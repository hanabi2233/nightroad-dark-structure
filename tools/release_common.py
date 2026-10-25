"""Publication packaging helpers. Original scientific implementations stay unchanged."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261004, 20261005, 20261006)
VARIANTS = ('A', 'SUM4', 'B', 'GTMEAN', 'MERGED_LUM_PCGRAD', 'MERGED_DROP_S', 'MERGED_DROP_M')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path, obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def fresh(path):
    p=Path(path).resolve()
    if p==ROOT or p in ROOT.parents or ROOT in p.parents:
        raise ValueError('Choose a new output directory outside the source repository')
    p.mkdir(parents=True,exist_ok=False)
    return p

def verify_sources():
    checked=0
    for row in rows(ROOT/'SOURCE_MANIFEST.csv'):
        p=ROOT/row['path']
        if not p.is_file() or p.stat().st_size!=int(row['bytes']) or digest(p)!=row['sha256']:
            raise ValueError('Source integrity mismatch: '+row['path'])
        checked+=1
    return checked

def sources(seed,variant):
    """Use the recorded package family for the requested training instance."""
    if seed not in SEEDS or variant not in VARIANTS: raise ValueError('Unknown reported instance')
    kind='confirmation' if seed==20261004 and variant in ('A','SUM4','B') else 'followup'
    folder=ROOT/'src/frozen'/kind
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(folder))
    import common
    import core
    sys.path.insert(0,str(ROOT/'src/frozen/merged'))
    sys.path.insert(0,str(ROOT/'src/frozen/components'))
    import merged_core, component_core
    return folder,common,core,merged_core,component_core

def checkpoint_record(seed,variant):
    matches=[r for r in rows(ROOT/'metadata/TRAINING_INSTANCES.csv')
             if int(r['seed'])==seed and r['implementation_variant']==variant]
    if len(matches)!=1: raise ValueError('Checkpoint registry must contain exactly one instance')
    return matches[0]

def check_checkpoint(path,seed,variant):
    record=checkpoint_record(seed,variant)
    if digest(path)!=record['checkpoint_file_sha256'].lower():
        raise ValueError('Checkpoint does not match the paper registry')
    return record

def load_definitions(path,names,namespace):
    """Compile selected exact source function bodies, with caller-supplied path bindings."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in definitions}!=set(names): raise ValueError('Missing source definition')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future]+definitions,type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace

def sealed_model_from_assets(common,folder,asset_root,seed):
    """Restore the original state AND CPU RNG; do not regenerate initial weights from a seed."""
    import torch
    from sealed_initialization import verify_state_mapping
    ident=read(folder/'initialization'/str(seed)/'IDENTITY.json')
    assets=Path(asset_root)/str(seed)
    state_path=assets/'zero_step.pt'; rng_path=assets/'post_constructor_rng.pt'
    if digest(state_path)!=ident['file_sha256'] or digest(rng_path)!=ident['rng_file_sha256']:
        raise ValueError('Initializer or CPU RNG identity mismatch')
    if digest(folder/'base/vendor/RetinexFormer_arch.py')!=ident['source_architecture_sha256']:
        raise ValueError('Architecture identity mismatch')
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    verify_state_mapping(state,ident,'serialized')
    model=common.construct(seed)
    model.load_state_dict(state,strict=True)
    verify_state_mapping(model.state_dict(),ident,'loaded')
    torch.set_rng_state(torch.load(rng_path,map_location='cpu',weights_only=True))
    return model,ident

def metric_functions(common,core,backbone):
    import numpy as np, torch
    import torch.nn.functional as F
    from PIL import Image
    ns=dict(np=np,torch=torch,F=F,Image=Image,Path=Path,sha=digest,
            BACKBONE=Path(backbone),core=core,__name__='published_frozen_metrics')
    path=ROOT/'src/frozen/followup/base/evaluate_pilot.py'
    return load_definitions(path,['rgb','metric_net','metrics','diagnostics','prediction'],ns)
