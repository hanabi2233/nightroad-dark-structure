"""Path/import adaptation only; upstream definitions are compiled without editing them."""
import ast
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def safe(relative):
    path = ROOT / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Nonrelative package path')
    current = ROOT
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink() or (hasattr(os.path, 'isjunction') and os.path.isjunction(current)):
            raise ValueError('Symlink/reparse path is forbidden: '+str(current))
    if ROOT not in path.resolve().parents or not path.is_file():
        raise ValueError('Missing/nonregular package file: '+str(path))
    return path

def write_new(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def extract(filename, names, namespace):
    path = safe('vendor/'+filename)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    definitions = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in definitions} != set(names):
        raise RuntimeError('Missing upstream definition')
    # Add future annotations only to avoid typing-import side effects; function bodies are untouched.
    future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future]+definitions,type_ignores=[]))
    exec(compile(module,str(path),'exec'),namespace)
    return namespace

def data_helpers():
    import numpy as np
    import torch
    import cv2
    from torch.utils.data import Dataset
    ns = dict(np=np,torch=torch,cv2=cv2,random=random,Path=Path,contextlib=contextlib,
              hashlib=hashlib,Dataset=Dataset,__name__='portable_upstream')
    extract('transforms.py', ['paired_random_crop','data_augmentation','random_augmentation'],ns)
    extract('img_util.py',['img2tensor'],ns)
    extract('image_restoration_model.py',['Mixing_Augment'],ns)
    extract('original_adapter.py',['LOLv2FrozenSplitPairedDataset'],ns)
    extract('original_runner.py',['data_rng','data_seed','batch_schedule','tensor_sha'],ns)
    return ns

class Train40:
    def __init__(self):
        self.helpers = data_helpers()
        with safe('selected40.csv').open(encoding='utf-8',newline='') as f:
            members = list(csv.DictReader(f))
        if len(members)!=40 or len({r['sample_id'] for r in members})!=40 or any(r['split']!='train' for r in members):
            raise ValueError('Only the fixed 40 train members are permitted')
        paths = json.loads(safe('paths.json').read_text())
        base = self.helpers['LOLv2FrozenSplitPairedDataset']
        # Bind ONLY authorized records; no original constructor that requires all 551 image files.
        self.dataset = object.__new__(base)
        self.dataset.phase='train'
        self.dataset.frozen_split='train'
        self.dataset.gt_size=256
        self.dataset.scale=1
        self.dataset.geometric_augs=True
        self.dataset.records=[]
        for r in members:
            if r['low_path'] != paths['low_root']+'/low'+r['sample_id']+'.png' or r['gt_path'] != paths['gt_root']+'/normal'+r['sample_id']+'.png':
                raise ValueError('Path contract conflict')
            self.dataset.records.append(dict(sample_id=r['sample_id'],lq_path=str(safe(r['low_path'])),gt_path=str(safe(r['gt_path']))))
        self.index={r['sample_id']:i for i,r in enumerate(members)}
        self.batches=json.loads(safe('schedule.json').read_text())['batches']
        if len(self.batches)!=5 or any(len(b)!=8 for b in self.batches) or set(sum(self.batches,[]))!=set(self.index):
            raise ValueError('Fixed five-batch schedule conflict')

    def batch(self, attempt):
        import torch
        if not 0<=attempt<5:
            raise ValueError('Engineering branch exceeds five updates')
        ids=self.batches[attempt]
        with self.helpers['data_rng'](attempt):
            samples=[self.dataset[self.index[s]] for s in ids]
            x=torch.stack([s['lq'] for s in samples])
            y=torch.stack([s['gt'] for s in samples])
            mixer=self.helpers['Mixing_Augment'](1.2,True,torch.device('cpu'))
            y,x=mixer(y,x)
        if x.shape!=(8,3,256,256) or x.dtype!=torch.float32 or y.shape!=x.shape:
            raise ValueError('Batch/crop/precision conflict')
        return x,y,dict(sample_ids=ids,input_tensor_sha256=self.helpers['tensor_sha'](x),
                       gt_tensor_sha256=self.helpers['tensor_sha'](y))

def scratch_cpu():
    import torch
    import numpy as np
    random.seed(20260910)
    np.random.seed(20260910)
    torch.random.default_generator.manual_seed(20260910)
    path=safe('vendor/RetinexFormer_arch.py')
    spec=importlib.util.spec_from_file_location('smoke_upstream_arch',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2]).train()
