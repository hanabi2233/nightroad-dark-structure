"""New experiment adapter. Imports do not launch training or inspect images."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import contextlib, csv, importlib.util, json, math, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / 'base'))
from pilot_common import sha, read, rows, dump, event, setup, save_trace

SEEDS = (20261004, 20261005, 20261006)
BASELINE_VARIANTS = ('A', 'SUM4', 'B')
CANDIDATE_VARIANTS = ('GTMEAN', 'B_DROP_R', 'B_DROP_S', 'B_DROP_OVER', 'B_DROP_UNDER')
VARIANTS = BASELINE_VARIANTS + CANDIDATE_VARIANTS
STEPS = 8000
WARMUP = 500

def regular(path):
    p = Path(path)
    for q in (p, *p.parents):
        if q.is_symlink() or (hasattr(os.path, 'isjunction') and os.path.isjunction(q)):
            raise ValueError('REPARSE_OR_SYMLINK: ' + str(q))
    if not p.is_file(): raise ValueError('NOT_REGULAR: ' + str(p))
    return p

def verify_package():
    manifest = rows(HERE / 'MANIFEST.csv')
    expected = set()
    for r in manifest:
        rel = Path(r['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('UNSAFE_MANIFEST')
        p = regular(HERE / rel)
        if p.stat().st_size != int(r['bytes']) or sha(p) != r['sha256']:
            raise ValueError('PACKAGE_HASH: ' + str(rel))
        expected.add(rel.as_posix())
    actual = {p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file()}
    if actual != expected | {'MANIFEST.csv'}: raise ValueError('UNREGISTERED_PACKAGE_FILES')
    return len(manifest)

def authorize(path, phase, seed=None, variant=None):
    a = read(regular(path))
    if a.get('approved') is not True or a.get('experiment') != 'PCGRAD_METHOD_FOLLOWUP_PACKAGE_V2':
        raise ValueError('EXPLICIT_APPROVAL_REQUIRED')
    if a.get('protocol_sha256') != sha(HERE / 'PROTOCOL.md'):
        raise ValueError('APPROVAL_PROTOCOL_MISMATCH')
    if a.get('manifest_sha256') != sha(HERE / 'MANIFEST.csv'):
        raise ValueError('APPROVAL_PACKAGE_MISMATCH')
    if phase not in a.get('phases', []): raise ValueError('PHASE_NOT_AUTHORIZED')
    if phase == 'train' and dict(seed=seed, variant=variant) not in a.get('allowed_runs', []):
        raise ValueError('EXACT_TRAIN_RUN_NOT_AUTHORIZED')
    if phase in ('train','evaluate') and not isinstance(a.get('budget_seconds'), int):
        raise ValueError('RESOURCE_BUDGET_NOT_APPROVED')
    return a

def fresh_output(path):
    p = Path(path).resolve()
    if p == HERE or HERE in p.parents: raise ValueError('OUTPUT_INSIDE_SEALED_PACKAGE')
    if p.exists(): raise FileExistsError('NO_OVERWRITE: ' + str(p))
    # Explicitly exclude ancestors of the sealed package as output roots too.
    if p in HERE.parents: raise ValueError('OUTPUT_IS_PACKAGE_ANCESTOR')
    p.mkdir(parents=True, exist_ok=False)
    return p

def check_deadline(deadline):
    if time.time() >= deadline: raise TimeoutError('APPROVED_SHARED_DEADLINE')

def four(variant, attempt):
    if variant not in VARIANTS or not 1 <= attempt <= STEPS: raise ValueError('BRANCH_OR_BUDGET')
    return variant != 'A' and attempt > WARMUP

def lr_at(index):
    if not 0 <= index < STEPS: raise ValueError('SCHEDULER_INDEX')
    return 1e-6 + .5 * (2e-4 - 1e-6) * (1 + math.cos(math.pi * index / STEPS))

def configuration(seed, variant):
    if seed not in SEEDS or variant not in VARIANTS: raise ValueError('CONFIGURATION_ID')
    c = read(HERE/'configs'/f'{seed}_{variant}.json')
    expected = dict(seed=seed, variant=variant, total_attempts=STEPS, l1_warmup=WARMUP,
                    batch=8, crop=256, precision='FP32', amp=False, resume_allowed=False,
                    historical_checkpoint=None, trace_steps=[500,501,8000], checkpoint_steps=[8000])
    if any(c[k] != value for k,value in expected.items()): raise ValueError('CONFIGURATION_IMPLEMENTATION_MISMATCH')
    if c['scheduler'] != dict(name='cosine',T=STEPS,eta_min=1e-6): raise ValueError('SCHEDULER_CONFIG')
    if c['optimizer'] != dict(name='Adam',lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_norm=.01): raise ValueError('OPTIMIZER_CONFIG')
    if c['mixup'] != dict(beta=1.2,use_identity=True): raise ValueError('AUGMENTATION_CONFIG')
    if variant in CANDIDATE_VARIANTS:
        import candidate_core
        scientific = (dict(type='GTMEAN',source_commit=candidate_core.SOURCE_COMMIT,
                           sigma=0.1,reduction='none_then_mean',warmup='R')
                      if variant == 'GTMEAN' else
                      dict(type='B_LEAVE_ONE_OUT',removed=candidate_core.DROP_VARIANTS[variant],
                           warmup='R',pcgrad='standard'))
        if c.get('candidate') != scientific: raise ValueError('CANDIDATE_SCIENCE_CONFIG')
    return c

def construct(seed):
    """CPU constructor only; never initialize CUDA here."""
    import numpy as np, torch
    if seed not in SEEDS: raise ValueError('UNAPPROVED_SEED')
    random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
    spec = importlib.util.spec_from_file_location('confirmation_arch', HERE/'base/vendor/RetinexFormer_arch.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2]).train()

def sealed_model(seed):
    import torch
    from sealed_initialization import verify_state_mapping, legacy_sha
    identity = read(HERE / 'initialization' / str(seed) / 'IDENTITY.json')
    p = regular(HERE / 'initialization' / str(seed) / 'zero_step.pt')
    if identity['role'] != 'ZERO_STEP_UNTRAINED_NOT_PRETRAINED' or identity['seed'] != seed:
        raise ValueError('INITIALIZER_ROLE')
    if sha(p) != identity['file_sha256'] or p.stat().st_size != identity['file_bytes']:
        raise ValueError('INITIALIZER_HASH')
    if sha(HERE/'base/vendor/RetinexFormer_arch.py') != identity['source_architecture_sha256']:
        raise ValueError('ARCHITECTURE_HASH')
    state = torch.load(p, map_location='cpu', weights_only=True)
    verify_state_mapping(state, identity, 'serialized')
    model = construct(seed)
    fresh = legacy_sha(model)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys: raise ValueError('STRICT_LOAD')
    verify_state_mapping(model.state_dict(), identity, 'loaded')
    # Bind the CPU random stream after the original Windows constructor, independent of platform.
    rng = HERE/'initialization'/str(seed)/'post_constructor_rng.pt'
    if sha(rng) != identity['rng_file_sha256']: raise ValueError('INITIALIZER_RNG_HASH')
    torch.set_rng_state(torch.load(rng, map_location='cpu', weights_only=True))
    return model, dict(seed=seed, file_sha256=identity['file_sha256'],
                      legacy_initializer_sha256=identity['legacy_initializer_sha256'],
                      fresh_status='MATCH' if fresh == identity['legacy_initializer_sha256'] else 'MISMATCH_DIAGNOSTIC_ONLY',
                      state_tensors=len(state), missing=0, unexpected=0)

@contextlib.contextmanager
def data_rng(seed, index):
    import numpy as np, torch
    py = random.getstate(); ns = np.random.get_state(); ts = torch.get_rng_state()
    value = seed + 100003 * (index + 1)
    try:
        random.seed(value); np.random.seed(value % 2**32)
        torch.random.default_generator.manual_seed(value)
        yield value
    finally:
        random.setstate(py); np.random.set_state(ns); torch.set_rng_state(ts)

def schedule(seed):
    from portable import data_helpers
    return data_helpers()['batch_schedule'](551, STEPS, seed)

def binding_check(path, split):
    binding = read(regular(path))
    members = [r for r in rows(HERE/'base/DATA_MANIFEST.csv') if r['split'] == split]
    expected = 551 if split == 'train' else 138 if split == 'validation' else None
    if len(members) != expected or set(binding) != {r['sample_id'] for r in members}:
        raise ValueError('BINDING_SCOPE_MUST_EXACTLY_MATCH_SPLIT')
    for r in members:
        for role in ('input','gt'):
            p = regular(binding[r['sample_id']][role])
            if p.stat().st_size != int(r[role+'_bytes']) or sha(p) != r[role+'_sha256']:
                raise ValueError('DATA_HASH: '+r['sample_id']+':'+role)
    return binding, members

class TrainingData:
    def __init__(self, binding, members, seed):
        from portable import data_helpers
        self.h = data_helpers(); self.members = members; self.seed = seed
        self.schedule = schedule(seed)
        frozen = read(HERE/'schedules'/f'{seed}.json')
        if [[members[i]['sample_id'] for i in b] for b in self.schedule] != frozen['sample_ids']:
            raise ValueError('SCHEDULE_CONFLICT')
        self.ds = object.__new__(self.h['LOLv2FrozenSplitPairedDataset'])
        self.ds.phase='train'; self.ds.frozen_split='train'; self.ds.gt_size=256
        self.ds.scale=1; self.ds.geometric_augs=True
        self.ds.records=[dict(sample_id=r['sample_id'], lq_path=binding[r['sample_id']]['input'],
                              gt_path=binding[r['sample_id']]['gt']) for r in members]
    def batch(self,index):
        import torch
        with data_rng(self.seed, index) as augseed:
            samples = [self.ds[i] for i in self.schedule[index]]
            x=torch.stack([s['lq'] for s in samples]); y=torch.stack([s['gt'] for s in samples])
            y,x=self.h['Mixing_Augment'](1.2, True, torch.device('cpu'))(y,x)
        if x.shape != y.shape or x.shape != (8,3,256,256) or x.dtype != torch.float32 or y.dtype != torch.float32:
            raise ValueError('BATCH_CONTRACT')
        return x,y,dict(sample_ids=[s['sample_id'] for s in samples], input_sha=self.h['tensor_sha'](x),
                        gt_sha=self.h['tensor_sha'](y), augmentation_seed=augseed)

def update(model,x,y,adam,variant,attempt,rng):
    import core, sum4_core, candidate_core
    if variant in CANDIDATE_VARIANTS:
        return candidate_core.training_update(core,model,x,y,adam,variant,attempt,rng,lr_at(attempt-1))
    target = sum4_core.training_update if variant == 'SUM4' else core.training_update
    detail,tensors = target(model,x,y,adam,variant,four(variant,attempt),rng,lr_at(attempt-1))
    detail.update(variant=variant, pcgrad_applied=variant=='B' and four(variant,attempt), optimizer_updates=1)
    return detail,tensors
