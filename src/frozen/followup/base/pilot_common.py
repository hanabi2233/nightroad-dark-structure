"""Independent formal entry helpers; frozen optimizer core is imported unchanged."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, csv, time, random, hashlib
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
TRACE_STEPS = (500, 501, 2000)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def dump(p, v):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
def event(p, v):
    with Path(p).open('a', encoding='utf-8') as f: f.write(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def check_time(deadline):
    if time.time() >= deadline: raise TimeoutError('SHARED_FOUR_HOUR_DEADLINE')
def setup(seed=20260910):
    import numpy as np, torch
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True, warn_only=False)
def four(variant, step):
    assert variant in 'ABC' and 1<=step<=2000
    return variant!='A' and step>500
def save_trace(path, tensors, adam):
    import torch
    payload={k:v.detach().cpu() for k,v in tensors.items()}
    payload.update(adam_first_moment=adam.m.detach().cpu(),adam_second_moment=adam.v.detach().cpu(),adam_step_count=adam.step_count,role='DIAGNOSTIC_NOT_RESUMABLE_MODEL')
    with Path(path).open('xb') as f: torch.save(payload,f)
def verify_package():
    manifest=rows(ROOT/'STAGING_MANIFEST.csv')
    for r in manifest:
        p=ROOT/r['path']
        assert not p.is_symlink() and p.is_file(), str(p)
        assert p.stat().st_size==int(r['bytes']) and sha(p)==r['sha256'],str(p)
    return len(manifest)
class TrainingData:
    def __init__(self, binding):
        from portable import data_helpers
        self.h=data_helpers()
        allrows=rows(ROOT/'DATA_MANIFEST.csv')
        self.members=[r for r in allrows if r['split']=='train']
        assert len(self.members)==551
        base=self.h['LOLv2FrozenSplitPairedDataset']
        self.ds=object.__new__(base)
        self.ds.phase='train'; self.ds.frozen_split='train'; self.ds.gt_size=256
        self.ds.scale=1; self.ds.geometric_augs=True
        self.ds.records=[{'sample_id':r['sample_id'],'lq_path':binding[r['sample_id']]['input'],'gt_path':binding[r['sample_id']]['gt']} for r in self.members]
        self.schedule=self.h['batch_schedule'](551,2000,20260910)
        expected=read(ROOT/'SCHEDULE.json')
        assert [[self.members[i]['sample_id'] for i in b] for b in self.schedule]==expected['sample_ids']
    def batch(self,index):
        import torch
        with self.h['data_rng'](index):
            samples=[self.ds[i] for i in self.schedule[index]]
            x=torch.stack([s['lq'] for s in samples]); y=torch.stack([s['gt'] for s in samples])
            mixer=self.h['Mixing_Augment'](1.2,True,torch.device('cpu'))
            y,x=mixer(y,x)
        assert x.shape==y.shape==(8,3,256,256) and x.dtype==y.dtype==torch.float32
        return x,y,{'sample_ids':[s['sample_id'] for s in samples], 'input_sha':self.h['tensor_sha'](x),'gt_sha':self.h['tensor_sha'](y),'augmentation_seed':self.h['data_seed'](index)}
