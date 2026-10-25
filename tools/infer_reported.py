"""Common-PNG inference using an original paper checkpoint and an explicit input list."""
import argparse, csv
from pathlib import Path
from release_common import ROOT,SEEDS,VARIANTS,rows,read,fresh,write_new,digest,verify_sources,sources,check_checkpoint,load_definitions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    p.add_argument('--variant',required=True,choices=VARIANTS)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--inputs',required=True,type=Path,help='CSV: sample_id,input_path,input_sha256')
    p.add_argument('--output',required=True)
    a=p.parse_args();verify_sources();record=check_checkpoint(a.checkpoint,a.seed,a.variant)
    folder,common,core,merged,component=sources(a.seed,a.variant)
    import numpy as np,torch
    import torch.nn.functional as F
    from PIL import Image
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the recorded implementation')
    common.setup(20260908)
    model=common.construct(a.seed);model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval().requires_grad_(False)
    ns=dict(np=np,torch=torch,F=F,Image=Image,__name__='frozen_prediction')
    predict=load_definitions(folder/'base/evaluate_pilot.py',['prediction'],ns)['prediction']
    inputs=rows(a.inputs)
    if not inputs or len({r['sample_id'] for r in inputs})!=len(inputs):raise ValueError('Empty or duplicate sample list')
    for row in inputs:
        sid=row['sample_id']
        if Path(sid).name!=sid or '/' in sid or '\\' in sid or sid in ('.','..'):raise ValueError('Unsafe sample ID')
        if digest(row['input_path'])!=row['input_sha256'].lower():raise ValueError('Input identity mismatch: '+sid)
    out=fresh(a.output);(out/'predictions').mkdir()
    manifest=[]
    for row in inputs:
        with Image.open(row['input_path']) as im: x=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        target=out/'predictions'/(row['sample_id']+'.png')
        Image.fromarray(predict(model,x)).save(target)
        manifest.append({'sample_id':row['sample_id'],'prediction':target.relative_to(out).as_posix(),
                         'input_sha256':row['input_sha256'],'prediction_sha256':digest(target)})
    with (out/'OUTPUT_MANIFEST.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    write_new(out/'SUMMARY.json',{'seed':a.seed,'variant':a.variant,'completed':len(manifest),
                               'checkpoint_sha256':record['checkpoint_file_sha256'],'metrics_computed':False})

if __name__=='__main__':main()
