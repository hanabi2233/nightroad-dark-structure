"""Publication adapter around frozen training cores. Does not run without --execute."""
import argparse, random, time
from pathlib import Path
from release_common import (ROOT,SEEDS,VARIANTS,fresh,verify_sources,sources,
                            sealed_model_from_assets,write_new,digest,checkpoint_record)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    p.add_argument('--variant',required=True,choices=VARIANTS)
    p.add_argument('--binding',required=True,help='Verified train.json from prepare_lolv2_bindings.py')
    p.add_argument('--initialization-root',type=Path,required=True,help='Extracted initialization/ directory')
    p.add_argument('--output',required=True,help='A new directory outside the repository')
    p.add_argument('--execute',action='store_true',help='Explicitly start the fixed 8000-step run')
    a=p.parse_args()
    if not a.execute:
        print('Plan only: 8000 steps, batch 8, crop 256, FP32, fixed-last checkpoint. Add --execute to run.')
        return
    verify_sources()
    folder,common,core,merged,component=sources(a.seed,a.variant)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU is required')
    common.setup(a.seed)
    bindings,members=common.binding_check(a.binding,'train')
    data=common.TrainingData(bindings,members,a.seed)
    model,ident=sealed_model_from_assets(common,folder,a.initialization_root,a.seed)
    model=model.cuda(0).train();torch.cuda.manual_seed_all(a.seed)
    from sealed_initialization import verify_state_mapping
    verify_state_mapping(model.state_dict(),ident,'cuda_loaded')
    adam=core.AdamCandidate(core.flatten([p.detach() for p in model.parameters()]))
    rng=random.Random(a.seed)
    config_variant=a.variant if a.variant in ('A','SUM4','B','GTMEAN') else 'B'
    config=common.configuration(a.seed,config_variant)
    out=fresh(a.output)
    active={'A':['R'],'SUM4':['R','S','over','under'],'B':['R','S','over','under'],
            'GTMEAN':['GTMEAN'],'MERGED_LUM_PCGRAD':['R','S','merged_luminance'],
            'MERGED_DROP_S':['R','merged_luminance'],'MERGED_DROP_M':['R','S']}[a.variant]
    write_new(out/'CONFIGURATION.json',{'published_adapter':True,'variant':a.variant,'base':config,
              'objectives_after_warmup':active,'warmup_objectives':['R'],
              'gradient_rule':'direct scalar sum' if a.variant in ('A','SUM4','GTMEAN') else 'standard PCGrad then sum',
              'remaining_loss_rescaling':False})
    import platform, importlib.metadata as md
    versions={k:md.version(k) for k in ('torch','numpy','scipy','einops','opencv-python','Pillow')}
    write_new(out/'ENVIRONMENT.json',{'python':platform.python_version(),'versions':versions,
              'gpu':torch.cuda.get_device_name(0),'bitwise_equivalence_to_original_run':'not established'})
    start=time.time()
    for index in range(common.STEPS):
        attempt=index+1;x,y,sample=data.batch(index);x=x.cuda(0);y=y.cuda(0)
        if a.variant=='MERGED_LUM_PCGRAD':
            detail,trace=merged.training_update(core,model,x,y,adam,attempt,rng,common.lr_at(index))
        elif a.variant in ('MERGED_DROP_S','MERGED_DROP_M'):
            detail,trace=component.training_update(core,merged,model,x,y,adam,a.variant,attempt,rng,common.lr_at(index))
        else:
            detail,trace=common.update(model,x,y,adam,a.variant,attempt,rng)
        if adam.step_count!=attempt: raise ValueError('Optimizer counter mismatch')
        if attempt in (500,501,8000): common.save_trace(out/f'trace_{attempt:04d}.pt',trace,adam)
        common.event(out/'attempts.jsonl',{'attempt':attempt,'seed':a.seed,**sample,**detail})
        del x,y,trace
        if attempt%100==0: print(a.seed,a.variant,f'{attempt}/8000',round(time.time()-start,1),flush=True)
    checkpoint=out/'final_iter8000.pth'
    with checkpoint.open('xb') as f: torch.save(model.cpu().state_dict(),f)
    write_new(out/'FINAL_CHECKPOINT.json',{'seed':a.seed,'variant':a.variant,'step':8000,
              'sha256':digest(checkpoint),'selection':'fixed last; no intermediate validation',
              'original_paper_checkpoint_sha256':checkpoint_record(a.seed,a.variant)['checkpoint_file_sha256'],
              'note':'A new reproduction run; never relabel as an original paper checkpoint.'})
    print('Completed:',checkpoint)

if __name__=='__main__':main()
