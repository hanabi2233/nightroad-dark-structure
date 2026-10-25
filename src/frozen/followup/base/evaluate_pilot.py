"""Local exact common-PNG metrics; no history checkpoints, downloads or best selection."""
import argparse, importlib.metadata as md, socket, traceback, platform, subprocess
from pilot_common import *
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F
import core
BACKBONE=Path('C:/Users/wlx/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')
def deny(*a,**k):raise RuntimeError('NETWORK_DOWNLOAD_FORBIDDEN')
socket.socket.connect=deny;socket.socket.connect_ex=deny;torch.hub.download_url_to_file=deny
def rgb(p):
    with Image.open(p) as im:return np.asarray(im.convert('RGB'),dtype=np.float32)/255.
def metric_net():
    import lpips
    linear=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    assert sha(BACKBONE).upper()=='7BE5BE791159472B1FBF3C69796F7CB30DCA7AD8466C2DF70058C37116CDEE02'
    assert sha(linear).upper()=='DF73285E35B22355A2DF87CDB6B70B343713B667EDDBDA73E1977E0C860835C0'
    torch.hub.set_dir(str(BACKBONE.parent.parent))
    return lpips.LPIPS(net='alex',version='0.1',pretrained=True,lpips=True,spatial=False,pnet_rand=False,pnet_tune=False,use_dropout=True,model_path=str(linear),eval_mode=True,verbose=False).eval().requires_grad_(False).cuda()
def metrics(pred,gt,net):
    # Numerically identical to M0/M1 common_runner.metrics; frozen source retained in vendor.
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    x=torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).cuda()*2-1
    y=torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).cuda()*2-1
    with torch.inference_mode():lp=float(net(x,y,normalize=False,retPerLayer=False).item())
    result=dict(PSNR=float(peak_signal_noise_ratio(gt,pred,data_range=1.0)),SSIM=float(structural_similarity(gt,pred,win_size=7,gradient=False,data_range=1.0,channel_axis=2,gaussian_weights=False,full=False,use_sample_covariance=True,K1=.01,K2=.03,sigma=1.5)),LPIPS=lp)
    assert all(np.isfinite(v) for v in result.values()),result
    return result
def diagnostics(x,y,z):
    tx,ty,tz=[torch.from_numpy(v).permute(2,0,1)[None] for v in (x,y,z)]
    mask=core.dark_mask(tx)[0,0].numpy();count=int(mask.sum())
    bias=(core.luma(tz)-core.luma(ty))[0,0].numpy()
    grad=torch.linalg.vector_norm(core.sobel(core.luma(tz))-core.sobel(core.luma(ty)),dim=1)[0].numpy()
    mae=np.abs(z-y).mean(2)
    result={'mask_pixels':count,'region_valid':count>=256}
    for region,m in [('full',np.ones(mask.shape,dtype=bool)),('dark',mask)]:
        fields=['O','U','bias','over_fraction','under_fraction','black_fraction','white_fraction','gradient_error','rgb_mae']
        if region=='dark' and count<256:
            result.update({region+'_'+k:None for k in fields});continue
        v=bias[m]
        vals=[float(np.maximum(v,0).mean()),float(np.maximum(-v,0).mean()),float(v.mean()),float((v>2/255).mean()),float((v < -2/255).mean()),float((z[m]==0).mean()),float((z[m]==1).mean()),float(grad[m].mean()),float(mae[m].mean())]
        result.update({region+'_'+k:v for k,v in zip(fields,vals)})
    return result
def prediction(model,x):
    t=torch.from_numpy(x).permute(2,0,1)[None].cuda();h,w=t.shape[-2:]
    with torch.inference_mode():z=model(F.pad(t,(0,(-w)%4,0,(-h)%4),mode='reflect'))[...,:h,:w].clamp(0,1)
    assert bool(torch.isfinite(z).all())
    return np.rint(z[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
def preflight(out):
    out=Path(out);out.mkdir(exist_ok=False);setup(20260908)
    expected={'numpy':'2.3.4','Pillow':'12.0.0','scikit-image':'0.25.2','lpips':'0.1.4','torch':'2.11.0.dev20260120+cu128','torchvision':'0.25.0.dev20260120+cu128'}
    actual={k:md.version(k) for k in expected};assert actual==expected,actual
    assert platform.python_version()=='3.11.9' and torch.cuda.is_available()
    chk=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True);assert chk.returncode==0,chk.stdout+chk.stderr
    net=metric_net()
    # Synthetic-only pipeline check, not validation screening or throughput measurement.
    from sealed_initialization import sealed_scratch_cpu
    model,_=sealed_scratch_cpu();model=model.cuda().eval().requires_grad_(False)
    x=np.random.default_rng(20260908).random((65,67,3),dtype=np.float32)
    png=prediction(model,x);Image.fromarray(png).save(out/'synthetic.png')
    z=rgb(out/'synthetic.png');assert z.shape==x.shape
    values=metrics(z,x,net);diag=diagnostics(x,x,z)
    assert np.isfinite(list(values.values())).all()
    dump(out/'RESULT.json',{'status':'PASS','executable':sys.executable,'python':platform.python_version(),'versions':actual,'gpu':torch.cuda.get_device_name(0),'pip_check':chk.stdout,'synthetic_only':True,'padding_unpadding':True,'metrics':values,'diagnostics':diag})
def evaluate(a):
    out=Path(a.output);out.mkdir(exist_ok=False);(out/'predictions').mkdir()
    start=time.time(); completed=0
    try:
        check_time(a.deadline);verify_package();setup(20260908)
        final=read(Path(a.checkpoint).parent/'FINAL_CHECKPOINT.json')
        assert final['attempt']==2000 and final['sha256']==sha(a.checkpoint)
        from portable import scratch_cpu
        model=scratch_cpu();state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        load=model.load_state_dict(state,strict=True);assert not load.missing_keys and not load.unexpected_keys and len(state)==122
        model=model.cuda().eval().requires_grad_(False);net=metric_net();setup(20260908)
        bindings=read(a.binding);members=[r for r in rows(ROOT/'DATA_MANIFEST.csv') if r['split']=='validation'];assert len(members)==138
        for r in members:
            check_time(a.deadline);sid=r['sample_id'];paths=bindings[sid]
            x=rgb(paths['input']);y=rgb(paths['gt']);assert x.shape==y.shape==(int(r['height']),int(r['width']),3)
            p=out/'predictions'/f'{sid}.png';assert not p.exists()
            Image.fromarray(prediction(model,x)).save(p)
            z=rgb(p);vals=metrics(z,y,net);diag=diagnostics(x,y,z)
            event(out/'per_image.jsonl',{'sample_id':sid,'group_id':r['group_id'],'variant':final['variant'],**vals,**diag,'prediction_sha256':sha(p)})
            completed+=1
            if completed%20==0:print(f"eval {final['variant']} {completed}/138",flush=True)
        status='EVALUATION_COMPLETE'
    except Exception as e:
        status='FAILED_INCOMPLETE';dump(out/'FAILURE.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':completed});print(traceback.format_exc(),flush=True)
    dump(out/'EVAL_SUMMARY.json',{'status':status,'completed':completed,'expected':138,'failures':0 if completed==138 else 1,'elapsed_seconds':time.time()-start})
    return 0 if completed==138 else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--binding');p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();check_time(a.deadline)
    if a.preflight:preflight(a.output)
    else:sys.exit(evaluate(a))
