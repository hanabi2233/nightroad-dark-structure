from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, csv, json, hashlib, time, platform, socket, subprocess, traceback, random
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata as metadata
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
LITE=ROOT/'experiments/PHASE4_PAPER_B_G2_LITE_PROTOCOL_FREEZE_V1/execution_retry_v2'
FULL=ROOT/'experiments/PHASE4_PAPER_B_G2_FULL_SERVER_EXECUTION_V2/execution'
SOURCE=ROOT/'third_party/Retinexformer_1e9a0efce4b306b6701b824768370ff26066c32a'
METHODS=['Stage1','ACDC_Specialist','RetinexNet','SCI','EnlightenGAN','Zero-DCE']
DIRS=[LITE/'01_RETINEXFORMER_STAGE1_FOUNDATION',LITE/'02_RETINEXFORMER_ACDC_SPECIALIST',FULL/'01_RetinexNet',FULL/'02_SCI',FULL/'03_EnlightenGAN',FULL/'04_Zero_DCE']
CKPTS=[ROOT/'experiments/R3H_RUN2/stage1/models/net_g_best.pth',ROOT/'experiments/R3F_PHASE1_RUN1/best_psnr_31.43_8000.pth']
CK_SHA=['8F88840ACA722E3EF92A05447DFDAAB1A8CE1675EE075691DC6D957C4C908DA1','59066E08B29F09E6BEC71573D2724F3A3A9CD78A933B1BEB7E43F4856A4EBD4D']
BACKBONE=Path('C:/Users/wlx/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')
LINEAR=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
METRICS=['PSNR','SSIM','LPIPS']
stage='initialization'; sample=''; all_records=[]; timings=[]
identity=[]

def now(): return datetime.now(timezone.utc).isoformat()
def norm(s): return str(s).replace('\\','/')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def readcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def readjson(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def csvwrite(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check(p,expected=None):
    p=Path(p);actual=sha(p)
    if expected and actual!=expected.upper(): raise RuntimeError(f'IDENTITY_CONFLICT: {p} expected {expected} actual {actual}')
    identity.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size,'expected':expected,'status':'PASS' if expected else 'RECORDED'})
    return actual
def rgb(p):
    with Image.open(p) as im: return np.asarray(im.convert('RGB'),dtype=np.float32)/255.0
def inspect_image(p,expected=None):
    digest=check(p,expected)
    with Image.open(p) as im:
        assert im.format=='PNG',str(p)
        im.load();size=im.size;mode=im.mode
    assert mode=='RGB', (p,mode)
    return digest,size
def deny(*args,**kwargs): raise RuntimeError('NETWORK_FORBIDDEN_IN_THIS_RUN')
socket.socket.connect=deny
socket.socket.connect_ex=deny
torch.hub.download_url_to_file=deny

def model_cpu(p):
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    model=RetinexFormer(in_channels=3,out_channels=3,n_feat=40,stage=1,num_blocks=[1,2,2])
    payload=torch.load(p,map_location='cpu',weights_only=True)
    sd=payload['params'] if isinstance(payload,dict) and isinstance(payload.get('params'),dict) else payload
    result=model.load_state_dict(sd,strict=True)
    assert len(sd)==122 and not result.missing_keys and not result.unexpected_keys
    model.requires_grad_(False);model.eval()
    return model

def preflight():
    global stage,sample
    stage='combined_preflight'
    assert not (OUT/'PREFLIGHT.json').exists(),'task output already exists; no automatic rerun'
    canonical_before={name:sha(ROOT/name) for name in ['PROJECT_CORRECTIONS.md','PROJECT_STATE.md','PROJECT_MASTER_STATE.md','EXPERIMENT_REGISTRY.md','DECISION_LOG.md']}
    if (OUT/'CANONICAL_BEFORE.json').exists():
        assert readjson(OUT/'CANONICAL_BEFORE.json')==canonical_before
    else:
        dump(OUT/'CANONICAL_BEFORE.json',canonical_before)
    expected_versions={'numpy':'2.3.4','Pillow':'12.0.0','scikit-image':'0.25.2','lpips':'0.1.4','lmdb':'2.3.0','torch':'2.11.0.dev20260120+cu128','torchvision':'0.25.0.dev20260120+cu128'}
    versions={p:metadata.version(p) for p in expected_versions}
    assert versions==expected_versions,versions
    assert platform.python_version()=='3.11.9'
    pipcheck=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    assert pipcheck.returncode==0,pipcheck.stdout+pipcheck.stderr
    assert torch.cuda.is_available(),'LOCAL_GPU_UNAVAILABLE'
    random.seed(20260908);np.random.seed(20260908);torch.manual_seed(20260908);torch.cuda.manual_seed_all(20260908)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    check(BACKBONE,'7BE5BE791159472B1FBF3C69796F7CB30DCA7AD8466C2DF70058C37116CDEE02')
    check(LINEAR,'DF73285E35B22355A2DF87CDB6B70B343713B667EDDBDA73E1977E0C860835C0')
    check(SOURCE/'basicsr/models/archs/RetinexFormer_arch.py','1567C89D55285A3C1A4D4CA33F288B1FD98DC6B6BBDF971EDA02D745F9087631')
    check(SOURCE/'LICENSE.txt','8C9E6D5A7B28E0FB08C5A4D0E72F582F514A2F2F1AB87A9E4FFFA30E0DD4BAF3')
    for p,h in zip(CKPTS,CK_SHA): check(p,h)
    sys.path.insert(0,str(SOURCE))
    models=[model_cpu(p) for p in CKPTS]
    check(ROOT/'data/manifests/ACDC_SYNTHETIC_VAL_V1.csv','4863B6D3985FCB1DE3D42EA94C8A57703F900EF6A147C05F30419597DCF8C3B3')
    cachep=ROOT/'experiments/R3F_PHASE1_RUN1/frozen_val_png_cache/FROZEN_VAL_PNG_CACHE_MANIFEST.csv'
    check(cachep,'EB1C258A08CE3575CDAB62B3865D738D0B94A08EB5EF78A2546BCD1134F62ACD')
    cache=readcsv(cachep);cache.sort(key=lambda r:norm(r['source_relative_path']))
    ids=[norm(r['source_relative_path']) for r in cache]
    assert len(ids)==len(set(ids))==106
    original_manifests=[]
    for p,h in [(LITE/'G2_LITE_OUTPUT_MANIFEST.csv','68A0D16CEA25456CE428031E7EEAA33CCA9201F3AFFAB3B1134DE15361ED7B7B'),(FULL/'G2_FULL_SERVER_OUTPUT_MANIFEST.csv','DC65AACBE42E88D5F7F5C7B69665654865F5E6FC700381023C7FE089AA1D6E37')]:
        check(p,h);original_manifests.append({norm(r['relative_path']):r['sha256'] for r in readcsv(p)})
    bindings=[];oldtables={}
    for mi,d in enumerate(DIRS):
        old=readcsv(d/'per_image_metrics.csv');oldmap={norm(r['source_relative_path']):r for r in old}
        assert len(old)==106 and set(oldmap)==set(ids)
        oldtables[METHODS[mi]]=oldmap
        root=LITE if mi<2 else FULL;fm=original_manifests[0 if mi<2 else 1]
        check(d/'per_image_metrics.csv',fm[norm((d/'per_image_metrics.csv').relative_to(root))])
    for ci,r in enumerate(cache):
        sample=norm(r['source_relative_path'])
        gt=Path(r['gt_path']);lq=Path(r['input_png_path'])
        gh,gs=inspect_image(gt,r['gt_sha256']);ih,isz=inspect_image(lq,r['input_png_sha256']);assert gs==isz
        for mi,d in enumerate(DIRS):
            p=d/'predictions'/sample;root=LITE if mi<2 else FULL;fm=original_manifests[0 if mi<2 else 1]
            ph,ps=inspect_image(p,fm[norm(p.relative_to(root))]);assert ps==gs
            bindings.append(dict(domain='M0_ACDC',method=METHODS[mi],sample_id=sample,input_path=str(lq),input_sha256=ih,gt_path=str(gt),gt_sha256=gh,prediction_path=str(p),prediction_sha256=ph,width=gs[0],height=gs[1]))
        if (ci+1)%20==0: print(f'preflight ACDC {ci+1}/106',flush=True)
    opt=yaml.safe_load((ROOT/'configs/R3H_RUN2_STAGE1.yaml').read_text(encoding='utf-8'))['datasets']['val']
    check(ROOT/'configs/R3H_RUN2_STAGE1.yaml');check(ROOT/'configs/R3F_PHASE1_RUN1.yaml')
    split=Path(opt['split_manifest']);check(split,'D1C054B03F8FE6FAE3FD99BB6DB673C2CEEAB482385391B5DC57D19B12690D06')
    allrows=readcsv(split);val=sorted([r for r in allrows if r['split']=='validation'],key=lambda r:r['sample_id'])
    train=[r for r in allrows if r['split']=='train']
    assert len(allrows)==len({r['sample_id'] for r in allrows})==689 and len(val)==138 and len(train)==551
    assert not {r['group_id'] for r in val}&{r['group_id'] for r in train}
    anchorp=ROOT/'experiments/R3I_RUN2_STAGE1_FINALIZATION/lolv2_real_frozen_validation/per_image_metrics.csv'
    check(anchorp);anchor={r['sample_id']:r for r in readcsv(anchorp)}
    assert set(anchor)=={r['sample_id'] for r in val}
    m1=[]
    for r in val:
        sample=r['sample_id'];lq=Path(opt['low_root'])/f'low{sample}.png';gt=Path(opt['gt_root'])/f'normal{sample}.png'
        assert lq.resolve()==Path(anchor[sample]['lq_path']).resolve() and gt.resolve()==Path(anchor[sample]['gt_path']).resolve()
        assert lq.parent.name=='Low' and gt.parent.name=='Normal' and lq.parent.parent.name=='Train'
        ih,isz=inspect_image(lq);gh,gs=inspect_image(gt);assert isz==gs
        m1.append(dict(sample_id=sample,group_id=r['group_id'],input_path=str(lq),input_sha256=ih,gt_path=str(gt),gt_sha256=gh,width=gs[0],height=gs[1]))
    csvwrite(OUT/'M0_BINDINGS.csv',bindings);csvwrite(OUT/'M1_VAL138_BINDINGS.csv',m1)
    csvwrite(OUT/'INPUT_IDENTITY_CHECKS.csv',identity)
    for p in [Path(lpips.__file__).parent/'lpips.py',Path(__import__('skimage.metrics',fromlist=['x']).__file__)]: check(p)
    metric=lpips.LPIPS(net='alex',version='0.1',pretrained=True,lpips=True,spatial=False,pnet_rand=False,pnet_tune=False,use_dropout=True,model_path=str(LINEAR),eval_mode=True,verbose=False).eval().requires_grad_(False)
    env=dict(time=now(),hostname=socket.gethostname(),os=platform.platform(),python=sys.version,executable=sys.executable,versions=versions,pip_check=pipcheck.stdout.strip(),pip_freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout,gpu=torch.cuda.get_device_name(0),gpu_properties=str(torch.cuda.get_device_properties(0)),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu_driver=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout,deterministic_algorithms=True,cudnn_deterministic=True,cudnn_benchmark=False,tf32=False,AMP=False)
    dump(OUT/'PREFLIGHT.json',dict(status='PASS',environment=env,ACDC_six_methods_each_106=True,lolv2_validation138=True,strict_load=[dict(method=m,keys=122,missing=0,unexpected=0,strict=True) for m in METHODS[:2]],reuse_search='R3I val138 contains only metrics/summary; Stage1 and Run1 save_img=false; registered Run1 final outputs ACDC106 or historical tests are ineligible; no matching frozen val138 predictions established',enhancer_inference_before_gate=False,metric_forward_before_gate=False,identity_checks=identity))
    seal={p.name:sha(p) for p in [OUT/'COMMON_PROTOCOL.md',Path(__file__),OUT/'M0_BINDINGS.csv',OUT/'M1_VAL138_BINDINGS.csv',OUT/'PREFLIGHT.json']}
    dump(OUT/'PROTOCOL_SEAL.json',dict(sealed_utc=now(),sha256=seal))
    print('MERGED_PREFLIGHT_PASS; protocol sealed; continuing execution',flush=True)
    return bindings,m1,models,metric,oldtables,anchor

def metrics(pred,gt,net):
    assert pred.shape==gt.shape and np.isfinite(pred).all()
    x=torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).to('cuda')*2-1
    y=torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).to('cuda')*2-1
    with torch.inference_mode(): lp=float(net(x,y,normalize=False,retPerLayer=False).item())
    vals=dict(PSNR=float(peak_signal_noise_ratio(gt,pred,data_range=1.0)),SSIM=float(structural_similarity(gt,pred,win_size=7,gradient=False,data_range=1.0,channel_axis=2,gaussian_weights=False,full=False,use_sample_covariance=True,K1=.01,K2=.03,sigma=1.5)),LPIPS=lp)
    assert all(np.isfinite(v) for v in vals.values()),vals
    return vals

def execute(bindings,m1,models,net):
    global stage,sample
    net=net.cuda()
    for domain in ['M0_ACDC','M1_LOLv2']:
        for mi,method in enumerate(METHODS if domain=='M0_ACDC' else METHODS[:2]):
            stage=domain+'/'+method;directory=OUT/domain/method;directory.mkdir(parents=True,exist_ok=False)
            items=[r for r in bindings if r['method']==method] if domain=='M0_ACDC' else m1
            n=106 if domain=='M0_ACDC' else 138;assert len(items)==n
            model=None
            if domain=='M1_LOLv2': model=models[mi].cuda().eval()
            start=now();t0=time.perf_counter();torch.cuda.reset_peak_memory_stats();records=[]
            with (directory/'per_image_metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
                fields=['sample_id','prediction_path','prediction_sha256','gt_sha256','PSNR','SSIM','LPIPS','seconds','status']
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for i,r in enumerate(items):
                    sample=r['sample_id'];ts=time.perf_counter()
                    if model is None: p=Path(r['prediction_path'])
                    else:
                        p=directory/'predictions'/f'{sample}.png';p.parent.mkdir(exist_ok=True)
                        assert not p.exists(),p
                        inp=rgb(r['input_path']);h,w=inp.shape[:2]
                        tensor=torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).cuda()
                        tensor=F.pad(tensor,(0,(-w)%4,0,(-h)%4),mode='reflect')
                        with torch.inference_mode(): output=model(tensor)[:,:,:h,:w].clamp(0,1)
                        arr=output.squeeze(0).permute(1,2,0).cpu().numpy()
                        assert np.isfinite(arr).all()
                        Image.fromarray(np.rint(np.clip(arr,0,1)*255).astype(np.uint8)).save(p,format='PNG')
                        del tensor,output,arr
                    vals=metrics(rgb(p),rgb(r['gt_path']),net)
                    rec=dict(sample_id=sample,prediction_path=str(p),prediction_sha256=sha(p),gt_sha256=r['gt_sha256'],**vals,seconds=time.perf_counter()-ts,status='PASS')
                    if model is None: assert rec['prediction_sha256']==r['prediction_sha256']
                    writer.writerow(rec);f.flush();records.append(rec)
                    all_records.append(dict(domain=domain,method=method,**rec))
                    if (i+1)%10==0 or i+1==n: print(f'{stage} {i+1}/{n} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
            assert len(records)==n
            timing=dict(domain=domain,method=method,start_utc=start,end_utc=now(),wall_seconds=time.perf_counter()-t0,expected=n,completed=n,failed=0,failure_fraction=0.0,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20,max_reserved_mib=torch.cuda.max_memory_reserved()/2**20,enhancer_inference=(model is not None))
            timings.append(timing);dump(directory/'COMPLETENESS.json',timing)
            if model is not None: model.cpu();torch.cuda.empty_cache()

def aggregate(oldtables,anchor):
    global stage
    stage='statistics'
    main=[];pairs=[];deltas=[];differences=[];directions=[]
    for domain in ['M0_ACDC','M1_LOLv2']:
        methods=METHODS if domain=='M0_ACDC' else METHODS[:2]
        n=106 if domain=='M0_ACDC' else 138
        rr={m:sorted([r for r in all_records if r['domain']==domain and r['method']==m],key=lambda r:r['sample_id']) for m in methods}
        ids=[r['sample_id'] for r in rr['Stage1']]
        assert len(ids)==len(set(ids))==n
        arrays={m:np.array([[r[k] for k in METRICS] for r in rr[m]],dtype=np.float64) for m in methods}
        indices=np.random.default_rng(20260908).integers(0,n,size=(10000,n))
        for m in methods:
            assert [r['sample_id'] for r in rr[m]]==ids
            for j,k in enumerate(METRICS):
                a=arrays[m][:,j]
                main.append(dict(domain=domain,method=m,role='selection-coupled reference' if m=='ACDC_Specialist' else 'fixed checkpoint',metric=k,n=n,mean=float(a.mean()),median=float(np.median(a)),population_sd=float(a.std()),minimum=float(a.min()),maximum=float(a.max()),failure_count=0,failure_fraction=0.0))
            if domain=='M0_ACDC':
                for r in rr[m]:
                    old=oldtables[m][r['sample_id']]
                    for k in METRICS: differences.append(dict(domain=domain,method=m,sample_id=r['sample_id'],metric=k,old=float(old[k]),new=r[k],new_minus_old=r[k]-float(old[k]),old_contract='A_FLOAT' if m in METHODS[:2] else 'B_PNG',interpretation='compound quantization/version/determinism bridge, not necessarily pure quantization'))
            elif m=='Stage1':
                for r in rr[m]:
                    for k in METRICS: differences.append(dict(domain=domain,method=m,sample_id=r['sample_id'],metric=k,old=float(anchor[r['sample_id']][k.lower()]),new=r[k],new_minus_old=r[k]-float(anchor[r['sample_id']][k.lower()]),old_contract='BasicSR anchor',interpretation='new PNG/skimage protocol; frozen retention gates NOT rejudged'))
        for other in methods[1:]:
            d=arrays['Stage1']-arrays[other]
            for i,key in enumerate(ids): deltas.append(dict(domain=domain,comparison='Stage1_minus_'+other,sample_id=key,PSNR=d[i,0],SSIM=d[i,1],LPIPS=d[i,2]))
            for j,k in enumerate(METRICS):
                a=d[:,j];b=a[indices];mean_ci=np.quantile(b.mean(axis=1),[.025,.975],method='linear');median_ci=np.quantile(np.median(b,axis=1),[.025,.975],method='linear')
                signed=a if k!='LPIPS' else -a
                wins=int((signed>1e-12).sum());losses=int((signed< -1e-12).sum());ties=int((abs(signed)<=1e-12).sum())
                pairs.append(dict(domain=domain,comparison='Stage1_minus_'+other,metric=k,n=n,mean_delta=float(a.mean()),mean_ci95_low=float(mean_ci[0]),mean_ci95_high=float(mean_ci[1]),median_delta=float(np.median(a)),median_ci95_low=float(median_ci[0]),median_ci95_high=float(median_ci[1]),wins=wins,losses=losses,ties=ties,win_fraction=wins/n,loss_fraction=losses/n,tie_fraction=ties/n,CI_type='marginal image-level percentile; NOT simultaneous',resamples=10000,seed=20260908))
                if domain=='M0_ACDC':
                    olddelta=np.array([float(oldtables['Stage1'][key][k])-float(oldtables[other][key][k]) for key in ids])
                    directions.append(dict(comparison='Stage1_minus_'+other,metric=k,old_mean_delta=float(olddelta.mean()),new_mean_delta=float(a.mean()),mean_direction_changed=bool(np.sign(olddelta.mean())!=np.sign(a.mean())),old_interpretation='mixed A_FLOAT/B_PNG arithmetic ONLY; no prior admissible paired inference' if other not in METHODS[:2] else 'old A_FLOAT reference'))
    csvwrite(OUT/'MAIN_TABLE.csv',main);csvwrite(OUT/'PAIRED_STATISTICS.csv',pairs);csvwrite(OUT/'PAIRED_DELTAS.csv',deltas)
    csvwrite(OUT/'NEW_OLD_PER_IMAGE_DIFFERENCES.csv',differences);csvwrite(OUT/'M0_DIRECTION_BRIDGE.csv',directions)
    grouped=[]
    for domain,m,k in sorted({(r['domain'],r['method'],r['metric']) for r in differences}):
        selected=[r for r in differences if (r['domain'],r['method'],r['metric'])==(domain,m,k)]
        grouped.append(dict(domain=domain,method=m,metric=k,n=len(selected),old_mean=float(np.mean([r['old'] for r in selected])),new_mean=float(np.mean([r['new'] for r in selected])),mean_change=float(np.mean([r['new_minus_old'] for r in selected])),max_absolute_change=float(np.max(np.abs([r['new_minus_old'] for r in selected]))),old_contract=selected[0]['old_contract']))
    csvwrite(OUT/'NEW_OLD_METRIC_CONTRACT_DIFFERENCES.csv',grouped);csvwrite(OUT/'RUNTIME_COMPLETENESS.csv',timings)
    for p,h in zip(CKPTS,CK_SHA): assert sha(p)==h
    seal=readjson(OUT/'PROTOCOL_SEAL.json')
    for name,h in seal['sha256'].items(): assert sha(OUT/name)==h
    csvwrite(OUT/'ALL_PER_IMAGE_METRICS.csv',all_records)
    outputs=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and not p.name.startswith('RUN_LOG'): outputs.append(dict(relative_path=norm(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    csvwrite(OUT/'EXECUTION_OUTPUT_MANIFEST.csv',outputs)
    summary=dict(status='M0_M1_COMMON_PROTOCOL_COMPLETE',finished_utc=now(),M0_methods=6,M0_each=106,M1_methods=2,M1_each=138,enhancer_forward_images=276,M0_enhancer_forward_images=0,metric_rows=len(all_records),main_table=main,paired_statistics=pairs,failures=0,CI='post-hoc descriptive marginal image-level, not simultaneous/independent-test',output_manifest_sha256=sha(OUT/'EXECUTION_OUTPUT_MANIFEST.csv'),protocol_sha256=sha(OUT/'COMMON_PROTOCOL.md'),runner_sha256=sha(__file__),canonical_synchronization='pending final verified handoff',server=False,formal_test=False,training=False,downloads=False)
    dump(OUT/'SUMMARY.json',summary)
    print('M0_M1_COMMON_PROTOCOL_COMPLETE',flush=True)

class Tee:
    def __init__(self,stream,handle): self.stream=stream;self.handle=handle
    def write(self,s): self.stream.write(s);self.handle.write(s);self.handle.flush()
    def flush(self): self.stream.flush();self.handle.flush()

if __name__=='__main__':
    log=(OUT/'RUN_LOG_RECOVERY_1.txt').open('x',encoding='utf-8')
    sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    try:
        b,v,models,net,old,anchor=preflight();execute(b,v,models,net);aggregate(old,anchor)
    except Exception as exc:
        dump(OUT/'FAILURE_RECOVERY_1.json',dict(status='FAILED_INCOMPLETE_OR_BLOCKED',time=now(),stage=stage,sample=sample,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),completed_metric_rows=len(all_records),completed_methods=timings))
        traceback.print_exc();sys.exit(1)
