"""New input binding only; metric creation/runtime use unchanged sealed functions."""
from pathlib import Path
import sys,os,csv,json,hashlib,importlib.util,time,io,traceback,math
from datetime import datetime,timezone
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
P=ROOT/'outputs/V11_WEEKEND_SUPPLEMENT_20261016/iqa_preparation'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def now():return datetime.now(timezone.utc).isoformat()
def main():
 lock=read(HERE/'SOURCE_LOCK.json')
 for path,h in lock['files'].items():assert sha(ROOT/path)==h,path
 spec=importlib.util.spec_from_file_location('fixed_iqa',P/'run_fixed_iqa.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.freeze_runtime()
 import torch,numpy as np
 from PIL import Image
 assert torch.cuda.mem_get_info()[0]>2*1024**3,'GPU memory below 2GiB'
 if '--preflight' in sys.argv:
  assert not (HERE/'PREFLIGHT.json').exists()
  result={'status':'STARTED','started_utc':now(),'runner_sha256':sha(Path(__file__)),'real_scores_produced':0,'gpu':torch.cuda.get_device_name(0),'cuda_free_total_bytes':list(torch.cuda.mem_get_info())}
  try:
   x=torch.rand((1,3,1080,1920),dtype=torch.float32)
   ramp=(torch.linspace(0,1,1920)[None,None,None,:]+torch.linspace(0,1,1080)[None,None,:,None])/2
   y=(.1+.65*ramp+.15*torch.rand_like(x)).clamp(0,1)
   models=m.model_pair();old=read(P/'SYNTHETIC_PREFLIGHT.json');entries=[]
   for metric in ['niqe','musiq']:
    values=[]
    for v in [x,x,y]:
     t=time.monotonic()
     with torch.inference_mode():score=models[metric](v)
     torch.cuda.synchronize();assert score.numel()==1 and torch.isfinite(score).all()
     values.append({'score':float(score.item()),'seconds':time.monotonic()-t})
    original=next(q for q in old['metrics'] if q['metric']==metric)
    diffs=[abs(a['score']-b['score']) for a,b in zip(values,original['calls'])]
    assert max(diffs)<=1e-6,(metric,diffs)
    entries.append({'metric':metric,'calls':values,'max_diff_old_synthetic':max(diffs)})
   result.update(status='PASS',metrics=entries)
  except Exception:result.update(status='FAILED_PREFLIGHT',error=traceback.format_exc());raise
  finally:result['finished_utc']=now();save(HERE/'PREFLIGHT.json',result)
  print(json.dumps(result),flush=True);return
 auth=read(HERE/'RUN_AUTHORIZATION.json')
 assert auth['action']=='EXTERNAL_FIVE_530_IQA_ONCE' and auth['records']==530
 assert auth['source_lock_sha256']==sha(HERE/'SOURCE_LOCK.json') and auth['preflight_sha256']==sha(HERE/'PREFLIGHT.json')
 assert auth['runner_sha256']==sha(Path(__file__)) and read(HERE/'PREFLIGHT.json')['status']=='PASS'
 assert time.time()<auth['deadline_epoch']<=1792296000
 with (HERE/'INPUT_BINDINGS.csv').open(encoding='utf8',newline='') as f:bindings=list(csv.DictReader(f))
 assert len(bindings)==530
 out=HERE/'results_v1';out.mkdir(exist_ok=False)
 status={'status':'STARTED','completed_records':0,'metric_events':0,'started_utc':now(),'pid':os.getpid()};start=time.monotonic();complete=[]
 save(out/'RUN_STARTED.json',{**status,'authorization_sha256':sha(HERE/'RUN_AUTHORIZATION.json'),'gpu':torch.cuda.get_device_name(0),'deadline_epoch':auth['deadline_epoch']})
 try:
  models=m.model_pair();torch.cuda.reset_peak_memory_stats()
  with (out/'METRIC_EVENTS.jsonl').open('x',encoding='utf8') as events,(out/'COMPLETE_RECORDS.jsonl').open('x',encoding='utf8') as log:
   for index,row in enumerate(bindings):
    assert time.time()<auth['deadline_epoch']
    data=m.checked_bytes(row)
    with Image.open(io.BytesIO(data)) as im:
     assert im.format=='PNG' and im.mode=='RGB' and im.size==(1920,1080)
     array=np.array(im,dtype=np.uint8,copy=True)
    inp=torch.from_numpy(array).permute(2,0,1).unsqueeze(0).contiguous().to(dtype=torch.float32)/255.0
    assert tuple(inp.shape)==(1,3,1080,1920) and torch.isfinite(inp).all() and 0<=inp.min()<=inp.max()<=1
    entry=dict(row,index=index)
    for metric in ['niqe','musiq']:
     assert time.time()<auth['deadline_epoch']
     torch.cuda.synchronize();t=time.monotonic()
     with torch.inference_mode():score=models[metric](inp)
     torch.cuda.synchronize();assert score.numel()==1 and torch.isfinite(score).all()
     entry[metric]=float(score.item());entry[metric+'_seconds']=time.monotonic()-t;assert math.isfinite(entry[metric])
     m.append_event(events,{'index':index,'condition':row['condition'],'sample_id':row['sample_id'],'metric':metric,'score':entry[metric],'seconds':entry[metric+'_seconds'],'utc':now()});status['metric_events']+=1
    entry['pyiqa_version']='0.1.16';entry['protocol_sha256']=sha(HERE/'PROTOCOL_ZH.md');entry['resource_lock_sha256']=sha(P/'OFFICIAL_RESOURCE_LOCK.json')
    complete.append(entry);m.append_event(log,entry);status['completed_records']=len(complete)
    if len(complete)%20==0 or len(complete)%106==0:print('IQA',len(complete),'/530',row['condition'],flush=True)
  assert len(complete)==530 and status['metric_events']==1060
  m.write_csv_new(out/'PER_IMAGE.csv',complete);status.update(status='COMPLETE_530_1060',peak_reserved_bytes=torch.cuda.max_memory_reserved())
 except Exception:status.update(status='FAILED_INCOMPLETE',error=traceback.format_exc())
 finally:
  status.update(seconds=time.monotonic()-start,finished_utc=now());save(out/'RUN_RESULT.json',status)
  save(out/'OUTPUT_MANIFEST.json',[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file()])
 print(json.dumps(status),flush=True)
 if status['status']!='COMPLETE_530_1060':raise SystemExit(1)
if __name__=='__main__':main()
