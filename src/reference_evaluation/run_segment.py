"""Isolated fixed MERGED real-night inference and segmentation adapter."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, os, socket, sys, time, traceback

ROOT = Path('/root/autodl-tmp/PAPER_B_MERGED_REAL_NIGHT_VAL106_20261014_SEGMENT_CONT01')
INFER_ROOT = Path('/root/autodl-tmp/PAPER_B_MERGED_REAL_NIGHT_VAL106_20261014_RUN01')
BASE = Path('/root/autodl-tmp')
OLD_INFER = BASE/'PAPER_B_PCGRAD_ROAD_TRANSFER_20261005_RUN01/assets/road_fixed_infer.py'
INPUT_MANIFEST = OLD_INFER.parent/'REAL_NIGHT_INPUT_BINDING.csv'
ARCH = BASE/'PAPER_B_PCGRAD_STAGE1_20261004_RUN01/package/base/vendor/RetinexFormer_arch.py'
INPUT_ROOT = BASE/'paper_nightroad_phase1/data/raw/ACDC'
SEG_PACK = BASE/'PAPER_B_NIGHTROAD_SERVER_SYNTHETIC_V1_RUN01/package'
DEEP_PACK = BASE/'PAPER_B_NIGHTROAD_SECOND_SEGMENTER_V1_RUN01/package'
DATA = DEEP_PACK.parent/'data'
OLD_SEG = BASE/'PAPER_B_PCGRAD_REAL_NIGHT_SEGMENTATION_V1_RUN01'
DEADLINE = 1791970200  # 2026-10-14 17:30 Asia/Shanghai
CHECKPOINTS = {
 '20261004': (BASE/'PAPER_B_MERGED_LUM_20261011_RUN01/train_output/final_iter8000.pth', 'FC196F83CA50EBAF632F130CF0224154D637A03E2E8041A5638A0841BE270054'),
 '20261005': (BASE/'PAPER_B_PCGRAD_MULTISEED_20261012_RUN01/train_20261005_MERGED_LUM_PCGRAD/final_iter8000.pth', '8157410FC1D14216CB824F2B8DA7CAE28464232C2A9B149BBAF9EDC16A01F7F2'),
 '20261006': (BASE/'PAPER_B_PCGRAD_MERGED_20261006_RERUN_20261014_RUN01/train_output/final_iter8000.pth', 'FEB46FFE454DD3BA88244CD8E93A0CBAD1C85AD857392CD13A8920ACC0534FE5'),
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest().upper()

def save(p,obj):
 with Path(p).open('x',encoding='utf8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def identity(p,h,size=None):
 p=Path(p)
 if not p.is_file() or p.is_symlink() or (size is not None and p.stat().st_size!=int(size)) or sha(p)!=h.upper():raise ValueError('IDENTITY '+str(p))

def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def tick():
 if time.time()>=DEADLINE:raise TimeoutError('absolute deadline')
 import shutil
 if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('DISK below 2GiB; stop without cleanup')

def check():
 import shutil
 if socket.gethostname()!='autodl-container-d0884780b4-eb6dfa59':raise ValueError('HOST')
 if shutil.disk_usage(ROOT).free<5*1024**3:raise ValueError('DISK less than 5GiB')
 expected={OLD_INFER:'3FE13D65B8E54690AFB574221167183279DBD139F62C6FCB83BB15A308CB939C',INPUT_MANIFEST:'0AD6E71361580C8511D4229A7845059534F8E277CE74F30EF49938CD17409D52',ARCH:'1567C89D55285A3C1A4D4CA33F288B1FD98DC6B6BBDF971EDA02D745F9087631',DEEP_PACK/'VAL106_BINDING_MANIFEST.csv':'7B0C6E748BE9BF82A713356811A085A0660DF4429B790CD5B40F14D7BF88E0D3',SEG_PACK/'segmenter.py':'F083E7C231AC98174EB797247A1D236565A1B9428E9DA20DAC9A0F7BDC11D9FA',DEEP_PACK/'segmenter_deeplab.py':'EE7D5C80DC63CE0340064159D6C2DD79FAFE6D0820C61462A008F6A612B1FF68'}
 for p,h in expected.items():identity(p,h)
 for pack,name,h,key in [(SEG_PACK,'RESOURCE_SEAL.csv','D790B8DCBA4A37EF5B45E39EA4D4D0B6167D4A003A93A2311D5082D70190832A','relative_path'),(DEEP_PACK,'PACKAGE_MANIFEST.csv','6F8EF7EF2EEBB24EB0A1D1FBA56C6533F8CFE7F363BF63D9116B3409F22BE75F','path')]:
  identity(pack/name,h)
  for x in rows(pack/name):identity(pack/x[key],x['sha256'],x['bytes'])
 for pack in (SEG_PACK,DEEP_PACK):identity(pack/'nightroad_core.py','57CD3C21B1B874D25EBF4B43705E12D4C1588D064A25CD681EE445EBC53C5702')
 for seed,(p,h) in CHECKPOINTS.items():
  identity(p,h); record=json.loads((p.parent/'FINAL_CHECKPOINT.json').read_text());summary=json.loads((p.parent/'TRAIN_SUMMARY.json').read_text())
  if record['seed']!=int(seed) or record['variant']!='MERGED_LUM_PCGRAD' or record['attempt']!=8000 or record['sha256'].upper()!=h or summary['committed']!=8000 or summary['status']!='TRAIN_COMPLETE':raise ValueError('TRAIN_IDENTITY '+seed)
  print('CHECKPOINT',seed,record,flush=True)
 r=rows(INPUT_MANIFEST); labels=rows(DEEP_PACK/'VAL106_BINDING_MANIFEST.csv')
 if len(r)!=106 or len(labels)!=106 or {x['sample_id'] for x in r}!={x['sample_id'] for x in labels}:raise ValueError('MEMBERS')
 for x in r:identity(INPUT_ROOT/x['input_relative'],x['input_sha256'])
 for x in labels:
  for role in ('Input','labelTrainIds','labelIds','invIds'):
   prefix='input' if role=='Input' else role
   identity(DATA/role/(x['sample_id']+'.png'),x[prefix+'_sha256'],x[prefix+'_bytes'])
 for evaluator,h in [('segformer','208CA7AD280258938F24D0922CC65F446E85F811D09A9C9B31A1A1C6D8731DBA'),('deeplab','6C53D3A632D7C5A9EE48C8CCA7BC2D93B648E3029FB1B5C22587A77B95DE8F8A')]:
  old=OLD_SEG/('results_'+evaluator);identity(old/'OUTPUT_MANIFEST.json',h)
  for x in json.loads((old/'OUTPUT_MANIFEST.json').read_text()):identity(old/x['path'],x['sha256'],x['bytes'])
 save(ROOT/'ASSET_CHECK.json',{'status':'PASS','members':106,'seeds':list(CHECKPOINTS),'checked_at':time.time(),'files':{str(p):h for p,h in expected.items()},'free_bytes':shutil.disk_usage(ROOT).free})

def infer(synthetic=False):
 import numpy as np,torch
 from PIL import Image
 m=load(OLD_INFER,'frozen_infer');m.EXPECTED_CHECKPOINTS={s:h for s,(p,h) in CHECKPOINTS.items()}
 if synthetic:
  model=m.load_model(ARCH,CHECKPOINTS['20261004'][0],'20261004');t=time.monotonic();out=m.predict(model,Image.fromarray(np.zeros((1080,1920,3),np.uint8)));torch.cuda.synchronize()
  if out.size!=(1920,1080):raise ValueError('SYNTHETIC_SHAPE')
  save(ROOT/'INFER_SYNTHETIC.json',{'status':'PASS','seconds':time.monotonic()-t,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)});return
 if not (ROOT/'ASSET_CHECK.json').exists() or not (ROOT/'INFER_SYNTHETIC.json').exists():raise ValueError('GATE')
 for seed,(p,h) in CHECKPOINTS.items():
  tick();print('INFER_START',seed,flush=True)
  a=argparse.Namespace(variant=seed,architecture=ARCH,checkpoint=p,synthetic_only=False,manifest=INPUT_MANIFEST,input_root=INPUT_ROOT,output=INFER_ROOT/'enhanced'/seed,cohort='real_night')
  original=m.predict
  def bounded(model,image):tick();return original(model,image)
  m.predict=bounded
  try:m.run(a)
  finally:m.predict=original
  print('INFER_COMPLETE',seed,flush=True)

def segment(evaluator,synthetic=False):
 import numpy as np,torch
 from PIL import Image
 pack=SEG_PACK if evaluator=='segformer' else DEEP_PACK
 sys.path.insert(0,str(pack));import nightroad_core as core
 core.MANIFEST=DEEP_PACK/'VAL106_BINDING_MANIFEST.csv'  # Explicit frozen manifest and its existing seal location.
 wrapper=load(pack/('segmenter.py' if evaluator=='segformer' else 'segmenter_deeplab.py'),'fixed_wrapper')
 torch.manual_seed(2026);np.random.seed(2026);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 model,rec=wrapper.build('cuda:0');start=time.monotonic()
 if synthetic:
  pred=wrapper.predict(model,np.zeros((1080,1920,3),np.uint8));torch.cuda.synchronize()
  if pred.shape!=(1080,1920) or pred.dtype!=np.uint8:raise ValueError('SEG_SHAPE')
  save(ROOT/('SYNTHETIC_'+evaluator+'.json'),{'status':'PASS','load':rec,'seconds':time.monotonic()-start,'torch':torch.__version__});return
 if not (INFER_ROOT/('SYNTHETIC_'+evaluator+'.json')).exists():raise ValueError('SEG_GATE')
 rr=core.read_manifest(DEEP_PACK/'VAL106_BINDING_MANIFEST.csv')
 old=OLD_SEG/('results_'+evaluator)
 def png(p,rgb=False):
  with Image.open(p) as im:
   if im.format!='PNG' or (rgb and im.mode!='RGB'):raise ValueError('PNG_CONTRACT')
   a=np.array(im)
  if a.dtype!=np.uint8:raise ValueError('PNG_DTYPE')
  return a
 sid=rr[0]['sample_id'];canary=wrapper.predict(model,png(DATA/'Input'/(sid+'.png'),True));reference=png(old/'segmented/Input'/(sid+'.png'))
 if not np.array_equal(canary,reference):raise ValueError('OLD_INPUT_PIXEL_EQUIVALENCE_FAILED')
 save(ROOT/('EQUIVALENCE_'+evaluator+'.json'),{'status':'PASS','sample_id':sid,'pixel_exact':True,'load':rec})
 for seed in CHECKPOINTS:
  tick();out=ROOT/('segmentation_'+evaluator)/seed;out.mkdir(parents=True,exist_ok=False);(out/'segmented').mkdir();mats={};per=[];deltas=[];pred_records=[]
  manifest={x['sample_id']:x for x in rows(INFER_ROOT/'enhanced'/seed/'OUTPUT_MANIFEST.csv')}
  if set(manifest)!={x['sample_id'] for x in rr}:raise ValueError('ENHANCED_MEMBERS')
  for i,x in enumerate(rr):
   tick();sid=x['sample_id'];p=INFER_ROOT/'enhanced'/seed/'predictions'/(sid+'.png');v=manifest[sid];identity(p,v['output_sha256'],v['output_bytes'])
   if v['input_sha256']!=x['input_sha256']:raise ValueError('INPUT_BINDING')
   gt=png(DATA/'labelTrainIds'/(sid+'.png'));ids=png(DATA/'labelIds'/(sid+'.png'));inv=png(DATA/'invIds'/(sid+'.png'))
   if not np.array_equal(gt,core.map_ids(ids)):raise ValueError('LABEL_MAPPING')
   image=png(p,True)
   if image.shape!=(int(x['height']),int(x['width']),3):raise ValueError('INPUT_SHAPE')
   pred=wrapper.predict(model,image);torch.cuda.synchronize();dst=out/'segmented'/(sid+'.png')
   with dst.open('xb') as f:Image.fromarray(pred).save(f,format='PNG')
   pred_records.append({'sample_id':sid,'bytes':dst.stat().st_size,'sha256':sha(dst)})
   for region in ('all','inv0','inv1'):
    mat=core.confusion(gt,pred,inv,region);score=core.scores(mat)
    per.append({'seed':seed,'sample_id':sid,'sequence':x['sequence'],'region':region,'mIoU':score['mIoU'],'aAcc':score['aAcc'],'valid_pixels':score['valid_pixels']})
    for group in ('ALL',x['sequence']):
     key=(group,region);mats[key]=mats.get(key,np.zeros((19,19),np.int64))+mat
   for method in (('Input','A','SUM4','B') if seed=='20261004' else ('Input',)):
    ref=png(old/'segmented'/method/(sid+'.png'))
    for region in ('all','inv0','inv1'):deltas.append({'seed':seed,'reference':method,'sample_id':sid,'region':region,**core.changes(gt,ref,pred,inv,region)})
   if (i+1)%10==0:print('SEG',evaluator,seed,i+1,'/106',flush=True)
  summary=[];classes=[]
  for (group,region),mat in mats.items():
   score=core.scores(mat);summary.append({'seed':seed,'evaluator':evaluator,'group':group,'region':region,**{k:score[k] for k in ('mIoU','aAcc','valid_pixels','defined_classes')}})
   for k,name in enumerate(core.CLASSES):classes.append({'seed':seed,'evaluator':evaluator,'group':group,'region':region,'class_name':name,**{n:score[n][k] for n in ('IoU','GT_pixels','TP','FP','FN')}})
  for name,value in [('SUMMARY',summary),('CLASSES',classes),('PER_IMAGE',per),('DELTAS',deltas),('PREDICTIONS',pred_records)]:save(out/(name+'.json'),value)
  save(out/'COMPLETION.json',{'status':'COMPLETE','seed':seed,'evaluator':evaluator,'members':106,'checkpoint_sha256':CHECKPOINTS[seed][1],'elapsed_cumulative_seconds':time.monotonic()-start})
  print('SEG_COMPLETE',evaluator,seed,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','segment']);p.add_argument('--evaluator',choices=['segformer','deeplab']);a=p.parse_args();tick()
 if socket.gethostname()!='autodl-container-d0884780b4-eb6dfa59':raise ValueError('HOST')
 if a.mode=='check':check()
 elif a.mode.startswith('infer'):infer(a.mode=='infer-synthetic')
 elif a.mode in ('seg-synthetic','segment'):segment(a.evaluator,a.mode=='seg-synthetic')
 else:
  files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ('stdout.log','RUN_PID.txt','FINAL_MANIFEST.json'))
  save(ROOT/'FINAL_MANIFEST.json',[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files])

if __name__=='__main__':
 try:main()
 except Exception:
  failure=ROOT/('FAILURE_'+str(time.time_ns())+'.json');save(failure,{'status':'FAILED_INCOMPLETE','time':time.time(),'error':traceback.format_exc()});raise
