"""New two-task binding; original computation functions are imported unchanged."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,socket,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
AUTH=json.loads((ROOT/'RUN_AUTHORIZATION.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCE=Path('/root/autodl-tmp/PAPER_B_MERGED_REAL_NIGHT_VAL106_20261014_SEGMENT_CONT01/run_segment.py')
assert sha(SOURCE)==AUTH['source_sha256']
assert sha(Path(__file__))==AUTH['runner_sha256']
assert sha(ROOT/'PROTOCOL_ZH.md')==AUTH['protocol_sha256']
assert sha(ROOT/'LOCAL_CHECK_V2.json')==AUTH['local_check_sha256']
spec=importlib.util.spec_from_file_location('original_night',SOURCE)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.ROOT=ROOT;r.INFER_ROOT=ROOT;r.DEADLINE=AUTH['deadline_epoch']
OLD_CP=r.CHECKPOINTS
RUNS=Path('/root/autodl-tmp/V11_WEEKEND_SUPPLEMENT_20261016_RUN01')
LOCK=json.loads((ROOT/'LOCAL_CHECK_V2.json').read_text())
NEW_CP={str(s):(RUNS/f'train_{s}_MERGED_DROP_M/final_iter8000.pth',LOCK['files'][f'train_{s}_MERGED_DROP_M/final_iter8000.pth']['sha256'].upper()) for s in AUTH['seeds']}

def check():
 # Original asset check covers input/labels/packages and all reused baseline outputs.
 r.CHECKPOINTS=OLD_CP;r.check()
 for rel,item in LOCK['files'].items():r.identity(RUNS/rel,item['sha256'],item['bytes'])
 for seed,(p,h) in NEW_CP.items():
  record=json.loads((p.parent/'FINAL_CHECKPOINT.json').read_text())
  summary=json.loads((p.parent/'TRAIN_SUMMARY.json').read_text())
  assert record['seed']==int(seed) and record['variant']=='MERGED_DROP_M' and record['attempt']==8000
  assert record['sha256'].upper()==h and summary['attempted']==summary['committed']==8000
  counts={'ATTEMPT_START':0,'COMMIT':0}
  for line in (p.parent/'attempts.jsonl').open():
   event=json.loads(line).get('event');counts[event]=counts.get(event,0)+1
  assert counts['ATTEMPT_START']==counts['COMMIT']==8000,counts
 r.CHECKPOINTS=NEW_CP
 r.save(ROOT/'TWO_TASK_ASSET_CHECK.json',{'status':'PASS','variant':'MERGED_DROP_M','seeds':AUTH['seeds'],'files_checked':len(LOCK['files']),'no_training':True})

def main():
 assert str(ROOT)==AUTH['remote_root'] and socket.gethostname()==AUTH['hostname']
 r.tick()
 p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--evaluator');a=p.parse_args()
 if a.mode=='check':return check()
 assert json.loads((ROOT/'TWO_TASK_ASSET_CHECK.json').read_text())['status']=='PASS'
 r.CHECKPOINTS=NEW_CP
 if a.mode in ('infer-synthetic','infer'):r.infer(a.mode=='infer-synthetic')
 elif a.mode in ('seg-synthetic','segment'):r.segment(a.evaluator,a.mode=='seg-synthetic')
 else:raise ValueError('MODE')
if __name__=='__main__':
 try:main()
 except Exception:
  r.save(ROOT/('FAILURE_'+str(time.time_ns())+'.json'),{'error':traceback.format_exc()});raise
