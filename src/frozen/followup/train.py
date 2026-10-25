"""Explicitly authorized future server training. No automatic resume or validation."""
import argparse, random, time, traceback
from common import *

def environment():
    import platform, subprocess, torch, numpy, scipy, einops, cv2, PIL
    expected=read(HERE/'SERVER_ENVIRONMENT_REFERENCE.json')['records'][1]
    actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
                scipy=scipy.__version__,einops=einops.__version__,opencv=cv2.__version__,Pillow=PIL.__version__,cuda_build=torch.version.cuda)
    if any(actual[k]!=expected[k] for k in actual): raise ValueError('TRAIN_ENVIRONMENT_VERSION_MISMATCH: '+str(actual))
    chk=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    if chk.returncode: raise ValueError('PIP_CHECK: '+chk.stdout+chk.stderr)
    if not torch.cuda.is_available(): raise ValueError('CUDA_UNAVAILABLE')
    torch.cuda.set_device(0); free,total=torch.cuda.mem_get_info(0)
    if total<24_000_000_000 or free<20*1024**3: raise ValueError('DEDICATED_GPU_MEMORY_CONTRACT')
    if '4090' not in torch.cuda.get_device_name(0): raise ValueError('GPU_SPEC_NOT_APPROVED')
    actual.update(gpu=torch.cuda.get_device_name(0),total_bytes=total,free_bytes=free,pip_check=chk.stdout)
    return actual

def run(a):
    authorize(a.authorization,'train',a.seed,a.variant); verify_package(); check_deadline(a.deadline)
    out=fresh_output(a.output); start=time.time(); attempt=commits=nonzero=0
    try:
        import torch, core
        setup(a.seed); dump(out/'CONFIGURATION.json',configuration(a.seed,a.variant))
        dump(out/'ENVIRONMENT.json',environment())
        binding,members=binding_check(a.binding,'train')
        dump(out/'INPUT_BINDING.json',dict(split='train551',binding_sha256=sha(a.binding),files=1102))
        data=TrainingData(binding,members,a.seed); model,identity=sealed_model(a.seed)
        model=model.cuda(0).train(); torch.cuda.manual_seed_all(a.seed)
        from sealed_initialization import verify_state_mapping
        verify_state_mapping(model.state_dict(),read(HERE/'initialization'/str(a.seed)/'IDENTITY.json'),'cuda_loaded')
        dump(out/'INITIALIZATION.json',identity)
        adam=core.AdamCandidate(core.flatten([p.detach() for p in model.parameters()])); rng=random.Random(a.seed)
        torch.cuda.reset_peak_memory_stats()
        for index in range(STEPS):
            check_deadline(a.deadline); attempt=index+1; t=time.perf_counter()
            x,y,sample=data.batch(index)
            event(out/'attempts.jsonl',dict(event='ATTEMPT_START',attempt=attempt,variant=a.variant,seed=a.seed,**sample))
            x=x.cuda(0); y=y.cuda(0); torch.cuda.synchronize(); prepared=time.perf_counter()
            detail,tensors=update(model,x,y,adam,a.variant,attempt,rng)
            torch.cuda.synchronize(); computed=time.perf_counter()
            if adam.step_count!=attempt: raise ValueError('OPTIMIZER_COUNTER')
            trace=None; trace_seconds=0.
            if attempt in (500,501,8000):
                ts=time.perf_counter(); trace=f'trace_{attempt:04d}.pt'; save_trace(out/trace,tensors,adam)
                trace_seconds=time.perf_counter()-ts
            commits+=1; nonzero+=int(detail['nonzero_update'])
            event(out/'attempts.jsonl',dict(event='COMMIT',attempt=attempt,seed=a.seed,
                    four=four(a.variant,attempt),data_seconds=prepared-t,core_seconds=computed-prepared,
                    trace_seconds=trace_seconds,trace=trace,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),**detail))
            del x,y,tensors
            if attempt%100==0: print(f'{a.seed} {a.variant} {attempt}/{STEPS} {time.time()-start:.1f}s',flush=True)
        check_deadline(a.deadline)
        state=model.cpu().state_dict(); p=out/'final_iter8000.pth'
        with p.open('xb') as f: torch.save(state,f)
        dump(out/'FINAL_CHECKPOINT.json',dict(role='NEW_SCRATCH_FIXED_LAST_METHOD_FOLLOWUP_V2',seed=a.seed,variant=a.variant,
              attempt=STEPS,sha256=sha(p),initializer_sha256=identity['file_sha256'],selection='fixed last; no intermediate validation'))
        status='TRAIN_COMPLETE'
    except Exception as error:
        status='FAILED_INCOMPLETE'; dump(out/'FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),attempt=attempt))
        print(traceback.format_exc(),flush=True)
    dump(out/'TRAIN_SUMMARY.json',dict(status=status,seed=a.seed,variant=a.variant,attempted=attempt,
          committed=commits,nonzero=nonzero,elapsed_seconds=time.time()-start))
    return 0 if status=='TRAIN_COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True,choices=SEEDS)
    p.add_argument('--variant',required=True,choices=VARIANTS); p.add_argument('--binding',required=True)
    p.add_argument('--output',required=True); p.add_argument('--authorization',required=True); p.add_argument('--deadline',type=float,required=True)
    sys.exit(run(p.parse_args()))
