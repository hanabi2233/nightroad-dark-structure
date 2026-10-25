"""Official MMSeg model/config wrapper. No silent key conversion or extra download."""
from pathlib import Path
import sys, socket, contextlib
import numpy as np
from nightroad_core import digest
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'resources/mmsegmentation'
CHECKPOINT=ROOT/'resources/segformer_b1.pth'
WEIGHT_SHA='655C7B3FE6E50E2D3AAD3690075EAF72C4852873D365D7BBCC322CEA3351270F'
CONFIG=SOURCE/'configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py'

@contextlib.contextmanager
def offline():
    original=socket.socket.connect
    def deny(*args,**kwargs): raise RuntimeError('NETWORK_FORBIDDEN_IN_MODEL_CHECK_OR_RUN')
    socket.socket.connect=deny
    try: yield
    finally: socket.socket.connect=original

def build(device='cpu'):
    import torch
    sys.path.insert(0,str(SOURCE))
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    register_all_modules()
    if digest(CHECKPOINT)!=WEIGHT_SHA: raise ValueError('SegFormer checkpoint SHA conflict')
    cfg=Config.fromfile(CONFIG)
    assert cfg.model.backbone.embed_dims==64
    assert cfg.model.decode_head.num_classes==19
    assert dict(cfg.model.test_cfg)==dict(mode='slide',crop_size=(1024,1024),stride=(768,768))
    assert cfg.model.data_preprocessor.mean==[123.675,116.28,103.53]
    assert cfg.model.data_preprocessor.std==[58.395,57.12,57.375]
    # Disable only unused pretrained initializer; full segmentation state is loaded below.
    # Do not call init_weights. This cannot create a missing-backbone fallback.
    cfg.model.pretrained=None;cfg.model.backbone.init_cfg=None
    with offline():
        model=MODELS.build(cfg.model)
        payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        state=payload['state_dict']
        result=model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys: raise ValueError(str(result))
        actual=model.state_dict()
        if set(actual)!=set(state): raise ValueError('complete key set mismatch')
        for key,value in state.items():
            if actual[key].shape!=value.shape or actual[key].dtype!=value.dtype or not torch.equal(actual[key],value): raise ValueError('tensor identity mismatch '+key)
            if value.is_floating_point() and not torch.isfinite(value).all(): raise ValueError('nonfinite weight '+key)
        model=model.to(device).eval()
    model.cfg=cfg
    record={'status':'PASS','device':device,'checkpoint_sha256':WEIGHT_SHA,'state_tensors':len(state),'parameters':sum(p.numel() for p in model.parameters()),'missing':0,'unexpected':0,'tensor_exact':True,'key_conversion':None,'backbone_download':False,'forward_executed':False,'source_commit':'c685fe6767c4cadf6b051983ca6208f1b9d1ccb8'}
    return model,record

def prepare_bgr(rgb):
    from mmcv.transforms import Compose
    # ndarray equivalent of official BGR LoadImageFromFile; same official Resize and PackSegInputs.
    pipeline=Compose([dict(type='LoadImageFromNDArray'),dict(type='Resize',scale=(2048,1024),keep_ratio=True),dict(type='PackSegInputs')])
    return pipeline(dict(img=np.ascontiguousarray(rgb[:,:,::-1])))

def predict(model,rgb):
    import torch
    data=prepare_bgr(rgb)
    with offline(),torch.inference_mode():
        result=model.test_step({'inputs':[data['inputs']],'data_samples':[data['data_samples']]})[0]
    p=result.pred_sem_seg.data[0].cpu().numpy()
    if p.shape!=rgb.shape[:2] or not np.isin(p,range(19)).all(): raise ValueError('prediction original-size/class contract')
    return p.astype(np.uint8)

if __name__=='__main__':
    import argparse,time,traceback
    from nightroad_core import write_json
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True);args=parser.parse_args()
    start=time.monotonic()
    try:
        _,record=build('cpu');record['seconds']=time.monotonic()-start
    except Exception:
        record={'status':'FAILED','seconds':time.monotonic()-start,'traceback':traceback.format_exc(),'forward_executed':False}
    write_json(args.record,record);print(record)
    if record['status']!='PASS': raise SystemExit(1)
