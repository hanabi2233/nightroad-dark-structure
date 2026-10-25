"""Fixed DeepLabV3+; no training initialization, key conversion or downloads."""
from pathlib import Path
import sys, socket, contextlib
import numpy as np
import nightroad_core as c

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
WEIGHT=ROOT/'deeplabv3plus_r50.pth'
SHA='D222FFCDBEFDC6D9396B30FD66A61401EFAB836DB8B991258C64A85C79A0EC21'
CONFIG=SOURCE/'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py'

@contextlib.contextmanager
def offline():
    original=socket.socket.connect
    def deny(*a,**k):raise RuntimeError('NETWORK_FORBIDDEN')
    socket.socket.connect=deny
    try:yield
    finally:socket.socket.connect=original

def build(device='cpu'):
    import torch
    from mmengine.config import Config
    sys.path.insert(0,str(SOURCE))
    from mmseg.utils import register_all_modules
    from mmseg.registry import MODELS
    register_all_modules()
    if c.digest(c.plain_file(WEIGHT))!=SHA:raise ValueError('checkpoint SHA conflict')
    cfg=Config.fromfile(CONFIG)
    assert cfg.model.backbone.type=='ResNetV1c' and cfg.model.backbone.depth==50
    assert cfg.model.decode_head.num_classes==cfg.model.auxiliary_head.num_classes==19
    assert dict(cfg.model.test_cfg)==dict(mode='whole')
    assert cfg.model.data_preprocessor.mean==[123.675,116.28,103.53]
    assert cfg.model.data_preprocessor.std==[58.395,57.12,57.375]
    assert cfg.model.data_preprocessor.bgr_to_rgb is True
    assert cfg.model.decode_head.align_corners is False
    assert cfg.test_pipeline[1].scale==(2048,1024) and cfg.test_pipeline[1].keep_ratio
    cfg.model.pretrained=None
    cfg.model.backbone.init_cfg=None
    with offline():
        model=MODELS.build(cfg.model)
        payload=torch.load(WEIGHT,map_location='cpu',weights_only=True)
        state=payload['state_dict']
        actual=model.state_dict()
        if set(actual)!=set(state):
            raise ValueError({'missing':sorted(set(actual)-set(state)), 'unexpected':sorted(set(state)-set(actual))})
        # Explicit precheck prevents old PyTorch BN compatibility hooks silently filling counters.
        for key,value in state.items():
            if not isinstance(value,torch.Tensor):raise TypeError(key)
            if actual[key].shape!=value.shape or actual[key].dtype!=value.dtype:raise ValueError('shape/dtype '+key)
        result=model.load_state_dict(state,strict=True)
        if result.missing_keys or result.unexpected_keys:raise ValueError(str(result))
        for key,value in model.state_dict().items():
            if not torch.equal(value,state[key]):raise ValueError('tensor mismatch '+key)
            if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('nonfinite '+key)
        model=model.to(device).eval()
    model.cfg=cfg
    return model,dict(status='PASS',device=str(device),state_tensors=len(state),missing=0,unexpected=0,
                      tensor_exact=True,key_conversion=None,checkpoint_sha256=SHA,
                      parameters=sum(p.numel() for p in model.parameters()),
                      pretrained_initializer_disabled=True,init_weights_called=False,mode='whole',
                      real_image_forward=False)

def prepare_bgr(rgb):
    from mmcv.transforms import Compose
    rgb=c.rgb_array(rgb)
    pipeline=Compose([dict(type='LoadImageFromNDArray'),dict(type='Resize',scale=(2048,1024),keep_ratio=True),dict(type='PackSegInputs')])
    return pipeline(dict(img=np.ascontiguousarray(rgb[:,:,::-1])))

def predict(model,rgb):
    import torch
    data=prepare_bgr(rgb)
    with offline(),torch.inference_mode():
        out=model.test_step({'inputs':[data['inputs']],'data_samples':[data['data_samples']]})[0]
    if not torch.isfinite(out.seg_logits.data).all():raise ValueError('nonfinite logits')
    pred=out.pred_sem_seg.data[0].cpu().numpy()
    if pred.shape!=rgb.shape[:2] or not np.isin(pred,range(19)).all():raise ValueError('shape/class conflict')
    return pred.astype(np.uint8)

if __name__=='__main__':
    import argparse,time,traceback
    ap=argparse.ArgumentParser();ap.add_argument('--record',type=Path,required=True);args=ap.parse_args()
    start=time.monotonic()
    try:
        _,r=build('cpu');r['seconds']=time.monotonic()-start
    except Exception:
        r=dict(status='FAILED',seconds=time.monotonic()-start,error=traceback.format_exc())
    c.write_json(args.record,r);print(r)
    if r['status']!='PASS':raise SystemExit(1)
