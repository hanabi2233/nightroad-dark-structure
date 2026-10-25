"""Pure CPU numerical contract. No model imports and no image inference."""
from pathlib import Path, PurePosixPath
import csv, hashlib, json, stat, re
import numpy as np

CONDITIONS=('Input','Gamma','Stage1','Specialist','SCI','EnlightenGAN')
LEARNED=CONDITIONS[2:]
TOL=1e-12
CLASSES=('road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle')
IDS=(7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33)
MANIFEST=Path(__file__).resolve().parent/'VAL106_BINDING_MANIFEST.csv'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest().upper()

def plain_file(path):
    p=Path(path).absolute()
    for part in (p,*p.parents):
        s=part.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400: raise ValueError(f'link/reparse: {part}')
    if not stat.S_ISREG(p.stat().st_mode): raise ValueError(f'not regular: {p}')
    return p

def night_relative(value):
    value=value.replace('\\','/')
    if not re.fullmatch(r'rgb_anon/night/val/GOPR\d{4}/GOPR\d{4}_frame_\d{6}_rgb_anon\.png',value): raise ValueError('not an explicit night/val path')
    p=PurePosixPath(value)
    if p.parts[3]!=p.name.split('_frame_')[0]: raise ValueError('sequence mismatch')
    return value

def read_manifest(path=MANIFEST):
    path=plain_file(path)
    # Only this sealed task manifest is accepted, not any similarly shaped CSV.
    seal=MANIFEST.with_suffix('.sha256').read_text().strip()
    if digest(path)!=seal: raise ValueError('manifest identity mismatch')
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=106 or len({r['sample_id'] for r in rows})!=106: raise ValueError('106 unique members required')
    for r in rows:
        rel=night_relative(r['input_relative'])
        if r['sample_id']!=PurePosixPath(rel).stem or r['split']!='val': raise ValueError('member binding mismatch')
    return rows

def resolve_member(row,field,root):
    relative=row[field].replace('\\','/')
    p=PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or ':' in relative or not relative: raise ValueError('unsafe member')
    # Check logical image roots independent of OS and relocated physical roots.
    if field=='input_relative': night_relative(relative)
    elif field in ('labelTrainIds_relative','labelIds_relative','invIds_relative'):
        suffix=field.replace('_relative','')
        expected='gt/night/val/'+row['sequence']+'/'+row['sample_id'].replace('_rgb_anon','_gt_'+suffix)+'.png'
        if relative!=expected: raise ValueError('annotation mapping mismatch')
    elif field=='specialist_relative':
        if relative!=row['sequence']+'/'+row['sample_id']+'.png': raise ValueError('reuse member mismatch')
    else: raise ValueError('unknown field')
    root=Path(root).absolute(); result=plain_file(root/Path(*p.parts))
    if not result.resolve().is_relative_to(root.resolve()): raise ValueError('path escaped root')
    return result

def rgb_array(a):
    a=np.asarray(a)
    if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3: raise ValueError('RGB uint8 HWC required')
    return a

def gamma(a):
    # Float64 arithmetic and ties-to-even np.rint explicitly fixed, gamma=0.5.
    x=rgb_array(a).astype(np.float64)/255.
    return np.rint(np.clip(np.sqrt(x),0,1)*255).astype(np.uint8)

def map_ids(a):
    a=np.asarray(a)
    if a.ndim!=2 or a.dtype!=np.uint8: raise ValueError('labelIds uint8 required')
    if not np.isin(a,list(range(34))+[255]).all(): raise ValueError('illegal labelId')
    lut=np.full(256,255,np.uint8)
    for train_id,label_id in enumerate(IDS): lut[label_id]=train_id
    return lut[a]

def label_mask(gt,pred,inv=None,region='all'):
    gt=np.asarray(gt); pred=np.asarray(pred)
    if gt.shape!=pred.shape or gt.ndim!=2 or not np.issubdtype(gt.dtype,np.integer) or not np.issubdtype(pred.dtype,np.integer): raise ValueError('label geometry/type')
    if not np.isin(gt,list(range(19))+[255]).all(): raise ValueError('invalid GT id')
    # All predictions valid, including ignored pixels: never silently mask illegal predictions.
    if not ((pred>=0)&(pred<19)).all(): raise ValueError('illegal prediction id')
    mask=gt!=255
    if inv is not None:
        inv=np.asarray(inv)
        if inv.shape!=gt.shape or not np.isin(inv,[0,1]).all(): raise ValueError('invalid invIds')
    if region!='all':
        if inv is None or region not in ('inv0','inv1'): raise ValueError('invalid region')
        mask=mask&(inv==int(region[-1]))
    return mask

def confusion(gt,pred,inv=None,region='all'):
    mask=label_mask(gt,pred,inv,region)
    return np.bincount(19*np.asarray(gt)[mask].astype(np.int64)+np.asarray(pred)[mask],minlength=361).reshape(19,19)

def scores(matrix):
    c=np.asarray(matrix)
    if c.shape!=(19,19) or not np.issubdtype(c.dtype,np.integer) or (c<0).any(): raise ValueError('integer confusion required')
    tp=c.diagonal(); gt=c.sum(1); pred=c.sum(0); union=gt+pred-tp
    iou=np.divide(tp,union,out=np.full(19,np.nan),where=union>0)
    return {'mIoU':float(np.nanmean(iou)) if (union>0).any() else None,'defined_classes':int((union>0).sum()),'IoU':[None if np.isnan(x) else float(x) for x in iou],'TP':tp.tolist(),'FP':(pred-tp).tolist(),'FN':(gt-tp).tolist(),'GT_pixels':gt.tolist(),'union':union.tolist(),'valid_pixels':int(c.sum()),'aAcc':float(tp.sum()/c.sum()) if c.sum() else None}

def changes(gt,before,after,inv=None,region='all'):
    mask=label_mask(gt,before,inv,region); label_mask(gt,after,inv,region)
    n=int(mask.sum())
    b=(before==gt)&mask; a=(after==gt)&mask
    fixed=int((~b&a&mask).sum()); introduced=int((b&~a&mask).sum())
    other=int((~b&~a&(before!=after)&mask).sum())
    return {'valid_pixels':n,'corrected_pixels':fixed,'new_error_pixels':introduced,'still_wrong_changed_class':other,'corrected_rate':fixed/n if n else None,'new_error_rate':introduced/n if n else None,'delta_aAcc':(fixed-introduced)/n if n else None}

def attributes(rgb):
    x=rgb_array(rgb).astype(np.float64)/255; h,w=x.shape[:2]
    if min(h,w)<4: raise ValueError('4x4 blocks require image >=4x4')
    y=x@np.array([.2126,.7152,.0722])
    blocks=[y[i*h//4:(i+1)*h//4,j*w//4:(j+1)*w//4].mean() for i in range(4) for j in range(4)]
    return {'muY':float(y.mean()),'D05':float((y<.05).mean()),'S16':float(np.quantile(blocks,.9,method='linear')-np.quantile(blocks,.1,method='linear'))}

def summarize(values):
    a=np.asarray([x for x in values if x is not None],np.float64)
    if not np.isfinite(a).all(): raise ValueError('nonfinite statistic')
    if not len(a): return {'n':0,'status':'NA'}
    q=np.quantile(a,[.25,.5,.75],method='linear'); negative=a[a < -TOL]
    return {'n':len(a),'mean':float(a.mean()),'median':float(q[1]),'q1':float(q[0]),'q3':float(q[2]),'min':float(a.min()),'max':float(a.max()),'win':int((a>TOL).sum()),'loss':int((a<-TOL).sum()),'tie':int((abs(a)<=TOL).sum()),'negative':{'n':len(negative),'median':float(np.median(negative)) if len(negative) else None,'q1':float(np.quantile(negative,.25)) if len(negative) else None,'q3':float(np.quantile(negative,.75)) if len(negative) else None,'min':float(negative.min()) if len(negative) else None}}

def fixed_bins(rows,attribute):
    if len(rows)!=106: raise ValueError('106 required')
    ordered=sorted(rows,key=lambda x:(x[attribute],x['sample_id']))
    return {r['sample_id']:1 if i<35 else 2 if i<70 else 3 for i,r in enumerate(ordered)}

def select_cases(input_rows,delta_rows):
    result=[]
    for sequence in sorted({r['sequence'] for r in input_rows}):
        group=sorted([r for r in input_rows if r['sequence']==sequence],key=lambda r:(r['muY'],r['sample_id']))
        result.append({'role':'ordinary','sequence':sequence,'sample_id':group[(len(group)-1)//2]['sample_id']})
    for method in LEARNED:
        for sign in ('improvement','deterioration'):
            group=[r for r in delta_rows if r['method']==method and r['delta_aAcc'] is not None and (r['delta_aAcc']>TOL if sign=='improvement' else r['delta_aAcc']<-TOL)]
            if not group: result.append({'role':sign,'method':method,'sample_id':None,'status':'无'}); continue
            median=float(np.median([r['delta_aAcc'] for r in group]))
            chosen=min(group,key=lambda r:(abs(r['delta_aAcc']-median),r['sample_id']))
            result.append({'role':sign,'method':method,'sample_id':chosen['sample_id']})
    return {'selections':result,'unique_sample_ids':sorted({r['sample_id'] for r in result if r['sample_id'] is not None}),'all_conditions':list(CONDITIONS),'roi_fraction':[.25,.50,.75,.90]}

def require_complete(records,sample_ids):
    expected={(m,i) for m in CONDITIONS for i in sample_ids}
    actual=[(r['method'],r['sample_id']) for r in records if r.get('status')=='COMPLETE']
    if len(records)!=len(expected) or len(set(actual))!=len(actual) or set(actual)!=expected: raise ValueError('six-condition output incomplete/duplicate/extra')

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError('cannot infer empty CSV schema')
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
