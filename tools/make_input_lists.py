"""Create explicit inference/scoring input lists from verified source images."""
import argparse,csv
from pathlib import Path
from release_common import ROOT,digest,rows,read,fresh,verify_sources

def save(path,records):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lolv2-validation-binding',type=Path)
    p.add_argument('--acdc-root',type=Path)
    p.add_argument('--synthetic-root',type=Path,help='Folder containing <synthetic_sample_id>.png')
    p.add_argument('--output',required=True)
    a=p.parse_args();verify_sources()
    if not a.lolv2_validation_binding and not a.acdc_root:p.error('Provide a validation binding and/or ACDC root')
    if a.synthetic_root and not a.acdc_root:p.error('--synthetic-root requires --acdc-root')
    lists={}
    if a.lolv2_validation_binding:
        binding=read(a.lolv2_validation_binding)
        members=[r for r in rows(ROOT/'metadata/DATA_MANIFEST.csv') if r['split']=='validation']
        if set(binding)!={r['sample_id'] for r in members}:raise ValueError('Validation membership differs')
        lists['lolv2_validation138.csv']=[]
        for r in members:
            paths=binding[r['sample_id']]
            for role in ('input','gt'):
                if digest(paths[role])!=r[role+'_sha256']:raise ValueError('LOLv2 identity differs')
            lists['lolv2_validation138.csv'].append({'sample_id':r['sample_id'],
                'input_path':paths['input'],'input_sha256':r['input_sha256'],
                'gt_path':paths['gt'],'gt_sha256':r['gt_sha256']})
    if a.acdc_root:
        lists['acdc_real_night106.csv']=[]
        if a.synthetic_root:lists['acdc_synthetic106.csv']=[]
        for r in rows(ROOT/'metadata/ACDC_PANEL_MEMBERS.csv'):
            night=a.acdc_root/r['night_member']
            if digest(night)!=r['night_encoded_sha256'].lower():raise ValueError('Night image identity differs')
            lists['acdc_real_night106.csv'].append({'sample_id':Path(r['night_member']).stem,
                'input_path':str(night.resolve()),'input_sha256':r['night_encoded_sha256'].lower()})
            if a.synthetic_root:
                normal=a.acdc_root/r['normal_member'];syn=a.synthetic_root/(r['synthetic_sample_id']+'.png')
                if digest(normal)!=r['normal_encoded_sha256'].lower() or digest(syn)!=r['synthetic_png_sha256'].lower():
                    raise ValueError('Synthetic panel identity differs')
                lists['acdc_synthetic106.csv'].append({'sample_id':r['synthetic_sample_id'],
                    'input_path':str(syn.resolve()),'input_sha256':r['synthetic_png_sha256'].lower(),
                    'gt_path':str(normal.resolve()),'gt_sha256':r['normal_encoded_sha256'].lower()})
    out=fresh(a.output)
    for name,records in lists.items():save(out/name,records)
    print('Created:',', '.join(lists))

if __name__=='__main__':main()
