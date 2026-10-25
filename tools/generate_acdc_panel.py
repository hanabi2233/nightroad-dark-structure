"""Rebuild only the published 106-image synthetic panel using its recorded parameters."""
import argparse,csv,importlib.util,json
from pathlib import Path
from release_common import ROOT,digest,rows,fresh,verify_sources

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acdc-root',required=True,type=Path,help='Directory containing rgb_anon/')
    p.add_argument('--output',required=True)
    a=p.parse_args();verify_sources()
    import numpy as np
    from PIL import Image
    script=ROOT/'metadata/minimal_degradation_v1.py'
    spec=importlib.util.spec_from_file_location('frozen_degradation',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parameters=rows(ROOT/'metadata/ACDC_SYNTHETIC_VAL_V1.csv')
    panel={r['normal_member']:r for r in rows(ROOT/'metadata/ACDC_PANEL_MEMBERS.csv')}
    if len(parameters)!=106 or len(panel)!=106:raise ValueError('Panel size mismatch')
    out=fresh(a.output);records=[];mismatches=[]
    for row in parameters:
        source=a.acdc_root/row['source_relative_path']
        if digest(source)!=row['source_hash'].lower():raise ValueError('Normal-source identity mismatch')
        pars={}
        for key in ('exposure_parameters','illumination_parameters','noise_parameters'):pars.update(json.loads(row[key]))
        with Image.open(source) as im:x=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        y=module.apply_degradation(x,pars,int(row['sample_seed']))
        member=panel[row['source_relative_path']]
        png=out/(member['synthetic_sample_id']+'.png')
        Image.fromarray(np.rint(y*255).clip(0,255).astype(np.uint8),mode='RGB').save(png)
        actual=digest(png);expected=member['synthetic_png_sha256'].lower()
        records.append({'sample_id':member['synthetic_sample_id'],'input_path':str(png.resolve()),
                        'input_sha256':actual,'expected_original_png_sha256':expected,
                        'encoded_png_hash_match':actual==expected,'gt_path':str(source.resolve()),
                        'gt_sha256':row['source_hash'].lower()})
        if actual!=expected:mismatches.append(member['synthetic_sample_id'])
    with (out/'INPUTS.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    if mismatches:
        raise RuntimeError(f'{len(mismatches)}/106 PNG hashes differ from the original cache. '
                           'Do not label these files as the original panel; inspect NumPy/Pillow environment differences.')
    print('All 106 generated PNGs match the published cache hashes.')

if __name__=='__main__':main()
