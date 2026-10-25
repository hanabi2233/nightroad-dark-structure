"""Read-only release, metadata and optional binary integrity check; no ML imports."""
import argparse, json
from pathlib import Path
from release_common import ROOT, digest, rows, verify_sources

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-assets',type=Path)
    a=p.parse_args()
    checked=verify_sources()
    full_manifest=ROOT/'SHA256SUMS.csv'
    all_checked=0
    if full_manifest.is_file():
        for r in rows(full_manifest):
            f=ROOT/r['path']
            if not f.is_file() or f.stat().st_size!=int(r['bytes']) or digest(f)!=r['sha256'].lower():
                raise ValueError('Release file mismatch: '+r['path'])
            all_checked+=1
    members=rows(ROOT/'metadata/DATA_MANIFEST.csv')
    counts={s:sum(r['split']==s for r in members) for s in ('train','validation')}
    assert counts=={'train':551,'validation':138}
    groups={s:{r['group_id'] for r in members if r['split']==s} for s in counts}
    assert not groups['train'] & groups['validation']
    records=rows(ROOT/'metadata/TRAINING_INSTANCES.csv')
    assert len(records)==21 and len({(r['seed'],r['implementation_variant']) for r in records})==21
    result={'source_files_verified':checked,'release_files_verified':all_checked,
            'split_counts':counts,'training_instances':len(records)}
    if a.release_assets:
        manifest=a.release_assets/'MODEL_ASSETS_MANIFEST.csv'
        if digest(manifest)!=digest(ROOT/'MODEL_ASSETS_MANIFEST.csv'):
            raise ValueError('Model asset manifest differs from the prepared release')
        assets=rows(manifest)
        for r in assets:
            f=a.release_assets/r['relative_path']
            if not f.is_file() or f.stat().st_size!=int(r['bytes']) or digest(f)!=r['expected_sha256'].lower(): raise ValueError('Asset mismatch: '+str(f))
        result['assets_verified']=len(assets)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
