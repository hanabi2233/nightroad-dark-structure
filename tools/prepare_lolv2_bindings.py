"""Bind the published 551/138 split to an author's lawful local copy of LOLv2-real."""
import argparse
from pathlib import Path
from release_common import ROOT, digest, rows, fresh, write_new, verify_sources

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low-root',type=Path,required=True,help='Real_captured/Train/Low')
    p.add_argument('--normal-root',type=Path,required=True,help='Real_captured/Train/Normal')
    p.add_argument('--output',required=True,help='A new directory outside this repository')
    a=p.parse_args();verify_sources()
    bindings={'train':{},'validation':{}}
    for row in rows(ROOT/'metadata/DATA_MANIFEST.csv'):
        sid=row['sample_id']
        candidates={'input':a.low_root/f'low{sid}.png','gt':a.normal_root/f'normal{sid}.png'}
        for role,path in candidates.items():
            if not path.is_file() or path.stat().st_size!=int(row[role+'_bytes']) or digest(path)!=row[role+'_sha256']:
                raise ValueError('Dataset identity mismatch: '+str(path))
        bindings[row['split']][sid]={k:str(v.resolve()) for k,v in candidates.items()}
    out=fresh(a.output)
    for split,content in bindings.items(): write_new(out/(split+'.json'),content)
    print('Verified train551 and validation138 bindings written to',out)

if __name__=='__main__': main()
