"""Join a full-reference input list to its inference output manifest without scoring."""
import argparse,csv
from pathlib import Path
from release_common import rows,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--prediction-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path,help='New CSV file')
    a=p.parse_args()
    inputs=rows(a.inputs);outputs=rows(a.prediction_root/'OUTPUT_MANIFEST.csv')
    index={r['sample_id']:r for r in outputs}
    if len(index)!=len(outputs) or set(index)!={r['sample_id'] for r in inputs}:raise ValueError('Member mismatch')
    result=[]
    for r in inputs:
        if 'gt_path' not in r or 'gt_sha256' not in r:
            raise ValueError('No pixel-aligned reference: do not use real-night normal correspondences as GT')
        out=index[r['sample_id']];rel=Path(out['prediction'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe prediction path')
        path=a.prediction_root/rel
        if digest(path)!=out['prediction_sha256'].lower():raise ValueError('Prediction hash differs')
        result.append({**r,'prediction_path':str(path.resolve()),'prediction_sha256':out['prediction_sha256'].lower()})
    with a.output.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)

if __name__=='__main__':main()
