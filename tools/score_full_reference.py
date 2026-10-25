"""Full-reference scoring with the exact archived metric and diagnostic function bodies."""
import argparse,csv
from pathlib import Path
from release_common import SEEDS,VARIANTS,rows,fresh,verify_sources,sources,metric_functions,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',required=True,type=Path,help='CSV: sample_id,input_path,gt_path,prediction_path,input_sha256,gt_sha256,prediction_sha256')
    p.add_argument('--alexnet-backbone',required=True,type=Path,help='Original alexnet-owt-7be5be79.pth; see docs/REPRODUCING.md')
    p.add_argument('--output',required=True)
    a=p.parse_args();verify_sources()
    folder,common,core,merged,component=sources(20261005,'MERGED_LUM_PCGRAD')
    common.setup(20260908)
    fn=metric_functions(common,core,a.alexnet_backbone)
    records=rows(a.pairs)
    if not records:raise ValueError('No image pairs')
    for row in records:
        for role in ('input','gt','prediction'):
            if digest(row[role+'_path'])!=row[role+'_sha256'].lower():raise ValueError('Identity mismatch: '+row['sample_id'])
    net=fn['metric_net']();out=fresh(a.output);result=[]
    for row in records:
        x,y,z=[fn['rgb'](row[k+'_path']) for k in ('input','gt','prediction')]
        if x.shape!=y.shape or y.shape!=z.shape:raise ValueError('Shape mismatch')
        result.append({'sample_id':row['sample_id'],**fn['metrics'](z,y,net),**fn['diagnostics'](x,y,z)})
    with (out/'per_image.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print('Scored',len(result),'pairs; no aggregation or checkpoint selection performed.')

if __name__=='__main__':main()
