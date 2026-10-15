"""Fit/evaluate a simple direct aggregate historical-error baseline."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.residuals import aggregate_inputs,residual_counts
from otg.cohort_scores import score_projected_counts
from otg.protocol import verify_final_protocol
ROOT=Path(__file__).resolve().parents[1]


def read(data,c,m):
    folder=data/c['folder'];q=pd.read_parquet(folder/'queries.parquet');p=np.load(folder/'cdf.npy');unit='m' if m['horizon_unit']=='minutes' else 'd'
    y=q[[f'exit_{h}{unit}' for h in m['horizons']]].to_numpy()
    return aggregate_inputs(p,q.state.to_numpy(),q[m['relations']].to_numpy(),m['relations'],y)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['borg','bpi']);ap.add_argument('--period',choices=['development','final'],default='development');ap.add_argument('--draws',type=int,default=512);a=ap.parse_args()
    if a.period=='final':verify_final_protocol(ROOT)
    data=ROOT/'data/processed'/f'{a.dataset}_joint_selected_v2';m=json.loads((data/'metadata.json').read_text());out=ROOT/'runs'/f'aggregate_residual_{a.dataset}_selected_v2';start=time.monotonic()
    if a.period=='development':
        out.mkdir(exist_ok=False);cm=json.loads((data/'cohorts/manifest.json').read_text());flat={};origins=[]
        for i,c in enumerate(cm['cohorts']):
            _,v=read(data/'cohorts',c,m);origins.append(c['cutoff_us'])
            for name,z in v.items():
                flat[f'{i}_{name}_size']=z['size'];flat[f'{i}_{name}_residual']=(z['truth']-z['mean'])/z['scale']
        np.savez_compressed(out/'templates.npz',**flat)
        fit={'origins':origins,'projections':list(v),'source_sha256':hashlib.sha256((data/'cohorts/manifest.json').read_bytes()).hexdigest(),
             'fitting':'Development residuals; leave-one-origin-out development scores. Final uses all development origins. No parameter selection. Aggregate forecasts need not preserve marginal CDFs or be jointly realizable across overlapping projections.'}
        (out/'fit_manifest.json').write_text(json.dumps(fit,indent=2)+'\n')
    fit=json.loads((out/'fit_manifest.json').read_text());z=np.load(out/'templates.npz')
    templates=[{name:{key:z[f'{i}_{name}_{key}'] for key in ['size','residual']} for name in fit['projections']} for i in range(len(fit['origins']))]
    directory=data/('cohorts' if a.period=='development' else 'final_cohorts');cm=json.loads((directory/'manifest.json').read_text());rows=[]
    output=out/f'{a.period}_scores_m{a.draws}.parquet'
    if output.exists():raise FileExistsError(output)
    for i,c in enumerate(cm['cohorts']):
        projection,inputs=read(directory,c,m);reference=[v for j,v in enumerate(templates) if a.period=='final' or fit['origins'][j]!=c['cutoff_us']]
        for seed in range(3):
            counts=residual_counts(inputs,reference,a.draws,700000+seed*10000+i)
            scores=score_projected_counts(projection,counts,{name:v['truth'] for name,v in inputs.items()})
            rows.extend([{**r,'family':'empirical','mode':'historical_residual','seed':seed,'cutoff_us':c['cutoff_us'],'horizon':m['horizons'][r['horizon_index']]} for r in scores])
    frame=pd.DataFrame(rows);frame.to_parquet(output,index=False)
    report={'config':vars(a),'n_origins':len(cm['cohorts']),'elapsed_s':time.monotonic()-start,'macro':frame.groupby('cutoff_us').mean(numeric_only=True).mean().to_dict(),'scores_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (out/f'{a.period}_report_m{a.draws}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
