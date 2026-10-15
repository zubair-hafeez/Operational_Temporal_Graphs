"""Compare identity and coherent shared calibration, using development only.

One positive slope/intercept across horizons preserves monotone raw CDFs.
Calibration-period predictions are recomputed from the saved earlier-fit model.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]


def fit_shared(p,y,weights):
    z=logit(np.clip(p,1e-6,1-1e-6));w=weights/weights.sum()/y.shape[1]
    def objective(ab):
        a,b=ab;eta=a*z+b;delta=(expit(eta)-y)*w[:,None]
        loss=np.sum((np.logaddexp(0,eta)-y*eta)*w[:,None])+.001*((a-1)**2+b*b)
        grad=np.array([(delta*z).sum()+.002*(a-1),delta.sum()+.002*b])
        return loss,grad
    opt=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.05,10),(-10,10)])
    if not opt.success:raise RuntimeError(opt.message)
    return opt.x


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--history',action='store_true');a=ap.parse_args()
    version='history_v1' if a.history else 'horizons_v1';tag='history_v1' if a.history else 'v1'
    src=ROOT/'data/processed'/f'{a.dataset}_development_{version}';q=pd.read_parquet(src/'queries.parquet');x=np.load(src/'features.npy',mmap_mode='r')
    horizons=[1,5,15,60] if a.dataset=='borg' else [1,3,7,14];unit='m' if a.dataset=='borg' else 'd';maximum=1/24 if a.dataset=='borg' else 14
    all_y=q[[f'exit_{h}{unit}' for h in horizons]].to_numpy();reports=[]
    with threadpool_limits(limits=4):
      for state in map(int,sorted(q.state.unique())):
       for arm in ['personal','global','relational']:
        run=ROOT/'runs'/f'{a.dataset}_marginals_{tag}_state{state}_{arm}_seed0'
        for folder in sorted(run.iterdir()):
            manifest_path=folder/'manifest.json'
            if not manifest_path.exists():continue
            m=json.loads(manifest_path.read_text())
            if m['status']!='complete':continue
            report_path=folder/'calibration_audit.json'
            if report_path.exists():reports.append(json.loads(report_path.read_text()));continue
            cal=(q.state==state)&(q.elapsed_day>=m['fit_end_day'])&(q.elapsed_day+maximum<=m['calibration_end_day'])
            ix=np.flatnonzero(cal);pp=[]
            for h in horizons:
                fitted=joblib.load(folder/f'horizon{h}{unit}.joblib')
                pp.append(fitted['model'].predict_proba(x[ix][:,fitted['cols']])[:,1])
            raw_cal=np.sort(np.column_stack(pp),axis=1);y=all_y[ix]
            node_w=q.loc[cal,'sampling_weight'].to_numpy()
            cutoff_w=1/q.loc[cal].groupby('cutoff_us').cutoff_us.transform('size').to_numpy()
            params={'shared_population':fit_shared(raw_cal,y,node_w),'shared_cutoff':fit_shared(raw_cal,y,cutoff_w)}
            original=np.load(folder/'predictions.npz');ids=original['query_ids'];raw=np.sort(original['raw'],axis=1)
            variants={'raw_sorted':raw,'individual_platt_sorted':original['cdf']}
            for name,(aa,bb) in params.items():variants[name]=expit(aa*logit(np.clip(raw,1e-6,1-1e-6))+bb).astype(np.float32)
            target=q.iloc[ids];tw=1/target.groupby('cutoff_us').cutoff_us.transform('size').to_numpy();yy=all_y[ids]
            metrics={}
            for name,p in variants.items():
                metrics[name]={'cutoff_mean_integrated_brier':float(np.average(((p-yy)**2).mean(1),weights=tw)),
                    'population_integrated_brier':float(np.average(((p-yy)**2).mean(1),weights=target.sampling_weight)),
                    'cutoff_mean_brier_by_horizon':np.average((p-yy)**2,weights=tw,axis=0).tolist()}
            np.savez_compressed(folder/'calibration_variants.npz',query_ids=ids,**variants)
            report={'dataset':a.dataset,'state':state,'arm':arm,'fold':folder.name,'params':{k:v.tolist() for k,v in params.items()},'metrics':metrics,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'marginal_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'calibration_contract':'Only earlier calibration-window labels. Coherent shared positive Platt map across all four horizons, identity-shrinkage penalty .001. Both population and cutoff weights retained.'}
            report_path.write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
            print(json.dumps({k:report[k] for k in ['dataset','state','arm','fold','metrics']}),flush=True)
    (ROOT/'reports'/f'{a.dataset}_{tag}_calibration_audit.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
