"""Shared-factor Gaussian-copula BASELINE, not a claimed novel architecture.

    Tests joint scenario quality while preserving identical calibrated marginal
    probabilities. Labels and scores concern only the sampled query population.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from otg.scores import ensemble_crps,interval_statistics
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True)
    ap.add_argument('--draws',type=int,default=256);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
    ds='borg_pilot_h15_step60_n1000' if a.dataset=='borg' else 'bpi_pilot_h7d'
    run=ROOT/'runs'/f'pilot_hgb_seed0_{ds}'
    pred=pd.read_parquet(run/'predictions.parquet')
    pred=pred[pred.arm=='relational'].copy()
    q=pd.read_parquet(ROOT/'data/processed'/ds/'queries.parquet')
    if a.dataset=='borg':
        taskmap=pd.read_parquet(ROOT/'data/processed/borg_prefix_7d/task_index.parquet',columns=['task','job_id'])
        q=q.merge(taskmap,on='task',validate='many_to_one');q['relation']=q.job_id
    else:
        meta=json.loads((ROOT/'data/processed/bpi_before_20161101/metadata.json').read_text())
        q['relation']=q.resource.astype(np.int64)
        # Every unknown-resource case is its own singleton, never one cluster.
        q.loc[q.resource==meta['resources'],'relation']=-1-q.loc[q.resource==meta['resources'],'case']
    pred=pred.merge(q[['query_id','relation']],on='query_id',validate='many_to_one')
    out=ROOT/'runs'/f'pilot_copula_{a.dataset}_seed{a.seed}';out.mkdir(parents=True,exist_ok=False)
    configs=[(0,0),(.05,0),(.2,0),(.5,0),(0,.2),(0,.5),(0,.8),(.05,.2),(.05,.5),(.2,.5)]
    rows=[];calibrations=[];rng=np.random.default_rng(a.seed);start=time.monotonic()
    for state in sorted(pred.state.unique()):
        d=pred[pred.state==state].copy();p=np.clip(d.probability.to_numpy(),1e-6,1-1e-6)
        z=np.log(p/(1-p))[:,None];val=d.split=='validation'
        with threadpool_limits(limits=4):
            cal=LogisticRegression(C=1,max_iter=1000).fit(z[val],d.loc[val,'exit'],sample_weight=d.loc[val,'sampling_weight']/d.loc[val,'sampling_weight'].mean())
            d['p']=cal.predict_proba(z)[:,1]
        calibrations.append({'state':int(state),'intercept':cal.intercept_.tolist(),'coef':cal.coef_.tolist()})
        for (split,t),f in d.groupby(['split','cutoff_us'],sort=True):
            group,_=pd.factorize(f.relation,sort=False);n=len(f);ng=group.max()+1
            counts=np.bincount(group);eligible=np.flatnonzero(counts>=5)
            y=f['exit'].to_numpy();p=f.p.to_numpy();threshold=ndtri(p)
            observed=np.bincount(group,weights=y,minlength=ng)
            # Identical underlying draws across configurations reduce MC noise.
            eps=rng.standard_normal((a.draws,n),dtype=np.float32)
            global_z=rng.standard_normal((a.draws,1),dtype=np.float32)
            relation_z=rng.standard_normal((a.draws,ng),dtype=np.float32)
            for global_rho,relation_rho in configs:
                latent=np.sqrt(1-global_rho-relation_rho)*eps+np.sqrt(global_rho)*global_z+np.sqrt(relation_rho)*relation_z[:,group]
                samples=(latent<threshold).astype(np.float32)
                totals=samples.sum(axis=1)[:,None]
                total_crps=float(ensemble_crps(totals,[y.sum()])[0]/n)
                coverage,width,interval=interval_statistics(totals,[y.sum()])
                if len(eligible):
                    gc=np.zeros((a.draws,ng),np.float32)
                    for j in range(a.draws):gc[j]=np.bincount(group,weights=samples[j],minlength=ng)
                    group_crps=float(np.mean(ensemble_crps(gc[:,eligible],observed[eligible])/counts[eligible]))
                    cov,w,isc=interval_statistics(gc[:,eligible],observed[eligible])
                    group_cov=float(cov.mean());group_width=float(np.mean(w/counts[eligible]));group_interval=float(np.mean(isc/counts[eligible]))
                else:group_crps=group_cov=group_width=group_interval=None
                rows.append(dict(state=int(state),split=split,cutoff_us=int(t),global_rho=global_rho,relation_rho=relation_rho,n=n,groups=len(eligible),
                    total_fraction_crps=total_crps,total_90_coverage=float(coverage[0]),total_90_width_fraction=float(width[0]/n),total_90_interval_score=float(interval[0]/n),
                    group_fraction_crps=group_crps,group_90_coverage=group_cov,group_90_width_fraction=group_width,group_90_interval_score=group_interval,
                    marginal_brier=float(np.mean((p-y)**2)),marginal_mean=float(p.mean()),observed_mean=float(y.mean())))
    frame=pd.DataFrame(rows);frame.to_parquet(out/'scores.parquet',index=False)
    metrics=frame.groupby(['state','split','global_rho','relation_rho']).mean(numeric_only=True).reset_index()
    metrics.to_json(out/'metrics.json',orient='records',indent=2)
    selected=[]
    for state in sorted(frame.state.unique()):
        validation=metrics[(metrics.state==state)&(metrics.split=='validation')].copy()
        validation['selection_score']=validation.total_fraction_crps+validation.group_fraction_crps.fillna(0)
        best=validation.sort_values(['selection_score','global_rho','relation_rho']).iloc[0]
        test=metrics[(metrics.state==state)&(metrics.split=='pilot_test')]
        selected.append({'state':int(state),'selection_rule':'Equal-weight sum of population and mean group count-fraction CRPS on validation; groups require >=5 query entities.',
            'chosen':test[(test.global_rho==best.global_rho)&(test.relation_rho==best.relation_rho)].to_dict('records'),
            'independent':test[(test.global_rho==0)&(test.relation_rho==0)].to_dict('records')})
    manifest={'config':vars(a),'calibration':calibrations,'selected':selected,'elapsed_s':time.monotonic()-start,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'predictor_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
        'caveats':['Development pilot; no final set used.','Classical shared-factor copula baseline, not a novel model.','Calibration and factor selection reuse validation period in this exploratory pilot.','Borg aggregate refers to unweighted sampled query population, not full cell.','Overlapping case outcomes require temporal block uncertainty in final evaluation.']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'reports'/f'{out.name}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
