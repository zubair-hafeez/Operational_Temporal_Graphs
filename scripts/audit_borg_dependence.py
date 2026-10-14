"""Pilot residual dependence, with matched permutation and calibration controls.

Evidence of residual structure is not proof of irreducible dependence given all
available history. This diagnostic neither trains nor evaluates a generator.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def pair_stat(residual,group):
    sums=np.bincount(group,weights=residual)
    squares=np.bincount(group,weights=residual**2)
    n=np.bincount(group)
    pairs=n*(n-1)
    return float((sums*sums-squares).sum()/max(pairs.sum(),1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--permutations',type=int,default=199);a=p.parse_args()
    run=ROOT/'runs/pilot_hgb_seed0_borg_pilot_h15_step60_n1000'
    pred=pd.read_parquet(run/'predictions.parquet')
    taskmap=pd.read_parquet(ROOT/'data/processed/borg_prefix_7d/task_index.parquet',columns=['task','job_id'])
    queries=pd.read_parquet(ROOT/'data/processed/borg_pilot_h15_step60_n1000/queries.parquet',columns=['query_id','task'])
    pred=pred.merge(queries,on='query_id',validate='many_to_one').merge(taskmap,on='task',validate='many_to_one')
    rng=np.random.default_rng(42);results=[]
    with threadpool_limits(limits=4):
        for state in (1,2):
            for arm in ('personal','global','relational'):
                part=pred[(pred.state==state)&(pred.arm==arm)].copy()
                probability=np.clip(part.probability.to_numpy(),1e-6,1-1e-6)
                logit=np.log(probability/(1-probability))[:,None]
                val=part.split=='validation'
                calibration=LogisticRegression(C=1,max_iter=1000)
                weights=part.sampling_weight.to_numpy()[val]
                calibration.fit(logit[val],part['exit'].to_numpy()[val],sample_weight=weights/weights.mean())
                part['calibrated_p']=calibration.predict_proba(logit)[:,1]
                test=part[part.split=='pilot_test'].copy().reset_index(drop=True)
                for cal,column in [('raw','probability'),('development_platt','calibrated_p')]:
                    p_hat=test[column].to_numpy();y=test['exit'].to_numpy()
                    # Within-cutoff centering removes a shared mean offset, but
                    # cannot remove all omitted-history/model misspecification.
                    r=y-p_hat
                    r-=pd.Series(r).groupby(test.cutoff_us).transform('mean').to_numpy()
                    group,_=pd.factorize(pd.MultiIndex.from_frame(test[['cutoff_us','job_id']]),sort=False)
                    observed=pair_stat(r,group)
                    bins=np.minimum((p_hat*10).astype(int),9)
                    strata,_=pd.factorize(pd.MultiIndex.from_arrays([test.cutoff_us,bins]),sort=False)
                    positions=[np.flatnonzero(strata==i) for i in np.unique(strata)]
                    null=[]
                    for _ in range(a.permutations):
                        shuffled=r.copy()
                        for ids in positions:
                            shuffled[ids]=rng.permutation(r[ids])
                        null.append(pair_stat(shuffled,group))
                    grouped=test.assign(p=p_hat).groupby(['cutoff_us','job_id']).agg(n=('exit','size'),actual=('exit','sum'),expected=('p','sum'))
                    variances=pd.Series(p_hat*(1-p_hat)).groupby([test.cutoff_us,test.job_id]).sum()
                    grouped['independent_variance']=variances
                    grouped=grouped[grouped.n>=5]
                    var_ratio=float(((grouped.actual-grouped.expected)**2).sum()/grouped.independent_variance.sum()) if len(grouped) else None
                    result=dict(state=state,arm=arm,calibration=cal,n_test=len(test),same_job_residual_pair_product=observed,matched_permutation_mean=float(np.mean(null)),matched_permutation_95=np.quantile(null,[.025,.975]).tolist(),permutation_p_one_sided=float((1+sum(x>=observed for x in null))/(a.permutations+1)),groups_size_at_least_5=len(grouped),squared_count_error_over_independent_variance=var_ratio,
                                caveat='Pilot only; two days, overlapping entity histories; misspecified marginals and omitted histories can produce residual dependence. Permutation is a descriptive conditional-association diagnostic, not final inferential evidence.')
                    results.append(result);print(json.dumps(result),flush=True)
    (ROOT/'reports/borg_pilot_dependence.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
