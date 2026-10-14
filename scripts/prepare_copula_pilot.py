"""Disjoint chronological marginal fit, calibration, dependence fit, tuning, test.

This deliberately smaller pilot trades marginal fit data for clean independence
between outcome-based marginal fitting and dependence fitting. No final data.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from otg.borg import ORIGIN_US,DAY_US
ROOT=Path(__file__).resolve().parents[1]


def make_pairs(q,part,per_cutoff,rng):
    pairs=[]
    for (_,t),f in q[q.part==part].groupby(['state','cutoff_us']):
        ids=f.index.to_numpy();groups=list(f.groupby('relation').groups.values())
        groups=[np.asarray(g) for g in groups if len(g)>1]
        # Half uniformly selected nonsingleton groups, half random entity pairs.
        if groups:
            selected=rng.integers(len(groups),size=per_cutoff//2)
            for g in selected:
                i,j=rng.choice(groups[g],2,replace=False);pairs.append((i,j))
        for _ in range(per_cutoff//2):
            i,j=rng.choice(ids,2,replace=False);pairs.append((i,j))
    return np.asarray(pairs,dtype=np.int32)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);a=ap.parse_args()
    ds='borg_pilot_h15_step60_n1000' if a.dataset=='borg' else 'bpi_pilot_h7d'
    src=ROOT/'data/processed'/ds;m=json.loads((src/'metadata.json').read_text())
    x=np.load(src/'features.npy');q=pd.read_parquet(src/'queries.parquet')
    if a.dataset=='borg':
        t=(q.cutoff_us-ORIGIN_US)/DAY_US;h=15/(24*60);b0,b1,b2,b3,b4=2.,2.5,4.,5.,7.
        mapping=pd.read_parquet(ROOT/'data/processed/borg_prefix_7d/task_index.parquet',columns=['task','job_id'])
        q=q.merge(mapping,on='task',validate='many_to_one');q['relation']=q.job_id
    else:
        t=q.cutoff_us/DAY_US;h=7.
        b0,b1,b2,b3,b4=[pd.Timestamp(s,tz='UTC').value/1000/DAY_US for s in ['2016-05-01','2016-06-01','2016-07-01','2016-08-01','2016-09-01']]
        meta=json.loads((ROOT/'data/processed/bpi_before_20161101/metadata.json').read_text())
        q['relation']=q.resource.astype(np.int64)
        q.loc[q.resource==meta['resources'],'relation']=-1-q.loc[q.resource==meta['resources'],'case']
    q['part']='embargo'
    for name,lo,hi in [('marginal',-np.inf,b0),('calibration',b0,b1),('dependence',b1,b2),('validation',b2,b3),('pilot_test',b3,b4)]:
        q.loc[(t>=lo)&(t+h<=hi),'part']=name
    out=ROOT/'data/processed'/f'{a.dataset}_copula_pilot';out.mkdir(parents=True,exist_ok=False)
    q['p']=np.nan;models=[]
    with threadpool_limits(limits=4):
        for state in map(int,sorted(q.state.unique())):
            fit=(q.state==state)&(q.part=='marginal');cal=(q.state==state)&(q.part=='calibration');val=(q.state==state)&(q.part=='validation')
            choices=[]
            for arm in ['personal','global','relational']:
                features=m['feature_groups']['personal']+(m['feature_groups']['global'] if arm!='personal' else [])+(m['feature_groups']['relational'] if arm=='relational' else [])
                cols=[m['features'].index(c) for c in features]
                model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=1.,learning_rate=.08,early_stopping=False,random_state=0)
                w=q.loc[fit,'sampling_weight'].to_numpy();model.fit(x[fit][:,cols],q.loc[fit,'exit'],sample_weight=w/w.mean())
                p=np.clip(model.predict_proba(x[:,cols])[:,1],1e-5,1-1e-5)
                logit=np.log(p/(1-p))[:,None];w=q.loc[cal,'sampling_weight'].to_numpy()
                calibrator=LogisticRegression(C=1,max_iter=1000).fit(logit[cal],q.loc[cal,'exit'],sample_weight=w/w.mean())
                pp=np.clip(calibrator.predict_proba(logit)[:,1],1e-5,1-1e-5)
                brier=float(np.average((q.loc[val,'exit']-pp[val])**2,weights=q.loc[val,'sampling_weight']))
                joblib.dump({'model':model,'calibrator':calibrator,'cols':cols},out/f'{arm}_state{state}.joblib')
                choices.append((brier,arm,pp));models.append({'state':state,'arm':arm,'validation_brier':brier})
            best=min(choices,key=lambda z:z[0]);q.loc[q.state==state,'p']=best[2][q.state==state]
            models.append({'state':state,'chosen_arm':best[1]})
    # Scaling is based only on dependence-fit covariates.
    fit=q.part=='dependence';mean=x[fit].mean(0);scale=x[fit].std(0);scale[scale<1e-5]=1
    np.save(out/'features.npy',np.clip((x-mean)/scale,-10,10).astype(np.float32))
    np.savez(out/'scaler.npz',mean=mean,scale=scale)
    q.to_parquet(out/'queries.parquet',index=False,compression='zstd')
    rng=np.random.default_rng(20261014)
    for name in ['dependence','validation','pilot_test']:np.save(out/f'pairs_{name}.npy',make_pairs(q,name,2000,rng))
    report={'dataset':a.dataset,'source':ds,'parts':q.part.value_counts().to_dict(),'models':models,'features':m['features'],'feature_groups':m['feature_groups'],
        'pair_sampling':'Each state/cutoff: 1000 uniform-group pairs plus 1000 uniform-node pairs; no outcome-based sampling. Coupling fitted only on dependence part.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_metadata_sha256':hashlib.sha256((src/'metadata.json').read_bytes()).hexdigest()}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
