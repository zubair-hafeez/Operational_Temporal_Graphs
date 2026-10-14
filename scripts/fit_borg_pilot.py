"""Cheap falsification: do relation-derived state summaries add predictive signal?

This is a discriminative development pilot, not a graph-generation result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def scores(y,p,weight):
    p=np.clip(p,1e-7,1-1e-7)
    nll=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return dict(n=len(y),prevalence=float(np.average(y,weights=weight)),prediction_mean=float(np.average(p,weights=weight)),
                brier=float(np.average((y-p)**2,weights=weight)),nll=float(np.average(nll,weights=weight)),
                auroc=float(roc_auc_score(y,p,sample_weight=weight)) if len(np.unique(y))>1 else None,
                ap=float(average_precision_score(y,p,sample_weight=weight)) if len(np.unique(y))>1 else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='borg_pilot_h15_step60_n1000')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max-iter',type=int,default=150)
    parser.add_argument('--arms',nargs='+',default=['personal','global','relational'])
    a=parser.parse_args()
    data=ROOT/'data/processed'/a.dataset
    meta=json.loads((data/'metadata.json').read_text())
    x=np.load(data/'features.npy');q=pd.read_parquet(data/'queries.parquet')
    if set(q.split)-{'train','validation','pilot_test','embargo'}:raise ValueError('Not a pilot dataset')
    if not np.isfinite(x).all():raise ValueError('Nonfinite inputs')
    run=ROOT/'runs'/f'pilot_hgb_seed{a.seed}_{a.dataset}'
    run.mkdir(parents=True,exist_ok=False)
    feature_sets={
        'personal':meta['feature_groups']['personal'],
        'global':meta['feature_groups']['personal']+meta['feature_groups']['global'],
        'relational':meta['features'],
    }
    manifest=dict(config=vars(a),dataset_metadata_sha256=hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  python=platform.python_version(),sklearn=sklearn.__version__,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  purpose='DEVELOPMENT ONLY. No proposed neural generator or final-test evaluation.',models=[])
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[];predictions=[]
    with threadpool_limits(limits=4):
        for state in (1,2):
            train=(q.split=='train')&(q.state==state)&q.observed
            y=q['exit'].to_numpy()
            weight=q.sampling_weight.to_numpy()
            fit_weight=weight[train]/weight[train].mean()
            for arm in a.arms:
                cols=[meta['features'].index(n) for n in feature_sets[arm]]
                model=HistGradientBoostingClassifier(max_iter=a.max_iter,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=1.0,learning_rate=.08,early_stopping=False,random_state=a.seed)
                start=time.monotonic();model.fit(x[train][:,cols],y[train],sample_weight=fit_weight)
                fit_seconds=time.monotonic()-start
                joblib.dump(model,run/f'{arm}_state{state}.joblib')
                for split in ['validation','pilot_test']:
                    mask=(q.split==split)&(q.state==state)&q.observed
                    p=model.predict_proba(x[mask][:,cols])[:,1]
                    result=dict(arm=arm,state=state,split=split,fit_seconds=fit_seconds,**scores(y[mask],p,weight[mask]))
                    rows.append(result);print(json.dumps(result),flush=True)
                    pred=q.loc[mask,['query_id','cutoff_us','elapsed_day','sampling_weight','state','split','exit']].copy()
                    pred['arm']=arm;pred['probability']=p;predictions.append(pred)
                manifest['models'].append(dict(arm=arm,state=state,features=feature_sets[arm],fit_seconds=fit_seconds))
                (run/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    pd.concat(predictions,ignore_index=True).to_parquet(run/'predictions.parquet',index=False)
    manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    manifest['status']='complete'
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(run,flush=True)


if __name__=='__main__':main()
