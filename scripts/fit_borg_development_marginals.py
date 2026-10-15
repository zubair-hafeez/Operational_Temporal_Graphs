"""Chronological cross-fitted multi-horizon marginals; final week remains locked."""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
HORIZONS=[1,5,15,60]
FOLDS={'oof8_10':(6,8,10),'oof12_14':(10,12,14),'oof16_18':(14,16,18),'development20_22':(18,20,22)}


def metric(y,p,w):
    pp=np.clip(p,1e-7,1-1e-7)
    return {'brier':float(np.average((y-p)**2,weights=w)),'nll':float(np.average(-y*np.log(pp)-(1-y)*np.log1p(-pp),weights=w)),
        'prevalence':float(np.average(y,weights=w)),'prediction_mean':float(np.average(p,weights=w))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state',type=int,choices=[1,2],required=True)
    ap.add_argument('--dataset',choices=['borg','bpi'],default='borg')
    ap.add_argument('--history',action='store_true',help='Use the Borg past-event history extension')
    ap.add_argument('--arm',choices=['personal','global','relational'],required=True)
    ap.add_argument('--folds',nargs='+');ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--max-iter',type=int,default=250);a=ap.parse_args()
    if a.history and a.dataset!='borg':raise ValueError('History extension is Borg-specific')
    version='history_v1' if a.history else 'horizons_v1'
    src=ROOT/'data/processed'/f'{a.dataset}_development_{version}';m=json.loads((src/'metadata.json').read_text())
    if a.dataset=='borg':folds=FOLDS;horizons=HORIZONS;unit='m';max_horizon_day=1/24
    else:
        origin=pd.Timestamp('2016-01-01',tz='UTC')
        def day(s):return (pd.Timestamp(s,tz='UTC')-origin).total_seconds()/86400
        folds={'oof_june':tuple(map(day,['2016-05-01','2016-06-01','2016-07-01'])),
               'oof_july':tuple(map(day,['2016-06-01','2016-07-01','2016-08-01'])),
               'oof_august':tuple(map(day,['2016-07-01','2016-08-01','2016-09-01'])),
               'development_october':tuple(map(day,['2016-09-01','2016-10-01','2016-11-01']))}
        horizons=[1,3,7,14];unit='d';max_horizon_day=14
    requested=a.folds or list(folds)
    if set(requested)-set(folds):raise ValueError('Unknown fold for dataset')
    q=pd.read_parquet(src/'queries.parquet');x=np.load(src/'features.npy',mmap_mode='r')
    names=m['feature_groups']['personal']+(m['feature_groups']['global'] if a.arm!='personal' else [])+(m['feature_groups']['relational'] if a.arm=='relational' else [])
    cols=[m['features'].index(c) for c in names];state=q.state.to_numpy()==a.state
    t=q.elapsed_day.to_numpy();w=q.sampling_weight.to_numpy();y=q[[f'exit_{h}{unit}' for h in horizons]].to_numpy()
    tag='history_v1' if a.history else 'v1'
    run=ROOT/'runs'/f'{a.dataset}_marginals_{tag}_state{a.state}_{a.arm}_seed{a.seed}';run.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with threadpool_limits(limits=4):
        for fold,(fit_end,cal_end,target_end) in folds.items():
            if fold not in requested:continue
            out=run/fold;out.mkdir(exist_ok=False)
            fit=state&(t+max_horizon_day<=fit_end);cal=state&(t>=fit_end)&(t+max_horizon_day<=cal_end);target=state&(t>=cal_end)&(t+max_horizon_day<=target_end)
            fit_ids=np.flatnonzero(fit);cal_ids=np.flatnonzero(cal);target_ids=np.flatnonzero(target)
            xx=np.asarray(x[fit_ids][:,cols]);xc=np.asarray(x[cal_ids][:,cols]);xt=np.asarray(x[target_ids][:,cols])
            pred=[];raw=[];calibrators=[]
            manifest={'config':vars(a),'fold':fold,'fit_end_day':fit_end,'calibration_end_day':cal_end,'target_end_day':target_end,'n_fit':len(fit_ids),'n_calibration':len(cal_ids),'n_target':len(target_ids),
                'features':names,'horizons':horizons,'horizon_unit':unit,'status':'running','python':platform.python_version(),'sklearn':sklearn.__version__,
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dataset_metadata_sha256':hashlib.sha256((src/'metadata.json').read_bytes()).hexdigest()}
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            for k,h in enumerate(horizons):
                model=HistGradientBoostingClassifier(max_iter=a.max_iter,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=1.,learning_rate=.05,early_stopping=False,random_state=a.seed)
                model.fit(xx,y[fit,k],sample_weight=w[fit]/w[fit].mean())
                pc=np.clip(model.predict_proba(xc)[:,1],1e-5,1-1e-5);pt=np.clip(model.predict_proba(xt)[:,1],1e-5,1-1e-5)
                calibration=LogisticRegression(C=1.,max_iter=1000).fit(np.log(pc/(1-pc))[:,None],y[cal,k],sample_weight=w[cal]/w[cal].mean())
                calibrated=np.clip(calibration.predict_proba(np.log(pt/(1-pt))[:,None])[:,1],1e-5,1-1e-5)
                joblib.dump({'model':model,'calibrator':calibration,'cols':cols},out/f'horizon{h}{unit}.joblib')
                pred.append(calibrated);raw.append(pt);calibrators.append({'horizon':h,'intercept':calibration.intercept_.tolist(),'coefficient':calibration.coef_.tolist()})
                print(json.dumps({'state':a.state,'arm':a.arm,'fold':fold,'horizon':h,'elapsed_s':time.monotonic()-start}),flush=True)
            pre=np.stack(pred,axis=1);cdf=np.sort(pre,axis=1)
            np.savez_compressed(out/'predictions.npz',query_ids=target_ids,raw=np.stack(raw,axis=1).astype(np.float32),calibrated=pre.astype(np.float32),cdf=cdf.astype(np.float32))
            metrics=[{'horizon':h,'horizon_unit':unit,'raw':metric(y[target,k],raw[k],w[target]),'calibrated':metric(y[target,k],pre[:,k],w[target]),'monotone_cdf':metric(y[target,k],cdf[:,k],w[target])} for k,h in enumerate(horizons)]
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
            manifest.update(status='complete',calibrators=calibrators,cdf_projection='Ascending rearrangement of four calibrated horizon probabilities; coherent first-exit CDF.',crossing_fraction=float(np.mean(np.any(np.diff(pre,axis=1)<0,axis=1))),elapsed_s=time.monotonic()-start,
                predictions_sha256=hashlib.sha256((out/'predictions.npz').read_bytes()).hexdigest())
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
