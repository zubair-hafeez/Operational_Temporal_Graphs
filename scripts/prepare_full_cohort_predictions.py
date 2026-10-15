"""Predict full development cohorts using frozen, selected marginal models."""
from __future__ import annotations
import argparse,hashlib,json,shutil,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--history',action='store_true');a=ap.parse_args()
    tag='history_v1' if a.history else 'v1';data=ROOT/'data/processed'/f'{a.dataset}_joint_{tag}';m=json.loads((data/'metadata.json').read_text())
    out=data/'cohorts';out.mkdir(exist_ok=False);source=ROOT/m['source_dataset'];sm=json.loads((source/'metadata.json').read_text())
    scaler=np.load(data/'scaler.npz');rows=[];start=time.monotonic()
    if a.dataset=='bpi':
        q=pd.read_parquet(data/'queries.parquet');p=np.load(data/'cdf.npy');x=np.load(data/'features.npy')
        for t,frame in q[q.part=='validation'].groupby('cutoff_us'):
            name=f'cohort_{int(t)}';folder=out/name;folder.mkdir();ids=frame.index.to_numpy()
            frame.to_parquet(folder/'queries.parquet',index=False);np.save(folder/'cdf.npy',p[ids]);np.save(folder/'features.npy',x[ids])
            rows.append({'folder':name,'cutoff_us':int(t),'n':len(frame)})
    else:
      with threadpool_limits(limits=4):
        models={}
        for choice in m['selected_marginals']:
            state=choice['state'];run=ROOT/'runs'/f'borg_marginals_{tag}_state{state}_{choice["arm"]}_seed0/development20_22'
            models[state]=(choice,[joblib.load(run/f'horizon{h}m.joblib') for h in m['horizons']],json.loads((run/'calibration_audit.json').read_text()))
        for cohort in sm['cohorts']:
            if cohort['elapsed_day']<20:continue
            old=source/cohort['folder'];q=pd.read_parquet(old/'queries.parquet');x=np.load(old/'features.npy');p=np.zeros((len(q),len(m['horizons'])),np.float32)
            for state,(choice,fitted,audit) in models.items():
                ids=np.flatnonzero(q.state.to_numpy()==state)
                if not len(ids):continue
                raw=[];individual=[]
                for model in fitted:
                    pp=np.clip(model['model'].predict_proba(x[ids][:,model['cols']])[:,1],1e-5,1-1e-5)
                    raw.append(pp);individual.append(model['calibrator'].predict_proba(logit(pp)[:,None])[:,1])
                raw=np.sort(np.column_stack(raw),axis=1);variant=choice['calibration_variant']
                if variant=='raw_sorted':cdf=raw
                elif variant=='individual_platt_sorted':cdf=np.sort(np.column_stack(individual),axis=1)
                else:
                    aa,bb=audit['params'][variant];cdf=expit(aa*logit(np.clip(raw,1e-6,1-1e-6))+bb)
                p[ids]=cdf
            folder=out/cohort['folder'];folder.mkdir();shutil.copy2(old/'queries.parquet',folder/'queries.parquet');np.save(folder/'cdf.npy',p)
            np.save(folder/'features.npy',np.clip((x-scaler['mean'])/scaler['scale'],-10,10).astype(np.float32))
            rows.append({'folder':cohort['folder'],'cutoff_us':cohort['cutoff_us'],'n':len(q),'elapsed_day':cohort['elapsed_day']})
            print(json.dumps({'cohort':cohort['folder'],'n':len(q),'elapsed_s':time.monotonic()-start}),flush=True)
    report={'config':vars(a),'cohorts':rows,'joint_metadata_sha256':hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    for row in rows:
        folder=out/row['folder'];row['cdf_sha256']=hashlib.sha256((folder/'cdf.npy').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
