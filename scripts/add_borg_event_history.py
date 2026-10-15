"""Add strictly past event histograms for tasks, global stream, jobs, machines.

This strengthens marginal-history controls before interpreting residual coupling.
No outcome, future assignment, or future event count enters these features.
"""
from __future__ import annotations
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EVENT_NAMES=['submit','schedule','evict','fail','finish','kill','lost','update_pending','update_running']
WINDOWS=[1,5,15,60]


def sparse_counts(frame,column,ids):
    codes=frame[column].to_numpy(dtype=np.int64)*9+frame.event_type.to_numpy()
    keys,counts=np.unique(codes,return_counts=True)
    target=np.asarray(ids,dtype=np.int64)[:,None]*9+np.arange(9)
    if not len(keys):return np.zeros(target.shape,np.float32)
    where=np.searchsorted(keys,target);safe=np.minimum(where,len(keys)-1)
    return np.log1p(np.where((where<len(keys))&(keys[safe]==target),counts[safe],0)).astype(np.float32)


def main():
    src=ROOT/'data/processed/borg_development_horizons_v1';m=json.loads((src/'metadata.json').read_text())
    out=ROOT/'data/processed/borg_development_history_v1';out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();print('Reading development prefix',flush=True)
    e=pd.read_parquet(ROOT/'data/processed/borg_prefix_22d/events.parquet',columns=['time_us','task','job','machine','event_type']);times=e.time_us.to_numpy()
    q=pd.read_parquet(src/'queries.parquet');x=np.load(src/'features.npy',mmap_mode='r')
    extra_names=[];groups={k:list(v) for k,v in m['feature_groups'].items()}
    for window in WINDOWS:
        for scope,group in [('task','personal'),('global','global'),('job','relational'),('machine','relational')]:
            names=[f'log_{scope}_{window}m_{event}_count' for event in EVENT_NAMES];extra_names.extend(names);groups[group].extend(names)
    names=m['features']+extra_names
    full=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(q),len(names)));full[:,:x.shape[1]]=x
    cohorts_by_time={c['cutoff_us']:c for c in m['cohorts']};cohorts=[]
    for k,(t,frame) in enumerate(q.groupby('cutoff_us',sort=True)):
        hi=np.searchsorted(times,t,side='right');lookups=[]
        for window in WINDOWS:
            lo=np.searchsorted(times,t-window*60_000_000,side='right');lookups.append(e.iloc[lo:hi])
        def features(query):
            pieces=[]
            for recent in lookups:
                pieces.append(sparse_counts(recent,'task',query.task))
                glob=np.log1p(np.bincount(recent.event_type.to_numpy(),minlength=9)).astype(np.float32)
                pieces.append(np.tile(glob,(len(query),1)))
                pieces.append(sparse_counts(recent,'job',query.job))
                pieces.append(sparse_counts(recent,'machine',query.machine))
            return np.concatenate(pieces,axis=1)
        full[frame.index,x.shape[1]:]=features(frame)
        if int(t) in cohorts_by_time:
            c=cohorts_by_time[int(t)];folder=out/c['folder'];folder.mkdir()
            old=src/c['folder'];cq=pd.read_parquet(old/'queries.parquet');cx=np.load(old/'features.npy')
            np.save(folder/'features.npy',np.concatenate([cx,features(cq)],axis=1));shutil.copy2(old/'queries.parquet',folder/'queries.parquet')
            cohorts.append({**c,'features_sha256':hashlib.sha256((folder/'features.npy').read_bytes()).hexdigest()})
        if k%48==0:print(json.dumps({'cutoff':k,'total':q.cutoff_us.nunique(),'elapsed_s':time.monotonic()-start}),flush=True)
    full.flush();del full
    shutil.copy2(src/'queries.parquet',out/'queries.parquet')
    report={**m,'features':names,'feature_groups':groups,'cohorts':cohorts,'history_windows_minutes':WINDOWS,'history_contract':'Events in (cutoff-window, cutoff]; ids are cutoff-observed. All pending/unknown machine contexts remain zero.',
        'features_sha256':hashlib.sha256((out/'features.npy').read_bytes()).hexdigest(),'source_metadata_sha256':hashlib.sha256((src/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports/borg_development_history.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'n_queries':len(q),'n_features':len(names),'cohorts':len(cohorts),'elapsed_s':report['elapsed_s']}),flush=True)


if __name__=='__main__':main()
