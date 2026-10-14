"""Prepare an explicitly bounded prefix. Does not score/open the final period."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
COLUMNS=['time_us','missing','job_id','task_index','machine_id','event_type','user','sched_class','priority','cpu','memory','disk','different_machine']
KEEP=['time_us','missing','job_id','task_index','machine_id','event_type','sched_class','priority','cpu','memory']
DTYPES={'time_us':'int64','missing':'float32','job_id':'int64','task_index':'int32','machine_id':'Int64','event_type':'int8','sched_class':'float32','priority':'int8','cpu':'float32','memory':'float32'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--through-days',type=float,default=7)
    args=p.parse_args()
    if not 0<args.through_days<=22:
        raise ValueError('Development preparation restricted to <=22 days; final access requires frozen evaluation machinery.')
    stop=int((600+args.through_days*86400)*1e6)
    name=f'borg_prefix_{args.through_days:g}d'
    out=ROOT/'data/processed'/name;out.mkdir(parents=True,exist_ok=True)
    frames=[];sources=[];rid=0;start=time.monotonic()
    for path in sorted((ROOT/'data/raw/borg/task_events').glob('*.csv.gz')):
        df=pd.read_csv(path,names=COLUMNS,usecols=KEEP,dtype=DTYPES)
        df['machine_id']=df['machine_id'].fillna(-1).astype('int64')
        df['missing']=df['missing'].fillna(-1).astype('int8')
        df['source_row']=np.arange(rid,rid+len(df),dtype=np.int64);rid+=len(df)
        max_time=int(df.time_us.max())
        df=df[df.time_us<stop].copy()
        if len(df):
            frames.append(df);sources.append(path.name)
        if len(sources)%25==0:
            print(json.dumps({'parts':len(sources),'rows':sum(len(x) for x in frames),'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
        if max_time>=stop:break
    if not frames or max_time<stop:
        raise RuntimeError('Insufficient downloaded data for requested prefix')
    df=pd.concat(frames,ignore_index=True);del frames
    was_sorted=bool(df.time_us.is_monotonic_increasing)
    if not was_sorted:df=df.sort_values(['time_us','source_row'],kind='stable').reset_index(drop=True)
    task_codes,task_keys=pd.factorize(pd.MultiIndex.from_frame(df[['job_id','task_index']]),sort=False)
    job_codes,job_keys=pd.factorize(df.job_id,sort=False)
    machine_codes,machine_keys=pd.factorize(df.machine_id,sort=False)
    df['task']=task_codes.astype(np.int32);df['job']=job_codes.astype(np.int32);df['machine']=machine_codes.astype(np.int32)
    unknown_machine=int(np.flatnonzero(machine_keys==-1)[0]) if np.any(machine_keys==-1) else -1
    # Storage-only mapping; no future node counts/identities are predictive features.
    taskmap=task_keys.to_frame(index=False);taskmap.columns=['job_id','task_index'];taskmap['task']=np.arange(len(taskmap),dtype=np.int32)
    taskmap.to_parquet(out/'task_index.parquet',index=False)
    pd.DataFrame({'job':np.arange(len(job_keys),dtype=np.int32),'job_id':job_keys}).to_parquet(out/'job_index.parquet',index=False)
    pd.DataFrame({'machine':np.arange(len(machine_keys),dtype=np.int32),'machine_id':machine_keys}).to_parquet(out/'machine_index.parquet',index=False)
    df.to_parquet(out/'events.parquet',index=False,compression='zstd')
    # Independent interval table for labels, isolated from cutoff feature construction.
    states=np.array([1,2,0,0,0,0,0,1,2],dtype=np.int8)
    df['state']=states[df.event_type.to_numpy()]
    prev=df.groupby('task',sort=False).state.shift(1)
    changed=prev.isna() | (prev != df.state)
    intervals=df.loc[changed,['task','job','time_us','state','event_type','source_row','missing']].copy()
    intervals['end_us']=intervals.groupby('task',sort=False).time_us.shift(-1).fillna(stop).astype(np.int64)
    intervals['end_type']=intervals.groupby('task',sort=False).event_type.shift(-1).fillna(-1).astype(np.int8)
    intervals['censored']=intervals.end_type<0
    intervals.to_parquet(out/'intervals.parquet',index=False,compression='zstd')
    bad=intervals[(intervals.end_us<intervals.time_us)]
    if len(bad):raise RuntimeError('Negative intervals')
    valid_prev=prev.notna() & (df.time_us>0) & (df['missing']==-1)
    matrix=pd.crosstab(prev[valid_prev].astype(int),df.loc[valid_prev,'event_type'])
    metadata=dict(through_days=args.through_days,stop_us=stop,rows=len(df),tasks=len(task_keys),jobs=len(job_keys),machines=len(machine_keys),unknown_machine=unknown_machine,
                  initial_rows=int((df.time_us==0).sum()),imputed_positive_rows=int(((df.time_us>0)&(df['missing']>=0)).sum()),was_time_sorted=was_sorted,
                  sources=sources,interval_count=len(intervals),zero_length_intervals=int((intervals.end_us==intervals.time_us).sum()),
                  transition_counts={str(k):{str(j):int(v) for j,v in row.items()} for k,row in matrix.iterrows()},
                  events_sha256=hashlib.sha256((out/'events.parquet').read_bytes()).hexdigest(),
                  source_manifest_sha256=hashlib.sha256((ROOT/'reports/source_manifest.json').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.monotonic()-start)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (ROOT/'reports'/f'{name}_audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['sources','transition_counts']}),flush=True)


if __name__=='__main__':main()
