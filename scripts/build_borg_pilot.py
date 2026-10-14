"""Construct as-of task queries while retaining the full cell's observed state."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.borg import BorgReplay, ORIGIN_US, DAY_US, PERSONAL, GLOBAL, RELATIONAL, assign_interval_labels

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prefix',default='borg_prefix_7d')
    p.add_argument('--step-minutes',type=int,default=60)
    p.add_argument('--horizon-minutes',type=int,default=15)
    p.add_argument('--per-state',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261014)
    a=p.parse_args()
    folder=ROOT/'data/processed'/a.prefix
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['through_days']>7:raise ValueError('Pilot only; implement frozen full protocol separately')
    events=pd.read_parquet(folder/'events.parquet')
    intervals=pd.read_parquet(folder/'intervals.parquet')
    state=BorgReplay(meta['tasks'],meta['jobs'],meta['machines'],meta['unknown_machine'])
    stop=meta['stop_us'];horizon=a.horizon_minutes*60*1_000_000
    rng=np.random.default_rng(a.seed);lo=0;rows=[];features=[];qid=0;start=time.monotonic();state_audit=[]
    cutoffs=np.arange(ORIGIN_US+a.step_minutes*60*1_000_000,stop-horizon+1,a.step_minutes*60*1_000_000,dtype=np.int64)
    times=events.time_us.to_numpy()
    for i,t in enumerate(cutoffs):
        hi=int(np.searchsorted(times,t,side='right'))
        state.apply(events.iloc[lo:hi]);lo=hi
        counts={int(s):int((state.state==s).sum()) for s in (1,2)}
        ids=[];weights=[]
        for s in (1,2):
            pool=np.flatnonzero(state.state==s)
            chosen=rng.choice(pool,size=min(a.per_state,len(pool)),replace=False)
            ids.extend(chosen);weights.extend([len(pool)/max(1,len(chosen))]*len(chosen))
        ids=np.asarray(ids,np.int32)
        features.append(state.features(ids,int(t)))
        rows.append(pd.DataFrame({'query_id':np.arange(qid,qid+len(ids),dtype=np.int64),'task':ids,'cutoff_us':t,'elapsed_day':(t-ORIGIN_US)/DAY_US,'state':state.state[ids],'sampling_weight':np.asarray(weights,np.float32)}))
        qid+=len(ids)
        state_audit.append(dict(cutoff_us=int(t),pending=counts[1],running=counts[2],events_seen=hi))
        if i%24==0:print(json.dumps({'cutoff':i,'total_cutoffs':len(cutoffs),'queries':qid,'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
    q=pd.concat(rows,ignore_index=True)
    y=assign_interval_labels(intervals,q,horizon,stop)
    if not np.array_equal(q.query_id.to_numpy(),y.query_id.to_numpy()):raise RuntimeError('label/query ordering mismatch')
    # Entire outcome interval must remain inside its designated temporal split.
    days=q.elapsed_day.to_numpy();end_days=days+a.horizon_minutes/(24*60)
    split=np.full(len(q),'embargo',dtype=object)
    split[(days<4)&(end_days<=4)]='train'
    split[(days>=4)&(end_days<=5)]='validation'
    split[(days>=5)&(end_days<=7)]='pilot_test'
    q['split']=split
    for key in ['exit','exit_type','observed','remaining_s']:q[key]=y[key].to_numpy()
    x=np.concatenate(features)
    out=ROOT/'data/processed'/f'borg_pilot_h{a.horizon_minutes}_step{a.step_minutes}_n{a.per_state}'
    out.mkdir(exist_ok=True,parents=True)
    np.save(out/'features.npy',x)
    q.to_parquet(out/'queries.parquet',index=False,compression='zstd')
    report=dict(config=vars(a),n_queries=len(q),n_features=x.shape[1],features=PERSONAL+GLOBAL+RELATIONAL,feature_groups={'personal':PERSONAL,'global':GLOBAL,'relational':RELATIONAL},splits=q.split.value_counts().to_dict(),
                label='First change of current task state during (cutoff, cutoff+horizon]; no future input.',
                sampling='Uniform within pending/running at each cutoff; full cell retained for context; inverse inclusion weights stored.',
                prefix_metadata=meta,queries_sha256=hashlib.sha256((out/'queries.parquet').read_bytes()).hexdigest(),features_sha256=hashlib.sha256((out/'features.npy').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),replay_sha256=hashlib.sha256((ROOT/'otg/borg.py').read_bytes()).hexdigest(),elapsed_s=time.monotonic()-start)
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'reports/borg_pilot_build.json').write_text(json.dumps({**report,'state_audit':state_audit},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['features','feature_groups','prefix_metadata']}),flush=True)


if __name__=='__main__':main()
