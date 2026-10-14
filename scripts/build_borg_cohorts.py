"""Full-development multi-horizon queries plus full-population scenario cohorts.

The final week remains locked. All jobs/machines are observed at the cutoff.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.borg import BorgReplay,ORIGIN_US,DAY_US,PERSONAL,GLOBAL,RELATIONAL,assign_interval_labels
ROOT=Path(__file__).resolve().parents[1]


def label(q,intervals,stop):
    target=assign_interval_labels(intervals,q,3600_000_000,stop)
    assert np.array_equal(target.query_id,q.query_id)
    for col in ['exit_type','remaining_s','observed']:q[col]=target[col].to_numpy()
    for minute in [1,5,15,60]:q[f'exit_{minute}m']=((q.remaining_s<=60*minute)&(q.exit_type>=0)).astype(np.int8)
    return q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--per-state',type=int,default=2000);a=ap.parse_args()
    src=ROOT/'data/processed/borg_prefix_22d';m=json.loads((src/'metadata.json').read_text())
    out=ROOT/'data/processed/borg_development_horizons_v1';out.mkdir(parents=True,exist_ok=False)
    print('Reading bounded development events',flush=True);start=time.monotonic()
    e=pd.read_parquet(src/'events.parquet');intervals=pd.read_parquet(src/'intervals.parquet')
    state=BorgReplay(m['tasks'],m['jobs'],m['machines'],m['unknown_machine'])
    times=e.time_us.to_numpy();lo=0;qid=0;rows=[];xx=[];cohorts=[];rng=np.random.default_rng(20261014)
    cutoffs=np.arange(ORIGIN_US+3600_000_000,m['stop_us']-3600_000_000+1,3600_000_000,dtype=np.int64)
    for k,t in enumerate(cutoffs):
        hi=int(np.searchsorted(times,t,side='right'));state.apply(e.iloc[lo:hi]);lo=hi
        elapsed=(t-ORIGIN_US)/DAY_US;ids=[];weights=[]
        for s in [1,2]:
            pool=np.flatnonzero(state.state==s);chosen=rng.choice(pool,min(len(pool),a.per_state),replace=False)
            ids.extend(chosen);weights.extend([len(pool)/max(1,len(chosen))]*len(chosen))
        ids=np.asarray(ids,np.int32)
        x=state.features(ids,int(t));xx.append(x)
        def query_frame(which,start_id,weight):
            which=np.asarray(which,dtype=np.int32)
            machine=state.machine[which].astype(np.int64).copy()
            unavailable=(state.state[which]!=2)|(machine==state.unknown_machine)
            machine[unavailable]=-1-which[unavailable].astype(np.int64)
            return pd.DataFrame({'query_id':np.arange(start_id,start_id+len(which),dtype=np.int64),'task':which,'cutoff_us':int(t),'elapsed_day':elapsed,'state':state.state[which],
                'job':state.job[which],'machine':machine,'last_source_row':state.last_source[which],'sampling_weight':weight})
        rows.append(query_frame(ids,qid,np.asarray(weights,np.float32)));qid+=len(ids)
        if elapsed>=18 and (k+1)%6==0:
            all_ids=np.flatnonzero(state.state>0);all_x=state.features(all_ids,int(t))
            cq=label(query_frame(all_ids,0,1.),intervals,m['stop_us'])
            name=f'cohort_hour{k+1:04d}';folder=out/name;folder.mkdir()
            np.save(folder/'features.npy',all_x);cq.to_parquet(folder/'queries.parquet',index=False,compression='zstd')
            cohorts.append({'folder':name,'cutoff_us':int(t),'elapsed_day':elapsed,'n':len(cq),'pending':int((cq.state==1).sum()),'running':int((cq.state==2).sum()),
                'queries_sha256':hashlib.sha256((folder/'queries.parquet').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((folder/'features.npy').read_bytes()).hexdigest()})
        if k%24==0:print(json.dumps({'cutoff':k+1,'queries':qid,'full_cohorts':len(cohorts),'elapsed_s':time.monotonic()-start}),flush=True)
    q=label(pd.concat(rows,ignore_index=True),intervals,m['stop_us'])
    q['split']='embargo';q.loc[q.elapsed_day+1/24<=18,'split']='train';q.loc[q.elapsed_day>=18,'split']='development'
    q.to_parquet(out/'queries.parquet',index=False,compression='zstd');np.save(out/'features.npy',np.concatenate(xx))
    report={'config':vars(a),'n_queries':len(q),'features':PERSONAL+GLOBAL+RELATIONAL,'feature_groups':{'personal':PERSONAL,'global':GLOBAL,'relational':RELATIONAL},
        'horizon_minutes':[1,5,15,60],'cohorts':cohorts,'splits':q.split.value_counts().to_dict(),
        'relation_contract':'Job and machine incidence observed at cutoff. Pending/unknown machine is a unique singleton; future assignment is never used.',
        'queries_sha256':hashlib.sha256((out/'queries.parquet').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((out/'features.npy').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256((ROOT/'otg/borg.py').read_bytes()).hexdigest(),'prefix_metadata_sha256':hashlib.sha256((src/'metadata.json').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports/borg_development_cohorts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cohorts','features','feature_groups']}),flush=True)


if __name__=='__main__':main()
