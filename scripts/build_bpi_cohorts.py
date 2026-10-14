"""Multi-horizon current-case resolution cohorts; no post-October observations."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.bpi import BPIReplay,DAY_US
ROOT=Path(__file__).resolve().parents[1]


def main():
    src=ROOT/'data/processed/bpi_before_20161101';m=json.loads((src/'metadata.json').read_text())
    out=ROOT/'data/processed/bpi_development_horizons_v1';out.mkdir(parents=True,exist_ok=False)
    e=pd.read_parquet(src/'events.parquet');outcomes=pd.read_parquet(src/'outcomes.parquet').set_index('case')
    # Only training-period activity names; unknown handles later names.
    vocab=sorted(e.loc[e.timestamp<pd.Timestamp('2016-09-01',tz='UTC'),'activity'].unique())
    state=BPIReplay(m['cases'],m['resources'],vocab);lo=qid=0;rows=[];xx=[];start=time.monotonic()
    origin=int(pd.Timestamp('2016-01-01',tz='UTC').value//1000);times=e.time_us.to_numpy()
    cutoff_end=pd.Timestamp('2016-11-01',tz='UTC')-pd.Timedelta(days=14)
    for date in pd.date_range(pd.Timestamp('2016-01-08 12:00',tz='UTC'),cutoff_end,freq='D'):
        t=int(date.value//1000);hi=int(np.searchsorted(times,t,side='right'));state.apply(e.iloc[lo:hi]);lo=hi
        ids=np.flatnonzero((state.created>=0)&~state.resolved).astype(np.int32)
        x,groups=state.features(ids,t);xx.append(x);y=outcomes.loc[ids]
        resource=state.resource[ids].astype(np.int64);resource[resource==m['resources']]=-1-ids[resource==m['resources']].astype(np.int64)
        q=pd.DataFrame({'query_id':np.arange(qid,qid+len(ids)),'case':ids,'cutoff_us':t,'elapsed_day':(t-origin)/DAY_US,'state':1,'resource':resource,'sampling_weight':1.,
            'remaining_s':(y.resolution_us.to_numpy()-t)/1e6,'resolution_observed':y.resolution_observed.to_numpy()})
        for horizon in [1,3,7,14]:q[f'exit_{horizon}d']=((q.remaining_s<=horizon*86400)&q.resolution_observed).astype(np.int8)
        q['split']='embargo'
        if date+pd.Timedelta(days=14)<=pd.Timestamp('2016-09-01',tz='UTC'):q['split']='train'
        elif date>=pd.Timestamp('2016-09-01',tz='UTC'):q['split']='development'
        rows.append(q);qid+=len(ids)
    q=pd.concat(rows,ignore_index=True);x=np.concatenate(xx)
    q.to_parquet(out/'queries.parquet',index=False,compression='zstd');np.save(out/'features.npy',x)
    report={'n_queries':len(q),'features':sum(groups.values(),[]),'feature_groups':groups,'horizon_days':[1,3,7,14],'cohort_unit':'Complete unresolved-case population at each daily cutoff; unknown handler is singleton.',
        'splits':q.split.value_counts().to_dict(),'queries_sha256':hashlib.sha256((out/'queries.parquet').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((out/'features.npy').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256((ROOT/'otg/bpi.py').read_bytes()).hexdigest(),'source_metadata_sha256':hashlib.sha256((src/'metadata.json').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports/bpi_development_cohorts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['features','feature_groups']}),flush=True)


if __name__=='__main__':main()
