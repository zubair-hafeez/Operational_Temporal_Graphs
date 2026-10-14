"""Daily unresolved-case queries with all-case prefix context and isolated labels."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.bpi import BPIReplay,DAY_US
ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.monotonic();src=ROOT/'data/processed/bpi_before_20161101'
    e=pd.read_parquet(src/'events.parquet');m=json.loads((src/'metadata.json').read_text())
    fit_end=int(pd.Timestamp('2016-07-01',tz='UTC').value//1000)
    vocab=sorted(e.loc[e.time_us<fit_end,'activity'].unique())
    state=BPIReplay(m['cases'],m['resources'],vocab)
    outcome=pd.read_parquet(src/'outcomes.parquet').set_index('case')
    dates=pd.date_range('2016-01-08 12:00','2016-08-31 12:00',freq='D',tz='UTC')
    times=e.time_us.to_numpy();lo=0;rows=[];xx=[];qid=0
    for date in dates:
        t=int(date.value//1000);hi=int(np.searchsorted(times,t,side='right'))
        state.apply(e.iloc[lo:hi]);lo=hi
        ids=np.flatnonzero((state.created>=0)&~state.resolved)
        x,groups=state.features(ids,t);xx.append(x)
        horizon=7*DAY_US;end=t+horizon
        split='embargo'
        if end<=fit_end:split='train'
        elif t>=fit_end and end<=int(pd.Timestamp('2016-08-01',tz='UTC').value//1000):split='validation'
        elif t>=int(pd.Timestamp('2016-08-01',tz='UTC').value//1000) and end<=int(pd.Timestamp('2016-09-01',tz='UTC').value//1000):split='pilot_test'
        y=outcome.loc[ids]
        rows.append(pd.DataFrame({'query_id':np.arange(qid,qid+len(ids)),'case':ids,'cutoff_us':t,'state':1,'sampling_weight':1.,'split':split,'resource':state.resource[ids],
            'exit':((y.resolution_us.to_numpy()<=end)&y.resolution_observed.to_numpy()).astype(np.int8),
            'remaining_s':(y.resolution_us.to_numpy()-t)/1e6,'resolution_observed':y.resolution_observed.to_numpy()}))
        qid+=len(ids)
    out=ROOT/'data/processed/bpi_pilot_h7d';out.mkdir(parents=True,exist_ok=True)
    q=pd.concat(rows,ignore_index=True);x=np.concatenate(xx)
    q.to_parquet(out/'queries.parquet',index=False,compression='zstd');np.save(out/'features.npy',x)
    report={'n_queries':len(q),'n_features':x.shape[1],'feature_groups':groups,'features':sum(groups.values(),[]),'splits':q.split.value_counts().to_dict(),'horizon_days':7,
        'resource_edge_semantics':'Most recent observed workflow handler; not current service occupancy or exclusive assignment. Unknown is not a shared resource.',
        'queries_sha256':hashlib.sha256((out/'queries.parquet').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((out/'features.npy').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256((ROOT/'otg/bpi.py').read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports/bpi_pilot_build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['features','feature_groups']}),flush=True)

if __name__=='__main__':main()
