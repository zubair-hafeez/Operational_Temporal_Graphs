"""Build fixed final forecast origins only after verifying the frozen protocol."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.protocol import verify_final_protocol
from otg.borg import BorgReplay,ORIGIN_US,DAY_US
from otg.bpi import BPIReplay
from scripts.build_borg_cohorts import label
from scripts.add_borg_event_history import sparse_counts,WINDOWS
ROOT=Path(__file__).resolve().parents[1]


def save_cohort(out,name,q,x,extra=None):
    folder=out/name;folder.mkdir();q.to_parquet(folder/'queries.parquet',index=False,compression='zstd');np.save(folder/'features.npy',x)
    return {'folder':name,'cutoff_us':int(q.cutoff_us.iloc[0]),'n':len(q),'queries_sha256':hashlib.sha256((folder/'queries.parquet').read_bytes()).hexdigest(),
            'features_sha256':hashlib.sha256((folder/'features.npy').read_bytes()).hexdigest(),**(extra or {})}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);a=ap.parse_args();protocol,protocol_hash=verify_final_protocol(ROOT)
    out=ROOT/'data/processed'/f'{a.dataset}_final_cohorts_v1';out.mkdir(parents=True,exist_ok=False);start=time.monotonic();cohorts=[]
    version='history_v1' if a.dataset=='borg' else 'horizons_v1';source=ROOT/'data/processed'/f'{a.dataset}_development_{version}';sm=json.loads((source/'metadata.json').read_text())
    prefix=ROOT/'data/processed'/('borg_prefix_29d' if a.dataset=='borg' else 'bpi_before_20170116');pm=json.loads((prefix/'metadata.json').read_text())
    e=pd.read_parquet(prefix/'events.parquet');times=e.time_us.to_numpy();lo=0
    if a.dataset=='borg':
        intervals=pd.read_parquet(prefix/'intervals.parquet');replay=BorgReplay(pm['tasks'],pm['jobs'],pm['machines'],pm['unknown_machine'])
        for hour in range(6,29*24,6):
            t=ORIGIN_US+hour*3600_000_000;hi=np.searchsorted(times,t,side='right');replay.apply(e.iloc[lo:hi]);lo=hi
            if hour<22*24:continue
            if t+3600_000_000>pm['stop_us']:raise ValueError('Final horizon exceeds observed endpoint')
            ids=np.flatnonzero(replay.state>0).astype(np.int32);machine=replay.machine[ids].astype(np.int64).copy();missing=(replay.state[ids]!=2)|(machine==replay.unknown_machine);machine[missing]=-1-ids[missing].astype(np.int64)
            q=pd.DataFrame({'query_id':np.arange(len(ids)),'task':ids,'cutoff_us':t,'elapsed_day':hour/24,'state':replay.state[ids],'job':replay.job[ids],'machine':machine,'last_source_row':replay.last_source[ids],'sampling_weight':1.})
            x=replay.features(ids,t);extra=[]
            for window in WINDOWS:
                first=np.searchsorted(times,t-window*60_000_000,side='right');recent=e.iloc[first:hi]
                extra.append(sparse_counts(recent,'task',q.task));extra.append(np.tile(np.log1p(np.bincount(recent.event_type.to_numpy(),minlength=9)).astype(np.float32),(len(q),1)))
                extra.append(sparse_counts(recent,'job',q.job));extra.append(sparse_counts(recent,'machine',q.machine))
            x=np.column_stack([x,*extra]);q=label(q,intervals,pm['stop_us']);name=f'cohort_hour{hour:04d}'
            cohorts.append(save_cohort(out,name,q,x,{'elapsed_day':hour/24,'pending':int((q.state==1).sum()),'running':int((q.state==2).sum())}));print(json.dumps({'cohort':name,'n':len(q),'elapsed_s':time.monotonic()-start}),flush=True)
    else:
        vocab=sorted(e.loc[e.timestamp<pd.Timestamp('2016-09-01',tz='UTC'),'activity'].unique());replay=BPIReplay(pm['cases'],pm['resources'],vocab)
        outcomes=pd.read_parquet(prefix/'outcomes.parquet').set_index('case');origin=int(pd.Timestamp('2016-01-01',tz='UTC').value//1000)
        # Replay only once to each origin; outcomes are joined after features.
        for date in pd.date_range('2016-11-01 12:00','2016-12-31 12:00',freq='D',tz='UTC'):
            t=int(date.value//1000);hi=np.searchsorted(times,t,side='right');replay.apply(e.iloc[lo:hi]);lo=hi
            if t+14*DAY_US>pm['observation_end_us']:raise ValueError('Final horizon exceeds observed endpoint')
            ids=np.flatnonzero((replay.created>=0)&~replay.resolved).astype(np.int32);x,groups=replay.features(ids,t)
            if sum(groups.values(),[])!=sm['features']:raise ValueError('Frozen feature vocabulary changed')
            resource=replay.resource[ids].astype(np.int64);missing=resource==pm['resources'];resource[missing]=-1-ids[missing].astype(np.int64)
            q=pd.DataFrame({'query_id':np.arange(len(ids)),'case':ids,'cutoff_us':t,'elapsed_day':(t-origin)/DAY_US,'state':1,'resource':resource,'sampling_weight':1.})
            y=outcomes.loc[ids];q['remaining_s']=(y.resolution_us.to_numpy()-t)/1e6;q['resolution_observed']=y.resolution_observed.to_numpy()
            for h in [1,3,7,14]:q[f'exit_{h}d']=((q.remaining_s<=h*86400)&q.resolution_observed).astype(np.int8)
            name=f'cohort_{t}';cohorts.append(save_cohort(out,name,q,x));print(json.dumps({'cohort':name,'n':len(q),'elapsed_s':time.monotonic()-start}),flush=True)
    report={'dataset':a.dataset,'period':'final','cohorts':cohorts,'features':sm['features'],'feature_groups':sm['feature_groups'],'prefix':str(prefix.relative_to(ROOT)),
            'frozen_protocol_sha256':protocol_hash,'prefix_metadata_sha256':hashlib.sha256((prefix/'metadata.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports'/f'{a.dataset}_final_cohorts.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
