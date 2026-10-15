"""Prepare only the declared prefix of the small Alibaba 2018 task table."""
import argparse,json,sys,tarfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from revision.common import ROOT,write_json,digest
from revision.alibaba_data import prefix_features,PERSONAL,GLOBAL,RELATIONAL,DAY,HOUR


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--period',choices=['development','final'],default='development');a=ap.parse_args();start=time.monotonic()
    if a.period=='final':
        from revision.alibaba_protocol import verify_alibaba_protocol
        verify_alibaba_protocol()
    cap=(6 if a.period=='development' else 8)*DAY;out=ROOT/'data/processed'/f'revision_alibaba_{a.period}_v1';out.mkdir(exist_ok=False);src=ROOT/'data/raw/revision_alibaba2018/batch_task.tar.gz';frames=[]
    with tarfile.open(src,'r:gz') as tar:
        for member in tar:
            if member.name!='batch_task.csv':continue
            with tar.extractfile(member) as f:
                for chunk in pd.read_csv(f,header=None,names=['task_name','instances','job_name','type','status','start','end','cpu','mem'],usecols=['task_name','job_name','type','start','end'],chunksize=500000):
                    q=chunk[(chunk.start>0)&(chunk.start<cap)].copy();q['end']=np.where((q.end>0)&(q.end<cap),q.end,cap);frames.append(q)
    d=pd.concat(frames,ignore_index=True);bad=d.end<d.start;invalid=int(bad.sum());d=d[~bad].sort_values(['start','job_name','task_name'],kind='stable').reset_index(drop=True)
    assert not d.duplicated(['job_name','task_name']).any()
    d['job'],keys=pd.factorize(d.job_name,sort=False);starts=d.start.to_numpy(np.int64);ends=d.end.to_numpy(np.int64);jobs=d.job.to_numpy(np.int64);types=d['type'].to_numpy(np.int64);assert np.all((types>=1)&(types<=12))
    d.to_parquet(out/'tasks.parquet',index=False);rows=[];xs=[];qid=0;cohorts=[]
    # Strictly-before endpoint prevents an integer-second event at the next
    # partition boundary from entering a development label.
    hours=range(1,6*24-1) if a.period=='development' else range(6*24,8*24-1)
    for hour in hours:
        t=hour*HOUR;ids,x=prefix_features(starts,ends,jobs,types,t);xs.append(x);q=pd.DataFrame({'query_id':np.arange(qid,qid+len(ids)),'task':ids,'cutoff_us':t*1_000_000,'elapsed_day':t/DAY,'state':1,'job':jobs[ids],'sampling_weight':1.})
        for h in [1,5,15,60]:q[f'exit_{h}m']=((ends[ids]<cap)&(ends[ids]<=t+60*h)).astype(np.int8)
        rows.append(q);qid+=len(ids);cohorts.append({'hour':hour,'n':len(ids)})
        if hour%12==0:print(json.dumps({'hour':hour,'queries':qid,'elapsed_s':time.monotonic()-start}),flush=True)
    q=pd.concat(rows,ignore_index=True);q.to_parquet(out/'queries.parquet',index=False);np.save(out/'features.npy',np.concatenate(xs))
    write_json(out/'metadata.json',{'period':a.period,'cap_seconds':cap,'rows':len(d),'jobs':len(keys),'negative_intervals_excluded':invalid,'zero_duration_records':int((ends==starts).sum()),'cohorts':cohorts,'features':PERSONAL+GLOBAL+RELATIONAL,'feature_groups':{'personal':PERSONAL,'global':GLOBAL,'relational':RELATIONAL},'horizons':[1,5,15,60],'horizon_unit':'minutes','relations':['job'],'target':'First recorded end of currently running positive-start task. No arrivals, machine incidence, inferred task-completion success or future DAG knowledge.','excluded_features':['eventual status','instance count','planned CPU/memory (availability not separately validated)','future task names/DAG','end time or censoring indicator'],'censoring':'Nonpositive or out-of-prefix end treated as administratively censored; zero-duration tasks never active at a cutoff. Negative intervals are malformed and excluded; their end precedes start, so the anomaly is already observable by start time.','script_sha256':digest(__file__),'feature_module_sha256':digest(ROOT/'revision/alibaba_data.py'),'source_sha256':digest(src),'queries_sha256':digest(out/'queries.parquet'),'elapsed_s':time.monotonic()-start})
    print(json.dumps({'period':a.period,'queries':len(q),'origins':q.cutoff_us.nunique(),'task_records':len(d),'elapsed_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
