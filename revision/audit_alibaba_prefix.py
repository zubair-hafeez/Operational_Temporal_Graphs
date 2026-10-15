"""Feasibility only; discard all duration information beyond elapsed day 2."""
import hashlib,json,sys,tarfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from revision.common import ROOT,write_json,digest


def main():
    src=ROOT/'data/raw/revision_alibaba2018/batch_task.tar.gz';assert digest(src)=='7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6';start=time.monotonic();cap=2*86400;frames=[]
    names=['task','instances','job','type','status','start','end','cpu','mem']
    with tarfile.open(src,'r:gz') as archive:
        for member in archive:
            if member.name!='batch_task.csv':continue
            with archive.extractfile(member) as f:
                for chunk in pd.read_csv(f,header=None,names=names,usecols=['task','job','type','start','end','cpu','mem'],chunksize=500000):
                    keep=chunk[(chunk.start>0)&(chunk.start<cap)].copy()
                    keep['end']=np.where((keep.end>0)&(keep.end<cap),keep.end,cap)
                    frames.append(keep)
    d=pd.concat(frames,ignore_index=True);out=ROOT/'data/processed/revision_alibaba_prefix2d_audit';out.mkdir(exist_ok=False);d.to_parquet(out/'tasks.parquet',index=False)
    origins=[]
    for t in range(24*3600,48*3600,3600):
        active=d[(d.start<=t)&(d.end>t)];groups=active.groupby('job').size();origins.append({'hour':t/3600,'tasks':len(active),'groups_ge5':int((groups>=5).sum()),'tasks_in_groups_ge5':int(groups[groups>=5].sum())})
    report={'scope':'Only positive task starts before elapsed day2; end times beyond this prefix or absent are both censored at day2. Eventual status excluded. No final-duration statistics.','rows':len(d),'jobs':d.job.nunique(),'duplicate_job_task_rows':int(d.duplicated(['job','task']).sum()),'nonpositive_intervals_in_prefix':int((d.end<=d.start).sum()),'types':d.type.value_counts().to_dict(),'fully_observed_ends':int((d.end<cap).sum()),'censored_at_prefix':int((d.end==cap).sum()),'cohorts':origins,'script_sha256':digest(__file__),'source_sha256':digest(src),'elapsed_s':time.monotonic()-start}
    write_json(ROOT/'reports/revision_alibaba_feasibility.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
