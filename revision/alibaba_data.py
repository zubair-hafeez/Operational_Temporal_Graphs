"""Prefix-observable features for running Alibaba tasks; no retrospective DAG."""
import numpy as np
import pandas as pd

HOUR=3600
DAY=86400
PERSONAL=['log_age_minutes']+[f'type_{k}' for k in range(1,13)]
GLOBAL=['hour_sin','hour_cos','log_active','log_active_jobs','log_starts_1h','log_ends_1h','log_starts_6h','log_ends_6h','log_active_same_type']
RELATIONAL=['log_job_active','log_job_started','log_job_ended','log_job_age_minutes','job_mean_log_age','log_since_job_start_minutes']


def prefix_features(starts,ends,jobs,types,t):
    """Only starts<=t and ends<=t can influence features or membership.

    Changing any ending time strictly after t leaves the output unchanged.
    End=observation-cap represents administrative censoring, not a feature.
    """
    seen=starts<=t;finished=ends<=t;active=seen&~finished;ids=np.flatnonzero(active);n_jobs=int(jobs.max())+1
    age=np.log1p((t-starts[ids])/60);typ=types[ids];onehot=np.eye(12,dtype=np.float32)[np.clip(typ,1,12)-1]
    personal=np.column_stack([age,onehot]);ja=np.bincount(jobs[ids],minlength=n_jobs);js=np.bincount(jobs[seen],minlength=n_jobs);je=np.bincount(jobs[finished],minlength=n_jobs)
    oldest=np.full(n_jobs,t,np.int64);np.minimum.at(oldest,jobs[seen],starts[seen]);latest=np.zeros(n_jobs,np.int64);np.maximum.at(latest,jobs[seen],starts[seen]);agesum=np.bincount(jobs[ids],weights=age,minlength=n_jobs)
    pertype=np.bincount(typ,minlength=13);phase=2*np.pi*(t%DAY)/DAY
    global_values=[np.sin(phase),np.cos(phase),np.log1p(len(ids)),np.log1p(np.count_nonzero(ja)),np.log1p(np.sum((starts>t-HOUR)&seen)),np.log1p(np.sum((ends>t-HOUR)&finished)),np.log1p(np.sum((starts>t-6*HOUR)&seen)),np.log1p(np.sum((ends>t-6*HOUR)&finished))]
    glob=np.column_stack([np.tile(global_values,(len(ids),1)),np.log1p(pertype[typ])]);j=jobs[ids]
    rel=np.column_stack([np.log1p(ja[j]),np.log1p(js[j]),np.log1p(je[j]),np.log1p((t-oldest[j])/60),agesum[j]/ja[j],np.log1p((t-latest[j])/60)])
    return ids,np.column_stack([personal,glob,rel]).astype(np.float32)
