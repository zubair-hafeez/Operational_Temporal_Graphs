"""Causal Borg state reconstruction for development experiments.

All context aggregates use state at the cutoff. Future intervals are labels only.
This module does not implement the proposed generative model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

EVENT_STATE=np.array([1,2,0,0,0,0,0,1,2],dtype=np.int8)
ORIGIN_US=600_000_000
DAY_US=86_400_000_000

PERSONAL=['state','log_state_age_s','unknown_state_age','log_since_event_s','priority','sched_class','cpu','memory','cpu_missing','memory_missing','log_seen_events','log_evictions','imputed_ever','log_since_first_seen_s']
GLOBAL=['hour_sin','hour_cos','week_sin','week_cos','log_global_pending','log_global_running']
RELATIONAL=['log_job_seen','log_job_pending','log_job_running','job_running_mean_age_log','job_running_cpu_sum','job_running_memory_sum','log_machine_running','machine_cpu_sum','machine_memory_sum','machine_priority_mean','machine_age_log_mean','log_job_running_machine_degree']


@dataclass
class BorgReplay:
    n_tasks:int
    n_jobs:int
    n_machines:int
    unknown_machine:int

    def __post_init__(self):
        n=self.n_tasks
        self.state=np.zeros(n,np.int8)
        self.job=np.zeros(n,np.int32)
        self.machine=np.full(n,self.unknown_machine,np.int32)
        self.state_since=np.full(n,ORIGIN_US,np.int64)
        self.last_time=np.full(n,ORIGIN_US,np.int64)
        self.first_time=np.full(n,-1,np.int64)
        self.unknown_age=np.ones(n,np.bool_)
        self.priority=np.zeros(n,np.float32)
        self.sched=np.zeros(n,np.float32)
        self.cpu=np.full(n,np.nan,np.float32)
        self.memory=np.full(n,np.nan,np.float32)
        self.event_count=np.zeros(n,np.int32)
        self.evictions=np.zeros(n,np.int32)
        self.imputed=np.zeros(n,np.bool_)
        self.last_source=np.full(n,-1,np.int64)
        self.time_us=-1

    def apply(self, events:pd.DataFrame):
        # Latest per-task values can be applied in bulk while first/last state
        # transitions and event counters retain every event's effect.
        if not len(events):return
        if int(events.time_us.min())<self.time_us:
            raise ValueError('Replay events must be chronological')
        task=events.task.to_numpy(dtype=np.int32)
        types=events.event_type.to_numpy(dtype=np.int8)
        states=EVENT_STATE[types]
        times=events.time_us.to_numpy(dtype=np.int64)
        old_series=events.assign(_state=states).groupby('task',sort=False)._state.shift(1)
        prior=old_series.to_numpy()
        first_in_batch=np.isnan(prior)
        prior[first_in_batch]=self.state[task[first_in_batch]]
        changed=(prior!=states) | (self.first_time[task]<0)&first_in_batch
        idx=np.flatnonzero(changed)
        if len(idx):
            change=pd.DataFrame({'task':task[idx],'t':times[idx]}).drop_duplicates('task',keep='last')
            ti=change.task.to_numpy();tt=change.t.to_numpy()
            self.state_since[ti]=np.maximum(tt,ORIGIN_US)
            self.unknown_age[ti]=tt==0
        first=events.drop_duplicates('task',keep='first')
        ids=first.task.to_numpy();unseen=self.first_time[ids]<0
        self.first_time[ids[unseen]]=np.maximum(first.time_us.to_numpy()[unseen],ORIGIN_US)
        np.add.at(self.event_count,task,1)
        np.add.at(self.evictions,task,types==2)
        imputed_ids=events.loc[events['missing']>=0,'task'].to_numpy()
        self.imputed[imputed_ids]=True
        last=events.drop_duplicates('task',keep='last')
        ids=last.task.to_numpy();self.state[ids]=EVENT_STATE[last.event_type.to_numpy()]
        self.job[ids]=last.job.to_numpy();self.machine[ids]=last.machine.to_numpy()
        self.last_time[ids]=np.maximum(last.time_us.to_numpy(),ORIGIN_US)
        self.last_source[ids]=last.source_row.to_numpy()
        self.priority[ids]=last.priority.to_numpy()
        # A missing request does not erase a previously known request.
        for column,attr in [('cpu','cpu'),('memory','memory'),('sched_class','sched')]:
            known=events.loc[events[column].notna(),['task',column]].drop_duplicates('task',keep='last')
            getattr(self,attr)[known.task.to_numpy()]=known[column].to_numpy()
        self.time_us=int(times[-1])

    def features(self, ids, cutoff_us):
        ids=np.asarray(ids,dtype=np.int32)
        if cutoff_us<self.time_us:raise ValueError('Features requested before replay time')
        if np.any(self.state[ids]==0):raise ValueError('Features require active tasks')
        running=np.flatnonzero(self.state==2);pending=np.flatnonzero(self.state==1)
        seen=np.flatnonzero(self.first_time>=0)
        job_seen=np.bincount(self.job[seen],minlength=self.n_jobs)
        job_pending=np.bincount(self.job[pending],minlength=self.n_jobs)
        job_running=np.bincount(self.job[running],minlength=self.n_jobs)
        age=np.log1p(np.maximum(0,(cutoff_us-self.state_since)/1e6))
        cpu=np.nan_to_num(self.cpu,nan=0.0);memory=np.nan_to_num(self.memory,nan=0.0)
        def job_sum(values):return np.bincount(self.job[running],weights=values[running],minlength=self.n_jobs)
        machine_valid=running[self.machine[running]!=self.unknown_machine]
        m_count=np.bincount(self.machine[machine_valid],minlength=self.n_machines)
        def machine_sum(values):return np.bincount(self.machine[machine_valid],weights=values[machine_valid],minlength=self.n_machines)
        # Degree of observed active job-to-machine incidence, not future topology.
        pairs=np.unique(self.job[machine_valid].astype(np.int64)*self.n_machines+self.machine[machine_valid])
        job_degree=np.bincount(pairs//self.n_machines,minlength=self.n_jobs)
        j=self.job[ids];m=self.machine[ids]
        elapsed=(cutoff_us-ORIGIN_US)/1e6
        hour=(elapsed/3600+19)%24
        week=(elapsed/86400+6+19/24)%7
        personal=np.column_stack([self.state[ids],age[ids],self.unknown_age[ids],np.log1p(np.maximum(0,(cutoff_us-self.last_time[ids])/1e6)),self.priority[ids],self.sched[ids],cpu[ids],memory[ids],np.isnan(self.cpu[ids]),np.isnan(self.memory[ids]),np.log1p(self.event_count[ids]),np.log1p(self.evictions[ids]),self.imputed[ids],np.log1p(np.maximum(0,(cutoff_us-self.first_time[ids])/1e6))])
        global_=np.tile([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*week/7),np.cos(2*np.pi*week/7),np.log1p(len(pending)),np.log1p(len(running))],(len(ids),1))
        rel=np.column_stack([np.log1p(job_seen[j]),np.log1p(job_pending[j]),np.log1p(job_running[j]),job_sum(age)[j]/np.maximum(job_running[j],1),job_sum(cpu)[j],job_sum(memory)[j],np.log1p(m_count[m]),machine_sum(cpu)[m],machine_sum(memory)[m],machine_sum(self.priority)[m]/np.maximum(m_count[m],1),machine_sum(age)[m]/np.maximum(m_count[m],1),np.log1p(job_degree[j])])
        # No machine assignment is known while pending. Endpoint fields on
        # termination/submission records must not reveal eventual destinations.
        rel[self.state[ids]!=2,6:11]=0
        return np.column_stack([personal,global_,rel]).astype(np.float32)


def assign_interval_labels(intervals, queries, horizon_us, observation_end_us):
    """Return first state-exit targets using isolated future interval information.

    Queries must be active after all events at cutoff. A tied zero-length interval
    is never selected. Output is separate from model input features.
    """
    left=queries[['task','cutoff_us','query_id']].sort_values('cutoff_us',kind='stable')
    right=intervals[['task','time_us','end_us','end_type','state']].sort_values('time_us',kind='stable')
    joined=pd.merge_asof(left,right,by='task',left_on='cutoff_us',right_on='time_us',direction='backward')
    if joined.state.isna().any() or (joined.state==0).any() or (joined.end_us<=joined.cutoff_us).any():
        raise ValueError('Query does not identify an active nonzero interval')
    observed=joined.cutoff_us.to_numpy()+horizon_us<=observation_end_us
    happened=(joined.end_us.to_numpy()<=joined.cutoff_us.to_numpy()+horizon_us)&(joined.end_type.to_numpy()>=0)
    result=pd.DataFrame({'query_id':joined.query_id.to_numpy(),'exit':happened.astype(np.int8),'exit_type':np.where(happened,joined.end_type.to_numpy(),-1).astype(np.int8),'observed':observed,'remaining_s':(joined.end_us.to_numpy()-joined.cutoff_us.to_numpy())/1e6})
    return result.sort_values('query_id',kind='stable').reset_index(drop=True)
