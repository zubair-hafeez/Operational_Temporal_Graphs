"""Prefix-only case state; resource edges mean last observed handling, not capacity."""
from __future__ import annotations
import numpy as np
import pandas as pd

DAY_US=86_400_000_000
RESOLUTION={'A_Pending','A_Denied','A_Cancelled'}
HANDLING={'start','resume','suspend','complete','ate_abort'}


class BPIReplay:
    def __init__(self,n_cases,n_resources,activities):
        self.n=n_cases;self.nr=n_resources;self.activities=list(activities)
        self.vocab={name:i+1 for i,name in enumerate(activities)}
        self.created=np.full(n_cases,-1,np.int64)
        self.last=np.zeros(n_cases,np.int64)
        self.resolved=np.zeros(n_cases,bool)
        self.last_activity=np.zeros(n_cases,np.int16)
        self.last_app=np.zeros(n_cases,np.int16)
        self.resource=np.full(n_cases,n_resources,np.int16)
        self.resource_time=np.zeros(n_cases,np.int64)
        self.counts=np.zeros((n_cases,len(activities)+1),np.int32)
        self.offers={};self.time_us=-1
        self.recent=pd.DataFrame()

    def apply(self,e):
        if not len(e):return
        if e.time_us.min()<self.time_us:raise ValueError('Out-of-order replay')
        created=e[e.activity=='A_Create Application'].drop_duplicates('case')
        ids=created.case.to_numpy();new=self.created[ids]<0
        self.created[ids[new]]=created.time_us.to_numpy()[new]
        self.resolved[e.loc[e.activity.isin(RESOLUTION),'case'].to_numpy()]=True
        c=e.case.to_numpy();a=e.activity.map(self.vocab).fillna(0).to_numpy(np.int16)
        np.add.at(self.counts,(c,a),1)
        last=e.drop_duplicates('case',keep='last');ids=last.case.to_numpy()
        self.last[ids]=last.time_us.to_numpy()
        self.last_activity[ids]=last.activity.map(self.vocab).fillna(0).to_numpy()
        app=e[e.activity.str.startswith('A_')].drop_duplicates('case',keep='last')
        self.last_app[app.case.to_numpy()]=app.activity.map(self.vocab).fillna(0).to_numpy()
        worker=e[e.activity.str.startswith('W_')&e.lifecycle.isin(HANDLING)].drop_duplicates('case',keep='last')
        self.resource[worker.case.to_numpy()]=worker.resource.to_numpy()
        self.resource_time[worker.case.to_numpy()]=worker.time_us.to_numpy()
        offers=e[e.offer_id.notna()].drop_duplicates('offer',keep='last')
        for row in offers.itertuples():self.offers[row.offer]=(row.case,row.activity)
        self.time_us=int(e.time_us.iloc[-1])
        self.recent=pd.concat([self.recent,e[['case','time_us','activity','resource','lifecycle']]],ignore_index=True)
        self.recent=self.recent[self.recent.time_us>self.time_us-7*DAY_US].copy()

    def features(self,ids,t):
        ids=np.asarray(ids,np.int32)
        if t<self.time_us or np.any(self.created[ids]<0) or np.any(self.resolved[ids]):raise ValueError('Invalid query')
        active=np.flatnonzero((self.created>=0)&~self.resolved)
        age=np.log1p(np.maximum(0,(t-self.created)/DAY_US))
        offer_count=np.zeros(self.n);open_count=np.zeros(self.n)
        for case,activity in self.offers.values():
            offer_count[case]+=1
            open_count[case]+=activity not in {'O_Cancelled','O_Refused','O_Accepted'}
        eye=np.eye(len(self.activities)+1,dtype=np.float32)
        personal=np.column_stack([age[ids],np.log1p((t-self.last[ids])/DAY_US),np.log1p(self.counts[ids].sum(1)),np.log1p(offer_count[ids]),np.log1p(open_count[ids]),eye[self.last_activity[ids]],eye[self.last_app[ids]],np.log1p(self.counts[ids])])
        names=['log_age_days','log_since_event_days','log_events','log_offers_seen','log_offers_open']+[f'last_{x}' for x in ['UNKNOWN']+self.activities]+[f'app_{x}' for x in ['UNKNOWN']+self.activities]+[f'count_{x}' for x in ['UNKNOWN']+self.activities]
        recent=self.recent[self.recent.time_us>t-7*DAY_US]
        day=recent[recent.time_us>t-DAY_US]
        date=pd.Timestamp(t,unit='us',tz='UTC')
        glob=[np.sin(2*np.pi*date.dayofweek/7),np.cos(2*np.pi*date.dayofweek/7),np.sin(2*np.pi*date.dayofyear/366),np.cos(2*np.pi*date.dayofyear/366),np.log1p(len(active))]
        gnames=['weekday_sin','weekday_cos','year_sin','year_cos','log_active_cases']
        for name,frame in [('day',day),('week',recent)]:
            glob.extend([np.log1p(len(frame)),np.log1p((frame.activity=='A_Create Application').sum()),np.log1p(frame.activity.isin(RESOLUTION).sum())])
            gnames.extend([f'log_{name}_events',f'log_{name}_arrivals',f'log_{name}_resolutions'])
        r=self.resource[ids];ra=self.resource[active]
        rc=np.bincount(ra,minlength=self.nr+1)
        age_sum=np.bincount(ra,weights=age[active],minlength=self.nr+1)
        stage_counts=np.zeros((self.nr+1,len(self.activities)+1))
        np.add.at(stage_counts,(ra,self.last_app[active]),1)
        handling=recent[recent.activity.str.startswith('W_')&recent.lifecycle.isin(HANDLING)]
        re=np.bincount(handling.resource.to_numpy(),minlength=self.nr+1)
        rel=np.column_stack([r==self.nr,np.log1p(np.maximum(0,(t-self.resource_time[ids])/DAY_US)),np.log1p(rc[r]),age_sum[r]/np.maximum(rc[r],1),np.log1p(stage_counts[r,self.last_app[ids]]),np.log1p(re[r])])
        rel[r==self.nr,1:]=0
        rnames=['resource_unknown','log_since_resource_days','log_resource_active_cases','resource_mean_log_age','log_resource_same_stage_cases','log_resource_week_handling_events']
        return np.column_stack([personal,np.tile(glob,(len(ids),1)),rel]).astype(np.float32),{'personal':names,'global':gnames,'relational':rnames}
