"""Investigate exceptional transitions before adopting lifecycle masks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_parquet(ROOT/'data/processed/borg_prefix_7d/events.parquet')
states=np.array([1,2,0,0,0,0,0,1,2],np.int8)
df['state']=states[df.event_type.to_numpy()]
df['prior_state']=df.groupby('task',sort=False).state.shift(1)
df['prior_us']=df.groupby('task',sort=False).time_us.shift(1)
bad=df[(df.time_us>0)&(df['missing']==-1)&(((df.prior_state==1)&df.event_type.isin([2,4,8]))|((df.prior_state==2)&df.event_type.isin([0,1,7])))]
summary={}
for (old,event),group in bad.groupby(['prior_state','event_type']):
    delta=(group.time_us-group.prior_us)/1e6
    summary[f'{int(old)}_event{event}']=dict(count=len(group),same_microsecond=int((delta==0).sum()),within_1ms=int((delta<=.001).sum()),within_1second=int((delta<=1).sum()),delta_s_quantiles=delta.quantile([0,.5,.9,.99,1]).to_dict())
bad.to_parquet(ROOT/'data/processed/borg_prefix_7d/exceptional_transitions.parquet',index=False)
sample_tasks=bad.task.drop_duplicates().iloc[:12]
examples=df[df.task.isin(sample_tasks)]
examples.to_parquet(ROOT/'data/processed/borg_prefix_7d/exceptional_task_examples.parquet',index=False)
report=dict(summary=summary,sample_tasks=sample_tasks.tolist(),sample_event_rows=len(examples),scope='7-day development prefix; raw time/source order, before tie reconciliation')
(ROOT/'reports/borg_transition_exceptions.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
