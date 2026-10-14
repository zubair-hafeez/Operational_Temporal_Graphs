"""Prepare BPI development events without copying retrospective trace attributes."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESOLUTION={'A_Pending','A_Denied','A_Cancelled'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',default='2016-11-01');a=p.parse_args()
    if pd.Timestamp(a.before)>pd.Timestamp('2016-11-01'):
        raise ValueError('Final period remains locked')
    src=ROOT/'data/raw/bpi/bpi2017.xes.gz'
    rows=[];total_cases=total_events=0;start=time.monotonic()
    with gzip.open(src,'rb') as f:
        for _,elem in ET.iterparse(f,events=['end']):
            if elem.tag.split('}')[-1]!='trace':continue
            attrs={x.get('key'):x.get('value') for x in elem if x.get('key')}
            case=attrs['concept:name'];total_cases+=1
            for child in elem:
                if child.tag.split('}')[-1]!='event':continue
                e={x.get('key'):x.get('value') for x in child};total_events+=1
                if e.get('time:timestamp','')>=a.before:continue
                rows.append((case,e.get('time:timestamp'),e.get('concept:name'),e.get('lifecycle:transition'),e.get('org:resource'),e.get('EventOrigin'),e.get('EventID'),e.get('OfferID'),total_events))
            elem.clear()
    df=pd.DataFrame(rows,columns=['case_id','timestamp','activity','lifecycle','resource_id','origin','event_id','offer_id','source_row']);del rows
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
    df=df.sort_values(['timestamp','source_row'],kind='stable').reset_index(drop=True)
    df['time_us']=df.timestamp.astype('int64')//1000
    df['case'],case_keys=pd.factorize(df.case_id,sort=False);df['case']=df.case.astype('int32')
    df['resource'],resource_keys=pd.factorize(df.resource_id.fillna('__UNKNOWN__'),sort=False);df['resource']=df.resource.astype('int16')
    df['offer'],offer_keys=pd.factorize(df.offer_id.fillna('__NONE__'),sort=False);df['offer']=df.offer.astype('int32')
    df['activity_code'],activity_keys=pd.factorize(df.activity,sort=False);df['activity_code']=df.activity_code.astype('int8')
    df['lifecycle_code'],lifecycle_keys=pd.factorize(df.lifecycle,sort=False);df['lifecycle_code']=df.lifecycle_code.astype('int8')
    out=ROOT/'data/processed'/('bpi_before_'+a.before.replace('-',''));out.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out/'events.parquet',index=False,compression='zstd')
    resolution=df[df.activity.isin(RESOLUTION)]
    first_res=resolution.drop_duplicates('case',keep='first')[['case','time_us','activity']].rename(columns={'time_us':'resolution_us','activity':'resolution_activity'})
    starts=df[df.activity=='A_Create Application'].drop_duplicates('case')[['case','time_us']].rename(columns={'time_us':'created_us'})
    outcomes=starts.merge(first_res,on='case',how='left')
    outcomes['resolution_observed']=outcomes.resolution_us.notna()
    observation_end=int(pd.Timestamp(a.before,tz='UTC').value//1000)
    outcomes['resolution_us']=outcomes.resolution_us.fillna(observation_end).astype('int64')
    outcomes.to_parquet(out/'outcomes.parquet',index=False)
    prefix=df[df.timestamp<pd.Timestamp('2016-09-01',tz='UTC')]
    # This diagnostic is restricted to fit-period observations.
    pres=prefix[prefix.activity.isin(RESOLUTION)]
    pfirst=pres.drop_duplicates('case')[['case','time_us']].rename(columns={'time_us':'resolution_us'})
    post=prefix.merge(pfirst,on='case',how='inner')
    post=post[post.time_us>post.resolution_us]
    post_application=post[post.activity.str.startswith('A_')]
    offer_case=prefix[prefix.offer_id.notna()].groupby('offer_id').case.nunique()
    meta=dict(before=a.before,rows=len(df),cases=len(case_keys),resources=len(resource_keys),offers_excluding_none=int(len(offer_keys)-('__NONE__' in offer_keys)),
              activity_names=list(activity_keys),lifecycle_names=list(lifecycle_keys),resource_names=list(resource_keys),offer_none_index=int(np.flatnonzero(offer_keys=='__NONE__')[0]),
              observation_end_us=observation_end,raw_case_count=total_cases,raw_event_count=total_events,
              excluded_inputs=['all trace attributes','offer Selected/Accepted flags','amounts or attributes not yet availability-audited'],
              fit_period_audit=dict(rows=len(prefix),cases=prefix.case.nunique(),duplicate_event_id_rows=int(prefix.event_id.duplicated().sum()),
                  resolutions=len(pres),cases_with_multiple_resolutions=int((pres.groupby('case').size()>1).sum()),
                  post_resolution_events=len(post),post_resolution_application_events=len(post_application),
                  post_resolution_application_types=post_application.activity.value_counts().to_dict(),
                  offers_linked_to_multiple_cases=int((offer_case>1).sum()),
                  workflow_lifecycle_counts=prefix[prefix.origin=='Workflow'].lifecycle.value_counts().to_dict()),
              raw_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),events_sha256=hashlib.sha256((out/'events.parquet').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_s=time.monotonic()-start)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=int)+'\n')
    (ROOT/'reports/bpi_development_audit.json').write_text(json.dumps(meta,indent=2,default=int)+'\n')
    print(json.dumps(meta,default=int),flush=True)


if __name__=='__main__':main()
