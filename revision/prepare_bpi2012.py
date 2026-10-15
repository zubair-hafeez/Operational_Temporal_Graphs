"""Causal BPI 2012 development replay; February outcomes remain gated."""
import argparse,gzip,json,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from otg.bpi import BPIReplay,RESOLUTION,DAY_US
from revision.common import ROOT,write_json,digest

ACTIVITY_MAP={'A_SUBMITTED':'A_Create Application','A_APPROVED':'A_Pending','A_DECLINED':'A_Denied','A_CANCELLED':'A_Cancelled'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--period',choices=['development','final'],default='development');a=ap.parse_args();start=time.monotonic()
    if a.period=='final':
        from revision.fresh_protocol import verify_fresh_protocol
        verify_fresh_protocol()
    before='2012-02-01' if a.period=='development' else '2012-03-14';end=int(pd.Timestamp(before,tz='UTC').value//1000)
    out=ROOT/'data/processed'/f'revision_bpi2012_{a.period}_v1';out.mkdir(exist_ok=False)
    source=ROOT/'data/raw/revision_bpi2012/bpi2012.xes.gz';rows=[];source_row=0
    with gzip.open(source,'rb') as f:
        for _,elem in ET.iterparse(f,events=['end']):
            if elem.tag.split('}')[-1]!='trace':continue
            case=next(c.get('value') for c in elem if c.get('key')=='concept:name')
            for child in elem:
                if child.tag.split('}')[-1]!='event':continue
                source_row+=1;stamp=next(c.get('value') for c in child if c.get('key')=='time:timestamp');t=int(pd.Timestamp(stamp).value//1000)
                if t>=end:continue  # no future activity/outcome is retained or summarized
                v={c.get('key'):c.get('value') for c in child};name=v['concept:name']
                rows.append((case,t,ACTIVITY_MAP.get(name,name),v.get('lifecycle:transition','').lower(),v.get('org:resource'),source_row))
            elem.clear()
    e=pd.DataFrame(rows,columns=['case_id','time_us','activity','lifecycle','resource_id','source_row']).sort_values(['time_us','source_row'],kind='stable').reset_index(drop=True)
    e['case'],cases=pd.factorize(e.case_id,sort=False);e['case']=e['case'].astype(np.int32)
    resources=list(pd.unique(e.resource_id.dropna()));rmap={v:i for i,v in enumerate(resources)};e['resource']=e.resource_id.map(rmap).fillna(len(resources)).astype(np.int16)
    e['offer_id']=None;e['offer']=-1
    fit_end=int(pd.Timestamp('2011-11-01',tz='UTC').value//1000)
    if a.period=='development':activities=sorted(e.loc[e.time_us<fit_end,'activity'].unique())
    else:activities=json.loads((ROOT/'data/processed/revision_bpi2012_development_v1/metadata.json').read_text())['activity_vocabulary']
    e.to_parquet(out/'events.parquet',index=False)
    resolution=e[e.activity.isin(RESOLUTION)].drop_duplicates('case').set_index('case').time_us
    resolution_us=np.full(len(cases),end,np.int64);observed=np.zeros(len(cases),bool)
    resolution_us[resolution.index]=resolution.to_numpy();observed[resolution.index]=True
    replay=BPIReplay(len(cases),len(resources),activities);times=e.time_us.to_numpy();lo=0;records=[];features=[];qid=0
    first=pd.Timestamp('2011-10-02T12:00:00Z');last=pd.Timestamp(before,tz='UTC')-pd.Timedelta(days=14)
    for stamp in pd.date_range(first,last,freq='D'):
        t=int(stamp.value//1000);hi=int(np.searchsorted(times,t,side='right'));replay.apply(e.iloc[lo:hi]);lo=hi
        # Retain cross-boundary origins solely for delayed calibration once
        # their labels mature; scoring still begins on February 1.
        if a.period=='final' and t<int(pd.Timestamp('2012-01-19',tz='UTC').value//1000):continue
        ids=np.flatnonzero((replay.created>=0)&~replay.resolved).astype(np.int32)
        if not len(ids):continue
        x,groups=replay.features(ids,t);features.append(x);resource=replay.resource[ids].astype(np.int64);resource[resource==len(resources)]=-1
        q=pd.DataFrame({'query_id':np.arange(qid,qid+len(ids)), 'case':ids,'cutoff_us':t,'state':1,'resource':resource,'sampling_weight':1.});qid+=len(ids)
        for h in [1,3,7,14]:q[f'exit_{h}d']=(observed[ids]&(resolution_us[ids]<=t+h*DAY_US)).astype(np.int8)
        assert np.all(resolution_us[ids]>t)
        records.append(q)
    q=pd.concat(records,ignore_index=True);q.to_parquet(out/'queries.parquet',index=False);np.save(out/'features.npy',np.concatenate(features))
    # Audit only retained period; February is unavailable during development.
    starts=e[e.activity=='A_Create Application'].drop_duplicates('case')
    audit={'events':len(e),'cases':len(cases),'created_cases':len(starts),'observed_resolutions':len(resolution),'resources':len(resources),'origins':q.cutoff_us.nunique(),'queries':len(q),'min_time_us':int(e.time_us.min()),'max_time_us':int(e.time_us.max()),'max_origin_us':int(q.cutoff_us.max())}
    write_json(out/'metadata.json',{'config':vars(a),'before':before,'target':'First documented A_APPROVED, A_DECLINED or A_CANCELLED after A_SUBMITTED; no repayment or final trace completion claim.','activity_aliases_for_reused_replay':ACTIVITY_MAP,'activity_vocabulary':activities,'features':sum(groups.values(),[]),'feature_groups':groups,'relations':['resource'],'horizons':[1,3,7,14],'horizon_unit':'days','excluded':'All retrospective trace attributes, including requested amount (not needed); future resource assignments and future activity vocabulary.','availability':'Recorded event timestamps; source does not establish real-time ingestion delay. Last observed workflow resource is a proxy for handling, not ownership/capacity.','audit':audit,'raw_sha256':digest(source),'queries_sha256':digest(out/'queries.parquet'),'features_sha256':digest(out/'features.npy'),'script_sha256':digest(__file__),'elapsed_s':time.monotonic()-start})
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
