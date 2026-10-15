"""Complete six-hour Borg OOF cohorts using existing earlier-fold predictors."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from otg.borg import BorgReplay,ORIGIN_US,DAY_US
from scripts.build_borg_cohorts import label
from revision.common import ROOT,write_json,digest


def main():
    threadpool_limits(limits=4);start=time.monotonic()
    src=ROOT/'data/processed/borg_prefix_22d';m=json.loads((src/'metadata.json').read_text())
    out=ROOT/'data/processed/revision_borg_oof_full_v1';out.mkdir(exist_ok=False)
    selections=json.loads((ROOT/'data/processed/borg_joint_selected_v2/metadata.json').read_text())['selected_marginals']
    models={}
    for fold in ['oof8_10','oof12_14','oof16_18']:
        models[fold]=[]
        for c in selections:
            assert c['kind']=='hgb' and c['tag']=='v1'
            base=ROOT/'runs'/f'borg_marginals_v1_state{c["state"]}_{c["arm"]}_seed0'/fold
            models[fold].append((c,[joblib.load(base/f'horizon{h}m.joblib') for h in [1,5,15,60]],json.loads((base/'calibration_audit.json').read_text())))
    e=pd.read_parquet(src/'events.parquet');intervals=pd.read_parquet(src/'intervals.parquet')
    replay=BorgReplay(m['tasks'],m['jobs'],m['machines'],m['unknown_machine']);times=e.time_us.to_numpy();lo=0;cohorts=[]
    for first,fold in [(8,'oof8_10'),(12,'oof12_14'),(16,'oof16_18')]:
        for hour in range(first*24,(first+2)*24,6):
            t=int(ORIGIN_US+hour*3600_000_000);hi=int(np.searchsorted(times,t,side='right'));replay.apply(e.iloc[lo:hi]);lo=hi
            ids=np.flatnonzero(replay.state>0).astype(np.int32);x=replay.features(ids,t);machine=replay.machine[ids].astype(np.int64).copy();missing=(replay.state[ids]!=2)|(machine==replay.unknown_machine);machine[missing]=-1-ids[missing].astype(np.int64)
            q=pd.DataFrame({'query_id':np.arange(len(ids),dtype=np.int64),'task':ids,'cutoff_us':t,'elapsed_day':hour/24,'state':replay.state[ids],'job':replay.job[ids],'machine':machine,'last_source_row':replay.last_source[ids],'sampling_weight':1.})
            q=label(q,intervals,int(ORIGIN_US+(first+2)*DAY_US));assert q.observed.all()
            p=np.zeros((len(ids),4),np.float32)
            for c,fitted,audit in models[fold]:
                mask=q.state.to_numpy()==c['state'];raw=[];individual=[]
                for model in fitted:
                    pr=np.clip(model['model'].predict_proba(x[mask][:,model['cols']])[:,1],1e-5,1-1e-5);raw.append(pr);individual.append(model['calibrator'].predict_proba(logit(pr)[:,None])[:,1])
                raw=np.sort(np.column_stack(raw),axis=1)
                if c['variant']=='individual_platt_sorted':pred=np.sort(np.column_stack(individual),axis=1)
                elif c['variant']=='raw_sorted':pred=raw
                else:
                    a,b=audit['params'][c['variant']];pred=expit(a*logit(np.clip(raw,1e-6,1-1e-6))+b)
                p[mask]=pred
            folder=out/f'cohort_hour{hour:04d}';folder.mkdir();q.to_parquet(folder/'queries.parquet',index=False);np.save(folder/'cdf.npy',p);np.save(folder/'base_features.npy',x)
            cohorts.append({'folder':folder.name,'cutoff_us':t,'n':len(q),'forward_fold':fold,'queries_sha256':digest(folder/'queries.parquet'),'cdf_sha256':digest(folder/'cdf.npy')})
            print(json.dumps({'cohorts':len(cohorts),'hour':hour,'n':len(q),'elapsed_s':time.monotonic()-start}),flush=True)
    write_json(out/'manifest.json',{'cohorts':cohorts,'role':'Full population OOF cohorts for retrospective aggregate baseline training; existing fold models reused without refit.','sampling':'Every six hours in original three OOF windows; full active population.','script_sha256':digest(__file__),'elapsed_s':time.monotonic()-start})


if __name__=='__main__':main()
