"""Frozen fresh replication inference with auditable delayed online updates."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision.common import ROOT,write_json,digest
from revision.calibration import correct,fit_intercept,eligible_history,DAY_US
from revision.fit_bpi2012_marginals import origin_records,boundary
from revision.fresh_protocol import verify_fresh_protocol


def main():
    verify_fresh_protocol();threadpool_limits(limits=4);start=time.monotonic();out=ROOT/'runs/revision_bpi2012_final_predictions_v1';out.mkdir(exist_ok=False)
    source=ROOT/'data/processed/revision_bpi2012_final_v1';marg=ROOT/'runs/revision_bpi2012_marginals_v1';m=json.loads((marg/'manifest.json').read_text());arm=m['selected']['arm'];q=pd.read_parquet(source/'queries.parquet');x=np.load(source/'features.npy');ps=[]
    for h in [1,3,7,14]:
        model=joblib.load(marg/f'{arm}_{h}d.joblib');ps.append(model['model'].predict_proba(x[:,model['cols']])[:,1])
    raw=np.sort(np.column_stack(ps),axis=1).astype(np.float32);static=correct(raw,m['static_intercepts'][arm])
    oldq=pd.read_parquet(ROOT/'data/processed/revision_bpi2012_development_v1/queries.parquet');oldp=np.load(marg/f'{arm}_raw_cdf.npy');history=origin_records(oldq,oldp,np.flatnonzero(oldq.cutoff_us.to_numpy()>=boundary('2011-11-01')))+origin_records(q,raw,np.arange(len(q)))
    times=[o['t'] for o in history];assert len(times)==len(set(times)), 'History origins overlap'
    online=raw.copy();fits=[]
    for o in history:
        ids=np.flatnonzero(q.cutoff_us.to_numpy()==o['t'])
        if not len(ids):continue
        past=eligible_history(history,o['t']);assert all(v['t']+14*DAY_US<=o['t'] for v in past);b=fit_intercept(past);online[ids]=correct(raw[ids],b);fits.append({'cutoff_us':o['t'],'intercept':b,'fit_origins':[v['t'] for v in past]})
    for name,p in [('identity',raw),('static',static),('rolling60',online)]:np.save(out/f'selected_{name}_cdf.npy',p)
    write_json(out/'online_fits.json',fits);write_json(out/'manifest.json',{'status':'complete','script_sha256':digest(__file__),'marginal_manifest_sha256':digest(marg/'manifest.json'),'source_metadata_sha256':digest(source/'metadata.json'),'interpretation':'Frozen HGB inference, fixed November calibration, and delayed same-version rolling correction. January19–31 origins supply history only; February1–28 origins are evaluated.','elapsed_s':time.monotonic()-start})
    print(json.dumps({'queries':len(q),'origins':q.cutoff_us.nunique(),'elapsed_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
