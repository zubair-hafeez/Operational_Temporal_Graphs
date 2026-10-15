"""Select fresh-replication individual models using January development only."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from revision.common import ROOT,write_json,digest
from revision.calibration import fit_intercept,correct,eligible_history,DAY_US


def boundary(s):return int(pd.Timestamp(s,tz='UTC').value//1000)


def origin_records(q,p,ids):
    part=q.iloc[ids];result=[]
    for t,ix in part.groupby('cutoff_us').groups.items():
        ix=np.asarray(ix);result.append({'t':int(t),'p':p[ix],'y':q.iloc[ix][['exit_1d','exit_3d','exit_7d','exit_14d']].to_numpy(np.int8)})
    return result


def main():
    threadpool_limits(limits=4);start=time.monotonic();out=ROOT/'runs/revision_bpi2012_marginals_v1';out.mkdir(exist_ok=False)
    source=ROOT/'data/processed/revision_bpi2012_development_v1';m=json.loads((source/'metadata.json').read_text());q=pd.read_parquet(source/'queries.parquet');x=np.load(source/'features.npy');y=q[['exit_1d','exit_3d','exit_7d','exit_14d']].to_numpy()
    t=q.cutoff_us.to_numpy();fit=np.flatnonzero(t+14*DAY_US<=boundary('2011-11-01'));cal=np.flatnonzero((t>=boundary('2011-11-01'))&(t+14*DAY_US<=boundary('2011-12-01')));dev=np.flatnonzero((t>=boundary('2012-01-01'))&(t+14*DAY_US<=boundary('2012-02-01')))
    weights=1/q.groupby('cutoff_us').cutoff_us.transform('size').to_numpy();results=[];predictions={};intercepts={}
    for arm in ['personal','global','relational']:
        names=m['feature_groups']['personal']+(m['feature_groups']['global'] if arm!='personal' else [])+(m['feature_groups']['relational'] if arm=='relational' else []);cols=[m['features'].index(n) for n in names];ps=[]
        for hi,h in enumerate([1,3,7,14]):
            model=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,min_samples_leaf=50,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=0)
            model.fit(x[fit][:,cols],y[fit,hi],sample_weight=weights[fit]/weights[fit].mean());ps.append(model.predict_proba(x[:,cols])[:,1]);joblib.dump({'model':model,'cols':cols},out/f'{arm}_{h}d.joblib')
        raw=np.sort(np.column_stack(ps),axis=1).astype(np.float32);b=fit_intercept(origin_records(q,raw,cal));intercepts[arm]=b
        for variant,p in [('identity',raw),('static',correct(raw,b))]:
            predictions[(arm,variant)]=p;score=float(np.average(((p[dev]-y[dev])**2).mean(1),weights=weights[dev]));results.append({'arm':arm,'calibration':variant,'development_ibrier':score})
        np.save(out/f'{arm}_raw_cdf.npy',raw)
        print(json.dumps({'arm':arm,'results':results[-2:],'elapsed_s':time.monotonic()-start}),flush=True)
    selected=min(results,key=lambda r:r['development_ibrier']);arm=selected['arm'];raw=predictions[(arm,'identity')]
    # Full post-fit origins, including gaps between labeled fitting windows,
    # provide delayed online updates from one predictor version only.
    history=origin_records(q,raw,np.flatnonzero(t>=boundary('2011-11-01')));online=raw.copy();fits=[]
    for o in history:
        if o['t']<boundary('2011-12-01'):continue
        past=eligible_history(history,o['t']);b=fit_intercept(past);mask=t==o['t'];online[mask]=correct(raw[mask],b);fits.append({'cutoff_us':o['t'],'intercept':b,'fit_origins':[v['t'] for v in past]})
    for name,p in [('identity',raw),('static',predictions[(arm,'static')]),('rolling60',online)]:np.save(out/f'selected_{name}_cdf.npy',p)
    write_json(out/'online_fits.json',fits)
    write_json(out/'manifest.json',{'status':'complete','selected':selected,'candidates':results,'static_intercepts':intercepts,'fit_queries':len(fit),'calibration_queries':len(cal),'development_queries':len(dev),'fit_origin_range':[int(t[fit].min()),int(t[fit].max())],'development_origin_range':[int(t[dev].min()),int(t[dev].max())],'selection':'Choose feature arm and identity/static correction by equal-origin January integrated Brier. Online correction is a prespecified ablation, not used for arm selection.','model':'HGB 250 iterations, 15 leaves, minimum leaf50, lr.05, L2=1, no random holdout/early stopping. Three feature arms.','source_metadata_sha256':digest(source/'metadata.json'),'script_sha256':digest(__file__),'elapsed_s':time.monotonic()-start})
    print(json.dumps({'selected':selected,'elapsed_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
