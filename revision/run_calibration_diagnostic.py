"""Retrospective controlled BPI marginal-bias diagnosis. CPU is sufficient."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from revision.common import ROOT,bpi_origins,frozen_loadings,write_json,digest
from revision.calibration import correct,fit_intercept,eligible_history,DAY_US
from otg.cohort_scores import cohort_scores
from otg.protocol import verify_final_protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,default=512);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a=ap.parse_args()
    out=ROOT/'runs/revision_bpi_calibration_v1';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);threadpool_limits(limits=2)
    _,protocol_hash=verify_final_protocol(ROOT);origins=bpi_origins();start=time.monotonic()
    train=[o for o in origins if o['period']=='train'];static=fit_intercept(train)
    scores=[];marginals=[];fits=[]
    for oi,o in enumerate(origins):
        if o['period']=='train':continue
        eligible=eligible_history(origins,o['t']);rolling=fit_intercept(eligible)
        assert all(v['t']+14*DAY_US<=o['t'] for v in eligible)
        fits.append({'cutoff_us':o['t'],'period':o['period'],'static_intercept':static,'rolling_intercept':rolling,'rolling_fit_origins':[v['t'] for v in eligible]})
        q=o['q'];rel=q[['resource']].to_numpy(np.int64);st=q.state.to_numpy();configs={s:frozen_loadings(o,s) for s in a.seeds}
        for cal,b in [('identity',0),('static',static),('rolling60',rolling)]:
            p=o['p'] if cal=='identity' else correct(o['p'],b)
            for h in range(4):
                marginals.append({'cutoff_us':o['t'],'period':o['period'],'calibration':cal,'horizon':[1,3,7,14][h],'brier':float(np.mean((p[:,h]-o['y'][:,h])**2)),'predicted':float(p[:,h].mean()),'observed':float(o['y'][:,h].mean()),'intercept':b})
            for seed in a.seeds:
                for mode,load in configs[seed].items():
                    rs=cohort_scores(p,load,rel,st,o['y'],['resource'],a.draws,910000+seed*10000+oi)
                    scores.extend([{**r,'cutoff_us':o['t'],'period':o['period'],'calibration':cal,'mode':mode,'seed':seed} for r in rs])
        if len(fits)%5==0:print(json.dumps({'origins':len(fits),'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(scores);df.to_parquet(out/'scores.parquet',index=False)
    mg=pd.DataFrame(marginals);mg.to_csv(out/'marginals.csv',index=False)
    df.groupby(['period','calibration','mode']).mean(numeric_only=True).to_csv(out/'macro.csv')
    mg.groupby(['period','calibration','horizon']).mean(numeric_only=True).to_csv(out/'marginal_summary.csv')
    write_json(out/'calibrator_fits.json',{'static_fit_origins':[o['t'] for o in train],'fits':fits})
    verify_final_protocol(ROOT)
    write_json(out/'manifest.json',{'status':'complete','interpretation':'Retrospective diagnostic; original final results already known. Dependence weights frozen, not refitted after marginal correction.','config':vars(a),'original_protocol_sha256':protocol_hash,'script_sha256':digest(__file__),'module_hashes':{n:digest(ROOT/'revision'/n) for n in ['common.py','calibration.py']},'scores_sha256':digest(out/'scores.parquet'),'elapsed_s':time.monotonic()-start})
    print(df.groupby(['period','calibration','mode'])[['fraction_crps','coverage_90','buffer_cost_ratio9']].mean().to_string(),flush=True)


if __name__=='__main__':main()
