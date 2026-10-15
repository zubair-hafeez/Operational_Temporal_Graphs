"""Strong direct count control fitted on the same period as historical residuals.

Retrospective only: use full development for the final forecast, selecting
concentration-only versus biased DM by leave-one-origin-out exact CRPS.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision.common import ROOT,bpi_origins,write_json,digest
from revision.borg_data import borg_origins
from revision.counts import group_data,fit_dm,parameters
from revision.exact_counts import exact_projected_scores


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['borg','bpi']);a=ap.parse_args();threadpool_limits(limits=2);start=time.monotonic();out=ROOT/'runs'/f'revision_{a.dataset}_development_fitted_dm_v1';out.mkdir(exist_ok=False)
    origins=bpi_origins() if a.dataset=='bpi' else borg_origins();names=['resource'] if a.dataset=='bpi' else ['job','machine'];dev=[];final=[]
    for o in origins:
        if o['period']=='train':continue
        _,gs=group_data(o,names);(dev if o['period']=='development' else final).append((o['t'],gs))
    def fit(records,bias):return {name:fit_dm([gs[name] for _,gs in records],bias) for name in names+['population_by_state']}
    rows=[]
    for i,(t,gs) in enumerate(dev):
        for mode,bias in [('dispersion_dm',False),('bias_dm',True)]:
            fits=fit([o for j,o in enumerate(dev) if j!=i],bias);rs=exact_projected_scores(gs,{n:parameters(z['prob'],fits[n]) for n,z in gs.items()});rows.extend([{**r,'mode':mode,'cutoff_us':t,'period':'development_loo'} for r in rs])
        print(json.dumps({'loo_origin':i+1,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);selected=df.groupby('mode').fraction_crps.mean().idxmin();fitted=fit(dev,selected=='bias_dm');write_json(out/'fits.json',fitted)
    for i,(t,gs) in enumerate(final):
        rs=exact_projected_scores(gs,{n:parameters(z['prob'],fitted[n]) for n,z in gs.items()});rows.extend([{**r,'mode':selected,'cutoff_us':t,'period':'retrospective_final'} for r in rs])
        if (i+1)%5==0:print(json.dumps({'final_origin':i+1,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'scores.parquet',index=False);df.groupby(['period','mode']).mean(numeric_only=True).to_csv(out/'macro.csv')
    write_json(out/'manifest.json',{'status':'complete','dataset':a.dataset,'selected_on_loo_development':selected,'training_origins':[t for t,_ in dev],'selection':'LOO development exact count CRPS; full development fits used at final. Future development origins may enter a LOO fit, so these are model-selection scores, not online validation.','scoring':'Exact beta-binomial cumulative-count marginals; no Monte Carlo seeds. Quantiles are exact discrete population quantiles, rather than finite-ensemble estimates.','interpretation':'Retrospective strong control. Same forecast version and fitting window as original historical-residual comparator; complements earlier OOF-fitted DM. No refitting on final labels.','script_sha256':digest(__file__),'score_module_sha256':digest(ROOT/'revision/exact_counts.py'),'elapsed_s':time.monotonic()-start})
    print(df.groupby(['period','mode'])[['fraction_crps','coverage_90','buffer_cost_ratio9']].mean().to_string(),flush=True)


if __name__=='__main__':main()
