"""Direct count baselines trained on complete out-of-fold active cohorts."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from revision.common import ROOT,bpi_origins,write_json,digest
from revision.counts import group_data,fit_dm,parameters,sample_dm
from otg.cohort_scores import score_projected_counts
from otg.protocol import verify_final_protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['bpi','borg'],default='bpi');ap.add_argument('--draws',type=int,default=512);a=ap.parse_args();threadpool_limits(limits=2)
    out=ROOT/'runs'/f'revision_{a.dataset}_direct_counts_v1';out.mkdir(exist_ok=False);start=time.monotonic();verify_final_protocol(ROOT)
    if a.dataset=='bpi':origins=bpi_origins();relations=['resource']
    else:
        from revision.borg_data import borg_origins
        origins=borg_origins();relations=['job','machine']
    train={};evals=[]
    for o in origins:
        proj,gs=group_data(o,relations)
        if o['period']=='train':
            for name,z in gs.items():train.setdefault(name,[]).append(z)
        else:evals.append((o['t'],o['period'],proj,gs))
    fits={mode:{name:fit_dm(records,bias=(mode=='bias_dm')) for name,records in train.items()} for mode in ['dispersion_dm','bias_dm']}
    write_json(out/'fits.json',fits);rows=[]
    for oi,(t,period,proj,gs) in enumerate(evals):
        for mode,fit in fits.items():
            for seed in range(3):
                rng=np.random.default_rng(930000+seed*10000+oi);counts={name:sample_dm(z['n'],parameters(z['prob'],fit[name]),a.draws,rng)[:,:,:4].cumsum(-1) for name,z in gs.items()}
                scores=score_projected_counts(proj,counts,{name:z['truth'] for name,z in gs.items()})
                rows.extend([{**r,'mode':mode,'seed':seed,'cutoff_us':t,'period':period} for r in scores])
        if (oi+1)%10==0:print(json.dumps({'origins':oi+1,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'scores.parquet',index=False)
    df.groupby(['period','mode']).mean(numeric_only=True).to_csv(out/'macro.csv')
    dev=df[df.period=='development'].groupby('mode').fraction_crps.mean();selected=dev.idxmin()
    write_json(out/'manifest.json',{'status':'complete','config':vars(a),'selected_on_development':selected,'development_crps':dev.to_dict(),'interpretation':'Retrospective follow-up to an already opened final period. Direct group counts are separately coherent trajectories; overlapping projections are not jointly reconciled.','training_origins':[o['t'] for o in origins if o['period']=='train'],'script_sha256':digest(__file__),'module_sha256':digest(ROOT/'revision/counts.py'),'scores_sha256':digest(out/'scores.parquet'),'elapsed_s':time.monotonic()-start})
    print(df.groupby(['period','mode'])[['fraction_crps','coverage_90','buffer_cost_ratio9']].mean().to_string(),flush=True)


if __name__=='__main__':main()
