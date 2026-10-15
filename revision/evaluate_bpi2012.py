"""Fresh replication evaluation with common counts and joint-profile scores."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from revision.common import ROOT,write_json,digest
from revision.bpi2012_data import origins_2012,loads_2012
from revision.counts import group_data,fit_dm,parameters,sample_dm,category_counts
from revision.reconcile import reconcile_trajectories
from revision.joint_profile import profile_scores
from otg.residuals import aggregate_inputs
from otg.cohort_scores import score_projected_counts
from otg.torch_scenarios import projected_counts_torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--period',choices=['development','final'],default='development');ap.add_argument('--draws',type=int,default=512);a=ap.parse_args();torch.set_num_threads(3);start=time.monotonic()
    if a.period=='final':
        from revision.fresh_protocol import verify_fresh_protocol
        verify_fresh_protocol()
    out=ROOT/'runs'/f'revision_bpi2012_{a.period}_evaluation_v1';out.mkdir(exist_ok=False)
    # Direct counts use the development-selected individual calibration.
    mm=json.loads((ROOT/'runs/revision_bpi2012_marginals_v1/manifest.json').read_text());basecal=mm['selected']['calibration']
    if a.period=='development':
        train={}
        for o in origins_2012('development',basecal):
            if o['period']!='train':continue
            _,gs=group_data(o,['resource'])
            for name,z in gs.items():train.setdefault(name,[]).append(z)
        fits={mode:{name:fit_dm(rs,bias=(mode=='bias_dm')) for name,rs in train.items()} for mode in ['dispersion_dm','bias_dm']};write_json(out/'direct_fits.json',fits)
    else:fits=json.loads((ROOT/'runs/revision_bpi2012_development_evaluation_v1/direct_fits.json').read_text())
    rows=[];joint=[];ess=[]
    def score(o,proj,gs,counts,mode,seed,calibration):
        base={'cutoff_us':o['t'],'mode':mode,'seed':seed,'calibration':calibration};truth={name:z['truth'] for name,z in gs.items()};rows.extend([{**r,**base} for r in score_projected_counts(proj,counts,truth)])
        _,ref=aggregate_inputs(o['raw_reference'],o['q'].state.to_numpy(),o['q'][['resource']].to_numpy(),['resource'])
        for name,z in gs.items():
            if name=='population_by_state':continue
            rr=profile_scores(counts[name],z['truth'],z['n'],ref[name]['mean'],ref[name]['scale'],seed=int(o['t']//86400_000_000));joint.extend([{**v,**base,'projection':name} for v in rr])
    for calibration in ['identity','static','rolling60']:
        for oi,o in enumerate(origins_2012(a.period,calibration)):
            if o['period']=='train':continue
            proj,gs=group_data(o,['resource']);rel=o['q'][['resource']].to_numpy(np.int64)
            for seed in range(3):
                configurations=loads_2012(o,seed,calibration)
                for mode,w in configurations.items():
                    counts=projected_counts_torch(o['p'],w,rel,proj,a.draws,1010000+seed*10000+oi,'gaussian','cpu');score(o,proj,gs,counts,mode,seed,calibration)
                if calibration!=basecal:continue
                for mode,fit in fits.items():
                    rng=np.random.default_rng(1020000+seed*10000+oi);counts={name:sample_dm(z['n'],parameters(z['prob'],fit[name]),a.draws,rng)[:,:,:4].cumsum(-1) for name,z in gs.items()};score(o,proj,gs,counts,mode,seed,calibration)
                    # Handler bottoms plus all small/unknown groups form a
                    # complete partition for the same population trajectory.
                    r=np.where(rel[:,0]<0,-1,rel[:,0]);values,code=np.unique(r,return_inverse=True);sizes=np.bincount(code);mean=np.column_stack([np.bincount(code,weights=o['p'][:,h]) for h in range(4)]);prob=np.maximum(category_counts(mean,sizes)/sizes[:,None],1e-6);prob/=prob.sum(-1,keepdims=True)
                    eligible=(values>=0)&(sizes>=5);alpha=parameters(prob,fit['resource']);upper=parameters(gs['population_by_state']['prob'],fit['population_by_state'])[0];bottom=sample_dm(sizes,alpha,8192,rng);reconc,diag=reconcile_trajectories(bottom,upper,a.draws,rng);ess.append({**diag,'cutoff_us':o['t'],'seed':seed,'base_mode':mode})
                    for how,b in [('bottom_up',bottom[:a.draws]),('conditioned',reconc)]:
                        c=b[:,:,:4].cumsum(-1);counts={'population_by_state':c.sum(1)[:,None,:],'resource':c[:,eligible,:]};score(o,proj,gs,counts,mode+'_'+how,seed,calibration)
            if oi%5==0:print(json.dumps({'calibration':calibration,'origin':oi,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'scores.parquet',index=False);pd.DataFrame(joint).to_parquet(out/'joint_profiles.parquet',index=False);pd.DataFrame(ess).to_csv(out/'importance_diagnostics.csv',index=False)
    df.groupby(['calibration','mode']).mean(numeric_only=True).to_csv(out/'macro.csv');pd.DataFrame(joint).groupby(['calibration','mode']).mean(numeric_only=True).to_csv(out/'joint_macro.csv')
    write_json(out/'manifest.json',{'status':'complete','config':vars(a),'base_calibration_for_direct_models':basecal,'script_sha256':digest(__file__),'scores_sha256':digest(out/'scores.parquet'),'joint_profile_module_sha256':digest(ROOT/'revision/joint_profile.py'),'elapsed_s':time.monotonic()-start})
    print(df.groupby(['calibration','mode'])[['fraction_crps','coverage_90','buffer_cost_ratio9']].mean().to_string(),flush=True)


if __name__=='__main__':main()
