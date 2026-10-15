"""Threshold-free retrospective joint profiles with strong aggregate controls."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from revision.common import ROOT,frozen_loadings,write_json,digest
from revision.calibration import correct
from revision.counts import group_data,parameters,sample_dm,category_counts
from revision.reconcile import reconcile_trajectories
from revision.joint_profile import profile_scores
from otg.residuals import aggregate_inputs
from otg.torch_scenarios import projected_counts_torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--draws',type=int,default=512);a=ap.parse_args();torch.set_num_threads(3);start=time.monotonic()
    out=ROOT/'runs'/f'revision_{a.dataset}_joint_profiles_v1';out.mkdir(exist_ok=False)
    base=ROOT/'runs'/f'revision_{a.dataset}_direct_counts_v2';chosen=json.loads((base/'manifest.json').read_text())['selected_on_development'];fit=json.loads((base/'fits.json').read_text())[chosen]
    data=ROOT/'data/processed'/f'{a.dataset}_joint_selected_v2';meta=json.loads((data/'metadata.json').read_text());names=meta['relations'];unit='m' if a.dataset=='borg' else 'd';rows=[]
    calibration={}
    if a.dataset=='bpi':calibration={v['cutoff_us']:v for v in json.loads((ROOT/'runs/revision_bpi_calibration_same_version_v2/calibrator_fits.json').read_text())['fits']}
    for period,sub in [('development','cohorts'),('retrospective_final','final_cohorts')]:
        directory=data/sub;cm=json.loads((directory/'manifest.json').read_text())
        for oi,c in enumerate(cm['cohorts']):
            f=directory/c['folder'];q=pd.read_parquet(f/'queries.parquet');p=np.load(f/'cdf.npy');x=np.load(f/'features.npy');y=q[[f'exit_{h}{unit}' for h in meta['horizons']]].to_numpy();rel=q[names].to_numpy(np.int64);o={'q':q,'p':p,'y':y,'x':x};proj,gs=group_data(o,names);_,ref=aggregate_inputs(p,q.state.to_numpy(),rel,names)
            def score(counts,mode,seed):
                for name,z in gs.items():
                    if name=='population_by_state':continue
                    rr=profile_scores(counts[name],z['truth'],z['n'],ref[name]['mean'],ref[name]['scale'],seed=oi+9000)
                    rows.extend([{**r,'period':period,'cutoff_us':c['cutoff_us'],'mode':mode,'seed':seed,'projection':name} for r in rr])
            for seed in range(3):
                configs=frozen_loadings(o,seed,a.dataset)
                for mode,w in configs.items():
                    counts=projected_counts_torch(p,w,rel,proj,a.draws,1030000+seed*10000+oi,'gaussian',a.device);score(counts,mode,seed)
                if a.dataset=='bpi':
                    cf=calibration[c['cutoff_us']]
                    for variant,b in [('static_calibration',cf['static_intercept']),('rolling_calibration',cf['rolling_intercept'])]:
                        for mode in ['independent','global','static']:
                            counts=projected_counts_torch(correct(p,b),configs[mode],rel,proj,a.draws,1030000+seed*10000+oi,'gaussian',a.device);score(counts,variant+'_'+mode,seed)
                rng=np.random.default_rng(1040000+seed*10000+oi);counts={name:sample_dm(z['n'],parameters(z['prob'],fit[name]),a.draws,rng)[:,:,:4].cumsum(-1) for name,z in gs.items()};score(counts,'direct_selected_dm',seed)
                if a.dataset=='bpi':
                    r=np.where(rel[:,0]<0,-1,rel[:,0]);values,code=np.unique(r,return_inverse=True);sizes=np.bincount(code);mean=np.column_stack([np.bincount(code,weights=p[:,h]) for h in range(4)]);prob=np.maximum(category_counts(mean,sizes)/sizes[:,None],1e-6);prob/=prob.sum(-1,keepdims=True);eligible=(values>=0)&(sizes>=5)
                    bottom=sample_dm(sizes,parameters(prob,fit['resource']),8192,rng);reconc,_=reconcile_trajectories(bottom,parameters(gs['population_by_state']['prob'],fit['population_by_state'])[0],a.draws,rng)
                    for mode,b in [('bottom_up_dm',bottom[:a.draws]),('conditioned_dm',reconc)]:
                        v=b[:,:,:4].cumsum(-1);score({'resource':v[:,eligible,:]},mode,seed)
            if (oi+1)%4==0:print(json.dumps({'period':period,'origins':oi+1,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'scores.parquet',index=False);df.groupby(['period','mode','projection']).mean(numeric_only=True).to_csv(out/'summary.csv')
    write_json(out/'manifest.json',{'status':'complete','config':vars(a),'script_sha256':digest(__file__),'profile_module_sha256':digest(ROOT/'revision/joint_profile.py'),'direct_model_selected':chosen,'interpretation':'Retrospective threshold-free joint scores. Reference peak-slowdown centering/scaling fixed to original individual forecasts for every method. BPI corrected marginals retain original factor weights as a controlled diagnostic. Direct group trajectories need not agree across overlapping projections.','elapsed_s':time.monotonic()-start})
    print(df.groupby(['period','mode'])[['joint_profile_energy','joint_variogram_fair','peak_slowdown_crps']].mean().to_string(),flush=True)


if __name__=='__main__':main()
