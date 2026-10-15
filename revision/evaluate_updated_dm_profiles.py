"""Joint-risk follow-up for development-fitted direct/reconciled count controls."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from revision.common import ROOT,bpi_origins,write_json,digest
from revision.borg_data import borg_origins
from revision.counts import group_data,parameters,sample_dm,category_counts
from revision.reconcile import reconcile_trajectories
from revision.joint_profile import profile_scores
from otg.residuals import aggregate_inputs
from otg.cohort_scores import score_projected_counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['borg','bpi']);a=ap.parse_args();start=time.monotonic();out=ROOT/'runs'/f'revision_{a.dataset}_updated_dm_profiles_v1';out.mkdir(exist_ok=False)
    base=ROOT/'runs'/f'revision_{a.dataset}_development_fitted_dm_v1';fit=json.loads((base/'fits.json').read_text());origins=bpi_origins() if a.dataset=='bpi' else borg_origins();names=['resource'] if a.dataset=='bpi' else ['job','machine'];rows=[];countscores=[];diagnostics=[]
    for oi,o in enumerate(origins):
        if o['period']!='retrospective_final':continue
        proj,gs=group_data(o,names);q=o['q'];p=o['p'];rel=q[names].to_numpy();_,ref=aggregate_inputs(p,q.state.to_numpy(),rel,names)
        def score(counts,mode,seed):
            base={'cutoff_us':o['t'],'mode':mode,'seed':seed}
            countscores.extend([{**v,**base} for v in score_projected_counts(proj,counts,{n:z['truth'] for n,z in gs.items()})])
            for name,z in gs.items():
                if name=='population_by_state':continue
                rr=profile_scores(counts[name],z['truth'],z['n'],ref[name]['mean'],ref[name]['scale'],seed=oi+9000);rows.extend([{**v,**base,'projection':name} for v in rr])
        for seed in range(3):
            rng=np.random.default_rng(1060000+seed*10000+oi);counts={name:sample_dm(z['n'],parameters(z['prob'],fit[name]),512,rng)[:,:,:4].cumsum(-1) for name,z in gs.items()};score(counts,'development_fitted_dm',seed)
            if a.dataset=='bpi':
                r=np.where(rel[:,0]<0,-1,rel[:,0]);values,code=np.unique(r,return_inverse=True);sizes=np.bincount(code);mean=np.column_stack([np.bincount(code,weights=p[:,h]) for h in range(4)]);prob=np.maximum(category_counts(mean,sizes)/sizes[:,None],1e-6);prob/=prob.sum(-1,keepdims=True);eligible=(values>=0)&(sizes>=5)
                bottom=sample_dm(sizes,parameters(prob,fit['resource']),8192,rng);reconc,diag=reconcile_trajectories(bottom,parameters(gs['population_by_state']['prob'],fit['population_by_state'])[0],512,rng);diagnostics.append({**diag,'cutoff_us':o['t'],'seed':seed})
                for mode,b in [('development_bottom_up_dm',bottom[:512]),('development_conditioned_dm',reconc)]:
                    v=b[:,:,:4].cumsum(-1);score({'population_by_state':v.sum(1)[:,None,:],'resource':v[:,eligible,:]},mode,seed)
        if (oi+1)%5==0:print(json.dumps({'origin_index':oi+1,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'scores.parquet',index=False);df.groupby(['mode','projection']).mean(numeric_only=True).to_csv(out/'summary.csv');pd.DataFrame(countscores).to_parquet(out/'count_scores.parquet',index=False)
    if diagnostics:pd.DataFrame(diagnostics).to_csv(out/'importance_diagnostics.csv',index=False)
    write_json(out/'manifest.json',{'status':'complete','config':vars(a),'script_sha256':digest(__file__),'fit_sha256':digest(base/'fits.json'),'interpretation':'Retrospective stronger baseline fitted to the current-version development forecasts; original inference targets unchanged.','elapsed_s':time.monotonic()-start})
    print(df.groupby('mode')[['joint_profile_energy','joint_variogram_fair','peak_slowdown_crps']].mean().to_string(),flush=True)


if __name__=='__main__':main()
