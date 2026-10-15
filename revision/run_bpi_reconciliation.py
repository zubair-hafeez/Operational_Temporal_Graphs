"""Full handler-to-population trajectory reconciliation on BPI (retrospective)."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from revision.common import ROOT,bpi_origins,write_json,digest
from revision.counts import group_data,category_counts,parameters,sample_dm
from revision.reconcile import reconcile_trajectories
from otg.cohort_scores import score_projected_counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--draws',type=int,default=512);ap.add_argument('--pool',type=int,default=8192);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);a=ap.parse_args()
    out=ROOT/'runs'/f'revision_bpi_reconciliation_v1_pool{a.pool}';out.mkdir(exist_ok=False);start=time.monotonic()
    base=ROOT/'runs/revision_bpi_direct_counts_v1';selected=json.loads((base/'manifest.json').read_text())['selected_on_development'];fit=json.loads((base/'fits.json').read_text())[selected]
    rows=[];diagnostics=[]
    for oi,o in enumerate(bpi_origins()):
        if o['period']=='train':continue
        proj,gs=group_data(o,['resource']);r=o['q'].resource.to_numpy();r=np.where(r<0,-1,r);values,code=np.unique(r,return_inverse=True);sizes=np.bincount(code)
        mean=np.column_stack([np.bincount(code,weights=o['p'][:,h]) for h in range(4)])
        prob=np.maximum(category_counts(mean,sizes)/sizes[:,None],1e-6);prob/=prob.sum(-1,keepdims=True)
        alpha=parameters(prob,fit['resource']);upper=parameters(gs['population_by_state']['prob'],fit['population_by_state'])[0]
        eligible=(values>=0)&(sizes>=5);assert np.array_equal(sizes[eligible],proj['resource']['size'])
        for seed in a.seeds:
            rng=np.random.default_rng(950000+seed*10000+oi);bottom=sample_dm(sizes,alpha,a.pool,rng)
            reconc,diag=reconcile_trajectories(bottom,upper,a.draws,rng);diagnostics.append({**diag,'cutoff_us':o['t'],'period':o['period'],'seed':seed})
            for mode,b in [('bottom_up_dm',bottom[:a.draws]),('conditioned_dm',reconc)]:
                cumulative=b[:,:,:4].cumsum(-1)
                counts={'population_by_state':cumulative.sum(1)[:,None,:],'resource':cumulative[:,eligible,:]}
                rs=score_projected_counts(proj,counts,{name:z['truth'] for name,z in gs.items()})
                rows.extend([{**v,'mode':mode,'cutoff_us':o['t'],'period':o['period'],'seed':seed} for v in rs])
        if len(diagnostics)%30==0:print(json.dumps({'origin':oi,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'scores.parquet',index=False);df.groupby(['period','mode']).mean(numeric_only=True).to_csv(out/'macro.csv');pd.DataFrame(diagnostics).to_csv(out/'importance_diagnostics.csv',index=False)
    write_json(out/'manifest.json',{'status':'complete','config':vars(a),'base_model':selected,'method':'Two-level discrete probabilistic reconciliation by conditioning; joint five-category first-exit vectors; independently sampled bottom handler distributions, reweighted by population Dirichlet-multinomial mass.','citation':'https://doi.org/10.1007/s11222-023-10343-y','adaptation':'Python implementation of a single hierarchical importance step, using multivariate count-vector forecasts. All handler groups including small/unknown groups contribute to totals; score original eligible groups. Does not preserve individual CDFs.','interpretation':'Retrospective follow-up, not untouched confirmation.','script_sha256':digest(__file__),'reconciliation_sha256':digest(ROOT/'revision/reconcile.py'),'base_fit_sha256':digest(base/'fits.json'),'scores_sha256':digest(out/'scores.parquet'),'elapsed_s':time.monotonic()-start})
    print(df.groupby(['period','mode'])[['fraction_crps','coverage_90','buffer_cost_ratio9']].mean().to_string(),flush=True);print(pd.DataFrame(diagnostics).ess_fraction.describe().to_string(),flush=True)


if __name__=='__main__':main()
