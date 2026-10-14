"""Evaluate disjoint-fit pilot samplers; no final period or new fitting."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import ndtri
from otg.scores import ensemble_crps,interval_statistics
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--draws',type=int,default=1024)
    ap.add_argument('--family',choices=['gaussian','commonshock'],default='gaussian');a=ap.parse_args()
    tag='copula' if a.family=='gaussian' else 'commonshock'
    data=ROOT/'data/processed'/f'{a.dataset}_copula_pilot';run=ROOT/'runs'/f'learned_{tag}_{a.dataset}_seed{a.seed}'
    if (run/'scenario_scores.parquet').exists():raise FileExistsError('Preserve prior evaluation')
    q=pd.read_parquet(data/'queries.parquet');rows=[];rng=np.random.default_rng(4096+a.seed)
    fixed=[(0,0),(.05,0),(.2,0),(.5,0),(0,.2),(0,.5),(0,.8),(.05,.2),(.05,.5),(.2,.5)]
    for state in map(int,sorted(q.state.unique())):
        loadings={mode:np.load(run/f'{mode}_state{state}_loadings.npy') for mode in ['global','static','context']}
        for (split,t),f in q[(q.state==state)&q.part.isin(['validation','pilot_test'])].groupby(['part','cutoff_us']):
            indices=f.index.to_numpy();group,_=pd.factorize(f.relation,sort=False);ng=group.max()+1;n=len(f)
            counts=np.bincount(group);eligible=np.flatnonzero(counts>=5);y=f['exit'].to_numpy();observed=np.bincount(group,weights=y,minlength=ng)
            p=f.p.to_numpy();threshold=ndtri(p) if a.family=='gaussian' else -np.log1p(-p)
            draw=rng.standard_normal if a.family=='gaussian' else rng.standard_exponential
            eps=draw((a.draws,n),dtype=np.float32);gz=draw((a.draws,1),dtype=np.float32);rz=draw((a.draws,ng),dtype=np.float32)
            configs={mode:loads[indices] for mode,loads in loadings.items()}
            for g,r in fixed:configs[f'fixed_g{g}_r{r}']=np.tile(np.sqrt([g,r,1-g-r]),(n,1))
            for mode,loads in configs.items():
                if a.family=='gaussian':samples=(loads[:,2]*eps+loads[:,0]*gz+loads[:,1]*rz[:,group]<threshold).astype(np.float32)
                else:
                    weights=loads**2
                    samples=((eps<threshold*weights[:,2])|(gz<threshold*weights[:,0])|(rz[:,group]<threshold*weights[:,1])).astype(np.float32)
                totals=samples.sum(1)[:,None];tc=ensemble_crps(totals,[y.sum()])[0]/n
                cov,width,isc=interval_statistics(totals,[y.sum()])
                gc=np.zeros((a.draws,ng),np.float32)
                for j in range(a.draws):gc[j]=np.bincount(group,weights=samples[j],minlength=ng)
                if len(eligible):
                    crps=np.mean(ensemble_crps(gc[:,eligible],observed[eligible])/counts[eligible]);cc,ww,ii=interval_statistics(gc[:,eligible],observed[eligible])
                    g_cov=float(cc.mean());g_width=float(np.mean(ww/counts[eligible]));g_isc=float(np.mean(ii/counts[eligible]))
                else:crps=g_cov=g_width=g_isc=np.nan
                rows.append({'state':state,'split':split,'cutoff_us':int(t),'mode':mode,'n':n,'groups':len(eligible),'total_fraction_crps':tc,'group_fraction_crps':crps,'total_90_coverage':float(cov[0]),'total_90_width_fraction':width[0]/n,'total_90_interval_score':isc[0]/n,
                    'group_90_coverage':g_cov,'group_90_width_fraction':g_width,'group_90_interval_score':g_isc,'mean_global_variance':float(np.mean(loads[:,0]**2)),'mean_relation_variance':float(np.mean(loads[:,1]**2)),
                    'marginal_brier':float(np.mean((p-y)**2)),'marginal_mean':float(p.mean()),'observed_mean':float(y.mean())})
    scores=pd.DataFrame(rows);scores.to_parquet(run/'scenario_scores.parquet',index=False)
    metrics=scores.groupby(['state','split','mode']).mean(numeric_only=True).reset_index();metrics.to_json(run/'scenario_metrics.json',orient='records',indent=2)
    selected=[]
    for state in sorted(scores.state.unique()):
        val=metrics[(metrics.state==state)&(metrics.split=='validation')&metrics['mode'].str.startswith('fixed')].copy()
        val['score']=val.total_fraction_crps+val.group_fraction_crps.fillna(0);fixed_best=val.sort_values('score').iloc[0]['mode']
        selected.append({'state':int(state),'fixed_grid_selected_on_validation':fixed_best,'pilot_test':metrics[(metrics.state==state)&(metrics.split=='pilot_test')&metrics['mode'].isin(['global','static','context','fixed_g0_r0',fixed_best])].to_dict('records')})
    report={'config':vars(a),'results':selected,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
        'caveats':['Exploratory pilot, not final held-out evidence.','Borg outcomes refer to sampled query cohort, not entire cell.','Fixed-grid factors tuned directly on validation CRPS; learned factors selected by validation pair likelihood.','Dependence has disjoint fitting outcomes from the marginal predictor and calibration.']}
    (run/'scenario_report.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports'/f'learned_{tag}_{a.dataset}_seed{a.seed}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
