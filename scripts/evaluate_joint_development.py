"""Full-cohort joint distributions and predeclared decision proxies on development."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from otg.copula import StratifiedFactors
from otg.cohort_scores import cohort_scores
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--history',action='store_true')
    ap.add_argument('--family',choices=['gaussian','commonshock'],default='gaussian');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--draws',type=int,default=512);a=ap.parse_args()
    torch.set_num_threads(4);tag='history_v1' if a.history else 'v1';data=ROOT/'data/processed'/f'{a.dataset}_joint_{tag}';m=json.loads((data/'metadata.json').read_text())
    run=ROOT/'runs'/f'joint_{a.dataset}_{tag}_{a.family}_seed{a.seed}';manifest=json.loads((run/'manifest.json').read_text())
    output=run/f'full_cohort_scores_m{a.draws}.parquet'
    if output.exists():raise FileExistsError('Preserve prior evaluation')
    cohort_manifest=json.loads((data/'cohorts/manifest.json').read_text());models={}
    for mode in ['global','static','context']:
        model=StratifiedFactors(manifest['input_dim'],len(manifest['states']),manifest['n_relations'],mode=mode,hidden=manifest['hidden'])
        model.load_state_dict(torch.load(run/f'{mode}.pt',weights_only=True,map_location='cpu'));model.eval();models[mode]=model
    rows=[];start=time.monotonic();unit='m' if a.dataset=='borg' else 'd'
    for cohort_index,c in enumerate(cohort_manifest['cohorts']):
        folder=data/'cohorts'/c['folder'];q=pd.read_parquet(folder/'queries.parquet');p=np.load(folder/'cdf.npy');x=torch.as_tensor(np.load(folder/'features.npy'))
        relation=q[m['relations']].to_numpy(np.int64);states=q.state.to_numpy();st=torch.as_tensor(np.searchsorted(manifest['states'],states),dtype=torch.long);available=torch.as_tensor(relation>=0,dtype=torch.float32)
        y=q[[f'exit_{h}{unit}' for h in m['horizons']]].to_numpy(np.int8);configs={}
        with torch.no_grad():
            for mode,model in models.items():configs[mode]=torch.cat([model(x[lo:lo+16384],st[lo:lo+16384],available[lo:lo+16384])[0] for lo in range(0,len(x),16384)]).numpy()
        independent=np.zeros_like(configs['static']);independent[:,-1]=1;configs['independent']=independent
        # Noise-incidence control holds all individual CDFs and fitted weights.
        shuffled=relation.copy();random=np.random.default_rng(4096+cohort_index)
        risk=np.floor(np.log(np.clip(p[:,-1],1e-5,1-1e-5)/(1-np.clip(p[:,-1],1e-5,1-1e-5)))*2).astype(int)
        for column in range(relation.shape[1]):
            for state in np.unique(states):
                for s in np.unique(risk[states==state]):
                    ids=np.flatnonzero((states==state)&(risk==s)&(relation[:,column]>=0));shuffled[ids,column]=random.permutation(relation[ids,column])
        configs['static_permuted']=configs['static']
        for mode,loads in configs.items():
            result=cohort_scores(p,loads,relation,states,y,m['relations'],a.draws,100000+a.seed*10000+cohort_index,a.family,sampling_relations=shuffled if mode=='static_permuted' else None)
            rows.extend([{**r,'mode':mode,'cutoff_us':c['cutoff_us'],'n':len(q),'horizon':m['horizons'][r['horizon_index']],'horizon_unit':m['horizon_unit']} for r in result])
        print(json.dumps({'cohort':c['folder'],'n':len(q),'elapsed_s':time.monotonic()-start}),flush=True)
    frame=pd.DataFrame(rows);frame.to_parquet(output,index=False)
    summary=frame.groupby(['mode','projection','horizon']).mean(numeric_only=True).reset_index();summary.to_json(run/f'full_cohort_metrics_m{a.draws}.json',orient='records',indent=2)
    macro=frame.groupby(['mode','cutoff_us']).fraction_crps.mean().groupby('mode').mean().to_dict()
    report={'config':vars(a),'macro_fraction_crps':macro,'n_cohorts':len(cohort_manifest['cohorts']),'elapsed_s':time.monotonic()-start,
        'macro_definition':'Equal cutoff, projection type, horizon, and group weights; population projection groups by current state. Relation groups require >=5 entities.',
        'decision_proxy':'Reserve capacity for unexited current cohort at quantiles .75/.9 under normalized asymmetric loss ratios 3/9. No causal policy or currency claim.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scores_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'training_manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()}
    (run/f'full_cohort_report_m{a.draws}.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'reports'/f'joint_{a.dataset}_{tag}_{a.family}_seed{a.seed}_m{a.draws}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
