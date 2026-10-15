"""Retrospective graph ablations and collective slow-progress risk."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from otg.cohort_scores import projections
from otg.torch_scenarios import projected_counts_torch
from otg.scores import ensemble_crps
from revision.common import ROOT,frozen_loadings,write_json,digest


def risk_scores(proj,counts,p,y):
    rows=[];strata=[]
    for name,v in proj.items():
        code=v['code'];keep=code>=0;size=v['size'];g=len(size)
        def sums(a):return np.column_stack([np.bincount(code[keep],weights=a[keep,h],minlength=g) for h in range(4)])
        mean=sums(p);truth=sums(y);sd=np.sqrt(1+sums(p*(1-p)));sample=counts[name]
        crps=ensemble_crps(sample,truth)/size[:,None]
        for group in range(g):
            strata.append({'projection':name,'size':int(size[group]),'size_bin':int(np.floor(np.log2(size[group]))),'crps':float(crps[group].mean()),'absolute_mean_bias':float(np.abs(sample[:,group].mean(0)-truth[group]).mean()/size[group])})
        if name=='population_by_state':continue
        # Two sigma below the reference individual's expected exit count means
        # unexpectedly slow progress. Thresholds use supplied reference CDFs,
        # never target labels, and are identical across dependence modes.
        threshold=mean-2*sd
        flags=sample<threshold[None,:,:];observed=truth<threshold
        for h in range(4):
            fraction=flags[:,:,h].mean(1);actual=observed[:,h].mean();minimum=max(2,int(np.ceil(.2*g)))
            event=bool(observed[:,h].sum()>=minimum);prob=float((flags[:,:,h].sum(1)>=minimum).mean())
            rows.append({'projection':name,'horizon_index':h,'groups':g,'slow_group_fraction_crps':float(ensemble_crps(fraction[:,None],[actual])[0]),'observed_slow_fraction':float(actual),'predicted_slow_fraction':float(fraction.mean()),'collective_event':int(event),'collective_probability':prob,'collective_brier':(prob-event)**2,'minimum_slow_groups':minimum})
    return rows,strata


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['borg','bpi']);ap.add_argument('--device',default='cpu');ap.add_argument('--draws',type=int,default=512);a=ap.parse_args();torch.set_num_threads(3)
    out=ROOT/'runs'/f'revision_{a.dataset}_collective_risk_v1';out.mkdir(exist_ok=False);start=time.monotonic();data=ROOT/'data/processed'/f'{a.dataset}_joint_selected_v2';meta=json.loads((data/'metadata.json').read_text());relations=meta['relations'];unit='m' if a.dataset=='borg' else 'd';rows=[];strata=[]
    for period,sub in [('development','cohorts'),('retrospective_final','final_cohorts')]:
        directory=data/sub;cm=json.loads((directory/'manifest.json').read_text())
        for oi,c in enumerate(cm['cohorts']):
            f=directory/c['folder'];q=pd.read_parquet(f/'queries.parquet');p=np.load(f/'cdf.npy');x=np.load(f/'features.npy');y=q[[f'exit_{h}{unit}' for h in meta['horizons']]].to_numpy();rel=q[relations].to_numpy(np.int64);states=q.state.to_numpy();proj=projections(states,rel,relations);o={'q':q,'x':x}
            for seed in range(3):
                configs=frozen_loadings(o,seed,a.dataset);shuffled=rel.copy();rng=np.random.default_rng(980000+seed*10000+oi)
                riskbin=np.floor(np.log(np.clip(p[:,-1],1e-5,1-1e-5)/(1-np.clip(p[:,-1],1e-5,1-1e-5)))*2).astype(int)
                for k in range(rel.shape[1]):
                    for state in np.unique(states):
                        for rb in np.unique(riskbin[states==state]):
                            ids=np.flatnonzero((states==state)&(riskbin==rb)&(rel[:,k]>=0));shuffled[ids,k]=rng.permutation(rel[ids,k])
                configs['static_permuted']=configs['static']
                if a.dataset=='borg':
                    for drop,k in [('no_job',1),('no_machine',2)]:
                        w=configs['static'].copy();w[:,-1]=np.sqrt(w[:,-1]**2+w[:,k]**2);w[:,k]=0;configs[drop]=w
                for mode,w in configs.items():
                    counts=projected_counts_torch(p,w,shuffled if mode=='static_permuted' else rel,proj,a.draws,970000+seed*10000+oi,'gaussian',a.device)
                    rr,ss=risk_scores(proj,counts,p,y);base={'period':period,'cutoff_us':c['cutoff_us'],'mode':mode,'seed':seed};rows.extend([{**v,**base} for v in rr]);strata.extend([{**v,**base} for v in ss])
            if (oi+1)%4==0:print(json.dumps({'period':period,'origins':oi+1,'elapsed_s':time.monotonic()-start}),flush=True)
    df=pd.DataFrame(rows);df.to_parquet(out/'risk_scores.parquet',index=False);pd.DataFrame(strata).to_parquet(out/'group_scores.parquet',index=False)
    df.groupby(['period','mode','projection']).mean(numeric_only=True).to_csv(out/'summary.csv')
    write_json(out/'manifest.json',{'status':'complete','config':vars(a),'scope':'Retrospective new endpoints and graph controls; thresholds fixed in code before execution, no fitted SLA.','risk_definition':'A group progresses slowly if exits are below its shared reference individual-CDF mean by more than 2 sqrt(1+sum p(1-p)); collective event requires at least max(2,ceil(.2*groups)) slow groups. Threshold does not depend on evaluated dependence mode or actual labels.','ablations':'Remove one typed variance component from frozen static loads and move it to private variance; keep individual CDFs and other weights fixed. Permute observed incidence within current state and fine risk strata.','script_sha256':digest(__file__),'elapsed_s':time.monotonic()-start})
    print(df.groupby(['period','mode'])[['slow_group_fraction_crps','collective_brier','collective_event']].mean().to_string(),flush=True)


if __name__=='__main__':main()
