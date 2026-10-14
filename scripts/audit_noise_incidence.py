"""Hold marginals/loadings fixed; permute which nodes share relation noise."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import ndtri
from otg.scores import ensemble_crps
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['borg','bpi'],required=True);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
    data=ROOT/'data/processed'/f'{a.dataset}_copula_pilot';run=ROOT/'runs'/f'learned_copula_{a.dataset}_seed{a.seed}'
    q=pd.read_parquet(data/'queries.parquet');rng=np.random.default_rng(9321);rows=[];draws=512
    for state in map(int,sorted(q.state.unique())):
        loads=np.load(run/f'static_state{state}_loadings.npy')
        for t,f in q[(q.state==state)&(q.part=='pilot_test')].groupby('cutoff_us'):
            idx=f.index.to_numpy();w=loads[idx];group,_=pd.factorize(f.relation,sort=False);counts=np.bincount(group);ng=len(counts);eligible=np.flatnonzero(counts>=5)
            p=f.p.to_numpy();y=f['exit'].to_numpy();threshold=ndtri(p);truth=np.bincount(group,weights=y,minlength=ng)
            strata=np.floor(np.log(p/(1-p))*2).astype(int)
            matched=group.copy()
            for s in np.unique(strata):
                ids=np.flatnonzero(strata==s);matched[ids]=rng.permutation(group[ids])
            shuffled=rng.permutation(group)
            assert np.array_equal(np.bincount(matched,minlength=ng),counts)
            eps=rng.standard_normal((draws,len(f)),dtype=np.float32);gz=rng.standard_normal((draws,1),dtype=np.float32);rz=rng.standard_normal((draws,ng),dtype=np.float32)
            for name,g in [('true_incidence',group),('risk_matched_permutation',matched),('unconditional_permutation',shuffled)]:
                samples=(eps*w[:,2]+gz*w[:,0]+rz[:,g]*w[:,1]<threshold).astype(np.float32)
                gc=np.zeros((draws,ng),np.float32)
                for j in range(draws):gc[j]=np.bincount(group,weights=samples[j],minlength=ng)
                rows.append({'state':state,'cutoff_us':int(t),'condition':name,'n':len(f),'groups':len(eligible),
                    'total_fraction_crps':float(ensemble_crps(samples.sum(1)[:,None],[y.sum()])[0]/len(f)),
                    'true_group_fraction_crps':float(np.mean(ensemble_crps(gc[:,eligible],truth[eligible])/counts[eligible])) if len(eligible) else None})
    result=pd.DataFrame(rows);result.to_parquet(run/'noise_incidence_audit.parquet',index=False)
    report={'dataset':a.dataset,'results':result.groupby(['state','condition']).mean(numeric_only=True).reset_index().to_dict('records'),
        'contract':'Same calibrated marginals and static factor weights; permute only relation-noise incidence. Preserve group-size multiset. Score original real groups.',
        'caveat':'Exploratory pilot; one permutation per cutoff, no inferential claim.'}
    (ROOT/'reports'/f'noise_incidence_{a.dataset}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
