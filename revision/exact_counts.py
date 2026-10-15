"""Exact marginal scores for DM cumulative counts via beta-binomial masses."""
import numpy as np
from scipy.special import gammaln,betaln


def beta_binomial_scores(n,a,b,y):
    n=int(n);k=np.arange(n+1,dtype=float)
    logp=gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)+betaln(k+a,n-k+b)-betaln(a,b)
    pmf=np.exp(logp-logp.max());pmf/=pmf.sum();cdf=pmf.cumsum();cdf[-1]=1
    # Half the pairwise absolute-distance expectation, evaluated in O(n).
    prev=cdf-pmf;weighted=(pmf*k).cumsum()-pmf*k
    crps=float(pmf@np.abs(k-y)-np.sum(pmf*(k*prev-weighted)))
    low=int(np.searchsorted(cdf,.05));high=int(np.searchsorted(cdf,.95));interval=high-low+20*max(low-y,0)+20*max(y-high,0)
    scores={'fraction_crps':crps/n,'coverage_90':float(low<=y<=high),'width_90_fraction':(high-low)/n,'interval_score_90_fraction':interval/n,'mean_prediction_fraction':float((pmf@k)/n),'observed_fraction':y/n}
    # Inverse CDF of remaining counts, with support reversed.
    rem=np.arange(n+1);rcdf=pmf[::-1].cumsum();rcdf[-1]=1
    for ratio in [3,9]:
        buffer=int(np.searchsorted(rcdf,ratio/(ratio+1)));actual=n-y;scores[f'buffer_cost_ratio{ratio}']=(ratio*max(actual-buffer,0)+max(buffer-actual,0))/n
    return scores


def exact_projected_scores(gs,alphas):
    rows=[]
    for name,z in gs.items():
        alpha=alphas[name];n=z['n'];target=z['truth']
        for h in range(4):
            per=[beta_binomial_scores(nn,aa[:h+1].sum(),aa[h+1:].sum(),yy[h]) for nn,aa,yy in zip(n,alpha,target)]
            rows.append({'projection':name,'horizon_index':h,'groups':len(n),**{k:float(np.mean([v[k] for v in per])) for k in per[0]}})
    return rows
