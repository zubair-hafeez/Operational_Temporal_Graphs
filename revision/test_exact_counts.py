import numpy as np
from scipy.stats import betabinom
from revision.exact_counts import beta_binomial_scores


def test_exact_crps_against_double_sum():
    n=9;a=2.3;b=.7;y=5;k=np.arange(n+1);p=betabinom.pmf(k,n,a,b)
    expected=p@abs(k-y)-.5*np.sum(p[:,None]*p[None,:]*abs(k[:,None]-k[None,:]))
    score=beta_binomial_scores(n,a,b,y)
    np.testing.assert_allclose(score['fraction_crps'],expected/n,atol=1e-12)
    remain=n-k;quantile=np.searchsorted(p[::-1].cumsum(),.9);actual=n-y
    assert score['buffer_cost_ratio9']==(9*max(actual-quantile,0)+max(quantile-actual,0))/n
