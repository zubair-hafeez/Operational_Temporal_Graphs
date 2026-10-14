"""Proper scores for samples, including finite-ensemble bias correction."""
import numpy as np


def ensemble_crps(samples,truth,fair=True):
    """Samples shape (draws, targets); output one CRPS per target.

    Fair mode uses distinct ensemble pairs, removing finite-M upward bias.
    Negative realized values can occur for an unbiased finite-ensemble estimate.
    """
    x=np.asarray(samples,dtype=np.float64)
    y=np.asarray(truth,dtype=np.float64)
    m=x.shape[0]
    if m<2:raise ValueError('Need at least two draws')
    ordered=np.sort(x,axis=0)
    coeff=(2*np.arange(m)-m+1).reshape((m,)+(1,)*(x.ndim-1))
    second=(coeff*ordered).sum(axis=0)/(m*(m-1) if fair else m*m)
    return np.abs(x-y).mean(axis=0)-second


def interval_statistics(samples,truth,alpha=.1):
    lo,hi=np.quantile(samples,[alpha/2,1-alpha/2],axis=0)
    y=np.asarray(truth)
    score=hi-lo+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0)
    return (y>=lo)&(y<=hi),hi-lo,score
