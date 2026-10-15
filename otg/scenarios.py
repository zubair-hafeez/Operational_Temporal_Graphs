"""Streaming joint first-exit scenarios with one latent uniform per entity.

Shared factor draws are created once per cohort and reused across node chunks,
so chunking cannot turn a job/machine into independent partial groups.
"""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr


def exponential_race_time(noise,weight):
    # Float32 RNGs can return exactly zero. A zero-weight factor is always
    # absent, including when its draw is zero; 0/0 must never create NaNs.
    shape=np.broadcast_shapes(noise.shape,weight.shape)
    return np.divide(noise,weight,out=np.full(shape,np.inf,np.float32),where=weight>0)


def factor_uniform_chunks(loadings,relations,draws,rng,family='gaussian',chunk_size=4096):
    loadings=np.asarray(loadings,np.float32);relations=np.asarray(relations,np.int64)
    if relations.ndim==1:relations=relations[:,None]
    n=len(loadings);nr=relations.shape[1]
    if loadings.shape!=(n,nr+2):raise ValueError('Factor shape mismatch')
    if not np.allclose((loadings**2).sum(1),1,atol=1e-5):raise ValueError('Unit marginal variance/weight required')
    if np.any(loadings<0):raise ValueError('Nonnegative loadings required')
    if family not in ['gaussian','commonshock']:raise ValueError(family)
    draw=rng.standard_normal if family=='gaussian' else rng.standard_exponential
    global_noise=draw((draws,1),dtype=np.float32)
    codes=[];shared=[]
    for r in range(nr):
        valid=relations[:,r]>=0;code=np.full(n,-1,np.int32)
        _,inverse=np.unique(relations[valid,r],return_inverse=True);code[valid]=inverse
        codes.append(code);shared.append(draw((draws,int(inverse.max())+1 if len(inverse) else 0),dtype=np.float32))
        if np.any(loadings[~valid,r+1]!=0):raise ValueError('Missing relation weight must be private')
    for lo in range(0,n,chunk_size):
        hi=min(lo+chunk_size,n);w=loadings[lo:hi];private=draw((draws,hi-lo),dtype=np.float32)
        if family=='gaussian':latent=private*w[:,-1]+global_noise*w[:,0]
        else:
            ww=w**2
            latent=np.minimum(exponential_race_time(private,ww[:,-1]),exponential_race_time(global_noise,ww[:,0]))
        for r in range(nr):
            code=codes[r][lo:hi];valid=code>=0
            if not np.any(valid):continue
            noise=shared[r][:,code[valid]]
            if family=='gaussian':latent[:,valid]+=noise*w[valid,r+1]
            else:
                latent[:,valid]=np.minimum(latent[:,valid],exponential_race_time(noise,ww[valid,r+1]))
        uniform=ndtr(latent) if family=='gaussian' else -np.expm1(-latent)
        uniform=np.clip(uniform,np.nextafter(np.float32(0),np.float32(1)),np.nextafter(np.float32(1),np.float32(0)))
        yield lo,hi,uniform


def first_exit_bins(cdf,uniform):
    """Return 0..H-1 for an exit bin, H for survival beyond the last horizon."""
    cdf=np.asarray(cdf)
    if np.any(np.diff(cdf,axis=1)<-1e-7) or np.any((cdf<0)|(cdf>1)):raise ValueError('Invalid CDF')
    return np.sum(uniform[:,:,None]>cdf[None,:,:],axis=-1).astype(np.int8)
