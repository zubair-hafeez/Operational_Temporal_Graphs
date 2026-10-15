"""Conditional Dirichlet–multinomial count trajectories (classical baseline)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln,softmax
from otg.residuals import aggregate_inputs


def category_counts(cumulative,size):
    return np.diff(np.concatenate([np.zeros((*cumulative.shape[:-1],1)),cumulative,np.broadcast_to(size,cumulative.shape[:-1])[...,None]],axis=-1),axis=-1)


def group_data(origin,relations):
    proj,inputs=aggregate_inputs(origin['p'],origin['q'].state.to_numpy(),origin['q'][relations].to_numpy(),relations,origin['y'])
    result={}
    for name,v in inputs.items():
        n=v['size'];prob=category_counts(v['mean'],n)/n[:,None]
        prob=np.maximum(prob,1e-6);prob/=prob.sum(1,keepdims=True)
        result[name]={'n':n,'prob':prob,'target':category_counts(v['truth'],n).astype(np.int64),'truth':v['truth']}
    return proj,result


def dm_logpmf(y,n,alpha):
    k=alpha.sum(-1)
    return gammaln(n+1)-gammaln(y+1).sum(-1)+gammaln(k)-gammaln(k+n)+(gammaln(alpha+y)-gammaln(alpha)).sum(-1)


def parameters(prob,fit):
    theta=np.asarray(fit['theta']);bias=np.r_[theta[:4],0.] if fit['bias'] else np.zeros(5)
    mean=softmax(np.log(prob)+bias,axis=-1)
    return np.maximum(mean*np.exp(theta[-1]),1e-8)


def fit_dm(records,bias):
    p=np.concatenate([z['prob'] for z in records]);n=np.concatenate([z['n'] for z in records]);y=np.concatenate([z['target'] for z in records])
    w=np.concatenate([np.full(len(z['n']),1/(len(records)*len(z['n']))) for z in records])
    def objective(theta):
        fit={'theta':theta,'bias':bias};alpha=parameters(p,fit)
        return -np.dot(w,dm_logpmf(y,n,alpha))+(1e-3*np.sum(theta[:-1]**2) if bias else 0)
    candidates=[]
    for init in [2.,5.]:
        theta=np.r_[np.zeros(4),init] if bias else np.array([init])
        opt=minimize(objective,theta,method='L-BFGS-B',bounds=([(-2,2)]*4 if bias else [])+[(-2,12)],options={'maxiter':500,'ftol':1e-10})
        candidates.append(opt)
    opt=min(candidates,key=lambda v:v.fun)
    if not opt.success:raise RuntimeError(str(opt.message))
    return {'bias':bias,'theta':opt.x.tolist(),'fit_objective':float(opt.fun),'origins':len(records),'groups':len(n),'regularization':.001}


def sample_dm(n,alpha,draws,rng):
    """Vectorized sequential beta-binomial representation, exact integer totals."""
    n=np.asarray(n,np.int64);alpha=np.asarray(alpha,float)
    remaining=np.broadcast_to(n,(draws,len(n))).copy();out=np.empty((draws,len(n),5),np.int32)
    for k in range(4):
        prob=rng.beta(alpha[None,:,k],alpha[None,:,k+1:].sum(-1),size=(draws,len(n)))
        v=rng.binomial(remaining,prob);out[:,:,k]=v;remaining-=v
    out[:,:,-1]=remaining
    return out
