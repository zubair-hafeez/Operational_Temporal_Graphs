"""Delayed, order-preserving marginal correction with explicit fit-origin audit."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit

DAY_US=86400*1_000_000


def correct(p, intercept):
    return expit(logit(np.clip(p,1e-7,1-1e-7))+intercept).astype(np.float32)


def fit_intercept(origins):
    if not origins:
        return 0.0
    # Origin balance avoids giving larger active populations more influence.
    logits=np.concatenate([logit(np.clip(o['p'],1e-7,1-1e-7)).ravel() for o in origins])
    labels=np.concatenate([o['y'].ravel() for o in origins])
    w=np.concatenate([np.full(o['p'].size,1/(len(origins)*o['p'].size)) for o in origins])
    def objective(b):
        v=logits+b
        return np.dot(w,np.logaddexp(0,v)-labels*v)+1e-4*b*b
    opt=minimize_scalar(objective,bounds=(-2,2),method='bounded',options={'xatol':1e-7})
    if not opt.success:raise RuntimeError(opt.message)
    return float(opt.x)


def eligible_history(origins,t,window_days=60,maturity_days=14):
    mature=[o for o in origins if o['t']+maturity_days*DAY_US<=t]
    recent=[o for o in mature if o['t']>=t-window_days*DAY_US]
    return recent if len(recent)>=4 else mature
