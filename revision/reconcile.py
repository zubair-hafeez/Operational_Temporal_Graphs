"""Two-level discrete probabilistic reconciliation by conditioning.

Adapts the importance-weighting step of Zambon et al. (2024),
doi:10.1007/s11222-023-10343-y, to joint first-exit category vectors.
This is a classical baseline, not a new reconciliation algorithm.
"""
import numpy as np
from scipy.special import logsumexp
from revision.counts import dm_logpmf


def reconcile_trajectories(bottom,upper_alpha,draws,rng):
    total=bottom.sum(1);n=total.sum(-1)
    logw=dm_logpmf(total,n,np.asarray(upper_alpha))
    weights=np.exp(logw-logsumexp(logw))
    if not np.isfinite(weights).all():raise ValueError('Invalid importance weights')
    ids=rng.choice(len(bottom),draws,replace=True,p=weights)
    ess=float(1/(weights@weights))
    return bottom[ids],{'pool':len(bottom),'ess':ess,'ess_fraction':ess/len(bottom),'max_weight':float(weights.max()),'unique_selected':int(len(np.unique(ids)))}
