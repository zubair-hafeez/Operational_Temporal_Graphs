"""Threshold-free joint count-profile and peak-deviation proper scores."""
import numpy as np
from otg.scores import ensemble_crps


def profile_scores(sample,truth,size,reference_mean,reference_scale,seed=123):
    draws,g,h=sample.shape;norm=sample/size[None,:,None];target=truth/size[:,None]
    # RMS across current groups keeps the scale comparable as groups change.
    first=np.sqrt(((norm-target[None])**2).mean(1)).mean(0)
    even=(draws//2)*2
    second=.5*np.sqrt(((norm[:even:2]-norm[1:even:2])**2).mean(1)).mean(0)
    # Under-counts of exits correspond to excess remaining backlog.
    peak=((reference_mean[None]-sample)/reference_scale[None]).max(1)
    observed_peak=((reference_mean-truth)/reference_scale).max(0)
    peak_crps=ensemble_crps(peak,observed_peak)
    rng=np.random.default_rng(seed)
    variogram=np.zeros(h)
    if g>1:
        i=rng.integers(g,size=2048);j=rng.integers(g-1,size=2048);j+=j>=i
        v=np.sqrt(np.abs(norm[:,i]-norm[:,j]));obs=np.sqrt(np.abs(target[i]-target[j]));mean=v.mean(0)
        # Remove finite-ensemble variance bias of the squared moment error.
        variogram=((mean-obs)**2-v.var(0,ddof=1)/draws).mean(0)
    return [{'horizon_index':k,'joint_profile_energy':float(first[k]-second[k]),'joint_variogram_fair':float(variogram[k]),'peak_slowdown_crps':float(peak_crps[k]),'observed_peak_slowdown':float(observed_peak[k])} for k in range(h)]
