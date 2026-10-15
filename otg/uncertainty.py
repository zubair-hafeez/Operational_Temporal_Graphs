"""Paired uncertainty for temporally dependent forecast-origin scores."""
import numpy as np


def paired_moving_block_interval(difference,block_length,repeats=10000,seed=0):
    """Circular moving-block bootstrap of the mean paired score difference.

    All outcomes/horizons/groups must first be averaged within each forecast
    origin. Seeds are Monte Carlo/training repetitions, not new observations.
    This interval remains an approximation for short dependent traces.
    """
    d=np.asarray(difference,np.float64)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():raise ValueError('Need finite origin-level differences')
    if not 1<=block_length<=len(d):raise ValueError('Invalid block length')
    rng=np.random.default_rng(seed);blocks=int(np.ceil(len(d)/block_length))
    starts=rng.integers(0,len(d),size=(repeats,blocks))
    indices=((starts[:,:,None]+np.arange(block_length))%len(d)).reshape(repeats,-1)[:,:len(d)]
    means=d[indices].mean(1)
    return {'mean_difference':float(d.mean()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975)),
            'n_origins':len(d),'block_length_origins':int(block_length),'bootstrap_repeats':repeats,'seed':seed}
