"""Historical aggregate-error resampling: a direct-count benchmark.

This forecasts separate group counts; it does not supply entity-level coherent
scenarios or preserve the individual CDFs. All residuals come from development.
"""
import numpy as np
from .cohort_scores import projections,score_projected_counts


def aggregate_inputs(cdf,states,relations,relation_names,labels=None):
    projection=projections(states,relations,relation_names);result={}
    for name,v in projection.items():
        code=v['code'];keep=code>=0;n=len(v['size'])
        def sums(values):
            return np.column_stack([np.bincount(code[keep],weights=values[keep,k],minlength=n) for k in range(values.shape[1])])
        result[name]={'size':v['size'],'mean':sums(cdf),'scale':np.sqrt(1+sums(cdf*(1-cdf)))}
        if labels is not None:result[name]['truth']=sums(labels)
    return projection,result


def residual_counts(target,templates,draws,seed):
    """One common historical origin per draw, then size-matched group residuals.

    The full four-horizon residual vector is resampled together. Counts are
    clipped, rounded, and monotonized. No target outcome enters this routine.
    """
    if not templates:raise ValueError('At least one earlier residual origin required')
    rng=np.random.default_rng(seed);origins=rng.integers(len(templates),size=draws);counts={}
    for name,v in target.items():
        n=len(v['size']);out=np.empty((draws,n,v['mean'].shape[1]),np.float32)
        bins=np.floor(np.log2(v['size'])).astype(int)
        for origin in np.unique(origins):
            rows=np.flatnonzero(origins==origin);ref=templates[origin][name]
            if name=='population_by_state':
                if len(ref['size'])!=n:raise ValueError('Population states changed')
                residual=np.broadcast_to(ref['residual'],(len(rows),*ref['residual'].shape))
            else:
                residual=np.empty((len(rows),n,v['mean'].shape[1]),np.float32)
                old_bins=np.floor(np.log2(ref['size'])).astype(int)
                for b in np.unique(bins):
                    ids=np.flatnonzero(bins==b);distance=np.abs(old_bins-b);pool=np.flatnonzero(distance==distance.min())
                    selected=rng.choice(pool,size=(len(rows),len(ids)))
                    residual[:,ids]=ref['residual'][selected]
            values=v['mean'][None]+v['scale'][None]*residual
            values=np.rint(np.clip(values,0,v['size'][None,:,None]))
            out[rows]=np.maximum.accumulate(values,axis=2)
        counts[name]=out
    return counts
