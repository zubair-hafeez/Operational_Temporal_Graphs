"""Operational projections of joint first-exit draws; observed labels score only."""
from __future__ import annotations
import numpy as np
from .scenarios import factor_uniform_chunks,first_exit_bins
from .scores import ensemble_crps,interval_statistics


def projections(states,relations,relation_names):
    result={};_,code=np.unique(states,return_inverse=True)
    result['population_by_state']={'code':code.astype(np.int32),'size':np.bincount(code)}
    for k,name in enumerate(relation_names):
        valid=relations[:,k]>=0;code=np.full(len(states),-1,np.int32)
        _,inverse=np.unique(relations[valid,k],return_inverse=True)
        sizes=np.bincount(inverse);eligible=sizes>=5;mapping=np.full(len(sizes),-1,np.int32);mapping[eligible]=np.arange(eligible.sum())
        code[valid]=mapping[inverse]
        if eligible.any():result[name]={'code':code,'size':sizes[eligible]}
    return result


def cohort_scores(cdf,loadings,relations,states,labels,relation_names,draws,seed,family='gaussian',sampling_relations=None,chunk_size=4096,device='cpu'):
    n,h=cdf.shape
    if np.any(np.diff(labels,axis=1)<0):raise ValueError('Nonmonotone outcome labels')
    projection=projections(states,relations,relation_names)
    counts={name:np.zeros((draws,len(v['size']),h),np.float32) for name,v in projection.items()}
    truth={}
    for name,v in projection.items():
        valid=v['code']>=0
        truth[name]=np.column_stack([np.bincount(v['code'][valid],weights=labels[valid,k],minlength=len(v['size'])) for k in range(h)])
    random=np.random.default_rng(seed)
    incidence=relations if sampling_relations is None else sampling_relations
    if str(device).startswith('cuda'):
        from .torch_scenarios import projected_counts_torch
        counts=projected_counts_torch(cdf,loadings,incidence,projection,draws,seed,family,device)
        chunks=[]
    else:chunks=factor_uniform_chunks(loadings,incidence,draws,random,family,chunk_size)
    for lo,hi,u in chunks:
        bins=first_exit_bins(cdf[lo:hi],u)
        for name,v in projection.items():
            code=v['code'][lo:hi];valid=code>=0
            if not valid.any():continue
            local,inverse=np.unique(code[valid],return_inverse=True);base=inverse*(h+1)
            # Batch draw indices into one bincount. Blocking bounds temporary
            # memory while avoiding a Python loop for each ensemble member.
            stride=len(local)*(h+1)
            for first_draw in range(0,draws,64):
                stop_draw=min(draws,first_draw+64);number=stop_draw-first_draw
                encoded=(np.arange(number)[:,None]*stride+base[None,:]+bins[first_draw:stop_draw,valid]).ravel()
                bc=np.bincount(encoded,minlength=number*stride).reshape(number,len(local),h+1)
                counts[name][first_draw:stop_draw,local]+=np.cumsum(bc[:,:,:h],axis=2)
    return score_projected_counts(projection,counts,truth)


def score_projected_counts(projection,counts,truth):
    """Score any declared aggregate forecast with the same operational metrics."""
    rows=[]
    for name,v in projection.items():
        size=v['size'];sample=counts[name];actual=truth[name]
        draws,_,h=sample.shape
        crps=ensemble_crps(sample,actual)/size[:,None]
        coverage,width,interval=interval_statistics(sample,actual)
        # Independent adjacent ensemble draws give an unbiased energy-score
        # estimator for each normalized multi-horizon group trajectory.
        normalized=sample/size[None,:,None];observed=actual/size[:,None]
        first=np.linalg.norm(normalized-observed[None,:,:],axis=2).mean(0)
        even=(draws//2)*2
        second=.5*np.linalg.norm(normalized[:even:2]-normalized[1:even:2],axis=2).mean(0)
        for k in range(h):
            # A declared proxy: reserve units for the current cohort that have
            # not exited. No arrivals, policy effects, or monetary cost inferred.
            costs={}
            for ratio in [3,9]:
                alpha=ratio/(ratio+1);remaining=size[None,:]-sample[:,:,k]
                buffer=np.quantile(remaining,alpha,axis=0,method='higher');real=size-actual[:,k]
                costs[f'buffer_cost_ratio{ratio}']=float(np.mean((ratio*np.maximum(real-buffer,0)+np.maximum(buffer-real,0))/size))
            threshold_fraction=.2
            alarm_samples=(sample[:,:,k]/size[None,:]>=threshold_fraction).mean(1)[:,None]
            alarm_actual=(actual[:,k]/size>=threshold_fraction).mean()
            rows.append({'projection':name,'horizon_index':k,'groups':len(size),'entities_in_projection':int(size.sum()),'fraction_crps':float(crps[:,k].mean()),
                'coverage_90':float(coverage[:,k].mean()),'width_90_fraction':float(np.mean(width[:,k]/size)),'interval_score_90_fraction':float(np.mean(interval[:,k]/size)),
                'mean_prediction_fraction':float(np.mean(sample[:,:,k].mean(0)/size)),'observed_fraction':float(np.mean(actual[:,k]/size)),
                'simultaneous_alarm_fraction_crps':float(ensemble_crps(alarm_samples,[alarm_actual])[0]),
                'trajectory_energy':float(np.mean(first-second)),**costs})
    return rows
