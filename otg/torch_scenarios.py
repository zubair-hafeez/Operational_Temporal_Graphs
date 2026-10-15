"""Accelerated streaming scenario counts with exact integer accumulation."""
import numpy as np
import torch


def projected_counts_torch(cdf,loadings,relations,projection,draws,seed,family,device,chunk_size=8192):
    device=torch.device(device);generator=torch.Generator(device=device).manual_seed(seed)
    p=torch.as_tensor(cdf,device=device,dtype=torch.float32);loads=torch.as_tensor(loadings,device=device,dtype=torch.float32)
    if not torch.allclose((loads**2).sum(1),torch.ones(len(loads),device=device),atol=1e-5):raise ValueError('Unnormalized factor weights')
    if (loads<0).any():raise ValueError('Negative loading')
    n,h=p.shape;nr=relations.shape[1]
    def noise(shape):
        if family=='gaussian':return torch.randn(shape,device=device,generator=generator)
        if family=='commonshock':return torch.empty(shape,device=device).exponential_(generator=generator)
        raise ValueError(family)
    global_noise=noise((draws,1));codes=[];shared=[]
    for k in range(nr):
        known=relations[:,k]>=0;code=np.full(n,-1,np.int64);_,inverse=np.unique(relations[known,k],return_inverse=True);code[known]=inverse
        if np.any(np.asarray(loadings)[~known,k+1]!=0):raise ValueError('Unknown relation weight is not private')
        codes.append(torch.as_tensor(code,device=device));shared.append(noise((draws,int(inverse.max())+1 if len(inverse) else 0)))
    projected_codes={name:torch.as_tensor(v['code'],device=device,dtype=torch.long) for name,v in projection.items()}
    # Integer additions make the result independent of GPU scatter scheduling.
    bins_by_group={name:torch.zeros((draws,len(v['size'])*(h+1)),device=device,dtype=torch.int32) for name,v in projection.items()}
    def race(e,w):return torch.where(w>0,e/w.clamp_min(torch.finfo(torch.float32).tiny),torch.inf)
    for lo in range(0,n,chunk_size):
        hi=min(n,lo+chunk_size);w=loads[lo:hi];private=noise((draws,hi-lo))
        if family=='gaussian':latent=private*w[:,-1]+global_noise*w[:,0]
        else:
            ww=w**2;latent=torch.minimum(race(private,ww[:,-1]),race(global_noise,ww[:,0]))
        for k in range(nr):
            code=codes[k][lo:hi];valid=code>=0
            if not valid.any():continue
            eps=shared[k][:,code[valid]]
            if family=='gaussian':latent[:,valid]+=eps*w[valid,k+1]
            else:latent[:,valid]=torch.minimum(latent[:,valid],race(eps,ww[valid,k+1]))
        u=torch.special.ndtr(latent) if family=='gaussian' else -torch.expm1(-latent)
        u=u.clamp(torch.finfo(torch.float32).tiny,1-torch.finfo(torch.float32).eps)
        event_bin=(u[:,:,None]>p[None,lo:hi,:]).sum(-1)
        for name,code in projected_codes.items():
            current=code[lo:hi];valid=current>=0
            if not valid.any():continue
            index=current[None,valid]*(h+1)+event_bin[:,valid]
            bins_by_group[name].scatter_add_(1,index,torch.ones(index.shape,device=device,dtype=torch.int32))
    return {name:values.reshape(draws,-1,h+1)[:,:,:h].cumsum(2).to(torch.float32).cpu().numpy() for name,values in bins_by_group.items()}
