"""Positive shared-factor Gaussian copula with exact Bernoulli marginals.

This is a model family built from established factor-copula ingredients. Novelty
must be assessed through the full operational formulation and empirical evidence.
"""
import numpy as np
import torch
from torch import nn


class FactorLoadings(nn.Module):
    def __init__(self,input_dim,mode='context',hidden=32,n_relations=1):
        super().__init__();self.mode=mode;self.n_relations=n_relations
        initial=torch.tensor([-2.]+[0.]*(n_relations+1))
        if mode=='context':
            self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,n_relations+2))
            nn.init.zeros_(self.net[-1].weight)
            self.net[-1].bias.data.copy_(initial)
        elif mode=='static':self.logits=nn.Parameter(initial)
        elif mode=='global':self.logits=nn.Parameter(torch.tensor([-2.,0.]))
        else:raise ValueError(mode)

    def forward(self,x):
        if self.mode=='context':prob=torch.softmax(self.net(x),dim=-1)
        elif self.mode=='static':prob=torch.softmax(self.logits,dim=-1).expand(len(x),-1)
        else:
            p=torch.softmax(self.logits,dim=-1).expand(len(x),-1)
            prob=torch.cat([p[:,:1],p.new_zeros((len(p),self.n_relations)),p[:,1:]],dim=-1)
        # Keep individual variance >=0.02. Every node's latent variance is one.
        shared=.98*prob[:,:-1]
        individual=1-shared.sum(-1,keepdim=True)
        return torch.sqrt(torch.cat([shared,individual],dim=-1))


class StratifiedFactors(nn.Module):
    """State-specific classical factors with optional bounded context offsets."""
    def __init__(self,input_dim,n_states,n_relations,mode='static',hidden=32):
        super().__init__();self.mode=mode;self.n_relations=n_relations
        width=2 if mode=='global' else n_relations+2
        initial=torch.tensor([-2.]+[0.]*(width-1)).repeat(n_states,1)
        self.base=nn.Parameter(initial)
        if mode=='context':
            self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,width))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        elif mode not in ['static','global']:raise ValueError(mode)

    def forward(self,x,state,available):
        logits=self.base[state];penalty=x.new_tensor(0.)
        if self.mode=='context':
            delta=2*torch.tanh(self.net(x)/2);logits=logits+delta;penalty=(delta**2).mean()
        prob=torch.softmax(logits,dim=-1)
        if self.mode=='global':
            shared=torch.cat([.98*prob[:,:1],prob.new_zeros((len(prob),self.n_relations))],dim=1)
        else:shared=.98*prob[:,:-1]
        loads=torch.sqrt(torch.cat([shared,1-shared.sum(1,keepdim=True)],dim=1))
        return mask_unavailable_relations(loads,available),penalty


def mask_unavailable_relations(loadings,available):
    """Move missing relation weight into private noise; never share unknown IDs."""
    local=loadings[:,1:-1]
    private=torch.sqrt(loadings[:,-1:]**2+((1-available)*local**2).sum(-1,keepdim=True))
    return torch.cat([loadings[:,:1],local*available,private],dim=-1)


def joint_bernoulli_probabilities(p_i,p_j,rho,n_quadrature=32):
    """Plackett integral for bivariate normal CDF, differentiable in rho.

    Output order: (00, 01, 10, 11). Float64 integration avoids cancellation for
    rare events. rho is constrained to [0,.98] by the factor representation.
    """
    p_i=p_i.double();p_j=p_j.double();rho=rho.double()
    z_i=torch.special.ndtri(p_i);z_j=torch.special.ndtri(p_j)
    nodes,weights=np.polynomial.legendre.leggauss(n_quadrature)
    nodes=torch.as_tensor((nodes+1)/2,device=rho.device,dtype=torch.float64)
    weights=torch.as_tensor(weights/2,device=rho.device,dtype=torch.float64)
    r=rho[:,None]*nodes
    variance=1-r*r
    exponent=-(z_i[:,None]**2-2*r*z_i[:,None]*z_j[:,None]+z_j[:,None]**2)/(2*variance)
    density=torch.exp(exponent)/(2*torch.pi*torch.sqrt(variance))
    both=p_i*p_j+rho*(density*weights).sum(-1)
    result=torch.stack([1-p_i-p_j+both,p_j-both,p_i-both,both],dim=-1)
    return result.clamp_min(1e-12)


def common_shock_probabilities(load_i,load_j,p_i,p_j,same_group):
    """Classical exponential common-shock/Marshall–Olkin factor construction.

    Squared loadings are nonnegative weights summing to one. Each node's latent
    minimum exponential race is Exp(1), preserving every supplied marginal.
    This is a dependence model, not an identified causal failure mechanism.
    """
    w_i=load_i.double()**2;w_j=load_j.double()**2
    p_i=p_i.double();p_j=p_j.double()
    h_i=-torch.log1p(-p_i);h_j=-torch.log1p(-p_j)
    if same_group.ndim==1:same_group=same_group[:,None]
    overlap=torch.minimum(w_i[:,0]*h_i,w_j[:,0]*h_j)+(same_group*torch.minimum(w_i[:,1:-1]*h_i[:,None],w_j[:,1:-1]*h_j[:,None])).sum(-1)
    neither=torch.exp(-h_i-h_j+overlap)
    both=p_i+p_j-1+neither
    return torch.stack([neither,1-p_i-neither,1-p_j-neither,both],dim=-1).clamp_min(1e-12)


def pair_nll(load_i,load_j,p_i,p_j,same_group,y_i,y_j,family='gaussian'):
    if family=='gaussian':
        if same_group.ndim==1:same_group=same_group[:,None]
        rho=load_i[:,0]*load_j[:,0]+(same_group*load_i[:,1:-1]*load_j[:,1:-1]).sum(-1)
        prob=joint_bernoulli_probabilities(p_i,p_j,rho)
    elif family=='commonshock':prob=common_shock_probabilities(load_i,load_j,p_i,p_j,same_group)
    else:raise ValueError(family)
    label=(2*y_i+y_j).long()
    return -torch.log(prob.gather(1,label[:,None]).squeeze(1))
