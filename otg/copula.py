"""Positive shared-factor Gaussian copula with exact Bernoulli marginals.

This is a model family built from established factor-copula ingredients. Novelty
must be assessed through the full operational formulation and empirical evidence.
"""
import numpy as np
import torch
from torch import nn


class FactorLoadings(nn.Module):
    def __init__(self,input_dim,mode='context',hidden=32):
        super().__init__();self.mode=mode
        if mode=='context':
            self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,3))
            nn.init.zeros_(self.net[-1].weight)
            self.net[-1].bias.data.copy_(torch.tensor([-2.,0.,0.]))
        elif mode=='static':self.logits=nn.Parameter(torch.tensor([-2.,0.,0.]))
        elif mode=='global':self.logits=nn.Parameter(torch.tensor([-2.,0.]))
        else:raise ValueError(mode)

    def forward(self,x):
        if self.mode=='context':prob=torch.softmax(self.net(x),dim=-1)
        elif self.mode=='static':prob=torch.softmax(self.logits,dim=-1).expand(len(x),-1)
        else:
            p=torch.softmax(self.logits,dim=-1).expand(len(x),-1)
            prob=torch.stack([p[:,0],torch.zeros_like(p[:,0]),p[:,1]],dim=-1)
        # Keep individual variance >=0.02. Every node's latent variance is one.
        shared=.98*prob[:,:2]
        individual=1-shared.sum(-1,keepdim=True)
        return torch.sqrt(torch.cat([shared,individual],dim=-1))


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
    overlap=torch.minimum(w_i[:,0]*h_i,w_j[:,0]*h_j)+same_group*torch.minimum(w_i[:,1]*h_i,w_j[:,1]*h_j)
    neither=torch.exp(-h_i-h_j+overlap)
    both=p_i+p_j-1+neither
    return torch.stack([neither,1-p_i-neither,1-p_j-neither,both],dim=-1).clamp_min(1e-12)


def pair_nll(load_i,load_j,p_i,p_j,same_group,y_i,y_j,family='gaussian'):
    if family=='gaussian':
        rho=load_i[:,0]*load_j[:,0]+same_group*load_i[:,1]*load_j[:,1]
        prob=joint_bernoulli_probabilities(p_i,p_j,rho)
    elif family=='commonshock':prob=common_shock_probabilities(load_i,load_j,p_i,p_j,same_group)
    else:raise ValueError(family)
    label=(2*y_i+y_j).long()
    return -torch.log(prob.gather(1,label[:,None]).squeeze(1))
