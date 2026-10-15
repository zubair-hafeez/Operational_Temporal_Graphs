"""Independent temporal survival encoders with a coherent categorical head.

The Mixer uses the published token/channel mixing idea, implemented here for
typed entity/relation histories and first-exit forecasting. It is not an exact
reproduction of GraphMixer's link-prediction model or benchmark.
"""
import torch
from torch import nn


class MixerBlock(nn.Module):
    def __init__(self, length, width):
        super().__init__()
        self.token_norm=nn.LayerNorm(width)
        self.channel_norm=nn.LayerNorm(width)
        self.token=nn.Sequential(nn.Linear(length,max(8,length//2)),nn.GELU(),nn.Linear(max(8,length//2),length))
        self.channel=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
    def forward(self,x):
        x=x+self.token(self.token_norm(x).transpose(-1,-2)).transpose(-1,-2)
        return x+self.channel(self.channel_norm(x))


class TemporalSurvival(nn.Module):
    def __init__(self,input_dim,vocab_sizes,continuous_dim,n_scopes,length=16,width=64,architecture='mixer'):
        super().__init__();self.architecture=architecture;self.n_scopes=n_scopes;self.width=width
        self.tabular=nn.Sequential(nn.Linear(input_dim,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
        if architecture!='mlp':
            self.category=nn.ModuleList([nn.Embedding(n,8) for n in vocab_sizes])
            self.scope=nn.Embedding(n_scopes,8)
            self.register_buffer('frequencies',torch.logspace(0,-8,16))
            self.token_input=nn.Linear(16+8+continuous_dim+17,width)
            if architecture=='mixer':self.encoder=nn.Sequential(MixerBlock(length,width),MixerBlock(length,width),nn.LayerNorm(width))
            elif architecture=='gru':self.encoder=nn.GRU(width,width,batch_first=True)
            else:raise ValueError(architecture)
        self.head=nn.Sequential(nn.Linear(width*(1+(n_scopes if architecture!='mlp' else 0)),width),nn.GELU(),nn.Linear(width,5))

    def forward(self,x,categories=None,values=None,lag_seconds=None,mask=None):
        tabular=self.tabular(x)
        if self.architecture=='mlp':return self.head(tabular)
        batch,scopes,length=mask.shape
        cats=torch.cat([emb(categories[...,k]) for k,emb in enumerate(self.category)],dim=-1)
        scope=self.scope(torch.arange(scopes,device=x.device))[None,:,None,:].expand(batch,scopes,length,-1)
        time=torch.cat([torch.log1p(lag_seconds)[...,None]/10,torch.cos(lag_seconds[...,None]*self.frequencies)],dim=-1)
        tokens=self.token_input(torch.cat([cats,scope,values,time],dim=-1))*mask[...,None]
        tokens=tokens.reshape(batch*scopes,length,self.width);valid=mask.reshape(batch*scopes,length)
        if self.architecture=='mixer':
            encoded=self.encoder(tokens)
            encoded=(encoded*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        else:
            # Left padding permits a fixed-size batch; zero-history scopes are
            # explicitly masked after the recurrent encoder.
            _,hidden=self.encoder(tokens);encoded=hidden[-1]
        encoded=encoded*valid.any(1)[:,None]
        return self.head(torch.cat([tabular,encoded.reshape(batch,-1)],dim=1))


def logits_to_cdf(logits):
    return logits.softmax(-1).cumsum(-1)[...,:-1]
