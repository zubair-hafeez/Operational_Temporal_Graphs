import pytest
import torch
from otg.temporal import TemporalSurvival,logits_to_cdf


@pytest.mark.parametrize('architecture',['mlp','mixer','gru'])
def test_temporal_survival_normalization_and_backward(architecture):
    torch.manual_seed(3)
    model=TemporalSurvival(7,[10,4],2,3,length=4,width=16,architecture=architecture)
    x=torch.randn(5,7);cats=torch.zeros(5,3,4,2,dtype=torch.long)
    values=torch.randn(5,3,4,2);lag=torch.rand(5,3,4)*100;mask=torch.ones(5,3,4,dtype=torch.bool);mask[0]=False
    logits=model(x,cats,values,lag,mask);cdf=logits_to_cdf(logits)
    assert cdf.shape==(5,4) and torch.isfinite(cdf).all()
    assert (cdf.diff(dim=-1)>=0).all() and (cdf>=0).all() and (cdf<=1).all()
    torch.nn.functional.cross_entropy(logits,torch.arange(5)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
