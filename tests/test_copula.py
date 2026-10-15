import numpy as np
import torch
from scipy.stats import multivariate_normal,norm
from otg.copula import FactorLoadings,joint_bernoulli_probabilities,pair_nll,common_shock_probabilities,mask_unavailable_relations


def test_joint_probabilities_have_fixed_marginals_and_match_integral():
    p=torch.tensor([.01,.2,.7,.99],dtype=torch.float64)
    q=torch.tensor([.3,.8,.4,.99],dtype=torch.float64)
    rho=torch.tensor([0.,.2,.7,.95],dtype=torch.float64,requires_grad=True)
    prob=joint_bernoulli_probabilities(p,q,rho)
    np.testing.assert_allclose(prob.detach().sum(-1),1,atol=1e-10)
    np.testing.assert_allclose((prob[:,2]+prob[:,3]).detach(),p,atol=1e-10)
    np.testing.assert_allclose((prob[:,1]+prob[:,3]).detach(),q,atol=1e-10)
    for i in range(4):
        reference=multivariate_normal.cdf([norm.ppf(p[i]),norm.ppf(q[i])],cov=[[1,float(rho[i].detach())],[float(rho[i].detach()),1]],maxpts=1000000,abseps=1e-8)
        assert abs(prob[i,3].item()-reference)<5e-5
    prob[:,3].sum().backward();assert torch.isfinite(rho.grad).all() and (rho.grad>0).all()


def test_loadings_define_positive_semidefinite_unit_diagonal_covariance():
    torch.manual_seed(5);x=torch.randn(5,3);model=FactorLoadings(3)
    loads=model(x).detach().numpy();groups=np.array([0,0,1,1,2])
    factor=np.zeros((5,1+3+5));factor[:,0]=loads[:,0]
    factor[np.arange(5),1+groups]=loads[:,1]
    factor[np.arange(5),4+np.arange(5)]=loads[:,2]
    cov=factor@factor.T
    np.testing.assert_allclose(np.diag(cov),1,atol=1e-6)
    assert np.linalg.eigvalsh(cov).min()>0


def test_independent_pair_likelihood_and_order():
    load=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    p=torch.tensor([.2,.7]);q=torch.tensor([.3,.4])
    actual=pair_nll(load,load,p,q,torch.ones(2),torch.tensor([1,0]),torch.tensor([0,1]))
    np.testing.assert_allclose(actual.numpy(),-np.log([.2*.7,.3*.4]),rtol=1e-6)


def test_common_shock_marginals_and_monte_carlo_joint_probability():
    weights=np.array([[.2,.6,.2],[.1,.4,.5]])
    load=torch.tensor(np.sqrt(weights));p=torch.tensor([.3]);q=torch.tensor([.4])
    prob=common_shock_probabilities(load[:1],load[1:],p,q,torch.ones(1)).numpy()[0]
    np.testing.assert_allclose(prob.sum(),1,atol=1e-8)
    np.testing.assert_allclose(prob[2]+prob[3],p,atol=1e-8)
    np.testing.assert_allclose(prob[1]+prob[3],q,atol=1e-8)
    rng=np.random.default_rng(123);n=200000
    shared=rng.exponential(size=(n,2));private=rng.exponential(size=(n,2))
    hazard=-np.log1p(-np.array([.3,.4]));y=np.zeros((n,2),bool)
    for i in range(2):y[:,i]=(shared[:,0]<weights[i,0]*hazard[i])|(shared[:,1]<weights[i,1]*hazard[i])|(private[:,i]<weights[i,2]*hazard[i])
    empirical=np.bincount(2*y[:,0].astype(int)+y[:,1],minlength=4)/n
    np.testing.assert_allclose(empirical,prob,atol=.003)


def test_common_shock_independence_and_complete_sharing_limits():
    p=torch.tensor([.1,.7]);q=torch.tensor([.4,.3])
    private=torch.tensor([[0.,0.,1.],[0.,0.,1.]])
    independent=common_shock_probabilities(private,private,p,q,torch.ones(2))
    np.testing.assert_allclose(independent[:,3],p*q,atol=1e-7)
    shared=torch.tensor([[0.,1.,0.],[0.,1.,0.]])
    full=common_shock_probabilities(shared,shared,p,q,torch.ones(2))
    np.testing.assert_allclose(full[:,3],torch.minimum(p,q),atol=1e-7)


def test_crossed_relations_and_unknown_weight_preserve_marginals():
    model=FactorLoadings(3,n_relations=2);x=torch.randn(4,3)
    loads=model(x);available=torch.tensor([[1.,1.],[1.,0.],[1.,1.],[0.,0.]])
    masked=mask_unavailable_relations(loads,available)
    np.testing.assert_allclose((masked.detach()**2).sum(1),1,atol=1e-6)
    assert masked[1,2]==0 and (masked[3,1:3]==0).all()
    loss=pair_nll(masked[:2],masked[2:],torch.tensor([.2,.3]),torch.tensor([.4,.5]),torch.tensor([[1.,0.],[0.,0.]]),torch.tensor([1,0]),torch.tensor([0,1])).mean()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
