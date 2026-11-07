"""Scientific invariants: proper scoring, valid joint paths, and graph controls."""
import numpy as np
import torch
from extension.models import CohortNet, local_codes, relation_summary
from extension.run_pilots import crps, scenario
from extension.prepare_pilots import interval_events
import pandas as pd


def test_crps_matches_pairwise_definition():
    x=np.array([[0.,2.],[1.,1.],[4.,2.]])
    y=np.array([2.,2.])
    exact=np.abs(x-y).mean(0)-.5*np.abs(x[:,None]-x[None]).mean((0,1))
    np.testing.assert_allclose(crps(x,y),exact)


def test_scenario_preserves_marginals_and_monotonicity():
    p=np.array([[.1,.3,.6,.9],[.2,.4,.7,.95],[.05,.1,.2,.3]])
    r=np.array([[1,2],[1,3],[-1,3]])
    for df in [None,4]:
        s=scenario(p,r,[.1,.2,.15],60000,7,df)
        assert np.all(np.diff(s.astype(int),axis=-1)>=0)
        np.testing.assert_allclose(s.mean(0),p,atol=.008)


def test_missing_relations_do_not_create_spurious_edges():
    r=np.array([[-1],[-1],[4],[4]])
    codes=local_codes(r)
    assert codes[0,0]!=codes[1,0]
    x=np.array([[1.],[9.],[3.],[5.]])
    result=relation_summary(x,r)
    np.testing.assert_array_equal(result[:,-2],[1,9,4,4])


def test_shuffling_zero_weight_relations_changes_no_samples():
    p=np.tile([.1,.2,.4,.8],(10,1));r=np.array([[1],[1],[1],[2],[2],[2],[2],[3],[-1],[-1]])
    np.testing.assert_array_equal(scenario(p,r,[.2,0],100,7),scenario(p,r,[.2,0],100,7,shuffled=True))


def test_shared_factor_matches_binary_gaussian_probability():
    p=np.full((2,4),.5);r=np.zeros((2,1),dtype=int)
    s=scenario(p,r,[0,.5],60000,18)
    expected=.25+np.arcsin(.5)/(2*np.pi)
    assert abs(np.mean(s[:,0,0]&s[:,1,0])-expected)<.007


def test_graph_output_is_permutation_equivariant():
    torch.manual_seed(2)
    x=torch.randn(7,3);r=torch.tensor([[0],[0],[1],[1],[1],[2],[3]])
    h=torch.zeros(7,1,8,5)
    perm=torch.tensor([3,0,6,1,4,5,2])
    for arch in ["mlp","set","graph","temporal_attention"]:
        net=CohortNet(3,1,arch).eval()
        torch.testing.assert_close(net(x,r,h)[perm],net(x[perm],r[perm],h[perm]),atol=2e-6,rtol=2e-6)


def test_future_end_time_is_censored_at_preparation_boundary():
    f=pd.DataFrame({"id":[1,2,3],"start":[1,2,20],"end":[30,5,40],"r":[1,1,2]})
    a,_,_,_=interval_events(f,[],[],["r"],10)
    f.loc[0,"end"]=1000
    b,_,_,_=interval_events(f,[],[],["r"],10)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==2 and np.isinf(a.iloc[0].end)
