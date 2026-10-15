import numpy as np
import pytest
from otg.cohort_scores import projections
from otg.torch_scenarios import projected_counts_torch


@pytest.mark.parametrize('family',['gaussian','commonshock'])
def test_torch_streamed_counts_shared_noise_and_margins(family):
    n=10;rel=np.zeros((n,1),np.int64);states=np.ones(n);cdf=np.tile([.2,.5,.8],(n,1));loads=np.tile([0,1,0],(n,1))
    projection=projections(states,rel,['job'])
    counts=projected_counts_torch(cdf,loads,rel,projection,100000,42,family,'cpu',chunk_size=3)
    for values in counts.values():
        assert np.all((values==0)|(values==n))
        np.testing.assert_allclose(values.mean(0)/n,cdf[:1],atol=.005)


def test_torch_counts_match_deterministic_nested_trajectories():
    n=13;rel=np.array([[3]]*6+[[4]]*6+[[-1]]);state=np.array([1,2]*6+[1]);p=np.array([[0,1,1]]*6+[[0,0,1]]*7)
    projection=projections(state,rel,['job']);loads=np.tile([0,0,1],(n,1))
    counts=projected_counts_torch(p,loads,rel,projection,7,9,'gaussian','cpu',chunk_size=4)
    for name,v in projection.items():
        truth=np.stack([p[v['code']==i].sum(0) for i in range(len(v['size']))])
        np.testing.assert_array_equal(counts[name],np.broadcast_to(truth,counts[name].shape))
