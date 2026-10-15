import numpy as np
import pytest
from otg.scenarios import factor_uniform_chunks,first_exit_bins,exponential_race_time


@pytest.mark.parametrize('family',['gaussian','commonshock'])
def test_shared_factor_across_chunks_and_correct_marginals(family):
    loads=np.array([[0,1,0],[0,1,0],[0,0,1]],np.float32)
    rel=np.array([[10],[10],[-1]])
    # One million draws keep a .003 check well beyond six standard errors.
    chunks=list(factor_uniform_chunks(loads,rel,1000000,np.random.default_rng(42),family,chunk_size=1))
    u=np.concatenate([part[2] for part in chunks],axis=1)
    np.testing.assert_allclose(u[:,0],u[:,1])
    np.testing.assert_allclose((u<.3).mean(0),.3,atol=.003)
    assert np.corrcoef(u[:,0],u[:,2])[0,1]<.02
    cdf=np.array([[.1,.3,.6,.8]]*3)
    bins=first_exit_bins(cdf,u)
    for k in range(4):np.testing.assert_allclose((bins<=k).mean(0),cdf[:,k],atol=.003)


def test_unknown_shared_factor_and_nonmonotone_cdf_rejected():
    with pytest.raises(ValueError):list(factor_uniform_chunks([[0,1,0]],[[-1]],2,np.random.default_rng(0)))
    with pytest.raises(ValueError):first_exit_bins([[.4,.3]],np.array([[.2]]))


def test_zero_weight_factor_is_absent_even_for_zero_draw():
    result=exponential_race_time(np.array([[0.,1.]],np.float32),np.array([0.,.5],np.float32))
    assert np.isinf(result[0,0]) and result[0,1]==2 and not np.isnan(result).any()
