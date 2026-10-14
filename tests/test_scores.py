import numpy as np
from otg.scores import ensemble_crps,interval_statistics


def test_fair_crps_matches_explicit_distinct_pair_formula():
    x=np.array([[1.,2.],[3.,1.],[4.,9.]])
    y=np.array([2.,5.]);m=len(x)
    pair=np.abs(x[:,None]-x[None,:]).sum(axis=(0,1))/(2*m*(m-1))
    np.testing.assert_allclose(ensemble_crps(x,y),np.abs(x-y).mean(0)-pair)


def test_interval_score_penalizes_misses():
    x=np.ones((10,2));cov,width,score=interval_statistics(x,[1,2])
    np.testing.assert_array_equal(cov,[True,False])
    np.testing.assert_array_equal(width,[0,0]);np.testing.assert_allclose(score,[0,20])
