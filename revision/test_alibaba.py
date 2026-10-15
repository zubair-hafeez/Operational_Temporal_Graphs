import numpy as np
from revision.alibaba_data import prefix_features


def test_future_task_ends_do_not_change_prefix_features():
    starts=np.array([10,20,30,120,200]);ends=np.array([40,400,600,900,1000]);jobs=np.array([0,0,1,1,2]);types=np.array([1,2,3,4,5])
    ids,x=prefix_features(starts,ends,jobs,types,100)
    perturbed=np.array([40,10000,101,100000,201]);ids2,x2=prefix_features(starts,perturbed,jobs,types,100)
    np.testing.assert_array_equal(ids,[1,2]);np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(x,x2)


def test_unseen_task_attributes_cannot_influence_prefix_features():
    starts=np.array([10,20,200]);ends=np.array([40,400,600]);jobs=np.array([0,0,1]);types=np.array([1,2,3])
    ids,x=prefix_features(starts,ends,jobs,types,100);jobs[-1]=999;types[-1]=12
    ids2,x2=prefix_features(starts,ends,jobs,types,100)
    np.testing.assert_array_equal(ids,ids2);np.testing.assert_array_equal(x,x2)
