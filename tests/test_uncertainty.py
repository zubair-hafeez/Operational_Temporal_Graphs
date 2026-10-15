import numpy as np
from otg.uncertainty import paired_moving_block_interval


def test_paired_block_interval_preserves_constant_improvement():
    r=paired_moving_block_interval(np.full(28,-.03),4,repeats=500)
    np.testing.assert_allclose([r['mean_difference'],r['ci95_low'],r['ci95_high']],-.03)


def test_block_interval_is_reproducible_and_retains_time_variation():
    d=np.sin(np.arange(40)/5)
    a=paired_moving_block_interval(d,7,repeats=1000,seed=31)
    assert a==paired_moving_block_interval(d,7,repeats=1000,seed=31)
    assert a['ci95_low']<a['mean_difference']<a['ci95_high']
