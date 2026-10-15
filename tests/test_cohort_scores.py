import numpy as np
from otg.cohort_scores import cohort_scores, projections
from otg.scores import ensemble_crps


def test_streaming_projection_scores_match_explicit_trajectories(monkeypatch):
    # Two relation groups, one unknown entity, and a group spanning chunks.
    n, draws, h = 13, 8, 3
    relation = np.array([[3]] * 6 + [[9]] * 6 + [[-1]])
    state = np.array([1, 2] * 6 + [1])
    cdf = np.tile([.2, .5, .8], (n, 1))
    u = np.random.default_rng(71).uniform(size=(draws, n))
    label = np.random.default_rng(4).uniform(size=n)[:, None] <= cdf
    def deterministic_chunks(*args, **kwargs):
        for lo in range(0, n, 4):
            yield lo, min(n, lo+4), u[:, lo:lo+4]
    monkeypatch.setattr('otg.cohort_scores.factor_uniform_chunks', deterministic_chunks)
    rows = cohort_scores(cdf, np.zeros((n, 3)), relation, state,
                         label.astype(int), ['job'], draws, 0)
    scenarios = u[:, :, None] <= cdf[None, :, :]
    for name, projection in projections(state, relation, ['job']).items():
        counts = np.stack([scenarios[:, projection['code'] == k].sum(1)
                           for k in range(len(projection['size']))], axis=1)
        truth = np.stack([label[projection['code'] == k].sum(0)
                          for k in range(len(projection['size']))])
        expected = (ensemble_crps(counts, truth) / projection['size'][:, None]).mean(0)
        actual = [r['fraction_crps'] for r in rows if r['projection'] == name]
        np.testing.assert_allclose(actual, expected)
        assert all(r['groups'] == 2 for r in rows if r['projection'] == name)


def test_perfect_deterministic_trajectories_have_zero_loss():
    cdf = np.array([[0, 1, 1]] * 10, dtype=float)
    loads = np.tile([0, 0, 1], (10, 1))
    rows = cohort_scores(cdf, loads, np.zeros((10, 1), int),
                         np.ones(10), cdf, ['resource'], 8, 3)
    for row in rows:
        for key in ['fraction_crps', 'trajectory_energy',
                    'buffer_cost_ratio3', 'buffer_cost_ratio9',
                    'simultaneous_alarm_fraction_crps']:
            assert row[key] == 0
        assert row['coverage_90'] == 1
