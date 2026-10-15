import numpy as np
from otg.history import last_event_indices


def test_temporal_relation_retrieval_rejects_future_and_unknown():
    ids = np.array([5, 7, 5, 5, 7, 9])
    times = np.array([1, 1, 3, 3, 5, 2])
    actual = last_event_indices(ids, times, np.array([5, 7, 9, -1, 10]),
                                np.array([3, 4, 1, 10, 9]), k=3)
    np.testing.assert_array_equal(actual, [[0, 2, 3], [-1, -1, 1],
                                          [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])


def test_retrieval_matches_explicit_history_with_unsorted_events():
    rng = np.random.default_rng(19)
    ids, times = rng.integers(0, 20, 200), rng.integers(0, 100, 200)
    qi, qt = rng.integers(0, 25, 50), rng.integers(0, 100, 50)
    actual = last_event_indices(ids, times, qi, qt, 7)
    for row, (i, t) in enumerate(zip(qi, qt)):
        eligible = np.flatnonzero((ids == i) & (times <= t))
        chosen = eligible[np.argsort(times[eligible], kind='stable')][-7:]
        expected = np.full(7, -1)
        if len(chosen): expected[-len(chosen):] = chosen
        np.testing.assert_array_equal(actual[row], expected)
