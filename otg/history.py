"""Outcome-free retrieval of timestamp-bounded entity and relation histories."""
import numpy as np


def last_event_indices(event_ids, event_times, query_ids, query_times, k=16):
    """Return oldest-to-newest row indices; -1 denotes absent history.

    Ties at the cutoff are observable under the replay contract. Negative query
    IDs denote unknown relations and never retrieve a shared history.
    """
    event_ids = np.asarray(event_ids, dtype=np.int64)
    event_times = np.asarray(event_times, dtype=np.int64)
    order = np.lexsort((event_times, event_ids))
    keys = np.empty(len(order), dtype=[('id', '<i8'), ('time', '<i8')])
    keys['id'], keys['time'] = event_ids[order], event_times[order]
    result = np.full((len(query_ids), k), -1, np.int32)
    for lo in range(0, len(query_ids), 100000):
        ids = np.asarray(query_ids[lo:lo+100000], dtype=np.int64)
        query = np.empty(len(ids), dtype=keys.dtype)
        query['id'], query['time'] = ids, query_times[lo:lo+100000]
        stop = np.searchsorted(keys, query, side='right')
        positions = stop[:, None] - np.arange(k, 0, -1)
        safe = np.clip(positions, 0, max(0, len(order)-1))
        valid = (positions >= 0) & (ids[:, None] >= 0) & (keys['id'][safe] == ids[:, None])
        result[lo:lo+len(ids)] = np.where(valid, order[safe], -1)
    return result
