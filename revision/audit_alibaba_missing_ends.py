"""Schema-level audit only: first two days, no eventual status/future duration."""
import sys, tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from revision.common import ROOT, write_json, digest


def main():
    src = ROOT/'data/raw/revision_alibaba2018/batch_task.tar.gz'
    cap = 2*86400
    pieces = []
    with tarfile.open(src, 'r:gz') as archive:
        with archive.extractfile('batch_task.csv') as f:
            for chunk in pd.read_csv(f, header=None, usecols=[5, 6], names=list(range(9)), chunksize=500000):
                q = chunk[(chunk[5] > 0) & (chunk[5] < cap)].copy()
                pieces.append(pd.DataFrame({'start': q[5], 'missing_end': q[6].isna() | (q[6] <= 0),
                    'prefix_end': np.where((q[6] > 0) & (q[6] < cap), q[6], cap)}))
    d = pd.concat(pieces, ignore_index=True)
    cohorts = []
    for hour in [24, 30, 36, 42, 47]:
        a = d[(d.start <= hour*3600) & (d.prefix_end > hour*3600)]
        cohorts.append({'hour': hour, 'candidate_active': len(a), 'no_recorded_end': int(a.missing_end.sum()),
                        'missing_end_fraction': float(a.missing_end.mean())})
    result = {'scope': 'Only positive starts before day 2; final status and actual end values at/after day 2 are not retained or summarized.',
              'rows': len(d), 'no_recorded_end': int(d.missing_end.sum()), 'cohorts': cohorts,
              'source_sha256': digest(src), 'script_sha256': digest(__file__),
              'decision': 'Defer: no endpoint-specific evidence establishes that absent ends imply continuous running until archive boundary. Do not label them as observed non-exits.'}
    out = ROOT/'reports/revision_alibaba_missing_ends.json'
    if out.exists():
        raise FileExistsError(out)
    write_json(out, result)
    print(result)


if __name__ == '__main__':
    main()
