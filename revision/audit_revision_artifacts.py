"""Check reported evidence, final-period alignment, and both immutable freezes."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from revision.common import ROOT, digest, write_json
from revision.fresh_protocol import verify_fresh_protocol
from otg.protocol import verify_final_protocol


def main():
    original, original_hash = verify_final_protocol(ROOT)
    fresh = verify_fresh_protocol()
    generated = ROOT/'paper_revision/generated'
    manifest = json.loads((generated/'manifest.json').read_text())
    assert manifest['script_sha256'] == digest(ROOT/'revision/build_revision_artifacts.py')
    for record in manifest['sources']:
        assert digest(ROOT/record['path']) == record['sha256'], record['path']
    macro = pd.read_csv(generated/'count_macro.csv')
    origins = pd.read_csv(generated/'count_origin_scores.csv')
    expected_origins = {'borg': 28, 'bpi': 61, 'bpi2012': 28}
    for (ds, method), rows in origins.groupby(['dataset', 'method']):
        assert len(rows) == expected_origins[ds], (ds, method, len(rows))
        assert rows.cutoff_us.nunique() == len(rows)
    for name in ['count_macro.csv', 'joint_macro.csv', 'paired_intervals.csv']:
        d = pd.read_csv(generated/name)
        assert np.isfinite(d.select_dtypes('number').to_numpy()).all(), name
    assert macro.coverage_90.between(0, 1).all()
    assert (macro.fraction_crps >= 0).all()
    def value(ds, method, metric):
        z = macro[(macro.dataset == ds) & (macro.method == method)]
        assert len(z) == 1
        return float(z.iloc[0][metric])
    effects = {}
    for ds, model, reference in [('borg', 'static', 'independent'), ('borg', 'static', 'global'),
                                ('bpi', 'global', 'independent'), ('bpi2012', 'global', 'independent'),
                                ('bpi2012', 'early_dm', 'global')]:
        effects[f'{ds}:{model}_vs_{reference}'] = {
            metric: 100*(1-value(ds, model, metric)/value(ds, reference, metric))
            for metric in ['fraction_crps', 'buffer_cost_ratio9']}
    # Confirm the table retains the original published-in-review aggregate scores.
    for ds in ['borg', 'bpi']:
        d = pd.read_parquet(ROOT/f'reports/summary_{ds}_selected_v2_final_m512/all_scores.parquet')
        d = d[d.family == 'gaussian']
        for mode in ['independent', 'global', 'static']:
            for metric in ['fraction_crps', 'coverage_90', 'buffer_cost_ratio9']:
                original_value = d[d['mode'] == mode].groupby('cutoff_us')[metric].mean().mean()
                assert abs(original_value-value(ds, mode, metric)) < 1e-12
        residual = pd.read_parquet(ROOT/f'runs/aggregate_residual_{ds}_selected_v2/final_scores_m512.parquet')
        assert set(residual.seed) == {0, 1, 2}
    final = ROOT/'data/processed/revision_bpi2012_final_v1'
    q = pd.read_parquet(final/'queries.parquet')
    feb = pd.Timestamp('2012-02-01 12:00', tz='UTC').value//1000
    end = pd.Timestamp('2012-03-14', tz='UTC').value//1000
    fq = q[q.cutoff_us >= feb]
    assert len(fq) == 23749 and fq.cutoff_us.nunique() == 28
    assert fq.cutoff_us.max()+14*86400_000_000 < end
    fbase = ROOT/'runs/revision_bpi2012_final_predictions_v1'
    for path in sorted(fbase.glob('*cdf.npy')):
        p = np.load(path)
        assert p.shape == (len(q), 4) and np.isfinite(p).all()
        assert np.all((p >= 0) & (p <= 1)) and np.all(np.diff(p, axis=1) >= 0)
    fits = json.loads((fbase/'online_fits.json').read_text())
    assert len(fits) == q.cutoff_us.nunique() == 41
    for fit in fits:
        assert all(t+14*86400_000_000 <= fit['cutoff_us'] for t in fit['fit_origins'])
    fresh_scores = pd.read_parquet(ROOT/'runs/revision_bpi2012_final_evaluation_v1/scores.parquet')
    for _, rows in fresh_scores.groupby(['calibration', 'mode', 'seed']):
        assert set(rows.cutoff_us) == set(fq.cutoff_us)
        assert len(rows) == 28*4*2
    assert set(fresh_scores.seed) == {0, 1, 2}
    # Compare exact and sampled versions of the SAME newly fitted DM controls.
    mc = {}
    for ds in ['borg', 'bpi']:
        d = pd.read_parquet(ROOT/f'runs/revision_{ds}_updated_dm_profiles_v1/count_scores.parquet')
        estimate = d[d['mode'] == 'development_fitted_dm'].groupby('cutoff_us').fraction_crps.mean().mean()
        mc[ds] = {'exact': value(ds, 'recent_dm', 'fraction_crps'), 'sampled_512_three_seeds': float(estimate),
                  'absolute_difference': abs(float(estimate)-value(ds, 'recent_dm', 'fraction_crps'))}
        assert mc[ds]['absolute_difference'] < .001
    ess = {}
    for ds, path in [('bpi', 'runs/revision_bpi_updated_dm_profiles_v1/importance_diagnostics.csv'),
                     ('bpi2012', 'runs/revision_bpi2012_final_evaluation_v1/importance_diagnostics.csv')]:
        d = pd.read_csv(ROOT/path)
        assert d.ess_fraction.between(0, 1).all() and (d.pool == 8192).all()
        ess[ds] = {'mean': float(d.ess_fraction.mean()), 'min': float(d.ess_fraction.min())}
    report = {'status': 'passed', 'original_protocol_sha256': original_hash,
              'original_frozen_assets': len(original['frozen_files']), 'fresh_frozen_assets': len(fresh['files']),
              'fresh_protocol_sha256': digest(ROOT/'revision/FRESH_FINAL_PROTOCOL.json'),
              'generated_inputs_verified': len(manifest['sources']), 'origins': expected_origins,
              'fresh_scored_queries': len(fq), 'online_fits_maturity_verified': len(fits),
              'exact_vs_sampled_dm': mc, 'importance_effective_fraction': ess,
              'percent_improvement_positive_is_better': effects,
              'reporting_correction': 'Original historical residual results use three seeds. Earlier one-seed caption was erroneous.',
              'scope': 'Evidence consistency and implementation audit, not independent peer review or proof of generalization.',
              'audit_script_sha256': digest(__file__)}
    write_json(ROOT/'reports/revision_artifact_audit.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
