"""Finish presentation-only artifacts without changing the frozen analysis.

Run after build_paper_artifacts.py and audit_final_artifacts.py. The original
analysis remains frozen; this script adds numerical audit prose and clarifies
two table captions. It does not alter any model, score, selection or interval.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/generated'


def main():
    audit = json.loads((ROOT / 'reports/final_artifact_audit.json').read_text())
    if audit['status'] != 'passed':
        raise ValueError('Final consistency audit must pass first')
    mc = pd.read_csv(OUT / 'monte_carlo_sensitivity.csv')
    runtime = pd.read_csv(OUT / 'runtimes.csv')
    stable = all(
        g.sort_values('crps_512')['mode'].tolist()
        == g.sort_values('crps_1024')['mode'].tolist()
        for _, g in mc.groupby(['dataset', 'family'])
    )
    if not stable:
        raise ValueError('Revise numerical audit prose: Monte Carlo ranks changed')
    b = runtime[runtime.dataset == 'borg']
    w = runtime[runtime.dataset == 'bpi']
    maximum = mc.difference.abs().max()
    gpu_gib = b.peak_torch_allocated_bytes.max() / 2**30
    host_gib = b.peak_host_rss_kib.max() / 2**20
    prose = rf'''\subsection{{Numerical sensitivity and execution}}
Doubling seed-0 scenario draws from 512 to 1,024 changes macro CRPS by at
most {maximum:.6f} across the two applications, families, and five controls.
Within each application and family, the ordering of the five controls is
unchanged. This checks the reported CRPS comparisons against finite-ensemble
variation; it does not replace uncertainty over future forecast origins.

On the shared RTX A6000 host, each 512-draw Borg evaluation of all five
controls across 28 origins takes {b.elapsed_s.min():.1f}--{b.elapsed_s.max():.1f}
seconds wall time, including sampling and scoring. The corresponding BPI
evaluation across 61 origins takes {w.elapsed_s.min():.2f}--{w.elapsed_s.max():.2f}
seconds. Borg peak allocated PyTorch memory is at most {gpu_gib:.2f}\,GiB
per evaluation process and peak host resident memory is {host_gib:.2f}\,GiB.
These are evaluation costs with individual CDFs already computed, measured
while other evaluations share the device; they exclude data reconstruction,
training, and individual-model inference. They are not an isolated latency
benchmark.

The final artifact audit verifies all {audit['frozen_files_verified']} frozen
file hashes, every final CDF and label trajectory, the exact frozen feature
transform, and the expected origins and comparisons. No frozen model or
evaluation file was modified during final evaluation. The separately audited
lossless timestamp correction is described in the appendix.
'''
    (OUT / 'numerical_audits.tex').write_text(prose)
    main_table = OUT / 'main_table.tex'
    main_table.write_text(main_table.read_text().replace(
        'Frozen final evaluation, averaged over three seeds and forecast origins.',
        'Frozen final evaluation, averaged over forecast origins. Fitted dependence models use three seeds; the historical-residual sampler uses one fixed seed.'
    ))
    supplement = OUT / 'supplementary_tables.tex'
    supplement.write_text(supplement.read_text().replace(
        'the full CSV also includes every family/mode and seed.',
        'the projection CSV includes all families and modes; the released origin-score tables retain individual seeds.'
    ))
    records = []
    for p in [OUT / 'monte_carlo_sensitivity.csv', OUT / 'runtimes.csv',
              ROOT / 'reports/final_artifact_audit.json']:
        records.append({'path': str(p.relative_to(ROOT)),
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    report = {'status': 'complete', 'crps_order_unchanged': stable,
              'max_abs_crps_change': maximum, 'inputs': records,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'caption_changes': 'Clarify residual seed count and origin-score versus projection CSV granularity; no numerical changes.'}
    (OUT / 'presentation_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
