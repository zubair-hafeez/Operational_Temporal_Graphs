Folder organization update (18 October 2026): the latest manuscript is `paper_revision/main.pdf` (10 pages), also accessible as `LATEST_PAPER.pdf`; `paper/main.pdf` is the earlier 12-page paper. The duplicate review ZIPs were removed after all unique contents were retained. The ZIP-based commands below describe the historical package workflow. To use them, first reconstruct the appropriate package from this folder:

```bash
python scripts/organized_review.py --restore revision --output /tmp/otg_revision_review.zip
```

Use that output path in place of the retired `releases/` ZIP. Every member is restored byte-for-byte; ZIP metadata/compression and its whole-archive checksum may differ. To check current storage without recreating a ZIP, run `python scripts/organized_review.py`. Large immutable duplicate arrays use relative links; preserve links when copying the project, or dereference them when creating a standalone subset. Treat frozen inputs as immutable and create new output paths for new experiments.

--- Historical reproduction instructions ---

# Reproducing the revised collective forecasting paper

The revised manuscript is `paper_revision/main.pdf`, with source `paper_revision/main.tex`. The original study remains in `paper/`, `REPRODUCE.md` and its immutable review ZIP. This document covers additional work. Use the Python environment in `requirements-research.txt`; the reference interpreter is Python 3.10.14 with PyTorch 2.5.1+cu124. Run commands from the project root. CPU table regeneration and fresh-replication evaluation do not require CUDA.

## Fast reproduction from the local review package

Extract `releases/operational_cohorts_revision_review.zip` into a clean directory. Its `REVISION_RELEASE_MANIFEST.json` records every included file's size and SHA256. The archive includes both frozen asset sets, all revised score inputs, fitted controls, the small BPI 2012 source and derived arrays, source code and manuscript. Large original raw logs/final feature arrays, the unused Alibaba data, environment binaries and TeX caches are excluded. Included third-party data retain their original terms; the package grants no new data license.

```bash
python -m pytest tests revision/test_revision.py revision/test_exact_counts.py revision/test_alibaba.py -q
python revision/build_revision_artifacts.py
python revision/audit_revision_artifacts.py
```

These regenerate all revised CSV/TeX tables, paired block intervals and the vector figure, then verify both freezes, input hashes, origin coverage, original-score preservation, fresh CDFs, delayed-update eligibility and exact-versus-sampled direct forecasts. The figure is deterministic under the reference plotting environment. A PDF compiled by another TeX installation need not have the same binary hash.

Compile `paper_revision/main.tex` using Tectonic (from its directory, `tectonic main.tex`) or a standard LaTeX/BibTeX toolchain. The original host uses the cached compiler under `paper/tools`, which is not bundled. `revision/verify_review_package.py` verifies every archived hash, extracts to a new temporary directory, regenerates numerical artifacts, runs the tests/audit and optionally compiles with a supplied Tectonic path. It writes `reports/revision_archive_verification.json` outside the immutable ZIP.

## Evidence map

| Evidence | Location |
| --- | --- |
| Original selected-factor final scores | `reports/summary_{borg,bpi}_selected_v2_final_m512/` |
| Original residual comparator, three seeds | `runs/aggregate_residual_{borg,bpi}_selected_v2/` |
| Early OOF direct count fits | `runs/revision_{borg,bpi}_direct_counts_v2/` |
| Same-version development-fitted direct count controls | `runs/revision_{borg,bpi}_development_fitted_dm_v1/` |
| Same-version delayed calibration | `runs/revision_bpi_calibration_same_version_v2/` |
| Earlier failed cross-version calibration | `runs/revision_bpi_calibration_v1/` |
| Job/machine removals, permutation and rare-alarm diagnostic | `runs/revision_{borg,bpi}_collective_risk_v1/` |
| Original-factor/calibration joint profiles | `runs/revision_{borg,bpi}_joint_profiles_v1/` |
| Stronger direct and reconciled joint profiles | `runs/revision_{borg,bpi}_updated_dm_profiles_v1/` |
| Fresh individual and dependence models | `runs/revision_bpi2012_marginals_v1/`, `runs/revision_bpi2012_joint_v1_seed{0,1,2}/` |
| Fresh development and final scores | `runs/revision_bpi2012_{development,final}_evaluation_v1/` |
| Fresh final predictions and online-fit origins | `runs/revision_bpi2012_final_predictions_v1/` |
| Main table input hashes and complete interval sensitivities | `paper_revision/generated/manifest.json`, `paired_intervals.csv` |

Main direct-count rows use exact beta-binomial marginals for the recent development-fitted control and 512 sampled draws/three seeds for the earlier and fresh frozen controls. Joint-profile scores always use samples. Every method's risk transform uses the same original individual CDF reference. Macro scores average seeds/horizons/projections within origin, then average origins. Source files retain the finer scores.

## Recompute additional original-period experiments

These analyses are retrospective because the original final outcomes were already known. Reconstruct original data/forecasts according to `REPRODUCE.md` first; the fast archive intentionally excludes those large inputs. The commands below refuse to overwrite existing run directories, so use a separate reproduction checkout with required inputs restored and only the target outputs absent. Do not delete or mutate the original workspace's evidence.

```bash
python revision/build_borg_oof.py
python revision/run_direct_counts.py --dataset borg --version v2
python revision/run_direct_counts.py --dataset bpi --version v2
python revision/predict_bpi_calibration_window.py
python revision/run_calibration_diagnostic.py --same-version
python revision/run_development_fitted_counts.py --dataset borg
python revision/run_development_fitted_counts.py --dataset bpi
python revision/collective_risk.py --dataset borg --device cuda:0
python revision/collective_risk.py --dataset bpi --device cuda:1
python revision/evaluate_joint_profiles.py --dataset borg --device cuda:0
python revision/evaluate_joint_profiles.py --dataset bpi --device cuda:1
python revision/evaluate_updated_dm_profiles.py --dataset borg
python revision/evaluate_updated_dm_profiles.py --dataset bpi
```

`evaluate_joint_profiles.py` samples its own earlier-fit reconciliation comparison from direct-count v2 fits. The separate, earlier `run_bpi_reconciliation.py` experiment reads the original `revision_bpi_direct_counts_v1` fit files, which are included in the package. Those v1 fits used the finite-difference implementation; `revision/provenance/counts_finite_difference.py.txt` preserves that source. The main updated reconciliation comparison instead reads the analytic-gradient development-fitted DM. The preserved failed Borg v1 directory contains no claimed result.

Original direct-fit v2 selection used complete earlier OOF cohorts. The stronger recent control uses leave-one-origin-out development selection and then all development origins for the final fit. Those LOO folds can include later development origins; they are not online validation. Both approaches are reported. New results must be described as reproductions/follow-ups, not newly untouched estimates.

## Fresh BPI 2012 confirmation

The source is [BPI Challenge 2012](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f), available from `https://ndownloader.figshare.com/files/24027287`. Store it as `data/raw/revision_bpi2012/bpi2012.xes.gz`. SHA256 is `5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`; supplied MD5 is `74c7ba9aba85bfcb181a22c9d565e5b5`. Publisher metadata is included beside it.

The executable protocol is `revision/FRESH_FINAL_PROTOCOL.json`, SHA256 `d40ee5bbfac35299e298c60521c27d6116ee7871d6e73b36c1661e49297e0e20`. Its 82 frozen assets include development data, selected models and code. Do not replace it with hashes of refitted models and imply that the result has the same prereview status.

With these frozen assets restored, a separate reproduction checkout can reconstruct the final derived data and scores:

```bash
python revision/prepare_bpi2012.py --period final
python revision/predict_bpi2012_final.py
python revision/evaluate_bpi2012.py --period final
```

The corresponding final output directories must be absent in that separate checkout; the distributed archive already contains them. The declared final origins are February 1–28 at noon UTC, with follow-up before March 14. January 19–31 origins provide calibration history only. Every online update waits for all 14-day labels to mature. No dependence or direct-count refit uses February outcomes.

For an independent from-scratch rerun, prepare development, fit individual HGB models, fit joint models for seeds 0/1/2, and evaluate development in that order. Scripts are `prepare_bpi2012.py --period development`, `fit_bpi2012_marginals.py`, `fit_bpi2012_joint.py --seed S --device cuda:N` and `evaluate_bpi2012.py --period development`. Use a new, explicitly labeled replication protocol for new hashes before proceeding; retain the original frozen archive. Minor platform differences can alter fitted parameters and selections.

## Provenance and limits

The original protocol SHA256 is `3fca720b01d98a36554b48d6f0d246965edcb6d5e839d80c5b821359e1a60f77`. Both protocols are local development-informed freezes, not externally registered studies. `LEDGER.md` retains chronology, failed attempts and source decisions. A GPU runner log records one failed fresh seed launch caused by a log-name collision before fitting; a UUID suffix fixed the wrapper and the successful run is retained. This did not change frozen scientific code.

The original review archive's one-seed residual caption is incorrect; saved evaluators/results use three seeds. The revised manuscript corrects it. Alibaba feasibility code is preserved for audit but its missing-end assumption is unvalidated; do not use its prepared data as a benchmark without resolving that issue. No full event-log generator, operational deployment or causal resource-allocation experiment is claimed.
