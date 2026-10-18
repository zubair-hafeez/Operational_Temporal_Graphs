Folder organization update (18 October 2026): the latest manuscript is `paper_revision/main.pdf` (10 pages), also accessible as `LATEST_PAPER.pdf`; `paper/main.pdf` is the earlier 12-page paper. The duplicate review ZIPs were removed after all unique contents were retained. The ZIP-based commands below describe the historical package workflow. To use them, first reconstruct the appropriate package from this folder:

```bash
python scripts/organized_review.py --restore original --output /tmp/otg_original_review.zip
```

Use that output path in place of the retired `releases/` ZIP. Every member is restored byte-for-byte; ZIP metadata/compression and its whole-archive checksum may differ. To check current storage without recreating a ZIP, run `python scripts/organized_review.py`. Large immutable duplicate arrays use relative links; preserve links when copying the project, or dereference them when creating a standalone subset. Treat frozen inputs as immutable and create new output paths for new experiments.

--- Historical reproduction instructions ---

# Reproducing the operational-cohort study

The implemented target is a conditional joint first-exit distribution for the population active at a timestamp. Read `docs/SCOPE_REVISION.md` for the boundary between this task and the original broader event-log proposal. `docs/FINAL_PROTOCOL.json` is the immutable development-informed final freeze. Final results must not be used to repeat model selection.

## Environment

The reference environment is Python 3.10.14, PyTorch 2.5.1+cu124 and the versions in `requirements-research.txt`, on NVIDIA RTX A6000 GPUs. Use a dedicated environment. The local interpreter used for this study is `/home/zubair/zubairEnv/bin/python`. The scripts find the project root from their own location, so the source does not require this absolute installation path on another host. All commands below run from the project root and use `python` for that interpreter.

Set `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. For plotting, set `MPLCONFIGDIR` to a writable temporary/cache directory. CUDA temporal inference is the reference path; a numerical audit records small CPU/CUDA differences in time-feature arithmetic. CPU and GPU samplers use different random streams, although both implement the stated distribution.

Run `python -m pytest -q tests` for implementation checks. Tests do not constitute a reproduction of the statistical findings.

## Fast reproduction from saved forecasts and scores

The local ZIP includes the manuscript, source, all 264 frozen model/selection files, final score tables, and the completed audit reports. It excludes raw provider logs and the large final cohort/history arrays. From that ZIP, `build_paper_artifacts.py`, `draw_forecast_contract.py`, and `complete_paper_artifacts.py` regenerate the manuscript tables and figures from saved numerical evidence. The completion script reads the supplied audit report. Independently rerunning `audit_final_artifacts.py` or individual diagnostics requires reconstructing the excluded final arrays by the later instructions. The archive's `RELEASE_MANIFEST.json` records every included file hash; it is a local review package, not an externally published release.

Final score tables are under `reports/summary_{borg,bpi}_selected_v2_final_m512/`. Each contains all seed/origin/projection/horizon scores, macro and projection tables, and paired block intervals. The empirical baseline is stored separately in `runs/aggregate_residual_{dataset}_selected_v2/`. `paper/generated/evidence_manifest.json` links all manuscript-generated outputs to source hashes.

With the preserved checkpoints, selected-data manifests and final forecasts present, run:

```bash
python scripts/final_diagnostics.py --dataset borg
python scripts/final_diagnostics.py --dataset bpi
python scripts/summarize_joint_results.py --dataset borg --period final
python scripts/summarize_joint_results.py --dataset bpi --period final
python scripts/build_paper_artifacts.py
python scripts/draw_forecast_contract.py
python scripts/audit_final_artifacts.py
python scripts/complete_paper_artifacts.py
```

The first four commands deliberately refuse to overwrite an existing result directory. Use a clean reproduction checkout with source assets restored; preserve existing results if auditing the original workspace. The remaining commands regenerate paper-only tables/figures, the separately labeled inference schematic, and numerical-audit prose. The presentation completion script clarifies two captions without changing numbers. The primary comparisons and all secondary controls must be retained, including negative results.

## Data acquisition and development reconstruction

`python scripts/fetch_data.py --borg-parts 500 --workers 4` downloads the required public Borg task-event shards, machine/schema metadata and BPI archive. It verifies existing downloads. Source URLs, byte counts, license records and SHA-256 values are in `reports/source_manifest.json` and related data documentation. Raw data remain subject to provider terms; this project does not relicense the BPI archive.

In a new workspace, the ordered preparation stages are:

```bash
python scripts/prepare_borg.py --through-days 22
python scripts/prepare_bpi.py --before 2016-11-01
python scripts/build_borg_cohorts.py
python scripts/build_bpi_cohorts.py
python scripts/add_borg_event_history.py
python scripts/build_temporal_histories.py --dataset borg
python scripts/build_temporal_histories.py --dataset bpi
```

Fit HGB with `scripts/fit_borg_development_marginals.py` for both datasets, each available state and each `--arm personal|global|relational`. For Borg, run both with and without `--history`. Use seed0 and default hyperparameters. The script runs all four chronological folds. Run `scripts/audit_marginal_calibration.py` for both datasets and the additional Borg history arm. The run manifests record exact commands/configurations; consult each script's `--help` instead of substituting a random split.

Run `scripts/run_development_suite.py --suite temporal --datasets borg bpi` on the two GPUs to fit Mixer, GRU and MLP seeds0/1/2 across all four folds. After all complete, run `scripts/select_development_marginals.py --dataset ...` for each dataset and `scripts/predict_selected_cohorts.py --dataset ... --device cuda:...`. These prepare `selected_v2`, the common individual CDF used by every dependence model.

Run `scripts/run_development_suite.py --suite joint --datasets borg bpi --data-tag selected_v2`, followed by `scripts/evaluate_development_suite.py --dataset ... --data-tag selected_v2 --device cuda:...` for each dataset. Run `scripts/aggregate_residual_baseline.py --dataset ...` to fit the historical-error bank and leave-one-origin-out development comparator. Summarize each dataset with `scripts/summarize_joint_results.py --dataset ... --data-tag selected_v2`.

The original artifact contains a completed freeze. A from-scratch rerun can reproduce selection and generate its own manifest in a separate research checkout; do not overwrite the original manifest or claim that a rerun freeze was created before the original final results were known. Minor numerical differences can change selected models; report them instead of substituting the expected winner.

## Frozen final evaluation

Original final access began only after the hash-verified manifest was written. With the original frozen assets restored, `python scripts/open_final_data.py` prepares the final prefixes, cohorts and histories. It records first access in `STATE.json` and refuses silent replacement of partial artifacts. Its two dataset workers are independent, so BPI may finish first. No model fitting occurs here.

The original Borg final preparation produced float64 Parquet timestamps and the integer-time replay rejected the join before producing any final cohort scores. The frozen research code was retained. `scripts/repair_final_borg_timestamps.py` checks every timestamp for finite, exact integer representation and casts it losslessly, preserving the original prefix under `borg_prefix_29d_original_float_schema`. The audit is `reports/final_borg_timestamp_correction.json`. If reproducing that same schema failure, run this correction once, then resume `build_final_cohorts.py --dataset borg` and `build_final_histories.py --dataset borg` directly with new logs. Do not apply it to a prefix already stored as int64. No model, cutoff, outcome value or evaluation choice is changed by the correction.

For each dataset, use the frozen model to predict and score:

```bash
python scripts/predict_selected_cohorts.py --dataset borg --period final --device cuda:0
python scripts/predict_selected_cohorts.py --dataset bpi --period final --device cuda:1
python scripts/evaluate_development_suite.py --dataset borg --data-tag selected_v2 --period final --device cuda:0 --workers 2
python scripts/evaluate_development_suite.py --dataset bpi --data-tag selected_v2 --period final --device cuda:1 --workers 2
python scripts/evaluate_development_suite.py --dataset borg --data-tag selected_v2 --period final --device cuda:0 --workers 2 --draws 1024 --seeds 0
python scripts/evaluate_development_suite.py --dataset bpi --data-tag selected_v2 --period final --device cuda:1 --workers 2 --draws 1024 --seeds 0
python scripts/aggregate_residual_baseline.py --dataset borg --period final
python scripts/aggregate_residual_baseline.py --dataset bpi --period final
```

On the original managed host, GPU calls are made through the persistently approved `scripts/gpu_research.py` runner. Its named tasks (`predict`, `evaluate-suite`, `suite`, etc.) correspond to the scripts above; it invokes no shell and logs internally. This avoids repeated CUDA approval prompts caused by shell redirection. The wrapper is a host permission convenience, not a scientific dependency.

Recreate diagnostics, summaries and figures as above. Keep every original output if a correctness bug requires an explicitly documented rerun. Report final results without any further calibration or model selection.

## Manuscript and limitations

`paper/main.tex` includes the complete manuscript source and BibTeX references. Generated quantitative tables are under `paper/generated/`; vector figures are under `paper/figures/`. A standard LaTeX/BibTeX toolchain or Tectonic can compile it. Author names and affiliations are intentionally not invented; they require the researcher's final review.

This artifact does not contain a reproduced CEP3, DG-Gen, TACTiS, or full business-process-simulation numerical benchmark. Pinned source archives support the protocol/related-work audit only. The study's simple reference models and application adapters must be assessed on the claims actually tested.
