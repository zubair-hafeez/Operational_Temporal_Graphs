# Scientific review notes

This is an application and benchmark manuscript on joint first-exit forecasting for changing operational cohorts. Its strongest evidence is a controlled comparison with identical individual probabilities, observed-incidence permutations, a simple direct aggregate comparator, and untouched future periods in two applications. The classical copulas and neural components are not new methods.

## Findings to check in the manuscript

- Borg selected Gaussian/static: count-fraction CRPS improves 24.3% and the declared ratio-9 reserve proxy improves 59.7% over independent draws. Most benefit is already available from a global factor; typed incidence adds a smaller gain concentrated in jobs.
- BPI selected Gaussian/global: count CRPS improves 8.3%, while the reserve proxy worsens 9.6%. The loss-difference interval crosses zero for 14-day blocks but excludes zero for 28-day blocks. The paper reports this dependence on block choice.
- True BPI handler incidence improves CRPS over risk-stratified permutation but worsens reserve loss. These are sampler interventions, not causal effects of reassignment.
- Context networks have slightly better mean CRPS than static factors in some comparisons, but the paired block intervals do not establish a consistent advantage. The historical-error baseline is strongest on BPI aggregate CRPS.
- Borg macro coverage near 90% conceals population coverage around 76% and relation-group coverage near 98%. The paper must not claim uniformly calibrated joint predictions.

## Main publication risks

The methodological novelty is limited: this paper should be assessed as a practical benchmark and empirical study. Avramenko et al. (2026) already study current-state process simulation and uncertainty on BPI2017; that close prior is explicitly cited. CEP3, DG-Gen, TACTiS and full business-process simulators are discussed and source-audited but not numerically reproduced on this target. A reviewer may require such adaptations or additional organizations/datasets. Those comparisons cannot be represented as completed.

The cluster's final period spans seven days in one site; the workflow spans two months in one bank. Millions of repeated entity queries do not provide independent temporal replication. Equal projection/group weighting and penalty ratios are declared benchmark choices. The reserve loss is an operational proxy, with no observed policy deployment or monetary savings. Cohort forecasts exclude future arrivals, repeated state transitions, and arbitrary graph evolution. These are substantive limits rather than footnotes.

The original exploration considered broader event-log generation. `docs/SCOPE_REVISION.md` records the evidence-driven change. Do not present this artifact as a completed synthetic-data factory or claim downstream synthetic-training-data benefits. The reusable foundation consists of timestamp-bounded adapters, outcome contracts, joint sampling and diagnostics.

## Integrity and authorship

Final model/code choices were frozen before final preparation; the manifest hashes 264 files. A final Borg timestamp type mismatch stopped replay before cohort scoring. The separate lossless correction verified every timestamp and retained the original files; it changed no values, model or evaluation choice. Read `reports/final_borg_timestamp_correction.json` and the ledger. Negative and failed experiments are preserved.

Author names, affiliations, contributions, target venue, and the eventual release license require the researcher's final review. They are not invented in the manuscript. No paper, message, repository release or dataset has been submitted or published externally. The local review package is intended to support that review, not to certify acceptance by a venue.
