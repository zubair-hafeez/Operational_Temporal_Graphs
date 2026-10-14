# Evidence ledger

## 2026-10-14 Initial execution authorization

User authorized implementation through complete manuscript for final review, including GPU experiments. Two RTX A6000 GPUs and PyTorch CUDA access confirmed through host execution. Other GPU processes were left untouched. No model results yet. Created isolated project, data downloader, and research protocol. Final evaluation remains locked.

## 2026-10-14 Data, predictive pilots, and classical joint-generation controls

Acquired all 500 Borg task-event shards (1.624 GB compressed) plus machine events, schema, and BPI original archive; `reports/source_manifest.json` records checksums and versions. Seven-day Borg development prefix contains 35,398,938 events, 5,695,929 tasks, and 154,039 jobs. Initial state and publisher imputation are explicit. Exceptional pending-to-evict/finish transitions are not timestamp ties and must not be silently removed by a mask.

Built 325,080 sampled Borg task queries retaining full-cell context. HGB seed-0 feasibility models: pending pilot-test Brier 0.113220 personal, 0.084558 global, 0.057600 relational; running 0.021188 personal, 0.022528 global, 0.021171 relational. Thus no robust running-state improvement over the personal baseline. These are first-state-exit predictors, not generators.

BPI development parser excludes retrospective trace attributes and uses first A_Pending/A_Denied/A_Cancelled resolution; training audit finds no later application-stage events. Built 410,629 daily unresolved-case queries, seven-day horizon, pilot fit Jan–Jun, calibration July, pilot holdout August, with horizon embargoes. Resource edge is last observed workflow handler, not current occupancy. BPI pilot-test Brier: personal 0.136447, global 0.143126, relational 0.141710. The proposed resource-context benefit is **not supported** in this pilot.

Implemented a classical shared-factor Gaussian-copula scenario baseline with identical calibrated marginals across independent/correlated samplers. Selected factor weights on development count CRPS. Borg pending selected global variance 0.2 + job variance 0.5; sampled-population count-fraction CRPS 0.072532 independent versus 0.047820 coupled. Running selected job variance 0.8, CRPS 0.012616 versus 0.011207. BPI selected global variance 0.05 and **zero resource variance**, CRPS 0.046003 versus 0.034036. Independent 90% aggregate intervals cover very poorly in these pilots. This could partly reflect marginal misspecification/common shifts; it is not proof that the fitted relations cause outcomes. Full-cell assessment, stronger marginals, forward cross-fitting, state-conditioned classical controls, and block uncertainty remain necessary.

Prior-art follow-up: TACTiS/TACTiS-2, spatial Gaussian copulas trained with scoring rules, graph-copula time-series models, and classical frailty establish that graph/correlated sampling alone is not a novel contribution. Any paper must demonstrate an operationally useful, scalable approach for changing active-entity populations and explicitly compare shared-factor controls. BPI may serve as a negative/weak-relation case; do not conceal that result or force a graph gain.

One BPI fit failed after its first model because a NumPy integer was not JSON serializable. Fixed the generic state loop, retained failed run under `runs/pilot_hgb_seed0_bpi_pilot_h7d_failed_json_serialization`, reran successfully. Twelve causal replay and proper-score tests pass. No GPU training yet and no final-test scoring. Started 22-day Borg development preparation.
