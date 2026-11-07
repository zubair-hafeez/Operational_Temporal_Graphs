> **Complete research snapshot:** Code and manuscript files are browsable here. Datasets, checkpoints, caches, ignored files and original local Git metadata are included in the split archives attached to [release research-snapshot-20261107T032249Z](https://github.com/zubair-hafeez/Operational_Temporal_Graphs/releases/tag/research-snapshot-20261107T032249Z). See [restore instructions](github_snapshot/RESTORE.md) and [verification report](github_snapshot/VERIFICATION_REPORT.md). A normal Git clone does not download those archives.

# Collective forecasting on operational temporal graphs

Latest paper: [OPERATION.LATEST_PAPER.pdf](OPERATION.LATEST_PAPER.pdf), pointing to [paper_revision/main.pdf](paper_revision/main.pdf). Editable [LaTeX](paper_revision/main.tex). Earlier manuscript versions remain deleted; no duplicate paper archive is needed.

“When Do Relations Help Collective Forecasts?” is a venue-neutral, six-source empirical study: Borg, Azure V2, BPI 2017/2012, UCI incidents and BPI 2014 incidents. It forecasts first exits of currently active cohorts, not full future logs or arrivals. The six sources have 201 scored origins; repeated entities and seeds are not independent replications.

The [extension review](extension/REVIEW.txt) maps all six requested improvements to experiments and limitations. [Reproduction instructions](extension/REPRODUCE.txt) cover saved models, score tables, source sampling and bounded downloads. Google2019, BPI change snapshots, CitiBike and airline pilots are transparently reported as deferred or exploratory—not extra confirmation sources.

Key confirmation findings: UCI typed Gaussian/Student-t reduce count CRPS by 3.3%/3.6% against matched global controls, but the strongest direct-count model is essentially tied. Azure's development-selected subscription-only Gaussian improves global CRPS by 4.7%, while its combined typed fit collapses to global. Strong trees often beat neural graphs. Original Borg's typed gain remains 2.2% beyond global, versus 24.3% beyond independence. The contribution is controlled evidence, not a universal graph-model win.

The [new protocol](extension/CONFIRMATION_PROTOCOL.json) freezes 165 assets before new-period target preparation and scoring. The original [264-entry](docs/FINAL_PROTOCOL.json) and [82-entry](revision/FRESH_FINAL_PROTOCOL.json) manifests remain immutable. [Pre-confirmation checks](reports/extension_20261021/pre_confirmation_audit.json) passed 52 tests; the [publication audit](reports/extension_20261021/publication_audit.json) records final checks and numerical sensitivity. New storage is about 1.1 GB, below the 2-GiB cap. No external submission or public release has occurred.

## Working state

Read [START_HERE.txt](START_HERE.txt), [RESUME.md](RESUME.md), [STATE.json](STATE.json) and [extension/REVIEW.txt](extension/REVIEW.txt). Large data/models are outside version control; URLs, hashes, versions and run configurations are retained.

Use /home/zubair/zubairEnv/bin/python. Two RTX A6000 GPUs were used within bounded allocations; other projects' GPU processes must remain untouched. This project is independent of RF work.

## Local verification (no downloads)

```bash
/home/zubair/zubairEnv/bin/python extension/build_publication.py
/home/zubair/zubairEnv/bin/python extension/audit_publication.py
```

Do not rerun completed experiment directories or alter frozen files. Historical notes can reference retired papers and archives; current instructions supersede those restoration recipes.
