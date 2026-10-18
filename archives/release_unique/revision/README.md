# Joint forecasting for operational temporal graphs

Research project on joint first-exit forecasts for populations active at a forecast origin. Applications are Google Borg task states and BPI 2017 application resolution. The [scope revision](docs/SCOPE_REVISION.md) records how experiments and prior work narrowed the original event-log-generation proposal. This implementation does not generate full future logs or arrivals.

The current [revised manuscript PDF](paper_revision/main.pdf) and [LaTeX source](paper_revision/main.tex) present a three-log empirical study of collective forecasting on temporal operational graphs. Start with the [revision review notes](REVISION_REVIEW_NOTES.md), then the [revision reproduction instructions](REVISION_REPRODUCE.md). The original study, PDF, archive and [264-file protocol](docs/FINAL_PROTOCOL.json) remain preserved. The new [82-asset protocol](revision/FRESH_FINAL_PROTOCOL.json) separately froze BPI 2012 confirmation before its final outcomes were retained.

The [saved revision plan](docs/CURRENT_PAPER_REVISION_PLAN.md) motivated the completed direct-count, reconciliation, calibration, graph and joint-risk controls. The [broader research extension plan](docs/RESEARCH_EXTENSION_PLAN.md) remains a separate proposal; no unimplemented method is claimed as a result.

The study evaluates 28 Borg, 61 BPI 2017 and 28 fresh BPI 2012 origins. Borg typed factors improve count CRPS 24.3% over independence and 2.2% beyond a global factor. Strong recent direct counts and same-version calibration change the BPI 2017 ranking. In fresh BPI 2012, global dependence improves count CRPS 12.7% while worsening reserve loss 20.4%; a development-selected direct model beats it on both. Count and joint-risk rankings can differ. These findings support a benchmark and controlled empirical contribution using established statistical models.

All 45 implementation tests pass. The [revision audit](reports/revision_artifact_audit.json) verifies both frozen asset sets, table inputs and final-period eligibility. A separate local revision archive includes source, fitted controls, numerical evidence, the small fresh-replication dataset and the PDF; large original raw logs and final arrays remain outside it. See `reports/revision_archive_verification.json` for clean-extraction verification when complete. No external submission or publication has occurred.

## Working state

Read [RESUME.md](RESUME.md), [STATE.json](STATE.json), [research protocol](docs/PROTOCOL.md), and [evidence ledger](LEDGER.md). Data and large artifacts are kept outside version control; source URLs, hashes, versions, and run configurations are retained.

Use `/home/zubair/zubairEnv/bin/python`. Two RTX A6000 GPUs are available through approved host execution. Existing allocations belong to other work and must remain untouched. This research is independent of RF projects.

## Initial commands

```bash
/home/zubair/zubairEnv/bin/python scripts/fetch_data.py --borg-parts 500 --workers 4
/home/zubair/zubairEnv/bin/python -m pytest tests -q
```

The data downloader validates existing files and resumes missing downloads. It downloads only task events, machine events, schema/documentation metadata, and the BPI archive, not the complete Google usage release.
