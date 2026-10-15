# Joint forecasting for operational temporal graphs

Research project on joint first-exit forecasts for populations active at a forecast origin. Applications are Google Borg task states and BPI 2017 application resolution. The [scope revision](docs/SCOPE_REVISION.md) records how experiments and prior work narrowed the original event-log-generation proposal. This implementation does not generate full future logs or arrivals.

The complete [manuscript PDF](paper/main.pdf) and [LaTeX source](paper/main.tex) are ready for author review. Start with the candid [scientific review notes](REVIEW_NOTES.md), then the [reproduction instructions](REPRODUCE.md). The [final evaluation protocol](docs/FINAL_PROTOCOL.json) freezes selected models, code hashes, periods and comparisons; it must not be silently changed after final data access.

The study evaluates 28 final cluster origins and 61 final workflow origins. The selected Gaussian factors reduce count-fraction CRPS by 24.3% and 8.3%, respectively, versus independent sampling. The ratio-9 reserve proxy improves by 59.7% in the cluster but worsens by 9.6% in the workflow. Most cluster improvement already comes from a global factor; typed relations add a smaller contribution. Historical aggregate errors have the lowest mean workflow CRPS. This is an application benchmark and empirical study using classical dependence models, not a claim of a new copula architecture.

All 36 implementation tests pass. The final artifact audit verifies all 264 frozen files and the complete final forecasts/scores. Doubling scenario draws preserves each family's ranking. The local review archive includes source, frozen model/selection assets, numerical evidence and the PDF; raw provider logs and full final feature arrays remain outside it. No external submission or publication has occurred.

## Working state

Read [RESUME.md](RESUME.md), [STATE.json](STATE.json), [research protocol](docs/PROTOCOL.md), and [evidence ledger](LEDGER.md). Data and large artifacts are kept outside version control; source URLs, hashes, versions, and run configurations are retained.

Use `/home/zubair/zubairEnv/bin/python`. Two RTX A6000 GPUs are available through approved host execution. Existing allocations belong to other work and must remain untouched. This research is independent of RF projects.

## Initial commands

```bash
/home/zubair/zubairEnv/bin/python scripts/fetch_data.py --borg-parts 500 --workers 4
/home/zubair/zubairEnv/bin/python -m pytest tests -q
```

The data downloader validates existing files and resumes missing downloads. It downloads only task events, machine events, schema/documentation metadata, and the BPI archive, not the complete Google usage release.
