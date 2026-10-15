# Joint forecasting for operational temporal graphs

Research project on joint first-exit forecasts for populations active at a forecast origin. Applications are Google Borg task states and BPI 2017 application resolution. The [scope revision](docs/SCOPE_REVISION.md) records how experiments and prior work narrowed the original event-log-generation proposal. This implementation does not generate full future logs or arrivals.

The [final evaluation protocol](docs/FINAL_PROTOCOL.json) freezes selected models, code hashes, periods and comparisons. It must not be silently changed after final data access. The manuscript source is [paper/main.tex](paper/main.tex); its abstract/results and PDF are still being completed.

The target deliverable is a complete research manuscript with reproducible evidence for the user's final review. Results must determine the final claims. No external submission or publication is authorized by this repository.

## Working state

Read [RESUME.md](RESUME.md), [STATE.json](STATE.json), [research protocol](docs/PROTOCOL.md), and [evidence ledger](LEDGER.md). Data and large artifacts are kept outside version control; source URLs, hashes, versions, and run configurations are retained.

Use `/home/zubair/zubairEnv/bin/python`. Two RTX A6000 GPUs are available through approved host execution. Existing allocations belong to other work and must remain untouched. This research is independent of RF projects.

## Initial commands

```bash
/home/zubair/zubairEnv/bin/python scripts/fetch_data.py --borg-parts 500 --workers 4
/home/zubair/zubairEnv/bin/python -m pytest tests -q
```

The data downloader validates existing files and resumes missing downloads. It downloads only task events, machine events, schema/documentation metadata, and the BPI archive, not the complete Google usage release.
