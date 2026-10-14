# Operational temporal graph generation

Research project on preserving operational behavior in generated temporal graphs. The primary application is Google Borg task scheduling; BPI 2017 supplies a second application. The full approved motivation is in [the research proposal](../Temporal_Graph_Papers_2024-2026/FIRST_PAPER_PROPOSAL.md).

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
