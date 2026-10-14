# Research continuation

The user authorized end-to-end implementation, experiments, analysis, and a complete publishable-quality manuscript for final review. Continue autonomously; do not return only a plan or an unfinished placeholder paper. Publication acceptance cannot be guaranteed and no results may be fabricated. Do not submit externally. RF is excluded.

The active thread goal tracks this work. Mark it complete only after the manuscript, necessary experiments, audits, and reproducible artifacts are complete. Do not mark a goal blocked unless the required repeated-blocking threshold has actually been met.

Current project: `/mnt/zubair/Operational_Temporal_Graphs`. Parent proposal: `/mnt/zubair/Temporal_Graph_Papers_2024-2026/FIRST_PAPER_PROPOSAL.md`. Detailed prior-art audit: `../Temporal_Graph_Papers_2024-2026/analysis/first_paper_20261014/NOVELTY_AUDIT.md`.

Hardware checked on host: two NVIDIA RTX A6000, 49,140 MiB each. Initial existing allocations approximately 9 GB on GPU0 and 5 GB on GPU1; do not kill or alter other jobs. The sandbox cannot communicate with the GPU driver; host execution works. Python environment has torch 2.5.1+cu124, PyG 2.7.0, NumPy 2.2.6, pandas 2.2.3, Arrow 22, SciPy 1.14.1, sklearn 1.6, pytest. No dependency installation is currently needed. Numba import fails; avoid relying on it.

Research test: active relational state must add useful signal over same-information sequence/set/state-summary controls, and gains must survive free rollouts and downstream real-data utility. State-feedback, masks, temporal generation, and workload simulation are established prior art. Mandatory competitors include DG-Gen, RIMS/RIMS+, AgentSimulator, and cloud TraceGen adaptations with transparent task differences.

Initial source files from earlier ideation are in `/tmp/borg_task_events_00000.csv.gz`, `/tmp/borg_task_events_00001.csv.gz`, `/tmp/borg_schema.csv`, `/tmp/borg_schema_doc.pdf`, `/tmp/borg_schema_doc.txt`, `/tmp/bpi2017.xes.gz`, `/tmp/bpi2017_record.json`, and `/tmp/rims2025.pdf`. They have not been used for model fitting. Initial cloud sample spans only ~2.78 hours; BPI early-feature audit used Jan–Feb. The raw Google task-event table has 500 shards totaling 1,624,496,362 compressed bytes. Complete bucket listing is `/tmp/borg_task_listing.xml`.

Keep STATE.json and LEDGER.md updated after meaningful changes. Preserve code/config/data hashes for results. On compaction, inspect the newest ledger entry and live process status before repeating work.
