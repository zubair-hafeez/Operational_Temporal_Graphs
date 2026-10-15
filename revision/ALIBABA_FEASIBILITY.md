# Independent-provider feasibility check

The strengthened study still has one compute provider and two releases of one workflow. Before final packaging, audit the small Alibaba 2018 batch-task table as a possible additional compute replication. This is a feasibility check, not a promise to add a favorable dataset.

Official description: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md . Schema: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt . Official download-script host supplies the task archive (130,310,549 bytes), SHA256 `7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6`, matching the publisher checksum. It contains an 802,261,444-byte task table. The 20 GB instance table is not downloaded.

Reserve outcomes at or after elapsed day 6 for any potential final evaluation. The initial audit retains only tasks started before day 2 and clips ending times at day 2. It must not summarize later durations/statuses or use eventual task status as an input. The candidate target is first recorded end of a currently running task, with job grouping; this table cannot support a machine-incidence comparison. Missing starts, censored/malformed ends, duplicates, group sizes and available predictors determine suitability. A full task DAG known only retrospectively must not enter a prefix.

If suitable, tentatively allocate fit before day 3, calibration day 3–4, dependence/direct fitting day 4–5, development day 5–6, and final hourly origins day 6 through day 7+23h at 1/5/15/60-minute horizons. These are candidate boundaries, not a frozen executable protocol. Freeze code and selections before retaining final outcomes. An unsuitable source is documented and deferred.

## Decision after the schema audit

Defer this source from the current paper. The development parser completed 9,887,832 task records and 1,923,476 task–origin queries through day 6, but its treatment of absent ends as administrative censoring is an unvalidated feasibility assumption. It must not be used for scientific scores.

The separate first-two-day missing-end audit (`reports/revision_alibaba_missing_ends.json`) finds 5,622 positive-start records with no recorded end. At hour 47, they account for 5,425 of 6,434 reconstructed active tasks (84.3%). At hour 24 the fraction is only 2.3%, so accumulation substantially changes the candidate cohort. The official 2018 schema does not establish whether every absent end represents continued running. The 2017 instance documentation discusses absent ends for both running and failed instances, but that is a different release and table and cannot settle the 2018 task semantics. Excluding records using eventual terminal status would also condition the cohort on future information.

This is a limitation of our proposed adapter and available semantic evidence, not a general judgment that the Alibaba dataset is unusable. No predictive models or scores were produced. Exact end times and statuses in the reserved day-6-and-later period remain unexamined; no final protocol was created. The archive and development outputs are preserved for a future timestamped-state or censoring-aware adapter.
