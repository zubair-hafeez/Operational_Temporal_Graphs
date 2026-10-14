# Research protocol

## Scope and success criteria

Generate conditional continuations of operational temporal graphs and assess whether explicit active relations preserve real behavior and improve useful training data. Start with Borg; port the tested representation and procedure to BPI. Separate architectural gains from additional information, hard masks, resampling, and richer workload features. Revise the method or claim if strong controls explain its advantage.

## Fixed periods

Google trace time zero is an initial-state marker. Physical day 1 begins at timestamp 600 seconds. Define `elapsed_seconds = timestamp_us / 1e6 - 600` for positive-time observations.

- Pilot fit: elapsed days [0,4); pilot development [4,5); pilot holdout [5,7). Pilot holdout results can guide subsequent full development but never stand in for final evaluation.
- Full fit: [0,18); development/calibration [18,22); final [22,29). Prediction labels must fit within the declared partition or use explicit censoring.
- BPI fit: 2016-01-01 to 2016-09-01; development: September–October; final cutoffs: November–December; follow-up through archive end when available. Exact dates use UTC.

The final set may be ingested and checksummed, but not scored or used for architecture, thresholds, or tuning before a frozen evaluation manifest exists. Preserve ongoing entities' factual prefix state across time boundaries; fit no future outcomes from them. Preprocessing and feature scales are fitted on permitted training data. Random event splits are forbidden.

## Observation contract

No future machine assignment, final task duration, post-cutoff event count, future entity list, eventual offer outcome, or future boundary events are model inputs. Entity index mappings are storage identifiers only, not predictors. Initial state is observed/censored as documented; zero-time rows are not training arrivals. Record publisher-imputed and ambiguous events and evaluate sensitivities.

Borg resource requests are not measured utilization or hard physical capacity. Scheduling is not an exact program-execution start. BPI resource IDs may include system users; no exclusive-capacity worker rule without evidence. Mark application resolution by an explicit supported milestone rather than retrospectively last observed row.

## Comparisons

Start with historical-window retrieval, bootstrap, semi-Markov/survival, event-history, state-summary, and set models. Add graph representation only after meaningful baseline comparisons. Give matched models the same facts and legal actions. Label all deviations from published baselines. An oracle future-arrival experiment is separate from end-to-end generation.

## Evaluation

Score complete generated state/outcome trajectories with proper scores and calibration. Include distributions of backlog, completion and eviction counts, and time-to-event outcomes. Evaluate existing-entity queries fixed at cutoff, and permutation-invariant population outcomes for newly generated entities. Use state-load strata assigned from cutoff information with training-defined thresholds.

Train fixed downstream predictors on real, resampled, generated, and mixed training data and evaluate on real future data. Include at least one downstream non-graph model. Repeated seed and paired block uncertainty are required. Days/blocks, not individual tasks, define independent temporal evidence. Report finite-month and single-cell limitations.

## Integrity

Raw downloads carry source metadata and hashes. Runs carry code/config/data hashes, seed, split, environment, command, start/end time, and exit state. Preserve failed/negative runs. Never silently replace an experiment after observing final outcomes. All table values and plots must be generated from saved results. No invented values, citations, or significance claims.

Use synthetic toy cases to validate replay, censoring, score calculation, and permutation handling. These tests establish code behavior, not real-world method effectiveness. A zero-validity-error result produced by hard masks is not the main research finding.
