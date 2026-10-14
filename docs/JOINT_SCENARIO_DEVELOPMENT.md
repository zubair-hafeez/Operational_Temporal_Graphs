# Joint scenario development and claim boundaries

This is an experimental design record, not a completed paper or a claim of a new copula family. The user authorized a complete paper; current pilots do not yet justify declaring the research successful.

## Operational question

For the work already present in a system, how much can finish, fail, or remain outstanding together over a future horizon? Per-entity risk accuracy is insufficient to identify this joint distribution. The experiments test the practical importance of this distinction on changing task/job/machine populations and case/offer/resource histories. The current implemented target is the first exit of a current active episode (Borg) or application resolution (BPI). It does **not** generate new arrivals, complete event logs, or causal scheduling interventions. These omissions must remain explicit unless subsequent experiments implement them.

## Evidence so far

Source acquisition and prefix replay are reproducible. Borg's pending-state relational features help individual prediction in the initial pilot; running-state gains are weak. BPI resource summaries do not beat personal history. Classical correlated scenarios improve count distribution scores with fixed individual probabilities. A context-dependent Gaussian factor model has **not consistently beaten the small static model** on pilot scenario scores. Treat this as a reason to strengthen comparisons and possibly make the contribution an operational benchmark and simple effective decoder, not to conceal the static baseline.

The second pilot separates marginal fit, calibration, dependence fit, tuning, and holdout chronologically. Its smaller marginal training set causes visibly poorer marginal forecasts than the earlier pilot. Results are therefore not directly comparable across the two pilot configurations. Full development will use forward cross-fitting and a stronger final marginal fit to avoid sacrificing most training data.

## Full-development test

Borg: 22-day prefix, 115,400,646 events. Sample training queries uniformly within current pending/running state, preserve full observed context, and evaluate full populations at development cutoffs. Predict first-exit CDF at 1, 5, 15, and 60 minutes. Store causal job and machine incidence. Unknown machine and pending nodes get singleton machine factors. Compare personal/history, system-state, relation-summary, and learned temporal encoders, with identical observed facts and no learned node-ID lookup for unseen entities.

Use forward chronological folds for dependence training: a marginal model and calibrator may only use outcomes that precede the coupling-fit window. A full marginal model may subsequently use days [0,18), calibration [18,20), and model selection [20,22). All label horizons must fit their fitting partition. The last week remains locked until an evaluation manifest freezes code, data, configurations, selection, metrics, seeds, and failure handling.

Coupling comparisons must include independence, a system-wide factor, fixed graph factors, learned context-dependent factors, graph permutation controls, and alternative common-shock/tail dependence. The initial Gaussian factor construction is standard. Classical Marshall–Olkin/common-shock models and empirical copula/ensemble coupling are relevant competitors, not inventions to rename. CDFs must be monotone across horizons, and generated event cardinality must emerge from samples rather than come from future logs.

Evaluate population and subgroup count CRPS, interval scores/coverage/width, joint trajectory scores, and operational decision proxies with predeclared costs. Individual scores remain necessary. Use temporal blocks, not millions of tasks as independent samples. For Borg, requested resources are not measured physical use; a current-episode survival forecast must not be described as total future cluster utilization. BPI's last-handler graph is not a capacity-constrained queueing system. Downstream synthetic-data utility remains unimplemented and must be tested or removed from the final claim with explanation.

## Prior art that rules out broad novelty claims

- [TACTiS, ICML 2022](https://proceedings.mlr.press/v162/drouin22a.html) and [TACTiS-2, ICLR 2024](https://arxiv.org/abs/2310.01327) already learn joint predictive distributions using copulas. Uneven observations and flexible forecasting are not new by themselves.
- [Huk et al., 2023](https://arxiv.org/abs/2308.09827) use neural marginal models and censored spatial Gaussian copulas trained with scoring rules. A marginal-preserving two-stage copula model is not a new general principle.
- [Graph-copula electricity forecasting, 2024](https://doi.org/10.1016/j.asoc.2024.111324) already combines spatial/temporal graphs with copula dependence.
- [Schefzik et al., 2013](https://arxiv.org/abs/1302.7149) establish ensemble copula coupling; [the Schaake shuffle](https://doi.org/10.1175/1525-7541%282004%29005%3C0243%3ATSSAMF%3E2.0.CO%3B2) reconstructs dependence through historical ranks. Showing dependence helps operational decisions is an established motivation in probabilistic forecasting.
- [CEP3, LoG 2022](https://proceedings.mlr.press/v198/wang22c.html) jointly forecasts multiple graph events and timestamps. Do not claim previous temporal graph work only predicts independent links.
- [DG-Gen](https://arxiv.org/abs/2412.15582), RIMS, AgentSimulator, TraceGen, and Mimesys remain relevant to the original wider generation direction. Stateful simulation and operational evaluation are established.
- [Borg: The Next Generation, EuroSys 2020](https://research.google/pubs/borg-the-next-generation/) already discusses job dependencies underlying failures. Observing job-level dependence is not a novel workload discovery.
- [Oberst and Sontag, ICML 2019](https://proceedings.mlr.press/v97/oberst19a.html) establish Gumbel-Max structural causal constructions. Shared Gumbel sampling must not be framed as new causal identification; observational forecasts here do not identify policy effects.

## Reproducible source-code findings

Archived source pins and tarball hashes are in `reports/baseline_source_manifest.json`.

DG-Gen commit `9d9c94d1c8f60d4e3578ef0eb6b8671d88df3e12`: `DGGen/generate.py` lines 32–35 chooses the number of interactions from 15% of the input dataset size; `inference.py` iterates a specified number of batches. This is a fixed-size synthetic graph protocol. It cannot directly establish forecast quality for unknown future event count without an explicitly documented horizon-stopping adaptation.

CEP3 commit `7ea12a5e6c9587959e98c694f1bfc1747f81cec2`: `pointprocess.py` takes source/destination node sets from `true_events_graph`, sets `num_edges = true_events_graph.num_edges()`, and uses that as the simulation step limit. Its `cont_sim` currently tests step count while the time condition is commented out. The default non-visualization branch uses expected inter-event time; stochastic exponential sampling occurs in the visualization branch. These are implementation/protocol facts, **not an accusation of improper research conduct**. For an operational fixed-horizon comparison, freeze the candidate universe from the observed prefix, sample inter-event times, and stop at the horizon; label this as an adaptation and retain an oracle-count sensitivity only as a diagnostic.

TACTiS source commit `64e9db6405438ebd3a756a072711c81f57fbcffb` is archived for architecture inspection. Adapting continuous time-series likelihoods to discrete event indicators requires an appropriate probability treatment; arbitrary jitter-density scores must not be presented as exact discrete likelihoods.

## What would justify continuing versus revising

Continue a method-centered paper only if a clearly specified scalable decoder adds reproducible operational value over the strongest simple dependence models, matched-information temporal encoders, and reasonable resampling/ensemble controls. Otherwise favor a rigorously justified benchmark/measurement paper if its protocol, datasets, and cross-domain findings are substantial enough. Neither route is automatically publishable. No final test has been used to choose between them.
