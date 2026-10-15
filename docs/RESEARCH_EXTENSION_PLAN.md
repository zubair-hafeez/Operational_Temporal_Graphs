# Research plan for reliable collective forecasting

Saved 15 October 2026. This is the broader research plan requested by the researcher. The proposed experiments and method extensions have not been executed. The existing manuscript, frozen protocol, results, and review archive remain the completed first study. The more focused [current paper revision plan](CURRENT_PAPER_REVISION_PLAN.md) should be considered first when the immediate objective is publication.

The research objective is to generate accurate future scenarios for changing operational populations, with consistent predictions across individuals, overlapping groups, and several forecast horizons. The applications remain computing workloads and business processes. RF is excluded.

The central question is: **How can joint forecasts correct systematic prediction errors while retaining consistent entity outcomes and useful group uncertainty as populations and relationships change?** This is a candidate research contribution whose novelty and practical value must be tested against established forecasting methods.

## 1 Evidence motivating the next problem

| Evidence from the completed study | Implication | Next question |
| --- | --- | --- |
| BPI predicted mean resolution at 14 days is approximately 0.525 versus 0.579 observed. | Coupling fixed individual probabilities cannot correct their expected count. | Does simple chronological recalibration explain the decision-loss problem? |
| Typed factors improve Borg CRPS by approximately 2.2% beyond the global-factor comparator. | Most of the improvement over independence is shared global variation. | Under which conditions does observed relation structure add information? |
| Borg static-factor macro coverage is approximately 0.906, but population coverage is approximately 0.763. | Macro averages can conceal unreliable projections. | Can reliability improve where decisions are actually made? |
| Historical aggregate residuals have the lowest mean BPI aggregate CRPS. | Direct aggregate forecasting is a serious competitor. | Can coherent entity scenarios match its accuracy and provide additional useful capabilities? |

Consistency and predictive accuracy are separate properties. The existing sampler already generates outcomes that can be aggregated consistently. The research extension must improve accuracy and reliability while preserving valid outcomes.

## 2 Novelty assessment before method development

Forecast reconciliation already addresses consistency between predictive distributions. Existing work includes discrete counts, general aggregation constraints, conditioning, score-based optimization, and time-varying weights. Overlap alone is not evidence that existing approaches are unsuitable.

Relevant starting points are [Efficient probabilistic reconciliation of forecasts for real-valued and count time series](https://link.springer.com/article/10.1007/s11222-023-10343-y), [Discrete forecast reconciliation](https://arxiv.org/abs/2305.18809), and [Scaleable Dynamic Forecast Reconciliation](https://arxiv.org/abs/2409.12856). Current-state operational simulation and uncertainty also have close prior work, including [Avramenko et al.](https://link.springer.com/article/10.1007/s44311-026-00045-9).

Build a comparison matrix covering changing cohort size, overlapping membership, newly appearing groups, bounded discrete outcomes, monotone multihorizon trajectories, history requirements, runtime, and memory. Verify restrictions from each method's formulation and code. Where possible, implement a small example before asserting a limitation.

The potential contribution lies in the combined operational setting and a demonstrated capability under its constraints. If an established method handles the task accurately and efficiently, use it and revise the contribution rather than rename it.

## 3 Diagnose bias and dependence separately

Compare the current marginal predictions and chronologically recalibrated predictions under independent, global-factor, and typed-factor sampling. Begin with simple global and state-specific corrections, then a modest correction conditioned on observable context.

Each update may use only outcomes that have become observable by that update time. A 14-day target does not become an available label simply because its forecast origin is earlier than the current origin. Reconstruct label availability explicitly.

Measure individual Brier scores, group mean bias, CRPS, interval coverage and width, and the existing reserve proxy. Break results down by horizon, projection, group size, activity regime, and whether the group is new. Define strata using training information or attributes available at the origin.

This experiment should determine whether correcting marginal bias removes the BPI decision-loss increase, whether graph factors still help afterward, and whether the benefits are concentrated in particular group types. Bias is a plausible explanation, not a demonstrated sole cause of the current loss increase.

If recalibration plus a global factor explains the gains, pursue the empirical contribution rather than force a graph-method claim.

## 4 Candidate method if the diagnostics justify one

Retain the strongest existing individual predictors as the starting point. Add a small correction mechanism using current state, recent activity, observed group composition, and previously observable forecast errors. Share parameters across groups instead of memorizing entity identities.

Correct hazards or exit-bin probabilities so the resulting CDF remains valid. Corrected probabilities are new marginals; a method that changes them must not claim to preserve the original biased marginals.

Generate one exit bin or survival outcome per entity and scenario. Derive every group count from those same entity outcomes. This gives integer counts, valid bounds, consistent aggregation, and monotone trajectories across horizons. Those properties are validity mechanisms, not new mathematical discoveries.

Investigate collective forecasting objectives alongside individual accuracy. Candidate objectives include group-count CRPS and multihorizon scores, with regularization limiting unjustified changes to individual forecasts. Begin with a small number of correction and dependence parameters. Establish improvement with exact discrete evaluation before adopting a large neural model or a relaxed training objective.

Discrete sampling complicates gradient estimation. A prototype must compare its optimization surrogate with exact sampled-score behavior and monitor optimization variance. If a simpler staged calibration and dependence fit works equally well, retain that simpler method.

Potential substantive capabilities are accurate prediction for newly appearing groups, useful forecasts for aggregation queries not directly trained on, or efficient inference for large changing cohorts. Each requires an explicit experiment.

## 5 Baselines and fair comparisons

| Comparator | Question it answers |
| --- | --- |
| Recalibrated marginals with independent sampling | Is correcting individual bias sufficient? |
| Recalibrated global and typed factors | Is a small extension of the current model sufficient? |
| Direct aggregate count forecasting | Is direct prediction preferable when the user needs group outcomes? |
| Historical residual resampling | Does the existing strong simple comparator retain its advantage? |
| Probabilistic or discrete reconciliation | Does an established coherence method solve the same task? |
| A relevant published temporal or process model | Does stronger temporal modeling or domain simulation explain the remaining improvement? |

Adapt competitors to identical outcome semantics, origins, horizons, and information availability. Document adaptations and tuning budgets. A domain-focused workflow paper should prioritize current-state process simulation; a joint forecasting paper should prioritize reconciliation and an expressive joint comparator.

For methods that cannot handle full cohorts, evaluate all competitors on the same smaller cohorts and report full-scale efficiency separately. A failure of one implementation is not proof of an inherent limitation of its method family.

## 6 Application evaluation that needs joint uncertainty

Use early warnings of collective backlog risk as an initial application. Given the population active now, estimate the probability that several observed groups simultaneously retain unusually large unresolved cohorts.

Candidate outcomes are simultaneous threshold exceedance, unusually large maximum group backlog, warning accuracy under a fixed false-alarm budget, warning lead time, and a declared tail-sensitive planning loss. Fix thresholds and operating budgets from development data or documented application requirements.

The dependence model should be useful for these joint events. Additive decisions based only on means or isolated group probabilities would not establish the need for a joint scenario generator.

BPI handler membership is historical association, not worker occupancy. Borg first state exit is not necessarily task completion. Retain these semantics and avoid presenting retrospective warning experiments as observed staffing or scheduling benefits.

## 7 Fresh evaluation data

The existing final outcomes have influenced the extension hypothesis. They can support diagnostics, but cannot serve as untouched confirmation for the new method. Preserve the original first-study results and protocol separately.

| Dataset | Planned role |
| --- | --- |
| Borg 2011 | Existing diagnostic evidence and development for the extension |
| BPI 2017 | Existing diagnostic evidence and cross-application development |
| Selected Google ClusterData 2019 cells | Preferred fresh cluster replication across cells |
| BPI 2015 municipalities | Alternative or additional workflow replication across organizations |

[Google ClusterData 2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md) covers eight cells during May 2019. The complete release is approximately 2.4 TiB compressed, so first assess bounded event-table extraction, access requirements, storage, and query cost. Select cells and periods before examining evaluation outcomes. Different Google cells broaden site evidence without establishing independence from the provider's operating practices.

[BPI 2015](https://research.tue.nl/en/datasets/bpi-challenge-2015-municipality-4/) contains building-permit processes from five municipalities over approximately four years. Audit resolution semantics, event ordering, and attribute availability before committing to its use. If adopting it, separate development organizations or periods from untouched evaluation organizations or periods.

Add one trustworthy new source first. Freeze partitions, model choices, tuning budgets, metrics, and primary comparisons before opening its evaluation outcomes. Any evaluation used to guide subsequent changes becomes development evidence for those changes.

## 8 Evidence required for a stronger paper

Require competitive performance against the strongest relevant comparator, useful reliability without indiscriminate interval widening, a graph contribution that survives stronger global controls, improved collective-risk predictions or declared loss, and practical behavior on new groups or sites.

Average seeds within forecast origins before temporal uncertainty analysis. Report projection and horizon results alongside macro scores. Repeated entities, seeds, and overlapping windows do not create independent organizational replications.

Test newly appearing groups and aggregation queries not directly optimized in training. Coherence for an additional query follows from entity sampling; predictive accuracy does not. Show both.

Choose practical improvement thresholds and uncertainty procedures before final evaluation. Do not search over group weights or decision penalties until a preferred model wins.

## 9 Stopping and redirection rules

- If recalibration plus a global factor matches the extension, simplify and pursue the empirical-paper route.
- If existing reconciliation is accurate and practical, make its operational adaptation central.
- If graph benefits are confined to a subset of groups, explain and scope that finding.
- If gains disappear on fresh data, revise the hypothesis and report the failed transfer.
- If a dataset cannot support reliable labels, change the dataset before building more model complexity.

Publication does not require winning every metric. It requires a consequential question and conclusions supported by the comparisons.

## 10 Prompt learning and the future dataset factory

Prompt learning is relevant if adaptation with few labels becomes a primary requirement. It would then need comparisons with recalibration, adapters, and regularized fine-tuning. It is not part of the initial extension.

The reusable factory components are timestamp-aware adapters, outcome definitions, information-availability checks, scenario validity tests, projection diagnostics, and versioned provenance. Full future event logs, arrivals, repeated transitions, and synthetic-training-data utility are separate extensions. RF remains excluded.

## Sequence and planning estimates

| Phase | Deliverable | Planning estimate for one focused researcher |
| --- | --- | --- |
| Diagnosis and novelty assessment | Failure analysis, literature matrix, simple correction controls | About one week |
| Stronger comparisons | Direct aggregate and reconciliation baselines | One to two weeks |
| Candidate extension | Small prototype and decisive ablations | One to two weeks |
| Fresh validation | Audited adapter and frozen evaluation | One to three weeks |
| Evidence-based rewrite | Claims, figures, limitations, reproducibility | About one week |

These estimates depend on data access and baseline adaptation. Two GPUs should support the proposed modeling scale; data semantics and evaluation design are likely to require more attention than raw training capacity. The first scientific step is the bias-versus-dependence diagnostic, after the proposed comparison has been specified.
