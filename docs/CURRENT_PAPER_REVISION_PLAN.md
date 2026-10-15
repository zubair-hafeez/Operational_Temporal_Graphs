# Revision plan for the current operational forecasting paper

Saved 15 October 2026. The objective is to make the existing manuscript a more convincing empirical application paper. This plan proposes revisions and additional experiments; it does not record completed improvements or predict an acceptance probability. The [broader research extension](RESEARCH_EXTENSION_PLAN.md) is a separate option if a new method contribution is needed.

The strongest existing evidence is a controlled comparison of joint forecasts with identical individual probabilities, relation-incidence interventions, two applications, and reproducible future-period evaluation. The principal weaknesses are limited methodological novelty, incomplete strong comparisons, uncertain practical implications, and restricted site and temporal replication.

## 1 Make the contribution specific

Organize the paper around three questions: when dependence improves collective forecasts, what observed relations add beyond global variation, and when forecast-score gains survive a declared decision objective. A possible title is **When Dependence Helps Collective Forecasts in Operational Temporal Graphs**. Choose the final title after the added evidence is available.

The introduction should explain the practical target in one concrete example, then state the contribution as a forecasting task, controlled comparison, and empirical finding. Present the current models as reference methods. State precise differences from operational simulation and forecast reconciliation instead of relying on the absence of an identical benchmark name.

Use the actual strongest comparisons in the abstract. The 24.3% cluster CRPS gain is versus independence; the graph-specific improvement over a global factor is approximately 2.2%. BPI selects a global joint factor and the residual comparator has the lowest mean aggregate CRPS. Those facts determine the scope of the contribution.

## 2 Add the two most consequential baseline families

First, add a direct aggregate forecasting model using observed cohort composition, group size, recent history, and summaries of individual risk. A bounded count model such as a beta-binomial predictor is a starting point; select and audit the distribution against the task rather than assume it fits. Include a multihorizon treatment or disclose separately fitted projections. A direct model does not automatically produce a consistent entity-level joint law.

Second, implement a suitable established probabilistic or discrete reconciliation method, beginning with [conditioning-based reconciliation](https://link.springer.com/article/10.1007/s11222-023-10343-y) or [discrete forecast reconciliation](https://arxiv.org/abs/2305.18809). Determine its actual applicability to the current outcomes and overlapping groups before adapting it. Use matched-size experiments if full-scale execution is impractical.

Retain the historical residual comparator, independent sampling, global factors, typed factors, and the incidence control. A relevant current-state process-simulation comparator becomes especially important if the final paper primarily claims workflow simulation value. [Avramenko et al.](https://link.springer.com/article/10.1007/s44311-026-00045-9) is a close reference for that route.

Give all newly compared methods a declared historical information budget, tuning budget, target, horizon and cohort definition. Construct adequate complete training cohorts for aggregate models rather than train a complicated model on only a handful of development origins. Preserve the distinction between the original frozen comparison and any revised experimental protocol.

## 3 Investigate the banking failure

Compare the original and chronologically recalibrated marginal predictions under independent, global and typed sampling. Examine bias, dispersion, group sizes, forecast horizon, and activity regime. Use only matured historical labels for calibration.

The question is whether marginal bias, dependence misspecification, the selected decision quantile, or their interaction explains the reserve-loss increase. The completed study does not establish a single cause. Do not tune against the observed final loss and call the resulting evaluation untouched.

If simple recalibration fixes the problem, report that result as a useful practical finding. If it does not, explain which parts of the distribution remain wrong. Retain negative results and distinguish lower average CRPS from improvement at a particular quantile.

## 4 Strengthen the graph evidence

Report incremental gains over a global factor and direct aggregate forecasting, not just independence. Analyze where true incidence helps: job versus machine versus handler, group size, horizon, and groups with little history.

Retain the existing risk-stratified permutation and inspect whether important group-size and risk characteristics remain comparable. Predefine any additional relation controls before inspecting their new evaluation outcomes. Incidence interventions establish properties of the fitted predictive model, not causal effects of moving tasks or reassigning cases.

A credible outcome may be that graph structure helps a specific class of collective outcomes. Scope the claim to that result. If the graph contribution is negligible against stronger controls, reposition the paper around operational joint forecasting rather than asserting broad temporal graph superiority.

## 5 Add an operational evaluation that uses dependence

Evaluate the probability of simultaneous backlog threshold exceedances across several current groups, or unusually large maximum group backlog. Add warning performance under a fixed false-alarm budget if the event semantics support it. These outcomes use joint uncertainty, whereas many additive mean-based objectives do not.

Choose thresholds from application documentation or development data and identify their origin. Do not present selected thresholds as observed organizational service-level agreements. Keep current-cohort forecasting distinct from total future workload, which also depends on new arrivals and later transitions.

Continue reporting the reserve proxy, but describe its normalized units and penalties plainly. No staffing or scheduling savings are established by replaying outcomes that were generated under a different policy.

## 6 Add fresh confirmation if the intended claim requires generalization

The existing Borg final period spans seven days and BPI covers one bank. Prefer a bounded independent replication with additional Google 2019 cells, or an audited workflow source such as BPI 2015 municipalities. New Google cells broaden site evidence within the same provider; they do not establish new-provider generalization.

For any new model motivated by the original final results, use a newly frozen evaluation on previously unexamined outcomes. Additional analyses of the original final periods can be informative, but must be labeled exploratory or retrospective. The original frozen results remain intact.

An additional dataset is most useful when it tests the same substantive hypothesis. If a trustworthy new source is infeasible, narrow the generalization claim and favor a focused specialist empirical submission rather than imply evidence that is missing.

## 7 Improve uncertainty reporting and presentation

Make the primary comparison and primary outcome explicit. Show paired differences with time-block intervals, projection-level coverage and width, and horizon-specific results. Explain why macro coverage can conceal poor population coverage. Keep block-size sensitivity visible, especially for the banking decision-loss result.

Separate development selection from final evaluation and retrospective analyses. Use plain descriptions of application semantics, data availability, and benchmark weights. Consolidate repeated qualifications into the appropriate methods or limitations paragraph while retaining the qualifications that materially change a result's interpretation.

Organize the main figures around the scientific story: marginal bias, incremental graph contribution, and collective-risk or decision behavior. Put secondary configurations and implementation details in the appendix. The paper should read as an argument supported by experiments, rather than a catalog of completed runs.

## 8 Review the evidence before choosing the submission

Arrange a technical review focused on the contribution, comparator fairness, outcome semantics, leakage, and inference from short dependent traces. An independent reader should be able to reconstruct a small cohort from raw events and understand every main-table number. Run the additional baselines from a documented environment and extend the reproduction package to include the revised experiment provenance.

Select the venue after the final contribution is clear. Assess its current scope, recent comparable papers, article type, length, fees, artifact requirements and disclosure policy. Venue fit can improve the chance of useful review, but does not remove a technical weakness.

## Priority and acceptance criteria

| Priority | Work | Evidence required before treating it as complete |
| --- | --- | --- |
| 1 | Sharper contribution and closest-work comparison | The paper states what is new or newly demonstrated, and why it matters. |
| 2 | Direct aggregate and reconciliation baselines | Strong relevant alternatives are evaluated fairly or their exact adaptation limits are documented. |
| 3 | Banking failure diagnosis and graph ablations | Claims about bias and relation value follow controlled comparisons. |
| 4 | Joint operational outcome | The evaluated application quantity actually depends on collective uncertainty. |
| 5 | Fresh replication | New claims receive confirmation beyond the outcomes that motivated them, or are explicitly scoped to exploratory evidence. |
| 6 | Review and venue-specific rewrite | Main claims are reproducible, readable and appropriate for the selected audience. |

The minimum useful revision combines priorities 1–3 and a stronger presentation of application relevance. A more convincing application-journal submission should also seek priorities 4–5. If a simple established model wins, the empirical finding can remain the contribution. A new architecture, prompt module, diffusion model, or full dataset factory is not required to complete this revision plan.

The immediate next experiment, if later authorized, is the controlled recalibration comparison. The immediate comparison to implement alongside it is direct aggregate count forecasting. Neither has been launched by saving this plan.
