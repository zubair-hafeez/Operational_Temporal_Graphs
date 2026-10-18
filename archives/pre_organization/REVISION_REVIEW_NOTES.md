# Revised paper for author review

Read `paper_revision/main.pdf`. The revised manuscript is a complete empirical paper on collective forecasting for temporal operational graphs. Its defensible contribution is a reproducible evaluation framework and application evidence: distinguish individual mean bias, shared uncertainty, typed graph incidence, and joint group risk. It does not introduce a new copula or establish that a temporal graph neural model universally wins.

## What is substantially stronger

| Earlier weakness | Completed revision |
| --- | --- |
| Independence was an insufficient main control | Global factors, direct count models fitted on the same recent development period, historical residuals, and workflow reconciliation are compared. |
| Workflow losses contradicted the apparent count improvement | Delayed same-version calibration isolates mean error; both beneficial and harmful reserve-quantile changes are reported. |
| Graph benefit was overstated by the independence comparison | Job/machine removals and incidence permutation locate the useful relation; the incremental global-to-typed improvement is stated separately. |
| Per-group scores did not validate joint risk | Group-profile energy, fair variogram and peak-deviation CRPS are evaluated. The rare binary alarm failed and remains disclosed. |
| Follow-up analyses reused known final periods | They are explicitly retrospective. BPI 2012 February was separately gated by an 82-asset protocol before outcome retention. |
| Reproduction covered only the first draft | Revised tables, fitted controls, score evidence, raw/processed fresh-replication assets, audit scripts and a clean-extraction verification are included in a separate package. |

## Findings to defend

1. **Borg:** typed factors reduce count CRPS 24.3% against independence, but only 2.2% against a global factor. Their .02587 CRPS remains better than the stronger recent direct model's .02853. Job incidence contributes much more than machine incidence.
2. **BPI 2017:** the stronger recent direct model reaches .04077 CRPS, essentially tied with delayed correction plus global dependence (.04072). The old uncorrected global model is .04595. The direct model improves counts but has worse peak-group-deviation CRPS than the original global model.
3. **Fresh BPI 2012:** global dependence improves count CRPS 12.7% against independence while worsening the reserve-loss proxy 20.4%. Development-selected bias-DM improves CRPS 37.7% and reserve loss 32.6% against that global model. This is a negative result for a universal factor-model claim and positive evidence for the evaluation argument.

The complete evidence has 28 Borg, 61 BPI 2017 and 28 fresh BPI 2012 forecast origins. Repeated entities and three sampling seeds do not create independent temporal replications. All 45 implementation tests pass. The artifact audit verifies all 264 original frozen files and 82 fresh frozen assets, 20 table-source hashes, final-origin alignment, CDF validity and the maturity delay for every fresh online update. Exact-versus-sampled direct CRPS differs by .000031 on Borg and .0000022 on BPI 2017. These are correctness checks, not an independent scientific review.

## Remaining scientific limitations

The evidence spans one compute provider and two releases of a loan workflow, not three independent organizations. Borg has only seven final days. Fresh BPI 2012 has 28 days and an individual model trained on only 4,854 earlier case–origin queries; its substantial mean drift limits conclusions about an optimally refreshed individual model. Long-block intervals describe these periods and are weak evidence for long-run generalization.

The operational score is an explicit remaining-cohort reserve proxy, not measured staffing or compute cost. Last handler is an association, not exclusive ownership. The task excludes future arrivals, repeated transitions and future memberships. Full crossed job–machine probabilistic reconciliation, TACTiS and complete process simulators are not implemented numerical baselines. Claims against those methods would be unsupported.

Alibaba 2018 was investigated before predictive fitting and deferred because missing terminal timestamps dominated some reconstructed active cohorts; their interpretation as continued running was unverified. BPI 2015 municipality 1 was deferred because our audit did not establish a reliable closure definition. Neither is used to manufacture a favorable additional result. See `revision/ALIBABA_FEASIBILITY.md` and the preserved replication design.

## Submission positioning

My assessment is that this is now a plausible application/empirical submission, subject to external peer review. It should be presented as **evaluation of collective forecasts on temporal graphs**, not as a new generative architecture or a guaranteed top-tier method paper. The manuscript already uses that positioning.

**PeerJ Computer Science is a reasonable broad computer-science venue to consider first.** Its published comparative-study example demonstrates room for empirical comparisons; that does not predict acceptance of this paper. See the publisher's [comparative learning study](https://peerj.com/articles/cs-832/) and [journal factsheet](https://static.peerj.com/docs/Discover_PeerJ_Institutional_Partnerships.pdf). I could not retrieve its current editorial-criteria page, so I do not rely on a claim that novelty is irrelevant to its review.

**Applied Network Science is a possible second fit if the network contribution is central.** Its current [scope](https://link.springer.com/journal/41109/aims-and-scope) includes temporal networks, statistical network methods and empirical applications, but requires network insights beyond routine technical use of a graph. Our job/machine ablations help; the modest incremental graph gain remains a scope risk. These are fit judgments, not journal rankings or acceptance probabilities. No submission, payment or external upload has occurred.

## Final author decisions

Review the target semantics and whether this empirical contribution matches your intended first-paper identity. Check the abstract, Tables 1–3 and the limitations first. Supply actual authors, affiliations, funding, contributions, conflicts and any applicable ethics statement; choose a venue and its format. Choose a permanent artifact destination and a source-code license after checking third-party terms. The local archive is ready to review but is not a public repository or DOI.

Substantial AI assistance was used for research planning, implementation, experiments, analysis and drafting. A submission disclosure should describe that scope accurately and follow the selected venue's policy. A possible author-reviewed statement is: “The authors used OpenAI Codex for assistance with research planning, code development, experiment execution, analysis and manuscript drafting. The authors reviewed and verified the final methods, results, references and text and take responsibility for the work.” The second sentence becomes true only after your review; do not submit it automatically.

The earlier PDF and archive remain unchanged. Their historical-residual one-seed caption was a reporting error: code and score files used three seeds. The revised paper corrects the caption without changing the numerical results.
