# BPI 2012 confirmation analysis

This protocol is written after development evaluation and before parsing any February outcomes. The original Borg/BPI 2017 final results and the revision diagnostics are known. The confirmation tests a fixed procedure on a new period/source, not a new independent provider. It does not establish acceptance or a novel architecture.

## Forecast task and endpoints

Daily 12:00 UTC origins February 1–28, 2012; 1/3/7/14-day first documented A_APPROVED, A_DECLINED or A_CANCELLED among cases already submitted and not yet resolved. Use the archive through March 14 00:00 UTC. No new arrivals enter a current-cohort target; no last-trace-event target. January 19–31 origins are generated only to supply matured online-calibration history. All trace-level retrospective attributes are excluded.

Primary outcome: fraction CRPS, equally weighted over origins, population/handler projections, four horizons, and eligible groups within a projection. Handler groups have at least five active cases. Use 512 scenario draws and all three seeds; never choose a best seed. Report coverage, width, reserve proxy ratios 3/9 and individual Brier/bias alongside the primary outcome.

Joint secondary outcomes: RMS-normalized energy score of the full current-handler count vector, fair variogram score (power .5, 2,048 outcome-independent ordered group pairs per horizon), and CRPS of the maximum standardized slow-progress residual over handlers. All methods use the same original individual-forecast centering/scaling for that last outcome. These are distributional forecast evaluations, not demonstrated staffing savings or organizational SLA predictions. The initial two-standard-deviation simultaneous-alarm endpoint was nearly absent in the retrospective banking data and is not a primary confirmation endpoint; do not conceal that failed diagnostic.

## Frozen predictors and comparisons

HGB feature arm and identity/static correction selected solely by January individual Brier: **personal plus global features, identity correction**. Three feature arms and their fixed November intercept corrections remain in the development record. HGB parameters: 250 iterations, 15 leaves, minimum leaf 50, learning rate .05, L2 1, no random early-stopping split. Individual training labels end before November 1. November labels fit static calibration. December complete cohorts fit dependence and direct count models; January pair likelihood selects dependence checkpoints.

The development-selected Gaussian configuration by aggregate CRPS is **identity marginals plus global dependence**. Compare it primarily with identity marginals plus independence. The development-selected direct count distribution is **bias-adjusted Dirichlet–multinomial**; compare it with the selected Gaussian and independence. The best conditioning baseline on development is **bias-DM conditioning**, retained alongside its bottom-up control. Report all prespecified modes, including static typed dependence and identity/static/rolling60 correction, regardless of outcome.

Rolling60 correction uses only fully matured 14-day windows from the same individual predictor version, with equal-origin Bernoulli loss and an expanding fallback if fewer than four origins occur within 60 days. Fit dependence separately for each correction treatment on December. Do not refit individual predictors or direct count parameters after opening February. Corrections may update online using earlier matured February labels as explicitly permitted here.

## Inference and limitations

Average seeds within origin first. Report paired mean differences with moving-block bootstrap intervals, using seven-day blocks primarily and fourteen-day blocks as sensitivity (10,000 replicates, fixed seed). The test has only 28 origins and four/two such blocks, so intervals are descriptive and weak evidence of long-run generalization; report the short duration prominently. Do not pool individual cases as independent statistical replicates or report seed variation as temporal uncertainty. Primary contrasts are selected global versus independence and selected direct-count versus selected global; other contrasts are secondary. No favorable-sign requirement, hidden model switching, or additional tuning on February.

Preserve every frozen source and fitted parameter checksum. A correctness failure requires a separate documented correction retaining originals, not silent alteration. Complete the original 264-file integrity check too. If a simpler method wins or calibration fails under drift, report that as evidence about the applicability of the approach.
