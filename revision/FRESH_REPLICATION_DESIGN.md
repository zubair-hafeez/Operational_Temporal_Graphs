# Fresh replication design, before final outcomes

BPI 2012 is the next bounded replication candidate. It supplies another loan-processing period; it must not be described as another independent bank without provider evidence. Source DOI: https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f . Dataset metadata and compressed-file checksum are recorded separately. No RF data or tasks are used.

## Why the municipal candidate is deferred

The municipality 1 audit found 1,199 cases and 52,217 events, inconsistent English labels under the same activity codes, many different administrative milestones, and explicit close-case events in only a minority of traces. Several timestamps differ from dateFinished, and trace-level status/end-date fields are retrospective. The first informal label inspection mistakenly sought `taskNameEN`; the actual event key is `activityNameEN`, and the corrected inspection exposed the ambiguity. This is not evidence that the dataset is unusable; it needs a separate target/availability study. Municipalities 4 and 5 remain unopened. Do not force a convenient completion target.

## Temporal gates

Before parsing BPI 2012 outcomes, reserve all outcomes at or after **2012-02-01 UTC** for final confirmation. Development preparation may retain events strictly before that date; an XML parser necessarily scans archive bytes, but must not report or retain later activities/outcomes. Do not inspect final-period outcome statistics or model performance until the executable protocol, fits, selected configurations and primary comparisons are frozen. If documented archive coverage cannot support the planned horizon, record a feasibility failure rather than move the boundary to obtain a favorable result.

Tentative windows, to be finalized using development evidence only: individual predictor fit before 2011-11-01; earlier calibration during November; dependence/direct-count fitting on complete December cohorts; model selection on January cohorts whose 14-day outcomes fit within January. Final origins start 2012-02-01 and end at least 14 days before the documented observation endpoint. Score 1/3/7/14-day first documented application resolution, defined by an explicit application milestone (approval, cancellation or decline), not last trace event. All current-case membership and last-handler relations come from the observed prefix.

## Hypotheses to confirm

1. Individual forecast accuracy/calibration alone does not establish calibrated population/group count uncertainty.
2. Simple shared dependence and direct count/reconciliation models are strong alternatives to typed graph dependence; incremental graph benefit must be measured against them.
3. Delayed recalibration can affect operational behavior materially; updates must use matured predictions from the **same fitted predictor version**, rather than blindly combine errors from changing OOF predictors.

The model-version restriction follows a retrospective BPI 2017 diagnostic. It is an evaluation/control requirement, not a claimed new calibration algorithm. Retain identity/static/delayed-update controls and all contrary evidence. The primary final comparison and direction of claims will be frozen after development. A new neural architecture is not presumed necessary.
