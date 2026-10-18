Folder organization update: the latest manuscript remains `paper_revision/main.pdf` (10 pages; shortcut `LATEST_PAPER.pdf`). References below to verified ZIP archives describe the completed historical packaging checks. The duplicate ZIP files were subsequently removed at the user's request; all unique member bytes are retained and portable packages can be reconstructed with `scripts/organized_review.py`. See the project-root `START_HERE.txt`.

# Revision study

The revision is complete for author review. Read `../REVISION_REVIEW_NOTES.md`, `../REVISION_REPRODUCE.md` and `../paper_revision/main.pdf`. Both frozen asset sets verify; all45 tests pass; the separate review ZIP passed clean-extraction reproduction of14 generated artifacts and PDF compilation. The material below records the initial diagnostic plan and its provenance. BPI2012 supplied fresh confirmation; BPI2015 and Alibaba candidates were deferred for adapter/endpoint issues, as documented in their audit notes.

Execution authorized 2026-10-15 after the original study and its final outcomes were reviewed. Original frozen assets and the immutable review archive are preserved. New analyses of those periods are **retrospective follow-ups**, not a second untouched test. RF remains excluded.

Immediate registered diagnostic: compare identity, static shared-logit recalibration, and delayed rolling shared-logit recalibration of the saved BPI individual CDFs, crossed with independent, global and typed Gaussian dependence. Fit the static map on the original June–August out-of-fold forecasts; rolling maps use only origins whose entire 14-day outcome window has matured, with a 60-day lookback and an expanding-history fallback if fewer than four origins are available. One shared logit intercept preserves horizon ordering. Fixed original dependence weights isolate the effect of marginal correction; they are not refitted in this diagnostic. Equal-origin, equal-horizon Bernoulli loss fits the intercept. No final-outcome-driven candidate selection.

Direct aggregate comparator: fit conditional Dirichlet–multinomial count distributions on complete out-of-fold cohorts. The five categories are four first-exit intervals plus survival. Pool changing groups by projection type, using their mean individual category probabilities as offsets. Compare concentration-only versus four category-bias parameters plus concentration, with weak fixed regularization. Select by original development count CRPS; report both in retrospective final follow-up. This preserves each group's integer bounds and horizon order, but separately forecast overlapping projections need not describe a common entity realization. A reconciliation comparison remains required.

Fresh replication candidates: audit the public BPI 2015 municipality 1 development source first. Municipalities 4 and 5 are reserved: do not inspect their outcomes before freezing a concrete protocol. Dataset suitability, target semantics, censoring, and sample size must be established before using them as confirmation.

## Reporting correction

The original historical-residual evaluator iterates over seeds 0, 1 and 2, and both saved final score files contain all three. The later manuscript caption claiming one seed is incorrect. Numerical results are unchanged. Correct the caption in the revised manuscript; preserve the original archive.

## Execution records

Revision code lives here; generated data and runs use dedicated revision directories under `data/processed/` and `runs/`. Record checksums and settings per run, refuse output overwrites, and verify the original 264-file freeze after changes.
