# Directed Research - MSc FE

Research on volatility relaxation after shocks: can a power-law decay be distinguished from exponential alternatives, and do causal shock features improve forecasts? This repository contains research code, synthetic tests, specifications, aggregate results and original analytical figures. It is a curated publication copy, not the private working archive or the final course report.

## Methods

The relaxation models are a single exponential (E1), a two-exponential mixture (E2), a shifted power law (PL) and a stretched exponential (SE). Responses are normalized against a pre-shock baseline, with 250 valid pre-onset sessions required by the pooled protocol. Bounded nonlinear least squares uses 32 starts; comparisons include fit starts at sessions 1 and 5 without resetting time. Boundary contacts are retained and affect interpretation. The historical pooled analysis considers 61-, 120- and 250-session windows, full and common-maturity panels, and event versus overlap-group resampling.

The sample-size experiment uses four generating truths, N = 19, 40, 80, 160, 320, independent and calendar-joint noise, and 200 replications per cell. N80 requires at least 80% true-PL recognition and at most 10% false-PL selection under each alternative at **both** fit starts. Missed-PL and false-PL rates are separate outcomes.

The forecast experiment compares HAR and HAR-X on overlapping average-log volatility targets at 5, 10 and 20 sessions. Features use information available at the forecast origin, not completed-event fitted curves or eventual peaks. Monthly estimation and annual validation select the shock-decay parameter. Inference combines Diebold–Mariano tests with Newey–West lag h−1 and the implemented finite-sample correction, whole-crisis bootstrap, leave-one-crisis-out checks and Holm adjustment for 23 secondary tests. Prefix truncation and perturbed-future audits check leakage.

## Saved results

* **Historical recovery calibration:** OU, two-OU and ARFIMA failed the specified calibration/eligibility requirements; no process-recovery rates were estimated. The empty rate fields in `stages/S02b_recovery/` are unavailable estimates, not zero recovery rates.
* **Identification:** N80 was not attained through N=320 in either noise design. Under calendar-joint noise at start 5, PL recognition was 10, 9, 18, 12 and 22 out of 200. At N=320 there were 178 PL misses; false-PL counts under E1/E2/SE were 13/22/25 out of 200. See `extension/E02_sample_size/`.
* **Forecasts:** primary VIX30D, 10-session active-period MSE was 0.01469217 for HAR and 0.01523579 for HAR-X: 3.7001% higher error for HAR-X. The HAR-minus-HAR-X loss difference was −0.0005436212, with crisis-bootstrap 95% interval [−0.0009460393, −0.0001455143], 15 crisis units and 2,512 active origins. All 24 differences were negative; no positive claim survived the specified rule. See `extension/E03_forecast/tests.csv` and `leave_one_crisis_out.csv`.
* **Expanded fitted comparison:** eight Cboe index cohorts supplied 84 trajectories in only seven global overlap components. Nine common VIX events formed one component. E1 had the lowest start-5 descriptive score in 11 of 12 panels; E2 did in RVX full. The equal-index pool's E1 time constant was 42.1361 sessions, with 666 valid and 333 missing-index failed draws of 999. All planned intervals and maturity significance tests were ineligible under the ten-component/900-valid-draw rule. See `extension/E05_larger_sample/fresh_public_v5/`.
* **Separate expanded event catalog:** 64 eligible index-event rows became 50 distinct dates after merging shared dates across indices; all belonged to one full-footprint overlap component. These are not 50 independent crisis observations and are not the 84 fitted trajectories above. See `extension/E05_larger_sample/cross_index_count/summary.json`.

These are saved historical results, checked against the project's completed independent reviews when preparing this repository. Publication checks do not rerun the full numerical experiments or recertify every historical fit.

## Limitations

Small numbers of crisis clusters, persistent overlapping targets, nested forecast models, active parameter bounds, heterogeneous index definitions and retrospective data revisions limit interpretation. The sample-size experiment's 19 empirical overlap components were all singletons, so its production calendar-joint design does not demonstrate robustness to observed overlapping crises. Descriptive score rankings do not establish model superiority, a universal decay law, trading profitability or causality. The larger-sample fresh run was review-informed; the original run's pre-outcome freeze chronology was not established, and fresh wall-clock versus monotonic timing disagreed. Source snapshots are not real-time historical vintages. No licensed OptionMetrics or Bloomberg outcome is claimed.

## Setup and quick verification

Use Python 3.13 for the closest match to the saved extension environment. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/check_publication.py
.venv/bin/python tools/run_quick_checks.py
```

The checks exercise model identities, normalization, optimizer recovery, synthetic overlap construction, forecast recursion, cohort eligibility and mocked extraction safety. They require no source download, paid service or heavy simulation. Results and the actual publication test environment are recorded in `VALIDATION.md`.

See [REPRODUCING.md](REPRODUCING.md) for workflows and known historical runner limitations; [DATA_ACCESS.md](DATA_ACCESS.md) for official retrieval locations and snapshot requirements; and [EXCLUSIONS.md](EXCLUSIONS.md) for the publication boundary. Original filenames retain historical stage labels to preserve imports; the course and deliverable are called **Directed Research - MSc FE**.

## Code map

* `omori_pipeline/`, `v2_analysis/`, `v3_analysis/`, `v3_run_2026-10-20/`: historical acquisition, detection and exploratory fitting. These do not supersede the later frozen experiments.
* `stages/S01_protocol/attempt_02/protocol_core.py`: shared scientific contracts and fits.
* `stages/S02a_pooled/`, `stages/S02b_recovery/`, `stages/S02c_review/`, `stages/S03_validation/`: pooled relaxation, recovery calibration, sensitivity checks and historical validation. The continuation code is retained separately; use its intended root layout when reconstructing a continuation.
* `extension/E02_sample_size/`, `extension/E03_forecast/`: identification and causal forecasting experiments.
* `extension/E05_larger_sample/fresh_public_v5/`: final public-data descriptive comparison; `cross_index_count/` is the separate date-merged inventory.
* `extension/E04_more_shocks/` and `wrds_extract/`: source inventory and optional historical extraction utilities. They are not dependencies for the public comparison; extraction requires separate legitimate access and confirmation of schema, security identity and rights.

`SOURCE_MAP.json` records copied source paths and hashes. No source repository history was imported. No open-source or third-party data redistribution licence is granted by this private repository.
