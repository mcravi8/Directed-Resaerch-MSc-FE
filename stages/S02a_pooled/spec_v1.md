# S02a execution specification v1

Authorized trigger: October 21, 2026, 8:05:42 PM America/Los_Angeles.
Hard deadline: October 22, 2026, 4:05:42 AM; analysis shutdown 3:35:42 AM.
All pauses, implementation, retries and exports count. This run executes S02a only.

The exact S01 analysis_protocol.md and recovery_protocol.md are incorporated in
full and frozen in this stage's manifest, together with S01 spec_v1.md, input
manifest, revision3 S02a brief, this file and all executable scripts before any
new outcomes. Known review outcomes were read during intake: review-informed,
not untouched preregistration. S01 Chief PASS and independent Vera PASS checked.

Four models E1/E2/PL/SE, bounds, 32 keyed starts, 1e-8 TRF tolerances and 5000
evaluations per start are inherited without alteration from protocol_core.py.
Starts are exactly 1/3/5; time remains elapsed since anchor. Horizons 120 primary,
61/250 sensitivity, free baseline primary and zero baseline sensitivity. Equal
event log-normalized trajectories; 250 strictly pre-onset observed closes,
denominator >1e-8; no undershoot flooring or imputation. Reject incomplete
post-anchor observed-date grids and truncate at next onset. Historical merged
catalog only for empirical S02a. Full per-maturity and four-maturity common-ID
panels are separate; group footprints span pre-onset baseline through horizon.
499 full-trajectory event resamples and 499 overlap-group resamples per empirical
panel; paired common panels share sampled IDs. Group counts <10 are descriptive.
All four models refit for every start and baseline. Report failed starts and
boundaries. Parameter-count score is descriptive, not an independence-based test.

Timing pilots use scalar 30D analogue, seeds keyed timing/process, 10000 retained
sessions. Exact stationary OU initialized from its invariant Gaussian distribution,
5000 discarded sessions. Two independent OU components have total SD .4 and fast
variance fraction .25. ARFIMA d=.4 uses exact stationary Toeplitz covariance by
circulant embedding with nonnegative-eigenvalue check, no finite filter; retain
10000 after discarding 5000. Theoretical covariance verified independently before
generation. Mean log(20), OU tau=90, two OU taus=(5,90). Date grid repeats saved
calendar gaps with a weekday-aligned cycle seam; no historical-name merges.
Original raw detector >=7 one-session OR >=10 five-session, cooldown >30 calendar
days, first argmax onset through onset+10 inclusive. Both argmax and onset+5
anchors, intersect eligible IDs, 199 shared event-bootstrap refits per panel;
all horizons, baseline sensitivities and starts use the same pipeline as empirical
fits, with the primary 120/free/start1 used for selection. Fewer than 10 eligible
events is undetermined; no fit is invented for an ineligible panel. Pilot outcomes
are synthetic timing evidence and never confirmatory recovery denominators.
Selection requires score advantage >=2 against all alternatives, bootstrap wins
>=.9; collapsed E2 and E1-like PL/SE are undetermined under inherited definitions.
U_p=2*T_p; N=min(100,floor(21600/(3*U_OU+4*U_2OU+2*U_LM))).
N=0 means PARTIAL; never force N=1. Full wall time includes generation, detection,
fitting, scoring and checkpoint writing. No extrapolation of incomplete pilots.

Implementation choices: sequential deterministic units, immutable JSON per unit,
full start diagnostics retained, process isolation and absolute-deadline watchdog.
Original diagnostic scripts copied unchanged and executed diag1, diag3, diag5,
diag6. Unsupported full-sample ACF claim is explicitly unsupported, not rerun
as an unplanned analysis. Original ML is skipped. Diagnostic claim reconciliation
and independent numerical verification are mandatory before PASS.
S03 validation, S02b recovery grid, report rendering, WRDS, paid data and external
communication are not applicable. Failed prerequisites or scientific verification
stop this run with an explicit checkpoint. Future runs require a fresh trigger.
