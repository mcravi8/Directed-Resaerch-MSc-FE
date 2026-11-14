#!/usr/bin/env python3
"""
v2 fits: Omori power law vs exponential decay of post-shock implied volatility.
Implements verification/v2_fit_spec_FROZEN.md (D-12, D-13, D-14, D-21).

STATUS: UNTESTED. Written by Chief on 2026-10-19 from the frozen spec and the
original scripts. It has NOT been executed yet. The first coding session must
run it, fix any crash without changing the frozen rules, and record any code fix
in the run log. Any change to a RULE must be logged in decision_log.md first.
Static review 2026-10-19 22:05 (Chief): input column names checked against
shock_events.csv, decay_fits.csv and 03_fit_decay.py; one crash risk fixed
(boolean flag columns became object dtype when any fit failed, so `~` on
Python bools would give -2/-1 instead of True/False). Code-only fix, no rule change.

Safety: the script copies its inputs into <out>/inputs_copy/ and reads only the
copies. It refuses to write anywhere inside omori_pipeline/.

Usage (run from any directory that is NOT the data directory):
  python -I fit_v2.py \
      --ts      <path>/vix_term_structure.csv \
      --events  <path>/shock_events.csv \
      --v1fits  <path>/decay_fits.csv \
      --out     ./v2_out

Requirements: numpy, pandas, scipy; matplotlib optional (figures).
"""
import argparse
import json
import os
import platform
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- constants
MATS = ["VIX9D", "VIX30D", "VIX3M", "VIX6M"]
WINDOW = 60            # trading days, t = 1..60 (t=1 is the peak day, as in v1)
MIN_POINTS = 15
BASE_WIN = 20          # trading days before onset for per-maturity baseline
PEAK_SEARCH = 10       # trading days after GFC peak searched for the merged max
LATE_PEAK = 10         # W1: window max more than 10 trading days after start
P_B = (0.01, 10.0)
C_B = (0.01, 500.0)
L_B = (1e-4, 5.0)
K_EXP, K_PL = 3, 4     # primary: exp (b, A, lam); PL (b, K, c, p)
K_EXP_B, K_PL_B = 2, 3 # robustness B (baseline fixed, not estimated)
REL = 0.01             # "within 1%" tolerance for bound flags
OOS_SPLITS = [(20, "fit1-20"), (30, "fit1-30"), (40, "leave-last-20-out")]
V1_COMPARE = ["COVID", "Volmageddon", "Quant Quake", "Flash Crash", "Yen Carry", "GFC"]
BOOL_COLS = ["W1", "F1", "F2v2", "F3", "F5", "F6", "strict_R2_lt_0.5",
             "primary_valid", "strict_valid", "overlap_S2"]


# ---------------------------------------------------------------- models
def exp_full(t, b, A, lam):
    return b + A * np.exp(-lam * t)


def pl_full(t, b, K, c, p):
    return b + K * np.power(t + c, -p)


def exp_nob(t, A, lam):
    return A * np.exp(-lam * t)


def pl_nob(t, K, c, p):
    return K * np.power(t + c, -p)


# ---------------------------------------------------------------- helpers
def multistart(f, t, y, starts, lo, hi):
    """Least squares from every start; keep the lowest SSE."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    best, nconv = None, 0
    for p0 in starts:
        p0 = np.asarray(p0, float)
        span = np.where(np.isfinite(hi - lo), (hi - lo) * 1e-6, 1e-6)
        p0 = np.minimum(np.maximum(p0, lo + span), np.where(np.isfinite(hi), hi - span, np.inf))
        try:
            popt, pcov = curve_fit(f, t, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        except Exception:
            continue
        yhat = f(t, *popt)
        if not np.all(np.isfinite(yhat)):
            continue
        nconv += 1
        sse = float(np.sum((y - yhat) ** 2))
        if best is None or sse < best[2]:
            best = (popt, pcov, sse)
    return best, nconv, len(starts)


def se(pcov, i):
    try:
        v = float(pcov[i, i])
        return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else np.nan
    except Exception:
        return np.nan


def metrics(y, yhat, k):
    n = len(y)
    e = y - yhat
    sse = max(float(np.sum(e ** 2)), 1e-300)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    ll = -n / 2 * (np.log(2 * np.pi * sse / n) + 1)
    return {"SSE": sse, "R2": r2, "LL": ll, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll}


def pointwise_ll(y, yhat):
    e = y - yhat
    s2 = max(float(np.mean(e ** 2)), 1e-300)
    return -0.5 * np.log(2 * np.pi * s2) - e ** 2 / (2 * s2)


def vuong(y, yhat_pl, yhat_exp, k_pl, k_exp):
    """Vuong non-nested test with Schwarz correction. z > 0 favours power law."""
    n = len(y)
    d = pointwise_ll(y, yhat_pl) - pointwise_ll(y, yhat_exp)
    lr = d.sum() - (k_pl - k_exp) / 2.0 * np.log(n)
    w = d.std(ddof=0)
    if not np.isfinite(w) or w <= 0:
        return np.nan, np.nan
    z = lr / (np.sqrt(n) * w)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def bh(pvals):
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(m)
    qq[order] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q


def near(x, lo, hi):
    if not np.isfinite(x):
        return True
    return bool(x <= lo * (1 + REL) or x >= hi * (1 - REL))


def strength(d):
    a = abs(d)
    if not np.isfinite(d):
        return "na"
    if a < 2:
        return "inconclusive"
    side = "PL" if d > 0 else "exp"
    if a < 6:
        return side + "_positive"
    if a < 10:
        return side + "_strong"
    return side + "_decisive"


# ---------------------------------------------------------------- fitting
def fit_primary(t, y, base_m):
    """D-13 option C: baseline b estimated in the fit, no flooring."""
    vpk = float(np.max(y))
    ymin = float(np.min(y))
    bs = []
    for b in (base_m, ymin, 0.5 * ymin):
        if np.isfinite(b):
            b = float(np.clip(b, 0.0, vpk))
            if all(abs(b - x) > 1e-9 for x in bs):
                bs.append(b)
    exp_starts = [(b, max(y[0] - b, 1e-3), lam) for b in bs
                  for lam in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
    pl_starts = [(b, max(y[0] - b, 1e-3) * (1 + c) ** p, c, p) for b in bs
                 for c in (0.1, 1, 5, 20, 100) for p in (0.3, 1, 2)]
    ex, ex_conv, ex_n = multistart(exp_full, t, y, exp_starts,
                                   [0.0, 1e-8, L_B[0]], [vpk, np.inf, L_B[1]])
    pl, pl_conv, pl_n = multistart(pl_full, t, y, pl_starts,
                                   [0.0, 1e-8, C_B[0], P_B[0]], [vpk, np.inf, C_B[1], P_B[1]])
    return ex, pl, (ex_conv, ex_n, pl_conv, pl_n), vpk


def fit_fixed_base(t, y):
    """Robustness B: baseline subtracted beforehand, no flooring."""
    exp_starts = [(max(y[0], 1e-3), lam) for lam in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)]
    pl_starts = [(max(y[0], 1e-3) * (1 + c) ** p, c, p)
                 for c in (0.1, 1, 5, 20, 100) for p in (0.3, 1, 2)]
    ex, _, _ = multistart(exp_nob, t, y, exp_starts, [1e-8, L_B[0]], [np.inf, L_B[1]])
    pl, _, _ = multistart(pl_nob, t, y, pl_starts, [1e-8, C_B[0], P_B[0]], [np.inf, C_B[1], P_B[1]])
    return ex, pl


# ---------------------------------------------------------------- events
def build_events(ev, ts, log):
    ev = ev.copy()
    ev["label"] = ev["label"].fillna("").astype(str)
    leh = ev[ev["label"] == "Lehman"]
    gfc = ev[ev["label"] == "GFC Peak"]
    if len(leh) == 1 and len(gfc) == 1:
        onset = leh["onset_date"].iloc[0]
        g_peak = gfc["peak_date"].iloc[0]
        end_loc = min(ts.index.searchsorted(g_peak) + PEAK_SEARCH, len(ts) - 1)
        seg = ts["VIX30D"].loc[onset: ts.index[end_loc]].dropna()
        pk = seg.idxmax()
        base = float(leh["baseline"].iloc[0])
        row = {"onset_date": onset, "peak_date": pk, "peak_level": round(float(seg.max()), 2),
               "baseline": base, "magnitude": round(float(seg.max()) - base, 2),
               "onset_daily_chg": leh["onset_daily_chg"].iloc[0],
               "label": "GFC (merged Lehman + GFC Peak)"}
        log.append(f"D-12 merge: Lehman onset {onset.date()} + GFC Peak {g_peak.date()} -> "
                   f"merged peak {pk.date()} (VIX30D max {seg.max():.2f} searched "
                   f"{onset.date()}..{ts.index[end_loc].date()}). NOTE: the CSV onset of the "
                   f"Lehman row is {onset.date()}; the spec text said 2008-09-29, which is the "
                   f"Lehman PEAK date. The CSV onset is used.")
        ev = ev.drop(index=list(leh.index) + list(gfc.index))
        ev = pd.concat([ev, pd.DataFrame([row])], ignore_index=True)
    else:
        log.append(f"WARNING: could not merge Lehman/GFC (found {len(leh)} Lehman, {len(gfc)} GFC rows).")
    ev = ev.sort_values("peak_date").reset_index(drop=True)
    dates = ev["peak_date"].dt.strftime("%Y-%m-%d").tolist()
    labels = ev["label"].tolist()
    ev["event_id"] = [d + (" " + l if l else "") for d, l in zip(dates, labels)]
    log.append(f"Events after merge: {len(ev)} (spec expects 33).")
    return ev


def overlaps(ev, ts):
    locs = [int(ts.index.searchsorted(d)) for d in ev["peak_date"]]
    pairs, bad = [], set()
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            if locs[j] <= locs[i] + WINDOW - 1:
                pairs.append((ev["event_id"].iloc[i], ev["event_id"].iloc[j],
                              locs[j] - locs[i]))
                bad.update([ev["event_id"].iloc[i], ev["event_id"].iloc[j]])
    return pairs, bad


# ---------------------------------------------------------------- summaries
def summarize(df, view):
    rows = []
    for mat in MATS + ["ALL"]:
        d = df if mat == "ALL" else df[df["maturity"] == mat]
        n = len(d)
        pl_aic = int((d["dAIC"] > 0).sum())
        ex_aic = int((d["dAIC"] < 0).sum())
        pl_bic = int((d["dBIC"] > 0).sum())
        ex_bic = int((d["dBIC"] < 0).sum())
        sign_p = stats.binomtest(pl_aic, pl_aic + ex_aic, 0.5).pvalue if pl_aic + ex_aic > 0 else np.nan
        sign_p_bic = stats.binomtest(pl_bic, pl_bic + ex_bic, 0.5).pvalue if pl_bic + ex_bic > 0 else np.nan
        r = {"view": view, "maturity": mat, "n_fits": n,
             "AIC_PL": pl_aic, "AIC_exp": ex_aic, "AIC_sign_p": sign_p,
             "BIC_PL": pl_bic, "BIC_exp": ex_bic, "BIC_sign_p": sign_p_bic,
             "Vuong_PL": int((d["vuong_class"] == "PL").sum()),
             "Vuong_exp": int((d["vuong_class"] == "exp").sum()),
             "Vuong_indist": int((d["vuong_class"] == "indistinguishable").sum()),
             "median_p_validPL": float(d["pl_p"].median()) if n else np.nan,
             "count_claim_allowed": bool(n >= 20)}
        for cls in ["exp_decisive", "exp_strong", "exp_positive", "inconclusive",
                    "PL_positive", "PL_strong", "PL_decisive"]:
            r["dAIC_" + cls] = int((d["dAIC_class"] == cls).sum())
        rows.append(r)
    return rows


def add_vuong_classes(d):
    d = d.copy()
    d["vuong_q_BH"] = bh(d["vuong_p"].values)
    d["vuong_class"] = np.where(d["vuong_q_BH"] < 0.05,
                                np.where(d["vuong_z"] > 0, "PL", "exp"), "indistinguishable")
    return d


def event_types(d):
    out = []
    for eid, g in d.groupby("event_id", sort=False):
        pl_w = int(((g["vuong_class"] == "PL") | (g["dAIC"] > 2)).sum())
        ex_w = int(((g["vuong_class"] == "exp") | (g["dAIC"] < -2)).sum())
        if pl_w >= 2 and ex_w == 0:
            typ = "PL-type"
        elif ex_w >= 2 and pl_w == 0:
            typ = "exp-type"
        else:
            typ = "mixed"
        out.append({"event_id": eid, "n_valid": len(g), "PL_wins": pl_w, "exp_wins": ex_w, "type": typ})
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--v1fits", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if "omori_pipeline" in out.split(os.sep):
        sys.exit("Refusing to write inside omori_pipeline/ (originals are read-only).")
    os.makedirs(out, exist_ok=True)
    cp = os.path.join(out, "inputs_copy")
    os.makedirs(cp, exist_ok=True)
    ts_p = shutil.copy2(a.ts, os.path.join(cp, "vix_term_structure.csv"))
    ev_p = shutil.copy2(a.events, os.path.join(cp, "shock_events.csv"))
    v1_p = shutil.copy2(a.v1fits, os.path.join(cp, "decay_fits_v1.csv")) if a.v1fits else None

    log = []
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": __import__("scipy").__version__, "argv": sys.argv}
    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except Exception:
        env["matplotlib"] = None

    ts = pd.read_csv(ts_p, parse_dates=["Date"], index_col="Date").sort_index()
    ev = pd.read_csv(ev_p, parse_dates=["onset_date", "peak_date"])
    log.append(f"Input events: {len(ev)}; maturities present: {[m for m in MATS if m in ts.columns]}")
    ev = build_events(ev, ts, log)
    pairs, overlap_ids = overlaps(ev, ts)
    log.append(f"S2 overlapping decay-window pairs: {len(pairs)}; events involved: {len(overlap_ids)}")

    fits, robB, oos, curves = [], [], [], {}
    for _, e in ev.iterrows():
        pk_loc = int(ts.index.searchsorted(e["peak_date"]))
        on_loc = int(ts.index.searchsorted(e["onset_date"]))
        for mat in MATS:
            if mat not in ts.columns:
                continue
            seg = ts[mat].iloc[pk_loc: pk_loc + WINDOW]
            mask = seg.notna().values
            y = seg.values[mask].astype(float)
            t = np.arange(1, len(seg) + 1, dtype=float)[mask]
            if len(y) < MIN_POINTS:
                continue
            pre = ts[mat].iloc[max(0, on_loc - BASE_WIN): on_loc].dropna()
            base_m = float(pre.mean()) if len(pre) else np.nan

            row = {"event_id": e["event_id"], "label": e["label"],
                   "peak_date": e["peak_date"].date(), "maturity": mat, "n_points": len(y),
                   "V_peak_window": float(np.max(y)), "V_t1": float(y[0]),
                   "baseline_pre_mat": base_m,
                   "window_argmax": int(np.argmax(y)),
                   "overlap_S2": e["event_id"] in overlap_ids}
            ex, pl, conv, vpk = fit_primary(t, y, base_m)
            row.update({"exp_conv": conv[0], "exp_starts": conv[1],
                        "pl_conv": conv[2], "pl_starts": conv[3]})
            if ex is None or pl is None:
                row["fit_failed"] = True
                fits.append(row)
                continue
            row["fit_failed"] = False
            (eb, eA, el), ecov = ex[0], ex[1]
            (pb, pK, pc, pp), pcov = pl[0], pl[1]
            yh_e = exp_full(t, eb, eA, el)
            yh_p = pl_full(t, pb, pK, pc, pp)
            me = metrics(y, yh_e, K_EXP)
            mp = metrics(y, yh_p, K_PL)
            z, vp = vuong(y, yh_p, yh_e, K_PL, K_EXP)
            row.update({
                "exp_b": eb, "exp_A": eA, "exp_lambda": el,
                "exp_b_se": se(ecov, 0), "exp_A_se": se(ecov, 1), "exp_lambda_se": se(ecov, 2),
                "exp_half_life_days": np.log(2) / el,
                "pl_b": pb, "pl_K": pK, "pl_c": pc, "pl_p": pp,
                "pl_b_se": se(pcov, 0), "pl_K_se": se(pcov, 1), "pl_c_se": se(pcov, 2), "pl_p_se": se(pcov, 3),
                "exp_R2": me["R2"], "exp_AIC": me["AIC"], "exp_BIC": me["BIC"],
                "pl_R2": mp["R2"], "pl_AIC": mp["AIC"], "pl_BIC": mp["BIC"],
                "dAIC": me["AIC"] - mp["AIC"], "dBIC": me["BIC"] - mp["BIC"],
                "vuong_z": z, "vuong_p": vp,
            })
            row["dAIC_class"] = strength(row["dAIC"])
            # ---- frozen filter flags (D-14 / D-21)
            row["W1"] = bool(row["window_argmax"] > LATE_PEAK)
            row["F1"] = near(pp, *P_B)
            row["F2v2"] = near(pc, *C_B)
            row["F3"] = near(el, *L_B)
            row["F5"] = bool((me["R2"] < 0) or (mp["R2"] < 0) or not np.isfinite(me["R2"]) or not np.isfinite(mp["R2"]))
            row["F6"] = bool(any((b <= REL * vpk) or (b >= (1 - REL) * vpk) for b in (eb, pb)))
            row["strict_R2_lt_0.5"] = bool((me["R2"] < 0.5) or (mp["R2"] < 0.5))
            reasons = [k for k in ("W1", "F1", "F2v2", "F3", "F5", "F6") if row[k]]
            row["exclusion_reasons"] = ";".join(reasons)
            row["primary_valid"] = len(reasons) == 0
            row["strict_valid"] = row["primary_valid"] and not row["strict_R2_lt_0.5"]
            fits.append(row)
            curves[(e["event_id"], mat)] = (t, y, yh_e, yh_p)

            # ---- robustness B: fixed per-maturity baseline, no flooring
            rb = {"event_id": e["event_id"], "maturity": mat, "baseline_pre_mat": base_m}
            if not np.isfinite(base_m):
                rb.update({"status": "no-baseline"})
            else:
                yx = y - base_m
                frac = float(np.mean(yx > 0))
                rb["frac_above_baseline"] = frac
                if frac < 0.8:
                    rb["status"] = "baseline-exceeded"
                else:
                    bex, bpl = fit_fixed_base(t, yx)
                    if bex is None or bpl is None:
                        rb["status"] = "fit-failed"
                    else:
                        (A2, l2), (K2, c2, p2) = bex[0], bpl[0]
                        ye2, yp2 = exp_nob(t, A2, l2), pl_nob(t, K2, c2, p2)
                        m2e, m2p = metrics(yx, ye2, K_EXP_B), metrics(yx, yp2, K_PL_B)
                        z2, vp2 = vuong(yx, yp2, ye2, K_PL_B, K_EXP_B)
                        flags = [k for k, v in (("W1", row["W1"]), ("F1", near(p2, *P_B)),
                                                ("F2v2", near(c2, *C_B)), ("F3", near(l2, *L_B)),
                                                ("F5", m2e["R2"] < 0 or m2p["R2"] < 0)) if v]
                        rb.update({"status": "fitted", "exp_A": A2, "exp_lambda": l2,
                                   "pl_K": K2, "pl_c": c2, "pl_p": p2,
                                   "exp_R2": m2e["R2"], "pl_R2": m2p["R2"],
                                   "dAIC": m2e["AIC"] - m2p["AIC"], "dBIC": m2e["BIC"] - m2p["BIC"],
                                   "vuong_z": z2, "vuong_p": vp2,
                                   "exclusion_reasons": ";".join(flags),
                                   "valid": len(flags) == 0})
            robB.append(rb)

            # ---- out-of-sample forecasts (primary design)
            for n_tr, name in OOS_SPLITS:
                tr, te = t <= n_tr, t > n_tr
                if tr.sum() < 10 or te.sum() < 5:
                    continue
                ex2, pl2, _, _ = fit_primary(t[tr], y[tr], base_m)
                if ex2 is None or pl2 is None:
                    oos.append({"event_id": e["event_id"], "maturity": mat, "split": name, "status": "fit-failed"})
                    continue
                fe = exp_full(t[te], *ex2[0])
                fp = pl_full(t[te], *pl2[0])
                oos.append({"event_id": e["event_id"], "maturity": mat, "split": name, "status": "ok",
                            "n_train": int(tr.sum()), "n_test": int(te.sum()),
                            "exp_RMSE": float(np.sqrt(np.mean((y[te] - fe) ** 2))),
                            "pl_RMSE": float(np.sqrt(np.mean((y[te] - fp) ** 2))),
                            "exp_MAE": float(np.mean(np.abs(y[te] - fe))),
                            "pl_MAE": float(np.mean(np.abs(y[te] - fp))),
                            "primary_valid_fullwindow": row["primary_valid"]})
        print(f"done {e['event_id']}", flush=True)

    F = pd.DataFrame(fits)
    F["fit_failed"] = F["fit_failed"].astype(bool)
    F.to_csv(os.path.join(out, "v2_fits.csv"), index=False)
    ok = F[~F["fit_failed"]].copy()
    # Flag columns are object dtype if any fit failed (NaN in failed rows); force real bools.
    for col in BOOL_COLS:
        if col in ok.columns:
            ok[col] = ok[col].fillna(False).astype(bool)
    excl = ok[~ok["primary_valid"]][["event_id", "maturity", "exclusion_reasons", "window_argmax",
                                     "pl_p", "pl_c", "exp_lambda", "exp_b", "pl_b", "exp_R2", "pl_R2"]]
    failed = F[F["fit_failed"]][["event_id", "maturity"]].assign(exclusion_reasons="fit-failed")
    pd.concat([excl, failed], ignore_index=True).to_csv(os.path.join(out, "v2_exclusions.csv"), index=False)

    views = {
        "unfiltered": ok,
        "primary": ok[ok["primary_valid"]],
        "strict": ok[ok["strict_valid"]],
        "primary_nonoverlap": ok[ok["primary_valid"] & ~ok["overlap_S2"]],
    }
    comp, types = [], []
    for vname, d in views.items():
        d = add_vuong_classes(d)
        d.to_csv(os.path.join(out, f"v2_view_{vname}.csv"), index=False)
        comp += summarize(d, vname)
        for r in event_types(d):
            r["view"] = vname
            types.append(r)
    C = pd.DataFrame(comp)
    C.to_csv(os.path.join(out, "v2_comparison_by_view.csv"), index=False)
    T = pd.DataFrame(types)
    T.to_csv(os.path.join(out, "v2_event_types.csv"), index=False)

    O = pd.DataFrame(oos)
    O.to_csv(os.path.join(out, "v2_oos.csv"), index=False)
    if len(O) and "exp_RMSE" in O:
        Ook = O[O["status"] == "ok"].copy()
        Ook["primary_valid_fullwindow"] = Ook["primary_valid_fullwindow"].fillna(False).astype(bool)
        Ook["pl_better_RMSE"] = Ook["pl_RMSE"] < Ook["exp_RMSE"]
        oos_sum = Ook.groupby(["split", "primary_valid_fullwindow"]).agg(
            n=("pl_better_RMSE", "size"), pl_better=("pl_better_RMSE", "sum"),
            med_exp_RMSE=("exp_RMSE", "median"), med_pl_RMSE=("pl_RMSE", "median")).reset_index()
        oos_sum.to_csv(os.path.join(out, "v2_oos_summary.csv"), index=False)
    else:
        oos_sum = pd.DataFrame()

    B = pd.DataFrame(robB)
    B.to_csv(os.path.join(out, "v2_robustnessB_fits.csv"), index=False)
    if len(B) and "valid" in B:
        Bv = B[B["status"].eq("fitted") & B["valid"].fillna(False).astype(bool)]
        robB_sum = {"fitted": int(B["status"].eq("fitted").sum()),
                    "baseline_exceeded": int(B["status"].eq("baseline-exceeded").sum()),
                    "valid": len(Bv), "AIC_PL": int((Bv["dAIC"] > 0).sum()),
                    "AIC_exp": int((Bv["dAIC"] < 0).sum())}
    else:
        robB_sum = {}

    pd.DataFrame(pairs, columns=["event_a", "event_b", "peak_gap_trading_days"]).to_csv(
        os.path.join(out, "v2_overlaps_S2.csv"), index=False)

    # ---- v1 comparison
    v1cmp = pd.DataFrame()
    if v1_p:
        v1 = pd.read_csv(v1_p)
        v1["label"] = v1["label"].fillna("").astype(str)
        rows = []
        for key in V1_COMPARE:
            v1k = v1[v1["label"].str.contains(key, regex=False)]
            v2k = ok[ok["label"].astype(str).str.contains(key, regex=False)]
            for mat in MATS:
                a1 = v1k[v1k["maturity"] == mat]
                a2 = v2k[v2k["maturity"] == mat]
                if len(a1) == 0 and len(a2) == 0:
                    continue
                rows.append({"event": key, "maturity": mat,
                             "v1_rows": ";".join(a1["label"].astype(str) + "@" + a1["event_date"].astype(str)),
                             "v1_p": ";".join(a1["pl_p"].astype(str)) if len(a1) else "",
                             "v1_c": ";".join(a1["pl_c"].astype(str)) if len(a1) else "",
                             "v1_winner_AIC": ";".join(a1["winner_AIC"].astype(str)) if len(a1) else "",
                             "v2_p": a2["pl_p"].iloc[0] if len(a2) else np.nan,
                             "v2_c": a2["pl_c"].iloc[0] if len(a2) else np.nan,
                             "v2_dAIC": a2["dAIC"].iloc[0] if len(a2) else np.nan,
                             "v2_primary_valid": bool(a2["primary_valid"].iloc[0]) if len(a2) else None,
                             "v2_exclusion": a2["exclusion_reasons"].iloc[0] if len(a2) else ""})
        v1cmp = pd.DataFrame(rows)
        v1cmp.to_csv(os.path.join(out, "v2_vs_v1.csv"), index=False)

    # ---- figures
    if not a.no_figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fd = os.path.join(out, "figures")
            os.makedirs(fd, exist_ok=True)
            for eid in ev["event_id"]:
                fig, axs = plt.subplots(2, 2, figsize=(10, 7))
                for ax, mat in zip(axs.ravel(), MATS):
                    if (eid, mat) in curves:
                        t, y, ye, yp = curves[(eid, mat)]
                        ax.plot(t, y, ".", ms=3, color="k", label="data")
                        ax.plot(t, ye, "-", label="exp")
                        ax.plot(t, yp, "--", label="power law")
                        rr = ok[(ok["event_id"] == eid) & (ok["maturity"] == mat)].iloc[0]
                        tag = "valid" if rr["primary_valid"] else "excl: " + str(rr["exclusion_reasons"])
                        ax.set_title(f"{mat}  dAIC={rr['dAIC']:.1f}  ({tag})", fontsize=8)
                    else:
                        ax.set_title(f"{mat}: no fit", fontsize=8)
                    ax.tick_params(labelsize=7)
                axs[0, 0].legend(fontsize=7)
                fig.suptitle(eid)
                fig.tight_layout()
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in eid)
                fig.savefig(os.path.join(fd, f"event_{safe}.png"), dpi=110)
                plt.close(fig)
            pv = views["primary"]
            fig, ax = plt.subplots(figsize=(7, 4))
            for mat in MATS:
                vals = pv[pv["maturity"] == mat]["pl_p"].dropna()
                if len(vals):
                    ax.hist(vals, bins=15, alpha=0.5, label=f"{mat} (n={len(vals)})")
            ax.set_xlabel("power-law exponent p (primary view)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(fd, "hist_p_by_maturity.png"), dpi=110)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * ok["event_id"].nunique())))
            for i, mat in enumerate(MATS):
                d = ok[ok["maturity"] == mat]
                ax.scatter(d["dAIC"].clip(-50, 50), d["event_id"], s=12, label=mat)
            ax.axvline(0, color="k", lw=0.5)
            ax.axvline(2, color="grey", lw=0.5, ls=":")
            ax.axvline(-2, color="grey", lw=0.5, ls=":")
            ax.set_xlabel("dAIC = AIC_exp - AIC_PL (>0 favours power law; clipped at +/-50)")
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(fd, "dAIC_by_event.png"), dpi=110)
            plt.close(fig)
        except Exception as ex_:
            log.append(f"Figures skipped: {ex_!r}")

    # ---- auto summary (to be copied into verification/v2_results_summary.md)
    with open(os.path.join(out, "v2_results_summary_AUTO.md"), "w") as fh:
        fh.write("# v2 results (auto-generated by fit_v2.py) — VERIFIED-RUN only after a clean run; Vera sign-off pending\n\n")
        fh.write("## Environment\n```\n" + json.dumps(env, indent=1) + "\n```\n\n## Run log\n")
        for line in log:
            fh.write(f"- {line}\n")
        fh.write(f"\n## Fits\n- total rows {len(F)}, fit failures {int(F['fit_failed'].sum())}, "
                 f"primary valid {int(ok['primary_valid'].sum())}, strict valid {int(ok['strict_valid'].sum())}\n")
        fh.write("- exclusion counts by rule: " + json.dumps(
            {k: int(ok[k].sum()) for k in ("W1", "F1", "F2v2", "F3", "F5", "F6")}) + "\n")
        fh.write("\n## Comparison by view\n```\n" + C.to_string(index=False) + "\n```\n")
        fh.write("\n## Event types\n```\n" + T.to_string(index=False) + "\n```\n")
        if len(oos_sum):
            fh.write("\n## Out-of-sample\n```\n" + oos_sum.to_string(index=False) + "\n```\n")
        fh.write("\n## Robustness B (fixed per-maturity baseline)\n" + json.dumps(robB_sum) + "\n")
        fh.write(f"\n## S2 overlaps\n{len(pairs)} pairs; see v2_overlaps_S2.csv\n")
        if len(v1cmp):
            fh.write("\n## v2 vs v1\n```\n" + v1cmp.to_string(index=False) + "\n```\n")
        fh.write("\nRule reminder: a count claim needs >= 20 primary-valid fits (count_claim_allowed).\n")
    print("\n".join(log))
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
