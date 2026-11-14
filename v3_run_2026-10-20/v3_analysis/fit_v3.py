#!/usr/bin/env python3
"""
v3 fits: power law vs exponential decay of post-shock implied vol, windows 61/120/250.
Implements verification/filter_rules_v3.md (frozen; body SHA-256 checked at start).

Usage:  python3 -I fit_v3.py --ts ... --events ... --rules ... --out OUT [--gate-only] [--jobs N]
Reads only copies (placed in OUT/inputs_copy). Refuses to write inside omori_pipeline/.
"""
import argparse, hashlib, json, os, platform, shutil, sys, warnings
import pickle, subprocess

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

EXPECTED_HASH = "010df754be8b239da2bb75e7df22db90933d03b858b9a88aad9f4ac190314415"
MATS = ["VIX9D", "VIX30D", "VIX3M", "VIX6M"]
HS = [61, 120, 250]
SPECS = ["free", "fixed"]
MIN_N = 15
BASE_WIN = 20
NSTART = 24
MAX_NFEV = 1500   # least_squares (trf); non-converged starts are kept (best SSE wins)
BOOT_REPS = 999
SEED0 = 20261020
C_B, P_B, TAU_B = (1e-3, 500.0), (0.01, 10.0), (0.2, 1000.0)
REL = 0.01
LATE_PEAK = 10
MIN_RESOLVED = 5


def pl_free(t, B, K, c, p): return B + K * np.power(t + c, -p)
def exp_free(t, B, A, tau): return B + A * np.exp(-t / tau)
def pl_fix(t, K, c, p): return K * np.power(t + c, -p)
def exp_fix(t, A, tau): return A * np.exp(-t / tau)


def check_hash(path):
    b = open(path, "rb").read()
    m = b"\n<!-- END FROZEN BODY -->\n"
    h = hashlib.sha256(b[: b.index(m) + 1]).hexdigest()
    if h != EXPECTED_HASH:
        sys.exit(f"RULES HASH MISMATCH: {h} != {EXPECTED_HASH}; aborting")
    return h


def near(x, lo, hi):
    return bool(x <= lo * (1 + REL) or x >= hi * (1 - REL))


# ------------------------------------------------------------------ fitting
def make_starts(kind, rng, y, base_pre):
    """24 starts. Starts 0-2 anchored (free: B in {pre-shock mean, ymin, 0.5 ymin}); rest random."""
    ymax, ymin = float(y.max()), float(y.min())
    out = []
    for i in range(NSTART):
        if kind in ("pl_free", "exp_free"):
            anchors = [base_pre if np.isfinite(base_pre) else ymin, ymin, 0.5 * ymin]
            B = anchors[i] if i < 3 else rng.uniform(0, ymax)
            B = float(np.clip(B, 0, ymax))
        else:
            B = 0.0
        amp0 = max(y[0] - B, 1e-3)
        c = float(np.exp(rng.uniform(np.log(0.05), np.log(100))))
        p = float(np.exp(rng.uniform(np.log(0.1), np.log(5))))
        tau = float(np.exp(rng.uniform(np.log(1), np.log(300))))
        if kind == "pl_free": out.append((B, amp0 * (1 + c) ** p, c, p))
        elif kind == "exp_free": out.append((B, amp0, tau))
        elif kind == "pl_fix": out.append((max(y[0], 1e-3) * (1 + c) ** p, c, p))
        else: out.append((max(y[0], 1e-3), tau))
    return out


def bounds(kind, ymax):
    if kind == "pl_free": return [0, 1e-8, C_B[0], P_B[0]], [ymax, np.inf, C_B[1], P_B[1]]
    if kind == "exp_free": return [0, 1e-8, TAU_B[0]], [ymax, np.inf, TAU_B[1]]
    if kind == "pl_fix": return [1e-8, C_B[0], P_B[0]], [np.inf, C_B[1], P_B[1]]
    return [1e-8, TAU_B[0]], [np.inf, TAU_B[1]]


FUN = {"pl_free": pl_free, "exp_free": exp_free, "pl_fix": pl_fix, "exp_fix": exp_fix}


def multistart(kind, t, y, rng, base_pre):
    f = FUN[kind]
    lo, hi = bounds(kind, float(y.max()))
    lo, hi = np.array(lo, float), np.array(hi, float)
    span = np.where(np.isfinite(hi - lo), (hi - lo) * 1e-6, 1e-6)
    sses, best, log = [], None, []
    for si, p0 in enumerate(make_starts(kind, rng, y, base_pre)):
        p0 = np.minimum(np.maximum(np.asarray(p0, float), lo + span), np.where(np.isfinite(hi), hi - span, np.inf))
        try:
            popt = least_squares(lambda q: f(t, *q) - y, p0, bounds=(lo, hi), method="trf", max_nfev=MAX_NFEV).x
            yh = f(t, *popt)
            if not np.all(np.isfinite(yh)): raise ValueError
            sse = float(np.sum((y - yh) ** 2))
        except Exception:
            sses.append(np.nan); log.append((si, "fail", np.nan)); continue
        sses.append(sse); log.append((si, "ok", sse))
        if best is None or sse < best[1]: best = (popt, sse)
    if best is None: return None, 0, log
    reach = int(sum(1 for s in sses if np.isfinite(s) and s <= best[1] * (1 + 1e-6) + 1e-12))
    return (best[0], best[1]), reach, log


def pll(y, yh):
    e = y - yh
    s2 = max(float(np.mean(e ** 2)), 1e-300)
    return -0.5 * np.log(2 * np.pi * s2) - e ** 2 / (2 * s2)


def ll_tot(y, yh):
    n = len(y); sse = max(float(np.sum((y - yh) ** 2)), 1e-300)
    return -n / 2 * (np.log(2 * np.pi * sse / n) + 1)


def r2(y, yh):
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1 - float(np.sum((y - yh) ** 2)) / sst if sst > 0 else np.nan


def boot_dAIC(d, dk2, L, rng):
    """Moving-block bootstrap of dAIC* = 2 sum d* - 2(kPL-kEXP); recentred percentile CI."""
    n = len(d); L = min(L, n)
    nb = -(-n // L)
    st = rng.integers(0, n - L + 1, size=(BOOT_REPS, nb))
    idx = (st[:, :, None] + np.arange(L)[None, None, :]).reshape(BOOT_REPS, -1)[:, :n]
    stat = 2 * d[idx].sum(axis=1) - dk2
    point = 2 * d.sum() - dk2
    rc = point + (stat - stat.mean())
    lo, hi = np.percentile(rc, [2.5, 97.5])
    cls = "PL" if lo > 0 else ("exp" if hi < 0 else "indistinguishable")
    return float(point), float(lo), float(hi), cls


def hac_vuong(d, dk, n):
    m = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    dc = d - d.mean()
    lrv = float(np.sum(dc * dc) / n)
    for j in range(1, m + 1):
        w = 1 - j / (m + 1)
        lrv += 2 * w * float(np.sum(dc[j:] * dc[:-j]) / n)
    if not np.isfinite(lrv) or lrv <= 0: return np.nan, np.nan, m
    lr = d.sum() - dk / 2.0 * np.log(n)
    z = lr / np.sqrt(n * lrv)
    return float(z), float(2 * stats.norm.sf(abs(z))), m


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    pv = p[ok]; m = len(pv)
    if m == 0: return q
    o = np.argsort(pv); r = pv[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(m); qq[o] = np.minimum(r, 1); q[ok] = qq
    return q


# ------------------------------------------------------------------ job

def oos_rows(key, t, yy, n, kind_pl, kind_ex, rs_oos, base_pre):
    fp, fe = FUN[kind_pl], FUN[kind_ex]
    oos = []
    for name, ntr in (("first33", int(round(0.33 * n))), ("first50", int(round(0.5 * n))), ("lastout20", n - 20)):
        if ntr < 8 or n - ntr < 5: continue
        tr = np.arange(n) < ntr; te = ~tr
        pl2, _, _ = multistart(kind_pl, t[tr], yy[tr], rs_oos, base_pre)
        ex2, _, _ = multistart(kind_ex, t[tr], yy[tr], rs_oos, base_pre)
        if pl2 is None or ex2 is None:
            oos.append(dict(key, split=name, status="fit-failed")); continue
        a, b = fp(t[te], *pl2[0]), fe(t[te], *ex2[0])
        oos.append(dict(key, split=name, status="ok", n_train=int(tr.sum()), n_test=int(te.sum()),
                        pl_RMSE=float(np.sqrt(np.mean((yy[te] - a) ** 2))), exp_RMSE=float(np.sqrt(np.mean((yy[te] - b) ** 2))),
                        pl_MAE=float(np.mean(np.abs(yy[te] - a))), exp_MAE=float(np.mean(np.abs(yy[te] - b)))))
    return oos

G = {}


def run_job(job):
    ei, mi, H, si, trunc_rule, ev_id, label, peak_loc, end_loc, trunc_flag, base_pre = job
    ts_vals = G["vals"][mi]
    seg = ts_vals[peak_loc: end_loc]
    tpos = np.arange(len(seg), dtype=float)
    mask = ~np.isnan(seg)
    y, t = seg[mask].astype(float), tpos[mask]
    n = len(y)
    key = dict(event_id=ev_id, label=label, event_index=ei, maturity=MATS[mi], H=H, spec=SPECS[si], window_rule=trunc_rule)
    if n < MIN_N: return None
    seed = np.random.SeedSequence([SEED0, ei, mi, H, si, 0])
    rs_pl, rs_ex, rs_oos = [np.random.default_rng(s) for s in seed.spawn(3)]
    fixed = si == 1
    yy = y - base_pre if fixed else y
    if fixed and not np.isfinite(base_pre): return None
    kp, ke = (4, 3) if fixed else (5, 4)
    kind_pl, kind_ex = ("pl_fix", "exp_fix") if fixed else ("pl_free", "exp_free")
    if os.environ.get("V3_OOS_ONLY"):
        return dict(fit=None, starts=[], boot=[], vuong=None, oos=oos_rows(key, t, yy, n, kind_pl, kind_ex, rs_oos, base_pre))
    pl, reach_pl, log_pl = multistart(kind_pl, t, yy, rs_pl, base_pre)
    ex, reach_ex, log_ex = multistart(kind_ex, t, yy, rs_ex, base_pre)
    startlog = [dict(key, model="PL", start=s, status=a, sse=b) for s, a, b in log_pl] + \
               [dict(key, model="EXP", start=s, status=a, sse=b) for s, a, b in log_ex]
    row = dict(key, n=n, trunc=trunc_flag, full=bool(n >= 0.8 * H), base_pre=base_pre,
               V_peak=float(y.max()), argmax=int(np.argmax(y)), reach_pl=reach_pl, reach_ex=reach_ex,
               first_date=str(G["dates"][peak_loc]), last_date=str(G["dates"][min(end_loc, len(G["dates"])) - 1]))
    if pl is None or ex is None:
        row["fit_failed"] = True
        return dict(fit=row, starts=startlog, boot=[], vuong=None, oos=[])
    row["fit_failed"] = False
    (pp, _), (ep, _) = pl, ex
    fp, fe = FUN[kind_pl], FUN[kind_ex]
    yh_p, yh_e = fp(t, *pp), fe(t, *ep)
    ll_p, ll_e = ll_tot(yy, yh_p), ll_tot(yy, yh_e)
    aic_p, aic_e = 2 * kp - 2 * ll_p, 2 * ke - 2 * ll_e
    bic_p, bic_e = kp * np.log(n) - 2 * ll_p, ke * np.log(n) - 2 * ll_e
    row.update(pl_R2=r2(yy, yh_p), exp_R2=r2(yy, yh_e), pl_AIC=aic_p, exp_AIC=aic_e, pl_BIC=bic_p, exp_BIC=bic_e,
               dAIC=aic_e - aic_p, dBIC=bic_e - bic_p, dAIC_eqk=2 * (ll_p - ll_e),
               pl_SSE=pl[1], exp_SSE=ex[1])
    if fixed:
        row.update(pl_K=pp[0], pl_c=pp[1], pl_p=pp[2], exp_A=ep[0], exp_tau=ep[1], pl_B=base_pre, exp_B=base_pre)
        row["frac_below_B"] = float(np.mean(y < base_pre)); row["B_minus"] = bool(row["frac_below_B"] > 0.2)
        c_b, p_b, tau_b, B_b = near(pp[1], *C_B), near(pp[2], *P_B), near(ep[1], *TAU_B), False
        resolved = int(np.sum(y > base_pre))
        row.update(pl_Bdiff=np.nan, exp_Bdiff=np.nan)
    else:
        row.update(pl_B=pp[0], pl_K=pp[1], pl_c=pp[2], pl_p=pp[3], exp_B=ep[0], exp_A=ep[1], exp_tau=ep[2])
        ymax = float(y.max())
        c_b, p_b, tau_b = near(pp[2], *C_B), near(pp[3], *P_B), near(ep[2], *TAU_B)
        B_b = any(b <= REL * ymax or b >= (1 - REL) * ymax for b in (pp[0], ep[0]))
        resolved = int(np.sum(y > base_pre)) if np.isfinite(base_pre) else int(np.sum(y > y.min()))
        row.update(pl_Bdiff=pp[0] - base_pre, exp_Bdiff=ep[0] - base_pre, frac_below_B=np.nan, B_minus=False)
    f = dict(W1=row["argmax"] > LATE_PEAK, W2=resolved < MIN_RESOLVED, F1=p_b, F3=tau_b,
             F5=bool(row["pl_R2"] < 0 or row["exp_R2"] < 0), F6=B_b, notfull=not row["full"])
    unstable = min(reach_pl, reach_ex) < 3
    row["resolved"] = resolved
    row["c_on_bound"] = c_b
    row["unstable"] = unstable
    row["excl_reasons"] = ";".join(k for k, v in f.items() if v)
    row["primary"] = not any(f.values())
    row["strict"] = bool(row["primary"] and not c_b and not (row["pl_R2"] < 0.5 or row["exp_R2"] < 0.5) and not unstable)
    # curve separation and Vuong (HAC)
    d = pll(yy, yh_p) - pll(yy, yh_e)
    sig = float(np.sqrt(np.mean((yy - (yh_p if pl[1] <= ex[1] else yh_e)) ** 2)))
    sep = float(np.max(np.abs(yh_p - yh_e)))
    testable = bool(sig > 0 and sep >= 0.5 * sig)
    row["sep_over_sigma"] = sep / sig if sig > 0 else np.nan
    row["testable"] = testable
    lag1 = np.corrcoef((yy - yh_p)[1:], (yy - yh_p)[:-1])[0, 1] if n > 3 else np.nan
    row["resid_lag1_PL"] = float(lag1)
    vz, vp, m = (np.nan, np.nan, np.nan)
    if testable: vz, vp, m = hac_vuong(d, kp - ke, n)
    # bootstrap
    seedb = np.random.SeedSequence([SEED0, ei, mi, H, si, 1])
    rb = np.random.default_rng(seedb)
    boots = []
    Ls = [5, 10] + ([20] if H == 250 else [])
    dk2 = 2 * (kp - ke)
    for L in Ls:
        pt, lo, hi, cls = boot_dAIC(d, dk2, L, rb)
        boots.append(dict(key, L=L, dAIC=pt, ci_lo=lo, ci_hi=hi, cls=cls))
    row["boot_cls_L5"] = boots[0]["cls"]; row["boot_cls_L10"] = boots[1]["cls"]
    row["block_sensitive"] = boots[0]["cls"] != boots[1]["cls"]
    row["ci_lo_L5"], row["ci_hi_L5"] = boots[0]["ci_lo"], boots[0]["ci_hi"]
    row["vuong_z"], row["vuong_p"], row["hac_bw"] = vz, vp, m
    oos = [] if os.environ.get("V3_SKIP_OOS") else oos_rows(key, t, yy, n, kind_pl, kind_ex, rs_oos, base_pre)
    for o in oos: o["primary"] = row["primary"]
    return dict(fit=row, starts=startlog, boot=boots, vuong=None, oos=oos)


# ------------------------------------------------------------------ events
def build_events(ev, ts):
    ev = ev.copy(); ev["label"] = ev["label"].fillna("").astype(str)
    leh, gfc = ev[ev.label == "Lehman"], ev[ev.label == "GFC Peak"]
    assert len(leh) == 1 and len(gfc) == 1
    row = dict(onset_date=leh.onset_date.iloc[0], peak_date=gfc.peak_date.iloc[0], peak_level=gfc.peak_level.iloc[0],
               baseline=leh.baseline.iloc[0], label="GFC (merged Lehman + GFC Peak)")
    ev = pd.concat([ev.drop(index=list(leh.index) + list(gfc.index)), pd.DataFrame([row])], ignore_index=True)
    ev = ev.sort_values("onset_date").reset_index(drop=True)
    assert len(ev) == 33, len(ev)
    ev["event_id"] = [d.strftime("%Y-%m-%d") + (" " + l if l else "") for d, l in zip(ev.peak_date, ev.label)]
    return ev


def make_jobs(ev, ts, windows):
    onset_locs = [int(ts.index.searchsorted(d)) for d in ev.onset_date]
    jobs = []
    for ei, e in ev.iterrows():
        pk = int(ts.index.searchsorted(e.peak_date)); on = int(ts.index.searchsorted(e.onset_date))
        later = [o for o in onset_locs if o > pk]
        for mi, mat in enumerate(MATS):
            pre = ts[mat].iloc[max(0, on - BASE_WIN): on].dropna()
            base = float(pre.mean()) if len(pre) else np.nan
            for H, rule in windows:
                end = pk + H; flag = ""
                if rule == "trunc":
                    if later and min(later) < end: end = min(later); flag = "T"
                if end > len(ts): end = len(ts); flag = "E" if flag == "" else flag
                for si in (0, 1):
                    jobs.append((ei, mi, H, si, rule, e.event_id, e.label, pk, end, flag, base))
    return jobs


def init(vals, dates):
    G["vals"], G["dates"] = vals, dates


def counts(F, cond_col=None):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True); ap.add_argument("--events", required=True)
    ap.add_argument("--rules", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gate-only", action="store_true"); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--worker", type=int, default=None)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if "omori_pipeline" in out.split(os.sep): sys.exit("Refusing to write inside omori_pipeline/")
    h = check_hash(a.rules)
    os.makedirs(os.path.join(out, "inputs_copy"), exist_ok=True)
    ts_p, ev_p = (os.path.join(out, "inputs_copy", n) for n in ("vix_term_structure.csv", "shock_events.csv"))
    if a.worker is None:  # workers read the copies made by the parent
        shutil.copy2(a.ts, ts_p); shutil.copy2(a.events, ev_p)
    env = dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
               scipy=__import__("scipy").__version__, platform=platform.platform(), rules_hash=h)
    if a.worker is None: json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=1)
    ts = pd.read_csv(ts_p, parse_dates=["Date"], index_col="Date").sort_index()
    ev = build_events(pd.read_csv(ev_p, parse_dates=["onset_date", "peak_date"]), ts)
    windows = [(61, "v2compat")] if a.gate_only else [(61, "v2compat")] + [(H, "trunc") for H in HS]
    jobs = make_jobs(ev, ts, windows)
    print(f"events={len(ev)} jobs={len(jobs)} env={env}", flush=True)
    vals = [ts[m].values.astype(float) for m in MATS]
    init(vals, [str(d.date()) for d in ts.index])
    shard_dir = os.path.join(out, "shards"); os.makedirs(shard_dir, exist_ok=True)
    def shard(i): return os.path.join(shard_dir, f"j{i}.pkl")
    if a.worker is not None:  # worker: serial run of every jobs-th job; one pickle per job (resumable)
        for i in range(a.worker, len(jobs), a.jobs):
            if os.path.exists(shard(i)): continue
            pickle.dump(run_job(jobs[i]), open(shard(i) + ".tmp", "wb")); os.replace(shard(i) + ".tmp", shard(i))
        return
    # (multiprocessing.Pool is blocked by the sandbox, so shards run as separate processes)
    procs = []
    for k in range(a.jobs):
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "--ts", a.ts, "--events", a.events, "--rules", a.rules,
               "--out", out, "--jobs", str(a.jobs), "--worker", str(k)] + (["--gate-only"] if a.gate_only else [])
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    for pr in procs:
        if pr.wait() != 0: sys.exit("worker failed")
    res = [r for r in (pickle.load(open(shard(i), "rb")) for i in range(len(jobs))) if r is not None]
    if os.environ.get("V3_OOS_ONLY"):
        O = pd.DataFrame([o for r in res for o in r["oos"]])
        O.to_csv(os.path.join(out, "v3_oos.csv"), index=False); shutil.rmtree(shard_dir); print("oos rows", len(O)); return
    F = pd.DataFrame([r["fit"] for r in res])
    S = pd.DataFrame([s for r in res for s in r["starts"]])
    Bt = pd.DataFrame([b for r in res for b in r["boot"]])
    O = pd.DataFrame([o for r in res for o in r["oos"]])
    ok = F[~F.fit_failed.astype(bool)].copy()
    for c in ("full", "primary", "strict", "testable", "block_sensitive", "c_on_bound", "unstable", "B_minus"):
        ok[c] = ok[c].astype(bool)
    # D flag: window identical to a shorter horizon (trunc windows)
    ok["dup"] = False
    tr = ok[ok.window_rule == "trunc"]
    for (eid, mat, sp), g in tr.groupby(["event_id", "maturity", "spec"]):
        g = g.sort_values("H"); seen = {}
        for ix, r in g.iterrows():
            k = (r.n, r.first_date, r.last_date)
            if k in seen: ok.loc[ix, "dup"] = True
            seen[k] = ix
    # BH-adjusted HAC Vuong within window x spec x view
    for v in ("unfiltered", "primary", "strict"):
        ok[f"vq_{v}"] = np.nan
        sel = ok.primary if v == "primary" else (ok.strict if v == "strict" else pd.Series(True, index=ok.index))
        for _, g in ok[sel].groupby(["window_rule", "H", "spec"]):
            ok.loc[g.index, f"vq_{v}"] = bh(g.vuong_p.values)
    F.to_csv(os.path.join(out, "v3_fits_all.csv"), index=False)
    ok.to_csv(os.path.join(out, "v3_fits.csv"), index=False)
    S.to_csv(os.path.join(out, "v3_starts_log.csv"), index=False)
    Bt.to_csv(os.path.join(out, "v3_bootstrap.csv"), index=False)
    O.to_csv(os.path.join(out, "v3_oos.csv"), index=False)
    ok[["event_id", "maturity", "H", "spec", "window_rule", "n", "trunc", "excl_reasons", "pl_p", "pl_c", "exp_tau", "pl_B", "exp_B"]][
        ~ok.primary].to_csv(os.path.join(out, "v3_exclusions.csv"), index=False)
    ok[ok.testable][["event_id", "maturity", "H", "spec", "window_rule", "n", "sep_over_sigma", "vuong_z", "vuong_p", "hac_bw",
                     "vq_unfiltered", "vq_primary", "vq_strict"]].to_csv(os.path.join(out, "v3_vuong.csv"), index=False)
    # per-event labels
    rows = []
    for view in ("unfiltered", "primary", "strict"):
        d = ok if view == "unfiltered" else ok[ok[view]]
        for (eid, H, sp, wr), g in d.groupby(["event_id", "H", "spec", "window_rule"]):
            npl, nex = int((g.boot_cls_L5 == "PL").sum()), int((g.boot_cls_L5 == "exp").sum())
            typ = "PL-type" if npl >= 2 and nex == 0 else ("exp-type" if nex >= 2 and npl == 0 else "mixed")
            rows.append(dict(view=view, event_id=eid, H=H, spec=sp, window_rule=wr, n_mat=len(g), PL=npl, EXP=nex, type=typ))
    pd.DataFrame(rows).to_csv(os.path.join(out, "v3_event_types.csv"), index=False)
    # headline counts
    cr = []
    for (wr, H, sp), g in ok.groupby(["window_rule", "H", "spec"]):
        for view in ("unfiltered", "primary", "strict"):
            d = g if view == "unfiltered" else g[g[view]]
            n = len(d)
            pl_a, ex_a = int((d.dAIC > 0).sum()), int((d.dAIC < 0).sum())
            bp, be, bi = [int((d.boot_cls_L5 == c).sum()) for c in ("PL", "exp", "indistinguishable")]
            sign = stats.binomtest(bp, bp + be, 0.5).pvalue if bp + be > 0 else np.nan
            cr.append(dict(window_rule=wr, H=H, spec=sp, view=view, n=n, AIC_PL=pl_a, AIC_EXP=ex_a,
                           boot_PL=bp, boot_EXP=be, boot_indist=bi, boot_sign_p=sign,
                           BIC_PL=int((d.dBIC > 0).sum()), BIC_EXP=int((d.dBIC < 0).sum()),
                           block_sensitive=int(d.block_sensitive.sum()), testable=int(d.testable.sum()),
                           hac_sig=int(((d[f"vq_{view}"] < 0.05) & d.testable).sum()),
                           trunc_T=int((d.trunc == "T").sum()), trunc_E=int((d.trunc == "E").sum()), dup=int(d.dup.sum()),
                           count_claim_allowed=bool(n >= 20 and view == "primary")))
    C = pd.DataFrame(cr); C.to_csv(os.path.join(out, "v3_counts.csv"), index=False)
    print(C.to_string(index=False))
    shutil.rmtree(shard_dir)
    print(f"fit_failed={int(F.fit_failed.sum())} rows={len(F)} outputs in {out}")


if __name__ == "__main__":
    main()
