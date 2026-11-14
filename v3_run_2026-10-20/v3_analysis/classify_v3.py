#!/usr/bin/env python3
"""Stage 2: per-event x maturity classification (power_law / exponential / undetermined).

Reads only v3 outputs (v3_fits.csv, v3_bootstrap.csv, v3_exclusions.csv) plus the
event and term-structure inputs (read-only, for two descriptive POST-HOC flags).
Verifies the frozen rules hash first. Writes classification_v3.{md,csv} and
classification_v3_exclusions.csv into --outdir. Fits nothing; changes no frozen rule.

  python3 -I classify_v3.py --v3out v3_out --ts <vix csv> --events <events csv> \
      --rules verification/filter_rules_v3.md --outdir verification
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

FROZEN = "010df754be8b239da2bb75e7df22db90933d03b858b9a88aad9f4ac190314415"
MATS = ["VIX9D", "VIX30D", "VIX3M", "VIX6M"]
PRIMARY = ("free", 250)
COREPORT = [("fixed", 250), ("free", 120), ("fixed", 120), ("free", 61), ("fixed", 61)]
CLS = {"PL": "power_law", "exp": "exponential"}


def rules_hash(path):
    b = open(path, "rb").read()
    m = b"\n<!-- END FROZEN BODY -->\n"
    return hashlib.sha256(b[: b.index(m) + 1]).hexdigest()


def classify(row_cls):
    """row_cls: list of bootstrap classes at each block length (PL / exp / indistinguishable)."""
    s = set(row_cls)
    if len(s) == 1:
        c = row_cls[0]
        if c in CLS:
            return CLS[c], "decisive_stable"
        return "undetermined", "indistinguishable"
    if s <= {"PL", "exp", "indistinguishable"} and ("PL" in s and "exp" in s):
        return "undetermined", "block_unstable_conflict"
    return "undetermined", "block_unstable"


def holm_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    holm = np.zeros(n, bool)
    for r, i in enumerate(o):
        if p[i] <= 0.05 / (n - r):
            holm[i] = True
        else:
            break
    bh = np.zeros(n, bool)
    k = [r for r, i in enumerate(o) if p[i] <= 0.05 * (r + 1) / n]
    if k:
        bh[o[: max(k) + 1]] = True
    return holm, bh


def load_inputs(a):
    f = pd.read_csv(os.path.join(a.v3out, "v3_fits.csv"))
    b = pd.read_csv(os.path.join(a.v3out, "v3_bootstrap.csv"))
    f = f[f.window_rule != "v2compat"].copy()  # 61 = truncated v3 rule; 61u (gate) is not classified
    b = b[b.window_rule != "v2compat"]
    f["ev"] = f.event_id.str[:10]
    return f, b


def bump_and_overlap(f, a):
    """POST-HOC descriptive flags (V3-3c, V3-4). Not filters."""
    ts = pd.read_csv(a.ts, parse_dates=["Date"]).set_index("Date")
    cal = ts["VIX30D"].dropna().index
    ev = pd.read_csv(a.events, parse_dates=["onset_date", "peak_date"])
    ev = ev[ev.peak_date != pd.Timestamp("2008-09-29")].copy()  # Lehman merged into GFC (rules §1)
    ev.loc[ev.peak_date == pd.Timestamp("2008-10-27"), "onset_date"] = pd.Timestamp("2008-09-17")
    ev = ev.sort_values("onset_date").reset_index(drop=True)
    pos = lambda d: int(cal.searchsorted(d))
    ov = {}
    for i, r in ev.iterrows():
        if i == 0:
            ov[r.peak_date.strftime("%Y-%m-%d")] = (False, "")
            continue
        p = ev.iloc[i - 1]
        prev_end = pos(p.peak_date) + 60  # previous event's 61-day window ends here (index)
        pre_start = pos(r.onset_date) - 20
        ov[r.peak_date.strftime("%Y-%m-%d")] = (prev_end >= pre_start, p.peak_date.strftime("%Y-%m-%d"))
    bump = []
    for _, r in f.iterrows():
        s = ts[r.maturity].loc[pd.Timestamp(r.first_date): pd.Timestamp(r.last_date)].dropna().values
        ex0 = r.V_peak - r.base_pre
        flag = False
        if len(s) > 10 and ex0 > 0:
            trough = np.minimum.accumulate(s)
            flag = bool(np.any((s - trough)[10:] >= 0.5 * ex0))
        bump.append(flag)
    f["later_bump_posthoc"] = bump
    f["prewin_overlap_prev61_posthoc"] = f.ev.map(lambda d: ov.get(d, (False, ""))[0])
    f["prewin_overlap_with"] = f.ev.map(lambda d: ov.get(d, (False, ""))[1])
    return f


def attach(f, b):
    """Add per-block-length classes, rule-based class, F1-retained and approx bootstrap p."""
    out = []
    for (spec, H), g in f.groupby(["spec", "H"]):
        g = g.copy()
        bb = b[(b.spec == spec) & (b.H == H)]
        key = ["event_id", "maturity"]
        piv = bb.pivot_table(index=key, columns="L", values="cls", aggfunc="first")
        piv.columns = [f"L{c}" for c in piv.columns]
        ci = bb[bb.L == 5].set_index(key)[["ci_lo", "ci_hi"]]
        g = g.join(piv, on=key).join(ci, on=key)
        Ls = [c for c in piv.columns]
        res = [classify([r[c] for c in Ls]) for _, r in g.iterrows()]
        g["cls_rule"] = [x[0] for x in res]
        g["cls_reason"] = [x[1] for x in res]
        se = (g.ci_hi - g.ci_lo) / (2 * 1.96)
        z = g.dAIC / se
        g["boot_p_approx_posthoc"] = 2 * stats.norm.sf(np.abs(z))
        g["holm_sig_posthoc"], g["bh_sig_posthoc"] = holm_bh(g.boot_p_approx_posthoc.fillna(1).values)
        g["penalty_only"] = (g.cls_rule == "exponential") & (~g.testable.astype(bool))
        g["cls_posthoc"] = np.where(g.penalty_only, "undetermined", g.cls_rule)
        g["primary_valid"] = g.excl_reasons.isna()
        rs = g.excl_reasons.fillna("").str.split(";")
        g["f1_retained_valid"] = rs.map(lambda x: set(x) <= {"", "F1"})
        out.append(g)
    return pd.concat(out)


def counts(g):
    c = g.cls_rule.value_counts()
    return [int(c.get(k, 0)) for k in ("power_law", "exponential", "undetermined")]


def md_table(rows, head):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3out", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    h = rules_hash(a.rules)
    if h != FROZEN:
        sys.exit(f"rules hash mismatch: {h}")
    f, b = load_inputs(a)
    f = bump_and_overlap(f, a)
    A = attach(f, b)
    os.makedirs(a.outdir, exist_ok=True)

    # ---- slot grid: all 33 events x 4 maturities; slots without a fit are not_fitted
    ev_tbl = f[["ev", "event_id", "label", "event_index"]].drop_duplicates().sort_values("event_index")
    grid = pd.MultiIndex.from_product([ev_tbl.event_id, MATS], names=["event_id", "maturity"]).to_frame(index=False)
    grid = grid.merge(ev_tbl, on="event_id")
    sp, H = PRIMARY
    P = A[(A.spec == sp) & (A.H == H)].set_index(["event_id", "maturity"])
    keep = ["n", "trunc", "full", "dup", "dAIC", "ci_lo", "ci_hi", "L5", "L10", "L20", "testable", "sep_over_sigma",
            "vuong_p", "boot_p_approx_posthoc", "holm_sig_posthoc", "bh_sig_posthoc", "cls_rule", "cls_reason",
            "penalty_only", "cls_posthoc", "primary_valid", "f1_retained_valid", "excl_reasons",
            "later_bump_posthoc", "prewin_overlap_prev61_posthoc", "prewin_overlap_with", "pl_p", "pl_B", "base_pre"]
    C = grid.join(P[keep].add_prefix("P250free_"), on=["event_id", "maturity"])
    C = C.rename(columns={"P250free_cls_rule": "class", "P250free_cls_posthoc": "class_posthoc",
                          "P250free_cls_reason": "class_reason"})
    C["class"] = C["class"].fillna("not_fitted")
    C["class_posthoc"] = C["class_posthoc"].fillna("not_fitted")
    C["class_reason"] = C["class_reason"].fillna("W3_or_no_data")
    for s2, H2 in COREPORT:
        Q = A[(A.spec == s2) & (A.H == H2)].set_index(["event_id", "maturity"])
        t = f"{s2}{H2}"
        q = Q[["cls_rule", "cls_posthoc", "primary_valid", "f1_retained_valid", "excl_reasons", "n", "full"]]
        q.columns = [f"{t}_{c}" for c in ["class", "class_posthoc", "primary_valid", "f1_retained_valid", "excl", "n", "full"]]
        C = C.join(q, on=["event_id", "maturity"])
        C[f"{t}_class"] = C[f"{t}_class"].fillna("not_fitted")
        C[f"{t}_class_posthoc"] = C[f"{t}_class_posthoc"].fillna("not_fitted")
    C.drop(columns=["ev"]).to_csv(os.path.join(a.outdir, "classification_v3.csv"), index=False)

    # ---- exclusion list (every spec x window), with rule ID
    X = A[A.excl_reasons.notna()][["event_id", "maturity", "spec", "H", "n", "excl_reasons", "pl_p", "pl_B"]]
    X = X.assign(rule=X.excl_reasons.str.split(";")).explode("rule").drop(columns="excl_reasons")
    miss = C[C["class"] == "not_fitted"][["event_id", "maturity"]].assign(spec="free", H=250, n=np.nan, rule="W3_or_no_data")
    X = pd.concat([X, miss], ignore_index=True)
    X.to_csv(os.path.join(a.outdir, "classification_v3_exclusions.csv"), index=False)

    # ---- markdown
    L = []
    L.append("# classification_v3 — per-event x maturity classification (stage 2)\n")
    L.append("Generated by `v3_analysis/classify_v3.py` (step 4 of `run_all_v3.sh`). Frozen rules hash verified in-script: "
             f"`{h}`. No frozen rule changed; nothing refitted. Inputs: v3 outputs only (fits, bootstrap CIs). "
             "Label for every number: **PRELIMINARY** until Vera signs off (see findings_log.md); counts are reproducible-run numbers (**VERIFIED-RUN** only where noted).\n")
    L.append("## 0. Rules used\n"
             "- **PRIMARY spec (supervisor decision, ASSUMPTION about which is best):** 250-day window, free baseline B. Co-reported: fixed pre-shock B at 250; free and fixed at 120 and at 61 (v3 truncated rule).\n"
             "- **Class rule (as instructed):** `power_law` / `exponential` only if the block-bootstrap ΔAIC 95% CI (rules §5.2) excludes 0 on the same side at every block length run (L = 5, 10 and, at H = 250, 20); `undetermined` otherwise (CI spans 0, or class flips with block length). Sign: ΔAIC = AIC_EXP − AIC_PL, positive favours PL. `not_fitted` = fewer than 15 points or no data at that maturity (rule W3 / data start).\n"
             "- **Penalty caveat (v3 §7b, VERIFIED-RUN):** when the two curves are nearly identical (max|PL−EXP| < 0.5σ, `testable=False`) PL has one extra parameter, ΔAIC ≈ −2 and a tight CI, so the bootstrap says `exponential` by parameter penalty alone. The rule above labels those `exponential`; they are flagged `penalty_only`. **POST-HOC (not a frozen rule, changes no primary count):** `class_posthoc` turns penalty-only exponentials into `undetermined`. Quote `class_posthoc` for any claim that data favour exponential shape.\n"
             "- **Views beside every count:** *all* = every fitted slot; *primary* = frozen §4 filters (W1, W2, W3, F1, F3, F5, F6, full-window); *primary+F1-retained* = same without F1 (Vera V2-1/V3-1). Where a count has n < 20 it is descriptive only (rules §4).\n")

    sp_name = "free-B, H=250 (PRIMARY)"

    def count_block(g, name):
        al, pr, f1 = g, g[g.primary_valid], g[g.f1_retained_valid]
        r = [name]
        for v in (al, pr, f1):
            pl, ex, un = counts(v)
            po = int(v.penalty_only.sum())
            r.append(f"n={len(v)}: {pl} PL / {ex} EXP ({po} penalty-only) / {un} und.")
        return r

    L.append("## 1. Headline counts (event x maturity slots)\n")
    rows = [count_block(A[(A.spec == sp) & (A.H == H)], sp_name)]
    for s2, H2 in COREPORT:
        rows.append(count_block(A[(A.spec == s2) & (A.H == H2)], f"{s2}-B, H={H2}"))
    L.append(md_table(rows, ["spec / window", "all fitted", "primary", "primary + F1-retained"]))
    nfit = int((C["class"] != "not_fitted").sum())
    L.append(f"\nPrimary spec covers {len(C)} event x maturity slots: {nfit} fitted, {len(C) - nfit} not fitted (W3 / data start: VIX9D begins 2011-01-03, VIX3M 2006-07-17, VIX6M 2008-01-02). "
             "`undetermined` is the largest honest class; the only decisive non-penalty results are listed in §3.\n")
    P2 = A[(A.spec == sp) & (A.H == H)]
    L.append(f"After correction (POST-HOC, approximate bootstrap p from CI width, normal approx, family = free-250 fits, n = {len(P2)}): "
             f"Holm-significant {int(P2.holm_sig_posthoc.sum())}, BH-significant {int(P2.bh_sig_posthoc.sum())}; "
             f"of those, penalty-only exponentials: {int((P2.holm_sig_posthoc & P2.penalty_only).sum())} (Holm). Exact bootstrap p-values need the replicates, which v3 did not save (v3_bootstrap.csv keeps CIs only) — ASSUMPTION-level approximation. "
             f"HAC-Vuong (BH within family) significant: {int(((P2.vuong_p.notna()) & (P2.testable.astype(bool)) & (P2.vuong_p < 0.05)).sum())} raw p<0.05 of {int(P2.testable.sum())} testable (v3 §2 reports 1 BH-significant).\n")

    L.append("## 2. Per-event table, primary spec (free-B, 250 d): class by maturity, with co-reported specs\n"
             "Cell = rule class. pl = power_law, ex = exponential (curves testable), ex(p) = exponential by penalty only, un = undetermined, nf = not fitted. `*` after a class = fit is in the primary (filtered) view; `~` = in F1-retained view only. Roll-up per event (ASSUMPTION convention, not a frozen rule): decisive classes agree → that class; both PL and EXP present → mixed(undetermined); none → undetermined; ex(p) counts as ex in `roll_rule`, as undetermined in `roll_posthoc`.\n")

    def short(c, pen, valid, f1v):
        m = {"power_law": "pl", "exponential": "ex(p)" if pen else "ex", "undetermined": "un", "not_fitted": "nf"}[c]
        return m + ("*" if valid else ("~" if f1v else ""))

    rows = []
    for eid, g in C.groupby("event_id", sort=False):
        cells, cl, cph = [], [], []
        for m in MATS:
            r = g[g.maturity == m].iloc[0]
            cells.append(short(r["class"], bool(r["P250free_penalty_only"]) if r["class"] != "not_fitted" else False,
                               bool(r["P250free_primary_valid"]) if r["class"] != "not_fitted" else False,
                               bool(r["P250free_f1_retained_valid"]) if r["class"] != "not_fitted" else False))
            cl.append(r["class"])
            cph.append(r["class_posthoc"])

        def roll(cs):
            d = {c for c in cs if c in ("power_law", "exponential")}
            if d == {"power_law"}:
                return "power_law"
            if d == {"exponential"}:
                return "exponential"
            return "mixed" if len(d) == 2 else "undetermined" if any(c != "not_fitted" for c in cs) else "not_fitted"
        co = []
        for s2, H2 in COREPORT:
            t = f"{s2}{H2}"
            co.append(roll([r[f"{t}_class_posthoc"] for _, r in g.iterrows()]).replace("undetermined", "un").replace("power_law", "pl").replace("exponential", "ex"))
        rows.append([eid] + cells + [roll(cl), roll(cph)] + co)
    L.append(md_table(rows, ["event", "9D", "30D", "3M", "6M", "roll_rule", "roll_posthoc"] + [f"{s2}{H2} (posthoc roll)" for s2, H2 in COREPORT]))
    L.append("\nCo-reported columns show the post-hoc roll-up (penalty-only exponentials counted as undetermined), all fitted slots regardless of filter view. Full per-slot detail (CI, block-length classes, Vuong p, flags) is in `classification_v3.csv`.\n")

    L.append("## 3. Decisive, stable, non-penalty results (primary spec, all views)\n")
    D = C[(C["class_posthoc"].isin(["power_law", "exponential"]))]
    rows = [[r.event_id, r.maturity, r.class_posthoc, f"{r.P250free_dAIC:.2f}", f"[{r.P250free_ci_lo:.2f}, {r.P250free_ci_hi:.2f}]",
             f"{r.P250free_L5}/{r.P250free_L10}/{r.P250free_L20}", r.P250free_primary_valid, r.P250free_excl_reasons if isinstance(r.P250free_excl_reasons, str) else ""]
            for r in D.itertuples()]
    L.append(md_table(rows, ["event", "maturity", "class_posthoc", "ΔAIC", "95% CI (L=5)", "L5/L10/L20", "primary-valid", "excl rules"]) if rows else "none\n")

    L.append("\n## 4. Within-event horizon comparison (V3-3b): slots with a FULL 250-day window, free-B, rule class\n")
    full = C[C.P250free_full.astype(object).eq(True)]
    ct = pd.crosstab(full.free61_class + " → " + full.free120_class, full["class"])
    L.append(f"{len(full)} slots ({full.event_id.nunique()} events) have a full 250-day window (n ≥ 200 and not truncated below 0.8·H). Cross-tab of class at 61 → 120 versus 250:\n")
    L.append(md_table([[i] + [int(v) for v in row] for i, row in ct.iterrows()], ["class 61 → 120"] + [str(c) for c in ct.columns]))
    L.append(f"Descriptive flags on these windows (POST-HOC): later rebound (a rise ≥ 50% of the original excess above the running trough after day 10; lenient definition, so it flags most long windows) in {int(full.P250free_later_bump_posthoc.sum())}/{len(full)}; pre-onset 20-day window overlapping previous event's 61-day window in {int(full.P250free_prewin_overlap_prev61_posthoc.sum())}/{len(full)}.\n")
    allp = C[C["class"] != "not_fitted"]
    L.append(f"On all {len(allp)} fitted primary-spec slots: rebound flag {int(allp.P250free_later_bump_posthoc.sum())}, pre-window overlap flag {int(allp.P250free_prewin_overlap_prev61_posthoc.sum())}, window truncated by next onset (T) {int((allp.P250free_trunc == 'T').sum())}, duplicate of a shorter horizon (D) {int(allp.P250free_dup.astype(object).eq(True).sum())}.\n")

    L.append("## 5. Exclusions with rule ID\n"
             "Rule IDs from filter_rules_v3.md §4: W1 window peak > 10 trading days after start; W2 < 5 resolved points; W3 n < 15 (no fit); F1 PL exponent on a bound; F3 τ on a bound; F5 R² < 0 (either model); F6 B on a bound (free spec); `notfull` = truncated window with n < 0.8·H. Exclusions remove a slot from the *primary view only*; its class is still reported. Counts by rule, primary spec (a slot can carry several rules):\n")
    Xp = X[(X.spec == sp) & (X.H == H)]
    rc = Xp.groupby("rule").size().sort_values(ascending=False)
    L.append(md_table([[k, int(v)] for k, v in rc.items()], ["rule", "slots flagged (free-250)"]))
    L.append(f"\nEvery exclusion for all 8 spec x window combinations ({len(X)} rows, rule per row) is in `classification_v3_exclusions.csv`. Primary spec, slot by slot:\n")
    rows = []
    for _, r in C.iterrows():
        e = r["P250free_excl_reasons"]
        if r["class"] == "not_fitted":
            rows.append([r.event_id, r.maturity, "W3/no data", "not fitted"])
        elif isinstance(e, str):
            rows.append([r.event_id, r.maturity, e.replace(";", ", "), r["class"]])
    L.append(md_table(rows, ["event", "maturity", "rules", "class (still reported)"]))
    open(os.path.join(a.outdir, "classification_v3.md"), "w").write("\n".join(L))
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()
