#!/usr/bin/env python3
"""61-day v3 (untruncated, window_rule=v2compat) vs v2 cross-check, using only the numbers printed in
verification/SUMMARY_v2.md (transcribed below; the executed v2 code/tables were not available).
  python3 -I crosscheck_v2_61d.py --v3out v3_out --outdir verification
"""
import argparse
import os

import numpy as np
import pandas as pd

# transcribed from SUMMARY_v2.md §4-§6
V2_COUNTS = {("free", "unfiltered"): (105, 18, 87), ("free", "primary"): (12, 7, 5), ("free", "strict"): (4, 4, 0),
             ("fixed", "unfiltered"): (105, 18, 87), ("fixed", "primary"): (24, 17, 7), ("fixed", "strict"): (6, 4, 2)}
V2_KEY = {  # (peak-date prefix, label) -> (free dAIC, fixed dAIC), VIX30D
    ("2007-03-05", "Quant Quake"): (-1.1, 17.4), ("2010-05-20", "Flash Crash"): (5.4, 7.5),
    ("2018-02-05", "Volmageddon"): (-2.8, 29.7), ("2020-03-09", "COVID"): (-2.2, -3.4),
    ("2024-08-05", "Yen Carry"): (-3.1, 18.0)}
V2_P = {"VIX9D": 0.79, "VIX30D": 0.62, "VIX3M": 0.28, "VIX6M": 0.31}  # fixed_own primary medians, c<500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3out", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    f = pd.read_csv(os.path.join(a.v3out, "v3_fits.csv"))
    g = f[(f.H == 61) & (f.window_rule == "v2compat")].copy()
    g["ev"] = g.event_id.str[:10]
    g["prim"] = g.excl_reasons.isna()
    g["strict_"] = g.strict.astype(bool)
    L = ["# crosscheck_v3_vs_v2_61d — 61-day v3 (untruncated) against v2 as printed in SUMMARY_v2.md\n",
         "**Status: PARTIAL and PRELIMINARY.** The files named in the brief (`omori_pipeline/verify/filter_rules_v2.md`, `run_all_v2.sh`, `fit_v2.py`, "
         "`filter_recount_v2.py`, `make_summary_v2.py`, `SUMMARY_v2_template.md`, `omori_pipeline/verification/SUMMARY_v2.md`) are NOT present in any folder "
         "this session can see (`omori_pipeline/` holds only the v1 scripts, `data/`, `venv/`, `plots/`); `v2_analysis/fit_v2.py` is the different, never-executed "
         "script written for `v2_fit_spec_FROZEN.md`. So v2 could not be re-run and `filter_rules_v2.md` could not be hash-checked. What follows compares v3 "
         "(H=61, untruncated) with the v2 numbers *printed* in `verification/SUMMARY_v2.md` (transcribed in this script).\n",
         "## 1. Counts (PL/EXP by point AIC)\n", "| spec | view | v2 n | v2 PL/EXP | v3 n | v3 PL/EXP | agree? |\n|---|---|---|---|---|---|---|"]
    allok = True
    for sp in ("free", "fixed"):
        d = g[g.spec == sp]
        for view, sel in (("unfiltered", d), ("primary", d[d.prim]), ("strict", d[d.strict_])):
            n2, p2, e2 = V2_COUNTS[(sp, view)]
            pl = int((sel.dAIC > 0).sum())
            ex = int((sel.dAIC < 0).sum())
            ok = (len(sel), pl, ex) == (n2, p2, e2)
            allok &= ok
            L.append(f"| {sp} | {view} | {n2} | {p2}/{e2} | {len(sel)} | {pl}/{ex} | {'yes' if ok else 'NO'} |")
    L.append("\nThe fixed-spec v2 'unfiltered' entry in SUMMARY_v2 §4 is 18/87 (same as free); v3 gives 24/81. "
             f"Exact agreement on all six rows: **{'yes' if allok else 'no'}**.\n")
    L.append("## 2. Parameter-at-bound diagnostic\n")
    fr = g[g.spec == "free"]
    L.append(f"Free-spec PL exponent within 1% of upper bound 10: v3 {int((fr.pl_p >= 10 * 0.99).sum())}/105 vs v2 53/105 (SUMMARY_v2 §0.1; "
             "the frozen F1 rule excludes these fits from the primary view).\n"
             f"Looser thresholds in v3 (same fits): p >= 3: {int((fr.pl_p >= 3).sum())}, p >= 5: {int((fr.pl_p >= 5).sum())}, p >= 8: {int((fr.pl_p >= 8).sum())} of 105. "
             "ASSUMPTION (not tested): the PL likelihood is flat along a ridge towards the exponential limit (p large, c large); v2's optimiser ran to the bound on about half of the fits, "
             "v3's stops short of the 1% tolerance on most of them, so the two tallies could be the same ridge counted at different thresholds. "
             "The near-equal ΔAIC at the key events in §3 (4 of 5 within 0.4) is consistent with that, but §4 shows primary-view p medians differ at 30D and 3M, so the filter flags are not interchangeable.\n")
    L.append("## 3. Key events, VIX30D, ΔAIC = AIC_EXP − AIC_PL (positive favours PL)\n")
    L.append("| event | v2 free | v3 free | v2 fixed | v3 fixed | sign agrees (free / fixed) |\n|---|---|---|---|---|---|")
    for (d, name), (v2f, v2x) in V2_KEY.items():
        r = []
        for sp in ("free", "fixed"):
            x = g[(g.spec == sp) & (g.ev == d) & (g.maturity == "VIX30D")]
            r.append(float(x.dAIC.iloc[0]) if len(x) else np.nan)
        s1 = "yes" if np.sign(r[0]) == np.sign(v2f) else "NO"
        s2 = "yes" if np.sign(r[1]) == np.sign(v2x) else "NO"
        L.append(f"| {name} {d} | {v2f} | {r[0]:.1f} | {v2x} | {r[1]:.1f} | {s1} / {s2} |")
    L.append("\n## 4. p by maturity, fixed spec, primary view, c < 500, median\n")
    L.append("| maturity | v2 | v3 61u | n (v3) |\n|---|---|---|---|")
    d = g[(g.spec == "fixed") & g.prim & (g.pl_c < 500)]
    for m, v in V2_P.items():
        x = d[d.maturity == m].pl_p
        L.append(f"| {m} | {v} | {x.median():.2f} | {len(x)} |")
    L.append("\n## 5. Reading (PRELIMINARY)\n"
             "Compare the rows above: the unfiltered free total (v3 21/84 vs v2 18/87), primary free size (35 vs 12) and the p-at-bound count (9 vs 53) disagree; "
             "this is the stage-1 failure again, now with the numbers printed side by side. Without v2's executed code the cause cannot be isolated: candidate causes "
             "are (i) different multistart/optimiser handling (v3 keeps non-converged starts, TRF with 1500 evaluations; v2 unknown), (ii) different W2/F1/F6 bound "
             "tolerances, (iii) v2 treating the 'p at 10' flat-likelihood optimum differently. This section does not decide which is right. To close it: copy "
             "`omori_pipeline/verify/` and `verification/SUMMARY_v2.md` from the earlier coding project into the Directed Research folder, re-run `sh verify/run_all_v2.sh`, and diff per-fit.\n")
    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, "crosscheck_v3_vs_v2_61d.md"), "w").write("\n".join(L))
    print("\n".join(L[-0:]))


if __name__ == "__main__":
    main()
