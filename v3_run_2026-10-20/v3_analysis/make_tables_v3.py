#!/usr/bin/env python3
"""Tables for SUMMARY_v3.md from v3_out (main) and v3_out_oos. Usage: make_tables_v3.py MAIN_OUT OOS_OUT > tables.md"""
import sys, os
import numpy as np, pandas as pd

main, oosd = sys.argv[1], sys.argv[2]
F = pd.read_csv(os.path.join(main, "v3_fits.csv"))
C = pd.read_csv(os.path.join(main, "v3_counts.csv"))
Bt = pd.read_csv(os.path.join(main, "v3_bootstrap.csv"))
O = pd.read_csv(os.path.join(oosd, "v3_oos.csv"))
ET = pd.read_csv(os.path.join(main, "v3_event_types.csv"))
for c in ("full", "primary", "strict", "testable", "block_sensitive", "dup"):
    F[c] = F[c].astype(bool)
F["trunc"] = F["trunc"].fillna("")
F["Hs"] = np.where(F.window_rule == "v2compat", "61u", F.H.astype(str))
HORD = ["61u", "61", "120", "250"]


def md(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:.{floatfmt}g}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v))) for v in r) + " |")
    return "\n".join(out)


p = print
p("## 2. Headline counts (PL / EXP / indistinguishable by block-bootstrap L=5; point-AIC PL/EXP in brackets)")
p("`61u` = untruncated 61-day windows (the gate; = v2's window rule). `61` = 61 days truncated at the next onset (v3 rule). 120/250 truncated. Sign test (two-sided) uses bootstrap-classified PL vs EXP only; count claims allowed only for primary views with n >= 20.\n")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        for view in ("unfiltered", "primary", "strict"):
            wr, H = ("v2compat", 61) if hs == "61u" else ("trunc", int(hs))
            r = C[(C.window_rule == wr) & (C.H == H) & (C.spec == spec) & (C.view == view)].iloc[0]
            sp = r.boot_sign_p
            rows.append(dict(spec=spec, H=hs, view=view, n=r.n, boot_PL=r.boot_PL, boot_EXP=r.boot_EXP, boot_indist=r.boot_indist,
                             AIC_PL_EXP=f"{r.AIC_PL}/{r.AIC_EXP}", BIC_PL_EXP=f"{r.BIC_PL}/{r.BIC_EXP}",
                             sign_p=("" if not np.isfinite(sp) else f"{sp:.3g}"),
                             count_claim="yes" if (view == "primary" and r.n >= 20) else "descriptive",
                             HAC_testable=r.testable, HAC_sig_BH=r.hac_sig, block_sens=r.block_sensitive))
p(md(pd.DataFrame(rows)))

p("\n## 3. Windows: truncation, duplicates, completeness")
rows = []
for hs in HORD[1:]:
    for spec in ("free", "fixed"):
        d = F[(F.Hs == hs) & (F.spec == spec)]
        rows.append(dict(H=hs, spec=spec, fits=len(d), truncated_T=int((d.trunc == "T").sum()), full=int(d.full.sum()),
                         not_full=int((~d.full).sum()), duplicates_of_shorter=int(d.dup.sum()), median_n=float(d.n.median())))
p(md(pd.DataFrame(rows)))

p("\n## 4. Exclusions by rule (count of fits flagged; a fit can carry several)")
rows = []
for hs in HORD:
    for spec in ("free", "fixed"):
        d = F[(F.Hs == hs) & (F.spec == spec)]
        r = dict(H=hs, spec=spec, fits=len(d))
        for k in ("W1", "W2", "F1", "F3", "F5", "F6", "notfull"):
            r[k] = int(d.excl_reasons.fillna("").str.contains(k).sum())
        r["c_on_bound(flag)"] = int(d.c_on_bound.sum()); r["B-(>20% below B)"] = int(d.B_minus.sum())
        r["primary"] = int(d.primary.sum()); r["p_at_upper_bound"] = int((d.pl_p >= 9.9).sum())
        rows.append(r)
p(md(pd.DataFrame(rows)))

p("\n## 5. B-hat minus pre-shock baseline (free spec; vol points). Median [IQR]; primary view and unfiltered")
rows = []
for hs in HORD:
    for view in ("unfiltered", "primary"):
        d = F[(F.Hs == hs) & (F.spec == "free")]
        d = d if view == "unfiltered" else d[d.primary]
        d = d[np.isfinite(d.pl_Bdiff)]
        q = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]" if len(s) else ""
        rows.append(dict(H=hs, view=view, n=len(d), PL_Bhat_minus_pre=q(d.pl_Bdiff), EXP_Bhat_minus_pre=q(d.exp_Bdiff),
                         share_PL_B_within_1pt=float((d.pl_Bdiff.abs() <= 1).mean()) if len(d) else np.nan,
                         share_EXP_B_within_1pt=float((d.exp_Bdiff.abs() <= 1).mean()) if len(d) else np.nan,
                         median_EXP_tau=float(d.exp_tau.median()) if len(d) else np.nan))
p(md(pd.DataFrame(rows)))

p("\n## 6. Power-law exponent p by maturity (median [n]; primary view; c < 500 only)")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        d = F[(F.Hs == hs) & (F.spec == spec) & F.primary & (F.pl_c < 500 * 0.99)]
        r = dict(spec=spec, H=hs)
        for m in ("VIX9D", "VIX30D", "VIX3M", "VIX6M"):
            x = d[d.maturity == m].pl_p
            r[m] = f"{x.median():.2f} [{len(x)}]" if len(x) else "- [0]"
        rows.append(r)
p(md(pd.DataFrame(rows)))
p("\nSame, unfiltered (all fits; p at bounds included):")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        d = F[(F.Hs == hs) & (F.spec == spec)]
        r = dict(spec=spec, H=hs)
        for m in ("VIX9D", "VIX30D", "VIX3M", "VIX6M"):
            x = d[d.maturity == m].pl_p
            r[m] = f"{x.median():.2f} [{len(x)}]"
        rows.append(r)
p(md(pd.DataFrame(rows)))

p("\n## 7. Bootstrap detail")
p("Share of ALL usable fits by class and block length (unfiltered). L=20 only at H=250.\n")
rows = []
for (spec, H, wr, L), g in Bt.groupby(["spec", "H", "window_rule", "L"]):
    hs = "61u" if wr == "v2compat" else str(H)
    rows.append(dict(spec=spec, H=hs, L=L, n=len(g), PL=int((g.cls == "PL").sum()), EXP=int((g.cls == "exp").sum()),
                     indist=int((g.cls == "indistinguishable").sum()), median_CI_width=float((g.ci_hi - g.ci_lo).median())))
B2 = pd.DataFrame(rows); B2["o"] = B2.H.map({h: i for i, h in enumerate(HORD)}); B2 = B2.sort_values(["spec", "o", "L"]).drop(columns="o")
p(md(B2))
p("\nHAC Vuong (Newey-West), tested only where curves differ by >= 0.5 sigma; BH within window x spec x view:\n")
rows = []
for hs in HORD:
    for spec in ("free", "fixed"):
        d = F[(F.Hs == hs) & (F.spec == spec)]
        t = d[d.testable]
        rows.append(dict(H=hs, spec=spec, fits=len(d), testable=len(t), share_testable=len(t) / len(d),
                         sig_unfiltered_BH=int((t.vq_unfiltered < .05).sum()),
                         sig_PL=int(((t.vq_unfiltered < .05) & (t.vuong_z > 0)).sum()),
                         sig_EXP=int(((t.vq_unfiltered < .05) & (t.vuong_z < 0)).sum()),
                         median_resid_lag1_PL=float(d.resid_lag1_PL.median())))
p(md(pd.DataFrame(rows)))

p("\n## 8. Per-event winners (bootstrap-classified, L=5; type from all usable maturities; PL-type = PL at >=2 maturities and EXP at none)")
for spec in ("free", "fixed"):
    p(f"\n### {spec} baseline: event type at H = 61 (trunc) / 120 / 250  [PL-wins/EXP-wins counts across maturities]")
    rows = []
    E = ET[(ET.view == "unfiltered") & (ET.spec == spec) & (ET.window_rule == "trunc")]
    for eid in sorted(F.event_id.unique(), key=lambda s: s):
        r = dict(event=eid)
        for H in (61, 120, 250):
            x = E[(E.event_id == eid) & (E.H == H)]
            r[f"H{H}"] = f"{x.type.iloc[0]} [{x.PL.iloc[0]}/{x.EXP.iloc[0]}] n={x.n_mat.iloc[0]}" if len(x) else "-"
        rows.append(r)
    p(md(pd.DataFrame(rows)))
    for view in ("primary",):
        E2 = ET[(ET.view == view) & (ET.spec == spec) & (ET.window_rule == "trunc")]
        p(f"\nEvent-type tallies, {view} view: " + "; ".join(
            f"H{H}: " + ", ".join(f"{k}={v}" for k, v in E2[E2.H == H].type.value_counts().items()) for H in (61, 120, 250)))
    E3 = ET[(ET.view == "unfiltered") & (ET.spec == spec) & (ET.window_rule == "trunc")]
    p(f"Event-type tallies, unfiltered: " + "; ".join(
        f"H{H}: " + ", ".join(f"{k}={v}" for k, v in E3[E3.H == H].type.value_counts().items()) for H in (61, 120, 250)))

p("\n## 9. Key events (point dAIC = AIC_EXP - AIC_PL, positive favours PL; boot class in brackets), 30D")
rows = []
for key in ("Quant Quake", "Flash Crash", "Volmageddon", "COVID", "Yen Carry", "GFC", "US Downgrade", "China Deval"):
    for spec in ("free", "fixed"):
        r = dict(event=key, spec=spec)
        for hs in HORD:
            d = F[(F.Hs == hs) & (F.spec == spec) & (F.maturity == "VIX30D") & F.event_id.str.contains(key, regex=False)]
            r["H" + hs] = f"{d.dAIC.iloc[0]:+.1f} ({d.boot_cls_L5.iloc[0][:3]}){'*' if d.primary.iloc[0] else ''} B-pre={d.pl_Bdiff.iloc[0]:+.1f}/{d.exp_Bdiff.iloc[0]:+.1f}" if len(d) and spec == "free" else (
                f"{d.dAIC.iloc[0]:+.1f} ({d.boot_cls_L5.iloc[0][:3]}){'*' if d.primary.iloc[0] else ''}" if len(d) else "-")
        rows.append(r)
p("`*` = primary-valid. free cells also show B-hat minus pre-shock (PL/EXP).\n")
p(md(pd.DataFrame(rows)))

p("\n## 10. Out-of-sample (descriptive; refit on the training part only; lower RMSE wins)")
P = F[["event_id", "maturity", "Hs", "spec", "primary"]].copy(); P["H"] = P.Hs
O["Hs"] = np.where(O.window_rule == "v2compat", "61u", O.H.astype(str))
O = O[O.status == "ok"].merge(P[["event_id", "maturity", "Hs", "spec", "primary"]], on=["event_id", "maturity", "Hs", "spec"], how="left")
O["pl_better"] = O.pl_RMSE < O.exp_RMSE
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        for split in ("first33", "first50", "lastout20"):
            for pv in (False, True):
                d = O[(O.spec == spec) & (O.Hs == hs) & (O.split == split)]
                d = d[d.primary.fillna(False).astype(bool)] if pv else d
                if len(d):
                    rows.append(dict(spec=spec, H=hs, split=split, view="primary" if pv else "all", n=len(d), PL_better_RMSE=int(d.pl_better.sum()),
                                     med_PL_RMSE=float(d.pl_RMSE.median()), med_EXP_RMSE=float(d.exp_RMSE.median())))
p(md(pd.DataFrame(rows)))

p("\n## 7b. Bootstrap class among curve-distinguishable fits (max |PL-EXP| >= 0.5 sigma) vs the rest (unfiltered)")
p("Where curves are near-identical the dAIC is about -2(k difference) with a tiny CI, so a bootstrap 'EXP' there is the parameter penalty, not evidence about shape. Read the 'testable' rows for model evidence.\n")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        d = F[(F.Hs == hs) & (F.spec == spec)]
        for nm, g in (("testable", d[d.testable]), ("not testable", d[~d.testable])):
            rows.append(dict(spec=spec, H=hs, group=nm, n=len(g), PL=int((g.boot_cls_L5 == "PL").sum()), EXP=int((g.boot_cls_L5 == "exp").sum()),
                             indist=int((g.boot_cls_L5 == "indistinguishable").sum()), median_dAIC=float(g.dAIC.median()) if len(g) else np.nan))
p(md(pd.DataFrame(rows)))
p("\nSame, primary view only, testable fits:\n")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        g = F[(F.Hs == hs) & (F.spec == spec) & F.primary & F.testable]
        rows.append(dict(spec=spec, H=hs, n=len(g), PL=int((g.boot_cls_L5 == "PL").sum()), EXP=int((g.boot_cls_L5 == "exp").sum()),
                         indist=int((g.boot_cls_L5 == "indistinguishable").sum())))
p(md(pd.DataFrame(rows)))
