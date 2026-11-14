#!/usr/bin/env python3
"""Extra SUMMARY_v3 tables (F1-retained counts, block-length decisiveness, per-event winners).
Reads v3_fits.csv and v3_bootstrap.csv only; no fitting. Usage: extra_tables_v3.py OUT_DIR > extra_tables.md

Definitions
- F1-retained primary = primary view with rule F1 (PL p on a bound) not applied; every other primary rule still applies.
- F1-retained strict = F1-retained primary and strict's extra conditions (c off bound, both R2 >= 0.5, stable multistart).
- Not decisive = bootstrap class differs across the block lengths run for that fit (L = 5, 10; plus 20 at H = 250).
  Not-decisive fits are left out of the decisive PL/EXP/indist counts and reported on their own.
"""
import sys, os
import numpy as np, pandas as pd

out = sys.argv[1]
F = pd.read_csv(os.path.join(out, "v3_fits.csv"))
B = pd.read_csv(os.path.join(out, "v3_bootstrap.csv"))
for c in ("primary", "strict", "c_on_bound", "unstable"):
    F[c] = F[c].astype(bool)
F["Hs"] = np.where(F.window_rule == "v2compat", "61u", F.H.astype(str))
HORD = ["61u", "61", "120", "250"]
key = ["event_id", "maturity", "H", "spec", "window_rule"]
nl = B.groupby(key).cls.nunique().rename("n_cls").reset_index()
F = F.merge(nl, on=key, how="left")
F["not_decisive"] = F.n_cls > 1
toks = F.excl_reasons.fillna("").str.split(";")
F["f1_retained"] = toks.apply(lambda t: set(x for x in t if x) <= {"F1"})
F["f1_strict"] = F.f1_retained & ~F.c_on_bound & ~F.unstable & (F.pl_R2 >= 0.5) & (F.exp_R2 >= 0.5)
assert (F.primary == (F.excl_reasons.fillna("") == "")).all()
assert not (F.strict & ~F.f1_strict).any() and not (F.primary & ~F.f1_retained).any()

VIEWS = [("unfiltered", lambda d: d), ("primary", lambda d: d[d.primary]), ("primary+F1-retained", lambda d: d[d.f1_retained]),
         ("strict", lambda d: d[d.strict]), ("strict+F1-retained", lambda d: d[d.f1_strict])]


def md(df):
    cols = list(df.columns)
    o = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(o)


def counts(d):
    return dict(n=len(d), AIC_PL_EXP=f"{int((d.dAIC > 0).sum())}/{int((d.dAIC < 0).sum())}",
                boot_PL=int((d.boot_cls_L5 == "PL").sum()), boot_EXP=int((d.boot_cls_L5 == "exp").sum()),
                boot_ind=int((d.boot_cls_L5 == "indistinguishable").sum()))


p = print
p("## 2a. F1-retained counts beside primary / strict (per window and spec)")
p("F1-retained = same view with rule F1 (PL exponent on a bound) not applied. Columns are bootstrap L=5 classes and point-AIC PL/EXP. "
  "F1 removes fits where PL has collapsed to p = 0.01 or 10; the F1-retained rows show what those removals do to the counts.\n")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        d0 = F[(F.spec == spec) & (F.Hs == hs)]
        for name, fn in VIEWS[1:]:
            rows.append(dict(spec=spec, H=hs, view=name, **counts(fn(d0))))
p(md(pd.DataFrame(rows)))

p("\n## 2c. Decisive counts: fits whose bootstrap verdict changes with block length are NOT decisive")
p("A fit is not decisive if its bootstrap class (PL / EXP / indistinguishable) is not the same at every block length run (L = 5 and 10; also 20 at H = 250). "
  "Decisive counts exclude those fits; `not_decisive` is shown separately. Decisive PL/EXP counts are the only bootstrap counts to quote.\n")
rows = []
for spec in ("free", "fixed"):
    for hs in HORD:
        d0 = F[(F.spec == spec) & (F.Hs == hs)]
        for name, fn in VIEWS:
            d = fn(d0); dd = d[~d.not_decisive]
            rows.append(dict(spec=spec, H=hs, view=name, n=len(d), not_decisive=int(d.not_decisive.sum()),
                             dec_PL=int((dd.boot_cls_L5 == "PL").sum()), dec_EXP=int((dd.boot_cls_L5 == "exp").sum()),
                             dec_ind=int((dd.boot_cls_L5 == "indistinguishable").sum())))
p(md(pd.DataFrame(rows)))
p("\nNot-decisive fits (every view; unfiltered list):\n")
nd = F[F.not_decisive].copy()
bc = B.groupby(key).apply(lambda g: ", ".join(f"L{int(l)}={c}" for l, c in zip(g.L, g.cls))).rename("classes").reset_index()
nd = nd.merge(bc, on=key)
nd = nd[["event_id", "maturity", "Hs", "spec", "primary", "strict", "classes"]].sort_values(["spec", "Hs", "event_id", "maturity"])
p(md(nd) if len(nd) else "none")

p("\n## 2b. Per-event winners at H = 61 (truncated) / 120 / 250, free and fixed baseline")
p("Cell = decisive bootstrap winner across the event's usable maturities [PL/EXP/ind counts; point-AIC PL/EXP wins]. "
  "Winner = PL if decisive PL > decisive EXP, EXP if the reverse, `tie` if equal and > 0, `ind` if no decisive PL or EXP. "
  "Fits that are not decisive across block lengths are excluded from the bootstrap counts (point-AIC counts use every maturity). "
  "EXP wins in curve-indistinguishable fits reflect the parameter penalty (§7b).\n")
for spec in ("free", "fixed"):
    p(f"\n### {spec} baseline\n")
    rows = []
    evs = F[["event_id", "label"]].drop_duplicates("event_id").sort_values("event_id")
    for _, e in evs.iterrows():
        r = dict(event=e.event_id)
        for hs in HORD[1:]:
            d = F[(F.spec == spec) & (F.Hs == hs) & (F.event_id == e.event_id)]
            if len(d) == 0:
                r["H" + hs] = "-"; continue
            dd = d[~d.not_decisive]
            npl, nex = int((dd.boot_cls_L5 == "PL").sum()), int((dd.boot_cls_L5 == "exp").sum())
            ni = int((dd.boot_cls_L5 == "indistinguishable").sum())
            w = "PL" if npl > nex else ("EXP" if nex > npl else ("tie" if npl else "ind"))
            r["H" + hs] = f"{w} [{npl}/{nex}/{ni}; {int((d.dAIC > 0).sum())}/{int((d.dAIC < 0).sum())}]" + (f" nd={int(d.not_decisive.sum())}" if d.not_decisive.any() else "")
        rows.append(r)
    R = pd.DataFrame(rows)
    p(md(R))
    for hs in HORD[1:]:
        c = R["H" + hs].str.split(" ").str[0].value_counts().to_dict()
        p(f"\nH{hs} winner tally ({spec}): " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
