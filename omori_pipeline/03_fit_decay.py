"""
Step 3: Fit Decay Models to Post-Shock Implied Volatility
==========================================================
For each shock event and each available maturity node, fits:
  - Exponential decay:  V(t) = A * exp(-lambda * t) + V_base
  - Power-law (Omori):  V(t) = K / (t + c)^p + V_base

where t = trading days since peak, V_base = pre-shock baseline.

Compares fits using R², AIC, and BIC.

Outputs:
  - data/decay_fits.csv         (all fit parameters and diagnostics)
  - data/decay_summary.csv      (per-event: which model wins, exponents)
  - plots/                      (one plot per event showing fits)

The numbers to verify against your proposal:
  COVID decay exponents: 1.93 (9D), 1.61 (30D), 0.74 (3M), 0.37 (6M)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data/vix_term_structure.csv"
EVENTS_PATH = "data/shock_events.csv"
OUT_DIR = "data"
PLOT_DIR = "plots"

DECAY_WINDOW = 60          # trading days after peak to fit
MIN_DECAY_POINTS = 15      # minimum data points needed for fitting
MATURITIES = ["VIX9D", "VIX30D", "VIX3M", "VIX6M"]  # add "VIX1Y" if available

# ── Model definitions ──────────────────────────────────────────────────────────
def exponential_decay(t, A, lam):
    """A * exp(-lam * t), constrained to positive A and lam."""
    return A * np.exp(-lam * t)

def power_law_decay(t, K, c, p):
    """K / (t + c)^p  (modified Omori law)."""
    return K / (t + c) ** p

# ── Fitting functions ──────────────────────────────────────────────────────────
def fit_exponential(t, y):
    """Fit exponential decay. Returns (params, r_squared, aic, bic) or Nones."""
    try:
        popt, pcov = curve_fit(
            exponential_decay, t, y,
            p0=[y[0], 0.05],
            bounds=([0, 1e-6], [np.inf, 5.0]),
            maxfev=10000,
        )
        y_pred = exponential_decay(t, *popt)
        r2 = compute_r2(y, y_pred)
        n, k = len(y), 2
        aic, bic = compute_ic(y, y_pred, k)
        return {"A": popt[0], "lambda": popt[1], "R2": r2, "AIC": aic, "BIC": bic}
    except Exception:
        return None

def fit_power_law(t, y):
    """Fit power-law (Omori) decay. Returns params dict or None."""
    try:
        popt, pcov = curve_fit(
            power_law_decay, t, y,
            p0=[y[0], 1.0, 1.0],
            bounds=([0, 0.01, 0.01], [np.inf, 50.0, 10.0]),
            maxfev=10000,
        )
        y_pred = power_law_decay(t, *popt)
        r2 = compute_r2(y, y_pred)
        n, k = len(y), 3
        aic, bic = compute_ic(y, y_pred, k)
        return {"K": popt[0], "c": popt[1], "p": popt[2], "R2": r2, "AIC": aic, "BIC": bic}
    except Exception:
        return None

def compute_r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def compute_ic(y, y_pred, k):
    n = len(y)
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    if ss_res <= 0 or n <= k:
        return np.nan, np.nan
    log_lik = -n / 2 * (np.log(2 * np.pi * ss_res / n) + 1)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    return aic, bic

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    ts = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    events = pd.read_csv(EVENTS_PATH, parse_dates=["onset_date", "peak_date"])

    # Check which maturities are actually in the data
    available_mats = [m for m in MATURITIES if m in ts.columns]
    if "VIX1Y" in ts.columns and "VIX1Y" not in available_mats:
        available_mats.append("VIX1Y")
    print(f"Maturities available: {available_mats}\n")

    all_fits = []

    for _, event in events.iterrows():
        peak_date = event["peak_date"]
        baseline = event["baseline"]
        label = event.get("label", "")
        event_id = peak_date.strftime("%Y-%m-%d")

        # Get post-peak window
        if peak_date not in ts.index:
            # Find nearest trading day
            idx = ts.index.searchsorted(peak_date)
            if idx >= len(ts.index):
                continue
            peak_date = ts.index[idx]

        peak_loc = ts.index.get_loc(peak_date)
        end_loc = min(peak_loc + DECAY_WINDOW, len(ts) - 1)

        print(f"Event: {event_id} ({label})")

        for mat in available_mats:
            series = ts[mat].iloc[peak_loc:end_loc + 1].dropna()
            if len(series) < MIN_DECAY_POINTS:
                print(f"  {mat}: insufficient data ({len(series)} points)")
                continue

            # Excess above baseline
            y_raw = series.values
            y = y_raw - baseline
            y = np.maximum(y, 0.01)  # floor to avoid log(0) issues

            # Time axis: trading days since peak, starting at 1 (not 0)
            t = np.arange(1, len(y) + 1, dtype=float)

            # Fit both models
            exp_result = fit_exponential(t, y)
            pl_result = fit_power_law(t, y)

            row = {
                "event_date": event_id,
                "label": label,
                "maturity": mat,
                "peak_level": round(y_raw[0], 2),
                "baseline": round(baseline, 2),
                "n_points": len(y),
            }

            if exp_result:
                row["exp_A"] = round(exp_result["A"], 4)
                row["exp_lambda"] = round(exp_result["lambda"], 4)
                row["exp_R2"] = round(exp_result["R2"], 4)
                row["exp_AIC"] = round(exp_result["AIC"], 2)
                row["exp_BIC"] = round(exp_result["BIC"], 2)

            if pl_result:
                row["pl_K"] = round(pl_result["K"], 4)
                row["pl_c"] = round(pl_result["c"], 4)
                row["pl_p"] = round(pl_result["p"], 4)
                row["pl_R2"] = round(pl_result["R2"], 4)
                row["pl_AIC"] = round(pl_result["AIC"], 2)
                row["pl_BIC"] = round(pl_result["BIC"], 2)

            # Determine winner by AIC
            if exp_result and pl_result:
                row["winner_AIC"] = "power_law" if pl_result["AIC"] < exp_result["AIC"] else "exponential"
                row["winner_R2"] = "power_law" if pl_result["R2"] > exp_result["R2"] else "exponential"
            elif exp_result:
                row["winner_AIC"] = "exponential"
                row["winner_R2"] = "exponential"
            elif pl_result:
                row["winner_AIC"] = "power_law"
                row["winner_R2"] = "power_law"
            else:
                row["winner_AIC"] = "neither"
                row["winner_R2"] = "neither"

            all_fits.append(row)

            # Print summary line
            exp_r2 = exp_result["R2"] if exp_result else float("nan")
            pl_r2 = pl_result["R2"] if pl_result else float("nan")
            pl_p = pl_result["p"] if pl_result else float("nan")
            winner = row.get("winner_AIC", "?")
            print(f"  {mat:8s}: exp_R2={exp_r2:.3f}  pl_R2={pl_r2:.3f}  pl_p={pl_p:.2f}  winner={winner}")

        print()

    # ── Save detailed fits ──────────────────────────────────────────────────────
    fits_df = pd.DataFrame(all_fits)
    fits_path = os.path.join(OUT_DIR, "decay_fits.csv")
    fits_df.to_csv(fits_path, index=False)
    print(f"Saved detailed fits: {fits_path}")

    # ── Generate summary: per-event power-law exponents across maturities ──────
    print("\n" + "=" * 80)
    print("POWER-LAW DECAY EXPONENTS BY EVENT AND MATURITY")
    print("=" * 80)

    pivot = fits_df.pivot_table(
        index=["event_date", "label"],
        columns="maturity",
        values="pl_p",
        aggfunc="first",
    )
    # Reorder columns by maturity
    mat_order = [m for m in ["VIX9D", "VIX30D", "VIX3M", "VIX6M", "VIX1Y"] if m in pivot.columns]
    pivot = pivot[mat_order]
    print(pivot.to_string())

    pivot_path = os.path.join(OUT_DIR, "decay_exponents_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"\nSaved: {pivot_path}")

    # ── Winner summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MODEL WINNER BY AIC — PER EVENT (across all maturities)")
    print("=" * 80)

    for (edate, label), group in fits_df.groupby(["event_date", "label"]):
        winners = group["winner_AIC"].value_counts()
        dominant = winners.index[0] if len(winners) > 0 else "?"
        print(f"  {edate} {label:20s} → {dominant} ({dict(winners)})")

    # ── Specific check: COVID exponents ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERIFICATION: COVID DECAY EXPONENTS")
    print("Compare to proposal: 1.93 (9D), 1.61 (30D), 0.74 (3M), 0.37 (6M)")
    print("=" * 80)

    covid_rows = fits_df[fits_df["label"].str.contains("COVID", na=False)]
    if len(covid_rows) > 0:
        for _, row in covid_rows.iterrows():
            pl_p = row.get("pl_p", "N/A")
            print(f"  {row['maturity']:8s}: p = {pl_p}")
    else:
        print("  COVID event not found — check shock detection labels.")

    print("\n" + "=" * 80)
    print("VERIFICATION: VOLMAGEDDON DECAY EXPONENTS")
    print("=" * 80)
    volma_rows = fits_df[fits_df["label"].str.contains("Volmageddon", na=False)]
    if len(volma_rows) > 0:
        for _, row in volma_rows.iterrows():
            pl_p = row.get("pl_p", "N/A")
            print(f"  {row['maturity']:8s}: p = {pl_p}")
    else:
        print("  Volmageddon event not found — check shock detection labels.")

if __name__ == "__main__":
    main()
