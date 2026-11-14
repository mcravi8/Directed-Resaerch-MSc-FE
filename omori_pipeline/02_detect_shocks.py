"""
Step 2: Detect Shock Events in the VIX
=======================================
Identifies shock events using a threshold on daily VIX changes.
For each shock, records the onset date, peak date, peak level, and 
pre-shock baseline.

Detection rule:
  A shock is triggered when the VIX rises by >= THRESHOLD points in a 
  single day OR >= THRESHOLD_MULTI points over MULTI_DAY_WINDOW days.
  Consecutive triggers within COOLDOWN days are merged into one event.
  The shock peak is the maximum VIX within PEAK_WINDOW days of onset.

Output: data/shock_events.csv

Review the output carefully. You should see events including:
  - COVID Feb 2020
  - Volmageddon Feb 2018
  - Yen-carry unwind Aug 2024
  - And others (GFC 2008, Aug 2015 flash crash, etc.)
"""

import os
import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_PATH = "data/vix_term_structure.csv"
OUT_DIR = "data"

# Detection thresholds — adjust these if event count doesn't match your prior run
THRESHOLD_1D = 7.0        # single-day VIX point increase
THRESHOLD_MULTI = 10.0    # multi-day VIX point increase
MULTI_DAY_WINDOW = 5      # days for multi-day threshold
COOLDOWN = 30             # merge events within this many days
PEAK_WINDOW = 10          # days after onset to search for peak
BASELINE_WINDOW = 20      # trading days before onset for baseline level

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    vix = df["VIX30D"].dropna()

    # Daily change
    daily_chg = vix.diff()

    # Multi-day change
    multi_chg = vix - vix.shift(MULTI_DAY_WINDOW)

    # Trigger dates: either threshold breached
    trigger_1d = daily_chg >= THRESHOLD_1D
    trigger_md = multi_chg >= THRESHOLD_MULTI
    triggers = vix.index[trigger_1d | trigger_md]

    if len(triggers) == 0:
        print("No shocks detected. Adjust thresholds.")
        return

    # Merge triggers within cooldown window
    events = []
    last_onset = None
    for date in triggers:
        if last_onset is None or (date - last_onset).days > COOLDOWN:
            events.append(date)
            last_onset = date

    # For each event, find peak, baseline, and metadata
    records = []
    for onset in events:
        onset_loc = vix.index.get_loc(onset)

        # Peak: max VIX within PEAK_WINDOW days after onset
        peak_end = min(onset_loc + PEAK_WINDOW, len(vix) - 1)
        peak_slice = vix.iloc[onset_loc:peak_end + 1]
        peak_date = peak_slice.idxmax()
        peak_level = peak_slice.max()

        # Baseline: mean VIX in BASELINE_WINDOW days before onset
        baseline_start = max(0, onset_loc - BASELINE_WINDOW)
        baseline = vix.iloc[baseline_start:onset_loc].mean()

        # Shock magnitude
        magnitude = peak_level - baseline

        # Daily change on onset
        onset_daily_chg = daily_chg.loc[onset] if onset in daily_chg.index else np.nan

        records.append({
            "onset_date": onset,
            "peak_date": peak_date,
            "peak_level": round(peak_level, 2),
            "baseline": round(baseline, 2),
            "magnitude": round(magnitude, 2),
            "onset_daily_chg": round(onset_daily_chg, 2),
        })

    result = pd.DataFrame(records)

    # Label known events for verification
    known = {
        "1997-10": "Asian Crisis",
        "1998-08": "LTCM / Russia",
        "2001-09": "9/11",
        "2002-07": "WorldCom",
        "2007-02": "Quant Quake",
        "2007-08": "Subprime",
        "2008-09": "Lehman",
        "2008-10": "GFC Peak",
        "2010-05": "Flash Crash",
        "2011-08": "US Downgrade",
        "2015-08": "China Deval",
        "2018-02": "Volmageddon",
        "2020-02": "COVID",
        "2020-03": "COVID (cont)",
        "2022-01": "Rate Shock",
        "2024-08": "Yen Carry",
    }

    def label_event(date):
        key = date.strftime("%Y-%m")
        return known.get(key, "")

    result["label"] = result["onset_date"].apply(label_event)

    # Save
    out_path = os.path.join(OUT_DIR, "shock_events.csv")
    result.to_csv(out_path, index=False)

    print(f"Detected {len(result)} shock events")
    print(f"Saved: {out_path}\n")
    print(result.to_string(index=False))
    print(f"\n--- REVIEW THESE EVENTS ---")
    print(f"Target: ~34 events. Detected: {len(result)}")
    print(f"If count is wrong, adjust THRESHOLD_1D, THRESHOLD_MULTI, or COOLDOWN in this script.")
    print(f"Key events to verify: COVID (2020-02/03), Volmageddon (2018-02), Yen-carry (2024-08)")

if __name__ == "__main__":
    main()
