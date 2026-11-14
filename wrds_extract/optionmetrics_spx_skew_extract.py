"""
OptionMetrics SPX volatility-surface extract for the Omori project (D-04).

FOR MATTEO — run once, under your own WRDS login (agents never use it):
  1. pip install wrds pandas
  2. From this folder:  python optionmetrics_spx_skew_extract.py
     (the wrds package prompts for your username/password/Duo itself; nothing is stored here)
  3. It writes ../optionm_spx_vsurf.csv (Directed Research folder) and prints the date range.
     If it fails, copy the printed error/table list into the Directed Research folder as wrds_error.txt.

Pulls SPX (secid 108105) from the OptionMetrics standardized volatility surface
(optionm.vsurfdYYYY per-year tables, fallback optionm.vsurfd), maturities 30 and 91 days,
deltas 50 (ATM call), 25 (25-delta call), -25 (25-delta put), 1996 onward.
Skew measure to be built later by Forge: IV(25d put) - IV(25d call), and IV(25d put) - IV(ATM).

Status: drafted by Chief 2026-10-19, not yet run. Table/column names follow the standard
IvyDB US schema (ASSUMPTION until the first run confirms them).
"""

import os
import sys
import pandas as pd

try:
    import wrds
except ImportError:
    sys.exit("Install the WRDS package first:  pip install wrds pandas")

SECID = 108105            # S&P 500 index (SPX) in OptionMetrics
DAYS = (30, 91)
DELTAS = (50, 25, -25)
START_YEAR = 1996
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "optionm_spx_vsurf.csv")


def main():
    db = wrds.Connection()  # interactive login handled by the wrds package
    try:
        tables = set(db.list_tables(library="optionm"))
    except Exception as e:
        sys.exit(f"Could not list OptionMetrics tables (no subscription access?): {e}")

    year_tables = sorted(t for t in tables if t.startswith("vsurfd") and t[6:].isdigit())
    days = ",".join(str(d) for d in DAYS)
    deltas = ",".join(str(d) for d in DELTAS)
    cols = "secid, date, days, delta, cp_flag, impl_volatility, impl_strike"

    frames = []
    if year_tables:
        for t in year_tables:
            if int(t[6:]) < START_YEAR:
                continue
            q = (f"select {cols} from optionm.{t} "
                 f"where secid = {SECID} and days in ({days}) and delta in ({deltas})")
            df = db.raw_sql(q, date_cols=["date"])
            print(f"{t}: {len(df)} rows")
            frames.append(df)
    elif "vsurfd" in tables:
        q = (f"select {cols} from optionm.vsurfd "
             f"where secid = {SECID} and days in ({days}) and delta in ({deltas}) "
             f"and date >= '{START_YEAR}-01-01'")
        frames.append(db.raw_sql(q, date_cols=["date"]))
    else:
        print("No vsurfd tables found. Available OptionMetrics tables:")
        for t in sorted(tables):
            print("  ", t)
        sys.exit(1)

    db.close()
    out = pd.concat(frames, ignore_index=True).sort_values(["date", "days", "delta"])
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out)} rows to {os.path.abspath(OUT)}")
    if len(out):
        print(f"Date range: {out['date'].min().date()} to {out['date'].max().date()}")


if __name__ == "__main__":
    main()
