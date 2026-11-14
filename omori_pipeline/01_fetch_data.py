"""
Step 1: Fetch VIX Term Structure Data
=====================================
Downloads daily closing values for VIX term structure indices from Yahoo Finance.
Saves a single CSV with columns: VIX9D, VIX30D, VIX3M, VIX6M.

If you have VIX1Y data from another source (WRDS, CBOE, Bloomberg), place it as
a CSV named 'vix1y_external.csv' in this directory with columns: Date, Close.
The script will merge it automatically.

Output: data/vix_term_structure.csv
"""

import os
import pandas as pd
import yfinance as yf

# ── Config ──────────────────────────────────────────────────────────────────────
START = "1990-01-01"
END   = "2026-09-01"
OUT_DIR = "data"

TICKERS = {
    "VIX9D":  "^VIX9D",   # 9-day implied vol,  available ~2011+
    "VIX30D": "^VIX",     # 30-day (the standard VIX), available 1990+
    "VIX3M":  "^VIX3M",   # 3-month, available ~2007+
    "VIX6M":  "^VIX6M",   # 6-month, available ~2008+
}

# ── Download ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    frames = {}
    for name, ticker in TICKERS.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(ticker, start=START, end=END, progress=False)
        if len(df) == 0:
            print(f"  WARNING: no data for {name}")
            continue
        close = df["Close"].squeeze()
        close.name = name
        frames[name] = close
        print(f"  {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}, {len(close)} rows")

    combined = pd.DataFrame(frames)
    combined.index.name = "Date"

    # ── Optional: merge external VIX1Y ──────────────────────────────────────────
    vix1y_path = "vix1y_external.csv"
    if os.path.exists(vix1y_path):
        print(f"\nMerging external VIX1Y from {vix1y_path}...")
        ext = pd.read_csv(vix1y_path, parse_dates=["Date"], index_col="Date")
        ext.columns = ["VIX1Y"]
        combined = combined.join(ext, how="left")
        non_null = combined["VIX1Y"].notna().sum()
        print(f"  Merged {non_null} VIX1Y observations")
    else:
        print(f"\nNo {vix1y_path} found — proceeding with 4 maturity nodes.")
        print("  (Place a CSV with columns Date,Close in this directory to add VIX1Y)")

    # ── Save ────────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "vix_term_structure.csv")
    combined.to_csv(out_path)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {combined.shape}")
    print(f"\nDate coverage per index:")
    for col in combined.columns:
        s = combined[col].dropna()
        print(f"  {col:8s}: {s.index[0].strftime('%Y-%m-%d')} to {s.index[-1].strftime('%Y-%m-%d')} ({len(s)} obs)")

if __name__ == "__main__":
    main()
