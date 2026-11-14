# Input retrieval and rights

No raw histories are included. Consult provider terms before downloading, using or redistributing data. Credentials must remain outside this repository.

## Official Cboe histories

The base URL is `https://cdn.cboe.com/api/global/us_indices/daily_prices/`.

| Index | File at base URL | Local destination |
|---|---|---|
| VIX30D | VIX_History.csv | extension/E03_forecast/raw/VIX30D.csv |
| VIX9D | VIX9D_History.csv | extension/E03_forecast/raw/VIX9D.csv |
| VIX3M | VIX3M_History.csv | extension/E03_forecast/raw/VIX3M.csv |
| VIX6M | VIX6M_History.csv | extension/E03_forecast/raw/VIX6M.csv |
| VXN, RVX, OVX, GVZ | INDEX_History.csv | extension/E05_larger_sample/fresh_public_v5/private_sources/INDEX_History.csv |

Create destination folders and save the original CSV bytes. `extension/E03_forecast/input_manifest.json` records expected hashes, byte sizes and the October 30, 2026 cutoff. The fitted extension's `source_manifest.csv` records the eight snapshots. Source columns use DATE plus CLOSE, or the index name for scalar files, with month/day/year dates. Do not fill missing closes. Live provider files change: truncating a newer download does not necessarily recreate the original bytes or historical revisions. A hash mismatch must be disclosed as a new-input reproduction, not silently passed as the frozen experiment.

## Original event and maturity inputs

Historical workflows expect `omori_pipeline/data/vix_term_structure.csv` (Date,VIX9D,VIX30D,VIX3M,VIX6M) and `shock_events.csv` (onset_date,peak_date,peak_level,baseline,magnitude,onset_daily_chg,label). The acquisition and detector scripts run from `omori_pipeline/`; they use Yahoo Finance tickers ^VIX9D, ^VIX, ^VIX3M and ^VIX6M and an end date of September 1, 2026. Reacquiring current history can change event membership and numerical results. Do not tune thresholds to force a historical event count. The later pooled pipeline merges the labelled GFC Peak into the Lehman onset namespace. Exact replication requires the original permitted snapshots; otherwise report the changed input and resulting cohort.

## Expanded inventory

`extension/E05_larger_sample/cross_index_count/source_manifest.json` records URLs, hashes and coverage for VXN, RVX, OVX, GVZ, VXEEM, VXAPL, VXAZN and Nikkei 225 VI. Save each as `cross_index_count/sources/INDEX.csv`. The Nikkei official URL is `https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_vi_daily_en.csv`; the historical file has 936 valid dated observations and one copyright footer, not 937 dated observations. This inventory is separate from the fitted comparison.

## Optional licensed inputs and literature

WRDS/OptionMetrics utilities are retained only as historical source code and offline tests. They are not required for the public-data findings. Use them only with independently established entitlement, schema/security checks and permitted storage; no extract is provided. Bloomberg inputs were conditional and are absent. Downloaded papers and third-party figures were excluded; obtain publications from their publisher or legitimate repository, including Marcaccioli et al., J. Stat. Mech. (2022) 023403. No paper download is required for the included tests.
