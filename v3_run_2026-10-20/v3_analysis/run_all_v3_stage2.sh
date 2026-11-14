#!/bin/sh
# One-command reproduction of the v3 run. Run from the project root:
#   sh v3_run_2026-10-20/v3_analysis/run_all_v3.sh            (writes v3_out/, v3_out_oos/)
#   OUT=v3_rerun sh v3_run_2026-10-20/v3_analysis/run_all_v3.sh  (writes v3_rerun/, v3_rerun_oos/)
# Requires: python3 (3.10.11), numpy 1.26.2, pandas 2.1.4, scipy 1.11.4.
# The 10-way worker split is done inside fit_v3.py (one subprocess per shard); no manual steps.
# Idempotent / resumable: per-job shards are cached, and a stage whose final output already exists is skipped,
# so re-running the same command after an interruption (e.g. a 300 s command limit) continues where it stopped.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
SD=$(cd "$(dirname "$0")" && pwd)
TS=omori_pipeline/data/vix_term_structure.csv
EV=omori_pipeline/data/shock_events.csv
RULES=verification/filter_rules_v3.md
JOBS=${JOBS:-10}
OUT=${OUT:-v3_out}
OOS=${OUT}_oos
# 1. main fits + bootstrap + HAC Vuong (hash of $RULES is verified inside; aborts on mismatch)
if [ ! -f "$OUT/v3_counts.csv" ]; then
  V3_SKIP_OOS=1 python3 -I "$SD/fit_v3.py" --ts $TS --events $EV --rules $RULES --out "$OUT" --jobs $JOBS
fi
# 2. out-of-sample refits (second pass, same seeds)
if [ ! -f "$OOS/v3_oos.csv" ]; then
  V3_OOS_ONLY=1 python3 -I "$SD/fit_v3.py" --ts $TS --events $EV --rules $RULES --out "$OOS" --jobs $JOBS
fi
# 3. tables and summary
python3 -I "$SD/make_tables_v3.py" "$OUT" "$OOS" > "$OUT/tables.md"
python3 -I "$SD/extra_tables_v3.py" "$OUT" > "$OUT/extra_tables.md"
cat "$SD/summary_head.md" "$OUT/tables.md" "$SD/summary_tail.md" > "$OUT/SUMMARY_v3.md"
# 4. stage 2: classification table + 61-day v2 cross-check (read-only on v3 outputs; no refit)
CLS_OUT=${CLS_OUT:-verification}
python3 -I "$SD/classify_v3.py" --v3out "$OUT" --ts $TS --events $EV --rules $RULES --outdir "$CLS_OUT"
python3 -I "$SD/crosscheck_v2_61d.py" --v3out "$OUT" --outdir "$CLS_OUT"
echo "done: $OUT/SUMMARY_v3.md, $CLS_OUT/classification_v3.md"
