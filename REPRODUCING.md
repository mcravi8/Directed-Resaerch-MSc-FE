# Reproduction workflows

Run in a disposable clone: research programs write results beside their code. Preserve the published summaries before executing production commands. The quick checks in README need no external data. Full empirical reproduction needs the inputs in DATA_ACCESS.md and can take hours; it was not rerun for publication.

## Historical acquisition and relaxation

From `omori_pipeline/`, run `python 01_fetch_data.py`, then `python 02_detect_shocks.py`. Inspect dates, missing values and event membership without changing thresholds to match a desired count. These historical scripts fetch Yahoo data, whereas the later forecast/expanded analysis uses the official Cboe files listed separately. They are not interchangeable byte snapshots.

From repository root, `python stages/S02a_pooled/scripts/pipeline.py empirical` performs pooled empirical fits after the two historical input CSVs exist. It saves atomic units and uses deterministic keyed seeds. The preserved `continuation_code_20261022_v4/stages/S02a_pooled/scripts/` contains the later parallel continuation and summarizers; those files expect to be restored to `stages/S02a_pooled/scripts/` in a separate reconstruction, not executed from the nested archival location. The original inherited pipeline remains in its canonical location because the extension calibration imported that version. Historical prepare/finalize/package helpers reference private administrative manifests that are deliberately excluded; they are not standalone publication entry points. Do not fabricate those records.

The flattened `stages/S02c_review/` contains the completed sensitivity-analysis code and tables, including the corrected plotting implementation. `v2_analysis/` and the two v3 paths preserve earlier versions, not alternative authoritative final results.

## Identification experiment

After obtaining the historical input CSVs (and retaining the published pooled_summary.json calibration), run from the root:

```sh
python extension/E02_sample_size/code/e2.py --mode prepare
python extension/E02_sample_size/code/e2.py --mode run --max-units 1
# Full production, only when intentionally requested:
python extension/E02_sample_size/code/e2.py --mode run
python extension/E02_sample_size/code/e2.py --mode summarize
```

The single-unit invocation is still a numerical simulation, not part of the publication quick checks. Production saves atomic replication units and can resume completed units. Summarization requires the full requested cells to state N80; missing cells remain pending. The historical synthetic unit archives are omitted; regenerate them rather than inventing archive receipts.

## Forecast experiment

The original `run_e3.py` is byte-preserved for its input-manifest code-hash contract. It saved forecasts before encountering a NumPy-string/Pandas timestamp error in its old audit and also contains a superseded loss-difference typo. **Do not interpret its original scoring output as the final result.** Historical corrected reproduction sequence, with the exact source snapshots and an empty monthly_models directory, is:

```sh
python extension/E03_forecast/code/run_e3.py
python extension/E03_forecast/code/audit_e3_v2.py
python extension/E03_forecast/code/audit_e3_byte_v3.py
python extension/E03_forecast/code/finish_e3_v2.py
python extension/E03_forecast/code/verify_local.py
```

The first command may stop at the known audit error after writing forecasts; establish that all expected monthly models and forecasts exist before running the corrected audits. Other failures are not expected and must be investigated. The corrected finisher uses HAR loss minus HAR-X loss and checks identities. The strict byte audit is mandatory for the stated leakage claim. The frozen runner refuses changed source hashes. If exact historical snapshots cannot be lawfully obtained, exact reproduction is blocked: establish and document a new-input study rather than bypassing the hash assertions. The 29,272 historical forecast rows and source-revealing model archives are excluded.

## Expanded public-data fits

Acquire the four Cboe maturity files, four scalar histories and historical shock catalog as documented. In a disposable clone, run:

```sh
python extension/E05_larger_sample/fresh_public_v5/e5_pipeline.py freeze
python extension/E05_larger_sample/fresh_public_v5/e5_pipeline.py prepare
python extension/E05_larger_sample/fresh_public_v5/run_pool.py freeze
python extension/E05_larger_sample/fresh_public_v5/run_public.py
python extension/E05_larger_sample/fresh_public_v5/run_pool.py run
python extension/E05_larger_sample/fresh_public_v5/verify_e5.py --require-complete
python extension/E05_larger_sample/fresh_public_v5/verify_pool.py
```

These are substantial jobs, with deterministic seeds and atomic units. Preserve freezes when resuming; do not reuse units with changed inputs. Historical packaging and independent archival verifiers also expect omitted private records or archives and are retained as source, not promised to pass in a bare clone. The eight-history catalog separately runs `python extension/E05_larger_sample/cross_index_count/count_events.py` after its own sources are present. Its date-merging and full-footprint clustering answer a different question from the fitted comparison.

No full empirical, bootstrap, licensed-data or figure rerun was performed during repository publication. The documented limitations distinguish a usable code-and-summary release from a self-contained raw-data archive.
