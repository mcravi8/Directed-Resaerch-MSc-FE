# Publication validation

Checked November 13, 2026 (America/Los_Angeles) in a separate task copy. The connected original research folder and frozen release/report were not modified.

Environment actually used: Python 3.13.9, NumPy 2.3.5, pandas 2.2.3, SciPy 1.16.3, matplotlib 3.10.6, pytest 8.4.2. Existing installed packages were used; a fresh virtual-environment installation was not tested.

`python3 tools/run_quick_checks.py` completed successfully: 12 scientific-contract unit tests, five pooled-method tests, the deterministic calendar-overlap/independent-noise fixture, six mocked extraction tests, three fail-closed extraction tests, and independent arithmetic/consistency checks of the identification, forecast, pooled-draw and date-merging summaries. Total: 26 unit tests plus the noise and saved-summary check scripts. No real extraction, network request, full bootstrap, empirical simulation run or report/figure regeneration was performed.

`python3 tools/check_publication.py` checks Python syntax, copied-file SHA-256 identity, excluded paths/file types and common credential/private-identifier patterns. It passed before staging. Pattern scans cannot prove absence of all possible sensitive information; the allowlisted package also received source/content review. No original Git repository was present, and no original history was imported. A fresh initial commit is used for publication.

The initial Git whitespace check flagged inherited trailing spaces and CRLF CSV endings. Those bytes were intentionally preserved to keep source hashes valid; whitespace normalization was not applied. `.gitattributes` disables text conversion. Staged blobs are compared byte-for-byte with the scanned working files before the publication archive is made.

Historical full-run independent reviews were read to reconcile the README's claims. This release's lightweight tests do not replace those historical audits. Exact source-to-output reexecution remains conditional on permitted original data snapshots; source-revealing raw data and archives are excluded. Reproduction caveats and known historical runner errors are documented in REPRODUCING.md.
