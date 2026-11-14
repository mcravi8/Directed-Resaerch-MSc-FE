"""Build a reproducibility bundle after verify_completion.py passes."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from verify_completion import verify

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "stages/S02c_review"
ARCHIVE = OUT / "S02c_completed_20261023.zip"
MANIFEST = OUT / "S02c_manifest_20261023.json"


def main():
    check = verify()
    local = [
        "spec_C1_C2_frozen.md", "spec_C1_C2_frozen.sha256", "simulation_parameters.json",
        "run_C1_C2.py", "run_C3_C6.py", "verify_completion.py", "package_completion.py",
        "C1_C2_requirement_mapping.md", "completed_scope_report.md", "interim_scope_report.md",
        "runpod_migration_stop_record.md", "runpod_migration_audit_20261023.md",
        "remote_setup_shard1.json", "remote_setup_shard2.json",
        "remote_C1_shard1_run2_summary.json", "remote_C1_shard2_run2_summary.json",
        "C1_confusion_and_win_fraction.csv", "C1_E1_tau_bias.csv", "C2_summary.json",
        "C3_group_sensitivity.csv", "C3_per_event_signs.csv", "C4_matched_RMSE.csv",
        "C4_figure1_correction.md", "C5_event_table.csv", "C6_E1_tau_intervals.csv",
    ]
    inputs = [
        "stages/S01_protocol/attempt_02/protocol_core.py",
        "stages/S02a_pooled/scripts/pipeline.py",
        "stages/S02a_pooled/pooled_summary.json",
        "stages/S02a_pooled/cohort_manifest.csv",
        "stages/S03_validation/event_losses.csv",
        "stages/S03_validation/train_test_manifest.csv",
        "omori_pipeline/data/vix_term_structure.csv",
        "omori_pipeline/data/shock_events.csv",
    ]
    paths = [OUT / name for name in local]
    paths += sorted((OUT / "checkpoints").rglob("*.json"))
    paths += sorted((OUT / "figures").glob("*.png"))
    paths += [ROOT / name for name in inputs]
    assert all(p.is_file() for p in paths), [str(p) for p in paths if not p.is_file()]
    data = {"verification": check, "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    MANIFEST.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(paths + [MANIFEST]):
            z.write(p, p.relative_to(ROOT))
    print(json.dumps({"archive": str(ARCHIVE.relative_to(ROOT)), "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), "files": len(paths) + 1, "bytes": ARCHIVE.stat().st_size}))


if __name__ == "__main__":
    main()
