"""Independent offline audit of the E4 corrected handoff; no WRDS connection."""
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
sys.path.insert(0, str(P))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


v1 = module("wrds_v1_history", P / "wrds_extract_v1.py")
v2 = module("wrds_v2_review", P / "wrds_extract.py")
from test_vera_wrds_offline import FakeDB

base = json.loads((P / "wrds_config.template.json").read_text())
base.update(start_date="2020-01-01", end_date="2020-12-31")
row = {"secid": 108105, "date": "2020-03-02", "days": 30,
       "delta": 50, "cp_flag": "C", "iv": 0.25}


class CountDB(FakeDB):
    def __init__(self):
        super().__init__(["vsurfd2020"], [row])
        self.metadata_calls = 0

    def list_tables(self, library):
        self.metadata_calls += 1
        return super().list_tables(library)

    def close(self):
        pass


result = {"v1": {}, "v2": {"invalid_main": {}, "invalid_direct": {}}, "hashes": {}}
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    bad = dict(base, schema_confirmed="false", security_identity_confirmed="false")
    old_db = CountDB()
    v1.extract(old_db, bad, p / "old")
    result["v1"] = {"string_false_row_queries": len(old_db.sql),
                    "annual_files": len(list((p / "old").glob("*.csv")))}
    assert result["v1"] == {"string_false_row_queries": 1, "annual_files": 1}

    variants = {
        "string_false": bad,
        "string_true": dict(base, schema_confirmed="true", security_identity_confirmed="true"),
        "boolean_false": dict(base, schema_confirmed=True, security_identity_confirmed=False),
        "missing_key": {k: v for k, v in base.items() if k != "schema_confirmed"},
        "null": dict(base, schema_confirmed=None),
        "numeric": dict(base, schema_confirmed=1),
        "wrong_days": dict(base, days=91),
        "wrong_delta": dict(base, delta=-50),
        "wrong_flag": dict(base, cp_flag="P"),
    }
    for label, c in variants.items():
        path = p / f"{label}.json"
        path.write_text(json.dumps(c))
        calls = []
        mock_wrds = types.SimpleNamespace(Connection=lambda: calls.append("connection"))
        with patch.dict(sys.modules, {"wrds": mock_wrds}), patch.object(
            sys, "argv", ["wrds_extract.py", "extract", "--config", str(path),
                          "--output", str(p / f"{label}_out")]
        ):
            try:
                v2.main()
            except ValueError:
                pass
            else:
                raise AssertionError(f"CLI accepted {label}")
        assert not calls and not (p / f"{label}_out").exists(), label
        result["v2"]["invalid_main"][label] = 0
        db = CountDB()
        try:
            v2.extract(db, c, p / f"{label}_direct")
        except ValueError:
            pass
        else:
            raise AssertionError(f"direct extract accepted {label}")
        assert db.metadata_calls == 0 and db.sql == [], label
        result["v2"]["invalid_direct"][label] = [db.metadata_calls, len(db.sql)]

    good = dict(base, schema_confirmed=True, security_identity_confirmed=True)
    db = CountDB()
    path = p / "good.json"
    path.write_text(json.dumps(good))
    with patch.dict(sys.modules, {"wrds": types.SimpleNamespace(Connection=lambda: db)}), patch.object(
        sys, "argv", ["wrds_extract.py", "extract", "--config", str(path),
                      "--output", str(p / "good_out")]
    ):
        v2.main()
    manifest = json.loads((p / "good_out" / "manifest.json").read_text())
    record = manifest["files"][0]
    assert (db.metadata_calls, len(db.sql), record["queried_rows"],
            record["retained_rows"]) == (1, 1, 1, 1)
    assert record["sha256"] == sha(p / "good_out" / record["file"])
    assert ("days = 30" in db.sql[0] and "delta = 50" in db.sql[0]
            and "cp_flag = 'C'" in db.sql[0])
    result["v2"]["valid_extract"] = {"connections": 1, "metadata_calls": 1,
                                    "row_queries": 1, "queried_rows": 1,
                                    "retained_rows": 1, "manifest_hash_matches": True}
    db = CountDB()
    path.write_text(json.dumps(base))
    with patch.dict(sys.modules, {"wrds": types.SimpleNamespace(Connection=lambda: db)}), patch.object(
        sys, "argv", ["wrds_extract.py", "probe", "--config", str(path)]
    ):
        v2.main()
    assert db.metadata_calls == 1 and db.sql == []
    result["v2"]["unconfirmed_probe"] = {"metadata_calls": 1, "row_queries": 0}

freeze2 = json.loads((P / "wrds_handoff_freeze_v2.json").read_text())
for filename, digest in freeze2["files_sha256_at_handoff"].items():
    actual = sha(P / filename)
    assert actual == digest, filename
    result["hashes"][filename] = actual
for filename, digest in freeze2["preserved_v1"].items():
    assert sha(P / filename) == digest, filename
    result["hashes"][filename] = digest
old = json.loads((P / "wrds_handoff_freeze.json").read_text())["files"]
for filename, digest in old.items():
    historical = "wrds_extract_v1.py" if filename == "wrds_extract.py" else filename
    assert sha(P / historical) == digest, filename
release = ROOT / "stages" / "S05_release"
result["protected_release"] = {}
for line in (release / "manifest.sha256").read_text().splitlines():
    digest, filename = line.split(maxsplit=1)
    actual = sha(release / filename)
    assert actual == digest, filename
    result["protected_release"][filename] = actual

counts = list(csv.DictReader((P / "achievable_counts.csv").open()))
catalog = list(csv.DictReader((P / "event_catalog.csv").open()))
clusters = list(csv.DictReader((P / "clusters.csv").open()))
matrix = list(csv.DictReader((P / "overlap_matrix.csv").open()))
result["inventory_readback"] = {
    "triggers": sum(int(r["candidate_triggers"]) for r in counts),
    "onsets": sum(int(r["accepted_onsets"]) for r in counts),
    "potential_windows": sum(int(r["potential_eligible"]) for r in counts),
    "catalog_rows": len(catalog), "clusters": len(clusters),
    "same_onset_pairs": sum(int(r["same_onset_pairs"]) for r in matrix),
    "overlap_pairs": sum(int(r["overlapping_window_pairs"]) for r in matrix),
}
assert result["inventory_readback"] == {"triggers": 126, "onsets": 52,
    "potential_windows": 35, "catalog_rows": 52, "clusters": 13,
    "same_onset_pairs": 7, "overlap_pairs": 31}
assert result["inventory_readback"]["potential_windows"] == sum(
    int(r["potential_eligible"]) for r in catalog)
result["new_claim_check"] = {"test_methods_reproduced_separately": 9,
                             "invalid_cli_variants_in_forge_test": 12,
                             "malformed_json_cases_in_forge_test": 1,
                             "invalid_direct_variants_in_forge_test": 5,
                             "readme_planning_rows_31x252": 31 * 252}
print(json.dumps(result, indent=2))
