"""Offline fail-closed regression tests. Synthetic rows and mocked WRDS only."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_vera_wrds_offline import FakeDB
from wrds_extract import config, extract, main

TEMPLATE = Path(__file__).with_name("wrds_config.template.json")


class CountingDB(FakeDB):
    def __init__(self):
        super().__init__(["vsurfd2020"], [{"secid": 108105, "date": "2020-03-02",
                                    "days": 30, "delta": 50, "cp_flag": "C", "iv": 0.25}])
        self.metadata_calls = 0

    def list_tables(self, library):
        self.metadata_calls += 1
        return super().list_tables(library)

    def close(self):
        pass


class GateTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads(TEMPLATE.read_text())
        self.base.update(start_date="2020-01-01", end_date="2020-12-31")

    def test_extract_rejects_invalid_before_connection(self):
        cases = {
            "string_false_bypass": dict(schema_confirmed="false", security_identity_confirmed="false"),
            "string_true": dict(schema_confirmed="true"),
            "missing_confirmation": {"drop": "schema_confirmed"},
            "null_confirmation": dict(schema_confirmed=None),
            "numeric_confirmation": dict(security_identity_confirmed=1),
            "boolean_false": dict(security_identity_confirmed=False),
            "wrong_days": dict(days=91),
            "wrong_delta": dict(delta=-50),
            "put_instead_of_call": dict(cp_flag="P"),
            "unsafe_identifier": dict(library="optionm;DROP"),
            "missing_columns": {"drop": "columns"},
            "extra_key": dict(unreviewed_setting=True),
        }
        for label, edits in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                c = dict(self.base, schema_confirmed=True, security_identity_confirmed=True)
                edits = dict(edits)
                if "drop" in edits:
                    c.pop(edits.pop("drop"))
                c.update(edits)
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(c))
                calls = []
                fake_wrds = types.SimpleNamespace(Connection=lambda: calls.append("connected"))
                with patch.dict(sys.modules, {"wrds": fake_wrds}), patch.object(
                    sys, "argv", ["wrds_extract.py", "extract", "--config", str(path),
                                 "--output", str(Path(tmp) / "out")]
                ):
                    with self.assertRaises((ValueError, TypeError)):
                        main()
                self.assertEqual(calls, [])
                self.assertFalse((Path(tmp) / "out").exists())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{malformed JSON")
            calls = []
            with patch.dict(sys.modules, {"wrds": types.SimpleNamespace(Connection=lambda: calls.append(1))}), patch.object(
                sys, "argv", ["wrds_extract.py", "extract", "--config", str(path),
                             "--output", str(Path(tmp) / "out")]
            ):
                with self.assertRaises(ValueError):
                    main()
            self.assertEqual(calls, [])

    def test_direct_extract_rejects_before_metadata(self):
        cases = [dict(self.base, schema_confirmed="false", security_identity_confirmed="false"),
                 dict(self.base, schema_confirmed=True, security_identity_confirmed=False),
                 dict(self.base, schema_confirmed=True, security_identity_confirmed=True, days=91),
                 dict(self.base, schema_confirmed=True, security_identity_confirmed=True, delta=-50),
                 dict(self.base, schema_confirmed=True, security_identity_confirmed=True, cp_flag="P")]
        for c in cases:
            with self.subTest(config=c), tempfile.TemporaryDirectory() as tmp:
                db = CountingDB()
                with self.assertRaises(ValueError):
                    extract(db, c, Path(tmp) / "out")
                self.assertEqual(db.metadata_calls, 0)
                self.assertEqual(db.sql, [])

    def test_confirmed_synthetic_extract_and_unconfirmed_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(self.base))
            self.assertFalse(config(path)["schema_confirmed"])
            db = CountingDB()
            with patch.dict(sys.modules, {"wrds": types.SimpleNamespace(Connection=lambda: db)}), patch.object(
                sys, "argv", ["wrds_extract.py", "probe", "--config", str(path)]
            ):
                main()
            self.assertGreater(db.metadata_calls, 0)
            self.assertEqual(db.sql, [])
            confirmed = dict(self.base, schema_confirmed=True, security_identity_confirmed=True)
            path.write_text(json.dumps(confirmed))
            out = Path(tmp) / "out"
            with patch.dict(sys.modules, {"wrds": types.SimpleNamespace(Connection=lambda: db)}), patch.object(
                sys, "argv", ["wrds_extract.py", "extract", "--config", str(path), "--output", str(out)]
            ):
                main()
            self.assertEqual(len(db.sql), 1)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "optionmetrics_iv_2020.csv").exists())


if __name__ == "__main__":
    unittest.main()
