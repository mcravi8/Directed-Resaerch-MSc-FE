"""Synthetic only: no WRDS import, connection, credentials or licensed rows."""
import json
import tempfile
import unittest
from pathlib import Path

from wrds_extract import checked_rows, config


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.c = config(Path(__file__).with_name("wrds_config.template.json"))
        self.row = dict(secid=108105, date="2020-03-02", days=30, delta=50,
                        cp_flag="C", iv=0.25, source_table="SYNTHETIC")

    def test_quality_and_bounds(self):
        bad = [dict(self.row, date="2027-01-01"), dict(self.row, date="2020-03-03", iv=0),
               dict(self.row, date="2020-03-04", iv=float("nan")),
               dict(self.row, date="2020-03-05", days=91)]
        rows, rejects = checked_rows([self.row, *bad], self.c)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rejects, {"outside_selection": 2, "invalid_iv": 2})
        self.assertEqual(rows[0]["iv"], 0.25)

    def test_duplicate_key_aborts(self):
        with self.assertRaisesRegex(ValueError, "Duplicate security/date"):
            checked_rows([self.row, self.row], self.c)

    def test_untrusted_identifier_rejected(self):
        broken = dict(self.c, library="optionm;drop")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "Unsafe SQL identifier"):
                config(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
