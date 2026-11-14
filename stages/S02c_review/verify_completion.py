"""Lightweight integrity check for the frozen S02c C1-C6 deliverable."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_SPEC = "eb4ca7b3bef9533c40dd5f05876199ae5fbc2e1df37839b8756548fb3ea6b1b7"
FAMILIES = ("E1", "E2", "PL", "SE")


def verify():
    digest = hashlib.sha256((ROOT / "spec_C1_C2_frozen.md").read_bytes()).hexdigest()
    assert digest == EXPECTED_SPEC, f"frozen spec changed: {digest}"
    counts = {}
    for truth in FAMILIES:
        folder = ROOT / "checkpoints" / "C1" / truth
        files = sorted(folder.glob("*.json"))
        assert [p.name for p in files] == [f"{i:03d}.json" for i in range(200)], truth
        for i, path in enumerate(files):
            record = json.loads(path.read_text())
            assert record["truth"] == truth and record["replication"] == i, path
            assert set(record["starts"]) == {"1", "5"}, path
            for start in ("1", "5"):
                block = record["starts"][start]
                assert set(block["bootstrap_wins"]) == set(FAMILIES), path
                assert sum(block["bootstrap_wins"].values()) + block["bootstrap_failures"] == 199, path
        counts[truth] = len(files)
    c2 = sorted((ROOT / "checkpoints" / "C2").glob("*.json"))
    assert [p.name for p in c2] == [f"{i:03d}.json" for i in range(200)]
    for i, path in enumerate(c2):
        assert json.loads(path.read_text())["replication"] == i, path
    for name in ("C1_confusion_and_win_fraction.csv", "C1_E1_tau_bias.csv", "C2_summary.json",
                 "C3_group_sensitivity.csv", "C3_per_event_signs.csv", "C4_matched_RMSE.csv",
                 "C5_event_table.csv", "C6_E1_tau_intervals.csv", "simulation_parameters.json"):
        assert (ROOT / name).stat().st_size > 0, name
    return {"C1": counts, "C2": len(c2), "frozen_spec_sha256": digest, "status": "verified"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
