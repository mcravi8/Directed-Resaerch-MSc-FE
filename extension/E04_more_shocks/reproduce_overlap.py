"""Reproduce Lyra's provisional interval counts from the saved dated catalog."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ("VXN", "RVX", "OVX", "GVZ")
rows = list(csv.DictReader((ROOT / "event_catalog.csv").open()))
eligible = [r for r in rows if r["potential_eligible"] == "1"]
parent = list(range(len(eligible)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


matrix = []
for ai, a in enumerate(NAMES):
    for b in NAMES[ai:]:
        pairs = [(i, j) for i, x in enumerate(eligible) for j, y in enumerate(eligible)
                 if x["index"] == a and y["index"] == b and (a != b or j > i)]
        same = sum(eligible[i]["onset"] == eligible[j]["onset"] for i, j in pairs)
        overlap = 0
        for i, j in pairs:
            x, y = eligible[i], eligible[j]
            if x["onset"] <= y["end_120"] and y["onset"] <= x["end_120"]:
                overlap += 1
                parent[find(i)] = find(j)
        matrix.append(dict(index_a=a, index_b=b, same_onset_pairs=same,
                           overlapping_window_pairs=overlap))

groups = defaultdict(list)
for i, row in enumerate(eligible):
    groups[find(i)].append(row)
count_by_index = {n: sum(r["index"] == n for r in eligible) for n in NAMES}
old_matrix = list(csv.DictReader((ROOT / "overlap_matrix.csv").open()))
old_clusters = list(csv.DictReader((ROOT / "clusters.csv").open()))
assert len(matrix) == len(old_matrix)
assert all(str(new[k]) == old[k] for new, old in zip(matrix, old_matrix)
           for k in ("index_a", "index_b", "same_onset_pairs", "overlapping_window_pairs"))
assert sorted(sorted(r["event_id"] for r in g) for g in groups.values()) == sorted(
    sorted(row["event_ids"].split(";")) for row in old_clusters)
summary = dict(input="saved E4 event_catalog.csv; source snapshots unavailable here",
               eligible_events=len(eligible), count_by_index=count_by_index,
               date_clusters=len(groups), same_onset_pairs=sum(x["same_onset_pairs"] for x in matrix),
               overlapping_window_pairs=sum(x["overlapping_window_pairs"] for x in matrix),
               pairwise=matrix, agrees_with_saved_matrix_and_membership=True,
               VSTOXX=None, MOVE=None, OptionMetrics=None)
(ROOT / "overlap_reproduction.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
