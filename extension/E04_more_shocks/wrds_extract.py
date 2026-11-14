"""Read-only OptionMetrics handoff. Only Matteo runs --probe/--extract under his WRDS login."""
import argparse
import csv
import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path

IDENT = re.compile(r"^[a-z][a-z0-9_]*$")
FIELDS = ("secid", "date", "days", "delta", "cp_flag", "iv")
CONFIG_KEYS = frozenset(("library", "annual_table_prefix", "start_date", "end_date",
                         "secids", "days", "delta", "cp_flag", "columns",
                         "schema_confirmed", "security_identity_confirmed"))


def validate_config(c, require_confirmed=False):
    if not isinstance(c, dict) or set(c) != CONFIG_KEYS:
        raise ValueError("Configuration must contain exactly the documented keys")
    if not isinstance(c["columns"], dict) or set(c["columns"]) != set(FIELDS):
        raise ValueError("Exactly six distinct confirmed columns are required")
    if len(set(map(str, c["columns"].values()))) != len(FIELDS):
        raise ValueError("Exactly six distinct confirmed columns are required")
    for value in [c["library"], c["annual_table_prefix"], *c["columns"].values()]:
        if type(value) is not str or not IDENT.fullmatch(value):
            raise ValueError(f"Unsafe SQL identifier: {value!r}")
    try:
        start, end = date.fromisoformat(c["start_date"]), date.fromisoformat(c["end_date"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid ISO dates") from exc
    if (type(c["start_date"]) is not str or type(c["end_date"]) is not str
            or start.isoformat() != c["start_date"] or end.isoformat() != c["end_date"]):
        raise ValueError("Dates must be YYYY-MM-DD")
    if (start > end or type(c["secids"]) is not list or not c["secids"]
            or any(type(x) is not int or x <= 0 for x in c["secids"])):
        raise ValueError("Invalid dates or secids")
    if len(set(c["secids"])) != len(c["secids"]):
        raise ValueError("Duplicate secids")
    if type(c["days"]) is not int or type(c["delta"]) is not int or type(c["cp_flag"]) is not str:
        raise ValueError("Selection has invalid types")
    if (c["days"], c["delta"], c["cp_flag"]) != (30, 50, "C"):
        raise ValueError("Frozen selection requires 30-day, 50-delta call IV")
    for key in ("schema_confirmed", "security_identity_confirmed"):
        if type(c[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
    if require_confirmed and (c["schema_confirmed"] is not True
                              or c["security_identity_confirmed"] is not True):
        raise ValueError("Confirm WRDS column meanings and security identity in config first")
    return c


def config(path, require_confirmed=False):
    return validate_config(json.loads(Path(path).read_text()), require_confirmed)


def checked_rows(rows, c):
    """Pure offline validator; yields normalized valid rows and reject counts."""
    out, rejects, keys = [], {}, set()
    lo, hi = date.fromisoformat(c["start_date"]), date.fromisoformat(c["end_date"])
    for r in rows:
        try:
            d = date.fromisoformat(str(r["date"])[:10])
            secid = int(r["secid"])
            days, delta = int(r["days"]), int(r["delta"])
            flag = str(r["cp_flag"]).upper()
            iv = float(r["iv"])
            if not (lo <= d <= hi and secid in c["secids"] and days == c["days"]
                    and delta == c["delta"] and flag == c["cp_flag"]):
                reason = "outside_selection"
            elif not math.isfinite(iv) or iv <= 0:
                reason = "invalid_iv"
            elif (secid, d) in keys:
                raise ValueError(f"Duplicate security/date key: {secid}, {d}")
            else:
                keys.add((secid, d))
                out.append({"secid": secid, "date": d.isoformat(), "days": days,
                            "delta": delta, "cp_flag": flag, "iv": iv,
                            "source_table": str(r["source_table"])})
                continue
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if str(exc).startswith("Duplicate security/date key"):
                raise
            reason = "malformed"
        rejects[reason] = rejects.get(reason, 0) + 1
    return out, rejects


def metadata_tables(db, c):
    names = set(db.list_tables(library=c["library"]))
    pattern = re.compile(re.escape(c["annual_table_prefix"]) + r"(\d{4})$")
    years = range(date.fromisoformat(c["start_date"]).year,
                  date.fromisoformat(c["end_date"]).year + 1)
    tables = [(y, c["annual_table_prefix"] + str(y)) for y in years]
    available = {n: int(pattern.fullmatch(n).group(1)) for n in names if pattern.fullmatch(n)}
    return tables, available


def probe(db, c):
    tables, available = metadata_tables(db, c)
    print(json.dumps({"requested_years": [tables[0][0], tables[-1][0]],
                      "candidate_tables": sorted(available),
                      "missing_requested_tables": [t for _, t in tables if t not in available]}, indent=2))
    for _, table in tables:
        if table in available:
            desc = db.describe_table(library=c["library"], table=table)
            print(f"{c['library']}.{table}: {desc.to_string(index=False)}")


def extract(db, c, output):
    validate_config(c, require_confirmed=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new private output directory; existing files are never replaced")
    tables, available = metadata_tables(db, c)
    missing = [t for _, t in tables if t not in available]
    if missing:
        raise RuntimeError(f"Requested annual tables unavailable: {missing}; revise scope and refreeze first")
    cols = c["columns"]
    for _, table in tables:
        desc = db.describe_table(library=c["library"], table=table)
        observed = set(desc["name"].str.lower()) if "name" in desc else set(desc["variable_name"].str.lower())
        if not set(cols.values()).issubset(observed):
            raise RuntimeError(f"Schema mismatch in {table}: missing {set(cols.values()) - observed}")
    output.mkdir(parents=True)
    records = []
    selected = ", ".join(f"{cols[k]} AS {k}" for k in FIELDS)
    secids = ",".join(map(str, c["secids"]))
    for year, table in tables:
        left = max(c["start_date"], f"{year}-01-01")
        right = min(c["end_date"], f"{year}-12-31")
        sql = (f"SELECT {selected} FROM {c['library']}.{table} "
               f"WHERE {cols['secid']} IN ({secids}) AND {cols['date']} >= '{left}' "
               f"AND {cols['date']} <= '{right}' AND {cols['days']} = {c['days']} "
               f"AND {cols['delta']} = {c['delta']} AND {cols['cp_flag']} = '{c['cp_flag']}'")
        frame = db.raw_sql(sql)
        raw = frame.to_dict("records")
        for r in raw:
            r["source_table"] = table
        rows, rejects = checked_rows(raw, c)
        target = output / f"optionmetrics_iv_{year}.csv"
        with target.open("x", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=(*FIELDS, "source_table"))
            writer.writeheader(); writer.writerows(rows)
        records.append({"year": year, "table": table, "queried_rows": len(raw),
                        "retained_rows": len(rows), "rejects": rejects,
                        "first_date": min((r["date"] for r in rows), default=None),
                        "last_date": max((r["date"] for r in rows), default=None),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "file": target.name})
    (output / "manifest.json").write_text(json.dumps({"selection": c, "files": records}, indent=2) + "\n")
    print(f"Saved {len(records)} annual files and manifest in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "extract"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="New private directory (extract only)")
    args = parser.parse_args()
    c = config(args.config, require_confirmed=args.mode == "extract")
    if args.mode == "extract" and not args.output:
        parser.error("--output is required for extract")
    import wrds  # Deferred: offline tests require no account or package.
    db = wrds.Connection()
    try:
        if args.mode == "probe":
            probe(db, c)
        else:
            extract(db, c, args.output)
    finally:
        db.close()


if __name__ == "__main__":
    main()
