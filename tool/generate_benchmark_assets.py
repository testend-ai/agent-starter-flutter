"""Normalize the frozen IVA_Test.csv golden dataset into the benchmark catalog.

Implements docs/iva-golden-dataset-implementation-plan.md section 4 (P0 ingest):

    assets/benchmark/
    ├── tasks_v1.json     normalized task specs (plan 4.2 canonical schema)
    ├── intents_v1.json   unique-intent catalog with ground-truth labels
    ├── personas_v1.json  persona clusters (name x gender x anrede)
    ├── manifest.json     schema version, source/artifact hashes, counts
    └── IVA_Test.csv      frozen original (hash recorded in manifest.json)

Pipeline (plan 4.3): utf-8-sig `;`-CSV -> validate required fields ->
drop ``Ausgeschlossen=Ja`` rows -> unique ``{intent}#{contact:04d}`` ids ->
pronouns derived from gender only -> deterministic JSON artifacts.

Stdlib-only and idempotent: re-running over an unchanged CSV reproduces the
three data artifacts byte-for-byte. Pass --timestamp <iso> to also pin the
manifest for diffable regeneration; default stamps the current UTC time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"
GENERATOR = "tool/generate_benchmark_assets.py"

REQUIRED_COLUMNS = ("intent_name", "description", "scenario", "caller_name", "gender", "anrede")

# plan 4.3 step 5: pronouns derive from gender only, never from the name.
PRONOUNS_DE = {
    "männlich": ["er", "ihm"],
    "weiblich": ["sie", "ihr"],
}

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "assets" / "benchmark" / "IVA_Test.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _int_or_none(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def _contact_index(contact_id: str, fallback: int, warnings: list[str]) -> int:
    """contact ids are numeric in the golden set; degrade gracefully if not."""
    if contact_id.isdigit():
        return int(contact_id)
    warnings.append(f"contact id '{contact_id}' is not numeric; using row {fallback} for task_id")
    return fallback


def normalize(csv_path: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    """CSV -> (tasks, intents, personas, counts). Pure: no I/O beyond reading."""
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    tasks: list[dict] = []
    seen_ids: set[str] = set()
    openers = Counter()
    counts = {
        "rows_total": len(rows),
        "dropped_excluded": 0,
        "dropped_missing_fields": 0,
        "duplicate_task_ids_dropped": 0,
        "tasks_emitted": 0,
    }
    warnings: list[str] = []

    for row_index, row in enumerate(rows):
        if (row.get("Ausgeschlossen") or "").strip().lower() == "ja":
            counts["dropped_excluded"] += 1
            continue

        missing = [c for c in REQUIRED_COLUMNS if not (row.get(c) or "").strip()]
        if missing:
            counts["dropped_missing_fields"] += 1
            warnings.append(f"contact {row.get('ID des Kontakts', '?')}: missing {', '.join(missing)}")
            continue

        intent = row["intent_name"].strip()
        contact_id = row["ID des Kontakts"].strip()
        contact_index = _contact_index(contact_id, row_index + 1, warnings)
        task_id = f"{intent}#{contact_index:04d}"
        if task_id in seen_ids:
            counts["duplicate_task_ids_dropped"] += 1
            warnings.append(f"duplicate task_id dropped: {task_id}")
            continue
        seen_ids.add(task_id)

        gender = row["gender"].strip() or "weiblich"
        if gender not in PRONOUNS_DE:
            warnings.append(f"task {task_id}: unknown gender '{gender}', pronouns left empty")

        opener = row["scenario"].strip()
        openers[opener] += 1
        tasks.append(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "source_row": row_index + 2,  # +1 header, +1 zero-based
                "intent": {
                    "name": intent,
                    "expected_output": row["expected_output"].strip(),
                },
                "goal": row["description"].strip(),
                "opener": opener,
                "persona": {
                    "caller_name": row["caller_name"].strip(),
                    "gender": gender,
                    "anrede": row["anrede"].strip() or "Sie",
                    "pronouns_de": PRONOUNS_DE.get(gender, []),
                },
                "metadata": {
                    "callernbr": (row.get("callernbr") or "").strip(),
                    "priority": _int_or_none((row.get("Priorität") or "").strip()),
                    "wrap_up_type": (row.get("Wrap-up-Typ") or "").strip(),
                    "contact_id": _int_or_none(contact_id) or contact_index,
                },
                "filters": {
                    "excluded": False,
                    "excluded_detail": (row.get("Ausgeschlossenes Detail") or "").strip() or None,
                },
            }
        )

    counts["tasks_emitted"] = len(tasks)
    counts["duplicate_openers"] = sum(n - 1 for n in openers.values() if n > 1)
    counts["unique_intents"] = len({t["intent"]["name"] for t in tasks})
    counts["unique_personas"] = len({(t["persona"]["caller_name"], t["persona"]["gender"], t["persona"]["anrede"]) for t in tasks})

    # ---- intents catalog: one entry per unique intent, ground truth attached
    by_intent: dict[str, list[dict]] = {}
    for task in tasks:
        by_intent.setdefault(task["intent"]["name"], []).append(task)
    intents = [
        {
            "name": name,
            "expected_output": group[0]["intent"]["expected_output"],
            "task_count": len(group),
            "task_ids": sorted(t["task_id"] for t in group),
        }
        for name, group in sorted(by_intent.items())
    ]
    conflicts = [
        name
        for name, group in by_intent.items()
        if len({t["intent"]["expected_output"] for t in group}) > 1
    ]
    for name in conflicts:
        warnings.append(f"intent {name}: inconsistent expected_output across tasks")

    # ---- persona clusters
    by_persona: dict[tuple[str, str, str], Counter] = {}
    for task in tasks:
        p = task["persona"]
        key = (p["caller_name"], p["gender"], p["anrede"])
        by_persona.setdefault(key, Counter())["tasks"] += 1
    personas = [
        {
            "caller_name": name,
            "gender": gender,
            "anrede": anrede,
            "pronouns_de": PRONOUNS_DE.get(gender, []),
            "task_count": counter["tasks"],
        }
        for (name, gender, anrede), counter in sorted(by_persona.items())
    ]

    tasks.sort(key=lambda t: t["task_id"])
    return tasks, intents, personas, {"counts": counts, "warnings": sorted(warnings)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="source IVA_Test.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CSV.parent, help="artifact output directory")
    parser.add_argument("--timestamp", default=None, help="ISO-8601 stamp for manifest (default: now UTC)")
    args = parser.parse_args(argv)

    if not args.csv.is_file():
        parser.error(f"dataset not found: {args.csv}")
    generated_at = args.timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    tasks, intents, personas, extra = normalize(args.csv)
    counts, warnings = extra["counts"], extra["warnings"]

    # Data artifacts carry only CSV-derived content: re-running the tool over
    # an unchanged dataset reproduces them byte-for-byte (plan P0 gate).
    artifacts = {
        "tasks_v1.json": {
            "schema_version": SCHEMA_VERSION,
            "count": len(tasks),
            "tasks": tasks,
        },
        "intents_v1.json": {
            "schema_version": SCHEMA_VERSION,
            "count": len(intents),
            "intents": intents,
        },
        "personas_v1.json": {
            "schema_version": SCHEMA_VERSION,
            "count": len(personas),
            "personas": personas,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in artifacts.items():
        write_json(args.out_dir / filename, payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "generated_at": generated_at,
        "source": {
            "dataset": args.csv.name,
            "sha256": sha256_file(args.csv),
        },
        "artifacts": {
            name.removesuffix(".json"): {"file": name, "sha256": sha256_file(args.out_dir / name)}
            for name in artifacts
        },
        "counts": counts,
        "distributions": {
            "gender": dict(sorted(Counter(t["persona"]["gender"] for t in tasks).items())),
            "anrede": dict(sorted(Counter(t["persona"]["anrede"] for t in tasks).items())),
        },
        "warnings": warnings,
    }
    write_json(args.out_dir / "manifest.json", manifest)

    print(f"source         : {args.csv} (sha256 {manifest['source']['sha256'][:12]}…)")
    print(f"tasks          : {counts['tasks_emitted']} emitted -> tasks_v1.json")
    print(f"intents        : {counts['unique_intents']} unique -> intents_v1.json")
    print(f"personas       : {counts['unique_personas']} clusters -> personas_v1.json")
    print(
        f"dropped        : {counts['dropped_excluded']} excluded, "
        f"{counts['dropped_missing_fields']} missing fields, "
        f"{counts['duplicate_task_ids_dropped']} duplicate ids"
    )
    if warnings:
        print(f"warnings       : {len(warnings)} (see manifest.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
