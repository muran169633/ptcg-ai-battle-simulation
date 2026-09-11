#!/usr/bin/env python3
"""Audit exact-state action memorization across archive train/valid splits.

The validation split is never used to construct the lookup.  Card-instance
``serial`` values are removed because they are not consumed by the BC model
and are not stable identifiers for strategically identical observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import orjson


def without_serials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_serials(item)
            for key, item in value.items()
            if key != "serial"
        }
    if isinstance(value, list):
        return [without_serials(item) for item in value]
    return value


def state_digest(row: dict[str, Any]) -> bytes:
    observation = without_serials(row.get("observation") or {})
    rendered = orjson.dumps(observation, option=orjson.OPT_SORT_KEYS)
    return hashlib.blake2b(rendered, digest_size=16).digest()


def action_tuple(row: dict[str, Any]) -> tuple[int, ...] | None:
    raw = row.get("action")
    if not isinstance(raw, list):
        return None
    try:
        return tuple(sorted(int(index) for index in raw))
    except (TypeError, ValueError):
        return None


def rows(archive: zipfile.ZipFile, split: str) -> Iterator[dict[str, Any]]:
    members = sorted(
        name
        for name in archive.namelist()
        if name.startswith(f"{split}/") and name.endswith(".jsonl")
    )
    for member in members:
        with archive.open(member) as handle:
            for line in handle:
                yield orjson.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    counts: dict[bytes, Counter[tuple[int, ...]]] = {}
    train_rows = 0
    with zipfile.ZipFile(args.data) as archive:
        for row in rows(archive, "train"):
            if args.max_train_rows is not None and train_rows >= args.max_train_rows:
                break
            action = action_tuple(row)
            if action is None:
                continue
            digest = state_digest(row)
            counter = counts.get(digest)
            if counter is None:
                counter = Counter()
                counts[digest] = counter
            counter[action] += 1
            train_rows += 1
            if train_rows % 100_000 == 0:
                print(
                    f"train_rows={train_rows:,} unique_states={len(counts):,}",
                    flush=True,
                )

        lookup = {
            digest: counter.most_common(1)[0][0]
            for digest, counter in counts.items()
        }
        ambiguous_states = sum(len(counter) > 1 for counter in counts.values())

        valid_rows = 0
        covered = 0
        correct = 0
        for row in rows(archive, "valid"):
            if args.max_valid_rows is not None and valid_rows >= args.max_valid_rows:
                break
            action = action_tuple(row)
            if action is None:
                continue
            prediction = lookup.get(state_digest(row))
            valid_rows += 1
            if prediction is not None:
                covered += 1
                correct += int(prediction == action)
            if valid_rows % 50_000 == 0:
                print(
                    f"valid_rows={valid_rows:,} covered={covered:,} correct={correct:,}",
                    flush=True,
                )

    result = {
        "schema_version": "ptcg-bc-exact-state-lookup-audit-v1",
        "data": str(args.data.resolve()),
        "signature": "blake2b128(canonical_observation_without_serial_fields)",
        "train_rows": train_rows,
        "train_unique_states": len(counts),
        "train_ambiguous_states": ambiguous_states,
        "valid_rows": valid_rows,
        "covered_rows": covered,
        "coverage": covered / valid_rows if valid_rows else 0.0,
        "correct_rows": correct,
        "accuracy_on_covered": correct / covered if covered else 0.0,
        "accuracy_with_uncovered_as_wrong": correct / valid_rows if valid_rows else 0.0,
        "validation_used_to_build_lookup": False,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
