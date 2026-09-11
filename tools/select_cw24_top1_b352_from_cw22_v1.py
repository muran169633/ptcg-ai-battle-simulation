#!/usr/bin/env python3
"""Build the frozen train-only CW24 B352 selection.

The first 256 rows are the frozen CW22 targeted panel.  The appended 96 rows
are every exact-CW11 train-profile row whose first action is correct while the
full ordered action is wrong.  No archive member, validation row, test row, or
model is opened by this selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
SCRIPT = ROOT / "tools/select_cw24_top1_b352_from_cw22_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
B256 = ROOT / "artifacts/cw22_targeted_b256_selection_v2_20260803.json"
B256_SHA256 = "035f071902c3f43acd5cec328400d6172a990bd19864d2d794cb2cdff5f7bee7"
PROFILE = ROOT / "artifacts/u468_cw11_full_train_margin_profile_v1_20260803.json"
PROFILE_SHA256 = "da684858d2c459416234cdc3cae1ffc32a5ec23a1ec376f66a0cf440776a846e"
OUTPUT = ROOT / "artifacts/cw24_top1_b352_selection_v1.json"
SCHEMA = "ptcg-cw24-top1-b352-train-selection-v1"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"


class ProtocolError(RuntimeError):
    """Fail-closed selection error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    def check(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtocolError("nonfinite JSON value")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ProtocolError("non-string JSON key")
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)

    check(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def regular_input(path: Path, expected_sha: str, mode: int, label: str) -> dict[str, Any]:
    before = path.lstat()
    raw = path.read_bytes()
    after = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == mode,
        "identity_stable": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": hashlib.sha256(raw).hexdigest() == expected_sha,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} is not an object")
    return value


def stratum(source: str, team: str) -> str:
    if source == "core5":
        return "top1_guard_szlach" if team == "szlachetny snieg" else "top1_guard_core_other"
    if source == "flg":
        return "top1_guard_flg"
    if source == "pokemonfan":
        return "top1_guard_pf"
    raise ProtocolError(f"unexpected profile source: {source}")


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise ProtocolError("short selection write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": sha256_file(path) == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"published selection drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "checks": checks,
    }


def build() -> dict[str, Any]:
    runtime = {
        "python_exact": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.dont_write_bytecode,
        "cwd_exact": Path.cwd().resolve() == ROOT.resolve(),
    }
    if not all(runtime.values()):
        raise ProtocolError(f"runtime drift: {runtime}")
    inputs = {
        "b256": regular_input(B256, B256_SHA256, 0o444, "CW22 B256"),
        "profile": regular_input(PROFILE, PROFILE_SHA256, 0o444, "CW11 profile"),
    }
    b256 = load_json(B256, "CW22 B256")
    profile = load_json(PROFILE, "CW11 profile")
    input_checks = {
        "b256_schema": b256.get("schema_version") == "ptcg-cw22-targeted-train-b256-selection-v2",
        "b256_status": b256.get("status") == "completed_frozen_train_only_selection",
        "b256_base": b256.get("base_model_state_sha256") == CW11_MODEL_SHA256,
        "b256_rows256": isinstance(b256.get("rows"), list) and len(b256["rows"]) == 256,
        "profile_schema": profile.get("schema_version")
        == "ptcg-u468-exact-cw11-full-train-margin-profile-v1",
        "profile_status": profile.get("status") == "completed_exact_CW11_train_only_profile",
        "profile_base": profile.get("base", {}).get("model_state_sha256") == CW11_MODEL_SHA256,
        "profile_split_train": profile.get("split") == "train",
        "profile_no_new_valid": profile.get("validation_opened_for_profile") is False,
        "profile_no_test": profile.get("test_opened") is False,
    }
    if not all(input_checks.values()):
        raise ProtocolError(f"input contract drift: {input_checks}")

    top1_rows: list[dict[str, Any]] = []
    profiles = profile.get("endpoint", {}).get("profiles", {})
    if set(profiles) != {"core5", "flg", "pokemonfan"}:
        raise ProtocolError("profile source set drift")
    for source in sorted(profiles):
        section = profiles[source]
        candidates = list(section.get("near_wrong", [])) + list(section.get("fragile_correct", []))
        for row in candidates:
            predicted = [int(value) for value in row.get("predicted_order", [])]
            expert = [int(value) for value in row.get("expert_order", [])]
            if not predicted or not expert:
                continue
            if predicted[0] != expert[0] or bool(row.get("ordered_correct")):
                continue
            selected = dict(row)
            selected["source"] = source
            selected["category"] = "top1_guard"
            selected["stratum"] = stratum(source, str(row.get("team_name", "")))
            selected["selection_origin"] = "exact_CW11_train_profile_top1_correct_ordered_wrong"
            top1_rows.append(selected)

    top1_rows.sort(
        key=lambda row: (
            str(row["source"]),
            str(row["stratum"]),
            str(row["team_name"]),
            float(row["selection_margin"]),
            str(row["line_sha256"]),
        )
    )
    base_rows = [dict(row) for row in b256["rows"]]
    all_rows = base_rows + top1_rows
    base_sha = {str(row["line_sha256"]) for row in base_rows}
    top_sha = {str(row["line_sha256"]) for row in top1_rows}
    counts = {
        "rows": len(all_rows),
        "base_rows": len(base_rows),
        "top1_guard_rows": len(top1_rows),
        "top1_strata": dict(sorted(Counter(str(row["stratum"]) for row in top1_rows).items())),
        "top1_sources": dict(sorted(Counter(str(row["source"]) for row in top1_rows).items())),
        "all_sources": dict(sorted(Counter(str(row["source"]) for row in all_rows).items())),
    }
    terminal_checks = {
        "rows352": len(all_rows) == 352,
        "base256": len(base_rows) == 256,
        "top1_guard96": len(top1_rows) == 96,
        "top1_unique96": len(top_sha) == 96,
        "base_unique256": len(base_sha) == 256,
        "no_overlap": not (base_sha & top_sha),
        "union_unique352": len(base_sha | top_sha) == 352,
        "strata_exact": counts["top1_strata"]
        == {
            "top1_guard_core_other": 18,
            "top1_guard_flg": 27,
            "top1_guard_pf": 36,
            "top1_guard_szlach": 15,
        },
        "sources_exact": counts["top1_sources"]
        == {"core5": 33, "flg": 27, "pokemonfan": 36},
        "all_members_train": all(str(row["member"]).startswith("train/") for row in all_rows),
        "all_top1_contract": all(
            row["predicted_order"][0] == row["expert_order"][0]
            and row["ordered_correct"] is False
            for row in top1_rows
        ),
    }
    if not all(terminal_checks.values()):
        raise ProtocolError(f"selection terminal gate failed: {terminal_checks}")
    row_payload_sha = hashlib.sha256(canonical_json(all_rows)).hexdigest()
    return {
        "schema_version": SCHEMA,
        "status": "completed_frozen_train_only_B352_selection",
        "base_model_state_sha256": CW11_MODEL_SHA256,
        "rows": all_rows,
        "counts": counts,
        "rows_canonical_sha256": row_payload_sha,
        "contract": {
            "first_256": "frozen CW22 B256 in original order",
            "appended_96": "all exact-CW11 train-profile top1-correct ordered-wrong rows",
            "selection_uses_train_profile_only": True,
            "archive_members_opened": 0,
            "validation_or_test_rows_opened": 0,
            "changed_candidate_created": False,
            "official_evaluation_count": 0,
            "submission_performed": False,
        },
        "runtime": runtime,
        "inputs": inputs,
        "input_checks": input_checks,
        "terminal_checks": terminal_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = build()
    if args.audit_only:
        print(canonical_json({
            "status": "audit_only_pass",
            "counts": result["counts"],
            "terminal_checks": result["terminal_checks"],
            "output_exists": OUTPUT.exists(),
            "writes_performed": 0,
        }).decode(), end="")
        return
    payload = canonical_json(result)
    evidence = publish_o_excl(OUTPUT, payload)
    print(canonical_json({
        "status": result["status"],
        "counts": result["counts"],
        "rows_canonical_sha256": result["rows_canonical_sha256"],
        "output": evidence,
    }).decode(), end="")


if __name__ == "__main__":
    main()
