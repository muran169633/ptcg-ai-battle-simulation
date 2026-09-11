#!/usr/bin/env python3
"""Freeze corrected CW22 train-only B256 selection without rewriting v1.

The correction changes exactly two strata from the frozen v1 selection:
Dominic ctx0 hard includes every exact-CW11 type10/area5 pair target, and the
FLG broad guard is restricted to context 3.  No dataset archive is opened.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "select_cw22_targeted_b256_from_cw11_profile_v2.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw22-targeted-train-b256-selection-v2"

V1_SELECTOR = TOOLS / "select_cw22_targeted_b256_from_cw11_profile_v1.py"
V1_SELECTOR_SHA256 = "ac240f65a8c35dbd4297019c39bad0714bc35044a7cafa61be5a0ee66fc96555"
V1_SELECTOR_MODE = 0o555
V1_SELECTION = ROOT / "artifacts/cw22_targeted_b256_selection_v1_20260803.json"
V1_SELECTION_SHA256 = "eb5e7dbde09d82648203bc0506d2d7dfe0d6e707a2351e1c4b5023c2965b4426"
V1_SELECTION_MODE = 0o444
V1_SELECTION_SCHEMA = "ptcg-cw22-targeted-train-b256-selection-v1"
V1_SELECTION_PAYLOAD_SHA256 = "d6e93edba87c61bdc35c6076a4a9f305de82b2b9224bbe2db965dc6307c9e72b"
PROFILE = ROOT / "artifacts/u468_cw11_full_train_margin_profile_v1_20260803.json"
PROFILE_SHA256 = "da684858d2c459416234cdc3cae1ffc32a5ec23a1ec376f66a0cf440776a846e"
PROFILE_MODE = 0o444
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"

DOMINIC = "dominic_ctx0_hard"
FLG = "broad_flg_retention"
REBUILT_STRATA = (DOMINIC, FLG)
BATCH_SIZE = 256
DOMINIC_QUOTA = 32
DOMINIC_SPECIAL_QUOTA = 9
FLG_CONTEXT3_QUOTA = 16


class SelectionError(RuntimeError):
    """Fail-closed corrected-selection error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def evidence(
    path: Path, expected_sha: str, expected_mode: int, label: str
) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise SelectionError(f"unsafe {label} identity/mode")
    digest = sha256_file(path)
    after = path.lstat()
    checks = {
        "sha_exact": digest == expected_sha,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact": stat.S_IMODE(after.st_mode) == expected_mode,
    }
    if not all(checks.values()):
        raise SelectionError(f"{label} evidence drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def load_json_strict(path: Path, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectionError(f"{label}: duplicate key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SelectionError(f"{label}: nonfinite constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise SelectionError(f"{label}: root is not an object")
    return payload


def load_v1() -> tuple[ModuleType, dict[str, Any]]:
    provenance = evidence(
        V1_SELECTOR, V1_SELECTOR_SHA256, V1_SELECTOR_MODE, "v1 selector"
    )
    source = V1_SELECTOR.read_bytes()
    if hashlib.sha256(source).hexdigest() != V1_SELECTOR_SHA256:
        raise SelectionError("v1 selector changed after evidence")
    name = "cw22_selection_v2_frozen_v1"
    if name in sys.modules:
        raise SelectionError("v1 module name already occupied")
    module = types.ModuleType(name)
    module.__file__ = str(V1_SELECTOR)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(source, str(V1_SELECTOR), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, provenance


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    ]
    forbidden = {
        "load_state_dict",
        "backward",
        "step",
        "save",
        "submit",
        "upload",
    }
    hits = sorted(name for name in calls if name in forbidden)
    checks = {
        "no_training_archive_or_submission_calls": not hits
        and "ZipFile" not in calls,
        "exact_one_publish_call": calls.count("publish_o_excl") == 1,
        "stdout_summary_present": "print" in calls,
        "dev_null_pycache_present": b'sys.pycache_prefix = "/dev/null"' in source,
    }
    if not all(checks.values()):
        raise SelectionError(f"source audit failed: {checks}; hits={hits}")
    return {
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "checks": checks,
        "forbidden_hits": hits,
        "pass": True,
    }


def validate_runtime() -> dict[str, Any]:
    checks = {
        "repo_root": Path.cwd().resolve() == ROOT,
        "my_project_env": Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1
        and sys.dont_write_bytecode is True,
        "pycache_prefix_dev_null": sys.pycache_prefix == "/dev/null",
    }
    if not all(checks.values()):
        raise SelectionError(f"runtime drift: {checks}")
    return {"python": str(Path(sys.executable).resolve()), "checks": checks, "pass": True}


def option_pair_matches(row: Mapping[str, Any]) -> bool:
    metadata = row.get("train_row_metadata")
    signatures = (
        metadata.get("selected_option_signatures")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(signatures, Mapping) or len(signatures) < 2:
        return False
    return sum(
        isinstance(signature, Mapping)
        and int(signature.get("type", -1)) == 10
        and int(signature.get("area", -1)) == 5
        for signature in signatures.values()
    ) >= 2


def hard_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["selection_margin"]),
        str(row["line_sha256"]),
        str(row["member"]),
        int(row["line_index"]),
    )


def retention_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["selection_margin"]),
        str(row["line_sha256"]),
        str(row["member"]),
        int(row["line_index"]),
    )


def unique_episode_select(
    rows: Sequence[Mapping[str, Any]],
    quota: int,
    used_lines: set[str],
    used_episodes: set[str],
    key: Any,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    for row in sorted(rows, key=key):
        line_sha = str(row["line_sha256"])
        episode_id = str(row["episode_id"])
        if line_sha in used_lines or episode_id in used_episodes:
            continue
        selected.append(row)
        used_lines.add(line_sha)
        used_episodes.add(episode_id)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise SelectionError(f"unique-episode capacity {len(selected)}/{quota}")
    return selected


def validate_v1(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = payload.get("rows")
    fields = payload.get("selection_sha256_payload_fields")
    if not isinstance(rows, list) or not isinstance(fields, list):
        raise SelectionError("v1 selection rows/fields malformed")
    compact = [{str(key): row[str(key)] for key in fields} for row in rows]
    checks = {
        "schema_exact": payload.get("schema_version") == V1_SELECTION_SCHEMA,
        "status_exact": payload.get("status") == "completed_frozen_train_only_selection",
        "base_exact_CW11": payload.get("base_model_state_sha256")
        == CW11_MODEL_SHA256,
        "payload_sha_field_exact": payload.get("selection_sha256")
        == V1_SELECTION_PAYLOAD_SHA256,
        "payload_sha_recomputed_exact": hashlib.sha256(canonical_json(compact)).hexdigest()
        == V1_SELECTION_PAYLOAD_SHA256,
        "rows_exact256": len(rows) == BATCH_SIZE,
        "terminal_checks_all_true": all(
            value is True for value in payload.get("terminal_checks", {}).values()
        ),
        "zero_nontrain_scope": payload.get("scope", {}).get(
            "validation_or_test_rows_opened"
        )
        == 0,
    }
    if not all(checks.values()):
        raise SelectionError(f"v1 contract drift: {checks}")
    return [dict(row) for row in rows], [str(key) for key in fields]


def rebuild_selection(
    profile: Mapping[str, Any],
    v1_rows: Sequence[Mapping[str, Any]],
    v1: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panels = profile["endpoint"]["profiles"]
    unchanged_rows = [row for row in v1_rows if row["stratum"] not in REBUILT_STRATA]
    globally_used = {str(row["line_sha256"]) for row in unchanged_rows}

    dominic_eligible = [
        row
        for row in panels["core5"]["near_wrong"]
        if v1.hard_eligible(row)
        and int(row.get("context", -1)) == 0
        and str(row.get("team_name", "")) == "Dominic Peel"
    ]
    dominic_special_pool = [row for row in dominic_eligible if option_pair_matches(row)]
    special_episode_count = len({str(row["episode_id"]) for row in dominic_special_pool})
    if len(dominic_special_pool) != DOMINIC_SPECIAL_QUOTA or special_episode_count != DOMINIC_SPECIAL_QUOTA:
        raise SelectionError("Dominic special pool is not exact 9 rows/episodes")
    dominic_lines = set(globally_used)
    dominic_episodes: set[str] = set()
    dominic_special = unique_episode_select(
        dominic_special_pool,
        DOMINIC_SPECIAL_QUOTA,
        dominic_lines,
        dominic_episodes,
        hard_key,
    )
    dominic_fill = unique_episode_select(
        dominic_eligible,
        DOMINIC_QUOTA - DOMINIC_SPECIAL_QUOTA,
        dominic_lines,
        dominic_episodes,
        hard_key,
    )
    dominic = [
        v1.compact_row("core5", DOMINIC, "hard", row)
        for row in dominic_special + dominic_fill
    ]

    flg_eligible = [
        row
        for row in panels["flg"]["fragile_correct"]
        if v1.retention_eligible(row)
        and int(row.get("context", -1)) == 3
    ]
    flg_lines = globally_used | {str(row["line_sha256"]) for row in dominic}
    flg_episode_ids: set[str] = set()
    flg_source = unique_episode_select(
        flg_eligible,
        FLG_CONTEXT3_QUOTA,
        flg_lines,
        flg_episode_ids,
        retention_key,
    )
    flg = [v1.compact_row("flg", FLG, "fragile", row) for row in flg_source]

    replacements = {DOMINIC: dominic, FLG: flg}
    selection: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for old_row in v1_rows:
        stratum = str(old_row["stratum"])
        if stratum in replacements:
            if stratum not in emitted:
                selection.extend(replacements[stratum])
                emitted.add(stratum)
            continue
        selection.append(dict(old_row))
    if emitted != set(REBUILT_STRATA):
        raise SelectionError("replacement strata missing from v1 ordering")
    for slot, row in enumerate(selection):
        row["final_b256_slot_zero_based"] = slot
    audit = {
        "Dominic": {
            "eligible_rows": len(dominic_eligible),
            "eligible_unique_episodes": len(
                {str(row["episode_id"]) for row in dominic_eligible}
            ),
            "forced_type10_area5_pair_rows": len(dominic_special),
            "forced_type10_area5_pair_unique_episodes": len(dominic_episodes & {
                str(row["episode_id"]) for row in dominic_special
            }),
            "selected_rows": len(dominic),
            "selected_unique_episodes": len(
                {str(row["episode_id"]) for row in dominic}
            ),
            "sort": "forced special9 by margin desc, then fill23 by margin desc",
        },
        "FLG": {
            "eligible_context3_rows": len(flg_eligible),
            "eligible_context3_unique_episodes": len(
                {str(row["episode_id"]) for row in flg_eligible}
            ),
            "selected_context3_rows": len(flg),
            "selected_unique_episodes": len(
                {str(row["episode_id"]) for row in flg}
            ),
            "sort": "selection margin asc",
        },
    }
    return selection, audit


def build_result() -> dict[str, Any]:
    runtime = validate_runtime()
    source = source_audit()
    profile_evidence = evidence(PROFILE, PROFILE_SHA256, PROFILE_MODE, "profile")
    v1_selection_evidence = evidence(
        V1_SELECTION, V1_SELECTION_SHA256, V1_SELECTION_MODE, "v1 selection"
    )
    v1, v1_selector_evidence = load_v1()
    profile = load_json_strict(PROFILE, "profile")
    v1_payload = load_json_strict(V1_SELECTION, "v1 selection")
    profile_checks = v1.validate_profile(profile)
    v1_rows, fields = validate_v1(v1_payload)
    selection, rebuild_audit = rebuild_selection(profile, v1_rows, v1)
    compact = [{key: row[key] for key in fields} for row in selection]
    selection_sha = hashlib.sha256(canonical_json(compact)).hexdigest()

    strata = Counter(str(row["stratum"]) for row in selection)
    categories = Counter(str(row["category"]) for row in selection)
    sources = Counter(str(row["source"]) for row in selection)
    unchanged_names = set(strata) - set(REBUILT_STRATA)
    checks = {
        "rows_exact256": len(selection) == BATCH_SIZE,
        "stratum_quotas_exact_v1": dict(sorted(strata.items()))
        == dict(sorted(v1.STRATUM_QUOTAS.items())),
        "category_counts_exact": dict(categories)
        == {"hard": 96, "fragile": 144, "c34": 16},
        "line_sha_global_unique": len({str(row["line_sha256"]) for row in selection})
        == BATCH_SIZE,
        "source_stratum_episode_unique": len(
            {
                (str(row["source"]), str(row["stratum"]), str(row["episode_id"]))
                for row in selection
            }
        )
        == BATCH_SIZE,
        "slots_exact": [int(row["final_b256_slot_zero_based"]) for row in selection]
        == list(range(BATCH_SIZE)),
        "all_train_members": all(
            str(row["member"]).startswith("train/")
            and str(row["member"]).endswith(".jsonl")
            and ".." not in str(row["member"]).split("/")
            for row in selection
        ),
        "hard_rows_baseline_wrong": all(
            row["ordered_correct"] is False and row["set_correct"] is False
            for row in selection
            if row["category"] == "hard"
        ),
        "retention_rows_baseline_correct": all(
            row["ordered_correct"] is True and row["set_correct"] is True
            for row in selection
            if row["category"] != "hard"
        ),
        "all_count_correct": all(row["count_correct"] is True for row in selection),
        "all_selection_margins_finite": all(
            isinstance(row["selection_margin"], (int, float))
            and math.isfinite(float(row["selection_margin"]))
            for row in selection
        ),
        "Dominic_special_exact_full9": sum(
            row["stratum"] == DOMINIC and option_pair_matches(row)
            for row in selection
        )
        == DOMINIC_SPECIAL_QUOTA,
        "Dominic_selected_exact32": strata[DOMINIC] == DOMINIC_QUOTA,
        "FLG_selected_exact16_all_context3": strata[FLG] == FLG_CONTEXT3_QUOTA
        and all(
            int(row["context"]) == 3
            for row in selection
            if row["stratum"] == FLG
        ),
        "other_eight_strata_line_identity_exact_v1": all(
            [str(row["line_sha256"]) for row in selection if row["stratum"] == name]
            == [str(row["line_sha256"]) for row in v1_rows if row["stratum"] == name]
            for name in unchanged_names
        ),
        "selection_changed_from_v1": selection_sha != V1_SELECTION_PAYLOAD_SHA256,
        "no_valid_fingerprint_or_identity_matching_used": True,
    }
    if not all(checks.values()):
        raise SelectionError(f"v2 terminal gate failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "completed_frozen_train_only_selection",
        "base_model_state_sha256": CW11_MODEL_SHA256,
        "profile_input": profile_evidence,
        "v1_selection_input": v1_selection_evidence,
        "v1_selector_input": v1_selector_evidence,
        "profile_contract_checks": profile_checks,
        "scope": {
            "selection_uses_train_profile_rows_only": True,
            "archive_members_opened": 0,
            "validation_or_test_rows_opened": 0,
            "historical_valid4_used_for_row_matching": False,
            "valid_identity_feature_logit_or_fingerprint_used": False,
            "changed_candidate_created": False,
            "official_evaluation_count": 0,
            "submission_performed": False,
        },
        "contract": {
            "batch_size": BATCH_SIZE,
            "stratum_quotas": v1.STRATUM_QUOTAS,
            "context34_source_quotas": v1.CONTEXT34_SOURCE_QUOTAS,
            "Dominic_special": "all 9 unique type10 area5 pair hard rows forced",
            "FLG_guard": "16 context3 retention rows",
            "hard_rank": "selection margin descending, then line SHA/member/index",
            "retention_rank": "selection margin ascending, then line SHA/member/index",
            "episode_uniqueness": (
                "one row per source/stratum/episode; cross-stratum episode reuse allowed"
            ),
            "unchanged_strata": "the other eight strata preserve v1 line identity and order",
        },
        "capacity_and_selection_audit": rebuild_audit,
        "terminal_checks": checks,
        "counts": {
            "strata": dict(sorted(strata.items())),
            "categories": dict(sorted(categories.items())),
            "sources": dict(sorted(sources.items())),
        },
        "selection_sha256": selection_sha,
        "selection_sha256_payload_fields": fields,
        "rows": selection,
        "tool": source,
        "runtime": runtime,
    }


def publish_o_excl(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise SelectionError("short selection write")
            written += count
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    observed = path.lstat()
    digest = sha256_file(path)
    checks = {
        "regular_single_link": stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and int(observed.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(observed.st_mode) == 0o444,
        "size_exact": int(observed.st_size) == len(payload),
        "sha_exact": digest == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise SelectionError(f"publication drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(observed.st_size),
        "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "nlink": int(observed.st_nlink),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    if args.mode == "static":
        if args.output is not None:
            raise SelectionError("static mode forbids output")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_passed",
                    "source": source_audit(),
                    "profile": evidence(PROFILE, PROFILE_SHA256, PROFILE_MODE, "profile"),
                    "v1_selector": evidence(
                        V1_SELECTOR, V1_SELECTOR_SHA256, V1_SELECTOR_MODE, "v1 selector"
                    ),
                    "v1_selection": evidence(
                        V1_SELECTION,
                        V1_SELECTION_SHA256,
                        V1_SELECTION_MODE,
                        "v1 selection",
                    ),
                    "writes": 0,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.output is None:
        raise SelectionError("run mode requires output")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise SelectionError("output must be direct artifacts child")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result = build_result()
    artifact = publish_o_excl(output, canonical_json(result))
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "selection_sha256": result["selection_sha256"],
                "counts": result["counts"],
                "output": artifact,
                "validation_or_test_rows_opened": 0,
                "changed_candidate_created": False,
                "submission_performed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
