#!/usr/bin/env python3
"""Freeze one targeted train-only B256 selection from the exact-CW11 profile."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "select_cw22_targeted_b256_from_cw11_profile_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-cw22-targeted-train-b256-selection-v1"
PROFILE = ROOT / "artifacts/u468_cw11_full_train_margin_profile_v1_20260803.json"
PROFILE_SHA256 = "da684858d2c459416234cdc3cae1ffc32a5ec23a1ec376f66a0cf440776a846e"
PROFILE_MODE = 0o444
PROFILE_SCHEMA = "ptcg-u468-exact-cw11-full-train-margin-profile-v1"
CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"

STRATUM_QUOTAS = {
    "pf_ctx0_hard": 32,
    "pf_ctx7_hard": 32,
    "dominic_ctx0_hard": 32,
    "pf_ctx0_retention": 32,
    "pf_ctx7_retention": 32,
    "dominic_ctx0_retention": 32,
    "broad_flg_retention": 16,
    "broad_pf_other_retention": 16,
    "broad_core_non_dominic_retention": 16,
    "broad_context34_retention": 16,
}
CONTEXT34_SOURCE_QUOTAS = {"flg": 5, "pokemonfan": 5, "core5": 6}
BATCH_SIZE = 256


class SelectionError(RuntimeError):
    """Fail-closed selection error."""


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


def evidence(path: Path, expected_sha: str, expected_mode: int) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
    ):
        raise SelectionError(f"unsafe input identity/mode: {path}")
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
        raise SelectionError(f"input evidence drift: {checks}")
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


def load_json_strict(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelectionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise SelectionError(f"nonfinite JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise SelectionError("profile root must be an object")
    return payload


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
        "no_training_or_submission_calls": not hits,
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


def finite_margin(row: Mapping[str, Any]) -> bool:
    value = row.get("selection_margin")
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def count_safe(row: Mapping[str, Any], allow_multi: bool = False) -> bool:
    expert = row.get("expert_order")
    predicted = row.get("predicted_order")
    if not isinstance(expert, list) or not isinstance(predicted, list) or not expert:
        return False
    if len(expert) != len(predicted) or (not allow_multi and len(expert) != 1):
        return False
    minimum = int(row.get("min_count", -1))
    maximum = int(row.get("max_count", -2))
    count = len(expert)
    if not (minimum <= count <= maximum):
        return False
    if minimum == maximum:
        return True
    value = row.get("count_margin")
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def hard_eligible(row: Mapping[str, Any]) -> bool:
    return (
        row.get("ordered_correct") is False
        and row.get("set_correct") is False
        and count_safe(row)
        and finite_margin(row)
        and float(row["selection_margin"]) <= 0.0
    )


def retention_eligible(row: Mapping[str, Any], allow_multi: bool = False) -> bool:
    return (
        row.get("ordered_correct") is True
        and row.get("set_correct") is True
        and count_safe(row, allow_multi=allow_multi)
        and finite_margin(row)
        and float(row["selection_margin"]) >= 0.0
    )


def compact_row(source: str, stratum: str, category: str, row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "member",
        "line_index",
        "line_sha256",
        "episode_id",
        "observation_step_index",
        "team_name",
        "context",
        "min_count",
        "max_count",
        "expert_order",
        "predicted_order",
        "set_correct",
        "ordered_correct",
        "selection_margin",
        "count_margin",
        "predicted_log_probability",
        "train_row_metadata",
    )
    result = {key: row.get(key) for key in keys}
    result.update(
        {
            "source": source,
            "stratum": stratum,
            "category": category,
            "expert_count": len(row["expert_order"]),
            "predicted_count": len(row["predicted_order"]),
            "count_correct": len(row["expert_order"]) == len(row["predicted_order"]),
            "profile_selection_margin_rank_basis": "exact_CW11_native_BF16",
        }
    )
    return result


def choose(
    source: str,
    stratum: str,
    category: str,
    rows: Sequence[Mapping[str, Any]],
    quota: int,
    predicate: Callable[[Mapping[str, Any]], bool],
    used_lines: set[str],
    used_episodes: set[tuple[str, str, str]],
    hard: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if predicate(row)]
    ordered = sorted(
        eligible,
        key=lambda row: (
            -float(row["selection_margin"]) if hard else float(row["selection_margin"]),
            str(row["line_sha256"]),
            str(row["member"]),
            int(row["line_index"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ordered:
        line_sha = str(row["line_sha256"])
        episode_key = (source, stratum, str(row["episode_id"]))
        if line_sha in used_lines or episode_key in used_episodes:
            continue
        selected.append(compact_row(source, stratum, category, row))
        used_lines.add(line_sha)
        used_episodes.add(episode_key)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise SelectionError(
            f"{stratum}: insufficient globally episode-unique capacity "
            f"({len(selected)}/{quota})"
        )
    return selected, {
        "eligible_rows_before_global_dedup": len(eligible),
        "eligible_unique_episodes_before_global_dedup": len(
            {str(row["episode_id"]) for row in eligible}
        ),
        "selected_rows": len(selected),
        "selected_unique_episodes": len({str(row["episode_id"]) for row in selected}),
        "selection_margin_min": min(float(row["selection_margin"]) for row in selected),
        "selection_margin_max": max(float(row["selection_margin"]) for row in selected),
        "sort": (
            "selection_margin_desc_then_line_sha_member_line"
            if hard
            else "selection_margin_asc_then_line_sha_member_line"
        ),
    }


def build_selection(profile: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint = profile["endpoint"]
    panels = endpoint["profiles"]
    used_lines: set[str] = set()
    used_episodes: set[tuple[str, str, str]] = set()
    selection: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}

    specs: list[tuple[str, str, str, Sequence[Mapping[str, Any]], Callable[[Mapping[str, Any]], bool], bool]] = [
        (
            "pokemonfan",
            "pf_ctx0_hard",
            "hard",
            panels["pokemonfan"]["near_wrong"],
            lambda row: hard_eligible(row) and int(row.get("context", -1)) == 0,
            True,
        ),
        (
            "pokemonfan",
            "pf_ctx7_hard",
            "hard",
            panels["pokemonfan"]["near_wrong"],
            lambda row: hard_eligible(row) and int(row.get("context", -1)) == 7,
            True,
        ),
        (
            "core5",
            "dominic_ctx0_hard",
            "hard",
            panels["core5"]["near_wrong"],
            lambda row: hard_eligible(row)
            and int(row.get("context", -1)) == 0
            and str(row.get("team_name", "")) == "Dominic Peel",
            True,
        ),
        (
            "pokemonfan",
            "pf_ctx0_retention",
            "fragile",
            panels["pokemonfan"]["fragile_correct"],
            lambda row: retention_eligible(row) and int(row.get("context", -1)) == 0,
            False,
        ),
        (
            "pokemonfan",
            "pf_ctx7_retention",
            "fragile",
            panels["pokemonfan"]["fragile_correct"],
            lambda row: retention_eligible(row) and int(row.get("context", -1)) == 7,
            False,
        ),
        (
            "core5",
            "dominic_ctx0_retention",
            "fragile",
            panels["core5"]["fragile_correct"],
            lambda row: retention_eligible(row)
            and int(row.get("context", -1)) == 0
            and str(row.get("team_name", "")) == "Dominic Peel",
            False,
        ),
        (
            "flg",
            "broad_flg_retention",
            "fragile",
            panels["flg"]["fragile_correct"],
            lambda row: retention_eligible(row) and int(row.get("context", -1)) != 34,
            False,
        ),
        (
            "pokemonfan",
            "broad_pf_other_retention",
            "fragile",
            panels["pokemonfan"]["fragile_correct"],
            lambda row: retention_eligible(row)
            and int(row.get("context", -1)) not in {0, 7, 34},
            False,
        ),
        (
            "core5",
            "broad_core_non_dominic_retention",
            "fragile",
            panels["core5"]["fragile_correct"],
            lambda row: retention_eligible(row)
            and int(row.get("context", -1)) != 34
            and str(row.get("team_name", "")) != "Dominic Peel",
            False,
        ),
    ]
    for source, stratum, category, rows, predicate, hard in specs:
        chosen, audit = choose(
            source,
            stratum,
            category,
            rows,
            STRATUM_QUOTAS[stratum],
            predicate,
            used_lines,
            used_episodes,
            hard,
        )
        selection.extend(chosen)
        audits[stratum] = audit

    c34_selected: list[dict[str, Any]] = []
    c34_audits: dict[str, Any] = {}
    for source in ("flg", "pokemonfan", "core5"):
        chosen, audit = choose(
            source,
            "broad_context34_retention",
            "c34",
            panels[source]["fragile_correct"],
            CONTEXT34_SOURCE_QUOTAS[source],
            lambda row: retention_eligible(row, allow_multi=True)
            and int(row.get("context", -1)) == 34,
            used_lines,
            used_episodes,
            False,
        )
        c34_selected.extend(chosen)
        c34_audits[source] = audit
    if len(c34_selected) != STRATUM_QUOTAS["broad_context34_retention"]:
        raise SelectionError("context34 aggregate quota drift")
    selection.extend(c34_selected)
    audits["broad_context34_retention"] = {
        "source_quotas": CONTEXT34_SOURCE_QUOTAS,
        "per_source": c34_audits,
        "selected_rows": len(c34_selected),
    }
    return selection, audits


def validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "schema_exact": profile.get("schema_version") == PROFILE_SCHEMA,
        "status_exact": profile.get("status")
        == "completed_exact_CW11_train_only_profile",
        "base_model_exact_CW11": profile.get("base", {}).get("model_state_sha256")
        == CW11_MODEL_SHA256,
        "profile_split_train": profile.get("split") == "train",
        "profile_archive_nontrain_false": profile.get("validation_opened_for_profile")
        is False,
        "historical_valid_replay_disclosed": profile.get(
            "historical_exact_CW11_replay", {}
        ).get("validation_replayed_for_historical_reconstruction")
        is True,
        "not_end_to_end_validation_free": profile.get("end_to_end_validation_free")
        is False,
        "profile_restore_pass": profile.get("endpoint", {}).get("restore_pass") is True,
        "target_capacity_pass": profile.get("endpoint", {})
        .get("targeted_CW22_capacity", {})
        .get("pass")
        is True,
        "no_changed_candidate": profile.get(
            "official_unique_changed_candidate_count_consumed"
        )
        == 0,
        "no_submission": profile.get("submission_performed") is False,
    }
    if not all(checks.values()):
        raise SelectionError(f"profile contract drift: {checks}")
    return checks


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
        raise SelectionError(f"published selection drift: {checks}")
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


def build_result() -> dict[str, Any]:
    profile_evidence = evidence(PROFILE, PROFILE_SHA256, PROFILE_MODE)
    profile = load_json_strict(PROFILE)
    profile_checks = validate_profile(profile)
    selection, audits = build_selection(profile)
    stratum_counts = Counter(str(row["stratum"]) for row in selection)
    category_counts = Counter(str(row["category"]) for row in selection)
    source_counts = Counter(str(row["source"]) for row in selection)
    selection_payload = [
        {
            key: row[key]
            for key in (
                "source",
                "stratum",
                "category",
                "member",
                "line_index",
                "line_sha256",
                "episode_id",
                "observation_step_index",
                "team_name",
                "context",
                "min_count",
                "max_count",
                "expert_order",
                "predicted_order",
                "selection_margin",
                "count_margin",
            )
        }
        for row in selection
    ]
    selection_sha = hashlib.sha256(canonical_json(selection_payload)).hexdigest()
    checks = {
        "rows_exact256": len(selection) == BATCH_SIZE,
        "stratum_quotas_exact": dict(sorted(stratum_counts.items()))
        == dict(sorted(STRATUM_QUOTAS.items())),
        "category_counts_exact": dict(category_counts)
        == {"hard": 96, "fragile": 144, "c34": 16},
        "line_sha_global_unique": len({row["line_sha256"] for row in selection})
        == BATCH_SIZE,
        "source_stratum_episode_unique": len(
            {(row["source"], row["stratum"], row["episode_id"]) for row in selection}
        )
        == BATCH_SIZE,
        "all_members_train": all(
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
            if row["category"] in {"fragile", "c34"}
        ),
        "all_count_correct": all(row["count_correct"] is True for row in selection),
        "all_selection_margins_finite": all(
            isinstance(row["selection_margin"], (int, float))
            and math.isfinite(float(row["selection_margin"]))
            for row in selection
        ),
        "no_valid_fingerprint_or_identity_matching_used": True,
    }
    if not all(checks.values()):
        raise SelectionError(f"terminal selection gate failed: {checks}")
    for slot, row in enumerate(selection):
        row["final_b256_slot_zero_based"] = slot
    return {
        "schema_version": SCHEMA,
        "status": "completed_frozen_train_only_selection",
        "base_model_state_sha256": CW11_MODEL_SHA256,
        "profile_input": profile_evidence,
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
            "stratum_quotas": STRATUM_QUOTAS,
            "context34_source_quotas": CONTEXT34_SOURCE_QUOTAS,
            "hard_rank": "selection_margin descending, then line SHA/member/index",
            "retention_rank": "selection_margin ascending, then line SHA/member/index",
            "episode_uniqueness": (
                "one row per source/stratum/episode; cross-stratum episode reuse allowed"
            ),
            "line_sha_uniqueness": "global across the entire B256",
            "hard_objective": "expert versus exact-CW11 wrong decision proxy",
            "retention_role": "train-only no-harm guard",
        },
        "capacity_and_selection_audit": audits,
        "terminal_checks": checks,
        "counts": {
            "strata": dict(sorted(stratum_counts.items())),
            "categories": dict(sorted(category_counts.items())),
            "sources": dict(sorted(source_counts.items())),
        },
        "selection_sha256": selection_sha,
        "selection_sha256_payload_fields": list(selection_payload[0]),
        "rows": selection,
        "tool": source_audit(),
        "runtime": validate_runtime(),
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
            raise SelectionError("static mode forbids --output")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "static_audit_passed",
                    "source": source_audit(),
                    "profile": evidence(PROFILE, PROFILE_SHA256, PROFILE_MODE),
                    "writes": 0,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return
    if args.output is None:
        raise SelectionError("run mode requires --output")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise SelectionError("output must be a direct child of artifacts/")
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
                "changed_candidate_created": False,
                "validation_or_test_rows_opened": 0,
                "submission_performed": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
