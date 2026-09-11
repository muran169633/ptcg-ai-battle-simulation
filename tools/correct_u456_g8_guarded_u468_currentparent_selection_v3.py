#!/usr/bin/env python3
"""Selection-only count-safe correction for the frozen guarded-U468 profile."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "correct_u456_g8_guarded_u468_currentparent_selection_v3.py"
SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-train-profile-selection-v3"
SOURCE_PROFILE = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143_v2.json"
SOURCE_PROFILE_SHA256 = "ad25c407d4c08e53fccf0c883ea0b09d224bf1ad25e4fe7dd9031196b50dd5ce"
CORRECTION_V4 = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_boundaryqp_design202608143.master_correction_v4.json"
CORRECTION_V4_SHA256 = "060fd238feba40908cb90fee81c81d4e4a981dfade274faf416233deae886857"
V1_ENGINE = TOOLS / "profile_u456_g8_guarded_u468_currentparent_selection_v1.py"
V1_ENGINE_SHA256 = "a398749d606d706d4622cfdfcdd36b7e6ad0c717a8d346af9f524554c18fa869"
ATTEMPT = ROOT / ".ptcg-u456-g8-guarded-u468-currentparent-selection-correction-v3-attempt.json"
OUTPUT = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143_v3.json"
TARGET_MARGIN = 1.0 / 512.0


class ProtocolError(RuntimeError):
    """Fail-closed selection correction error."""


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_regular(
    path: Path, expected_sha256: str | None, mode: int, label: str
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or stat.S_IMODE(before.st_mode) != mode
        ):
            raise ProtocolError(f"{label}: unsafe type/link/mode")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    visible = os.lstat(path)
    identity = lambda item: (
        int(item.st_dev), int(item.st_ino), int(item.st_size), int(item.st_mtime_ns)
    )
    if (
        identity(before) != identity(after)
        or identity(after) != identity(visible)
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or int(visible.st_nlink) != 1
    ):
        raise ProtocolError(f"{label}: identity changed during read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if len(payload) != int(after.st_size) or (
        expected_sha256 is not None and digest != expected_sha256
    ):
        raise ProtocolError(f"{label}: SHA/size drift")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "held_fd_identity_exact": True,
    }


def publish(path: Path, payload: bytes) -> dict[str, Any]:
    if path not in {ATTEMPT, OUTPUT} or path.parent.resolve() not in {
        ROOT.resolve(), (ROOT / "artifacts").resolve()
    }:
        raise ProtocolError("publication target outside exact allowlist")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(fd, payload[offset:])
            if count <= 0:
                raise ProtocolError("short O_EXCL publication")
            offset += count
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    reloaded, evidence = read_regular(path, sha256_bytes(payload), 0o444, f"published {path.name}")
    if reloaded != payload:
        raise ProtocolError("published payload mismatch")
    return evidence


def load_engine() -> tuple[ModuleType, dict[str, Any]]:
    source, evidence = read_regular(V1_ENGINE, V1_ENGINE_SHA256, 0o555, "frozen selection engine")
    spec = importlib.util.spec_from_file_location("selection_v3_frozen_engine", V1_ENGINE)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot import frozen selection engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise ProtocolError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ProtocolError("requires exact Python -I -B")


def source_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_imports = {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
    forbidden_calls = {"save", "backward", "step", "unlink", "rename", "replace", "rmtree"}
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            imports.extend(name for name in names if name.split(".", 1)[0] in forbidden_imports)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden_calls:
                calls.append(node.func.attr)
    checks = {
        "ast_parse": True,
        "no_network_submission_import": not imports,
        "no_training_checkpoint_or_destructive_call": not calls,
        "single_output": source.count(str(OUTPUT.relative_to(ROOT)).encode()) == 1,
        "single_attempt": source.count(str(ATTEMPT.relative_to(ROOT)).encode()) == 1,
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}")
    return {"checks": checks, "forbidden_imports": imports, "forbidden_calls": calls}


def count_safe(row: Mapping[str, Any], *, allow_multi: bool) -> bool:
    expert = row.get("expert_order")
    predicted = row.get("predicted_order")
    if (
        not isinstance(expert, list)
        or not isinstance(predicted, list)
        or not expert
        or len(expert) != len(predicted)
        or (not allow_multi and len(expert) != 1)
    ):
        return False
    count = len(expert)
    minimum = int(row.get("min_count", -1))
    maximum = int(row.get("max_count", -2))
    if not minimum <= count <= maximum:
        return False
    if minimum == maximum:
        return True
    margin = row.get("count_margin")
    return isinstance(margin, (int, float)) and math.isfinite(float(margin)) and float(margin) > 0.0


def select_targets(all_rows: Sequence[Mapping[str, Any]], engine: ModuleType) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for context, quota in ((0, 3), (7, 2)):
        pool = [
            row for row in all_rows
            if row.get("source") == "pokemonfan"
            and int(row.get("context", -1)) == context
            and row.get("ordered_correct") is False
            and row.get("set_correct") is False
            and count_safe(row, allow_multi=False)
            and isinstance(row.get("selection_margin"), (int, float))
            and math.isfinite(float(row["selection_margin"]))
            and float(row["selection_margin"]) < 0.0
        ]
        ranked = sorted(
            pool,
            key=lambda row: (
                TARGET_MARGIN - float(row["selection_margin"]),
                str(row["line_sha256"]),
            ),
        )
        if len(ranked) < quota:
            raise ProtocolError(f"context {context}: insufficient count-safe targets")
        for rank, row in enumerate(ranked[:quota], start=1):
            item = engine.row_copy(row)
            item.update(
                {
                    "role": "target",
                    "target_context_quota": context,
                    "target_rank_within_context": rank,
                    "required_linear_margin_change": TARGET_MARGIN - float(row["selection_margin"]),
                    "endpoint_margin_minimum": TARGET_MARGIN,
                    "count_safe_semantics": "fixed_count_or_flexible_positive_parent_count_margin",
                }
            )
            selected.append(item)
    if len(selected) != 5 or len({row["line_sha256"] for row in selected}) != 5:
        raise ProtocolError("corrected target cardinality drift")
    return selected


def build_b256(
    all_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    guards: Sequence[Mapping[str, Any]],
    engine: ModuleType,
) -> list[dict[str, Any]]:
    forced_by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in targets:
        stratum = engine.classify_stratum(row, hard=True)
        if stratum is None:
            raise ProtocolError("corrected target has no hard stratum")
        forced_by_stratum[stratum].append(row)
    for row in guards:
        stratum = engine.classify_stratum(row, hard=False)
        if stratum is None:
            raise ProtocolError("guard has no retention stratum")
        forced_by_stratum[stratum].append(row)
    result: list[dict[str, Any]] = []
    for stratum in engine.STRATUM_ORDER:
        hard = stratum.endswith("_hard")
        pool = [
            row for row in all_rows
            if engine.classify_stratum(row, hard=hard) == stratum
            and isinstance(row.get("selection_margin"), (int, float))
            and math.isfinite(float(row["selection_margin"]))
            and (
                not hard
                or (
                    isinstance(row.get("decision_margin"), (int, float))
                    and math.isfinite(float(row["decision_margin"]))
                )
            )
            and (
                (
                    row.get("ordered_correct") is False
                    and row.get("set_correct") is False
                    and count_safe(row, allow_multi=False)
                    and float(row["selection_margin"]) <= 0.0
                )
                if hard
                else (
                    row.get("ordered_correct") is True
                    and row.get("set_correct") is True
                    and count_safe(row, allow_multi=True)
                    and float(row["selection_margin"]) >= 0.0
                )
            )
        ]
        key: Callable[[Mapping[str, Any]], Any] = (
            (lambda row: (-float(row["decision_margin"]), str(row["line_sha256"])))
            if hard
            else (lambda row: (float(row["selection_margin"]), str(row["line_sha256"])))
        )
        chosen = engine.select_with_episode_cap(
            pool,
            engine.STRATUM_QUOTAS[stratum],
            forced_by_stratum.get(stratum, []),
            key,
            stratum,
        )
        for row in chosen:
            row["stratum"] = stratum
            row["category"] = "hard" if hard else ("c34" if stratum == "broad_context34_retention" else "fragile")
            row["profile_selection_margin_rank_basis"] = (
                "count_safe_singleton_descending_decision_margin_then_line_sha"
                if hard
                else "count_safe_ascending_selection_margin_then_line_sha"
            )
        result.extend(chosen)
    if len(result) != 256 or len({row["line_sha256"] for row in result}) != 256:
        raise ProtocolError("corrected B256 size/uniqueness drift")
    forced = {row["line_sha256"] for row in (*targets, *guards)}
    if not forced.issubset({row["line_sha256"] for row in result}):
        raise ProtocolError("corrected B256 lost a target or guard")
    for slot, row in enumerate(result):
        row["final_b256_slot_zero_based"] = slot
    if dict(Counter(row["stratum"] for row in result)) != engine.STRATUM_QUOTAS:
        raise ProtocolError("corrected B256 quota drift")
    return result


def build_result(
    source_profile: Mapping[str, Any],
    profile_evidence: Mapping[str, Any],
    correction_evidence: Mapping[str, Any],
    engine: ModuleType,
    engine_evidence: Mapping[str, Any],
    self_evidence: Mapping[str, Any],
    attempt_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        source_profile.get("schema_version")
        != "ptcg-u456-g8-guarded-u468-currentparent-train-profile-selection-v2"
        or source_profile.get("status")
        != "completed_frozen_parent_only_train_profile_selection"
        or source_profile.get("base", {}).get("model_state_sha256")
        != engine.PARENT_MODEL_SHA256
        or source_profile.get("scope_audit", {}).get("changed_candidates_constructed_or_evaluated") != 0
    ):
        raise ProtocolError("source profile identity/scope drift")
    profiles = source_profile.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(engine.PANEL_ORDER):
        raise ProtocolError("source profile panels drift")
    all_rows = [
        row
        for panel in engine.PANEL_ORDER
        for list_name in ("near_wrong", "fragile_correct")
        for row in profiles[panel][list_name]
    ]
    if len(all_rows) != sum(engine.EXPECTED_ROWS.values()):
        raise ProtocolError("source profile retained-row count drift")
    targets = select_targets(all_rows, engine)
    guards = engine.select_guards(all_rows)
    b256 = build_b256(all_rows, targets, guards, engine)
    target_sha = [row["line_sha256"] for row in targets]
    expected_target_sha = [
        "d1d1aba7ae2c6949e30c6a4b6d1292cc9459f867b550e3b9b71b969fd74c0f54",
        "498a7f841b66e44756c04e93312d683c979c201d8eec7713823876e3cdc3fda7",
        "5994ba14fd56d323743e2ba297621bb1e3385dde20116b5ae33f8ad9c7ffbb36",
        "2ed31b529dee80b049e76f753b2c4c5b2f33cc32a9f5bdace7566da616ca2cac",
        "6f1591a836cfa2858da431140641f6ee01db6684b4fc6a1a81caea003c5cfbd4",
    ]
    contracts = {
        "target_count_exact5": len(targets) == 5,
        "guard_count_exact14": len(guards) == 14,
        "b256_count_exact256": len(b256) == 256,
        "corrected_target_identity_exact": target_sha == expected_target_sha,
        "corrected_context7_margin_exact": [row["selection_margin"] for row in targets if row["context"] == 7] == [-0.001953125, -0.00390625],
        "corrected_context7_count_margin_exact": [row["count_margin"] for row in targets if row["context"] == 7] == [8.4609375, 11.015625],
        "all_targets_count_safe": all(count_safe(row, allow_multi=False) for row in targets),
        "all_target_guard_rows_in_b256": {row["line_sha256"] for row in (*targets, *guards)}.issubset({row["line_sha256"] for row in b256}),
        "all_rows_unique": len({row["line_sha256"] for row in b256}) == 256,
        "stratum_quotas_exact": dict(Counter(row["stratum"] for row in b256)) == engine.STRATUM_QUOTAS,
        "source_profile_not_mutated_or_rerun": True,
        "zero_changed_candidates_constructed_or_evaluated": True,
    }
    if not all(contracts.values()):
        raise ProtocolError(f"selection correction terminal gate failed: {contracts}")
    result = dict(source_profile)
    result["schema_version"] = SCHEMA
    result["status"] = "completed_frozen_parent_only_train_profile_selection"
    result["selection"] = {
        "status": "completed_count_safe_singleton_selection_correction_v3",
        "target_margin": TARGET_MARGIN,
        "targets": targets,
        "guards": guards,
        "b256_rows": b256,
        "b256_rows_canonical_sha256": sha256_bytes(canonical_json(b256)),
        "contracts": contracts,
    }
    result["selection_correction_v3"] = {
        "source_profile": dict(profile_evidence),
        "master_correction_v4": dict(correction_evidence),
        "frozen_selection_engine": dict(engine_evidence),
        "tool": dict(self_evidence),
        "attempt_marker": dict(attempt_evidence),
        "profile_rows_recomputed": 0,
        "archive_members_opened": 0,
        "cuda_accessed": False,
        "changed_candidates_constructed_or_evaluated": 0,
        "only_change": "count-safe singleton target and B256 hard-row selection",
    }
    result["scope_audit"] = {
        **result["scope_audit"],
        "selection_correction_archive_members_opened": 0,
        "selection_correction_cuda_accessed": False,
        "selection_correction_changed_candidates": 0,
    }
    canonical_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "run"), required=True)
    parser.add_argument("--expected-tool-sha256", required=True)
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular(SCRIPT, None, 0o555, "selection correction tool")
    if args.expected_tool_sha256 != self_evidence["sha256"]:
        raise ProtocolError("selection correction tool SHA drift")
    audit = source_audit(source)
    profile_bytes, profile_evidence = read_regular(
        SOURCE_PROFILE, SOURCE_PROFILE_SHA256, 0o444, "frozen parent profile v2"
    )
    correction_bytes, correction_evidence = read_regular(
        CORRECTION_V4, CORRECTION_V4_SHA256, 0o444, "master correction v4"
    )
    correction = json.loads(correction_bytes)
    if (
        correction.get("status")
        != "locked_after_parent_profile_before_any_changed_candidate_construction"
        or correction.get("scope", {}).get("selection_only_correction_authorized") is not True
    ):
        raise ProtocolError("master correction v4 does not authorize selection correction")
    engine, engine_evidence = load_engine()
    absent = not (ATTEMPT.exists() or ATTEMPT.is_symlink() or OUTPUT.exists() or OUTPUT.is_symlink())
    if args.mode == "audit":
        if not absent:
            raise ProtocolError("selection correction targets already exist")
        print(canonical_json({
            "schema_version": f"{SCHEMA}-audit-v1",
            "status": "STATIC_AUDIT_PASS",
            "tool": self_evidence,
            "source_audit": audit,
            "source_profile": profile_evidence,
            "master_correction_v4": correction_evidence,
            "frozen_engine": engine_evidence,
            "archive_members_opened": 0,
            "cuda_accessed": False,
            "changed_candidates": 0,
            "writes": 0,
            "targets_absent": True,
        }).decode("utf-8"), end="")
        return
    if not absent:
        raise ProtocolError("selection correction one-shot targets already exist")
    marker = publish(ATTEMPT, canonical_json({
        "schema_version": f"{SCHEMA}-attempt-v1",
        "attempt": 1,
        "tool_sha256": self_evidence["sha256"],
        "source_profile_sha256": SOURCE_PROFILE_SHA256,
        "master_correction_v4_sha256": CORRECTION_V4_SHA256,
        "changed_candidate_constructed_or_evaluated": False,
        "retry_authorized": False,
    }))
    profile = json.loads(profile_bytes)
    result = build_result(
        profile,
        profile_evidence,
        correction_evidence,
        engine,
        engine_evidence,
        self_evidence,
        marker,
    )
    output = publish(OUTPUT, canonical_json(result))
    print(canonical_json({
        "status": result["status"],
        "output": output,
        "selection": {
            "targets": len(result["selection"]["targets"]),
            "guards": len(result["selection"]["guards"]),
            "b256": len(result["selection"]["b256_rows"]),
            "b256_sha256": result["selection"]["b256_rows_canonical_sha256"],
        },
    }).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
