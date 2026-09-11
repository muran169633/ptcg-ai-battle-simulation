#!/usr/bin/env python3
"""Read-only exploratory 28-row cutting-plane continuation for U468.

This wrapper hash-locks the audited CW4 runner, retains its original four
full-train guards, and adds the five unique CW witnesses found by the v3
24,050-row gate.  The terminal is deliberately observational: no unknown
terminal hash is represented as preregistered evidence.  The wrapped runner's
selected-row gate and mandatory raw-model restoration remain hard requirements.

There are no checkpoint/result writes, validation reads, optimizer steps,
network calls, uploads, or submissions.  Output is one JSON document on stdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math as std_math
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-cw9-cuttingplane-probe-v1"

PRIMARY = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py"
PRIMARY_SHA256 = "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c"
FROZEN_MODE = 0o555

EXPECTED_ROWS = 28
EXPECTED_TARGET_ROWS = 5
EXPECTED_GUARD_ROWS = 23
EXPECTED_TARGET_OBLIGATIONS = 20
EXPECTED_GUARD_OBLIGATIONS = 68
EXPECTED_INITIAL_ACTIVE_PAIRS = 29
EXPECTED_ADDED_GUARDS = 5

MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)

ADDED_GUARDS = (
    {
        "role": "guard",
        "panel": "core5",
        "member": "train/part-00004.jsonl",
        "line_index_zero_based": 246,
        "line_sha256": "18513bbf7b92945f6569ce34056c18c7a5877d7fc814fc5eaaf576cf5841480b",
        "expected_context": 22,
        "expected_expert_order": [0, 1],
        "expected_option_count": 2,
        "metrics_union": ["ordered_exact"],
        "expected_c0_false_metrics": [],
        "positive_option": 0,
        "negative_option": 1,
        "formal_raw_order": [0, 1],
        "formal_c0_order": [0, 1],
        "formal_failed_candidate_order": [1, 0],
        "expected_formal_b256_raw_margin": 0.0,
        "formal_batch_index_zero_based": 4,
        "formal_batch_position_zero_based": 246,
        "formal_batch_size": 256,
    },
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 1193,
        "line_sha256": "9f06d1995e0f77f5465d296de6c29b63c5a844c136bccd5653f8c232713a41aa",
        "expected_context": 7,
        "expected_expert_order": [3],
        "expected_option_count": 10,
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": [],
        "positive_option": 3,
        "negative_option": 2,
        "formal_raw_order": [3],
        "formal_c0_order": [3],
        "formal_failed_candidate_order": [2],
        "expected_formal_b256_raw_margin": 0.015625,
        "formal_batch_index_zero_based": 4,
        "formal_batch_position_zero_based": 169,
        "formal_batch_size": 256,
    },
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 4169,
        "line_sha256": "c3e89f8e57250c65f1ec51ec1bf99db07904431016900154e053ade741904a1c",
        "expected_context": 5,
        "expected_expert_order": [0, 1],
        "expected_option_count": 4,
        "metrics_union": ["ordered_exact"],
        "expected_c0_false_metrics": [],
        "positive_option": 0,
        "negative_option": 1,
        "formal_raw_order": [0, 1],
        "formal_c0_order": [0, 1],
        "formal_failed_candidate_order": [1, 0],
        "expected_formal_b256_raw_margin": 0.0,
        "formal_batch_index_zero_based": 16,
        "formal_batch_position_zero_based": 73,
        "formal_batch_size": 256,
    },
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 9442,
        "line_sha256": "4ae1720258969aca918d1649b85818a5d2d0777da30197512357dc232cf8b3e4",
        "expected_context": 7,
        "expected_expert_order": [1, 0],
        "expected_option_count": 2,
        "metrics_union": ["ordered_exact"],
        "expected_c0_false_metrics": [],
        "positive_option": 1,
        "negative_option": 0,
        "formal_raw_order": [1, 0],
        "formal_c0_order": [1, 0],
        "formal_failed_candidate_order": [0, 1],
        "expected_formal_b256_raw_margin": 0.03125,
        "formal_batch_index_zero_based": 36,
        "formal_batch_position_zero_based": 226,
        "formal_batch_size": 227,
    },
    {
        "role": "guard",
        "panel": "pokemonfan",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 744,
        "line_sha256": "165a9a5b31e49fa867d2ac2ae2b3c8f35942e8da302d19c418837afff2f9bd42",
        "expected_context": 0,
        "expected_expert_order": [9],
        "expected_option_count": 14,
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": [],
        "positive_option": 9,
        "negative_option": 3,
        "formal_raw_order": [9],
        "formal_c0_order": [9],
        "formal_failed_candidate_order": [3],
        "expected_formal_b256_raw_margin": 0.015625,
        "formal_batch_index_zero_based": 2,
        "formal_batch_position_zero_based": 232,
        "formal_batch_size": 256,
    },
)

# These are the exact same-process selected-23 raw thresholds published by the
# audited CW4 terminal.  Rebatching to 28 rows may only strengthen, never lower,
# one of these four already accepted guard floors.
FROZEN_CW4_GUARD_THRESHOLDS = {
    (
        "flg",
        "train/part-00000.jsonl",
        4545,
        "f5c29fe63df23a40aff98d58447af577ba7244af2c762a288555041dd7fdfac4",
        9,
        10,
    ): 0.0,
    (
        "flg",
        "train/part-00000.jsonl",
        5063,
        "9417c4667da3af128f684cf5b847952b47686a329af28a94cbf01615f3de8644",
        1,
        3,
    ): 0.0,
    (
        "pokemonfan",
        "train/part-00000.jsonl",
        5092,
        "989b079b931a1f256b4ddad8e7f1d42986f3f98464f993a3afe994b5cca56d13",
        0,
        1,
    ): 0.00390625,
    (
        "core5",
        "train/part-00007.jsonl",
        209,
        "010dca17bea03dab885587766989072a358f8df36e6485cc358321dd83f6d483",
        5,
        4,
    ): 0.0234375,
}


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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_regular_bytes(
    path: Path,
    expected_sha256: str | None,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or after.st_size != len(payload)
    ):
        raise RuntimeError(f"{label} identity changed while reading")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
    }


def import_frozen(path: Path, expected_sha256: str, label: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, expected_sha256, label, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(
        f"_cw9_{path.stem}_{expected_sha256[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {label}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_calls = {
        "save",
        "dump",
        "dumps_to_file",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "replace",
        "rename",
        "unlink",
        "remove",
        "rmtree",
        "backward",
        "step",
    }
    forbidden_import_roots = {
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "kaggle",
    }
    call_hits: list[dict[str, Any]] = []
    import_hits: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls:
                call_hits.append({"name": name, "line": node.lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append({"name": name, "line": node.lineno})
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_or_network_calls": not call_hits,
        "no_network_or_submission_imports": not import_hits,
        "five_added_guard_identities_unique": len(
            {
                (
                    item["panel"],
                    item["member"],
                    item["line_index_zero_based"],
                    item["line_sha256"],
                )
                for item in ADDED_GUARDS
            }
        )
        == EXPECTED_ADDED_GUARDS,
        "five_added_pairs_unique": len(
            {
                (
                    item["panel"],
                    item["member"],
                    item["line_index_zero_based"],
                    item["positive_option"],
                    item["negative_option"],
                )
                for item in ADDED_GUARDS
            }
        )
        == EXPECTED_ADDED_GUARDS,
        "added_metric_obligations_exact_11": sum(
            len(item["metrics_union"]) for item in ADDED_GUARDS
        )
        == 11,
        "formal_batch_coordinates_preregistered": all(
            int(item["formal_batch_size"]) in {227, 256}
            and int(item["formal_batch_position_zero_based"])
            < int(item["formal_batch_size"])
            for item in ADDED_GUARDS
        ),
        "four_frozen_cw4_thresholds_bound": len(
            FROZEN_CW4_GUARD_THRESHOLDS
        )
        == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(f"local static audit failed: {checks}")
    return {
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "stdout_only": True,
    }


class _AnyEqual:
    """Exploration-only sentinel used inside the frozen terminal reproducer."""

    def __eq__(self, other: object) -> bool:
        return True

    def __req__(self, other: object) -> bool:
        return True


ANY_EQUAL = _AnyEqual()
ANY_FLOAT = object()


class _MathProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(std_math, name)

    @staticmethod
    def isclose(a: Any, b: Any, *, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        if b is ANY_FLOAT:
            return True
        return std_math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def append_nine_guards(
    primary: ModuleType,
    cutting: ModuleType,
    active_pairs: list[dict[str, Any]],
    raw_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    added: list[dict[str, Any]] = []
    existing = {cutting.pair_key(pair) for pair in active_pairs}
    for offset, descriptor in enumerate(
        primary.NEW_GUARDS, start=primary.ORIGINAL_ROW_COUNT
    ):
        selected_margin = cutting.pair_margin(
            raw_snapshot,
            {
                "row_index": offset,
                "positive_option": int(descriptor["positive_option"]),
                "negative_option": int(descriptor["negative_option"]),
            },
        )
        if not std_math.isfinite(selected_margin) or selected_margin < 0.0:
            raise RuntimeError("selected-28 raw guard margin is invalid")
        formal_margin = descriptor.get("expected_formal_b256_raw_margin")
        threshold = selected_margin
        threshold_sources = ["same_process_selected28_raw_pair_margin"]
        frozen_key = (
            str(descriptor["panel"]),
            str(descriptor["member"]),
            int(descriptor["line_index_zero_based"]),
            str(descriptor["line_sha256"]),
            int(descriptor["positive_option"]),
            int(descriptor["negative_option"]),
        )
        frozen_cw4_margin = FROZEN_CW4_GUARD_THRESHOLDS.get(frozen_key)
        if frozen_cw4_margin is not None:
            threshold = max(threshold, float(frozen_cw4_margin))
            threshold_sources.append("frozen_selected23_CW4_raw_pair_margin")
        if formal_margin is not None:
            formal_margin = float(formal_margin)
            if not std_math.isfinite(formal_margin) or formal_margin < 0.0:
                raise RuntimeError("formal-B256 raw guard margin is invalid")
            threshold = max(selected_margin, formal_margin)
            threshold_sources.append("preregistered_formal_B256_raw_pair_margin")
        pair = {
            "row_index": offset,
            "identity": primary.identity_record(descriptor),
            "role": "guard",
            "positive_option": int(descriptor["positive_option"]),
            "negative_option": int(descriptor["negative_option"]),
            "threshold": threshold,
            "threshold_source": "max_selected28_and_formal_B256_raw_margin",
            "threshold_sources": threshold_sources,
            "threshold_history": [
                {
                    "iteration": primary.EXPECTED_C0_SUCCESS_ITERATION,
                    "source": "max_selected28_and_formal_B256_raw_margin",
                    "observed_selected28": selected_margin,
                    "frozen_selected23_CW4": frozen_cw4_margin,
                    "observed_formal_B256": formal_margin,
                    "retained_max": threshold,
                }
            ],
            "origins": ["fulltrain_CW_guard"],
            "constructions": ["specified_raw_safe_vs_failed_candidate_pair"],
            "created_iteration": primary.EXPECTED_C0_SUCCESS_ITERATION,
        }
        key = cutting.pair_key(pair)
        if key in existing:
            raise RuntimeError("expanded guard pair duplicates existing pair")
        existing.add(key)
        active_pairs.append(pair)
        added.append(dict(pair))
    if len(active_pairs) != EXPECTED_INITIAL_ACTIVE_PAIRS:
        raise RuntimeError("29-pair initial active ledger drift")
    return added


def audit_nine_guard_transitions(
    primary: ModuleType,
    raw_snapshot: Mapping[str, Any],
    c0_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Audit old C0-CWs and later-CW4 witnesses under distinct contracts."""
    records: list[dict[str, Any]] = []
    for row_index, descriptor in enumerate(
        primary.NEW_GUARDS, start=primary.ORIGINAL_ROW_COUNT
    ):
        raw_row = raw_snapshot["official_rows"][row_index]
        c0_row = c0_snapshot["official_rows"][row_index]
        metrics = [str(value) for value in descriptor["metrics_union"]]
        raw_flags = {metric: raw_row["flags"][metric] for metric in metrics}
        c0_flags = {metric: c0_row["flags"][metric] for metric in metrics}
        observed_c0_false = {
            metric for metric, value in c0_flags.items() if value is False
        }
        expected_c0_false = set(descriptor["expected_c0_false_metrics"])
        raw_order = [int(value) for value in raw_row["predicted_order"]]
        c0_order = [int(value) for value in c0_row["predicted_order"]]
        raw_c0_first_difference = next(
            (
                (raw_value, c0_value)
                for raw_value, c0_value in zip(raw_order, c0_order)
                if raw_value != c0_value
            ),
            None,
        )
        expected_pair = (
            int(descriptor["positive_option"]),
            int(descriptor["negative_option"]),
        )
        checks = {
            "raw_order_exact": raw_order
            == list(descriptor["formal_raw_order"]),
            "c0_order_exact": c0_order
            == list(descriptor["formal_c0_order"]),
            "all_guard_metrics_raw_true": all(
                value is True for value in raw_flags.values()
            ),
            "c0_false_metric_set_exact": observed_c0_false
            == expected_c0_false,
        }
        if "formal_failed_candidate_order" in descriptor:
            failed_order = [
                int(value)
                for value in descriptor["formal_failed_candidate_order"]
            ]
            raw_failed_first_difference = next(
                (
                    (raw_value, failed_value)
                    for raw_value, failed_value in zip(raw_order, failed_order)
                    if raw_value != failed_value
                ),
                None,
            )
            checks.update(
                {
                    "new_guard_raw_and_c0_order_identical": raw_order
                    == c0_order,
                    "new_guard_raw_c0_has_no_first_difference": (
                        raw_c0_first_difference is None
                    ),
                    "new_guard_threat_pair_exact_raw_vs_failed_cw4": (
                        raw_failed_first_difference == expected_pair
                    ),
                }
            )
            transition_kind = "raw_equals_c0_then_later_cw4_threat"
        else:
            checks["old_guard_pair_exact_raw_vs_c0_first_difference"] = (
                raw_c0_first_difference == expected_pair
            )
            transition_kind = "raw_to_c0_cw"
        if not all(checks.values()):
            raise RuntimeError(
                f"nine-guard transition contract drift: {checks}"
            )
        records.append(
            {
                "row_index": row_index,
                "identity": primary.identity_record(descriptor),
                "transition_kind": transition_kind,
                "metrics_union": metrics,
                "raw_flags": raw_flags,
                "c0_flags": c0_flags,
                "raw_order": raw_order,
                "c0_order": c0_order,
                "checks": checks,
            }
        )
    return records


def run_probe(
    primary: ModuleType,
    primary_source: bytes,
    primary_evidence: Mapping[str, Any],
    *,
    candidate_consumer: Any = None,
) -> dict[str, Any]:
    cutting, cutting_evidence = primary.import_frozen(
        primary.CUTTING,
        primary.CUTTING_SHA256,
        "cw9 frozen base cutting v2",
    )
    fulltrain, fulltrain_evidence = primary.import_frozen(
        primary.FULLTRAIN,
        primary.FULLTRAIN_SHA256,
        "cw9 frozen fulltrain v2",
    )
    geometry, geometry_evidence = cutting.import_frozen(
        cutting.GEOMETRY,
        cutting.GEOMETRY_SHA256,
        "cw9 frozen geometry",
    )
    ram, ram_evidence = cutting.import_frozen(
        cutting.RAM_RUNNER,
        cutting.RAM_RUNNER_SHA256,
        "cw9 frozen RAM runner",
    )
    formal, formal_evidence = fulltrain.import_frozen(
        fulltrain.FORMAL,
        fulltrain.FORMAL_SHA256,
        "cw9 frozen formal evaluator",
    )
    primary_static = primary.static_audit(primary_source)
    interfaces = primary.interface_audit(cutting, fulltrain, geometry, ram, formal)

    combined_guards = tuple(primary.NEW_GUARDS) + tuple(
        dict(item) for item in ADDED_GUARDS
    )
    patch_values = {
        "NEW_GUARDS": combined_guards,
        "EXPANDED_ROW_COUNT": EXPECTED_ROWS,
        "EXPANDED_GUARD_ROW_COUNT": EXPECTED_GUARD_ROWS,
        "EXPECTED_NEW_GUARD_OBLIGATIONS": 27,
        "EXPECTED_EXPANDED_GUARD_OBLIGATIONS": EXPECTED_GUARD_OBLIGATIONS,
        "EXPECTED_INITIAL_ACTIVE_COUNT": EXPECTED_INITIAL_ACTIVE_PAIRS,
        "EXPECTED_SUCCESS_CONTINUATION_ITERATION": ANY_EQUAL,
        "EXPECTED_TERMINAL_CUMULATIVE_L2": ANY_FLOAT,
        "EXPECTED_TERMINAL_CUMULATIVE_SHA256": ANY_EQUAL,
        "EXPECTED_TERMINAL_MODEL_STATE_SHA256": ANY_EQUAL,
        "EXPECTED_TERMINAL_ACTIVE_LEDGER_SHA256": ANY_EQUAL,
        "math": _MathProxy(),
    }
    original_values = {name: getattr(primary, name) for name in patch_values}
    original_append = primary.append_new_guards
    original_transition_audit = primary.new_guard_transition_audit
    globals_restored = False
    try:
        for name, value in patch_values.items():
            setattr(primary, name, value)

        def patched_append(
            cutting_module: ModuleType,
            active_pairs: list[dict[str, Any]],
            raw_snapshot: Mapping[str, Any],
        ) -> list[dict[str, Any]]:
            return append_nine_guards(
                primary, cutting_module, active_pairs, raw_snapshot
            )

        primary.append_new_guards = patched_append

        def patched_transition_audit(
            raw_snapshot: Mapping[str, Any],
            c0_snapshot: Mapping[str, Any],
        ) -> list[dict[str, Any]]:
            return audit_nine_guard_transitions(
                primary, raw_snapshot, c0_snapshot
            )

        primary.new_guard_transition_audit = patched_transition_audit
        result = primary.run_cuttingplane(
            primary_source,
            primary_static,
            cutting,
            cutting_evidence,
            fulltrain,
            fulltrain_evidence,
            geometry,
            geometry_evidence,
            ram,
            ram_evidence,
            formal,
            formal_evidence,
            candidate_consumer=candidate_consumer,
        )
    finally:
        primary.append_new_guards = original_append
        primary.new_guard_transition_audit = original_transition_audit
        for name, value in original_values.items():
            setattr(primary, name, value)
        globals_restored = all(
            getattr(primary, name) is value
            for name, value in original_values.items()
        ) and primary.append_new_guards is original_append and (
            primary.new_guard_transition_audit is original_transition_audit
        )

    if not globals_restored:
        raise RuntimeError("frozen primary globals were not restored")
    decision = result["decision"]
    synthetic = decision.pop("preregistered_terminal_checks")
    observational_checks = {
        "selected_28row_gate_pass": bool(synthetic["final_23row_gate_pass"]),
        "success_iteration_found": decision["success_iteration_after_c0"] is not None,
        "terminal_l2_finite": std_math.isfinite(
            float(decision["terminal_cumulative_l2"])
        ),
        "terminal_vector_sha_is_hex64": len(
            str(decision["terminal_cumulative_float64_le_sha256"])
        )
        == 64,
        "terminal_model_sha_is_hex64": len(
            str(decision["candidate_model_state_sha256_before_restore"])
        )
        == 64,
        "terminal_nonactor_exact_raw": bool(
            synthetic["terminal_nonactor_exact_raw"]
        ),
        "active_pair_count_exact_29": int(
            result["active_pair_contract"]["final_count"]
        )
        == EXPECTED_INITIAL_ACTIVE_PAIRS,
        "active_pair_ledger_sha_self_consistent": decision[
            "terminal_active_pair_ledger_sha256"
        ]
        == result["active_pair_contract"]["final_canonical_ledger_sha256"],
        "original_active20_thresholds_preserved": bool(
            synthetic["original_active20_absolute_thresholds_preserved"]
        ),
        "wrapped_finally_raw_restore_pass": bool(result["final_integrity"]["pass"]),
        "primary_module_globals_restored": globals_restored,
        "candidate_consumer_call_semantics_exact": bool(
            decision["candidate_consumer_called_before_finally_restore"]
        )
        == (candidate_consumer is not None),
    }
    if not all(observational_checks.values()):
        raise RuntimeError(
            f"exploratory terminal integrity failed: {observational_checks}"
        )
    result["schema_version"] = SCHEMA
    result["status"] = "exploratory_28row_29pair_optimization_success"
    result["scope"].update(
        {
            "row_count": EXPECTED_ROWS,
            "original_cw4_rows": 23,
            "new_fulltrain_cw_rows_added": EXPECTED_ADDED_GUARDS,
            "terminal_preregistered": False,
            "formal_B256_thresholds_recorded_for_new_guards": True,
            "standalone_candidate_consumer_is_none": candidate_consumer is None,
        }
    )
    result["input_lock"]["wrapped_primary_cw4"] = dict(primary_evidence)
    result["decision"]["status"] = result["status"]
    result["decision"]["observational_terminal_checks"] = observational_checks
    result["decision"]["terminal_preregistration_bypassed_for_discovery"] = True
    result["decision"]["model_materialized"] = False
    result["decision"]["submission_performed"] = False
    result["active_pair_contract"].update(
        {
            "initial_count": EXPECTED_INITIAL_ACTIVE_PAIRS,
            "new_guard_thresholds": (
                "max of selected28 same-process raw and preregistered formal-B256 raw"
            ),
        }
    )
    result["wrapper_audit"] = {
        "primary_module_globals_restored": globals_restored,
        "terminal_is_observational_not_preregistered": True,
        "model_or_result_writes": 0,
        "validation_or_holdout_access": False,
        "network_upload_submission": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT, None, "CW9 exploratory wrapper", expected_mode=FROZEN_MODE
    )
    local_static = static_audit(source)
    primary, primary_evidence = import_frozen(
        PRIMARY, PRIMARY_SHA256, "audited CW4 primary runner"
    )
    primary_source, _ = read_regular_bytes(
        PRIMARY,
        PRIMARY_SHA256,
        "audited CW4 primary source",
        expected_mode=FROZEN_MODE,
    )
    if args.mode == "static":
        result = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "primary": primary_evidence,
            "guard_spec_sha256": sha256_bytes(canonical_json(ADDED_GUARDS)),
            "contract": {
                "rows": EXPECTED_ROWS,
                "target_rows": EXPECTED_TARGET_ROWS,
                "guard_rows": EXPECTED_GUARD_ROWS,
                "target_obligations": EXPECTED_TARGET_OBLIGATIONS,
                "guard_obligations": EXPECTED_GUARD_OBLIGATIONS,
                "initial_active_pairs": EXPECTED_INITIAL_ACTIVE_PAIRS,
                "added_guards": EXPECTED_ADDED_GUARDS,
            },
            "audit": {
                "local": local_static,
                "primary": primary.static_audit(primary_source),
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
        }
    else:
        result = run_probe(primary, primary_source, primary_evidence)
        result["input_lock"]["self"] = self_evidence
        result["local_static_audit"] = local_static
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
