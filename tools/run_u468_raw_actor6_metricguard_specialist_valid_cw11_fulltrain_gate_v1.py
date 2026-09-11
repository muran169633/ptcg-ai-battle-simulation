#!/usr/bin/env python3
"""Formal full-train gate for the frozen specialist-valid CW11 terminal.

This read-only adapter reconstructs the exact CW11 terminal in RAM and reuses
the audited v5 -> v4 -> v3 formal evaluator over raw/candidate x
FLG/PokemonFan/core5.  It binds the raw U468 and E904 anchors plus the CW11
iteration/vector/model/ledger identity before the candidate consumer can run.

The only compatibility work is translating CW11 result-field names and the
33-row/34-pair context into labels expected by the older audited evaluator.
All numerical, hash, 2x3, 24,050-row, policy, count, and value gates remain
unchanged.  Candidate state is CPU-cloned in RAM and never materialized.

CW11 has already consumed four frozen specialist-valid guards for optimization,
so those rows are explicitly not promotion evidence.  This formal evaluator
opens no additional validation, broad, or Gold data.  There are no optimizer,
backward, checkpoint/result writes, network calls, uploads, or submissions.
Output is JSON on stdout only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
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
SCRIPT = TOOLS / (
    "run_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_fulltrain_gate_v1.py"
)
SCHEMA = (
    "ptcg-u468-raw-actor6-metricguard-specialist-valid-"
    "cw11-fulltrain-gate-v1"
)
FROZEN_MODE = 0o555

CW11 = TOOLS / (
    "probe_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_cuttingplane_v1.py"
)
CW11_SHA256 = "23bc74022210942115eaf339a38e710e88ad81ba75d62237e8eb9e2c11e074ee"
FULLTRAIN_V5 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v5.py"
FULLTRAIN_V5_SHA256 = (
    "90ab439a0ca75e7f0a2cf2c8e7c117a07ee5ff9b8b6462ffc5e8380114b404a0"
)

RAW_CHECKPOINT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_CHECKPOINT_FILE_SHA256 = (
    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
)
EXPECTED_RAW_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_RAW_NONACTOR_SHA256 = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)

E904_CHECKPOINT = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_fulltrain_"
    "cw10_materialized_v1_20260802/"
    "u468-cw10-fulltrain-pass-eval-only.pt"
)
E904_CHECKPOINT_FILE_SHA256 = (
    "91b64ddf754b149297dd6177038de871fdc69bcea0d86787cff851e4fd79e0dd"
)
EXPECTED_E904_MODEL_STATE_SHA256 = (
    "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
)

EXPECTED_SUCCESS_ITERATION_AFTER_CW10 = 3
EXPECTED_TERMINAL_CUMULATIVE_L2 = 0.00792176975336988
EXPECTED_TERMINAL_CUMULATIVE_SHA256 = (
    "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
)
EXPECTED_TERMINAL_MODEL_STATE_SHA256 = (
    "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
)
EXPECTED_ACTIVE_PAIR_COUNT = 34
EXPECTED_ACTIVE_PAIR_LEDGER_SHA256 = (
    "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"
)
EXPECTED_SELECTED_ROW_COUNT = 33
CW11_STATUS = "exploratory_33row_specialist_valid_CW11_success"
LEGACY_V4_EXPECTED_STATUS = "exploratory_28row_29pair_optimization_success"

EXPECTED_PANEL_ORDER = ("flg", "pokemonfan", "core5")
EXPECTED_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_TOTAL_ROWS = 24050
EXPECTED_BATCH_SIZE = 256
EXPECTED_EVALUATIONS = 6
EXPECTED_MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
EXPECTED_SAFETY_METRICS = EXPECTED_MAIN_METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
EXPECTED_PF_MINIMUM_WC = {
    "set_exact": 3,
    "hybrid_order_exact": 3,
    "ordered_exact": 5,
    "top1_correct": 3,
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
        raise RuntimeError(f"{label} changed while reading")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
    }


def import_frozen(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path,
        expected_sha256,
        label,
        expected_mode=FROZEN_MODE,
    )
    spec = importlib.util.spec_from_file_location(
        f"_cw11_fulltrain_{path.stem}_{expected_sha256[:12]}", path
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


def static_audit(
    source: bytes,
    cw11: ModuleType,
    v5: ModuleType,
    v4: ModuleType,
    base: ModuleType,
) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_calls = {
        "save",
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

    parameters = inspect.signature(cw11.run_probe).parameters
    terminal_hashes = (
        EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
    )
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_or_network_calls": not call_hits,
        "no_network_or_submission_imports": not import_hits,
        "cw11_consumer_present_and_default_none": (
            "candidate_consumer" in parameters
            and parameters["candidate_consumer"].default is None
        ),
        "v5_run_gate_callable": callable(v5.run_gate),
        "v4_run_gate_callable": callable(v4.run_gate),
        "v4_capture_callable": callable(v4.capture_candidate),
        "base_decide_fulltrain_callable": callable(base.decide_fulltrain),
        "cw11_script_identity_exact": str(cw11.SCRIPT) == str(CW11),
        "cw11_raw_model_anchor_exact": cw11.EXPECTED_RAW_MODEL_STATE_SHA256
        == EXPECTED_RAW_MODEL_STATE_SHA256,
        "cw11_raw_nonactor_anchor_exact": cw11.EXPECTED_RAW_NONACTOR_SHA256
        == EXPECTED_RAW_NONACTOR_SHA256,
        "cw11_e904_model_anchor_exact": cw11.EXPECTED_CW10_MODEL_STATE_SHA256
        == EXPECTED_E904_MODEL_STATE_SHA256,
        "cw11_rows_exact_33": int(cw11.EXPECTED_ROW_COUNT)
        == EXPECTED_SELECTED_ROW_COUNT,
        "cw11_initial_pairs_exact_34": int(cw11.EXPECTED_INITIAL_ACTIVE_COUNT)
        == EXPECTED_ACTIVE_PAIR_COUNT,
        "terminal_hashes_hex64": all(len(value) == 64 for value in terminal_hashes),
        "success_iteration_exact_3": EXPECTED_SUCCESS_ITERATION_AFTER_CW10 == 3,
        "selected_row_count_exact_33": EXPECTED_SELECTED_ROW_COUNT == 33,
        "active_pair_count_exact_34": EXPECTED_ACTIVE_PAIR_COUNT == 34,
        "panel_order_exact": tuple(base.PANEL_ORDER) == EXPECTED_PANEL_ORDER,
        "panel_rows_exact_9443_9487_5120": dict(base.EXPECTED_ROWS)
        == EXPECTED_ROWS,
        "total_unique_train_rows_exact_24050": (
            int(base.EXPECTED_TOTAL_ROWS) == EXPECTED_TOTAL_ROWS
            and sum(EXPECTED_ROWS.values()) == EXPECTED_TOTAL_ROWS
        ),
        "batch_size_exact_256": int(base.EXPECTED_BATCH_SIZE)
        == EXPECTED_BATCH_SIZE,
        "evaluation_count_exact_6": int(base.EXPECTED_EVALUATIONS)
        == EXPECTED_EVALUATIONS,
        "main_metrics_exact": tuple(base.MAIN_METRICS)
        == EXPECTED_MAIN_METRICS,
        "six_policy_safety_metrics_exact": tuple(base.SAFETY_METRICS)
        == EXPECTED_SAFETY_METRICS,
        "pokemonfan_thresholds_exact_3_3_5_3": dict(base.PF_MINIMUM_WC)
        == EXPECTED_PF_MINIMUM_WC,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW11 formal fulltrain static audit failed: {checks}")
    return {
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "candidate_consumer_RAM_only": True,
        "count_value_fingerprints_must_equal_raw": True,
        "specialist_valid_consumed_for_optimization": True,
        "promotion_evidence": False,
        "formal_evaluator_opens_no_additional_validation_broad_gold": True,
        "stdout_only": True,
    }


def _pop_rename(mapping: dict[str, Any], old: str, new: str) -> None:
    if old not in mapping:
        raise RuntimeError(f"legacy compatibility label missing: {old}")
    if new in mapping:
        raise RuntimeError(f"compatibility target label already exists: {new}")
    mapping[new] = mapping.pop(old)


def relabel_legacy_audit_fields(result: dict[str, Any]) -> None:
    frozen = result["probe_frozen_reproduction_checks"]
    _pop_rename(
        frozen,
        "probe_success_iteration_exact_6",
        "cw11_success_iteration_after_CW10_exact_3",
    )

    capture_checks = result["candidate_capture"]["checks"]
    _pop_rename(
        capture_checks,
        "success_iteration_exact_6",
        "cw11_success_iteration_after_CW10_exact_3",
    )
    _pop_rename(
        capture_checks,
        "actual_expanded_row_count_exact_29",
        "actual_expanded_row_count_exact_33",
    )
    _pop_rename(
        capture_checks,
        "active_pair_count_exact_29",
        "active_pair_count_exact_34",
    )

    scope = result["scope_audit"]
    if not bool(scope.pop("selected_29_train_rows_then_all_24050_train_rows")):
        raise RuntimeError("trusted v5 selected/fulltrain scope gate failed")
    scope["selected_33_rows_then_all_24050_unique_train_rows"] = True
    scope["selected_rows_include_four_consumed_specialist_valid_guards"] = True
    scope["specialist_valid_consumed_for_optimization"] = True
    scope["promotion_evidence"] = False
    if bool(scope.pop("validation_members_opened")):
        raise RuntimeError("formal evaluator unexpectedly opened additional validation")
    if bool(scope.pop("validation_results_read")):
        raise RuntimeError("formal evaluator unexpectedly read additional validation results")
    scope["additional_validation_members_opened_by_formal_evaluator"] = False
    scope["additional_validation_results_read_by_formal_evaluator"] = False
    scope["additional_validation_broad_gold_opened_by_formal_evaluator"] = False

    execution_checks = result["execution"]["checks"]
    _pop_rename(
        execution_checks,
        "validation_members_not_opened",
        "no_additional_validation_members_opened_by_formal_evaluator",
    )

    adapter = result.pop("v5_compatibility_adapter_audit")
    raw = adapter["raw"]
    _pop_rename(
        raw,
        "actual_selected_row_count_exact_29_before_capture_adapter",
        "actual_selected_row_count_exact_33_before_capture_adapter",
    )
    adapter_checks = adapter["checks"]
    _pop_rename(
        adapter_checks,
        "actual_selected_row_count_exact_29",
        "actual_selected_row_count_exact_33",
    )
    _pop_rename(
        adapter_checks,
        "active_pair_count_exact_30",
        "active_pair_count_exact_34",
    )
    result["trusted_v5_numeric_adapter_audit"] = adapter


def run_gate(
    source: bytes,
    static: Mapping[str, Any],
    cw11: ModuleType,
    cw11_evidence: Mapping[str, Any],
    v5: ModuleType,
    v5_evidence: Mapping[str, Any],
    v4: ModuleType,
    v4_evidence: Mapping[str, Any],
    base: ModuleType,
    base_evidence: Mapping[str, Any],
    raw_checkpoint_evidence: Mapping[str, Any],
    e904_checkpoint_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    original_cw11_run = cw11.run_probe
    patch_values = {
        "SCRIPT": SCRIPT,
        "SCHEMA": SCHEMA,
        "EXPECTED_SUCCESS_ITERATION": EXPECTED_SUCCESS_ITERATION_AFTER_CW10,
        "EXPECTED_TERMINAL_CUMULATIVE_L2": EXPECTED_TERMINAL_CUMULATIVE_L2,
        "EXPECTED_TERMINAL_CUMULATIVE_SHA256": EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "EXPECTED_TERMINAL_MODEL_STATE_SHA256": EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "EXPECTED_ACTIVE_PAIR_COUNT": EXPECTED_ACTIVE_PAIR_COUNT,
        "EXPECTED_ACTIVE_PAIR_LEDGER_SHA256": EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
        "EXPECTED_SELECTED_ROW_COUNT": EXPECTED_SELECTED_ROW_COUNT,
        "CW10_STATUS": CW11_STATUS,
        "LEGACY_V4_EXPECTED_STATUS": LEGACY_V4_EXPECTED_STATUS,
    }
    original_v5_values = {name: getattr(v5, name) for name in patch_values}
    normalization_audit: dict[str, Any] = {
        "cw11_run_calls": 0,
        "actual_contract_verified_before_aliasing": False,
    }
    restored = False
    try:
        for name, value in patch_values.items():
            setattr(v5, name, value)

        def normalized_cw11_run(
            primary: ModuleType,
            primary_source: bytes,
            primary_evidence: Mapping[str, Any],
            *,
            candidate_consumer: Any = None,
        ) -> dict[str, Any]:
            normalization_audit["cw11_run_calls"] += 1
            observed = original_cw11_run(
                primary,
                primary_source,
                primary_evidence,
                candidate_consumer=candidate_consumer,
            )
            decision = observed.get("second_stage", {}).get("decision", {})
            active = observed.get("second_stage", {}).get(
                "active_pair_contract", {}
            )
            actual_checks = {
                "status_exact": observed.get("status") == CW11_STATUS,
                "success_iteration_after_CW10_exact_3": int(
                    decision.get("success_iteration_after_CW10", -1)
                )
                == EXPECTED_SUCCESS_ITERATION_AFTER_CW10,
                "terminal_l2_exact": math.isclose(
                    float(decision.get("terminal_cumulative_l2", -1.0)),
                    EXPECTED_TERMINAL_CUMULATIVE_L2,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "terminal_vector_sha_exact": decision.get(
                    "terminal_cumulative_float64_le_sha256"
                )
                == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
                "terminal_model_sha_exact": decision.get(
                    "candidate_model_state_sha256_before_CW10_finally_restore"
                )
                == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
                "terminal_nonactor_exact_raw": decision.get(
                    "terminal_nonactor_sha256"
                )
                == EXPECTED_RAW_NONACTOR_SHA256,
                "active_pair_count_exact_34": int(
                    active.get("final_count", -1)
                )
                == EXPECTED_ACTIVE_PAIR_COUNT,
                "active_pair_ledger_sha_exact": active.get(
                    "final_canonical_ledger_sha256"
                )
                == EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
                "candidate_consumer_called": bool(
                    decision.get("candidate_consumer_called")
                ),
                "selected_33row_gate_pass": bool(
                    decision.get("terminal_checks", {}).get(
                        "selected_33row_gate_pass"
                    )
                ),
                "outer_finally_raw_restore_pass": bool(
                    observed.get("final_integrity", {}).get("pass")
                ),
            }
            if not all(actual_checks.values()):
                raise RuntimeError(
                    f"CW11 terminal drift before compatibility aliases: {actual_checks}"
                )
            normalization_audit["actual_contract_verified_before_aliasing"] = True
            normalization_audit["actual_contract_checks"] = actual_checks

            decision["success_iteration_after_c0"] = decision[
                "success_iteration_after_CW10"
            ]
            decision["candidate_model_state_sha256_before_restore"] = decision[
                "candidate_model_state_sha256_before_CW10_finally_restore"
            ]
            decision["candidate_consumer_called_before_finally_restore"] = decision[
                "candidate_consumer_called"
            ]
            observed["decision"] = decision
            observed["active_pair_contract"] = active
            return observed

        cw11.run_probe = normalized_cw11_run
        result = v5.run_gate(
            source,
            static,
            cw11,
            cw11_evidence,
            v4,
            v4_evidence,
            base,
            base_evidence,
        )
    finally:
        cw11.run_probe = original_cw11_run
        for name, value in original_v5_values.items():
            setattr(v5, name, value)
        restored = (
            cw11.run_probe is original_cw11_run
            and all(
                getattr(v5, name) is value
                for name, value in original_v5_values.items()
            )
        )

    if int(normalization_audit["cw11_run_calls"]) != 1:
        raise RuntimeError("CW11 formal adapter did not run exactly once")
    if not normalization_audit["actual_contract_verified_before_aliasing"]:
        raise RuntimeError("CW11 actual contract was not verified")
    if not restored:
        raise RuntimeError("CW11/v5 compatibility globals were not restored")

    relabel_legacy_audit_fields(result)
    probe = result["probe_result"]
    second_decision = probe["second_stage"]["decision"]
    final_checks = {
        "cw11_status_exact": probe.get("status") == CW11_STATUS,
        "terminal_model_sha_exact": result["candidate_capture"][
            "candidate_model_state_sha256"
        ]
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "terminal_vector_sha_exact": result["candidate_capture"][
            "terminal_cumulative_float64_le_sha256"
        ]
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "terminal_l2_exact": math.isclose(
            float(result["candidate_capture"]["terminal_cumulative_l2"]),
            EXPECTED_TERMINAL_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "terminal_nonactor_exact_raw": second_decision[
            "terminal_nonactor_sha256"
        ]
        == EXPECTED_RAW_NONACTOR_SHA256,
        "active_pair_count_exact_34": int(
            result["candidate_capture"]["active_pair_count"]
        )
        == EXPECTED_ACTIVE_PAIR_COUNT,
        "active_pair_ledger_sha_exact": result["candidate_capture"][
            "active_pair_ledger_sha256"
        ]
        == EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
        "all_compatibility_state_restored": restored,
        "six_evaluations": int(result["execution"]["evaluation_count_exact"])
        == EXPECTED_EVALUATIONS,
        "all_24050_unique_train_rows": bool(
            result["execution"]["checks"]["total_unique_train_rows_24050"]
        ),
        "evaluator_finally_raw_restore_pass": bool(
            result["execution"]["fulltrain_evaluator_finally_raw_restore"]
            ["finally_raw_restore_pass"]
        ),
    }
    if not all(final_checks.values()):
        raise RuntimeError(f"CW11 formal fulltrain final integrity failed: {final_checks}")

    input_lock = result["input_lock"]
    input_lock["self"] = {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": sha256_bytes(source),
    }
    input_lock["cw11_probe"] = input_lock.pop("cw10_probe")
    input_lock["fulltrain_v5_adapter"] = dict(v5_evidence)
    input_lock["raw_u468_checkpoint"] = dict(raw_checkpoint_evidence)
    input_lock["e904_eval_only_anchor"] = dict(e904_checkpoint_evidence)
    result["schema_version"] = SCHEMA
    result["cw11_terminal_contract"] = {
        "success_iteration_after_CW10": EXPECTED_SUCCESS_ITERATION_AFTER_CW10,
        "cumulative_l2": EXPECTED_TERMINAL_CUMULATIVE_L2,
        "cumulative_sha256": EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "model_state_sha256": EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "nonactor_sha256": EXPECTED_RAW_NONACTOR_SHA256,
        "active_pair_count": EXPECTED_ACTIVE_PAIR_COUNT,
        "active_pair_ledger_sha256": EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
        "selected_row_count": EXPECTED_SELECTED_ROW_COUNT,
    }
    result["cw11_normalization_audit"] = {
        **normalization_audit,
        "compatibility_aliases_only": True,
        "no_numeric_or_hash_gate_relaxed": True,
        "all_dependency_functions_and_globals_restored": restored,
        "final_checks": final_checks,
    }
    result["local_static_audit"] = dict(static)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()

    source, self_evidence = read_regular_bytes(
        SCRIPT,
        None,
        "CW11 formal fulltrain gate v1",
        expected_mode=FROZEN_MODE,
    )
    cw11, cw11_evidence = import_frozen(
        CW11,
        CW11_SHA256,
        "frozen CW11 probe",
    )
    v5, v5_evidence = import_frozen(
        FULLTRAIN_V5,
        FULLTRAIN_V5_SHA256,
        "frozen fulltrain gate v5 adapter",
    )
    v4, v4_evidence = import_frozen(
        v5.V4,
        v5.V4_SHA256,
        "frozen fulltrain gate v4 implementation",
    )
    base, base_evidence = import_frozen(
        v4.BASE_GATE,
        v4.BASE_GATE_SHA256,
        "frozen fulltrain gate v3 utilities",
    )
    _, raw_checkpoint_evidence = read_regular_bytes(
        RAW_CHECKPOINT,
        RAW_CHECKPOINT_FILE_SHA256,
        "raw U468 checkpoint anchor",
        expected_mode=0o664,
    )
    _, e904_checkpoint_evidence = read_regular_bytes(
        E904_CHECKPOINT,
        E904_CHECKPOINT_FILE_SHA256,
        "E904 eval-only checkpoint anchor",
        expected_mode=0o444,
    )
    local_static = static_audit(source, cw11, v5, v4, base)
    interfaces = v4.interface_audit(cw11, base)

    if args.mode == "static":
        cw11_source, _ = read_regular_bytes(
            CW11,
            CW11_SHA256,
            "static CW11 source",
            expected_mode=FROZEN_MODE,
        )
        v5_source, _ = read_regular_bytes(
            FULLTRAIN_V5,
            FULLTRAIN_V5_SHA256,
            "static fulltrain v5 source",
            expected_mode=FROZEN_MODE,
        )
        v4_source, _ = read_regular_bytes(
            v5.V4,
            v5.V4_SHA256,
            "static fulltrain v4 source",
            expected_mode=FROZEN_MODE,
        )
        base_source, _ = read_regular_bytes(
            v4.BASE_GATE,
            v4.BASE_GATE_SHA256,
            "static fulltrain v3 source",
            expected_mode=FROZEN_MODE,
        )
        cw10, _ = cw11.import_frozen(
            cw11.CW10,
            cw11.CW10_SHA256,
            "static frozen CW10 dependency",
        )
        result = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {
                "cw11_probe": cw11_evidence,
                "fulltrain_v5_adapter": v5_evidence,
                "fulltrain_v4_implementation": v4_evidence,
                "fulltrain_v3_utilities": base_evidence,
                "raw_u468_checkpoint": raw_checkpoint_evidence,
                "e904_eval_only_anchor": e904_checkpoint_evidence,
            },
            "interface_audit": interfaces,
            "terminal_contract": {
                "success_iteration_after_CW10": (
                    EXPECTED_SUCCESS_ITERATION_AFTER_CW10
                ),
                "cumulative_l2": EXPECTED_TERMINAL_CUMULATIVE_L2,
                "cumulative_sha256": EXPECTED_TERMINAL_CUMULATIVE_SHA256,
                "model_state_sha256": EXPECTED_TERMINAL_MODEL_STATE_SHA256,
                "raw_nonactor_sha256": EXPECTED_RAW_NONACTOR_SHA256,
                "active_pair_count": EXPECTED_ACTIVE_PAIR_COUNT,
                "active_pair_ledger_sha256": EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
                "selected_row_count": EXPECTED_SELECTED_ROW_COUNT,
            },
            "formal_gate_contract": {
                "models": ["raw", "candidate"],
                "panels": list(EXPECTED_PANEL_ORDER),
                "evaluation_count": EXPECTED_EVALUATIONS,
                "panel_rows": dict(EXPECTED_ROWS),
                "total_unique_train_rows": EXPECTED_TOTAL_ROWS,
                "batch_size": EXPECTED_BATCH_SIZE,
                "pokemonfan_minimum_wc_and_net": dict(EXPECTED_PF_MINIMUM_WC),
                "all_three_panels_six_policy_metrics_zero_cw": True,
                "count_value_logits_fingerprints_exact_raw": True,
                "candidate_RAM_only_no_checkpoint_write": True,
                "specialist_valid_consumed_for_optimization": True,
                "promotion_evidence": False,
                "additional_validation_broad_gold_opened_by_formal_evaluator": False,
            },
            "audit": {
                "local": local_static,
                "cw11": cw11.static_audit(cw11_source),
                "fulltrain_v5_anchor": v5.static_audit(v5_source, cw10, v4),
                "fulltrain_v4": v4.static_audit(v4_source),
                "fulltrain_v3": base.local_static_audit(base_source),
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
            "network_upload_submission_performed": False,
        }
    else:
        result = run_gate(
            source,
            local_static,
            cw11,
            cw11_evidence,
            v5,
            v5_evidence,
            v4,
            v4_evidence,
            base,
            base_evidence,
            raw_checkpoint_evidence,
            e904_checkpoint_evidence,
        )
        result["input_lock"]["self"] = self_evidence

    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
