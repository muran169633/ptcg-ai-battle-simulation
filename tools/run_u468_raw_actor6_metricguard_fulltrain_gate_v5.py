#!/usr/bin/env python3
"""Formal full-train gate for the frozen CW10 exploratory terminal.

This read-only adapter reuses the audited v4 full-train implementation while
binding the CW10 terminal identity (iteration/vector/model/ledger) and the
29-row/30-pair candidate consumer contract.  The compatibility adapter only
renames legacy CW9 labels; all numeric and hash gates remain exact.

No validation reads, training, file writes, network calls, uploads, or
submissions are performed.  Candidate state remains RAM-only.
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
SCRIPT = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v5.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-gate-v5"

CW10 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1.py"
CW10_SHA256 = "546e90c5b3ca35c84b8efc08109b2f310b45b2895d24aeee70561d13bffa146c"
V4 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v4.py"
V4_SHA256 = "75974150f83455c10bbf6a90b3c571e4daf319f4e06e9c77f67673fbf34b9ea8"
FROZEN_MODE = 0o555

EXPECTED_SUCCESS_ITERATION = 9
EXPECTED_TERMINAL_CUMULATIVE_L2 = 0.007627603437990896
EXPECTED_TERMINAL_CUMULATIVE_SHA256 = (
    "42e02c90f65b81e1e8b801fa173fcf953b347f73e65b58000719107fa83a6c28"
)
EXPECTED_TERMINAL_MODEL_STATE_SHA256 = (
    "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
)
EXPECTED_ACTIVE_PAIR_COUNT = 30
EXPECTED_ACTIVE_PAIR_LEDGER_SHA256 = (
    "22b20b8630d63c6fbfd249a82a188397b468a84d1b747a2159c17b05a6f4983d"
)
EXPECTED_SELECTED_ROW_COUNT = 29
CW10_STATUS = "exploratory_29row_30pair_optimization_success"
LEGACY_V4_EXPECTED_STATUS = "exploratory_28row_29pair_optimization_success"


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


def import_frozen(path: Path, expected_sha256: str, label: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, expected_sha256, label, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(
        f"_fulltrain_v5_{path.stem}_{expected_sha256[:12]}", path
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


def static_audit(source: bytes, cw10: ModuleType, v4: ModuleType) -> dict[str, Any]:
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
    cw10_parameters = inspect.signature(cw10.run_probe).parameters
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_or_network_calls": not call_hits,
        "no_network_or_submission_imports": not import_hits,
        "cw10_consumer_present_and_default_none": (
            "candidate_consumer" in cw10_parameters
            and cw10_parameters["candidate_consumer"].default is None
        ),
        "v4_run_gate_callable": callable(v4.run_gate),
        "v4_capture_callable": callable(v4.capture_candidate),
        "terminal_hashes_hex64": all(
            len(value) == 64
            for value in (
                EXPECTED_TERMINAL_CUMULATIVE_SHA256,
                EXPECTED_TERMINAL_MODEL_STATE_SHA256,
                EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
            )
        ),
        "selected_row_count_exact_29": EXPECTED_SELECTED_ROW_COUNT == 29,
        "active_pair_count_exact_30": EXPECTED_ACTIVE_PAIR_COUNT == 30,
    }
    if not all(checks.values()):
        raise RuntimeError(f"fulltrain v5 static audit failed: {checks}")
    return {
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "stdout_only": True,
    }


def run_gate(
    source: bytes,
    static: Mapping[str, Any],
    cw10: ModuleType,
    cw10_evidence: Mapping[str, Any],
    v4: ModuleType,
    v4_evidence: Mapping[str, Any],
    base: ModuleType,
    base_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    interfaces = v4.interface_audit(cw10, base)
    original_cw10_run = cw10.run_probe
    original_capture = v4.capture_candidate
    patch_values = {
        "SCRIPT": SCRIPT,
        "SCHEMA": SCHEMA,
        "EXPECTED_SUCCESS_ITERATION": EXPECTED_SUCCESS_ITERATION,
        "EXPECTED_TERMINAL_CUMULATIVE_L2": EXPECTED_TERMINAL_CUMULATIVE_L2,
        "EXPECTED_TERMINAL_CUMULATIVE_SHA256": EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "EXPECTED_TERMINAL_MODEL_STATE_SHA256": EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "EXPECTED_ACTIVE_PAIR_COUNT": EXPECTED_ACTIVE_PAIR_COUNT,
        "EXPECTED_ACTIVE_PAIR_LEDGER_SHA256": EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
    }
    original_values = {name: getattr(v4, name) for name in patch_values}
    adapter_audit: dict[str, Any] = {
        "cw10_run_calls": 0,
        "capture_calls": 0,
        "actual_cw10_status_verified_before_legacy_label_adapter": False,
        "actual_selected_row_count_exact_29_before_capture_adapter": False,
    }
    restored = False
    try:
        for name, value in patch_values.items():
            setattr(v4, name, value)

        def adapted_cw10_run(
            primary: ModuleType,
            primary_source: bytes,
            primary_evidence: Mapping[str, Any],
            *,
            candidate_consumer: Any = None,
        ) -> dict[str, Any]:
            adapter_audit["cw10_run_calls"] += 1
            observed = original_cw10_run(
                primary,
                primary_source,
                primary_evidence,
                candidate_consumer=candidate_consumer,
            )
            if observed.get("status") != CW10_STATUS:
                raise RuntimeError("actual CW10 status drift before adapter")
            adapter_audit[
                "actual_cw10_status_verified_before_legacy_label_adapter"
            ] = True
            observed["cw10_actual_status_before_v4_compatibility_adapter"] = (
                observed["status"]
            )
            observed["status"] = LEGACY_V4_EXPECTED_STATUS
            return observed

        def adapted_capture(
            context: Mapping[str, Any],
            holder: dict[str, Any],
            base_module: ModuleType,
            geometry: ModuleType,
        ) -> None:
            adapter_audit["capture_calls"] += 1
            actual_rows = int(context.get("expanded_row_count", -1))
            if actual_rows != EXPECTED_SELECTED_ROW_COUNT:
                raise RuntimeError("actual CW10 selected-row count drift")
            adapter_audit[
                "actual_selected_row_count_exact_29_before_capture_adapter"
            ] = True
            legacy_context = dict(context)
            legacy_context["expanded_row_count"] = 28
            original_capture(
                legacy_context,
                holder,
                base_module,
                geometry,
            )
            checks = holder["audit"]["checks"]
            if not checks.pop("expanded_row_count_exact_28"):
                raise RuntimeError("legacy row-count compatibility check failed")
            checks["actual_expanded_row_count_exact_29"] = True

        cw10.run_probe = adapted_cw10_run
        v4.capture_candidate = adapted_capture
        result = v4.run_gate(
            source,
            static,
            cw10,
            cw10_evidence,
            base,
            base_evidence,
            interfaces,
        )
    finally:
        cw10.run_probe = original_cw10_run
        v4.capture_candidate = original_capture
        for name, value in original_values.items():
            setattr(v4, name, value)
        restored = (
            cw10.run_probe is original_cw10_run
            and v4.capture_candidate is original_capture
            and all(
                getattr(v4, name) is value
                for name, value in original_values.items()
            )
        )
    compatibility_checks = {
        "cw10_run_called_once": int(adapter_audit["cw10_run_calls"]) == 1,
        "capture_called_once": int(adapter_audit["capture_calls"]) == 1,
        "actual_cw10_status_verified": bool(
            adapter_audit[
                "actual_cw10_status_verified_before_legacy_label_adapter"
            ]
        ),
        "actual_selected_row_count_exact_29": bool(
            adapter_audit[
                "actual_selected_row_count_exact_29_before_capture_adapter"
            ]
        ),
        "all_dependency_globals_and_functions_restored": restored,
        "candidate_model_sha_exact": result["candidate_capture"][
            "candidate_model_state_sha256"
        ]
        == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
        "terminal_vector_sha_exact": result["candidate_capture"][
            "terminal_cumulative_float64_le_sha256"
        ]
        == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
        "active_pair_count_exact_30": int(
            result["candidate_capture"]["active_pair_count"]
        )
        == EXPECTED_ACTIVE_PAIR_COUNT,
        "active_pair_ledger_sha_exact": result["candidate_capture"][
            "active_pair_ledger_sha256"
        ]
        == EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
    }
    if not all(compatibility_checks.values()):
        raise RuntimeError(
            f"fulltrain v5 compatibility integrity failed: {compatibility_checks}"
        )
    probe_result = result["probe_result"]
    actual_status = probe_result.pop(
        "cw10_actual_status_before_v4_compatibility_adapter"
    )
    if actual_status != CW10_STATUS:
        raise RuntimeError("stored actual CW10 status drift")
    probe_result["status"] = actual_status
    frozen_checks = result["probe_frozen_reproduction_checks"]
    if not frozen_checks.pop("probe_status_success"):
        raise RuntimeError("legacy v4 probe status gate did not pass")
    frozen_checks["actual_cw10_status_exact"] = True
    result["schema_version"] = SCHEMA
    result["input_lock"]["self"] = {
        "path": str(SCRIPT.relative_to(ROOT)),
        "sha256": sha256_bytes(source),
    }
    result["input_lock"]["fulltrain_v4_implementation"] = dict(v4_evidence)
    result["input_lock"]["cw10_probe"] = dict(cw10_evidence)
    scope = result["scope_audit"]
    if not scope.pop("selected_28_train_rows_then_all_24050_train_rows"):
        raise RuntimeError("legacy v4 selected/fulltrain scope gate failed")
    scope["selected_29_train_rows_then_all_24050_train_rows"] = True
    result["v5_compatibility_adapter_audit"] = {
        "raw": adapter_audit,
        "checks": compatibility_checks,
        "legacy_labels_only_no_numeric_or_hash_gate_relaxed": True,
        "all_dependency_globals_and_functions_restored": restored,
    }
    result["local_static_audit"] = dict(static)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT, None, "fulltrain gate v5", expected_mode=FROZEN_MODE
    )
    cw10, cw10_evidence = import_frozen(
        CW10, CW10_SHA256, "frozen CW10 probe"
    )
    v4, v4_evidence = import_frozen(V4, V4_SHA256, "frozen fulltrain v4")
    base, base_evidence = import_frozen(
        v4.BASE_GATE,
        v4.BASE_GATE_SHA256,
        "frozen fulltrain v3 utilities",
    )
    local_static = static_audit(source, cw10, v4)
    interfaces = v4.interface_audit(cw10, base)
    if args.mode == "static":
        cw10_source, _ = read_regular_bytes(
            CW10, CW10_SHA256, "static CW10 source", expected_mode=FROZEN_MODE
        )
        v4_source, _ = read_regular_bytes(
            V4, V4_SHA256, "static fulltrain v4 source", expected_mode=FROZEN_MODE
        )
        cw9, _ = cw10.import_frozen(
            cw10.CW9, cw10.CW9_SHA256, "static CW9 dependency"
        )
        result = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {
                "cw10_probe": cw10_evidence,
                "fulltrain_v4": v4_evidence,
                "base_gate_v3": base_evidence,
            },
            "interface_audit": interfaces,
            "terminal_contract": {
                "success_iteration": EXPECTED_SUCCESS_ITERATION,
                "cumulative_l2": EXPECTED_TERMINAL_CUMULATIVE_L2,
                "cumulative_sha256": EXPECTED_TERMINAL_CUMULATIVE_SHA256,
                "model_state_sha256": EXPECTED_TERMINAL_MODEL_STATE_SHA256,
                "active_pair_count": EXPECTED_ACTIVE_PAIR_COUNT,
                "active_pair_ledger_sha256": EXPECTED_ACTIVE_PAIR_LEDGER_SHA256,
                "selected_row_count": EXPECTED_SELECTED_ROW_COUNT,
            },
            "audit": {
                "local": local_static,
                "cw10": cw10.static_audit(cw10_source, cw9),
                "fulltrain_v4": v4.static_audit(v4_source),
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
        }
    else:
        result = run_gate(
            source,
            local_static,
            cw10,
            cw10_evidence,
            v4,
            v4_evidence,
            base,
            base_evidence,
        )
        result["input_lock"]["self"] = self_evidence
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
