#!/usr/bin/env python3
"""Read-only exploratory 29-row continuation adding the final v4 FLG CW.

This wrapper hash-locks the audited CW9 probe and adds one ordered-only FLG
guard.  Its terminal remains observational and RAM-only; formal full-train
evaluation is required afterward.  No files, validation data, network, upload,
or submission paths are used.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
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
SCRIPT = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-cw10-cuttingplane-probe-v1"

CW9 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1.py"
CW9_SHA256 = "f77a76f2ac91732e8bc8d9a436ca7a49058b5de50f7b3114db14cffa1fbf5292"
PRIMARY = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py"
PRIMARY_SHA256 = "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c"
FROZEN_MODE = 0o555

EXPECTED_ROWS = 29
EXPECTED_TARGET_ROWS = 5
EXPECTED_GUARD_ROWS = 24
EXPECTED_TARGET_OBLIGATIONS = 20
EXPECTED_GUARD_OBLIGATIONS = 69
EXPECTED_ACTIVE_PAIRS = 30
EXPECTED_ADDED_GUARDS = 6

FINAL_ADDED_GUARD = {
    "role": "guard",
    "panel": "flg",
    "member": "train/part-00000.jsonl",
    "line_index_zero_based": 6908,
    "line_sha256": "65799675c3e6bae838679412f0ee172968e358264f49424e811752a93805ae64",
    "expected_context": 5,
    "expected_expert_order": [3, 1],
    "expected_option_count": 4,
    "metrics_union": ["ordered_exact"],
    "expected_c0_false_metrics": [],
    "positive_option": 3,
    "negative_option": 1,
    "formal_raw_order": [3, 1],
    "formal_c0_order": [3, 1],
    "formal_failed_candidate_order": [1, 3],
    "expected_formal_b256_raw_margin": 0.01171875,
    "formal_batch_index_zero_based": 26,
    "formal_batch_position_zero_based": 252,
    "formal_batch_size": 256,
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


def import_frozen(path: Path, expected_sha256: str, label: str) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, expected_sha256, label, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(
        f"_cw10_{path.stem}_{expected_sha256[:12]}", path
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


def static_audit(source: bytes, cw9: ModuleType) -> dict[str, Any]:
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
    parameters = inspect.signature(run_probe).parameters
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_or_network_calls": not call_hits,
        "no_network_or_submission_imports": not import_hits,
        "cw9_guard_count_exact_5": len(cw9.ADDED_GUARDS) == 5,
        "cw10_combined_guard_count_exact_6": len(cw9.ADDED_GUARDS) + 1
        == EXPECTED_ADDED_GUARDS,
        "final_guard_metric_obligation_exact_1": len(
            FINAL_ADDED_GUARD["metrics_union"]
        )
        == 1,
        "formal_batch_coordinate_exact": (
            int(FINAL_ADDED_GUARD["formal_batch_index_zero_based"]) == 26
            and int(FINAL_ADDED_GUARD["formal_batch_position_zero_based"])
            == 252
            and int(FINAL_ADDED_GUARD["formal_batch_size"]) == 256
        ),
        "candidate_consumer_present": "candidate_consumer" in parameters,
        "candidate_consumer_default_none": parameters[
            "candidate_consumer"
        ].default
        is None,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW10 static audit failed: {checks}")
    return {
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "stdout_only": True,
    }


def run_probe(
    primary: ModuleType,
    primary_source: bytes,
    primary_evidence: Mapping[str, Any],
    *,
    candidate_consumer: Any = None,
) -> dict[str, Any]:
    cw9, cw9_evidence = import_frozen(CW9, CW9_SHA256, "frozen CW9 probe")
    combined_guards = tuple(cw9.ADDED_GUARDS) + (dict(FINAL_ADDED_GUARD),)
    patch_values = {
        "ADDED_GUARDS": combined_guards,
        "EXPECTED_ROWS": EXPECTED_ROWS,
        "EXPECTED_GUARD_ROWS": EXPECTED_GUARD_ROWS,
        "EXPECTED_GUARD_OBLIGATIONS": EXPECTED_GUARD_OBLIGATIONS,
        "EXPECTED_INITIAL_ACTIVE_PAIRS": EXPECTED_ACTIVE_PAIRS,
        "EXPECTED_ADDED_GUARDS": EXPECTED_ADDED_GUARDS,
    }
    original_values = {name: getattr(cw9, name) for name in patch_values}
    restored = False
    try:
        for name, value in patch_values.items():
            setattr(cw9, name, value)
        result = cw9.run_probe(
            primary,
            primary_source,
            primary_evidence,
            candidate_consumer=candidate_consumer,
        )
    finally:
        for name, value in original_values.items():
            setattr(cw9, name, value)
        restored = all(
            getattr(cw9, name) is value
            for name, value in original_values.items()
        )
    if not restored:
        raise RuntimeError("CW9 dependency globals not restored")
    checks = {
        "wrapped_selected_gate_pass": bool(
            result["decision"]["observational_terminal_checks"][
                "selected_28row_gate_pass"
            ]
        ),
        "terminal_success_iteration_found": result["decision"][
            "success_iteration_after_c0"
        ]
        is not None,
        "active_pair_count_exact_30": int(
            result["active_pair_contract"]["final_count"]
        )
        == EXPECTED_ACTIVE_PAIRS,
        "selected_row_count_exact_29": int(result["scope"]["row_count"])
        == EXPECTED_ROWS,
        "candidate_consumer_semantics_exact": bool(
            result["decision"][
                "candidate_consumer_called_before_finally_restore"
            ]
        )
        == (candidate_consumer is not None),
        "wrapped_primary_finally_raw_restore_pass": bool(
            result["final_integrity"]["pass"]
        ),
        "cw9_module_globals_restored": restored,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW10 exploratory integrity failed: {checks}")
    observations = result["decision"]["observational_terminal_checks"]
    observations["active_pair_count_exact_30"] = observations.pop(
        "active_pair_count_exact_29"
    )
    observations["selected_29row_gate_pass"] = observations.pop(
        "selected_28row_gate_pass"
    )
    result["schema_version"] = SCHEMA
    result["status"] = "exploratory_29row_30pair_optimization_success"
    result["decision"]["status"] = result["status"]
    result["decision"]["cw10_wrapper_checks"] = checks
    result["scope"].update(
        {
            "row_count": EXPECTED_ROWS,
            "guard_rows": EXPECTED_GUARD_ROWS,
            "new_fulltrain_cw_rows": EXPECTED_ADDED_GUARDS,
            "new_fulltrain_cw_rows_added": EXPECTED_ADDED_GUARDS,
            "original_cw9_rows": 28,
        }
    )
    result["input_lock"]["wrapped_cw9_probe"] = dict(cw9_evidence)
    result["cw10_wrapper_audit"] = {
        "checks": checks,
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
        SCRIPT, None, "CW10 exploratory wrapper", expected_mode=FROZEN_MODE
    )
    cw9, cw9_evidence = import_frozen(CW9, CW9_SHA256, "frozen CW9 probe")
    local_static = static_audit(source, cw9)
    if args.mode == "static":
        cw9_source, _ = read_regular_bytes(
            CW9, CW9_SHA256, "static CW9 source", expected_mode=FROZEN_MODE
        )
        result = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {"cw9_probe": cw9_evidence},
            "new_guard_spec_sha256": sha256_bytes(
                canonical_json(FINAL_ADDED_GUARD)
            ),
            "contract": {
                "rows": EXPECTED_ROWS,
                "target_rows": EXPECTED_TARGET_ROWS,
                "guard_rows": EXPECTED_GUARD_ROWS,
                "target_obligations": EXPECTED_TARGET_OBLIGATIONS,
                "guard_obligations": EXPECTED_GUARD_OBLIGATIONS,
                "active_pairs": EXPECTED_ACTIVE_PAIRS,
                "added_guards": EXPECTED_ADDED_GUARDS,
            },
            "audit": {
                "local": local_static,
                "cw9_probe": cw9.static_audit(cw9_source),
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
        }
    else:
        primary, primary_evidence = import_frozen(
            PRIMARY, PRIMARY_SHA256, "frozen CW4 primary"
        )
        primary_source, _ = read_regular_bytes(
            PRIMARY,
            PRIMARY_SHA256,
            "frozen CW4 primary source",
            expected_mode=FROZEN_MODE,
        )
        result = run_probe(primary, primary_source, primary_evidence)
        result["input_lock"]["self"] = self_evidence
        result["local_static_audit"] = local_static
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
