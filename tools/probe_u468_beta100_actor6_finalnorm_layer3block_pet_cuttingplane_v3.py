#!/usr/bin/env python3
"""Thin frozen extension of combined v1 with the complete layer-3 block.

The mutable vector is the frozen actor6 prefix, final LayerNorm, and all
twelve parameters of transformer layer 3: twenty FP32 tensors / 264,321
scalars.  Only the actor6 prefix is projected out of the frozen P/E/T span.
The hard combined L2 radius remains 1e-3 and every weak-BF16, Direct512,
complete-train, consumer, and restoration rule is delegated to hash-bound
combined v1.

Standalone execution is RAM-only, always passes ``candidate_consumer=None``,
and prints one JSON document.  No checkpoint or result file is written.
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
import random
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_beta100_actor6_finalnorm_layer3block_pet_cuttingplane_v3.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta100-actor6-finalnorm-layer3block-pet-cuttingplane-v3"

BASE_TOOL = TOOLS / "probe_u468_beta100_actor6_finalnorm_pet_cuttingplane_v1.py"
BASE_TOOL_SHA256 = "2dd81aa5837a0a3eb7a29a4f3335c9cff41e9a1e0be86df4486481a9de09b66a"
BASE_TOOL_MODE = 0o444

ACTOR_ELEMENTS = 65793
FINALNORM_ELEMENTS = 256
LAYER3BLOCK_ELEMENTS = 198272
SHARED_TAIL_ELEMENTS = FINALNORM_ELEMENTS + LAYER3BLOCK_ELEMENTS
NORM_ELEMENTS = SHARED_TAIL_ELEMENTS
COMBINED_ELEMENTS = ACTOR_ELEMENTS + NORM_ELEMENTS
HARD_COMBINED_L2 = 1.0e-3

FINAL_NORM_NAMES = ("transformer.norm.weight", "transformer.norm.bias")
LAYER3_BLOCK_NAMES = (
    "transformer.layers.3.self_attn.in_proj_weight",
    "transformer.layers.3.self_attn.in_proj_bias",
    "transformer.layers.3.self_attn.out_proj.weight",
    "transformer.layers.3.self_attn.out_proj.bias",
    "transformer.layers.3.linear1.weight",
    "transformer.layers.3.linear1.bias",
    "transformer.layers.3.linear2.weight",
    "transformer.layers.3.linear2.bias",
    "transformer.layers.3.norm1.weight",
    "transformer.layers.3.norm1.bias",
    "transformer.layers.3.norm2.weight",
    "transformer.layers.3.norm2.bias",
)
EXPECTED_BETA_VARIABLE20_SHA256 = (
    "23119f1735a43916b1f7df14bc5ac45edcdd3addb18e09dc4ed2a6b98e7feb89"
)
EXPECTED_BETA_OTHER60_SHA256 = (
    "6e79320edc16e9308ab74f2063489d692905c8b84018cd014a5df39ad02476b6"
)


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
    visible = os.lstat(path)
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
    ):
        raise RuntimeError(f"{label} changed during held-fd read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"{label} SHA drift: expected {expected_sha256}, observed {digest}"
        )
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(
            f"{label} mode drift: expected {oct(expected_mode)}, observed {oct(mode)}"
        )
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "single_link_regular_held_fd_identity_exact": True,
    }


def import_locked(
    path: Path, expected_sha256: str, module_name: str, *, expected_mode: int
) -> tuple[ModuleType, dict[str, Any]]:
    payload, evidence = read_regular_bytes(
        path, expected_sha256, module_name, expected_mode=expected_mode
    )
    spec = importlib.util.spec_from_file_location(
        f"_{module_name}_{sha256_bytes(payload)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


BASE, BASE_EVIDENCE = import_locked(
    BASE_TOOL,
    BASE_TOOL_SHA256,
    "frozen_actor6_finalnorm_combined_v1",
    expected_mode=BASE_TOOL_MODE,
)
torch = BASE.torch
ACTOR_NAMES = tuple(BASE.ACTOR_NAMES)
NORM_NAMES = FINAL_NORM_NAMES + LAYER3_BLOCK_NAMES
VARIABLE_NAMES = ACTOR_NAMES + NORM_NAMES

_ORIGINAL_BUILD_GEOMETRY = BASE.build_combined_geometry
_ORIGINAL_ENDPOINT_AUDIT = BASE.endpoint_audit


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")
    if len(VARIABLE_NAMES) != 20 or COMBINED_ELEMENTS != 264321:
        raise RuntimeError("v3 variable manifest drift")
    if HARD_COMBINED_L2 != BASE.HARD_COMBINED_L2:
        raise RuntimeError("v3 attempted to relax the combined radius")


def endpoint_audit_v3(vector: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (COMBINED_ELEMENTS,):
        raise RuntimeError(f"v3 endpoint shape drift: {value.shape}")
    if q.shape != (COMBINED_ELEMENTS, 3):
        raise RuntimeError(f"v3 embedded-Q shape drift: {q.shape}")
    components = q.T @ value
    actor = value[:ACTOR_ELEMENTS]
    finalnorm = value[ACTOR_ELEMENTS : ACTOR_ELEMENTS + FINALNORM_ELEMENTS]
    layer3block = value[ACTOR_ELEMENTS + FINALNORM_ELEMENTS :]
    combined_l2 = float(np.linalg.norm(value))
    actor_l2 = float(np.linalg.norm(actor))
    finalnorm_l2 = float(np.linalg.norm(finalnorm))
    layer3block_l2 = float(np.linalg.norm(layer3block))
    maximum = float(np.max(np.abs(components), initial=0.0))
    relative = float(np.linalg.norm(components)) / actor_l2 if actor_l2 > 0.0 else 0.0
    passed = (
        maximum <= BASE.PET_ABS_TOLERANCE
        and relative <= BASE.PET_RELATIVE_TOLERANCE
        and combined_l2 <= HARD_COMBINED_L2 + BASE.RADIUS_TOLERANCE
    )
    return {
        "combined_l2": combined_l2,
        "actor6_l2": actor_l2,
        "finalnorm_l2": finalnorm_l2,
        "layer3block_l2": layer3block_l2,
        "all_shared_tail_l2": float(math.hypot(finalnorm_l2, layer3block_l2)),
        "pet_components": [float(item) for item in components],
        "pet_component_max_abs": maximum,
        "pet_component_relative_to_actor_l2": relative,
        "pet_abs_pass": maximum <= BASE.PET_ABS_TOLERANCE,
        "pet_relative_pass": relative <= BASE.PET_RELATIVE_TOLERANCE,
        "combined_radius_pass": combined_l2 <= HARD_COMBINED_L2 + BASE.RADIUS_TOLERANCE,
        "pass": passed,
    }


def build_combined_geometry_v3(basis: Mapping[str, Any]) -> dict[str, Any]:
    q_actor = np.ascontiguousarray(basis["q"].numpy(), dtype=np.float64)
    if q_actor.shape != (ACTOR_ELEMENTS, 3):
        raise RuntimeError("v3 actor PET Q shape drift")
    q = np.zeros((COMBINED_ELEMENTS, 3), dtype=np.float64)
    q[:ACTOR_ELEMENTS] = q_actor
    gram = q.T @ q
    residual = float(np.max(np.abs(gram - np.eye(3))))
    if residual > 1.0e-12 or bool(np.any(q[ACTOR_ELEMENTS:] != 0.0)):
        raise RuntimeError("v3 embedded PET geometry drift")
    p_state = basis["payloads"]["P"]["model_state_dict"]
    variable = {name: p_state[name] for name in VARIABLE_NAMES}
    other_names = sorted(set(p_state) - set(VARIABLE_NAMES))
    other = {name: p_state[name] for name in other_names}
    checks = {
        "variable_tensor_count_exact_20": len(variable) == 20,
        "combined_elements_exact_264321": (
            sum(int(value.numel()) for value in variable.values())
            == COMBINED_ELEMENTS
        ),
        "variable_all_fp32": all(
            value.dtype == torch.float32 for value in variable.values()
        ),
        "other_tensor_count_exact_60": len(other) == 60,
        "variable20_sha_exact": (
            BASE.PET.frozen.model_state_sha256(variable)
            == EXPECTED_BETA_VARIABLE20_SHA256
        ),
        "other60_sha_exact": (
            BASE.PET.frozen.model_state_sha256(other)
            == EXPECTED_BETA_OTHER60_SHA256
        ),
        "embedded_shared_tail_rows_exact_zero": bool(
            np.all(q[ACTOR_ELEMENTS:] == 0.0)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v3 beta100 state geometry drift: {checks}")
    return {
        "q": q,
        "q_actor": q_actor,
        "report": {
            "variable_names": list(VARIABLE_NAMES),
            "actor_elements": ACTOR_ELEMENTS,
            "finalnorm_elements": FINALNORM_ELEMENTS,
            "layer3block_elements": LAYER3BLOCK_ELEMENTS,
            "shared_tail_elements": SHARED_TAIL_ELEMENTS,
            "combined_elements": COMBINED_ELEMENTS,
            "embedded_q_shape": list(q.shape),
            "embedded_q_float64_sha256": BASE.vector_sha256(q.reshape(-1), "<f8"),
            "orthonormality_max_abs_residual": residual,
            "pet_span": "frozen actor6 P/E/T with 198528 zero shared-tail rows",
            "beta_variable20_sha256": EXPECTED_BETA_VARIABLE20_SHA256,
            "beta_other60_sha256": EXPECTED_BETA_OTHER60_SHA256,
            "checks": checks,
        },
    }


def _binding_values() -> dict[str, Any]:
    return {
        "SCRIPT": SCRIPT,
        "SCHEMA": SCHEMA,
        "NORM_NAMES": NORM_NAMES,
        "VARIABLE_NAMES": VARIABLE_NAMES,
        "NORM_ELEMENTS": NORM_ELEMENTS,
        "COMBINED_ELEMENTS": COMBINED_ELEMENTS,
        "EXPECTED_BETA_VARIABLE8_SHA256": EXPECTED_BETA_VARIABLE20_SHA256,
        "EXPECTED_BETA_OTHER72_SHA256": EXPECTED_BETA_OTHER60_SHA256,
        "build_combined_geometry": build_combined_geometry_v3,
        "endpoint_audit": endpoint_audit_v3,
    }


def _apply_bindings() -> dict[str, Any]:
    values = _binding_values()
    old = {name: getattr(BASE, name) for name in values}
    for name, value in values.items():
        setattr(BASE, name, value)
    return old


def _restore_bindings(old: Mapping[str, Any]) -> None:
    for name, value in old.items():
        setattr(BASE, name, value)


def _standalone_consumer_none(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if name != "run_probe":
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "candidate_consumer"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
            ):
                return True
    return False


def static_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_calls = {
        "backward", "step", "save", "savez", "write", "write_text",
        "write_bytes", "touch", "mkdir", "makedirs", "unlink", "remove",
        "rmtree", "rename", "replace",
    }
    forbidden_imports = {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
    calls: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    write_flags: list[dict[str, Any]] = []
    copy_sites: list[int] = []
    print_sites: list[int] = []
    optimizer_sites: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_imports:
                    imports.append({"line": node.lineno, "name": name})
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
        ):
            write_flags.append({"line": node.lineno, "name": node.attr})
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        if name in forbidden_calls:
            calls.append({"line": node.lineno, "name": name})
        if name == "copy_":
            copy_sites.append(node.lineno)
        if name == "print":
            print_sites.append(node.lineno)
        dotted = ast.unparse(node.func).lower()
        if dotted.startswith("torch.optim") or ".optimizer" in dotted:
            optimizer_sites.append(node.lineno)
    base_static = BASE.static_audit()
    binding_names = tuple(_binding_values())
    binding_before = {name: getattr(BASE, name) for name in binding_names}
    old = _apply_bindings()
    try:
        solver_test = BASE.solver_synthetic_self_test()
        zero_q = BASE.zero_positive_bf16_q()
    finally:
        _restore_bindings(old)
    binding_restore_exact = all(
        getattr(BASE, name) is binding_before[name] for name in binding_names
    )
    signature = inspect.signature(run_probe)
    checks = {
        "ast_parse": True,
        "no_local_write_backward_step_save_or_network": not (
            calls or imports or write_flags
        ),
        "no_local_optimizer": not optimizer_sites,
        "local_copy_sites_exact_zero_delegated_to_frozen_base": not copy_sites,
        "single_stdout_print_site": len(print_sites) == 1,
        "frozen_base_static_all_pass": all(base_static["checks"].values()),
        "scope_exact_20_tensors_264321": (
            len(VARIABLE_NAMES) == 20 and COMBINED_ELEMENTS == 264321
        ),
        "shared_partition_exact_256_plus_198272": (
            FINALNORM_ELEMENTS == 256
            and LAYER3BLOCK_ELEMENTS == 198272
            and SHARED_TAIL_ELEMENTS == 198528
            and NORM_ELEMENTS == SHARED_TAIL_ELEMENTS
        ),
        "one_hard_combined_radius_unchanged": HARD_COMBINED_L2 == 1.0e-3,
        "solver_self_test_under_v3_bindings": solver_test["pass"],
        "value_zero_quantum_exact": zero_q == math.ldexp(1.0, -133),
        "delegated_module_bindings_restored_exact": binding_restore_exact,
        "candidate_consumer_default_none": (
            signature.parameters["candidate_consumer"].default is None
        ),
        "standalone_candidate_consumer_none": _standalone_consumer_none(tree),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v3 static audit failed: {checks}")
    return {
        "status": "static_layer3block_v3_audit_passed",
        "checks": checks,
        "forbidden_call_hits": calls,
        "forbidden_import_hits": imports,
        "forbidden_os_write_flags": write_flags,
        "optimizer_sites": optimizer_sites,
        "local_ram_copy_sites": copy_sites,
        "stdout_print_sites": print_sites,
        "frozen_base_audit": base_static,
        "solver_synthetic_self_test": solver_test,
        "value_zero_positive_bf16_quantum": zero_q,
    }


def _normalize_result_labels(result: dict[str, Any]) -> dict[str, Any]:
    result["schema_version"] = SCHEMA
    result.setdefault("input_lock", {})["frozen_combined_v1"] = BASE_EVIDENCE
    scope = result.get("scope")
    if isinstance(scope, dict):
        scope["other_state_tensors_exact"] = 60
        scope["finalnorm_elements"] = FINALNORM_ELEMENTS
        scope["layer3block_elements"] = LAYER3BLOCK_ELEMENTS
        scope["shared_tail_elements"] = SHARED_TAIL_ELEMENTS
    integrity = result.get("final_integrity")
    if isinstance(integrity, dict):
        if "variable8_tensor_restored_exact" in integrity:
            integrity["variable20_tensor_restored_exact"] = integrity.pop(
                "variable8_tensor_restored_exact"
            )
        if "other72_sha_after" in integrity:
            integrity["other60_sha_after"] = integrity.pop("other72_sha_after")
        if "other72_restored_exact" in integrity:
            integrity["other60_restored_exact"] = integrity.pop(
                "other72_restored_exact"
            )
    return result


def run_probe(
    transition_context: Mapping[str, Any] | None = None,
    candidate_consumer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if candidate_consumer is not None and not callable(candidate_consumer):
        raise TypeError("candidate_consumer must be callable or None")

    def enriched_consumer(payload_source: Mapping[str, Any]) -> None:
        if candidate_consumer is None:
            raise RuntimeError("internal v3 consumer contract drift")
        payload = dict(payload_source)
        vector_tensor = payload["actual_combined_delta_float32_cpu"]
        vector = vector_tensor.detach().cpu().float().reshape(-1)
        if int(vector.numel()) != COMBINED_ELEMENTS:
            raise RuntimeError("v3 consumer vector shape drift")
        final = vector[ACTOR_ELEMENTS : ACTOR_ELEMENTS + FINALNORM_ELEMENTS]
        layer3block = vector[ACTOR_ELEMENTS + FINALNORM_ELEMENTS :]
        payload["actual_finalnorm_delta_l2"] = float(final.double().norm())
        payload["actual_layer3block_delta_l2"] = float(
            layer3block.double().norm()
        )
        payload["actual_shared_tail_delta_l2"] = float(
            vector[ACTOR_ELEMENTS:].double().norm()
        )
        payload["v3_variable_names"] = list(VARIABLE_NAMES)
        payload["frozen_combined_v1"] = BASE_EVIDENCE
        candidate_consumer(payload)

    delegated_consumer = enriched_consumer if candidate_consumer is not None else None
    old = _apply_bindings()
    try:
        result = BASE.run_probe(
            transition_context=transition_context,
            candidate_consumer=delegated_consumer,
        )
    finally:
        _restore_bindings(old)
    return _normalize_result_labels(result)


def cache_audit() -> dict[str, Any]:
    old = _apply_bindings()
    try:
        result = BASE.cache_audit()
    finally:
        _restore_bindings(old)
    result = _normalize_result_labels(result)
    result["status"] = "layer3block_v3_cache_audit_passed_zero_write_train_only"
    result["self"] = read_regular_bytes(SCRIPT, None, "layer3block v3 self")[1]
    result["frozen_combined_v1"] = BASE_EVIDENCE
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("static-audit", "cache-audit", "actual"), required=True
    )
    args = parser.parse_args()
    validate_runtime()
    static = static_audit()
    if args.mode == "static-audit":
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_audit_passed",
            "self": read_regular_bytes(SCRIPT, None, "layer3block v3 self")[1],
            "frozen_combined_v1": BASE_EVIDENCE,
            "audit": static,
            "actual_executed": False,
        }
    elif args.mode == "cache-audit":
        result = {"static": static, **cache_audit()}
    else:
        result = run_probe(transition_context=None, candidate_consumer=None)
        result["static_audit"] = static
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
