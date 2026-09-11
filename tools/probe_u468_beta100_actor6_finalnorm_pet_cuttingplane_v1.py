#!/usr/bin/env python3
"""Train-only beta100 actor6 + final-LayerNorm PET cutting-plane probe.

This version expands the frozen actor6 PET search by exactly the affine
``transformer.norm`` weight and bias.  The 65,793 actor coordinates remain in
the orthogonal complement of the frozen P/E/T span; the 256 FinalNorm
coordinates are appended without changing that span.  The combined endpoint
has one hard L2 radius of 1e-3.

Policy and count cuts use one *current native-BF16 positive quantum*; value
sign cuts use exactly the smallest positive native-BF16 quantum above zero.
They never preserve a full beta100 margin.  Every evaluated endpoint is
scanned on the complete 24,050-row train stream; new policy harms, changed
count behaviour, and changed value signs become same-row cuts for the next
round.  The thirteen Direct512 ordered-NLL losses are actual endpoint gates
and local linear constraints.

The standalone program is RAM-only, always passes ``candidate_consumer=None``,
writes no checkpoint or result, and prints one JSON document.  A callable
consumer may be injected programmatically and is invoked only for a fully
passing endpoint, before the mandatory finally restoration.  Construction of
this file authorizes only static/cache modes; actual mode exists for a later
separately reviewed launch.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import optimize


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_beta100_actor6_finalnorm_pet_cuttingplane_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta100-actor6-finalnorm-pet-cuttingplane-v1"

V1_TOOL = TOOLS / "probe_u468_beta100_actor6_pet_orthogonal_cuttingplane_v1.py"
V1_TOOL_SHA256 = "9b36284b2cfa8a6307c5ecfe542fe76735d74bf9f1bc6702728f46e9e94c75b2"
V1_EXPECTED_MODE = 0o444

SEED = 202608128
ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
NORM_NAMES = ("transformer.norm.weight", "transformer.norm.bias")
VARIABLE_NAMES = ACTOR_NAMES + NORM_NAMES
ACTOR_ELEMENTS = 65793
NORM_ELEMENTS = 256
COMBINED_ELEMENTS = 66049
EXPECTED_BETA_VARIABLE8_SHA256 = (
    "6e96d949db2ff2d4809f58142327e83b66a9c9e1efe1d28c980d5a80e4fe3838"
)
EXPECTED_BETA_OTHER72_SHA256 = (
    "04bc158d0dd807b6d53a7d8350271bfdf5a0d3b39d0300c326ac58122dcccfc6"
)

METRICS = ("set_exact", "hybrid_order_exact", "ordered_exact", "top1_correct")
FULLTRAIN_METRICS = METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
    "count_correct",
    "value_correct",
)
MAX_ENDPOINTS = 6
MAX_LATTICE_WRITES = 8
HARD_COMBINED_L2 = 1.0e-3
RADIUS_TOLERANCE = 1.0e-12
PET_ABS_TOLERANCE = 2.0e-9
PET_RELATIVE_TOLERANCE = 1.0e-5
DIRECT_LOSS_TOLERANCE = 1.0e-7
LINEAR_TOLERANCE = 2.0e-8
DUAL_CERT_TOLERANCE = 1.0e-12

EXPECTED_TARGETS = {
    "9417c4667da3af128f684cf5b847952b47686a329af28a94cbf01615f3de8644": {
        "source": "flg",
        "metrics": ("ordered_exact",),
    },
    "5b06cd5facc20017464010692049d50ff20ed25adadaf25a3bd773e217e1ac67": {
        "source": "core5",
        "metrics": METRICS,
    },
    "002afbcdd4f248353808c4d743785d1a0aabbdee0627486352945749a57d81b2": {
        "source": "core5",
        "metrics": METRICS,
    },
}
EXPECTED_PF_GUARD = {
    "3b8e123d82ea6b8233d3d6088830527ec635f0ac363e72bb95bbd8a99d99cd21": {
        "source": "pokemonfan",
        "metrics": METRICS,
    }
}
DENOMINATORS = {
    "flg": dict(
        set_exact=9443,
        hybrid_order_exact=9443,
        ordered_exact=9443,
        top1_correct=9426,
        context34_hybrid_order_exact=42,
        context34_ordered_exact=42,
        count_correct=9443,
        value_correct=9443,
    ),
    "pokemonfan": dict(
        set_exact=9487,
        hybrid_order_exact=9487,
        ordered_exact=9487,
        top1_correct=9450,
        context34_hybrid_order_exact=38,
        context34_ordered_exact=38,
        count_correct=9487,
        value_correct=9487,
    ),
    "core5": dict(
        set_exact=5120,
        hybrid_order_exact=5120,
        ordered_exact=5120,
        top1_correct=5106,
        context34_hybrid_order_exact=20,
        context34_ordered_exact=20,
        count_correct=5120,
        value_correct=5120,
    ),
}
MINIMUM_COUNTS = {
    "flg": dict(
        set_exact=7239,
        hybrid_order_exact=7156,
        ordered_exact=7097,
        top1_correct=7308,
        context34_hybrid_order_exact=30,
        context34_ordered_exact=30,
        count_correct=9407,
        value_correct=6439,
    ),
    "pokemonfan": dict(
        set_exact=8280,
        hybrid_order_exact=8279,
        ordered_exact=8201,
        top1_correct=8329,
        context34_hybrid_order_exact=37,
        context34_ordered_exact=37,
        count_correct=9409,
        value_correct=6585,
    ),
    "core5": dict(
        set_exact=4073,
        hybrid_order_exact=4042,
        ordered_exact=4025,
        top1_correct=4130,
        context34_hybrid_order_exact=16,
        context34_ordered_exact=16,
        count_correct=5061,
        value_correct=3822,
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
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


def vector_sha256(vector: np.ndarray, dtype: str) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype=dtype))
    return hashlib.sha256(value.tobytes()).hexdigest()


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
    path: Path,
    expected_sha256: str,
    module_name: str,
    *,
    expected_mode: int | None = None,
) -> tuple[ModuleType, dict[str, Any]]:
    payload, evidence = read_regular_bytes(
        path,
        expected_sha256,
        module_name,
        expected_mode=expected_mode,
    )
    spec = importlib.util.spec_from_file_location(
        f"_{module_name}_{sha256_bytes(payload)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


V1, V1_EVIDENCE = import_locked(
    V1_TOOL,
    V1_TOOL_SHA256,
    "frozen_actor6_pet_v1",
    expected_mode=V1_EXPECTED_MODE,
)
PET = V1.PET
torch = V1.torch
ppo = V1.ppo


class ProbeClosed(RuntimeError):
    """Expected fail-closed search termination."""


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")
    if tuple(PET.ACTOR_NAMES) != ACTOR_NAMES or PET.EXPECTED_ACTOR_ELEMENTS != ACTOR_ELEMENTS:
        raise RuntimeError("frozen actor6 manifest drift")
    if V1.ORTHOGONAL_ABS_TOLERANCE != PET_ABS_TOLERANCE:
        raise RuntimeError("frozen PET absolute tolerance drift")
    if V1.ORTHOGONAL_RELATIVE_L2_TOLERANCE != PET_RELATIVE_TOLERANCE:
        raise RuntimeError("frozen PET relative tolerance drift")


def static_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_calls = {
        "backward",
        "step",
        "save",
        "savez",
        "write",
        "write_text",
        "write_bytes",
        "touch",
        "mkdir",
        "makedirs",
        "unlink",
        "remove",
        "rmtree",
        "rename",
        "replace",
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
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = ""
        if name in forbidden_calls:
            calls.append({"line": node.lineno, "name": name})
        if name == "copy_":
            copy_sites.append(node.lineno)
        if name == "print":
            print_sites.append(node.lineno)
        dotted_call = ast.unparse(node.func).lower()
        if dotted_call.startswith("torch.optim") or ".optimizer" in dotted_call:
            optimizer_sites.append(node.lineno)
    main_definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    standalone_run_probe_calls = []
    if len(main_definitions) == 1:
        standalone_run_probe_calls = [
            node
            for node in ast.walk(main_definitions[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_probe"
        ]
    standalone_consumer_none = (
        len(standalone_run_probe_calls) == 1
        and any(
            keyword.arg == "candidate_consumer"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is None
            for keyword in standalone_run_probe_calls[0].keywords
        )
    )
    signature = inspect.signature(run_probe)
    solver_self_test = solver_synthetic_self_test()
    checks = {
        "ast_parse": True,
        "no_write_backward_step_save_or_network": not (calls or imports or write_flags),
        "no_optimizer": not optimizer_sites,
        "single_ram_copy_site": len(copy_sites) == 1,
        "single_stdout_print_site": len(print_sites) == 1,
        "combined_scope_exact_8_tensors_66049": (
            len(VARIABLE_NAMES) == 8
            and ACTOR_ELEMENTS + NORM_ELEMENTS == COMBINED_ELEMENTS == 66049
        ),
        "other_state_exact_72_contract": len(V1.PET.CHECKPOINT_SPECS) == 5,
        "embedded_pet_tolerances_exact": (
            PET_ABS_TOLERANCE == 2.0e-9 and PET_RELATIVE_TOLERANCE == 1.0e-5
        ),
        "one_hard_combined_radius": HARD_COMBINED_L2 == 1.0e-3,
        "max_endpoints_exact_6": MAX_ENDPOINTS == 6,
        "max_lattice_writes_at_most_8": MAX_LATTICE_WRITES <= 8,
        "minimal_positive_bf16_cut_constructor_present": callable(positive_bf16_quantum),
        "value_sign_zero_positive_bf16_quantum_exact": (
            zero_positive_bf16_q() == 2.0**-133
        ),
        "row_scaled_cholesky_nnls_solver_present": callable(solve_min_norm_endpoint),
        "dual_solver_feasible_and_radius_certificate_self_test": solver_self_test[
            "pass"
        ],
        "dynamic_fulltrain_gate_present": callable(fulltrain_behavior_gate),
        "candidate_consumer_default_none": signature.parameters["candidate_consumer"].default is None,
        "standalone_candidate_consumer_none": standalone_consumer_none,
    }
    if not all(checks.values()):
        raise RuntimeError(f"static audit failed: {checks}")
    return {
        "status": "static_combined_probe_audit_passed",
        "checks": checks,
        "forbidden_call_hits": calls,
        "forbidden_import_hits": imports,
        "forbidden_os_write_flags": write_flags,
        "optimizer_sites": optimizer_sites,
        "ram_copy_sites": copy_sites,
        "stdout_print_sites": print_sites,
        "solver_synthetic_self_test": solver_self_test,
    }


def build_combined_geometry(basis: Mapping[str, Any]) -> dict[str, Any]:
    q_actor = np.ascontiguousarray(basis["q"].numpy(), dtype=np.float64)
    if q_actor.shape != (ACTOR_ELEMENTS, 3):
        raise RuntimeError("actor PET Q shape drift")
    q = np.zeros((COMBINED_ELEMENTS, 3), dtype=np.float64)
    q[:ACTOR_ELEMENTS] = q_actor
    gram = q.T @ q
    residual = float(np.max(np.abs(gram - np.eye(3))))
    if residual > 1.0e-12:
        raise RuntimeError("embedded PET Q is not orthonormal")
    p_state = basis["payloads"]["P"]["model_state_dict"]
    if tuple(ACTOR_NAMES) != tuple(PET.ACTOR_NAMES):
        raise RuntimeError("actor order drift")
    variable = {name: p_state[name] for name in VARIABLE_NAMES}
    other_names = sorted(set(p_state) - set(VARIABLE_NAMES))
    other = {name: p_state[name] for name in other_names}
    variable_elements = sum(int(value.numel()) for value in variable.values())
    checks = {
        "combined_elements_exact": variable_elements == COMBINED_ELEMENTS,
        "variable_tensor_count_exact_8": len(variable) == 8,
        "other_tensor_count_exact_72": len(other) == 72,
        "variable_all_fp32": all(value.dtype == torch.float32 for value in variable.values()),
        "variable_sha_exact": PET.frozen.model_state_sha256(variable)
        == EXPECTED_BETA_VARIABLE8_SHA256,
        "other72_sha_exact": PET.frozen.model_state_sha256(other)
        == EXPECTED_BETA_OTHER72_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"combined beta100 state geometry drift: {checks}")
    return {
        "q": q,
        "q_actor": q_actor,
        "report": {
            "variable_names": list(VARIABLE_NAMES),
            "actor_elements": ACTOR_ELEMENTS,
            "finalnorm_elements": NORM_ELEMENTS,
            "combined_elements": COMBINED_ELEMENTS,
            "embedded_q_shape": list(q.shape),
            "embedded_q_float64_sha256": vector_sha256(q.reshape(-1), "<f8"),
            "orthonormality_max_abs_residual": residual,
            "pet_span": "frozen actor6 P/E/T with zero FinalNorm rows",
            "beta_variable8_sha256": EXPECTED_BETA_VARIABLE8_SHA256,
            "beta_other72_sha256": EXPECTED_BETA_OTHER72_SHA256,
            "checks": checks,
        },
    }


def project_combined(vector: np.ndarray, q: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (COMBINED_ELEMENTS,):
        raise RuntimeError(f"combined vector shape drift: {value.shape}")
    result = value - q @ (q.T @ value)
    if not bool(np.all(np.isfinite(result))):
        raise FloatingPointError("nonfinite combined PET projection")
    return result


def endpoint_audit(vector: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    value = np.asarray(vector, dtype=np.float64)
    components = q.T @ value
    combined_l2 = float(np.linalg.norm(value))
    actor_l2 = float(np.linalg.norm(value[:ACTOR_ELEMENTS]))
    norm_l2 = float(np.linalg.norm(value[ACTOR_ELEMENTS:]))
    maximum = float(np.max(np.abs(components), initial=0.0))
    relative = float(np.linalg.norm(components)) / actor_l2 if actor_l2 > 0.0 else 0.0
    return {
        "combined_l2": combined_l2,
        "actor6_l2": actor_l2,
        "finalnorm_l2": norm_l2,
        "pet_components": [float(item) for item in components],
        "pet_component_max_abs": maximum,
        "pet_component_relative_to_actor_l2": relative,
        "pet_abs_pass": maximum <= PET_ABS_TOLERANCE,
        "pet_relative_pass": relative <= PET_RELATIVE_TOLERANCE,
        "combined_radius_pass": combined_l2 <= HARD_COMBINED_L2 + RADIUS_TOLERANCE,
        "pass": (
            maximum <= PET_ABS_TOLERANCE
            and relative <= PET_RELATIVE_TOLERANCE
            and combined_l2 <= HARD_COMBINED_L2 + RADIUS_TOLERANCE
        ),
    }


def configure_combined(model: Any) -> dict[str, Any]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    result = {}
    for name in VARIABLE_NAMES:
        parameter = named.get(name)
        if parameter is None or parameter.dtype != torch.float32:
            raise RuntimeError(f"missing/non-FP32 combined parameter: {name}")
        parameter.requires_grad_(True)
        result[name] = parameter
    active = {name for name, value in model.named_parameters() if value.requires_grad}
    if active != set(VARIABLE_NAMES):
        raise RuntimeError(f"combined trainable scope drift: {sorted(active)}")
    if sum(int(value.numel()) for value in result.values()) != COMBINED_ELEMENTS:
        raise RuntimeError("combined runtime element-count drift")
    return result


def combined_delta_from_model(
    parameters: Mapping[str, Any], beta_variables: Mapping[str, Any]
) -> np.ndarray:
    pieces = [
        (
            parameters[name].detach().cpu().float()
            - beta_variables[name].detach().cpu().float()
        )
        .reshape(-1)
        .double()
        .numpy()
        for name in VARIABLE_NAMES
    ]
    result = np.concatenate(pieces).astype(np.float64, copy=False)
    if result.shape != (COMBINED_ELEMENTS,):
        raise RuntimeError("combined actual delta shape drift")
    return result


def set_combined_delta(
    parameters: Mapping[str, Any],
    beta_variables: Mapping[str, Any],
    delta: np.ndarray,
) -> None:
    value = np.asarray(delta, dtype=np.float64)
    if value.shape != (COMBINED_ELEMENTS,) or not bool(np.all(np.isfinite(value))):
        raise RuntimeError("invalid combined overlay vector")
    offset = 0
    with torch.no_grad():
        for name in VARIABLE_NAMES:
            parameter = parameters[name]
            count = int(parameter.numel())
            piece = torch.from_numpy(
                value[offset : offset + count].astype(np.float32, copy=True)
            ).to(device=parameter.device, dtype=parameter.dtype).reshape(parameter.shape)
            parameter.copy_(beta_variables[name] + piece)
            offset += count
    if offset != COMBINED_ELEMENTS:
        raise RuntimeError("combined overlay slice drift")


def materialize_quantized_endpoint(
    parameters: Mapping[str, Any],
    beta_variables: Mapping[str, Any],
    planned: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = project_combined(planned, q)
    target_audit = endpoint_audit(target, q)
    if not target_audit["pass"]:
        raise ProbeClosed("planned combined endpoint failed PET/radius")
    request = target.copy()
    seen: dict[tuple[str, str], int] = {}
    iterations = []
    for iteration in range(1, MAX_LATTICE_WRITES + 1):
        set_combined_delta(parameters, beta_variables, request)
        actual = combined_delta_from_model(parameters, beta_variables)
        audit = endpoint_audit(actual, q)
        request_sha = vector_sha256(request, "<f8")
        actual_sha = vector_sha256(actual, "<f4")
        key = (request_sha, actual_sha)
        repeated = seen.get(key)
        seen[key] = iteration
        iterations.append(
            {
                "iteration": iteration,
                "request_float64_sha256": request_sha,
                "request_combined_l2": float(np.linalg.norm(request)),
                "actual_float32_sha256": actual_sha,
                "actual": audit,
                "repeated_request_actual_from_iteration": repeated,
                "pass": audit["pass"],
            }
        )
        if audit["pass"]:
            return actual, {
                "status": "bounded_span_only_quantization_passed",
                "rule": "u_next_actor=u_actor-Q_actor(Q_actor^T actual_actor); norm request unchanged",
                "maximum_writes": MAX_LATTICE_WRITES,
                "iterations": iterations,
            }
        if repeated is not None:
            break
        span_error = q @ (q.T @ actual)
        request = request - span_error
        if float(np.linalg.norm(request)) > HARD_COMBINED_L2 + RADIUS_TOLERANCE:
            break
    raise ProbeClosed("combined FP32 lattice PET correction failed within 8 writes")


def positive_bf16_quantum(value_a: Any, value_b: Any) -> float:
    if value_a.dtype != torch.bfloat16 or value_b.dtype != torch.bfloat16:
        raise RuntimeError("positive quantum requires native BF16 scalars")
    spacings = []
    for value in (value_a, value_b):
        spacings.extend(
            (
                torch.nextafter(value, torch.full_like(value, float("inf"))) - value,
                value - torch.nextafter(value, torch.full_like(value, float("-inf"))),
            )
        )
    result = max(float(item.float()) for item in spacings)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("invalid native-BF16 positive quantum")
    return result


def zero_positive_bf16_q() -> float:
    zero = torch.zeros((), dtype=torch.bfloat16)
    result = float(
        torch.nextafter(zero, torch.full_like(zero, float("inf"))).float()
    )
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("invalid native-BF16 positive quantum around zero")
    return result


def flatten_projected_gradient(
    scalar: Any,
    parameters: Mapping[str, Any],
    q: np.ndarray,
    *,
    retain_graph: bool,
) -> np.ndarray:
    if any(parameters[name].grad is not None for name in VARIABLE_NAMES):
        raise RuntimeError("unexpected accumulated .grad before autograd.grad")
    values = torch.autograd.grad(
        scalar,
        tuple(parameters[name] for name in VARIABLE_NAMES),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
        materialize_grads=False,
    )
    pieces = []
    for name, value in zip(VARIABLE_NAMES, values):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite combined gradient: {name}")
        pieces.append(value.detach().reshape(-1).cpu().double().numpy())
    flat = np.concatenate(pieces).astype(np.float64, copy=False)
    if flat.shape != (COMBINED_ELEMENTS,):
        raise RuntimeError("combined gradient shape drift")
    projected = project_combined(flat, q)
    if any(parameters[name].grad is not None for name in VARIABLE_NAMES):
        raise RuntimeError("autograd.grad unexpectedly populated .grad")
    return projected


def constraint_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    base: tuple[Any, ...] = (
        str(item["batch_key"]),
        int(item["local_index"]),
        str(item["kind"]),
    )
    if item["kind"] in {"policy", "count"}:
        return base + (int(item["positive_index"]), int(item["negative_index"]))
    if item["kind"] == "value":
        return base + (int(item["sign_multiplier"]),)
    raise RuntimeError(f"unknown constraint kind: {item['kind']}")


def add_constraint(
    active: dict[tuple[Any, ...], dict[str, Any]], item_source: Mapping[str, Any]
) -> dict[str, Any]:
    item = dict(item_source)
    item["origins"] = list(item.get("origins", ()))
    key = constraint_key(item)
    if key not in active:
        active[key] = item
        return {"operation": "added", "key": list(key), "threshold": item["threshold"]}
    old = active[key]
    previous = float(old["threshold"])
    old["threshold"] = max(previous, float(item["threshold"]))
    for origin in item["origins"]:
        if origin not in old["origins"]:
            old["origins"].append(origin)
    return {
        "operation": "strengthened" if old["threshold"] > previous else "deduplicated",
        "key": list(key),
        "previous": previous,
        "threshold": old["threshold"],
    }


def constraint_margin(outputs: Mapping[str, Any], item: Mapping[str, Any]) -> Any:
    row = int(item["local_index"])
    if item["kind"] == "policy":
        logits = outputs["policy_logits"]
        return logits[row, int(item["positive_index"])].float() - logits[
            row, int(item["negative_index"])
        ].float()
    if item["kind"] == "count":
        logits = outputs["count_logits"]
        return logits[row, int(item["positive_index"])].float() - logits[
            row, int(item["negative_index"])
        ].float()
    if item["kind"] == "value":
        return outputs["value_logits"][row].float() * int(item["sign_multiplier"])
    raise RuntimeError(f"unknown constraint kind: {item['kind']}")


def cpu_batches_tensor_exact(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return set(left) == set(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def policy_pair_from_record(
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    record: Mapping[str, Any],
    row: int,
    metric: str,
) -> tuple[int, int] | None:
    expert = [int(value) for value in record["expert_order"]]
    predicted = [int(value) for value in record["predicted_order"]]
    if metric == "top1_correct":
        logits = outputs_cpu["policy_logits"][row].float()
        targets = cpu_batch["targets"][row].bool()
        positives = targets.nonzero(as_tuple=False).squeeze(1)
        negatives = (~targets).nonzero(as_tuple=False).squeeze(1)
        if positives.numel() == 0 or negatives.numel() == 0:
            return None
        return (
            int(positives[torch.argmax(logits[positives])]),
            int(negatives[torch.argmax(logits[negatives])]),
        )
    if metric == "set_exact" or (
        metric == "hybrid_order_exact" and int(record["context"]) != 34
    ):
        expected_set = set(expert)
        predicted_set = set(predicted)
        missing = [value for value in expert if value not in predicted_set]
        extra = [value for value in predicted if value not in expected_set]
        return (missing[0], extra[0]) if missing and extra else None
    for expected, observed in zip(expert, predicted):
        if expected != observed:
            return expected, observed
    return None


def policy_constraint(
    *,
    batch_key: str,
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    record: Mapping[str, Any],
    row: int,
    identity: Mapping[str, Any],
    metric: str,
    role: str,
    origin: str,
) -> dict[str, Any] | None:
    pair = policy_pair_from_record(cpu_batch, outputs_cpu, record, row, metric)
    if pair is None or pair[0] == pair[1]:
        return None
    logits = outputs_cpu["policy_logits"]
    threshold = positive_bf16_quantum(logits[row, pair[0]], logits[row, pair[1]])
    return {
        "kind": "policy",
        "batch_key": batch_key,
        "cpu_batch": cpu_batch,
        "local_index": row,
        "identity": dict(identity),
        "role": role,
        "metric": metric,
        "positive_index": pair[0],
        "negative_index": pair[1],
        "threshold": threshold,
        "threshold_basis": "one_current_native_bf16_positive_quantum",
        "origins": [origin],
    }


def initialize_selected_constraints(
    descriptors: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    active: dict[tuple[Any, ...], dict[str, Any]] = {}
    for descriptor in descriptors:
        key = str(descriptor["batch_key"])
        row = int(descriptor["local_index"])
        current_record = baseline[key]["official_rows"][row]
        if descriptor["role"] == "target_transition":
            metrics = [
                metric
                for metric in descriptor["required_metrics"]
                if not current_record["flags"][metric]
            ]
            for metric in metrics:
                item = policy_constraint(
                    batch_key=key,
                    cpu_batch=descriptor["cpu_batch"],
                    outputs_cpu=baseline[key]["outputs_cpu"],
                    record=current_record,
                    row=row,
                    identity=descriptor["identity"],
                    metric=metric,
                    role=str(descriptor["role"]),
                    origin=f"initial_target:{metric}",
                )
                if item is None:
                    raise ProbeClosed("initial target has no actor/finalnorm policy pair")
                add_constraint(active, item)
            continue
        shell = dict(descriptor)
        shell["cpu_batch"] = descriptor["cpu_batch"]
        for positive, negative, support in V1.guard_support_pairs(shell, baseline):
            logits = baseline[key]["outputs_cpu"]["policy_logits"]
            threshold = positive_bf16_quantum(
                logits[row, positive], logits[row, negative]
            )
            add_constraint(
                active,
                {
                    "kind": "policy",
                    "batch_key": key,
                    "cpu_batch": descriptor["cpu_batch"],
                    "local_index": row,
                    "identity": dict(descriptor["identity"]),
                    "role": str(descriptor["role"]),
                    "metric": "support",
                    "positive_index": positive,
                    "negative_index": negative,
                    "threshold": threshold,
                    "threshold_basis": "one_current_native_bf16_positive_quantum",
                    "origins": [f"initial_guard:{support}"],
                },
            )
    if not active:
        raise ProbeClosed("no initial combined constraints")
    return active


def collect_constraint_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    active: Mapping[tuple[Any, ...], Mapping[str, Any]],
    q: np.ndarray,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for key in sorted(active):
        item = active[key]
        grouped.setdefault(str(item["batch_key"]), []).append(item)
    gradients = []
    rhs = []
    records = []
    for batch_key in sorted(grouped):
        items = grouped[batch_key]
        cpu_batch = items[0]["cpu_batch"]
        if any(
            not cpu_batches_tensor_exact(cpu_batch, item["cpu_batch"])
            for item in items
        ):
            raise RuntimeError("same active batch key has nonidentical canonical tensors")
        batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
        outputs = ppo.model_forward(model, batch, device)
        if any(value.dtype != torch.bfloat16 for value in outputs.values()):
            raise RuntimeError("constraint forward is not native BF16")
        margins = [constraint_margin(outputs, item) for item in items]
        for index, (item, margin) in enumerate(zip(items, margins)):
            gradient = flatten_projected_gradient(
                margin,
                parameters,
                q,
                retain_graph=index + 1 < len(items),
            )
            observed = float(margin.detach().cpu())
            required = float(item["threshold"]) - observed
            gradients.append(gradient)
            rhs.append(required)
            records.append(
                {
                    "key": list(constraint_key(item)),
                    "gradient_l2": float(np.linalg.norm(gradient)),
                    "observed_margin": observed,
                    "threshold": float(item["threshold"]),
                    "correction_rhs": required,
                }
            )
    return (
        np.stack(gradients),
        np.asarray(rhs, dtype=np.float64),
        {"rows": len(records), "records": records},
    )


def collect_loss_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    cache: Mapping[str, Any],
    q: np.ndarray,
    device: Any,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in cache["direct_union"].items()
    }
    outputs = ppo.model_forward(model, batch, device)
    if outputs["policy_logits"].dtype != torch.bfloat16:
        raise RuntimeError("Direct512 forward is not native BF16")
    per_row = PET.frozen.ordered_nll_per_row(outputs, batch)
    weights = batch["sample_weights"].float() * torch.where(
        batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
        torch.full_like(
            batch["sample_weights"].float(), PET.aggregate.ORDER_CONTEXT_WEIGHT
        ),
        torch.ones_like(batch["sample_weights"].float()),
    )
    scalars = []
    names = list(PET.LOSS_NAMES)
    losses: dict[str, float] = {}
    for name in names:
        mask = cache["loss_masks"][name].to(device=device, dtype=torch.bool)
        scalar = (per_row[mask] * weights[mask]).sum() / weights[mask].sum()
        if not bool(torch.isfinite(scalar)):
            raise FloatingPointError(f"nonfinite Direct512 loss: {name}")
        scalars.append(scalar)
        losses[name] = float(scalar.detach().cpu())
    gradients = []
    records = []
    for index, (name, scalar) in enumerate(zip(names, scalars)):
        gradient = flatten_projected_gradient(
            scalar,
            parameters,
            q,
            retain_graph=index + 1 < len(scalars),
        )
        gradients.append(gradient)
        records.append(
            {
                "name": name,
                "loss": losses[name],
                "projected_gradient_l2": float(np.linalg.norm(gradient)),
            }
        )
    return np.stack(gradients), losses, {"rows": len(records), "records": records}


def solve_min_norm_endpoint(
    pair_gradients: np.ndarray,
    pair_rhs: np.ndarray,
    loss_gradients: np.ndarray,
    loss_rhs: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Row-scaled Cholesky+NNLS dual solution and radius certificate."""
    if pair_gradients.ndim != 2 or pair_rhs.shape != (pair_gradients.shape[0],):
        raise RuntimeError("pair matrix/RHS shape drift")
    if loss_gradients.shape != (13, COMBINED_ELEMENTS) or loss_rhs.shape != (13,):
        raise RuntimeError("Direct512 matrix/RHS shape drift")
    matrix = np.concatenate((pair_gradients, -loss_gradients), axis=0)
    correction_rhs = np.concatenate((pair_rhs, loss_rhs))
    endpoint_rhs = correction_rhs + matrix @ current
    row_norms = np.linalg.norm(matrix, axis=1)
    if bool(np.any(~np.isfinite(row_norms))) or bool(np.any(row_norms <= 0.0)):
        raise ProbeClosed("zero/nonfinite combined constraint row")
    scaled_matrix = matrix / row_norms[:, None]
    scaled_rhs = endpoint_rhs / row_norms
    gram = scaled_matrix @ scaled_matrix.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    if float(eigenvalues[0]) <= 0.0:
        return None, {
            "status": "closed_singular_row_scaled_gram_uncertified",
            "rows": int(matrix.shape[0]),
            "eigen_min": float(eigenvalues[0]),
            "eigen_max": float(eigenvalues[-1]),
        }
    try:
        cholesky = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        return None, {
            "status": "closed_cholesky_failure_uncertified",
            "rows": int(matrix.shape[0]),
            "eigen_min": float(eigenvalues[0]),
            "eigen_max": float(eigenvalues[-1]),
        }
    target = np.linalg.solve(cholesky, scaled_rhs)
    dual, nnls_residual = optimize.nnls(
        cholesky.T,
        target,
        maxiter=max(1000, 20 * int(matrix.shape[0])),
        atol=1.0e-14,
    )
    endpoint = scaled_matrix.T @ dual
    primal_residual = scaled_matrix @ endpoint - scaled_rhs
    dual_gradient = gram @ dual - scaled_rhs
    active = dual > 1.0e-12
    complementarity = dual * primal_residual
    primal_objective = 0.5 * float(endpoint @ endpoint)
    dual_objective = float(scaled_rhs @ dual - 0.5 * (endpoint @ endpoint))
    gap = primal_objective - dual_objective
    radius_objective = 0.5 * HARD_COMBINED_L2**2
    certificate_gap = dual_objective - radius_objective
    kkt_pass = (
        float(primal_residual.min()) >= -LINEAR_TOLERANCE
        and float(dual.min(initial=0.0)) >= 0.0
        and float(np.max(np.abs(complementarity), initial=0.0)) <= 1.0e-8
        and (
            not bool(np.any(active))
            or float(np.max(np.abs(dual_gradient[active]), initial=0.0)) <= 1.0e-8
        )
        and gap >= -1.0e-10
        and gap <= 1.0e-8
    )
    report = {
        "status": "dual_solution_computed",
        "solver": "row-scaled FP64 Gram Cholesky plus scipy.optimize.nnls",
        "rows": int(matrix.shape[0]),
        "columns": COMBINED_ELEMENTS,
        "row_scale_min": float(row_norms.min()),
        "row_scale_max": float(row_norms.max()),
        "gram_eigen_min": float(eigenvalues[0]),
        "gram_eigen_max": float(eigenvalues[-1]),
        "gram_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "nnls_residual": float(nnls_residual),
        "endpoint_l2": float(np.linalg.norm(endpoint)),
        "primal_residual_min": float(primal_residual.min()),
        "dual_lambda_min": float(dual.min(initial=0.0)),
        "complementarity_max_abs": float(
            np.max(np.abs(complementarity), initial=0.0)
        ),
        "active_dual_gradient_max_abs": (
            float(np.max(np.abs(dual_gradient[active]), initial=0.0))
            if bool(np.any(active))
            else 0.0
        ),
        "primal_objective": primal_objective,
        "dual_objective": dual_objective,
        "primal_dual_gap": gap,
        "radius_objective": radius_objective,
        "dual_radius_certificate_gap": certificate_gap,
        "dual_sha256": vector_sha256(dual, "<f8"),
        "kkt_pass": kkt_pass,
    }
    if not kkt_pass:
        report["status"] = "closed_nnls_kkt_failure_uncertified"
        return None, report
    if float(np.linalg.norm(endpoint)) > HARD_COMBINED_L2 + RADIUS_TOLERANCE:
        certified = certificate_gap > max(
            DUAL_CERT_TOLERANCE,
            100.0
            * np.finfo(np.float64).eps
            * (
                abs(float(scaled_rhs @ dual))
                + 0.5 * float(endpoint @ endpoint)
                + HARD_COMBINED_L2**2
            ),
        )
        report["status"] = (
            "certified_min_norm_exceeds_hard_radius"
            if certified
            else "closed_radius_excess_uncertified"
        )
        report["radius_infeasible_certified"] = certified
        return None, report
    report["status"] = "feasible_min_norm_endpoint"
    report["radius_infeasible_certified"] = False
    return endpoint, report


def solver_synthetic_self_test() -> dict[str, Any]:
    pair = np.zeros((1, COMBINED_ELEMENTS), dtype=np.float64)
    pair[0, 0] = 1.0
    losses = np.zeros((13, COMBINED_ELEMENTS), dtype=np.float64)
    for index in range(13):
        losses[index, index + 1] = 1.0
    current = np.zeros(COMBINED_ELEMENTS, dtype=np.float64)
    feasible, feasible_report = solve_min_norm_endpoint(
        pair,
        np.asarray([1.0e-4], dtype=np.float64),
        losses,
        np.full(13, -1.0e-7, dtype=np.float64),
        current,
    )
    infeasible, infeasible_report = solve_min_norm_endpoint(
        pair,
        np.asarray([2.0e-3], dtype=np.float64),
        losses,
        np.full(13, -1.0e-7, dtype=np.float64),
        current,
    )
    passed = (
        feasible is not None
        and feasible_report["status"] == "feasible_min_norm_endpoint"
        and abs(float(feasible[0]) - 1.0e-4) <= 1.0e-15
        and infeasible is None
        and infeasible_report["status"]
        == "certified_min_norm_exceeds_hard_radius"
        and infeasible_report["radius_infeasible_certified"] is True
    )
    return {
        "pass": passed,
        "feasible_status": feasible_report["status"],
        "feasible_endpoint_l2": feasible_report["endpoint_l2"],
        "infeasible_status": infeasible_report["status"],
        "infeasible_endpoint_l2": infeasible_report["endpoint_l2"],
        "infeasible_dual_radius_certificate_gap": infeasible_report[
            "dual_radius_certificate_gap"
        ],
    }


def active_constraint_gate(
    model: Any,
    active: Mapping[tuple[Any, ...], Mapping[str, Any]],
    device: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for key in sorted(active):
        item = active[key]
        grouped.setdefault(str(item["batch_key"]), []).append(item)
    records = []
    with torch.no_grad():
        for batch_key in sorted(grouped):
            items = grouped[batch_key]
            cpu_batch = items[0]["cpu_batch"]
            if any(
                not cpu_batches_tensor_exact(cpu_batch, item["cpu_batch"])
                for item in items
            ):
                raise RuntimeError(
                    "active gate batch key has nonidentical canonical tensors"
                )
            batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
            outputs = ppo.model_forward(model, batch, device)
            for item in items:
                margin = float(constraint_margin(outputs, item).detach().cpu())
                records.append(
                    {
                        "key": list(constraint_key(item)),
                        "margin": margin,
                        "threshold": float(item["threshold"]),
                        "pass": margin >= float(item["threshold"]),
                    }
                )
    return {
        "pass": all(item["pass"] for item in records),
        "count": len(records),
        "minimum_residual": min(
            (item["margin"] - item["threshold"] for item in records),
            default=None,
        ),
        "records": records,
    }


def _stable_identity(transition: ModuleType, identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": str(identity["panel"]),
        "member": str(identity["member"]),
        "line_index_zero_based": int(identity["line_index_zero_based"]),
        "line_sha256": str(identity["line_sha256"]),
    }


def fulltrain_behavior_gate(
    transition: ModuleType,
    model: Any,
    device: Any,
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authoritative beta->candidate gate plus current-quantum dynamic cuts."""
    cells = {
        panel: {
            metric: {cell: 0 for cell in ("cc", "cw", "wc", "ww")}
            for metric in FULLTRAIN_METRICS
        }
        for panel in transition.PANEL_ORDER
    }
    batch_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    row_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    identity_digests = {panel: hashlib.sha256() for panel in transition.PANEL_ORDER}
    behavior_mismatch = {"count": 0, "value": 0}
    dynamic: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    selected_lookup = {
        (str(item["source"]), str(item["identity"]["line_sha256"])): item
        for item in descriptors
    }
    selected_records: dict[tuple[str, str, str], str] = {}
    state_before = ppo.model_state_sha256(model)

    def callback(
        panel: str,
        cpu_batch: Mapping[str, Any],
        identities: Sequence[Mapping[str, Any]],
        raw_snapshot: Mapping[str, Any],
        beta_snapshot: Mapping[str, Any],
    ) -> None:
        del raw_snapshot
        batch_index = batch_counts[panel]
        batch_counts[panel] += 1
        row_counts[panel] += len(identities)
        batch_key = f"{panel}:B{batch_index:03d}"
        for identity in identities:
            identity_digests[panel].update(
                transition.canonical_json(_stable_identity(transition, identity))
            )
        batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
        with torch.no_grad():
            outputs = ppo.model_forward(model, batch, device)
            actions, _, _, _ = ppo.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=False,
            )
        if set(outputs) != {"policy_logits", "count_logits", "value_logits"}:
            raise RuntimeError("fulltrain candidate output-key drift")
        if any(value.dtype != torch.bfloat16 for value in outputs.values()):
            raise RuntimeError("fulltrain candidate output is not native BF16")
        outputs_cpu = {name: value.detach().cpu().contiguous() for name, value in outputs.items()}
        for row, identity in enumerate(identities):
            record = V1.official_row_record(cpu_batch, outputs_cpu, actions, row)
            context = int(identity["context"])
            nonempty = int(cpu_batch["action_counts"][row]) > 0
            candidate_flags = dict(record["flags"])
            candidate_flags["count_correct"] = (
                int(record["predicted_count"]) == int(cpu_batch["action_counts"][row])
            )
            candidate_value_sign = bool(outputs_cpu["value_logits"][row] >= 0)
            candidate_flags["value_correct"] = candidate_value_sign == bool(
                cpu_batch["win_targets"][row]
            )
            beta_prediction = beta_snapshot["predictions"][row]
            beta_count = int(beta_prediction["predicted_count"])
            beta_value_sign = bool(beta_prediction["value_sign"])
            behavior_mismatch["count"] += int(record["predicted_count"] != beta_count)
            behavior_mismatch["value"] += int(candidate_value_sign != beta_value_sign)
            line_sha = str(identity["line_sha256"])
            for metric in FULLTRAIN_METRICS:
                if metric == "top1_correct" and not nonempty:
                    continue
                if metric.startswith("context34_") and context != 34:
                    continue
                base_metric = metric.removeprefix("context34_")
                beta_correct = bool(beta_snapshot["flags"][row][base_metric])
                candidate_correct = bool(candidate_flags[base_metric])
                cell = (
                    ("c" if beta_correct else "w")
                    + ("c" if candidate_correct else "w")
                )
                cells[panel][metric][cell] += 1
                selected_records[(panel, line_sha, metric)] = cell
                if (
                    base_metric in METRICS
                    and beta_correct
                    and not candidate_correct
                    and not metric.startswith("context34_")
                ):
                    item = policy_constraint(
                        batch_key=batch_key,
                        cpu_batch=cpu_batch,
                        outputs_cpu=outputs_cpu,
                        record=record,
                        row=row,
                        identity=identity,
                        metric=base_metric,
                        role="fulltrain_policy_cw",
                        origin=f"dynamic_fulltrain_cw:{base_metric}",
                    )
                    if item is None:
                        unresolved.append(
                            {"identity": dict(identity), "kind": "policy", "metric": base_metric}
                        )
                    else:
                        dynamic.append(item)
            selected = selected_lookup.get((panel, line_sha))
            if selected is not None:
                for metric in selected["required_metrics"]:
                    if not candidate_flags[metric]:
                        item = policy_constraint(
                            batch_key=batch_key,
                            cpu_batch=cpu_batch,
                            outputs_cpu=outputs_cpu,
                            record=record,
                            row=row,
                            identity=identity,
                            metric=str(metric),
                            role=str(selected["role"]),
                            origin=f"dynamic_selected:{metric}",
                        )
                        if item is None:
                            unresolved.append(
                                {"identity": dict(identity), "kind": "selected", "metric": metric}
                            )
                        else:
                            dynamic.append(item)
            if int(record["predicted_count"]) != beta_count:
                fixed = bool(cpu_batch["min_counts"][row] == cpu_batch["max_counts"][row])
                if fixed:
                    unresolved.append(
                        {"identity": dict(identity), "kind": "fixed_count_behavior_drift"}
                    )
                else:
                    positive = beta_count
                    negative = int(record["predicted_count"])
                    logits = outputs_cpu["count_logits"]
                    dynamic.append(
                        {
                            "kind": "count",
                            "batch_key": batch_key,
                            "cpu_batch": cpu_batch,
                            "local_index": row,
                            "identity": dict(identity),
                            "role": "fulltrain_count_behavior",
                            "metric": "predicted_count_exact_beta100",
                            "positive_index": positive,
                            "negative_index": negative,
                            "threshold": positive_bf16_quantum(
                                logits[row, positive], logits[row, negative]
                            ),
                            "threshold_basis": "one_current_native_bf16_positive_quantum",
                            "origins": ["dynamic_count_behavior"],
                        }
                    )
            if candidate_value_sign != beta_value_sign:
                dynamic.append(
                    {
                        "kind": "value",
                        "batch_key": batch_key,
                        "cpu_batch": cpu_batch,
                        "local_index": row,
                        "identity": dict(identity),
                        "role": "fulltrain_value_behavior",
                        "metric": "value_sign_exact_beta100",
                        "sign_multiplier": 1 if beta_value_sign else -1,
                        "threshold": zero_positive_bf16_q(),
                        "threshold_basis": "zero_to_nextafter_positive_native_bf16_quantum",
                        "origins": ["dynamic_value_behavior:zero_positive_bf16_q"],
                    }
                )

    report = transition.run_transition_audit(row_consumer=callback, candidate_consumer=None)
    transition_closure = V1.transition_absolute_closure(report)
    expected_batches = {
        panel: math.ceil(int(transition.EXPECTED_TRAIN[panel]["rows"]) / 256)
        for panel in transition.PANEL_ORDER
    }
    identity_sha = {
        panel: identity_digests[panel].hexdigest() for panel in transition.PANEL_ORDER
    }
    coverage = (
        batch_counts == expected_batches
        and identity_sha == transition.EXPECTED_IDENTITY_SHA256
        and sum(row_counts.values()) == 24050
    )
    records = []
    for panel in transition.PANEL_ORDER:
        for metric in FULLTRAIN_METRICS:
            item = cells[panel][metric]
            denominator = sum(item.values())
            beta_correct = item["cc"] + item["cw"]
            candidate_correct = item["cc"] + item["wc"]
            expected_beta = int(transition.EXPECTED_AGGREGATES[panel]["beta100"][metric])
            passed = (
                denominator == DENOMINATORS[panel][metric]
                and beta_correct == expected_beta
                and item["cw"] == 0
                and candidate_correct >= MINIMUM_COUNTS[panel][metric]
                and item["wc"] - item["cw"] == candidate_correct - beta_correct
            )
            records.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "cells": dict(item),
                    "denominator": denominator,
                    "expected_denominator": DENOMINATORS[panel][metric],
                    "beta_correct": beta_correct,
                    "expected_beta_correct": expected_beta,
                    "candidate_correct": candidate_correct,
                    "minimum_candidate_correct": MINIMUM_COUNTS[panel][metric],
                    "pass": passed,
                }
            )
    selected_audit = []
    for descriptor in descriptors:
        panel = str(descriptor["source"])
        sha = str(descriptor["identity"]["line_sha256"])
        expected = "wc" if descriptor["role"] == "target_transition" else "cc"
        for metric in descriptor["required_metrics"]:
            observed = selected_records.get((panel, sha, str(metric)))
            selected_audit.append(
                {
                    "identity": descriptor["identity"],
                    "role": descriptor["role"],
                    "metric": metric,
                    "expected_cell": expected,
                    "observed_cell": observed,
                    "pass": observed == expected,
                }
            )
    state_after = ppo.model_state_sha256(model)
    passed = (
        report.get("status") == "complete_train_transition_audit_passed"
        and transition_closure["pass"]
        and coverage
        and len(records) == 24
        and all(item["pass"] for item in records)
        and behavior_mismatch == {"count": 0, "value": 0}
        and len(selected_audit) == 397
        and all(item["pass"] for item in selected_audit)
        and not unresolved
        and state_after == state_before
    )
    return (
        {
            "pass": passed,
            "coverage_pass": coverage,
            "batch_counts": batch_counts,
            "row_counts": row_counts,
            "identity_stream_sha256": identity_sha,
            "metric_records": records,
            "metric_record_count_exact_24": len(records) == 24,
            "count_value_behavior_mismatch_rows": behavior_mismatch,
            "selected_obligations": selected_audit,
            "selected_obligation_count_exact_397": len(selected_audit) == 397,
            "unresolved": unresolved,
            "dynamic_cut_count": len(dynamic),
            "candidate_state_sha256_before": state_before,
            "candidate_state_sha256_after": state_after,
            "candidate_state_unchanged_by_fulltrain": state_after == state_before,
            "transition_absolute_closure": transition_closure,
        },
        dynamic,
    )


def canonical_constraint_ledger(
    active: Mapping[tuple[Any, ...], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(active):
        item = active[key]
        result.append(
            {
                "key": list(key),
                "identity": item["identity"],
                "role": item["role"],
                "metric": item["metric"],
                "threshold": float(item["threshold"]),
                "threshold_basis": item["threshold_basis"],
                "origins": list(item["origins"]),
            }
        )
    return result


def run_probe(
    transition_context: Mapping[str, Any] | None = None,
    candidate_consumer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if candidate_consumer is not None and not callable(candidate_consumer):
        raise TypeError("candidate_consumer must be callable or None")
    if transition_context is None:
        transition, transition_evidence = V1.load_transition_module()
        transition_context = V1.build_actual_transition_context(transition)
    else:
        transition_evidence = {"injected_context": True}
        transition = transition_context.get("_transition_module")
    if not isinstance(transition, ModuleType):
        raise RuntimeError("actual combined probe requires hash-bound transition module")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("actual combined probe requires native CUDA BF16")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    context_audit = V1.validate_transition_context(transition_context)
    model = transition_context["beta_model"]
    beta_checkpoint = transition_context["beta_checkpoint"]
    descriptors = transition_context["descriptors"]
    batches = transition_context["selected_batches"]
    if ppo.model_state_sha256(model) != transition.BETA_MODEL_SHA256:
        raise RuntimeError("combined runtime beta100 state SHA drift")
    basis = PET.build_basis_geometry()
    geometry = build_combined_geometry(basis)
    q = geometry["q"]
    cache = PET.build_cache_context(basis["payloads"]["R"])
    parameters = configure_combined(model)
    beta_variables = {
        name: parameters[name].detach().clone() for name in VARIABLE_NAMES
    }
    other_names = sorted(set(model.state_dict()) - set(VARIABLE_NAMES))
    beta_other_sha = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in other_names}
    )
    if beta_other_sha != EXPECTED_BETA_OTHER72_SHA256:
        raise RuntimeError("runtime beta100 other72 SHA drift")
    beta_full_sha = ppo.model_state_sha256(model)
    baseline = V1.snapshot_batches(model, batches, device)
    for descriptor in descriptors:
        observed = baseline[str(descriptor["batch_key"])]["official_rows"][
            int(descriptor["local_index"])
        ]["flags"]
        if any(
            bool(observed[metric]) != bool(descriptor["beta_flags"][metric])
            for metric in METRICS
        ):
            raise RuntimeError("combined independent beta selected-flag drift")
    baseline_losses = V1.evaluate_direct_losses(
        model, cache["direct_union"], cache["loss_masks"], device
    )
    active = initialize_selected_constraints(descriptors, baseline)
    current = np.zeros(COMBINED_ELEMENTS, dtype=np.float64)
    endpoints = 0
    status = "closed_no_candidate"
    close_reason = "maximum_endpoint_budget_reached"
    ledger = []
    success = False
    consumer_called = False
    final_integrity: dict[str, Any] = {}
    try:
        while endpoints < MAX_ENDPOINTS:
            pair_g, pair_rhs, pair_audit = collect_constraint_gradients(
                model, parameters, active, q, device
            )
            loss_g, current_losses, loss_audit = collect_loss_gradients(
                model, parameters, cache, q, device
            )
            loss_rhs = np.asarray(
                [
                    current_losses[name]
                    - baseline_losses[name]
                    - DIRECT_LOSS_TOLERANCE
                    for name in PET.LOSS_NAMES
                ],
                dtype=np.float64,
            )
            endpoint, solver = solve_min_norm_endpoint(
                pair_g, pair_rhs, loss_g, loss_rhs, current
            )
            if endpoint is None:
                close_reason = str(solver["status"])
                ledger.append(
                    {
                        "endpoint_index": endpoints + 1,
                        "kind": "closed_before_endpoint",
                        "solver": solver,
                        "pair_gradients": pair_audit,
                        "loss_gradients": loss_audit,
                    }
                )
                break
            planned = project_combined(endpoint, q)
            actual, quantization = materialize_quantized_endpoint(
                parameters, beta_variables, planned, q
            )
            current = actual.copy()
            endpoints += 1
            other_sha = PET.frozen.model_state_sha256(
                {name: model.state_dict()[name] for name in other_names}
            )
            if other_sha != beta_other_sha:
                raise RuntimeError("other72 state changed during combined endpoint")
            active_gate = active_constraint_gate(model, active, device)
            actual_losses = V1.evaluate_direct_losses(
                model, cache["direct_union"], cache["loss_masks"], device
            )
            direct_gate = V1.direct_loss_gate(baseline_losses, actual_losses)
            full_gate, new_constraints = fulltrain_behavior_gate(
                transition, model, device, descriptors
            )
            additions = [add_constraint(active, item) for item in new_constraints]
            ledger.append(
                {
                    "endpoint_index": endpoints,
                    "kind": "combined_actor6_finalnorm_endpoint",
                    "solver": solver,
                    "pair_gradients": pair_audit,
                    "loss_gradients": loss_audit,
                    "planned": endpoint_audit(planned, q),
                    "actual": endpoint_audit(actual, q),
                    "actual_float32_sha256": vector_sha256(actual, "<f4"),
                    "quantization": quantization,
                    "active_constraint_gate": active_gate,
                    "actual_direct512_gate": direct_gate,
                    "fulltrain_behavior_gate": full_gate,
                    "dynamic_constraint_updates": additions,
                }
            )
            if active_gate["pass"] and direct_gate["pass"] and full_gate["pass"]:
                success = True
                status = "combined_train_endpoint_success"
                close_reason = "all_selected_fulltrain_behavior_and_direct_loss_gates_pass"
                break
        if not success and endpoints >= MAX_ENDPOINTS:
            close_reason = "maximum_endpoint_budget_reached"
        constraint_ledger = canonical_constraint_ledger(active)
        if success and candidate_consumer is not None:
            consumer_called = True
            actual_delta_by_name = {}
            offset = 0
            for name in VARIABLE_NAMES:
                parameter = parameters[name]
                count = int(parameter.numel())
                actual_delta_by_name[name] = torch.from_numpy(
                    current[offset : offset + count].astype(np.float32, copy=True)
                ).reshape(parameter.shape)
                offset += count
            if offset != COMBINED_ELEMENTS:
                raise RuntimeError("combined consumer delta split drift")
            candidate_consumer(
                {
                    "model": model,
                    "beta_checkpoint": beta_checkpoint,
                    "beta_model_state_sha256": beta_full_sha,
                    "actual_delta_float32_by_name_cpu": actual_delta_by_name,
                    "actual_combined_delta_float32_cpu": torch.from_numpy(
                        current.astype(np.float32, copy=True)
                    ),
                    "actual_combined_delta_float32_sha256": vector_sha256(
                        current, "<f4"
                    ),
                    "actual_combined_delta_l2": float(np.linalg.norm(current)),
                    "actual_actor6_delta_l2": float(
                        np.linalg.norm(current[:ACTOR_ELEMENTS])
                    ),
                    "actual_finalnorm_delta_l2": float(
                        np.linalg.norm(current[ACTOR_ELEMENTS:])
                    ),
                    "actor_pet_projection": endpoint_audit(current, q),
                    "identity_ledger": context_audit,
                    "active_constraint_ledger": constraint_ledger,
                    "active_constraint_gate": active_gate,
                    "actual_direct512_loss_gate": direct_gate,
                    "fulltrain_behavior_gate": full_gate,
                    "success_endpoint": endpoints,
                }
            )
    except ProbeClosed as error:
        close_reason = str(error)
    finally:
        set_combined_delta(
            parameters,
            beta_variables,
            np.zeros(COMBINED_ELEMENTS, dtype=np.float64),
        )
        restored_full = ppo.model_state_sha256(model)
        restored_other = PET.frozen.model_state_sha256(
            {name: model.state_dict()[name] for name in other_names}
        )
        restored_variable_exact = all(
            torch.equal(parameters[name].detach(), beta_variables[name])
            for name in VARIABLE_NAMES
        )
        final_integrity = {
            "finally_restore_executed": True,
            "beta100_full_sha_before": beta_full_sha,
            "beta100_full_sha_after": restored_full,
            "full_state_restored_exact": restored_full == beta_full_sha,
            "variable8_tensor_restored_exact": restored_variable_exact,
            "other72_sha_after": restored_other,
            "other72_restored_exact": restored_other == beta_other_sha,
        }
        if not all(
            (
                final_integrity["full_state_restored_exact"],
                final_integrity["variable8_tensor_restored_exact"],
                final_integrity["other72_restored_exact"],
            )
        ):
            raise RuntimeError("combined finally restoration failed")
    constraint_ledger = canonical_constraint_ledger(active)
    return {
        "schema_version": SCHEMA,
        "status": status,
        "scope": {
            "base": "beta100 complete state",
            "train_only": True,
            "validation_opened": False,
            "variable_names": list(VARIABLE_NAMES),
            "combined_elements": COMBINED_ELEMENTS,
            "other_state_tensors_exact": 72,
            "hard_combined_l2": HARD_COMBINED_L2,
            "max_endpoints": MAX_ENDPOINTS,
            "standalone_candidate_consumer_is_none": candidate_consumer is None,
            "checkpoint_or_result_writes": 0,
        },
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(SCRIPT.read_bytes()),
            },
            "frozen_actor6_v1": V1_EVIDENCE,
            "transition": transition_evidence,
        },
        "transition_context": context_audit,
        "combined_geometry": geometry["report"],
        "direct512": cache["report"]["direct512"],
        "baseline_direct512_losses": baseline_losses,
        "iterations": ledger,
        "decision": {
            "status": status,
            "close_reason": close_reason,
            "endpoints_evaluated": endpoints,
            "success": success,
            "active_constraint_count": len(constraint_ledger),
            "active_constraint_ledger_sha256": sha256_bytes(
                canonical_json_bytes(constraint_ledger)
            ),
            "candidate_consumer_called_before_finally_restore": consumer_called,
            "model_materialized": False,
        },
        "final_integrity": final_integrity,
    }


def cache_audit() -> dict[str, Any]:
    transition, transition_evidence = V1.load_transition_module()
    transition_cache = V1.cache_audit(transition)
    basis = PET.build_basis_geometry()
    geometry = build_combined_geometry(basis)
    cache = PET.build_cache_context(basis["payloads"]["R"])
    if transition_cache.get("status") != "cache_audit_passed_zero_write_train_only":
        raise RuntimeError("frozen transition cache audit failed")
    if cache["report"]["direct512"]["rows"] != 512:
        raise RuntimeError("Direct512 cache row count drift")
    return {
        "schema_version": SCHEMA,
        "status": "combined_cache_audit_passed_zero_write_train_only",
        "frozen_actor6_v1": V1_EVIDENCE,
        "transition": transition_evidence,
        "transition_cache": transition_cache,
        "combined_geometry": geometry["report"],
        "direct512_cache": cache["report"]["direct512"],
        "actual_executed": False,
        "validation_opened": False,
        "writes": 0,
    }


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
            "self": read_regular_bytes(SCRIPT, None, "combined self")[1],
            "frozen_actor6_v1": V1_EVIDENCE,
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
