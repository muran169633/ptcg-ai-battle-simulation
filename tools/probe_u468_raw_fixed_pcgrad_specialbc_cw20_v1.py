#!/usr/bin/env python3
"""One raw-U468 fixed-PCGrad special-BC endpoint, train-only and stdout-only."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import lzma
import math
import os
import random
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_raw_fixed_pcgrad_specialbc_cw20_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-raw-fixed-pcgrad-specialbc-cw20-v1"
SEED = 202608302

CW19 = TOOLS / "probe_u468_cw11_bootstrap_pcgrad_specialbc_cw19_v1.py"
CW19_SHA256 = "65f16009641481cd13538714520828640d00aed42aae920e24d9b70910423ae8"
CW19_MODE = 0o555

RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
ACTOR_DIMENSION = 65793
TASK_ORDER = (
    "union_mixed",
    "flg_hard",
    "pokemonfan_hard",
    "core5_hard",
)
RETENTION_OBJECTIVE = "union_retention"
PLANNED_STEP_L2 = 2.75e-5
ACTUAL_STEP_L2_MIN = 2.70e-5
ACTUAL_STEP_L2_MAX = 2.80e-5
UNION_LOSS_IMPROVEMENT_MIN = 1e-6
HARD_LOSS_IMPROVEMENT_MIN = 1e-8
RETENTION_LOSS_IMPROVEMENT_MIN = 1e-8
FIRST_ORDER_COSINE_EPSILON = 1e-12


class ProtocolError(RuntimeError):
    """Fail-closed CW20 error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(
    path: Path, expected_sha: str | None, expected_mode: int | None
) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
    ):
        raise ProtocolError(f"file identity/mode drift: {path}")
    digest = sha256_file(path)
    after = path.lstat()
    checks = {
        "regular": stat.S_ISREG(after.st_mode) and not stat.S_ISLNK(after.st_mode),
        "single_link": int(after.st_nlink) == 1,
        "identity_stable": (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "sha_exact": expected_sha is None or digest == expected_sha,
        "mode_exact": expected_mode is None
        or stat.S_IMODE(after.st_mode) == expected_mode,
    }
    if not all(checks.values()):
        raise ProtocolError(f"file changed: {path}: {checks}")
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


def import_cw19() -> tuple[ModuleType, dict[str, Any]]:
    evidence = regular_evidence(CW19, CW19_SHA256, CW19_MODE)
    spec = importlib.util.spec_from_file_location("cw20_frozen_cw19", CW19)
    if spec is None or spec.loader is None:
        raise ProtocolError("cannot import frozen CW19 helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, evidence


def runtime_audit(require_cuda: bool) -> dict[str, Any]:
    checks = {
        "cwd_exact": Path.cwd().resolve() == ROOT,
        "python_exact_my_project_env": Path(sys.executable).resolve()
        == EXPECTED_PYTHON.resolve(),
        "isolated": sys.flags.isolated == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "cublas_workspace_exact": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        == ":4096:8",
    }
    if not all(checks.values()):
        raise ProtocolError(f"runtime drift: {checks}")
    result: dict[str, Any] = {"checks": checks, "pass": True}
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise ProtocolError("CUDA unavailable")
        properties = torch.cuda.get_device_properties(0)
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "native_bf16": bool(torch.cuda.is_bf16_supported()),
            "device_count_exact_1": torch.cuda.device_count() == 1,
            "current_device_exact_0": torch.cuda.current_device() == 0,
            "device_name_exact": properties.name == "NVIDIA GeForce RTX 5090",
            "compute_capability_exact_12_0": (
                int(properties.major), int(properties.minor)
            )
            == (12, 0),
            "torch_version_exact": torch.__version__ == "2.8.0+cu128",
            "cuda_runtime_exact": torch.version.cuda == "12.8",
        }
        if not all(cuda.values()):
            raise ProtocolError(f"CUDA BF16 unavailable: {cuda}")
        result["cuda"] = cuda
    return result


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden = {
        "save",
        "backward",
        "step",
        "evaluate_candidate_once",
        "package_submission",
        "upload",
        "submit",
    }
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in forbidden:
            hits.append(f"{name}@{node.lineno}")
    checks = {
        "no_forbidden_training_or_external_calls": not hits,
        "stdout_print_present": any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}: {hits}")
    return {
        "checks": checks,
        "pass": True,
        "source_sha256": sha256_bytes(source),
        "source_bytes": len(source),
        "forbidden_hits": hits,
    }


def actor_bytes(parameters: Mapping[str, Any], names: Sequence[str], np: Any) -> bytes:
    pieces = []
    for name in names:
        parameter = parameters[name]
        if str(parameter.dtype) != "torch.float32":
            raise ProtocolError(f"actor tensor is not float32: {name}")
        value = parameter.detach().cpu().contiguous().numpy()
        pieces.append(np.ascontiguousarray(value.astype("<f4", copy=False)).tobytes())
    payload = b"".join(pieces)
    if len(payload) != ACTOR_DIMENSION * 4:
        raise ProtocolError("actor6 byte length drift")
    return payload


def actor_layout(parameters: Mapping[str, Any], names: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    offset = 0
    for name in names:
        parameter = parameters[name]
        count = int(parameter.numel())
        result.append(
            {
                "name": name,
                "dtype": str(parameter.dtype),
                "shape": [int(value) for value in parameter.shape],
                "numel": count,
                "start": offset,
                "stop": offset + count,
            }
        )
        offset += count
    if offset != ACTOR_DIMENSION:
        raise ProtocolError("actor layout dimension drift")
    return result


def flat_actor(parameters: Mapping[str, Any], names: Sequence[str], np: Any) -> Any:
    result = np.concatenate(
        [
            parameters[name].detach().cpu().to(dtype=__import__("torch").float64)
            .contiguous()
            .numpy()
            .reshape(-1)
            for name in names
        ]
    )
    if result.shape != (ACTOR_DIMENSION,) or not np.isfinite(result).all():
        raise ProtocolError("flat actor vector drift")
    return result


def apply_flat_actor(
    parameters: Mapping[str, Any], names: Sequence[str], flat: Any, torch: Any
) -> None:
    offset = 0
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            count = int(parameter.numel())
            value = torch.from_numpy(flat[offset : offset + count].reshape(parameter.shape))
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            offset += count
    if offset != ACTOR_DIMENSION:
        raise ProtocolError("flat actor application length drift")


def clone_actor(parameters: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    return {name: parameters[name].detach().clone() for name in names}


def restore_actor(
    parameters: Mapping[str, Any], names: Sequence[str], snapshot: Mapping[str, Any], torch: Any
) -> None:
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(snapshot[name])


def nonactor_sha(model: Any, names: Sequence[str], helper: Any) -> str:
    state = model.state_dict()
    actor = set(names)
    digest = hashlib.sha256()
    for name, tensor in sorted(
        (name, value) for name, value in state.items() if name not in actor
    ):
        value = tensor.detach().cpu().contiguous()
        if not bool(helper.torch.isfinite(value).all()):
            raise ProtocolError(f"nonfinite nonactor tensor: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(helper.torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def xz_payload(raw: bytes) -> dict[str, Any]:
    compressed = lzma.compress(
        raw,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )
    encoded = base64.standard_b64encode(compressed).decode("ascii")
    return {
        "dtype": "<f4",
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "compression": "XZ_preset9_extreme_CRC64",
        "compressed_bytes": len(compressed),
        "compressed_sha256": sha256_bytes(compressed),
        "base64_chunks_76": [
            encoded[index : index + 76] for index in range(0, len(encoded), 76)
        ],
    }


def decode_xz(value: Mapping[str, Any]) -> bytes:
    compressed = base64.b64decode(
        "".join(str(part) for part in value["base64_chunks_76"]), validate=True
    )
    if sha256_bytes(compressed) != value["compressed_sha256"]:
        raise ProtocolError("compressed actor payload SHA drift")
    raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if len(raw) != int(value["raw_bytes"]) or sha256_bytes(raw) != value["raw_sha256"]:
        raise ProtocolError("actor payload raw identity drift")
    return raw


def copy_actor_bytes(
    parameters: Mapping[str, Any], names: Sequence[str], layout: Sequence[Mapping[str, Any]], raw: bytes, np: Any, torch: Any
) -> None:
    vector = np.frombuffer(raw, dtype="<f4")
    if vector.shape != (ACTOR_DIMENSION,):
        raise ProtocolError("decoded actor payload shape drift")
    if [str(record.get("name")) for record in layout] != list(names):
        raise ProtocolError("payload actor order drift")
    expected_start = 0
    with torch.no_grad():
        for record in layout:
            name = str(record["name"])
            if name not in parameters or name not in names:
                raise ProtocolError("payload actor name drift")
            start, stop = int(record["start"]), int(record["stop"])
            parameter = parameters[name]
            if (
                str(record.get("dtype")) != "torch.float32"
                or str(parameter.dtype) != "torch.float32"
                or start != expected_start
                or stop - start != int(parameter.numel())
                or int(record["numel"]) != int(parameter.numel())
                or tuple(int(value) for value in record["shape"])
                != tuple(int(value) for value in parameter.shape)
            ):
                raise ProtocolError("payload actor layout drift")
            value = torch.from_numpy(
                vector[start:stop].copy().reshape(tuple(record["shape"]))
            )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            expected_start = stop
    if expected_start != ACTOR_DIMENSION:
        raise ProtocolError("payload actor layout coverage drift")


def retention_rows(selections: Sequence[Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    return [
        row
        for batch in selections
        for row in batch
        if str(row["category"]) in {"fragile", "c34"}
    ]


def endpoint_gate(
    cw19: ModuleType,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_correct: Mapping[str, bool],
    candidate_correct: Mapping[str, bool],
    selections: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    improvements = {
        "union_mixed": float(baseline["mixed_ordered_loss"])
        - float(candidate["mixed_ordered_loss"]),
        **{
            f"{source}_hard": float(
                baseline["by_source_and_bucket"][source]["hard"]["ordered_loss"]
            )
            - float(candidate["by_source_and_bucket"][source]["hard"]["ordered_loss"])
            for source in ("flg", "pokemonfan", "core5")
        },
    }
    rows = retention_rows(selections)
    flips = [
        str(row["line_sha256"])
        for row in rows
        if baseline_correct[str(row["line_sha256"])]
        and not candidate_correct[str(row["line_sha256"])]
    ]
    hard_correct = {}
    for source in ("flg", "pokemonfan", "core5"):
        before = int(baseline["by_source_and_bucket"][source]["hard"]["ordered_correct"])
        after = int(candidate["by_source_and_bucket"][source]["hard"]["ordered_correct"])
        hard_correct[source] = {
            "baseline": before,
            "candidate": after,
            "net": after - before,
            "nondegrade": after >= before,
        }
    core5_hard_rows = [
        row
        for batch in selections
        for row in batch
        if str(row["source"]) == "core5" and str(row["category"]) == "hard"
    ]
    core5_hard_teams = sorted({str(row["team_name"]) for row in core5_hard_rows})
    core5_hard_team_correct = {}
    for team in core5_hard_teams:
        team_rows = [row for row in core5_hard_rows if str(row["team_name"]) == team]
        before = sum(
            baseline_correct[str(row["line_sha256"])] for row in team_rows
        )
        after = sum(
            candidate_correct[str(row["line_sha256"])] for row in team_rows
        )
        core5_hard_team_correct[team] = {
            "rows": len(team_rows),
            "baseline": before,
            "candidate": after,
            "net": after - before,
            "nondegrade": after >= before,
        }
    base_retention_loss = cw19.retention_loss(baseline)
    candidate_retention_loss = cw19.retention_loss(candidate)
    retention_improvement = base_retention_loss - candidate_retention_loss
    baseline_retention_contract = baseline["retention"] == {
        "rows": 288,
        "ordered_correct": 288,
        "fragile_rows": 282,
        "fragile_ordered_correct": 282,
        "context34_rows": 6,
        "context34_ordered_correct": 6,
    }
    checks = {
        "raw_retention_contract_exact": baseline_retention_contract,
        "union_loss_improvement": improvements["union_mixed"]
        >= UNION_LOSS_IMPROVEMENT_MIN,
        "three_hard_loss_improvements": all(
            improvements[f"{source}_hard"] >= HARD_LOSS_IMPROVEMENT_MIN
            for source in ("flg", "pokemonfan", "core5")
        ),
        "retention_weighted_loss_improvement": retention_improvement
        >= RETENTION_LOSS_IMPROVEMENT_MIN,
        "retention_correct_to_wrong_zero": len(flips) == 0,
        "each_hard_correct_nondegrade": all(
            record["nondegrade"] for record in hard_correct.values()
        ),
        "core5_five_hard_teams_each_nondegrade": len(core5_hard_team_correct) == 5
        and all(record["nondegrade"] for record in core5_hard_team_correct.values()),
        "at_least_one_hard_discrete_repair": sum(
            record["net"] for record in hard_correct.values()
        )
        >= 1,
        "pokemonfan_hard_discrete_repair": hard_correct["pokemonfan"]["net"]
        >= 1,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "loss_improvements": improvements,
        "union_minimum": UNION_LOSS_IMPROVEMENT_MIN,
        "hard_minimum": HARD_LOSS_IMPROVEMENT_MIN,
        "baseline_retention_weighted_loss": base_retention_loss,
        "candidate_retention_weighted_loss": candidate_retention_loss,
        "retention_weighted_loss_improvement": retention_improvement,
        "retention_minimum": RETENTION_LOSS_IMPROVEMENT_MIN,
        "retention_rows": len(rows),
        "baseline_correct_retention_rows": sum(
            baseline_correct[str(row["line_sha256"])] for row in rows
        ),
        "retention_correct_to_wrong_count": len(flips),
        "retention_correct_to_wrong_line_sha256": flips,
        "hard_correct": hard_correct,
        "core5_hard_team_correct": core5_hard_team_correct,
    }


def production_run() -> dict[str, Any]:
    import numpy as np
    import torch

    production_runtime = runtime_audit(require_cuda=True)
    production_source = source_audit()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    precision_before = torch.get_float32_matmul_precision()
    cw19, cw19_evidence = import_cw19()
    runner, runner_evidence = cw19.import_locked(
        cw19.RUNNER, "run_u468_raw_trainhard_actor6_balanced_mix_sweep"
    )
    grad, grad_evidence = cw19.import_locked(
        cw19.GRADIENT, "cw20_frozen_gradient_probe"
    )
    if grad.sweep is not runner:
        raise ProtocolError("gradient module did not bind locked runner")
    selections, selection_summary, selection_evidence = grad.load_frozen_selection()
    cache, cache_audit, cache_inputs = grad.load_frozen_cache(selections)
    inventory = grad.objective_inventory(cache, selections)
    device = torch.device("cuda")
    model, _ = grad.sweep.load_raw_u468(device)
    model.eval()
    names = tuple(grad.sweep.ACTOR_NAMES)
    parameters = grad.sweep.configure_actor6(model)
    raw_actor = clone_actor(parameters, names)
    raw_flat = flat_actor(parameters, names, np)
    raw_actor_bytes = actor_bytes(parameters, names, np)
    raw_model_hash = grad.sweep.ppo.model_state_sha256(model)
    raw_nonactor_hash = nonactor_sha(model, names, grad.sweep.ppo)
    raw_checks = {
        "raw_model_hash_exact": raw_model_hash == RAW_MODEL_SHA256,
        "raw_nonactor_hash_exact": raw_nonactor_hash == RAW_NONACTOR_SHA256,
        "actor_dimension": raw_flat.shape == (ACTOR_DIMENSION,),
        "actor6_all_float32": all(
            parameter.dtype == torch.float32 for parameter in parameters.values()
        ),
        "actor_names_exact": names
        == (
            "actor_query.weight",
            "actor_key.weight",
            "actor_residual.0.weight",
            "actor_residual.0.bias",
            "actor_residual.2.weight",
            "actor_residual.2.bias",
        ),
    }
    if not all(raw_checks.values()):
        raise ProtocolError(f"raw U468 identity drift: {raw_checks}")

    result: dict[str, Any]
    try:
        torch.set_float32_matmul_precision("high")
        if torch.get_float32_matmul_precision() != "high":
            raise ProtocolError("failed to enter high matmul precision scope")
        baseline_report, baseline_correct = grad.sweep.evaluate_selected_union(
            model, cache, selections, device
        )
        objective_gradients, baseline_losses, autograd_audit = (
            grad.compute_objective_gradients(
                model, parameters, cache, selections, inventory, device
            )
        )
        if grad.sweep.ppo.model_state_sha256(model) != raw_model_hash:
            raise ProtocolError("gradient computation changed raw model")
        retention_gradient, retention_loss_value, retention_weight = (
            grad.derive_union_retention(objective_gradients, baseline_losses, inventory)
        )
        effect_gradients = {
            **objective_gradients,
            RETENTION_OBJECTIVE: retention_gradient,
        }
        pcgrad, pcgrad_audit = grad.fixed_order_pcgrad(objective_gradients)
        effects = grad.candidate_effects(pcgrad, effect_gradients)
        direction = cw19.flatten_gradient(pcgrad, names, np)
        direction_l2 = float(np.linalg.norm(direction))
        direction_gate = {
            "task_order_exact": tuple(pcgrad_audit["task_order"]) == TASK_ORDER,
            "original_unprojected_references": pcgrad_audit["reference_gradient_kind"]
            == "original_unprojected",
            "arithmetic_mean": pcgrad_audit["aggregation"]
            == "arithmetic_mean_of_projected_task_gradients",
            "four_tasks_plus_retention_robust_descent": effects[
                "all_directional_gates_numerically_robust_first_order_descent"
            ]
            is True,
            "minimum_directional_gate_cosine": float(
                effects["minimum_directional_gate_cosine"]
            )
            > FIRST_ORDER_COSINE_EPSILON,
            "direction_finite_nonzero": math.isfinite(direction_l2)
            and direction_l2 > 0.0,
        }
        common = {
            "raw_checks": raw_checks,
            "baseline_train512": baseline_report,
            "baseline_losses": {
                **{name: float(value) for name, value in baseline_losses.items()},
                RETENTION_OBJECTIVE: float(retention_loss_value),
            },
            "pcgrad": {
                "direction_gate": direction_gate,
                "direction_l2": direction_l2,
                "direction_float64_le_sha256": cw19.float64_sha(direction, np),
                "audit": pcgrad_audit,
                "effects": effects,
                "retention_effective_weight": float(retention_weight),
            },
            "autograd_audit": autograd_audit,
            "gradient_definition": {
                "native_bf16_task_gradients_independent": True,
                "same_two_direct_256_row_forward_graphs": True,
                "union_from_source_gradient_additivity_required": False,
                "union_from_source_reconstruction_gate_used": False,
            },
        }
        if not all(direction_gate.values()):
            result = {
                **common,
                "decision": "NO_GO_CW20_CACHE512_PREFLIGHT",
                "reason": "PCGRAD_DIRECTION_GATE_FAILED",
                "candidate_payload": None,
                "changed_candidate_train_endpoint_count": 0,
            }
        else:
            planned_delta = -PLANNED_STEP_L2 * direction / direction_l2
            planned_target = raw_flat + planned_delta
            apply_flat_actor(parameters, names, planned_target, torch)
            candidate_hash = grad.sweep.ppo.model_state_sha256(model)
            actual_flat = flat_actor(parameters, names, np)
            actual_delta = actual_flat - raw_flat
            actual_l2 = float(np.linalg.norm(actual_delta))
            changed_names = [
                name
                for name in names
                if not torch.equal(parameters[name].detach(), raw_actor[name])
            ]
            integrity_checks = {
                "planned_step_l2_exact": math.isclose(
                    float(np.linalg.norm(planned_delta)),
                    PLANNED_STEP_L2,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "actual_step_l2_in_frozen_interval": ACTUAL_STEP_L2_MIN
                <= actual_l2
                <= ACTUAL_STEP_L2_MAX,
                "changed_scope_exact_actor6": tuple(changed_names) == names,
                "nonactor_exact_raw": nonactor_sha(model, names, grad.sweep.ppo)
                == RAW_NONACTOR_SHA256,
                "candidate_model_changed": candidate_hash != raw_model_hash,
                "all_finite": bool(np.isfinite(actual_flat).all()),
                "no_clip_or_parameter_space_projection": True,
            }
            actual_predicted = {
                name: -float(
                    cw19.flatten_gradient(effect_gradients[name], names, np)
                    @ actual_delta
                )
                for name in (*TASK_ORDER, RETENTION_OBJECTIVE)
            }
            integrity_checks[
                "actual_four_tasks_plus_retention_first_order_improvement"
            ] = all(
                actual_predicted[name] > 0.0
                for name in (*TASK_ORDER, RETENTION_OBJECTIVE)
            )
            candidate_report, candidate_correct = grad.sweep.evaluate_selected_union(
                model, cache, selections, device
            )
            train_gate = endpoint_gate(
                cw19,
                baseline_report,
                candidate_report,
                baseline_correct,
                candidate_correct,
                selections,
            )
            pre_payload_checks = {
                "integrity": all(integrity_checks.values()),
                "train_gate": train_gate["pass"] is True,
            }
            payload = None
            reconstruction = None
            final_checks = dict(pre_payload_checks)
            if all(pre_payload_checks.values()):
                layout = actor_layout(parameters, names)
                candidate_actor_bytes = actor_bytes(parameters, names, np)
                payload = {
                    "anchor": {
                        "raw_checkpoint": str(grad.sweep.U468.relative_to(ROOT)),
                        "raw_checkpoint_sha256": grad.sweep.U468_SHA256,
                        "raw_model_state_sha256": raw_model_hash,
                        "raw_nonactor_sha256": raw_nonactor_hash,
                        "terminal_model_state_sha256": candidate_hash,
                    },
                    "formula": "replace_only_six_actor_float32_tensors_from_frozen_payload",
                    "actor_names": list(names),
                    "actor_layout": layout,
                    "actor_layout_sha256": hashlib.sha256(
                        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "raw_actor_float32_le_sha256": sha256_bytes(raw_actor_bytes),
                    "candidate_actor_float32_le": xz_payload(candidate_actor_bytes),
                    "planned_step_l2": PLANNED_STEP_L2,
                    "actual_step_l2": actual_l2,
                    "actual_delta_float64_le_sha256": cw19.float64_sha(actual_delta, np),
                    "pcgrad_direction_float64_le_sha256": cw19.float64_sha(direction, np),
                }
                restore_actor(parameters, names, raw_actor, torch)
                restored_raw_hash = grad.sweep.ppo.model_state_sha256(model)
                decoded = decode_xz(payload["candidate_actor_float32_le"])
                copy_actor_bytes(parameters, names, layout, decoded, np, torch)
                reconstructed_hash = grad.sweep.ppo.model_state_sha256(model)
                reconstructed_bytes = actor_bytes(parameters, names, np)
                reconstruction_checks = {
                    "raw_restore_exact": restored_raw_hash == raw_model_hash,
                    "decoded_actor_bytes_exact": reconstructed_bytes
                    == candidate_actor_bytes,
                    "terminal_model_hash_exact": reconstructed_hash == candidate_hash,
                    "nonactor_still_exact": nonactor_sha(model, names, grad.sweep.ppo)
                    == RAW_NONACTOR_SHA256,
                }
                reconstruction = {
                    "checks": reconstruction_checks,
                    "pass": all(reconstruction_checks.values()),
                    "model_state_sha256": reconstructed_hash,
                }
                final_checks["pure_payload_reconstruction"] = reconstruction["pass"]
            passed = bool(final_checks) and all(final_checks.values())
            failure_reason = (
                "all_cache512_train_only_gates_passed"
                if passed
                else "integrity_gate_failed"
                if not pre_payload_checks["integrity"]
                else "cache512_train_gate_failed"
                if not pre_payload_checks["train_gate"]
                else "payload_reconstruction_gate_failed"
            )
            result = {
                **common,
                "decision": (
                    "GO_CW20_CACHE512_PREFLIGHT"
                    if passed
                    else "NO_GO_CW20_CACHE512_PREFLIGHT"
                ),
                "reason": failure_reason,
                "candidate_model_state_sha256": candidate_hash,
                "candidate_train512": candidate_report,
                "planned_step": {
                    "l2": float(np.linalg.norm(planned_delta)),
                    "float64_le_sha256": cw19.float64_sha(planned_delta, np),
                },
                "actual_step": {
                    "l2": actual_l2,
                    "float64_le_sha256": cw19.float64_sha(actual_delta, np),
                    "changed_parameter_names": changed_names,
                    "predicted_loss_improvements": actual_predicted,
                },
                "integrity_checks": integrity_checks,
                "train_gate": train_gate,
                "pre_payload_checks": pre_payload_checks,
                "final_checks": final_checks,
                "pure_payload_reconstruction": reconstruction,
                "candidate_payload": payload if passed else None,
                "changed_candidate_train_endpoint_count": 1,
            }
    finally:
        restore_actor(parameters, names, raw_actor, torch)
        torch.set_float32_matmul_precision(precision_before)
        if grad.sweep.ppo.model_state_sha256(model) != raw_model_hash:
            raise ProtocolError("failed final RAM restore to raw U468")
        if torch.get_float32_matmul_precision() != precision_before:
            raise ProtocolError("failed to restore incoming matmul precision")

    decision = result["decision"]
    output = {
        "schema_version": SCHEMA,
        "status": decision,
        "decision": decision,
        "seed": SEED,
        "endpoint": result,
        "train_cache": {
            "selection": selection_summary,
            "cache": cache_audit,
            "inventory": inventory,
            "non_train_members_opened": False,
        },
        "contract": {
            "base": "raw_U468_after_generalBC_plus_PPO",
            "special_BC": "fixed_order_PCGrad_direct_normalized_actor6_step",
            "gradient_definition": (
                "independent_native_BF16_objectives_on_two_shared_B256_graphs; "
                "no_union_from_source_additivity_assumption"
            ),
            "task_order": list(TASK_ORDER),
            "planned_step_l2": PLANNED_STEP_L2,
            "actual_step_l2_interval": [ACTUAL_STEP_L2_MIN, ACTUAL_STEP_L2_MAX],
            "candidate_count": 1,
            "promotion_scope": "cache512_train_only_preflight_not_fulltrain_or_specialist",
            "cw11_no_harm_proven": False,
            "next_gate_if_GO": "frozen_fulltrain_before_official_specialist",
            "sweeps": 0,
            "line_searches": 0,
            "parameter_space_projections": 0,
            "pcgrad_gradient_conflict_projection": True,
            "clips": 0,
            "optimizer_instances": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "changed_candidate_official_or_validation_evaluation_count": 0,
            "official_candidate_budget_consumed": 0,
            "checkpoint_writes": 0,
            "model_writes": 0,
            "result_artifact_writes": 0,
            "stdout_only": True,
        },
        "integrity": {
            "runtime": production_runtime,
            "source": production_source,
            "frozen_inputs": {
                "cw19_helper": cw19_evidence,
                "runner": runner_evidence,
                "gradient": grad_evidence,
                "selection": selection_evidence,
                "cache_inputs": cache_inputs,
            },
        },
        "writes_performed": 0,
        "submission_performed": False,
    }
    if not grad.sweep.repair.finite_nested(output):
        raise ProtocolError("CW20 output contains nonfinite values")
    return output


def selftest_result() -> dict[str, Any]:
    raw = bytes((index * 17) % 256 for index in range(4096))
    encoded = xz_payload(raw)
    decoded = decode_xz(encoded)
    checks = {
        "payload_roundtrip": decoded == raw,
        "planned_step_literal": PLANNED_STEP_L2 == 2.75e-5,
        "actual_interval_contains_planned": ACTUAL_STEP_L2_MIN
        <= PLANNED_STEP_L2
        <= ACTUAL_STEP_L2_MAX,
        "retention_minimum_literal": RETENTION_LOSS_IMPROVEMENT_MIN == 1e-8,
        "task_order_exact": TASK_ORDER
        == ("union_mixed", "flg_hard", "pokemonfan_hard", "core5_hard"),
    }
    if not all(checks.values()):
        raise ProtocolError(f"selftest failed: {checks}")
    return {
        "schema_version": SCHEMA,
        "status": "selftest_passed",
        "pass": True,
        "checks": checks,
        "source": source_audit(),
        "runtime": runtime_audit(require_cuda=False),
        "writes_performed": 0,
        "submission_performed": False,
    }


def static_result() -> dict[str, Any]:
    cw19_evidence = regular_evidence(CW19, CW19_SHA256, CW19_MODE)
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "pass": True,
        "source": source_audit(),
        "runtime": runtime_audit(require_cuda=False),
        "frozen_CW19_helper": cw19_evidence,
        "protocol": {
            "base": "raw_U468",
            "task_order": list(TASK_ORDER),
            "planned_step_l2": PLANNED_STEP_L2,
            "single_endpoint": True,
            "train_only": True,
            "no_parameter_space_projection": True,
            "pcgrad_gradient_conflict_projection": True,
            "no_sweep": True,
        },
        "writes_performed": 0,
        "submission_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "selftest", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ProtocolError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise ProtocolError("requires Python -I -B")
    if args.mode == "run" and args.device != "cuda":
        raise ProtocolError("production run is CUDA-only")
    if args.mode != "run" and args.device != "cpu":
        raise ProtocolError("static/selftest are CPU-only")
    result = (
        static_result()
        if args.mode == "static"
        else selftest_result()
        if args.mode == "selftest"
        else production_run()
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
