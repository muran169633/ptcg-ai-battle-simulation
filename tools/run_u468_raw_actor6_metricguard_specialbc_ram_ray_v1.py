#!/usr/bin/env python3
"""Zero-write selected-row RAM ray for the frozen raw-U468 metric guard.

The hash-bound geometry module selects the direction before any endpoint is
created.  This runner then evaluates all three preregistered radii in RAM,
restoring raw actor6 tensors before and after every point.  Only the selected
19 train rows are opened.  Results go to stdout; no model/result file is ever
created and there is no optimizer, backward call, parameter step, network
action, validation access, holdout, or full-panel evaluation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
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
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_ram_ray_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-specialbc-ram-ray-v1"
SEED = 202608203

GEOMETRY = TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py"
GEOMETRY_SHA256 = (
    "ddecd3a85bc2b43c28854afc56678c21943198ac3fde3613765d6a49e440eedb"
)
GEOMETRY_MODE = 0o555
GEOMETRY_SCHEMA = "ptcg-u468-raw-actor6-metricguard-specialbc-geometry-v2"
EXPECTED_DIRECTION_SHA256 = (
    "e4714580607e4c9543c026fc4851326526b33612ca9ed39818122e3ec3cffd84"
)
RADII = (
    0.0028737480729610368,
    0.0038316640972813824,
    0.004789580121601728,
)
DISPLACEMENT_L2_ATOL = 5e-6
DISPLACEMENT_COSINE_MIN = 0.9999
MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
NATIVE_OUTPUT_KEYS = ("policy_logits", "count_logits", "value_logits")


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
    expected_sha256: str,
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
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(mode),
        "single_link_regular_held_fd_identity_exact": True,
    }


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_attributes = {
        "backward",
        "step",
        "save",
        "savez",
        "write",
        "write_bytes",
        "write_text",
        "touch",
        "mkdir",
        "unlink",
        "rename",
        "replace",
    }
    forbidden_names = {"open", "exec", "eval", "compile", "DataLoader"}
    forbidden_os_flags = {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
    forbidden_import_roots = {"requests", "urllib", "subprocess"}
    attribute_hits: list[tuple[int, str]] = []
    name_hits: list[tuple[int, str]] = []
    os_flag_hits: list[tuple[int, str]] = []
    import_hits: list[tuple[int, str]] = []
    print_sites: list[int] = []
    os_open_sites: list[int] = []
    copy_sites: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append((node.lineno, name))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in forbidden_os_flags
        ):
            os_flag_hits.append((node.lineno, node.attr))
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in forbidden_attributes:
                attribute_hits.append((node.lineno, function.attr))
            if function.attr == "copy_":
                copy_sites.append(node.lineno)
            if (
                isinstance(function.value, ast.Name)
                and function.value.id == "os"
                and function.attr == "open"
            ):
                os_open_sites.append(node.lineno)
        elif isinstance(function, ast.Name):
            if function.id in forbidden_names:
                name_hits.append((node.lineno, function.id))
            if function.id == "print":
                print_sites.append(node.lineno)
    if attribute_hits or name_hits or os_flag_hits or import_hits:
        raise RuntimeError(
            "zero-write static audit failed: "
            f"{attribute_hits=} {name_hits=} {os_flag_hits=} {import_hits=}"
        )
    if len(print_sites) != 1 or len(os_open_sites) != 1 or len(copy_sites) != 1:
        raise RuntimeError("static stdout/read/overlay call-site cardinality drift")
    return {
        "ast_parse": True,
        "forbidden_attribute_call_sites": attribute_hits,
        "forbidden_name_call_sites": name_hits,
        "forbidden_os_write_flags": os_flag_hits,
        "forbidden_network_imports": import_hits,
        "stdout_print_call_sites": print_sites,
        "os_open_read_only_call_sites": os_open_sites,
        "ram_actor_copy_call_sites": copy_sites,
        "direct_autograd_grad_call_sites": 0,
        "hash_bound_geometry_owns_autograd_and_solver": True,
        "no_optimizer_backward_step_save_or_write": True,
        "no_dataloader_holdout_or_full_panel": True,
    }


def load_geometry() -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        GEOMETRY,
        GEOMETRY_SHA256,
        "frozen metric-guard geometry",
        expected_mode=GEOMETRY_MODE,
    )
    spec = importlib.util.spec_from_file_location("u468_metricguard_geometry_v2", GEOMETRY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct frozen geometry import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.SCHEMA != GEOMETRY_SCHEMA
        or module.EXPECTED_DIRECTION_SHA256 != EXPECTED_DIRECTION_SHA256
        or tuple(module.TARGET_METRICS) != MAIN_METRICS
        or module.EXPECTED_ROW_COUNT != 19
        or module.EXPECTED_TARGET_COUNT != 5
        or module.EXPECTED_GUARD_COUNT != 14
    ):
        raise RuntimeError("frozen geometry exported contract drift")
    return module, evidence


def tensor_sha256(tensor: Any, torch: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256_bytes(payload)


def official_row_record(
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    actions: Sequence[Sequence[int]],
    row_index: int,
) -> dict[str, Any]:
    targets = cpu_batch["targets"][row_index].bool()
    option_mask = cpu_batch["option_mask"][row_index].bool()
    prediction = targets.new_zeros(targets.shape)
    order = [int(value) for value in actions[row_index]]
    if order:
        prediction[order] = True
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert_order = cpu_batch["expert_ordered_actions"][
        row_index, :expert_count
    ].tolist()
    context = int(cpu_batch["contexts"][row_index])
    hybrid_order = order if context == 34 else sorted(order)
    top1_index = int(outputs_cpu["policy_logits"][row_index].argmax())
    value_sign = bool(outputs_cpu["value_logits"][row_index] >= 0)
    value_target = bool(cpu_batch["win_targets"][row_index])
    flags: dict[str, bool | None] = {
        "set_exact": bool(((prediction == targets) | ~option_mask).all()),
        "hybrid_order_exact": hybrid_order == expert_order,
        "ordered_exact": order == expert_order,
        "top1_correct": bool(targets[top1_index]),
        "context34_hybrid_order_exact": (
            hybrid_order == expert_order if context == 34 else None
        ),
        "context34_ordered_exact": (
            order == expert_order if context == 34 else None
        ),
        "count_correct": len(order) == int(cpu_batch["action_counts"][row_index]),
        "value_correct": value_sign == value_target,
        "nonempty": int(cpu_batch["action_counts"][row_index]) > 0,
    }
    return {
        "flags": flags,
        "predicted_order": order,
        "expert_order": [int(value) for value in expert_order],
        "hybrid_order": [int(value) for value in hybrid_order],
        "predicted_count": len(order),
        "expert_count": int(cpu_batch["action_counts"][row_index]),
        "top1_index_full_policy_vector": top1_index,
        "value_sign": value_sign,
        "value_target": value_target,
        "context": context,
    }


def snapshot_forward(
    helper: ModuleType,
    model: Any,
    batch: Mapping[str, Any],
    cpu_batch: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    torch = helper.torch
    with torch.no_grad():
        outputs = helper.ppo.model_forward(model, dict(batch), device)
        if set(outputs) != set(NATIVE_OUTPUT_KEYS):
            raise RuntimeError("selected-row forward output key drift")
        if any(value.dtype != torch.bfloat16 for value in outputs.values()):
            raise RuntimeError("selected-row forward is not native CUDA BF16")
        actions, _, _, _ = helper.ppo.sample_ordered_actions(
            outputs,
            dict(batch),
            deterministic=True,
            canonicalize_order=False,
        )
        outputs_cpu = {
            key: value.detach().cpu().contiguous() for key, value in outputs.items()
        }
    return {
        "actions": [[int(value) for value in row] for row in actions],
        "outputs_cpu": outputs_cpu,
        "official_rows": [
            official_row_record(cpu_batch, outputs_cpu, actions, index)
            for index in range(len(actions))
        ],
        "fingerprints": {
            key: tensor_sha256(value, torch) for key, value in outputs_cpu.items()
        },
    }


def set_actor_overlay(
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    direction_tensors: Sequence[Any],
    radius: float,
    torch: Any,
) -> None:
    with torch.no_grad():
        for parameter, raw_value, direction in zip(
            parameters, raw_actor, direction_tensors
        ):
            value = raw_value if radius == 0.0 else raw_value + float(radius) * direction
            parameter.copy_(value)


def actor_displacement(
    parameters: Sequence[Any],
    raw_actor: Sequence[Any],
    direction: Any,
    radius: float,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    flat_parts = []
    maximum = 0.0
    for parameter, raw_value in zip(parameters, raw_actor):
        delta = parameter.detach().float().cpu() - raw_value.detach().float().cpu()
        flat_parts.append(delta.reshape(-1).double())
        maximum = max(maximum, float(delta.abs().max()))
    flat = torch.cat(flat_parts).numpy()
    l2 = float(np.linalg.norm(flat))
    cosine = float((flat @ direction) / (l2 * np.linalg.norm(direction)))
    return {
        "actor6_l2": l2,
        "actor6_max_abs": maximum,
        "cosine_to_frozen_direction": cosine,
        "l2_matches_radius": abs(l2 - radius) <= DISPLACEMENT_L2_ATOL,
        "cosine_at_least_threshold": cosine >= DISPLACEMENT_COSINE_MIN,
    }


def evaluate_endpoint(
    radius: float,
    descriptors: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cpu_batch: Mapping[str, Any],
    torch: Any,
    nonactor_exact: bool,
    displacement: Mapping[str, Any],
) -> dict[str, Any]:
    raw_outputs = raw_snapshot["outputs_cpu"]
    candidate_outputs = candidate["outputs_cpu"]
    count_exact_rows = (
        (candidate_outputs["count_logits"] == raw_outputs["count_logits"])
        .reshape(len(descriptors), -1)
        .all(dim=1)
        .tolist()
    )
    value_exact_rows = (
        (candidate_outputs["value_logits"] == raw_outputs["value_logits"])
        .reshape(len(descriptors), -1)
        .all(dim=1)
        .tolist()
    )
    target_records = []
    guard_records = []
    target_gate = True
    guard_gate = True
    target_metric_obligations: list[str] = []
    for index, descriptor in enumerate(descriptors):
        raw_official = raw_snapshot["official_rows"][index]
        candidate_official = candidate["official_rows"][index]
        if descriptor["role"] == "target":
            main_flags = {
                metric: bool(candidate_official["flags"][metric])
                for metric in MAIN_METRICS
            }
            target_metric_obligations.extend(
                f"{descriptor['line_sha256']}:{metric}" for metric in MAIN_METRICS
            )
            positive = int(descriptor["positive_option"])
            negative = int(descriptor["negative_option"])
            raw_margin = float(
                raw_outputs["policy_logits"][index, positive].float()
                - raw_outputs["policy_logits"][index, negative].float()
            )
            candidate_margin = float(
                candidate_outputs["policy_logits"][index, positive].float()
                - candidate_outputs["policy_logits"][index, negative].float()
            )
            required_positive = float(descriptor["expected_positive_bf16_margin"])
            pair_margin_gate = candidate_margin >= required_positive
            row_pass = all(main_flags.values()) and pair_margin_gate
            target_gate = target_gate and row_pass
            target_records.append(
                {
                    "identity": {
                        "panel": descriptor["panel"],
                        "member": descriptor["member"],
                        "line_index_zero_based": descriptor[
                            "line_index_zero_based"
                        ],
                        "line_sha256": descriptor["line_sha256"],
                    },
                    "raw_official": raw_official,
                    "candidate_official": candidate_official,
                    "observed_pair": [positive, negative],
                    "raw_pair_margin": raw_margin,
                    "candidate_pair_margin": candidate_margin,
                    "required_positive_bf16_margin": required_positive,
                    "candidate_pair_margin_at_least_positive_bf16": pair_margin_gate,
                    "all_four_main_metrics_correct": row_pass,
                }
            )
            continue
        metrics = [str(value) for value in descriptor["metrics_union"]]
        raw_metric_flags = {
            metric: raw_official["flags"][metric] for metric in metrics
        }
        candidate_metric_flags = {
            metric: candidate_official["flags"][metric] for metric in metrics
        }
        if not all(value is True for value in raw_metric_flags.values()):
            raise RuntimeError("guard metrics_union contains a raw-false metric")
        metric_local_zero_cw = all(
            value is True for value in candidate_metric_flags.values()
        )
        positive = int(descriptor["positive_option"])
        negative = int(descriptor["negative_option"])
        raw_margin = float(
            raw_outputs["policy_logits"][index, positive].float()
            - raw_outputs["policy_logits"][index, negative].float()
        )
        candidate_margin = float(
            candidate_outputs["policy_logits"][index, positive].float()
            - candidate_outputs["policy_logits"][index, negative].float()
        )
        margin_non_decrease = candidate_margin >= raw_margin
        row_pass = metric_local_zero_cw and margin_non_decrease
        guard_gate = guard_gate and row_pass
        guard_records.append(
            {
                "identity": {
                    "panel": descriptor["panel"],
                    "member": descriptor["member"],
                    "line_index_zero_based": descriptor["line_index_zero_based"],
                    "line_sha256": descriptor["line_sha256"],
                },
                "metrics_union_only": metrics,
                "raw_metric_flags": raw_metric_flags,
                "candidate_metric_flags": candidate_metric_flags,
                "metric_local_zero_cw": metric_local_zero_cw,
                "observed_pair": [positive, negative],
                "raw_pair_margin": raw_margin,
                "candidate_pair_margin": candidate_margin,
                "observed_pair_margin_at_least_raw": margin_non_decrease,
            }
        )
    count_native_exact = bool(all(count_exact_rows)) and torch.equal(
        candidate_outputs["count_logits"], raw_outputs["count_logits"]
    )
    value_native_exact = bool(all(value_exact_rows)) and torch.equal(
        candidate_outputs["value_logits"], raw_outputs["value_logits"]
    )
    immutable_gate = count_native_exact and value_native_exact and nonactor_exact
    policy_native_differs_raw = not torch.equal(
        candidate_outputs["policy_logits"], raw_outputs["policy_logits"]
    )
    obligation_count = len(target_metric_obligations)
    obligation_unique_count = len(set(target_metric_obligations))
    if obligation_count != 20 or obligation_unique_count != 20:
        raise RuntimeError("five-target by four-metric obligation ledger drift")
    eligible = target_gate and guard_gate and immutable_gate and policy_native_differs_raw
    return {
        "radius": radius,
        "actor6_displacement": dict(displacement),
        "targets": target_records,
        "guards": guard_records,
        "immutables": {
            "count_logits_native_tensor_exact_raw": count_native_exact,
            "value_logits_native_tensor_exact_raw": value_native_exact,
            "count_logits_exact_rows": [bool(value) for value in count_exact_rows],
            "value_logits_exact_rows": [bool(value) for value in value_exact_rows],
            "nonactor_state_bit_exact_raw": nonactor_exact,
            "policy_logits_native_tensor_differs_raw": policy_native_differs_raw,
            "raw_fingerprints": {
                "count_logits": raw_snapshot["fingerprints"]["count_logits"],
                "value_logits": raw_snapshot["fingerprints"]["value_logits"],
            },
            "candidate_fingerprints": {
                "count_logits": candidate["fingerprints"]["count_logits"],
                "value_logits": candidate["fingerprints"]["value_logits"],
            },
        },
        "gates": {
            "five_targets_all_four_main_metrics_and_positive_pair_margin": target_gate,
            "target_metric_obligation_count": obligation_count,
            "target_metric_obligation_unique_count": obligation_unique_count,
            "target_metric_obligation_20_of_20_exact": True,
            "fourteen_guards_metric_local_zero_cw_and_margin_non_decrease": guard_gate,
            "count_value_native_and_nonactor_exact": immutable_gate,
            "displacement_geometry_global_raise_gate_passed": True,
            "eligible": eligible,
        },
    }


def run_ray(
    source: bytes,
    static: Mapping[str, Any],
    geometry: ModuleType,
    geometry_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper, dependency_evidence = geometry.load_helper()
    torch = helper.torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("requires CUDA with native BF16 support")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")

    formal_payload, formal_evidence = geometry.read_regular_bytes(
        geometry.FORMAL_RESULT,
        geometry.FORMAL_RESULT_SHA256,
        "formal v3 ray result",
    )
    formal = geometry.strict_json_bytes(formal_payload, "formal v3 ray result")
    if (
        formal.get("schema_version") != geometry.FORMAL_RESULT_SCHEMA
        or formal.get("status") != "closed_no_candidate"
        or formal.get("decision", {}).get("selected_candidate") is not None
    ):
        raise RuntimeError("formal v3 result status drift")
    targets = geometry.extract_targets(formal)
    guards = geometry.extract_guards(formal)
    descriptors = [*targets, *guards]
    if len(descriptors) != 19 or len({geometry.identity_key(x) for x in descriptors}) != 19:
        raise RuntimeError("selected 19-row identity drift")

    parent_payload, parent_evidence = geometry.read_regular_bytes(
        geometry.PARENT,
        geometry.PARENT_FILE_SHA256,
        "raw U468 checkpoint",
    )
    checkpoint = torch.load(
        io.BytesIO(parent_payload), map_location="cpu", weights_only=False
    )
    if int(checkpoint.get("update", -1)) != 468:
        raise RuntimeError("raw parent update drift")
    model, model_config, kind = helper.instantiate_checkpoint(checkpoint, device)
    if (
        kind != "ppo"
        or helper.model_state_sha256(model.state_dict())
        != geometry.PARENT_MODEL_STATE_SHA256
    ):
        raise RuntimeError("raw U468 model construction/hash drift")
    raw_full_state_sha = helper.model_state_sha256(model.state_dict())
    parameters = geometry.configure_actor6(model)
    raw_actor = [parameter.detach().clone() for parameter in parameters]
    nonactor_names = sorted(set(model.state_dict()) - set(geometry.ACTOR6_NAMES))
    raw_nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )

    selected_rows, archive_evidence = geometry.load_selected_rows(
        helper, descriptors, model_config
    )
    cpu_batch = helper.evaluator.collate_ordered(
        [item["features"] for item in selected_rows],
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    model.eval()
    outputs = helper.ppo.model_forward(model, batch, device)
    if set(outputs) != set(NATIVE_OUTPUT_KEYS) or any(
        value.dtype != torch.bfloat16 for value in outputs.values()
    ):
        raise RuntimeError("raw geometry forward dtype/key drift")
    with torch.no_grad():
        detached = {key: value.detach() for key, value in outputs.items()}
        raw_actions, _, _, _ = helper.ppo.sample_ordered_actions(
            detached, batch, deterministic=True, canonicalize_order=False
        )
    raw_snapshot = {
        "actions": [[int(value) for value in row] for row in raw_actions],
        "outputs_cpu": {
            key: value.detach().cpu().contiguous() for key, value in outputs.items()
        },
    }
    raw_snapshot["official_rows"] = [
        official_row_record(
            cpu_batch, raw_snapshot["outputs_cpu"], raw_actions, index
        )
        for index in range(19)
    ]
    raw_snapshot["fingerprints"] = {
        key: tensor_sha256(value, torch)
        for key, value in raw_snapshot["outputs_cpu"].items()
    }

    native_logits = outputs["policy_logits"]
    logits = native_logits.float()
    margins = []
    target_required = []
    for index, (descriptor, raw_order) in enumerate(zip(descriptors, raw_actions)):
        if [int(value) for value in raw_order] != descriptor["formal_raw_order"]:
            raise RuntimeError("formal/live raw selected-row order drift")
        positive = int(descriptor["positive_option"])
        negative = int(descriptor["negative_option"])
        margin = logits[index, positive] - logits[index, negative]
        observed = float(margin.detach().cpu())
        if descriptor["role"] == "target":
            if observed != descriptor["expected_raw_margin"]:
                raise RuntimeError("target raw margin drift")
            native_positive = native_logits[index, positive].detach()
            native_negative = native_logits[index, negative].detach()
            spacings = (
                torch.nextafter(
                    native_positive, torch.full_like(native_positive, float("inf"))
                )
                - native_positive,
                native_positive
                - torch.nextafter(
                    native_positive,
                    torch.full_like(native_positive, float("-inf")),
                ),
                torch.nextafter(
                    native_negative, torch.full_like(native_negative, float("inf"))
                )
                - native_negative,
                native_negative
                - torch.nextafter(
                    native_negative,
                    torch.full_like(native_negative, float("-inf")),
                ),
            )
            positive_bf16 = max(float(value.float().cpu()) for value in spacings)
            required = positive_bf16 - observed
            if (
                positive_bf16 != descriptor["expected_positive_bf16_margin"]
                or required != descriptor["required_positive_margin"]
            ):
                raise RuntimeError("target BF16 closure requirement drift")
            target_required.append(required)
        margins.append(margin)

    normalized_gradients = []
    gradient_norms = []
    for index, margin in enumerate(margins):
        flat, _ = geometry.gradient_for_margin(
            margin,
            parameters,
            torch,
            retain_graph=index + 1 < len(margins),
        )
        norm = float(np.linalg.norm(flat))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError("selected-row actor6 gradient norm invalid")
        normalized_gradients.append(flat / norm)
        gradient_norms.append(norm)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("autograd.grad materialized actor6 .grad")
    A = np.stack(normalized_gradients, axis=0)
    direction, direction_audit = geometry.solve_closure_rate_direction(
        A,
        np.asarray(gradient_norms[:5], dtype=np.float64),
        np.asarray(target_required, dtype=np.float64),
        np,
        optimize,
    )
    direction_sha = geometry.vector_sha256_float64_le(direction, np)
    if direction_sha != EXPECTED_DIRECTION_SHA256:
        raise RuntimeError(f"RAM ray direction SHA drift: {direction_sha}")

    direction_tensors = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        value = torch.from_numpy(direction[offset : offset + count].copy())
        direction_tensors.append(
            value.to(device=parameter.device, dtype=parameter.dtype).reshape(
                parameter.shape
            )
        )
        offset += count
    if offset != geometry.EXPECTED_ACTOR6_FLAT_LENGTH:
        raise RuntimeError("direction actor6 slice accounting drift")
    del outputs, logits, margins

    endpoint_results = []
    restore_audits = []
    for radius in RADII:
        set_actor_overlay(parameters, raw_actor, direction_tensors, 0.0, torch)
        pre_hash = helper.model_state_sha256(model.state_dict())
        if pre_hash != raw_full_state_sha:
            raise RuntimeError("raw restore before RAM endpoint failed")
        set_actor_overlay(parameters, raw_actor, direction_tensors, radius, torch)
        nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        displacement = actor_displacement(
            parameters, raw_actor, direction, radius, torch, np
        )
        if not (
            displacement["l2_matches_radius"]
            and displacement["cosine_at_least_threshold"]
        ):
            raise RuntimeError("RAM endpoint displacement geometry drift")
        candidate = snapshot_forward(helper, model, batch, cpu_batch, device)
        endpoint = evaluate_endpoint(
            radius,
            descriptors,
            raw_snapshot,
            candidate,
            cpu_batch,
            torch,
            nonactor_sha == raw_nonactor_sha,
            displacement,
        )
        if (
            not endpoint["immutables"]["count_logits_native_tensor_exact_raw"]
            or not endpoint["immutables"]["value_logits_native_tensor_exact_raw"]
            or not endpoint["immutables"]["nonactor_state_bit_exact_raw"]
            or not endpoint["immutables"]["policy_logits_native_tensor_differs_raw"]
        ):
            raise RuntimeError("RAM endpoint structural/immutable anomaly")
        endpoint_results.append(endpoint)
        set_actor_overlay(parameters, raw_actor, direction_tensors, 0.0, torch)
        post_hash = helper.model_state_sha256(model.state_dict())
        if post_hash != raw_full_state_sha:
            raise RuntimeError("raw restore after RAM endpoint failed")
        restore_audits.append(
            {"radius": radius, "pre_raw_sha256": pre_hash, "post_raw_sha256": post_hash}
        )
    if len(endpoint_results) != len(RADII):
        raise RuntimeError("not all preregistered RAM endpoints were evaluated")

    final_state_sha = helper.model_state_sha256(model.state_dict())
    final_nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    if final_state_sha != raw_full_state_sha or final_nonactor_sha != raw_nonactor_sha:
        raise RuntimeError("final raw/nonactor state is not bit exact")
    final_raw_snapshot = snapshot_forward(helper, model, batch, cpu_batch, device)
    raw_pre_post_recompute_exact = (
        final_raw_snapshot["actions"] == raw_snapshot["actions"]
        and final_raw_snapshot["official_rows"] == raw_snapshot["official_rows"]
        and all(
            torch.equal(
                final_raw_snapshot["outputs_cpu"][key],
                raw_snapshot["outputs_cpu"][key],
            )
            for key in NATIVE_OUTPUT_KEYS
        )
    )
    if not raw_pre_post_recompute_exact:
        raise RuntimeError("same-process raw pre/post recomputation drift")
    eligible = [item for item in endpoint_results if item["gates"]["eligible"]]
    selected = min(eligible, key=lambda item: item["radius"]) if eligible else None
    decision = {
        "status": "selected_ram_candidate" if selected else "closed_no_candidate",
        "all_endpoints_evaluated_before_selection": True,
        "evaluated_radii": list(RADII),
        "selected_radius": selected["radius"] if selected else None,
        "selection_rule": (
            "smallest radius with all five targets correct on four main metrics "
            "and each specified pair margin at its positive-BF16 threshold, "
            "all fourteen guards zero-CW on metrics_union with pair margin not "
            "below raw, and count/value/nonactor exact"
        ),
        "model_materialized": False,
    }
    return {
        "schema_version": SCHEMA,
        "status": decision["status"],
        "scope": {
            "selected_train_rows_only": True,
            "target_rows": 5,
            "guard_rows": 14,
            "ram_endpoints": len(RADII),
            "validation_or_holdout_opened": False,
            "full_panel_evaluation": False,
            "optimizer_created": False,
            "backward_called": False,
            "checkpoint_or_result_write": False,
            "stdout_only": True,
        },
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "geometry": dict(geometry_evidence),
            "parent": parent_evidence,
            "formal_result": formal_evidence,
            "dependencies": dependency_evidence,
            "train_archives": archive_evidence,
        },
        "direction": {
            "sha256": direction_sha,
            "expected_sha256": EXPECTED_DIRECTION_SHA256,
            "hash_exact": True,
            "geometry_audit": direction_audit,
        },
        "ray": {
            "radii": list(RADII),
            "restore_audits": restore_audits,
            "endpoints": endpoint_results,
        },
        "decision": decision,
        "final_integrity": {
            "raw_model_state_sha256": raw_full_state_sha,
            "final_model_state_sha256": final_state_sha,
            "raw_model_final_bit_exact": True,
            "raw_nonactor_sha256": raw_nonactor_sha,
            "final_nonactor_sha256": final_nonactor_sha,
            "nonactor_final_bit_exact": True,
            "all_actor6_grad_buffers_none": all(
                parameter.grad is None for parameter in parameters
            ),
            "same_process_raw_pre_post_recompute_exact": (
                raw_pre_post_recompute_exact
            ),
            "static_zero_write_audit": dict(static),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "ray"), default="ray")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT, sha256_bytes(SCRIPT.read_bytes()), "RAM ray runner"
    )
    static = static_audit(source)
    geometry, geometry_evidence = load_geometry()
    if args.mode == "static":
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "geometry": geometry_evidence,
            "radii": list(RADII),
            "expected_direction_sha256": EXPECTED_DIRECTION_SHA256,
            "audit": static,
        }
    else:
        result = run_ray(source, static, geometry, geometry_evidence)
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
