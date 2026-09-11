#!/usr/bin/env python3
"""V2 zero-write direct512 multi-objective probe.

V1 correctly froze the sole equal-blend direction but used an invalid hard
assumption: one FP32 parameter-space backward over the union was required to
equal six separately reduced backwards to 1e-6 relative L2.  The first formal
probe failed before candidate selection because deterministic FP32 reductions
are reproducible, not algebraically associative.

V2 SHA-binds V1 and leaves its candidate, priority, and SGD suggestion
unchanged.  Before V1 candidate geometry is allowed to run, V2 proves the
actual structural invariant at the policy-logit boundary in float64: the six
disjoint ordered Plackett-Luce objectives exactly reconstruct union loss and
its logit gradient.  V1's FP32 parameter reconstruction is retained only as a
diagnostic.  This tool has no optimizer, backward, step, save, or file-write
path; it prints one JSON object to stdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
V1_TOOL = TOOLS / "probe_u468_raw_direct512_actor6_multiobjective.py"
V1_TOOL_SHA256 = (
    "92e6c599da44ac42a2a497280b0db86e78fc5cd91a40423c9447b4f29c177119"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_link_regular(
    path: Path, expected_sha256: str | None, label: str
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a single-link regular file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


V1_EVIDENCE = require_single_link_regular(
    V1_TOOL, V1_TOOL_SHA256, "frozen v1 direct512 probe"
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_V1_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_direct512_multiobjective_v1_92e6c599", V1_TOOL
)
if _V1_SPEC is None or _V1_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen v1 probe import spec")
v1: ModuleType = importlib.util.module_from_spec(_V1_SPEC)
_V1_SPEC.loader.exec_module(v1)

torch = v1.torch
ppo = v1.ppo
frozen = v1.frozen
aggregate = v1.aggregate


SCHEMA = "ptcg-u468-raw-direct512-actor6-multiobjective-probe-v2"
ROWS = v1.ROWS
PARTITIONS = tuple(v1.PARTITIONS)
STRUCTURAL_OBJECTIVES = ("union", *PARTITIONS)
FLOAT64_TOLERANCE = 1e-12
V1_PARAMETER_RECONSTRUCTION_BYPASS_RELATIVE = 1.0
V1_PARAMETER_RECONSTRUCTION_BYPASS_MAX_ABS = 1.0


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def ast_audit() -> dict[str, Any]:
    source = Path(__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__).resolve()))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [
        dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    forbidden_suffixes = (
        ".backward",
        ".step",
        ".zero_grad",
        ".save",
        ".write_text",
        ".write_bytes",
    )
    forbidden = sorted(
        name
        for name in calls
        if any(name.endswith(suffix) for suffix in forbidden_suffixes)
        or name.startswith("torch.optim")
    )
    if forbidden:
        raise RuntimeError(f"v2 zero-write AST gate failed: {forbidden}")
    expected = {
        "torch.autograd.grad": 1,
        "ppo.model_forward": 1,
        "v1.run_probe": 1,
    }
    observed = {name: calls.count(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"v2 call-site count drift: {observed}")
    return {
        "status": "zero_write_v2_ast_audit_passed",
        "call_sites": observed,
        "forbidden_update_or_write_calls": [],
        "v1_sha_checked_before_import": True,
        "stdout_only": True,
    }


def ordered_plackett_luce_nll_float64(
    logits: torch.Tensor, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    if logits.dtype != torch.float64 or logits.ndim != 2:
        raise RuntimeError("structural logits must be rank-2 float64")
    option_mask = batch["option_mask"].bool()
    action_counts = batch["action_counts"]
    sequences = batch["action_sequences"]
    if logits.shape != option_mask.shape:
        raise RuntimeError("structural logits/option-mask shape drift")
    selected = torch.zeros_like(option_mask)
    per_row = torch.zeros(
        logits.shape[0], dtype=torch.float64, device=logits.device
    )
    for step in range(sequences.shape[1]):
        active = step < action_counts
        if not bool(active.any()):
            break
        raw_chosen = sequences[:, step]
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        active_chosen = raw_chosen[active]
        if (
            bool((active_chosen < 0).any())
            or bool((active_chosen >= logits.shape[1]).any())
            or not bool(
                (
                    option_mask[active_rows, active_chosen]
                    & ~selected[active_rows, active_chosen]
                ).all()
            )
        ):
            raise RuntimeError("illegal or duplicate structural expert action")
        allowed = option_mask & ~selected
        log_probs = torch.log_softmax(
            logits.masked_fill(~allowed, -1e300), dim=1
        )
        chosen = raw_chosen.clamp(0, logits.shape[1] - 1)
        chosen_log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
        per_row -= torch.where(
            active, chosen_log_prob, torch.zeros_like(chosen_log_prob)
        )
        selected.scatter_(1, chosen.unsqueeze(1), active.unsqueeze(1))
    if not bool(torch.isfinite(per_row).all()):
        raise FloatingPointError("float64 structural NLL is nonfinite")
    return per_row


def float64_logit_structural_audit(
    native_logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    masks_cpu: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    logits = native_logits.detach().to(torch.float64).requires_grad_(True)
    per_row = ordered_plackett_luce_nll_float64(logits, batch)
    weights = batch["sample_weights"].to(torch.float64) * torch.where(
        batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
        torch.full_like(batch["sample_weights"].to(torch.float64), 8.0),
        torch.ones_like(batch["sample_weights"].to(torch.float64)),
    )
    masks = {
        name: masks_cpu[name].to(device=logits.device, dtype=torch.bool)
        for name in STRUCTURAL_OBJECTIVES
    }
    if tuple(masks) != STRUCTURAL_OBJECTIVES:
        raise RuntimeError(f"structural mask order/set drift: {tuple(masks)}")
    cover = torch.stack([masks[name] for name in PARTITIONS]).sum(dim=0)
    partition_rows = {name: int(masks[name].sum()) for name in PARTITIONS}
    if (
        not torch.equal(cover, torch.ones_like(cover))
        or partition_rows != v1.EXPECTED_PARTITION_ROWS
        or int(masks["union"].sum()) != ROWS
    ):
        raise RuntimeError("float64 structural partitions are not exact")

    scalars: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    effective_weights: dict[str, float] = {}
    for index, name in enumerate(STRUCTURAL_OBJECTIVES):
        mask = masks[name]
        denominator = weights[mask].sum()
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
            raise RuntimeError(f"invalid structural denominator: {name}")
        scalar = (per_row[mask] * weights[mask]).sum() / denominator
        gradient = torch.autograd.grad(
            scalar,
            logits,
            retain_graph=index + 1 < len(STRUCTURAL_OBJECTIVES),
            create_graph=False,
            allow_unused=False,
            materialize_grads=False,
        )[0]
        if not bool(torch.isfinite(scalar)) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"nonfinite structural objective: {name}")
        scalars[name] = scalar.detach()
        gradients[name] = gradient.detach()
        effective_weights[name] = float(denominator.detach())

    union_weight = effective_weights["union"]
    partition_weight_sum = sum(effective_weights[name] for name in PARTITIONS)
    reconstructed_loss = sum(
        effective_weights[name] * float(scalars[name]) for name in PARTITIONS
    ) / union_weight
    reconstructed_gradient = sum(
        (
            effective_weights[name] * gradients[name]
            for name in PARTITIONS
        ),
        torch.zeros_like(gradients["union"]),
    ) / union_weight
    gradient_delta = reconstructed_gradient - gradients["union"]
    gradient_absolute_l2 = float(gradient_delta.norm())
    gradient_reference_l2 = float(gradients["union"].norm())
    gradient_relative_l2 = gradient_absolute_l2 / max(
        gradient_reference_l2, 1e-300
    )
    gradient_max_abs = float(gradient_delta.abs().max())
    loss_absolute = abs(float(scalars["union"]) - reconstructed_loss)
    loss_relative = loss_absolute / max(abs(float(scalars["union"])), 1e-300)
    weight_absolute = abs(union_weight - partition_weight_sum)
    passed = bool(
        weight_absolute <= FLOAT64_TOLERANCE
        and loss_absolute <= FLOAT64_TOLERANCE
        and loss_relative <= FLOAT64_TOLERANCE
        and gradient_relative_l2 <= FLOAT64_TOLERANCE
        and gradient_max_abs <= FLOAT64_TOLERANCE
    )
    result = {
        "kind": "detached_native_policy_logits_float64_ordered_plackett_luce",
        "hard_gate": True,
        "tolerance": FLOAT64_TOLERANCE,
        "rows": ROWS,
        "partition_rows": partition_rows,
        "six_partitions_disjoint_and_exhaustive": True,
        "effective_weights": effective_weights,
        "union_effective_weight": union_weight,
        "partition_effective_weight_sum": partition_weight_sum,
        "weight_absolute_difference": weight_absolute,
        "union_loss": float(scalars["union"]),
        "reconstructed_union_loss": reconstructed_loss,
        "loss_absolute_difference": loss_absolute,
        "loss_relative_difference": loss_relative,
        "union_logit_gradient_l2": gradient_reference_l2,
        "gradient_absolute_l2": gradient_absolute_l2,
        "gradient_relative_l2": gradient_relative_l2,
        "gradient_max_abs": gradient_max_abs,
        "pure_float64_loss_implementation": True,
        "bc_expert_actor_loss_not_used": True,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(
            "float64 logit structural reconstruction gate failed: "
            + json.dumps(result, sort_keys=True)
        )
    return result


def build_cache_context() -> tuple[
    dict[str, Any],
    list[list[dict[str, Any]]],
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    cache_audit, selections, union = aggregate.build_cache_audit()
    identities = v1.flat_identities(selections)
    masks, weight_audit, _ = v1.build_masks_and_weights(union, identities)
    if (
        cache_audit["aggregate512"]["cache_sha256"]
        != v1.EXPECTED_DIRECT_CACHE_SHA256
        or cache_audit["aggregate512"]["batch_sha256"]
        != v1.EXPECTED_DIRECT_BATCH_SHA256
    ):
        raise RuntimeError("v2 direct512 cache identity drift")
    return cache_audit, selections, union, identities, masks, weight_audit


def run_v2_probe(
    union: Mapping[str, torch.Tensor],
    identities: Sequence[Mapping[str, Any]],
    masks: Mapping[str, torch.Tensor],
    weight_audit: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v2 probe is CUDA-only")
    random.seed(v1.PROBE_SEED)
    torch.manual_seed(v1.PROBE_SEED)
    torch.cuda.manual_seed_all(v1.PROBE_SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)

    structural_model, _ = frozen.load_raw_u468(device)
    structural_model.eval()
    frozen.configure_actor6(structural_model)
    structural_hash_before = ppo.model_state_sha256(structural_model)
    batch = {
        key: value.to(device, non_blocking=True) for key, value in union.items()
    }
    with torch.no_grad():
        native_outputs = ppo.model_forward(structural_model, batch, device)
    if native_outputs["policy_logits"].dtype != torch.bfloat16:
        raise RuntimeError("v2 native policy logits are not BF16")
    structural = float64_logit_structural_audit(
        native_outputs["policy_logits"], batch, masks
    )
    structural_hash_after = ppo.model_state_sha256(structural_model)
    if structural_hash_after != structural_hash_before:
        raise RuntimeError("v2 structural audit changed model tensors")
    if any(parameter.grad is not None for parameter in structural_model.parameters()):
        raise RuntimeError("v2 structural audit materialized parameter gradients")
    del native_outputs, structural_model

    original_relative = v1.FP32_RECONSTRUCTION_RELATIVE_L2_MAX
    original_max_abs = v1.FP32_RECONSTRUCTION_MAX_ABS_MAX
    if original_relative != 1e-6 or original_max_abs != 5e-7:
        raise RuntimeError("frozen v1 reconstruction thresholds drift")
    v1.FP32_RECONSTRUCTION_RELATIVE_L2_MAX = (
        V1_PARAMETER_RECONSTRUCTION_BYPASS_RELATIVE
    )
    v1.FP32_RECONSTRUCTION_MAX_ABS_MAX = (
        V1_PARAMETER_RECONSTRUCTION_BYPASS_MAX_ABS
    )
    try:
        probe = v1.run_probe(union, identities, masks, weight_audit, device)
    finally:
        v1.FP32_RECONSTRUCTION_RELATIVE_L2_MAX = original_relative
        v1.FP32_RECONSTRUCTION_MAX_ABS_MAX = original_max_abs
    if (
        v1.FP32_RECONSTRUCTION_RELATIVE_L2_MAX != original_relative
        or v1.FP32_RECONSTRUCTION_MAX_ABS_MAX != original_max_abs
    ):
        raise RuntimeError("v1 diagnostic thresholds were not restored")

    parameter_reconstruction = probe.pop("fp32_structural_reconstruction")
    internal_bypass_pass = bool(parameter_reconstruction.pop("pass"))
    original_parameter_pass = bool(
        parameter_reconstruction["relative_l2"] <= original_relative
        and parameter_reconstruction["max_abs"] <= original_max_abs
    )
    parameter_reconstruction.update(
        {
            "relative_l2_max": original_relative,
            "max_abs_max": original_max_abs,
            "pass": original_parameter_pass,
            "pass_under_original_v1_threshold": original_parameter_pass,
            "pass_under_temporary_internal_bypass": internal_bypass_pass,
            "gate_enforced_in_v2": False,
            "classification": "diagnostic_only_fp32_parameter_reduction",
            "reason": (
                "deterministic FP32 parameter reductions are reproducible but "
                "not algebraically associative across union and six backwards"
            ),
            "temporary_internal_bypass_relative": (
                V1_PARAMETER_RECONSTRUCTION_BYPASS_RELATIVE
            ),
            "temporary_internal_bypass_max_abs": (
                V1_PARAMETER_RECONSTRUCTION_BYPASS_MAX_ABS
            ),
            "temporary_thresholds_restored": True,
        }
    )
    probe["status"] = (
        "completed_zero_weight_write_direct512_multiobjective_probe_v2"
    )
    probe["float64_logit_structural_reconstruction"] = structural
    probe["fp32_parameter_reconstruction_diagnostic_only"] = (
        parameter_reconstruction
    )
    probe["v2_gate_model"] = {
        "hard_gate": "float64_logit_structural_reconstruction",
        "fp32_parameter_reconstruction_gate": False,
        "candidate_formula_changed_from_v1": False,
        "candidate_priority_changed_from_v1": False,
        "sgd_suggestion_changed_from_v1": False,
        "hard_gate_passed_before_v1_candidate_geometry": True,
    }
    probe["integrity"].update(
        {
            "v1_probe_tool_sha256": V1_TOOL_SHA256,
            "v2_structural_model_state_sha256_before": structural_hash_before,
            "v2_structural_model_state_sha256_after": structural_hash_after,
            "v2_structural_model_weights_unchanged": (
                structural_hash_before == structural_hash_after
            ),
            "v2_extra_native_no_grad_forward_calls": 1,
            "v2_float64_logit_autograd_calls": len(STRUCTURAL_OBJECTIVES),
            "v2_parameter_updates": 0,
            "v2_result_artifact_writes": 0,
        }
    )
    if not frozen.repair.finite_nested(probe):
        raise FloatingPointError("v2 probe result contains nonfinite values")
    return probe


def common_report(mode: str, self_evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "zero_write_v2_audit_passed",
        "mode": mode,
        "tool": dict(self_evidence),
        "frozen_v1_probe": dict(V1_EVIDENCE),
        "aggregate_runner": dict(v1.AGGREGATE_RUNNER_EVIDENCE),
        "ast_audit": ast_audit(),
        "contract": {
            "base": "raw full U468",
            "rows": ROWS,
            "direct512_cache_sha256": v1.EXPECTED_DIRECT_CACHE_SHA256,
            "direct512_batch_sha256": v1.EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": v1.EXPECTED_FLAT_IDENTITY_SHA256,
            "sole_selectable_candidate": v1.EQUAL_BLEND_CANDIDATE,
            "equal_blend_formula": (
                "0.5*L_union + "
                "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
            ),
            "sole_optimizer_suggestion": {
                "name": "SGD",
                "learning_rate": v1.SUGGESTED_SGD_LEARNING_RATE,
                "momentum": v1.SUGGESTED_SGD_MOMENTUM,
                "weight_decay": v1.SUGGESTED_SGD_WEIGHT_DECAY,
            },
            "hard_structural_gate": {
                "space": "detached native policy logits",
                "dtype": "torch.float64",
                "loss": "pure ordered Plackett-Luce",
                "objectives": list(STRUCTURAL_OBJECTIVES),
                "tolerance": FLOAT64_TOLERANCE,
            },
            "fp32_parameter_reconstruction": "diagnostic_only",
            "train_only": True,
            "validation": False,
            "optimizer": False,
            "backward": False,
            "optimizer_step": False,
            "parameter_update": False,
            "checkpoint_write": False,
            "model_write": False,
            "result_artifact_write": False,
            "stdout_only": True,
        },
        "writes_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("static-audit", "cache-audit", "probe"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    v1.assert_frozen_aggregate_contract()
    self_path = Path(__file__).resolve()
    if self_path.parent != TOOLS:
        raise RuntimeError("v2 probe must remain in repository tools directory")
    self_evidence = require_single_link_regular(self_path, None, "v2 probe tool")
    common = common_report(args.mode, self_evidence)
    if args.mode == "static-audit":
        if args.device != "cpu":
            raise ValueError("static-audit uses the default CPU device argument")
        print(json.dumps(common, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return

    cache_audit, selections, union, identities, masks, weight_audit = (
        build_cache_context()
    )
    cache_report = common | {
        "status": "zero_write_v2_cache_audit_passed",
        "cache": cache_audit,
        "flat_identity": {
            "rows": len(identities),
            "sha256": v1.EXPECTED_FLAT_IDENTITY_SHA256,
            "unique": True,
        },
        "weight_and_partition_audit": weight_audit,
    }
    if args.mode == "cache-audit":
        if args.device != "cpu":
            raise ValueError("cache-audit uses the default CPU device argument")
        print(
            json.dumps(
                cache_report, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        )
        return
    if args.device != "cuda":
        raise ValueError("formal v2 probe requires --device cuda")
    probe = run_v2_probe(
        union, identities, masks, weight_audit, torch.device("cuda")
    )
    final = cache_report | {
        "status": probe["status"],
        "mode": "probe",
        "probe": probe,
    }
    print(json.dumps(final, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
