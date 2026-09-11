#!/usr/bin/env python3
"""Train-only RAM probe of BF16 behavior along the frozen step-1 PCGrad ray.

The probe rehydrates only the 301 treatment and 467 retention rows frozen in
the v1 profile.  It recomputes the step-1 treatment/retention gradients at the
source, forms the same symmetric PCGrad direction as the frozen runner, and
evaluates a fixed list of actor6 L2 radii.  Every radius starts from a fresh
source model and policy behavior is decoded by a real 61-count-class clone.

The only output is a new JSON audit.  No model artifact is serialized and no
radius is selected by this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_yanz_source_error_actor6_v1 as profile_tool  # noqa: E402
import run_yanz_actor6_pcgrad_specialbc_v1 as runner  # noqa: E402


SCHEMA = "ptcg-yanz-actor6-pcgrad-bf16-radii-probe-v1"
SCRIPT = Path(__file__).resolve()
PROFILE = ROOT / (
    "data/yanz_alakazam_actor6_pcgrad_specialbc_20260810_v1/"
    "train_only_profile.json"
)
PROTOCOL = ROOT / (
    "artifacts/yanz_alakazam_actor6_pcgrad_specialbc_20260810_v1/"
    "protocol.json"
)
PRIOR_AUDIT = ROOT / (
    "artifacts/yanz_alakazam_actor6_pcgrad_specialbc_20260810_v1/"
    "train_only_audit.json"
)

SOURCE_SHA256 = (
    "f5500086c16a02c19f3f2abce5e144446bd079248fd9fc3f4a620f3d079c7626"
)
PROFILE_SHA256 = (
    "f6b92793e38dedba5a4ee4bfadbbade9e076dd71f83d80b4b6735edde5996e91"
)
RUNNER_SHA256 = (
    "286106d7d646c88b07e529724d858418514f1f88a6791fc6a1744b0341d568d7"
)
PROTOCOL_SHA256 = (
    "55aef2884f0de00609fb724e0aba9daa92b6f39075c938c8382f3f8d6789c5a0"
)
PRIOR_AUDIT_SHA256 = (
    "dda36349201363038c7c02577b04f9e3b8f6a1380682785c654ac7b662735ad4"
)

RADII = (
    0.0,
    2.5e-5,
    5.0e-5,
    1.0e-4,
    2.0e-4,
    4.0e-4,
    8.0e-4,
    1.6e-3,
    3.2e-3,
)
EVAL_BATCH_SIZE = 256
ACTOR6 = runner.ACTOR6


class ProbeError(RuntimeError):
    """A fail-closed binding or train-only probe error."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError(f"{label} is not an object")
    return value


def validate_frozen_bindings() -> dict[str, dict[str, str]]:
    """Validate only frozen code/metadata plus source and selected train data."""

    expected_files = {
        "source_checkpoint": (profile_tool.SOURCE, SOURCE_SHA256),
        "profile": (PROFILE, PROFILE_SHA256),
        "runner": (runner.RUNNER, RUNNER_SHA256),
        "protocol": (PROTOCOL, PROTOCOL_SHA256),
        "prior_train_only_audit": (PRIOR_AUDIT, PRIOR_AUDIT_SHA256),
    }
    result: dict[str, dict[str, str]] = {}
    for label, (path, expected) in expected_files.items():
        actual = file_sha256(path)
        if actual != expected:
            raise ProbeError(f"frozen {label} SHA drift: {actual} != {expected}")
        result[label] = {"path": str(path.resolve()), "sha256": actual}

    if profile_tool.SOURCE_SHA256 != SOURCE_SHA256:
        raise ProbeError("profiler source constant drift")
    profile = runner.load_json_strict(PROFILE)
    runner.validate_profile(profile)

    protocol = runner.load_json_strict(PROTOCOL)
    runner.validate_protocol(protocol)
    protocol_bindings = _require_mapping(protocol.get("bindings"), "protocol bindings")
    protocol_source = _require_mapping(
        protocol_bindings.get("source_checkpoint"), "protocol source binding"
    )
    protocol_profile = _require_mapping(
        protocol_bindings.get("profile"), "protocol profile binding"
    )
    protocol_tools = _require_mapping(
        protocol_bindings.get("tools"), "protocol tool bindings"
    )
    protocol_runner = _require_mapping(
        protocol_tools.get("runner"), "protocol runner binding"
    )
    if protocol_source.get("sha256") != SOURCE_SHA256:
        raise ProbeError("frozen protocol source SHA drift")
    if protocol_profile.get("sha256") != PROFILE_SHA256:
        raise ProbeError("frozen protocol profile SHA drift")
    if protocol_runner.get("sha256") != RUNNER_SHA256:
        raise ProbeError("frozen protocol runner SHA drift")

    prior = runner.load_json_strict(PRIOR_AUDIT)
    if prior.get("schema_version") != runner.AUDIT_SCHEMA:
        raise ProbeError("prior train-only audit schema drift")
    if prior.get("status") != "rejected_train_only":
        raise ProbeError("prior train-only audit status drift")
    prior_protocol = _require_mapping(prior.get("protocol"), "prior protocol binding")
    prior_profile = _require_mapping(prior.get("profile"), "prior profile binding")
    if prior_protocol.get("sha256") != PROTOCOL_SHA256:
        raise ProbeError("prior audit protocol SHA drift")
    if prior_profile.get("sha256") != PROFILE_SHA256:
        raise ProbeError("prior audit profile SHA drift")
    if prior.get("candidate_written") is not False:
        raise ProbeError("prior audit unexpectedly wrote a model artifact")
    if prior.get("heldout_evaluation_performed") is not False:
        raise ProbeError("prior audit unexpectedly performed non-train evaluation")
    if prior.get("submission_performed") is not False:
        raise ProbeError("prior audit unexpectedly performed a submission")
    replay_a = _require_mapping(prior.get("replay_a"), "prior replay A")
    steps = replay_a.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ProbeError("prior rejected replay step ledger drift")
    first = _require_mapping(steps[0], "prior replay step 1")
    rows = _require_mapping(first.get("rows"), "prior replay step-1 rows")
    if int(first.get("step", -1)) != 1 or rows != {
        "treatment": 101,
        "retention": 155,
    }:
        raise ProbeError("prior replay is not the bound failed step 1")
    return result


def vector_dot(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> float:
    if len(left) != len(right):
        raise ProbeError("vector tensor-count mismatch")
    return float(
        sum(
            (
                left_value.detach().cpu().double().reshape(-1)
                * right_value.detach().cpu().double().reshape(-1)
            ).sum()
            for left_value, right_value in zip(left, right)
        )
    )


def vector_norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(vector_dot(values, values), 0.0))


def vector_sha256(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ptcg-actor6-vector-float64-v1\0")
    for name, value in zip(ACTOR6, values):
        tensor = value.detach().cpu().double().contiguous()
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(len(tensor.shape).to_bytes(4, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=True))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    if len(values) != len(ACTOR6):
        raise ProbeError("actor6 vector tensor-count drift")
    return digest.hexdigest()


def scaled_descent(
    direction: Sequence[torch.Tensor], radius: float
) -> tuple[torch.Tensor, ...]:
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    norm = vector_norm(direction)
    if norm <= 0.0:
        raise ProbeError("cannot scale a zero direction")
    scale = radius / norm
    return tuple(value.detach().double() * scale for value in direction)


def delta_geometry(
    actual_delta: Sequence[torch.Tensor],
    treatment_gradient: Sequence[torch.Tensor],
    retention_gradient: Sequence[torch.Tensor],
    pcgrad_direction: Sequence[torch.Tensor] | None = None,
) -> dict[str, Any]:
    delta_norm = vector_norm(actual_delta)
    effects: dict[str, dict[str, float | None]] = {}
    references: list[tuple[str, Sequence[torch.Tensor]]] = [
        ("treatment", treatment_gradient),
        ("retention", retention_gradient),
    ]
    if pcgrad_direction is not None:
        references.append(("pcgrad_direction", pcgrad_direction))
    for label, reference in references:
        reference_norm = vector_norm(reference)
        dot = vector_dot(actual_delta, reference)
        cosine = (
            dot / (delta_norm * reference_norm)
            if delta_norm > 0.0 and reference_norm > 0.0
            else None
        )
        effects[label] = {"dot": dot, "cosine": cosine}
    return {
        "actual_delta_l2": delta_norm,
        "actual_nonzero_scalar_count": sum(
            int(torch.count_nonzero(value.detach()).item()) for value in actual_delta
        ),
        "effects": effects,
    }


def apply_scaled_descent(
    model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
    intended_delta: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Apply one FP32 SGD-style displacement and return the actual cast delta."""

    named = dict(model.named_parameters())
    if len(intended_delta) != len(ACTOR6):
        raise ProbeError("intended actor6 delta tensor-count drift")
    with torch.no_grad():
        for name, value in zip(ACTOR6, intended_delta):
            parameter = named[name]
            source = source_state[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
            parameter.copy_(source)
            parameter.add_(
                value.to(device=parameter.device, dtype=parameter.dtype), alpha=-1.0
            )
    return tuple(
        source_state[name].double()
        - named[name].detach().cpu().double()
        for name in ACTOR6
    )


def _evaluate_rows(
    raw_model: torch.nn.Module,
    expanded_model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate selected rows only, decoding actions with the expanded clone."""

    row_ids: list[str] = []
    contexts: list[int] = []
    actions: dict[str, list[int]] = {}
    correct: dict[str, bool] = {}
    per_row_nll: dict[str, float] = {}
    policy_logits: list[torch.Tensor] = []
    weighted_numerator = 0.0
    weight_total = 0.0
    raw_expanded_policy_bitwise = True
    expanded_count_classes: int | None = None
    raw_model.eval()
    expanded_model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH_SIZE):
            chunk = rows[start : start + EVAL_BATCH_SIZE]
            batch, chunk_ids = runner.concatenate_batches(chunk, config, device)
            raw_outputs = runner.ppo.model_forward(raw_model, dict(batch), device)
            expanded_outputs = runner.ppo.model_forward(
                expanded_model, dict(batch), device
            )
            classes = int(expanded_outputs["count_logits"].shape[1])
            if expanded_count_classes is None:
                expanded_count_classes = classes
            if classes != 61 or expanded_count_classes != classes:
                raise ProbeError("behavior path is not a stable real expanded61 clone")
            policy_exact = torch.equal(
                raw_outputs["policy_logits"], expanded_outputs["policy_logits"]
            )
            raw_expanded_policy_bitwise = raw_expanded_policy_bitwise and policy_exact
            if not policy_exact:
                raise ProbeError("raw17 and expanded61 policy logits differ")
            row_loss, weights = runner.hybrid_policy_nll_per_row(raw_outputs, batch)
            weighted_numerator += float(
                (row_loss.detach().double() * weights.detach().double()).sum().cpu()
            )
            weight_total += float(weights.detach().double().sum().cpu())
            predicted = profile_tool.deterministic_policy_actions(
                expanded_outputs, batch
            )
            sequences = batch["action_sequences"].detach().cpu()
            counts = batch["action_counts"].detach().cpu().tolist()
            chunk_contexts = batch["contexts"].detach().cpu().tolist()
            losses = row_loss.detach().cpu().tolist()
            for index, line_hash in enumerate(chunk_ids):
                if line_hash in correct:
                    raise ProbeError("selected row repeats during behavior evaluation")
                context = int(chunk_contexts[index])
                expert = sequences[index, : int(counts[index])].tolist()
                policy_action = [int(value) for value in predicted[index]]
                hybrid = policy_action if context == 34 else sorted(policy_action)
                row_ids.append(line_hash)
                contexts.append(context)
                actions[line_hash] = policy_action
                correct[line_hash] = hybrid == expert
                per_row_nll[line_hash] = float(losses[index])
            policy_logits.append(
                expanded_outputs["policy_logits"].detach().cpu().clone()
            )
    if len(row_ids) != len(rows) or len(set(row_ids)) != len(rows):
        raise ProbeError("selected behavior row cardinality drift")
    if weight_total <= 0.0:
        raise ProbeError("selected behavior weight is not positive")
    return {
        "row_ids": row_ids,
        "contexts": contexts,
        "actions": actions,
        "correct": correct,
        "per_row_nll": per_row_nll,
        "weighted_loss": weighted_numerator / weight_total,
        "policy_logits": tuple(policy_logits),
        "expanded_count_classes": expanded_count_classes,
        "raw_expanded_policy_bitwise": raw_expanded_policy_bitwise,
    }


def compare_behavior(
    source: Mapping[str, Any], moved: Mapping[str, Any]
) -> dict[str, Any]:
    source_ids = list(source["row_ids"])
    moved_ids = list(moved["row_ids"])
    if source_ids != moved_ids:
        raise ProbeError("behavior row order drift")
    if source["contexts"] != moved["contexts"]:
        raise ProbeError("behavior context drift")
    if source["expanded_count_classes"] != 61 or moved["expanded_count_classes"] != 61:
        raise ProbeError("behavior comparison did not use expanded61 models")
    source_correct = _require_mapping(source["correct"], "source correctness")
    moved_correct = _require_mapping(moved["correct"], "moved correctness")
    source_actions = _require_mapping(source["actions"], "source actions")
    moved_actions = _require_mapping(moved["actions"], "moved actions")
    source_nll = _require_mapping(source["per_row_nll"], "source row NLL")
    moved_nll = _require_mapping(moved["per_row_nll"], "moved row NLL")

    repairs = [
        key for key in source_ids if not source_correct[key] and moved_correct[key]
    ]
    correctness_harm = [
        key for key in source_ids if source_correct[key] and not moved_correct[key]
    ]
    changed_actions = [
        key for key in source_ids if source_actions[key] != moved_actions[key]
    ]
    wrong_to_other_wrong = [
        key
        for key in changed_actions
        if not source_correct[key] and not moved_correct[key]
    ]
    nll_improved = [key for key in source_ids if moved_nll[key] < source_nll[key]]
    nll_worsened = [key for key in source_ids if moved_nll[key] > source_nll[key]]
    nll_unchanged = [
        key for key in source_ids if moved_nll[key] == source_nll[key]
    ]

    source_logits = source["policy_logits"]
    moved_logits = moved["policy_logits"]
    if not isinstance(source_logits, tuple) or not isinstance(moved_logits, tuple):
        raise ProbeError("behavior policy-logit chunks missing")
    if len(source_logits) != len(moved_logits):
        raise ProbeError("behavior policy-logit chunk count drift")
    changed_elements = 0
    max_abs_delta = 0.0
    for left, right in zip(source_logits, moved_logits):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise ProbeError("behavior policy-logit chunk type drift")
        if left.shape != right.shape:
            raise ProbeError("behavior policy-logit shape drift")
        changed_elements += int(torch.count_nonzero(left != right).item())
        if left.numel():
            max_abs_delta = max(
                max_abs_delta,
                float((left.float() - right.float()).abs().max()),
            )
    return {
        "rows": len(source_ids),
        "loss": {
            "source": float(source["weighted_loss"]),
            "moved": float(moved["weighted_loss"]),
            "delta": float(moved["weighted_loss"])
            - float(source["weighted_loss"]),
        },
        "correct": {
            "source": sum(bool(source_correct[key]) for key in source_ids),
            "moved": sum(bool(moved_correct[key]) for key in source_ids),
        },
        "repairs": {"count": len(repairs), "line_sha256": repairs},
        "harm": {
            "definition": (
                "correctness harm is source-correct to moved-wrong; NLL harm is "
                "strict per-row hybrid-policy NLL increase"
            ),
            "source_correct_to_wrong_count": len(correctness_harm),
            "source_correct_to_wrong_line_sha256": correctness_harm,
            "nll_worsened_count": len(nll_worsened),
            "nll_worsened_line_sha256": nll_worsened,
            "wrong_to_different_wrong_action_count": len(wrong_to_other_wrong),
            "wrong_to_different_wrong_action_line_sha256": wrong_to_other_wrong,
        },
        "nll_row_transitions": {
            "improved": len(nll_improved),
            "worsened": len(nll_worsened),
            "unchanged": len(nll_unchanged),
        },
        "action_changed_count": len(changed_actions),
        "action_changed_line_sha256": changed_actions,
        "policy_quantization": {
            "changed_elements": changed_elements,
            "max_abs_delta": max_abs_delta,
            "all_policy_logits_bitwise_source": changed_elements == 0,
        },
        "expanded_count_classes": 61,
        "raw_expanded_policy_bitwise": bool(
            source["raw_expanded_policy_bitwise"]
            and moved["raw_expanded_policy_bitwise"]
        ),
    }


def _loss_only_comparison(
    source: Mapping[str, Any], moved: Mapping[str, Any]
) -> dict[str, Any]:
    full = compare_behavior(source, moved)
    return {
        "rows": full["rows"],
        "loss": full["loss"],
        "correct": full["correct"],
        "action_changed_count": full["action_changed_count"],
        "policy_quantization": full["policy_quantization"],
        "expanded_count_classes": full["expanded_count_classes"],
        "raw_expanded_policy_bitwise": full["raw_expanded_policy_bitwise"],
    }


def _flatten_selected(
    selected: Sequence[Mapping[str, Sequence[dict[str, Any]]]], label: str
) -> list[dict[str, Any]]:
    rows = [row for step in selected for row in step[label]]
    hashes = [str(row["line_sha256"]) for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ProbeError(f"duplicate {label} row in frozen profile")
    return rows


def run_probe(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    if not torch.cuda.is_available():
        raise ProbeError("CUDA is required for the BF16 behavior probe")
    if not torch.cuda.is_bf16_supported():
        raise ProbeError("the CUDA device does not support BF16")
    device = torch.device("cuda")
    bindings = validate_frozen_bindings()
    profile = runner.load_json_strict(PROFILE)
    source_checkpoint, source_model, config = runner.load_source(device)
    source_deployment = runner.deployment_clone_from_raw(
        source_checkpoint, source_model, device
    )
    if source_deployment.count_head[-1].out_features != 61:
        raise ProbeError("source deployment clone is not expanded61")
    selected = runner.load_selected_features(profile, config)
    if len(selected) != 3:
        raise ProbeError("frozen selected-step count drift")
    treatment_all = _flatten_selected(selected, "treatment")
    retention_all = _flatten_selected(selected, "retention")
    if len(treatment_all) != 301 or len(retention_all) != 467:
        raise ProbeError("frozen selected-row union cardinality drift")
    step1_treatment = list(selected[0]["treatment"])
    step1_retention = list(selected[0]["retention"])
    if len(step1_treatment) != 101 or len(step1_retention) != 155:
        raise ProbeError("frozen step-1 row cardinality drift")

    parameters = runner.configure_actor6(source_model)
    source_state = runner.tensor_state_clone(source_model)
    step1_t_batch, _ = runner.concatenate_batches(
        step1_treatment, config, device
    )
    step1_r_batch, _ = runner.concatenate_batches(
        step1_retention, config, device
    )
    treatment_loss, treatment_gradient = runner.task_gradient(
        source_model, parameters, step1_t_batch
    )
    retention_loss, retention_gradient = runner.task_gradient(
        source_model, parameters, step1_r_batch
    )
    direction, pcgrad = runner.symmetric_two_task_pcgrad(
        treatment_gradient, retention_gradient
    )
    if not pcgrad["pass"]:
        raise ProbeError("bound step-1 PCGrad direction is no longer valid")
    if any(parameter.grad is not None for parameter in parameters):
        raise ProbeError("autograd unexpectedly materialized actor6 grad buffers")

    source_behaviors = {
        "step1_treatment": _evaluate_rows(
            source_model, source_deployment, step1_treatment, config, device
        ),
        "step1_retention": _evaluate_rows(
            source_model, source_deployment, step1_retention, config, device
        ),
        "all301_treatment": _evaluate_rows(
            source_model, source_deployment, treatment_all, config, device
        ),
        "all467_retention": _evaluate_rows(
            source_model, source_deployment, retention_all, config, device
        ),
    }
    if any(source_behaviors["all301_treatment"]["correct"].values()):
        raise ProbeError("all301 treatment source baseline contains a correct row")
    if not all(source_behaviors["all467_retention"]["correct"].values()):
        raise ProbeError("all467 retention source baseline contains a wrong row")

    radius_reports: list[dict[str, Any]] = []
    for radius in RADII:
        moved_model, _ = profile_tool.instantiate_raw_bc(
            source_checkpoint, device
        )
        before_hash = profile_tool.bitwise_model_state_sha256(
            moved_model.state_dict()
        )
        if before_hash != profile_tool.SOURCE_BITWISE_STATE_SHA256:
            raise ProbeError("radius did not start independently from source")
        intended_delta = scaled_descent(direction, radius)
        actual_delta = apply_scaled_descent(
            moved_model, source_state, intended_delta
        )
        moved_deployment = runner.deployment_clone_from_raw(
            source_checkpoint, moved_model, device
        )
        moved_behaviors = {
            "step1_treatment": _evaluate_rows(
                moved_model, moved_deployment, step1_treatment, config, device
            ),
            "step1_retention": _evaluate_rows(
                moved_model, moved_deployment, step1_retention, config, device
            ),
            "all301_treatment": _evaluate_rows(
                moved_model, moved_deployment, treatment_all, config, device
            ),
            "all467_retention": _evaluate_rows(
                moved_model, moved_deployment, retention_all, config, device
            ),
        }
        all301 = compare_behavior(
            source_behaviors["all301_treatment"],
            moved_behaviors["all301_treatment"],
        )
        all467 = compare_behavior(
            source_behaviors["all467_retention"],
            moved_behaviors["all467_retention"],
        )
        integrity = runner.integrity_gate(source_state, moved_model)
        geometry = delta_geometry(
            actual_delta,
            treatment_gradient,
            retention_gradient,
            direction,
        )
        geometry["requested_radius_l2"] = radius
        geometry["requested_minus_actual_l2"] = (
            radius - float(geometry["actual_delta_l2"])
        )
        radius_reports.append(
            {
                "radius_actor6_l2": radius,
                "independent_source_start_model_state_sha256": before_hash,
                "transient_model_state_sha256": (
                    profile_tool.bitwise_model_state_sha256(
                        moved_model.state_dict()
                    )
                ),
                "actual_delta": geometry,
                "step1_treatment": _loss_only_comparison(
                    source_behaviors["step1_treatment"],
                    moved_behaviors["step1_treatment"],
                ),
                "step1_retention": _loss_only_comparison(
                    source_behaviors["step1_retention"],
                    moved_behaviors["step1_retention"],
                ),
                "all301_treatment": all301,
                "all467_retention": {
                    "rows": all467["rows"],
                    "loss": all467["loss"],
                    "source_correct": all467["correct"]["source"],
                    "moved_correct": all467["correct"]["moved"],
                    "flips_count": all467["harm"][
                        "source_correct_to_wrong_count"
                    ],
                    "flips_line_sha256": all467["harm"][
                        "source_correct_to_wrong_line_sha256"
                    ],
                    "action_changed_count": all467["action_changed_count"],
                    "action_changed_line_sha256": all467[
                        "action_changed_line_sha256"
                    ],
                    "policy_quantization": all467["policy_quantization"],
                    "expanded_count_classes": all467["expanded_count_classes"],
                    "raw_expanded_policy_bitwise": all467[
                        "raw_expanded_policy_bitwise"
                    ],
                },
                "integrity": {
                    "source_tensor_count": integrity["source_tensor_count"],
                    "transient_tensor_count": integrity["candidate_tensor_count"],
                    "transient_count_classes": integrity[
                        "candidate_count_classes"
                    ],
                    "changed_parameter_names": integrity[
                        "changed_parameter_names"
                    ],
                    "frozen_nonactor_tensors_bitwise_source": integrity[
                        "frozen_nonactor_tensors_bitwise_source"
                    ],
                    "all_transient_tensors_finite": integrity[
                        "all_candidate_tensors_finite"
                    ],
                },
            }
        )
        del moved_deployment, moved_model

    if file_sha256(profile_tool.SOURCE) != SOURCE_SHA256:
        raise ProbeError("source changed during probe")
    if file_sha256(PROFILE) != PROFILE_SHA256:
        raise ProbeError("profile changed during probe")
    if file_sha256(runner.RUNNER) != RUNNER_SHA256:
        raise ProbeError("runner changed during probe")
    audit: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed_train_only_ram_probe",
        "bindings": {
            **bindings,
            "probe": {"path": str(SCRIPT), "sha256": file_sha256(SCRIPT)},
        },
        "scope": {
            "rows": {
                "step1_treatment": 101,
                "step1_retention": 155,
                "all_treatment": 301,
                "all_retention": 467,
            },
            "selected_train_rows_only": True,
            "every_radius_independent_from_source": True,
            "radius_selection_performed": False,
            "model_artifact_written": False,
            "submission_performed": False,
        },
        "runtime": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(device),
            "autocast_dtype": "torch.bfloat16",
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "eval_batch_size": EVAL_BATCH_SIZE,
        },
        "fixed_radii_actor6_l2": list(RADII),
        "step1_gradients": {
            "treatment_loss": float(treatment_loss),
            "retention_loss": float(retention_loss),
            "treatment_l2": vector_norm(treatment_gradient),
            "retention_l2": vector_norm(retention_gradient),
            "treatment_sha256": vector_sha256(treatment_gradient),
            "retention_sha256": vector_sha256(retention_gradient),
            "pcgrad_direction_sha256": vector_sha256(direction),
            "pcgrad": pcgrad,
            "actor6_parameter_names": list(ACTOR6),
            "actor6_parameter_count": sum(
                value.numel() for value in parameters
            ),
            "grad_buffers_remain_none": all(
                parameter.grad is None for parameter in parameters
            ),
        },
        "radii": radius_reports,
        "output": {
            "json_audit_only": True,
            "model_checkpoint_written": False,
        },
    }
    payload = runner.canonical_json_bytes(audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
    return {
        "status": audit["status"],
        "output": str(output.resolve()),
        "sha256": file_sha256(output),
        "radii": len(radius_reports),
        "model_artifact_written": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    result = run_probe(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
