#!/usr/bin/env python3
"""Train-only second SCP round for the frozen PPO actor6 dual-anchor probe.

The first target-only endpoint is reconstructed in RAM from the frozen v1
mathprobe.  This program then appends affine cuts, all expressed relative to
the original PPO actor coordinates, for:

* every still-unrepaired mandatory recovery target (expert action);
* every harmed PPO-correct row (its PPO/expert action); and
* every non-target PPO-wrong row that changed to another wrong action (its
  original PPO action).

Mandatory targets are deliberately excluded from the third class: their PPO
action is wrong and preserving it would contradict the mandatory expert cut.
The combined minimum-L2 QP is applied to a freshly loaded PPO model in RAM and
both complete train archives are scanned.  Nothing is written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
V1_PATH = TOOLS / "mathprobe_yanz_ppo_actor6_target_qp_v1.py"
V1_SHA256 = "1a8bbf32127093214c1cbc4c039ab82a87b1324f5d0590fb5545d300a44266b8"
TAU = 1.0e-4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if file_sha256(V1_PATH) != V1_SHA256:
    raise RuntimeError("frozen first-round mathprobe SHA drift")
spec = importlib.util.spec_from_file_location("target_qp_mathprobe_v1", V1_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen first-round mathprobe")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)


class Round2Error(RuntimeError):
    pass


def actor_vector(model: torch.nn.Module) -> torch.Tensor:
    named = dict(model.named_parameters())
    return torch.cat([named[name].detach().double().reshape(-1) for name in v1.ACTOR6])


def apply_absolute_delta(
    model: torch.nn.Module,
    base_actor: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    named = dict(model.named_parameters())
    offset = 0
    with torch.no_grad():
        for name in v1.ACTOR6:
            parameter = named[name]
            count = parameter.numel()
            parameter.add_(delta[offset : offset + count].reshape(parameter.shape).float())
            offset += count
    if offset != delta.numel():
        raise Round2Error("actor6 coordinate count drift")
    model.eval().requires_grad_(False)
    after = model.state_dict()
    changed = sorted(name for name in after if not torch.equal(before[name], after[name]))
    if any(name not in v1.ACTOR6 for name in changed):
        raise Round2Error("nonactor tensor changed")
    actual = actor_vector(model) - base_actor
    return actual, {
        "requested_l2": float(torch.linalg.vector_norm(delta)),
        "actual_fp32_l2": float(torch.linalg.vector_norm(actual)),
        "changed_actor_subset": changed,
        "all_nonactor_bitwise": all(
            torch.equal(before[name], after[name])
            for name in after
            if name not in v1.ACTOR6
        ),
        "ram_state_sha256": v1.frozen.bitwise_model_state_sha256(after),
    }


def scan_and_collect(
    source: str,
    path: Path,
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    target_ids: set[str],
    *,
    collect_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = Counter()
    event_ids: dict[str, list[tuple[str, int]]] = {
        "mandatory_repaired": [],
        "mandatory_missed": [],
        "correct_to_wrong": [],
        "wrong_to_different_wrong": [],
        "non_target_wrong_to_different_wrong": [],
        "count_changes": [],
        "repairs": [],
    }
    requests: list[dict[str, Any]] = []
    for row, line_hash in v1.iter_train(path):
        record = records.get(line_hash)
        if record is None:
            raise Round2Error(f"profile row absent: {source}:{line_hash}")
        batch = v1.inference_batch(row, runtime, config)
        context = int(batch["contexts"][0])
        with torch.inference_mode():
            outputs = model(batch)
        decoded = v1.frozen.decode_hybrid(outputs, batch)
        candidate = [int(value) for value in decoded["hybrid_action"]]
        ppo_action = [int(value) for value in record["ppo_hybrid_action"]]
        expert = [int(value) for value in row.get("action", [])]
        ppo_correct = bool(record["ppo_hybrid_correct"])
        candidate_correct = candidate == expert
        is_target = line_hash in target_ids
        counts["rows"] += 1
        counts["ppo_correct"] += int(ppo_correct)
        counts["candidate_correct"] += int(candidate_correct)
        if is_target:
            name = "mandatory_repaired" if candidate_correct else "mandatory_missed"
            event_ids[name].append((line_hash, context))
            if not candidate_correct and collect_rows:
                requests.append(
                    {
                        "source": source,
                        "line_sha256": line_hash,
                        "row": row,
                        "desired_action": expert,
                        "purpose": "mandatory_expert_recovery",
                    }
                )
        if not ppo_correct and candidate_correct:
            event_ids["repairs"].append((line_hash, context))
        if ppo_correct and not candidate_correct:
            event_ids["correct_to_wrong"].append((line_hash, context))
            if collect_rows:
                requests.append(
                    {
                        "source": source,
                        "line_sha256": line_hash,
                        "row": row,
                        "desired_action": ppo_action,
                        "purpose": "ppo_correct_action_guard",
                    }
                )
        if not ppo_correct and not candidate_correct and candidate != ppo_action:
            event_ids["wrong_to_different_wrong"].append((line_hash, context))
            if not is_target:
                event_ids["non_target_wrong_to_different_wrong"].append(
                    (line_hash, context)
                )
                if collect_rows:
                    requests.append(
                        {
                            "source": source,
                            "line_sha256": line_hash,
                            "row": row,
                            "desired_action": ppo_action,
                            "purpose": "non_target_ppo_wrong_action_guard",
                        }
                    )
        if int(decoded["count"]) != int(record["ppo_predicted_count"]):
            event_ids["count_changes"].append((line_hash, context))
            if collect_rows:
                requests.append(
                    {
                        "source": source,
                        "line_sha256": line_hash,
                        "row": row,
                        "desired_action": ppo_action,
                        "purpose": "ppo_count_guard",
                        "count_only": True,
                        "desired_count": int(record["ppo_predicted_count"]),
                    }
                )
        if counts["rows"] % 5000 == 0:
            print(f"scan {source}: {counts['rows']}", file=sys.stderr, flush=True)
    report: dict[str, Any] = dict(counts)
    report.update({name: v1.event_summary(values) for name, values in event_ids.items()})
    return report, requests


def count_cells(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    desired_count: int,
) -> list[tuple[torch.Tensor, tuple[int, int]]]:
    minimum = int(batch["min_counts"][0])
    maximum = int(batch["max_counts"][0])
    if not minimum <= desired_count <= maximum:
        raise Round2Error("desired source count is outside legal bounds")
    logits = outputs["count_logits"][0]
    return [
        (logits[desired_count] - logits[other], (desired_count, other))
        for other in range(minimum, maximum + 1)
        if other != desired_count
    ]


def append_endpoint_cuts(
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    requests: Sequence[Mapping[str, Any]],
    current_absolute_delta: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters = [named[name] for name in v1.ACTOR6]
    for parameter in parameters:
        parameter.requires_grad_(True)
    gradients: list[torch.Tensor] = []
    rhs: list[float] = []
    metadata: list[dict[str, Any]] = []
    request_counts = Counter()
    selected_counts = Counter()
    for request in requests:
        row = request["row"]
        batch = v1.inference_batch(row, runtime, config)
        outputs = model(batch)
        if request.get("count_only"):
            cells = count_cells(
                outputs,
                batch,
                int(request["desired_count"]),
            )
            head = "count"
        else:
            option_count = int(batch["option_mask"][0].sum())
            logits = outputs["policy_logits"][0, :option_count]
            cells = v1.desired_cells(
                logits,
                [int(value) for value in request["desired_action"]],
                int(batch["contexts"][0]),
            )
            head = "policy"
        purpose = str(request["purpose"])
        request_counts[purpose] += 1
        selected = [
            (value, pair)
            for value, pair in cells
            if float(value.detach()) < TAU
        ]
        if not selected:
            # A count-only request may have been fixed by a simultaneous action
            # request before gradients are collected.  Otherwise an exact
            # action violation without a sub-tau cell is a protocol drift.
            if request.get("count_only"):
                continue
            raise Round2Error(
                f"violating action has no sub-tau cell: {request['source']}:"
                f"{request['line_sha256']}:{purpose}"
            )
        selected_counts[purpose] += len(selected)
        for position, (value, pair) in enumerate(selected):
            parts = torch.autograd.grad(
                value,
                parameters,
                retain_graph=position + 1 < len(selected),
            )
            gradient = torch.cat([part.detach().reshape(-1) for part in parts]).float()
            margin = float(value.detach())
            absolute_rhs = (
                TAU
                - margin
                + float(torch.dot(gradient.double(), current_absolute_delta))
            )
            gradients.append(gradient)
            rhs.append(absolute_rhs)
            metadata.append(
                {
                    "source": str(request["source"]),
                    "line_sha256": str(request["line_sha256"]),
                    "purpose": purpose,
                    "head": head,
                    "positive": int(pair[0]),
                    "negative": int(pair[1]),
                    "endpoint_margin": margin,
                    "absolute_rhs": absolute_rhs,
                }
            )
    if not gradients:
        raise Round2Error("second round produced no affine cuts")
    matrix = torch.stack(gradients)
    return matrix, np.asarray(rhs, dtype=np.float64), metadata, {
        "request_rows_by_purpose": dict(sorted(request_counts.items())),
        "appended_cells_by_purpose": dict(sorted(selected_counts.items())),
        "appended_cell_count": len(gradients),
        "gradient_shape": list(matrix.shape),
        "gradient_storage_bytes": matrix.numel() * matrix.element_size(),
    }


def main() -> None:
    started = time.time()
    _, records, targets = v1.bind_profile()
    runtime = v1.frozen.load_bound_runtime()
    bc_model, first_model, config, _ = v1.dual.load_bound_models(runtime)
    del bc_model
    raw_targets = v1.rehydrate_targets(targets)
    first_gradients, first_base, _, first_cells = v1.target_gradients(
        first_model, runtime, config, raw_targets, records, targets
    )
    first_delta, first_solver = v1.solve_min_l2(first_gradients, first_base)
    first_base_actor = actor_vector(first_model)
    first_actual_delta, first_mutation = apply_absolute_delta(
        first_model, first_base_actor, first_delta
    )
    target_sets = {source: set(targets[source]) for source in ("yanz", "old")}
    first_scans: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    for source, path in (("yanz", v1.dual.YANZ), ("old", v1.dual.OLD)):
        scan, source_requests = scan_and_collect(
            source,
            path,
            first_model,
            runtime,
            config,
            records[source],
            target_sets[source],
            collect_rows=True,
        )
        first_scans[source] = scan
        requests.extend(source_requests)
    if sum(first_scans[source]["mandatory_missed"]["count"] for source in first_scans) != 192:
        raise Round2Error("first endpoint mandatory-miss count drift")
    if sum(first_scans[source]["correct_to_wrong"]["count"] for source in first_scans) != 201:
        raise Round2Error("first endpoint harm count drift")
    if sum(
        first_scans[source]["wrong_to_different_wrong"]["count"] for source in first_scans
    ) != 116:
        raise Round2Error("first endpoint wrong-different count drift")
    appended_gradients, appended_rhs, appended_meta, appended_report = (
        append_endpoint_cuts(
            first_model,
            runtime,
            config,
            requests,
            first_actual_delta,
        )
    )
    initial_rhs = TAU - first_base
    combined_gradients = torch.cat([first_gradients, appended_gradients], dim=0)
    combined_rhs = np.concatenate([initial_rhs, appended_rhs])
    # The frozen solver accepts margins and internally forms TAU - margin.
    synthetic_base = TAU - combined_rhs
    second_delta, second_solver = v1.solve_min_l2(combined_gradients, synthetic_base)
    del combined_gradients, appended_gradients

    bc_model, second_model, second_config, _ = v1.dual.load_bound_models(runtime)
    del bc_model
    if dict(second_config) != dict(config):
        raise Round2Error("fresh PPO config drift")
    second_base_actor = actor_vector(second_model)
    _, second_mutation = apply_absolute_delta(second_model, second_base_actor, second_delta)
    second_scans: dict[str, Any] = {}
    for source, path in (("yanz", v1.dual.YANZ), ("old", v1.dual.OLD)):
        scan, _ = scan_and_collect(
            source,
            path,
            second_model,
            runtime,
            config,
            records[source],
            target_sets[source],
            collect_rows=False,
        )
        second_scans[source] = scan
    output = {
        "schema_version": "ptcg-yanz-ppo-actor6-scp-round2-mathprobe-v1",
        "status": "completed_train_only_ram_scp_round2_probe",
        "bindings": {
            "tool_sha256": file_sha256(Path(__file__).resolve()),
            "frozen_round1_tool_sha256": V1_SHA256,
            "dual_anchor_profile_sha256": v1.PROFILE_SHA256,
            "ppo_parent_state_sha256": v1.dual.PPO_PARENT_BITWISE_STATE_SHA256,
        },
        "scope": {
            "train_only": True,
            "checkpoint_written": False,
            "artifact_written": False,
            "holdout_opened": False,
            "submission_performed": False,
            "ram_endpoints_only": True,
        },
        "round1": {
            "target_cells": first_cells,
            "solver": first_solver,
            "mutation": first_mutation,
            "scan": first_scans,
        },
        "round2": {
            "affine_cut_append": appended_report,
            "affine_cut_metadata_digest": hashlib.sha256(
                (json.dumps(appended_meta, sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest(),
            "total_constraint_cells": int(len(combined_rhs)),
            "solver": second_solver,
            "mutation": second_mutation,
            "scan": second_scans,
        },
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
