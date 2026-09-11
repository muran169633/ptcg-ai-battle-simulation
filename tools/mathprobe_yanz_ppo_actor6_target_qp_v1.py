#!/usr/bin/env python3
"""Disposable, train-only PPO-origin actor6 target-QP numerical probe.

This diagnostic deliberately omits protection constraints.  It builds only the
currently violated expert-action cells for all 458 BC-correct/PPO-wrong rows,
solves their minimum-L2 first-order projection, applies the delta to a CPU FP32
model in RAM, and scans both complete train archives under the frozen hybrid
decoder.  It writes no output or model artifact; the JSON report goes to stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import orjson
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_yanz_ppo_dual_anchor_deployment_v1 as dual  # noqa: E402
import profile_yanz_source_error_actor6_deployment_v2 as frozen  # noqa: E402


PROFILE = ROOT / "data/yanz_alakazam_ppo_dual_anchor_20260810_v1/train_only_profile.json"
PROFILE_SHA256 = "e10e2e6d4ee1b1225b82c1e43af089969bbdad61dfd70e2c42d151e85015b7b0"
TAU = 1.0e-4
ACTOR6 = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EXPECTED_TARGET_ROWS = {"yanz": 15, "old": 443}
EXPECTED_VIOLATED_CELLS = 474
EXPECTED_ALL_TARGET_CELLS = 4144


class ProbeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeError(f"JSON root drift: {path}")
    return value


def iter_train(path: Path):
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members:
            raise ProbeError(f"train members absent: {path}")
        for member in members:
            with archive.open(member) as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = orjson.loads(line)
                    if not isinstance(row, dict) or row.get("split") != "train":
                        raise ProbeError(f"non-train row: {member}:{line_number}")
                    yield row, hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()


def bind_profile() -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]], dict[str, list[str]]]:
    if sha256_file(PROFILE) != PROFILE_SHA256:
        raise ProbeError("dual-anchor profile SHA drift")
    profile = load_json(PROFILE)
    if profile.get("schema_version") != dual.SCHEMA:
        raise ProbeError("dual-anchor profile schema drift")
    if profile.get("status") != "frozen_complete_train_only_dual_anchor_deployment_profile":
        raise ProbeError("dual-anchor profile incomplete")
    records: dict[str, dict[str, dict[str, Any]]] = {"yanz": {}, "old": {}}
    for descriptor in profile["cache"]["immutable_shards"]:
        path = Path(str(descriptor["path"]))
        if sha256_file(path) != descriptor["sha256"]:
            raise ProbeError(f"cache shard SHA drift: {path}")
        payload = load_json(path)
        source = str(payload.get("source"))
        if source not in records:
            raise ProbeError("cache source drift")
        if payload.get("status") != "complete_immutable_train_only_dual_anchor_chunk":
            raise ProbeError("cache status drift")
        for record in payload["records"]:
            line_hash = str(record["line_sha256"])
            if line_hash in records[source]:
                raise ProbeError(f"duplicate record: {source}:{line_hash}")
            records[source][line_hash] = record
    if {source: len(values) for source, values in records.items()} != {
        "yanz": dual.EXPECTED_YANZ_ROWS,
        "old": dual.EXPECTED_OLD_ROWS,
    }:
        raise ProbeError("cache row cardinality drift")
    index = profile["hybrid_category_index"]
    targets = {
        source: [
            str(value)
            for value in index[source]["bc_correct_ppo_wrong_recovery_target"]
        ]
        for source in ("yanz", "old")
    }
    if {source: len(values) for source, values in targets.items()} != EXPECTED_TARGET_ROWS:
        raise ProbeError("target row cardinality drift")
    for source, ids in targets.items():
        if len(ids) != len(set(ids)):
            raise ProbeError("duplicate target ID")
        for line_hash in ids:
            record = records[source][line_hash]
            if record["hybrid_category"] != "bc_correct_ppo_wrong_recovery_target":
                raise ProbeError("target category drift")
            if int(record["ppo_predicted_count"]) != len(record["expert_action_order"]):
                raise ProbeError("actor6-infeasible target count")
    return profile, records, targets


def inference_batch(
    row: Mapping[str, Any], runtime: Any, config: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    observation = row.get("observation")
    if not isinstance(observation, dict):
        raise ProbeError("observation missing")
    current = observation.get("current") or {}
    feature = runtime.featurize_row(
        {
            "observation": observation,
            "seat": int(current.get("yourIndex", 0) or 0),
            "deck_hash": dual.DECK_HASH,
            "team_name": "",
            "action": [],
            "terminal_reward": 0.0,
            "sample_weight": 1.0,
        },
        hash_size=int(config["hash_size"]),
        max_state_entities=int(config["max_state_entities"]),
    )
    if feature is None:
        raise ProbeError("featurization failed")
    batch = runtime.collate_decisions(
        [feature],
        max_state_entities=int(config["max_state_entities"]),
        entity_fields=int(config["entity_fields"]),
        option_fields=int(config["option_fields"]),
    )
    if any(value.shape[0] != 1 or value.device.type != "cpu" for value in batch.values()):
        raise ProbeError("runtime is not CPU batch=1")
    return batch


def desired_cells(
    logits: torch.Tensor, expert: Sequence[int], context: int
) -> list[tuple[torch.Tensor, tuple[int, int]]]:
    cells: list[tuple[torch.Tensor, tuple[int, int]]] = []
    option_count = int(logits.numel())
    if context == frozen.SKILL_ORDER_CONTEXT:
        remaining = list(range(option_count))
        for target in expert:
            for other in remaining:
                if other != target:
                    cells.append((logits[target] - logits[other], (int(target), other)))
            remaining.remove(int(target))
    else:
        selected = set(int(value) for value in expert)
        for target in sorted(selected):
            for other in range(option_count):
                if other not in selected:
                    cells.append((logits[target] - logits[other], (target, other)))
    return cells


def rehydrate_targets(targets: Mapping[str, Sequence[str]]) -> dict[str, dict[str, Any]]:
    paths = {"yanz": dual.YANZ, "old": dual.OLD}
    output: dict[str, dict[str, Any]] = {}
    for source in ("yanz", "old"):
        wanted = set(targets[source])
        for row, line_hash in iter_train(paths[source]):
            if line_hash in wanted:
                output[line_hash] = row
    expected = set(targets["yanz"]) | set(targets["old"])
    if set(output) != expected:
        raise ProbeError("target rehydration incomplete")
    return output


def target_gradients(
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    raw_rows: Mapping[str, Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    targets: Mapping[str, Sequence[str]],
) -> tuple[torch.Tensor, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters = [named[name] for name in ACTOR6]
    for parameter in parameters:
        parameter.requires_grad_(True)
    gradients: list[torch.Tensor] = []
    base: list[float] = []
    metadata: list[dict[str, Any]] = []
    all_cell_count = 0
    violating_by_row: Counter[int] = Counter()
    for source in ("yanz", "old"):
        for line_hash in targets[source]:
            row = raw_rows[line_hash]
            batch = inference_batch(row, runtime, config)
            outputs = model(batch)
            decoded = frozen.decode_hybrid(outputs, batch)
            record = records[source][line_hash]
            if decoded["hybrid_action"] != record["ppo_hybrid_action"]:
                raise ProbeError(f"PPO origin action mismatch: {source}:{line_hash}")
            option_count = int(batch["option_mask"][0].sum())
            logits = outputs["policy_logits"][0, :option_count]
            expert = [int(value) for value in row.get("action", [])]
            context = int(batch["contexts"][0])
            cells = desired_cells(logits, expert, context)
            all_cell_count += len(cells)
            selected = [
                (index, value, pair)
                for index, (value, pair) in enumerate(cells)
                if float(value.detach()) < TAU
            ]
            violating_by_row[len(selected)] += 1
            if not selected:
                raise ProbeError(f"wrong target has no violated cell: {source}:{line_hash}")
            for position, (_, value, pair) in enumerate(selected):
                parts = torch.autograd.grad(
                    value,
                    parameters,
                    retain_graph=position + 1 < len(selected),
                )
                gradient = torch.cat([part.detach().reshape(-1) for part in parts]).float()
                gradients.append(gradient)
                base.append(float(value.detach()))
                metadata.append(
                    {
                        "source": source,
                        "line_sha256": line_hash,
                        "context": context,
                        "target": pair[0],
                        "other": pair[1],
                    }
                )
    matrix = torch.stack(gradients)
    if matrix.shape != (EXPECTED_VIOLATED_CELLS, 65793):
        raise ProbeError(f"gradient matrix drift: {tuple(matrix.shape)}")
    if all_cell_count != EXPECTED_ALL_TARGET_CELLS:
        raise ProbeError(f"all target cell count drift: {all_cell_count}")
    if any(parameter.grad is not None for parameter in parameters):
        raise ProbeError("unexpected materialized .grad buffer")
    return matrix, np.asarray(base, dtype=np.float64), metadata, {
        "all_target_action_cells": all_cell_count,
        "currently_below_tau_cells": len(gradients),
        "violating_cells_per_row": {
            str(key): int(value) for key, value in sorted(violating_by_row.items())
        },
        "gradient_shape": list(matrix.shape),
        "gradient_storage_bytes": matrix.numel() * matrix.element_size(),
        "gradient_row_norm_min": float(torch.linalg.vector_norm(matrix, dim=1).min()),
        "gradient_row_norm_max": float(torch.linalg.vector_norm(matrix, dim=1).max()),
    }


def solve_min_l2(gradients: torch.Tensor, base: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
    # At 474 rows the exact FP64 Gram is small and avoids the FP32 selection
    # ambiguity of the older pure-BC audit solver.
    gradients64 = gradients.double()
    gram = (gradients64 @ gradients64.T).numpy()
    deficit = TAU - base
    eigenvalues = np.linalg.eigvalsh(gram)
    maximum = float(max(eigenvalues[-1], 0.0))
    default_tol = max(gram.shape) * np.finfo(np.float64).eps * maximum
    rank_default = int(np.sum(eigenvalues > default_tol))
    rank_rcond_1e10 = int(np.sum(eigenvalues > maximum * 1.0e-10))

    active: list[int] = []
    multipliers = np.empty(0, dtype=np.float64)
    iterations = 0
    for iterations in range(1, 2001):
        while active:
            active_gram = gram[np.ix_(active, active)]
            multipliers = np.linalg.lstsq(active_gram, deficit[active], rcond=1.0e-12)[0]
            if float(multipliers.min()) >= -1.0e-10:
                break
            active.pop(int(np.argmin(multipliers)))
        effect = gram[:, active] @ multipliers if active else np.zeros(len(deficit))
        violation = deficit - effect
        worst = int(np.argmax(violation))
        if float(violation[worst]) <= 2.0e-9:
            break
        if worst in active:
            raise ProbeError("active-set solver stalled")
        active.append(worst)
    else:
        raise ProbeError("active-set solver iteration cap")

    active_gram = gram[np.ix_(active, active)]
    multipliers = np.linalg.lstsq(active_gram, deficit[active], rcond=1.0e-13)[0]
    delta = gradients64[active].T @ torch.from_numpy(multipliers)
    effect = (gradients64 @ delta).numpy()
    slack = effect - deficit
    if float(slack.min()) < -1.0e-8:
        raise ProbeError(f"linear QP slack failure: {slack.min()}")
    active_residual = effect[active] - deficit[active]
    stationarity = delta - gradients64[active].T @ torch.from_numpy(multipliers)
    report = {
        "solver": "deterministic minimum-L2 dual active-set; exact FP64 dense Gram",
        "iterations": iterations,
        "gram_shape": list(gram.shape),
        "gram_storage_bytes": int(gram.nbytes),
        "gram_eigen_min": float(eigenvalues[0]),
        "gram_eigen_max": float(eigenvalues[-1]),
        "rank_default_tolerance": rank_default,
        "rank_relative_1e-10": rank_rcond_1e10,
        "active_count": len(active),
        "active_indices": active,
        "multiplier_min": float(multipliers.min()),
        "linear_l2": float(torch.linalg.vector_norm(delta)),
        "linear_min_endpoint_margin": float((base + effect).min()),
        "linear_min_slack": float(slack.min()),
        "linear_max_slack": float(slack.max()),
        "active_residual_abs_max": float(np.abs(active_residual).max()),
        "stationarity_l2": float(torch.linalg.vector_norm(stationarity)),
    }
    return delta, report


def apply_ram_delta(model: torch.nn.Module, delta: torch.Tensor) -> dict[str, Any]:
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    named = dict(model.named_parameters())
    offset = 0
    actual_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for name in ACTOR6:
            parameter = named[name]
            count = parameter.numel()
            source = parameter.detach().clone()
            parameter.add_(delta[offset : offset + count].reshape(parameter.shape).float())
            actual_parts.append((parameter.detach() - source).double().reshape(-1))
            offset += count
    if offset != delta.numel():
        raise ProbeError("delta coordinate count drift")
    model.eval().requires_grad_(False)
    after = model.state_dict()
    changed = sorted(name for name in after if not torch.equal(before[name], after[name]))
    if any(name not in ACTOR6 for name in changed):
        raise ProbeError("nonactor tensor changed")
    return {
        "requested_fp64_l2": float(torch.linalg.vector_norm(delta)),
        "actual_fp32_l2": float(torch.linalg.vector_norm(torch.cat(actual_parts))),
        "changed_tensors": changed,
        "all_nonactor_tensors_bitwise_unchanged": all(
            torch.equal(before[name], after[name]) for name in after if name not in ACTOR6
        ),
        "count_head_bitwise_unchanged": all(
            torch.equal(before[name], after[name]) for name in after if name.startswith("count_head.")
        ),
        "ram_endpoint_state_sha256": frozen.bitwise_model_state_sha256(after),
    }


def event_summary(values: Sequence[tuple[str, int]]) -> dict[str, Any]:
    ordered = sorted(str(line_hash) for line_hash, _ in values)
    return {
        "count": len(values),
        "by_context": dict(
            sorted(Counter(str(context) for _, context in values).items(), key=lambda item: int(item[0]))
        ),
        "line_sha256_digest": hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest(),
        "first_20_line_sha256": ordered[:20],
    }


def scan_source(
    source: str,
    path: Path,
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    target_ids: set[str],
) -> dict[str, Any]:
    counts = Counter()
    events: dict[str, list[tuple[str, int]]] = {
        "repairs": [],
        "correct_to_wrong": [],
        "wrong_to_different_wrong": [],
        "hybrid_action_changes": [],
        "count_changes": [],
        "mandatory_repairs": [],
        "mandatory_not_repaired": [],
    }
    for row, line_hash in iter_train(path):
        record = records.get(line_hash)
        if record is None:
            raise ProbeError(f"profile row absent: {source}:{line_hash}")
        batch = inference_batch(row, runtime, config)
        context = int(batch["contexts"][0])
        if context != int(record["context"]):
            raise ProbeError("context drift")
        with torch.inference_mode():
            outputs = model(batch)
        decoded = frozen.decode_hybrid(outputs, batch)
        candidate = [int(value) for value in decoded["hybrid_action"]]
        ppo_action = [int(value) for value in record["ppo_hybrid_action"]]
        expert = [int(value) for value in row.get("action", [])]
        ppo_correct = bool(record["ppo_hybrid_correct"])
        candidate_correct = candidate == expert
        counts["rows"] += 1
        counts["ppo_correct"] += int(ppo_correct)
        counts["candidate_correct"] += int(candidate_correct)
        if not ppo_correct and candidate_correct:
            events["repairs"].append((line_hash, context))
        if ppo_correct and not candidate_correct:
            events["correct_to_wrong"].append((line_hash, context))
        if not ppo_correct and not candidate_correct and candidate != ppo_action:
            events["wrong_to_different_wrong"].append((line_hash, context))
        if candidate != ppo_action:
            events["hybrid_action_changes"].append((line_hash, context))
        if int(decoded["count"]) != int(record["ppo_predicted_count"]):
            events["count_changes"].append((line_hash, context))
        if line_hash in target_ids:
            key = "mandatory_repairs" if candidate_correct else "mandatory_not_repaired"
            events[key].append((line_hash, context))
        if counts["rows"] % 5000 == 0:
            print(f"scan {source}: {counts['rows']}", file=sys.stderr, flush=True)
    result: dict[str, Any] = dict(counts)
    result.update({name: event_summary(values) for name, values in events.items()})
    return result


def main() -> None:
    started = time.time()
    profile, records, targets = bind_profile()
    static = dual.validate_static_bindings()
    runtime = frozen.load_bound_runtime()
    bc_model, ppo_model, config, model_audit = dual.load_bound_models(runtime)
    del bc_model
    raw_rows = rehydrate_targets(targets)
    gradients, base, metadata, cell_report = target_gradients(
        ppo_model, runtime, config, raw_rows, records, targets
    )
    delta, solver = solve_min_l2(gradients, base)
    mutation = apply_ram_delta(ppo_model, delta)
    del gradients, delta
    scan = {
        "yanz": scan_source(
            "yanz", dual.YANZ, ppo_model, runtime, config, records["yanz"], set(targets["yanz"])
        ),
        "old": scan_source(
            "old", dual.OLD, ppo_model, runtime, config, records["old"], set(targets["old"])
        ),
    }
    report = {
        "schema_version": "ptcg-yanz-ppo-actor6-target-only-mathprobe-v1",
        "status": "completed_train_only_ram_target_only_linear_qp_probe",
        "bindings": {
            "mathprobe_tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "dual_anchor_profile": {"path": str(PROFILE.resolve()), "sha256": PROFILE_SHA256},
            "dual_anchor_profiler_sha256": sha256_file(dual.SCRIPT),
            "ppo_parent_checkpoint_sha256": dual.PPO_PARENT_SHA256,
            "ppo_parent_state_sha256": dual.PPO_PARENT_BITWISE_STATE_SHA256,
            "pure_bc_expanded61_state_sha256": dual.PURE_BC_EXPANDED61_BITWISE_STATE_SHA256,
            "hybrid_main_sha256": dual.HYBRID_MAIN_SHA256,
            "policy_runtime_sha256": dual.RUNTIME_SHA256,
            "profile_category_index_sha256": profile["hybrid_category_index"]["sha256"],
            "profile_cache_shards": len(profile["cache"]["immutable_shards"]),
        },
        "scope": {
            "opened_archive_members": "train/*.jsonl only",
            "holdout_opened": False,
            "checkpoint_written": False,
            "model_artifact_written": False,
            "audit_artifact_written": False,
            "submission_performed": False,
            "candidate_exists_in_ram_only": True,
            "protection_constraints_included": False,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "torch.float32 endpoint; torch.float64 QP",
            "batch_size": 1,
            "torch_num_threads": torch.get_num_threads(),
            "autocast": False,
            "torch_version": torch.__version__,
        },
        "target_contract": {
            "rows": {source: len(values) for source, values in targets.items()},
            "total_rows": sum(len(values) for values in targets.values()),
            "tau": TAU,
            "actor6": list(ACTOR6),
            "actor_coordinates": 65793,
            "target_id_digest": hashlib.sha256(
                ("\n".join(sorted(targets["yanz"] + targets["old"])) + "\n").encode()
            ).hexdigest(),
        },
        "gradient_geometry": cell_report,
        "solver": solver,
        "ram_mutation": mutation,
        "full_train_scan_relative_to_ppo": scan,
        "diagnostic_gates": {
            "all_458_mandatory_repaired": sum(
                scan[source]["mandatory_repairs"]["count"] for source in ("yanz", "old")
            ) == 458,
            "ppo_correct_to_wrong_zero": all(
                scan[source]["correct_to_wrong"]["count"] == 0 for source in ("yanz", "old")
            ),
            "ppo_wrong_to_different_wrong_zero": all(
                scan[source]["wrong_to_different_wrong"]["count"] == 0
                for source in ("yanz", "old")
            ),
            "count_changes_zero": all(
                scan[source]["count_changes"]["count"] == 0 for source in ("yanz", "old")
            ),
            "yanz_at_least_1422": scan["yanz"]["candidate_correct"] >= 1422,
            "old_at_least_56241": scan["old"]["candidate_correct"] >= 56241,
        },
        "model_load_audit": model_audit,
        "static_runtime_binding": static["runtime_semantics"],
        "elapsed_seconds": time.time() - started,
        "interpretation": (
            "Target-only first-order diagnostic. Any protection harm makes this endpoint non-promotable; "
            "failure does not prove full actor6 infeasibility."
        ),
        "constraint_metadata_digest": hashlib.sha256(
            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
