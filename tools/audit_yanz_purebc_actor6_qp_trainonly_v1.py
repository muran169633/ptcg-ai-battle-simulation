#!/usr/bin/env python3
"""Reproduce the exact-CPU pure-BC actor6 QP as a train-only audit.

The program is intentionally narrow and fail closed.  It binds the frozen
policy-boundary profile, the hybrid submission runtime, three fixed yanz
targets, all 40 target policy cells, and one source-critical policy guard for
each of the 467 selected old-train retention rows.  It solves a minimum-L2
linearized actor6 edit at tau=1e-4, applies that edit only to an in-memory FP32
model, and scans every train row in the yanz and old archives at batch size 1.

Only the requested JSON audit is written.  No checkpoint, evaluation split,
package, upload, or submission is created or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

import profile_yanz_source_error_actor6_deployment_v2 as profile_tool  # noqa: E402


SCHEMA = "ptcg-yanz-purebc-actor6-qp-trainonly-audit-v1"
SCRIPT = Path(__file__).resolve()
PROFILE = ROOT / (
    "data/yanz_alakazam_actor6_pcgrad_specialbc_20260810_v2_"
    "deployment_policyboundary/train_only_profile.json"
)
PROFILE_SHA256 = "87891f8985d5f23b8114e5aff11426a4c80c717dc2782eae4cb9cd38d71eb529"
PROFILER_SHA256 = "386d23e280f42225b1752e987b1eb92df8b30b41a1be5ebbd9260b8f6388927e"
TAU = 1.0e-4
GUARD_FLOOR = 0.0
TARGET_IDS = (
    "ca657545345192c96417bffa5da89f7479c136d8e7dbabd5338bd5952a2f9c91",
    "7fc63a1145d9ccb5f52e19e37472b11b6128ebea0f6eb9e76c1d4dd482bef70e",
    "3432a6e6c8c934801dd4ddb1a24c7e05c7fe598771ccc85da20a9a717ebf040b",
)
ACTOR6 = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EXPECTED_CHANGED_ACTOR_TENSORS = tuple(
    name for name in ACTOR6 if name != "actor_residual.2.bias"
)
EXPECTED = {
    "yanz": {
        "rows": 1725,
        "source_hybrid_correct": 1421,
        "source_set_correct": 1422,
        "source_count_correct": 1723,
        "candidate_hybrid_correct": 1424,
        "candidate_set_correct": 1425,
        "candidate_count_correct": 1723,
        "repairs": 3,
        "source_correct_to_wrong": 0,
        "set_harm": 0,
        "wrong_to_different_wrong": 0,
        "hybrid_action_changes": 3,
        "raw_policy_action_changes": 5,
        "count_changes": 0,
    },
    "old": {
        "rows": 59875,
        "source_hybrid_correct": 56241,
        "source_set_correct": 56482,
        "source_count_correct": 59864,
        "candidate_hybrid_correct": 56242,
        "candidate_set_correct": 56483,
        "candidate_count_correct": 59864,
        "repairs": 1,
        "source_correct_to_wrong": 0,
        "set_harm": 0,
        "wrong_to_different_wrong": 0,
        "hybrid_action_changes": 1,
        "raw_policy_action_changes": 45,
        "count_changes": 0,
        "selected467_flips": 0,
    },
}


class AuditError(RuntimeError):
    """A frozen binding, geometry, or behavior gate failed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {path}")
    return value


def bind_profile() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if file_sha256(PROFILE) != PROFILE_SHA256:
        raise AuditError("policy-boundary profile SHA drift")
    if file_sha256(profile_tool.SCRIPT) != PROFILER_SHA256:
        raise AuditError("deployment profiler SHA drift")
    profile = load_json(PROFILE)
    if profile.get("schema_version") != profile_tool.SCHEMA:
        raise AuditError("policy-boundary profile schema drift")
    bindings = profile.get("bindings")
    if not isinstance(bindings, dict):
        raise AuditError("profile bindings missing")
    profiler = bindings.get("profiler")
    if not isinstance(profiler, dict) or profiler.get("sha256") != PROFILER_SHA256:
        raise AuditError("profile does not bind the frozen profiler")
    static = profile_tool.validate_static_bindings()
    steps = profile.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        raise AuditError("profile step contract drift")
    retention = [row for step in steps for row in step["retention"]]
    ids = [str(row["line_sha256"]) for row in retention]
    if len(ids) != 467 or len(set(ids)) != 467:
        raise AuditError("profile retention is not 467 unique rows")
    return profile, retention, static


def load_source_records(profile: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {"yanz": {}, "old": {}}
    cache = profile.get("cache")
    if not isinstance(cache, Mapping):
        raise AuditError("profile cache ledger missing")
    shards = cache.get("immutable_shards")
    if not isinstance(shards, list):
        raise AuditError("profile cache shard ledger missing")
    for descriptor in shards:
        if not isinstance(descriptor, Mapping):
            raise AuditError("profile shard descriptor drift")
        path = Path(str(descriptor["path"]))
        if file_sha256(path) != descriptor.get("sha256"):
            raise AuditError(f"profile cache shard SHA drift: {path}")
        payload = load_json(path)
        source = str(payload.get("source"))
        if source not in output or payload.get("status") != "complete_immutable_train_chunk":
            raise AuditError(f"profile cache shard status drift: {path}")
        records = payload.get("records")
        if not isinstance(records, list):
            raise AuditError(f"profile cache records missing: {path}")
        for record in records:
            if not isinstance(record, dict):
                raise AuditError("profile cache row drift")
            line_hash = str(record["line_sha256"])
            if line_hash in output[source]:
                raise AuditError(f"duplicate cached row: {source}:{line_hash}")
            output[source][line_hash] = record
    if len(output["yanz"]) != 1725 or len(output["old"]) != 59875:
        raise AuditError("profile cache full-train cardinality drift")
    return output


def iter_train_lines(path: Path):
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members:
            raise AuditError(f"no train JSONL members: {path}")
        for member in members:
            with archive.open(member) as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = orjson.loads(line)
                    if not isinstance(row, dict) or row.get("split") != "train":
                        raise AuditError(f"non-train row: {path}:{member}:{line_number}")
                    identity = {
                        "archive_member": member,
                        "member_line_number": line_number,
                        "raw_line_sha256": hashlib.sha256(line).hexdigest(),
                        "line_sha256": hashlib.sha256(line.rstrip(b"\r\n")).hexdigest(),
                    }
                    yield row, identity


def load_constraint_rows(retention: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    needs = {
        str(profile_tool.YANZ): set(TARGET_IDS),
        str(profile_tool.OLD): {str(row["line_sha256"]) for row in retention},
    }
    output: dict[str, dict[str, Any]] = {}
    for path in (profile_tool.YANZ, profile_tool.OLD):
        wanted = needs[str(path)]
        for row, identity in iter_train_lines(path):
            line_hash = identity["line_sha256"]
            if line_hash in wanted:
                output[line_hash] = row
    expected = set(TARGET_IDS) | needs[str(profile_tool.OLD)]
    if set(output) != expected:
        raise AuditError("constraint rows could not be rehydrated exactly")
    return output


def inference_batch(
    row: Mapping[str, Any], runtime: Any, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    observation = row.get("observation")
    if not isinstance(observation, dict):
        raise AuditError("row observation missing")
    current = observation.get("current") or {}
    feature = runtime.featurize_row(
        {
            "observation": observation,
            "seat": int(current.get("yourIndex", 0) or 0),
            "deck_hash": profile_tool.DECK_HASH,
            "team_name": "",
            "action": [],
            "terminal_reward": 0.0,
            "sample_weight": 1.0,
        },
        hash_size=int(config["hash_size"]),
        max_state_entities=int(config["max_state_entities"]),
    )
    if feature is None:
        raise AuditError("deployment featurization failed")
    batch = runtime.collate_decisions(
        [feature],
        max_state_entities=int(config["max_state_entities"]),
        entity_fields=int(config["entity_fields"]),
        option_fields=int(config["option_fields"]),
    )
    if any(value.shape[0] != 1 or value.device.type != "cpu" for value in batch.values()):
        raise AuditError("deployment batch is not CPU batch=1")
    return feature, batch


def policy_cells(
    model: torch.nn.Module,
    row: Mapping[str, Any],
    runtime: Any,
    config: Mapping[str, Any],
) -> tuple[list[tuple[torch.Tensor, dict[str, int]]], int]:
    _, batch = inference_batch(row, runtime, config)
    outputs = model(batch)
    option_count = int(batch["option_mask"][0].sum())
    logits = outputs["policy_logits"][0, :option_count]
    expert = [int(value) for value in row.get("action", [])]
    context = int(batch["contexts"][0])
    cells: list[tuple[torch.Tensor, dict[str, int]]] = []
    if context == profile_tool.SKILL_ORDER_CONTEXT:
        remaining = list(range(option_count))
        for step, target in enumerate(expert):
            for other in remaining:
                if other != target:
                    cells.append(
                        (
                            logits[target] - logits[other],
                            {"step": step, "target": target, "other": other},
                        )
                    )
            remaining.remove(target)
    else:
        targets = sorted(set(expert))
        target_set = set(targets)
        for target in targets:
            for other in range(option_count):
                if other not in target_set:
                    cells.append(
                        (logits[target] - logits[other], {"target": target, "other": other})
                    )
    if not cells:
        raise AuditError("selected policy row has no finite correctness cell")
    return cells, context


def actor6_gradients(
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    raw_rows: Mapping[str, Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, np.ndarray, list[str], list[dict[str, Any]], int]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters = [named[name] for name in ACTOR6]
    for parameter in parameters:
        parameter.requires_grad_(True)
    gradients: list[torch.Tensor] = []
    base: list[float] = []
    kinds: list[str] = []
    metadata: list[dict[str, Any]] = []
    for line_hash in TARGET_IDS:
        cells, context = policy_cells(model, raw_rows[line_hash], runtime, config)
        for index, (value, cell) in enumerate(cells):
            parts = torch.autograd.grad(
                value,
                parameters,
                retain_graph=index + 1 < len(cells),
            )
            gradients.append(torch.cat([part.detach().reshape(-1) for part in parts]))
            base.append(float(value.detach()))
            kinds.append("target")
            metadata.append({"row": line_hash, "context": context, "cell": cell})
    target_cells = len(gradients)
    if target_cells != 40:
        raise AuditError(f"target policy-cell count drift: {target_cells}")
    for descriptor in retention:
        line_hash = str(descriptor["line_sha256"])
        cells, context = policy_cells(model, raw_rows[line_hash], runtime, config)
        values = [float(value.detach()) for value, _ in cells]
        critical = min(range(len(values)), key=lambda index: (values[index], index))
        value, cell = cells[critical]
        if abs(values[critical] - float(descriptor["policy_hybrid_margin"])) >= 2.0e-6:
            raise AuditError(f"source-critical cell mismatch: {line_hash}")
        parts = torch.autograd.grad(value, parameters)
        gradients.append(torch.cat([part.detach().reshape(-1) for part in parts]))
        base.append(values[critical])
        kinds.append("guard")
        metadata.append({"row": line_hash, "context": context, "cell": cell})
    matrix = torch.stack(gradients).float()
    if tuple(matrix.shape) != (507, 65793):
        raise AuditError(f"constraint-gradient shape drift: {tuple(matrix.shape)}")
    if any(parameter.grad is not None for parameter in parameters):
        raise AuditError("autograd materialized actor6 grad buffers")
    return matrix, np.asarray(base), kinds, metadata, target_cells


def solve_projection(
    gradients: torch.Tensor,
    base: np.ndarray,
    kinds: Sequence[str],
    target_cells: int,
    *,
    enforce_frozen_active_identity: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gram = (gradients @ gradients.T).double().numpy()
    threshold = np.asarray([TAU if kind == "target" else GUARD_FLOOR for kind in kinds])
    deficit = threshold - base
    active: list[int] = []
    multipliers = np.empty(0)
    for iteration in range(400):
        while active:
            active_gram = gram[np.ix_(active, active)]
            multipliers = np.linalg.lstsq(
                active_gram, deficit[active], rcond=1.0e-11
            )[0]
            if float(multipliers.min()) >= -1.0e-10:
                break
            active.pop(int(np.argmin(multipliers)))
        effect = gram[:, active] @ multipliers if active else np.zeros(len(deficit))
        violation = deficit - effect
        worst = int(np.argmax(violation))
        if float(violation[worst]) <= 3.0e-8:
            break
        if worst in active:
            raise AuditError("active-set projection stalled")
        active.append(worst)
    else:
        raise AuditError("active-set projection did not converge")

    active_gram64 = np.asarray(
        [
            [
                float(torch.dot(gradients[left].double(), gradients[right].double()))
                for right in active
            ]
            for left in active
        ]
    )
    multipliers = np.linalg.solve(active_gram64, deficit[active])
    delta = torch.zeros(gradients.shape[1], dtype=torch.float64)
    for multiplier, index in zip(multipliers, active):
        delta.add_(gradients[index].double(), alpha=float(multiplier))
    effect = np.asarray(
        [float(torch.dot(gradient.double(), delta)) for gradient in gradients]
    )
    slack = effect - deficit
    if float(slack.min()) < -2.0e-9:
        raise AuditError("refined projection violates a linear constraint")
    report = {
        "tau": TAU,
        "guard_floor": GUARD_FLOOR,
        "linear_l2": float(torch.linalg.vector_norm(delta)),
        "active_count": len(active),
        "active_target": sum(index < target_cells for index in active),
        "active_guard": sum(index >= target_cells for index in active),
        "linear_min_target_margin": float((base[:target_cells] + effect[:target_cells]).min()),
        "linear_min_guard_margin": float((base[target_cells:] + effect[target_cells:]).min()),
        "linear_min_slack": float(slack.min()),
        "active_indices": active,
    }
    if enforce_frozen_active_identity and (
        report["active_count"] != 4
        or report["active_target"] != 4
        or report["active_guard"] != 0
    ):
        raise AuditError("active-set identity drift")
    return delta, report


def apply_ram_delta(
    model: torch.nn.Module,
    delta: torch.Tensor,
) -> tuple[float, dict[str, Any]]:
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    named = dict(model.named_parameters())
    offset = 0
    actual: list[torch.Tensor] = []
    with torch.no_grad():
        for name in ACTOR6:
            parameter = named[name]
            count = parameter.numel()
            piece = delta[offset : offset + count].reshape(parameter.shape).to(parameter.dtype)
            source = parameter.detach().clone()
            parameter.add_(piece)
            actual.append((parameter.detach() - source).double().reshape(-1))
            offset += count
    if offset != delta.numel():
        raise AuditError("actor6 delta length drift")
    after = model.state_dict()
    changed = {name for name in after if not torch.equal(before[name], after[name])}
    if changed != set(EXPECTED_CHANGED_ACTOR_TENSORS):
        raise AuditError(f"RAM endpoint changed unexpected tensors: {sorted(changed)}")
    model.eval().requires_grad_(False)
    return float(torch.linalg.vector_norm(torch.cat(actual))), {
        "changed_actor_tensors": sorted(changed),
        "unchanged_actor_tensors": sorted(set(ACTOR6) - changed),
        "all_nonactor_tensors_bitwise_unchanged": all(
            torch.equal(before[name], after[name]) for name in after if name not in ACTOR6
        ),
    }


def exact_constraint_audit(
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    raw_rows: Mapping[str, Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    target_cells: int,
) -> dict[str, Any]:
    target_values: list[float] = []
    for line_hash in TARGET_IDS:
        cells, _ = policy_cells(model, raw_rows[line_hash], runtime, config)
        target_values.extend(float(value.detach()) for value, _ in cells)
    guard_values: list[float] = []
    for offset, descriptor in enumerate(retention):
        line_hash = str(descriptor["line_sha256"])
        source_cell = metadata[target_cells + offset]["cell"]
        cells, _ = policy_cells(model, raw_rows[line_hash], runtime, config)
        matches = [float(value.detach()) for value, cell in cells if cell == source_cell]
        if len(matches) != 1:
            raise AuditError(f"source-critical endpoint cell missing: {line_hash}")
        guard_values.append(matches[0])
    return {
        "exact_min_target_cell_margin": min(target_values),
        "exact_target_cells_below_nominal_tau": sum(value < TAU for value in target_values),
        "exact_all_target_cells_positive": all(value > 0.0 for value in target_values),
        "exact_min_source_critical_guard_margin": min(guard_values),
        "exact_guard_violations": sum(value < GUARD_FLOOR for value in guard_values),
        "numeric_nominal_tau_gate_pass": all(value >= TAU for value in target_values),
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "rows": 0,
        "source_hybrid_correct": 0,
        "source_set_correct": 0,
        "source_count_correct": 0,
        "candidate_hybrid_correct": 0,
        "candidate_set_correct": 0,
        "candidate_count_correct": 0,
        "repairs": [],
        "source_correct_to_wrong": [],
        "set_harm": [],
        "wrong_to_different_wrong": [],
        "hybrid_action_changes": [],
        "raw_policy_action_changes": [],
        "count_changes": [],
        "selected467_flips": [],
    }


def add_event(
    bucket: list[dict[str, Any]],
    line_hash: str,
    context: int,
    source_action: Sequence[int],
    candidate_action: Sequence[int],
) -> None:
    bucket.append(
        {
            "line_sha256": line_hash,
            "context": context,
            "source_action": [int(value) for value in source_action],
            "candidate_action": [int(value) for value in candidate_action],
        }
    )


def event_report(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = sorted(str(value["line_sha256"]) for value in values)
    return {
        "count": len(values),
        "by_context": dict(
            sorted(
                Counter(str(int(value["context"])) for value in values).items(),
                key=lambda item: int(item[0]),
            )
        ),
        "line_sha256_digest": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "details": list(values),
    }


def finalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in metrics.items() if not isinstance(value, list)}
    for key, value in metrics.items():
        if isinstance(value, list):
            result[key] = event_report(value)
    return result


def scan_archive(
    source: str,
    path: Path,
    model: torch.nn.Module,
    runtime: Any,
    config: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
    retention_ids: set[str],
) -> dict[str, Any]:
    metrics = empty_metrics()
    for row, identity in iter_train_lines(path):
        line_hash = identity["line_sha256"]
        if line_hash not in source_records:
            raise AuditError(f"source cache row missing: {source}:{line_hash}")
        source_row = source_records[line_hash]
        _, batch = inference_batch(row, runtime, config)
        expert = [int(value) for value in row.get("action", [])]
        context = int(batch["contexts"][0])
        with torch.inference_mode():
            outputs = model(batch)
        decoded = profile_tool.decode_hybrid(outputs, batch)
        candidate_hybrid = [int(value) for value in decoded["hybrid_action"]]
        candidate_raw = [int(value) for value in decoded["raw_action"]]
        candidate_hybrid_correct = candidate_hybrid == expert
        candidate_set_correct = sorted(candidate_raw) == sorted(expert)
        candidate_count_correct = len(candidate_hybrid) == len(expert)
        source_hybrid = [int(value) for value in source_row["source_hybrid_action"]]
        source_raw = [int(value) for value in source_row["source_policy_action"]]
        source_hybrid_correct = bool(source_row["source_hybrid_correct"])
        source_set_correct = bool(source_row["source_set_correct"])
        source_count_correct = bool(source_row["source_count_correct"])
        metrics["rows"] += 1
        metrics["source_hybrid_correct"] += int(source_hybrid_correct)
        metrics["source_set_correct"] += int(source_set_correct)
        metrics["source_count_correct"] += int(source_count_correct)
        metrics["candidate_hybrid_correct"] += int(candidate_hybrid_correct)
        metrics["candidate_set_correct"] += int(candidate_set_correct)
        metrics["candidate_count_correct"] += int(candidate_count_correct)
        if not source_hybrid_correct and candidate_hybrid_correct:
            add_event(metrics["repairs"], line_hash, context, source_hybrid, candidate_hybrid)
        if source_hybrid_correct and not candidate_hybrid_correct:
            add_event(
                metrics["source_correct_to_wrong"],
                line_hash,
                context,
                source_hybrid,
                candidate_hybrid,
            )
        if source_set_correct and not candidate_set_correct:
            add_event(metrics["set_harm"], line_hash, context, source_hybrid, candidate_hybrid)
        if (
            not source_hybrid_correct
            and not candidate_hybrid_correct
            and source_hybrid != candidate_hybrid
        ):
            add_event(
                metrics["wrong_to_different_wrong"],
                line_hash,
                context,
                source_hybrid,
                candidate_hybrid,
            )
        if source_hybrid != candidate_hybrid:
            add_event(
                metrics["hybrid_action_changes"],
                line_hash,
                context,
                source_hybrid,
                candidate_hybrid,
            )
        if source_raw != candidate_raw:
            add_event(
                metrics["raw_policy_action_changes"],
                line_hash,
                context,
                source_raw,
                candidate_raw,
            )
        if len(source_hybrid) != len(candidate_hybrid):
            add_event(metrics["count_changes"], line_hash, context, source_hybrid, candidate_hybrid)
        if source == "old" and line_hash in retention_ids and not candidate_hybrid_correct:
            add_event(
                metrics["selected467_flips"],
                line_hash,
                context,
                source_hybrid,
                candidate_hybrid,
            )
        if metrics["rows"] % 5000 == 0:
            print(f"scan {source}: {metrics['rows']}", flush=True)
    return finalize_metrics(metrics)


def validate_expected(scan: Mapping[str, Mapping[str, Any]]) -> None:
    for source, expected in EXPECTED.items():
        actual = scan[source]
        for key, value in expected.items():
            observed = actual[key]["count"] if isinstance(actual.get(key), Mapping) else actual.get(key)
            if observed != value:
                raise AuditError(
                    f"full-train gate drift {source}.{key}: {observed!r} != {value!r}"
                )
    yanz_contexts = scan["yanz"]["repairs"]["by_context"]
    old_contexts = scan["old"]["repairs"]["by_context"]
    if yanz_contexts != {"0": 2, "8": 1} or old_contexts != {"0": 1}:
        raise AuditError("repair context distribution drift")


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    started = time.time()
    startup_script_sha = file_sha256(SCRIPT)
    profile, retention, static = bind_profile()
    source_records = load_source_records(profile)
    raw_rows = load_constraint_rows(retention)
    runtime = profile_tool.load_bound_runtime()
    _, source_model, config, source_audit = profile_tool.instantiate_deployment_source(runtime)
    source_state_sha = profile_tool.bitwise_model_state_sha256(source_model.state_dict())
    if source_state_sha != profile_tool.SOURCE_EXPANDED61_BITWISE_STATE_SHA256:
        raise AuditError("expanded pure-BC source state drift")
    gradients, base, kinds, metadata, target_cells = actor6_gradients(
        source_model, runtime, config, raw_rows, retention
    )
    delta, geometry = solve_projection(gradients, base, kinds, target_cells)
    actual_l2, mutation = apply_ram_delta(source_model, delta)
    geometry["actual_fp32_l2"] = actual_l2
    geometry["requested_minus_actual_l2"] = geometry["linear_l2"] - actual_l2
    geometry["ram_model_state_sha256"] = profile_tool.bitwise_model_state_sha256(
        source_model.state_dict()
    )
    geometry["ram_mutation"] = mutation
    geometry.update(
        exact_constraint_audit(
            source_model,
            runtime,
            config,
            raw_rows,
            retention,
            metadata,
            target_cells,
        )
    )
    if not geometry["exact_all_target_cells_positive"] or geometry["exact_guard_violations"]:
        raise AuditError("exact endpoint behavior-cell gate failed")

    del gradients, delta
    scan = {
        "yanz": scan_archive(
            "yanz",
            profile_tool.YANZ,
            source_model,
            runtime,
            config,
            source_records["yanz"],
            {str(row["line_sha256"]) for row in retention},
        ),
        "old": scan_archive(
            "old",
            profile_tool.OLD,
            source_model,
            runtime,
            config,
            source_records["old"],
            {str(row["line_sha256"]) for row in retention},
        ),
    }
    validate_expected(scan)
    if file_sha256(SCRIPT) != startup_script_sha:
        raise AuditError("audit script changed while running")
    result = {
        "schema_version": SCHEMA,
        "status": "train_only_behavior_pass_numeric_nominal_tau_shortfall",
        "bindings": {
            "audit_tool": {"path": str(SCRIPT.resolve()), "sha256": startup_script_sha},
            "profile": {"path": str(PROFILE.resolve()), "sha256": PROFILE_SHA256},
            "profiler_sha256": PROFILER_SHA256,
            "hybrid_main_sha256": profile_tool.HYBRID_MAIN_SHA256,
            "policy_runtime_sha256": profile_tool.RUNTIME_SHA256,
            "source_expanded_state_sha256": source_state_sha,
            "static": static,
            "profile_cache_shard_count": len(profile["cache"]["immutable_shards"]),
        },
        "contract": {
            "runtime": "CPU FP32 batch=1 exact hybrid deployment semantics",
            "target_ids": list(TARGET_IDS),
            "target_policy_cells": target_cells,
            "retention_source_critical_policy_guards": 467,
            "tau": TAU,
            "guard_floor": GUARD_FLOOR,
            "actor_tensors": list(ACTOR6),
            "checkpoint_written": False,
            "opened_members": "train/*.jsonl only",
        },
        "source_model": source_audit,
        "geometry": geometry,
        "full_train_scan": scan,
        "gates": {
            "behavior_expected_counts_exact": True,
            "all_target_actions_repaired": scan["yanz"]["repairs"]["count"] == 3,
            "no_source_correct_to_wrong": all(
                scan[source]["source_correct_to_wrong"]["count"] == 0
                for source in ("yanz", "old")
            ),
            "no_set_harm": all(
                scan[source]["set_harm"]["count"] == 0 for source in ("yanz", "old")
            ),
            "no_wrong_to_different_wrong": all(
                scan[source]["wrong_to_different_wrong"]["count"] == 0
                for source in ("yanz", "old")
            ),
            "no_count_change": all(
                scan[source]["count_changes"]["count"] == 0
                for source in ("yanz", "old")
            ),
            "selected467_zero_flips": scan["old"]["selected467_flips"]["count"] == 0,
            "numeric_nominal_tau_gate_pass": geometry["numeric_nominal_tau_gate_pass"],
        },
        "scope": {
            "train_members_only": True,
            "checkpoint_written": False,
            "holdout_opened": False,
            "submission_performed": False,
            "ram_endpoint_only": True,
        },
        "seconds": time.time() - started,
    }
    profile_tool.write_new_json(result, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output),
                "geometry": result["geometry"],
                "full_train_counts": {
                    source: {
                        key: value
                        for key, value in result["full_train_scan"][source].items()
                        if not isinstance(value, dict)
                    }
                    for source in ("yanz", "old")
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
