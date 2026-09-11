#!/usr/bin/env python3
"""Freeze the train-only row profile for the yanz actor6 repair.

This program is deliberately a *selector*, not a trainer.  It opens only the
``train/`` members of the two hash-bound archives, evaluates the frozen BC
source with deployment-compatible deterministic decoding, and writes an exact
row ledger.  No valid/test/time-forward archive is accepted by the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import orjson
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


SCHEMA = "ptcg-yanz-source-error-actor6-profile-v1"
SOURCE = ROOT / "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/best.pt"
YANZ = ROOT / (
    "data/yanz_alakazam_live_20260810_v1/snapshot_20260810T0433Z/"
    "policies/yanzhou06_alakazam_live_55394520.zip"
)
OLD = ROOT / "data/gold8_recent7_20260808/archives/alakazam_control.zip"
SOURCE_SHA256 = "f5500086c16a02c19f3f2abce5e144446bd079248fd9fc3f4a620f3d079c7626"
SOURCE_BITWISE_STATE_SHA256 = (
    "b0bfe317017a9d4f3cbd367cd21ff96c7fdc60abd1924a6d8edbdf7b32917f91"
)
SOURCE_PPO_STYLE_STATE_SHA256 = (
    "843e2f497115d99863518a6d984ee50b8a5d6e0a4aacd26cd2dd3ec66968be84"
)
SOURCE_EXPANDED61_PPO_STYLE_STATE_SHA256 = (
    "45ae40783c28ad140c0a625b0950bd0d2f943a7cbff72cf95178abffa8d5dded"
)
YANZ_SHA256 = "bbbb6f69b809c60e900f7d2992333b645d117b2a6c836642fecca0e1f4226720"
OLD_SHA256 = "4bd0de193cfb88b060435f84bbb5dc83a9eff3ab1aa4a498ef0859d2346f3c6c"
DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
ACTOR6 = tuple(bc.TRANSFORMER_ACTOR_HEAD_PARAMETER_NAMES)
ACTOR6_TENSORS = 6
ACTOR6_PARAMETERS = 65_793
TREATMENT_STEP_SIZES = (101, 100, 100)
BOUNDARY_STEP_SIZES = (123, 124, 124)
BROAD_PER_STEP = 32
MIN_CONTEXT34_BOUNDARY_PER_STEP = 4
SELECTION_SALT = "ptcg-yanz-actor6-pcgrad-specialbc-selection-v1"


class ProtocolError(RuntimeError):
    """A fail-closed profile contract violation."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def bitwise_model_state_sha256(state: Mapping[str, Any]) -> str:
    """The explicitly named ``ptcg-model-state-bitwise-v1`` digest."""

    digest = hashlib.sha256()
    digest.update(b"ptcg-model-state-bitwise-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("model state must map string names to tensors")
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "big"))
        digest.update(dtype_bytes)
        digest.update(len(tensor.shape).to_bytes(4, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=True))
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def raw_train_rows(path: Path) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield raw rows and immutable identities from train members only."""

    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not members:
            raise ProtocolError(f"no train JSONL members: {path}")
        for member in members:
            with archive.open(member) as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = orjson.loads(line)
                    if str(row.get("split", "")) != "train":
                        raise ProtocolError(f"non-train row in {path}:{member}")
                    if str(row.get("deck_hash", "")) != DECK_HASH:
                        raise ProtocolError(f"deck drift in {path}:{member}")
                    yield row, {
                        "archive_member": member,
                        "member_line_number": line_number,
                        "raw_line_sha256": hashlib.sha256(line).hexdigest(),
                        "line_sha256": hashlib.sha256(
                            line.rstrip(b"\r\n")
                        ).hexdigest(),
                        "line_sha256_algorithm": "sha256(raw_line_rstrip_CR_LF)",
                    }


def feature_visible_signature(feature: Mapping[str, Any]) -> str:
    visible = {
        "global_fields": feature["global_fields"],
        "global_numeric": feature["global_numeric"],
        "entity_fields": feature["entity_fields"],
        "entity_numeric": feature["entity_numeric"],
        "option_fields": feature["option_fields"],
        "option_numeric": feature["option_numeric"],
        "min_count": feature["min_count"],
        "max_count": feature["max_count"],
        "context": feature["context"],
    }
    return hashlib.sha256(
        orjson.dumps(visible, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def featurize_deployment(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Featurize under the deployment evaluator's 0..60 action contract."""

    previous = bc.MAX_ACTION_COUNT
    bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
    try:
        return bc.featurize_row(
            dict(row),
            int(config["hash_size"]),
            int(config["max_state_entities"]),
        )
    finally:
        bc.MAX_ACTION_COUNT = previous


def featurize_training(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Featurize the actor6 selection/training pool under BC's 0..16 cap."""

    if bc.MAX_ACTION_COUNT != 16:
        raise ProtocolError("BC training action-count cap drift")
    return bc.featurize_row(
        dict(row),
        int(config["hash_size"]),
        int(config["max_state_entities"]),
    )


# Backwards-compatible internal alias: every selector/rehydrator call is the
# <=16 training pool.  The full behavior gate calls featurize_deployment.
featurize = featurize_training


def instantiate_raw_bc(
    checkpoint: Mapping[str, Any], device: torch.device
) -> tuple[bc.EntityOptionPolicy, dict[str, Any]]:
    config = ppo.checkpoint_model_config(dict(checkpoint))
    model = bc.EntityOptionPolicy(
        hash_size=config["hash_size"],
        categorical_dim=config["categorical_dim"],
        model_dim=config["model_dim"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
        max_state_entities=config["max_state_entities"],
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or len(state) != 80:
        raise ProtocolError("source is not the expected 80-tensor BC state")
    model.load_state_dict(state)
    if model.count_head[-1].out_features != 17:
        raise ProtocolError("source count head is not the original 17-class head")
    model.to(device).eval()
    return model, config


def deployment_count_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    """Exactly emulate ``instantiate_model_from_bc`` count-head expansion."""

    if raw_logits.shape[1] != 17:
        raise ProtocolError("raw count logits must have 17 classes")
    expanded = raw_logits.new_full((raw_logits.shape[0], 61), -10.0)
    expanded[:, :17] = raw_logits
    return expanded


def collate(
    features: Sequence[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    return bc.collate_decisions(
        list(features),
        max_state_entities=int(config["max_state_entities"]),
        entity_fields=int(config["entity_fields"]),
        option_fields=int(config["option_fields"]),
    )


def deterministic_policy_actions(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> list[list[int]]:
    expanded = dict(outputs)
    if outputs["count_logits"].shape[1] == 17:
        expanded["count_logits"] = deployment_count_logits(
            outputs["count_logits"].float()
        )
    elif outputs["count_logits"].shape[1] != 61:
        raise ProtocolError("deployment decoder requires 17 or 61 count classes")
    actions, _, _, _ = ppo.sample_ordered_actions(
        expanded,
        dict(batch),
        deterministic=True,
        canonicalize_order=False,
    )
    return actions


def hybrid_margin(
    logits: torch.Tensor,
    expert_order: Sequence[int],
    context: int,
    option_count: int,
) -> float:
    """Positive distance to the exact actor decision boundary."""

    row_logits = logits[:option_count].float()
    if context == ppo.SKILL_ORDER_CONTEXT:
        if not expert_order:
            return math.inf
        remaining = torch.ones(option_count, dtype=torch.bool, device=logits.device)
        margins: list[float] = []
        for chosen in expert_order:
            alternatives = remaining.clone()
            alternatives[int(chosen)] = False
            if alternatives.any():
                margins.append(
                    float(row_logits[int(chosen)] - row_logits[alternatives].max())
                )
            remaining[int(chosen)] = False
        return min(margins, default=math.inf)
    if not expert_order or len(expert_order) == option_count:
        return math.inf
    target = torch.zeros(option_count, dtype=torch.bool, device=logits.device)
    target[list(expert_order)] = True
    if not target.any() or target.all():
        return math.inf
    return float(row_logits[target].min() - row_logits[~target].max())


def compact_record(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    feature: Mapping[str, Any],
    predicted: Sequence[int],
    policy_logits: torch.Tensor,
) -> dict[str, Any]:
    expert_order = [int(value) for value in row.get("action", [])]
    context = int(feature["context"])
    option_count = len(feature["targets"])
    return {
        **identity,
        "decision_key": (
            f"{row.get('episode_id')}:"
            f"{row.get('observation_step_index')}->"
            f"{row.get('action_step_index')}"
        ),
        "episode_id": str(row.get("episode_id", "")),
        "episode_uuid": str(row.get("episode_uuid", "")),
        "visible_signature": feature_visible_signature(feature),
        "context": context,
        "team_name": str(row.get("team_name", "")),
        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        "expert_action_order": expert_order,
        "source_policy_action": [int(value) for value in predicted],
        "source_set_correct": sorted(predicted) == sorted(expert_order),
        "source_hybrid_correct": (
            list(predicted) if context == ppo.SKILL_ORDER_CONTEXT else sorted(predicted)
        )
        == expert_order,
        "source_count_correct": len(predicted) == len(expert_order),
        "hybrid_margin": hybrid_margin(
            policy_logits,
            expert_order,
            context,
            option_count,
        ),
    }


def profile_archive(
    path: Path,
    source_name: str,
    model: bc.EntityOptionPolicy,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    raw_rows = 0
    skipped = 0

    def flush() -> None:
        if not pending:
            return
        cpu = collate([item[2] for item in pending], config)
        batch = {name: value.to(device) for name, value in cpu.items()}
        with torch.no_grad():
            outputs = ppo.model_forward(model, batch, device)
            actions = deterministic_policy_actions(outputs, batch)
        for index, (row, identity, feature) in enumerate(pending):
            record = compact_record(
                row,
                identity,
                feature,
                actions[index],
                outputs["policy_logits"][index],
            )
            record["source"] = source_name
            records.append(record)
        pending.clear()

    for row, identity in raw_train_rows(path):
        raw_rows += 1
        feature = featurize(row, config)
        if feature is None:
            skipped += 1
            continue
        pending.append((row, identity, feature))
        if len(pending) >= batch_size:
            flush()
    flush()
    line_hashes = [row["line_sha256"] for row in records]
    if len(line_hashes) != len(set(line_hashes)):
        raise ProtocolError(f"duplicate accepted line SHA in {source_name}")
    return records, {
        "raw_rows": raw_rows,
        "accepted_rows": len(records),
        "skipped_rows": skipped,
    }


def stable_hash(label: str, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        f"{SELECTION_SALT}:{label}:{row['line_sha256']}".encode("utf-8")
    ).hexdigest()


def assign_fixed_sizes(
    rows: Sequence[dict[str, Any]], sizes: Sequence[int], label: str
) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (stable_hash(label, row), row["line_sha256"]))
    if len(ordered) != sum(sizes):
        raise ProtocolError(f"{label} size drift: {len(ordered)}")
    output: list[list[dict[str, Any]]] = []
    start = 0
    for size in sizes:
        output.append(ordered[start : start + size])
        start += size
    return output


def choose_boundary_rows(
    eligible: Sequence[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    finite = [
        row
        for row in eligible
        if math.isfinite(float(row["hybrid_margin"]))
        and float(row["hybrid_margin"]) > 0.0
    ]
    ordered = sorted(
        finite,
        key=lambda row: (
            float(row["hybrid_margin"]),
            stable_hash("boundary-tie", row),
            row["line_sha256"],
        ),
    )
    total = sum(BOUNDARY_STEP_SIZES)
    if len(ordered) < total:
        raise ProtocolError("insufficient positive-margin old-train rows")
    pool = ordered[:total]
    context34 = [row for row in pool if int(row["context"]) == 34]
    required = MIN_CONTEXT34_BOUNDARY_PER_STEP * 3
    replacements: list[dict[str, Any]] = []
    if len(context34) < required:
        outside34 = [row for row in ordered[total:] if int(row["context"]) == 34]
        need = required - len(context34)
        if len(outside34) < need:
            raise ProtocolError("insufficient context34 boundary retention rows")
        replaceable = sorted(
            [row for row in pool if int(row["context"]) != 34],
            key=lambda row: (float(row["hybrid_margin"]), row["line_sha256"]),
            reverse=True,
        )[:need]
        replacement_ids = {row["line_sha256"] for row in replaceable}
        replacements = outside34[:need]
        pool = [row for row in pool if row["line_sha256"] not in replacement_ids]
        pool.extend(replacements)

    steps: list[list[dict[str, Any]]] = [[] for _ in range(3)]
    capacities = list(BOUNDARY_STEP_SIZES)
    c34_rows = sorted(
        [row for row in pool if int(row["context"]) == 34],
        key=lambda row: (float(row["hybrid_margin"]), row["line_sha256"]),
    )
    used: set[str] = set()
    for step_index in range(3):
        for row in c34_rows[
            step_index * MIN_CONTEXT34_BOUNDARY_PER_STEP :
            (step_index + 1) * MIN_CONTEXT34_BOUNDARY_PER_STEP
        ]:
            steps[step_index].append(row)
            used.add(row["line_sha256"])
    remaining = sorted(
        [row for row in pool if row["line_sha256"] not in used],
        key=lambda row: (
            float(row["hybrid_margin"]),
            stable_hash("boundary-assign", row),
        ),
    )
    for row in remaining:
        candidates = [
            index for index in range(3) if len(steps[index]) < capacities[index]
        ]
        if not candidates:
            raise ProtocolError("boundary assignment overflow")
        chosen = min(
            candidates,
            key=lambda index: (
                len(steps[index]) / capacities[index],
                stable_hash(f"boundary-step-{index + 1}", row),
            ),
        )
        steps[chosen].append(row)
    if [len(rows) for rows in steps] != capacities:
        raise ProtocolError("boundary step quota drift")
    if any(sum(int(row["context"]) == 34 for row in rows) < 4 for rows in steps):
        raise ProtocolError("context34 boundary quota drift")
    return steps, {
        "eligible_positive_margin_rows": len(finite),
        "selected_rows": len(pool),
        "context34_rows": sum(int(row["context"]) == 34 for row in pool),
        "quota_replacements": len(replacements),
        "selection": (
            "lowest positive hybrid margins globally; if required, replace the "
            "largest non-context34 boundary rows by the lowest outside context34 "
            "rows until the preregistered 4-per-step minimum is feasible"
        ),
    }


def choose_broad_rows(
    eligible: Sequence[dict[str, Any]], excluded: set[str]
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in eligible:
        if row["line_sha256"] in excluded:
            continue
        groups[(int(row["context"]), str(row["team_name"]))].append(row)
    for key, values in list(groups.items()):
        groups[key] = deque(
            sorted(values, key=lambda row: stable_hash(f"broad:{key}", row))
        )
    output: list[list[dict[str, Any]]] = []
    for step_index in range(3):
        chosen: list[dict[str, Any]] = []
        round_index = 0
        while len(chosen) < BROAD_PER_STEP:
            active = [key for key, values in groups.items() if values]
            if not active:
                raise ProtocolError("insufficient broad retention rows")
            active.sort(
                key=lambda key: hashlib.sha256(
                    f"{SELECTION_SALT}:broad-step-{step_index + 1}:"
                    f"round-{round_index}:{key[0]}:{key[1]}".encode()
                ).hexdigest()
            )
            made_progress = False
            for key in active:
                if len(chosen) >= BROAD_PER_STEP:
                    break
                chosen.append(groups[key].popleft())
                made_progress = True
            if not made_progress:
                raise ProtocolError("broad stratification made no progress")
            round_index += 1
        output.append(chosen)
    return output


def descriptor(row: Mapping[str, Any], category: str) -> dict[str, Any]:
    keys = (
        "source",
        "archive_member",
        "member_line_number",
        "raw_line_sha256",
        "line_sha256",
        "line_sha256_algorithm",
        "decision_key",
        "episode_id",
        "episode_uuid",
        "visible_signature",
        "context",
        "team_name",
        "sample_weight",
        "expert_action_order",
        "source_policy_action",
        "source_set_correct",
        "source_hybrid_correct",
        "source_count_correct",
        "hybrid_margin",
    )
    result = {key: row[key] for key in keys}
    margin = float(result["hybrid_margin"])
    result["hybrid_margin"] = margin if math.isfinite(margin) else None
    result["category"] = category
    return result


def build_profile(
    model: bc.EntityOptionPolicy,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    yanz, yanz_counts = profile_archive(
        YANZ, "yanz", model, config, device, batch_size
    )
    old, old_counts = profile_archive(OLD, "old", model, config, device, batch_size)
    if yanz_counts != {"raw_rows": 1725, "accepted_rows": 1725, "skipped_rows": 0}:
        raise ProtocolError(f"yanz row count drift: {yanz_counts}")
    if old_counts != {"raw_rows": 59875, "accepted_rows": 59846, "skipped_rows": 29}:
        raise ProtocolError(f"old row count drift: {old_counts}")

    set_errors = [row for row in yanz if not row["source_set_correct"]]
    treatment = [row for row in set_errors if row["source_count_correct"]]
    count_errors = [row for row in set_errors if not row["source_count_correct"]]
    if (len(set_errors), len(treatment), len(count_errors)) != (303, 301, 2):
        raise ProtocolError(
            "source error contract drift: "
            f"{len(set_errors)}/{len(treatment)}/{len(count_errors)}"
        )
    treatment_steps = assign_fixed_sizes(
        treatment, TREATMENT_STEP_SIZES, "treatment-partition"
    )

    yanz_episode_ids = {row["episode_id"] for row in yanz}
    yanz_episode_uuids = {row["episode_uuid"] for row in yanz}
    yanz_signatures = {row["visible_signature"] for row in yanz}
    eligible_old = [
        row
        for row in old
        if row["source_hybrid_correct"]
        and row["source_count_correct"]
        and row["episode_id"] not in yanz_episode_ids
        and row["episode_uuid"] not in yanz_episode_uuids
        and row["visible_signature"] not in yanz_signatures
        and math.isfinite(float(row["hybrid_margin"]))
        and float(row["hybrid_margin"]) > 0.0
    ]
    boundary_steps, boundary_audit = choose_boundary_rows(eligible_old)
    boundary_ids = {
        row["line_sha256"] for rows in boundary_steps for row in rows
    }
    broad_steps = choose_broad_rows(eligible_old, boundary_ids)
    all_retention = [
        row for step in range(3) for row in boundary_steps[step] + broad_steps[step]
    ]
    retention_ids = [row["line_sha256"] for row in all_retention]
    if len(retention_ids) != 467 or len(set(retention_ids)) != 467:
        raise ProtocolError("retention union is not 467 mutually exclusive rows")

    steps = []
    for index in range(3):
        treatment_rows = [descriptor(row, "treatment") for row in treatment_steps[index]]
        boundary_rows = [descriptor(row, "retention_boundary") for row in boundary_steps[index]]
        broad_rows = [descriptor(row, "retention_broad") for row in broad_steps[index]]
        retention_rows = boundary_rows + broad_rows
        steps.append(
            {
                "step": index + 1,
                "treatment": treatment_rows,
                "retention_boundary": boundary_rows,
                "retention_broad": broad_rows,
                "retention": retention_rows,
                "counts": {
                    "treatment": len(treatment_rows),
                    "retention_boundary": len(boundary_rows),
                    "retention_broad": len(broad_rows),
                    "retention": len(retention_rows),
                    "retention_context34": sum(
                        int(row["context"]) == 34 for row in boundary_rows
                    ),
                },
            }
        )

    return {
        "schema_version": SCHEMA,
        "status": "frozen_train_only_row_selection",
        "bindings": {
            "source_checkpoint": {
                "path": str(SOURCE),
                "sha256": SOURCE_SHA256,
                "model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
                "model_state_sha256": SOURCE_BITWISE_STATE_SHA256,
                "train_ppo_model_state_sha256_algorithm": (
                    "name-NUL-dtype-NUL-shape-NUL-raw-NUL-v1"
                ),
                "train_ppo_model_state_sha256": SOURCE_PPO_STYLE_STATE_SHA256,
                "expanded61_train_ppo_model_state_sha256": (
                    SOURCE_EXPANDED61_PPO_STYLE_STATE_SHA256
                ),
                "tensor_count": 80,
                "count_classes": 17,
            },
            "yanz_train_archive": {"path": str(YANZ), "sha256": YANZ_SHA256},
            "old_train_archive": {"path": str(OLD), "sha256": OLD_SHA256},
            "deck_hash": DECK_HASH,
        },
        "selection_contract": {
            "opened_members": "train/*.jsonl only",
            "forbidden": (
                "valid/test members, timeforward archives, leaderboard scores, "
                "and any held-out label or metric"
            ),
            "treatment": (
                "all source set errors with source-correct deployment cardinality; "
                "the two count-involved errors are excluded"
            ),
            "treatment_partition": (
                "ascending salted SHA-256, consecutive fixed blocks 101/100/100"
            ),
            "retention_eligibility": (
                "old-train source hybrid-correct and count-correct, with no yanz "
                "episode id, episode UUID, or exact visible-signature overlap"
            ),
            "boundary": boundary_audit,
            "broad": (
                "32 per step by deterministic context/team round-robin with "
                "salted SHA order inside each stratum"
            ),
            "cross_step_overlap": 0,
        },
        "source_counts": {
            "yanz": yanz_counts,
            "yanz_set_correct": sum(row["source_set_correct"] for row in yanz),
            "yanz_hybrid_correct": sum(row["source_hybrid_correct"] for row in yanz),
            "yanz_count_correct": sum(row["source_count_correct"] for row in yanz),
            "yanz_set_errors": len(set_errors),
            "treatment_selection_only": len(treatment),
            "excluded_count_errors": len(count_errors),
            "old": old_counts,
            "old_hybrid_correct": sum(row["source_hybrid_correct"] for row in old),
            "eligible_old_retention": len(eligible_old),
            "retention_selected": len(retention_ids),
        },
        "excluded_count_error_rows": [
            descriptor(row, "excluded_count_error") for row in count_errors
        ],
        "steps": steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen profile: {args.output}")
    for path, expected in ((SOURCE, SOURCE_SHA256), (YANZ, YANZ_SHA256), (OLD, OLD_SHA256)):
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"hash-bound input drift: {path}")
    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ProtocolError("source checkpoint root is not a mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ProtocolError("source checkpoint has no model_state_dict")
    state_hash = bitwise_model_state_sha256(state)
    if state_hash != SOURCE_BITWISE_STATE_SHA256:
        raise ProtocolError(f"source bitwise state hash drift: {state_hash}")
    device = torch.device(args.device)
    raw_model, config = instantiate_raw_bc(checkpoint, device)
    if ppo.model_state_sha256(raw_model) != SOURCE_PPO_STYLE_STATE_SHA256:
        raise ProtocolError("source train_ppo-style state hash drift")
    model = ppo.instantiate_model_from_bc(dict(checkpoint), device)
    if ppo.model_state_sha256(model) != SOURCE_EXPANDED61_PPO_STYLE_STATE_SHA256:
        raise ProtocolError("source expanded61 train_ppo-style state hash drift")
    torch.set_float32_matmul_precision("high")
    profile = build_profile(model, config, device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(profile)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "steps": [step["counts"] for step in profile["steps"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    # Keep repeated GPU executions stable enough for a byte-identical row ledger.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
