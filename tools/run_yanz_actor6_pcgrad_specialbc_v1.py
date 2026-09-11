#!/usr/bin/env python3
"""Freeze or execute one train-only yanz actor6 PCGrad special-BC route.

The executable has no validation/test/time-forward arguments.  ``freeze``
binds the source, archives, exact row profile, code hashes, optimizer, gates,
and output paths.  ``execute`` requires the resulting protocol SHA-256, runs
the three fixed updates twice from RAM, and writes a 17-class BC checkpoint
only when every train-only gate passes in both replays.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import orjson
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_yanz_source_error_actor6_v1 as profile_tool  # noqa: E402
import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


SCHEMA = "ptcg-yanz-actor6-pcgrad-specialbc-protocol-v1"
AUDIT_SCHEMA = "ptcg-yanz-actor6-pcgrad-specialbc-audit-v1"
RUNNER = Path(__file__).resolve()
PROFILER = TOOLS / "profile_yanz_source_error_actor6_v1.py"
BC_TOOL = TOOLS / "train_bc_orbit.py"
PPO_TOOL = TOOLS / "train_ppo.py"
EVALUATOR = TOOLS / "evaluate_policy_bc.py"
H2H_TOOL = TOOLS / "evaluate_ppo_head_to_head.py"
TIMEFORWARD = ROOT / (
    "data/yanz_alakazam_timeforward_20260810_v1/"
    "snapshot_20260810T0506Z_strict_holdout/policies/"
    "yanzhou06_alakazam_timeforward_after_043806.zip"
)
TIMEFORWARD_SHA256 = (
    "f133c627788669b9b1cbcd05e2a1d2f06143842d850269df468c647fc619c18c"
)
DECK_CSV = ROOT / (
    "data/gold8_recent7_20260808/decks/"
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv"
)
DECK_CSV_SHA256 = (
    "0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451"
)
ACTOR6 = tuple(profile_tool.ACTOR6)
LEARNING_RATE = 1.0e-4
GRAD_CLIP_NORM = 0.5
STEP_L2_MAX = 5.0e-5
CUMULATIVE_L2_MAX = 1.5e-4
RETENTION_NLL_TOLERANCE = 1.0e-6
PCGRAD_COSINE_MIN = 1.0e-8
CONTEXT34_WEIGHT = 8.0
SET_BCE_WEIGHT = 0.25
EXPECTED_REPAIRS = 3
EXPECTED_REPAIR_CONTEXTS = 2
EXPECTED_YANZ_HYBRID = 1425


class ProtocolError(RuntimeError):
    """A fail-closed execution or integrity error."""


def file_sha256(path: Path) -> str:
    return profile_tool.file_sha256(path)


def canonical_json_bytes(value: Any) -> bytes:
    return profile_tool.canonical_json_bytes(value)


def load_json_strict(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root is not an object: {path}")
    return value


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != profile_tool.SCHEMA:
        raise ProtocolError("profile schema drift")
    if profile.get("status") != "frozen_train_only_row_selection":
        raise ProtocolError("profile is not frozen train-only selection")
    bindings = profile.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ProtocolError("profile bindings missing")
    expected = {
        "source_checkpoint": profile_tool.SOURCE_SHA256,
        "yanz_train_archive": profile_tool.YANZ_SHA256,
        "old_train_archive": profile_tool.OLD_SHA256,
    }
    for key, digest in expected.items():
        entry = bindings.get(key)
        if not isinstance(entry, Mapping) or entry.get("sha256") != digest:
            raise ProtocolError(f"profile binding drift: {key}")
    if bindings.get("deck_hash") != profile_tool.DECK_HASH:
        raise ProtocolError("profile deck hash drift")
    steps = profile.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        raise ProtocolError("profile must contain exactly three steps")
    treatment_ids: list[str] = []
    retention_ids: list[str] = []
    treatment_episodes: set[str] = set()
    treatment_uuids: set[str] = set()
    treatment_signatures: set[str] = set()
    retention_episodes: set[str] = set()
    retention_uuids: set[str] = set()
    retention_signatures: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or int(step.get("step", -1)) != index + 1:
            raise ProtocolError("profile step order drift")
        treatment = step.get("treatment")
        boundary = step.get("retention_boundary")
        broad = step.get("retention_broad")
        retention = step.get("retention")
        expected_t = profile_tool.TREATMENT_STEP_SIZES[index]
        expected_b = profile_tool.BOUNDARY_STEP_SIZES[index]
        if not all(isinstance(value, list) for value in (treatment, boundary, broad, retention)):
            raise ProtocolError("profile step row lists missing")
        if (len(treatment), len(boundary), len(broad), len(retention)) != (
            expected_t,
            expected_b,
            profile_tool.BROAD_PER_STEP,
            expected_b + profile_tool.BROAD_PER_STEP,
        ):
            raise ProtocolError("profile step quota drift")
        if retention != boundary + broad:
            raise ProtocolError("profile retention concatenation drift")
        if sum(int(row["context"]) == 34 for row in boundary) < 4:
            raise ProtocolError("profile context34 boundary quota drift")
        for row in treatment:
            if (
                row.get("category") != "treatment"
                or row.get("source") != "yanz"
                or bool(row.get("source_set_correct"))
                or bool(row.get("source_hybrid_correct"))
                or not bool(row.get("source_count_correct"))
            ):
                raise ProtocolError("profile treatment source predicate drift")
            treatment_episodes.add(str(row.get("episode_id")))
            treatment_uuids.add(str(row.get("episode_uuid")))
            treatment_signatures.add(str(row.get("visible_signature")))
        for row in boundary + broad:
            margin = row.get("hybrid_margin")
            if (
                row.get("category")
                not in {"retention_boundary", "retention_broad"}
                or row.get("source") != "old"
                or not bool(row.get("source_hybrid_correct"))
                or not bool(row.get("source_count_correct"))
                or not isinstance(margin, (int, float))
                or isinstance(margin, bool)
                or not math.isfinite(float(margin))
                or float(margin) <= 0.0
            ):
                raise ProtocolError("profile retention source/margin predicate drift")
            retention_episodes.add(str(row.get("episode_id")))
            retention_uuids.add(str(row.get("episode_uuid")))
            retention_signatures.add(str(row.get("visible_signature")))
        treatment_ids.extend(str(row["line_sha256"]) for row in treatment)
        retention_ids.extend(str(row["line_sha256"]) for row in retention)
    if len(treatment_ids) != 301 or len(set(treatment_ids)) != 301:
        raise ProtocolError("treatment ledger overlap/size drift")
    if len(retention_ids) != 467 or len(set(retention_ids)) != 467:
        raise ProtocolError("retention ledger overlap/size drift")
    if (
        treatment_episodes & retention_episodes
        or treatment_uuids & retention_uuids
        or treatment_signatures & retention_signatures
    ):
        raise ProtocolError("profile yanz/old episode or visible-signature overlap")
    source_counts = profile.get("source_counts")
    if not isinstance(source_counts, Mapping):
        raise ProtocolError("profile source counts missing")
    expected_source_counts = {
        "yanz_set_correct": 1422,
        "yanz_hybrid_correct": 1422,
        "yanz_count_correct": 1723,
        "yanz_set_errors": 303,
        "treatment_selection_only": 301,
        "excluded_count_errors": 2,
        "retention_selected": 467,
    }
    for key, expected in expected_source_counts.items():
        if int(source_counts.get(key, -1)) != expected:
            raise ProtocolError(f"profile source count drift: {key}")


def freeze_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {args.protocol}")
    if args.candidate_output.exists() or args.audit_output.exists():
        raise FileExistsError("candidate/audit output must be absent at freeze")
    profile = load_json_strict(args.profile)
    validate_profile(profile)
    bindings = {
        "source_checkpoint": {
            "path": str(profile_tool.SOURCE),
            "sha256": profile_tool.SOURCE_SHA256,
            "model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
            "model_state_sha256": profile_tool.SOURCE_BITWISE_STATE_SHA256,
            "train_ppo_model_state_sha256_algorithm": (
                "name-NUL-dtype-NUL-shape-NUL-raw-NUL-v1"
            ),
            "train_ppo_model_state_sha256": profile_tool.SOURCE_PPO_STYLE_STATE_SHA256,
            "expanded61_train_ppo_model_state_sha256": (
                profile_tool.SOURCE_EXPANDED61_PPO_STYLE_STATE_SHA256
            ),
            "raw_bc_tensor_count": 80,
            "raw_bc_count_classes": 17,
        },
        "yanz_train_archive": {
            "path": str(profile_tool.YANZ),
            "sha256": profile_tool.YANZ_SHA256,
        },
        "old_train_archive": {
            "path": str(profile_tool.OLD),
            "sha256": profile_tool.OLD_SHA256,
        },
        "profile": {
            "path": str(args.profile.resolve()),
            "sha256": file_sha256(args.profile),
        },
        "tools": {
            "runner": {"path": str(RUNNER), "sha256": file_sha256(RUNNER)},
            "profiler": {"path": str(PROFILER), "sha256": file_sha256(PROFILER)},
            "train_bc_orbit": {"path": str(BC_TOOL), "sha256": file_sha256(BC_TOOL)},
            "train_ppo": {"path": str(PPO_TOOL), "sha256": file_sha256(PPO_TOOL)},
            "deployment_evaluator": {
                "path": str(EVALUATOR),
                "sha256": file_sha256(EVALUATOR),
            },
            "head_to_head_evaluator": {
                "path": str(H2H_TOOL),
                "sha256": file_sha256(H2H_TOOL),
            },
        },
        "deck_hash": profile_tool.DECK_HASH,
        "deck_csv": {"path": str(DECK_CSV), "sha256": DECK_CSV_SHA256},
        "consumed_timeforward_holdout": {
            "path": str(TIMEFORWARD),
            "sha256": TIMEFORWARD_SHA256,
            "binding_only_not_opened_by_training": True,
        },
    }
    protocol = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "route": {
            "candidate_count": 1,
            "steps": 3,
            "candidate_endpoint": "step3_only",
            "candidate_kind": (
                "raw17_80_tensor_training_artifact; not directly submittable until "
                "a separately gated real61 PPO-compatible deployment package is "
                "materialized and verified"
            ),
            "two_identical_ram_replays_required": True,
            "validation_selection": False,
            "test_selection": False,
            "timeforward_selection": False,
            "sweep": False,
            "backtracking": False,
            "soup": False,
            "ppo": False,
        },
        "bindings": bindings,
        "training": {
            "parameter_scope": {
                "names": list(ACTOR6),
                "tensor_count": 6,
                "parameter_count": 65793,
                "frozen_tensor_count": 74,
            },
            "loss": {
                "name": "policy_only_hybrid_ordered",
                "context34": "ordered Plackett-Luce NLL plus 0.25 masked BCE",
                "context34_row_weight_multiplier": CONTEXT34_WEIGHT,
                "other_contexts": "normalized pointer NLL plus 0.25 masked BCE",
                "repository_exact_semantics": (
                    "selection and BCE are each weighted by sample_weight times "
                    "8 for context34; shared normalization makes this exactly the "
                    "per-row combined expression used here"
                ),
                "sample_weight_preserved": True,
                "count_loss": 0.0,
                "value_loss": 0.0,
            },
            "optimizer": {
                "name": "plain_SGD",
                "learning_rate": LEARNING_RATE,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "gradient_clip_l2": GRAD_CLIP_NORM,
            },
            "pcgrad": {
                "tasks": ["treatment", "retention"],
                "formula": (
                    "symmetric_two_task_PCGrad; independently project each task "
                    "against the other original gradient only when dot<0, then mean"
                ),
                "final_direction_dot_each_original_strictly_positive": True,
                "final_direction_cosine_each_original_strictly_above": PCGRAD_COSINE_MIN,
            },
            "step_l2_max": STEP_L2_MAX,
            "cumulative_l2_max": CUMULATIVE_L2_MAX,
            "device": args.device,
            "selected_batch_size": 256,
            "full_gate_batch_size": args.full_gate_batch_size,
        },
        "train_only_gates": {
            "each_step": {
                "treatment_loss_strictly_decreases": True,
                "current_retention_nll_delta_max": RETENTION_NLL_TOLERANCE,
                "cumulative_retention_nll_delta_from_source_max": RETENTION_NLL_TOLERANCE,
                "cumulative_retention_correct_to_wrong": 0,
                "previously_repaired_treatment_regressions": (
                    "0 across the frozen global union of all 301 treatment rows; "
                    "includes future-step rows repaired early"
                ),
            },
            "final": {
                "treatment_repairs_min": EXPECTED_REPAIRS,
                "repair_contexts_min": EXPECTED_REPAIR_CONTEXTS,
                "full_yanz_train_hybrid_correct_min": EXPECTED_YANZ_HYBRID,
                "full_yanz_source_correct_flips": 0,
                "full_old_train_hybrid_not_below_source": True,
                "raw17_and_deployment61_count_logits_bitwise_source": True,
                "value_logits_bitwise_source": True,
                "changed_parameter_names_exact_actor6": True,
                "frozen_nonactor_tensors_bitwise_source": True,
            },
        },
        "post_train_validation": {
            "selection_rule": (
                "only the unique step3 endpoint may be evaluated; no metric may "
                "select a step, learning rate, mixture, soup, or decoding rule"
            ),
            "deployment": {
                "checkpoint_form": "real expanded61 clone of the raw17 artifact",
                "prediction_order": "canonical except context34 skill order",
                "workers_0_and_2_integer_metrics_identical": True,
            },
            "offline_gates": {
                "old_valid_8116": {
                    "set_correct_min": 6774,
                    "ordered_correct_min": 6762,
                    "hybrid_correct_min": 6757,
                    "count_correct_exact": 8109,
                    "value_correct_exact": 6283,
                    "context34_set_and_hybrid_correct_exact": 14,
                },
                "old_test_3747": {
                    "set_correct_min": 3000,
                    "ordered_correct_min": 2996,
                    "hybrid_correct_min": 2992,
                    "count_correct_exact": 3741,
                    "value_correct_exact": 2548,
                    "context34_set_and_hybrid_correct_exact": 7,
                },
                "yanz_valid_525": {
                    "set_correct_min": 420,
                    "ordered_correct_min": 420,
                    "hybrid_correct_min": 420,
                    "count_correct_exact": 525,
                    "value_correct_exact": 411,
                },
                "consumed_timeforward_616": {
                    "set_correct_min": 473,
                    "ordered_correct_min": 473,
                    "hybrid_correct_min": 472,
                    "count_correct_exact": 616,
                    "value_correct_exact": 364,
                    "role": "development gate only, never untouched final",
                },
            },
            "head_to_head": {
                "opponent": "frozen source checkpoint with the exact same deck",
                "balanced_seats": True,
                "invalid_games_count_as_candidate_losses": True,
                "maximum_invalid_rate": 0.02,
                "screen": {
                    "games": 512,
                    "seed": 2026081052,
                    "minimum_conservative_win_rate": 0.5,
                },
                "formal": {
                    "games": 4096,
                    "seed": 2026081053,
                    "wilson_95_lower_bound_strictly_above": 0.5,
                    "minimum_candidate_wins_if_all_games_count": 2111,
                },
            },
            "untouched_future_blind": {
                "collection_starts_strictly_after_candidate_file_and_state_hashes_lock": True,
                "minimum_new_episodes": 8,
                "minimum_new_rows": 600,
                "episode_overlap_with_consumed_data": 0,
                "single_label_reveal": True,
                "set_ordered_hybrid_each_not_below_source": True,
                "at_least_one_action_metric_strict_gain_rows": 1,
                "count_and_value_exactly_source": True,
            },
        },
        "submission": {
            "authorized_opportunities": 1,
            "used_by_this_route": 0,
            "raw17_candidate_directly_submittable": False,
            "requires_separately_frozen_real61_materializer_and_package_audit": True,
            "submit_only_after_every_train_offline_h2h_blind_and_package_gate": True,
            "runner_performs_submission": False,
        },
        "outputs": {
            "candidate": str(args.candidate_output.resolve()),
            "audit": str(args.audit_output.resolve()),
            "failure": "write audit only; never write checkpoint",
        },
    }
    args.protocol.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(protocol)
    with args.protocol.open("xb") as handle:
        handle.write(payload)
    return {
        "protocol": str(args.protocol.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != SCHEMA:
        raise ProtocolError("protocol schema drift")
    if protocol.get("status") != "frozen_before_execution":
        raise ProtocolError("protocol status drift")
    bindings = protocol.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ProtocolError("protocol bindings missing")
    fixed_files = (
        ("source_checkpoint", profile_tool.SOURCE, profile_tool.SOURCE_SHA256),
        ("yanz_train_archive", profile_tool.YANZ, profile_tool.YANZ_SHA256),
        ("old_train_archive", profile_tool.OLD, profile_tool.OLD_SHA256),
    )
    for key, path, digest in fixed_files:
        entry = bindings.get(key)
        if not isinstance(entry, Mapping):
            raise ProtocolError(f"missing binding: {key}")
        if Path(str(entry.get("path"))).resolve() != path.resolve():
            raise ProtocolError(f"binding path drift: {key}")
        if entry.get("sha256") != digest or file_sha256(path) != digest:
            raise ProtocolError(f"binding hash drift: {key}")
    tool_paths = {
        "runner": RUNNER,
        "profiler": PROFILER,
        "train_bc_orbit": BC_TOOL,
        "train_ppo": PPO_TOOL,
        "deployment_evaluator": EVALUATOR,
        "head_to_head_evaluator": H2H_TOOL,
    }
    tools = bindings.get("tools")
    if not isinstance(tools, Mapping):
        raise ProtocolError("tool bindings missing")
    for key, path in tool_paths.items():
        entry = tools.get(key)
        if not isinstance(entry, Mapping):
            raise ProtocolError(f"tool binding missing: {key}")
        if Path(str(entry.get("path"))).resolve() != path.resolve():
            raise ProtocolError(f"tool path drift: {key}")
        if entry.get("sha256") != file_sha256(path):
            raise ProtocolError(f"tool hash drift: {key}")


def load_source(device: torch.device) -> tuple[dict[str, Any], bc.EntityOptionPolicy, dict[str, Any]]:
    checkpoint = torch.load(
        profile_tool.SOURCE, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ProtocolError("source checkpoint root drift")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ProtocolError("source model state missing")
    if profile_tool.bitwise_model_state_sha256(state) != profile_tool.SOURCE_BITWISE_STATE_SHA256:
        raise ProtocolError("source bitwise model-state hash drift")
    model, config = profile_tool.instantiate_raw_bc(checkpoint, device)
    if ppo.model_state_sha256(model) != profile_tool.SOURCE_PPO_STYLE_STATE_SHA256:
        raise ProtocolError("source train_ppo-style hash drift")
    return checkpoint, model, config


def featurize_descriptor_row(
    row: Mapping[str, Any], descriptor: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    feature = profile_tool.featurize(row, config)
    if feature is None:
        raise ProtocolError(f"selected row no longer featurizes: {descriptor['line_sha256']}")
    checks = {
        "visible_signature": profile_tool.feature_visible_signature(feature),
        "context": int(feature["context"]),
        "sample_weight": float(feature["sample_weight"]),
        "expert_action_order": [int(value) for value in row.get("action", [])],
        "episode_id": str(row.get("episode_id", "")),
        "episode_uuid": str(row.get("episode_uuid", "")),
        "team_name": str(row.get("team_name", "")),
    }
    for key, actual in checks.items():
        expected = descriptor.get(key)
        if actual != expected:
            raise ProtocolError(
                f"selected descriptor drift {descriptor['line_sha256']} {key}: "
                f"{actual!r} != {expected!r}"
            )
    feature["expert_action_order"] = checks["expert_action_order"]
    feature["line_sha256"] = str(descriptor["line_sha256"])
    feature["source"] = str(descriptor["source"])
    feature["category"] = str(descriptor["category"])
    return feature


def load_selected_features(
    profile: Mapping[str, Any], config: Mapping[str, Any]
) -> list[dict[str, list[dict[str, Any]]]]:
    descriptors: dict[str, Mapping[str, Any]] = {}
    for step in profile["steps"]:
        for row in step["treatment"] + step["retention"]:
            line_hash = str(row["line_sha256"])
            if line_hash in descriptors:
                raise ProtocolError("selected line hash repeats")
            descriptors[line_hash] = row
    found: dict[str, dict[str, Any]] = {}
    for source, archive_path in (("yanz", profile_tool.YANZ), ("old", profile_tool.OLD)):
        needed = {
            key for key, value in descriptors.items() if value.get("source") == source
        }
        for row, identity in profile_tool.raw_train_rows(archive_path):
            line_hash = str(identity["line_sha256"])
            if line_hash not in needed:
                continue
            descriptor = descriptors[line_hash]
            if (
                identity["archive_member"] != descriptor["archive_member"]
                or identity["member_line_number"] != descriptor["member_line_number"]
                or identity["raw_line_sha256"] != descriptor["raw_line_sha256"]
                or identity["line_sha256_algorithm"]
                != descriptor["line_sha256_algorithm"]
            ):
                raise ProtocolError(f"selected archive identity drift: {line_hash}")
            found[line_hash] = featurize_descriptor_row(row, descriptor, config)
    if set(found) != set(descriptors):
        missing = sorted(set(descriptors) - set(found))
        raise ProtocolError(f"selected rows missing from archives: {missing[:5]}")
    output = []
    for step in profile["steps"]:
        output.append(
            {
                "treatment": [found[row["line_sha256"]] for row in step["treatment"]],
                "retention": [found[row["line_sha256"]] for row in step["retention"]],
            }
        )
    return output


def collate_features(
    rows: Sequence[dict[str, Any]], config: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    cpu = profile_tool.collate(rows, config)
    return {name: value.to(device) for name, value in cpu.items()}


def ordered_nll_per_row(
    logits: torch.Tensor, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    mask = batch["option_mask"].bool()
    counts = batch["action_counts"]
    sequences = batch["action_sequences"]
    selected = torch.zeros_like(mask)
    result = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    for sequence_step in range(sequences.shape[1]):
        active = sequence_step < counts
        if not active.any():
            break
        chosen = sequences[:, sequence_step]
        active_rows = active.nonzero(as_tuple=False).squeeze(1)
        active_chosen = chosen[active]
        if not (mask[active_rows, active_chosen] & ~selected[active_rows, active_chosen]).all():
            raise ProtocolError("illegal ordered expert action")
        allowed = mask & ~selected
        log_probs = F.log_softmax(logits.masked_fill(~allowed, -1e9), dim=1)
        safe = chosen.clamp(0, logits.shape[1] - 1)
        chosen_logp = log_probs.gather(1, safe.unsqueeze(1)).squeeze(1)
        result -= torch.where(active, chosen_logp, torch.zeros_like(result))
        selected.scatter_(1, safe.unsqueeze(1), active.unsqueeze(1))
    return result


def hybrid_policy_nll_per_row(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row objective and effective row weights."""

    logits = outputs["policy_logits"].float()
    mask = batch["option_mask"].bool()
    targets = batch["targets"].float()
    counts = batch["action_counts"]
    log_probs = F.log_softmax(logits, dim=1)
    normalized = targets / counts.clamp_min(1).unsqueeze(1)
    pointer = -(normalized * log_probs).sum(dim=1)
    bce_raw = F.binary_cross_entropy_with_logits(
        logits.masked_fill(~mask, 0.0), targets, reduction="none"
    )
    bce = (bce_raw * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    ordered = ordered_nll_per_row(logits, batch)
    context34 = batch["contexts"] == ppo.SKILL_ORDER_CONTEXT
    selection = torch.where(context34, ordered, pointer)
    per_row = selection + SET_BCE_WEIGHT * bce
    weights = batch["sample_weights"].float() * torch.where(
        context34,
        torch.full_like(batch["sample_weights"].float(), CONTEXT34_WEIGHT),
        torch.ones_like(batch["sample_weights"].float()),
    )
    return per_row, weights


def weighted_policy_loss(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    per_row, weights = hybrid_policy_nll_per_row(outputs, batch)
    return (per_row * weights).sum() / weights.sum().clamp_min(1.0)


def configure_actor6(model: bc.EntityOptionPolicy) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    missing = sorted(set(ACTOR6) - set(named))
    if missing:
        raise ProtocolError(f"actor6 missing tensors: {missing}")
    for name in ACTOR6:
        named[name].requires_grad_(True)
    selected_names = [name for name, value in model.named_parameters() if value.requires_grad]
    parameters = [named[name] for name in ACTOR6]
    if tuple(selected_names) != ACTOR6:
        raise ProtocolError(f"actor6 scope/order drift: {selected_names}")
    if len(parameters) != 6 or sum(value.numel() for value in parameters) != 65793:
        raise ProtocolError("actor6 tensor/parameter count drift")
    return parameters


def task_gradient(
    model: bc.EntityOptionPolicy,
    parameters: Sequence[torch.nn.Parameter],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    outputs = ppo.model_forward(model, dict(batch), next(model.parameters()).device)
    loss = weighted_policy_loss(outputs, batch)
    values = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
        materialize_grads=False,
    )
    gradients = tuple(value.detach().float() for value in values)
    if not all(torch.isfinite(value).all() for value in gradients):
        raise ProtocolError("nonfinite actor6 task gradient")
    return loss.detach(), gradients


def gradient_dot(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> torch.Tensor:
    return sum(
        (a.detach().double().reshape(-1) * b.detach().double().reshape(-1)).sum()
        for a, b in zip(left, right)
    )


def gradient_norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(float(gradient_dot(values, values)), 0.0))


def symmetric_two_task_pcgrad(
    treatment: Sequence[torch.Tensor], retention: Sequence[torch.Tensor]
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    """Symmetric two-task PCGrad against the original task gradients."""

    dot = float(gradient_dot(treatment, retention))
    t_sq = float(gradient_dot(treatment, treatment))
    r_sq = float(gradient_dot(retention, retention))
    if t_sq <= 0.0 or r_sq <= 0.0:
        raise ProtocolError("zero task gradient")
    t_coefficient = dot / r_sq if dot < 0.0 else 0.0
    r_coefficient = dot / t_sq if dot < 0.0 else 0.0
    projected_t = tuple(
        value.double() - t_coefficient * reference.double()
        for value, reference in zip(treatment, retention)
    )
    projected_r = tuple(
        value.double() - r_coefficient * reference.double()
        for value, reference in zip(retention, treatment)
    )
    direction = tuple((left + right) * 0.5 for left, right in zip(projected_t, projected_r))
    direction_norm = gradient_norm(direction)
    effects: dict[str, Any] = {}
    for name, original in (("treatment", treatment), ("retention", retention)):
        original_norm = gradient_norm(original)
        effect_dot = float(gradient_dot(direction, original))
        cosine = effect_dot / max(direction_norm * original_norm, 1e-300)
        effects[name] = {
            "dot": effect_dot,
            "cosine": cosine,
            "pass": effect_dot > 0.0 and cosine > PCGRAD_COSINE_MIN,
        }
    return direction, {
        "original_dot": dot,
        "original_cosine": dot / math.sqrt(t_sq * r_sq),
        "projection_applied": dot < 0.0,
        "treatment_projection_coefficient": t_coefficient,
        "retention_projection_coefficient": r_coefficient,
        "direction_l2": direction_norm,
        "effects": effects,
        "pass": all(value["pass"] for value in effects.values()),
    }


def tensor_state_clone(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def actor_l2_from_state(
    state: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> float:
    named = dict(model.named_parameters())
    total = 0.0
    for name in ACTOR6:
        delta = named[name].detach().cpu().double() - state[name].double()
        total += float(delta.square().sum())
    return math.sqrt(total)


def apply_plain_sgd(
    parameters: Sequence[torch.nn.Parameter], direction: Sequence[torch.Tensor]
) -> dict[str, float]:
    optimizer = torch.optim.SGD(
        parameters,
        lr=LEARNING_RATE,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
    )
    optimizer.zero_grad(set_to_none=True)
    for parameter, value in zip(parameters, direction):
        parameter.grad = value.to(device=parameter.device, dtype=parameter.dtype).clone()
    preclip = float(torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM))
    postclip = math.sqrt(
        sum(float(parameter.grad.detach().double().square().sum()) for parameter in parameters)
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"preclip_gradient_l2": preclip, "postclip_gradient_l2": postclip}


def actual_descent_effects(
    pre_step_state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    treatment_gradient: Sequence[torch.Tensor],
    retention_gradient: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """Audit the FP32-cast, clipped, actually applied SGD displacement."""

    named = dict(model.named_parameters())
    actual_direction = tuple(
        (
            pre_step_state[name].to(
                device=named[name].device, dtype=named[name].dtype
            )
            - named[name].detach()
        ).float()
        / LEARNING_RATE
        for name in ACTOR6
    )
    actual_norm = gradient_norm(actual_direction)
    effects: dict[str, Any] = {}
    for label, original in (
        ("treatment", treatment_gradient),
        ("retention", retention_gradient),
    ):
        dot = float(gradient_dot(actual_direction, original))
        cosine = dot / max(actual_norm * gradient_norm(original), 1e-300)
        effects[label] = {
            "dot": dot,
            "cosine": cosine,
            "pass": dot > 0.0 and cosine > PCGRAD_COSINE_MIN,
        }
    return {
        "actual_descent_direction_l2": actual_norm,
        "effects": effects,
        "pass": all(value["pass"] for value in effects.values()),
    }


def batch_behavior(
    raw_model: bc.EntityOptionPolicy,
    deployment_model: bc.EntityOptionPolicy,
    batch: Mapping[str, torch.Tensor],
    row_ids: Sequence[str],
) -> dict[str, Any]:
    """Raw17 policy NLL plus correctness from the real expanded61 clone."""

    raw_model.eval()
    deployment_model.eval()
    with torch.no_grad():
        raw_outputs = ppo.model_forward(
            raw_model, dict(batch), next(raw_model.parameters()).device
        )
        deployment_outputs = ppo.model_forward(
            deployment_model,
            dict(batch),
            next(deployment_model.parameters()).device,
        )
        if deployment_outputs["count_logits"].shape[1] != 61:
            raise ProtocolError("step behavior did not use a real expanded61 clone")
        if not torch.equal(
            raw_outputs["policy_logits"], deployment_outputs["policy_logits"]
        ):
            raise ProtocolError("raw17/expanded61 policy logits differ on step gate")
        loss = float(weighted_policy_loss(raw_outputs, batch))
        actions = profile_tool.deterministic_policy_actions(
            deployment_outputs, batch
        )
    contexts = batch["contexts"].detach().cpu().tolist()
    sequences = batch["action_sequences"].detach().cpu()
    counts = batch["action_counts"].detach().cpu().tolist()
    correct: dict[str, bool] = {}
    for index, line_hash in enumerate(row_ids):
        expert = sequences[index, : int(counts[index])].tolist()
        predicted = actions[index]
        hybrid = predicted if int(contexts[index]) == 34 else sorted(predicted)
        correct[line_hash] = hybrid == expert
    return {"loss": loss, "correct": correct}


def sync_actor6_to_deployment(
    raw_model: bc.EntityOptionPolicy,
    deployment_model: bc.EntityOptionPolicy,
) -> None:
    raw_named = dict(raw_model.named_parameters())
    deployment_named = dict(deployment_model.named_parameters())
    with torch.no_grad():
        for name in ACTOR6:
            deployment_named[name].copy_(raw_named[name])
    if not all(
        torch.equal(raw_named[name], deployment_named[name]) for name in ACTOR6
    ):
        raise ProtocolError("failed to synchronize actor6 into expanded61 clone")


def concatenate_batches(
    rows: Sequence[dict[str, Any]], config: Mapping[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], list[str]]:
    return (
        collate_features(rows, config, device),
        [str(row["line_sha256"]) for row in rows],
    )


def run_replay(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    selected: Sequence[Mapping[str, Sequence[dict[str, Any]]]],
    device: torch.device,
) -> tuple[bc.EntityOptionPolicy, dict[str, Any]]:
    model, _ = profile_tool.instantiate_raw_bc(checkpoint, device)
    deployment_model = deployment_clone_from_raw(checkpoint, model, device)
    sync_actor6_to_deployment(model, deployment_model)
    parameters = configure_actor6(model)
    source_state = tensor_state_clone(model)
    treatment_seen: list[dict[str, Any]] = []
    retention_seen: list[dict[str, Any]] = []
    repaired_previous: set[str] = set()
    steps: list[dict[str, Any]] = []
    replay_pass = True

    # Freeze every source baseline before the first update.  Future-step rows
    # must never inherit behavior from an already-mutated earlier endpoint.
    all_treatment_rows = [
        row for step_rows in selected for row in step_rows["treatment"]
    ]
    all_retention_rows = [
        row for step_rows in selected for row in step_rows["retention"]
    ]
    all_t_batch, all_t_ids = concatenate_batches(
        all_treatment_rows, config, device
    )
    all_r_batch, all_r_ids = concatenate_batches(
        all_retention_rows, config, device
    )
    source_treatment = batch_behavior(
        model, deployment_model, all_t_batch, all_t_ids
    )
    source_retention = batch_behavior(
        model, deployment_model, all_r_batch, all_r_ids
    )
    source_treatment_correct = source_treatment["correct"]
    source_retention_correct = source_retention["correct"]
    if any(source_treatment_correct.values()):
        raise ProtocolError("treatment ledger contains a source-correct row")
    if not all(source_retention_correct.values()):
        raise ProtocolError("retention ledger contains a source-wrong row")
    source_cumulative_losses: dict[str, float] = {}
    growing_source_retention: list[dict[str, Any]] = []
    for future_index, future in enumerate(selected, start=1):
        growing_source_retention.extend(list(future["retention"]))
        source_batch, source_ids = concatenate_batches(
            growing_source_retention, config, device
        )
        source_cumulative_losses[str(future_index)] = batch_behavior(
            model, deployment_model, source_batch, source_ids
        )["loss"]

    for step_index, step_rows in enumerate(selected, start=1):
        treatment_rows = list(step_rows["treatment"])
        retention_rows = list(step_rows["retention"])
        treatment_seen.extend(treatment_rows)
        retention_seen.extend(retention_rows)
        t_batch, t_ids = concatenate_batches(treatment_rows, config, device)
        r_batch, r_ids = concatenate_batches(retention_rows, config, device)
        cumulative_t_batch, cumulative_t_ids = concatenate_batches(
            treatment_seen, config, device
        )
        cumulative_r_batch, cumulative_r_ids = concatenate_batches(
            retention_seen, config, device
        )
        before_t = batch_behavior(model, deployment_model, t_batch, t_ids)
        before_r = batch_behavior(model, deployment_model, r_batch, r_ids)
        before_ct = batch_behavior(
            model, deployment_model, cumulative_t_batch, cumulative_t_ids
        )
        before_cr = batch_behavior(
            model, deployment_model, cumulative_r_batch, cumulative_r_ids
        )
        if any(source_treatment_correct[key] for key in t_ids):
            raise ProtocolError("treatment contains a source-correct row")
        if not all(source_retention_correct[key] for key in r_ids):
            raise ProtocolError("retention contains a source-wrong row")

        treatment_loss, treatment_grad = task_gradient(model, parameters, t_batch)
        retention_loss, retention_grad = task_gradient(model, parameters, r_batch)
        direction, pcgrad = symmetric_two_task_pcgrad(treatment_grad, retention_grad)
        pre_step = tensor_state_clone(model)
        optimizer_audit: dict[str, float] | None = None
        if pcgrad["pass"]:
            optimizer_audit = apply_plain_sgd(parameters, direction)
        actual_direction = actual_descent_effects(
            pre_step, model, treatment_grad, retention_grad
        )
        sync_actor6_to_deployment(model, deployment_model)
        step_l2 = actor_l2_from_state(pre_step, model)
        cumulative_l2 = actor_l2_from_state(source_state, model)
        after_t = batch_behavior(model, deployment_model, t_batch, t_ids)
        after_r = batch_behavior(model, deployment_model, r_batch, r_ids)
        after_ct = batch_behavior(
            model, deployment_model, cumulative_t_batch, cumulative_t_ids
        )
        after_cr = batch_behavior(
            model, deployment_model, cumulative_r_batch, cumulative_r_ids
        )
        after_all_t = batch_behavior(
            model, deployment_model, all_t_batch, all_t_ids
        )

        current_retention_flips = [
            key for key in r_ids if before_r["correct"][key] and not after_r["correct"][key]
        ]
        cumulative_retention_flips = [
            key
            for key in cumulative_r_ids
            if source_retention_correct[key] and not after_cr["correct"][key]
        ]
        regressed_repairs = [
            key
            for key in repaired_previous
            if not after_all_t["correct"].get(key, False)
        ]
        repaired_now = {
            key
            for key in all_t_ids
            if not source_treatment_correct[key] and after_all_t["correct"][key]
        }
        gates = {
            "pcgrad_direction": bool(pcgrad["pass"]),
            "actual_cast_clip_sgd_direction": bool(actual_direction["pass"]),
            "treatment_loss_strict_decrease": after_t["loss"] < before_t["loss"],
            "current_retention_nll_delta": (
                after_r["loss"] - before_r["loss"] <= RETENTION_NLL_TOLERANCE
            ),
            "cumulative_retention_nll_delta_from_source": (
                after_cr["loss"] - source_cumulative_losses[str(step_index)]
                <= RETENTION_NLL_TOLERANCE
            ),
            "current_retention_zero_flips": not current_retention_flips,
            "cumulative_retention_zero_flips": not cumulative_retention_flips,
            "previously_repaired_treatment_zero_regressions": not regressed_repairs,
            "step_l2": step_l2 <= STEP_L2_MAX,
            "cumulative_l2": cumulative_l2 <= CUMULATIVE_L2_MAX,
        }

        step_pass = all(gates.values())
        report = {
            "step": step_index,
            "rows": {"treatment": len(t_ids), "retention": len(r_ids)},
            "losses": {
                "autograd_treatment": float(treatment_loss),
                "autograd_retention": float(retention_loss),
                "treatment_before": before_t["loss"],
                "treatment_after": after_t["loss"],
                "retention_before": before_r["loss"],
                "retention_after": after_r["loss"],
                "cumulative_treatment_before": before_ct["loss"],
                "cumulative_treatment_after": after_ct["loss"],
                "all301_treatment_after": after_all_t["loss"],
                "cumulative_retention_before": before_cr["loss"],
                "cumulative_retention_after": after_cr["loss"],
                "cumulative_retention_source": source_cumulative_losses[
                    str(step_index)
                ],
            },
            "pcgrad": pcgrad,
            "actual_cast_clip_sgd_direction": actual_direction,
            "optimizer": optimizer_audit,
            "movement": {"step_l2": step_l2, "cumulative_l2": cumulative_l2},
            "repairs": {
                "cumulative": len(repaired_now),
                "new": sorted(repaired_now - repaired_previous),
                "regressed_previous": sorted(regressed_repairs),
            },
            "retention_flips": {
                "current": sorted(current_retention_flips),
                "cumulative": sorted(cumulative_retention_flips),
            },
            "gates": gates,
            "pass": step_pass,
        }
        if step_index == 1:
            report["source_cumulative_retention_loss_by_step"] = (
                source_cumulative_losses
            )
        steps.append(report)
        replay_pass = replay_pass and step_pass
        repaired_previous = repaired_now
        if not step_pass:
            break

    state = model.state_dict()
    return model, {
        "steps": steps,
        "pass": replay_pass and len(steps) == 3,
        "terminal_model_state_sha256": profile_tool.bitwise_model_state_sha256(state),
        "terminal_train_ppo_model_state_sha256": ppo.model_state_sha256(model),
    }


def replay_comparable(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "steps": report["steps"],
        "pass": report["pass"],
        "terminal_model_state_sha256": report["terminal_model_state_sha256"],
        "terminal_train_ppo_model_state_sha256": report[
            "terminal_train_ppo_model_state_sha256"
        ],
    }


def iter_archive_feature_batches(
    path: Path,
    config: Mapping[str, Any],
    batch_size: int,
) -> Iterable[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    features: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row, identity in profile_tool.raw_train_rows(path):
        feature = profile_tool.featurize_deployment(row, config)
        if feature is None:
            continue
        feature["expert_action_order"] = [int(value) for value in row.get("action", [])]
        features.append(feature)
        metadata.append(
            {
                "line_sha256": identity["line_sha256"],
                "context": int(feature["context"]),
            }
        )
        if len(features) >= batch_size:
            yield features, metadata
            features, metadata = [], []
    if features:
        yield features, metadata


def paired_full_train_gate(
    source_raw: bc.EntityOptionPolicy,
    candidate_raw: bc.EntityOptionPolicy,
    source_deployment: bc.EntityOptionPolicy,
    candidate_deployment: bc.EntityOptionPolicy,
    path: Path,
    config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    repair_contexts: set[int] = set()
    source_raw.eval()
    candidate_raw.eval()
    source_deployment.eval()
    candidate_deployment.eval()
    with torch.no_grad():
        for rows, metadata in iter_archive_feature_batches(path, config, batch_size):
            batch = collate_features(rows, config, device)
            source_raw_outputs = ppo.model_forward(source_raw, batch, device)
            candidate_raw_outputs = ppo.model_forward(candidate_raw, batch, device)
            source_outputs = ppo.model_forward(source_deployment, batch, device)
            candidate_outputs = ppo.model_forward(candidate_deployment, batch, device)
            if source_outputs["count_logits"].shape[1] != 61:
                raise ProtocolError("source deployment clone is not 61-class")
            if candidate_outputs["count_logits"].shape[1] != 61:
                raise ProtocolError("candidate deployment clone is not 61-class")
            source_actions = profile_tool.deterministic_policy_actions(
                source_outputs, batch
            )
            candidate_actions = profile_tool.deterministic_policy_actions(
                candidate_outputs, batch
            )
            raw_count_exact = torch.equal(
                source_raw_outputs["count_logits"],
                candidate_raw_outputs["count_logits"],
            )
            deployment_count_exact = torch.equal(
                source_outputs["count_logits"], candidate_outputs["count_logits"]
            )
            raw_value_exact = torch.equal(
                source_raw_outputs["value_logits"],
                candidate_raw_outputs["value_logits"],
            )
            deployment_value_exact = torch.equal(
                source_outputs["value_logits"],
                candidate_outputs["value_logits"],
            )
            raw_deployment_policy_exact_source = torch.equal(
                source_raw_outputs["policy_logits"], source_outputs["policy_logits"]
            )
            raw_deployment_policy_exact_candidate = torch.equal(
                candidate_raw_outputs["policy_logits"],
                candidate_outputs["policy_logits"],
            )
            totals["raw17_count_logits_mismatch_batches"] += int(not raw_count_exact)
            totals["deployment61_count_logits_mismatch_batches"] += int(
                not deployment_count_exact
            )
            totals["raw_value_logits_mismatch_batches"] += int(not raw_value_exact)
            totals["deployment_value_logits_mismatch_batches"] += int(
                not deployment_value_exact
            )
            totals["source_raw_deployment_policy_mismatch_batches"] += int(
                not raw_deployment_policy_exact_source
            )
            totals["candidate_raw_deployment_policy_mismatch_batches"] += int(
                not raw_deployment_policy_exact_candidate
            )
            sequences = batch["action_sequences"].cpu()
            counts = batch["action_counts"].cpu().tolist()
            contexts = batch["contexts"].cpu().tolist()
            for index, meta in enumerate(metadata):
                expert = sequences[index, : int(counts[index])].tolist()
                context = int(contexts[index])
                source_hybrid = (
                    source_actions[index] if context == 34 else sorted(source_actions[index])
                ) == expert
                candidate_hybrid = (
                    candidate_actions[index]
                    if context == 34
                    else sorted(candidate_actions[index])
                ) == expert
                totals["rows"] += 1
                totals["source_hybrid_correct"] += int(source_hybrid)
                totals["candidate_hybrid_correct"] += int(candidate_hybrid)
                totals["source_correct_to_wrong"] += int(
                    source_hybrid and not candidate_hybrid
                )
                totals["source_wrong_to_correct"] += int(
                    not source_hybrid and candidate_hybrid
                )
                if not source_hybrid and candidate_hybrid:
                    repair_contexts.add(context)
    return {
        **dict(totals),
        "repair_contexts": sorted(repair_contexts),
        "repair_context_count": len(repair_contexts),
    }


def integrity_gate(
    source_state: Mapping[str, torch.Tensor], candidate: bc.EntityOptionPolicy
) -> dict[str, Any]:
    candidate_state = candidate.state_dict()
    changed = sorted(
        name
        for name in source_state
        if not torch.equal(source_state[name], candidate_state[name].detach().cpu())
    )
    nonactor_exact = all(
        torch.equal(source_state[name], candidate_state[name].detach().cpu())
        for name in source_state
        if name not in ACTOR6
    )
    return {
        "source_tensor_count": len(source_state),
        "candidate_tensor_count": len(candidate_state),
        "candidate_count_classes": int(candidate_state["count_head.2.weight"].shape[0]),
        "changed_parameter_names": changed,
        "changed_parameter_names_exact_actor6": changed == sorted(ACTOR6),
        "frozen_nonactor_tensors_bitwise_source": nonactor_exact,
        "all_candidate_tensors_finite": all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in candidate_state.values()
        ),
    }


def candidate_checkpoint(
    source_checkpoint: Mapping[str, Any],
    candidate: bc.EntityOptionPolicy,
    protocol_sha256: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(source_checkpoint))
    source_state = source_checkpoint["model_state_dict"]
    candidate_state = candidate.state_dict()
    state = {
        name: (
            candidate_state[name].detach().cpu().clone()
            if name in ACTOR6
            else source_state[name].detach().cpu().clone()
        )
        for name in source_state
    }
    if len(state) != 80 or int(state["count_head.2.weight"].shape[0]) != 17:
        raise ProtocolError("candidate checkpoint schema expansion detected")
    output["model_state_dict"] = state
    output["special_bc"] = {
        "schema_version": "ptcg-yanz-actor6-pcgrad-specialbc-checkpoint-v1",
        "parent_checkpoint_sha256": profile_tool.SOURCE_SHA256,
        "protocol_sha256": protocol_sha256,
        "steps": 3,
        "parameter_scope": "actor6",
        "model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
        "model_state_sha256": audit["candidate_model_state_sha256"],
        "raw17_training_artifact": True,
        "direct_submission_ready": False,
        "required_next_stage": (
            "separately materialize and verify a real61 PPO-compatible deployment package"
        ),
    }
    return output


def deployment_clone_from_raw(
    source_checkpoint: Mapping[str, Any],
    raw_model: bc.EntityOptionPolicy,
    device: torch.device,
) -> bc.EntityOptionPolicy:
    """Build the real 61-class live clone without changing checkpoint schema."""

    transient = dict(source_checkpoint)
    transient["model_state_dict"] = {
        name: value.detach().cpu().clone()
        for name, value in raw_model.state_dict().items()
    }
    if len(transient["model_state_dict"]) != 80:
        raise ProtocolError("raw deployment-clone source is not 80 tensors")
    if transient["model_state_dict"]["count_head.2.weight"].shape[0] != 17:
        raise ProtocolError("raw deployment-clone source is not 17-class")
    return ppo.instantiate_model_from_bc(transient, device)


def atomic_torch_save_new(value: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        if path.exists():
            raise FileExistsError(f"candidate appeared during save: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_audit_new(audit: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(audit))


def execute_protocol(
    protocol_path: Path, expected_protocol_sha256: str
) -> dict[str, Any]:
    actual_protocol_sha256 = file_sha256(protocol_path)
    if actual_protocol_sha256 != expected_protocol_sha256:
        raise ProtocolError(
            f"protocol SHA mismatch: {actual_protocol_sha256} != "
            f"{expected_protocol_sha256}"
        )
    protocol = load_json_strict(protocol_path)
    validate_protocol(protocol)
    profile_entry = protocol["bindings"]["profile"]
    profile_path = Path(profile_entry["path"])
    if file_sha256(profile_path) != profile_entry["sha256"]:
        raise ProtocolError("profile hash drift")
    profile = load_json_strict(profile_path)
    validate_profile(profile)
    outputs = protocol["outputs"]
    candidate_path = Path(outputs["candidate"])
    audit_path = Path(outputs["audit"])
    if candidate_path.exists() or audit_path.exists():
        raise FileExistsError("candidate/audit output must be absent at execute")
    device = torch.device(protocol["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("frozen CUDA execution requested but unavailable")
    source_checkpoint, source_model, config = load_source(device)
    source_deployment = deployment_clone_from_raw(
        source_checkpoint, source_model, device
    )
    if (
        ppo.model_state_sha256(source_deployment)
        != profile_tool.SOURCE_EXPANDED61_PPO_STYLE_STATE_SHA256
    ):
        raise ProtocolError("source expanded61 runtime state hash drift")
    selected = load_selected_features(profile, config)
    source_state = tensor_state_clone(source_model)

    candidate_a, replay_a = run_replay(
        source_checkpoint, config, selected, device
    )
    candidate_b, replay_b = run_replay(
        source_checkpoint, config, selected, device
    )
    replay_identical = canonical_json_bytes(replay_comparable(replay_a)) == canonical_json_bytes(
        replay_comparable(replay_b)
    )
    replay_gate = replay_identical and replay_a["pass"] and replay_b["pass"]
    integrity = integrity_gate(source_state, candidate_a)
    full_gates: dict[str, Any] | None = None
    final_gate_values: dict[str, bool] = {
        "double_ram_replay_identical": replay_identical,
        "both_replays_pass": replay_a["pass"] and replay_b["pass"],
        "artifact_integrity": all(
            value
            for key, value in integrity.items()
            if key
            in {
                "changed_parameter_names_exact_actor6",
                "frozen_nonactor_tensors_bitwise_source",
                "all_candidate_tensors_finite",
            }
        )
        and integrity["source_tensor_count"] == 80
        and integrity["candidate_tensor_count"] == 80
        and integrity["candidate_count_classes"] == 17,
    }
    if replay_gate and final_gate_values["artifact_integrity"]:
        batch_size = int(protocol["training"]["full_gate_batch_size"])
        candidate_deployment = deployment_clone_from_raw(
            source_checkpoint, candidate_a, device
        )
        yanz = paired_full_train_gate(
            source_model,
            candidate_a,
            source_deployment,
            candidate_deployment,
            profile_tool.YANZ,
            config,
            device,
            batch_size,
        )
        old = paired_full_train_gate(
            source_model,
            candidate_a,
            source_deployment,
            candidate_deployment,
            profile_tool.OLD,
            config,
            device,
            batch_size,
        )
        full_gates = {"yanz_train": yanz, "old_train": old}
        final_gate_values.update(
            {
                "yanz_rows_exact1725": yanz["rows"] == 1725,
                "old_rows_exact59875": old["rows"] == 59875,
                "yanz_source_hybrid_exact1422": (
                    yanz["source_hybrid_correct"] == 1422
                ),
                "yanz_hybrid_min1425": yanz["candidate_hybrid_correct"] >= 1425,
                "yanz_source_correct_zero_flips": yanz["source_correct_to_wrong"] == 0,
                "repairs_min3": yanz["source_wrong_to_correct"] >= 3,
                "repair_contexts_min2": yanz["repair_context_count"] >= 2,
                "old_hybrid_not_below_source": (
                    old["candidate_hybrid_correct"] >= old["source_hybrid_correct"]
                ),
                "raw17_count_logits_bitwise": (
                    yanz["raw17_count_logits_mismatch_batches"] == 0
                    and old["raw17_count_logits_mismatch_batches"] == 0
                ),
                "deployment61_count_logits_bitwise": (
                    yanz["deployment61_count_logits_mismatch_batches"] == 0
                    and old["deployment61_count_logits_mismatch_batches"] == 0
                ),
                "value_logits_bitwise": (
                    yanz["raw_value_logits_mismatch_batches"] == 0
                    and old["raw_value_logits_mismatch_batches"] == 0
                    and yanz["deployment_value_logits_mismatch_batches"] == 0
                    and old["deployment_value_logits_mismatch_batches"] == 0
                ),
                "raw_and_deployment_policy_logits_match": (
                    yanz["source_raw_deployment_policy_mismatch_batches"] == 0
                    and old["source_raw_deployment_policy_mismatch_batches"] == 0
                    and yanz["candidate_raw_deployment_policy_mismatch_batches"] == 0
                    and old["candidate_raw_deployment_policy_mismatch_batches"] == 0
                ),
            }
        )
    passed = all(final_gate_values.values())
    candidate_state_hash = profile_tool.bitwise_model_state_sha256(
        candidate_a.state_dict()
    )
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "accepted_train_only_candidate" if passed else "rejected_train_only",
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": actual_protocol_sha256,
        },
        "profile": {
            "path": str(profile_path.resolve()),
            "sha256": profile_entry["sha256"],
        },
        "replay_a": replay_a,
        "replay_b": replay_b,
        "replay_metrics_bitwise_canonical_json_identical": replay_identical,
        "integrity": integrity,
        "full_train_gates": full_gates,
        "final_gates": final_gate_values,
        "candidate_model_state_sha256_algorithm": "ptcg-model-state-bitwise-v1",
        "candidate_model_state_sha256": candidate_state_hash,
        "candidate_written": passed,
        "heldout_evaluation_performed": False,
        "submission_performed": False,
    }
    if passed:
        output_checkpoint = candidate_checkpoint(
            source_checkpoint, candidate_a, actual_protocol_sha256, audit
        )
        if profile_tool.bitwise_model_state_sha256(
            output_checkpoint["model_state_dict"]
        ) != candidate_state_hash:
            raise ProtocolError("17-class checkpoint state hash mismatch")
        atomic_torch_save_new(output_checkpoint, candidate_path)
        audit["candidate"] = {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
            "tensor_count": len(output_checkpoint["model_state_dict"]),
            "count_classes": int(
                output_checkpoint["model_state_dict"]["count_head.2.weight"].shape[0]
            ),
        }
    write_audit_new(audit, audit_path)
    return {
        "status": audit["status"],
        "audit": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "candidate": str(candidate_path) if passed else None,
        "candidate_sha256": file_sha256(candidate_path) if passed else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--profile", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--candidate-output", type=Path, required=True)
    freeze.add_argument("--audit-output", type=Path, required=True)
    freeze.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    freeze.add_argument("--full-gate-batch-size", type=int, default=256)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--protocol-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    if args.command == "freeze":
        if args.full_gate_batch_size < 1:
            raise ValueError("--full-gate-batch-size must be positive")
        result = freeze_protocol(args)
    else:
        if len(args.protocol_sha256) != 64:
            raise ValueError("--protocol-sha256 must be a SHA-256 hex digest")
        result = execute_protocol(args.protocol, args.protocol_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
