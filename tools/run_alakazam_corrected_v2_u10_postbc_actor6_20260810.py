#!/usr/bin/env python3
"""Apply a tiny, actor6-only BC repair to corrected-v2 Alakazam U10.

This is a frozen, two-phase protocol.  ``--audit-only`` binds the parent,
dependencies, and deterministic replay cache without taking an optimizer step
or writing a model checkpoint.  A training invocation must provide that cache
hash and emits only the S2 and S4 evaluation endpoints.

The PPO parent is never modified.  Every parameter except the six option
selection tensors is bitwise preserved, and output checkpoints deliberately
omit optimizer state and forbid training resume.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import stat
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    # Isolated mode intentionally omits the script directory from sys.path.
    # Add only this repository's frozen tools directory so sibling core imports
    # work under the required ``python -I -B`` invocation.
    sys.path.insert(0, str(TOOLS))

import torch

import run_ppo_bc_repair as repair
import train_ppo as ppo


EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PARENT = (
    ROOT
    / "artifacts/top3_bc77_20260809/alakazam_control/"
    "ppo_terminal01_corrected_v2_full/checkpoints/update-0010.pt"
)
PARENT_SHA256 = (
    "d20dbaa091186fd2f3a0b66b10bc11014d9d09c72fcf2992e17eb442864e1bca"
)
PARENT_UPDATE = 10
GENERAL_BC = (
    ROOT
    / "artifacts/gold8_recent7_20260808/alakazam_control/"
    "specialist_bc/best.pt"
)
GENERAL_BC_SHA256 = (
    "f5500086c16a02c19f3f2abce5e144446bd079248fd9fc3f4a620f3d079c7626"
)
REPLAY_ARCHIVE = ROOT / "data/gold8_recent7_20260808/archives/alakazam_control.zip"
REPLAY_ARCHIVE_SHA256 = (
    "4bd0de193cfb88b060435f84bbb5dc83a9eff3ab1aa4a498ef0859d2346f3c6c"
)
TRAIN_PPO = ROOT / "tools/train_ppo.py"
TRAIN_PPO_SHA256 = (
    "7abf34c073c14988e3ca5ef7fe069ae0fde447caa0551228ce8d9b5ae4750aa5"
)
EVALUATE_POLICY_BC = ROOT / "tools/evaluate_policy_bc.py"
EVALUATE_POLICY_BC_SHA256 = (
    "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
)
PARENT_BC_EVAL = (
    ROOT
    / "artifacts/top3_bc77_20260809/alakazam_control/"
    "ppo_terminal01_corrected_v2_full/bc_eval/update-0010.json"
)
PARENT_BC_EVAL_SHA256 = (
    "72a1cb488bd61eb6675e6d856058d3abb8511cc2c907c7e1a99fa4d152aed9ea"
)
LEARNER_DECK_HASH = (
    "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
)

SCHEMA_VERSION = "ptcg-alakazam-corrected-v2-u10-postbc-actor6-20260810-v1"
OUTPUT_ROOT = (
    ROOT
    / "artifacts/top3_bc77_20260809/alakazam_control/"
    "ppo_terminal01_corrected_v2_u10_postbc_actor6_lr4e7_s2s4_20260810_v1"
)
PREFLIGHT_ROOT = Path(str(OUTPUT_ROOT) + "_preflight")

ACTOR6 = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
ENDPOINTS = (2, 4)
REPAIR_STEPS = 4
REPAIR_SEED = 202608101
REPLAY_BATCHES = 72
BATCH_SIZE = 256
CONTEXT34_ROWS_PER_BATCH = 4
BATCH_INDICES = (11, 37, 5, 61)
LEARNING_RATE = 4e-7
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

# These are the corrected-v2 U10 full-valid counts, frozen by
# PARENT_BC_EVAL_SHA256.  S2/S4 may enter H2H only after a separate full-valid
# evaluation meets every floor.  Among eligible endpoints, rank only these BC
# counts and send at most one endpoint to H2H; never use H2H for preselection.
BC_PRESELECTION_FLOORS: dict[str, int] = {
    "rows": 8116,
    "count_correct": 8111,
    "set_exact_correct": 6323,
    "hybrid_order_exact_correct": 6305,
    "ordered_exact_correct": 6189,
    "context34_rows": 14,
    "context34_count_correct": 14,
    "context34_set_exact_correct": 14,
    "context34_hybrid_order_exact_correct": 9,
    "context34_ordered_exact_correct": 9,
}
BC_PRESELECTION_RANKING = (
    "ordered_exact_correct",
    "hybrid_order_exact_correct",
    "set_exact_correct",
    "count_correct",
    "context34_ordered_exact_correct",
    "context34_hybrid_order_exact_correct",
    "smaller_endpoint_step",
)

EXPECTED_PARENT_CONFIG: dict[str, Any] = {
    "bc_checkpoint": str(GENERAL_BC),
    "kl_reference_checkpoint": str(GENERAL_BC),
    "actor_reduction": "transition_mean",
    "advantage_normalization": "per_opponent",
    "gamma": 1.0,
    "gae_lambda": 1.0,
    "trainable_scope": "last_block_heads",
    "learning_rate": 3e-5,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": MAX_GRAD_NORM,
    "bc_replay_data": str(REPLAY_ARCHIVE),
    "bc_replay_split": "train",
    "bc_replay_batches": REPLAY_BATCHES,
    "bc_replay_batch_size": BATCH_SIZE,
    "bc_replay_workers": 8,
    "bc_replay_steps": 2,
    "bc_replay_lr_scale": 0.075,
    "bc_replay_loss": "ordered",
    "bc_replay_order_context_weight": 8.0,
    "bc_replay_context34_rows_per_batch": CONTEXT34_ROWS_PER_BATCH,
    "bc_replay_non_context34_fixed_multi_action_order_weight": 1.0,
}

PAYLOAD_KEYS = (
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "action_distribution",
)
OMITTED_OPTIMIZER_KEYS = (
    "optimizer_state_dict",
    "bc_replay_optimizer_state_dict",
    "optimizer_parameter_names",
    "opponent_quota_state",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish(path: Path, payload: bytes) -> None:
    """Write a new file without permitting replacement."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"Short write while publishing {path}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(path: Path, payload: dict[str, Any]) -> None:
    publish(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def require_regular_file(path: Path, expected_sha256: str, label: str) -> None:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"{label} is not a regular non-symlink file: {path}")
    actual = repair.file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )


def validate_parent_config(raw_config: Any) -> ppo.PPOConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("Parent checkpoint has no PPO config")
    for name, expected in EXPECTED_PARENT_CONFIG.items():
        actual = raw_config.get(name)
        if actual != expected:
            raise ValueError(
                f"Parent config {name!r} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    return ppo.PPOConfig(**raw_config)


def validate_parent(parent: Any) -> ppo.PPOConfig:
    if not isinstance(parent, dict):
        raise ValueError("Parent checkpoint payload is not a mapping")
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError("Parent is not a compatible terminal-01 PPO checkpoint")
    if int(parent.get("update", -1)) != PARENT_UPDATE:
        raise ValueError(
            f"Parent update must be {PARENT_UPDATE}, got {parent.get('update')!r}"
        )
    if parent.get("learner_deck_hash") != LEARNER_DECK_HASH:
        raise ValueError("Parent learner deck hash drifted")
    model_state = parent.get("model_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("Parent has no model_state_dict")
    return validate_parent_config(parent.get("config"))


def validate_request(audit_only: bool, expected_cache_sha256: str | None) -> None:
    if REPAIR_STEPS != max(ENDPOINTS) or ENDPOINTS != (2, 4):
        raise RuntimeError("Endpoint protocol drifted")
    if len(BATCH_INDICES) != REPAIR_STEPS:
        raise RuntimeError("Batch-index count does not match repair steps")
    if len(set(BATCH_INDICES)) != len(BATCH_INDICES):
        raise RuntimeError("Repair batch indices must be unique")
    if not all(0 <= index < REPLAY_BATCHES for index in BATCH_INDICES):
        raise RuntimeError("Repair batch index is outside the replay cache")
    if audit_only and expected_cache_sha256 is not None:
        raise ValueError("Audit-only discovers the cache hash; do not supply one")
    if not audit_only:
        if expected_cache_sha256 is None:
            raise ValueError("Training requires --expected-cache-sha256")
        if len(expected_cache_sha256) != 64:
            raise ValueError("Expected cache SHA256 must contain 64 hex characters")
        try:
            bytes.fromhex(expected_cache_sha256)
        except ValueError as error:
            raise ValueError("Expected cache SHA256 is not hexadecimal") from error


def require_exact_parent_model(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    loaded = repair.clone_model_state(model)
    changed = repair.changed_tensor_names(parent_state, loaded)
    if changed:
        raise RuntimeError(
            "Instantiated model does not exactly match PPO parent: "
            + json.dumps(changed)
        )
    return loaded


def configure_actor6(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    missing = sorted(set(ACTOR6) - set(named))
    if missing:
        raise RuntimeError("Actor6 parameter schema mismatch: " + json.dumps(missing))
    parameters: list[torch.nn.Parameter] = []
    for name in ACTOR6:
        named[name].requires_grad_(True)
        parameters.append(named[name])
    actual = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if actual != set(ACTOR6):
        raise RuntimeError(
            "Trainable parameter scope is not exactly actor6: "
            + json.dumps(sorted(actual))
        )
    if len({id(parameter) for parameter in parameters}) != len(ACTOR6):
        raise RuntimeError("Actor6 contains aliased parameter tensors")
    return parameters, named


def validate_endpoint_state(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> list[str]:
    if not repair.finite_nested(after):
        raise FloatingPointError("Post-BC endpoint contains non-finite tensors")
    changed = repair.changed_tensor_names(before, after)
    unexpected = sorted(set(changed) - set(ACTOR6))
    if unexpected:
        raise RuntimeError(
            "Post-BC changed tensors outside actor6: " + json.dumps(unexpected)
        )
    if set(changed) != set(ACTOR6):
        raise RuntimeError(
            "Post-BC did not change exactly actor6: " + json.dumps(changed)
        )
    return changed


def endpoint_payload(
    parent: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    step: int,
    endpoint_provenance: dict[str, Any],
) -> dict[str, Any]:
    if step not in ENDPOINTS:
        raise ValueError(f"Unsupported endpoint step: {step}")
    missing = [key for key in PAYLOAD_KEYS if key not in parent]
    if missing:
        raise ValueError("Parent lacks endpoint payload keys: " + json.dumps(missing))
    payload = {key: copy.deepcopy(parent[key]) for key in PAYLOAD_KEYS}
    payload.update(
        {
            "model_state_dict": copy.deepcopy(model_state),
            "update": PARENT_UPDATE,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": list(OMITTED_OPTIMIZER_KEYS),
            "post_ppo_special_bc": copy.deepcopy(endpoint_provenance),
        }
    )
    if any(key in payload for key in OMITTED_OPTIMIZER_KEYS):
        raise RuntimeError("Endpoint payload retained an optimizer state")
    return payload


def serialize_checkpoint(payload: dict[str, Any]) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    # A round trip catches accidental unserializable or lossy payload changes.
    loaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if repair.nested_sha256(loaded["model_state_dict"]) != repair.nested_sha256(
        payload["model_state_dict"]
    ):
        raise RuntimeError("Serialized endpoint model state failed round-trip")
    return raw, sha256_bytes(raw)


def build_replay_cache(
    config: ppo.PPOConfig,
    model_config: dict[str, Any],
) -> tuple[list[dict[str, torch.Tensor]], str, list[str]]:
    repair_config = copy.deepcopy(config)
    repair_config.bc_replay_data = str(REPLAY_ARCHIVE)
    repair_config.bc_replay_split = "train"
    repair_config.bc_replay_batches = REPLAY_BATCHES
    repair_config.bc_replay_batch_size = BATCH_SIZE
    repair_config.bc_replay_workers = EXPECTED_PARENT_CONFIG["bc_replay_workers"]
    repair_config.bc_replay_steps = 1
    repair_config.bc_replay_lr_scale = 1.0
    repair_config.bc_replay_loss = "ordered"
    repair_config.bc_replay_order_context_weight = 8.0
    repair_config.bc_replay_context34_rows_per_batch = CONTEXT34_ROWS_PER_BATCH
    repair_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    repair_config.max_grad_norm = MAX_GRAD_NORM
    repair_config.seed = REPAIR_SEED
    batches = ppo.build_bc_replay_batches(repair_config, model_config)
    if len(batches) != REPLAY_BATCHES:
        raise RuntimeError("Replay cache does not contain exactly 72 batches")
    if any(int(batch["contexts"].shape[0]) != BATCH_SIZE for batch in batches):
        raise RuntimeError("Replay cache contains a short batch")
    context_counts = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in batches
    ]
    if set(context_counts) != {CONTEXT34_ROWS_PER_BATCH}:
        raise RuntimeError("Replay cache context-34 quota drifted")
    cache_sha256, batch_sha256 = repair.replay_cache_manifest(batches)
    return batches, cache_sha256, batch_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-tool-sha256", required=True)
    parser.add_argument("--expected-cache-sha256")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_request(args.audit_only, args.expected_cache_sha256)
    tool = Path(__file__).resolve()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"Requires my_project_env Python: {EXPECTED_PYTHON}")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("Run with my_project_env Python -I -B")

    require_regular_file(tool, args.expected_tool_sha256, "post-BC tool")
    require_regular_file(TRAIN_PPO, TRAIN_PPO_SHA256, "train_ppo")
    require_regular_file(
        EVALUATE_POLICY_BC,
        EVALUATE_POLICY_BC_SHA256,
        "evaluate_policy_bc",
    )
    require_regular_file(PARENT, PARENT_SHA256, "PPO parent")
    require_regular_file(
        PARENT_BC_EVAL,
        PARENT_BC_EVAL_SHA256,
        "parent BC evaluation",
    )
    require_regular_file(GENERAL_BC, GENERAL_BC_SHA256, "general BC")
    require_regular_file(REPLAY_ARCHIVE, REPLAY_ARCHIVE_SHA256, "replay archive")

    target = PREFLIGHT_ROOT if args.audit_only else OUTPUT_ROOT
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to reuse output path: {target}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this frozen protocol")
    torch.use_deterministic_algorithms(True)
    random.seed(REPAIR_SEED)
    torch.manual_seed(REPAIR_SEED)
    torch.cuda.manual_seed_all(REPAIR_SEED)
    device = torch.device("cuda")

    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    config = validate_parent(parent)
    general_bc = torch.load(GENERAL_BC, map_location="cpu", weights_only=False)
    model = ppo.instantiate_model_from_checkpoint(parent, general_bc, device)
    before = require_exact_parent_model(model, parent["model_state_dict"])
    model_hash_before = ppo.model_state_sha256(model)
    actor_parameters, _ = configure_actor6(model)

    batches, cache_sha256, batch_sha256 = build_replay_cache(
        config,
        parent["model_config"],
    )
    if args.expected_cache_sha256 is not None and (
        cache_sha256 != args.expected_cache_sha256
    ):
        raise RuntimeError(
            "Replay cache SHA256 mismatch: "
            f"expected {args.expected_cache_sha256}, got {cache_sha256}"
        )

    common: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "audit_only" if args.audit_only else "post_bc",
        "parent": {
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": PARENT_SHA256,
            "update": PARENT_UPDATE,
            "model_state_sha256": model_hash_before,
        },
        "sources": {
            "general_bc": {
                "path": str(GENERAL_BC.relative_to(ROOT)),
                "sha256": GENERAL_BC_SHA256,
            },
            "replay_archive": {
                "path": str(REPLAY_ARCHIVE.relative_to(ROOT)),
                "sha256": REPLAY_ARCHIVE_SHA256,
                "split": "train",
            },
            "train_ppo": {
                "path": str(TRAIN_PPO.relative_to(ROOT)),
                "sha256": TRAIN_PPO_SHA256,
            },
            "bc_evaluator": {
                "path": str(EVALUATE_POLICY_BC.relative_to(ROOT)),
                "sha256": EVALUATE_POLICY_BC_SHA256,
            },
            "parent_bc_evaluation": {
                "path": str(PARENT_BC_EVAL.relative_to(ROOT)),
                "sha256": PARENT_BC_EVAL_SHA256,
            },
            "tool": {
                "path": str(tool.relative_to(ROOT)),
                "sha256": args.expected_tool_sha256,
            },
        },
        "protocol": {
            "seed": REPAIR_SEED,
            "steps": REPAIR_STEPS,
            "endpoints": list(ENDPOINTS),
            "batch_indices": list(BATCH_INDICES),
            "batch_sha256": [batch_sha256[index] for index in BATCH_INDICES],
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "fresh_adamw",
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "loss": "ordered",
            "context34_rows_per_batch": CONTEXT34_ROWS_PER_BATCH,
            "order_context_weight": 8.0,
            "trainable_parameter_names": list(ACTOR6),
        },
        "replay_cache": {
            "sha256": cache_sha256,
            "batches": REPLAY_BATCHES,
            "rows": REPLAY_BATCHES * BATCH_SIZE,
            "all_batches_have_exact_context34_quota": True,
        },
        "bc_only_preselection": {
            "status": "not_run_by_this_training_tool",
            "candidate_endpoint_steps": list(ENDPOINTS),
            "evaluation_data": str(REPLAY_ARCHIVE.relative_to(ROOT)),
            "evaluation_split": "valid",
            "evaluation_scope": "full_valid",
            "baseline_checkpoint_sha256": PARENT_SHA256,
            "baseline_evaluation_sha256": PARENT_BC_EVAL_SHA256,
            "eligibility_floors_inclusive": BC_PRESELECTION_FLOORS,
            "ranking_descending_except_last": list(BC_PRESELECTION_RANKING),
            "tie_break": "smaller_endpoint_step",
            "h2h_results_must_not_be_used_for_preselection": True,
            "maximum_candidates_entering_h2h": 1,
            "if_no_endpoint_meets_all_floors": "run_no_h2h",
        },
        "scope": {
            "local_only": True,
            "package": False,
            "upload": False,
            "submission": False,
            "validation_rows_used_for_training": 0,
        },
    }

    if args.audit_only:
        after = repair.clone_model_state(model)
        result = common | {
            "status": "audit_passed",
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "model_state_unchanged": repair.changed_tensor_names(before, after) == [],
            "parent_checkpoint_unchanged": repair.file_sha256(PARENT)
            == PARENT_SHA256,
        }
        if not result["model_state_unchanged"] or not result[
            "parent_checkpoint_unchanged"
        ]:
            raise RuntimeError("Audit-only mutated a frozen input")
        os.mkdir(target, mode=0o700)
        publish_json(target / "preflight_audit.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0

    repair_config = copy.deepcopy(config)
    repair_config.bc_replay_steps = 1
    repair_config.bc_replay_lr_scale = 1.0
    repair_config.bc_replay_loss = "ordered"
    repair_config.bc_replay_order_context_weight = 8.0
    repair_config.bc_replay_context34_rows_per_batch = CONTEXT34_ROWS_PER_BATCH
    repair_config.bc_replay_non_context34_fixed_multi_action_order_weight = 1.0
    repair_config.max_grad_norm = MAX_GRAD_NORM
    optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=LEARNING_RATE,
        eps=1e-5,
        weight_decay=WEIGHT_DECAY,
    )

    per_step: list[dict[str, Any]] = []
    endpoint_bytes: dict[int, bytes] = {}
    endpoint_records: dict[str, Any] = {}
    for step, batch_index in enumerate(BATCH_INDICES, start=1):
        metrics = ppo.bc_replay_update(
            model,
            optimizer,
            [batches[batch_index]],
            repair_config,
            device,
            LEARNING_RATE,
        )
        if not isinstance(metrics, dict):
            raise RuntimeError("BC replay core returned no metrics")
        if metrics.get("steps") != 1 or int(metrics.get("rows", -1)) != BATCH_SIZE:
            raise RuntimeError("BC replay core violated the one-step row contract")
        if int(metrics.get("context_34_rows", -1)) != CONTEXT34_ROWS_PER_BATCH:
            raise RuntimeError("BC replay core consumed a wrong context-34 quota")
        if float(metrics.get("learning_rate", -1.0)) != LEARNING_RATE:
            raise RuntimeError("BC replay core used an unexpected learning rate")
        if metrics.get("loss_mode") != "ordered" or not repair.finite_nested(metrics):
            raise RuntimeError("BC replay metrics violated the frozen loss contract")
        per_step.append(
            {
                "step": step,
                "batch_index": batch_index,
                "batch_sha256": batch_sha256[batch_index],
                "metrics": metrics,
            }
        )
        if step not in ENDPOINTS:
            continue

        state = repair.clone_model_state(model)
        changed = validate_endpoint_state(before, state)
        optimizer_steps = repair.optimizer_steps(optimizer.state_dict())
        if set(optimizer_steps) != {step} or len(optimizer_steps) != len(ACTOR6):
            raise RuntimeError("Fresh AdamW state does not match endpoint step")
        endpoint_provenance = {
            "schema_version": SCHEMA_VERSION,
            "parent_sha256": PARENT_SHA256,
            "parent_update": PARENT_UPDATE,
            "steps": step,
            "learning_rate": LEARNING_RATE,
            "optimizer": "fresh_adamw",
            "trainable_parameter_names": list(ACTOR6),
            "changed_parameter_names": changed,
            "value_count_and_transformer_unchanged": True,
            "evaluation_only": True,
            "resume_forbidden": True,
            "validation_rows_used_for_training": 0,
        }
        payload = endpoint_payload(parent, state, step, endpoint_provenance)
        raw, checkpoint_sha256 = serialize_checkpoint(payload)
        endpoint_bytes[step] = raw
        endpoint_records[str(step)] = {
            "checkpoint": f"special-bc-s{step:02d}.pt",
            "sha256": checkpoint_sha256,
            "model_state_sha256": repair.nested_sha256(state),
            "changed_parameter_names": changed,
            "optimizer_steps": step,
            "rows_consumed": step * BATCH_SIZE,
            "evaluation_only": True,
            "resume_forbidden": True,
        }

    final_state = repair.clone_model_state(model)
    final_changed = validate_endpoint_state(before, final_state)
    if set(endpoint_bytes) != set(ENDPOINTS):
        raise RuntimeError("Training did not materialize both frozen endpoints")
    if repair.file_sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError("PPO parent changed during post-BC")
    if repair.file_sha256(GENERAL_BC) != GENERAL_BC_SHA256:
        raise RuntimeError("General BC checkpoint changed during post-BC")
    if repair.file_sha256(REPLAY_ARCHIVE) != REPLAY_ARCHIVE_SHA256:
        raise RuntimeError("Replay archive changed during post-BC")

    result = common | {
        "status": "post_bc_completed",
        "per_step": per_step,
        "endpoints": endpoint_records,
        "integrity": {
            "optimizer_steps": REPAIR_STEPS,
            "checkpoint_writes": len(endpoint_bytes),
            "changed_exactly_actor6": set(final_changed) == set(ACTOR6),
            "changed_parameter_names": final_changed,
            "all_other_model_tensors_bitwise_unchanged": True,
            "optimizer_states_omitted": list(OMITTED_OPTIMIZER_KEYS),
            "parent_checkpoint_unchanged": True,
            "general_bc_checkpoint_unchanged": True,
            "replay_archive_unchanged": True,
            "all_metrics_and_model_tensors_finite": True,
            "validation_rows_used_for_training": 0,
        },
    }
    if result["integrity"]["checkpoint_writes"] != len(ENDPOINTS):
        raise RuntimeError("Checkpoint write count drifted")
    # Delay creating the immutable versioned directory until all computation,
    # serialization, and frozen-input checks have succeeded.  A failed run
    # therefore cannot strand an empty directory that blocks a clean retry.
    os.mkdir(target, mode=0o700)
    for step in ENDPOINTS:
        publish(target / f"special-bc-s{step:02d}.pt", endpoint_bytes[step])
    publish_json(target / "special_bc_manifest.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
