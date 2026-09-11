#!/usr/bin/env python3
"""Freeze, then execute, a deterministic actor6-only post-PPO BC tail repair.

The training sources are the frozen Raihan and kdcyberdude Marnie train-win
archives.  Their replay batches alternate for exactly eight optimizer steps.
The general Marnie archive is read only from ``valid`` and is never passed to
the optimizer.  Endpoints are emitted after steps 2, 4, and 8, with an explicit
general-valid promotion guard.

Freezing a manifest performs no optimizer step and creates no output directory.
Execution requires the saved manifest and its reviewed SHA-256, rebuilds all
three deterministic replay caches, and refuses any drift.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import torch  # noqa: E402
import run_ppo_bc_repair as audit  # noqa: E402
import train_ppo as ppo  # noqa: E402


SCHEMA_VERSION = "ptcg-gold-push-postppo-tail-repair-v1"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAINER = TOOLS / "train_ppo.py"
MARNIE_DECK_HASH = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)

ARCHIVE_ROOT = ROOT / "data/gold_push_recent7_20260810_v1/archives"
RAIHAN_ARCHIVE = ARCHIVE_ROOT / "marnie_raihan_trainwins.zip"
RAIHAN_SHA256 = (
    "36c8b407eb27890cbb00f22a7b5df8f30bdba9b970aa136017bf405603f890f3"
)
KD_ARCHIVE = ARCHIVE_ROOT / "marnie_kdcyberdude_trainwins.zip"
KD_SHA256 = (
    "24fd53320193ec299282b5dd8e609486d565c35fa1ee7b7febe698f872c360c5"
)
GENERAL_ARCHIVE = ARCHIVE_ROOT / "marnie_trainwins.zip"
GENERAL_SHA256 = (
    "7cc2a4cb38b857ccdacf9cc84fe4610400ca3295c9ab1dbefbcb75e4341e8eab"
)

SEED = 202608210
TRAIN_BATCHES_PER_SOURCE = 32
VALID_BATCHES = 16
BATCH_SIZE = 256
TRAIN_CONTEXT34_ROWS_PER_BATCH = 1
VALID_CONTEXT34_ROWS_PER_BATCH = 4
ORDER_CONTEXT_WEIGHT = 8.0
RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT = 2.0
LEARNING_RATE = 4e-7
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 1.0
ENDPOINTS = (2, 4, 8)
GENERAL_VALID_MAX_ABSOLUTE_LOSS_INCREASE = 0.002
GENERAL_VALID_MAX_RAW_MULTI_ORDERED_LOSS_INCREASE = 0.01
GENERAL_VALID_MAX_CONTEXT34_ORDERED_LOSS_INCREASE = 0.01

ACTOR6 = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
RAIHAN_BATCH_INDICES = (7, 19, 3, 27)
KD_BATCH_INDICES = (11, 5, 23, 1)
STEP_SCHEDULE = tuple(
    entry
    for pair in zip(
        (("raihan", index) for index in RAIHAN_BATCH_INDICES),
        (("kdcyberdude", index) for index in KD_BATCH_INDICES),
    )
    for entry in pair
)


@dataclass
class PreparedRun:
    manifest: dict[str, Any]
    manifest_sha256: str
    parent: dict[str, Any]
    bc_checkpoint: dict[str, Any]
    configs: dict[str, argparse.Namespace]
    caches: dict[str, list[dict[str, torch.Tensor]]]
    batch_sha256: dict[str, list[str]]
    parent_path: Path
    bc_path: Path


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


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def validate_sha256_text(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def require_regular_file(path: Path, expected_sha256: str, label: str) -> str:
    validate_sha256_text(expected_sha256, f"{label} SHA-256")
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file: {path}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return observed


def require_unhashed_regular_file(path: Path, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file: {path}")


def assert_output_absent(output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    parent = output_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise FileNotFoundError(
            f"Output parent must already be a non-symlink directory: {parent}"
        )


def write_exclusive_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"Short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_json(path: Path, payload: Any) -> None:
    write_exclusive_bytes(path, canonical_json_bytes(payload))


def validate_protocol_constants() -> None:
    if ENDPOINTS != (2, 4, 8):
        raise RuntimeError("Tail-repair endpoints drifted")
    if len(STEP_SCHEDULE) != 8:
        raise RuntimeError("Tail-repair schedule must contain exactly eight steps")
    if tuple(source for source, _ in STEP_SCHEDULE) != (
        "raihan",
        "kdcyberdude",
    ) * 4:
        raise RuntimeError("Tail-repair sources must alternate Raihan/KD")
    for source, indices in (
        ("raihan", RAIHAN_BATCH_INDICES),
        ("kdcyberdude", KD_BATCH_INDICES),
    ):
        if len(indices) != 4 or len(set(indices)) != 4:
            raise RuntimeError(f"{source} batch indices must be unique")
        if not all(0 <= index < TRAIN_BATCHES_PER_SOURCE for index in indices):
            raise RuntimeError(f"{source} batch index outside the frozen cache")
    if set(ACTOR6) & {
        name for name in ACTOR6 if name.startswith(("count_head.", "value_head."))
    }:
        raise RuntimeError("actor6 scope may not include count/value tensors")


def replay_config(
    parent_config: dict[str, Any],
    *,
    archive: Path,
    split: str,
    batches: int,
    context34_rows_per_batch: int,
    seed: int,
) -> argparse.Namespace:
    # Replay construction and expert loss consume only the attributes frozen
    # below.  A Namespace deliberately keeps this launcher compatible with
    # older PPO parents whose serialized config predates unrelated PPO fields.
    config = argparse.Namespace(**copy.deepcopy(parent_config))
    config.bc_replay_data = str(archive.resolve())
    config.bc_replay_split = split
    config.bc_replay_batches = batches
    config.bc_replay_batch_size = BATCH_SIZE
    config.bc_replay_workers = 1
    config.bc_replay_steps = 1
    config.bc_replay_lr_scale = 1.0
    config.bc_replay_loss = "ordered"
    config.bc_replay_order_context_weight = ORDER_CONTEXT_WEIGHT
    config.bc_replay_non_context34_fixed_multi_action_order_weight = (
        RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT
    )
    config.bc_replay_context34_rows_per_batch = context34_rows_per_batch
    config.max_grad_norm = MAX_GRAD_NORM
    config.seed = seed
    return config


def cache_profile(
    batches: list[dict[str, torch.Tensor]],
    *,
    expected_batches: int,
    expected_context34_rows_per_batch: int,
) -> dict[str, Any]:
    if len(batches) != expected_batches:
        raise RuntimeError(
            f"Replay cache batch count mismatch: {len(batches)} != {expected_batches}"
        )
    row_counts = [int(batch["action_counts"].shape[0]) for batch in batches]
    if set(row_counts) != {BATCH_SIZE}:
        raise RuntimeError("Replay cache contains a partial batch")
    context_counts = [
        int((batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum())
        for batch in batches
    ]
    if set(context_counts) != {expected_context34_rows_per_batch}:
        raise RuntimeError("Replay cache context-34 quota drifted")
    fixed_multi_rows = sum(
        int(
            (
                (batch["contexts"] != ppo.SKILL_ORDER_CONTEXT)
                & (batch["min_counts"] == batch["max_counts"])
                & (batch["action_counts"] > 1)
            ).sum()
        )
        for batch in batches
    )
    if fixed_multi_rows <= 0:
        raise RuntimeError("Replay cache has no ordinary fixed multi-action rows")
    return {
        "batches": len(batches),
        "batch_size": BATCH_SIZE,
        "rows": sum(row_counts),
        "context34_rows": sum(context_counts),
        "context34_rows_per_batch": expected_context34_rows_per_batch,
        "non_context34_fixed_multi_action_rows": fixed_multi_rows,
    }


def configure_actor6(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    missing = sorted(set(ACTOR6) - set(named))
    if missing:
        raise RuntimeError(f"Parent model lacks actor6 tensors: {missing}")
    parameters: list[torch.nn.Parameter] = []
    for name in ACTOR6:
        named[name].requires_grad_(True)
        parameters.append(named[name])
    trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if trainable != ACTOR6:
        raise RuntimeError(f"Trainable scope drifted from actor6: {trainable}")
    return parameters


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def validate_endpoint_state(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> dict[str, Any]:
    changed = audit.changed_tensor_names(before, after)
    if changed != sorted(ACTOR6):
        raise RuntimeError(
            "Tail repair changed tensors outside actor6 or left actor6 untouched: "
            f"{changed}"
        )
    count_names = sorted(name for name in before if name.startswith("count_head."))
    value_names = sorted(name for name in before if name.startswith("value_head."))
    if not count_names or not value_names:
        raise RuntimeError("Model lacks count/value integrity tensors")
    if any(not torch.equal(before[name], after[name]) for name in count_names):
        raise RuntimeError("Count head changed during actor6-only repair")
    if any(not torch.equal(before[name], after[name]) for name in value_names):
        raise RuntimeError("Value head changed during actor6-only repair")
    if not audit.finite_nested(after):
        raise FloatingPointError("Non-finite endpoint model state")
    actor_l2 = math.sqrt(
        sum(
            float((after[name].double() - before[name].double()).square().sum())
            for name in ACTOR6
        )
    )
    return {
        "changed_parameter_names": changed,
        "changed_exactly_actor6": True,
        "count_head_bit_identical": True,
        "value_head_bit_identical": True,
        "all_non_actor6_tensors_bit_identical": True,
        "actor6_l2_from_parent": actor_l2,
        "all_finite": True,
    }


def build_manifest_payload(
    *,
    parent_path: Path,
    parent_sha256: str,
    parent: dict[str, Any],
    bc_path: Path,
    bc_sha256: str,
    output_dir: Path,
    cache_records: dict[str, dict[str, Any]],
    actor_shapes: dict[str, list[int]],
    tool_sha256: str,
    trainer_sha256: str,
    python_sha256: str,
    device: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "frozen_before_training",
        "inputs": {
            "parent_checkpoint": {
                "path": display_path(parent_path),
                "sha256": parent_sha256,
                "feature_version": parent.get("feature_version"),
                "update": int(parent.get("update", -1)),
                "model_state_sha256": audit.nested_sha256(
                    parent["model_state_dict"]
                ),
            },
            "bc_architecture_checkpoint": {
                "path": display_path(bc_path),
                "sha256": bc_sha256,
            },
            "raihan_train": {
                "path": display_path(RAIHAN_ARCHIVE),
                "sha256": RAIHAN_SHA256,
                "split": "train",
                "used_for_optimizer_steps": True,
            },
            "kdcyberdude_train": {
                "path": display_path(KD_ARCHIVE),
                "sha256": KD_SHA256,
                "split": "train",
                "used_for_optimizer_steps": True,
            },
            "general_valid": {
                "path": display_path(GENERAL_ARCHIVE),
                "sha256": GENERAL_SHA256,
                "split": "valid",
                "used_for_optimizer_steps": False,
            },
            "launcher": {
                "path": display_path(Path(__file__).resolve()),
                "sha256": tool_sha256,
            },
            "trainer_module": {
                "path": display_path(TRAINER),
                "sha256": trainer_sha256,
            },
            "python": {
                "path": str(EXPECTED_PYTHON.resolve()),
                "sha256": python_sha256,
            },
        },
        "lineage": {
            "learner_deck_hash": parent.get("learner_deck_hash"),
            "required_learner_deck_hash": MARNIE_DECK_HASH,
            "parent_config_sha256": sha256_json(parent.get("config")),
            "parent_model_config_sha256": sha256_json(parent.get("model_config")),
        },
        "runtime": {
            "python": str(EXPECTED_PYTHON.resolve()),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "deterministic_algorithms_required": True,
            "cublas_workspace_config": ":4096:8",
        },
        "protocol": {
            "seed": SEED,
            "optimizer": "fresh_adamw",
            "optimizer_state_from_parent": False,
            "steps": len(STEP_SCHEDULE),
            "endpoints": list(ENDPOINTS),
            "step_schedule": [
                {"step": step, "source": source, "batch_index": index}
                for step, (source, index) in enumerate(STEP_SCHEDULE, 1)
            ],
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "adam_eps": ADAM_EPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "trainable_scope": "actor6",
            "trainable_parameter_names": list(ACTOR6),
            "trainable_parameter_shapes": actor_shapes,
            "loss": "ordered",
            "context34_order_weight": ORDER_CONTEXT_WEIGHT,
            "non_context34_fixed_multi_action_order_weight": (
                RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT
            ),
            "batch_size": BATCH_SIZE,
            "train_batches_per_source": TRAIN_BATCHES_PER_SOURCE,
            "general_valid_batches": VALID_BATCHES,
            "train_context34_rows_per_batch": TRAIN_CONTEXT34_ROWS_PER_BATCH,
            "general_valid_context34_rows_per_batch": (
                VALID_CONTEXT34_ROWS_PER_BATCH
            ),
            "device": device,
        },
        "replay_caches": cache_records,
        "guards": {
            "general_valid_read_only": True,
            "general_valid_rows_used_for_training": 0,
            "general_valid_max_absolute_loss_increase": (
                GENERAL_VALID_MAX_ABSOLUTE_LOSS_INCREASE
            ),
            "general_valid_max_raw_multi_ordered_loss_increase": (
                GENERAL_VALID_MAX_RAW_MULTI_ORDERED_LOSS_INCREASE
            ),
            "general_valid_max_context34_ordered_loss_increase": (
                GENERAL_VALID_MAX_CONTEXT34_ORDERED_LOSS_INCREASE
            ),
            "specialist_mean_loss_must_improve": True,
            "each_specialist_loss_must_improve": True,
            "each_specialist_raw_multi_ordered_loss_must_improve": True,
            "count_head_must_be_bit_identical": True,
            "value_head_must_be_bit_identical": True,
            "all_non_actor6_tensors_must_be_bit_identical": True,
            "failed_endpoint_is_not_promotion_eligible": True,
        },
        "output": {
            "directory": str(output_dir.resolve()),
            "must_not_exist_before_execute": True,
            "checkpoint_steps": list(ENDPOINTS),
            "evaluation_only": True,
            "resume_forbidden": True,
        },
        "scope": {
            "training": True,
            "local_only": True,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }


def prepare_run(
    *,
    parent_path: Path,
    expected_parent_sha256: str,
    bc_path: Path,
    expected_bc_sha256: str,
    output_dir: Path,
    device: str,
) -> PreparedRun:
    validate_protocol_constants()
    assert_output_absent(output_dir)
    parent_path = absolute_path(parent_path)
    bc_path = absolute_path(bc_path)
    require_regular_file(parent_path, expected_parent_sha256, "parent checkpoint")
    require_regular_file(bc_path, expected_bc_sha256, "BC architecture checkpoint")
    require_regular_file(RAIHAN_ARCHIVE, RAIHAN_SHA256, "Raihan replay archive")
    require_regular_file(KD_ARCHIVE, KD_SHA256, "KD replay archive")
    require_regular_file(GENERAL_ARCHIVE, GENERAL_SHA256, "general replay archive")
    tool_sha256 = file_sha256(Path(__file__).resolve())
    trainer_sha256 = file_sha256(TRAINER)
    python_sha256 = file_sha256(EXPECTED_PYTHON.resolve())

    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(bc_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, dict) or not isinstance(bc_checkpoint, dict):
        raise TypeError("Parent and BC checkpoints must contain dictionaries")
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise RuntimeError("Parent is not a compatible PPO checkpoint")
    if parent.get("learner_deck_hash") != MARNIE_DECK_HASH:
        raise RuntimeError("Parent is not bound to the frozen Marnie deck")
    if not isinstance(parent.get("config"), dict):
        raise RuntimeError("Parent checkpoint lacks PPO config")
    if not isinstance(parent.get("model_config"), dict):
        raise RuntimeError("Parent checkpoint lacks model_config")
    if not isinstance(parent.get("model_state_dict"), dict):
        raise RuntimeError("Parent checkpoint lacks model_state_dict")
    if bc_checkpoint.get("feature_version") != ppo.BC_FEATURE_VERSION:
        raise RuntimeError("Architecture checkpoint is not a compatible BC checkpoint")
    bc_config = bc_checkpoint.get("config")
    if not isinstance(bc_config, dict):
        raise RuntimeError("Architecture checkpoint lacks BC config")
    architecture_mismatches = {
        name: {"parent": expected, "bc": bc_config.get(name)}
        for name, expected in parent["model_config"].items()
        if bc_config.get(name) != expected
    }
    if architecture_mismatches:
        raise RuntimeError(
            "Parent and BC architecture settings differ: "
            f"{architecture_mismatches}"
        )

    verification_model = ppo.instantiate_model_from_checkpoint(
        parent,
        bc_checkpoint,
        torch.device("cpu"),
    )
    configure_actor6(verification_model)
    actor_shapes = {
        name: list(dict(verification_model.named_parameters())[name].shape)
        for name in ACTOR6
    }
    del verification_model

    configs = {
        "raihan": replay_config(
            parent["config"],
            archive=RAIHAN_ARCHIVE,
            split="train",
            batches=TRAIN_BATCHES_PER_SOURCE,
            context34_rows_per_batch=TRAIN_CONTEXT34_ROWS_PER_BATCH,
            seed=SEED + 101,
        ),
        "kdcyberdude": replay_config(
            parent["config"],
            archive=KD_ARCHIVE,
            split="train",
            batches=TRAIN_BATCHES_PER_SOURCE,
            context34_rows_per_batch=TRAIN_CONTEXT34_ROWS_PER_BATCH,
            seed=SEED + 202,
        ),
        "general_valid": replay_config(
            parent["config"],
            archive=GENERAL_ARCHIVE,
            split="valid",
            batches=VALID_BATCHES,
            context34_rows_per_batch=VALID_CONTEXT34_ROWS_PER_BATCH,
            seed=SEED + 303,
        ),
    }
    caches = {
        name: ppo.build_bc_replay_batches(config, parent["model_config"])
        for name, config in configs.items()
    }
    profiles = {
        "raihan": cache_profile(
            caches["raihan"],
            expected_batches=TRAIN_BATCHES_PER_SOURCE,
            expected_context34_rows_per_batch=TRAIN_CONTEXT34_ROWS_PER_BATCH,
        ),
        "kdcyberdude": cache_profile(
            caches["kdcyberdude"],
            expected_batches=TRAIN_BATCHES_PER_SOURCE,
            expected_context34_rows_per_batch=TRAIN_CONTEXT34_ROWS_PER_BATCH,
        ),
        "general_valid": cache_profile(
            caches["general_valid"],
            expected_batches=VALID_BATCHES,
            expected_context34_rows_per_batch=VALID_CONTEXT34_ROWS_PER_BATCH,
        ),
    }
    batch_sha256: dict[str, list[str]] = {}
    cache_records: dict[str, dict[str, Any]] = {}
    for name, batches in caches.items():
        cache_sha256, per_batch = audit.replay_cache_manifest(batches)
        batch_sha256[name] = per_batch
        cache_records[name] = {
            **profiles[name],
            "sha256": cache_sha256,
            "batch_sha256": per_batch,
            "split": "valid" if name == "general_valid" else "train",
            "used_for_optimizer_steps": name != "general_valid",
        }
    scheduled_indices = {
        "raihan": list(RAIHAN_BATCH_INDICES),
        "kdcyberdude": list(KD_BATCH_INDICES),
    }
    for source, indices in scheduled_indices.items():
        scheduled_fixed_multi_rows: list[int] = []
        for index in indices:
            batch = caches[source][index]
            rows = int(
                (
                    (batch["contexts"] != ppo.SKILL_ORDER_CONTEXT)
                    & (batch["min_counts"] == batch["max_counts"])
                    & (batch["action_counts"] > 1)
                ).sum()
            )
            if rows <= 0:
                raise RuntimeError(
                    f"Scheduled {source} replay batch {index} has no raw "
                    "fixed multi-action supervision"
                )
            scheduled_fixed_multi_rows.append(rows)
        cache_records[source]["scheduled_batch_indices"] = indices
        cache_records[source]["scheduled_batch_sha256"] = [
            batch_sha256[source][index] for index in indices
        ]
        cache_records[source][
            "scheduled_non_context34_fixed_multi_action_rows"
        ] = scheduled_fixed_multi_rows

    manifest = build_manifest_payload(
        parent_path=parent_path,
        parent_sha256=expected_parent_sha256,
        parent=parent,
        bc_path=bc_path,
        bc_sha256=expected_bc_sha256,
        output_dir=output_dir,
        cache_records=cache_records,
        actor_shapes=actor_shapes,
        tool_sha256=tool_sha256,
        trainer_sha256=trainer_sha256,
        python_sha256=python_sha256,
        device=device,
    )
    return PreparedRun(
        manifest=manifest,
        manifest_sha256=sha256_json(manifest),
        parent=parent,
        bc_checkpoint=bc_checkpoint,
        configs=configs,
        caches=caches,
        batch_sha256=batch_sha256,
        parent_path=parent_path,
        bc_path=bc_path,
    )


def frozen_manifest_envelope(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-frozen-manifest-envelope",
        "manifest_sha256": sha256_json(manifest),
        "manifest": manifest,
    }


def load_frozen_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    require_unhashed_regular_file(path, "frozen manifest")
    validate_sha256_text(expected_sha256, "expected manifest SHA-256")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("manifest"), dict):
        raise RuntimeError("Frozen manifest envelope is malformed")
    manifest = raw["manifest"]
    observed = sha256_json(manifest)
    if raw.get("manifest_sha256") != observed:
        raise RuntimeError("Frozen manifest envelope has an invalid embedded digest")
    if observed != expected_sha256:
        raise RuntimeError(
            f"Frozen manifest SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    expected_envelope_schema = SCHEMA_VERSION + "-frozen-manifest-envelope"
    if raw.get("schema_version") != expected_envelope_schema:
        raise RuntimeError("Frozen manifest envelope schema mismatch")
    return manifest


@torch.inference_mode()
def cache_loss(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    config: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    rows = 0
    for cpu_batch in batches:
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in cpu_batch.items()
        }
        outputs = ppo.model_forward(model, batch, device)
        loss, parts = ppo.bc_expert_actor_loss(
            outputs,
            batch,
            loss_mode=config.bc_replay_loss,
            order_context_weight=config.bc_replay_order_context_weight,
            non_context34_fixed_multi_action_order_weight=(
                config.bc_replay_non_context34_fixed_multi_action_order_weight
            ),
        )
        batch_rows = int(batch["action_counts"].shape[0])
        rows += batch_rows
        totals["loss"] = totals.get("loss", 0.0) + float(loss) * batch_rows
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_rows
    result = {"rows": rows} | {
        name: total / max(rows, 1) for name, total in totals.items()
    }
    if rows <= 0 or not audit.finite_nested(result):
        raise FloatingPointError("Invalid replay cache evaluation metrics")
    return result


def actor6_optimizer_step(
    *,
    model: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    cpu_batch: dict[str, torch.Tensor],
    config: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batch = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in cpu_batch.items()
    }
    model.zero_grad(set_to_none=True)
    outputs = ppo.model_forward(model, batch, device)
    loss, parts = ppo.bc_expert_actor_loss(
        outputs,
        batch,
        loss_mode=config.bc_replay_loss,
        order_context_weight=config.bc_replay_order_context_weight,
        non_context34_fixed_multi_action_order_weight=(
            config.bc_replay_non_context34_fixed_multi_action_order_weight
        ),
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)
    optimizer.step()
    result = {
        "rows": int(batch["action_counts"].shape[0]),
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "context34_rows": int(
            (batch["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()
        ),
        "non_context34_fixed_multi_action_rows": int(
            (
                (batch["contexts"] != ppo.SKILL_ORDER_CONTEXT)
                & (batch["min_counts"] == batch["max_counts"])
                & (batch["action_counts"] > 1)
            ).sum()
        ),
        "parts": {name: float(value.detach()) for name, value in parts.items()},
    }
    if result["rows"] != BATCH_SIZE or not audit.finite_nested(result):
        raise FloatingPointError("Invalid actor6 optimizer-step metrics")
    if result["non_context34_fixed_multi_action_rows"] <= 0:
        raise RuntimeError("Scheduled batch lacks raw fixed multi-action supervision")
    return result


def specialist_mean_loss(metrics: dict[str, dict[str, Any]]) -> float:
    return 0.5 * (
        float(metrics["raihan"]["loss"])
        + float(metrics["kdcyberdude"]["loss"])
    )


def build_endpoint_guard(
    *,
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    specialist_loss_deltas = {
        source: float(current[source]["loss"]) - float(baseline[source]["loss"])
        for source in ("raihan", "kdcyberdude")
    }
    raw_metric = "non_context34_fixed_multi_action_ordered_loss"
    specialist_raw_multi_deltas = {
        source: float(current[source][raw_metric])
        - float(baseline[source][raw_metric])
        for source in ("raihan", "kdcyberdude")
    }
    specialist_mean_delta = (
        specialist_mean_loss(current) - specialist_mean_loss(baseline)
    )
    general_delta = (
        float(current["general_valid"]["loss"])
        - float(baseline["general_valid"]["loss"])
    )
    general_raw_multi_delta = (
        float(current["general_valid"][raw_metric])
        - float(baseline["general_valid"][raw_metric])
    )
    context34_metric = "context_34_ordered_loss"
    general_context34_delta = (
        float(current["general_valid"][context34_metric])
        - float(baseline["general_valid"][context34_metric])
    )
    guard = {
        "general_valid_loss_delta": general_delta,
        "general_valid_max_absolute_loss_increase": (
            GENERAL_VALID_MAX_ABSOLUTE_LOSS_INCREASE
        ),
        "general_valid_pass": (
            general_delta <= GENERAL_VALID_MAX_ABSOLUTE_LOSS_INCREASE
        ),
        "general_valid_raw_multi_ordered_loss_delta": general_raw_multi_delta,
        "general_valid_max_raw_multi_ordered_loss_increase": (
            GENERAL_VALID_MAX_RAW_MULTI_ORDERED_LOSS_INCREASE
        ),
        "general_valid_raw_multi_ordered_pass": (
            general_raw_multi_delta
            <= GENERAL_VALID_MAX_RAW_MULTI_ORDERED_LOSS_INCREASE
        ),
        "general_valid_context34_ordered_loss_delta": general_context34_delta,
        "general_valid_max_context34_ordered_loss_increase": (
            GENERAL_VALID_MAX_CONTEXT34_ORDERED_LOSS_INCREASE
        ),
        "general_valid_context34_ordered_pass": (
            general_context34_delta
            <= GENERAL_VALID_MAX_CONTEXT34_ORDERED_LOSS_INCREASE
        ),
        "specialist_mean_loss_delta": specialist_mean_delta,
        "specialist_mean_improved": specialist_mean_delta < 0.0,
        "specialist_loss_deltas": specialist_loss_deltas,
        "each_specialist_loss_improved": all(
            delta < 0.0 for delta in specialist_loss_deltas.values()
        ),
        "specialist_raw_multi_ordered_loss_deltas": (
            specialist_raw_multi_deltas
        ),
        "each_specialist_raw_multi_ordered_loss_improved": all(
            delta < 0.0 for delta in specialist_raw_multi_deltas.values()
        ),
    }
    guard["promotion_eligible"] = all(
        guard[field]
        for field in (
            "general_valid_pass",
            "general_valid_raw_multi_ordered_pass",
            "general_valid_context34_ordered_pass",
            "specialist_mean_improved",
            "each_specialist_loss_improved",
            "each_specialist_raw_multi_ordered_loss_improved",
        )
    )
    return guard


def endpoint_payload(
    *,
    parent: dict[str, Any],
    state: dict[str, torch.Tensor],
    manifest_sha256: str,
    step: int,
    eligible: bool,
    guard: dict[str, Any],
) -> dict[str, Any]:
    retained = {}
    for name in (
        "feature_version",
        "bc_feature_version",
        "config",
        "model_config",
        "learner_deck_hash",
        "reward",
        "value_trunk_gradient",
        "actor_value_gradient",
        "action_distribution",
    ):
        if name in parent:
            retained[name] = copy.deepcopy(parent[name])
    retained.update(
        {
            "model_state_dict": state,
            "update": int(parent.get("update", -1)),
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": [
                "optimizer_state_dict",
                "bc_replay_optimizer_state_dict",
                "opponent_quota_state",
            ],
            "post_ppo_special_bc": {
                "schema_version": SCHEMA_VERSION,
                "frozen_manifest_sha256": manifest_sha256,
                "steps": step,
                "sources": ["raihan", "kdcyberdude"],
                "general_valid_rows_used_for_training": 0,
                "trainable_scope": "actor6",
                "fresh_optimizer": True,
                "promotion_eligible": eligible,
                "guard": guard,
            },
        }
    )
    return retained


def execute(prepared: PreparedRun, output_dir: Path, device_name: str) -> int:
    assert_output_absent(output_dir)
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Formal tail repair requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    model = ppo.instantiate_model_from_checkpoint(
        prepared.parent,
        prepared.bc_checkpoint,
        device,
    )
    parameters = configure_actor6(model)
    parent_state = clone_model_state(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )

    baseline = {
        name: cache_loss(model, prepared.caches[name], prepared.configs[name], device)
        for name in ("raihan", "kdcyberdude", "general_valid")
    }
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    write_exclusive_json(
        output_dir / "frozen_training_manifest.json",
        frozen_manifest_envelope(prepared.manifest),
    )

    per_step: list[dict[str, Any]] = []
    endpoints: dict[str, Any] = {}
    eligible_steps: list[int] = []
    for step, (source, batch_index) in enumerate(STEP_SCHEDULE, 1):
        step_metrics = actor6_optimizer_step(
            model=model,
            parameters=parameters,
            optimizer=optimizer,
            cpu_batch=prepared.caches[source][batch_index],
            config=prepared.configs[source],
            device=device,
        )
        per_step.append(
            {
                "step": step,
                "source": source,
                "batch_index": batch_index,
                "batch_sha256": prepared.batch_sha256[source][batch_index],
                "metrics": step_metrics,
            }
        )
        if step not in ENDPOINTS:
            continue

        state = clone_model_state(model)
        integrity = validate_endpoint_state(parent_state, state)
        current = {
            name: cache_loss(
                model,
                prepared.caches[name],
                prepared.configs[name],
                device,
            )
            for name in ("raihan", "kdcyberdude", "general_valid")
        }
        guard = build_endpoint_guard(baseline=baseline, current=current)
        eligible = bool(guard["promotion_eligible"])
        if eligible:
            eligible_steps.append(step)
        payload = endpoint_payload(
            parent=prepared.parent,
            state=state,
            manifest_sha256=prepared.manifest_sha256,
            step=step,
            eligible=eligible,
            guard=guard,
        )
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        raw_checkpoint = buffer.getvalue()
        checkpoint_name = f"postppo-special-bc-s{step:02d}.pt"
        write_exclusive_bytes(output_dir / checkpoint_name, raw_checkpoint)
        endpoints[str(step)] = {
            "checkpoint": checkpoint_name,
            "sha256": hashlib.sha256(raw_checkpoint).hexdigest(),
            "promotion_eligible": eligible,
            "guard": guard,
            "integrity": integrity,
            "cache_metrics": current,
        }

    final_state = clone_model_state(model)
    final_integrity = validate_endpoint_state(parent_state, final_state)
    result = {
        "schema_version": SCHEMA_VERSION + "-result",
        "status": "completed",
        "frozen_manifest_sha256": prepared.manifest_sha256,
        "optimizer_steps": len(STEP_SCHEDULE),
        "endpoint_steps": list(ENDPOINTS),
        "baseline_cache_metrics": baseline,
        "per_step": per_step,
        "endpoints": endpoints,
        "promotion_eligible_endpoints": eligible_steps,
        "general_valid_rows_used_for_training": 0,
        "final_integrity": final_integrity,
        "inputs_unchanged_after_training": {
            "parent": file_sha256(prepared.parent_path)
            == prepared.manifest["inputs"]["parent_checkpoint"]["sha256"],
            "bc_architecture": file_sha256(prepared.bc_path)
            == prepared.manifest["inputs"]["bc_architecture_checkpoint"]["sha256"],
            "raihan": file_sha256(RAIHAN_ARCHIVE) == RAIHAN_SHA256,
            "kdcyberdude": file_sha256(KD_ARCHIVE) == KD_SHA256,
            "general_valid": file_sha256(GENERAL_ARCHIVE) == GENERAL_SHA256,
            "trainer": file_sha256(TRAINER)
            == prepared.manifest["inputs"]["trainer_module"]["sha256"],
            "launcher": file_sha256(Path(__file__).resolve())
            == prepared.manifest["inputs"]["launcher"]["sha256"],
        },
        "scope": prepared.manifest["scope"],
    }
    if len(endpoints) != len(ENDPOINTS):
        raise RuntimeError("Not every preregistered endpoint was written")
    if not all(result["inputs_unchanged_after_training"].values()):
        raise RuntimeError("A frozen input changed during tail repair")
    write_exclusive_json(output_dir / "tail_repair_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def enforce_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"Wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("Run with my_project_env Python flags -I -B")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--freeze-manifest",
        type=Path,
        metavar="PATH",
        help="Write a new frozen manifest; performs zero optimizer steps.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute only after a frozen manifest has been reviewed.",
    )
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-bc-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def validate_mode_args(args: argparse.Namespace) -> None:
    if args.execute:
        if args.frozen_manifest is None or not args.expected_manifest_sha256:
            raise ValueError(
                "--execute requires --frozen-manifest and "
                "--expected-manifest-sha256"
            )
    elif args.frozen_manifest is not None or args.expected_manifest_sha256:
        raise ValueError(
            "--frozen-manifest/--expected-manifest-sha256 are execute-only"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_mode_args(args)
    enforce_runtime()
    output_dir = absolute_path(args.output_dir)
    prepared = prepare_run(
        parent_path=args.parent_checkpoint,
        expected_parent_sha256=args.expected_parent_sha256,
        bc_path=args.bc_checkpoint,
        expected_bc_sha256=args.expected_bc_sha256,
        output_dir=output_dir,
        device=args.device,
    )
    report = frozen_manifest_envelope(prepared.manifest)
    if args.freeze_manifest is not None:
        manifest_path = absolute_path(args.freeze_manifest)
        if manifest_path == output_dir:
            raise ValueError("Frozen manifest path must differ from output directory")
        if manifest_path.exists() or manifest_path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite manifest: {manifest_path}")
        if not manifest_path.parent.is_dir() or manifest_path.parent.is_symlink():
            raise FileNotFoundError(
                f"Manifest parent must already exist: {manifest_path.parent}"
            )
        write_exclusive_json(manifest_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    frozen = load_frozen_manifest(
        absolute_path(args.frozen_manifest),
        args.expected_manifest_sha256,
    )
    if frozen != prepared.manifest or sha256_json(frozen) != prepared.manifest_sha256:
        raise RuntimeError(
            "Current inputs/protocol do not match the reviewed frozen manifest"
        )
    return execute(prepared, output_dir, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
