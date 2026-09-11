#!/usr/bin/env python3
"""Deterministically calibrate only a PPO checkpoint's value head on BC data.

The calibration contract is intentionally narrow:

* the canonical archive ``train/`` split is consumed exactly once;
* the four value-head tensors are initialized from a canonical fresh-BC
  checkpoint;
* every non-value parameter is frozen while the whole model remains in eval
  mode;
* the loss and optimizer are fixed to unweighted BCEWithLogits and AdamW;
* no PPO optimizer, scheduler, or scaler state is copied to the output.

The resulting checkpoint is suitable for evaluation or packaging.  It is not
an optimizer-compatible PPO resume checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import orjson
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_bc_orbit as bc
import train_ppo as ppo


CALIBRATION_VERSION = "ppo-value-head-bc-calibration-v1"
VALUE_HEAD_NAMES = (
    "value_head.0.weight",
    "value_head.0.bias",
    "value_head.2.weight",
    "value_head.2.bias",
)
EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
WORKERS = 8
DEFAULT_SEED = 20260922
TRAIN_EPOCH = 1
PROBE_EPOCH = 0
TRAIN_SHUFFLE_SEED_OFFSET = 1
PROBE_SHUFFLE_SEED_OFFSET = 17
CANONICAL_DONOR_EPOCH = 7
CANONICAL_TRAIN_ROWS = 1_185_094
CANONICAL_TRAIN_BATCHES = 4_630
CANONICAL_POSITIVE_TARGETS = 687_068
CANONICAL_NEGATIVE_TARGETS = 498_026
CANONICAL_TRAIN_MEMBERS = 48
CANONICAL_DECK_HASH = (
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
)


@dataclass(frozen=True)
class CalibrationRequest:
    ppo_checkpoint: Path
    ppo_checkpoint_sha256: str
    canonical_bc_checkpoint: Path
    canonical_bc_checkpoint_sha256: str
    data: Path
    data_sha256: str
    output: Path
    manifest: Path | None = None
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    seed: int = DEFAULT_SEED
    device: str = "cuda"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(tensor_sha256(values[name])))
    return digest.hexdigest()


def batch_sha256(batch: Mapping[str, torch.Tensor]) -> str:
    return tensor_mapping_sha256(batch)


def validate_expected_sha256(path: Path, expected: str, label: str) -> str:
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"{label} expected SHA256 must be 64 lowercase hex")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def is_optimizer_continuation_key(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in ("optimizer", "scheduler", "scaler")
    )


def default_manifest_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".calibration.json")


def validate_request(request: CalibrationRequest) -> tuple[Path, Path]:
    if request.epochs != EPOCHS:
        raise ValueError(
            f"value-head calibration requires exactly {EPOCHS} epoch"
        )
    if request.batch_size != BATCH_SIZE:
        raise ValueError(
            f"value-head calibration requires batch size {BATCH_SIZE}"
        )
    if request.learning_rate != LEARNING_RATE:
        raise ValueError(
            f"value-head calibration requires learning rate {LEARNING_RATE}"
        )
    if request.weight_decay != WEIGHT_DECAY:
        raise ValueError(
            f"value-head calibration requires weight decay {WEIGHT_DECAY}"
        )
    if request.seed != DEFAULT_SEED:
        raise ValueError(
            f"value-head calibration requires seed {DEFAULT_SEED}"
        )

    sources = (
        request.ppo_checkpoint,
        request.canonical_bc_checkpoint,
        request.data,
    )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    output = request.output.resolve()
    manifest = (
        request.manifest.resolve()
        if request.manifest is not None
        else default_manifest_path(output)
    )
    source_paths = {source.resolve() for source in sources}
    if output in source_paths or manifest in source_paths:
        raise ValueError("output paths must not alias an input")
    if output == manifest:
        raise ValueError("checkpoint and manifest outputs must differ")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if manifest.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest}"
        )
    return output, manifest


def read_archive_contract(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, mode="r") as archive:
        names = archive.namelist()
        train_members = sorted(
            name
            for name in names
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not train_members:
            raise ValueError("canonical BC archive has no train JSONL members")
        if "manifest.json" not in names:
            raise ValueError("canonical BC archive is missing manifest.json")
        manifest = orjson.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict):
        raise ValueError("canonical BC archive manifest root must be a mapping")
    split_decisions = manifest.get("split_decisions")
    if not isinstance(split_decisions, dict):
        raise ValueError(
            "canonical BC archive manifest lacks split_decisions"
        )
    expected_rows = split_decisions.get("train")
    if (
        not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows <= 0
    ):
        raise ValueError(
            "canonical BC archive manifest train row count must be positive"
        )
    deck_hash_filter = manifest.get("deck_hash_filter")
    if expected_rows != CANONICAL_TRAIN_ROWS:
        raise ValueError(
            "canonical BC archive train row gate failed: "
            f"{expected_rows} != {CANONICAL_TRAIN_ROWS}"
        )
    if len(train_members) != CANONICAL_TRAIN_MEMBERS:
        raise ValueError(
            "canonical BC archive train member gate failed: "
            f"{len(train_members)} != {CANONICAL_TRAIN_MEMBERS}"
        )
    if deck_hash_filter != CANONICAL_DECK_HASH:
        raise ValueError(
            "canonical BC archive deck hash gate failed: "
            f"{deck_hash_filter!r} != {CANONICAL_DECK_HASH!r}"
        )
    return {
        "expected_train_rows": expected_rows,
        "train_member_count": len(train_members),
        "train_members": train_members,
        "deck_hash_filter": deck_hash_filter,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_split_policy": manifest.get("split_policy"),
    }


def load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{label} checkpoint root must be a mapping")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"{label} checkpoint has no model_state_dict")
    return checkpoint


def validate_model_configs(
    parent: dict[str, Any],
    canonical_bc: dict[str, Any],
) -> dict[str, Any]:
    parent_config = ppo.checkpoint_model_config(parent)
    bc_config = ppo.checkpoint_model_config(canonical_bc)
    if parent_config != bc_config:
        raise ValueError(
            "PPO and canonical fresh-BC model configurations differ: "
            + json.dumps(
                {"ppo": parent_config, "canonical_bc": bc_config},
                sort_keys=True,
            )
        )
    return parent_config


def validate_canonical_bc_donor(
    checkpoint: dict[str, Any],
    archive_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint.get("epoch") != CANONICAL_DONOR_EPOCH:
        raise ValueError(
            "canonical fresh-BC donor epoch gate failed: "
            f"{checkpoint.get('epoch')!r} != {CANONICAL_DONOR_EPOCH}"
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("canonical fresh-BC checkpoint lacks training config")
    expected = {
        "seed": DEFAULT_SEED,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "split_mode": "archive",
        "use_trajectory_weights": False,
        "max_train_rows": None,
        "expected_train_rows": CANONICAL_TRAIN_ROWS,
    }
    mismatches = {
        name: {"expected": value, "actual": config.get(name)}
        for name, value in expected.items()
        if config.get(name) != value
    }
    deck_hashes = tuple(config.get("deck_hashes", ()))
    team_names = tuple(config.get("team_names", ()))
    shuffle_buffer_rows = int(
        config.get("train_shuffle_buffer_rows_per_worker", 0) or 0
    )
    if deck_hashes != (CANONICAL_DECK_HASH,):
        mismatches["deck_hashes"] = {
            "expected": [CANONICAL_DECK_HASH],
            "actual": list(deck_hashes),
        }
    if team_names:
        mismatches["team_names"] = {
            "expected": [],
            "actual": list(team_names),
        }
    if shuffle_buffer_rows != 0:
        mismatches["train_shuffle_buffer_rows_per_worker"] = {
            "expected": 0,
            "actual": shuffle_buffer_rows,
        }
    if (
        int(archive_contract["expected_train_rows"])
        != int(config.get("expected_train_rows", -1))
    ):
        mismatches["archive_vs_checkpoint_train_rows"] = {
            "expected": archive_contract["expected_train_rows"],
            "actual": config.get("expected_train_rows"),
        }
    if mismatches:
        raise ValueError(
            "canonical fresh-BC donor training contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        **expected,
        "donor_epoch": CANONICAL_DONOR_EPOCH,
        "deck_hashes": list(deck_hashes),
        "team_names": list(team_names),
        "train_shuffle_buffer_rows_per_worker": shuffle_buffer_rows,
    }


def clone_model_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def extract_value_state(
    state: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    actual_names = tuple(sorted(
        name for name in state if name.startswith("value_head.")
    ))
    if actual_names != tuple(sorted(VALUE_HEAD_NAMES)):
        raise ValueError(
            f"{label} must contain exactly the four canonical value-head "
            f"tensors; got {actual_names}"
        )
    result: dict[str, torch.Tensor] = {}
    for name in VALUE_HEAD_NAMES:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label} tensor {name!r} is not a tensor")
        result[name] = tensor.detach().cpu().clone()
    return result


def initialize_value_head_from_bc(
    model: torch.nn.Module,
    canonical_bc: dict[str, Any],
) -> dict[str, dict[str, str]]:
    model_state = model.state_dict()
    parent_value = extract_value_state(model_state, label="PPO checkpoint")
    canonical_value = extract_value_state(
        canonical_bc["model_state_dict"],
        label="canonical fresh-BC checkpoint",
    )
    for name in VALUE_HEAD_NAMES:
        if (
            model_state[name].shape != canonical_value[name].shape
            or model_state[name].dtype != canonical_value[name].dtype
        ):
            raise ValueError(
                f"canonical fresh-BC value tensor {name!r} is incompatible"
            )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in canonical_value:
                parameter.copy_(
                    canonical_value[name].to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
    initialized = extract_value_state(
        model.state_dict(),
        label="initialized learner",
    )
    audit: dict[str, dict[str, str]] = {}
    for name in VALUE_HEAD_NAMES:
        canonical_hash = tensor_sha256(canonical_value[name])
        initialized_hash = tensor_sha256(initialized[name])
        if initialized_hash != canonical_hash:
            raise RuntimeError(
                f"value tensor {name!r} did not initialize exactly from BC"
            )
        audit[name] = {
            "ppo_parent_sha256": tensor_sha256(parent_value[name]),
            "canonical_bc_sha256": canonical_hash,
            "initialized_sha256": initialized_hash,
        }
    return audit


def freeze_to_value_head(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    names = [name for name, _ in named]
    if tuple(names) != VALUE_HEAD_NAMES:
        raise RuntimeError(
            "trainable parameters are not exactly the four value-head "
            f"tensors: {names}"
        )
    model.eval()
    if model.training or any(
        module.training for module in model.modules()
    ):
        raise RuntimeError("model trunk is not fully in eval mode")
    return [parameter for _, parameter in named], names


def make_train_loader(
    *,
    data: Path,
    model_config: Mapping[str, Any],
    seed: int,
    epoch: int,
    seed_offset: int,
    device: torch.device,
) -> DataLoader:
    dataset = bc.ZipDecisionDataset(
        archive_path=data,
        split="train",
        max_rows=None,
        split_seed=seed,
        shuffle_seed=seed + seed_offset,
        epoch=epoch,
        hash_size=int(model_config["hash_size"]),
        max_state_entities=int(model_config["max_state_entities"]),
        use_trajectory_weights=False,
        deck_hashes=(CANONICAL_DECK_HASH,),
        team_names=(),
        split_mode="archive",
        shuffle_buffer_rows=0,
        policy_team_weights=None,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        collate_fn=partial(
            bc.collate_decisions,
            max_state_entities=int(model_config["max_state_entities"]),
            entity_fields=int(model_config["entity_fields"]),
            option_fields=int(model_config["option_fields"]),
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=2,
        generator=generator,
        in_order=True,
        multiprocessing_context="spawn",
    )


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def fixed_probe(
    *,
    model: torch.nn.Module,
    data: Path,
    model_config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    loader = make_train_loader(
        data=data,
        model_config=model_config,
        seed=seed,
        epoch=PROBE_EPOCH,
        seed_offset=PROBE_SHUFFLE_SEED_OFFSET,
        device=device,
    )
    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as error:
        raise ValueError("canonical BC archive train split produced no rows") from error
    finally:
        del iterator
    moved = move_batch(batch, device)
    with torch.no_grad():
        outputs = model(moved)
    logits = {
        name: outputs[name].detach().cpu().clone()
        for name in ("policy_logits", "count_logits")
    }
    audit = {
        "rows": int(batch["win_targets"].numel()),
        "batch_sha256": batch_sha256(batch),
        "epoch": PROBE_EPOCH,
        "shuffle_seed": seed + PROBE_SHUFFLE_SEED_OFFSET,
        "workers": WORKERS,
        "in_order": True,
        "multiprocessing_context": "spawn",
        "policy_logits_sha256": tensor_sha256(logits["policy_logits"]),
        "count_logits_sha256": tensor_sha256(logits["count_logits"]),
    }
    return logits, audit


def train_one_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Path,
    model_config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    loader = make_train_loader(
        data=data,
        model_config=model_config,
        seed=seed,
        epoch=TRAIN_EPOCH,
        seed_offset=TRAIN_SHUFFLE_SEED_OFFSET,
        device=device,
    )
    rows = 0
    batches = 0
    positive_targets = 0
    loss_sum = 0.0
    first_batch_hash: str | None = None
    last_batch_hash: str | None = None
    for batch in loader:
        current_hash = batch_sha256(batch)
        if first_batch_hash is None:
            first_batch_hash = current_hash
        last_batch_hash = current_hash
        moved = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(moved)
        value_logits = outputs["value_logits"]
        targets = moved["win_targets"]
        loss = F.binary_cross_entropy_with_logits(value_logits, targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite value BCE during calibration")
        loss.backward()
        unexpected_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not name.startswith("value_head.") and parameter.grad is not None
        ]
        if unexpected_gradients:
            raise RuntimeError(
                "non-value parameters received gradients: "
                f"{unexpected_gradients[:12]}"
            )
        missing_or_bad_gradients = [
            name
            for name, parameter in model.named_parameters()
            if name.startswith("value_head.")
            and (
                parameter.grad is None
                or not bool(torch.isfinite(parameter.grad).all())
            )
        ]
        if missing_or_bad_gradients:
            raise RuntimeError(
                "value-head gradient is missing or non-finite: "
                f"{missing_or_bad_gradients}"
            )
        optimizer.step()
        current_rows = int(targets.numel())
        rows += current_rows
        batches += 1
        positive_targets += int(targets.detach().sum().item())
        loss_sum += float(loss.detach().item()) * current_rows
    if rows <= 0 or first_batch_hash is None or last_batch_hash is None:
        raise ValueError("canonical BC archive train split produced no rows")
    return {
        "epochs": EPOCHS,
        "rows": rows,
        "batches": batches,
        "positive_targets": positive_targets,
        "negative_targets": rows - positive_targets,
        "mean_bce": loss_sum / rows,
        "first_batch_sha256": first_batch_hash,
        "last_batch_sha256": last_batch_hash,
        "epoch": TRAIN_EPOCH,
        "shuffle_seed": seed + TRAIN_SHUFFLE_SEED_OFFSET,
        "workers": WORKERS,
        "in_order": True,
        "multiprocessing_context": "spawn",
    }


def validate_canonical_training_result(
    metrics: Mapping[str, Any],
) -> dict[str, int]:
    expected = {
        "rows": CANONICAL_TRAIN_ROWS,
        "batches": CANONICAL_TRAIN_BATCHES,
        "positive_targets": CANONICAL_POSITIVE_TARGETS,
        "negative_targets": CANONICAL_NEGATIVE_TARGETS,
    }
    mismatches = {
        name: {"expected": value, "actual": metrics.get(name)}
        for name, value in expected.items()
        if metrics.get(name) != value
    }
    if mismatches:
        raise RuntimeError(
            "canonical BC calibration exposure gate failed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return expected


def non_value_tensor_audit(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    if set(before) != set(after):
        raise RuntimeError("model state tensor names changed during calibration")
    audit: list[dict[str, Any]] = []
    changed: list[str] = []
    for name in sorted(before):
        if name.startswith("value_head."):
            continue
        before_hash = tensor_sha256(before[name])
        after_hash = tensor_sha256(after[name])
        unchanged = (
            before_hash == after_hash
            and torch.equal(before[name], after[name])
        )
        if not unchanged:
            changed.append(name)
        audit.append(
            {
                "name": name,
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "unchanged": unchanged,
            }
        )
    if changed:
        raise RuntimeError(
            f"non-value tensors changed during calibration: {changed[:12]}"
        )
    return audit


def atomic_save_checkpoint(
    payload: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    output: Path,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        reloaded = torch.load(
            temporary,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(reloaded, dict):
            raise RuntimeError("saved checkpoint root is not a mapping")
        reloaded_state = reloaded.get("model_state_dict")
        if not isinstance(reloaded_state, dict):
            raise RuntimeError("saved checkpoint lacks model_state_dict")
        if set(reloaded_state) != set(expected_state):
            raise RuntimeError("saved checkpoint model tensor names changed")
        changed = [
            name
            for name in expected_state
            if not torch.equal(reloaded_state[name], expected_state[name])
        ]
        if changed:
            raise RuntimeError(
                f"saved checkpoint tensors changed: {changed[:12]}"
            )
        inherited = [
            name for name in reloaded if is_optimizer_continuation_key(name)
        ]
        if inherited:
            raise RuntimeError(
                f"saved checkpoint inherited optimizer state: {inherited}"
            )
        os.link(temporary, output)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def build_output_payload(
    parent: dict[str, Any],
    model_state: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    stripped = sorted(
        name for name in parent if is_optimizer_continuation_key(name)
    )
    payload = {
        name: copy.deepcopy(value)
        for name, value in parent.items()
        if name != "model_state_dict"
        and not is_optimizer_continuation_key(name)
    }
    payload["model_state_dict"] = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model_state.items()
    }
    payload["post_ppo_value_head_calibration"] = copy.deepcopy(provenance)
    return payload, stripped


def run_calibration(request: CalibrationRequest) -> dict[str, Any]:
    output, manifest_path = validate_request(request)
    ppo_sha = validate_expected_sha256(
        request.ppo_checkpoint,
        request.ppo_checkpoint_sha256,
        "PPO checkpoint",
    )
    bc_sha = validate_expected_sha256(
        request.canonical_bc_checkpoint,
        request.canonical_bc_checkpoint_sha256,
        "canonical fresh-BC checkpoint",
    )
    data_sha = validate_expected_sha256(
        request.data,
        request.data_sha256,
        "canonical BC archive",
    )
    archive_contract = read_archive_contract(request.data)

    device = torch.device(request.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    parent = load_checkpoint(request.ppo_checkpoint, "PPO")
    canonical_bc = load_checkpoint(
        request.canonical_bc_checkpoint,
        "canonical fresh-BC",
    )
    if parent.get("feature_version") != ppo.PPO_FEATURE_VERSION:
        raise ValueError(
            "input learner must be a PPO checkpoint with feature_version "
            f"{ppo.PPO_FEATURE_VERSION!r}"
        )
    if canonical_bc.get("feature_version") != bc.FEATURE_VERSION:
        raise ValueError(
            "canonical fresh-BC checkpoint has wrong feature_version"
        )
    model_config = validate_model_configs(parent, canonical_bc)
    canonical_donor_contract = validate_canonical_bc_donor(
        canonical_bc,
        archive_contract,
    )

    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    random.seed(request.seed)
    torch.manual_seed(request.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(request.seed)
    try:
        model = ppo.instantiate_model_from_checkpoint(
            parent,
            canonical_bc,
            device,
        )
        parent_state = clone_model_state(model)
        value_initialization = initialize_value_head_from_bc(
            model,
            canonical_bc,
        )
        initialized_state = clone_model_state(model)
        non_value_initialization_audit = non_value_tensor_audit(
            parent_state,
            initialized_state,
        )
        parameters, optimizer_parameter_names = freeze_to_value_head(model)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        probe_before, probe_audit = fixed_probe(
            model=model,
            data=request.data,
            model_config=model_config,
            seed=request.seed,
            device=device,
        )
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data=request.data,
            model_config=model_config,
            seed=request.seed,
            device=device,
        )
        if train_metrics["rows"] != archive_contract["expected_train_rows"]:
            raise RuntimeError(
                "consumed train rows do not match canonical archive manifest: "
                f"{train_metrics['rows']} != "
                f"{archive_contract['expected_train_rows']}"
            )
        canonical_exposure_gate = validate_canonical_training_result(
            train_metrics
        )
        final_state = clone_model_state(model)
        non_value_audit = non_value_tensor_audit(
            initialized_state,
            final_state,
        )
        final_value = extract_value_state(
            final_state,
            label="calibrated learner",
        )
        value_changed = [
            name
            for name in VALUE_HEAD_NAMES
            if not torch.equal(initialized_state[name], final_state[name])
        ]
        if not value_changed:
            raise RuntimeError("value-head calibration changed no value tensor")

        probe_loader = make_train_loader(
            data=request.data,
            model_config=model_config,
            seed=request.seed,
            epoch=PROBE_EPOCH,
            seed_offset=PROBE_SHUFFLE_SEED_OFFSET,
            device=device,
        )
        probe_iterator = iter(probe_loader)
        try:
            probe_batch_after = next(probe_iterator)
        except StopIteration as error:
            raise RuntimeError("fixed probe disappeared after calibration") from error
        finally:
            del probe_iterator
        if batch_sha256(probe_batch_after) != probe_audit["batch_sha256"]:
            raise RuntimeError("fixed probe batch changed across calibration")
        with torch.no_grad():
            probe_outputs_after = model(move_batch(probe_batch_after, device))
        probe_after = {
            name: probe_outputs_after[name].detach().cpu().clone()
            for name in ("policy_logits", "count_logits")
        }
        policy_equal = torch.equal(
            probe_before["policy_logits"],
            probe_after["policy_logits"],
        )
        count_equal = torch.equal(
            probe_before["count_logits"],
            probe_after["count_logits"],
        )
        if not policy_equal or not count_equal:
            raise RuntimeError(
                "fixed-probe policy/count logits changed during value-only "
                "calibration"
            )
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    for path, expected, label in (
        (request.ppo_checkpoint, ppo_sha, "PPO checkpoint"),
        (
            request.canonical_bc_checkpoint,
            bc_sha,
            "canonical fresh-BC checkpoint",
        ),
        (request.data, data_sha, "canonical BC archive"),
    ):
        if file_sha256(path) != expected:
            raise RuntimeError(f"{label} changed during calibration")

    value_final_audit = {}
    for name in VALUE_HEAD_NAMES:
        value_final_audit[name] = {
            **value_initialization[name],
            "calibrated_sha256": tensor_sha256(final_value[name]),
            "changed_by_calibration": name in value_changed,
        }
    probe_integrity = {
        **probe_audit,
        "policy_logits_before_sha256": tensor_sha256(
            probe_before["policy_logits"]
        ),
        "policy_logits_after_sha256": tensor_sha256(
            probe_after["policy_logits"]
        ),
        "policy_logits_bitwise_equal": policy_equal,
        "count_logits_before_sha256": tensor_sha256(
            probe_before["count_logits"]
        ),
        "count_logits_after_sha256": tensor_sha256(
            probe_after["count_logits"]
        ),
        "count_logits_bitwise_equal": count_equal,
    }
    provenance: dict[str, Any] = {
        "version": CALIBRATION_VERSION,
        "status": "completed",
        "not_a_new_ppo_update": True,
        "not_optimizer_compatible_for_resume": True,
        "inputs": {
            "ppo_checkpoint": str(request.ppo_checkpoint.resolve()),
            "ppo_checkpoint_sha256": ppo_sha,
            "canonical_fresh_bc_checkpoint": str(
                request.canonical_bc_checkpoint.resolve()
            ),
            "canonical_fresh_bc_checkpoint_sha256": bc_sha,
            "canonical_bc_archive": str(request.data.resolve()),
            "canonical_bc_archive_sha256": data_sha,
        },
        "data": {
            **archive_contract,
            "split": "train",
            "split_mode": "archive",
            "rows_consumed": train_metrics["rows"],
            "source_read_only_and_unchanged": True,
        },
        "model_config": model_config,
        "canonical_fresh_bc_donor_contract": canonical_donor_contract,
        "canonical_exposure_gate": {
            **canonical_exposure_gate,
            "passed": True,
        },
        "training": {
            "loss": "binary_cross_entropy_with_logits",
            "loss_weighting": "uniform_rows",
            "optimizer": "AdamW",
            "optimizer_parameter_names": optimizer_parameter_names,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "seed": request.seed,
            "model_mode": "eval",
            "train_shuffle": "fixed_member_shuffle_no_row_buffer",
            "loader_order": "pytorch_in_order_true",
            "multiprocessing_context": "spawn",
            "worker_seed_generator": request.seed
            + TRAIN_SHUFFLE_SEED_OFFSET,
            "deterministic_algorithms": True,
            **train_metrics,
        },
        "value_head_tensors": value_final_audit,
        "value_head_changed_names": value_changed,
        "non_value_initialization_audit": non_value_initialization_audit,
        "non_value_tensor_audit": non_value_audit,
        "all_non_value_tensors_unchanged": True,
        "fixed_probe": probe_integrity,
        "model_state_sha256_before_calibration": tensor_mapping_sha256(
            initialized_state
        ),
        "model_state_sha256_after_calibration": tensor_mapping_sha256(
            final_state
        ),
    }
    output_payload, stripped = build_output_payload(
        parent,
        final_state,
        provenance,
    )
    provenance["stripped_parent_training_state_keys"] = stripped
    output_payload["post_ppo_value_head_calibration"] = copy.deepcopy(provenance)
    atomic_save_checkpoint(output_payload, final_state, output)
    output_sha = file_sha256(output)

    manifest_payload = {
        **provenance,
        "output": {
            "checkpoint": str(output),
            "checkpoint_sha256": output_sha,
            "checkpoint_bytes": output.stat().st_size,
            "manifest": str(manifest_path),
            "optimizer_continuation_state_present": False,
        },
    }
    atomic_write_json(manifest_path, manifest_payload)
    result = {
        "status": "completed",
        "checkpoint": str(output),
        "checkpoint_sha256": output_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "rows": train_metrics["rows"],
        "optimizer_parameter_names": optimizer_parameter_names,
        "all_non_value_tensors_unchanged": True,
        "policy_logits_bitwise_equal": policy_equal,
        "count_logits_bitwise_equal": count_equal,
        "stripped_parent_training_state_keys": stripped,
    }
    return result


def parse_args() -> CalibrationRequest:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint-sha256", required=True)
    parser.add_argument(
        "--canonical-bc-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--canonical-bc-checkpoint-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return CalibrationRequest(
        ppo_checkpoint=args.ppo_checkpoint,
        ppo_checkpoint_sha256=args.ppo_checkpoint_sha256,
        canonical_bc_checkpoint=args.canonical_bc_checkpoint,
        canonical_bc_checkpoint_sha256=(
            args.canonical_bc_checkpoint_sha256
        ),
        data=args.data,
        data_sha256=args.data_sha256,
        output=args.output,
        manifest=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


def main() -> None:
    result = run_calibration(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
