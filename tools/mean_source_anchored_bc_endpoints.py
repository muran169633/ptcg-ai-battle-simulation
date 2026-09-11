#!/usr/bin/env python3
"""Build one fail-closed source-anchored mean of two BC endpoints.

Only the complete ``transformer.*`` stack and the exact six actor tensors are
averaged.  Every other tensor must still be bitwise identical to both the
frozen source checkpoint and both independently trained endpoints.  The
result is a deployment-only checkpoint; it deliberately omits optimizer and
scheduler state and cannot be used as a faithful training resume checkpoint.
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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import compose_bc_checkpoint_blocks as block_utils  # noqa: E402
import interpolate_bc_checkpoints as checkpoint_utils  # noqa: E402


PROTOCOL_SCHEMA_VERSION = "ptcg-yanz-source-anchored-bc-preregister-v1"
OUTPUT_SCHEMA_VERSION = "ptcg-source-anchored-endpoint-mean-v1"
FORMULA = "source_complement_plus_selected_endpoint_mean_0p5"
ALPHA = 0.5
SELECTED_SCOPE = "transformer_actor"
EXPECTED_SEEDS = (2026081021, 2026081022)
EXPECTED_SELECTED_TENSORS = 56
EXPECTED_SELECTED_PARAMETERS = 859_137
EXPECTED_COMPLEMENT_TENSORS = 24
EXPECTED_COMPLEMENT_PARAMETERS = 4_287_506
SCHEDULE_ALGORITHM = "warmup_then_cosine_floor_0p05"
WARMUP_FRACTION = 0.03
MINIMUM_LR_MULTIPLIER = 0.05

REQUIRED_ENDPOINT_CONFIG_KEYS = frozenset(
    {
        "data",
        "epochs",
        "batch_size",
        "workers",
        "learning_rate",
        "weight_decay",
        "set_bce_weight",
        "count_loss_weight",
        "value_loss_weight",
        "target_accuracy",
        "use_trajectory_weights",
        "trajectory_weight_scope",
        "train_shuffle_buffer_rows_per_worker",
        "flexible_selection_loss_weight",
        "count_trunk_gradient_scale",
        "expected_train_rows",
        "split_mode",
        "max_train_rows",
        "max_valid_rows",
        "max_test_rows",
        "policy_team_balance",
        "init_checkpoint",
        "init_checkpoint_sha256",
        "trainable_scope",
        "expected_trainable_tensors",
        "expected_trainable_parameters",
    }
)
ALLOWED_ENDPOINT_CONFIG_DIFFERENCES = frozenset({"seed", "output_dir"})


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Protocol JSON contains forbidden constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Protocol JSON contains duplicate key {key!r}")
        output[key] = value
    return output


def load_protocol(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Protocol does not exist: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError("Protocol must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Protocol is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise TypeError("Protocol JSON root must be an object")
    return document


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must all be strings")
    return value


def require_non_bool_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a non-bool integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def require_exact_number(value: Any, expected: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(f"{label} must equal {expected!r}; got {value!r}")
    return float(value)


def json_compatible(value: Any, label: str = "value") -> Any:
    """Return a deterministic JSON-compatible representation or fail."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} contains a non-string mapping key")
            output[key] = json_compatible(child, f"{label}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            json_compatible(child, f"{label}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(
        f"{label} contains unsupported value type {type(value).__name__}"
    )


def resolved_existing_file(path: Path, label: str) -> Path:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    return resolved


def resolve_bound_path(value: Any, protocol_path: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label}.path must be a nonempty string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = protocol_path.parent / candidate
    return resolved_existing_file(candidate, f"{label}.path")


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def validate_bound_input(
    inputs: Mapping[str, Any],
    key: str,
    actual_path: Path,
    protocol_path: Path,
) -> dict[str, str]:
    entry = require_mapping(inputs.get(key), f"bindings.inputs.{key}")
    bound_path = resolve_bound_path(
        entry.get("path"),
        protocol_path,
        f"bindings.inputs.{key}",
    )
    if bound_path != actual_path:
        raise ValueError(
            f"bindings.inputs.{key}.path mismatch: protocol={bound_path}, "
            f"CLI={actual_path}"
        )
    expected_sha = require_sha256(
        entry.get("sha256"),
        f"bindings.inputs.{key}.sha256",
    )
    actual_sha = checkpoint_utils.sha256_file(actual_path)
    if expected_sha != actual_sha:
        raise ValueError(
            f"bindings.inputs.{key}.sha256 mismatch: "
            f"protocol={expected_sha}, actual={actual_sha}"
        )
    return {"path": str(actual_path), "sha256": actual_sha}


def tensor_raw_bytes(value: torch.Tensor) -> bytes:
    if value.layout != torch.strided:
        raise ValueError("Only strided model-state tensors are supported")
    contiguous = value.detach().cpu().contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def tensors_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.layout == right.layout
        and tensor_raw_bytes(left) == tensor_raw_bytes(right)
    )


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ptcg-model-state-bitwise-v1\0")
    for name in sorted(state):
        if not isinstance(name, str):
            raise TypeError("model_state_dict keys must be strings")
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model_state_dict entry {name!r} is not a tensor")
        if value.layout != torch.strided:
            raise ValueError(f"model_state_dict tensor {name!r} must be strided")
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "big"))
        digest.update(dtype_bytes)
        digest.update(len(value.shape).to_bytes(4, "big"))
        for dimension in value.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=True))
        raw = tensor_raw_bytes(value)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def source_anchored_mean_state(
    source_state: Mapping[str, Any],
    endpoint_a_state: Mapping[str, Any],
    endpoint_b_state: Mapping[str, Any],
    *,
    expected_selected_tensors: int = EXPECTED_SELECTED_TENSORS,
    expected_selected_parameters: int = EXPECTED_SELECTED_PARAMETERS,
    expected_complement_tensors: int = EXPECTED_COMPLEMENT_TENSORS,
    expected_complement_parameters: int = EXPECTED_COMPLEMENT_PARAMETERS,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    for label, value in (
        ("expected_selected_tensors", expected_selected_tensors),
        ("expected_selected_parameters", expected_selected_parameters),
        ("expected_complement_tensors", expected_complement_tensors),
        ("expected_complement_parameters", expected_complement_parameters),
    ):
        require_non_bool_int(value, label, minimum=1)

    keys_source = set(source_state)
    keys_a = set(endpoint_a_state)
    keys_b = set(endpoint_b_state)
    if keys_source != keys_a or keys_source != keys_b:
        raise ValueError(
            "model_state_dict key mismatch: "
            f"missing_from_A={sorted(keys_source - keys_a)[:10]}, "
            f"extra_in_A={sorted(keys_a - keys_source)[:10]}, "
            f"missing_from_B={sorted(keys_source - keys_b)[:10]}, "
            f"extra_in_B={sorted(keys_b - keys_source)[:10]}"
        )
    if any(not isinstance(name, str) for name in keys_source):
        raise TypeError("model_state_dict keys must all be strings")

    selected_keys = sorted(
        name
        for name in keys_source
        if block_utils.selected_by_transformer_actor(name)
    )
    missing_actor = sorted(block_utils.ACTOR_KEYS - set(selected_keys))
    if missing_actor:
        raise KeyError(f"Missing required actor tensors: {missing_actor}")
    transformer_keys = [
        name for name in selected_keys if name.startswith("transformer.")
    ]
    actor_keys = [name for name in selected_keys if name in block_utils.ACTOR_KEYS]
    if not transformer_keys:
        raise ValueError("No transformer.* tensors were selected")
    if set(actor_keys) != block_utils.ACTOR_KEYS:
        raise RuntimeError("Selected actor tensors are not the exact actor6 set")
    complement_keys = sorted(keys_source - set(selected_keys))

    selected_parameters = 0
    complement_parameters = 0
    output: dict[str, torch.Tensor] = {}
    for name in sorted(keys_source):
        source = source_state[name]
        endpoint_a = endpoint_a_state[name]
        endpoint_b = endpoint_b_state[name]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (source, endpoint_a, endpoint_b)
        ):
            raise TypeError(
                f"model_state_dict entry {name!r} must be a tensor in source, "
                "endpoint A, and endpoint B"
            )
        assert isinstance(source, torch.Tensor)
        assert isinstance(endpoint_a, torch.Tensor)
        assert isinstance(endpoint_b, torch.Tensor)
        if source.shape != endpoint_a.shape or source.shape != endpoint_b.shape:
            raise ValueError(
                f"Tensor shape mismatch for {name!r}: "
                f"source={tuple(source.shape)}, A={tuple(endpoint_a.shape)}, "
                f"B={tuple(endpoint_b.shape)}"
            )
        if source.dtype != endpoint_a.dtype or source.dtype != endpoint_b.dtype:
            raise ValueError(
                f"Tensor dtype mismatch for {name!r}: "
                f"source={source.dtype}, A={endpoint_a.dtype}, "
                f"B={endpoint_b.dtype}"
            )
        if source.layout != endpoint_a.layout or source.layout != endpoint_b.layout:
            raise ValueError(
                f"Tensor layout mismatch for {name!r}: "
                f"source={source.layout}, A={endpoint_a.layout}, "
                f"B={endpoint_b.layout}"
            )
        if source.layout != torch.strided:
            raise ValueError(f"Tensor {name!r} must have strided layout")

        if not source.is_floating_point() and not (
            tensors_bitwise_equal(source, endpoint_a)
            and tensors_bitwise_equal(source, endpoint_b)
        ):
            raise ValueError(
                f"Non-floating tensor {name!r} differs across source/A/B; "
                "non-floating tensors are never converted or averaged"
            )

        if name in selected_keys:
            if source.dtype != torch.float32:
                raise ValueError(
                    f"Selected tensor {name!r} must be FP32; got {source.dtype}"
                )
            for label, value in (
                ("source", source),
                ("endpoint A", endpoint_a),
                ("endpoint B", endpoint_b),
            ):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(
                        f"Selected {label} tensor {name!r} contains non-finite values"
                    )
            averaged = torch.lerp(endpoint_a, endpoint_b, ALPHA)
            if averaged.dtype != torch.float32:
                raise RuntimeError(f"Averaged tensor {name!r} is no longer FP32")
            if not bool(torch.isfinite(averaged).all()):
                raise ValueError(f"Averaged tensor {name!r} contains non-finite values")
            output[name] = averaged
            selected_parameters += averaged.numel()
        else:
            if not tensors_bitwise_equal(source, endpoint_a):
                raise ValueError(
                    f"Endpoint A complement tensor {name!r} differs bitwise "
                    "from source"
                )
            if not tensors_bitwise_equal(source, endpoint_b):
                raise ValueError(
                    f"Endpoint B complement tensor {name!r} differs bitwise "
                    "from source"
                )
            output[name] = source.clone()
            complement_parameters += source.numel()

    observed = {
        "selected_tensors": len(selected_keys),
        "selected_parameters": selected_parameters,
        "complement_tensors": len(complement_keys),
        "complement_parameters": complement_parameters,
    }
    expected = {
        "selected_tensors": expected_selected_tensors,
        "selected_parameters": expected_selected_parameters,
        "complement_tensors": expected_complement_tensors,
        "complement_parameters": expected_complement_parameters,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if mismatches:
        raise ValueError(f"Source-anchored parameter audit mismatch: {mismatches}")

    for name in complement_keys:
        if not tensors_bitwise_equal(output[name], source_state[name]):
            raise RuntimeError(f"Output complement tensor {name!r} changed from source")
        if output[name].data_ptr() == source_state[name].data_ptr():
            raise RuntimeError(f"Output complement tensor {name!r} was not cloned")

    return output, {
        "selected_scope": SELECTED_SCOPE,
        "selected_tensor_count": len(selected_keys),
        "selected_parameter_count": selected_parameters,
        "selected_keys": selected_keys,
        "transformer_tensor_count": len(transformer_keys),
        "actor_tensor_count": len(actor_keys),
        "actor_keys": sorted(block_utils.ACTOR_KEYS),
        "complement_tensor_count": len(complement_keys),
        "complement_parameter_count": complement_parameters,
        "complement_keys": complement_keys,
        "total_tensor_count": len(keys_source),
        "total_parameter_count": selected_parameters + complement_parameters,
    }


def shared_checkpoint_contract(
    source: Mapping[str, Any],
    endpoint_a: Mapping[str, Any],
    endpoint_b: Mapping[str, Any],
) -> tuple[dict[str, int | float], dict[str, str], str, Any, tuple[int, int, int]]:
    for checkpoint, label in (
        (source, "source"),
        (endpoint_a, "endpoint A"),
        (endpoint_b, "endpoint B"),
    ):
        checkpoint_utils.require_bc_feature_version(checkpoint, label)

    configs_with_sources = [
        checkpoint_utils.runtime_model_config(checkpoint, label)
        for checkpoint, label in (
            (source, "source"),
            (endpoint_a, "endpoint A"),
            (endpoint_b, "endpoint B"),
        )
    ]
    runtime_configs = [item[0] for item in configs_with_sources]
    if runtime_configs[0] != runtime_configs[1] or runtime_configs[0] != runtime_configs[2]:
        raise ValueError("Runtime model config mismatch across source/endpoints")

    deck_hashes = [
        checkpoint_utils.single_deck_hash(checkpoint, label)
        for checkpoint, label in (
            (source, "source"),
            (endpoint_a, "endpoint A"),
            (endpoint_b, "endpoint B"),
        )
    ]
    if len(set(deck_hashes)) != 1:
        raise ValueError(f"Checkpoint deck hash mismatch: {deck_hashes}")

    interface = checkpoint_utils.shared_ppo_interface(source, endpoint_a)
    if endpoint_b.get("ppo_interface") != interface:
        raise ValueError("Checkpoint ppo_interface mismatch for endpoint B")

    epochs: list[int] = []
    for checkpoint, label in (
        (source, "source"),
        (endpoint_a, "endpoint A"),
        (endpoint_b, "endpoint B"),
    ):
        epochs.append(
            require_non_bool_int(checkpoint.get("epoch"), f"{label} epoch", minimum=1)
        )
    if epochs[1] != epochs[2]:
        raise ValueError(
            f"Endpoint epoch mismatch: A={epochs[1]}, B={epochs[2]}"
        )

    config_sources = {
        "source": configs_with_sources[0][1],
        "endpoint_a": configs_with_sources[1][1],
        "endpoint_b": configs_with_sources[2][1],
    }
    return runtime_configs[0], config_sources, deck_hashes[0], interface, tuple(epochs)


def validate_endpoint_training_bindings(
    endpoint_a: Mapping[str, Any],
    endpoint_b: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    mixed_archive_path: Path,
    endpoint_a_seed: int,
    endpoint_b_seed: int,
    endpoint_epoch: int,
) -> dict[str, Any]:
    config_a = require_mapping(endpoint_a.get("config"), "endpoint A config")
    config_b = require_mapping(endpoint_b.get("config"), "endpoint B config")
    if set(config_a) != set(config_b):
        raise ValueError(
            "Endpoint config key mismatch: "
            f"only_A={sorted(set(config_a) - set(config_b))}, "
            f"only_B={sorted(set(config_b) - set(config_a))}"
        )
    for key in sorted(set(config_a) - ALLOWED_ENDPOINT_CONFIG_DIFFERENCES):
        if json_compatible(config_a[key], f"endpoint A config.{key}") != json_compatible(
            config_b[key], f"endpoint B config.{key}"
        ):
            raise ValueError(
                f"Endpoint configs differ at forbidden key {key!r}: "
                f"A={config_a[key]!r}, B={config_b[key]!r}"
            )

    if config_a.get("seed") != endpoint_a_seed:
        raise ValueError("Endpoint A checkpoint config.seed differs from CLI seed")
    if config_b.get("seed") != endpoint_b_seed:
        raise ValueError("Endpoint B checkpoint config.seed differs from CLI seed")
    if config_a.get("output_dir") == config_b.get("output_dir"):
        raise ValueError("Endpoint output_dir values must be distinct")

    endpoint_config = require_mapping(
        training.get("endpoint_config"),
        "training.endpoint_config",
    )
    missing_protocol_keys = sorted(REQUIRED_ENDPOINT_CONFIG_KEYS - set(endpoint_config))
    if missing_protocol_keys:
        raise KeyError(
            "training.endpoint_config is missing required keys: "
            f"{missing_protocol_keys}"
        )
    for key, expected_value in endpoint_config.items():
        if key not in config_a or key not in config_b:
            raise KeyError(
                f"Protocol training.endpoint_config key {key!r} is absent from endpoints"
            )
        expected_json = json_compatible(
            expected_value,
            f"training.endpoint_config.{key}",
        )
        for label, config in (("A", config_a), ("B", config_b)):
            actual_json = json_compatible(
                config[key],
                f"endpoint {label} config.{key}",
            )
            if actual_json != expected_json:
                raise ValueError(
                    f"Endpoint {label} config.{key} violates protocol: "
                    f"expected={expected_value!r}, actual={config[key]!r}"
                )

    configured_epochs = require_non_bool_int(
        endpoint_config.get("epochs"),
        "training.endpoint_config.epochs",
        minimum=1,
    )
    if endpoint_epoch != configured_epochs:
        raise ValueError(
            f"Endpoint checkpoint epoch {endpoint_epoch} differs from "
            f"protocol epochs {configured_epochs}"
        )
    expected_rows = require_non_bool_int(
        endpoint_config.get("expected_train_rows"),
        "training.endpoint_config.expected_train_rows",
        minimum=1,
    )
    batch_size = require_non_bool_int(
        endpoint_config.get("batch_size"),
        "training.endpoint_config.batch_size",
        minimum=1,
    )

    for label, config in (("A", config_a), ("B", config_b)):
        data_path = resolved_existing_file(Path(str(config.get("data"))), f"endpoint {label} data")
        if data_path != mixed_archive_path:
            raise ValueError(
                f"Endpoint {label} config.data is not the bound mixed archive"
            )
        init_path = resolved_existing_file(
            Path(str(config.get("init_checkpoint"))),
            f"endpoint {label} init_checkpoint",
        )
        if init_path != source_path:
            raise ValueError(
                f"Endpoint {label} config.init_checkpoint is not the bound source"
            )
        if config.get("init_checkpoint_sha256") != source_sha256:
            raise ValueError(
                f"Endpoint {label} config.init_checkpoint_sha256 differs from source"
            )

    schedule = require_mapping(training.get("schedule"), "training.schedule")
    if schedule.get("algorithm") != SCHEDULE_ALGORITHM:
        raise ValueError(
            f"training.schedule.algorithm must be {SCHEDULE_ALGORITHM!r}"
        )
    steps_per_epoch = math.ceil(expected_rows / batch_size)
    total_steps = max(steps_per_epoch * configured_epochs, 1)
    warmup_steps = max(int(total_steps * WARMUP_FRACTION), 1)
    derived = {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    for key, expected in derived.items():
        observed = require_non_bool_int(
            schedule.get(key),
            f"training.schedule.{key}",
            minimum=1,
        )
        if observed != expected:
            raise ValueError(
                f"training.schedule.{key} mismatch: expected {expected}, got {observed}"
            )
    if "warmup_fraction" in schedule:
        require_exact_number(
            schedule["warmup_fraction"],
            WARMUP_FRACTION,
            "training.schedule.warmup_fraction",
        )
    if "minimum_lr_multiplier" in schedule:
        require_exact_number(
            schedule["minimum_lr_multiplier"],
            MINIMUM_LR_MULTIPLIER,
            "training.schedule.minimum_lr_multiplier",
        )
    return {
        "endpoint_config": json_compatible(endpoint_config, "training.endpoint_config"),
        "schedule": json_compatible(schedule, "training.schedule"),
        "derived_schedule": derived,
    }


def verify_reloaded_checkpoint(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    expected_state_sha256: str,
) -> None:
    if set(actual) != set(expected):
        raise RuntimeError("Saved checkpoint root keys changed after reload")
    state = actual.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Saved checkpoint has no model_state_dict mapping")
    if set(state) != set(expected_state):
        raise RuntimeError("Saved model_state_dict keys changed after reload")
    for name, expected_tensor in expected_state.items():
        actual_tensor = state[name]
        if not isinstance(actual_tensor, torch.Tensor):
            raise RuntimeError(f"Reloaded model-state entry {name!r} is not a tensor")
        if not tensors_bitwise_equal(actual_tensor, expected_tensor):
            raise RuntimeError(
                f"Reloaded model-state tensor {name!r} changed bitwise"
            )
    if model_state_sha256(state) != expected_state_sha256:
        raise RuntimeError("Reloaded model-state SHA-256 changed")
    expected_metadata = {
        key: value for key, value in expected.items() if key != "model_state_dict"
    }
    actual_metadata = {
        key: value for key, value in actual.items() if key != "model_state_dict"
    }
    if actual_metadata != expected_metadata:
        raise RuntimeError("Saved checkpoint metadata changed after reload")
    for forbidden in (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "optimizer",
        "scheduler",
    ):
        if forbidden in actual:
            raise RuntimeError(f"Deployment checkpoint unexpectedly contains {forbidden}")


def atomic_publish_checkpoint(
    checkpoint: dict[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    expected_state_sha256: str,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    published = False
    try:
        torch.save(checkpoint, temporary_path)
        temporary_reload = torch.load(
            temporary_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(temporary_reload, Mapping):
            raise RuntimeError("Temporary checkpoint root changed after reload")
        verify_reloaded_checkpoint(
            temporary_reload,
            checkpoint,
            expected_state,
            expected_state_sha256,
        )
        # A same-directory hard link gives O_EXCL semantics: link() fails if a
        # competing publisher has created output_path since the initial check.
        os.link(temporary_path, output_path)
        published = True
        temporary_path.unlink()
        output_reload = torch.load(
            output_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(output_reload, Mapping):
            raise RuntimeError("Published checkpoint root changed after reload")
        verify_reloaded_checkpoint(
            output_reload,
            checkpoint,
            expected_state,
            expected_state_sha256,
        )
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        if published and output_path.exists():
            output_path.unlink()
        raise


def create_source_anchored_endpoint_mean(
    *,
    protocol_path: Path,
    source_checkpoint_path: Path,
    endpoint_a_path: Path,
    endpoint_a_seed: int,
    endpoint_b_path: Path,
    endpoint_b_seed: int,
    mixed_archive_path: Path,
    trainer_path: Path,
    materializer_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    protocol_path = resolved_existing_file(Path(protocol_path), "protocol")
    source_checkpoint_path = resolved_existing_file(
        Path(source_checkpoint_path), "source checkpoint"
    )
    endpoint_a_path = resolved_existing_file(Path(endpoint_a_path), "endpoint A")
    endpoint_b_path = resolved_existing_file(Path(endpoint_b_path), "endpoint B")
    mixed_archive_path = resolved_existing_file(Path(mixed_archive_path), "mixed archive")
    trainer_path = resolved_existing_file(Path(trainer_path), "trainer")
    materializer_path = resolved_existing_file(Path(materializer_path), "materializer")
    if endpoint_a_path == endpoint_b_path:
        raise ValueError("Endpoint A and endpoint B paths must be distinct")
    if source_checkpoint_path in {endpoint_a_path, endpoint_b_path}:
        raise ValueError("Source and endpoint checkpoint paths must be distinct")

    endpoint_a_seed = require_non_bool_int(endpoint_a_seed, "endpoint A seed", minimum=0)
    endpoint_b_seed = require_non_bool_int(endpoint_b_seed, "endpoint B seed", minimum=0)
    if (endpoint_a_seed, endpoint_b_seed) != EXPECTED_SEEDS:
        raise ValueError(
            f"Endpoint seeds must equal preregistered ordered seeds {EXPECTED_SEEDS}; "
            f"got {(endpoint_a_seed, endpoint_b_seed)}"
        )

    protocol = load_protocol(protocol_path)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"Protocol schema_version must be {PROTOCOL_SCHEMA_VERSION!r}"
        )
    bindings = require_mapping(protocol.get("bindings"), "bindings")
    mean_candidate = require_mapping(
        bindings.get("mean_candidate"),
        "bindings.mean_candidate",
    )
    exact_mean_bindings: dict[str, Any] = {
        "formula": FORMULA,
        "alpha": ALPHA,
        "seeds": list(EXPECTED_SEEDS),
        "selected_scope": SELECTED_SCOPE,
        "expected_selected_tensors": EXPECTED_SELECTED_TENSORS,
        "expected_selected_parameters": EXPECTED_SELECTED_PARAMETERS,
        "expected_complement_tensors": EXPECTED_COMPLEMENT_TENSORS,
        "expected_complement_parameters": EXPECTED_COMPLEMENT_PARAMETERS,
    }
    for key, expected in exact_mean_bindings.items():
        observed = json_compatible(
            mean_candidate.get(key),
            f"bindings.mean_candidate.{key}",
        )
        if observed != expected:
            raise ValueError(
                f"bindings.mean_candidate.{key} mismatch: "
                f"expected={expected!r}, got={observed!r}"
            )
    if list(mean_candidate.get("seeds", [])) != [endpoint_a_seed, endpoint_b_seed]:
        raise ValueError("CLI endpoint seeds differ from protocol mean_candidate.seeds")

    inputs = require_mapping(bindings.get("inputs"), "bindings.inputs")
    source_identity = validate_bound_input(
        inputs,
        "source_checkpoint",
        source_checkpoint_path,
        protocol_path,
    )
    mixed_identity = validate_bound_input(
        inputs,
        "mixed_archive",
        mixed_archive_path,
        protocol_path,
    )
    trainer_identity = validate_bound_input(
        inputs,
        "trainer",
        trainer_path,
        protocol_path,
    )
    materializer_identity = validate_bound_input(
        inputs,
        "materializer",
        materializer_path,
        protocol_path,
    )
    tool_path = Path(__file__).resolve()
    helper_paths = {
        "block_selection": Path(block_utils.__file__).resolve(),
        "checkpoint_validation": Path(checkpoint_utils.__file__).resolve(),
    }
    composer_identity = validate_bound_input(
        inputs,
        "composer_tool",
        tool_path,
        protocol_path,
    )
    helper_identities = {
        label: validate_bound_input(
            inputs,
            label,
            helper_path,
            protocol_path,
        )
        for label, helper_path in helper_paths.items()
    }

    source = checkpoint_utils.load_checkpoint(source_checkpoint_path, "source")
    endpoint_a = checkpoint_utils.load_checkpoint(endpoint_a_path, "endpoint A")
    endpoint_b = checkpoint_utils.load_checkpoint(endpoint_b_path, "endpoint B")
    runtime_config, config_sources, deck_hash, ppo_interface, epochs = (
        shared_checkpoint_contract(source, endpoint_a, endpoint_b)
    )
    source_epoch, endpoint_a_epoch, endpoint_b_epoch = epochs
    assert endpoint_a_epoch == endpoint_b_epoch

    training = require_mapping(protocol.get("training"), "training")
    training_audit = validate_endpoint_training_bindings(
        endpoint_a,
        endpoint_b,
        training=training,
        source_path=source_checkpoint_path,
        source_sha256=source_identity["sha256"],
        mixed_archive_path=mixed_archive_path,
        endpoint_a_seed=endpoint_a_seed,
        endpoint_b_seed=endpoint_b_seed,
        endpoint_epoch=endpoint_a_epoch,
    )

    source_state = source.get("model_state_dict")
    endpoint_a_state = endpoint_a.get("model_state_dict")
    endpoint_b_state = endpoint_b.get("model_state_dict")
    if not isinstance(source_state, Mapping):
        raise TypeError("Source model_state_dict must be a mapping")
    if not isinstance(endpoint_a_state, Mapping):
        raise TypeError("Endpoint A model_state_dict must be a mapping")
    if not isinstance(endpoint_b_state, Mapping):
        raise TypeError("Endpoint B model_state_dict must be a mapping")
    mean_state, parameter_audit = source_anchored_mean_state(
        source_state,
        endpoint_a_state,
        endpoint_b_state,
    )

    source_state_sha = model_state_sha256(source_state)
    endpoint_a_state_sha = model_state_sha256(endpoint_a_state)
    endpoint_b_state_sha = model_state_sha256(endpoint_b_state)
    output_state_sha = model_state_sha256(mean_state)
    protocol_sha = checkpoint_utils.sha256_file(protocol_path)
    endpoint_a_file_sha = checkpoint_utils.sha256_file(endpoint_a_path)
    endpoint_b_file_sha = checkpoint_utils.sha256_file(endpoint_b_path)

    minimal_config: dict[str, Any] = {
        **runtime_config,
        "deck_hashes": [deck_hash],
    }
    provenance = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "formula": FORMULA,
        "alpha": ALPHA,
        "selected_scope": SELECTED_SCOPE,
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha,
            "schema_version": PROTOCOL_SCHEMA_VERSION,
        },
        "source_checkpoint": {
            **source_identity,
            "model_state_sha256": source_state_sha,
            "epoch": source_epoch,
        },
        "endpoint_a": {
            "path": str(endpoint_a_path),
            "sha256": endpoint_a_file_sha,
            "model_state_sha256": endpoint_a_state_sha,
            "seed": endpoint_a_seed,
            "epoch": endpoint_a_epoch,
        },
        "endpoint_b": {
            "path": str(endpoint_b_path),
            "sha256": endpoint_b_file_sha,
            "model_state_sha256": endpoint_b_state_sha,
            "seed": endpoint_b_seed,
            "epoch": endpoint_b_epoch,
        },
        "mixed_archive": mixed_identity,
        "trainer": trainer_identity,
        "materializer": materializer_identity,
        "composer_tool": composer_identity,
        "helper_tools": helper_identities,
        "feature_version": checkpoint_utils.bc.FEATURE_VERSION,
        "deck_hash": deck_hash,
        "runtime_model_config": copy.deepcopy(runtime_config),
        "config_provenance": {
            "strategy": "validated_shared_runtime_fields_only",
            "runtime_model_keys": list(checkpoint_utils.RUNTIME_MODEL_CONFIG_KEYS),
            "deck_hashes": "validated_equal_singleton",
            "checkpoint_config_sources": config_sources,
        },
        "training_protocol": training_audit,
        **parameter_audit,
        "output_model_state_sha256": output_state_sha,
        "output_epoch_policy": "shared_endpoint_epoch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid_metrics_null": True,
        "optimizer_state_omitted": True,
        "scheduler_state_omitted": True,
        "deployment_only": True,
        "resume_training": False,
    }
    output_checkpoint: dict[str, Any] = {
        "feature_version": checkpoint_utils.bc.FEATURE_VERSION,
        "config": minimal_config,
        "model_state_dict": mean_state,
        "epoch": endpoint_a_epoch,
        "valid_metrics": None,
        "ppo_interface": copy.deepcopy(ppo_interface),
        "source_anchored_endpoint_mean": provenance,
    }
    atomic_publish_checkpoint(
        output_checkpoint,
        mean_state,
        output_state_sha,
        output_path,
    )
    return {
        "output": str(output_path),
        "output_sha256": checkpoint_utils.sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_model_state_sha256": output_state_sha,
        "source_model_state_sha256": source_state_sha,
        "endpoint_a_model_state_sha256": endpoint_a_state_sha,
        "endpoint_b_model_state_sha256": endpoint_b_state_sha,
        "endpoint_seeds": [endpoint_a_seed, endpoint_b_seed],
        "epoch": endpoint_a_epoch,
        "deck_hash": deck_hash,
        "protocol_sha256": protocol_sha,
        "selected_tensor_count": parameter_audit["selected_tensor_count"],
        "selected_parameter_count": parameter_audit["selected_parameter_count"],
        "complement_tensor_count": parameter_audit["complement_tensor_count"],
        "complement_parameter_count": parameter_audit[
            "complement_parameter_count"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one deployment-only source-anchored mean of two "
            "preregistered transformer_actor BC endpoints."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--endpoint-a", type=Path, required=True)
    parser.add_argument("--endpoint-a-seed", type=int, required=True)
    parser.add_argument("--endpoint-b", type=Path, required=True)
    parser.add_argument("--endpoint-b-seed", type=int, required=True)
    parser.add_argument("--mixed-archive", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_source_anchored_endpoint_mean(
        protocol_path=args.protocol,
        source_checkpoint_path=args.source_checkpoint,
        endpoint_a_path=args.endpoint_a,
        endpoint_a_seed=args.endpoint_a_seed,
        endpoint_b_path=args.endpoint_b,
        endpoint_b_seed=args.endpoint_b_seed,
        mixed_archive_path=args.mixed_archive,
        trainer_path=args.trainer,
        materializer_path=args.materializer,
        output_path=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
