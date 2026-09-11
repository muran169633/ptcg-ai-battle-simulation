#!/usr/bin/env python3
"""Evaluate and select the frozen beta1.157 + FLG head10 P1/P2/P4/P8 sweep.

This launcher deliberately does not train.  It consumes a completed, hash-bound
training manifest, evaluates every one of the four published endpoints on the
same six specialist/retention panels, and only then selects the smallest fully
eligible step.  Evaluation uses ``tools/evaluate_policy_bc.py`` through the
PPO ``model_forward`` path (CUDA bfloat16 autocast) with policy-greedy ordering.

The training-manifest layout is intentionally not hard-coded.  A separately
hash-bound evaluation preregistration supplies RFC 6901 JSON pointers for the
manifest status and the four endpoint records.  Each record must nevertheless
contain the canonical ``step``, ``path``, ``sha256``, and
``model_state_sha256`` fields, and all four checkpoint identities are checked
again before any output is written.

Formal execution is one-shot.  It preclaims all 24 output files, runs all 24
commands even when an earlier child fails, writes a terminal manifest and a
decision, and seals the directory.  It contains no broad, Gold, packaging,
network, upload, or submission implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PREREGISTRATION_SCHEMA = (
    "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-"
    "preregistration-v1"
)
AUTHORIZATION_SCHEMA = (
    "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-"
    "formal-authorization-v1"
)
EXPECTED_STEPS = (1, 2, 4, 8)
EXPECTED_ENDPOINT_NAMES = ("p001", "p002", "p004", "p008")
EXPECTED_PANELS = (
    ("pokemonfan", "pokemonfan", None),
    ("flg", "flg", None),
    ("core5", "core5", None),
    ("dominic", "core5", "Dominic Peel"),
    ("luca", "core5", "Luca"),
    ("szlach", "core5", "szlachetny snieg"),
)
METRIC_KEYS = (
    "set",
    "hybrid",
    "ordered",
    "value",
    "count",
    "top1",
    "context34_hybrid",
    "context34_ordered",
)

# Absolute gates relative to the frozen U468 parent.  These are code-level
# minima: a preregistration can bind them exactly but cannot weaken them.
EXPECTED_PANEL_GATES: dict[str, dict[str, Any]] = {
    "pokemonfan": {
        "rows": 15152,
        "context34_rows": 58,
        "minimum": {
            "set": 13020,
            "hybrid": 13019,
            "ordered": 12869,
            "value": 11033,
            "count": 15022,
            "top1": 13119,
            "context34_hybrid": 57,
            "context34_ordered": 57,
        },
    },
    "flg": {
        "rows": 2312,
        "context34_rows": 3,
        "minimum": {
            "set": 1761,
            "hybrid": 1748,
            "ordered": 1731,
            "value": 1807,
            "count": 2299,
            "top1": 1775,
            "context34_hybrid": 3,
            "context34_ordered": 3,
        },
    },
    "core5": {
        "rows": 8319,
        "context34_rows": 43,
        "minimum": {
            "set": 6550,
            "hybrid": 6534,
            "ordered": 6505,
            "value": 6127,
            "count": 8210,
            "top1": 6653,
            "context34_hybrid": 41,
            "context34_ordered": 41,
        },
    },
    "dominic": {
        "rows": 2623,
        "context34_rows": 15,
        "minimum": {
            "set": 1981,
            "hybrid": 1974,
            "ordered": 1973,
            "value": 1940,
            "count": 2579,
            "top1": 2019,
            "context34_hybrid": 13,
            "context34_ordered": 13,
        },
    },
    "luca": {
        "rows": 1378,
        "context34_rows": 7,
        "minimum": {
            "set": 1069,
            "hybrid": 1069,
            "ordered": 1059,
            "value": 960,
            "count": 1377,
            "top1": 1086,
            "context34_hybrid": 7,
            "context34_ordered": 7,
        },
    },
    "szlach": {
        "rows": 2567,
        "context34_rows": 7,
        "minimum": {
            "set": 2071,
            "hybrid": 2071,
            "ordered": 2062,
            "value": 1906,
            "count": 2510,
            "top1": 2109,
            "context34_hybrid": 7,
            "context34_ordered": 7,
        },
    },
}

EXPECTED_SHARED_PROTOCOL = {
    "split": "valid",
    "split_mode": "archive",
    "split_seed": 20260723,
    "batch_size": 256,
    "workers": 8,
    "prediction_order_argument": "policy",
    "expected_prediction_order": "policy_greedy",
    "forward_precision": "cuda_bfloat16_autocast_via_train_ppo_model_forward",
    "device": "cuda",
    "compact": True,
    "progress_interval": 0,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_file_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def strict_json_loads(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite constant {value}")

    value = json.loads(
        payload,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def root_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty root-relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"{label} must be a safe root-relative path")
    path = ROOT / rel
    if path.resolve(strict=False) != path:
        raise ValueError(f"{label} is not normalized or traverses a symlink")
    return path


def normalize_cli_path(path: Path, label: str) -> Path:
    if path.is_absolute():
        resolved = path.resolve(strict=False)
        if resolved.parent != ROOT and ROOT not in resolved.parents:
            raise ValueError(f"{label} must be inside the frozen repository")
        return resolved
    return root_relative_path(str(path), label)


def read_plain_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"{label} must be a regular single-link file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError(f"{label} changed while held")
        if len(payload) != after.st_size:
            raise RuntimeError(f"{label} size changed while held")
        return payload, {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "device": after.st_dev,
            "inode": after.st_ino,
            "mode": oct(after.st_mode & 0o777),
            "nlink": after.st_nlink,
        }
    finally:
        os.close(fd)


def file_binding(binding: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, dict):
        raise TypeError(f"{label} binding must be an object")
    path = root_relative_path(binding.get("path"), f"{label}.path")
    _, evidence = read_plain_file(path, label)
    if evidence["sha256"] != binding.get("sha256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, evidence


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor model state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite model state entry: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_binding(binding: Any, label: str) -> tuple[Path, dict[str, Any]]:
    path, evidence = file_binding(binding, label)
    raw, _ = read_plain_file(path, label)
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), Mapping
    ):
        raise TypeError(f"{label} is not a model checkpoint")
    observed_model_sha = model_state_sha256(checkpoint["model_state_dict"])
    if observed_model_sha != binding.get("model_state_sha256"):
        raise ValueError(f"{label} model-state SHA-256 mismatch")
    evidence["model_state_sha256"] = observed_model_sha
    return path, evidence


def decode_json_pointer_token(token: str) -> str:
    index = 0
    result: list[str] = []
    while index < len(token):
        if token[index] != "~":
            result.append(token[index])
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise ValueError("Invalid RFC 6901 escape in JSON pointer")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def json_pointer_get(document: Any, pointer: Any, label: str) -> Any:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError(f"{label} must be an RFC 6901 JSON pointer")
    current = document
    if pointer == "":
        return current
    for raw_token in pointer[1:].split("/"):
        token = decode_json_pointer_token(raw_token)
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(f"{label} missing object token {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (token != "0" and token.startswith("0")):
                raise ValueError(f"{label} has invalid list token {token!r}")
            index = int(token)
            if index >= len(current):
                raise IndexError(f"{label} list token out of range: {token}")
            current = current[index]
        else:
            raise TypeError(f"{label} traverses a scalar at token {token!r}")
    return current


def validate_runtime() -> None:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise ValueError("Current working directory differs from frozen root")
    if Path(sys.executable).resolve(strict=True) != EXPECTED_PYTHON.resolve(strict=True):
        raise ValueError("Launcher must run under my_project_env Python")
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
        raise ValueError("Launcher requires exact -I -B flags")


def expected_command(
    *,
    evaluator: Path,
    checkpoint: Path,
    data: Path,
    team_name: str | None,
    output: Path,
) -> list[str]:
    command = [
        str(EXPECTED_PYTHON),
        "-I",
        "-B",
        str(evaluator),
        "--checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--data",
        str(data.relative_to(ROOT)),
        "--split",
        "valid",
        "--split-mode",
        "archive",
        "--split-seed",
        "20260723",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--prediction-order",
        "policy",
        "--device",
        "cuda",
        "--compact",
        "--progress-interval",
        "0",
        "--json-output",
        str(output.relative_to(ROOT)),
    ]
    if team_name is not None:
        command.extend(["--team-name", team_name])
    return command


def validate_protocol(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Path]]:
    validate_runtime()
    preregistration_path = normalize_cli_path(args.preregistration, "preregistration")
    preregistration_raw, preregistration_file = read_plain_file(
        preregistration_path,
        "evaluation preregistration",
    )
    preregistration_sha256 = preregistration_file["sha256"]
    if preregistration_sha256 != args.expected_preregistration_sha256:
        raise ValueError("Evaluation preregistration SHA-256 differs from CLI binding")
    protocol = strict_json_loads(preregistration_raw, "evaluation preregistration")
    if protocol.get("schema_version") != PREREGISTRATION_SCHEMA or (
        protocol.get("status") != "locked_before_single_zero_write_preflight"
    ):
        raise ValueError("Evaluation preregistration schema/status mismatch")
    if protocol.get("required_cwd") != str(ROOT):
        raise ValueError("Evaluation preregistration cwd mismatch")

    launcher = protocol.get("launcher")
    if not isinstance(launcher, dict):
        raise TypeError("Evaluation preregistration lacks launcher binding")
    launcher_path, launcher_file = file_binding(launcher, "launcher")
    if launcher_path != Path(__file__).resolve(strict=True):
        raise ValueError("Launcher binding differs from this executable")
    if launcher_file["sha256"] != launcher.get("sha256") or (
        launcher.get("python") != str(EXPECTED_PYTHON)
    ) or launcher.get("flags") != ["-I", "-B"]:
        raise ValueError("Launcher runtime/hash binding mismatch")

    training_binding = protocol.get("training_manifest")
    if not isinstance(training_binding, dict):
        raise TypeError("Evaluation preregistration lacks training manifest binding")
    cli_training_path = normalize_cli_path(args.training_manifest, "training manifest")
    protocol_training_path = root_relative_path(
        training_binding.get("path"),
        "training_manifest.path",
    )
    if cli_training_path != protocol_training_path or (
        args.expected_training_manifest_sha256 != training_binding.get("sha256")
    ):
        raise ValueError("Training manifest CLI binding differs from preregistration")
    training_raw, training_file = read_plain_file(
        protocol_training_path,
        "training manifest",
    )
    if training_file["sha256"] != args.expected_training_manifest_sha256:
        raise ValueError("Training manifest SHA-256 differs from CLI binding")
    training_manifest = strict_json_loads(training_raw, "training manifest")
    status = json_pointer_get(
        training_manifest,
        training_binding.get("status_json_pointer"),
        "training_manifest.status_json_pointer",
    )
    if training_binding.get("required_status") != (
        "training_completed_all_endpoints_published"
    ) or status != "training_completed_all_endpoints_published":
        raise ValueError("Training manifest is not in an accepted completed status")
    training_seal_binding = protocol.get("training_completion_seal")
    training_seal_path, _ = file_binding(
        training_seal_binding,
        "training completion seal",
    )
    training_seal_raw, _ = read_plain_file(
        training_seal_path,
        "training completion seal",
    )
    training_seal = strict_json_loads(
        training_seal_raw,
        "training completion seal",
    )
    if training_seal.get("schema_version") != training_manifest.get(
        "schema_version"
    ) or training_seal.get("branch") != training_manifest.get("branch") or (
        training_seal.get("endpoint_steps") != list(EXPECTED_STEPS)
    ) or training_seal.get("status") != "completed" or (
        training_seal.get("training_manifest")
        != {
            "path": str(protocol_training_path.relative_to(ROOT)),
            "sha256": training_file["sha256"],
            "bytes": training_file["bytes"],
            "mode": training_file["mode"],
            "inode": training_file["inode"],
            "device": training_file["device"],
            "nlink": training_file["nlink"],
            "descriptor_reload_exact": True,
            "visible_identity_exact": True,
        }
    ):
        raise ValueError("Training completion seal does not bind the completed manifest")

    evaluator_binding = protocol.get("evaluator")
    evaluator_path, _ = file_binding(evaluator_binding, "evaluator")
    if evaluator_path != ROOT / "tools" / "evaluate_policy_bc.py" or (
        evaluator_binding.get("python") != str(EXPECTED_PYTHON)
    ) or evaluator_binding.get("flags") != ["-I", "-B"]:
        raise ValueError("Evaluator path/runtime binding mismatch")
    dependencies = protocol.get("evaluator_dependencies")
    if not isinstance(dependencies, list) or len(dependencies) < 2:
        raise ValueError("At least train_ppo.py and train_bc_orbit.py must be bound")
    dependency_paths = [
        file_binding(binding, f"evaluator dependency {index}")[0]
        for index, binding in enumerate(dependencies)
    ]
    if ROOT / "tools" / "train_ppo.py" not in dependency_paths or (
        ROOT / "tools" / "train_bc_orbit.py" not in dependency_paths
    ):
        raise ValueError("Evaluator dependencies must bind bf16 PPO and BC sources")

    endpoint_bindings = protocol.get("endpoints")
    if not isinstance(endpoint_bindings, list) or len(endpoint_bindings) != 4:
        raise ValueError("Exactly four endpoint bindings are required")
    observed_endpoint_identity = [
        (item.get("name"), item.get("step"))
        if isinstance(item, dict)
        else (None, None)
        for item in endpoint_bindings
    ]
    if observed_endpoint_identity != list(zip(EXPECTED_ENDPOINT_NAMES, EXPECTED_STEPS)):
        raise ValueError("Endpoint identity/order must be P1/P2/P4/P8")
    endpoint_paths: dict[str, Path] = {}
    for item in endpoint_bindings:
        assert isinstance(item, dict)
        name = item["name"]
        record = json_pointer_get(
            training_manifest,
            item.get("manifest_record_json_pointer"),
            f"endpoint {name} training-manifest record",
        )
        if not isinstance(record, dict):
            raise TypeError(f"Endpoint {name} manifest record must be an object")
        expected_record = {
            "step": item["step"],
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "model_state_sha256": item.get("model_state_sha256"),
        }
        observed_record = {key: record.get(key) for key in expected_record}
        if observed_record != expected_record:
            raise ValueError(f"Endpoint {name} differs from its training-manifest record")
        path, _ = checkpoint_binding(item, f"endpoint {name}")
        endpoint_paths[name] = path
    if len(set(endpoint_paths.values())) != 4 or len({
        item["sha256"] for item in endpoint_bindings
    }) != 4 or len({
        item["model_state_sha256"] for item in endpoint_bindings
    }) != 4 or len({
        item["manifest_record_json_pointer"] for item in endpoint_bindings
    }) != 4:
        raise ValueError("P1/P2/P4/P8 endpoint paths, hashes, and pointers must be unique")

    data_bindings = protocol.get("data_bindings")
    if not isinstance(data_bindings, dict) or set(data_bindings) != {
        "pokemonfan",
        "flg",
        "core5",
    }:
        raise ValueError("Data binding names differ from the fixed six-panel design")
    data_paths = {
        name: file_binding(binding, f"data {name}")[0]
        for name, binding in data_bindings.items()
    }

    if protocol.get("authoritative_panel_gates") != EXPECTED_PANEL_GATES:
        raise ValueError("Authoritative panel gates differ from code-level floors")
    if protocol.get("shared_protocol") != EXPECTED_SHARED_PROTOCOL:
        raise ValueError("Shared official bf16/policy evaluator protocol mismatch")
    if protocol.get("selection_rule") != {
        "all_24_outputs_required_before_decision": True,
        "run_all_endpoints_without_early_stopping": True,
        "endpoint_requires_all_six_panels": True,
        "select": "smallest_step_among_fully_eligible_endpoints",
        "step_order": [1, 2, 4, 8],
        "none_eligible": "close_branch_without_broad_or_gold",
        "execution_failure": "no_selection_no_broad_no_gold",
    }:
        raise ValueError("Selection rule mismatch")
    if protocol.get("scope") != {
        "local_only": True,
        "network": False,
        "training": False,
        "specialist_behavior": True,
        "broad": False,
        "gold": False,
        "package": False,
        "upload": False,
        "submission": False,
    }:
        raise ValueError("Evaluation scope mismatch")

    output_rule = protocol.get("output_rule")
    if not isinstance(output_rule, dict):
        raise TypeError("Evaluation preregistration lacks output rule")
    output_root = root_relative_path(output_rule.get("root"), "output root")
    marker = root_relative_path(output_rule.get("attempt_marker"), "attempt marker")
    if output_root.parent != ROOT / "artifacts" or marker.parent != ROOT:
        raise ValueError("Output root and marker must use frozen direct parents")
    if output_root.exists() or output_root.is_symlink() or marker.exists() or marker.is_symlink():
        raise FileExistsError("Evaluation root or one-shot marker has already been consumed")
    if output_rule != {
        "root": str(output_root.relative_to(ROOT)),
        "attempt_marker": str(marker.relative_to(ROOT)),
        "evaluation_count_exact": 24,
        "attempts_per_evaluation": 1,
        "formal_attempts_authorized": 1,
        "retry_authorized": False,
        "run_all_before_decision": True,
        "manifest": "specialist_execution_manifest.json",
        "decision": "specialist_selection_decision.json",
        "completion_seal": "COMPLETED.json",
        "final_directory_mode": "0o500",
    }:
        raise ValueError("Output rule mismatch")

    evaluations = protocol.get("ordered_evaluations")
    expected_matrix = [
        (name, step, panel, data_name, team)
        for name, step in zip(EXPECTED_ENDPOINT_NAMES, EXPECTED_STEPS)
        for panel, data_name, team in EXPECTED_PANELS
    ]
    if not isinstance(evaluations, list) or len(evaluations) != len(expected_matrix):
        raise ValueError("Ordered evaluation matrix must contain exactly 24 entries")
    commands: list[list[str]] = []
    outputs: list[Path] = []
    for index, (item, expected) in enumerate(zip(evaluations, expected_matrix, strict=True)):
        if not isinstance(item, dict):
            raise TypeError(f"Evaluation {index + 1} must be an object")
        endpoint, step, panel, data_name, team = expected
        if {
            "order": item.get("order"),
            "endpoint": item.get("endpoint"),
            "step": item.get("step"),
            "panel": item.get("panel"),
            "team_name": item.get("team_name"),
        } != {
            "order": index + 1,
            "endpoint": endpoint,
            "step": step,
            "panel": panel,
            "team_name": team,
        }:
            raise ValueError(f"Evaluation {index + 1} identity/order mismatch")
        output = output_root / endpoint / f"{panel}.json"
        if root_relative_path(item.get("output"), f"evaluation {index + 1} output") != output:
            raise ValueError(f"Evaluation {index + 1} output mismatch")
        command = expected_command(
            evaluator=evaluator_path,
            checkpoint=endpoint_paths[endpoint],
            data=data_paths[data_name],
            team_name=team,
            output=output,
        )
        if item.get("command") != command or (
            item.get("command_sha256") != sha256_bytes(canonical_json_bytes(command))
        ) or item.get("attempts_authorized") != 1 or (
            item.get("expected_success_exit_code") != 0
        ):
            raise ValueError(f"Evaluation {index + 1} command/attempt binding mismatch")
        commands.append(command)
        outputs.append(output)
    if len(set(outputs)) != 24 or protocol.get("ordered_command_matrix_sha256") != (
        sha256_bytes(canonical_json_bytes(commands))
    ):
        raise ValueError("Ordered command matrix/output uniqueness mismatch")

    return protocol, training_manifest, preregistration_sha256, {
        "preregistration": preregistration_path,
        "training_manifest": protocol_training_path,
        "training_completion_seal": training_seal_path,
        "launcher": launcher_path,
        "evaluator": evaluator_path,
        "output_root": output_root,
        "marker": marker,
    }


def validate_authorization(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    preregistration_sha256: str,
) -> dict[str, Any]:
    if args.formal_authorization is None or not args.expected_formal_authorization_sha256:
        raise ValueError("Formal mode requires a hash-bound authorization")
    authorization_path = normalize_cli_path(
        args.formal_authorization,
        "formal authorization",
    )
    raw, evidence = read_plain_file(authorization_path, "formal authorization")
    if evidence["sha256"] != args.expected_formal_authorization_sha256:
        raise ValueError("Formal authorization SHA-256 differs from CLI binding")
    authorization = strict_json_loads(raw, "formal authorization")
    if authorization != {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "formal_execution_authorized",
        "preregistration_sha256": preregistration_sha256,
        "training_manifest_sha256": protocol["training_manifest"]["sha256"],
        "launcher_sha256": protocol["launcher"]["sha256"],
        "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
        "training_completion_seal_sha256": protocol["training_completion_seal"]["sha256"],
        "formal_attempts_authorized": 1,
        "retry_authorized": False,
        "broad_gold_package_upload_submission_authorized": False,
    }:
        raise ValueError("Formal authorization content mismatch")
    return evidence


def cuda_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA preflight failed: no visible CUDA device")
    probe = torch.empty(1, device="cuda")
    probe.fill_(1.0)
    if float(probe.item()) != 1.0:
        raise RuntimeError("CUDA preflight tensor returned an unexpected value")
    return {
        "available": True,
        "device_count": int(torch.cuda.device_count()),
        "device_index": int(torch.cuda.current_device()),
        "device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
    }


def hold_bound_input(binding: Mapping[str, Any], label: str) -> tuple[Path, int, str, str]:
    path = root_relative_path(binding.get("path"), f"{label}.path")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"{label} must be regular and single-link")
        payload = read_held_bytes(fd, label)
        digest = sha256_bytes(payload)
        if digest != binding.get("sha256"):
            raise RuntimeError(f"{label} SHA-256 mismatch")
        visible = os.stat(path, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != (visible.st_dev, visible.st_ino):
            raise RuntimeError(f"{label} path/descriptor identity mismatch")
        return path, fd, digest, label
    except BaseException:
        os.close(fd)
        raise


def read_held_bytes(fd: int, label: str) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    payload = b"".join(chunks)
    if len(payload) != os.fstat(fd).st_size:
        raise RuntimeError(f"{label} changed size while held")
    return payload


def revalidate_bound_input(item: tuple[Path, int, str, str]) -> None:
    path, fd, expected_sha256, label = item
    held = os.fstat(fd)
    visible = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(held.st_mode) or held.st_nlink != 1 or (
        held.st_dev,
        held.st_ino,
    ) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity changed")
    if sha256_bytes(read_held_bytes(fd, label)) != expected_sha256:
        raise RuntimeError(f"{label} held bytes changed")


def open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def claim_file(path: Path, mode: int = 0o600) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    os.fchmod(fd, mode)
    os.fsync(fd)
    return fd


def write_fd(fd: int, payload: bytes) -> None:
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    written = 0
    while written < len(view):
        written += os.write(fd, view[written:])
    os.fsync(fd)


def validate_held_output(
    *,
    path: Path,
    fd: int,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    held = os.fstat(fd)
    visible = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(held.st_mode) or held.st_nlink != 1 or (
        held.st_dev,
        held.st_ino,
    ) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"Output identity changed: {path}")
    os.fchmod(fd, 0o600)
    os.fsync(fd)
    payload = read_held_bytes(fd, str(path))
    digest = sha256_bytes(payload)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"Output SHA-256 changed: {path}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "device": held.st_dev,
        "inode": held.st_ino,
        "mode": "0o600",
        "nlink": held.st_nlink,
        "path_descriptor_identity": True,
    }


def validate_held_directory(
    *,
    path: Path,
    fd: int,
    expected_mode: int,
    expected_names: Sequence[str],
) -> dict[str, Any]:
    held = os.fstat(fd)
    visible = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(visible.st_mode) or (
        held.st_dev,
        held.st_ino,
    ) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"Directory path/descriptor identity changed: {path}")
    if (held.st_mode & 0o777) != expected_mode:
        raise RuntimeError(f"Directory mode mismatch: {path}")
    observed_names = sorted(os.listdir(fd))
    if observed_names != sorted(expected_names):
        raise RuntimeError(
            f"Directory contents mismatch for {path}: {observed_names}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "device": held.st_dev,
        "inode": held.st_ino,
        "mode": oct(held.st_mode & 0o777),
        "nlink": held.st_nlink,
        "contents": observed_names,
        "path_descriptor_identity": True,
    }


def metric_projection(result: dict[str, Any]) -> dict[str, int]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Evaluator output has no metrics object")
    context = metrics.get("by_context")
    if not isinstance(context, dict):
        raise ValueError("Evaluator output has no by_context object")
    context34 = context.get("34", {})
    if not isinstance(context34, dict):
        raise ValueError("Evaluator output has invalid context 34 metrics")
    projection = {
        "rows": metrics.get("rows"),
        "context34_rows": context34.get("rows", 0),
        "set": metrics.get("set_exact_correct"),
        "hybrid": metrics.get("hybrid_order_exact_correct"),
        "ordered": metrics.get("ordered_exact_correct"),
        "value": metrics.get("value_correct"),
        "count": metrics.get("count_correct"),
        "top1": metrics.get("top1_correct"),
        "context34_hybrid": context34.get("hybrid_order_exact_correct", 0),
        "context34_ordered": context34.get("ordered_exact_correct", 0),
    }
    if any(type(value) is not int or value < 0 for value in projection.values()):
        raise ValueError("Evaluator output metric projection is non-integer/negative")
    return projection


def validate_evaluator_output(
    *,
    payload: bytes,
    stdout: bytes,
    evaluation: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    if payload != stdout:
        raise ValueError("Evaluator stdout differs from --json-output bytes")
    result = strict_json_loads(payload, "evaluator output")
    endpoint = next(
        item for item in protocol["endpoints"]
        if item["name"] == evaluation["endpoint"]
    )
    panel = evaluation["panel"]
    data_name = "core5" if panel in {"core5", "dominic", "luca", "szlach"} else panel
    data = protocol["data_bindings"][data_name]
    expected_team_names = [] if evaluation["team_name"] is None else [evaluation["team_name"]]
    if Path(result.get("checkpoint", "")) != root_relative_path(
        endpoint["path"],
        "evaluator checkpoint",
    ) or result.get("checkpoint_sha256") != endpoint["sha256"]:
        raise ValueError("Evaluator output checkpoint binding mismatch")
    if Path(result.get("data", "")) != root_relative_path(
        data["path"],
        "evaluator data",
    ):
        raise ValueError("Evaluator output data binding mismatch")
    if {
        "split": result.get("split"),
        "split_mode": result.get("split_mode"),
        "split_seed": result.get("split_seed"),
        "prediction_order": result.get("prediction_order"),
        "device": result.get("device"),
    } != {
        "split": "valid",
        "split_mode": "archive",
        "split_seed": 20260723,
        "prediction_order": "policy_greedy",
        "device": "cuda",
    }:
        raise ValueError("Evaluator output protocol fields mismatch")
    filters = result.get("filters")
    if not isinstance(filters, dict) or filters.get("deck_hashes") != [] or (
        filters.get("team_names") != expected_team_names
    ):
        raise ValueError("Evaluator output filter mismatch")
    projection = metric_projection(result)
    gate = EXPECTED_PANEL_GATES[panel]
    if projection["rows"] != gate["rows"] or (
        projection["context34_rows"] != gate["context34_rows"]
    ):
        raise ValueError("Evaluator output row/context shape differs from frozen panel")
    return result, projection


def evaluate_gate(metrics: dict[str, int], panel: str) -> dict[str, Any]:
    expected = EXPECTED_PANEL_GATES[panel]
    checks = {
        "rows_exact": {
            "actual": metrics["rows"],
            "required": expected["rows"],
            "pass": metrics["rows"] == expected["rows"],
        },
        "context34_rows_exact": {
            "actual": metrics["context34_rows"],
            "required": expected["context34_rows"],
            "pass": metrics["context34_rows"] == expected["context34_rows"],
        },
    }
    checks.update({
        f"{key}_minimum": {
            "actual": metrics[key],
            "required": expected["minimum"][key],
            "pass": metrics[key] >= expected["minimum"][key],
        }
        for key in METRIC_KEYS
    })
    return {
        "panel": panel,
        "pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
    }


def formal_run(
    *,
    protocol: dict[str, Any],
    preregistration_sha256: str,
    paths: dict[str, Path],
    authorization: dict[str, Any],
    cuda: dict[str, Any],
) -> dict[str, Any]:
    output_root = paths["output_root"]
    marker = paths["marker"]
    created_at = datetime.now(timezone.utc).isoformat()
    marker_fd = -1
    output_root_fd = -1
    manifest_fd = -1
    decision_fd = -1
    completion_fd = -1
    output_claims: dict[int, tuple[Path, int]] = {}
    endpoint_fds: dict[str, int] = {}
    held_inputs: list[tuple[Path, int, str, str]] = []
    try:
        immutable_bindings: list[tuple[Mapping[str, Any], str]] = [
            ({"path": str(paths["preregistration"].relative_to(ROOT)), "sha256": preregistration_sha256}, "preregistration"),
            (protocol["training_manifest"], "training manifest"),
            (protocol["training_completion_seal"], "training completion seal"),
            (protocol["launcher"], "launcher"),
            (protocol["evaluator"], "evaluator"),
            *[(item, f"evaluator dependency {index}") for index, item in enumerate(protocol["evaluator_dependencies"])],
            *[(item, f"endpoint {item['name']}") for item in protocol["endpoints"]],
            *[(item, f"data {name}") for name, item in protocol["data_bindings"].items()],
            (authorization, "formal authorization"),
        ]
        held_inputs = [hold_bound_input(binding, label) for binding, label in immutable_bindings]
        for item in held_inputs:
            revalidate_bound_input(item)

        marker_fd = claim_file(marker)
        marker_payload = canonical_json_file_bytes({
            "schema_version": "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-attempt-v1",
            "status": "formal_attempt_claimed",
            "created_at_utc": created_at,
            "attempt": 1,
            "attempts_authorized": 1,
            "retry_authorized": False,
            "preregistration_sha256": preregistration_sha256,
            "training_manifest_sha256": protocol["training_manifest"]["sha256"],
            "training_completion_seal_sha256": protocol["training_completion_seal"]["sha256"],
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "formal_authorization_sha256": authorization["sha256"],
        })
        write_fd(marker_fd, marker_payload)
        _, marker_evidence = validate_held_output(path=marker, fd=marker_fd)

        os.mkdir(output_root, mode=0o700)
        output_root_fd = open_directory(output_root)
        os.fchmod(output_root_fd, 0o700)
        for endpoint in EXPECTED_ENDPOINT_NAMES:
            endpoint_dir = output_root / endpoint
            os.mkdir(endpoint_dir, mode=0o700)
            endpoint_fd = open_directory(endpoint_dir)
            os.fchmod(endpoint_fd, 0o700)
            endpoint_fds[endpoint] = endpoint_fd
        os.fsync(output_root_fd)

        for evaluation in protocol["ordered_evaluations"]:
            output_path = root_relative_path(evaluation["output"], "evaluation output")
            output_claims[evaluation["order"]] = (output_path, claim_file(output_path))

        results: list[dict[str, Any]] = []
        # This loop intentionally has no break or metric-based branch.  Every
        # endpoint/panel command consumes its one authorized attempt.
        for evaluation in protocol["ordered_evaluations"]:
            started = datetime.now(timezone.utc)
            record: dict[str, Any] = {
                "order": evaluation["order"],
                "endpoint": evaluation["endpoint"],
                "step": evaluation["step"],
                "panel": evaluation["panel"],
                "command_sha256": evaluation["command_sha256"],
                "attempt_consumed": True,
            }
            output_path, output_fd = output_claims[evaluation["order"]]
            completed: subprocess.CompletedProcess[bytes] | None = None
            try:
                for item in held_inputs:
                    revalidate_bound_input(item)
                completed = subprocess.run(
                    evaluation["command"],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                for item in held_inputs:
                    revalidate_bound_input(item)
                record.update({
                    "terminal_exit_code": completed.returncode,
                    "stdout_bytes": len(completed.stdout),
                    "stdout_sha256": sha256_bytes(completed.stdout),
                    "stderr_bytes": len(completed.stderr),
                    "stderr_sha256": sha256_bytes(completed.stderr),
                    "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-2000:],
                })
                if completed.returncode != 0:
                    raise RuntimeError(f"Evaluator exited {completed.returncode}")
                output_raw, output_evidence = validate_held_output(
                    path=output_path,
                    fd=output_fd,
                )
                _, metrics = validate_evaluator_output(
                    payload=output_raw,
                    stdout=completed.stdout,
                    evaluation=evaluation,
                    protocol=protocol,
                )
                record.update({
                    "status": "completed",
                    "output": output_evidence,
                    "metrics": metrics,
                })
            except BaseException as error:
                failure = {
                    "schema_version": "ptcg-specialist-evaluation-terminal-failure-v1",
                    "status": "failed",
                    "order": evaluation["order"],
                    "endpoint": evaluation["endpoint"],
                    "step": evaluation["step"],
                    "panel": evaluation["panel"],
                    "command_sha256": evaluation["command_sha256"],
                    "terminal_exit_code": None if completed is None else completed.returncode,
                    "error": f"{type(error).__name__}: {error}",
                }
                write_fd(output_fd, canonical_json_file_bytes(failure))
                _, output_evidence = validate_held_output(path=output_path, fd=output_fd)
                record.update({
                    "status": "failed",
                    "error": failure["error"],
                    "output": output_evidence,
                })
            record["elapsed_seconds"] = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
            results.append(record)

        if len(results) != 24:
            raise RuntimeError("Internal error: formal loop did not produce 24 records")
        for evaluation, record in zip(protocol["ordered_evaluations"], results, strict=True):
            output_path, output_fd = output_claims[evaluation["order"]]
            validate_held_output(
                path=output_path,
                fd=output_fd,
                expected_sha256=record["output"]["sha256"],
            )
        for item in held_inputs:
            revalidate_bound_input(item)

        all_completed = all(record.get("status") == "completed" for record in results)
        manifest = {
            "schema_version": "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-manifest-v1",
            "status": "evaluations_completed" if all_completed else "terminal_evaluation_failure",
            "created_at_utc": created_at,
            "preregistration_sha256": preregistration_sha256,
            "training_manifest": {
                "path": protocol["training_manifest"]["path"],
                "sha256": protocol["training_manifest"]["sha256"],
                "required_status": protocol["training_manifest"]["required_status"],
            },
            "training_completion_seal": protocol["training_completion_seal"],
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "formal_authorization_sha256": authorization["sha256"],
            "attempt_marker": marker_evidence,
            "cuda_preflight": cuda,
            "evaluation_count_required": 24,
            "evaluations_attempted": len(results),
            "evaluations_completed": sum(record.get("status") == "completed" for record in results),
            "run_all_before_decision": True,
            "no_early_stopping": True,
            "results": results,
        }
        manifest_path = output_root / protocol["output_rule"]["manifest"]
        manifest_fd = claim_file(manifest_path)
        manifest_payload = canonical_json_file_bytes(manifest)
        write_fd(manifest_fd, manifest_payload)
        _, manifest_evidence = validate_held_output(path=manifest_path, fd=manifest_fd)

        endpoint_decisions: list[dict[str, Any]] = []
        eligible_steps: list[int] = []
        if all_completed:
            for endpoint, step in zip(EXPECTED_ENDPOINT_NAMES, EXPECTED_STEPS):
                endpoint_results = [
                    record for record in results if record["endpoint"] == endpoint
                ]
                panel_gates = {
                    record["panel"]: evaluate_gate(record["metrics"], record["panel"])
                    for record in endpoint_results
                }
                eligible = len(endpoint_results) == 6 and all(
                    panel_gates[panel]["pass"]
                    for panel, _, _ in EXPECTED_PANELS
                )
                if eligible:
                    eligible_steps.append(step)
                endpoint_decisions.append({
                    "endpoint": endpoint,
                    "step": step,
                    "eligible": eligible,
                    "panel_gates": panel_gates,
                })
        else:
            endpoint_decisions = [
                {
                    "endpoint": endpoint,
                    "step": step,
                    "eligible": False,
                    "reason": "one_or_more_of_24_evaluations_failed",
                }
                for endpoint, step in zip(EXPECTED_ENDPOINT_NAMES, EXPECTED_STEPS)
            ]

        selected_step = min(eligible_steps) if eligible_steps else None
        selected_endpoint = None
        if selected_step is not None:
            selected_endpoint = next(
                {
                    "name": item["name"],
                    "step": item["step"],
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "model_state_sha256": item["model_state_sha256"],
                }
                for item in protocol["endpoints"]
                if item["step"] == selected_step
            )
        if not all_completed:
            decision_status = "terminal_evaluation_failure_no_selection"
        elif selected_endpoint is None:
            decision_status = "failed_no_eligible_endpoint_branch_closed"
        else:
            decision_status = "passed_smallest_fully_eligible_step_selected"
        decision = {
            "schema_version": "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-decision-v1",
            "status": decision_status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": preregistration_sha256,
            "training_manifest_sha256": protocol["training_manifest"]["sha256"],
            "training_completion_seal_sha256": protocol["training_completion_seal"]["sha256"],
            "manifest": manifest_evidence,
            "all_24_outputs_completed_before_decision": all_completed,
            "selection_rule": "smallest_step_among_fully_eligible_endpoints",
            "eligible_steps": eligible_steps,
            "selected_endpoint": selected_endpoint,
            "endpoint_decisions": endpoint_decisions,
            "downstream": {
                "separate_broad_evaluation_authorized": selected_endpoint is not None,
                "broad_executed_by_this_tool": False,
                "gold_authorized": False,
                "gold_executed": False,
                "package_upload_submission_authorized": False,
            },
        }
        decision_path = output_root / protocol["output_rule"]["decision"]
        decision_fd = claim_file(decision_path)
        decision_payload = canonical_json_file_bytes(decision)
        write_fd(decision_fd, decision_payload)
        _, decision_evidence = validate_held_output(path=decision_path, fd=decision_fd)

        completion_path = output_root / protocol["output_rule"]["completion_seal"]
        completion_fd = claim_file(completion_path)
        empty_completion, _ = validate_held_output(
            path=completion_path,
            fd=completion_fd,
        )
        if empty_completion:
            raise RuntimeError("Completion seal preclaim was not empty")

        for endpoint in EXPECTED_ENDPOINT_NAMES:
            endpoint_path = output_root / endpoint
            validate_held_directory(
                path=endpoint_path,
                fd=endpoint_fds[endpoint],
                expected_mode=0o700,
                expected_names=[f"{panel}.json" for panel, _, _ in EXPECTED_PANELS],
            )
            os.fchmod(endpoint_fds[endpoint], 0o500)
            os.fsync(endpoint_fds[endpoint])
        expected_root_names = sorted([
            *EXPECTED_ENDPOINT_NAMES,
            protocol["output_rule"]["manifest"],
            protocol["output_rule"]["decision"],
            protocol["output_rule"]["completion_seal"],
        ])
        validate_held_directory(
            path=output_root,
            fd=output_root_fd,
            expected_mode=0o700,
            expected_names=expected_root_names,
        )
        os.fchmod(output_root_fd, 0o500)
        os.fsync(output_root_fd)

        # Terminal seal: after all publications and directory mode changes,
        # rebind every directory and every held output descriptor to its visible
        # path and exact SHA.  The authoritative completion record is not
        # returned until this full closed-world check passes.
        endpoint_directories = {
            endpoint: validate_held_directory(
                path=output_root / endpoint,
                fd=endpoint_fds[endpoint],
                expected_mode=0o500,
                expected_names=[f"{panel}.json" for panel, _, _ in EXPECTED_PANELS],
            )
            for endpoint in EXPECTED_ENDPOINT_NAMES
        }
        output_directory = validate_held_directory(
            path=output_root,
            fd=output_root_fd,
            expected_mode=0o500,
            expected_names=expected_root_names,
        )
        for evaluation, record in zip(
            protocol["ordered_evaluations"],
            results,
            strict=True,
        ):
            output_path, output_fd = output_claims[evaluation["order"]]
            validate_held_output(
                path=output_path,
                fd=output_fd,
                expected_sha256=record["output"]["sha256"],
            )
        validate_held_output(
            path=manifest_path,
            fd=manifest_fd,
            expected_sha256=manifest_evidence["sha256"],
        )
        validate_held_output(
            path=decision_path,
            fd=decision_fd,
            expected_sha256=decision_evidence["sha256"],
        )
        validate_held_output(
            path=marker,
            fd=marker_fd,
            expected_sha256=marker_evidence["sha256"],
        )
        for item in held_inputs:
            revalidate_bound_input(item)

        # Populate the previously claimed empty completion file only after the
        # exact tree, every evaluator output, manifest, decision, marker, all
        # immutable inputs, and sealed directory identities have passed.
        completion = {
            "schema_version": "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-completion-v1",
            "status": (
                "completed"
                if all_completed
                else "terminal_evaluation_failure_no_retry"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_sha256": preregistration_sha256,
            "training_manifest_sha256": protocol["training_manifest"]["sha256"],
            "training_completion_seal_sha256": protocol["training_completion_seal"]["sha256"],
            "manifest": manifest_evidence,
            "decision": decision_evidence,
            "evaluations_attempted": 24,
            "evaluations_completed": manifest["evaluations_completed"],
            "selected_endpoint": selected_endpoint,
            "retry_authorized": False,
            "broad_executed": False,
            "gold_package_upload_submission_authorized": False,
            "output_directory": output_directory,
            "endpoint_directories": endpoint_directories,
        }
        write_fd(completion_fd, canonical_json_file_bytes(completion))
        _, completion_evidence = validate_held_output(
            path=completion_path,
            fd=completion_fd,
        )
        validate_held_directory(
            path=output_root,
            fd=output_root_fd,
            expected_mode=0o500,
            expected_names=expected_root_names,
        )
        for item in held_inputs:
            revalidate_bound_input(item)
        return {
            **completion,
            "completion_publication": completion_evidence,
            "decision_status": decision_status,
            "output_directory": output_directory,
            "endpoint_directories": endpoint_directories,
        }
    finally:
        for _, fd in output_claims.values():
            try:
                os.close(fd)
            except OSError:
                pass
        for _, fd, _, _ in held_inputs:
            try:
                os.close(fd)
            except OSError:
                pass
        for fd in endpoint_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        for fd in (completion_fd, decision_fd, manifest_fd, output_root_fd, marker_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--expected-training-manifest-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--formal-authorization", type=Path)
    parser.add_argument("--expected-formal-authorization-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    protocol, _, preregistration_sha256, paths = validate_protocol(args)
    cuda = cuda_preflight()
    if args.mode == "preflight":
        if args.formal_authorization is not None or (
            args.expected_formal_authorization_sha256 is not None
        ):
            raise ValueError("Preflight must not receive formal authorization")
        for target in (paths["marker"], paths["output_root"]):
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Preflight target appeared: {target}")
        print(json.dumps({
            "schema_version": "ptcg-u468-beta1157-flg-fresh-adamw-head10-specialist-preflight-v1",
            "status": "preflight_passed_no_writes",
            "preregistration_sha256": preregistration_sha256,
            "training_manifest_sha256": protocol["training_manifest"]["sha256"],
            "training_completion_seal_sha256": protocol["training_completion_seal"]["sha256"],
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "evaluation_count": 24,
            "cuda_preflight": cuda,
            "writes": 0,
            "targets_remained_absent": True,
        }, ensure_ascii=False, sort_keys=True, indent=2))
        return
    authorization = validate_authorization(args, protocol, preregistration_sha256)
    result = formal_run(
        protocol=protocol,
        preregistration_sha256=preregistration_sha256,
        paths=paths,
        authorization=authorization,
        cuda=cuda,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
