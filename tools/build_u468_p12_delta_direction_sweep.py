#!/usr/bin/env python3
"""Build a frozen evaluation-only U468 P12-delta direction sweep.

For the ten actor/count tensors changed by the historical P12 repair, form

    D_beta = (1 - beta) * D_current + beta * D_historical
    W_beta = W_U468 + D_beta

All other model tensors are copied byte-for-byte from U468.  The emitted
checkpoints intentionally contain no optimizer or quota state and must never
be used to resume training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
SCHEMA = "ptcg-u468-p12-delta-direction-sweep-design-v1"
MUTABLE_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
    "count_head.0.weight",
    "count_head.0.bias",
    "count_head.2.weight",
    "count_head.2.bias",
)
EXPECTED_ENDPOINTS = (
    ("beta050", 0.50, "transport-beta-050.pt"),
    ("beta075", 0.75, "transport-beta-075.pt"),
    ("beta100", 1.00, "transport-beta-100.pt"),
)
REQUIRED_SLIM_KEYS = (
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "action_distribution",
)


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json_loads(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    value = json.loads(
        payload,
        object_pairs_hook=reject_duplicate_pairs,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def model_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Non-tensor model state entry: {name}")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
        identity = (before.st_dev, before.st_ino, before.st_size)
        if identity != (after.st_dev, after.st_ino, after.st_size):
            raise RuntimeError(f"{label} changed while its descriptor was held")
        if len(payload) != after.st_size:
            raise RuntimeError(f"{label} size changed while reading")
        evidence = {
            "sha256": bytes_sha256(payload),
            "bytes": len(payload),
            "device": after.st_dev,
            "inode": after.st_ino,
            "mode": oct(after.st_mode & 0o777),
            "nlink": after.st_nlink,
        }
        return payload, evidence
    finally:
        os.close(fd)


def relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"{label} must be a safe relative path")
    path = ROOT / rel
    if path.resolve(strict=False) != path:
        raise ValueError(f"{label} resolves through a symlink or is not normalized")
    return path


def load_checkpoint_bytes(payload: bytes, label: str) -> dict[str, Any]:
    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a checkpoint dict")
    state = value.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError(f"{label}.model_state_dict must be a mapping")
    return value


def assert_state_compatible(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    label: str,
) -> None:
    if list(reference) != list(candidate):
        raise ValueError(f"{label} model-state key order/schema mismatch")
    for name in reference:
        left = reference[name]
        right = candidate[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"{label} contains non-tensor model state {name!r}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"{label} tensor schema mismatch for {name!r}")
        if left.layout != torch.strided or right.layout != torch.strided:
            raise ValueError(f"{label} tensor {name!r} must use strided layout")


def changed_tensor_names(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> list[str]:
    return sorted(name for name in left if not torch.equal(left[name], right[name]))


def flat_delta(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return torch.cat(
        [
            (target[name].detach().cpu().double() - source[name].detach().cpu().double()).reshape(-1)
            for name in MUTABLE_NAMES
        ]
    )


def build_endpoint_state(
    *,
    beta: float,
    new_parent: Mapping[str, torch.Tensor],
    new_exact: Mapping[str, torch.Tensor],
    old_parent: Mapping[str, torch.Tensor],
    old_p12: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = {name: tensor.detach().cpu().clone() for name, tensor in new_parent.items()}
    for name in MUTABLE_NAMES:
        base = new_parent[name].detach().cpu()
        if not base.is_floating_point() or not bool(torch.isfinite(base).all()):
            raise ValueError(f"Mutable tensor {name!r} must be finite floating point")
        current_delta = new_exact[name].detach().cpu().double() - base.double()
        historical_delta = (
            old_p12[name].detach().cpu().double()
            - old_parent[name].detach().cpu().double()
        )
        transported = (
            base.double()
            + (1.0 - beta) * current_delta
            + beta * historical_delta
        ).to(dtype=base.dtype)
        if not bool(torch.isfinite(transported).all()):
            raise ValueError(f"Endpoint tensor {name!r} is not finite")
        output[name] = transported
    return output


def publish_o_excl_at(
    *,
    directory_fd: int,
    name: str,
    payload: bytes,
    label: str,
    expected_model_state_sha256: str | None = None,
) -> tuple[dict[str, Any], int]:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} name must be one normalized path component")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode):
            raise RuntimeError(f"{label} is not regular after publication")
        if (current.st_mode & 0o777) != 0o600 or current.st_nlink != 1:
            raise RuntimeError(f"{label} publication mode/nlink mismatch")
        if current.st_size != len(payload):
            raise RuntimeError(f"{label} publication size mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        reloaded_payload = b"".join(chunks)
        if reloaded_payload != payload:
            raise RuntimeError(f"{label} descriptor-held byte reload mismatch")
        if expected_model_state_sha256 is not None:
            checkpoint = torch.load(
                io.BytesIO(reloaded_payload),
                map_location="cpu",
                weights_only=False,
            )
            if model_state_sha256(checkpoint["model_state_dict"]) != (
                expected_model_state_sha256
            ):
                raise RuntimeError(f"{label} descriptor-held model reload mismatch")
            if checkpoint.get("evaluation_only") is not True or (
                checkpoint.get("resume_forbidden") is not True
            ):
                raise RuntimeError(f"{label} lost evaluation-only resume gates")
            forbidden = {
                "optimizer_state_dict",
                "bc_replay_optimizer_state_dict",
                "opponent_quota_state",
            }
            if forbidden.intersection(checkpoint):
                raise RuntimeError(f"{label} unexpectedly contains resume state")
        else:
            strict_json_loads(reloaded_payload, label)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (entry.st_dev, entry.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError(f"{label} path/descriptor identity mismatch")
        evidence = {
            "sha256": bytes_sha256(reloaded_payload),
            "bytes": current.st_size,
            "device": current.st_dev,
            "inode": current.st_ino,
            "mode": oct(current.st_mode & 0o777),
            "nlink": current.st_nlink,
            "descriptor_held_exact_reload": True,
            "path_descriptor_identity": True,
        }
        return evidence, fd
    except BaseException:
        os.close(fd)
        raise


def revalidate_published_at(
    *,
    directory_fd: int,
    name: str,
    file_fd: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    held = os.fstat(file_fd)
    visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path no longer names its held descriptor")
    if not stat.S_ISREG(held.st_mode) or (held.st_mode & 0o777) != 0o600:
        raise RuntimeError(f"{label} final type/mode mismatch")
    if held.st_nlink != 1 or visible.st_nlink != 1:
        raise RuntimeError(f"{label} final nlink mismatch")
    os.lseek(file_fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    payload = b"".join(chunks)
    digest = bytes_sha256(payload)
    if digest != expected_sha256 or len(payload) != held.st_size:
        raise RuntimeError(f"{label} final held-byte hash/size mismatch")
    return {
        "sha256": digest,
        "bytes": len(payload),
        "device": held.st_dev,
        "inode": held.st_ino,
        "mode": oct(held.st_mode & 0o777),
        "nlink": held.st_nlink,
        "path_descriptor_identity": True,
        "descriptor_held_final_revalidation": True,
    }


def claim_empty_o_excl_at(
    *,
    directory_fd: int,
    name: str,
    label: str,
) -> tuple[dict[str, Any], int]:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} name must be one normalized path component")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        os.fsync(fd)
        held = os.fstat(fd)
        visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
            raise RuntimeError(f"{label} claim path/descriptor identity mismatch")
        if not stat.S_ISREG(held.st_mode) or (held.st_mode & 0o777) != 0o600:
            raise RuntimeError(f"{label} claim type/mode mismatch")
        if held.st_nlink != 1 or held.st_size != 0:
            raise RuntimeError(f"{label} claim nlink/size mismatch")
        return {
            "device": held.st_dev,
            "inode": held.st_ino,
            "mode": oct(held.st_mode & 0o777),
            "nlink": held.st_nlink,
            "empty_claimed": True,
            "path_descriptor_identity": True,
        }, fd
    except BaseException:
        os.close(fd)
        raise


def complete_claimed_json(
    *,
    directory_fd: int,
    name: str,
    file_fd: int,
    payload: bytes,
    label: str,
) -> dict[str, Any]:
    before = os.fstat(file_fd)
    visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (before.st_dev, before.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} no longer names its empty held claim")
    if before.st_size != 0:
        raise RuntimeError(f"{label} empty claim was unexpectedly populated")
    os.lseek(file_fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    written = 0
    while written < len(view):
        written += os.write(file_fd, view[written:])
    os.fsync(file_fd)
    result = revalidate_published_at(
        directory_fd=directory_fd,
        name=name,
        file_fd=file_fd,
        expected_sha256=bytes_sha256(payload),
        label=label,
    )
    strict_json_loads(payload, label)
    return result


def revalidate_directory_at(
    *,
    parent_fd: int,
    name: str,
    directory_fd: int,
    expected_mode: int | None,
    label: str,
) -> dict[str, Any]:
    held = os.fstat(directory_fd)
    visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity mismatch")
    if not stat.S_ISDIR(held.st_mode):
        raise RuntimeError(f"{label} is not a directory")
    mode = held.st_mode & 0o777
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode mismatch: {oct(mode)}")
    return {
        "device": held.st_dev,
        "inode": held.st_ino,
        "mode": oct(mode),
        "nlink": held.st_nlink,
        "path_descriptor_identity": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--formal-authorization", type=Path)
    parser.add_argument("--expected-formal-authorization-sha256")
    return parser.parse_args(argv)


def load_and_validate_design(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise ValueError("Current working directory differs from the frozen root")
    if args.design.is_absolute():
        raise ValueError("--design must use the frozen root-relative path")
    design_path = relative_path(str(args.design), "design")
    raw, design_file_evidence = read_plain_file(design_path, "design")
    digest = design_file_evidence["sha256"]
    if digest != args.expected_design_sha256:
        raise ValueError("Design SHA-256 differs from the expected CLI binding")
    design = strict_json_loads(raw, "design")
    if design.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported design schema")
    builder = design.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("Design has no builder binding")
    builder_path = relative_path(builder.get("path"), "builder.path")
    if builder_path != Path(__file__).resolve(strict=True):
        raise ValueError("Design builder path differs from this executable")
    builder_raw, _ = read_plain_file(builder_path, "builder")
    if bytes_sha256(builder_raw) != builder.get("sha256"):
        raise ValueError("Builder SHA-256 mismatch")
    if builder.get("flags") != ["-I", "-B"]:
        raise ValueError("Design builder flags must be exactly -I -B")
    expected_python = Path(builder.get("python", ""))
    if not expected_python.is_absolute() or expected_python.resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ValueError("Builder must run under the frozen Python interpreter")
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
        raise ValueError("Builder requires exact -I -B Python flags")
    expected = [
        {"name": name, "beta": beta, "filename": filename}
        for name, beta, filename in EXPECTED_ENDPOINTS
    ]
    if design.get("endpoints") != expected:
        raise ValueError("Design endpoint grid differs from the frozen beta grid")
    if design.get("formula") != (
        "W_U468 + (1-beta)*(W_exactP12_U468-W_U468) + "
        "beta*(W_histP12_U464-W_U464)"
    ):
        raise ValueError("Design transport formula mismatch")
    if design.get("mutable_parameter_names") != list(MUTABLE_NAMES):
        raise ValueError("Design mutable-parameter list mismatch")
    if design.get("arithmetic") != "float64_accumulate_then_cast_to_parent_dtype":
        raise ValueError("Design arithmetic rule mismatch")
    if design.get("resume_forbidden") is not True:
        raise ValueError("Design must forbid resume from evaluation-only endpoints")
    if design.get("l2_closed_interval") != [0.00177912, 0.00266868]:
        raise ValueError("Design L2 interval mismatch")
    execution_rule = design.get("execution_rule")
    if not isinstance(execution_rule, dict) or any(
        execution_rule.get(key) != expected_value
        for key, expected_value in (
            ("preflight_attempts", 1),
            ("formal_attempts", 1),
            ("retry_authorized", False),
            ("formal_only_after_preflight_and_independent_static_audits_pass", True),
            ("no_training_steps", True),
        )
    ):
        raise ValueError("Design execution rule mismatch")
    scope = design.get("scope")
    if not isinstance(scope, dict) or scope != {
        "local_only": True,
        "network": False,
        "training": False,
        "gold": False,
        "package": False,
        "upload": False,
        "submission": False,
    }:
        raise ValueError("Design scope mismatch")
    downstream = design.get("downstream")
    if not isinstance(downstream, dict) or downstream.get(
        "endpoint_materialization_authorized_now"
    ) is not False:
        raise ValueError("Design must defer materialization to formal authorization")
    authorization = design.get("authorization")
    if not isinstance(authorization, dict):
        raise ValueError("Design has no authorization binding")
    if authorization.get("required_status") != (
        "failed_specialist_behavior_gate_broad_and_gold_forbidden"
    ) or authorization.get("scope") != (
        "one_local_direction_correction_sweep_without_gold"
    ):
        raise ValueError("Design authorization status/scope declaration mismatch")
    auth_path = relative_path(authorization.get("path"), "authorization.path")
    auth_raw, auth_evidence = read_plain_file(auth_path, "authorization")
    if auth_evidence["sha256"] != authorization.get("sha256"):
        raise ValueError("Design authorization SHA-256 mismatch")
    auth = strict_json_loads(auth_raw, "authorization")
    if auth.get("status") != authorization.get("required_status"):
        raise ValueError("Design authorization status mismatch")
    auth_decision = auth.get("decision")
    if not isinstance(auth_decision, dict):
        raise ValueError("Design authorization has no decision object")
    if any(
        auth_decision.get(key) is not False
        for key in (
            "candidate_promoted",
            "broad_behavior_authorized",
            "gold_authorized",
            "retry_same_candidate_authorized",
            "package_upload_or_submission_authorized",
        )
    ):
        raise ValueError("Design authorization does not close the failed candidate")
    return design, digest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    design, design_sha256 = load_and_validate_design(args)
    if args.mode == "preflight":
        if args.formal_authorization is not None or (
            args.expected_formal_authorization_sha256 is not None
        ):
            raise ValueError("Preflight must not receive formal authorization")
        formal_authorization_evidence = None
    else:
        if args.formal_authorization is None or not (
            args.expected_formal_authorization_sha256
        ):
            raise ValueError("Formal mode requires a hash-bound authorization")
        if args.formal_authorization.is_absolute():
            raise ValueError("Formal authorization must be root-relative")
        formal_auth_path = relative_path(
            str(args.formal_authorization),
            "formal authorization",
        )
        formal_auth_raw, formal_auth_file = read_plain_file(
            formal_auth_path,
            "formal authorization",
        )
        if formal_auth_file["sha256"] != (
            args.expected_formal_authorization_sha256
        ):
            raise ValueError("Formal authorization CLI SHA-256 mismatch")
        formal_auth = strict_json_loads(
            formal_auth_raw,
            "formal authorization",
        )
        if formal_auth.get("schema_version") != (
            "ptcg-u468-p12-delta-direction-formal-authorization-v1"
        ):
            raise ValueError("Formal authorization schema mismatch")
        if formal_auth.get("status") != (
            "passed_preflight_and_static_audits_formal_materialization_authorized"
        ):
            raise ValueError("Formal authorization status mismatch")
        if formal_auth.get("design_sha256") != design_sha256:
            raise ValueError("Formal authorization design SHA-256 mismatch")
        if formal_auth.get("builder_sha256") != design["builder"]["sha256"]:
            raise ValueError("Formal authorization builder SHA-256 mismatch")
        if formal_auth.get("formal_attempts_authorized") != 1 or (
            formal_auth.get("retry_authorized") is not False
        ):
            raise ValueError("Formal authorization attempt policy mismatch")
        if formal_auth.get("preflight_status") != "preflight_passed_no_writes":
            raise ValueError("Formal authorization preflight status mismatch")
        if formal_auth.get("preflight_writes") != 0:
            raise ValueError("Formal authorization preflight writes mismatch")
        if formal_auth.get("preflight_targets_remained_absent") is not True:
            raise ValueError("Formal authorization target-absence proof mismatch")
        if formal_auth.get("independent_static_audits_passed") != 3:
            raise ValueError("Formal authorization audit-count mismatch")
        expected_models = [
            design["endpoint_geometry"][name]["model_state_sha256"]
            for name, _, _ in EXPECTED_ENDPOINTS
        ]
        if formal_auth.get("preflight_endpoint_model_state_sha256") != (
            expected_models
        ):
            raise ValueError("Formal authorization preflight model hashes mismatch")
        frozen_evidence: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        expected_evidence_status = {
            "preflight_protocol": "locked_before_single_zero_write_preflight",
            "preflight_result": "preflight_passed_no_writes",
            "static_review_decision": (
                "passed_three_independent_static_audits_formal_authorized"
            ),
        }
        for evidence_name, expected_status in expected_evidence_status.items():
            binding = formal_auth.get(evidence_name)
            if not isinstance(binding, dict):
                raise ValueError(
                    f"Formal authorization lacks {evidence_name} binding"
                )
            evidence_path = relative_path(
                binding.get("path"),
                f"formal_authorization.{evidence_name}.path",
            )
            evidence_raw, evidence_file = read_plain_file(
                evidence_path,
                evidence_name,
            )
            if evidence_file["sha256"] != binding.get("sha256"):
                raise ValueError(f"{evidence_name} SHA-256 mismatch")
            evidence_json = strict_json_loads(evidence_raw, evidence_name)
            if evidence_json.get("status") != expected_status or (
                binding.get("required_status") != expected_status
            ):
                raise ValueError(f"{evidence_name} status mismatch")
            frozen_evidence[evidence_name] = (evidence_json, evidence_file)
        preflight_protocol, preflight_protocol_file = frozen_evidence[
            "preflight_protocol"
        ]
        if preflight_protocol.get("design", {}).get("sha256") != design_sha256:
            raise ValueError("Preflight protocol design binding mismatch")
        if preflight_protocol.get("builder", {}).get("sha256") != (
            design["builder"]["sha256"]
        ):
            raise ValueError("Preflight protocol builder binding mismatch")
        if preflight_protocol.get("attempts_authorized") != 1 or (
            preflight_protocol.get("retry_authorized") is not False
        ):
            raise ValueError("Preflight protocol attempt policy mismatch")
        if preflight_protocol.get("expected", {}).get(
            "model_state_sha256"
        ) != expected_models:
            raise ValueError("Preflight protocol endpoint hashes mismatch")
        preflight_result, preflight_result_file = frozen_evidence[
            "preflight_result"
        ]
        if preflight_result.get("design_sha256") != design_sha256 or (
            preflight_result.get("builder_sha256")
            != design["builder"]["sha256"]
        ):
            raise ValueError("Preflight result source binding mismatch")
        if preflight_result.get("preflight_protocol_sha256") != (
            preflight_protocol_file["sha256"]
        ):
            raise ValueError("Preflight result protocol binding mismatch")
        if preflight_result.get("writes") != 0 or (
            preflight_result.get("targets_remained_absent") is not True
        ):
            raise ValueError("Preflight result zero-write/absence mismatch")
        if preflight_result.get("endpoint_model_state_sha256") != expected_models:
            raise ValueError("Preflight result endpoint hashes mismatch")
        static_review, static_review_file = frozen_evidence[
            "static_review_decision"
        ]
        if static_review.get("design_sha256") != design_sha256 or (
            static_review.get("builder_sha256") != design["builder"]["sha256"]
        ):
            raise ValueError("Static review source binding mismatch")
        if static_review.get("preflight_result_sha256") != (
            preflight_result_file["sha256"]
        ):
            raise ValueError("Static review preflight-result binding mismatch")
        audits = static_review.get("independent_audits")
        if not isinstance(audits, list) or len(audits) != 3 or any(
            not isinstance(item, dict) or item.get("result") != "PASS"
            for item in audits
        ):
            raise ValueError("Static review independent-audit evidence mismatch")
        formal_authorization_evidence = {
            "path": str(formal_auth_path.relative_to(ROOT)),
            "sha256": formal_auth_file["sha256"],
            "status": formal_auth["status"],
            "preflight_protocol_sha256": preflight_protocol_file["sha256"],
            "preflight_result_sha256": preflight_result_file["sha256"],
            "static_review_decision_sha256": static_review_file["sha256"],
        }
    bindings = design.get("input_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "old_parent", "old_p12", "new_parent", "new_exact_p12"
    }:
        raise ValueError("Design input bindings are incomplete")

    checkpoints: dict[str, dict[str, Any]] = {}
    input_evidence: dict[str, Any] = {}
    for label, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"input_bindings.{label} must be an object")
        path = relative_path(binding.get("path"), f"input_bindings.{label}.path")
        raw_checkpoint, file_evidence = read_plain_file(path, label)
        digest = file_evidence["sha256"]
        if digest != binding.get("sha256"):
            raise ValueError(f"Input SHA-256 mismatch: {label}")
        checkpoint = load_checkpoint_bytes(raw_checkpoint, label)
        checkpoints[label] = checkpoint
        model_hash = model_state_sha256(checkpoint["model_state_dict"])
        if model_hash != binding.get("model_state_sha256"):
            raise ValueError(f"Input model-state SHA-256 mismatch: {label}")
        input_evidence[label] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "update": checkpoint.get("update"),
            "model_state_sha256": model_hash,
            "device": file_evidence["device"],
            "inode": file_evidence["inode"],
            "mode": file_evidence["mode"],
            "nlink": file_evidence["nlink"],
            "descriptor_held_read": True,
        }

    old_parent = checkpoints["old_parent"]["model_state_dict"]
    old_p12 = checkpoints["old_p12"]["model_state_dict"]
    new_parent = checkpoints["new_parent"]["model_state_dict"]
    new_exact = checkpoints["new_exact_p12"]["model_state_dict"]
    for label, state in (
        ("old_parent", old_parent),
        ("old_p12", old_p12),
        ("new_parent", new_parent),
        ("new_exact_p12", new_exact),
    ):
        assert_state_compatible(old_parent, state, label)
        for name, tensor in state.items():
            if not bool(torch.isfinite(tensor.detach().cpu()).all()):
                raise ValueError(f"{label} has non-finite model tensor {name!r}")
    expected_changed = sorted(MUTABLE_NAMES)
    if changed_tensor_names(old_parent, old_p12) != expected_changed:
        raise ValueError("Historical P12 did not change exactly mutable10")
    if changed_tensor_names(new_parent, new_exact) != expected_changed:
        raise ValueError("Current exact P12 did not change exactly mutable10")
    if checkpoints["new_parent"].get("update") != 468:
        raise ValueError("New parent must be update 468")
    for key in REQUIRED_SLIM_KEYS:
        if key not in checkpoints["new_parent"]:
            raise KeyError(f"New parent lacks required slim key {key!r}")

    historical_delta = flat_delta(old_parent, old_p12)
    current_delta = flat_delta(new_parent, new_exact)
    historical_l2 = float(torch.linalg.vector_norm(historical_delta).item())
    current_l2 = float(torch.linalg.vector_norm(current_delta).item())
    source_cosine = float(
        torch.dot(historical_delta, current_delta).item()
        / (historical_l2 * current_l2)
    )
    source_difference_l2 = float(
        torch.linalg.vector_norm(historical_delta - current_delta).item()
    )
    source_expected = design.get("source_geometry")
    if not isinstance(source_expected, dict):
        raise ValueError("Design has no source_geometry")
    for key, actual in (
        ("historical_l2", historical_l2),
        ("current_l2", current_l2),
        ("cosine", source_cosine),
        ("difference_l2", source_difference_l2),
    ):
        if not math.isclose(actual, float(source_expected.get(key)), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Source geometry mismatch for {key}: {actual}")

    endpoint_states: dict[str, dict[str, torch.Tensor]] = {}
    endpoint_evidence: list[dict[str, Any]] = []
    for name, beta, filename in EXPECTED_ENDPOINTS:
        state = build_endpoint_state(
            beta=beta,
            new_parent=new_parent,
            new_exact=new_exact,
            old_parent=old_parent,
            old_p12=old_p12,
        )
        if changed_tensor_names(new_parent, state) != expected_changed:
            raise ValueError(f"Endpoint {name} did not change exactly mutable10")
        for tensor_name in new_parent:
            if tensor_name not in MUTABLE_NAMES and not torch.equal(
                new_parent[tensor_name], state[tensor_name]
            ):
                raise ValueError(f"Endpoint {name} changed frozen tensor {tensor_name!r}")
        if not all(
            bool(torch.isfinite(tensor.detach().cpu()).all())
            for tensor in state.values()
        ):
            raise ValueError(f"Endpoint {name} has a non-finite model tensor")
        delta = flat_delta(new_parent, state)
        l2 = float(torch.linalg.vector_norm(delta).item())
        cosine_historical = float(
            torch.dot(delta, historical_delta).item()
            / (l2 * historical_l2)
        )
        expected_geometry = design["endpoint_geometry"][name]
        if not math.isclose(l2, float(expected_geometry["l2"]), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Endpoint {name} L2 mismatch: {l2}")
        if not math.isclose(
            cosine_historical,
            float(expected_geometry["cosine_historical"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Endpoint {name} cosine mismatch: {cosine_historical}")
        endpoint_model_hash = model_state_sha256(state)
        if endpoint_model_hash != expected_geometry.get("model_state_sha256"):
            raise ValueError(f"Endpoint {name} model-state SHA-256 mismatch")
        low, high = design["l2_closed_interval"]
        if not float(low) <= l2 <= float(high):
            raise ValueError(f"Endpoint {name} lies outside the frozen L2 interval")
        endpoint_states[name] = state
        endpoint_evidence.append(
            {
                "name": name,
                "beta": beta,
                "filename": filename,
                "l2_vs_u468": l2,
                "cosine_historical": cosine_historical,
                "model_state_sha256": endpoint_model_hash,
                "changed_tensor_names": expected_changed,
                "frozen_tensor_count": len(state) - len(MUTABLE_NAMES),
            }
        )
    if len({item["model_state_sha256"] for item in endpoint_evidence}) != len(EXPECTED_ENDPOINTS):
        raise ValueError("Endpoint model-state hashes must be unique")

    output_root = relative_path(design.get("output_root"), "output_root")
    marker = relative_path(design.get("attempt_marker"), "attempt_marker")
    manifest_path = output_root / "delta_transport_manifest.json"
    if output_root.exists() or marker.exists():
        raise FileExistsError("Formal/preflight targets have already been consumed")

    preview = {
        "schema_version": "ptcg-u468-p12-delta-direction-sweep-preflight-v1",
        "status": "preflight_passed_no_writes" if args.mode == "preflight" else "formal_ready",
        "design_sha256": design_sha256,
        "input_evidence": input_evidence,
        "source_geometry": {
            "historical_l2": historical_l2,
            "current_l2": current_l2,
            "cosine": source_cosine,
            "difference_l2": source_difference_l2,
        },
        "endpoints": endpoint_evidence,
        "writes": 0,
        "formal_authorization": formal_authorization_evidence,
    }
    if args.mode == "preflight":
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True, indent=2))
        return

    created_at = datetime.now(timezone.utc).isoformat()
    marker_payload = {
        "schema_version": "ptcg-u468-p12-delta-direction-attempt-v1",
        "status": "formal_attempt_claimed",
        "created_at_utc": created_at,
        "attempt": 1,
        "attempts_authorized": 1,
        "retry_authorized": False,
        "design_sha256": design_sha256,
        "formal_authorization": formal_authorization_evidence,
        "planned_endpoints": [name for name, _, _ in EXPECTED_ENDPOINTS],
    }
    if marker.parent != ROOT or output_root.parent != ROOT / "artifacts":
        raise ValueError("Formal marker/output root must use frozen direct parents")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    root_parent_fd = os.open(ROOT.parent, directory_flags)
    root_fd = os.open(ROOT.name, directory_flags, dir_fd=root_parent_fd)
    cwd_fd = os.open(".", directory_flags)
    if (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino) != (
        os.fstat(cwd_fd).st_dev,
        os.fstat(cwd_fd).st_ino,
    ):
        raise RuntimeError("Frozen root differs from the held current directory")
    os.close(cwd_fd)
    root_binding = revalidate_directory_at(
        parent_fd=root_parent_fd,
        name=ROOT.name,
        directory_fd=root_fd,
        expected_mode=None,
        label="frozen root",
    )
    artifacts_fd = os.open("artifacts", directory_flags, dir_fd=root_fd)
    artifacts_binding = revalidate_directory_at(
        parent_fd=root_fd,
        name="artifacts",
        directory_fd=artifacts_fd,
        expected_mode=None,
        label="artifacts directory",
    )
    for parent_fd, name, label in (
        (root_fd, marker.name, "attempt marker"),
        (artifacts_fd, output_root.name, "output root"),
    ):
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"{label} appeared before descriptor claim")
    marker_publish, marker_fd = publish_o_excl_at(
        directory_fd=root_fd,
        name=marker.name,
        payload=canonical_json_bytes(marker_payload),
        label="attempt marker",
    )
    os.fsync(root_fd)
    os.mkdir(output_root.name, mode=0o700, dir_fd=artifacts_fd)
    os.fsync(artifacts_fd)
    output_fd = os.open(output_root.name, directory_flags, dir_fd=artifacts_fd)
    output_binding = revalidate_directory_at(
        parent_fd=artifacts_fd,
        name=output_root.name,
        directory_fd=output_fd,
        expected_mode=0o700,
        label="output directory",
    )
    if os.listdir(output_fd):
        raise RuntimeError("Fresh output directory is not empty")

    published: list[dict[str, Any]] = []
    endpoint_fds: dict[str, int] = {}
    for endpoint, state in zip(endpoint_evidence, endpoint_states.values(), strict=True):
        slim = {
            key: copy.deepcopy(checkpoints["new_parent"][key])
            for key in REQUIRED_SLIM_KEYS
        }
        slim.update(
            {
                "model_state_dict": state,
                "update": 468,
                "evaluation_only": True,
                "resume_forbidden": True,
                "optimizer_states_omitted": [
                    "optimizer_state_dict",
                    "bc_replay_optimizer_state_dict",
                    "opponent_quota_state",
                ],
                "actor_delta_transport": {
                    "schema_version": "ptcg-u468-p12-delta-direction-endpoint-v1",
                    "created_at_utc": created_at,
                    "design_sha256": design_sha256,
                    "formula": "W_U468 + (1-beta)*(W_exactP12_U468-W_U468) + beta*(W_histP12_U464-W_U464)",
                    "arithmetic": design["arithmetic"],
                    "beta": endpoint["beta"],
                    "endpoint": endpoint["name"],
                    "input_evidence": input_evidence,
                    "model_state_sha256": endpoint["model_state_sha256"],
                    "mutable_parameter_names": list(MUTABLE_NAMES),
                    "resume_forbidden": True,
                },
            }
        )
        buffer = io.BytesIO()
        torch.save(slim, buffer)
        raw = buffer.getvalue()
        reloaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if model_state_sha256(reloaded["model_state_dict"]) != endpoint["model_state_sha256"]:
            raise RuntimeError(f"Endpoint {endpoint['name']} changed during serialization")
        if reloaded.get("resume_forbidden") is not True:
            raise RuntimeError("Serialized endpoint lost resume_forbidden")
        output_path = output_root / endpoint["filename"]
        publish_evidence, endpoint_fd = publish_o_excl_at(
            directory_fd=output_fd,
            name=endpoint["filename"],
            payload=raw,
            label=f"endpoint {endpoint['name']}",
            expected_model_state_sha256=endpoint["model_state_sha256"],
        )
        endpoint_fds[endpoint["filename"]] = endpoint_fd
        os.fsync(output_fd)
        published.append(
            {
                **endpoint,
                "path": str(output_path.relative_to(ROOT)),
                **publish_evidence,
            }
        )

    expected_endpoint_names = sorted(
        filename for _, _, filename in EXPECTED_ENDPOINTS
    )
    observed_endpoint_names = sorted(os.listdir(output_fd))
    if observed_endpoint_names != expected_endpoint_names:
        raise RuntimeError(
            "Endpoint directory contents mismatch before terminal manifest"
        )

    revalidate_directory_at(
        parent_fd=root_parent_fd,
        name=ROOT.name,
        directory_fd=root_fd,
        expected_mode=None,
        label="frozen root before manifest",
    )
    revalidate_directory_at(
        parent_fd=root_fd,
        name="artifacts",
        directory_fd=artifacts_fd,
        expected_mode=None,
        label="artifacts directory before manifest",
    )
    output_binding_before_manifest = revalidate_directory_at(
        parent_fd=artifacts_fd,
        name=output_root.name,
        directory_fd=output_fd,
        expected_mode=0o700,
        label="output directory before manifest",
    )
    marker_final = revalidate_published_at(
        directory_fd=root_fd,
        name=marker.name,
        file_fd=marker_fd,
        expected_sha256=marker_publish["sha256"],
        label="attempt marker before manifest",
    )
    endpoint_final: dict[str, dict[str, Any]] = {}
    for item in published:
        endpoint_final[item["filename"]] = revalidate_published_at(
            directory_fd=output_fd,
            name=item["filename"],
            file_fd=endpoint_fds[item["filename"]],
            expected_sha256=item["sha256"],
            label=f"endpoint {item['name']} before manifest",
        )

    manifest = {
        "schema_version": "ptcg-u468-p12-delta-direction-sweep-manifest-v1",
        "status": "materialized_pending_completion_seal",
        "created_at_utc": created_at,
        "design_sha256": design_sha256,
        "attempt_marker": {
            "path": str(marker.relative_to(ROOT)),
            **marker_publish,
        },
        "formal_authorization": formal_authorization_evidence,
        "input_evidence": input_evidence,
        "source_geometry": preview["source_geometry"],
        "endpoints": published,
        "output_directory_binding": {
            "path": str(output_root.relative_to(ROOT)),
            **output_binding_before_manifest,
            "contents_before_manifest": observed_endpoint_names,
            "expected_contents_after_manifest": sorted(
                expected_endpoint_names
                + [manifest_path.name, "COMPLETED.json"]
            ),
        },
        "integrity": {
            "endpoint_count_exact": len(published) == 3,
            "changed_tensors_exactly_mutable10_all": True,
            "all_other_model_tensors_byte_exact_u468": True,
            "all_model_states_finite": True,
            "all_l2_inside_closed_interval": True,
            "optimizer_and_quota_states_omitted": True,
            "resume_forbidden": True,
            "checkpoint_writes_exact": 3,
            "manifest_written_after_all_checkpoints": True,
            "completion_requires_separate_seal": True,
            "endpoint_directory_gate_before_manifest": True,
            "all_published_paths_match_held_descriptors_before_manifest": True,
            "descriptor_held_input_reads": True,
            "descriptor_held_output_reloads": True,
            "directory_fsync_after_every_publication": True,
        },
        "scope": {
            "training": False,
            "evaluation": False,
            "gold": False,
            "network": False,
            "package_upload_or_submission": False,
        },
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_publish, manifest_fd = publish_o_excl_at(
        directory_fd=output_fd,
        name=manifest_path.name,
        payload=manifest_payload,
        label="pending materialization manifest",
    )
    os.fsync(output_fd)
    completion_name = "COMPLETED.json"
    completion_claim, completion_fd = claim_empty_o_excl_at(
        directory_fd=output_fd,
        name=completion_name,
        label="completion seal",
    )
    os.fsync(output_fd)
    expected_names = sorted(
        [filename for _, _, filename in EXPECTED_ENDPOINTS]
        + [manifest_path.name, completion_name]
    )
    observed_names = sorted(os.listdir(output_fd))
    if observed_names != expected_names:
        raise RuntimeError("Pre-seal output directory contents mismatch")
    root_final = revalidate_directory_at(
        parent_fd=root_parent_fd,
        name=ROOT.name,
        directory_fd=root_fd,
        expected_mode=None,
        label="frozen root before completion seal",
    )
    artifacts_final = revalidate_directory_at(
        parent_fd=root_fd,
        name="artifacts",
        directory_fd=artifacts_fd,
        expected_mode=None,
        label="artifacts directory before completion seal",
    )
    revalidate_directory_at(
        parent_fd=artifacts_fd,
        name=output_root.name,
        directory_fd=output_fd,
        expected_mode=0o700,
        label="output directory before completion seal",
    )
    marker_final = revalidate_published_at(
        directory_fd=root_fd,
        name=marker.name,
        file_fd=marker_fd,
        expected_sha256=marker_final["sha256"],
        label="attempt marker before completion seal",
    )
    for item in published:
        endpoint_final[item["filename"]] = revalidate_published_at(
            directory_fd=output_fd,
            name=item["filename"],
            file_fd=endpoint_fds[item["filename"]],
            expected_sha256=endpoint_final[item["filename"]]["sha256"],
            label=f"endpoint {item['name']} before completion seal",
        )
    manifest_final = revalidate_published_at(
        directory_fd=output_fd,
        name=manifest_path.name,
        file_fd=manifest_fd,
        expected_sha256=manifest_publish["sha256"],
        label="pending manifest before completion seal",
    )
    os.fchmod(output_fd, 0o500)
    os.fsync(output_fd)
    output_final = revalidate_directory_at(
        parent_fd=artifacts_fd,
        name=output_root.name,
        directory_fd=output_fd,
        expected_mode=0o500,
        label="sealed output directory",
    )
    if sorted(os.listdir(output_fd)) != expected_names:
        raise RuntimeError("Sealed output directory contents mismatch")
    completion = {
        "schema_version": "ptcg-u468-p12-delta-direction-completion-seal-v1",
        "status": "completed",
        "created_at_utc": created_at,
        "design_sha256": design_sha256,
        "formal_authorization": formal_authorization_evidence,
        "attempt_marker": marker_final,
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            **manifest_final,
        },
        "endpoints": [
            {
                "name": item["name"],
                "path": item["path"],
                **endpoint_final[item["filename"]],
            }
            for item in published
        ],
        "output_directory": {
            "path": str(output_root.relative_to(ROOT)),
            **output_final,
            "contents": expected_names,
        },
        "held_parent_directories": {
            "root": root_final,
            "artifacts": artifacts_final,
        },
        "integrity": {
            "all_preterminal_gates_passed_before_seal_content": True,
            "completion_entry_claimed_and_parent_fsynced_while_empty": True,
            "output_directory_write_locked_before_seal_content": True,
            "completion_content_written_and_fsynced_through_held_descriptor": True,
            "retry_authorized": False,
            "resume_forbidden": True,
        },
    }
    completion_payload = canonical_json_bytes(completion)
    completion_publish = complete_claimed_json(
        directory_fd=output_fd,
        name=completion_name,
        file_fd=completion_fd,
        payload=completion_payload,
        label="completion seal",
    )
    terminal = {
        **completion,
        "completion_publication": completion_publish,
        "empty_completion_claim": completion_claim,
    }
    for fd in endpoint_fds.values():
        os.close(fd)
    os.close(manifest_fd)
    os.close(completion_fd)
    os.close(marker_fd)
    os.close(output_fd)
    os.close(artifacts_fd)
    os.close(root_fd)
    os.close(root_parent_fd)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
