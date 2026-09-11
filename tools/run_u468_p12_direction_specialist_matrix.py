#!/usr/bin/env python3
"""Run the frozen U468 P12-direction 3x6 specialist matrix exactly once."""

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
SCHEMA = "ptcg-u468-p12-direction-specialist-matrix-preregistration-v1"
AUTH_SCHEMA = "ptcg-u468-p12-direction-specialist-formal-authorization-v1"
EXPECTED_ENDPOINTS = ("beta050", "beta075", "beta100")
EXPECTED_PANELS = (
    ("pokemonfan", "pokemonfan", None),
    ("flg", "flg", None),
    ("core5", "core5", None),
    ("dominic", "core5", "Dominic Peel"),
    ("luca", "core5", "Luca"),
    ("szlach", "core5", "szlachetny snieg"),
)
EXPECTED_OUTPUT_ROOT = (
    "artifacts/ppo_u468_p12delta_direction_beta050_075_100_"
    "design202608092.specialist_behavior"
)
EXPECTED_MARKER = (
    ".ptcg-u468-p12delta-direction-specialist-behavior-attempt-202608092.json"
)


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


def relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"{label} must be a safe root-relative path")
    path = ROOT / rel
    if path.resolve(strict=False) != path:
        raise ValueError(f"{label} is not normalized or resolves through a symlink")
    return path


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
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError(f"{label} changed while held")
        if len(payload) != after.st_size:
            raise RuntimeError(f"{label} size changed while held")
        return payload, {
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "device": after.st_dev,
            "inode": after.st_ino,
            "mode": oct(after.st_mode & 0o777),
            "nlink": after.st_nlink,
        }
    finally:
        os.close(fd)


def read_json_binding(
    binding: Any,
    label: str,
    *,
    expected_status: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not isinstance(binding, dict):
        raise TypeError(f"{label} binding must be an object")
    path = relative_path(binding.get("path"), f"{label}.path")
    raw, evidence = read_plain_file(path, label)
    if evidence["sha256"] != binding.get("sha256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    value = strict_json_loads(raw, label)
    if expected_status is not None:
        if binding.get("required_status") != expected_status:
            raise ValueError(f"{label} binding status mismatch")
        if value.get("status") != expected_status:
            raise ValueError(f"{label} actual status mismatch")
    return value, evidence, path


def load_checkpoint(payload: bytes, label: str) -> dict[str, Any]:
    checkpoint = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label} must contain a checkpoint dict")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError(f"{label} has no model_state_dict mapping")
    if not all(
        isinstance(tensor, torch.Tensor)
        and bool(torch.isfinite(tensor.detach().cpu()).all())
        for tensor in state.values()
    ):
        raise ValueError(f"{label} contains a non-finite/non-tensor model entry")
    return checkpoint


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


def file_binding(binding: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(binding, dict):
        raise TypeError(f"{label} binding must be an object")
    path = relative_path(binding.get("path"), f"{label}.path")
    raw, evidence = read_plain_file(path, label)
    if evidence["sha256"] != binding.get("sha256"):
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise ValueError("Current working directory differs from frozen root")
    expected_python = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
    if Path(sys.executable).resolve(strict=True) != expected_python.resolve(strict=True):
        raise ValueError("Launcher must run under my_project_env Python")
    if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
        raise ValueError("Launcher requires exact -I -B flags")


def validate_protocol(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, Path]]:
    validate_runtime()
    if args.preregistration.is_absolute():
        raise ValueError("--preregistration must be root-relative")
    protocol_path = relative_path(str(args.preregistration), "preregistration")
    protocol_raw, protocol_file = read_plain_file(protocol_path, "preregistration")
    protocol_sha256 = protocol_file["sha256"]
    if protocol_sha256 != args.expected_preregistration_sha256:
        raise ValueError("Preregistration SHA-256 differs from CLI binding")
    protocol = strict_json_loads(protocol_raw, "preregistration")
    if protocol.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported specialist preregistration schema")
    if protocol.get("status") != "locked_before_single_zero_write_preflight":
        raise ValueError("Specialist preregistration status mismatch")
    if protocol.get("required_cwd") != str(ROOT):
        raise ValueError("Specialist preregistration cwd mismatch")

    launcher = protocol.get("launcher")
    if not isinstance(launcher, dict):
        raise TypeError("Preregistration lacks launcher binding")
    launcher_path = relative_path(launcher.get("path"), "launcher.path")
    if launcher_path != Path(__file__).resolve(strict=True):
        raise ValueError("Launcher path differs from this executable")
    launcher_raw, launcher_file = read_plain_file(launcher_path, "launcher")
    if launcher_file["sha256"] != launcher.get("sha256"):
        raise ValueError("Launcher SHA-256 mismatch")
    if launcher.get("python") != "/home/xxc/miniconda3/envs/my_project_env/bin/python":
        raise ValueError("Launcher Python binding mismatch")
    if launcher.get("flags") != ["-I", "-B"]:
        raise ValueError("Launcher flags binding mismatch")

    decision, _, _ = read_json_binding(
        protocol.get("materialization_integrity_decision"),
        "materialization integrity decision",
        expected_status="passed_specialist_behavior_preregistration_authorized",
    )
    if decision.get("decision", {}).get(
        "specialist_behavior_preregistration_authorized"
    ) is not True:
        raise ValueError("Materialization decision does not authorize preregistration")
    if decision.get("decision", {}).get(
        "specialist_behavior_execution_authorized_by_this_file"
    ) is not False:
        raise ValueError("Materialization decision scope is unexpectedly broad")
    seal, _, _ = read_json_binding(
        protocol.get("materialization_completion_seal"),
        "materialization completion seal",
        expected_status="completed",
    )
    if seal.get("schema_version") != (
        "ptcg-u468-p12-delta-direction-completion-seal-v1"
    ):
        raise ValueError("Materialization completion seal schema mismatch")

    evaluator = protocol.get("evaluator")
    if not isinstance(evaluator, dict):
        raise TypeError("Preregistration lacks evaluator binding")
    evaluator_path, _ = file_binding(evaluator, "evaluator")
    expected_evaluator = ROOT / "tools" / "evaluate_policy_bc.py"
    if evaluator_path != expected_evaluator:
        raise ValueError("Evaluator path mismatch")
    if evaluator.get("python") != launcher.get("python") or (
        evaluator.get("flags") != ["-I", "-B"]
    ):
        raise ValueError("Evaluator runtime binding mismatch")
    dependencies = protocol.get("evaluator_dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("Evaluator dependencies must be a non-empty list")
    for index, binding in enumerate(dependencies):
        file_binding(binding, f"evaluator dependency {index}")

    endpoint_bindings = protocol.get("endpoints")
    if not isinstance(endpoint_bindings, list) or [
        item.get("name") if isinstance(item, dict) else None
        for item in endpoint_bindings
    ] != list(EXPECTED_ENDPOINTS):
        raise ValueError("Endpoint list/order mismatch")
    endpoint_paths: dict[str, Path] = {}
    for item in endpoint_bindings:
        assert isinstance(item, dict)
        name = item["name"]
        path = relative_path(item.get("path"), f"endpoint {name}.path")
        raw, endpoint_file = read_plain_file(path, f"endpoint {name} checkpoint")
        if endpoint_file["sha256"] != item.get("sha256"):
            raise ValueError(f"Endpoint {name} SHA-256 mismatch")
        checkpoint = load_checkpoint(raw, f"endpoint {name}")
        model_hash = model_state_sha256(checkpoint["model_state_dict"])
        if model_hash != item.get("model_state_sha256"):
            raise ValueError(f"Endpoint {name} model-state SHA-256 mismatch")
        if checkpoint.get("evaluation_only") is not True or (
            checkpoint.get("resume_forbidden") is not True
        ):
            raise ValueError(f"Endpoint {name} is not frozen evaluation-only")
        endpoint_paths[name] = path
    decision_endpoints = decision.get("endpoints")
    if not isinstance(decision_endpoints, list) or [
        {
            "name": item.get("name"),
            "beta": item.get("beta"),
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "model_state_sha256": item.get("model_state_sha256"),
        }
        for item in decision_endpoints
    ] != endpoint_bindings:
        raise ValueError("Endpoint bindings differ from materialization decision")
    seal_endpoints = seal.get("endpoints")
    if not isinstance(seal_endpoints, list) or [
        {
            "path": item.get("path"),
            "sha256": item.get("sha256"),
        }
        for item in seal_endpoints
    ] != [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in endpoint_bindings
    ]:
        raise ValueError("Endpoint bindings differ from completion seal")

    parent = protocol.get("parent")
    if not isinstance(parent, dict) or parent.get("reuse_frozen_outputs") is not True:
        raise ValueError("Preregistration lacks frozen parent binding")
    parent_path = relative_path(parent.get("checkpoint"), "parent checkpoint")
    parent_raw, parent_file = read_plain_file(parent_path, "parent checkpoint")
    if parent_file["sha256"] != parent.get("sha256"):
        raise ValueError("Parent checkpoint SHA-256 mismatch")
    parent_checkpoint = load_checkpoint(parent_raw, "parent checkpoint")
    if model_state_sha256(parent_checkpoint["model_state_dict"]) != parent.get(
        "model_state_sha256"
    ):
        raise ValueError("Parent checkpoint model-state SHA-256 mismatch")
    source_decision = parent.get("source_decision")
    read_json_binding(source_decision, "parent source decision")

    data_bindings = protocol.get("data_bindings")
    if not isinstance(data_bindings, dict) or set(data_bindings) != {
        "pokemonfan",
        "flg",
        "core5",
    }:
        raise ValueError("Data binding set mismatch")
    data_paths: dict[str, Path] = {}
    for name, binding in data_bindings.items():
        path, _ = file_binding(binding, f"data {name}")
        data_paths[name] = path

    frozen_outputs = protocol.get("frozen_parent_outputs")
    if not isinstance(frozen_outputs, dict) or set(frozen_outputs) != {
        name for name, _, _ in EXPECTED_PANELS
    }:
        raise ValueError("Frozen parent output set mismatch")
    for name, binding in frozen_outputs.items():
        parent_json, _, _ = read_json_binding(binding, f"parent output {name}")
        expected_metrics = binding.get("metrics")
        if not isinstance(expected_metrics, dict):
            raise TypeError(f"Parent output {name} has no metric binding")
        metrics = parent_json.get("metrics")
        context34 = metrics.get("by_context", {}).get("34", {})
        observed = {
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
        if observed != expected_metrics:
            raise ValueError(f"Frozen parent output {name} metrics mismatch")

    output_rule = protocol.get("output_rule")
    if not isinstance(output_rule, dict):
        raise TypeError("Preregistration lacks output rule")
    output_root = relative_path(output_rule.get("root"), "output root")
    marker = relative_path(output_rule.get("attempt_marker"), "attempt marker")
    if str(output_root.relative_to(ROOT)) != EXPECTED_OUTPUT_ROOT or (
        str(marker.relative_to(ROOT)) != EXPECTED_MARKER
    ):
        raise ValueError("Output root/marker differs from the frozen branch")
    if output_root.parent != ROOT / "artifacts" or marker.parent != ROOT:
        raise ValueError("Output root/marker must use frozen direct parents")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("Specialist output root has already been consumed")
    if marker.exists() or marker.is_symlink():
        raise FileExistsError("Specialist attempt marker has already been consumed")
    if output_rule.get("evaluation_count_exact") != 18 or (
        output_rule.get("attempts_per_evaluation") != 1
    ) or output_rule.get("retry_authorized") is not False or (
        output_rule.get("run_all_before_metric_decision") is not True
    ) or output_rule.get("root_absent_at_lock") is not True or (
        output_rule.get("marker_absent_at_lock") is not True
    ) or output_rule.get("pending_manifest") != (
        "specialist_execution_manifest.json"
    ) or output_rule.get("authoritative_completion_seal") != (
        "COMPLETED.json"
    ) or output_rule.get("final_directory_mode") != "0o500":
        raise ValueError("Output/attempt rule mismatch")
    if protocol.get("downstream") != {
        "separate_metric_decision_required": True,
        "broad_forbidden_until_metric_decision_passes": True,
        "gold_forbidden": True,
        "package_upload_or_submission_forbidden": True,
    }:
        raise ValueError("Downstream authorization rule mismatch")

    expected_gates = {
        "rows_context34_rows_and_value_exact_parent_all_panels": True,
        "pokemonfan_minimum_delta_candidate_minus_parent": {
            "set": 3,
            "hybrid": 3,
            "ordered": 5,
            "count": 0,
            "top1": 3,
            "context34_hybrid": 0,
            "context34_ordered": 0,
        },
        "other_panels_minimum_delta_candidate_minus_parent": {
            "set": 0,
            "hybrid": 0,
            "ordered": 0,
            "count": 0,
            "top1": 0,
            "context34_hybrid": 0,
            "context34_ordered": 0,
        },
        "all_eighteen_outputs_required_before_decision": True,
        "eligible_endpoint_requires_all_gates": True,
        "selection_rule": "smallest_beta_among_fully_eligible_endpoints",
        "none_eligible_rule": "close_direction_sweep_without_broad_or_gold",
    }
    if protocol.get("authoritative_gates") != expected_gates:
        raise ValueError("Authoritative specialist gates mismatch")

    shared = protocol.get("shared_protocol")
    expected_shared = {
        "split": "valid",
        "split_mode": "archive",
        "split_seed": 20260723,
        "batch_size": 256,
        "workers": 8,
        "prediction_order_argument": "policy",
        "expected_prediction_order": "policy_greedy",
        "device": "cuda",
        "compact": True,
        "progress_interval": 0,
    }
    if shared != expected_shared:
        raise ValueError("Shared evaluator protocol mismatch")

    evaluations = protocol.get("ordered_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 18:
        raise ValueError("Ordered evaluation count must be exactly 18")
    expected_matrix = [
        (endpoint, panel, data_name, team)
        for endpoint in EXPECTED_ENDPOINTS
        for panel, data_name, team in EXPECTED_PANELS
    ]
    commands: list[list[str]] = []
    output_paths: list[Path] = []
    for index, (item, expected) in enumerate(zip(evaluations, expected_matrix, strict=True)):
        if not isinstance(item, dict):
            raise TypeError(f"Evaluation {index + 1} must be an object")
        endpoint, panel, data_name, team = expected
        if item.get("order") != index + 1 or item.get("endpoint") != endpoint or (
            item.get("panel") != panel
        ) or item.get("team_name") != team:
            raise ValueError(f"Evaluation {index + 1} identity/order mismatch")
        output = output_root / endpoint / f"{panel}.json"
        if relative_path(item.get("output"), f"evaluation {index + 1} output") != output:
            raise ValueError(f"Evaluation {index + 1} output mismatch")
        command = [
            evaluator["python"],
            "-I",
            "-B",
            str(evaluator_path),
            "--checkpoint",
            str(endpoint_paths[endpoint].relative_to(ROOT)),
            "--data",
            str(data_paths[data_name].relative_to(ROOT)),
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
        if team is not None:
            command.extend(["--team-name", team])
        if item.get("command") != command:
            raise ValueError(f"Evaluation {index + 1} command mismatch")
        if item.get("command_tokens") != len(command):
            raise ValueError(f"Evaluation {index + 1} command token mismatch")
        if item.get("command_sha256") != sha256_bytes(canonical_json_bytes(command)):
            raise ValueError(f"Evaluation {index + 1} command SHA-256 mismatch")
        if item.get("attempts_authorized") != 1 or (
            item.get("expected_success_exit_code") != 0
        ):
            raise ValueError(f"Evaluation {index + 1} attempt/exit rule mismatch")
        commands.append(command)
        output_paths.append(output)
    if len(set(output_paths)) != 18:
        raise ValueError("Evaluation output paths are not unique")
    if protocol.get("ordered_command_matrix_sha256") != sha256_bytes(
        canonical_json_bytes(commands)
    ):
        raise ValueError("Ordered command matrix SHA-256 mismatch")

    if protocol.get("threat_model") != {
        "concurrent_same_uid_malicious_aba_path_replacement": "excluded",
        "accidental_or_persistent_input_drift": (
            "detected_by_held_fd_hash_and_path_identity_before_and_after_each_child"
        ),
        "output_publication": (
            "o_excl_preclaim_held_fd_fsync_reload_exact_tree_terminal_seal"
        ),
    }:
        raise ValueError("Specialist threat model declaration mismatch")

    scope = protocol.get("scope")
    if scope != {
        "local_only": True,
        "network": False,
        "training": False,
        "behavior": True,
        "broad": False,
        "gold": False,
        "package": False,
        "upload": False,
        "submission": False,
    }:
        raise ValueError("Specialist scope mismatch")
    return protocol, protocol_sha256, {
        "protocol": protocol_path,
        "launcher": launcher_path,
        "evaluator": evaluator_path,
        "output_root": output_root,
        "marker": marker,
    }


def validate_formal_authorization(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    if args.formal_authorization is None or not args.expected_formal_authorization_sha256:
        raise ValueError("Formal mode requires a hash-bound authorization")
    if args.formal_authorization.is_absolute():
        raise ValueError("Formal authorization must be root-relative")
    path = relative_path(str(args.formal_authorization), "formal authorization")
    raw, evidence = read_plain_file(path, "formal authorization")
    if evidence["sha256"] != args.expected_formal_authorization_sha256:
        raise ValueError("Formal authorization SHA-256 differs from CLI binding")
    auth = strict_json_loads(raw, "formal authorization")
    if auth.get("schema_version") != AUTH_SCHEMA or auth.get("status") != (
        "passed_preflight_and_three_static_audits_formal_execution_authorized"
    ):
        raise ValueError("Formal authorization schema/status mismatch")
    if auth.get("preregistration_sha256") != protocol_sha256 or (
        auth.get("launcher_sha256") != protocol["launcher"]["sha256"]
    ):
        raise ValueError("Formal authorization source binding mismatch")
    if auth.get("formal_attempts_authorized") != 1 or (
        auth.get("retry_authorized") is not False
    ):
        raise ValueError("Formal authorization attempt policy mismatch")
    evidence_values: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, status in (
        ("preflight_protocol", "locked_before_single_zero_write_preflight"),
        ("preflight_result", "preflight_passed_no_writes"),
        ("static_review_decision", "passed_three_independent_static_audits_formal_authorized"),
    ):
        value, file_evidence, _ = read_json_binding(
            auth.get(name),
            f"formal authorization {name}",
            expected_status=status,
        )
        evidence_values[name] = (value, file_evidence)
    preflight_protocol, preflight_protocol_file = evidence_values["preflight_protocol"]
    preflight_result, preflight_result_file = evidence_values["preflight_result"]
    static_review, _ = evidence_values["static_review_decision"]
    if preflight_protocol.get("preregistration_sha256") != protocol_sha256 or (
        preflight_protocol.get("launcher_sha256") != protocol["launcher"]["sha256"]
    ):
        raise ValueError("Preflight protocol source binding mismatch")
    if preflight_result.get("preregistration_sha256") != protocol_sha256 or (
        preflight_result.get("launcher_sha256") != protocol["launcher"]["sha256"]
    ) or preflight_result.get("preflight_protocol_sha256") != (
        preflight_protocol_file["sha256"]
    ) or preflight_result.get("writes") != 0 or (
        preflight_result.get("targets_remained_absent") is not True
    ):
        raise ValueError("Preflight result binding/zero-write proof mismatch")
    audits = static_review.get("independent_audits")
    if static_review.get("preregistration_sha256") != protocol_sha256 or (
        static_review.get("launcher_sha256") != protocol["launcher"]["sha256"]
    ) or static_review.get("preflight_result_sha256") != (
        preflight_result_file["sha256"]
    ) or not isinstance(audits, list) or len(audits) != 3 or any(
        not isinstance(item, dict) or item.get("result") != "PASS"
        for item in audits
    ):
        raise ValueError("Static review evidence mismatch")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": evidence["sha256"],
        "status": auth["status"],
        "preflight_protocol_sha256": preflight_protocol_file["sha256"],
        "preflight_result_sha256": preflight_result_file["sha256"],
        "static_review_decision_sha256": auth["static_review_decision"]["sha256"],
    }


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


def directory_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def read_held_bytes(fd: int, label: str) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    payload = b"".join(chunks)
    info = os.fstat(fd)
    if len(payload) != info.st_size:
        raise RuntimeError(f"{label} changed size while held")
    return payload


def revalidate_directory_at(
    *,
    parent_fd: int,
    name: str,
    directory_fd: int,
    expected_mode: int,
    expected_contents: list[str],
    label: str,
) -> dict[str, Any]:
    held = os.fstat(directory_fd)
    visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity mismatch")
    if not stat.S_ISDIR(held.st_mode) or (held.st_mode & 0o777) != expected_mode:
        raise RuntimeError(f"{label} type/mode mismatch")
    contents = sorted(os.listdir(directory_fd))
    if contents != sorted(expected_contents):
        raise RuntimeError(f"{label} contents mismatch: {contents}")
    return {
        "device": held.st_dev,
        "inode": held.st_ino,
        "mode": oct(held.st_mode & 0o777),
        "nlink": held.st_nlink,
        "contents": contents,
        "path_descriptor_identity": True,
    }


def revalidate_directory_identity_at(
    *,
    parent_fd: int,
    name: str,
    directory_fd: int,
    label: str,
) -> dict[str, Any]:
    held = os.fstat(directory_fd)
    visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(held.st_mode) or not stat.S_ISDIR(visible.st_mode):
        raise RuntimeError(f"{label} must remain a directory")
    if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity mismatch")
    return {
        "device": held.st_dev,
        "inode": held.st_ino,
        "mode": oct(held.st_mode & 0o777),
        "nlink": held.st_nlink,
        "path_descriptor_identity": True,
    }


def publish_exclusive_at(
    *,
    directory_fd: int,
    name: str,
    payload: bytes,
    path: Path,
    label: str,
    mode: int = 0o600,
) -> tuple[dict[str, Any], int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        os.fchmod(fd, mode)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        evidence = revalidate_published_at(
            directory_fd=directory_fd,
            name=name,
            file_fd=fd,
            expected_sha256=sha256_bytes(payload),
            expected_mode=mode,
            path=path,
            label=label,
        )
        return evidence, fd
    except BaseException:
        os.close(fd)
        raise


def validate_claimed_output_at(
    *,
    directory_fd: int,
    name: str,
    file_fd: int,
    path: Path,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    info = os.fstat(file_fd)
    visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"{label} must be regular and single-link")
    if (info.st_dev, info.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity mismatch")
    os.fchmod(file_fd, 0o600)
    os.fsync(file_fd)
    payload = read_held_bytes(file_fd, label)
    after = os.fstat(file_fd)
    if (after.st_mode & 0o777) != 0o600 or after.st_nlink != 1:
        raise RuntimeError(f"{label} mode/link mismatch after fchmod")
    evidence = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": "0o600",
        "nlink": after.st_nlink,
        "path_descriptor_identity": True,
        "descriptor_preclaimed_held_fsync_and_reload": True,
    }
    return payload, evidence


def revalidate_published_at(
    *,
    directory_fd: int,
    name: str,
    file_fd: int,
    expected_sha256: str,
    expected_mode: int,
    path: Path,
    label: str,
) -> dict[str, Any]:
    held = os.fstat(file_fd)
    visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(held.st_mode) or held.st_nlink != 1:
        raise RuntimeError(f"{label} must remain regular and single-link")
    if (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity mismatch")
    if (held.st_mode & 0o777) != expected_mode:
        raise RuntimeError(f"{label} mode mismatch")
    payload = read_held_bytes(file_fd, label)
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 changed while held")
    final = os.fstat(file_fd)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "device": final.st_dev,
        "inode": final.st_ino,
        "mode": oct(final.st_mode & 0o777),
        "nlink": final.st_nlink,
        "path_descriptor_identity": True,
        "descriptor_held_final_revalidation": True,
    }


def claim_empty_at(
    *,
    directory_fd: int,
    name: str,
    mode: int = 0o600,
) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, mode, dir_fd=directory_fd)
    os.fchmod(fd, mode)
    os.fsync(fd)
    return fd


def populate_claim_at(
    *,
    directory_fd: int,
    name: str,
    file_fd: int,
    path: Path,
    payload: bytes,
) -> dict[str, Any]:
    before = os.fstat(file_fd)
    visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or (
        before.st_mode & 0o777
    ) != 0o600:
        raise RuntimeError("Completion seal empty claim metadata mismatch")
    if (before.st_dev, before.st_ino) != (visible.st_dev, visible.st_ino) or (
        before.st_size != 0
    ):
        raise RuntimeError("Completion seal no longer names its empty held claim")
    os.lseek(file_fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    written = 0
    while written < len(view):
        written += os.write(file_fd, view[written:])
    os.fsync(file_fd)
    reloaded = read_held_bytes(file_fd, "completion seal")
    if reloaded != payload:
        raise RuntimeError("Completion seal held-FD reload mismatch")
    return revalidate_published_at(
        directory_fd=directory_fd,
        name=name,
        file_fd=file_fd,
        expected_sha256=sha256_bytes(payload),
        expected_mode=0o600,
        path=path,
        label="completion seal",
    )


def metric_shape(result: dict[str, Any]) -> tuple[int, int]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Evaluator output has no metrics object")
    rows = metrics.get("rows")
    context = metrics.get("by_context")
    if type(rows) is not int or rows <= 0 or not isinstance(context, dict):
        raise ValueError("Evaluator output rows/context schema mismatch")
    context34 = context.get("34", {})
    if not isinstance(context34, dict):
        raise ValueError("Evaluator output context34 schema mismatch")
    context_rows = context34.get("rows", 0)
    if type(context_rows) is not int or context_rows < 0:
        raise ValueError("Evaluator output context34 row count mismatch")
    return rows, context_rows


def validate_evaluator_output(
    *,
    payload: bytes,
    stdout: bytes,
    evaluation: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    if payload != stdout:
        raise ValueError("Evaluator stdout differs from --json-output bytes")
    result = strict_json_loads(payload, "evaluator output")
    endpoint = next(
        item for item in protocol["endpoints"]
        if item["name"] == evaluation["endpoint"]
    )
    panel_name = evaluation["panel"]
    data_name = "core5" if panel_name in {"core5", "dominic", "luca", "szlach"} else panel_name
    data = protocol["data_bindings"][data_name]
    expected_team = [] if evaluation["team_name"] is None else [evaluation["team_name"]]
    if Path(result.get("checkpoint", "")) != relative_path(
        endpoint["path"], "endpoint output checkpoint"
    ) or result.get("checkpoint_sha256") != endpoint["sha256"]:
        raise ValueError("Evaluator output checkpoint binding mismatch")
    if Path(result.get("data", "")) != relative_path(
        data["path"], "evaluator output data"
    ):
        raise ValueError("Evaluator output data binding mismatch")
    if result.get("split") != "valid" or result.get("split_mode") != "archive" or (
        result.get("split_seed") != 20260723
    ) or result.get("prediction_order") != "policy_greedy" or (
        result.get("device") != "cuda"
    ):
        raise ValueError("Evaluator output protocol fields mismatch")
    filters = result.get("filters")
    if not isinstance(filters, dict) or filters.get("deck_hashes") != [] or (
        filters.get("team_names") != expected_team
    ):
        raise ValueError("Evaluator output filter mismatch")
    rows, context34_rows = metric_shape(result)
    expected_metrics = protocol["frozen_parent_outputs"][panel_name]["metrics"]
    if rows != expected_metrics["rows"] or context34_rows != expected_metrics["context34_rows"]:
        raise ValueError("Evaluator output row/context shape differs from frozen parent")
    return result


def hold_bound_input(
    binding: dict[str, Any],
    label: str,
) -> tuple[Path, int, str]:
    path = relative_path(binding.get("path"), f"{label}.path")
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
        return path, fd, digest
    except BaseException:
        os.close(fd)
        raise


def revalidate_bound_input(
    *,
    path: Path,
    file_fd: int,
    expected_sha256: str,
    label: str,
) -> None:
    info = os.fstat(file_fd)
    visible = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"{label} no longer regular/single-link")
    if (info.st_dev, info.st_ino) != (visible.st_dev, visible.st_ino):
        raise RuntimeError(f"{label} path/descriptor identity changed")
    if sha256_bytes(read_held_bytes(file_fd, label)) != expected_sha256:
        raise RuntimeError(f"{label} held bytes changed")


def run_formal(
    protocol: dict[str, Any],
    protocol_sha256: str,
    paths: dict[str, Path],
    formal_authorization: dict[str, Any],
    cuda: dict[str, Any],
) -> dict[str, Any]:
    output_root = paths["output_root"]
    marker = paths["marker"]
    created_at = datetime.now(timezone.utc).isoformat()
    repo_parent_fd = os.open(ROOT.parent, directory_flags())
    repo_fd = os.open(ROOT.name, directory_flags(), dir_fd=repo_parent_fd)
    cwd_fd = os.open(".", directory_flags())
    if (os.fstat(repo_fd).st_dev, os.fstat(repo_fd).st_ino) != (
        os.fstat(cwd_fd).st_dev,
        os.fstat(cwd_fd).st_ino,
    ):
        raise RuntimeError("Held current directory differs from frozen repository")
    artifacts_fd = os.open("artifacts", directory_flags(), dir_fd=repo_fd)
    output_root_fd = -1
    endpoint_fds: dict[str, int] = {}
    marker_fd = -1
    manifest_fd = -1
    completion_fd = -1
    output_claims: dict[int, dict[str, Any]] = {}
    long_held_inputs: list[tuple[Path, int, str, str]] = []

    def revalidate_namespace(label: str) -> dict[str, Any]:
        root_binding = revalidate_directory_identity_at(
            parent_fd=repo_parent_fd,
            name=ROOT.name,
            directory_fd=repo_fd,
            label=f"frozen repository {label}",
        )
        if (os.fstat(cwd_fd).st_dev, os.fstat(cwd_fd).st_ino) != (
            os.fstat(repo_fd).st_dev,
            os.fstat(repo_fd).st_ino,
        ):
            raise RuntimeError(f"Held cwd/repository identity mismatch {label}")
        artifacts_binding = revalidate_directory_identity_at(
            parent_fd=repo_fd,
            name="artifacts",
            directory_fd=artifacts_fd,
            label=f"artifacts directory {label}",
        )
        result = {
            "root": root_binding,
            "artifacts": artifacts_binding,
            "cwd_repository_identity": True,
        }
        if output_root_fd >= 0:
            result["output_root"] = revalidate_directory_identity_at(
                parent_fd=artifacts_fd,
                name=output_root.name,
                directory_fd=output_root_fd,
                label=f"specialist output root {label}",
            )
        return result

    try:
        revalidate_namespace("at formal startup")
        protocol_binding = {
            "path": str(paths["protocol"].relative_to(ROOT)),
            "sha256": protocol_sha256,
        }
        authorization_binding = {
            "path": formal_authorization["path"],
            "sha256": formal_authorization["sha256"],
        }
        for binding, label in (
            (protocol_binding, "preregistration"),
            (protocol["launcher"], "launcher"),
            (authorization_binding, "formal authorization"),
        ):
            path, file_fd, digest = hold_bound_input(binding, label)
            long_held_inputs.append((path, file_fd, digest, label))
        immutable_bindings: list[tuple[dict[str, Any], str]] = [
            (
                protocol["materialization_integrity_decision"],
                "materialization integrity decision",
            ),
            (
                protocol["materialization_completion_seal"],
                "materialization completion seal",
            ),
            (protocol["evaluator"], "evaluator"),
            *[
                (binding, f"evaluator dependency {index}")
                for index, binding in enumerate(protocol["evaluator_dependencies"])
            ],
            *[
                (binding, f"endpoint {binding['name']}")
                for binding in protocol["endpoints"]
            ],
            *[
                (binding, f"data {name}")
                for name, binding in protocol["data_bindings"].items()
            ],
            (
                {
                    "path": protocol["parent"]["checkpoint"],
                    "sha256": protocol["parent"]["sha256"],
                },
                "parent checkpoint",
            ),
            (protocol["parent"]["source_decision"], "parent source decision"),
            *[
                (binding, f"frozen parent output {name}")
                for name, binding in protocol["frozen_parent_outputs"].items()
            ],
        ]
        for binding, label in immutable_bindings:
            path, file_fd, digest = hold_bound_input(binding, label)
            long_held_inputs.append((path, file_fd, digest, label))

        marker_payload = canonical_json_file_bytes({
            "schema_version": "ptcg-u468-p12-direction-specialist-attempt-v1",
            "status": "formal_attempt_claimed",
            "created_at_utc": created_at,
            "attempt": 1,
            "attempts_authorized": 1,
            "retry_authorized": False,
            "preregistration_sha256": protocol_sha256,
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "formal_authorization": formal_authorization,
        })
        marker_evidence, marker_fd = publish_exclusive_at(
            directory_fd=repo_fd,
            name=marker.name,
            payload=marker_payload,
            path=marker,
            label="specialist attempt marker",
        )
        os.fsync(repo_fd)

        os.mkdir(output_root.name, 0o700, dir_fd=artifacts_fd)
        os.fsync(artifacts_fd)
        output_root_fd = os.open(
            output_root.name,
            directory_flags(),
            dir_fd=artifacts_fd,
        )
        os.fchmod(output_root_fd, 0o700)
        for endpoint in EXPECTED_ENDPOINTS:
            os.mkdir(endpoint, 0o700, dir_fd=output_root_fd)
            endpoint_fd = os.open(endpoint, directory_flags(), dir_fd=output_root_fd)
            os.fchmod(endpoint_fd, 0o700)
            endpoint_fds[endpoint] = endpoint_fd
        os.fsync(output_root_fd)

        for evaluation in protocol["ordered_evaluations"]:
            endpoint_fd = endpoint_fds[evaluation["endpoint"]]
            name = f"{evaluation['panel']}.json"
            path = relative_path(evaluation["output"], "evaluation output")
            claim_fd = claim_empty_at(directory_fd=endpoint_fd, name=name)
            output_claims[evaluation["order"]] = {
                "fd": claim_fd,
                "directory_fd": endpoint_fd,
                "name": name,
                "path": path,
            }
        for endpoint_fd in endpoint_fds.values():
            os.fsync(endpoint_fd)

        panel_names = [name for name, _, _ in EXPECTED_PANELS]
        for endpoint in EXPECTED_ENDPOINTS:
            revalidate_directory_at(
                parent_fd=output_root_fd,
                name=endpoint,
                directory_fd=endpoint_fds[endpoint],
                expected_mode=0o700,
                expected_contents=[f"{name}.json" for name in panel_names],
                label=f"endpoint directory {endpoint} after claims",
            )
        revalidate_directory_at(
            parent_fd=artifacts_fd,
            name=output_root.name,
            directory_fd=output_root_fd,
            expected_mode=0o700,
            expected_contents=list(EXPECTED_ENDPOINTS),
            label="specialist output root after claims",
        )

        results: list[dict[str, Any]] = []
        for evaluation in protocol["ordered_evaluations"]:
            record: dict[str, Any] = {
                "order": evaluation["order"],
                "endpoint": evaluation["endpoint"],
                "panel": evaluation["panel"],
                "command_sha256": evaluation["command_sha256"],
                "attempt_consumed": True,
            }
            started = datetime.now(timezone.utc)
            held_inputs: list[tuple[Path, int, str, str]] = []
            try:
                for path, file_fd, digest, label in long_held_inputs + held_inputs:
                    revalidate_bound_input(
                        path=path,
                        file_fd=file_fd,
                        expected_sha256=digest,
                        label=label,
                    )
                revalidate_namespace(
                    f"before evaluation {evaluation['order']} child"
                )
                completed = subprocess.run(
                    evaluation["command"],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                revalidate_namespace(
                    f"after evaluation {evaluation['order']} child"
                )
                record.update({
                    "terminal_exit_code": completed.returncode,
                    "stdout_bytes": len(completed.stdout),
                    "stdout_sha256": sha256_bytes(completed.stdout),
                    "stderr_bytes": len(completed.stderr),
                    "stderr_sha256": sha256_bytes(completed.stderr),
                    "stderr_tail": completed.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-2000:],
                })
                for path, file_fd, digest, label in long_held_inputs + held_inputs:
                    revalidate_bound_input(
                        path=path,
                        file_fd=file_fd,
                        expected_sha256=digest,
                        label=label,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(f"Evaluator exited {completed.returncode}")
                claim = output_claims[evaluation["order"]]
                output_raw, output_evidence = validate_claimed_output_at(
                    directory_fd=claim["directory_fd"],
                    name=claim["name"],
                    file_fd=claim["fd"],
                    path=claim["path"],
                    label=f"evaluation output {evaluation['order']}",
                )
                parsed = validate_evaluator_output(
                    payload=output_raw,
                    stdout=completed.stdout,
                    evaluation=evaluation,
                    protocol=protocol,
                )
                metrics = parsed["metrics"]
                context34 = metrics.get("by_context", {}).get("34", {})
                record.update({
                    "status": "completed",
                    "output": output_evidence,
                    "metrics": {
                        "rows": metrics["rows"],
                        "context34_rows": context34.get("rows", 0),
                        "set": metrics["set_exact_correct"],
                        "hybrid": metrics["hybrid_order_exact_correct"],
                        "ordered": metrics["ordered_exact_correct"],
                        "value": metrics["value_correct"],
                        "count": metrics["count_correct"],
                        "top1": metrics["top1_correct"],
                        "context34_hybrid": context34.get(
                            "hybrid_order_exact_correct",
                            0,
                        ),
                        "context34_ordered": context34.get(
                            "ordered_exact_correct",
                            0,
                        ),
                    },
                })
            except BaseException as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
            finally:
                for _, file_fd, _, _ in held_inputs:
                    os.close(file_fd)
            record["elapsed_seconds"] = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
            results.append(record)

        for evaluation, record in zip(
            protocol["ordered_evaluations"],
            results,
            strict=True,
        ):
            claim = output_claims[evaluation["order"]]
            payload, evidence = validate_claimed_output_at(
                directory_fd=claim["directory_fd"],
                name=claim["name"],
                file_fd=claim["fd"],
                path=claim["path"],
                label=f"terminal evaluation output {evaluation['order']}",
            )
            if record.get("status") == "completed":
                if evidence["sha256"] != record["output"]["sha256"]:
                    raise RuntimeError("Completed output changed before manifest")
            else:
                record["terminal_output"] = evidence
                record["terminal_output_json_valid"] = False
                if payload:
                    try:
                        strict_json_loads(payload, "failed evaluator output")
                    except BaseException:
                        pass
                    else:
                        record["terminal_output_json_valid"] = True

        all_passed = len(results) == 18 and all(
            item.get("status") == "completed"
            and item.get("terminal_exit_code") == 0
            for item in results
        )
        manifest = {
            "schema_version": "ptcg-u468-p12-direction-specialist-manifest-v1",
            "status": (
                "evaluations_completed_pending_completion_seal"
                if all_passed
                else "evaluations_failed_pending_terminal_seal"
            ),
            "created_at_utc": created_at,
            "preregistration_sha256": protocol_sha256,
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "formal_authorization": formal_authorization,
            "attempt_marker": marker_evidence,
            "cuda_preflight": cuda,
            "evaluations_attempted": len(results),
            "evaluations_completed": sum(
                item.get("status") == "completed" for item in results
            ),
            "run_all_before_metric_decision": True,
            "results": results,
        }
        manifest_path = output_root / "specialist_execution_manifest.json"
        manifest_payload = canonical_json_file_bytes(manifest)
        manifest_evidence, manifest_fd = publish_exclusive_at(
            directory_fd=output_root_fd,
            name=manifest_path.name,
            payload=manifest_payload,
            path=manifest_path,
            label="specialist execution manifest",
        )
        completion_path = output_root / "COMPLETED.json"
        completion_fd = claim_empty_at(
            directory_fd=output_root_fd,
            name=completion_path.name,
        )
        os.fsync(output_root_fd)

        expected_root_contents = [
            *EXPECTED_ENDPOINTS,
            manifest_path.name,
            completion_path.name,
        ]
        endpoint_bindings: dict[str, Any] = {}
        for endpoint in EXPECTED_ENDPOINTS:
            endpoint_bindings[endpoint] = revalidate_directory_at(
                parent_fd=output_root_fd,
                name=endpoint,
                directory_fd=endpoint_fds[endpoint],
                expected_mode=0o700,
                expected_contents=[f"{name}.json" for name in panel_names],
                label=f"endpoint directory {endpoint} before seal",
            )
        output_root_binding = revalidate_directory_at(
            parent_fd=artifacts_fd,
            name=output_root.name,
            directory_fd=output_root_fd,
            expected_mode=0o700,
            expected_contents=expected_root_contents,
            label="specialist output root before seal",
        )
        marker_evidence = revalidate_published_at(
            directory_fd=repo_fd,
            name=marker.name,
            file_fd=marker_fd,
            expected_sha256=marker_evidence["sha256"],
            expected_mode=0o600,
            path=marker,
            label="specialist attempt marker before seal",
        )
        manifest_evidence = revalidate_published_at(
            directory_fd=output_root_fd,
            name=manifest_path.name,
            file_fd=manifest_fd,
            expected_sha256=manifest_evidence["sha256"],
            expected_mode=0o600,
            path=manifest_path,
            label="specialist execution manifest before seal",
        )
        for evaluation, record in zip(
            protocol["ordered_evaluations"],
            results,
            strict=True,
        ):
            claim = output_claims[evaluation["order"]]
            expected_sha = (
                record["output"]["sha256"]
                if record.get("status") == "completed"
                else record["terminal_output"]["sha256"]
            )
            revalidate_published_at(
                directory_fd=claim["directory_fd"],
                name=claim["name"],
                file_fd=claim["fd"],
                expected_sha256=expected_sha,
                expected_mode=0o600,
                path=claim["path"],
                label=f"evaluation output {evaluation['order']} before seal",
            )
        for path, file_fd, digest, label in long_held_inputs:
            revalidate_bound_input(
                path=path,
                file_fd=file_fd,
                expected_sha256=digest,
                label=label,
            )
        namespace_binding = revalidate_namespace("before directory seal")

        for endpoint_fd in endpoint_fds.values():
            os.fchmod(endpoint_fd, 0o500)
            os.fsync(endpoint_fd)
        os.fchmod(output_root_fd, 0o500)
        os.fsync(output_root_fd)
        for endpoint in EXPECTED_ENDPOINTS:
            endpoint_bindings[endpoint] = revalidate_directory_at(
                parent_fd=output_root_fd,
                name=endpoint,
                directory_fd=endpoint_fds[endpoint],
                expected_mode=0o500,
                expected_contents=[f"{name}.json" for name in panel_names],
                label=f"sealed endpoint directory {endpoint}",
            )
        output_root_binding = revalidate_directory_at(
            parent_fd=artifacts_fd,
            name=output_root.name,
            directory_fd=output_root_fd,
            expected_mode=0o500,
            expected_contents=expected_root_contents,
            label="sealed specialist output root",
        )
        manifest_evidence = revalidate_published_at(
            directory_fd=output_root_fd,
            name=manifest_path.name,
            file_fd=manifest_fd,
            expected_sha256=manifest_evidence["sha256"],
            expected_mode=0o600,
            path=manifest_path,
            label="sealed specialist execution manifest",
        )
        for evaluation, record in zip(
            protocol["ordered_evaluations"],
            results,
            strict=True,
        ):
            claim = output_claims[evaluation["order"]]
            expected_sha = (
                record["output"]["sha256"]
                if record.get("status") == "completed"
                else record["terminal_output"]["sha256"]
            )
            revalidate_published_at(
                directory_fd=claim["directory_fd"],
                name=claim["name"],
                file_fd=claim["fd"],
                expected_sha256=expected_sha,
                expected_mode=0o600,
                path=claim["path"],
                label=f"sealed evaluation output {evaluation['order']}",
            )

        completion = {
            "schema_version": "ptcg-u468-p12-direction-specialist-completion-seal-v1",
            "status": "completed" if all_passed else "terminal_failed_no_retry",
            "created_at_utc": created_at,
            "preregistration_sha256": protocol_sha256,
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "formal_authorization": formal_authorization,
            "attempt_marker": marker_evidence,
            "manifest": manifest_evidence,
            "evaluations_attempted": len(results),
            "evaluations_completed": sum(
                item.get("status") == "completed" for item in results
            ),
            "retry_authorized": False,
            "all_outputs_require_separate_metric_decision": True,
            "broad_gold_package_upload_submission_authorized": False,
            "output_directory": output_root_binding,
            "endpoint_directories": endpoint_bindings,
            "namespace_binding": namespace_binding,
        }
        completion_evidence = populate_claim_at(
            directory_fd=output_root_fd,
            name=completion_path.name,
            file_fd=completion_fd,
            path=completion_path,
            payload=canonical_json_file_bytes(completion),
        )
        return {**completion, "completion_publication": completion_evidence}
    finally:
        for claim in output_claims.values():
            try:
                os.close(claim["fd"])
            except OSError:
                pass
        for _, file_fd, _, _ in long_held_inputs:
            try:
                os.close(file_fd)
            except OSError:
                pass
        for file_fd in (completion_fd, manifest_fd, marker_fd):
            if file_fd >= 0:
                try:
                    os.close(file_fd)
                except OSError:
                    pass
        for endpoint_fd in endpoint_fds.values():
            try:
                os.close(endpoint_fd)
            except OSError:
                pass
        for file_fd in (
            output_root_fd,
            artifacts_fd,
            cwd_fd,
            repo_fd,
            repo_parent_fd,
        ):
            if file_fd >= 0:
                try:
                    os.close(file_fd)
                except OSError:
                    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--formal-authorization", type=Path)
    parser.add_argument("--expected-formal-authorization-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    protocol, protocol_sha256, paths = validate_protocol(args)
    cuda = cuda_preflight()
    if args.mode == "preflight":
        if args.formal_authorization is not None or (
            args.expected_formal_authorization_sha256 is not None
        ):
            raise ValueError("Preflight must not receive formal authorization")
        for target, label in (
            (paths["marker"], "specialist attempt marker"),
            (paths["output_root"], "specialist output root"),
        ):
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    f"{label} appeared during zero-write preflight"
                )
        preview = {
            "schema_version": "ptcg-u468-p12-direction-specialist-preflight-v1",
            "status": "preflight_passed_no_writes",
            "preregistration_sha256": protocol_sha256,
            "launcher_sha256": protocol["launcher"]["sha256"],
            "ordered_command_matrix_sha256": protocol["ordered_command_matrix_sha256"],
            "evaluation_count": len(protocol["ordered_evaluations"]),
            "cuda_preflight": cuda,
            "writes": 0,
            "targets_remained_absent": True,
        }
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True, indent=2))
        return
    formal_authorization = validate_formal_authorization(
        args,
        protocol,
        protocol_sha256,
    )
    result = run_formal(
        protocol,
        protocol_sha256,
        paths,
        formal_authorization,
        cuda,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
