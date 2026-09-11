#!/usr/bin/env python3
"""Materialize the frozen CW10 full-train-PASS candidate exactly once.

Static mode is read-only and CPU-only.  Materialize mode reruns the exact
hash-bound v5 gate, wraps its same-process candidate consumer to retain one CPU
clone, and refuses to serialize unless every terminal and formal full-train
gate is exact.  The raw U468 checkpoint is the template: its inference
metadata is preserved, ``model_state_dict`` is replaced, stale metrics and all
resumable-training state are omitted, and one namespaced audit record is
appended.  The resulting checkpoint is explicitly evaluation-only.

Publication uses a fresh fixed staging directory followed by Linux
``renameat2(RENAME_NOREPLACE)``.  No output path is ever overwritten.  This
tool has no network, upload, packaging, or submission path.
"""

from __future__ import annotations

import argparse
import ast
import copy
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import os
import stat
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "materialize_u468_raw_actor6_metricguard_fulltrain_cw10_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-cw10-materializer-v1"
V5_SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-gate-v5"
MANIFEST_SCHEMA = f"{SCHEMA}-manifest"
COMPLETION_SCHEMA = f"{SCHEMA}-completion"
AUDIT_KEY = "metricguard_fulltrain_cw10_materialization"

V5 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v5.py"
V5_SHA256 = "90ab439a0ca75e7f0a2cf2c8e7c117a07ee5ff9b8b6462ffc5e8380114b404a0"
CW10 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1.py"
CW10_SHA256 = "546e90c5b3ca35c84b8efc08109b2f310b45b2895d24aeee70561d13bffa146c"
V4 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v4.py"
V4_SHA256 = "75974150f83455c10bbf6a90b3c571e4daf319f4e06e9c77f67673fbf34b9ea8"
BASE = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v3.py"
BASE_SHA256 = "f32c077c0d577bcc7c0ad2e42bdfda1ec1641244004b29efd5c82008cfe92338"
RAW_U468 = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_U468_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
FROZEN_MODE = 0o555

EXPECTED_ITERATION = 9
EXPECTED_L2 = 0.007627603437990896
EXPECTED_VECTOR_SHA256 = "42e02c90f65b81e1e8b801fa173fcf953b347f73e65b58000719107fa83a6c28"
EXPECTED_MODEL_SHA256 = "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
EXPECTED_LEDGER_SHA256 = "22b20b8630d63c6fbfd249a82a188397b468a84d1b747a2159c17b05a6f4983d"
EXPECTED_ROWS_SELECTED = 29
EXPECTED_ACTIVE_PAIRS = 30
EXPECTED_PANEL_ROWS = {"flg": 9443, "pokemonfan": 9487, "core5": 5120}
EXPECTED_PF_WC_NET = {
    "set_exact": 6,
    "hybrid_order_exact": 6,
    "ordered_exact": 7,
    "top1_correct": 7,
}
EXPECTED_CHANGED = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
)
EXPECTED_UNCHANGED_ACTOR = ("actor_residual.2.bias",)
REQUIRED_SLIM_KEYS = (
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "action_distribution",
)
FORBIDDEN_RESUME_OR_STALE_KEYS = {
    "optimizer_state_dict",
    "optimizer_parameter_names",
    "bc_replay_optimizer_state_dict",
    "opponent_quota_state",
    "fresh_special_optimizer_state_dict",
    "metrics",
    "value_trunk_gradient",
    "actor_value_gradient",
}

BRANCH = "ppo_u468_raw_actor6_metricguard_fulltrain_cw10_materialized_v1_20260802"
OUTPUT_ROOT = ROOT / "artifacts" / BRANCH
STAGING_ROOT = OUTPUT_ROOT.with_name(f".{BRANCH}.staging")
CHECKPOINT_NAME = "u468-cw10-fulltrain-pass-eval-only.pt"
MANIFEST_NAME = "materialization_manifest.json"
COMPLETION_NAME = "COMPLETED.json"


def canonical_json(value: Any) -> bytes:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_regular_bytes(
    path: Path,
    expected_sha256: str | None,
    label: str,
    *,
    expected_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    payload = b"".join(chunks)
    visible = os.lstat(path)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or visible.st_nlink != 1
        or after.st_size != len(payload)
    ):
        raise RuntimeError(f"{label} changed while reading")
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "device": after.st_dev,
        "inode": after.st_ino,
        "nlink": after.st_nlink,
    }


def import_frozen(
    path: Path, expected_sha256: str, label: str
) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, expected_sha256, label, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(
        f"_materializer_{path.stem}_{expected_sha256[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {label}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def require_lower_hex64(value: str, label: str) -> None:
    if len(value) != 64 or value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"{label} must be lowercase SHA-256 hex")


def require_outputs_absent() -> dict[str, bool]:
    result = {
        "output_root_absent": not (OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink()),
        "staging_root_absent": not (
            STAGING_ROOT.exists() or STAGING_ROOT.is_symlink()
        ),
    }
    if not all(result.values()):
        raise FileExistsError(f"materialization target already claimed: {result}")
    parent = OUTPUT_ROOT.parent
    observed = os.lstat(parent)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise RuntimeError("artifacts parent is not a real directory")
    if parent.resolve() != parent:
        raise RuntimeError("artifacts parent path identity drift")
    return result


def fixed_inputs() -> tuple[dict[str, Any], bytes]:
    evidence: dict[str, Any] = {}
    for name, path, digest, mode in (
        ("fulltrain_v5", V5, V5_SHA256, FROZEN_MODE),
        ("cw10_probe", CW10, CW10_SHA256, FROZEN_MODE),
        ("fulltrain_v4", V4, V4_SHA256, FROZEN_MODE),
        ("fulltrain_v3", BASE, BASE_SHA256, FROZEN_MODE),
    ):
        _, evidence[name] = read_regular_bytes(
            path, digest, name.replace("_", " "), expected_mode=mode
        )
    raw_payload, evidence["raw_u468"] = read_regular_bytes(
        RAW_U468, RAW_U468_SHA256, "raw U468 checkpoint"
    )
    return evidence, raw_payload


def load_modules() -> tuple[
    ModuleType,
    ModuleType,
    ModuleType,
    ModuleType,
    dict[str, Any],
]:
    v5, v5_evidence = import_frozen(V5, V5_SHA256, "frozen fulltrain v5")
    cw10, cw10_evidence = import_frozen(CW10, CW10_SHA256, "frozen CW10 probe")
    v4, v4_evidence = import_frozen(V4, V4_SHA256, "frozen fulltrain v4")
    base, base_evidence = import_frozen(BASE, BASE_SHA256, "frozen fulltrain v3")
    bindings = {
        "v5_constants_exact": (
            Path(v5.CW10) == CW10
            and v5.CW10_SHA256 == CW10_SHA256
            and Path(v5.V4) == V4
            and v5.V4_SHA256 == V4_SHA256
            and Path(v4.BASE_GATE) == BASE
            and v4.BASE_GATE_SHA256 == BASE_SHA256
        ),
        "terminal_constants_exact": (
            v5.EXPECTED_SUCCESS_ITERATION == EXPECTED_ITERATION
            and v5.EXPECTED_TERMINAL_CUMULATIVE_L2 == EXPECTED_L2
            and v5.EXPECTED_TERMINAL_CUMULATIVE_SHA256 == EXPECTED_VECTOR_SHA256
            and v5.EXPECTED_TERMINAL_MODEL_STATE_SHA256 == EXPECTED_MODEL_SHA256
            and v5.EXPECTED_ACTIVE_PAIR_COUNT == EXPECTED_ACTIVE_PAIRS
            and v5.EXPECTED_ACTIVE_PAIR_LEDGER_SHA256 == EXPECTED_LEDGER_SHA256
            and v5.EXPECTED_SELECTED_ROW_COUNT == EXPECTED_ROWS_SELECTED
        ),
    }
    if not all(bindings.values()):
        raise RuntimeError(f"frozen module binding drift: {bindings}")
    return v5, cw10, v4, base, {
        "fulltrain_v5": v5_evidence,
        "cw10_probe": cw10_evidence,
        "fulltrain_v4": v4_evidence,
        "fulltrain_v3": base_evidence,
        "binding_checks": bindings,
    }


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    calls = [
        dotted_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    forbidden_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        forbidden_imports.extend(
            name
            for name in names
            if name.split(".", 1)[0]
            in {"requests", "urllib", "http", "socket", "subprocess", "kaggle"}
        )
    forbidden_calls = sorted(
        name
        for name in calls
        if name.rsplit(".", 1)[-1]
        in {"backward", "step", "unlink", "remove", "rmtree"}
        or name in {"os.replace", "os.rename", "Path.replace", "Path.rename"}
    )
    counts = {
        "v5.run_gate": calls.count("v5.run_gate"),
        "torch.save": calls.count("torch.save"),
        "os.mkdir": calls.count("os.mkdir"),
        "publish_staged_file": calls.count("publish_staged_file"),
        "atomic_publish_directory": calls.count("atomic_publish_directory"),
    }
    expected = {
        "v5.run_gate": 1,
        "torch.save": 1,
        "os.mkdir": 1,
        "publish_staged_file": 3,
        "atomic_publish_directory": 1,
    }
    checks = {
        "ast_parse": True,
        "no_network_submission_imports": not forbidden_imports,
        "no_optimizer_backward_delete_or_overwrite_calls": not forbidden_calls,
        "actual_path_call_counts_exact": counts == expected,
        "fixed_versioned_output_root": OUTPUT_ROOT.name == BRANCH,
        "staging_and_output_distinct": STAGING_ROOT != OUTPUT_ROOT,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"materializer static audit failed: {checks}, {counts}, {forbidden_calls}"
        )
    return {
        "checks": checks,
        "observed_calls": counts,
        "forbidden_imports": forbidden_imports,
        "forbidden_calls": forbidden_calls,
        "static_writes": 0,
        "static_cuda_access": False,
    }


def update_digest(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def typed_fingerprint(value: Any, torch: ModuleType) -> str:
    digest = hashlib.sha256()

    def walk(item: Any) -> None:
        update_digest(digest, type(item).__module__.encode())
        update_digest(digest, type(item).__qualname__.encode())
        if item is None:
            return
        if isinstance(item, bool):
            update_digest(digest, b"1" if item else b"0")
        elif isinstance(item, int):
            update_digest(digest, str(item).encode("ascii"))
        elif isinstance(item, float):
            update_digest(digest, struct.pack("<d", item))
        elif isinstance(item, str):
            update_digest(digest, item.encode("utf-8"))
        elif isinstance(item, bytes):
            update_digest(digest, item)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            update_digest(digest, str(tensor.dtype).encode("ascii"))
            update_digest(digest, canonical_json(list(tensor.shape)))
            update_digest(digest, canonical_json(list(tensor.stride())))
            update_digest(
                digest,
                tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
        elif isinstance(item, Mapping):
            update_digest(digest, struct.pack("<Q", len(item)))
            for key, child in item.items():
                walk(key)
                walk(child)
        elif isinstance(item, (list, tuple)):
            update_digest(digest, struct.pack("<Q", len(item)))
            for child in item:
                walk(child)
        else:
            raise TypeError(f"unsupported checkpoint metadata type: {type(item)!r}")

    walk(value)
    return digest.hexdigest()


def metadata_without_model(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}


def validate_gate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    decision = result.get("decision")
    capture = result.get("candidate_capture")
    execution = result.get("execution")
    adapter = result.get("v5_compatibility_adapter_audit")
    scope = result.get("scope_audit")
    if not all(isinstance(value, Mapping) for value in (decision, capture, execution, adapter, scope)):
        raise RuntimeError("v5 result schema drift")
    panel_gates = decision.get("panel_gates")
    if not isinstance(panel_gates, Mapping) or set(panel_gates) != set(EXPECTED_PANEL_ROWS):
        raise RuntimeError("v5 panel-gate schema drift")
    panel_checks = {}
    for panel, rows in EXPECTED_PANEL_ROWS.items():
        gate = panel_gates[panel]
        panel_checks[panel] = (
            gate.get("pass") is True
            and int(gate.get("rows", -1)) == rows
            and all(int(value) == 0 for value in gate["safety_cw_counts"].values())
            and all(gate["checks"].values())
        )
    checks = {
        "schema_exact": result.get("schema_version") == V5_SCHEMA,
        "status_and_decision_pass": (
            result.get("status") == "full_train_gate_pass"
            and decision.get("status") == "full_train_gate_pass"
            and decision.get("pass") is True
            and decision.get("materialization_allowed_next") is True
            and decision.get("model_materialized") is False
            and decision.get("submission_performed") is False
        ),
        "terminal_candidate_exact": (
            capture.get("candidate_model_state_sha256") == EXPECTED_MODEL_SHA256
            and capture.get("terminal_cumulative_float64_le_sha256") == EXPECTED_VECTOR_SHA256
            and capture.get("active_pair_ledger_sha256") == EXPECTED_LEDGER_SHA256
            and int(capture.get("active_pair_count", -1)) == EXPECTED_ACTIVE_PAIRS
            and abs(float(capture.get("terminal_cumulative_l2")) - EXPECTED_L2) <= 1e-15
        ),
        "probe_terminal_exact": (
            result["probe_result"].get("status")
            == "exploratory_29row_30pair_optimization_success"
            and int(result["probe_result"]["decision"]["success_iteration_after_c0"])
            == EXPECTED_ITERATION
            and result["probe_result"]["decision"]["candidate_model_state_sha256_before_restore"]
            == EXPECTED_MODEL_SHA256
        ),
        "actual_29row_adapter_exact": (
            adapter["checks"].get("actual_selected_row_count_exact_29") is True
            and adapter.get("all_dependency_globals_and_functions_restored") is True
        ),
        "formal_2x3_exact": (
            int(execution.get("batch_size", -1)) == 256
            and int(execution.get("evaluation_count_exact", -1)) == 6
            and all(execution["checks"].values())
            and len(execution.get("completion_ledger", [])) == 6
        ),
        "all_panels_zero_cw_and_integrity": all(panel_checks.values()),
        "pokemonfan_wc_exact": decision.get("pokemonfan_observed_wc") == EXPECTED_PF_WC_NET,
        "pokemonfan_net_exact": decision.get("pokemonfan_observed_net_gains") == EXPECTED_PF_WC_NET,
        "zero_write_validation_submission_scope": (
            scope.get("selected_29_train_rows_then_all_24050_train_rows") is True
            and scope.get("validation_members_opened") is False
            and scope.get("validation_results_read") is False
            and int(scope.get("model_artifact_writes", -1)) == 0
            and int(scope.get("evidence_file_writes", -1)) == 0
            and scope.get("network_upload_submission") is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v5 full-train materialization authorization failed: {checks}")
    return {"checks": checks, "panel_checks": panel_checks}


def run_and_capture_candidate(
    source: bytes,
    v5: ModuleType,
    cw10: ModuleType,
    v4: ModuleType,
    base: ModuleType,
    module_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    local_static = v5.static_audit(source, cw10, v4)
    original_capture = v4.capture_candidate
    captured: dict[str, Any] = {"calls": 0}

    def capture_for_materialization(
        context: Mapping[str, Any],
        holder: dict[str, Any],
        base_module: ModuleType,
        geometry: ModuleType,
    ) -> None:
        original_capture(context, holder, base_module, geometry)
        captured["calls"] += 1
        if captured["calls"] != 1:
            raise RuntimeError("materialization candidate captured more than once")
        helper = context["helper"]
        candidate_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in holder["candidate_state"].items()
        }
        if helper.model_state_sha256(candidate_state) != EXPECTED_MODEL_SHA256:
            raise RuntimeError("captured candidate SHA drift")
        captured.update(
            {
                "helper": helper,
                "candidate_state": candidate_state,
                "checkpoint_keys": list(context["checkpoint"].keys()),
                "checkpoint_metadata_fingerprint": typed_fingerprint(
                    metadata_without_model(context["checkpoint"]), helper.torch
                ),
                "legacy_row_count_seen_by_frozen_v4": int(context["expanded_row_count"]),
                "candidate_model_state_sha256": EXPECTED_MODEL_SHA256,
            }
        )

    v4.capture_candidate = capture_for_materialization
    restored = False
    try:
        result = v5.run_gate(
            source,
            local_static,
            cw10,
            module_evidence["cw10_probe"],
            v4,
            module_evidence["fulltrain_v4"],
            base,
            module_evidence["fulltrain_v3"],
        )
    finally:
        v4.capture_candidate = original_capture
        restored = v4.capture_candidate is original_capture
    if not restored or captured.get("calls") != 1:
        raise RuntimeError("materialization capture hook integrity failed")
    gate_audit = validate_gate_result(result)
    captured["capture_hook_restored"] = True
    captured["gate_audit"] = gate_audit
    return result, captured


def verify_checkpoint(
    checkpoint: Mapping[str, Any],
    raw_checkpoint: Mapping[str, Any],
    helper: ModuleType,
    expected_gate_sha256: str,
) -> dict[str, Any]:
    torch = helper.torch
    expected_keys = [
        *REQUIRED_SLIM_KEYS,
        "model_state_dict",
        "update",
        "evaluation_only",
        "resume_forbidden",
        "optimizer_states_omitted",
        AUDIT_KEY,
    ]
    if list(checkpoint.keys()) != expected_keys:
        raise RuntimeError("materialized checkpoint top-level key/order drift")
    raw_state = raw_checkpoint.get("model_state_dict")
    candidate_state = checkpoint.get("model_state_dict")
    if not isinstance(raw_state, Mapping) or not isinstance(candidate_state, Mapping):
        raise RuntimeError("checkpoint lacks model_state_dict mapping")
    raw_sha = helper.model_state_sha256(raw_state)
    candidate_sha = helper.model_state_sha256(candidate_state)
    changed = []
    for name in raw_state:
        if name not in candidate_state:
            raise RuntimeError("candidate state key set drift")
        raw_tensor = raw_state[name]
        candidate_tensor = candidate_state[name]
        if (
            raw_tensor.shape != candidate_tensor.shape
            or raw_tensor.dtype != candidate_tensor.dtype
            or not bool(torch.isfinite(candidate_tensor).all())
        ):
            raise RuntimeError(f"candidate tensor schema/finite drift: {name}")
        if not torch.equal(raw_tensor, candidate_tensor):
            changed.append(name)
    audit = checkpoint.get(AUDIT_KEY)
    checks = {
        "raw_model_sha_exact": raw_sha == RAW_MODEL_SHA256,
        "candidate_model_sha_exact": candidate_sha == EXPECTED_MODEL_SHA256,
        "state_key_count_exact_80": len(raw_state) == len(candidate_state) == 80,
        "changed_actor_set_exact_five": set(changed) == set(EXPECTED_CHANGED),
        "evaluation_only_top_level": checkpoint.get("evaluation_only") is True,
        "resume_forbidden_top_level": checkpoint.get("resume_forbidden") is True,
        "update_exact_468": checkpoint.get("update") == 468,
        "forbidden_resume_or_stale_keys_absent": not (
            FORBIDDEN_RESUME_OR_STALE_KEYS.intersection(checkpoint)
        ),
        "optimizer_states_omitted_exact": checkpoint.get("optimizer_states_omitted")
        == sorted(FORBIDDEN_RESUME_OR_STALE_KEYS.intersection(raw_checkpoint)),
        "audit_record_exact": (
            isinstance(audit, Mapping)
            and audit.get("schema_version") == SCHEMA
            and audit.get("candidate_model_state_sha256") == EXPECTED_MODEL_SHA256
            and audit.get("fulltrain_gate_result_canonical_sha256")
            == expected_gate_sha256
            and audit.get("evaluation_only") is True
            and audit.get("resume_forbidden") is True
        ),
    }
    for key in REQUIRED_SLIM_KEYS:
        if typed_fingerprint(checkpoint[key], torch) != typed_fingerprint(
            raw_checkpoint[key], torch
        ):
            raise RuntimeError(f"raw checkpoint metadata changed: {key}")
    checks["required_slim_metadata_exact"] = True
    if not all(checks.values()):
        raise RuntimeError(f"materialized checkpoint verification failed: {checks}")
    return {
        "checks": checks,
        "raw_model_state_sha256": raw_sha,
        "candidate_model_state_sha256": candidate_sha,
        "state_tensor_count": len(candidate_state),
        "changed_parameter_names": changed,
        "unchanged_actor_parameter_names": list(EXPECTED_UNCHANGED_ACTOR),
        "preserved_metadata_keys": list(REQUIRED_SLIM_KEYS),
        "omitted_resume_or_stale_keys": sorted(
            FORBIDDEN_RESUME_OR_STALE_KEYS.intersection(raw_checkpoint)
        ),
        "slim_metadata_preserved_exact": True,
        "optimizer_quota_and_stale_metrics_absent": True,
    }


def publish_staged_file(dir_fd: int, name: str, payload: bytes) -> dict[str, Any]:
    if Path(name).name != name:
        raise RuntimeError("staged publication name must be a basename")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o400, dir_fd=dir_fd)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise RuntimeError("short staged publication write")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.lseek(fd, 0, os.SEEK_SET)
        reloaded = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            reloaded += chunk
        observed = os.fstat(fd)
        visible = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        if (
            reloaded != payload
            or not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (visible.st_dev, visible.st_ino, visible.st_size)
        ):
            raise RuntimeError("staged file descriptor/path verification failed")
        return {
            "name": name,
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "mode_octal": format(stat.S_IMODE(observed.st_mode), "04o"),
            "device": observed.st_dev,
            "inode": observed.st_ino,
        }
    finally:
        os.close(fd)


def renameat2_available() -> bool:
    return getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None) is not None


def atomic_publish_directory() -> dict[str, Any]:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 unavailable; refusing non-atomic publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    before = os.lstat(STAGING_ROOT)
    ctypes.set_errno(0)
    rc = renameat2(
        -100,
        os.fsencode(STAGING_ROOT),
        -100,
        os.fsencode(OUTPUT_ROOT),
        1,
    )
    if rc != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(f"output root won atomic no-replace race: {OUTPUT_ROOT}")
        raise OSError(error_number, os.strerror(error_number))
    after = os.lstat(OUTPUT_ROOT)
    if STAGING_ROOT.exists() or STAGING_ROOT.is_symlink():
        raise RuntimeError("staging path remains after atomic publish")
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise RuntimeError("atomic directory publication inode drift")
    parent_fd = os.open(OUTPUT_ROOT.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {
        "method": "renameat2(RENAME_NOREPLACE)",
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "no_overwrite": True,
        "atomic_directory_visibility": True,
    }


def materialize(
    self_evidence: Mapping[str, Any],
    initial_inputs: Mapping[str, Any],
    raw_payload: bytes,
    v5: ModuleType,
    cw10: ModuleType,
    v4: ModuleType,
    base: ModuleType,
    module_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if not renameat2_available():
        raise RuntimeError("renameat2 unavailable")
    require_outputs_absent()
    v5_source, _ = read_regular_bytes(V5, V5_SHA256, "fulltrain v5", expected_mode=FROZEN_MODE)
    gate_result, captured = run_and_capture_candidate(
        v5_source, v5, cw10, v4, base, module_evidence
    )
    gate_sha256 = sha256_bytes(canonical_json(gate_result))
    final_inputs, final_raw_payload = fixed_inputs()
    if final_inputs != initial_inputs or final_raw_payload != raw_payload:
        raise RuntimeError("fixed input identity changed during full-train gate")
    require_outputs_absent()

    helper = captured["helper"]
    torch = helper.torch
    raw_checkpoint = torch.load(io.BytesIO(raw_payload), map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, dict) or AUDIT_KEY in raw_checkpoint:
        raise RuntimeError("raw U468 checkpoint root/audit namespace drift")
    raw_metadata_fingerprint = typed_fingerprint(
        metadata_without_model(raw_checkpoint), torch
    )
    if (
        captured["checkpoint_keys"] != list(raw_checkpoint.keys())
        or captured["checkpoint_metadata_fingerprint"] != raw_metadata_fingerprint
    ):
        raise RuntimeError("same-process consumer checkpoint differs from raw template")

    created_at = utc_now()
    audit_metadata = {
        "schema_version": SCHEMA,
        "created_at_utc": created_at,
        "branch": BRANCH,
        "materializer": dict(self_evidence),
        "fixed_inputs": dict(initial_inputs),
        "frozen_modules": dict(module_evidence),
        "raw_checkpoint_file_sha256": RAW_U468_SHA256,
        "raw_model_state_sha256": RAW_MODEL_SHA256,
        "raw_nonactor_sha256": RAW_NONACTOR_SHA256,
        "raw_metadata_typed_fingerprint_sha256": raw_metadata_fingerprint,
        "candidate_model_state_sha256": EXPECTED_MODEL_SHA256,
        "terminal": {
            "success_iteration": EXPECTED_ITERATION,
            "cumulative_l2": EXPECTED_L2,
            "cumulative_float64_le_sha256": EXPECTED_VECTOR_SHA256,
            "active_pair_count": EXPECTED_ACTIVE_PAIRS,
            "active_pair_ledger_sha256": EXPECTED_LEDGER_SHA256,
            "selected_row_count": EXPECTED_ROWS_SELECTED,
        },
        "fulltrain_gate_result_canonical_sha256": gate_sha256,
        "fulltrain_gate_result": gate_result,
        "same_process_candidate_consumer": {
            "calls": captured["calls"],
            "capture_hook_restored": captured["capture_hook_restored"],
            "candidate_cpu_clone_before_generator_raw_restore": True,
            "generator_and_formal_finally_raw_restore_passed": True,
        },
        "evaluation_only": True,
        "resume_forbidden": True,
        "resume_warning": (
            "optimizer/quota state and stale raw metrics are intentionally omitted; "
            "this endpoint must never be used to resume training"
        ),
        "submission_performed": False,
    }
    missing = [key for key in REQUIRED_SLIM_KEYS if key not in raw_checkpoint]
    if missing:
        raise RuntimeError(f"raw U468 lacks required slim metadata: {missing}")
    checkpoint = {
        key: copy.deepcopy(raw_checkpoint[key]) for key in REQUIRED_SLIM_KEYS
    }
    checkpoint["model_state_dict"] = captured["candidate_state"]
    checkpoint["update"] = 468
    checkpoint["evaluation_only"] = True
    checkpoint["resume_forbidden"] = True
    checkpoint["optimizer_states_omitted"] = sorted(
        FORBIDDEN_RESUME_OR_STALE_KEYS.intersection(raw_checkpoint)
    )
    checkpoint[AUDIT_KEY] = audit_metadata
    pre_serialize_audit = verify_checkpoint(
        checkpoint, raw_checkpoint, helper, gate_sha256
    )
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    checkpoint_payload = buffer.getvalue()
    reloaded_checkpoint = torch.load(
        io.BytesIO(checkpoint_payload), map_location="cpu", weights_only=False
    )
    reload_audit = verify_checkpoint(
        reloaded_checkpoint, raw_checkpoint, helper, gate_sha256
    )
    if reload_audit != pre_serialize_audit:
        raise RuntimeError("in-memory serialized checkpoint semantic drift")

    checkpoint_record = {
        "path": str((OUTPUT_ROOT / CHECKPOINT_NAME).relative_to(ROOT)),
        "sha256": sha256_bytes(checkpoint_payload),
        "bytes": len(checkpoint_payload),
        "model_state_sha256": EXPECTED_MODEL_SHA256,
        "verification": reload_audit,
    }
    manifest_payload = canonical_json(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "staged_complete_pending_atomic_directory_publish",
            "created_at_utc": created_at,
            "branch": BRANCH,
            "checkpoint": checkpoint_record,
            "raw_template": initial_inputs["raw_u468"],
            "frozen_modules": module_evidence,
            "terminal": audit_metadata["terminal"],
            "fulltrain_gate_result_canonical_sha256": gate_sha256,
            "fulltrain_gate_audit": captured["gate_audit"],
            "fulltrain_completion_ledger": gate_result["execution"]["completion_ledger"],
            "pokemonfan_observed_wc": gate_result["decision"]["pokemonfan_observed_wc"],
            "pokemonfan_observed_net_gains": gate_result["decision"]["pokemonfan_observed_net_gains"],
            "metadata_policy": {
                "required_slim_metadata_preserved": list(REQUIRED_SLIM_KEYS),
                "omitted_resume_or_stale_keys": sorted(
                    FORBIDDEN_RESUME_OR_STALE_KEYS.intersection(raw_checkpoint)
                ),
                "model_state_dict_replaced": True,
                "one_namespaced_audit_entry_appended": AUDIT_KEY,
                "evaluation_only": True,
                "resume_forbidden": True,
            },
            "publication": {
                "staging_root": str(STAGING_ROOT.relative_to(ROOT)),
                "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
                "method": "renameat2(RENAME_NOREPLACE)",
                "no_overwrite": True,
                "files_mode": "0444",
                "directory_mode": "0555",
            },
            "validation_members_opened": False,
            "network_upload_submission": False,
        }
    )
    manifest_record = {
        "path": str((OUTPUT_ROOT / MANIFEST_NAME).relative_to(ROOT)),
        "sha256": sha256_bytes(manifest_payload),
        "bytes": len(manifest_payload),
    }
    completion_payload = canonical_json(
        {
            "schema_version": COMPLETION_SCHEMA,
            "status": "complete_only_after_atomic_directory_publish",
            "branch": BRANCH,
            "checkpoint": checkpoint_record,
            "manifest": manifest_record,
            "evaluation_only": True,
            "resume_forbidden": True,
            "submission_performed": False,
        }
    )

    os.mkdir(STAGING_ROOT, mode=0o700)
    staging_fd = os.open(STAGING_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        staged_checkpoint = publish_staged_file(
            staging_fd, CHECKPOINT_NAME, checkpoint_payload
        )
        staged_manifest = publish_staged_file(
            staging_fd, MANIFEST_NAME, manifest_payload
        )
        staged_completion = publish_staged_file(
            staging_fd, COMPLETION_NAME, completion_payload
        )
        os.fsync(staging_fd)
        os.fchmod(staging_fd, 0o555)
        os.fsync(staging_fd)
    finally:
        os.close(staging_fd)
    if (
        staged_checkpoint["sha256"] != checkpoint_record["sha256"]
        or staged_manifest["sha256"] != manifest_record["sha256"]
    ):
        raise RuntimeError("staged output hash drift")
    publication = atomic_publish_directory()
    return {
        "schema_version": COMPLETION_SCHEMA,
        "status": "materialized_eval_only_atomic_no_overwrite",
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "checkpoint": checkpoint_record,
        "manifest": manifest_record,
        "completion_sha256": staged_completion["sha256"],
        "publication": publication,
        "evaluation_only": True,
        "resume_forbidden": True,
        "submission_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "materialize"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expected-self-sha256")
    parser.add_argument("--confirm-output-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    self_source, self_evidence = read_regular_bytes(
        SCRIPT, None, "materializer", expected_mode=FROZEN_MODE
    )
    static = static_audit(self_source)
    initial_inputs, raw_payload = fixed_inputs()
    v5, cw10, v4, base, module_evidence = load_modules()
    v5_source, _ = read_regular_bytes(V5, V5_SHA256, "fulltrain v5", expected_mode=FROZEN_MODE)
    v5_static = v5.static_audit(v5_source, cw10, v4)
    output_absence = require_outputs_absent()

    if args.mode == "static":
        if args.device != "cpu" or args.expected_self_sha256 is not None or args.confirm_output_root is not None:
            raise RuntimeError("static mode accepts only default CPU arguments")
        print(
            canonical_json(
                {
                    "schema_version": SCHEMA,
                    "status": "static_zero_write_cpu_only_pass",
                    "self": self_evidence,
                    "fixed_inputs": initial_inputs,
                    "frozen_modules": module_evidence,
                    "materializer_static_audit": static,
                    "v5_static_audit": v5_static,
                    "output_absence": output_absence,
                    "renameat2_noreplace_symbol_available": renameat2_available(),
                    "cuda_accessed": False,
                    "writes_performed": False,
                    "materialization_executed": False,
                    "submission_performed": False,
                }
            ).decode("utf-8"),
            end="",
        )
        return

    if args.device != "cuda":
        raise RuntimeError("materialize mode requires --device cuda")
    if args.expected_self_sha256 is None:
        raise RuntimeError("materialize mode requires --expected-self-sha256")
    require_lower_hex64(args.expected_self_sha256, "expected self SHA")
    if self_evidence["sha256"] != args.expected_self_sha256:
        raise RuntimeError("materializer self SHA does not match external lock")
    expected_output = str(OUTPUT_ROOT.relative_to(ROOT))
    if args.confirm_output_root != expected_output:
        raise RuntimeError(f"confirm exact output root with --confirm-output-root {expected_output}")
    result = materialize(
        self_evidence,
        initial_inputs,
        raw_payload,
        v5,
        cw10,
        v4,
        base,
        module_evidence,
    )
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
