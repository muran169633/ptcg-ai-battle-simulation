#!/usr/bin/env python3
"""One-step direct512 equal-blend actor6 SGD shadow from raw full U468.

The direct512 cache builder is imported only after authenticating the frozen
aggregate runner.  The sole candidate uses one native CUDA BF16 forward and
four complete ordered composite scalars from that same output graph::

    L = 0.5 * L_union
        + (L_flg_hard + L_pokemonfan_hard + L_core5_hard) / 6

The combined scalar has exactly one backward, clip, and plain-SGD step.  Static
and cache modes are zero-write.  Contract/actual modes fail closed until a
frozen v2 probe result authorizes ``DIRECTION_FOUND`` and the equal-blend
direction.  Actual is CUDA-only, exact-preregistered, O_EXCL-locked, RAM-only,
and has no validation/checkpoint/model-artifact path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
AGGREGATE_RUNNER = TOOLS / "run_u468_raw_actor6_aggregate512_shadow.py"
AGGREGATE_RUNNER_SHA256 = (
    "db7d6ca06b83d9d035e42476a4282c0ab25b8d532a589f3be80bb2299748b641"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_link_regular(
    path: Path, expected_sha256: str | None, label: str
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a single-link regular file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


# Authentication deliberately precedes execution of the imported runner.
AGGREGATE_RUNNER_EVIDENCE = require_single_link_regular(
    AGGREGATE_RUNNER,
    AGGREGATE_RUNNER_SHA256,
    "frozen aggregate512 runner",
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_AGGREGATE_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_u468_direct512_db7d6ca0", AGGREGATE_RUNNER
)
if _AGGREGATE_SPEC is None or _AGGREGATE_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen aggregate512 import spec")
aggregate: ModuleType = importlib.util.module_from_spec(_AGGREGATE_SPEC)
_AGGREGATE_SPEC.loader.exec_module(aggregate)

torch = aggregate.torch
ppo = aggregate.ppo
repair = aggregate.repair
frozen = aggregate.frozen


SCHEMA = "ptcg-u468-raw-actor6-equalblend-sgd512-shadow-v1"
PREREGISTRATION_SCHEMA = (
    "ptcg-u468-raw-actor6-equalblend-sgd512-preregistration-v1"
)
ATTEMPT_SCHEMA = "ptcg-u468-raw-actor6-equalblend-sgd512-attempt-v1"
BRANCH = "ppo_u468_raw_actor6_equalblend_sgd512_shadow_design202608115"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"

EXECUTION_SEED = 202608115
ROWS = 512
SOURCES = ("flg", "pokemonfan", "core5")
ACTOR_NAMES = tuple(aggregate.ACTOR_NAMES)
ORDER_CONTEXT_WEIGHT = 8.0
LEARNING_RATE = 5e-5
MAX_GRAD_NORM = 0.5
SGD_MOMENTUM = 0.0
SGD_DAMPENING = 0.0
SGD_WEIGHT_DECAY = 0.0
EQUAL_BLEND_ALPHA = 0.5
EXPECTED_DIRECT_CACHE_SHA256 = (
    "ef5ab1b8f7e7162316e8ae80a6621f902e9a8e8cf73bb54daaf99e1355d0114a"
)
EXPECTED_DIRECT_BATCH_SHA256 = (
    "1c0bd23912b85dcbc64318d42ae41ba1fd7217f80e9a741e8970d91adeb1db05"
)
EXPECTED_FLAT_IDENTITY_SHA256 = (
    "d0c148e1c992030a9c5442b7a407b50816a30fdec27488f4f79b26b9e5916ce8"
)
CRITICAL_LINE_SHA256 = (
    "51eb642eb4c5a82cad4ac22c82bca296841ee8e348a19028aa847b9b658b4305"
)
EXPECTED_HARD_ROWS = {"flg": 96, "pokemonfan": 64, "core5": 64}
EXPECTED_RETENTION_ROWS = {"flg": 32, "pokemonfan": 128, "core5": 128}

# Frozen zero-write v2 probe authorization and its two-step code provenance.
V2_PROBE_RESULT: Path | None = (
    ROOT / "artifacts/u468_raw_direct512_actor6_multiobjective_probe_v2_20260802.json"
)
V2_PROBE_RESULT_SHA256: str | None = (
    "be822bb253f7a1e7ec2dbe2f01e0a0aacda78cc1902f239a59e5b74d057a61af"
)
V2_PROBE_EXPECTED_SCHEMA: str | None = (
    "ptcg-u468-raw-direct512-actor6-multiobjective-probe-v2"
)
V2_PROBE_TOOL = TOOLS / "probe_u468_raw_direct512_actor6_multiobjective_v2.py"
V2_PROBE_TOOL_SHA256 = (
    "4385a2b78ff125d0dee7c0eff2bbf4a61638f407e1730dbe979646b9a3e20828"
)
V1_PROBE_TOOL = TOOLS / "probe_u468_raw_direct512_actor6_multiobjective.py"
V1_PROBE_TOOL_SHA256 = (
    "92e6c599da44ac42a2a497280b0db86e78fc5cd91a40423c9447b4f29c177119"
)
V2_PROBE_EXPECTED_SELECTED_DIRECTION = (
    "native_bf16_equal_blend_direct_composite_plain_sgd"
)

GATE_CONTRACT = {
    "union_mixed_ordered_loss_improvement_at_least": 1e-5,
    "each_source_hard_ordered_loss_improvement_at_least": 1e-5,
    "each_source_all_ordered_loss_improvement_at_least": 1e-8,
    "aggregate_retention_ordered_loss_improvement_at_least": 1e-6,
    "each_source_retention_ordered_loss_improvement_at_least": 1e-8,
    "endpoint_retention_ordered_correct": 288,
    "endpoint_context34_ordered_correct": 6,
    "retention_correct_to_wrong_flips": 0,
    "critical_line_sha256": CRITICAL_LINE_SHA256,
    "critical_line_raw_and_endpoint_ordered_correct": True,
    "gradient_scope_exact_actor6_and_all_six_nonzero": True,
    "changed_scope_exact_actor6": True,
    "frozen_tensors_bit_exact": True,
    "displacement_l2_closed_interval": [1.5e-5, 2.5e-5],
    "displacement_max_abs_at_most": 3e-6,
    "native_bf16_gradient_forward_calls": 1,
    "complete_ordered_composite_scalar_calls": 4,
    "combined_scalar_backward_calls": 1,
    "gradient_clip_calls": 1,
    "optimizer_step_calls": 1,
    "plain_sgd_state_entries_after_step": 0,
    "single_candidate_only": True,
    "decision": "GO iff every gate passes, otherwise NO_GO",
}


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def normalize_repo_path(raw_path: str | Path, label: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay inside repository") from error
    return path


def require_absent_target(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{label} already exists: {path}")
    if not path.parent.is_dir() or path.parent.resolve() != path.parent:
        raise RuntimeError(f"{label} parent must be an existing real directory")


def publish_o_excl(path: Path, payload: bytes, mode: int = 0o600) -> dict[str, Any]:
    path = normalize_repo_path(path, "evidence output")
    require_absent_target(path, "evidence output")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(fd, view[offset:])
            if count <= 0:
                raise RuntimeError("short O_EXCL evidence write")
            offset += count
        os.fsync(fd)
        observed = os.fstat(fd)
        visible = os.lstat(path)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_ISLNK(visible.st_mode)
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (visible.st_dev, visible.st_ino, visible.st_size)
            or observed.st_size != len(payload)
        ):
            raise RuntimeError("unsafe O_EXCL evidence publication")
        os.lseek(fd, 0, os.SEEK_SET)
        reloaded = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            reloaded += chunk
        if reloaded != payload:
            raise RuntimeError("O_EXCL evidence payload drift")
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(reloaded),
            "bytes": len(reloaded),
            "inode": observed.st_ino,
            "device": observed.st_dev,
            "nlink": observed.st_nlink,
        }
    finally:
        os.close(fd)


def self_evidence() -> dict[str, Any]:
    return require_single_link_regular(
        normalize_repo_path(__file__, "equalblend runner"),
        None,
        "equalblend runner",
    )


def verify_fixed_inputs() -> dict[str, Any]:
    aggregate_evidence = require_single_link_regular(
        AGGREGATE_RUNNER,
        AGGREGATE_RUNNER_SHA256,
        "frozen aggregate512 runner",
    )
    if aggregate_evidence["sha256"] != AGGREGATE_RUNNER_EVIDENCE["sha256"]:
        raise RuntimeError("aggregate runner changed after hash-bound import")
    return {
        "aggregate_runner": aggregate_evidence,
        "aggregate_fixed_inputs": aggregate.verify_fixed_inputs(),
        "v2_probe_tool": require_single_link_regular(
            V2_PROBE_TOOL, V2_PROBE_TOOL_SHA256, "frozen v2 probe tool"
        ),
        "v1_probe_tool": require_single_link_regular(
            V1_PROBE_TOOL, V1_PROBE_TOOL_SHA256, "frozen v1 probe tool"
        ),
        "v2_probe_result": require_single_link_regular(
            V2_PROBE_RESULT,
            V2_PROBE_RESULT_SHA256,
            "frozen v2 probe result",
        ),
    }


def flat_identities(
    selections: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    fields = (
        "source",
        "member",
        "line_index",
        "line_sha256",
        "episode_id",
        "team_name",
        "category",
        "context",
        "min_count",
        "max_count",
        "expert_order",
    )
    flat = [{key: row[key] for key in fields} for batch in selections for row in batch]
    if len(flat) != ROWS:
        raise RuntimeError("direct512 identity count drift")
    digest = sha256_bytes(canonical_json(flat))
    if digest != EXPECTED_FLAT_IDENTITY_SHA256:
        raise RuntimeError(f"direct512 flat-identity SHA drift: {digest}")
    if len({(row["source"], row["member"], row["line_index"]) for row in flat}) != ROWS:
        raise RuntimeError("direct512 identities are not unique")
    return flat


def resolve_critical_line_from_archive(
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = [
        row for row in identities if str(row["line_sha256"]) == CRITICAL_LINE_SHA256
    ]
    if len(matches) != 1:
        raise RuntimeError(f"critical full SHA match-count drift: {len(matches)}")
    identity = matches[0]
    if (
        identity["source"] != "pokemonfan"
        or identity["category"] != "fragile"
        or identity["member"] != "train/part-00000.jsonl"
        or int(identity["line_index"]) != 243
        or int(identity["context"]) != 0
        or [int(value) for value in identity["expert_order"]] != [6]
    ):
        raise RuntimeError(f"critical-line selected identity drift: {identity}")
    archive_path = frozen.DATASETS["pokemonfan"]
    member = str(identity["member"])
    with zipfile.ZipFile(archive_path) as archive:
        archive.getinfo(member)
        with archive.open(member) as handle:
            raw = next(
                payload
                for index, payload in enumerate(handle)
                if index == int(identity["line_index"])
            )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CRITICAL_LINE_SHA256:
        raise RuntimeError(f"critical raw-line SHA drift: {digest}")
    record = aggregate.orjson.loads(raw)
    if (
        str(record.get("split")) != "train"
        or str(record.get("episode_id")) != str(identity["episode_id"])
        or [int(value) for value in record.get("action", [])] != [6]
    ):
        raise RuntimeError("critical raw train record metadata drift")
    return {
        "source": "pokemonfan",
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": frozen.DATA_SHA256["pokemonfan"],
        "member": member,
        "line_index": int(identity["line_index"]),
        "line_sha256": digest,
        "episode_id": str(identity["episode_id"]),
        "context": int(identity["context"]),
        "expert_order": [6],
        "category": "fragile",
        "resolved_from_raw_train_archive": True,
        "validation_member_payloads_opened": False,
    }


def build_masks(
    union: Mapping[str, torch.Tensor],
    identities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if int(union["action_counts"].shape[0]) != ROWS or len(identities) != ROWS:
        raise RuntimeError("direct512 tensor/identity alignment drift")
    masks = {"union": torch.ones(ROWS, dtype=torch.bool)}
    for source in SOURCES:
        masks[f"{source}_hard"] = torch.tensor(
            [row["source"] == source and row["category"] == "hard" for row in identities],
            dtype=torch.bool,
        )
        masks[f"{source}_retention"] = torch.tensor(
            [
                row["source"] == source and row["category"] in {"fragile", "c34"}
                for row in identities
            ],
            dtype=torch.bool,
        )
    hard_rows = {
        source: int(masks[f"{source}_hard"].sum()) for source in SOURCES
    }
    retention_rows = {
        source: int(masks[f"{source}_retention"].sum()) for source in SOURCES
    }
    if hard_rows != EXPECTED_HARD_ROWS or retention_rows != EXPECTED_RETENTION_ROWS:
        raise RuntimeError(
            f"objective partition row drift: hard={hard_rows} retention={retention_rows}"
        )
    cover = sum(
        masks[f"{source}_hard"].to(torch.int64)
        + masks[f"{source}_retention"].to(torch.int64)
        for source in SOURCES
    )
    if not torch.equal(cover, torch.ones(ROWS, dtype=torch.int64)):
        raise RuntimeError("six source/category partitions are not disjoint/exhaustive")
    return masks, {
        "hard_rows": hard_rows,
        "retention_rows": retention_rows,
        "aggregate_retention_rows": sum(retention_rows.values()),
        "six_partitions_disjoint_and_exhaustive": True,
        "objective_masks_aligned_to_flat_identity_order": True,
    }


def build_cache_and_objective_audit() -> tuple[
    dict[str, Any],
    list[list[dict[str, Any]]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    cache_audit, selections, union = aggregate.build_cache_audit()
    if (
        cache_audit["aggregate512"]["cache_sha256"]
        != EXPECTED_DIRECT_CACHE_SHA256
        or cache_audit["aggregate512"]["batch_sha256"]
        != EXPECTED_DIRECT_BATCH_SHA256
        or cache_audit["aggregate512"]["rows"] != ROWS
        or not cache_audit["padded_concat_tensor_exact_to_direct_single512"]
    ):
        raise RuntimeError("frozen direct512 cache binding drift")
    identities = flat_identities(selections)
    critical = resolve_critical_line_from_archive(identities)
    masks, partition_audit = build_masks(union, identities)
    result = {
        "status": "zero_write_equalblend_cache_audit_passed",
        "aggregate_cache": cache_audit,
        "direct512": {
            "rows": ROWS,
            "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            "single_batch": True,
        },
        "partition_audit": partition_audit,
        "critical_line": critical,
        "objective": {
            "scalar_formula": (
                "0.5*L_union + "
                "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
            ),
            "base_complete_ordered_composite_scalars": [
                "union",
                "flg_hard",
                "pokemonfan_hard",
                "core5_hard",
            ],
            "same_native_bf16_forward": True,
            "combined_scalar_backward_calls": 1,
        },
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }
    return result, selections, union, masks


def probe_binding_status() -> dict[str, Any]:
    resolved = (
        V2_PROBE_RESULT is not None
        and V2_PROBE_RESULT_SHA256 is not None
        and V2_PROBE_EXPECTED_SCHEMA is not None
    )
    return {
        "status": "resolved" if resolved else "pending_v2_probe_freeze",
        "resolved": resolved,
        "path": None if V2_PROBE_RESULT is None else str(V2_PROBE_RESULT),
        "sha256": V2_PROBE_RESULT_SHA256,
        "expected_schema": V2_PROBE_EXPECTED_SCHEMA,
        "required_decision": "DIRECTION_FOUND",
        "required_selected_direction": V2_PROBE_EXPECTED_SELECTED_DIRECTION,
        "contract_and_actual_enabled": resolved,
    }


def load_json_no_duplicates(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise RuntimeError(f"{label} contains nonfinite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def require_probe_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        V2_PROBE_RESULT is None
        or V2_PROBE_RESULT_SHA256 is None
        or V2_PROBE_EXPECTED_SCHEMA is None
    ):
        raise RuntimeError(
            "v2 probe binding is unresolved; contract/actual remain disabled"
        )
    path = normalize_repo_path(V2_PROBE_RESULT, "v2 probe result")
    evidence = require_single_link_regular(
        path, V2_PROBE_RESULT_SHA256, "v2 direct512 probe result"
    )
    payload = load_json_no_duplicates(path, "v2 direct512 probe result")
    probe = payload.get("probe")
    if not isinstance(probe, dict):
        raise RuntimeError("v2 probe result lacks nested probe payload")
    selection = probe.get("selection")
    direct512 = probe.get("direct512")
    integrity = probe.get("integrity")
    structural = probe.get("float64_logit_structural_reconstruction")
    candidate_contract = probe.get("candidate_contract")
    objective_contract = probe.get("objective_contract")
    directions = probe.get("directions")
    gate_model = probe.get("v2_gate_model")
    tool_binding = require_single_link_regular(
        V2_PROBE_TOOL, V2_PROBE_TOOL_SHA256, "v2 probe tool"
    )
    v1_tool_binding = require_single_link_regular(
        V1_PROBE_TOOL, V1_PROBE_TOOL_SHA256, "v1 probe tool"
    )
    top_tool = payload.get("tool")
    top_v1_tool = payload.get("frozen_v1_probe")
    selected_report = (
        directions.get(V2_PROBE_EXPECTED_SELECTED_DIRECTION)
        if isinstance(directions, dict)
        else None
    )
    if (
        payload.get("schema_version") != V2_PROBE_EXPECTED_SCHEMA
        or payload.get("status")
        != "completed_zero_weight_write_direct512_multiobjective_probe_v2"
        or payload.get("mode") != "probe"
        or payload.get("writes_performed") is not False
        or not isinstance(top_tool, dict)
        or top_tool.get("path") != str(V2_PROBE_TOOL.relative_to(ROOT))
        or top_tool.get("sha256") != V2_PROBE_TOOL_SHA256
        or not isinstance(top_v1_tool, dict)
        or top_v1_tool.get("path") != str(V1_PROBE_TOOL.relative_to(ROOT))
        or top_v1_tool.get("sha256") != V1_PROBE_TOOL_SHA256
        or payload.get("aggregate_runner", {}).get("sha256")
        != AGGREGATE_RUNNER_SHA256
        or probe.get("status")
        != "completed_zero_weight_write_direct512_multiobjective_probe_v2"
        or probe.get("decision") != "DIRECTION_FOUND"
        or probe.get("device") != "cuda"
        or not isinstance(selection, dict)
        or selection.get("selected_direction")
        != V2_PROBE_EXPECTED_SELECTED_DIRECTION
        or selection.get("equal_blend_passes") is not True
        or not isinstance(direct512, dict)
        or direct512.get("rows") != ROWS
        or direct512.get("cache_sha256") != EXPECTED_DIRECT_CACHE_SHA256
        or direct512.get("batch_sha256") != EXPECTED_DIRECT_BATCH_SHA256
        or direct512.get("flat_identity_sha256")
        != EXPECTED_FLAT_IDENTITY_SHA256
        or direct512.get("single_forward_batch") is not True
        or not isinstance(structural, dict)
        or structural.get("pass") is not True
        or structural.get("hard_gate") is not True
        or structural.get("kind")
        != "detached_native_policy_logits_float64_ordered_plackett_luce"
        or structural.get("pure_float64_loss_implementation") is not True
        or structural.get("bc_expert_actor_loss_not_used") is not True
        or structural.get("rows") != ROWS
        or structural.get("six_partitions_disjoint_and_exhaustive") is not True
        or structural.get("tolerance") != 1e-12
        or not isinstance(candidate_contract, dict)
        or candidate_contract.get("selectable_candidates")
        != [V2_PROBE_EXPECTED_SELECTED_DIRECTION]
        or candidate_contract.get("maximum_authorized_endpoints") != 1
        or candidate_contract.get("automatic_fallback") is not False
        or candidate_contract.get("equal_blend_alpha") != EQUAL_BLEND_ALPHA
        or candidate_contract.get("equal_blend_single_scalar_vjp") is not True
        or candidate_contract.get(
            "equal_blend_not_reconstructed_from_extracted_gradients"
        )
        is not True
        or candidate_contract.get("sole_optimizer_suggestion")
        != {
            "name": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": SGD_MOMENTUM,
            "weight_decay": SGD_WEIGHT_DECAY,
        }
        or not isinstance(objective_contract, dict)
        or objective_contract.get("loss")
        != "complete ordered bc_expert_actor_loss composite"
        or objective_contract.get("equal_blend_formula")
        != (
            "0.5*L_union + "
            "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
        )
        or objective_contract.get("equal_blend_single_scalar_vjp") is not True
        or objective_contract.get(
            "no_native_bf16_objective_gradient_algebraic_substitution"
        )
        is not True
        or objective_contract.get("order_context_weight")
        != ORDER_CONTEXT_WEIGHT
        or objective_contract.get(
            "non_context34_fixed_multi_action_order_weight"
        )
        != 1.0
        or not isinstance(selected_report, dict)
        or selected_report.get("all_gate_objectives_robust_descent") is not True
        or selected_report.get("direction", {}).get("finite") is not True
        or selected_report.get("direction", {}).get("all_six_tensors_nonzero")
        is not True
        or selected_report.get("frozen_plain_sgd_suggestion")
        != {
            "convention": "theta_new = theta - lr * raw_direction",
            "learning_rate": LEARNING_RATE,
            "momentum": SGD_MOMENTUM,
            "predicted_displacement_l2": (
                selected_report.get("frozen_plain_sgd_suggestion", {}).get(
                    "predicted_displacement_l2"
                )
            ),
            "weight_decay": SGD_WEIGHT_DECAY,
        }
        or not math.isclose(
            float(
                selected_report.get("frozen_plain_sgd_suggestion", {}).get(
                    "predicted_displacement_l2", math.nan
                )
            ),
            2.156803242859215e-05,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not isinstance(gate_model, dict)
        or gate_model.get("hard_gate")
        != "float64_logit_structural_reconstruction"
        or gate_model.get("hard_gate_passed_before_v1_candidate_geometry")
        is not True
        or gate_model.get("candidate_formula_changed_from_v1") is not False
        or gate_model.get("candidate_priority_changed_from_v1") is not False
        or gate_model.get("sgd_suggestion_changed_from_v1") is not False
        or not isinstance(integrity, dict)
        or integrity.get("aggregate_runner_sha256") != AGGREGATE_RUNNER_SHA256
        or integrity.get("v1_probe_tool_sha256") != V1_PROBE_TOOL_SHA256
        or integrity.get("actor6_scope_exact") is not True
        or integrity.get("model_weights_unchanged") is not True
        or integrity.get("model_state_sha256_before")
        != frozen.BASE_MODEL_SHA256
        or integrity.get("model_state_sha256_after")
        != frozen.BASE_MODEL_SHA256
        or integrity.get("v2_structural_model_weights_unchanged") is not True
        or integrity.get("v2_structural_model_state_sha256_before")
        != frozen.BASE_MODEL_SHA256
        or integrity.get("v2_structural_model_state_sha256_after")
        != frozen.BASE_MODEL_SHA256
        or integrity.get("v2_parameter_updates") != 0
        or integrity.get("optimizer_instances_created") != 0
        or integrity.get("optimizer_step_calls") != 0
        or integrity.get("backward_calls") != 0
        or integrity.get("parameter_grad_buffers_materialized") != 0
        or integrity.get("validation_member_payloads_opened") is not False
        or integrity.get("checkpoint_writes") != 0
        or integrity.get("model_writes") != 0
        or integrity.get("result_artifact_writes") != 0
        or integrity.get("v2_result_artifact_writes") != 0
    ):
        raise RuntimeError("v2 probe did not exactly authorize equal-blend SGD")
    authorization = {
        "schema_version": payload["schema_version"],
        "probe_status": probe["status"],
        "decision": probe["decision"],
        "selected_direction": selection["selected_direction"],
        "equal_blend_passes": selection["equal_blend_passes"],
        "direct512": {
            "rows": direct512["rows"],
            "cache_sha256": direct512["cache_sha256"],
            "batch_sha256": direct512["batch_sha256"],
            "flat_identity_sha256": direct512["flat_identity_sha256"],
        },
        "v2_probe_tool": {
            "path": tool_binding["path"],
            "sha256": tool_binding["sha256"],
        },
        "v1_probe_tool": {
            "path": v1_tool_binding["path"],
            "sha256": v1_tool_binding["sha256"],
        },
        "float64_logit_structural_reconstruction": {
            "kind": structural["kind"],
            "hard_gate": structural["hard_gate"],
            "pass": structural["pass"],
            "tolerance": structural["tolerance"],
        },
        "sole_optimizer_suggestion": dict(
            candidate_contract["sole_optimizer_suggestion"]
        ),
        "maximum_authorized_endpoints": 1,
        "aggregate_runner_sha256": integrity["aggregate_runner_sha256"],
        "probe_wrote_no_model_or_optimizer_update": True,
    }
    return evidence, authorization


def ast_audit() -> dict[str, Any]:
    source_path = normalize_repo_path(__file__, "equalblend runner")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    counts = {
        "ppo.model_forward": calls.count("ppo.model_forward"),
        "complete_ordered_scalar": calls.count("complete_ordered_scalar"),
        "combined_loss.backward": calls.count("combined_loss.backward"),
        "torch.nn.utils.clip_grad_norm_": calls.count("torch.nn.utils.clip_grad_norm_"),
        "optimizer.step": calls.count("optimizer.step"),
        "torch.optim.SGD": calls.count("torch.optim.SGD"),
        "torch.optim.AdamW": calls.count("torch.optim.AdamW"),
        "torch.autograd.grad": calls.count("torch.autograd.grad"),
        "torch.save": calls.count("torch.save"),
    }
    expected = {
        "ppo.model_forward": 1,
        "complete_ordered_scalar": 4,
        "combined_loss.backward": 1,
        "torch.nn.utils.clip_grad_norm_": 1,
        "optimizer.step": 1,
        "torch.optim.SGD": 1,
        "torch.optim.AdamW": 0,
        "torch.autograd.grad": 0,
        "torch.save": 0,
    }
    if counts != expected:
        raise RuntimeError(f"equalblend source training-call drift: {counts}")
    if "AGGREGATE_RUNNER_EVIDENCE = require_single_link_regular" not in source:
        raise RuntimeError("aggregate runner is not visibly authenticated before import")
    return {
        "status": "zero_write_static_ast_audit_passed",
        "observed_training_calls": counts,
        "expected_training_calls": expected,
        "one_native_forward_four_scalars_one_backward_clip_sgd_step": True,
        "no_gradient_reconstruction_or_probe_autograd_reuse": True,
        "checkpoint_serialization_calls": 0,
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }


def cuda_contract() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for equalblend contract/actual")
    index = torch.cuda.current_device()
    return {
        "device": "cuda",
        "cuda_device_index": index,
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "float32_matmul_precision": "high",
        "native_forward_policy_logits_dtype": "torch.bfloat16",
    }


def expected_contract(
    runner: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    probe_evidence: Mapping[str, Any],
    probe_authorization: Mapping[str, Any],
    result_output: Path,
) -> dict[str, Any]:
    return {
        "runner": {"path": runner["path"], "sha256": runner["sha256"]},
        "aggregate_runner": {
            "path": AGGREGATE_RUNNER_EVIDENCE["path"],
            "sha256": AGGREGATE_RUNNER_SHA256,
            "hash_verified_before_import": True,
        },
        "branch": BRANCH,
        "runtime": cuda_contract(),
        "execution_seed": EXECUTION_SEED,
        "training_base": {
            "kind": "raw_full_u468",
            "path": str(frozen.U468.relative_to(ROOT)),
            "checkpoint_sha256": frozen.U468_SHA256,
            "runtime_model_state_sha256": frozen.BASE_MODEL_SHA256,
            "update": 468,
        },
        "v2_probe": {
            "result": {
                "path": probe_evidence["path"],
                "sha256": probe_evidence["sha256"],
            },
            "tool": dict(probe_authorization["v2_probe_tool"]),
            "frozen_v1_tool": dict(probe_authorization["v1_probe_tool"]),
            "authorization": dict(probe_authorization),
        },
        "direct512": {
            "cache_sha256": EXPECTED_DIRECT_CACHE_SHA256,
            "batch_sha256": EXPECTED_DIRECT_BATCH_SHA256,
            "flat_identity_sha256": EXPECTED_FLAT_IDENTITY_SHA256,
            "rows": ROWS,
            "single_batch": True,
            "padded_concat_tensor_exact_to_direct_single512": cache_audit[
                "aggregate_cache"
            ]["padded_concat_tensor_exact_to_direct_single512"],
        },
        "critical_line": cache_audit["critical_line"],
        "objective": {
            "native_bf16_gradient_forward_calls": 1,
            "complete_ordered_composite_scalar_calls": 4,
            "loss_mode": "ordered",
            "order_context_weight": ORDER_CONTEXT_WEIGHT,
            "non_context34_fixed_multi_action_order_weight": 1.0,
            "formula": (
                "0.5*L_union + "
                "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
            ),
            "base_scalars": [
                "union",
                "flg_hard",
                "pokemonfan_hard",
                "core5_hard",
            ],
            "combined_scalar_backward_calls": 1,
            "no_extracted_gradient_reconstruction": True,
            "no_probe_gradient_reuse": True,
        },
        "optimizer": {
            "name": "SGD",
            "plain": True,
            "fresh": True,
            "lr": LEARNING_RATE,
            "momentum": SGD_MOMENTUM,
            "dampening": SGD_DAMPENING,
            "weight_decay": SGD_WEIGHT_DECAY,
            "nesterov": False,
            "maximize": False,
            "foreach": False,
            "differentiable": False,
            "fused": False,
            "max_grad_norm": MAX_GRAD_NORM,
            "state_entries_before_step": 0,
            "state_entries_after_step": 0,
        },
        "trajectory": {
            "candidate_count": 1,
            "actor_parameter_names": list(ACTOR_NAMES),
            "gradient_forward_calls": 1,
            "backward_calls": 1,
            "clip_calls": 1,
            "optimizer_step_calls": 1,
            "evaluation_forward_calls": 2,
            "model_mutation": "RAM_only",
        },
        "gates": GATE_CONTRACT,
        "decision_rule": "sole candidate GO iff all gates pass; otherwise NO_GO",
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "result_output": str(result_output.relative_to(ROOT)),
        "scope": {
            "train_only": True,
            "validation": False,
            "checkpoint_write": False,
            "model_artifact_write": False,
            "optimizer_artifact_write": False,
            "submission": False,
            "evidence_writes": ["O_EXCL attempt marker", "O_EXCL result JSON"],
        },
    }


def load_preregistration(
    path: Path, expected_sha256: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError("expected preregistration SHA must be 64 lowercase hex")
    evidence = require_single_link_regular(
        path, expected_sha256, "equalblend preregistration"
    )
    payload = load_json_no_duplicates(path, "equalblend preregistration")
    if payload != {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "locked_before_actual",
        "shadow_contract": contract,
    }:
        raise RuntimeError("equalblend preregistration exact-contract mismatch")
    return evidence


def masked_batch(
    batch: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    if mask.shape != batch["sample_weights"].shape:
        raise RuntimeError("objective mask shape drift")
    result = dict(batch)
    result["sample_weights"] = batch["sample_weights"] * mask.to(
        device=batch["sample_weights"].device,
        dtype=batch["sample_weights"].dtype,
    )
    return result


def complete_ordered_scalar(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scalar, parts = ppo.bc_expert_actor_loss(
        dict(outputs),
        dict(batch),
        loss_mode="ordered",
        order_context_weight=ORDER_CONTEXT_WEIGHT,
        non_context34_fixed_multi_action_order_weight=1.0,
    )
    if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
        raise FloatingPointError("complete ordered composite scalar is nonfinite")
    return scalar, parts


def execute_equalblend_step(
    model: torch.nn.Module,
    union_cpu: Mapping[str, torch.Tensor],
    masks_cpu: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    parameters = frozen.configure_actor6(model)
    if tuple(parameters) != ACTOR_NAMES:
        raise RuntimeError("actor6 parameter order drift")
    if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise RuntimeError("raw U468 actor6 storage dtype is not FP32")
    optimizer = torch.optim.SGD(
        list(parameters.values()),
        lr=LEARNING_RATE,
        momentum=SGD_MOMENTUM,
        dampening=SGD_DAMPENING,
        weight_decay=SGD_WEIGHT_DECAY,
        nesterov=False,
        maximize=False,
        foreach=False,
        differentiable=False,
        fused=False,
    )
    if optimizer.state or optimizer.state_dict()["state"]:
        raise RuntimeError("fresh plain-SGD state is not empty")
    counters = {
        "native_bf16_gradient_forward_calls": 0,
        "complete_ordered_composite_scalar_calls": 0,
        "combined_scalar_backward_calls": 0,
        "gradient_clip_calls": 0,
        "optimizer_step_calls": 0,
    }
    optimizer.zero_grad(set_to_none=True)
    batch = {
        key: value.to(device, non_blocking=True) for key, value in union_cpu.items()
    }
    masks = {
        key: value.to(device, non_blocking=True) for key, value in masks_cpu.items()
    }
    if int(batch["action_counts"].shape[0]) != ROWS:
        raise RuntimeError("equalblend gradient batch is not direct512")
    counters["native_bf16_gradient_forward_calls"] += 1
    outputs = ppo.model_forward(model, batch, device)
    output_dtypes = {key: str(value.dtype) for key, value in outputs.items()}
    if outputs["policy_logits"].dtype != torch.bfloat16:
        raise RuntimeError(f"native policy logits are not BF16: {output_dtypes}")

    union_loss, union_parts = complete_ordered_scalar(
        outputs, masked_batch(batch, masks["union"])
    )
    counters["complete_ordered_composite_scalar_calls"] += 1
    flg_hard_loss, flg_hard_parts = complete_ordered_scalar(
        outputs, masked_batch(batch, masks["flg_hard"])
    )
    counters["complete_ordered_composite_scalar_calls"] += 1
    pokemonfan_hard_loss, pokemonfan_hard_parts = complete_ordered_scalar(
        outputs, masked_batch(batch, masks["pokemonfan_hard"])
    )
    counters["complete_ordered_composite_scalar_calls"] += 1
    core5_hard_loss, core5_hard_parts = complete_ordered_scalar(
        outputs, masked_batch(batch, masks["core5_hard"])
    )
    counters["complete_ordered_composite_scalar_calls"] += 1
    combined_loss = EQUAL_BLEND_ALPHA * union_loss + (
        flg_hard_loss + pokemonfan_hard_loss + core5_hard_loss
    ) / 6.0
    if combined_loss.ndim != 0 or not bool(torch.isfinite(combined_loss)):
        raise FloatingPointError("equalblend combined scalar is nonfinite")
    counters["combined_scalar_backward_calls"] += 1
    combined_loss.backward()

    gradient_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    )
    per_tensor_gradient: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"missing/nonfinite equalblend gradient: {name}")
        gradient = parameter.grad.detach()
        per_tensor_gradient[name] = {
            "l2": math.sqrt(float(gradient.double().square().sum())),
            "max_abs": float(gradient.abs().max()),
            "nonzero_elements": int(torch.count_nonzero(gradient)),
        }
    counters["gradient_clip_calls"] += 1
    preclip = torch.nn.utils.clip_grad_norm_(list(parameters.values()), MAX_GRAD_NORM)
    preclip_l2 = float(preclip.detach().cpu())
    postclip_l2 = math.sqrt(
        sum(
            float(parameter.grad.detach().double().square().sum())
            for parameter in parameters.values()
            if parameter.grad is not None
        )
    )
    counters["optimizer_step_calls"] += 1
    optimizer.step()
    if optimizer.state or optimizer.state_dict()["state"]:
        raise RuntimeError("plain SGD unexpectedly materialized optimizer state")
    expected_counters = {
        "native_bf16_gradient_forward_calls": 1,
        "complete_ordered_composite_scalar_calls": 4,
        "combined_scalar_backward_calls": 1,
        "gradient_clip_calls": 1,
        "optimizer_step_calls": 1,
    }
    if counters != expected_counters:
        raise RuntimeError(f"equalblend operation-count drift: {counters}")

    scalar_records = {
        "union": (union_loss, union_parts),
        "flg_hard": (flg_hard_loss, flg_hard_parts),
        "pokemonfan_hard": (pokemonfan_hard_loss, pokemonfan_hard_parts),
        "core5_hard": (core5_hard_loss, core5_hard_parts),
    }
    return {
        "combined_loss": float(combined_loss.detach().cpu()),
        "base_complete_ordered_composite_scalars": {
            name: {
                "loss": float(scalar.detach().cpu()),
                "loss_parts": {
                    key: float(value.detach().cpu()) for key, value in parts.items()
                },
            }
            for name, (scalar, parts) in scalar_records.items()
        },
        "formula": (
            "0.5*L_union + "
            "(L_flg_hard+L_pokemonfan_hard+L_core5_hard)/6"
        ),
        "native_output_dtypes": output_dtypes,
        "gradient_parameter_names": gradient_names,
        "gradient_scope_exact_actor6": gradient_names == sorted(ACTOR_NAMES),
        "all_six_gradient_tensors_nonzero": all(
            item["nonzero_elements"] > 0 for item in per_tensor_gradient.values()
        ),
        "per_tensor_preclip_gradient": per_tensor_gradient,
        "preclip_gradient_l2": preclip_l2,
        "postclip_gradient_l2": postclip_l2,
        "clip_max_norm": MAX_GRAD_NORM,
        "optimizer_state_count_before_step": 0,
        "optimizer_state_count_after_step": 0,
        "operation_counts": counters,
        "single_direct512_graph": True,
        "extracted_probe_gradients_reused": False,
    }


def combined_group(
    report: Mapping[str, Any], source: str, categories: Sequence[str]
) -> dict[str, Any]:
    groups = [report["by_source_and_bucket"][source][category] for category in categories]
    weight = sum(float(group["effective_weight"]) for group in groups)
    if weight <= 0.0:
        raise RuntimeError("nonpositive combined evaluation weight")
    return {
        "rows": sum(int(group["rows"]) for group in groups),
        "effective_weight": weight,
        "ordered_loss": sum(
            float(group["ordered_loss"]) * float(group["effective_weight"])
            for group in groups
        )
        / weight,
        "ordered_correct": sum(int(group["ordered_correct"]) for group in groups),
    }


def evaluation_improvements(
    raw: Mapping[str, Any], endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    source_reports: dict[str, Any] = {}
    for source in SOURCES:
        raw_all = raw["by_source_and_bucket"][source]["all"]
        endpoint_all = endpoint["by_source_and_bucket"][source]["all"]
        raw_hard = raw["by_source_and_bucket"][source]["hard"]
        endpoint_hard = endpoint["by_source_and_bucket"][source]["hard"]
        raw_retention = combined_group(raw, source, ("fragile", "c34"))
        endpoint_retention = combined_group(endpoint, source, ("fragile", "c34"))
        source_reports[source] = {
            "all": {
                "raw": float(raw_all["ordered_loss"]),
                "endpoint": float(endpoint_all["ordered_loss"]),
                "improvement": float(raw_all["ordered_loss"])
                - float(endpoint_all["ordered_loss"]),
            },
            "hard": {
                "raw": float(raw_hard["ordered_loss"]),
                "endpoint": float(endpoint_hard["ordered_loss"]),
                "improvement": float(raw_hard["ordered_loss"])
                - float(endpoint_hard["ordered_loss"]),
            },
            "retention": {
                "raw": raw_retention,
                "endpoint": endpoint_retention,
                "improvement": float(raw_retention["ordered_loss"])
                - float(endpoint_retention["ordered_loss"]),
            },
        }
    raw_retention_groups = [source_reports[source]["retention"]["raw"] for source in SOURCES]
    endpoint_retention_groups = [
        source_reports[source]["retention"]["endpoint"] for source in SOURCES
    ]

    def aggregate_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        weight = sum(float(group["effective_weight"]) for group in groups)
        return {
            "rows": sum(int(group["rows"]) for group in groups),
            "effective_weight": weight,
            "ordered_loss": sum(
                float(group["ordered_loss"]) * float(group["effective_weight"])
                for group in groups
            )
            / weight,
            "ordered_correct": sum(int(group["ordered_correct"]) for group in groups),
        }

    raw_retention = aggregate_groups(raw_retention_groups)
    endpoint_retention = aggregate_groups(endpoint_retention_groups)
    return {
        "union": {
            "raw": float(raw["mixed_ordered_loss"]),
            "endpoint": float(endpoint["mixed_ordered_loss"]),
            "improvement": float(raw["mixed_ordered_loss"])
            - float(endpoint["mixed_ordered_loss"]),
        },
        "sources": source_reports,
        "aggregate_retention": {
            "raw": raw_retention,
            "endpoint": endpoint_retention,
            "improvement": float(raw_retention["ordered_loss"])
            - float(endpoint_retention["ordered_loss"]),
        },
    }


def run_single_candidate(
    union: Mapping[str, torch.Tensor],
    selections: Sequence[Sequence[dict[str, Any]]],
    masks: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    model, _ = frozen.load_raw_u468(device)
    base_state = repair.clone_model_state(model)
    raw_evaluation, raw_correct = aggregate.evaluate_union(
        model, union, selections, device
    )
    if raw_evaluation["retention"] != {
        "rows": 288,
        "ordered_correct": 288,
        "fragile_rows": 282,
        "fragile_ordered_correct": 282,
        "context34_rows": 6,
        "context34_ordered_correct": 6,
    }:
        raise RuntimeError("raw U468 direct512 retention correctness drift")
    if raw_correct.get(CRITICAL_LINE_SHA256) is not True:
        raise RuntimeError("critical line is not raw ordered-correct")

    training = execute_equalblend_step(model, union, masks, device)
    endpoint_evaluation, endpoint_correct = aggregate.evaluate_union(
        model, union, selections, device
    )
    displacement = frozen.displacement_report(base_state, model)
    endpoint_state = repair.clone_model_state(model)
    frozen_names = sorted(set(base_state) - set(ACTOR_NAMES))
    frozen_unchanged = all(
        torch.equal(endpoint_state[name], base_state[name]) for name in frozen_names
    )
    retention_flips = frozen.correct_to_wrong_flips(
        raw_correct, endpoint_correct, selections
    )
    improvements = evaluation_improvements(raw_evaluation, endpoint_evaluation)
    displacement_l2 = float(displacement["l2"])
    displacement_max_abs = float(displacement["max_abs"])
    gate_checks = {
        "all_finite": repair.finite_nested(
            {
                "training": training,
                "raw_evaluation": raw_evaluation,
                "endpoint_evaluation": endpoint_evaluation,
                "displacement": displacement,
                "retention_flips": retention_flips,
                "improvements": improvements,
                "endpoint_state": endpoint_state,
            }
        ),
        "union_improves_at_least_1e-5": improvements["union"]["improvement"] >= 1e-5,
        "three_hard_each_improve_at_least_1e-5": all(
            improvements["sources"][source]["hard"]["improvement"] >= 1e-5
            for source in SOURCES
        ),
        "three_source_all_each_improve_at_least_1e-8": all(
            improvements["sources"][source]["all"]["improvement"] >= 1e-8
            for source in SOURCES
        ),
        "aggregate_retention_improves_at_least_1e-6": improvements[
            "aggregate_retention"
        ]["improvement"]
        >= 1e-6,
        "three_source_retention_each_improve_at_least_1e-8": all(
            improvements["sources"][source]["retention"]["improvement"] >= 1e-8
            for source in SOURCES
        ),
        "endpoint_retention_288_context34_6_zero_flips": bool(
            endpoint_evaluation["retention"]["ordered_correct"] == 288
            and endpoint_evaluation["retention"]["context34_ordered_correct"] == 6
            and retention_flips["retention_rows"] == 288
            and retention_flips["raw_retention_correct"] == 288
            and retention_flips["total_correct_to_wrong"] == 0
        ),
        "critical_line_pinned_correct": bool(
            raw_correct.get(CRITICAL_LINE_SHA256) is True
            and endpoint_correct.get(CRITICAL_LINE_SHA256) is True
        ),
        "gradient_scope_exact_actor6_all_six_nonzero": bool(
            training["gradient_scope_exact_actor6"]
            and training["all_six_gradient_tensors_nonzero"]
        ),
        "changed_scope_exact_actor6_frozen_exact": bool(
            displacement["changed_scope_exact_actor6"] and frozen_unchanged
        ),
        "displacement_l2_in_closed_interval": 1.5e-5
        <= displacement_l2
        <= 2.5e-5,
        "displacement_max_abs_at_most_3e-6": displacement_max_abs <= 3e-6,
        "one_forward_four_scalars_one_backward_clip_step": training[
            "operation_counts"
        ]
        == {
            "native_bf16_gradient_forward_calls": 1,
            "complete_ordered_composite_scalar_calls": 4,
            "combined_scalar_backward_calls": 1,
            "gradient_clip_calls": 1,
            "optimizer_step_calls": 1,
        },
        "plain_sgd_state_empty": bool(
            training["optimizer_state_count_before_step"] == 0
            and training["optimizer_state_count_after_step"] == 0
        ),
    }
    return {
        "candidate": "raw_u468_actor6_direct512_equalblend_plain_sgd_lr5e-5",
        "candidate_count": 1,
        "training": training,
        "evaluations": {"raw": raw_evaluation, "endpoint": endpoint_evaluation},
        "improvements": improvements,
        "retention_flips_from_raw": retention_flips,
        "critical_line": {
            "line_sha256": CRITICAL_LINE_SHA256,
            "raw_ordered_correct": raw_correct[CRITICAL_LINE_SHA256],
            "endpoint_ordered_correct": endpoint_correct[CRITICAL_LINE_SHA256],
        },
        "displacement_from_raw": displacement,
        "frozen_tensors_unchanged": frozen_unchanged,
        "gate_checks": gate_checks,
        "fully_passes": all(gate_checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("static-audit", "cache-audit", "shadow-contract", "actual"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--result-output")
    parser.add_argument("--preregistration")
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    runner = self_evidence()
    fixed_inputs = verify_fixed_inputs()
    static = ast_audit()
    probe_evidence, probe_authorization = require_probe_authorization()
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)
    evidence_args = (
        args.result_output,
        args.preregistration,
        args.expected_preregistration_sha256,
    )
    if args.mode == "static-audit":
        if args.device != "cpu" or any(value is not None for value in evidence_args):
            raise RuntimeError("static-audit is CPU/default and accepts no evidence args")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "zero_write_static_audit_passed",
                    "runner": runner,
                    "fixed_inputs": fixed_inputs,
                    "ast_audit": static,
                    "probe_binding": probe_binding_status(),
                    "probe_evidence": probe_evidence,
                    "probe_authorization": probe_authorization,
                    "contract_and_actual_executed": False,
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    cache_audit, selections, union, masks = build_cache_and_objective_audit()
    if args.mode == "cache-audit":
        if args.device != "cpu" or any(value is not None for value in evidence_args):
            raise RuntimeError("cache-audit is CPU/default and accepts no evidence args")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "zero_write_cache_audit_passed",
                    "runner": runner,
                    "fixed_inputs": fixed_inputs,
                    "ast_audit": static,
                    "cache": cache_audit,
                    "probe_binding": probe_binding_status(),
                    "probe_evidence": probe_evidence,
                    "probe_authorization": probe_authorization,
                    "contract_and_actual_executed": False,
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    if args.mode == "shadow-contract":
        if (
            args.device != "cuda"
            or args.result_output is None
            or args.preregistration is not None
            or args.expected_preregistration_sha256 is not None
        ):
            raise RuntimeError(
                "shadow-contract requires --device cuda and --result-output only"
            )
        result_output = normalize_repo_path(args.result_output, "shadow result")
        if result_output == ATTEMPT_MARKER:
            raise RuntimeError("result path collides with attempt marker")
        require_absent_target(result_output, "shadow result")
        contract = expected_contract(
            runner, cache_audit, probe_evidence, probe_authorization, result_output
        )
        print(
            json.dumps(
                {
                    "schema_version": PREREGISTRATION_SCHEMA,
                    "status": "locked_before_actual",
                    "shadow_contract": contract,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return


    if (
        args.device != "cuda"
        or args.result_output is None
        or args.preregistration is None
        or args.expected_preregistration_sha256 is None
    ):
        raise RuntimeError(
            "actual requires CUDA, result, preregistration, and exact prereg SHA"
        )
    result_output = normalize_repo_path(args.result_output, "shadow result")
    preregistration = normalize_repo_path(args.preregistration, "preregistration")
    if result_output in {ATTEMPT_MARKER, preregistration}:
        raise RuntimeError("actual control-plane path collision")
    require_absent_target(result_output, "shadow result")
    contract = expected_contract(
        runner, cache_audit, probe_evidence, probe_authorization, result_output
    )
    preregistration_evidence = load_preregistration(
        preregistration, args.expected_preregistration_sha256, contract
    )

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    # Recheck all mutable code/data/probe/prereg bindings immediately before
    # the one-shot lock.  No model or optimizer exists before the marker.
    fixed_inputs_at_lock = verify_fixed_inputs()
    probe_evidence_at_lock, probe_authorization_at_lock = require_probe_authorization()
    preregistration_at_lock = load_preregistration(
        preregistration, args.expected_preregistration_sha256, contract
    )
    if (
        fixed_inputs_at_lock != fixed_inputs
        or probe_evidence_at_lock != probe_evidence
        or probe_authorization_at_lock != probe_authorization
        or preregistration_at_lock != preregistration_evidence
        or self_evidence()["sha256"] != runner["sha256"]
    ):
        raise RuntimeError("formal identity changed immediately before lock")
    require_absent_target(result_output, "shadow result")
    require_absent_target(ATTEMPT_MARKER, "attempt marker")
    marker_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "actual_attempt_consumed_before_model_or_optimizer_creation",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "runner": runner,
        "aggregate_runner": AGGREGATE_RUNNER_EVIDENCE,
        "probe": probe_evidence,
        "probe_authorization": probe_authorization,
        "preregistration": preregistration_evidence,
        "shadow_contract": contract,
        "cuda_verified_before_lock": True,
        "model_instances_created_before_lock": 0,
        "optimizer_instances_created_before_lock": 0,
        "backward_calls_before_lock": 0,
        "optimizer_step_calls_before_lock": 0,
        "checkpoint_writes_before_lock": 0,
        "validation_member_payloads_opened": False,
    }
    marker = publish_o_excl(ATTEMPT_MARKER, canonical_json(marker_payload))

    candidate = run_single_candidate(union, selections, masks, device)
    torch.cuda.synchronize(device)
    decision = "GO" if candidate["fully_passes"] else "NO_GO"
    result = {
        "schema_version": SCHEMA,
        "status": "completed_actual_ram_only_equalblend_sgd512_step",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "decision": decision,
        "single_candidate_only": True,
        "candidate": candidate,
        "runner": runner,
        "aggregate_runner": AGGREGATE_RUNNER_EVIDENCE,
        "fixed_inputs": fixed_inputs,
        "ast_audit": static,
        "cache": cache_audit,
        "probe": probe_evidence,
        "probe_authorization": probe_authorization,
        "preregistration": preregistration_evidence,
        "shadow_contract": contract,
        "attempt_marker": marker,
        "model_artifact_writes": 0,
        "checkpoint_writes": 0,
        "optimizer_artifact_writes": 0,
        "training_artifact_writes": 0,
        "evidence_writes": 2,
        "validation_member_payloads_opened": False,
        "submission_performed": False,
    }
    if not repair.finite_nested(result):
        raise FloatingPointError("equalblend result contains nonfinite values")
    result_evidence = publish_o_excl(result_output, canonical_json(result), mode=0o444)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": decision,
                "result": result_evidence,
                "attempt_marker": marker,
                "validation_member_payloads_opened": False,
                "checkpoint_writes": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
