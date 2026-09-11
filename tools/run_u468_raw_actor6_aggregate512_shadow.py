#!/usr/bin/env python3
"""Audit one aggregate-512 actor6 AdamW shadow step from raw full U468.

The frozen balanced-selection runner is SHA-bound before import.  Its two
256-row train-only caches are reused, padded along the option axis, and
concatenated.  An independently collated 512-row batch must be tensor-exact to
that padded concatenation.  The only state-changing model path is a CUDA-only,
hash-preregistered RAM shadow with exactly one 512-row backward, one gradient
clip, and one optimizer step.  It never writes a checkpoint or opens a
validation member.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_SELECTION_RUNNER = (
    TOOLS / "run_u468_raw_trainhard_actor6_balanced_mix_sweep.py"
)
FROZEN_SELECTION_RUNNER_SHA256 = (
    "419feeeb5644adf34a8d56142cfcd87b3a4b79446d95b9b566bf9fcabdb721b1"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_frozen_selection_runner_before_import() -> dict[str, Any]:
    observed = os.lstat(FROZEN_SELECTION_RUNNER)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError("frozen selection runner is not a single-link regular file")
    digest = sha256_file(FROZEN_SELECTION_RUNNER)
    if digest != FROZEN_SELECTION_RUNNER_SHA256:
        raise RuntimeError(f"frozen selection runner SHA-256 drift: {digest}")
    return {
        "path": str(FROZEN_SELECTION_RUNNER.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


FROZEN_SELECTION_EVIDENCE = require_frozen_selection_runner_before_import()
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_SELECTION_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_u468_balanced_selection_419feeeb",
    FROZEN_SELECTION_RUNNER,
)
if _SELECTION_SPEC is None or _SELECTION_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen selection-runner import spec")
frozen: ModuleType = importlib.util.module_from_spec(_SELECTION_SPEC)
_SELECTION_SPEC.loader.exec_module(frozen)

torch = frozen.torch
orjson = frozen.orjson
ppo = frozen.ppo
bc = frozen.bc
repair = frozen.repair


SCHEMA = "ptcg-u468-raw-actor6-aggregate512-shadow-v1"
PREREGISTRATION_SCHEMA = "ptcg-u468-raw-actor6-aggregate512-preregistration-v1"
ATTEMPT_SCHEMA = "ptcg-u468-raw-actor6-aggregate512-attempt-v1"
BRANCH = "ppo_u468_raw_actor6_aggregate512_shadow_design202608112"
ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"

EXECUTION_SEED = 202608113
ROWS = 512
BATCH_ROWS = 256
LEARNING_RATE = 1.125e-7
ORDER_CONTEXT_WEIGHT = 8.0
MAX_GRAD_NORM = 0.5
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-5
WEIGHT_DECAY = 1e-4
ACTOR_NAMES = tuple(frozen.ACTOR_NAMES)
EXPECTED_SOURCE_COUNTS = {"flg": 128, "pokemonfan": 192, "core5": 192}
EXPECTED_CATEGORY_COUNTS = {
    "flg": {"hard": 96, "fragile": 30, "c34": 2},
    "pokemonfan": {"hard": 64, "fragile": 126, "c34": 2},
    "core5": {"hard": 64, "fragile": 126, "c34": 2},
}
OPTION_AXIS_KEYS = {
    "option_fields",
    "option_field_mask",
    "option_numeric",
    "option_mask",
    "targets",
}
EXPECTED_FROZEN_CACHE_SHA256 = frozen.EXPECTED_CACHE_SHA256
EXPECTED_FROZEN_BATCH_SHA256 = tuple(frozen.EXPECTED_BATCH_SHA256)
# Frozen from the zero-write direct-collate/padded-concat cache audit.  The
# preregistration independently binds both values again.
EXPECTED_UNION_CACHE_SHA256: str | None = (
    "ef5ab1b8f7e7162316e8ae80a6621f902e9a8e8cf73bb54daaf99e1355d0114a"
)
EXPECTED_UNION_BATCH_SHA256: str | None = (
    "1c0bd23912b85dcbc64318d42ae41ba1fd7217f80e9a741e8970d91adeb1db05"
)

GATE_CONTRACT = {
    "all_finite": True,
    "gradient_scope_exact_actor6_and_all_six_nonzero": True,
    "changed_scope_exact_actor6_and_frozen_tensors_unchanged": True,
    "fresh_adamw_optimizer_state_count": 6,
    "optimizer_step_values_exact": [1],
    "displacement_l2_strictly_greater_than": 0.0,
    "displacement_l2_at_most": 3.10e-5,
    "union_mixed_ordered_loss_improves_by_at_least": 1e-8,
    "each_source_hard_ordered_loss_improves_by_at_least": 1e-8,
    "retention_rows": 288,
    "retention_correct_to_wrong_flips": 0,
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
        raise ValueError(f"{label} parent must be an existing real directory")


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


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires Python -I -B")


def runner_evidence() -> dict[str, Any]:
    path = normalize_repo_path(__file__, "aggregate runner")
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError("aggregate runner is not a single-link regular file")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "bytes": observed.st_size,
        "inode": observed.st_ino,
        "device": observed.st_dev,
        "nlink": observed.st_nlink,
    }


def verify_fixed_inputs() -> dict[str, Any]:
    if sha256_file(FROZEN_SELECTION_RUNNER) != FROZEN_SELECTION_RUNNER_SHA256:
        raise RuntimeError("selection runner changed after its hash-bound import")
    evidence = {
        "selection_runner": frozen.require_regular(
            FROZEN_SELECTION_RUNNER,
            FROZEN_SELECTION_RUNNER_SHA256,
            "frozen balanced selection runner",
        ),
        "u468": frozen.require_regular(
            frozen.U468, frozen.U468_SHA256, "raw U468"
        ),
        "general_bc": frozen.require_regular(
            frozen.GENERAL_BC, frozen.GENERAL_BC_SHA256, "general BC"
        ),
        "profile": frozen.require_regular(
            frozen.PROFILE, frozen.PROFILE_SHA256, "train-only profile"
        ),
        "datasets": {
            source: frozen.require_regular(
                path, frozen.DATA_SHA256[source], f"{source} train archive"
            )
            for source, path in frozen.DATASETS.items()
        },
        "dependencies": {
            str(path.relative_to(ROOT)): frozen.require_regular(path, digest, str(path))
            for path, digest in frozen.DEPENDENCY_SHA256.items()
        },
    }
    return evidence


def load_profile_and_selections() -> tuple[dict[str, Any], list[list[dict[str, Any]]], dict[str, Any]]:
    profile = json.loads(frozen.PROFILE.read_text(encoding="utf-8"))
    selections, summary = frozen.select_batches(profile)
    if len(selections) != 2 or any(len(batch) != BATCH_ROWS for batch in selections):
        raise RuntimeError("frozen selection must remain exactly B1/B2 = 256/256")
    flat = [row for batch in selections for row in batch]
    source_counts = {
        source: sum(row["source"] == source for row in flat)
        for source in frozen.DATASETS
    }
    category_counts = {
        source: {
            category: sum(
                row["source"] == source and row["category"] == category
                for row in flat
            )
            for category in ("hard", "fragile", "c34")
        }
        for source in frozen.DATASETS
    }
    if source_counts != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(f"aggregate source-count drift: {source_counts}")
    if category_counts != EXPECTED_CATEGORY_COUNTS:
        raise RuntimeError(f"aggregate category-count drift: {category_counts}")
    return profile, selections, summary


def load_direct_union_features(
    selections: Sequence[Sequence[dict[str, Any]]], model_config: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Independently collate all selected train rows as one 512-row batch."""
    flat = [record for batch in selections for record in batch]
    if len(flat) != ROWS:
        raise RuntimeError("direct union row-count drift")
    wanted: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        source: {} for source in frozen.DATASETS
    }
    for record in flat:
        source = str(record["source"])
        key = (str(record["member"]), int(record["line_index"]))
        if key in wanted[source]:
            raise RuntimeError("direct union repeats selected raw row")
        wanted[source][key] = record

    features: dict[tuple[str, str, int], dict[str, Any]] = {}
    opened_members: dict[str, list[str]] = {source: [] for source in frozen.DATASETS}
    for source, path in frozen.DATASETS.items():
        by_member: dict[str, set[int]] = defaultdict(set)
        for member, line_index in wanted[source]:
            if not member.startswith("train/") or not member.endswith(".jsonl"):
                raise RuntimeError("direct union attempted a non-train member")
            by_member[member].add(line_index)
        with zipfile.ZipFile(path) as archive:
            for member in sorted(by_member):
                archive.getinfo(member)
                remaining = set(by_member[member])
                with archive.open(member) as handle:
                    opened_members[source].append(member)
                    for line_index, raw in enumerate(handle):
                        if line_index not in remaining:
                            continue
                        record = wanted[source][(member, line_index)]
                        if hashlib.sha256(raw).hexdigest() != record["line_sha256"]:
                            raise RuntimeError("direct union raw-line SHA drift")
                        row = orjson.loads(raw)
                        if str(row.get("split", "")) != "train":
                            raise RuntimeError("direct union selected a non-train row")
                        if (
                            str(row.get("episode_id", ""))
                            != str(record["episode_id"])
                            or str(row.get("team_name", ""))
                            != str(record["team_name"])
                        ):
                            raise RuntimeError("direct union row identity drift")
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            feature = bc.featurize_row(
                                row,
                                int(model_config["hash_size"]),
                                int(model_config["max_state_entities"]),
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if feature is None:
                            raise RuntimeError("direct union selected row no longer featurizes")
                        expert = [int(value) for value in row.get("action", [])]
                        if (
                            expert != [int(value) for value in record["expert_order"]]
                            or int(feature["context"]) != int(record["context"])
                            or int(feature["min_count"]) != int(record["min_count"])
                            or int(feature["max_count"]) != int(record["max_count"])
                        ):
                            raise RuntimeError("direct union decision metadata drift")
                        feature["action_sequence"] = expert
                        if int(feature["context"]) == ppo.SKILL_ORDER_CONTEXT:
                            feature["sample_weight"] = frozen.CONTEXT34_SAMPLE_WEIGHT
                        features[(source, member, line_index)] = feature
                        remaining.remove(line_index)
                        if not remaining:
                            break
                if remaining:
                    raise RuntimeError(f"direct union selected lines missing: {member}")

    rows = [
        features[(str(row["source"]), str(row["member"]), int(row["line_index"]))]
        for row in flat
    ]
    union = bc.collate_decisions(
        rows,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    if int(union["action_counts"].shape[0]) != ROWS:
        raise RuntimeError("direct single512 collate row-count drift")
    context_mask = union["contexts"] == ppo.SKILL_ORDER_CONTEXT
    if int(context_mask.sum()) != 6:
        raise RuntimeError("direct single512 context34 physical quota drift")
    expected_weight = torch.full_like(
        union["sample_weights"][context_mask], frozen.CONTEXT34_SAMPLE_WEIGHT
    )
    if not torch.equal(union["sample_weights"][context_mask], expected_weight):
        raise RuntimeError("direct single512 context34 sample-weight drift")
    if any(
        not member.startswith("train/")
        for members in opened_members.values()
        for member in members
    ):
        raise RuntimeError("non-train member opened during direct collate")
    return union, {
        "opened_members": opened_members,
        "non_train_members_opened": False,
        "rows": ROWS,
    }


def pad_option_axis(
    tensor: torch.Tensor, target_width: int, key: str
) -> torch.Tensor:
    if key not in OPTION_AXIS_KEYS:
        raise ValueError(f"unexpected option-axis tensor: {key}")
    if tensor.ndim < 2:
        raise RuntimeError(f"option-axis tensor has rank < 2: {key}")
    width = int(tensor.shape[1])
    if width > target_width:
        raise RuntimeError(f"cannot shrink option axis for {key}")
    if width == target_width:
        return tensor
    padding_shape = list(tensor.shape)
    padding_shape[1] = target_width - width
    padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=1)


def padded_concat_batches(
    batches: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if len(batches) != 2 or any(int(batch["action_counts"].shape[0]) != 256 for batch in batches):
        raise RuntimeError("padding input must be exactly two 256-row batches")
    key_sets = [set(batch) for batch in batches]
    if key_sets[0] != key_sets[1]:
        raise RuntimeError("B1/B2 tensor-key drift")
    widths = [int(batch["option_mask"].shape[1]) for batch in batches]
    target_width = max(widths)
    result: dict[str, torch.Tensor] = {}
    padded_keys: list[str] = []
    for key in sorted(key_sets[0]):
        left = batches[0][key]
        right = batches[1][key]
        if left.dtype != right.dtype or left.ndim != right.ndim:
            raise RuntimeError(f"B1/B2 dtype/rank drift for {key}")
        if key in OPTION_AXIS_KEYS:
            if left.shape[2:] != right.shape[2:]:
                raise RuntimeError(f"B1/B2 trailing option shape drift for {key}")
            pieces = [pad_option_axis(batch[key], target_width, key) for batch in batches]
            result[key] = torch.cat(pieces, dim=0)
            if widths[0] != widths[1]:
                padded_keys.append(key)
        else:
            if left.shape[1:] != right.shape[1:]:
                raise RuntimeError(f"non-option B1/B2 shape drift for {key}")
            result[key] = torch.cat((left, right), dim=0)
        if int(result[key].shape[0]) != ROWS:
            raise RuntimeError(f"padded union row-count drift for {key}")

    for batch_index, (batch, width) in enumerate(zip(batches, widths)):
        row_slice = slice(batch_index * BATCH_ROWS, (batch_index + 1) * BATCH_ROWS)
        for key, original in batch.items():
            merged = result[key][row_slice]
            if key in OPTION_AXIS_KEYS:
                if not torch.equal(merged[:, :width], original):
                    raise RuntimeError(f"valid option payload changed for {key}/B{batch_index + 1}")
                if width < target_width and bool(torch.count_nonzero(merged[:, width:])):
                    raise RuntimeError(f"option padding is not zero for {key}/B{batch_index + 1}")
            elif not torch.equal(merged, original):
                raise RuntimeError(f"non-option payload changed for {key}/B{batch_index + 1}")
        active_steps = torch.arange(batch["action_sequences"].shape[1]).unsqueeze(0) < batch[
            "action_counts"
        ].unsqueeze(1)
        active_actions = batch["action_sequences"][active_steps]
        if active_actions.numel() and int(active_actions.max()) >= width:
            raise RuntimeError(f"B{batch_index + 1} active expert action exceeds option width")
    if set(padded_keys) != OPTION_AXIS_KEYS:
        raise RuntimeError(f"option-padding key set drift: {sorted(padded_keys)}")
    return result, {
        "input_option_widths": widths,
        "target_option_width": target_width,
        "padded_tensor_keys": sorted(padded_keys),
        "padding_value": "numeric_zero_or_boolean_false",
        "valid_prefix_payloads_tensor_exact": True,
        "active_expert_actions_within_original_width": True,
    }


def tensor_manifest(batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        key: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in sorted(batch.items())
    }


def build_cache_audit() -> tuple[
    dict[str, Any],
    list[list[dict[str, Any]]],
    dict[str, torch.Tensor],
]:
    _, selections, selection_summary = load_profile_and_selections()
    parent = torch.load(frozen.U468, map_location="cpu", weights_only=False)
    if int(parent.get("update", -1)) != 468:
        raise RuntimeError("raw U468 update drift")
    model_config = parent["model_config"]
    frozen_batches = frozen.load_selected_features(selections, model_config)
    frozen_cache_sha, frozen_batch_sha = repair.replay_cache_manifest(frozen_batches)
    if frozen_cache_sha != EXPECTED_FROZEN_CACHE_SHA256:
        raise RuntimeError(f"frozen B1/B2 cache SHA drift: {frozen_cache_sha}")
    if tuple(frozen_batch_sha) != EXPECTED_FROZEN_BATCH_SHA256:
        raise RuntimeError(f"frozen B1/B2 batch SHA drift: {frozen_batch_sha}")

    padded_union, padding_audit = padded_concat_batches(frozen_batches)
    direct_union, direct_audit = load_direct_union_features(selections, model_config)
    if set(padded_union) != set(direct_union):
        raise RuntimeError("padded/direct single512 tensor-key drift")
    mismatches = [
        key
        for key in sorted(direct_union)
        if not torch.equal(padded_union[key], direct_union[key])
    ]
    if mismatches:
        raise RuntimeError(f"padded union != direct single512: {mismatches}")
    padded_cache_sha, padded_batch_sha = repair.replay_cache_manifest([padded_union])
    direct_cache_sha, direct_batch_sha = repair.replay_cache_manifest([direct_union])
    if (padded_cache_sha, padded_batch_sha) != (direct_cache_sha, direct_batch_sha):
        raise RuntimeError("padded/direct single512 cache-manifest drift")
    if (
        EXPECTED_UNION_CACHE_SHA256 is not None
        and direct_cache_sha != EXPECTED_UNION_CACHE_SHA256
    ):
        raise RuntimeError(f"frozen union cache SHA drift: {direct_cache_sha}")
    if (
        EXPECTED_UNION_BATCH_SHA256 is not None
        and direct_batch_sha != [EXPECTED_UNION_BATCH_SHA256]
    ):
        raise RuntimeError(f"frozen union batch SHA drift: {direct_batch_sha}")
    if int(direct_union["action_counts"].shape[0]) != ROWS:
        raise RuntimeError("single512 action-count row drift")
    if not all(torch.isfinite(value).all() for value in direct_union.values() if value.is_floating_point()):
        raise FloatingPointError("single512 cache contains nonfinite values")
    cache_audit = {
        "status": "zero_write_cache_audit_passed",
        "frozen_b1_b2": {
            "cache_sha256": frozen_cache_sha,
            "batch_sha256": frozen_batch_sha,
            "tensor_manifests": [tensor_manifest(batch) for batch in frozen_batches],
        },
        "aggregate512": {
            "cache_sha256": direct_cache_sha,
            "batch_sha256": direct_batch_sha[0],
            "rows": ROWS,
            "batches": 1,
            "tensor_manifest": tensor_manifest(direct_union),
            "physical_context34_rows": int(
                (direct_union["contexts"] == ppo.SKILL_ORDER_CONTEXT).sum()
            ),
            "effective_context34_sample_weight": float(
                direct_union["sample_weights"][
                    direct_union["contexts"] == ppo.SKILL_ORDER_CONTEXT
                ].sum()
            ),
        },
        "padding_audit": padding_audit,
        "direct_collate_audit": direct_audit,
        "padded_concat_tensor_exact_to_direct_single512": True,
        "gradient_equivalence_basis": (
            "The padded B1+B2 union and independently collated single512 are "
            "tensor-exact for every model/loss input; the actual backward uses "
            "that single direct512 graph, so its union gradient is exactly the "
            "single512 gradient without any B1/B2 backward or optimizer step."
        ),
        "selection": selection_summary,
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }
    return cache_audit, selections, direct_union


def ast_audit() -> dict[str, Any]:
    source_path = normalize_repo_path(__file__, "aggregate runner")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    def dotted_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [
        dotted_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    counts = {
        "loss.backward": calls.count("loss.backward"),
        "torch.nn.utils.clip_grad_norm_": calls.count(
            "torch.nn.utils.clip_grad_norm_"
        ),
        "optimizer.step": calls.count("optimizer.step"),
        "optimizer.zero_grad": calls.count("optimizer.zero_grad"),
        "torch.optim.AdamW": calls.count("torch.optim.AdamW"),
        "torch.save": calls.count("torch.save"),
        "frozen.shadow_training_step": calls.count("frozen.shadow_training_step"),
        "frozen.run_shadow_candidate": calls.count("frozen.run_shadow_candidate"),
    }
    expected = {
        "loss.backward": 1,
        "torch.nn.utils.clip_grad_norm_": 1,
        "optimizer.step": 1,
        "optimizer.zero_grad": 1,
        "torch.optim.AdamW": 1,
        "torch.save": 0,
        "frozen.shadow_training_step": 0,
        "frozen.run_shadow_candidate": 0,
    }
    if counts != expected:
        raise RuntimeError(f"aggregate runner training AST drift: {counts}")
    if "os.O_EXCL" not in source or "FROZEN_SELECTION_RUNNER_SHA256" not in source:
        raise RuntimeError("aggregate runner lacks O_EXCL/hash-bound source contract")
    return {
        "status": "static_ast_audit_passed",
        "training_call_counts_in_source": counts,
        "expected_training_call_counts_in_source": expected,
        "exactly_one_backward_clip_step_source_path": True,
        "microbatch_training_helpers_not_called": True,
        "checkpoint_serialization_calls": 0,
        "selection_runner_hash_checked_before_import": True,
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }


def cuda_runtime_contract() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for aggregate512 preregistration/actual")
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
    }


def expected_contract(
    runner: Mapping[str, Any],
    fixed_inputs: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    result_output: Path,
) -> dict[str, Any]:
    return {
        "runner": {"path": runner["path"], "sha256": runner["sha256"]},
        "frozen_selection_runner": {
            "path": FROZEN_SELECTION_EVIDENCE["path"],
            "sha256": FROZEN_SELECTION_RUNNER_SHA256,
            "hash_verified_before_import": True,
        },
        "branch": BRANCH,
        "runtime": cuda_runtime_contract(),
        "execution_seed": EXECUTION_SEED,
        "training_base": {
            "kind": "raw_full_u468",
            "path": str(frozen.U468.relative_to(ROOT)),
            "checkpoint_sha256": frozen.U468_SHA256,
            "runtime_model_state_sha256": frozen.BASE_MODEL_SHA256,
            "update": 468,
        },
        "selection_profile": {
            "path": str(frozen.PROFILE.relative_to(ROOT)),
            "sha256": frozen.PROFILE_SHA256,
        },
        "dataset_sha256": dict(frozen.DATA_SHA256),
        "dependency_sha256": {
            str(path.relative_to(ROOT)): digest
            for path, digest in frozen.DEPENDENCY_SHA256.items()
        },
        "fixed_input_sha256_verified": True,
        "selection": {
            "seed": frozen.SELECTION_SEED,
            "step_selection_sha256": list(frozen.EXPECTED_STEP_SELECTION_SHA256),
            "frozen_b1_b2_cache_sha256": EXPECTED_FROZEN_CACHE_SHA256,
            "frozen_b1_b2_batch_sha256": list(EXPECTED_FROZEN_BATCH_SHA256),
            "source_counts": EXPECTED_SOURCE_COUNTS,
            "category_counts": EXPECTED_CATEGORY_COUNTS,
        },
        "aggregate512_cache": {
            "cache_sha256": cache_audit["aggregate512"]["cache_sha256"],
            "batch_sha256": cache_audit["aggregate512"]["batch_sha256"],
            "rows": ROWS,
            "batches": 1,
            "input_option_widths": cache_audit["padding_audit"][
                "input_option_widths"
            ],
            "target_option_width": cache_audit["padding_audit"][
                "target_option_width"
            ],
            "padded_tensor_keys": sorted(OPTION_AXIS_KEYS),
            "padded_concat_tensor_exact_to_direct_single512": True,
        },
        "trajectory": {
            "candidate_count": 1,
            "candidate": "raw U468 plus one actor6-only AdamW update",
            "training_batch": "one independently collated union512",
            "forward_for_gradient_calls": 1,
            "backward_calls": 1,
            "gradient_clip_calls": 1,
            "optimizer_step_calls": 1,
            "b1_backward_calls": 0,
            "b2_backward_calls": 0,
            "b1_optimizer_step_calls": 0,
            "b2_optimizer_step_calls": 0,
            "gradient_equivalence": (
                "padded B1+B2 union tensor-exact to direct single512; actual "
                "gradient is taken only from direct single512"
            ),
            "actor_parameter_names": list(ACTOR_NAMES),
            "order_context_weight": ORDER_CONTEXT_WEIGHT,
        },
        "optimizer": {
            "name": "AdamW",
            "fresh": True,
            "lr": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "gates": GATE_CONTRACT,
        "decision_rule": "the sole candidate is GO iff all frozen gates pass",
        "attempt_marker": str(ATTEMPT_MARKER.relative_to(ROOT)),
        "result_output": str(result_output.relative_to(ROOT)),
        "scope": {
            "train_only": True,
            "validation": False,
            "checkpoint_write": False,
            "optimizer_artifact_write": False,
            "model_artifact_write": False,
            "submission": False,
            "RAM_only_model_mutation": True,
            "evidence_writes": ["O_EXCL attempt marker", "O_EXCL result JSON"],
        },
    }


def load_json_object_without_duplicate_keys(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"{label} contains nonfinite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_preregistration(
    path: Path,
    expected_sha256: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        len(expected_sha256) != 64
        or expected_sha256 != expected_sha256.lower()
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected preregistration SHA-256 must be 64 lowercase hex")
    evidence = frozen.require_regular(path, expected_sha256, "aggregate preregistration")
    payload = load_json_object_without_duplicate_keys(path, "aggregate preregistration")
    if payload != {
        "schema_version": PREREGISTRATION_SCHEMA,
        "status": "locked_before_actual",
        "shadow_contract": expected,
    }:
        raise RuntimeError("aggregate preregistration exact-contract mismatch")
    return evidence


def evaluate_union(
    model: torch.nn.Module,
    union: Mapping[str, torch.Tensor],
    selections: Sequence[Sequence[dict[str, Any]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, bool]]:
    flat = [record for batch in selections for record in batch]
    return frozen.evaluate_selected_union(model, [dict(union)], [flat], device)


def execute_single_union_step(
    model: torch.nn.Module,
    union: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    parameters = frozen.configure_actor6(model)
    if tuple(parameters) != ACTOR_NAMES:
        raise RuntimeError("actor6 parameter order drift")
    optimizer = torch.optim.AdamW(
        list(parameters.values()),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if optimizer.state:
        raise RuntimeError("fresh aggregate512 AdamW unexpectedly has state")
    counters = {
        "forward_for_gradient_calls": 0,
        "backward_calls": 0,
        "gradient_clip_calls": 0,
        "optimizer_step_calls": 0,
        "b1_backward_calls": 0,
        "b2_backward_calls": 0,
        "b1_optimizer_step_calls": 0,
        "b2_optimizer_step_calls": 0,
    }
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    batch = {key: value.to(device, non_blocking=True) for key, value in union.items()}
    if int(batch["action_counts"].shape[0]) != ROWS:
        raise RuntimeError("gradient batch is not single512")
    counters["forward_for_gradient_calls"] += 1
    outputs = ppo.model_forward(model, batch, device)
    loss, parts = ppo.bc_expert_actor_loss(
        outputs,
        batch,
        loss_mode="ordered",
        order_context_weight=ORDER_CONTEXT_WEIGHT,
        non_context34_fixed_multi_action_order_weight=1.0,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("aggregate512 scalar loss is nonfinite")
    counters["backward_calls"] += 1
    loss.backward()
    gradient_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    )
    per_tensor_gradient: dict[str, dict[str, Any]] = {}
    for name, parameter in parameters.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"missing/nonfinite aggregate512 gradient: {name}")
        gradient = parameter.grad.detach()
        per_tensor_gradient[name] = {
            "l2": math.sqrt(float(gradient.double().square().sum())),
            "max_abs": float(gradient.abs().max()),
            "nonzero_elements": int(torch.count_nonzero(gradient)),
        }
    counters["gradient_clip_calls"] += 1
    preclip = torch.nn.utils.clip_grad_norm_(list(parameters.values()), MAX_GRAD_NORM)
    preclip_value = float(preclip.detach().cpu())
    postclip_value = math.sqrt(
        sum(
            float(parameter.grad.detach().double().square().sum())
            for parameter in parameters.values()
            if parameter.grad is not None
        )
    )
    counters["optimizer_step_calls"] += 1
    optimizer.step()
    state = optimizer.state_dict()
    steps = repair.optimizer_steps(state)
    if len(state["state"]) != 6 or len(steps) != 6 or set(steps) != {1}:
        raise RuntimeError("aggregate512 fresh AdamW state/step drift")
    expected_counters = {
        "forward_for_gradient_calls": 1,
        "backward_calls": 1,
        "gradient_clip_calls": 1,
        "optimizer_step_calls": 1,
        "b1_backward_calls": 0,
        "b2_backward_calls": 0,
        "b1_optimizer_step_calls": 0,
        "b2_optimizer_step_calls": 0,
    }
    if counters != expected_counters:
        raise RuntimeError(f"aggregate512 operation-count drift: {counters}")
    return {
        "loss": float(loss.detach().cpu()),
        "loss_parts": {key: float(value.detach().cpu()) for key, value in parts.items()},
        "gradient_parameter_names": gradient_names,
        "gradient_scope_exact_actor6": gradient_names == sorted(ACTOR_NAMES),
        "all_six_gradient_tensors_nonzero": all(
            item["nonzero_elements"] > 0 for item in per_tensor_gradient.values()
        ),
        "per_tensor_preclip_gradient": per_tensor_gradient,
        "preclip_gradient_l2": preclip_value,
        "postclip_gradient_l2": postclip_value,
        "clip_max_norm": MAX_GRAD_NORM,
        "optimizer_state_count": len(state["state"]),
        "optimizer_steps": sorted(set(steps)),
        "optimizer_state_finite": repair.finite_nested(state),
        "operation_counts": counters,
        "training_batch_rows": ROWS,
        "training_batch_kind": "independently_collated_direct_single512",
        "gradient_matches_padded_union_by_tensor_exact_input_identity": True,
    }


def run_single_candidate(
    union: Mapping[str, torch.Tensor],
    selections: Sequence[Sequence[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    model, _ = frozen.load_raw_u468(device)
    base_state = repair.clone_model_state(model)
    raw_evaluation, raw_correct = evaluate_union(model, union, selections, device)
    if raw_evaluation["retention"] != {
        "rows": 288,
        "ordered_correct": 288,
        "fragile_rows": 282,
        "fragile_ordered_correct": 282,
        "context34_rows": 6,
        "context34_ordered_correct": 6,
    }:
        raise RuntimeError("raw U468 aggregate512 retention correctness drift")
    training = execute_single_union_step(model, union, device)
    endpoint_evaluation, endpoint_correct = evaluate_union(
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
    union_improvement = float(raw_evaluation["mixed_ordered_loss"]) - float(
        endpoint_evaluation["mixed_ordered_loss"]
    )
    hard_loss_by_source: dict[str, Any] = {}
    for source in frozen.DATASETS:
        raw_loss = float(
            raw_evaluation["by_source_and_bucket"][source]["hard"]["ordered_loss"]
        )
        endpoint_loss = float(
            endpoint_evaluation["by_source_and_bucket"][source]["hard"][
                "ordered_loss"
            ]
        )
        improvement = raw_loss - endpoint_loss
        hard_loss_by_source[source] = {
            "raw": raw_loss,
            "endpoint": endpoint_loss,
            "improvement": improvement,
            "pass": improvement >= 1e-8,
        }
    displacement_l2 = float(displacement["l2"])
    gate_checks = {
        "all_finite": repair.finite_nested(
            {
                "training": training,
                "raw_evaluation": raw_evaluation,
                "endpoint_evaluation": endpoint_evaluation,
                "displacement": displacement,
                "retention_flips": retention_flips,
                "hard_loss_by_source": hard_loss_by_source,
                "endpoint_state": endpoint_state,
            }
        ),
        "gradient_scope_exact_actor6_and_all_six_nonzero": bool(
            training["gradient_scope_exact_actor6"]
            and training["all_six_gradient_tensors_nonzero"]
        ),
        "changed_scope_exact_actor6_and_frozen_tensors_unchanged": bool(
            displacement["changed_scope_exact_actor6"] and frozen_unchanged
        ),
        "fresh_adamw_optimizer_state_count_6_step_1": bool(
            training["optimizer_state_count"] == 6
            and training["optimizer_steps"] == [1]
            and training["optimizer_state_finite"]
        ),
        "displacement_l2_in_open_closed_interval": 0.0 < displacement_l2 <= 3.10e-5,
        "union_mixed_ordered_loss_improvement": union_improvement >= 1e-8,
        "each_source_hard_ordered_loss_improvement": all(
            item["pass"] for item in hard_loss_by_source.values()
        ),
        "retention_288_zero_correct_to_wrong": bool(
            retention_flips["retention_rows"] == 288
            and retention_flips["raw_retention_correct"] == 288
            and retention_flips["total_correct_to_wrong"] == 0
        ),
        "exactly_one_union512_backward_clip_step": training["operation_counts"]
        == {
            "forward_for_gradient_calls": 1,
            "backward_calls": 1,
            "gradient_clip_calls": 1,
            "optimizer_step_calls": 1,
            "b1_backward_calls": 0,
            "b2_backward_calls": 0,
            "b1_optimizer_step_calls": 0,
            "b2_optimizer_step_calls": 0,
        },
        "padded_union_gradient_equals_direct_single512": bool(
            training["gradient_matches_padded_union_by_tensor_exact_input_identity"]
        ),
    }
    return {
        "candidate": "raw_u468_actor6_aggregate512_lr1.125e-7_one_step",
        "candidate_count": 1,
        "learning_rate": LEARNING_RATE,
        "optimizer": {
            "name": "AdamW",
            "fresh": True,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "foreach": False,
            "fused": False,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "training": training,
        "evaluations": {"raw": raw_evaluation, "endpoint": endpoint_evaluation},
        "displacement_from_raw": displacement,
        "frozen_tensors_unchanged": frozen_unchanged,
        "union_mixed_ordered_loss_improvement": union_improvement,
        "hard_ordered_loss_by_source": hard_loss_by_source,
        "retention_flips_from_raw": retention_flips,
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
    runner = runner_evidence()
    fixed_inputs = verify_fixed_inputs()
    ast_result = ast_audit()
    if ATTEMPT_MARKER.exists() or ATTEMPT_MARKER.is_symlink():
        raise FileExistsError(ATTEMPT_MARKER)

    evidence_args = (
        args.result_output,
        args.preregistration,
        args.expected_preregistration_sha256,
    )
    if args.mode == "static-audit":
        if args.device != "cpu" or any(value is not None for value in evidence_args):
            raise ValueError("static-audit is CPU/default and accepts no evidence args")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "zero_write_static_audit_passed",
                    "runner": runner,
                    "fixed_inputs": fixed_inputs,
                    "ast_audit": ast_result,
                    "attempt_marker_absent": True,
                    "validation_member_payloads_opened": False,
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    cache_audit, selections, union = build_cache_audit()
    if args.mode == "cache-audit":
        if args.device != "cpu" or any(value is not None for value in evidence_args):
            raise ValueError("cache-audit is CPU/default and accepts no evidence args")
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA,
                    "status": "zero_write_cache_audit_passed",
                    "runner": runner,
                    "fixed_inputs": fixed_inputs,
                    "ast_audit": ast_result,
                    "cache": cache_audit,
                    "attempt_marker_absent": True,
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
            raise ValueError(
                "shadow-contract requires --device cuda and --result-output only"
            )
        result_output = normalize_repo_path(args.result_output, "shadow result")
        if result_output == ATTEMPT_MARKER:
            raise ValueError("result path collides with attempt marker")
        require_absent_target(result_output, "shadow result")
        contract = expected_contract(runner, fixed_inputs, cache_audit, result_output)
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
        raise ValueError(
            "actual requires --device cuda, result, preregistration, and exact SHA"
        )
    result_output = normalize_repo_path(args.result_output, "shadow result")
    preregistration = normalize_repo_path(args.preregistration, "preregistration")
    if result_output in {ATTEMPT_MARKER, preregistration}:
        raise ValueError("actual control-plane path collision")
    require_absent_target(result_output, "shadow result")
    contract = expected_contract(runner, fixed_inputs, cache_audit, result_output)
    preregistration_evidence = validate_preregistration(
        preregistration,
        args.expected_preregistration_sha256,
        contract,
    )

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    random.seed(EXECUTION_SEED)
    torch.manual_seed(EXECUTION_SEED)
    torch.cuda.manual_seed_all(EXECUTION_SEED)
    # Recheck all mutable bindings and output absence immediately before the
    # durable one-shot marker.  No model or optimizer exists before this lock.
    fixed_inputs_at_lock = verify_fixed_inputs()
    if fixed_inputs_at_lock != fixed_inputs:
        raise RuntimeError("fixed-input identity changed before actual lock")
    if runner_evidence()["sha256"] != runner["sha256"]:
        raise RuntimeError("aggregate runner changed before actual lock")
    require_absent_target(result_output, "shadow result")
    require_absent_target(ATTEMPT_MARKER, "attempt marker")
    marker_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "actual_attempt_consumed_before_model_or_optimizer_creation",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "runner": runner,
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

    candidate = run_single_candidate(union, selections, device)
    torch.cuda.synchronize(device)
    decision = "GO" if candidate["fully_passes"] else "NO_GO"
    result = {
        "schema_version": SCHEMA,
        "status": "completed_actual_ram_only_single_union512_step",
        "created_at_utc": utc_now(),
        "branch": BRANCH,
        "decision": decision,
        "single_candidate_only": True,
        "candidate": candidate,
        "runner": runner,
        "fixed_inputs": fixed_inputs,
        "ast_audit": ast_result,
        "cache": cache_audit,
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
        raise FloatingPointError("aggregate512 result contains nonfinite values")
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
