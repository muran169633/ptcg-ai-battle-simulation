#!/usr/bin/env python3
"""Reprofile frozen train archives at the exact historical CW11 RAM state.

The CW11 state exists only inside the already-consumed historical callback.  It
is never loaded from the evaluation-only materialization.  Only ``train/``
members of the three frozen specialist archives are opened, and no changed
candidate is created or evaluated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import stat
import sys
import types
import zipfile
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "profile_u468_cw11_full_train_margins_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-exact-cw11-full-train-margin-profile-v1"
SEED = 202608304

CW21 = TOOLS / "probe_u468_cw11_fixed_pcgrad_specialbc_cw21_v1.py"
CW21_SHA256 = "9c18db25e16fe6b9feb1ca72d0682a1baa31ad6dc19e1a6283fd7db2da0f9cf4"
CW21_MODE = 0o555
RAW_PROFILE = TOOLS / "profile_u468_raw_full_train_margins.py"
RAW_PROFILE_SHA256 = "c70e52e2398b8551baf0fa3a3a403fbf6636b56c70010fb948a5c85c298977ae"
RAW_PROFILE_MODE = 0o555
FRAMEWORK = TOOLS / "profile_u468_beta1157_train_margins.py"
FRAMEWORK_SHA256 = "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142"
FRAMEWORK_MODE = 0o664

CW11_MODEL_SHA256 = "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
CW11_VECTOR_SHA256 = "e7183412ec13e9b69b2b50c95b4ebc6fc603a3d934229ca47b00538abb95abcd"
CW11_LEDGER_SHA256 = "6ae0c07a94627cb6fd0155f265f75d13bd7505b0e5f62cd62c3e26ceb7714436"
CW11_ACTOR_FLOAT32_LE_SHA256 = (
    "fc13661fec1801768a11d2c0366602727df08d6773c982f2ba273e194b8fafa6"
)
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
RAW_NONACTOR_SHA256 = "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
EXPECTED_ACTOR_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)

DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
EXPECTED_DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}


class ProtocolError(RuntimeError):
    """Fail-closed profile protocol error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_evidence(
    path: Path,
    expected_sha256: str,
    label: str,
    expected_mode: int | None = None,
) -> dict[str, Any]:
    before = path.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ProtocolError(f"{label} must be a single-link regular file")
    digest = sha256_file(path)
    after = path.lstat()
    identity_stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    checks = {
        "identity_stable": identity_stable,
        "sha256_exact": digest == expected_sha256,
        "regular_single_link": stat.S_ISREG(after.st_mode)
        and not stat.S_ISLNK(after.st_mode)
        and int(after.st_nlink) == 1,
        "mode_exact_if_bound": expected_mode is None
        or stat.S_IMODE(after.st_mode) == expected_mode,
    }
    if not all(checks.values()):
        raise ProtocolError(f"{label} evidence drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(after.st_size),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "nlink": int(after.st_nlink),
        "checks": checks,
    }


def load_locked_source(
    path: Path,
    expected_sha256: str,
    expected_mode: int,
    module_name: str,
) -> tuple[ModuleType, dict[str, Any]]:
    evidence = regular_evidence(path, expected_sha256, module_name, expected_mode)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ProtocolError(f"{module_name}: source changed after evidence read")
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    previous = sys.modules.get(module_name)
    if previous is not None:
        raise ProtocolError(f"{module_name}: canonical module name already occupied")
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module, evidence


def source_audit() -> dict[str, Any]:
    source = SCRIPT.read_bytes()
    tree = ast.parse(source, filename=str(SCRIPT))
    torch_load_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    load_state_dict_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
    ]
    run_probe_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_probe"
    ]
    profile_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "profile_archive"
    ]
    checks = {
        "no_direct_torch_checkpoint_load": not torch_load_lines,
        "no_load_state_dict": not load_state_dict_lines,
        "exact_one_historical_run_probe_site": len(run_probe_lines) == 1,
        "exact_one_profile_archive_call_site": len(profile_lines) == 1,
        "dev_null_pycache_literal_present": b'sys.pycache_prefix = "/dev/null"'
        in source,
        "stdout_summary_present": any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            for node in ast.walk(tree)
        ),
    }
    if not all(checks.values()):
        raise ProtocolError(f"source audit failed: {checks}")
    return {
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
        "checks": checks,
        "torch_load_lines": torch_load_lines,
        "load_state_dict_lines": load_state_dict_lines,
        "run_probe_lines": run_probe_lines,
        "profile_archive_lines": profile_lines,
        "pass": True,
    }


def validate_runtime(require_cuda: bool) -> dict[str, Any]:
    if Path.cwd().resolve() != ROOT:
        raise ProtocolError("run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError("requires my_project_env Python")
    checks = {
        "isolated_mode": sys.flags.isolated == 1,
        "dont_write_bytecode_flag": sys.flags.dont_write_bytecode == 1,
        "dont_write_bytecode_runtime": sys.dont_write_bytecode is True,
        "pycache_prefix_exact_dev_null": sys.pycache_prefix == "/dev/null",
        "dev_null_character_device": stat.S_ISCHR(Path("/dev/null").stat().st_mode),
    }
    if not all(checks.values()):
        raise ProtocolError(f"runtime isolation failed: {checks}")
    result: dict[str, Any] = {
        "python": str(Path(sys.executable).resolve()),
        "checks": checks,
        "pass": True,
    }
    if require_cuda:
        import torch

        cuda_checks = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count_positive": torch.cuda.device_count() > 0,
        }
        if not all(cuda_checks.values()):
            raise ProtocolError(f"CUDA unavailable: {cuda_checks}")
        result["cuda"] = {
            "checks": cuda_checks,
            "device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    return result


def archive_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence: dict[str, Any] = {}
    inventory: dict[str, Any] = {}
    for label, path in DATASETS.items():
        evidence[label] = regular_evidence(
            path, EXPECTED_DATA_SHA256[label], f"{label} train archive"
        )
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if len(names) != len(set(names)):
            raise ProtocolError(f"{label}: duplicate ZIP member names")
        train_members = sorted(
            name
            for name in names
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if not train_members:
            raise ProtocolError(f"{label}: no train JSONL members")
        if any(
            name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            for name in train_members
        ):
            raise ProtocolError(f"{label}: noncanonical train ZIP member")
        inventory[label] = {
            "all_member_count": len(names),
            "train_jsonl_member_count": len(train_members),
            "train_jsonl_members": train_members,
            "archive_members_opened_during_inventory": 0,
        }
    return evidence, inventory


def scalar_option_signature(option: Any) -> dict[str, Any]:
    if not isinstance(option, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("type", "area", "index", "playerIndex"):
        value = option.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            result[key] = value
        elif isinstance(value, str):
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def int_or_default(value: Any, default: int = -1) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def enrich_retained_train_rows(
    path: Path,
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    wanted: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for list_name in ("near_wrong", "fragile_correct"):
        records = panel.get(list_name)
        if not isinstance(records, list):
            raise ProtocolError(f"malformed retained list: {list_name}")
        for record in records:
            if not isinstance(record, dict):
                raise ProtocolError(f"malformed retained record: {list_name}")
            member = str(record.get("member", ""))
            line_index = int(record.get("line_index", -1))
            if not member.startswith("train/") or line_index < 0:
                raise ProtocolError("retained identity is not train-only")
            wanted[(member, line_index)].append(record)

    by_member: dict[str, set[int]] = defaultdict(set)
    for member, line_index in wanted:
        by_member[member].add(line_index)
    found: set[tuple[str, int]] = set()
    opened: list[str] = []
    with zipfile.ZipFile(path) as archive:
        archive_names = set(archive.namelist())
        for member in sorted(by_member):
            if member not in archive_names or not member.startswith("train/"):
                raise ProtocolError(f"unsafe enrichment member: {member}")
            opened.append(member)
            remaining = set(by_member[member])
            with archive.open(member) as handle:
                for line_index, raw in enumerate(handle):
                    if line_index not in remaining:
                        continue
                    row = json.loads(raw)
                    if str(row.get("split", "")) != "train":
                        raise ProtocolError(f"non-train enrichment row: {member}")
                    key = (member, line_index)
                    for record in wanted[key]:
                        if hashlib.sha256(raw).hexdigest() != str(
                            record.get("line_sha256", "")
                        ):
                            raise ProtocolError("retained line SHA drift")
                        observation = row.get("observation")
                        select = (
                            observation.get("select")
                            if isinstance(observation, Mapping)
                            else None
                        )
                        options = (
                            select.get("option")
                            if isinstance(select, Mapping)
                            else None
                        )
                        if not isinstance(options, list):
                            options = []
                        indices = sorted(
                            {
                                int(value)
                                for field in ("expert_order", "predicted_order")
                                for value in record.get(field, [])
                                if isinstance(value, int)
                                and 0 <= int(value) < len(options)
                            }
                        )
                        record["train_row_metadata"] = {
                            "select_type": int_or_default(row.get("select_type")),
                            "select_context": int_or_default(
                                row.get("select_context")
                            ),
                            "seat": int_or_default(row.get("seat")),
                            "opponent_team_name": str(
                                row.get("opponent_team_name", "")
                            ),
                            "selected_option_signatures": {
                                str(index): scalar_option_signature(options[index])
                                for index in indices
                            },
                        }
                    found.add(key)
                    remaining.remove(line_index)
                    if not remaining:
                        break
            if remaining:
                raise ProtocolError(f"missing enrichment rows: {member}: {remaining}")
    if found != set(wanted):
        raise ProtocolError("enrichment identity coverage drift")
    return {
        "opened_members": opened,
        "non_train_members_opened": False,
        "unique_retained_rows_enriched": len(wanted),
        "retained_record_references_enriched": sum(len(value) for value in wanted.values()),
        "coverage_exact": True,
    }


def nonempty(record: Mapping[str, Any]) -> bool:
    expert = record.get("expert_order")
    return isinstance(expert, list) and bool(expert)


def fixed_cardinality(record: Mapping[str, Any]) -> bool:
    return int(record.get("min_count", -1)) == int(record.get("max_count", -2))


def single_action_count_safe(record: Mapping[str, Any]) -> bool:
    expert = record.get("expert_order")
    predicted = record.get("predicted_order")
    if (
        not isinstance(expert, list)
        or not isinstance(predicted, list)
        or len(expert) != 1
        or len(predicted) != 1
    ):
        return False
    minimum = int(record.get("min_count", -1))
    maximum = int(record.get("max_count", -2))
    if not (minimum <= 1 <= maximum):
        return False
    margin = record.get("count_margin")
    return fixed_cardinality(record) or (
        isinstance(margin, (int, float)) and math.isfinite(float(margin)) and margin > 0
    )


def sanitize_profile_nonfinite(
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace JSON-forbidden diagnostic infinities with explicit nulls."""

    replacements: dict[str, int] = defaultdict(int)
    for label, panel in profiles.items():
        for list_name in ("near_wrong", "fragile_correct"):
            records = panel.get(list_name)
            if not isinstance(records, list):
                raise ProtocolError(f"{label}: malformed retained list before sanitize")
            for record in records:
                for key in (
                    "selection_margin",
                    "count_margin",
                    "decision_margin",
                    "predicted_log_probability",
                ):
                    value = record.get(key)
                    if isinstance(value, float) and not math.isfinite(value):
                        record[key] = None
                        record[f"{key}_nonfinite_reason"] = (
                            "no_competing_allowed_choice"
                            if value > 0
                            else "nonfinite_model_diagnostic"
                        )
                        replacements[f"{label}.{list_name}.{key}"] += 1
    return {
        "encoding": "JSON null plus sibling nonfinite_reason field",
        "replacement_counts": dict(sorted(replacements.items())),
        "replacement_total": sum(replacements.values()),
    }


def unique_capacity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "unique_line_sha256": len(
            {str(record.get("line_sha256", "")) for record in records}
        ),
        "unique_episodes": len(
            {str(record.get("episode_id", "")) for record in records}
        ),
    }


def option_pair_matches(record: Mapping[str, Any], expected: Mapping[str, int]) -> bool:
    metadata = record.get("train_row_metadata")
    signatures = (
        metadata.get("selected_option_signatures")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(signatures, Mapping) or len(signatures) < 2:
        return False
    matching = 0
    for signature in signatures.values():
        if isinstance(signature, Mapping) and all(
            int(signature.get(key, -1)) == value for key, value in expected.items()
        ):
            matching += 1
    return matching >= 2


def targeted_capacity(profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pf_wrong = [
        record
        for record in profiles["pokemonfan"]["near_wrong"]
        if record.get("ordered_correct") is False
        and record.get("set_correct") is False
        and single_action_count_safe(record)
    ]
    pf_retention = [
        record
        for record in profiles["pokemonfan"]["fragile_correct"]
        if record.get("ordered_correct") is True
        and record.get("set_correct") is True
        and single_action_count_safe(record)
    ]
    core_wrong = [
        record
        for record in profiles["core5"]["near_wrong"]
        if record.get("ordered_correct") is False
        and record.get("set_correct") is False
        and single_action_count_safe(record)
    ]
    core_retention = [
        record
        for record in profiles["core5"]["fragile_correct"]
        if record.get("ordered_correct") is True
        and record.get("set_correct") is True
        and single_action_count_safe(record)
    ]
    flg_retention = [
        record
        for record in profiles["flg"]["fragile_correct"]
        if record.get("ordered_correct") is True
        and record.get("set_correct") is True
        and single_action_count_safe(record)
    ]
    panels: dict[str, Any] = {
        "pokemonfan_context0_hard": unique_capacity(
            [record for record in pf_wrong if int(record.get("context", -1)) == 0]
        ),
        "pokemonfan_context7_hard": unique_capacity(
            [record for record in pf_wrong if int(record.get("context", -1)) == 7]
        ),
        "pokemonfan_context0_retention": unique_capacity(
            [
                record
                for record in pf_retention
                if int(record.get("context", -1)) == 0
            ]
        ),
        "pokemonfan_context7_retention": unique_capacity(
            [
                record
                for record in pf_retention
                if int(record.get("context", -1)) == 7
            ]
        ),
        "dominic_context0_hard": unique_capacity(
            [
                record
                for record in core_wrong
                if str(record.get("team_name", "")) == "Dominic Peel"
                and int(record.get("context", -1)) == 0
            ]
        ),
        "dominic_context0_type10_area5_pair_hard": unique_capacity(
            [
                record
                for record in core_wrong
                if str(record.get("team_name", "")) == "Dominic Peel"
                and int(record.get("context", -1)) == 0
                and option_pair_matches(record, {"type": 10, "area": 5})
            ]
        ),
        "dominic_context0_retention": unique_capacity(
            [
                record
                for record in core_retention
                if str(record.get("team_name", "")) == "Dominic Peel"
                and int(record.get("context", -1)) == 0
            ]
        ),
        "flg_context3_retention": unique_capacity(
            [
                record
                for record in flg_retention
                if int(record.get("context", -1)) == 3
            ]
        ),
    }
    required_minimums = {
        "pokemonfan_context0_hard": 32,
        "pokemonfan_context7_hard": 32,
        "pokemonfan_context0_retention": 32,
        "pokemonfan_context7_retention": 32,
        "dominic_context0_hard": 32,
        "dominic_context0_retention": 32,
        "flg_context3_retention": 16,
    }
    checks = {
        key: panels[key]["unique_episodes"] >= minimum
        for key, minimum in required_minimums.items()
    }
    return {
        "purpose": (
            "capacity only for a separately frozen targeted CW22 endpoint; "
            "not a training selection"
        ),
        "known_CW11_harm_targets": {
            "pokemonfan_context0_valid_flips": 3,
            "pokemonfan_context7_valid_flips": 2,
            "dominic_context0_shared_core5_valid_flip": 1,
            "valid_rows_used_as_training_examples": 0,
        },
        "panels": panels,
        "required_minimum_unique_episodes": required_minimums,
        "checks": checks,
        "pass": all(checks.values()),
    }


def validate_exact_cw11_context(
    context: Mapping[str, Any],
    cw20: ModuleType,
    cw15: ModuleType,
) -> dict[str, Any]:
    import numpy as np

    required = {
        "helper",
        "model",
        "checkpoint",
        "model_config",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "success_iteration",
        "cw10_success_iteration",
        "selected_row_gate",
        "active_pair_ledger",
        "expanded_row_count",
        "candidate_model_state_sha256",
    }
    if not required.issubset(context):
        raise ProtocolError("historical CW11 callback schema drift")
    helper = context["helper"]
    model = context["model"]
    state = model.state_dict()
    device = next(model.parameters()).device
    terminal = np.asarray(context["terminal_cumulative_float64"], dtype=np.float64)
    ledger_sha = hashlib.sha256(
        cw15.canonical_json(context["active_pair_ledger"])
    ).hexdigest()
    checkpoint = context["checkpoint"]
    checkpoint_state = (
        checkpoint.get("model_state_dict") if isinstance(checkpoint, Mapping) else None
    )
    checkpoint_model_sha = (
        helper.model_state_sha256(checkpoint_state)
        if isinstance(checkpoint_state, Mapping)
        else None
    )
    named = dict(model.named_parameters())
    actor_bytes = cw20.actor_bytes(
        named, EXPECTED_ACTOR_NAMES, np
    )
    model_config = context["model_config"]
    checkpoint_model_config = (
        checkpoint.get("model_config") if isinstance(checkpoint, Mapping) else None
    )
    checks = {
        "device_cuda": device.type == "cuda",
        "exact_CW11_model": helper.model_state_sha256(state) == CW11_MODEL_SHA256,
        "context_candidate_exact_CW11": str(context["candidate_model_state_sha256"])
        == CW11_MODEL_SHA256,
        "exact_CW11_vector_field": str(
            context["terminal_cumulative_float64_le_sha256"]
        )
        == CW11_VECTOR_SHA256,
        "exact_CW11_vector_recomputed": cw15.float64_vector_sha256(terminal)
        == CW11_VECTOR_SHA256,
        "CW11_vector_shape_finite": terminal.shape == (cw15.ACTOR6_FLAT_LENGTH,)
        and bool(np.isfinite(terminal).all()),
        "CW11_vector_l2_exact": math.isclose(
            float(np.linalg.norm(terminal)),
            0.00792176975336988,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "exact_CW11_ledger": ledger_sha == CW11_LEDGER_SHA256,
        "raw_model_anchor": str(context["raw_model_state_sha256"])
        == RAW_MODEL_SHA256,
        "raw_nonactor_anchor": str(context["raw_nonactor_sha256"])
        == RAW_NONACTOR_SHA256,
        "model_eval": model.training is False,
        "legacy_ledger34": len(context["active_pair_ledger"]) == 34,
        "success_iteration_exact3": int(context["success_iteration"]) == 3,
        "cw10_success_iteration_exact9": int(context["cw10_success_iteration"]) == 9,
        "expanded_row_count_exact33": int(context["expanded_row_count"]) == 33,
        "selected_row_gate_pass": context["selected_row_gate"].get("pass") is True,
        "model_tensor_count80": len(state) == 80,
        "actor_names_present": all(name in named for name in EXPECTED_ACTOR_NAMES),
        "actor_dimension_exact65793": sum(
            int(named[name].numel()) for name in EXPECTED_ACTOR_NAMES
        )
        == cw15.ACTOR6_FLAT_LENGTH,
        "actor6_all_float32": all(
            named[name].dtype == helper.torch.float32 for name in EXPECTED_ACTOR_NAMES
        ),
        "actor_bytes_exact": hashlib.sha256(actor_bytes).hexdigest()
        == CW11_ACTOR_FLOAT32_LE_SHA256,
        "nonactor_exact_raw": cw20.nonactor_sha(
            model, EXPECTED_ACTOR_NAMES, context["helper"]
        )
        == RAW_NONACTOR_SHA256,
        "raw_parent_dictionary": isinstance(checkpoint, Mapping),
        "raw_parent_update468": isinstance(checkpoint, Mapping)
        and int(checkpoint.get("update", -1)) == 468,
        "raw_parent_model_exact": checkpoint_model_sha == RAW_MODEL_SHA256,
        "model_config_mapping": isinstance(model_config, Mapping),
        "model_config_exact_raw_parent": isinstance(checkpoint_model_config, Mapping)
        and dict(model_config) == dict(checkpoint_model_config),
        "raw_parent_not_eval_only": isinstance(checkpoint, Mapping)
        and "evaluation_only" not in checkpoint
        and "resume_forbidden" not in checkpoint,
    }
    if not all(checks.values()):
        raise ProtocolError(f"exact CW11 context drift: {checks}")
    return checks


def profile_callback(
    context: Mapping[str, Any],
    cw20: ModuleType,
    cw15: ModuleType,
    framework: ModuleType,
    raw_profile: ModuleType,
    batch_size: int,
    keep: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    checks = validate_exact_cw11_context(context, cw20, cw15)
    model = context["model"]
    helper = context["helper"]
    named = dict(model.named_parameters())
    state_before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    requires_grad_before = {
        name: bool(parameter.requires_grad) for name, parameter in named.items()
    }
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in named.items()
    }
    if any(value is not None for value in gradients_before.values()):
        raise ProtocolError("incoming exact CW11 unexpectedly has gradients")
    training_before = bool(model.training)
    precision_before = torch.get_float32_matmul_precision()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state_all()
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    bc_max_action_before = framework.bc.MAX_ACTION_COUNT
    deterministic_debug_before = torch.get_deterministic_debug_mode()
    cuda_matmul_tf32_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32_before = torch.backends.cudnn.allow_tf32
    result: dict[str, Any]
    try:
        torch.set_float32_matmul_precision("high")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        profiles: dict[str, Any] = {}
        enrichment: dict[str, Any] = {}
        for label, path in DATASETS.items():
            panel = framework.profile_archive(
                label,
                path,
                model,
                context["model_config"],
                next(model.parameters()).device,
                batch_size,
                keep,
            )
            opened = panel.get("opened_members")
            panel_checks = {
                "opened_nonempty_list": isinstance(opened, list) and bool(opened),
                "all_opened_members_train": isinstance(opened, list)
                and all(str(member).startswith("train/") for member in opened),
                "non_train_members_opened_false": panel.get(
                    "non_train_members_opened"
                )
                is False,
                "archive_sha_exact": panel.get("archive_sha256")
                == EXPECTED_DATA_SHA256[label],
            }
            if not all(panel_checks.values()):
                raise ProtocolError(f"{label}: train-only profile drift: {panel_checks}")
            panel["train_only_checks"] = panel_checks
            enrichment[label] = enrich_retained_train_rows(path, panel)
            per_archive_integrity = {
                "model_hash_exact_CW11": helper.model_state_sha256(
                    model.state_dict()
                )
                == CW11_MODEL_SHA256,
                "model_still_eval": model.training is False,
                "all_gradients_absent": all(
                    parameter.grad is None for parameter in named.values()
                ),
            }
            if not all(per_archive_integrity.values()):
                raise ProtocolError(
                    f"{label}: model integrity drift after archive: "
                    f"{per_archive_integrity}"
                )
            panel["post_archive_model_integrity"] = per_archive_integrity
            profiles[label] = panel
        state_unchanged = helper.model_state_sha256(model.state_dict())
        if state_unchanged != CW11_MODEL_SHA256:
            raise ProtocolError("profile mutated exact CW11 model")
        if any(parameter.grad is not None for parameter in named.values()):
            raise ProtocolError("profile created model gradient buffers")
        capacity = raw_profile.recipe_capacity(profiles)
        targeted = targeted_capacity(profiles)
        nonfinite_encoding = sanitize_profile_nonfinite(profiles)
        result = {
            "status": "completed_exact_CW11_train_only_profile",
            "context_checks": checks,
            "profiles": profiles,
            "enrichment": enrichment,
            "balanced_recipe_capacity": capacity,
            "targeted_CW22_capacity": targeted,
            "nonfinite_margin_encoding": nonfinite_encoding,
            "model_config_canonical_sha256": hashlib.sha256(
                (
                    json.dumps(
                        dict(context["model_config"]),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest(),
            "profile_model_state_sha256_after": state_unchanged,
            "changed_candidate_created": False,
            "changed_candidate_train_endpoint_count": 0,
            "changed_candidate_validation_evaluation_count": 0,
        }
    finally:
        with torch.no_grad():
            current = model.state_dict()
            if set(current) != set(state_before):
                raise ProtocolError("model state key drift during profiling")
            for name, value in state_before.items():
                current[name].copy_(value)
        for name, parameter in named.items():
            parameter.requires_grad_(requires_grad_before[name])
            parameter.grad = (
                None
                if gradients_before[name] is None
                else gradients_before[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
            )
        model.train(training_before)
        framework.bc.MAX_ACTION_COUNT = bc_max_action_before
        torch.set_float32_matmul_precision(precision_before)
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cuda.matmul.allow_tf32 = cuda_matmul_tf32_before
        torch.backends.cudnn.allow_tf32 = cudnn_tf32_before
        torch.random.set_rng_state(cpu_rng_before)
        torch.cuda.set_rng_state_all(cuda_rng_before)
        random.setstate(python_rng_before)
        np.random.set_state(numpy_rng_before)
    restore_checks = {
        "model_hash_exact": helper.model_state_sha256(model.state_dict())
        == CW11_MODEL_SHA256,
        "training_flag_exact": bool(model.training) == training_before,
        "requires_grad_exact": all(
            bool(parameter.requires_grad) == requires_grad_before[name]
            for name, parameter in named.items()
        ),
        "gradient_presence_exact": all(
            (parameter.grad is None) == (gradients_before[name] is None)
            for name, parameter in named.items()
        ),
        "precision_exact": torch.get_float32_matmul_precision() == precision_before,
        "deterministic_setting_exact": torch.are_deterministic_algorithms_enabled()
        == deterministic_before,
        "deterministic_warn_only_exact": torch.is_deterministic_algorithms_warn_only_enabled()
        == warn_only_before,
        "deterministic_debug_mode_exact": torch.get_deterministic_debug_mode()
        == deterministic_debug_before,
        "cudnn_benchmark_exact": torch.backends.cudnn.benchmark
        == cudnn_benchmark_before,
        "cudnn_deterministic_exact": torch.backends.cudnn.deterministic
        == cudnn_deterministic_before,
        "bc_MAX_ACTION_COUNT_exact": framework.bc.MAX_ACTION_COUNT
        == bc_max_action_before,
        "cuda_matmul_tf32_exact": torch.backends.cuda.matmul.allow_tf32
        == cuda_matmul_tf32_before,
        "cudnn_tf32_exact": torch.backends.cudnn.allow_tf32 == cudnn_tf32_before,
        "cpu_rng_exact": torch.equal(torch.random.get_rng_state(), cpu_rng_before),
        "cuda_rng_exact": all(
            torch.equal(left, right)
            for left, right in zip(torch.cuda.get_rng_state_all(), cuda_rng_before)
        ),
        "python_rng_exact": random.getstate() == python_rng_before,
        "numpy_rng_exact": all(
            np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
            for left, right in zip(np.random.get_state(), numpy_rng_before)
        ),
    }
    if not all(restore_checks.values()):
        raise ProtocolError(f"profile restore shield failed: {restore_checks}")
    result["restore_checks"] = restore_checks
    result["restore_pass"] = True
    return result


def publish_o_excl_readonly(path: Path, payload: bytes) -> dict[str, Any]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(fd, payload[written:])
            if count <= 0:
                raise ProtocolError("short profile artifact write")
            written += count
        os.fsync(fd)
        observed = os.fstat(fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or int(observed.st_nlink) != 1
            or int(observed.st_size) != len(payload)
        ):
            raise ProtocolError("unsafe profile artifact publication")
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    final = path.lstat()
    digest = sha256_file(path)
    checks = {
        "regular_single_link": stat.S_ISREG(final.st_mode)
        and not stat.S_ISLNK(final.st_mode)
        and int(final.st_nlink) == 1,
        "mode_0444": stat.S_IMODE(final.st_mode) == 0o444,
        "size_exact": int(final.st_size) == len(payload),
        "payload_sha_exact": digest == hashlib.sha256(payload).hexdigest(),
    }
    if not all(checks.values()):
        raise ProtocolError(f"published artifact drift: {checks}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": int(final.st_size),
        "mode_octal": format(stat.S_IMODE(final.st_mode), "04o"),
        "device": int(final.st_dev),
        "inode": int(final.st_ino),
        "nlink": int(final.st_nlink),
        "checks": checks,
    }


def production_run(batch_size: int, keep: int) -> dict[str, Any]:
    import numpy as np
    import torch

    runtime = validate_runtime(require_cuda=True)
    source = source_audit()
    archive_evidence, inventories = archive_inventory()
    cw21, cw21_evidence = load_locked_source(
        CW21, CW21_SHA256, CW21_MODE, "cw22_profile_frozen_cw21"
    )
    cw20, cw20_evidence = cw21.import_cw20()
    cw20_runtime = cw21.runtime_audit(cw20, require_cuda=True)
    cw19, cw19_evidence = cw20.import_cw19()
    cw15, cw15_evidence = cw19.import_locked(
        cw19.CW15, "cw22_profile_frozen_cw15"
    )
    modules = cw15.frozen_modules()

    framework, framework_evidence = load_locked_source(
        FRAMEWORK,
        FRAMEWORK_SHA256,
        FRAMEWORK_MODE,
        "profile_u468_beta1157_train_margins",
    )
    raw_profile, raw_profile_evidence = load_locked_source(
        RAW_PROFILE,
        RAW_PROFILE_SHA256,
        RAW_PROFILE_MODE,
        "cw22_profile_frozen_raw_profile",
    )
    module_checks = {
        "raw_profile_bound_exact_framework_object": raw_profile.framework is framework,
        "framework_path_exact": Path(framework.__file__).resolve() == FRAMEWORK,
        "raw_profile_dataset_paths_exact": dict(raw_profile.DATASETS) == DATASETS,
        "raw_profile_dataset_hashes_exact": dict(raw_profile.EXPECTED_DATA_SHA256)
        == EXPECTED_DATA_SHA256,
        "framework_dataset_paths_exact": dict(framework.DATASETS) == DATASETS,
        "framework_dataset_hashes_exact": dict(framework.EXPECTED_DATA_SHA256)
        == EXPECTED_DATA_SHA256,
        "framework_BC_sha_bound_by_CW15": sha256_file(Path(framework.bc.__file__))
        == cw15.MODULE_SHAS[cw15.BC],
        "framework_PPO_sha_bound_by_CW15": sha256_file(Path(framework.ppo.__file__))
        == cw15.MODULE_SHAS[cw15.PPO],
    }
    if not all(module_checks.values()):
        raise ProtocolError(f"profile module binding drift: {module_checks}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    primary_source, primary_evidence = modules["cw11"].read_regular_bytes(
        cw15.PRIMARY,
        cw15.MODULE_SHAS[cw15.PRIMARY],
        "CW22 profile frozen primary source",
        expected_mode=0o555,
    )
    holder: dict[str, Any] = {}

    def consume(context: Mapping[str, Any]) -> None:
        if holder:
            raise ProtocolError("exact CW11 profile callback called more than once")
        holder["profile"] = profile_callback(
            context,
            cw20,
            cw15,
            framework,
            raw_profile,
            batch_size,
            keep,
        )

    historical = modules["cw11"].run_probe(
        modules["primary"],
        primary_source,
        primary_evidence,
        candidate_consumer=consume,
    )
    archive_evidence_after, inventories_after = archive_inventory()
    archive_stability_checks = {
        label: {
            "sha_exact_before_after": archive_evidence[label]["sha256"]
            == archive_evidence_after[label]["sha256"]
            == EXPECTED_DATA_SHA256[label],
            "identity_exact_before_after": (
                archive_evidence[label]["device"],
                archive_evidence[label]["inode"],
                archive_evidence[label]["bytes"],
            )
            == (
                archive_evidence_after[label]["device"],
                archive_evidence_after[label]["inode"],
                archive_evidence_after[label]["bytes"],
            ),
            "inventory_exact_before_after": inventories[label]
            == inventories_after[label],
        }
        for label in DATASETS
    }
    if not all(
        all(checks.values()) for checks in archive_stability_checks.values()
    ):
        raise ProtocolError(f"archive changed during profile: {archive_stability_checks}")
    historical_checks = {
        "callback_once": "profile" in holder,
        "exact_historical_CW11_status": historical.get("status")
        == "exploratory_33row_specialist_valid_CW11_success",
        "historical_consumer_called": historical.get("second_stage", {})
        .get("decision", {})
        .get("candidate_consumer_called")
        is True,
        "historical_CW11_model_exact": historical.get("second_stage", {})
        .get("decision", {})
        .get("candidate_model_state_sha256_before_CW10_finally_restore")
        == CW11_MODEL_SHA256,
        "historical_CW11_vector_exact": historical.get("second_stage", {})
        .get("decision", {})
        .get("terminal_cumulative_float64_le_sha256")
        == CW11_VECTOR_SHA256,
        "historical_CW11_ledger_exact": historical.get("second_stage", {})
        .get("active_pair_contract", {})
        .get("final_canonical_ledger_sha256")
        == CW11_LEDGER_SHA256,
        "outer_historical_restore_pass": historical.get("final_integrity", {}).get(
            "pass"
        )
        is True,
    }
    if not all(historical_checks.values()):
        raise ProtocolError(f"historical exact CW11 replay drift: {historical_checks}")

    endpoint = holder["profile"]
    return {
        "schema_version": SCHEMA,
        "status": endpoint["status"],
        "seed": SEED,
        "base": {
            "kind": "exact_CW11_historical_RAM_replay_after_general_BC_plus_PPO",
            "model_state_sha256": CW11_MODEL_SHA256,
            "actor_float32_le_sha256": CW11_ACTOR_FLOAT32_LE_SHA256,
            "terminal_vector_float64_le_sha256": CW11_VECTOR_SHA256,
            "active_ledger_sha256": CW11_LEDGER_SHA256,
            "raw_parent_model_state_sha256": RAW_MODEL_SHA256,
            "raw_parent_nonactor_sha256": RAW_NONACTOR_SHA256,
            "materialized_eval_only_CW11_opened": False,
            "resume_forbidden_respected": True,
        },
        "split": "train",
        "validation_opened_for_profile": False,
        "end_to_end_validation_free": False,
        "test_opened": False,
        "selection": {
            "near_wrong": (
                "ordered-wrong rows by descending exact-CW11 decision margin, "
                "then line SHA"
            ),
            "fragile_correct": (
                "ordered-correct rows by ascending exact-CW11 decision margin, "
                "then line SHA"
            ),
            "decision_margin": (
                "minimum expert-vs-best-alternative margin over ordered selection "
                "and flexible count"
            ),
            "keep_per_list": keep,
            "batch_size": batch_size,
        },
        "historical_exact_CW11_replay": {
            "checks": historical_checks,
            "pass": True,
            "historical_replay_count": 1,
            "consumed_evidence_only": True,
            "historical_specialist_valid_rows_replayed": 4,
            "validation_replayed_for_historical_reconstruction": True,
            "specialist_valid_consumed_for_optimization": True,
            "promotion_evidence": False,
            "changed_candidate_created": False,
            "changed_candidate_official_or_validation_evaluation_count": 0,
        },
        "endpoint": endpoint,
        "inputs": {
            "archives": archive_evidence,
            "archive_inventory": inventories,
            "archive_stability_checks": archive_stability_checks,
            "frozen_CW21": cw21_evidence,
            "frozen_CW20": cw20_evidence,
            "frozen_CW19": cw19_evidence,
            "frozen_CW15": cw15_evidence,
            "frozen_primary": primary_evidence,
            "profile_framework": framework_evidence,
            "raw_profile_capacity_helper": raw_profile_evidence,
            "module_binding_checks": module_checks,
        },
        "audit": {
            "source": source,
            "runtime": runtime,
            "CW20_runtime": cw20_runtime,
        },
        "official_unique_changed_candidate_count_consumed": 0,
        "submission_performed": False,
        "package_upload_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--keep", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.keep < 1:
        raise ProtocolError("batch-size and keep must be positive")
    if args.mode == "static":
        if args.device != "cpu" or args.output is not None:
            raise ProtocolError("static mode requires --device cpu and forbids --output")
        result = {
            "schema_version": SCHEMA,
            "status": "static_audit_passed_zero_archive_members_opened",
            "source": source_audit(),
            "runtime": validate_runtime(require_cuda=False),
            "inputs": {
                "CW21": regular_evidence(CW21, CW21_SHA256, "CW21", CW21_MODE),
                "raw_profile": regular_evidence(
                    RAW_PROFILE,
                    RAW_PROFILE_SHA256,
                    "raw profile",
                    RAW_PROFILE_MODE,
                ),
                "framework": regular_evidence(
                    FRAMEWORK,
                    FRAMEWORK_SHA256,
                    "profile framework",
                    FRAMEWORK_MODE,
                ),
            },
            "archive_member_opens": 0,
            "writes_performed": 0,
            "validation_opened": False,
            "submission_performed": False,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return

    if args.device != "cuda" or args.output is None:
        raise ProtocolError("run mode requires --device cuda and --output")
    output = args.output.resolve()
    if output.parent != ROOT / "artifacts":
        raise ProtocolError("output must be a direct child of artifacts/")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result = production_run(args.batch_size, args.keep)
    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    artifact = publish_o_excl_readonly(output, payload)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": result["status"],
                "output": artifact,
                "base_model_state_sha256": CW11_MODEL_SHA256,
                "rows": {
                    label: panel["rows"]
                    for label, panel in result["endpoint"]["profiles"].items()
                },
                "ordered_wrong": {
                    label: panel["ordered_wrong"]
                    for label, panel in result["endpoint"]["profiles"].items()
                },
                "balanced_recipe_capacity_pass": result["endpoint"][
                    "balanced_recipe_capacity"
                ]["all_sources_have_capacity"],
                "targeted_CW22_capacity_pass": result["endpoint"][
                    "targeted_CW22_capacity"
                ]["pass"],
                "validation_opened_for_profile": False,
                "official_unique_changed_candidate_count_consumed": 0,
                "submission_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
