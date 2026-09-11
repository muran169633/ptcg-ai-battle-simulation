#!/usr/bin/env python3
"""Read-only 23-row cutting-plane repair seeded by the exact C0 candidate.

The frozen cutting-plane v2 runner first reconstructs C0 (its iteration-11
candidate) and restores raw U468.  This runner then rebuilds the original 19
rows plus four full-train CW witnesses in one native-BF16 batch, remaps C0's
20 active constraints by full row identity, refreshes only old guard floors
against the same-process 23-row raw margins, appends four raw-margin guards,
and performs at most twelve more minimum-L2 trust-region corrections.

There is no validation access, training optimizer, backward call, sweep,
checkpoint/result write, materialization, network access, upload, or
submission.  Output is one JSON document on stdout only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-fulltrain-cw4-cuttingplane-v1"
SEED = 202608205

CUTTING = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py"
CUTTING_SHA256 = "c2866ba8b00eba6b424197a520419a5717401335cc47202b4fcc711611f67503"
FULLTRAIN = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v2.py"
FULLTRAIN_SHA256 = "e6bce38044f8230662cb6d61d560baa529cf507070045db53c4196bce89794b6"
FROZEN_MODE = 0o555

EXPECTED_RAW_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_RAW_NONACTOR_SHA256 = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)
EXPECTED_C0_MODEL_STATE_SHA256 = (
    "c79ce4a258f673248c7abab6e7ccffa10bd9dd4250f43d0e841fd54e44586c82"
)
EXPECTED_C0_SUCCESS_ITERATION = 11
EXPECTED_C0_CUMULATIVE_L2 = 0.006294165313358753
EXPECTED_C0_CUMULATIVE_SHA256 = (
    "47a001d201db1bba7255ce0b8ccb993cce1649f78f14826782cc0b37e0b31ffc"
)
EXPECTED_C0_ACTIVE_COUNT = 20
EXPECTED_C0_ACTIVE_LEDGER_SHA256 = (
    "8643f1c6f88fdae789fb9b9d505a0156a2971d065b1cd00696c54b562d32e699"
)
EXPECTED_C0_GUARD_ACTIVE_COUNT = 15
EXPECTED_SUCCESS_CONTINUATION_ITERATION = 5
EXPECTED_TERMINAL_CUMULATIVE_L2 = 0.007014818833558696
EXPECTED_TERMINAL_CUMULATIVE_SHA256 = (
    "f2f3d554cf1cffa8f81c7b637ca4a2e531e97777188446b2dc999592800e4b9e"
)
EXPECTED_TERMINAL_MODEL_STATE_SHA256 = (
    "f1130d1701619609737c5a277502a41367c05a261115f5daaaaccf967822fd6b"
)
EXPECTED_TERMINAL_ACTIVE_LEDGER_SHA256 = (
    "ad48597cef3e0d43a6e1e2dd34dc00d585e1bd1b1ef904972f8436aa0455508e"
)

ORIGINAL_ROW_COUNT = 19
EXPANDED_ROW_COUNT = 23
TARGET_ROW_COUNT = 5
EXPANDED_GUARD_ROW_COUNT = 18
EXPECTED_TARGET_OBLIGATIONS = 20
EXPECTED_ORIGINAL_GUARD_OBLIGATIONS = 41
EXPECTED_NEW_GUARD_OBLIGATIONS = 16
EXPECTED_C0_FALSE_OBLIGATIONS = 13
EXPECTED_EXPANDED_GUARD_OBLIGATIONS = 57
EXPECTED_INITIAL_ACTIVE_COUNT = 24
MAX_ITER = 12
STEP_L2_CAP = 0.001
MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)

NEW_GUARDS = (
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 4545,
        "line_sha256": "f5c29fe63df23a40aff98d58447af577ba7244af2c762a288555041dd7fdfac4",
        "expected_context": 0,
        "expected_expert_order": [9],
        "expected_option_count": 13,
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": list(MAIN_METRICS),
        "positive_option": 9,
        "negative_option": 10,
        "formal_raw_order": [9],
        "formal_c0_order": [10],
    },
    {
        "role": "guard",
        "panel": "flg",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 5063,
        "line_sha256": "9417c4667da3af128f684cf5b847952b47686a329af28a94cbf01615f3de8644",
        "expected_context": 5,
        "expected_expert_order": [1, 3],
        "expected_option_count": 4,
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": ["ordered_exact"],
        "positive_option": 1,
        "negative_option": 3,
        "formal_raw_order": [1, 3],
        "formal_c0_order": [3, 1],
    },
    {
        "role": "guard",
        "panel": "pokemonfan",
        "member": "train/part-00000.jsonl",
        "line_index_zero_based": 5092,
        "line_sha256": "989b079b931a1f256b4ddad8e7f1d42986f3f98464f993a3afe994b5cca56d13",
        "expected_context": 0,
        "expected_expert_order": [0],
        "expected_option_count": 6,
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": list(MAIN_METRICS),
        "positive_option": 0,
        "negative_option": 1,
        "formal_raw_order": [0],
        "formal_c0_order": [1],
    },
    {
        "role": "guard",
        "panel": "core5",
        "member": "train/part-00007.jsonl",
        "line_index_zero_based": 209,
        "line_sha256": "010dca17bea03dab885587766989072a358f8df36e6485cc358321dd83f6d483",
        "expected_context": 0,
        "expected_expert_order": [5],
        "expected_option_count": 9,
        "metrics_union": list(MAIN_METRICS),
        "expected_c0_false_metrics": list(MAIN_METRICS),
        "positive_option": 5,
        "negative_option": 4,
        "formal_raw_order": [5],
        "formal_c0_order": [4],
    },
)


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


def read_regular_bytes(
    path: Path,
    expected_sha256: str,
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
    visible = os.lstat(path)
    identity = (after.st_dev, after.st_ino, after.st_size)
    if (
        (before.st_dev, before.st_ino, before.st_size) != identity
        or stat.S_ISLNK(visible.st_mode)
        or not stat.S_ISREG(visible.st_mode)
        or visible.st_nlink != 1
        or (visible.st_dev, visible.st_ino, visible.st_size) != identity
    ):
        raise RuntimeError(f"{label} changed during held-fd read")
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    mode = stat.S_IMODE(after.st_mode)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(mode),
        "single_link_regular_held_fd_identity_exact": True,
    }


def import_frozen(
    path: Path, digest: str, module_name: str
) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, digest, module_name, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct frozen import {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_attributes = {
        "backward",
        "step",
        "save",
        "savez",
        "write",
        "write_bytes",
        "write_text",
        "touch",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "copy_",
        "add_",
    }
    forbidden_names = {"open", "exec", "eval", "compile", "DataLoader"}
    forbidden_os_flags = {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
    forbidden_import_roots = {"requests", "urllib", "subprocess"}
    attribute_hits: list[tuple[int, str]] = []
    name_hits: list[tuple[int, str]] = []
    os_flag_hits: list[tuple[int, str]] = []
    import_hits: list[tuple[int, str]] = []
    print_sites: list[int] = []
    os_open_sites: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append((node.lineno, name))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in forbidden_os_flags
        ):
            os_flag_hits.append((node.lineno, node.attr))
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in forbidden_attributes:
                attribute_hits.append((node.lineno, function.attr))
            if (
                isinstance(function.value, ast.Name)
                and function.value.id == "os"
                and function.attr == "open"
            ):
                os_open_sites.append(node.lineno)
        elif isinstance(function, ast.Name):
            if function.id in forbidden_names:
                name_hits.append((node.lineno, function.id))
            if function.id == "print":
                print_sites.append(node.lineno)
    if attribute_hits or name_hits or os_flag_hits or import_hits:
        raise RuntimeError(
            "zero-write static audit failed: "
            f"{attribute_hits=} {name_hits=} {os_flag_hits=} {import_hits=}"
        )
    if len(print_sites) != 1 or len(os_open_sites) != 1:
        raise RuntimeError("static stdout/read call-site cardinality drift")
    return {
        "ast_parse": True,
        "forbidden_attribute_call_sites": attribute_hits,
        "forbidden_name_call_sites": name_hits,
        "forbidden_os_write_flags": os_flag_hits,
        "forbidden_network_imports": import_hits,
        "stdout_print_call_sites": print_sites,
        "os_open_read_only_call_sites": os_open_sites,
        "controlled_dependency_globals": 2,
        "checkpoint_or_result_write_call_sites": 0,
        "no_optimizer_backward_step_save_write_network_or_submission": True,
    }


def interface_audit(
    cutting: ModuleType,
    fulltrain: ModuleType,
    geometry: ModuleType,
    ram: ModuleType,
    formal: ModuleType,
) -> dict[str, Any]:
    cutting_parameters = inspect.signature(cutting.run_cuttingplane).parameters
    if (
        "candidate_consumer" not in cutting_parameters
        or cutting_parameters["candidate_consumer"].default is not None
    ):
        raise RuntimeError("cutting v2 candidate-consumer interface drift")
    if (
        cutting.STEP_L2_CAP != STEP_L2_CAP
        or cutting.MAX_ITER != MAX_ITER
        or tuple(cutting.MAIN_METRICS) != MAIN_METRICS
        or geometry.EXPECTED_ROW_COUNT != ORIGINAL_ROW_COUNT
        or ram.EXPECTED_DIRECTION_SHA256 != cutting.EXPECTED_DIRECTION_SHA256
        or fulltrain.EXPECTED_SELECTED_SUCCESS_ITERATION
        != EXPECTED_C0_SUCCESS_ITERATION
        or fulltrain.EXPECTED_TERMINAL_CUMULATIVE_SHA256
        != EXPECTED_C0_CUMULATIVE_SHA256
        or fulltrain.EXPECTED_FINAL_ACTIVE_PAIR_LEDGER_SHA256
        != EXPECTED_C0_ACTIVE_LEDGER_SHA256
        or Path(fulltrain.FORMAL) != Path(formal.SCRIPT)
    ):
        raise RuntimeError("frozen dependency constants drift")
    run_parameters = inspect.signature(run_cuttingplane).parameters
    if (
        "candidate_consumer" not in run_parameters
        or run_parameters["candidate_consumer"].default is not None
    ):
        raise RuntimeError("expanded candidate-consumer interface drift")
    return {
        "cutting_v2_candidate_consumer_default_none": True,
        "expanded_candidate_consumer_default_none": True,
        "step_l2_cap_exact": STEP_L2_CAP,
        "max_additional_iterations": MAX_ITER,
        "original_rows": ORIGINAL_ROW_COUNT,
        "expanded_rows": EXPANDED_ROW_COUNT,
        "formal_evaluator_bound_through_fulltrain_v2": True,
    }


def identity_key(value: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(value["panel"]),
        str(value["member"]),
        int(value["line_index_zero_based"]),
        str(value["line_sha256"]),
    )


def identity_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": str(value["panel"]),
        "member": str(value["member"]),
        "line_index_zero_based": int(value["line_index_zero_based"]),
        "line_sha256": str(value["line_sha256"]),
    }


def load_original_descriptors(
    geometry: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, evidence = geometry.read_regular_bytes(
        geometry.FORMAL_RESULT,
        geometry.FORMAL_RESULT_SHA256,
        "formal v3 ray result",
    )
    formal = geometry.strict_json_bytes(payload, "formal v3 ray result")
    if formal.get("status") != "closed_no_candidate":
        raise RuntimeError("formal v3 result status drift")
    descriptors = [*geometry.extract_targets(formal), *geometry.extract_guards(formal)]
    if (
        len(descriptors) != ORIGINAL_ROW_COUNT
        or sum(item["role"] == "target" for item in descriptors) != TARGET_ROW_COUNT
        or sum(item["role"] == "guard" for item in descriptors)
        != ORIGINAL_ROW_COUNT - TARGET_ROW_COUNT
        or len({identity_key(item) for item in descriptors}) != ORIGINAL_ROW_COUNT
    ):
        raise RuntimeError("original descriptor contract drift")
    return descriptors, evidence


def capture_c0(context: Mapping[str, Any], holder: dict[str, Any]) -> None:
    if holder:
        raise RuntimeError("C0 consumer called more than once")
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
        "selected_row_gate",
        "active_pair_ledger",
    }
    if not required.issubset(context):
        raise RuntimeError("C0 consumer context schema drift")
    helper = context["helper"]
    model = context["model"]
    live_sha = helper.model_state_sha256(model.state_dict())
    if live_sha != EXPECTED_C0_MODEL_STATE_SHA256:
        raise RuntimeError(f"C0 live model SHA drift: {live_sha}")
    cumulative = context["terminal_cumulative_float64"].copy()
    holder.update(
        {
            "helper": helper,
            "checkpoint": context["checkpoint"],
            "model_config": dict(context["model_config"]),
            "raw_model_state_sha256": str(context["raw_model_state_sha256"]),
            "raw_nonactor_sha256": str(context["raw_nonactor_sha256"]),
            "cumulative": cumulative,
            "cumulative_sha256": str(
                context["terminal_cumulative_float64_le_sha256"]
            ),
            "success_iteration": int(context["success_iteration"]),
            "selected_row_gate": context["selected_row_gate"],
            "active_pair_ledger": context["active_pair_ledger"],
            "live_c0_model_state_sha256": live_sha,
        }
    )


def reconstruct_c0(
    cutting: ModuleType,
    cutting_source: bytes,
    cutting_static: Mapping[str, Any],
    geometry: ModuleType,
    geometry_evidence: Mapping[str, Any],
    ram: ModuleType,
    ram_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    holder: dict[str, Any] = {}

    def consumer(context: Mapping[str, Any]) -> None:
        capture_c0(context, holder)

    result = cutting.run_cuttingplane(
        cutting_source,
        cutting_static,
        geometry,
        geometry_evidence,
        ram,
        ram_evidence,
        candidate_consumer=consumer,
    )
    checks = {
        "status_success": result.get("status")
        == "selected_row_adaptive_optimization_success",
        "consumer_called_once": bool(holder),
        "success_iteration_exact_11": int(
            result.get("decision", {}).get("success_iteration", -1)
        )
        == EXPECTED_C0_SUCCESS_ITERATION,
        "cumulative_l2_exact": math.isclose(
            float(result.get("decision", {}).get("terminal_cumulative_l2", -1.0)),
            EXPECTED_C0_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "cumulative_sha_exact": result.get("decision", {}).get(
            "terminal_cumulative_float64_le_sha256"
        )
        == EXPECTED_C0_CUMULATIVE_SHA256,
        "active_count_exact_20": int(
            result.get("active_pair_contract", {}).get("final_count", -1)
        )
        == EXPECTED_C0_ACTIVE_COUNT,
        "active_ledger_sha_exact": result.get("active_pair_contract", {}).get(
            "final_canonical_ledger_sha256"
        )
        == EXPECTED_C0_ACTIVE_LEDGER_SHA256,
        "c0_model_sha_exact": holder.get("live_c0_model_state_sha256")
        == EXPECTED_C0_MODEL_STATE_SHA256,
        "raw_model_sha_exact": holder.get("raw_model_state_sha256")
        == EXPECTED_RAW_MODEL_STATE_SHA256,
        "raw_nonactor_sha_exact": holder.get("raw_nonactor_sha256")
        == EXPECTED_RAW_NONACTOR_SHA256,
        "v2_finally_raw_restore_pass": bool(
            result.get("final_integrity", {}).get("pass")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"C0 exact reconstruction gate failed: {checks}")
    ledger = holder["active_pair_ledger"]
    if (
        len(ledger) != EXPECTED_C0_ACTIVE_COUNT
        or sha256_bytes(canonical_json(ledger)) != EXPECTED_C0_ACTIVE_LEDGER_SHA256
    ):
        raise RuntimeError("consumer C0 active ledger count/SHA drift")
    return holder, {"checks": checks, "frozen_v2_result": result}


def load_expanded_rows(
    geometry: ModuleType,
    helper: ModuleType,
    descriptors: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    previous = geometry.EXPECTED_ROW_COUNT
    restored = False
    try:
        geometry.EXPECTED_ROW_COUNT = EXPANDED_ROW_COUNT
        rows, evidence = geometry.load_selected_rows(
            helper, descriptors, model_config
        )
    finally:
        geometry.EXPECTED_ROW_COUNT = previous
        restored = geometry.EXPECTED_ROW_COUNT == ORIGINAL_ROW_COUNT
    if not restored or previous != ORIGINAL_ROW_COUNT:
        raise RuntimeError("geometry EXPECTED_ROW_COUNT global restoration failed")
    if len(rows) != EXPANDED_ROW_COUNT:
        raise RuntimeError("expanded selected-row cardinality drift")
    metadata_checks = []
    for offset, guard in enumerate(NEW_GUARDS, start=ORIGINAL_ROW_COUNT):
        metadata = rows[offset]["raw_metadata"]
        checks = {
            "identity_position_exact": identity_key(descriptors[offset])
            == identity_key(guard),
            "context_exact": int(metadata["context"])
            == int(guard["expected_context"]),
            "expert_order_exact": list(metadata["expert_order"])
            == list(guard["expected_expert_order"]),
            "option_count_exact": int(metadata["option_count"])
            == int(guard["expected_option_count"]),
            "pair_options_in_bounds": 0
            <= int(guard["positive_option"])
            < int(metadata["option_count"])
            and 0
            <= int(guard["negative_option"])
            < int(metadata["option_count"]),
            "non_context34": int(metadata["context"]) != 34,
        }
        if not all(checks.values()):
            raise RuntimeError(f"new guard row metadata drift: {checks}")
        metadata_checks.append(
            {"identity": identity_record(guard), "checks": checks}
        )
    return rows, evidence, {
        "temporary_expected_row_count": EXPANDED_ROW_COUNT,
        "geometry_global_restored_to_19": restored,
        "new_guard_metadata": metadata_checks,
    }


def canonical_to_internal_and_remap(
    cutting: ModuleType,
    canonical_ledger: Sequence[Mapping[str, Any]],
    original_descriptors: Sequence[Mapping[str, Any]],
    expanded_descriptors: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        len(canonical_ledger) != EXPECTED_C0_ACTIVE_COUNT
        or sha256_bytes(canonical_json(canonical_ledger))
        != EXPECTED_C0_ACTIVE_LEDGER_SHA256
    ):
        raise RuntimeError("C0 canonical ledger pre-remap lock failed")
    expanded_index = {
        identity_key(descriptor): index
        for index, descriptor in enumerate(expanded_descriptors)
    }
    if len(expanded_index) != EXPANDED_ROW_COUNT:
        raise RuntimeError("expanded descriptor identities are not unique")
    active: list[dict[str, Any]] = []
    remap: list[dict[str, Any]] = []
    guard_threshold_refresh: list[dict[str, Any]] = []
    for record in canonical_ledger:
        old_key = tuple(int(value) for value in record["key"])
        old_row, positive, negative = old_key
        if not 0 <= old_row < ORIGINAL_ROW_COUNT:
            raise RuntimeError("C0 active pair old row index out of range")
        identity = identity_key(record["identity"])
        descriptor = original_descriptors[old_row]
        if identity != identity_key(descriptor):
            raise RuntimeError("C0 active pair row/identity mismatch")
        new_row = expanded_index.get(identity)
        if new_row is None:
            raise RuntimeError("C0 active pair identity absent from expanded rows")
        if new_row != old_row:
            raise RuntimeError("original 0..18 identity-preserving remap drift")
        option_count = int(selected_rows[new_row]["raw_metadata"]["option_count"])
        if not (0 <= positive < option_count and 0 <= negative < option_count):
            raise RuntimeError("C0 active pair option out of expanded-row bounds")
        role = str(record["role"])
        if role != str(descriptor["role"]):
            raise RuntimeError("C0 active pair role/descriptor mismatch")
        threshold = float(record["threshold"])
        if not math.isfinite(threshold):
            raise RuntimeError("C0 active threshold is nonfinite")
        threshold_sources = [str(value) for value in record["threshold_sources"]]
        threshold_history = [dict(value) for value in record["threshold_history"]]
        threshold_source = str(record["threshold_source"])
        pair = {
            "row_index": new_row,
            "identity": identity_record(record["identity"]),
            "role": role,
            "positive_option": positive,
            "negative_option": negative,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "threshold_sources": threshold_sources,
            "threshold_history": threshold_history,
            "origins": [str(value) for value in record["origins"]],
            "constructions": [str(value) for value in record["construction"]],
            "created_iteration": int(record["created_iteration"]),
        }
        raw23_margin = cutting.pair_margin(raw_snapshot, pair)
        if not math.isfinite(raw23_margin):
            raise RuntimeError("old active pair raw23 margin is nonfinite")
        if role == "guard":
            if raw23_margin < 0.0:
                raise RuntimeError("old guard raw23 pair margin is negative")
            if raw23_margin != threshold:
                raise RuntimeError(
                    "old guard canonical threshold differs from same-process "
                    "raw 23-row pair margin"
                )
            guard_threshold_refresh.append(
                {
                    "key_before": list(old_key),
                    "key_after": [new_row, positive, negative],
                    "canonical_threshold": threshold,
                    "raw23_margin": raw23_margin,
                    "raw23_minus_canonical": raw23_margin - threshold,
                    "exact": raw23_margin == threshold,
                    "max_rule_would_raise": raw23_margin > threshold,
                }
            )
        active.append(pair)
        remap.append(
            {
                "identity": identity_record(record["identity"]),
                "old_key": list(old_key),
                "new_key": [new_row, positive, negative],
                "identity_preserved": True,
                "role": role,
                "descriptor_role": str(descriptor["role"]),
                "role_exact": role == str(descriptor["role"]),
            }
        )
    if (
        len(active) != EXPECTED_C0_ACTIVE_COUNT
        or len({cutting.pair_key(pair) for pair in active})
        != EXPECTED_C0_ACTIVE_COUNT
        or len(guard_threshold_refresh) != EXPECTED_C0_GUARD_ACTIVE_COUNT
    ):
        raise RuntimeError("C0 internal/remap/guard refresh cardinality drift")
    target_thresholds_preserved = all(
        float(pair["threshold"])
        == float(canonical_ledger[index]["threshold"])
        for index, pair in enumerate(active)
        if pair["role"] == "target"
    )
    if not target_thresholds_preserved:
        raise RuntimeError("C0 target threshold changed during remap")
    return active, {
        "pre_remap_count": len(canonical_ledger),
        "pre_remap_canonical_ledger_sha256": sha256_bytes(
            canonical_json(canonical_ledger)
        ),
        "full_identity_remap": remap,
        "active20_full_identity_role_records_exact_20": (
            len(remap) == EXPECTED_C0_ACTIVE_COUNT
            and all(item["identity_preserved"] and item["role_exact"] for item in remap)
        ),
        "old_guard_same_process_raw23_max_refresh": guard_threshold_refresh,
        "old_guard_count_exact_15": len(guard_threshold_refresh)
        == EXPECTED_C0_GUARD_ACTIVE_COUNT,
        "old_guard_raw23_exact_canonical_15_of_15": all(
            item["exact"] for item in guard_threshold_refresh
        ),
        "old_guard_max_rule_raise_count": sum(
            item["max_rule_would_raise"] for item in guard_threshold_refresh
        ),
        "target_absolute_thresholds_preserved": target_thresholds_preserved,
        "original_indices_identity_preserved": all(
            item["old_key"] == item["new_key"] for item in remap
        ),
    }


def append_new_guards(
    cutting: ModuleType,
    active_pairs: list[dict[str, Any]],
    raw_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    added: list[dict[str, Any]] = []
    existing = {cutting.pair_key(pair) for pair in active_pairs}
    for offset, descriptor in enumerate(NEW_GUARDS, start=ORIGINAL_ROW_COUNT):
        pair = {
            "row_index": offset,
            "identity": identity_record(descriptor),
            "role": "guard",
            "positive_option": int(descriptor["positive_option"]),
            "negative_option": int(descriptor["negative_option"]),
            "threshold": 0.0,
            "threshold_source": "same_process_raw_fulltrain_cw_pair_margin",
            "threshold_sources": ["same_process_raw_fulltrain_cw_pair_margin"],
            "threshold_history": [],
            "origins": ["fulltrain_CW_guard"],
            "constructions": ["specified_raw_safe_vs_C0_threat_pair"],
            "created_iteration": EXPECTED_C0_SUCCESS_ITERATION,
        }
        threshold = cutting.pair_margin(raw_snapshot, pair)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise RuntimeError("new guard same-process raw23 margin invalid")
        pair["threshold"] = threshold
        pair["threshold_history"] = [
            {
                "iteration": EXPECTED_C0_SUCCESS_ITERATION,
                "source": pair["threshold_source"],
                "observed": threshold,
                "retained_max": threshold,
            }
        ]
        key = cutting.pair_key(pair)
        if key in existing:
            raise RuntimeError("new guard pair duplicates remapped C0 pair")
        existing.add(key)
        active_pairs.append(pair)
        added.append(dict(pair))
    if len(active_pairs) != EXPECTED_INITIAL_ACTIVE_COUNT:
        raise RuntimeError("expanded initial active-pair count drift")
    return added


def expanded_gate_snapshot(
    cutting: ModuleType,
    descriptors: Sequence[Mapping[str, Any]],
    active_pairs: Sequence[Mapping[str, Any]],
    raw_snapshot: Mapping[str, Any],
    current: Mapping[str, Any],
    nonactor_exact: bool,
    torch: Any,
    patch_audit: dict[str, Any],
) -> dict[str, Any]:
    previous = cutting.EXPECTED_SELECTED_ROW_COUNT
    restored = False
    try:
        cutting.EXPECTED_SELECTED_ROW_COUNT = EXPANDED_ROW_COUNT
        result = cutting.gate_snapshot(
            descriptors,
            active_pairs,
            raw_snapshot,
            current,
            nonactor_exact,
            torch,
        )
    finally:
        cutting.EXPECTED_SELECTED_ROW_COUNT = previous
        restored = cutting.EXPECTED_SELECTED_ROW_COUNT == ORIGINAL_ROW_COUNT
    patch_audit["gate_call_count"] = int(patch_audit.get("gate_call_count", 0)) + 1
    patch_audit["cutting_global_restored_every_call"] = bool(
        patch_audit.get("cutting_global_restored_every_call", True)
    ) and restored
    if not restored or previous != ORIGINAL_ROW_COUNT:
        raise RuntimeError("cutting EXPECTED_SELECTED_ROW_COUNT restoration failed")
    exact_legacy = result.pop("selected_official_row_count_exact_19")
    result["selected_official_row_count_exact_23"] = bool(exact_legacy)
    hard = {
        "official_rows_exact_23": int(result["selected_official_row_count"])
        == EXPANDED_ROW_COUNT
        and bool(result["selected_official_row_count_exact_23"]),
        "target_obligations_exact_20": int(
            result["target_5x4_obligation_count"]
        )
        == EXPECTED_TARGET_OBLIGATIONS
        and int(result["target_5x4_obligation_unique_count"])
        == EXPECTED_TARGET_OBLIGATIONS,
        "guard_obligations_exact_57": int(result["guard_obligation_count"])
        == EXPECTED_EXPANDED_GUARD_OBLIGATIONS,
        "active_pair_count_at_least_initial_24": int(result["active_pair_count"])
        >= EXPECTED_INITIAL_ACTIVE_COUNT,
        "active_pair_keys_unique": len(
            {tuple(item["key"]) for item in result["active_pair_gates"]}
        )
        == int(result["active_pair_count"]),
        "count_value_nonactor_exact": bool(
            result["count_logits_native_exact_raw"]
            and result["value_logits_native_exact_raw"]
            and result["nonactor_state_exact_raw"]
        ),
    }
    if not all(hard.values()):
        raise RuntimeError(f"expanded gate structural contract failed: {hard}")
    result["expanded_structural_checks"] = hard
    return result


def new_guard_transition_audit(
    raw_snapshot: Mapping[str, Any],
    c0_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for row_index, descriptor in enumerate(NEW_GUARDS, start=ORIGINAL_ROW_COUNT):
        raw_row = raw_snapshot["official_rows"][row_index]
        c0_row = c0_snapshot["official_rows"][row_index]
        metrics = [str(value) for value in descriptor["metrics_union"]]
        raw_flags = {metric: raw_row["flags"][metric] for metric in metrics}
        c0_flags = {metric: c0_row["flags"][metric] for metric in metrics}
        observed_c0_false = {
            metric for metric, value in c0_flags.items() if value is False
        }
        expected_c0_false = set(descriptor["expected_c0_false_metrics"])
        raw_order = [int(value) for value in raw_row["predicted_order"]]
        c0_order = [int(value) for value in c0_row["predicted_order"]]
        first_difference = next(
            (
                (raw_value, c0_value)
                for raw_value, c0_value in zip(raw_order, c0_order)
                if raw_value != c0_value
            ),
            None,
        )
        checks = {
            "raw_order_exact": raw_order
            == list(descriptor["formal_raw_order"]),
            "c0_order_exact": c0_order
            == list(descriptor["formal_c0_order"]),
            "frozen_pair_exact_raw_vs_c0_first_difference": first_difference
            == (
                int(descriptor["positive_option"]),
                int(descriptor["negative_option"]),
            ),
            "all_guard_metrics_raw_true": all(value is True for value in raw_flags.values()),
            "c0_false_metric_set_exact": observed_c0_false
            == expected_c0_false,
        }
        if not all(checks.values()):
            raise RuntimeError(f"new guard frozen C0 CW transition drift: {checks}")
        records.append(
            {
                "row_index": row_index,
                "identity": identity_record(descriptor),
                "metrics_union": metrics,
                "raw_flags": raw_flags,
                "c0_flags": c0_flags,
                "raw_order": raw_row["predicted_order"],
                "c0_order": c0_row["predicted_order"],
                "checks": checks,
            }
        )
    return records


def run_cuttingplane(
    source: bytes,
    static: Mapping[str, Any],
    cutting: ModuleType,
    cutting_evidence: Mapping[str, Any],
    fulltrain: ModuleType,
    fulltrain_evidence: Mapping[str, Any],
    geometry: ModuleType,
    geometry_evidence: Mapping[str, Any],
    ram: ModuleType,
    ram_evidence: Mapping[str, Any],
    formal: ModuleType,
    formal_evidence: Mapping[str, Any],
    candidate_consumer: Any | None = None,
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    design, _, design_evidence = formal.verify_frozen_design()
    if (
        tuple(design.DATASETS) != tuple(fulltrain.PANEL_ORDER)
        or any(
            Path(design.DATASETS[panel]) != Path(geometry.DATASETS[panel])
            or str(design.DATA_SHA256[panel]) != str(geometry.DATA_SHA256[panel])
            for panel in fulltrain.PANEL_ORDER
        )
    ):
        raise RuntimeError("formal/geometry train archive binding drift")

    cutting_source, _ = cutting.read_regular_bytes(
        cutting.SCRIPT,
        CUTTING_SHA256,
        "expanded runner cutting v2",
        expected_mode=FROZEN_MODE,
    )
    cutting_static = cutting.static_audit(cutting_source)
    c0, c0_audit = reconstruct_c0(
        cutting,
        cutting_source,
        cutting_static,
        geometry,
        geometry_evidence,
        ram,
        ram_evidence,
    )
    original_descriptors, formal_result_evidence = load_original_descriptors(geometry)
    descriptors = [*original_descriptors, *[dict(value) for value in NEW_GUARDS]]
    if (
        len(descriptors) != EXPANDED_ROW_COUNT
        or len({identity_key(value) for value in descriptors}) != EXPANDED_ROW_COUNT
        or sum(value["role"] == "target" for value in descriptors) != TARGET_ROW_COUNT
        or sum(value["role"] == "guard" for value in descriptors)
        != EXPANDED_GUARD_ROW_COUNT
        or sum(len(value["metrics_union"]) for value in descriptors if value["role"] == "guard")
        != EXPECTED_EXPANDED_GUARD_OBLIGATIONS
    ):
        raise RuntimeError("expanded descriptor/obligation contract drift")

    helper = c0["helper"]
    torch = helper.torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("requires CUDA with native BF16 support")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")

    checkpoint = c0["checkpoint"]
    model, model_config, kind = helper.instantiate_checkpoint(checkpoint, device)
    if kind != "ppo" or model_config != c0["model_config"]:
        raise RuntimeError("fresh raw model/config differs from C0 reconstruction")
    raw_state_sha = helper.model_state_sha256(model.state_dict())
    if raw_state_sha != EXPECTED_RAW_MODEL_STATE_SHA256:
        raise RuntimeError("fresh raw U468 model-state SHA drift")
    parameters = geometry.configure_actor6(model)
    raw_actor = [parameter.detach().clone() for parameter in parameters]
    nonactor_names = sorted(set(model.state_dict()) - set(geometry.ACTOR6_NAMES))
    raw_nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    if raw_nonactor_sha != EXPECTED_RAW_NONACTOR_SHA256:
        raise RuntimeError("fresh raw nonactor SHA drift")

    rows, archive_evidence, loader_patch_audit = load_expanded_rows(
        geometry, helper, descriptors, model_config
    )
    cpu_batch = helper.evaluator.collate_ordered(
        [item["features"] for item in rows],
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    model.eval()
    raw_snapshot = ram.snapshot_forward(helper, model, batch, cpu_batch, device)

    active_pairs, remap_audit = canonical_to_internal_and_remap(
        cutting,
        c0["active_pair_ledger"],
        original_descriptors,
        descriptors,
        rows,
        raw_snapshot,
    )
    original_active_thresholds = {
        cutting.pair_key(pair): float(pair["threshold"])
        for pair in active_pairs
    }
    new_initial_pairs = append_new_guards(cutting, active_pairs, raw_snapshot)
    initial_active_ledger = cutting.canonical_active_pair_ledger(active_pairs)
    if (
        len(initial_active_ledger) != EXPECTED_INITIAL_ACTIVE_COUNT
        or len({tuple(value["key"]) for value in initial_active_ledger})
        != EXPECTED_INITIAL_ACTIVE_COUNT
    ):
        raise RuntimeError("expanded initial active ledger drift")

    cumulative = c0["cumulative"].copy()
    if (
        geometry.vector_sha256_float64_le(cumulative, np)
        != EXPECTED_C0_CUMULATIVE_SHA256
        or not math.isclose(
            float(np.linalg.norm(cumulative)),
            EXPECTED_C0_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("C0 cumulative vector drift before expanded run")

    ledger: list[dict[str, Any]] = []
    patch_audit: dict[str, Any] = {
        "gate_call_count": 0,
        "cutting_global_restored_every_call": True,
    }
    success_iteration: int | None = None
    close_reason = "maximum_additional_correction_count_reached_without_pass"
    overlay_started = False
    consumer_called = False
    final_integrity: dict[str, Any] = {}
    current: Mapping[str, Any] | None = None
    current_gate: dict[str, Any] | None = None
    terminal_cumulative = cumulative.copy()

    try:
        overlay_started = True
        cutting.apply_cumulative_from_raw(
            ram, parameters, raw_actor, cumulative, torch
        )
        c0_live_sha = helper.model_state_sha256(model.state_dict())
        if c0_live_sha != EXPECTED_C0_MODEL_STATE_SHA256:
            raise RuntimeError(f"fresh reconstructed C0 model SHA drift: {c0_live_sha}")
        current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
        nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        original_gate = cutting.gate_snapshot(
            original_descriptors,
            active_pairs[:EXPECTED_C0_ACTIVE_COUNT],
            raw_snapshot,
            current,
            nonactor_sha == raw_nonactor_sha,
            torch,
        )
        if not bool(original_gate["pass"]):
            raise RuntimeError("C0 no longer passes original 19-row/20-pair gate")
        transitions = new_guard_transition_audit(raw_snapshot, current)
        current_gate = expanded_gate_snapshot(
            cutting,
            descriptors,
            active_pairs,
            raw_snapshot,
            current,
            nonactor_sha == raw_nonactor_sha,
            torch,
            patch_audit,
        )
        if (
            current_gate["pass"]
            or len(current_gate["false_obligations"])
            != EXPECTED_C0_FALSE_OBLIGATIONS
            or any(
                int(item["row_index"]) < ORIGINAL_ROW_COUNT
                for item in current_gate["false_obligations"]
            )
        ):
            raise RuntimeError("C0 expanded false-obligation ledger drift")
        ledger.append(
            {
                "iteration": 0,
                "kind": "exact_c0_plus_four_frozen_fulltrain_cw_guards",
                "new_pairs": new_initial_pairs,
                "strengthened_pairs": remap_audit[
                    "old_guard_same_process_raw23_max_refresh"
                ],
                "active_pair_ledger": cutting.active_pair_ledger_audit(active_pairs),
                "step_l2": 0.0,
                "cumulative_l2": float(np.linalg.norm(cumulative)),
                "cumulative_float64_le_sha256": (
                    geometry.vector_sha256_float64_le(cumulative, np)
                ),
                "original_19row_gate": original_gate,
                "new_guard_c0_transitions": transitions,
                "post_gate": current_gate,
            }
        )

        for iteration in range(1, MAX_ITER + 1):
            if success_iteration is not None:
                break
            absolute_iteration = EXPECTED_C0_SUCCESS_ITERATION + iteration
            pair_update = cutting.add_dynamic_pairs(
                active_pairs,
                current_gate["false_obligations"],
                descriptors,
                raw_snapshot,
                current,
                torch,
                absolute_iteration,
            )
            nonactor_sha = helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            pre_gate = expanded_gate_snapshot(
                cutting,
                descriptors,
                active_pairs,
                raw_snapshot,
                current,
                nonactor_sha == raw_nonactor_sha,
                torch,
                patch_audit,
            )
            active_violated = sum(
                not bool(record["pass"])
                for record in pre_gate["active_pair_gates"]
            )
            has_separating_cut = (
                bool(pair_update["all_false_obligations_have_separating_cuts"])
                and active_violated > 0
            )
            if not has_separating_cut:
                close_reason = "fail_closed_no_violated_separating_cut"
                ledger.append(
                    {
                        "iteration": iteration,
                        "absolute_iteration": absolute_iteration,
                        "kind": "fail_closed_without_correction",
                        "new_pairs": pair_update["added"],
                        "strengthened_pairs": pair_update["strengthened"],
                        "separation_audit": pair_update,
                        "active_pair_ledger": cutting.active_pair_ledger_audit(
                            active_pairs
                        ),
                        "pre_gate": pre_gate,
                        "step_l2": 0.0,
                        "cumulative_l2": float(np.linalg.norm(cumulative)),
                        "cumulative_float64_le_sha256": (
                            geometry.vector_sha256_float64_le(cumulative, np)
                        ),
                    }
                )
                break

            gradients, rhs, gradient_audit = cutting.active_pair_gradients(
                geometry,
                helper,
                model,
                batch,
                current,
                active_pairs,
                parameters,
                device,
                np,
            )
            correction, qp_audit = cutting.solve_minimum_l2_correction(
                gradients, rhs, np, optimize
            )
            correction_l2 = float(np.linalg.norm(correction))
            if (
                not math.isfinite(correction_l2)
                or correction_l2 <= 0.0
                or correction_l2 > STEP_L2_CAP + 1e-12
            ):
                raise RuntimeError("expanded correction trust-region hard gate failed")
            cumulative = cumulative + correction
            cutting.restore_raw_actor(ram, parameters, raw_actor, torch)
            if helper.model_state_sha256(model.state_dict()) != raw_state_sha:
                raise RuntimeError("expanded per-iteration raw restoration failed")
            cutting.apply_cumulative_from_raw(
                ram, parameters, raw_actor, cumulative, torch
            )
            current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
            nonactor_sha = helper.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            current_gate = expanded_gate_snapshot(
                cutting,
                descriptors,
                active_pairs,
                raw_snapshot,
                current,
                nonactor_sha == raw_nonactor_sha,
                torch,
                patch_audit,
            )
            ledger.append(
                {
                    "iteration": iteration,
                    "absolute_iteration": absolute_iteration,
                    "kind": "expanded_cumulative_cuttingplane_correction",
                    "new_pairs": pair_update["added"],
                    "strengthened_pairs": pair_update["strengthened"],
                    "separation_audit": pair_update,
                    "active_pair_ledger": cutting.active_pair_ledger_audit(
                        active_pairs
                    ),
                    "pre_gate": pre_gate,
                    "gradient_audit": gradient_audit,
                    "qp": qp_audit,
                    "step_l2": correction_l2,
                    "cumulative_l2": float(np.linalg.norm(cumulative)),
                    "cumulative_float64_le_sha256": (
                        geometry.vector_sha256_float64_le(cumulative, np)
                    ),
                    "post_gate": current_gate,
                }
            )
            if current_gate["pass"]:
                success_iteration = iteration
                close_reason = "all_23_rows_and_all_active_thresholds_pass"

        terminal_cumulative = cumulative.copy()
        terminal_l2 = float(np.linalg.norm(terminal_cumulative))
        terminal_sha = geometry.vector_sha256_float64_le(
            terminal_cumulative, np
        )
        terminal_model_sha = helper.model_state_sha256(model.state_dict())
        terminal_nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        terminal_active_ledger = cutting.canonical_active_pair_ledger(
            active_pairs
        )
        terminal_active_ledger_sha = sha256_bytes(
            canonical_json(terminal_active_ledger)
        )
        final_original_thresholds = {
            tuple(record["key"]): float(record["threshold"])
            for record in terminal_active_ledger
            if int(record["key"][0]) < ORIGINAL_ROW_COUNT
        }
        terminal_checks = {
            "success_iteration_exact_5": success_iteration
            == EXPECTED_SUCCESS_CONTINUATION_ITERATION,
            "terminal_l2_exact": math.isclose(
                terminal_l2,
                EXPECTED_TERMINAL_CUMULATIVE_L2,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "terminal_cumulative_sha_exact": terminal_sha
            == EXPECTED_TERMINAL_CUMULATIVE_SHA256,
            "terminal_model_sha_exact": terminal_model_sha
            == EXPECTED_TERMINAL_MODEL_STATE_SHA256,
            "terminal_nonactor_exact_raw": terminal_nonactor_sha
            == raw_nonactor_sha,
            "final_active_count_exact_24": len(terminal_active_ledger)
            == EXPECTED_INITIAL_ACTIVE_COUNT,
            "final_active_ledger_sha_exact": terminal_active_ledger_sha
            == EXPECTED_TERMINAL_ACTIVE_LEDGER_SHA256,
            "original_active20_absolute_thresholds_preserved": (
                final_original_thresholds == original_active_thresholds
            ),
            "final_23row_gate_pass": bool(current_gate["pass"]),
        }
        if not all(terminal_checks.values()):
            raise RuntimeError(
                f"preregistered terminal reproduction gate failed: {terminal_checks}"
            )
        if success_iteration is not None and candidate_consumer is not None:
            consumer_called = True
            candidate_consumer(
                {
                    "helper": helper,
                    "model": model,
                    "checkpoint": checkpoint,
                    "model_config": model_config,
                    "raw_actor": raw_actor,
                    "raw_model_state_sha256": raw_state_sha,
                    "raw_nonactor_sha256": raw_nonactor_sha,
                    "terminal_cumulative_float64": terminal_cumulative.copy(),
                    "terminal_cumulative_float64_le_sha256": terminal_sha,
                    "success_iteration": success_iteration,
                    "selected_row_gate": current_gate,
                    "active_pair_ledger": terminal_active_ledger,
                    "c0_model_state_sha256": EXPECTED_C0_MODEL_STATE_SHA256,
                    "expanded_row_count": EXPANDED_ROW_COUNT,
                }
            )
    finally:
        if overlay_started:
            cutting.restore_raw_actor(ram, parameters, raw_actor, torch)
            final_integrity = cutting.raw_restore_integrity(
                helper,
                ram,
                model,
                parameters,
                nonactor_names,
                raw_state_sha,
                raw_nonactor_sha,
                raw_snapshot,
                batch,
                cpu_batch,
                device,
                torch,
            )
            if not bool(final_integrity["pass"]):
                raise RuntimeError("expanded finally raw restoration failed")

    if not bool(patch_audit["cutting_global_restored_every_call"]):
        raise RuntimeError("cutting gate global was not restored on every call")
    status = (
        "expanded_selected_row_adaptive_optimization_success"
        if success_iteration is not None
        else "closed_no_candidate"
    )
    final_active_ledger = cutting.canonical_active_pair_ledger(active_pairs)
    return {
        "schema_version": SCHEMA,
        "status": status,
        "scope": {
            "train_rows_only": True,
            "row_count": EXPANDED_ROW_COUNT,
            "original_rows": ORIGINAL_ROW_COUNT,
            "new_fulltrain_cw_rows": len(NEW_GUARDS),
            "max_additional_corrections": MAX_ITER,
            "step_l2_cap": STEP_L2_CAP,
            "hyperparameter_sweep": False,
            "candidate_RAM_only": True,
            "model_or_result_writes": 0,
            "validation_or_holdout_access": False,
            "network_upload_submission": False,
            "stdout_only": True,
            "standalone_candidate_consumer_is_none": candidate_consumer is None,
        },
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "cutting_v2": dict(cutting_evidence),
            "fulltrain_v2": dict(fulltrain_evidence),
            "geometry": dict(geometry_evidence),
            "ram_runner": dict(ram_evidence),
            "formal_evaluator": dict(formal_evidence),
            "formal_result": formal_result_evidence,
            "formal_design": {
                "tool": design_evidence["tool"],
                "artifact": design_evidence["artifact"],
            },
            "expanded_train_archives": archive_evidence,
        },
        "c0_exact_reconstruction": c0_audit,
        "expanded_descriptors": [
            {
                **identity_record(value),
                "role": value["role"],
                "metrics_union": (
                    list(MAIN_METRICS)
                    if value["role"] == "target"
                    else list(value["metrics_union"])
                ),
                "positive_option": int(value["positive_option"]),
                "negative_option": int(value["negative_option"]),
            }
            for value in descriptors
        ],
        "row_loading_audit": loader_patch_audit,
        "c0_active_pair_remap": remap_audit,
        "initial_active_pair_contract": {
            "count": len(initial_active_ledger),
            "expected_count": EXPECTED_INITIAL_ACTIVE_COUNT,
            "canonical_json_sha256": sha256_bytes(
                canonical_json(initial_active_ledger)
            ),
            "canonical_ledger": initial_active_ledger,
        },
        "iterations": ledger,
        "active_pair_contract": {
            "initial_count": EXPECTED_INITIAL_ACTIVE_COUNT,
            "final_count": len(final_active_ledger),
            "old_target_thresholds": "absolute C0 canonical thresholds unchanged",
            "old_guard_thresholds": (
                "C0 canonical threshold hard-equals same-process raw 23-row margin"
            ),
            "new_guard_thresholds": (
                "same-process raw fulltrain 23-row pair margin"
            ),
            "dynamic_thresholds": {
                "target": "current exact pair native-BF16 nextafter local q",
                "guard": "same-process raw 23-row exact pair margin",
            },
            "dedup_key": ["row_index", "positive_option", "negative_option"],
            "final_canonical_ledger_sha256": sha256_bytes(
                canonical_json(final_active_ledger)
            ),
            "final_canonical_ledger": final_active_ledger,
        },
        "dependency_global_patch_audit": {
            **loader_patch_audit,
            **patch_audit,
            "geometry_expected_row_count_final": geometry.EXPECTED_ROW_COUNT,
            "cutting_expected_selected_row_count_final": (
                cutting.EXPECTED_SELECTED_ROW_COUNT
            ),
            "all_globals_restored": (
                geometry.EXPECTED_ROW_COUNT == ORIGINAL_ROW_COUNT
                and cutting.EXPECTED_SELECTED_ROW_COUNT == ORIGINAL_ROW_COUNT
            ),
        },
        "decision": {
            "status": status,
            "close_reason": close_reason,
            "success_iteration_after_c0": success_iteration,
            "first_passing_iteration_selected": success_iteration is not None,
            "terminal_cumulative_l2": terminal_l2,
            "terminal_cumulative_float64_le_sha256": terminal_sha,
            "candidate_model_state_sha256_before_restore": terminal_model_sha,
            "terminal_nonactor_sha256_before_restore": terminal_nonactor_sha,
            "terminal_active_pair_ledger_sha256": terminal_active_ledger_sha,
            "preregistered_terminal_checks": terminal_checks,
            "candidate_consumer_called_before_finally_restore": consumer_called,
            "model_materialized": False,
            "submission_performed": False,
        },
        "final_integrity": {
            **final_integrity,
            "finally_restore_executed_after_overlay": overlay_started,
            "static_zero_write_audit": dict(static),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT,
        sha256_bytes(SCRIPT.read_bytes()),
        "expanded CW4 cutting-plane runner",
        expected_mode=FROZEN_MODE,
    )
    static = static_audit(source)
    cutting, cutting_evidence = import_frozen(
        CUTTING, CUTTING_SHA256, "u468_cuttingplane_v2_frozen_for_cw4"
    )
    fulltrain, fulltrain_evidence = import_frozen(
        FULLTRAIN, FULLTRAIN_SHA256, "u468_fulltrain_v2_frozen_for_cw4"
    )
    geometry, geometry_evidence = cutting.import_frozen(
        cutting.GEOMETRY,
        cutting.GEOMETRY_SHA256,
        "u468_geometry_frozen_for_cw4",
    )
    ram, ram_evidence = cutting.import_frozen(
        cutting.RAM_RUNNER,
        cutting.RAM_RUNNER_SHA256,
        "u468_ram_frozen_for_cw4",
    )
    formal, formal_evidence = fulltrain.import_frozen(
        fulltrain.FORMAL,
        fulltrain.FORMAL_SHA256,
        "u468_formal_v3_frozen_for_cw4",
    )
    interfaces = interface_audit(cutting, fulltrain, geometry, ram, formal)
    if args.mode == "static":
        cutting_source, _ = cutting.read_regular_bytes(
            cutting.SCRIPT,
            CUTTING_SHA256,
            "static cutting v2",
            expected_mode=FROZEN_MODE,
        )
        fulltrain_source, _ = fulltrain.read_regular_bytes(
            fulltrain.SCRIPT,
            FULLTRAIN_SHA256,
            "static fulltrain v2",
            expected_mode=FROZEN_MODE,
        )
        geometry_source, _ = read_regular_bytes(
            geometry.SCRIPT,
            cutting.GEOMETRY_SHA256,
            "static geometry",
            expected_mode=FROZEN_MODE,
        )
        ram_source, _ = read_regular_bytes(
            ram.SCRIPT,
            cutting.RAM_RUNNER_SHA256,
            "static RAM runner",
            expected_mode=FROZEN_MODE,
        )
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {
                "cutting_v2": cutting_evidence,
                "fulltrain_v2": fulltrain_evidence,
                "geometry": geometry_evidence,
                "ram_runner": ram_evidence,
                "formal_evaluator": formal_evidence,
            },
            "interface_audit": interfaces,
            "new_guard_spec_sha256": sha256_bytes(canonical_json(NEW_GUARDS)),
            "audit": {
                "local": static,
                "cutting_v2": cutting.static_audit(cutting_source),
                "fulltrain_v2": fulltrain.local_static_audit(fulltrain_source),
                "geometry": geometry.static_audit(geometry_source),
                "ram_runner": ram.static_audit(ram_source),
                "formal_evaluator": formal.ast_audit(),
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
        }
    else:
        result = run_cuttingplane(
            source,
            static,
            cutting,
            cutting_evidence,
            fulltrain,
            fulltrain_evidence,
            geometry,
            geometry_evidence,
            ram,
            ram_evidence,
            formal,
            formal_evidence,
        )
        result["interface_audit"] = interfaces
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
