#!/usr/bin/env python3
"""Read-only CW11 continuation from the exact CW10/E904 RAM terminal.

The frozen CW10 runner reconstructs E904 and invokes this wrapper while the
candidate is still live.  CW11 then reloads the original 29 selected train
rows plus four exact, already-consumed specialist-valid counterexamples,
preserves the complete frozen 30-pair ledger, appends four same-process raw
guard floors, and performs bounded minimum-L2 actor6 corrections.

This is an exploratory/development repair.  Because exact specialist-valid
rows participate in optimization, that specialist split is contaminated and
may only be used afterward as a consistency check.  Formal all-train, broad,
and Gold evaluation remain separate.  There are no checkpoint/result writes,
training optimizers, backward calls, network calls, uploads, or submissions.
Output is one JSON document on stdout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import stat
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "probe_u468_raw_actor6_metricguard_specialist_valid_cw11_cuttingplane_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-specialist-valid-cw11-probe-v1"
SEED = 202608206
FROZEN_MODE = 0o555

CW10 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw10_cuttingplane_v1.py"
CW10_SHA256 = "546e90c5b3ca35c84b8efc08109b2f310b45b2895d24aeee70561d13bffa146c"
FULLTRAIN_V5 = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_gate_v5.py"
FULLTRAIN_V5_SHA256 = "90ab439a0ca75e7f0a2cf2c8e7c117a07ee5ff9b8b6462ffc5e8380114b404a0"
CW9 = TOOLS / "probe_u468_raw_actor6_metricguard_fulltrain_cw9_cuttingplane_v1.py"
CW9_SHA256 = "f77a76f2ac91732e8bc8d9a436ca7a49058b5de50f7b3114db14cffa1fbf5292"
PRIMARY = TOOLS / "run_u468_raw_actor6_metricguard_fulltrain_cw4_cuttingplane_v1.py"
PRIMARY_SHA256 = "40715927549457b161fc907757279e4d85466ded1d1e6bfed62f67c13b529e7c"
CUTTING = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_cuttingplane_v2.py"
CUTTING_SHA256 = "c2866ba8b00eba6b424197a520419a5717401335cc47202b4fcc711611f67503"
GEOMETRY = TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py"
GEOMETRY_SHA256 = "ddecd3a85bc2b43c28854afc56678c21943198ac3fde3613765d6a49e440eedb"
RAM = TOOLS / "run_u468_raw_actor6_metricguard_specialbc_ram_ray_v1.py"
RAM_SHA256 = "86b05d4f826576717907141aef2c534f140a1531b87c8e2d28aeb624a2657e4d"
FLIP_REPORT = ROOT / "artifacts/e904_specialist_row_flips_readonly_20260802_v1.json"
FLIP_REPORT_SHA256 = "ad269467b80b1757960aec635b2db73ff29053b15605a4207540d5fce27bb7db"

EXPECTED_RAW_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
EXPECTED_RAW_NONACTOR_SHA256 = (
    "4f2287215392270ccb86d73e0af89b01d89e0f757f7227ce811038ab5d92fafb"
)
EXPECTED_CW10_MODEL_STATE_SHA256 = (
    "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
)
EXPECTED_CW10_CUMULATIVE_SHA256 = (
    "42e02c90f65b81e1e8b801fa173fcf953b347f73e65b58000719107fa83a6c28"
)
EXPECTED_CW10_CUMULATIVE_L2 = 0.007627603437990896
EXPECTED_CW10_SUCCESS_ITERATION = 9
EXPECTED_CW10_ACTIVE_COUNT = 30
EXPECTED_CW10_ACTIVE_LEDGER_SHA256 = (
    "22b20b8630d63c6fbfd249a82a188397b468a84d1b747a2159c17b05a6f4983d"
)

OLD_ROW_COUNT = 29
ADDED_VALID_GUARD_COUNT = 4
EXPECTED_ROW_COUNT = 33
EXPECTED_TARGET_ROW_COUNT = 5
EXPECTED_GUARD_ROW_COUNT = 28
EXPECTED_TARGET_OBLIGATIONS = 20
EXPECTED_OLD_GUARD_OBLIGATIONS = 69
EXPECTED_ADDED_GUARD_OBLIGATIONS = 15
EXPECTED_GUARD_OBLIGATIONS = 84
EXPECTED_INITIAL_ACTIVE_COUNT = 34
MAX_ITER = 12
STEP_L2_CAP = 0.001
MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)

ADDED_VALID_GUARDS = (
    {
        "role": "guard",
        "panel": "flg",
        "specialist_views": ["flg"],
        "member": "valid/part-00000.jsonl",
        "line_index_zero_based": 566,
        "line_sha256": "d400c5d20ab5d2d8942c07a652cd2005607fcc0bdf7a291112457270faa690dc",
        "expected_context": 16,
        "expected_team_name": "flg",
        "expected_expert_order": [0],
        "expected_raw_order": [0],
        "expected_cw10_order": [1],
        "expected_option_count": 2,
        "metrics_union": list(MAIN_METRICS),
        "positive_option": 0,
        "negative_option": 1,
        "reported_specialist_b256_raw_margin": 0.0,
        "reported_specialist_b256_cw10_margin": -0.00390625,
    },
    {
        "role": "guard",
        "panel": "core5",
        "specialist_views": ["core5", "dominic"],
        "member": "valid/part-00000.jsonl",
        "line_index_zero_based": 734,
        "line_sha256": "518e152619376bd0d8ded7624bec1c22741dfbc0199a78e0185607423b4ae576",
        "expected_context": 0,
        "expected_team_name": "Dominic Peel",
        "expected_expert_order": [4],
        "expected_raw_order": [4],
        "expected_cw10_order": [3],
        "expected_option_count": 16,
        "metrics_union": list(MAIN_METRICS),
        "positive_option": 4,
        "negative_option": 3,
        "reported_specialist_b256_raw_margin": 0.015625,
        "reported_specialist_b256_cw10_margin": -0.0078125,
    },
    {
        "role": "guard",
        "panel": "core5",
        "specialist_views": ["core5", "szlach"],
        "member": "valid/part-00004.jsonl",
        "line_index_zero_based": 2442,
        "line_sha256": "059e8a9dae616851e8a45b5768c7d1fad99b8274acd42d6faad1211e158495d5",
        "expected_context": 7,
        "expected_team_name": "szlachetny snieg",
        "expected_expert_order": [16],
        "expected_raw_order": [16],
        "expected_cw10_order": [12],
        "expected_option_count": 17,
        "metrics_union": list(MAIN_METRICS),
        "positive_option": 16,
        "negative_option": 12,
        "reported_specialist_b256_raw_margin": 0.00390625,
        "reported_specialist_b256_cw10_margin": -0.00390625,
    },
    {
        "role": "guard",
        "panel": "pokemonfan",
        "specialist_views": ["pokemonfan"],
        "member": "valid/part-00000.jsonl",
        "line_index_zero_based": 10655,
        "line_sha256": "349a4f1cea874065c083b34082fe0b1fa0cef3b31ab5aeccb4257be90461e9aa",
        "expected_context": 7,
        "expected_team_name": "Pokemon Fan",
        "expected_expert_order": [0, 1, 2],
        "expected_raw_order": [0, 1, 2],
        "expected_cw10_order": [0, 1, 5],
        "expected_option_count": 6,
        "metrics_union": [
            "set_exact",
            "hybrid_order_exact",
            "ordered_exact",
        ],
        "positive_option": 2,
        "negative_option": 5,
        "reported_specialist_b256_raw_margin": 0.00390625,
        "reported_specialist_b256_cw10_margin": -0.00390625,
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
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA drift: {digest}")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(mode, "04o"),
        "single_link_regular_held_fd_identity_exact": True,
    }


def import_frozen(
    path: Path, expected_sha256: str, label: str
) -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_regular_bytes(
        path, expected_sha256, label, expected_mode=FROZEN_MODE
    )
    spec = importlib.util.spec_from_file_location(
        f"_cw11_{path.stem}_{expected_sha256[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {label}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def static_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"), filename=str(SCRIPT))
    forbidden_calls = {
        "save",
        "savez",
        "write",
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "replace",
        "rename",
        "unlink",
        "remove",
        "rmtree",
        "backward",
        "step",
    }
    forbidden_import_roots = {
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "kaggle",
    }
    call_hits: list[dict[str, Any]] = []
    import_hits: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls:
                call_hits.append({"name": name, "line": node.lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    import_hits.append({"name": name, "line": node.lineno})
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_or_network_calls": not call_hits,
        "no_network_or_submission_imports": not import_hits,
        "four_added_valid_guards": len(ADDED_VALID_GUARDS)
        == ADDED_VALID_GUARD_COUNT,
        "added_guard_identities_unique": len(
            {identity_key(value) for value in ADDED_VALID_GUARDS}
        )
        == ADDED_VALID_GUARD_COUNT,
        "added_guard_obligations_exact_15": sum(
            len(value["metrics_union"]) for value in ADDED_VALID_GUARDS
        )
        == EXPECTED_ADDED_GUARD_OBLIGATIONS,
        "row_accounting_exact_33": OLD_ROW_COUNT + ADDED_VALID_GUARD_COUNT
        == EXPECTED_ROW_COUNT,
        "active_pair_accounting_exact_34": (
            EXPECTED_CW10_ACTIVE_COUNT + ADDED_VALID_GUARD_COUNT
            == EXPECTED_INITIAL_ACTIVE_COUNT
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW11 static audit failed: {checks}")
    return {
        "checks": checks,
        "forbidden_call_hits": call_hits,
        "forbidden_import_hits": import_hits,
        "stdout_only": True,
    }


def validate_interfaces(
    cw10: ModuleType,
    cw9: ModuleType,
    primary: ModuleType,
    cutting: ModuleType,
    geometry: ModuleType,
    ram: ModuleType,
) -> dict[str, Any]:
    checks = {
        "cw10_to_cw9_exact": Path(cw10.CW9) == CW9
        and str(cw10.CW9_SHA256) == CW9_SHA256,
        "cw9_to_primary_exact": Path(cw9.PRIMARY) == PRIMARY
        and str(cw9.PRIMARY_SHA256) == PRIMARY_SHA256,
        "primary_to_cutting_exact": Path(primary.CUTTING) == CUTTING
        and str(primary.CUTTING_SHA256) == CUTTING_SHA256,
        "cutting_to_geometry_exact": Path(cutting.GEOMETRY) == GEOMETRY
        and str(cutting.GEOMETRY_SHA256) == GEOMETRY_SHA256,
        "cutting_to_ram_exact": Path(cutting.RAM_RUNNER) == RAM
        and str(cutting.RAM_RUNNER_SHA256) == RAM_SHA256,
        "actor6_names_exact": tuple(geometry.ACTOR6_NAMES)
        == (
            "actor_query.weight",
            "actor_key.weight",
            "actor_residual.0.weight",
            "actor_residual.0.bias",
            "actor_residual.2.weight",
            "actor_residual.2.bias",
        ),
        "step_l2_cap_exact": float(cutting.STEP_L2_CAP) == STEP_L2_CAP,
        "max_iter_exact": int(cutting.MAX_ITER) == MAX_ITER,
        "main_metrics_exact": tuple(cutting.MAIN_METRICS) == MAIN_METRICS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW11 interface audit failed: {checks}")
    return checks


def validate_flip_report(payload: bytes) -> dict[str, Any]:
    report = json.loads(payload)
    bindings = report.get("checkpoint_bindings", {})
    checks = {
        "status_pass": report.get("status")
        == "completed_read_only_valid_diagnostic",
        "raw_model_sha_exact": bindings.get("raw", {}).get(
            "model_state_sha256"
        )
        == EXPECTED_RAW_MODEL_STATE_SHA256,
        "candidate_model_sha_exact": bindings.get("candidate", {}).get(
            "model_state_sha256"
        )
        == EXPECTED_CW10_MODEL_STATE_SHA256,
    }
    report_rows: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for panel in ("flg", "core5", "pokemonfan"):
        for row in report.get("panels", {}).get(panel, {}).get("flips", []):
            key = (
                panel,
                str(row["member"]),
                int(row["line_index"]),
                str(row["line_sha256"]),
            )
            report_rows[key] = row
    row_checks = []
    for descriptor in ADDED_VALID_GUARDS:
        key = identity_key(descriptor)
        row = report_rows.get(key)
        if row is None:
            raise RuntimeError(f"CW11 guard absent from frozen flip report: {key}")
        first = row["first_prediction_difference"]
        local = {
            "context_exact": int(row["context"])
            == int(descriptor["expected_context"]),
            "team_exact": str(row["team_name"])
            == str(descriptor["expected_team_name"]),
            "expert_exact": list(row["expert_order"])
            == list(descriptor["expected_expert_order"]),
            "raw_order_exact": list(row["raw"]["order"])
            == list(descriptor["expected_raw_order"]),
            "cw10_order_exact": list(row["candidate"]["order"])
            == list(descriptor["expected_cw10_order"]),
            "positive_exact": int(first["raw_choice"])
            == int(descriptor["positive_option"]),
            "negative_exact": int(first["candidate_choice"])
            == int(descriptor["negative_option"]),
            "raw_margin_exact": float(
                first["raw_logit_raw_minus_candidate_choice"]
            )
            == float(descriptor["reported_specialist_b256_raw_margin"]),
            "cw10_margin_exact": float(
                first["candidate_logit_raw_minus_candidate_choice"]
            )
            == float(descriptor["reported_specialist_b256_cw10_margin"]),
        }
        if not all(local.values()):
            raise RuntimeError(f"CW11 flip report row drift: {key} {local}")
        row_checks.append({"identity": identity_record(descriptor), "checks": local})
    if not all(checks.values()):
        raise RuntimeError(f"CW11 flip report binding drift: {checks}")
    return {
        "checks": checks,
        "row_checks": row_checks,
        "selected_strategy": (
            "minimum four physical guards from live E904: one PokemonFan "
            "aggregate repair, one FLG aggregate repair, and the mandatory "
            "Dominic/Szlach subgroup repairs"
        ),
        "specialist_valid_contaminated_after_optimization": True,
    }


def reconstruct_old_descriptors(
    primary: ModuleType,
    geometry: ModuleType,
    cw9: ModuleType,
    cw10: ModuleType,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original, formal_evidence = primary.load_original_descriptors(geometry)
    # CW10 invokes its consumer while CW9's temporary patch is live.  At this
    # exact point primary.NEW_GUARDS is already the full ordered 4+5+1 tuple.
    # Consuming that live tuple avoids duplicating CW9/CW10 witnesses.
    live_guards = [dict(value) for value in primary.NEW_GUARDS]
    descriptors = [
        *[dict(value) for value in original],
        *live_guards,
    ]
    expected_tail = [
        *[identity_key(value) for value in cw9.ADDED_GUARDS],
        identity_key(cw10.FINAL_ADDED_GUARD),
    ]
    checks = {
        "live_CW10_patched_guard_count_exact_10": len(live_guards) == 10,
        "live_CW9_plus_CW10_tail_exact_6": [
            identity_key(value) for value in live_guards[-6:]
        ]
        == expected_tail,
        "old_rows_exact_29": len(descriptors) == OLD_ROW_COUNT,
        "old_identities_unique_29": len(
            {identity_key(value) for value in descriptors}
        )
        == OLD_ROW_COUNT,
        "old_target_rows_exact_5": sum(
            value["role"] == "target" for value in descriptors
        )
        == EXPECTED_TARGET_ROW_COUNT,
        "old_guard_rows_exact_24": sum(
            value["role"] == "guard" for value in descriptors
        )
        == 24,
        "old_guard_obligations_exact_69": sum(
            len(value["metrics_union"])
            for value in descriptors
            if value["role"] == "guard"
        )
        == EXPECTED_OLD_GUARD_OBLIGATIONS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW11 old descriptor reconstruction drift: {checks}")
    return descriptors, {
        "checks": checks,
        "formal_result": formal_evidence,
        "canonical_identity_sha256": sha256_bytes(
            canonical_json([identity_record(value) for value in descriptors])
        ),
    }


def load_selected_train_and_exact_valid_rows(
    geometry: ModuleType,
    helper: ModuleType,
    descriptors: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid_allowlist = {identity_key(value) for value in ADDED_VALID_GUARDS}
    requested: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for descriptor in descriptors:
        panel = str(descriptor["panel"])
        member = str(descriptor["member"])
        line_index = int(descriptor["line_index_zero_based"])
        target = requested.setdefault(panel, {}).setdefault(member, {})
        if line_index in target:
            raise RuntimeError("CW11 selected identity overlap")
        target[line_index] = descriptor

    loaded: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    archives: dict[str, Any] = {}
    opened_valid: set[tuple[str, str]] = set()
    for panel in geometry.PANEL_SCAN_ORDER:
        if panel not in requested:
            continue
        payload, evidence = geometry.read_regular_bytes(
            geometry.DATASETS[panel],
            geometry.DATA_SHA256[panel],
            f"CW11 {panel} frozen train-valid archive",
        )
        opened_members = []
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError(f"CW11 duplicate zip member names for {panel}")
            for member, index_map in sorted(requested[panel].items()):
                if member not in names or not member.endswith(".jsonl"):
                    raise RuntimeError(f"CW11 missing selected member {panel}/{member}")
                is_train = member.startswith("train/")
                is_valid = member.startswith("valid/")
                if not is_train and not is_valid:
                    raise RuntimeError(f"CW11 refused split member {panel}/{member}")
                if is_valid:
                    for line_index, descriptor in index_map.items():
                        if identity_key(descriptor) not in valid_allowlist:
                            raise RuntimeError("CW11 non-allowlisted valid row requested")
                    opened_valid.add((panel, member))
                opened_members.append(member)
                remaining = set(index_map)
                maximum = max(remaining)
                with archive.open(member, "r") as handle:
                    for line_index, raw_line in enumerate(handle):
                        if line_index in remaining:
                            descriptor = index_map[line_index]
                            digest = sha256_bytes(raw_line)
                            if digest != str(descriptor["line_sha256"]):
                                raise RuntimeError(
                                    f"CW11 selected line SHA drift at "
                                    f"{panel}/{member}:{line_index}"
                                )
                            row = helper.orjson.loads(raw_line)
                            expected_split = "train" if is_train else "valid"
                            if (
                                not isinstance(row, dict)
                                or str(row.get("split", "")) != expected_split
                            ):
                                raise RuntimeError("CW11 selected row split drift")
                            raw_action = row.get("action")
                            if not isinstance(raw_action, list):
                                raise RuntimeError("CW11 row action is not a list")
                            expert_order = [int(value) for value in raw_action]
                            previous_max = helper.bc.MAX_ACTION_COUNT
                            helper.bc.MAX_ACTION_COUNT = helper.ppo.MAX_ACTION_COUNT
                            try:
                                features = helper.bc.featurize_row(
                                    row,
                                    int(model_config["hash_size"]),
                                    int(model_config["max_state_entities"]),
                                )
                            finally:
                                helper.bc.MAX_ACTION_COUNT = previous_max
                            options = (
                                ((row.get("observation") or {}).get("select") or {}).get(
                                    "option"
                                )
                                or []
                            )
                            if (
                                features is None
                                or len(expert_order) > helper.ppo.MAX_ACTION_COUNT
                                or len(set(expert_order)) != len(expert_order)
                                or any(
                                    value < 0 or value >= len(options)
                                    for value in expert_order
                                )
                            ):
                                raise RuntimeError("CW11 selected row is not evaluable")
                            features["expert_action_order"] = expert_order
                            metadata = {
                                "team_name": str(row.get("team_name", "")),
                                "context": int(features["context"]),
                                "expert_order": expert_order,
                                "option_count": len(options),
                            }
                            if is_valid:
                                exact = {
                                    "context": metadata["context"]
                                    == int(descriptor["expected_context"]),
                                    "team": metadata["team_name"]
                                    == str(descriptor["expected_team_name"]),
                                    "expert": metadata["expert_order"]
                                    == list(descriptor["expected_expert_order"]),
                                    "option_count": metadata["option_count"]
                                    == int(descriptor["expected_option_count"]),
                                }
                                if not all(exact.values()):
                                    raise RuntimeError(
                                        f"CW11 valid metadata drift: {exact}"
                                    )
                            key = identity_key(descriptor)
                            loaded[key] = {
                                "features": features,
                                "raw_metadata": metadata,
                            }
                            remaining.discard(line_index)
                        if line_index >= maximum:
                            break
                if remaining:
                    raise RuntimeError(
                        f"CW11 selected rows missing in {panel}/{member}: {remaining}"
                    )
        archives[panel] = {
            **evidence,
            "opened_members": opened_members,
            "selected_row_count": sum(
                len(index_map) for index_map in requested[panel].values()
            ),
        }

    rows = []
    for descriptor in descriptors:
        item = loaded.get(identity_key(descriptor))
        if item is None:
            raise RuntimeError(f"CW11 selected row not loaded: {identity_key(descriptor)}")
        rows.append(item)
    checks = {
        "loaded_rows_exact_33": len(rows) == EXPECTED_ROW_COUNT
        and len(loaded) == EXPECTED_ROW_COUNT,
        "valid_identity_allowlist_exact_4": {
            identity_key(value)
            for value in descriptors
            if str(value["member"]).startswith("valid/")
        }
        == valid_allowlist,
        "only_expected_valid_members_opened": opened_valid
        == {
            (str(value["panel"]), str(value["member"]))
            for value in ADDED_VALID_GUARDS
        },
    }
    if not all(checks.values()):
        raise RuntimeError(f"CW11 selected loader gate failed: {checks}")
    return rows, {
        "checks": checks,
        "archives": archives,
        "validation_member_payloads_opened": True,
        "opened_validation_rows_exactly_allowlisted": True,
        "opened_validation_row_count": ADDED_VALID_GUARD_COUNT,
    }


def remap_old_active_pairs(
    cutting: ModuleType,
    ledger: Sequence[Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        len(ledger) != EXPECTED_CW10_ACTIVE_COUNT
        or sha256_bytes(canonical_json(ledger))
        != EXPECTED_CW10_ACTIVE_LEDGER_SHA256
    ):
        raise RuntimeError("CW11 frozen CW10 active ledger drift")
    active = []
    records = []
    for record in ledger:
        row_index, positive, negative = [int(value) for value in record["key"]]
        if not 0 <= row_index < OLD_ROW_COUNT:
            raise RuntimeError("CW11 old active row index out of range")
        expected_identity = identity_record(descriptors[row_index])
        if dict(record["identity"]) != expected_identity:
            raise RuntimeError("CW11 old active identity/index drift")
        pair = {
            "row_index": row_index,
            "identity": expected_identity,
            "role": str(record["role"]),
            "positive_option": positive,
            "negative_option": negative,
            "threshold": float(record["threshold"]),
            "threshold_source": str(record["threshold_source"]),
            "threshold_sources": [str(value) for value in record["threshold_sources"]],
            "threshold_history": [dict(value) for value in record["threshold_history"]],
            "origins": [str(value) for value in record["origins"]],
            "constructions": [str(value) for value in record["construction"]],
            "created_iteration": int(record["created_iteration"]),
        }
        active.append(pair)
        records.append(
            {
                "key": [row_index, positive, negative],
                "identity": expected_identity,
                "threshold_preserved_exact": True,
            }
        )
    if (
        len(active) != EXPECTED_CW10_ACTIVE_COUNT
        or len({cutting.pair_key(value) for value in active})
        != EXPECTED_CW10_ACTIVE_COUNT
    ):
        raise RuntimeError("CW11 old active pair cardinality drift")
    return active, {
        "old_count_exact_30": True,
        "old_canonical_ledger_sha256": sha256_bytes(canonical_json(ledger)),
        "old_absolute_thresholds_preserved": True,
        "records": records,
    }


def append_valid_guards(
    cutting: ModuleType,
    active_pairs: list[dict[str, Any]],
    raw_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    existing = {cutting.pair_key(pair) for pair in active_pairs}
    added = []
    for row_index, descriptor in enumerate(
        ADDED_VALID_GUARDS, start=OLD_ROW_COUNT
    ):
        pair = {
            "row_index": row_index,
            "identity": identity_record(descriptor),
            "role": "guard",
            "positive_option": int(descriptor["positive_option"]),
            "negative_option": int(descriptor["negative_option"]),
            "threshold": 0.0,
            "threshold_source": "same_process_raw_33row_valid_guard_margin",
            "threshold_sources": [
                "same_process_raw_33row_valid_guard_margin",
                "reported_specialist_B256_raw_margin_advisory",
            ],
            "threshold_history": [],
            "origins": ["consumed_specialist_valid_CW_guard"],
            "constructions": ["raw_safe_vs_CW10_first_prediction_difference"],
            "created_iteration": EXPECTED_CW10_SUCCESS_ITERATION,
        }
        threshold = cutting.pair_margin(raw_snapshot, pair)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise RuntimeError("CW11 same-process valid raw guard margin invalid")
        pair["threshold"] = threshold
        pair["threshold_history"] = [
            {
                "iteration": EXPECTED_CW10_SUCCESS_ITERATION,
                "source": pair["threshold_source"],
                "observed": threshold,
                "reported_specialist_B256_raw_margin": float(
                    descriptor["reported_specialist_b256_raw_margin"]
                ),
                "retained_max": threshold,
            }
        ]
        key = cutting.pair_key(pair)
        if key in existing:
            raise RuntimeError("CW11 valid guard pair duplicates old pair")
        existing.add(key)
        active_pairs.append(pair)
        added.append(dict(pair))
    if len(active_pairs) != EXPECTED_INITIAL_ACTIVE_COUNT:
        raise RuntimeError("CW11 initial active ledger cardinality drift")
    return added


def gate33(
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
        cutting.EXPECTED_SELECTED_ROW_COUNT = EXPECTED_ROW_COUNT
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
        restored = cutting.EXPECTED_SELECTED_ROW_COUNT == previous
    patch_audit["gate_call_count"] = int(patch_audit.get("gate_call_count", 0)) + 1
    patch_audit["global_restored_every_call"] = bool(
        patch_audit.get("global_restored_every_call", True)
    ) and restored
    if not restored or previous != 19:
        raise RuntimeError("CW11 cutting row-count global restoration failed")
    legacy = result.pop("selected_official_row_count_exact_19")
    result["selected_official_row_count_exact_33"] = bool(legacy)
    hard = {
        "rows_exact_33": int(result["selected_official_row_count"])
        == EXPECTED_ROW_COUNT
        and bool(result["selected_official_row_count_exact_33"]),
        "target_obligations_exact_20": int(
            result["target_5x4_obligation_count"]
        )
        == EXPECTED_TARGET_OBLIGATIONS,
        "guard_obligations_exact_84": int(result["guard_obligation_count"])
        == EXPECTED_GUARD_OBLIGATIONS,
        "active_count_at_least_34": int(result["active_pair_count"])
        >= EXPECTED_INITIAL_ACTIVE_COUNT,
        "count_value_nonactor_exact": bool(
            result["count_logits_native_exact_raw"]
            and result["value_logits_native_exact_raw"]
            and result["nonactor_state_exact_raw"]
        ),
    }
    if not all(hard.values()):
        raise RuntimeError(f"CW11 selected gate structure failed: {hard}")
    result["cw11_structural_checks"] = hard
    return result


def audit_valid_transitions(
    raw_snapshot: Mapping[str, Any],
    cw10_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for row_index, descriptor in enumerate(
        ADDED_VALID_GUARDS, start=OLD_ROW_COUNT
    ):
        raw_row = raw_snapshot["official_rows"][row_index]
        cw10_row = cw10_snapshot["official_rows"][row_index]
        metrics = [str(value) for value in descriptor["metrics_union"]]
        raw_flags = {metric: raw_row["flags"][metric] for metric in metrics}
        cw10_flags = {metric: cw10_row["flags"][metric] for metric in metrics}
        pair = {
            "row_index": row_index,
            "positive_option": int(descriptor["positive_option"]),
            "negative_option": int(descriptor["negative_option"]),
        }
        checks = {
            "raw_order_exact": list(raw_row["predicted_order"])
            == list(descriptor["expected_raw_order"]),
            "cw10_order_exact": list(cw10_row["predicted_order"])
            == list(descriptor["expected_cw10_order"]),
            "raw_required_metrics_all_true": all(
                value is True for value in raw_flags.values()
            ),
            "cw10_required_metrics_all_false": all(
                value is False for value in cw10_flags.values()
            ),
            "raw_pair_margin_matches_added_threshold_source": (
                cutting_pair_margin_placeholder(raw_snapshot, pair) >= 0.0
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"CW11 valid transition drift: {checks}")
        records.append(
            {
                "row_index": row_index,
                "identity": identity_record(descriptor),
                "specialist_views": list(descriptor["specialist_views"]),
                "metrics_union": metrics,
                "raw_flags": raw_flags,
                "cw10_flags": cw10_flags,
                "raw_order": list(raw_row["predicted_order"]),
                "cw10_order": list(cw10_row["predicted_order"]),
                "raw_pair_margin": cutting_pair_margin_placeholder(
                    raw_snapshot, pair
                ),
                "cw10_pair_margin": cutting_pair_margin_placeholder(
                    cw10_snapshot, pair
                ),
                "checks": checks,
            }
        )
    return records


def cutting_pair_margin_placeholder(
    snapshot: Mapping[str, Any], pair: Mapping[str, Any]
) -> float:
    logits = snapshot["outputs_cpu"]["policy_logits"]
    return float(
        logits[int(pair["row_index"]), int(pair["positive_option"])].float()
        - logits[int(pair["row_index"]), int(pair["negative_option"])].float()
    )


def run_second_stage(
    context: Mapping[str, Any],
    *,
    primary: ModuleType,
    cw9: ModuleType,
    cw10: ModuleType,
    cutting: ModuleType,
    geometry: ModuleType,
    ram: ModuleType,
    user_candidate_consumer: Any = None,
) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    required = {
        "helper",
        "model",
        "checkpoint",
        "model_config",
        "raw_actor",
        "raw_model_state_sha256",
        "raw_nonactor_sha256",
        "terminal_cumulative_float64",
        "terminal_cumulative_float64_le_sha256",
        "success_iteration",
        "selected_row_gate",
        "active_pair_ledger",
        "expanded_row_count",
    }
    if not required.issubset(context):
        raise RuntimeError("CW11 CW10 consumer context schema drift")
    helper = context["helper"]
    torch = helper.torch
    model = context["model"]
    if (
        str(context["raw_model_state_sha256"])
        != EXPECTED_RAW_MODEL_STATE_SHA256
        or str(context["raw_nonactor_sha256"])
        != EXPECTED_RAW_NONACTOR_SHA256
        or str(context["terminal_cumulative_float64_le_sha256"])
        != EXPECTED_CW10_CUMULATIVE_SHA256
        or int(context["success_iteration"]) != EXPECTED_CW10_SUCCESS_ITERATION
        or int(context["expanded_row_count"]) != OLD_ROW_COUNT
        or helper.model_state_sha256(model.state_dict())
        != EXPECTED_CW10_MODEL_STATE_SHA256
    ):
        raise RuntimeError("CW11 exact live CW10 terminal binding failed")
    cumulative = context["terminal_cumulative_float64"].copy()
    if (
        geometry.vector_sha256_float64_le(cumulative, np)
        != EXPECTED_CW10_CUMULATIVE_SHA256
        or not math.isclose(
            float(np.linalg.norm(cumulative)),
            EXPECTED_CW10_CUMULATIVE_L2,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise RuntimeError("CW11 CW10 cumulative vector drift")

    old_descriptors, descriptor_audit = reconstruct_old_descriptors(
        primary, geometry, cw9, cw10
    )
    descriptors = [
        *old_descriptors,
        *[dict(value) for value in ADDED_VALID_GUARDS],
    ]
    descriptor_checks = {
        "rows_exact_33": len(descriptors) == EXPECTED_ROW_COUNT,
        "identities_unique_33": len(
            {identity_key(value) for value in descriptors}
        )
        == EXPECTED_ROW_COUNT,
        "target_rows_exact_5": sum(
            value["role"] == "target" for value in descriptors
        )
        == EXPECTED_TARGET_ROW_COUNT,
        "guard_rows_exact_28": sum(
            value["role"] == "guard" for value in descriptors
        )
        == EXPECTED_GUARD_ROW_COUNT,
        "guard_obligations_exact_84": sum(
            len(value["metrics_union"])
            for value in descriptors
            if value["role"] == "guard"
        )
        == EXPECTED_GUARD_OBLIGATIONS,
    }
    if not all(descriptor_checks.values()):
        raise RuntimeError(f"CW11 descriptor accounting failed: {descriptor_checks}")

    device = next(model.parameters()).device
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CW11 requires CUDA with native BF16")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    parameters = geometry.configure_actor6(model)
    raw_actor = context["raw_actor"]
    if len(parameters) != len(raw_actor):
        raise RuntimeError("CW11 raw actor tensor cardinality drift")
    nonactor_names = sorted(set(model.state_dict()) - set(geometry.ACTOR6_NAMES))
    nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    if nonactor_sha != EXPECTED_RAW_NONACTOR_SHA256:
        raise RuntimeError("CW11 live nonactor SHA drift")

    rows, row_loading_audit = load_selected_train_and_exact_valid_rows(
        geometry, helper, descriptors, context["model_config"]
    )
    cpu_batch = helper.evaluator.collate_ordered(
        [item["features"] for item in rows],
        max_state_entities=int(context["model_config"]["max_state_entities"]),
        entity_fields=int(context["model_config"]["entity_fields"]),
        option_fields=int(context["model_config"]["option_fields"]),
    )
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    model.eval()

    cutting.restore_raw_actor(ram, parameters, raw_actor, torch)
    if helper.model_state_sha256(model.state_dict()) != EXPECTED_RAW_MODEL_STATE_SHA256:
        raise RuntimeError("CW11 temporary raw restoration failed")
    raw_snapshot = ram.snapshot_forward(
        helper, model, batch, cpu_batch, device
    )
    cutting.apply_cumulative_from_raw(
        ram, parameters, raw_actor, cumulative, torch
    )
    if helper.model_state_sha256(model.state_dict()) != EXPECTED_CW10_MODEL_STATE_SHA256:
        raise RuntimeError("CW11 live CW10 reapplication failed")
    current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)

    active_pairs, remap_audit = remap_old_active_pairs(
        cutting, context["active_pair_ledger"], descriptors
    )
    added_initial = append_valid_guards(cutting, active_pairs, raw_snapshot)
    transitions = audit_valid_transitions(raw_snapshot, current)
    patch_audit: dict[str, Any] = {
        "gate_call_count": 0,
        "global_restored_every_call": True,
    }
    current_gate = gate33(
        cutting,
        descriptors,
        active_pairs,
        raw_snapshot,
        current,
        nonactor_sha == EXPECTED_RAW_NONACTOR_SHA256,
        torch,
        patch_audit,
    )
    initial_false = [
        {
            "row_index": int(value["row_index"]),
            "role": str(value["role"]),
            "metric": str(value["metric"]),
        }
        for value in current_gate["false_obligations"]
    ]
    expected_false = [
        {
            "row_index": row_index,
            "role": "guard",
            "metric": metric,
        }
        for row_index, descriptor in enumerate(
            ADDED_VALID_GUARDS, start=OLD_ROW_COUNT
        )
        for metric in descriptor["metrics_union"]
    ]
    initial_checks = {
        "cw10_fails_expanded_gate_before_repair": not bool(current_gate["pass"]),
        "false_obligations_exactly_four_valid_rows_15_cells": initial_false
        == expected_false,
        "old_29_official_rows_remain_pass": all(
            bool(value["pass"])
            for value in current_gate["official_row_gates"][:OLD_ROW_COUNT]
        ),
        "new_four_guard_rows_fail": all(
            not bool(value["pass"])
            for value in current_gate["official_row_gates"][OLD_ROW_COUNT:]
        ),
    }
    if not all(initial_checks.values()):
        raise RuntimeError(f"CW11 initial expanded transition gate failed: {initial_checks}")

    iterations = [
        {
            "iteration": 0,
            "kind": "exact_live_CW10_plus_four_specialist_valid_guards",
            "new_pairs": added_initial,
            "active_pair_ledger": cutting.active_pair_ledger_audit(active_pairs),
            "step_l2": 0.0,
            "cumulative_l2": float(np.linalg.norm(cumulative)),
            "cumulative_float64_le_sha256": geometry.vector_sha256_float64_le(
                cumulative, np
            ),
            "post_gate": current_gate,
        }
    ]
    success_iteration: int | None = None
    close_reason = "maximum_CW11_correction_count_reached_without_pass"
    for iteration in range(1, MAX_ITER + 1):
        pair_update = cutting.add_dynamic_pairs(
            active_pairs,
            current_gate["false_obligations"],
            descriptors,
            raw_snapshot,
            current,
            torch,
            EXPECTED_CW10_SUCCESS_ITERATION + iteration,
        )
        current_nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        pre_gate = gate33(
            cutting,
            descriptors,
            active_pairs,
            raw_snapshot,
            current,
            current_nonactor_sha == EXPECTED_RAW_NONACTOR_SHA256,
            torch,
            patch_audit,
        )
        active_violated = sum(
            not bool(value["pass"])
            for value in pre_gate["active_pair_gates"]
        )
        if not (
            bool(pair_update["all_false_obligations_have_separating_cuts"])
            and active_violated > 0
        ):
            close_reason = "fail_closed_no_violated_separating_cut"
            iterations.append(
                {
                    "iteration": iteration,
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
            raise RuntimeError("CW11 correction trust-region gate failed")
        cumulative = cumulative + correction
        cutting.restore_raw_actor(ram, parameters, raw_actor, torch)
        if helper.model_state_sha256(model.state_dict()) != EXPECTED_RAW_MODEL_STATE_SHA256:
            raise RuntimeError("CW11 per-iteration raw restoration failed")
        cutting.apply_cumulative_from_raw(
            ram, parameters, raw_actor, cumulative, torch
        )
        current = ram.snapshot_forward(helper, model, batch, cpu_batch, device)
        current_nonactor_sha = helper.model_state_sha256(
            {name: model.state_dict()[name] for name in nonactor_names}
        )
        current_gate = gate33(
            cutting,
            descriptors,
            active_pairs,
            raw_snapshot,
            current,
            current_nonactor_sha == EXPECTED_RAW_NONACTOR_SHA256,
            torch,
            patch_audit,
        )
        iterations.append(
            {
                "iteration": iteration,
                "kind": "CW11_minimum_l2_cumulative_correction",
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
                "cumulative_float64_le_sha256": geometry.vector_sha256_float64_le(
                    cumulative, np
                ),
                "post_gate": current_gate,
            }
        )
        if current_gate["pass"]:
            success_iteration = iteration
            close_reason = "all_33_rows_and_all_active_thresholds_pass"
            break

    terminal_l2 = float(np.linalg.norm(cumulative))
    terminal_vector_sha = geometry.vector_sha256_float64_le(cumulative, np)
    terminal_model_sha = helper.model_state_sha256(model.state_dict())
    terminal_nonactor_sha = helper.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    terminal_ledger = cutting.canonical_active_pair_ledger(active_pairs)
    terminal_ledger_sha = sha256_bytes(canonical_json(terminal_ledger))
    terminal_checks = {
        "success_iteration_found": success_iteration is not None,
        "selected_33row_gate_pass": bool(current_gate["pass"]),
        "terminal_vector_sha_hex64": len(terminal_vector_sha) == 64,
        "terminal_model_sha_hex64": len(terminal_model_sha) == 64,
        "terminal_nonactor_exact_raw": terminal_nonactor_sha
        == EXPECTED_RAW_NONACTOR_SHA256,
        "active_pair_count_at_least_34": len(terminal_ledger)
        >= EXPECTED_INITIAL_ACTIVE_COUNT,
        "ledger_sha_self_consistent": terminal_ledger_sha
        == sha256_bytes(canonical_json(terminal_ledger)),
        "cutting_global_restored_every_gate": bool(
            patch_audit["global_restored_every_call"]
        ),
    }
    selected_pass = all(terminal_checks.values())
    user_consumer_called = False
    if selected_pass and user_candidate_consumer is not None:
        user_candidate_consumer(
            {
                "helper": helper,
                "model": model,
                "checkpoint": context["checkpoint"],
                "model_config": dict(context["model_config"]),
                "raw_actor": raw_actor,
                "raw_model_state_sha256": EXPECTED_RAW_MODEL_STATE_SHA256,
                "raw_nonactor_sha256": EXPECTED_RAW_NONACTOR_SHA256,
                "terminal_cumulative_float64": cumulative.copy(),
                "terminal_cumulative_float64_le_sha256": terminal_vector_sha,
                "success_iteration": success_iteration,
                "cw10_success_iteration": EXPECTED_CW10_SUCCESS_ITERATION,
                "selected_row_gate": current_gate,
                "active_pair_ledger": terminal_ledger,
                "expanded_row_count": EXPECTED_ROW_COUNT,
                "candidate_model_state_sha256": terminal_model_sha,
            }
        )
        user_consumer_called = True
    return {
        "status": (
            "exploratory_33row_specialist_valid_CW11_success"
            if selected_pass
            else "closed_no_CW11_candidate"
        ),
        "descriptor_audit": descriptor_audit,
        "descriptor_checks": descriptor_checks,
        "row_loading_audit": row_loading_audit,
        "old_active_pair_remap": remap_audit,
        "valid_transition_audit": transitions,
        "initial_checks": initial_checks,
        "iterations": iterations,
        "decision": {
            "success_iteration_after_CW10": success_iteration,
            "close_reason": close_reason,
            "terminal_cumulative_l2": terminal_l2,
            "terminal_cumulative_float64_le_sha256": terminal_vector_sha,
            "candidate_model_state_sha256_before_CW10_finally_restore": terminal_model_sha,
            "terminal_nonactor_sha256": terminal_nonactor_sha,
            "terminal_active_pair_count": len(terminal_ledger),
            "terminal_active_pair_ledger_sha256": terminal_ledger_sha,
            "terminal_checks": terminal_checks,
            "candidate_consumer_called": user_consumer_called,
            "eligible_for_formal_fulltrain_and_specialist_consistency": selected_pass,
            "model_materialized": False,
            "submission_performed": False,
        },
        "active_pair_contract": {
            "initial_count": EXPECTED_INITIAL_ACTIVE_COUNT,
            "final_count": len(terminal_ledger),
            "old_30_absolute_thresholds_preserved": True,
            "new_valid_guard_thresholds": "same-process raw 33-row pair margins",
            "dynamic_guard_thresholds": "same-process raw 33-row pair margins",
            "final_canonical_ledger_sha256": terminal_ledger_sha,
            "final_canonical_ledger": terminal_ledger,
        },
        "dependency_global_patch_audit": patch_audit,
    }


def run_probe(
    primary: ModuleType,
    primary_source: bytes,
    primary_evidence: Mapping[str, Any],
    *,
    candidate_consumer: Any = None,
) -> dict[str, Any]:
    cw10, cw10_evidence = import_frozen(CW10, CW10_SHA256, "CW11 frozen CW10")
    cw9, cw9_evidence = import_frozen(CW9, CW9_SHA256, "CW11 frozen CW9")
    cutting, cutting_evidence = import_frozen(
        CUTTING, CUTTING_SHA256, "CW11 frozen cutting v2"
    )
    geometry, geometry_evidence = import_frozen(
        GEOMETRY, GEOMETRY_SHA256, "CW11 frozen geometry"
    )
    ram, ram_evidence = import_frozen(RAM, RAM_SHA256, "CW11 frozen RAM runner")
    interface_checks = validate_interfaces(
        cw10, cw9, primary, cutting, geometry, ram
    )
    holder: dict[str, Any] = {}

    def internal_consumer(context: Mapping[str, Any]) -> None:
        if holder:
            raise RuntimeError("CW11 internal consumer called more than once")
        holder.update(
            run_second_stage(
                context,
                primary=primary,
                cw9=cw9,
                cw10=cw10,
                cutting=cutting,
                geometry=geometry,
                ram=ram,
                user_candidate_consumer=candidate_consumer,
            )
        )

    cw10_result = cw10.run_probe(
        primary,
        primary_source,
        primary_evidence,
        candidate_consumer=internal_consumer,
    )
    if not holder:
        raise RuntimeError("CW11 internal consumer was not called")
    outer_checks = {
        "cw10_status_exact": cw10_result.get("status")
        == "exploratory_29row_30pair_optimization_success",
        "cw10_model_sha_exact": cw10_result["decision"]
        ["candidate_model_state_sha256_before_restore"]
        == EXPECTED_CW10_MODEL_STATE_SHA256,
        "cw10_vector_sha_exact": cw10_result["decision"]
        ["terminal_cumulative_float64_le_sha256"]
        == EXPECTED_CW10_CUMULATIVE_SHA256,
        "cw10_ledger_sha_exact": cw10_result["decision"]
        ["terminal_active_pair_ledger_sha256"]
        == EXPECTED_CW10_ACTIVE_LEDGER_SHA256,
        "cw10_finally_raw_restore_pass": bool(cw10_result["final_integrity"]["pass"]),
        "second_stage_status_success": holder.get("status")
        == "exploratory_33row_specialist_valid_CW11_success",
    }
    if not all(outer_checks.values()):
        raise RuntimeError(f"CW11 outer integrity failed: {outer_checks}")
    return {
        "schema_version": SCHEMA,
        "status": holder["status"],
        "scope": {
            "CW10_exact_reconstruction": True,
            "selected_train_rows": OLD_ROW_COUNT,
            "consumed_specialist_valid_rows": ADDED_VALID_GUARD_COUNT,
            "selected_rows_total": EXPECTED_ROW_COUNT,
            "candidate_RAM_only": True,
            "training_optimizer_backward": False,
            "model_or_result_writes": 0,
            "network_upload_submission": False,
            "broad_or_gold_access": False,
            "specialist_valid_contaminated_for_future_promotion": True,
        },
        "input_lock": {
            "cw10": cw10_evidence,
            "cw9": cw9_evidence,
            "primary": dict(primary_evidence),
            "cutting": cutting_evidence,
            "geometry": geometry_evidence,
            "ram": ram_evidence,
        },
        "interface_checks": interface_checks,
        "outer_checks": outer_checks,
        "cw10_reproduction_summary": {
            "status": cw10_result["status"],
            "success_iteration": cw10_result["decision"]["success_iteration_after_c0"],
            "cumulative_l2": cw10_result["decision"]["terminal_cumulative_l2"],
            "vector_sha256": cw10_result["decision"]
            ["terminal_cumulative_float64_le_sha256"],
            "model_state_sha256": cw10_result["decision"]
            ["candidate_model_state_sha256_before_restore"],
            "active_pair_count": cw10_result["active_pair_contract"]["final_count"],
            "active_pair_ledger_sha256": cw10_result["active_pair_contract"]
            ["final_canonical_ledger_sha256"],
            "finally_raw_restore_pass": cw10_result["final_integrity"]["pass"],
        },
        "second_stage": holder,
        "final_integrity": {
            "outer_CW10_finally_raw_restore_pass": cw10_result["final_integrity"]["pass"],
            "dependency_globals_restored": bool(
                holder["dependency_global_patch_audit"]["global_restored_every_call"]
            ),
            "pass": bool(cw10_result["final_integrity"]["pass"])
            and bool(
                holder["dependency_global_patch_audit"]["global_restored_every_call"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    args = parser.parse_args()
    validate_runtime()
    source, self_evidence = read_regular_bytes(
        SCRIPT, None, "CW11 exploratory runner", expected_mode=FROZEN_MODE
    )
    local_static = static_audit(source)
    primary, primary_evidence = import_frozen(
        PRIMARY, PRIMARY_SHA256, "CW11 frozen primary"
    )
    primary_source, _ = read_regular_bytes(
        PRIMARY, PRIMARY_SHA256, "CW11 frozen primary source", expected_mode=FROZEN_MODE
    )
    cw10, cw10_evidence = import_frozen(CW10, CW10_SHA256, "CW11 static CW10")
    cw9, cw9_evidence = import_frozen(CW9, CW9_SHA256, "CW11 static CW9")
    cutting, cutting_evidence = import_frozen(
        CUTTING, CUTTING_SHA256, "CW11 static cutting"
    )
    geometry, geometry_evidence = import_frozen(
        GEOMETRY, GEOMETRY_SHA256, "CW11 static geometry"
    )
    ram, ram_evidence = import_frozen(RAM, RAM_SHA256, "CW11 static RAM"
    )
    _, fulltrain_v5_evidence = read_regular_bytes(
        FULLTRAIN_V5,
        FULLTRAIN_V5_SHA256,
        "CW11 frozen fulltrain v5 anchor",
        expected_mode=FROZEN_MODE,
    )
    report_payload, report_evidence = read_regular_bytes(
        FLIP_REPORT,
        FLIP_REPORT_SHA256,
        "CW11 frozen specialist flip report",
        expected_mode=0o444,
    )
    report_audit = validate_flip_report(report_payload)
    interface_checks = validate_interfaces(
        cw10, cw9, primary, cutting, geometry, ram
    )
    if args.mode == "static":
        result = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": self_evidence,
            "dependencies": {
                "cw10": cw10_evidence,
                "cw9": cw9_evidence,
                "primary": primary_evidence,
                "cutting": cutting_evidence,
                "geometry": geometry_evidence,
                "ram": ram_evidence,
                "fulltrain_v5_anchor": fulltrain_v5_evidence,
                "specialist_flip_report": report_evidence,
            },
            "contract": {
                "rows": EXPECTED_ROW_COUNT,
                "old_train_rows": OLD_ROW_COUNT,
                "added_consumed_valid_guards": ADDED_VALID_GUARD_COUNT,
                "target_rows": EXPECTED_TARGET_ROW_COUNT,
                "guard_rows": EXPECTED_GUARD_ROW_COUNT,
                "target_obligations": EXPECTED_TARGET_OBLIGATIONS,
                "guard_obligations": EXPECTED_GUARD_OBLIGATIONS,
                "initial_active_pairs": EXPECTED_INITIAL_ACTIVE_COUNT,
                "max_corrections": MAX_ITER,
                "step_l2_cap": STEP_L2_CAP,
            },
            "added_valid_guard_spec_sha256": sha256_bytes(
                canonical_json(ADDED_VALID_GUARDS)
            ),
            "audit": {
                "local": local_static,
                "interfaces": interface_checks,
                "flip_report": report_audit,
            },
            "run_executed": False,
            "cuda_accessed": False,
            "writes_performed": False,
        }
    else:
        result = run_probe(
            primary,
            primary_source,
            primary_evidence,
        )
        result["input_lock"]["self"] = self_evidence
        result["input_lock"]["fulltrain_v5_anchor"] = fulltrain_v5_evidence
        result["input_lock"]["specialist_flip_report"] = report_evidence
        result["local_static_audit"] = local_static
        result["flip_report_audit"] = report_audit
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
