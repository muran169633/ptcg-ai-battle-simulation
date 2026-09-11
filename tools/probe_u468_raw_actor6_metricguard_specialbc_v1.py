#!/usr/bin/env python3
"""Zero-write selected-row actor6 metric-guard geometry probe for raw U468.

The probe reads five frozen Pokemon Fan common-WC target rows and derives one
guard per distinct correct-to-wrong row from the hash-bound formal v3 result.
Every scalar is only the policy-logit margin at the first ranking stage where
the frozen raw and threat orders differ; later/unrelated ordering stages are
not constrained.  It performs one native CUDA BF16 forward, differentiates
the FP32 margins with ``torch.autograd.grad``, and solves one preregistered
closure-rate problem in the 19-gradient row space: minimize the slowest of the
five linear radii needed to reach each row's first positive BF16 margin while
holding every normalized guard cosine at or above 0.10.

Both modes are stdout-only.  There is no optimizer, backward call, parameter
step, checkpoint serialization, result publication, full-panel evaluation,
validation-member access, or network action.
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
SCRIPT = TOOLS / "probe_u468_raw_actor6_metricguard_specialbc_v1.py"
SCHEMA = "ptcg-u468-raw-actor6-metricguard-specialbc-geometry-v2"
SEED = 202608202

PARENT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
PARENT_FILE_SHA256 = (
    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
)
PARENT_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)

FORMAL_RESULT = ROOT / (
    "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_"
    "design202608122.result.json"
)
FORMAL_RESULT_SHA256 = (
    "d43a1242b0ad28d71544e7e2b4112ce1559818e9e6fb70b8666efa439e91322e"
)
FORMAL_RESULT_SCHEMA = (
    "ptcg-u468-raw-equalblend-ray-threshold-probe-result-v2"
)

HELPER = TOOLS / "run_u468_raw_balanced_train_only_endpoint_gate.py"
HELPER_SHA256 = (
    "84b51d8ef51e2184271c4bf358e5d59b74ea631daa79b84d73fffaf44a37037d"
)
DEPENDENCIES = {
    TOOLS / "evaluate_policy_bc.py": (
        "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4"
    ),
    TOOLS / "train_ppo.py": (
        "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3"
    ),
    TOOLS / "train_bc_orbit.py": (
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
    ),
}

DATASETS = {
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "core5": ROOT
    / "data/bc_gold19_exact_core5_balanced20_seed202608017_20260801.zip",
}
DATA_SHA256 = {
    "flg": "4cdb18faa91084606476dc4c03fbd8fe92fb35e4459628035c800bd611bac4e8",
    "pokemonfan": "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598",
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
EXPECTED_ACTOR6_FLAT_LENGTH = 65793
ALPHAS = (2, 4, 8, 16, 32, 64, 128, 256)
PANEL_SCAN_ORDER = ("flg", "pokemonfan", "core5")
SAFETY_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
TARGET_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)

TARGET_SPECS = (
    {
        "line_index_zero_based": 823,
        "line_sha256": "130403ae3ffedadf68d42a31fc972fa05d65f7012301367f043c15af037023e4",
        "target_option": 7,
        "raw_option": 13,
        "expected_raw_margin": -0.00390625,
        "expected_positive_bf16_margin": 0.001953125,
        "required_positive_margin": 0.005859375,
    },
    {
        "line_index_zero_based": 1179,
        "line_sha256": "3b8e123d82ea6b8233d3d6088830527ec635f0ac363e72bb95bbd8a99d99cd21",
        "target_option": 4,
        "raw_option": 18,
        "expected_raw_margin": -0.015625,
        "expected_positive_bf16_margin": 0.015625,
        "required_positive_margin": 0.03125,
    },
    {
        "line_index_zero_based": 3259,
        "line_sha256": "d456bafc089282ed4053d9ad1d0bb795be0366c3256f6d3992fea6937fadfb50",
        "target_option": 3,
        "raw_option": 2,
        "expected_raw_margin": 0.0,
        "expected_positive_bf16_margin": 0.00390625,
        "required_positive_margin": 0.00390625,
    },
    {
        "line_index_zero_based": 4753,
        "line_sha256": "7ee1eff087e34b39c039d3b043b595dc6d605681e41514a06d7d0886df4b607c",
        "target_option": 0,
        "raw_option": 3,
        "expected_raw_margin": -0.015625,
        "expected_positive_bf16_margin": 0.001953125,
        "required_positive_margin": 0.017578125,
    },
    {
        "line_index_zero_based": 6250,
        "line_sha256": "a35ef5e40dbded01968854707764cfb7745ffd2d0677677d992508989c10c232",
        "target_option": 0,
        "raw_option": 2,
        "expected_raw_margin": -0.015625,
        "expected_positive_bf16_margin": 0.0078125,
        "required_positive_margin": 0.0234375,
    },
)
TARGET_MEMBER = "train/part-00000.jsonl"

EXPECTED_TARGET_COUNT = 5
EXPECTED_GUARD_COUNT = 14
EXPECTED_ROW_COUNT = 19
EXPECTED_DIRECTION_SHA256 = (
    "e4714580607e4c9543c026fc4851326526b33612ca9ed39818122e3ec3cffd84"
)
EXPECTED_TARGET_DIRECTIONAL_DERIVATIVES = (
    1.5291985025924653,
    8.155725347159827,
    1.0194656683949792,
    4.5875955077774035,
    6.116794010369868,
)
EXPECTED_BINDING_RADIUS = 0.0038316640972813824
EXPECTED_MIN_GUARD_COSINE = 0.09999999999999956
EXPECTED_SCALED_SOLVER_T = 1.0194656683949779
EXPECTED_ACTIVE_DUAL_MIN = 0.014094132319374111
CLOSURE_RATE_SCALE = 256.0
GUARD_COSINE_FLOOR = 0.10
SVD_RELATIVE_RANK_TOLERANCE = 1e-12
SOLVER_FTOL = 1e-12
SOLVER_MAXITER = 5000
EXPECTED_SOLVER_ITERATIONS = 1
NORM_RESIDUAL_TOLERANCE = 1e-10
DIRECT_ABSOLUTE_TOLERANCE = 1e-10
DIRECT_RADIUS_TOLERANCE = 1e-12
GUARD_FLOOR_TOLERANCE = 1e-12


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


def strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise RuntimeError(f"{label} has forbidden JSON constant {value}")

    parsed = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} root is not an object")
    return parsed


def read_regular_bytes(path: Path, expected_sha256: str, label: str) -> tuple[bytes, dict[str, Any]]:
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
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(stat.S_IMODE(after.st_mode)),
        "single_link_regular_held_fd_identity_exact": True,
    }


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
    }
    forbidden_names = {"open", "exec", "eval", "compile", "DataLoader"}
    forbidden_os_flags = {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
    forbidden_import_roots = {"requests", "urllib", "subprocess"}
    attribute_hits: list[tuple[int, str]] = []
    name_hits: list[tuple[int, str]] = []
    autograd_grad_sites: list[int] = []
    print_sites: list[int] = []
    os_open_sites: list[int] = []
    os_flag_hits: list[tuple[int, str]] = []
    import_hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root in forbidden_import_roots:
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
                isinstance(function.value, ast.Attribute)
                and isinstance(function.value.value, ast.Name)
                and function.value.value.id == "torch"
                and function.value.attr == "autograd"
                and function.attr == "grad"
            ):
                autograd_grad_sites.append(node.lineno)
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
    if len(autograd_grad_sites) != 1 or len(print_sites) != 1 or len(os_open_sites) != 1:
        raise RuntimeError("static call-site cardinality drift")
    return {
        "ast_parse": True,
        "forbidden_attribute_call_sites": attribute_hits,
        "forbidden_name_call_sites": name_hits,
        "forbidden_os_write_flags": os_flag_hits,
        "forbidden_network_imports": import_hits,
        "torch_autograd_grad_call_sites": autograd_grad_sites,
        "stdout_print_call_sites": print_sites,
        "os_open_read_only_call_sites": os_open_sites,
        "no_optimizer_backward_step_save_or_write": True,
        "no_full_panel_dataloader": True,
        "no_network_modules": True,
    }


def load_helper() -> tuple[ModuleType, dict[str, Any]]:
    dependency_evidence: dict[str, Any] = {}
    for path, digest in DEPENDENCIES.items():
        _, evidence = read_regular_bytes(path, digest, path.name)
        dependency_evidence[path.name] = evidence
    _, helper_evidence = read_regular_bytes(HELPER, HELPER_SHA256, "geometry helper")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("u468_metricguard_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct helper import spec")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, {"helper": helper_evidence, "dependencies": dependency_evidence}


def identity_key(record: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(record["panel"]),
        str(record["member"]),
        int(record["line_index_zero_based"]),
        str(record["line_sha256"]),
    )


def first_different_stage(raw_order: Sequence[int], threat_order: Sequence[int]) -> int:
    if len(raw_order) != len(threat_order) or not raw_order:
        raise RuntimeError("ranking-stage guard requires equal nonempty order lengths")
    for stage, (raw_option, threat_option) in enumerate(zip(raw_order, threat_order)):
        if int(raw_option) != int(threat_option):
            return stage
    raise RuntimeError("CW record has no different ranking stage")


def extract_targets(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel = result["evaluations"]["alpha128"]["pokemonfan"]
    per_metric: dict[str, dict[int, dict[str, Any]]] = {}
    for metric in TARGET_METRICS:
        expanded = panel["raw_transition_evidence"][metric]["cells"]["wc"].get(
            "expanded_full_line_identities_and_orders", []
        )
        per_metric[metric] = {
            int(record["line_index_zero_based"]): record for record in expanded
        }
    targets: list[dict[str, Any]] = []
    for ordinal, frozen in enumerate(TARGET_SPECS):
        line_index = int(frozen["line_index_zero_based"])
        records = [per_metric[metric].get(line_index) for metric in TARGET_METRICS]
        if any(record is None for record in records):
            raise RuntimeError(f"target {line_index} is not alpha128 common-WC")
        first = records[0]
        assert first is not None
        if any(record != first for record in records[1:]):
            raise RuntimeError(f"target {line_index} identity/order differs by metric")
        if (
            first["panel"] != "pokemonfan"
            or first["member"] != TARGET_MEMBER
            or first["line_sha256"] != frozen["line_sha256"]
            or first["raw_order"] != [frozen["raw_option"]]
            or first["candidate_order"] != [frozen["target_option"]]
        ):
            raise RuntimeError(f"target {line_index} frozen identity/pair drift")
        targets.append(
            {
                "role": "target",
                "ordinal": ordinal,
                "panel": "pokemonfan",
                "member": TARGET_MEMBER,
                "line_index_zero_based": line_index,
                "line_sha256": frozen["line_sha256"],
                "metrics_intersection": list(TARGET_METRICS),
                "first_different_ranking_stage": 0,
                "positive_option": int(frozen["target_option"]),
                "negative_option": int(frozen["raw_option"]),
                "margin_semantics": "expert_option_minus_raw_top1_option",
                "expected_raw_margin": float(frozen["expected_raw_margin"]),
                "expected_positive_bf16_margin": float(
                    frozen["expected_positive_bf16_margin"]
                ),
                "required_positive_margin": float(
                    frozen["required_positive_margin"]
                ),
                "formal_raw_order": list(first["raw_order"]),
                "formal_threat_order": list(first["candidate_order"]),
            }
        )
    return targets


def extract_guards(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    guards: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for alpha in ALPHAS:
        model_name = f"alpha{alpha}"
        for panel in PANEL_SCAN_ORDER:
            evidence = result["evaluations"][model_name][panel][
                "raw_transition_evidence"
            ]
            for metric in SAFETY_METRICS:
                cell = evidence[metric]["cells"]["cw"]
                expanded = cell.get("expanded_full_line_identities_and_orders", [])
                if int(cell["count"]) != len(expanded):
                    raise RuntimeError(f"{model_name}/{panel}/{metric} CW expansion drift")
                for record in expanded:
                    if record["panel"] != panel:
                        raise RuntimeError("CW panel identity drift")
                    key = identity_key(record)
                    occurrence = {
                        "alpha": alpha,
                        "metric": metric,
                        "raw_order": [int(value) for value in record["raw_order"]],
                        "candidate_order": [
                            int(value) for value in record["candidate_order"]
                        ],
                    }
                    if key not in guards:
                        stage = first_different_stage(
                            occurrence["raw_order"], occurrence["candidate_order"]
                        )
                        guards[key] = {
                            "role": "guard",
                            "panel": key[0],
                            "member": key[1],
                            "line_index_zero_based": key[2],
                            "line_sha256": key[3],
                            "context": int(record["context"]),
                            "first_alpha": alpha,
                            "first_metric": metric,
                            "metrics_union": [metric],
                            "first_different_ranking_stage": stage,
                            "positive_option": occurrence["raw_order"][stage],
                            "negative_option": occurrence["candidate_order"][stage],
                            "margin_semantics": (
                                "raw_safe_option_minus_observed_threat_at_"
                                "first_different_metric_local_ranking_stage"
                            ),
                            "formal_raw_order": occurrence["raw_order"],
                            "formal_threat_order": occurrence["candidate_order"],
                            "occurrences": [occurrence],
                        }
                    else:
                        guard = guards[key]
                        if metric not in guard["metrics_union"]:
                            guard["metrics_union"].append(metric)
                        guard["occurrences"].append(occurrence)
    ordered = sorted(
        guards.values(),
        key=lambda item: (
            item["panel"],
            item["member"],
            item["line_index_zero_based"],
            item["line_sha256"],
        ),
    )
    if len(ordered) != EXPECTED_GUARD_COUNT:
        raise RuntimeError(f"expected 14 distinct CW guards, observed {len(ordered)}")
    for ordinal, guard in enumerate(ordered):
        guard["ordinal"] = ordinal
    return ordered


def load_selected_rows(
    helper: ModuleType,
    descriptors: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {}
    for descriptor in descriptors:
        panel = str(descriptor["panel"])
        member = str(descriptor["member"])
        index = int(descriptor["line_index_zero_based"])
        target = requested.setdefault(panel, {}).setdefault(member, {})
        if index in target:
            raise RuntimeError("target/guard selected-row identity overlap")
        target[index] = descriptor

    loaded: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    archive_evidence: dict[str, Any] = {}
    for panel in PANEL_SCAN_ORDER:
        if panel not in requested:
            continue
        payload, evidence = read_regular_bytes(
            DATASETS[panel], DATA_SHA256[panel], f"{panel} train archive"
        )
        opened_members: list[str] = []
        lines_scanned = 0
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise RuntimeError(f"{panel} archive has duplicate member names")
            for member, index_map in sorted(requested[panel].items()):
                if not member.startswith("train/") or not member.endswith(".jsonl"):
                    raise RuntimeError(f"refusing non-train member {member}")
                if member not in names:
                    raise RuntimeError(f"missing selected train member {member}")
                opened_members.append(member)
                remaining = set(index_map)
                maximum = max(remaining)
                with archive.open(member, "r") as handle:
                    for line_index, raw_line in enumerate(handle):
                        lines_scanned += 1
                        if line_index in remaining:
                            descriptor = index_map[line_index]
                            digest = sha256_bytes(raw_line)
                            if digest != descriptor["line_sha256"]:
                                raise RuntimeError(
                                    f"selected row SHA drift at {panel}/{member}:{line_index}"
                                )
                            row = helper.orjson.loads(raw_line)
                            if not isinstance(row, dict) or row.get("split") != "train":
                                raise RuntimeError("selected row is not a train object")
                            raw_action = row.get("action")
                            if not isinstance(raw_action, list):
                                raise RuntimeError("selected row action is not a list")
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
                                or any(value < 0 or value >= len(options) for value in expert_order)
                            ):
                                raise RuntimeError("selected row is not evaluable")
                            features["expert_action_order"] = expert_order
                            key = identity_key(descriptor)
                            loaded[key] = {
                                "features": features,
                                "raw_metadata": {
                                    "episode_id": row.get("episode_id"),
                                    "observation_step_index": row.get(
                                        "observation_step_index"
                                    ),
                                    "team_name": row.get("team_name"),
                                    "context": int(features["context"]),
                                    "expert_order": expert_order,
                                    "option_count": len(options),
                                },
                            }
                            remaining.remove(line_index)
                        if line_index >= maximum:
                            break
                if remaining:
                    raise RuntimeError(f"selected row indices missing in {member}: {remaining}")
        archive_evidence[panel] = {
            **evidence,
            "opened_members": opened_members,
            "opened_members_train_only": True,
            "selected_rows": sum(len(value) for value in requested[panel].values()),
            "lines_scanned_until_last_selected_row": lines_scanned,
            "validation_member_payloads_opened": False,
        }

    rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        key = identity_key(descriptor)
        item = loaded.get(key)
        if item is None:
            raise RuntimeError(f"selected row not loaded: {key}")
        rows.append(item)
    if len(rows) != EXPECTED_ROW_COUNT or len(loaded) != EXPECTED_ROW_COUNT:
        raise RuntimeError("selected-row cardinality drift")
    return rows, archive_evidence


def configure_actor6(model: Any) -> list[Any]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters: list[Any] = []
    for name in ACTOR6_NAMES:
        if name not in named:
            raise RuntimeError(f"missing actor6 parameter {name}")
        named[name].requires_grad_(True)
        parameters.append(named[name])
    observed = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if observed != ACTOR6_NAMES:
        raise RuntimeError(f"actor6 trainable scope drift: {observed}")
    if sum(parameter.numel() for parameter in parameters) != EXPECTED_ACTOR6_FLAT_LENGTH:
        raise RuntimeError("actor6 flattened length drift")
    return parameters


def vector_sha256_float64_le(vector: Any, np: Any) -> str:
    payload = np.ascontiguousarray(vector.astype("<f8")).tobytes()
    return sha256_bytes(payload)


def gradient_for_margin(
    margin: Any,
    parameters: Sequence[Any],
    torch: Any,
    *,
    retain_graph: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    values = torch.autograd.grad(
        margin,
        tuple(parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    flat_parts = []
    tensor_records: list[dict[str, Any]] = []
    for name, value in zip(ACTOR6_NAMES, values):
        cpu_float = value.detach().float().cpu().contiguous()
        payload = cpu_float.reshape(-1).view(torch.uint8).numpy().tobytes()
        flat_parts.append(cpu_float.reshape(-1))
        tensor_records.append(
            {
                "name": name,
                "shape": [int(item) for item in cpu_float.shape],
                "dtype_before_float64_concat": str(cpu_float.dtype),
                "float32_native_endian_sha256": sha256_bytes(payload),
                "l2": float(cpu_float.double().norm()),
            }
        )
    flat = torch.cat(flat_parts).double().numpy()
    return flat, tensor_records


def solve_closure_rate_direction(
    A: Any,
    target_gradient_norms: Any,
    target_required_positive_margins: Any,
    np: Any,
    optimize: Any,
) -> tuple[Any, dict[str, Any]]:
    if A.shape != (EXPECTED_ROW_COUNT, EXPECTED_ACTOR6_FLAT_LENGTH):
        raise RuntimeError(f"normalized gradient matrix shape drift: {A.shape}")
    expected_required = np.asarray(
        [float(item["required_positive_margin"]) for item in TARGET_SPECS],
        dtype=np.float64,
    )
    required = np.asarray(target_required_positive_margins, dtype=np.float64)
    gradient_norms = np.asarray(target_gradient_norms, dtype=np.float64)
    if (
        required.shape != (EXPECTED_TARGET_COUNT,)
        or gradient_norms.shape != (EXPECTED_TARGET_COUNT,)
        or not np.array_equal(required, expected_required)
        or not bool(np.all(np.isfinite(gradient_norms)))
        or not bool(np.all(gradient_norms > 0.0))
    ):
        raise RuntimeError("target closure-rate inputs drifted")

    u, singular_values, vh = np.linalg.svd(A, full_matrices=False)
    rank = int((singular_values > singular_values[0] * SVD_RELATIVE_RANK_TOLERANCE).sum())
    if rank != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"expected full 19-row gradient rank, observed {rank}")
    C = u[:, :rank] * singular_values[:rank]
    target_closure_matrix = C[:EXPECTED_TARGET_COUNT] * (
        gradient_norms / required
    )[:, None]
    scaled_target_matrix = target_closure_matrix / CLOSURE_RATE_SCALE
    guard_matrix = C[EXPECTED_TARGET_COUNT:]

    # The audited optimum has all five target and all fourteen guard
    # constraints active.  Construct that boundary point without using an
    # endpoint, then require nonnegative KKT multipliers and an independent
    # SLSQP success before accepting it.
    active_matrix = np.vstack([scaled_target_matrix, guard_matrix])
    target_basis = np.linalg.solve(
        active_matrix,
        np.r_[np.ones(EXPECTED_TARGET_COUNT), np.zeros(EXPECTED_GUARD_COUNT)],
    )
    guard_basis = np.linalg.solve(
        active_matrix,
        np.r_[
            np.zeros(EXPECTED_TARGET_COUNT),
            np.full(EXPECTED_GUARD_COUNT, GUARD_COSINE_FLOOR),
        ],
    )
    quadratic = (
        float(target_basis @ target_basis),
        float(2.0 * (target_basis @ guard_basis)),
        float(guard_basis @ guard_basis - 1.0),
    )
    quadratic_roots = np.roots(quadratic)
    positive_real_roots = [
        float(value.real)
        for value in quadratic_roots
        if abs(float(value.imag)) <= DIRECT_ABSOLUTE_TOLERANCE
        and float(value.real) > 0.0
    ]
    if len(positive_real_roots) != 1:
        raise RuntimeError(f"closure boundary root drift: {quadratic_roots}")
    closed_form_t = positive_real_roots[0]
    closed_form_y = closed_form_t * target_basis + guard_basis
    closed_form_norm_before = float(np.linalg.norm(closed_form_y))
    closed_form_y /= closed_form_norm_before
    closed_form_t = float((scaled_target_matrix @ closed_form_y).min())
    x0 = np.r_[closed_form_y, closed_form_t]

    def objective(x: Any) -> float:
        return -float(x[-1])

    def objective_jacobian(x: Any) -> Any:
        value = np.zeros_like(x)
        value[-1] = -1.0
        return value

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: scaled_target_matrix @ x[:-1] - x[-1],
            "jac": lambda x: np.c_[
                scaled_target_matrix,
                -np.ones(scaled_target_matrix.shape[0]),
            ],
        },
        {
            "type": "ineq",
            "fun": lambda x: guard_matrix @ x[:-1] - GUARD_COSINE_FLOOR,
            "jac": lambda x: np.c_[
                guard_matrix,
                np.zeros(guard_matrix.shape[0]),
            ],
        },
        {
            "type": "ineq",
            "fun": lambda x: 1.0 - float(x[:-1] @ x[:-1]),
            "jac": lambda x: np.r_[-2.0 * x[:-1], 0.0],
        },
    ]
    solution = optimize.minimize(
        objective,
        x0,
        jac=objective_jacobian,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": SOLVER_FTOL, "maxiter": SOLVER_MAXITER, "disp": False},
    )
    if not bool(solution.success):
        raise RuntimeError(f"SLSQP failed: {solution.status} {solution.message}")
    direction = vh[:rank].T @ solution.x[:-1]
    direction /= np.linalg.norm(direction)
    direct_target_cosines = A[:EXPECTED_TARGET_COUNT] @ direction
    direct_guard_cosines = A[EXPECTED_TARGET_COUNT:] @ direction
    direct_target_derivatives = direct_target_cosines * gradient_norms
    direct_closure_rates = direct_target_derivatives / required
    direct_positive_radii = required / direct_target_derivatives
    direct_scaled_closure_rates = direct_closure_rates / CLOSURE_RATE_SCALE
    binding_radius = float(direct_positive_radii.max())
    norm_residual = abs(float(np.linalg.norm(direction)) - 1.0)

    dual_basis = np.linalg.solve(active_matrix.T, solution.x[:-1])
    target_dual_denominator = float(dual_basis[:EXPECTED_TARGET_COUNT].sum())
    if not math.isfinite(target_dual_denominator) or target_dual_denominator <= 0.0:
        raise RuntimeError("closure-rate KKT dual normalization failed")
    active_duals = dual_basis / target_dual_denominator
    active_dual_min = float(active_duals.min())

    expected_derivatives = np.asarray(
        EXPECTED_TARGET_DIRECTIONAL_DERIVATIVES, dtype=np.float64
    )
    if (
        int(solution.nit) != EXPECTED_SOLVER_ITERATIONS
        or norm_residual > NORM_RESIDUAL_TOLERANCE
        or not bool(np.all(direct_target_derivatives > 0.0))
        or float(direct_guard_cosines.min())
        < GUARD_COSINE_FLOOR - GUARD_FLOOR_TOLERANCE
        or float(np.max(np.abs(direct_guard_cosines - GUARD_COSINE_FLOOR)))
        > DIRECT_ABSOLUTE_TOLERANCE
        or not bool(
            np.allclose(
                direct_target_derivatives,
                expected_derivatives,
                rtol=0.0,
                atol=DIRECT_ABSOLUTE_TOLERANCE,
            )
        )
        or float(np.ptp(direct_closure_rates)) > DIRECT_ABSOLUTE_TOLERANCE
        or float(np.ptp(direct_positive_radii)) > DIRECT_RADIUS_TOLERANCE
        or abs(binding_radius - EXPECTED_BINDING_RADIUS)
        > DIRECT_RADIUS_TOLERANCE
        or abs(float(direct_guard_cosines.min()) - EXPECTED_MIN_GUARD_COSINE)
        > GUARD_FLOOR_TOLERANCE
        or abs(float(solution.x[-1]) - EXPECTED_SCALED_SOLVER_T)
        > DIRECT_ABSOLUTE_TOLERANCE
        or abs(float(direct_scaled_closure_rates.min()) - float(solution.x[-1]))
        > DIRECT_ABSOLUTE_TOLERANCE
        or active_dual_min < 0.0
        or abs(active_dual_min - EXPECTED_ACTIVE_DUAL_MIN)
        > DIRECT_ABSOLUTE_TOLERANCE
    ):
        raise RuntimeError("direct closure-rate geometry hard gate failed")
    direction_sha256 = vector_sha256_float64_le(direction, np)
    if direction_sha256 != EXPECTED_DIRECTION_SHA256:
        raise RuntimeError(f"unit actor6 direction SHA drift: {direction_sha256}")
    return direction, {
        "formulation": (
            "maximize the minimum of five target directional derivatives divided "
            "by exact required positive-BF16 margins, subject to fourteen "
            "normalized guard cosines >=0.10 and unit actor6 L2"
        ),
        "pre_endpoint_selection_rule": (
            "before any RAM endpoint exists, choose the unique unit direction "
            "with the smallest worst five-target linear positive-BF16 closure "
            "radius in the guard-cosine>=0.10 feasible set"
        ),
        "endpoint_results_used_for_direction_selection": False,
        "target_required_positive_margins": [float(value) for value in required],
        "closure_rate_scale_for_solver_conditioning": CLOSURE_RATE_SCALE,
        "guard_cosine_floor": GUARD_COSINE_FLOOR,
        "svd": {
            "implementation": "numpy.linalg.svd(A, full_matrices=False)",
            "relative_rank_tolerance": SVD_RELATIVE_RANK_TOLERANCE,
            "rank": rank,
            "singular_values": [float(value) for value in singular_values],
        },
        "all_active_closed_form_initialization": {
            "active_matrix_shape": list(active_matrix.shape),
            "active_matrix_condition_number": float(np.linalg.cond(active_matrix)),
            "quadratic_coefficients": [float(value) for value in quadratic],
            "quadratic_roots": [
                {"real": float(value.real), "imag": float(value.imag)}
                for value in quadratic_roots
            ],
            "selected_positive_scaled_closure_root": closed_form_t,
            "y_l2_before_normalization": closed_form_norm_before,
            "used_as_slsqp_initial_point_only": True,
        },
        "solver": {
            "implementation": "scipy.optimize.minimize/SLSQP",
            "ftol": SOLVER_FTOL,
            "maxiter": SOLVER_MAXITER,
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "iterations": int(solution.nit),
            "function_evaluations": int(solution.nfev),
            "jacobian_evaluations": int(solution.njev),
            "reported_scaled_closure_rate_t": float(solution.x[-1]),
            "direct_recomputed_constraints_are_authority": True,
        },
        "direction": {
            "length": int(direction.size),
            "dtype": "little-endian float64",
            "l2": float(np.linalg.norm(direction)),
            "norm_residual": norm_residual,
            "sha256_preimage": "np.ascontiguousarray(d.astype('<f8')).tobytes()",
            "sha256": direction_sha256,
        },
        "direct_target_cosines": [float(value) for value in direct_target_cosines],
        "direct_target_directional_derivatives": [
            float(value) for value in direct_target_derivatives
        ],
        "direct_target_closure_rates": [float(value) for value in direct_closure_rates],
        "direct_target_positive_bf16_linear_radii": [
            float(value) for value in direct_positive_radii
        ],
        "direct_binding_radius": binding_radius,
        "direct_guard_cosines": [float(value) for value in direct_guard_cosines],
        "direct_min_guard_cosine": float(direct_guard_cosines.min()),
        "direct_guard_floor_residual_min": float(
            (direct_guard_cosines - GUARD_COSINE_FLOOR).min()
        ),
        "all_target_derivatives_strictly_positive": True,
        "all_guard_cosines_at_floor_within_tolerance": True,
        "kkt": {
            "all_nineteen_constraints_active": True,
            "active_duals": [float(value) for value in active_duals],
            "active_dual_min": active_dual_min,
            "all_active_duals_nonnegative": True,
        },
    }


def run_probe(source: bytes, static: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy import optimize

    helper, dependency_evidence = load_helper()
    torch = helper.torch
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("requires CUDA with native BF16 support")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")

    result_payload, result_evidence = read_regular_bytes(
        FORMAL_RESULT, FORMAL_RESULT_SHA256, "formal v3 ray result"
    )
    formal = strict_json_bytes(result_payload, "formal v3 ray result")
    if (
        formal.get("schema_version") != FORMAL_RESULT_SCHEMA
        or formal.get("status") != "closed_no_candidate"
        or formal.get("decision", {}).get("status") != "closed_no_candidate"
        or formal.get("decision", {}).get("selected_candidate") is not None
        or formal.get("execution", {}).get("evaluation_count_exact") != 30
    ):
        raise RuntimeError("formal v3 result schema/status drift")
    targets = extract_targets(formal)
    guards = extract_guards(formal)
    descriptors = [*targets, *guards]
    if len(descriptors) != EXPECTED_ROW_COUNT or len({identity_key(x) for x in descriptors}) != EXPECTED_ROW_COUNT:
        raise RuntimeError("19-row target/guard identity cardinality drift")

    parent_payload, parent_evidence = read_regular_bytes(
        PARENT, PARENT_FILE_SHA256, "raw U468 checkpoint"
    )
    checkpoint = torch.load(io.BytesIO(parent_payload), map_location="cpu", weights_only=False)
    if int(checkpoint.get("update", -1)) != 468:
        raise RuntimeError("raw parent update drift")
    model, model_config, kind = helper.instantiate_checkpoint(checkpoint, device)
    if kind != "ppo" or helper.model_state_sha256(model.state_dict()) != PARENT_MODEL_STATE_SHA256:
        raise RuntimeError("raw U468 model construction/hash drift")
    state_hash_before = helper.model_state_sha256(model.state_dict())
    parameters = configure_actor6(model)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("actor6 .grad buffer existed before probe")

    selected_rows, archive_evidence = load_selected_rows(
        helper, descriptors, model_config
    )
    features = [item["features"] for item in selected_rows]
    cpu_batch = helper.evaluator.collate_ordered(
        features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    model.eval()
    outputs = helper.ppo.model_forward(model, batch, device)
    if set(outputs) != {"policy_logits", "count_logits", "value_logits"}:
        raise RuntimeError("raw forward output key drift")
    if any(value.dtype != torch.bfloat16 for value in outputs.values()):
        raise RuntimeError("forward outputs are not native CUDA BF16")
    with torch.no_grad():
        detached = {key: value.detach() for key, value in outputs.items()}
        raw_orders, _, _, _ = helper.ppo.sample_ordered_actions(
            detached, batch, deterministic=True, canonicalize_order=False
        )
    if len(raw_orders) != EXPECTED_ROW_COUNT:
        raise RuntimeError("official raw prediction row count drift")

    native_logits = outputs["policy_logits"]
    logits = native_logits.float()
    margins = []
    target_positive_bf16_margins: list[float] = []
    target_required_positive_margins: list[float] = []
    for row_index, (descriptor, selected, raw_order) in enumerate(
        zip(descriptors, selected_rows, raw_orders)
    ):
        expected_raw_order = descriptor["formal_raw_order"]
        if [int(value) for value in raw_order] != expected_raw_order:
            raise RuntimeError(f"formal/live raw order drift at selected row {row_index}")
        metadata = selected["raw_metadata"]
        if descriptor["role"] == "target":
            if metadata["expert_order"] != [descriptor["positive_option"]]:
                raise RuntimeError("target is not the frozen single-action expert row")
        positive = int(descriptor["positive_option"])
        negative = int(descriptor["negative_option"])
        if not bool(cpu_batch["option_mask"][row_index, positive]) or not bool(
            cpu_batch["option_mask"][row_index, negative]
        ):
            raise RuntimeError("margin option is outside the selected row option mask")
        margin = logits[row_index, positive] - logits[row_index, negative]
        if margin.dtype != torch.float32 or not bool(torch.isfinite(margin)):
            raise RuntimeError("margin is not finite FP32")
        observed_margin = float(margin.detach().cpu())
        if descriptor["role"] == "target" and observed_margin != descriptor["expected_raw_margin"]:
            raise RuntimeError("frozen target raw margin drift")
        if descriptor["role"] == "target":
            native_positive = native_logits[row_index, positive].detach()
            native_negative = native_logits[row_index, negative].detach()
            spacings = (
                torch.nextafter(
                    native_positive,
                    torch.full_like(native_positive, float("inf")),
                )
                - native_positive,
                native_positive
                - torch.nextafter(
                    native_positive,
                    torch.full_like(native_positive, float("-inf")),
                ),
                torch.nextafter(
                    native_negative,
                    torch.full_like(native_negative, float("inf")),
                )
                - native_negative,
                native_negative
                - torch.nextafter(
                    native_negative,
                    torch.full_like(native_negative, float("-inf")),
                ),
            )
            positive_bf16_margin = max(
                float(value.float().cpu()) for value in spacings
            )
            required_positive_margin = positive_bf16_margin - observed_margin
            if (
                positive_bf16_margin
                != descriptor["expected_positive_bf16_margin"]
                or required_positive_margin
                != descriptor["required_positive_margin"]
            ):
                raise RuntimeError("target positive-BF16 closure requirement drift")
            target_positive_bf16_margins.append(positive_bf16_margin)
            target_required_positive_margins.append(required_positive_margin)
        margins.append(margin)

    raw_gradients = []
    raw_gradient_norms: list[float] = []
    gradient_records = []
    for row_index, margin in enumerate(margins):
        flat, tensor_records = gradient_for_margin(
            margin,
            parameters,
            torch,
            retain_graph=row_index + 1 < len(margins),
        )
        norm = float(np.linalg.norm(flat))
        if not math.isfinite(norm) or norm <= 0.0 or int(flat.size) != EXPECTED_ACTOR6_FLAT_LENGTH:
            raise RuntimeError("selected-row actor6 margin gradient is invalid")
        unit = flat / norm
        raw_gradients.append(unit)
        raw_gradient_norms.append(norm)
        gradient_records.append(
            {
                "raw_float64_l2": norm,
                "raw_float64_le_sha256": vector_sha256_float64_le(flat, np),
                "unit_float64_le_sha256": vector_sha256_float64_le(unit, np),
                "actor6_tensor_gradients": tensor_records,
            }
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("autograd.grad unexpectedly materialized actor6 .grad")
    A = np.stack(raw_gradients, axis=0)
    direction, solver_record = solve_closure_rate_direction(
        A,
        np.asarray(raw_gradient_norms[:EXPECTED_TARGET_COUNT], dtype=np.float64),
        np.asarray(target_required_positive_margins, dtype=np.float64),
        np,
        optimize,
    )
    direct_cosines = A @ direction
    direct_derivatives = direct_cosines * np.asarray(
        raw_gradient_norms, dtype=np.float64
    )
    direct_target_closure_rates = (
        direct_derivatives[:EXPECTED_TARGET_COUNT]
        / np.asarray(target_required_positive_margins, dtype=np.float64)
    )
    direct_target_positive_radii = 1.0 / direct_target_closure_rates
    gram = A @ A.T

    row_records = []
    for index, (descriptor, selected, gradient, margin, cosine, derivative) in enumerate(
        zip(
            descriptors,
            selected_rows,
            gradient_records,
            margins,
            direct_cosines,
            direct_derivatives,
        )
    ):
        record = {
            "geometry_row": index,
            **descriptor,
            "raw_metadata": selected["raw_metadata"],
            "raw_margin": float(margin.detach().cpu()),
            "gradient": gradient,
            "unit_gradient_dot_unit_direction": float(cosine),
            "raw_gradient_dot_unit_direction": float(derivative),
        }
        if descriptor["role"] == "target":
            record.update(
                {
                    "observed_positive_bf16_margin": (
                        target_positive_bf16_margins[index]
                    ),
                    "observed_required_positive_margin": (
                        target_required_positive_margins[index]
                    ),
                    "direct_closure_rate": float(
                        direct_target_closure_rates[index]
                    ),
                    "direct_positive_bf16_linear_radius": float(
                        direct_target_positive_radii[index]
                    ),
                }
            )
        row_records.append(record)

    state_hash_after = helper.model_state_sha256(model.state_dict())
    if state_hash_after != state_hash_before or state_hash_after != PARENT_MODEL_STATE_SHA256:
        raise RuntimeError("model state changed during geometry probe")
    direction_slices = []
    offset = 0
    for name, parameter in zip(ACTOR6_NAMES, parameters):
        count = parameter.numel()
        value = direction[offset : offset + count]
        direction_slices.append(
            {
                "name": name,
                "shape": [int(item) for item in parameter.shape],
                "offset": offset,
                "count": count,
                "float64_le_sha256": vector_sha256_float64_le(value, np),
                "l2": float(np.linalg.norm(value)),
            }
        )
        offset += count
    if offset != EXPECTED_ACTOR6_FLAT_LENGTH:
        raise RuntimeError("direction slice accounting drift")

    return {
        "schema_version": SCHEMA,
        "status": "completed_geometry_only_stdout",
        "scope": {
            "parent": "raw U468",
            "selected_rows_only": True,
            "targets": EXPECTED_TARGET_COUNT,
            "distinct_cw_guards": EXPECTED_GUARD_COUNT,
            "full_panel_evaluation": False,
            "validation_member_payloads_opened": False,
            "optimizer_created": False,
            "backward_called": False,
            "parameter_step_or_mutation": False,
            "checkpoint_or_result_write": False,
            "stdout_only": True,
        },
        "runtime": {
            "python": str(Path(sys.executable).resolve()),
            "flags": ["-I", "-B"],
            "device": "cuda:0",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "numpy_version": str(np.__version__),
            "scipy_version": str(importlib.import_module("scipy").__version__),
            "native_forward_dtype": "torch.bfloat16",
            "margin_autograd_dtype": "torch.float32",
            "flattened_geometry_dtype": "little-endian float64",
            "seed": SEED,
        },
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(source),
            },
            "parent": parent_evidence,
            "parent_model_state_sha256": PARENT_MODEL_STATE_SHA256,
            "formal_v3_result": result_evidence,
            "dependencies": dependency_evidence,
            "train_archives": archive_evidence,
        },
        "selection": {
            "target_rule": (
                "fixed PF single-action line indices; exact alpha128 common-WC "
                "intersection across the four main metrics"
            ),
            "guard_rule": (
                "alpha then panel then metric scan; distinct line identity keeps "
                "first observed threat and unions later metric occurrences"
            ),
            "guard_sort": ["panel", "member", "line_index_zero_based", "line_sha256"],
            "margin_rule": (
                "one option-pair margin at the first differing metric-local "
                "ranking stage; no later or unrelated order frozen"
            ),
            "rows": row_records,
        },
        "geometry": {
            "actor6_parameter_order": list(ACTOR6_NAMES),
            "actor6_flat_length": EXPECTED_ACTOR6_FLAT_LENGTH,
            "normalized_gradient_matrix_shape": list(A.shape),
            "normalized_gradient_gram": gram.tolist(),
            "solver_and_direction": solver_record,
            "direction_actor6_slices": direction_slices,
            "direct_target_dots": [
                float(value) for value in direct_cosines[:EXPECTED_TARGET_COUNT]
            ],
            "direct_guard_dots": [
                float(value) for value in direct_cosines[EXPECTED_TARGET_COUNT:]
            ],
            "direct_target_directional_derivatives": [
                float(value)
                for value in direct_derivatives[:EXPECTED_TARGET_COUNT]
            ],
            "direct_target_closure_rates": [
                float(value) for value in direct_target_closure_rates
            ],
            "direct_target_positive_bf16_linear_radii": [
                float(value) for value in direct_target_positive_radii
            ],
        },
        "integrity": {
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_bit_exact_unchanged": True,
            "all_actor6_grad_buffers_remain_none": True,
            "target_raw_margins_exact": [
                float(margin.detach().cpu()) for margin in margins[:EXPECTED_TARGET_COUNT]
            ],
            "target_positive_bf16_margins_exact": list(
                target_positive_bf16_margins
            ),
            "target_required_positive_margins_exact": list(
                target_required_positive_margins
            ),
            "expected_unit_direction_sha256": EXPECTED_DIRECTION_SHA256,
            "unit_direction_sha256_exact": True,
            "direct_target_derivatives_match_frozen_geometry": True,
            "direct_target_closure_rates_equalized": True,
            "direct_guard_cosines_at_least_0_10": True,
            "static_zero_write_audit": dict(static),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "probe"), default="probe")
    args = parser.parse_args()
    validate_runtime()
    source, source_evidence = read_regular_bytes(
        SCRIPT, sha256_bytes(SCRIPT.read_bytes()), "metric-guard probe"
    )
    static = static_audit(source)
    if args.mode == "static":
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_zero_write_audit_only",
            "self": source_evidence,
            "audit": static,
        }
    else:
        result = run_probe(source, static)
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
