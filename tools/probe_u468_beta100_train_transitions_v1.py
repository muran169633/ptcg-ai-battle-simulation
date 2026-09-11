#!/usr/bin/env python3
"""Read-only raw-U468 versus beta100 full-train transition audit.

The public ``run_transition_audit`` entry point is intended for a hash-bound
in-process import by a later probe.  It returns every expanded CW/WC record and
never publishes a file.  The CLI has three zero-write modes:

* ``static`` authenticates the frozen files and the source contract;
* ``cache`` additionally opens only ``train/*.jsonl`` archive members and
  verifies the complete row-identity streams;
* ``run`` evaluates two independently instantiated complete checkpoints on
  the same CUDA batches and prints one canonical JSON object to stdout.

No mode trains, opens a validation-member payload, serializes a checkpoint,
uses network access, or writes an evidence artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import os
import stat
import sys
import time
import zipfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_beta100_train_transitions_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

SCHEMA = "ptcg-u468-beta100-complete-train-transition-audit-v1"
BATCH_SIZE = 256
WORKERS = 0
SEED = 20260802
PANEL_ORDER = ("flg", "pokemonfan", "core5")
POLICY_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
COUNT_VALUE_METRICS = ("count_correct", "value_correct")
ALL_METRICS = POLICY_METRICS + COUNT_VALUE_METRICS
TRANSITION_CELLS = ("cc", "cw", "wc", "ww")

RAW_CHECKPOINT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_FILE_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
RAW_MODEL_SHA256 = "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"

BETA_CHECKPOINT = ROOT / (
    "artifacts/ppo_u468_p12delta_direction_beta050_075_100_design202608092/"
    "transport-beta-100.pt"
)
BETA_FILE_SHA256 = "53284b1d4e94f09bff5b92a7fb0d24efd26ee67a2a332014cdc97572b00c3beb"
BETA_MODEL_SHA256 = "bed8734e0dc8a4eca2443737976724a7785fb2f4abea21e60e51f0e99b09390d"

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
EXPECTED_TRAIN = {
    "flg": {"rows": 9443, "members": 1, "nonempty": 9426, "context34": 42},
    "pokemonfan": {
        "rows": 9487,
        "members": 1,
        "nonempty": 9450,
        "context34": 38,
    },
    "core5": {"rows": 5120, "members": 20, "nonempty": 5106, "context34": 20},
}
EXPECTED_IDENTITY_SHA256 = {
    "flg": "c4ecb8cd6b6744b823ba12f1395c2900b6168b4bba63cca1a1756e0389ad0e0c",
    "pokemonfan": "7457709480f8b7929106079ab0c255f1528c2aa7086f0b536e6922af588838cb",
    "core5": "c494d6c13787dc934097ea3896f93838b299562664467b2c7fe5b215ce9958e7",
}
EXPECTED_UNION_IDENTITY_SUMMARY_SHA256 = (
    "f8b2df4951c87c2d0b1e9c2358223494db10cf9bc7de02b1667b7d2ded00792d"
)

# Frozen native-CUDA-BF16 policy-greedy aggregates.  These are not inferred
# from a net delta: both complete endpoint totals must independently match.
EXPECTED_AGGREGATES = {
    "flg": {
        "raw": {
            "set_exact": 7239,
            "hybrid_order_exact": 7156,
            "ordered_exact": 7095,
            "top1_correct": 7308,
            "count_correct": 9407,
            "value_correct": 6439,
            "context34_hybrid_order_exact": 30,
            "context34_ordered_exact": 30,
        },
        "beta100": {
            "set_exact": 7239,
            "hybrid_order_exact": 7156,
            "ordered_exact": 7096,
            "top1_correct": 7308,
            "count_correct": 9407,
            "value_correct": 6439,
            "context34_hybrid_order_exact": 30,
            "context34_ordered_exact": 30,
        },
    },
    "pokemonfan": {
        "raw": {
            "set_exact": 8282,
            "hybrid_order_exact": 8281,
            "ordered_exact": 8203,
            "top1_correct": 8332,
            "count_correct": 9409,
            "value_correct": 6585,
            "context34_hybrid_order_exact": 37,
            "context34_ordered_exact": 37,
        },
        "beta100": {
            "set_exact": 8280,
            "hybrid_order_exact": 8279,
            "ordered_exact": 8201,
            "top1_correct": 8329,
            "count_correct": 9409,
            "value_correct": 6585,
            "context34_hybrid_order_exact": 37,
            "context34_ordered_exact": 37,
        },
    },
    "core5": {
        "raw": {
            "set_exact": 4073,
            "hybrid_order_exact": 4042,
            "ordered_exact": 4025,
            "top1_correct": 4130,
            "count_correct": 5061,
            "value_correct": 3822,
            "context34_hybrid_order_exact": 16,
            "context34_ordered_exact": 16,
        },
        "beta100": {
            "set_exact": 4071,
            "hybrid_order_exact": 4040,
            "ordered_exact": 4023,
            "top1_correct": 4128,
            "count_correct": 5061,
            "value_correct": 3822,
            "context34_hybrid_order_exact": 16,
            "context34_ordered_exact": 16,
        },
    },
}

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
COUNT4_NAMES = (
    "count_head.0.weight",
    "count_head.0.bias",
    "count_head.2.weight",
    "count_head.2.bias",
)
EXPECTED_CHANGED_NAMES = tuple(sorted(ACTOR6_NAMES + COUNT4_NAMES))
BETA_SUBSET_SHA256 = {
    "actor6": "c854849bfe4686909b9de5a60dba02957cc397f25c50c8db00fd94fbfa39f1aa",
    "count4": "b8de52a289dd6dddd78e5eecf287a84a52408a9bfd3937b06fb30a2514b3b5a2",
    "value4": "c808854db453276b66e3f05eb18804269941ddff091eb0cb4322d16bb42cf901",
}
VALUE4_NAMES = (
    "value_head.0.weight",
    "value_head.0.bias",
    "value_head.2.weight",
    "value_head.2.bias",
)

DEPENDENCY_SHA256 = {
    TOOLS / "evaluate_policy_bc.py": "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4",
    TOOLS / "train_ppo.py": "de3d1467d2b18a551b66c2e897e774d251a898b276dac709268b03d05da41ba3",
    TOOLS / "train_bc_orbit.py": "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_bytes(
    path: Path,
    expected_sha256: str | None,
    label: str,
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
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode": oct(stat.S_IMODE(after.st_mode)),
        "nlink": after.st_nlink,
        "device": after.st_dev,
        "inode": after.st_ino,
    }


def _validate_runtime() -> None:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve(strict=True) != EXPECTED_PYTHON.resolve(strict=True):
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B flags")


def _archive_inventory(payload: bytes, panel: str) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
    members = sorted(
        name
        for name in names
        if name.startswith("train/") and name.endswith(".jsonl")
    )
    if len(members) != int(EXPECTED_TRAIN[panel]["members"]):
        raise RuntimeError(f"{panel} train-member inventory drift")
    return {
        "archive_member_count": len(names),
        "train_members": members,
        "train_member_count": len(members),
        "member_payloads_opened": 0,
    }


def _fixed_inputs() -> tuple[dict[str, Any], dict[str, bytes]]:
    evidence: dict[str, Any] = {"checkpoints": {}, "datasets": {}, "dependencies": {}}
    payloads: dict[str, bytes] = {}
    for name, path, expected in (
        ("raw", RAW_CHECKPOINT, RAW_FILE_SHA256),
        ("beta100", BETA_CHECKPOINT, BETA_FILE_SHA256),
    ):
        payload, item = _read_regular_bytes(path, expected, f"{name} checkpoint")
        payloads[name] = payload
        evidence["checkpoints"][name] = item
    for panel in PANEL_ORDER:
        payload, item = _read_regular_bytes(
            DATASETS[panel], DATA_SHA256[panel], f"{panel} archive"
        )
        payloads[panel] = payload
        evidence["datasets"][panel] = {
            **item,
            "inventory": _archive_inventory(payload, panel),
        }
    for path, expected in DEPENDENCY_SHA256.items():
        _, item = _read_regular_bytes(path, expected, f"dependency {path.name}")
        evidence["dependencies"][str(path.relative_to(ROOT))] = item
    return evidence, payloads


def _ast_audit(source: bytes) -> dict[str, Any]:
    text = source.decode("utf-8")
    tree = ast.parse(text, filename=str(SCRIPT))
    forbidden_import_roots = {
        "socket",
        "subprocess",
        "requests",
        "urllib",
        "http",
        "ftplib",
    }
    imports: list[str] = []
    calls: list[str] = []

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            calls.append(dotted(node.func))
    forbidden_imports = sorted(
        name for name in imports if name.split(".", 1)[0] in forbidden_import_roots
    )
    forbidden_calls = sorted(
        name
        for name in calls
        if name
        in {
            "open",
            "Path.open",
            "Path.write_bytes",
            "Path.write_text",
            "os.mkdir",
            "os.makedirs",
            "os.remove",
            "os.rename",
            "os.replace",
            "os.unlink",
            "torch.save",
        }
        or name.endswith(".backward")
        or name.endswith(".step")
    )
    if forbidden_imports or forbidden_calls:
        raise RuntimeError(
            f"source scope audit failed: imports={forbidden_imports}, calls={forbidden_calls}"
        )
    return {
        "status": "static_ast_scope_audit_passed",
        "forbidden_network_or_subprocess_imports": forbidden_imports,
        "forbidden_write_train_or_serialize_calls": forbidden_calls,
        "os_open_sites": calls.count("os.open"),
        "os_open_read_only_by_source_review": True,
        "stdout_print_sites": calls.count("print"),
        "validation_member_payload_open_sites": 0,
    }


def _new_transition_book(metrics: Sequence[str]) -> dict[str, Any]:
    return {
        metric: {
            cell: {"count": 0, "identity_digest": hashlib.sha256(), "expanded": []}
            for cell in TRANSITION_CELLS
        }
        for metric in metrics
    }


def _stable_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel": str(identity["panel"]),
        "archive": str(identity["archive"]),
        "member": str(identity["member"]),
        "line_index_zero_based": int(identity["line_index_zero_based"]),
        "line_sha256": str(identity["line_sha256"]),
        "context": int(identity["context"]),
    }


def _metric_applicable(metric: str, flags: Mapping[str, bool], context: int) -> bool:
    if metric.startswith("context34_"):
        return context == 34
    if metric == "top1_correct":
        return bool(flags["nonempty"])
    return True


def _update_transition_book(
    book: Mapping[str, Any],
    identity: Mapping[str, Any],
    raw_flags: Mapping[str, bool],
    beta_flags: Mapping[str, bool],
    raw_prediction: Mapping[str, Any],
    beta_prediction: Mapping[str, Any],
) -> None:
    context = int(identity["context"])
    for metric, cells in book.items():
        raw_applicable = _metric_applicable(metric, raw_flags, context)
        if raw_applicable != _metric_applicable(metric, beta_flags, context):
            raise RuntimeError(f"applicability drift for {metric}")
        if not raw_applicable:
            continue
        key = metric.removeprefix("context34_")
        raw_correct = bool(raw_flags[key])
        beta_correct = bool(beta_flags[key])
        cell = ("c" if raw_correct else "w") + ("c" if beta_correct else "w")
        record = cells[cell]
        record["count"] += 1
        record["identity_digest"].update(canonical_json(_stable_identity(identity)))
        if cell in {"cw", "wc"}:
            record["expanded"].append(
                {
                    **_stable_identity(identity),
                    "raw_order": [int(value) for value in raw_prediction["order"]],
                    "beta100_order": [int(value) for value in beta_prediction["order"]],
                    "raw": {
                        "correct": raw_correct,
                        "predicted_count": int(raw_prediction["predicted_count"]),
                        "top1_index": int(raw_prediction["top1_index"]),
                        "value_sign_nonnegative": bool(raw_prediction["value_sign"]),
                    },
                    "beta100": {
                        "correct": beta_correct,
                        "predicted_count": int(beta_prediction["predicted_count"]),
                        "top1_index": int(beta_prediction["top1_index"]),
                        "value_sign_nonnegative": bool(beta_prediction["value_sign"]),
                    },
                }
            )


def _denominator(panel: str, metric: str) -> int:
    expected = EXPECTED_TRAIN[panel]
    if metric == "top1_correct":
        return int(expected["nonempty"])
    if metric.startswith("context34_"):
        return int(expected["context34"])
    return int(expected["rows"])


def _finalize_transition_book(
    panel: str,
    book: Mapping[str, Any],
    raw_counts: Mapping[str, int],
    beta_counts: Mapping[str, int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric, cells in book.items():
        serialized: dict[str, Any] = {}
        for cell in TRANSITION_CELLS:
            value = cells[cell]
            item: dict[str, Any] = {
                "count": int(value["count"]),
                "canonical_identity_stream_sha256": value[
                    "identity_digest"
                ].hexdigest(),
            }
            if cell in {"cw", "wc"}:
                item["expanded_stable_identities_and_raw_beta_orders"] = value["expanded"]
                item["expanded_count"] = len(value["expanded"])
                if item["expanded_count"] != item["count"]:
                    raise RuntimeError(f"{panel}/{metric}/{cell} expansion drift")
            serialized[cell] = item
        denominator = sum(serialized[cell]["count"] for cell in TRANSITION_CELLS)
        expected_denominator = _denominator(panel, metric)
        if denominator != expected_denominator:
            raise RuntimeError(f"{panel}/{metric} denominator drift: {denominator}")
        aggregate_delta = int(beta_counts[metric]) - int(raw_counts[metric])
        wc_minus_cw = serialized["wc"]["count"] - serialized["cw"]["count"]
        if wc_minus_cw != aggregate_delta:
            raise RuntimeError(f"{panel}/{metric} WC-CW != aggregate delta")
        output[metric] = {
            "cells": serialized,
            "applicable_denominator": denominator,
            "four_cells_close_denominator": True,
            "raw_correct": int(raw_counts[metric]),
            "beta100_correct": int(beta_counts[metric]),
            "aggregate_delta": aggregate_delta,
            "wc_minus_cw": wc_minus_cw,
            "wc_minus_cw_equals_aggregate_delta": True,
        }
    return output


def _synthetic_self_test() -> dict[str, Any]:
    book = _new_transition_book(("count_correct",))
    patterns = ((True, True), (True, False), (False, True), (False, False))
    for index, (raw_correct, beta_correct) in enumerate(patterns):
        identity = {
            "panel": "synthetic",
            "archive": "synthetic.zip",
            "member": "train/x.jsonl",
            "line_index_zero_based": index,
            "line_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "context": 0,
        }
        raw_flags = {"count_correct": raw_correct, "nonempty": True}
        beta_flags = {"count_correct": beta_correct, "nonempty": True}
        prediction = {
            "order": [index],
            "predicted_count": 1,
            "top1_index": index,
            "value_sign": True,
        }
        _update_transition_book(
            book, identity, raw_flags, beta_flags, prediction, prediction
        )
    cells = book["count_correct"]
    if [cells[cell]["count"] for cell in TRANSITION_CELLS] != [1, 1, 1, 1]:
        raise RuntimeError("synthetic four-cell transition self-test failed")
    if len(cells["cw"]["expanded"]) != 1 or len(cells["wc"]["expanded"]) != 1:
        raise RuntimeError("synthetic CW/WC expansion self-test failed")
    if cells["wc"]["count"] - cells["cw"]["count"] != 0:
        raise RuntimeError("synthetic WC-CW self-test failed")
    return {
        "status": "synthetic_transition_self_test_passed",
        "all_four_cells_exercised": True,
        "cw_wc_expanded": True,
        "wc_minus_cw_identity": True,
    }


def static_audit() -> dict[str, Any]:
    """Authenticate fixed inputs without deserializing a model or using CUDA."""

    _validate_runtime()
    source, source_evidence = _read_regular_bytes(SCRIPT, None, "transition tool")
    fixed, _ = _fixed_inputs()
    return {
        "schema_version": SCHEMA,
        "status": "static_audit_passed",
        "mode": "static",
        "formal_evaluation_executed": False,
        "source": source_evidence,
        "ast_audit": _ast_audit(source),
        "synthetic_self_test": _synthetic_self_test(),
        "fixed_inputs": fixed,
        "model_bindings": {
            "raw_model_state_sha256": RAW_MODEL_SHA256,
            "beta100_model_state_sha256": BETA_MODEL_SHA256,
            "expected_complete_state_changed_names": list(EXPECTED_CHANGED_NAMES),
            "beta100_subset_sha256": BETA_SUBSET_SHA256,
        },
        "evaluation_contract": {
            "panels": list(PANEL_ORDER),
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "device": "cuda:0",
            "native_autocast_dtype": "torch.bfloat16",
            "prediction": "policy_greedy_deterministic_noncanonical",
            "full_checkpoints_independently_instantiated": True,
            "archive_member_payload_scope": "train/*.jsonl only",
            "expected_rows": {panel: EXPECTED_TRAIN[panel]["rows"] for panel in PANEL_ORDER},
            "expected_identity_stream_sha256": EXPECTED_IDENTITY_SHA256,
            "expected_union_identity_summary_sha256": (
                EXPECTED_UNION_IDENTITY_SUMMARY_SHA256
            ),
            "known_aggregate_counts": EXPECTED_AGGREGATES,
        },
        "scope": {
            "writes_performed": False,
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "training_backward_optimizer": False,
            "network": False,
        },
    }


def _scan_train_identity(panel: str, payload: bytes) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    opened: list[str] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if len(members) != int(EXPECTED_TRAIN[panel]["members"]):
            raise RuntimeError(f"{panel} train member count drift")
        for member in members:
            if not member.startswith("train/") or not member.endswith(".jsonl"):
                raise RuntimeError(f"refusing non-train member {member}")
            opened.append(member)
            with archive.open(member) as handle:
                for line_index, raw_line in enumerate(handle):
                    row = json.loads(raw_line)
                    if not isinstance(row, dict) or row.get("split") != "train":
                        raise RuntimeError(f"non-train row in {panel}/{member}")
                    identity = {
                        "panel": panel,
                        "member": member,
                        "line_index_zero_based": line_index,
                        "line_sha256": sha256_bytes(raw_line),
                    }
                    digest.update(canonical_json(identity))
                    rows += 1
    observed = digest.hexdigest()
    if rows != int(EXPECTED_TRAIN[panel]["rows"]):
        raise RuntimeError(f"{panel} train row count drift: {rows}")
    if observed != EXPECTED_IDENTITY_SHA256[panel]:
        raise RuntimeError(f"{panel} identity stream SHA drift: {observed}")
    return {
        "rows": rows,
        "train_members": opened,
        "train_member_count": len(opened),
        "identity_stream_sha256": observed,
        "identity_schema": [
            "panel",
            "member",
            "line_index_zero_based",
            "line_sha256",
        ],
        "validation_member_payloads_opened": False,
    }


def _cache_from_payloads(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    panels = {
        panel: _scan_train_identity(panel, payloads[panel]) for panel in PANEL_ORDER
    }
    union_summary = {
        panel: {
            "rows": panels[panel]["rows"],
            "identity_stream_sha256": panels[panel]["identity_stream_sha256"],
        }
        for panel in PANEL_ORDER
    }
    union_sha = sha256_bytes(canonical_json(union_summary))
    if union_sha != EXPECTED_UNION_IDENTITY_SUMMARY_SHA256:
        raise RuntimeError(f"union identity summary SHA drift: {union_sha}")
    return {
        "status": "zero_write_train_identity_cache_audit_passed",
        "panels": panels,
        "union_identity_summary": union_summary,
        "union_identity_summary_canonical_sha256": union_sha,
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }


def cache_audit() -> dict[str, Any]:
    """Verify every frozen train line identity; opens no non-train payload."""

    _validate_runtime()
    fixed, payloads = _fixed_inputs()
    return {
        "schema_version": SCHEMA,
        "status": "cache_audit_passed",
        "mode": "cache",
        "fixed_inputs": fixed,
        "cache": _cache_from_payloads(payloads),
        "formal_evaluation_executed": False,
        "scope": {
            "writes_performed": False,
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "network": False,
        },
    }


def _load_runtime() -> SimpleNamespace:
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    torch = importlib.import_module("torch")
    orjson = importlib.import_module("orjson")
    evaluator = importlib.import_module("evaluate_policy_bc")
    ppo = importlib.import_module("train_ppo")
    bc = importlib.import_module("train_bc_orbit")
    for module, expected_path in (
        (evaluator, TOOLS / "evaluate_policy_bc.py"),
        (ppo, TOOLS / "train_ppo.py"),
        (bc, TOOLS / "train_bc_orbit.py"),
    ):
        module_path = Path(str(module.__file__)).resolve(strict=True)
        if module_path != expected_path.resolve(strict=True):
            raise RuntimeError(f"runtime module path drift: {module.__name__}")
    return SimpleNamespace(
        torch=torch,
        orjson=orjson,
        evaluator=evaluator,
        ppo=ppo,
        bc=bc,
    )


def _model_state_sha256(state: Mapping[str, Any], runtime: SimpleNamespace) -> str:
    torch = runtime.torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"non-tensor model state entry {name}")
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


def _subset_sha256(
    state: Mapping[str, Any], names: Sequence[str], runtime: SimpleNamespace
) -> str:
    return _model_state_sha256({name: state[name] for name in names}, runtime)


def _load_checkpoint(payload: bytes, label: str, runtime: SimpleNamespace) -> dict[str, Any]:
    checkpoint = runtime.torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise RuntimeError(f"{label} is not a complete model checkpoint")
    return checkpoint


def _validate_complete_states(
    raw: Mapping[str, Any], beta: Mapping[str, Any], runtime: SimpleNamespace
) -> dict[str, Any]:
    torch = runtime.torch
    if raw.get("update") != 468 or beta.get("update") != 468:
        raise RuntimeError("raw/beta checkpoint update is not U468")
    if runtime.evaluator.checkpoint_kind(dict(raw)) != "ppo" or (
        runtime.evaluator.checkpoint_kind(dict(beta)) != "ppo"
    ):
        raise RuntimeError("raw/beta checkpoint kind is not PPO")
    if (
        raw.get("model_config") != beta.get("model_config")
        or raw.get("feature_version") != beta.get("feature_version")
        or raw.get("bc_feature_version") != beta.get("bc_feature_version")
    ):
        raise RuntimeError("raw/beta inference schema differs")
    if beta.get("evaluation_only") is not True or beta.get("resume_forbidden") is not True:
        raise RuntimeError("beta100 evaluation-only/resume-forbidden metadata drift")
    forbidden_resume_keys = (
        "optimizer_state_dict",
        "bc_replay_optimizer_state_dict",
        "opponent_quota_state",
    )
    present_resume_keys = [name for name in forbidden_resume_keys if name in beta]
    if present_resume_keys:
        raise RuntimeError(f"beta100 unexpectedly carries resume state: {present_resume_keys}")
    raw_state = raw["model_state_dict"]
    beta_state = beta["model_state_dict"]
    raw_hash = _model_state_sha256(raw_state, runtime)
    beta_hash = _model_state_sha256(beta_state, runtime)
    if raw_hash != RAW_MODEL_SHA256 or beta_hash != BETA_MODEL_SHA256:
        raise RuntimeError("raw/beta complete model-state SHA drift")
    if set(raw_state) != set(beta_state):
        raise RuntimeError("raw/beta complete state key sets differ")
    changed: list[str] = []
    for name in sorted(raw_state):
        left = raw_state[name]
        right = beta_state[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"raw/beta tensor schema drift at {name}")
        if not torch.equal(left, right):
            changed.append(name)
    if tuple(changed) != EXPECTED_CHANGED_NAMES:
        raise RuntimeError(f"raw/beta changed tensor scope drift: {changed}")
    subset_hashes = {
        "actor6": _subset_sha256(beta_state, ACTOR6_NAMES, runtime),
        "count4": _subset_sha256(beta_state, COUNT4_NAMES, runtime),
        "value4": _subset_sha256(beta_state, VALUE4_NAMES, runtime),
    }
    if subset_hashes != BETA_SUBSET_SHA256:
        raise RuntimeError(f"beta100 subset SHA drift: {subset_hashes}")
    return {
        "raw_complete_model_state_sha256": raw_hash,
        "beta100_complete_model_state_sha256": beta_hash,
        "state_key_count": len(raw_state),
        "changed_parameter_names": changed,
        "changed_parameter_count": len(changed),
        "changed_scope_exact_actor6_plus_count4": True,
        "beta100_subset_sha256": subset_hashes,
        "beta100_evaluation_only": True,
        "beta100_resume_forbidden": True,
        "beta100_resume_state_keys_absent": list(forbidden_resume_keys),
    }


def _instantiate_complete_model(
    checkpoint: Mapping[str, Any], runtime: SimpleNamespace, device: Any
) -> tuple[Any, dict[str, Any]]:
    config = runtime.evaluator.model_config_from_checkpoint(dict(checkpoint), "ppo")
    model = runtime.evaluator.instantiate_ppo_checkpoint(
        dict(checkpoint), config, device
    )
    model.eval()
    return model, config


def _identity_collate(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    runtime: SimpleNamespace,
    model_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = [row[0] for row in rows]
    identities = [row[1] for row in rows]
    batch = runtime.evaluator.collate_ordered(
        features,
        max_state_entities=int(model_config["max_state_entities"]),
        entity_fields=int(model_config["entity_fields"]),
        option_fields=int(model_config["option_fields"]),
    )
    if int(batch["contexts"].shape[0]) != len(identities):
        raise RuntimeError("collated identity row count drift")
    for index, identity in enumerate(identities):
        expert_count = int(batch["expert_ordered_action_counts"][index])
        expert_order = batch["expert_ordered_actions"][index, :expert_count].tolist()
        if (
            int(batch["contexts"][index]) != int(identity["context"])
            or expert_order != identity["expert_order"]
            or int(batch["option_mask"][index].sum()) != int(identity["option_count"])
        ):
            raise RuntimeError("identity/features synchronization drift")
    return batch, identities


def _build_loader(
    panel: str,
    archive_payload: bytes,
    runtime: SimpleNamespace,
    model_config: Mapping[str, Any],
) -> tuple[Any, Any]:
    torch = runtime.torch

    class IdentityTrainDataset(torch.utils.data.IterableDataset):
        def __init__(self) -> None:
            super().__init__()
            self.opened_members: list[str] = []
            self.inventory_members: list[str] = []
            self.rows = 0
            self.nonempty = 0
            self.context34 = 0
            self.identity_digest = hashlib.sha256()
            self.validation_member_payloads_opened = False

        def __iter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
            if torch.utils.data.get_worker_info() is not None:
                raise RuntimeError("transition audit requires workers=0")
            with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
                members = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("train/") and name.endswith(".jsonl")
                )
                if len(members) != int(EXPECTED_TRAIN[panel]["members"]):
                    raise RuntimeError(f"{panel} train member inventory drift")
                self.inventory_members = members
                for member in members:
                    if not member.startswith("train/") or not member.endswith(".jsonl"):
                        self.validation_member_payloads_opened = True
                        raise RuntimeError(f"refusing non-train member {member}")
                    self.opened_members.append(member)
                    with archive.open(member) as handle:
                        for line_index, raw_line in enumerate(handle):
                            row = runtime.orjson.loads(raw_line)
                            if not isinstance(row, dict) or row.get("split") != "train":
                                raise RuntimeError(f"non-train row in {panel}/{member}")
                            raw_action = row.get("action")
                            if not isinstance(raw_action, list):
                                raise RuntimeError(f"invalid action in {panel}/{member}")
                            expert_order = [int(value) for value in raw_action]
                            previous_max = runtime.bc.MAX_ACTION_COUNT
                            runtime.bc.MAX_ACTION_COUNT = runtime.ppo.MAX_ACTION_COUNT
                            try:
                                features = runtime.bc.featurize_row(
                                    row,
                                    int(model_config["hash_size"]),
                                    int(model_config["max_state_entities"]),
                                )
                            finally:
                                runtime.bc.MAX_ACTION_COUNT = previous_max
                            options = (
                                ((row.get("observation") or {}).get("select") or {}).get(
                                    "option"
                                )
                                or []
                            )
                            if (
                                features is None
                                or len(expert_order) > runtime.ppo.MAX_ACTION_COUNT
                                or len(set(expert_order)) != len(expert_order)
                                or any(
                                    value < 0 or value >= len(options)
                                    for value in expert_order
                                )
                            ):
                                raise RuntimeError(
                                    f"non-evaluable row in {panel}/{member}:{line_index}"
                                )
                            features["expert_action_order"] = expert_order
                            context = int(features["context"])
                            identity = {
                                "panel": panel,
                                "archive": str(DATASETS[panel].relative_to(ROOT)),
                                "member": member,
                                "line_index_zero_based": line_index,
                                "line_sha256": sha256_bytes(raw_line),
                                "context": context,
                                "expert_order": expert_order,
                                "option_count": len(options),
                            }
                            identity_for_digest = {
                                "panel": panel,
                                "member": member,
                                "line_index_zero_based": line_index,
                                "line_sha256": identity["line_sha256"],
                            }
                            self.identity_digest.update(canonical_json(identity_for_digest))
                            self.rows += 1
                            self.nonempty += int(bool(expert_order))
                            self.context34 += int(context == 34)
                            yield features, identity

    dataset = IdentityTrainDataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        collate_fn=partial(
            _identity_collate,
            runtime=runtime,
            model_config=model_config,
        ),
        pin_memory=True,
        persistent_workers=False,
    )
    return loader, dataset


def _finalize_dataset(panel: str, dataset: Any) -> dict[str, Any]:
    expected = EXPECTED_TRAIN[panel]
    observed_sha = dataset.identity_digest.hexdigest()
    if (
        dataset.validation_member_payloads_opened
        or dataset.opened_members != dataset.inventory_members
        or dataset.rows != int(expected["rows"])
        or dataset.nonempty != int(expected["nonempty"])
        or dataset.context34 != int(expected["context34"])
        or observed_sha != EXPECTED_IDENTITY_SHA256[panel]
    ):
        raise RuntimeError(f"{panel} evaluated dataset audit failed")
    return {
        "rows": dataset.rows,
        "nonempty_rows": dataset.nonempty,
        "context34_rows": dataset.context34,
        "train_members": dataset.opened_members,
        "workers": WORKERS,
        "identity_stream_sha256": observed_sha,
        "validation_member_payloads_opened": False,
    }


def _row_flags(
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    actions: Sequence[Sequence[int]],
    row_index: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    targets = cpu_batch["targets"][row_index].bool()
    option_mask = cpu_batch["option_mask"][row_index].bool()
    prediction = targets.new_zeros(targets.shape)
    order = [int(value) for value in actions[row_index]]
    if order:
        prediction[order] = True
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert_order = cpu_batch["expert_ordered_actions"][
        row_index, :expert_count
    ].tolist()
    context = int(cpu_batch["contexts"][row_index])
    top1_index = int(outputs_cpu["policy_logits"][row_index].argmax())
    value_sign = bool(outputs_cpu["value_logits"][row_index] >= 0)
    flags = {
        "set_exact": bool(((prediction == targets) | ~option_mask).all()),
        "ordered_exact": order == expert_order,
        "hybrid_order_exact": (
            order == expert_order if context == 34 else sorted(order) == expert_order
        ),
        "top1_correct": bool(targets[top1_index]),
        "count_correct": len(order) == int(cpu_batch["action_counts"][row_index]),
        "value_correct": value_sign == bool(cpu_batch["win_targets"][row_index]),
        "nonempty": int(cpu_batch["action_counts"][row_index]) > 0,
    }
    return flags, {
        "order": order,
        "predicted_count": len(order),
        "top1_index": top1_index,
        "value_sign": value_sign,
    }


def _aggregate_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    context34 = summary.get("by_context", {}).get("34")
    if not isinstance(context34, dict):
        raise RuntimeError("official metric summary lacks context 34")
    return {
        "set_exact": int(summary["set_exact_correct"]),
        "hybrid_order_exact": int(summary["hybrid_order_exact_correct"]),
        "ordered_exact": int(summary["ordered_exact_correct"]),
        "top1_correct": int(summary["top1_correct"]),
        "count_correct": int(summary["count_correct"]),
        "value_correct": int(summary["value_correct"]),
        "context34_hybrid_order_exact": int(
            context34["hybrid_order_exact_correct"]
        ),
        "context34_ordered_exact": int(context34["ordered_exact_correct"]),
    }


def _new_logit_audit() -> dict[str, Any]:
    return {
        head: {
            "raw_digest": hashlib.sha256(),
            "beta_digest": hashlib.sha256(),
            "mismatch_identity_digest": hashlib.sha256(),
            "row_mismatch_count": 0,
            "element_mismatch_count": 0,
            "sum_abs_difference": 0.0,
            "max_abs_difference": 0.0,
            "rows": 0,
        }
        for head in ("count_logits", "value_logits")
    }


def _fingerprint_tensor_row(
    digest: Any,
    identity: Mapping[str, Any],
    tensor: Any,
    runtime: SimpleNamespace,
) -> None:
    value = tensor.detach().to(device="cpu", dtype=runtime.torch.float32).contiguous()
    array = value.numpy()
    if sys.byteorder != "little":
        array = array.byteswap().newbyteorder("<")
    else:
        array = array.astype("<f4", copy=False)
    digest.update(canonical_json(_stable_identity(identity)))
    digest.update(canonical_json({"shape": [int(item) for item in value.shape]}))
    digest.update(array.tobytes(order="C"))
    digest.update(b"\0")


def _update_logit_audit(
    audit: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    raw_outputs: Mapping[str, Any],
    beta_outputs: Mapping[str, Any],
    runtime: SimpleNamespace,
) -> None:
    torch = runtime.torch
    for head in ("count_logits", "value_logits"):
        raw = raw_outputs[head]
        beta = beta_outputs[head]
        if raw.shape != beta.shape:
            raise RuntimeError(f"{head} shape differs raw versus beta100")
        rows = len(identities)
        equal_rows = (raw == beta).reshape(rows, -1).all(dim=1).cpu().tolist()
        different_elements = raw != beta
        delta = (beta.float() - raw.float()).abs()
        item = audit[head]
        item["element_mismatch_count"] += int(different_elements.sum())
        item["sum_abs_difference"] += float(delta.double().sum().cpu())
        item["max_abs_difference"] = max(
            float(item["max_abs_difference"]), float(delta.max().cpu())
        )
        item["rows"] += rows
        for index, identity in enumerate(identities):
            _fingerprint_tensor_row(item["raw_digest"], identity, raw[index], runtime)
            _fingerprint_tensor_row(item["beta_digest"], identity, beta[index], runtime)
            if not bool(equal_rows[index]):
                item["row_mismatch_count"] += 1
                item["mismatch_identity_digest"].update(
                    canonical_json(_stable_identity(identity))
                )


def _finalize_logit_audit(audit: Mapping[str, Any], rows: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head, item in audit.items():
        if int(item["rows"]) != rows:
            raise RuntimeError(f"{head} audit row count drift")
        result[head] = {
            "rows": rows,
            "native_bfloat16_row_mismatch_count": int(item["row_mismatch_count"]),
            "native_bfloat16_row_match_count": rows - int(item["row_mismatch_count"]),
            "native_bfloat16_element_mismatch_count": int(
                item["element_mismatch_count"]
            ),
            "native_bfloat16_sum_abs_difference": float(item["sum_abs_difference"]),
            "native_bfloat16_max_abs_difference": float(item["max_abs_difference"]),
            "mismatch_stable_identity_stream_sha256": item[
                "mismatch_identity_digest"
            ].hexdigest(),
            "raw_float32_le_per_row_fingerprint_sha256": item[
                "raw_digest"
            ].hexdigest(),
            "beta100_float32_le_per_row_fingerprint_sha256": item[
                "beta_digest"
            ].hexdigest(),
            "raw_beta_fingerprint_exact": (
                item["raw_digest"].hexdigest() == item["beta_digest"].hexdigest()
            ),
            "relative_logits_measured_not_assumed": True,
        }
    return result


def _evaluate_transitions(
    raw_model: Any,
    beta_model: Any,
    model_config: Mapping[str, Any],
    archive_payloads: Mapping[str, bytes],
    runtime: SimpleNamespace,
    device: Any,
    row_consumer: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = runtime.torch
    panels: dict[str, Any] = {}
    dataset_audits: dict[str, Any] = {}
    raw_model.eval()
    beta_model.eval()
    for panel in PANEL_ORDER:
        loader, dataset = _build_loader(
            panel, archive_payloads[panel], runtime, model_config
        )
        accumulators = {
            "raw": runtime.evaluator.MetricAccumulator(),
            "beta100": runtime.evaluator.MetricAccumulator(),
        }
        book = _new_transition_book(ALL_METRICS)
        logit_audit = _new_logit_audit()
        rows = 0
        dtype_evidence: dict[str, dict[str, str]] = {}
        for cpu_batch, identities in loader:
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
            }
            state_results: dict[str, Any] = {}
            for name, model in (("raw", raw_model), ("beta100", beta_model)):
                with torch.inference_mode():
                    outputs = runtime.ppo.model_forward(model, batch, device)
                    actions, _, _, _ = runtime.ppo.sample_ordered_actions(
                        outputs,
                        batch,
                        deterministic=True,
                        canonicalize_order=False,
                    )
                if set(outputs) != {"policy_logits", "count_logits", "value_logits"}:
                    raise RuntimeError(f"{panel}/{name} output-key drift")
                if any(value.dtype != torch.bfloat16 for value in outputs.values()):
                    raise RuntimeError(f"{panel}/{name} output is not native CUDA BF16")
                observed_dtypes = {key: str(value.dtype) for key, value in outputs.items()}
                if name in dtype_evidence and dtype_evidence[name] != observed_dtypes:
                    raise RuntimeError(f"{panel}/{name} output dtype drift")
                dtype_evidence[name] = observed_dtypes
                accumulators[name].update(batch, outputs, actions, actions)
                outputs_cpu = {
                    key: value.detach().float().cpu().contiguous()
                    for key, value in outputs.items()
                }
                flags: list[dict[str, bool]] = []
                predictions: list[dict[str, Any]] = []
                for row_index in range(len(identities)):
                    row_flags, prediction = _row_flags(
                        cpu_batch, outputs_cpu, actions, row_index
                    )
                    flags.append(row_flags)
                    predictions.append(prediction)
                state_results[name] = {
                    "outputs": outputs,
                    "outputs_cpu": outputs_cpu,
                    "actions": [[int(value) for value in row] for row in actions],
                    "flags": flags,
                    "predictions": predictions,
                }
            _update_logit_audit(
                logit_audit,
                identities,
                state_results["raw"]["outputs"],
                state_results["beta100"]["outputs"],
                runtime,
            )
            for index, identity in enumerate(identities):
                _update_transition_book(
                    book,
                    identity,
                    state_results["raw"]["flags"][index],
                    state_results["beta100"]["flags"][index],
                    state_results["raw"]["predictions"][index],
                    state_results["beta100"]["predictions"][index],
                )
            if row_consumer is not None:
                row_consumer(
                    panel,
                    cpu_batch,
                    identities,
                    {
                        "outputs_cpu": state_results["raw"]["outputs_cpu"],
                        "actions": state_results["raw"]["actions"],
                        "flags": state_results["raw"]["flags"],
                        "predictions": state_results["raw"]["predictions"],
                    },
                    {
                        "outputs_cpu": state_results["beta100"]["outputs_cpu"],
                        "actions": state_results["beta100"]["actions"],
                        "flags": state_results["beta100"]["flags"],
                        "predictions": state_results["beta100"]["predictions"],
                    },
                )
            rows += len(identities)
            del state_results
        dataset_audits[panel] = _finalize_dataset(panel, dataset)
        summaries = {
            name: accumulator.summary() for name, accumulator in accumulators.items()
        }
        aggregates = {
            name: _aggregate_counts(summary) for name, summary in summaries.items()
        }
        if aggregates != EXPECTED_AGGREGATES[panel]:
            raise RuntimeError(
                f"{panel} frozen aggregate drift: observed={aggregates}, "
                f"expected={EXPECTED_AGGREGATES[panel]}"
            )
        transitions = _finalize_transition_book(
            panel, book, aggregates["raw"], aggregates["beta100"]
        )
        panels[panel] = {
            "rows": rows,
            "aggregates": aggregates,
            "aggregate_delta": {
                metric: aggregates["beta100"][metric] - aggregates["raw"][metric]
                for metric in ALL_METRICS
            },
            "policy_metric_transitions": {
                metric: transitions[metric] for metric in POLICY_METRICS
            },
            "count_value_correctness_transitions": {
                metric: transitions[metric] for metric in COUNT_VALUE_METRICS
            },
            "transitions": transitions,
            "count_value_logit_relations": _finalize_logit_audit(logit_audit, rows),
            "native_output_dtypes": dtype_evidence,
            "frozen_aggregate_counts_exact": True,
            "all_transition_net_identities_exact": True,
        }
    return panels, dataset_audits


def run_transition_audit(
    *,
    row_consumer: Callable[..., Any] | None = None,
    candidate_consumer: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run and return the fixed complete-state transition audit in memory.

    ``row_consumer``, when supplied, receives exactly five positional values
    for every same-batch evaluation: ``panel, cpu_batch, identities,
    raw_snapshot, beta_snapshot``.  Snapshots contain CPU logits, actions,
    flags and predictions.  ``candidate_consumer`` receives the completed
    report once.  Both are in-memory callbacks; this audit never writes an
    artifact.
    """

    _validate_runtime()
    if row_consumer is not None and not callable(row_consumer):
        raise TypeError("row_consumer must be callable or None")
    if candidate_consumer is not None and not callable(candidate_consumer):
        raise TypeError("candidate_consumer must be callable or None")
    source, source_evidence = _read_regular_bytes(SCRIPT, None, "transition tool")
    fixed, payloads = _fixed_inputs()
    cache = _cache_from_payloads(payloads)
    runtime = _load_runtime()
    torch = runtime.torch
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("transition run requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("transition run requires native CUDA BF16 support")
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")

    raw_checkpoint = _load_checkpoint(payloads["raw"], "raw U468", runtime)
    beta_checkpoint = _load_checkpoint(payloads["beta100"], "beta100", runtime)
    state_integrity = _validate_complete_states(
        raw_checkpoint, beta_checkpoint, runtime
    )
    raw_model, raw_config = _instantiate_complete_model(
        raw_checkpoint, runtime, device
    )
    beta_model, beta_config = _instantiate_complete_model(
        beta_checkpoint, runtime, device
    )
    if raw_model is beta_model or raw_config != beta_config:
        raise RuntimeError("models were not independently and compatibly instantiated")
    live_hashes = {
        "raw": _model_state_sha256(raw_model.state_dict(), runtime),
        "beta100": _model_state_sha256(beta_model.state_dict(), runtime),
    }
    if live_hashes != {"raw": RAW_MODEL_SHA256, "beta100": BETA_MODEL_SHA256}:
        raise RuntimeError(f"live complete model hash drift: {live_hashes}")

    started = time.time()
    panels, dataset_audits = _evaluate_transitions(
        raw_model,
        beta_model,
        raw_config,
        {panel: payloads[panel] for panel in PANEL_ORDER},
        runtime,
        device,
        row_consumer,
    )
    result = {
        "schema_version": SCHEMA,
        "status": "complete_train_transition_audit_passed",
        "mode": "run",
        "source": source_evidence,
        "ast_audit": _ast_audit(source),
        "synthetic_self_test": _synthetic_self_test(),
        "fixed_inputs": fixed,
        "cache": cache,
        "state_integrity": state_integrity,
        "live_complete_model_state_sha256": live_hashes,
        "panels": panels,
        "execution": {
            "device": "cuda:0",
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "seed": SEED,
            "native_bfloat16": True,
            "policy_greedy": True,
            "deterministic": True,
            "canonicalize_order": False,
            "model_instance_count": 2,
            "complete_checkpoint_instances": ["raw", "beta100"],
            "actor_only_overlay_used": False,
            "same_batch_raw_then_beta100": True,
            "dataset_audits": dataset_audits,
            "seconds": time.time() - started,
        },
        "scope": {
            "writes_performed": False,
            "stdout_only_when_cli": True,
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "training_backward_optimizer": False,
            "model_serialization": False,
            "network": False,
        },
    }
    if candidate_consumer is not None:
        candidate_consumer(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "cache", "run"), default="static")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "static":
            result = static_audit()
        elif args.mode == "cache":
            result = cache_audit()
        else:
            result = run_transition_audit()
    except Exception as error:
        result = {
            "schema_version": SCHEMA,
            "status": "failed",
            "mode": args.mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "writes_performed": False,
        }
        print(canonical_json(result).decode("utf-8"), end="")
        raise SystemExit(1) from None
    print(canonical_json(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
