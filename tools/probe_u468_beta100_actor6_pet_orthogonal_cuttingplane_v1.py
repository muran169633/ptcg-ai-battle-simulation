#!/usr/bin/env python3
"""Beta100 train-only PET-orthogonal actor6 cutting-plane probe.

This module is the endpoint/geometry half of a two-part audit.  A separately
hash-bound transition module owns the complete raw-U468 versus beta100
official-B256 train stream and returns only train-backed canonical contexts.
This module turns the harmful FLG/core5 transitions into repair obligations,
protects favorable PokemonFan transitions and beta-near correct rows, and
solves a minimum-L2 actor6 correction in the strict orthogonal complement of
the frozen P/E/T span.

All endpoints live in RAM.  The standalone entry point never supplies a
candidate consumer, never serializes the 65,793-vector, never writes a result
or checkpoint, and emits exactly one JSON document on stdout.  A caller may
provide ``candidate_consumer`` to ``run_probe``; it is invoked only for a
passing live endpoint and before the unconditional beta100 restoration.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import optimize


sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "probe_u468_beta100_actor6_pet_orthogonal_cuttingplane_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta100-actor6-pet-orthogonal-cuttingplane-v1"

PET_TOOL = TOOLS / "probe_u468_actor6_pet_cone_geometry.py"
PET_TOOL_SHA256 = "4f233db368f0150ee68109ee7d974a5a647ce8a35c919a5217f1714dddd98b2f"
TRANSITION_TOOL = TOOLS / "probe_u468_beta100_train_transitions_v1.py"
TRANSITION_TOOL_SHA256 = "5f25ffbd54705b282edf4512195bc83fe783ef300bc42d9361b48c627d0bffbb"

SEED = 202608126
METRICS = ("set_exact", "hybrid_order_exact", "ordered_exact", "top1_correct")
FULLTRAIN_POLICY_METRICS = METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
TARGET_SOURCES = ("flg", "core5")
PF_SOURCE = "pokemonfan"
NEAR_GUARD_PER_SOURCE = 32
INITIAL_CUMULATIVE_L2 = 5.0e-4
EXPANDED_CUMULATIVE_L2 = 1.0e-3
MAX_RADIUS_EXPANSIONS = 1
MAX_ROUNDS = 8
MAX_ENDPOINTS = 6
SVD_RELATIVE_RANK_TOLERANCE = 1.0e-12
QP_FTOL = 1.0e-12
QP_MAXITER = 5000
QP_LINEAR_TOLERANCE = 2.0e-8
ORTHOGONAL_ABS_TOLERANCE = 2.0e-9
ORTHOGONAL_RELATIVE_L2_TOLERANCE = 1.0e-5
RADIUS_ABS_TOLERANCE = 1.0e-12
PAIR_TOLERANCE = 0.0
DIRECT_LOSS_ABS_TOLERANCE = 1.0e-7

EXPECTED_TARGETS = {
    "9417c4667da3af128f684cf5b847952b47686a329af28a94cbf01615f3de8644": {
        "source": "flg",
        "metrics": ("ordered_exact",),
    },
    "5b06cd5facc20017464010692049d50ff20ed25adadaf25a3bd773e217e1ac67": {
        "source": "core5",
        "metrics": METRICS,
    },
    "002afbcdd4f248353808c4d743785d1a0aabbdee0627486352945749a57d81b2": {
        "source": "core5",
        "metrics": METRICS,
    },
}
EXPECTED_PF_GAIN_GUARDS = {
    "3b8e123d82ea6b8233d3d6088830527ec635f0ac363e72bb95bbd8a99d99cd21": {
        "source": "pokemonfan",
        "metrics": METRICS,
    }
}
TERMINAL_POLICY_DENOMINATORS = {
    "flg": {
        "set_exact": 9443,
        "hybrid_order_exact": 9443,
        "ordered_exact": 9443,
        "top1_correct": 9426,
        "context34_hybrid_order_exact": 42,
        "context34_ordered_exact": 42,
    },
    "pokemonfan": {
        "set_exact": 9487,
        "hybrid_order_exact": 9487,
        "ordered_exact": 9487,
        "top1_correct": 9450,
        "context34_hybrid_order_exact": 38,
        "context34_ordered_exact": 38,
    },
    "core5": {
        "set_exact": 5120,
        "hybrid_order_exact": 5120,
        "ordered_exact": 5120,
        "top1_correct": 5106,
        "context34_hybrid_order_exact": 20,
        "context34_ordered_exact": 20,
    },
}
TERMINAL_MINIMUM_CANDIDATE_COUNTS = {
    "flg": {
        "set_exact": 7239,
        "hybrid_order_exact": 7156,
        "ordered_exact": 7097,
        "top1_correct": 7308,
    },
    "pokemonfan": {
        "set_exact": 8280,
        "hybrid_order_exact": 8279,
        "ordered_exact": 8201,
        "top1_correct": 8329,
    },
    "core5": {
        "set_exact": 4073,
        "hybrid_order_exact": 4042,
        "ordered_exact": 4025,
        "top1_correct": 4130,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
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


def vector_sha256_float64(vector: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype="<f8"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def vector_sha256_float32(vector: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype="<f4"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def read_regular_bytes(
    path: Path, expected_sha256: str | None, label: str
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
        raise RuntimeError(
            f"{label} SHA-256 drift: expected {expected_sha256}, observed {digest}"
        )
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
        "single_link_regular_held_fd_identity_exact": True,
    }


def import_locked(
    path: Path, expected_sha256: str, module_name: str
) -> tuple[ModuleType, dict[str, Any]]:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RuntimeError(f"{module_name} does not yet have a frozen SHA-256")
    payload, evidence = read_regular_bytes(path, expected_sha256, module_name)
    if any(token in str(path).lower() for token in ("valid", "cw11", "cw12", "official6")):
        raise RuntimeError(f"forbidden consumed-valid dependency: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_{module_name}_{sha256_bytes(payload)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import spec for {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


PET, PET_EVIDENCE = import_locked(PET_TOOL, PET_TOOL_SHA256, "pet_geometry")
torch = PET.torch
ppo = PET.ppo


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")
    if tuple(PET.ACTOR_NAMES) != tuple(PET.EXPECTED_ACTOR_NAMES):
        raise RuntimeError("PET actor6 scope drift")
    if PET.EXPECTED_ACTOR_ELEMENTS != 65793:
        raise RuntimeError("PET actor6 dimension drift")


def static_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_calls = {
        "backward",
        "step",
        "save",
        "savez",
        "write",
        "write_text",
        "write_bytes",
        "touch",
        "mkdir",
        "makedirs",
        "unlink",
        "remove",
        "rmtree",
        "rename",
        "replace",
    }
    forbidden_import_roots = {
        "requests",
        "urllib",
        "http",
        "socket",
        "subprocess",
        "kaggle",
    }
    calls: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    copy_sites: list[int] = []
    print_sites: list[int] = []
    write_flags: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                if name.split(".", 1)[0] in forbidden_import_roots:
                    imports.append({"line": node.lineno, "name": name})
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"}
        ):
            write_flags.append({"line": node.lineno, "name": node.attr})
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = ""
        if name in forbidden_calls:
            calls.append({"line": node.lineno, "name": name})
        if name == "copy_":
            copy_sites.append(node.lineno)
        if name == "print":
            print_sites.append(node.lineno)
    signature = inspect.signature(run_probe)
    checks = {
        "ast_parse": True,
        "no_write_optimizer_backward_step_save_or_network": not (
            calls or imports or write_flags
        ),
        "single_ram_parameter_copy_site": len(copy_sites) == 1,
        "single_stdout_print_site": len(print_sites) == 1,
        "candidate_consumer_default_none": (
            "candidate_consumer" in signature.parameters
            and signature.parameters["candidate_consumer"].default is None
        ),
        "max_rounds_at_most_8": MAX_ROUNDS <= 8,
        "max_endpoints_at_most_6": MAX_ENDPOINTS <= 6,
        "initial_radius_exact": INITIAL_CUMULATIVE_L2 == 5.0e-4,
        "one_expansion_to_exact_radius": (
            MAX_RADIUS_EXPANSIONS == 1
            and EXPANDED_CUMULATIVE_L2 == 1.0e-3
        ),
        "hard_radius_numerical_tolerance_at_most_1e_12": (
            RADIUS_ABS_TOLERANCE <= 1.0e-12
        ),
        "pet_absolute_tolerance_exact_2e_9": (
            ORTHOGONAL_ABS_TOLERANCE == 2.0e-9
        ),
        "pet_relative_l2_tolerance_exact_1e_5": (
            ORTHOGONAL_RELATIVE_L2_TOLERANCE == 1.0e-5
        ),
        "terminal_policy_metrics_include_main4_and_context34_2": (
            len(FULLTRAIN_POLICY_METRICS) == 6
            and FULLTRAIN_POLICY_METRICS[:4] == METRICS
        ),
        "exact_target_count_3_and_pf_gain_count_1": (
            len(EXPECTED_TARGETS) == 3 and len(EXPECTED_PF_GAIN_GUARDS) == 1
        ),
        "direct_loss_tolerance_fixed": DIRECT_LOSS_ABS_TOLERANCE == 1.0e-7,
        "standalone_consumer_is_none": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"static audit failed: {checks}")
    return {
        "status": "static_zero_write_ram_endpoint_audit_passed",
        "checks": checks,
        "forbidden_call_hits": calls,
        "forbidden_import_hits": imports,
        "forbidden_os_write_flags": write_flags,
        "ram_copy_sites": copy_sites,
        "stdout_print_sites": print_sites,
    }


def load_transition_module() -> tuple[ModuleType, dict[str, Any]]:
    module, evidence = import_locked(
        TRANSITION_TOOL, TRANSITION_TOOL_SHA256, "beta100_train_transitions"
    )
    required = ("cache_audit", "run_transition_audit")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"transition module API missing: {missing}")
    return module, evidence


def ordered_selection_margin(
    logits: Any, option_mask: Any, expert_order: Sequence[int]
) -> float:
    remaining = option_mask.bool().clone()
    margins: list[float] = []
    for chosen in expert_order:
        chosen = int(chosen)
        if chosen < 0 or chosen >= int(remaining.shape[0]) or not bool(remaining[chosen]):
            raise RuntimeError("illegal expert action in transition callback")
        competitors = remaining.nonzero(as_tuple=False).squeeze(1)
        competitors = competitors[competitors != chosen]
        if competitors.numel() > 0:
            margins.append(float(logits[chosen] - logits[competitors].max()))
        remaining[chosen] = False
    return min(margins, default=float("inf"))


def transition_absolute_closure(report: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for panel in ("flg", "pokemonfan", "core5"):
        panel_report = report["panels"][panel]
        for metric in tuple(getattr(report.get("_module"), "ALL_METRICS", ())) or (
            "set_exact",
            "hybrid_order_exact",
            "ordered_exact",
            "top1_correct",
            "context34_hybrid_order_exact",
            "context34_ordered_exact",
            "count_correct",
            "value_correct",
        ):
            transition = panel_report["transitions"][metric]
            cells = transition["cells"]
            raw_from_cells = int(cells["cc"]["count"]) + int(cells["cw"]["count"])
            beta_from_cells = int(cells["cc"]["count"]) + int(cells["wc"]["count"])
            passed = (
                raw_from_cells == int(transition["raw_correct"])
                and beta_from_cells == int(transition["beta100_correct"])
            )
            records.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "raw_cc_plus_cw": raw_from_cells,
                    "raw_correct": int(transition["raw_correct"]),
                    "beta_cc_plus_wc": beta_from_cells,
                    "beta100_correct": int(transition["beta100_correct"]),
                    "pass": passed,
                }
            )
    if len(records) != 24 or not all(item["pass"] for item in records):
        raise RuntimeError("transition 3-panel x 8-metric absolute closure failed")
    return {"checks": records, "check_count_exact_24": True, "pass": True}


def build_actual_transition_context(transition: ModuleType) -> dict[str, Any]:
    batch_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    identity_digests = {panel: hashlib.sha256() for panel in transition.PANEL_ORDER}
    primary: dict[tuple[str, str], dict[str, Any]] = {}
    near_pool: dict[str, list[dict[str, Any]]] = {
        panel: [] for panel in transition.PANEL_ORDER
    }

    def callback(
        panel: str,
        cpu_batch: Mapping[str, Any],
        identities: Sequence[Mapping[str, Any]],
        raw_snapshot: Mapping[str, Any],
        beta_snapshot: Mapping[str, Any],
    ) -> None:
        batch_index = batch_counts[panel]
        batch_counts[panel] += 1
        batch_key = f"{panel}:B{batch_index:03d}"
        for local_index, identity_source in enumerate(identities):
            identity = dict(identity_source)
            identity_digests[panel].update(
                transition.canonical_json(
                    {
                        "panel": panel,
                        "member": identity["member"],
                        "line_index_zero_based": int(identity["line_index_zero_based"]),
                        "line_sha256": identity["line_sha256"],
                    }
                )
            )
            raw_flags = {
                metric: bool(raw_snapshot["flags"][local_index][metric])
                for metric in METRICS
            }
            beta_flags = {
                metric: bool(beta_snapshot["flags"][local_index][metric])
                for metric in METRICS
            }
            line_sha = str(identity["line_sha256"])
            if panel in TARGET_SOURCES:
                changed = [
                    metric
                    for metric in METRICS
                    if raw_flags[metric] and not beta_flags[metric]
                ]
                role = "target_transition"
            elif panel == PF_SOURCE:
                changed = [
                    metric
                    for metric in METRICS
                    if not raw_flags[metric] and beta_flags[metric]
                ]
                role = "pf_gain_guard"
            else:
                raise RuntimeError(f"unexpected transition panel: {panel}")
            if changed:
                key = (panel, line_sha)
                if key not in primary:
                    primary[key] = {
                        "identity": identity,
                        "source": panel,
                        "role": role,
                        "required_metrics": set(),
                        "batch_key": batch_key,
                        "local_index": local_index,
                        "raw_flags": raw_flags,
                        "beta_flags": beta_flags,
                        "cpu_batch": cpu_batch,
                    }
                primary[key]["required_metrics"].update(changed)
            expert = [int(value) for value in identity["expert_order"]]
            if expert and all(beta_flags.values()):
                selection_margin = ordered_selection_margin(
                    beta_snapshot["outputs_cpu"]["policy_logits"][local_index],
                    cpu_batch["option_mask"][local_index],
                    expert,
                )
                candidate = {
                    "identity": identity,
                    "source": panel,
                    "role": "near_guard",
                    "required_metrics": METRICS,
                    "batch_key": batch_key,
                    "local_index": local_index,
                    "raw_flags": raw_flags,
                    "beta_flags": beta_flags,
                    "beta_selection_margin": selection_margin,
                    "cpu_batch": cpu_batch,
                }
                near_pool[panel].append(candidate)
                near_pool[panel].sort(
                    key=lambda item: (
                        float(item["beta_selection_margin"]),
                        str(item["identity"]["line_sha256"]),
                    )
                )
                del near_pool[panel][NEAR_GUARD_PER_SOURCE + 8 :]

    report = transition.run_transition_audit(
        row_consumer=callback,
        candidate_consumer=None,
    )
    if (
        report.get("status") != "complete_train_transition_audit_passed"
        or report.get("scope", {}).get("validation_member_payloads_opened") is not False
    ):
        raise RuntimeError("transition dependency did not pass strict train-only audit")
    absolute_closure = transition_absolute_closure(report)
    expected_batches = {
        panel: math.ceil(int(transition.EXPECTED_TRAIN[panel]["rows"]) / 256)
        for panel in transition.PANEL_ORDER
    }
    identity_sha = {
        panel: identity_digests[panel].hexdigest() for panel in transition.PANEL_ORDER
    }
    if batch_counts != expected_batches or identity_sha != transition.EXPECTED_IDENTITY_SHA256:
        raise RuntimeError(
            f"external row_consumer coverage drift: {batch_counts=} {identity_sha=}"
        )

    descriptors = []
    for item in primary.values():
        item["required_metrics"] = tuple(
            metric for metric in METRICS if metric in item["required_metrics"]
        )
        descriptors.append(item)
    observed_targets = {
        item["identity"]["line_sha256"]: {
            "source": item["source"],
            "metrics": tuple(item["required_metrics"]),
        }
        for item in descriptors
        if item["role"] == "target_transition"
    }
    observed_pf_guards = {
        item["identity"]["line_sha256"]: {
            "source": item["source"],
            "metrics": tuple(item["required_metrics"]),
        }
        for item in descriptors
        if item["role"] == "pf_gain_guard"
    }
    if observed_targets != EXPECTED_TARGETS:
        raise RuntimeError(f"target transition identity drift: {observed_targets}")
    if observed_pf_guards != EXPECTED_PF_GAIN_GUARDS:
        raise RuntimeError(f"PokemonFan gain-guard identity drift: {observed_pf_guards}")
    primary_hashes = set(observed_targets) | set(observed_pf_guards)
    near_ledger = {}
    for panel in transition.PANEL_ORDER:
        candidates = [
            item
            for item in near_pool[panel]
            if item["identity"]["line_sha256"] not in primary_hashes
        ][:NEAR_GUARD_PER_SOURCE]
        if len(candidates) != NEAR_GUARD_PER_SOURCE:
            raise RuntimeError(f"{panel}: insufficient deterministic beta-near guards")
        descriptors.extend(candidates)
        near_ledger[panel] = [
            {
                "identity": item["identity"],
                "beta_selection_margin": float(item["beta_selection_margin"]),
            }
            for item in candidates
        ]

    descriptors.sort(
        key=lambda item: (
            {"target_transition": 0, "pf_gain_guard": 1, "near_guard": 2}[
                item["role"]
            ],
            str(item["source"]),
            str(item["identity"]["line_sha256"]),
        )
    )
    selected_batches: dict[str, dict[str, Any]] = {}
    for item in descriptors:
        key = str(item["batch_key"])
        selected_batches.setdefault(
            key, {"key": key, "cpu_batch": item.pop("cpu_batch")}
        )

    runtime = transition._load_runtime()
    beta_checkpoint = torch.load(
        PET.CHECKPOINT_SPECS["P"][0], map_location="cpu", weights_only=False
    )
    beta_model, model_config = transition._instantiate_complete_model(
        beta_checkpoint, runtime, torch.device("cuda:0")
    )
    if ppo.model_state_sha256(beta_model) != transition.BETA_MODEL_SHA256:
        raise RuntimeError("independently instantiated beta100 model SHA drift")
    return {
        "split": "train",
        "validation_opened": False,
        "batch_size": 256,
        "beta_model": beta_model,
        "beta_checkpoint": beta_checkpoint,
        "model_config": model_config,
        "beta_model_state_sha256": transition.BETA_MODEL_SHA256,
        "descriptors": descriptors,
        "selected_batches": [selected_batches[key] for key in sorted(selected_batches)],
        "transition_report": report,
        "absolute_closure": absolute_closure,
        "row_consumer_coverage": {
            "batch_counts": batch_counts,
            "expected_batch_counts": expected_batches,
            "identity_stream_sha256": identity_sha,
            "expected_identity_stream_sha256": transition.EXPECTED_IDENTITY_SHA256,
            "pass": True,
        },
        "near_selection_ledger": near_ledger,
        "near_margin_basis": "beta100 selection_margin",
        "_transition_module": transition,
    }


def project_orthogonal(vector: np.ndarray, q: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (PET.EXPECTED_ACTOR_ELEMENTS,):
        raise RuntimeError(f"actor6 vector shape drift: {value.shape}")
    result = value - q @ (q.T @ value)
    if not bool(np.all(np.isfinite(result))):
        raise FloatingPointError("nonfinite PET-orthogonal projection")
    return result


def projection_audit(vector: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    value = np.asarray(vector, dtype=np.float64)
    components = q.T @ value
    norm = float(np.linalg.norm(value))
    maximum = float(np.max(np.abs(components), initial=0.0))
    relative = float(np.linalg.norm(components)) / norm if norm > 0.0 else 0.0
    return {
        "l2": norm,
        "pet_components": [float(item) for item in components],
        "pet_component_max_abs": maximum,
        "pet_component_relative_l2": relative,
        "pet_component_abs_tolerance": ORTHOGONAL_ABS_TOLERANCE,
        "pet_component_relative_l2_tolerance": (
            ORTHOGONAL_RELATIVE_L2_TOLERANCE
        ),
        "strict_orthogonal_within_tolerance": (
            maximum <= ORTHOGONAL_ABS_TOLERANCE
            and relative <= ORTHOGONAL_RELATIVE_L2_TOLERANCE
        ),
    }


def solve_minimum_endpoint_l2(
    pair_gradients: np.ndarray,
    pair_rhs: np.ndarray,
    loss_gradients: np.ndarray,
    loss_rhs: np.ndarray,
    cumulative: np.ndarray,
    radius: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Solve for the minimum-L2 endpoint under linearized cuts and a ball."""
    if pair_gradients.ndim != 2 or pair_rhs.shape != (pair_gradients.shape[0],):
        raise RuntimeError("pair gradient/RHS shape drift")
    if loss_gradients.ndim != 2 or loss_gradients.shape[1] != pair_gradients.shape[1]:
        raise RuntimeError("loss gradient shape drift")
    if loss_gradients.shape[0] != 13:
        raise RuntimeError("direct512 loss-gradient count is not 13")
    if loss_rhs.shape != (13,):
        raise RuntimeError("direct512 loss RHS count is not 13")
    # g_pair dot correction >= pair_rhs.  For each loss, the linearized
    # beta-nonregression condition is
    #   -g dot correction >= current_loss - beta_loss - tolerance.
    matrix = np.concatenate((pair_gradients, -loss_gradients), axis=0)
    rhs = np.concatenate((pair_rhs, loss_rhs))
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise RuntimeError("cutting-plane constraint matrix has zero rank")
    rank = int(
        np.count_nonzero(
            singular_values
            > singular_values[0] * SVD_RELATIVE_RANK_TOLERANCE
        )
    )
    basis = vh[:rank]
    reduced = matrix @ basis.T

    # Parameterize the *endpoint* in the current constraint row-space.  The
    # minimum-norm feasible endpoint has no component in its null-space.  In
    # later rounds this is materially different from restricting only a new
    # correction to that row-space: the latter would strand obsolete
    # cumulative null components and can falsely collide with the radius.
    effective_rhs = rhs + matrix @ cumulative

    def endpoint(y: np.ndarray) -> np.ndarray:
        return basis.T @ y

    def objective(y: np.ndarray) -> float:
        return 0.5 * float(y @ y)

    def objective_jacobian(y: np.ndarray) -> np.ndarray:
        return y

    def radius_residual(y: np.ndarray) -> float:
        return radius - float(np.linalg.norm(endpoint(y)))

    def radius_jacobian(y: np.ndarray) -> np.ndarray:
        value = endpoint(y)
        norm = float(np.linalg.norm(value))
        if norm == 0.0:
            return np.zeros(rank, dtype=np.float64)
        return -(basis @ value) / norm

    solution = optimize.minimize(
        objective,
        np.zeros(rank, dtype=np.float64),
        jac=objective_jacobian,
        constraints=(
            {
                "type": "ineq",
                "fun": lambda y: reduced @ y - effective_rhs,
                "jac": lambda y: reduced,
            },
            {
                "type": "ineq",
                "fun": radius_residual,
                "jac": radius_jacobian,
            },
        ),
        method="SLSQP",
        options={"ftol": QP_FTOL, "maxiter": QP_MAXITER, "disp": False},
    )
    audit: dict[str, Any] = {
        "objective": "minimum cumulative actor6 endpoint L2",
        "solver": "scipy.optimize.minimize(method=SLSQP) CPU float64",
        "success": bool(solution.success),
        "status": int(solution.status),
        "message": str(solution.message),
        "iterations": int(solution.nit),
        "constraint_rows": int(matrix.shape[0]),
        "pair_rows": int(pair_gradients.shape[0]),
            "loss_slope_rows": int(loss_gradients.shape[0]),
            "loss_rhs_min": float(loss_rhs.min()),
            "loss_rhs_max": float(loss_rhs.max()),
        "svd_rank": rank,
        "radius": radius,
    }
    if not bool(solution.success) or solution.x is None:
        audit["failure"] = "infeasible_or_solver_failure_inside_radius"
        return None, audit
    candidate = endpoint(np.asarray(solution.x, dtype=np.float64))
    correction = candidate - cumulative
    linear_residual = matrix @ correction - rhs
    candidate_l2 = float(np.linalg.norm(candidate))
    audit.update(
        {
            "correction_l2": float(np.linalg.norm(correction)),
            "candidate_l2": candidate_l2,
            "linear_residual_min": float(linear_residual.min()),
            "pair_predicted_residual_min": float(
                (pair_gradients @ correction - pair_rhs).min()
            ),
            "loss_predicted_slope_max": float(
                (loss_gradients @ correction).max()
            ),
            "loss_predicted_constraint_residual_min": float(
                (-loss_gradients @ correction - loss_rhs).min()
            ),
            "radius_residual": radius - candidate_l2,
        }
    )
    if (
        float(linear_residual.min()) < -QP_LINEAR_TOLERANCE
        or candidate_l2 > radius + RADIUS_ABS_TOLERANCE
    ):
        audit["success"] = False
        audit["failure"] = "direct_recomputation_gate_failed"
        return None, audit
    return correction, audit


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def official_row_record(
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    actions: Sequence[Sequence[int]],
    row_index: int,
) -> dict[str, Any]:
    targets = cpu_batch["targets"][row_index].bool()
    option_mask = cpu_batch["option_mask"][row_index].bool()
    prediction = targets.new_zeros(targets.shape)
    order = [int(value) for value in actions[row_index]]
    if order:
        prediction[order] = True
    expert_count = int(cpu_batch["expert_ordered_action_counts"][row_index])
    expert_order = [
        int(value)
        for value in cpu_batch["expert_ordered_actions"][row_index, :expert_count]
    ]
    context = int(cpu_batch["contexts"][row_index])
    hybrid = order if context == ppo.SKILL_ORDER_CONTEXT else sorted(order)
    top1 = int(outputs_cpu["policy_logits"][row_index].argmax())
    return {
        "flags": {
            "set_exact": bool(((prediction == targets) | ~option_mask).all()),
            "hybrid_order_exact": hybrid == expert_order,
            "ordered_exact": order == expert_order,
            "top1_correct": bool(targets[top1]),
        },
        "predicted_order": order,
        "expert_order": expert_order,
        "context": context,
        "top1_index_full_policy_vector": top1,
        "predicted_count": len(order),
        "expert_count": int(cpu_batch["action_counts"][row_index]),
    }


def snapshot_batches(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    device: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with torch.no_grad():
        for item in batches:
            key = str(item["key"])
            cpu_batch = item["cpu_batch"]
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in cpu_batch.items()
            }
            outputs = ppo.model_forward(model, batch, device)
            if any(value.dtype != torch.bfloat16 for value in outputs.values()):
                raise RuntimeError(f"{key}: selected forward is not native BF16")
            actions, _, _, _ = ppo.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=False,
            )
            outputs_cpu = {
                name: value.detach().cpu().contiguous()
                for name, value in outputs.items()
            }
            result[key] = {
                "outputs_cpu": outputs_cpu,
                "official_rows": [
                    official_row_record(cpu_batch, outputs_cpu, actions, index)
                    for index in range(len(actions))
                ],
                "fingerprints": {
                    name: tensor_sha256(value)
                    for name, value in outputs_cpu.items()
                },
            }
    return result


def pair_key(pair: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(pair["batch_key"]),
        int(pair["local_index"]),
        int(pair["positive_option"]),
        int(pair["negative_option"]),
    )


def pair_margin(snapshot: Mapping[str, Any], pair: Mapping[str, Any]) -> float:
    outputs = snapshot[str(pair["batch_key"])]["outputs_cpu"]
    logits = outputs["policy_logits"]
    row = int(pair["local_index"])
    return float(
        logits[row, int(pair["positive_option"])].float()
        - logits[row, int(pair["negative_option"])].float()
    )


def local_positive_bf16_q(
    snapshot: Mapping[str, Any], pair: Mapping[str, Any]
) -> float:
    outputs = snapshot[str(pair["batch_key"])]["outputs_cpu"]
    logits = outputs["policy_logits"]
    row = int(pair["local_index"])
    values = (
        logits[row, int(pair["positive_option"])],
        logits[row, int(pair["negative_option"])],
    )
    spacings = []
    for value in values:
        spacings.extend(
            (
                torch.nextafter(value, torch.full_like(value, float("inf"))) - value,
                value - torch.nextafter(value, torch.full_like(value, float("-inf"))),
            )
        )
    result = max(float(value.float()) for value in spacings)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("invalid native-BF16 local positive quantum")
    return result


def first_different_pair(
    expert_order: Sequence[int], predicted_order: Sequence[int]
) -> tuple[int, int] | None:
    for expected, observed in zip(expert_order, predicted_order):
        if int(expected) != int(observed):
            return int(expected), int(observed)
    return None


def set_pair(
    expert_order: Sequence[int], predicted_order: Sequence[int]
) -> tuple[int, int] | None:
    expected_set = {int(value) for value in expert_order}
    predicted_set = {int(value) for value in predicted_order}
    missing = [int(value) for value in expert_order if int(value) not in predicted_set]
    included = [int(value) for value in predicted_order if int(value) not in expected_set]
    return (missing[0], included[0]) if missing and included else None


def top1_pair(
    descriptor: Mapping[str, Any],
    current_snapshot: Mapping[str, Any],
) -> tuple[int, int] | None:
    batch_key = str(descriptor["batch_key"])
    row = int(descriptor["local_index"])
    outputs = current_snapshot[batch_key]["outputs_cpu"]
    logits = outputs["policy_logits"][row].float()
    cpu_batch = descriptor["cpu_batch"]
    targets = cpu_batch["targets"][row].bool()
    # Official top1 is the argmax of the full policy vector, not the masked
    # action sampler.  The separating pair must therefore use the full target
    # and non-target coordinate sets as well.
    positive_indices = targets.nonzero(as_tuple=False).squeeze(1)
    negative_indices = (~targets).nonzero(as_tuple=False).squeeze(1)
    if positive_indices.numel() == 0 or negative_indices.numel() == 0:
        return None
    positive = int(positive_indices[torch.argmax(logits[positive_indices])])
    negative = int(negative_indices[torch.argmax(logits[negative_indices])])
    if positive == negative:
        return None
    return positive, negative


def derive_pair(
    descriptor: Mapping[str, Any],
    metric: str,
    current_snapshot: Mapping[str, Any],
) -> tuple[int, int] | None:
    record = current_snapshot[str(descriptor["batch_key"])]["official_rows"][
        int(descriptor["local_index"])
    ]
    expert = record["expert_order"]
    predicted = record["predicted_order"]
    if metric == "top1_correct":
        return top1_pair(descriptor, current_snapshot)
    if metric == "set_exact":
        return set_pair(expert, predicted)
    if metric == "hybrid_order_exact" and int(record["context"]) != ppo.SKILL_ORDER_CONTEXT:
        return set_pair(expert, predicted)
    if metric in {"hybrid_order_exact", "ordered_exact"}:
        return first_different_pair(expert, predicted)
    raise RuntimeError(f"unsupported official metric: {metric}")


def guard_support_pairs(
    descriptor: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[tuple[int, int, str]]:
    batch_key = str(descriptor["batch_key"])
    row = int(descriptor["local_index"])
    outputs = baseline[batch_key]["outputs_cpu"]
    logits = outputs["policy_logits"][row].float()
    cpu_batch = descriptor["cpu_batch"]
    option_mask = cpu_batch["option_mask"][row].bool().clone()
    official = baseline[batch_key]["official_rows"][row]
    result: list[tuple[int, int, str]] = []
    if "top1_correct" in descriptor["required_metrics"]:
        pair = top1_pair(descriptor, baseline)
        if pair is not None:
            result.append((pair[0], pair[1], "baseline_top1_support"))
    if any(
        metric in descriptor["required_metrics"]
        for metric in ("set_exact", "hybrid_order_exact", "ordered_exact")
    ):
        for stage, chosen in enumerate(official["expert_order"]):
            competitors = option_mask.nonzero(as_tuple=False).squeeze(1)
            competitors = competitors[competitors != int(chosen)]
            if competitors.numel() > 0:
                values = logits[competitors]
                best = values.max()
                for competitor in competitors[values == best].tolist():
                    result.append(
                        (int(chosen), int(competitor), f"baseline_stage_{stage}_support")
                    )
            option_mask[int(chosen)] = False
    return result


def add_or_strengthen_pair(
    active: dict[tuple[str, int, int, int], dict[str, Any]],
    descriptor: Mapping[str, Any],
    positive: int,
    negative: int,
    threshold: float,
    origin: str,
    round_index: int,
) -> dict[str, Any]:
    candidate = {
        "batch_key": str(descriptor["batch_key"]),
        "local_index": int(descriptor["local_index"]),
        "identity": dict(descriptor["identity"]),
        "row_role": str(descriptor["role"]),
        "positive_option": int(positive),
        "negative_option": int(negative),
        "threshold": float(threshold),
        "origins": [origin],
        "created_round": round_index,
    }
    key = pair_key(candidate)
    if key not in active:
        active[key] = candidate
        return {"kind": "added", "key": list(key), "threshold": float(threshold)}
    existing = active[key]
    previous = float(existing["threshold"])
    existing["threshold"] = max(previous, float(threshold))
    if origin not in existing["origins"]:
        existing["origins"].append(origin)
    return {
        "kind": "strengthened" if float(existing["threshold"]) > previous else "deduplicated",
        "key": list(key),
        "previous": previous,
        "threshold": float(existing["threshold"]),
    }


def validate_transition_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if context.get("split") != "train" or context.get("validation_opened") is not False:
        raise RuntimeError("transition context is not strict train-only")
    if context.get("batch_size") != 256:
        raise RuntimeError("transition context is not official B256")
    descriptors = context.get("descriptors")
    batches = context.get("selected_batches")
    if not isinstance(descriptors, list) or not descriptors:
        raise RuntimeError("transition context has no selected descriptors")
    if not isinstance(batches, list) or not batches:
        raise RuntimeError("transition context has no canonical selected batches")
    batch_by_key = {str(item["key"]): item for item in batches}
    if len(batch_by_key) != len(batches):
        raise RuntimeError("transition context repeats a canonical batch key")
    role_counts: dict[str, int] = {}
    seen: set[tuple[str, int, str]] = set()
    for descriptor in descriptors:
        role = str(descriptor["role"])
        source = str(descriptor["source"])
        metrics = tuple(str(value) for value in descriptor["required_metrics"])
        identity = descriptor["identity"]
        if role not in {"target_transition", "pf_gain_guard", "near_guard"}:
            raise RuntimeError(f"unexpected transition role: {role}")
        if not metrics or any(metric not in METRICS for metric in metrics):
            raise RuntimeError("descriptor official metric drift")
        if role == "target_transition" and source not in TARGET_SOURCES:
            raise RuntimeError("only FLG/core5 harmful transitions may be targets")
        if role == "pf_gain_guard" and source != PF_SOURCE:
            raise RuntimeError("only PokemonFan favorable transitions may be gain guards")
        if role == "near_guard" and source not in (*TARGET_SOURCES, PF_SOURCE):
            raise RuntimeError("near guard source drift")
        key = str(descriptor["batch_key"])
        local = int(descriptor["local_index"])
        if key not in batch_by_key:
            raise RuntimeError("descriptor references an absent canonical batch")
        cpu_batch = batch_by_key[key]["cpu_batch"]
        if not (0 <= local < int(cpu_batch["action_counts"].shape[0])):
            raise RuntimeError("descriptor local index outside canonical batch")
        descriptor["cpu_batch"] = cpu_batch
        occurrence = (key, local, role)
        if occurrence in seen:
            raise RuntimeError("duplicate descriptor role occurrence")
        seen.add(occurrence)
        role_counts[role] = role_counts.get(role, 0) + 1
        member = str(identity.get("member", ""))
        if not member.startswith("train/") or "valid" in member.lower():
            raise RuntimeError("descriptor is not backed by a train member")
    if role_counts.get("target_transition", 0) < 1:
        raise RuntimeError("no train harmful transition target was selected")
    if role_counts.get("pf_gain_guard", 0) < 1:
        raise RuntimeError("no PokemonFan train gain guard was selected")
    near_sources = {
        str(item["source"])
        for item in descriptors
        if item["role"] == "near_guard"
    }
    if near_sources != {"flg", "pokemonfan", "core5"}:
        raise RuntimeError(f"near-margin source coverage drift: {near_sources}")
    expected_role_counts = {
        "target_transition": 3,
        "pf_gain_guard": 1,
        "near_guard": 3 * NEAR_GUARD_PER_SOURCE,
    }
    if role_counts != expected_role_counts:
        raise RuntimeError(
            f"transition/near descriptor count drift: {role_counts}"
        )
    row_identities = [
        (
            str(item["source"]),
            str(item["identity"]["member"]),
            int(item["identity"]["line_index_zero_based"]),
            str(item["identity"]["line_sha256"]),
        )
        for item in descriptors
    ]
    if len(row_identities) != len(set(row_identities)):
        raise RuntimeError("selected descriptor identities overlap across roles")
    target_source_counts = {
        source: sum(
            item["role"] == "target_transition" and item["source"] == source
            for item in descriptors
        )
        for source in TARGET_SOURCES
    }
    if target_source_counts != {"flg": 1, "core5": 2}:
        raise RuntimeError(f"target panel count drift: {target_source_counts}")
    observed_targets = {
        item["identity"]["line_sha256"]: {
            "source": item["source"],
            "metrics": tuple(item["required_metrics"]),
        }
        for item in descriptors
        if item["role"] == "target_transition"
    }
    observed_pf_guards = {
        item["identity"]["line_sha256"]: {
            "source": item["source"],
            "metrics": tuple(item["required_metrics"]),
        }
        for item in descriptors
        if item["role"] == "pf_gain_guard"
    }
    if observed_targets != EXPECTED_TARGETS:
        raise RuntimeError("exact 3-target identity/metric ledger drift")
    if observed_pf_guards != EXPECTED_PF_GAIN_GUARDS:
        raise RuntimeError("exact one-PokemonFan-gain identity/metric ledger drift")
    near_counts = {
        source: sum(
            item["role"] == "near_guard" and item["source"] == source
            for item in descriptors
        )
        for source in ("flg", "pokemonfan", "core5")
    }
    if any(value != NEAR_GUARD_PER_SOURCE for value in near_counts.values()):
        raise RuntimeError(f"near guard per-source count drift: {near_counts}")
    return {
        "descriptor_count": len(descriptors),
        "canonical_batch_count": len(batches),
        "role_counts": role_counts,
        "target_source_counts": target_source_counts,
        "descriptor_identities_unique_and_role_disjoint": True,
        "near_guard_counts": near_counts,
        "near_guard_sources": sorted(near_sources),
        "near_selection_ledger": context.get("near_selection_ledger"),
        "row_consumer_coverage": context.get("row_consumer_coverage"),
        "transition_absolute_closure": context.get("absolute_closure"),
        "transition_selection_sha256": sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "identity": item["identity"],
                        "source": item["source"],
                        "role": item["role"],
                        "required_metrics": list(item["required_metrics"]),
                        "batch_key": item["batch_key"],
                        "local_index": item["local_index"],
                    }
                    for item in descriptors
                ]
            )
        ),
    }


def gate_selected(
    descriptors: Sequence[Mapping[str, Any]],
    active: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    false_obligations: list[dict[str, Any]] = []
    row_gates: list[dict[str, Any]] = []
    for descriptor_index, descriptor in enumerate(descriptors):
        official = snapshot[str(descriptor["batch_key"])]["official_rows"][
            int(descriptor["local_index"])
        ]
        flags = {
            metric: bool(official["flags"][metric])
            for metric in descriptor["required_metrics"]
        }
        for metric, passed in flags.items():
            if not passed:
                false_obligations.append(
                    {
                        "descriptor_index": descriptor_index,
                        "role": descriptor["role"],
                        "metric": metric,
                    }
                )
        row_gates.append(
            {
                "identity": descriptor["identity"],
                "role": descriptor["role"],
                "source": descriptor["source"],
                "required_metrics": list(descriptor["required_metrics"]),
                "flags": flags,
                "pass": all(flags.values()),
            }
        )
    pair_gates = []
    for key in sorted(active):
        pair = active[key]
        margin = pair_margin(snapshot, pair)
        threshold = float(pair["threshold"])
        pair_gates.append(
            {
                "key": list(key),
                "role": pair["row_role"],
                "margin": margin,
                "threshold": threshold,
                "residual": margin - threshold,
                "pass": margin + PAIR_TOLERANCE >= threshold,
            }
        )
    return {
        "pass": not false_obligations and all(item["pass"] for item in pair_gates),
        "false_obligations": false_obligations,
        "row_gate_count": len(row_gates),
        "row_gates": row_gates,
        "active_pair_count": len(pair_gates),
        "active_pair_residual_min": (
            min(item["residual"] for item in pair_gates) if pair_gates else None
        ),
        "active_pairs_all_pass": all(item["pass"] for item in pair_gates),
    }


def add_dynamic_cuts(
    descriptors: Sequence[Mapping[str, Any]],
    false_obligations: Sequence[Mapping[str, Any]],
    active: dict[tuple[str, int, int, int], dict[str, Any]],
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    round_index: int,
) -> dict[str, Any]:
    updates = []
    unresolved = []
    for obligation in false_obligations:
        descriptor = descriptors[int(obligation["descriptor_index"])]
        metric = str(obligation["metric"])
        pair = derive_pair(descriptor, metric, current)
        if pair is None or pair[0] == pair[1]:
            unresolved.append(
                {
                    "identity": descriptor["identity"],
                    "metric": metric,
                    "reason": "false_official_cell_has_no_actor_pair",
                }
            )
            continue
        shell = {
            "batch_key": descriptor["batch_key"],
            "local_index": descriptor["local_index"],
            "positive_option": pair[0],
            "negative_option": pair[1],
        }
        if descriptor["role"] == "target_transition":
            threshold = local_positive_bf16_q(current, shell)
            source = "current_native_bf16_positive_q"
        else:
            threshold = pair_margin(baseline, shell)
            if threshold < 0.0:
                unresolved.append(
                    {
                        "identity": descriptor["identity"],
                        "metric": metric,
                        "reason": "baseline_guard_pair_margin_negative",
                        "margin": threshold,
                    }
                )
                continue
            source = "same_process_beta100_guard_margin"
        updates.append(
            add_or_strengthen_pair(
                active,
                descriptor,
                pair[0],
                pair[1],
                threshold,
                f"dynamic_{metric}:{source}",
                round_index,
            )
        )
    return {"updates": updates, "unresolved": unresolved, "pass": not unresolved}


def initialize_guard_pairs(
    descriptors: Sequence[Mapping[str, Any]],
    active: dict[tuple[str, int, int, int], dict[str, Any]],
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    updates = []
    for descriptor in descriptors:
        if descriptor["role"] == "target_transition":
            continue
        for positive, negative, origin in guard_support_pairs(descriptor, baseline):
            shell = {
                "batch_key": descriptor["batch_key"],
                "local_index": descriptor["local_index"],
                "positive_option": positive,
                "negative_option": negative,
            }
            threshold = pair_margin(baseline, shell)
            if threshold < 0.0:
                raise RuntimeError("beta100 guard support pair is negative")
            updates.append(
                add_or_strengthen_pair(
                    active,
                    descriptor,
                    positive,
                    negative,
                    threshold,
                    f"initial_guard:{origin}",
                    0,
                )
            )
    return updates


def collect_pair_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    active: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
    current: Mapping[str, Any],
    q: np.ndarray,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    by_batch: dict[str, list[Mapping[str, Any]]] = {}
    for key in sorted(active):
        pair = active[key]
        by_batch.setdefault(str(pair["batch_key"]), []).append(pair)
    batch_by_key = {str(item["key"]): item for item in batches}
    gradients: list[np.ndarray] = []
    rhs: list[float] = []
    projection_records = []
    for batch_key in sorted(by_batch):
        cpu_batch = batch_by_key[batch_key]["cpu_batch"]
        batch = {
            name: value.to(device, non_blocking=True)
            for name, value in cpu_batch.items()
        }
        outputs = ppo.model_forward(model, batch, device)
        if not all(
            torch.equal(outputs[name].detach().cpu(), current[batch_key]["outputs_cpu"][name])
            for name in outputs
        ):
            raise RuntimeError("gradient/no-grad selected forward mismatch")
        pairs = by_batch[batch_key]
        margins = [
            outputs["policy_logits"][
                int(pair["local_index"]), int(pair["positive_option"])
            ].float()
            - outputs["policy_logits"][
                int(pair["local_index"]), int(pair["negative_option"])
            ].float()
            for pair in pairs
        ]
        for index, (pair, margin) in enumerate(zip(pairs, margins)):
            full = PET.flat_actor6_gradient(
                margin,
                parameters,
                retain_graph=index + 1 < len(margins),
            ).numpy()
            projected = project_orthogonal(full, q)
            audit = projection_audit(projected, q)
            if not audit["strict_orthogonal_within_tolerance"]:
                raise RuntimeError("pair gradient PET projection residual too large")
            gradients.append(projected)
            observed = float(margin.detach().cpu())
            rhs.append(float(pair["threshold"]) - observed)
            projection_records.append(
                {
                    "key": list(pair_key(pair)),
                    "full_l2": float(np.linalg.norm(full)),
                    "projected_l2": audit["l2"],
                    "pet_component_max_abs": audit["pet_component_max_abs"],
                }
            )
    if not gradients:
        raise RuntimeError("no active pair gradients")
    return (
        np.stack(gradients, axis=0),
        np.asarray(rhs, dtype=np.float64),
        {
            "count": len(gradients),
            "records": projection_records,
            "rhs_min": float(min(rhs)),
            "rhs_max": float(max(rhs)),
        },
    )


def collect_projected_loss_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    direct_union: Mapping[str, Any],
    masks: Mapping[str, Any],
    q: np.ndarray,
    device: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    records = PET.collect_direct_loss_gradients(
        model, parameters, direct_union, masks, device
    )
    if [str(item["name"]) for item in records] != list(PET.LOSS_NAMES):
        raise RuntimeError("direct512 13-loss order drift")
    rows = []
    report = []
    for item in records:
        full = item.pop("gradient").numpy()
        projected = project_orthogonal(full, q)
        audit = projection_audit(projected, q)
        if not audit["strict_orthogonal_within_tolerance"]:
            raise RuntimeError("loss gradient PET projection residual too large")
        rows.append(projected)
        report.append(
            {
                **item,
                "full_gradient_l2": float(np.linalg.norm(full)),
                "projected_gradient_l2": audit["l2"],
                "pet_component_max_abs": audit["pet_component_max_abs"],
            }
        )
    return np.stack(rows, axis=0), {"count": len(rows), "records": report}


def evaluate_direct_losses(
    model: Any,
    direct_union: Mapping[str, Any],
    masks_cpu: Mapping[str, Any],
    device: Any,
) -> dict[str, float]:
    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in direct_union.items()
    }
    with torch.no_grad():
        outputs = ppo.model_forward(model, batch, device)
        per_row = PET.frozen.ordered_nll_per_row(outputs, batch)
        weights = batch["sample_weights"].float() * torch.where(
            batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
            torch.full_like(
                batch["sample_weights"].float(), PET.aggregate.ORDER_CONTEXT_WEIGHT
            ),
            torch.ones_like(batch["sample_weights"].float()),
        )
        result = {}
        for name in PET.LOSS_NAMES:
            mask = masks_cpu[name].to(device=device, dtype=torch.bool)
            denominator = weights[mask].sum()
            scalar = (per_row[mask] * weights[mask]).sum() / denominator
            result[name] = float(scalar.detach().cpu())
    if set(result) != set(PET.LOSS_NAMES) or not all(
        math.isfinite(value) for value in result.values()
    ):
        raise RuntimeError("direct512 actual ordered-NLL evaluation drift")
    return result


def direct_loss_gate(
    baseline: Mapping[str, float], current: Mapping[str, float]
) -> dict[str, Any]:
    records = []
    for name in PET.LOSS_NAMES:
        delta = float(current[name]) - float(baseline[name])
        records.append(
            {
                "name": name,
                "beta100_loss": float(baseline[name]),
                "endpoint_loss": float(current[name]),
                "endpoint_minus_beta100": delta,
                "tolerance": DIRECT_LOSS_ABS_TOLERANCE,
                "pass": delta <= DIRECT_LOSS_ABS_TOLERANCE,
            }
        )
    return {
        "pass": all(item["pass"] for item in records),
        "count_exact_13": len(records) == 13,
        "maximum_endpoint_minus_beta100": max(
            item["endpoint_minus_beta100"] for item in records
        ),
        "records": records,
    }


def selected_head_exact_gate(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    records = []
    for batch_key in sorted(baseline):
        for head in ("count_logits", "value_logits"):
            exact = torch.equal(
                baseline[batch_key]["outputs_cpu"][head],
                current[batch_key]["outputs_cpu"][head],
            )
            records.append({"batch_key": batch_key, "head": head, "exact": exact})
    return {
        "pass": all(item["exact"] for item in records),
        "records": records,
        "selected_batch_count": len(baseline),
        "tensor_comparison_count": len(records),
    }


def terminal_fulltrain_policy_gate(
    transition: ModuleType,
    model: Any,
    device: Any,
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    batch_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    row_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    identity_digests = {panel: hashlib.sha256() for panel in transition.PANEL_ORDER}
    mismatch_rows = {"count_logits": 0, "value_logits": 0}
    cells = {
        panel: {
            metric: {cell: 0 for cell in ("cc", "cw", "wc", "ww")}
            for metric in FULLTRAIN_POLICY_METRICS
        }
        for panel in transition.PANEL_ORDER
    }
    selected_cells: dict[tuple[str, str, str], str] = {}
    output_key_checks: list[bool] = []
    native_dtype_checks: list[bool] = []
    candidate_state_sha_before = ppo.model_state_sha256(model)
    nonactor_names = sorted(set(model.state_dict()) - set(PET.ACTOR_NAMES))
    candidate_nonactor_sha_before = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )

    def callback(
        panel: str,
        cpu_batch: Mapping[str, Any],
        identities: Sequence[Mapping[str, Any]],
        raw_snapshot: Mapping[str, Any],
        beta_snapshot: Mapping[str, Any],
    ) -> None:
        del raw_snapshot
        batch_counts[panel] += 1
        row_counts[panel] += len(identities)
        for identity in identities:
            identity_digests[panel].update(
                transition.canonical_json(
                    {
                        "panel": panel,
                        "member": identity["member"],
                        "line_index_zero_based": int(identity["line_index_zero_based"]),
                        "line_sha256": identity["line_sha256"],
                    }
                )
            )
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in cpu_batch.items()
        }
        with torch.no_grad():
            outputs = ppo.model_forward(model, batch, device)
            actions, _, _, _ = ppo.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=False,
            )
        output_key_checks.append(
            set(outputs) == {"policy_logits", "count_logits", "value_logits"}
        )
        native_dtype_checks.append(
            all(value.dtype == torch.bfloat16 for value in outputs.values())
        )
        outputs_cpu = {
            key: value.detach().cpu().contiguous() for key, value in outputs.items()
        }
        for head in ("count_logits", "value_logits"):
            candidate = outputs_cpu[head].float()
            beta = beta_snapshot["outputs_cpu"][head]
            if candidate.shape != beta.shape:
                raise RuntimeError("terminal full-train head shape drift")
            unequal = (candidate != beta).reshape(candidate.shape[0], -1).any(dim=1)
            mismatch_rows[head] += int(unequal.sum())
        for row_index, identity in enumerate(identities):
            candidate_record = official_row_record(
                cpu_batch, outputs_cpu, actions, row_index
            )
            context = int(identity["context"])
            nonempty = int(cpu_batch["action_counts"][row_index]) > 0
            line_sha = str(identity["line_sha256"])
            for metric in FULLTRAIN_POLICY_METRICS:
                if metric == "top1_correct" and not nonempty:
                    continue
                if metric.startswith("context34_") and context != 34:
                    continue
                base_metric = metric.removeprefix("context34_")
                beta_correct = bool(
                    beta_snapshot["flags"][row_index][base_metric]
                )
                candidate_correct = bool(candidate_record["flags"][base_metric])
                cell = (
                    ("c" if beta_correct else "w")
                    + ("c" if candidate_correct else "w")
                )
                cells[panel][metric][cell] += 1
                selected_cells[(panel, line_sha, metric)] = cell

    report = transition.run_transition_audit(
        row_consumer=callback,
        candidate_consumer=None,
    )
    closure = transition_absolute_closure(report)
    expected_batches = {
        panel: math.ceil(int(transition.EXPECTED_TRAIN[panel]["rows"]) / 256)
        for panel in transition.PANEL_ORDER
    }
    identity_sha = {
        panel: identity_digests[panel].hexdigest() for panel in transition.PANEL_ORDER
    }
    coverage = (
        batch_counts == expected_batches
        and identity_sha == transition.EXPECTED_IDENTITY_SHA256
        and row_counts
        == {
            panel: int(transition.EXPECTED_TRAIN[panel]["rows"])
            for panel in transition.PANEL_ORDER
        }
    )
    policy_records = []
    for panel in transition.PANEL_ORDER:
        for metric in FULLTRAIN_POLICY_METRICS:
            counts = cells[panel][metric]
            denominator = sum(counts.values())
            beta_correct = counts["cc"] + counts["cw"]
            candidate_correct = counts["cc"] + counts["wc"]
            frozen_beta = int(
                transition.EXPECTED_AGGREGATES[panel]["beta100"][metric]
            )
            minimum_candidate = int(
                TERMINAL_MINIMUM_CANDIDATE_COUNTS.get(panel, {}).get(
                    metric, frozen_beta
                )
            )
            expected_denominator = int(
                TERMINAL_POLICY_DENOMINATORS[panel][metric]
            )
            record_pass = (
                denominator == expected_denominator
                and beta_correct == frozen_beta
                and candidate_correct >= frozen_beta
                and candidate_correct >= minimum_candidate
                and counts["cw"] == 0
                and counts["wc"] - counts["cw"]
                == candidate_correct - beta_correct
            )
            policy_records.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "cells": dict(counts),
                    "denominator": denominator,
                    "expected_denominator": expected_denominator,
                    "beta_cc_plus_cw": beta_correct,
                    "frozen_beta_correct": frozen_beta,
                    "candidate_cc_plus_wc": candidate_correct,
                    "minimum_candidate_correct": minimum_candidate,
                    "wc_minus_cw": counts["wc"] - counts["cw"],
                    "candidate_minus_beta": candidate_correct - beta_correct,
                    "cw_exact_zero": counts["cw"] == 0,
                    "pass": record_pass,
                }
            )
    exact_selected_records = []
    for descriptor in descriptors:
        expected_cell = (
            "wc" if descriptor["role"] == "target_transition" else "cc"
        )
        panel = str(descriptor["source"])
        line_sha = str(descriptor["identity"]["line_sha256"])
        for metric in descriptor["required_metrics"]:
            observed_cell = selected_cells.get((panel, line_sha, str(metric)))
            exact_selected_records.append(
                {
                    "identity": descriptor["identity"],
                    "role": descriptor["role"],
                    "metric": metric,
                    "expected_cell": expected_cell,
                    "observed_cell": observed_cell,
                    "pass": observed_cell == expected_cell,
                }
            )
    target_records = [
        item
        for item in exact_selected_records
        if item["role"] == "target_transition"
    ]
    guard_records = [
        item
        for item in exact_selected_records
        if item["role"] != "target_transition"
    ]
    candidate_state_sha_after = ppo.model_state_sha256(model)
    candidate_nonactor_sha_after = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )
    state_pass = (
        candidate_state_sha_after == candidate_state_sha_before
        and candidate_nonactor_sha_after == candidate_nonactor_sha_before
    )
    passed = (
        coverage
        and not any(mismatch_rows.values())
        and closure["pass"]
        and all(output_key_checks)
        and all(native_dtype_checks)
        and len(policy_records) == 18
        and all(item["pass"] for item in policy_records)
        and len(target_records) == 9
        and all(item["pass"] for item in target_records)
        and len(guard_records) == 388
        and all(item["pass"] for item in guard_records)
        and state_pass
    )
    return {
        "pass": passed,
        "candidate_consumer_used": False,
        "batch_counts": batch_counts,
        "batch_count_total": sum(batch_counts.values()),
        "batch_count_total_exact_95": sum(batch_counts.values()) == 95,
        "expected_batch_counts": expected_batches,
        "row_counts": row_counts,
        "row_count_total": sum(row_counts.values()),
        "row_count_total_exact_24050": sum(row_counts.values()) == 24050,
        "identity_stream_sha256": identity_sha,
        "expected_identity_stream_sha256": transition.EXPECTED_IDENTITY_SHA256,
        "row_consumer_full_coverage": coverage,
        "count_value_mismatch_rows": mismatch_rows,
        "count_logits_tensor_exact_beta100_full_train": mismatch_rows[
            "count_logits"
        ]
        == 0,
        "value_logits_tensor_exact_beta100_full_train": mismatch_rows[
            "value_logits"
        ]
        == 0,
        "candidate_output_keys_exact_every_batch": all(output_key_checks),
        "candidate_outputs_native_bf16_every_batch": all(native_dtype_checks),
        "policy_transition_records": policy_records,
        "policy_record_count_exact_18": len(policy_records) == 18,
        "all_policy_cw_exact_zero_and_aggregate_gates_pass": all(
            item["pass"] for item in policy_records
        ),
        "exact_target_wc_records": target_records,
        "exact_target_wc_obligation_count_exact_9": len(target_records) == 9,
        "exact_target_wc_all_pass": all(item["pass"] for item in target_records),
        "pf_gain_and_96_near_cc_records": guard_records,
        "pf_gain_and_96_near_cc_obligation_count_exact_388": (
            len(guard_records) == 388
        ),
        "pf_gain_and_96_near_cc_all_pass": all(
            item["pass"] for item in guard_records
        ),
        "candidate_model_state_sha256_before_fulltrain": (
            candidate_state_sha_before
        ),
        "candidate_model_state_sha256_after_fulltrain": candidate_state_sha_after,
        "candidate_full_state_unchanged_by_fulltrain_forward": (
            candidate_state_sha_after == candidate_state_sha_before
        ),
        "candidate_nonactor_state_sha256_before_fulltrain": (
            candidate_nonactor_sha_before
        ),
        "candidate_nonactor_state_sha256_after_fulltrain": (
            candidate_nonactor_sha_after
        ),
        "candidate_nonactor_state_unchanged_by_fulltrain_forward": (
            candidate_nonactor_sha_after == candidate_nonactor_sha_before
        ),
        "transition_absolute_closure": closure,
    }


def actor_delta_from_model(
    parameters: Mapping[str, Any], beta_actor: Mapping[str, Any]
) -> np.ndarray:
    pieces = [
        (
            parameters[name].detach().cpu().float()
            - beta_actor[name].detach().cpu().float()
        )
        .reshape(-1)
        .double()
        .numpy()
        for name in PET.ACTOR_NAMES
    ]
    return np.concatenate(pieces).astype(np.float64, copy=False)


def set_actor_delta(
    parameters: Mapping[str, Any],
    beta_actor: Mapping[str, Any],
    delta: np.ndarray,
) -> None:
    offset = 0
    with torch.no_grad():
        for name in PET.ACTOR_NAMES:
            parameter = parameters[name]
            count = int(parameter.numel())
            piece = torch.from_numpy(delta[offset : offset + count].copy()).to(
                device=parameter.device, dtype=parameter.dtype
            ).reshape(parameter.shape)
            parameter.copy_(beta_actor[name] + piece)
            offset += count
    if offset != PET.EXPECTED_ACTOR_ELEMENTS:
        raise RuntimeError("actor6 overlay vector slice drift")


def canonical_active_ledger(
    active: Mapping[tuple[str, int, int, int], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "key": list(key),
            "identity": active[key]["identity"],
            "row_role": active[key]["row_role"],
            "positive_option": active[key]["positive_option"],
            "negative_option": active[key]["negative_option"],
            "threshold": active[key]["threshold"],
            "origins": list(active[key]["origins"]),
            "created_round": active[key]["created_round"],
        }
        for key in sorted(active)
    ]


def run_probe(
    transition_context: Mapping[str, Any] | None = None,
    candidate_consumer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if transition_context is None:
        transition, transition_evidence = load_transition_module()
        transition_context = build_actual_transition_context(transition)
    else:
        transition_evidence = {"injected_context": True}
        transition = transition_context.get("_transition_module")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("actual probe requires CUDA with native BF16")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    context_audit = validate_transition_context(transition_context)
    model = transition_context["beta_model"]
    beta_checkpoint = transition_context["beta_checkpoint"]
    descriptors = transition_context["descriptors"]
    batches = transition_context["selected_batches"]
    if next(model.parameters()).device != device:
        raise RuntimeError("transition beta model is not on cuda:0")
    if ppo.model_state_sha256(model) != transition_context["beta_model_state_sha256"]:
        raise RuntimeError("beta100 runtime model SHA drift")

    basis = PET.build_basis_geometry()
    q_torch = basis["q"]
    q = np.ascontiguousarray(q_torch.numpy(), dtype=np.float64)
    cache = PET.build_cache_context(basis["payloads"]["R"])
    parameters = PET.configure_actor6(model)
    beta_actor = {
        name: parameters[name].detach().clone() for name in PET.ACTOR_NAMES
    }
    beta_state_sha = ppo.model_state_sha256(model)
    nonactor_names = sorted(set(model.state_dict()) - set(PET.ACTOR_NAMES))
    beta_nonactor_sha = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in nonactor_names}
    )

    baseline = snapshot_batches(model, batches, device)
    for descriptor in descriptors:
        beta_flags = descriptor["beta_flags"]
        observed = baseline[str(descriptor["batch_key"])]["official_rows"][
            int(descriptor["local_index"])
        ]["flags"]
        if any(bool(observed[metric]) != bool(beta_flags[metric]) for metric in METRICS):
            raise RuntimeError("transition/baseline official flag mismatch")

    active: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    initial_guard_updates = initialize_guard_pairs(descriptors, active, baseline)
    current = baseline
    gate = gate_selected(descriptors, active, current)
    beta_direct_losses = evaluate_direct_losses(
        model, cache["direct_union"], cache["loss_masks"], device
    )
    current_direct_gate = direct_loss_gate(beta_direct_losses, beta_direct_losses)
    current_head_gate = selected_head_exact_gate(baseline, current)
    combined_gate_pass = (
        gate["pass"] and current_direct_gate["pass"] and current_head_gate["pass"]
    )
    initial_target_update = add_dynamic_cuts(
        descriptors,
        gate["false_obligations"],
        active,
        baseline,
        current,
        0,
    )
    if not initial_target_update["pass"]:
        return {
            "schema_version": SCHEMA,
            "status": "closed_no_candidate",
            "close_reason": "initial_false_obligation_has_no_actor_pair",
            "transition_context": context_audit,
            "unresolved": initial_target_update["unresolved"],
            "integrity": {"overlay_started": False, "writes": 0},
        }
    gate = gate_selected(descriptors, active, current)

    cumulative = np.zeros(PET.EXPECTED_ACTOR_ELEMENTS, dtype=np.float64)
    radius = INITIAL_CUMULATIVE_L2
    expansions = 0
    endpoints = 0
    ledger: list[dict[str, Any]] = []
    success_round: int | None = None
    close_reason = "maximum_round_or_endpoint_budget_reached"
    overlay_started = False
    consumer_called = False
    terminal_actual = cumulative.copy()
    final_integrity: dict[str, Any] = {}
    terminal_fulltrain_head_gate: dict[str, Any] | None = None

    try:
        for round_index in range(1, MAX_ROUNDS + 1):
            if endpoints >= MAX_ENDPOINTS:
                close_reason = "maximum_endpoint_budget_reached"
                break
            dynamic = add_dynamic_cuts(
                descriptors,
                gate["false_obligations"],
                active,
                baseline,
                current,
                round_index,
            )
            if not dynamic["pass"]:
                close_reason = "false_official_obligation_has_no_separating_actor_pair"
                ledger.append(
                    {
                        "round": round_index,
                        "kind": "closed_without_endpoint",
                        "dynamic": dynamic,
                        "pre_gate": gate,
                    }
                )
                break
            pair_gradients, pair_rhs, pair_audit = collect_pair_gradients(
                model, parameters, batches, active, current, q, device
            )
            loss_gradients, loss_audit = collect_projected_loss_gradients(
                model,
                parameters,
                cache["direct_union"],
                cache["loss_masks"],
                q,
                device,
            )
            current_losses = {
                str(item["name"]): float(item["loss"])
                for item in loss_audit["records"]
            }
            if set(current_losses) != set(PET.LOSS_NAMES):
                raise RuntimeError("current gradient loss ledger drift")
            loss_rhs = np.asarray(
                [
                    current_losses[name]
                    - beta_direct_losses[name]
                    - DIRECT_LOSS_ABS_TOLERANCE
                    for name in PET.LOSS_NAMES
                ],
                dtype=np.float64,
            )
            correction, qp = solve_minimum_endpoint_l2(
                pair_gradients,
                pair_rhs,
                loss_gradients,
                loss_rhs,
                cumulative,
                radius,
            )
            if correction is None and expansions < MAX_RADIUS_EXPANSIONS:
                expansions += 1
                radius = EXPANDED_CUMULATIVE_L2
                correction, qp_expanded = solve_minimum_endpoint_l2(
                    pair_gradients,
                    pair_rhs,
                    loss_gradients,
                    loss_rhs,
                    cumulative,
                    radius,
                )
                qp = {"initial_radius_failure": qp, "expanded_radius": qp_expanded}
            if correction is None:
                close_reason = "projected_qp_infeasible_inside_final_radius"
                ledger.append(
                    {
                        "round": round_index,
                        "kind": "closed_without_endpoint",
                        "dynamic": dynamic,
                        "pair_gradients": pair_audit,
                        "loss_gradients": loss_audit,
                        "qp": qp,
                        "pre_gate": gate,
                    }
                )
                break
            proposed = project_orthogonal(cumulative + correction, q)
            proposed_audit = projection_audit(proposed, q)
            if (
                not proposed_audit["strict_orthogonal_within_tolerance"]
                or proposed_audit["l2"] > radius + RADIUS_ABS_TOLERANCE
            ):
                raise RuntimeError("planned endpoint PET/radius gate failed")
            overlay_started = True
            set_actor_delta(parameters, beta_actor, proposed)
            actual = actor_delta_from_model(parameters, beta_actor)
            actual_audit = projection_audit(actual, q)
            if (
                not actual_audit["strict_orthogonal_within_tolerance"]
                or actual_audit["l2"] > radius + RADIUS_ABS_TOLERANCE
            ):
                close_reason = "actual_fp32_endpoint_failed_pet_or_radius_gate"
                terminal_actual = actual.copy()
                ledger.append(
                    {
                        "round": round_index,
                        "kind": "rejected_fp32_endpoint",
                        "planned_projection": proposed_audit,
                        "actual_projection": actual_audit,
                        "qp": qp,
                    }
                )
                break
            cumulative = actual.copy()
            terminal_actual = actual.copy()
            endpoints += 1
            current = snapshot_batches(model, batches, device)
            gate = gate_selected(descriptors, active, current)
            endpoint_losses = evaluate_direct_losses(
                model, cache["direct_union"], cache["loss_masks"], device
            )
            current_direct_gate = direct_loss_gate(
                beta_direct_losses, endpoint_losses
            )
            current_head_gate = selected_head_exact_gate(baseline, current)
            combined_gate_pass = (
                gate["pass"]
                and current_direct_gate["pass"]
                and current_head_gate["pass"]
            )
            nonactor_sha = PET.frozen.model_state_sha256(
                {name: model.state_dict()[name] for name in nonactor_names}
            )
            if nonactor_sha != beta_nonactor_sha:
                raise RuntimeError("nonactor state changed during actor6 endpoint")
            ledger.append(
                {
                    "round": round_index,
                    "kind": "pet_orthogonal_cuttingplane_endpoint",
                    "radius": radius,
                    "radius_expansions_used": expansions,
                    "endpoint_index": endpoints,
                    "dynamic": dynamic,
                    "pair_gradients": pair_audit,
                    "loss_gradients": loss_audit,
                    "qp": qp,
                    "planned_projection": proposed_audit,
                    "actual_projection": actual_audit,
                    "actual_delta_float32_sha256": vector_sha256_float32(actual),
                    "post_gate": gate,
                    "actual_direct512_loss_gate": current_direct_gate,
                    "selected_count_value_tensor_exact_beta100": current_head_gate,
                    "combined_selected_and_direct_gate_pass": combined_gate_pass,
                }
            )
            if combined_gate_pass:
                success_round = round_index
                close_reason = (
                    "all_selected_train_obligations_active_pairs_and_13_actual_losses_pass"
                )
                break

        final_ledger = canonical_active_ledger(active)
        if success_round is not None:
            terminal_projection = projection_audit(terminal_actual, q)
            if (
                not terminal_projection["strict_orthogonal_within_tolerance"]
                or terminal_projection["l2"]
                > EXPANDED_CUMULATIVE_L2 + RADIUS_ABS_TOLERANCE
            ):
                success_round = None
                close_reason = "terminal_actual_fp32_pet_or_hard_radius_gate_failed"
        if success_round is not None:
            if not isinstance(transition, ModuleType):
                raise RuntimeError(
                    "passing endpoint requires hash-bound transition module for "
                    "terminal full-train head verification"
                )
            terminal_fulltrain_head_gate = terminal_fulltrain_policy_gate(
                transition, model, device, descriptors
            )
            if not terminal_fulltrain_head_gate["pass"]:
                success_round = None
                close_reason = "terminal_fulltrain_authoritative_policy_or_head_gate_failed"
        if success_round is not None and candidate_consumer is not None:
            consumer_called = True
            actual_delta_by_name = {}
            offset = 0
            for name in PET.ACTOR_NAMES:
                parameter = parameters[name]
                count = int(parameter.numel())
                actual_delta_by_name[name] = torch.from_numpy(
                    terminal_actual[offset : offset + count].astype(np.float32, copy=True)
                ).reshape(parameter.shape)
                offset += count
            candidate_consumer(
                {
                    "model": model,
                    "beta_checkpoint": beta_checkpoint,
                    "beta_model_state_sha256": beta_state_sha,
                    "beta_actor": beta_actor,
                    "actual_delta_float32_by_name_cpu": actual_delta_by_name,
                    "actual_delta_float32_sha256": vector_sha256_float32(terminal_actual),
                    "actual_delta_l2": float(np.linalg.norm(terminal_actual)),
                    "pet_projection": projection_audit(terminal_actual, q),
                    "identity_ledger": context_audit,
                    "active_pair_ledger": final_ledger,
                    "selected_row_gate": gate,
                    "actual_direct512_loss_gate": current_direct_gate,
                    "selected_count_value_tensor_exact_beta100": current_head_gate,
                    "terminal_fulltrain_authoritative_policy_and_head_gate": (
                        terminal_fulltrain_head_gate
                    ),
                    "success_round": success_round,
                }
            )
    finally:
        if overlay_started:
            set_actor_delta(
                parameters,
                beta_actor,
                np.zeros(PET.EXPECTED_ACTOR_ELEMENTS, dtype=np.float64),
            )
            restored_sha = ppo.model_state_sha256(model)
            restored_delta = actor_delta_from_model(parameters, beta_actor)
            final_integrity = {
                "finally_beta100_restore_executed": True,
                "beta100_model_state_sha256_before": beta_state_sha,
                "beta100_model_state_sha256_after": restored_sha,
                "beta100_model_state_restored_exact": restored_sha == beta_state_sha,
                "restored_actor6_delta_l2": float(np.linalg.norm(restored_delta)),
                "restored_actor6_tensor_exact": all(
                    torch.equal(parameters[name].detach(), beta_actor[name])
                    for name in PET.ACTOR_NAMES
                ),
                "nonactor_state_sha256": PET.frozen.model_state_sha256(
                    {name: model.state_dict()[name] for name in nonactor_names}
                ),
            }
            if not (
                final_integrity["beta100_model_state_restored_exact"]
                and final_integrity["restored_actor6_tensor_exact"]
                and final_integrity["nonactor_state_sha256"] == beta_nonactor_sha
            ):
                raise RuntimeError("finally beta100 restoration failed")

    status = (
        "selected_train_pet_orthogonal_endpoint_success"
        if success_round is not None
        else "closed_no_candidate"
    )
    final_ledger = canonical_active_ledger(active)
    return {
        "schema_version": SCHEMA,
        "status": status,
        "scope": {
            "start": "beta100",
            "train_only": True,
            "validation_opened": False,
            "target_sources": list(TARGET_SOURCES),
            "pokemonfan_gain_guards": True,
            "near_margin_basis": "beta100 selection_margin",
            "near_guard_per_source": NEAR_GUARD_PER_SOURCE,
            "direct512_loss_slopes": list(PET.LOSS_NAMES),
            "pet_orthogonal_complement": True,
            "max_rounds": MAX_ROUNDS,
            "max_endpoints": MAX_ENDPOINTS,
            "initial_cumulative_l2": INITIAL_CUMULATIVE_L2,
            "single_expanded_cumulative_l2": EXPANDED_CUMULATIVE_L2,
            "checkpoint_writes": 0,
            "stdout_only": True,
            "standalone_candidate_consumer_is_none": candidate_consumer is None,
        },
        "input_lock": {
            "self": {
                "path": str(SCRIPT.relative_to(ROOT)),
                "sha256": sha256_bytes(SCRIPT.read_bytes()),
            },
            "pet_geometry": PET_EVIDENCE,
            "transition": transition_evidence,
            "pet_inputs": PET.CHECKPOINT_EVIDENCE,
        },
        "transition_context": context_audit,
        "pet_geometry": basis["report"],
        "direct512_cache": cache["report"]["direct512"],
        "beta100_direct512_ordered_losses": beta_direct_losses,
        "initial_guard_pair_updates": initial_guard_updates,
        "initial_target_pair_updates": initial_target_update,
        "iterations": ledger,
        "decision": {
            "status": status,
            "close_reason": close_reason,
            "success_round": success_round,
            "endpoints_evaluated": endpoints,
            "radius_expansions_used": expansions,
            "terminal_actual_delta_l2": float(np.linalg.norm(terminal_actual)),
            "terminal_actual_delta_float32_sha256": vector_sha256_float32(
                terminal_actual
            ),
            "terminal_pet_projection": projection_audit(terminal_actual, q),
            "active_pair_count": len(final_ledger),
            "active_pair_ledger_sha256": sha256_bytes(
                canonical_json_bytes(final_ledger)
            ),
            "candidate_consumer_called_before_finally_restore": consumer_called,
            "model_materialized": False,
        },
        "final_selected_gate": gate,
        "final_actual_direct512_loss_gate": current_direct_gate,
        "final_selected_count_value_tensor_exact_beta100": current_head_gate,
        "terminal_fulltrain_authoritative_policy_and_head_gate": (
            terminal_fulltrain_head_gate
        ),
        "final_integrity": final_integrity,
    }


def cache_audit(transition: ModuleType) -> dict[str, Any]:
    basis = PET.build_basis_geometry()
    cache = PET.build_cache_context(basis["payloads"]["R"])
    transition_cache = transition.cache_audit()
    if transition_cache.get("status") != "cache_audit_passed":
        raise RuntimeError("transition cache audit status drift")
    if (
        transition_cache.get("scope", {}).get(
            "validation_member_payloads_opened"
        )
        is not False
    ):
        raise RuntimeError("transition cache opened validation")
    rows = sum(
        int(panel["rows"])
        for panel in transition_cache["cache"]["panels"].values()
    )
    if rows != 24050:
        raise RuntimeError(f"transition cache row-count drift: {rows}")
    return {
        "status": "cache_audit_passed_zero_write_train_only",
        "transition": transition_cache,
        "pet_geometry": basis["report"],
        "pet_cache": cache["report"],
        "validation_opened": False,
        "near_margin_basis": "beta100 selection_margin (actual-mode row_consumer)",
        "writes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("static-audit", "cache-audit", "actual"), required=True
    )
    args = parser.parse_args()
    validate_runtime()
    static = static_audit()
    transition, transition_evidence = load_transition_module()
    if args.mode == "static-audit":
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "static_audit_passed",
            "self": read_regular_bytes(SCRIPT, None, "self")[1],
            "pet_geometry": PET_EVIDENCE,
            "transition": transition_evidence,
            "audit": static,
        }
    elif args.mode == "cache-audit":
        result = {
            "schema_version": SCHEMA,
            "static": static,
            "transition_input": transition_evidence,
            **cache_audit(transition),
        }
    else:
        result = run_probe(
            transition_context=build_actual_transition_context(transition),
            candidate_consumer=None,
        )
        result["static_audit"] = static
        result["transition_input"] = transition_evidence
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
