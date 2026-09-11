#!/usr/bin/env python3
"""Train-only beta100 FinalNorm sequential cutting-plane probe.

Only ``transformer.norm.weight`` and ``transformer.norm.bias`` (256 FP32
scalars) may move.  The probe begins from the SHA-bound beta100 checkpoint,
uses the frozen complete-train transition stream and the frozen Direct512
cache, and keeps every candidate in RAM.  It has no optimizer, backward call,
serialization, result path, validation input, or network path.

The complete 24,050-row terminal gate is part of the cutting plane rather
than a terminal-only rejection.  A beta-correct policy regression, a changed
variable-cardinality prediction, or a changed value sign is converted into a
same-row differentiable margin cut and the solver continues.  A fixed-count
prediction change is an integrity error because the official sampler forces
that count independently of logits.

The command-line entry point always passes ``candidate_consumer=None`` and
prints exactly one JSON document.  Static and cache modes are CPU-only.  The
actual mode is implemented for a later separately authorized CUDA run; this
file's construction does not run or freeze it.
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
SCRIPT = TOOLS / "probe_u468_beta100_finalnorm_train_cuttingplane_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCHEMA = "ptcg-u468-beta100-finalnorm-train-cuttingplane-v1"

PET_TOOL = TOOLS / "probe_u468_actor6_pet_cone_geometry.py"
PET_TOOL_SHA256 = "4f233db368f0150ee68109ee7d974a5a647ce8a35c919a5217f1714dddd98b2f"
TRANSITION_TOOL = TOOLS / "probe_u468_beta100_train_transitions_v1.py"
TRANSITION_TOOL_SHA256 = "5f25ffbd54705b282edf4512195bc83fe783ef300bc42d9361b48c627d0bffbb"

SEED = 202608127
NORM_NAMES = ("transformer.norm.weight", "transformer.norm.bias")
NORM_ELEMENTS = 256
EXPECTED_BETA_NORM2_SHA256 = (
    "0d40adbaca88fec93dfc5ec297069dfb9cf172de623253a30e4d75fa5511566f"
)
EXPECTED_BETA_NONNORM78_SHA256 = (
    "303244ce7e1d4bdec6e0c4b097a85402f78f5f75109b7374415ac1df79341058"
)

METRICS = ("set_exact", "hybrid_order_exact", "ordered_exact", "top1_correct")
FULLTRAIN_POLICY_METRICS = METRICS + (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
FULLTRAIN_ALL_METRICS = FULLTRAIN_POLICY_METRICS + (
    "count_correct",
    "value_correct",
)
TARGET_SOURCES = ("flg", "core5")
PF_SOURCE = "pokemonfan"
NEAR_GUARD_PER_SOURCE = 32
EXPECTED_INITIAL_CONSTRAINT_ROWS = 158
MAX_ENDPOINTS = 6
DIRECT_LOSS_ABS_TOLERANCE = 1.0e-7
QP_FTOL = 1.0e-12
QP_MAXITER = 5000
QP_LINEAR_TOLERANCE = 2.0e-8
RADIUS_ABS_TOLERANCE = 1.0e-12
MAXABS_ABS_TOLERANCE = 1.0e-12
PAIR_TOLERANCE = 0.0
DUAL_NNLS_MAXITER = 100_000
DUAL_ACTIVE_LAMBDA_TOLERANCE = 1.0e-10
DUAL_PRIMAL_RESIDUAL_TOLERANCE = 1.0e-9
DUAL_GRADIENT_TOLERANCE = 1.0e-9
DUAL_COMPLEMENTARITY_TOLERANCE = 1.0e-9
DUALITY_GAP_TOLERANCE = 1.0e-9
DUAL_MIN_GRAM_EIGENVALUE = 1.0e-14
LADDER = (
    {"radius": 5.0e-3, "maxabs": 1.0e-3},
    {"radius": 1.0e-2, "maxabs": 2.0e-3},
)

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
TERMINAL_DENOMINATORS = {
    "flg": {
        "set_exact": 9443,
        "hybrid_order_exact": 9443,
        "ordered_exact": 9443,
        "top1_correct": 9426,
        "context34_hybrid_order_exact": 42,
        "context34_ordered_exact": 42,
        "count_correct": 9443,
        "value_correct": 9443,
    },
    "pokemonfan": {
        "set_exact": 9487,
        "hybrid_order_exact": 9487,
        "ordered_exact": 9487,
        "top1_correct": 9450,
        "context34_hybrid_order_exact": 38,
        "context34_ordered_exact": 38,
        "count_correct": 9487,
        "value_correct": 9487,
    },
    "core5": {
        "set_exact": 5120,
        "hybrid_order_exact": 5120,
        "ordered_exact": 5120,
        "top1_correct": 5106,
        "context34_hybrid_order_exact": 20,
        "context34_ordered_exact": 20,
        "count_correct": 5120,
        "value_correct": 5120,
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


def load_transition_module() -> tuple[ModuleType, dict[str, Any]]:
    module, evidence = import_locked(
        TRANSITION_TOOL, TRANSITION_TOOL_SHA256, "beta100_train_transitions"
    )
    for name in ("cache_audit", "run_transition_audit"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"transition module API missing: {name}")
    return module, evidence


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def static_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))
    forbidden_calls = {
        "backward", "step", "save", "savez", "write", "write_text",
        "write_bytes", "touch", "mkdir", "makedirs", "unlink", "remove",
        "rmtree", "rename", "replace",
    }
    forbidden_import_roots = {
        "requests", "urllib", "http", "socket", "subprocess", "kaggle",
    }
    calls: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    write_flags: list[dict[str, Any]] = []
    copy_sites: list[int] = []
    print_sites: list[int] = []
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
            signature.parameters["candidate_consumer"].default is None
        ),
        "norm_scope_exact_2_tensors_256_fp32": (
            NORM_NAMES == ("transformer.norm.weight", "transformer.norm.bias")
            and NORM_ELEMENTS == 256
        ),
        "radius_maxabs_ladder_exact": LADDER == (
            {"radius": 5.0e-3, "maxabs": 1.0e-3},
            {"radius": 1.0e-2, "maxabs": 2.0e-3},
        ),
        "max_endpoints_exact_6": MAX_ENDPOINTS == 6,
        "direct_loss_tolerance_exact": DIRECT_LOSS_ABS_TOLERANCE == 1.0e-7,
        "exact_3_targets_1_pf_gain_96_near": (
            len(EXPECTED_TARGETS) == 3
            and len(EXPECTED_PF_GAIN_GUARDS) == 1
            and 3 * NEAR_GUARD_PER_SOURCE == 96
        ),
        "initial_constraint_rows_exact_158": EXPECTED_INITIAL_CONSTRAINT_ROWS == 158,
        "terminal_policy6_count_value2": (
            len(FULLTRAIN_POLICY_METRICS) == 6
            and len(FULLTRAIN_ALL_METRICS) == 8
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"static audit failed: {checks}")
    return {
        "status": "static_zero_write_finalnorm_audit_passed",
        "checks": checks,
        "forbidden_call_hits": calls,
        "forbidden_import_hits": imports,
        "forbidden_os_write_flags": write_flags,
        "ram_copy_sites": copy_sites,
        "stdout_print_sites": print_sites,
    }


def transition_absolute_closure(report: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for panel in ("flg", "pokemonfan", "core5"):
        for metric in FULLTRAIN_ALL_METRICS:
            transition = report["panels"][panel]["transitions"][metric]
            cells = transition["cells"]
            raw_from_cells = int(cells["cc"]["count"]) + int(cells["cw"]["count"])
            beta_from_cells = int(cells["cc"]["count"]) + int(cells["wc"]["count"])
            passed = (
                raw_from_cells == int(transition["raw_correct"])
                and beta_from_cells == int(transition["beta100_correct"])
            )
            records.append({
                "panel": panel,
                "metric": metric,
                "raw_cc_plus_cw": raw_from_cells,
                "beta_cc_plus_wc": beta_from_cells,
                "pass": passed,
            })
    if len(records) != 24 or not all(item["pass"] for item in records):
        raise RuntimeError("transition absolute 3x8 closure failed")
    return {"pass": True, "check_count_exact_24": True, "checks": records}


def ordered_selection_margin(
    logits: Any, option_mask: Any, expert_order: Sequence[int]
) -> float:
    remaining = option_mask.bool().clone()
    margins: list[float] = []
    for chosen_source in expert_order:
        chosen = int(chosen_source)
        if not (0 <= chosen < int(remaining.shape[0])) or not bool(remaining[chosen]):
            raise RuntimeError("illegal expert action in transition callback")
        competitors = remaining.nonzero(as_tuple=False).squeeze(1)
        competitors = competitors[competitors != chosen]
        if competitors.numel() > 0:
            margins.append(float(logits[chosen] - logits[competitors].max()))
        remaining[chosen] = False
    return min(margins, default=float("inf"))


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
            identity_digests[panel].update(transition.canonical_json({
                "panel": panel,
                "member": identity["member"],
                "line_index_zero_based": int(identity["line_index_zero_based"]),
                "line_sha256": identity["line_sha256"],
            }))
            raw_flags = {metric: bool(raw_snapshot["flags"][local_index][metric]) for metric in METRICS}
            beta_flags = {metric: bool(beta_snapshot["flags"][local_index][metric]) for metric in METRICS}
            line_sha = str(identity["line_sha256"])
            if panel in TARGET_SOURCES:
                changed = [metric for metric in METRICS if raw_flags[metric] and not beta_flags[metric]]
                role = "target_transition"
            elif panel == PF_SOURCE:
                changed = [metric for metric in METRICS if not raw_flags[metric] and beta_flags[metric]]
                role = "pf_gain_guard"
            else:
                raise RuntimeError(f"unexpected panel: {panel}")
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
                near_pool[panel].append({
                    "identity": identity,
                    "source": panel,
                    "role": "near_guard",
                    "required_metrics": METRICS,
                    "batch_key": batch_key,
                    "local_index": local_index,
                    "raw_flags": raw_flags,
                    "beta_flags": beta_flags,
                    "beta_selection_margin": ordered_selection_margin(
                        beta_snapshot["outputs_cpu"]["policy_logits"][local_index],
                        cpu_batch["option_mask"][local_index],
                        expert,
                    ),
                    "cpu_batch": cpu_batch,
                })

    report = transition.run_transition_audit(
        row_consumer=callback, candidate_consumer=None
    )
    if (
        report.get("status") != "complete_train_transition_audit_passed"
        or report.get("scope", {}).get("validation_member_payloads_opened") is not False
    ):
        raise RuntimeError("strict train-only transition audit did not pass")
    closure = transition_absolute_closure(report)
    expected_batches = {
        panel: math.ceil(int(transition.EXPECTED_TRAIN[panel]["rows"]) / 256)
        for panel in transition.PANEL_ORDER
    }
    identity_sha = {
        panel: identity_digests[panel].hexdigest() for panel in transition.PANEL_ORDER
    }
    if batch_counts != expected_batches or identity_sha != transition.EXPECTED_IDENTITY_SHA256:
        raise RuntimeError("transition row_consumer coverage drift")

    descriptors = []
    for item in primary.values():
        item["required_metrics"] = tuple(
            metric for metric in METRICS if metric in item["required_metrics"]
        )
        descriptors.append(item)
    observed_targets = {
        item["identity"]["line_sha256"]: {
            "source": item["source"], "metrics": tuple(item["required_metrics"])
        }
        for item in descriptors if item["role"] == "target_transition"
    }
    observed_pf = {
        item["identity"]["line_sha256"]: {
            "source": item["source"], "metrics": tuple(item["required_metrics"])
        }
        for item in descriptors if item["role"] == "pf_gain_guard"
    }
    if observed_targets != EXPECTED_TARGETS or observed_pf != EXPECTED_PF_GAIN_GUARDS:
        raise RuntimeError("exact target/PokemonFan identity ledger drift")
    primary_hashes = set(observed_targets) | set(observed_pf)
    near_ledger: dict[str, list[dict[str, Any]]] = {}
    for panel in transition.PANEL_ORDER:
        candidates = sorted(
            (
                item for item in near_pool[panel]
                if item["identity"]["line_sha256"] not in primary_hashes
            ),
            key=lambda item: (
                float(item["beta_selection_margin"]),
                str(item["identity"]["line_sha256"]),
            ),
        )[:NEAR_GUARD_PER_SOURCE]
        if len(candidates) != NEAR_GUARD_PER_SOURCE:
            raise RuntimeError(f"{panel}: insufficient beta-near train guards")
        descriptors.extend(candidates)
        near_ledger[panel] = [
            {"identity": item["identity"], "beta_selection_margin": float(item["beta_selection_margin"])}
            for item in candidates
        ]
    descriptors.sort(key=lambda item: (
        {"target_transition": 0, "pf_gain_guard": 1, "near_guard": 2}[item["role"]],
        str(item["source"]),
        str(item["identity"]["line_sha256"]),
    ))
    selected_batches: dict[str, dict[str, Any]] = {}
    for item in descriptors:
        cpu_batch = item.pop("cpu_batch")
        selected_batches.setdefault(
            str(item["batch_key"]),
            {"key": str(item["batch_key"]), "cpu_batch": cpu_batch},
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
        "absolute_closure": closure,
        "row_consumer_coverage": {
            "batch_counts": batch_counts,
            "expected_batch_counts": expected_batches,
            "identity_stream_sha256": identity_sha,
            "expected_identity_stream_sha256": transition.EXPECTED_IDENTITY_SHA256,
            "pass": True,
        },
        "near_selection_ledger": near_ledger,
        "_transition_module": transition,
    }


def validate_transition_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if context.get("split") != "train" or context.get("validation_opened") is not False:
        raise RuntimeError("transition context is not strict train-only")
    if context.get("batch_size") != 256:
        raise RuntimeError("transition context is not official B256")
    descriptors = context.get("descriptors")
    batches = context.get("selected_batches")
    if not isinstance(descriptors, list) or not isinstance(batches, list):
        raise RuntimeError("transition descriptor/batch contract drift")
    batch_by_key = {str(item["key"]): item for item in batches}
    if len(batch_by_key) != len(batches):
        raise RuntimeError("canonical selected batch keys repeat")
    role_counts: dict[str, int] = {}
    row_identities: list[tuple[str, str, int, str]] = []
    for descriptor in descriptors:
        role = str(descriptor["role"])
        source = str(descriptor["source"])
        metrics = tuple(str(value) for value in descriptor["required_metrics"])
        key = str(descriptor["batch_key"])
        local = int(descriptor["local_index"])
        if role not in {"target_transition", "pf_gain_guard", "near_guard"}:
            raise RuntimeError(f"unexpected descriptor role: {role}")
        if not metrics or any(metric not in METRICS for metric in metrics):
            raise RuntimeError("descriptor policy metric drift")
        if role == "target_transition" and source not in TARGET_SOURCES:
            raise RuntimeError("target source drift")
        if role == "pf_gain_guard" and source != PF_SOURCE:
            raise RuntimeError("PokemonFan guard source drift")
        if key not in batch_by_key:
            raise RuntimeError("descriptor canonical batch absent")
        cpu_batch = batch_by_key[key]["cpu_batch"]
        if not (0 <= local < int(cpu_batch["action_counts"].shape[0])):
            raise RuntimeError("descriptor local index outside batch")
        descriptor["cpu_batch"] = cpu_batch
        identity = descriptor["identity"]
        member = str(identity.get("member", ""))
        if not member.startswith("train/") or "valid" in member.lower():
            raise RuntimeError("descriptor is not train-backed")
        row_identities.append((
            source,
            member,
            int(identity["line_index_zero_based"]),
            str(identity["line_sha256"]),
        ))
        role_counts[role] = role_counts.get(role, 0) + 1
    expected_roles = {
        "target_transition": 3,
        "pf_gain_guard": 1,
        "near_guard": 3 * NEAR_GUARD_PER_SOURCE,
    }
    if role_counts != expected_roles or len(row_identities) != len(set(row_identities)):
        raise RuntimeError("descriptor count, uniqueness, or role-disjointness drift")
    targets = {
        item["identity"]["line_sha256"]: {
            "source": item["source"], "metrics": tuple(item["required_metrics"])
        }
        for item in descriptors if item["role"] == "target_transition"
    }
    pf = {
        item["identity"]["line_sha256"]: {
            "source": item["source"], "metrics": tuple(item["required_metrics"])
        }
        for item in descriptors if item["role"] == "pf_gain_guard"
    }
    near_counts = {
        source: sum(
            item["role"] == "near_guard" and item["source"] == source
            for item in descriptors
        )
        for source in ("flg", "pokemonfan", "core5")
    }
    if targets != EXPECTED_TARGETS or pf != EXPECTED_PF_GAIN_GUARDS:
        raise RuntimeError("primary descriptor ledger drift")
    if any(value != NEAR_GUARD_PER_SOURCE for value in near_counts.values()):
        raise RuntimeError(f"near guard count drift: {near_counts}")
    selection = [
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
    return {
        "descriptor_count_exact_100": len(descriptors) == 100,
        "canonical_batch_count": len(batches),
        "role_counts": role_counts,
        "near_guard_counts": near_counts,
        "identities_unique_and_roles_disjoint": True,
        "transition_selection_sha256": sha256_bytes(canonical_json_bytes(selection)),
        "row_consumer_coverage": context.get("row_consumer_coverage"),
        "transition_absolute_closure": context.get("absolute_closure"),
    }


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        value.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


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
        for value in cpu_batch["expert_ordered_actions"][
            row_index, :expert_count
        ]
    ]
    context = int(cpu_batch["contexts"][row_index])
    hybrid = order if context == ppo.SKILL_ORDER_CONTEXT else sorted(order)
    top1 = int(outputs_cpu["policy_logits"][row_index].argmax())
    value_sign = bool(outputs_cpu["value_logits"][row_index] >= 0)
    return {
        "flags": {
            "set_exact": bool(((prediction == targets) | ~option_mask).all()),
            "hybrid_order_exact": hybrid == expert_order,
            "ordered_exact": order == expert_order,
            "top1_correct": bool(targets[top1]),
            "count_correct": len(order) == int(cpu_batch["action_counts"][row_index]),
            "value_correct": value_sign == bool(cpu_batch["win_targets"][row_index]),
        },
        "predicted_order": order,
        "expert_order": expert_order,
        "context": context,
        "top1_index": top1,
        "predicted_count": len(order),
        "value_sign": value_sign,
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
            batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
            outputs = ppo.model_forward(model, batch, device)
            if set(outputs) != {"policy_logits", "count_logits", "value_logits"}:
                raise RuntimeError(f"{key}: output-key drift")
            if any(value.dtype != torch.bfloat16 for value in outputs.values()):
                raise RuntimeError(f"{key}: selected output is not native BF16")
            actions, _, _, _ = ppo.sample_ordered_actions(
                outputs, batch, deterministic=True, canonicalize_order=False
            )
            outputs_cpu = {
                name: value.detach().float().cpu().contiguous()
                for name, value in outputs.items()
            }
            result[key] = {
                "outputs_cpu": outputs_cpu,
                "official_rows": [
                    official_row_record(cpu_batch, outputs_cpu, actions, row)
                    for row in range(len(actions))
                ],
                "fingerprints": {
                    name: tensor_sha256(value) for name, value in outputs_cpu.items()
                },
            }
    return result


def first_different_pair(
    desired_order: Sequence[int], predicted_order: Sequence[int]
) -> tuple[int, int] | None:
    for desired, observed in zip(desired_order, predicted_order):
        if int(desired) != int(observed):
            return int(desired), int(observed)
    return None


def set_pair(
    desired_order: Sequence[int], predicted_order: Sequence[int]
) -> tuple[int, int] | None:
    desired_set = {int(value) for value in desired_order}
    predicted_set = {int(value) for value in predicted_order}
    missing = [int(value) for value in desired_order if int(value) not in predicted_set]
    included = [int(value) for value in predicted_order if int(value) not in desired_set]
    return (missing[0], included[0]) if missing and included else None


def top1_pair(
    cpu_batch: Mapping[str, Any],
    outputs_cpu: Mapping[str, Any],
    row: int,
) -> tuple[int, int] | None:
    logits = outputs_cpu["policy_logits"][row]
    targets = cpu_batch["targets"][row].bool()
    positive_indices = targets.nonzero(as_tuple=False).squeeze(1)
    negative_indices = (~targets).nonzero(as_tuple=False).squeeze(1)
    if positive_indices.numel() == 0 or negative_indices.numel() == 0:
        return None
    positive = int(positive_indices[torch.argmax(logits[positive_indices])])
    negative = int(negative_indices[torch.argmax(logits[negative_indices])])
    return (positive, negative) if positive != negative else None


def derive_policy_pair(
    cpu_batch: Mapping[str, Any],
    current_outputs: Mapping[str, Any],
    current_record: Mapping[str, Any],
    desired_order: Sequence[int],
    metric_source: str,
    row: int,
) -> tuple[int, int] | None:
    metric = metric_source.removeprefix("context34_")
    if metric == "top1_correct":
        return top1_pair(cpu_batch, current_outputs, row)
    if metric == "set_exact":
        return set_pair(desired_order, current_record["predicted_order"])
    if metric == "hybrid_order_exact" and int(current_record["context"]) != 34:
        return set_pair(desired_order, current_record["predicted_order"])
    if metric in {"hybrid_order_exact", "ordered_exact"}:
        return first_different_pair(desired_order, current_record["predicted_order"])
    raise RuntimeError(f"unsupported policy metric: {metric_source}")


def local_positive_bf16_q(value_a: Any, value_b: Any) -> float:
    spacings = []
    for value in (value_a, value_b):
        native = value.to(dtype=torch.bfloat16)
        spacings.extend((
            torch.nextafter(native, torch.full_like(native, float("inf"))) - native,
            native - torch.nextafter(native, torch.full_like(native, float("-inf"))),
        ))
    result = max(float(item.float()) for item in spacings)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("invalid native-BF16 local positive quantum")
    return result


def zero_positive_bf16_q() -> float:
    zero = torch.zeros((), dtype=torch.bfloat16)
    result = float(
        torch.nextafter(zero, torch.full_like(zero, float("inf"))).float()
    )
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("invalid native-BF16 positive quantum around zero")
    return result


def constraint_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = str(item["kind"])
    base: tuple[Any, ...] = (
        str(item["batch_key"]), int(item["local_index"]), kind
    )
    if kind in {"policy", "count"}:
        return base + (int(item["positive_index"]), int(item["negative_index"]))
    if kind == "value":
        return base + (int(item["sign_multiplier"]),)
    raise RuntimeError(f"unknown constraint kind: {kind}")


def add_or_strengthen_constraint(
    active: dict[tuple[Any, ...], dict[str, Any]],
    item_source: Mapping[str, Any],
) -> dict[str, Any]:
    item = dict(item_source)
    item["origins"] = list(item.get("origins", ()))
    key = constraint_key(item)
    if key not in active:
        active[key] = item
        return {"kind": "added", "key": list(key), "threshold": float(item["threshold"])}
    existing = active[key]
    previous = float(existing["threshold"])
    existing["threshold"] = max(previous, float(item["threshold"]))
    for origin in item["origins"]:
        if origin not in existing["origins"]:
            existing["origins"].append(origin)
    return {
        "kind": "strengthened" if float(existing["threshold"]) > previous else "deduplicated",
        "key": list(key),
        "previous": previous,
        "threshold": float(existing["threshold"]),
    }


def constraint_margin_from_outputs(
    outputs: Mapping[str, Any], item: Mapping[str, Any]
) -> Any:
    row = int(item["local_index"])
    kind = str(item["kind"])
    if kind == "policy":
        logits = outputs["policy_logits"]
        return logits[row, int(item["positive_index"])].float() - logits[
            row, int(item["negative_index"])
        ].float()
    if kind == "count":
        logits = outputs["count_logits"]
        return logits[row, int(item["positive_index"])].float() - logits[
            row, int(item["negative_index"])
        ].float()
    if kind == "value":
        return outputs["value_logits"][row].float() * int(item["sign_multiplier"])
    raise RuntimeError(f"unknown constraint kind: {kind}")


def configure_finalnorm(model: Any) -> dict[str, Any]:
    model.requires_grad_(False)
    named = dict(model.named_parameters())
    parameters: dict[str, Any] = {}
    for name in NORM_NAMES:
        parameter = named.get(name)
        if parameter is None or parameter.dtype != torch.float32:
            raise RuntimeError(f"FinalNorm FP32 parameter drift: {name}")
        parameter.requires_grad_(True)
        parameters[name] = parameter
    active_names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if active_names != NORM_NAMES:
        raise RuntimeError(f"FinalNorm trainable scope drift: {active_names}")
    if sum(int(value.numel()) for value in parameters.values()) != NORM_ELEMENTS:
        raise RuntimeError("FinalNorm element-count drift")
    return parameters


def flat_norm_gradient(
    scalar: Any,
    parameters: Mapping[str, Any],
    *,
    retain_graph: bool,
) -> np.ndarray:
    values = torch.autograd.grad(
        scalar,
        tuple(parameters[name] for name in NORM_NAMES),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
        materialize_grads=False,
    )
    pieces = []
    for name, value in zip(NORM_NAMES, values):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite FinalNorm gradient: {name}")
        pieces.append(value.detach().reshape(-1).cpu().double().numpy())
    flat = np.concatenate(pieces).astype(np.float64, copy=False)
    if flat.shape != (NORM_ELEMENTS,) or not bool(np.all(np.isfinite(flat))):
        raise RuntimeError("invalid FinalNorm flattened gradient")
    return flat


def norm_delta_from_model(
    parameters: Mapping[str, Any], beta_norm: Mapping[str, Any]
) -> np.ndarray:
    return np.concatenate([
        (parameters[name].detach().cpu().float() - beta_norm[name].detach().cpu().float())
        .reshape(-1).double().numpy()
        for name in NORM_NAMES
    ]).astype(np.float64, copy=False)


def set_norm_delta(
    parameters: Mapping[str, Any],
    beta_norm: Mapping[str, Any],
    delta: np.ndarray,
) -> None:
    offset = 0
    with torch.no_grad():
        for name in NORM_NAMES:
            parameter = parameters[name]
            count = int(parameter.numel())
            piece = torch.from_numpy(delta[offset:offset + count].astype(np.float32, copy=True)).to(
                device=parameter.device, dtype=parameter.dtype
            ).reshape(parameter.shape)
            parameter.copy_(beta_norm[name] + piece)
            offset += count
    if offset != NORM_ELEMENTS:
        raise RuntimeError("FinalNorm overlay slice drift")


def selected_gate(
    descriptors: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    false_obligations: list[dict[str, Any]] = []
    records = []
    for descriptor_index, descriptor in enumerate(descriptors):
        record = snapshot[str(descriptor["batch_key"])]["official_rows"][
            int(descriptor["local_index"])
        ]
        flags = {
            metric: bool(record["flags"][metric])
            for metric in descriptor["required_metrics"]
        }
        for metric, passed in flags.items():
            if not passed:
                false_obligations.append({
                    "descriptor_index": descriptor_index,
                    "metric": metric,
                })
        records.append({
            "identity": descriptor["identity"],
            "role": descriptor["role"],
            "source": descriptor["source"],
            "required_metrics": list(descriptor["required_metrics"]),
            "flags": flags,
            "pass": all(flags.values()),
        })
    return {
        "pass": not false_obligations,
        "false_obligations": false_obligations,
        "records": records,
    }


def policy_margin(
    outputs_cpu: Mapping[str, Any], row: int, positive: int, negative: int
) -> float:
    logits = outputs_cpu["policy_logits"]
    return float(logits[row, positive] - logits[row, negative])


def guard_support_pairs(
    descriptor: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[tuple[int, int, str]]:
    batch_key = str(descriptor["batch_key"])
    row = int(descriptor["local_index"])
    snapshot = baseline[batch_key]
    outputs = snapshot["outputs_cpu"]
    record = snapshot["official_rows"][row]
    cpu_batch = descriptor["cpu_batch"]
    option_mask = cpu_batch["option_mask"][row].bool().clone()
    result: list[tuple[int, int, str]] = []
    if "top1_correct" in descriptor["required_metrics"]:
        pair = top1_pair(cpu_batch, outputs, row)
        if pair is not None:
            result.append((pair[0], pair[1], "baseline_top1_support"))
    if any(
        metric in descriptor["required_metrics"]
        for metric in ("set_exact", "hybrid_order_exact", "ordered_exact")
    ):
        logits = outputs["policy_logits"][row]
        for stage, chosen_source in enumerate(record["expert_order"]):
            chosen = int(chosen_source)
            competitors = option_mask.nonzero(as_tuple=False).squeeze(1)
            competitors = competitors[competitors != chosen]
            if competitors.numel() > 0:
                values = logits[competitors]
                best = values.max()
                for competitor in competitors[values == best].tolist():
                    result.append((chosen, int(competitor), f"baseline_stage_{stage}_support"))
            option_mask[chosen] = False
    return result


def initialize_selected_constraints(
    descriptors: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    active: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    updates = []
    unresolved = []
    for descriptor in descriptors:
        if descriptor["role"] == "target_transition":
            continue
        row = int(descriptor["local_index"])
        outputs = baseline[str(descriptor["batch_key"])]["outputs_cpu"]
        for positive, negative, origin in guard_support_pairs(descriptor, baseline):
            beta_margin = policy_margin(outputs, row, positive, negative)
            if beta_margin < 0.0:
                raise RuntimeError("beta100 selected guard margin is negative")
            threshold = local_positive_bf16_q(
                outputs["policy_logits"][row, positive],
                outputs["policy_logits"][row, negative],
            )
            updates.append(add_or_strengthen_constraint(active, {
                "kind": "policy",
                "batch_key": descriptor["batch_key"],
                "local_index": row,
                "positive_index": positive,
                "negative_index": negative,
                "threshold": threshold,
                "identity": descriptor["identity"],
                "row_role": descriptor["role"],
                "origins": [f"initial_guard:{origin}:positive_bf16_q"],
            }))
    gate = selected_gate(descriptors, baseline)
    for obligation in gate["false_obligations"]:
        descriptor = descriptors[int(obligation["descriptor_index"])]
        if descriptor["role"] != "target_transition":
            unresolved.append({
                "identity": descriptor["identity"],
                "metric": obligation["metric"],
                "reason": "beta100_guard_not_correct_at_baseline",
            })
            continue
        key = str(descriptor["batch_key"])
        row = int(descriptor["local_index"])
        current = baseline[key]
        pair = derive_policy_pair(
            descriptor["cpu_batch"],
            current["outputs_cpu"],
            current["official_rows"][row],
            current["official_rows"][row]["expert_order"],
            str(obligation["metric"]),
            row,
        )
        if pair is None:
            unresolved.append({
                "identity": descriptor["identity"],
                "metric": obligation["metric"],
                "reason": "target_has_no_policy_separating_pair",
            })
            continue
        logits = current["outputs_cpu"]["policy_logits"]
        threshold = local_positive_bf16_q(logits[row, pair[0]], logits[row, pair[1]])
        updates.append(add_or_strengthen_constraint(active, {
            "kind": "policy",
            "batch_key": key,
            "local_index": row,
            "positive_index": pair[0],
            "negative_index": pair[1],
            "threshold": threshold,
            "identity": descriptor["identity"],
            "row_role": descriptor["role"],
            "origins": [f"initial_target:{obligation['metric']}:positive_bf16_q"],
        }))
    return {
        "pass": not unresolved,
        "updates": updates,
        "unresolved": unresolved,
        "active_constraint_count": len(active),
        "baseline_selected_gate": gate,
    }


def collect_constraint_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    batch_registry: Mapping[str, Mapping[str, Any]],
    active: Mapping[tuple[Any, ...], Mapping[str, Any]],
    device: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    by_batch: dict[str, list[Mapping[str, Any]]] = {}
    for key in sorted(active):
        item = active[key]
        by_batch.setdefault(str(item["batch_key"]), []).append(item)
    gradients: list[np.ndarray] = []
    rhs: list[float] = []
    records = []
    for batch_key in sorted(by_batch):
        if batch_key not in batch_registry:
            raise RuntimeError(f"active constraint batch absent: {batch_key}")
        cpu_batch = batch_registry[batch_key]["cpu_batch"]
        batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
        outputs = ppo.model_forward(model, batch, device)
        constraints = by_batch[batch_key]
        scalars = [constraint_margin_from_outputs(outputs, item) for item in constraints]
        for index, (item, scalar) in enumerate(zip(constraints, scalars)):
            gradient = flat_norm_gradient(
                scalar,
                parameters,
                retain_graph=index + 1 < len(scalars),
            )
            observed = float(scalar.detach().cpu())
            gradients.append(gradient)
            rhs.append(float(item["threshold"]) - observed)
            records.append({
                "key": list(constraint_key(item)),
                "kind": item["kind"],
                "gradient_l2": float(np.linalg.norm(gradient)),
                "observed_margin": observed,
                "threshold": float(item["threshold"]),
                "correction_rhs": float(item["threshold"]) - observed,
            })
    if not gradients or any(float(np.linalg.norm(row)) <= 0.0 for row in gradients):
        raise RuntimeError("empty or zero FinalNorm constraint gradient")
    return (
        np.stack(gradients, axis=0),
        np.asarray(rhs, dtype=np.float64),
        {
            "count": len(gradients),
            "rhs_min": float(min(rhs)),
            "rhs_max": float(max(rhs)),
            "records": records,
        },
    )


def collect_direct_loss_gradients(
    model: Any,
    parameters: Mapping[str, Any],
    direct_union: Mapping[str, Any],
    masks_cpu: Mapping[str, Any],
    device: Any,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    batch = {key: value.to(device, non_blocking=True) for key, value in direct_union.items()}
    outputs = ppo.model_forward(model, batch, device)
    if outputs["policy_logits"].dtype != torch.bfloat16:
        raise RuntimeError("Direct512 policy output is not native BF16")
    per_row = PET.frozen.ordered_nll_per_row(outputs, batch)
    weights = batch["sample_weights"].float() * torch.where(
        batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
        torch.full_like(batch["sample_weights"].float(), PET.aggregate.ORDER_CONTEXT_WEIGHT),
        torch.ones_like(batch["sample_weights"].float()),
    )
    scalars = []
    records = []
    for name in PET.LOSS_NAMES:
        mask = masks_cpu[name].to(device=device, dtype=torch.bool)
        denominator = weights[mask].sum()
        if int(mask.sum()) != int(PET.EXPECTED_GROUP_ROWS[name]) or float(denominator) <= 0.0:
            raise RuntimeError(f"invalid Direct512 group: {name}")
        scalar = (per_row[mask] * weights[mask]).sum() / denominator
        if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
            raise FloatingPointError(f"nonfinite Direct512 loss: {name}")
        scalars.append(scalar)
        records.append({
            "name": name,
            "rows": int(mask.sum()),
            "effective_weight": float(denominator.detach().cpu()),
            "loss": float(scalar.detach().cpu()),
        })
    gradients = []
    for index, (scalar, record) in enumerate(zip(scalars, records)):
        gradient = flat_norm_gradient(
            scalar,
            parameters,
            retain_graph=index + 1 < len(scalars),
        )
        gradients.append(gradient)
        record["gradient_l2"] = float(np.linalg.norm(gradient))
    losses = {str(item["name"]): float(item["loss"]) for item in records}
    if len(gradients) != 13 or tuple(losses) != tuple(PET.LOSS_NAMES):
        raise RuntimeError("Direct512 13-loss order drift")
    return np.stack(gradients, axis=0), losses, {"count": 13, "records": records}


def evaluate_direct_losses(
    model: Any,
    direct_union: Mapping[str, Any],
    masks_cpu: Mapping[str, Any],
    device: Any,
) -> dict[str, float]:
    batch = {key: value.to(device, non_blocking=True) for key, value in direct_union.items()}
    with torch.no_grad():
        outputs = ppo.model_forward(model, batch, device)
        per_row = PET.frozen.ordered_nll_per_row(outputs, batch)
        weights = batch["sample_weights"].float() * torch.where(
            batch["contexts"] == ppo.SKILL_ORDER_CONTEXT,
            torch.full_like(batch["sample_weights"].float(), PET.aggregate.ORDER_CONTEXT_WEIGHT),
            torch.ones_like(batch["sample_weights"].float()),
        )
        result = {}
        for name in PET.LOSS_NAMES:
            mask = masks_cpu[name].to(device=device, dtype=torch.bool)
            result[name] = float(((per_row[mask] * weights[mask]).sum() / weights[mask].sum()).cpu())
    if tuple(result) != tuple(PET.LOSS_NAMES) or not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Direct512 actual loss evaluation drift")
    return result


def direct_loss_gate(
    beta_losses: Mapping[str, float], endpoint_losses: Mapping[str, float]
) -> dict[str, Any]:
    records = []
    for name in PET.LOSS_NAMES:
        delta = float(endpoint_losses[name]) - float(beta_losses[name])
        records.append({
            "name": name,
            "beta100_loss": float(beta_losses[name]),
            "endpoint_loss": float(endpoint_losses[name]),
            "endpoint_minus_beta100": delta,
            "tolerance": DIRECT_LOSS_ABS_TOLERANCE,
            "pass": delta <= DIRECT_LOSS_ABS_TOLERANCE,
        })
    return {
        "pass": len(records) == 13 and all(item["pass"] for item in records),
        "count_exact_13": len(records) == 13,
        "maximum_endpoint_minus_beta100": max(item["endpoint_minus_beta100"] for item in records),
        "records": records,
    }


def vector_sha256_float64(vector: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype="<f8"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def dual_minimum_norm_certificate(
    constraint_gradients: np.ndarray,
    constraint_rhs: np.ndarray,
    loss_gradients: np.ndarray,
    loss_rhs: np.ndarray,
    cumulative: np.ndarray,
    radius: float,
    maxabs: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Solve and certify the row-scaled convex dual; never infer from SLSQP."""
    report: dict[str, Any] = {
        "solver": "row-scaled Cholesky(AA^T) plus scipy.optimize.nnls active dual",
        "classification": "solver_failure_certificate_inconclusive",
        "certificate_pass": False,
        "radius": float(radius),
        "maxabs": float(maxabs),
    }
    try:
        matrix = np.concatenate((constraint_gradients, -loss_gradients), axis=0)
        step_rhs = np.concatenate((constraint_rhs, loss_rhs)).astype(np.float64, copy=False)
        cumulative_value = np.asarray(cumulative, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != NORM_ELEMENTS
            or step_rhs.shape != (matrix.shape[0],)
            or cumulative_value.shape != (NORM_ELEMENTS,)
            or not bool(np.all(np.isfinite(matrix)))
            or not bool(np.all(np.isfinite(step_rhs)))
        ):
            raise RuntimeError("dual input shape/finite gate failed")
        effective_rhs = step_rhs + matrix @ cumulative_value
        row_norms = np.linalg.norm(matrix, axis=1)
        if bool(np.any(row_norms <= 0.0)):
            raise RuntimeError("dual constraint contains a zero gradient row")
        scaled_matrix = matrix / row_norms[:, None]
        scaled_rhs = effective_rhs / row_norms
        rank = int(np.linalg.matrix_rank(scaled_matrix, tol=1.0e-12))
        if rank != matrix.shape[0]:
            raise RuntimeError(
                f"dual rows are not independent: rank={rank}, rows={matrix.shape[0]}"
            )
        gram = scaled_matrix @ scaled_matrix.T
        gram = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        if float(eigenvalues[0]) <= DUAL_MIN_GRAM_EIGENVALUE:
            raise RuntimeError("dual Gram matrix is not certifiably positive definite")
        cholesky = np.linalg.cholesky(gram)
        nnls_target = np.linalg.solve(cholesky, scaled_rhs)
        dual_lambda, nnls_rnorm = optimize.nnls(
            cholesky.T, nnls_target, maxiter=DUAL_NNLS_MAXITER
        )
        dual_lambda = np.asarray(dual_lambda, dtype=np.float64)
        endpoint = scaled_matrix.T @ dual_lambda
        scaled_residual = scaled_matrix @ endpoint - scaled_rhs
        raw_residual = matrix @ endpoint - effective_rhs
        dual_gradient = gram @ dual_lambda - scaled_rhs
        active = dual_lambda > DUAL_ACTIVE_LAMBDA_TOLERANCE
        inactive = ~active
        complementarity = dual_lambda * scaled_residual
        primal_objective = 0.5 * float(endpoint @ endpoint)
        dual_objective = float(scaled_rhs @ dual_lambda) - 0.5 * float(
            dual_lambda @ (gram @ dual_lambda)
        )
        duality_gap = primal_objective - dual_objective
        active_gradient = (
            float(np.abs(dual_gradient[active]).max()) if bool(active.any()) else 0.0
        )
        inactive_gradient = (
            float(dual_gradient[inactive].min()) if bool(inactive.any()) else None
        )
        endpoint_l2 = float(np.linalg.norm(endpoint))
        endpoint_maxabs = float(np.max(np.abs(endpoint), initial=0.0))
        dual_dot = float(scaled_rhs @ dual_lambda)
        dual_vector_norm = float(np.linalg.norm(scaled_matrix.T @ dual_lambda))
        dual_norm_lower_bound = (
            dual_dot / dual_vector_norm if dual_dot > 0.0 and dual_vector_norm > 0.0 else 0.0
        )
        kkt_checks = {
            "dual_lambda_nonnegative": float(dual_lambda.min()) >= -DUAL_GRADIENT_TOLERANCE,
            "scaled_primal_residual": float(scaled_residual.min()) >= -DUAL_PRIMAL_RESIDUAL_TOLERANCE,
            "raw_primal_residual": float(raw_residual.min()) >= -DUAL_PRIMAL_RESIDUAL_TOLERANCE,
            "active_dual_gradient": active_gradient <= DUAL_GRADIENT_TOLERANCE,
            "inactive_dual_margin": inactive_gradient is None or inactive_gradient >= -DUAL_GRADIENT_TOLERANCE,
            "complementarity": float(np.abs(complementarity).max(initial=0.0)) <= DUAL_COMPLEMENTARITY_TOLERANCE,
            "strong_duality": abs(duality_gap) <= DUALITY_GAP_TOLERANCE,
        }
        certificate_pass = all(kkt_checks.values())
        if certificate_pass and dual_norm_lower_bound > radius + RADIUS_ABS_TOLERANCE:
            classification = "certified_linearized_radius_infeasible"
            returned = None
        elif certificate_pass and endpoint_maxabs > maxabs + MAXABS_ABS_TOLERANCE:
            # The unconstrained minimum-norm point is not a box-feasibility
            # certificate.  Keep this distinct and inconclusive rather than
            # claiming that the maxabs-constrained polytope is empty.
            classification = "solver_unresolved_maxabs_active_bounds_required"
            returned = None
        elif certificate_pass:
            classification = "certified_feasible_minimum_norm_endpoint"
            returned = endpoint
        else:
            classification = "solver_failure_certificate_inconclusive"
            returned = None
        report.update({
            "classification": classification,
            "certificate_pass": certificate_pass,
            "constraint_rows": int(matrix.shape[0]),
            "policy_count_value_rows": int(constraint_gradients.shape[0]),
            "loss_rows_exact_13": int(loss_gradients.shape[0]),
            "columns_exact_256": int(matrix.shape[1]),
            "row_rank": rank,
            "raw_matrix_sha256_float64": vector_sha256_float64(matrix),
            "effective_rhs_sha256_float64": vector_sha256_float64(effective_rhs),
            "scaled_matrix_sha256_float64": vector_sha256_float64(scaled_matrix),
            "scaled_rhs_sha256_float64": vector_sha256_float64(scaled_rhs),
            "gram_min_eigenvalue": float(eigenvalues[0]),
            "gram_max_eigenvalue": float(eigenvalues[-1]),
            "gram_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "nnls_max_iterations": DUAL_NNLS_MAXITER,
            "nnls_residual_l2": float(nnls_rnorm),
            "lambda_active_count": int(active.sum()),
            "lambda_sha256_float64": vector_sha256_float64(dual_lambda),
            "endpoint_sha256_float64": vector_sha256_float64(endpoint),
            "endpoint_l2": endpoint_l2,
            "endpoint_maxabs": endpoint_maxabs,
            "dual_norm_lower_bound": dual_norm_lower_bound,
            "radius_certificate_margin": dual_norm_lower_bound - float(radius),
            "raw_primal_residual_min": float(raw_residual.min()),
            "scaled_primal_residual_min": float(scaled_residual.min()),
            "active_dual_gradient_max_abs": active_gradient,
            "inactive_dual_margin_min": inactive_gradient,
            "complementarity_max_abs": float(np.abs(complementarity).max(initial=0.0)),
            "primal_objective": primal_objective,
            "dual_objective": dual_objective,
            "duality_gap": duality_gap,
            "kkt_checks": kkt_checks,
            "kkt_tolerances": {
                "primal_residual": DUAL_PRIMAL_RESIDUAL_TOLERANCE,
                "dual_gradient": DUAL_GRADIENT_TOLERANCE,
                "complementarity": DUAL_COMPLEMENTARITY_TOLERANCE,
                "duality_gap": DUALITY_GAP_TOLERANCE,
            },
        })
        return returned, report
    except Exception as error:
        report.update({
            "classification": "solver_failure_certificate_inconclusive",
            "certificate_pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        return None, report


def terminal_fulltrain_dynamic_gate(
    transition: ModuleType,
    model: Any,
    device: Any,
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Authoritative full-train gate and exact dynamic-cut extractor."""
    batch_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    row_counts = {panel: 0 for panel in transition.PANEL_ORDER}
    identity_digests = {panel: hashlib.sha256() for panel in transition.PANEL_ORDER}
    cells = {
        panel: {
            metric: {cell: 0 for cell in ("cc", "cw", "wc", "ww")}
            for metric in FULLTRAIN_ALL_METRICS
        }
        for panel in transition.PANEL_ORDER
    }
    selected_cells: dict[tuple[str, str, str], str] = {}
    dynamic_specs: list[dict[str, Any]] = []
    dynamic_batches: dict[str, dict[str, Any]] = {}
    policy_regressions: list[dict[str, Any]] = []
    count_mismatches: list[dict[str, Any]] = []
    value_mismatches: list[dict[str, Any]] = []
    integrity_errors: list[dict[str, Any]] = []
    output_key_checks: list[bool] = []
    dtype_checks: list[bool] = []
    state_before = ppo.model_state_sha256(model)
    nonnorm_names = sorted(set(model.state_dict()) - set(NORM_NAMES))
    nonnorm_before = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in nonnorm_names}
    )

    def callback(
        panel: str,
        cpu_batch: Mapping[str, Any],
        identities: Sequence[Mapping[str, Any]],
        raw_snapshot: Mapping[str, Any],
        beta_snapshot: Mapping[str, Any],
    ) -> None:
        del raw_snapshot
        batch_index = batch_counts[panel]
        batch_counts[panel] += 1
        row_counts[panel] += len(identities)
        batch_key = f"{panel}:B{batch_index:03d}"
        for identity in identities:
            identity_digests[panel].update(transition.canonical_json({
                "panel": panel,
                "member": identity["member"],
                "line_index_zero_based": int(identity["line_index_zero_based"]),
                "line_sha256": identity["line_sha256"],
            }))
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        with torch.no_grad():
            outputs = ppo.model_forward(model, batch, device)
            actions, _, _, _ = ppo.sample_ordered_actions(
                outputs, batch, deterministic=True, canonicalize_order=False
            )
        output_key_checks.append(
            set(outputs) == {"policy_logits", "count_logits", "value_logits"}
        )
        dtype_checks.append(all(value.dtype == torch.bfloat16 for value in outputs.values()))
        outputs_cpu = {
            key: value.detach().float().cpu().contiguous()
            for key, value in outputs.items()
        }
        allowed_counts = ppo.count_allowed_mask(cpu_batch)
        for row, identity_source in enumerate(identities):
            identity = dict(identity_source)
            line_sha = str(identity["line_sha256"])
            candidate_record = official_row_record(cpu_batch, outputs_cpu, actions, row)
            beta_prediction = beta_snapshot["predictions"][row]
            context = int(identity["context"])
            nonempty = int(cpu_batch["action_counts"][row]) > 0
            count_changed = (
                int(candidate_record["predicted_count"])
                != int(beta_prediction["predicted_count"])
            )
            value_changed = (
                bool(candidate_record["value_sign"])
                != bool(beta_prediction["value_sign"])
            )
            for metric in FULLTRAIN_ALL_METRICS:
                if metric == "top1_correct" and not nonempty:
                    continue
                if metric.startswith("context34_") and context != 34:
                    continue
                base_metric = metric.removeprefix("context34_")
                beta_correct = bool(beta_snapshot["flags"][row][base_metric])
                candidate_correct = bool(candidate_record["flags"][base_metric])
                cell = (
                    ("c" if beta_correct else "w")
                    + ("c" if candidate_correct else "w")
                )
                cells[panel][metric][cell] += 1
                selected_cells[(panel, line_sha, metric)] = cell
                if metric not in FULLTRAIN_POLICY_METRICS or cell != "cw":
                    continue
                desired_order = [int(value) for value in beta_prediction["order"]]
                pair = derive_policy_pair(
                    cpu_batch,
                    outputs_cpu,
                    candidate_record,
                    desired_order,
                    metric,
                    row,
                )
                regression = {
                    "identity": identity,
                    "panel": panel,
                    "metric": metric,
                    "count_behavior_changed": count_changed,
                }
                if pair is None:
                    regression["dynamic_cut"] = "delegated_to_count_cut" if count_changed else "missing"
                    if not count_changed:
                        integrity_errors.append({
                            **regression,
                            "reason": "policy_cw_has_no_pair_and_no_count_flip",
                        })
                else:
                    beta_outputs = beta_snapshot["outputs_cpu"]
                    is_top1 = metric.removeprefix("context34_") == "top1_correct"
                    beta_margin = policy_margin(
                        beta_outputs, row, pair[0], pair[1]
                    )
                    if beta_margin < 0.0 and not is_top1:
                        integrity_errors.append({
                            **regression,
                            "reason": "beta_correct_policy_pair_margin_negative",
                            "beta_margin": beta_margin,
                        })
                    else:
                        current_q = local_positive_bf16_q(
                            outputs_cpu["policy_logits"][row, pair[0]],
                            outputs_cpu["policy_logits"][row, pair[1]],
                        )
                        threshold = (
                            current_q
                            if is_top1
                            else max(
                                current_q,
                                local_positive_bf16_q(
                                    beta_outputs["policy_logits"][row, pair[0]],
                                    beta_outputs["policy_logits"][row, pair[1]],
                                ),
                            )
                        )
                        dynamic_specs.append({
                            "kind": "policy",
                            "batch_key": batch_key,
                            "local_index": row,
                            "positive_index": pair[0],
                            "negative_index": pair[1],
                            "threshold": threshold,
                            "identity": identity,
                            "row_role": "fulltrain_policy_cw",
                            "origins": [f"terminal_dynamic:{metric}:positive_bf16_q"],
                        })
                        dynamic_batches[batch_key] = {"key": batch_key, "cpu_batch": cpu_batch}
                        regression["dynamic_cut"] = "policy_positive_bf16_q"
                policy_regressions.append(regression)

            if count_changed:
                fixed = bool(cpu_batch["min_counts"][row] == cpu_batch["max_counts"][row])
                record = {
                    "identity": identity,
                    "panel": panel,
                    "beta100_predicted_count": int(beta_prediction["predicted_count"]),
                    "candidate_predicted_count": int(candidate_record["predicted_count"]),
                    "fixed_count": fixed,
                }
                if fixed:
                    record["dynamic_cut"] = "integrity_error"
                    integrity_errors.append({
                        **record,
                        "reason": "fixed_count_behavior_flip_is_impossible_under_official_sampler",
                    })
                else:
                    allowed = allowed_counts[row].bool()
                    beta_logits = beta_snapshot["outputs_cpu"]["count_logits"][row].masked_fill(~allowed, -1.0e9)
                    candidate_logits = outputs_cpu["count_logits"][row].masked_fill(~allowed, -1.0e9)
                    positive = int(beta_logits.argmax())
                    negative = int(candidate_logits.argmax())
                    if positive == negative:
                        record["dynamic_cut"] = "integrity_error"
                        integrity_errors.append({
                            **record,
                            "reason": "variable_count_flip_without_masked_argmax_flip",
                        })
                    else:
                        beta_margin = float(beta_logits[positive] - beta_logits[negative])
                        if beta_margin < 0.0:
                            raise RuntimeError("beta masked-count winner margin is negative")
                        threshold = max(
                            local_positive_bf16_q(
                                beta_logits[positive], beta_logits[negative]
                            ),
                            local_positive_bf16_q(
                                candidate_logits[positive], candidate_logits[negative]
                            ),
                        )
                        dynamic_specs.append({
                            "kind": "count",
                            "batch_key": batch_key,
                            "local_index": row,
                            "positive_index": positive,
                            "negative_index": negative,
                            "threshold": threshold,
                            "identity": identity,
                            "row_role": "fulltrain_count_behavior_flip",
                            "origins": ["terminal_dynamic:beta_masked_count_positive_bf16_q"],
                        })
                        dynamic_batches[batch_key] = {"key": batch_key, "cpu_batch": cpu_batch}
                        record["dynamic_cut"] = "count_positive_bf16_q"
                count_mismatches.append(record)

            if value_changed:
                beta_sign = bool(beta_prediction["value_sign"])
                multiplier = 1 if beta_sign else -1
                beta_signed_margin = float(
                    beta_snapshot["outputs_cpu"]["value_logits"][row]
                ) * multiplier
                if beta_signed_margin < 0.0:
                    raise RuntimeError("beta value sign/margin inconsistency")
                dynamic_specs.append({
                    "kind": "value",
                    "batch_key": batch_key,
                    "local_index": row,
                    "sign_multiplier": multiplier,
                    "threshold": zero_positive_bf16_q(),
                    "identity": identity,
                    "row_role": "fulltrain_value_sign_flip",
                    "origins": ["terminal_dynamic:beta_value_sign_zero_positive_bf16_q"],
                })
                dynamic_batches[batch_key] = {"key": batch_key, "cpu_batch": cpu_batch}
                value_mismatches.append({
                    "identity": identity,
                    "panel": panel,
                    "beta100_value_sign": beta_sign,
                    "candidate_value_sign": bool(candidate_record["value_sign"]),
                    "dynamic_cut": "value_beta_signed_margin",
                })

    transition_report = transition.run_transition_audit(
        row_consumer=callback, candidate_consumer=None
    )
    frozen_closure = transition_absolute_closure(transition_report)
    expected_batches = {
        panel: math.ceil(int(transition.EXPECTED_TRAIN[panel]["rows"]) / 256)
        for panel in transition.PANEL_ORDER
    }
    expected_rows = {
        panel: int(transition.EXPECTED_TRAIN[panel]["rows"])
        for panel in transition.PANEL_ORDER
    }
    identity_sha = {
        panel: identity_digests[panel].hexdigest() for panel in transition.PANEL_ORDER
    }
    coverage = (
        batch_counts == expected_batches
        and row_counts == expected_rows
        and identity_sha == transition.EXPECTED_IDENTITY_SHA256
    )
    transition_records = []
    for panel in transition.PANEL_ORDER:
        for metric in FULLTRAIN_ALL_METRICS:
            counts = cells[panel][metric]
            denominator = sum(counts.values())
            beta_correct = counts["cc"] + counts["cw"]
            candidate_correct = counts["cc"] + counts["wc"]
            frozen_beta = int(transition.EXPECTED_AGGREGATES[panel]["beta100"][metric])
            expected_denominator = int(TERMINAL_DENOMINATORS[panel][metric])
            minimum = int(
                TERMINAL_MINIMUM_CANDIDATE_COUNTS.get(panel, {}).get(metric, frozen_beta)
            )
            policy = metric in FULLTRAIN_POLICY_METRICS
            passed = (
                denominator == expected_denominator
                and beta_correct == frozen_beta
                and counts["wc"] - counts["cw"] == candidate_correct - beta_correct
                and (
                    candidate_correct >= minimum and counts["cw"] == 0
                    if policy
                    else True
                )
            )
            transition_records.append({
                "panel": panel,
                "metric": metric,
                "cells": dict(counts),
                "denominator": denominator,
                "expected_denominator": expected_denominator,
                "beta_cc_plus_cw": beta_correct,
                "frozen_beta_correct": frozen_beta,
                "candidate_cc_plus_wc": candidate_correct,
                "minimum_candidate_correct": minimum,
                "cw_exact_zero_required": policy,
                "pass": passed,
            })
    selected_records = []
    for descriptor in descriptors:
        expected_cell = "wc" if descriptor["role"] == "target_transition" else "cc"
        panel = str(descriptor["source"])
        line_sha = str(descriptor["identity"]["line_sha256"])
        for metric in descriptor["required_metrics"]:
            observed = selected_cells.get((panel, line_sha, str(metric)))
            selected_records.append({
                "identity": descriptor["identity"],
                "role": descriptor["role"],
                "metric": metric,
                "expected_cell": expected_cell,
                "observed_cell": observed,
                "pass": observed == expected_cell,
            })
    target_records = [item for item in selected_records if item["role"] == "target_transition"]
    guard_records = [item for item in selected_records if item["role"] != "target_transition"]
    state_after = ppo.model_state_sha256(model)
    nonnorm_after = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in nonnorm_names}
    )
    behavior_exact = not count_mismatches and not value_mismatches
    passed = (
        coverage
        and frozen_closure["pass"]
        and all(output_key_checks)
        and all(dtype_checks)
        and len(transition_records) == 24
        and all(item["pass"] for item in transition_records)
        and behavior_exact
        and not integrity_errors
        and len(target_records) == 9
        and all(item["pass"] for item in target_records)
        and len(guard_records) == 388
        and all(item["pass"] for item in guard_records)
        and state_after == state_before
        and nonnorm_before == nonnorm_after == EXPECTED_BETA_NONNORM78_SHA256
    )
    report = {
        "pass": passed,
        "candidate_consumer_used": False,
        "batch_counts": batch_counts,
        "batch_count_total_exact_95": sum(batch_counts.values()) == 95,
        "row_counts": row_counts,
        "row_count_total_exact_24050": sum(row_counts.values()) == 24050,
        "identity_stream_sha256": identity_sha,
        "row_consumer_full_coverage": coverage,
        "transition_records": transition_records,
        "transition_record_count_exact_24": len(transition_records) == 24,
        "policy6_cw0_and_target_minima_pass": all(
            item["pass"] for item in transition_records
            if item["metric"] in FULLTRAIN_POLICY_METRICS
        ),
        "count_predicted_behavior_mismatch_count": len(count_mismatches),
        "count_predicted_behavior_exact_beta100_row_by_row": not count_mismatches,
        "value_sign_mismatch_count": len(value_mismatches),
        "value_sign_exact_beta100_row_by_row": not value_mismatches,
        "count_behavior_mismatches": count_mismatches,
        "value_behavior_mismatches": value_mismatches,
        "policy_cw_regressions": policy_regressions,
        "integrity_errors": integrity_errors,
        "dynamic_cut_spec_count": len(dynamic_specs),
        "all_behavior_flips_have_dynamic_cuts_or_explicit_integrity_error": (
            len(dynamic_specs) + len(integrity_errors) > 0
            if policy_regressions or count_mismatches or value_mismatches
            else True
        ),
        "exact_target_wc_records": target_records,
        "exact_target_obligation_count_exact_9": len(target_records) == 9,
        "pf_gain_and_96_near_cc_records": guard_records,
        "guard_obligation_count_exact_388": len(guard_records) == 388,
        "frozen_raw_beta_transition_absolute_closure": frozen_closure,
        "candidate_model_state_unchanged_by_forward": state_after == state_before,
        "candidate_nonnorm78_sha256_before": nonnorm_before,
        "candidate_nonnorm78_sha256_after": nonnorm_after,
        "candidate_nonnorm78_exact_beta100": (
            nonnorm_before == nonnorm_after == EXPECTED_BETA_NONNORM78_SHA256
        ),
    }
    return report, dynamic_specs, dynamic_batches


def active_constraint_gate(
    model: Any,
    batch_registry: Mapping[str, Mapping[str, Any]],
    active: Mapping[tuple[Any, ...], Mapping[str, Any]],
    device: Any,
) -> dict[str, Any]:
    by_batch: dict[str, list[Mapping[str, Any]]] = {}
    for key in sorted(active):
        item = active[key]
        by_batch.setdefault(str(item["batch_key"]), []).append(item)
    records = []
    with torch.no_grad():
        for batch_key in sorted(by_batch):
            cpu_batch = batch_registry[batch_key]["cpu_batch"]
            batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
            outputs = ppo.model_forward(model, batch, device)
            for item in by_batch[batch_key]:
                margin = float(constraint_margin_from_outputs(outputs, item).cpu())
                threshold = float(item["threshold"])
                records.append({
                    "key": list(constraint_key(item)),
                    "kind": item["kind"],
                    "margin": margin,
                    "threshold": threshold,
                    "residual": margin - threshold,
                    "pass": margin + PAIR_TOLERANCE >= threshold,
                })
    return {
        "pass": len(records) == len(active) and all(item["pass"] for item in records),
        "constraint_count": len(records),
        "residual_min": min((item["residual"] for item in records), default=None),
        "records": records,
    }


def add_selected_dynamic_constraints(
    descriptors: Sequence[Mapping[str, Any]],
    beta_snapshot: Mapping[str, Any],
    current_snapshot: Mapping[str, Any],
    active: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    gate = selected_gate(descriptors, current_snapshot)
    updates = []
    unresolved = []
    for obligation in gate["false_obligations"]:
        descriptor = descriptors[int(obligation["descriptor_index"])]
        key = str(descriptor["batch_key"])
        row = int(descriptor["local_index"])
        current = current_snapshot[key]
        desired = beta_snapshot[key]["official_rows"][row]["expert_order"]
        if descriptor["role"] == "target_transition":
            desired = current["official_rows"][row]["expert_order"]
        pair = derive_policy_pair(
            descriptor["cpu_batch"],
            current["outputs_cpu"],
            current["official_rows"][row],
            desired,
            str(obligation["metric"]),
            row,
        )
        if pair is None:
            unresolved.append({
                "identity": descriptor["identity"],
                "metric": obligation["metric"],
                "reason": "selected_false_obligation_has_no_policy_pair",
            })
            continue
        if descriptor["role"] == "target_transition":
            logits = current["outputs_cpu"]["policy_logits"]
            threshold = local_positive_bf16_q(logits[row, pair[0]], logits[row, pair[1]])
            origin = "selected_dynamic_target_positive_bf16_q"
        else:
            beta_outputs = beta_snapshot[key]["outputs_cpu"]
            is_top1 = (
                str(obligation["metric"]).removeprefix("context34_")
                == "top1_correct"
            )
            beta_margin = policy_margin(
                beta_outputs, row, pair[0], pair[1]
            )
            if beta_margin < 0.0 and not is_top1:
                unresolved.append({
                    "identity": descriptor["identity"],
                    "metric": obligation["metric"],
                    "reason": "selected_dynamic_beta_guard_margin_negative",
                })
                continue
            current_q = local_positive_bf16_q(
                current["outputs_cpu"]["policy_logits"][row, pair[0]],
                current["outputs_cpu"]["policy_logits"][row, pair[1]],
            )
            threshold = (
                current_q
                if is_top1
                else max(
                    current_q,
                    local_positive_bf16_q(
                        beta_outputs["policy_logits"][row, pair[0]],
                        beta_outputs["policy_logits"][row, pair[1]],
                    ),
                )
            )
            origin = "selected_dynamic_guard_positive_bf16_q"
        if threshold < 0.0:
            unresolved.append({
                "identity": descriptor["identity"],
                "metric": obligation["metric"],
                "reason": "selected_dynamic_threshold_negative",
            })
            continue
        updates.append(add_or_strengthen_constraint(active, {
            "kind": "policy",
            "batch_key": key,
            "local_index": row,
            "positive_index": pair[0],
            "negative_index": pair[1],
            "threshold": threshold,
            "identity": descriptor["identity"],
            "row_role": descriptor["role"],
            "origins": [f"{origin}:{obligation['metric']}"],
        }))
    return {
        "pass": not unresolved,
        "updates": updates,
        "unresolved": unresolved,
        "selected_gate": gate,
    }


def canonical_active_ledger(
    active: Mapping[tuple[Any, ...], Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for key in sorted(active):
        item = active[key]
        record = {
            "key": list(key),
            "kind": item["kind"],
            "identity": item["identity"],
            "row_role": item["row_role"],
            "threshold": float(item["threshold"]),
            "origins": list(item["origins"]),
        }
        if item["kind"] in {"policy", "count"}:
            record.update({
                "positive_index": int(item["positive_index"]),
                "negative_index": int(item["negative_index"]),
            })
        else:
            record["sign_multiplier"] = int(item["sign_multiplier"])
        output.append(record)
    return output


def run_probe(
    transition_context: Mapping[str, Any] | None = None,
    candidate_consumer: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if candidate_consumer is not None and not callable(candidate_consumer):
        raise TypeError("candidate_consumer must be callable or None")
    if transition_context is None:
        transition, transition_evidence = load_transition_module()
        transition_context = build_actual_transition_context(transition)
    else:
        transition_evidence = {"injected_context": True}
        transition = transition_context.get("_transition_module")
    if not isinstance(transition, ModuleType):
        raise RuntimeError("actual probe requires hash-bound transition module")
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
    selected_batches = transition_context["selected_batches"]
    if next(model.parameters()).device != device:
        raise RuntimeError("beta100 model is not on cuda:0")
    beta_state_sha = ppo.model_state_sha256(model)
    if beta_state_sha != transition.BETA_MODEL_SHA256:
        raise RuntimeError("clean beta100 start SHA drift")

    parameters = configure_finalnorm(model)
    beta_norm = {name: parameters[name].detach().clone() for name in NORM_NAMES}
    norm_sha = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in NORM_NAMES}
    )
    nonnorm_names = sorted(set(model.state_dict()) - set(NORM_NAMES))
    nonnorm_sha = PET.frozen.model_state_sha256(
        {name: model.state_dict()[name] for name in nonnorm_names}
    )
    if norm_sha != EXPECTED_BETA_NORM2_SHA256 or nonnorm_sha != EXPECTED_BETA_NONNORM78_SHA256:
        raise RuntimeError("beta100 norm2/nonnorm78 subset SHA drift")

    basis = PET.build_basis_geometry()
    cache = PET.build_cache_context(basis["payloads"]["R"])
    beta_selected = snapshot_batches(model, selected_batches, device)
    for descriptor in descriptors:
        observed = beta_selected[str(descriptor["batch_key"])]["official_rows"][
            int(descriptor["local_index"])
        ]["flags"]
        if any(
            bool(observed[metric]) != bool(descriptor["beta_flags"][metric])
            for metric in METRICS
        ):
            raise RuntimeError("transition/beta selected official flag mismatch")

    active: dict[tuple[Any, ...], dict[str, Any]] = {}
    initial = initialize_selected_constraints(descriptors, beta_selected, active)
    if not initial["pass"]:
        return {
            "schema_version": SCHEMA,
            "status": "integrity_error_no_candidate",
            "decision": {
                "classification": "initial_selected_constraint_construction_integrity_error",
                "unresolved": initial["unresolved"],
            },
            "transition_context": context_audit,
            "writes": 0,
        }
    if len(active) != EXPECTED_INITIAL_CONSTRAINT_ROWS:
        raise RuntimeError(
            f"initial FinalNorm constraint row drift: {len(active)} != {EXPECTED_INITIAL_CONSTRAINT_ROWS}"
        )
    batch_registry = {str(item["key"]): item for item in selected_batches}
    beta_direct_losses = evaluate_direct_losses(
        model, cache["direct_union"], cache["loss_masks"], device
    )

    cumulative = np.zeros(NORM_ELEMENTS, dtype=np.float64)
    endpoints = 0
    overlay_started = False
    consumer_called = False
    success = False
    classification = "endpoint_budget_exhausted_without_certificate"
    iterations: list[dict[str, Any]] = []
    terminal_gate: dict[str, Any] | None = None
    current_selected = beta_selected
    current_selected_gate = selected_gate(descriptors, current_selected)
    current_active_gate = active_constraint_gate(
        model, batch_registry, active, device
    )
    current_direct_gate = direct_loss_gate(beta_direct_losses, beta_direct_losses)
    final_integrity: dict[str, Any] = {}

    try:
        while endpoints < MAX_ENDPOINTS and not success:
            selected_dynamic = add_selected_dynamic_constraints(
                descriptors, beta_selected, current_selected, active
            )
            if not selected_dynamic["pass"]:
                classification = "integrity_error_selected_obligation_has_no_exact_cut"
                iterations.append({
                    "endpoint_index": endpoints + 1,
                    "kind": "closed_before_solver",
                    "selected_dynamic": selected_dynamic,
                })
                break
            constraint_gradients, constraint_rhs, constraint_audit = collect_constraint_gradients(
                model, parameters, batch_registry, active, device
            )
            loss_gradients, current_losses, loss_audit = collect_direct_loss_gradients(
                model,
                parameters,
                cache["direct_union"],
                cache["loss_masks"],
                device,
            )
            loss_rhs = np.asarray([
                current_losses[name] - beta_direct_losses[name] - DIRECT_LOSS_ABS_TOLERANCE
                for name in PET.LOSS_NAMES
            ], dtype=np.float64)
            ladder_audits = []
            endpoint: np.ndarray | None = None
            chosen_stage: int | None = None
            for stage, limits in enumerate(LADDER):
                endpoint_candidate, certificate = dual_minimum_norm_certificate(
                    constraint_gradients,
                    constraint_rhs,
                    loss_gradients,
                    loss_rhs,
                    cumulative,
                    float(limits["radius"]),
                    float(limits["maxabs"]),
                )
                ladder_audits.append(certificate)
                if endpoint_candidate is not None:
                    endpoint = endpoint_candidate
                    chosen_stage = stage
                    break
                if certificate["classification"] == "solver_failure_certificate_inconclusive":
                    classification = "solver_failure_feasibility_not_certified"
                    break
                if certificate["classification"] == "solver_unresolved_maxabs_active_bounds_required":
                    if stage + 1 == len(LADDER):
                        classification = "solver_unresolved_maxabs_feasibility_not_certified"
                    continue
                if stage + 1 == len(LADDER):
                    classification = "certified_linearized_infeasible_final_radius"
            if endpoint is None:
                iterations.append({
                    "endpoint_index": endpoints + 1,
                    "kind": "closed_before_endpoint",
                    "constraint_gradients": constraint_audit,
                    "loss_gradients": loss_audit,
                    "loss_rhs": [float(value) for value in loss_rhs],
                    "ladder_certificates": ladder_audits,
                    "classification": classification,
                })
                break
            limits = LADDER[int(chosen_stage)]
            planned_l2 = float(np.linalg.norm(endpoint))
            planned_maxabs = float(np.max(np.abs(endpoint), initial=0.0))
            if (
                planned_l2 > float(limits["radius"]) + RADIUS_ABS_TOLERANCE
                or planned_maxabs > float(limits["maxabs"]) + MAXABS_ABS_TOLERANCE
            ):
                raise RuntimeError("certified endpoint violates its radius/maxabs ladder")
            overlay_started = True
            set_norm_delta(parameters, beta_norm, endpoint)
            actual = norm_delta_from_model(parameters, beta_norm)
            actual_l2 = float(np.linalg.norm(actual))
            actual_maxabs = float(np.max(np.abs(actual), initial=0.0))
            if (
                actual_l2 > float(limits["radius"]) + RADIUS_ABS_TOLERANCE
                or actual_maxabs > float(limits["maxabs"]) + MAXABS_ABS_TOLERANCE
            ):
                classification = "actual_fp32_endpoint_failed_radius_or_maxabs"
                iterations.append({
                    "endpoint_index": endpoints + 1,
                    "kind": "rejected_fp32_endpoint",
                    "planned_l2": planned_l2,
                    "planned_maxabs": planned_maxabs,
                    "actual_l2": actual_l2,
                    "actual_maxabs": actual_maxabs,
                    "ladder_certificates": ladder_audits,
                })
                break
            cumulative = actual.copy()
            endpoints += 1
            observed_nonnorm = PET.frozen.model_state_sha256(
                {name: model.state_dict()[name] for name in nonnorm_names}
            )
            if observed_nonnorm != EXPECTED_BETA_NONNORM78_SHA256:
                raise RuntimeError("nonnorm78 changed during FinalNorm endpoint")
            current_selected = snapshot_batches(model, selected_batches, device)
            current_selected_gate = selected_gate(descriptors, current_selected)
            current_active_gate = active_constraint_gate(
                model, batch_registry, active, device
            )
            endpoint_losses = evaluate_direct_losses(
                model, cache["direct_union"], cache["loss_masks"], device
            )
            current_direct_gate = direct_loss_gate(beta_direct_losses, endpoint_losses)
            endpoint_record: dict[str, Any] = {
                "endpoint_index": endpoints,
                "kind": "finalnorm_cuttingplane_endpoint",
                "chosen_ladder_stage": chosen_stage,
                "limits": dict(limits),
                "constraint_gradients": constraint_audit,
                "loss_gradients": loss_audit,
                "ladder_certificates": ladder_audits,
                "actual_delta_l2": actual_l2,
                "actual_delta_maxabs": actual_maxabs,
                "actual_delta_float32_sha256": vector_sha256_float32(actual),
                "selected_gate": current_selected_gate,
                "active_constraint_gate": current_active_gate,
                "direct512_actual_loss_gate": current_direct_gate,
                "nonnorm78_sha256": observed_nonnorm,
            }
            local_pass = (
                current_selected_gate["pass"]
                and current_active_gate["pass"]
                and current_direct_gate["pass"]
            )
            # Run the authoritative stream for every actual endpoint, not
            # only after the selected-row gate passes.  In particular, a
            # selected policy failure can be caused by a changed predicted
            # count; only the full behavior gate can produce the exact
            # beta-masked-count cut for that row.
            terminal_gate, dynamic_specs, dynamic_batches = terminal_fulltrain_dynamic_gate(
                transition, model, device, descriptors
            )
            endpoint_record["terminal_fulltrain_dynamic_gate"] = terminal_gate
            if terminal_gate["integrity_errors"]:
                classification = "integrity_error_in_terminal_fulltrain_gate"
            else:
                if dynamic_specs:
                    batch_registry.update(dynamic_batches)
                    updates = [
                        add_or_strengthen_constraint(active, spec)
                        for spec in dynamic_specs
                    ]
                    endpoint_record["terminal_dynamic_cut_updates"] = updates
                if local_pass and terminal_gate["pass"]:
                    success = True
                    classification = "certified_fulltrain_finalnorm_endpoint_success"
                elif local_pass and not dynamic_specs:
                    classification = "terminal_gate_failed_without_exact_dynamic_cut"
            iterations.append(endpoint_record)
            if success or classification in {
                "integrity_error_in_terminal_fulltrain_gate",
                "terminal_gate_failed_without_exact_dynamic_cut",
                "actual_fp32_endpoint_failed_radius_or_maxabs",
            }:
                break

        ledger = canonical_active_ledger(active)
        if success and candidate_consumer is not None:
            consumer_called = True
            candidate_consumer({
                "model": model,
                "beta_checkpoint": beta_checkpoint,
                "beta_model_state_sha256": beta_state_sha,
                "actual_finalnorm_delta_float32_cpu": torch.from_numpy(
                    cumulative.astype(np.float32, copy=True)
                ),
                "actual_finalnorm_delta_float32_sha256": vector_sha256_float32(cumulative),
                "actual_finalnorm_delta_l2": float(np.linalg.norm(cumulative)),
                "active_constraint_ledger": ledger,
                "terminal_fulltrain_dynamic_gate": terminal_gate,
            })
    finally:
        if overlay_started:
            set_norm_delta(parameters, beta_norm, np.zeros(NORM_ELEMENTS, dtype=np.float64))
        restored_sha = ppo.model_state_sha256(model)
        restored_nonnorm = PET.frozen.model_state_sha256(
            {name: model.state_dict()[name] for name in nonnorm_names}
        )
        restored_norm = PET.frozen.model_state_sha256(
            {name: model.state_dict()[name] for name in NORM_NAMES}
        )
        final_integrity = {
            "finally_beta100_restore_executed": overlay_started,
            "beta100_model_state_sha256_before": beta_state_sha,
            "beta100_model_state_sha256_after": restored_sha,
            "beta100_model_state_restored_exact": restored_sha == beta_state_sha,
            "restored_norm2_sha256": restored_norm,
            "restored_norm2_exact_beta100": restored_norm == EXPECTED_BETA_NORM2_SHA256,
            "restored_nonnorm78_sha256": restored_nonnorm,
            "restored_nonnorm78_exact_beta100": restored_nonnorm == EXPECTED_BETA_NONNORM78_SHA256,
            "restored_norm_delta_l2": float(np.linalg.norm(norm_delta_from_model(parameters, beta_norm))),
        }
        if not all((
            final_integrity["beta100_model_state_restored_exact"],
            final_integrity["restored_norm2_exact_beta100"],
            final_integrity["restored_nonnorm78_exact_beta100"],
            final_integrity["restored_norm_delta_l2"] == 0.0,
        )):
            raise RuntimeError("finally beta100 full restoration failed")

    if success:
        status = "certified_fulltrain_candidate_success"
    elif classification.startswith("certified_linearized_infeasible"):
        status = "certified_no_candidate_within_final_radius"
    elif classification.startswith("solver_"):
        status = "solver_inconclusive_no_candidate"
    elif classification.startswith("integrity_error"):
        status = "integrity_error_no_candidate"
    else:
        status = "budget_exhausted_no_candidate_no_infeasibility_certificate"
    ledger = canonical_active_ledger(active)
    return {
        "schema_version": SCHEMA,
        "status": status,
        "scope": {
            "start": "clean beta100",
            "mutable_tensors": list(NORM_NAMES),
            "mutable_elements_fp32": NORM_ELEMENTS,
            "nonnorm78_exact_beta100": True,
            "train_only": True,
            "validation_opened": False,
            "target_count": 3,
            "pokemonfan_gain_guard_count": 1,
            "near_guard_count": 96,
            "direct512_loss_count": 13,
            "direct512_tolerance": DIRECT_LOSS_ABS_TOLERANCE,
            "radius_maxabs_ladder": [dict(item) for item in LADDER],
            "max_endpoints": MAX_ENDPOINTS,
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
        "pet_geometry_input_report": basis["report"],
        "direct512_cache": cache["report"]["direct512"],
        "beta100_subset_sha256": {
            "norm2": norm_sha,
            "nonnorm78": nonnorm_sha,
        },
        "beta100_direct512_ordered_losses": beta_direct_losses,
        "initial_selected_constraints": initial,
        "iterations": iterations,
        "decision": {
            "status": status,
            "classification": classification,
            "endpoints_evaluated": endpoints,
            "success": success,
            "terminal_delta_l2_before_restore": float(np.linalg.norm(cumulative)),
            "terminal_delta_maxabs_before_restore": float(np.max(np.abs(cumulative), initial=0.0)),
            "terminal_delta_float32_sha256": vector_sha256_float32(cumulative),
            "active_constraint_count": len(ledger),
            "active_constraint_ledger_sha256": sha256_bytes(canonical_json_bytes(ledger)),
            "candidate_consumer_called_before_finally_restore": consumer_called,
            "model_materialized": False,
        },
        "final_selected_gate": current_selected_gate,
        "final_active_constraint_gate": current_active_gate,
        "final_direct512_actual_loss_gate": current_direct_gate,
        "terminal_fulltrain_dynamic_gate": terminal_gate,
        "final_integrity": final_integrity,
    }


def cache_audit(transition: ModuleType) -> dict[str, Any]:
    basis = PET.build_basis_geometry()
    cache = PET.build_cache_context(basis["payloads"]["R"])
    transition_cache = transition.cache_audit()
    if transition_cache.get("status") != "cache_audit_passed":
        raise RuntimeError("transition cache audit status drift")
    if (
        transition_cache.get("scope", {}).get("validation_member_payloads_opened")
        is not False
    ):
        raise RuntimeError("transition cache opened validation")
    rows = sum(
        int(panel["rows"])
        for panel in transition_cache["cache"]["panels"].values()
    )
    if rows != 24050:
        raise RuntimeError(f"transition cache row-count drift: {rows}")
    beta_state = basis["payloads"]["P"]["model_state_dict"]
    if len(beta_state) != 80 or set(NORM_NAMES) - set(beta_state):
        raise RuntimeError("beta100 cache state schema drift")
    norm_sha = PET.frozen.model_state_sha256(
        {name: beta_state[name] for name in NORM_NAMES}
    )
    nonnorm_names = sorted(set(beta_state) - set(NORM_NAMES))
    nonnorm_sha = PET.frozen.model_state_sha256(
        {name: beta_state[name] for name in nonnorm_names}
    )
    if (
        norm_sha != EXPECTED_BETA_NORM2_SHA256
        or nonnorm_sha != EXPECTED_BETA_NONNORM78_SHA256
        or len(nonnorm_names) != 78
    ):
        raise RuntimeError("beta100 norm2/nonnorm78 cache hash drift")
    return {
        "status": "cache_audit_passed_zero_write_train_only",
        "transition": transition_cache,
        "pet_geometry": basis["report"],
        "pet_cache": cache["report"],
        "beta100_scope": {
            "state_tensor_count_exact_80": len(beta_state) == 80,
            "norm2_names": list(NORM_NAMES),
            "norm2_elements_exact_256": sum(int(beta_state[name].numel()) for name in NORM_NAMES) == NORM_ELEMENTS,
            "norm2_all_fp32": all(beta_state[name].dtype == torch.float32 for name in NORM_NAMES),
            "norm2_sha256": norm_sha,
            "nonnorm78_tensor_count": len(nonnorm_names),
            "nonnorm78_sha256": nonnorm_sha,
        },
        "complete_train_rows_exact_24050": rows == 24050,
        "target_identity_count": len(EXPECTED_TARGETS),
        "pokemonfan_gain_identity_count": len(EXPECTED_PF_GAIN_GUARDS),
        "near_identity_count_selected_in_actual_callback": 3 * NEAR_GUARD_PER_SOURCE,
        "validation_opened": False,
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
            "input_lock": {
                "self": read_regular_bytes(SCRIPT, None, "self")[1],
                "pet_geometry": PET_EVIDENCE,
                "transition": transition_evidence,
            },
            **cache_audit(transition),
        }
    else:
        result = run_probe(candidate_consumer=None)
    print(canonical_json_bytes(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
