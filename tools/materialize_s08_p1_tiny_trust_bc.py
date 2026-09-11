#!/usr/bin/env python3
"""Materialize one gate-passing tiny P1 actor6 step from the frozen S8.

The P1 direction is never loaded from an untrusted sidecar.  This tool
recomputes the complete natural ordered-selection gradient for each of the
three frozen fit sources and requires the resulting unit P1 vector to exactly
match the already reviewed geometry report.  It then tries the preregistered
R50 radius and, only when R50 fails calibration, the smaller R25 radius.

Radius selection uses only the time-forward calibration partitions.  Once a
single radius is locked, the anti-KD dev view and the two Aug-7 behavior views
are evaluated for a same-run S8 baseline and that candidate only.  The unique
candidate is repeated once as a deterministic confirmation.  A final failure
rejects the route; it can never fall back to the other radius.

Every behavior gate is evaluated under both CPU/FP32 and CUDA/BF16 forward
semantics.  The former is a batched arithmetic proxy for the batch-one CPU
deployment path; the latter matches the strict local H2H arithmetic path.
Selection losses are aggregated over the whole domain by their effective
weight sums, never by a second per-batch row average.

Formal operation is two-stage: freeze and review a canonical plan, then
execute exactly that plan.  Only a fully passing unique evaluation-only
checkpoint is written.  A rejection writes JSON evidence but no checkpoint.
There is no gameplay, H2H, packaging, upload, or submission path here.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
EXPECTED_ENV_PREFIX = Path("/home/xxc/miniconda3/envs/my_project_env")
EXPECTED_CUBLAS = ":4096:8"
EXPECTED_CUDA_VISIBLE_DEVICES = "0"

SCHEMA_VERSION = "ptcg-s08-p1-tiny-trust-bc-materializer-v1"
PLAN_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v1.reviewed_plan.json"
)
RESULT_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v1.result.json"
)
CHECKPOINT_PATH = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_p1_tiny_trust_bc_v1.pt"
)

GEOMETRY_TOOL = TOOLS / "probe_s08_antikd_fullsource_bc_geometry.py"
GEOMETRY_TOOL_SHA256 = (
    "6358457c88dd515eb16d72dcd89c261a59ede96758055438905a19295cfed575"
)
GEOMETRY_PLAN = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_antikd_fullsource_bc_geometry_v1.reviewed_plan.json"
)
GEOMETRY_PLAN_FILE_SHA256 = (
    "884db290fec44aa7ac1782fdc9243ce14ac6eae6fd788b48f69852d41fab6114"
)
GEOMETRY_PLAN_SHA256 = (
    "a2d9c1160f78ba46f9b738799c12dc346042406b9bc986b7226c12e1e1841f8e"
)
GEOMETRY_REPORT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_antikd_fullsource_bc_geometry_v1.json"
)
GEOMETRY_REPORT_SHA256 = (
    "03ef63e55def30415f873c08b1c2d3a181a9c7f7fd1dc1671cfc99b86d115ca0"
)

S8_PARENT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/postppo_tail_repair_r2best_v1/"
    "postppo-special-bc-s08.pt"
)
BC_ARCHITECTURE = (
    ROOT
    / "artifacts/gold_push_20260810_v1/bc/"
    "marnie_trainwins_seed1011/best.pt"
)
ANTI_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_antikd_exact_20260810_v1/"
    "marnie_exact_anti_kd_trainwins.zip"
)
FROS_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_fros_exact_20260810_v1/"
    "marnie_vs_froslass_trainwins.zip"
)
GENERAL_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_general_anchor_20260810_v1/"
    "marnie_general_anchor_train.zip"
)
FROS_VALID_ARCHIVE = (
    ROOT
    / "data/gold_push_marnie_fros_valid_20260810_v1/"
    "marnie_exact_fros_valid_view.zip"
)
GENERAL_VALID_ARCHIVE = (
    ROOT / "data/gold_push_recent7_20260810_v1/archives/marnie.zip"
)

BEHAVIOR_CORE = TOOLS / "evaluate_policy_bc.py"
BC_CORE = TOOLS / "train_bc_orbit.py"
TRAINING_CORE = TOOLS / "run_gold_push_postppo_tail_repair.py"
TRAINER_MODULE = TOOLS / "train_ppo.py"

# These direct dependencies are authenticated before the frozen geometry
# module is imported and allowed to execute.
PREIMPORT_INPUTS: dict[str, tuple[Path, str]] = {
    "geometry_tool": (GEOMETRY_TOOL, GEOMETRY_TOOL_SHA256),
    "behavior_core": (
        BEHAVIOR_CORE,
        "69d3d7046edadc86e360667307abfb60aaa2af2e153c074ea726943f490c13a4",
    ),
    "bc_core": (
        BC_CORE,
        "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc",
    ),
    "training_core": (
        TRAINING_CORE,
        "d1484f37030b128db6b0bffa9e07b7e387fcc6b95d2d5e57b937e615421ca53e",
    ),
    "trainer_module": (
        TRAINER_MODULE,
        "321a4e25fb3ccecb0dfb3c37803a20369b8083619f6b238c3d067990b7e397cf",
    ),
}

IMMUTABLE_INPUTS: dict[str, tuple[Path, str]] = {
    **PREIMPORT_INPUTS,
    "geometry_plan": (GEOMETRY_PLAN, GEOMETRY_PLAN_FILE_SHA256),
    "geometry_report": (GEOMETRY_REPORT, GEOMETRY_REPORT_SHA256),
    "s8_parent": (
        S8_PARENT,
        "d7443bda57cb89a5c12d9d776710d1b8b2c01151541573e01f7e04b032ea25e9",
    ),
    "bc_architecture": (
        BC_ARCHITECTURE,
        "dda68d51d9b922526709149143aa8409ba287fb8f31ddbb2293f0ea543ef01a0",
    ),
    "anti_archive": (
        ANTI_ARCHIVE,
        "4ee00aa4fdc632bf3e9218796b055d148a3e1d388ed65192354253b6ea09179d",
    ),
    "fros_archive": (
        FROS_ARCHIVE,
        "a148e42baf8e84f94150cf9d2437c1e04f30a8b376575e0db3976ee9da31e044",
    ),
    "general_archive": (
        GENERAL_ARCHIVE,
        "3a95c8c62ba706e1226b79404b4f2a1ce8663e57582eafde49bcdb41e43db9db",
    ),
    "fros_valid_archive": (
        FROS_VALID_ARCHIVE,
        "559fc2678c45f0c1509dc9c38c55a6dd09fc15fbdf1d1c3a3bc0f7b086bbddf3",
    ),
    "general_valid_archive": (
        GENERAL_VALID_ARCHIVE,
        "6b3873b28bfad70377d0a3b22fe1163516a3d40b5af429885c0ce15f75b402a0",
    ),
    "python": (
        EXPECTED_PYTHON.resolve(),
        "35010543d1379740c163ebf34e92108891c70cceb393367d71f463733c8be497",
    ),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, expected_sha256: str, label: str) -> Path:
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    return path


for _input_name, (_input_path, _input_sha) in PREIMPORT_INPUTS.items():
    require_regular_file(_input_path, _input_sha, _input_name)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_PROBE_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_s08_geometry_6358457c", GEOMETRY_TOOL
)
if _PROBE_SPEC is None or _PROBE_SPEC.loader is None:
    raise RuntimeError("Cannot construct frozen geometry probe import")
probe: ModuleType = importlib.util.module_from_spec(_PROBE_SPEC)
sys.modules[_PROBE_SPEC.name] = probe
_PROBE_SPEC.loader.exec_module(probe)

torch = probe.torch
behavior_core = probe.behavior_core
training_core = probe.training_core
ppo = probe.ppo

EXPECTED_TORCH_VERSION = "2.8.0+cu128"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_CUDNN_VERSION = 91002
EXPECTED_ORJSON_VERSION = "3.11.9"

ACTOR6 = tuple(probe.ACTOR6)
P1_NAME = "P1_raw_source_mean_2_1_1"
P1_VECTOR_SHA256 = (
    "35a56e2b3060f7d239bf0ed03307716be5032d3c692f61a4982d18427937b1e5"
)
P1_ELEMENTS = 65793
RADIUS_PRIORITY = (5.0e-5, 2.5e-5)
RADIUS_LABELS = {5.0e-5: "R50", 2.5e-5: "R25"}
REFERENCE_E100_DELTA_L2 = 1.896e-4
BATCH_SIZE = 256
SEED = 2026081057
SEMANTICS = ("cpu_fp32", "cuda_bf16")
GENERAL_ACCURACY_DROP_MAX = 0.001
GROUP_HYBRID_DROP_MAX = 0.005

EXPECTED_FIT_GRADIENTS = {
    "anti_fit": {
        "float64_le_sha256": "a81f810c96c7fec227e93e9edf34db74305422098a30d715fb0e1dcc46bc62cf",
        "l2": 0.16034722588704536,
        "effective_weight_sum": 3010.0,
    },
    "fros_fit": {
        "float64_le_sha256": "b54886cde191a7086e6a3d3c7bdcd0298ce263304479f95b0a7380eaad888b8f",
        "l2": 0.16770721016217777,
        "effective_weight_sum": 9139.0,
    },
    "general_fit": {
        "float64_le_sha256": "d64e8bbcb1e685a29f74f501a2a2e17af85751a5ec0c05560867fbb5cac9e6fc",
        "l2": 0.11746854289792587,
        "effective_weight_sum": 157673.0,
    },
}

FIT_SPECS = tuple(spec for spec in probe.DOMAIN_SPECS if spec.role == "fit")
CAL_SPECS = tuple(spec for spec in probe.DOMAIN_SPECS if spec.role == "cal")
FINAL_SPECS = tuple(spec for spec in probe.DOMAIN_SPECS if spec.role == "final")
if tuple(spec.name for spec in FIT_SPECS) != probe.FIT_TARGETS:
    raise RuntimeError("Frozen fit spec order drift")
if tuple(spec.name for spec in CAL_SPECS) != probe.CAL_TARGETS:
    raise RuntimeError("Frozen calibration spec order drift")
if tuple(spec.name for spec in FINAL_SPECS) != probe.FINAL_TARGETS:
    raise RuntimeError("Frozen final spec order drift")


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


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def strict_json_bytes(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Invalid strict JSON in {label}") from error


def assert_output_absent(path: Path, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(f"{label} must be absent: {path}")


def write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    assert_output_absent(path, "exclusive output")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(all_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(child) for child in value)
    return True


def static_scope_audit(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source)
    dotted_calls: list[str] = []

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted_calls.append(dotted(node.func))
    leaf_calls = [name.rsplit(".", 1)[-1] for name in dotted_calls]
    forbidden = sorted(
        name
        for name in leaf_calls
        if name in {
            "backward",
            "step",
            "package",
            "upload",
            "submit",
            "submission",
        }
    )
    if forbidden:
        raise RuntimeError(f"Forbidden materializer call site: {forbidden}")
    if dotted_calls.count("torch.save") != 1:
        raise RuntimeError("Materializer requires exactly one torch.save call site")
    if any(name.startswith("torch.optim") for name in dotted_calls):
        raise RuntimeError("Materializer cannot instantiate an optimizer")
    return {
        "torch_save_call_sites_exactly_one": True,
        "no_backward_or_optimizer_step_call_site": True,
        "no_gameplay_h2h_package_upload_or_submission_call_site": True,
        "persistent_outputs": [
            str(PLAN_PATH),
            str(RESULT_PATH),
            str(CHECKPOINT_PATH),
        ],
    }


def load_geometry_evidence() -> dict[str, Any]:
    require_regular_file(
        GEOMETRY_PLAN, GEOMETRY_PLAN_FILE_SHA256, "frozen geometry plan"
    )
    require_regular_file(
        GEOMETRY_REPORT, GEOMETRY_REPORT_SHA256, "frozen geometry report"
    )
    plan_raw = GEOMETRY_PLAN.read_bytes()
    plan_envelope = strict_json_bytes(plan_raw, "frozen geometry plan")
    if (
        not isinstance(plan_envelope, dict)
        or not isinstance(plan_envelope.get("plan"), dict)
        or plan_raw != canonical_json_bytes(plan_envelope)
    ):
        raise RuntimeError("Frozen geometry plan envelope drift")
    inner_plan = plan_envelope["plan"]
    if (
        plan_envelope.get("plan_sha256") != GEOMETRY_PLAN_SHA256
        or sha256_json(inner_plan) != GEOMETRY_PLAN_SHA256
    ):
        raise RuntimeError("Frozen geometry inner plan SHA mismatch")

    report_raw = GEOMETRY_REPORT.read_bytes()
    report = strict_json_bytes(report_raw, "frozen geometry report")
    if not isinstance(report, dict) or report_raw != canonical_json_bytes(report):
        raise RuntimeError("Frozen geometry report canonical encoding drift")
    selection = report.get("candidate_selection")
    if not isinstance(selection, dict) or not isinstance(selection.get("reports"), dict):
        raise RuntimeError("Frozen geometry report lacks candidate selection")
    selected_report = selection["reports"].get(P1_NAME)
    if not isinstance(selected_report, dict):
        raise RuntimeError("Frozen geometry report lacks selected P1")
    direction = selected_report.get("direction")
    if (
        report.get("frozen_plan_sha256") != GEOMETRY_PLAN_SHA256
        or report.get("status") != "candidate_selected"
        or selection.get("selected") != P1_NAME
        or selection.get("selection_used_only_fit_and_cal") is not True
        or selection.get("final_views_used_for_selection") is not False
        or not isinstance(direction, dict)
        or direction.get("elements") != P1_ELEMENTS
        or direction.get("float64_le_sha256") != P1_VECTOR_SHA256
        or not math.isclose(float(direction.get("l2", float("nan"))), 1.0, abs_tol=1e-12)
        or selected_report.get("selection_eligible") is not True
    ):
        raise RuntimeError("Frozen geometry P1 selection drift")
    gradient_reports = report.get("gradient_reports")
    if not isinstance(gradient_reports, dict):
        raise RuntimeError("Frozen geometry report lacks gradients")
    fit_records: dict[str, Any] = {}
    for name, expected in EXPECTED_FIT_GRADIENTS.items():
        record = gradient_reports.get(name)
        natural = record.get("natural") if isinstance(record, dict) else None
        gradient = natural.get("gradient") if isinstance(natural, dict) else None
        if (
            not isinstance(gradient, dict)
            or gradient.get("float64_le_sha256")
            != expected["float64_le_sha256"]
            or gradient.get("elements") != P1_ELEMENTS
            or not math.isclose(
                float(gradient.get("l2", float("nan"))),
                float(expected["l2"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or natural.get("selection_effective_weight_sum")
            != expected["effective_weight_sum"]
        ):
            raise RuntimeError(f"Frozen geometry fit gradient drift for {name}")
        fit_records[name] = {
            "gradient": dict(gradient),
            "selection_effective_weight_sum": natural[
                "selection_effective_weight_sum"
            ],
        }
    return {
        "plan_file_sha256": GEOMETRY_PLAN_FILE_SHA256,
        "plan_sha256": GEOMETRY_PLAN_SHA256,
        "report_sha256": GEOMETRY_REPORT_SHA256,
        "selected_candidate": P1_NAME,
        "selected_direction": dict(direction),
        "fit_gradients": fit_records,
        "model_state_sha256": report.get("model_state_sha256_before"),
        "final_diagnostics_were_report_only": True,
    }


def snapshot_input_hashes(*, include_frozen_plan: bool) -> dict[str, str]:
    records = {
        name: file_sha256(path)
        for name, (path, _expected) in IMMUTABLE_INPUTS.items()
    }
    records["materializer"] = file_sha256(Path(__file__).resolve())
    if include_frozen_plan:
        records["materializer_frozen_plan"] = file_sha256(PLAN_PATH)
    return records


def build_plan() -> dict[str, Any]:
    assert_output_absent(RESULT_PATH, "materialization result")
    assert_output_absent(CHECKPOINT_PATH, "materialized checkpoint")
    inputs: dict[str, Any] = {}
    for name, (path, expected_sha) in IMMUTABLE_INPUTS.items():
        require_regular_file(path, expected_sha, name)
        inputs[name] = {"path": str(path), "sha256": expected_sha}
    tool_path = Path(__file__).resolve()
    tool_source = tool_path.read_bytes()
    geometry = load_geometry_evidence()
    domains = {
        spec.name: {
            "source": spec.source,
            "role": spec.role,
            "archive": str(spec.archive),
            "member_split": spec.member_split,
            "dates": list(spec.dates),
            **copy.deepcopy(probe.EXPECTED_DOMAIN_STATS[spec.name]),
        }
        for spec in FIT_SPECS + CAL_SPECS + FINAL_SPECS
    }
    plan = {
        "schema_version": SCHEMA_VERSION + "-plan",
        "purpose": "one unique tiny actor6 P1 BC endpoint from frozen S8",
        "inputs": inputs
        | {
            "materializer": {
                "path": str(tool_path),
                "sha256": sha256_bytes(tool_source),
            }
        },
        "geometry_lineage": geometry,
        "domains": domains,
        "runtime": {
            "python": str(EXPECTED_PYTHON),
            "torch_version": EXPECTED_TORCH_VERSION,
            "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
            "cudnn_version": EXPECTED_CUDNN_VERSION,
            "orjson_version": EXPECTED_ORJSON_VERSION,
            "isolated": True,
            "dont_write_bytecode": True,
            "CUBLAS_WORKSPACE_CONFIG": EXPECTED_CUBLAS,
            "CUDA_VISIBLE_DEVICES": EXPECTED_CUDA_VISIBLE_DEVICES,
            "deterministic_algorithms": True,
            "tf32": False,
            "batch_size": BATCH_SIZE,
            "forward_semantics": {
                "cpu_fp32": {
                    "device": "cpu",
                    "autocast": False,
                    "dtype": "float32",
                    "batch_size": BATCH_SIZE,
                    "meaning": (
                        "batched FP32 arithmetic proxy for the production "
                        "CPU/FP32 batch-one main.py path"
                    ),
                },
                "cuda_bf16": {
                    "device": "cuda:0",
                    "autocast": True,
                    "dtype": "bfloat16",
                    "batch_size": BATCH_SIZE,
                    "meaning": "strict local H2H forward arithmetic",
                },
            },
        },
        "direction_reproduction": {
            "sources": list(probe.FIT_TARGETS),
            "all_rows_once": True,
            "loss": "natural ordered selection (context34=1,fixed_multi=1)",
            "global_gradient_aggregation": (
                "sum(batch_gradient * selection_effective_weight_sum) / "
                "sum(selection_effective_weight_sum)"
            ),
            "P1_formula": "unit(0.5*g_anti+0.25*g_fros+0.25*g_general)",
            "required_source_gradient_records": copy.deepcopy(
                EXPECTED_FIT_GRADIENTS
            ),
            "required_P1_float64_le_sha256": P1_VECTOR_SHA256,
            "required_P1_elements": P1_ELEMENTS,
            "geometry_report_vector_is_hash_only_not_loaded": True,
        },
        "candidate_protocol": {
            "parameterization": "theta_candidate = theta_S8 - radius * unit(P1)",
            "trainable_scope": list(ACTOR6),
            "radius_priority": [
                {"label": RADIUS_LABELS[radius], "l2": radius}
                for radius in RADIUS_PRIORITY
            ],
            "reference_rejected_E100_delta_l2": REFERENCE_E100_DELTA_L2,
            "R50_fraction_of_reference_E100": RADIUS_PRIORITY[0]
            / REFERENCE_E100_DELTA_L2,
            "R25_fraction_of_reference_E100": RADIUS_PRIORITY[1]
            / REFERENCE_E100_DELTA_L2,
            "materialization_precision": (
                "parent and P1 combined in float64, then each actor tensor "
                "rounded once to the parent float32 dtype"
            ),
            "quantized_integrity": {
                "actual_delta_l2_relative_error_max": 0.05,
                "actual_descent_direction_cosine_min": 0.999,
                "changed_names_nonempty_subset_of_actor6": True,
                "all_nonactor_tensors_bit_identical": True,
                "count_head_tensors_bit_identical": True,
            },
        },
        "behavior_measurement": {
            "semantics_required": list(SEMANTICS),
            "same_run_S8_baseline": True,
            "loss_modes": {
                "natural": {"context34": 1.0, "fixed_multi": 1.0},
                "legacy_8_2": {"context34": 8.0, "fixed_multi": 2.0},
            },
            "global_selection_loss_aggregation": (
                "math.fsum(batch_selection_loss * "
                "batch_selection_effective_weight_sum) / "
                "math.fsum(batch_selection_effective_weight_sum)"
            ),
            "forbidden_row_reaverage": True,
            "accuracy_metrics": ["set_exact", "hybrid_order_exact", "ordered_exact"],
            "count_proofs": [
                "count_logits_float32_le_sha256",
                "action_count_prediction_sha256",
                "count_correct exact",
            ],
            "subgroups": ["team_name", "seat"],
        },
        "calibration_protocol": {
            "views": list(probe.CAL_TARGETS),
            "final_behavior_rows_not_iterated_before_selection": True,
            "final_archive_bytes_authenticated_by_SHA_before_selection": True,
            "preexisting_geometry_final_diagnostics_not_used_for_radius_selection": True,
            "sequential_priority": ["R50", "R25"],
            "R25_evaluated_only_if_R50_fails_complete_dual_semantics_gate": True,
            "selection_requires_both_forward_semantics": True,
            "per_semantics_per_view_gates": {
                "natural_global_selection_loss_decrease_min": 2e-7,
                "legacy_8_2_global_selection_loss_decrease_min": 2e-7,
                "rho_definition": (
                    "(S8_global_loss-candidate_global_loss) / "
                    "(radius * frozen_P1_gradient_dot)"
                ),
                "rho_min_each_loss_mode": 0.10,
                "anti_and_fros_set_hybrid_ordered_correct_counts": "non-decrease",
                "anti_and_fros_all_three_accuracies": "non-decrease",
                "general_all_three_accuracy_drop_max": GENERAL_ACCURACY_DROP_MAX,
                "each_team_and_seat_hybrid_drop_max": GROUP_HYBRID_DROP_MAX,
                "general_team_macro_hybrid": "non-decrease",
                "count_proofs": "bit/exact identical",
            },
            "selected_candidate_confirmation": {
                "complete_second_evaluation": True,
                "policy_action_count_and_count_logit_digests": "exact match",
                "selection_loss_absolute_difference_max": 1e-9,
                "required_before_radius_lock": True,
            },
        },
        "final_protocol": {
            "views": list(probe.FINAL_TARGETS),
            "opened_only_after_unique_radius_locked": True,
            "same_run_S8_and_unique_candidate_only": True,
            "both_forward_semantics_required": True,
            "gates_identical_to_calibration": True,
            "selected_candidate_confirmation": {
                "complete_second_evaluation": True,
                "policy_action_count_and_count_logit_digests": "exact match",
                "selection_loss_absolute_difference_max": 1e-9,
            },
            "failure_action": "reject route; never evaluate or fall back to other radius",
        },
        "scope_audit": static_scope_audit(tool_source),
        "outputs": {
            "frozen_plan": str(PLAN_PATH),
            "result": str(RESULT_PATH),
            "checkpoint": str(CHECKPOINT_PATH),
            "exclusive_create": True,
            "checkpoint_only_when_all_final_gates_pass": True,
            "checkpoint_count_max": 1,
        },
        "scope": {
            "optimizer": False,
            "backward": False,
            "gameplay": False,
            "H2H": False,
            "package": False,
            "upload": False,
            "submission": False,
            "local_only": True,
        },
    }
    return plan


def plan_envelope(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-frozen-plan-envelope",
        "plan_sha256": sha256_json(plan),
        "plan": plan,
    }


def load_frozen_plan(path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    if path.resolve() != PLAN_PATH.resolve():
        raise RuntimeError(f"Frozen plan path must be {PLAN_PATH}")
    try:
        info = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing frozen plan: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("Frozen plan must be a non-symlink regular file")
    raw = path.read_bytes()
    envelope = strict_json_bytes(raw, "frozen materializer plan")
    if (
        not isinstance(envelope, dict)
        or not isinstance(envelope.get("plan"), dict)
        or raw != canonical_json_bytes(envelope)
    ):
        raise RuntimeError("Frozen materializer plan envelope drift")
    plan = envelope["plan"]
    observed = sha256_json(plan)
    if envelope.get("plan_sha256") != observed or observed != expected_plan_sha256:
        raise RuntimeError("Frozen materializer plan SHA-256 mismatch")
    return plan


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def state_integrity(
    parent_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    radius: float,
) -> dict[str, Any]:
    if set(parent_state) != set(candidate_state):
        raise RuntimeError("Candidate state schema differs from S8")
    changed: list[str] = []
    nonactor_identical = True
    count_head_identical = True
    for name in parent_state:
        parent_tensor = parent_state[name]
        candidate_tensor = candidate_state[name]
        if (
            parent_tensor.shape != candidate_tensor.shape
            or parent_tensor.dtype != candidate_tensor.dtype
        ):
            raise RuntimeError(f"Candidate tensor schema drift at {name}")
        equal = torch.equal(parent_tensor, candidate_tensor)
        if not equal:
            changed.append(name)
        if name not in ACTOR6:
            nonactor_identical &= equal
        if name.startswith("count_head."):
            count_head_identical &= equal
    delta = probe.flatten_actor6(candidate_state) - probe.flatten_actor6(parent_state)
    delta_l2 = float(torch.linalg.vector_norm(delta))
    if not math.isfinite(delta_l2) or delta_l2 <= 0.0:
        raise RuntimeError("Candidate has zero/non-finite actor6 delta")
    actual_descent = probe.unit(-delta)
    desired_delta = -float(radius) * direction.detach().cpu().double()
    quantization_error = delta - desired_delta
    cosine = float(torch.dot(actual_descent, probe.unit(direction)))
    relative_l2_error = abs(delta_l2 - radius) / radius
    report = {
        "parent_state_sha256": state_dict_sha256(parent_state),
        "candidate_state_sha256": state_dict_sha256(candidate_state),
        "changed_parameter_names": sorted(changed),
        "changed_names_nonempty_subset_of_actor6": bool(changed)
        and set(changed).issubset(ACTOR6),
        "all_nonactor_tensors_bit_identical": nonactor_identical,
        "count_head_tensors_bit_identical": count_head_identical,
        "actual_delta": probe.vector_record(delta),
        "desired_unquantized_delta": probe.vector_record(desired_delta),
        "float32_quantization_error": probe.vector_record(quantization_error),
        "actual_delta_l2_relative_error": relative_l2_error,
        "actual_descent_direction_cosine_to_P1": cosine,
    }
    report["gates"] = {
        "changed_names_nonempty_subset_of_actor6": report[
            "changed_names_nonempty_subset_of_actor6"
        ],
        "all_nonactor_tensors_bit_identical": nonactor_identical,
        "count_head_tensors_bit_identical": count_head_identical,
        "actual_delta_l2_relative_error_at_most_0_05": relative_l2_error <= 0.05,
        "actual_descent_direction_cosine_at_least_0_999": cosine >= 0.999,
    }
    report["pass"] = all(report["gates"].values())
    return report


def materialize_candidate_state(
    parent_state: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    radius: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if radius not in RADIUS_PRIORITY:
        raise RuntimeError(f"Unregistered trust radius: {radius}")
    flat = direction.detach().cpu().double().reshape(-1)
    if flat.numel() != P1_ELEMENTS:
        raise RuntimeError("P1 direction element count drift")
    if probe.vector_sha256(flat) != P1_VECTOR_SHA256:
        raise RuntimeError("P1 direction hash drift before materialization")
    if not math.isclose(
        float(torch.linalg.vector_norm(flat)), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("P1 direction is not unit norm")
    candidate = clone_state(parent_state)
    offset = 0
    for name in ACTOR6:
        parent_tensor = parent_state[name].detach().cpu()
        elements = parent_tensor.numel()
        chunk = flat[offset : offset + elements].reshape(parent_tensor.shape)
        updated = (
            parent_tensor.double() - float(radius) * chunk
        ).to(dtype=parent_tensor.dtype)
        candidate[name] = updated.contiguous()
        offset += elements
    if offset != flat.numel():
        raise RuntimeError("P1 actor6 slicing did not consume the vector")
    integrity = state_integrity(parent_state, candidate, flat, radius)
    if not integrity["pass"]:
        raise RuntimeError(f"Quantized candidate integrity failed: {integrity['gates']}")
    return candidate, integrity


def apply_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Model state load drift: {incompatible}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("State application found stale gradient buffers")


def _subset_tensor_dict(
    values: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long, device=device)
    return {name: value.index_select(0, index) for name, value in values.items()}


def _metric_summary_without_context(
    accumulator: Any,
) -> dict[str, Any]:
    summary = accumulator.summary()
    summary.pop("by_context", None)
    return summary


class GlobalSelectionLoss:
    """Effective-weight aggregation for one complete domain."""

    def __init__(self) -> None:
        self.numerators: list[float] = []
        self.effective_weights: list[float] = []
        self.batches = 0

    def add(self, loss: float, effective_weight: float) -> None:
        if (
            not math.isfinite(loss)
            or not math.isfinite(effective_weight)
            or effective_weight <= 0.0
        ):
            raise RuntimeError("Invalid global selection-loss contribution")
        self.numerators.append(loss * effective_weight)
        self.effective_weights.append(effective_weight)
        self.batches += 1

    def finish(self) -> dict[str, Any]:
        denominator = math.fsum(self.effective_weights)
        numerator = math.fsum(self.numerators)
        if self.batches <= 0 or denominator <= 0.0:
            raise RuntimeError("No global selection-loss batches")
        value = numerator / denominator
        if not all(math.isfinite(item) for item in (numerator, denominator, value)):
            raise FloatingPointError("Non-finite global selection loss")
        return {
            "selection_loss": value,
            "weighted_numerator": numerator,
            "selection_effective_weight_sum": denominator,
            "batches": self.batches,
            "aggregation": (
                "fsum(batch_selection_loss * effective_weight_sum) / "
                "fsum(effective_weight_sum)"
            ),
        }


@torch.inference_mode()
def evaluate_behavior_domain(
    model: torch.nn.Module,
    spec: Any,
    model_config: Mapping[str, Any],
    device: torch.device,
    semantics: str,
) -> dict[str, Any]:
    expected_device = {
        "cpu_fp32": "cpu",
        "cuda_bf16": "cuda:0",
    }.get(semantics)
    if expected_device is None or str(device) != expected_device:
        raise RuntimeError(f"Behavior semantic/device mismatch: {semantics}/{device}")
    model.eval()
    total = behavior_core.MetricAccumulator()
    by_team: dict[str, Any] = defaultdict(behavior_core.MetricAccumulator)
    by_seat: dict[str, Any] = defaultdict(behavior_core.MetricAccumulator)
    losses = {
        "natural": GlobalSelectionLoss(),
        "legacy_8_2": GlobalSelectionLoss(),
    }
    count_logits_digest = hashlib.sha256()
    action_count_digest = hashlib.sha256()
    policy_action_digest = hashlib.sha256()
    row_key_digest = hashlib.sha256()
    rows = 0
    batch_rows: list[dict[str, Any]] = []

    def consume(raw_rows: list[dict[str, Any]]) -> None:
        nonlocal rows
        features = list(probe._featurize_rows(raw_rows, model_config))
        if len(features) != len(raw_rows):
            raise RuntimeError(f"Featurization count drift for {spec.name}")
        cpu_batch = probe._collate_features(features, model_config)
        batch = {
            key: value.to(device, non_blocking=False)
            for key, value in cpu_batch.items()
        }
        batch["action_sequences"] = batch["expert_ordered_actions"]
        outputs = ppo.model_forward(model, batch, device)
        if semantics == "cpu_fp32" and any(
            value.dtype != torch.float32
            for key, value in outputs.items()
            if key in {"policy_logits", "count_logits", "value_logits"}
        ):
            raise RuntimeError("CPU semantic emitted non-FP32 logits")
        policy_actions, _, _, _ = ppo.sample_ordered_actions(
            outputs,
            batch,
            deterministic=True,
            canonicalize_order=False,
        )
        total.update(batch, outputs, policy_actions, policy_actions)

        team_indices: dict[str, list[int]] = defaultdict(list)
        seat_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(raw_rows):
            team = row.get("team_name")
            seat = row.get("seat")
            if not isinstance(team, str) or not team:
                raise RuntimeError(f"Missing team_name in {spec.name}")
            if isinstance(seat, bool) or not isinstance(seat, int) or seat not in (0, 1):
                raise RuntimeError(f"Invalid seat in {spec.name}")
            team_indices[team].append(index)
            seat_indices[str(seat)].append(index)
            row_key_digest.update(
                canonical_json_bytes(
                    [
                        probe.row_date(row),
                        row.get("episode_id"),
                        seat,
                        team,
                        row.get("action_step_index"),
                    ]
                )
            )
        metric_outputs = {
            key: outputs[key]
            for key in ("policy_logits", "value_logits")
        }
        for group, indices in team_indices.items():
            sub_batch = _subset_tensor_dict(batch, indices, device)
            sub_outputs = _subset_tensor_dict(metric_outputs, indices, device)
            sub_actions = [policy_actions[index] for index in indices]
            by_team[group].update(
                sub_batch, sub_outputs, sub_actions, sub_actions
            )
        for group, indices in seat_indices.items():
            sub_batch = _subset_tensor_dict(batch, indices, device)
            sub_outputs = _subset_tensor_dict(metric_outputs, indices, device)
            sub_actions = [policy_actions[index] for index in indices]
            by_seat[group].update(
                sub_batch, sub_outputs, sub_actions, sub_actions
            )

        count_logits = (
            outputs["count_logits"].detach().cpu().float().contiguous().numpy()
        )
        count_logits_digest.update(
            count_logits.astype("<f4", copy=False).tobytes()
        )
        predicted_counts = torch.tensor(
            [len(action) for action in policy_actions], dtype=torch.int16
        )
        action_count_digest.update(
            predicted_counts.numpy().astype("<i2", copy=False).tobytes()
        )
        for action in policy_actions:
            policy_action_digest.update(canonical_json_bytes(action))

        for mode, weights in (
            ("natural", probe.NATURAL_WEIGHTS),
            ("legacy_8_2", probe.LEGACY_WEIGHTS),
        ):
            _total_loss, parts = ppo.bc_expert_actor_loss(
                outputs,
                batch,
                loss_mode="ordered",
                order_context_weight=weights[0],
                non_context34_fixed_multi_action_order_weight=weights[1],
            )
            losses[mode].add(
                float(parts["selection_loss"].detach().cpu()),
                float(
                    parts["selection_effective_weight_sum"].detach().cpu()
                ),
            )
        rows += len(raw_rows)

    for raw_row in probe.iter_raw_domain_rows(spec):
        batch_rows.append(raw_row)
        if len(batch_rows) == BATCH_SIZE:
            consume(batch_rows)
            batch_rows = []
    if batch_rows:
        consume(batch_rows)
    expected = probe.EXPECTED_DOMAIN_STATS[spec.name]
    if rows != expected["rows"] or total.total["rows"] != expected["rows"]:
        raise RuntimeError(f"Behavior rows drift for {spec.name}: {rows}")
    result = {
        "semantics": semantics,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "rows": rows,
        "row_key_sha256": row_key_digest.hexdigest(),
        "metrics": total.summary(),
        "by_team": {
            name: _metric_summary_without_context(accumulator)
            for name, accumulator in sorted(by_team.items())
        },
        "by_seat": {
            name: _metric_summary_without_context(accumulator)
            for name, accumulator in sorted(by_seat.items())
        },
        "losses": {name: accumulator.finish() for name, accumulator in losses.items()},
        "count_logits_float32_le_sha256": count_logits_digest.hexdigest(),
        "action_count_prediction_sha256": action_count_digest.hexdigest(),
        "policy_action_sha256": policy_action_digest.hexdigest(),
    }
    for mode in ("natural", "legacy_8_2"):
        expected_weight = float(
            load_geometry_report_cached()["gradient_reports"][spec.name][mode][
                "selection_effective_weight_sum"
            ]
        )
        observed_weight = result["losses"][mode][
            "selection_effective_weight_sum"
        ]
        if observed_weight != expected_weight:
            raise RuntimeError(
                f"Effective-weight drift for {spec.name}/{mode}: "
                f"{observed_weight} != {expected_weight}"
            )
    if not all_finite(result):
        raise FloatingPointError(f"Non-finite behavior result for {spec.name}")
    return result


_GEOMETRY_REPORT_CACHE: dict[str, Any] | None = None


def load_geometry_report_cached() -> dict[str, Any]:
    global _GEOMETRY_REPORT_CACHE
    if _GEOMETRY_REPORT_CACHE is None:
        require_regular_file(
            GEOMETRY_REPORT, GEOMETRY_REPORT_SHA256, "frozen geometry report"
        )
        value = strict_json_bytes(GEOMETRY_REPORT.read_bytes(), "geometry report")
        if not isinstance(value, dict):
            raise RuntimeError("Geometry report root drift")
        _GEOMETRY_REPORT_CACHE = value
    return _GEOMETRY_REPORT_CACHE


def frozen_p1_dot(target: str, mode: str) -> float:
    report = load_geometry_report_cached()
    if target in probe.FIT_TARGETS + probe.CAL_TARGETS:
        value = report["candidate_selection"]["reports"][P1_NAME][mode][target][
            "dot_gradient_direction"
        ]
    elif target in probe.FINAL_TARGETS:
        value = report["final_diagnostics_report_only"][P1_NAME][mode][target][
            "dot_gradient_direction"
        ]
    else:
        raise RuntimeError(f"Unknown frozen geometry target: {target}")
    dot = float(value)
    if not math.isfinite(dot) or dot <= 0.0:
        raise RuntimeError(f"Frozen P1 dot is not positive for {target}/{mode}")
    return dot


def _accuracy_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    key: str,
) -> float:
    base_value = baseline.get(key)
    candidate_value = candidate.get(key)
    if not isinstance(base_value, (int, float)) or not isinstance(
        candidate_value, (int, float)
    ):
        raise RuntimeError(f"Accuracy metric {key} is not numeric")
    value = float(candidate_value) - float(base_value)
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite accuracy delta at {key}")
    return value


def _group_hybrid_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    axis: str,
) -> dict[str, float]:
    base_groups = baseline.get(axis)
    candidate_groups = candidate.get(axis)
    if not isinstance(base_groups, Mapping) or not isinstance(
        candidate_groups, Mapping
    ):
        raise RuntimeError(f"Missing behavior subgroup axis: {axis}")
    if set(base_groups) != set(candidate_groups):
        raise RuntimeError(f"Behavior subgroup membership drift: {axis}")
    deltas: dict[str, float] = {}
    for name in sorted(base_groups):
        base = base_groups[name]
        cand = candidate_groups[name]
        if base.get("rows") != cand.get("rows"):
            raise RuntimeError(f"Behavior subgroup row drift: {axis}/{name}")
        deltas[name] = _accuracy_delta(
            base, cand, "hybrid_order_exact_accuracy"
        )
    return deltas


def compare_behavior_view(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    target: str,
    source: str,
    radius: float,
) -> dict[str, Any]:
    if baseline.get("semantics") != candidate.get("semantics"):
        raise RuntimeError("Baseline/candidate forward semantics drift")
    if baseline.get("rows") != candidate.get("rows"):
        raise RuntimeError("Baseline/candidate behavior row drift")
    metrics_base = baseline.get("metrics")
    metrics_candidate = candidate.get("metrics")
    if not isinstance(metrics_base, Mapping) or not isinstance(
        metrics_candidate, Mapping
    ):
        raise RuntimeError("Behavior comparison lacks metrics")

    losses: dict[str, Any] = {}
    loss_gates: dict[str, bool] = {}
    for mode in ("natural", "legacy_8_2"):
        base_loss = float(baseline["losses"][mode]["selection_loss"])
        candidate_loss = float(candidate["losses"][mode]["selection_loss"])
        decrease = base_loss - candidate_loss
        prediction = radius * frozen_p1_dot(target, mode)
        rho = decrease / prediction
        losses[mode] = {
            "baseline_global_selection_loss": base_loss,
            "candidate_global_selection_loss": candidate_loss,
            "actual_decrease": decrease,
            "predicted_first_order_decrease": prediction,
            "rho_actual_over_predicted": rho,
            "frozen_P1_gradient_dot": frozen_p1_dot(target, mode),
        }
        loss_gates[f"{mode}_decrease_at_least_2e_7"] = decrease >= 2e-7
        loss_gates[f"{mode}_rho_at_least_0_10"] = rho >= 0.10

    accuracy_keys = (
        "set_exact_accuracy",
        "hybrid_order_exact_accuracy",
        "ordered_exact_accuracy",
    )
    accuracy_deltas = {
        key: _accuracy_delta(metrics_base, metrics_candidate, key)
        for key in accuracy_keys
    }
    correct_keys = (
        "set_exact_correct",
        "hybrid_order_exact_correct",
        "ordered_exact_correct",
    )
    correct_deltas = {
        key: int(metrics_candidate[key]) - int(metrics_base[key])
        for key in correct_keys
    }
    allowed_drop = GENERAL_ACCURACY_DROP_MAX if source == "general" else 0.0
    accuracy_gates = {
        f"{key}_drop_within_source_limit": value >= -allowed_drop
        for key, value in accuracy_deltas.items()
    }
    correct_count_gates = {
        f"{key}_non_decrease_for_specialist": (
            source == "general" or value >= 0
        )
        for key, value in correct_deltas.items()
    }
    team_deltas = _group_hybrid_deltas(baseline, candidate, "by_team")
    seat_deltas = _group_hybrid_deltas(baseline, candidate, "by_seat")
    team_macro_base = math.fsum(
        float(record["hybrid_order_exact_accuracy"])
        for record in baseline["by_team"].values()
    ) / len(baseline["by_team"])
    team_macro_candidate = math.fsum(
        float(record["hybrid_order_exact_accuracy"])
        for record in candidate["by_team"].values()
    ) / len(candidate["by_team"])
    subgroup_gates = {
        "each_team_hybrid_drop_at_most_0_005": all(
            value >= -GROUP_HYBRID_DROP_MAX for value in team_deltas.values()
        ),
        "each_seat_hybrid_drop_at_most_0_005": all(
            value >= -GROUP_HYBRID_DROP_MAX for value in seat_deltas.values()
        ),
        "general_team_macro_hybrid_non_decrease": (
            source != "general" or team_macro_candidate >= team_macro_base
        ),
    }
    identity_gates = {
        "row_keys_identical": baseline.get("row_key_sha256")
        == candidate.get("row_key_sha256"),
        "count_logits_bit_identical": baseline.get(
            "count_logits_float32_le_sha256"
        )
        == candidate.get("count_logits_float32_le_sha256"),
        "action_count_predictions_identical": baseline.get(
            "action_count_prediction_sha256"
        )
        == candidate.get("action_count_prediction_sha256"),
        "count_correct_identical": metrics_base.get("count_correct")
        == metrics_candidate.get("count_correct"),
    }
    gates = (
        loss_gates
        | accuracy_gates
        | correct_count_gates
        | subgroup_gates
        | identity_gates
    )
    report = {
        "target": target,
        "source": source,
        "semantics": baseline["semantics"],
        "radius": radius,
        "losses": losses,
        "accuracy_deltas": accuracy_deltas,
        "correct_count_deltas": correct_deltas,
        "allowed_accuracy_drop": allowed_drop,
        "team_hybrid_deltas": team_deltas,
        "seat_hybrid_deltas": seat_deltas,
        "team_macro_hybrid": {
            "baseline": team_macro_base,
            "candidate": team_macro_candidate,
            "delta": team_macro_candidate - team_macro_base,
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    if not all_finite(report):
        raise FloatingPointError("Non-finite behavior gate report")
    return report


def compare_behavior_bundle(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Any],
    radius: float,
) -> dict[str, Any]:
    expected_names = {spec.name for spec in specs}
    if set(baseline) != set(SEMANTICS) or set(candidate) != set(SEMANTICS):
        raise RuntimeError("Dual behavior semantics are incomplete")
    reports: dict[str, Any] = {}
    for semantics in SEMANTICS:
        if (
            set(baseline[semantics]) != expected_names
            or set(candidate[semantics]) != expected_names
        ):
            raise RuntimeError("Behavior bundle domain drift")
        reports[semantics] = {
            spec.name: compare_behavior_view(
                baseline[semantics][spec.name],
                candidate[semantics][spec.name],
                target=spec.name,
                source=spec.source,
                radius=radius,
            )
            for spec in specs
        }
    gates = {
        f"{semantics}__{spec.name}": reports[semantics][spec.name]["pass"]
        for semantics in SEMANTICS
        for spec in specs
    }
    loss_item_gates = {
        f"{semantics}__{spec.name}__{mode}": all(
            reports[semantics][spec.name]["gates"][key]
            for key in (
                f"{mode}_decrease_at_least_2e_7",
                f"{mode}_rho_at_least_0_10",
            )
        )
        for semantics in SEMANTICS
        for spec in specs
        for mode in ("natural", "legacy_8_2")
    }
    if len(loss_item_gates) != 12:
        raise RuntimeError("Dual-semantics loss gate count must be exactly 12")
    return {
        "views": reports,
        "view_gates": gates,
        "loss_item_gates": loss_item_gates,
        "loss_item_gate_count": len(loss_item_gates),
        "pass": all(gates.values()) and all(loss_item_gates.values()),
    }


def confirm_behavior_bundle(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Any],
) -> dict[str, Any]:
    expected_names = {spec.name for spec in specs}
    if set(first) != set(SEMANTICS) or set(second) != set(SEMANTICS):
        raise RuntimeError("Confirmation semantics are incomplete")
    reports: dict[str, Any] = {}
    for semantics in SEMANTICS:
        if set(first[semantics]) != expected_names or set(
            second[semantics]
        ) != expected_names:
            raise RuntimeError("Confirmation domain drift")
        reports[semantics] = {}
        for spec in specs:
            left = first[semantics][spec.name]
            right = second[semantics][spec.name]
            loss_differences = {
                mode: abs(
                    float(left["losses"][mode]["selection_loss"])
                    - float(right["losses"][mode]["selection_loss"])
                )
                for mode in ("natural", "legacy_8_2")
            }
            gates = {
                "policy_action_digest_identical": left.get("policy_action_sha256")
                == right.get("policy_action_sha256"),
                "action_count_digest_identical": left.get(
                    "action_count_prediction_sha256"
                )
                == right.get("action_count_prediction_sha256"),
                "count_logit_digest_identical": left.get(
                    "count_logits_float32_le_sha256"
                )
                == right.get("count_logits_float32_le_sha256"),
                "metrics_exactly_identical": left.get("metrics")
                == right.get("metrics"),
                "subgroups_exactly_identical": left.get("by_team")
                == right.get("by_team")
                and left.get("by_seat") == right.get("by_seat"),
                "natural_loss_difference_at_most_1e_9": loss_differences[
                    "natural"
                ]
                <= 1e-9,
                "legacy_loss_difference_at_most_1e_9": loss_differences[
                    "legacy_8_2"
                ]
                <= 1e-9,
            }
            reports[semantics][spec.name] = {
                "loss_absolute_differences": loss_differences,
                "gates": gates,
                "pass": all(gates.values()),
            }
    flat = {
        f"{semantics}__{spec.name}": reports[semantics][spec.name]["pass"]
        for semantics in SEMANTICS
        for spec in specs
    }
    return {"views": reports, "gates": flat, "pass": all(flat.values())}


def evaluate_bundle(
    models: Mapping[str, torch.nn.Module],
    devices: Mapping[str, torch.device],
    state: Mapping[str, torch.Tensor],
    model_config: Mapping[str, Any],
    specs: Sequence[Any],
) -> dict[str, dict[str, Any]]:
    if set(models) != set(SEMANTICS) or set(devices) != set(SEMANTICS):
        raise RuntimeError("Behavior models/devices lack dual semantics")
    result: dict[str, dict[str, Any]] = {}
    expected_state_sha = state_dict_sha256(state)
    for semantics in SEMANTICS:
        model = models[semantics]
        apply_state(model, state)
        observed_state = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
        if state_dict_sha256(observed_state) != expected_state_sha:
            raise RuntimeError(f"Applied state hash drift under {semantics}")
        result[semantics] = {
            spec.name: evaluate_behavior_domain(
                model,
                spec,
                model_config,
                devices[semantics],
                semantics,
            )
            for spec in specs
        }
        after_state = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
        if state_dict_sha256(after_state) != expected_state_sha:
            raise RuntimeError(f"Behavior evaluation changed model under {semantics}")
    return result


def select_radius_from_records(
    records: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if "R50" not in records:
        raise RuntimeError("R50 must be evaluated first")
    if records["R50"].get("complete_calibration_pass") is True:
        if "R25" in records:
            raise RuntimeError("R25 cannot be evaluated after R50 passes")
        return "R50"
    if "R25" not in records:
        raise RuntimeError("R25 is required only after R50 fails")
    if records["R25"].get("complete_calibration_pass") is True:
        return "R25"
    return None


def recompute_p1_direction(
    parent: Mapping[str, Any],
    bc_checkpoint: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], torch.nn.Module]:
    model = ppo.instantiate_model_from_checkpoint(parent, bc_checkpoint, device)
    parameters = training_core.configure_actor6(model)
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if trainable_names != ACTOR6 or len(parameters) != len(ACTOR6):
        raise RuntimeError("P1 recomputation trainable scope differs from actor6")
    model.eval()
    state_before = ppo.model_state_sha256(model)
    natural_fit: dict[str, torch.Tensor] = {}
    domain_reports: dict[str, Any] = {}
    for spec in FIT_SPECS:
        gradients, report = probe.gradient_for_domain(
            model,
            parameters,
            spec,
            parent["model_config"],
            device,
        )
        gradient = gradients["natural"]
        record = probe.vector_record(gradient)
        expected = EXPECTED_FIT_GRADIENTS[spec.name]
        checks = {
            "gradient_hash_exact": record["float64_le_sha256"]
            == expected["float64_le_sha256"],
            "gradient_l2_exact_within_1e_15": math.isclose(
                record["l2"], expected["l2"], rel_tol=0.0, abs_tol=1e-15
            ),
            "effective_weight_sum_exact": report["natural"][
                "selection_effective_weight_sum"
            ]
            == expected["effective_weight_sum"],
            "all_rows_covered_once": report["feature_checks"][
                "all_rows_covered_once"
            ]
            is True,
            "count_loss_actor6_independent": report[
                "count_actor6_dependency"
            ]["all_parameters_unused_or_exact_zero"]
            is True,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Recomputed fit gradient mismatch for {spec.name}: {checks}"
            )
        natural_fit[spec.name] = gradient
        domain_reports[spec.name] = {
            "checks": checks,
            "gradient": record,
            "selection_effective_weight_sum": report["natural"][
                "selection_effective_weight_sum"
            ],
            "rows": report["rows"],
            "batches": report["batches"],
            "count_actor6_dependency": report["count_actor6_dependency"],
        }
    candidates, construction = probe.construct_candidates(natural_fit)
    direction = candidates[P1_NAME].detach().cpu().double()
    direction_record = probe.vector_record(direction)
    if (
        direction_record["float64_le_sha256"] != P1_VECTOR_SHA256
        or direction_record["elements"] != P1_ELEMENTS
        or not math.isclose(direction_record["l2"], 1.0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Recomputed P1 direction mismatch: {direction_record}")
    state_after = ppo.model_state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("P1 recomputation changed model state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("P1 recomputation left gradient buffers")
    report = {
        "fit_domains": domain_reports,
        "construction": construction,
        "direction": direction_record,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_bit_identical": True,
        "optimizer_used": False,
        "backward_used": False,
    }
    return direction, report, model


CHECKPOINT_RETAIN_KEYS = (
    "feature_version",
    "bc_feature_version",
    "config",
    "model_config",
    "learner_deck_hash",
    "reward",
    "value_trunk_gradient",
    "actor_value_gradient",
    "action_distribution",
    "post_ppo_special_bc",
)
FORBIDDEN_CHECKPOINT_KEYS = {
    "optimizer_state_dict",
    "bc_replay_optimizer_state_dict",
    "opponent_quota_state",
    "fresh_special_optimizer_state_dict",
}


def build_checkpoint(
    parent: Mapping[str, Any],
    candidate_state: Mapping[str, torch.Tensor],
    *,
    frozen_plan_sha256: str,
    radius_label: str,
    radius: float,
    integrity: Mapping[str, Any],
    calibration_record: Mapping[str, Any],
    final_record: Mapping[str, Any],
) -> dict[str, Any]:
    retained = {
        key: copy.deepcopy(parent[key])
        for key in CHECKPOINT_RETAIN_KEYS
        if key in parent
    }
    retained.update(
        {
            "model_state_dict": clone_state(candidate_state),
            "update": 0,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_states_omitted": sorted(FORBIDDEN_CHECKPOINT_KEYS),
            "s08_p1_tiny_trust_bc": {
                "schema_version": SCHEMA_VERSION,
                "frozen_plan_sha256": frozen_plan_sha256,
                "parent": "S8",
                "parent_sha256": IMMUTABLE_INPUTS["s8_parent"][1],
                "geometry_plan_sha256": GEOMETRY_PLAN_SHA256,
                "geometry_report_sha256": GEOMETRY_REPORT_SHA256,
                "P1_float64_le_sha256": P1_VECTOR_SHA256,
                "radius_label": radius_label,
                "radius_l2": radius,
                "trainable_scope": list(ACTOR6),
                "integrity": copy.deepcopy(dict(integrity)),
                "calibration_record_canonical_sha256": sha256_json(
                    calibration_record
                ),
                "final_record_canonical_sha256": sha256_json(final_record),
                "dual_forward_semantics_pass": True,
                "selected_calibration_confirmed_twice": True,
                "selected_final_confirmed_twice": True,
                "H2H_eligible": True,
                "H2H_not_performed": True,
                "promotion_eligible": False,
                "package_not_performed": True,
                "upload_not_performed": True,
                "submission_authorized": False,
                "submission_not_performed": True,
            },
        }
    )
    if FORBIDDEN_CHECKPOINT_KEYS.intersection(retained):
        raise RuntimeError("Materialized checkpoint retained optimizer state")
    if set(retained["model_state_dict"]) != set(parent["model_state_dict"]):
        raise RuntimeError("Materialized checkpoint state schema drift")
    return retained


def serialize_checkpoint(checkpoint: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(checkpoint), stream)
    return stream.getvalue()


def verify_checkpoint_payload(
    payload: bytes,
    expected_state: Mapping[str, torch.Tensor],
    expected_plan_sha256: str,
    expected_radius_label: str,
) -> dict[str, Any]:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Serialized materialized checkpoint root drift")
    metadata = checkpoint.get("s08_p1_tiny_trust_bc")
    state = checkpoint.get("model_state_dict")
    if (
        not isinstance(metadata, dict)
        or not isinstance(state, dict)
        or metadata.get("frozen_plan_sha256") != expected_plan_sha256
        or metadata.get("radius_label") != expected_radius_label
        or metadata.get("H2H_not_performed") is not True
        or metadata.get("submission_not_performed") is not True
        or checkpoint.get("evaluation_only") is not True
        or checkpoint.get("resume_forbidden") is not True
        or FORBIDDEN_CHECKPOINT_KEYS.intersection(checkpoint)
    ):
        raise RuntimeError("Serialized materialized checkpoint metadata drift")
    observed_state_sha = state_dict_sha256(state)
    expected_state_sha = state_dict_sha256(expected_state)
    if observed_state_sha != expected_state_sha:
        raise RuntimeError("Serialized materialized checkpoint state drift")
    return {
        "payload_sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "model_state_sha256": observed_state_sha,
        "metadata_canonical_sha256": sha256_json(metadata),
    }


def make_result_base(
    plan: Mapping[str, Any],
    initial_hashes: Mapping[str, str],
    direction_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION + "-result",
        "frozen_plan_sha256": sha256_json(plan),
        "geometry_plan_sha256": GEOMETRY_PLAN_SHA256,
        "geometry_report_sha256": GEOMETRY_REPORT_SHA256,
        "direction_reproduction": copy.deepcopy(dict(direction_report)),
        "initial_input_hashes": dict(initial_hashes),
        "radius_priority": [RADIUS_LABELS[value] for value in RADIUS_PRIORITY],
        "calibration": {},
        "selected_radius_label": None,
        "selected_radius": None,
        "final": None,
        "checkpoint": None,
        "H2H_performed": False,
        "gameplay_performed": False,
        "package_performed": False,
        "upload_performed": False,
        "submission_performed": False,
    }


def publish_result(result: dict[str, Any], initial_hashes: Mapping[str, str]) -> None:
    final_hashes = snapshot_input_hashes(include_frozen_plan=True)
    if dict(initial_hashes) != final_hashes:
        raise RuntimeError("A frozen materializer input changed during execution")
    result["final_input_hashes"] = final_hashes
    result["inputs_unchanged_after_execution"] = True
    result["persistent_result"] = str(RESULT_PATH)
    if not all_finite(result):
        raise FloatingPointError("Materializer result contains non-finite values")
    write_exclusive(RESULT_PATH, canonical_json_bytes(result))


def execute(plan: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    assert_output_absent(RESULT_PATH, "materialization result")
    assert_output_absent(CHECKPOINT_PATH, "materialized checkpoint")
    rebuilt = build_plan()
    if plan != rebuilt:
        raise RuntimeError("Frozen materializer plan no longer matches rebuilt plan")
    frozen_plan_sha256 = sha256_json(plan)
    initial_hashes = snapshot_input_hashes(include_frozen_plan=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal P1 materialization requires cuda:0")
    parent = torch.load(S8_PARENT, map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(
        BC_ARCHITECTURE, map_location="cpu", weights_only=False
    )
    if not isinstance(parent, dict) or not isinstance(bc_checkpoint, dict):
        raise RuntimeError("Bound checkpoint roots must be dictionaries")
    parent_state_raw = parent.get("model_state_dict")
    if not isinstance(parent_state_raw, dict):
        raise RuntimeError("S8 lacks model_state_dict")
    parent_state = clone_state(parent_state_raw)
    direction, direction_report, cuda_model = recompute_p1_direction(
        parent, bc_checkpoint, cuda_device
    )
    cpu_model = ppo.instantiate_model_from_checkpoint(
        parent, bc_checkpoint, cpu_device
    )
    models = {"cpu_fp32": cpu_model, "cuda_bf16": cuda_model}
    devices = {"cpu_fp32": cpu_device, "cuda_bf16": cuda_device}
    result = make_result_base(plan, initial_hashes, direction_report)
    result["parent_model_state_sha256"] = state_dict_sha256(parent_state)

    # Calibration is the sole source of the radius decision.  The final
    # archives have not been iterated before this block completes.
    calibration_baseline = evaluate_bundle(
        models,
        devices,
        parent_state,
        parent["model_config"],
        CAL_SPECS,
    )
    result["calibration"]["S8_baseline"] = calibration_baseline
    candidate_states: dict[str, dict[str, torch.Tensor]] = {}
    candidate_integrity: dict[str, dict[str, Any]] = {}
    radius_records: dict[str, dict[str, Any]] = {}

    for radius in RADIUS_PRIORITY:
        label = RADIUS_LABELS[radius]
        if label == "R25" and radius_records["R50"][
            "complete_calibration_pass"
        ]:
            raise RuntimeError("R25 evaluation attempted after R50 passed")
        state, integrity = materialize_candidate_state(
            parent_state, direction, radius
        )
        candidate_states[label] = state
        candidate_integrity[label] = integrity
        first = evaluate_bundle(
            models,
            devices,
            state,
            parent["model_config"],
            CAL_SPECS,
        )
        gate = compare_behavior_bundle(
            calibration_baseline, first, CAL_SPECS, radius
        )
        confirmation: dict[str, Any]
        second: dict[str, Any] | None = None
        if gate["pass"]:
            second = evaluate_bundle(
                models,
                devices,
                state,
                parent["model_config"],
                CAL_SPECS,
            )
            confirmation = confirm_behavior_bundle(first, second, CAL_SPECS)
        else:
            confirmation = {
                "performed": False,
                "reason": "first calibration gate failed",
                "pass": False,
            }
        complete_pass = bool(
            integrity["pass"] and gate["pass"] and confirmation["pass"]
        )
        radius_records[label] = {
            "radius": radius,
            "integrity": integrity,
            "first_evaluation": first,
            "first_gate": gate,
            "confirmation_evaluation": second,
            "confirmation": confirmation,
            "complete_calibration_pass": complete_pass,
        }
        result["calibration"][label] = radius_records[label]
        if complete_pass:
            break

    selected_label = select_radius_from_records(radius_records)
    if selected_label is None:
        result.update(
            {
                "status": "calibration_rejected",
                "selected_radius_label": None,
                "selected_radius": None,
                "final": {
                    "opened": False,
                    "reason": "no radius passed complete dual-semantics calibration",
                },
                "checkpoint_written": False,
            }
        )
        assert_output_absent(CHECKPOINT_PATH, "rejected checkpoint")
        publish_result(result, initial_hashes)
        return 42, result

    selected_radius = next(
        radius for radius in RADIUS_PRIORITY if RADIUS_LABELS[radius] == selected_label
    )
    selected_state = candidate_states[selected_label]
    result["selected_radius_label"] = selected_label
    result["selected_radius"] = selected_radius
    result["calibration_selection"] = {
        "selected": selected_label,
        "priority": [RADIUS_LABELS[value] for value in RADIUS_PRIORITY],
        "R25_evaluated": "R25" in radius_records,
        "selected_before_final_opened": True,
        "final_used_for_selection": False,
    }

    # The unique radius is now immutable.  A final failure cannot trigger a
    # different candidate or another radius.
    final_baseline = evaluate_bundle(
        models,
        devices,
        parent_state,
        parent["model_config"],
        FINAL_SPECS,
    )
    final_first = evaluate_bundle(
        models,
        devices,
        selected_state,
        parent["model_config"],
        FINAL_SPECS,
    )
    final_gate = compare_behavior_bundle(
        final_baseline, final_first, FINAL_SPECS, selected_radius
    )
    final_second: dict[str, Any] | None = None
    if final_gate["pass"]:
        final_second = evaluate_bundle(
            models,
            devices,
            selected_state,
            parent["model_config"],
            FINAL_SPECS,
        )
        final_confirmation = confirm_behavior_bundle(
            final_first, final_second, FINAL_SPECS
        )
    else:
        final_confirmation = {
            "performed": False,
            "reason": "first final gate failed",
            "pass": False,
        }
    final_pass = bool(final_gate["pass"] and final_confirmation["pass"])
    final_record = {
        "opened": True,
        "opened_after_radius_lock": True,
        "selected_radius_label": selected_label,
        "S8_baseline": final_baseline,
        "first_evaluation": final_first,
        "first_gate": final_gate,
        "confirmation_evaluation": final_second,
        "confirmation": final_confirmation,
        "pass": final_pass,
        "fallback_allowed": False,
    }
    result["final"] = final_record
    if not final_pass:
        result.update(
            {
                "status": "final_rejected",
                "checkpoint_written": False,
                "final_failure_did_not_trigger_radius_fallback": True,
            }
        )
        assert_output_absent(CHECKPOINT_PATH, "rejected checkpoint")
        publish_result(result, initial_hashes)
        return 42, result

    selected_calibration_record = radius_records[selected_label]
    checkpoint = build_checkpoint(
        parent,
        selected_state,
        frozen_plan_sha256=frozen_plan_sha256,
        radius_label=selected_label,
        radius=selected_radius,
        integrity=candidate_integrity[selected_label],
        calibration_record=selected_calibration_record,
        final_record=final_record,
    )
    checkpoint_payload = serialize_checkpoint(checkpoint)
    checkpoint_audit = verify_checkpoint_payload(
        checkpoint_payload,
        selected_state,
        frozen_plan_sha256,
        selected_label,
    )
    write_exclusive(CHECKPOINT_PATH, checkpoint_payload)
    if file_sha256(CHECKPOINT_PATH) != checkpoint_audit["payload_sha256"]:
        raise RuntimeError("Published checkpoint payload hash drift")
    result.update(
        {
            "status": "candidate_materialized",
            "checkpoint_written": True,
            "checkpoint": {
                "path": str(CHECKPOINT_PATH),
                **checkpoint_audit,
            },
            "H2H_eligible": True,
            "promotion_eligible": False,
        }
    )
    publish_result(result, initial_hashes)
    return 0, result


def enforce_runtime(*, formal_cuda: bool) -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run from repository root: {ROOT}")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"Wrong Python runtime: {sys.executable}")
    if Path(sys.prefix).resolve() != EXPECTED_ENV_PREFIX.resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("Run with my_project_env Python flags -I -B")
    observed_versions = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "orjson": probe.orjson.__version__,
    }
    expected_versions = {
        "torch": EXPECTED_TORCH_VERSION,
        "torch_cuda": EXPECTED_TORCH_CUDA_VERSION,
        "cudnn": EXPECTED_CUDNN_VERSION,
        "orjson": EXPECTED_ORJSON_VERSION,
    }
    if observed_versions != expected_versions:
        raise RuntimeError(
            f"Runtime library version drift: expected={expected_versions}, "
            f"observed={observed_versions}"
        )
    if formal_cuda:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != EXPECTED_CUBLAS:
            raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {EXPECTED_CUBLAS}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_CUDA_VISIBLE_DEVICES:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES must be {EXPECTED_CUDA_VISIBLE_DEVICES}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--print-plan", action="store_true")
    actions.add_argument("--freeze-plan", action="store_true")
    actions.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--expected-plan-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enforce_runtime(formal_cuda=args.execute)
    if args.print_plan:
        plan = build_plan()
        print(canonical_json_bytes(plan).decode("utf-8"), end="")
        return 0
    if args.freeze_plan:
        if args.plan.resolve() != PLAN_PATH.resolve():
            raise RuntimeError(f"Frozen plan path must be {PLAN_PATH}")
        assert_output_absent(PLAN_PATH, "frozen materializer plan")
        plan = build_plan()
        envelope = plan_envelope(plan)
        write_exclusive(PLAN_PATH, canonical_json_bytes(envelope))
        print(
            json.dumps(
                {
                    "path": str(PLAN_PATH),
                    "plan_sha256": envelope["plan_sha256"],
                    "file_sha256": file_sha256(PLAN_PATH),
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.expected_plan_sha256:
        raise RuntimeError("--execute requires --expected-plan-sha256")
    plan = load_frozen_plan(args.plan, args.expected_plan_sha256)
    code, result = execute(plan)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_radius_label": result.get("selected_radius_label"),
                "checkpoint_written": result.get("checkpoint_written"),
                "result": str(RESULT_PATH),
            },
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
