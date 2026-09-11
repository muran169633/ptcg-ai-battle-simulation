#!/usr/bin/env python3
"""Contract-only full-train promotion gate for the equal-blend endpoint.

This draft is deliberately unarmed.  It has no ``formal`` or ``actual`` CLI
mode and never instantiates/evaluates the candidate.  Its purpose is to freeze
the train-only data, metric, checkpoint-format, and promotion contracts before
the sole equal-blend endpoint is serialized.  Candidate path/file/model and
generation-manifest SHA-256 values remain explicit ``None`` placeholders.

The eventual armed revision may reuse the hash-bound evaluator helpers named
by :func:`promotion_contract`, but must preserve the contract byte-for-byte,
bind the completed endpoint before any evaluation, and run raw plus the sole
endpoint on every train row before making one promotion decision.  It must not
open a validation member or read any validation result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate.py"

HELPER = TOOLS / "run_u468_raw_balanced_train_only_endpoint_gate.py"
HELPER_SHA256 = "84b51d8ef51e2184271c4bf358e5d59b74ea631daa79b84d73fffaf44a37037d"

PARENT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
PARENT_FILE_SHA256 = "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
PARENT_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)
PROFILE = ROOT / "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json"
PROFILE_SHA256 = "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9"

SHADOW_RUNNER = TOOLS / "run_u468_raw_actor6_equalblend_sgd512_shadow.py"
SHADOW_RUNNER_SHA256 = (
    "e04f7b7579ef42d0c6f643db833779fb837ae86e3205d948b5564d08f8b30f8e"
)
SHADOW_PREREGISTRATION = ROOT / (
    "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_shadow_design202608115."
    "preregistration.json"
)
SHADOW_PREREGISTRATION_SHA256 = (
    "961ed51abcc5e502f603e5fc73385c64f9bc7fd0408154825920ed48002cb6da"
)
SHADOW_RESULT = ROOT / (
    "artifacts/ppo_u468_raw_actor6_equalblend_sgd512_shadow_design202608115."
    "result.json"
)
SHADOW_RESULT_SHA256 = (
    "a35d2050e92a9f0035847d3438241555573c053c4f99c0c4269a39bcd80bfe45"
)

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
DEPENDENCY_SHA256 = {
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
EXPECTED_TRAIN_SHAPE = {
    "flg": {
        "rows": 9443,
        "nonempty_rows": 9426,
        "context34_rows": 42,
        "context34_nonempty_rows": 42,
        "profile_set_exact_correct": 7239,
        "profile_ordered_exact_correct": 7095,
        "train_member_count": 1,
    },
    "pokemonfan": {
        "rows": 9487,
        "nonempty_rows": 9450,
        "context34_rows": 38,
        "context34_nonempty_rows": 38,
        "profile_set_exact_correct": 8282,
        "profile_ordered_exact_correct": 8203,
        "train_member_count": 1,
    },
    "core5": {
        "rows": 5120,
        "nonempty_rows": 5106,
        "context34_rows": 20,
        "context34_nonempty_rows": 20,
        "profile_set_exact_correct": 4073,
        "profile_ordered_exact_correct": 4025,
        "train_member_count": 20,
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
STRUCTURAL_METRICS = (
    "rows",
    "nonempty_rows",
    "by_context.34.rows",
    "by_context.34.nonempty_rows",
)
IMMUTABLE_METRICS = ("count_correct", "value_correct")
MAIN_QUALITY_METRICS = (
    "set_exact_correct",
    "hybrid_order_exact_correct",
    "ordered_exact_correct",
    "top1_correct",
)
CONTEXT34_SAFETY_METRICS = (
    "by_context.34.hybrid_order_exact_correct",
    "by_context.34.ordered_exact_correct",
)
NON_REGRESSION_METRICS = MAIN_QUALITY_METRICS + CONTEXT34_SAFETY_METRICS

BATCH_SIZE = 256
WORKERS = 0
MAX_ROWS = None
DEVICE = "cuda"

# These four values must be frozen together after the sole endpoint is
# serialized and its completed generation manifest independently audited.
# Keeping them None makes this source mechanically incapable of formal use.
ENDPOINT_PATH: Path | None = None
ENDPOINT_FILE_SHA256: str | None = None
ENDPOINT_MODEL_STATE_SHA256: str | None = None
ENDPOINT_GENERATION_MANIFEST_PATH: Path | None = None
ENDPOINT_GENERATION_MANIFEST_SHA256: str | None = None

SCHEMA = "ptcg-u468-raw-equalblend-train-only-promotion-gate-contract-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single_link_regular(
    path: Path, expected_sha256: str, label: str
) -> dict[str, Any]:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
    ):
        raise RuntimeError(f"{label} is not a single-link regular file")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 drift: {digest}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "nlink": observed.st_nlink,
    }


# Authenticate the reusable evaluator before importing any of its code.
HELPER_EVIDENCE = require_single_link_regular(
    HELPER, HELPER_SHA256, "frozen full-train evaluator helper"
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "ptcg_frozen_train_only_gate_helper_84b51d8e", HELPER
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError("cannot construct frozen train-only helper import spec")
helper: ModuleType = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(helper)


def validate_runtime() -> None:
    if Path.cwd().resolve(strict=True) != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve(strict=True) != EXPECTED_PYTHON.resolve(strict=True):
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B flags")


def metric_value(metrics: Mapping[str, Any], path: str) -> int:
    value: Any = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"missing metric path {path}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"metric {path} must be an integer count")
    return value


def gate_panel(
    panel: str,
    raw: Mapping[str, Any],
    endpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if panel not in DATASETS:
        raise ValueError(f"unknown panel: {panel}")
    checks: dict[str, Any] = {}
    passed = True
    for family, metrics, predicate in (
        ("structural_exact", STRUCTURAL_METRICS, lambda new, old: new == old),
        ("immutable_exact", IMMUTABLE_METRICS, lambda new, old: new == old),
        ("quality_non_regression", NON_REGRESSION_METRICS, lambda new, old: new >= old),
    ):
        for metric in metrics:
            baseline = metric_value(raw, metric)
            candidate = metric_value(endpoint, metric)
            item_pass = bool(predicate(candidate, baseline))
            passed &= item_pass
            checks[f"{family}:{metric}"] = {
                "raw": baseline,
                "endpoint": candidate,
                "delta": candidate - baseline,
                "pass": item_pass,
            }

    main_improvements = [
        metric
        for metric in MAIN_QUALITY_METRICS
        if metric_value(endpoint, metric) >= metric_value(raw, metric) + 1
    ]
    pokemonfan_improvement_required = panel == "pokemonfan"
    improvement_pass = (
        not pokemonfan_improvement_required or bool(main_improvements)
    )
    passed &= improvement_pass
    checks["pokemonfan_at_least_one_main_quality_count_plus_one"] = {
        "required": pokemonfan_improvement_required,
        "eligible_metrics": list(MAIN_QUALITY_METRICS),
        "improved_metrics": main_improvements,
        "context34_safety_metrics_cannot_satisfy_improvement": True,
        "count_and_value_cannot_satisfy_improvement": True,
        "pass": improvement_pass,
    }
    return {"panel": panel, "pass": bool(passed), "checks": checks}


def synthetic_gate_self_test() -> dict[str, Any]:
    raw = {
        "rows": 100,
        "nonempty_rows": 99,
        "count_correct": 97,
        "value_correct": 71,
        "set_exact_correct": 80,
        "hybrid_order_exact_correct": 79,
        "ordered_exact_correct": 78,
        "top1_correct": 82,
        "by_context": {
            "34": {
                "rows": 5,
                "nonempty_rows": 5,
                "hybrid_order_exact_correct": 4,
                "ordered_exact_correct": 4,
            }
        },
    }

    def changed(path: str, delta: int) -> dict[str, Any]:
        result = json.loads(json.dumps(raw))
        target: dict[str, Any] = result
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] += delta
        return result

    cases = {
        "pokemonfan_main_plus_one_passes": gate_panel(
            "pokemonfan", raw, changed("ordered_exact_correct", 1)
        )["pass"],
        "pokemonfan_no_improvement_fails": not gate_panel(
            "pokemonfan", raw, raw
        )["pass"],
        "pokemonfan_context34_only_cannot_satisfy_plus_one": not gate_panel(
            "pokemonfan",
            raw,
            changed("by_context.34.ordered_exact_correct", 1),
        )["pass"],
        "pokemonfan_main_regression_fails": not gate_panel(
            "pokemonfan", raw, changed("top1_correct", -1)
        )["pass"],
        "pokemonfan_count_change_fails": not gate_panel(
            "pokemonfan", raw, changed("count_correct", 1)
        )["pass"],
        "flg_exact_tie_passes": gate_panel("flg", raw, raw)["pass"],
        "core5_exact_tie_passes": gate_panel("core5", raw, raw)["pass"],
    }
    if not all(cases.values()):
        raise RuntimeError(f"synthetic promotion-gate self-test failed: {cases}")
    return {"status": "passed", "cases": cases}


def source_ast_audit() -> dict[str, Any]:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SCRIPT))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    calls = [dotted(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = sorted(
        call
        for call in calls
        if call in {
            "helper.evaluate_panel",
            "helper.formal",
            "torch.load",
            "torch.save",
        }
        or call.endswith(".backward")
        or call.endswith(".step")
        or call.endswith(".write_text")
        or call.endswith(".write_bytes")
    )
    if forbidden:
        raise RuntimeError(f"contract-only AST gate failed: {forbidden}")
    return {
        "status": "contract_only_ast_audit_passed",
        "candidate_evaluation_calls": calls.count("helper.evaluate_panel"),
        "formal_calls": calls.count("helper.formal"),
        "model_load_or_save_calls": calls.count("torch.load")
        + calls.count("torch.save"),
        "optimizer_or_backward_calls": sum(
            call.endswith(".backward") or call.endswith(".step") for call in calls
        ),
        "file_write_calls": sum(
            call.endswith(".write_text") or call.endswith(".write_bytes")
            for call in calls
        ),
    }


def candidate_placeholders() -> dict[str, Any]:
    values = {
        "path": None if ENDPOINT_PATH is None else str(ENDPOINT_PATH.relative_to(ROOT)),
        "file_sha256": ENDPOINT_FILE_SHA256,
        "model_state_sha256": ENDPOINT_MODEL_STATE_SHA256,
        "generation_manifest_path": (
            None
            if ENDPOINT_GENERATION_MANIFEST_PATH is None
            else str(ENDPOINT_GENERATION_MANIFEST_PATH.relative_to(ROOT))
        ),
        "generation_manifest_sha256": ENDPOINT_GENERATION_MANIFEST_SHA256,
    }
    return {
        **values,
        "all_frozen": all(value is not None for value in values.values()),
        "actual_enabled": False,
    }


def promotion_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "contract_only_unarmed_endpoint_placeholders_required",
        "scope": {
            "split": "train",
            "all_train_rows": True,
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "candidate_evaluation_executed": False,
            "training": False,
            "broad": False,
            "gold": False,
            "network": False,
            "package_upload_submission": False,
        },
        "lineage": {
            "parent": {
                "path": str(PARENT.relative_to(ROOT)),
                "file_sha256": PARENT_FILE_SHA256,
                "model_state_sha256": PARENT_MODEL_STATE_SHA256,
                "update": 468,
            },
            "shadow_runner": {
                "path": str(SHADOW_RUNNER.relative_to(ROOT)),
                "sha256": SHADOW_RUNNER_SHA256,
            },
            "shadow_preregistration": {
                "path": str(SHADOW_PREREGISTRATION.relative_to(ROOT)),
                "sha256": SHADOW_PREREGISTRATION_SHA256,
            },
            "shadow_result": {
                "path": str(SHADOW_RESULT.relative_to(ROOT)),
                "sha256": SHADOW_RESULT_SHA256,
                "required_decision": "GO",
                "required_candidate": (
                    "raw_u468_actor6_direct512_equalblend_plain_sgd_lr5e-5"
                ),
            },
            "sole_endpoint": candidate_placeholders(),
        },
        "checkpoint_format": {
            "kind": "ppo_eval_only",
            "root": "dict",
            "required_keys": [
                "model_state_dict",
                "feature_version",
                "bc_feature_version",
                "config",
                "model_config",
                "learner_deck_hash",
                "reward",
                "action_distribution",
                "update",
                "evaluation_only",
                "resume_forbidden",
                "optimizer_states_omitted",
            ],
            "update": 468,
            "evaluation_only": True,
            "resume_forbidden": True,
            "optimizer_state_keys_forbidden": [
                "optimizer_state_dict",
                "bc_replay_optimizer_state_dict",
                "fresh_special_optimizer_state_dict",
                "opponent_quota_state",
            ],
            "inference_schema_exact_parent": [
                "feature_version",
                "bc_feature_version",
                "model_config",
                "count_head_output_classes",
            ],
            "model_state_key_shape_dtype_exact_parent": True,
            "changed_tensor_names_exact": list(ACTOR6_NAMES),
            "all_six_actor_tensors_must_change": True,
            "all_other_tensors_bit_exact_parent": True,
        },
        "data": {
            panel: {
                "path": str(DATASETS[panel].relative_to(ROOT)),
                "sha256": DATA_SHA256[panel],
                "expected_train_shape": EXPECTED_TRAIN_SHAPE[panel],
            }
            for panel in DATASETS
        },
        "raw_profile": {
            "path": str(PROFILE.relative_to(ROOT)),
            "sha256": PROFILE_SHA256,
            "status": "completed_train_only",
            "validation_opened": False,
            "raw_set_and_ordered_counts_must_match_profile": True,
        },
        "dependencies": {
            str(path.relative_to(ROOT)): sha256
            for path, sha256 in DEPENDENCY_SHA256.items()
        },
        "evaluation": {
            "models": ["raw_full_u468", "sole_equalblend_endpoint"],
            "panels": list(DATASETS),
            "evaluation_count": 6,
            "run_all_six_before_decision": True,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "max_rows": MAX_ROWS,
            "device": DEVICE,
            "prediction_order": "policy_greedy",
            "canonicalize_order": False,
            "archive_rule": (
                "open every train/*.jsonl member exactly once; require row split "
                "train; open no other member"
            ),
            "reusable_hash_bound_helper": HELPER_EVIDENCE,
            "reusable_functions": [
                "StrictTrainDataset",
                "instantiate_checkpoint",
                "evaluate_panel",
                "metric_value",
                "model_state_sha256",
                "verify_actor6_only",
                "read_regular_file",
                "checkpoint_from_bytes",
            ],
        },
        "promotion_gate": {
            "structural_exact_raw": list(STRUCTURAL_METRICS),
            "immutable_exact_raw": list(IMMUTABLE_METRICS),
            "all_panels_non_regression": list(NON_REGRESSION_METRICS),
            "pokemonfan_plus_one_required_among": list(MAIN_QUALITY_METRICS),
            "pokemonfan_plus_one_minimum_delta": 1,
            "pokemonfan_context34_cannot_satisfy_plus_one": True,
            "pokemonfan_count_or_value_cannot_satisfy_plus_one": True,
            "flg_extra_improvement_required": False,
            "core5_extra_improvement_required": False,
            "decision": "PROMOTE iff every check passes; otherwise CLOSE before valid",
            "maximum_promoted_endpoints": 1,
        },
        "arming_requirements": {
            "freeze_endpoint_path_file_sha_model_sha_and_generation_manifest_sha": True,
            "generation_manifest_must_bind_shadow_result_and_exact_recipe": True,
            "gate_contract_must_be_frozen_before_endpoint_materialization": True,
            "formal_attempt_marker": "O_EXCL and one-shot",
            "result_publication": "O_EXCL then sealed read-only",
            "actual_mode_in_this_revision": False,
        },
    }


def static_audit() -> dict[str, Any]:
    ast_result = source_ast_audit()
    if candidate_placeholders()["all_frozen"]:
        raise RuntimeError("contract-only draft unexpectedly has armed endpoint values")

    fixed_evidence, fixed_payloads = helper.verify_fixed_inputs()
    profile_summary = helper.verify_profile(fixed_payloads["profile"])
    parent = helper.verify_parent_checkpoint(fixed_payloads["parent"])
    if helper.model_state_sha256(parent["model_state_dict"]) != PARENT_MODEL_STATE_SHA256:
        raise RuntimeError("raw U468 model-state hash drift")
    if helper.ACTOR6_NAMES != ACTOR6_NAMES:
        raise RuntimeError("actor6 scope drift from frozen evaluator helper")
    if helper.DATA_SHA256 != DATA_SHA256 or helper.EXPECTED_TRAIN_SHAPE != EXPECTED_TRAIN_SHAPE:
        raise RuntimeError("full-train data contract drift from frozen evaluator helper")
    if helper.BATCH_SIZE != BATCH_SIZE:
        raise RuntimeError("full-train batch-size drift from frozen evaluator helper")
    if helper.DEPENDENCY_SHA256 != DEPENDENCY_SHA256:
        raise RuntimeError("full-train dependency hash contract drift")

    lineage_evidence = {
        "shadow_runner": require_single_link_regular(
            SHADOW_RUNNER, SHADOW_RUNNER_SHA256, "equal-blend shadow runner"
        ),
        "shadow_preregistration": require_single_link_regular(
            SHADOW_PREREGISTRATION,
            SHADOW_PREREGISTRATION_SHA256,
            "equal-blend shadow preregistration",
        ),
        "shadow_result": require_single_link_regular(
            SHADOW_RESULT, SHADOW_RESULT_SHA256, "equal-blend shadow result"
        ),
    }
    shadow_result = json.loads(SHADOW_RESULT.read_text(encoding="utf-8"))
    candidate = shadow_result.get("candidate")
    if (
        shadow_result.get("decision") != "GO"
        or not isinstance(candidate, dict)
        or candidate.get("candidate")
        != "raw_u468_actor6_direct512_equalblend_plain_sgd_lr5e-5"
        or candidate.get("fully_passes") is not True
    ):
        raise RuntimeError("equal-blend shadow authorization drift")

    return {
        "schema_version": SCHEMA,
        "status": "contract_only_static_audit_passed_actual_disabled",
        "tool": {
            "path": str(SCRIPT.relative_to(ROOT)),
            "sha256": sha256_file(SCRIPT),
        },
        "ast_audit": ast_result,
        "synthetic_gate_self_test": synthetic_gate_self_test(),
        "fixed_inputs": fixed_evidence,
        "profile_summary": profile_summary,
        "lineage_evidence": lineage_evidence,
        "candidate_placeholders": candidate_placeholders(),
        "candidate_model_loaded": False,
        "candidate_evaluation_executed": False,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
        "writes_performed": False,
        "contract": promotion_contract(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "static-audit"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    result = promotion_contract() if args.mode == "contract" else static_audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
