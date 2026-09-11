#!/usr/bin/env python3
"""Freeze a zero-write contract for a train-only equal-blend ray probe.

The completed alpha=1 equal-blend endpoint tied the raw U468 aggregate counts
on all three full-train panels.  Aggregate equality is not a row-identity
proof, and the endpoint is too small to locate the next policy-ranking
boundary by counts alone.  This tool therefore freezes a future RAM-only
probe contract.  It binds raw U468, the exact alpha=1 materialization, the
closed full-train gate, the old unarmed gate precommit, the raw margin profile,
the evaluator, and every train archive.

This file is deliberately contract-only.  Static and cache modes remain
zero-write; contract mode emits the complete immutable design to stdout.  A
separate future formal runner must bind this frozen tool and its frozen design
artifact before it may consume a one-shot attempt marker.  This file never
loads a model, imports torch, opens validation payloads, or writes any file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping


sys.dont_write_bytecode = True


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
SCRIPT = TOOLS / "probe_u468_raw_equalblend_ray_thresholds.py"

SCHEMA = "ptcg-u468-raw-equalblend-ray-threshold-probe-contract-v4"
PREREGISTRATION_SCHEMA = (
    "ptcg-u468-raw-equalblend-ray-threshold-probe-preregistration-v4"
)
BRANCH = "ppo_u468_raw_equalblend_ray_threshold_probe_design202608121"
FUTURE_ATTEMPT_MARKER = ROOT / f".ptcg-{BRANCH}-attempt.json"
FUTURE_RESULT = ROOT / f"artifacts/{BRANCH}.result.json"
FORMAL_PREREGISTRATION = ROOT / f"artifacts/{BRANCH}.preregistration.json"
FUTURE_FORMAL_RUNNER = TOOLS / "run_u468_raw_equalblend_ray_threshold_probe_formal.py"
DESIGN_ARTIFACT = ROOT / (
    "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_contract_"
    "design202608121.json"
)
SUPERSEDED_DESIGN_ARTIFACTS = [
    "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_contract_design202608118.json",
    "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_contract_design202608119.json",
    "artifacts/ppo_u468_raw_equalblend_ray_threshold_probe_contract_design202608120.json",
]

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

MATERIALIZER = TOOLS / "materialize_u468_raw_actor6_equalblend_sgd512_endpoint.py"
MATERIALIZER_SHA256 = (
    "fdd259ba846234db8358a630c392c637773645105116d3be7fe2089cbdf3ccd3"
)
MATERIALIZATION_BRANCH = (
    "ppo_u468_raw_actor6_equalblend_sgd512_materialize_design202608116"
)
MATERIALIZATION_PREREGISTRATION = ROOT / (
    f"artifacts/{MATERIALIZATION_BRANCH}.preregistration.json"
)
MATERIALIZATION_PREREGISTRATION_SHA256 = (
    "1d076b31e82d189ef829bc77bd89fe3808fd212bc3113298ae29aed437392882"
)
MATERIALIZATION_ATTEMPT_MARKER = ROOT / (
    f".ptcg-{MATERIALIZATION_BRANCH}-attempt.json"
)
MATERIALIZATION_ATTEMPT_MARKER_SHA256 = (
    "0a131e37a27f33c14a1e4f91ed7dba79ac2b340407c94f9e3704b624be20c460"
)
MATERIALIZATION_ROOT = ROOT / f"artifacts/{MATERIALIZATION_BRANCH}"
ENDPOINT = MATERIALIZATION_ROOT / "equalblend-sgd512-eval-only.pt"
ENDPOINT_FILE_SHA256 = (
    "1684493b48b9c6a77696150d925150c21d8028c5fdd45f80e9e838a94d472be2"
)
ENDPOINT_MODEL_STATE_SHA256 = (
    "5fd532338caf54abab95943d482dc35228559257e07d325bd1cacb20db62648d"
)
MATERIALIZATION_MANIFEST = MATERIALIZATION_ROOT / "materialization_manifest.json"
MATERIALIZATION_MANIFEST_SHA256 = (
    "51a133ec362ee489b05018728d43fb17f8ab12eb4adf1b223426afa6fafdccb1"
)
MATERIALIZATION_RESULT = MATERIALIZATION_ROOT / "materialization_result.json"
MATERIALIZATION_RESULT_SHA256 = (
    "9b0395c63478434ed6b577566ebd7663c6c4dfb20742ba747f374aeb618896e3"
)

UNARMED_GATE = TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate.py"
UNARMED_GATE_SHA256 = (
    "4f36bd0f86cbb811d8142f155027445b4bfb43a1b3e55b2fbf3b5787981aeae0"
)
UNARMED_GATE_CONTRACT = ROOT / (
    "artifacts/ppo_u468_raw_equalblend_train_only_promotion_gate_contract_"
    "design202608116.json"
)
UNARMED_GATE_CONTRACT_SHA256 = (
    "e0d80b72aed237c3de4b643157b058acf2bff513d11420dd7579db0308b970c2"
)

CLOSED_GATE_RUNNER = (
    TOOLS / "run_u468_raw_equalblend_train_only_promotion_gate_armed.py"
)
CLOSED_GATE_RUNNER_SHA256 = (
    "c5db208e544f9f7165b189faafc8cbe2b87ea29e0f79f876a6aa6bf42dc837a6"
)
CLOSED_GATE_BRANCH = (
    "ppo_u468_raw_equalblend_train_only_promotion_gate_armed_design202608117"
)
CLOSED_GATE_PREREGISTRATION = ROOT / (
    f"artifacts/{CLOSED_GATE_BRANCH}.preregistration.json"
)
CLOSED_GATE_PREREGISTRATION_SHA256 = (
    "43eee0d5d247fd280d9b81499735d35b6fb693dbe0f734c5ef17d2830375bb3c"
)
CLOSED_GATE_ATTEMPT_MARKER = ROOT / f".ptcg-{CLOSED_GATE_BRANCH}-attempt.json"
CLOSED_GATE_ATTEMPT_MARKER_SHA256 = (
    "457c9359f13788c98dc2b0af22e221075710df56a9e1f7fdb3b33313a81728c6"
)
CLOSED_GATE_ROOT = ROOT / f"artifacts/{CLOSED_GATE_BRANCH}"
CLOSED_GATE_DECISION = CLOSED_GATE_ROOT / "gate_decision.json"
CLOSED_GATE_DECISION_SHA256 = (
    "e2585f2b393b15f234d26507c918a4356459439a3c3687bd9253267153d74e90"
)
CLOSED_GATE_MANIFEST = CLOSED_GATE_ROOT / "execution_manifest.json"
CLOSED_GATE_MANIFEST_SHA256 = (
    "9dd33205db048f072e207d09fa28d42632c3912f1276ae269e09fe3f9bf03ace"
)
CLOSED_GATE_COMPLETED = CLOSED_GATE_ROOT / "COMPLETED.json"
CLOSED_GATE_COMPLETED_SHA256 = (
    "86526fecf340ffeefdb5052dfaacd06588f7c9324d8d2dd7dc3b5db805841186"
)
CLOSED_EVALUATIONS = {
    "raw_full_u468/flg.json": (
        "b1b83587d39ee29f6d6327082ad3570e754bce77e5193f2feebcd1eeb173295e"
    ),
    "raw_full_u468/pokemonfan.json": (
        "e9e89927b45eabb175fdd160802613b9a76912bc40b074fef444ff2331840027"
    ),
    "raw_full_u468/core5.json": (
        "29249f098a9c961f1e73e8fadb538278c6698e5328f13152f55fa59a02528dd4"
    ),
    "sole_equalblend_endpoint/flg.json": (
        "1a200383535de971e75e94009b167b60ca602493e1d2d8af47ec5d14d19bf8c9"
    ),
    "sole_equalblend_endpoint/pokemonfan.json": (
        "4744823636f7b6179820930a01a844eba823642e366f4b635e61680c9a044977"
    ),
    "sole_equalblend_endpoint/core5.json": (
        "1dc0df6af28ce8232b9e629d56ac60228e1d4a0385e005b6b4129a5f9bfe01ae"
    ),
}

PROFILE = ROOT / "artifacts/u468_raw_full_train_margin_profile_v3_20260802.json"
PROFILE_SHA256 = (
    "17af98e175167398a80cd2e1f8930c3d1d85cfe6ceea3143dfe2ba44dc89f1f9"
)
PROFILE_WRAPPER = TOOLS / "profile_u468_raw_full_train_margins.py"
PROFILE_WRAPPER_SHA256 = (
    "c70e52e2398b8551baf0fa3a3a403fbf6636b56c70010fb948a5c85c298977ae"
)
PROFILE_FRAMEWORK = TOOLS / "profile_u468_beta1157_train_margins.py"
PROFILE_FRAMEWORK_SHA256 = (
    "f1018994c48ffadcbc5277f0dfd9233086693f90bf30f5bf67462965b4e18142"
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
    "pokemonfan": (
        "71da8819f89ea9fba986152de4294f263f4a14e286659759be78516fa5ba6598"
    ),
    "core5": "bbfc8d645bf5d189ce65a3abf30cf35ed86b8b8386aef1e08c72496c07b06d2a",
}
EXPECTED_TRAIN = {
    "flg": {
        "rows": 9443,
        "members": 1,
        "context34_rows": 42,
        "nonempty_rows": 9426,
    },
    "pokemonfan": {
        "rows": 9487,
        "members": 1,
        "context34_rows": 38,
        "nonempty_rows": 9450,
    },
    "core5": {
        "rows": 5120,
        "members": 20,
        "context34_rows": 20,
        "nonempty_rows": 5106,
    },
}
EXPECTED_TRAIN_IDENTITY_SHA256 = {
    "flg": "c4ecb8cd6b6744b823ba12f1395c2900b6168b4bba63cca1a1756e0389ad0e0c",
    "pokemonfan": (
        "7457709480f8b7929106079ab0c255f1528c2aa7086f0b536e6922af588838cb"
    ),
    "core5": "c494d6c13787dc934097ea3896f93838b299562664467b2c7fe5b215ce9958e7",
}
EXPECTED_UNION_IDENTITY_SUMMARY_SHA256 = (
    "f8b2df4951c87c2d0b1e9c2358223494db10cf9bc7de02b1667b7d2ded00792d"
)

ACTOR6_NAMES = (
    "actor_query.weight",
    "actor_key.weight",
    "actor_residual.0.weight",
    "actor_residual.0.bias",
    "actor_residual.2.weight",
    "actor_residual.2.bias",
)
BASE_LEARNING_RATE = 5e-5
ALPHA1_DISPLACEMENT_L2 = 2.1571429878723395e-5
ALPHA1_DISPLACEMENT_MAX_ABS = 2.3543834686279297e-6

# Eight powers of two form one bounded observed grid through alpha=256.  This
# probes selected points; it can miss a flip/revert between adjacent points and
# does not claim to solve an exact continuous root.
CANDIDATE_ALPHAS = (2, 4, 8, 16, 32, 64, 128, 256)

MAIN_METRICS = (
    "set_exact",
    "hybrid_order_exact",
    "ordered_exact",
    "top1_correct",
)
CONTEXT34_METRICS = (
    "context34_hybrid_order_exact",
    "context34_ordered_exact",
)
SAFETY_METRICS = MAIN_METRICS + CONTEXT34_METRICS
TRANSITION_KEYS = ("cc", "cw", "wc", "ww")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(
    path: Path,
    expected_sha256: str,
    label: str,
    *,
    expected_mode: int | None = None,
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
    mode = stat.S_IMODE(observed.st_mode)
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(f"{label} mode drift: {oct(mode)}")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": observed.st_size,
        "mode": oct(mode),
        "nlink": observed.st_nlink,
    }


def require_sealed_tree(
    root: Path,
    expected_files: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    observed = os.lstat(root)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o555
    ):
        raise RuntimeError(f"{label} root is not a real mode-0555 directory")
    observed_files = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )
    if observed_files != sorted(expected_files):
        raise RuntimeError(f"{label} exact-file inventory drift: {observed_files}")
    evidence = {
        name: require_regular(
            root / name,
            digest,
            f"{label} {name}",
            expected_mode=0o444,
        )
        for name, digest in expected_files.items()
    }
    for path in root.rglob("*"):
        if path.is_dir():
            mode = stat.S_IMODE(os.lstat(path).st_mode)
            if path.is_symlink() or mode != 0o555:
                raise RuntimeError(f"{label} subdirectory mode/link drift: {path}")
    return {
        "path": str(root.relative_to(ROOT)),
        "mode": "0o555",
        "exact_files": observed_files,
        "files": evidence,
    }


def strict_json(
    path: Path,
    label: str,
    *,
    allow_positive_infinity: bool = False,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} has duplicate JSON key {key}")
            result[key] = value
        return result

    def parse_constant(value: str) -> Any:
        if allow_positive_infinity and value == "Infinity":
            return math.inf
        raise RuntimeError(f"{label} has forbidden JSON constant {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=parse_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root is not an object")
    return value


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the frozen repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError("requires my_project_env Python")
    if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
        raise RuntimeError("requires exact Python -I -B")


def verify_materialization() -> dict[str, Any]:
    files = {
        "runner": require_regular(
            MATERIALIZER,
            MATERIALIZER_SHA256,
            "equalblend materializer",
            expected_mode=0o555,
        ),
        "preregistration": require_regular(
            MATERIALIZATION_PREREGISTRATION,
            MATERIALIZATION_PREREGISTRATION_SHA256,
            "materialization preregistration",
            expected_mode=0o444,
        ),
        "attempt_marker": require_regular(
            MATERIALIZATION_ATTEMPT_MARKER,
            MATERIALIZATION_ATTEMPT_MARKER_SHA256,
            "materialization attempt marker",
            expected_mode=0o444,
        ),
    }
    tree = require_sealed_tree(
        MATERIALIZATION_ROOT,
        {
            ENDPOINT.name: ENDPOINT_FILE_SHA256,
            MATERIALIZATION_MANIFEST.name: MATERIALIZATION_MANIFEST_SHA256,
            MATERIALIZATION_RESULT.name: MATERIALIZATION_RESULT_SHA256,
        },
        "equalblend materialization",
    )
    preregistration = strict_json(
        MATERIALIZATION_PREREGISTRATION, "materialization preregistration"
    )
    marker = strict_json(
        MATERIALIZATION_ATTEMPT_MARKER, "materialization attempt marker"
    )
    manifest = strict_json(MATERIALIZATION_MANIFEST, "materialization manifest")
    result = strict_json(MATERIALIZATION_RESULT, "materialization result")
    endpoint = manifest.get("endpoint", {})
    integrity = manifest.get("integrity", {})
    if (
        preregistration.get("status") != "locked_before_materialization"
        or marker.get("status")
        != "materialization_attempt_consumed_before_model_or_optimizer"
        or marker.get("materialization_contract")
        != preregistration.get("materialization_contract")
        or manifest.get("materialization_contract")
        != preregistration.get("materialization_contract")
        or endpoint.get("sha256") != ENDPOINT_FILE_SHA256
        or endpoint.get("model_state_sha256") != ENDPOINT_MODEL_STATE_SHA256
        or endpoint.get("in_memory_reload", {}).get("evaluation_only") is not True
        or endpoint.get("in_memory_reload", {}).get("resume_forbidden") is not True
        or endpoint.get("in_memory_reload", {}).get(
            "optimizer_and_quota_states_absent"
        )
        is not True
        or integrity.get("shadow_full_training_record_exact") is not True
        or integrity.get("shadow_full_evaluation_record_exact") is not True
        or integrity.get("shadow_displacement_record_exact") is not True
        or integrity.get("validation_member_payloads_opened") is not False
        or integrity.get("submission_performed") is not False
        or result.get("decision") != "MATERIALIZED"
        or result.get("endpoint", {}).get("sha256") != ENDPOINT_FILE_SHA256
        or result.get("endpoint", {}).get("model_state_sha256")
        != ENDPOINT_MODEL_STATE_SHA256
        or result.get("manifest", {}).get("sha256")
        != MATERIALIZATION_MANIFEST_SHA256
        or result.get("validation_member_payloads_opened") is not False
        or result.get("submission_performed") is not False
    ):
        raise RuntimeError("equalblend materialization semantic binding drift")
    displacement = manifest.get("reproduction", {}).get("displacement_from_raw")
    if (
        not isinstance(displacement, dict)
        or displacement.get("changed_parameter_names") != sorted(ACTOR6_NAMES)
        or displacement.get("changed_scope_exact_actor6") is not True
        or displacement.get("l2") != ALPHA1_DISPLACEMENT_L2
        or displacement.get("max_abs") != ALPHA1_DISPLACEMENT_MAX_ABS
    ):
        raise RuntimeError("alpha=1 displacement binding drift")
    return {
        "files": files,
        "sealed_tree": tree,
        "endpoint_model_state_sha256": ENDPOINT_MODEL_STATE_SHA256,
        "actor6_only": True,
        "evaluation_only": True,
        "resume_forbidden": True,
        "optimizer_and_quota_states_absent": True,
        "validation_member_payloads_opened": False,
        "submission_performed": False,
    }


def flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        result.update(flatten_leaves(item, path))
    return result


def verify_closed_gate() -> dict[str, Any]:
    outer = {
        "runner": require_regular(
            CLOSED_GATE_RUNNER,
            CLOSED_GATE_RUNNER_SHA256,
            "closed full-train gate runner",
            expected_mode=0o555,
        ),
        "preregistration": require_regular(
            CLOSED_GATE_PREREGISTRATION,
            CLOSED_GATE_PREREGISTRATION_SHA256,
            "closed full-train gate preregistration",
            expected_mode=0o444,
        ),
        "attempt_marker": require_regular(
            CLOSED_GATE_ATTEMPT_MARKER,
            CLOSED_GATE_ATTEMPT_MARKER_SHA256,
            "closed full-train gate marker",
            expected_mode=0o444,
        ),
    }
    expected_tree = {
        "gate_decision.json": CLOSED_GATE_DECISION_SHA256,
        "execution_manifest.json": CLOSED_GATE_MANIFEST_SHA256,
        "COMPLETED.json": CLOSED_GATE_COMPLETED_SHA256,
        **CLOSED_EVALUATIONS,
    }
    tree = require_sealed_tree(CLOSED_GATE_ROOT, expected_tree, "closed full-train gate")
    decision = strict_json(CLOSED_GATE_DECISION, "closed gate decision")
    manifest = strict_json(CLOSED_GATE_MANIFEST, "closed gate manifest")
    completed = strict_json(CLOSED_GATE_COMPLETED, "closed gate completion seal")
    if (
        decision.get("status") != "not_promoted_branch_closed_before_valid"
        or decision.get("promoted") is not False
        or decision.get("promoted_endpoint") is not None
        or decision.get("promoted_endpoint_count") != 0
        or decision.get("evaluation_count") != 6
        or decision.get("all_six_full_train_evaluations_completed_before_decision")
        is not True
        or any(decision.get("downstream_executed", {}).values())
        or decision.get("validation_member_payloads_opened") is not False
        or decision.get("validation_results_read") is not False
        or manifest.get("evaluation_count") != 6
        or manifest.get("validation_member_payloads_opened") is not False
        or manifest.get("validation_results_read") is not False
        or completed.get("status") != "completed"
        or completed.get("promoted") is not False
        or completed.get("promoted_endpoint") is not None
        or completed.get("validation_member_payloads_opened") is not False
        or completed.get("validation_results_read") is not False
    ):
        raise RuntimeError("closed full-train gate status drift")
    panel_gates = decision.get("panel_gates")
    if not isinstance(panel_gates, dict) or set(panel_gates) != set(DATASETS):
        raise RuntimeError("closed gate panel inventory drift")
    failures = {
        panel: sorted(
            key
            for key, value in gate.get("checks", {}).items()
            if value.get("pass") is not True
        )
        for panel, gate in panel_gates.items()
    }
    expected_failures = {
        "flg": [],
        "pokemonfan": ["pokemonfan_at_least_one_main_quality_count_plus_one"],
        "core5": [],
    }
    if failures != expected_failures:
        raise RuntimeError(f"closed-gate failure-set drift: {failures}")
    metric_equality: dict[str, Any] = {}
    for panel in DATASETS:
        raw = strict_json(
            CLOSED_GATE_ROOT / f"raw_full_u468/{panel}.json",
            f"closed raw {panel} evaluation",
        )
        endpoint = strict_json(
            CLOSED_GATE_ROOT / f"sole_equalblend_endpoint/{panel}.json",
            f"closed endpoint {panel} evaluation",
        )
        if (
            raw.get("split") != "train"
            or endpoint.get("split") != "train"
            or raw.get("validation_members_opened") is not False
            or endpoint.get("validation_members_opened") is not False
            or raw.get("validation_results_read") is not False
            or endpoint.get("validation_results_read") is not False
            or raw.get("opened_members") != raw.get("archive_inventory_train_members")
            or endpoint.get("opened_members")
            != endpoint.get("archive_inventory_train_members")
            or raw.get("metrics") != endpoint.get("metrics")
        ):
            raise RuntimeError(f"closed {panel} aggregate equality/member audit drift")
        leaves = flatten_leaves(raw["metrics"])
        if len(leaves) != 572:
            raise RuntimeError(f"closed {panel} metric leaf-count drift: {len(leaves)}")
        metric_equality[panel] = {
            "aggregate_and_by_context_metrics_exact": True,
            "scalar_leaf_count": len(leaves),
            "row_identity_equality_not_proven_by_aggregate_counts": True,
        }
    return {
        "outer_files": outer,
        "sealed_tree": tree,
        "decision": "CLOSE",
        "only_failed_gate": (
            "pokemonfan requires at least one +1 among four main quality counts"
        ),
        "metric_equality": metric_equality,
        "validation_member_payloads_opened": False,
        "validation_results_read": False,
        "submission_performed": False,
    }


def profile_observations(profile: Mapping[str, Any]) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for panel in DATASETS:
        payload = profile.get("profiles", {}).get(panel)
        if not isinstance(payload, dict):
            raise RuntimeError(f"profile lacks {panel}")
        near_wrong = payload.get("near_wrong")
        fragile = payload.get("fragile_correct")
        if not isinstance(near_wrong, list) or not isinstance(fragile, list):
            raise RuntimeError(f"profile retained lists malformed for {panel}")
        identities: set[tuple[str, int, str]] = set()
        for record in near_wrong + fragile:
            identity = (
                str(record.get("member")),
                int(record.get("line_index", -1)),
                str(record.get("line_sha256")),
            )
            if (
                not identity[0].startswith("train/")
                or not identity[0].endswith(".jsonl")
                or identity[1] < 0
                or len(identity[2]) != 64
                or identity in identities
            ):
                raise RuntimeError(f"profile retained identity drift for {panel}")
            identities.add(identity)
        actionable_wrong = [
            record
            for record in near_wrong
            if bool(record.get("expert_order"))
            and (
                int(record.get("min_count", -1))
                == int(record.get("max_count", -2))
                or float(record.get("count_margin", -math.inf)) > 0.0
            )
            and float(record.get("selection_margin", math.inf)) <= 0.0
        ]
        negative_actionable = sorted(
            float(record["selection_margin"])
            for record in actionable_wrong
            if float(record["selection_margin"]) < 0.0
        )
        observations[panel] = {
            "rows": int(payload.get("rows", -1)),
            "ordered_correct": int(payload.get("ordered_correct", -1)),
            "ordered_wrong": int(payload.get("ordered_wrong", -1)),
            "near_wrong_retained": len(near_wrong),
            "fragile_correct_retained": len(fragile),
            "retained_identity_count": len(identities),
            "actionable_policy_wrong_retained": len(actionable_wrong),
            "actionable_wrong_at_zero_selection_margin": sum(
                float(record["selection_margin"]) == 0.0
                for record in actionable_wrong
            ),
            "fragile_correct_at_zero_selection_margin": sum(
                float(record["selection_margin"]) == 0.0 for record in fragile
            ),
            "closest_negative_actionable_selection_margin": (
                max(negative_actionable) if negative_actionable else None
            ),
        }
    expected = {
        "flg": {
            "rows": 9443,
            "ordered_correct": 7095,
            "ordered_wrong": 2348,
            "near_wrong_retained": 2048,
            "fragile_correct_retained": 2048,
            "retained_identity_count": 4096,
            "actionable_policy_wrong_retained": 2022,
            "actionable_wrong_at_zero_selection_margin": 3,
            "fragile_correct_at_zero_selection_margin": 6,
            "closest_negative_actionable_selection_margin": -0.001953125,
        },
        "pokemonfan": {
            "rows": 9487,
            "ordered_correct": 8203,
            "ordered_wrong": 1284,
            "near_wrong_retained": 1284,
            "fragile_correct_retained": 2048,
            "retained_identity_count": 3332,
            "actionable_policy_wrong_retained": 1206,
            "actionable_wrong_at_zero_selection_margin": 3,
            "fragile_correct_at_zero_selection_margin": 8,
            "closest_negative_actionable_selection_margin": -0.00390625,
        },
        "core5": {
            "rows": 5120,
            "ordered_correct": 4025,
            "ordered_wrong": 1095,
            "near_wrong_retained": 1095,
            "fragile_correct_retained": 2048,
            "retained_identity_count": 3143,
            "actionable_policy_wrong_retained": 1036,
            "actionable_wrong_at_zero_selection_margin": 1,
            "fragile_correct_at_zero_selection_margin": 3,
            "closest_negative_actionable_selection_margin": -0.00390625,
        },
    }
    if observations != expected:
        raise RuntimeError(f"raw margin-profile observation drift: {observations}")
    return observations


def verify_profile() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = {
        "profile": require_regular(
            PROFILE,
            PROFILE_SHA256,
            "raw full-train margin profile",
            expected_mode=0o444,
        ),
        "wrapper": require_regular(
            PROFILE_WRAPPER,
            PROFILE_WRAPPER_SHA256,
            "raw margin-profile wrapper",
            expected_mode=0o555,
        ),
        "framework": require_regular(
            PROFILE_FRAMEWORK,
            PROFILE_FRAMEWORK_SHA256,
            "raw margin-profile framework",
        ),
    }
    profile = strict_json(
        PROFILE,
        "raw full-train margin profile",
        allow_positive_infinity=True,
    )
    if (
        profile.get("schema_version") != "ptcg-u468-raw-full-train-margin-profile-v3"
        or profile.get("status") != "completed_train_only"
        or profile.get("split") != "train"
        or profile.get("validation_opened") is not False
        or profile.get("base", {}).get("checkpoint_file_sha256")
        != PARENT_FILE_SHA256
        or profile.get("base", {}).get("model_state_sha256")
        != PARENT_MODEL_STATE_SHA256
        or profile.get("base", {}).get("update") != 468
    ):
        raise RuntimeError("raw margin-profile binding drift")
    return evidence, {
        "observations": profile_observations(profile),
        "profile": profile,
    }


def verify_unarmed_precommit() -> dict[str, Any]:
    tool = require_regular(
        UNARMED_GATE,
        UNARMED_GATE_SHA256,
        "unarmed promotion-gate tool",
        expected_mode=0o555,
    )
    artifact = require_regular(
        UNARMED_GATE_CONTRACT,
        UNARMED_GATE_CONTRACT_SHA256,
        "unarmed promotion-gate contract",
        expected_mode=0o444,
    )
    contract = strict_json(UNARMED_GATE_CONTRACT, "unarmed promotion-gate contract")
    sole = contract.get("lineage", {}).get("sole_endpoint")
    if (
        contract.get("status")
        != "contract_only_unarmed_endpoint_placeholders_required"
        or sole
        != {
            "path": None,
            "file_sha256": None,
            "model_state_sha256": None,
            "generation_manifest_path": None,
            "generation_manifest_sha256": None,
            "all_frozen": False,
            "actual_enabled": False,
        }
        or contract.get("scope", {}).get("validation_member_payloads_opened")
        is not False
        or contract.get("scope", {}).get("validation_results_read") is not False
        or contract.get("scope", {}).get("training") is not False
        or contract.get("scope", {}).get("package_upload_submission") is not False
    ):
        raise RuntimeError("unarmed promotion-gate precommit drift")
    return {"tool": tool, "contract": artifact, "payload_exact": True}


def verify_fixed_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    profile_evidence, profile_payload = verify_profile()
    evidence = {
        "self": require_regular(SCRIPT, sha256_file(SCRIPT), "ray probe contract tool"),
        "parent": require_regular(PARENT, PARENT_FILE_SHA256, "raw U468 parent"),
        "materialization": verify_materialization(),
        "unarmed_gate_precommit": verify_unarmed_precommit(),
        "closed_full_train_gate": verify_closed_gate(),
        "profile": profile_evidence,
        "helper": require_regular(
            HELPER,
            HELPER_SHA256,
            "full-train evaluator helper",
            expected_mode=0o555,
        ),
        "dependencies": {
            str(path.relative_to(ROOT)): require_regular(
                path, digest, f"dependency {path.name}"
            )
            for path, digest in DEPENDENCIES.items()
        },
        "datasets": {
            panel: require_regular(path, DATA_SHA256[panel], f"{panel} train archive")
            for panel, path in DATASETS.items()
        },
    }
    return evidence, profile_payload


def ast_audit() -> dict[str, Any]:
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
    duplicate_constant_dict_keys: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen: dict[tuple[type[Any], Any], int] = {}
        for key_node in node.keys:
            if not isinstance(key_node, ast.Constant):
                continue
            value = key_node.value
            try:
                identity = (type(value), value)
                hash(identity)
            except TypeError:
                continue
            if identity in seen:
                duplicate_constant_dict_keys.append(
                    {
                        "key": repr(value),
                        "first_line": seen[identity],
                        "duplicate_line": int(getattr(key_node, "lineno", -1)),
                    }
                )
            else:
                seen[identity] = int(getattr(key_node, "lineno", -1))
    forbidden_calls = sorted(
        call
        for call in calls
        if call
        in {
            "torch.load",
            "torch.save",
            "helper.evaluate_panel",
            "evaluator.evaluate",
            "ppo.model_forward",
            "os.open",
            "os.mkdir",
            "os.chmod",
            "os.unlink",
            "os.rename",
            "os.replace",
        }
        or call.endswith(".backward")
        or call.endswith(".step")
        or call.endswith(".write_text")
        or call.endswith(".write_bytes")
    )
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden_imports = sorted(
        imported_roots.intersection(
            {"requests", "urllib", "socket", "http", "kaggle", "torch"}
        )
    )
    if forbidden_calls or forbidden_imports or duplicate_constant_dict_keys:
        raise RuntimeError(
            f"contract-only AST audit failed: calls={forbidden_calls}, "
            f"imports={forbidden_imports}, "
            f"duplicate_dict_keys={duplicate_constant_dict_keys}"
        )
    return {
        "status": "contract_only_zero_write_ast_audit_passed",
        "forbidden_calls": forbidden_calls,
        "forbidden_imports": forbidden_imports,
        "duplicate_constant_dict_keys": duplicate_constant_dict_keys,
        "model_loads": 0,
        "forward_calls": 0,
        "training_calls": 0,
        "file_writes": 0,
        "network_calls": 0,
        "actual_or_formal_mode_present": False,
    }


def scan_train_identities(
    panel: str,
    path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, int], str]]:
    digest = hashlib.sha256()
    identities: dict[tuple[str, int], str] = {}
    opened: list[str] = []
    rows = 0
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".jsonl")
        )
        if len(members) != EXPECTED_TRAIN[panel]["members"]:
            raise RuntimeError(f"{panel} train-member count drift")
        for member in members:
            if not member.startswith("train/") or not member.endswith(".jsonl"):
                raise RuntimeError(f"refusing non-train member {member}")
            opened.append(member)
            with archive.open(member) as handle:
                for line_index, raw in enumerate(handle):
                    row = json.loads(raw)
                    if not isinstance(row, dict) or row.get("split") != "train":
                        raise RuntimeError(f"{panel} non-train row in {member}")
                    line_sha256 = sha256_bytes(raw)
                    key = (member, line_index)
                    if key in identities:
                        raise RuntimeError(f"{panel} repeated train identity {key}")
                    identities[key] = line_sha256
                    identity = {
                        "panel": panel,
                        "member": member,
                        "line_index_zero_based": line_index,
                        "line_sha256": line_sha256,
                    }
                    digest.update(canonical_json(identity))
                    rows += 1
    if rows != EXPECTED_TRAIN[panel]["rows"]:
        raise RuntimeError(f"{panel} row-count drift: {rows}")
    identity_digest = digest.hexdigest()
    if identity_digest != EXPECTED_TRAIN_IDENTITY_SHA256[panel]:
        raise RuntimeError(
            f"{panel} train identity stream SHA drift: {identity_digest}"
        )
    return (
        {
            "panel": panel,
            "archive": str(path.relative_to(ROOT)),
            "archive_sha256": DATA_SHA256[panel],
            "train_members": opened,
            "train_member_count": len(opened),
            "rows": rows,
            "identity_record_schema": [
                "panel",
                "member",
                "line_index_zero_based",
                "line_sha256",
            ],
            "identity_stream_sha256": identity_digest,
            "identity_hash_recipe": (
                "SHA256 over concatenated canonical-JSON-newline identity records "
                "in sorted train-member and ascending zero-based line order"
            ),
            "non_train_member_payloads_opened": False,
        },
        identities,
    )


def cache_audit(profile: Mapping[str, Any]) -> dict[str, Any]:
    panels: dict[str, Any] = {}
    profile_cross_checks: dict[str, Any] = {}
    for panel, path in DATASETS.items():
        summary, identities = scan_train_identities(panel, path)
        panels[panel] = summary
        records = (
            profile["profiles"][panel]["near_wrong"]
            + profile["profiles"][panel]["fragile_correct"]
        )
        for record in records:
            key = (str(record["member"]), int(record["line_index"]))
            if identities.get(key) != str(record["line_sha256"]):
                raise RuntimeError(f"{panel} profile/archive identity mismatch: {key}")
        profile_cross_checks[panel] = {
            "retained_records": len(records),
            "all_member_line_sha_identities_exact": True,
        }
    union_payload = {
        panel: {
            "rows": panels[panel]["rows"],
            "identity_stream_sha256": panels[panel]["identity_stream_sha256"],
        }
        for panel in DATASETS
    }
    union_digest = sha256_bytes(canonical_json(union_payload))
    if union_digest != EXPECTED_UNION_IDENTITY_SUMMARY_SHA256:
        raise RuntimeError(f"union train identity summary SHA drift: {union_digest}")
    return {
        "status": "zero_write_full_train_identity_cache_audit_passed",
        "panels": panels,
        "union_identity_summary_canonical_sha256": union_digest,
        "profile_cross_checks": profile_cross_checks,
        "validation_member_payloads_opened": False,
        "writes_performed": False,
    }


def candidate_design() -> dict[str, Any]:
    return {
        "baseline": {"alpha": 0, "kind": "exact raw U468"},
        "safety_anchor": {
            "alpha": 1,
            "kind": "exact serialized equalblend endpoint",
            "selectable": False,
            "purpose": (
                "establish formal row-level raw-to-alpha1 fingerprints, transition "
                "matrices, and native-BF16 endpoint logit secants"
            ),
        },
        "candidate_alphas": [
            {
                "alpha": alpha,
                "nominal_scaled_lr_only": alpha * BASE_LEARNING_RATE,
                "nominal_scaled_lr_is_not_recomputed_SGD": True,
                "expected_displacement_l2_if_affine_exact": (
                    alpha * ALPHA1_DISPLACEMENT_L2
                ),
                "expected_displacement_max_abs_if_affine_exact": (
                    alpha * ALPHA1_DISPLACEMENT_MAX_ABS
                ),
            }
            for alpha in CANDIDATE_ALPHAS
        ],
        "candidate_count": len(CANDIDATE_ALPHAS),
        "fixed_before_any_formal_forward": True,
        "ray_definition": {
            "mutable": list(ACTOR6_NAMES),
            "formula": (
                "for actor6 FP32 tensors only, execute three explicit CPU FP32 "
                "operations in order: D=E-R; S=D*alpha; C=R+S; clone every "
                "other raw tensor bit-exact"
            ),
            "operation_dtype": "torch.float32",
            "operation_device": "cpu",
            "operation_order_exact": ["D=E-R", "S=D*alpha", "C=R+S"],
            "lerp_forbidden": True,
            "fused_multiply_add_forbidden": True,
            "alpha1_must_reconstruct_endpoint_tensor_exact": True,
            "all_other_model_tensors_bit_exact_raw": True,
            "candidate_states_exist_in_RAM_only": True,
            "torch_save_forbidden": True,
            "optimizer_forbidden": True,
            "gradient_or_training_step_forbidden": True,
            "recomputed_multi_step_SGD_is_not_this_ray": True,
            "nominal_scaled_lr_is_descriptive_only_not_equivalent_training": True,
        },
        "why_fixed_2_4_8_16_32_64_128_256": (
            "alpha=1 produced no aggregate gain; powers of two provide a bounded "
            "observed grid of stronger points. The grid can miss a flip and revert "
            "between adjacent alphas. At alpha=256 the absolute actor6 L2 "
            "displacement remains below 0.00553. Only each measured point's exact "
            "native-BF16 row gates decide safety."
        ),
    }


def threshold_contract() -> dict[str, Any]:
    return {
        "official_forward": {
            "device": "cuda",
            "autocast_dtype": "torch.bfloat16",
            "model_forward": "hash-bound train_ppo.model_forward",
            "prediction": "hash-bound sample_ordered_actions deterministic=True",
            "prediction_order": "policy_greedy",
            "canonicalize_order": False,
        },
        "raw_and_alpha1_per_row_record": {
            "identity_fields": [
                "panel",
                "archive",
                "member",
                "line_index_zero_based",
                "line_sha256",
            ],
            "full_per_row_logit_arrays_required": False,
            "storage": (
                "construct row records transiently in RAM and stream them into "
                "hashes/transitions; do not publish full raw/alpha1 row records"
            ),
            "full_per_row_records_published": False,
            "required_evidence": [
                "raw and alpha1 identity-bound greedy-order fingerprints",
                "raw and alpha1 identity-bound allowed-logit float32 fingerprints",
                "exact metric transition matrices and CW/WC expanded identities",
                "exact sequence, length, and top1 change identities",
            ],
            "prediction_fingerprint": (
                "canonical SHA256 over every identity plus exact greedy order, "
                "processed in the frozen train identity order"
            ),
            "allowed_logit_fingerprint": (
                "for each row in frozen identity order, hash canonical identity "
                "JSON then NUL; literal <f4 then NUL; canonical JSON shape then "
                "NUL; allowed option indices in ascending original option-index "
                "order as canonical JSON then NUL; CPU contiguous IEEE-754 "
                "float32 little-endian bytes then NUL. Empty allowed slices use "
                "shape [0], indices [], and zero data bytes between the final "
                "two NUL delimiters"
            ),
        },
        "secant_prediction": {
            "probe_kind": "bounded_geometric_threshold_bracket",
            "does_not_claim_an_exact_continuous_root": True,
            "may_miss_flip_or_revert_between_adjacent_grid_points": True,
            "logit_model": "z_hat(alpha)=z_raw+alpha*(z_alpha1-z_raw)",
            "pair_crossing": (
                "for each allowed option pair i,j with nonzero secant slope, "
                "solve z_hat_i(alpha)=z_hat_j(alpha)"
            ),
            "ordered_margin": (
                "minimum expert-choice logit minus best remaining competitor at "
                "each official greedy stage"
            ),
            "set_margin": (
                "minimum expert logit minus maximum non-expert logit when the "
                "frozen count equals expert count"
            ),
            "top1_margin": "maximum expert logit minus maximum non-expert logit",
            "required_advisory_summaries": {
                "pokemonfan_each_main_metric": (
                    "earliest secant-estimated wrong-to-correct alpha and line/"
                    "limiting-pair evidence, or None plus an explicit reason"
                ),
                "each_panel_and_safety_metric": (
                    "earliest secant-estimated correct-to-wrong alpha and line/"
                    "limiting-pair evidence, or None plus an explicit reason"
                ),
                "observed_grid": [
                    "earliest_observed_grid_gain_bracket_by_pokemonfan_main_metric",
                    "earliest_observed_grid_regression_bracket_by_panel_and_safety_metric",
                ],
            },
            "required_for_formal_completion": True,
            "summary_only": True,
            "must_not_participate_in_candidate_selection": True,
            "advisory_only": True,
            "reason": (
                "query-key interaction is quadratic in ray alpha, residual layers "
                "contain GELU, and native BF16 introduces quantization boundaries"
            ),
            "candidate_native_bf16_transitions_are_the_only_gate_authority": True,
            "hybrid_margin_semantics": {
                "context34": "same ordered-stage margin as ordered_exact",
                "non_context34": (
                    "sorted selected set compared with expert order; if expert "
                    "order is not compatible with sorted-set semantics, unreachable"
                ),
                "empty_competitor_set_margin": "+Infinity",
                "argmax_tie_break": "lowest option index exactly as torch.argmax",
            },
        },
    }


def transition_contract() -> dict[str, Any]:
    return {
        "comparisons": "raw versus alpha1 and raw versus every candidate alpha",
        "metrics": list(SAFETY_METRICS),
        "transition_matrix_keys": list(TRANSITION_KEYS),
        "matrix_meaning": {
            "cc": "raw correct, candidate correct",
            "cw": "raw correct, candidate wrong",
            "wc": "raw wrong, candidate correct",
            "ww": "raw wrong, candidate wrong",
        },
        "required_for_all_four_cells": {
            "count": True,
            "canonical_line_identity_digest": True,
            "digest_record_fields": [
                "panel",
                "member",
                "line_index_zero_based",
                "line_sha256",
                "context",
            ],
            "cc_and_ww_full_identity_expansion": False,
        },
        "expanded_changed_cells": {
            "cells": ["cw", "wc"],
            "full_line_identities": [
                "panel",
                "member",
                "line_index_zero_based",
                "line_sha256",
                "context",
            ],
            "raw_and_candidate_orders": True,
            "maximum_expanded_records": None,
            "no_truncation": True,
        },
        "additional_row_change_audits": [
            "exact greedy-sequence changed count and identities",
            "greedy-length changed count and identities",
            "top1-index changed count and identities",
            "prediction fingerprint SHA256 for every panel and alpha",
        ],
        "official_metric_semantics": {
            "argmax_ties_choose_lowest_option_index_exactly_as_torch_argmax": True,
            "count_mismatch_set_order_hybrid_semantics_exact_hash_bound_evaluator": True,
            "no_custom_correctness_shortcuts": True,
            "applicable_denominators": {
                panel: {
                    "set_exact": EXPECTED_TRAIN[panel]["rows"],
                    "hybrid_order_exact": EXPECTED_TRAIN[panel]["rows"],
                    "ordered_exact": EXPECTED_TRAIN[panel]["rows"],
                    "top1_correct": EXPECTED_TRAIN[panel]["nonempty_rows"],
                    "context34_hybrid_order_exact": EXPECTED_TRAIN[panel][
                        "context34_rows"
                    ],
                    "context34_ordered_exact": EXPECTED_TRAIN[panel][
                        "context34_rows"
                    ],
                }
                for panel in DATASETS
            },
        },
        "count_and_value": {
            "state_tensors_bit_exact_raw": True,
            "logits_must_be_tensor_exact_raw_for_every_row_and_alpha": True,
            "correctness_transitions_forbidden": True,
            "required_per_panel_and_alpha": {
                "raw_count_logits_identity_bound_contiguous_float32_fingerprint": True,
                "candidate_count_logits_identity_bound_contiguous_float32_fingerprint": True,
                "count_logits_mismatch_count": 0,
                "count_logits_mismatch_full_line_identities": True,
                "raw_value_logits_identity_bound_contiguous_float32_fingerprint": True,
                "candidate_value_logits_identity_bound_contiguous_float32_fingerprint": True,
                "value_logits_mismatch_count": 0,
                "value_logits_mismatch_full_line_identities": True,
                "predicted_count_fingerprint_exact_raw": True,
                "greedy_length_fingerprint_exact_raw": True,
                "count_correct_transition_matrix_and_fingerprint_exact_raw": True,
                "value_sign_fingerprint_exact_raw": True,
                "value_correct_transition_matrix_and_fingerprint_exact_raw": True,
            },
            "fingerprint_recipe": (
                "for each row in frozen identity order: canonical identity JSON, "
                "NUL, literal <f4, NUL, canonical JSON shape, NUL, CPU "
                "contiguous IEEE-754 float32 little-endian bytes, NUL; a zero-"
                "element tensor contributes its exact shape and zero data bytes"
            ),
            "all_mismatch_identity_lists_untruncated": True,
        },
        "hard_safety_gate": {
            "all_three_panels": True,
            "zero_correct_to_wrong_for_every_safety_metric": True,
            "context34_denominators": {
                panel: EXPECTED_TRAIN[panel]["context34_rows"] for panel in DATASETS
            },
            "aggregate_net_ties_do_not_satisfy_row_safety": True,
        },
        "alpha1_integrity_hard_gate": {
            "selectable": False,
            "all_three_panels_zero_cw_for_every_safety_metric": True,
            "count_and_value_logits_and_correctness_exact_raw": True,
            "failure_action": "CLOSE with no selected candidate",
        },
        "gain_gate": {
            "panel": "pokemonfan",
            "at_least_one_main_metric_net_gain": 1,
            "with_zero_cw_this_requires_wc_at_least_one": True,
            "eligible_metrics": list(MAIN_METRICS),
        },
        "selection": {
            "evaluate_raw_alpha1_and_all_eight_candidates_before_decision": True,
            "each_candidate_constructed_directly_from_raw": True,
            "intermediate_grid_points_are_not_a_trajectory": True,
            "candidate_does_not_require_smaller_alphas_to_be_safe": True,
            "eligible": (
                "alpha1 integrity gate passes, and this candidate's own hard "
                "safety gates and PokemonFan gain gate pass"
            ),
            "choose": "smallest eligible alpha",
            "maximum_selected_candidates": 1,
            "selected_candidate_is_not_materialized_by_probe": True,
            "separate_materialization_preregistration_required": True,
        },
        "fixed_evaluation_ledger": {
            "model_order": [
                "raw",
                "alpha1",
                "alpha2",
                "alpha4",
                "alpha8",
                "alpha16",
                "alpha32",
                "alpha64",
                "alpha128",
                "alpha256",
            ],
            "panel_order_per_model": ["flg", "pokemonfan", "core5"],
            "evaluation_count_exact": 30,
            "all_30_complete_before_any_selection_decision": True,
            "short_circuit_forbidden": True,
            "batch_protocol": {
                "dataset_yields": "(features, exact line identity)",
                "custom_collate": (
                    "separate identities, then call hash-bound evaluator.collate_ordered "
                    "on features only"
                ),
                "workers": 0,
                "same_batch_evaluation": (
                    "evaluate raw, alpha1, alpha2, alpha4, alpha8, alpha16, "
                    "alpha32, alpha64, alpha128, alpha256 sequentially before "
                    "advancing to the next batch"
                ),
                "row_order_and_padding_shared_across_all_ten_states": True,
            },
        },
        "transition_invariants": {
            "four_cells_sum_to_metric_applicable_denominator": True,
            "candidate_correct_minus_raw_correct_equals_wc_minus_cw": True,
            "top1_excludes_empty_expert_order_rows": True,
        },
        "official_summary_integrity": {
            "raw_and_alpha1_each_panel_must_exactly_match_frozen_gate_JSON": True,
            "comparison": "full nested metrics object exact equality",
            "scalar_leaf_count_per_panel_per_state": 572,
            "frozen_sources": {
                "raw": {
                    panel: f"raw_full_u468/{panel}.json" for panel in DATASETS
                },
                "alpha1": {
                    panel: f"sole_equalblend_endpoint/{panel}.json"
                    for panel in DATASETS
                },
            },
        },
    }


def probe_contract(
    self_evidence: Mapping[str, Any],
    fixed_inputs: Mapping[str, Any],
    cache: Mapping[str, Any],
    profile_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "contract_only_future_formal_revision_required",
        "branch": BRANCH,
        "design_artifact": str(DESIGN_ARTIFACT.relative_to(ROOT)),
        "supersedes_contract_only_design_artifacts": SUPERSEDED_DESIGN_ARTIFACTS,
        "runner": {
            "path": self_evidence["path"],
            "sha256": self_evidence["sha256"],
            "formal_enabled": False,
            "actual_enabled": False,
        },
        "scope": {
            "split": "train",
            "all_train_rows": True,
            "validation_member_payloads_opened": False,
            "validation_results_read": False,
            "training": False,
            "optimizer": False,
            "backward": False,
            "model_checkpoint_writes": 0,
            "candidate_states_RAM_only": True,
            "network": False,
            "broad": False,
            "gold": False,
            "package_upload_submission": False,
        },
        "fixed_bindings": fixed_inputs,
        "train_identity_cache": cache,
        "raw_profile_observations": profile_summary["observations"],
        "candidate_design": candidate_design(),
        "threshold_analysis": threshold_contract(),
        "row_transition_evidence": transition_contract(),
        "future_formal_requirements": {
            "separate_hash_frozen_runner": True,
            "future_formal_runner_path": str(FUTURE_FORMAL_RUNNER.relative_to(ROOT)),
            "contract_only_runner_is_frozen_before_formal_runner_is_bound": True,
            "formal_runner_binds_visible_contract_only_runner_SHA256": True,
            "formal_runner_binds_frozen_design_artifact_SHA256": True,
            "frozen_design_artifact_path": str(DESIGN_ARTIFACT.relative_to(ROOT)),
            "exact_external_preregistration_SHA256_required": True,
            "O_EXCL_one_shot_attempt_marker_before_model_instantiation": True,
            "CUDA_available_before_marker": True,
            "my_project_env_python_I_B": True,
            "fixed_inputs_and_train_cache_reverified_at_lock": True,
            "at_lock_binary_reads": (
                "O_NOFOLLOW read-only fd, pre/post fstat identity, visible lstat "
                "identity, exact SHA; fail on any TOCTOU drift"
            ),
            "each_alpha_state_integrity": [
                "all model tensors finite",
                "exact model-state fingerprint recorded",
                "changed tensor set exact actor6",
                "every non-actor tensor bit-exact raw",
            ],
            "result_schema_and_every_line_identity_frozen_before_execution": True,
            "result_publication": "one O_EXCL JSON result, then mode 0444",
            "future_attempt_marker": str(FUTURE_ATTEMPT_MARKER.relative_to(ROOT)),
            "future_result": str(FUTURE_RESULT.relative_to(ROOT)),
            "future_formal_preregistration": str(
                FORMAL_PREREGISTRATION.relative_to(ROOT)
            ),
            "validation_member_payloads_opened": False,
            "model_artifact_writes": 0,
            "submission_performed": False,
        },
    }


def outputs_absent() -> dict[str, bool]:
    return {
        "future_formal_preregistration_absent": not (
            FORMAL_PREREGISTRATION.exists() or FORMAL_PREREGISTRATION.is_symlink()
        ),
        "future_attempt_marker_absent": not (
            FUTURE_ATTEMPT_MARKER.exists() or FUTURE_ATTEMPT_MARKER.is_symlink()
        ),
        "future_result_absent": not (
            FUTURE_RESULT.exists() or FUTURE_RESULT.is_symlink()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("static-audit", "cache-audit", "contract"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    static = ast_audit()
    fixed_inputs, profile_payload = verify_fixed_inputs()
    absence = outputs_absent()
    if not all(absence.values()):
        raise FileExistsError(f"reserved future probe outputs already exist: {absence}")
    if args.mode == "static-audit":
        result = {
            "schema_version": SCHEMA,
            "status": "zero_write_static_audit_passed",
            "ast_audit": static,
            "fixed_inputs": fixed_inputs,
            "raw_profile_observations": profile_payload["observations"],
            "candidate_design": candidate_design(),
            "outputs_absent": absence,
            "actual_or_formal_executed": False,
            "validation_member_payloads_opened": False,
            "writes_performed": False,
        }
    else:
        cache = cache_audit(profile_payload["profile"])
        if args.mode == "cache-audit":
            result = {
                "schema_version": SCHEMA,
                "status": "zero_write_cache_audit_passed",
                "ast_audit": static,
                "fixed_inputs": fixed_inputs,
                "cache": cache,
                "outputs_absent": absence,
                "actual_or_formal_executed": False,
                "validation_member_payloads_opened": False,
                "writes_performed": False,
            }
        else:
            result = {
                "schema_version": PREREGISTRATION_SCHEMA,
                "status": "contract_only_future_formal_revision_required",
                "probe_contract": probe_contract(
                    fixed_inputs["self"],
                    fixed_inputs,
                    cache,
                    profile_payload,
                ),
            }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
