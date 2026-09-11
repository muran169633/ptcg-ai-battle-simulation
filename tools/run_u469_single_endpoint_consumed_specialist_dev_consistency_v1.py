#!/usr/bin/env python3
"""Run frozen U469 on six consumed specialist panels as dev consistency.

These specialist-valid panels have already been consumed by earlier U468
development.  The source two-update training branch also failed as a whole:
U470 crossed the BC-anchor-KL cap and the trainer child omitted the design-
listed ``-I -B`` flags.  U469 itself passed its endpoint-local numerical gates
and was explicitly retained for a separately preregistered future experiment.

This one-shot run is therefore diagnostic consistency evidence only, never
promotion evidence.  A pass cannot cleanse the source-training taint or
authorize broad, Gold, packaging, upload, or submission.

The executor validates one hash-bound preregistration, consumes an attempt
marker before the first panel, attempts all six panels exactly once in their
declared order, requires all 60 gates, and rehashes every immutable input both
before and after execution.  It performs no training or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "run_u469_single_endpoint_consumed_specialist_dev_consistency_v1.py"
BUILDER = TOOLS / "build_u469_consumed_specialist_dev_consistency_preregistration_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_MODE = 0o555

LEGACY = TOOLS / "run_e904_single_endpoint_specialist.py"
LEGACY_SHA256 = "f749fd660209e2369d6cdf21e91ca0a5234eb3306d50af98165cac502f37e6c7"

PREREGISTRATION_SCHEMA = (
    "ptcg-u469-single-endpoint-consumed-specialist-dev-consistency-preregistration-v1"
)
MANIFEST_SCHEMA = (
    "ptcg-u469-single-endpoint-consumed-specialist-dev-consistency-manifest-v1"
)
DECISION_SCHEMA = (
    "ptcg-u469-single-endpoint-consumed-specialist-dev-consistency-decision-v1"
)
ATTEMPT_SCHEMA = (
    "ptcg-u469-single-endpoint-consumed-specialist-dev-consistency-attempt-v1"
)
PREREGISTRATION_STATUS = "locked_before_specialist_dev_consistency"

ARTIFACT_ID = "u469_single_endpoint_consumed_specialist_dev_consistency_20260802_v1"
PREREGISTRATION = ROOT / "artifacts" / (
    f"{ARTIFACT_ID}.preregistration.json"
)
OUTPUT_ROOT = ROOT / "artifacts" / (
    f"{ARTIFACT_ID}.specialist_dev_consistency"
)
ATTEMPT_MARKER = ROOT / "artifacts" / (
    f".ptcg-{ARTIFACT_ID}-specialist-dev-consistency-attempt.json"
)

CANDIDATE_CHECKPOINT = ROOT / (
    "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110/"
    "B_gold_league/seed-202607336/checkpoints/update-0469.pt"
)
CANDIDATE_CHECKPOINT_SHA256 = (
    "ca0d63ca0c61a8f3d09788d1861f4a6449b711ab85ac96aa986188f2ea4f3a3f"
)
CANDIDATE_MODEL_STATE_SHA256 = (
    "69223aa0d2559e27b84d094370a3fea7608f118a0b93f631fbafbde7602382b6"
)
CANDIDATE_CHECKPOINT_UPDATE = 469
RAW_CHECKPOINT_UPDATE = 468

TRAINING_ROOT = ROOT / "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110"
TRAINING_DECISION = ROOT / (
    "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110."
    "training_integrity_decision.json"
)
TRAINING_DESIGN = ROOT / (
    "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110."
    "design_preregistration.json"
)
TRAINING_EXECUTION = ROOT / (
    "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110."
    "execution_preregistration.json"
)
TRAINING_RUN_CONFIG = TRAINING_ROOT / "B_gold_league/seed-202607336/run_config.json"
TRAINING_METRICS = TRAINING_ROOT / "B_gold_league/seed-202607336/metrics.jsonl"
TRAINING_SUMMARY = TRAINING_ROOT / "B_gold_league/seed-202607336/summary.json"
TRAINING_LOG = ROOT / (
    "artifacts/ppo_u468_exactresume_ppo2_u469u470_design202608110.log"
)
TRAINING_ATTEMPT = ROOT / (
    ".ptcg-u468-exactresume-ppo2-u470-attempt-202607336-202608110.json"
)
TRAINING_LAUNCH_CLAIM = TRAINING_ROOT / ".u464_to_u468_launch_claim.json"

TAINTED_TRAINING_PROVENANCE: dict[str, tuple[Path, str]] = {
    "training_decision": (
        TRAINING_DECISION,
        "65515793edc70ab49c9bc81db1d5b59313bf3d5f27cfdc6699c2aa2314ed76dc",
    ),
    "design_preregistration": (
        TRAINING_DESIGN,
        "ebe4d1f789aad1ff1c4e0d4f110db87ee83a4b8cd34aea09dd9cd6b350f23035",
    ),
    "execution_preregistration": (
        TRAINING_EXECUTION,
        "a6290a654a2b5a681f0f3580abf9647107cd9ae16ed26d64ca23c613f2bae1bd",
    ),
    "run_config": (
        TRAINING_RUN_CONFIG,
        "ba56c122f9da26c3663609e617d84314c9e1570b929dbb7fe27734adf217c58b",
    ),
    "metrics": (
        TRAINING_METRICS,
        "43dce0b527cdaf993f473154222b5ffaf28b9dff4ab44eb54461d66e29d94487",
    ),
    "summary": (
        TRAINING_SUMMARY,
        "70844c7d277348277ae2fbba42a8e64140bc2f9dbc54f0cd82d3f153e31ed2b2",
    ),
    "log": (
        TRAINING_LOG,
        "73eb32663471cb12b1076b17744f4cd315c00d37cb84794c6ef27e53d3c9f8de",
    ),
    "attempt_marker": (
        TRAINING_ATTEMPT,
        "0d84fcd6493a0bf3f3d6a38e4d420de889f918635c061f21141992f8951fcab5",
    ),
    "launch_claim": (
        TRAINING_LAUNCH_CLAIM,
        "0d84fcd6493a0bf3f3d6a38e4d420de889f918635c061f21141992f8951fcab5",
    ),
}

RAW_U468_CHECKPOINT = ROOT / (
    "artifacts/ppo_u464_g8generalbc_then_ppo4_then_s32eqp12_design202608090/"
    "ppo_stage/B_gold_league/seed-202607336/checkpoints/update-0468.pt"
)
RAW_U468_CHECKPOINT_SHA256 = (
    "11ccd396b5c6453afe23c7595aa7ef8bf50ad3aa311aa8039c92d2f372ed532f"
)
RAW_U468_MODEL_STATE_SHA256 = (
    "6fff0b2d6cb83eae47bacd1be0c4bfb1cd38a70c37bb290a2e3f5e3ae5e9d512"
)

CLASSIFICATION = {
    "specialist_valid_consumed": True,
    "dev_consistency_only": True,
    "promotion_evidence": False,
    "source_training_integrity_failed": True,
    "source_training_taint_preserved": True,
}
SCOPE = {
    "local_only": True,
    "network": False,
    "training": False,
    "specialist_behavior": True,
    "specialist_valid_consumed": True,
    "dev_consistency_only": True,
    "promotion_evidence": False,
    "source_training_taint_preserved": True,
    "broad": False,
    "gold": False,
    "package": False,
    "upload": False,
    "submission": False,
}
AUTHORIZATION = {
    "broad_behavior_authorized": False,
    "gold_authorized": False,
    "package_authorized": False,
    "upload_authorized": False,
    "submission_authorized": False,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def read_frozen_source(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
            or stat.S_IMODE(before.st_mode) != FROZEN_MODE
        ):
            raise RuntimeError(f"{label} must be a frozen single-link file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    digest = sha256_bytes(payload)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != int(after.st_size)
        or digest != expected_sha256
    ):
        raise RuntimeError(f"{label} identity drift")
    return payload, {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "bytes": len(payload),
        "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
    }


def import_legacy() -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_frozen_source(LEGACY, LEGACY_SHA256, "legacy specialist executor")
    spec = importlib.util.spec_from_file_location(
        f"_u469_consumed_dev_legacy_{LEGACY_SHA256[:12]}", LEGACY
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen legacy specialist executor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


legacy, LEGACY_EVIDENCE = import_legacy()
ProtocolError = legacy.ProtocolError


def root_relative_text(path: Path) -> str:
    return legacy.root_relative_text(path)


def expected_candidate_binding() -> dict[str, Any]:
    return {
        "path": root_relative_text(CANDIDATE_CHECKPOINT),
        "sha256": CANDIDATE_CHECKPOINT_SHA256,
        "model_state_sha256": CANDIDATE_MODEL_STATE_SHA256,
        "checkpoint_update": CANDIDATE_CHECKPOINT_UPDATE,
    }


def expected_raw_binding() -> dict[str, Any]:
    return {
        "path": root_relative_text(RAW_U468_CHECKPOINT),
        "sha256": RAW_U468_CHECKPOINT_SHA256,
        "model_state_sha256": RAW_U468_MODEL_STATE_SHA256,
        "checkpoint_update": RAW_CHECKPOINT_UPDATE,
    }


def expected_provenance_bindings() -> dict[str, dict[str, str]]:
    return {
        name: {"path": root_relative_text(path), "sha256": digest}
        for name, (path, digest) in TAINTED_TRAINING_PROVENANCE.items()
    }


def verify_candidate_checkpoint(
    payload: bytes,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        checkpoint = legacy.torch.load(
            legacy.io.BytesIO(payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ProtocolError("U469 candidate checkpoint cannot be loaded on CPU") from error
    if not isinstance(checkpoint, dict):
        raise ProtocolError("U469 candidate checkpoint must contain a dict")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ProtocolError("U469 candidate has no model_state_dict mapping")
    observed_model_sha = legacy.model_state_sha256(state)
    checks = {
        "file_sha_declared_exact": binding.get("sha256")
        == CANDIDATE_CHECKPOINT_SHA256,
        "model_sha_declared_exact": binding.get("model_state_sha256")
        == CANDIDATE_MODEL_STATE_SHA256,
        "model_sha_observed_exact": observed_model_sha
        == CANDIDATE_MODEL_STATE_SHA256,
        "checkpoint_update_declared_exact_469": binding.get("checkpoint_update")
        == CANDIDATE_CHECKPOINT_UPDATE,
        "checkpoint_update_observed_exact_469": checkpoint.get("update")
        == CANDIDATE_CHECKPOINT_UPDATE,
        "full_ppo_optimizer_present": isinstance(
            checkpoint.get("optimizer_state_dict"), Mapping
        ),
        "full_bc_replay_optimizer_present": isinstance(
            checkpoint.get("bc_replay_optimizer_state_dict"), Mapping
        ),
        "opponent_quota_state_present": isinstance(
            checkpoint.get("opponent_quota_state"), Mapping
        ),
        "not_marked_evaluation_only": checkpoint.get("evaluation_only") is not True,
        "not_marked_resume_forbidden": checkpoint.get("resume_forbidden") is not True,
    }
    if not all(checks.values()):
        raise ProtocolError(f"U469 candidate full-checkpoint drift: {checks}")
    return {
        "model_state_sha256": observed_model_sha,
        "checkpoint_update": checkpoint["update"],
        "model_tensor_count": len(state),
        "full_checkpoint_checks": checks,
    }


def verify_tainted_training_provenance(
    declared: Any,
) -> tuple[dict[str, Any], dict[str, Path]]:
    expected = expected_provenance_bindings()
    if not isinstance(declared, Mapping) or dict(declared) != expected:
        raise ProtocolError("tainted training provenance binding drift")
    evidence: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, binding in expected.items():
        path, payload, observed = legacy.verify_binding(
            binding,
            f"tainted training provenance {name}",
            expected_path=ROOT / binding["path"],
            expected_sha256=binding["sha256"],
        )
        paths[name] = path
        evidence[name] = observed
        if name == "training_decision":
            decision = legacy.strict_json_loads(payload, "training integrity decision")

    candidate = expected_candidate_binding()
    raw = expected_raw_binding()
    endpoint_integrity = decision.get("endpoint_integrity")
    u469 = endpoint_integrity[0] if isinstance(endpoint_integrity, list) and len(endpoint_integrity) == 2 else {}
    u470 = endpoint_integrity[1] if isinstance(endpoint_integrity, list) and len(endpoint_integrity) == 2 else {}
    frozen = decision.get("frozen_gate_decision", {})
    isolated = frozen.get("implementation_isolated_flags", {}) if isinstance(frozen, Mapping) else {}
    strict = decision.get("strict_interpretation", {})
    provenance_checks = {
        "decision_schema_exact": decision.get("schema_version")
        == "ptcg-u468-exactresume-ppo2-u469-u470-training-integrity-decision-v1",
        "whole_source_branch_failed_exact": decision.get("status")
        == "failed_bc_anchor_kl_gate_whole_branch_closed_before_behavior",
        "design_binding_exact": decision.get("design_preregistration")
        == expected["design_preregistration"],
        "execution_binding_exact": {
            key: decision.get("execution_preregistration", {}).get(key)
            for key in ("path", "sha256")
        }
        == expected["execution_preregistration"],
        "run_config_binding_exact": decision.get("run_artifacts", {}).get("run_config")
        == expected["run_config"],
        "metrics_binding_exact": {
            key: decision.get("run_artifacts", {}).get("metrics", {}).get(key)
            for key in ("path", "sha256")
        }
        == expected["metrics"],
        "summary_binding_exact": {
            key: decision.get("run_artifacts", {}).get("summary", {}).get(key)
            for key in ("path", "sha256")
        }
        == expected["summary"],
        "log_binding_exact": decision.get("transport", {}).get("log")
        == {
            "path": expected["log"]["path"],
            "sha256": expected["log"]["sha256"],
        },
        "attempt_binding_exact": decision.get("transport", {}).get("attempt_marker")
        == expected["attempt_marker"],
        "launch_claim_binding_exact": decision.get("transport", {}).get("launch_claim")
        == expected["launch_claim"],
        "raw_u468_source_exact": decision.get("source_parent")
        == {
            "path": raw["path"],
            "sha256_before_and_after": raw["sha256"],
            "runtime_model_state_sha256": raw["model_state_sha256"],
        },
        "u469_checkpoint_exact": {
            "update": u469.get("update"),
            "path": u469.get("path"),
            "sha256": u469.get("sha256"),
            "runtime_model_state_sha256": u469.get("runtime_model_state_sha256"),
        }
        == {
            "update": candidate["checkpoint_update"],
            "path": candidate["path"],
            "sha256": candidate["sha256"],
            "runtime_model_state_sha256": candidate["model_state_sha256"],
        },
        "u469_endpoint_local_integrity_passed": u469.get("endpoint_local_integrity")
        == "passed",
        "u469_anchor_kl_within_cap": u469.get("bc_anchor_kl")
        == 0.01343894525341693,
        "u469_optimizer_dry_load_passed": u469.get("optimizer_dry_load")
        == "passed",
        "u470_quarantined_endpoint_failed": u470.get("update") == 470
        and u470.get("endpoint_local_integrity") == "failed_bc_anchor_kl_gate"
        and u470.get("bc_anchor_kl") == 0.015720198649913072,
        "source_child_was_my_project_env": isolated.get(
            "environment_binary_is_my_project_env"
        )
        is True,
        "source_child_missing_isolated_flags_preserved": isolated.get(
            "actual_child_argv_contains_required_flags"
        )
        is False
        and isolated.get("pass") is False,
        "source_overall_integrity_failed": frozen.get("overall_training_integrity")
        == "failed",
        "old_design_did_not_authorize_u469_direct_eval": strict.get(
            "u469_direct_specialist_evaluation_authorized"
        )
        is False,
        "future_separate_preregistration_explicitly_allowed": strict.get(
            "u469_preserved_for_read_only_diagnostics_or_a_future_separately_preregistered_experiment"
        )
        is True
        and strict.get("next_experiment_requires_new_preregistration") is True,
        "u470_quarantine_preserved": strict.get(
            "u470_quarantined_from_all_downstream_use"
        )
        is True,
        "old_downstream_authorizations_all_false": all(
            strict.get(name) is False
            for name in (
                "broad_authorized",
                "gold_authorized",
                "package_upload_or_submission_authorized",
            )
        ),
    }
    if not all(provenance_checks.values()):
        raise ProtocolError(
            f"tainted training provenance content drift: {provenance_checks}"
        )
    evidence["checks"] = provenance_checks
    evidence["classification"] = {
        "source_training_integrity_failed": True,
        "source_training_taint_preserved": True,
        "u469_endpoint_local_integrity_passed": True,
        "u470_quarantined": True,
    }
    return evidence, paths


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise ProtocolError("executor must run from the repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ProtocolError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise ProtocolError("executor requires Python flags -I -B")


def binding_for(path: Path, label: str) -> dict[str, str]:
    _, evidence = legacy.read_plain_file(path, label)
    return {"path": root_relative_text(path), "sha256": evidence["sha256"]}


def absent_by_lstat(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    return False


def validate_preregistration(
    preregistration_path: Path,
    expected_preregistration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_payload, prereg_evidence = legacy.read_plain_file(
        preregistration_path, "U469 consumed dev-consistency preregistration"
    )
    expected_digest = legacy.require_sha256(
        expected_preregistration_sha256,
        "expected preregistration SHA-256",
    )
    if prereg_evidence["sha256"] != expected_digest:
        raise ProtocolError("preregistration SHA-256 mismatch")
    prereg = legacy.strict_json_loads(prereg_payload, "preregistration")
    if prereg.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ProtocolError("unsupported dev-consistency preregistration schema")
    if prereg.get("status") != PREREGISTRATION_STATUS:
        raise ProtocolError("dev-consistency preregistration is not locked")
    if prereg.get("classification") != CLASSIFICATION:
        raise ProtocolError("dev-consistency classification drift")
    if prereg.get("authorization") != AUTHORIZATION or any(
        bool(value) for value in prereg.get("authorization", {}).values()
    ):
        raise ProtocolError("all preregistered downstream authorizations must be false")

    executor_path, _, executor_evidence = legacy.verify_binding(
        prereg.get("executor"), "executor", expected_path=SCRIPT
    )
    executor = prereg["executor"]
    if executor.get("python") != str(EXPECTED_PYTHON) or executor.get("flags") != [
        "-I",
        "-B",
    ]:
        raise ProtocolError("executor Python/flags binding drift")
    legacy_path, _, legacy_evidence = legacy.verify_binding(
        prereg.get("legacy_specialist_implementation"),
        "legacy specialist implementation",
        expected_path=LEGACY,
        expected_sha256=LEGACY_SHA256,
    )
    generator_path, _, generator_evidence = legacy.verify_binding(
        prereg.get("generator"), "generator", expected_path=BUILDER
    )

    evaluator = prereg.get("evaluator")
    evaluator_path, _, evaluator_evidence = legacy.verify_binding(
        evaluator,
        "evaluator",
        expected_path=legacy.EXPECTED_EVALUATOR,
        expected_sha256=legacy.EXPECTED_EVALUATOR_SHA256,
    )
    if evaluator.get("python") != str(EXPECTED_PYTHON) or evaluator.get("flags") != [
        "-I",
        "-B",
    ]:
        raise ProtocolError("evaluator Python/flags binding drift")
    dependencies = prereg.get("evaluator_dependencies")
    if not isinstance(dependencies, Mapping) or set(dependencies) != set(
        legacy.EXPECTED_DEPENDENCIES
    ):
        raise ProtocolError("evaluator dependency set mismatch")
    dependency_paths: dict[str, Path] = {}
    dependency_evidence: dict[str, Any] = {}
    for name, (path, digest) in legacy.EXPECTED_DEPENDENCIES.items():
        dep_path, _, dep_evidence = legacy.verify_binding(
            dependencies[name],
            f"evaluator dependency {name}",
            expected_path=path,
            expected_sha256=digest,
        )
        dependency_paths[name] = dep_path
        dependency_evidence[name] = dep_evidence

    candidate = prereg.get("candidate")
    if not isinstance(candidate, Mapping) or dict(candidate) != expected_candidate_binding():
        raise ProtocolError("candidate binding differs from frozen U469 checkpoint")
    checkpoint, checkpoint_payload, checkpoint_evidence = legacy.verify_binding(
        candidate,
        "U469 candidate checkpoint",
        expected_path=CANDIDATE_CHECKPOINT,
        expected_sha256=CANDIDATE_CHECKPOINT_SHA256,
    )
    checkpoint_evidence.update(verify_candidate_checkpoint(checkpoint_payload, candidate))

    raw_parent = prereg.get("raw_u468_parent")
    if not isinstance(raw_parent, Mapping) or dict(raw_parent) != expected_raw_binding():
        raise ProtocolError("raw U468 parent binding drift")
    raw_path, raw_payload, raw_evidence = legacy.verify_binding(
        raw_parent,
        "raw U468 parent checkpoint",
        expected_path=RAW_U468_CHECKPOINT,
        expected_sha256=RAW_U468_CHECKPOINT_SHA256,
    )
    raw_evidence.update(
        legacy.load_checkpoint_identity(
            raw_payload,
            raw_parent,
            expected_model_state_sha256=RAW_U468_MODEL_STATE_SHA256,
            label="raw U468 parent",
        )
    )
    provenance_evidence, provenance_paths = verify_tainted_training_provenance(
        prereg.get("tainted_training_provenance")
    )

    if prereg.get("shared_protocol") != legacy.SHARED_PROTOCOL:
        raise ProtocolError("shared six-panel protocol mismatch")
    expected_gates = {
        name: legacy.expected_thresholds(name) for name in legacy.BASELINE_METRICS
    }
    if prereg.get("authoritative_panel_gates") != expected_gates:
        raise ProtocolError("authoritative 60 gates drift")

    data_bindings = prereg.get("data_bindings")
    baseline_bindings = prereg.get("baseline_bindings")
    if not isinstance(data_bindings, Mapping) or set(data_bindings) != {
        "pokemonfan",
        "flg",
        "core5",
    }:
        raise ProtocolError("specialist data binding set mismatch")
    if not isinstance(baseline_bindings, Mapping) or set(baseline_bindings) != {
        panel["name"] for panel in legacy.PANEL_SPECS
    }:
        raise ProtocolError("specialist baseline binding set mismatch")
    data_evidence: dict[str, Any] = {}
    baseline_evidence: dict[str, Any] = {}
    for panel in legacy.PANEL_SPECS:
        data_name = str(panel["data_name"])
        if data_name not in data_evidence:
            _, _, evidence = legacy.verify_binding(
                data_bindings[data_name],
                f"data {data_name}",
                expected_path=ROOT / str(panel["data_path"]),
                expected_sha256=str(panel["data_sha256"]),
            )
            data_evidence[data_name] = evidence
        baseline_path, baseline_payload, evidence = legacy.verify_binding(
            baseline_bindings[str(panel["name"])],
            f"baseline {panel['name']}",
            expected_path=ROOT / str(panel["baseline_path"]),
            expected_sha256=str(panel["baseline_sha256"]),
        )
        baseline = legacy.strict_json_loads(
            baseline_payload, f"baseline {panel['name']}"
        )
        legacy.validate_output_identity(
            baseline,
            checkpoint=raw_path,
            checkpoint_sha256=RAW_U468_CHECKPOINT_SHA256,
            panel=panel,
        )
        metrics = legacy.observed_metrics(baseline)
        if metrics != legacy.BASELINE_METRICS[str(panel["name"])]:
            raise ProtocolError(f"baseline metric drift for {panel['name']}")
        evidence["metrics"] = metrics
        evidence["resolved_path"] = str(baseline_path)
        baseline_evidence[str(panel["name"])] = evidence

    output_rule = prereg.get("output_rule")
    manifest_path = OUTPUT_ROOT / "dev_consistency_execution_manifest.json"
    decision_path = OUTPUT_ROOT / "dev_consistency_decision.json"
    expected_output_rule = {
        "root": root_relative_text(OUTPUT_ROOT),
        "root_absent_at_lock": True,
        "attempt_marker": root_relative_text(ATTEMPT_MARKER),
        "attempt_marker_absent_at_lock": True,
        "evaluation_count_exact": 6,
        "gate_count_exact": 60,
        "attempts_per_evaluation": 1,
        "retry_authorized": False,
        "run_all_before_decision": True,
        "manifest": manifest_path.name,
        "decision": decision_path.name,
        "final_directory_mode": "0o500",
    }
    if output_rule != expected_output_rule:
        raise ProtocolError("dev-consistency one-shot output rule drift")

    evaluations = prereg.get("ordered_evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 6:
        raise ProtocolError("ordered evaluations must contain exactly six panels")
    output_paths: list[Path] = []
    for order, (evaluation, panel) in enumerate(
        zip(evaluations, legacy.PANEL_SPECS, strict=True), start=1
    ):
        output = OUTPUT_ROOT / f"{panel['name']}.json"
        command = legacy.expected_command(checkpoint, panel, output)
        expected_evaluation = {
            "order": order,
            "panel": panel["name"],
            "team_name": panel["team_name"],
            "output": root_relative_text(output),
            "output_absent_at_lock": True,
            "command": command,
            "command_sha256": sha256_bytes(canonical_json_bytes(command)),
            "attempts_authorized": 1,
        }
        if not isinstance(evaluation, Mapping) or dict(evaluation) != expected_evaluation:
            raise ProtocolError(f"evaluation {order} command drift")
        output_paths.append(output)
    if prereg.get("scope") != SCOPE:
        raise ProtocolError("dev-consistency scope drift")

    immutable_bindings: list[dict[str, Any]] = [
        {
            "label": "preregistration",
            "path": preregistration_path,
            "sha256": prereg_evidence["sha256"],
        },
        {"label": "executor", "path": executor_path, "sha256": executor_evidence["sha256"]},
        {"label": "generator", "path": generator_path, "sha256": generator_evidence["sha256"]},
        {"label": "legacy specialist implementation", "path": legacy_path, "sha256": legacy_evidence["sha256"]},
        {"label": "evaluator", "path": evaluator_path, "sha256": evaluator_evidence["sha256"]},
        {"label": "candidate checkpoint", "path": checkpoint, "sha256": checkpoint_evidence["sha256"]},
        {"label": "raw U468 parent checkpoint", "path": raw_path, "sha256": raw_evidence["sha256"]},
    ]
    immutable_bindings.extend(
        {
            "label": f"tainted training provenance {name}",
            "path": provenance_paths[name],
            "sha256": expected_provenance_bindings()[name]["sha256"],
        }
        for name in sorted(provenance_paths)
    )
    immutable_bindings.extend(
        {
            "label": f"evaluator dependency {name}",
            "path": dependency_paths[name],
            "sha256": dependency_evidence[name]["sha256"],
        }
        for name in sorted(dependency_paths)
    )
    immutable_bindings.extend(
        {
            "label": f"data {name}",
            "path": ROOT / str(data_bindings[name]["path"]),
            "sha256": data_evidence[name]["sha256"],
        }
        for name in sorted(data_evidence)
    )
    immutable_bindings.extend(
        {
            "label": f"baseline {name}",
            "path": ROOT / str(baseline_bindings[name]["path"]),
            "sha256": baseline_evidence[name]["sha256"],
        }
        for name in sorted(baseline_evidence)
    )
    if len(immutable_bindings) != 27:
        raise ProtocolError("immutable input binding count must be exactly 27")

    return prereg, {
        "preregistration": prereg_evidence,
        "executor": executor_evidence,
        "generator": generator_evidence,
        "legacy_specialist_implementation": legacy_evidence,
        "evaluator": evaluator_evidence,
        "evaluator_dependencies": dependency_evidence,
        "candidate": checkpoint_evidence,
        "raw_u468_parent": raw_evidence,
        "tainted_training_provenance": provenance_evidence,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": checkpoint_evidence["sha256"],
        "data": data_evidence,
        "baselines": baseline_evidence,
        "output_root": OUTPUT_ROOT,
        "marker": ATTEMPT_MARKER,
        "manifest_path": manifest_path,
        "decision_path": decision_path,
        "output_paths": output_paths,
        "immutable_bindings": immutable_bindings,
    }


def require_all_outputs_absent(evidence: Mapping[str, Any]) -> None:
    legacy.require_absent(evidence["output_root"], "dev-consistency output root")
    legacy.require_absent(evidence["marker"], "dev-consistency attempt marker")
    for path in (
        evidence["manifest_path"],
        evidence["decision_path"],
        *evidence["output_paths"],
    ):
        legacy.require_absent(path, "dev-consistency artifact")


def static_audit() -> dict[str, Any]:
    self_payload, self_evidence = legacy.read_plain_file(SCRIPT, "U469 dev executor")
    del self_payload
    candidate_path, candidate_payload, candidate_evidence = legacy.verify_binding(
        expected_candidate_binding(),
        "static U469 candidate checkpoint",
        expected_path=CANDIDATE_CHECKPOINT,
        expected_sha256=CANDIDATE_CHECKPOINT_SHA256,
    )
    candidate_evidence.update(
        verify_candidate_checkpoint(candidate_payload, expected_candidate_binding())
    )
    raw_path, raw_payload, raw_evidence = legacy.verify_binding(
        expected_raw_binding(),
        "static raw U468 parent",
        expected_path=RAW_U468_CHECKPOINT,
        expected_sha256=RAW_U468_CHECKPOINT_SHA256,
    )
    raw_evidence.update(
        legacy.load_checkpoint_identity(
            raw_payload,
            expected_raw_binding(),
            expected_model_state_sha256=RAW_U468_MODEL_STATE_SHA256,
            label="static raw U468 parent",
        )
    )
    provenance, _ = verify_tainted_training_provenance(
        expected_provenance_bindings()
    )
    evaluator = binding_for(legacy.EXPECTED_EVALUATOR, "static evaluator")
    if evaluator["sha256"] != legacy.EXPECTED_EVALUATOR_SHA256:
        raise ProtocolError("static evaluator SHA drift")
    dependencies = {
        name: binding_for(path, f"static dependency {name}")
        for name, (path, _) in legacy.EXPECTED_DEPENDENCIES.items()
    }
    for name, (_, digest) in legacy.EXPECTED_DEPENDENCIES.items():
        if dependencies[name]["sha256"] != digest:
            raise ProtocolError(f"static dependency drift: {name}")
    data = {
        str(panel["data_name"]): binding_for(
            ROOT / str(panel["data_path"]), f"static data {panel['data_name']}"
        )
        for panel in legacy.PANEL_SPECS
    }
    baselines = {
        str(panel["name"]): binding_for(
            ROOT / str(panel["baseline_path"]), f"static baseline {panel['name']}"
        )
        for panel in legacy.PANEL_SPECS
    }
    gates = {
        name: legacy.expected_thresholds(name) for name in legacy.BASELINE_METRICS
    }
    checks = {
        "self_mode_0555": self_evidence["mode_octal"] == "0555",
        "legacy_executor_hash_exact": LEGACY_EVIDENCE["sha256"] == LEGACY_SHA256,
        "candidate_path_exact": candidate_path == CANDIDATE_CHECKPOINT,
        "candidate_model_exact": candidate_evidence["model_state_sha256"]
        == CANDIDATE_MODEL_STATE_SHA256,
        "raw_path_exact": raw_path == RAW_U468_CHECKPOINT,
        "raw_model_exact": raw_evidence["model_state_sha256"]
        == RAW_U468_MODEL_STATE_SHA256,
        "six_panels_exact": len(legacy.PANEL_SPECS) == 6,
        "sixty_gates_exact": sum(len(value) for value in gates.values()) == 60,
        "classification_exact": CLASSIFICATION
        == {
            "specialist_valid_consumed": True,
            "dev_consistency_only": True,
            "promotion_evidence": False,
            "source_training_integrity_failed": True,
            "source_training_taint_preserved": True,
        },
        "all_downstream_authorizations_false": not any(AUTHORIZATION.values()),
        "output_root_absent_by_lstat": absent_by_lstat(OUTPUT_ROOT),
        "attempt_marker_absent_by_lstat": absent_by_lstat(ATTEMPT_MARKER),
    }
    if not all(checks.values()):
        raise ProtocolError(f"U469 dev executor static audit failed: {checks}")
    return {
        "schema_version": "ptcg-u469-consumed-specialist-dev-consistency-static-v1",
        "status": "static_zero_write_audit_only",
        "self": self_evidence,
        "legacy_specialist_implementation": LEGACY_EVIDENCE,
        "candidate": candidate_evidence,
        "raw_u468_parent": raw_evidence,
        "tainted_training_provenance": provenance,
        "evaluator": evaluator,
        "evaluator_dependencies": dependencies,
        "data_bindings": data,
        "baseline_bindings": baselines,
        "authoritative_panel_gates": gates,
        "classification": CLASSIFICATION,
        "authorization": AUTHORIZATION,
        "checks": checks,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }


def validate_u469_output_identity(
    document: Mapping[str, Any],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    panel: Mapping[str, Any],
) -> None:
    """Validate evaluator identity without legacy's hard-coded update 468."""

    if Path(str(document.get("checkpoint"))).resolve() != checkpoint:
        raise ProtocolError("U469 evaluator output checkpoint path mismatch")
    if document.get("checkpoint_sha256") != checkpoint_sha256:
        raise ProtocolError("U469 evaluator output checkpoint SHA-256 mismatch")
    if document.get("checkpoint_update") != CANDIDATE_CHECKPOINT_UPDATE:
        raise ProtocolError("U469 evaluator output checkpoint update mismatch")
    if Path(str(document.get("data"))).resolve() != (
        ROOT / str(panel["data_path"])
    ).resolve():
        raise ProtocolError("U469 evaluator output data path mismatch")
    expected_scalars = {
        "split": "valid",
        "split_mode": "archive",
        "split_seed": 20260723,
        "device": "cuda",
        "prediction_order": "policy_greedy",
        "max_rows": None,
        "evaluator": "tools/evaluate_policy_bc.py",
    }
    for key, expected in expected_scalars.items():
        if document.get(key) != expected:
            raise ProtocolError(f"U469 evaluator output field {key} mismatch")
    expected_teams = [] if panel.get("team_name") is None else [panel["team_name"]]
    filters = document.get("filters")
    if not isinstance(filters, Mapping):
        raise ProtocolError("U469 evaluator output filters must be an object")
    if filters.get("deck_hashes") != [] or filters.get("team_names") != expected_teams:
        raise ProtocolError("U469 evaluator output filters mismatch")


def run_all_panels_u469(
    prereg: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all six panels once while expecting candidate update 469."""

    records: list[dict[str, Any]] = []
    all_gates: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    evaluations = prereg["ordered_evaluations"]
    checkpoint = evidence["checkpoint_path"]
    checkpoint_sha256 = evidence["checkpoint_sha256"]
    for evaluation, panel, output in zip(
        evaluations,
        legacy.PANEL_SPECS,
        evidence["output_paths"],
        strict=True,
    ):
        started = legacy.utc_now()
        command = list(evaluation["command"])
        returncode: int | None = None
        stdout = b""
        stderr = b""
        launch_error: str | None = None
        try:
            completed = legacy.subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdin=legacy.subprocess.DEVNULL,
                stdout=legacy.subprocess.PIPE,
                stderr=legacy.subprocess.PIPE,
                check=False,
                shell=False,
            )
            returncode = int(completed.returncode)
            stdout = completed.stdout
            stderr = completed.stderr
        except BaseException as error:
            launch_error = f"{type(error).__name__}: {error}"

        record: dict[str, Any] = {
            "order": evaluation["order"],
            "panel": panel["name"],
            "started_at_utc": started,
            "finished_at_utc": legacy.utc_now(),
            "command_sha256": evaluation["command_sha256"],
            "attempt_count": 1,
            "returncode": returncode,
            "launch_error": launch_error,
            "stdout": {
                "sha256": sha256_bytes(stdout),
                "bytes": len(stdout),
            },
            "stderr": {
                "sha256": sha256_bytes(stderr),
                "bytes": len(stderr),
            },
            "output": root_relative_text(output),
            "output_present": output.is_file() and not output.is_symlink(),
        }
        panel_gates: list[dict[str, Any]] = []
        output_error: str | None = None
        if record["output_present"]:
            try:
                payload, output_evidence = legacy.read_plain_file(
                    output,
                    f"U469 candidate output {panel['name']}",
                )
                document = legacy.strict_json_loads(
                    payload,
                    f"U469 candidate output {panel['name']}",
                )
                validate_u469_output_identity(
                    document,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    panel=panel,
                )
                metrics = legacy.observed_metrics(document)
                panel_gates = legacy.evaluate_gates(str(panel["name"]), metrics)
                record["output_evidence"] = output_evidence
                record["metrics"] = metrics
            except BaseException as error:
                output_error = f"{type(error).__name__}: {error}"
            finally:
                try:
                    os.chmod(output, 0o444, follow_symlinks=False)
                except OSError as error:
                    if output_error is None:
                        output_error = f"could not freeze output: {error}"
        else:
            output_error = "candidate output is absent"
        record["output_error"] = output_error
        record["gates"] = panel_gates
        record["evaluation_pass"] = (
            returncode == 0
            and launch_error is None
            and output_error is None
            and len(panel_gates) == 10
            and all(gate["pass"] for gate in panel_gates)
        )
        records.append(record)
        all_gates.extend(panel_gates)
    return records, all_gates


def run(preregistration_path: Path, expected_sha256: str) -> dict[str, Any]:
    prereg, evidence = validate_preregistration(
        preregistration_path, expected_sha256
    )
    require_all_outputs_absent(evidence)
    pre_execution_rehash = legacy.rehash_bound_inputs(
        evidence["immutable_bindings"],
        phase="pre_execution",
        raise_on_failure=True,
    )
    cuda = legacy.cuda_preflight()
    marker_payload = canonical_json_bytes(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "status": "one_shot_dev_consistency_attempt_consumed",
            "created_at_utc": legacy.utc_now(),
            "pid": os.getpid(),
            "preregistration": {
                "path": root_relative_text(preregistration_path),
                "sha256": evidence["preregistration"]["sha256"],
            },
            "candidate": expected_candidate_binding(),
            "raw_u468_parent": expected_raw_binding(),
            "classification": CLASSIFICATION,
            "authorization": AUTHORIZATION,
            "pre_execution_rehash": pre_execution_rehash,
            "cuda_preflight": cuda,
        }
    )
    legacy.atomic_create(ATTEMPT_MARKER, marker_payload, mode=0o444)
    OUTPUT_ROOT.mkdir(mode=0o700)

    records, gates = run_all_panels_u469(prereg, evidence)
    post_execution_rehash = legacy.rehash_bound_inputs(
        evidence["immutable_bindings"],
        phase="post_execution",
        raise_on_failure=False,
    )
    complete = len(records) == 6 and all(
        int(record["attempt_count"]) == 1 for record in records
    )
    passed = (
        complete
        and bool(post_execution_rehash["pass"])
        and len(gates) == 60
        and all(bool(record["evaluation_pass"]) for record in records)
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "six_dev_consistency_panels_completed"
        if complete
        else "dev_consistency_execution_incomplete",
        "created_at_utc": legacy.utc_now(),
        "preregistration": evidence["preregistration"],
        "executor": evidence["executor"],
        "generator": evidence["generator"],
        "legacy_specialist_implementation": evidence[
            "legacy_specialist_implementation"
        ],
        "evaluator": evidence["evaluator"],
        "evaluator_dependencies": evidence["evaluator_dependencies"],
        "candidate": evidence["candidate"],
        "raw_u468_parent": evidence["raw_u468_parent"],
        "tainted_training_provenance": evidence["tainted_training_provenance"],
        "data": evidence["data"],
        "baselines": evidence["baselines"],
        "immutable_input_rechecks": {
            "pre_execution": pre_execution_rehash,
            "post_execution": post_execution_rehash,
        },
        "cuda_preflight": cuda,
        "ordered_evaluations": records,
        "gate_count": len(gates),
        "all_six_attempted_once": complete,
        "classification": CLASSIFICATION,
        "authorization": AUTHORIZATION,
        "scope": SCOPE,
    }
    manifest_payload = canonical_json_bytes(manifest)
    legacy.atomic_create(evidence["manifest_path"], manifest_payload, mode=0o444)
    decision = {
        "schema_version": DECISION_SCHEMA,
        "status": "passed_dev_consistency_only"
        if passed
        else "failed_dev_consistency",
        "created_at_utc": legacy.utc_now(),
        "pass": passed,
        "manifest": {
            "path": root_relative_text(evidence["manifest_path"]),
            "sha256": sha256_bytes(manifest_payload),
        },
        "preregistration": {
            "path": root_relative_text(preregistration_path),
            "sha256": evidence["preregistration"]["sha256"],
        },
        "candidate": expected_candidate_binding(),
        "raw_u468_parent": expected_raw_binding(),
        "tainted_training_provenance": evidence["tainted_training_provenance"],
        "classification": CLASSIFICATION,
        "immutable_input_rechecks_pass": bool(pre_execution_rehash["pass"])
        and bool(post_execution_rehash["pass"]),
        "immutable_input_rechecks": {
            "pre_execution": pre_execution_rehash,
            "post_execution": post_execution_rehash,
        },
        "all_six_attempted_once": complete,
        "all_60_gates_required": True,
        "gate_count": len(gates),
        "passed_gate_count": sum(bool(gate["pass"]) for gate in gates),
        "failed_gates": [gate for gate in gates if not bool(gate["pass"])],
        "panel_pass": {
            record["panel"]: bool(record["evaluation_pass"]) for record in records
        },
        "authorization": AUTHORIZATION,
    }
    decision_payload = canonical_json_bytes(decision)
    legacy.atomic_create(evidence["decision_path"], decision_payload, mode=0o444)
    os.chmod(OUTPUT_ROOT, 0o500)
    summary = {
        "decision": root_relative_text(evidence["decision_path"]),
        "decision_sha256": sha256_bytes(decision_payload),
        "pass": passed,
        "classification": CLASSIFICATION,
        "authorization": AUTHORIZATION,
    }
    if not passed:
        print(json.dumps(summary, sort_keys=True))
        raise SystemExit(2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="run")
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--expected-preregistration-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    if args.mode == "static":
        result = static_audit()
    else:
        if args.preregistration is None or args.expected_preregistration_sha256 is None:
            raise ProtocolError("run mode requires preregistration path and SHA-256")
        argument = args.preregistration
        preregistration_path = Path(
            os.path.normpath(
                os.fspath(argument if argument.is_absolute() else Path.cwd() / argument)
            )
        )
        try:
            preregistration_path.relative_to(ROOT.absolute())
        except ValueError as error:
            raise ProtocolError("preregistration must remain inside repository") from error
        result = run(preregistration_path, args.expected_preregistration_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
