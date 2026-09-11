#!/usr/bin/env python3
"""Run the frozen CW11 endpoint on six specialist panels as dev consistency.

The four specialist-valid guards were consumed while constructing CW11.
Consequently this one-shot six-panel rerun is consistency evidence only and is
never promotion evidence.  A pass cannot authorize broad, Gold, packaging,
upload, or submission.

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
SCRIPT = TOOLS / "run_cw11_single_endpoint_specialist_dev_consistency_v1.py"
BUILDER = TOOLS / "build_cw11_specialist_dev_consistency_preregistration_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_MODE = 0o555

LEGACY = TOOLS / "run_e904_single_endpoint_specialist.py"
LEGACY_SHA256 = "f749fd660209e2369d6cdf21e91ca0a5234eb3306d50af98165cac502f37e6c7"

PREREGISTRATION_SCHEMA = (
    "ptcg-cw11-single-endpoint-specialist-dev-consistency-preregistration-v1"
)
MANIFEST_SCHEMA = (
    "ptcg-cw11-single-endpoint-specialist-dev-consistency-manifest-v1"
)
DECISION_SCHEMA = (
    "ptcg-cw11-single-endpoint-specialist-dev-consistency-decision-v1"
)
ATTEMPT_SCHEMA = (
    "ptcg-cw11-single-endpoint-specialist-dev-consistency-attempt-v1"
)
PREREGISTRATION_STATUS = "locked_before_specialist_dev_consistency"

ARTIFACT_ID = "cw11_4317_specialist_devconsistency_20260802_v1"
PREREGISTRATION = ROOT / "artifacts" / (
    f"{ARTIFACT_ID}.specialist_dev_consistency_preregistration.json"
)
OUTPUT_ROOT = ROOT / "artifacts" / (
    f"{ARTIFACT_ID}.specialist_dev_consistency"
)
ATTEMPT_MARKER = ROOT / "artifacts" / (
    f".ptcg-{ARTIFACT_ID}-specialist-dev-consistency-attempt.json"
)

CANDIDATE_CHECKPOINT = ROOT / (
    "artifacts/ppo_u468_raw_actor6_metricguard_specialist_valid_"
    "cw11_materialized_v1_20260802/u468-cw11-formal-pass-eval-only.pt"
)
CANDIDATE_CHECKPOINT_SHA256 = (
    "bea774c30cd3113d984ba8252324c330d293a8ce5042d7d17f357f8132e86775"
)
CANDIDATE_MODEL_STATE_SHA256 = (
    "4317c492fcf03d1c931c95f1baa1bcc8b9991036c3e8f9e3cc8d6e816d18302e"
)
EXPECTED_CHECKPOINT_UPDATE = 468

MATERIALIZATION_MANIFEST = CANDIDATE_CHECKPOINT.parent / "materialization_manifest.json"
MATERIALIZATION_MANIFEST_SHA256 = (
    "6356f08505b5777fdb30e4f8cfce8a776ca2918af540ebc324227678aeb02016"
)
MATERIALIZATION_COMPLETION = CANDIDATE_CHECKPOINT.parent / "COMPLETED.json"
MATERIALIZATION_COMPLETION_SHA256 = (
    "786281007347ee40a0ac3dd0ad65202c26aed9d5901b3866720980f8cd77a646"
)
MATERIALIZER = TOOLS / (
    "materialize_u468_raw_actor6_metricguard_specialist_valid_cw11_v1.py"
)
MATERIALIZER_SHA256 = (
    "b7c356ab78085f44cbbdbfcf1a46691e126820dc9a6ca0aa2199312ba8a3bbc1"
)

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
}
SCOPE = {
    "local_only": True,
    "network": False,
    "training": False,
    "specialist_behavior": True,
    "specialist_valid_consumed": True,
    "dev_consistency_only": True,
    "promotion_evidence": False,
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
        f"_cw11_dev_legacy_{LEGACY_SHA256[:12]}", LEGACY
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
        "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
    }


def expected_raw_binding() -> dict[str, Any]:
    return {
        "path": root_relative_text(RAW_U468_CHECKPOINT),
        "sha256": RAW_U468_CHECKPOINT_SHA256,
        "model_state_sha256": RAW_U468_MODEL_STATE_SHA256,
        "checkpoint_update": EXPECTED_CHECKPOINT_UPDATE,
    }


def expected_provenance_bindings() -> dict[str, dict[str, str]]:
    return {
        "materializer": {
            "path": root_relative_text(MATERIALIZER),
            "sha256": MATERIALIZER_SHA256,
        },
        "manifest": {
            "path": root_relative_text(MATERIALIZATION_MANIFEST),
            "sha256": MATERIALIZATION_MANIFEST_SHA256,
        },
        "completion": {
            "path": root_relative_text(MATERIALIZATION_COMPLETION),
            "sha256": MATERIALIZATION_COMPLETION_SHA256,
        },
    }


def verify_candidate_checkpoint(
    payload: bytes,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    identity = legacy.load_checkpoint_identity(
        payload,
        binding,
        expected_model_state_sha256=CANDIDATE_MODEL_STATE_SHA256,
        label="CW11 candidate",
    )
    try:
        checkpoint = legacy.torch.load(
            legacy.io.BytesIO(payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ProtocolError("CW11 candidate checkpoint cannot be reloaded") from error
    forbidden = {
        "actor_value_gradient",
        "bc_replay_optimizer_state_dict",
        "metrics",
        "opponent_quota_state",
        "optimizer_parameter_names",
        "optimizer_state_dict",
        "value_trunk_gradient",
    }
    checks = {
        "evaluation_only_exact_true": checkpoint.get("evaluation_only") is True,
        "resume_forbidden_exact_true": checkpoint.get("resume_forbidden") is True,
        "resume_and_stale_keys_absent": not bool(forbidden.intersection(checkpoint)),
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW11 candidate slim-checkpoint drift: {checks}")
    return {**identity, "slim_checkpoint_checks": checks}


def verify_materialization_provenance(
    declared: Any,
) -> tuple[dict[str, Any], dict[str, Path]]:
    expected = expected_provenance_bindings()
    if not isinstance(declared, Mapping) or dict(declared) != expected:
        raise ProtocolError("materialization provenance binding drift")
    documents: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for name, binding in expected.items():
        path, payload, observed = legacy.verify_binding(
            binding,
            f"materialization {name}",
            expected_path=ROOT / binding["path"],
            expected_sha256=binding["sha256"],
        )
        paths[name] = path
        evidence[name] = observed
        if name != "materializer":
            documents[name] = legacy.strict_json_loads(
                payload, f"materialization {name}"
            )
    manifest = documents["manifest"]
    completion = documents["completion"]
    expected_checkpoint = expected_candidate_binding()
    provenance_checks = {
        "manifest_schema_exact": manifest.get("schema_version")
        == "ptcg-u468-raw-actor6-metricguard-specialist-valid-cw11-materializer-v1-manifest",
        "manifest_checkpoint_exact": {
            key: manifest.get("checkpoint", {}).get(key)
            for key in ("path", "sha256", "model_state_sha256")
        }
        == {
            key: expected_checkpoint[key]
            for key in ("path", "sha256", "model_state_sha256")
        },
        "manifest_specialist_consumed": manifest.get(
            "specialist_valid_consumed_for_optimization"
        )
        is True,
        "manifest_promotion_false": manifest.get("promotion_evidence") is False,
        "completion_schema_exact": completion.get("schema_version")
        == "ptcg-u468-raw-actor6-metricguard-specialist-valid-cw11-materializer-v1-completion",
        "completion_status_exact": completion.get("status")
        == "complete_only_after_atomic_directory_publish",
        "completion_checkpoint_exact": {
            key: completion.get("checkpoint", {}).get(key)
            for key in ("path", "sha256", "model_state_sha256")
        }
        == {
            key: expected_checkpoint[key]
            for key in ("path", "sha256", "model_state_sha256")
        },
        "completion_manifest_sha_exact": completion.get("manifest", {}).get(
            "sha256"
        )
        == MATERIALIZATION_MANIFEST_SHA256,
        "completion_evaluation_only": completion.get("evaluation_only") is True,
        "completion_resume_forbidden": completion.get("resume_forbidden") is True,
        "completion_specialist_consumed": completion.get(
            "specialist_valid_consumed_for_optimization"
        )
        is True,
        "completion_promotion_false": completion.get("promotion_evidence") is False,
        "completion_submission_false": completion.get("submission_performed") is False,
    }
    if not all(provenance_checks.values()):
        raise ProtocolError(f"materialization provenance content drift: {provenance_checks}")
    evidence["checks"] = provenance_checks
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
        preregistration_path, "CW11 dev-consistency preregistration"
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
        raise ProtocolError("candidate binding differs from frozen CW11 checkpoint")
    checkpoint, checkpoint_payload, checkpoint_evidence = legacy.verify_binding(
        candidate,
        "CW11 candidate checkpoint",
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
    provenance_evidence, provenance_paths = verify_materialization_provenance(
        prereg.get("materialization_provenance")
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
            "label": f"materialization {name}",
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
    if len(immutable_bindings) != 21:
        raise ProtocolError("immutable input binding count must be exactly 21")

    return prereg, {
        "preregistration": prereg_evidence,
        "executor": executor_evidence,
        "generator": generator_evidence,
        "legacy_specialist_implementation": legacy_evidence,
        "evaluator": evaluator_evidence,
        "evaluator_dependencies": dependency_evidence,
        "candidate": checkpoint_evidence,
        "raw_u468_parent": raw_evidence,
        "materialization_provenance": provenance_evidence,
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
    self_payload, self_evidence = legacy.read_plain_file(SCRIPT, "CW11 dev executor")
    del self_payload
    candidate_path, candidate_payload, candidate_evidence = legacy.verify_binding(
        expected_candidate_binding(),
        "static CW11 candidate checkpoint",
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
    provenance, _ = verify_materialization_provenance(
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
        },
        "all_downstream_authorizations_false": not any(AUTHORIZATION.values()),
        "output_root_absent_by_lstat": absent_by_lstat(OUTPUT_ROOT),
        "attempt_marker_absent_by_lstat": absent_by_lstat(ATTEMPT_MARKER),
    }
    if not all(checks.values()):
        raise ProtocolError(f"CW11 dev executor static audit failed: {checks}")
    return {
        "schema_version": "ptcg-cw11-specialist-dev-consistency-static-v1",
        "status": "static_zero_write_audit_only",
        "self": self_evidence,
        "legacy_specialist_implementation": LEGACY_EVIDENCE,
        "candidate": candidate_evidence,
        "raw_u468_parent": raw_evidence,
        "materialization_provenance": provenance,
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

    records, gates = legacy.run_all_panels(prereg, evidence)
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
        "materialization_provenance": evidence["materialization_provenance"],
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
        "materialization_provenance": evidence["materialization_provenance"],
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
