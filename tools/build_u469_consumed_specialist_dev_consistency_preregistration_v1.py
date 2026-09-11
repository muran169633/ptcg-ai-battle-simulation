#!/usr/bin/env python3
"""Build only the U469 one-shot consumed specialist dev preregistration.

This generator never evaluates a panel and never creates a broad template.  It
binds the full U469 checkpoint, the failed two-update source-training decision
and all of its frozen provenance, raw U468, the executor, evaluator and
dependencies, three frozen data archives, six raw baselines, all 60 gates, and
six exactly-once commands.

The specialist-valid rows are consumed development data.  The generated
protocol is dev consistency only, never promotion evidence, cannot cleanse the
source child-flags taint, and cannot authorize broad, Gold, packaging, upload,
or submission.
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
from typing import Any


sys.dont_write_bytecode = True

ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "build_u469_consumed_specialist_dev_consistency_preregistration_v1.py"
EXPECTED_PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FROZEN_MODE = 0o555

EXECUTOR = TOOLS / "run_u469_single_endpoint_consumed_specialist_dev_consistency_v1.py"
EXECUTOR_SHA256 = "83768c1fe6fce5252b95c015e3be97e4acb0ede07f4248b2f4143c5013bcc47d"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def import_executor() -> tuple[ModuleType, dict[str, Any]]:
    _, evidence = read_frozen_source(EXECUTOR, EXECUTOR_SHA256, "U469 dev executor")
    spec = importlib.util.spec_from_file_location(
        f"_u469_consumed_dev_executor_{EXECUTOR_SHA256[:12]}", EXECUTOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import U469 dev executor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, evidence


executor, EXECUTOR_EVIDENCE = import_executor()
legacy = executor.legacy


def validate_runtime() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError("generator must run from repository root")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise RuntimeError(
            f"wrong Python runtime: {sys.executable}; expected {EXPECTED_PYTHON}"
        )
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise RuntimeError("generator requires Python flags -I -B")


def binding(path: Path, label: str) -> dict[str, str]:
    _, evidence = legacy.read_plain_file(path, label)
    return {
        "path": executor.root_relative_text(path),
        "sha256": evidence["sha256"],
    }


def absent_by_lstat(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    return False


def require_targets_absent(*, include_preregistration: bool) -> dict[str, bool]:
    targets = {
        "output_root": executor.OUTPUT_ROOT,
        "attempt_marker": executor.ATTEMPT_MARKER,
        "manifest": executor.OUTPUT_ROOT / "dev_consistency_execution_manifest.json",
        "decision": executor.OUTPUT_ROOT / "dev_consistency_decision.json",
    }
    targets.update(
        {
            f"panel_{panel['name']}": executor.OUTPUT_ROOT / f"{panel['name']}.json"
            for panel in legacy.PANEL_SPECS
        }
    )
    if include_preregistration:
        targets["preregistration"] = executor.PREREGISTRATION
    absent = {name: absent_by_lstat(path) for name, path in targets.items()}
    if not all(absent.values()):
        present = [name for name, value in absent.items() if not value]
        raise FileExistsError(f"fresh U469 dev targets must be absent: {present}")
    return absent


def build_document() -> tuple[dict[str, Any], dict[str, Any]]:
    static = executor.static_audit()
    if static.get("status") != "static_zero_write_audit_only":
        raise RuntimeError("executor static audit did not pass")

    candidate_binding = executor.expected_candidate_binding()
    _, candidate_payload, candidate_evidence = legacy.verify_binding(
        candidate_binding,
        "U469 candidate checkpoint",
        expected_path=executor.CANDIDATE_CHECKPOINT,
        expected_sha256=executor.CANDIDATE_CHECKPOINT_SHA256,
    )
    candidate_evidence.update(
        executor.verify_candidate_checkpoint(candidate_payload, candidate_binding)
    )
    raw_binding = executor.expected_raw_binding()
    raw_path, raw_payload, raw_evidence = legacy.verify_binding(
        raw_binding,
        "raw U468 parent checkpoint",
        expected_path=executor.RAW_U468_CHECKPOINT,
        expected_sha256=executor.RAW_U468_CHECKPOINT_SHA256,
    )
    raw_evidence.update(
        legacy.load_checkpoint_identity(
            raw_payload,
            raw_binding,
            expected_model_state_sha256=executor.RAW_U468_MODEL_STATE_SHA256,
            label="raw U468 parent",
        )
    )
    provenance_bindings = executor.expected_provenance_bindings()
    provenance_evidence, _ = executor.verify_tainted_training_provenance(
        provenance_bindings
    )

    data_bindings: dict[str, dict[str, str]] = {}
    baseline_bindings: dict[str, dict[str, str]] = {}
    evaluations: list[dict[str, Any]] = []
    for order, panel in enumerate(legacy.PANEL_SPECS, start=1):
        data_name = str(panel["data_name"])
        data_bindings.setdefault(
            data_name,
            {
                "path": str(panel["data_path"]),
                "sha256": str(panel["data_sha256"]),
            },
        )
        baseline_bindings[str(panel["name"])] = {
            "path": str(panel["baseline_path"]),
            "sha256": str(panel["baseline_sha256"]),
        }
        output = executor.OUTPUT_ROOT / f"{panel['name']}.json"
        command = legacy.expected_command(executor.CANDIDATE_CHECKPOINT, panel, output)
        evaluations.append(
            {
                "order": order,
                "panel": panel["name"],
                "team_name": panel["team_name"],
                "output": executor.root_relative_text(output),
                "output_absent_at_lock": True,
                "command": command,
                "command_sha256": executor.sha256_bytes(
                    executor.canonical_json_bytes(command)
                ),
                "attempts_authorized": 1,
            }
        )

    for name, item in data_bindings.items():
        actual = binding(ROOT / item["path"], f"data {name}")
        if actual != item:
            raise RuntimeError(f"data binding drift for {name}")
    for name, item in baseline_bindings.items():
        actual = binding(ROOT / item["path"], f"baseline {name}")
        if actual != item:
            raise RuntimeError(f"baseline binding drift for {name}")
        _, payload, _ = legacy.verify_binding(item, f"baseline {name}")
        document = legacy.strict_json_loads(payload, f"baseline {name}")
        panel = next(value for value in legacy.PANEL_SPECS if value["name"] == name)
        legacy.validate_output_identity(
            document,
            checkpoint=raw_path,
            checkpoint_sha256=executor.RAW_U468_CHECKPOINT_SHA256,
            panel=panel,
        )
        if legacy.observed_metrics(document) != legacy.BASELINE_METRICS[name]:
            raise RuntimeError(f"baseline metrics drift for {name}")

    executor_binding = binding(EXECUTOR, "U469 dev executor") | {
        "python": str(EXPECTED_PYTHON),
        "flags": ["-I", "-B"],
    }
    if executor_binding["sha256"] != EXECUTOR_SHA256:
        raise RuntimeError("U469 dev executor SHA drift")
    evaluator_binding = binding(legacy.EXPECTED_EVALUATOR, "evaluator") | {
        "python": str(EXPECTED_PYTHON),
        "flags": ["-I", "-B"],
    }
    if evaluator_binding["sha256"] != legacy.EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("evaluator SHA drift")
    dependencies = {
        name: binding(path, f"evaluator dependency {name}")
        for name, (path, _) in legacy.EXPECTED_DEPENDENCIES.items()
    }
    for name, (_, digest) in legacy.EXPECTED_DEPENDENCIES.items():
        if dependencies[name]["sha256"] != digest:
            raise RuntimeError(f"evaluator dependency SHA drift for {name}")

    manifest_name = "dev_consistency_execution_manifest.json"
    decision_name = "dev_consistency_decision.json"
    document = {
        "schema_version": executor.PREREGISTRATION_SCHEMA,
        "status": executor.PREREGISTRATION_STATUS,
        "classification": dict(executor.CLASSIFICATION),
        "generator": binding(SCRIPT, "U469 dev preregistration generator"),
        "executor": executor_binding,
        "legacy_specialist_implementation": binding(
            executor.LEGACY, "legacy specialist implementation"
        ),
        "candidate": candidate_binding,
        "tainted_training_provenance": provenance_bindings,
        "raw_u468_parent": raw_binding,
        "evaluator": evaluator_binding,
        "evaluator_dependencies": dependencies,
        "data_bindings": data_bindings,
        "baseline_bindings": baseline_bindings,
        "shared_protocol": legacy.SHARED_PROTOCOL,
        "authoritative_panel_gates": {
            name: legacy.expected_thresholds(name)
            for name in legacy.BASELINE_METRICS
        },
        "ordered_evaluations": evaluations,
        "output_rule": {
            "root": executor.root_relative_text(executor.OUTPUT_ROOT),
            "root_absent_at_lock": True,
            "attempt_marker": executor.root_relative_text(executor.ATTEMPT_MARKER),
            "attempt_marker_absent_at_lock": True,
            "evaluation_count_exact": 6,
            "gate_count_exact": 60,
            "attempts_per_evaluation": 1,
            "retry_authorized": False,
            "run_all_before_decision": True,
            "manifest": manifest_name,
            "decision": decision_name,
            "final_directory_mode": "0o500",
        },
        "scope": dict(executor.SCOPE),
        "authorization": dict(executor.AUTHORIZATION),
    }
    if document["authorization"] != executor.AUTHORIZATION or any(
        document["authorization"].values()
    ):
        raise RuntimeError("all downstream authorizations must stay false")
    return document, {
        "executor_static": static,
        "candidate": candidate_evidence,
        "raw_u468_parent": raw_evidence,
        "tainted_training_provenance": provenance_evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "build"), default="build")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_runtime()
    self_payload, self_evidence = legacy.read_plain_file(
        SCRIPT, "U469 dev preregistration generator"
    )
    del self_payload
    if self_evidence["mode_octal"] != "0555":
        raise RuntimeError("generator must be frozen mode 0555")
    document, audit = build_document()
    output_absence = require_targets_absent(include_preregistration=args.mode == "build")
    payload = executor.canonical_json_bytes(document)
    result = {
        "schema_version": "ptcg-u469-consumed-specialist-dev-consistency-builder-result-v1",
        "status": "static_zero_write_audit_only",
        "self": self_evidence,
        "executor": EXECUTOR_EVIDENCE,
        "candidate": audit["candidate"],
        "tainted_training_provenance": audit["tainted_training_provenance"],
        "raw_u468_parent": audit["raw_u468_parent"],
        "classification": executor.CLASSIFICATION,
        "authorization": executor.AUTHORIZATION,
        "ordered_evaluation_count": len(document["ordered_evaluations"]),
        "gate_count": sum(
            len(value) for value in document["authoritative_panel_gates"].values()
        ),
        "fresh_target_absence": output_absence,
        "preregistration": {
            "path": executor.root_relative_text(executor.PREREGISTRATION),
            "prospective_sha256": executor.sha256_bytes(payload),
        },
        "broad_template_created": False,
        "run_executed": False,
        "cuda_accessed": False,
        "writes_performed": False,
    }
    if args.mode == "build":
        legacy.atomic_create(executor.PREREGISTRATION, payload, mode=0o444)
        result["status"] = "dev_consistency_preregistration_built"
        result["preregistration"]["sha256"] = executor.sha256_bytes(payload)
        result["preregistration"]["mode_octal"] = "0444"
        result["writes_performed"] = True
        result["writes"] = [executor.root_relative_text(executor.PREREGISTRATION)]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
