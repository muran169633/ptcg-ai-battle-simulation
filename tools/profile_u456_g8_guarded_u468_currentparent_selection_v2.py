#!/usr/bin/env python3
"""Hash-bound JSON-sanitization recovery for current-parent profile v1.

The frozen v1 implementation is reused wholesale.  The sole behavioral
change is to encode nonfinite diagnostic margins in copied target, guard and
B256 records as JSON null plus an explicit reason before canonical hashing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import stat
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path("/home/xxc/The Pokémon Company - PTCG AI Battle Challenge Simulation")
TOOLS = ROOT / "tools"
SCRIPT = TOOLS / "profile_u456_g8_guarded_u468_currentparent_selection_v2.py"
V1 = TOOLS / "profile_u456_g8_guarded_u468_currentparent_selection_v1.py"
V1_SHA256 = "a398749d606d706d4622cfdfcdd36b7e6ad0c717a8d346af9f524554c18fa869"
RECOVERY = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143.recovery_v2_preregistration.json"
RECOVERY_SHA256 = "ca4a25a45a21603ee51e9df276c96be478d445aa31d3d3c6fa610dcfa4c44e8a"
FAILURE = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143.v1_failure_decision.json"
FAILURE_SHA256 = "d789bfdcf1c3ab120c33b1c5c13eadb46f467de7445438be369cca4d86679aed"
V1_MARKER = ROOT / ".ptcg-u456-g8-guarded-u468-currentparent-profile-selection-attempt-v1.json"
V1_MARKER_SHA256 = "17afd5e5106b5e108fe59d13f5cd83395937e3d36fda07f3ae8416ec2cccdb52"


def load_v1() -> ModuleType:
    observed = V1.lstat()
    source = V1.read_bytes()
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or int(observed.st_nlink) != 1
        or stat.S_IMODE(observed.st_mode) != 0o555
        or hashlib.sha256(source).hexdigest() != V1_SHA256
    ):
        raise RuntimeError("frozen profile v1 identity drift")
    spec = importlib.util.spec_from_file_location("current_parent_profile_v1_frozen", V1)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen profile v1")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_v1()

v1.SCRIPT = SCRIPT
v1.SCHEMA = "ptcg-u456-g8-guarded-u468-currentparent-train-profile-selection-v2"
v1.PREREG_SCHEMA = f"{v1.SCHEMA}-execution-preregistration-v1"
v1.PREREGISTRATION = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143_v2.execution_preregistration.json"
v1.ATTEMPT_MARKER = ROOT / ".ptcg-u456-g8-guarded-u468-currentparent-profile-selection-attempt-v2.json"
v1.OUTPUT = ROOT / "artifacts/ppo_u456_g8_guarded3x4x96_currentparent_profile_selection_design202608143_v2.json"

_original_expected_bindings = v1.expected_bindings
_original_expected_contract = v1.expected_contract
_original_load_static = v1.load_static
_original_select_targets = v1.select_targets
_original_select_guards = v1.select_guards
_original_build_b256 = v1.build_b256


def recovery_evidence() -> dict[str, Any]:
    recovery_bytes, recovery = v1.read_regular(
        RECOVERY, RECOVERY_SHA256, "profile v2 recovery preregistration", 0o444
    )
    failure_bytes, failure = v1.read_regular(
        FAILURE, FAILURE_SHA256, "profile v1 failure decision", 0o444
    )
    _, marker = v1.read_regular(
        V1_MARKER, V1_MARKER_SHA256, "profile v1 consumed marker", 0o444
    )
    recovery_json = json.loads(recovery_bytes)
    failure_json = json.loads(failure_bytes)
    checks = {
        "recovery_locked": recovery_json.get("status") == "locked_before_v2_audit_or_formal_recovery",
        "only_sanitization_change": str(recovery_json.get("only_authorized_code_change", "")).startswith("Sanitize nonfinite selection diagnostic margins"),
        "v1_failure_closed": failure_json.get("status") == "PROFILE_V1_TECHNICAL_ENCODING_FAILURE_CLOSED",
        "v1_same_retry_forbidden": failure_json.get("decision", {}).get("same_v1_attempt_or_runner_retry_authorized") is False,
        "v2_authorized": failure_json.get("decision", {}).get("separately_versioned_v2_parent_only_recovery_authorized") is True,
        "v1_output_absent": failure_json.get("failure", {}).get("original_result_path_absent") is True,
        "zero_changed_candidates": failure_json.get("scope_audit", {}).get("changed_candidate_constructed_or_evaluated") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"profile v2 recovery evidence drift: {checks}")
    return {"recovery": recovery, "failure": failure, "v1_marker": marker, "checks": checks}


def expected_bindings(self_sha256: str) -> dict[str, Any]:
    result = _original_expected_bindings(self_sha256)
    result.update(
        {
            "frozen_v1_runner": {
                "path": str(V1.relative_to(ROOT)),
                "sha256": V1_SHA256,
                "mode_octal": "0555",
            },
            "recovery_v2": {
                "path": str(RECOVERY.relative_to(ROOT)),
                "sha256": RECOVERY_SHA256,
                "mode_octal": "0444",
            },
            "v1_failure_decision": {
                "path": str(FAILURE.relative_to(ROOT)),
                "sha256": FAILURE_SHA256,
                "mode_octal": "0444",
            },
            "v1_attempt_marker": {
                "path": str(V1_MARKER.relative_to(ROOT)),
                "sha256": V1_MARKER_SHA256,
                "mode_octal": "0444",
            },
        }
    )
    return result


def expected_contract() -> dict[str, Any]:
    result = _original_expected_contract()
    result["recovery_version"] = 2
    result["only_change_from_v1"] = (
        "nonfinite copied selection diagnostics encoded as JSON null plus sibling nonfinite_reason before canonical hashing"
    )
    result["v1_result_reused"] = False
    return result


def load_static() -> dict[str, Any]:
    inputs = _original_load_static()
    inputs["recovery_v2"] = recovery_evidence()
    inputs["semantic"]["v2_recovery_authorized"] = True
    return inputs


def sanitize_copied_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for key in (
            "selection_margin",
            "count_margin",
            "decision_margin",
            "predicted_log_probability",
        ):
            value = row.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
                row[f"{key}_nonfinite_reason"] = (
                    "no_competing_allowed_choice"
                    if value > 0
                    else "nonfinite_model_diagnostic"
                )
        result.append(row)
    return result


def select_targets(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sanitize_copied_rows(_original_select_targets(all_rows))


def select_guards(all_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sanitize_copied_rows(_original_select_guards(all_rows))


def build_b256(
    all_rows: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    guards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sanitize_copied_rows(_original_build_b256(all_rows, targets, guards))


v1.expected_bindings = expected_bindings
v1.expected_contract = expected_contract
v1.load_static = load_static
v1.select_targets = select_targets
v1.select_guards = select_guards
v1.build_b256 = build_b256


if __name__ == "__main__":
    v1.main()
