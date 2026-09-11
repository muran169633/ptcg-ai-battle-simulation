#!/usr/bin/env python3
"""Freeze and optionally launch the parent/S08 full-panel confirmation.

Dry-run is the default and has no filesystem side effects.  Execution requires
``--execute`` plus the exact manifest SHA-256 printed by a reviewed dry-run.
The two candidates use identical full-16 manifest order, seeds, and explicit
hybrid action order.  Their jobs=1 league runners are started concurrently so
the per-policy runs are as interleaved as practical.

The official engine does not expose RNG seed control.  Consequently the two
samples are independent binomial samples, not paired games.  All invalid,
timeout, or truncated attempts are counted as candidate losses; zero invalid
attempts is also a separate hard gate.

This launcher never trains, packages, uploads, or submits.  Every failure is
fail-closed and no candidate is made package/submission eligible unless the
atomically installed terminal decision passes every frozen gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TOOLS = ROOT / "tools"
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
PYTHON_RESOLVED = Path(
    "/home/xxc/miniconda3/envs/my_project_env/bin/python3.11"
)
RUNNER = TOOLS / "run_gold_league_h2h.py"
EVALUATOR = TOOLS / "evaluate_ppo_head_to_head.py"
PANEL_AUDITOR = TOOLS / "audit_gold_push_panel.py"
TEMPLATE_CONTRACT = (
    ROOT / "submission_templates/ptcg_ppo_marnie_hybrid_order_v1/CONTRACT.json"
)

RUNNER_SHA256 = "d19e1eb5f6e178754f055c2845e2735d48df68839c1f0b38992342c306360a03"
EVALUATOR_SHA256 = "f94f40869192aa522f2ff985edd1b0e2476895fb1a56c3e9ebec6eeb21b3d8a6"
PANEL_AUDITOR_SHA256 = "adf3fae9c1f798d5d27e95a1a27140df53616da0fea4a41ecf11478057fe4754"
PYTHON_SHA256 = "35010543d1379740c163ebf34e92108891c70cceb393367d71f463733c8be497"
TEMPLATE_CONTRACT_SHA256 = (
    "97267986ee62e26eeaeafa374e75f151ba3d0510343a5bf96e27e476ca14b200"
)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if file_sha256(PANEL_AUDITOR) != PANEL_AUDITOR_SHA256:
    raise RuntimeError("Authenticated panel auditor SHA-256 mismatch")

import audit_gold_push_panel as panel  # noqa: E402


OUTPUT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "s08_full16_independent_confirmation_v1"
)
PARENT_OUTPUT = OUTPUT / "parent"
S08_OUTPUT = OUTPUT / "s08"
PARENT_DEPLOYMENT_CONTRACT = OUTPUT / "parent.deployment_contract.json"
S08_DEPLOYMENT_CONTRACT = OUTPUT / "s08.deployment_contract.json"
MANIFEST = ROOT / "artifacts/gold_push_20260810_v1/panel_opponents_v1.json"
PARENT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "ppo_marnie_tail32_v3_cvar_tailrepair_smoke_r2/best.pt"
)
S08 = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "postppo_tail_repair_r2best_v1/postppo-special-bc-s08.pt"
)
BC_CHECKPOINT = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "bc_soups/marnie_source50_seedmean50.pt"
)
CANDIDATE_DECK = (
    ROOT
    / "data/gold_push_recent7_20260810_v1/decks/"
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv"
)
ENDPOINT_DECISION = (
    ROOT / "artifacts/gold_push_20260810_v1/endpoint_screen_decision_v1.json"
)
S08_VALID_BEHAVIOR = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "postppo_tail_repair_behavior_v1/s08_valid.json"
)
S08_TEST_BEHAVIOR = (
    ROOT
    / "artifacts/gold_push_20260810_v1/"
    "postppo_tail_repair_behavior_v1/s08_test.json"
)

FROZEN_SHA256 = {
    "python": PYTHON_SHA256,
    "runner": RUNNER_SHA256,
    "evaluator": EVALUATOR_SHA256,
    "panel_auditor": PANEL_AUDITOR_SHA256,
    "panel_manifest": "1ed4fdc713f78b850184522b8a5019a0169f59483bd0f30e06698e6912a07609",
    "parent": "0f1e0a8654e692de7af27f095c2b2a5f02c3ada3164b15437bb38f81ce96d4c6",
    "s08": "d7443bda57cb89a5c12d9d776710d1b8b2c01151541573e01f7e04b032ea25e9",
    "bc_checkpoint": "8ee633d3df1bfe7bce536da4ad7dde71844a2ac293d6e0339731602db44a7036",
    "candidate_deck": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
    "endpoint_decision": "8b59a905151aa9539fd22be6910bd8e3edfad64a43e4f5236073d1303038b17d",
    "s08_valid_behavior": "86d73d71fb2ea9bff0b1485654ce9ff9b9b44bc48dbb58af573c20e85040e35c",
    "s08_test_behavior": "bc844d7e54d26a4015a5b77efd52f66f298c9a911d096f7c2b88d682a949bbf2",
    "template_contract": TEMPLATE_CONTRACT_SHA256,
}

POLICY_IDS = (
    "marnie_source_bc_aug08",
    "marnie_v1_u200",
    "marnie_v3_u440",
    "marnie_u472_updated_records",
    "marnie_raihan_recent7_clone",
    "marnie_kanto_recent7_clone",
    "marnie_kdcyberdude_recent7_clone",
    "marnie_pokemonfan_gold_clone",
    "lucario_source_bc_aug08",
    "lucario_recent7_uptake_bc",
    "froslass_lopunny_source_bc_aug08",
    "froslass_lopunny_recent7_uptake_bc",
    "alakazam_source_bc_aug08",
    "alakazam_yushin_gold_clone",
    "kangaskhan_crustle_source_bc_aug08",
    "dragapult_lumen_gold_clone",
)
TARGET_POLICY_IDS = (
    "marnie_kdcyberdude_recent7_clone",
    "marnie_raihan_recent7_clone",
)
BOTTOM4_POLICY_IDS = frozenset(
    {
        "marnie_source_bc_aug08",
        "marnie_kanto_recent7_clone",
        "marnie_kdcyberdude_recent7_clone",
        "froslass_lopunny_recent7_uptake_bc",
    }
)
OTHER_POLICY_IDS = frozenset(POLICY_IDS) - frozenset(TARGET_POLICY_IDS)

SEED = 202608213
GAMES_PER_POLICY = 512
GAMES_PER_SEAT = 256
POLICY_COUNT = 16
TOTAL_GAMES = POLICY_COUNT * GAMES_PER_POLICY
ENVIRONMENTS = 32
MAX_GAME_DECISIONS = 1000
DEVICE = "cuda"
Z_95 = 1.959963984540054
POSTERIOR_REJECT_EXIT = 42
INPUT_DRIFT_EXIT = 43
LAUNCH_FAILURE_EXIT = 44

ABSOLUTE_THRESHOLDS = {
    "attempted_games_exact": TOTAL_GAMES,
    "valid_games_exact": TOTAL_GAMES,
    "pooled_wilson_95_lower_minimum": 0.56,
    "macro_win_rate_minimum": 0.58,
    "bottom4_cvar_win_rate_minimum": 0.53,
    "minimum_archetype_win_rate": 0.52,
    "seat0_win_rate_minimum": 0.55,
    "seat1_win_rate_minimum": 0.55,
    "invalid_games_exact": 0,
}
RELATIVE_THRESHOLDS = {
    "target_policy_rate_delta_minimum": 0.0,
    "target_combined_rate_gain_minimum": 2.0 / 256.0,
    "pooled_rate_gain_minimum": 4.0 / 768.0,
    "bottom4_rate_gain_minimum": 4.0 / 512.0,
    "minimum_other_policy_rate_delta": -8.0 / 128.0,
    "newcombe_lower_bound_minimum": {
        "pooled": -0.050,
        "bottom4": -0.0625,
        "seat0": -0.075,
        "seat1": -0.075,
    },
}

EXPECTED_EVALUATION = {
    "screening_games": GAMES_PER_POLICY,
    "confirmation": "none",
    "confirmation_games": GAMES_PER_POLICY,
    "environments": ENVIRONMENTS,
    "max_game_decisions": MAX_GAME_DECISIONS,
    "candidate_order_mode": "hybrid",
    "candidate_raw_order": False,
    "candidate_canonical_order": False,
    "candidate_hybrid_order": True,
    "opponent_canonical_order_source": "per_opponent_manifest",
    "legacy_opponent_canonical_order_all_true_assertion": False,
    "seed": SEED,
    "device": DEVICE,
}
EXPECTED_RUNNER_GATES = {
    "cvar_alpha": 0.25,
    "screen_min_policy_win_rate": 0.0,
    "screen_min_policy_wilson_low": 0.0,
    "screen_min_macro_win_rate": 0.0,
    "screen_min_cvar_win_rate": 0.0,
    "promotion_min_policies": POLICY_COUNT,
    "promotion_min_policy_wilson_low": 0.0,
    "promotion_min_archetype_wilson_low": 0.0,
    "promotion_min_seat_wilson_low": 0.0,
    "promotion_min_macro_win_rate": 0.0,
    "promotion_min_policy_win_rate": 0.0,
    "promotion_min_cvar_win_rate": 0.0,
}


class ConfirmationError(RuntimeError):
    """A fail-closed protocol or evidence error."""


@dataclass(frozen=True)
class Preflight:
    manifest: dict[str, Any]
    manifest_sha256: str
    commands: Mapping[str, tuple[str, ...]]
    opponents: tuple[panel.Opponent, ...]


@dataclass
class InputLock:
    label: str
    path: Path
    expected_sha256: str
    handle: BinaryIO
    identity: tuple[int, int, int]


@dataclass(frozen=True)
class RunEvidence:
    label: str
    run_dir: Path
    run_config: dict[str, Any]
    summary: dict[str, Any]
    counts: tuple[panel.PolicyCounts, ...]
    result_paths: Mapping[str, Path]
    result_sha256: Mapping[str, str]
    result_times: Mapping[str, tuple[datetime, datetime]]

    @property
    def by_policy(self) -> dict[str, panel.PolicyCounts]:
        return {row.policy_id: row for row in self.counts}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def render_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def rendered_json_sha256(value: Any) -> str:
    return hashlib.sha256(render_json(value)).hexdigest()


def require_file(label: str, path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ConfirmationError(f"{label} must be a regular non-symlink file: {resolved}")
    observed = file_sha256(resolved)
    if observed != expected_sha256:
        raise ConfirmationError(
            f"{label} SHA-256 drift: expected {expected_sha256}, observed {observed}"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": observed,
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmationError(f"Invalid {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ConfirmationError(f"{label} must be a JSON object: {path}")
    return value


def parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ConfirmationError(f"{label} must be a timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ConfirmationError(f"{label} must be ISO-8601") from error
    if result.tzinfo is None:
        raise ConfirmationError(f"{label} must have a timezone")
    return result


def validate_route_evidence() -> dict[str, Any]:
    decision = read_json(ENDPOINT_DECISION, "endpoint decision")
    comparison = decision.get("endpoint_comparisons", {}).get("s08", {})
    checks = {
        "audit_complete": decision.get("audit_complete") is True,
        "parent_remains_incumbent": decision.get("selected_endpoint") is None,
        "s08_failed_only_kd_gate": comparison.get("failed_gates")
        == ["kdcyberdude_win_delta"],
        "s08_kd_delta_wins": comparison.get(
            "policy_win_deltas_vs_parent", {}
        ).get("marnie_kdcyberdude_recent7_clone")
        == -5,
        "screening_not_submission_evidence": decision.get(
            "screening_only_not_submission_evidence"
        )
        is True,
        "submission_not_authorized": decision.get(
            "submission_authorized_by_this_audit"
        )
        is False,
    }
    if not all(checks.values()):
        raise ConfirmationError(f"Frozen S08 route evidence drifted: {checks}")
    behavior: dict[str, Any] = {}
    for label, path in (
        ("valid", S08_VALID_BEHAVIOR),
        ("test", S08_TEST_BEHAVIOR),
    ):
        report = read_json(path, f"S08 {label} behavior")
        metrics = report.get("metrics") or {}
        row = {
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "checkpoint_update": report.get("checkpoint_update"),
            "prediction_order": report.get("prediction_order"),
            "hybrid_prediction_order": report.get("hybrid_prediction_order"),
            "hybrid_order_exact_accuracy": metrics.get(
                "hybrid_order_exact_accuracy"
            ),
            "count_accuracy": metrics.get("count_accuracy"),
        }
        if row["checkpoint_sha256"] != FROZEN_SHA256["s08"]:
            raise ConfirmationError(f"S08 {label} behavior checkpoint drifted")
        if row["checkpoint_update"] != 0:
            raise ConfirmationError(f"S08 {label} behavior update drifted")
        if row["prediction_order"] != "policy_greedy":
            raise ConfirmationError(f"S08 {label} behavior order drifted")
        if row["hybrid_prediction_order"] != (
            "policy_greedy_for_context_34_skill_order;canonical_ascending_otherwise"
        ):
            raise ConfirmationError(f"S08 {label} hybrid order drifted")
        behavior[label] = row
    return {"checks": checks, "behavior": behavior}


def build_runner_command(
    candidate: Path,
    output_dir: Path,
    deployment_contract: Path,
) -> tuple[str, ...]:
    return (
        str(PYTHON),
        "-I",
        "-B",
        str(RUNNER),
        "--league-manifest",
        str(MANIFEST),
        "--deployment-contract",
        str(deployment_contract),
        "--candidate",
        str(candidate),
        "--candidate-deck",
        str(CANDIDATE_DECK),
        "--output-dir",
        str(output_dir),
        "--bc-checkpoint",
        str(BC_CHECKPOINT),
        "--jobs",
        "1",
        "--screening-games",
        str(GAMES_PER_POLICY),
        "--confirmation",
        "none",
        "--confirmation-games",
        str(GAMES_PER_POLICY),
        "--environments",
        str(ENVIRONMENTS),
        "--max-game-decisions",
        str(MAX_GAME_DECISIONS),
        "--candidate-hybrid-order",
        "--seed",
        str(SEED),
        "--device",
        DEVICE,
        "--cvar-alpha",
        "0.25",
        "--screen-min-policy-win-rate",
        "0",
        "--screen-min-policy-wilson-low",
        "0",
        "--screen-min-macro-win-rate",
        "0",
        "--screen-min-cvar-win-rate",
        "0",
        "--promotion-min-policies",
        str(POLICY_COUNT),
        "--promotion-min-policy-wilson-low",
        "0",
        "--promotion-min-archetype-wilson-low",
        "0",
        "--promotion-min-seat-wilson-low",
        "0",
        "--promotion-min-macro-win-rate",
        "0",
        "--promotion-min-policy-win-rate",
        "0",
        "--promotion-min-cvar-win-rate",
        "0",
    )


def collect_inputs(opponents: Sequence[panel.Opponent]) -> dict[str, dict[str, Any]]:
    paths = {
        "python": PYTHON,
        "runner": RUNNER,
        "evaluator": EVALUATOR,
        "panel_auditor": PANEL_AUDITOR,
        "panel_manifest": MANIFEST,
        "parent": PARENT,
        "s08": S08,
        "bc_checkpoint": BC_CHECKPOINT,
        "candidate_deck": CANDIDATE_DECK,
        "endpoint_decision": ENDPOINT_DECISION,
        "s08_valid_behavior": S08_VALID_BEHAVIOR,
        "s08_test_behavior": S08_TEST_BEHAVIOR,
        "template_contract": TEMPLATE_CONTRACT,
    }
    records: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        if label == "python":
            if not path.is_symlink() or os.readlink(path) != "python3.11":
                raise ConfirmationError(
                    f"my_project_env Python entry symlink drifted: {path}"
                )
            if path.resolve() != PYTHON_RESOLVED:
                raise ConfirmationError(
                    f"my_project_env Python resolved identity drifted: {path.resolve()}"
                )
            row = require_file(label, PYTHON_RESOLVED, FROZEN_SHA256[label])
            row.update(
                {
                    "path": str(path),
                    "entry_is_symlink": True,
                    "entry_symlink_target": os.readlink(path),
                    "resolved_path": str(PYTHON_RESOLVED),
                }
            )
            records[label] = row
        else:
            records[label] = require_file(label, path, FROZEN_SHA256[label])
    records["launcher"] = {
        "path": str(SELF),
        "bytes": SELF.stat().st_size,
        "sha256": file_sha256(SELF),
        "self_hash_frozen_into_manifest": True,
    }
    seen_paths = {Path(row["path"]) for row in records.values()}
    for opponent in opponents:
        for kind, path, expected in (
            ("checkpoint", opponent.checkpoint, opponent.checkpoint_sha256),
            ("deck", opponent.deck, opponent.deck_file_sha256),
        ):
            if path in seen_paths:
                continue
            label = f"opponent_{kind}_{opponent.policy_id}"
            records[label] = require_file(label, path, expected)
            seen_paths.add(path)
    return records


def validate_command(
    label: str,
    command: Sequence[str],
    candidate: Path,
    output: Path,
    deployment_contract: Path,
) -> None:
    if command[:4] != (str(PYTHON), "-I", "-B", str(RUNNER)):
        raise ConfirmationError(f"{label} interpreter/runner contract drifted")
    if command.count("--candidate-hybrid-order") != 1:
        raise ConfirmationError(f"{label} must use explicit hybrid order")
    forbidden = {
        "--candidate-raw-order",
        "--candidate-canonical-order",
        "--resume",
        "--dry-run",
        "--wall-seconds",
        "--policy-id",
        "--archetype",
        "--policy-regex",
        "--max-policies",
    }
    present = forbidden.intersection(command)
    if present:
        raise ConfirmationError(f"{label} forbidden flags: {sorted(present)}")

    def value(flag: str) -> str:
        if command.count(flag) != 1:
            raise ConfirmationError(f"{label} requires exactly one {flag}")
        return command[command.index(flag) + 1]

    expected = {
        "--candidate": str(candidate),
        "--output-dir": str(output),
        "--deployment-contract": str(deployment_contract),
        "--league-manifest": str(MANIFEST),
        "--candidate-deck": str(CANDIDATE_DECK),
        "--bc-checkpoint": str(BC_CHECKPOINT),
        "--jobs": "1",
        "--screening-games": "512",
        "--confirmation": "none",
        "--confirmation-games": "512",
        "--environments": "32",
        "--max-game-decisions": "1000",
        "--seed": str(SEED),
        "--device": DEVICE,
        "--promotion-min-policies": "16",
    }
    observed = {flag: value(flag) for flag in expected}
    if observed != expected:
        raise ConfirmationError(f"{label} command contract drifted: {observed}")


def validate_checkpoint_metadata() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise ConfirmationError("my_project_env must provide torch") from error
    records: dict[str, Any] = {}
    for label, path in (("parent", PARENT), ("s08", S08)):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        row = {
            "feature_version": checkpoint.get("feature_version"),
            "bc_feature_version": checkpoint.get("bc_feature_version"),
            "update": checkpoint.get("update"),
            "learner_deck_hash": checkpoint.get("learner_deck_hash"),
            "evaluation_only": checkpoint.get("evaluation_only", False),
            "resume_forbidden": checkpoint.get("resume_forbidden", False),
        }
        if row["feature_version"] != "ptcg-selfplay-ppo-terminal01-v1":
            raise ConfirmationError(f"{label} PPO feature version drifted")
        if row["bc_feature_version"] != "ptcg-bc-orbit-entity-transformer-v5":
            raise ConfirmationError(f"{label} BC feature version drifted")
        if row["update"] != 0:
            raise ConfirmationError(f"{label} must remain update 0")
        if row["learner_deck_hash"] != panel.deck_semantic_hash(CANDIDATE_DECK):
            raise ConfirmationError(f"{label} learner deck drifted")
        records[label] = row
    if records["s08"]["evaluation_only"] is not True:
        raise ConfirmationError("S08 must remain evaluation-only")
    if records["s08"]["resume_forbidden"] is not True:
        raise ConfirmationError("S08 must remain resume-forbidden")
    return records


def build_deployment_contract(
    *,
    label: str,
    candidate: Path,
    candidate_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": panel.DEPLOYMENT_SCHEMA,
        "frozen_by": "run_gold_push_s08_full16_confirmation.py",
        "candidate_label": label,
        "candidate": {
            "path": str(candidate),
            "sha256": candidate_sha256,
            "feature_version": "ptcg-selfplay-ppo-terminal01-v1",
            "update": 0,
        },
        "candidate_deck": {
            "path": str(CANDIDATE_DECK),
            "file_sha256": FROZEN_SHA256["candidate_deck"],
            "semantic_hash": panel.deck_semantic_hash(CANDIDATE_DECK),
        },
        "action_order": {
            "mode": "hybrid",
            "canonical_order": False,
            "hybrid_order": True,
            "preserve_greedy_plackett_luce_contexts": [34],
            "sort_selected_indices_for_all_other_contexts": True,
        },
        "panel": {
            "manifest": str(MANIFEST),
            "manifest_sha256": FROZEN_SHA256["panel_manifest"],
            "output_dir": str(output_dir),
            "legacy_raw_panel_is_evidence_for_this_contract": False,
        },
        "submission_template": {
            "contract": str(TEMPLATE_CONTRACT),
            "contract_sha256": TEMPLATE_CONTRACT_SHA256,
            "template_version": panel.HYBRID_TEMPLATE_VERSION,
        },
        "required_identity_chain": [
            "deployment_contract",
            "league_run_config",
            "evaluator_result",
            "separate_package_review_if_all_confirmation_gates_pass",
        ],
        "scope": {
            "confirmation_started_by_contract_creation": False,
            "package_created_by_contract_creation": False,
            "upload_performed": False,
            "external_submission_performed": False,
        },
    }


def build_preflight() -> Preflight:
    if Path.cwd().resolve() != ROOT:
        raise ConfirmationError(f"Launcher must run from repository root: {ROOT}")
    if Path(sys.executable) != PYTHON or Path(sys.executable).resolve() != PYTHON_RESOLVED:
        raise ConfirmationError(
            f"Launcher must use my_project_env Python {PYTHON}; got {sys.executable}"
        )
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ConfirmationError(f"One-shot output already exists: {OUTPUT}")
    _, loaded = panel.load_opponents(MANIFEST)
    opponents = tuple(loaded)
    if tuple(row.policy_id for row in opponents) != POLICY_IDS:
        raise ConfirmationError("Frozen full-16 policy identities/order drifted")
    if len({row.policy_id for row in opponents}) != POLICY_COUNT:
        raise ConfirmationError("Full-16 panel identities are not unique")
    inputs = collect_inputs(opponents)
    candidate_hashes = {FROZEN_SHA256["parent"], FROZEN_SHA256["s08"]}
    opponent_hashes = {row.checkpoint_sha256 for row in opponents}
    if len(candidate_hashes) != 2 or candidate_hashes & opponent_hashes:
        raise ConfirmationError("Candidate identities collide with each other/opponents")
    deployment_contracts = {
        "parent": build_deployment_contract(
            label="parent",
            candidate=PARENT,
            candidate_sha256=FROZEN_SHA256["parent"],
            output_dir=PARENT_OUTPUT,
        ),
        "s08": build_deployment_contract(
            label="s08",
            candidate=S08,
            candidate_sha256=FROZEN_SHA256["s08"],
            output_dir=S08_OUTPUT,
        ),
    }
    deployment_contract_paths = {
        "parent": PARENT_DEPLOYMENT_CONTRACT,
        "s08": S08_DEPLOYMENT_CONTRACT,
    }
    commands = {
        "parent": build_runner_command(
            PARENT, PARENT_OUTPUT, PARENT_DEPLOYMENT_CONTRACT
        ),
        "s08": build_runner_command(S08, S08_OUTPUT, S08_DEPLOYMENT_CONTRACT),
    }
    validate_command(
        "parent",
        commands["parent"],
        PARENT,
        PARENT_OUTPUT,
        PARENT_DEPLOYMENT_CONTRACT,
    )
    validate_command(
        "s08", commands["s08"], S08, S08_OUTPUT, S08_DEPLOYMENT_CONTRACT
    )
    checkpoint_metadata = validate_checkpoint_metadata()
    route_evidence = validate_route_evidence()
    manifest: dict[str, Any] = {
        "schema_version": "ptcg-gold-push-s08-full16-independent-confirmation-launch-v1",
        "status": "frozen_before_execute",
        "output_dir": str(OUTPUT),
        "inputs": inputs,
        "candidate_checkpoints": {
            "parent": {
                "path": str(PARENT),
                "sha256": FROZEN_SHA256["parent"],
                "metadata": checkpoint_metadata["parent"],
                "role": "incumbent_anchor",
            },
            "s08": {
                "path": str(S08),
                "sha256": FROZEN_SHA256["s08"],
                "metadata": checkpoint_metadata["s08"],
                "role": "only_frozen_challenger",
            },
        },
        "candidate_deck": {
            "path": str(CANDIDATE_DECK),
            "file_sha256": FROZEN_SHA256["candidate_deck"],
            "semantic_hash": panel.deck_semantic_hash(CANDIDATE_DECK),
        },
        "deployment_contracts": {
            label: {
                "path": str(deployment_contract_paths[label]),
                "payload": deployment_contracts[label],
                "file_sha256": rendered_json_sha256(
                    deployment_contracts[label]
                ),
                "atomic_exclusive_write_before_runner": True,
                "runner_validation_required": True,
            }
            for label in ("parent", "s08")
        },
        "panel": {
            "manifest": str(MANIFEST),
            "manifest_sha256": FROZEN_SHA256["panel_manifest"],
            "policy_ids_in_execution_order": list(POLICY_IDS),
            "policies_exact": POLICY_COUNT,
            "opponents": [
                {
                    "policy_id": row.policy_id,
                    "archetype": row.archetype,
                    "checkpoint": str(row.checkpoint),
                    "checkpoint_sha256": row.checkpoint_sha256,
                    "deck": str(row.deck),
                    "deck_file_sha256": row.deck_file_sha256,
                    "deck_hash": row.deck_hash,
                    "canonical_order": row.canonical_order,
                }
                for row in opponents
            ],
        },
        "evaluation_protocol": {
            "games_per_candidate_per_policy": GAMES_PER_POLICY,
            "valid_games_per_candidate_per_policy_exact": GAMES_PER_POLICY,
            "valid_games_per_candidate_per_policy_per_seat_exact": GAMES_PER_SEAT,
            "valid_games_per_candidate_exact": TOTAL_GAMES,
            "jobs_per_candidate": 1,
            "candidates_launched_concurrently": True,
            "same_manifest_order": True,
            "same_base_and_derived_python_torch_seeds": True,
            "seed": SEED,
            "engine_seed_control": False,
            "statistical_treatment": "independent_binomial_not_paired",
            "wall_clock_interleaving": (
                "best_effort_descriptive_only_not_a_selection_gate"
            ),
            "candidate_action_order": {
                "mode": "hybrid",
                "canonical_order": False,
                "hybrid_order": True,
                "preserve_greedy_plackett_luce_contexts": [34],
                "sort_selected_indices_for_all_other_contexts": True,
            },
            "invalid_timeout_truncation_treatment": "candidate_loss",
            "runner_evaluation": EXPECTED_EVALUATION,
            "runner_internal_gates_ignored_for_terminal_decision": True,
        },
        "absolute_s08_gates": {
            **ABSOLUTE_THRESHOLDS,
            "bottom4_cvar_alpha": 0.25,
            "pooled_wilson_z": Z_95,
            "invalid_timeout_truncation_counted_as_loss": True,
        },
        "relative_s08_vs_parent_gates": {
            **RELATIVE_THRESHOLDS,
            "target_policy_ids": list(TARGET_POLICY_IDS),
            "fixed_bottom4_policy_ids": sorted(BOTTOM4_POLICY_IDS),
            "other_policy_ids": sorted(OTHER_POLICY_IDS),
            "interval_method": "Newcombe_difference_from_independent_Wilson_score_intervals",
            "interval_z": Z_95,
            "not_paired": True,
        },
        "route_evidence": route_evidence,
        "commands": {label: list(command) for label, command in commands.items()},
        "commands_sha256": {
            label: canonical_sha256(list(command))
            for label, command in commands.items()
        },
        "terminal_decision": {
            "path": str(OUTPUT / "decision.json"),
            "atomic_exclusive_no_overwrite": True,
            "any_failure": "reject_s08_preserve_parent_no_package_no_submit",
            "all_gates_pass": "s08_eligible_for_separate_package_review_only",
            "passing_does_not_package_or_submit": True,
        },
        "scope": {
            "training": False,
            "local_evaluation": True,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    return Preflight(
        manifest=manifest,
        manifest_sha256=canonical_sha256(manifest),
        commands=commands,
        opponents=opponents,
    )


def wilson_interval(
    successes: int, trials: int, *, z: float = Z_95
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ConfirmationError("Wilson counts are invalid")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def newcombe_lower_bound(
    s08_wins: int,
    s08_trials: int,
    parent_wins: int,
    parent_trials: int,
) -> dict[str, Any]:
    s08_rate = s08_wins / s08_trials
    parent_rate = parent_wins / parent_trials
    s08_low, s08_high = wilson_interval(s08_wins, s08_trials)
    parent_low, parent_high = wilson_interval(parent_wins, parent_trials)
    difference = s08_rate - parent_rate
    lower = difference - math.sqrt(
        (s08_rate - s08_low) ** 2 + (parent_high - parent_rate) ** 2
    )
    return {
        "s08": {
            "wins": s08_wins,
            "trials": s08_trials,
            "win_rate": s08_rate,
            "wilson_low": s08_low,
            "wilson_high": s08_high,
        },
        "parent": {
            "wins": parent_wins,
            "trials": parent_trials,
            "win_rate": parent_rate,
            "wilson_low": parent_low,
            "wilson_high": parent_high,
        },
        "point_difference": difference,
        "newcombe_lower_bound": lower,
        "z": Z_95,
        "statistical_treatment": "independent_binomial_not_paired",
    }


def aggregate_counts(
    counts: Sequence[panel.PolicyCounts],
    *,
    policy_ids: Iterable[str] = POLICY_IDS,
    seat: str | None = None,
) -> tuple[int, int]:
    selected = frozenset(policy_ids)
    wins = 0
    trials = 0
    for row in counts:
        if row.policy_id not in selected:
            continue
        if seat is None:
            wins += row.wins
            trials += row.attempted_games
        else:
            wins += row.seat_wins[seat]
            trials += row.seat_valid_games[seat] + row.invalid_by_seat[seat]
    if trials <= 0:
        raise ConfirmationError("Aggregate has no trials")
    return wins, trials


def evaluate_gates(
    parent_counts: Sequence[panel.PolicyCounts],
    s08_counts: Sequence[panel.PolicyCounts],
) -> dict[str, Any]:
    for label, counts in (("parent", parent_counts), ("s08", s08_counts)):
        if tuple(row.policy_id for row in counts) != POLICY_IDS:
            raise ConfirmationError(f"{label} policy identities/order drifted")

    s08_by_policy = {row.policy_id: row for row in s08_counts}
    parent_by_policy = {row.policy_id: row for row in parent_counts}
    s08_wins, s08_trials = aggregate_counts(s08_counts)
    s08_valid = sum(row.valid_games for row in s08_counts)
    s08_invalid = sum(row.invalid_games for row in s08_counts)
    parent_valid = sum(row.valid_games for row in parent_counts)
    parent_invalid = sum(row.invalid_games for row in parent_counts)
    policy_rates = {
        row.policy_id: row.conservative_win_rate for row in s08_counts
    }
    macro = sum(policy_rates.values()) / POLICY_COUNT
    bottom_rows = sorted(
        policy_rates.items(), key=lambda item: (item[1], item[0])
    )[:4]
    bottom_cvar = sum(rate for _, rate in bottom_rows) / 4.0
    archetype_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in s08_counts:
        archetype_counts[row.archetype][0] += row.wins
        archetype_counts[row.archetype][1] += row.attempted_games
    archetypes = {
        name: {
            "wins": values[0],
            "trials": values[1],
            "win_rate": values[0] / values[1],
        }
        for name, values in sorted(archetype_counts.items())
    }
    min_archetype = min(row["win_rate"] for row in archetypes.values())
    seat_metrics: dict[str, dict[str, Any]] = {}
    for seat in ("0", "1"):
        wins, trials = aggregate_counts(s08_counts, seat=seat)
        seat_metrics[seat] = {
            "wins": wins,
            "trials": trials,
            "win_rate": wins / trials,
        }
    pooled_wilson_low, pooled_wilson_high = wilson_interval(s08_wins, s08_trials)
    absolute_checks = {
        "s08_attempted_games_exact": s08_trials
        == ABSOLUTE_THRESHOLDS["attempted_games_exact"],
        "s08_valid_games_exact": s08_valid
        == ABSOLUTE_THRESHOLDS["valid_games_exact"],
        "parent_attempted_games_exact": sum(
            row.attempted_games for row in parent_counts
        )
        == TOTAL_GAMES,
        "parent_valid_games_exact": parent_valid == TOTAL_GAMES,
        "s08_pooled_wilson_95_lower": pooled_wilson_low
        >= ABSOLUTE_THRESHOLDS["pooled_wilson_95_lower_minimum"],
        "s08_macro_win_rate": macro
        >= ABSOLUTE_THRESHOLDS["macro_win_rate_minimum"],
        "s08_bottom4_cvar_win_rate": bottom_cvar
        >= ABSOLUTE_THRESHOLDS["bottom4_cvar_win_rate_minimum"],
        "s08_minimum_archetype_win_rate": min_archetype
        >= ABSOLUTE_THRESHOLDS["minimum_archetype_win_rate"],
        "s08_seat0_win_rate": seat_metrics["0"]["win_rate"]
        >= ABSOLUTE_THRESHOLDS["seat0_win_rate_minimum"],
        "s08_seat1_win_rate": seat_metrics["1"]["win_rate"]
        >= ABSOLUTE_THRESHOLDS["seat1_win_rate_minimum"],
        "s08_zero_invalid": s08_invalid == 0,
        "parent_zero_invalid": parent_invalid == 0,
    }

    policy_deltas = {
        policy_id: (
            s08_by_policy[policy_id].conservative_win_rate
            - parent_by_policy[policy_id].conservative_win_rate
        )
        for policy_id in POLICY_IDS
    }
    groups = {
        "pooled": (POLICY_IDS, None),
        "bottom4": (BOTTOM4_POLICY_IDS, None),
        "seat0": (POLICY_IDS, "0"),
        "seat1": (POLICY_IDS, "1"),
    }
    intervals: dict[str, Any] = {}
    for name, (policy_ids, seat) in groups.items():
        sw, st = aggregate_counts(s08_counts, policy_ids=policy_ids, seat=seat)
        pw, pt = aggregate_counts(parent_counts, policy_ids=policy_ids, seat=seat)
        row = newcombe_lower_bound(sw, st, pw, pt)
        minimum = RELATIVE_THRESHOLDS["newcombe_lower_bound_minimum"][name]
        row.update(
            {
                "minimum_newcombe_lower_bound": minimum,
                "pass": row["newcombe_lower_bound"] >= minimum,
            }
        )
        intervals[name] = row

    def rate_gain(policy_ids: Iterable[str], seat: str | None = None) -> float:
        sw, st = aggregate_counts(s08_counts, policy_ids=policy_ids, seat=seat)
        pw, pt = aggregate_counts(parent_counts, policy_ids=policy_ids, seat=seat)
        return sw / st - pw / pt

    point_gates = {
        "kdcyberdude_nonregression": {
            "actual": policy_deltas[TARGET_POLICY_IDS[0]],
            "minimum": RELATIVE_THRESHOLDS[
                "target_policy_rate_delta_minimum"
            ],
        },
        "raihan_nonregression": {
            "actual": policy_deltas[TARGET_POLICY_IDS[1]],
            "minimum": RELATIVE_THRESHOLDS[
                "target_policy_rate_delta_minimum"
            ],
        },
        "target_combined_rate_gain": {
            "actual": rate_gain(TARGET_POLICY_IDS),
            "minimum": RELATIVE_THRESHOLDS[
                "target_combined_rate_gain_minimum"
            ],
        },
        "pooled_rate_gain": {
            "actual": rate_gain(POLICY_IDS),
            "minimum": RELATIVE_THRESHOLDS["pooled_rate_gain_minimum"],
        },
        "bottom4_rate_gain": {
            "actual": rate_gain(BOTTOM4_POLICY_IDS),
            "minimum": RELATIVE_THRESHOLDS["bottom4_rate_gain_minimum"],
        },
        "minimum_other_policy_rate_delta": {
            "actual": min(policy_deltas[row] for row in OTHER_POLICY_IDS),
            "minimum": RELATIVE_THRESHOLDS[
                "minimum_other_policy_rate_delta"
            ],
        },
    }
    for row in point_gates.values():
        row["pass"] = row["actual"] >= row["minimum"]
    relative_checks = {
        **{name: bool(row["pass"]) for name, row in point_gates.items()},
        **{
            f"{name}_newcombe_noninferiority": bool(row["pass"])
            for name, row in intervals.items()
        },
    }
    checks = {**absolute_checks, **relative_checks}
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "pass": not failed,
        "failed_gates": failed,
        "checks": checks,
        "absolute_s08": {
            "attempted_games": s08_trials,
            "valid_games": s08_valid,
            "invalid_games": s08_invalid,
            "wins": s08_wins,
            "pooled_win_rate": s08_wins / s08_trials,
            "pooled_wilson_95_lower": pooled_wilson_low,
            "pooled_wilson_95_upper": pooled_wilson_high,
            "macro_win_rate": macro,
            "bottom4_cvar_alpha": 0.25,
            "bottom4_policy_ids_observed": [row[0] for row in bottom_rows],
            "bottom4_cvar_win_rate": bottom_cvar,
            "archetypes": archetypes,
            "minimum_archetype_win_rate": min_archetype,
            "seats": seat_metrics,
            "thresholds": ABSOLUTE_THRESHOLDS,
        },
        "parent_integrity": {
            "attempted_games": sum(row.attempted_games for row in parent_counts),
            "valid_games": parent_valid,
            "invalid_games": parent_invalid,
        },
        "relative_s08_vs_parent": {
            "policy_rate_deltas": policy_deltas,
            "point_estimate_gates": point_gates,
            "newcombe_noninferiority": intervals,
            "thresholds": RELATIVE_THRESHOLDS,
            "statistical_treatment": "independent_binomial_not_paired",
        },
    }


def validate_runner_signature(run_config: Mapping[str, Any], label: str) -> str:
    signature = run_config.get("run_signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ConfirmationError(f"{label} runner signature is malformed")
    excluded = {
        "created_at_utc",
        "run_signature",
        "jobs",
        "wall_seconds_per_invocation",
        "local_only",
        "uploads_or_submissions_performed",
    }
    payload = {key: value for key, value in run_config.items() if key not in excluded}
    if canonical_sha256(payload) != signature:
        raise ConfirmationError(f"{label} runner signature does not recompute")
    return signature


def validate_run_output(
    *,
    label: str,
    run_dir: Path,
    candidate: Path,
    candidate_sha256: str,
    deployment_contract_path: Path,
    deployment_contract_payload: Mapping[str, Any],
    opponents: Sequence[panel.Opponent],
) -> RunEvidence:
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    run_config = read_json(run_config_path, f"{label} run config")
    summary = read_json(summary_path, f"{label} summary")
    panel.require_equal(run_config.get("schema_version"), panel.RUN_SCHEMA, f"{label} run schema")
    panel.require_equal(summary.get("schema_version"), panel.SUMMARY_SCHEMA, f"{label} summary schema")
    signature = validate_runner_signature(run_config, label)
    panel.require_equal(summary.get("run_signature"), signature, f"{label} summary signature")
    manifest_sha = FROZEN_SHA256["panel_manifest"]
    for document_name, document, base in (
        ("run", run_config, run_config_path.parent),
        ("summary", summary, summary_path.parent),
    ):
        panel.require_equal(
            panel.resolve_path(document.get("league_manifest"), base, f"{label} {document_name} manifest"),
            MANIFEST,
            f"{label} {document_name} manifest",
        )
    panel.require_equal(run_config.get("league_manifest_sha256"), manifest_sha, f"{label} manifest hash")
    panel.require_equal(
        panel.resolve_path(run_config.get("candidate"), run_dir, f"{label} candidate"),
        candidate,
        f"{label} candidate",
    )
    panel.require_equal(run_config.get("candidate_sha256"), candidate_sha256, f"{label} candidate hash")
    panel.require_equal(
        panel.resolve_path(run_config.get("candidate_deck"), run_dir, f"{label} candidate deck"),
        CANDIDATE_DECK,
        f"{label} candidate deck",
    )
    panel.require_equal(
        run_config.get("candidate_deck_file_sha256"),
        FROZEN_SHA256["candidate_deck"],
        f"{label} candidate deck hash",
    )
    summary_candidate = panel.require_mapping(summary.get("candidate"), f"{label} summary candidate")
    panel.require_equal(
        panel.resolve_path(summary_candidate.get("checkpoint"), run_dir, f"{label} summary checkpoint"),
        candidate,
        f"{label} summary checkpoint",
    )
    panel.require_equal(summary_candidate.get("checkpoint_sha256"), candidate_sha256, f"{label} summary candidate hash")
    panel.require_equal(
        panel.resolve_path(summary_candidate.get("deck"), run_dir, f"{label} summary deck"),
        CANDIDATE_DECK,
        f"{label} summary deck",
    )
    panel.require_equal(summary_candidate.get("deck_file_sha256"), FROZEN_SHA256["candidate_deck"], f"{label} summary deck hash")
    panel.require_equal(summary_candidate.get("order_mode"), "hybrid", f"{label} summary order")
    panel.require_equal(run_config.get("selected_opponents"), panel.expected_selected_opponents(opponents), f"{label} selected opponents")
    panel.require_equal(summary.get("selected_policy_ids"), list(POLICY_IDS), f"{label} policy order")
    panel.require_equal(run_config.get("evaluation"), EXPECTED_EVALUATION, f"{label} evaluation protocol")
    panel.require_equal(run_config.get("gates"), EXPECTED_RUNNER_GATES, f"{label} runner gates")
    panel.require_equal(run_config.get("jobs"), 1, f"{label} jobs")
    panel.require_equal(run_config.get("engine_seed_control"), False, f"{label} engine seed control")
    actual_contract = read_json(
        deployment_contract_path, f"{label} deployment contract"
    )
    panel.require_equal(
        actual_contract,
        dict(deployment_contract_payload),
        f"{label} frozen deployment contract payload",
    )
    deployment_sha = rendered_json_sha256(deployment_contract_payload)
    panel.require_equal(
        file_sha256(deployment_contract_path),
        deployment_sha,
        f"{label} deployment contract hash",
    )
    expected_binding = {
        "path": str(deployment_contract_path),
        "sha256": deployment_sha,
        "schema_version": panel.DEPLOYMENT_SCHEMA,
    }
    panel.require_equal(
        run_config.get("deployment_contract"),
        expected_binding,
        f"{label} deployment binding",
    )
    panel.require_equal(
        summary.get("deployment_contract"),
        expected_binding,
        f"{label} summary deployment binding",
    )
    for document_name, document in (("run", run_config), ("summary", summary)):
        panel.require_equal(document.get("local_only"), True, f"{label} {document_name} local only")
        panel.require_equal(document.get("uploads_or_submissions_performed"), False, f"{label} {document_name} side effects")
    panel.require_equal(
        panel.resolve_path(run_config.get("bc_checkpoint"), run_dir, f"{label} BC"),
        BC_CHECKPOINT,
        f"{label} BC path",
    )
    panel.require_equal(run_config.get("bc_checkpoint_sha256"), FROZEN_SHA256["bc_checkpoint"], f"{label} BC hash")
    panel.require_equal(
        panel.resolve_path(run_config.get("evaluator"), run_dir, f"{label} evaluator"),
        EVALUATOR,
        f"{label} evaluator path",
    )
    panel.require_equal(run_config.get("evaluator_sha256"), EVALUATOR_SHA256, f"{label} evaluator hash")
    order_contract = panel.candidate_order_contract(
        panel.require_mapping(run_config.get("evaluation"), f"{label} evaluation")
    )
    panel.validate_deployment_contract_identity(
        run_config=run_config,
        summary=summary,
        run_config_path=run_config_path,
        summary_path=summary_path,
        order_contract=order_contract,
        candidate=candidate,
        candidate_sha256=candidate_sha256,
        candidate_deck=CANDIDATE_DECK,
        candidate_deck_file_sha256=FROZEN_SHA256["candidate_deck"],
        panel_manifest=MANIFEST,
        panel_manifest_sha256=FROZEN_SHA256["panel_manifest"],
        run_dir=run_dir,
    )
    confirmation = panel.require_mapping(summary.get("confirmation"), f"{label} confirmation")
    panel.require_equal(confirmation.get("mode"), "none", f"{label} confirmation mode")
    panel.require_equal(confirmation.get("summary"), None, f"{label} confirmation summary")
    panel.require_equal(summary.get("final_evidence_phase"), "screening", f"{label} evidence phase")
    summary_engine = panel.require_mapping(summary.get("engine"), f"{label} engine")
    panel.require_equal(summary_engine.get("engine_seed_control"), False, f"{label} summary engine seed")
    panel.require_equal(summary_engine.get("strict_even_valid_games_by_candidate_seat_required"), True, f"{label} strict seats")
    screening = panel.require_mapping(summary.get("screening"), f"{label} screening")
    for field, expected in {
        "phase": "screening",
        "games_requested_per_policy": GAMES_PER_POLICY,
        "selected_policies": POLICY_COUNT,
        "completed_policies": POLICY_COUNT,
        "coverage_complete": True,
        "strict_even_seat_balance_verified": True,
        "failures": {},
        "wall_budget_exhausted": False,
    }.items():
        panel.require_equal(screening.get(field), expected, f"{label} screening.{field}")

    results_dir = run_dir / "results/screening"
    expected_paths = {
        row.policy_id: (results_dir / f"{panel.safe_name(row.policy_id)}.json").resolve()
        for row in opponents
    }
    panel.require_equal(
        {path.resolve() for path in results_dir.glob("*.json")},
        set(expected_paths.values()),
        f"{label} exact raw result coverage",
    )
    deck_hash = panel.deck_semantic_hash(CANDIDATE_DECK)
    counts: list[panel.PolicyCounts] = []
    result_hashes: dict[str, str] = {}
    result_times: dict[str, tuple[datetime, datetime]] = {}
    for opponent in opponents:
        result_path = expected_paths[opponent.policy_id]
        result = read_json(result_path, f"{label} result {opponent.policy_id}")
        counts.append(
            panel.validate_result(
                path=result_path,
                result=result,
                opponent=opponent,
                candidate=candidate,
                candidate_sha256=candidate_sha256,
                candidate_deck=CANDIDATE_DECK,
                candidate_deck_hash=deck_hash,
                run_config=run_config,
                games_per_policy=GAMES_PER_POLICY,
            )
        )
        started = parse_timestamp(result.get("started_at_utc"), f"{label} {opponent.policy_id} start")
        finished = parse_timestamp(result.get("finished_at_utc"), f"{label} {opponent.policy_id} finish")
        if finished < started:
            raise ConfirmationError(f"{label} {opponent.policy_id} finishes before start")
        result_times[opponent.policy_id] = (started, finished)
        result_hashes[opponent.policy_id] = file_sha256(result_path)
    panel.validate_summary_counts(screening, counts, expected_paths)
    return RunEvidence(
        label=label,
        run_dir=run_dir,
        run_config=run_config,
        summary=summary,
        counts=tuple(counts),
        result_paths=expected_paths,
        result_sha256=result_hashes,
        result_times=result_times,
    )


def audit_completed_outputs(preflight: Preflight, launch: Mapping[str, Any]) -> dict[str, Any]:
    frozen_contracts = preflight.manifest["deployment_contracts"]
    parent = validate_run_output(
        label="parent",
        run_dir=PARENT_OUTPUT,
        candidate=PARENT,
        candidate_sha256=FROZEN_SHA256["parent"],
        deployment_contract_path=PARENT_DEPLOYMENT_CONTRACT,
        deployment_contract_payload=frozen_contracts["parent"]["payload"],
        opponents=preflight.opponents,
    )
    s08 = validate_run_output(
        label="s08",
        run_dir=S08_OUTPUT,
        candidate=S08,
        candidate_sha256=FROZEN_SHA256["s08"],
        deployment_contract_path=S08_DEPLOYMENT_CONTRACT,
        deployment_contract_payload=frozen_contracts["s08"]["payload"],
        opponents=preflight.opponents,
    )
    shared_fields = (
        "league_manifest",
        "league_manifest_sha256",
        "candidate_deck",
        "candidate_deck_file_sha256",
        "bc_checkpoint",
        "bc_checkpoint_sha256",
        "evaluator",
        "evaluator_sha256",
        "selected_opponents",
        "evaluation",
        "gates",
        "jobs",
        "engine_seed_control",
    )
    for field in shared_fields:
        panel.require_equal(s08.run_config.get(field), parent.run_config.get(field), f"shared runner field {field}")
    if s08.run_config["run_signature"] == parent.run_config["run_signature"]:
        raise ConfirmationError("Parent and S08 runner signatures must differ")
    sequence = {
        run.label: [
            policy_id
            for policy_id, _ in sorted(run.result_times.items(), key=lambda item: item[1][0])
        ]
        for run in (parent, s08)
    }
    start_skews = {
        policy_id: abs(
            (s08.result_times[policy_id][0] - parent.result_times[policy_id][0]).total_seconds()
        )
        for policy_id in POLICY_IDS
    }
    gate_result = evaluate_gates(parent.counts, s08.counts)
    evidence = {
        run.label: {
            "run_dir": str(run.run_dir),
            "run_config": str(run.run_dir / "run_config.json"),
            "run_config_sha256": file_sha256(run.run_dir / "run_config.json"),
            "summary": str(run.run_dir / "summary.json"),
            "summary_sha256": file_sha256(run.run_dir / "summary.json"),
            "run_signature": run.run_config["run_signature"],
            "raw_results": {
                policy_id: {
                    "path": str(run.result_paths[policy_id]),
                    "sha256": run.result_sha256[policy_id],
                }
                for policy_id in POLICY_IDS
            },
        }
        for run in (parent, s08)
    }
    return {
        "schema_version": "ptcg-gold-push-s08-full16-independent-confirmation-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": preflight.manifest_sha256,
        "pass": gate_result["pass"],
        "decision": (
            "s08_eligible_for_separate_package_review_only"
            if gate_result["pass"]
            else "reject_s08_preserve_parent_no_package_no_submit"
        ),
        "failed_gates": gate_result["failed_gates"],
        "gate_audit": gate_result,
        "shared_protocol": {
            "pass": True,
            "same_manifest_order": True,
            "same_runner_protocol": True,
            "same_base_and_derived_python_torch_seeds": True,
            "engine_seed_control": False,
            "statistical_treatment": "independent_binomial_not_paired",
            "candidate_process_launch": dict(launch),
            "observed_sequence_by_run": sequence,
            "observed_sequences_match_manifest_order": all(
                row == list(POLICY_IDS) for row in sequence.values()
            ),
            "per_policy_start_skew_seconds": start_skews,
            "maximum_per_policy_start_skew_seconds": max(start_skews.values()),
            "wall_clock_interleaving_is_descriptive_not_a_gate": True,
        },
        "evidence": evidence,
        "scope": {
            "training_performed": False,
            "package_performed": False,
            "upload_performed": False,
            "submission_performed": False,
            "passing_only_allows_separate_package_review": True,
        },
    }


def atomic_write_json_exclusive(path: Path, value: Any) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_json(value)
    temporary: Path | None = None
    try:
        fd, raw_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(raw_path)
        with os.fdopen(fd, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ConfirmationError(f"Refusing to overwrite evidence: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return hashlib.sha256(rendered).hexdigest()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def lock_bindings(preflight: Preflight) -> list[tuple[str, Path, str]]:
    bindings: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()
    for label, record in preflight.manifest["inputs"].items():
        path = Path(record.get("resolved_path", record["path"])).resolve()
        if path in seen:
            continue
        bindings.append((label, path, str(record["sha256"])))
        seen.add(path)
    return bindings


def sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def acquire_input_locks(preflight: Preflight) -> list[InputLock]:
    locks: list[InputLock] = []
    try:
        for label, path, expected in lock_bindings(preflight):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            handle = os.fdopen(fd, "rb", closefd=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            info = os.fstat(handle.fileno())
            identity = (info.st_dev, info.st_ino, info.st_size)
            if not stat.S_ISREG(info.st_mode) or sha256_handle(handle) != expected:
                handle.close()
                raise ConfirmationError(f"Execution-time input drift: {label}")
            locks.append(InputLock(label, path, expected, handle, identity))
        return locks
    except BaseException:
        release_input_locks(locks)
        raise


def assert_inputs_unchanged(preflight: Preflight, locks: Sequence[InputLock]) -> None:
    for locked in locks:
        info = os.fstat(locked.handle.fileno())
        if (info.st_dev, info.st_ino, info.st_size) != locked.identity:
            raise ConfirmationError(f"Locked input identity changed: {locked.label}")
        if sha256_handle(locked.handle) != locked.expected_sha256:
            raise ConfirmationError(f"Locked input bytes changed: {locked.label}")
    for label, path, expected in lock_bindings(preflight):
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise ConfirmationError(f"Live input path changed: {label}")
    python_record = preflight.manifest["inputs"]["python"]
    python_entry = Path(python_record["path"])
    if (
        not python_entry.is_symlink()
        or os.readlink(python_entry) != python_record["entry_symlink_target"]
        or python_entry.resolve() != Path(python_record["resolved_path"])
    ):
        raise ConfirmationError("my_project_env Python entry identity changed")


def release_input_locks(locks: Sequence[InputLock]) -> None:
    for locked in reversed(locks):
        try:
            fcntl.flock(locked.handle.fileno(), fcntl.LOCK_UN)
        finally:
            locked.handle.close()


def terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_fail_fast(
    processes: Mapping[str, subprocess.Popen[Any]],
    *,
    poll_seconds: float = 0.2,
) -> dict[str, int]:
    """Reap both groups, terminating the sibling group on first failure."""
    remaining = set(processes)
    return_codes: dict[str, int] = {}
    while remaining:
        for label in tuple(sorted(remaining)):
            code = processes[label].poll()
            if code is None:
                continue
            return_codes[label] = int(code)
            remaining.remove(label)
            if code != 0:
                for other in sorted(remaining):
                    terminate_process(processes[other])
                    observed = processes[other].poll()
                    return_codes[other] = (
                        int(observed) if observed is not None else -signal.SIGKILL
                    )
                remaining.clear()
                break
        if remaining:
            time.sleep(poll_seconds)
    return return_codes


def failure_decision(
    preflight: Preflight,
    *,
    reason: str,
    return_code: int,
    child_return_codes: Mapping[str, int | None],
    traceback_text: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "ptcg-gold-push-s08-full16-independent-confirmation-decision-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": preflight.manifest_sha256,
        "pass": False,
        "decision": "reject_s08_preserve_parent_no_package_no_submit",
        "failed_gates": [reason],
        "failure": {
            "reason": reason,
            "return_code": return_code,
            "child_return_codes": dict(child_return_codes),
            "traceback": traceback_text,
        },
        "scope": {
            "training_performed": False,
            "package_performed": False,
            "upload_performed": False,
            "submission_performed": False,
        },
    }


def execute(preflight: Preflight) -> int:
    locks = acquire_input_locks(preflight)
    parent_process: subprocess.Popen[Any] | None = None
    s08_process: subprocess.Popen[Any] | None = None
    logs: list[Any] = []
    child_codes: dict[str, int | None] = {"parent": None, "s08": None}
    try:
        assert_inputs_unchanged(preflight, locks)
        if OUTPUT.exists() or OUTPUT.is_symlink():
            raise ConfirmationError(f"One-shot output already exists: {OUTPUT}")
        OUTPUT.mkdir(parents=False, exist_ok=False, mode=0o700)
        atomic_write_json_exclusive(
            OUTPUT / "launcher_manifest.json",
            {
                "manifest_sha256": preflight.manifest_sha256,
                "manifest": preflight.manifest,
            },
        )
        for label, contract_path in (
            ("parent", PARENT_DEPLOYMENT_CONTRACT),
            ("s08", S08_DEPLOYMENT_CONTRACT),
        ):
            frozen = preflight.manifest["deployment_contracts"][label]
            observed = atomic_write_json_exclusive(
                contract_path, frozen["payload"]
            )
            if observed != frozen["file_sha256"]:
                raise ConfirmationError(
                    f"{label} deployment contract write hash mismatch"
                )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        launch_times: dict[str, float] = {}
        for label in ("parent", "s08"):
            stdout = (OUTPUT / f"{label}.runner.stdout.log").open("xb")
            stderr = (OUTPUT / f"{label}.runner.stderr.log").open("xb")
            logs.extend((stdout, stderr))
            launch_times[label] = time.monotonic()
            process = subprocess.Popen(
                preflight.commands[label],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                start_new_session=True,
            )
            if label == "parent":
                parent_process = process
            else:
                s08_process = process
        launch = {
            "mode": "two_concurrent_jobs1_runners",
            "parent_pid": parent_process.pid,
            "s08_pid": s08_process.pid,
            "launch_skew_seconds": abs(launch_times["s08"] - launch_times["parent"]),
            "launch_skew_is_descriptive_not_a_selection_gate": True,
        }
        observed_codes = wait_fail_fast(
            {"parent": parent_process, "s08": s08_process}
        )
        child_codes.update(observed_codes)
        for handle in logs:
            handle.close()
        logs.clear()
        try:
            assert_inputs_unchanged(preflight, locks)
        except BaseException:
            decision = failure_decision(
                preflight,
                reason="input_drift_after_children",
                return_code=INPUT_DRIFT_EXIT,
                child_return_codes=child_codes,
                traceback_text=traceback.format_exc(),
            )
            atomic_write_json_exclusive(OUTPUT / "decision.json", decision)
            return INPUT_DRIFT_EXIT
        if any(code != 0 for code in child_codes.values()):
            decision = failure_decision(
                preflight,
                reason="runner_child_failure",
                return_code=LAUNCH_FAILURE_EXIT,
                child_return_codes=child_codes,
            )
            atomic_write_json_exclusive(OUTPUT / "decision.json", decision)
            return LAUNCH_FAILURE_EXIT
        try:
            decision = audit_completed_outputs(preflight, launch)
        except BaseException:
            decision = failure_decision(
                preflight,
                reason="posterior_audit_error",
                return_code=POSTERIOR_REJECT_EXIT,
                child_return_codes=child_codes,
                traceback_text=traceback.format_exc(),
            )
        atomic_write_json_exclusive(OUTPUT / "decision.json", decision)
        return 0 if decision.get("pass") is True else POSTERIOR_REJECT_EXIT
    except BaseException:
        terminate_process(parent_process)
        terminate_process(s08_process)
        for label, process in (("parent", parent_process), ("s08", s08_process)):
            if process is not None:
                child_codes[label] = process.poll()
        if OUTPUT.is_dir() and not (OUTPUT / "decision.json").exists():
            decision = failure_decision(
                preflight,
                reason="launcher_failure",
                return_code=LAUNCH_FAILURE_EXIT,
                child_return_codes=child_codes,
                traceback_text=traceback.format_exc(),
            )
            atomic_write_json_exclusive(OUTPUT / "decision.json", decision)
        return LAUNCH_FAILURE_EXIT
    finally:
        for handle in logs:
            handle.close()
        release_input_locks(locks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute and not args.expected_manifest_sha256:
        raise ValueError("--execute requires --expected-manifest-sha256")
    if not args.execute and args.expected_manifest_sha256:
        raise ValueError("--expected-manifest-sha256 is execute-only")
    preflight = build_preflight()
    rendered = {
        "manifest_sha256": preflight.manifest_sha256,
        "manifest": preflight.manifest,
    }
    print(json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True))
    for label in ("parent", "s08"):
        print(f"{label}: {shlex.join(preflight.commands[label])}")
    if not args.execute:
        return 0
    if args.expected_manifest_sha256 != preflight.manifest_sha256:
        raise RuntimeError("Manifest SHA-256 mismatch; repeat and review dry-run")
    return execute(preflight)


if __name__ == "__main__":
    raise SystemExit(main())
