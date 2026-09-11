#!/usr/bin/env python3
"""Read-only, hash-bound plan for hybrid evaluation and PPO packaging."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from package_ppo_submission import read_template_contract, sha256_file
from train_ppo import compute_deck_hash, read_deck


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
ROOT = REPO_ROOT / "artifacts/gold_push_20260810_v1"
CANDIDATE_FREEZE = ROOT / "ppo_candidate_freeze_v1.json"
PREREGISTRATION = ROOT / "preregistration.json"
PANEL_MANIFEST = ROOT / "panel_opponents_v1.json"
DEPLOYMENT_CONTRACT = ROOT / "hybrid_deployment_contract_v1.json"
CANDIDATE = ROOT / "ppo_marnie_tail32_v1/best.pt"
BC_ANCHOR = ROOT / "bc_soups/marnie_source50_seedmean50.pt"
DECK = (
    REPO_ROOT
    / "data/gold_push_recent7_20260810_v1/decks/"
    "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv"
)
HYBRID_TEMPLATE = (
    REPO_ROOT / "submission_templates/ptcg_ppo_marnie_hybrid_order_v1"
)
BASE_TEMPLATE = REPO_ROOT / "submissions/ptcg_ppo_terminal01_v1"
HYBRID_MAIN = HYBRID_TEMPLATE / "main.py"
HYBRID_CONTRACT = HYBRID_TEMPLATE / "CONTRACT.json"
POLICY_RUNTIME = BASE_TEMPLATE / "policy_runtime.py"
HYBRID_PANEL_OUTPUT = ROOT / "panel_frozen_best_hybrid_v1"
VALIDATION_REPORT = ROOT / "package_action_exact_hybrid_build_a_32.json"
PACKAGE_BASE = REPO_ROOT / "submissions/ptcg_gold_push_marnie_hybrid_20260810"

EXPECTED = {
    "candidate_freeze": "d777fe3cc018ffcafd0138931b5f6645c0c6c762ce4fa170ded00d1c923fab57",
    "preregistration": "07d70c6af6cdbe462149b88a61f9a01fa52e88a09f3767ea82a9705bab1388ad",
    "panel_manifest": "1ed4fdc713f78b850184522b8a5019a0169f59483bd0f30e06698e6912a07609",
    "deployment_contract": "ad5516bf09ccb8e0dd2da6e0dae4e9503c6fb6a28a205f61f6d2c434992ca961",
    "candidate": "a205210bbe201f88eb4942d29c0c840799047df38b9f6abb24c1959e0b683300",
    "bc_anchor": "8ee633d3df1bfe7bce536da4ad7dde71844a2ac293d6e0339731602db44a7036",
    "deck": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
}
DECK_HASH = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"
PPO_FEATURE = "ptcg-selfplay-ppo-terminal01-v1"
BC_FEATURE = "ptcg-bc-orbit-entity-transformer-v5"
ACTION_DISTRIBUTION = (
    "masked cardinality categorical + ordered Plackett-Luce without replacement"
)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def normalized_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("model_config")
    if not isinstance(raw, dict):
        raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint lacks model configuration")
    keys = (
        "hash_size",
        "categorical_dim",
        "model_dim",
        "layers",
        "heads",
        "dropout",
        "max_state_entities",
        "entity_fields",
        "option_fields",
    )
    return {key: raw.get(key) for key in keys}


def package_paths(suffix: str) -> dict[str, Path]:
    stem = Path(f"{PACKAGE_BASE}_{suffix}")
    archive = Path(f"{stem}.tar.gz")
    return {
        "output_dir": stem,
        "archive": archive,
        "manifest": Path(f"{archive}.manifest.json"),
    }


def package_command(paths: dict[str, Path]) -> list[str]:
    return [
        str(PYTHON),
        str((REPO_ROOT / "tools/package_ppo_submission.py").resolve()),
        "--checkpoint",
        str(CANDIDATE.resolve()),
        "--deployment-contract",
        str(DEPLOYMENT_CONTRACT.resolve()),
        "--template-dir",
        str(BASE_TEMPLATE.resolve()),
        "--main-file",
        str(HYBRID_MAIN.resolve()),
        "--deck-file",
        str(DECK.resolve()),
        "--policy-runtime-file",
        str(POLICY_RUNTIME.resolve()),
        "--template-contract",
        str(HYBRID_CONTRACT.resolve()),
        "--action-order-mode",
        "hybrid",
        "--output-dir",
        str(paths["output_dir"].resolve()),
        "--archive",
        str(paths["archive"].resolve()),
        "--manifest",
        str(paths["manifest"].resolve()),
    ]


def audit_and_plan() -> dict[str, Any]:
    files = {
        "candidate_freeze": CANDIDATE_FREEZE,
        "preregistration": PREREGISTRATION,
        "panel_manifest": PANEL_MANIFEST,
        "deployment_contract": DEPLOYMENT_CONTRACT,
        "candidate": CANDIDATE,
        "bc_anchor": BC_ANCHOR,
        "deck": DECK,
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        require_equal(sha256_file(path), EXPECTED[label], f"{label} SHA-256")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)

    freeze = json.loads(CANDIDATE_FREEZE.read_text(encoding="utf-8"))
    require_equal(
        (freeze.get("candidate") or {}).get("sha256"),
        EXPECTED["candidate"],
        "freeze candidate",
    )
    require_equal(
        (freeze.get("candidate_deck") or {}).get("semantic_deck_hash"),
        DECK_HASH,
        "freeze deck hash",
    )
    candidate = torch.load(CANDIDATE, map_location="cpu", weights_only=False)
    bc_anchor = torch.load(BC_ANCHOR, map_location="cpu", weights_only=False)
    require_equal(candidate.get("feature_version"), PPO_FEATURE, "PPO feature")
    require_equal(candidate.get("bc_feature_version"), BC_FEATURE, "BC feature")
    require_equal(candidate.get("learner_deck_hash"), DECK_HASH, "learner deck")
    require_equal(
        candidate.get("action_distribution"),
        ACTION_DISTRIBUTION,
        "action distribution",
    )
    require_equal(bc_anchor.get("feature_version"), BC_FEATURE, "BC anchor")
    require_equal(
        normalized_config(candidate),
        normalized_config(bc_anchor),
        "candidate/BC model config",
    )
    deck = read_deck(DECK)
    require_equal(len(deck), 60, "deck card count")
    require_equal(compute_deck_hash(deck), DECK_HASH, "semantic deck hash")

    template_sources = {
        "main.py": HYBRID_MAIN,
        "deck.csv": DECK,
        "policy_runtime.py": POLICY_RUNTIME,
    }
    contract = read_template_contract(
        HYBRID_CONTRACT,
        action_order_mode="hybrid",
        source_files=template_sources,
    )
    build_a = package_paths("build_a")
    build_b = package_paths("build_b")
    future_paths = [
        HYBRID_PANEL_OUTPUT,
        VALIDATION_REPORT,
        *build_a.values(),
        *build_b.values(),
    ]
    existing = [str(path) for path in future_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing reused hybrid output paths: {existing}")

    runner = REPO_ROOT / "tools/run_gold_league_h2h.py"
    evaluator = REPO_ROOT / "tools/evaluate_ppo_head_to_head.py"
    packager = REPO_ROOT / "tools/package_ppo_submission.py"
    validator = REPO_ROOT / "tools/validate_ppo_submission.py"
    pair_verifier = REPO_ROOT / "tools/verify_ppo_package_pair.py"
    tools = (runner, evaluator, packager, validator, pair_verifier)
    for path in tools:
        if not path.is_file():
            raise FileNotFoundError(path)

    panel_command = [
        str(PYTHON),
        "-I",
        "-B",
        str(runner.resolve()),
        "--league-manifest",
        str(PANEL_MANIFEST.resolve()),
        "--deployment-contract",
        str(DEPLOYMENT_CONTRACT.resolve()),
        "--candidate",
        str(CANDIDATE.resolve()),
        "--candidate-deck",
        str(DECK.resolve()),
        "--bc-checkpoint",
        str(BC_ANCHOR.resolve()),
        "--output-dir",
        str(HYBRID_PANEL_OUTPUT.resolve()),
        "--candidate-hybrid-order",
        "--jobs",
        "4",
        "--screening-games",
        "256",
        "--confirmation",
        "none",
        "--confirmation-games",
        "512",
        "--environments",
        "32",
        "--max-game-decisions",
        "1000",
        "--seed",
        "202608104",
        "--device",
        "cuda",
        "--cvar-alpha",
        "0.25",
        "--screen-min-policy-win-rate",
        "0",
        "--screen-min-policy-wilson-low",
        "0",
        "--screen-min-macro-win-rate",
        "0.58",
        "--screen-min-cvar-win-rate",
        "0.53",
        "--promotion-min-policies",
        "16",
        "--promotion-min-policy-wilson-low",
        "0",
        "--promotion-min-archetype-wilson-low",
        "0",
        "--promotion-min-seat-wilson-low",
        "0",
        "--promotion-min-macro-win-rate",
        "0.58",
        "--promotion-min-policy-win-rate",
        "0",
        "--promotion-min-cvar-win-rate",
        "0.53",
    ]
    pair_command = [
        str(PYTHON),
        str(pair_verifier.resolve()),
        "--archive-a",
        str(build_a["archive"].resolve()),
        "--archive-b",
        str(build_b["archive"].resolve()),
        "--manifest-a",
        str(build_a["manifest"].resolve()),
        "--manifest-b",
        str(build_b["manifest"].resolve()),
        "--checkpoint",
        str(CANDIDATE.resolve()),
        "--main-file",
        str(HYBRID_MAIN.resolve()),
        "--deck-file",
        str(DECK.resolve()),
        "--policy-runtime-file",
        str(POLICY_RUNTIME.resolve()),
        "--template-contract",
        str(HYBRID_CONTRACT.resolve()),
        "--deployment-contract",
        str(DEPLOYMENT_CONTRACT.resolve()),
        "--action-order-mode",
        "hybrid",
    ]
    validation_command = [
        str(PYTHON),
        str(validator.resolve()),
        "--archive",
        str(build_a["archive"].resolve()),
        "--checkpoint",
        str(CANDIDATE.resolve()),
        "--bc-checkpoint",
        str(BC_ANCHOR.resolve()),
        "--games",
        "32",
        "--max-game-decisions",
        "1000",
        "--action-order-mode",
        "hybrid",
        "--deployment-contract",
        str(DEPLOYMENT_CONTRACT.resolve()),
        "--output",
        str(VALIDATION_REPORT.resolve()),
    ]
    return {
        "schema_version": "ptcg-gold-push-marnie-hybrid-plan-v1",
        "read_only": True,
        "candidate": {
            "path": str(CANDIDATE.resolve()),
            "sha256": EXPECTED["candidate"],
            "update": candidate.get("update"),
            "learner_deck_hash": DECK_HASH,
        },
        "action_order_contract": {
            "mode": "hybrid",
            "context_34": "preserve_greedy_plackett_luce_order",
            "all_other_contexts": "sort_selected_indices_ascending",
            "must_match": [
                "league_run_config",
                "evaluator_result_identity",
                "submission_template_contract",
                "package_manifest",
                "32_game_action_exact_validation",
            ],
        },
        "template": {
            "contract": str(HYBRID_CONTRACT.resolve()),
            "contract_sha256": sha256_file(HYBRID_CONTRACT),
            "template_version": contract.get("template_version"),
            "sources": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for name, path in template_sources.items()
            },
        },
        "deployment_contract": {
            "path": str(DEPLOYMENT_CONTRACT.resolve()),
            "sha256": EXPECTED["deployment_contract"],
        },
        "tools": {
            path.name: sha256_file(path)
            for path in tools
        },
        "hybrid_panel": {
            "output_dir": str(HYBRID_PANEL_OUTPUT.resolve()),
            "command": panel_command,
            "executed": False,
            "must_use_new_run_signature": True,
        },
        "package_after_all_gates_only": {
            "build_a_command": package_command(build_a),
            "build_b_command": package_command(build_b),
            "pair_identity_command": pair_command,
            "action_exact_32_command": validation_command,
            "required_pair_archive_sha_equal": True,
            "required_pair_member_payloads_equal": True,
            "required_valid_games": 32,
            "required_action_mismatches": 0,
            "required_invalid_games": 0,
            "official_archive_format": "tar.gz",
            "executed": False,
        },
        "remaining_gate_evidence_required_before_packaging": [
            "hybrid behavior gate",
            "passing hybrid terminal panel audit",
            "passing 4x1024 hybrid anchor audit",
            "passing independent 16x512 hybrid confirmation audit",
        ],
        "future_output_paths_exist": False,
        "archive_created": False,
        "external_submission_performed": False,
    }


def main() -> None:
    print(json.dumps(audit_and_plan(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
