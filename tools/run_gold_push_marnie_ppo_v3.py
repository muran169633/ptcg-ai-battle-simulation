#!/usr/bin/env python3
"""Launch the frozen v3 Marnie CVaR-tail repair experiment.

The launcher is dry-run by default.  A real local training run requires both
``--execute`` and the exact manifest SHA-256 printed by a preceding dry run.
It never packages, uploads, or submits a model.  PPO keeps its existing raw
learner action order; every promoted/deployed candidate must instead be
independently evaluated with hybrid action ordering.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
BASE_LAUNCHER = ROOT / "tools/run_gold_push_marnie_ppo.py"
BASE_LAUNCHER_SHA256 = "efc2c308c6165201604aed289eafc22d8c0ef771e4d6bcd6924616a37387d2ac"


def raw_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if raw_sha256_file(BASE_LAUNCHER) != BASE_LAUNCHER_SHA256:
    raise RuntimeError("Authenticated v1 launcher SHA-256 mismatch")
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import run_gold_push_marnie_ppo as base  # noqa: E402


OUTPUT_ROOT = ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v3_cvar_tailrepair"
SMOKE_OUTPUT = ROOT / "artifacts/gold_push_20260810_v1/ppo_marnie_tail32_v3_cvar_tailrepair_smoke"

RAW_ROOT = ROOT / "artifacts/gold_push_20260810_v1"
STRICT_GATE = RAW_ROOT / "panel_frozen_best_v1/strict_gold_gate.json"
PANEL_SUMMARY = RAW_ROOT / "panel_frozen_best_v1/summary.json"
PANEL_MANIFEST = RAW_ROOT / "panel_opponents_v1.json"
PANEL_AUDIT = RAW_ROOT / "panel_opponents_v1.audit.json"
BEHAVIOR_GATE = RAW_ROOT / "ppo_behavior/frozen_best_bc_gate.json"
BEHAVIOR_VALID = RAW_ROOT / "ppo_behavior/frozen_best_valid.json"
BEHAVIOR_TEST = RAW_ROOT / "ppo_behavior/frozen_best_test.json"
SOURCE_TEST = RAW_ROOT / "bc_test/marnie_source_test.json"
CANDIDATE_FREEZE = RAW_ROOT / "ppo_candidate_freeze_v1.json"
RAW_CANDIDATE = RAW_ROOT / "ppo_marnie_tail32_v1/best.pt"

RAIHAN = RAW_ROOT / "clones/marnie_raihan_seed1041/best.pt"
KDCYBERDUDE_PANEL = RAW_ROOT / "clones/marnie_kdcyberdude_seed1043/best.pt"
# last.pt is byte-identical to the panel's best.pt, but its distinct stem keeps
# train_ppo's generated opponent name unique without copying or changing data.
KDCYBERDUDE_TRAIN = RAW_ROOT / "clones/marnie_kdcyberdude_seed1043/last.pt"
MARNIE_NAMED_DECK = ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"

EXTRA_SHA256 = {
    "strict_gate": "2690a8a2933fc6ea173f5243d4c54960303e8f416bf3a560888ac99c941e2919",
    "panel_summary": "a46f3f0f90b3216a11843587993f8cdc3ab79ec0ff5514731148d9287676cda1",
    "panel_manifest": "1ed4fdc713f78b850184522b8a5019a0169f59483bd0f30e06698e6912a07609",
    "panel_audit": "1311e004ea65bec15e15e15165a548250835dbbcab2a1cbb5998efd181214958",
    "behavior_gate": "0190339a091c9d13b1a559063c30d7d41ff2df18fdb63ebb397c9011b7edafcb",
    "behavior_valid": "ee2fde2d9009c7f0c354fb96634f50056296fed85e151f186ff669da32b78c6e",
    "behavior_test": "00e849d1f64567b2d2865f2befbd65b1aafbef180ce61b1a657497a5cc44ef0a",
    "source_test": "97390d8d190f6abb5399c83e33a1f4fb6c3ff69256c6951739473d4565b7ac3d",
    "candidate_freeze": "d777fe3cc018ffcafd0138931b5f6645c0c6c762ce4fa170ded00d1c923fab57",
    "raw_candidate": "a205210bbe201f88eb4942d29c0c840799047df38b9f6abb24c1959e0b683300",
    "raihan": "aa7c0cae1d2652f063d9bbc4f432bdbfc2765846630e8a796455faf4fc87dc1d",
    "kdcyberdude_panel": "c79d9655577cddd9640f95090fc379547d8586fd06eb40ee786b640c26d0e719",
    "kdcyberdude_train": "c79d9655577cddd9640f95090fc379547d8586fd06eb40ee786b640c26d0e719",
    "marnie_named_deck": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
}

EXTRA_PATHS = {
    "strict_gate": STRICT_GATE,
    "panel_summary": PANEL_SUMMARY,
    "panel_manifest": PANEL_MANIFEST,
    "panel_audit": PANEL_AUDIT,
    "behavior_gate": BEHAVIOR_GATE,
    "behavior_valid": BEHAVIOR_VALID,
    "behavior_test": BEHAVIOR_TEST,
    "source_test": SOURCE_TEST,
    "candidate_freeze": CANDIDATE_FREEZE,
    "raw_candidate": RAW_CANDIDATE,
    "raihan": RAIHAN,
    "kdcyberdude_panel": KDCYBERDUDE_PANEL,
    "kdcyberdude_train": KDCYBERDUDE_TRAIN,
    "marnie_named_deck": MARNIE_NAMED_DECK,
}


@dataclass(frozen=True)
class Phase:
    name: str
    updates: int
    games_per_update: int
    own_bc_quota: int
    raihan_quota: int
    kdcyberdude_quota: int
    source_marnie_quota: int
    u472_quota: int
    froslass_quota: int
    eval_games: int
    seed: int


PHASES = {
    "smoke": Phase("smoke", 1, 96, 24, 20, 16, 12, 12, 12, 32, 202608103),
    "full": Phase("full", 2, 192, 48, 40, 32, 24, 24, 24, 256, 202608104),
}


@dataclass(frozen=True)
class Opponent:
    label: str
    checkpoint: Path
    deck: Path
    quota_field: str


OPPONENTS = (
    Opponent("raihan_tail", RAIHAN, base.MARNIE_DECK, "raihan_quota"),
    Opponent("kdcyberdude_tail", KDCYBERDUDE_TRAIN, base.MARNIE_DECK, "kdcyberdude_quota"),
    Opponent("source_marnie_guard", base.SOURCE_MARNIE_BC, MARNIE_NAMED_DECK, "source_marnie_quota"),
    Opponent("u472_guard", base.U472, base.MARNIE_DECK, "u472_quota"),
    Opponent("froslass_uptake_guard", base.FROSLASS_BC, base.FROSLASS_DECK, "froslass_quota"),
)


@dataclass(frozen=True)
class Preflight:
    phase: Phase
    target: Path
    command: list[str]
    manifest: dict[str, Any]
    manifest_sha256: str


@dataclass
class InputLock:
    label: str
    path: Path
    expected_sha256: str
    handle: BinaryIO
    identity: tuple[int, int, int]


def output_dir(phase: Phase) -> Path:
    return SMOKE_OUTPUT if phase.name == "smoke" else OUTPUT_ROOT


def opponent_name(checkpoint: Path, deck: Path) -> str:
    return f"{checkpoint.stem}@{deck.stem}"


def quotas(phase: Phase) -> dict[str, int]:
    values = {"bc": phase.own_bc_quota}
    for opponent in OPPONENTS:
        name = opponent_name(opponent.checkpoint, opponent.deck)
        if name in values:
            raise RuntimeError(f"Duplicate permanent opponent name: {name}")
        values[name] = int(getattr(phase, opponent.quota_field))
    if sum(values.values()) != phase.games_per_update:
        raise RuntimeError("v3 quotas do not sum to games-per-update")
    if any(value <= 0 or value % 2 for value in values.values()):
        raise RuntimeError("v3 quotas must be positive and exactly seat-balanced")
    return values


def set_single_value(command: list[str], flag: str, value: str) -> None:
    if command.count(flag) != 1:
        raise RuntimeError(f"Expected exactly one command flag: {flag}")
    command[command.index(flag) + 1] = value


def strip_repeated_triplets(command: list[str], flag: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(command):
        if command[index] == flag:
            if index + 2 >= len(command):
                raise RuntimeError(f"Malformed repeated command flag: {flag}")
            index += 3
            continue
        result.append(command[index])
        index += 1
    return result


def build_command(phase: Phase) -> list[str]:
    command = base.build_command(base.PHASES[phase.name])
    command = strip_repeated_triplets(command, "--extra-opponent")
    command = strip_repeated_triplets(command, "--opponent-base-quota")
    replacements = {
        "--kl-reference-checkpoint": str(base.SOURCE_MARNIE_BC),
        "--output-dir": str(output_dir(phase)),
        "--updates": str(phase.updates),
        "--games-per-update": str(phase.games_per_update),
        "--learning-rate": "0.000006",
        "--bc-kl-start": "0.030",
        "--bc-kl-end": "0.030",
        "--bc-replay-steps": "4",
        "--bc-replay-lr-scale": "0.10",
        "--eval-games": str(phase.eval_games),
        "--eval-interval": "1",
        "--seed": str(phase.seed),
    }
    for flag, value in replacements.items():
        set_single_value(command, flag, value)
    for opponent in OPPONENTS:
        command.extend(["--extra-opponent", str(opponent.checkpoint), str(opponent.deck)])
    for name, quota in quotas(phase).items():
        command.extend(["--opponent-base-quota", name, str(quota)])
    base.assert_cli_contract(command)
    return command


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_evidence() -> dict[str, Any]:
    strict = read_json(STRICT_GATE)
    panel = read_json(PANEL_MANIFEST)
    panel_audit = read_json(PANEL_AUDIT)
    behavior = read_json(BEHAVIOR_GATE)
    test = read_json(BEHAVIOR_TEST)
    source = read_json(SOURCE_TEST)
    freeze = read_json(CANDIDATE_FREEZE)

    expect(strict.get("schema_version") == "ptcg-gold-push-panel-audit-v1", "strict gate schema drifted")
    expect(strict.get("read_only_audit") is True, "strict audit is no longer read-only")
    expect(strict.get("pass") is False, "raw candidate terminal decision drifted")
    failed = sorted(name for name, gate in strict["gates"].items() if not gate["pass"])
    expect(failed == ["bottom_quartile_cvar"], "strict terminal failure set drifted")
    expected_actuals = {
        "macro_policy_win_rate": 0.630615234375,
        "candidate_seat_0_aggregate_win_rate": 0.638671875,
        "candidate_seat_1_aggregate_win_rate": 0.62255859375,
        "minimum_archetype_win_rate": 0.548828125,
        "bottom_quartile_cvar": 0.4931640625,
    }
    for name, actual in expected_actuals.items():
        expect(float(strict["gates"][name]["actual"]) == actual, f"strict {name} drifted")
    bottom_ids = [
        "marnie_raihan_recent7_clone",
        "marnie_kdcyberdude_recent7_clone",
        "marnie_source_bc_aug08",
        "froslass_lopunny_recent7_uptake_bc",
    ]
    expect(strict["metrics"]["bottom_quartile_cvar"]["policy_ids"] == bottom_ids, "bottom-four identities drifted")
    policies = {row["policy_id"]: row for row in strict["metrics"]["policies"]}
    bottom_rates = {
        policy_id: float(policies[policy_id]["conservative_win_rate"])
        for policy_id in bottom_ids
    }
    expect(bottom_rates == {
        "marnie_raihan_recent7_clone": 0.453125,
        "marnie_kdcyberdude_recent7_clone": 0.484375,
        "marnie_source_bc_aug08": 0.515625,
        "froslass_lopunny_recent7_uptake_bc": 0.51953125,
    }, "bottom-four rates drifted")
    expect(float(policies["marnie_u472_updated_records"]["conservative_win_rate"]) == 0.57421875, "U472 guard rate drifted")

    panel_rows = {row["policy_id"]: row for row in panel["opponents"]}
    expected_panel_hashes = {
        "marnie_raihan_recent7_clone": EXTRA_SHA256["raihan"],
        "marnie_kdcyberdude_recent7_clone": EXTRA_SHA256["kdcyberdude_panel"],
        "marnie_source_bc_aug08": base.FILE_SHA256["source_marnie_bc"],
        "marnie_u472_updated_records": base.FILE_SHA256["u472"],
        "froslass_lopunny_recent7_uptake_bc": base.FILE_SHA256["froslass_bc"],
    }
    for policy_id, digest in expected_panel_hashes.items():
        expect(panel_rows[policy_id]["checkpoint_sha256"] == digest, f"panel checkpoint drifted: {policy_id}")
    integrity = panel_audit["integrity_checks"]
    expect(integrity["policy_count"] == 16 and integrity["all_checkpoint_sha256_match"] is True, "panel integrity drifted")

    expect(behavior["all_offline_behavior_gates_pass"] is False, "raw behavior decision drifted")
    behavior_failed = sorted(name for name, gate in behavior["gates"].items() if not gate["pass"])
    expect(behavior_failed == ["major_context_drop_vs_source_at_most_0_01"], "raw behavior failure set drifted")
    expect(float(behavior["metrics"]["sealed_test_ordered_action_exact_accuracy"]) == 0.777246857595033, "raw ordered accuracy drifted")
    expect(behavior["metrics"]["worst_major_context"] == "22", "worst context drifted")
    expect(float(behavior["metrics"]["worst_major_context_delta_vs_source"]) == -0.1628614916286149, "context-22 delta drifted")
    candidate22 = test["metrics"]["by_context"]["22"]
    source22 = source["metrics"]["by_context"]["22"]
    expect((candidate22["rows"], candidate22["fixed_rows"], candidate22["flexible_rows"]) == (657, 0, 657), "candidate context-22 composition drifted")
    expect(float(candidate22["ordered_action_exact_accuracy"]) == 0.2831050228310502, "candidate context-22 accuracy drifted")
    expect(float(candidate22["hybrid_order_exact_accuracy"]) == 0.45662100456621, "candidate context-22 hybrid accuracy drifted")
    expect(float(source22["ordered_action_exact_accuracy"]) == 0.4459665144596651, "source context-22 accuracy drifted")
    expect(freeze["candidate"]["selected_update"] == 0, "frozen candidate is no longer update zero")
    expect(freeze["candidate"]["sha256"] == EXTRA_SHA256["raw_candidate"], "frozen candidate SHA drifted")
    expect(freeze["conversion_audit"]["source_bc_sha256"] == base.FILE_SHA256["learner_bc"], "frozen BC parent drifted")

    return {
        "raw_terminal_panel": {
            "candidate_sha256": EXTRA_SHA256["raw_candidate"],
            "games": int(strict["accounting"]["attempted_games"]),
            "policy_count": int(strict["identity_and_coverage"]["policies"]),
            "macro": expected_actuals["macro_policy_win_rate"],
            "seat_0": expected_actuals["candidate_seat_0_aggregate_win_rate"],
            "seat_1": expected_actuals["candidate_seat_1_aggregate_win_rate"],
            "minimum_archetype": expected_actuals["minimum_archetype_win_rate"],
            "bottom_quartile_cvar": expected_actuals["bottom_quartile_cvar"],
            "only_failed_gate": "bottom_quartile_cvar",
            "bottom_four": bottom_rates,
            "u472": 0.57421875,
        },
        "raw_behavior": {
            "test_ordered_accuracy": 0.777246857595033,
            "test_hybrid_accuracy": float(behavior["metric_interpretation"]["sealed_test_hybrid_ordered_accuracy"]),
            "only_failed_gate": "major_context_drop_vs_source_at_most_0_01",
            "context_22": {
                "rows": 657,
                "fixed_rows": 0,
                "flexible_rows": 657,
                "candidate_raw_ordered": 0.2831050228310502,
                "candidate_hybrid_ordered": 0.45662100456621,
                "source_ordered": 0.4459665144596651,
                "raw_delta": -0.1628614916286149,
            },
        },
    }


def collect_inputs() -> dict[str, dict[str, Any]]:
    records = base.collect_file_records()
    records["v1_launcher"] = records.pop("launcher")
    for label, path in EXTRA_PATHS.items():
        records[label] = base.validate_file_binding(label, path, EXTRA_SHA256[label])
    records["launcher"] = {
        "path": str(SELF),
        "resolved_path": str(SELF.resolve()),
        "bytes": SELF.stat().st_size,
        "sha256": raw_sha256_file(SELF),
        "protocol_pins_live_launcher_sha256": True,
    }
    return records


def validate_named_marnie_deck() -> dict[str, Any]:
    cards = base.read_deck(MARNIE_NAMED_DECK)
    semantic_hash = base.compute_deck_hash(cards)
    expect(semantic_hash == base.MARNIE_DECK_HASH, "named Marnie deck semantic hash drifted")
    expect(MARNIE_NAMED_DECK.read_bytes() == base.MARNIE_DECK.read_bytes(), "named Marnie deck bytes drifted")
    return {
        "path": str(MARNIE_NAMED_DECK),
        "cards": 60,
        "file_sha256": EXTRA_SHA256["marnie_named_deck"],
        "semantic_deck_hash": semantic_hash,
        "byte_identical_to_frozen_hash_named_deck": True,
        "purpose": "unique train_ppo opponent name for the source-Marnie guard",
    }


def build_preflight(phase: Phase) -> Preflight:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Launcher must run from repository root: {ROOT}")
    target = output_dir(phase)
    base.assert_target_absent(target)
    inputs = collect_inputs()
    evidence = load_evidence()
    expect(EXTRA_SHA256["kdcyberdude_panel"] == EXTRA_SHA256["kdcyberdude_train"], "kd checkpoint alias SHA mismatch")
    named_deck = validate_named_marnie_deck()
    checkpoints = {
        "learner_bc": base.validate_checkpoint("learner_bc", base.LEARNER, base.BC_FEATURE, base.MARNIE_DECK_HASH),
        "source_marnie_guard_and_kl_reference": base.validate_checkpoint("source_marnie_guard", base.SOURCE_MARNIE_BC, base.BC_FEATURE, base.MARNIE_DECK_HASH),
        "raihan_tail": base.validate_checkpoint("raihan_tail", RAIHAN, base.BC_FEATURE, base.MARNIE_DECK_HASH),
        "kdcyberdude_tail": base.validate_checkpoint("kdcyberdude_tail", KDCYBERDUDE_TRAIN, base.BC_FEATURE, base.MARNIE_DECK_HASH),
        "u472_guard": base.validate_checkpoint("submitted_u472", base.U472, base.PPO_FEATURE, base.MARNIE_DECK_HASH),
        "froslass_uptake_guard": base.validate_checkpoint("froslass_uptake_guard", base.FROSLASS_BC, base.BC_FEATURE, base.FROSLASS_DECK_HASH),
    }
    learner_payload = base.torch.load(base.LEARNER, map_location="cpu", weights_only=False)
    learner_lineage = base.validate_learner_lineage(learner_payload)
    replay = base.validate_replay(base.REPLAY)
    with zipfile.ZipFile(base.REPLAY) as archive:
        replay_manifest = json.loads(archive.read("manifest.json"))
    expect(int(replay_manifest["context_decisions"]["22"]) == 4804, "replay context-22 count drifted")
    replay["archive_context22_rows_all_splits"] = 4804

    command = build_command(phase)
    phase_quotas = quotas(phase)
    expected_names = {"bc", *(opponent_name(item.checkpoint, item.deck) for item in OPPONENTS)}
    expect(set(phase_quotas) == expected_names and len(expected_names) == 6, "fixed opponent set drifted")
    eval_rounds = 1 + phase.updates
    manifest: dict[str, Any] = {
        "schema_version": "ptcg-gold-push-marnie-v3-cvar-tailrepair-launch-v1",
        "status": "locked_before_training",
        "phase": phase.name,
        "output_dir": str(target),
        "expected_terminal_checkpoint": str(target / f"checkpoints/update-{phase.updates:04d}.pt"),
        "inputs": inputs,
        "decks": {
            "marnie": base.validate_deck(base.MARNIE_DECK, base.MARNIE_DECK_HASH),
            "marnie_named_alias": named_deck,
            "froslass": base.validate_deck(base.FROSLASS_DECK, base.FROSLASS_DECK_HASH),
        },
        "checkpoints": checkpoints,
        "learner_lineage": learner_lineage,
        "frozen_raw_audit": evidence,
        "parent_choice": {
            "initial_checkpoint": str(base.LEARNER),
            "initial_checkpoint_sha256": base.FILE_SHA256["learner_bc"],
            "reason": "the audited v1 winner was update-0 conversion of this BC; all learned v1 updates lost the fixed-panel minimum",
            "kl_reference": str(base.SOURCE_MARNIE_BC),
            "kl_reference_reason": "the only offline failure is a raw-order drop versus this source, so a strong source anchor is safer than anchoring the failing soup to itself",
            "fresh_run": True,
        },
        "rollout": {
            "updates": phase.updates,
            "games_per_update": phase.games_per_update,
            "total_training_games": phase.updates * phase.games_per_update,
            "fixed_per_update_quotas": phase_quotas,
            "tail_quota_games": phase.raihan_quota + phase.kdcyberdude_quota,
            "tail_and_bottom4_quota_games": phase.raihan_quota + phase.kdcyberdude_quota + phase.source_marnie_quota + phase.froslass_quota,
            "all_quotas_even": all(value % 2 == 0 for value in phase_quotas.values()),
            "seat_balance": "exact_half_per_opponent",
        },
        "objective": {
            "reward": {"win": 1.0, "loss": 0.0, "draw": 0.0},
            "ppo_updates": phase.updates,
            "ppo_epochs": 2,
            "actor_learning_rate": 6e-6,
            "value_learning_rate": 2.5e-5,
            "bc_kl": {"start": 0.030, "end": 0.030, "reference": "source_marnie_guard"},
            "target_kl": 0.004,
            "trainable_scope": "last_block_heads",
        },
        "bc_replay": {
            "split": "train",
            "loss": "ordered",
            "steps_per_update": 4,
            "learning_rate_scale": 0.10,
            "effective_learning_rate": 6e-7,
            "context34_rows_per_batch": 4,
            "context22_direct_stratification": False,
            "context22_note": "context 22 is entirely flexible, so the existing fixed-row weight cannot target it; ordered replay is strengthened globally and hybrid deployment is independently gated",
            "replay": replay,
        },
        "selection": {
            "internal": {
                "fixed_opponents": ["own_bc", *(item.label for item in OPPONENTS)],
                "aggregation": "min",
                "native_cvar_available": False,
                "role": "conservative tail proxy only",
                "games_per_opponent_per_round": phase.eval_games,
                "rounds": eval_rounds,
                "total_evaluation_games": eval_rounds * len(expected_names) * phase.eval_games,
            },
            "external_required": "unchanged 16-policy strict panel with true bottom-quartile CVaR",
        },
        "deployment_order": {
            "ppo_training_learner_order": "existing raw Plackett-Luce order",
            "candidate_canonical_order": False,
            "candidate_hybrid_order": True,
            "required_evaluator_flag": "--candidate-hybrid-order",
            "new_checkpoint_must_be_independently_evaluated": True,
        },
        "promotion_gates": {
            "terminal_panel_unchanged": {
                "policies": 16,
                "games_per_policy": 256,
                "attempted_games_min": 4096,
                "macro_min": 0.58,
                "each_seat_min": 0.55,
                "minimum_archetype_min": 0.52,
                "bottom_quartile_cvar_min": 0.53,
                "invalid_rate_max": 0.001,
            },
            "hybrid_behavior_for_new_checkpoint": {
                "test_hybrid_ordered_accuracy_min": 0.77,
                "valid_to_test_drop_max": 0.01,
                "context_0_delta_vs_source_min": -0.01,
                "every_major_context_delta_vs_source_min": -0.01,
                "count_delta_vs_source_min": -0.002,
                "old_candidate_metrics_are_not_reusable": True,
            },
            "all_must_pass_before_packaging_or_submission": True,
        },
        "capacity": {
            "permanent_opponents": len(expected_names),
            "max_pool_size": 8,
            "training_games": phase.updates * phase.games_per_update,
            "internal_evaluation_games": eval_rounds * len(expected_names) * phase.eval_games,
            "assessment": "fits configured pool; full phase is 384 training plus 4608 internal evaluation games",
        },
        "gates": {
            "all_input_sha256_exact": True,
            "kd_last_byte_identical_to_panel_best": True,
            "named_marnie_deck_byte_identical_to_frozen_deck": True,
            "fixed_quota_sum_exact": sum(phase_quotas.values()) == phase.games_per_update,
            "output_absent_before_launch": True,
            "trainer_cli_contract": base.assert_cli_contract(command),
        },
        "seed": phase.seed,
        "command": command,
        "command_sha256": base.sha256_json(command),
        "scope": {
            "local_training_only": True,
            "training_started_by_dry_run": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }
    return Preflight(phase, target, command, manifest, base.sha256_json(manifest))


def lock_bindings(preflight: Preflight) -> dict[str, tuple[Path, str]]:
    bindings: dict[str, tuple[Path, str]] = {}
    for label, record in preflight.manifest["inputs"].items():
        path = Path(record["resolved_path"] if label == "python" else record["path"])
        bindings[label] = (path, str(record["sha256"]))
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
        for label, (path, expected_sha256) in lock_bindings(preflight).items():
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            handle = os.fdopen(fd, "rb", closefd=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or sha256_handle(handle) != expected_sha256:
                raise RuntimeError(f"Execution-time input mismatch for {label}")
            locks.append(InputLock(label, path, expected_sha256, handle, (info.st_dev, info.st_ino, info.st_size)))
    except BaseException:
        release_input_locks(locks)
        raise
    return locks


def assert_input_locks_unchanged(locks: Sequence[InputLock]) -> None:
    for locked in locks:
        info = os.fstat(locked.handle.fileno())
        if (info.st_dev, info.st_ino, info.st_size) != locked.identity or sha256_handle(locked.handle) != locked.expected_sha256:
            raise RuntimeError(f"Locked input changed during launch: {locked.path}")


def release_input_locks(locks: Sequence[InputLock]) -> None:
    for locked in reversed(locks):
        try:
            fcntl.flock(locked.handle.fileno(), fcntl.LOCK_UN)
        finally:
            locked.handle.close()


def execute(preflight: Preflight) -> int:
    locks = acquire_input_locks(preflight)
    try:
        assert_input_locks_unchanged(locks)
        base.assert_target_absent(preflight.target)
        preflight.target.mkdir(parents=False, exist_ok=False, mode=0o700)
        base.write_exclusive(preflight.target / "launcher_manifest.json", {"manifest_sha256": preflight.manifest_sha256, "manifest": preflight.manifest})
        environment = os.environ.copy()
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(preflight.command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, shell=False, check=False)
        assert_input_locks_unchanged(locks)
        base.write_exclusive(preflight.target / "launcher_result.json", {
            "schema_version": "ptcg-gold-push-marnie-v3-cvar-tailrepair-result-v1",
            "manifest_sha256": preflight.manifest_sha256,
            "return_code": int(completed.returncode),
            "inputs_unchanged_after_child": True,
            "package_upload_submission_performed": False,
        })
        return int(completed.returncode)
    finally:
        release_input_locks(locks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASES), default="smoke")
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
        raise ValueError("--expected-manifest-sha256 is only valid with --execute")
    preflight = build_preflight(PHASES[args.phase])
    print(json.dumps({"manifest_sha256": preflight.manifest_sha256, "manifest": preflight.manifest}, ensure_ascii=False, indent=2, sort_keys=True))
    print(shlex.join(preflight.command), flush=True)
    if not args.execute:
        return 0
    if args.expected_manifest_sha256 != preflight.manifest_sha256:
        raise RuntimeError("Manifest SHA-256 mismatch; repeat and review dry-run")
    return execute(preflight)


if __name__ == "__main__":
    raise SystemExit(main())
