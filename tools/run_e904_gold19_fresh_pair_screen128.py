#!/usr/bin/env python3
"""Run the frozen, one-shot U456-vs-E904 Gold19 screen pair.

This is a deliberately narrow local execution wrapper around the hash-locked
isolated adapter for ``run_gold_league_h2h.py``.  It binds the frozen inputs,
runs a fresh U456 control before a fresh E904 candidate exactly once, and
computes the authoritative fourteen screen gates.  The wrapped runner's own
``summary.promotion`` is diagnostic and is never used for the formal decision.

The default mode is preflight only.  ``--execute`` is required to create an
attempt directory and launch the two CUDA runs.  Once an attempt directory is
created it can never be reused or resumed by this wrapper, even after an
interruption or failed arm.  This script has no network, packaging, upload, or
submission operation.

The official battle engine does not expose RNG seed control.  The two frozen
fresh seeds control Python/Torch policy-side randomness only; the arms are
independent samples, not engine-seed-paired trials.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")

RUNNER = REPO_ROOT / "tools/run_gold_league_h2h_isolated_adapter.py"
RUNNER_SHA256 = "c83270a3c1d3e01b727f0a14d2731bf0c3336733faedd7fbbf6a941a4b8cd4a4"
EVALUATOR = REPO_ROOT / "tools/evaluate_ppo_head_to_head.py"
EVALUATOR_SHA256 = "32e0100805b6d7bcd052433a663dc0464a92fbde6225c0347fc1a3f0b62e7005"

LEAGUE_MANIFEST = (
    REPO_ROOT / "artifacts/gold_clone_league_combined_20260727/league_manifest.json"
)
LEAGUE_MANIFEST_SHA256 = (
    "8f70f68c69b9576d958af8bd78e07cd87127c79264533e30717dd23abe8b6571"
)
CANDIDATE_DECK = REPO_ROOT / "data/decks/marnie_grimmsnarl_froslass_luca.csv"
CANDIDATE_DECK_SHA256 = (
    "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
)
BC_CHECKPOINT = (
    REPO_ROOT
    / "artifacts/bc_marnie_train28_valid29_gold21_orbit_v5_seed20260922_20260731/best.pt"
)
BC_CHECKPOINT_SHA256 = (
    "c75c7146cb4b48a20de3c5ca16ad98ddeec669eb2f803e2b6c6ab107b56154eb"
)
CONTROL_CHECKPOINT = (
    REPO_ROOT
    / "artifacts/ppo_bc28init_currentgold10_tailfocus_replay2_ctx34q4_episode_mean_actorlr2x_u448meta_to_u456_seed20260736"
    / "B_gold_league/seed-20260736/checkpoints/update-0456.pt"
)
CONTROL_FILE_SHA256 = (
    "b7ed9580543e4a2374ffcc618bb2eed74b90e762117a587768c90daf0019c83b"
)
CONTROL_MODEL_STATE_SHA256 = (
    "e9baf1917d24aeaab66341835daf86105d071caab9b2985082f921611db18b2c"
)
E904_MODEL_STATE_SHA256 = (
    "e904efb322c15ca4bee62573f4f439309d1d4ecb3ae62620d13fd70e8acaa91f"
)

# These values had zero references in artifacts/**/*.json and tools/**/*.py at
# the static lock scan, apart from this file after it was created.  They are not
# battle-engine seeds and do not make the two arms paired.
CONTROL_SEED = 202608125
CANDIDATE_SEED = 202608126

POLICY_IDS = (
    "rank01_flg",
    "rank02_dominic",
    "rank03_dries",
    "rank04_liam",
    "rank06_etoppo",
    "rank08_hancang",
    "rank11_luca",
    "rank12_taichicchi",
    "rank15_jz",
    "rank17_213tubo",
    "rank18_tuna",
    "rank19_szlachetny",
    "rank05_yushin",
    "rank14_majkel",
    "rank21_cununn",
    "rank13_lumen",
    "rank20_third",
    "rank07_james",
    "rank16_orbit",
)
MARNIE_POLICY_IDS = (
    "rank01_flg",
    "rank02_dominic",
    "rank03_dries",
    "rank04_liam",
    "rank06_etoppo",
    "rank08_hancang",
    "rank11_luca",
    "rank12_taichicchi",
    "rank15_jz",
    "rank17_213tubo",
    "rank18_tuna",
    "rank19_szlachetny",
)
FIXED_HARD_FIVE = (
    "rank03_dries",
    "rank19_szlachetny",
    "rank02_dominic",
    "rank01_flg",
    "rank17_213tubo",
)
MARNIE_ARCHETYPE = "Marnie Grimmsnarl Froslass Munkidori"

GAMES_PER_POLICY = 128
GAMES_PER_POLICY_PER_SEAT = 64
POLICY_COUNT = 19
TOTAL_GAMES = POLICY_COUNT * GAMES_PER_POLICY
TOTAL_GAMES_PER_SEAT = POLICY_COUNT * GAMES_PER_POLICY_PER_SEAT
MARNIE_GAMES = len(MARNIE_POLICY_IDS) * GAMES_PER_POLICY
BOTTOM_FIVE_GAMES = 5 * GAMES_PER_POLICY

FROZEN_TAIL_ARGS = (
    "--jobs", "1",
    "--wall-seconds", "3600",
    "--screening-games", "128",
    "--confirmation", "none",
    "--environments", "32",
    "--max-game-decisions", "1000",
    "--opponent-canonical-order",
    "--device", "cuda",
    "--cvar-alpha", "0.25",
    "--screen-min-policy-win-rate", "0.390625",
    "--screen-min-policy-wilson-low", "0.31047019100495765",
    "--screen-min-macro-win-rate", "0.6414473684210527",
    "--screen-min-cvar-win-rate", "0.45",
    "--promotion-min-policies", "19",
    "--promotion-min-policy-wilson-low", "0.31047019100495765",
    "--promotion-min-archetype-wilson-low", "0",
    "--promotion-min-seat-wilson-low", "0",
    "--promotion-min-macro-win-rate", "0.6414473684210527",
    "--promotion-min-policy-win-rate", "0.390625",
    "--promotion-min-cvar-win-rate", "0.45",
)

ABSOLUTE_MINIMUMS = {
    "candidate_overall_wins": 1560,
    "candidate_marnie_wins": 806,
    "candidate_worst_policy_wins": 50,
    "candidate_own_dynamic_bottom_five_wins": 288,
    "candidate_seat_0_wins": 730,
    "candidate_seat_1_wins": 730,
}
DELTA_MINIMUMS = {
    "candidate_minus_control_overall_wins": 50,
    "candidate_minus_control_marnie_wins": 32,
    "candidate_minus_control_own_dynamic_bottom_five_wins": -8,
    "candidate_minus_control_seat_0_wins": -24,
    "candidate_minus_control_seat_1_wins": -24,
    "candidate_minus_control_fixed_hard_five_wins": 14,
}

SCHEMA = "ptcg-e904-gold19-fresh-pair-screen128-v1"
ATTEMPT_SCHEMA = "ptcg-e904-gold19-fresh-pair-attempt-v1"
DECISION_SCHEMA = "ptcg-e904-gold19-fresh-pair-decision-v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
PYTHON_ENV_KEYS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONINSPECT",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def exclusive_write_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def require_file_sha256(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "sha256": observed, "size": path.stat().st_size}


def canonical_deck_sha256(path: Path) -> str:
    try:
        cards = [
            int(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except ValueError as error:
        raise ValueError(f"{path}: deck must contain one integer per line") from error
    if len(cards) != 60:
        raise ValueError(f"{path}: expected 60 cards, observed {len(cards)}")
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_state_sha256(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid model state entry: {name!r}")
        value = tensor.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite model state entry: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_model_binding(
    path: Path,
    expected_file_sha256: str,
    expected_model_sha256: str,
    label: str,
) -> dict[str, Any]:
    evidence = require_file_sha256(path, expected_file_sha256, label)
    import torch  # Deliberately lazy: --help and syntax checks stay CPU-library free.

    raw = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label}: checkpoint must be a dictionary")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"{label}: missing non-empty model_state_dict")
    observed_model_sha256 = model_state_sha256(state, torch)
    if observed_model_sha256 != expected_model_sha256:
        raise ValueError(
            f"{label} runtime model SHA-256 mismatch: expected "
            f"{expected_model_sha256}, observed {observed_model_sha256}"
        )
    evidence["model_state_sha256"] = observed_model_sha256
    evidence["model_tensor_count"] = len(state)
    return evidence


def root_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def validate_output_root(path: Path, *, require_absent: bool) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve(strict=False)
    artifacts_root = (REPO_ROOT / "artifacts").resolve()
    try:
        relative = resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("--output-root must be below this repository's artifacts/") from error
    if not relative.parts:
        raise ValueError("--output-root cannot be the artifacts directory itself")
    if len(relative.parts) != 1:
        raise ValueError("--output-root must be a new direct child of artifacts/")
    if resolved.parent != artifacts_root or not artifacts_root.is_dir():
        raise ValueError("The bound artifacts parent is unavailable")
    if require_absent and os.path.lexists(resolved):
        raise FileExistsError(
            f"One-shot output target already exists and cannot be resumed: {resolved}"
        )
    return resolved


def validate_manifest() -> dict[str, Any]:
    manifest = read_json(LEAGUE_MANIFEST)
    if manifest.get("schema_version") != "ptcg-ppo-opponent-league-v1":
        raise ValueError("Frozen Gold19 manifest schema drift")
    raw_opponents = manifest.get("opponents")
    if not isinstance(raw_opponents, list) or len(raw_opponents) != POLICY_COUNT:
        raise ValueError("Frozen Gold19 manifest must contain exactly 19 opponents")
    observed_ids = tuple(row.get("policy_id") for row in raw_opponents)
    if observed_ids != POLICY_IDS:
        raise ValueError("Frozen Gold19 manifest policy identities/order drift")
    observed_marnie = tuple(
        row.get("policy_id")
        for row in raw_opponents
        if row.get("archetype") == MARNIE_ARCHETYPE
    )
    if observed_marnie != MARNIE_POLICY_IDS:
        raise ValueError("Frozen twelve-policy Marnie subset drift")

    checked_opponents: list[dict[str, Any]] = []
    for index, row in enumerate(raw_opponents):
        if not isinstance(row, dict):
            raise TypeError(f"opponents[{index}] is not an object")
        checkpoint_value = row.get("checkpoint")
        deck_value = row.get("deck")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            raise ValueError(f"opponents[{index}].checkpoint is invalid")
        if not isinstance(deck_value, str) or not deck_value:
            raise ValueError(f"opponents[{index}].deck is invalid")
        checkpoint = Path(checkpoint_value)
        deck = Path(deck_value)
        if not checkpoint.is_absolute():
            checkpoint = LEAGUE_MANIFEST.parent / checkpoint
        if not deck.is_absolute():
            deck = LEAGUE_MANIFEST.parent / deck
        checkpoint = checkpoint.resolve()
        deck = deck.resolve()

        declared_checkpoint_sha256 = str(row.get("checkpoint_sha256"))
        declared_deck_hash = str(row.get("deck_hash"))
        if not HEX64.fullmatch(declared_checkpoint_sha256):
            raise ValueError(f"opponents[{index}].checkpoint_sha256 is invalid")
        if not HEX64.fullmatch(declared_deck_hash):
            raise ValueError(f"opponents[{index}].deck_hash is invalid")
        checkpoint_evidence = require_file_sha256(
            checkpoint,
            declared_checkpoint_sha256,
            f"opponents[{index}].checkpoint",
        )
        if not deck.is_file():
            raise FileNotFoundError(f"opponents[{index}].deck: {deck}")
        observed_deck_hash = canonical_deck_sha256(deck)
        if observed_deck_hash != declared_deck_hash:
            raise ValueError(
                f"opponents[{index}].deck canonical SHA-256 mismatch: expected "
                f"{declared_deck_hash}, observed {observed_deck_hash}"
            )
        checked_opponents.append(
            {
                "policy_id": row["policy_id"],
                "checkpoint": checkpoint_evidence,
                "deck": {
                    "path": str(deck),
                    "file_sha256": sha256_file(deck),
                    "canonical_deck_sha256": observed_deck_hash,
                    "declared_deck_hash": declared_deck_hash,
                },
            }
        )
    return {
        "policy_ids": list(observed_ids),
        "marnie_policy_ids": list(observed_marnie),
        "opponent_binding_count": len(checked_opponents),
        "all_checkpoint_file_sha256_match_manifest": True,
        "all_canonical_deck_sha256_match_manifest": True,
        "opponents": checked_opponents,
    }


def scan_seed_freshness() -> dict[str, Any]:
    matches: dict[str, list[str]] = {str(CONTROL_SEED): [], str(CANDIDATE_SEED): []}
    this_file = Path(__file__).resolve()
    for base, suffixes in (
        (REPO_ROOT / "tools", {".py"}),
        (REPO_ROOT / "artifacts", {".json"}),
    ):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if path.resolve() == this_file:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for seed in matches:
                if seed in content:
                    matches[seed].append(root_relative(path))
    occupied = {seed: paths for seed, paths in matches.items() if paths}
    if occupied:
        raise ValueError(f"Fresh-seed lock is no longer fresh: {occupied}")
    return {
        "control_seed": CONTROL_SEED,
        "candidate_seed": CANDIDATE_SEED,
        "fresh_at_static_lock_scan": True,
        "engine_seed_control": False,
        "arms_are_independent_not_paired": True,
    }


def verify_static_bindings(
    candidate: Path,
    candidate_file_sha256: str,
    *,
    check_seed_freshness: bool,
) -> dict[str, Any]:
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(
            f"Wrong Python environment: expected {PYTHON.resolve()}, "
            f"observed {Path(sys.executable).resolve()}"
        )
    if int(sys.flags.isolated) != 1:
        raise RuntimeError("Gold pair runner itself requires Python -I")
    if int(sys.flags.dont_write_bytecode) != 1:
        raise RuntimeError("Gold pair runner itself requires Python -B")
    if not HEX64.fullmatch(candidate_file_sha256):
        raise ValueError("--candidate-file-sha256 must be lowercase 64-hex")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("--candidate must resolve inside this repository") from error
    if candidate == CONTROL_CHECKPOINT.resolve():
        raise ValueError("Candidate cannot alias the U456 control checkpoint")

    evidence = {
        "python": str(PYTHON),
        "runner": require_file_sha256(RUNNER, RUNNER_SHA256, "Gold19 runner"),
        "evaluator": require_file_sha256(
            EVALUATOR, EVALUATOR_SHA256, "Gold19 evaluator"
        ),
        "league_manifest": require_file_sha256(
            LEAGUE_MANIFEST, LEAGUE_MANIFEST_SHA256, "Gold19 manifest"
        ),
        "candidate_deck": require_file_sha256(
            CANDIDATE_DECK, CANDIDATE_DECK_SHA256, "candidate deck"
        ),
        "bc_decode_anchor": require_file_sha256(
            BC_CHECKPOINT, BC_CHECKPOINT_SHA256, "BC decode anchor"
        ),
        "u456_control": checkpoint_model_binding(
            CONTROL_CHECKPOINT,
            CONTROL_FILE_SHA256,
            CONTROL_MODEL_STATE_SHA256,
            "U456 control checkpoint",
        ),
        "e904_candidate": checkpoint_model_binding(
            candidate,
            candidate_file_sha256,
            E904_MODEL_STATE_SHA256,
            "E904 candidate checkpoint",
        ),
        "manifest_contract": validate_manifest(),
    }
    if check_seed_freshness:
        evidence["seed_lock"] = scan_seed_freshness()
    return evidence


def arm_output_dir(output_root: Path, arm: str) -> Path:
    if arm == "control":
        return output_root / f"fresh_u456_control_seed{CONTROL_SEED}"
    if arm == "candidate":
        return output_root / f"fresh_e904_candidate_seed{CANDIDATE_SEED}"
    raise ValueError(arm)


def build_command(
    *,
    candidate: Path,
    output_dir: Path,
    seed: int,
) -> list[str]:
    command = [
        str(PYTHON),
        "-I",
        "-B",
        str(RUNNER),
        "--league-manifest", str(LEAGUE_MANIFEST),
        "--candidate", str(candidate),
        "--candidate-deck", str(CANDIDATE_DECK),
        "--bc-checkpoint", str(BC_CHECKPOINT),
        "--output-dir", str(output_dir),
        *FROZEN_TAIL_ARGS[:13],
        "--seed", str(seed),
        *FROZEN_TAIL_ARGS[13:],
    ]
    forbidden = {
        "--resume",
        "--dry-run",
        "--candidate-canonical-order",
        "--policy-id",
        "--archetype",
        "--policy-regex",
        "--max-policies",
    }
    present = forbidden.intersection(command)
    if present:
        raise AssertionError(f"Forbidden Gold19 command flags: {sorted(present)}")
    if command.count("--opponent-canonical-order") != 1:
        raise AssertionError("Gold opponents must use canonical order exactly once")
    if command.count("--seed") != 1:
        raise AssertionError("Each arm must have exactly one frozen seed")
    return command


def execution_plan(
    candidate: Path,
    candidate_file_sha256: str,
    output_root: Path,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    arms = [
        {
            "order": 1,
            "arm": "control",
            "seed": CONTROL_SEED,
            "checkpoint": root_relative(CONTROL_CHECKPOINT),
            "checkpoint_file_sha256": CONTROL_FILE_SHA256,
            "runtime_model_state_sha256": CONTROL_MODEL_STATE_SHA256,
            "output_dir": root_relative(arm_output_dir(output_root, "control")),
            "command": build_command(
                candidate=CONTROL_CHECKPOINT,
                output_dir=arm_output_dir(output_root, "control"),
                seed=CONTROL_SEED,
            ),
        },
        {
            "order": 2,
            "arm": "candidate",
            "seed": CANDIDATE_SEED,
            "checkpoint": root_relative(candidate),
            "checkpoint_file_sha256": candidate_file_sha256,
            "runtime_model_state_sha256": E904_MODEL_STATE_SHA256,
            "output_dir": root_relative(arm_output_dir(output_root, "candidate")),
            "command": build_command(
                candidate=candidate,
                output_dir=arm_output_dir(output_root, "candidate"),
                seed=CANDIDATE_SEED,
            ),
        },
    ]
    return {
        "schema_version": SCHEMA,
        "created_at_utc": utc_now(),
        "mode": "preflight",
        "output_root": root_relative(output_root),
        "bindings": bindings,
        "ordered_runs": arms,
        "protocol": {
            "policies": POLICY_COUNT,
            "games_per_policy": GAMES_PER_POLICY,
            "games_per_arm": TOTAL_GAMES,
            "candidate_decode": "policy_greedy_dynamic",
            "opponent_decode": "canonical_ascending",
            "candidate_canonical_order": False,
            "opponent_canonical_order": True,
            "formal_gate_count": 14,
            "all_gates_required": True,
            "runner_promotion_is_diagnostic_only": True,
            "no_resume": True,
            "no_retry": True,
            "no_prior_sample_pooling": True,
            "no_seed_selection": True,
            "engine_seed_control": False,
            "arms_are_independent_not_paired": True,
            "full_python_chain_isolated_and_no_bytecode": True,
        },
        "scope": {
            "local_only": True,
            "network": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }


def expected_run_config(
    *, candidate: Path, candidate_sha256: str, seed: int
) -> dict[str, Any]:
    return {
        "schema_version": "ptcg-gold-league-h2h-run-v1",
        "candidate": str(candidate),
        "candidate_sha256": candidate_sha256,
        "candidate_deck": str(CANDIDATE_DECK),
        "candidate_deck_file_sha256": CANDIDATE_DECK_SHA256,
        "bc_checkpoint": str(BC_CHECKPOINT),
        "bc_checkpoint_sha256": BC_CHECKPOINT_SHA256,
        "league_manifest": str(LEAGUE_MANIFEST),
        "league_manifest_sha256": LEAGUE_MANIFEST_SHA256,
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": EVALUATOR_SHA256,
        "engine_seed_control": False,
        "jobs": 1,
        "wall_seconds_per_invocation": 3600.0,
        "local_only": True,
        "uploads_or_submissions_performed": False,
        "evaluation": {
            "candidate_canonical_order": False,
            "confirmation": "none",
            "confirmation_games": 2048,
            "device": "cuda",
            "environments": 32,
            "max_game_decisions": 1000,
            "opponent_canonical_order": True,
            "screening_games": 128,
            "seed": seed,
        },
        "gates": {
            "cvar_alpha": 0.25,
            "promotion_min_archetype_wilson_low": 0.0,
            "promotion_min_cvar_win_rate": 0.45,
            "promotion_min_macro_win_rate": 0.6414473684210527,
            "promotion_min_policies": 19,
            "promotion_min_policy_wilson_low": 0.31047019100495765,
            "promotion_min_policy_win_rate": 0.390625,
            "promotion_min_seat_wilson_low": 0.0,
            "screen_min_cvar_win_rate": 0.45,
            "screen_min_macro_win_rate": 0.6414473684210527,
            "screen_min_policy_wilson_low": 0.31047019100495765,
            "screen_min_policy_win_rate": 0.390625,
        },
    }


def exact_subset(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    mismatches: dict[str, Any] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                mismatches[key] = {"actual": actual_value, "expected": expected_value}
            else:
                try:
                    exact_subset(actual_value, expected_value, f"{label}.{key}")
                except ValueError as error:
                    mismatches[key] = str(error)
        elif actual_value != expected_value:
            mismatches[key] = {"actual": actual_value, "expected": expected_value}
    if mismatches:
        raise ValueError(f"{label} mismatch: {mismatches}")


def valid_wld(row: Mapping[str, Any], games: int, label: str) -> tuple[int, int, int]:
    try:
        wins = int(row["wins"])
        losses = int(row["losses"])
        draws = int(row["draws"])
        observed_games = int(row["games"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid W-L-D") from error
    if min(wins, losses, draws) < 0 or observed_games != games:
        raise ValueError(f"{label}: bad game count")
    if wins + losses + draws != games:
        raise ValueError(f"{label}: W-L-D does not sum to games")
    return wins, losses, draws


def validate_arm(
    output_dir: Path,
    *,
    candidate: Path,
    candidate_sha256: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    run_config = read_json(run_config_path)
    exact_subset(
        run_config,
        expected_run_config(
            candidate=candidate,
            candidate_sha256=candidate_sha256,
            seed=seed,
        ),
        "run_config",
    )
    selected_opponents = run_config.get("selected_opponents")
    if not isinstance(selected_opponents, list) or tuple(
        row.get("policy_id") for row in selected_opponents
    ) != POLICY_IDS:
        raise ValueError("run_config selected opponent identities/order drift")

    summary = read_json(summary_path)
    exact_subset(
        summary,
        {
            "schema_version": "ptcg-gold-league-h2h-summary-v1",
            "run_signature": run_config.get("run_signature"),
            "league_manifest": str(LEAGUE_MANIFEST),
            "selected_policy_ids": list(POLICY_IDS),
            "final_evidence_phase": "screening",
            "local_only": True,
            "uploads_or_submissions_performed": False,
            "candidate": {
                "checkpoint": str(candidate),
                "checkpoint_sha256": candidate_sha256,
                "deck": str(CANDIDATE_DECK),
                "deck_file_sha256": CANDIDATE_DECK_SHA256,
            },
            "engine": {
                "engine_seed_control": False,
                "strict_even_valid_games_by_candidate_seat_required": True,
            },
            "confirmation": {
                "mode": "none",
                "reason": "confirmation_not_requested",
                "summary": None,
                "failures": {},
                "wall_budget_exhausted": False,
                "confirmation_is_independent_not_cumulative": True,
                "confirmation_replaces_screening_for_promotion": True,
            },
        },
        "summary",
    )
    screening = summary.get("screening")
    if not isinstance(screening, dict):
        raise ValueError("summary.screening missing")
    exact_subset(
        screening,
        {
            "phase": "screening",
            "games_requested_per_policy": GAMES_PER_POLICY,
            "selected_policies": POLICY_COUNT,
            "completed_policies": POLICY_COUNT,
            "coverage_complete": True,
            "strict_even_seat_balance_verified": True,
            "failures": {},
            "wall_budget_exhausted": False,
            "engine_seed_control": False,
        },
        "summary.screening",
    )
    pooled = screening.get("pooled")
    seats = screening.get("by_candidate_seat")
    policies = screening.get("policies")
    if not isinstance(pooled, dict) or not isinstance(seats, dict):
        raise ValueError("summary screening aggregate sections missing")
    valid_wld(pooled, TOTAL_GAMES, "pooled")
    for seat in ("0", "1"):
        row = seats.get(seat)
        if not isinstance(row, dict):
            raise ValueError(f"summary seat {seat} missing")
        valid_wld(row, TOTAL_GAMES_PER_SEAT, f"summary seat {seat}")
    if not isinstance(policies, list) or len(policies) != POLICY_COUNT:
        raise ValueError("summary must contain exactly 19 policy rows")
    policy_rows: dict[str, dict[str, Any]] = {}
    result_paths: set[Path] = set()
    for row in policies:
        if not isinstance(row, dict):
            raise ValueError("summary policy row is not an object")
        policy_id = row.get("policy_id")
        if policy_id not in POLICY_IDS or policy_id in policy_rows:
            raise ValueError(f"summary policy identity invalid: {policy_id!r}")
        if row.get("status") != "complete":
            raise ValueError(f"summary policy incomplete: {policy_id}")
        if row.get("strict_even_seat_balance_verified") is not True:
            raise ValueError(f"summary policy seat balance failed: {policy_id}")
        valid_wld(row, GAMES_PER_POLICY, f"policy {policy_id}")
        by_seat = row.get("by_candidate_seat")
        if not isinstance(by_seat, dict):
            raise ValueError(f"policy {policy_id} seat section missing")
        for seat in ("0", "1"):
            seat_row = by_seat.get(seat)
            if not isinstance(seat_row, dict):
                raise ValueError(f"policy {policy_id} seat {seat} missing")
            valid_wld(
                seat_row,
                GAMES_PER_POLICY_PER_SEAT,
                f"policy {policy_id} seat {seat}",
            )
        result_path_value = row.get("result_path")
        if not isinstance(result_path_value, str):
            raise ValueError(f"policy {policy_id} result path missing")
        result_path = Path(result_path_value).resolve()
        expected_parent = (output_dir / "results/screening").resolve()
        if result_path.parent != expected_parent or not result_path.is_file():
            raise ValueError(f"policy {policy_id} result path escaped/missing")
        result_paths.add(result_path)
        policy_rows[str(policy_id)] = row
    if tuple(row["policy_id"] for row in policies) != POLICY_IDS:
        raise ValueError("summary policy order drift")
    disk_results = {
        path.resolve()
        for path in (output_dir / "results/screening").glob("*.json")
        if not path.name.endswith(".running.json")
    }
    if disk_results != result_paths or len(disk_results) != POLICY_COUNT:
        raise ValueError("per-policy result JSON coverage is not exact")

    invalid_games = 0
    for result_path in sorted(result_paths):
        result = read_json(result_path)
        engine = result.get("engine")
        evaluation = result.get("evaluation")
        result_candidate = result.get("candidate")
        result_opponent = result.get("opponent")
        if not all(
            isinstance(item, dict)
            for item in (engine, evaluation, result_candidate, result_opponent)
        ):
            raise ValueError(f"{result_path}: incomplete evaluator result")
        assert isinstance(engine, dict)
        assert isinstance(evaluation, dict)
        assert isinstance(result_candidate, dict)
        assert isinstance(result_opponent, dict)
        exact_subset(
            engine,
            {
                "games_requested": GAMES_PER_POLICY,
                "environments": 32,
                "max_game_decisions": 1000,
                "engine_seed_control": False,
                "device": "cuda",
            },
            f"{result_path}.engine",
        )
        exact_subset(
            result_candidate,
            {
                "path": str(candidate),
                "sha256": candidate_sha256,
                "deck": str(CANDIDATE_DECK),
                "canonical_order": False,
            },
            f"{result_path}.candidate",
        )
        if result_opponent.get("canonical_order") is not True:
            raise ValueError(f"{result_path}: opponent is not canonical")
        if int(evaluation.get("valid_games", -1)) != GAMES_PER_POLICY:
            raise ValueError(f"{result_path}: valid games are incomplete")
        invalid_games += int(evaluation.get("invalid_games", -1))
        by_seat = evaluation.get("by_candidate_seat")
        balance = evaluation.get("seat_balance")
        if not isinstance(by_seat, dict) or not isinstance(balance, dict):
            raise ValueError(f"{result_path}: evaluator seat audit missing")
        if any(
            int(by_seat.get(seat, {}).get("valid_games", -1))
            != GAMES_PER_POLICY_PER_SEAT
            for seat in ("0", "1")
        ):
            raise ValueError(f"{result_path}: evaluator seat quota failed")
        if balance.get("strict_even_balance_verified") is not True:
            raise ValueError(f"{result_path}: evaluator seat balance failed")
        if balance.get("aggregate_equals_seat_sum_verified") is not True:
            raise ValueError(f"{result_path}: evaluator seat accounting failed")
    if invalid_games != 0:
        raise ValueError(f"arm contains {invalid_games} invalid games")

    sorted_by_wins = sorted(
        policy_rows.values(), key=lambda row: (int(row["wins"]), str(row["policy_id"]))
    )
    bottom_five = sorted_by_wins[:5]
    marnie_rows = [policy_rows[policy_id] for policy_id in MARNIE_POLICY_IDS]
    hard_rows = [policy_rows[policy_id] for policy_id in FIXED_HARD_FIVE]
    observed = {
        "overall_wins": int(pooled["wins"]),
        "marnie_wins": sum(int(row["wins"]) for row in marnie_rows),
        "worst_policy_wins": min(int(row["wins"]) for row in policy_rows.values()),
        "own_dynamic_bottom_five_wins": sum(int(row["wins"]) for row in bottom_five),
        "own_dynamic_bottom_five_policy_ids": [row["policy_id"] for row in bottom_five],
        "candidate_seat_0_wins": int(seats["0"]["wins"]),
        "candidate_seat_1_wins": int(seats["1"]["wins"]),
        "fixed_hard_five_wins": sum(int(row["wins"]) for row in hard_rows),
    }
    integrity = {
        "pass": True,
        "completed_policies": POLICY_COUNT,
        "result_jsons": len(result_paths),
        "valid_games": TOTAL_GAMES,
        "invalid_games": invalid_games,
        "games_per_policy": GAMES_PER_POLICY,
        "games_per_policy_per_candidate_seat": GAMES_PER_POLICY_PER_SEAT,
        "candidate_seat_0_games": TOTAL_GAMES_PER_SEAT,
        "candidate_seat_1_games": TOTAL_GAMES_PER_SEAT,
        "strict_even_seat_balance": True,
        "failures": 0,
        "wall_budget_exhausted": False,
        "candidate_dynamic": True,
        "opponent_canonical": True,
    }
    evidence = {
        "run_config": root_relative(run_config_path),
        "run_config_sha256": sha256_file(run_config_path),
        "summary": root_relative(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "run_signature": summary.get("run_signature"),
        "integrity": integrity,
        "observed": observed,
    }
    return evidence, observed


def failed_arm_evidence(
    output_dir: Path,
    return_code: int | None,
    error: BaseException | None,
) -> dict[str, Any]:
    return {
        "integrity": {"pass": False},
        "output_dir": root_relative(output_dir),
        "runner_return_code": return_code,
        "validation_error": None if error is None else f"{type(error).__name__}: {error}",
        "attempt_consumed": True,
    }


def minimum_gate(name: str, observed: int | None, required: int, category: str) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "rule": "minimum",
        "required": required,
        "observed": observed,
        "passed": observed is not None and observed >= required,
    }


def make_decision(
    plan: dict[str, Any],
    arm_records: Mapping[str, dict[str, Any]],
    metrics: Mapping[str, dict[str, Any] | None],
) -> dict[str, Any]:
    control_metrics = metrics.get("control")
    candidate_metrics = metrics.get("candidate")
    control_integrity = bool(
        arm_records.get("control", {}).get("integrity", {}).get("pass")
    )
    candidate_integrity = bool(
        arm_records.get("candidate", {}).get("integrity", {}).get("pass")
    )
    gates: list[dict[str, Any]] = [
        {
            "name": "control_integrity",
            "category": "integrity",
            "rule": "all_frozen_integrity_checks",
            "required": True,
            "observed": control_integrity,
            "passed": control_integrity,
        },
        {
            "name": "candidate_integrity",
            "category": "integrity",
            "rule": "all_frozen_integrity_checks",
            "required": True,
            "observed": candidate_integrity,
            "passed": candidate_integrity,
        },
    ]
    absolute_keys = {
        "candidate_overall_wins": "overall_wins",
        "candidate_marnie_wins": "marnie_wins",
        "candidate_worst_policy_wins": "worst_policy_wins",
        "candidate_own_dynamic_bottom_five_wins": "own_dynamic_bottom_five_wins",
        "candidate_seat_0_wins": "candidate_seat_0_wins",
        "candidate_seat_1_wins": "candidate_seat_1_wins",
    }
    for gate_name, metric_name in absolute_keys.items():
        observed = (
            int(candidate_metrics[metric_name])
            if isinstance(candidate_metrics, dict)
            else None
        )
        gates.append(
            minimum_gate(
                gate_name,
                observed,
                ABSOLUTE_MINIMUMS[gate_name],
                "candidate_absolute",
            )
        )
    delta_keys = {
        "candidate_minus_control_overall_wins": "overall_wins",
        "candidate_minus_control_marnie_wins": "marnie_wins",
        "candidate_minus_control_own_dynamic_bottom_five_wins": "own_dynamic_bottom_five_wins",
        "candidate_minus_control_seat_0_wins": "candidate_seat_0_wins",
        "candidate_minus_control_seat_1_wins": "candidate_seat_1_wins",
        "candidate_minus_control_fixed_hard_five_wins": "fixed_hard_five_wins",
    }
    deltas: dict[str, int] | None = None
    if isinstance(control_metrics, dict) and isinstance(candidate_metrics, dict):
        deltas = {
            metric_name: int(candidate_metrics[metric_name]) - int(control_metrics[metric_name])
            for metric_name in delta_keys.values()
        }
    for gate_name, metric_name in delta_keys.items():
        observed = deltas[metric_name] if deltas is not None else None
        gates.append(
            minimum_gate(
                gate_name,
                observed,
                DELTA_MINIMUMS[gate_name],
                "candidate_minus_fresh_control",
            )
        )
    if len(gates) != 14 or len({gate["name"] for gate in gates}) != 14:
        raise AssertionError("Formal Gold19 gate construction is not exactly fourteen")
    failed = [gate["name"] for gate in gates if not gate["passed"]]
    all_pass = not failed
    return {
        "schema_version": DECISION_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "screen_passed" if all_pass else "screen_failed_or_incomplete",
        "screen_pass": all_pass,
        "attempt_consumed": True,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        "arms": dict(arm_records),
        "observed": {
            "control": control_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_fresh_control": deltas,
        },
        "fixed_subsets": {
            "marnie_policy_ids": list(MARNIE_POLICY_IDS),
            "fixed_hard_five_policy_ids": list(FIXED_HARD_FIVE),
            "dynamic_bottom_five_is_computed_separately_for_each_arm": True,
        },
        "formal_gates": gates,
        "gate_summary": {
            "passed_gate_count": len(gates) - len(failed),
            "required_gate_count": 14,
            "failed_gates": failed,
            "all_14_gates_pass": all_pass,
        },
        "decision": {
            "candidate_promoted_by_screen": all_pass,
            "candidate_becomes_incumbent": False,
            "candidate_submit_ready": False,
            "confirm512_executed": False,
            "separate_confirm512_preregistration_creation_authorized": all_pass,
            "package_upload_or_submission_authorized": False,
        },
        "rules_applied": {
            "formal_runs_order": ["fresh_u456_control", "fresh_e904_candidate"],
            "candidate_run_attempted_even_if_control_failed": True,
            "all_14_gates_required": True,
            "runner_promotion_diagnostics_ignored": True,
            "no_resume": True,
            "no_retry": True,
            "no_prior_sample_pooling": True,
            "no_seed_selection": True,
            "engine_seed_control": False,
            "arms_are_independent_not_paired": True,
        },
        "scope": {
            "local_only": True,
            "network": False,
            "package": False,
            "upload": False,
            "submission": False,
        },
    }


def clean_subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in PYTHON_ENV_KEYS or key.upper().startswith("PYTHON"):
            environment.pop(key, None)
    remaining = {
        key: value
        for key, value in environment.items()
        if key.upper().startswith("PYTHON")
    }
    if remaining:
        raise RuntimeError(f"Python subprocess environment scrub failed: {remaining}")
    return environment


def run_arm(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> tuple[int | None, str | None]:
    try:
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                list(command),
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
                env=clean_subprocess_environment(),
            )
        return int(completed.returncode), None
    except BaseException as error:  # Record the consumed attempt; never retry.
        return None, f"{type(error).__name__}: {error}"


def execute(
    candidate: Path,
    candidate_file_sha256: str,
    output_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    os.mkdir(output_root, 0o755)
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "created_at_utc": utc_now(),
        "status": "attempt_consumed_before_first_gold_process",
        "plan": plan,
        "plan_sha256": hashlib.sha256(canonical_json_bytes(plan)).hexdigest(),
        "no_resume_no_retry": True,
        "uploads_or_submissions_performed": False,
    }
    exclusive_write_json(output_root / "ATTEMPT.json", attempt)

    arm_specs = (
        ("control", CONTROL_CHECKPOINT, CONTROL_FILE_SHA256, CONTROL_SEED),
        ("candidate", candidate, candidate_file_sha256, CANDIDATE_SEED),
    )
    arm_records: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any] | None] = {}
    for arm, checkpoint, checkpoint_sha256, seed in arm_specs:
        # Recheck all mutable inputs immediately before each arm.  Seed freshness
        # is checked only before creating ATTEMPT.json; the attempt itself now
        # legitimately contains both seed strings.
        before = verify_static_bindings(
            candidate,
            candidate_file_sha256,
            check_seed_freshness=False,
        )
        output_dir = arm_output_dir(output_root, arm)
        if os.path.lexists(output_dir):
            raise FileExistsError(f"One-shot arm output already exists: {output_dir}")
        command = build_command(candidate=checkpoint, output_dir=output_dir, seed=seed)
        stdout_path = output_root / f"{arm}.runner.stdout.log"
        stderr_path = output_root / f"{arm}.runner.stderr.log"
        return_code, launch_error = run_arm(command, stdout_path, stderr_path)
        record: dict[str, Any] = {
            "order": 1 if arm == "control" else 2,
            "seed": seed,
            "checkpoint": root_relative(checkpoint),
            "checkpoint_file_sha256": checkpoint_sha256,
            "output_dir": root_relative(output_dir),
            "command": list(command),
            "runner_return_code": return_code,
            "launch_error": launch_error,
            "attempt_consumed": True,
            "bindings_immediately_before_arm": before,
            "runner_stdout": root_relative(stdout_path),
            "runner_stderr": root_relative(stderr_path),
        }
        validation_error: BaseException | None = None
        observed: dict[str, Any] | None = None
        if return_code == 0 and launch_error is None:
            try:
                evidence, observed = validate_arm(
                    output_dir,
                    candidate=checkpoint.resolve(),
                    candidate_sha256=checkpoint_sha256,
                    seed=seed,
                )
                record.update(evidence)
            except BaseException as error:
                validation_error = error
        if observed is None:
            record.update(failed_arm_evidence(output_dir, return_code, validation_error))
        arm_records[arm] = record
        metrics[arm] = observed
        exclusive_write_json(output_root / f"{arm}.receipt.json", record)

    # Bind the inputs once more after both independent arms to detect drift that
    # occurred during execution.  Any failure is reflected by a consumed-attempt
    # terminal record; the wrapper never launches another Gold sample.
    postflight_error: str | None = None
    try:
        postflight = verify_static_bindings(
            candidate,
            candidate_file_sha256,
            check_seed_freshness=False,
        )
    except BaseException as error:
        postflight = None
        postflight_error = f"{type(error).__name__}: {error}"
        for arm in arm_records.values():
            arm["integrity"] = {"pass": False, "postflight_binding_error": postflight_error}

    decision = make_decision(plan, arm_records, metrics)
    decision["postflight_bindings"] = postflight
    decision["postflight_binding_error"] = postflight_error
    if postflight_error is not None:
        decision["status"] = "screen_failed_or_incomplete"
        decision["screen_pass"] = False
        decision["gate_summary"]["all_14_gates_pass"] = False
        for gate in decision["formal_gates"][:2]:
            gate["observed"] = False
            gate["passed"] = False
        decision["gate_summary"]["failed_gates"] = [
            gate["name"] for gate in decision["formal_gates"] if not gate["passed"]
        ]
        decision["gate_summary"]["passed_gate_count"] = (
            14 - len(decision["gate_summary"]["failed_gates"])
        )
        decision["decision"]["candidate_promoted_by_screen"] = False
        decision["decision"]["separate_confirm512_preregistration_creation_authorized"] = False
    exclusive_write_json(output_root / "formal_decision.json", decision)
    os.chmod(output_root, 0o555)
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or explicitly execute the one-shot frozen U456/E904 "
            "Gold19 19x128 screen pair. Default: preflight only."
        )
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="consume the one-shot attempt and launch U456 then E904 on CUDA",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = args.candidate.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    output_root = validate_output_root(args.output_root, require_absent=True)
    bindings = verify_static_bindings(
        candidate,
        args.candidate_file_sha256,
        check_seed_freshness=True,
    )
    plan = execution_plan(
        candidate,
        args.candidate_file_sha256,
        output_root,
        bindings,
    )
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0
    decision = execute(
        candidate,
        args.candidate_file_sha256,
        output_root,
        plan,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    # A completed formal rejection is not an infrastructure error and must not
    # invite a retry.  The immutable decision carries pass/fail.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
