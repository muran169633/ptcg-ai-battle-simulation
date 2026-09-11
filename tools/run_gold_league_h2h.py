#!/usr/bin/env python3
"""Run a local, resumable candidate-vs-gold-clone evaluation league.

The runner consumes ``ptcg-ppo-opponent-league-v1`` manifests emitted by
``run_gold_clone_league.py`` and invokes ``evaluate_ppo_head_to_head.py`` once
per selected clone and phase. It never calls Kaggle, uploads, or submits.

The official engine does not expose RNG seed control. ``--seed`` only seeds
Python/Torch around deterministic policy inference; it must not be interpreted
as a reproducible or paired battle-engine seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "tools" / "evaluate_ppo_head_to_head.py"
LEAGUE_SCHEMA = "ptcg-ppo-opponent-league-v1"
RUN_SCHEMA = "ptcg-gold-league-h2h-run-v1"
SUMMARY_SCHEMA = "ptcg-gold-league-h2h-summary-v1"
DEPLOYMENT_SCHEMA = "ptcg-gold-push-deployment-contract-v1"
PHASES = ("screening", "confirmation")
ORDER_MODES = ("raw", "canonical", "hybrid")
ENGINE_WARNING = (
    "engine_seed_control=false: Python/Torch seeds do not control the official "
    "battle engine RNG. Repeated runs are independent samples, not "
    "reproducible or paired engine-seed trials."
)


@dataclass(frozen=True)
class OpponentSpec:
    policy_id: str
    submission_id: int | None
    team_name: str
    archetype: str
    checkpoint: Path
    checkpoint_sha256: str
    deck: Path
    deck_file_sha256: str
    deck_hash: str
    canonical_order: bool
    quality: dict[str, Any] | None


@dataclass(frozen=True)
class EvalTask:
    opponent: OpponentSpec
    phase: str
    games: int
    seed: int
    result_path: Path
    command: tuple[str, ...]


@dataclass
class RunningTask:
    task: EvalTask
    process: subprocess.Popen[str]
    temporary_output: Path
    stdout_handle: Any
    stderr_handle: Any
    stdout_path: Path
    stderr_path: Path
    started_monotonic: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_deployment_contract(
    path: Path,
    *,
    args: argparse.Namespace,
    candidate_sha256: str,
    candidate_deck_file_sha256: str,
    order_mode: str,
) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError(f"{path}: unexpected deployment contract schema")
    candidate = contract.get("candidate")
    deck = contract.get("candidate_deck")
    action_order = contract.get("action_order")
    panel = contract.get("panel")
    if not all(isinstance(row, dict) for row in (candidate, deck, action_order, panel)):
        raise ValueError(f"{path}: deployment contract sections are missing")
    assert isinstance(candidate, dict)
    assert isinstance(deck, dict)
    assert isinstance(action_order, dict)
    assert isinstance(panel, dict)
    expected = {
        "candidate.path": (
            resolve_path(candidate.get("path"), path.parent, "candidate.path"),
            args.candidate,
        ),
        "candidate.sha256": (candidate.get("sha256"), candidate_sha256),
        "candidate_deck.path": (
            resolve_path(deck.get("path"), path.parent, "candidate_deck.path"),
            args.candidate_deck,
        ),
        "candidate_deck.file_sha256": (
            deck.get("file_sha256"),
            candidate_deck_file_sha256,
        ),
        "action_order.mode": (action_order.get("mode"), order_mode),
        "action_order.canonical_order": (
            action_order.get("canonical_order"),
            order_mode == "canonical",
        ),
        "action_order.hybrid_order": (
            action_order.get("hybrid_order"),
            order_mode == "hybrid",
        ),
        "panel.manifest": (
            resolve_path(panel.get("manifest"), path.parent, "panel.manifest"),
            args.league_manifest,
        ),
        "panel.manifest_sha256": (
            panel.get("manifest_sha256"),
            file_sha256(args.league_manifest),
        ),
        "panel.output_dir": (
            resolve_path(panel.get("output_dir"), path.parent, "panel.output_dir"),
            args.output_dir,
        ),
    }
    mismatches = {
        label: {"actual": actual, "expected": expected_value}
        for label, (actual, expected_value) in expected.items()
        if actual != expected_value
    }
    if mismatches:
        raise ValueError(f"Deployment contract mismatch: {mismatches}")
    return contract


def resolve_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_sha256(value: Any, label: str) -> str:
    parsed = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", parsed):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return parsed


def safe_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{(slug[:72] or 'policy')}-{suffix}"


def load_league_manifest(path: Path) -> tuple[dict[str, Any], list[OpponentSpec]]:
    manifest = read_json(path)
    if manifest.get("schema_version") != LEAGUE_SCHEMA:
        raise ValueError(
            f"{path}: unsupported schema {manifest.get('schema_version')!r}; "
            f"expected {LEAGUE_SCHEMA!r}"
        )
    raw_opponents = manifest.get("opponents")
    if not isinstance(raw_opponents, list) or not raw_opponents:
        raise ValueError(f"{path}: opponents must be a non-empty list")

    opponents: list[OpponentSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_opponents):
        if not isinstance(raw, dict):
            raise ValueError(f"opponents[{index}] must be an object")
        policy_id = raw.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError(f"opponents[{index}].policy_id must be non-empty")
        if policy_id in seen:
            raise ValueError(f"Duplicate policy_id {policy_id!r}")
        seen.add(policy_id)
        quality = raw.get("quality")
        if isinstance(quality, dict) and quality.get("pass") is not True:
            raise ValueError(
                f"opponents[{index}] is not a quality-qualified clone"
            )
        checkpoint = resolve_path(
            raw.get("checkpoint"),
            path.parent,
            f"opponents[{index}].checkpoint",
        )
        deck = resolve_path(
            raw.get("deck"),
            path.parent,
            f"opponents[{index}].deck",
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not deck.is_file():
            raise FileNotFoundError(deck)
        declared_checkpoint_sha = require_sha256(
            raw.get("checkpoint_sha256"),
            f"opponents[{index}].checkpoint_sha256",
        )
        actual_checkpoint_sha = file_sha256(checkpoint)
        if actual_checkpoint_sha != declared_checkpoint_sha:
            raise ValueError(
                f"{policy_id}: clone checkpoint SHA-256 does not match "
                "league_manifest"
            )
        raw_submission_id = raw.get("submission_id")
        submission_id = (
            int(raw_submission_id) if raw_submission_id is not None else None
        )
        canonical_order = raw.get("canonical_order")
        if type(canonical_order) is not bool:
            raise ValueError(
                f"opponents[{index}].canonical_order must be an explicit "
                "boolean"
            )
        opponents.append(
            OpponentSpec(
                policy_id=policy_id,
                submission_id=submission_id,
                team_name=str(raw.get("team_name", "")),
                archetype=str(raw.get("archetype", "unknown")),
                checkpoint=checkpoint,
                checkpoint_sha256=actual_checkpoint_sha,
                deck=deck,
                deck_file_sha256=file_sha256(deck),
                deck_hash=require_sha256(
                    raw.get("deck_hash"),
                    f"opponents[{index}].deck_hash",
                ),
                canonical_order=canonical_order,
                quality=quality if isinstance(quality, dict) else None,
            )
        )
    return manifest, opponents


def select_opponents(
    opponents: list[OpponentSpec],
    *,
    policy_ids: Iterable[str],
    archetypes: Iterable[str],
    policy_regex: str | None,
    max_policies: int | None,
) -> list[OpponentSpec]:
    requested_ids = set(policy_ids)
    known_ids = {opponent.policy_id for opponent in opponents}
    unknown = sorted(requested_ids - known_ids)
    if unknown:
        raise ValueError(f"Unknown --policy-id values: {unknown}")
    requested_archetypes = set(archetypes)
    known_archetypes = {opponent.archetype for opponent in opponents}
    unknown_archetypes = sorted(requested_archetypes - known_archetypes)
    if unknown_archetypes:
        raise ValueError(
            f"Unknown --archetype values: {unknown_archetypes}"
        )
    pattern = re.compile(policy_regex) if policy_regex else None
    selected = [
        opponent
        for opponent in opponents
        if (not requested_ids or opponent.policy_id in requested_ids)
        and (
            not requested_archetypes
            or opponent.archetype in requested_archetypes
        )
        and (pattern is None or pattern.search(opponent.policy_id))
    ]
    if max_policies is not None:
        selected = selected[:max_policies]
    if not selected:
        raise ValueError("No opponents selected")
    return selected


def wilson_interval(
    successes: int,
    trials: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - margin, center + margin


def wld_metrics(wins: int, losses: int, draws: int) -> dict[str, Any]:
    games = wins + losses + draws
    low, high = wilson_interval(wins, games)
    decisive = wins + losses
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else None,
        "decisive_win_rate": wins / decisive if decisive else None,
        "wilson_95_low": low if games else None,
        "wilson_95_high": high if games else None,
    }


def cvar_lower_tail(values: list[float], alpha: float) -> dict[str, Any]:
    if not values:
        return {"alpha": alpha, "count": 0, "value": None}
    count = max(1, math.ceil(len(values) * alpha))
    ordered = sorted(values)
    return {
        "alpha": alpha,
        "count": count,
        "value": sum(ordered[:count]) / count,
    }


def result_path(output_dir: Path, phase: str, policy_id: str) -> Path:
    return output_dir / "results" / phase / f"{safe_name(policy_id)}.json"


def derived_seed(base_seed: int, phase: str, policy_id: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{phase}:{policy_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def candidate_order_mode(args: argparse.Namespace) -> str:
    enabled = {
        "raw": bool(getattr(args, "candidate_raw_order", False)),
        "canonical": bool(args.candidate_canonical_order),
        "hybrid": bool(getattr(args, "candidate_hybrid_order", False)),
    }
    selected = [mode for mode, active in enabled.items() if active]
    if len(selected) != 1:
        raise ValueError(
            "Exactly one explicit candidate action-order mode is required: "
            "--candidate-raw-order, --candidate-canonical-order, or "
            "--candidate-hybrid-order"
        )
    return selected[0]


def build_eval_command(
    args: argparse.Namespace,
    opponent: OpponentSpec,
    *,
    phase: str,
    games: int,
    output: Path,
) -> tuple[str, ...]:
    order_mode = candidate_order_mode(args)
    if args.opponent_canonical_order and not opponent.canonical_order:
        raise ValueError(
            "--opponent-canonical-order is a deprecated all-true assertion "
            f"and conflicts with {opponent.policy_id!r}, whose manifest "
            "canonical_order is false"
        )
    command = [
        sys.executable,
        str(EVALUATOR),
        "--candidate",
        str(args.candidate),
        "--opponent",
        str(opponent.checkpoint),
        "--bc-checkpoint",
        str(args.bc_checkpoint),
        "--candidate-deck",
        str(args.candidate_deck),
        "--opponent-deck",
        str(opponent.deck),
        "--games",
        str(games),
        "--environments",
        str(args.environments),
        "--max-game-decisions",
        str(args.max_game_decisions),
        "--seed",
        str(derived_seed(args.seed, phase, opponent.policy_id)),
        "--device",
        args.device,
        "--output",
        str(output),
    ]
    if order_mode == "canonical":
        command.append("--candidate-canonical-order")
    elif order_mode == "hybrid":
        command.append("--candidate-hybrid-order")
    if opponent.canonical_order:
        command.append("--opponent-canonical-order")
    return tuple(command)


def validate_evaluator_result(
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    opponent: OpponentSpec,
    phase: str,
    games: int,
    candidate_sha256: str,
) -> None:
    candidate = result.get("candidate")
    evaluated_opponent = result.get("opponent")
    engine = result.get("engine")
    evaluation = result.get("evaluation")
    if not all(
        isinstance(item, dict)
        for item in (candidate, evaluated_opponent, engine, evaluation)
    ):
        raise ValueError("Evaluator result is missing top-level sections")
    assert isinstance(candidate, dict)
    assert isinstance(evaluated_opponent, dict)
    assert isinstance(engine, dict)
    assert isinstance(evaluation, dict)
    expected_candidate_path = str(args.candidate.resolve())
    expected_candidate_deck = str(args.candidate_deck.resolve())
    expected_opponent_path = str(opponent.checkpoint.resolve())
    expected_opponent_deck = str(opponent.deck.resolve())
    expected_candidate_order = candidate_order_mode(args)
    expected_opponent_order = (
        "canonical" if opponent.canonical_order else "raw"
    )
    if type(candidate.get("canonical_order")) is not bool:
        raise ValueError(
            "Evaluator result candidate.canonical_order must be a boolean"
        )
    if type(candidate.get("hybrid_order")) is not bool:
        raise ValueError(
            "Evaluator result candidate.hybrid_order must be a boolean"
        )
    if type(evaluated_opponent.get("canonical_order")) is not bool:
        raise ValueError(
            "Evaluator result opponent.canonical_order must be a boolean"
        )
    if type(evaluated_opponent.get("hybrid_order")) is not bool:
        raise ValueError(
            "Evaluator result opponent.hybrid_order must be a boolean"
        )
    expected = {
        "candidate.path": (candidate.get("path"), expected_candidate_path),
        "candidate.sha256": (candidate.get("sha256"), candidate_sha256),
        "candidate.deck": (candidate.get("deck"), expected_candidate_deck),
        "candidate.canonical_order": (
            candidate.get("canonical_order"),
            expected_candidate_order == "canonical",
        ),
        "candidate.hybrid_order": (
            candidate.get("hybrid_order"),
            expected_candidate_order == "hybrid",
        ),
        "candidate.order_mode": (
            candidate.get("order_mode"),
            expected_candidate_order,
        ),
        "opponent.path": (
            evaluated_opponent.get("path"),
            expected_opponent_path,
        ),
        "opponent.sha256": (
            evaluated_opponent.get("sha256"),
            opponent.checkpoint_sha256,
        ),
        "opponent.deck": (
            evaluated_opponent.get("deck"),
            expected_opponent_deck,
        ),
        "opponent.canonical_order": (
            evaluated_opponent.get("canonical_order"),
            opponent.canonical_order,
        ),
        "opponent.hybrid_order": (
            evaluated_opponent.get("hybrid_order"),
            False,
        ),
        "opponent.order_mode": (
            evaluated_opponent.get("order_mode"),
            expected_opponent_order,
        ),
        "engine.games_requested": (engine.get("games_requested"), games),
        "engine.environments": (
            engine.get("environments"),
            args.environments,
        ),
        "engine.max_game_decisions": (
            engine.get("max_game_decisions"),
            args.max_game_decisions,
        ),
        "engine.python_torch_seed": (
            engine.get("python_torch_seed"),
            derived_seed(args.seed, phase, opponent.policy_id),
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise ValueError(f"Evaluator result identity mismatch: {mismatches}")
    if engine.get("engine_seed_control") is not False:
        raise ValueError("Evaluator must explicitly report engine_seed_control=false")

    wins = int(evaluation.get("wins", -1))
    losses = int(evaluation.get("losses", -1))
    draws = int(evaluation.get("draws", -1))
    valid_games = int(evaluation.get("valid_games", -1))
    if valid_games != games or wins + losses + draws != valid_games:
        raise ValueError("Evaluator result has incomplete aggregate W-L-D")
    by_seat = evaluation.get("by_candidate_seat")
    seat_balance = evaluation.get("seat_balance")
    if not isinstance(by_seat, dict) or not isinstance(seat_balance, dict):
        raise ValueError("Evaluator result lacks candidate-seat audit")
    expected_seat_games = games // 2
    seat_totals = {"wins": 0, "losses": 0, "draws": 0, "valid_games": 0}
    for seat in ("0", "1"):
        row = by_seat.get(seat)
        if not isinstance(row, dict):
            raise ValueError(f"Evaluator result lacks candidate seat {seat}")
        seat_games = int(row.get("valid_games", -1))
        seat_wins = int(row.get("wins", -1))
        seat_losses = int(row.get("losses", -1))
        seat_draws = int(row.get("draws", -1))
        if (
            seat_games != expected_seat_games
            or seat_wins + seat_losses + seat_draws != seat_games
        ):
            raise ValueError(
                f"Candidate seat {seat} is not exactly balanced and complete"
            )
        for key, value in (
            ("wins", seat_wins),
            ("losses", seat_losses),
            ("draws", seat_draws),
            ("valid_games", seat_games),
        ):
            seat_totals[key] += value
    if seat_totals != {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "valid_games": valid_games,
    }:
        raise ValueError("Aggregate W-L-D does not equal candidate-seat sums")
    if seat_balance.get("strict_even_balance_verified") is not True:
        raise ValueError("Evaluator did not verify strict even seat balance")
    if seat_balance.get("aggregate_equals_seat_sum_verified") is not True:
        raise ValueError("Evaluator did not verify aggregate seat accounting")


def tail_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def make_task(
    args: argparse.Namespace,
    opponent: OpponentSpec,
    *,
    phase: str,
    games: int,
) -> EvalTask:
    destination = result_path(args.output_dir, phase, opponent.policy_id)
    temporary = destination.with_suffix(".running.json")
    return EvalTask(
        opponent=opponent,
        phase=phase,
        games=games,
        seed=derived_seed(args.seed, phase, opponent.policy_id),
        result_path=destination,
        command=build_eval_command(
            args,
            opponent,
            phase=phase,
            games=games,
            output=temporary,
        ),
    )


def load_valid_result(
    path: Path,
    *,
    args: argparse.Namespace,
    opponent: OpponentSpec,
    phase: str,
    games: int,
    candidate_sha256: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = read_json(path)
        validate_evaluator_result(
            result,
            args=args,
            opponent=opponent,
            phase=phase,
            games=games,
            candidate_sha256=candidate_sha256,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return result


def terminate_running(running: list[RunningTask]) -> None:
    for item in running:
        if item.process.poll() is None:
            item.process.terminate()
    deadline = time.monotonic() + 5.0
    while any(item.process.poll() is None for item in running):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for item in running:
        if item.process.poll() is None:
            item.process.kill()
    for item in running:
        try:
            item.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        item.stdout_handle.close()
        item.stderr_handle.close()


def execute_phase(
    args: argparse.Namespace,
    opponents: list[OpponentSpec],
    *,
    phase: str,
    games: int,
    candidate_sha256: str,
    deadline: float | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], bool]:
    results: dict[str, dict[str, Any]] = {}
    tasks: list[EvalTask] = []
    failures: dict[str, dict[str, Any]] = {}
    for opponent in opponents:
        destination = result_path(args.output_dir, phase, opponent.policy_id)
        existing = (
            load_valid_result(
                destination,
                args=args,
                opponent=opponent,
                phase=phase,
                games=games,
                candidate_sha256=candidate_sha256,
            )
            if args.resume
            else None
        )
        if existing is not None:
            results[opponent.policy_id] = existing
        else:
            tasks.append(
                make_task(
                    args,
                    opponent,
                    phase=phase,
                    games=games,
                )
            )

    pending = list(tasks)
    running: list[RunningTask] = []
    wall_exhausted = False
    try:
        while pending or running:
            if deadline is not None and time.monotonic() >= deadline:
                wall_exhausted = True
                break
            while pending and len(running) < args.jobs:
                if deadline is not None and time.monotonic() >= deadline:
                    wall_exhausted = True
                    break
                task = pending.pop(0)
                task.result_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = task.result_path.with_suffix(".running.json")
                if temporary.is_file():
                    temporary.unlink()
                stdout_path = task.result_path.with_suffix(".stdout.log")
                stderr_path = task.result_path.with_suffix(".stderr.log")
                stdout_handle = stdout_path.open("w", encoding="utf-8")
                stderr_handle = stderr_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    task.command,
                    cwd=REPO_ROOT,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
                running.append(
                    RunningTask(
                        task=task,
                        process=process,
                        temporary_output=temporary,
                        stdout_handle=stdout_handle,
                        stderr_handle=stderr_handle,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        started_monotonic=time.monotonic(),
                    )
                )

            finished = [
                item for item in running if item.process.poll() is not None
            ]
            for item in finished:
                running.remove(item)
                item.stdout_handle.close()
                item.stderr_handle.close()
                return_code = int(item.process.returncode or 0)
                if return_code != 0:
                    failures[item.task.opponent.policy_id] = {
                        "phase": phase,
                        "return_code": return_code,
                        "elapsed_seconds": (
                            time.monotonic() - item.started_monotonic
                        ),
                        "stderr_tail": tail_text(item.stderr_path),
                    }
                    continue
                try:
                    result = read_json(item.temporary_output)
                    validate_evaluator_result(
                        result,
                        args=args,
                        opponent=item.task.opponent,
                        phase=phase,
                        games=games,
                        candidate_sha256=candidate_sha256,
                    )
                except (
                    OSError,
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                ) as error:
                    failures[item.task.opponent.policy_id] = {
                        "phase": phase,
                        "return_code": return_code,
                        "elapsed_seconds": (
                            time.monotonic() - item.started_monotonic
                        ),
                        "validation_error": str(error),
                        "stderr_tail": tail_text(item.stderr_path),
                    }
                    continue
                os.replace(item.temporary_output, item.task.result_path)
                results[item.task.opponent.policy_id] = result
            if not finished and running:
                time.sleep(0.1)
    finally:
        if running:
            terminate_running(running)
            for item in running:
                failures[item.task.opponent.policy_id] = {
                    "phase": phase,
                    "status": "terminated_at_wall_budget",
                    "elapsed_seconds": (
                        time.monotonic() - item.started_monotonic
                    ),
                    "stderr_tail": tail_text(item.stderr_path),
                }
        if wall_exhausted:
            for task in pending:
                failures[task.opponent.policy_id] = {
                    "phase": phase,
                    "status": "not_started_at_wall_budget",
                }
    return results, failures, wall_exhausted


def evaluation_wld(evaluation: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(evaluation["wins"]),
        int(evaluation["losses"]),
        int(evaluation["draws"]),
    )


def aggregate_phase(
    opponents: list[OpponentSpec],
    results: dict[str, dict[str, Any]],
    *,
    phase: str,
    games_requested_per_policy: int,
    cvar_alpha: float,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    policies: list[dict[str, Any]] = []
    archetype_counts: dict[str, list[int]] = {}
    archetype_seat_counts: dict[str, dict[str, list[int]]] = {}
    seat_counts = {"0": [0, 0, 0], "1": [0, 0, 0]}
    total = [0, 0, 0]
    policy_rates: list[float] = []
    all_strictly_balanced = True

    for opponent in opponents:
        result = results.get(opponent.policy_id)
        if result is None:
            policies.append(
                {
                    "policy_id": opponent.policy_id,
                    "submission_id": opponent.submission_id,
                    "team_name": opponent.team_name,
                    "archetype": opponent.archetype,
                    "opponent_canonical_order": opponent.canonical_order,
                    "status": "missing",
                }
            )
            all_strictly_balanced = False
            continue
        evaluation = result["evaluation"]
        wins, losses, draws = evaluation_wld(evaluation)
        metrics = wld_metrics(wins, losses, draws)
        assert metrics["win_rate"] is not None
        policy_rates.append(float(metrics["win_rate"]))
        for index, count in enumerate((wins, losses, draws)):
            total[index] += count
        archetype_counts.setdefault(opponent.archetype, [0, 0, 0])
        archetype_seat_counts.setdefault(
            opponent.archetype,
            {"0": [0, 0, 0], "1": [0, 0, 0]},
        )
        for index, count in enumerate((wins, losses, draws)):
            archetype_counts[opponent.archetype][index] += count
        by_seat: dict[str, Any] = {}
        for seat in ("0", "1"):
            seat_row = evaluation["by_candidate_seat"][seat]
            seat_wld = (
                int(seat_row["wins"]),
                int(seat_row["losses"]),
                int(seat_row["draws"]),
            )
            by_seat[seat] = wld_metrics(*seat_wld)
            for index, count in enumerate(seat_wld):
                seat_counts[seat][index] += count
                archetype_seat_counts[opponent.archetype][seat][index] += (
                    count
                )
        strict = bool(
            evaluation["seat_balance"].get(
                "strict_even_balance_verified",
                False,
            )
        )
        all_strictly_balanced = all_strictly_balanced and strict
        policies.append(
            {
                "policy_id": opponent.policy_id,
                "submission_id": opponent.submission_id,
                "team_name": opponent.team_name,
                "archetype": opponent.archetype,
                "opponent_canonical_order": opponent.canonical_order,
                "status": "complete",
                **metrics,
                "by_candidate_seat": by_seat,
                "strict_even_seat_balance_verified": strict,
                "result_path": str(
                    result_path(
                        output_dir,
                        phase,
                        opponent.policy_id,
                    )
                )
                if output_dir is not None
                else None,
            }
        )

    archetypes = [
        {
            "archetype": archetype,
            **wld_metrics(*counts),
            "by_candidate_seat": {
                seat: wld_metrics(*seat_counts_by_archetype)
                for seat, seat_counts_by_archetype in (
                    archetype_seat_counts[archetype].items()
                )
            },
            "policies": sum(
                row["status"] == "complete"
                and row["archetype"] == archetype
                for row in policies
            ),
        }
        for archetype, counts in sorted(archetype_counts.items())
    ]
    complete_count = len(results)
    coverage_complete = complete_count == len(opponents)
    return {
        "phase": phase,
        "games_requested_per_policy": games_requested_per_policy,
        "selected_policies": len(opponents),
        "completed_policies": complete_count,
        "coverage_complete": coverage_complete,
        "strict_even_seat_balance_verified": (
            coverage_complete and all_strictly_balanced
        ),
        "policies": policies,
        "archetypes": archetypes,
        "by_candidate_seat": {
            seat: wld_metrics(*counts)
            for seat, counts in seat_counts.items()
        },
        "pooled": wld_metrics(*total),
        "macro_policy_win_rate": (
            sum(policy_rates) / len(policy_rates) if policy_rates else None
        ),
        "minimum_policy_win_rate": min(policy_rates) if policy_rates else None,
        "policy_win_rate_cvar": cvar_lower_tail(policy_rates, cvar_alpha),
        "engine_seed_control": False,
        "engine_randomness_warning": ENGINE_WARNING,
    }


def screen_gate(
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    complete_policies = [
        row for row in summary["policies"] if row["status"] == "complete"
    ]
    minimum_wilson = min(
        (float(row["wilson_95_low"]) for row in complete_policies),
        default=None,
    )
    gates = {
        "coverage_complete": {
            "actual": summary["coverage_complete"],
            "required": True,
            "pass": summary["coverage_complete"] is True,
        },
        "strict_even_seat_balance": {
            "actual": summary["strict_even_seat_balance_verified"],
            "required": True,
            "pass": summary["strict_even_seat_balance_verified"] is True,
        },
        "minimum_policy_win_rate": {
            "actual": summary["minimum_policy_win_rate"],
            "minimum": args.screen_min_policy_win_rate,
            "pass": (
                summary["minimum_policy_win_rate"] is not None
                and summary["minimum_policy_win_rate"]
                >= args.screen_min_policy_win_rate
            ),
        },
        "minimum_policy_wilson_95_low": {
            "actual": minimum_wilson,
            "minimum": args.screen_min_policy_wilson_low,
            "pass": (
                minimum_wilson is not None
                and minimum_wilson >= args.screen_min_policy_wilson_low
            ),
        },
        "macro_policy_win_rate": {
            "actual": summary["macro_policy_win_rate"],
            "minimum": args.screen_min_macro_win_rate,
            "pass": (
                summary["macro_policy_win_rate"] is not None
                and summary["macro_policy_win_rate"]
                >= args.screen_min_macro_win_rate
            ),
        },
        "policy_win_rate_cvar": {
            "actual": summary["policy_win_rate_cvar"]["value"],
            "minimum": args.screen_min_cvar_win_rate,
            "pass": (
                summary["policy_win_rate_cvar"]["value"] is not None
                and summary["policy_win_rate_cvar"]["value"]
                >= args.screen_min_cvar_win_rate
            ),
        },
    }
    return {
        "pass": all(gate["pass"] for gate in gates.values()),
        "gates": gates,
    }


def promotion_gate(
    summary: dict[str, Any],
    args: argparse.Namespace,
    *,
    confirmation_required: bool,
    confirmation_complete: bool,
) -> dict[str, Any]:
    complete_policies = [
        row for row in summary["policies"] if row["status"] == "complete"
    ]
    policy_wilson = [
        float(row["wilson_95_low"]) for row in complete_policies
    ]
    archetype_wilson = [
        float(row["wilson_95_low"]) for row in summary["archetypes"]
    ]
    seat_wilson = [
        float(seat_row["wilson_95_low"])
        for row in complete_policies
        for seat_row in row["by_candidate_seat"].values()
    ]
    actual_confirmation = (
        confirmation_complete if confirmation_required else True
    )
    gates = {
        "confirmation_complete_when_requested": {
            "actual": actual_confirmation,
            "required": True,
            "pass": actual_confirmation,
        },
        "minimum_policy_count": {
            "actual": summary["completed_policies"],
            "minimum": args.promotion_min_policies,
            "pass": (
                summary["completed_policies"]
                >= args.promotion_min_policies
            ),
        },
        "coverage_complete": {
            "actual": summary["coverage_complete"],
            "required": True,
            "pass": summary["coverage_complete"] is True,
        },
        "strict_even_seat_balance": {
            "actual": summary["strict_even_seat_balance_verified"],
            "required": True,
            "pass": summary["strict_even_seat_balance_verified"] is True,
        },
        "minimum_policy_wilson_95_low": {
            "actual": min(policy_wilson) if policy_wilson else None,
            "minimum": args.promotion_min_policy_wilson_low,
            "pass": bool(policy_wilson)
            and min(policy_wilson)
            >= args.promotion_min_policy_wilson_low,
        },
        "minimum_archetype_wilson_95_low": {
            "actual": min(archetype_wilson) if archetype_wilson else None,
            "minimum": args.promotion_min_archetype_wilson_low,
            "pass": bool(archetype_wilson)
            and min(archetype_wilson)
            >= args.promotion_min_archetype_wilson_low,
        },
        "minimum_policy_seat_wilson_95_low": {
            "actual": min(seat_wilson) if seat_wilson else None,
            "minimum": args.promotion_min_seat_wilson_low,
            "pass": bool(seat_wilson)
            and min(seat_wilson)
            >= args.promotion_min_seat_wilson_low,
        },
        "macro_policy_win_rate": {
            "actual": summary["macro_policy_win_rate"],
            "minimum": args.promotion_min_macro_win_rate,
            "pass": (
                summary["macro_policy_win_rate"] is not None
                and summary["macro_policy_win_rate"]
                >= args.promotion_min_macro_win_rate
            ),
        },
        "minimum_policy_win_rate": {
            "actual": summary["minimum_policy_win_rate"],
            "minimum": args.promotion_min_policy_win_rate,
            "pass": (
                summary["minimum_policy_win_rate"] is not None
                and summary["minimum_policy_win_rate"]
                >= args.promotion_min_policy_win_rate
            ),
        },
        "policy_win_rate_cvar": {
            "actual": summary["policy_win_rate_cvar"]["value"],
            "minimum": args.promotion_min_cvar_win_rate,
            "pass": (
                summary["policy_win_rate_cvar"]["value"] is not None
                and summary["policy_win_rate_cvar"]["value"]
                >= args.promotion_min_cvar_win_rate
            ),
        },
    }
    return {
        "promote": all(gate["pass"] for gate in gates.values()),
        "gates": gates,
        "interpretation": (
            "Promotion requires every gate. Wilson intervals treat wins as "
            "successes and draws as non-wins. Macro/min/CVaR are computed over "
            "per-policy raw win rates; CVaR is the configured lower tail."
        ),
    }


def validate_args(args: argparse.Namespace) -> None:
    candidate_order_mode(args)
    for name in (
        "screening_games",
        "confirmation_games",
        "environments",
        "max_game_decisions",
        "jobs",
        "promotion_min_policies",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("screening_games", "confirmation_games"):
        if getattr(args, name) % 2:
            raise ValueError(
                f"--{name.replace('_', '-')} must be even for strict seat balance"
            )
    if args.confirmation_games < args.screening_games:
        raise ValueError(
            "--confirmation-games must be at least --screening-games"
        )
    if args.max_policies is not None and args.max_policies < 1:
        raise ValueError("--max-policies must be positive")
    if args.wall_seconds is not None and args.wall_seconds <= 0:
        raise ValueError("--wall-seconds must be positive")
    probability_names = (
        "cvar_alpha",
        "screen_min_policy_win_rate",
        "screen_min_policy_wilson_low",
        "screen_min_macro_win_rate",
        "screen_min_cvar_win_rate",
        "promotion_min_policy_wilson_low",
        "promotion_min_archetype_wilson_low",
        "promotion_min_seat_wilson_low",
        "promotion_min_macro_win_rate",
        "promotion_min_policy_win_rate",
        "promotion_min_cvar_win_rate",
    )
    for name in probability_names:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.cvar_alpha <= 0.0:
        raise ValueError("--cvar-alpha must be in (0, 1]")
    if args.resume and args.dry_run:
        raise ValueError("--resume and --dry-run are mutually exclusive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one local PPO/BC candidate against a frozen gold-clone "
            "league. This command never uploads or submits."
        )
    )
    parser.add_argument("--league-manifest", type=Path, required=True)
    parser.add_argument("--deployment-contract", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("artifacts/bc_marnie_luca_orbit_v5/best.pt"),
    )
    parser.add_argument("--policy-id", action="append", default=[])
    parser.add_argument("--archetype", action="append", default=[])
    parser.add_argument("--policy-regex")
    parser.add_argument("--max-policies", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wall-seconds", type=float)

    parser.add_argument("--screening-games", type=int, default=512)
    parser.add_argument(
        "--confirmation",
        choices=("none", "if-screen-pass", "always"),
        default="none",
    )
    parser.add_argument("--confirmation-games", type=int, default=2048)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    candidate_order = parser.add_mutually_exclusive_group(required=True)
    candidate_order.add_argument(
        "--candidate-raw-order",
        action="store_true",
        help="Preserve raw PPO Plackett-Luce order in every context.",
    )
    candidate_order.add_argument(
        "--candidate-canonical-order",
        action="store_true",
        help="Sort every selected action index into canonical order.",
    )
    candidate_order.add_argument(
        "--candidate-hybrid-order",
        action="store_true",
        help=(
            "Preserve Plackett-Luce order only for select.context=34 and "
            "sort every other candidate action."
        ),
    )
    parser.add_argument(
        "--opponent-canonical-order",
        action="store_true",
        help=(
            "Deprecated compatibility assertion: require every selected "
            "opponent manifest entry to set canonical_order=true. The "
            "per-opponent manifest field is always authoritative."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--cvar-alpha", type=float, default=0.25)
    parser.add_argument("--screen-min-policy-win-rate", type=float, default=0.45)
    parser.add_argument(
        "--screen-min-policy-wilson-low",
        type=float,
        default=0.40,
    )
    parser.add_argument("--screen-min-macro-win-rate", type=float, default=0.50)
    parser.add_argument("--screen-min-cvar-win-rate", type=float, default=0.45)

    parser.add_argument("--promotion-min-policies", type=int, default=1)
    parser.add_argument(
        "--promotion-min-policy-wilson-low",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--promotion-min-archetype-wilson-low",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--promotion-min-seat-wilson-low",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--promotion-min-macro-win-rate",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--promotion-min-policy-win-rate",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--promotion-min-cvar-win-rate",
        type=float,
        default=0.50,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.league_manifest = args.league_manifest.expanduser().resolve()
    if args.deployment_contract is not None:
        args.deployment_contract = args.deployment_contract.expanduser().resolve()
    args.candidate = args.candidate.expanduser().resolve()
    args.candidate_deck = args.candidate_deck.expanduser().resolve()
    args.bc_checkpoint = args.bc_checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    validate_args(args)
    for path in (
        args.league_manifest,
        args.candidate,
        args.candidate_deck,
        args.bc_checkpoint,
        EVALUATOR,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    _, all_opponents = load_league_manifest(args.league_manifest)
    opponents = select_opponents(
        all_opponents,
        policy_ids=args.policy_id,
        archetypes=args.archetype,
        policy_regex=args.policy_regex,
        max_policies=args.max_policies,
    )
    if args.opponent_canonical_order:
        conflicting = [
            opponent.policy_id
            for opponent in opponents
            if not opponent.canonical_order
        ]
        if conflicting:
            raise ValueError(
                "--opponent-canonical-order is a deprecated all-true "
                "assertion and conflicts with manifest canonical_order=false "
                f"for: {conflicting}"
            )
    candidate_sha = file_sha256(args.candidate)
    candidate_deck_file_sha = file_sha256(args.candidate_deck)
    bc_sha = file_sha256(args.bc_checkpoint)
    selected_candidate_order_mode = candidate_order_mode(args)
    deployment_contract: dict[str, Any] | None = None
    if args.deployment_contract is not None:
        if not args.deployment_contract.is_file():
            raise FileNotFoundError(args.deployment_contract)
        deployment_contract = validate_deployment_contract(
            args.deployment_contract,
            args=args,
            candidate_sha256=candidate_sha,
            candidate_deck_file_sha256=candidate_deck_file_sha,
            order_mode=selected_candidate_order_mode,
        )
    signature_payload = {
        "schema_version": RUN_SCHEMA,
        "league_manifest": str(args.league_manifest),
        "league_manifest_sha256": file_sha256(args.league_manifest),
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": file_sha256(EVALUATOR),
        "candidate": str(args.candidate),
        "candidate_sha256": candidate_sha,
        "candidate_deck": str(args.candidate_deck),
        "candidate_deck_file_sha256": candidate_deck_file_sha,
        "bc_checkpoint": str(args.bc_checkpoint),
        "bc_checkpoint_sha256": bc_sha,
        "deployment_contract": (
            {
                "path": str(args.deployment_contract),
                "sha256": file_sha256(args.deployment_contract),
                "schema_version": deployment_contract.get("schema_version"),
            }
            if args.deployment_contract is not None
            and deployment_contract is not None
            else None
        ),
        "selected_opponents": [
            {
                "policy_id": opponent.policy_id,
                "checkpoint_sha256": opponent.checkpoint_sha256,
                "deck_file_sha256": opponent.deck_file_sha256,
                "deck_hash": opponent.deck_hash,
                "canonical_order": opponent.canonical_order,
            }
            for opponent in opponents
        ],
        "evaluation": {
            "screening_games": args.screening_games,
            "confirmation": args.confirmation,
            "confirmation_games": args.confirmation_games,
            "environments": args.environments,
            "max_game_decisions": args.max_game_decisions,
            "candidate_order_mode": selected_candidate_order_mode,
            "candidate_raw_order": (
                selected_candidate_order_mode == "raw"
            ),
            "candidate_canonical_order": args.candidate_canonical_order,
            "candidate_hybrid_order": (
                selected_candidate_order_mode == "hybrid"
            ),
            "opponent_canonical_order_source": "per_opponent_manifest",
            "legacy_opponent_canonical_order_all_true_assertion": (
                args.opponent_canonical_order
            ),
            "seed": args.seed,
            "device": args.device,
        },
        "gates": {
            key: getattr(args, key)
            for key in (
                "cvar_alpha",
                "screen_min_policy_win_rate",
                "screen_min_policy_wilson_low",
                "screen_min_macro_win_rate",
                "screen_min_cvar_win_rate",
                "promotion_min_policies",
                "promotion_min_policy_wilson_low",
                "promotion_min_archetype_wilson_low",
                "promotion_min_seat_wilson_low",
                "promotion_min_macro_win_rate",
                "promotion_min_policy_win_rate",
                "promotion_min_cvar_win_rate",
            )
        },
        "engine_seed_control": False,
    }
    signature = canonical_sha256(signature_payload)

    dry_plan = {
        "schema_version": RUN_SCHEMA,
        "dry_run": True,
        "run_signature": signature,
        "engine_seed_control": False,
        "engine_randomness_warning": ENGINE_WARNING,
        "selected_policies": len(opponents),
        "candidate_order_mode": selected_candidate_order_mode,
        "screening_commands": [
            list(
                make_task(
                    args,
                    opponent,
                    phase="screening",
                    games=args.screening_games,
                ).command
            )
            for opponent in opponents
        ],
        "confirmation_mode": args.confirmation,
        "confirmation_commands": [
            list(
                make_task(
                    args,
                    opponent,
                    phase="confirmation",
                    games=args.confirmation_games,
                ).command
            )
            for opponent in opponents
        ]
        if args.confirmation != "none"
        else [],
        "uploads_or_submissions_performed": False,
    }
    if args.dry_run:
        return dry_plan

    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)
    run_config_path = args.output_dir / "run_config.json"
    if args.resume:
        if not run_config_path.is_file():
            raise FileNotFoundError(
                f"{run_config_path}: --resume requires an existing run config"
            )
        existing_config = read_json(run_config_path)
        if existing_config.get("run_signature") != signature:
            raise ValueError(
                "Resume run signature mismatch; inputs, filters, evaluator, "
                "games, ordering, or gates changed"
            )
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} is non-empty; use --resume or a new output dir"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_write_json(
            run_config_path,
            {
                **signature_payload,
                "created_at_utc": utc_now(),
                "run_signature": signature,
                "jobs": args.jobs,
                "wall_seconds_per_invocation": args.wall_seconds,
                "local_only": True,
                "uploads_or_submissions_performed": False,
            },
        )

    started = time.monotonic()
    deadline = (
        started + args.wall_seconds
        if args.wall_seconds is not None
        else None
    )
    screening_results, screening_failures, screen_wall_exhausted = execute_phase(
        args,
        opponents,
        phase="screening",
        games=args.screening_games,
        candidate_sha256=candidate_sha,
        deadline=deadline,
    )
    screening_summary = aggregate_phase(
        opponents,
        screening_results,
        phase="screening",
        games_requested_per_policy=args.screening_games,
        cvar_alpha=args.cvar_alpha,
        output_dir=args.output_dir,
    )
    screening_gate = screen_gate(screening_summary, args)

    confirmation_results: dict[str, dict[str, Any]] = {}
    confirmation_failures: dict[str, dict[str, Any]] = {}
    confirmation_wall_exhausted = False
    confirmation_reason: str
    should_confirm = False
    if args.confirmation == "none":
        confirmation_reason = "confirmation_not_requested"
    elif not screening_summary["coverage_complete"]:
        confirmation_reason = "screening_incomplete"
    elif args.confirmation == "if-screen-pass" and not screening_gate["pass"]:
        confirmation_reason = "screen_gate_failed"
    else:
        should_confirm = True
        confirmation_reason = "confirmation_started"
    if should_confirm:
        if deadline is not None and time.monotonic() >= deadline:
            confirmation_wall_exhausted = True
            confirmation_reason = "wall_budget_exhausted_before_confirmation"
        else:
            (
                confirmation_results,
                confirmation_failures,
                confirmation_wall_exhausted,
            ) = execute_phase(
                args,
                opponents,
                phase="confirmation",
                games=args.confirmation_games,
                candidate_sha256=candidate_sha,
                deadline=deadline,
            )
            confirmation_reason = (
                "confirmation_complete"
                if len(confirmation_results) == len(opponents)
                else "confirmation_incomplete"
            )

    confirmation_summary = (
        aggregate_phase(
            opponents,
            confirmation_results,
            phase="confirmation",
            games_requested_per_policy=args.confirmation_games,
            cvar_alpha=args.cvar_alpha,
            output_dir=args.output_dir,
        )
        if args.confirmation != "none"
        else None
    )
    confirmation_complete = bool(
        confirmation_summary
        and confirmation_summary["coverage_complete"]
    )
    final_summary = (
        confirmation_summary
        if confirmation_complete
        else screening_summary
    )
    final_phase = (
        "confirmation" if confirmation_complete else "screening"
    )
    promotion = promotion_gate(
        final_summary,
        args,
        confirmation_required=args.confirmation != "none",
        confirmation_complete=confirmation_complete,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_at_utc": utc_now(),
        "run_signature": signature,
        "candidate": {
            "checkpoint": str(args.candidate),
            "checkpoint_sha256": candidate_sha,
            "deck": str(args.candidate_deck),
            "deck_file_sha256": candidate_deck_file_sha,
            "order_mode": selected_candidate_order_mode,
        },
        "league_manifest": str(args.league_manifest),
        "deployment_contract": signature_payload["deployment_contract"],
        "selected_policy_ids": [
            opponent.policy_id for opponent in opponents
        ],
        "screening": {
            **screening_summary,
            "gate": screening_gate,
            "failures": screening_failures,
            "wall_budget_exhausted": screen_wall_exhausted,
        },
        "confirmation": {
            "mode": args.confirmation,
            "reason": confirmation_reason,
            "summary": confirmation_summary,
            "failures": confirmation_failures,
            "wall_budget_exhausted": confirmation_wall_exhausted,
            "confirmation_is_independent_not_cumulative": True,
            "confirmation_replaces_screening_for_promotion": True,
        },
        "final_evidence_phase": final_phase,
        "promotion": promotion,
        "engine": {
            "engine_seed_control": False,
            "warning": ENGINE_WARNING,
            "strict_even_valid_games_by_candidate_seat_required": True,
        },
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "local_only": True,
        "uploads_or_submissions_performed": False,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
