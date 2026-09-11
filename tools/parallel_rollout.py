#!/usr/bin/env python3
"""Multi-process official-engine rollout with one central CUDA policy server.

Workers own RawBattle pointers, JSON parsing, public-log history and feature
construction.  The parent process batches requests from every worker, performs
one GPU forward per shared policy, records learner transitions, and applies the
unchanged terminal reward/GAE contract.
"""

from __future__ import annotations

import atexit
import copy
import math
import multiprocessing as mp
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from typing import Any

import torch

import train_ppo as legacy


@dataclass
class WorkerBattle:
    battle: Any
    uid: int
    opponent_index: int
    opponent_name: str | None
    learner_seat: int | None
    trainable_seats: tuple[int, ...]
    seat_deck_hash: dict[int, str]
    decisions: int = 0
    public_logs: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: {0: [], 1: []}
    )


def _remote_metadata(game: WorkerBattle) -> dict[str, Any]:
    return {
        "uid": game.uid,
        "opponent_index": game.opponent_index,
        "opponent_name": game.opponent_name,
        "learner_seat": game.learner_seat,
        "trainable_seats": list(game.trainable_seats),
        "seat_deck_hash": game.seat_deck_hash,
    }


def _worker_rollout(connection: Connection, spec: dict[str, Any]) -> None:
    feature_adapter = str(spec.get("feature_adapter", "nonar"))
    if feature_adapter == "nonar":
        import train_nonar_league_ppo as nonar_entry

        nonar_entry.install_nonar_adapter()
    elif feature_adapter != "legacy":
        raise ValueError(f"Unknown rollout feature adapter: {feature_adapter}")
    torch.set_num_threads(1)
    random_generator = random.Random(int(spec["seed"]))
    worker_id = int(spec["worker_id"])
    target_games = int(spec["target_games"])
    active_limit = min(int(spec["active_environments"]), target_games)
    learner_deck = list(spec["learner_deck"])
    learner_deck_hash = str(spec["learner_deck_hash"])
    opponents = list(spec["opponents"])
    weights = [float(value) for value in spec["opponent_weights"]]
    league_probability = float(spec.get("league_probability", 1.0))
    model_config = dict(spec["model_config"])
    max_game_decisions = int(spec["max_game_decisions"])
    update = int(spec["update"])
    uid_counter = update * 100_000_000 + worker_id * 1_000_000
    start_ordinal = 0
    valid_completed = 0
    attempts = 0
    start_errors = 0
    active: dict[int, WorkerBattle] = {}
    pending_events: list[dict[str, Any]] = []
    pending_start_errors: list[dict[str, Any]] = []

    def start_one() -> bool:
        nonlocal uid_counter, start_ordinal, start_errors
        frozen_game = bool(opponents) and (
            random_generator.random() < league_probability
        )
        learner_seat = (
            (worker_id + start_ordinal) % 2 if frozen_game else None
        )
        start_ordinal += 1
        if frozen_game:
            assert learner_seat is not None
            opponent_index = random_generator.choices(
                range(len(opponents)),
                weights=weights,
                k=1,
            )[0]
            opponent = opponents[opponent_index]
            opponent_name: str | None = str(opponent["name"])
            trainable_seats = (learner_seat,)
            seat_decks = {
                learner_seat: learner_deck,
                1 - learner_seat: list(opponent["deck"]),
            }
            seat_hashes = {
                learner_seat: learner_deck_hash,
                1 - learner_seat: str(opponent["deck_hash"]),
            }
        else:
            opponent_index = -1
            opponent_name = None
            trainable_seats = (0, 1)
            seat_decks = {0: learner_deck, 1: learner_deck}
            seat_hashes = {0: learner_deck_hash, 1: learner_deck_hash}
        try:
            battle = legacy.RawBattle(seat_decks[0], seat_decks[1])
        except ValueError:
            start_errors += 1
            pending_start_errors.append(
                {
                    "opponent_name": (
                        legacy.SELFPLAY_OPPONENT_GROUP
                        if opponent_name is None
                        else opponent_name
                    ),
                    "learner_seat": (
                        "selfplay_both"
                        if learner_seat is None
                        else str(learner_seat)
                    ),
                }
            )
            return False
        game = WorkerBattle(
            battle=battle,
            uid=uid_counter,
            opponent_index=opponent_index,
            opponent_name=opponent_name,
            learner_seat=learner_seat,
            trainable_seats=trainable_seats,
            seat_deck_hash=seat_hashes,
        )
        uid_counter += 1
        active[game.uid] = game
        return True

    def refill() -> None:
        failures = 0
        while (
            len(active) < active_limit
            and valid_completed + len(active) < target_games
        ):
            if start_one():
                failures = 0
            else:
                failures += 1
                if failures > target_games * 5:
                    raise RuntimeError("Repeated BattleStart failures in worker")

    try:
        refill()
        while active:
            requests: list[dict[str, Any]] = []
            invalid_uids: list[int] = []
            for game in list(active.values()):
                observation = game.battle.observation
                select = observation.get("select")
                current = observation.get("current") or {}
                options = select.get("option") if isinstance(select, dict) else None
                minimum = int((select or {}).get("minCount", 0) or 0)
                maximum = int((select or {}).get("maxCount", 0) or 0)
                if (
                    not isinstance(select, dict)
                    or not isinstance(options, list)
                    or not options
                    or minimum < 0
                    or maximum < minimum
                    or maximum > len(options)
                    or maximum > legacy.MAX_ACTION_COUNT
                ):
                    pending_events.append(
                        {
                            **_remote_metadata(game),
                            "valid": False,
                            "reason": "invalid_observation",
                        }
                    )
                    invalid_uids.append(game.uid)
                    continue
                seat = int(current.get("yourIndex", 0) or 0)
                current_logs = observation.get("logs") or []
                if isinstance(current_logs, list):
                    game.public_logs[seat].extend(
                        event for event in current_logs if isinstance(event, dict)
                    )
                    game.public_logs[seat] = game.public_logs[seat][-64:]
                feature = legacy.live_feature(
                    observation,
                    model_config,
                    game.seat_deck_hash[seat],
                    game.public_logs[seat],
                )
                if feature is None:
                    pending_events.append(
                        {
                            **_remote_metadata(game),
                            "valid": False,
                            "reason": "invalid_observation",
                        }
                    )
                    invalid_uids.append(game.uid)
                    continue
                requests.append(
                    {
                        **_remote_metadata(game),
                        "seat": seat,
                        "policy_index": (
                            -1
                            if seat in game.trainable_seats
                            else game.opponent_index
                        ),
                        "feature": feature,
                    }
                )

            for uid in invalid_uids:
                game = active.pop(uid)
                game.battle.close()
                attempts += 1
            refill()
            if not requests:
                continue

            connection.send(
                {
                    "type": "infer",
                    "requests": requests,
                    "events": pending_events,
                    "start_errors": pending_start_errors,
                }
            )
            pending_events = []
            response = connection.recv()
            if response.get("type") == "stop":
                break
            actions = list(response.get("actions") or [])
            if len(actions) != len(requests):
                raise RuntimeError("Inference response length mismatch")

            finished_uids: list[int] = []
            for request, action in zip(requests, actions):
                uid = int(request["uid"])
                game = active.get(uid)
                if game is None:
                    raise RuntimeError(f"Worker lost active game {uid}")
                _, error = game.battle.step(list(action))
                game.decisions += 1
                if error:
                    pending_events.append(
                        {
                            **_remote_metadata(game),
                            "valid": False,
                            "reason": "battle_step_error",
                            "select_error": int(error),
                        }
                    )
                    finished_uids.append(uid)
                    continue
                result = game.battle.result
                if result != -1:
                    valid_completed += 1
                    pending_events.append(
                        {
                            **_remote_metadata(game),
                            "valid": True,
                            "result": int(result),
                            "decisions": game.decisions,
                        }
                    )
                    finished_uids.append(uid)
                    continue
                if game.decisions >= max_game_decisions:
                    pending_events.append(
                        {
                            **_remote_metadata(game),
                            "valid": False,
                            "reason": "max_game_decisions",
                            "decisions": game.decisions,
                        }
                    )
                    finished_uids.append(uid)

            for uid in finished_uids:
                game = active.pop(uid)
                game.battle.close()
                attempts += 1
            refill()

        connection.send(
            {
                "type": "done",
                "events": pending_events,
                "start_errors": pending_start_errors,
                "valid_games": valid_completed,
                "attempts": attempts,
                "start_error_count": start_errors,
            }
        )
    except BaseException as error:
        for game in active.values():
            game.battle.close()
        connection.send(
            {
                "type": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )


def _worker_process(connection: Connection, worker_id: int) -> None:
    while True:
        message = connection.recv()
        message_type = message.get("type")
        if message_type == "stop":
            return
        if message_type != "start":
            raise RuntimeError(f"Unexpected worker command: {message_type!r}")
        spec = dict(message["spec"])
        spec["worker_id"] = worker_id
        _worker_rollout(connection, spec)


class ParallelRolloutPool:
    def __init__(
        self,
        workers: int,
        environments_per_worker: int,
        batch_wait_ms: float,
        feature_adapter: str,
    ) -> None:
        if workers < 1 or environments_per_worker < 1 or batch_wait_ms < 0.0:
            raise ValueError("Invalid parallel rollout configuration")
        self.workers = workers
        self.environments_per_worker = environments_per_worker
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        if feature_adapter not in {"legacy", "nonar"}:
            raise ValueError("feature_adapter must be legacy or nonar")
        self.feature_adapter = feature_adapter
        context = mp.get_context("spawn")
        self.connections: list[Connection] = []
        self.processes: list[mp.Process] = []
        for worker_id in range(workers):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_worker_process,
                args=(child, worker_id),
                name=f"ptcg-rollout-{worker_id:02d}",
                daemon=True,
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send({"type": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for connection in self.connections:
            connection.close()

    def _collect_ready(self, active: set[Connection]) -> list[Connection]:
        ready = list(wait(active))
        if self.batch_wait_seconds <= 0.0 or len(ready) == len(active):
            return ready
        deadline = time.monotonic() + self.batch_wait_seconds
        ready_set = set(ready)
        while len(ready_set) < len(active):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            more = wait(active - ready_set, timeout=remaining)
            if not more:
                break
            ready_set.update(more)
        return list(ready_set)


_POOL: ParallelRolloutPool | None = None


def install_parallel_rollout(
    workers: int,
    environments_per_worker: int,
    batch_wait_ms: float,
    feature_adapter: str = "nonar",
) -> None:
    global _POOL
    if _POOL is not None:
        raise RuntimeError("Parallel rollout pool is already installed")
    _POOL = ParallelRolloutPool(
        workers,
        environments_per_worker,
        batch_wait_ms,
        feature_adapter,
    )
    atexit.register(_POOL.close)


def _game_record(payload: dict[str, Any]) -> legacy.RunningGame:
    trainable_seats = {
        int(seat) for seat in payload.get("trainable_seats") or []
    }
    if trainable_seats == {0, 1}:
        seat_policy = {0: -1, 1: -1}
        opponent_name = None
    else:
        learner_seat = int(payload["learner_seat"])
        if trainable_seats != {learner_seat}:
            raise ValueError("Remote frozen game has invalid trainable seats")
        opponent_index = int(payload["opponent_index"])
        seat_policy = {learner_seat: -1, 1 - learner_seat: opponent_index}
        opponent_name = str(payload["opponent_name"])
    return legacy.RunningGame(
        battle=None,
        uid=int(payload["uid"]),
        seat_policy=seat_policy,
        seat_deck_hash={
            int(key): str(value)
            for key, value in dict(payload["seat_deck_hash"]).items()
        },
        trainable_seats=trainable_seats,
        transition_indices={0: [], 1: []},
        opponent_name=opponent_name,
    )


def collect_rollout_parallel(
    current_model: torch.nn.Module,
    opponents: list[legacy.FrozenOpponent],
    deck: list[int],
    model_config: dict[str, Any],
    config: legacy.PPOConfig,
    device: torch.device,
    update: int,
    opponent_quota_controller: Any = None,
    opponent_sampling_reweighter: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _POOL is None:
        raise RuntimeError("Parallel rollout pool is not installed")
    if opponent_quota_controller is not None:
        raise ValueError("Parallel rollout currently requires legacy sampling")
    if (
        not 0.0 <= config.league_probability <= 1.0
        or config.opponent_sampling != "per_game"
        or not opponents
        or config.failed_attempt_as_loss
        or config.truncation_as_loss
    ):
        raise ValueError(
            "Parallel rollout requires per-game league sampling with at least "
            "one frozen opponent, terminal valid-game rewards, and replacement "
            "of failed games"
        )
    current_model.eval()
    for opponent in opponents:
        opponent.model.eval()
    opponent_weights = (
        opponent_sampling_reweighter.sampling_weights(opponents)
        if opponent_sampling_reweighter is not None
        else legacy.resolve_opponent_sampling_weights(
            opponents,
            config.opponent_weights,
            config.history_opponent_weight,
        )
    )
    if opponent_weights is None:
        opponent_weights = [1.0] * len(opponents)
    if len(opponent_weights) != len(opponents) or sum(opponent_weights) <= 0.0:
        raise ValueError("Invalid parallel opponent sampling weights")

    target_base, target_remainder = divmod(config.games_per_update, _POOL.workers)
    targets = [
        target_base + (1 if worker < target_remainder else 0)
        for worker in range(_POOL.workers)
    ]
    if any(target <= 0 for target in targets):
        raise ValueError("More rollout workers than games per update")
    opponent_specs = [
        {
            "name": opponent.name,
            "deck": list(opponent.deck),
            "deck_hash": opponent.deck_hash,
        }
        for opponent in opponents
    ]
    learner_hash = legacy.compute_deck_hash(deck)
    for worker_id, (connection, target) in enumerate(
        zip(_POOL.connections, targets)
    ):
        connection.send(
            {
                "type": "start",
                "spec": {
                    "seed": config.seed + update * 65_537 + worker_id * 1_000_003,
                    "target_games": target,
                    "active_environments": min(
                        _POOL.environments_per_worker,
                        target,
                    ),
                    "learner_deck": list(deck),
                    "learner_deck_hash": learner_hash,
                    "opponents": opponent_specs,
                    "opponent_weights": opponent_weights,
                    "league_probability": config.league_probability,
                    "feature_adapter": _POOL.feature_adapter,
                    "model_config": model_config,
                    "max_game_decisions": config.max_game_decisions,
                    "update": update,
                },
            }
        )

    transitions: list[dict[str, Any]] = []
    games: dict[int, legacy.RunningGame] = {}
    stats: Counter[str] = Counter()
    league_by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    outcome_sequences: dict[str, list[str]] = defaultdict(list)
    failure_audit = legacy.RolloutFailureAudit()
    active_connections = set(_POOL.connections)
    started = time.time()

    def ensure_game(payload: dict[str, Any]) -> legacy.RunningGame:
        uid = int(payload["uid"])
        game = games.get(uid)
        if game is None:
            game = _game_record(payload)
            games[uid] = game
        return game

    def process_events(message: dict[str, Any]) -> None:
        for start_error in message.get("start_errors") or []:
            stats["start_errors"] += 1
            failure_audit.record_start_error(
                str(start_error["opponent_name"]),
                str(start_error["learner_seat"]),
            )
        for event in message.get("events") or []:
            game = ensure_game(event)
            stats["attempted_games"] += 1
            if not bool(event["valid"]):
                reason = str(event["reason"])
                stats[f"{reason}_games"] += 1
                if "select_error" in event:
                    stats[f"select_error_{int(event['select_error'])}"] += 1
                available = legacy.learner_transition_count(game)
                legacy.finalize_episode(
                    game,
                    transitions,
                    result=2,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    valid=False,
                )
                failure_audit.record_game(game, reason, available, 0)
                games.pop(game.uid, None)
                continue
            result = int(event["result"])
            stats["valid_games"] += 1
            stats[f"result_{result}"] += 1
            if len(game.trainable_seats) == 1:
                learner_seat = next(iter(game.trainable_seats))
                opponent_name = str(game.opponent_name)
                stats["league_games"] += 1
                league_by_opponent[opponent_name]["games"] += 1
                seat_prefix = f"seat_{learner_seat}_"
                league_by_opponent[opponent_name][seat_prefix + "games"] += 1
                if result == learner_seat:
                    stats["league_current_wins"] += 1
                    league_by_opponent[opponent_name]["wins"] += 1
                    league_by_opponent[opponent_name][seat_prefix + "wins"] += 1
                    outcome_sequences[opponent_name].append("win")
                elif result == 2:
                    stats["league_draws"] += 1
                    league_by_opponent[opponent_name]["draws"] += 1
                    league_by_opponent[opponent_name][seat_prefix + "draws"] += 1
                    outcome_sequences[opponent_name].append("draw")
                else:
                    stats["league_current_losses"] += 1
                    league_by_opponent[opponent_name]["losses"] += 1
                    league_by_opponent[opponent_name][seat_prefix + "losses"] += 1
                    outcome_sequences[opponent_name].append("loss")
            else:
                stats["selfplay_games"] += 1
            legacy.finalize_episode(
                game,
                transitions,
                result=result,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                valid=True,
            )
            games.pop(game.uid, None)

    try:
        while active_connections:
            ready = _POOL._collect_ready(active_connections)
            inference_messages: dict[Connection, dict[str, Any]] = {}
            requests: list[tuple[Connection, int, dict[str, Any]]] = []
            for connection in ready:
                message = connection.recv()
                message_type = message.get("type")
                if message_type == "error":
                    raise RuntimeError(
                        f"Rollout worker failed: {message.get('error_type')}: "
                        f"{message.get('error')}"
                    )
                process_events(message)
                if message_type == "done":
                    if int(message.get("valid_games", -1)) <= 0:
                        raise RuntimeError("Rollout worker completed no valid games")
                    active_connections.remove(connection)
                    continue
                if message_type != "infer":
                    raise RuntimeError(f"Unexpected worker message {message_type!r}")
                inference_messages[connection] = message
                for request_index, request in enumerate(message["requests"]):
                    requests.append((connection, request_index, request))

            groups: dict[tuple[str, int], list[int]] = defaultdict(list)
            for flat_index, (_, _, request) in enumerate(requests):
                policy_index = int(request["policy_index"])
                key = (
                    ("learner", -1)
                    if policy_index == -1
                    else ("frozen", id(opponents[policy_index].model))
                )
                groups[key].append(flat_index)
            stats["inference_batches"] += 1
            stats["inference_request_rows"] += len(requests)
            stats["inference_ready_worker_events"] += len(inference_messages)
            stats["inference_max_request_rows"] = max(
                stats["inference_max_request_rows"], len(requests)
            )
            stats["model_forward_calls"] += len(groups)
            if groups:
                stats["model_forward_max_rows"] = max(
                    stats["model_forward_max_rows"],
                    max(len(indices) for indices in groups.values()),
                )
            flat_actions: list[list[int] | None] = [None] * len(requests)
            for indices in groups.values():
                stats["model_forward_rows"] += len(indices)
                first_request = requests[indices[0]][2]
                policy_index = int(first_request["policy_index"])
                features = [requests[index][2]["feature"] for index in indices]
                batch = legacy.collate_features(features, model_config, device)
                model = (
                    current_model
                    if policy_index == -1
                    else opponents[policy_index].model
                )
                with torch.no_grad():
                    outputs = legacy.model_forward(model, batch, device)
                    actions, log_probs, entropies, values = (
                        legacy.sample_ordered_actions(
                            outputs,
                            batch,
                            deterministic=policy_index != -1,
                            canonicalize_order=(
                                policy_index != -1
                                and opponents[policy_index].canonical_order
                            ),
                            temperature=(
                                config.policy_temperature
                                if policy_index == -1
                                else 1.0
                            ),
                        )
                    )
                log_prob_rows = log_probs.detach().cpu().tolist()
                entropy_rows = entropies.detach().cpu().tolist()
                value_rows = values.detach().cpu().tolist()
                for local_index, flat_index in enumerate(indices):
                    request = requests[flat_index][2]
                    action = actions[local_index]
                    flat_actions[flat_index] = action
                    stats["engine_decisions"] += 1
                    if policy_index == -1:
                        game = ensure_game(request)
                        seat = int(request["seat"])
                        transition_index = len(transitions)
                        transitions.append(
                            {
                                "feature": request["feature"],
                                "action": action,
                                "action_count": len(action),
                                "old_log_prob": float(log_prob_rows[local_index]),
                                "old_value": float(value_rows[local_index]),
                                "old_entropy": float(entropy_rows[local_index]),
                                "game_uid": game.uid,
                                "seat": seat,
                                "opponent_name": game.opponent_name,
                                "keep": True,
                            }
                        )
                        game.transition_indices[seat].append(transition_index)

            response_actions: dict[Connection, list[list[int] | None]] = {
                connection: [None] * len(message["requests"])
                for connection, message in inference_messages.items()
            }
            for flat_index, (connection, request_index, _) in enumerate(requests):
                response_actions[connection][request_index] = flat_actions[flat_index]
            for connection, values in response_actions.items():
                if any(value is None for value in values):
                    raise RuntimeError("Missing parallel inference action")
                connection.send({"type": "actions", "actions": values})
    except BaseException:
        _POOL.close()
        raise

    if stats["valid_games"] != config.games_per_update:
        raise RuntimeError(
            f"Parallel rollout completed {stats['valid_games']} valid games; "
            f"expected {config.games_per_update}"
        )
    if games:
        raise RuntimeError(f"Parallel rollout leaked {len(games)} game records")
    kept = [transition for transition in transitions if transition.get("keep")]
    if not kept:
        raise RuntimeError("Parallel rollout produced no learner transitions")
    attempted_games = int(stats["attempted_games"])
    failure_metrics = failure_audit.render(
        started_game_attempts=attempted_games,
        valid_games=int(stats["valid_games"]),
        battle_start_errors=int(stats["start_errors"]),
    )
    stats["transitions_total"] = len(transitions)
    stats["transitions_kept"] = len(kept)
    stats["kept_episode_count"] = legacy.kept_episode_count(kept)
    stats["attempted_games_including_start_errors"] = (
        attempted_games + stats["start_errors"]
    )
    stats["failed_started_games"] = attempted_games - stats["valid_games"]
    stats["hard_failure_attempts"] = failure_metrics["hard_failure_attempts"]
    stats["max_active_games"] = sum(
        min(_POOL.environments_per_worker, target) for target in targets
    )
    stats["seconds"] = time.time() - started
    stats["decisions_per_second"] = stats["engine_decisions"] / max(
        stats["seconds"], 1e-6
    )
    stats["mean_episode_decisions"] = stats["engine_decisions"] / max(
        stats["valid_games"], 1
    )
    stats["mean_decisions_per_started_attempt"] = stats[
        "engine_decisions"
    ] / max(attempted_games, 1)
    stats["mean_old_value"] = sum(float(row["old_value"]) for row in kept) / len(kept)
    stats["mean_terminal_return"] = sum(
        float(row["return"]) for row in kept
    ) / len(kept)
    result = dict(stats)
    result["failure_audit"] = failure_metrics
    result["league_by_opponent"] = {
        name: dict(values) for name, values in sorted(league_by_opponent.items())
    }
    result["league_by_opponent_seat"] = {
        name: {
            seat: {
                key: int(values.get(f"seat_{seat}_{key}", 0))
                for key in ("games", "wins", "losses", "draws")
            }
            for seat in ("0", "1")
        }
        for name, values in sorted(league_by_opponent.items())
    }
    transitions_by_opponent_seat: dict[str, Counter[str]] = defaultdict(Counter)
    for transition in kept:
        opponent_name = transition.get("opponent_name")
        group_name = (
            legacy.SELFPLAY_OPPONENT_GROUP
            if opponent_name is None
            else str(opponent_name)
        )
        transitions_by_opponent_seat[group_name][
            str(int(transition["seat"]))
        ] += 1
    result["transitions_by_opponent_seat"] = {
        name: {seat: int(values.get(seat, 0)) for seat in ("0", "1")}
        for name, values in sorted(transitions_by_opponent_seat.items())
    }
    total_weight = sum(opponent_weights)
    result["opponent_sampling_probabilities"] = {
        opponent.name: weight / total_weight
        for opponent, weight in zip(opponents, opponent_weights)
        if weight > 0.0
    }
    if opponent_sampling_reweighter is not None:
        result["opponent_sampling_reweight_before"] = (
            opponent_sampling_reweighter.audit()
        )
        result["opponent_outcome_sequences"] = {
            name: list(values) for name, values in outcome_sequences.items()
        }
        opponent_sampling_reweighter.observe(result["opponent_outcome_sequences"])
        result["opponent_sampling_reweight_after"] = (
            opponent_sampling_reweighter.audit()
        )
    result["parallel_rollout"] = {
        "workers": _POOL.workers,
        "environments_per_worker": _POOL.environments_per_worker,
        "active_environments": stats["max_active_games"],
        "batch_wait_ms": _POOL.batch_wait_seconds * 1000.0,
        "engine_process_model": "spawned persistent workers",
        "inference_model": "single parent CUDA batch server",
        "inference_batches": int(stats["inference_batches"]),
        "mean_requests_per_batch": float(stats["inference_request_rows"])
        / max(int(stats["inference_batches"]), 1),
        "max_requests_per_batch": int(stats["inference_max_request_rows"]),
        "mean_ready_workers_per_batch": float(
            stats["inference_ready_worker_events"]
        )
        / max(int(stats["inference_batches"]), 1),
        "model_forward_calls": int(stats["model_forward_calls"]),
        "mean_rows_per_model_forward": float(stats["model_forward_rows"])
        / max(int(stats["model_forward_calls"]), 1),
        "max_rows_per_model_forward": int(stats["model_forward_max_rows"]),
    }
    return kept, result
