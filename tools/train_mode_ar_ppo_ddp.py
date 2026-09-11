#!/usr/bin/env python3
"""Synchronous multi-GPU adapter for the mode-aware AR PPO trainer.

Each torchrun rank collects an independent official-engine rollout with the
same 70/20/10 recent-meta, inverse-window and current-policy self-play
contract. PPO actor/value gradients are averaged by NCCL DDP, while rank 0 is
the only checkpoint writer and evaluator. The opponent outcome window and
gate result are synchronized before the next update.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import train_mode_ar_ppo as mode_ar
from parallel_rollout import collect_rollout_parallel, install_parallel_rollout


def _extract_parallel_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout-workers", type=int, default=4)
    parser.add_argument("--rollout-envs-per-worker", type=int, default=32)
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=10.0)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    if args.rollout_workers < 1 or args.rollout_envs_per_worker < 1:
        raise ValueError("Parallel rollout worker counts must be positive")
    if args.rollout_batch_wait_ms < 0.0:
        raise ValueError("Rollout batch wait must be non-negative")
    return args


def _cli_value(flag: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _replace_cli_value(flag: str, value: str) -> None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        sys.argv.extend((flag, value))
    else:
        if index + 1 >= len(sys.argv):
            raise ValueError(f"{flag} requires a value")
        sys.argv[index + 1] = value


class DistributedRuntime:
    def __init__(self) -> None:
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError("Start this trainer with torchrun")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        if self.world_size < 2:
            raise RuntimeError("Mode-AR DDP requires at least two ranks")
        if not torch.cuda.is_available():
            raise RuntimeError("Mode-AR DDP requires CUDA")
        if self.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={self.local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        self.device = torch.device("cuda", self.local_rank)
        output = _cli_value("--output-dir")
        if output is None:
            raise ValueError("--output-dir is required")
        self.authoritative_output = Path(output).resolve()

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def broadcast_object(self, value: Any, source: int = 0) -> Any:
        payload = [value if self.rank == source else None]
        dist.broadcast_object_list(payload, src=source, device=self.device)
        return payload[0]

    def all_gather_object(self, value: Any) -> list[Any]:
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def install_rank_arguments(self) -> None:
        _replace_cli_value("--device", f"cuda:{self.local_rank}")
        base_seed = int(_cli_value("--seed") or "0")
        _replace_cli_value("--seed", str(base_seed + self.rank * 1_000_003))
        if not self.is_primary:
            # Keep worker-local bookkeeping outside the authoritative run
            # directory.  Putting it underneath ``authoritative_output``
            # lets a non-zero rank create that parent before rank 0 enters
            # ``mode_ar.train``; rank 0 then correctly-but-unhelpfully rejects
            # the just-created directory as an attempted reuse.  A sibling
            # scratch tree removes that startup race while retaining the base
            # trainer's deliberate no-overwrite guard on every rank.
            worker_output = (
                self.authoritative_output.parent
                / f".{self.authoritative_output.name}.ddp_worker_state"
                / f"rank-{self.rank:02d}"
            )
            _replace_cli_value("--output-dir", str(worker_output))


def _sum_nested_counts(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        for name, counts in row.items():
            target = result.setdefault(str(name), {})
            if not isinstance(counts, Mapping):
                continue
            for key, value in counts.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    target[str(key)] = target.get(str(key), 0) + int(value)
    return result


def merge_rollout_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged = copy.deepcopy(rows[0])
    seconds = max(float(row.get("seconds", 0.0)) for row in rows)
    integer_keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    for key in integer_keys:
        merged[key] = sum(int(row.get(key, 0)) for row in rows)
    decisions = sum(int(row.get("engine_decisions", 0)) for row in rows)
    games = sum(int(row.get("valid_games", 0)) for row in rows)
    transitions = sum(int(row.get("transitions_kept", 0)) for row in rows)
    merged["seconds"] = seconds
    merged["decisions_per_second"] = decisions / max(seconds, 1e-6)
    merged["mean_episode_decisions"] = decisions / max(games, 1)
    for key in ("mean_old_value", "mean_terminal_return"):
        merged[key] = sum(
            float(row.get(key, 0.0))
            * int(row.get("transitions_kept", 0))
            for row in rows
        ) / max(transitions, 1)
    merged["league_by_opponent"] = _sum_nested_counts(
        [dict(row.get("league_by_opponent") or {}) for row in rows]
    )
    merged["distributed"] = {
        "world_size": len(rows),
        "global_games": games,
        "global_engine_decisions": decisions,
        "global_transitions": transitions,
        "wall_seconds": seconds,
        "aggregate_decisions_per_second": merged["decisions_per_second"],
        "per_rank": [
            {
                "rank": rank,
                "games": int(row.get("valid_games", 0)),
                "engine_decisions": int(row.get("engine_decisions", 0)),
                "transitions": int(row.get("transitions_kept", 0)),
                "seconds": float(row.get("seconds", 0.0)),
                "decisions_per_second": float(
                    row.get("decisions_per_second", 0.0)
                ),
            }
            for rank, row in enumerate(rows)
        ],
    }
    return merged


class DistributedActionStatistics(nn.Module):
    """Put the complete pointer decode path inside the DDP forward graph."""

    def __init__(
        self,
        policy: mode_ar.bc.ModeAwareARPolicy,
        reference: mode_ar.bc.ModeAwareARPolicy,
        base_statistics: Any,
    ) -> None:
        super().__init__()
        self.policy = policy
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_base_statistics", base_statistics)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        action_sequences: torch.Tensor,
        action_counts: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._base_statistics(
            self.policy,
            batch,
            action_sequences,
            action_counts,
            temperature=float(temperature),
            reference_model=self._reference,
        )


def install_distributed_adapter(runtime: DistributedRuntime) -> None:
    base_collect = mode_ar.legacy.collect_rollout
    base_update = mode_ar.ppo_update
    base_evaluate = mode_ar.evaluate_models
    base_atomic_save = mode_ar.atomic_torch_save
    base_log = mode_ar.log
    base_statistics = mode_ar.action_path_statistics
    base_sync_kl = mode_ar.synchronize_ppo_epoch_kl

    def distributed_log(message: str) -> None:
        if runtime.is_primary:
            base_log(message)

    def distributed_collect(*args: Any, **kwargs: Any) -> Any:
        reweighter = kwargs.get("opponent_sampling_reweighter")
        if reweighter is None and len(args) >= 9:
            reweighter = args[8]
        state_before = (
            copy.deepcopy(reweighter.state_dict())
            if reweighter is not None
            else None
        )
        transitions, local_metrics = base_collect(*args, **kwargs)
        local_sequences = dict(
            local_metrics.get("opponent_outcome_sequences") or {}
        )
        gathered_sequences = runtime.all_gather_object(local_sequences)
        combined_sequences: dict[str, list[str]] = {}
        for rank_sequences in gathered_sequences:
            for name, values in rank_sequences.items():
                combined_sequences.setdefault(str(name), []).extend(values)
        if reweighter is not None:
            if state_before is None:
                raise RuntimeError("Distributed opponent window state is missing")
            reweighter.load_state_dict(state_before)
            reweighter.observe(combined_sequences)
            synchronized_state = runtime.broadcast_object(
                reweighter.state_dict() if runtime.is_primary else None
            )
            reweighter.load_state_dict(synchronized_state)
        gathered_metrics = runtime.all_gather_object(local_metrics)
        merged = (
            merge_rollout_metrics(gathered_metrics)
            if runtime.is_primary
            else None
        )
        merged = runtime.broadcast_object(merged)
        merged["opponent_outcome_sequences"] = combined_sequences
        if reweighter is not None:
            merged["opponent_sampling_reweight_after"] = reweighter.audit()
        return transitions, merged

    def distributed_update(
        model: mode_ar.bc.ModeAwareARPolicy,
        reference_model: mode_ar.bc.ModeAwareARPolicy,
        optimizer: torch.optim.Optimizer,
        transitions: list[dict[str, Any]],
        model_cfg: dict[str, Any],
        config: mode_ar.TrainConfig,
        device: torch.device,
        update: int,
    ) -> dict[str, Any]:
        local_count = len(transitions)
        if local_count < 1:
            raise RuntimeError("A DDP rank produced no PPO transitions")
        maximum = torch.tensor(local_count, device=runtime.device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        padded_count = int(
            math.ceil(int(maximum.item()) / config.minibatch_size)
            * config.minibatch_size
        )
        padded = list(transitions)
        if len(padded) < padded_count:
            rng = random.Random(config.seed + update * 104_729)
            padded.extend(
                transitions[rng.randrange(local_count)]
                for _ in range(padded_count - local_count)
            )

        statistics_module = DistributedActionStatistics(
            model,
            reference_model,
            base_statistics,
        )
        ddp_statistics = DistributedDataParallel(
            statistics_module,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        def statistics_dispatch(
            candidate: mode_ar.bc.ModeAwareARPolicy,
            batch: dict[str, torch.Tensor],
            action_sequences: torch.Tensor,
            action_counts: torch.Tensor,
            *,
            temperature: float,
            reference_model: mode_ar.bc.ModeAwareARPolicy | None = None,
        ) -> Any:
            if candidate is model:
                if reference_model is None:
                    raise RuntimeError("DDP PPO requires the frozen reference")
                return ddp_statistics(
                    batch,
                    action_sequences,
                    action_counts,
                    float(temperature),
                )
            return base_statistics(
                candidate,
                batch,
                action_sequences,
                action_counts,
                temperature=temperature,
                reference_model=reference_model,
            )

        def synchronize_epoch_kl(value: float, sync_device: torch.device) -> float:
            del sync_device
            tensor = torch.tensor(float(value), device=runtime.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return float(tensor.item()) / runtime.world_size

        mode_ar.action_path_statistics = statistics_dispatch
        mode_ar.synchronize_ppo_epoch_kl = synchronize_epoch_kl
        try:
            local_result = base_update(
                model,
                reference_model,
                optimizer,
                padded,
                model_cfg,
                config,
                device,
                update,
            )
        finally:
            mode_ar.action_path_statistics = base_statistics
            mode_ar.synchronize_ppo_epoch_kl = base_sync_kl
            del ddp_statistics
            del statistics_module

        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

        local_counts = runtime.all_gather_object(local_count)
        gathered = runtime.all_gather_object(local_result)
        merged = copy.deepcopy(gathered[0]) if runtime.is_primary else None
        if runtime.is_primary:
            numeric_keys = {
                key
                for key in gathered[0]
                if all(
                    isinstance(row.get(key), (int, float))
                    and not isinstance(row.get(key), bool)
                    for row in gathered
                )
            }
            for key in numeric_keys:
                merged[key] = sum(float(row[key]) for row in gathered) / len(
                    gathered
                )
            merged["transitions"] = sum(local_counts)
            merged["rows"] = sum(int(row.get("rows", 0)) for row in gathered)
            merged["optimizer_steps"] = int(
                gathered[0].get("optimizer_steps", 0)
            )
            merged["epochs_completed"] = int(
                gathered[0].get("epochs_completed", 0)
            )
            merged["early_stop"] = bool(gathered[0].get("early_stop", False))
            merged["seconds"] = max(
                float(row.get("seconds", 0.0)) for row in gathered
            )
            merged["distributed"] = {
                "world_size": runtime.world_size,
                "local_transition_counts": local_counts,
                "padded_transitions_per_rank": padded_count,
                "effective_global_minibatch_size": (
                    config.minibatch_size * runtime.world_size
                ),
                "gradient_synchronization": "NCCL DDP mean",
                "early_stop_metric": "world_mean_approx_kl",
            }
        return runtime.broadcast_object(merged)

    def distributed_evaluate(*args: Any, **kwargs: Any) -> Any:
        result = base_evaluate(*args, **kwargs) if runtime.is_primary else None
        return runtime.broadcast_object(result)

    def primary_atomic_save(payload: Mapping[str, Any], path: Path) -> None:
        if runtime.is_primary:
            base_atomic_save(payload, path)

    mode_ar.log = distributed_log
    mode_ar.legacy.collect_rollout = distributed_collect
    mode_ar.ppo_update = distributed_update
    mode_ar.evaluate_models = distributed_evaluate
    mode_ar.atomic_torch_save = primary_atomic_save


def main() -> None:
    parallel = _extract_parallel_args()
    runtime = DistributedRuntime()
    runtime.install_rank_arguments()
    install_parallel_rollout(
        workers=parallel.rollout_workers,
        environments_per_worker=parallel.rollout_envs_per_worker,
        batch_wait_ms=parallel.rollout_batch_wait_ms,
        feature_adapter="legacy",
    )
    mode_ar.legacy.collect_rollout = collect_rollout_parallel
    install_distributed_adapter(runtime)
    try:
        mode_ar.main()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
