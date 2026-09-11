#!/usr/bin/env python3
"""Synchronous multi-GPU adapter for the non-AR league PPO trainer.

Every torchrun rank collects an independent official-engine rollout on its own
GPU. PPO gradients are synchronized with DistributedDataParallel, so the eight
ranks update one policy rather than producing eight unrelated policies. Rank 0
is the sole checkpoint/league writer and the sole evaluator; evaluation results
and rolling opponent-window state are broadcast to every worker.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import train_nonar_league_ppo as nonar_entry
import train_ppo as legacy
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
            raise RuntimeError(
                "Distributed trainer must be started with torchrun; use "
                "run_8gpu_ppo.py from the portable package"
            )
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        if self.world_size < 2:
            raise RuntimeError("Distributed PPO requires at least two ranks")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed PPO requires CUDA")
        if self.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={self.local_rank} but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        self.device = torch.device("cuda", self.local_rank)
        self.authoritative_output = Path(
            _cli_value("--output-dir") or "runs/nonar_ppo_8gpu"
        ).resolve()

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

    def install_rank_paths(self) -> None:
        _replace_cli_value("--device", f"cuda:{self.local_rank}")
        if not self.is_primary:
            worker_output = (
                self.authoritative_output
                / ".ddp_worker_state"
                / f"rank-{self.rank:02d}"
            )
            _replace_cli_value("--output-dir", str(worker_output))


def _sum_nested_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        for name, counts in row.items():
            target = result.setdefault(name, {})
            for key, value in counts.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    target[key] = target.get(key, 0) + value
    return result


def _merge_rollout_metrics(
    rows: list[dict[str, Any]],
    combined_sequences: dict[str, list[str]] | None,
    reweight_before: dict[str, Any] | None,
    reweight_after: dict[str, Any] | None,
) -> dict[str, Any]:
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
    merged["seconds"] = seconds
    merged["decisions_per_second"] = sum(
        int(row.get("engine_decisions", 0)) for row in rows
    ) / max(seconds, 1e-6)
    total_games = sum(int(row.get("valid_games", 0)) for row in rows)
    total_transitions = sum(
        int(row.get("transitions_kept", 0)) for row in rows
    )
    merged["mean_episode_decisions"] = sum(
        int(row.get("engine_decisions", 0)) for row in rows
    ) / max(total_games, 1)
    for key in ("mean_old_value", "mean_terminal_return"):
        merged[key] = sum(
            float(row.get(key, 0.0))
            * int(row.get("transitions_kept", 0))
            for row in rows
        ) / max(total_transitions, 1)
    merged["league_by_opponent"] = _sum_nested_counts(
        [dict(row.get("league_by_opponent") or {}) for row in rows]
    )
    if combined_sequences is not None:
        merged["opponent_outcome_sequences"] = combined_sequences
    if reweight_before is not None:
        merged["opponent_sampling_reweight_before"] = reweight_before
    if reweight_after is not None:
        merged["opponent_sampling_reweight_after"] = reweight_after
    merged["distributed"] = {
        "world_size": len(rows),
        "global_games": total_games,
        "global_engine_decisions": int(merged.get("engine_decisions", 0)),
        "global_transitions": total_transitions,
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


def install_distributed_adapter(runtime: DistributedRuntime) -> None:
    base_seed_everything = legacy.seed_everything
    base_collect_rollout = legacy.collect_rollout
    base_ppo_update = legacy.ppo_update
    base_evaluate_head_to_head = legacy.evaluate_head_to_head
    base_row_weighted_mean = legacy.row_weighted_minibatch_mean
    base_save_checkpoint = legacy.save_ppo_checkpoint
    base_torch_save = torch.save
    base_log = legacy.log

    def distributed_log(message: str) -> None:
        if runtime.is_primary:
            base_log(message)

    def distributed_seed(seed: int) -> None:
        # All model weights come from a checkpoint; rank offsets only diversify
        # opponent selection and stochastic policy actions during rollout.
        base_seed_everything(seed + runtime.rank * 1_000_003)

    def distributed_collect_rollout(*args: Any, **kwargs: Any) -> Any:
        reweighter = kwargs.get("opponent_sampling_reweighter")
        if reweighter is None and len(args) >= 9:
            reweighter = args[8]
        state_before = (
            copy.deepcopy(reweighter.state_dict())
            if reweighter is not None
            else None
        )
        transitions, local_metrics = base_collect_rollout(*args, **kwargs)
        local_sequences = dict(
            local_metrics.get("opponent_outcome_sequences") or {}
        )
        gathered_sequences = runtime.all_gather_object(local_sequences)
        combined_sequences: dict[str, list[str]] | None = None
        before_audit = None
        after_audit = None
        if reweighter is not None:
            if state_before is None:
                raise RuntimeError("Distributed reweighter state was not saved")
            combined_sequences = {}
            for rank_sequences in gathered_sequences:
                for name, values in rank_sequences.items():
                    combined_sequences.setdefault(name, []).extend(values)
            reweighter.load_state_dict(state_before)
            before_audit = reweighter.audit()
            reweighter.observe(combined_sequences)
            synchronized_state = runtime.broadcast_object(
                reweighter.state_dict() if runtime.is_primary else None
            )
            reweighter.load_state_dict(synchronized_state)
            after_audit = reweighter.audit()

        gathered_metrics = runtime.all_gather_object(local_metrics)
        merged = (
            _merge_rollout_metrics(
                gathered_metrics,
                combined_sequences,
                before_audit,
                after_audit,
            )
            if runtime.is_primary
            else None
        )
        merged = runtime.broadcast_object(merged)
        return transitions, merged

    def distributed_row_weighted_mean(values: Any) -> float:
        local_value = float(base_row_weighted_mean(values))
        tensor = torch.tensor(local_value, device=runtime.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def distributed_ppo_update(
        model: torch.nn.Module,
        reference_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        transitions: list[dict[str, Any]],
        model_config: dict[str, Any],
        config: Any,
        device: torch.device,
        update: int,
        objective_state: Any = None,
    ) -> dict[str, Any]:
        if config.ppo_objective != "standard":
            raise ValueError("Portable DDP adapter currently supports standard PPO")
        local_count = len(transitions)
        maximum = torch.tensor(local_count, device=runtime.device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        padded_count = int(
            math.ceil(int(maximum.item()) / config.minibatch_size)
            * config.minibatch_size
        )
        padded = list(transitions)
        if len(padded) < padded_count:
            rng = random.Random(config.seed + update * 104_729 + runtime.rank)
            padded.extend(
                transitions[rng.randrange(local_count)]
                for _ in range(padded_count - local_count)
            )

        ddp_model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        local_result = base_ppo_update(
            ddp_model,
            reference_model,
            optimizer,
            padded,
            model_config,
            config,
            device,
            update,
            objective_state,
        )
        del ddp_model
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)

        local_transition_counts = runtime.all_gather_object(local_count)
        gathered = runtime.all_gather_object(local_result)
        merged = copy.deepcopy(gathered[0]) if runtime.is_primary else None
        if runtime.is_primary:
            numeric_mean_keys = {
                key
                for key in gathered[0]
                if all(
                    isinstance(row.get(key), (int, float))
                    and not isinstance(row.get(key), bool)
                    for row in gathered
                )
            }
            for key in numeric_mean_keys:
                merged[key] = sum(float(row[key]) for row in gathered) / len(
                    gathered
                )
            merged["rows"] = sum(int(row.get("rows", 0)) for row in gathered)
            merged["transitions"] = sum(local_transition_counts)
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
                "local_transition_counts": local_transition_counts,
                "padded_transitions_per_rank": padded_count,
                "effective_global_minibatch_size": (
                    config.minibatch_size * runtime.world_size
                ),
                "gradient_synchronization": "DDP mean before gradient clipping",
            }
        merged = runtime.broadcast_object(merged)
        return merged

    def distributed_evaluate_head_to_head(*args: Any, **kwargs: Any) -> Any:
        result = (
            base_evaluate_head_to_head(*args, **kwargs)
            if runtime.is_primary
            else None
        )
        return runtime.broadcast_object(result)

    def primary_checkpoint_only(*args: Any, **kwargs: Any) -> None:
        if runtime.is_primary:
            base_save_checkpoint(*args, **kwargs)

    def primary_torch_save(*args: Any, **kwargs: Any) -> Any:
        if runtime.is_primary:
            return base_torch_save(*args, **kwargs)
        return None

    legacy.log = distributed_log
    legacy.seed_everything = distributed_seed
    legacy.collect_rollout = distributed_collect_rollout
    legacy.ppo_update = distributed_ppo_update
    legacy.evaluate_head_to_head = distributed_evaluate_head_to_head
    legacy.row_weighted_minibatch_mean = distributed_row_weighted_mean
    legacy.save_ppo_checkpoint = primary_checkpoint_only
    torch.save = primary_torch_save


def main() -> None:
    parallel = _extract_parallel_args()
    runtime = DistributedRuntime()
    runtime.install_rank_paths()
    nonar_entry.install_nonar_adapter()
    install_parallel_rollout(
        workers=parallel.rollout_workers,
        environments_per_worker=parallel.rollout_envs_per_worker,
        batch_wait_ms=parallel.rollout_batch_wait_ms,
    )
    legacy.collect_rollout = collect_rollout_parallel
    install_distributed_adapter(runtime)
    try:
        legacy.main()
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
