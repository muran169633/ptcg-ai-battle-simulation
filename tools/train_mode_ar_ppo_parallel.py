#!/usr/bin/env python3
"""Run mode-aware AR PPO with process-parallel official-engine rollouts."""

from __future__ import annotations

import argparse
import sys

import train_mode_ar_ppo as mode_ar
from parallel_rollout import collect_rollout_parallel, install_parallel_rollout


def _extract_parallel_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--rollout-envs-per-worker", type=int, default=16)
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=10.0)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    if args.rollout_workers < 1 or args.rollout_envs_per_worker < 1:
        raise ValueError("Parallel rollout worker counts must be positive")
    if args.rollout_batch_wait_ms < 0.0:
        raise ValueError("Rollout batch wait must be non-negative")
    return args


def main() -> None:
    parallel = _extract_parallel_args()
    install_parallel_rollout(
        workers=parallel.rollout_workers,
        environments_per_worker=parallel.rollout_envs_per_worker,
        batch_wait_ms=parallel.rollout_batch_wait_ms,
        feature_adapter="legacy",
    )
    mode_ar.legacy.collect_rollout = collect_rollout_parallel
    mode_ar.main()


if __name__ == "__main__":
    main()
