#!/usr/bin/env python3
"""Run non-AR league PPO with local CPU process-parallel rollout."""

from __future__ import annotations

import argparse
import sys

import train_nonar_league_ppo as nonar_entry
import train_ppo as legacy
from parallel_rollout import collect_rollout_parallel, install_parallel_rollout


def _extract_parallel_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--rollout-envs-per-worker", type=int, default=16)
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=10.0)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    return args


def main() -> None:
    parallel = _extract_parallel_args()
    nonar_entry.install_nonar_adapter()
    install_parallel_rollout(
        workers=parallel.rollout_workers,
        environments_per_worker=parallel.rollout_envs_per_worker,
        batch_wait_ms=parallel.rollout_batch_wait_ms,
    )
    legacy.collect_rollout = collect_rollout_parallel
    legacy.main()


if __name__ == "__main__":
    main()
