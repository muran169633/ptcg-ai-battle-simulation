#!/usr/bin/env python3
"""Preflight and launch one synchronized non-AR PPO policy on 8 GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BC_DIR = (
    ROOT
    / "artifacts"
    / "bc_top100_recent14_nonar_order_v7_end0813_b2048_20260816_v1"
)
DEFAULT_META = (
    ROOT / "data" / "recent_day_meta_pool_20260813_top23_v1" / "meta_pool.json"
)
DEFAULT_DECK = (
    ROOT
    / "data"
    / "recent_day_meta_pool_20260813_top23_v1"
    / "decks"
    / "rank02_07bedfffbfad.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "dragapult_8gpu_1m")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument(
        "--updates",
        type=int,
        default=8_000,
        help="8000 * 8 ranks * 256 games = 16,384,000 training games",
    )
    parser.add_argument("--environments-per-rank", type=int, default=128)
    parser.add_argument("--games-per-rank-update", type=int, default=256)
    parser.add_argument(
        "--minibatch-size-per-rank",
        type=int,
        default=4096,
        help="Per-GPU PPO minibatch; 4096 targets the packaged 8x A100-40GB host",
    )
    parser.add_argument("--rollout-workers-per-rank", type=int, default=4)
    parser.add_argument("--rollout-envs-per-worker", type=int, default=32)
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=10.0)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save a resumable rank-0 checkpoint every N global updates",
    )
    parser.add_argument("--bc-dir", type=Path, default=DEFAULT_BC_DIR)
    parser.add_argument("--meta-pool", type=Path, default=DEFAULT_META)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2026081621)
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.nproc_per_node < 2:
        raise ValueError("--nproc-per-node must be at least 2")
    visible_gpus = torch.cuda.device_count()
    if not args.dry_run and visible_gpus < args.nproc_per_node:
        raise RuntimeError(
            f"Requested {args.nproc_per_node} ranks but only {visible_gpus} "
            "CUDA devices are visible"
        )
    if (
        args.updates < 1
        or args.games_per_rank_update < 1
        or args.checkpoint_interval < 1
    ):
        raise ValueError("updates and games per rank must be positive")
    if args.rollout_workers_per_rank < 1 or args.rollout_envs_per_worker < 1:
        raise ValueError("Parallel rollout worker counts must be positive")
    active_environments_per_rank = min(
        args.environments_per_rank,
        args.rollout_workers_per_rank * args.rollout_envs_per_worker,
    )
    if active_environments_per_rank < args.environments_per_rank:
        raise ValueError(
            "rollout workers * environments per worker must cover "
            "--environments-per-rank"
        )

    output_dir = args.output_dir.resolve()
    resume = args.resume.resolve() if args.resume is not None else None
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bc_best = (args.bc_dir / "best.pt").resolve()
    required = [
        bc_best,
        args.meta_pool.resolve(),
        args.deck.resolve(),
        ROOT / "tools" / "train_nonar_league_ppo_ddp.py",
        ROOT / "tools" / "parallel_rollout.py",
        ROOT / "dataset" / "sample_submission" / "sample_submission" / "cg" / "libcg.so",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    global_games_per_update = (
        args.nproc_per_node * args.games_per_rank_update
    )
    planned_games = args.updates * global_games_per_update
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--master-port={args.master_port}",
        str(ROOT / "tools" / "launch_nonar_dragapult_ppo_fast.py"),
        "--trainer",
        str(ROOT / "tools" / "train_nonar_league_ppo_ddp.py"),
        "--output-dir",
        str(output_dir),
        "--bc-dir",
        str(args.bc_dir.resolve()),
        "--meta-pool",
        str(args.meta_pool.resolve()),
        "--deck",
        str(args.deck.resolve()),
        "--updates",
        str(args.updates),
        "--environments",
        str(args.environments_per_rank),
        "--games-per-update",
        str(args.games_per_rank_update),
        "--minibatch-size",
        str(args.minibatch_size_per_rank),
        "--seed",
        str(args.seed),
        "--rollout-workers",
        str(args.rollout_workers_per_rank),
        "--rollout-envs-per-worker",
        str(args.rollout_envs_per_worker),
        "--rollout-batch-wait-ms",
        str(args.rollout_batch_wait_ms),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        # Preserve the intended 5,000/15,000-game gate cooldowns after the
        # eight-fold increase in games collected per learner update.
        "--champion-initial-wait",
        "1",
        "--champion-failure-cooldown",
        str(max(1, math.ceil(5_000 / global_games_per_update))),
        "--champion-success-cooldown",
        str(max(1, math.ceil(15_000 / global_games_per_update))),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))

    manifest = {
        "schema_version": "ptcg-nonar-v7-ddp-launch-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_size": args.nproc_per_node,
        "visible_gpus": visible_gpus,
        "updates": args.updates,
        "games_per_rank_update": args.games_per_rank_update,
        "global_games_per_update": global_games_per_update,
        "planned_training_games": planned_games,
        "environments_per_rank": args.environments_per_rank,
        "rollout_workers_per_rank": args.rollout_workers_per_rank,
        "rollout_envs_per_worker": args.rollout_envs_per_worker,
        "total_cpu_rollout_workers": (
            args.rollout_workers_per_rank * args.nproc_per_node
        ),
        "rollout_batch_wait_ms": args.rollout_batch_wait_ms,
        "checkpoint_interval": args.checkpoint_interval,
        "planned_periodic_checkpoints": math.ceil(
            args.updates / args.checkpoint_interval
        ),
        "minibatch_size_per_rank": args.minibatch_size_per_rank,
        "effective_global_minibatch_size": (
            args.minibatch_size_per_rank * args.nproc_per_node
        ),
        "bc_checkpoint": str(bc_best),
        "bc_checkpoint_sha256": sha256(bc_best),
        "resume_checkpoint": str(resume) if resume is not None else None,
        "command": command,
        "checkpoint_writer": "rank0_only",
        "gradient_sync": "torch DistributedDataParallel NCCL",
    }
    (output_dir / "distributed_launch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    environment.setdefault("NCCL_DEBUG", "WARN")
    log_mode = "a" if resume is not None else "w"
    with (output_dir / "train.log").open(log_mode, encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("torchrun stdout pipe was not created")
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
        return_code = process.wait()
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
