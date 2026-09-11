#!/usr/bin/env python3
"""Preflight and launch synchronized mode-AR PPO on an 8x A100 host."""

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
FROZEN_REL = Path(
    "artifacts/frozen_incumbents/"
    "dragapult_mode_ar_submit55527088_20260815_v1"
)
DEFAULT_ANCHOR = ROOT / FROZEN_REL / "submitted_policy_update0040.pt"
DEFAULT_META = (
    ROOT / "data" / "recent_day_meta_pool_20260813_top23_v1" / "meta_pool.json"
)
DEFAULT_DECK = ROOT / FROZEN_REL / "deck.csv"
EXPECTED_ANCHOR_SHA256 = (
    "5c8e2659a9a6528202e4f24a75cb0058321584ba4160b0514aca59fb3ffb8e0e"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "dragapult_mode_ar_u40_8gpu",
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--updates", type=int, default=8_000)
    parser.add_argument("--environments-per-rank", type=int, default=128)
    parser.add_argument("--games-per-rank-update", type=int, default=256)
    parser.add_argument("--minibatch-size-per-rank", type=int, default=4096)
    parser.add_argument("--rollout-workers-per-rank", type=int, default=4)
    parser.add_argument("--rollout-envs-per-worker", type=int, default=32)
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=10.0)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--candidate-snapshot-interval", type=int, default=5)
    parser.add_argument("--champion-gate-interval", type=int, default=5)
    parser.add_argument("--champion-gate-games", type=int, default=1000)
    parser.add_argument("--bc-eval-games", type=int, default=1000)
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--meta-pool", type=Path, default=DEFAULT_META)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--seed", type=int, default=2026081640)
    parser.add_argument("--master-port", type=int, default=29527)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    positive = (
        args.nproc_per_node,
        args.updates,
        args.environments_per_rank,
        args.games_per_rank_update,
        args.minibatch_size_per_rank,
        args.rollout_workers_per_rank,
        args.rollout_envs_per_worker,
        args.checkpoint_interval,
        args.candidate_snapshot_interval,
        args.champion_gate_interval,
        args.champion_gate_games,
        args.bc_eval_games,
        args.evaluation_workers,
    )
    if any(value < 1 for value in positive):
        raise ValueError("All count and interval arguments must be positive")
    if args.nproc_per_node < 2:
        raise ValueError("--nproc-per-node must be at least 2")
    if args.rollout_batch_wait_ms < 0.0:
        raise ValueError("--rollout-batch-wait-ms must be non-negative")
    if (
        args.rollout_workers_per_rank * args.rollout_envs_per_worker
        < args.environments_per_rank
    ):
        raise ValueError(
            "rollout workers * environments per worker must cover "
            "--environments-per-rank"
        )
    if args.champion_gate_games % 2 or args.bc_eval_games % 2:
        raise ValueError("Evaluation game counts must be even")
    visible_gpus = torch.cuda.device_count()
    if not args.dry_run and visible_gpus < args.nproc_per_node:
        raise RuntimeError(
            f"Requested {args.nproc_per_node} ranks but only {visible_gpus} "
            "CUDA devices are visible"
        )

    anchor = args.anchor_checkpoint.resolve()
    meta_pool = args.meta_pool.resolve()
    deck = args.deck.resolve()
    required = (
        anchor,
        meta_pool,
        deck,
        ROOT / "tools" / "train_mode_ar_ppo_ddp.py",
        ROOT / "tools" / "train_mode_ar_ppo.py",
        ROOT / "dataset" / "sample_submission" / "sample_submission" / "cg" / "libcg.so",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    anchor_sha = sha256(anchor)
    if anchor == DEFAULT_ANCHOR.resolve() and anchor_sha != EXPECTED_ANCHOR_SHA256:
        raise RuntimeError(
            "Frozen submitted update40 hash mismatch: "
            f"actual={anchor_sha} expected={EXPECTED_ANCHOR_SHA256}"
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {output_dir}"
        )
    if output_dir.exists():
        raise FileExistsError(
            "Mode-AR trainer requires a new output path; remove the empty "
            f"directory or choose another: {output_dir}"
        )

    global_games_per_update = (
        args.nproc_per_node * args.games_per_rank_update
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--master-port={args.master_port}",
        str(ROOT / "tools" / "train_mode_ar_ppo_ddp.py"),
        "--bc-checkpoint",
        str(anchor),
        "--meta-pool",
        str(meta_pool),
        "--route",
        "dragapult",
        "--learner-deck",
        str(deck),
        "--output-dir",
        str(output_dir),
        "--updates",
        str(args.updates),
        "--environments",
        str(args.environments_per_rank),
        "--games-per-update",
        str(args.games_per_rank_update),
        "--ppo-epochs",
        "4",
        "--minibatch-size",
        str(args.minibatch_size_per_rank),
        "--actor-learning-rate",
        "3e-6",
        "--value-learning-rate",
        "1.5e-6",
        "--weight-decay",
        "0",
        "--gamma",
        "1",
        "--gae-lambda",
        "1",
        "--policy-temperature",
        "0.8",
        "--clip-ratio",
        "0.10",
        "--value-coefficient",
        "0.2",
        "--entropy-coefficient",
        "0.0005",
        "--anchor-kl-coefficient",
        "0.04",
        "--target-kl",
        "0.0005",
        "--max-grad-norm",
        "0.25",
        "--league-probability",
        "0.90",
        "--fixed-meta-probability",
        "0.70",
        "--inverse-meta-probability",
        "0.20",
        "--opponent-window-games",
        "200",
        "--opponent-inverse-min-factor",
        "0.5",
        "--opponent-inverse-max-factor",
        "2.5",
        "--initialize-opponent-window-from-anchor",
        "--champion-gate-interval",
        str(args.champion_gate_interval),
        "--champion-gate-games",
        str(args.champion_gate_games),
        "--champion-gate-min-win-rate",
        "0.54",
        "--champion-gate-error-margin",
        "0.05",
        "--bc-eval-games",
        str(args.bc_eval_games),
        "--evaluation-workers",
        str(args.evaluation_workers),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--candidate-snapshot-interval",
        str(args.candidate_snapshot_interval),
        "--rollback-on-gate-failure",
        "--reset-optimizer-on-gate-rollback",
        "--max-game-decisions",
        "1000",
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--rollout-workers",
        str(args.rollout_workers_per_rank),
        "--rollout-envs-per-worker",
        str(args.rollout_envs_per_worker),
        "--rollout-batch-wait-ms",
        str(args.rollout_batch_wait_ms),
    ]
    manifest = {
        "schema_version": "ptcg-mode-ar-u40-ddp-launch-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "world_size": args.nproc_per_node,
        "visible_gpus": visible_gpus,
        "target_gpu": "NVIDIA A100 40GB",
        "updates": args.updates,
        "games_per_rank_update": args.games_per_rank_update,
        "global_games_per_update": global_games_per_update,
        "planned_training_games": args.updates * global_games_per_update,
        "environments_per_rank": args.environments_per_rank,
        "minibatch_size_per_rank": args.minibatch_size_per_rank,
        "effective_global_minibatch_size": (
            args.minibatch_size_per_rank * args.nproc_per_node
        ),
        "rollout_workers_per_rank": args.rollout_workers_per_rank,
        "rollout_envs_per_worker": args.rollout_envs_per_worker,
        "total_rollout_workers": (
            args.rollout_workers_per_rank * args.nproc_per_node
        ),
        "rollout_batch_wait_ms": args.rollout_batch_wait_ms,
        "anchor_checkpoint": str(anchor),
        "anchor_checkpoint_sha256": anchor_sha,
        "anchor_expected_submission_update": 40,
        "sampling": {"fixed_recent_meta": 0.70, "inverse_window": 0.20, "current_selfplay": 0.10},
        "checkpoint_interval": args.checkpoint_interval,
        "candidate_snapshot_interval": args.candidate_snapshot_interval,
        "champion_gate": {
            "interval": args.champion_gate_interval,
            "games": args.champion_gate_games,
            "minimum_win_rate": 0.54,
            "error_margin": 0.05,
            "rollback_on_failure": True,
            "reset_optimizer_on_rollback": True,
        },
        "checkpoint_writer": "rank0_only",
        "gradient_sync": "torch DistributedDataParallel NCCL",
        "command": command,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    launch_manifest = output_dir.parent / (
        output_dir.name + ".distributed_launch_manifest.json"
    )
    launch_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log_path = output_dir.parent / (output_dir.name + ".train.log")
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    environment.setdefault("NCCL_DEBUG", "WARN")
    with log_path.open("w", encoding="utf-8") as log_file:
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
