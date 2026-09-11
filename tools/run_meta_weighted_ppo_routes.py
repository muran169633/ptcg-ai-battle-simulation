#!/usr/bin/env python3
"""Launch corrected Dragapult or Rmy PPO from a freshly trained general BC.

The frozen-opponent draw keeps 10% live same-deck self-play.  The remaining
90% is distributed by exact-deck frequency from the latest complete replay
day, then adjusted online by inverse Beta(2, 2)-smoothed win rate over each
opponent's latest 200 completed games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from build_meta_weighted_ppo_protocol import atomic_write_json, build_protocol


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
FEATURE_SOURCE = REPO / (
    "frozen_behavior_source__sha256_"
    "f6c92841f2f6974408be05600b80f532b099a816a40ef2a5b2168fe45dc713ab"
) / "train_bc_orbit.py"
FEATURE_SOURCE_SHA256 = (
    "ee68efc864511d2c88da0429aba8500611e9a2a5a1738573c8073a58cb73a9cc"
)
META_POOL = REPO / "data/recent_day_meta_pool_20260813_top23_v1/meta_pool.json"
META_POOL_SHA256 = (
    "2ea9e93775c37c8723785399271a66e2558822d46650ecd97f8dade7fac8977c"
)


@dataclass(frozen=True)
class Route:
    learner_deck: Path
    replay_archive: Path
    replay_sha256: str
    seed: int
    default_output: Path
    default_protocol: Path


ROUTES = {
    "dragapult": Route(
        learner_deck=REPO
        / "data/recent_day_meta_pool_20260813_top23_v1/decks/"
        "rank02_07bedfffbfad.csv",
        replay_archive=REPO
        / "data/dragapult_current_07bed_20260812_v1/dragapult_07bed_aug12.zip",
        replay_sha256=(
            "a67cad9ab1382706d79a7ba00e40f7ed332b8f801e37c5fae54399c819faa60a"
        ),
        seed=2026081411,
        default_output=REPO
        / "artifacts/dragapult_metaweighted_20260814_v2/"
        "ppo_recent1_freq_invwin200_90pool10self_u120_seed2026081411",
        default_protocol=REPO
        / "artifacts/meta_weighted_protocols_20260814_v2/dragapult_protocol.json",
    ),
    "rmy": Route(
        learner_deck=REPO
        / "data/recent_day_meta_pool_20260813_top23_v1/decks/"
        "rank05_7e3984370203.csv",
        replay_archive=REPO
        / "data/rmy_exact_public_20260814_v1/snapshot_20260814T081156Z/"
        "policies/rmy_ogerpon_hydrapple_live_55482569.zip",
        replay_sha256=(
            "0c80943ffee5b8460883202e9cf0a2e9c03c34b3ef40190dce847425be1e290e"
        ),
        seed=2026081412,
        default_output=REPO
        / "artifacts/rmy_metaweighted_20260814_v2/"
        "ppo_recent1_freq_invwin200_90pool10self_u120_seed2026081412",
        default_protocol=REPO
        / "artifacts/meta_weighted_protocols_20260814_v2/rmy_protocol.json",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, expected_sha256: str | None = None) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}"
            )
    return path


def build_command(
    route_name: str,
    bc_checkpoint: Path,
    output_dir: Path,
    protocol_output: Path,
    updates: int,
) -> tuple[list[str], dict[str, object]]:
    route = ROUTES[route_name]
    bc_checkpoint = checked_file(bc_checkpoint)
    checked_file(FEATURE_SOURCE, FEATURE_SOURCE_SHA256)
    checked_file(META_POOL, META_POOL_SHA256)
    checked_file(route.learner_deck)
    checked_file(route.replay_archive, route.replay_sha256)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    if updates < 1:
        raise ValueError("--updates must be positive")

    protocol = build_protocol(
        META_POOL,
        route.learner_deck,
        bc_checkpoint,
        league_probability=0.90,
        window_games=200,
        inverse_min_factor=0.5,
        inverse_max_factor=2.5,
    )
    atomic_write_json(protocol_output.resolve(), protocol)

    command = [
        str(PYTHON),
        "tools/train_ppo.py",
        "--bc-checkpoint",
        str(bc_checkpoint),
        "--kl-reference-checkpoint",
        str(bc_checkpoint),
        "--deck",
        str(route.learner_deck),
        "--output-dir",
        str(output_dir),
        "--updates",
        str(updates),
        "--environments",
        "64",
        "--games-per-update",
        "1024",
        "--ppo-epochs",
        "4",
        "--minibatch-size",
        "4096",
        "--learning-rate",
        "3e-6",
        "--value-learning-rate",
        "1.5e-6",
        "--weight-decay",
        "0",
        "--gamma",
        "1",
        "--gae-lambda",
        "1",
        "--advantage-normalization",
        "per_opponent",
        "--clip-ratio",
        "0.10",
        "--value-coefficient",
        "0.2",
        "--value-trunk-gradient-scale",
        "0",
        "--entropy-coefficient",
        "0.0005",
        "--max-grad-norm",
        "0.25",
        "--policy-temperature",
        "0.8",
        "--trainable-scope",
        "last_block_heads",
        "--learning-rate-schedule",
        "cosine",
        "--schedule-start-update",
        "1",
        "--bc-kl-start",
        "0.04",
        "--bc-kl-end",
        "0.04",
        "--target-kl",
        "0.0005",
        "--history-opponent-weight",
        "0",
        "--opponent-quota-mode",
        "legacy",
        "--ppo-objective",
        "standard",
        "--actor-reduction",
        "episode_mean",
        "--actor-value-gradient-mode",
        "scalar",
        "--snapshot-interval",
        "1000000",
        "--bc-replay-data",
        str(route.replay_archive),
        "--bc-replay-split",
        "train",
        "--bc-replay-batches",
        "64",
        "--bc-replay-batch-size",
        "128",
        "--bc-replay-workers",
        "4",
        "--bc-replay-steps",
        "16",
        "--bc-replay-lr-scale",
        "0.15",
        "--bc-replay-loss",
        "hybrid_ordered",
        "--eval-interval",
        "10",
        "--eval-games",
        "32",
        "--eval-all-permanent-opponents",
        "--selection-aggregation",
        "mean",
        "--champion-gate-interval",
        "10",
        "--champion-gate-games",
        "200",
        "--champion-gate-min-win-rate",
        "0.54",
        "--champion-gate-threshold-inclusive",
        "--checkpoint-interval",
        "10",
        "--max-game-decisions",
        "1000",
        "--seed",
        str(route.seed),
        "--skip-initial-eval",
        "--device",
        "cuda",
        *[str(value) for value in protocol["train_ppo_argument_fragment"]],
    ]
    return command, protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=sorted(ROUTES), required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol-output", type=Path)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route = ROUTES[args.route]
    command, protocol = build_command(
        args.route,
        args.bc_checkpoint,
        args.output_dir or route.default_output,
        args.protocol_output or route.default_protocol,
        args.updates,
    )
    audit = {
        "route": args.route,
        "command": command,
        "protocol": protocol,
        "feature_source": {
            "path": str(FEATURE_SOURCE),
            "sha256": FEATURE_SOURCE_SHA256,
        },
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment["PTCG_BC_FEATURE_SOURCE"] = str(FEATURE_SOURCE)
    environment["PTCG_BC_FEATURE_SOURCE_SHA256"] = FEATURE_SOURCE_SHA256
    subprocess.run(command, cwd=REPO, env=environment, check=True)


if __name__ == "__main__":
    main()
