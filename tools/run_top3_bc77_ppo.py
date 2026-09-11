#!/usr/bin/env python3
"""Run guarded fresh PPO for the BC-qualified members of the Top-3 portfolio.

The portfolio-level gate is a row-weighted full-valid exact accuracy of at
least 77%.  A learner must also exceed 77% on its own full validation split;
therefore the current Froslass route can be used as an opponent but cannot
start PPO.  Commands are printed by default and all outputs are versioned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAIN_PPO = ROOT / "tools/train_ppo.py"
OUTPUT_ROOT = ROOT / "artifacts/top3_bc77_20260809"
BC_FEATURE = "ptcg-bc-orbit-entity-transformer-v5"
GATE = 0.77


@dataclass(frozen=True)
class Profile:
    slug: str
    checkpoint: Path
    summary: Path
    archive: Path
    deck: Path
    deck_hash: str
    seed: int


PROFILES = {
    "alakazam_control": Profile(
        slug="alakazam_control",
        checkpoint=ROOT / "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/best.pt",
        summary=ROOT / "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/summary.json",
        archive=ROOT / "data/gold8_recent7_20260808/archives/alakazam_control.zip",
        deck=ROOT / "data/gold8_recent7_20260808/decks/3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv",
        deck_hash="3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf",
        seed=202608971,
    ),
    "marnie": Profile(
        slug="marnie",
        checkpoint=ROOT / "artifacts/gold8_recent7_20260808/marnie/counttrunk0_light_v1/best.pt",
        summary=ROOT / "artifacts/gold8_recent7_20260808/marnie/counttrunk0_light_v1/summary.json",
        archive=ROOT / "data/gold8_recent7_20260808/archives/marnie.zip",
        deck=ROOT / "data/gold8_recent7_20260808/decks/c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv",
        deck_hash="c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af",
        seed=202608972,
    ),
    "mega_froslass_lopunny": Profile(
        slug="mega_froslass_lopunny",
        checkpoint=OUTPUT_ROOT / "froslass_all_from_recent_shuffle4096_flex125_v1/best.pt",
        summary=OUTPUT_ROOT / "froslass_all_from_recent_shuffle4096_flex125_v1/summary.json",
        archive=ROOT / "data/gold8_recent7_20260808/archives/mega_froslass_lopunny.zip",
        deck=ROOT / "data/gold8_recent7_20260808/decks/dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc.csv",
        deck_hash="dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc",
        seed=202608973,
    ),
}


AUXILIARIES = (
    (
        ROOT / "artifacts/gold8_recent7_20260808/mega_lucario/specialist_bc/best.pt",
        ROOT / "data/gold8_recent7_20260808/decks/77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8.csv",
        8,
    ),
    (
        ROOT / "artifacts/gold8_recent7_20260808/hydrapple_ogerpon/specialist_bc/best.pt",
        ROOT / "data/gold8_recent7_20260808/decks/0a6ca2ca3e72f1d6c0860cb653f16b05c4ba9b8d7774d5669c608e43f4c154f2.csv",
        8,
    ),
    (
        ROOT / "artifacts/gold8_recent7_20260808/mega_kangaskhan_crustle/specialist_bc/best.pt",
        ROOT / "data/gold8_recent7_20260808/decks/4bf59ca589c2d685d74e3535424c2dbe3c11389dffba59eddba4567be7e437df.csv",
        8,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_metrics(profile: Profile) -> tuple[int, float]:
    payload = json.loads(profile.summary.read_text(encoding="utf-8"))
    metrics = payload.get("best_valid_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{profile.summary}: missing best_valid_metrics")
    rows = int(metrics.get("rows", 0))
    exact = float(metrics.get("exact_action_set_accuracy", -1.0))
    if rows <= 0 or not 0.0 <= exact <= 1.0:
        raise ValueError(f"{profile.summary}: invalid full-valid metrics")
    return rows, exact


def portfolio_gate() -> dict[str, Any]:
    members: dict[str, Any] = {}
    correct = 0.0
    rows_total = 0
    for slug, profile in PROFILES.items():
        rows, exact = valid_metrics(profile)
        members[slug] = {
            "rows": rows,
            "exact_action_set_accuracy": exact,
            "checkpoint": str(profile.checkpoint),
            "checkpoint_sha256": sha256_file(profile.checkpoint),
        }
        correct += rows * exact
        rows_total += rows
    weighted = correct / rows_total
    return {
        "threshold": GATE,
        "rows": rows_total,
        "row_weighted_exact_action_set_accuracy": weighted,
        "passed": weighted >= GATE,
        "members": members,
    }


def opponent_name(checkpoint: Path, deck: Path) -> str:
    return f"{checkpoint.stem}@{deck.stem}"


def output_dir(profile: Profile) -> Path:
    return OUTPUT_ROOT / profile.slug / "ppo_light_top3_v1"


def ppo_command(profile: Profile) -> list[str]:
    command = [
        str(PYTHON),
        "-B",
        str(TRAIN_PPO),
        "--bc-checkpoint",
        str(profile.checkpoint),
        "--kl-reference-checkpoint",
        str(profile.checkpoint),
        "--deck",
        str(profile.deck),
        "--output-dir",
        str(output_dir(profile)),
        "--updates",
        "4",
        "--schedule-start-update",
        "1",
        "--environments",
        "16",
        "--games-per-update",
        "96",
        "--ppo-epochs",
        "2",
        "--minibatch-size",
        "512",
        "--learning-rate",
        "0.000012",
        "--value-learning-rate",
        "0.0000075",
        "--weight-decay",
        "0.0001",
        "--learning-rate-schedule",
        "constant",
        "--gamma",
        "1.0",
        "--gae-lambda",
        "0.97",
        "--advantage-normalization",
        "global",
        "--clip-ratio",
        "0.15",
        "--value-coefficient",
        "0.25",
        "--value-trunk-gradient-scale",
        "0.0",
        "--entropy-coefficient",
        "0.001",
        "--max-grad-norm",
        "0.5",
        "--policy-temperature",
        "0.8",
        "--trainable-scope",
        "last_block_heads",
        "--bc-kl-start",
        "0.016",
        "--bc-kl-end",
        "0.012",
        "--target-kl",
        "0.006",
        "--league-probability",
        "1.0",
        "--opponent-sampling",
        "per_game",
        "--opponent-quota-mode",
        "fixed",
        "--opponent-quota-seat-balance",
        "--ppo-objective",
        "standard",
        "--actor-reduction",
        "episode_mean",
        "--actor-value-gradient-mode",
        "scalar",
        "--constrained-gradient-mode",
        "scalar",
        "--snapshot-interval",
        "1000000",
        "--max-pool-size",
        "8",
        "--bc-replay-data",
        str(profile.archive),
        "--bc-replay-split",
        "train",
        "--bc-replay-batches",
        "72",
        "--bc-replay-batch-size",
        "256",
        "--bc-replay-workers",
        "8",
        "--bc-replay-steps",
        "2",
        "--bc-replay-lr-scale",
        "0.075",
        "--bc-replay-loss",
        "ordered",
        "--bc-replay-order-context-weight",
        "8.0",
        "--bc-replay-non-context34-fixed-multi-action-order-weight",
        "1.0",
        "--bc-replay-context34-rows-per-batch",
        "4",
        "--eval-interval",
        "2",
        "--eval-games",
        "32",
        "--eval-all-permanent-opponents",
        "--selection-aggregation",
        "mean",
        "--checkpoint-interval",
        "2",
        "--seed",
        str(profile.seed),
        "--device",
        "cuda",
        "--opponent-base-quota",
        "bc",
        "36",
    ]
    extras: list[tuple[Path, Path, int]] = []
    for other in PROFILES.values():
        if other.slug != profile.slug:
            extras.append((other.checkpoint, other.deck, 18))
    extras.extend(AUXILIARIES)
    if sum(quota for _, _, quota in extras) + 36 != 96:
        raise AssertionError("Top-3 PPO quotas must sum to 96")
    for checkpoint, deck, quota in extras:
        command.extend(("--extra-opponent", str(checkpoint), str(deck)))
        command.extend(
            (
                "--opponent-base-quota",
                opponent_name(checkpoint, deck),
                str(quota),
            )
        )
    return command


def validate(profile: Profile, execute: bool) -> dict[str, Any]:
    required = [PYTHON, TRAIN_PPO]
    for member in PROFILES.values():
        required.extend(
            (member.checkpoint, member.summary, member.archive, member.deck)
        )
    for checkpoint, deck, _ in AUXILIARIES:
        required.extend((checkpoint, deck))
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))

    gate = portfolio_gate()
    if not gate["passed"]:
        raise RuntimeError(
            "Portfolio BC77 gate failed: "
            f"{gate['row_weighted_exact_action_set_accuracy']:.6f} < {GATE:.2f}"
        )
    _, route_exact = valid_metrics(profile)
    if route_exact < GATE:
        raise RuntimeError(
            f"{profile.slug} route BC77 gate failed: {route_exact:.6f} < {GATE:.2f}"
        )
    checkpoint = torch.load(
        profile.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("feature_version") != BC_FEATURE:
        raise ValueError(f"{profile.checkpoint}: not a V5 BC checkpoint")
    declared = set((checkpoint.get("config") or {}).get("deck_hashes") or [])
    if declared and declared != {profile.deck_hash}:
        raise ValueError(
            f"{profile.checkpoint}: checkpoint deck hashes {declared} do not "
            f"match {profile.deck_hash}"
        )
    target = output_dir(profile)
    if execute and target.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {target}")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("alakazam_control", "marnie", "mega_froslass_lopunny"),
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    gate = validate(profile, args.execute)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    command = ppo_command(profile)
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
