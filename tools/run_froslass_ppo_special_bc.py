#!/usr/bin/env python3
"""Plan or execute guarded Froslass BC -> PPO -> special-BC stages.

The learner starts from the validation-selected recent-demonstrator BC.  PPO
uses a fixed, seat-balanced eight-opponent league, a reduced actor learning
rate, no value-loss gradient into the shared trunk, and exact-deck BC replay.
The post-PPO stage is the repository's deterministic actor-only special BC.

Commands are printed by default.  Pass ``--execute`` to mutate versioned
artifact directories.  Packaging and Kaggle submission are deliberately not
part of this launcher.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAIN_PPO = ROOT / "tools/train_ppo.py"
SPECIAL_BC = ROOT / "tools/apply_gold8_special_bc.py"

PROFILE_ROOT = ROOT / "artifacts/portfolio4_recent7_20260809/mega_froslass_lopunny"
BC = PROFILE_ROOT / "bc_recent_demo_counttrunk0_v1/best.pt"
PPO_OUTPUT = PROFILE_ROOT / "ppo_light_from_recent_demo_v1"
SPECIAL_OUTPUT = PROFILE_ROOT / "special_bc_after_ppo_light_v1"
DATA = ROOT / "data/gold8_recent7_20260808/archives/mega_froslass_lopunny.zip"
DECK = ROOT / "data/gold8_recent7_20260808/decks/dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc.csv"
DECK_HASH = "dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc"

OLD_FROSLASS = ROOT / "artifacts/gold8_recent7_20260808/mega_froslass_lopunny/specialist_bc/best.pt"
ALAKAZAM = ROOT / "artifacts/gold8_recent7_20260808/alakazam_control/specialist_bc/best.pt"
ALAKAZAM_DECK = ROOT / "data/gold8_recent7_20260808/decks/3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf.csv"
MARNIE = ROOT / "artifacts/gold8_recent7_20260808/marnie/counttrunk0_light_v1/best.pt"
MARNIE_DECK = ROOT / "data/gold8_recent7_20260808/decks/c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af.csv"
GENERAL = ROOT / "artifacts/gold8_recent7_20260808/general_bc/best.pt"
SLOWKING_DECK = ROOT / "data/portfolio4_recent7_20260809/decks/df6f7443719675711c2dcead3559087de996f3b8c43192d55ebfe6182e43cb7c.csv"
CRUSTLE_DECK = ROOT / "data/gold8_recent7_20260808/decks/4bf59ca589c2d685d74e3535424c2dbe3c11389dffba59eddba4567be7e437df.csv"
MEGA_LUCARIO = ROOT / "artifacts/gold8_recent7_20260808/mega_lucario/specialist_bc/best.pt"
MEGA_LUCARIO_DECK = ROOT / "data/gold8_recent7_20260808/decks/77a53ffc32f89b22562f6b4ac0b8cbde9e8210923cd0ef512551b8a8eb9003f8.csv"
HYDRAPPLE = ROOT / "artifacts/gold8_recent7_20260808/hydrapple_ogerpon/specialist_bc/best.pt"
HYDRAPPLE_DECK = ROOT / "data/gold8_recent7_20260808/decks/0a6ca2ca3e72f1d6c0860cb653f16b05c4ba9b8d7774d5669c608e43f4c154f2.csv"


def extra_name(checkpoint: Path, deck: Path) -> str:
    return f"{checkpoint.stem}@{deck.stem}"


def ppo_command() -> list[str]:
    command = [
        str(PYTHON),
        "-B",
        str(TRAIN_PPO),
        "--bc-checkpoint",
        str(BC),
        "--deck",
        str(DECK),
        "--output-dir",
        str(PPO_OUTPUT),
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
        "--actor-value-gradient-mode",
        "scalar",
        "--constrained-gradient-mode",
        "scalar",
        "--actor-reduction",
        "episode_mean",
        "--snapshot-interval",
        "1000000",
        "--max-pool-size",
        "8",
        "--bc-replay-data",
        str(DATA),
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
        "202608931",
        "--device",
        "cuda",
    ]
    extras = (
        (OLD_FROSLASS, DECK, 12),
        (ALAKAZAM, ALAKAZAM_DECK, 12),
        (MARNIE, MARNIE_DECK, 12),
        (GENERAL, SLOWKING_DECK, 12),
        (GENERAL, CRUSTLE_DECK, 12),
        (MEGA_LUCARIO, MEGA_LUCARIO_DECK, 12),
        (HYDRAPPLE, HYDRAPPLE_DECK, 12),
    )
    command.extend(("--opponent-base-quota", "bc", "12"))
    for checkpoint, deck, quota in extras:
        command.extend(("--extra-opponent", str(checkpoint), str(deck)))
        command.extend(
            ("--opponent-base-quota", extra_name(checkpoint, deck), str(quota))
        )
    return command


def special_command(parent_checkpoint: Path) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        str(SPECIAL_BC),
        "--parent-checkpoint",
        str(parent_checkpoint),
        "--special-data",
        str(DATA),
        "--expected-deck-hash",
        DECK_HASH,
        "--steps",
        "4",
        "--cache-batches",
        "32",
        "--batch-size",
        "256",
        "--workers",
        "8",
        "--learning-rate-scale",
        "0.025",
        "--context34-rows-per-batch",
        "1",
        "--seed",
        "202608932",
        "--output-dir",
        str(SPECIAL_OUTPUT),
        "--device",
        "cuda",
    ]


def validate_inputs(
    stage: str,
    execute: bool,
    parent_checkpoint: Path | None,
) -> None:
    required = [
        PYTHON,
        TRAIN_PPO,
        SPECIAL_BC,
        BC,
        DATA,
        DECK,
        OLD_FROSLASS,
        ALAKAZAM,
        ALAKAZAM_DECK,
        MARNIE,
        MARNIE_DECK,
        GENERAL,
        SLOWKING_DECK,
        CRUSTLE_DECK,
        MEGA_LUCARIO,
        MEGA_LUCARIO_DECK,
        HYDRAPPLE,
        HYDRAPPLE_DECK,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))
    if stage == "special-bc":
        if parent_checkpoint is None:
            raise ValueError("--parent-checkpoint is required for special-bc")
        if not parent_checkpoint.is_file():
            raise FileNotFoundError(parent_checkpoint)
    target = PPO_OUTPUT if stage == "ppo" else SPECIAL_OUTPUT
    if execute and target.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("ppo", "special-bc"), required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parent = args.parent_checkpoint.resolve() if args.parent_checkpoint else None
    validate_inputs(args.stage, args.execute, parent)
    command = ppo_command() if args.stage == "ppo" else special_command(parent)
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
