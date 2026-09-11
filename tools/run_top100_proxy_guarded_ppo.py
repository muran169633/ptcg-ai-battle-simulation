#!/usr/bin/env python3
"""Run one conservative Marnie PPO update against eight Top100 proxy decks.

Teal Mask Ogerpon is the primary improvement objective after the frozen panel
identified it as the BC model's dominant failure mode.  Current
Kangaskhan/Crustle remains the guard objective because the preceding standard
PPO probe regressed that matchup.  A primal-dual guard constraint and a frozen
BC KL anchor bound the update across all eight opponent groups.  No packaging
or submission code exists in this launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/xxc/miniconda3/envs/my_project_env/bin/python")
TRAINER = ROOT / "tools/train_ppo.py"
PROXY_ROOT = ROOT / "artifacts/top100_proxy_recent14_20260811_v1"
DATA_ROOT = ROOT / "data/top100_proxy_recent14_20260811_v1"
PARENT = ROOT / "artifacts/gold_push_marnie_recent7_seed817recipe_20260811_v1/bc_counttrunk0_balhalf_seed3601/best.pt"
GENERAL_REPLAY = ROOT / "data/gold_push_marnie_recent7_20260811_v1/marnie.zip"
DEFAULT_OUTPUT = ROOT / "artifacts/top100_proxy_recent14_20260811_v1/ppo_ogerpon_guarded1x640_seed4206"
MARNIE_HASH = "c20a8a46f5c635773754f03103652f5c534b13dc622448ed2255a97234c103af"


QUOTAS = {
    "bc": 128,
    "alakazam_control": 48,
    "mega_froslass_lopunny": 64,
    "mega_lopunny": 64,
    "mega_lucario": 48,
    "teal_mask_ogerpon": 96,
    "dragapult_ex": 64,
    "mega_kangaskhan_crustle_current": 80,
    "cynthias_garchomp_ex": 48,
}

LOSS_WEIGHTS = {
    "bc": 0.5,
    "alakazam_control": 0.7,
    "mega_froslass_lopunny": 1.0,
    "mega_lopunny": 1.0,
    "mega_lucario": 0.7,
    "teal_mask_ogerpon": 1.2,
    "dragapult_ex": 1.0,
    "mega_kangaskhan_crustle_current": 1.2,
    "cynthias_garchomp_ex": 0.7,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    proxy_selection = json.loads(
        (PROXY_ROOT / "proxy_checkpoint_selection_compatible.json").read_text(encoding="utf-8")
    )
    for path in (PYTHON, TRAINER, PARENT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sum(QUOTAS.values()) != 640 or any(value % 2 for value in QUOTAS.values()):
        raise RuntimeError("PPO quotas must sum to 640 and remain exactly seat-balanced")
    if set(LOSS_WEIGHTS) != set(QUOTAS):
        raise RuntimeError("Loss-weight and quota groups differ")

    learner_deck = Path(manifest["profiles"]["marnie"]["deck"])
    opponents: dict[str, dict[str, str]] = {}
    command = [
        str(PYTHON), "-I", "-B", str(TRAINER),
        "--bc-checkpoint", str(PARENT),
        "--kl-reference-checkpoint", str(PARENT),
        "--deck", str(learner_deck),
    ]
    for slug in QUOTAS:
        if slug == "bc":
            continue
        checkpoint = Path(proxy_selection["selections"][slug]["selected_checkpoint"])
        deck = Path(manifest["profiles"][slug]["deck"])
        for path in (checkpoint, deck):
            if not path.is_file():
                raise FileNotFoundError(path)
        name = f"{checkpoint.stem}@{deck.stem}"
        opponents[slug] = {
            "name": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "deck": str(deck),
            "deck_sha256": sha256(deck),
        }
        command.extend(["--extra-opponent", str(checkpoint), str(deck)])

    primary = opponents["teal_mask_ogerpon"]["name"]
    guard = opponents["mega_kangaskhan_crustle_current"]["name"]
    resolved_names = {"bc": "bc"} | {slug: row["name"] for slug, row in opponents.items()}
    command.extend([
        "--output-dir", str(output),
        "--updates", "1",
        "--schedule-start-update", "1",
        "--environments", "32",
        "--games-per-update", "640",
        "--ppo-epochs", "1",
        "--minibatch-size", "1024",
        "--learning-rate", "0.000002",
        "--value-learning-rate", "0.000002",
        "--weight-decay", "0.0",
        "--gamma", "1.0",
        "--gae-lambda", "1.0",
        "--advantage-normalization", "per_opponent",
        "--clip-ratio", "0.08",
        "--value-coefficient", "0.25",
        "--entropy-coefficient", "0.0",
        "--max-grad-norm", "0.20",
        "--policy-temperature", "0.8",
        "--trainable-scope", "heads",
        "--learning-rate-schedule", "constant",
        "--bc-kl-start", "0.04",
        "--bc-kl-end", "0.04",
        "--target-kl", "0.00005",
        "--league-probability", "1.0",
        "--opponent-sampling", "per_game",
        "--history-opponent-weight", "0.0",
        "--opponent-quota-mode", "fixed",
        "--opponent-quota-seat-balance",
        "--ppo-objective", "constrained",
        "--constrained-gradient-mode", "scalar",
        "--primary-opponent-name", primary,
        "--guard-opponent-name", guard,
        "--primary-policy-weight", "1.0",
        "--guard-policy-weight", "0.75",
        "--auxiliary-policy-weight", "0.35",
        "--guard-surrogate-floor", "0.0",
        "--constraint-dual-initial", "1.0",
        "--constraint-dual-lr", "0.02",
        "--constraint-dual-max", "5.0",
        "--snapshot-interval", "1000000",
        "--max-pool-size", "12",
        "--eval-interval", "1",
        "--eval-games", "64",
        "--eval-all-permanent-opponents",
        "--selection-aggregation", "min",
        "--checkpoint-interval", "1",
        "--max-game-decisions", "1000",
        "--seed", "2026084206",
        "--device", "cuda",
        "--failed-attempt-as-loss",
        "--failed-loss-tail-transitions", "32",
    ])
    for slug, quota in QUOTAS.items():
        command.extend(["--opponent-base-quota", resolved_names[slug], str(quota)])
    for slug, weight in LOSS_WEIGHTS.items():
        command.extend(["--opponent-loss-weight", resolved_names[slug], str(weight)])

    plan = {
        "schema_version": "ptcg-top100-proxy-guarded-ppo-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute": args.execute,
        "parent": {"path": str(PARENT), "sha256": sha256(PARENT)},
        "learner_deck": {"path": str(learner_deck), "sha256": sha256(learner_deck), "deck_hash": MARNIE_HASH},
        "opponents": opponents,
        "quotas": QUOTAS,
        "resolved_names": resolved_names,
        "primary": primary,
        "guard": guard,
        "command": command,
        "contracts": {
            "terminal_reward_source": "official engine terminal outcome",
            "win_target": 1,
            "loss_target": 0,
            "draw_target": 0,
            "exact_seat_balance": True,
            "single_update": True,
            "submission": False,
        },
    }
    plan_path = output.parent / f"{output.name}.plan.json"
    if plan_path.exists():
        raise FileExistsError(plan_path)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(shlex.join(command), flush=True)
    print(json.dumps({"plan": str(plan_path), "execute": args.execute}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
