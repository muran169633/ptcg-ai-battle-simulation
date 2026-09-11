#!/usr/bin/env python3
"""Verify a packaged mode-aware agent against its source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from evaluate_mode_ar_vs_submission import mode_ar_forward
from train_mode_ar_ppo import load_anchor
from train_ppo import RawBattle, collate_features, compute_deck_hash, live_feature, read_deck


EXPECTED_MEMBERS = ("main.py", "deck.csv", "model.pt", "policy_runtime.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.archive, args.checkpoint, args.deck):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.games < 1 or args.max_game_decisions < 1:
        raise ValueError("games and max-game-decisions must be positive")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    deck = read_deck(args.deck)
    deck_hash = compute_deck_hash(deck)
    checkpoint, source_model, config = load_anchor(args.checkpoint, device)
    if checkpoint.get("learner_deck_hash") != deck_hash:
        raise ValueError("Source checkpoint and requested deck do not match")

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="ptcg-mode-ar-submit-") as temporary:
        root = Path(temporary)
        with tarfile.open(args.archive, "r:gz") as tar:
            members = tuple(member.name for member in tar.getmembers())
            if members != EXPECTED_MEMBERS:
                raise RuntimeError(f"Unexpected archive members: {members}")
            tar.extractall(root, filter="data")
        if (root / "deck.csv").read_bytes() != args.deck.read_bytes():
            raise RuntimeError("Packaged deck order differs from the requested deck")

        sys.path.insert(0, str(root))
        sys.modules.pop("policy_runtime", None)
        namespace: dict[str, Any] = {}
        main_path = root / "main.py"
        exec(compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec"), namespace)
        packaged_agent = namespace.get("agent")
        if not callable(packaged_agent):
            raise TypeError("Bundle does not expose agent()")
        if packaged_agent({"select": None}) != deck:
            raise RuntimeError("Packaged initial deck response is incorrect")

        stats: dict[str, Any] = {
            "schema_version": "ptcg-mode-ar-runtime-validation-v1",
            "games_requested": args.games,
            "valid_games": 0,
            "invalid_games": 0,
            "engine_decisions": 0,
            "exact_action_matches": 0,
            "action_mismatches": 0,
            "deck_order_exact": True,
            "source_update": checkpoint.get("update"),
            "source_checkpoint_sha256": sha256_file(args.checkpoint),
            "archive_sha256": sha256_file(args.archive),
        }
        for _ in range(args.games):
            battle = RawBattle(deck, deck)
            valid = True
            game_decisions = 0
            try:
                while battle.result == -1:
                    observation = battle.observation
                    packaged_action = packaged_agent(observation)
                    feature = live_feature(observation, config, deck_hash)
                    if feature is None:
                        valid = False
                        break
                    batch = collate_features([feature], config, device)
                    source_action = mode_ar_forward(source_model, batch, device)["actions"][0]
                    if packaged_action != source_action:
                        stats["action_mismatches"] += 1
                        valid = False
                        break
                    stats["exact_action_matches"] += 1
                    _, error = battle.step(packaged_action)
                    stats["engine_decisions"] += 1
                    game_decisions += 1
                    if error or (battle.result == -1 and game_decisions >= args.max_game_decisions):
                        valid = False
                        break
                stats["valid_games" if valid else "invalid_games"] += 1
            finally:
                battle.close()

        stats["elapsed_seconds"] = time.time() - started
        stats["passed"] = (
            stats["valid_games"] == args.games
            and stats["invalid_games"] == 0
            and stats["action_mismatches"] == 0
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if not stats["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
