#!/usr/bin/env python3
"""Validate a BC submission archive, source tensors, and engine legality."""

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

from train_ppo import RawBattle, compute_deck_hash, read_deck


EXPECTED_MEMBERS = ("main.py", "deck.csv", "model.pt", "policy_runtime.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.archive, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.games < 1 or args.max_game_decisions < 1:
        raise ValueError("games and max-game-decisions must be positive")

    with tempfile.TemporaryDirectory(prefix="ptcg-bc-submit-validate-") as temporary:
        root = Path(temporary)
        with tarfile.open(args.archive, mode="r:gz") as archive:
            members = tuple(member.name for member in archive.getmembers())
            if members != EXPECTED_MEMBERS:
                raise RuntimeError(f"Unexpected archive members: {members}")
            if any(not member.isfile() for member in archive.getmembers()):
                raise RuntimeError("Archive contains a non-file member")
            archive.extractall(root, filter="data")

        source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        bundled = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        source_state = source.get("model_state_dict")
        bundled_state = bundled.get("model_state_dict")
        if not isinstance(source_state, dict) or not isinstance(bundled_state, dict):
            raise ValueError("Source or bundled checkpoint has no model state")
        tensor_keys_equal = source_state.keys() == bundled_state.keys()
        unequal_tensors = (
            []
            if not tensor_keys_equal
            else [
                name
                for name in source_state
                if not torch.equal(source_state[name], bundled_state[name])
            ]
        )
        feature_version_equal = (
            source.get("feature_version") == bundled.get("feature_version")
        )

        sys.path.insert(0, str(root))
        sys.modules.pop("policy_runtime", None)
        namespace: dict[str, Any] = {}
        main_path = root / "main.py"
        exec(
            compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec"),
            namespace,
        )
        packaged_agent = namespace.get("agent")
        if not callable(packaged_agent):
            raise TypeError("Bundle did not expose callable agent()")

        deck = read_deck(root / "deck.csv")
        if packaged_agent({"select": None}) != deck:
            raise RuntimeError("Initial packaged agent deck response is incorrect")

        stats: dict[str, Any] = {
            "archive": str(args.archive.resolve()),
            "archive_sha256": sha256_file(args.archive),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "feature_version": source.get("feature_version"),
            "feature_version_equal": feature_version_equal,
            "model_tensor_count": len(source_state),
            "model_tensor_keys_equal": tensor_keys_equal,
            "unequal_tensors": unequal_tensors,
            "deck_hash": compute_deck_hash(deck),
            "members": list(EXPECTED_MEMBERS),
            "games_requested": args.games,
            "valid_games": 0,
            "invalid_games": 0,
            "engine_decisions": 0,
            "packaged_agent_seconds": 0.0,
        }
        for _ in range(args.games):
            battle = RawBattle(deck, deck)
            valid = True
            game_decisions = 0
            try:
                while battle.result == -1:
                    started = time.perf_counter()
                    action = packaged_agent(battle.observation)
                    stats["packaged_agent_seconds"] += time.perf_counter() - started
                    _, error = battle.step(action)
                    stats["engine_decisions"] += 1
                    game_decisions += 1
                    if error or (
                        battle.result == -1
                        and game_decisions >= args.max_game_decisions
                    ):
                        valid = False
                        break
            finally:
                battle.close()
            stats["valid_games" if valid else "invalid_games"] += 1

        stats["mean_packaged_agent_ms"] = (
            stats["packaged_agent_seconds"]
            * 1000.0
            / max(stats["engine_decisions"], 1)
        )
        stats["passed"] = (
            feature_version_equal
            and tensor_keys_equal
            and not unequal_tensors
            and stats["valid_games"] == args.games
            and stats["invalid_games"] == 0
        )
        rendered = json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
        if args.output is not None:
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        if not stats["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
