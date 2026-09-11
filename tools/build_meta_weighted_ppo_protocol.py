#!/usr/bin/env python3
"""Bind a recent-day meta pool to reproducible PPO opponent flags."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from train_ppo import compute_deck_hash, read_deck  # noqa: E402


SCHEMA_VERSION = "ptcg-meta-weighted-ppo-protocol-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_protocol(
    meta_pool_path: Path,
    learner_deck_path: Path,
    opponent_checkpoint_path: Path,
    *,
    league_probability: float = 0.90,
    window_games: int = 200,
    inverse_min_factor: float = 0.5,
    inverse_max_factor: float = 2.5,
) -> dict[str, Any]:
    meta_pool_path = meta_pool_path.resolve()
    learner_deck_path = learner_deck_path.resolve()
    opponent_checkpoint_path = opponent_checkpoint_path.resolve()
    for path in (
        meta_pool_path,
        learner_deck_path,
        opponent_checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 < league_probability <= 1.0:
        raise ValueError("league_probability must be in (0, 1]")
    if window_games < 1:
        raise ValueError("window_games must be positive")
    if (
        not math.isfinite(inverse_min_factor)
        or not math.isfinite(inverse_max_factor)
        or inverse_min_factor <= 0.0
        or inverse_max_factor <= 0.0
        or inverse_min_factor > inverse_max_factor
    ):
        raise ValueError("Invalid inverse factor bounds")

    meta = json.loads(meta_pool_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != "ptcg-recent-day-exact-deck-meta-pool-v1":
        raise ValueError("Unexpected meta pool schema")
    opponents = meta.get("opponents")
    if not isinstance(opponents, list) or not opponents:
        raise ValueError("Meta pool has no opponents")
    learner_hash = compute_deck_hash(read_deck(learner_deck_path))
    learner_matches = [
        row for row in opponents if row.get("deck_hash") == learner_hash
    ]
    if len(learner_matches) != 1:
        raise ValueError(
            f"Learner deck must occur exactly once in selected meta pool; "
            f"found {len(learner_matches)}"
        )

    flags = [
        "--league-probability",
        f"{league_probability:.12g}",
        "--opponent-sampling",
        "per_game",
        "--opponent-win-rate-window-games",
        str(window_games),
        "--opponent-inverse-min-factor",
        f"{inverse_min_factor:.12g}",
        "--opponent-inverse-max-factor",
        f"{inverse_max_factor:.12g}",
        "--max-pool-size",
        str(len(opponents)),
    ]
    bindings: list[dict[str, Any]] = []
    for row in opponents:
        deck_hash = str(row["deck_hash"])
        weight = float(row["pool_base_probability"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Invalid pool weight for {deck_hash}")
        deck_path = Path(str(row["deck_path"])).resolve()
        if not deck_path.is_file():
            raise FileNotFoundError(deck_path)
        observed_hash = compute_deck_hash(read_deck(deck_path))
        if observed_hash != deck_hash:
            raise ValueError(f"Meta deck hash mismatch for {deck_path}")
        if deck_hash == learner_hash:
            name = "bc"
            role = "base_frozen_same_deck"
        else:
            name = f"{opponent_checkpoint_path.stem}@{deck_path.stem}"
            role = "extra_recent_meta_deck"
            flags.extend(
                [
                    "--extra-opponent",
                    str(opponent_checkpoint_path),
                    str(deck_path),
                ]
            )
        flags.extend(["--opponent-meta-weight", name, f"{weight:.17g}"])
        bindings.append(
            {
                "name": name,
                "role": role,
                "rank": int(row["rank"]),
                "deck_hash": deck_hash,
                "deck_path": str(deck_path),
                "deck_sha256": sha256_file(deck_path),
                "meta_appearances": int(row["appearances"]),
                "meta_probability": weight,
                "opponent_checkpoint": (
                    None
                    if role == "base_frozen_same_deck"
                    else str(opponent_checkpoint_path)
                ),
            }
        )

    probability_sum = sum(row["meta_probability"] for row in bindings)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Meta probabilities do not sum to one: {probability_sum}")
    return {
        "schema_version": SCHEMA_VERSION,
        "meta_pool": {
            "path": str(meta_pool_path),
            "sha256": sha256_file(meta_pool_path),
            "dataset_date": meta.get("dataset_date"),
            "coverage": (meta.get("selection") or {}).get("coverage"),
        },
        "learner_deck": {
            "path": str(learner_deck_path),
            "sha256": sha256_file(learner_deck_path),
            "semantic_hash": learner_hash,
        },
        "generic_opponent_checkpoint": {
            "path": str(opponent_checkpoint_path),
            "sha256": sha256_file(opponent_checkpoint_path),
        },
        "sampling": {
            "selfplay_probability": round(1.0 - league_probability, 12),
            "frozen_opponent_probability": league_probability,
            "conditional_base_distribution": "recent_day_exact_deck_frequency",
            "dynamic_adjustment": "inverse_beta22_smoothed_rolling_win_rate",
            "window_games_per_opponent": window_games,
            "inverse_min_factor": inverse_min_factor,
            "inverse_max_factor": inverse_max_factor,
            "probability_sum": probability_sum,
        },
        "bindings": bindings,
        "train_ppo_argument_fragment": flags,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-pool", type=Path, required=True)
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--league-probability", type=float, default=0.90)
    parser.add_argument("--window-games", type=int, default=200)
    parser.add_argument("--inverse-min-factor", type=float, default=0.5)
    parser.add_argument("--inverse-max-factor", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_protocol(
        args.meta_pool,
        args.learner_deck,
        args.opponent_checkpoint,
        league_probability=args.league_probability,
        window_games=args.window_games,
        inverse_min_factor=args.inverse_min_factor,
        inverse_max_factor=args.inverse_max_factor,
    )
    atomic_write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bindings": len(payload["bindings"]),
                "selfplay_probability": payload["sampling"][
                    "selfplay_probability"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
