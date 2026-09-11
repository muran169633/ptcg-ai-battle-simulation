#!/usr/bin/env python3
"""Validate an extracted PTCG BC/PPO bundle against its source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import torch

from train_ppo import (
    BC_FEATURE_VERSION,
    RawBattle,
    checkpoint_model_config,
    collate_features,
    compute_deck_hash,
    instantiate_model_from_checkpoint,
    live_feature,
    model_forward,
    read_deck,
    sample_ordered_actions,
)
from evaluate_ppo_head_to_head import apply_hybrid_action_order


EXPECTED_MEMBERS = ("main.py", "deck.csv", "model.pt", "policy_runtime.py")
DEPLOYMENT_SCHEMA = "ptcg-gold-push-deployment-contract-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_deployment_contract(
    path: Path,
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    deck_file_sha256: str,
    deck_hash: str,
    action_order_mode: str,
) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != DEPLOYMENT_SCHEMA:
        raise ValueError("Unexpected deployment contract schema")
    candidate = value.get("candidate") or {}
    deck = value.get("candidate_deck") or {}
    action_order = value.get("action_order") or {}
    if not all(isinstance(row, dict) for row in (candidate, deck, action_order)):
        raise ValueError("Deployment contract sections are malformed")
    expected = {
        "candidate.path": (
            Path(str(candidate.get("path"))).expanduser().resolve(),
            checkpoint.resolve(),
        ),
        "candidate.sha256": (candidate.get("sha256"), checkpoint_sha256),
        "candidate_deck.file_sha256": (
            deck.get("file_sha256"),
            deck_file_sha256,
        ),
        "candidate_deck.semantic_hash": (
            deck.get("semantic_hash"),
            deck_hash,
        ),
        "action_order.mode": (
            action_order.get("mode"),
            action_order_mode,
        ),
        "action_order.canonical_order": (
            action_order.get("canonical_order"),
            action_order_mode == "canonical",
        ),
        "action_order.hybrid_order": (
            action_order.get("hybrid_order"),
            action_order_mode == "hybrid",
        ),
    }
    mismatches = {
        label: {"actual": actual, "expected": wanted}
        for label, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise ValueError(f"Deployment contract mismatch: {mismatches}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("artifacts/bc_marnie_luca_orbit_v5/best.pt"),
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument(
        "--action-order-mode",
        choices=("raw", "canonical", "hybrid"),
        required=True,
    )
    parser.add_argument("--deployment-contract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (args.archive, args.checkpoint, args.bc_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix="ptcg-submit-validate-") as temporary:
        root = Path(temporary)
        with tarfile.open(args.archive, mode="r:gz") as archive:
            members = tuple(member.name for member in archive.getmembers())
            if members != EXPECTED_MEMBERS:
                raise RuntimeError(f"Unexpected archive members: {members}")
            archive.extractall(root, filter="data")

        sys.path.insert(0, str(root))
        sys.modules.pop("policy_runtime", None)
        namespace: dict[str, object] = {}
        main_path = root / "main.py"
        exec(
            compile(main_path.read_text(encoding="utf-8"), str(main_path), "exec"),
            namespace,
        )
        packaged_agent = namespace["agent"]
        if not callable(packaged_agent):
            raise TypeError("Bundle did not expose callable agent()")

        deck = read_deck(root / "deck.csv")
        deck_hash = compute_deck_hash(deck)
        deck_file_sha = sha256_file(root / "deck.csv")
        deployment_contract: dict[str, object] | None = None
        if args.deployment_contract is not None:
            if not args.deployment_contract.is_file():
                raise FileNotFoundError(args.deployment_contract)
            deployment_contract = validate_deployment_contract(
                args.deployment_contract,
                checkpoint=args.checkpoint,
                checkpoint_sha256=sha256_file(args.checkpoint),
                deck_file_sha256=deck_file_sha,
                deck_hash=deck_hash,
                action_order_mode=args.action_order_mode,
            )
        if packaged_agent({"select": None}) != deck:
            raise RuntimeError("Initial packaged agent deck response is incorrect")

        device = torch.device("cpu")
        bc_checkpoint = torch.load(
            args.bc_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if bc_checkpoint.get("feature_version") != BC_FEATURE_VERSION:
            raise ValueError("Unexpected BC anchor feature version")
        source_checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model_config = checkpoint_model_config(bc_checkpoint)
        source_model = instantiate_model_from_checkpoint(
            source_checkpoint,
            bc_checkpoint,
            device,
        )

        stats = {
            "games_requested": args.games,
            "valid_games": 0,
            "invalid_games": 0,
            "engine_decisions": 0,
            "exact_action_matches": 0,
            "action_mismatches": 0,
            "action_order_mode": args.action_order_mode,
            "deployment_contract": (
                {
                    "path": str(args.deployment_contract.resolve()),
                    "sha256": sha256_file(args.deployment_contract),
                    "schema_version": deployment_contract.get("schema_version"),
                }
                if args.deployment_contract is not None
                and deployment_contract is not None
                else None
            ),
            "packaged_agent_seconds": 0.0,
        }
        for _ in range(args.games):
            battle = RawBattle(deck, deck)
            valid = True
            game_decisions = 0
            try:
                while battle.result == -1:
                    observation = battle.observation
                    started = time.perf_counter()
                    packaged_action = packaged_agent(observation)
                    stats["packaged_agent_seconds"] += time.perf_counter() - started

                    feature = live_feature(
                        observation,
                        model_config,
                        deck_hash,
                    )
                    if feature is None:
                        valid = False
                        break
                    batch = collate_features(
                        [feature],
                        model_config,
                        device,
                    )
                    with torch.inference_mode():
                        outputs = model_forward(source_model, batch, device)
                        source_actions, _, _, _ = sample_ordered_actions(
                            outputs,
                            batch,
                            deterministic=True,
                            canonicalize_order=(
                                args.action_order_mode == "canonical"
                            ),
                        )
                    source_action = source_actions[0]
                    if args.action_order_mode == "hybrid":
                        select = observation.get("select") or {}
                        source_action = apply_hybrid_action_order(
                            source_action,
                            select_context=int(
                                select.get("context", -1) or -1
                            ),
                            enabled=True,
                        )
                    if packaged_action != source_action:
                        stats["action_mismatches"] += 1
                        valid = False
                        break
                    stats["exact_action_matches"] += 1

                    _, error = battle.step(packaged_action)
                    stats["engine_decisions"] += 1
                    game_decisions += 1
                    if error:
                        valid = False
                        break
                    if (
                        battle.result == -1
                        and game_decisions >= args.max_game_decisions
                    ):
                        valid = False
                        break
            finally:
                battle.close()
            if valid:
                stats["valid_games"] += 1
            else:
                stats["invalid_games"] += 1

        decisions = max(stats["engine_decisions"], 1)
        stats["mean_packaged_agent_ms"] = (
            stats["packaged_agent_seconds"] * 1000.0 / decisions
        )
        stats["passed"] = (
            stats["valid_games"] == args.games
            and stats["invalid_games"] == 0
            and stats["action_mismatches"] == 0
            and stats["exact_action_matches"] == stats["engine_decisions"]
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
