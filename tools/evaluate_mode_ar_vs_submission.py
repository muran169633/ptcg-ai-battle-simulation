#!/usr/bin/env python3
"""Strict same-deck H2H: Mode-aware AR BC versus a submitted PPO policy.

The official-engine loop lives in ``evaluate_ppo_head_to_head.py``.  This
adapter changes only the candidate inference route so the newer pointer
decoder can be evaluated under the same exact-valid-game and seat-balance
contract as the deployed legacy policy.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

import evaluate_ppo_head_to_head as legacy
from train_bc_mode_ar_v7 import (
    FEATURE_VERSION,
    ModeAwareARPolicy,
    greedy_decode,
)


MODE_AR_MARKER = "__ptcg_mode_ar_actions_v1__"


def load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: actual={actual!r} expected={expected!r}")


def mode_ar_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Mode-aware checkpoint is missing config")
    required = (
        "hash_size",
        "categorical_dim",
        "model_dim",
        "layers",
        "heads",
        "dropout",
        "max_state_entities",
        "entity_fields",
        "option_fields",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Mode-aware checkpoint config is missing {missing}")
    return {
        "hash_size": int(config["hash_size"]),
        "categorical_dim": int(config["categorical_dim"]),
        "model_dim": int(config["model_dim"]),
        "layers": int(config["layers"]),
        "heads": int(config["heads"]),
        "dropout": float(config["dropout"]),
        "max_state_entities": int(config["max_state_entities"]),
        "entity_fields": int(config["entity_fields"]),
        "option_fields": int(config["option_fields"]),
    }


def load_mode_ar_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], ModeAwareARPolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint mapping")
    require_equal("candidate feature version", checkpoint.get("feature_version"), FEATURE_VERSION)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path} is missing model_state_dict")
    option_positions = state.get("option_position.weight")
    if not isinstance(option_positions, torch.Tensor) or option_positions.ndim != 2:
        raise ValueError(f"{path} has an invalid option position table")
    config = mode_ar_model_config(checkpoint)
    model = ModeAwareARPolicy(
        hash_size=config["hash_size"],
        categorical_dim=config["categorical_dim"],
        model_dim=config["model_dim"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
        max_state_entities=config["max_state_entities"],
        max_options=int(option_positions.shape[0]),
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    return checkpoint, model, config


@torch.no_grad()
def mode_ar_forward(
    model: ModeAwareARPolicy,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    with torch.amp.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        encoded = model(batch)
        decoded = greedy_decode(model, encoded, batch)
    sequences = decoded["sequences"].detach().cpu()
    counts = decoded["counts"].detach().cpu()
    actions: list[list[int]] = []
    for row, count in zip(sequences, counts):
        action = [int(value) for value in row.tolist() if int(value) >= 0]
        if len(action) != int(count):
            raise RuntimeError("Mode-aware decoded sequence/count mismatch")
        if len(action) != len(set(action)):
            raise RuntimeError("Mode-aware decoder returned duplicate options")
        actions.append(action)
    return {
        MODE_AR_MARKER: True,
        "actions": actions,
        "counts": counts,
    }


@contextlib.contextmanager
def install_mode_ar_inference_adapter() -> Iterator[None]:
    """Temporarily dispatch the candidate through its pointer decoder."""

    original_forward = legacy.model_forward
    original_sample = legacy.sample_ordered_actions

    def forward_dispatch(
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(model, ModeAwareARPolicy):
            if args or kwargs:
                raise TypeError("Mode-aware evaluator received unexpected forward arguments")
            return mode_ar_forward(model, batch, device)
        return original_forward(model, batch, device, *args, **kwargs)

    def sample_dispatch(
        outputs: Any,
        batch: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(outputs, dict) and outputs.get(MODE_AR_MARKER) is True:
            if not bool(kwargs.get("deterministic", False)):
                raise ValueError("Mode-aware H2H requires deterministic decoding")
            if bool(kwargs.get("canonicalize_order", False)):
                raise ValueError("Do not post-sort mode-aware decoded actions")
            actions = outputs["actions"]
            zeros = torch.zeros(len(actions), device=batch["targets"].device)
            return actions, zeros, zeros, zeros
        return original_sample(outputs, batch, *args, **kwargs)

    legacy.model_forward = forward_dispatch
    legacy.sample_ordered_actions = sample_dispatch
    try:
        yield
    finally:
        legacy.model_forward = original_forward
        legacy.sample_ordered_actions = original_sample


def audit_completed_training(
    checkpoint: dict[str, Any],
    metrics_path: Path,
) -> dict[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("candidate checkpoint has no config")
    expected_epochs = int(config.get("epochs", 0))
    if expected_epochs < 1:
        raise ValueError("candidate checkpoint has invalid configured epoch count")
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [int(row["epoch"]) for row in rows]
    require_equal("completed training epochs", epochs, list(range(1, expected_epochs + 1)))
    best_epoch = int(checkpoint.get("epoch", 0))
    if best_epoch not in epochs:
        raise ValueError("best checkpoint epoch is absent from completed metrics")
    best_row = rows[best_epoch - 1]
    checkpoint_metrics = checkpoint.get("valid_metrics")
    if not isinstance(checkpoint_metrics, dict):
        raise ValueError("candidate best checkpoint is missing validation metrics")
    require_equal(
        "best daily macro choice accuracy",
        checkpoint_metrics.get("daily_macro_choice_accuracy"),
        best_row["valid"].get("daily_macro_choice_accuracy"),
    )
    dates = [str(value) for value in config.get("dates", ())]
    if "2026-08-13" not in dates:
        raise ValueError("candidate training dates do not include 2026-08-13")
    if config.get("daily_equal_weighting") is not True:
        raise ValueError("candidate was not trained with daily equal weighting")
    return {
        "configured_epochs": expected_epochs,
        "completed_epochs": len(epochs),
        "best_epoch": best_epoch,
        "best_daily_macro_choice_accuracy": checkpoint_metrics.get(
            "daily_macro_choice_accuracy"
        ),
        "dates": dates,
        "daily_equal_weighting": True,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": legacy.sha256_file(metrics_path),
    }


def audit_submitted_baseline(
    *,
    checkpoint: Path,
    deck_path: Path,
    deck_hash: str,
    archive: Path,
    manifest_path: Path,
    contract_path: Path,
    validation_path: Path,
) -> dict[str, Any]:
    manifest = load_json_mapping(manifest_path)
    contract = load_json_mapping(contract_path)
    validation = load_json_mapping(validation_path)
    checkpoint_sha = legacy.sha256_file(checkpoint)
    deck_sha = legacy.sha256_file(deck_path)
    archive_sha = legacy.sha256_file(archive)
    require_equal("submitted checkpoint SHA256", checkpoint_sha, manifest.get("source_checkpoint_sha256"))
    require_equal("submitted archive SHA256", archive_sha, manifest.get("archive_sha256"))
    require_equal(
        "submitted deck file SHA256",
        deck_sha,
        (manifest.get("template_sources") or {}).get("deck.csv", {}).get("sha256"),
    )
    require_equal("submitted semantic deck hash", deck_hash, manifest.get("semantic_deck_hash"))
    require_equal("submitted action order", manifest.get("action_order_mode"), "raw")
    require_equal("deployment checkpoint SHA256", checkpoint_sha, (contract.get("candidate") or {}).get("sha256"))
    require_equal("deployment semantic deck hash", deck_hash, (contract.get("candidate_deck") or {}).get("semantic_hash"))
    require_equal("deployment action order", (contract.get("action_order") or {}).get("mode"), "raw")
    require_equal("packaged runtime validation", validation.get("passed"), True)
    require_equal("packaged runtime mismatches", validation.get("action_mismatches"), 0)
    require_equal("packaged runtime invalid games", validation.get("invalid_games"), 0)
    return {
        "archive": str(archive.resolve()),
        "archive_sha256": archive_sha,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": legacy.sha256_file(manifest_path),
        "deployment_contract": str(contract_path.resolve()),
        "deployment_contract_sha256": legacy.sha256_file(contract_path),
        "packaged_runtime_validation": str(validation_path.resolve()),
        "packaged_runtime_validation_sha256": legacy.sha256_file(validation_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_update": manifest.get("source_update"),
        "deck_file_sha256": deck_sha,
        "semantic_deck_hash": deck_hash,
        "action_order_mode": "raw",
        "package_identity_verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--submitted-checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--submission-archive", type=Path, required=True)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--deployment-contract", type=Path, required=True)
    parser.add_argument("--submission-validation", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--environments", type=int, default=32)
    parser.add_argument("--max-game-decisions", type=int, default=1000)
    parser.add_argument("--loop-diagnostic-tail", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = (
        args.candidate,
        args.submitted_checkpoint,
        args.deck,
        args.submission_archive,
        args.submission_manifest,
        args.deployment_contract,
        args.submission_validation,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics_path = args.training_metrics or args.candidate.with_name("metrics.jsonl")
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if min(args.games, args.environments, args.max_game_decisions) < 1:
        raise ValueError("games, environments, and max-game-decisions must be positive")
    legacy.LoopDiagnosticCollector(args.loop_diagnostic_tail)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    deck = legacy.read_deck(args.deck)
    deck_hash = legacy.compute_deck_hash(deck)
    baseline_audit = audit_submitted_baseline(
        checkpoint=args.submitted_checkpoint,
        deck_path=args.deck,
        deck_hash=deck_hash,
        archive=args.submission_archive,
        manifest_path=args.submission_manifest,
        contract_path=args.deployment_contract,
        validation_path=args.submission_validation,
    )
    candidate_checkpoint, candidate_model, candidate_config = load_mode_ar_checkpoint(
        args.candidate, device
    )
    training_audit = audit_completed_training(candidate_checkpoint, metrics_path)
    opponent_checkpoint, opponent_model, opponent_config = legacy.load_checkpoint(
        args.submitted_checkpoint, deck_hash, args.deck, device
    )
    require_equal(
        "submitted update",
        opponent_checkpoint.get("update"),
        baseline_audit["source_update"],
    )

    started_at = datetime.now(timezone.utc)
    started = time.time()
    with install_mode_ar_inference_adapter():
        evaluation = legacy.evaluate_head_to_head_with_seats(
            candidate_model,
            opponent_model,
            deck,
            candidate_config,
            device,
            args.games,
            min(args.environments, args.games),
            args.max_game_decisions,
            current_canonical_order=False,
            opponent_canonical_order=False,
            opponent_deck=deck,
            loop_diagnostic_tail=args.loop_diagnostic_tail,
            current_hybrid_order=False,
            opponent_hybrid_order=False,
            opponent_model_config=opponent_config,
        )
    ci_low, ci_high = legacy.wilson_interval(
        int(evaluation["wins"]), int(evaluation["valid_games"])
    )
    evaluation.update(
        {
            "wilson_95_low": ci_low,
            "wilson_95_high": ci_high,
            "gate_ci_low_above_0_5": ci_low > 0.5,
        }
    )
    if int(evaluation["invalid_games"]) != 0:
        raise RuntimeError(
            f"strict H2H requires zero invalid games, got {evaluation['invalid_games']}"
        )
    result = {
        "schema_version": "ptcg-mode-ar-vs-submitted-h2h-v1",
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": legacy.sha256_file(args.candidate),
            "feature_version": candidate_checkpoint.get("feature_version"),
            "checkpoint_epoch": candidate_checkpoint.get("epoch"),
            "model_config": candidate_config,
            "decode_order": "mode-aware: raw AR for context 34, canonical set otherwise",
            "training_audit": training_audit,
        },
        "submitted_baseline": {
            "checkpoint": str(args.submitted_checkpoint.resolve()),
            "feature_version": opponent_checkpoint.get("feature_version"),
            "update": opponent_checkpoint.get("update"),
            "model_config": opponent_config,
            **baseline_audit,
        },
        "match": {
            "deck": str(args.deck.resolve()),
            "deck_hash": deck_hash,
            "same_deck_mirror": True,
            "exact_candidate_seat_balance": True,
            "games_requested": args.games,
            "environments": args.environments,
            "max_game_decisions": args.max_game_decisions,
            "engine_seed_control": False,
            "engine_seed_warning": (
                "Python/Torch seeds do not control official-engine RNG; seat quotas are exact but trials are not paired by engine seed."
            ),
            "python_torch_seed": args.seed,
            "device": str(device),
        },
        "evaluation": evaluation,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
