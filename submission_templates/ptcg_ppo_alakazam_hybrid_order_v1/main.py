"""Kaggle inference entrypoint for the Alakazam hybrid-order PPO policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import policy_runtime as _runtime
from policy_runtime import EntityOptionPolicy, collate_decisions, featurize_row


# Kaggle executes main.py with exec() and does not inject __file__. Imported
# sibling modules do receive it, so use policy_runtime as the bundle anchor.
AGENT_DIR = Path(_runtime.__file__).resolve().parent
MODEL_PATH = AGENT_DIR / "model.pt"
DECK_PATH = AGENT_DIR / "deck.csv"
DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
MAX_ACTION_COUNT = 60
SKILL_ORDER_CONTEXT = 34


def _configure_cpu() -> None:
    # Kaggle agents receive a small CPU allocation. A single inference request
    # is faster and more predictable without large OpenMP thread pools.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits changing this before parallel work starts.
        pass


_configure_cpu()


def _read_deck() -> list[int]:
    deck = [
        int(line.strip())
        for line in DECK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 cards, found {len(deck)}")
    return deck


DECK = _read_deck()
_MODEL: EntityOptionPolicy | None = None
_MODEL_CONFIG: dict[str, Any] | None = None


def _load_checkpoint() -> dict[str, Any]:
    try:
        return torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch builds.
        return torch.load(MODEL_PATH, map_location="cpu")


def _load_model() -> tuple[EntityOptionPolicy, dict[str, Any]]:
    global _MODEL, _MODEL_CONFIG
    if _MODEL is not None and _MODEL_CONFIG is not None:
        return _MODEL, _MODEL_CONFIG

    checkpoint = _load_checkpoint()
    if checkpoint.get("feature_version") != "ptcg-selfplay-ppo-terminal01-v1":
        raise ValueError("Unexpected PPO feature version")
    config = dict(checkpoint["model_config"])
    model = EntityOptionPolicy(
        hash_size=int(config["hash_size"]),
        categorical_dim=int(config["categorical_dim"]),
        model_dim=int(config["model_dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        max_state_entities=int(config["max_state_entities"]),
    )

    old_count_layer = model.count_head[-1]
    if not isinstance(old_count_layer, nn.Linear):
        raise TypeError("Unexpected count head")
    if old_count_layer.out_features != MAX_ACTION_COUNT + 1:
        model.count_head[-1] = nn.Linear(
            old_count_layer.in_features,
            MAX_ACTION_COUNT + 1,
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)

    _MODEL = model
    _MODEL_CONFIG = config
    return model, config


def _legal_bounds(select: dict[str, Any]) -> tuple[int, int, int]:
    options = select.get("option")
    if not isinstance(options, list):
        raise TypeError("select.option must be a list")
    option_count = len(options)
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    if (
        minimum < 0
        or maximum < minimum
        or maximum > option_count
        or maximum > MAX_ACTION_COUNT
    ):
        raise ValueError(
            f"Invalid legal bounds min={minimum} max={maximum} "
            f"options={option_count}"
        )
    return minimum, maximum, option_count


def _apply_hybrid_order(action: list[int], select_context: int) -> list[int]:
    """Preserve SKILL_ORDER semantics and canonicalize set-like contexts."""
    if select_context == SKILL_ORDER_CONTEXT:
        return action
    return sorted(action)


def _policy_action(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, dict):
        raise TypeError("Missing select object")
    minimum, maximum, option_count = _legal_bounds(select)
    if option_count == 0:
        return []

    model, config = _load_model()
    current = observation.get("current") or {}
    feature = featurize_row(
        {
            "observation": observation,
            "seat": int(current.get("yourIndex", 0) or 0),
            "deck_hash": DECK_HASH,
            "team_name": "",
            "action": [],
            "terminal_reward": 0.0,
            "sample_weight": 1.0,
        },
        hash_size=int(config["hash_size"]),
        max_state_entities=int(config["max_state_entities"]),
    )
    if feature is None:
        # The official engine should not reach this branch. Returning the
        # minimum legal prefix keeps the agent valid if a zero-information
        # selection is introduced.
        return list(range(minimum))

    batch = collate_decisions(
        [feature],
        max_state_entities=int(config["max_state_entities"]),
        entity_fields=int(config["entity_fields"]),
        option_fields=int(config["option_fields"]),
    )
    with torch.inference_mode():
        outputs = model(batch)

    if minimum == maximum:
        count = minimum
    else:
        count_logits = outputs["count_logits"][0].float()
        values = torch.arange(count_logits.numel())
        allowed = (values >= minimum) & (values <= maximum)
        count = int(count_logits.masked_fill(~allowed, -1e9).argmax())

    policy_logits = outputs["policy_logits"][0, :option_count].float()
    remaining = torch.ones(option_count, dtype=torch.bool)
    action: list[int] = []
    for _ in range(count):
        chosen = int(policy_logits.masked_fill(~remaining, -1e9).argmax())
        action.append(chosen)
        remaining[chosen] = False
    action = _apply_hybrid_order(
        action,
        int(select.get("context", -1) or -1),
    )
    if (
        len(action) < minimum
        or len(action) > maximum
        or len(action) != len(set(action))
        or any(index < 0 or index >= option_count for index in action)
    ):
        raise RuntimeError("Policy produced an illegal action")
    return action


def agent(obs_dict: dict[str, Any]) -> list[int]:
    """Return the deck initially, then hybrid-ordered PPO option indices."""
    if obs_dict.get("select") is None:
        return list(DECK)
    return _policy_action(obs_dict)
