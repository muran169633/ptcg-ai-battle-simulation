"""Kaggle inference entrypoint for the frozen Alakazam standard PPO policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import policy_runtime as _runtime
from policy_runtime import EntityOptionPolicy, collate_decisions, featurize_row


TEMPLATE_VERSION = "ptcg-ppo-alakazam-standard-pl-v1"
PPO_FEATURE_VERSION = "ptcg-selfplay-ppo-terminal01-v1"
BC_FEATURE_VERSION = "ptcg-bc-orbit-entity-transformer-v5"
SOURCE_CHECKPOINT_SHA256 = (
    "bc3bbf185e4f61e4d3adc5dd11241f0f6001be274e90ef42e3f341f22b052393"
)
DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"
MAX_ACTION_COUNT = 60
# Kaggle executes main.py with exec() and does not inject __file__. Imported
# sibling modules do receive it, so use policy_runtime as the bundle anchor.
AGENT_DIR = Path(_runtime.__file__).resolve().parent
MODEL_PATH = AGENT_DIR / "model.pt"
DECK_PATH = AGENT_DIR / "deck.csv"


def _configure_cpu() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_configure_cpu()


def _compute_deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(deck))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_deck() -> list[int]:
    deck = [
        int(line.strip())
        for line in DECK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 cards, found {len(deck)}")
    actual_hash = _compute_deck_hash(deck)
    if actual_hash != DECK_HASH:
        raise ValueError(
            f"Unexpected Alakazam deck hash {actual_hash}; expected {DECK_HASH}"
        )
    return deck


DECK = _read_deck()
_MODEL: EntityOptionPolicy | None = None
_MODEL_CONFIG: dict[str, Any] | None = None


def _load_checkpoint() -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    return checkpoint


def _load_model() -> tuple[EntityOptionPolicy, dict[str, Any]]:
    global _MODEL, _MODEL_CONFIG
    if _MODEL is not None and _MODEL_CONFIG is not None:
        return _MODEL, _MODEL_CONFIG

    checkpoint = _load_checkpoint()
    if checkpoint.get("feature_version") != PPO_FEATURE_VERSION:
        raise ValueError("Unexpected PPO feature version")
    if checkpoint.get("bc_feature_version") != BC_FEATURE_VERSION:
        raise ValueError("Unexpected BC feature version")
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


def _greedy_plackett_luce(
    policy_logits: torch.Tensor,
    *,
    count: int,
    option_count: int,
) -> list[int]:
    """Match evaluator greedy ordered sampling; never canonicalize or sort."""
    if policy_logits.ndim != 1:
        raise ValueError("policy_logits must be one-dimensional")
    if option_count < 0 or option_count > policy_logits.numel():
        raise ValueError("option_count is outside policy_logits")
    if count < 0 or count > option_count:
        raise ValueError("count is outside the legal option range")
    logits = policy_logits[:option_count].float()
    remaining = torch.ones(
        option_count,
        dtype=torch.bool,
        device=logits.device,
    )
    action: list[int] = []
    for _ in range(count):
        chosen = int(logits.masked_fill(~remaining, -1e9).argmax())
        action.append(chosen)
        remaining[chosen] = False
    return action


def _greedy_legal_count(
    count_logits: torch.Tensor,
    *,
    minimum: int,
    maximum: int,
    option_count: int,
) -> int:
    """Match evaluator deterministic cardinality selection."""
    if count_logits.ndim != 1:
        raise ValueError("count_logits must be one-dimensional")
    if (
        minimum < 0
        or maximum < minimum
        or maximum > option_count
        or maximum >= count_logits.numel()
    ):
        raise ValueError("invalid legal count bounds")
    if minimum == maximum:
        return minimum
    values = torch.arange(count_logits.numel(), device=count_logits.device)
    allowed = (values >= minimum) & (values <= maximum)
    return int(count_logits.float().masked_fill(~allowed, -1e9).argmax())


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
        return list(range(minimum))

    batch = collate_decisions(
        [feature],
        max_state_entities=int(config["max_state_entities"]),
        entity_fields=int(config["entity_fields"]),
        option_fields=int(config["option_fields"]),
    )
    with torch.inference_mode():
        outputs = model(batch)

    count = _greedy_legal_count(
        outputs["count_logits"][0],
        minimum=minimum,
        maximum=maximum,
        option_count=option_count,
    )

    action = _greedy_plackett_luce(
        outputs["policy_logits"][0],
        count=count,
        option_count=option_count,
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
    """Return the Alakazam deck initially, then standard PPO actions."""
    if obs_dict.get("select") is None:
        return list(DECK)
    return _policy_action(obs_dict)
