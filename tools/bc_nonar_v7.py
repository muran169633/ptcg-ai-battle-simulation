#!/usr/bin/env python3
"""Non-autoregressive V7 BC components for PTCG.

This module deliberately keeps the deployed action policy feed-forward.  It
adds richer public-log features, a bounded cross-decision public history,
explicit selection-mode routing, option-to-state cross attention, and a joint
SKIP candidate for optional-single decisions.  It does not contain a GRU,
recurrent neural state, STOP sequence decoder, or autoregressive action loop.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from copy import deepcopy
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import train_bc_orbit as base
except ImportError:  # Executed with tools/ as sys.path[0].
    import train_bc_orbit as base


# Keep immutable references before install_into_trainer() replaces the base
# module hooks.  Looking them up through ``base`` afterwards would recurse.
BASE_FEATURIZE_ROW = base.featurize_row
BASE_MASKED_LOSSES = base.masked_losses


FEATURE_VERSION = "ptcg-bc-orbit-nonar-v7-ordered-rank"
GLOBAL_NUMERIC_SIZE = base.GLOBAL_NUMERIC_SIZE
ENTITY_NUMERIC_SIZE = base.ENTITY_NUMERIC_SIZE
OPTION_NUMERIC_SIZE = base.OPTION_NUMERIC_SIZE
MAX_ACTION_COUNT = base.MAX_ACTION_COUNT

PUBLIC_LOG_FIELDS = (
    "type",
    "playerIndex",
    "hasBasicPokemon",
    "cardId",
    "serial",
    "fromArea",
    "toArea",
    "cardIdActive",
    "serialActive",
    "cardIdBench",
    "serialBench",
    "cardIdBefore",
    "serialBefore",
    "cardIdAfter",
    "serialAfter",
    "cardIdTarget",
    "serialTarget",
    "attackId",
    "value",
    "putDamageCounter",
    "isRecover",
    "head",
    "result",
    "reason",
)
MAX_PERSISTENT_LOGS = 64
MAX_CURRENT_LOG_ENTITIES = 16
MAX_HISTORY_LOG_ENTITIES = 8

MODE_SINGLE = 0
MODE_OPTIONAL_SINGLE = 1
MODE_ORDERED_SEQUENCE = 2
MODE_UNORDERED_SET = 3
MODE_NAMES = {
    MODE_SINGLE: "single",
    MODE_OPTIONAL_SINGLE: "optional_single",
    MODE_ORDERED_SEQUENCE: "ordered_sequence",
    MODE_UNORDERED_SET: "unordered_set",
}
ORDERED_CONTEXT = 34
ORDERED_RANK_LOSS_WEIGHT = 0.20


def normalize_public_log(event: Any) -> dict[str, Any] | None:
    """Copy only documented, player-visible scalar log fields."""

    if not isinstance(event, dict):
        return None
    normalized: dict[str, Any] = {}
    for key in PUBLIC_LOG_FIELDS:
        value = event.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            if value is not None:
                normalized[key] = value
    return normalized


class PersistentPublicLogState:
    """Deterministic bounded public history used identically offline/online."""

    def __init__(self, max_logs: int = MAX_PERSISTENT_LOGS) -> None:
        if max_logs < 1:
            raise ValueError("max_logs must be positive")
        self.max_logs = int(max_logs)
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_logs)
        self._last_signature: str | None = None

    def reset(self) -> None:
        self._events.clear()
        self._last_signature = None

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def ingest_observation(self, observation: dict[str, Any]) -> bool:
        """Append this decision delta once; return whether it was new."""

        logs = observation.get("logs") or []
        normalized = [
            event
            for raw in logs
            if (event := normalize_public_log(raw)) is not None
        ]
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        signature_payload = {
            "turn": current.get("turn"),
            "turnActionCount": current.get("turnActionCount"),
            "yourIndex": current.get("yourIndex"),
            "selectType": select.get("type"),
            "selectContext": select.get("context"),
            "logs": normalized,
        }
        signature = json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if signature == self._last_signature:
            return False
        self._last_signature = signature
        self._events.extend(normalized)
        return True


def _event_signature(event: dict[str, Any], keys: Iterable[str]) -> str:
    return "|".join(f"{key}={event.get(key)}" for key in keys)


def log_entity(
    event: dict[str, Any],
    recent_position: int,
    your_index: int,
    hash_size: int,
    *,
    source: str,
) -> dict[str, list[int] | list[float]]:
    """Encode the full documented public Log contract in a compact token."""

    fields: list[int] = []
    event_type = int(event.get("type", -1) or -1)
    player_index = int(event.get("playerIndex", -1) or -1)
    rel = (
        base.relation(player_index, your_index)
        if player_index in (0, 1)
        else "none"
    )
    for namespace, value in (
        ("entity:kind", "log"),
        ("log:source", source),
        ("log:type", event_type),
        ("log:relation", rel),
        ("log:type_relation", f"{event_type}:{rel}"),
        ("log:recent_slot", min(recent_position, 31)),
        ("log:cardId", event.get("cardId")),
        ("log:attackId", event.get("attackId")),
        ("log:fromArea", event.get("fromArea")),
        ("log:toArea", event.get("toArea")),
        ("log:cardIdTarget", event.get("cardIdTarget")),
        (
            "log:transition",
            _event_signature(
                event,
                (
                    "fromArea",
                    "toArea",
                    "cardIdBefore",
                    "cardIdAfter",
                    "cardIdActive",
                    "cardIdBench",
                    "cardIdTarget",
                ),
            ),
        ),
        (
            "log:outcome",
            _event_signature(
                event,
                (
                    "value",
                    "putDamageCounter",
                    "isRecover",
                    "head",
                    "result",
                    "reason",
                ),
            ),
        ),
    ):
        if value is not None:
            base.add_field(fields, namespace, value, hash_size)
    numeric = [
        base.bounded(recent_position, 32),
        base.bounded(event_type, 64),
        base.bounded(event.get("fromArea", 0), 15),
        base.bounded(event.get("toArea", 0), 15),
        base.bounded(event.get("value", 0), 350),
        base.bounded(event.get("attackId", 0), 32),
        float(bool(event.get("putDamageCounter"))),
        float(bool(event.get("isRecover"))),
        float(bool(event.get("head"))),
        base.bounded(event.get("result", 0), 8),
        base.bounded(event.get("reason", 0), 32),
        1.0 if source == "history" else 0.0,
    ]
    return {"fields": fields, "numeric": numeric}


def history_summary_entity(
    events: list[dict[str, Any]],
    your_index: int,
    hash_size: int,
) -> dict[str, list[int] | list[float]]:
    types = Counter(int(event.get("type", -1) or -1) for event in events)
    relations = Counter()
    moves = damage = recovery = coin_heads = 0
    for event in events:
        player_index = int(event.get("playerIndex", -1) or -1)
        rel = (
            base.relation(player_index, your_index)
            if player_index in (0, 1)
            else "none"
        )
        relations[rel] += 1
        moves += int(event.get("fromArea") is not None or event.get("toArea") is not None)
        damage += int(bool(event.get("putDamageCounter")))
        recovery += int(bool(event.get("isRecover")))
        coin_heads += int(bool(event.get("head")))
    fields: list[int] = []
    base.add_field(fields, "entity:kind", "log_history_summary", hash_size)
    base.add_field(fields, "history:length_bucket", min(len(events) // 4, 15), hash_size)
    for event_type, count in types.most_common(8):
        base.add_field(
            fields,
            "history:type_count",
            f"{event_type}:{min(count, 15)}",
            hash_size,
        )
    numeric = [
        base.bounded(len(events), MAX_PERSISTENT_LOGS),
        base.bounded(len(types), 32),
        base.bounded(relations["self"], MAX_PERSISTENT_LOGS),
        base.bounded(relations["opponent"], MAX_PERSISTENT_LOGS),
        base.bounded(relations["none"], MAX_PERSISTENT_LOGS),
        base.bounded(moves, MAX_PERSISTENT_LOGS),
        base.bounded(damage, MAX_PERSISTENT_LOGS),
        base.bounded(recovery, MAX_PERSISTENT_LOGS),
        base.bounded(coin_heads, MAX_PERSISTENT_LOGS),
        0.0,
        0.0,
        0.0,
    ]
    return {"fields": fields, "numeric": numeric}


def state_entities(
    row: dict[str, Any],
    hash_size: int,
    max_entities: int,
) -> list[dict[str, list[int] | list[float]]]:
    """V6 priority ordering with complete current logs and bounded history."""

    observation = row["observation"]
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    entities: list[dict[str, list[int] | list[float]]] = []
    players = current.get("players") or []
    visible_players = [
        (player_index, player)
        for player_index, player in enumerate(players[:2])
        if isinstance(player, dict)
    ]

    for player_index, player in visible_players:
        entities.append(
            base.player_summary_entity(
                player,
                base.relation(player_index, your_index),
                hash_size,
            )
        )
    for zone in ("active", "bench", "hand"):
        for player_index, player in visible_players:
            cards = player.get(zone)
            if not isinstance(cards, list):
                continue
            rel = base.relation(player_index, your_index)
            for position, card in enumerate(cards):
                entities.append(base.card_entity(card, rel, zone, position, hash_size))

    stadium_cards = current.get("stadium")
    if isinstance(stadium_cards, list):
        for position, card in enumerate(stadium_cards):
            entities.append(base.card_entity(card, "public", "stadium", position, hash_size))
    for role, key in (("context_card", "contextCard"), ("effect_card", "effect")):
        card = select.get(key)
        if isinstance(card, dict):
            entities.append(
                base.card_entity(
                    card,
                    "self",
                    "select_context",
                    0,
                    hash_size,
                    role=role,
                )
            )

    current_logs = observation.get("logs") or []
    for recent_position, event in enumerate(
        reversed(current_logs[-MAX_CURRENT_LOG_ENTITIES:])
    ):
        normalized = normalize_public_log(event)
        if normalized is not None:
            entities.append(
                log_entity(
                    normalized,
                    recent_position,
                    your_index,
                    hash_size,
                    source="current",
                )
            )

    # Search/reveal pools remain ahead of older history because they directly
    # define the current legal decision.
    looking_cards = current.get("looking")
    if isinstance(looking_cards, list):
        for position, card in enumerate(looking_cards):
            entities.append(base.card_entity(card, "public", "looking", position, hash_size))
    deck_cards = select.get("deck")
    if isinstance(deck_cards, list):
        for position, card in enumerate(deck_cards):
            entities.append(
                base.card_entity(
                    card,
                    "self",
                    "select_deck",
                    position,
                    hash_size,
                    role="revealed_choice",
                )
            )

    history = [
        normalized
        for event in row.get("_persistent_logs", [])
        if (normalized := normalize_public_log(event)) is not None
    ][-MAX_PERSISTENT_LOGS:]
    if history:
        entities.append(history_summary_entity(history, your_index, hash_size))
        for recent_position, event in enumerate(
            reversed(history[-MAX_HISTORY_LOG_ENTITIES:])
        ):
            entities.append(
                log_entity(
                    event,
                    recent_position,
                    your_index,
                    hash_size,
                    source="history",
                )
            )

    for zone in ("discard", "prize"):
        for player_index, player in visible_players:
            cards = player.get(zone)
            if not isinstance(cards, list):
                continue
            rel = base.relation(player_index, your_index)
            for position, card in enumerate(cards):
                entities.append(base.card_entity(card, rel, zone, position, hash_size))
    return entities[:max_entities]


def featurize_row(
    row: dict[str, Any],
    hash_size: int,
    max_state_entities: int,
) -> dict[str, Any] | None:
    features = BASE_FEATURIZE_ROW(row, hash_size, max_state_entities)
    if features is None:
        return None
    entities = state_entities(row, hash_size, max_state_entities)
    features["entity_fields"] = [entity["fields"] for entity in entities]
    features["entity_numeric"] = [entity["numeric"] for entity in entities]
    return features


def selection_modes(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    minimum = batch["min_counts"]
    maximum = batch["max_counts"]
    contexts = batch["contexts"]
    modes = torch.full_like(minimum, MODE_UNORDERED_SET)
    fixed = minimum == maximum
    modes[fixed & (minimum == 1)] = MODE_SINGLE
    modes[(minimum == 0) & (maximum == 1)] = MODE_OPTIONAL_SINGLE
    modes[(contexts == ORDERED_CONTEXT) & (maximum > 1)] = MODE_ORDERED_SEQUENCE
    return modes


def ordered_rank_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Plackett-Luce ranking loss on static pointer scores.

    The network produces every option score in one feed-forward pass.  For
    context 34 only, the replay's raw action order supervises the descending
    ranking of those scores.  Removing earlier expert choices from each loss
    denominator is label-side bookkeeping, not autoregressive decoding.
    """

    ordered = selection_modes(batch) == MODE_ORDERED_SEQUENCE
    if not ordered.any():
        return logits.sum() * 0.0

    ordered_logits = logits[ordered]
    legal = batch["option_mask"][ordered].clone()
    sequences = batch["action_sequences"][ordered]
    row_weights = batch["sample_weights"][ordered]
    if "policy_team_weights" in batch:
        row_weights = row_weights * batch["policy_team_weights"][ordered]

    per_row = torch.zeros_like(row_weights)
    steps = torch.zeros_like(row_weights)
    for position in range(sequences.shape[1]):
        choice = sequences[:, position]
        active = choice >= 0
        if not active.any():
            break
        safe_choice = choice.clamp_min(0)
        chosen_is_legal = legal.gather(1, safe_choice.unsqueeze(1)).squeeze(1)
        if not bool(chosen_is_legal[active].all()):
            raise ValueError("ordered action_sequence contains an illegal/repeated option")
        denominator = torch.logsumexp(
            ordered_logits.masked_fill(~legal, -30.0), dim=1
        )
        numerator = ordered_logits.gather(1, safe_choice.unsqueeze(1)).squeeze(1)
        per_row = per_row + torch.where(active, denominator - numerator, 0.0)
        steps = steps + active.to(steps.dtype)
        legal.scatter_(1, safe_choice.unsqueeze(1), False)
    per_row = per_row / steps.clamp_min(1.0)
    return (per_row * row_weights).sum() / row_weights.sum().clamp_min(1.0)


class EntityOptionPolicy(nn.Module):
    """Feed-forward mode-routed pointer policy; intentionally non-AR."""

    def __init__(
        self,
        hash_size: int,
        categorical_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        max_state_entities: int,
        max_options: int = 128,
    ) -> None:
        super().__init__()
        self.hash_size = hash_size
        self.categorical_dim = categorical_dim
        self.model_dim = model_dim
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.max_state_entities = max_state_entities
        self.max_options = max_options

        # Names and shapes through value_head intentionally match V5/V6 so a
        # proven BC checkpoint can initialize every legacy tensor exactly.
        self.embedding = nn.Embedding(hash_size, categorical_dim, padding_idx=0)
        self.global_encoder = nn.Sequential(
            nn.Linear(categorical_dim + GLOBAL_NUMERIC_SIZE, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(categorical_dim + ENTITY_NUMERIC_SIZE, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(categorical_dim + OPTION_NUMERIC_SIZE, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.kind_embedding = nn.Embedding(3, model_dim)
        self.state_position = nn.Embedding(max_state_entities, model_dim)
        self.option_position = nn.Embedding(max_options, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.actor_query = nn.Linear(model_dim, model_dim, bias=False)
        self.actor_key = nn.Linear(model_dim, model_dim, bias=False)
        self.actor_residual = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, MAX_ACTION_COUNT + 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

        self.option_cross_attn = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.option_cross_norm1 = nn.LayerNorm(model_dim)
        self.option_cross_ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.option_cross_norm2 = nn.LayerNorm(model_dim)
        self.option_cross_gate = nn.Parameter(torch.tensor(-2.0))
        self.mode_embedding = nn.Embedding(4, model_dim)
        self.select_encoder = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.LayerNorm(model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.select_gate = nn.Parameter(torch.tensor(-2.0))
        self.skip_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    masked_field_mean = staticmethod(base.EntityOptionPolicy.masked_field_mean)

    @staticmethod
    def _masked_option_pool(
        options: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(options.dtype).unsqueeze(-1)
        mean = (options * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        maximum = options.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        count_trunk_gradient_scale: float = 1.0,
        value_trunk_gradient_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        global_cat = self.masked_field_mean(
            self.embedding(batch["global_fields"]), batch["global_field_mask"]
        )
        global_token = self.global_encoder(
            torch.cat((global_cat, batch["global_numeric"]), dim=-1)
        ) + self.kind_embedding.weight[0]

        state_cat = self.masked_field_mean(
            self.embedding(batch["state_fields"]), batch["state_field_mask"]
        )
        state = self.entity_encoder(torch.cat((state_cat, batch["state_numeric"]), dim=-1))
        state_positions = torch.arange(state.shape[1], device=state.device).clamp_max(
            self.max_state_entities - 1
        )
        state = state + self.kind_embedding.weight[1] + self.state_position(
            state_positions
        ).unsqueeze(0)

        option_cat = self.masked_field_mean(
            self.embedding(batch["option_fields"]), batch["option_field_mask"]
        )
        options = self.option_encoder(
            torch.cat((option_cat, batch["option_numeric"]), dim=-1)
        )
        option_positions = torch.arange(options.shape[1], device=options.device).clamp_max(
            self.max_options - 1
        )
        options = options + self.kind_embedding.weight[2] + self.option_position(
            option_positions
        ).unsqueeze(0)

        sequence = torch.cat((global_token.unsqueeze(1), state, options), dim=1)
        sequence_mask = torch.cat(
            (
                torch.ones((state.shape[0], 1), dtype=torch.bool, device=state.device),
                batch["state_mask"],
                batch["option_mask"],
            ),
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=~sequence_mask)
        global_encoded = encoded[:, 0]
        state_encoded = encoded[:, 1 : 1 + state.shape[1]]
        option_encoded = encoded[:, 1 + state.shape[1] :]

        memory = torch.cat((global_encoded.unsqueeze(1), state_encoded), dim=1)
        memory_mask = torch.cat(
            (
                torch.ones((state.shape[0], 1), dtype=torch.bool, device=state.device),
                batch["state_mask"],
            ),
            dim=1,
        )
        cross, _ = self.option_cross_attn(
            option_encoded,
            memory,
            memory,
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        option_cross = self.option_cross_norm1(option_encoded + cross)
        option_cross = self.option_cross_norm2(
            option_cross + self.option_cross_ffn(option_cross)
        )
        option_fused = option_encoded + torch.sigmoid(self.option_cross_gate) * (
            option_cross - option_encoded
        )

        option_mean, option_max = self._masked_option_pool(
            option_fused, batch["option_mask"]
        )
        mode_features = self.mode_embedding(selection_modes(batch))
        select_delta = self.select_encoder(
            torch.cat((global_encoded, option_mean, option_max, mode_features), dim=-1)
        )
        select_summary = global_encoded + torch.sigmoid(self.select_gate) * select_delta

        pointer = torch.einsum(
            "bd,bod->bo",
            self.actor_query(select_summary),
            self.actor_key(option_fused),
        ) / math.sqrt(self.model_dim)
        residual = self.actor_residual(
            torch.cat((option_fused, select_summary.unsqueeze(1).expand_as(option_fused)), dim=-1)
        ).squeeze(-1)
        policy_logits = (pointer + residual).masked_fill(~batch["option_mask"], -30.0)

        def scaled(features: torch.Tensor, scale: float) -> torch.Tensor:
            if scale == 1.0:
                return features
            detached = features.detach()
            if scale == 0.0:
                return detached
            return detached + scale * (features - detached)

        return {
            "policy_logits": policy_logits,
            "count_logits": self.count_head(
                scaled(select_summary, count_trunk_gradient_scale)
            ),
            "value_logits": self.value_head(
                scaled(select_summary, value_trunk_gradient_scale)
            ).squeeze(-1),
            "skip_logits": self.skip_head(select_summary).squeeze(-1),
            "selection_modes": selection_modes(batch),
        }


def allowed_count_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return base.allowed_count_mask(batch)


def masked_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: base.TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Base actor losses plus legal count masking and optional joint SKIP CE."""

    legal_outputs = dict(outputs)
    legal_outputs["count_logits"] = outputs["count_logits"].masked_fill(
        ~allowed_count_mask(batch), -30.0
    )
    total, components = BASE_MASKED_LOSSES(legal_outputs, batch, config)
    order_rank = ordered_rank_loss(outputs["policy_logits"], batch)
    optional = selection_modes(batch) == MODE_OPTIONAL_SINGLE
    if optional.any():
        joint_logits = torch.cat(
            (outputs["policy_logits"], outputs["skip_logits"].unsqueeze(1)), dim=1
        )
        skip_index = outputs["policy_logits"].shape[1]
        target_option = batch["targets"].argmax(dim=1)
        joint_target = torch.where(
            batch["action_counts"] == 0,
            torch.full_like(target_option, skip_index),
            target_option,
        )
        optional_raw = F.cross_entropy(joint_logits, joint_target, reduction="none")
        policy_weights, _ = base.resolve_trajectory_loss_weights(
            batch["sample_weights"], config.trajectory_weight_scope
        )
        if "policy_team_weights" in batch:
            policy_weights = policy_weights * batch["policy_team_weights"]
        optional_joint = (
            optional_raw[optional] * policy_weights[optional]
        ).sum() / policy_weights[optional].sum().clamp_min(1.0)
    else:
        optional_joint = outputs["skip_logits"].sum() * 0.0
    total = (
        total
        + 0.5 * optional_joint
        + ORDERED_RANK_LOSS_WEIGHT * order_rank
    )
    return total, {
        **components,
        "optional_joint": optional_joint,
        "ordered_rank": order_rank,
    }


def predict_action_mask(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    modes = selection_modes(batch)
    option_mask = batch["option_mask"]
    option_count = option_mask.sum(dim=1)
    count_logits = outputs["count_logits"].masked_fill(
        ~allowed_count_mask(batch), -30.0
    )
    predicted_counts = count_logits.argmax(dim=1)
    fixed = batch["min_counts"] == batch["max_counts"]
    predicted_counts = torch.where(
        fixed,
        batch["min_counts"].clamp(0, MAX_ACTION_COUNT),
        predicted_counts,
    )

    optional = modes == MODE_OPTIONAL_SINGLE
    joint_logits = torch.cat(
        (outputs["policy_logits"], outputs["skip_logits"].unsqueeze(1)), dim=1
    )
    joint_choice = joint_logits.argmax(dim=1)
    skip_index = outputs["policy_logits"].shape[1]
    predicted_counts = torch.where(
        optional,
        (joint_choice != skip_index).to(predicted_counts.dtype),
        predicted_counts,
    )
    predicted_counts = torch.minimum(predicted_counts, option_count)

    ranks = torch.argsort(
        torch.argsort(outputs["policy_logits"], dim=1, descending=True), dim=1
    )
    prediction = (ranks < predicted_counts.unsqueeze(1)) & option_mask
    if optional.any():
        optional_prediction = torch.zeros_like(prediction)
        choose_option = optional & (joint_choice != skip_index)
        if choose_option.any():
            optional_prediction[choose_option, joint_choice[choose_option]] = True
        prediction = torch.where(optional.unsqueeze(1), optional_prediction, prediction)
    return prediction, predicted_counts


@torch.no_grad()
def evaluate(
    model: EntityOptionPolicy,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    started = __import__("time").time()
    totals: Counter[str] = Counter()
    context_total: Counter[int] = Counter()
    context_correct: Counter[int] = Counter()
    mode_total: Counter[int] = Counter()
    mode_correct: Counter[int] = Counter()
    for cpu_batch in loader:
        batch = base.move_batch(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(batch)
        prediction, predicted_counts = predict_action_mask(outputs, batch)
        set_exact = (
            (prediction == batch["targets"].bool()) | ~batch["option_mask"]
        ).all(dim=1)
        modes = selection_modes(batch)
        ordered = modes == MODE_ORDERED_SEQUENCE
        ranked = outputs["policy_logits"].argsort(dim=1, descending=True)
        target_sequence = batch["action_sequences"]
        sequence_width = target_sequence.shape[1]
        if ranked.shape[1] < sequence_width:
            ranked = F.pad(
                ranked,
                (0, sequence_width - ranked.shape[1]),
                value=-1,
            )
        predicted_sequence = ranked[:, :sequence_width]
        positions = torch.arange(sequence_width, device=device).unsqueeze(0)
        sequence_mask = positions < batch["action_counts"].unsqueeze(1)
        order_exact = (
            (predicted_sequence == target_sequence) | ~sequence_mask
        ).all(dim=1)
        exact = torch.where(ordered, set_exact & order_exact, set_exact)
        fixed = batch["min_counts"] == batch["max_counts"]
        flexible = ~fixed
        count_correct = predicted_counts == batch["action_counts"]
        nonempty = batch["action_counts"] > 0
        top1 = outputs["policy_logits"].argmax(dim=1)
        top1_correct = batch["targets"].gather(1, top1.unsqueeze(1)).squeeze(1).bool()
        value_prediction = outputs["value_logits"] >= 0
        value_target = batch["win_targets"].bool()

        totals["rows"] += exact.numel()
        totals["correct"] += int(exact.sum())
        totals["set_correct"] += int(set_exact.sum())
        totals["ordered"] += int(ordered.sum())
        totals["ordered_correct"] += int((order_exact & ordered).sum())
        totals["fixed"] += int(fixed.sum())
        totals["fixed_correct"] += int((exact & fixed).sum())
        totals["flexible"] += int(flexible.sum())
        totals["flexible_correct"] += int((exact & flexible).sum())
        totals["count_correct"] += int(count_correct.sum())
        totals["nonempty"] += int(nonempty.sum())
        totals["top1_correct"] += int((top1_correct & nonempty).sum())
        totals["value_correct"] += int((value_prediction == value_target).sum())

        contexts_cpu = batch["contexts"].cpu()
        modes_cpu = modes.cpu()
        exact_cpu = exact.cpu()
        for context_value in torch.unique(contexts_cpu):
            context = int(context_value)
            selected = contexts_cpu == context
            context_total[context] += int(selected.sum())
            context_correct[context] += int(exact_cpu[selected].sum())
        for mode_value in torch.unique(modes_cpu):
            mode = int(mode_value)
            selected = modes_cpu == mode
            mode_total[mode] += int(selected.sum())
            mode_correct[mode] += int(exact_cpu[selected].sum())

    rows = totals["rows"]
    return {
        "rows": rows,
        "seconds": __import__("time").time() - started,
        "exact_action_set_accuracy": totals["correct"] / max(rows, 1),
        "semantic_exact_accuracy": totals["correct"] / max(rows, 1),
        "action_set_exact_accuracy": totals["set_correct"] / max(rows, 1),
        "ordered_sequence_exact_accuracy": (
            totals["ordered_correct"] / max(totals["ordered"], 1)
        ),
        "fixed_cardinality_accuracy": totals["fixed_correct"] / max(totals["fixed"], 1),
        "flexible_cardinality_accuracy": totals["flexible_correct"] / max(totals["flexible"], 1),
        "count_accuracy": totals["count_correct"] / max(rows, 1),
        "nonempty_top1_accuracy": totals["top1_correct"] / max(totals["nonempty"], 1),
        "value_win_accuracy": totals["value_correct"] / max(rows, 1),
        "by_mode": {
            MODE_NAMES[mode]: {
                "rows": count,
                "exact_accuracy": mode_correct[mode] / count,
            }
            for mode, count in sorted(mode_total.items())
        },
        "by_context": {
            str(context): {
                "rows": count,
                "exact_accuracy": context_correct[context] / count,
            }
            for context, count in context_total.most_common()
        },
    }


def load_partial_bc_initialization(
    model: EntityOptionPolicy,
    checkpoint_path: Any,
    allowed_source_feature_versions: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Load every exact-shape legacy tensor; require all V5/V6 core keys."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("--init-checkpoint root must be a dictionary")
    source_version = str(checkpoint.get("feature_version", ""))
    if source_version != FEATURE_VERSION and source_version not in allowed_source_feature_versions:
        raise ValueError(
            f"Initialization feature version {source_version!r} is not explicitly allowed"
        )
    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise ValueError("--init-checkpoint is missing model_state_dict")
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target
        and isinstance(value, torch.Tensor)
        and tuple(value.shape) == tuple(target[key].shape)
    }
    missing_legacy = sorted(
        key
        for key in source
        if key in target and key not in compatible
    )
    if missing_legacy:
        raise ValueError(f"Legacy tensor shape mismatch: {missing_legacy[:12]}")
    required_prefixes = (
        "embedding.",
        "global_encoder.",
        "entity_encoder.",
        "option_encoder.",
        "transformer.",
        "actor_query.",
        "actor_key.",
        "actor_residual.",
        "count_head.",
        "value_head.",
    )
    required = {key for key in target if key.startswith(required_prefixes)}
    absent_required = sorted(required - set(compatible))
    if absent_required:
        raise ValueError(f"Initialization omits legacy core tensors: {absent_required[:12]}")
    result = model.load_state_dict(compatible, strict=False)
    new_keys = sorted(result.missing_keys)
    allowed_new_prefixes = (
        "option_cross_",
        "mode_embedding.",
        "select_encoder.",
        "select_gate",
        "skip_head.",
    )
    unexpected_missing = [
        key for key in new_keys if not key.startswith(allowed_new_prefixes)
    ]
    if unexpected_missing or result.unexpected_keys:
        raise ValueError(
            "Unexpected partial initialization result: "
            f"missing={unexpected_missing[:12]} unexpected={result.unexpected_keys[:12]}"
        )
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": base.file_sha256(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_valid_metrics": checkpoint.get("valid_metrics"),
        "source_feature_version": source_version,
        "target_feature_version": FEATURE_VERSION,
        "cross_feature_initialization": source_version != FEATURE_VERSION,
        "loaded_tensor_count": len(compatible),
        "new_tensor_count": len(new_keys),
        "new_tensor_names": new_keys,
    }


def install_into_trainer() -> None:
    """Install V7 components into the versioned base training harness."""

    base.FEATURE_VERSION = FEATURE_VERSION
    base.EntityOptionPolicy = EntityOptionPolicy
    base.featurize_row = featurize_row
    base.masked_losses = masked_losses
    base.predict_action_mask = predict_action_mask
    base.evaluate = evaluate
    base.load_bc_initialization = load_partial_bc_initialization
