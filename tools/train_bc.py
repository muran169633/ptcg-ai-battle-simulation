#!/usr/bin/env python3
"""Train and evaluate a behavior-cloning policy on PTCG decision frames.

The policy scores a dynamic list of legal options. At inference time, candidate
scores are converted to an action set while respecting select.minCount and
select.maxCount. Validation reports exact action-set accuracy, which is the
relevant decision metric for this dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


FEATURE_VERSION = "ptcg-bc-hashed-visible-v4"
DEFAULT_HASH_SIZE = 262_144
DEFAULT_STATE_TOKENS = 192
DEFAULT_CANDIDATE_TOKENS = 32
STATE_NUMERIC_SIZE = 18
CANDIDATE_NUMERIC_SIZE = 10

# Engine AreaType values.
AREA_TO_ZONE = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
    6: "prize",
    7: "stadium",
    12: "looking",
    14: "deck",
}


def log(message: str) -> None:
    print(message, flush=True)


@lru_cache(maxsize=500_000)
def token_id(text: str, hash_size: int = DEFAULT_HASH_SIZE) -> int:
    return zlib.crc32(text.encode("utf-8")) % (hash_size - 1) + 1


def bounded(value: Any, scale: float, limit: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-limit, min(limit, number / scale))


def add_token(tokens: list[int], namespace: str, value: Any) -> None:
    if value is None:
        return
    tokens.append(token_id(f"{namespace}={value}"))


def card_id(card: Any) -> int:
    if isinstance(card, dict):
        try:
            return int(card.get("id", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def add_card_tokens(
    tokens: list[int],
    card: Any,
    zone: str,
    position: int | None = None,
) -> None:
    if not isinstance(card, dict):
        add_token(tokens, f"{zone}:card", "hidden")
        return
    cid = card_id(card)
    if cid:
        add_token(tokens, f"{zone}:card", cid)
        if position is not None and position < 8:
            add_token(tokens, f"{zone}:slot{position}:card", cid)
    if "hp" in card:
        hp = int(card.get("hp", 0) or 0)
        max_hp = int(card.get("maxHp", 0) or 0)
        add_token(tokens, f"{zone}:hp_bucket", max(0, hp) // 30)
        add_token(tokens, f"{zone}:damage_bucket", max(0, max_hp - hp) // 30)
        add_token(tokens, f"{zone}:appear", int(bool(card.get("appearThisTurn"))))
    for energy in card.get("energies", []) or []:
        add_token(tokens, f"{zone}:energy", energy)
    for child_zone, key in (
        ("energy_card", "energyCards"),
        ("tool", "tools"),
        ("pre_evolution", "preEvolution"),
    ):
        for child in card.get(key, []) or []:
            child_id = card_id(child)
            if child_id:
                add_token(tokens, f"{zone}:{child_zone}", child_id)


def relation(player_index: int, your_index: int) -> str:
    return "self" if player_index == your_index else "opponent"


def get_zone(
    observation: dict[str, Any],
    area: int,
    player_index: int,
) -> list[Any] | None:
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    zone = AREA_TO_ZONE.get(int(area))
    if zone == "stadium":
        return current.get("stadium") or []
    if zone == "looking":
        return current.get("looking") or []
    if zone == "deck":
        if select.get("deck") is not None:
            return select.get("deck") or []
    players = current.get("players") or []
    if 0 <= player_index < len(players) and isinstance(players[player_index], dict):
        values = players[player_index].get(zone) if zone else None
        return values if isinstance(values, list) else None
    return None


def resolve_card(
    observation: dict[str, Any],
    area: int,
    index: int,
    player_index: int,
) -> Any:
    values = get_zone(observation, area, player_index)
    if values is not None and 0 <= index < len(values):
        return values[index]
    return None


def state_features(
    row: dict[str, Any],
    max_tokens: int,
) -> tuple[list[int], list[float]]:
    observation = row["observation"]
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    tokens: list[int] = []

    select_type = int(select.get("type", row.get("select_type", 0)) or 0)
    context = int(select.get("context", row.get("select_context", 0)) or 0)
    min_count = int(select.get("minCount", row.get("min_count", 0)) or 0)
    max_count = int(select.get("maxCount", row.get("max_count", 0)) or 0)
    options = select.get("option") or []
    team_name = str(row.get("team_name", "") or "")
    deck_hash = str(row.get("deck_hash", "") or "")
    if team_name:
        add_token(tokens, "policy:demonstrator", team_name)
        add_token(tokens, "policy:demonstrator_context", f"{team_name}:{context}")
    if deck_hash:
        add_token(tokens, "policy:deck", deck_hash)
        add_token(tokens, "policy:deck_context", f"{deck_hash}:{context}")
    for namespace, value in (
        ("select:type", select_type),
        ("select:context", context),
        ("select:min", min_count),
        ("select:max", max_count),
        ("select:option_count", min(len(options), 63)),
        ("select:type_context", f"{select_type}:{context}"),
        ("select:context_card", card_id(select.get("contextCard"))),
        ("select:effect_card", card_id(select.get("effect"))),
        ("current:turn_bucket", int(current.get("turn", 0) or 0) // 2),
        ("current:action_count", min(int(current.get("turnActionCount", 0) or 0), 31)),
        ("current:first_relative", int(current.get("firstPlayer", 0) == your_index)),
    ):
        add_token(tokens, namespace, value)

    for flag in (
        "supporterPlayed",
        "stadiumPlayed",
        "energyAttached",
        "retreated",
    ):
        add_token(tokens, f"current:{flag}", int(bool(current.get(flag))))

    players = current.get("players") or []
    for player_index, player in enumerate(players[:2]):
        if not isinstance(player, dict):
            continue
        rel = relation(player_index, your_index)
        for key, cap in (
            ("deckCount", 60),
            ("handCount", 30),
            ("benchMax", 8),
        ):
            add_token(tokens, f"{rel}:{key}", min(int(player.get(key, 0) or 0), cap))
        for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
            add_token(tokens, f"{rel}:{flag}", int(bool(player.get(flag))))
        for zone in ("hand", "active", "bench", "discard", "prize"):
            cards = player.get(zone)
            if cards is None:
                add_token(tokens, f"{rel}:{zone}", "hidden")
                continue
            if not isinstance(cards, list):
                continue
            add_token(tokens, f"{rel}:{zone}:count", min(len(cards), 63))
            for position, card in enumerate(cards):
                add_card_tokens(tokens, card, f"{rel}:{zone}", position)

    for zone in ("stadium", "looking"):
        cards = current.get(zone)
        if isinstance(cards, list):
            for position, card in enumerate(cards):
                add_card_tokens(tokens, card, zone, position)

    for zone in ("deck",):
        cards = select.get(zone)
        if isinstance(cards, list):
            for position, card in enumerate(cards):
                add_card_tokens(tokens, card, f"select:{zone}", position)

    # Recent public events often disambiguate otherwise identical board states.
    logs = observation.get("logs") or []
    for recent_position, event in enumerate(reversed(logs[-24:])):
        if not isinstance(event, dict):
            continue
        event_type = event.get("type", -1)
        player_index = int(event.get("playerIndex", -1) or -1)
        rel = relation(player_index, your_index) if player_index in (0, 1) else "none"
        add_token(tokens, "log:type_rel", f"{event_type}:{rel}")
        if recent_position < 6:
            add_token(tokens, f"log:recent{recent_position}:type_rel", f"{event_type}:{rel}")
        if event.get("cardId") is not None:
            add_token(tokens, f"log:{event_type}:card", event.get("cardId"))
        if event.get("attackId") is not None:
            add_token(tokens, f"log:{event_type}:attack", event.get("attackId"))

    self_player = (
        players[your_index]
        if 0 <= your_index < len(players) and isinstance(players[your_index], dict)
        else {}
    )
    opponent_index = 1 - your_index
    opponent = (
        players[opponent_index]
        if 0 <= opponent_index < len(players)
        and isinstance(players[opponent_index], dict)
        else {}
    )
    numeric = [
        bounded(current.get("turn", 0), 60),
        bounded(current.get("turnActionCount", 0), 20),
        bounded(self_player.get("deckCount", 0), 60),
        bounded(self_player.get("handCount", 0), 20),
        bounded(len(self_player.get("active") or []), 2),
        bounded(len(self_player.get("bench") or []), 5),
        bounded(len(self_player.get("discard") or []), 40),
        bounded(len(self_player.get("prize") or []), 6),
        bounded(opponent.get("deckCount", 0), 60),
        bounded(opponent.get("handCount", 0), 20),
        bounded(len(opponent.get("active") or []), 2),
        bounded(len(opponent.get("bench") or []), 5),
        bounded(len(opponent.get("discard") or []), 40),
        bounded(len(opponent.get("prize") or []), 6),
        bounded(select.get("remainDamageCounter", 0), 30),
        bounded(select.get("remainEnergyCost", 0), 8),
        bounded(min_count, 10),
        bounded(max_count, 10),
    ]
    return tokens[:max_tokens], numeric


def candidate_features(
    row: dict[str, Any],
    option: dict[str, Any],
    option_position: int,
    max_tokens: int,
) -> tuple[list[int], list[float]]:
    observation = row["observation"]
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    option_type = int(option.get("type", -1) or 0)
    context = int(select.get("context", row.get("select_context", 0)) or 0)
    select_type = int(select.get("type", row.get("select_type", 0)) or 0)
    tokens: list[int] = []

    add_token(tokens, "option:type", option_type)
    add_token(tokens, "option:position", min(option_position, 63))
    add_token(tokens, "option:type_position", f"{option_type}:{min(option_position, 15)}")
    add_token(tokens, "cross:context_option", f"{context}:{option_type}")
    add_token(tokens, "cross:select_option", f"{select_type}:{option_type}")
    for key, value in sorted(option.items()):
        if key == "serial":
            continue
        add_token(tokens, f"option:{key}", value)

    source = target = attached = None
    area = int(option.get("area", 0) or 0)
    index = int(option.get("index", -1) if option.get("index") is not None else -1)
    player_index = int(option.get("playerIndex", your_index) or 0)
    if option_type == 7:  # Play uses a hand index.
        area, player_index = 2, your_index
    if option_type in (3, 4, 5, 6, 8, 9, 10, 11) and index >= 0:
        source = resolve_card(observation, area, index, player_index)
    if option_type in (8, 9):
        target_area = int(option.get("inPlayArea", 0) or 0)
        target_index = int(option.get("inPlayIndex", -1) or 0)
        target = resolve_card(observation, target_area, target_index, your_index)
    if option_type in (4, 5, 6) and isinstance(source, dict):
        attached_key = "tools" if option_type == 4 else "energyCards"
        attached_index_key = "toolIndex" if option_type == 4 else "energyIndex"
        attached_index = int(option.get(attached_index_key, -1) or 0)
        attached_values = source.get(attached_key) or []
        if 0 <= attached_index < len(attached_values):
            attached = attached_values[attached_index]

    source_id = card_id(source)
    target_id = card_id(target)
    attached_id = card_id(attached)
    for namespace, cid in (
        ("candidate:source_card", source_id),
        ("candidate:target_card", target_id),
        ("candidate:attached_card", attached_id),
    ):
        if cid:
            add_token(tokens, namespace, cid)
    if source_id:
        add_token(tokens, "cross:context_source", f"{context}:{source_id}")
        add_token(tokens, "cross:type_source", f"{option_type}:{source_id}")
    if target_id:
        add_token(tokens, "cross:context_target", f"{context}:{target_id}")
        add_token(tokens, "cross:source_target", f"{source_id}:{target_id}")

    source_hp = float(source.get("hp", 0) or 0) if isinstance(source, dict) else 0.0
    source_max_hp = (
        float(source.get("maxHp", 0) or 0) if isinstance(source, dict) else 0.0
    )
    source_energy_count = (
        len(source.get("energies") or []) if isinstance(source, dict) else 0
    )
    numeric = [
        bounded(option_position, 32),
        bounded(area, 15),
        bounded(index, 60),
        bounded(option.get("playerIndex", your_index), 2),
        bounded(option.get("inPlayArea", 0), 15),
        bounded(option.get("inPlayIndex", 0), 8),
        bounded(option.get("attackId", 0), 32),
        bounded(option.get("number", option.get("count", 0)), 30),
        source_hp / source_max_hp if source_max_hp > 0 else 0.0,
        bounded(source_energy_count, 8),
    ]
    return tokens[:max_tokens], numeric


def featurize_row(
    row: dict[str, Any],
    max_state_tokens: int,
    max_candidate_tokens: int,
) -> dict[str, Any] | None:
    observation = row.get("observation") or {}
    select = observation.get("select")
    if not isinstance(select, dict):
        return None
    options = select.get("option") or []
    if not options:
        return None
    state_tokens, state_numeric = state_features(row, max_state_tokens)
    candidate_tokens = []
    candidate_numeric = []
    for position, option in enumerate(options):
        tokens, numeric = candidate_features(
            row,
            option,
            position,
            max_candidate_tokens,
        )
        candidate_tokens.append(tokens)
        candidate_numeric.append(numeric)
    action = {int(index) for index in row.get("action", [])}
    targets = [1.0 if index in action else 0.0 for index in range(len(options))]
    team_name = str(row.get("team_name", "") or "")
    deck_hash = str(row.get("deck_hash", "") or "")
    context = int(select.get("context", row.get("select_context", 0)) or 0)
    select_type = int(select.get("type", row.get("select_type", 0)) or 0)
    policy_tokens = [
        token_id(f"direct:team={team_name}"),
        token_id(f"direct:deck={deck_hash}"),
        token_id(f"direct:context={context}"),
        token_id(f"direct:select_type={select_type}"),
        token_id(f"direct:team_context={team_name}:{context}"),
        token_id(f"direct:deck_context={deck_hash}:{context}"),
    ]
    return {
        "state_tokens": state_tokens,
        "state_numeric": state_numeric,
        "candidate_tokens": candidate_tokens,
        "candidate_numeric": candidate_numeric,
        "targets": targets,
        "policy_tokens": policy_tokens,
        "min_count": int(select.get("minCount", row.get("min_count", 0)) or 0),
        "max_count": int(select.get("maxCount", row.get("max_count", 0)) or 0),
        "context": int(select.get("context", row.get("select_context", 0)) or 0),
        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
    }


class ZipDecisionDataset(IterableDataset):
    def __init__(
        self,
        archive_path: Path,
        split: str,
        max_rows: int | None,
        split_seed: int,
        shuffle_seed: int,
        epoch: int,
        max_state_tokens: int,
        max_candidate_tokens: int,
        use_trajectory_weights: bool,
        deck_hashes: tuple[str, ...],
        team_names: tuple[str, ...],
        split_mode: str,
    ) -> None:
        super().__init__()
        self.archive_path = archive_path
        self.split = split
        self.max_rows = max_rows
        self.split_seed = split_seed
        self.shuffle_seed = shuffle_seed
        self.epoch = epoch
        self.max_state_tokens = max_state_tokens
        self.max_candidate_tokens = max_candidate_tokens
        self.use_trajectory_weights = use_trajectory_weights
        self.deck_hashes = set(deck_hashes)
        self.team_names = set(team_names)
        self.split_mode = split_mode

    def row_split(self, row: dict[str, Any]) -> str:
        if self.split_mode == "archive":
            return str(row.get("split", ""))
        episode_id = str(row.get("episode_id", ""))
        digest = hashlib.sha256(f"{self.split_seed}:{episode_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        if value < 0.80:
            return "train"
        if value < 0.90:
            return "valid"
        return "test"

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rng = random.Random(self.shuffle_seed + self.epoch * 10_007)
        with zipfile.ZipFile(self.archive_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(".jsonl")
                and (
                    self.split_mode == "episode_hash"
                    or name.startswith(f"{self.split}/")
                )
            ]
            rng.shuffle(members)
            members = members[worker_id::workers]
            worker_limit = (
                math.ceil(self.max_rows / workers)
                if self.max_rows is not None
                else None
            )
            emitted = 0
            for member in members:
                with archive.open(member) as handle:
                    for line in handle:
                        row = orjson.loads(line)
                        if self.row_split(row) != self.split:
                            continue
                        if (
                            self.deck_hashes
                            and str(row.get("deck_hash", "")) not in self.deck_hashes
                        ):
                            continue
                        if (
                            self.team_names
                            and str(row.get("team_name", "")) not in self.team_names
                        ):
                            continue
                        features = featurize_row(
                            row,
                            self.max_state_tokens,
                            self.max_candidate_tokens,
                        )
                        if features is None:
                            continue
                        if not self.use_trajectory_weights:
                            features["sample_weight"] = 1.0
                        yield features
                        emitted += 1
                        if worker_limit is not None and emitted >= worker_limit:
                            return


def collate_decisions(
    rows: list[dict[str, Any]],
    max_state_tokens: int,
    max_candidate_tokens: int,
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    max_options = max(len(row["targets"]) for row in rows)
    state_tokens = torch.zeros((batch_size, max_state_tokens), dtype=torch.long)
    state_mask = torch.zeros((batch_size, max_state_tokens), dtype=torch.bool)
    state_numeric = torch.zeros((batch_size, STATE_NUMERIC_SIZE), dtype=torch.float32)
    policy_tokens = torch.zeros((batch_size, 6), dtype=torch.long)
    candidate_tokens = torch.zeros(
        (batch_size, max_options, max_candidate_tokens),
        dtype=torch.long,
    )
    candidate_token_mask = torch.zeros(
        (batch_size, max_options, max_candidate_tokens),
        dtype=torch.bool,
    )
    candidate_numeric = torch.zeros(
        (batch_size, max_options, CANDIDATE_NUMERIC_SIZE),
        dtype=torch.float32,
    )
    option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool)
    targets = torch.zeros((batch_size, max_options), dtype=torch.float32)
    min_counts = torch.zeros(batch_size, dtype=torch.long)
    max_counts = torch.zeros(batch_size, dtype=torch.long)
    contexts = torch.zeros(batch_size, dtype=torch.long)
    sample_weights = torch.ones(batch_size, dtype=torch.float32)

    for batch_index, row in enumerate(rows):
        state = row["state_tokens"]
        state_tokens[batch_index, : len(state)] = torch.tensor(state)
        state_mask[batch_index, : len(state)] = True
        state_numeric[batch_index] = torch.tensor(row["state_numeric"])
        policy_tokens[batch_index] = torch.tensor(row["policy_tokens"])
        option_count = len(row["targets"])
        option_mask[batch_index, :option_count] = True
        targets[batch_index, :option_count] = torch.tensor(row["targets"])
        for option_index, tokens in enumerate(row["candidate_tokens"]):
            candidate_tokens[batch_index, option_index, : len(tokens)] = torch.tensor(
                tokens
            )
            candidate_token_mask[
                batch_index, option_index, : len(tokens)
            ] = True
        candidate_numeric[batch_index, :option_count] = torch.tensor(
            row["candidate_numeric"]
        )
        min_counts[batch_index] = row["min_count"]
        max_counts[batch_index] = row["max_count"]
        contexts[batch_index] = row["context"]
        sample_weights[batch_index] = row["sample_weight"]

    return {
        "state_tokens": state_tokens,
        "state_mask": state_mask,
        "state_numeric": state_numeric,
        "policy_tokens": policy_tokens,
        "candidate_tokens": candidate_tokens,
        "candidate_token_mask": candidate_token_mask,
        "candidate_numeric": candidate_numeric,
        "option_mask": option_mask,
        "targets": targets,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "contexts": contexts,
        "sample_weights": sample_weights,
    }


class CandidatePolicy(nn.Module):
    def __init__(
        self,
        hash_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hash_size = hash_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.embedding = nn.Embedding(hash_size, embedding_dim, padding_idx=0)
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + STATE_NUMERIC_SIZE, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.state_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.candidate_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + CANDIDATE_NUMERIC_SIZE, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def masked_mean(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        total = (embeddings * weights).sum(dim=dim)
        count = weights.sum(dim=dim).clamp_min(1.0)
        return total / count.sqrt()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state_token_embeddings = self.embedding(batch["state_tokens"])
        state_embedding = self.masked_mean(
            state_token_embeddings,
            batch["state_mask"],
            dim=1,
        )
        policy_embedding = self.embedding(batch["policy_tokens"]).mean(dim=1)
        state = self.state_encoder(
            torch.cat(
                (state_embedding, policy_embedding, batch["state_numeric"]),
                dim=-1,
            )
        )
        candidate_token_embeddings = self.embedding(batch["candidate_tokens"])
        candidate_embedding = self.masked_mean(
            candidate_token_embeddings,
            batch["candidate_token_mask"],
            dim=2,
        )
        attention_logits = torch.einsum(
            "bod,bsd->bos",
            self.candidate_query(candidate_embedding),
            self.state_key(state_token_embeddings),
        ) / math.sqrt(self.embedding_dim)
        attention_logits = attention_logits.masked_fill(
            ~batch["state_mask"].unsqueeze(1),
            -30.0,
        )
        attention = torch.softmax(attention_logits, dim=-1)
        candidate_state = torch.einsum(
            "bos,bsd->bod",
            attention,
            state_token_embeddings,
        )
        candidate = self.candidate_encoder(
            torch.cat(
                (
                    candidate_embedding,
                    candidate_state,
                    batch["candidate_numeric"],
                ),
                dim=-1,
            )
        )
        expanded_state = state.unsqueeze(1).expand_as(candidate)
        combined = torch.cat(
            (
                expanded_state,
                candidate,
                expanded_state * candidate,
                torch.abs(expanded_state - candidate),
            ),
            dim=-1,
        )
        logits = self.scorer(combined).squeeze(-1)
        return logits.masked_fill(~batch["option_mask"], -30.0)


@dataclass
class TrainConfig:
    data: str
    output_dir: str
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    positive_weight: float
    embedding_dim: int
    hidden_dim: int
    dropout: float
    hash_size: int
    max_state_tokens: int
    max_candidate_tokens: int
    seed: int
    max_train_rows: int | None
    max_valid_rows: int | None
    target_accuracy: float
    use_trajectory_weights: bool
    deck_hashes: tuple[str, ...]
    team_names: tuple[str, ...]
    expected_train_rows: int | None
    split_mode: str


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def make_loader(
    config: TrainConfig,
    split: str,
    epoch: int,
    max_rows: int | None,
    shuffle_seed: int,
) -> DataLoader:
    dataset = ZipDecisionDataset(
        Path(config.data),
        split,
        max_rows,
        config.seed,
        shuffle_seed,
        epoch,
        config.max_state_tokens,
        config.max_candidate_tokens,
        config.use_trajectory_weights,
        config.deck_hashes,
        config.team_names,
        config.split_mode,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "num_workers": config.workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": lambda rows: collate_decisions(
            rows,
            config.max_state_tokens,
            config.max_candidate_tokens,
        ),
    }
    if config.workers:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
    return DataLoader(**kwargs)


def train_epoch(
    model: CandidatePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    positive_weight: float,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_decisions = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, cpu_batch in enumerate(loader, 1):
        batch = move_batch(cpu_batch, device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast:
            logits = model(batch)
            mask = batch["option_mask"]
            targets = batch["targets"]
            raw_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            class_weights = 1.0 + targets * (positive_weight - 1.0)
            decision_loss = (
                raw_loss * class_weights * mask
            ).sum(dim=1) / (class_weights * mask).sum(dim=1).clamp_min(1.0)
            fixed_single = (
                (batch["min_counts"] == 1)
                & (batch["max_counts"] == 1)
                & (targets.sum(dim=1) == 1)
            )
            if fixed_single.any():
                target_index = targets.argmax(dim=1)
                single_loss = F.cross_entropy(
                    logits,
                    target_index,
                    reduction="none",
                )
                decision_loss = torch.where(
                    fixed_single,
                    single_loss,
                    decision_loss,
                )
            loss = (decision_loss * batch["sample_weights"]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        decisions = int(mask.shape[0])
        total_decisions += decisions
        total_loss += float(loss.detach()) * decisions
        if step % 100 == 0:
            elapsed = max(time.time() - started, 1e-6)
            log(
                f"epoch={epoch} step={step} decisions={total_decisions} "
                f"loss={total_loss / total_decisions:.5f} "
                f"rate={total_decisions / elapsed:.0f}/s "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
    elapsed = time.time() - started
    return {
        "loss": total_loss / max(total_decisions, 1),
        "decisions": total_decisions,
        "seconds": elapsed,
        "decisions_per_second": total_decisions / max(elapsed, 1e-6),
    }


@torch.no_grad()
def evaluate(
    model: CandidatePolicy,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> dict[str, Any]:
    model.eval()
    totals = Counter()
    correct_by_threshold = Counter()
    fixed_by_threshold = Counter()
    flexible_by_threshold = Counter()
    context_total = Counter()
    context_correct: dict[float, Counter[int]] = {
        threshold: Counter() for threshold in thresholds
    }
    started = time.time()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast:
            logits = model(batch)
        mask = batch["option_mask"]
        targets = batch["targets"].bool()
        option_count = mask.sum(dim=1)
        mins = torch.minimum(batch["min_counts"], option_count)
        maxs = torch.minimum(batch["max_counts"], option_count)
        fixed = mins == maxs
        scores = logits.float().masked_fill(~mask, -1e9)
        ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)

        batch_size = mask.shape[0]
        totals["rows"] += batch_size
        totals["fixed"] += int(fixed.sum())
        totals["flexible"] += int((~fixed).sum())
        single = (mins == 1) & (maxs == 1)
        totals["single"] += int(single.sum())
        first_min = (
            torch.arange(mask.shape[1], device=device).unsqueeze(0)
            < mins.unsqueeze(1)
        ) & mask
        totals["first_min_correct"] += int(
            ((first_min == targets) | ~mask).all(dim=1).sum()
        )

        contexts_cpu = batch["contexts"].cpu()
        for context_value, count in zip(*torch.unique(contexts_cpu, return_counts=True)):
            context_total[int(context_value)] += int(count)

        probabilities = torch.sigmoid(scores)
        for threshold in thresholds:
            chosen_count = ((probabilities >= threshold) & mask).sum(dim=1)
            chosen_count = torch.maximum(chosen_count, mins)
            chosen_count = torch.minimum(chosen_count, maxs)
            prediction = (ranks < chosen_count.unsqueeze(1)) & mask
            exact = ((prediction == targets) | ~mask).all(dim=1)
            correct_by_threshold[threshold] += int(exact.sum())
            fixed_by_threshold[threshold] += int((exact & fixed).sum())
            flexible_by_threshold[threshold] += int((exact & ~fixed).sum())
            exact_cpu = exact.cpu()
            for context_value in torch.unique(contexts_cpu):
                context = int(context_value)
                context_correct[threshold][context] += int(
                    exact_cpu[contexts_cpu == context].sum()
                )

    best_threshold = max(
        thresholds,
        key=lambda threshold: correct_by_threshold[threshold],
    )
    rows = totals["rows"]
    by_threshold = {
        str(threshold): correct_by_threshold[threshold] / max(rows, 1)
        for threshold in thresholds
    }
    by_context = {
        str(context): {
            "rows": count,
            "exact_accuracy": context_correct[best_threshold][context] / count,
        }
        for context, count in context_total.most_common()
    }
    return {
        "rows": rows,
        "seconds": time.time() - started,
        "best_threshold": best_threshold,
        "exact_action_set_accuracy": correct_by_threshold[best_threshold]
        / max(rows, 1),
        "fixed_cardinality_accuracy": fixed_by_threshold[best_threshold]
        / max(totals["fixed"], 1),
        "flexible_cardinality_accuracy": flexible_by_threshold[best_threshold]
        / max(totals["flexible"], 1),
        "first_min_baseline_accuracy": totals["first_min_correct"] / max(rows, 1),
        "threshold_accuracy": by_threshold,
        "by_context": by_context,
    }


def read_manifest(data_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(data_path) as archive:
        return orjson.loads(archive.read("manifest.json"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/bc_recent7_top20.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bc_v1"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=2.5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE)
    parser.add_argument("--max-state-tokens", type=int, default=DEFAULT_STATE_TOKENS)
    parser.add_argument(
        "--max-candidate-tokens",
        type=int,
        default=DEFAULT_CANDIDATE_TOKENS,
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--target-accuracy", type=float, default=0.75)
    parser.add_argument(
        "--use-trajectory-weights",
        action="store_true",
        help="Downweight late losing decisions; off by default for exact imitation.",
    )
    parser.add_argument(
        "--deck-hash",
        action="append",
        default=[],
        help="Only train/evaluate rows from this deck hash; repeatable.",
    )
    parser.add_argument(
        "--expected-train-rows",
        type=int,
        help="Filtered train row count used to build the LR schedule.",
    )
    parser.add_argument(
        "--team-name",
        action="append",
        default=[],
        help="Only train/evaluate rows from this demonstrator; repeatable.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("archive", "episode_hash"),
        default="archive",
        help="Use stored time split or re-split whole episodes across all dates.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.hash_size != DEFAULT_HASH_SIZE:
        raise ValueError(
            f"This feature version requires --hash-size={DEFAULT_HASH_SIZE}"
        )

    config = TrainConfig(
        data=str(args.data.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_weight=args.positive_weight,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        hash_size=args.hash_size,
        max_state_tokens=args.max_state_tokens,
        max_candidate_tokens=args.max_candidate_tokens,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        target_accuracy=args.target_accuracy,
        use_trajectory_weights=args.use_trajectory_weights,
        deck_hashes=tuple(args.deck_hash),
        team_names=tuple(args.team_name),
        expected_train_rows=args.expected_train_rows,
        split_mode=args.split_mode,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest = read_manifest(args.data)
    expected_train = (
        args.expected_train_rows
        or args.max_train_rows
        or int(manifest["split_decisions"]["train"])
    )
    model = CandidatePolicy(
        config.hash_size,
        config.embedding_dim,
        config.hidden_dim,
        config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(expected_train / config.batch_size)
    total_steps = max(steps_per_epoch * config.epochs, 1)
    warmup_steps = max(int(total_steps * 0.03), 1)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step / warmup_steps, 0.05)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    thresholds = [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_info = {
        "feature_version": FEATURE_VERSION,
        "config": asdict(config),
        "manifest_dates": manifest.get("dates"),
        "manifest_splits": manifest.get("split_decisions"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "parameter_count": parameter_count,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        f"device={device} model_parameters={parameter_count:,} "
        f"train_rows={expected_train:,}"
    )

    best_accuracy = -1.0
    best_result: dict[str, Any] | None = None
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, config.epochs + 1):
            train_loader = make_loader(
                config,
                "train",
                epoch,
                config.max_train_rows,
                config.seed,
            )
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                config.positive_weight,
                epoch,
            )
            valid_loader = make_loader(
                config,
                "valid",
                0,
                config.max_valid_rows,
                config.seed + 17,
            )
            valid_metrics = evaluate(model, valid_loader, device, thresholds)
            result = {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
            }
            metrics_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            metrics_file.flush()
            accuracy = valid_metrics["exact_action_set_accuracy"]
            log(
                f"epoch={epoch} valid_exact={accuracy:.6f} "
                f"fixed={valid_metrics['fixed_cardinality_accuracy']:.6f} "
                f"flexible={valid_metrics['flexible_cardinality_accuracy']:.6f} "
                f"threshold={valid_metrics['best_threshold']}"
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_result = result
                torch.save(
                    {
                        "feature_version": FEATURE_VERSION,
                        "config": asdict(config),
                        "model_state_dict": model.state_dict(),
                        "threshold": valid_metrics["best_threshold"],
                        "epoch": epoch,
                        "valid_metrics": valid_metrics,
                    },
                    output_dir / "best.pt",
                )
            torch.save(
                {
                    "feature_version": FEATURE_VERSION,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                },
                output_dir / "last.pt",
            )

    assert best_result is not None
    summary = {
        "target_accuracy": config.target_accuracy,
        "target_reached": best_accuracy >= config.target_accuracy,
        "best_epoch": best_result["epoch"],
        "best_valid_exact_action_set_accuracy": best_accuracy,
        "best_threshold": best_result["valid"]["best_threshold"],
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "metrics": str(metrics_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
