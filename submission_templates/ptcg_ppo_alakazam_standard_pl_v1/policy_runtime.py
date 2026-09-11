#!/usr/bin/env python3
"""Orbit-style entity Transformer behavior cloning for PTCG.

The model consumes:

* one global decision token;
* one token per visible card/player-summary/recent-log entity;
* one token per currently legal action option.

It produces pointer logits over the dynamic legal options, a cardinality head
for flexible multi-select decisions, and a win-probability value head that can
be reused by PPO.  Fixed-cardinality decisions always obey minCount/maxCount;
flexible decisions select the predicted number of highest-scoring options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterator

try:
    import orjson
except ModuleNotFoundError:
    # Inference does not read the BC JSONL archive, but retaining a stdlib
    # fallback keeps this module importable in a minimal Kaggle runtime.
    orjson = json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


FEATURE_VERSION = "ptcg-bc-orbit-entity-transformer-v5"
DEFAULT_HASH_SIZE = 65_536
DEFAULT_MAX_STATE_ENTITIES = 80
DEFAULT_ENTITY_FIELDS = 20
DEFAULT_OPTION_FIELDS = 24
ENTITY_NUMERIC_SIZE = 12
OPTION_NUMERIC_SIZE = 12
GLOBAL_NUMERIC_SIZE = 16
MAX_ACTION_COUNT = 60

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


def bounded(value: Any, scale: float, limit: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-limit, min(limit, number / scale))


def stable_id(namespace: str, value: Any, hash_size: int) -> int:
    text = f"{namespace}={value}"
    return zlib.crc32(text.encode("utf-8")) % (hash_size - 1) + 1


def add_field(
    fields: list[int],
    namespace: str,
    value: Any,
    hash_size: int,
) -> None:
    if value is not None:
        fields.append(stable_id(namespace, value, hash_size))


def card_id(card: Any) -> int:
    if not isinstance(card, dict):
        return 0
    try:
        return int(card.get("id", 0) or 0)
    except (TypeError, ValueError):
        return 0


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
        values = current.get("stadium")
        return values if isinstance(values, list) else None
    if zone == "looking":
        values = current.get("looking")
        return values if isinstance(values, list) else None
    if zone == "deck" and select.get("deck") is not None:
        values = select.get("deck")
        return values if isinstance(values, list) else None
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


def card_entity(
    card: Any,
    rel: str,
    zone: str,
    position: int,
    hash_size: int,
    role: str = "board",
) -> dict[str, list[int] | list[float]]:
    fields: list[int] = []
    add_field(fields, "entity:kind", "card", hash_size)
    add_field(fields, "card:role", role, hash_size)
    add_field(fields, "card:owner", rel, hash_size)
    add_field(fields, "card:zone", zone, hash_size)
    add_field(fields, "card:owner_zone", f"{rel}:{zone}", hash_size)
    add_field(fields, "card:slot", min(position, 63), hash_size)
    cid = card_id(card)
    if not isinstance(card, dict):
        add_field(fields, "card:hidden", 1, hash_size)
        return {
            "fields": fields,
            "numeric": [0.0] * ENTITY_NUMERIC_SIZE,
        }

    add_field(fields, "card:id", cid, hash_size)
    add_field(fields, "card:id_zone", f"{cid}:{rel}:{zone}", hash_size)
    hp = float(card.get("hp", 0) or 0)
    max_hp = float(card.get("maxHp", 0) or 0)
    add_field(fields, "card:hp_bucket", max(int(hp), 0) // 30, hash_size)
    add_field(
        fields,
        "card:damage_bucket",
        max(int(max_hp - hp), 0) // 30,
        hash_size,
    )
    for flag in (
        "appearThisTurn",
        "poisoned",
        "burned",
        "asleep",
        "paralyzed",
        "confused",
    ):
        if flag in card:
            add_field(fields, f"card:{flag}", int(bool(card.get(flag))), hash_size)

    energies = card.get("energies") or []
    for energy in energies[:8]:
        add_field(fields, "card:energy_type", energy, hash_size)
    energy_cards = card.get("energyCards") or []
    tools = card.get("tools") or []
    evolutions = card.get("preEvolution") or []
    for attached in energy_cards[:6]:
        add_field(fields, "card:energy_card", card_id(attached), hash_size)
    for tool in tools[:4]:
        add_field(fields, "card:tool", card_id(tool), hash_size)
    for evolution in evolutions[:4]:
        add_field(fields, "card:pre_evolution", card_id(evolution), hash_size)

    numeric = [
        hp / max_hp if max_hp > 0 else 0.0,
        bounded(hp, 350),
        bounded(max_hp - hp, 350),
        bounded(len(energies), 8),
        bounded(len(energy_cards), 8),
        bounded(len(tools), 4),
        bounded(len(evolutions), 3),
        bounded(position, 60),
        float(bool(card.get("appearThisTurn"))),
        float(bool(card.get("poisoned") or card.get("burned"))),
        float(bool(card.get("asleep") or card.get("paralyzed"))),
        float(bool(card.get("confused"))),
    ]
    return {"fields": fields, "numeric": numeric}


def player_summary_entity(
    player: dict[str, Any],
    rel: str,
    hash_size: int,
) -> dict[str, list[int] | list[float]]:
    fields: list[int] = []
    add_field(fields, "entity:kind", "player_summary", hash_size)
    add_field(fields, "player:relation", rel, hash_size)
    for key, cap in (("deckCount", 60), ("handCount", 30), ("benchMax", 8)):
        value = min(int(player.get(key, 0) or 0), cap)
        add_field(fields, f"player:{rel}:{key}", value, hash_size)
    for flag in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        add_field(
            fields,
            f"player:{rel}:{flag}",
            int(bool(player.get(flag))),
            hash_size,
        )
    numeric = [
        bounded(player.get("deckCount", 0), 60),
        bounded(player.get("handCount", 0), 20),
        bounded(len(player.get("active") or []), 2),
        bounded(len(player.get("bench") or []), 5),
        bounded(len(player.get("discard") or []), 40),
        bounded(len(player.get("prize") or []), 6),
        bounded(player.get("benchMax", 5), 8),
        float(bool(player.get("poisoned"))),
        float(bool(player.get("burned"))),
        float(bool(player.get("asleep"))),
        float(bool(player.get("paralyzed"))),
        float(bool(player.get("confused"))),
    ]
    return {"fields": fields, "numeric": numeric}


def log_entity(
    event: dict[str, Any],
    recent_position: int,
    your_index: int,
    hash_size: int,
) -> dict[str, list[int] | list[float]]:
    fields: list[int] = []
    event_type = int(event.get("type", -1) or -1)
    player_index = int(event.get("playerIndex", -1) or -1)
    rel = relation(player_index, your_index) if player_index in (0, 1) else "none"
    add_field(fields, "entity:kind", "log", hash_size)
    add_field(fields, "log:type", event_type, hash_size)
    add_field(fields, "log:relation", rel, hash_size)
    add_field(fields, "log:type_relation", f"{event_type}:{rel}", hash_size)
    add_field(fields, "log:recent_slot", min(recent_position, 15), hash_size)
    for key in ("cardId", "attackId", "area", "index", "number", "count"):
        if event.get(key) is not None:
            add_field(fields, f"log:{key}", event.get(key), hash_size)
    numeric = [
        bounded(recent_position, 12),
        bounded(event_type, 64),
        bounded(event.get("cardId", 0), 4096),
        bounded(event.get("attackId", 0), 32),
        bounded(event.get("area", 0), 15),
        bounded(event.get("index", 0), 60),
        bounded(event.get("number", event.get("count", 0)), 30),
    ] + [0.0] * 5
    return {"fields": fields, "numeric": numeric}


def global_features(
    row: dict[str, Any],
    hash_size: int,
) -> tuple[list[int], list[float]]:
    observation = row["observation"]
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    select_type = int(select.get("type", row.get("select_type", 0)) or 0)
    context = int(select.get("context", row.get("select_context", 0)) or 0)
    min_count = int(select.get("minCount", row.get("min_count", 0)) or 0)
    max_count = int(select.get("maxCount", row.get("max_count", 0)) or 0)
    options = select.get("option") or []
    fields: list[int] = []
    for namespace, value in (
        ("global:select_type", select_type),
        ("global:context", context),
        ("global:type_context", f"{select_type}:{context}"),
        ("global:min_count", min_count),
        ("global:max_count", max_count),
        ("global:option_count", min(len(options), 63)),
        ("global:context_card", card_id(select.get("contextCard"))),
        ("global:effect_card", card_id(select.get("effect"))),
        ("global:first_relative", int(current.get("firstPlayer", 0) == your_index)),
        ("global:deck_hash", row.get("deck_hash", "")),
    ):
        add_field(fields, namespace, value, hash_size)
    for flag in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated"):
        add_field(fields, f"global:{flag}", int(bool(current.get(flag))), hash_size)

    players = current.get("players") or []
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
        bounded(len(self_player.get("bench") or []), 5),
        bounded(len(self_player.get("prize") or []), 6),
        bounded(opponent.get("deckCount", 0), 60),
        bounded(opponent.get("handCount", 0), 20),
        bounded(len(opponent.get("bench") or []), 5),
        bounded(len(opponent.get("prize") or []), 6),
        bounded(select.get("remainDamageCounter", 0), 30),
        bounded(select.get("remainEnergyCost", 0), 8),
        bounded(min_count, 10),
        bounded(max_count, 10),
        bounded(len(options), 40),
        float(current.get("firstPlayer", 0) == your_index),
    ]
    return fields, numeric


def state_entities(
    row: dict[str, Any],
    hash_size: int,
    max_entities: int,
) -> list[dict[str, list[int] | list[float]]]:
    observation = row["observation"]
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    entities: list[dict[str, list[int] | list[float]]] = []

    players = current.get("players") or []
    for player_index, player in enumerate(players[:2]):
        if not isinstance(player, dict):
            continue
        rel = relation(player_index, your_index)
        entities.append(player_summary_entity(player, rel, hash_size))
        for zone in ("hand", "active", "bench", "discard", "prize"):
            cards = player.get(zone)
            if not isinstance(cards, list):
                continue
            for position, card in enumerate(cards):
                entities.append(
                    card_entity(card, rel, zone, position, hash_size)
                )

    for zone in ("stadium", "looking"):
        cards = current.get(zone)
        if isinstance(cards, list):
            for position, card in enumerate(cards):
                entities.append(
                    card_entity(card, "public", zone, position, hash_size)
                )
    deck_cards = select.get("deck")
    if isinstance(deck_cards, list):
        for position, card in enumerate(deck_cards):
            entities.append(
                card_entity(
                    card,
                    "self",
                    "select_deck",
                    position,
                    hash_size,
                    role="revealed_choice",
                )
            )

    for role, key in (("context_card", "contextCard"), ("effect_card", "effect")):
        card = select.get(key)
        if isinstance(card, dict):
            entities.append(
                card_entity(card, "self", "select_context", 0, hash_size, role=role)
            )

    logs = observation.get("logs") or []
    for recent_position, event in enumerate(reversed(logs[-12:])):
        if isinstance(event, dict):
            entities.append(
                log_entity(event, recent_position, your_index, hash_size)
            )

    # Keep summaries, in-play/hand/revealed cards and newest logs first.
    return entities[:max_entities]


def option_features(
    row: dict[str, Any],
    option: dict[str, Any],
    option_position: int,
    hash_size: int,
) -> tuple[list[int], list[float]]:
    observation = row["observation"]
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    option_type = int(option.get("type", -1) or 0)
    context = int(select.get("context", row.get("select_context", 0)) or 0)
    select_type = int(select.get("type", row.get("select_type", 0)) or 0)
    fields: list[int] = []
    add_field(fields, "entity:kind", "legal_option", hash_size)
    add_field(fields, "option:type", option_type, hash_size)
    add_field(fields, "option:position", min(option_position, 63), hash_size)
    add_field(fields, "option:type_position", f"{option_type}:{min(option_position, 15)}", hash_size)
    add_field(fields, "option:context_type", f"{context}:{option_type}", hash_size)
    add_field(fields, "option:select_type", f"{select_type}:{option_type}", hash_size)
    for key, value in sorted(option.items()):
        if key != "serial":
            add_field(fields, f"option:{key}", value, hash_size)

    area = int(option.get("area", 0) or 0)
    index = int(option.get("index", -1) if option.get("index") is not None else -1)
    player_index = int(option.get("playerIndex", your_index) or 0)
    if option_type == 7:
        area, player_index = 2, your_index
    source = None
    if option_type in (3, 4, 5, 6, 7, 8, 9, 10, 11) and index >= 0:
        source = resolve_card(observation, area, index, player_index)
    target = None
    if option_type in (8, 9):
        target = resolve_card(
            observation,
            int(option.get("inPlayArea", 0) or 0),
            int(option.get("inPlayIndex", -1) or 0),
            your_index,
        )
    attached = None
    if option_type in (4, 5, 6) and isinstance(source, dict):
        attached_key = "tools" if option_type == 4 else "energyCards"
        index_key = "toolIndex" if option_type == 4 else "energyIndex"
        attached_values = source.get(attached_key) or []
        attached_index = int(option.get(index_key, -1) or 0)
        if 0 <= attached_index < len(attached_values):
            attached = attached_values[attached_index]

    source_id = card_id(source)
    target_id = card_id(target)
    attached_id = card_id(attached)
    for namespace, cid in (
        ("option:source_card", source_id),
        ("option:target_card", target_id),
        ("option:attached_card", attached_id),
    ):
        if cid:
            add_field(fields, namespace, cid, hash_size)
    if source_id:
        add_field(fields, "option:context_source", f"{context}:{source_id}", hash_size)
        add_field(fields, "option:type_source", f"{option_type}:{source_id}", hash_size)
    if target_id:
        add_field(fields, "option:source_target", f"{source_id}:{target_id}", hash_size)

    source_hp = float(source.get("hp", 0) or 0) if isinstance(source, dict) else 0.0
    source_max_hp = (
        float(source.get("maxHp", 0) or 0) if isinstance(source, dict) else 0.0
    )
    target_hp = float(target.get("hp", 0) or 0) if isinstance(target, dict) else 0.0
    target_max_hp = (
        float(target.get("maxHp", 0) or 0) if isinstance(target, dict) else 0.0
    )
    numeric = [
        bounded(option_position, 40),
        bounded(option_type, 16),
        bounded(area, 15),
        bounded(index, 60),
        bounded(option.get("playerIndex", your_index), 2),
        bounded(option.get("inPlayArea", 0), 15),
        bounded(option.get("inPlayIndex", 0), 8),
        bounded(option.get("attackId", 0), 32),
        bounded(option.get("number", option.get("count", 0)), 30),
        source_hp / source_max_hp if source_max_hp > 0 else 0.0,
        target_hp / target_max_hp if target_max_hp > 0 else 0.0,
        bounded(len(source.get("energies") or []) if isinstance(source, dict) else 0, 8),
    ]
    return fields, numeric


def featurize_row(
    row: dict[str, Any],
    hash_size: int,
    max_state_entities: int,
) -> dict[str, Any] | None:
    observation = row.get("observation") or {}
    select = observation.get("select")
    if not isinstance(select, dict):
        return None
    options = select.get("option") or []
    if not options:
        return None
    action = sorted({int(index) for index in row.get("action", [])})
    if any(index < 0 or index >= len(options) for index in action):
        return None
    if len(action) > MAX_ACTION_COUNT:
        return None

    global_fields, global_numeric = global_features(row, hash_size)
    entities = state_entities(row, hash_size, max_state_entities)
    option_fields: list[list[int]] = []
    option_numeric: list[list[float]] = []
    for position, option in enumerate(options):
        fields, numeric = option_features(row, option, position, hash_size)
        option_fields.append(fields)
        option_numeric.append(numeric)

    reward = float(row.get("terminal_reward", 0.0) or 0.0)
    return {
        "global_fields": global_fields,
        "global_numeric": global_numeric,
        "entity_fields": [entity["fields"] for entity in entities],
        "entity_numeric": [entity["numeric"] for entity in entities],
        "option_fields": option_fields,
        "option_numeric": option_numeric,
        "targets": [1.0 if i in action else 0.0 for i in range(len(options))],
        "action_count": len(action),
        "min_count": int(select.get("minCount", row.get("min_count", 0)) or 0),
        "max_count": int(select.get("maxCount", row.get("max_count", 0)) or 0),
        "context": int(select.get("context", row.get("select_context", 0)) or 0),
        "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
        "win_target": 1.0 if reward > 0 else 0.0,
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
        hash_size: int,
        max_state_entities: int,
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
        self.hash_size = hash_size
        self.max_state_entities = max_state_entities
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
        worker_count = worker.num_workers if worker else 1
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
            members = members[worker_id::worker_count]
            worker_limit = (
                math.ceil(self.max_rows / worker_count)
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
                            self.hash_size,
                            self.max_state_entities,
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
    max_state_entities: int,
    entity_fields: int,
    option_fields: int,
) -> dict[str, torch.Tensor]:
    batch_size = len(rows)
    max_options = max(len(row["targets"]) for row in rows)
    global_fields = torch.zeros((batch_size, entity_fields), dtype=torch.long)
    global_field_mask = torch.zeros((batch_size, entity_fields), dtype=torch.bool)
    global_numeric = torch.zeros((batch_size, GLOBAL_NUMERIC_SIZE))
    state_fields = torch.zeros(
        (batch_size, max_state_entities, entity_fields),
        dtype=torch.long,
    )
    state_field_mask = torch.zeros_like(state_fields, dtype=torch.bool)
    state_numeric = torch.zeros(
        (batch_size, max_state_entities, ENTITY_NUMERIC_SIZE)
    )
    state_mask = torch.zeros((batch_size, max_state_entities), dtype=torch.bool)
    option_field_tensor = torch.zeros(
        (batch_size, max_options, option_fields),
        dtype=torch.long,
    )
    option_field_mask = torch.zeros_like(option_field_tensor, dtype=torch.bool)
    option_numeric = torch.zeros((batch_size, max_options, OPTION_NUMERIC_SIZE))
    option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool)
    targets = torch.zeros((batch_size, max_options))
    action_counts = torch.zeros(batch_size, dtype=torch.long)
    min_counts = torch.zeros(batch_size, dtype=torch.long)
    max_counts = torch.zeros(batch_size, dtype=torch.long)
    contexts = torch.zeros(batch_size, dtype=torch.long)
    sample_weights = torch.ones(batch_size)
    win_targets = torch.zeros(batch_size)

    for batch_index, row in enumerate(rows):
        gf = row["global_fields"][:entity_fields]
        global_fields[batch_index, : len(gf)] = torch.tensor(gf)
        global_field_mask[batch_index, : len(gf)] = True
        global_numeric[batch_index] = torch.tensor(row["global_numeric"])

        entity_count = min(len(row["entity_fields"]), max_state_entities)
        state_mask[batch_index, :entity_count] = True
        for entity_index in range(entity_count):
            ef = row["entity_fields"][entity_index][:entity_fields]
            state_fields[batch_index, entity_index, : len(ef)] = torch.tensor(ef)
            state_field_mask[batch_index, entity_index, : len(ef)] = True
            state_numeric[batch_index, entity_index] = torch.tensor(
                row["entity_numeric"][entity_index]
            )

        option_count = len(row["targets"])
        option_mask[batch_index, :option_count] = True
        targets[batch_index, :option_count] = torch.tensor(row["targets"])
        for option_index, fields in enumerate(row["option_fields"]):
            fields = fields[:option_fields]
            option_field_tensor[
                batch_index, option_index, : len(fields)
            ] = torch.tensor(fields)
            option_field_mask[
                batch_index, option_index, : len(fields)
            ] = True
            option_numeric[batch_index, option_index] = torch.tensor(
                row["option_numeric"][option_index]
            )

        action_counts[batch_index] = row["action_count"]
        min_counts[batch_index] = row["min_count"]
        max_counts[batch_index] = row["max_count"]
        contexts[batch_index] = row["context"]
        sample_weights[batch_index] = row["sample_weight"]
        win_targets[batch_index] = row["win_target"]

    return {
        "global_fields": global_fields,
        "global_field_mask": global_field_mask,
        "global_numeric": global_numeric,
        "state_fields": state_fields,
        "state_field_mask": state_field_mask,
        "state_numeric": state_numeric,
        "state_mask": state_mask,
        "option_fields": option_field_tensor,
        "option_field_mask": option_field_mask,
        "option_numeric": option_numeric,
        "option_mask": option_mask,
        "targets": targets,
        "action_counts": action_counts,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "contexts": contexts,
        "sample_weights": sample_weights,
        "win_targets": win_targets,
    }


class EntityOptionPolicy(nn.Module):
    """Shared actor-critic entity Transformer suitable for BC then PPO."""

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

    @staticmethod
    def masked_field_mean(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        total = (embeddings * weights).sum(dim=-2)
        count = weights.sum(dim=-2).clamp_min(1.0)
        return total / count.sqrt()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        global_cat = self.masked_field_mean(
            self.embedding(batch["global_fields"]),
            batch["global_field_mask"],
        )
        global_token = self.global_encoder(
            torch.cat((global_cat, batch["global_numeric"]), dim=-1)
        )
        global_token = global_token + self.kind_embedding.weight[0]

        state_cat = self.masked_field_mean(
            self.embedding(batch["state_fields"]),
            batch["state_field_mask"],
        )
        state = self.entity_encoder(
            torch.cat((state_cat, batch["state_numeric"]), dim=-1)
        )
        state_positions = torch.arange(
            state.shape[1], device=state.device
        ).clamp_max(self.max_state_entities - 1)
        state = (
            state
            + self.kind_embedding.weight[1]
            + self.state_position(state_positions).unsqueeze(0)
        )

        option_cat = self.masked_field_mean(
            self.embedding(batch["option_fields"]),
            batch["option_field_mask"],
        )
        options = self.option_encoder(
            torch.cat((option_cat, batch["option_numeric"]), dim=-1)
        )
        option_positions = torch.arange(
            options.shape[1], device=options.device
        ).clamp_max(self.max_options - 1)
        options = (
            options
            + self.kind_embedding.weight[2]
            + self.option_position(option_positions).unsqueeze(0)
        )

        sequence = torch.cat((global_token.unsqueeze(1), state, options), dim=1)
        sequence_mask = torch.cat(
            (
                torch.ones(
                    (state.shape[0], 1),
                    dtype=torch.bool,
                    device=state.device,
                ),
                batch["state_mask"],
                batch["option_mask"],
            ),
            dim=1,
        )
        encoded = self.transformer(
            sequence,
            src_key_padding_mask=~sequence_mask,
        )
        global_encoded = encoded[:, 0]
        option_start = 1 + state.shape[1]
        option_encoded = encoded[:, option_start:]
        pointer = torch.einsum(
            "bd,bod->bo",
            self.actor_query(global_encoded),
            self.actor_key(option_encoded),
        ) / math.sqrt(self.model_dim)
        residual = self.actor_residual(
            torch.cat(
                (
                    option_encoded,
                    global_encoded.unsqueeze(1).expand_as(option_encoded),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        policy_logits = (pointer + residual).masked_fill(
            ~batch["option_mask"],
            -30.0,
        )
        return {
            "policy_logits": policy_logits,
            "count_logits": self.count_head(global_encoded),
            "value_logits": self.value_head(global_encoded).squeeze(-1),
        }


@dataclass
class TrainConfig:
    data: str
    output_dir: str
    epochs: int
    batch_size: int
    workers: int
    learning_rate: float
    weight_decay: float
    categorical_dim: int
    model_dim: int
    layers: int
    heads: int
    dropout: float
    hash_size: int
    max_state_entities: int
    entity_fields: int
    option_fields: int
    set_bce_weight: float
    count_loss_weight: float
    value_loss_weight: float
    seed: int
    max_train_rows: int | None
    max_valid_rows: int | None
    max_test_rows: int | None
    target_accuracy: float
    use_trajectory_weights: bool
    deck_hashes: tuple[str, ...]
    team_names: tuple[str, ...]
    expected_train_rows: int
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
        archive_path=Path(config.data),
        split=split,
        max_rows=max_rows,
        split_seed=config.seed,
        shuffle_seed=shuffle_seed,
        epoch=epoch,
        hash_size=config.hash_size,
        max_state_entities=config.max_state_entities,
        use_trajectory_weights=config.use_trajectory_weights,
        deck_hashes=config.deck_hashes,
        team_names=config.team_names,
        split_mode=config.split_mode,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.workers,
        collate_fn=partial(
            collate_decisions,
            max_state_entities=config.max_state_entities,
            entity_fields=config.entity_fields,
            option_fields=config.option_fields,
        ),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if config.workers else None,
    )


def masked_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs["policy_logits"]
    mask = batch["option_mask"]
    targets = batch["targets"]
    weights = batch["sample_weights"]
    action_counts = batch["action_counts"]

    log_probs = F.log_softmax(logits, dim=-1)
    normalized_targets = targets / action_counts.clamp_min(1).unsqueeze(1)
    pointer_per_row = -(normalized_targets * log_probs).sum(dim=1)
    pointer_active = action_counts > 0
    pointer_loss = (
        pointer_per_row[pointer_active] * weights[pointer_active]
    ).sum() / weights[pointer_active].sum().clamp_min(1.0)

    bce_raw = F.binary_cross_entropy_with_logits(
        logits.masked_fill(~mask, 0.0),
        targets,
        reduction="none",
    )
    bce_per_row = (bce_raw * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    set_bce_loss = (bce_per_row * weights).sum() / weights.sum().clamp_min(1.0)

    flexible = batch["min_counts"] != batch["max_counts"]
    count_ce = F.cross_entropy(
        outputs["count_logits"],
        action_counts.clamp_max(MAX_ACTION_COUNT),
        reduction="none",
    )
    count_loss = (
        (count_ce[flexible] * weights[flexible]).sum()
        / weights[flexible].sum().clamp_min(1.0)
        if flexible.any()
        else count_ce.sum() * 0.0
    )

    value_raw = F.binary_cross_entropy_with_logits(
        outputs["value_logits"],
        batch["win_targets"],
        reduction="none",
    )
    value_loss = (value_raw * weights).sum() / weights.sum().clamp_min(1.0)
    total = (
        pointer_loss
        + config.set_bce_weight * set_bce_loss
        + config.count_loss_weight * count_loss
        + config.value_loss_weight * value_loss
    )
    return total, {
        "pointer": pointer_loss,
        "set_bce": set_bce_loss,
        "count": count_loss,
        "value": value_loss,
    }


def allowed_count_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    count_values = torch.arange(
        MAX_ACTION_COUNT + 1,
        device=batch["min_counts"].device,
    ).unsqueeze(0)
    option_counts = batch["option_mask"].sum(dim=1).unsqueeze(1)
    minimum = batch["min_counts"].clamp(0, MAX_ACTION_COUNT).unsqueeze(1)
    maximum = torch.minimum(
        batch["max_counts"].clamp(0, MAX_ACTION_COUNT).unsqueeze(1),
        option_counts,
    )
    return (count_values >= minimum) & (count_values <= maximum)


def predict_action_mask(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed = batch["min_counts"] == batch["max_counts"]
    count_logits = outputs["count_logits"].masked_fill(
        ~allowed_count_mask(batch),
        -30.0,
    )
    predicted_counts = count_logits.argmax(dim=1)
    predicted_counts = torch.where(
        fixed,
        batch["min_counts"].clamp(0, MAX_ACTION_COUNT),
        predicted_counts,
    )
    predicted_counts = torch.minimum(
        predicted_counts,
        batch["option_mask"].sum(dim=1),
    )
    logits = outputs["policy_logits"]
    ranks = torch.argsort(
        torch.argsort(logits, dim=1, descending=True),
        dim=1,
    )
    prediction = (
        ranks < predicted_counts.unsqueeze(1)
    ) & batch["option_mask"]
    return prediction, predicted_counts


def train_epoch(
    model: EntityOptionPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    started = time.time()
    rows = 0
    correct = 0
    loss_total = 0.0
    component_totals: Counter[str] = Counter()
    for step, cpu_batch in enumerate(loader, start=1):
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(batch)
            loss, components = masked_losses(outputs, batch, config)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            prediction, _ = predict_action_mask(outputs, batch)
            exact = (
                (prediction == batch["targets"].bool()) | ~batch["option_mask"]
            ).all(dim=1)
        batch_rows = batch["targets"].shape[0]
        rows += batch_rows
        correct += int(exact.sum())
        loss_total += float(loss.detach()) * batch_rows
        for name, value in components.items():
            component_totals[name] += float(value.detach()) * batch_rows
        if step % 100 == 0:
            elapsed = time.time() - started
            log(
                f"epoch={epoch} step={step} rows={rows:,} "
                f"loss={loss_total / rows:.5f} exact={correct / rows:.5f} "
                f"rows_per_s={rows / max(elapsed, 1):.0f}"
            )
    return {
        "rows": rows,
        "seconds": time.time() - started,
        "loss": loss_total / max(rows, 1),
        "exact_action_set_accuracy": correct / max(rows, 1),
        "loss_components": {
            name: total / max(rows, 1)
            for name, total in component_totals.items()
        },
    }


@torch.no_grad()
def evaluate(
    model: EntityOptionPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    started = time.time()
    totals: Counter[str] = Counter()
    context_total: Counter[int] = Counter()
    context_correct: Counter[int] = Counter()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(batch)
        prediction, predicted_counts = predict_action_mask(outputs, batch)
        exact = (
            (prediction == batch["targets"].bool()) | ~batch["option_mask"]
        ).all(dim=1)
        fixed = batch["min_counts"] == batch["max_counts"]
        flexible = ~fixed
        count_correct = predicted_counts == batch["action_counts"]
        nonempty = batch["action_counts"] > 0
        top1 = outputs["policy_logits"].argmax(dim=1)
        top1_correct = batch["targets"].gather(1, top1.unsqueeze(1)).squeeze(1).bool()

        totals["rows"] += exact.numel()
        totals["correct"] += int(exact.sum())
        totals["fixed"] += int(fixed.sum())
        totals["fixed_correct"] += int((exact & fixed).sum())
        totals["flexible"] += int(flexible.sum())
        totals["flexible_correct"] += int((exact & flexible).sum())
        totals["count_correct"] += int(count_correct.sum())
        totals["nonempty"] += int(nonempty.sum())
        totals["top1_correct"] += int((top1_correct & nonempty).sum())
        value_prediction = outputs["value_logits"] >= 0
        value_target = batch["win_targets"].bool()
        totals["value_correct"] += int((value_prediction == value_target).sum())

        contexts_cpu = batch["contexts"].cpu()
        exact_cpu = exact.cpu()
        for context_value in torch.unique(contexts_cpu):
            context = int(context_value)
            selected = contexts_cpu == context
            context_total[context] += int(selected.sum())
            context_correct[context] += int(exact_cpu[selected].sum())

    rows = totals["rows"]
    return {
        "rows": rows,
        "seconds": time.time() - started,
        "exact_action_set_accuracy": totals["correct"] / max(rows, 1),
        "fixed_cardinality_accuracy": totals["fixed_correct"]
        / max(totals["fixed"], 1),
        "flexible_cardinality_accuracy": totals["flexible_correct"]
        / max(totals["flexible"], 1),
        "count_accuracy": totals["count_correct"] / max(rows, 1),
        "nonempty_top1_accuracy": totals["top1_correct"]
        / max(totals["nonempty"], 1),
        "value_win_accuracy": totals["value_correct"] / max(rows, 1),
        "by_context": {
            str(context): {
                "rows": count,
                "exact_accuracy": context_correct[context] / count,
            }
            for context, count in context_total.most_common()
        },
    }


def read_manifest(data_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(data_path) as archive:
        return orjson.loads(archive.read("manifest.json"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: EntityOptionPolicy,
    config: TrainConfig,
    epoch: int,
    valid_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "feature_version": FEATURE_VERSION,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "ppo_interface": {
                "actor": "policy_logits over option_mask",
                "cardinality": "count_logits masked to minCount..maxCount",
                "critic": "sigmoid(value_logits) predicts terminal win probability",
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/bc_recent7_top20.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bc_orbit_v5"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--categorical-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hash-size", type=int, default=DEFAULT_HASH_SIZE)
    parser.add_argument(
        "--max-state-entities",
        type=int,
        default=DEFAULT_MAX_STATE_ENTITIES,
    )
    parser.add_argument("--entity-fields", type=int, default=DEFAULT_ENTITY_FIELDS)
    parser.add_argument("--option-fields", type=int, default=DEFAULT_OPTION_FIELDS)
    parser.add_argument("--set-bce-weight", type=float, default=0.25)
    parser.add_argument("--count-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    parser.add_argument("--target-accuracy", type=float, default=0.75)
    parser.add_argument("--use-trajectory-weights", action="store_true")
    parser.add_argument("--deck-hash", action="append", default=[])
    parser.add_argument("--team-name", action="append", default=[])
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument(
        "--split-mode",
        choices=("archive", "episode_hash"),
        default="archive",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.model_dim % args.heads:
        raise ValueError("--model-dim must be divisible by --heads")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")

    manifest = read_manifest(args.data)
    expected_train_rows = (
        args.expected_train_rows
        or args.max_train_rows
        or int(manifest["split_decisions"]["train"])
    )
    config = TrainConfig(
        data=str(args.data.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        categorical_dim=args.categorical_dim,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        hash_size=args.hash_size,
        max_state_entities=args.max_state_entities,
        entity_fields=args.entity_fields,
        option_fields=args.option_fields,
        set_bce_weight=args.set_bce_weight,
        count_loss_weight=args.count_loss_weight,
        value_loss_weight=args.value_loss_weight,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        max_test_rows=args.max_test_rows,
        target_accuracy=args.target_accuracy,
        use_trajectory_weights=args.use_trajectory_weights,
        deck_hashes=tuple(args.deck_hash),
        team_names=tuple(args.team_name),
        expected_train_rows=expected_train_rows,
        split_mode=args.split_mode,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model = EntityOptionPolicy(
        hash_size=config.hash_size,
        categorical_dim=config.categorical_dim,
        model_dim=config.model_dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        max_state_entities=config.max_state_entities,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(expected_train_rows / config.batch_size)
    total_steps = max(steps_per_epoch * config.epochs, 1)
    warmup_steps = max(int(total_steps * 0.03), 1)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step / warmup_steps, 0.05)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda",
    )
    parameter_count = sum(p.numel() for p in model.parameters())
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
        f"expected_train_rows={expected_train_rows:,}"
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
                config.seed + epoch,
            )
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                config,
                epoch,
            )
            valid_loader = make_loader(
                config,
                "valid",
                0,
                config.max_valid_rows,
                config.seed + 17,
            )
            valid_metrics = evaluate(model, valid_loader, device)
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
                f"count={valid_metrics['count_accuracy']:.6f} "
                f"top1={valid_metrics['nonempty_top1_accuracy']:.6f}"
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_result = result
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    config,
                    epoch,
                    valid_metrics,
                )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                config,
                epoch,
                valid_metrics,
            )

    test_metrics = None
    if not args.skip_test:
        checkpoint = torch.load(
            output_dir / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loader = make_loader(
            config,
            "test",
            0,
            config.max_test_rows,
            config.seed + 29,
        )
        test_metrics = evaluate(model, test_loader, device)
        log(
            f"test_exact={test_metrics['exact_action_set_accuracy']:.6f} "
            f"fixed={test_metrics['fixed_cardinality_accuracy']:.6f} "
            f"flexible={test_metrics['flexible_cardinality_accuracy']:.6f} "
            f"count={test_metrics['count_accuracy']:.6f}"
        )

    summary = {
        "target_accuracy": config.target_accuracy,
        "target_reached_on_valid": best_accuracy >= config.target_accuracy,
        "target_reached_on_test": (
            test_metrics is not None
            and test_metrics["exact_action_set_accuracy"] >= config.target_accuracy
        ),
        "best_epoch": best_result["epoch"] if best_result else None,
        "best_valid_metrics": best_result["valid"] if best_result else None,
        "test_metrics": test_metrics,
        "best_checkpoint": str(output_dir / "best.pt"),
        "parameter_count": parameter_count,
        "ppo_ready_architecture": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
