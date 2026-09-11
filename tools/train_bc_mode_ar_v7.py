#!/usr/bin/env python3
"""Mode-aware autoregressive BC for PTCG dynamic legal actions.

This trainer deliberately keeps the existing v6 feature contract while
replacing the static multi-hot/top-k actor with four explicit decode modes:

* single: exactly one option;
* optional_single: one option or STOP;
* ordered_sequence: expert order is supervised autoregressively;
* unordered_set: a canonical ascending order removes permutation ambiguity.

The board is encoded independently, legal options cross-attend to that board,
and a GRU pointer decoder conditions every next choice on prior choices.  The
terminal value head is detached from the actor trunk; cardinality is auxiliary
only and never controls decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_bc_orbit as base


FEATURE_VERSION = "ptcg-bc-mode-ar-pointer-v7"
DATA_FEATURE_VERSIONS = {
    "ptcg-bc-orbit-entity-transformer-v6-priority",
}
ORDERED_CONTEXT = 34
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


def log(message: str) -> None:
    print(message, flush=True)


def action_mode_ids(
    contexts: torch.Tensor,
    min_counts: torch.Tensor,
    max_counts: torch.Tensor,
) -> torch.Tensor:
    """Route each decision without pretending every multi-select is ordered."""

    modes = torch.full_like(contexts, MODE_UNORDERED_SET)
    single = (min_counts == 1) & (max_counts == 1)
    optional = (min_counts == 0) & (max_counts == 1)
    ordered = (contexts == ORDERED_CONTEXT) & (max_counts > 1)
    modes[single] = MODE_SINGLE
    modes[optional] = MODE_OPTIONAL_SINGLE
    modes[ordered] = MODE_ORDERED_SEQUENCE
    return modes


def canonical_action_sequences(
    batch: dict[str, torch.Tensor],
    modes: torch.Tensor,
) -> torch.Tensor:
    """Use raw replay order only where the game semantics say order matters."""

    targets = batch["targets"].bool() & batch["option_mask"]
    option_count = targets.shape[1]
    indices = torch.arange(option_count, device=targets.device).unsqueeze(0)
    canonical = torch.where(
        targets,
        indices,
        torch.full_like(indices, option_count),
    ).sort(dim=1).values
    canonical = canonical.masked_fill(canonical == option_count, -1)
    raw = batch["action_sequences"][:, :option_count]
    if raw.shape[1] < option_count:
        raw = F.pad(raw, (0, option_count - raw.shape[1]), value=-1)
    ordered = modes == MODE_ORDERED_SEQUENCE
    return torch.where(ordered.unsqueeze(1), raw, canonical)


def validate_action_targets(
    batch: dict[str, torch.Tensor],
    modes: torch.Tensor,
    sequences: torch.Tensor,
) -> None:
    target_counts = (batch["targets"].bool() & batch["option_mask"]).sum(1)
    if not torch.equal(target_counts, batch["action_counts"]):
        raise ValueError("action_count differs from the target set cardinality")
    sequence_counts = (sequences >= 0).sum(1)
    if not torch.equal(sequence_counts, batch["action_counts"]):
        raise ValueError("action sequence length differs from action_count")
    ordered = modes == MODE_ORDERED_SEQUENCE
    if ordered.any():
        sequence_counts = torch.zeros_like(batch["targets"], dtype=torch.long)
        valid = sequences >= 0
        safe = sequences.clamp_min(0)
        sequence_counts.scatter_add_(1, safe, valid.long())
        sequence_set = sequence_counts > 0
        expected = batch["targets"].bool() & batch["option_mask"]
        if not torch.equal(sequence_set[ordered], expected[ordered]):
            raise ValueError("ordered action_sequence differs from target set")


def collate_mode_ar(
    rows: list[dict[str, Any]],
    *,
    max_state_entities: int,
    entity_fields: int,
    option_fields: int,
    date_to_id: dict[str, int],
    daily_weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    batch = base.collate_decisions(
        rows,
        max_state_entities=max_state_entities,
        entity_fields=entity_fields,
        option_fields=option_fields,
    )
    dates = [str(row.get("dataset_date", "")) for row in rows]
    unknown = sorted(set(dates) - set(date_to_id))
    if unknown:
        raise ValueError(f"batch contains unknown dataset_date values: {unknown}")
    batch["dataset_date_ids"] = torch.tensor(
        [date_to_id[value] for value in dates],
        dtype=torch.long,
    )
    batch["sample_weights"] *= torch.tensor(
        [daily_weights.get(value, 1.0) for value in dates],
        dtype=torch.float32,
    )
    return batch


class ModeAwareARPolicy(nn.Module):
    """Board Transformer + option cross-attention + conditional pointer GRU."""

    def __init__(
        self,
        *,
        hash_size: int = base.DEFAULT_HASH_SIZE,
        categorical_dim: int = 64,
        model_dim: int = 192,
        layers: int = 4,
        heads: int = 6,
        dropout: float = 0.05,
        max_state_entities: int = base.DEFAULT_MAX_STATE_ENTITIES,
        max_options: int = 128,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
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
            nn.Linear(categorical_dim + base.GLOBAL_NUMERIC_SIZE, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(categorical_dim + base.ENTITY_NUMERIC_SIZE, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(categorical_dim + base.OPTION_NUMERIC_SIZE, model_dim),
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
        self.board_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.option_cross_attention = nn.MultiheadAttention(
            model_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.option_cross_norm = nn.LayerNorm(model_dim)
        self.option_ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.option_ffn_norm = nn.LayerNorm(model_dim)

        self.context_embedding = nn.Embedding(256, model_dim // 4)
        self.count_embedding = nn.Embedding(
            base.MAX_ACTION_COUNT + 1,
            model_dim // 4,
        )
        self.mode_embedding = nn.Embedding(len(MODE_NAMES), model_dim // 4)
        select_width = model_dim * 2 + (model_dim // 4) * 4
        self.select_encoder = nn.Sequential(
            nn.Linear(select_width, model_dim * 2),
            nn.LayerNorm(model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.bos = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.bos, std=0.02)
        self.decoder = nn.GRUCell(model_dim, model_dim)
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.option_key = nn.Linear(model_dim, model_dim, bias=False)
        self.pointer_residual = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, base.MAX_ACTION_COUNT + 1),
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

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_cat = self.masked_field_mean(
            self.embedding(batch["global_fields"]),
            batch["global_field_mask"],
        )
        global_token = self.global_encoder(
            torch.cat((global_cat, batch["global_numeric"]), dim=-1)
        ) + self.kind_embedding.weight[0]

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

        board = torch.cat((global_token.unsqueeze(1), state), dim=1)
        board_mask = torch.cat(
            (
                torch.ones(
                    (state.shape[0], 1),
                    dtype=torch.bool,
                    device=state.device,
                ),
                batch["state_mask"],
            ),
            dim=1,
        )
        board = self.board_transformer(
            board,
            src_key_padding_mask=~board_mask,
        )
        global_encoded = board[:, 0]

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
        cross, _ = self.option_cross_attention(
            query=options,
            key=board,
            value=board,
            key_padding_mask=~board_mask,
            need_weights=False,
        )
        options = self.option_cross_norm(options + cross)
        options = self.option_ffn_norm(options + self.option_ffn(options))
        options = options.masked_fill(
            ~batch["option_mask"].unsqueeze(-1),
            0.0,
        )
        option_weights = batch["option_mask"].to(options.dtype).unsqueeze(-1)
        option_pool = (options * option_weights).sum(1) / option_weights.sum(1).clamp_min(1)

        modes = action_mode_ids(
            batch["contexts"],
            batch["min_counts"],
            batch["max_counts"],
        )
        context_ids = batch["contexts"].clamp(0, 255)
        min_ids = batch["min_counts"].clamp(0, base.MAX_ACTION_COUNT)
        max_ids = batch["max_counts"].clamp(0, base.MAX_ACTION_COUNT)
        select_summary = self.select_encoder(
            torch.cat(
                (
                    global_encoded,
                    option_pool,
                    self.context_embedding(context_ids),
                    self.count_embedding(min_ids),
                    self.count_embedding(max_ids),
                    self.mode_embedding(modes),
                ),
                dim=-1,
            )
        )
        return {
            "board_memory": board,
            "board_mask": board_mask,
            "option_vectors": options,
            "select_summary": select_summary,
            "modes": modes,
            "count_logits": self.count_head(select_summary),
            # The critic cannot distort the shared policy representation.
            "value_predictions": self.value_head(
                select_summary.detach()
            ).squeeze(-1),
        }

    def decoder_step(
        self,
        previous: torch.Tensor,
        hidden: torch.Tensor,
        option_vectors: torch.Tensor,
        select_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.decoder(previous, hidden)
        query = self.query(hidden)
        pointer = torch.einsum(
            "bd,bod->bo",
            query,
            self.option_key(option_vectors),
        ) / math.sqrt(self.model_dim)
        residual = self.pointer_residual(
            torch.cat(
                (
                    option_vectors,
                    hidden.unsqueeze(1).expand_as(option_vectors),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        stop = self.stop_head(
            torch.cat((hidden, select_summary), dim=-1)
        ).squeeze(-1)
        return hidden, pointer + residual, stop

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encode(batch)
        previous = self.bos.unsqueeze(0).expand(batch["targets"].shape[0], -1)
        hidden, logits, stop = self.decoder_step(
            previous,
            encoded["select_summary"],
            encoded["option_vectors"],
            encoded["select_summary"],
        )
        encoded.update(
            {
                "first_hidden": hidden,
                "policy_logits": logits.masked_fill(
                    ~batch["option_mask"], -30.0
                ),
                "stop_logits": stop,
            }
        )
        return encoded


def legal_decode_mask(
    batch: dict[str, torch.Tensor],
    modes: torch.Tensor,
    selected: torch.Tensor,
    step: int,
    previous_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    option_mask = batch["option_mask"] & ~selected
    option_count = batch["option_mask"].sum(1)
    below_max = step < torch.minimum(batch["max_counts"], option_count)
    option_mask &= below_max.unsqueeze(1)

    unordered = modes == MODE_UNORDERED_SET
    indices = torch.arange(option_mask.shape[1], device=option_mask.device).unsqueeze(0)
    canonical = indices > previous_indices.unsqueeze(1)
    remaining_needed = (batch["min_counts"] - (step + 1)).clamp_min(0)
    later_options = option_count.unsqueeze(1) - indices - 1
    reachable = later_options >= remaining_needed.unsqueeze(1)
    option_mask &= (~unordered).unsqueeze(1) | (canonical & reachable)

    stop_legal = (
        (modes != MODE_SINGLE)
        & (step >= batch["min_counts"])
        & (step <= batch["max_counts"])
    )
    return option_mask, stop_legal


def autoregressive_losses(
    model: ModeAwareARPolicy,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    count_loss_weight: float,
    value_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    modes = encoded["modes"]
    sequences = canonical_action_sequences(batch, modes)
    validate_action_targets(batch, modes, sequences)
    batch_size, option_width = batch["targets"].shape
    option_count = batch["option_mask"].sum(1)
    effective_max = torch.minimum(batch["max_counts"], option_count)
    needs_stop = (modes != MODE_SINGLE) & (
        batch["action_counts"] < effective_max
    )
    learned_steps = batch["action_counts"] + needs_stop.long()
    maximum_steps = int(learned_steps.max().item())
    if maximum_steps < 1:
        raise ValueError("batch contains no supervised decoder step")

    previous = model.bos.unsqueeze(0).expand(batch_size, -1)
    hidden = encoded["select_summary"]
    selected = torch.zeros_like(batch["option_mask"])
    previous_indices = torch.full(
        (batch_size,), -1, dtype=torch.long, device=previous.device
    )
    per_row_nll = torch.zeros(batch_size, device=previous.device)
    per_row_tokens = torch.zeros(batch_size, device=previous.device)
    for step in range(maximum_steps):
        hidden, option_logits, stop_logits = model.decoder_step(
            previous,
            hidden,
            encoded["option_vectors"],
            encoded["select_summary"],
        )
        legal_options, stop_legal = legal_decode_mask(
            batch,
            modes,
            selected,
            step,
            previous_indices,
        )
        logits = torch.cat((option_logits, stop_logits.unsqueeze(1)), dim=1)
        legal = torch.cat((legal_options, stop_legal.unsqueeze(1)), dim=1)
        logits = logits.masked_fill(~legal, -1e9)
        active = step < learned_steps
        action_step = step < batch["action_counts"]
        target = torch.where(
            action_step,
            sequences[:, step].clamp_min(0),
            torch.full_like(batch["action_counts"], option_width),
        )
        chosen_legal = legal.gather(1, target.unsqueeze(1)).squeeze(1)
        if not bool(chosen_legal[active].all()):
            raise ValueError("expert target violates mode-specific decode constraints")
        nll = F.cross_entropy(logits, target, reduction="none")
        per_row_nll += nll * active
        per_row_tokens += active

        chosen_action = active & action_step
        safe_target = target.clamp_max(option_width - 1)
        selected |= F.one_hot(
            safe_target, num_classes=option_width
        ).bool() & chosen_action.unsqueeze(1)
        gathered = encoded["option_vectors"].gather(
            1,
            safe_target.view(batch_size, 1, 1).expand(-1, 1, model.model_dim),
        ).squeeze(1)
        previous = torch.where(chosen_action.unsqueeze(1), gathered, previous)
        previous_indices = torch.where(chosen_action, safe_target, previous_indices)

    row_weights = batch["sample_weights"]
    ar_per_row = per_row_nll / per_row_tokens.clamp_min(1)
    ar_loss = (ar_per_row * row_weights).sum() / row_weights.sum().clamp_min(1)

    count_targets = batch["action_counts"].clamp_max(base.MAX_ACTION_COUNT)
    count_raw = F.cross_entropy(
        encoded["count_logits"], count_targets, reduction="none"
    )
    count_loss = (count_raw * row_weights).sum() / row_weights.sum().clamp_min(1)
    signed_outcomes = batch["win_targets"] * 2.0 - 1.0
    value_raw = F.mse_loss(
        encoded["value_predictions"], signed_outcomes, reduction="none"
    )
    value_loss = (value_raw * row_weights).sum() / row_weights.sum().clamp_min(1)
    total = ar_loss + count_loss_weight * count_loss + value_loss_weight * value_loss
    return total, {
        "autoregressive": ar_loss,
        "cardinality_aux": count_loss,
        "detached_value_aux": value_loss,
    }


@torch.no_grad()
def greedy_decode(
    model: ModeAwareARPolicy,
    encoded: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    modes = encoded["modes"]
    batch_size, option_width = batch["targets"].shape
    option_count = batch["option_mask"].sum(1)
    effective_max = torch.minimum(batch["max_counts"], option_count)
    maximum_steps = max(int(effective_max.max().item()), 1)
    sequences = torch.full(
        (batch_size, maximum_steps),
        -1,
        dtype=torch.long,
        device=option_count.device,
    )
    selected = torch.zeros_like(batch["option_mask"])
    previous_indices = torch.full(
        (batch_size,), -1, dtype=torch.long, device=option_count.device
    )
    previous = model.bos.unsqueeze(0).expand(batch_size, -1)
    hidden = encoded["select_summary"]
    finished = effective_max == 0
    counts = torch.zeros(batch_size, dtype=torch.long, device=option_count.device)
    for step in range(maximum_steps):
        hidden, option_logits, stop_logits = model.decoder_step(
            previous,
            hidden,
            encoded["option_vectors"],
            encoded["select_summary"],
        )
        legal_options, stop_legal = legal_decode_mask(
            batch,
            modes,
            selected,
            step,
            previous_indices,
        )
        legal_options &= ~finished.unsqueeze(1)
        stop_legal &= ~finished
        logits = torch.cat((option_logits, stop_logits.unsqueeze(1)), dim=1)
        legal = torch.cat((legal_options, stop_legal.unsqueeze(1)), dim=1)
        # Finished rows are dummy-routed to STOP to keep argmax defined.
        legal[finished, option_width] = True
        choice = logits.masked_fill(~legal, -1e9).argmax(1)
        chose_stop = choice == option_width
        chose_action = ~finished & ~chose_stop
        safe_choice = choice.clamp_max(option_width - 1)
        sequences[:, step] = torch.where(
            chose_action,
            safe_choice,
            torch.full_like(safe_choice, -1),
        )
        selected |= F.one_hot(
            safe_choice, num_classes=option_width
        ).bool() & chose_action.unsqueeze(1)
        gathered = encoded["option_vectors"].gather(
            1,
            safe_choice.view(batch_size, 1, 1).expand(-1, 1, model.model_dim),
        ).squeeze(1)
        previous = torch.where(chose_action.unsqueeze(1), gathered, previous)
        previous_indices = torch.where(chose_action, safe_choice, previous_indices)
        counts += chose_action.long()
        finished |= chose_stop | (counts >= effective_max)
        if bool(finished.all()):
            break
    return {"sequences": sequences, "selected": selected, "counts": counts}


def semantic_exact(
    decoded: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    modes: torch.Tensor,
) -> torch.Tensor:
    set_exact = (
        (decoded["selected"] == batch["targets"].bool())
        | ~batch["option_mask"]
    ).all(1)
    target_sequence = canonical_action_sequences(batch, modes)
    width = max(decoded["sequences"].shape[1], target_sequence.shape[1])
    predicted = F.pad(
        decoded["sequences"],
        (0, width - decoded["sequences"].shape[1]),
        value=-1,
    )
    target_sequence = F.pad(
        target_sequence,
        (0, width - target_sequence.shape[1]),
        value=-1,
    )
    sequence_exact = (predicted == target_sequence).all(1)
    return torch.where(modes == MODE_ORDERED_SEQUENCE, sequence_exact, set_exact)


def choice_bearing_rows(
    batch: dict[str, torch.Tensor],
    modes: torch.Tensor,
) -> torch.Tensor:
    """Exclude decisions whose legal constraints force the only outcome."""

    option_count = batch["option_mask"].sum(1)
    forced_single = (modes == MODE_SINGLE) & (option_count == 1)
    fixed = batch["min_counts"] == batch["max_counts"]
    forced_set = (
        (modes == MODE_UNORDERED_SET)
        & fixed
        & (
            (batch["min_counts"] == 0)
            | (batch["min_counts"] == option_count)
        )
    )
    return ~(forced_single | forced_set)


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def scan_date_split_counts(data_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}

    def add(dataset_date: str, split: str) -> None:
        counts.setdefault(dataset_date, Counter())[split] += 1

    if data_path.is_dir():
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet date audit requires pyarrow") from error
        for member in sorted(data_path.rglob("*.parquet")):
            parquet_file = pq.ParquetFile(member)
            for batch in parquet_file.iter_batches(
                batch_size=65_536,
                columns=["dataset_date", "split"],
            ):
                dates = batch.column(0).to_pylist()
                splits = batch.column(1).to_pylist()
                for dataset_date, split in zip(dates, splits):
                    add(str(dataset_date), str(split))
    else:
        with zipfile.ZipFile(data_path) as archive:
            members = sorted(
                name for name in archive.namelist() if name.endswith(".jsonl")
            )
            for member in members:
                with archive.open(member) as handle:
                    for line in handle:
                        row = orjson.loads(line)
                        add(str(row.get("dataset_date", "")), str(row.get("split", "")))
    return {
        dataset_date: dict(split_counts)
        for dataset_date, split_counts in sorted(counts.items())
    }


def daily_equal_weights(
    date_split_counts: dict[str, dict[str, int]],
) -> dict[str, float]:
    train_counts = {
        dataset_date: int(splits.get("train", 0))
        for dataset_date, splits in date_split_counts.items()
        if int(splits.get("train", 0)) > 0
    }
    if not train_counts:
        raise ValueError("dataset has no train rows")
    total = sum(train_counts.values())
    date_count = len(train_counts)
    weights = {
        dataset_date: total / (date_count * rows)
        for dataset_date, rows in train_counts.items()
    }
    weighted_mean = sum(
        train_counts[dataset_date] * weights[dataset_date]
        for dataset_date in train_counts
    ) / total
    if not math.isclose(weighted_mean, 1.0, abs_tol=1e-12):
        raise RuntimeError("daily equal weights are not train-row mean normalized")
    return weights


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
    count_loss_weight: float
    value_loss_weight: float
    seed: int
    max_train_rows: int | None
    max_valid_rows: int | None
    expected_train_rows: int
    train_shuffle_buffer_rows_per_worker: int
    use_trajectory_weights: bool
    daily_equal_weighting: bool
    dates: tuple[str, ...]
    date_split_counts: dict[str, dict[str, int]]
    daily_weights: dict[str, float]


def make_loader(
    config: TrainConfig,
    split: str,
    epoch: int,
    max_rows: int | None,
) -> DataLoader:
    data_path = Path(config.data)
    common = {
        "split": split,
        "max_rows": max_rows,
        "split_seed": config.seed,
        "shuffle_seed": config.seed + epoch,
        "epoch": epoch,
        "use_trajectory_weights": config.use_trajectory_weights,
        "deck_hashes": (),
        "team_names": (),
        "select_contexts": (),
        "split_mode": "archive",
        "shuffle_buffer_rows": (
            config.train_shuffle_buffer_rows_per_worker if split == "train" else 0
        ),
        "policy_team_weights": None,
    }
    if data_path.is_dir():
        dataset = base.ParquetDecisionDataset(dataset_dir=data_path, **common)
    else:
        dataset = base.ZipDecisionDataset(
            archive_path=data_path,
            hash_size=config.hash_size,
            max_state_entities=config.max_state_entities,
            **common,
        )
    date_to_id = {value: index for index, value in enumerate(config.dates)}
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.workers,
        collate_fn=partial(
            collate_mode_ar,
            max_state_entities=config.max_state_entities,
            entity_fields=config.entity_fields,
            option_fields=config.option_fields,
            date_to_id=date_to_id,
            daily_weights=config.daily_weights,
        ),
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if config.workers else None,
    )


def metric_update(
    totals: Counter[str],
    exact: torch.Tensor,
    modes: torch.Tensor,
    date_ids: torch.Tensor,
    choice_rows: torch.Tensor,
    by_mode: dict[int, Counter[str]],
    by_date: dict[int, Counter[str]],
) -> None:
    totals["rows"] += exact.numel()
    totals["correct"] += int(exact.sum())
    totals["choice_rows"] += int(choice_rows.sum())
    totals["choice_correct"] += int((exact & choice_rows).sum())
    for mode in torch.unique(modes):
        key = int(mode)
        selected = modes == mode
        by_mode.setdefault(key, Counter())["rows"] += int(selected.sum())
        by_mode[key]["correct"] += int(exact[selected].sum())
        by_mode[key]["choice_rows"] += int((choice_rows & selected).sum())
        by_mode[key]["choice_correct"] += int(
            (exact & choice_rows & selected).sum()
        )
    for date_id in torch.unique(date_ids):
        key = int(date_id)
        selected = date_ids == date_id
        by_date.setdefault(key, Counter())["rows"] += int(selected.sum())
        by_date[key]["correct"] += int(exact[selected].sum())
        by_date[key]["choice_rows"] += int((choice_rows & selected).sum())
        by_date[key]["choice_correct"] += int(
            (exact & choice_rows & selected).sum()
        )


def finalize_metrics(
    totals: Counter[str],
    by_mode: dict[int, Counter[str]],
    by_date: dict[int, Counter[str]],
    dates: tuple[str, ...],
    started: float,
) -> dict[str, Any]:
    date_metrics = {
        dates[date_id]: {
            "rows": values["rows"],
            "exact_accuracy": values["correct"] / max(values["rows"], 1),
            "choice_rows": values["choice_rows"],
            "choice_exact_accuracy": values["choice_correct"]
            / max(values["choice_rows"], 1),
        }
        for date_id, values in sorted(by_date.items())
    }
    macro = sum(value["exact_accuracy"] for value in date_metrics.values()) / max(
        len(date_metrics), 1
    )
    macro_choice = sum(
        value["choice_exact_accuracy"] for value in date_metrics.values()
    ) / max(len(date_metrics), 1)
    return {
        "rows": totals["rows"],
        "seconds": time.time() - started,
        "semantic_exact_accuracy": totals["correct"] / max(totals["rows"], 1),
        "choice_bearing_rows": totals["choice_rows"],
        "choice_bearing_exact_accuracy": totals["choice_correct"]
        / max(totals["choice_rows"], 1),
        "daily_macro_exact_accuracy": macro,
        "daily_macro_choice_accuracy": macro_choice,
        "by_mode": {
            MODE_NAMES[mode]: {
                "rows": values["rows"],
                "exact_accuracy": values["correct"] / max(values["rows"], 1),
                "choice_rows": values["choice_rows"],
                "choice_exact_accuracy": values["choice_correct"]
                / max(values["choice_rows"], 1),
            }
            for mode, values in sorted(by_mode.items())
        },
        "by_date": date_metrics,
    }


def train_epoch(
    model: ModeAwareARPolicy,
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
    totals: Counter[str] = Counter()
    by_mode: dict[int, Counter[str]] = {}
    by_date: dict[int, Counter[str]] = {}
    component_totals: Counter[str] = Counter()
    loss_total = 0.0
    for step, cpu_batch in enumerate(loader, start=1):
        batch = move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            encoded = model(batch)
            loss, components = autoregressive_losses(
                model,
                encoded,
                batch,
                count_loss_weight=config.count_loss_weight,
                value_loss_weight=config.value_loss_weight,
            )
            with torch.no_grad():
                decoded = greedy_decode(model, encoded, batch)
                exact = semantic_exact(decoded, batch, encoded["modes"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_rows = exact.numel()
        metric_update(
            totals,
            exact,
            encoded["modes"],
            batch["dataset_date_ids"],
            choice_bearing_rows(batch, encoded["modes"]),
            by_mode,
            by_date,
        )
        loss_total += float(loss.detach()) * batch_rows
        for name, value in components.items():
            component_totals[name] += float(value.detach()) * batch_rows
        if step % 100 == 0:
            elapsed = time.time() - started
            log(
                f"epoch={epoch} step={step} rows={totals['rows']:,} "
                f"loss={loss_total / totals['rows']:.5f} "
                f"exact={totals['correct'] / totals['rows']:.5f} "
                f"rows_per_s={totals['rows'] / max(elapsed, 1):.0f}"
            )
    metrics = finalize_metrics(totals, by_mode, by_date, config.dates, started)
    metrics["loss"] = loss_total / max(totals["rows"], 1)
    metrics["loss_components"] = {
        name: value / max(totals["rows"], 1)
        for name, value in component_totals.items()
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: ModeAwareARPolicy,
    loader: DataLoader,
    device: torch.device,
    dates: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    started = time.time()
    totals: Counter[str] = Counter()
    by_mode: dict[int, Counter[str]] = {}
    by_date: dict[int, Counter[str]] = {}
    count_correct = 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            encoded = model(batch)
            decoded = greedy_decode(model, encoded, batch)
        exact = semantic_exact(decoded, batch, encoded["modes"])
        metric_update(
            totals,
            exact,
            encoded["modes"],
            batch["dataset_date_ids"],
            choice_bearing_rows(batch, encoded["modes"]),
            by_mode,
            by_date,
        )
        count_correct += int((decoded["counts"] == batch["action_counts"]).sum())
    metrics = finalize_metrics(totals, by_mode, by_date, dates, started)
    metrics["decoded_count_accuracy"] = count_correct / max(totals["rows"], 1)
    return metrics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: ModeAwareARPolicy,
    config: TrainConfig,
    epoch: int,
    valid_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "feature_version": FEATURE_VERSION,
            "data_feature_version": base.FEATURE_VERSION,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "action_contract": {
                "ordered_context": ORDERED_CONTEXT,
                "modes": MODE_NAMES,
                "decoder": "GRU pointer with explicit STOP",
                "unordered_canonicalization": "ascending legal option index",
                "cardinality_head_controls_decode": False,
                "value_head_actor_trunk_gradient": False,
            },
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--categorical-dim", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hash-size", type=int, default=base.DEFAULT_HASH_SIZE)
    parser.add_argument(
        "--max-state-entities", type=int, default=base.DEFAULT_MAX_STATE_ENTITIES
    )
    parser.add_argument("--entity-fields", type=int, default=base.DEFAULT_ENTITY_FIELDS)
    parser.add_argument("--option-fields", type=int, default=base.DEFAULT_OPTION_FIELDS)
    parser.add_argument("--count-loss-weight", type=float, default=0.05)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-valid-rows", type=int)
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument(
        "--train-shuffle-buffer-rows-per-worker", type=int, default=65_536
    )
    parser.add_argument("--use-trajectory-weights", action="store_true")
    parser.add_argument(
        "--no-daily-equal-weighting",
        action="store_true",
        help="Disable train-row inverse-date weights",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = args.data.resolve()
    if not data.exists():
        raise FileNotFoundError(data)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.count_loss_weight < 0 or args.value_loss_weight < 0:
        raise ValueError("auxiliary loss weights must be non-negative")

    manifest = base.read_manifest(data)
    data_feature_version = str(manifest.get("feature_version", base.FEATURE_VERSION))
    if data.is_dir() and data_feature_version not in DATA_FEATURE_VERSIONS:
        raise ValueError(
            f"unsupported Parquet feature version: {data_feature_version!r}"
        )
    date_split_counts = scan_date_split_counts(data)
    dates = tuple(sorted(date_split_counts))
    if not dates:
        raise ValueError("dataset has no dated rows")
    observed_train_rows = sum(
        values.get("train", 0) for values in date_split_counts.values()
    )
    expected_train_rows = args.expected_train_rows or args.max_train_rows or observed_train_rows
    if args.expected_train_rows is not None and args.expected_train_rows != observed_train_rows:
        raise ValueError(
            "--expected-train-rows differs from audited train rows: "
            f"{args.expected_train_rows} != {observed_train_rows}"
        )
    weights = daily_equal_weights(date_split_counts)
    if args.no_daily_equal_weighting:
        weights = {dataset_date: 1.0 for dataset_date in dates}
    missing_valid = [
        dataset_date
        for dataset_date, values in date_split_counts.items()
        if values.get("valid", 0) == 0
    ]
    if missing_valid:
        raise ValueError(f"dates without validation rows: {missing_valid}")

    config = TrainConfig(
        data=str(data),
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
        count_loss_weight=args.count_loss_weight,
        value_loss_weight=args.value_loss_weight,
        seed=args.seed,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        expected_train_rows=expected_train_rows,
        train_shuffle_buffer_rows_per_worker=args.train_shuffle_buffer_rows_per_worker,
        use_trajectory_weights=args.use_trajectory_weights,
        daily_equal_weighting=not args.no_daily_equal_weighting,
        dates=dates,
        date_split_counts=date_split_counts,
        daily_weights=weights,
    )
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = ModeAwareARPolicy(
        hash_size=config.hash_size,
        categorical_dim=config.categorical_dim,
        model_dim=config.model_dim,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        max_state_entities=config.max_state_entities,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_info = {
        "feature_version": FEATURE_VERSION,
        "data_feature_version": data_feature_version,
        "config": asdict(config),
        "parameter_count": parameter_count,
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "validation_contract": {
            "episode_disjoint": True,
            "primary_metric": "daily_macro_choice_accuracy",
            "secondary_metrics": [
                "choice_bearing_exact_accuracy",
                "daily_macro_exact_accuracy",
                "semantic_exact_accuracy",
            ],
            "ordered_exact_uses_sequence": True,
            "other_modes_exact_use_action_set": True,
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"device={device} parameters={parameter_count:,} "
        f"dates={dates[0]}..{dates[-1]} train_rows={observed_train_rows:,} "
        f"epochs={config.epochs} daily_equal={config.daily_equal_weighting}"
    )

    best_macro = -1.0
    best_result: dict[str, Any] | None = None
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, config.epochs + 1):
            train_loader = make_loader(
                config, "train", epoch, config.max_train_rows
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
            if config.max_train_rows is None and train_metrics["rows"] != observed_train_rows:
                raise RuntimeError(
                    "training row audit failed: "
                    f"{train_metrics['rows']} != {observed_train_rows}"
                )
            valid_loader = make_loader(
                config, "valid", 0, config.max_valid_rows
            )
            valid_metrics = evaluate(model, valid_loader, device, config.dates)
            result = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
            metrics_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            metrics_file.flush()
            log(
                f"epoch={epoch} valid_macro_choice={valid_metrics['daily_macro_choice_accuracy']:.6f} "
                f"valid_choice={valid_metrics['choice_bearing_exact_accuracy']:.6f} "
                f"valid_micro={valid_metrics['semantic_exact_accuracy']:.6f} "
                f"count={valid_metrics['decoded_count_accuracy']:.6f}"
            )
            if valid_metrics["daily_macro_choice_accuracy"] > best_macro:
                best_macro = valid_metrics["daily_macro_choice_accuracy"]
                best_result = result
                save_checkpoint(output_dir / "best.pt", model, config, epoch, valid_metrics)
            save_checkpoint(output_dir / "last.pt", model, config, epoch, valid_metrics)

    summary = {
        "best": best_result,
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(output_dir / "best.pt"),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "last_checkpoint_sha256": file_sha256(output_dir / "last.pt"),
        "target_80_percent_reached": bool(
            best_result
            and best_result["valid"]["daily_macro_choice_accuracy"] >= 0.80
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
