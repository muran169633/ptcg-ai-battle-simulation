#!/usr/bin/env python3
"""Evaluate a BC or PPO policy checkpoint on a streamed BC archive split.

The evaluator deliberately keeps the raw expert action sequence in addition to
the set target produced by ``train_bc_orbit.featurize_row``.  This makes the two
notions of correctness explicit:

* set exact: predicted and expert action indices form the same set;
* ordered exact: predicted and expert action index lists are identical.
* hybrid-order exact: preserve policy order only for ``SKILL_ORDER`` context
  34 and canonicalize all other predicted action lists to ascending order.

By default results are printed as JSON and no artifact is modified.  Pass
``--json-output`` only when a persistent machine-readable report is desired.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import zipfile
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import orjson
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import train_bc_orbit as bc  # noqa: E402
import train_ppo as ppo  # noqa: E402


SKILL_ORDER_CONTEXT = 34


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bc_feature_source(path: Path) -> Any:
    """Load an explicitly pinned historical BC featurizer module."""

    spec = importlib.util.spec_from_file_location(
        "evaluate_policy_bc_feature_source",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load BC feature source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def option_equivalence_ids(row: dict[str, Any]) -> list[int]:
    """Group options that produce the same visible game-state operation.

    Physical ``serial`` and zone ``index`` values distinguish interchangeable
    copies in the replay label but not their game effect.  IDs are local to a
    decision row and are used only for the reported semantic-exact metric; the
    historical strict index metrics remain unchanged.
    """

    observation = row.get("observation") or {}
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", row.get("seat", 0)) or 0)
    options = select.get("option") or []
    classes: dict[bytes, int] = {}
    result: list[int] = []

    def normalized_card(card: Any) -> Any:
        if not isinstance(card, dict):
            return None
        return {
            key: value
            for key, value in card.items()
            if key != "serial"
        }

    for option in options:
        option_type = int(option.get("type", -1) or 0)
        area = int(option.get("area", 0) or 0)
        index = int(
            option.get("index", -1)
            if option.get("index") is not None
            else -1
        )
        player_index = int(option.get("playerIndex", your_index) or 0)
        if option_type == 7:
            area, player_index = 2, your_index
        source = (
            bc.resolve_card(observation, area, index, player_index)
            if option_type in (3, 4, 5, 6, 7, 8, 9, 10, 11)
            and index >= 0
            else None
        )
        target = (
            bc.resolve_card(
                observation,
                int(option.get("inPlayArea", 0) or 0),
                int(option.get("inPlayIndex", -1) or 0),
                your_index,
            )
            if option_type in (8, 9)
            else None
        )
        normalized_option = {
            key: value
            for key, value in option.items()
            if key
            not in {"index", "inPlayIndex", "serial"}
        }
        signature = orjson.dumps(
            (normalized_option, normalized_card(source), normalized_card(target)),
            option=orjson.OPT_SORT_KEYS,
        )
        if signature not in classes:
            classes[signature] = len(classes)
        result.append(classes[signature])
    return result


def checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    feature_version = str(checkpoint.get("feature_version", ""))
    if feature_version == bc.FEATURE_VERSION:
        return "bc"
    if (
        feature_version == ppo.PPO_FEATURE_VERSION
        or checkpoint.get("bc_feature_version") == bc.FEATURE_VERSION
    ):
        return "ppo"
    raise ValueError(
        "Unsupported checkpoint feature_version "
        f"{feature_version!r}; expected {bc.FEATURE_VERSION!r} or "
        f"{ppo.PPO_FEATURE_VERSION!r}"
    )


def model_config_from_checkpoint(
    checkpoint: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    if kind == "bc":
        return ppo.checkpoint_model_config(checkpoint)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("PPO checkpoint is missing model_config")
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
    missing = [key for key in required if key not in model_config]
    if missing:
        raise ValueError(f"PPO model_config is missing keys: {missing}")
    return {
        key: (
            float(model_config[key])
            if key == "dropout"
            else int(model_config[key])
        )
        for key in required
    }


def instantiate_ppo_checkpoint(
    checkpoint: dict[str, Any],
    model_config: dict[str, Any],
    device: torch.device,
) -> bc.EntityOptionPolicy:
    model = bc.EntityOptionPolicy(
        hash_size=model_config["hash_size"],
        categorical_dim=model_config["categorical_dim"],
        model_dim=model_config["model_dim"],
        layers=model_config["layers"],
        heads=model_config["heads"],
        dropout=model_config["dropout"],
        max_state_entities=model_config["max_state_entities"],
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict")
    count_weight = state_dict.get("count_head.2.weight")
    if not isinstance(count_weight, torch.Tensor) or count_weight.ndim != 2:
        raise ValueError("Checkpoint has an unexpected count_head.2.weight")
    count_classes = int(count_weight.shape[0])
    count_layer = model.count_head[-1]
    if not isinstance(count_layer, nn.Linear):
        raise TypeError("Unexpected EntityOptionPolicy count head")
    if count_layer.out_features != count_classes:
        model.count_head[-1] = nn.Linear(count_layer.in_features, count_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[bc.EntityOptionPolicy, dict[str, Any], dict[str, Any], str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    kind = checkpoint_kind(checkpoint)
    model_config = model_config_from_checkpoint(checkpoint, kind)
    if kind == "bc":
        # Match the live PPO runtime: preserve learned count classes 0..16 and
        # conservatively expand the engine-facing head to 0..60.
        model = ppo.instantiate_model_from_bc(checkpoint, device)
    else:
        model = instantiate_ppo_checkpoint(checkpoint, model_config, device)
    return model, model_config, checkpoint, kind


class OrderedZipDecisionDataset(IterableDataset):
    """Stream decision rows from zip members without materializing the split."""

    def __init__(
        self,
        archive_path: Path,
        split: str,
        split_mode: str,
        split_seed: int,
        hash_size: int,
        max_state_entities: int,
        deck_hashes: tuple[str, ...],
        team_names: tuple[str, ...],
        dataset_dates: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.archive_path = archive_path
        self.split = split
        self.split_mode = split_mode
        self.split_seed = split_seed
        self.hash_size = hash_size
        self.max_state_entities = max_state_entities
        self.deck_hashes = set(deck_hashes)
        self.team_names = set(team_names)
        self.dataset_dates = set(dataset_dates)

    def row_split(self, row: dict[str, Any]) -> str:
        if self.split_mode == "archive":
            return str(row.get("split", ""))
        episode_id = str(row.get("episode_id", ""))
        digest = hashlib.sha256(
            f"{self.split_seed}:{episode_id}".encode()
        ).digest()
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
        with zipfile.ZipFile(self.archive_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.endswith(".jsonl")
                and (
                    self.split_mode == "episode_hash"
                    or name.startswith(f"{self.split}/")
                )
            )
            for member in members[worker_id::worker_count]:
                with archive.open(member) as handle:
                    for line in handle:
                        row = orjson.loads(line)
                        if self.row_split(row) != self.split:
                            continue
                        if (
                            self.deck_hashes
                            and str(row.get("deck_hash", ""))
                            not in self.deck_hashes
                        ):
                            continue
                        if (
                            self.team_names
                            and str(row.get("team_name", ""))
                            not in self.team_names
                        ):
                            continue
                        if (
                            self.dataset_dates
                            and str(row.get("dataset_date", ""))
                            not in self.dataset_dates
                        ):
                            continue
                        raw_action = row.get("action", [])
                        if not isinstance(raw_action, list):
                            continue
                        try:
                            expert_order = [int(index) for index in raw_action]
                        except (TypeError, ValueError):
                            continue

                        # featurize_row historically capped BC at 16 actions.
                        # Evaluation supports the PPO/engine 0..60 head without
                        # changing the training module on disk.
                        previous_max = bc.MAX_ACTION_COUNT
                        bc.MAX_ACTION_COUNT = ppo.MAX_ACTION_COUNT
                        try:
                            features = bc.featurize_row(
                                row,
                                self.hash_size,
                                self.max_state_entities,
                            )
                        finally:
                            bc.MAX_ACTION_COUNT = previous_max
                        if features is None:
                            continue
                        options = (
                            ((row.get("observation") or {}).get("select") or {})
                            .get("option")
                            or []
                        )
                        if (
                            len(expert_order) > ppo.MAX_ACTION_COUNT
                            or len(set(expert_order)) != len(expert_order)
                            or any(
                                index < 0 or index >= len(options)
                                for index in expert_order
                            )
                        ):
                            continue
                        features["expert_action_order"] = expert_order
                        features["option_equivalence_ids"] = (
                            option_equivalence_ids(row)
                        )
                        yield features


def collate_ordered(
    rows: list[dict[str, Any]],
    max_state_entities: int,
    entity_fields: int,
    option_fields: int,
) -> dict[str, torch.Tensor]:
    batch = bc.collate_decisions(
        rows,
        max_state_entities=max_state_entities,
        entity_fields=entity_fields,
        option_fields=option_fields,
    )
    ordered_actions = torch.full(
        (len(rows), ppo.MAX_ACTION_COUNT),
        -1,
        dtype=torch.long,
    )
    ordered_action_counts = torch.zeros(len(rows), dtype=torch.long)
    equivalence_ids = torch.full(
        (len(rows), batch["option_mask"].shape[1]),
        -1,
        dtype=torch.long,
    )
    for row_index, row in enumerate(rows):
        action = row["expert_action_order"]
        ordered_action_counts[row_index] = len(action)
        if action:
            ordered_actions[row_index, : len(action)] = torch.tensor(action)
        row_equivalence_ids = row["option_equivalence_ids"]
        equivalence_ids[row_index, : len(row_equivalence_ids)] = torch.tensor(
            row_equivalence_ids,
            dtype=torch.long,
        )
    batch["expert_ordered_actions"] = ordered_actions
    batch["expert_ordered_action_counts"] = ordered_action_counts
    batch["option_equivalence_ids"] = equivalence_ids
    return batch


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_counter(counter: Counter[str]) -> dict[str, Any]:
    set_accuracy = ratio(
        counter["set_exact_correct"],
        counter["rows"],
    )
    ordered_accuracy = ratio(
        counter["ordered_exact_correct"],
        counter["rows"],
    )
    hybrid_accuracy = ratio(
        counter["hybrid_order_exact_correct"],
        counter["rows"],
    )
    fixed_accuracy = ratio(
        counter["fixed_set_exact_correct"],
        counter["fixed_rows"],
    )
    flexible_accuracy = ratio(
        counter["flexible_set_exact_correct"],
        counter["flexible_rows"],
    )
    return {
        "rows": counter["rows"],
        "set_exact_correct": counter["set_exact_correct"],
        "set_exact_accuracy": set_accuracy,
        "exact_action_set_accuracy": set_accuracy,
        "semantic_set_exact_correct": counter["semantic_set_exact_correct"],
        "semantic_set_exact_accuracy": ratio(
            counter["semantic_set_exact_correct"],
            counter["rows"],
        ),
        "ordered_exact_correct": counter["ordered_exact_correct"],
        "ordered_exact_accuracy": ordered_accuracy,
        "ordered_action_exact_accuracy": ordered_accuracy,
        "hybrid_order_exact_correct": counter["hybrid_order_exact_correct"],
        "hybrid_order_exact_accuracy": hybrid_accuracy,
        "fixed_rows": counter["fixed_rows"],
        "fixed_cardinality_accuracy": fixed_accuracy,
        "fixed_set_exact_accuracy": fixed_accuracy,
        "fixed_ordered_exact_accuracy": ratio(
            counter["fixed_ordered_exact_correct"],
            counter["fixed_rows"],
        ),
        "fixed_hybrid_order_exact_accuracy": ratio(
            counter["fixed_hybrid_order_exact_correct"],
            counter["fixed_rows"],
        ),
        "flexible_rows": counter["flexible_rows"],
        "flexible_cardinality_accuracy": flexible_accuracy,
        "flexible_set_exact_accuracy": flexible_accuracy,
        "flexible_ordered_exact_accuracy": ratio(
            counter["flexible_ordered_exact_correct"],
            counter["flexible_rows"],
        ),
        "flexible_hybrid_order_exact_accuracy": ratio(
            counter["flexible_hybrid_order_exact_correct"],
            counter["flexible_rows"],
        ),
        "count_correct": counter["count_correct"],
        "count_accuracy": ratio(
            counter["count_correct"],
            counter["rows"],
        ),
        "nonempty_rows": counter["nonempty_rows"],
        "top1_correct": counter["top1_correct"],
        "nonempty_top1_accuracy": ratio(
            counter["top1_correct"],
            counter["nonempty_rows"],
        ),
        "value_correct": counter["value_correct"],
        "value_win_accuracy": ratio(
            counter["value_correct"],
            counter["rows"],
        ),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.total: Counter[str] = Counter()
        self.by_context: dict[int, Counter[str]] = defaultdict(Counter)

    @staticmethod
    def update_counter(
        counter: Counter[str],
        *,
        set_exact: bool,
        semantic_set_exact: bool,
        ordered_exact: bool,
        hybrid_order_exact: bool,
        fixed: bool,
        count_correct: bool,
        nonempty: bool,
        top1_correct: bool,
        value_correct: bool,
    ) -> None:
        counter["rows"] += 1
        counter["set_exact_correct"] += int(set_exact)
        counter["semantic_set_exact_correct"] += int(semantic_set_exact)
        counter["ordered_exact_correct"] += int(ordered_exact)
        counter["hybrid_order_exact_correct"] += int(hybrid_order_exact)
        cardinality = "fixed" if fixed else "flexible"
        counter[f"{cardinality}_rows"] += 1
        counter[f"{cardinality}_set_exact_correct"] += int(set_exact)
        counter[f"{cardinality}_ordered_exact_correct"] += int(ordered_exact)
        counter[f"{cardinality}_hybrid_order_exact_correct"] += int(
            hybrid_order_exact
        )
        counter["count_correct"] += int(count_correct)
        counter["nonempty_rows"] += int(nonempty)
        counter["top1_correct"] += int(top1_correct and nonempty)
        counter["value_correct"] += int(value_correct)

    def update(
        self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        actions: list[list[int]],
        policy_actions: list[list[int]],
    ) -> None:
        targets = batch["targets"].bool().cpu()
        option_mask = batch["option_mask"].bool().cpu()
        action_counts = batch["action_counts"].cpu()
        min_counts = batch["min_counts"].cpu()
        max_counts = batch["max_counts"].cpu()
        contexts = batch["contexts"].cpu()
        expert_actions = batch["expert_ordered_actions"].cpu()
        expert_counts = batch["expert_ordered_action_counts"].cpu()
        equivalence_ids = batch["option_equivalence_ids"].cpu()
        top1 = outputs["policy_logits"].argmax(dim=1).cpu()
        value_predictions = (outputs["value_logits"] >= 0).cpu()
        value_targets = batch["win_targets"].bool().cpu()

        prediction = torch.zeros_like(targets)
        for row_index, action in enumerate(actions):
            if action:
                prediction[row_index, action] = True
        set_exact_values = (
            (prediction == targets) | ~option_mask
        ).all(dim=1)

        for row_index, action in enumerate(actions):
            expert_count = int(expert_counts[row_index])
            expert_order = expert_actions[
                row_index,
                :expert_count,
            ].tolist()
            context = int(contexts[row_index])
            hybrid_action = (
                policy_actions[row_index]
                if context == SKILL_ORDER_CONTEXT
                else sorted(policy_actions[row_index])
            )
            set_exact = bool(set_exact_values[row_index])
            predicted_equivalence = sorted(
                int(equivalence_ids[row_index, index]) for index in action
            )
            expert_equivalence = sorted(
                int(equivalence_ids[row_index, index])
                for index in expert_order
            )
            semantic_set_exact = predicted_equivalence == expert_equivalence
            ordered_exact = action == expert_order
            hybrid_order_exact = hybrid_action == expert_order
            fixed = bool(min_counts[row_index] == max_counts[row_index])
            count_correct = len(action) == int(action_counts[row_index])
            nonempty = int(action_counts[row_index]) > 0
            top1_correct = bool(
                targets[row_index, int(top1[row_index])]
            )
            value_correct = bool(
                value_predictions[row_index] == value_targets[row_index]
            )
            values = {
                "set_exact": set_exact,
                "semantic_set_exact": semantic_set_exact,
                "ordered_exact": ordered_exact,
                "hybrid_order_exact": hybrid_order_exact,
                "fixed": fixed,
                "count_correct": count_correct,
                "nonempty": nonempty,
                "top1_correct": top1_correct,
                "value_correct": value_correct,
            }
            self.update_counter(self.total, **values)
            self.update_counter(
                self.by_context[context],
                **values,
            )

    def summary(self) -> dict[str, Any]:
        return {
            **summarize_counter(self.total),
            "by_context": {
                str(context): summarize_counter(counter)
                for context, counter in sorted(self.by_context.items())
            },
        }


def slice_batch(
    batch: dict[str, torch.Tensor],
    rows: int,
) -> dict[str, torch.Tensor]:
    return {key: value[:rows] for key, value in batch.items()}


def evaluate(
    models: list[bc.EntityOptionPolicy],
    loader: DataLoader,
    device: torch.device,
    *,
    ensemble_weights: tuple[float, ...],
    count_ensemble_weights: tuple[float, ...],
    ensemble_combination: str,
    canonicalize_order: bool,
    max_rows: int | None,
    progress_interval: int,
) -> tuple[dict[str, Any], float]:
    accumulator = MetricAccumulator()
    if not models:
        raise ValueError("At least one policy model is required")
    if len(ensemble_weights) != len(models):
        raise ValueError("Ensemble weights must match the policy model count")
    weight_tensor = torch.tensor(
        ensemble_weights,
        dtype=torch.float32,
        device=device,
    )
    weight_tensor = weight_tensor / weight_tensor.sum()
    if len(count_ensemble_weights) != len(models):
        raise ValueError("Count ensemble weights must match the policy model count")
    count_weight_tensor = torch.tensor(
        count_ensemble_weights,
        dtype=torch.float32,
        device=device,
    )
    count_weight_tensor = count_weight_tensor / count_weight_tensor.sum()
    for model in models:
        model.eval()
    started = time.time()
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(loader, start=1):
            if max_rows is not None:
                remaining = max_rows - accumulator.total["rows"]
                if remaining <= 0:
                    break
                if remaining < cpu_batch["targets"].shape[0]:
                    cpu_batch = slice_batch(cpu_batch, remaining)
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in cpu_batch.items()
            }
            model_outputs = [
                ppo.model_forward(model, batch, device) for model in models
            ]
            output_keys = tuple(model_outputs[0])
            if any(tuple(outputs) != output_keys for outputs in model_outputs[1:]):
                raise ValueError("Ensemble checkpoints produced different outputs")
            outputs = {
                key: (
                    torch.stack(
                        [model_output[key] for model_output in model_outputs],
                        dim=0,
                    )
                    * weight_tensor.view(
                        (-1,) + (1,) * model_outputs[0][key].ndim
                    )
                ).sum(dim=0)
                for key in output_keys
            }
            if len(models) > 1:
                outputs["count_logits"] = (
                    torch.stack(
                        [
                            model_output["count_logits"]
                            for model_output in model_outputs
                        ],
                        dim=0,
                    )
                    * count_weight_tensor.view(
                        (-1,) + (1,) * model_outputs[0]["count_logits"].ndim
                    )
                ).sum(dim=0)
            if ensemble_combination == "probabilities" and len(models) > 1:
                for key in ("policy_logits", "count_logits"):
                    combination_weights = (
                        count_weight_tensor
                        if key == "count_logits"
                        else weight_tensor
                    )
                    probabilities = torch.stack(
                        [
                            torch.softmax(model_output[key].float(), dim=-1)
                            for model_output in model_outputs
                        ],
                        dim=0,
                    )
                    averaged = (
                        probabilities
                        * combination_weights.view(
                            (-1,) + (1,) * model_outputs[0][key].ndim
                        )
                    ).sum(dim=0)
                    outputs[key] = averaged.clamp_min(1e-12).log()
            policy_actions, _, _, _ = ppo.sample_ordered_actions(
                outputs,
                batch,
                deterministic=True,
                canonicalize_order=False,
            )
            actions = (
                [sorted(action) for action in policy_actions]
                if canonicalize_order
                else policy_actions
            )
            accumulator.update(
                batch,
                outputs,
                actions,
                policy_actions,
            )
            if (
                progress_interval > 0
                and batch_index % progress_interval == 0
            ):
                elapsed = time.time() - started
                print(
                    f"rows={accumulator.total['rows']:,} "
                    f"rows_per_second="
                    f"{accumulator.total['rows'] / max(elapsed, 1e-6):.0f}",
                    file=sys.stderr,
                    flush=True,
                )
    seconds = time.time() - started
    if accumulator.total["rows"] == 0:
        raise RuntimeError("No evaluation rows matched the requested split/filters")
    return accumulator.summary(), seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a BC archive split and evaluate a BC or PPO policy "
            "checkpoint without modifying training artifacts."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--bc-feature-source",
        type=Path,
        help=(
            "Optional pinned train_bc_orbit.py used to reproduce an older "
            "checkpoint feature version. Its feature version must match all "
            "evaluated checkpoints."
        ),
    )
    parser.add_argument(
        "--ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional additional architecture-compatible checkpoint. "
            "Policy, set, count, and value logits are averaged at inference."
        ),
    )
    parser.add_argument(
        "--ensemble-weight",
        type=float,
        action="append",
        default=[],
        help=(
            "Optional positive checkpoint weights in primary-then-ensemble "
            "order. Omit for equal weighting."
        ),
    )
    parser.add_argument(
        "--ensemble-combination",
        choices=("logits", "probabilities"),
        default="logits",
        help="Combine raw logits or normalized class probabilities.",
    )
    parser.add_argument(
        "--count-ensemble-weight",
        type=float,
        action="append",
        default=[],
        help=(
            "Optional nonnegative count-head weights with positive sum. "
            "Defaults to the policy "
            "ensemble weights."
        ),
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="test",
    )
    parser.add_argument(
        "--split-mode",
        choices=("archive", "episode_hash"),
        default="archive",
    )
    parser.add_argument("--split-seed", type=int, default=20260723)
    parser.add_argument("--deck-hash", action="append", default=[])
    parser.add_argument("--team-name", action="append", default=[])
    parser.add_argument("--dataset-date", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--prediction-order",
        choices=("auto", "policy", "canonical"),
        default="auto",
        help=(
            "auto canonicalizes BC predictions and preserves PPO policy order; "
            "policy always preserves greedy order; canonical always sorts"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional result path. Without this flag only stdout is used.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact rather than indented JSON.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Report progress to stderr every N batches; 0 disables it.",
    )
    return parser.parse_args()


def main() -> None:
    global bc
    args = parse_args()
    if args.bc_feature_source is not None:
        if not args.bc_feature_source.is_file():
            raise FileNotFoundError(args.bc_feature_source)
        bc = load_bc_feature_source(args.bc_feature_source.resolve())
    checkpoint_paths = [args.checkpoint, *args.ensemble_checkpoint]
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative")
    if args.ensemble_weight:
        if len(args.ensemble_weight) != len(checkpoint_paths):
            raise ValueError(
                "--ensemble-weight count must match all checkpoints"
            )
        if any(
            not math.isfinite(weight) or weight <= 0.0
            for weight in args.ensemble_weight
        ):
            raise ValueError("--ensemble-weight values must be finite and positive")
        ensemble_weights = tuple(args.ensemble_weight)
    else:
        ensemble_weights = (1.0,) * len(checkpoint_paths)
    if args.count_ensemble_weight:
        if len(args.count_ensemble_weight) != len(checkpoint_paths):
            raise ValueError(
                "--count-ensemble-weight count must match all checkpoints"
            )
        if any(
            not math.isfinite(weight) or weight < 0.0
            for weight in args.count_ensemble_weight
        ) or sum(args.count_ensemble_weight) <= 0.0:
            raise ValueError(
                "--count-ensemble-weight values must be finite and "
                "nonnegative with a positive sum"
            )
        count_ensemble_weights = tuple(args.count_ensemble_weight)
    else:
        count_ensemble_weights = ensemble_weights

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    loaded = [load_policy(path, device) for path in checkpoint_paths]
    models = [item[0] for item in loaded]
    model_config, checkpoint, kind = loaded[0][1:]
    for checkpoint_path, (_, other_config, _, other_kind) in zip(
        checkpoint_paths[1:], loaded[1:]
    ):
        if other_config != model_config:
            raise ValueError(
                f"Ensemble architecture mismatch: {checkpoint_path}"
            )
        if other_kind != kind:
            raise ValueError(
                f"Ensemble checkpoint kind mismatch: {checkpoint_path}"
            )
    if args.prediction_order == "auto":
        canonicalize_order = kind == "bc"
    else:
        canonicalize_order = args.prediction_order == "canonical"

    dataset = OrderedZipDecisionDataset(
        archive_path=args.data,
        split=args.split,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        hash_size=model_config["hash_size"],
        max_state_entities=model_config["max_state_entities"],
        deck_hashes=tuple(args.deck_hash),
        team_names=tuple(args.team_name),
        dataset_dates=tuple(args.dataset_date),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=partial(
            collate_ordered,
            max_state_entities=model_config["max_state_entities"],
            entity_fields=model_config["entity_fields"],
            option_fields=model_config["option_fields"],
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=2 if args.workers else None,
    )
    metrics, seconds = evaluate(
        models,
        loader,
        device,
        ensemble_weights=ensemble_weights,
        count_ensemble_weights=count_ensemble_weights,
        ensemble_combination=args.ensemble_combination,
        canonicalize_order=canonicalize_order,
        max_rows=args.max_rows,
        progress_interval=args.progress_interval,
    )
    count_weight = checkpoint["model_state_dict"]["count_head.2.weight"]
    inference_count_layer = models[0].count_head[-1]
    if not isinstance(inference_count_layer, nn.Linear):
        raise TypeError("Unexpected inference count head")
    result = {
        "evaluator": "tools/evaluate_policy_bc.py",
        "bc_feature_source": (
            str(args.bc_feature_source.resolve())
            if args.bc_feature_source is not None
            else str((TOOLS_ROOT / "train_bc_orbit.py").resolve())
        ),
        "bc_feature_source_sha256": file_sha256(
            args.bc_feature_source.resolve()
            if args.bc_feature_source is not None
            else TOOLS_ROOT / "train_bc_orbit.py"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "ensemble_checkpoints": [
            {
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": file_sha256(path),
            }
            for path in checkpoint_paths
        ],
        "ensemble_size": len(checkpoint_paths),
        "ensemble_weights": list(ensemble_weights),
        "count_ensemble_weights": list(count_ensemble_weights),
        "ensemble_combination": args.ensemble_combination,
        "checkpoint_kind": kind,
        "feature_version": checkpoint.get("feature_version"),
        "checkpoint_update": checkpoint.get("update"),
        "data": str(args.data.resolve()),
        "split": args.split,
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "filters": {
            "deck_hashes": args.deck_hash,
            "team_names": args.team_name,
            "dataset_dates": args.dataset_date,
        },
        "prediction_order": (
            "canonical_ascending"
            if canonicalize_order
            else "policy_greedy"
        ),
        "hybrid_prediction_order": (
            "policy_greedy_for_context_34_skill_order;"
            "canonical_ascending_otherwise"
        ),
        "device": str(device),
        "checkpoint_count_classes": int(count_weight.shape[0]),
        "inference_count_classes": inference_count_layer.out_features,
        "max_rows": args.max_rows,
        "metrics": metrics,
        "seconds": seconds,
        "rows_per_second": metrics["rows"] / max(seconds, 1e-6),
    }
    serialized = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        sort_keys=True,
    )
    print(serialized)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
