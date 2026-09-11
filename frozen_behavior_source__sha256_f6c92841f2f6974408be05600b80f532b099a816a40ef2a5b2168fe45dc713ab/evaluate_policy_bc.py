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
import json
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
    for row_index, row in enumerate(rows):
        action = row["expert_action_order"]
        ordered_action_counts[row_index] = len(action)
        if action:
            ordered_actions[row_index, : len(action)] = torch.tensor(action)
    batch["expert_ordered_actions"] = ordered_actions
    batch["expert_ordered_action_counts"] = ordered_action_counts
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
    model: bc.EntityOptionPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    canonicalize_order: bool,
    max_rows: int | None,
    progress_interval: int,
) -> tuple[dict[str, Any], float]:
    accumulator = MetricAccumulator()
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
            outputs = ppo.model_forward(model, batch, device)
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
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
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

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    model, model_config, checkpoint, kind = load_policy(
        args.checkpoint,
        device,
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
        model,
        loader,
        device,
        canonicalize_order=canonicalize_order,
        max_rows=args.max_rows,
        progress_interval=args.progress_interval,
    )
    count_weight = checkpoint["model_state_dict"]["count_head.2.weight"]
    inference_count_layer = model.count_head[-1]
    if not isinstance(inference_count_layer, nn.Linear):
        raise TypeError("Unexpected inference count head")
    result = {
        "evaluator": "tools/evaluate_policy_bc.py",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
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
