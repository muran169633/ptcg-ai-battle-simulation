#!/usr/bin/env python3
"""Read-only PF/FLG screen for a transported historical repair residual.

For each requested alpha, evaluate

    beta100 + alpha * (historical_full_P16 - historical_actor_only_P12)

in memory.  No checkpoint is serialized; an optional JSON report is the only
possible write.  The script deliberately reuses the official bf16 evaluator
path so boundary decisions match the frozen specialist protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate_policy_bc as evaluator  # noqa: E402


BETA100 = ROOT / (
    "artifacts/ppo_u468_p12delta_direction_beta050_075_100_design202608092/"
    "transport-beta-100.pt"
)
BETA100_SHA256 = "53284b1d4e94f09bff5b92a7fb0d24efd26ee67a2a332014cdc97572b00c3beb"
HISTORICAL_ACTOR_P12 = ROOT / (
    "artifacts/ppo_u464mb384_actorheadonly_pokemonfan_sweep_p4_p8_p12_p16_"
    "design202608060/sweep_stage/"
    "special-bc-actorheadonly-pokemonfan-prefix-0012.pt"
)
HISTORICAL_ACTOR_P12_SHA256 = (
    "8aafef92f0898f46cfe45f076f6fec5bfc383d2429f2d6087805cb976968c3fd"
)
HISTORICAL_FULL_P16 = ROOT / (
    "artifacts/ppo_u464mb384_pokemonfan_prefix_sweep_p16_p20_p24_p28_"
    "design202608050/sweep_stage/special-bc-pokemonfan-prefix-0016.pt"
)
HISTORICAL_FULL_P16_SHA256 = (
    "737de285e13ff0b3c177914a52376c4542d249a57a987b5c0526104658fc98dd"
)

PANELS = {
    "pokemonfan": ROOT
    / "data/bc_marnie_gold21_pokemonfan_timeforward_train28_valid29_special_20260801.zip",
    "flg": ROOT / "data/gold_league/top21_exact_20260727/policies/rank01_flg.zip",
}
METRIC_KEYS = (
    "rows",
    "set_exact_correct",
    "hybrid_order_exact_correct",
    "ordered_exact_correct",
    "count_correct",
    "top1_correct",
    "value_correct",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"checkpoint hash drift for {path}: {observed}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise TypeError(f"invalid checkpoint: {path}")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        action="append",
        type=float,
        required=True,
        help="Residual coefficient; repeat for a grid.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.alpha or any(not 0.0 <= value <= 1.0 for value in args.alpha):
        raise ValueError("all alpha values must lie in [0, 1]")
    if len(set(args.alpha)) != len(args.alpha):
        raise ValueError("alpha values must be unique")

    beta = checked_checkpoint(BETA100, BETA100_SHA256)
    actor_p12 = checked_checkpoint(
        HISTORICAL_ACTOR_P12, HISTORICAL_ACTOR_P12_SHA256
    )
    full_p16 = checked_checkpoint(HISTORICAL_FULL_P16, HISTORICAL_FULL_P16_SHA256)
    states = [
        value["model_state_dict"] for value in (beta, actor_p12, full_p16)
    ]
    if not (list(states[0]) == list(states[1]) == list(states[2])):
        raise ValueError("model-state schemas differ")
    changed = sorted(
        name
        for name in states[0]
        if not torch.equal(states[1][name], states[2][name])
    )
    if len(changed) != 24:
        raise ValueError(f"expected a 24-tensor residual, observed {len(changed)}")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    model, model_config, _, kind = evaluator.load_policy(BETA100, device)
    if kind != "ppo":
        raise ValueError("beta100 must be a PPO checkpoint")
    loaders: dict[str, DataLoader] = {}
    for name, archive in PANELS.items():
        if archive.is_symlink() or not archive.is_file():
            raise FileNotFoundError(archive)
        dataset = evaluator.OrderedZipDecisionDataset(
            archive_path=archive,
            split="valid",
            split_mode="archive",
            split_seed=20260723,
            hash_size=model_config["hash_size"],
            max_state_entities=model_config["max_state_entities"],
            deck_hashes=(),
            team_names=(),
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=partial(
                evaluator.collate_ordered,
                max_state_entities=model_config["max_state_entities"],
                entity_fields=model_config["entity_fields"],
                option_fields=model_config["option_fields"],
            ),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            prefetch_factor=2 if args.workers else None,
        )

    base_state = {
        name: tensor.detach().cpu().clone() for name, tensor in states[0].items()
    }
    results: list[dict[str, Any]] = []
    for alpha in args.alpha:
        candidate = dict(base_state)
        for name in changed:
            base = base_state[name]
            residual = states[2][name].double() - states[1][name].double()
            candidate[name] = (base.double() + alpha * residual).to(base.dtype)
        model.load_state_dict(candidate)
        panels: dict[str, Any] = {}
        for panel, loader in loaders.items():
            metrics, seconds = evaluator.evaluate(
                model,
                loader,
                device,
                canonicalize_order=False,
                max_rows=None,
                progress_interval=0,
            )
            panels[panel] = {
                "metrics": {key: metrics[key] for key in METRIC_KEYS},
                "seconds": seconds,
            }
        delta_l2 = sum(
            float(
                (
                    candidate[name].double() - base_state[name].double()
                ).square().sum()
            )
            for name in candidate
        ) ** 0.5
        results.append(
            {"alpha": alpha, "delta_from_beta100_l2": delta_l2, "panels": panels}
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    report = {
        "schema_version": "ptcg-u468-fullp16-minus-actorp12-readonly-screen-v1",
        "writes_checkpoint": False,
        "formula": "beta100 + alpha * (historical_full_p16 - historical_actor_p12)",
        "inputs": {
            "beta100": {"path": str(BETA100), "sha256": BETA100_SHA256},
            "historical_actor_p12": {
                "path": str(HISTORICAL_ACTOR_P12),
                "sha256": HISTORICAL_ACTOR_P12_SHA256,
            },
            "historical_full_p16": {
                "path": str(HISTORICAL_FULL_P16),
                "sha256": HISTORICAL_FULL_P16_SHA256,
            },
        },
        "changed_tensors": changed,
        "results": results,
    }
    if args.json_output is not None:
        output = args.json_output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
