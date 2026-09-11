#!/usr/bin/env python3
"""Package a mode-aware AR BC/PPO checkpoint for the Kaggle simulator.

The training implementation intentionally lives in two modules.  Kaggle only
loads the four files at the archive root, so this packager materializes a
standalone inference runtime from the audited training sources and verifies
that every bundled tensor is identical to the source checkpoint.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import torch


EXPECTED_MEMBERS = ("main.py", "deck.csv", "model.pt", "policy_runtime.py")
BC_FEATURE_VERSION = "ptcg-bc-mode-ar-pointer-v7"
PPO_FEATURE_VERSION = "ptcg-mode-ar-ppo-terminal01-v1"
MODEL_CONFIG_KEYS = (
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"{path} contains {len(deck)} cards; expected 60")
    return deck


def semantic_deck_hash(deck: list[int]) -> str:
    canonical = ",".join(str(card) for card in sorted(deck))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_source(repo_root: Path) -> str:
    base_path = repo_root / "tools/train_bc_orbit.py"
    mode_path = repo_root / "tools/train_bc_mode_ar_v7.py"
    base = base_path.read_text(encoding="utf-8")
    base = base.replace(
        "import orjson\n",
        "try:\n    import orjson\nexcept ModuleNotFoundError:\n    orjson = json\n",
        1,
    )
    base = base.replace(
        '\nif __name__ == "__main__":\n    main()\n',
        "\n",
        1,
    )

    mode = mode_path.read_text(encoding="utf-8")
    architecture_start = mode.index(
        f'FEATURE_VERSION = "{BC_FEATURE_VERSION}"'
    )
    architecture_end = mode.index("\ndef autoregressive_losses(")
    architecture = mode[architecture_start:architecture_end]
    decode_start = mode.index("@torch.no_grad()\ndef greedy_decode(")
    decode_end = mode.index("\ndef semantic_exact(", decode_start)
    decode = mode[decode_start:decode_end]
    overlay = (architecture + "\n\n" + decode).replace("base.", "")
    return (
        base
        + "\n\n# Generated mode-aware inference overlay.\n"
        + overlay
        + "\n"
    )


def main_source(deck_hash: str, *, boss_guard: bool = False) -> str:
    return f'''"""Kaggle entrypoint for a mode-aware Dragapult PPO policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

import policy_runtime as _runtime
from policy_runtime import ModeAwareARPolicy, collate_decisions, featurize_row, greedy_decode


AGENT_DIR = Path(_runtime.__file__).resolve().parent
MODEL_PATH = AGENT_DIR / "model.pt"
DECK_PATH = AGENT_DIR / "deck.csv"
DECK_HASH = "{deck_hash}"
BC_FEATURE_VERSION = "{BC_FEATURE_VERSION}"
PPO_FEATURE_VERSION = "{PPO_FEATURE_VERSION}"
ENGINE_MAX_ACTION_COUNT = 60
BOSS_GUARD_ENABLED = {boss_guard!r}
BOSS_ORDERS_CARD_ID = 1182
BUDEW_ATTACK_ID = 323


def _configure_cpu() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


_configure_cpu()


def _read_deck() -> list[int]:
    deck = [
        int(line.strip())
        for line in DECK_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 cards, found {{len(deck)}}")
    canonical = ",".join(str(card) for card in sorted(deck))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != DECK_HASH:
        raise ValueError("Unexpected deck.csv semantic hash")
    return deck


DECK = _read_deck()
_MODEL: ModeAwareARPolicy | None = None
_MODEL_CONFIG: dict[str, Any] | None = None


def _load_checkpoint() -> dict[str, Any]:
    try:
        return torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(MODEL_PATH, map_location="cpu")


def _load_model() -> tuple[ModeAwareARPolicy, dict[str, Any]]:
    global _MODEL, _MODEL_CONFIG
    if _MODEL is not None and _MODEL_CONFIG is not None:
        return _MODEL, _MODEL_CONFIG
    checkpoint = _load_checkpoint()
    feature_version = checkpoint.get("feature_version")
    if feature_version not in {{BC_FEATURE_VERSION, PPO_FEATURE_VERSION}}:
        raise ValueError(f"Unexpected mode-aware feature version: {{feature_version!r}}")
    if (
        feature_version == PPO_FEATURE_VERSION
        and checkpoint.get("bc_feature_version") != BC_FEATURE_VERSION
    ):
        raise ValueError("PPO checkpoint has an incompatible BC feature contract")
    config = dict(checkpoint["model_config"])
    state = checkpoint["model_state_dict"]
    option_positions = state.get("option_position.weight")
    if not isinstance(option_positions, torch.Tensor) or option_positions.ndim != 2:
        raise ValueError("Checkpoint has no valid option position table")
    model = ModeAwareARPolicy(
        hash_size=int(config["hash_size"]),
        categorical_dim=int(config["categorical_dim"]),
        model_dim=int(config["model_dim"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        max_state_entities=int(config["max_state_entities"]),
        max_options=int(option_positions.shape[0]),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    _MODEL = model
    _MODEL_CONFIG = config
    return model, config


def _raw_policy_action(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, dict):
        raise TypeError("Missing select object")
    options = select.get("option")
    if not isinstance(options, list):
        raise TypeError("select.option must be a list")
    option_count = len(options)
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    if minimum < 0 or maximum < minimum or maximum > option_count or maximum > ENGINE_MAX_ACTION_COUNT:
        raise ValueError(
            f"Invalid legal bounds min={{minimum}} max={{maximum}} options={{option_count}}"
        )
    if option_count == 0:
        return []

    model, config = _load_model()
    current = observation.get("current") or {{}}
    feature = featurize_row(
        {{
            "observation": observation,
            "seat": int(current.get("yourIndex", 0) or 0),
            "deck_hash": DECK_HASH,
            "team_name": "",
            "action": [],
            "action_sequence": [],
            "terminal_reward": 0.0,
            "sample_weight": 1.0,
        }},
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
    blocked = _boss_guard_positions(observation)
    if blocked:
        batch["option_mask"][0, sorted(blocked)] = False
    with torch.inference_mode():
        encoded = model(batch)
        decoded = greedy_decode(model, encoded, batch)
    count = int(decoded["counts"][0])
    action = [
        int(value)
        for value in decoded["sequences"][0].tolist()
        if int(value) >= 0
    ]
    if (
        len(action) != count
        or len(action) < minimum
        or len(action) > maximum
        or len(action) != len(set(action))
        or any(index < 0 or index >= option_count for index in action)
    ):
        raise RuntimeError("Mode-aware policy produced an illegal action")
    return action


def _boss_guard_positions(observation: dict[str, Any]) -> set[int]:
    """Identify premature Boss's Orders options for inference masking.

    The guard is intentionally narrow: it applies only to a top-level action
    menu when the learner has no attack option, or its only available attack
    is Budew's non-damaging attack.  It never guesses future prize conversion.
    """

    if not BOSS_GUARD_ENABLED:
        return set()
    select = observation.get("select")
    current = observation.get("current")
    if not isinstance(select, dict) or not isinstance(current, dict):
        return set()
    if int(select.get("context", -1) or 0) != 0 or int(select.get("type", -1) or 0) != 0:
        return set()
    options = select.get("option")
    players = current.get("players")
    seat = int(current.get("yourIndex", 0) or 0)
    if not isinstance(options, list) or not isinstance(players, list) or not 0 <= seat < len(players):
        return set()
    hand = players[seat].get("hand") if isinstance(players[seat], dict) else None
    if not isinstance(hand, list):
        return None

    blocked: set[int] = set()
    for option_position, option in enumerate(options):
        if not isinstance(option, dict) or int(option.get("type", -1) or -1) != 7:
            continue
        hand_index = option.get("index")
        if not isinstance(hand_index, int) or not 0 <= hand_index < len(hand):
            continue
        card = hand[hand_index]
        if isinstance(card, dict) and int(card.get("id", -1) or -1) == BOSS_ORDERS_CARD_ID:
            blocked.add(option_position)
    if not blocked:
        return set()

    attacks = [
        option for option in options
        if isinstance(option, dict) and int(option.get("type", -1) or -1) == 13
    ]
    has_effective_attack = bool(attacks) and any(
        int(option.get("attackId", -1) or -1) != BUDEW_ATTACK_ID
        for option in attacks
    )
    if has_effective_attack:
        return set()

    kept_indices = [index for index in range(len(options)) if index not in blocked]
    minimum = int(select.get("minCount", 0) or 0)
    if len(kept_indices) < minimum:
        return set()
    return blocked


def _boss_guard_view(
    observation: dict[str, Any],
) -> tuple[dict[str, Any], list[int]] | None:
    """Return a filtered diagnostic view used only by package audits."""

    blocked = _boss_guard_positions(observation)
    if not blocked:
        return None
    select = observation["select"]
    options = select["option"]
    kept_indices = [index for index in range(len(options)) if index not in blocked]
    guarded_select = dict(select)
    guarded_select["option"] = [options[index] for index in kept_indices]
    guarded_select["maxCount"] = min(
        int(select.get("maxCount", 0) or 0), len(kept_indices)
    )
    guarded_observation = dict(observation)
    guarded_observation["select"] = guarded_select
    return guarded_observation, kept_indices


def _policy_action(observation: dict[str, Any]) -> list[int]:
    return _raw_policy_action(observation)


def agent(obs_dict: dict[str, Any]) -> list[int]:
    if obs_dict.get("select") is None:
        return list(DECK)
    return _policy_action(obs_dict)
'''


def deterministic_torch_save(value: Any, path: Path) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    path.write_bytes(buffer.getvalue())


def write_archive(source_dir: Path, archive_path: Path) -> None:
    with archive_path.open("xb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name in EXPECTED_MEMBERS:
                    payload = (source_dir / name).read_bytes()
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--boss-guard",
        action="store_true",
        help=(
            "Suppress Boss's Orders only when no effective damaging attack is "
            "currently available (including Budew-only attack states)."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for path in (args.checkpoint, args.deck):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.archive.exists():
        raise FileExistsError("Refusing to overwrite an existing package")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a mapping")
    feature_version = checkpoint.get("feature_version")
    if feature_version not in {BC_FEATURE_VERSION, PPO_FEATURE_VERSION}:
        raise ValueError(f"Unsupported feature version: {feature_version!r}")
    if feature_version == PPO_FEATURE_VERSION and checkpoint.get("bc_feature_version") != BC_FEATURE_VERSION:
        raise ValueError("PPO checkpoint BC feature contract mismatch")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state")
    config_source = checkpoint.get("model_config", checkpoint.get("config"))
    if not isinstance(config_source, dict):
        raise ValueError("Checkpoint has no model configuration")
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config_source]
    if missing:
        raise KeyError(f"Model configuration is missing {missing}")
    model_config = {key: config_source[key] for key in MODEL_CONFIG_KEYS}

    deck = read_deck(args.deck)
    deck_hash = semantic_deck_hash(deck)
    if checkpoint.get("learner_deck_hash") != deck_hash:
        raise ValueError("Checkpoint learner deck does not match deck.csv")

    args.output_dir.mkdir(parents=True)
    shutil.copy2(args.deck, args.output_dir / "deck.csv")
    (args.output_dir / "main.py").write_text(
        main_source(deck_hash, boss_guard=args.boss_guard), encoding="utf-8"
    )
    (args.output_dir / "policy_runtime.py").write_text(runtime_source(repo_root), encoding="utf-8")
    bundled = {
        "feature_version": feature_version,
        "bc_feature_version": checkpoint.get("bc_feature_version", BC_FEATURE_VERSION),
        "model_config": model_config,
        "model_state_dict": state,
        "update": checkpoint.get("update", 0),
        "reward": checkpoint.get("reward"),
        "action_distribution": checkpoint.get("action_distribution"),
        "learner_deck_hash": deck_hash,
    }
    deterministic_torch_save(bundled, args.output_dir / "model.pt")
    reloaded = torch.load(args.output_dir / "model.pt", map_location="cpu", weights_only=True)
    unequal = [
        name for name in state
        if name not in reloaded["model_state_dict"]
        or not torch.equal(state[name], reloaded["model_state_dict"][name])
    ]
    if unequal or state.keys() != reloaded["model_state_dict"].keys():
        raise RuntimeError(f"Bundled model differs from source: {unequal[:5]}")
    for name in ("main.py", "policy_runtime.py"):
        source = (args.output_dir / name).read_text(encoding="utf-8")
        compile(source, str(args.output_dir / name), "exec")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    write_archive(args.output_dir, args.archive)
    with tarfile.open(args.archive, "r:gz") as tar:
        members = tuple(member.name for member in tar.getmembers())
        if members != EXPECTED_MEMBERS or any(not member.isfile() for member in tar.getmembers()):
            raise RuntimeError(f"Invalid archive members: {members}")

    archive_hash = sha256_file(args.archive)
    sidecar = args.archive.with_suffix(args.archive.suffix + ".sha256")
    sidecar.write_text(f"{archive_hash}  {args.archive.name}\n", encoding="utf-8")
    manifest_path = args.manifest or args.archive.with_suffix(args.archive.suffix + ".manifest.json")
    manifest = {
        "schema_version": "ptcg-mode-ar-submission-package-v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "source_update": checkpoint.get("update"),
        "feature_version": feature_version,
        "deck": str(args.deck.resolve()),
        "deck_file_sha256": sha256_file(args.deck),
        "semantic_deck_hash": deck_hash,
        "deck_order_preserved": (args.output_dir / "deck.csv").read_bytes() == args.deck.read_bytes(),
        "archive": str(args.archive.resolve()),
        "archive_sha256": archive_hash,
        "archive_bytes": args.archive.stat().st_size,
        "members": {
            name: {"bytes": (args.output_dir / name).stat().st_size, "sha256": sha256_file(args.output_dir / name)}
            for name in EXPECTED_MEMBERS
        },
        "model_tensor_count": len(state),
        "model_tensors_equal_source": True,
        "decode": "mode-aware greedy pointer; raw order for context 34, canonical ascending constraint for unordered sets",
        "boss_orders_guard": {
            "enabled": bool(args.boss_guard),
            "card_id": 1182,
            "condition": "top-level action with no attack or Budew attack 323 only",
        },
        "runtime_sources": {
            "train_bc_orbit.py": sha256_file(repo_root / "tools/train_bc_orbit.py"),
            "train_bc_mode_ar_v7.py": sha256_file(repo_root / "tools/train_bc_mode_ar_v7.py"),
        },
        "byte_reproducible_packaging": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
