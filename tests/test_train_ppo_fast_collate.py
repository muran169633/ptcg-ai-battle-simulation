from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import train_bc_orbit  # noqa: E402
import train_ppo  # noqa: E402


def test_fast_live_collate_is_elementwise_identical_to_general_collate() -> None:
    config = {
        "hash_size": 64,
        "max_state_entities": 8,
        "entity_fields": 12,
        "option_fields": 12,
    }
    features = []
    for context, options in ((0, 2), (7, 5), (34, 3), (14, 1)):
        feature = train_bc_orbit.featurize_row(
            {
                "observation": {
                    "select": {
                        "option": [
                            {"index": index} for index in range(options)
                        ],
                        "minCount": 0 if context == 0 else 1,
                        "maxCount": min(options, 3),
                        "context": context,
                    }
                },
                "action": [],
            },
            hash_size=64,
            max_state_entities=8,
        )
        assert feature is not None
        features.append(feature)
    # Real engine action modes may emit different categorical widths inside
    # one batch; keep the optimized path equivalent for those ragged rows.
    features[1]["option_fields"][0] = features[1]["option_fields"][0][:-3]

    expected = train_bc_orbit.collate_decisions(
        features,
        max_state_entities=8,
        entity_fields=12,
        option_fields=12,
    )
    actual = train_ppo.collate_live_features_fast_cpu(features, config)

    assert actual is not None
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name]), name


def test_fast_live_collate_defers_action_rows_to_general_path() -> None:
    assert train_ppo.collate_live_features_fast_cpu(
        [{"action_count": 1, "action_sequence": [0]}],
        {
            "max_state_entities": 8,
            "entity_fields": 12,
            "option_fields": 12,
        },
    ) is None
