from __future__ import annotations

from unittest import mock

from tools import bc_feature_orbit_v5_pinned as pinned


def test_adapter_exports_and_forwards_temporary_action_limit() -> None:
    assert pinned.resolve_card is pinned._module.resolve_card
    assert pinned.collate_decisions is pinned._module.collate_decisions
    assert pinned.EntityOptionPolicy is pinned._module.EntityOptionPolicy
    original_wrapper_limit = pinned.MAX_ACTION_COUNT
    original_source_limit = pinned._module.MAX_ACTION_COUNT
    observed: list[int] = []

    def fake_featurize(*_args: object) -> dict[str, object]:
        observed.append(pinned._module.MAX_ACTION_COUNT)
        return {"action_sequence": [1], "kept": True}

    try:
        pinned.MAX_ACTION_COUNT = 60
        with mock.patch.object(
            pinned._module,
            "featurize_row",
            side_effect=fake_featurize,
        ):
            result = pinned.featurize_row({}, 16, 8)
    finally:
        pinned.MAX_ACTION_COUNT = original_wrapper_limit

    assert observed == [60]
    assert pinned._module.MAX_ACTION_COUNT == original_source_limit
    assert result == {"kept": True}
