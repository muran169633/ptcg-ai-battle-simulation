from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_meta_weighted_ppo_routes as runner


def test_routes_use_distinct_exact_replays_and_fresh_bc_defaults() -> None:
    drag = runner.ROUTES["dragapult"]
    rmy = runner.ROUTES["rmy"]
    assert drag.replay_archive != rmy.replay_archive
    assert "rmy_exact_public_20260814_v1" in str(rmy.replay_archive)
    assert drag.default_output != rmy.default_output
    assert drag.seed != rmy.seed


def test_common_protocol_is_recent_day_frequency_inverse_window() -> None:
    payload = runner.json.loads(runner.META_POOL.read_text(encoding="utf-8"))
    assert payload["dataset_date"] == "2026-08-13"
    assert payload["selection"]["top_n"] == 23
    assert runner.sha256_file(runner.META_POOL) == runner.META_POOL_SHA256
