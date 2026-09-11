from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import train_mode_ar_ppo_ddp as ddp  # noqa: E402


class ModeAwareDDPTests(unittest.TestCase):
    def test_non_primary_worker_output_is_sibling_of_authoritative_run(self) -> None:
        runtime = object.__new__(ddp.DistributedRuntime)
        runtime.rank = 3
        runtime.local_rank = 3
        runtime.world_size = 8
        runtime.authoritative_output = Path("/tmp/runs/formal").resolve()
        original_argv = sys.argv
        try:
            sys.argv = [
                "train_mode_ar_ppo_ddp.py",
                "--output-dir",
                str(runtime.authoritative_output),
                "--seed",
                "40",
                "--device",
                "cuda",
            ]
            runtime.install_rank_arguments()
            worker_output = Path(ddp._cli_value("--output-dir") or "")
            self.assertEqual(
                worker_output,
                Path("/tmp/runs/.formal.ddp_worker_state/rank-03").resolve(),
            )
            self.assertNotIn(runtime.authoritative_output, worker_output.parents)
            self.assertEqual(ddp._cli_value("--seed"), str(40 + 3_000_009))
            self.assertEqual(ddp._cli_value("--device"), "cuda:3")
        finally:
            sys.argv = original_argv

    def test_merge_rollout_metrics_uses_global_counts_and_wall_time(self) -> None:
        rows = [
            {
                "valid_games": 2,
                "engine_decisions": 20,
                "transitions_kept": 12,
                "seconds": 4.0,
                "decisions_per_second": 5.0,
                "mean_old_value": 0.25,
                "mean_terminal_return": 0.5,
                "league_by_opponent": {"a": {"games": 2, "wins": 1}},
            },
            {
                "valid_games": 3,
                "engine_decisions": 30,
                "transitions_kept": 18,
                "seconds": 5.0,
                "decisions_per_second": 6.0,
                "mean_old_value": 0.75,
                "mean_terminal_return": 1.0,
                "league_by_opponent": {"a": {"games": 3, "wins": 2}},
            },
        ]
        merged = ddp.merge_rollout_metrics(rows)
        self.assertEqual(merged["valid_games"], 5)
        self.assertEqual(merged["engine_decisions"], 50)
        self.assertEqual(merged["transitions_kept"], 30)
        self.assertEqual(merged["seconds"], 5.0)
        self.assertEqual(merged["decisions_per_second"], 10.0)
        self.assertAlmostEqual(merged["mean_old_value"], 0.55)
        self.assertAlmostEqual(merged["mean_terminal_return"], 0.8)
        self.assertEqual(
            merged["league_by_opponent"]["a"],
            {"games": 5, "wins": 3},
        )
        self.assertEqual(merged["distributed"]["world_size"], 2)
        self.assertEqual(merged["distributed"]["global_games"], 5)


if __name__ == "__main__":
    unittest.main()
