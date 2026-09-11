from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_top3_bc77_trunc_loss_ppo as launcher  # noqa: E402


class TruncationLossLauncherTests(unittest.TestCase):
    def test_command_is_versioned_and_enables_truncation_loss(self) -> None:
        profile = launcher.PROFILES["alakazam_control"]
        phase = launcher.PHASES["smoke"]
        command = launcher.ppo_command(profile, phase)

        self.assertEqual(command.count("--truncation-as-loss"), 1)
        output_index = command.index("--output-dir") + 1
        self.assertEqual(
            Path(command[output_index]),
            launcher.output_dir(profile, phase),
        )
        self.assertIn("ppo_terminal01_trunc_loss_v3_smoke", command[output_index])

    def test_manifest_keeps_terminal_01_and_exact_replacement(self) -> None:
        profile = launcher.PROFILES["alakazam_control"]
        phase = launcher.PHASES["full"]
        command = launcher.ppo_command(profile, phase)
        manifest = launcher.protocol_manifest(profile, phase, command)

        self.assertEqual(
            manifest["reward"],
            {"win": 1.0, "loss": 0.0, "draw": 0.0},
        )
        self.assertEqual(
            manifest["truncation_handling"],
            {
                "training_target": "loss_return_0",
                "valid_game_quota": "replace",
                "max_game_decisions": 1000,
            },
        )
        self.assertEqual(sum(manifest["opponent_game_quotas"].values()), 256)


if __name__ == "__main__":
    unittest.main()
