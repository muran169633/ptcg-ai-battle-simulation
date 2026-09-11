from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_top3_bc77_corrected_ppo as corrected  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def quota_values(command: list[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for index, token in enumerate(command):
        if token == "--opponent-base-quota":
            values[command[index + 1]] = int(command[index + 2])
    return values


class CorrectedTop3PpoLauncherTests(unittest.TestCase):
    def test_full_protocol_uses_corrected_objective_and_budget(self) -> None:
        phase = corrected.PHASES["full"]
        profile = corrected.PROFILES["alakazam_control"]
        command = corrected.ppo_command(profile, phase)

        self.assertEqual(command[0], str(corrected.PYTHON))
        self.assertEqual(command_value(command, "--updates"), "12")
        self.assertEqual(command_value(command, "--games-per-update"), "256")
        self.assertEqual(command_value(command, "--gae-lambda"), "1.0")
        self.assertEqual(
            command_value(command, "--advantage-normalization"),
            "per_opponent",
        )
        self.assertEqual(
            command_value(command, "--actor-reduction"),
            "transition_mean",
        )
        self.assertEqual(
            command_value(command, "--value-trunk-gradient-scale"),
            "0.05",
        )
        self.assertEqual(phase.updates * phase.games_per_update, 3072)

    def test_only_top3_opponents_and_50_25_25_quotas(self) -> None:
        phase = corrected.PHASES["full"]
        profile = corrected.PROFILES["marnie"]
        command = corrected.ppo_command(profile, phase)
        quotas = quota_values(command)

        self.assertEqual(command.count("--extra-opponent"), 2)
        self.assertEqual(sorted(quotas.values()), [64, 64, 128])
        self.assertEqual(sum(quotas.values()), phase.games_per_update)
        for other in corrected.PROFILES.values():
            if other.slug != profile.slug:
                self.assertIn(str(other.checkpoint), command)
                self.assertIn(str(other.deck), command)
        for checkpoint, deck, _ in corrected.baseline.AUXILIARIES:
            self.assertNotIn(str(checkpoint), command)
            self.assertNotIn(str(deck), command)

    def test_outputs_are_versioned_and_do_not_reuse_previous_run(self) -> None:
        profile = corrected.PROFILES["alakazam_control"]
        smoke = corrected.output_dir(profile, corrected.PHASES["smoke"])
        full = corrected.output_dir(profile, corrected.PHASES["full"])

        self.assertNotEqual(smoke, full)
        self.assertNotEqual(smoke, corrected.baseline.output_dir(profile))
        self.assertNotEqual(full, corrected.baseline.output_dir(profile))
        self.assertIn(corrected.EXPERIMENT_NAME, smoke.name)
        self.assertIn(corrected.EXPERIMENT_NAME, full.name)


if __name__ == "__main__":
    unittest.main()
