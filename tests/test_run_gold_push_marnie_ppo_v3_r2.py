from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import run_gold_push_marnie_ppo_v3_r2 as r2  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class GoldPushMarniePpoV3R2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase = r2.PHASES["full"]
        cls.command = r2.build_command(cls.phase)
        cls.preflight = r2.build_preflight(cls.phase)

    def test_r2_uses_fresh_smoke_and_full_outputs(self) -> None:
        self.assertEqual(
            command_value(self.command, "--output-dir"), str(r2.OUTPUT_ROOT)
        )
        self.assertEqual(
            self.preflight.manifest["expected_terminal_checkpoint"],
            str(r2.OUTPUT_ROOT / "checkpoints/update-0002.pt"),
        )
        self.assertNotEqual(r2.OUTPUT_ROOT, r2.v3.OUTPUT_ROOT)
        self.assertNotEqual(r2.SMOKE_OUTPUT, r2.v3.SMOKE_OUTPUT)
        self.assertFalse(r2.OUTPUT_ROOT.exists())
        self.assertFalse(r2.SMOKE_OUTPUT.exists())

    def test_training_command_only_changes_output_directory(self) -> None:
        original = r2.v3.build_command(self.phase)
        differences = [
            index
            for index, (before, after) in enumerate(
                zip(original, self.command, strict=True)
            )
            if before != after
        ]
        self.assertEqual(differences, [original.index("--output-dir") + 1])
        protocol = self.preflight.manifest["revision"]["protocol_diff"]
        self.assertTrue(protocol["training_protocol_unchanged"])
        self.assertEqual(
            protocol["only_command_difference"], "--output-dir value"
        )

    def test_current_panel_audit_is_sha_pinned(self) -> None:
        inputs = self.preflight.manifest["inputs"]
        self.assertEqual(
            inputs["panel_audit"]["sha256"],
            r2.CURRENT_PANEL_AUDIT_SHA256,
        )
        self.assertEqual(
            r2.v3.raw_sha256_file(r2.v3.PANEL_AUDIT),
            r2.CURRENT_PANEL_AUDIT_SHA256,
        )
        self.assertTrue(
            self.preflight.manifest["gates"][
                "current_panel_audit_sha256_exact"
            ]
        )

    def test_invalidated_smoke_is_bound_and_forbidden(self) -> None:
        revision = self.preflight.manifest["revision"]
        invalidated = revision["invalidated_predecessor"]
        self.assertEqual(
            invalidated["record_sha256"], r2.INVALIDATED_SMOKE_SHA256
        )
        self.assertFalse(invalidated["may_select_or_promote_checkpoint"])
        self.assertFalse(invalidated["may_seed_later_training"])
        self.assertFalse(invalidated["may_package_or_submit"])
        self.assertEqual(
            self.preflight.manifest["inputs"]["invalidated_v3_smoke"][
                "sha256"
            ],
            r2.INVALIDATED_SMOKE_SHA256,
        )

    def test_launcher_lineage_is_explicit(self) -> None:
        inputs = self.preflight.manifest["inputs"]
        self.assertEqual(inputs["v3_launcher"]["sha256"], r2.V3_LAUNCHER_SHA256)
        self.assertEqual(
            inputs["launcher"]["sha256"], r2.v3.raw_sha256_file(r2.SELF)
        )
        self.assertEqual(
            self.preflight.manifest["schema_version"],
            "ptcg-gold-push-marnie-v3-cvar-tailrepair-launch-r2",
        )

    def test_template_global_overrides_are_restored(self) -> None:
        self.assertNotEqual(r2.v3.OUTPUT_ROOT, r2.OUTPUT_ROOT)
        self.assertNotEqual(r2.v3.SMOKE_OUTPUT, r2.SMOKE_OUTPUT)
        self.assertEqual(
            r2.v3.EXTRA_SHA256["panel_audit"],
            "1311e004ea65bec15e15e15165a548250835dbbcab2a1cbb5998efd181214958",
        )

    def test_smoke_protocol_is_unchanged_except_fresh_output(self) -> None:
        phase = r2.PHASES["smoke"]
        command = r2.build_command(phase)
        original = r2.v3.build_command(phase)
        self.assertEqual(command_value(command, "--updates"), "1")
        self.assertEqual(command_value(command, "--games-per-update"), "96")
        self.assertEqual(
            command_value(command, "--output-dir"), str(r2.SMOKE_OUTPUT)
        )
        expected = list(original)
        r2.v3.set_single_value(expected, "--output-dir", str(r2.SMOKE_OUTPUT))
        self.assertEqual(command, expected)

    def test_default_dry_run_is_zero_write_and_zero_child(self) -> None:
        stdout = io.StringIO()
        with mock.patch.object(r2.subprocess, "run") as child:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(r2.main(["--phase", "full", "--dry-run"]), 0)
        child.assert_not_called()
        self.assertFalse(r2.OUTPUT_ROOT.exists())
        self.assertIn(self.preflight.manifest_sha256, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
