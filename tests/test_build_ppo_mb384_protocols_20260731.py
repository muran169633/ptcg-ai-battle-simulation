from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import build_ppo_mb384_protocols_20260731 as builder


FINAL_BEHAVIOR_RUNNER_SHA256 = (
    "69ef3e4b650c76d1814a03b6ea5350ed5d94118643f611428b47f3dfbf132507"
)
FINAL_TRAINING_LAUNCHER_SHA256 = (
    "baa1b1a119b48a423a1411916fbf57e740672357adaeb1eaf7660d70b00fa94c"
)
FINAL_RECEIPT_WRITER_SHA256 = (
    "e9897ebeca75309168bcc7e0cdb25f2b53ac7887e9132a2147057315f957872a"
)


def _named_values(value: object, name: str) -> list[object]:
    matches: list[object] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == name:
                matches.append(child)
            matches.extend(_named_values(child, name))
    elif isinstance(value, list):
        for child in value:
            matches.extend(_named_values(child, name))
    return matches


class ReceiptWriterBindingTest(unittest.TestCase):
    def test_frozen_receipt_writer_bytes_are_accepted(self) -> None:
        contract = builder.load_terminal_receipt_writer_contract(
            FINAL_RECEIPT_WRITER_SHA256
        )
        self.assertEqual(
            contract.path,
            builder.repo_path(builder.TERMINAL_RECEIPT_WRITER_REL),
        )
        self.assertEqual(contract.sha256, FINAL_RECEIPT_WRITER_SHA256)

    def test_missing_receipt_writer_hash_is_rejected_by_cli(self) -> None:
        with mock.patch.object(sys, "argv", ["protocol-builder"]):
            with self.assertRaisesRegex(
                builder.ProtocolBuildError,
                "--expected-receipt-writer-sha256 is required",
            ):
                builder.main()

    def test_wrong_receipt_writer_hash_is_rejected_by_cli(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "protocol-builder",
                "--expected-receipt-writer-sha256",
                "0" * 64,
            ],
        ):
            with self.assertRaisesRegex(
                builder.ProtocolBuildError,
                "does not equal the frozen final writer hash",
            ):
                builder.main()

    def test_local_receipt_writer_byte_drift_is_rejected(self) -> None:
        with mock.patch.object(
            builder,
            "file_sha256",
            return_value="f" * 64,
        ):
            with self.assertRaisesRegex(
                builder.ProtocolBuildError,
                "bytes differ from the frozen final hash",
            ):
                builder.load_terminal_receipt_writer_contract(
                    FINAL_RECEIPT_WRITER_SHA256
                )


class ProtocolPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle, cls.upstream, cls.launcher = builder.build_protocol_bundle(
            created_at_utc="2026-07-31T08:00:00Z",
            expected_behavior_runner_sha256=(
                FINAL_BEHAVIOR_RUNNER_SHA256
            ),
            expected_training_launcher_sha256=(
                FINAL_TRAINING_LAUNCHER_SHA256
            ),
            expected_receipt_writer_sha256=(
                FINAL_RECEIPT_WRITER_SHA256
            ),
        )

    def test_final_hashes_are_explicit_and_no_formal_files_exist(self) -> None:
        self.assertFalse(self.bundle.provisional_infrastructure_hashes)
        self.assertEqual(self.bundle.pending_final_sha_fields, ())
        self.assertEqual(
            self.bundle.behavior_runner_sha256,
            FINAL_BEHAVIOR_RUNNER_SHA256,
        )
        self.assertEqual(
            self.bundle.training_launcher_sha256,
            FINAL_TRAINING_LAUNCHER_SHA256,
        )
        self.assertEqual(
            self.bundle.terminal_receipt_writer_sha256,
            FINAL_RECEIPT_WRITER_SHA256,
        )
        for relative in (
            builder.SOURCE_PROTOCOL_REL,
            builder.COMPREHENSIVE_PREREGISTRATION_REL,
            builder.TRANSPORT_LOCK_REL,
        ):
            self.assertFalse(os.path.lexists(builder.repo_path(relative)))

    def test_upstream_preregistration_hashes_are_reused_exactly(self) -> None:
        source = self.bundle.source_protocol
        binding = source["upstream_preregistration_binding"]
        self.assertEqual(
            binding["runner_preregistration"]["sha256"],
            builder.EXPECTED_RUNNER_PREREGISTRATION_SHA256,
        )
        self.assertEqual(
            binding["branch_preregistration"]["sha256"],
            builder.EXPECTED_BRANCH_PREREGISTRATION_SHA256,
        )
        self.assertEqual(
            binding["branch_preregistration"]["binding_sha256"],
            builder.EXPECTED_BRANCH_BINDING_SHA256,
        )
        self.assertEqual(
            binding["branch_preregistration"]["command_sha256"],
            builder.EXPECTED_BRANCH_COMMAND_SHA256,
        )

    def test_training_resume_reset_loss_and_steps_are_frozen(self) -> None:
        fixed = self.bundle.source_protocol["fixed_training"]
        self.assertEqual(
            fixed["resume_checkpoint"]["sha256"],
            builder.EXPECTED_FULL_INCUMBENT_U456_SHA256,
        )
        self.assertEqual(fixed["resume_checkpoint"]["update"], 456)
        self.assertEqual(fixed["updates"], list(range(457, 465)))
        self.assertEqual(fixed["minibatch_size"], 384)
        self.assertFalse(fixed["optimizer_state_loaded"])
        self.assertFalse(fixed["bc_replay_optimizer_state_loaded"])
        self.assertFalse(fixed["opponent_quota_state_loaded"])
        self.assertTrue(fixed["reset_optimizer_on_resume"])
        self.assertTrue(
            fixed["reset_bc_replay_optimizer_via_optimizer_reset"]
        )
        self.assertTrue(fixed["reset_opponent_quota_on_resume"])
        self.assertEqual(
            fixed["loss"]["identity"],
            "policy_loss + 0.25 * value_loss - "
            "0.001 * entropy + 0.012 * BC_KL",
        )
        self.assertEqual(
            fixed["optimizer_steps_each_update"],
            "2 * ceil(optimization transitions / 384)",
        )
        self.assertEqual(
            fixed["optimizer_steps_function"]["boundary_examples"],
            {"1": 2, "384": 2, "385": 4, "768": 4},
        )
        self.assertEqual(
            sum(fixed["fixed_opponent_quota_vector"].values()),
            64,
        )

    def test_single_knob_is_mechanical_but_not_claimed_causal(self) -> None:
        evidence = self.bundle.source_protocol[
            "single_selected_knob_command_audit"
        ]
        self.assertEqual(evidence["reference_minibatch_size"], 512)
        self.assertEqual(evidence["candidate_minibatch_size"], 384)
        self.assertEqual(len(evidence["differing_token_indices"]), 1)
        self.assertTrue(evidence["all_other_tokens_equal"])
        interpretation = self.bundle.source_protocol["interpretation"]
        self.assertFalse(interpretation["strict_causal_ablation"])
        self.assertTrue(interpretation["continued_training_confound"])
        self.assertIn(
            "U457-U464",
            interpretation["continued_training_confound_detail"],
        )

    def test_behavior_source_matches_final_runner_parse_contract(self) -> None:
        behavior = self.bundle.source_protocol["behavior_protocol"]
        implementation = behavior["implementation"]
        self.assertEqual(
            implementation["inprocess_runner_sha256"],
            FINAL_BEHAVIOR_RUNNER_SHA256,
        )
        self.assertEqual(implementation["device"], "cuda")
        self.assertEqual(implementation["workers"], 8)
        self.assertEqual(implementation["effective_workers"], 0)
        self.assertTrue(
            implementation["same_process_two_panel_runner_required"]
        )
        self.assertEqual(len(behavior["ordered_panels"]), 2)
        self.assertEqual(
            set(behavior["metric_mapping_and_gates"]),
            {
                "metrics_object",
                "context34_object",
                "old_retention",
                "valid29",
                "all_required",
            },
        )
        self.assertEqual(
            behavior["metric_mapping_and_gates"],
            builder.expected_gate_mapping(behavior["ordered_panels"]),
        )
        self.assertTrue(behavior["rules"]["all_20_gates_required"])

    def test_gold19_thresholds_are_reused_with_fresh_seeds(self) -> None:
        comprehensive = self.bundle.comprehensive_preregistration
        prior = builder.load_prior_vcoef_comprehensive()
        builder.validate_gold19_gate_reuse(
            prior,
            comprehensive["gold19_screen128_protocol"],
            comprehensive["gold19_confirm512_protocol_if_screen_passes"],
        )
        self.assertEqual(
            comprehensive["fresh_gold19_seeds"],
            [202607332, 202607333, 202607334, 202607335],
        )

    def test_every_sensitive_action_is_prohibited(self) -> None:
        prohibited = self.bundle.source_protocol["prohibited_actions"]
        self.assertEqual(
            set(prohibited),
            {
                "blind_result_or_replay_access",
                "current_gold_result_or_replay_access",
                "day30_content_access",
                "day31_content_access",
                "prior_gold19_result_reuse_or_pooling",
                "network",
                "package",
                "upload",
                "submission",
            },
        )
        self.assertTrue(all(prohibited.values()))

    def test_transport_lock_uses_final_launcher_schema_and_helpers(self) -> None:
        lock = self.bundle.transport_lock
        self.assertEqual(set(lock), set(self.launcher.transport_lock_keys))
        self.assertEqual(
            set(lock["child_command"]),
            set(self.launcher.child_command_keys),
        )
        self.assertEqual(
            set(lock["attempt"]),
            set(self.launcher.attempt_keys),
        )
        effective, indices = self.launcher.build_effective_command(
            self.upstream.command,
            root=builder.REPO_ROOT,
        )
        self.assertEqual(
            lock["child_command"]["effective_repo_path_token_indices"],
            list(indices),
        )
        self.assertEqual(
            lock["child_command"]["effective_command_canonical_sha256"],
            self.launcher.launcher_canonical_json_sha256(effective),
        )
        self.assertEqual(lock["child_command"]["effective_root_fd"], 197)
        self.assertEqual(lock["child_command"]["effective_trainer_fd"], 198)
        self.assertEqual(
            lock["child_command"]["sealed_trainer_payload_fd"],
            199,
        )
        self.assertEqual(
            lock["child_command"]["sealed_bootstrap_sha256"],
            builder.launcher_bootstrap_sha256(self.launcher),
        )
        self.assertEqual(
            lock["expected_transport"]["topology"],
            "supervised_fork_exec_v1",
        )
        self.assertEqual(
            lock["attempt"]["attempt_start_marker"],
            str(
                builder.launcher_attempt_start_marker(
                    self.launcher,
                    self.upstream.log,
                )
            ),
        )
        self.assertEqual(
            Path(lock["attempt"]["attempt_start_marker"]).parent,
            builder.REPO_ROOT,
        )
        self.assertTrue(
            Path(lock["attempt"]["attempt_start_marker"]).name.startswith(
                f".ptcg-ppo-attempt-{builder.TRAINING_SEED}-"
            )
        )
        self.assertEqual(
            lock["attempt"]["unpublished_log_witness"],
            str(
                self.upstream.log.with_name(
                    self.upstream.log.name + ".unpublished-witness"
                )
            ),
        )
        self.assertEqual(
            set(lock["attempt"]["absent_at_lock"]),
            {
                "output_dir",
                "log",
                "terminal_receipt",
                "terminal_checkpoint",
                "attempt_start_marker",
                "unpublished_log_witness",
            },
        )
        transport = self.bundle.source_protocol[
            "training_transport_binding"
        ]
        self.assertEqual(
            transport["launcher"]["sha256"],
            self.bundle.training_launcher_sha256,
        )
        self.assertEqual(transport["sealed_bootstrap_fd"], 198)
        self.assertEqual(transport["sealed_trainer_payload_fd"], 199)
        self.assertEqual(
            transport["topology"],
            "supervised_fork_exec_v1",
        )

    def test_all_three_json_trees_recursively_bind_receipt_writer(self) -> None:
        relative_binding = {
            "path": str(builder.TERMINAL_RECEIPT_WRITER_REL),
            "sha256": FINAL_RECEIPT_WRITER_SHA256,
        }
        absolute_binding = {
            "path": str(
                builder.repo_path(builder.TERMINAL_RECEIPT_WRITER_REL)
            ),
            "sha256": FINAL_RECEIPT_WRITER_SHA256,
        }
        for tree in (
            self.bundle.source_protocol,
            self.bundle.comprehensive_preregistration,
        ):
            matches = _named_values(tree, "terminal_receipt_writer")
            self.assertGreaterEqual(len(matches), 1)
            self.assertTrue(
                all(match == relative_binding for match in matches)
            )
        transport_matches = _named_values(
            self.bundle.transport_lock,
            "terminal_receipt_writer",
        )
        self.assertEqual(transport_matches, [absolute_binding])

    def test_serialized_bundle_is_self_consistent(self) -> None:
        source = json.loads(self.bundle.source_bytes)
        comprehensive = json.loads(self.bundle.comprehensive_bytes)
        lock = json.loads(self.bundle.transport_bytes)
        self.assertEqual(source, self.bundle.source_protocol)
        self.assertEqual(
            comprehensive,
            self.bundle.comprehensive_preregistration,
        )
        self.assertEqual(lock, self.bundle.transport_lock)
        self.assertEqual(
            lock["bindings"]["source_protocol"]["sha256"],
            builder.raw_sha256(self.bundle.source_bytes),
        )
        self.assertEqual(
            lock["bindings"]["comprehensive_preregistration"]["sha256"],
            builder.raw_sha256(self.bundle.comprehensive_bytes),
        )


class LocalHelperTest(unittest.TestCase):
    def test_optimizer_step_boundaries(self) -> None:
        self.assertEqual(builder.optimizer_steps_for_transitions(1), 2)
        self.assertEqual(builder.optimizer_steps_for_transitions(384), 2)
        self.assertEqual(builder.optimizer_steps_for_transitions(385), 4)
        self.assertEqual(builder.optimizer_steps_for_transitions(768), 4)
        with self.assertRaises(builder.ProtocolBuildError):
            builder.optimizer_steps_for_transitions(0)
        with self.assertRaises(builder.ProtocolBuildError):
            builder.optimizer_steps_for_transitions(True)

    def test_o_excl_success_and_collision_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            builder.write_files_o_excl(
                ((first, b'{"first":true}\n'), (second, b'{"second":true}\n'))
            )
            self.assertEqual(first.read_bytes(), b'{"first":true}\n')
            self.assertEqual(second.read_bytes(), b'{"second":true}\n')
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(second.stat().st_mode), 0o444)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            collision = root / "collision.json"
            collision.write_bytes(b"preserve me")
            with self.assertRaises(FileExistsError):
                builder.write_files_o_excl(
                    ((first, b"new"), (collision, b"must not replace"))
                )
            self.assertFalse(first.exists())
            self.assertEqual(collision.read_bytes(), b"preserve me")


if __name__ == "__main__":
    unittest.main()
