from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_gold_push_postppo_tail_repair as repair  # noqa: E402


class ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(3, 3)
        self.actor_query = torch.nn.Linear(3, 3, bias=False)
        self.actor_key = torch.nn.Linear(3, 3, bias=False)
        self.actor_residual = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 1),
        )
        self.count_head = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 2),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 1),
        )


class PostPpoTailRepairTests(unittest.TestCase):
    def test_frozen_archive_hashes_are_current(self) -> None:
        expected = (
            (repair.RAIHAN_ARCHIVE, repair.RAIHAN_SHA256),
            (repair.KD_ARCHIVE, repair.KD_SHA256),
            (repair.GENERAL_ARCHIVE, repair.GENERAL_SHA256),
        )
        for path, digest in expected:
            with self.subTest(path=path.name):
                self.assertEqual(repair.file_sha256(path), digest)

    def test_exact_alternating_schedule_and_endpoints(self) -> None:
        repair.validate_protocol_constants()
        self.assertEqual(repair.ENDPOINTS, (2, 4, 8))
        self.assertEqual(len(repair.STEP_SCHEDULE), 8)
        self.assertEqual(
            tuple(source for source, _ in repair.STEP_SCHEDULE),
            ("raihan", "kdcyberdude") * 4,
        )
        self.assertEqual(
            tuple(index for source, index in repair.STEP_SCHEDULE if source == "raihan"),
            repair.RAIHAN_BATCH_INDICES,
        )
        self.assertEqual(
            tuple(
                index
                for source, index in repair.STEP_SCHEDULE
                if source == "kdcyberdude"
            ),
            repair.KD_BATCH_INDICES,
        )
        self.assertNotIn("general_valid", {source for source, _ in repair.STEP_SCHEDULE})

    def test_raw_multi_action_order_is_explicitly_weighted(self) -> None:
        self.assertEqual(repair.ORDER_CONTEXT_WEIGHT, 8.0)
        self.assertEqual(repair.RAW_FIXED_MULTI_ACTION_ORDER_WEIGHT, 2.0)
        self.assertEqual(repair.TRAIN_CONTEXT34_ROWS_PER_BATCH, 1)
        self.assertEqual(repair.VALID_CONTEXT34_ROWS_PER_BATCH, 4)

    def test_actor6_configuration_freezes_count_value_and_trunk(self) -> None:
        model = ToyPolicy()
        parameters = repair.configure_actor6(model)
        trainable = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        self.assertEqual(trainable, repair.ACTOR6)
        self.assertEqual(len(parameters), 6)
        for name, parameter in model.named_parameters():
            if name.startswith(("count_head.", "value_head.", "trunk.")):
                self.assertFalse(parameter.requires_grad, name)

    def test_endpoint_integrity_accepts_only_exact_actor6_delta(self) -> None:
        model = ToyPolicy()
        before = repair.clone_model_state(model)
        after = {name: tensor.clone() for name, tensor in before.items()}
        for name in repair.ACTOR6:
            after[name].add_(0.25)
        result = repair.validate_endpoint_state(before, after)
        self.assertTrue(result["changed_exactly_actor6"])
        self.assertTrue(result["count_head_bit_identical"])
        self.assertTrue(result["value_head_bit_identical"])

    def test_endpoint_integrity_rejects_count_or_value_mutation(self) -> None:
        model = ToyPolicy()
        before = repair.clone_model_state(model)
        for forbidden in ("count_head.0.weight", "value_head.0.weight"):
            with self.subTest(forbidden=forbidden):
                after = {name: tensor.clone() for name, tensor in before.items()}
                for name in repair.ACTOR6:
                    after[name].add_(0.25)
                after[forbidden].add_(0.25)
                with self.assertRaisesRegex(RuntimeError, "outside actor6"):
                    repair.validate_endpoint_state(before, after)

    def test_manifest_binds_parent_archives_schedule_and_read_only_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_path = root / "parent.pt"
            bc_path = root / "bc.pt"
            output_dir = root / "new-output"
            parent = {
                "feature_version": repair.ppo.PPO_FEATURE_VERSION,
                "update": 123,
                "learner_deck_hash": repair.MARNIE_DECK_HASH,
                "model_state_dict": {"actor_query.weight": torch.ones(1)},
                "config": {"seed": 1},
                "model_config": {"model_dim": 3},
            }
            cache_records = {
                "raihan": {
                    "sha256": "a" * 64,
                    "used_for_optimizer_steps": True,
                },
                "kdcyberdude": {
                    "sha256": "b" * 64,
                    "used_for_optimizer_steps": True,
                },
                "general_valid": {
                    "sha256": "c" * 64,
                    "used_for_optimizer_steps": False,
                },
            }
            manifest = repair.build_manifest_payload(
                parent_path=parent_path,
                parent_sha256="d" * 64,
                parent=parent,
                bc_path=bc_path,
                bc_sha256="e" * 64,
                output_dir=output_dir,
                cache_records=cache_records,
                actor_shapes={name: [3, 3] for name in repair.ACTOR6},
                tool_sha256="f" * 64,
                trainer_sha256="1" * 64,
                python_sha256="2" * 64,
                device="cuda",
            )
        self.assertEqual(
            manifest["inputs"]["parent_checkpoint"]["sha256"], "d" * 64
        )
        self.assertEqual(
            manifest["protocol"]["trainable_parameter_names"], list(repair.ACTOR6)
        )
        self.assertEqual(
            manifest["protocol"]["endpoints"], list(repair.ENDPOINTS)
        )
        self.assertFalse(
            manifest["replay_caches"]["general_valid"]["used_for_optimizer_steps"]
        )
        self.assertEqual(manifest["guards"]["general_valid_rows_used_for_training"], 0)
        self.assertTrue(manifest["guards"]["count_head_must_be_bit_identical"])
        self.assertTrue(manifest["guards"]["value_head_must_be_bit_identical"])
        self.assertTrue(manifest["guards"]["each_specialist_loss_must_improve"])
        self.assertTrue(
            manifest["guards"][
                "each_specialist_raw_multi_ordered_loss_must_improve"
            ]
        )
        self.assertFalse(manifest["scope"]["package"])
        self.assertFalse(manifest["scope"]["upload"])
        self.assertFalse(manifest["scope"]["submission"])

    def test_frozen_manifest_requires_reviewed_digest_and_detects_tampering(self) -> None:
        manifest = {"schema_version": repair.SCHEMA_VERSION, "value": 7}
        envelope = repair.frozen_manifest_envelope(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_bytes(repair.canonical_json_bytes(envelope))
            loaded = repair.load_frozen_manifest(path, repair.sha256_json(manifest))
            self.assertEqual(loaded, manifest)
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                repair.load_frozen_manifest(path, "0" * 64)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["manifest"]["value"] = 8
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid embedded digest"):
                repair.load_frozen_manifest(path, repair.sha256_json(manifest))

    def test_execute_mode_requires_a_frozen_manifest_and_digest(self) -> None:
        for frozen, digest in ((None, None), (Path("frozen.json"), None), (None, "a" * 64)):
            with self.subTest(frozen=frozen, digest=digest):
                args = SimpleNamespace(
                    execute=True,
                    frozen_manifest=frozen,
                    expected_manifest_sha256=digest,
                )
                with self.assertRaisesRegex(ValueError, "requires"):
                    repair.validate_mode_args(args)
        repair.validate_mode_args(
            SimpleNamespace(
                execute=True,
                frozen_manifest=Path("frozen.json"),
                expected_manifest_sha256="a" * 64,
            )
        )

    def test_freeze_mode_rejects_execute_only_bindings(self) -> None:
        args = SimpleNamespace(
            execute=False,
            frozen_manifest=Path("frozen.json"),
            expected_manifest_sha256=None,
        )
        with self.assertRaisesRegex(ValueError, "execute-only"):
            repair.validate_mode_args(args)

    def test_output_directory_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = root / "absent"
            repair.assert_output_absent(absent)
            absent.mkdir()
            with self.assertRaises(FileExistsError):
                repair.assert_output_absent(absent)

    def test_endpoint_payload_is_evaluation_only_without_optimizers(self) -> None:
        parent = {
            "feature_version": repair.ppo.PPO_FEATURE_VERSION,
            "config": {"seed": 1},
            "model_config": {"model_dim": 3},
            "learner_deck_hash": repair.MARNIE_DECK_HASH,
            "update": 55,
            "optimizer_state_dict": {"must": "not survive"},
            "bc_replay_optimizer_state_dict": {"must": "not survive"},
        }
        payload = repair.endpoint_payload(
            parent=parent,
            state={"actor_query.weight": torch.ones(1)},
            manifest_sha256="a" * 64,
            step=2,
            eligible=False,
            guard={"general_valid_pass": False},
        )
        self.assertTrue(payload["evaluation_only"])
        self.assertTrue(payload["resume_forbidden"])
        self.assertNotIn("optimizer_state_dict", payload)
        self.assertNotIn("bc_replay_optimizer_state_dict", payload)
        self.assertEqual(
            payload["post_ppo_special_bc"]["general_valid_rows_used_for_training"],
            0,
        )
        self.assertFalse(payload["post_ppo_special_bc"]["promotion_eligible"])

    @staticmethod
    def guard_metrics(
        *,
        raihan_loss: float,
        raihan_raw: float,
        kd_loss: float,
        kd_raw: float,
        general_loss: float,
        general_raw: float,
        general_context34: float,
    ) -> dict[str, dict[str, float]]:
        return {
            "raihan": {
                "loss": raihan_loss,
                "non_context34_fixed_multi_action_ordered_loss": raihan_raw,
            },
            "kdcyberdude": {
                "loss": kd_loss,
                "non_context34_fixed_multi_action_ordered_loss": kd_raw,
            },
            "general_valid": {
                "loss": general_loss,
                "non_context34_fixed_multi_action_ordered_loss": general_raw,
                "context_34_ordered_loss": general_context34,
            },
        }

    def test_endpoint_guard_requires_both_specialists_and_raw_multi_improve(
        self,
    ) -> None:
        baseline = self.guard_metrics(
            raihan_loss=1.0,
            raihan_raw=2.0,
            kd_loss=1.0,
            kd_raw=2.0,
            general_loss=1.0,
            general_raw=2.0,
            general_context34=1.5,
        )
        safe = self.guard_metrics(
            raihan_loss=0.99,
            raihan_raw=1.99,
            kd_loss=0.99,
            kd_raw=1.99,
            general_loss=1.001,
            general_raw=2.005,
            general_context34=1.505,
        )
        self.assertTrue(
            repair.build_endpoint_guard(
                baseline=baseline,
                current=safe,
            )["promotion_eligible"]
        )

        # The mean specialist loss still improves, but Raihan's raw-order
        # subgroup regresses.  It must be quarantined.
        unsafe = self.guard_metrics(
            raihan_loss=0.99,
            raihan_raw=2.02,
            kd_loss=0.98,
            kd_raw=1.90,
            general_loss=1.0,
            general_raw=2.0,
            general_context34=1.5,
        )
        guard = repair.build_endpoint_guard(
            baseline=baseline,
            current=unsafe,
        )
        self.assertTrue(guard["specialist_mean_improved"])
        self.assertFalse(
            guard["each_specialist_raw_multi_ordered_loss_improved"]
        )
        self.assertFalse(guard["promotion_eligible"])

    def test_endpoint_guard_caps_general_component_regressions(self) -> None:
        baseline = self.guard_metrics(
            raihan_loss=1.0,
            raihan_raw=2.0,
            kd_loss=1.0,
            kd_raw=2.0,
            general_loss=1.0,
            general_raw=2.0,
            general_context34=1.5,
        )
        current = self.guard_metrics(
            raihan_loss=0.99,
            raihan_raw=1.99,
            kd_loss=0.99,
            kd_raw=1.99,
            # Overall general loss looks safe while raw order exceeds its
            # independently frozen component cap.
            general_loss=1.001,
            general_raw=2.02,
            general_context34=1.5,
        )
        guard = repair.build_endpoint_guard(
            baseline=baseline,
            current=current,
        )
        self.assertTrue(guard["general_valid_pass"])
        self.assertFalse(guard["general_valid_raw_multi_ordered_pass"])
        self.assertFalse(guard["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
