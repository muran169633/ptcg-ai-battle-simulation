from __future__ import annotations

import json
import math
import copy
import sys
import unittest
import zipfile
from pathlib import Path

import orjson
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_yanz_source_error_actor6_v1 as profile  # noqa: E402
import run_yanz_actor6_pcgrad_specialbc_v1 as runner  # noqa: E402
import train_ppo as ppo  # noqa: E402


class ProfileContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint = torch.load(
            profile.SOURCE, map_location="cpu", weights_only=False
        )
        cls.raw, cls.config = profile.instantiate_raw_bc(
            cls.checkpoint, torch.device("cpu")
        )

    def test_three_explicit_state_hash_algorithms_are_not_conflated(self) -> None:
        self.assertEqual(
            profile.bitwise_model_state_sha256(
                self.checkpoint["model_state_dict"]
            ),
            profile.SOURCE_BITWISE_STATE_SHA256,
        )
        self.assertEqual(
            ppo.model_state_sha256(self.raw),
            profile.SOURCE_PPO_STYLE_STATE_SHA256,
        )
        expanded = ppo.instantiate_model_from_bc(
            self.checkpoint, torch.device("cpu")
        )
        self.assertEqual(
            ppo.model_state_sha256(expanded),
            profile.SOURCE_EXPANDED61_PPO_STYLE_STATE_SHA256,
        )
        self.assertEqual(expanded.count_head[-1].out_features, 61)
        self.assertEqual(self.raw.count_head[-1].out_features, 17)

    def test_training_selector_excludes_but_deployment_accepts_gt16(self) -> None:
        found = None
        with zipfile.ZipFile(profile.OLD) as archive:
            for member in sorted(
                name
                for name in archive.namelist()
                if name.startswith("train/") and name.endswith(".jsonl")
            ):
                with archive.open(member) as handle:
                    for line in handle:
                        row = orjson.loads(line)
                        if len(row.get("action", [])) > 16:
                            found = row
                            break
                if found is not None:
                    break
        self.assertIsNotNone(found)
        self.assertIsNone(profile.featurize_training(found, self.config))
        self.assertIsNotNone(profile.featurize_deployment(found, self.config))

    def test_descriptor_normalizes_infinite_margin_for_strict_json(self) -> None:
        row = {
            "source": "old",
            "archive_member": "train/x.jsonl",
            "member_line_number": 1,
            "raw_line_sha256": "0" * 64,
            "line_sha256": "1" * 64,
            "line_sha256_algorithm": "sha256(raw_line_rstrip_CR_LF)",
            "decision_key": "e:1->2",
            "episode_id": "e",
            "episode_uuid": "u",
            "visible_signature": "2" * 64,
            "context": 0,
            "team_name": "t",
            "sample_weight": 1.0,
            "expert_action_order": [],
            "source_policy_action": [],
            "source_set_correct": True,
            "source_hybrid_correct": True,
            "source_count_correct": True,
            "hybrid_margin": math.inf,
        }
        compact = profile.descriptor(row, "retention")
        self.assertIsNone(compact["hybrid_margin"])
        json.dumps(compact, allow_nan=False)

    @staticmethod
    def synthetic_descriptor(
        source: str, index: int, category: str, context: int = 0
    ) -> dict[str, object]:
        treatment = source == "yanz"
        prefix = "t" if treatment else "r"
        return {
            "source": source,
            "archive_member": "train/part-00000.jsonl",
            "member_line_number": index + 1,
            "raw_line_sha256": f"{index + (1 if treatment else 1000):064x}",
            "line_sha256": f"{index + (2000 if treatment else 3000):064x}",
            "line_sha256_algorithm": "sha256(raw_line_rstrip_CR_LF)",
            "decision_key": f"{prefix}{index}:1->2",
            "episode_id": f"{prefix}-episode-{index}",
            "episode_uuid": f"{prefix}-uuid-{index}",
            "visible_signature": f"{prefix}-signature-{index}",
            "context": context,
            "team_name": f"team-{index % 7}",
            "sample_weight": 1.0,
            "expert_action_order": [0],
            "source_policy_action": [1] if treatment else [0],
            "source_set_correct": not treatment,
            "source_hybrid_correct": not treatment,
            "source_count_correct": True,
            "hybrid_margin": -1.0 if treatment else 0.25 + index * 1e-6,
            "category": category,
        }

    @classmethod
    def synthetic_profile(cls) -> dict[str, object]:
        treatment_cursor = 0
        retention_cursor = 0
        steps = []
        for step_index, (treatment_size, boundary_size) in enumerate(
            zip(profile.TREATMENT_STEP_SIZES, profile.BOUNDARY_STEP_SIZES),
            start=1,
        ):
            treatment = [
                cls.synthetic_descriptor(
                    "yanz", treatment_cursor + offset, "treatment"
                )
                for offset in range(treatment_size)
            ]
            treatment_cursor += treatment_size
            boundary = [
                cls.synthetic_descriptor(
                    "old",
                    retention_cursor + offset,
                    "retention_boundary",
                    context=34 if offset < 4 else 0,
                )
                for offset in range(boundary_size)
            ]
            retention_cursor += boundary_size
            broad = [
                cls.synthetic_descriptor(
                    "old",
                    retention_cursor + offset,
                    "retention_broad",
                    context=(offset % 10),
                )
                for offset in range(profile.BROAD_PER_STEP)
            ]
            retention_cursor += profile.BROAD_PER_STEP
            steps.append(
                {
                    "step": step_index,
                    "treatment": treatment,
                    "retention_boundary": boundary,
                    "retention_broad": broad,
                    "retention": boundary + broad,
                }
            )
        return {
            "schema_version": profile.SCHEMA,
            "status": "frozen_train_only_row_selection",
            "bindings": {
                "source_checkpoint": {"sha256": profile.SOURCE_SHA256},
                "yanz_train_archive": {"sha256": profile.YANZ_SHA256},
                "old_train_archive": {"sha256": profile.OLD_SHA256},
                "deck_hash": profile.DECK_HASH,
            },
            "source_counts": {
                "yanz_set_correct": 1422,
                "yanz_hybrid_correct": 1422,
                "yanz_count_correct": 1723,
                "yanz_set_errors": 303,
                "treatment_selection_only": 301,
                "excluded_count_errors": 2,
                "retention_selected": 467,
            },
            "steps": steps,
        }

    def test_validate_profile_locks_all_row_quotas_predicates_and_overlap(self) -> None:
        fixture = self.synthetic_profile()
        runner.validate_profile(fixture)
        self.assertEqual(
            [len(step["treatment"]) for step in fixture["steps"]],
            [101, 100, 100],
        )
        self.assertEqual(
            [len(step["retention"]) for step in fixture["steps"]],
            [155, 156, 156],
        )

        bad_margin = copy.deepcopy(fixture)
        bad_margin["steps"][0]["retention_broad"][0]["hybrid_margin"] = None
        bad_margin["steps"][0]["retention"] = (
            bad_margin["steps"][0]["retention_boundary"]
            + bad_margin["steps"][0]["retention_broad"]
        )
        with self.assertRaisesRegex(runner.ProtocolError, "retention source/margin"):
            runner.validate_profile(bad_margin)

        overlap = copy.deepcopy(fixture)
        overlap["steps"][0]["retention_boundary"][0]["visible_signature"] = (
            overlap["steps"][0]["treatment"][0]["visible_signature"]
        )
        overlap["steps"][0]["retention"] = (
            overlap["steps"][0]["retention_boundary"]
            + overlap["steps"][0]["retention_broad"]
        )
        with self.assertRaisesRegex(runner.ProtocolError, "episode or visible"):
            runner.validate_profile(overlap)

    def test_step_behavior_uses_real_expanded61_clone(self) -> None:
        row, _ = next(iter(profile.raw_train_rows(profile.YANZ)))
        feature = profile.featurize_training(row, self.config)
        self.assertIsNotNone(feature)
        feature["expert_action_order"] = [int(value) for value in row["action"]]
        feature["line_sha256"] = "a" * 64
        batch = runner.collate_features(
            [feature], self.config, torch.device("cpu")
        )
        expanded = ppo.instantiate_model_from_bc(
            self.checkpoint, torch.device("cpu")
        )
        result = runner.batch_behavior(
            self.raw, expanded, batch, ["a" * 64]
        )
        self.assertEqual(set(result["correct"]), {"a" * 64})
        self.assertEqual(expanded.count_head[-1].out_features, 61)


class LossAndPCGradTests(unittest.TestCase):
    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        return {
            "option_mask": torch.tensor([[True, True], [True, True]]),
            "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "action_counts": torch.tensor([1, 1]),
            "action_sequences": torch.tensor([[0] + [-1] * 15, [1] + [-1] * 15]),
            "contexts": torch.tensor([34, 0]),
            "sample_weights": torch.tensor([2.0, 0.5]),
            "min_counts": torch.tensor([1, 0]),
            "max_counts": torch.tensor([1, 2]),
        }

    def test_policy_loss_has_no_count_or_value_dependency_and_includes_ctx34_bce(self) -> None:
        policy = torch.tensor([[0.5, -0.5], [-0.2, 0.2]], requires_grad=True)
        count = torch.randn(2, 17, requires_grad=True)
        value = torch.randn(2, requires_grad=True)
        outputs = {
            "policy_logits": policy,
            "count_logits": count,
            "value_logits": value,
        }
        per_row, weights = runner.hybrid_policy_nll_per_row(
            outputs, self.batch()
        )
        self.assertEqual(weights.tolist(), [16.0, 0.5])
        ordered = -torch.log_softmax(policy[0], dim=0)[0]
        ctx34_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            policy[0], torch.tensor([1.0, 0.0])
        )
        self.assertTrue(
            torch.allclose(per_row[0], ordered + 0.25 * ctx34_bce)
        )
        loss = runner.weighted_policy_loss(outputs, self.batch())
        _, repository_parts = ppo.bc_expert_actor_loss(
            outputs,
            self.batch(),
            loss_mode="hybrid_ordered",
            order_context_weight=8.0,
        )
        repository_policy_component = (
            repository_parts["selection_loss"]
            + 0.25 * repository_parts["set_bce_loss"]
        )
        self.assertTrue(torch.allclose(loss, repository_policy_component))
        gradients = torch.autograd.grad(
            loss, (policy, count, value), allow_unused=True
        )
        self.assertIsNotNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])

    def test_symmetric_pcgrad_projects_conflict_and_descends_both(self) -> None:
        treatment = (torch.tensor([1.0, 0.0]),)
        retention = (torch.tensor([-0.5, 1.0]),)
        direction, audit = runner.symmetric_two_task_pcgrad(
            treatment, retention
        )
        self.assertTrue(audit["projection_applied"])
        self.assertTrue(audit["pass"])
        self.assertGreater(float(runner.gradient_dot(direction, treatment)), 0.0)
        self.assertGreater(float(runner.gradient_dot(direction, retention)), 0.0)


class ArtifactTests(unittest.TestCase):
    def test_candidate_checkpoint_stays_raw17_80_tensor_training_artifact(self) -> None:
        checkpoint = torch.load(
            profile.SOURCE, map_location="cpu", weights_only=False
        )
        model, _ = profile.instantiate_raw_bc(
            checkpoint, torch.device("cpu")
        )
        with torch.no_grad():
            dict(model.named_parameters())[runner.ACTOR6[0]].view(-1)[0].add_(1e-4)
        state_hash = profile.bitwise_model_state_sha256(model.state_dict())
        output = runner.candidate_checkpoint(
            checkpoint,
            model,
            "a" * 64,
            {"candidate_model_state_sha256": state_hash},
        )
        self.assertEqual(len(output["model_state_dict"]), 80)
        self.assertEqual(output["model_state_dict"]["count_head.2.weight"].shape[0], 17)
        self.assertFalse(output["special_bc"]["direct_submission_ready"])
        changed = sorted(
            name
            for name, value in checkpoint["model_state_dict"].items()
            if not torch.equal(value, output["model_state_dict"][name])
        )
        self.assertEqual(changed, [runner.ACTOR6[0]])

    def test_double_replay_comparison_includes_steps_and_both_hashes(self) -> None:
        report = {
            "steps": [{"step": 1, "loss": 1.0}],
            "pass": False,
            "terminal_model_state_sha256": "1" * 64,
            "terminal_train_ppo_model_state_sha256": "2" * 64,
            "ignored": "not compared",
        }
        comparable = runner.replay_comparable(report)
        self.assertNotIn("ignored", comparable)
        self.assertEqual(
            runner.canonical_json_bytes(comparable),
            runner.canonical_json_bytes(runner.replay_comparable(copy.deepcopy(report))),
        )


if __name__ == "__main__":
    unittest.main()
