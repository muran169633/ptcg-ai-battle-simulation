from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_alakazam_corrected_v2_u10_postbc_actor6_20260810 as runner  # noqa: E402


class TinyActorPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_query = torch.nn.Linear(2, 2, bias=False)
        self.actor_key = torch.nn.Linear(2, 2, bias=False)
        self.actor_residual = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 2),
        )
        self.count_head = torch.nn.Linear(2, 2)
        self.value_head = torch.nn.Linear(2, 1)
        self.transformer_stub = torch.nn.Linear(2, 2)


class AlakazamCorrectedV2U10PostBCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parent = torch.load(
            runner.PARENT,
            map_location="cpu",
            weights_only=False,
        )

    def test_frozen_parent_identity_and_config_are_accepted(self) -> None:
        self.assertEqual(
            runner.repair.file_sha256(runner.PARENT),
            runner.PARENT_SHA256,
        )
        config = runner.validate_parent(self.parent)
        self.assertEqual(config.actor_reduction, "transition_mean")
        self.assertEqual(config.advantage_normalization, "per_opponent")
        self.assertEqual(config.gae_lambda, 1.0)
        self.assertEqual(config.bc_replay_context34_rows_per_batch, 4)

    def test_parent_config_and_identity_drift_fail_closed(self) -> None:
        for name, expected in runner.EXPECTED_PARENT_CONFIG.items():
            changed = copy.deepcopy(self.parent)
            replacement = (
                expected + 1
                if isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                else f"{expected}.drift"
            )
            changed["config"][name] = replacement
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    runner.validate_parent(changed)

        wrong_update = copy.deepcopy(self.parent)
        wrong_update["update"] = 11
        with self.assertRaisesRegex(ValueError, "update"):
            runner.validate_parent(wrong_update)

        wrong_deck = copy.deepcopy(self.parent)
        wrong_deck["learner_deck_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "deck hash"):
            runner.validate_parent(wrong_deck)

    def test_actor6_scope_freezes_every_other_parameter(self) -> None:
        model = TinyActorPolicy()
        parameters, named = runner.configure_actor6(model)
        self.assertEqual(len(parameters), 6)
        self.assertEqual(
            {
                name
                for name, parameter in named.items()
                if parameter.requires_grad
            },
            set(runner.ACTOR6),
        )
        self.assertFalse(model.count_head.weight.requires_grad)
        self.assertFalse(model.value_head.weight.requires_grad)
        self.assertFalse(model.transformer_stub.weight.requires_grad)

    def test_endpoint_integrity_allows_exactly_actor6(self) -> None:
        before = {
            name: torch.zeros(2)
            for name in (*runner.ACTOR6, "value_head.weight")
        }
        after = copy.deepcopy(before)
        for index, name in enumerate(runner.ACTOR6, start=1):
            after[name].add_(float(index))
        self.assertEqual(
            runner.validate_endpoint_state(before, after),
            sorted(runner.ACTOR6),
        )

        bad = copy.deepcopy(after)
        bad["value_head.weight"].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "outside actor6"):
            runner.validate_endpoint_state(before, bad)

        incomplete = copy.deepcopy(before)
        incomplete[runner.ACTOR6[0]].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "exactly actor6"):
            runner.validate_endpoint_state(before, incomplete)

    def test_endpoint_payload_is_evaluation_only_and_has_no_optimizer(self) -> None:
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.parent["model_state_dict"].items()
        }
        provenance = {"steps": 2, "schema_version": runner.SCHEMA_VERSION}
        payload = runner.endpoint_payload(self.parent, state, 2, provenance)
        self.assertTrue(payload["evaluation_only"])
        self.assertTrue(payload["resume_forbidden"])
        self.assertEqual(payload["update"], 10)
        self.assertEqual(payload["post_ppo_special_bc"], provenance)
        for name in runner.OMITTED_OPTIMIZER_KEYS:
            self.assertNotIn(name, payload)
        raw, digest = runner.serialize_checkpoint(payload)
        self.assertGreater(len(raw), 0)
        self.assertEqual(len(digest), 64)

    def test_two_phase_request_and_versioned_outputs(self) -> None:
        runner.validate_request(True, None)
        runner.validate_request(False, "a" * 64)
        with self.assertRaisesRegex(ValueError, "do not supply"):
            runner.validate_request(True, "a" * 64)
        with self.assertRaisesRegex(ValueError, "requires"):
            runner.validate_request(False, None)
        with self.assertRaisesRegex(ValueError, "hexadecimal"):
            runner.validate_request(False, "z" * 64)
        self.assertIn("20260810", runner.OUTPUT_ROOT.name)
        self.assertNotEqual(runner.OUTPUT_ROOT, runner.PREFLIGHT_ROOT)
        self.assertEqual(runner.ENDPOINTS, (2, 4))
        self.assertEqual(runner.LEARNING_RATE, 4e-7)
        self.assertEqual(runner.CONTEXT34_ROWS_PER_BATCH, 4)

    def test_bc_only_preselection_is_frozen_and_h2h_blind(self) -> None:
        self.assertEqual(
            runner.repair.file_sha256(runner.PARENT_BC_EVAL),
            runner.PARENT_BC_EVAL_SHA256,
        )
        baseline = json.loads(runner.PARENT_BC_EVAL.read_text())
        metrics = baseline["metrics"]
        context34 = metrics["by_context"]["34"]
        observed = {
            "rows": metrics["rows"],
            "count_correct": metrics["count_correct"],
            "set_exact_correct": metrics["set_exact_correct"],
            "hybrid_order_exact_correct": metrics[
                "hybrid_order_exact_correct"
            ],
            "ordered_exact_correct": metrics["ordered_exact_correct"],
            "context34_rows": context34["rows"],
            "context34_count_correct": context34["count_correct"],
            "context34_set_exact_correct": context34["set_exact_correct"],
            "context34_hybrid_order_exact_correct": context34[
                "hybrid_order_exact_correct"
            ],
            "context34_ordered_exact_correct": context34[
                "ordered_exact_correct"
            ],
        }
        self.assertEqual(observed, runner.BC_PRESELECTION_FLOORS)
        self.assertEqual(runner.BC_PRESELECTION_RANKING[-1], "smaller_endpoint_step")

    def test_required_isolated_cli_can_import_frozen_sibling_cores(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", runner.__file__, "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--expected-tool-sha256", completed.stdout)


if __name__ == "__main__":
    unittest.main()
