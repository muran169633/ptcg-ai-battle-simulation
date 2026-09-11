from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TEMPLATE = (
    ROOT / "submission_templates/ptcg_ppo_alakazam_standard_pl_v1"
)
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


packaging = load_path(
    "package_alakazam_standard_ppo_v1_for_test",
    TOOLS / "package_alakazam_standard_ppo_v1.py",
)
train_ppo = load_path("train_ppo_alakazam_package_test", TOOLS / "train_ppo.py")


def load_template_main():
    prior_runtime = sys.modules.pop("policy_runtime", None)
    sys.path.insert(0, str(TEMPLATE))
    try:
        module = load_path(
            "ptcg_ppo_alakazam_standard_pl_v1_main_for_test",
            TEMPLATE / "main.py",
        )
    finally:
        sys.path.remove(str(TEMPLATE))
        sys.modules.pop("policy_runtime", None)
        if prior_runtime is not None:
            sys.modules["policy_runtime"] = prior_runtime
    return module


template_main = load_template_main()


class AlakazamStandardPPOTemplateTest(unittest.TestCase):
    def test_fixed_identity_audit_produces_read_only_existing_tool_plan(self) -> None:
        plan = packaging.audit_and_plan(ROOT)
        self.assertTrue(plan["read_only"])
        self.assertFalse(plan["archive_created"])
        self.assertFalse(plan["external_submission_performed"])
        self.assertEqual(
            plan["candidate"]["sha256"],
            packaging.EXPECTED_CANDIDATE_SHA256,
        )
        self.assertEqual(
            plan["deck"]["deck_hash"],
            packaging.EXPECTED_DECK_HASH,
        )
        self.assertTrue(plan["legacy_marnie_template_unchanged"])
        package_command = plan["package"]["command"]
        self.assertEqual(
            Path(package_command[1]),
            ROOT / "tools/package_ppo_submission.py",
        )
        self.assertEqual(
            Path(package_command[package_command.index("--template-dir") + 1]),
            TEMPLATE,
        )
        validation = plan["action_exact_validation"]
        self.assertEqual(
            Path(validation["command"][1]),
            ROOT / "tools/validate_ppo_submission.py",
        )
        self.assertIn("canonicalize_order=False", validation["comparison"])
        self.assertEqual(validation["required_action_mismatches"], 0)
        self.assertTrue(validation["required_before_submission"])

    def test_template_deck_and_runtime_are_independent_alakazam_assets(self) -> None:
        deck = packaging.read_deck(TEMPLATE / "deck.csv")
        self.assertEqual(len(deck), 60)
        self.assertEqual(
            packaging.compute_deck_hash(deck),
            packaging.EXPECTED_DECK_HASH,
        )
        self.assertEqual(template_main.agent({"select": None}), deck)
        self.assertEqual(
            packaging.sha256_file(TEMPLATE / "policy_runtime.py"),
            packaging.EXPECTED_TEMPLATE_FILES["policy_runtime.py"],
        )
        legacy = ROOT / "submissions/ptcg_ppo_terminal01_v1"
        for name, expected in packaging.LEGACY_MARNIE_FILES.items():
            self.assertEqual(packaging.sha256_file(legacy / name), expected)

    def test_decoder_is_action_exact_with_evaluator_full_plackett_luce(self) -> None:
        policy_logits = torch.tensor(
            [
                [0.1, 9.0, 3.0, 7.0],
                [4.0, 1.0, 8.0, -2.0],
            ],
            dtype=torch.float32,
        )
        count_logits = torch.full((2, 61), -10.0)
        count_logits[0, 3] = 5.0
        count_logits[1, 2] = 5.0
        outputs = {
            "policy_logits": policy_logits,
            "count_logits": count_logits,
            "value_logits": torch.zeros(2),
        }
        batch = {
            "min_counts": torch.tensor([3, 1]),
            "max_counts": torch.tensor([3, 3]),
            "option_mask": torch.tensor(
                [[True, True, True, True], [True, True, True, False]]
            ),
        }
        evaluator_actions, _, _, _ = train_ppo.sample_ordered_actions(
            outputs,
            batch,
            deterministic=True,
            canonicalize_order=False,
        )
        template_actions: list[list[int]] = []
        for row in range(2):
            option_count = int(batch["option_mask"][row].sum())
            count = template_main._greedy_legal_count(
                count_logits[row],
                minimum=int(batch["min_counts"][row]),
                maximum=int(batch["max_counts"][row]),
                option_count=option_count,
            )
            template_actions.append(
                template_main._greedy_plackett_luce(
                    policy_logits[row],
                    count=count,
                    option_count=option_count,
                )
            )
        self.assertEqual(template_actions, evaluator_actions)
        self.assertEqual(template_actions, [[1, 3, 2], [2, 0]])
        self.assertNotEqual(template_actions[0], sorted(template_actions[0]))
        self.assertNotEqual(template_actions[1], sorted(template_actions[1]))


if __name__ == "__main__":
    unittest.main()
