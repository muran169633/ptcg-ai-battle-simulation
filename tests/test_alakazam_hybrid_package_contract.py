from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
HYBRID_TEMPLATE = ROOT / "submission_templates/ptcg_ppo_alakazam_hybrid_order_v1"
BASE_TEMPLATE = ROOT / "submission_templates/ptcg_ppo_alakazam_standard_pl_v1"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


packager = load_path(
    "package_ppo_submission_alakazam_hybrid_contract_test",
    TOOLS / "package_ppo_submission.py",
)
evaluator = load_path(
    "evaluate_ppo_head_to_head_alakazam_hybrid_contract_test",
    TOOLS / "evaluate_ppo_head_to_head.py",
)


def load_hybrid_main():
    prior_runtime = sys.modules.pop("policy_runtime", None)
    sys.path.insert(0, str(BASE_TEMPLATE))
    try:
        return load_path(
            "ptcg_ppo_alakazam_hybrid_order_v1_main_for_test",
            HYBRID_TEMPLATE / "main.py",
        )
    finally:
        sys.path.remove(str(BASE_TEMPLATE))
        sys.modules.pop("policy_runtime", None)
        if prior_runtime is not None:
            sys.modules["policy_runtime"] = prior_runtime


hybrid_main = load_hybrid_main()


class AlakazamHybridPackageContractTest(unittest.TestCase):
    def source_files(self) -> dict[str, Path]:
        return {
            "main.py": HYBRID_TEMPLATE / "main.py",
            "deck.csv": BASE_TEMPLATE / "deck.csv",
            "policy_runtime.py": BASE_TEMPLATE / "policy_runtime.py",
        }

    def test_contract_binds_exact_hybrid_sources_and_deck(self) -> None:
        contract = packager.read_template_contract(
            HYBRID_TEMPLATE / "CONTRACT.json",
            action_order_mode="hybrid",
            source_files=self.source_files(),
        )
        self.assertEqual(contract["decode"]["order_mode"], "hybrid")
        self.assertEqual(contract["decode"]["preserve_order_contexts"], [34])
        self.assertEqual(
            contract["deck"]["semantic_hash"],
            hybrid_main.DECK_HASH,
        )
        with self.assertRaisesRegex(ValueError, "action-order mode"):
            packager.read_template_contract(
                HYBRID_TEMPLATE / "CONTRACT.json",
                action_order_mode="raw",
                source_files=self.source_files(),
            )

    def test_hybrid_helper_matches_evaluator(self) -> None:
        action = [2, 0, 1]
        for context in (34, 7, 0, 43):
            with self.subTest(context=context):
                self.assertEqual(
                    hybrid_main._apply_hybrid_order(list(action), context),
                    evaluator.apply_hybrid_action_order(
                        list(action),
                        select_context=context,
                        enabled=True,
                    ),
                )

    def test_policy_action_preserves_34_and_sorts_other_contexts(self) -> None:
        class DummyModel:
            def __call__(self, _batch):
                count_logits = torch.full((1, 61), -10.0)
                count_logits[0, 3] = 5.0
                return {
                    "policy_logits": torch.tensor([[0.1, 9.0, 3.0, 7.0]]),
                    "count_logits": count_logits,
                    "value_logits": torch.zeros(1),
                }

        config = {
            "hash_size": 8,
            "max_state_entities": 4,
            "entity_fields": 20,
            "option_fields": 24,
        }

        def run(context: int) -> list[int]:
            observation = {
                "current": {"yourIndex": 0},
                "select": {
                    "context": context,
                    "minCount": 3,
                    "maxCount": 3,
                    "option": [{}, {}, {}, {}],
                },
            }
            with mock.patch.object(
                hybrid_main,
                "_load_model",
                return_value=(DummyModel(), config),
            ), mock.patch.object(
                hybrid_main,
                "featurize_row",
                return_value={"synthetic": True},
            ), mock.patch.object(
                hybrid_main,
                "collate_decisions",
                return_value={"synthetic": True},
            ):
                return hybrid_main._policy_action(observation)

        self.assertEqual(run(34), [1, 3, 2])
        self.assertEqual(run(7), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
