from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module():
    path = TOOLS / "plan_gold_push_marnie_hybrid.py"
    spec = importlib.util.spec_from_file_location(
        "plan_gold_push_marnie_hybrid_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner = load_module()


class GoldPushMarnieHybridPlanTest(unittest.TestCase):
    def test_read_only_plan_binds_hybrid_across_all_future_stages(self) -> None:
        plan = planner.audit_and_plan()
        self.assertTrue(plan["read_only"])
        self.assertEqual(plan["action_order_contract"]["mode"], "hybrid")
        self.assertFalse(plan["archive_created"])
        self.assertFalse(plan["external_submission_performed"])
        self.assertFalse(planner.HYBRID_PANEL_OUTPUT.exists())

        panel_command = plan["hybrid_panel"]["command"]
        self.assertIn("--candidate-hybrid-order", panel_command)
        self.assertNotIn("--candidate-raw-order", panel_command)
        self.assertNotIn("--candidate-canonical-order", panel_command)
        self.assertEqual(
            panel_command[panel_command.index("--output-dir") + 1],
            str(planner.HYBRID_PANEL_OUTPUT.resolve()),
        )

        package = plan["package_after_all_gates_only"]
        for key in ("build_a_command", "build_b_command"):
            command = package[key]
            self.assertEqual(
                command[command.index("--action-order-mode") + 1],
                "hybrid",
            )
            self.assertEqual(
                Path(command[command.index("--main-file") + 1]),
                planner.HYBRID_MAIN,
            )
        self.assertNotEqual(
            package["build_a_command"][
                package["build_a_command"].index("--archive") + 1
            ],
            package["build_b_command"][
                package["build_b_command"].index("--archive") + 1
            ],
        )
        validator = package["action_exact_32_command"]
        self.assertEqual(validator[validator.index("--games") + 1], "32")
        self.assertEqual(
            validator[validator.index("--action-order-mode") + 1],
            "hybrid",
        )

    def test_candidate_deck_and_template_are_hash_locked(self) -> None:
        plan = planner.audit_and_plan()
        self.assertEqual(
            plan["candidate"]["sha256"],
            planner.EXPECTED["candidate"],
        )
        self.assertEqual(
            plan["candidate"]["learner_deck_hash"],
            planner.DECK_HASH,
        )
        self.assertEqual(
            plan["template"]["template_version"],
            "ptcg-ppo-marnie-hybrid-order-v1",
        )


if __name__ == "__main__":
    unittest.main()
