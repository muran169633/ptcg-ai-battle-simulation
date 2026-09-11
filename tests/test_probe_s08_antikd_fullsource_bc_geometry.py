from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/probe_s08_antikd_fullsource_bc_geometry.py"
SPEC = importlib.util.spec_from_file_location("s08_fullsource_geometry", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def row(episode: str, date: str, action_step: int = 1) -> dict:
    return {
        "dataset_date": date,
        "episode_id": episode,
        "seat": 0,
        "team_name": "fixture",
        "action_step_index": action_step,
        "action": [0],
        "select_context": "1",
        "min_count": 1,
        "max_count": 1,
    }


class TemporalSplitTests(unittest.TestCase):
    def test_frozen_time_forward_boundaries_and_counts(self) -> None:
        specs = {spec.name: spec for spec in probe.DOMAIN_SPECS}
        expected = {
            "anti_fit": (("2026-08-02", "2026-08-03"), 27, 3020),
            "anti_cal": (("2026-08-04", "2026-08-05"), 8, 885),
            "fros_fit": (
                ("2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"),
                89,
                9166,
            ),
            "fros_cal": (("2026-08-06",), 15, 1448),
            "general_fit": (
                ("2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05"),
                1620,
                158394,
            ),
            "general_cal": (("2026-08-06",), 142, 13730),
        }
        for name, (dates, episodes, rows) in expected.items():
            self.assertEqual(specs[name].dates, dates)
            self.assertEqual(probe.EXPECTED_DOMAIN_STATS[name]["episodes"], episodes)
            self.assertEqual(probe.EXPECTED_DOMAIN_STATS[name]["rows"], rows)
        for source in probe.SOURCE_ORDER:
            self.assertTrue(
                set(specs[f"{source}_fit"].dates).isdisjoint(
                    specs[f"{source}_cal"].dates
                )
            )

    def test_single_shard_is_fully_scanned_not_prefix_sampled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.zip"
            payload = b"".join(
                json.dumps(item, separators=(",", ":")).encode() + b"\n"
                for item in (
                    row("first", "2026-08-02", 1),
                    row("excluded", "2026-08-06", 2),
                    row("last", "2026-08-02", 3),
                )
            )
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("train/part-00000.jsonl", payload)
            spec = probe.DomainSpec(
                "fixture",
                "fixture",
                "fit",
                archive_path,
                "train",
                ("2026-08-02",),
            )
            observed = [item["episode_id"] for item in probe.iter_raw_domain_rows(spec)]
            self.assertEqual(observed, ["first", "last"])

    def test_partition_overlap_is_fail_closed(self) -> None:
        domains = {
            "anti_fit": frozenset({("d", "1", 0, "x")}),
            "anti_cal": frozenset({("d", "1", 0, "x")}),
            "anti_final": frozenset(),
            "fros_fit": frozenset(),
            "fros_cal": frozenset(),
            "fros_final": frozenset(),
            "general_fit": frozenset(),
            "general_cal": frozenset(),
            "general_final": frozenset(),
        }
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            probe.validate_temporal_partitions(domains)


class GradientAggregationTests(unittest.TestCase):
    def test_selection_effective_weight_aggregation_is_exact(self) -> None:
        accumulator = probe.WeightedGradientAccumulator(2)
        accumulator.add(torch.tensor([2.0, -2.0]), 1.0)
        accumulator.add(torch.tensor([4.0, 6.0]), 3.0)
        expected = torch.tensor([3.5, 4.0], dtype=torch.float64)
        self.assertTrue(torch.equal(accumulator.finish(), expected))
        self.assertEqual(accumulator.weight_sum, 4.0)
        self.assertEqual(accumulator.chunks, 2)

    def test_zero_or_invalid_effective_weight_rejected(self) -> None:
        accumulator = probe.WeightedGradientAccumulator(1)
        for value in (0.0, -1.0, float("nan")):
            with self.assertRaises(RuntimeError):
                accumulator.add(torch.ones(1), value)

    def test_gradient_once_proves_all_requested_parameters_unused(self) -> None:
        first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        second = torch.nn.Parameter(torch.tensor([3.0]))
        independent = torch.nn.Parameter(torch.tensor([4.0]))
        loss = independent.square().sum()
        vector, audit = probe._gradient_once(
            loss,
            (first, second),
            retain_graph=False,
            allow_unused=True,
        )
        self.assertTrue(torch.equal(vector, torch.zeros(3)))
        self.assertEqual(audit["unused_parameter_indices"], [0, 1])
        self.assertEqual(audit["unused_parameter_count"], 2)
        self.assertTrue(audit["all_parameters_unused_or_exact_zero"])

    def test_gradient_once_accepts_graphless_loss_only_for_unused_audit(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        vector, audit = probe._gradient_once(
            torch.tensor(3.0),
            (parameter,),
            retain_graph=False,
            allow_unused=True,
        )
        self.assertTrue(torch.equal(vector, torch.zeros(2)))
        self.assertFalse(audit["loss_requires_grad"])
        self.assertFalse(audit["autograd_invoked"])
        self.assertEqual(audit["unused_parameter_indices"], [0])
        with self.assertRaisesRegex(RuntimeError, "no autograd graph"):
            probe._gradient_once(
                torch.tensor(3.0),
                (parameter,),
                retain_graph=False,
                allow_unused=False,
            )

    def test_pairwise_gradient_matrix_is_symmetric_with_unit_diagonal(self) -> None:
        gradients = {
            "a": torch.tensor([1.0, 0.0]),
            "b": torch.tensor([1.0, 1.0]),
            "c": torch.tensor([0.0, -2.0]),
        }
        report = probe.pairwise_gradient_matrix(gradients, ("a", "b", "c"))
        self.assertEqual(report["targets"], ["a", "b", "c"])
        for matrix in (report["dot"], report["cosine"]):
            for row_index in range(3):
                for column_index in range(3):
                    self.assertAlmostEqual(
                        matrix[row_index][column_index],
                        matrix[column_index][row_index],
                    )
        for index in range(3):
            self.assertAlmostEqual(report["cosine"][index][index], 1.0)


class CandidateGeometryTests(unittest.TestCase):
    def test_candidates_are_exactly_latest_frozen_protocol(self) -> None:
        fit = {
            "anti_fit": torch.tensor([2.0, 0.0]),
            "fros_fit": torch.tensor([0.0, 1.0]),
            "general_fit": torch.tensor([0.0, 3.0]),
        }
        candidates, audit = probe.construct_candidates(fit)
        self.assertEqual(
            tuple(candidates),
            (
                "P1_raw_source_mean_2_1_1",
                "P2_unit_source_mean_2_1_1",
                "P3_fixed_cyclic_pcgrad",
            ),
        )
        self.assertEqual(audit["pcgrad"]["order"], ["anti", "fros", "general"])
        for direction in candidates.values():
            self.assertAlmostEqual(float(torch.linalg.vector_norm(direction)), 1.0)

    def test_fixed_cyclic_pcgrad_projects_conflicts_deterministically(self) -> None:
        gradients = {
            "anti": torch.tensor([1.0, 0.0]),
            "fros": torch.tensor([-1.0, 1.0]),
            "general": torch.tensor([0.0, 1.0]),
        }
        first, first_audit = probe.fixed_cyclic_pcgrad(gradients)
        second, second_audit = probe.fixed_cyclic_pcgrad(gradients)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_audit, second_audit)
        self.assertTrue(
            any(
                item["conflict_projection_applied"]
                for item in first_audit["projection_trace"]
            )
        )

    def _targets(self, vector: torch.Tensor, suffix: str) -> dict[str, torch.Tensor]:
        names = probe.FIT_TARGETS if suffix == "fit" else probe.CAL_TARGETS
        return {name: vector.clone() for name in names}

    def test_priority_selects_first_full_gate_candidate(self) -> None:
        candidates = {
            "P1_raw_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P2_unit_source_mean_2_1_1": torch.tensor([0.0, 1.0]),
            "P3_fixed_cyclic_pcgrad": torch.tensor([1.0, 1.0]),
        }
        y = torch.tensor([0.0, 1.0])
        selected, reports = probe.evaluate_and_select_candidates(
            candidates,
            self._targets(y, "fit"),
            self._targets(y, "cal"),
            self._targets(y, "fit"),
            self._targets(y, "cal"),
        )
        self.assertEqual(selected, "P2_unit_source_mean_2_1_1")
        self.assertFalse(
            reports["P3_fixed_cyclic_pcgrad"]["pcgrad_fallback_eligible"]
        )

    def test_pcgrad_fallback_uses_complete_base_gate_failures(self) -> None:
        candidates = {
            "P1_raw_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P2_unit_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P3_fixed_cyclic_pcgrad": torch.tensor([0.0, 1.0]),
        }
        y = torch.tensor([0.0, 1.0])
        selected, reports = probe.evaluate_and_select_candidates(
            candidates,
            self._targets(y, "fit"),
            self._targets(y, "cal"),
            self._targets(y, "fit"),
            self._targets(y, "cal"),
        )
        self.assertEqual(selected, "P3_fixed_cyclic_pcgrad")
        self.assertTrue(reports[selected]["pcgrad_fallback_eligible"])

    def test_pcgrad_can_fallback_when_earlier_fit_passes_but_cal_fails(self) -> None:
        candidates = {
            "P1_raw_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P2_unit_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P3_fixed_cyclic_pcgrad": torch.tensor([1.0, 1.0]),
        }
        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        selected, reports = probe.evaluate_and_select_candidates(
            candidates,
            self._targets(x, "fit"),
            self._targets(y, "cal"),
            self._targets(torch.tensor([1.0, 1.0]), "fit"),
            self._targets(torch.tensor([1.0, 1.0]), "cal"),
        )
        self.assertTrue(
            reports["P1_raw_source_mean_2_1_1"]["natural_fit_common_descent"]
        )
        self.assertFalse(
            reports["P1_raw_source_mean_2_1_1"]["base_selection_eligible"]
        )
        self.assertEqual(selected, "P3_fixed_cyclic_pcgrad")
        self.assertTrue(reports[selected]["pcgrad_fallback_eligible"])

    def test_legacy_gradient_must_be_non_ascent(self) -> None:
        candidates = {
            "P1_raw_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P2_unit_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P3_fixed_cyclic_pcgrad": torch.tensor([1.0, 0.0]),
        }
        positive = torch.tensor([1.0, 0.0])
        negative = torch.tensor([-1.0, 0.0])
        selected, reports = probe.evaluate_and_select_candidates(
            candidates,
            self._targets(positive, "fit"),
            self._targets(positive, "cal"),
            self._targets(negative, "fit"),
            self._targets(negative, "cal"),
        )
        self.assertIsNone(selected)
        self.assertFalse(
            reports["P1_raw_source_mean_2_1_1"]["legacy_fit_and_cal_non_ascent"]
        )

    def test_final_views_cannot_enter_selection_api(self) -> None:
        signature = inspect.signature(probe.evaluate_and_select_candidates)
        self.assertNotIn("final", signature.parameters)
        fit = self._targets(torch.tensor([1.0, 0.0]), "fit")
        fit["anti_final"] = torch.tensor([-1.0, 0.0])
        candidates = {
            "P1_raw_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P2_unit_source_mean_2_1_1": torch.tensor([1.0, 0.0]),
            "P3_fixed_cyclic_pcgrad": torch.tensor([1.0, 0.0]),
        }
        with self.assertRaisesRegex(RuntimeError, "six frozen fit/cal"):
            probe.evaluate_and_select_candidates(
                candidates,
                fit,
                self._targets(torch.tensor([1.0, 0.0]), "cal"),
                self._targets(torch.tensor([1.0, 0.0]), "fit"),
                self._targets(torch.tensor([1.0, 0.0]), "cal"),
            )


class StaticScopeTests(unittest.TestCase):
    def test_probe_has_one_autograd_site_and_no_update_or_model_write(self) -> None:
        raw = TOOL_PATH.read_bytes()
        audit = probe.static_scope_audit(raw)
        self.assertTrue(audit["one_autograd_grad_call_site"])
        self.assertTrue(audit["no_backward_optimizer_step_or_model_save_call_site"])
        tree = ast.parse(raw)
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue({"backward", "step", "save", "load_state_dict"}.isdisjoint(calls))

    def test_scope_declares_no_external_or_checkpoint_actions(self) -> None:
        source = TOOL_PATH.read_text()
        self.assertNotIn("torch.optim", source)
        self.assertNotIn("torch.save(", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("requests", source)


if __name__ == "__main__":
    unittest.main()
