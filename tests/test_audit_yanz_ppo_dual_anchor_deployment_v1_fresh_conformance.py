from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_yanz_ppo_dual_anchor_deployment_v1_fresh_conformance as audit  # noqa: E402


class FrozenContractTests(unittest.TestCase):
    def test_profiler_profile_models_main_runtime_and_archives_are_bound(self) -> None:
        static = audit.validate_frozen_bindings()
        self.assertEqual(
            audit.file_sha256(audit.DUAL_PROFILER_PATH), audit.DUAL_PROFILER_SHA256
        )
        self.assertEqual(
            audit.file_sha256(audit.ACCEPTED_PROFILE), audit.ACCEPTED_PROFILE_SHA256
        )
        self.assertEqual(
            static["ppo_parent_checkpoint"]["sha256"], audit.dual.PPO_PARENT_SHA256
        )
        self.assertEqual(
            static["pure_bc_checkpoint"]["sha256"], audit.dual.PURE_BC_SHA256
        )
        self.assertEqual(
            static["hybrid_main"]["sha256"], audit.dual.HYBRID_MAIN_SHA256
        )

    def test_accepted_profile_has_exact_counts_and_bitwise_anchors(self) -> None:
        profile = audit.load_json_strict(audit.ACCEPTED_PROFILE)
        audit.validate_profile_contract(profile, accepted=True)
        self.assertEqual(profile["cache"]["shard_count"], 122)
        self.assertEqual(
            profile["bindings"]["anchors"]["pure_bc_teacher"][
                "expanded_model_state_sha256"
            ],
            audit.dual.PURE_BC_EXPANDED61_BITWISE_STATE_SHA256,
        )
        self.assertEqual(
            profile["bindings"]["anchors"]["ppo_parent"]["model_state_sha256"],
            audit.dual.PPO_PARENT_BITWISE_STATE_SHA256,
        )

    def test_semantic_hash_excludes_only_cache_paths_and_hit_write_stats(self) -> None:
        accepted = audit.load_json_strict(audit.ACCEPTED_PROFILE)
        changed_execution = copy.deepcopy(accepted)
        changed_execution["source_counts"]["yanz"]["cache_hits"] = 4
        changed_execution["source_counts"]["yanz"]["cache_writes"] = 0
        changed_execution["source_counts"]["old"]["cache_hits"] = 118
        changed_execution["source_counts"]["old"]["cache_writes"] = 0
        for index, shard in enumerate(changed_execution["cache"]["immutable_shards"]):
            shard["path"] = f"/fresh-independent-cache/{index}.json"
        self.assertEqual(
            audit.semantic_profile_sha256(accepted),
            audit.semantic_profile_sha256(changed_execution),
        )
        changed_semantic = copy.deepcopy(changed_execution)
        changed_semantic["source_counts"]["yanz"]["ppo"]["hybrid_correct"] += 1
        self.assertNotEqual(
            audit.semantic_profile_sha256(accepted),
            audit.semantic_profile_sha256(changed_semantic),
        )


class ClassificationAndRuntimeTests(unittest.TestCase):
    def test_independent_four_way_category_truth_table(self) -> None:
        expected = {
            (True, True): "ppo_correct_protection",
            (True, False): "bc_correct_ppo_wrong_recovery_target",
            (False, True): "ppo_only_repair_protection",
            (False, False): "both_wrong",
        }
        self.assertEqual(
            {key: audit.independent_category(*key) for key in expected}, expected
        )
        self.assertEqual(tuple(audit.dual.CATEGORY_ORDER), audit.CATEGORIES)

    def test_both_loaded_anchors_are_exact_cpu1_fp32_bitwise_states(self) -> None:
        runtime = audit.frozen.load_bound_runtime()
        bc_model, ppo_model, _, anchor_audit = audit.dual.load_bound_models(runtime)
        exact = audit.validate_loaded_anchors(bc_model, ppo_model)
        self.assertEqual(exact["torch_num_threads"], 1)
        self.assertEqual(
            exact["pure_bc_expanded61_bitwise_state_sha256"],
            audit.dual.PURE_BC_EXPANDED61_BITWISE_STATE_SHA256,
        )
        self.assertEqual(
            exact["ppo_parent_bitwise_state_sha256"],
            audit.dual.PPO_PARENT_BITWISE_STATE_SHA256,
        )
        self.assertEqual(
            anchor_audit["ppo_parent"]["model_state_sha256"],
            audit.dual.PPO_PARENT_BITWISE_STATE_SHA256,
        )
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in bc_model.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in ppo_model.parameters()))

    def test_real_hybrid_main_matches_both_anchors_on_set_and_order_rows(self) -> None:
        runtime = audit.frozen.load_bound_runtime()
        bc_model, ppo_model, config, _ = audit.dual.load_bound_models(runtime)
        bc_main = audit.load_actual_hybrid_main(
            label="test_bc", runtime=runtime, model=bc_model, config=config
        )
        ppo_main = audit.load_actual_hybrid_main(
            label="test_ppo", runtime=runtime, model=ppo_model, config=config
        )
        wanted = {0, 34}
        found: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
        for chunk in audit.frozen.iter_train_chunks(
            audit.dual.YANZ, "yanz", chunk_size=64
        ):
            for row, identity in chunk["items"]:
                raw_context = row["observation"]["select"].get("context", -1)
                context = -1 if raw_context is None else int(raw_context)
                if context in wanted and context not in found:
                    found[context] = (row, identity)
            if set(found) == wanted:
                break
        self.assertEqual(set(found), wanted)
        for context, (row, identity) in sorted(found.items()):
            record = audit.dual.evaluate_dual_row(
                row,
                identity,
                "yanz",
                runtime,
                bc_model,
                ppo_model,
                config,
            )
            observation = dict(row["observation"])
            self.assertEqual(
                bc_main._policy_action(observation), record["bc_hybrid_action"], context
            )
            self.assertEqual(bc_main.agent(observation), record["bc_hybrid_action"], context)
            self.assertEqual(
                ppo_main._policy_action(observation), record["ppo_hybrid_action"], context
            )
            self.assertEqual(ppo_main.agent(observation), record["ppo_hybrid_action"], context)


class OutputBoundaryTests(unittest.TestCase):
    def test_second_cache_must_start_absent_and_cli_has_no_mutable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            fresh = root / "fresh.json"
            output = root / "audit.json"
            audit.assert_fresh_targets(cache, fresh, output)
            args = audit.parse_args(
                [
                    "--cache-dir",
                    str(cache),
                    "--fresh-profile",
                    str(fresh),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(args.cache_dir, cache)
            cache.mkdir()
            with self.assertRaisesRegex(FileExistsError, "initially absent"):
                audit.assert_fresh_targets(cache, fresh, output)
        for forbidden in ("--source", "--archive", "--profile", "--main", "--model"):
            with self.assertRaises(SystemExit):
                audit.parse_args(
                    [
                        "--cache-dir",
                        "cache",
                        "--fresh-profile",
                        "fresh.json",
                        "--output",
                        "audit.json",
                        forbidden,
                        "mutable",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
