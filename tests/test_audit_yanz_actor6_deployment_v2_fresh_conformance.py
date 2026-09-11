from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_yanz_actor6_deployment_v2_fresh_conformance as audit  # noqa: E402
import profile_yanz_source_error_actor6_deployment_v2 as profiler  # noqa: E402


class FrozenBindingAndSemanticHashTests(unittest.TestCase):
    def test_frozen_profiler_profile_main_runtime_and_source_bindings(self) -> None:
        static = audit.validate_frozen_bindings()
        self.assertEqual(audit.file_sha256(audit.PROFILER), audit.PROFILER_SHA256)
        self.assertEqual(
            audit.file_sha256(audit.ACCEPTED_PROFILE),
            audit.ACCEPTED_PROFILE_SHA256,
        )
        self.assertEqual(
            static["hybrid_main"]["sha256"], profiler.HYBRID_MAIN_SHA256
        )
        self.assertEqual(
            static["policy_runtime"]["sha256"], profiler.RUNTIME_SHA256
        )
        self.assertEqual(
            static["source_checkpoint"]["sha256"], profiler.SOURCE_SHA256
        )

    def test_semantic_hash_ignores_only_cache_paths_and_hit_write_stats(self) -> None:
        accepted = audit.load_json_strict(audit.ACCEPTED_PROFILE)
        audit.validate_profile_contract(accepted, accepted=True)
        changed_execution = copy.deepcopy(accepted)
        changed_execution["source_counts"]["yanz"]["cache_hits"] = 4
        changed_execution["source_counts"]["yanz"]["cache_writes"] = 0
        changed_execution["source_counts"]["old"]["cache_hits"] = 118
        changed_execution["source_counts"]["old"]["cache_writes"] = 0
        for index, shard in enumerate(changed_execution["cache"]["immutable_shards"]):
            shard["path"] = f"/independent/fresh/cache/{index}.json"
        self.assertEqual(
            audit.semantic_profile_sha256(accepted),
            audit.semantic_profile_sha256(changed_execution),
        )
        changed_semantic = copy.deepcopy(changed_execution)
        changed_semantic["source_counts"]["retention_selected"] += 1
        self.assertNotEqual(
            audit.semantic_profile_sha256(accepted),
            audit.semantic_profile_sha256(changed_semantic),
        )

    def test_second_cache_must_be_absent_and_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            fresh_profile = root / "fresh.json"
            output = root / "audit.json"
            audit.assert_fresh_targets(cache, fresh_profile, output)
            cache.mkdir()
            with self.assertRaisesRegex(FileExistsError, "initially absent"):
                audit.assert_fresh_targets(cache, fresh_profile, output)


class RealHybridMainConformanceTests(unittest.TestCase):
    def test_real_policy_action_and_agent_match_fresh_semantics_on_both_contexts(self) -> None:
        runtime = profiler.load_bound_runtime()
        _, model, config, _ = profiler.instantiate_deployment_source(runtime)
        main_module = audit.load_actual_hybrid_main(runtime, model, config)
        wanted = {0, 34}
        found: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
        for chunk in profiler.iter_train_chunks(profiler.YANZ, "yanz", chunk_size=64):
            for row, identity in chunk["items"]:
                raw_context = row["observation"]["select"].get("context", -1)
                context = -1 if raw_context is None else int(raw_context)
                if context in wanted and context not in found:
                    found[context] = (row, identity)
            if set(found) == wanted:
                break
        self.assertEqual(set(found), wanted)
        for context, (row, identity) in sorted(found.items()):
            record = profiler.evaluate_row(
                row, identity, "yanz", runtime, model, config
            )
            policy_action = main_module._policy_action(dict(row["observation"]))
            agent_action = main_module.agent(dict(row["observation"]))
            self.assertEqual(policy_action, record["source_hybrid_action"], context)
            self.assertEqual(agent_action, record["source_hybrid_action"], context)
        self.assertEqual(main_module.agent({}), list(main_module.DECK))

    def test_cli_exposes_no_mutable_model_archive_or_profile_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = audit.parse_args(
                [
                    "--cache-dir",
                    str(root / "cache"),
                    "--fresh-profile",
                    str(root / "fresh.json"),
                    "--output",
                    str(root / "audit.json"),
                ]
            )
            self.assertEqual(args.cache_dir, root / "cache")
        for forbidden in ("--source", "--archive", "--profile", "--main"):
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
