from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import profile_yanz_ppo_dual_anchor_deployment_v1 as dual  # noqa: E402


class BindingAndAnchorTests(unittest.TestCase):
    def test_all_fixed_dependencies_are_hash_bound(self) -> None:
        bindings = dual.validate_static_bindings()
        self.assertEqual(
            bindings["ppo_parent_checkpoint"]["sha256"], dual.PPO_PARENT_SHA256
        )
        self.assertEqual(
            bindings["pure_bc_checkpoint"]["sha256"], dual.PURE_BC_SHA256
        )
        self.assertEqual(bindings["hybrid_main"]["sha256"], dual.HYBRID_MAIN_SHA256)
        self.assertEqual(bindings["policy_runtime"]["sha256"], dual.RUNTIME_SHA256)
        self.assertEqual(bindings["deck_csv"]["sha256"], dual.DECK_CSV_SHA256)
        self.assertEqual(
            bindings["frozen_profiler_dependency"]["sha256"],
            dual.FROZEN_PROFILER_SHA256,
        )
        self.assertEqual(bindings["runtime_semantics"]["device"], "cpu")
        self.assertEqual(bindings["runtime_semantics"]["dtype"], "torch.float32")
        self.assertEqual(bindings["runtime_semantics"]["batch_size"], 1)
        self.assertEqual(bindings["runtime_semantics"]["models_per_row"], 2)

    def test_bound_ppo_and_bc_are_real61_cpu_fp32_exact_states(self) -> None:
        runtime = dual.frozen.load_bound_runtime()
        bc_model, ppo_model, config, audit = dual.load_bound_models(runtime)
        self.assertEqual(bc_model.count_head[-1].out_features, 61)
        self.assertEqual(ppo_model.count_head[-1].out_features, 61)
        for model in (bc_model, ppo_model):
            self.assertTrue(all(value.device.type == "cpu" for value in model.parameters()))
            self.assertTrue(all(value.dtype == torch.float32 for value in model.parameters()))
            self.assertFalse(model.training)
            self.assertTrue(all(not value.requires_grad for value in model.parameters()))
        self.assertEqual(
            audit["pure_bc_teacher"]["expanded_model_state_sha256"],
            dual.PURE_BC_EXPANDED61_BITWISE_STATE_SHA256,
        )
        self.assertEqual(
            audit["ppo_parent"]["model_state_sha256"],
            dual.PPO_PARENT_BITWISE_STATE_SHA256,
        )
        self.assertEqual(config["entity_fields"], 20)
        self.assertEqual(config["option_fields"], 24)

    def test_one_real_row_records_both_anchors_and_matches_actual_main(self) -> None:
        runtime = dual.frozen.load_bound_runtime()
        bc_model, ppo_model, config, _ = dual.load_bound_models(runtime)
        chunk = next(dual.frozen.iter_train_chunks(dual.YANZ, "yanz", chunk_size=1))
        row, identity = chunk["items"][0]
        record = dual.evaluate_dual_row(
            row, identity, "yanz", runtime, bc_model, ppo_model, config
        )
        self.assertEqual(record["record_schema"], dual.RECORD_SCHEMA)
        self.assertIn(record["hybrid_category"], dual.CATEGORY_ORDER)
        for prefix in ("bc", "ppo"):
            for suffix in (
                "policy_action",
                "hybrid_action",
                "set_correct",
                "hybrid_correct",
                "count_correct",
                "policy_hybrid_margin",
                "count_margin",
                "deployment_margin",
            ):
                self.assertIn(f"{prefix}_{suffix}", record)
        self.assertEqual(
            dual.frozen.actual_hybrid_main_action(
                row["observation"], runtime, bc_model, config
            ),
            record["bc_hybrid_action"],
        )
        self.assertEqual(
            dual.frozen.actual_hybrid_main_action(
                row["observation"], runtime, ppo_model, config
            ),
            record["ppo_hybrid_action"],
        )


class CategoryTests(unittest.TestCase):
    def test_four_categories_are_exhaustive_and_mutually_exclusive(self) -> None:
        expected = {
            (True, True): "ppo_correct_protection",
            (True, False): "bc_correct_ppo_wrong_recovery_target",
            (False, True): "ppo_only_repair_protection",
            (False, False): "both_wrong",
        }
        actual = {
            pair: dual.classify_transition(*pair)
            for pair in ((True, True), (True, False), (False, True), (False, False))
        }
        self.assertEqual(actual, expected)
        self.assertEqual(set(actual.values()), set(dual.CATEGORY_ORDER))

    def test_category_counts_reject_unknown_or_missing_assignment(self) -> None:
        rows = [{"hybrid_category": category} for category in dual.CATEGORY_ORDER]
        self.assertEqual(
            dual._category_counts(rows, "hybrid_category"),
            {category: 1 for category in dual.CATEGORY_ORDER},
        )
        with self.assertRaisesRegex(dual.ProtocolError, "non-exhaustive"):
            dual._category_counts(
                rows + [{"hybrid_category": "unknown"}], "hybrid_category"
            )


class TrainOnlyAndCacheTests(unittest.TestCase):
    def test_bound_reader_opens_train_members_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.zip"
            train_rows = [
                {
                    "split": "train",
                    "deck_hash": dual.DECK_HASH,
                    "episode_id": f"train-{index}",
                }
                for index in range(3)
            ]
            forbidden = {"split": "valid", "marker": "must-not-open"}
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "train/part-00000.jsonl",
                    "".join(json.dumps(row) + "\n" for row in train_rows),
                )
                archive.writestr("valid/part-00000.jsonl", json.dumps(forbidden) + "\n")
                archive.writestr("test/part-00000.jsonl", json.dumps(forbidden) + "\n")
            chunks = list(
                dual.frozen.iter_train_chunks(archive_path, "fixture", chunk_size=2)
            )
            self.assertEqual([len(chunk["items"]) for chunk in chunks], [2, 1])
            self.assertTrue(
                all(chunk["archive_member"].startswith("train/") for chunk in chunks)
            )
            self.assertEqual(
                [row["episode_id"] for chunk in chunks for row, _ in chunk["items"]],
                ["train-0", "train-1", "train-2"],
            )

    def test_immutable_cache_rejects_binding_and_record_drift(self) -> None:
        bindings = {"cache_contract_sha256": "a" * 64}
        records = [
            {
                "record_schema": dual.RECORD_SCHEMA,
                "hybrid_category": "ppo_correct_protection",
                "line_sha256": "1" * 64,
            }
        ]
        payload = dual.make_cache_payload(
            source="yanz",
            archive_sha256="b" * 64,
            archive_member="train/part-00000.jsonl",
            first_line=1,
            last_line=1,
            raw_chunk_sha256="c" * 64,
            records=records,
            cache_bindings=bindings,
        )
        self.assertEqual(
            dual.validate_cache_payload(
                payload,
                source="yanz",
                archive_sha256="b" * 64,
                archive_member="train/part-00000.jsonl",
                first_line=1,
                last_line=1,
                raw_chunk_sha256="c" * 64,
                cache_bindings=bindings,
            ),
            records,
        )
        with self.assertRaisesRegex(dual.ProtocolError, "cache binding"):
            dual.validate_cache_payload(
                payload,
                source="yanz",
                archive_sha256="b" * 64,
                archive_member="train/part-00000.jsonl",
                first_line=1,
                last_line=1,
                raw_chunk_sha256="c" * 64,
                cache_bindings={"cache_contract_sha256": "d" * 64},
            )
        bad = dict(payload)
        bad["records"] = [dict(records[0], hybrid_category="unknown")]
        bad["records_sha256"] = hashlib.sha256(
            dual.canonical_json_bytes(bad["records"])
        ).hexdigest()
        with self.assertRaisesRegex(dual.ProtocolError, "hybrid category"):
            dual.validate_cache_payload(
                bad,
                source="yanz",
                archive_sha256="b" * 64,
                archive_member="train/part-00000.jsonl",
                first_line=1,
                last_line=1,
                raw_chunk_sha256="c" * 64,
                cache_bindings=bindings,
            )

    def test_json_writer_and_cli_refuse_mutable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            dual.write_new_json({"status": "first"}, path)
            with self.assertRaises(FileExistsError):
                dual.write_new_json({"status": "second"}, path)
            args = dual.parse_args(
                [
                    "build",
                    "--cache-dir",
                    str(Path(directory) / "cache"),
                    "--output",
                    str(Path(directory) / "profile.json"),
                ]
            )
            self.assertEqual(args.command, "build")
        with self.assertRaises(SystemExit):
            dual.parse_args(["build", "--checkpoint", "other.pt"])
        with self.assertRaises(SystemExit):
            dual.parse_args(["smoke", "--archive", "other.zip"])

    def test_script_sha_lock_fails_closed(self) -> None:
        current = dual.file_sha256(dual.SCRIPT)
        dual.assert_script_sha(current)
        with self.assertRaisesRegex(dual.ProtocolError, "changed during run"):
            dual.assert_script_sha("0" * 64)


if __name__ == "__main__":
    unittest.main()
