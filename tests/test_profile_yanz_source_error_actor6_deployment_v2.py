from __future__ import annotations

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

import profile_yanz_source_error_actor6_deployment_v2 as v2  # noqa: E402


class BindingAndRuntimeTests(unittest.TestCase):
    def test_static_source_main_runtime_and_contract_bindings(self) -> None:
        bindings = v2.validate_static_bindings()
        self.assertEqual(bindings["source_checkpoint"]["sha256"], v2.SOURCE_SHA256)
        self.assertEqual(bindings["hybrid_main"]["sha256"], v2.HYBRID_MAIN_SHA256)
        self.assertEqual(bindings["policy_runtime"]["sha256"], v2.RUNTIME_SHA256)
        self.assertEqual(bindings["template_contract"]["sha256"], v2.CONTRACT_SHA256)
        self.assertEqual(bindings["runtime_semantics"]["device"], "cpu")
        self.assertEqual(bindings["runtime_semantics"]["dtype"], "torch.float32")
        self.assertEqual(bindings["runtime_semantics"]["batch_size"], 1)

    def test_real_runtime_model_is_expanded61_cpu_float32(self) -> None:
        runtime = v2.load_bound_runtime()
        _, model, config, audit = v2.instantiate_deployment_source(runtime)
        self.assertEqual(model.count_head[-1].out_features, 61)
        self.assertTrue(all(value.device.type == "cpu" for value in model.parameters()))
        self.assertTrue(all(value.dtype == torch.float32 for value in model.parameters()))
        self.assertEqual(audit["expanded_count_classes"], 61)
        self.assertEqual(audit["tensor_count"], 80)
        self.assertEqual(
            audit["expanded_model_state_sha256"],
            v2.SOURCE_EXPANDED61_BITWISE_STATE_SHA256,
        )
        self.assertEqual(config["entity_fields"], 20)
        self.assertEqual(config["option_fields"], 24)

    def test_actual_frozen_hybrid_main_matches_profiler_on_set_and_order_fixtures(self) -> None:
        runtime = v2.load_bound_runtime()
        _, model, config, _ = v2.instantiate_deployment_source(runtime)
        wanted = {0, 34}
        found: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
        for chunk in v2.iter_train_chunks(v2.YANZ, "yanz", chunk_size=64):
            for row, identity in chunk["items"]:
                raw_context = row["observation"]["select"].get("context", -1)
                context = -1 if raw_context is None else int(raw_context)
                if context in wanted and context not in found:
                    found[context] = (row, identity)
            if set(found) == wanted:
                break
        self.assertEqual(set(found), wanted)
        for context in sorted(found):
            row, identity = found[context]
            record = v2.evaluate_row(
                row, identity, "yanz", runtime, model, config
            )
            actual = v2.actual_hybrid_main_action(
                row["observation"], runtime, model, config
            )
            self.assertEqual(actual, record["source_hybrid_action"])


class DecoderTests(unittest.TestCase):
    @staticmethod
    def batch(
        *, minimum: int, maximum: int, context: int, options: int = 4
    ) -> dict[str, torch.Tensor]:
        return {
            "min_counts": torch.tensor([minimum]),
            "max_counts": torch.tensor([maximum]),
            "contexts": torch.tensor([context]),
            "option_mask": torch.tensor(
                [[index < options for index in range(6)]], dtype=torch.bool
            ),
        }

    def test_hybrid_decoder_sorts_set_context_and_preserves_context34(self) -> None:
        outputs = {
            "policy_logits": torch.tensor([[1.0, 4.0, 3.0, 2.0, -30.0, -30.0]]),
            "count_logits": torch.zeros(1, 61),
        }
        set_result = v2.decode_hybrid(
            outputs, self.batch(minimum=2, maximum=2, context=0)
        )
        ordered_result = v2.decode_hybrid(
            outputs, self.batch(minimum=2, maximum=2, context=34)
        )
        self.assertEqual(set_result["raw_action"], [1, 2])
        self.assertEqual(set_result["hybrid_action"], [1, 2])
        self.assertEqual(ordered_result["hybrid_action"], [1, 2])

        reverse_outputs = dict(outputs)
        reverse_outputs["policy_logits"] = torch.tensor(
            [[1.0, 3.0, 4.0, 2.0, -30.0, -30.0]]
        )
        set_reverse = v2.decode_hybrid(
            reverse_outputs, self.batch(minimum=2, maximum=2, context=0)
        )
        ordered_reverse = v2.decode_hybrid(
            reverse_outputs, self.batch(minimum=2, maximum=2, context=34)
        )
        self.assertEqual(set_reverse["raw_action"], [2, 1])
        self.assertEqual(set_reverse["hybrid_action"], [1, 2])
        self.assertEqual(ordered_reverse["hybrid_action"], [2, 1])

    def test_flexible_count_masks_illegal_classes_across_real61_head(self) -> None:
        count_logits = torch.full((1, 61), -20.0)
        count_logits[0, 60] = 100.0
        count_logits[0, 3] = 5.0
        count_logits[0, 2] = 4.0
        outputs = {
            "policy_logits": torch.tensor([[1.0, 4.0, 3.0, 2.0, -30.0, -30.0]]),
            "count_logits": count_logits,
        }
        result = v2.decode_hybrid(
            outputs, self.batch(minimum=1, maximum=3, context=0)
        )
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["hybrid_action"], [1, 2, 3])

    def test_decoder_rejects_raw17_or_multirow_outputs(self) -> None:
        outputs = {
            "policy_logits": torch.zeros(1, 6),
            "count_logits": torch.zeros(1, 17),
        }
        with self.assertRaisesRegex(v2.ProtocolError, "61"):
            v2.decode_hybrid(outputs, self.batch(minimum=1, maximum=1, context=0))
        outputs["count_logits"] = torch.zeros(2, 61)
        outputs["policy_logits"] = torch.zeros(2, 6)
        with self.assertRaisesRegex(v2.ProtocolError, "batch=1"):
            v2.decode_hybrid(outputs, self.batch(minimum=1, maximum=1, context=0))


class TrainOnlyArchiveTests(unittest.TestCase):
    def test_chunk_reader_opens_only_train_members_and_preserves_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.zip"
            train_rows = [
                {
                    "split": "train",
                    "deck_hash": v2.DECK_HASH,
                    "episode_id": f"e{index}",
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
            chunks = list(v2.iter_train_chunks(archive_path, "fixture", chunk_size=2))
            self.assertEqual([len(chunk["items"]) for chunk in chunks], [2, 1])
            self.assertEqual(
                [item[0]["episode_id"] for chunk in chunks for item in chunk["items"]],
                ["e0", "e1", "e2"],
            )
            self.assertTrue(
                all(chunk["archive_member"] == "train/part-00000.jsonl" for chunk in chunks)
            )
            self.assertEqual(chunks[0]["first_line"], 1)
            self.assertEqual(chunks[1]["first_line"], 3)

    def test_nontrain_row_inside_train_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "train/part-00000.jsonl",
                    json.dumps({"split": "valid", "deck_hash": v2.DECK_HASH}) + "\n",
                )
            with self.assertRaisesRegex(v2.ProtocolError, "non-train row"):
                list(v2.iter_train_chunks(archive_path, "fixture", chunk_size=2))


class SelectionAndCacheTests(unittest.TestCase):
    def test_dynamic_treatment_sizes_preserve_v1_partition_when_count_is301(self) -> None:
        self.assertEqual(v2.balanced_treatment_sizes(301), (101, 100, 100))
        self.assertEqual(v2.balanced_treatment_sizes(300), (100, 100, 100))
        self.assertEqual(v2.balanced_treatment_sizes(302), (101, 101, 100))

    def test_actor6_retention_boundary_uses_policy_margin_not_count_margin(self) -> None:
        row = {
            "policy_hybrid_margin": 0.25,
            "count_margin": 0.001,
            "deployment_margin": 0.001,
        }
        self.assertEqual(v2.positive_actor6_policy_margin(row), 0.25)
        bad = dict(row)
        bad["policy_hybrid_margin"] = 0.0
        self.assertIsNone(v2.positive_actor6_policy_margin(bad))

    def test_script_sha_lock_fails_closed(self) -> None:
        current = v2.file_sha256(v2.SCRIPT)
        v2.assert_script_sha(current)
        with self.assertRaisesRegex(v2.ProtocolError, "profiler script changed"):
            v2.assert_script_sha("0" * 64)

    def test_cache_payload_rejects_binding_or_raw_chunk_drift(self) -> None:
        bindings = {"cache_contract_sha256": "a" * 64}
        rows = [{"line_sha256": "1" * 64, "source": "yanz"}]
        payload = v2.make_cache_payload(
            source="yanz",
            archive_sha256="b" * 64,
            archive_member="train/part-00000.jsonl",
            first_line=1,
            last_line=1,
            raw_chunk_sha256="c" * 64,
            records=rows,
            cache_bindings=bindings,
        )
        self.assertEqual(
            v2.validate_cache_payload(
                payload,
                source="yanz",
                archive_sha256="b" * 64,
                archive_member="train/part-00000.jsonl",
                first_line=1,
                last_line=1,
                raw_chunk_sha256="c" * 64,
                cache_bindings=bindings,
            ),
            rows,
        )
        bad = dict(payload)
        bad["raw_chunk_sha256"] = "d" * 64
        with self.assertRaisesRegex(v2.ProtocolError, "cache raw chunk"):
            v2.validate_cache_payload(
                bad,
                source="yanz",
                archive_sha256="b" * 64,
                archive_member="train/part-00000.jsonl",
                first_line=1,
                last_line=1,
                raw_chunk_sha256="c" * 64,
                cache_bindings=bindings,
            )

    def test_cli_has_no_mutable_model_or_archive_inputs(self) -> None:
        smoke = v2.parse_args(["smoke", "--rows-per-source", "2"])
        self.assertEqual(smoke.command, "smoke")
        with tempfile.TemporaryDirectory() as directory:
            build = v2.parse_args(
                [
                    "build",
                    "--cache-dir",
                    str(Path(directory) / "cache"),
                    "--output",
                    str(Path(directory) / "profile.json"),
                ]
            )
            self.assertEqual(build.command, "build")
        with self.assertRaises(SystemExit):
            v2.parse_args(["smoke", "--archive", "other.zip"])
        with self.assertRaises(SystemExit):
            v2.parse_args(["build", "--source", "other.pt"])


if __name__ == "__main__":
    unittest.main()
