from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mean_source_anchored_bc_endpoints as mean  # noqa: E402
import train_bc_orbit as bc  # noqa: E402


class SourceAnchoredMeanStateTests(unittest.TestCase):
    @staticmethod
    def states() -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        selected = {
            "transformer.weight": torch.tensor([2.0, 4.0]),
            **{
                name: torch.tensor([float(index + 1)])
                for index, name in enumerate(sorted(mean.block_utils.ACTOR_KEYS))
            },
        }
        source = {
            **selected,
            "encoder.weight": torch.tensor([9.0, -3.0, 1.0]),
            "counter": torch.tensor([7], dtype=torch.int64),
        }
        endpoint_a = {name: value.clone() for name, value in source.items()}
        endpoint_b = {name: value.clone() for name, value in source.items()}
        for name in selected:
            endpoint_a[name].add_(2.0)
            endpoint_b[name].sub_(4.0)
        return source, endpoint_a, endpoint_b

    def call(
        self,
        source: dict[str, torch.Tensor],
        endpoint_a: dict[str, torch.Tensor],
        endpoint_b: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        selected = [
            name
            for name in source
            if mean.block_utils.selected_by_transformer_actor(name)
        ]
        complement = [name for name in source if name not in selected]
        return mean.source_anchored_mean_state(
            source,
            endpoint_a,
            endpoint_b,
            expected_selected_tensors=len(selected),
            expected_selected_parameters=sum(source[name].numel() for name in selected),
            expected_complement_tensors=len(complement),
            expected_complement_parameters=sum(
                source[name].numel() for name in complement
            ),
        )

    def test_only_selected_fp32_tensors_are_averaged(self) -> None:
        source, endpoint_a, endpoint_b = self.states()
        output, audit = self.call(source, endpoint_a, endpoint_b)
        selected = set(audit["selected_keys"])
        self.assertEqual(set(audit["actor_keys"]), mean.block_utils.ACTOR_KEYS)
        for name, value in output.items():
            if name in selected:
                expected = torch.lerp(endpoint_a[name], endpoint_b[name], 0.5)
            else:
                expected = source[name]
                self.assertNotEqual(value.data_ptr(), source[name].data_ptr())
            self.assertTrue(mean.tensors_bitwise_equal(value, expected), name)

    def test_rejects_any_complement_drift_from_source(self) -> None:
        source, endpoint_a, endpoint_b = self.states()
        endpoint_b["encoder.weight"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "complement tensor"):
            self.call(source, endpoint_a, endpoint_b)

    def test_rejects_nonfloating_drift_without_conversion(self) -> None:
        source, endpoint_a, endpoint_b = self.states()
        endpoint_a["counter"][0] += 1
        with self.assertRaisesRegex(ValueError, "Non-floating tensor"):
            self.call(source, endpoint_a, endpoint_b)

    def test_rejects_selected_non_fp32_or_nonfinite(self) -> None:
        source, endpoint_a, endpoint_b = self.states()
        for state in (source, endpoint_a, endpoint_b):
            state["transformer.weight"] = state["transformer.weight"].double()
        with self.assertRaisesRegex(ValueError, "must be FP32"):
            self.call(source, endpoint_a, endpoint_b)

        source, endpoint_a, endpoint_b = self.states()
        endpoint_b["actor_query.weight"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.call(source, endpoint_a, endpoint_b)


class SourceAnchoredMeanIntegrationTests(unittest.TestCase):
    PPO_INTERFACE = {
        "actor": "policy_logits over option_mask",
        "cardinality": "count_logits masked to minCount..maxCount",
        "critic": "sigmoid(value_logits) predicts terminal win probability",
    }
    DECK_HASH = "3f4515092dc59df397f365a9b79c7cf0c1cb73b9aa38bc47c1b18e9df4c2fdaf"

    @staticmethod
    def runtime_config() -> dict[str, int | float]:
        return {
            "hash_size": 65536,
            "categorical_dim": 64,
            "model_dim": 128,
            "layers": 4,
            "heads": 4,
            "dropout": 0.05,
            "max_state_entities": 80,
            "entity_fields": 20,
            "option_fields": 24,
        }

    @classmethod
    def production_model(cls) -> bc.EntityOptionPolicy:
        config = cls.runtime_config()
        return bc.EntityOptionPolicy(
            hash_size=int(config["hash_size"]),
            categorical_dim=int(config["categorical_dim"]),
            model_dim=int(config["model_dim"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            dropout=float(config["dropout"]),
            max_state_entities=int(config["max_state_entities"]),
        )

    @classmethod
    def endpoint_static_config(
        cls,
        *,
        source: Path,
        source_sha256: str,
        mixed_archive: Path,
    ) -> dict[str, object]:
        return {
            **cls.runtime_config(),
            "data": str(mixed_archive.resolve()),
            "epochs": 1,
            "batch_size": 4,
            "workers": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "set_bce_weight": 0.25,
            "count_loss_weight": 1.0,
            "value_loss_weight": 0.05,
            "target_accuracy": 0.80,
            "use_trajectory_weights": True,
            "trajectory_weight_scope": "policy_only",
            "train_shuffle_buffer_rows_per_worker": 70_000,
            "flexible_selection_loss_weight": 1.0,
            "count_trunk_gradient_scale": 1.0,
            "expected_train_rows": 10,
            "split_mode": "archive",
            "max_train_rows": None,
            "max_valid_rows": None,
            "max_test_rows": None,
            "policy_team_balance": "none",
            "policy_team_counts": None,
            "policy_team_weights": None,
            "init_checkpoint": str(source.resolve()),
            "init_checkpoint_sha256": source_sha256,
            "trainable_scope": "transformer_actor",
            "expected_trainable_tensors": mean.EXPECTED_SELECTED_TENSORS,
            "expected_trainable_parameters": mean.EXPECTED_SELECTED_PARAMETERS,
            "deck_hashes": (cls.DECK_HASH,),
            "team_names": (),
        }

    @classmethod
    def checkpoint(
        cls,
        state: dict[str, torch.Tensor],
        config: dict[str, object],
        epoch: int,
    ) -> dict[str, object]:
        return {
            "feature_version": bc.FEATURE_VERSION,
            "config": config,
            "model_state_dict": state,
            "epoch": epoch,
            "valid_metrics": {"must": "not survive"},
            "ppo_interface": copy.deepcopy(cls.PPO_INTERFACE),
            "optimizer_state_dict": {"must": "not survive"},
            "scheduler_state_dict": {"must": "not survive"},
        }

    @staticmethod
    def write_protocol(
        path: Path,
        *,
        source: Path,
        mixed_archive: Path,
        trainer: Path,
        materializer: Path,
        endpoint_config: dict[str, object],
    ) -> None:
        protocol_endpoint_config = {
            key: endpoint_config[key]
            for key in mean.REQUIRED_ENDPOINT_CONFIG_KEYS
        }
        protocol = {
            "schema_version": mean.PROTOCOL_SCHEMA_VERSION,
            "bindings": {
                "mean_candidate": {
                    "formula": mean.FORMULA,
                    "alpha": mean.ALPHA,
                    "seeds": list(mean.EXPECTED_SEEDS),
                    "selected_scope": mean.SELECTED_SCOPE,
                    "expected_selected_tensors": mean.EXPECTED_SELECTED_TENSORS,
                    "expected_selected_parameters": mean.EXPECTED_SELECTED_PARAMETERS,
                    "expected_complement_tensors": mean.EXPECTED_COMPLEMENT_TENSORS,
                    "expected_complement_parameters": mean.EXPECTED_COMPLEMENT_PARAMETERS,
                },
                "inputs": {
                    key: {
                        "path": str(value.resolve()),
                        "sha256": mean.checkpoint_utils.sha256_file(value),
                    }
                    for key, value in {
                        "source_checkpoint": source,
                        "mixed_archive": mixed_archive,
                        "trainer": trainer,
                        "materializer": materializer,
                        "composer_tool": Path(mean.__file__),
                        "block_selection": Path(mean.block_utils.__file__),
                        "checkpoint_validation": Path(
                            mean.checkpoint_utils.__file__
                        ),
                    }.items()
                },
            },
            "training": {
                "endpoint_config": protocol_endpoint_config,
                "schedule": {
                    "algorithm": mean.SCHEDULE_ALGORITHM,
                    "steps_per_epoch": 3,
                    "total_steps": 3,
                    "warmup_steps": 1,
                    "warmup_fraction": mean.WARMUP_FRACTION,
                    "minimum_lr_multiplier": mean.MINIMUM_LR_MULTIPLIER,
                },
            },
        }
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    def test_preregistered_production_shape_mean_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.pt"
            endpoint_a_path = root / "endpoint-a.pt"
            endpoint_b_path = root / "endpoint-b.pt"
            mixed_archive = root / "mixed.zip"
            trainer = root / "train_bc_orbit.py"
            materializer = root / "build_weighted_bc_mix.py"
            protocol = root / "protocol.json"
            output = root / "nested" / "mean.pt"
            mixed_archive.write_bytes(b"mixed-archive-test")
            trainer.write_bytes(b"trainer-test")
            materializer.write_bytes(b"materializer-test")

            torch.manual_seed(17)
            source_state = {
                name: value.clone()
                for name, value in self.production_model().state_dict().items()
            }
            endpoint_a_state = {
                name: value.clone() for name, value in source_state.items()
            }
            endpoint_b_state = {
                name: value.clone() for name, value in source_state.items()
            }
            selected = {
                name
                for name in source_state
                if mean.block_utils.selected_by_transformer_actor(name)
            }
            for name in selected:
                endpoint_a_state[name].add_(0.01)
                endpoint_b_state[name].sub_(0.03)

            source_config = {
                **self.runtime_config(),
                "deck_hashes": (self.DECK_HASH,),
            }
            torch.save(
                self.checkpoint(source_state, source_config, epoch=12),
                source_path,
            )
            source_sha = mean.checkpoint_utils.sha256_file(source_path)
            static = self.endpoint_static_config(
                source=source_path,
                source_sha256=source_sha,
                mixed_archive=mixed_archive,
            )
            config_a = {
                **static,
                "seed": mean.EXPECTED_SEEDS[0],
                "output_dir": str((root / "run-a").resolve()),
            }
            config_b = {
                **static,
                "seed": mean.EXPECTED_SEEDS[1],
                "output_dir": str((root / "run-b").resolve()),
            }
            torch.save(
                self.checkpoint(endpoint_a_state, config_a, epoch=1),
                endpoint_a_path,
            )
            torch.save(
                self.checkpoint(endpoint_b_state, config_b, epoch=1),
                endpoint_b_path,
            )
            self.write_protocol(
                protocol,
                source=source_path,
                mixed_archive=mixed_archive,
                trainer=trainer,
                materializer=materializer,
                endpoint_config=static,
            )

            result = mean.create_source_anchored_endpoint_mean(
                protocol_path=protocol,
                source_checkpoint_path=source_path,
                endpoint_a_path=endpoint_a_path,
                endpoint_a_seed=mean.EXPECTED_SEEDS[0],
                endpoint_b_path=endpoint_b_path,
                endpoint_b_seed=mean.EXPECTED_SEEDS[1],
                mixed_archive_path=mixed_archive,
                trainer_path=trainer,
                materializer_path=materializer,
                output_path=output,
            )
            saved = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(result["selected_tensor_count"], 56)
            self.assertEqual(result["selected_parameter_count"], 859_137)
            self.assertEqual(result["complement_tensor_count"], 24)
            self.assertEqual(result["complement_parameter_count"], 4_287_506)
            self.assertEqual(saved["epoch"], 1)
            self.assertIsNone(saved["valid_metrics"])
            self.assertNotIn("optimizer_state_dict", saved)
            self.assertNotIn("scheduler_state_dict", saved)
            metadata = saved["source_anchored_endpoint_mean"]
            self.assertTrue(metadata["deployment_only"])
            self.assertFalse(metadata["resume_training"])
            self.assertTrue(metadata["valid_metrics_null"])
            self.assertTrue(metadata["optimizer_state_omitted"])
            self.assertTrue(metadata["scheduler_state_omitted"])
            self.assertEqual(metadata["formula"], mean.FORMULA)
            self.assertEqual(metadata["alpha"], 0.5)
            self.assertEqual(set(metadata["selected_keys"]), selected)
            self.assertEqual(
                metadata["training_protocol"]["derived_schedule"],
                {"steps_per_epoch": 3, "total_steps": 3, "warmup_steps": 1},
            )
            self.assertEqual(
                metadata["output_model_state_sha256"],
                mean.model_state_sha256(saved["model_state_dict"]),
            )
            for name, actual in saved["model_state_dict"].items():
                expected = (
                    torch.lerp(endpoint_a_state[name], endpoint_b_state[name], 0.5)
                    if name in selected
                    else source_state[name]
                )
                self.assertTrue(mean.tensors_bitwise_equal(actual, expected), name)

            protocol_document = json.loads(protocol.read_text(encoding="utf-8"))
            for missing_tool in (
                "composer_tool",
                "block_selection",
                "checkpoint_validation",
            ):
                old_protocol = copy.deepcopy(protocol_document)
                del old_protocol["bindings"]["inputs"][missing_tool]
                protocol.write_text(json.dumps(old_protocol), encoding="utf-8")
                with self.subTest(missing_tool=missing_tool), self.assertRaisesRegex(
                    TypeError,
                    f"bindings.inputs.{missing_tool} must be an object",
                ):
                    mean.create_source_anchored_endpoint_mean(
                        protocol_path=protocol,
                        source_checkpoint_path=source_path,
                        endpoint_a_path=endpoint_a_path,
                        endpoint_a_seed=mean.EXPECTED_SEEDS[0],
                        endpoint_b_path=endpoint_b_path,
                        endpoint_b_seed=mean.EXPECTED_SEEDS[1],
                        mixed_archive_path=mixed_archive,
                        trainer_path=trainer,
                        materializer_path=materializer,
                        output_path=root / f"missing-{missing_tool}.pt",
                    )
                self.assertFalse((root / f"missing-{missing_tool}.pt").exists())

            bad_sha_protocol = copy.deepcopy(protocol_document)
            bad_sha_protocol["bindings"]["inputs"]["source_checkpoint"][
                "sha256"
            ] = "0" * 64
            protocol.write_text(json.dumps(bad_sha_protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_checkpoint.sha256 mismatch"):
                mean.create_source_anchored_endpoint_mean(
                    protocol_path=protocol,
                    source_checkpoint_path=source_path,
                    endpoint_a_path=endpoint_a_path,
                    endpoint_a_seed=mean.EXPECTED_SEEDS[0],
                    endpoint_b_path=endpoint_b_path,
                    endpoint_b_seed=mean.EXPECTED_SEEDS[1],
                    mixed_archive_path=mixed_archive,
                    trainer_path=trainer,
                    materializer_path=materializer,
                    output_path=root / "must-not-exist.pt",
                )
            self.assertFalse((root / "must-not-exist.pt").exists())

    def test_atomic_publish_race_preserves_competing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mean.pt"
            state = {"weight": torch.tensor([1.0])}
            checkpoint = {
                "model_state_dict": state,
                "valid_metrics": None,
            }
            real_link = mean.os.link

            def competing_link(source: object, destination: object) -> None:
                Path(destination).write_bytes(b"competitor")
                real_link(source, destination)

            with (
                mock.patch.object(mean.os, "link", side_effect=competing_link),
                self.assertRaises(FileExistsError),
            ):
                mean.atomic_publish_checkpoint(
                    checkpoint,
                    state,
                    mean.model_state_sha256(state),
                    output,
                )
            self.assertEqual(output.read_bytes(), b"competitor")

    def test_legacy_protocol_without_composer_binding_fails_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            endpoint_a = root / "endpoint-a.pt"
            endpoint_b = root / "endpoint-b.pt"
            mixed_archive = root / "mixed.zip"
            trainer = root / "trainer.py"
            materializer = root / "materializer.py"
            protocol = root / "legacy-protocol.json"
            for path in (
                source,
                endpoint_a,
                endpoint_b,
                mixed_archive,
                trainer,
                materializer,
            ):
                path.write_bytes(path.name.encode("utf-8"))
            source_sha = mean.checkpoint_utils.sha256_file(source)
            static = self.endpoint_static_config(
                source=source,
                source_sha256=source_sha,
                mixed_archive=mixed_archive,
            )
            self.write_protocol(
                protocol,
                source=source,
                mixed_archive=mixed_archive,
                trainer=trainer,
                materializer=materializer,
                endpoint_config=static,
            )
            legacy = json.loads(protocol.read_text(encoding="utf-8"))
            del legacy["bindings"]["inputs"]["composer_tool"]
            protocol.write_text(json.dumps(legacy), encoding="utf-8")

            with self.assertRaisesRegex(
                TypeError,
                "bindings.inputs.composer_tool must be an object",
            ):
                mean.create_source_anchored_endpoint_mean(
                    protocol_path=protocol,
                    source_checkpoint_path=source,
                    endpoint_a_path=endpoint_a,
                    endpoint_a_seed=mean.EXPECTED_SEEDS[0],
                    endpoint_b_path=endpoint_b,
                    endpoint_b_seed=mean.EXPECTED_SEEDS[1],
                    mixed_archive_path=mixed_archive,
                    trainer_path=trainer,
                    materializer_path=materializer,
                    output_path=root / "must-not-exist.pt",
                )
            self.assertFalse((root / "must-not-exist.pt").exists())

    def test_existing_output_is_rejected_before_input_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing.pt"
            output.write_bytes(b"preserve")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                mean.create_source_anchored_endpoint_mean(
                    protocol_path=root / "missing-protocol.json",
                    source_checkpoint_path=root / "missing-source.pt",
                    endpoint_a_path=root / "missing-a.pt",
                    endpoint_a_seed=mean.EXPECTED_SEEDS[0],
                    endpoint_b_path=root / "missing-b.pt",
                    endpoint_b_seed=mean.EXPECTED_SEEDS[1],
                    mixed_archive_path=root / "missing.zip",
                    trainer_path=root / "missing-trainer.py",
                    materializer_path=root / "missing-materializer.py",
                    output_path=output,
                )
            self.assertEqual(output.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
