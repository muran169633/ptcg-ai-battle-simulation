from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import train_bc_orbit as bc  # noqa: E402


class BcInitializationTests(unittest.TestCase):
    @staticmethod
    def model(seed: int, *, model_dim: int = 16) -> bc.EntityOptionPolicy:
        torch.manual_seed(seed)
        return bc.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=model_dim,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )

    def test_loads_full_bc_state_and_reports_provenance(self) -> None:
        source = self.model(1)
        target = self.model(2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "source.pt"
            torch.save(
                {
                    "feature_version": bc.FEATURE_VERSION,
                    "model_state_dict": source.state_dict(),
                    "epoch": 3,
                    "valid_metrics": {"exact_action_set_accuracy": 0.8},
                },
                checkpoint_path,
            )

            provenance = bc.load_bc_initialization(target, checkpoint_path)

            self.assertEqual(provenance["source_epoch"], 3)
            self.assertEqual(len(provenance["sha256"]), 64)
            for name, tensor in target.state_dict().items():
                self.assertTrue(
                    torch.equal(tensor, source.state_dict()[name]),
                    name,
                )

    def test_rejects_architecture_mismatch(self) -> None:
        source = self.model(1, model_dim=32)
        target = self.model(2, model_dim=16)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "source.pt"
            torch.save(
                {
                    "feature_version": bc.FEATURE_VERSION,
                    "model_state_dict": source.state_dict(),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(
                ValueError,
                "not architecture-compatible",
            ):
                bc.load_bc_initialization(target, checkpoint_path)

    def test_rejects_ppo_checkpoint(self) -> None:
        target = self.model(2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "source.pt"
            torch.save(
                {
                    "feature_version": "ptcg-ppo-orbit-v1",
                    "model_state_dict": target.state_dict(),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "must be a BC checkpoint"):
                bc.load_bc_initialization(target, checkpoint_path)


class ParquetAllTrainTests(unittest.TestCase):
    def test_all_train_multiworker_yields_every_row_once(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            rows = [
                {
                    "episode_id": f"episode-{index}",
                    "split": "train" if index % 3 else "test",
                    "sample_weight": 0.5,
                }
                for index in range(103)
            ]
            pq.write_table(
                pa.Table.from_pylist(rows),
                dataset_dir / "part.parquet",
                row_group_size=7,
            )
            dataset = bc.ParquetDecisionDataset(
                dataset_dir=dataset_dir,
                split="train",
                max_rows=None,
                split_seed=17,
                shuffle_seed=23,
                epoch=1,
                use_trajectory_weights=False,
                deck_hashes=(),
                team_names=(),
                select_contexts=(),
                split_mode="all_train",
                shuffle_buffer_rows=0,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=4,
            )

            observed = [row["episode_id"] for row in loader]

            self.assertEqual(len(observed), len(rows))
            self.assertEqual(set(observed), {row["episode_id"] for row in rows})

class TrainableScopeTests(unittest.TestCase):
    @staticmethod
    def four_layer_model() -> bc.EntityOptionPolicy:
        return bc.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=64,
            model_dim=128,
            layers=4,
            heads=4,
            dropout=0.0,
            max_state_entities=8,
        )

    def test_all_is_default_and_preserves_full_model_training(self) -> None:
        field = bc.TrainConfig.__dataclass_fields__["trainable_scope"]
        self.assertEqual(field.default, "all")
        model = BcInitializationTests.model(7)
        next(model.parameters()).requires_grad_(False)

        parameters, audit = bc.configure_trainable_scope(model, "all")

        self.assertTrue(
            all(parameter.requires_grad for parameter in model.parameters())
        )
        self.assertEqual(
            [id(parameter) for parameter in parameters],
            [id(parameter) for parameter in model.parameters()],
        )
        self.assertEqual(audit["scope"], "all")
        self.assertEqual(
            audit["trainable_tensor_count"],
            audit["total_tensor_count"],
        )
        self.assertEqual(
            audit["trainable_parameter_count"],
            audit["total_parameter_count"],
        )
        self.assertEqual(
            audit["trainable_parameter_names"],
            [name for name, _ in model.named_parameters()],
        )

    def test_transformer_actor_is_exact_actor6_and_four_layer_count(
        self,
    ) -> None:
        model = self.four_layer_model()

        parameters, audit = bc.configure_trainable_scope(
            model,
            "transformer_actor",
            expected_trainable_tensors=56,
            expected_trainable_parameters=859_137,
        )

        expected_names = [
            name
            for name, _ in model.named_parameters()
            if name.startswith("transformer.")
            or name in bc.TRANSFORMER_ACTOR_HEAD_PARAMETER_NAMES
        ]
        observed_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(observed_names, expected_names)
        self.assertEqual(
            {
                name
                for name in observed_names
                if not name.startswith("transformer.")
            },
            set(bc.TRANSFORMER_ACTOR_HEAD_PARAMETER_NAMES),
        )
        self.assertEqual(audit["trainable_tensor_count"], 56)
        self.assertEqual(audit["trainable_parameter_count"], 859_137)
        self.assertEqual(audit["trainable_parameter_names"], expected_names)
        self.assertEqual(len(parameters), 56)
        self.assertTrue(
            all(parameter.requires_grad for parameter in parameters)
        )
        self.assertTrue(
            all(
                parameter.requires_grad == (name in expected_names)
                for name, parameter in model.named_parameters()
            )
        )

        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(
            [id(parameter) for parameter in optimizer_parameters],
            [id(parameter) for parameter in parameters],
        )

    def test_actor6_is_exact_six_tensor_optimizer_scope_and_metadata(
        self,
    ) -> None:
        model = self.four_layer_model()

        parameters, audit = bc.configure_trainable_scope(
            model,
            "actor6",
            expected_trainable_tensors=6,
            expected_trainable_parameters=65_793,
        )

        observed_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(
            observed_names,
            list(bc.TRANSFORMER_ACTOR_HEAD_PARAMETER_NAMES),
        )
        self.assertEqual(audit["scope"], "actor6")
        self.assertEqual(audit["trainable_tensor_count"], 6)
        self.assertEqual(audit["trainable_parameter_count"], 65_793)
        self.assertEqual(
            audit["trainable_parameter_names"],
            list(bc.TRANSFORMER_ACTOR_HEAD_PARAMETER_NAMES),
        )
        self.assertEqual(audit["expected_trainable_tensors"], 6)
        self.assertEqual(audit["expected_trainable_parameters"], 65_793)

        optimizer = torch.optim.AdamW(parameters, lr=1e-4)
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(
            [id(parameter) for parameter in optimizer_parameters],
            [id(parameter) for parameter in parameters],
        )

    def test_actor6_optimizer_step_leaves_frozen_complement_bitwise_equal(
        self,
    ) -> None:
        model = self.four_layer_model()
        parameters, _ = bc.configure_trainable_scope(model, "actor6")
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)

        loss = sum(parameter.square().sum() for parameter in parameters)
        loss.backward()
        optimizer.step()

        changed_trainable = []
        for name, parameter in model.named_parameters():
            changed = not torch.equal(before[name], parameter.detach())
            if name in bc.TRANSFORMER_ACTOR_HEAD_PARAMETER_NAMES:
                changed_trainable.append(changed)
            else:
                self.assertFalse(changed, name)
        self.assertTrue(all(changed_trainable))

    def test_actor6_expected_counts_fail_closed(self) -> None:
        for keyword, value, message in (
            ("expected_trainable_tensors", 5, "trainable_tensor_count"),
            (
                "expected_trainable_parameters",
                65_792,
                "trainable_parameter_count",
            ),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, message):
                    bc.configure_trainable_scope(
                        self.four_layer_model(),
                        "actor6",
                        **{keyword: value},
                    )

    def test_cli_help_lists_actor6_scope(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(REPO_ROOT / "tools" / "train_bc_orbit.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("{all,transformer_actor,actor6}", completed.stdout)
        self.assertIn("'actor6' trains only the six named", completed.stdout)
        self.assertIn("actor-head tensors", completed.stdout)

    def test_expected_counts_fail_closed(self) -> None:
        for keyword, value, message in (
            ("expected_trainable_tensors", 55, "trainable_tensor_count"),
            (
                "expected_trainable_parameters",
                859_136,
                "trainable_parameter_count",
            ),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, message):
                    bc.configure_trainable_scope(
                        self.four_layer_model(),
                        "transformer_actor",
                        **{keyword: value},
                    )

    def test_scoped_checkpoint_still_contains_full_state(self) -> None:
        model = BcInitializationTests.model(9)
        _, audit = bc.configure_trainable_scope(
            model,
            "actor6",
        )
        with (
            mock.patch.object(bc, "asdict", return_value={}),
            mock.patch.object(torch, "save") as save,
        ):
            bc.save_checkpoint(
                Path("unused.pt"),
                model,
                SimpleNamespace(),  # type: ignore[arg-type]
                1,
                {},
            )

        payload = save.call_args.args[0]
        self.assertEqual(
            set(payload["model_state_dict"]),
            set(model.state_dict()),
        )
        self.assertGreater(
            len(payload["model_state_dict"]),
            audit["trainable_tensor_count"],
        )


class TrajectoryWeightScopeTests(unittest.TestCase):
    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        return {
            "option_mask": torch.ones((2, 2), dtype=torch.bool),
            "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "sample_weights": torch.tensor([1.0, 0.25]),
            "action_counts": torch.tensor([1, 1]),
            "min_counts": torch.tensor([0, 0]),
            "max_counts": torch.tensor([2, 2]),
            "win_targets": torch.tensor([1.0, 0.0]),
        }

    @staticmethod
    def outputs() -> dict[str, torch.Tensor]:
        count_logits = torch.zeros((2, bc.MAX_ACTION_COUNT + 1))
        count_logits[0, 1] = 2.0
        count_logits[1, 1] = -2.0
        return {
            "policy_logits": torch.tensor([[2.0, -1.0], [0.5, -0.5]]),
            "count_logits": count_logits,
            "value_logits": torch.tensor([2.0, 2.0]),
        }

    @staticmethod
    def config(scope: str) -> SimpleNamespace:
        return SimpleNamespace(
            set_bce_weight=0.25,
            count_loss_weight=1.0,
            value_loss_weight=0.5,
            trajectory_weight_scope=scope,
        )

    def test_default_scope_preserves_historical_all_loss_weighting(self) -> None:
        field = bc.TrainConfig.__dataclass_fields__["trajectory_weight_scope"]
        self.assertEqual(field.default, "all_losses")

    def test_policy_only_changes_value_loss_but_not_policy_losses(self) -> None:
        batch = self.batch()
        outputs = self.outputs()

        all_total, all_parts = bc.masked_losses(
            outputs,
            batch,
            self.config("all_losses"),
        )
        policy_total, policy_parts = bc.masked_losses(
            outputs,
            batch,
            self.config("policy_only"),
        )

        weights = batch["sample_weights"]
        pointer_per_row = -(
            batch["targets"]
            * F.log_softmax(outputs["policy_logits"], dim=-1)
        ).sum(dim=1)
        expected_pointer = (pointer_per_row * weights).sum() / weights.sum()
        bce_per_row = F.binary_cross_entropy_with_logits(
            outputs["policy_logits"],
            batch["targets"],
            reduction="none",
        ).mean(dim=1)
        expected_set_bce = (bce_per_row * weights).sum() / weights.sum()
        count_per_row = F.cross_entropy(
            outputs["count_logits"],
            batch["action_counts"],
            reduction="none",
        )
        expected_count = (count_per_row * weights).sum() / weights.sum()
        torch.testing.assert_close(all_parts["pointer"], expected_pointer)
        torch.testing.assert_close(all_parts["set_bce"], expected_set_bce)
        torch.testing.assert_close(all_parts["count"], expected_count)
        for name in ("pointer", "set_bce", "count"):
            torch.testing.assert_close(policy_parts[name], all_parts[name])

        value_per_row = F.binary_cross_entropy_with_logits(
            outputs["value_logits"],
            batch["win_targets"],
            reduction="none",
        )
        expected_weighted = (
            value_per_row * batch["sample_weights"]
        ).sum() / batch["sample_weights"].sum()
        expected_unweighted = value_per_row.mean()
        torch.testing.assert_close(all_parts["value"], expected_weighted)
        torch.testing.assert_close(policy_parts["value"], expected_unweighted)
        torch.testing.assert_close(
            policy_total - all_total,
            0.5 * (expected_unweighted - expected_weighted),
        )

    def test_invalid_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported trajectory weight scope",
        ):
            bc.masked_losses(
                self.outputs(),
                self.batch(),
                self.config("invalid"),
            )


class PolicyTeamBalanceTests(unittest.TestCase):
    @staticmethod
    def loss_batch(
        team_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = {
            "option_mask": torch.ones((2, 2), dtype=torch.bool),
            "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "sample_weights": torch.ones(2),
            "action_counts": torch.tensor([1, 1]),
            "min_counts": torch.tensor([0, 0]),
            "max_counts": torch.tensor([2, 2]),
            "win_targets": torch.tensor([1.0, 0.0]),
        }
        if team_weights is not None:
            batch["policy_team_weights"] = team_weights
        return batch

    @staticmethod
    def loss_outputs(
        *,
        requires_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        count_logits = torch.zeros((2, bc.MAX_ACTION_COUNT + 1))
        count_logits[0, 1] = 1.5
        count_logits[1, 1] = -0.5
        outputs = {
            "policy_logits": torch.tensor(
                [[2.0, -1.0], [0.5, -0.5]]
            ),
            "count_logits": count_logits,
            "value_logits": torch.tensor([2.0, 2.0]),
        }
        if requires_grad:
            return {
                name: tensor.clone().requires_grad_()
                for name, tensor in outputs.items()
            }
        return outputs

    @staticmethod
    def loss_config(mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            set_bce_weight=0.25,
            count_loss_weight=1.0,
            value_loss_weight=0.5,
            trajectory_weight_scope="all_losses",
            flexible_selection_loss_weight=1.0,
            policy_team_balance=mode,
        )

    @staticmethod
    def feature_row(
        team_weight: float | None,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "global_fields": [],
            "global_numeric": [0.0] * bc.GLOBAL_NUMERIC_SIZE,
            "entity_fields": [],
            "entity_numeric": [],
            "option_fields": [[]],
            "option_numeric": [[0.0] * bc.OPTION_NUMERIC_SIZE],
            "targets": [1.0],
            "action_count": 1,
            "action_sequence": [0],
            "min_count": 1,
            "max_count": 1,
            "context": 0,
            "sample_weight": 1.0,
            "win_target": 1.0,
        }
        if team_weight is not None:
            row["policy_team_weight"] = team_weight
        return row

    def test_default_mode_is_none(self) -> None:
        field = bc.TrainConfig.__dataclass_fields__["policy_team_balance"]
        self.assertEqual(field.default, "none")

    def test_sqrt_clip2_weights_are_bounded_deterministic_and_normalized(
        self,
    ) -> None:
        counts = {"large": 900, "medium": 90, "rare": 10}
        first = bc.sqrt_clip2_team_weights(counts)
        self.assertEqual(first, bc.sqrt_clip2_team_weights(counts))
        self.assertLess(first["large"], first["medium"])
        self.assertLessEqual(first["medium"], first["rare"])
        self.assertEqual(first["rare"], 2.0)
        self.assertTrue(all(0.5 <= value <= 2.0 for value in first.values()))
        weighted_mean = sum(
            counts[team] * first[team] for team in counts
        ) / sum(counts.values())
        self.assertAlmostEqual(weighted_mean, 1.0, places=12)

        for invalid in ({}, {"a": 0}, {"a": -1}, {"a": 1.5}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    bc.sqrt_clip2_team_weights(invalid)  # type: ignore[arg-type]

    def test_sqrt_clip2_half_is_exact_midpoint_and_normalized(self) -> None:
        counts = {"large": 900, "medium": 90, "rare": 10}
        full = bc.sqrt_clip2_team_weights(counts)
        half = bc.sqrt_clip2_half_team_weights(counts)
        self.assertEqual(half, bc.sqrt_clip2_half_team_weights(counts))
        for team_id in counts:
            self.assertAlmostEqual(
                half[team_id],
                1.0 + 0.5 * (full[team_id] - 1.0),
                places=15,
            )
        self.assertTrue(
            all(0.75 <= value <= 1.5 for value in half.values())
        )
        weighted_mean = sum(
            counts[team] * half[team] for team in counts
        ) / sum(counts.values())
        self.assertAlmostEqual(weighted_mean, 1.0, places=12)

    def test_focus_team_weights_are_relative_and_normalized(self) -> None:
        counts = {"AlphaStarmie": 20, "M Sato": 60, "other": 20}
        weights = bc.focus_team_weights(counts, "AlphaStarmie", 1.25)
        self.assertAlmostEqual(
            weights["AlphaStarmie"] / weights["M Sato"],
            1.25,
            places=15,
        )
        self.assertEqual(weights["M Sato"], weights["other"])
        weighted_mean = sum(
            counts[team] * weights[team] for team in counts
        ) / sum(counts.values())
        self.assertAlmostEqual(weighted_mean, 1.0, places=12)
        for focus, relative in (
            ("missing", 1.25),
            ("AlphaStarmie", 0.0),
            ("AlphaStarmie", float("nan")),
        ):
            with self.subTest(focus=focus, relative=relative):
                with self.assertRaises(ValueError):
                    bc.focus_team_weights(counts, focus, relative)

    def test_train_team_scan_uses_train_demonstrator_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "rows.zip"
            train_rows = [
                {
                    "split": "train",
                    "team_name": "A",
                    "opponent_team_name": "ignored",
                },
                {
                    "split": "train",
                    "team_name": "B",
                    "opponent_team_name": "A",
                },
                {
                    "split": "train",
                    "team_name": "A",
                    "opponent_team_name": "B",
                },
            ]
            valid_rows = [
                {
                    "split": "valid",
                    "team_name": "valid-only",
                    "opponent_team_name": "A",
                }
            ]
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "train/part-00000.jsonl",
                    "".join(json.dumps(row) + "\n" for row in train_rows),
                )
                archive.writestr(
                    "valid/part-00000.jsonl",
                    "".join(json.dumps(row) + "\n" for row in valid_rows),
                )
            self.assertEqual(
                bc.count_training_team_rows(archive_path),
                {"A": 2, "B": 1},
            )

            missing_path = Path(directory) / "missing.zip"
            with zipfile.ZipFile(missing_path, "w") as archive:
                archive.writestr(
                    "train/part-00000.jsonl",
                    json.dumps({"split": "train"}) + "\n",
                )
            with self.assertRaisesRegex(ValueError, "team_name"):
                bc.count_training_team_rows(missing_path)

    def test_dataset_weights_train_only_and_unknown_team_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "rows.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "train/part-00000.jsonl",
                    json.dumps(
                        {
                            "split": "train",
                            "team_name": "A",
                            "row_id": 1,
                        }
                    )
                    + "\n",
                )
                archive.writestr(
                    "valid/part-00000.jsonl",
                    json.dumps(
                        {
                            "split": "valid",
                            "team_name": "B",
                            "row_id": 2,
                        }
                    )
                    + "\n",
                )

            def dataset(
                split: str,
                weights: dict[str, float] | None,
            ) -> bc.ZipDecisionDataset:
                return bc.ZipDecisionDataset(
                    archive_path=archive_path,
                    split=split,
                    max_rows=None,
                    split_seed=3,
                    shuffle_seed=5,
                    epoch=0,
                    hash_size=bc.DEFAULT_HASH_SIZE,
                    max_state_entities=bc.DEFAULT_MAX_STATE_ENTITIES,
                    use_trajectory_weights=False,
                    deck_hashes=(),
                    team_names=(),
                    split_mode="archive",
                    policy_team_weights=weights,
                )

            def fake_featurize(row, _hash_size, _max_state_entities):
                return {
                    "row_id": row["row_id"],
                    "sample_weight": 7.0,
                }

            with mock.patch.object(
                bc,
                "featurize_row",
                side_effect=fake_featurize,
            ):
                train_row = list(dataset("train", {"A": 1.75}))[0]
                valid_row = list(dataset("valid", {"A": 1.75}))[0]
                default_row = list(dataset("valid", None))[0]
            self.assertEqual(train_row["sample_weight"], 1.0)
            self.assertEqual(train_row["policy_team_weight"], 1.75)
            self.assertEqual(valid_row["policy_team_weight"], 1.0)
            self.assertNotIn("policy_team_weight", default_row)

            with (
                mock.patch.object(
                    bc,
                    "featurize_row",
                    side_effect=fake_featurize,
                ),
                self.assertRaisesRegex(ValueError, "unknown team_name"),
            ):
                list(dataset("train", {"other": 1.0}))

    def test_collate_preserves_policy_team_weight_order(self) -> None:
        batch = bc.collate_decisions(
            [self.feature_row(2.0), self.feature_row(0.5)],
            max_state_entities=2,
            entity_fields=2,
            option_fields=2,
        )
        self.assertTrue(
            torch.equal(
                batch["policy_team_weights"],
                torch.tensor([2.0, 0.5]),
            )
        )

    def test_collate_default_schema_omits_policy_team_weights(self) -> None:
        batch = bc.collate_decisions(
            [self.feature_row(None), self.feature_row(None)],
            max_state_entities=2,
            entity_fields=2,
            option_fields=2,
        )
        self.assertNotIn("policy_team_weights", batch)

    def test_collate_rejects_mixed_policy_team_weight_schema(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "either all or no batch rows",
        ):
            bc.collate_decisions(
                [self.feature_row(1.25), self.feature_row(None)],
                max_state_entities=2,
                entity_fields=2,
                option_fields=2,
            )

    def test_actor_losses_use_absolute_team_weights_and_value_is_uniform(
        self,
    ) -> None:
        team_weights = torch.tensor([2.0, 0.5])
        batch = self.loss_batch(team_weights)
        outputs = self.loss_outputs(requires_grad=True)
        _, parts = bc.masked_losses(
            outputs,
            batch,
            self.loss_config("sqrt_clip2"),
        )

        pointer_rows = -(
            batch["targets"]
            * F.log_softmax(outputs["policy_logits"], dim=-1)
        ).sum(dim=1)
        bce_rows = F.binary_cross_entropy_with_logits(
            outputs["policy_logits"],
            batch["targets"],
            reduction="none",
        ).mean(dim=1)
        count_rows = F.cross_entropy(
            outputs["count_logits"],
            batch["action_counts"],
            reduction="none",
        )
        torch.testing.assert_close(
            parts["pointer"],
            (pointer_rows * team_weights).sum() / 2,
        )
        torch.testing.assert_close(
            parts["set_bce"],
            (bce_rows * team_weights).sum() / 2,
        )
        torch.testing.assert_close(
            parts["count"],
            (count_rows * team_weights).sum() / 2,
        )
        expected_value = F.binary_cross_entropy_with_logits(
            outputs["value_logits"],
            batch["win_targets"],
        )
        torch.testing.assert_close(parts["value"], expected_value)

        parts["value"].backward()
        weighted_value_gradient = outputs["value_logits"].grad.detach().clone()
        baseline_outputs = self.loss_outputs(requires_grad=True)
        _, baseline_parts = bc.masked_losses(
            baseline_outputs,
            self.loss_batch(),
            self.loss_config("none"),
        )
        baseline_parts["value"].backward()
        self.assertTrue(
            torch.equal(
                weighted_value_gradient,
                baseline_outputs["value_logits"].grad,
            )
        )

    def test_homogeneous_team_batch_does_not_cancel_weight(self) -> None:
        baseline_total, baseline = bc.masked_losses(
            self.loss_outputs(),
            self.loss_batch(),
            self.loss_config("none"),
        )
        weighted_total, weighted = bc.masked_losses(
            self.loss_outputs(),
            self.loss_batch(torch.tensor([2.0, 2.0])),
            self.loss_config("sqrt_clip2"),
        )
        del baseline_total, weighted_total
        for name in ("pointer", "set_bce", "count"):
            torch.testing.assert_close(
                weighted[name],
                baseline[name] * 2.0,
            )
        self.assertTrue(torch.equal(weighted["value"], baseline["value"]))

    def test_treatment_validation_is_fail_closed(self) -> None:
        bc.validate_policy_team_balance_configuration(
            "none",
            split_mode="episode_hash",
            max_train_rows=10,
            use_trajectory_weights=True,
        )
        for kwargs, message in (
            (
                {
                    "split_mode": "episode_hash",
                    "max_train_rows": None,
                    "use_trajectory_weights": False,
                },
                "split-mode archive",
            ),
            (
                {
                    "split_mode": "archive",
                    "max_train_rows": 10,
                    "use_trajectory_weights": False,
                },
                "complete train split",
            ),
            (
                {
                    "split_mode": "archive",
                    "max_train_rows": None,
                    "use_trajectory_weights": True,
                },
                "use-trajectory-weights",
            ),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    bc.validate_policy_team_balance_configuration(
                        "sqrt_clip2",
                        **kwargs,
                    )


class FlexibleSelectionLossWeightTests(unittest.TestCase):
    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        return {
            "option_mask": torch.ones((2, 2), dtype=torch.bool),
            "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "sample_weights": torch.tensor([1.0, 0.25]),
            "action_counts": torch.tensor([1, 1]),
            "min_counts": torch.tensor([1, 0]),
            "max_counts": torch.tensor([1, 2]),
            "win_targets": torch.tensor([1.0, 0.0]),
        }

    @staticmethod
    def outputs(
        *,
        requires_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        count_logits = torch.zeros((2, bc.MAX_ACTION_COUNT + 1))
        count_logits[0, 1] = 2.0
        count_logits[1, 1] = -2.0
        tensors = {
            "policy_logits": torch.tensor(
                [[2.0, -1.0], [0.5, -0.5]]
            ),
            "count_logits": count_logits,
            "value_logits": torch.tensor([2.0, 2.0]),
        }
        if requires_grad:
            return {
                name: tensor.clone().requires_grad_()
                for name, tensor in tensors.items()
            }
        return tensors

    @staticmethod
    def config(
        weight: float,
        scope: str = "all_losses",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            set_bce_weight=0.25,
            count_loss_weight=1.0,
            value_loss_weight=0.5,
            trajectory_weight_scope=scope,
            flexible_selection_loss_weight=weight,
        )

    def test_default_weight_preserves_historical_behavior(self) -> None:
        field = bc.TrainConfig.__dataclass_fields__[
            "flexible_selection_loss_weight"
        ]
        self.assertEqual(field.default, 1.0)

        batch = self.batch()
        outputs = self.outputs()
        total, parts = bc.masked_losses(
            outputs,
            batch,
            self.config(1.0),
        )
        weights = batch["sample_weights"]
        pointer_per_row = -(
            batch["targets"]
            * F.log_softmax(outputs["policy_logits"], dim=-1)
        ).sum(dim=1)
        expected_pointer = (pointer_per_row * weights).sum() / weights.sum()
        bce_per_row = F.binary_cross_entropy_with_logits(
            outputs["policy_logits"],
            batch["targets"],
            reduction="none",
        ).mean(dim=1)
        expected_bce = (bce_per_row * weights).sum() / weights.sum()
        torch.testing.assert_close(parts["pointer"], expected_pointer)
        torch.testing.assert_close(parts["set_bce"], expected_bce)
        torch.testing.assert_close(
            total,
            parts["pointer"]
            + 0.25 * parts["set_bce"]
            + parts["count"]
            + 0.5 * parts["value"],
        )

    def test_weight_only_rebalances_pointer_and_set_bce(self) -> None:
        batch = self.batch()
        outputs = self.outputs()
        baseline_total, baseline_parts = bc.masked_losses(
            outputs,
            batch,
            self.config(1.0),
        )
        weighted_total, weighted_parts = bc.masked_losses(
            outputs,
            batch,
            self.config(2.5),
        )

        selection_weights = torch.tensor([1.0, 0.25 * 2.5])
        pointer_per_row = -(
            batch["targets"]
            * F.log_softmax(outputs["policy_logits"], dim=-1)
        ).sum(dim=1)
        expected_pointer = (
            pointer_per_row * selection_weights
        ).sum() / selection_weights.sum()
        bce_per_row = F.binary_cross_entropy_with_logits(
            outputs["policy_logits"],
            batch["targets"],
            reduction="none",
        ).mean(dim=1)
        expected_bce = (
            bce_per_row * selection_weights
        ).sum() / selection_weights.sum()

        torch.testing.assert_close(weighted_parts["pointer"], expected_pointer)
        torch.testing.assert_close(weighted_parts["set_bce"], expected_bce)
        torch.testing.assert_close(
            weighted_parts["count"],
            baseline_parts["count"],
        )
        torch.testing.assert_close(
            weighted_parts["value"],
            baseline_parts["value"],
        )
        torch.testing.assert_close(
            weighted_total - baseline_total,
            (weighted_parts["pointer"] - baseline_parts["pointer"])
            + 0.25
            * (weighted_parts["set_bce"] - baseline_parts["set_bce"]),
        )

    def test_trajectory_scope_composes_without_value_leakage(self) -> None:
        batch = self.batch()
        outputs = self.outputs()
        all_total, all_parts = bc.masked_losses(
            outputs,
            batch,
            self.config(2.5, "all_losses"),
        )
        policy_total, policy_parts = bc.masked_losses(
            outputs,
            batch,
            self.config(2.5, "policy_only"),
        )
        for name in ("pointer", "set_bce", "count"):
            torch.testing.assert_close(policy_parts[name], all_parts[name])
        self.assertFalse(
            torch.isclose(policy_parts["value"], all_parts["value"])
        )
        torch.testing.assert_close(
            policy_total - all_total,
            0.5 * (policy_parts["value"] - all_parts["value"]),
        )

    def test_all_fixed_and_all_flexible_batches_are_finite(self) -> None:
        for flexible in (False, True):
            batch = self.batch()
            if flexible:
                batch["min_counts"] = torch.tensor([0, 0])
                batch["max_counts"] = torch.tensor([2, 2])
            else:
                batch["min_counts"] = torch.tensor([1, 1])
                batch["max_counts"] = torch.tensor([1, 1])
            total, parts = bc.masked_losses(
                self.outputs(),
                batch,
                self.config(2.5),
            )
            self.assertTrue(bool(torch.isfinite(total)))
            self.assertTrue(
                all(bool(torch.isfinite(value)) for value in parts.values())
            )

    def test_rejects_nonpositive_or_nonfinite_weight(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and positive",
                ):
                    bc.validate_flexible_selection_loss_weight(value)

    def test_backward_gradients_are_finite(self) -> None:
        outputs = self.outputs(requires_grad=True)
        total, _ = bc.masked_losses(
            outputs,
            self.batch(),
            self.config(2.5),
        )
        total.backward()
        for name, tensor in outputs.items():
            self.assertIsNotNone(tensor.grad, name)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()), name)


class CountTrunkGradientScaleTests(unittest.TestCase):
    @staticmethod
    def model() -> bc.EntityOptionPolicy:
        torch.manual_seed(41)
        model = bc.EntityOptionPolicy(
            hash_size=128,
            categorical_dim=8,
            model_dim=16,
            layers=2,
            heads=4,
            dropout=0.0,
            max_state_entities=4,
            max_options=4,
        )
        model.eval()
        return model

    @staticmethod
    def batch() -> dict[str, torch.Tensor]:
        torch.manual_seed(43)
        batch_size = 3
        state_count = 2
        option_count = 4
        field_count = 3
        return {
            "global_fields": torch.randint(
                1,
                128,
                (batch_size, field_count),
            ),
            "global_field_mask": torch.ones(
                (batch_size, field_count),
                dtype=torch.bool,
            ),
            "global_numeric": torch.randn(
                batch_size,
                bc.GLOBAL_NUMERIC_SIZE,
            ),
            "state_fields": torch.randint(
                1,
                128,
                (batch_size, state_count, field_count),
            ),
            "state_field_mask": torch.ones(
                (batch_size, state_count, field_count),
                dtype=torch.bool,
            ),
            "state_numeric": torch.randn(
                batch_size,
                state_count,
                bc.ENTITY_NUMERIC_SIZE,
            ),
            "state_mask": torch.ones(
                (batch_size, state_count),
                dtype=torch.bool,
            ),
            "option_fields": torch.randint(
                1,
                128,
                (batch_size, option_count, field_count),
            ),
            "option_field_mask": torch.ones(
                (batch_size, option_count, field_count),
                dtype=torch.bool,
            ),
            "option_numeric": torch.randn(
                batch_size,
                option_count,
                bc.OPTION_NUMERIC_SIZE,
            ),
            "option_mask": torch.ones(
                (batch_size, option_count),
                dtype=torch.bool,
            ),
        }

    def test_default_scale_preserves_historical_behavior(self) -> None:
        field = bc.TrainConfig.__dataclass_fields__[
            "count_trunk_gradient_scale"
        ]
        self.assertEqual(field.default, 1.0)

        model = self.model()
        batch = self.batch()
        default_outputs = model(batch)
        explicit_outputs = model(
            batch,
            count_trunk_gradient_scale=1.0,
        )
        for name in ("policy_logits", "count_logits", "value_logits"):
            self.assertTrue(
                torch.equal(default_outputs[name], explicit_outputs[name]),
                name,
            )

    def test_forward_values_are_identical_at_all_scales(self) -> None:
        model = self.model()
        batch = self.batch()
        baseline = model(batch, count_trunk_gradient_scale=1.0)
        for scale in (0.0, 0.5):
            outputs = model(
                batch,
                count_trunk_gradient_scale=scale,
            )
            for name in ("policy_logits", "count_logits", "value_logits"):
                self.assertTrue(
                    torch.equal(outputs[name], baseline[name]),
                    f"{scale}/{name}",
                )

    def test_scale_only_changes_count_gradient_into_shared_trunk(self) -> None:
        model = self.model()
        batch = self.batch()
        targets = torch.tensor([1, 2, 3])

        def gradients(
            scale: float,
        ) -> tuple[torch.Tensor | None, torch.Tensor]:
            model.zero_grad(set_to_none=True)
            outputs = model(
                batch,
                count_trunk_gradient_scale=scale,
            )
            F.cross_entropy(outputs["count_logits"], targets).backward()
            shared = model.global_encoder[0].weight.grad
            head = model.count_head[0].weight.grad
            self.assertIsNotNone(head)
            return (
                None if shared is None else shared.detach().clone(),
                head.detach().clone(),
            )

        shared_one, head_one = gradients(1.0)
        shared_half, head_half = gradients(0.5)
        shared_zero, head_zero = gradients(0.0)

        self.assertIsNotNone(shared_one)
        self.assertIsNotNone(shared_half)
        self.assertGreater(float(shared_one.norm()), 0.0)
        torch.testing.assert_close(shared_half, shared_one * 0.5)
        self.assertIsNone(shared_zero)
        self.assertTrue(torch.equal(head_half, head_one))
        self.assertTrue(torch.equal(head_zero, head_one))

    def test_policy_and_value_gradients_are_unchanged(self) -> None:
        model = self.model()
        batch = self.batch()

        def gradients(
            output_name: str,
            scale: float,
        ) -> dict[str, torch.Tensor]:
            model.zero_grad(set_to_none=True)
            outputs = model(
                batch,
                count_trunk_gradient_scale=scale,
            )
            outputs[output_name].square().mean().backward()
            return {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }

        for output_name in ("policy_logits", "value_logits"):
            baseline = gradients(output_name, 1.0)
            detached = gradients(output_name, 0.0)
            self.assertEqual(detached.keys(), baseline.keys(), output_name)
            for name in baseline:
                self.assertTrue(
                    torch.equal(detached[name], baseline[name]),
                    f"{output_name}/{name}",
                )

    def test_scale_does_not_change_state_dict_schema(self) -> None:
        model = self.model()
        before = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        model(
            self.batch(),
            count_trunk_gradient_scale=0.0,
        )
        after = model.state_dict()
        self.assertEqual(after.keys(), before.keys())
        for name, tensor in after.items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_rejects_out_of_range_or_nonfinite_scale(self) -> None:
        for value in (
            -0.01,
            1.01,
            float("nan"),
            float("inf"),
            -float("inf"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and between 0 and 1",
                ):
                    bc.validate_count_trunk_gradient_scale(value)
        for value in (0.0, 0.5, 1.0):
            with self.subTest(valid=value):
                bc.validate_count_trunk_gradient_scale(value)


class StreamingRowShuffleTests(unittest.TestCase):
    def test_zero_buffer_preserves_order(self) -> None:
        rows = list(range(32))
        shuffled = list(
            bc.bounded_stream_shuffle(
                iter(rows),
                buffer_rows=0,
                rng=random.Random(17),
            )
        )
        self.assertEqual(shuffled, rows)

    def test_shuffle_is_reproducible_complete_and_epoch_scoped(self) -> None:
        rows = list(range(100))

        def shuffled(epoch: int) -> list[int]:
            rng = random.Random(bc.row_shuffle_seed(23, epoch, 2))
            return list(
                bc.bounded_stream_shuffle(
                    iter(rows),
                    buffer_rows=11,
                    rng=rng,
                )
            )

        first = shuffled(4)
        self.assertEqual(first, shuffled(4))
        self.assertNotEqual(first, shuffled(5))
        worker_rng = random.Random(bc.row_shuffle_seed(23, 4, 3))
        worker_output = list(
            bc.bounded_stream_shuffle(
                iter(rows),
                buffer_rows=11,
                rng=worker_rng,
            )
        )
        self.assertNotEqual(first, worker_output)
        self.assertNotEqual(first, rows)
        self.assertEqual(len(first), len(rows))
        self.assertEqual(sorted(first), rows)

    def test_buffered_stream_never_holds_more_than_configured_rows(self) -> None:
        state = {"consumed": 0, "emitted": 0, "max_lag": 0}

        def source():
            for value in range(50):
                state["consumed"] += 1
                yield value

        output = []
        for value in bc.bounded_stream_shuffle(
            source(),
            buffer_rows=7,
            rng=random.Random(5),
        ):
            output.append(value)
            state["emitted"] += 1
            state["max_lag"] = max(
                state["max_lag"],
                state["consumed"] - state["emitted"],
            )

        self.assertEqual(state["max_lag"], 7)
        self.assertEqual(sorted(output), list(range(50)))

    def test_worker_row_limits_never_exceed_global_cap(self) -> None:
        limits = [bc.worker_row_limit(10, worker, 3) for worker in range(3)]
        self.assertEqual(limits, [4, 3, 3])
        self.assertEqual(sum(limit for limit in limits if limit is not None), 10)
        self.assertEqual(
            [bc.worker_row_limit(2, worker, 4) for worker in range(4)],
            [1, 1, 0, 0],
        )
        self.assertIsNone(bc.worker_row_limit(None, 0, 8))
        self.assertEqual(bc.effective_shuffle_buffer_rows("train", 4096), 4096)
        self.assertEqual(bc.effective_shuffle_buffer_rows("valid", 4096), 0)
        self.assertEqual(bc.effective_shuffle_buffer_rows("test", 4096), 0)

    def test_dataset_caps_before_shuffle_and_never_shuffles_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "rows.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for split, count in (("train", 20), ("valid", 6)):
                    payload = "".join(
                        json.dumps(
                            {
                                "row_id": row_id,
                                "sample_weight": row_id + 1,
                                "split": split,
                            }
                        )
                        + "\n"
                        for row_id in range(count)
                    )
                    archive.writestr(f"{split}/part-00000.jsonl", payload)

            def dataset(
                split: str,
                *,
                max_rows: int | None,
                epoch: int,
                buffer_rows: int,
            ) -> bc.ZipDecisionDataset:
                return bc.ZipDecisionDataset(
                    archive_path=archive_path,
                    split=split,
                    max_rows=max_rows,
                    split_seed=3,
                    shuffle_seed=29,
                    epoch=epoch,
                    hash_size=bc.DEFAULT_HASH_SIZE,
                    max_state_entities=bc.DEFAULT_MAX_STATE_ENTITIES,
                    use_trajectory_weights=True,
                    deck_hashes=(),
                    team_names=(),
                    split_mode="archive",
                    shuffle_buffer_rows=buffer_rows,
                )

            accepted: list[int] = []

            def fake_featurize(row, _hash_size, _max_state_entities):
                accepted.append(row["row_id"])
                return {
                    "row_id": row["row_id"],
                    "sample_weight": float(row["sample_weight"]),
                }

            train = dataset(
                "train",
                max_rows=7,
                epoch=2,
                buffer_rows=4,
            )
            with mock.patch.object(
                bc,
                "featurize_row",
                side_effect=fake_featurize,
            ):
                first = [row["row_id"] for row in train]
            self.assertEqual(accepted, list(range(7)))
            self.assertEqual(sorted(first), list(range(7)))

            accepted.clear()
            with mock.patch.object(
                bc,
                "featurize_row",
                side_effect=fake_featurize,
            ):
                second = [row["row_id"] for row in train]
            self.assertEqual(second, first)
            self.assertEqual(accepted, list(range(7)))

            legacy = dataset(
                "train",
                max_rows=7,
                epoch=2,
                buffer_rows=0,
            )
            with mock.patch.object(
                bc,
                "featurize_row",
                side_effect=fake_featurize,
            ):
                legacy_ids = [row["row_id"] for row in legacy]
            self.assertEqual(legacy_ids, list(range(7)))

            valid = dataset(
                "valid",
                max_rows=None,
                epoch=0,
                buffer_rows=4,
            )
            self.assertEqual(valid.shuffle_buffer_rows, 0)
            with mock.patch.object(
                bc,
                "featurize_row",
                side_effect=fake_featurize,
            ):
                valid_ids = [row["row_id"] for row in valid]
            self.assertEqual(valid_ids, list(range(6)))

    def test_multiworker_quotas_emit_exact_global_max_without_duplicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "rows.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for member_index in range(3):
                    payload = "".join(
                        json.dumps(
                            {
                                "row_id": member_index * 100 + row_index,
                                "sample_weight": 1.0,
                                "split": "train",
                            }
                        )
                        + "\n"
                        for row_index in range(10)
                    )
                    archive.writestr(
                        f"train/part-{member_index:05d}.jsonl",
                        payload,
                    )

            def fake_featurize(row, _hash_size, _max_state_entities):
                return {
                    "row_id": row["row_id"],
                    "sample_weight": 1.0,
                }

            outputs: list[int] = []
            worker_lengths: list[int] = []
            for worker_id in range(3):
                dataset = bc.ZipDecisionDataset(
                    archive_path=archive_path,
                    split="train",
                    max_rows=10,
                    split_seed=3,
                    shuffle_seed=31,
                    epoch=2,
                    hash_size=bc.DEFAULT_HASH_SIZE,
                    max_state_entities=bc.DEFAULT_MAX_STATE_ENTITIES,
                    use_trajectory_weights=True,
                    deck_hashes=(),
                    team_names=(),
                    split_mode="archive",
                    shuffle_buffer_rows=4,
                )
                worker = SimpleNamespace(id=worker_id, num_workers=3)
                with (
                    mock.patch.object(
                        bc,
                        "get_worker_info",
                        return_value=worker,
                    ),
                    mock.patch.object(
                        bc,
                        "featurize_row",
                        side_effect=fake_featurize,
                    ),
                ):
                    worker_output = [
                        row["row_id"]
                        for row in dataset
                    ]
                worker_lengths.append(len(worker_output))
                outputs.extend(worker_output)

            self.assertEqual(worker_lengths, [4, 3, 3])
            self.assertEqual(len(outputs), 10)
            self.assertEqual(len(set(outputs)), 10)

    def test_default_train_shuffle_buffer_is_disabled(self) -> None:
        field = bc.TrainConfig.__dataclass_fields__[
            "train_shuffle_buffer_rows_per_worker"
        ]
        self.assertEqual(field.default, 0)


if __name__ == "__main__":
    unittest.main()
