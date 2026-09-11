from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "run_ppo_gold_ab",
    TOOLS_ROOT / "run_ppo_gold_ab.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def write_deck(path: Path, offset: int) -> str:
    cards = [offset + index for index in range(60)]
    path.write_text(
        "\n".join(str(card) for card in cards) + "\n",
        encoding="utf-8",
    )
    canonical = ",".join(str(card) for card in sorted(cards))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class GoldPPOABRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.learner_deck = self.root / "marnie.csv"
        write_deck(self.learner_deck, 1)
        self.bc_checkpoint = self.root / "fresh-bc.pt"
        self.bc_checkpoint.write_bytes(b"fresh bc learner")
        self.resume_checkpoint = self.root / "update-0440.pt"
        self.resume_checkpoint.write_bytes(b"resume metadata only")
        self.kl_reference = self.root / "fresh-bc-anchor.pt"
        self.kl_reference.write_bytes(b"explicit kl reference")
        self.bc_replay_data = self.root / "bc-replay.zip"
        self.bc_replay_data.write_bytes(b"small replay fixture")

        raw_opponents: list[dict[str, object]] = []
        specifications = [
            ("marnie-a", "Marnie Grimmsnarl", True),
            ("marnie-b", "Marnie Grimmsnarl", True),
            ("alakazam-a", "Alakazam Dudunsparce", True),
            ("mewtwo-a", "Team Rocket Mewtwo", True),
            ("garchomp-rejected", "Garchomp", False),
        ]
        for index, (policy_id, archetype, quality_pass) in enumerate(
            specifications
        ):
            checkpoint = self.root / f"clone-{index}.pt"
            checkpoint.write_bytes(f"checkpoint-{index}".encode("ascii"))
            deck = self.root / f"deck-{index}.csv"
            deck_hash = write_deck(deck, 1000 * (index + 1))
            name = runner.expected_extra_opponent_name(checkpoint, deck)
            raw_opponents.append(
                {
                    "policy_id": policy_id,
                    "submission_id": 55000000 + index,
                    "team_name": f"team-{index}",
                    "archetype": archetype,
                    "name": name,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": runner.file_sha256(checkpoint),
                    "deck": str(deck),
                    "deck_hash": deck_hash,
                    "quality": {
                        "pass": quality_pass,
                        "rows": 1000,
                    },
                }
            )
        self.manifest = self.root / "league_manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": runner.LEAGUE_SCHEMA,
                    "opponents": raw_opponents,
                    "excluded": [],
                    "safety": {
                        "uses_public_replay_actions_only": True,
                        "uses_open_submission_code": False,
                        "uploads_or_submissions_performed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        (
            self.gold_opponents,
            self.quality_excluded,
            _,
        ) = runner.load_quality_passing_league(self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def build_plan(
        self,
        *,
        profile_name: str = "pilot",
        control_mode: str = "bc_v3",
        actor_learning_rate: float = runner.DEFAULT_ACTOR_LEARNING_RATE,
        gae_lambda: float = runner.DEFAULT_GAE_LAMBDA,
        actor_reduction: str = runner.DEFAULT_ACTOR_REDUCTION,
        advantage_normalization: str = (
            runner.DEFAULT_ADVANTAGE_NORMALIZATION
        ),
        value_learning_rate: float = runner.DEFAULT_VALUE_LEARNING_RATE,
        value_coefficient: float = runner.DEFAULT_VALUE_COEFFICIENT,
        entropy_coefficient: float = runner.DEFAULT_ENTROPY_COEFFICIENT,
        policy_temperature: float = runner.DEFAULT_POLICY_TEMPERATURE,
        value_trunk_gradient_scale: float = (
            runner.DEFAULT_VALUE_TRUNK_GRADIENT_SCALE
        ),
        trainable_scope: str = runner.DEFAULT_TRAINABLE_SCOPE,
        actor_value_gradient_mode: str = (
            runner.DEFAULT_ACTOR_VALUE_GRADIENT_MODE
        ),
        bc_kl_coefficient: float = runner.DEFAULT_BC_KL_COEFFICIENT,
        bc_replay_data: Path | None = None,
        bc_replay_split: str = "train",
        bc_replay_batches: int = runner.DEFAULT_BC_REPLAY_BATCHES,
        bc_replay_batch_size: int = runner.DEFAULT_BC_REPLAY_BATCH_SIZE,
        bc_replay_workers: int = runner.DEFAULT_BC_REPLAY_WORKERS,
        bc_replay_steps: int = runner.DEFAULT_BC_REPLAY_STEPS,
        bc_replay_lr_scale: float = runner.DEFAULT_BC_REPLAY_LR_SCALE,
        bc_replay_loss: str = runner.DEFAULT_BC_REPLAY_LOSS,
        bc_replay_order_context_weight: float = (
            runner.DEFAULT_BC_REPLAY_ORDER_CONTEXT_WEIGHT
        ),
        bc_replay_non_context34_fixed_multi_action_order_weight: float = (
            runner
            .DEFAULT_BC_REPLAY_NON_CONTEXT34_FIXED_MULTI_ACTION_ORDER_WEIGHT
        ),
        bc_replay_context34_rows_per_batch: int = (
            runner.DEFAULT_BC_REPLAY_CONTEXT34_ROWS_PER_BATCH
        ),
        resume_learner_weights: str = (
            runner.DEFAULT_RESUME_LEARNER_WEIGHTS
        ),
        minibatch_size: int | None = None,
    ) -> dict[str, object]:
        return runner.build_plan(
            league_manifest=self.manifest,
            output_root=self.root / f"output-{profile_name}-{control_mode}",
            profile=runner.PROFILES[profile_name],
            seeds=runner.PROFILES[profile_name].seeds,
            control_mode=control_mode,
            bc_checkpoint=self.bc_checkpoint,
            kl_reference_checkpoint=self.kl_reference,
            learner_deck=self.learner_deck,
            resume_checkpoint=self.resume_checkpoint,
            resume_update=440,
            resume_feature_version="ptcg-selfplay-ppo-terminal01-v1",
            gold_opponents=self.gold_opponents,
            quality_excluded=self.quality_excluded,
            device="cpu",
            resume_learner_weights=resume_learner_weights,
            minibatch_size=minibatch_size,
            trainable_scope=trainable_scope,
            actor_learning_rate=actor_learning_rate,
            gae_lambda=gae_lambda,
            actor_reduction=actor_reduction,
            advantage_normalization=advantage_normalization,
            value_learning_rate=value_learning_rate,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            policy_temperature=policy_temperature,
            value_trunk_gradient_scale=value_trunk_gradient_scale,
            actor_value_gradient_mode=actor_value_gradient_mode,
            bc_kl_coefficient=bc_kl_coefficient,
            bc_replay_data=bc_replay_data,
            bc_replay_split=bc_replay_split,
            bc_replay_batches=bc_replay_batches,
            bc_replay_batch_size=bc_replay_batch_size,
            bc_replay_workers=bc_replay_workers,
            bc_replay_steps=bc_replay_steps,
            bc_replay_lr_scale=bc_replay_lr_scale,
            bc_replay_loss=bc_replay_loss,
            bc_replay_order_context_weight=(
                bc_replay_order_context_weight
            ),
            bc_replay_non_context34_fixed_multi_action_order_weight=(
                bc_replay_non_context34_fixed_multi_action_order_weight
            ),
            bc_replay_context34_rows_per_batch=(
                bc_replay_context34_rows_per_batch
            ),
        )

    def test_quality_gate_excludes_failed_clone(self) -> None:
        self.assertEqual(len(self.gold_opponents), 4)
        self.assertNotIn(
            "garchomp-rejected",
            {opponent.policy_id for opponent in self.gold_opponents},
        )
        self.assertEqual(
            self.quality_excluded,
            [
                {
                    "policy_id": "garchomp-rejected",
                    "archetype": "Garchomp",
                    "reason": "quality_pass_is_not_true",
                }
            ],
        )

    def test_explicit_gold_policy_allowlist_is_exact_and_audited(self) -> None:
        selected, excluded = runner.select_gold_policy_allowlist(
            self.gold_opponents,
            ["marnie-b", "mewtwo-a"],
        )
        self.assertEqual(
            [opponent.policy_id for opponent in selected],
            ["marnie-b", "mewtwo-a"],
        )
        self.assertEqual(
            {entry["policy_id"] for entry in excluded},
            {"marnie-a", "alakazam-a"},
        )
        self.assertTrue(
            all(
                entry["reason"] == "protocol_policy_allowlist_excluded"
                for entry in excluded
            )
        )

        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-allowlist"),
                "--gold-policy-id",
                "marnie-b",
                "--gold-policy-id",
                "mewtwo-a",
            ]
        )
        self.assertEqual(args.gold_policy_id, ["marnie-b", "mewtwo-a"])

    def test_bc_anchor_games_parser_accepts_even_override(self) -> None:
        default_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-bc-anchor-default"),
            ]
        )
        self.assertIsNone(default_args.bc_anchor_games)

        override_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-bc-anchor-override"),
                "--bc-anchor-games",
                "4",
            ]
        )
        self.assertEqual(override_args.bc_anchor_games, 4)

        with mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        "--league-manifest",
                        str(self.manifest),
                        "--output-root",
                        str(self.root / "output-bc-anchor-odd"),
                        "--bc-anchor-games",
                        "3",
                    ]
                )

    def test_ppo_epochs_parser_accepts_positive_override(self) -> None:
        default_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-ppo-epochs-default"),
            ]
        )
        self.assertIsNone(default_args.ppo_epochs)

        override_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-ppo-epochs-override"),
                "--ppo-epochs",
                "3",
            ]
        )
        self.assertEqual(override_args.ppo_epochs, 3)

        with mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        "--league-manifest",
                        str(self.manifest),
                        "--output-root",
                        str(self.root / "output-ppo-epochs-zero"),
                        "--ppo-epochs",
                        "0",
                    ]
                )

    def test_resume_weights_and_minibatch_parser_are_explicit(self) -> None:
        default_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-resume-default"),
            ]
        )
        self.assertEqual(default_args.resume_learner_weights, "bc")
        self.assertIsNone(default_args.minibatch_size)

        override_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-resume-override"),
                "--resume-learner-weights",
                "resume",
                "--minibatch-size",
                "384",
            ]
        )
        self.assertEqual(override_args.resume_learner_weights, "resume")
        self.assertEqual(override_args.minibatch_size, 384)

        invalid_argv = (
            ("--resume-learner-weights", "unknown"),
            ("--minibatch-size", "0"),
            ("--minibatch-size", "-1"),
        )
        for flag, value in invalid_argv:
            with self.subTest(flag=flag, value=value):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid"),
                                flag,
                                value,
                            ]
                        )

    def test_trainable_scope_parser_is_safely_restricted(self) -> None:
        default_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-scope-default"),
            ]
        )
        self.assertEqual(
            default_args.trainable_scope,
            "last_block_heads",
        )
        candidate_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-scope-candidate"),
                "--trainable-scope",
                "last_two_blocks_heads",
            ]
        )
        self.assertEqual(
            candidate_args.trainable_scope,
            "last_two_blocks_heads",
        )
        for rejected in ("heads", "full", "unknown"):
            with self.subTest(rejected=rejected):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / f"output-{rejected}"),
                                "--trainable-scope",
                                rejected,
                            ]
                        )

    def test_gold_policy_allowlist_rejects_unknown_duplicate_and_empty(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            runner.select_gold_policy_allowlist(
                self.gold_opponents,
                ["not-present"],
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.select_gold_policy_allowlist(
                self.gold_opponents,
                ["marnie-a", "marnie-a"],
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            runner.select_gold_policy_allowlist(
                self.gold_opponents,
                [""],
            )

    def test_merged_historical_exclusions_are_unwrapped_and_recovered_removed(
        self,
    ) -> None:
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["excluded"] = [
            {
                "entry": {
                    "policy_id": "marnie-a",
                    "archetype": "Marnie Grimmsnarl",
                    "status": "quality_rejected",
                    "quality": {"pass": False},
                }
            },
            {
                "entry": {
                    "policy_id": "still-rejected",
                    "archetype": "Garchomp",
                    "ppo_ineligible_reason": "quality_threshold",
                    "quality": {"pass": False},
                }
            },
            {
                "entry": {
                    "policy_id": "still-rejected",
                    "archetype": "Garchomp",
                    "ppo_ineligible_reason": "quality_threshold",
                    "quality": {"pass": False, "set_exact_accuracy": 0.59},
                }
            },
        ]
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")

        included, excluded, _ = runner.load_quality_passing_league(self.manifest)

        self.assertIn("marnie-a", {item.policy_id for item in included})
        self.assertEqual(
            excluded,
            [
                {
                    "policy_id": "still-rejected",
                    "archetype": "Garchomp",
                    "reason": "quality_threshold",
                    "quality": {
                        "pass": False,
                        "set_exact_accuracy": 0.59,
                    },
                },
                {
                    "policy_id": "garchomp-rejected",
                    "archetype": "Garchomp",
                    "reason": "quality_pass_is_not_true",
                },
            ],
        )

    def test_branches_have_exactly_equal_budget_and_seed_protocol(self) -> None:
        plan = self.build_plan(profile_name="formal")
        branches = plan["branches"]
        control = branches["A_marnie_control"]
        gold = branches["B_gold_league"]
        self.assertEqual(control["budget"], gold["budget"])
        self.assertEqual(
            control["budget"]["rollout_games_per_branch"],
            20 * 128 * 3,
        )
        self.assertEqual(
            [run["seed"] for run in control["runs"]],
            [run["seed"] for run in gold["runs"]],
        )
        self.assertEqual(
            [run["seed"] for run in control["runs"]],
            [20260727, 20260728, 20260729],
        )
        for branch in (control, gold):
            for run in branch["runs"]:
                command = run["command"]
                self.assertEqual(command_value(command, "--updates"), "460")
                self.assertEqual(
                    command_value(command, "--games-per-update"),
                    "128",
                )
                self.assertEqual(
                    command_value(command, "--ppo-epochs"),
                    "4",
                )

    def test_duration8_changes_only_duration_and_terminal_eval_size(self) -> None:
        pilot = runner.PROFILES["pilot"]
        duration = runner.PROFILES["duration8"]
        self.assertEqual(duration.additional_updates, 8)
        self.assertEqual(duration.eval_games, 64)
        for field in (
            "games_per_update",
            "ppo_epochs",
            "environments",
            "minibatch_size",
            "bc_anchor_games",
            "seeds",
        ):
            self.assertEqual(getattr(duration, field), getattr(pilot, field))

    def test_duration12_changes_only_named_duration(self) -> None:
        duration8 = runner.PROFILES["duration8"]
        duration12 = runner.PROFILES["duration12"]
        self.assertEqual(duration12.name, "duration12")
        self.assertEqual(duration12.additional_updates, 12)
        for field in (
            "games_per_update",
            "ppo_epochs",
            "environments",
            "minibatch_size",
            "eval_games",
            "bc_anchor_games",
            "seeds",
        ):
            self.assertEqual(
                getattr(duration12, field),
                getattr(duration8, field),
            )

        plan = self.build_plan(profile_name="duration12")
        for branch in plan["branches"].values():
            self.assertEqual(
                branch["budget"]["rollout_games_per_branch"],
                12 * 64,
            )
            for run in branch["runs"]:
                command = run["command"]
                self.assertEqual(command_value(command, "--updates"), "452")
                self.assertEqual(
                    command_value(command, "--schedule-start-update"),
                    "441",
                )
                self.assertEqual(
                    command_value(command, "--eval-interval"),
                    "452",
                )
                self.assertEqual(
                    command_value(command, "--checkpoint-interval"),
                    "452",
                )

    def test_b_covers_every_eligible_archetype_policy_and_both_seats(self) -> None:
        plan = self.build_plan()
        branch = plan["branches"]["B_gold_league"]
        expected_archetypes = {
            opponent.archetype for opponent in self.gold_opponents
        }
        expected_policies = {
            opponent.policy_id for opponent in self.gold_opponents
        }
        self.assertEqual(set(branch["archetypes"]), expected_archetypes)
        self.assertEqual(set(branch["policy_ids"]), expected_policies)
        self.assertEqual(
            set(branch["opponent_quotas_per_update"]),
            {"bc", *(opponent.name for opponent in self.gold_opponents)},
        )
        self.assertEqual(
            sum(branch["opponent_quotas_per_update"].values()),
            runner.PROFILES["pilot"].games_per_update,
        )
        for opponent in self.gold_opponents:
            quota = branch["opponent_quotas_per_update"][opponent.name]
            seats = branch["learner_seat_quotas_per_update"][opponent.name]
            self.assertGreaterEqual(quota, 2)
            self.assertEqual(quota % 2, 0)
            self.assertGreater(seats["learner_seat_0"], 0)

    def test_explicit_gold_policy_quotas_are_exact_and_seat_balanced(self) -> None:
        explicit = {
            "marnie-a": 18,
            "marnie-b": 14,
            "alakazam-a": 12,
            "mewtwo-a": 12,
        }
        plan = runner.build_plan(
            league_manifest=self.manifest,
            output_root=self.root / "output-explicit",
            profile=runner.PROFILES["pilot"],
            seeds=runner.PROFILES["pilot"].seeds,
            control_mode="bc_v3",
            bc_checkpoint=self.bc_checkpoint,
            kl_reference_checkpoint=self.kl_reference,
            learner_deck=self.learner_deck,
            resume_checkpoint=self.resume_checkpoint,
            resume_update=440,
            resume_feature_version="ptcg-selfplay-ppo-terminal01-v1",
            gold_opponents=self.gold_opponents,
            quality_excluded=self.quality_excluded,
            device="cpu",
            gold_policy_quotas=explicit,
        )
        branch = plan["branches"]["B_gold_league"]
        by_policy = {
            opponent["policy_id"]: branch["opponent_quotas_per_update"][
                opponent["name"]
            ]
            for opponent in branch["opponents"]
            if opponent["policy_id"] != "base_bc_anchor"
        }
        self.assertEqual(by_policy, explicit)
        self.assertEqual(plan["protocol"]["gold_quota_mode"], "explicit_policy")
        self.assertEqual(
            plan["protocol"]["explicit_gold_policy_quotas"],
            dict(sorted(explicit.items())),
        )
        for policy_id, games in explicit.items():
            opponent = next(
                item
                for item in branch["opponents"]
                if item["policy_id"] == policy_id
            )
            seats = branch["learner_seat_quotas_per_update"][
                opponent["name"]
            ]
            self.assertEqual(seats["learner_seat_0"], games // 2)
            self.assertEqual(seats["learner_seat_1"], games // 2)

    def test_explicit_gold_policy_quotas_reject_bad_coverage_and_sum(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every eligible policy"):
            runner.allocate_explicit_policy_quotas(
                56,
                self.gold_opponents,
                {"marnie-a": 56},
            )
        with self.assertRaisesRegex(ValueError, "sum to"):
            runner.allocate_explicit_policy_quotas(
                56,
                self.gold_opponents,
                {
                    "marnie-a": 2,
                    "marnie-b": 2,
                    "alakazam-a": 2,
                    "mewtwo-a": 2,
                },
            )
            self.assertGreater(seats["learner_seat_1"], 0)
            self.assertEqual(
                seats["learner_seat_0"],
                seats["learner_seat_1"],
            )

    def test_commands_use_fresh_bc_standard_ppo_and_last_block_heads(self) -> None:
        plan = self.build_plan()
        self.assertEqual(
            plan["protocol"]["actor_learning_rate"],
            runner.DEFAULT_ACTOR_LEARNING_RATE,
        )
        self.assertEqual(
            plan["protocol"]["actor_reduction"],
            runner.DEFAULT_ACTOR_REDUCTION,
        )
        self.assertEqual(
            plan["protocol"]["advantage_normalization"],
            runner.DEFAULT_ADVANTAGE_NORMALIZATION,
        )
        self.assertEqual(
            plan["protocol"]["gae_lambda"],
            runner.DEFAULT_GAE_LAMBDA,
        )
        self.assertEqual(
            plan["protocol"]["bc_kl_coefficient"],
            runner.DEFAULT_BC_KL_COEFFICIENT,
        )
        self.assertEqual(
            plan["protocol"]["value_learning_rate"],
            runner.DEFAULT_VALUE_LEARNING_RATE,
        )
        self.assertEqual(
            plan["protocol"]["value_coefficient"],
            runner.DEFAULT_VALUE_COEFFICIENT,
        )
        self.assertEqual(
            plan["protocol"]["entropy_coefficient"],
            runner.DEFAULT_ENTROPY_COEFFICIENT,
        )
        self.assertEqual(
            plan["protocol"]["policy_temperature"],
            runner.DEFAULT_POLICY_TEMPERATURE,
        )
        self.assertEqual(
            plan["protocol"]["value_trunk_gradient_scale"],
            runner.DEFAULT_VALUE_TRUNK_GRADIENT_SCALE,
        )
        self.assertEqual(
            plan["protocol"]["actor_value_gradient_mode"],
            runner.DEFAULT_ACTOR_VALUE_GRADIENT_MODE,
        )
        self.assertEqual(
            plan["protocol"]["bc_replay"],
            {
                "enabled": False,
                "data": None,
                "data_sha256": None,
                "split": "train",
                "batches": 0,
                "batch_size": 256,
                "workers": 8,
                "steps": 0,
                "lr_scale": 0.25,
                "loss": "set",
                "order_context_weight": 1.0,
                "non_context34_fixed_multi_action_order_weight": 1.0,
                "context34_rows_per_batch": 0,
            },
        )
        for branch in plan["branches"].values():
            for run in branch["runs"]:
                command = run["command"]
                self.assertFalse(
                    any(
                        value.startswith("--bc-replay-")
                        for value in command
                    )
                )
                self.assertEqual(
                    float(command_value(command, "--learning-rate")),
                    runner.DEFAULT_ACTOR_LEARNING_RATE,
                )
                self.assertEqual(
                    command_value(command, "--gae-lambda"),
                    "0.97",
                )
                self.assertEqual(
                    float(command_value(command, "--value-learning-rate")),
                    runner.DEFAULT_VALUE_LEARNING_RATE,
                )
                self.assertEqual(
                    float(command_value(command, "--value-coefficient")),
                    runner.DEFAULT_VALUE_COEFFICIENT,
                )
                self.assertEqual(
                    float(command_value(command, "--entropy-coefficient")),
                    runner.DEFAULT_ENTROPY_COEFFICIENT,
                )
                self.assertEqual(
                    float(
                        command_value(
                            command,
                            "--value-trunk-gradient-scale",
                        )
                    ),
                    runner.DEFAULT_VALUE_TRUNK_GRADIENT_SCALE,
                )
                self.assertEqual(
                    command_value(
                        command,
                        "--actor-value-gradient-mode",
                    ),
                    runner.DEFAULT_ACTOR_VALUE_GRADIENT_MODE,
                )
                self.assertEqual(
                    float(command_value(command, "--bc-kl-start")),
                    runner.DEFAULT_BC_KL_COEFFICIENT,
                )
                self.assertEqual(
                    float(command_value(command, "--bc-kl-end")),
                    runner.DEFAULT_BC_KL_COEFFICIENT,
                )
                self.assertEqual(
                    command_value(command, "--resume-learner-weights"),
                    "bc",
                )
                self.assertIn("--reset-optimizer-on-resume", command)
                self.assertIn("--reset-opponent-quota-on-resume", command)
                self.assertEqual(
                    command_value(command, "--kl-reference-checkpoint"),
                    str(self.kl_reference),
                )
                self.assertEqual(
                    command_value(command, "--ppo-objective"),
                    "standard",
                )
                self.assertEqual(
                    command_value(command, "--actor-reduction"),
                    runner.DEFAULT_ACTOR_REDUCTION,
                )
                self.assertEqual(
                    command_value(command, "--advantage-normalization"),
                    "global",
                )
                self.assertEqual(
                    command_value(command, "--trainable-scope"),
                    runner.DEFAULT_TRAINABLE_SCOPE,
                )
                self.assertIn("--opponent-quota-seat-balance", command)
                self.assertNotIn("--constrained-gradient-mode", command)
                self.assertNotIn("--primary-opponent-name", command)
                self.assertNotIn("--guard-opponent-name", command)

    def test_incumbent_resume_weights_and_minibatch_384_are_audited(
        self,
    ) -> None:
        plan = self.build_plan(
            resume_learner_weights="resume",
            minibatch_size=384,
        )
        inputs = plan["inputs"]
        protocol = plan["protocol"]
        invariants = plan["invariants"]

        self.assertEqual(
            inputs["resume_usage"],
            "global update metadata and full learner weights",
        )
        self.assertEqual(
            inputs["learner_weight_source"],
            "full incumbent PPO resume checkpoint",
        )
        self.assertEqual(
            inputs["learner_weight_checkpoint"],
            str(self.resume_checkpoint),
        )
        self.assertEqual(
            inputs["learner_weight_checkpoint_sha256"],
            runner.file_sha256(self.resume_checkpoint),
        )
        self.assertEqual(protocol["resume_learner_weights"], "resume")
        self.assertEqual(protocol["minibatch_size"], 384)
        self.assertEqual(protocol["common_budget"]["minibatch_size"], 384)
        self.assertNotIn(
            "learner_initialization_is_full_fresh_bc",
            invariants,
        )
        self.assertNotIn("resume_is_update_metadata_only", invariants)
        self.assertTrue(
            invariants["learner_initialization_is_full_incumbent_ppo"]
        )
        self.assertTrue(
            invariants[
                "resume_provides_update_metadata_and_learner_weights"
            ]
        )
        self.assertTrue(
            invariants["resume_learner_weights_matches_every_command"]
        )
        self.assertTrue(invariants["minibatch_size_matches_every_command"])

        for branch in plan["branches"].values():
            self.assertEqual(branch["budget"]["minibatch_size"], 384)
            for run in branch["runs"]:
                command = run["command"]
                self.assertEqual(
                    command.count("--resume-learner-weights"),
                    1,
                )
                self.assertEqual(
                    command_value(command, "--resume-learner-weights"),
                    "resume",
                )
                self.assertEqual(
                    command_value(command, "--minibatch-size"),
                    "384",
                )
                self.assertIn("--reset-optimizer-on-resume", command)
                self.assertIn(
                    "--reset-opponent-quota-on-resume",
                    command,
                )

    def test_resume_weight_source_and_minibatch_reject_invalid_api_values(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported resume learner weight source",
        ):
            self.build_plan(resume_learner_weights="incumbent")
        for invalid in (0, -1, True, 384.0):
            with self.subTest(minibatch_size=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "Minibatch size must be a positive integer",
                ):
                    self.build_plan(minibatch_size=invalid)

    def test_last_two_blocks_scope_is_explicit_and_only_changes_scope(
        self,
    ) -> None:
        baseline = self.build_plan()
        candidate = self.build_plan(
            trainable_scope="last_two_blocks_heads"
        )
        self.assertEqual(
            baseline["protocol"]["trainable_scope"],
            "last_block_heads",
        )
        self.assertEqual(
            candidate["protocol"]["trainable_scope"],
            "last_two_blocks_heads",
        )
        baseline_protocol = dict(baseline["protocol"])
        candidate_protocol = dict(candidate["protocol"])
        baseline_protocol.pop("trainable_scope")
        candidate_protocol.pop("trainable_scope")
        self.assertEqual(candidate_protocol, baseline_protocol)
        self.assertTrue(
            candidate["invariants"][
                "trainable_scope_matches_every_command"
            ]
        )
        for branch_name, baseline_branch in baseline["branches"].items():
            candidate_branch = candidate["branches"][branch_name]
            for baseline_run, candidate_run in zip(
                baseline_branch["runs"],
                candidate_branch["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                candidate_command = list(candidate_run["command"])
                scope_index = baseline_command.index("--trainable-scope") + 1
                self.assertEqual(scope_index, candidate_command.index(
                    "--trainable-scope"
                ) + 1)
                self.assertEqual(
                    baseline_command[scope_index],
                    "last_block_heads",
                )
                self.assertEqual(
                    candidate_command[scope_index],
                    "last_two_blocks_heads",
                )
                candidate_command[scope_index] = baseline_command[scope_index]
                self.assertEqual(candidate_command, baseline_command)

    def test_enabled_bc_replay_is_exactly_audited_and_passed_through(
        self,
    ) -> None:
        baseline = self.build_plan()
        enabled = self.build_plan(
            bc_replay_data=self.bc_replay_data,
            bc_replay_split="valid",
            bc_replay_batches=3,
            bc_replay_batch_size=64,
            bc_replay_workers=2,
            bc_replay_steps=4,
            bc_replay_lr_scale=0.5,
            bc_replay_loss="hybrid_ordered",
            bc_replay_order_context_weight=2.0,
            bc_replay_non_context34_fixed_multi_action_order_weight=1.0,
            bc_replay_context34_rows_per_batch=7,
        )
        self.assertEqual(
            enabled["protocol"]["bc_replay"],
            {
                "enabled": True,
                "data": str(self.bc_replay_data.resolve()),
                "data_sha256": runner.file_sha256(self.bc_replay_data),
                "split": "valid",
                "batches": 3,
                "batch_size": 64,
                "workers": 2,
                "steps": 4,
                "lr_scale": 0.5,
                "loss": "hybrid_ordered",
                "order_context_weight": 2.0,
                "non_context34_fixed_multi_action_order_weight": 1.0,
                "context34_rows_per_batch": 7,
            },
        )
        expected_values = {
            "--bc-replay-data": str(self.bc_replay_data.resolve()),
            "--bc-replay-split": "valid",
            "--bc-replay-batches": "3",
            "--bc-replay-batch-size": "64",
            "--bc-replay-workers": "2",
            "--bc-replay-steps": "4",
            "--bc-replay-lr-scale": "0.5",
            "--bc-replay-loss": "hybrid_ordered",
            "--bc-replay-order-context-weight": "2.0",
            "--bc-replay-non-context34-fixed-multi-action-order-weight": "1.0",
            "--bc-replay-context34-rows-per-batch": "7",
        }
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            enabled_runs = enabled["branches"][branch_name]["runs"]
            for baseline_run, enabled_run in zip(
                baseline_runs,
                enabled_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                enabled_command = list(enabled_run["command"])
                for flag, expected in expected_values.items():
                    self.assertEqual(
                        command_value(enabled_command, flag),
                        expected,
                    )
                replay_value_indices = sorted(
                    (
                        enabled_command.index(flag)
                        for flag in expected_values
                    ),
                    reverse=True,
                )
                for flag_index in replay_value_indices:
                    del enabled_command[flag_index : flag_index + 2]
                self.assertEqual(enabled_command, baseline_command)

    def test_bc_replay_requires_data_batches_and_steps_together(self) -> None:
        incomplete = (
            {"bc_replay_data": self.bc_replay_data},
            {"bc_replay_batches": 2},
            {"bc_replay_steps": 1},
            {
                "bc_replay_data": self.bc_replay_data,
                "bc_replay_batches": 2,
            },
            {
                "bc_replay_data": self.bc_replay_data,
                "bc_replay_steps": 1,
            },
        )
        for values in incomplete:
            with self.subTest(values=values):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires data, batches > 0, and steps > 0",
                ):
                    self.build_plan(**values)
        with self.assertRaisesRegex(
            ValueError,
            "tuning parameters require",
        ):
            self.build_plan(bc_replay_loss="ordered")
        with self.assertRaisesRegex(
            ValueError,
            "tuning parameters require",
        ):
            self.build_plan(bc_replay_context34_rows_per_batch=1)
        with self.assertRaisesRegex(
            ValueError,
            "tuning parameters require",
        ):
            self.build_plan(
                bc_replay_non_context34_fixed_multi_action_order_weight=2.0
            )
        with self.assertRaises(FileNotFoundError):
            self.build_plan(
                bc_replay_data=self.root / "missing-replay.zip",
                bc_replay_batches=2,
                bc_replay_steps=1,
            )

    def test_fixed_multi_action_weight_requires_ordered_replay(self) -> None:
        common = {
            "bc_replay_data": self.bc_replay_data,
            "bc_replay_batches": 2,
            "bc_replay_steps": 1,
            "bc_replay_non_context34_fixed_multi_action_order_weight": 2.0,
        }
        with self.assertRaisesRegex(ValueError, "requires ordered loss"):
            self.build_plan(
                **common,
                bc_replay_loss="hybrid_ordered",
            )

        plan = self.build_plan(
            **common,
            bc_replay_loss="ordered",
        )
        replay = plan["protocol"]["bc_replay"]
        self.assertEqual(
            replay["non_context34_fixed_multi_action_order_weight"],
            2.0,
        )
        for branch in plan["branches"].values():
            for run in branch["runs"]:
                self.assertEqual(
                    command_value(
                        run["command"],
                        (
                            "--bc-replay-non-context34-fixed-multi-action-"
                            "order-weight"
                        ),
                    ),
                    "2.0",
                )

    def test_bc_replay_rejects_invalid_numeric_and_enum_settings(self) -> None:
        invalid_settings = (
            ({"bc_replay_batches": -1}, "non-negative integer"),
            ({"bc_replay_batch_size": 0}, "positive integer"),
            ({"bc_replay_workers": 0}, "positive integer"),
            ({"bc_replay_steps": -1}, "non-negative integer"),
            (
                {"bc_replay_context34_rows_per_batch": -1},
                "non-negative integer",
            ),
            ({"bc_replay_lr_scale": 0.0}, r"in \(0, 1\]"),
            ({"bc_replay_lr_scale": 1.1}, r"in \(0, 1\]"),
            ({"bc_replay_lr_scale": float("nan")}, r"in \(0, 1\]"),
            (
                {"bc_replay_order_context_weight": 0.0},
                "finite and positive",
            ),
            (
                {
                    "bc_replay_non_context34_fixed_multi_action_order_weight": (
                        0.0
                    )
                },
                "finite and positive",
            ),
            ({"bc_replay_split": "holdout"}, "Unsupported BC replay split"),
            ({"bc_replay_loss": "sequence"}, "Unsupported BC replay loss"),
        )
        for values, message in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    self.build_plan(**values)

    def test_bc_replay_context34_quota_must_be_below_batch_size(self) -> None:
        common = {
            "bc_replay_data": self.bc_replay_data,
            "bc_replay_batches": 2,
            "bc_replay_batch_size": 16,
            "bc_replay_steps": 1,
        }
        for invalid in (16, 17):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "strictly less than batch_size",
                ):
                    self.build_plan(
                        **common,
                        bc_replay_context34_rows_per_batch=invalid,
                    )

    def test_parser_accepts_and_rejects_bc_replay_settings(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-replay"),
                "--bc-replay-data",
                str(self.bc_replay_data),
                "--bc-replay-split",
                "test",
                "--bc-replay-batches",
                "5",
                "--bc-replay-batch-size",
                "128",
                "--bc-replay-workers",
                "3",
                "--bc-replay-steps",
                "2",
                "--bc-replay-lr-scale",
                "0.4",
                "--bc-replay-loss",
                "ordered",
                "--bc-replay-order-context-weight",
                "1.5",
                "--bc-replay-non-context34-fixed-multi-action-order-weight",
                "1.75",
                "--bc-replay-context34-rows-per-batch",
                "9",
            ]
        )
        self.assertEqual(args.bc_replay_data, self.bc_replay_data)
        self.assertEqual(args.bc_replay_split, "test")
        self.assertEqual(args.bc_replay_batches, 5)
        self.assertEqual(args.bc_replay_batch_size, 128)
        self.assertEqual(args.bc_replay_workers, 3)
        self.assertEqual(args.bc_replay_steps, 2)
        self.assertEqual(args.bc_replay_lr_scale, 0.4)
        self.assertEqual(args.bc_replay_loss, "ordered")
        self.assertEqual(args.bc_replay_order_context_weight, 1.5)
        self.assertEqual(
            args.bc_replay_non_context34_fixed_multi_action_order_weight,
            1.75,
        )
        self.assertEqual(args.bc_replay_context34_rows_per_batch, 9)

        invalid_cli = (
            ("--bc-replay-batches", "-1"),
            ("--bc-replay-batch-size", "0"),
            ("--bc-replay-workers", "0"),
            ("--bc-replay-steps", "-1"),
            ("--bc-replay-lr-scale", "0"),
            ("--bc-replay-lr-scale", "1.1"),
            ("--bc-replay-lr-scale", "nan"),
            ("--bc-replay-order-context-weight", "0"),
            ("--bc-replay-order-context-weight", "inf"),
            (
                "--bc-replay-non-context34-fixed-multi-action-order-weight",
                "0",
            ),
            (
                "--bc-replay-non-context34-fixed-multi-action-order-weight",
                "inf",
            ),
            ("--bc-replay-context34-rows-per-batch", "-1"),
            ("--bc-replay-split", "holdout"),
            ("--bc-replay-loss", "sequence"),
        )
        for flag, value in invalid_cli:
            with self.subTest(flag=flag, value=value):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid-replay"),
                                flag,
                                value,
                            ]
                        )

    def test_actor_learning_rate_override_is_audited_and_only_changes_lr(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(actor_learning_rate=1.8e-5)
        self.assertEqual(overridden["protocol"]["actor_learning_rate"], 1.8e-5)
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            overridden_runs = overridden["branches"][branch_name]["runs"]
            self.assertEqual(len(baseline_runs), len(overridden_runs))
            for baseline_run, overridden_run in zip(
                baseline_runs,
                overridden_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                lr_index = baseline_command.index("--learning-rate") + 1
                self.assertEqual(
                    lr_index,
                    overridden_command.index("--learning-rate") + 1,
                )
                self.assertEqual(float(overridden_command[lr_index]), 1.8e-5)
                baseline_command[lr_index] = "<actor-learning-rate>"
                overridden_command[lr_index] = "<actor-learning-rate>"
                self.assertEqual(overridden_command, baseline_command)

    def test_actor_learning_rate_rejects_nonpositive_and_nonfinite_values(
        self,
    ) -> None:
        for invalid in (0.0, -1e-5, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite positive number",
                ):
                    self.build_plan(actor_learning_rate=invalid)

        for invalid in ("0", "-1e-5", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid-lr"),
                                "--actor-learning-rate",
                                invalid,
                            ]
                        )

    def test_parser_accepts_actor_learning_rate_override(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-lr-override"),
                "--actor-learning-rate",
                "1.8e-5",
            ]
        )
        self.assertEqual(args.actor_learning_rate, 1.8e-5)

    def test_gae_lambda_override_is_audited_and_only_changes_gae_lambda(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(gae_lambda=0.8)
        self.assertEqual(overridden["protocol"]["gae_lambda"], 0.8)
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            overridden_runs = overridden["branches"][branch_name]["runs"]
            self.assertEqual(len(baseline_runs), len(overridden_runs))
            for baseline_run, overridden_run in zip(
                baseline_runs,
                overridden_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_index = (
                    baseline_command.index("--gae-lambda") + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(
                    float(overridden_command[expected_index]),
                    0.8,
                )

    def test_gae_lambda_rejects_out_of_range_and_nonfinite_values(
        self,
    ) -> None:
        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    r"finite and in \[0, 1\]",
                ):
                    self.build_plan(gae_lambda=invalid)

        for invalid in ("-0.01", "1.01", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid-gae-lambda"),
                                "--gae-lambda",
                                invalid,
                            ]
                        )

    def test_parser_accepts_gae_lambda_override_and_boundaries(self) -> None:
        for raw, expected in (("0", 0.0), ("0.8", 0.8), ("1", 1.0)):
            with self.subTest(raw=raw):
                args = runner.parse_args(
                    [
                        "--league-manifest",
                        str(self.manifest),
                        "--output-root",
                        str(self.root / "output-gae-lambda"),
                        "--gae-lambda",
                        raw,
                    ]
                )
                self.assertEqual(args.gae_lambda, expected)

    def test_quota_group_actor_reduction_is_single_command_change(self) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(actor_reduction="quota_group_mean")
        self.assertEqual(
            overridden["protocol"]["actor_reduction"],
            "quota_group_mean",
        )
        for branch_name in baseline["branches"]:
            for baseline_run, overridden_run in zip(
                baseline["branches"][branch_name]["runs"],
                overridden["branches"][branch_name]["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                reduction_index = (
                    baseline_command.index("--actor-reduction") + 1
                )
                self.assertEqual(
                    reduction_index,
                    overridden_command.index("--actor-reduction") + 1,
                )
                self.assertEqual(
                    overridden_command[reduction_index],
                    "quota_group_mean",
                )
                baseline_command[reduction_index] = "<actor-reduction>"
                overridden_command[reduction_index] = "<actor-reduction>"
                self.assertEqual(overridden_command, baseline_command)

    def test_parser_accepts_quota_group_actor_reduction(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-actor-reduction"),
                "--actor-reduction",
                "quota_group_mean",
            ]
        )
        self.assertEqual(args.actor_reduction, "quota_group_mean")

    def test_episode_mean_actor_reduction_is_single_command_change(self) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(actor_reduction="episode_mean")
        self.assertEqual(
            overridden["protocol"]["actor_reduction"],
            "episode_mean",
        )
        for branch_name in baseline["branches"]:
            for baseline_run, overridden_run in zip(
                baseline["branches"][branch_name]["runs"],
                overridden["branches"][branch_name]["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                reduction_index = (
                    baseline_command.index("--actor-reduction") + 1
                )
                self.assertEqual(
                    overridden_command[reduction_index],
                    "episode_mean",
                )
                baseline_command[reduction_index] = "<actor-reduction>"
                overridden_command[reduction_index] = "<actor-reduction>"
                self.assertEqual(overridden_command, baseline_command)

    def test_parser_accepts_episode_mean_actor_reduction(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-episode-reduction"),
                "--actor-reduction",
                "episode_mean",
            ]
        )
        self.assertEqual(args.actor_reduction, "episode_mean")

    def test_per_opponent_normalization_is_single_command_change(self) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(
            advantage_normalization="per_opponent",
        )
        self.assertEqual(
            overridden["protocol"]["advantage_normalization"],
            "per_opponent",
        )
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            for baseline_run, overridden_run in zip(
                baseline["branches"][branch_name]["runs"],
                overridden["branches"][branch_name]["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                normalization_index = (
                    baseline_command.index("--advantage-normalization") + 1
                )
                self.assertEqual(
                    normalization_index,
                    overridden_command.index("--advantage-normalization") + 1,
                )
                self.assertEqual(
                    overridden_command[normalization_index],
                    "per_opponent",
                )
                baseline_command[normalization_index] = (
                    "<advantage-normalization>"
                )
                overridden_command[normalization_index] = (
                    "<advantage-normalization>"
                )
                self.assertEqual(overridden_command, baseline_command)

    def test_parser_accepts_per_opponent_normalization(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-advantage-normalization"),
                "--advantage-normalization",
                "per_opponent",
            ]
        )
        self.assertEqual(args.advantage_normalization, "per_opponent")

    def test_quota_group_reduction_still_requires_global_advantages(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires global advantage normalization",
        ):
            self.build_plan(
                actor_reduction="quota_group_mean",
                advantage_normalization="per_opponent",
            )

    def test_value_learning_rate_override_is_audited_and_only_changes_lr(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(value_learning_rate=7.5e-6)
        self.assertEqual(
            overridden["protocol"]["value_learning_rate"],
            7.5e-6,
        )
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            overridden_runs = overridden["branches"][branch_name]["runs"]
            self.assertEqual(len(baseline_runs), len(overridden_runs))
            for baseline_run, overridden_run in zip(
                baseline_runs,
                overridden_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_index = (
                    baseline_command.index("--value-learning-rate") + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(
                    float(overridden_command[expected_index]),
                    7.5e-6,
                )

    def test_value_learning_rate_rejects_nonpositive_and_nonfinite_values(
        self,
    ) -> None:
        for invalid in (0.0, -1e-5, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite positive number",
                ):
                    self.build_plan(value_learning_rate=invalid)

        for invalid in ("0", "-1e-5", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid-value-lr"),
                                "--value-learning-rate",
                                invalid,
                            ]
                        )

    def test_parser_accepts_value_learning_rate_override(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-value-lr-override"),
                "--value-learning-rate",
                "7.5e-6",
            ]
        )
        self.assertEqual(args.value_learning_rate, 7.5e-6)

    def test_value_coefficient_override_is_audited_and_only_changes_value(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(value_coefficient=0.25)
        self.assertEqual(overridden["protocol"]["value_coefficient"], 0.25)
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            overridden_runs = overridden["branches"][branch_name]["runs"]
            self.assertEqual(len(baseline_runs), len(overridden_runs))
            for baseline_run, overridden_run in zip(
                baseline_runs,
                overridden_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_index = (
                    baseline_command.index("--value-coefficient") + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(
                    float(overridden_command[expected_index]),
                    0.25,
                )

    def test_value_coefficient_rejects_nonpositive_and_nonfinite_values(
        self,
    ) -> None:
        for invalid in (0.0, -0.5, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite positive number",
                ):
                    self.build_plan(value_coefficient=invalid)

        for invalid in ("0", "-0.5", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(
                                    self.root
                                    / "output-invalid-value-coefficient"
                                ),
                                "--value-coefficient",
                                invalid,
                            ]
                        )

    def test_parser_accepts_value_coefficient_override(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-value-coefficient-override"),
                "--value-coefficient",
                "0.25",
            ]
        )
        self.assertEqual(args.value_coefficient, 0.25)

    def test_entropy_zero_is_audited_and_only_changes_entropy_value(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(entropy_coefficient=0.0)
        self.assertEqual(overridden["protocol"]["entropy_coefficient"], 0.0)
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            overridden_runs = overridden["branches"][branch_name]["runs"]
            self.assertEqual(len(baseline_runs), len(overridden_runs))
            for baseline_run, overridden_run in zip(
                baseline_runs,
                overridden_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_index = (
                    baseline_command.index("--entropy-coefficient") + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(float(overridden_command[expected_index]), 0.0)

    def test_entropy_coefficient_rejects_negative_and_nonfinite_values(
        self,
    ) -> None:
        for invalid in (-0.001, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite non-negative number",
                ):
                    self.build_plan(entropy_coefficient=invalid)

        for invalid in ("-0.001", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(
                                    self.root
                                    / "output-invalid-entropy-coefficient"
                                ),
                                "--entropy-coefficient",
                                invalid,
                            ]
                        )

    def test_parser_accepts_entropy_zero(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-entropy-zero"),
                "--entropy-coefficient",
                "0",
            ]
        )
        self.assertEqual(args.entropy_coefficient, 0.0)

    def test_policy_temperature_override_is_single_command_change(self) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(policy_temperature=0.7)
        self.assertEqual(overridden["protocol"]["policy_temperature"], 0.7)
        self.assertEqual(
            baseline["protocol"]["policy_temperature"],
            runner.DEFAULT_POLICY_TEMPERATURE,
        )
        for branch_name in baseline["branches"]:
            for baseline_run, overridden_run in zip(
                baseline["branches"][branch_name]["runs"],
                overridden["branches"][branch_name]["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_index = (
                    baseline_command.index("--policy-temperature") + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(
                    float(overridden_command[expected_index]),
                    0.7,
                )

    def test_policy_temperature_rejects_nonpositive_and_nonfinite(self) -> None:
        for invalid in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite positive number",
                ):
                    self.build_plan(policy_temperature=invalid)

        for invalid in ("0", "-0.1", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid-temperature"),
                                "--policy-temperature",
                                invalid,
                            ]
                        )

    def test_parser_accepts_policy_temperature_override(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-temperature"),
                "--policy-temperature",
                "0.7",
            ]
        )
        self.assertEqual(args.policy_temperature, 0.7)

    def test_value_trunk_scale_zero_is_audited_as_single_command_change(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(value_trunk_gradient_scale=0.0)
        self.assertEqual(
            overridden["protocol"]["value_trunk_gradient_scale"],
            0.0,
        )
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            for baseline_run, overridden_run in zip(
                baseline["branches"][branch_name]["runs"],
                overridden["branches"][branch_name]["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_index = (
                    baseline_command.index(
                        "--value-trunk-gradient-scale"
                    )
                    + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(
                    float(overridden_command[expected_index]),
                    0.0,
                )

    def test_value_trunk_scale_parser_boundaries_and_invalids(self) -> None:
        for raw, expected in (("0", 0.0), ("0.5", 0.5), ("1", 1.0)):
            with self.subTest(valid=raw):
                args = runner.parse_args(
                    [
                        "--league-manifest",
                        str(self.manifest),
                        "--output-root",
                        str(self.root / "output-value-trunk-scale"),
                        "--value-trunk-gradient-scale",
                        raw,
                    ]
                )
                self.assertEqual(args.value_trunk_gradient_scale, expected)

        for invalid in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(build_invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite.*\\[0, 1\\]"):
                    self.build_plan(value_trunk_gradient_scale=invalid)
        for invalid in ("-0.01", "1.01", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(
                                    self.root
                                    / "output-invalid-value-trunk-scale"
                                ),
                                "--value-trunk-gradient-scale",
                                invalid,
                            ]
                        )

    def test_actor_priority_value_pcgrad_is_one_frozen_protocol_change(
        self,
    ) -> None:
        baseline = self.build_plan()
        candidate = self.build_plan(
            actor_value_gradient_mode="actor_priority_value_pcgrad",
        )
        self.assertEqual(
            candidate["protocol"]["actor_value_gradient_mode"],
            "actor_priority_value_pcgrad",
        )
        frozen_protocol_fields = (
            "common_budget",
            "paired_seeds",
            "actor_learning_rate",
            "gae_lambda",
            "actor_reduction",
            "advantage_normalization",
            "value_learning_rate",
            "value_coefficient",
            "entropy_coefficient",
            "value_trunk_gradient_scale",
            "bc_kl_coefficient",
            "bc_replay",
        )
        for field in frozen_protocol_fields:
            with self.subTest(protocol_field=field):
                self.assertEqual(
                    candidate["protocol"][field],
                    baseline["protocol"][field],
                )
        for branch_name in baseline["branches"]:
            for baseline_run, candidate_run in zip(
                baseline["branches"][branch_name]["runs"],
                candidate["branches"][branch_name]["runs"],
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                candidate_command = list(candidate_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, candidate_value) in enumerate(
                        zip(
                            baseline_command,
                            candidate_command,
                            strict=True,
                        )
                    )
                    if baseline_value != candidate_value
                }
                expected_index = (
                    baseline_command.index(
                        "--actor-value-gradient-mode"
                    )
                    + 1
                )
                self.assertEqual(changed_indices, {expected_index})
                self.assertEqual(
                    candidate_command[expected_index],
                    "actor_priority_value_pcgrad",
                )

    def test_actor_value_gradient_mode_parser_and_validation(self) -> None:
        default_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-gradient-mode-default"),
            ]
        )
        self.assertEqual(
            default_args.actor_value_gradient_mode,
            "scalar",
        )
        candidate_args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-gradient-mode-candidate"),
                "--actor-value-gradient-mode",
                "actor_priority_value_pcgrad",
            ]
        )
        self.assertEqual(
            candidate_args.actor_value_gradient_mode,
            "actor_priority_value_pcgrad",
        )
        with mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        "--league-manifest",
                        str(self.manifest),
                        "--output-root",
                        str(self.root / "output-gradient-mode-invalid"),
                        "--actor-value-gradient-mode",
                        "unknown",
                    ]
                )
        with self.assertRaisesRegex(
            ValueError,
            "requires.*value_trunk_gradient_scale=1.0",
        ):
            self.build_plan(
                actor_value_gradient_mode=(
                    "actor_priority_value_pcgrad"
                ),
                value_trunk_gradient_scale=0.0,
            )

    def test_bc_kl_override_is_audited_and_only_changes_schedule_endpoints(
        self,
    ) -> None:
        baseline = self.build_plan()
        overridden = self.build_plan(bc_kl_coefficient=0.012)
        self.assertEqual(overridden["protocol"]["bc_kl_coefficient"], 0.012)
        self.assertEqual(
            overridden["protocol"]["common_budget"],
            baseline["protocol"]["common_budget"],
        )
        for branch_name in baseline["branches"]:
            baseline_runs = baseline["branches"][branch_name]["runs"]
            overridden_runs = overridden["branches"][branch_name]["runs"]
            self.assertEqual(len(baseline_runs), len(overridden_runs))
            for baseline_run, overridden_run in zip(
                baseline_runs,
                overridden_runs,
                strict=True,
            ):
                baseline_command = list(baseline_run["command"])
                overridden_command = list(overridden_run["command"])
                changed_indices = {
                    index
                    for index, (baseline_value, overridden_value) in enumerate(
                        zip(
                            baseline_command,
                            overridden_command,
                            strict=True,
                        )
                    )
                    if baseline_value != overridden_value
                }
                expected_indices = {
                    baseline_command.index("--bc-kl-start") + 1,
                    baseline_command.index("--bc-kl-end") + 1,
                }
                self.assertEqual(changed_indices, expected_indices)
                for index in expected_indices:
                    self.assertEqual(float(overridden_command[index]), 0.012)

    def test_bc_kl_coefficient_rejects_nonpositive_and_nonfinite_values(
        self,
    ) -> None:
        for invalid in (0.0, -0.004, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite positive number",
                ):
                    self.build_plan(bc_kl_coefficient=invalid)

        for invalid in ("0", "-0.004", "nan", "inf"):
            with self.subTest(cli_invalid=invalid):
                with mock.patch.object(sys, "stderr"):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                "--league-manifest",
                                str(self.manifest),
                                "--output-root",
                                str(self.root / "output-invalid-bc-kl"),
                                "--bc-kl-coefficient",
                                invalid,
                            ]
                        )

    def test_parser_accepts_bc_kl_coefficient_override(self) -> None:
        args = runner.parse_args(
            [
                "--league-manifest",
                str(self.manifest),
                "--output-root",
                str(self.root / "output-bc-kl-override"),
                "--bc-kl-coefficient",
                "0.012",
            ]
        )
        self.assertEqual(args.bc_kl_coefficient, 0.012)

    def test_control_a_is_marnie_only(self) -> None:
        plan = self.build_plan(control_mode="bc_v3")
        control = plan["branches"]["A_marnie_control"]
        self.assertEqual(
            {opponent["policy_id"] for opponent in control["opponents"]},
            {"base_bc_anchor", "incumbent_v3"},
        )
        self.assertTrue(
            all(
                "marnie" in opponent["archetype"].casefold()
                for opponent in control["opponents"]
            )
        )
        marnie_clone_plan = self.build_plan(
            control_mode="marnie_league"
        )
        self.assertTrue(
            all(
                "marnie" in opponent["archetype"].casefold()
                for opponent in marnie_clone_plan["branches"][
                    "A_marnie_control"
                ]["opponents"]
            )
        )

    def test_plan_has_no_external_action_and_executor_is_train_only(self) -> None:
        plan = self.build_plan()
        self.assertEqual(
            plan["safety"],
            {
                "local_training_only": True,
                "allowed_child_program": str(runner.TRAIN_SCRIPT),
                "network_calls": False,
                "uploads": False,
                "submission": False,
                "packaging": False,
                "uses_open_submission_code": False,
                "uses_quality_passing_public_replay_clones_only": True,
            },
        )
        forbidden = ("kaggle", "submit", "upload", "package")
        for branch in plan["branches"].values():
            for run in branch["runs"]:
                command = run["command"]
                self.assertEqual(
                    Path(command[0]).resolve(),
                    Path(sys.executable).resolve(),
                )
                self.assertEqual(
                    Path(command[1]).resolve(),
                    runner.TRAIN_SCRIPT.resolve(),
                )
                lowered = [value.casefold() for value in command]
                self.assertFalse(
                    any(
                        word in value
                        for value in lowered
                        for word in forbidden
                    )
                )

        output_root = Path(plan["output_root"])
        registration = runner.preregistration_envelope(plan)
        with mock.patch.object(
            runner.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as child:
            summary = runner.execute_plan(
                plan,
                output_root,
                registration,
            )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(child.call_count, 2)
        for call in child.call_args_list:
            command = call.args[0]
            self.assertEqual(
                Path(command[1]).resolve(),
                runner.TRAIN_SCRIPT.resolve(),
            )
            self.assertIs(call.kwargs["stdin"], subprocess.DEVNULL)
            self.assertIs(call.kwargs["stderr"], subprocess.STDOUT)

    def test_refuses_output_overwrite_and_changed_preregistration(self) -> None:
        plan = self.build_plan()
        output_root = Path(plan["output_root"])
        output_root.mkdir()
        with self.assertRaises(FileExistsError):
            runner.execute_plan(
                plan,
                output_root,
                runner.preregistration_envelope(plan),
            )

        registration_path = self.root / "registration.json"
        registration = runner.preregistration_envelope(plan)
        registration_path.write_text(
            json.dumps(registration),
            encoding="utf-8",
        )
        changed = json.loads(json.dumps(plan))
        changed["protocol"]["common_budget"]["ppo_epochs"] = 999
        with self.assertRaisesRegex(ValueError, "differs"):
            runner.validate_existing_preregistration(
                registration_path,
                changed,
            )

    def test_budget_too_small_for_two_seats_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "both learner seats"):
            runner.allocate_even_policy_quotas(
                6,
                self.gold_opponents,
            )


if __name__ == "__main__":
    unittest.main()
