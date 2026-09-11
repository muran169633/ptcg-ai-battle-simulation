from __future__ import annotations

import argparse
import contextlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_gold_league_h2h as league
import run_gold_push_s08_full16_confirmation as launcher


ARCHETYPES = {
    **{policy_id: "Marnie Grimmsnarl" for policy_id in launcher.POLICY_IDS[:8]},
    **{policy_id: "Mega Lucario" for policy_id in launcher.POLICY_IDS[8:10]},
    **{
        policy_id: "Mega Froslass + Mega Lopunny"
        for policy_id in launcher.POLICY_IDS[10:12]
    },
    **{policy_id: "Alakazam Control" for policy_id in launcher.POLICY_IDS[12:14]},
    launcher.POLICY_IDS[14]: "Mega Kangaskhan + Crustle",
    launcher.POLICY_IDS[15]: "Dragapult + Dusknoir",
}


def make_counts(
    default_wins: int,
    *,
    wins: dict[str, int] | None = None,
    seat0_wins: int | dict[str, int] | None = None,
    invalid: dict[str, int] | None = None,
) -> tuple[launcher.panel.PolicyCounts, ...]:
    wins = wins or {}
    invalid = invalid or {}
    rows = []
    for policy_id in launcher.POLICY_IDS:
        policy_wins = wins.get(policy_id, default_wins)
        if isinstance(seat0_wins, dict):
            seat0 = seat0_wins.get(policy_id, policy_wins // 2)
        elif isinstance(seat0_wins, int):
            seat0 = seat0_wins
        else:
            seat0 = policy_wins // 2
        seat1 = policy_wins - seat0
        invalid_games = invalid.get(policy_id, 0)
        rows.append(
            launcher.panel.PolicyCounts(
                policy_id=policy_id,
                archetype=ARCHETYPES[policy_id],
                opponent_canonical_order=True,
                wins=policy_wins,
                losses=launcher.GAMES_PER_POLICY - policy_wins,
                draws=0,
                valid_games=launcher.GAMES_PER_POLICY,
                invalid_games=invalid_games,
                invalid_by_seat={"0": invalid_games, "1": 0},
                seat_wins={"0": seat0, "1": seat1},
                seat_valid_games={
                    "0": launcher.GAMES_PER_SEAT,
                    "1": launcher.GAMES_PER_SEAT,
                },
            )
        )
    return tuple(rows)


class FakeProcess:
    def __init__(
        self, pid: int, polls: list[int | None], *, timeout_once: bool = False
    ) -> None:
        self.pid = pid
        self.polls = list(polls)
        self.returncode: int | None = None
        self.timeout_once = timeout_once
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self.polls:
            value = self.polls.pop(0)
            if value is not None:
                self.returncode = value
            return value
        return None

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("fake", timeout)
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


class S08Full16ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preflight = launcher.build_preflight()

    def test_manifest_freezes_full16_hybrid_and_preregistered_gate(self) -> None:
        manifest = self.preflight.manifest
        panel = manifest["panel"]
        protocol = manifest["evaluation_protocol"]
        absolute = manifest["absolute_s08_gates"]
        self.assertEqual(panel["policies_exact"], 16)
        self.assertEqual(
            panel["policy_ids_in_execution_order"], list(launcher.POLICY_IDS)
        )
        self.assertEqual(
            sum(not row["canonical_order"] for row in panel["opponents"]), 3
        )
        self.assertEqual(protocol["games_per_candidate_per_policy"], 512)
        self.assertEqual(
            protocol["valid_games_per_candidate_per_policy_per_seat_exact"], 256
        )
        self.assertEqual(protocol["valid_games_per_candidate_exact"], 8192)
        self.assertEqual(protocol["jobs_per_candidate"], 1)
        self.assertIs(protocol["candidates_launched_concurrently"], True)
        self.assertEqual(
            protocol["candidate_action_order"],
            {
                "mode": "hybrid",
                "canonical_order": False,
                "hybrid_order": True,
                "preserve_greedy_plackett_luce_contexts": [34],
                "sort_selected_indices_for_all_other_contexts": True,
            },
        )
        self.assertEqual(
            protocol["statistical_treatment"], "independent_binomial_not_paired"
        )
        self.assertEqual(absolute["attempted_games_exact"], 8192)
        self.assertEqual(absolute["valid_games_exact"], 8192)
        self.assertEqual(absolute["pooled_wilson_95_lower_minimum"], 0.56)
        self.assertIs(absolute["invalid_timeout_truncation_counted_as_loss"], True)
        self.assertEqual(
            manifest["scope"],
            {
                "training": False,
                "local_evaluation": True,
                "package": False,
                "upload": False,
                "submission": False,
            },
        )

    def test_python_entry_commands_and_mixed_opponent_order(self) -> None:
        python = self.preflight.manifest["inputs"]["python"]
        self.assertEqual(
            python["path"], "/home/xxc/miniconda3/envs/my_project_env/bin/python"
        )
        self.assertTrue(python["resolved_path"].endswith("/bin/python3.11"))
        self.assertIs(python["entry_is_symlink"], True)
        self.assertEqual(python["sha256"], launcher.PYTHON_SHA256)
        for label, command in self.preflight.commands.items():
            with self.subTest(label=label):
                self.assertEqual(command[0], str(launcher.PYTHON))
                self.assertEqual(
                    command[1:4], ("-I", "-B", str(launcher.RUNNER))
                )
                self.assertEqual(command.count("--candidate-hybrid-order"), 1)
                self.assertNotIn("--opponent-canonical-order", command)
                self.assertNotIn("--candidate-raw-order", command)
                self.assertNotIn("--candidate-canonical-order", command)
                self.assertEqual(command[command.index("--jobs") + 1], "1")
                self.assertEqual(
                    command[command.index("--screening-games") + 1], "512"
                )
                expected = launcher.OUTPUT / f"{label}.deployment_contract.json"
                self.assertEqual(
                    command[command.index("--deployment-contract") + 1],
                    str(expected),
                )

    def test_candidate_specific_contracts_are_frozen_and_runner_valid(self) -> None:
        cases = (
            (
                "parent",
                launcher.PARENT,
                launcher.FROZEN_SHA256["parent"],
                launcher.PARENT_OUTPUT,
            ),
            (
                "s08",
                launcher.S08,
                launcher.FROZEN_SHA256["s08"],
                launcher.S08_OUTPUT,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_temp:
            temporary = Path(raw_temp)
            for label, candidate, candidate_sha, run_output in cases:
                with self.subTest(label=label):
                    frozen = self.preflight.manifest["deployment_contracts"][label]
                    payload = frozen["payload"]
                    self.assertEqual(payload["candidate"]["path"], str(candidate))
                    self.assertEqual(payload["candidate"]["sha256"], candidate_sha)
                    self.assertEqual(payload["panel"]["output_dir"], str(run_output))
                    self.assertEqual(payload["action_order"]["mode"], "hybrid")
                    self.assertEqual(
                        payload["action_order"][
                            "preserve_greedy_plackett_luce_contexts"
                        ],
                        [34],
                    )
                    self.assertEqual(
                        frozen["file_sha256"],
                        launcher.rendered_json_sha256(payload),
                    )
                    contract = temporary / f"{label}.json"
                    contract.write_bytes(launcher.render_json(payload))
                    args = argparse.Namespace(
                        candidate=candidate,
                        candidate_deck=launcher.CANDIDATE_DECK,
                        league_manifest=launcher.MANIFEST,
                        output_dir=run_output,
                    )
                    observed = league.validate_deployment_contract(
                        contract,
                        args=args,
                        candidate_sha256=candidate_sha,
                        candidate_deck_file_sha256=launcher.FROZEN_SHA256[
                            "candidate_deck"
                        ],
                        order_mode="hybrid",
                    )
                    self.assertEqual(observed, payload)

    def test_default_dry_run_is_write_free_and_does_not_spawn(self) -> None:
        with mock.patch.object(
            launcher.subprocess,
            "Popen",
            side_effect=AssertionError("dry-run must not spawn"),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(launcher.main(["--dry-run"]), 0)
        self.assertIn(self.preflight.manifest_sha256, output.getvalue())
        self.assertFalse(launcher.OUTPUT.exists())

    def test_execute_requires_reviewed_manifest_digest(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "requires --expected-manifest-sha256"
        ):
            launcher.main(["--execute"])
        with self.assertRaisesRegex(ValueError, "execute-only"):
            launcher.main(["--expected-manifest-sha256", "0" * 64])

    def test_all_gates_pass_on_strong_synthetic_panel(self) -> None:
        result = launcher.evaluate_gates(make_counts(260), make_counts(320))
        self.assertIs(result["pass"], True)
        self.assertEqual(result["failed_gates"], [])
        self.assertEqual(result["absolute_s08"]["attempted_games"], 8192)
        self.assertEqual(result["absolute_s08"]["valid_games"], 8192)
        self.assertGreaterEqual(
            result["absolute_s08"]["pooled_wilson_95_lower"], 0.56
        )
        self.assertTrue(all(result["checks"].values()))

    def test_pooled_wilson_and_macro_gates_are_enforced(self) -> None:
        result = launcher.evaluate_gates(make_counts(250), make_counts(290))
        self.assertIs(result["checks"]["s08_pooled_wilson_95_lower"], False)
        self.assertIs(result["checks"]["s08_macro_win_rate"], False)
        self.assertLess(result["absolute_s08"]["pooled_wilson_95_lower"], 0.56)

    def test_bottom4_cvar_gate_uses_lowest_four(self) -> None:
        weak = {policy_id: 260 for policy_id in launcher.BOTTOM4_POLICY_IDS}
        result = launcher.evaluate_gates(
            make_counts(240), make_counts(330, wins=weak)
        )
        self.assertIs(result["checks"]["s08_macro_win_rate"], True)
        self.assertIs(result["checks"]["s08_bottom4_cvar_win_rate"], False)
        self.assertEqual(
            set(result["absolute_s08"]["bottom4_policy_ids_observed"]),
            set(launcher.BOTTOM4_POLICY_IDS),
        )

    def test_minimum_archetype_gate_catches_single_policy_archetype(self) -> None:
        weak = {"kangaskhan_crustle_source_bc_aug08": 260}
        result = launcher.evaluate_gates(
            make_counts(240), make_counts(330, wins=weak)
        )
        self.assertIs(result["checks"]["s08_macro_win_rate"], True)
        self.assertIs(
            result["checks"]["s08_minimum_archetype_win_rate"], False
        )

    def test_each_seat_absolute_gate_is_separate(self) -> None:
        result = launcher.evaluate_gates(
            make_counts(260, seat0_wins=130),
            make_counts(320, seat0_wins=140),
        )
        self.assertIs(result["checks"]["s08_seat0_win_rate"], False)
        self.assertIs(result["checks"]["s08_seat1_win_rate"], True)

    def test_invalid_is_loss_and_breaks_exact_and_zero_invalid(self) -> None:
        result = launcher.evaluate_gates(
            make_counts(260),
            make_counts(320, invalid={launcher.POLICY_IDS[0]: 1}),
        )
        self.assertEqual(result["absolute_s08"]["attempted_games"], 8193)
        self.assertEqual(result["absolute_s08"]["wins"], 16 * 320)
        self.assertIs(result["checks"]["s08_attempted_games_exact"], False)
        self.assertIs(result["checks"]["s08_zero_invalid"], False)

    def test_each_original_target_nonregression_gate_is_retained(self) -> None:
        cases = (
            ("marnie_kdcyberdude_recent7_clone", "kdcyberdude_nonregression"),
            ("marnie_raihan_recent7_clone", "raihan_nonregression"),
        )
        for policy_id, gate in cases:
            with self.subTest(policy_id=policy_id):
                result = launcher.evaluate_gates(
                    make_counts(300), make_counts(330, wins={policy_id: 299})
                )
                point = result["relative_s08_vs_parent"][
                    "point_estimate_gates"
                ][gate]
                self.assertIs(point["pass"], False)
                self.assertIs(result["pass"], False)

    def test_other_drop_limit_is_rate_scaled_from_original(self) -> None:
        result = launcher.evaluate_gates(
            make_counts(300), make_counts(330, wins={"marnie_v1_u200": 267})
        )
        gate = result["relative_s08_vs_parent"]["point_estimate_gates"][
            "minimum_other_policy_rate_delta"
        ]
        self.assertAlmostEqual(gate["actual"], -33 / 512)
        self.assertEqual(gate["minimum"], -8 / 128)
        self.assertIs(gate["pass"], False)

    def test_newcombe_is_independent_not_paired_and_can_fail_seat(self) -> None:
        result = launcher.evaluate_gates(
            make_counts(300, seat0_wins=192),
            make_counts(330, seat0_wins=145),
        )
        interval = result["relative_s08_vs_parent"][
            "newcombe_noninferiority"
        ]["seat0"]
        self.assertEqual(
            interval["statistical_treatment"], "independent_binomial_not_paired"
        )
        self.assertLess(interval["newcombe_lower_bound"], -0.075)
        self.assertIs(interval["pass"], False)

    def test_atomic_decision_refuses_overwrite_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temporary = Path(raw_temp)
            target = temporary / "decision.json"
            digest = launcher.atomic_write_json_exclusive(target, {"pass": False})
            self.assertEqual(digest, launcher.file_sha256(target))
            with self.assertRaisesRegex(
                launcher.ConfirmationError, "Refusing to overwrite"
            ):
                launcher.atomic_write_json_exclusive(target, {"pass": True})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")), {"pass": False}
            )
            self.assertEqual(list(temporary.glob(".decision.json.*.tmp")), [])

    def test_wait_fail_fast_terminates_sibling_process_group(self) -> None:
        parent = FakeProcess(1001, [7])
        sibling = FakeProcess(1002, [None, None])
        signals: list[tuple[int, int]] = []
        with mock.patch.object(
            launcher.os,
            "killpg",
            side_effect=lambda pid, sig: signals.append((pid, sig)),
        ):
            result = launcher.wait_fail_fast(  # type: ignore[arg-type]
                {"parent": parent, "s08": sibling}, poll_seconds=0
            )
        self.assertEqual(result["parent"], 7)
        self.assertEqual(result["s08"], -signal.SIGTERM)
        self.assertEqual(signals, [(1002, signal.SIGTERM)])

    def test_process_group_cleanup_escalates_after_timeout(self) -> None:
        process = FakeProcess(2001, [None], timeout_once=True)
        signals: list[tuple[int, int]] = []
        with mock.patch.object(
            launcher.os,
            "killpg",
            side_effect=lambda pid, sig: signals.append((pid, sig)),
        ):
            launcher.terminate_process(process)  # type: ignore[arg-type]
        self.assertEqual(
            signals,
            [(2001, signal.SIGTERM), (2001, signal.SIGKILL)],
        )

    def test_execute_starts_both_process_groups_before_polling(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temporary = Path(raw_temp)
            output = temporary / "confirmation"
            parent_contract = output / "parent.deployment_contract.json"
            s08_contract = output / "s08.deployment_contract.json"
            contracts = {
                label: {
                    "payload": {"label": label},
                    "file_sha256": launcher.rendered_json_sha256(
                        {"label": label}
                    ),
                }
                for label in ("parent", "s08")
            }
            preflight = launcher.Preflight(
                manifest={"inputs": {}, "deployment_contracts": contracts},
                manifest_sha256="a" * 64,
                commands={
                    "parent": ("parent-command",),
                    "s08": ("s08-command",),
                },
                opponents=(),
            )
            created: list[Any] = []

            class ImmediateProcess:
                def __init__(inner_self, command: Any, **kwargs: Any) -> None:
                    self.assertIs(kwargs["start_new_session"], True)
                    self.assertIs(kwargs["shell"], False)
                    inner_self.command = command
                    inner_self.pid = 3000 + len(created)
                    created.append(inner_self)

                def poll(inner_self) -> int:
                    self.assertEqual(len(created), 2)
                    return 0

            passing = {
                "pass": True,
                "decision": "s08_eligible_for_separate_package_review_only",
                "scope": {
                    "package_performed": False,
                    "upload_performed": False,
                    "submission_performed": False,
                },
            }
            with (
                mock.patch.object(launcher, "OUTPUT", output),
                mock.patch.object(
                    launcher, "PARENT_DEPLOYMENT_CONTRACT", parent_contract
                ),
                mock.patch.object(
                    launcher, "S08_DEPLOYMENT_CONTRACT", s08_contract
                ),
                mock.patch.object(launcher, "acquire_input_locks", return_value=[]),
                mock.patch.object(launcher, "assert_inputs_unchanged"),
                mock.patch.object(launcher.subprocess, "Popen", ImmediateProcess),
                mock.patch.object(
                    launcher, "audit_completed_outputs", return_value=passing
                ),
            ):
                self.assertEqual(launcher.execute(preflight), 0)
            self.assertEqual(
                [row.command for row in created],
                [("parent-command",), ("s08-command",)],
            )
            decision = json.loads(
                (output / "decision.json").read_text(encoding="utf-8")
            )
            self.assertIs(decision["pass"], True)
            self.assertTrue(parent_contract.is_file())
            self.assertTrue(s08_contract.is_file())


if __name__ == "__main__":
    unittest.main()
