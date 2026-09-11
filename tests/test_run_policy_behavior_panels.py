from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "run_policy_behavior_panels",
    TOOLS_ROOT / "run_policy_behavior_panels.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value
        pieces = value.split(":", 1)
        self.type = pieces[0]
        self.index = int(pieces[1]) if len(pieces) == 2 else None

    def __str__(self) -> str:
        return self.value


class FakeProbe:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def add_(self, value: int) -> FakeProbe:
        self.events.append(f"probe_add:{value}")
        return self

    def item(self) -> float:
        self.events.append("probe_item")
        return 2.0


class FakeCuda:
    def __init__(self, events: list[str], *, fail_init: bool = False) -> None:
        self.events = events
        self.fail_init = fail_init

    def init(self) -> None:
        self.events.append("cuda_init")
        if self.fail_init:
            raise RuntimeError("synthetic CUDA init failure")

    def synchronize(self, device: FakeDevice) -> None:
        self.events.append(f"cuda_sync:{device}")

    def current_device(self) -> int:
        return 0

    def get_device_capability(self, index: int) -> tuple[int, int]:
        self.events.append(f"cuda_capability:{index}")
        return 12, 0

    def get_device_name(self, index: int) -> str:
        self.events.append(f"cuda_name:{index}")
        return "Fake RTX"


class FakeParameter:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = FakeDevice(device)


class FakeCountWeight:
    shape = (61, 8)


class FakeCountLayer:
    out_features = 61


class FakeModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.count_head = [FakeCountLayer()]
        self._parameters = [FakeParameter()]

    def parameters(self) -> list[FakeParameter]:
        return self._parameters

    def buffers(self) -> list[FakeParameter]:
        return []

    def eval(self) -> FakeModel:
        self.events.append("model_eval")
        return self


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PolicyBehaviorPanelsRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "candidate.pt"
        self.checkpoint.write_bytes(b"synthetic checkpoint bytes")
        self.old_data = self.root / "old.zip"
        self.old_data.write_bytes(b"synthetic old panel bytes")
        self.valid_data = self.root / "valid.zip"
        self.valid_data.write_bytes(b"synthetic valid panel bytes")
        self.old_output = self.root / "old-output.json"
        self.valid_output = self.root / "valid-output.json"
        self.preflight_evidence_directory = self.root / "preflight-evidence"
        self.marker = self.root / "formal-attempt.json"
        self.success = self.root / "terminal-success.json"
        self.failure = self.root / "terminal-failure.json"
        self.preregistration = self.root / "behavior-preregistration.json"
        self.source_protocol = self.root / "source-protocol.json"
        self.training_integrity_decision = (
            self.root / "training-integrity-decision.json"
        )
        self.evaluator_path = Path(runner.policy_eval.__file__).resolve()
        self.evaluator_sha256 = runner.file_sha256(self.evaluator_path)
        self.evaluator_dependencies = {
            "train_bc_orbit": {
                "path": str(Path(runner.policy_eval.bc.__file__).resolve()),
                "sha256": runner.file_sha256(
                    Path(runner.policy_eval.bc.__file__).resolve()
                ),
            },
            "train_ppo": {
                "path": str(Path(runner.policy_eval.ppo.__file__).resolve()),
                "sha256": runner.file_sha256(
                    Path(runner.policy_eval.ppo.__file__).resolve()
                ),
            },
        }
        self.runner_path = Path(runner.__file__).resolve()
        self.runner_sha256 = runner.file_sha256(self.runner_path)
        self._write_authoritative_sources()
        self._write_preregistration()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _panel_command(
        self,
        data: Path,
        output: Path,
    ) -> list[str]:
        return [
            sys.executable,
            str(self.evaluator_path),
            "--checkpoint",
            str(self.checkpoint),
            "--data",
            str(data),
            "--split",
            "valid",
            "--split-mode",
            "archive",
            "--split-seed",
            "20260723",
            "--batch-size",
            "256",
            "--workers",
            "8",
            "--prediction-order",
            "policy",
            "--device",
            "cuda",
            "--json-output",
            str(output),
            "--compact",
            "--progress-interval",
            "0",
        ]

    def _source_panel(
        self,
        *,
        order: int,
        data_panel: str,
        data: Path,
        data_sha256: str,
        output: Path,
    ) -> dict[str, object]:
        return {
            "order": order,
            "name": f"{data_panel}-policy",
            "data_panel": data_panel,
            "data": str(data),
            "data_sha256": data_sha256,
            "split": "valid",
            "split_mode": "archive",
            "split_seed": 20260723,
            "output": str(output),
            "rows_exact": 7,
            "context34_rows_exact": 0,
            "minimum_correct": {
                "set_exact": 0,
                "hybrid_order_exact": 0,
                "ordered_exact": 0,
                "value": 0,
                "count": 0,
                "top1": 0,
                "context34_hybrid_order_exact": 0,
                "context34_ordered_exact": 0,
            },
        }

    def _write_authoritative_sources(self) -> None:
        self.source_panels = [
            self._source_panel(
                order=1,
                data_panel="old",
                data=self.old_data,
                data_sha256=sha256(self.old_data),
                output=self.old_output,
            ),
            self._source_panel(
                order=2,
                data_panel="valid",
                data=self.valid_data,
                data_sha256=sha256(self.valid_data),
                output=self.valid_output,
            ),
        ]
        source = {
            "schema_version": "test-comprehensive-v1",
            "status": "locked_before_training",
            "candidate_binding": {
                "terminal_checkpoint": str(self.checkpoint),
            },
            "behavior_protocol": {
                "implementation": {
                    "python": runner.bound_path_text(
                        Path(sys.executable).resolve()
                    ),
                    "inprocess_runner": runner.bound_path_text(
                        self.runner_path
                    ),
                    "inprocess_runner_sha256": self.runner_sha256,
                    "metric_evaluator": runner.bound_path_text(
                        self.evaluator_path
                    ),
                    "metric_evaluator_sha256": self.evaluator_sha256,
                    "local_dependencies": {
                        name: {
                            "path": runner.bound_path_text(
                                Path(str(binding["path"]))
                            ),
                            "sha256": binding["sha256"],
                        }
                        for name, binding in self.evaluator_dependencies.items()
                    },
                    "device": "cuda",
                    "batch_size": 256,
                    "workers": 8,
                    "effective_workers": 0,
                    "prediction_order_cli_argument": "policy",
                    "expected_output_prediction_order": "policy_greedy",
                    "compact": True,
                    "progress_interval": 0,
                    "same_process_two_panel_runner_required": True,
                    "runner_path_and_sha256_bind_after_implementation_before_any_behavior_data_access": True,
                },
                "cuda_preflight_and_formal_attempt_boundary": {
                    "same_pid_and_cuda_context_for_preflight_and_both_panels": True,
                    "preflight_requires_cuda_init_tensor_kernel_synchronize_and_exact_checkpoint_model_load": True,
                    "preflight_must_not_open_behavior_archives_or_process_behavior_rows": True,
                    "preflight_certificate_o_excl_and_fsync": True,
                    "formal_attempt_marker_o_excl_and_fsync_after_certificate": True,
                    "model_loaded_once_and_reused_for_both_panels": True,
                    "no_exec_shell_or_subprocess_after_preflight": True,
                },
                "ordered_panels": self.source_panels,
                "rules": {
                    "all_20_gates_required": True,
                    "run_valid29_even_if_old_retention_gate_fails": True,
                    "run_panels_once_in_declared_order_after_formal_marker": True,
                    "no_alternate_decode": True,
                },
            },
        }
        self.source_protocol.write_text(
            json.dumps(source, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        decision = {
            "schema_version": "test-training-integrity-decision-v1",
            "status": "passed_and_behavior_authorized",
            "pass": True,
            "authoritative_source_protocol": {
                "path": str(self.source_protocol),
                "sha256": sha256(self.source_protocol),
            },
            "training_content_integrity": {
                "pass": True,
                "terminal_checkpoint": {
                    "path": str(self.checkpoint),
                    "sha256": sha256(self.checkpoint),
                    "update": 456,
                },
                "independent_read_only_recomputations": {
                    "count": 2,
                    "numeric_content_pass_count": 2,
                },
            },
            "transport_receipt_integrity": {"pass": True},
            "decision": {
                "behavior_execution_authorized": True,
                "gold19_authorized": False,
            },
        }
        self.training_integrity_decision.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_preregistration(self) -> None:
        payload = {
            "schema_version": runner.EXECUTION_PREREGISTRATION_SCHEMA,
            "status": "locked_before_behavior_evaluation",
            "source_protocol": {
                "path": str(self.source_protocol),
                "sha256": sha256(self.source_protocol),
            },
            "authorization": {
                "training_integrity_decision": {
                    "path": str(self.training_integrity_decision),
                    "sha256": sha256(self.training_integrity_decision),
                    "status": "passed_and_behavior_authorized",
                    "all_required_pass": True,
                    "behavior_execution_authorized": True,
                }
            },
            "candidate": {
                "checkpoint": str(self.checkpoint),
                "checkpoint_sha256": sha256(self.checkpoint),
                "checkpoint_update": 456,
            },
            "evaluator": {
                "path": str(self.evaluator_path),
                "sha256": self.evaluator_sha256,
                "local_dependencies": self.evaluator_dependencies,
                "python": sys.executable,
                "device": "cuda",
                "batch_size": 256,
                "workers": 8,
                "effective_workers": 0,
                "prediction_order_cli_argument": "policy",
                "expected_output_prediction_order": "policy_greedy",
                "progress_interval": 0,
                "compact": True,
            },
            "data": {
                "old": {
                    "path": str(self.old_data),
                    "sha256": sha256(self.old_data),
                    "split": "valid",
                    "split_mode": "archive",
                    "split_seed": 20260723,
                },
                "valid": {
                    "path": str(self.valid_data),
                    "sha256": sha256(self.valid_data),
                    "split": "valid",
                    "split_mode": "archive",
                    "split_seed": 20260723,
                },
            },
            "ordered_evaluations": [
                {
                    "order": 1,
                    "name": "old-policy",
                    "data_panel": "old",
                    "output": str(self.old_output),
                    "output_absent_at_lock": True,
                    "command": self._panel_command(
                        self.old_data,
                        self.old_output,
                    ),
                },
                {
                    "order": 2,
                    "name": "valid-policy",
                    "data_panel": "valid",
                    "output": str(self.valid_output),
                    "output_absent_at_lock": True,
                    "command": self._panel_command(
                        self.valid_data,
                        self.valid_output,
                    ),
                },
            ],
            "metric_mapping_and_gates": runner.expected_gate_mapping(
                self.source_panels
            ),
            "inprocess_runner": {
                "path": str(self.runner_path),
                "sha256": self.runner_sha256,
                "preflight_evidence_directory": str(
                    self.preflight_evidence_directory
                ),
                "formal_attempt_marker": str(self.marker),
                "success_result": str(self.success),
                "failure_result": str(self.failure),
            },
        }
        self.preregistration.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.preregistration_sha256 = sha256(self.preregistration)

    def _rebind_authority_chain(
        self,
        *,
        source: dict[str, object] | None = None,
        decision: dict[str, object] | None = None,
        preregistration: dict[str, object] | None = None,
    ) -> None:
        source_value = source or json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        self.source_protocol.write_text(
            json.dumps(source_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        decision_value = decision or json.loads(
            self.training_integrity_decision.read_text(encoding="utf-8")
        )
        decision_value["authoritative_source_protocol"] = {
            "path": str(self.source_protocol),
            "sha256": sha256(self.source_protocol),
        }
        self.training_integrity_decision.write_text(
            json.dumps(decision_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preregistration_value = preregistration or json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration_value["source_protocol"]["sha256"] = sha256(
            self.source_protocol
        )
        preregistration_value["authorization"][
            "training_integrity_decision"
        ]["sha256"] = sha256(self.training_integrity_decision)
        self.preregistration.write_text(
            json.dumps(
                preregistration_value,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.preregistration_sha256 = sha256(self.preregistration)

    def _fake_torch(
        self,
        events: list[str],
        *,
        fail_init: bool = False,
    ) -> types.SimpleNamespace:
        cuda = FakeCuda(events, fail_init=fail_init)

        def set_precision(value: str) -> None:
            events.append(f"precision:{value}")

        def ones(size: int, *, device: FakeDevice) -> FakeProbe:
            events.append(f"ones:{size}:{device}")
            return FakeProbe(events)

        return types.SimpleNamespace(
            __version__="test-torch",
            version=types.SimpleNamespace(cuda="test-cuda"),
            cuda=cuda,
            device=lambda value: FakeDevice(str(value)),
            ones=ones,
            set_float32_matmul_precision=set_precision,
        )

    def _fake_evaluator(
        self,
        events: list[str],
        *,
        fail_panel: str | None = None,
        replace_data_on_dataset: str | None = None,
        replace_checkpoint_on_load: bool = False,
    ) -> types.SimpleNamespace:
        model = FakeModel(events)
        checkpoint_payload = {
            "feature_version": "ptcg-selfplay-ppo-terminal01-v1",
            "update": 456,
            "model_state_dict": {
                "count_head.2.weight": FakeCountWeight(),
            },
        }
        model_config = {
            "hash_size": 64,
            "max_state_entities": 16,
            "entity_fields": 4,
            "option_fields": 3,
        }

        def load_policy(
            checkpoint: Path,
            device: FakeDevice,
        ) -> tuple[FakeModel, dict[str, int], dict[str, object], str]:
            self.assertEqual(checkpoint.parent, Path("/proc/self/fd"))
            if replace_checkpoint_on_load:
                replacement = self.root / "replacement-checkpoint.pt"
                replacement.write_bytes(b"replacement checkpoint bytes")
                replacement.replace(self.checkpoint)
            self.assertEqual(
                checkpoint.read_bytes(),
                b"synthetic checkpoint bytes",
            )
            events.append(f"load_policy:{self.checkpoint.name}:{device}")
            return model, model_config, checkpoint_payload, "ppo"

        def dataset_factory(**kwargs: object) -> dict[str, object]:
            data_path = Path(str(kwargs["archive_path"]))
            self.assertEqual(data_path.parent, Path("/proc/self/fd"))
            original_payload = data_path.read_bytes()
            if original_payload == b"synthetic old panel bytes":
                source_name = self.old_data.name
                source_path = self.old_data
            elif original_payload == b"synthetic valid panel bytes":
                source_name = self.valid_data.name
                source_path = self.valid_data
            else:
                self.fail(f"unexpected held dataset payload: {original_payload!r}")
            if replace_data_on_dataset == source_name:
                replacement = self.root / f"replacement-{source_name}"
                replacement.write_bytes(b"replacement panel bytes")
                replacement.replace(source_path)
            self.assertEqual(data_path.read_bytes(), original_payload)
            events.append(f"dataset:{source_name}")
            result = dict(kwargs)
            result["source_name"] = source_name
            result["source_payload"] = original_payload
            return result

        def data_loader(
            dataset: dict[str, object],
            **kwargs: object,
        ) -> dict[str, object]:
            events.append(f"loader:{dataset['source_name']}")
            events.append(f"loader_workers:{kwargs['num_workers']}")
            return {"dataset": dataset, "loader_kwargs": kwargs}

        def evaluate(
            model_value: FakeModel,
            loader: dict[str, object],
            device: FakeDevice,
            **kwargs: object,
        ) -> tuple[dict[str, object], float]:
            self.assertIs(model_value, model)
            source_name = str(
                loader["dataset"]["source_name"]  # type: ignore[index]
            )
            events.append(f"evaluate:{source_name}:{device}")
            if fail_panel == source_name:
                raise RuntimeError(f"synthetic failure for {source_name}")
            return (
                {
                    "rows": 7,
                    "set_exact_correct": 6,
                    "by_context": {},
                },
                0.5,
            )

        return types.SimpleNamespace(
            __file__=str(self.evaluator_path),
            bc=types.SimpleNamespace(
                __file__=self.evaluator_dependencies["train_bc_orbit"]["path"]
            ),
            ppo=types.SimpleNamespace(
                __file__=self.evaluator_dependencies["train_ppo"]["path"]
            ),
            load_policy=load_policy,
            OrderedZipDecisionDataset=dataset_factory,
            DataLoader=data_loader,
            collate_ordered=lambda *args, **kwargs: None,
            evaluate=evaluate,
        )

    def test_authoritative_decision_must_explicitly_authorize_behavior(
        self,
    ) -> None:
        decision = json.loads(
            self.training_integrity_decision.read_text(encoding="utf-8")
        )
        decision["pass"] = False
        decision["decision"]["behavior_execution_authorized"] = False
        self.training_integrity_decision.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["authorization"]["training_integrity_decision"][
            "sha256"
        ] = sha256(self.training_integrity_decision)
        self.preregistration.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "decision did not pass",
        ):
            runner.execute(
                self.preregistration,
                sha256(self.preregistration),
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.old_output.exists())
        self.assertFalse(self.valid_output.exists())

    def test_decision_must_authorize_the_exact_source_protocol(self) -> None:
        decision = json.loads(
            self.training_integrity_decision.read_text(encoding="utf-8")
        )
        decision["authoritative_source_protocol"]["sha256"] = "0" * 64
        self.training_integrity_decision.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["authorization"]["training_integrity_decision"][
            "sha256"
        ] = sha256(self.training_integrity_decision)
        self.preregistration.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "authorizes a different source protocol",
        ):
            runner.execute(
                self.preregistration,
                sha256(self.preregistration),
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())

    def test_source_runtime_and_rule_drift_is_rejected(self) -> None:
        source = json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        source["behavior_protocol"]["rules"]["no_alternate_decode"] = False
        self._rebind_authority_chain(source=source)
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "does not prohibit alternate decode",
        ):
            runner.parse_execution_spec(
                self.preregistration,
                self.preregistration_sha256,
            )

    def test_missing_checkpoint_update_and_reserved_panel_name_rejected(
        self,
    ) -> None:
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["candidate"].pop("checkpoint_update")
        self.preregistration.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "checkpoint_update is required",
        ):
            runner.parse_execution_spec(
                self.preregistration,
                sha256(self.preregistration),
            )
        reserved_panel = dict(self.source_panels[0])
        reserved_panel["data_panel"] = "all_required"
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "invalid data_panel names",
        ):
            runner.expected_gate_mapping(
                [reserved_panel, self.source_panels[1]]
            )

    def test_integer_parser_rejects_boolean_and_fractional_values(self) -> None:
        for value in (True, False, 1.0, 1.5, "1.5"):
            with self.subTest(value=value):
                with self.assertRaises(runner.ProtocolError):
                    runner.parse_positive_int(
                        value,
                        "probe",
                        allow_zero=True,
                    )

    def test_metric_gate_drift_is_rejected_before_cuda_or_data(self) -> None:
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["metric_mapping_and_gates"]["old"]["set_exact"][
            "minimum"
        ] = 1
        self.preregistration.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "metric mapping or thresholds differ",
        ):
            runner.execute(
                self.preregistration,
                sha256(self.preregistration),
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())

    def test_formal_filters_and_max_rows_are_forbidden(self) -> None:
        for extra_tokens, expected_error in (
            (["--max-rows", "1"], "--max-rows is forbidden"),
            (["--deck-hash", "abc"], "filters are forbidden"),
            (["--team-name", "name"], "filters are forbidden"),
        ):
            with self.subTest(extra_tokens=extra_tokens):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = root / "output.json"
                    command = self._panel_command(self.old_data, output)
                    command.extend(extra_tokens)
                    payload = json.loads(
                        self.preregistration.read_text(encoding="utf-8")
                    )
                    payload["ordered_evaluations"][0]["command"] = command
                    payload["ordered_evaluations"][0]["output"] = str(output)
                    payload["ordered_evaluations"][0]["command"][
                        payload["ordered_evaluations"][0]["command"].index(
                            "--json-output"
                        )
                        + 1
                    ] = str(output)
                    probe = root / "preregistration.json"
                    probe.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        runner.ProtocolError,
                        expected_error,
                    ):
                        runner.parse_execution_spec(probe, sha256(probe))

    def test_frozen_evaluator_settings_reject_drift(self) -> None:
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["evaluator"]["batch_size"] = 128
        for panel in preregistration["ordered_evaluations"]:
            index = panel["command"].index("--batch-size") + 1
            panel["command"][index] = "128"
        self.preregistration.write_text(
            json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "batch_size differs from the frozen behavior setting",
        ):
            runner.parse_execution_spec(
                self.preregistration,
                sha256(self.preregistration),
            )

    def test_preflight_failure_has_no_marker_dataset_or_output(self) -> None:
        events: list[str] = []
        fake_evaluator = self._fake_evaluator(events)
        fake_torch = self._fake_torch(events, fail_init=True)
        real_verify_file_hash = runner.verify_file_hash

        def recording_verify_file_hash(
            path: Path,
            expected: str,
            label: str,
        ) -> None:
            events.append(f"hash:{label}")
            real_verify_file_hash(path, expected, label)

        with self.assertRaisesRegex(
            RuntimeError,
            "synthetic CUDA init failure",
        ):
            with mock.patch.object(
                runner,
                "verify_file_hash",
                side_effect=recording_verify_file_hash,
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=fake_evaluator,
                    torch_module=fake_torch,
                )

        self.assertFalse(self.marker.exists())
        self.assertFalse(self.success.exists())
        self.assertFalse(self.failure.exists())
        self.assertFalse(self.old_output.exists())
        self.assertFalse(self.valid_output.exists())
        self.assertFalse(any(event.startswith("dataset:") for event in events))
        self.assertFalse(any(event.startswith("load_policy:") for event in events))
        self.assertFalse(any(event.endswith(" data") for event in events))
        failures = list(
            self.preflight_evidence_directory.glob(
                "infrastructure-failure.*.json"
            )
        )
        self.assertEqual(len(failures), 1)

    def test_runner_hash_mismatch_prevents_cuda_model_or_data(self) -> None:
        source = json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        source["behavior_protocol"]["implementation"][
            "inprocess_runner_sha256"
        ] = "0" * 64
        self.source_protocol.write_text(
            json.dumps(source, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        decision = json.loads(
            self.training_integrity_decision.read_text(encoding="utf-8")
        )
        decision["authoritative_source_protocol"]["sha256"] = sha256(
            self.source_protocol
        )
        self.training_integrity_decision.write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload = json.loads(self.preregistration.read_text(encoding="utf-8"))
        payload["inprocess_runner"]["sha256"] = "0" * 64
        payload["source_protocol"]["sha256"] = sha256(self.source_protocol)
        payload["authorization"]["training_integrity_decision"][
            "sha256"
        ] = sha256(self.training_integrity_decision)
        self.preregistration.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preregistration_sha256 = sha256(self.preregistration)
        events: list[str] = []

        with self.assertRaisesRegex(
            runner.ProtocolError,
            "in-process behavior runner SHA256 mismatch",
        ):
            runner.execute(
                self.preregistration,
                preregistration_sha256,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )

        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.success.exists())
        self.assertFalse(self.failure.exists())
        self.assertFalse(self.old_output.exists())
        self.assertFalse(self.valid_output.exists())
        failures = list(
            self.preflight_evidence_directory.glob(
                "infrastructure-failure.*.json"
            )
        )
        self.assertEqual(len(failures), 1)

    def test_dependency_hash_mismatch_prevents_cuda_or_data(self) -> None:
        source = json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        source["behavior_protocol"]["implementation"]["local_dependencies"][
            "train_ppo"
        ]["sha256"] = "0" * 64
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["evaluator"]["local_dependencies"]["train_ppo"][
            "sha256"
        ] = "0" * 64
        self._rebind_authority_chain(
            source=source,
            preregistration=preregistration,
        )
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "evaluator dependency train_ppo SHA256 mismatch",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.old_output.exists())

    def test_preflight_failure_evidence_survives_same_prereg_retry(self) -> None:
        first_events: list[str] = []
        with self.assertRaisesRegex(
            RuntimeError,
            "synthetic CUDA init failure",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(first_events),
                torch_module=self._fake_torch(first_events, fail_init=True),
            )

        failures = list(
            self.preflight_evidence_directory.glob(
                "infrastructure-failure.*.json"
            )
        )
        self.assertEqual(len(failures), 1)
        original_failure = failures[0]
        original_failure_sha256 = sha256(original_failure)

        retry_events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(retry_events),
            torch_module=self._fake_torch(retry_events),
        )

        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertTrue(original_failure.is_file())
        self.assertEqual(sha256(original_failure), original_failure_sha256)
        self.assertEqual(
            len(
                list(
                    self.preflight_evidence_directory.glob(
                        "infrastructure-failure.*.json"
                    )
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                list(
                    self.preflight_evidence_directory.glob(
                        "certificate.*.json"
                    )
                )
            ),
            1,
        )
        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.success.is_file())
        self.assertTrue(self.old_output.is_file())
        self.assertTrue(self.valid_output.is_file())

    def test_success_loads_once_and_runs_both_panels_in_order(self) -> None:
        events: list[str] = []
        fake_evaluator = self._fake_evaluator(events)
        fake_torch = self._fake_torch(events)
        real_atomic_write = runner.atomic_write_json_exclusive
        real_verify_file_hash = runner.verify_file_hash
        real_open_verified_held_file = runner.open_verified_held_file

        def recording_write(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> str:
            events.append(f"write:{path.name}")
            return real_atomic_write(path, payload, compact=compact)

        def recording_verify_file_hash(
            path: Path,
            expected: str,
            label: str,
        ) -> None:
            events.append(f"hash:{label}")
            real_verify_file_hash(path, expected, label)

        def recording_open_verified_held_file(
            path: Path,
            expected: str,
            label: str,
        ) -> runner.HeldFile:
            events.append(f"hold:{label}")
            return real_open_verified_held_file(path, expected, label)

        with mock.patch.object(
            runner,
            "atomic_write_json_exclusive",
            side_effect=recording_write,
        ), mock.patch.object(
            runner,
            "verify_file_hash",
            side_effect=recording_verify_file_hash,
        ), mock.patch.object(
            runner,
            "open_verified_held_file",
            side_effect=recording_open_verified_held_file,
        ):
            completed = runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=fake_evaluator,
                torch_module=fake_torch,
            )

        self.assertEqual(
            sum(event.startswith("load_policy:") for event in events),
            1,
        )
        self.assertLess(
            events.index(f"load_policy:{self.checkpoint.name}:cuda"),
            events.index("hold:old data"),
        )
        certificate_events = [
            event for event in events if event.startswith("write:certificate.")
        ]
        self.assertEqual(len(certificate_events), 1)
        certificate_event = certificate_events[0]
        self.assertLess(events.index(certificate_event), events.index(f"write:{self.marker.name}"))
        self.assertLess(
            events.index(f"write:{self.marker.name}"),
            events.index("hold:old data"),
        )
        self.assertLess(
            events.index(f"evaluate:{self.old_data.name}:cuda"),
            events.index("hold:valid data"),
        )
        self.assertLess(
            events.index("hold:valid data"),
            events.index(f"dataset:{self.valid_data.name}"),
        )
        self.assertLess(
            events.index(f"evaluate:{self.old_data.name}:cuda"),
            events.index(f"dataset:{self.valid_data.name}"),
        )
        self.assertEqual(
            [item["order"] for item in completed],
            [1, 2],
        )
        self.assertEqual(events.count("loader_workers:0"), 2)
        self.assertTrue(self.old_output.is_file())
        self.assertTrue(self.valid_output.is_file())
        self.assertTrue(self.success.is_file())
        self.assertFalse(self.failure.exists())

        certificates = list(
            self.preflight_evidence_directory.glob("certificate.*.json")
        )
        self.assertEqual(len(certificates), 1)
        certificate_path = certificates[0]
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        marker = json.loads(self.marker.read_text(encoding="utf-8"))
        self.assertFalse(certificate["formal_attempt_started"])
        self.assertEqual(certificate["raw_data_files_read_for_sha256"], 0)
        self.assertEqual(certificate["raw_data_files_sha256_verified"], 0)
        self.assertEqual(certificate["zip_archive_members_opened"], 0)
        self.assertEqual(certificate["dataset_objects_constructed"], 0)
        self.assertEqual(certificate["behavior_rows_evaluated"], 0)
        self.assertEqual(
            certificate["process"]["pid"],
            marker["process"]["pid"],
        )
        self.assertEqual(
            certificate["process"]["start_ticks"],
            marker["process"]["start_ticks"],
        )
        self.assertEqual(
            marker["preflight_certificate"]["sha256"],
            sha256(certificate_path),
        )
        self.assertLessEqual(
            certificate_path.stat().st_mtime_ns,
            self.marker.stat().st_mtime_ns,
        )

        old_result = json.loads(self.old_output.read_text(encoding="utf-8"))
        self.assertEqual(old_result["checkpoint_sha256"], sha256(self.checkpoint))
        self.assertEqual(old_result["prediction_order"], "policy_greedy")
        self.assertEqual(old_result["metrics"]["rows"], 7)
        self.assertEqual(old_result["checkpoint_count_classes"], 61)
        self.assertEqual(old_result["inference_count_classes"], 61)
        self.assertEqual(
            old_result["workers"]["requested_by_evaluator_command"],
            8,
        )
        self.assertEqual(
            old_result["workers"]["effective_inprocess_dataloader"],
            0,
        )
        self.assertEqual(
            old_result["data_held_file"]["original_path"],
            str(self.old_data),
        )
        self.assertTrue(
            old_result["data_held_file"]["pre_post_sha256_match"]
        )
        self.assertTrue(
            old_result["checkpoint_held_file"]["pre_post_sha256_match"]
        )
        self.assertEqual(
            len(self.old_output.read_text(encoding="utf-8").splitlines()),
            1,
        )

    def test_marker_publish_error_writes_terminal_failure(self) -> None:
        events: list[str] = []
        fake_evaluator = self._fake_evaluator(events)
        fake_torch = self._fake_torch(events)
        real_atomic_write = runner.atomic_write_json_exclusive

        def write_then_fail_for_marker(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> str:
            digest = real_atomic_write(path, payload, compact=compact)
            if path == self.marker:
                raise OSError("synthetic post-publish marker fsync failure")
            return digest

        with self.assertRaisesRegex(
            OSError,
            "synthetic post-publish marker fsync failure",
        ):
            with mock.patch.object(
                runner,
                "atomic_write_json_exclusive",
                side_effect=write_then_fail_for_marker,
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=fake_evaluator,
                    torch_module=fake_torch,
                )

        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.failure.is_file())
        self.assertFalse(self.success.exists())
        self.assertFalse(self.old_output.exists())
        self.assertFalse(self.valid_output.exists())
        self.assertFalse(any(event.startswith("dataset:") for event in events))
        failure = json.loads(self.failure.read_text(encoding="utf-8"))
        self.assertEqual(
            failure["status"],
            "terminal_failure_after_formal_attempt_marker",
        )
        self.assertIsNone(failure["current_panel"])
        self.assertEqual(failure["completed_panels"], [])
        self.assertEqual(
            failure["formal_attempt_marker"]["sha256"],
            sha256(self.marker),
        )

    def test_foreign_marker_is_not_claimed_or_given_shared_failure(self) -> None:
        events: list[str] = []
        real_atomic_write = runner.atomic_write_json_exclusive

        def inject_foreign_marker(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> str:
            if path == self.marker:
                path.write_bytes(b"foreign marker bytes")
            return real_atomic_write(path, payload, compact=compact)

        with self.assertRaises(runner.RestartRefusedError):
            with mock.patch.object(
                runner,
                "atomic_write_json_exclusive",
                side_effect=inject_foreign_marker,
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=self._fake_evaluator(events),
                    torch_module=self._fake_torch(events),
                )
        self.assertEqual(self.marker.read_bytes(), b"foreign marker bytes")
        self.assertFalse(self.failure.exists())
        self.assertFalse(self.success.exists())
        self.assertFalse(self.old_output.exists())

    def test_panel_output_replacement_is_detected_before_success(self) -> None:
        events: list[str] = []
        real_atomic_write = runner.atomic_write_json_exclusive

        def replace_old_output_after_write(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> str:
            digest = real_atomic_write(path, payload, compact=compact)
            if path == self.old_output:
                replacement = self.root / "replacement-output.json"
                replacement.write_bytes(b"replacement output bytes")
                replacement.replace(path)
            return digest

        with self.assertRaisesRegex(
            runner.ProtocolError,
            "persisted behavior output SHA256 mismatch",
        ):
            with mock.patch.object(
                runner,
                "atomic_write_json_exclusive",
                side_effect=replace_old_output_after_write,
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=self._fake_evaluator(events),
                    torch_module=self._fake_torch(events),
                )
        self.assertTrue(self.marker.exists())
        self.assertTrue(self.failure.exists())
        self.assertFalse(self.success.exists())
        self.assertFalse(self.valid_output.exists())

    def test_held_data_inode_survives_source_path_replacement(self) -> None:
        original_sha256 = sha256(self.old_data)
        events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(
                events,
                replace_data_on_dataset=self.old_data.name,
            ),
            torch_module=self._fake_torch(events),
        )

        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertEqual(self.old_data.read_bytes(), b"replacement panel bytes")
        self.assertIn(f"evaluate:{self.old_data.name}:cuda", events)
        result = json.loads(self.old_output.read_text(encoding="utf-8"))
        evidence = result["data_held_file"]
        self.assertEqual(evidence["expected_sha256"], original_sha256)
        self.assertEqual(evidence["pre_sha256"], original_sha256)
        self.assertEqual(evidence["post_sha256"], original_sha256)
        self.assertTrue(evidence["pre_post_sha256_match"])

    def test_held_checkpoint_inode_survives_source_path_replacement(self) -> None:
        original_sha256 = sha256(self.checkpoint)
        events: list[str] = []
        runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(
                events,
                replace_checkpoint_on_load=True,
            ),
            torch_module=self._fake_torch(events),
        )

        self.assertEqual(
            self.checkpoint.read_bytes(),
            b"replacement checkpoint bytes",
        )
        certificates = list(
            self.preflight_evidence_directory.glob("certificate.*.json")
        )
        self.assertEqual(len(certificates), 1)
        certificate = json.loads(
            certificates[0].read_text(encoding="utf-8")
        )
        evidence = certificate["checkpoint"]["held_file"]
        self.assertEqual(evidence["expected_sha256"], original_sha256)
        self.assertEqual(evidence["pre_sha256"], original_sha256)
        self.assertEqual(evidence["post_sha256"], original_sha256)
        self.assertTrue(evidence["pre_post_sha256_match"])

    def test_trusted_raw_preregistration_parse_failure_is_append_only(self) -> None:
        self.preregistration.write_bytes(b"{")
        trusted_sha256 = sha256(self.preregistration)
        events: list[str] = []

        for expected_count in (1, 2):
            with self.assertRaisesRegex(
                runner.ProtocolError,
                "preregistration is not valid JSON",
            ):
                runner.execute(
                    self.preregistration,
                    trusted_sha256,
                    evaluator_module=self._fake_evaluator(events),
                    torch_module=self._fake_torch(events),
                )
            evidence_directory = self.root / (
                f".{self.preregistration.name}.preflight-parse-evidence"
            )
            evidence_paths = sorted(
                evidence_directory.glob(
                    "preregistration-parse-failure.*.json"
                )
            )
            self.assertEqual(len(evidence_paths), expected_count)
            evidence = json.loads(
                evidence_paths[-1].read_text(encoding="utf-8")
            )
            self.assertTrue(
                evidence["preregistration"]["raw_bytes_sha256_verified"]
            )
            self.assertFalse(evidence["formal_attempt_consumed"])

        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())

    def test_preregistration_hash_mismatch_writes_no_evidence(self) -> None:
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "preregistration SHA256 mismatch",
        ):
            runner.execute(
                self.preregistration,
                "0" * 64,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )
        evidence_directory = self.root / (
            f".{self.preregistration.name}.preflight-parse-evidence"
        )
        self.assertFalse(evidence_directory.exists())
        self.assertEqual(events, [])

    def test_concurrent_execution_lock_refuses_second_instance(self) -> None:
        descriptor = runner.os.open(
            self.preregistration,
            runner.os.O_RDONLY,
        )
        try:
            runner.fcntl.flock(
                descriptor,
                runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB,
            )
            events: list[str] = []
            with self.assertRaisesRegex(
                runner.ConcurrentExecutionError,
                "another process owns",
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=self._fake_evaluator(events),
                    torch_module=self._fake_torch(events),
                )
            self.assertEqual(events, [])
            self.assertFalse(self.marker.exists())
            self.assertFalse(self.success.exists())
            self.assertFalse(self.failure.exists())
        finally:
            runner.fcntl.flock(descriptor, runner.fcntl.LOCK_UN)
            runner.os.close(descriptor)

    def test_restart_refusal_precedes_changed_external_bindings(self) -> None:
        events: list[str] = []
        runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(events),
            torch_module=self._fake_torch(events),
        )
        self.training_integrity_decision.write_bytes(b"changed after attempt")
        evidence_directory = self.root / (
            f".{self.preregistration.name}.preflight-parse-evidence"
        )

        restart_events: list[str] = []
        with self.assertRaises(runner.RestartRefusedError):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(restart_events),
                torch_module=self._fake_torch(restart_events),
            )

        self.assertEqual(restart_events, [])
        self.assertFalse(evidence_directory.exists())

    def test_direct_target_o_excl_refuses_overwrite(self) -> None:
        target = self.root / "exclusive.json"
        target.write_bytes(b"original bytes")
        real_open = runner.os.open
        observed_flags: list[int] = []

        def recording_open(
            path: object,
            flags: int,
            mode: int = 0o777,
        ) -> int:
            if Path(path) == target:
                observed_flags.append(flags)
            return real_open(path, flags, mode)

        with mock.patch.object(runner.os, "open", side_effect=recording_open):
            with self.assertRaises(runner.RestartRefusedError):
                runner.atomic_write_json_exclusive(target, {"new": True})

        self.assertEqual(target.read_bytes(), b"original bytes")
        self.assertEqual(len(observed_flags), 1)
        self.assertTrue(observed_flags[0] & runner.os.O_CREAT)
        self.assertTrue(observed_flags[0] & runner.os.O_EXCL)

    def test_post_marker_failure_is_terminal_and_restart_is_refused(self) -> None:
        events: list[str] = []
        fake_evaluator = self._fake_evaluator(
            events,
            fail_panel=self.valid_data.name,
        )
        fake_torch = self._fake_torch(events)

        with self.assertRaisesRegex(
            RuntimeError,
            f"synthetic failure for {self.valid_data.name}",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=fake_evaluator,
                torch_module=fake_torch,
            )

        self.assertEqual(
            len(
                list(
                    self.preflight_evidence_directory.glob(
                        "certificate.*.json"
                    )
                )
            ),
            1,
        )
        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.old_output.is_file())
        self.assertFalse(self.valid_output.exists())
        self.assertFalse(self.success.exists())
        self.assertTrue(self.failure.is_file())
        failure = json.loads(self.failure.read_text(encoding="utf-8"))
        self.assertEqual(
            failure["status"],
            "terminal_failure_after_formal_attempt_marker",
        )
        self.assertEqual(failure["current_panel"]["order"], 2)
        self.assertEqual(len(failure["completed_panels"]), 1)
        self.assertTrue(failure["no_retry"])

        restart_events: list[str] = []
        with self.assertRaises(runner.RestartRefusedError):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(restart_events),
                torch_module=self._fake_torch(restart_events),
            )
        self.assertEqual(restart_events, [])


if __name__ == "__main__":
    unittest.main()
