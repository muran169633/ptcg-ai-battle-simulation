from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "run_policy_behavior_panels_frozen_candidate_20260731",
    TOOLS_ROOT / "run_policy_behavior_panels_frozen_candidate_20260731.py",
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
        self.temporary = tempfile.TemporaryDirectory(
            dir=runner.REPO_ROOT,
            prefix=".frozen-runner-test-",
        )
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
            "cg/__init__.py": {
                "path": str(
                    Path(sys.modules["cg"].__file__).resolve()
                ),
                "sha256": runner.file_sha256(
                    Path(sys.modules["cg"].__file__).resolve()
                ),
            },
            "cg/sim.py": {
                "path": str(
                    Path(sys.modules["cg.sim"].__file__).resolve()
                ),
                "sha256": runner.file_sha256(
                    Path(sys.modules["cg.sim"].__file__).resolve()
                ),
            },
            "cg/libcg.so": {
                "path": str(
                    Path(sys.modules["cg.sim"].lib._name).resolve()
                ),
                "sha256": runner.file_sha256(
                    Path(sys.modules["cg.sim"].lib._name).resolve()
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
            "data": runner.bound_path_text(data),
            "data_sha256": data_sha256,
            "split": "valid",
            "split_mode": "archive",
            "split_seed": 20260723,
            "output": runner.bound_path_text(output),
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
        fail_panel_with_base_exception: bool = False,
        replace_data_on_dataset: str | None = None,
        replace_checkpoint_on_load: bool = False,
        invalid_gate_fields: bool = False,
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
                if fail_panel_with_base_exception:
                    raise KeyboardInterrupt(
                        f"synthetic base failure for {source_name}"
                    )
                raise RuntimeError(f"synthetic failure for {source_name}")
            metrics: dict[str, object] = {
                "rows": 7,
                "set_exact_correct": 6,
                "hybrid_order_exact_correct": 5,
                "ordered_exact_correct": 4,
                "value_correct": 3,
                "count_correct": 2,
                "top1_correct": 1,
                "by_context": {
                    "34": {
                        "rows": 0,
                        "hybrid_order_exact_correct": 0,
                        "ordered_exact_correct": 0,
                    }
                },
            }
            if invalid_gate_fields:
                metrics["rows"] = True
                metrics["set_exact_correct"] = True
                del metrics["count_correct"]
            return metrics, 0.5

        return types.SimpleNamespace(
            __file__=str(self.evaluator_path),
            bc=types.SimpleNamespace(
                __file__=self.evaluator_dependencies["train_bc_orbit"]["path"]
            ),
            ppo=types.SimpleNamespace(
                __file__=self.evaluator_dependencies["train_ppo"]["path"],
                lib=sys.modules["cg.sim"].lib,
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
                with tempfile.TemporaryDirectory(
                    dir=runner.REPO_ROOT,
                    prefix=".frozen-runner-filter-test-",
                ) as directory:
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

    def test_execution_and_source_protocol_bind_all_five_dependencies(
        self,
    ) -> None:
        expected_names = {
            "train_bc_orbit",
            "train_ppo",
            "cg/__init__.py",
            "cg/sim.py",
            "cg/libcg.so",
        }
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        source = json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(preregistration["evaluator"]["local_dependencies"]),
            expected_names,
        )
        self.assertEqual(
            set(
                source["behavior_protocol"]["implementation"][
                    "local_dependencies"
                ]
            ),
            expected_names,
        )
        runner.parse_execution_spec(
            self.preregistration,
            self.preregistration_sha256,
        )

        for missing_name in sorted(expected_names):
            with self.subTest(missing_name=missing_name):
                probe_value = json.loads(
                    self.preregistration.read_text(encoding="utf-8")
                )
                del probe_value["evaluator"]["local_dependencies"][
                    missing_name
                ]
                probe = self.root / (
                    "missing-"
                    + missing_name.replace("/", "-").replace(".", "-")
                    + ".json"
                )
                probe.write_text(
                    json.dumps(probe_value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    runner.ProtocolError,
                    "must bind exactly",
                ):
                    runner.parse_execution_spec(probe, sha256(probe))

    def test_cg_dependency_hash_is_verified(self) -> None:
        source = json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        source["behavior_protocol"]["implementation"]["local_dependencies"][
            "cg/sim.py"
        ]["sha256"] = "0" * 64
        preregistration["evaluator"]["local_dependencies"]["cg/sim.py"][
            "sha256"
        ] = "0" * 64
        self._rebind_authority_chain(
            source=source,
            preregistration=preregistration,
        )
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "evaluator dependency cg/sim.py SHA256 mismatch",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())

    def test_loaded_train_ppo_must_share_the_bound_cg_lib_handle(self) -> None:
        events: list[str] = []
        fake_evaluator = self._fake_evaluator(events)
        fake_evaluator.ppo.lib = object()
        with self.assertRaisesRegex(
            runner.ProtocolError,
            "train_ppo is not associated",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=fake_evaluator,
                torch_module=self._fake_torch(events),
            )
        self.assertEqual(events, [])
        self.assertFalse(self.marker.exists())

    def test_dangling_symlink_for_every_execution_artifact_is_restart(
        self,
    ) -> None:
        for target in (
            self.marker,
            self.success,
            self.failure,
            self.old_output,
            self.valid_output,
        ):
            with self.subTest(target=target.name):
                target.symlink_to(self.root / f"missing-{target.name}")
                self.assertTrue(runner.artifact_path_lexists(target))
                self.assertEqual(
                    runner.resolve_bound_artifact_path(target),
                    target,
                )
                events: list[str] = []
                with self.assertRaisesRegex(
                    runner.RestartRefusedError,
                    "existing symlink",
                ):
                    runner.execute(
                        self.preregistration,
                        self.preregistration_sha256,
                        evaluator_module=self._fake_evaluator(events),
                        torch_module=self._fake_torch(events),
                    )
                self.assertEqual(events, [])
                self.assertTrue(runner.artifact_path_lexists(target))
                target.unlink()

    def test_atomic_writer_refuses_dangling_symlink_without_following_it(
        self,
    ) -> None:
        target = self.root / "dangling-output.json"
        missing = self.root / "must-not-be-created.json"
        target.symlink_to(missing)
        with self.assertRaisesRegex(
            runner.RestartRefusedError,
            "existing symlink",
        ):
            runner.atomic_write_json_exclusive(target, {"new": True})
        self.assertTrue(runner.artifact_path_lexists(target))
        self.assertFalse(missing.exists())

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
        real_create_held = runner.create_json_artifact_exclusive_held
        real_verify_file_hash = runner.verify_file_hash
        real_open_verified_held_file = runner.open_verified_held_file

        def recording_create_held(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> runner.HeldArtifact:
            events.append(f"write:{path.name}")
            return real_create_held(path, payload, compact=compact)

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
            "create_json_artifact_exclusive_held",
            side_effect=recording_create_held,
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
        success = json.loads(self.success.read_text(encoding="utf-8"))
        self.assertTrue(success["quality_gates_evaluated"])
        self.assertTrue(success["all_20_gates_passed"])
        self.assertEqual(success["quality_gates"]["gate_count"], 20)
        self.assertEqual(len(success["quality_gates"]["gates"]), 20)
        self.assertFalse(success["gold19_authorized"])

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
        real_verify_held = runner.verify_held_artifact

        def fail_after_marker_publish(
            held: runner.HeldArtifact,
            label: str,
        ) -> str:
            digest = real_verify_held(held, label)
            if label == "formal attempt marker after persistence":
                raise OSError("synthetic post-publish marker fsync failure")
            return digest

        with self.assertRaisesRegex(
            OSError,
            "synthetic post-publish marker fsync failure",
        ):
            with mock.patch.object(
                runner,
                "verify_held_artifact",
                side_effect=fail_after_marker_publish,
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
        self.assertEqual(
            failure["failure_class"],
            "execution_integrity_failure",
        )
        self.assertEqual(failure["completed_panels"], [])
        self.assertEqual(
            failure["formal_attempt_marker"]["sha256"],
            sha256(self.marker),
        )

    def test_marker_create_error_recovers_exact_inode_as_consumed(self) -> None:
        events: list[str] = []
        real_create = runner.create_json_artifact_exclusive_held

        def raise_after_marker_create(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> runner.HeldArtifact:
            held = real_create(path, payload, compact=compact)
            if path == self.marker:
                held.close()
                raise OSError("synthetic marker publication return failure")
            return held

        with mock.patch.object(
            runner,
            "create_json_artifact_exclusive_held",
            side_effect=raise_after_marker_create,
        ), self.assertRaisesRegex(
            OSError,
            "synthetic marker publication return failure",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )

        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.failure.is_file())
        self.assertFalse(self.success.exists())
        self.assertFalse(self.old_output.exists())
        self.assertFalse(self.valid_output.exists())
        self.assertFalse(any(event.startswith("dataset:") for event in events))
        failure = json.loads(self.failure.read_text(encoding="utf-8"))
        self.assertEqual(
            failure["failure_class"],
            "execution_integrity_failure",
        )
        self.assertEqual(
            failure["formal_attempt_marker"]["sha256"],
            sha256(self.marker),
        )
        self.assertTrue(failure["no_retry"])

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

    def test_digest_lock_survives_same_byte_preregistration_replace(
        self,
    ) -> None:
        first_process_lock = runner.acquire_process_execution_lock(
            self.preregistration,
            self.preregistration_sha256,
        )
        first_inode_lock = runner.open_locked_verified_preregistration(
            self.preregistration,
            self.preregistration_sha256,
        )
        try:
            replacement = self.root / "same-byte-replacement.json"
            replacement.write_bytes(self.preregistration.read_bytes())
            replacement.replace(self.preregistration)

            blocked_events: list[str] = []
            with self.assertRaisesRegex(
                runner.ConcurrentExecutionError,
                "authoritative repo-",
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=self._fake_evaluator(blocked_events),
                    torch_module=self._fake_torch(blocked_events),
                )
            self.assertEqual(blocked_events, [])
            self.assertFalse(self.marker.exists())
        finally:
            first_inode_lock.close()
            first_process_lock.close()

        retry_events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(retry_events),
            torch_module=self._fake_torch(retry_events),
        )
        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.success.is_file())
        self.assertTrue(first_process_lock.path.is_file())

    def test_repo_root_lock_survives_digest_sidecar_file_replace(
        self,
    ) -> None:
        first_process_lock = runner.acquire_process_execution_lock(
            self.preregistration,
            self.preregistration_sha256,
        )
        try:
            replacement = first_process_lock.path.parent / "replacement.lock"
            replacement.write_bytes(b"replacement sidecar inode")
            os.replace(replacement, first_process_lock.path)

            real_open = runner.os.open
            opened_paths: list[Path] = []

            def recording_open(
                path: object,
                flags: int,
                mode: int = 0o777,
                **kwargs: object,
            ) -> int:
                opened_paths.append(Path(path))
                return real_open(path, flags, mode, **kwargs)

            blocked_events: list[str] = []
            with mock.patch.object(
                runner.os,
                "open",
                side_effect=recording_open,
            ), self.assertRaisesRegex(
                runner.ConcurrentExecutionError,
                "authoritative repo-",
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=self._fake_evaluator(blocked_events),
                    torch_module=self._fake_torch(blocked_events),
                )
            self.assertEqual(opened_paths, [Path(".")])
            self.assertEqual(blocked_events, [])
            self.assertFalse(self.marker.exists())
        finally:
            first_process_lock.close()

        retry_events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(retry_events),
            torch_module=self._fake_torch(retry_events),
        )
        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.success.is_file())

    def test_repo_root_lock_survives_digest_sidecar_directory_replace(
        self,
    ) -> None:
        first_process_lock = runner.acquire_process_execution_lock(
            self.preregistration,
            self.preregistration_sha256,
        )
        lock_directory = first_process_lock.path.parent
        displaced_directory = lock_directory.with_name(
            lock_directory.name + "-displaced"
        )
        replacement_directory = lock_directory.with_name(
            lock_directory.name + "-replacement"
        )
        replacement_directory.mkdir()
        try:
            os.replace(lock_directory, displaced_directory)
            os.replace(replacement_directory, lock_directory)
            self.assertEqual(list(lock_directory.iterdir()), [])

            blocked_events: list[str] = []
            with self.assertRaisesRegex(
                runner.ConcurrentExecutionError,
                "authoritative repo-",
            ):
                runner.execute(
                    self.preregistration,
                    self.preregistration_sha256,
                    evaluator_module=self._fake_evaluator(blocked_events),
                    torch_module=self._fake_torch(blocked_events),
                )
            self.assertEqual(blocked_events, [])
            self.assertEqual(list(lock_directory.iterdir()), [])
            self.assertFalse(self.marker.exists())
        finally:
            first_process_lock.close()

        retry_events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(retry_events),
            torch_module=self._fake_torch(retry_events),
        )
        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertTrue(self.marker.is_file())
        self.assertTrue(self.success.is_file())

    def test_pre_marker_lock_close_does_not_consume_attempt(self) -> None:
        process_lock = runner.acquire_process_execution_lock(
            self.preregistration,
            self.preregistration_sha256,
        )
        lock_path = process_lock.path
        process_lock.close()
        self.assertTrue(lock_path.is_file())
        self.assertFalse(self.marker.exists())

        events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(events),
            torch_module=self._fake_torch(events),
        )
        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertTrue(self.marker.is_file())

    def test_pre_marker_process_exit_releases_digest_lock(self) -> None:
        ready_read, ready_write = os.pipe()
        exit_read, exit_write = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            try:
                os.close(ready_read)
                os.close(exit_write)
                runner.acquire_process_execution_lock(
                    self.preregistration,
                    self.preregistration_sha256,
                )
                os.write(ready_write, b"locked")
                os.read(exit_read, 1)
                os._exit(0)
            except BaseException:
                os._exit(17)

        os.close(ready_write)
        os.close(exit_read)
        try:
            self.assertEqual(os.read(ready_read, 6), b"locked")
            self.assertFalse(self.marker.exists())
            os.write(exit_write, b"x")
            _, wait_status = os.waitpid(child_pid, 0)
            self.assertTrue(os.WIFEXITED(wait_status))
            self.assertEqual(os.WEXITSTATUS(wait_status), 0)
        finally:
            os.close(ready_read)
            os.close(exit_write)

        events: list[str] = []
        completed = runner.execute(
            self.preregistration,
            self.preregistration_sha256,
            evaluator_module=self._fake_evaluator(events),
            torch_module=self._fake_torch(events),
        )
        self.assertEqual([item["order"] for item in completed], [1, 2])
        self.assertTrue(self.marker.is_file())

    def test_trusted_root_rename_cannot_move_concurrency_to_new_inode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=runner.REPO_ROOT,
            prefix=".frozen-root-anchor-test-",
        ) as parent_text:
            parent = Path(parent_text)
            trusted_root = parent / "repo"
            trusted_root.mkdir()
            preregistration = trusted_root / "preregistration.json"
            preregistration.write_bytes(b"{}\n")
            expected_sha256 = hashlib.sha256(
                preregistration.read_bytes()
            ).hexdigest()
            anchor = runner.open_trusted_root_anchor(trusted_root)
            first_lock = runner.acquire_process_execution_lock(
                preregistration,
                expected_sha256,
                root_anchor=anchor,
            )
            moved_root = parent / "repo-original"
            try:
                trusted_root.rename(moved_root)
                trusted_root.mkdir()
                with self.assertRaisesRegex(
                    runner.ProtocolError,
                    "startup path changed|identity",
                ):
                    runner.validate_process_root_lock(first_lock)
                with self.assertRaisesRegex(
                    runner.ConcurrentExecutionError,
                    "repo-parent",
                ):
                    runner.acquire_process_execution_lock(
                        preregistration,
                        expected_sha256,
                        root_anchor=anchor,
                    )
                self.assertEqual(list(trusted_root.iterdir()), [])
            finally:
                trusted_root.rmdir()
                moved_root.rename(trusted_root)
                first_lock.close()

            retry_lock = runner.acquire_process_execution_lock(
                preregistration,
                expected_sha256,
                root_anchor=anchor,
            )
            retry_lock.close()
            os.close(anchor.root_descriptor)
            os.close(anchor.parent_descriptor)

    def test_openat_artifact_write_never_follows_swapped_ancestor(
        self,
    ) -> None:
        branch = self.root / "artifact-branch"
        nested = branch / "nested"
        nested.mkdir(parents=True)
        moved_branch = self.root / "artifact-branch-original"
        attacker = self.root / "attacker"
        (attacker / "nested").mkdir(parents=True)
        target = nested / "victim.json"
        real_open = runner.os.open
        swapped = False

        def swap_before_leaf_open(
            path: object,
            flags: int,
            mode: int = 0o777,
            **kwargs: object,
        ) -> int:
            nonlocal swapped
            if (
                not swapped
                and Path(path).name == target.name
                and flags & runner.os.O_CREAT
            ):
                swapped = True
                branch.rename(moved_branch)
                branch.symlink_to(attacker, target_is_directory=True)
            return real_open(path, flags, mode, **kwargs)

        with mock.patch.object(
            runner.os,
            "open",
            side_effect=swap_before_leaf_open,
        ), self.assertRaisesRegex(
            runner.ProtocolError,
            "ancestor|canonical pathname",
        ):
            runner.atomic_write_json_exclusive(target, {"trusted": True})

        self.assertTrue(swapped)
        self.assertFalse((attacker / "nested" / target.name).exists())
        created = moved_branch / "nested" / target.name
        self.assertTrue(created.is_file())
        self.assertEqual(json.loads(created.read_text()), {"trusted": True})

    def test_secure_artifact_reread_rejects_swapped_ancestor(self) -> None:
        branch = self.root / "reread-branch"
        nested = branch / "nested"
        nested.mkdir(parents=True)
        target = nested / "persisted.json"
        expected_sha256 = runner.atomic_write_json_exclusive(
            target,
            {"trusted": True},
        )
        moved_branch = self.root / "reread-branch-original"
        attacker = self.root / "reread-attacker"
        (attacker / "nested").mkdir(parents=True)
        real_open = runner.os.open
        swapped = False

        def swap_before_leaf_read(
            path: object,
            flags: int,
            mode: int = 0o777,
            **kwargs: object,
        ) -> int:
            nonlocal swapped
            if not swapped and Path(path).name == target.name:
                swapped = True
                branch.rename(moved_branch)
                branch.symlink_to(attacker, target_is_directory=True)
            return real_open(path, flags, mode, **kwargs)

        with mock.patch.object(
            runner.os,
            "open",
            side_effect=swap_before_leaf_read,
        ), self.assertRaisesRegex(
            runner.ProtocolError,
            "ancestor|changed",
        ):
            runner.read_json_artifact_verified(
                target,
                expected_sha256,
                "persisted behavior output",
            )

        self.assertTrue(swapped)
        self.assertTrue((moved_branch / "nested" / target.name).is_file())
        self.assertFalse((attacker / "nested" / target.name).exists())

    def test_unlinked_certificate_is_rejected_before_formal_marker(
        self,
    ) -> None:
        events: list[str] = []
        real_create = runner.create_json_artifact_exclusive_held
        certificate_descriptors: list[int] = []

        def unlink_certificate_after_publish(
            path: Path,
            payload: dict[str, object],
            *,
            compact: bool = False,
        ) -> runner.HeldArtifact:
            held = real_create(path, payload, compact=compact)
            if path.name.startswith("certificate."):
                certificate_descriptors.append(held.descriptor)
                held.path.unlink()
            return held

        with mock.patch.object(
            runner,
            "create_json_artifact_exclusive_held",
            side_effect=unlink_certificate_after_publish,
        ), self.assertRaisesRegex(
            runner.ProtocolError,
            "preflight certificate.*changed|pathname disappeared",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )

        self.assertEqual(len(certificate_descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(certificate_descriptors[0])
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.old_output.exists())
        self.assertFalse(self.valid_output.exists())
        self.assertFalse(self.success.exists())
        self.assertFalse(any(event.startswith("dataset:") for event in events))

    def test_panel_one_runtime_failure_still_runs_panel_two(self) -> None:
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.PanelExecutionAggregateError,
            "1 behavior panel",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(
                    events,
                    fail_panel=self.old_data.name,
                    fail_panel_with_base_exception=True,
                ),
                torch_module=self._fake_torch(events),
            )

        self.assertIn(f"dataset:{self.valid_data.name}", events)
        self.assertIn(f"evaluate:{self.valid_data.name}:cuda", events)
        self.assertFalse(self.old_output.exists())
        self.assertTrue(self.valid_output.is_file())
        self.assertFalse(self.success.exists())
        failure = json.loads(self.failure.read_text(encoding="utf-8"))
        self.assertIsNone(failure["current_panel"])
        self.assertEqual(failure["failure_class"], "panel_runtime_failure")
        self.assertEqual(
            [item["order"] for item in failure["panel_failures"]],
            [1],
        )
        self.assertEqual(
            [item["order"] for item in failure["completed_panels"]],
            [2],
        )
        self.assertFalse(failure["gold19_authorized"])

    def test_one_failed_behavior_gate_forces_terminal_failure(self) -> None:
        source = json.loads(
            self.source_protocol.read_text(encoding="utf-8")
        )
        source_panels = source["behavior_protocol"]["ordered_panels"]
        source_panels[0]["minimum_correct"]["set_exact"] = 7
        preregistration = json.loads(
            self.preregistration.read_text(encoding="utf-8")
        )
        preregistration["metric_mapping_and_gates"] = (
            runner.expected_gate_mapping(source_panels)
        )
        self._rebind_authority_chain(
            source=source,
            preregistration=preregistration,
        )

        events: list[str] = []
        with self.assertRaisesRegex(
            runner.BehaviorGateFailure,
            "1 of 20",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(events),
                torch_module=self._fake_torch(events),
            )

        self.assertTrue(self.old_output.is_file())
        self.assertTrue(self.valid_output.is_file())
        self.assertFalse(self.success.exists())
        failure = json.loads(self.failure.read_text(encoding="utf-8"))
        self.assertIsNone(failure["current_panel"])
        self.assertEqual(failure["failure_class"], "behavior_gate_rejection")
        self.assertTrue(failure["quality_gates_evaluated"])
        self.assertFalse(failure["all_20_gates_passed"])
        self.assertFalse(failure["gold19_authorized"])
        report = failure["quality_gates"]
        self.assertEqual(report["gate_count"], 20)
        self.assertEqual(report["passed_gate_count"], 19)
        self.assertEqual(report["failing_gates"], ["old.set_exact"])
        self.assertEqual(len(report["gates"]), 20)

    def test_invalid_gate_fields_fail_without_short_circuiting_twenty(self) -> None:
        events: list[str] = []
        with self.assertRaisesRegex(
            runner.BehaviorGateFailure,
            "6 of 20",
        ):
            runner.execute(
                self.preregistration,
                self.preregistration_sha256,
                evaluator_module=self._fake_evaluator(
                    events,
                    invalid_gate_fields=True,
                ),
                torch_module=self._fake_torch(events),
            )

        self.assertTrue(self.old_output.is_file())
        self.assertTrue(self.valid_output.is_file())
        self.assertFalse(self.success.exists())
        failure = json.loads(self.failure.read_text(encoding="utf-8"))
        report = failure["quality_gates"]
        self.assertTrue(report["all_20_gates_evaluated"])
        self.assertFalse(report["all_20_gates_passed"])
        self.assertEqual(report["gate_count"], 20)
        self.assertEqual(report["passed_gate_count"], 14)
        self.assertEqual(len(report["gates"]), 20)
        self.assertEqual(
            report["failing_gates"],
            [
                "old.rows",
                "old.set_exact",
                "old.count",
                "valid.rows",
                "valid.set_exact",
                "valid.count",
            ],
        )
        failed = {
            item["gate"]: item
            for item in report["gates"]
            if item["pass"] is False
        }
        self.assertIsNone(failed["old.rows"]["observed"])
        self.assertIn(
            "strict integer",
            failed["old.rows"]["validation_error"],
        )
        self.assertIsNone(failed["old.set_exact"]["observed"])
        self.assertIn(
            "strict integer",
            failed["old.set_exact"]["validation_error"],
        )
        self.assertIsNone(failed["valid.count"]["observed"])
        self.assertIn(
            "missing output field",
            failed["valid.count"]["validation_error"],
        )
        self.assertFalse(failure["all_20_gates_passed"])
        self.assertFalse(failure["gold19_authorized"])

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
            **kwargs: object,
        ) -> int:
            if Path(path).name == target.name:
                observed_flags.append(flags)
            return real_open(path, flags, mode, **kwargs)

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
            runner.PanelExecutionAggregateError,
            "1 behavior panel",
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
        self.assertIsNone(failure["current_panel"])
        self.assertEqual(len(failure["completed_panels"]), 1)
        self.assertEqual(len(failure["panel_failures"]), 1)
        self.assertIn(
            f"synthetic failure for {self.valid_data.name}",
            failure["panel_failures"][0]["error_message"],
        )
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
