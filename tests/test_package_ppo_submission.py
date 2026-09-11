from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGER_PATH = ROOT / "tools/package_ppo_submission.py"
if str(PACKAGER_PATH.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGER_PATH.parent))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "package_ppo_submission_for_test",
        PACKAGER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


packager = load_module()
PAIR_VERIFIER_PATH = ROOT / "tools/verify_ppo_package_pair.py"


def load_pair_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_ppo_package_pair_for_test",
        PAIR_VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pair_verifier = load_pair_verifier()


class DeterministicPPOPackageTest(unittest.TestCase):
    def test_torch_serialization_is_path_independent_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first" / "model.pt"
            second = root / "second" / "model.pt"
            first.parent.mkdir()
            second.parent.mkdir()
            checkpoint = {
                "feature_version": "ptcg-selfplay-ppo-terminal01-v1",
                "model_state_dict": {
                    "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)
                },
                "update": 7,
            }

            packager.deterministic_torch_save(checkpoint, first)
            packager.deterministic_torch_save(checkpoint, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            reloaded = torch.load(first, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(
                    reloaded["model_state_dict"]["weight"],
                    checkpoint["model_state_dict"]["weight"],
                )
            )

    def test_archive_is_byte_identical_with_normalized_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            for index, name in enumerate(packager.EXPECTED_MEMBERS):
                (bundle / name).write_bytes(
                    f"member {index}: {name}\n".encode("utf-8")
                )
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"

            packager.write_deterministic_archive(bundle, first)
            packager.write_deterministic_archive(bundle, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            )
            self.assertEqual(first.read_bytes()[4:8], b"\x00\x00\x00\x00")
            with tarfile.open(first, mode="r:gz") as archive:
                members = archive.getmembers()
                self.assertEqual(
                    tuple(member.name for member in members),
                    packager.EXPECTED_MEMBERS,
                )
                for member in members:
                    self.assertTrue(member.isfile())
                    self.assertEqual(member.mtime, 0)
                    self.assertEqual(member.mode, 0o644)
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.uname, "")
                    self.assertEqual(member.gname, "")

    def test_two_complete_synthetic_builds_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root / "template"
            template.mkdir()
            (template / "main.py").write_text("def agent(_obs):\n    return []\n")
            (template / "policy_runtime.py").write_text("VALUE = 1\n")
            (template / "deck.csv").write_text("".join("1\n" for _ in range(60)))
            contract = root / "CONTRACT.json"
            contract.write_text(
                json.dumps(
                    {
                        "schema_version": "synthetic-template-v1",
                        "template_version": "synthetic-raw-v1",
                        "decode": {"order_mode": "raw"},
                        "deck": {
                            "semantic_hash": hashlib.sha256(
                                ",".join("1" for _ in range(60)).encode("utf-8")
                            ).hexdigest(),
                            "cards": 60,
                        },
                        "template_files": {
                            name: packager.sha256_file(template / name)
                            for name in (
                                "main.py",
                                "deck.csv",
                                "policy_runtime.py",
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "source.pt"
            checkpoint = {
                "feature_version": "ptcg-selfplay-ppo-terminal01-v1",
                "bc_feature_version": "ptcg-bc-orbit-entity-transformer-v5",
                "model_config": {"model_dim": 2},
                "model_state_dict": {
                    "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)
                },
                "reward": {"win": 1.0, "other": 0.0},
                "update": 7,
                "action_distribution": "synthetic ordered policy",
                "learner_deck_hash": hashlib.sha256(
                    ",".join("1" for _ in range(60)).encode("utf-8")
                ).hexdigest(),
            }
            torch.save(checkpoint, checkpoint_path)
            deployment_contract = root / "DEPLOYMENT.json"
            deployment_contract.write_text(
                json.dumps(
                    {
                        "schema_version": "ptcg-gold-push-deployment-contract-v1",
                        "candidate": {
                            "path": str(checkpoint_path.resolve()),
                            "sha256": packager.sha256_file(checkpoint_path),
                        },
                        "candidate_deck": {
                            "path": str((template / "deck.csv").resolve()),
                            "file_sha256": packager.sha256_file(
                                template / "deck.csv"
                            ),
                            "semantic_hash": checkpoint["learner_deck_hash"],
                        },
                        "action_order": {
                            "mode": "raw",
                            "canonical_order": False,
                            "hybrid_order": False,
                        },
                        "submission_template": {
                            "contract": str(contract.resolve()),
                            "contract_sha256": packager.sha256_file(contract),
                        },
                    }
                ),
                encoding="utf-8",
            )
            archives: list[Path] = []
            manifests: list[dict[str, object]] = []
            for index in range(2):
                output_dir = root / f"bundle-{index}"
                archive = root / f"bundle-{index}.tar.gz"
                manifest = root / f"bundle-{index}.manifest.json"
                argv = [
                    str(PACKAGER_PATH),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--template-dir",
                    str(template),
                    "--deployment-contract",
                    str(deployment_contract),
                    "--template-contract",
                    str(contract),
                    "--action-order-mode",
                    "raw",
                    "--output-dir",
                    str(output_dir),
                    "--archive",
                    str(archive),
                    "--manifest",
                    str(manifest),
                ]
                with mock.patch.object(sys, "argv", argv):
                    with redirect_stdout(io.StringIO()):
                        packager.main()
                archives.append(archive)
                manifests.append(json.loads(manifest.read_text()))

            self.assertEqual(archives[0].read_bytes(), archives[1].read_bytes())
            self.assertEqual(
                manifests[0]["archive_sha256"],
                manifests[1]["archive_sha256"],
            )
            self.assertEqual(
                manifests[0]["members"],
                manifests[1]["members"],
            )
            self.assertTrue(manifests[0]["byte_reproducible_packaging"])
            self.assertTrue(manifests[1]["byte_reproducible_packaging"])
            audit = pair_verifier.audit(
                SimpleNamespace(
                    archive_a=archives[0],
                    archive_b=archives[1],
                    manifest_a=root / "bundle-0.manifest.json",
                    manifest_b=root / "bundle-1.manifest.json",
                    checkpoint=checkpoint_path,
                    main_file=template / "main.py",
                    deck_file=template / "deck.csv",
                    policy_runtime_file=template / "policy_runtime.py",
                    template_contract=contract,
                    deployment_contract=deployment_contract,
                    action_order_mode="raw",
                )
            )
            self.assertTrue(audit["pass"])
            self.assertTrue(audit["independent_archive_bytes_identical"])
            self.assertTrue(audit["model_tensors_equal_source"])
            self.assertEqual(
                audit["deployment_contract"]["sha256"],
                packager.sha256_file(deployment_contract),
            )


if __name__ == "__main__":
    unittest.main()
