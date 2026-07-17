from __future__ import annotations

import stat
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from api import (
    ApiError,
    TemplateSpec,
    create_pod_document,
    is_capacity_failure,
    validate_template,
)
from controller import validate_result_identity, validate_runner_identity
from lease import Lease, LeaseStore


class RunPodContract(unittest.TestCase):
    def test_graphql_documents_use_variables_for_operator_values(self) -> None:
        document = create_pod_document("SECURE", True)
        self.assertIn("imageName: $image", document)
        self.assertIn("gpuTypeId: $gpu", document)
        self.assertIn("dataCenterId: $dataCenter", document)
        self.assertNotIn("example/image", document)
        with self.assertRaises(ValueError):
            create_pod_document("UNTRUSTED", False)

        templated = create_pod_document("SECURE", False, True)
        self.assertIn("templateId: $templateId", templated)
        self.assertNotIn("imageName: $image", templated)

    def test_private_template_contract_rejects_drift(self) -> None:
        spec = TemplateSpec(
            name="nml-contracts-v1",
            image="ghcr.io/example/nml@sha256:" + "a" * 64,
            container_disk_gb=20,
        )
        template = {"id": "template-id", **spec.payload()}
        validate_template(template, spec)
        template["ports"] = ["22/tcp"]
        with self.assertRaises(ApiError):
            validate_template(template, spec)

    def test_only_capacity_failures_enable_gpu_fallback(self) -> None:
        self.assertTrue(is_capacity_failure("Unable to find a machine with capacity"))
        self.assertTrue(
            is_capacity_failure(
                "There are no longer any instances available with the requested specifications"
            )
        )
        self.assertTrue(
            is_capacity_failure(
                "This machine does not have the resources to deploy your pod"
            )
        )
        self.assertFalse(is_capacity_failure("GraphQL field imageName is invalid"))
        self.assertFalse(is_capacity_failure("authorization failed"))

    def test_lease_write_is_atomic_private_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LeaseStore(Path(directory) / "leases")
            lease = Lease.create(
                image="ghcr.io/example/nml@sha256:" + "a" * 64,
                image_digest="sha256:" + "a" * 64,
                source_commit="b" * 40,
                source_dirty=True,
                requested_gpus=["GPU A", "GPU B"],
                deadline_at=datetime.now(UTC) + timedelta(minutes=10),
                lease_token="secret-token",
            )
            path = store.save(lease)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(store.load(lease.lease_id), lease)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_runner_identity_must_match_the_exact_lease(self) -> None:
        lease = Lease.create(
            image="ghcr.io/example/nml@sha256:" + "a" * 64,
            image_digest="sha256:" + "a" * 64,
            source_commit="b" * 40,
            source_dirty=False,
            requested_gpus=["GPU"],
            deadline_at=datetime.now(UTC) + timedelta(minutes=10),
            lease_token="token",
        )
        payload = {
            "schema_version": 1,
            "artifact": {
                "image_digest": lease.image_digest,
                "source_commit": lease.source_commit,
                "source_dirty": False,
            },
        }
        validate_runner_identity(lease, payload)
        payload["artifact"]["image_digest"] = "sha256:" + "c" * 64
        with self.assertRaises(RuntimeError):
            validate_runner_identity(lease, payload)

    def test_terminal_result_requires_typed_gpu_identity(self) -> None:
        lease = Lease.create(
            image="ghcr.io/example/nml@sha256:" + "a" * 64,
            image_digest="sha256:" + "a" * 64,
            source_commit="b" * 40,
            source_dirty=False,
            requested_gpus=["GPU"],
            deadline_at=datetime.now(UTC) + timedelta(minutes=10),
            lease_token="token",
        )
        result = {
            "schema_version": 1,
            "artifact": {
                "image_digest": lease.image_digest,
                "source_commit": lease.source_commit,
                "source_dirty": False,
            },
            "hardware": [
                {
                    "index": 0,
                    "name": "NVIDIA RTX A6000",
                    "uuid": "GPU-example",
                    "compute_capability": "8.6",
                    "driver_version": "590.48.01",
                }
            ],
            "status": "succeeded",
        }
        validate_result_identity(lease, result)
        result["hardware"][0]["compute_capability"] = ""
        with self.assertRaises(RuntimeError):
            validate_result_identity(lease, result)


if __name__ == "__main__":
    unittest.main()
