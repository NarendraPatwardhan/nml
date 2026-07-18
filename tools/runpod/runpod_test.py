from __future__ import annotations

import stat
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from api import (
    ApiError,
    NETWORK_VOLUME_MOUNT_PATH,
    RunPodClient,
    TemplateSpec,
    create_pod_document,
    create_ssh_job_pod_document,
    is_capacity_failure,
    require_contract_inputs,
    require_workspace_path,
    same_exact_image,
    ssh_endpoint,
    validate_template,
)
from controller import (
    record_network_volume_result,
    validate_result_identity,
    validate_runner_identity,
)
from lease import Lease, LeaseStore

DATA_CENTER = "US-WA-1"


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

        volume_backed = create_pod_document("SECURE", True, False, True)
        self.assertIn("$networkVolumeId: String!", volume_backed)
        self.assertIn("networkVolumeId: $networkVolumeId", volume_backed)
        self.assertIn(
            f'volumeMountPath: "{NETWORK_VOLUME_MOUNT_PATH}"', volume_backed
        )
        self.assertIn("dataCenterId: $dataCenter", volume_backed)
        self.assertNotIn("HF_TOKEN", volume_backed)
        self.assertNotIn("MODEL", volume_backed)

        input_backed = create_pod_document(
            "SECURE", True, False, True, ("NML_MODEL", "NML_ORACLE")
        )
        self.assertIn("$contractInput0: String!", input_backed)
        self.assertIn("$contractInput1: String!", input_backed)
        self.assertIn(
            '{key: "NML_MODEL", value: $contractInput0}', input_backed
        )
        self.assertIn('{key: "NML_ORACLE", value: $contractInput1}', input_backed)
        with self.assertRaises(ValueError):
            create_pod_document("SECURE", False, False, True)
        with self.assertRaises(ValueError):
            create_pod_document("SECURE", True, False, False, ("NML_MODEL",))
        with self.assertRaises(ValueError):
            create_pod_document(
                "SECURE", True, False, True, ("NML_SOURCE_COMMIT",)
            )

        ephemeral = create_pod_document("SECURE", True)
        self.assertNotIn("networkVolumeId", ephemeral)

        ssh_job = create_ssh_job_pod_document("COMMUNITY", True)
        self.assertIn("imageName: $image", ssh_job)
        self.assertIn("startSsh: true", ssh_job)
        self.assertIn('ports: "22/tcp"', ssh_job)
        self.assertIn("dataCenterId: $dataCenter", ssh_job)
        self.assertNotIn("HF_TOKEN", ssh_job)

    def test_device_contract_pod_passes_existing_volume_as_graphql_variable(
        self,
    ) -> None:
        image = "ghcr.io/example/nml@sha256:" + "a" * 64

        class RecordingClient(RunPodClient):
            def __init__(self) -> None:
                super().__init__("api-key")
                self.document = ""
                self.variables: dict[str, object] = {}

            def graphql(
                self, operation: str, document: str, variables: dict[str, object]
            ) -> dict[str, object]:
                self.document = document
                self.variables = variables
                return {
                    "podFindAndDeployOnDemand": {
                        "id": "pod-123",
                        "imageName": image,
                        "machineId": "machine-123",
                    }
                }

        client = RecordingClient()
        client.create_device_contract_pod(
            name="nml-contracts-test",
            image=image,
            gpu_types=["GPU"],
            gpu_count=1,
            cloud="SECURE",
            container_disk_gb=20,
            lease_token="lease-token",
            image_digest="sha256:" + "a" * 64,
            source_commit="b" * 40,
            source_dirty=False,
            data_center=DATA_CENTER,
            template_id=None,
            network_volume_id="volume-123",
            contract_inputs={
                "NML_MODEL": "/workspace/models/selected",
                "NML_ORACLE": "/workspace/fixtures/generation.json",
            },
        )
        self.assertEqual(client.variables["dataCenter"], DATA_CENTER)
        self.assertEqual(client.variables["networkVolumeId"], "volume-123")
        self.assertEqual(
            client.variables["contractInput0"],
            "/workspace/models/selected",
        )
        self.assertIn("networkVolumeId: $networkVolumeId", client.document)
        self.assertIn("NML_MODEL", client.document)
        self.assertEqual(
            client.variables["contractInput1"],
            "/workspace/fixtures/generation.json",
        )
        self.assertIn("NML_ORACLE", client.document)
        self.assertNotIn("HF_TOKEN", client.document)

        with self.assertRaises(ValueError):
            client.create_device_contract_pod(
                name="nml-contracts-test",
                image=image,
                gpu_types=["GPU"],
                gpu_count=1,
                cloud="SECURE",
                container_disk_gb=20,
                lease_token="lease-token",
                image_digest="sha256:" + "a" * 64,
                source_commit="b" * 40,
                source_dirty=False,
                data_center=DATA_CENTER,
                template_id=None,
                network_volume_id=None,
                contract_inputs={"NML_INPUT": "/workspace/fixtures/value.json"},
            )

        with self.assertRaises(ValueError):
            client.create_device_contract_pod(
                name="nml-contracts-test",
                image=image,
                gpu_types=["GPU"],
                gpu_count=1,
                cloud="SECURE",
                container_disk_gb=20,
                lease_token="lease-token",
                image_digest="sha256:" + "a" * 64,
                source_commit="b" * 40,
                source_dirty=False,
                data_center=None,
                template_id=None,
                network_volume_id="volume-123",
            )

    def test_contract_inputs_are_canonical_and_belong_to_the_volume_mount(self) -> None:
        input_path = "/workspace/models/selected"
        self.assertEqual(require_workspace_path("input path", input_path), input_path)
        self.assertEqual(
            require_contract_inputs({"NML_MODEL": input_path}),
            {"NML_MODEL": input_path},
        )
        for invalid in (
            "models/selected",
            "/workspace",
            "/workspace/../secrets",
            "/tmp/model",
            "/workspace/models/",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                require_workspace_path("input path", invalid)
        for name in ("lowercase", "9INVALID", "NML-SPLIT", "NML_SOURCE_DIRTY"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                require_contract_inputs({name: input_path})

    def test_ssh_endpoint_requires_the_dynamic_public_tcp_mapping(self) -> None:
        pod = {
            "runtime": {
                "ports": [
                    {
                        "ip": "192.0.2.1",
                        "isIpPublic": True,
                        "privatePort": 22,
                        "publicPort": 23456,
                        "type": "tcp",
                    }
                ]
            }
        }
        self.assertEqual(ssh_endpoint(pod), ("192.0.2.1", 23456))
        pod["runtime"]["ports"][0]["isIpPublic"] = False
        self.assertIsNone(ssh_endpoint(pod))

    def test_exact_image_comparison_allows_only_registry_host_normalization(self) -> None:
        digest = "sha256:" + "a" * 64
        self.assertTrue(
            same_exact_image(
                f"docker.io/runpod/pytorch@{digest}", f"runpod/pytorch@{digest}"
            )
        )
        self.assertFalse(
            same_exact_image(
                f"docker.io/runpod/pytorch@{digest}",
                "runpod/pytorch@sha256:" + "b" * 64,
            )
        )
        self.assertFalse(same_exact_image("runpod/pytorch:latest", "runpod/pytorch:latest"))

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
                network_volume_id="volume-123",
                network_volume_data_center=DATA_CENTER,
                network_volume_mount_path=NETWORK_VOLUME_MOUNT_PATH,
                contract_inputs={
                    "NML_MODEL": "/workspace/models/example",
                    "NML_ORACLE": "/workspace/fixtures/example.json",
                },
            )
            path = store.save(lease)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(store.load(lease.lease_id), lease)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            self.assertEqual(
                lease.network_volume_identity(),
                {
                    "id": "volume-123",
                    "data_center": DATA_CENTER,
                    "mount_path": NETWORK_VOLUME_MOUNT_PATH,
                    "contract_inputs": {
                        "NML_MODEL": "/workspace/models/example",
                        "NML_ORACLE": "/workspace/fixtures/example.json",
                    },
                },
            )
            public = lease.public_record()
            self.assertEqual(public["network_volume_id"], "volume-123")
            self.assertEqual(public["network_volume"], lease.network_volume_identity())
            self.assertNotIn("lease_token", public)

            result: dict[str, object] = {"status": "succeeded"}
            record_network_volume_result(lease, result)
            self.assertEqual(result["network_volume"], lease.network_volume_identity())

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
