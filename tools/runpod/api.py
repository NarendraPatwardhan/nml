"""Typed HTTP boundaries for the intentionally split RunPod APIs."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
EXACT_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
NETWORK_VOLUME_MOUNT_PATH = "/workspace"
ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
RESERVED_RUNNER_ENVIRONMENT = frozenset(
    {
        "NML_CONTRACT_LEASE_TOKEN",
        "NML_IMAGE_DIGEST",
        "NML_SOURCE_COMMIT",
        "NML_SOURCE_DIRTY",
    }
)

POD_QUERY = """
query NmlPod($podId: String!) {
  pod(input: {podId: $podId}) {
    id
    name
    desiredStatus
    imageName
    costPerHr
    machineId
    runtime {
      uptimeInSeconds
      ports { ip isIpPublic privatePort publicPort type }
      gpus { id gpuUtilPercent memoryUtilPercent }
    }
  }
}
"""

TERMINATE_POD = """
mutation NmlTerminatePod($podId: String!) {
  podTerminate(input: {podId: $podId})
}
"""

_CREATE_POD = """
mutation NmlCreatePod(
  $name: String!
  {image_variable}
  $gpu: String!
  $gpuCount: Int!
  $containerDisk: Int!
  $leaseToken: String!
  $imageDigest: String!
  $sourceCommit: String!
  $sourceDirty: String!
  {template_variable}
  {data_center_variable}
  {network_volume_variable}
  {contract_input_variables}
) {{
  podFindAndDeployOnDemand(input: {{
    name: $name
    {artifact_fields}
    gpuTypeId: $gpu
    gpuCount: $gpuCount
    cloudType: {cloud}
    containerDiskInGb: $containerDisk
    volumeInGb: 0
    volumeMountPath: "{volume_mount_path}"
    {network_volume_field}
    dockerArgs: ""
    ports: "8080/http"
    supportPublicIp: true
    {data_center_field}
    env: [
      {{key: "NML_CONTRACT_LEASE_TOKEN", value: $leaseToken}}
      {{key: "NML_IMAGE_DIGEST", value: $imageDigest}}
      {{key: "NML_SOURCE_COMMIT", value: $sourceCommit}}
      {{key: "NML_SOURCE_DIRTY", value: $sourceDirty}}
      {contract_input_env}
    ]
  }}) {{
    id
    imageName
    machineId
  }}
}}
"""

_CREATE_SSH_JOB_POD = """
mutation NmlCreateSshJobPod(
  $name: String!
  $image: String!
  $gpu: String!
  $gpuCount: Int!
  $containerDisk: Int!
  $publicKey: String!
  {data_center_variable}
) {{
  podFindAndDeployOnDemand(input: {{
    name: $name
    imageName: $image
    gpuTypeId: $gpu
    gpuCount: $gpuCount
    cloudType: {cloud}
    containerDiskInGb: $containerDisk
    volumeInGb: 0
    volumeMountPath: "/workspace"
    dockerArgs: ""
    ports: "22/tcp"
    startSsh: true
    supportPublicIp: true
    {data_center_field}
    env: [{{key: "PUBLIC_KEY", value: $publicKey}}]
  }}) {{
    id
    imageName
    machineId
  }}
}}
"""


class ApiError(RuntimeError):
    """A transport or schema failure, with no safe placement fallback."""


class CapacityError(ApiError):
    """A placement refusal for which the next requested GPU may be tried."""


@dataclass(frozen=True)
class CreatedPod:
    pod_id: str
    image: str
    machine_id: str | None
    requested_gpu: str


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    image: str
    container_disk_gb: int

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("template name must be non-empty without surrounding whitespace")
        if EXACT_IMAGE.fullmatch(self.image) is None:
            raise ValueError("template image must be an exact lowercase sha256 digest")
        if self.container_disk_gb <= 0:
            raise ValueError("template container disk must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "imageName": self.image,
            "category": "NVIDIA",
            "containerDiskInGb": self.container_disk_gb,
            "dockerEntrypoint": [],
            "dockerStartCmd": [],
            "env": {},
            "isPublic": False,
            "isServerless": False,
            "ports": ["8080/http"],
            "readme": "NML immutable CUDA device-contract runner.",
            "volumeInGb": 0,
            "volumeMountPath": "/workspace",
        }


class RunPodClient:
    def __init__(
        self,
        api_key: str,
        *,
        rest_url: str = REST_URL,
        graphql_url: str = GRAPHQL_URL,
    ) -> None:
        if not api_key.strip():
            raise ValueError("RunPod API key must not be empty")
        self._api_key = api_key.strip()
        self._rest_url = rest_url.rstrip("/")
        self._graphql_url = graphql_url

    def create_device_contract_pod(
        self,
        *,
        name: str,
        image: str,
        gpu_types: list[str],
        gpu_count: int,
        cloud: str,
        container_disk_gb: int,
        lease_token: str,
        image_digest: str,
        source_commit: str,
        source_dirty: bool,
        data_center: str | None,
        template_id: str | None,
        network_volume_id: str | None = None,
        contract_inputs: dict[str, str] | None = None,
    ) -> CreatedPod:
        if not gpu_types:
            raise ValueError("at least one GPU type is required")
        if network_volume_id is not None:
            require_network_volume_id(network_volume_id)
            if (
                data_center is None
                or not data_center.strip()
                or data_center != data_center.strip()
            ):
                raise ValueError(
                    "network-volume data center must be non-empty without surrounding whitespace"
                )
        inputs = require_contract_inputs(contract_inputs or {})
        if inputs and network_volume_id is None:
            raise ValueError("contract inputs require an attached network volume")
        failures: list[str] = []
        for gpu_type in gpu_types:
            try:
                data = self.graphql(
                    "NmlCreatePod",
                    create_pod_document(
                        cloud,
                        data_center is not None,
                        template_id is not None,
                        network_volume_id is not None,
                        tuple(inputs),
                    ),
                    {
                        "name": name,
                        "gpu": gpu_type,
                        "gpuCount": gpu_count,
                        "containerDisk": container_disk_gb,
                        "leaseToken": lease_token,
                        "imageDigest": image_digest,
                        "sourceCommit": source_commit,
                        "sourceDirty": str(source_dirty).lower(),
                        **(
                            {"templateId": template_id}
                            if template_id
                            else {"image": image}
                        ),
                        **({"dataCenter": data_center} if data_center else {}),
                        **(
                            {"networkVolumeId": network_volume_id}
                            if network_volume_id
                            else {}
                        ),
                        **{
                            f"contractInput{index}": value
                            for index, (_, value) in enumerate(inputs.items())
                        },
                    },
                )
            except CapacityError as error:
                failures.append(f"{gpu_type}: {error}")
                continue
            pod = require_mapping(data.get("podFindAndDeployOnDemand"), "created Pod")
            pod_id = require_string(pod.get("id"), "created Pod id")
            returned_image = require_string(pod.get("imageName"), "created Pod imageName")
            self._require_created_image(pod_id, returned_image, image)
            machine_id = pod.get("machineId")
            if machine_id is not None and not isinstance(machine_id, str):
                raise ApiError("created Pod machineId is not a string or null")
            return CreatedPod(pod_id, returned_image, machine_id, gpu_type)
        raise CapacityError("; ".join(failures) or "no requested GPU had capacity")

    def create_ssh_job_pod(
        self,
        *,
        name: str,
        image: str,
        gpu_types: list[str],
        gpu_count: int,
        cloud: str,
        container_disk_gb: int,
        public_key: str,
        data_center: str | None,
    ) -> CreatedPod:
        """Creates an ephemeral SSH worker without putting job secrets in Pod env."""
        if EXACT_IMAGE.fullmatch(image) is None:
            raise ValueError("SSH job image must be an exact lowercase sha256 digest")
        if not gpu_types:
            raise ValueError("at least one GPU type is required")
        if not public_key.strip():
            raise ValueError("SSH public key must not be empty")
        failures: list[str] = []
        for gpu_type in gpu_types:
            try:
                data = self.graphql(
                    "NmlCreateSshJobPod",
                    create_ssh_job_pod_document(cloud, data_center is not None),
                    {
                        "name": name,
                        "image": image,
                        "gpu": gpu_type,
                        "gpuCount": gpu_count,
                        "containerDisk": container_disk_gb,
                        "publicKey": public_key.strip(),
                        **({"dataCenter": data_center} if data_center else {}),
                    },
                )
            except CapacityError as error:
                failures.append(f"{gpu_type}: {error}")
                continue
            pod = require_mapping(data.get("podFindAndDeployOnDemand"), "created Pod")
            pod_id = require_string(pod.get("id"), "created Pod id")
            returned_image = require_string(pod.get("imageName"), "created Pod imageName")
            self._require_created_image(pod_id, returned_image, image)
            machine_id = pod.get("machineId")
            if machine_id is not None and not isinstance(machine_id, str):
                raise ApiError("created Pod machineId is not a string or null")
            return CreatedPod(pod_id, returned_image, machine_id, gpu_type)
        raise CapacityError("; ".join(failures) or "no requested GPU had capacity")

    def _require_created_image(
        self, pod_id: str, returned_image: str, expected_image: str
    ) -> None:
        if same_exact_image(returned_image, expected_image):
            return
        try:
            self.terminate(pod_id)
        except Exception as cleanup_error:
            raise ApiError(
                f"RunPod created image {returned_image!r}, expected exact image "
                f"{expected_image!r}; Pod {pod_id} may still be billable because "
                f"termination failed: {cleanup_error}"
            ) from cleanup_error
        raise ApiError(
            f"RunPod created image {returned_image!r}, expected exact image "
            f"{expected_image!r}; mismatched Pod {pod_id} was terminated"
        )

    def pod(self, pod_id: str) -> dict[str, Any] | None:
        data = self.graphql("NmlPod", POD_QUERY, {"podId": pod_id})
        pod = data.get("pod")
        if pod is None:
            return None
        return require_mapping(pod, "Pod query")

    def terminate(self, pod_id: str) -> None:
        self.graphql("NmlTerminatePod", TERMINATE_POD, {"podId": pod_id})

    def templates(self) -> list[dict[str, Any]]:
        result = self.rest("GET", "/templates")
        if isinstance(result, list):
            return [require_mapping(item, "template") for item in result]
        result = require_mapping(result, "template-list response")
        for key in ("templates", "data", "items"):
            items = result.get(key)
            if isinstance(items, list):
                return [require_mapping(item, "template") for item in items]
        raise ApiError("template-list response contains no template array")

    def create_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        return require_mapping(self.rest("POST", "/templates", payload), "created template")

    def ensure_template(self, spec: TemplateSpec) -> tuple[dict[str, Any], bool]:
        matches = [item for item in self.templates() if item.get("name") == spec.name]
        if len(matches) > 1:
            raise ApiError(f"multiple RunPod templates are named {spec.name!r}")
        if matches:
            validate_template(matches[0], spec)
            return matches[0], False

        self.create_template(spec.payload())
        matches = [item for item in self.templates() if item.get("name") == spec.name]
        if len(matches) != 1:
            raise ApiError(
                f"created template {spec.name!r} was not uniquely observable through REST"
            )
        validate_template(matches[0], spec)
        return matches[0], True

    def graphql(
        self, operation: str, document: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        result = require_mapping(
            self._request(
                "POST",
                self._graphql_url,
                {"query": document, "variables": variables, "operationName": operation},
            ),
            "GraphQL response",
        )
        errors = result.get("errors")
        if errors:
            if not isinstance(errors, list):
                raise ApiError("GraphQL errors field is not an array")
            message = compact_errors(errors)
            if is_capacity_failure(message):
                raise CapacityError(message)
            raise ApiError(f"GraphQL {operation} failed: {message}")
        return require_mapping(result.get("data"), f"GraphQL {operation} data")

    def rest(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        if not path.startswith("/"):
            raise ValueError("REST path must be absolute")
        return self._request(method, f"{self._rest_url}{path}", body)

    def _request(self, method: str, url: str, body: dict[str, Any] | None) -> Any:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        request = urllib.request.Request(
            url,
            data=encoded,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "nml-runpod/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(16 * 1024).decode(errors="replace")
            raise ApiError(f"{method} {url} returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ApiError(f"{method} {url} failed: {error.reason}") from error
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ApiError(f"{method} {url} exceeded the response size limit")
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise ApiError(f"{method} {url} returned invalid JSON") from error


def create_pod_document(
    cloud: str,
    with_data_center: bool,
    with_template: bool = False,
    with_network_volume: bool = False,
    contract_inputs: tuple[str, ...] = (),
) -> str:
    if cloud not in {"SECURE", "COMMUNITY", "ALL"}:
        raise ValueError(f"unsupported cloud policy {cloud!r}")
    if with_network_volume and not with_data_center:
        raise ValueError("network volume requires explicit data-center placement")
    if len(set(contract_inputs)) != len(contract_inputs):
        raise ValueError("contract input environment names must be unique")
    names = tuple(require_contract_inputs({name: "/workspace/input" for name in contract_inputs}))
    if names and not with_network_volume:
        raise ValueError("contract inputs require an attached network volume")
    return _CREATE_POD.format(
        cloud=cloud,
        image_variable="" if with_template else "$image: String!",
        template_variable="$templateId: String!" if with_template else "",
        artifact_fields=(
            "templateId: $templateId" if with_template else "imageName: $image"
        ),
        data_center_variable="$dataCenter: String!" if with_data_center else "",
        data_center_field="dataCenterId: $dataCenter" if with_data_center else "",
        network_volume_variable=(
            "$networkVolumeId: String!" if with_network_volume else ""
        ),
        network_volume_field=(
            "networkVolumeId: $networkVolumeId" if with_network_volume else ""
        ),
        contract_input_variables="\n  ".join(
            f"$contractInput{index}: String!" for index, _ in enumerate(names)
        ),
        contract_input_env="\n      ".join(
            f'{{key: "{name}", value: $contractInput{index}}}'
            for index, name in enumerate(names)
        ),
        volume_mount_path=NETWORK_VOLUME_MOUNT_PATH,
    )


def create_ssh_job_pod_document(cloud: str, with_data_center: bool) -> str:
    if cloud not in {"SECURE", "COMMUNITY", "ALL"}:
        raise ValueError(f"unsupported cloud policy {cloud!r}")
    return _CREATE_SSH_JOB_POD.format(
        cloud=cloud,
        data_center_variable="$dataCenter: String!" if with_data_center else "",
        data_center_field="dataCenterId: $dataCenter" if with_data_center else "",
    )


def require_network_volume_id(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError(
            "network volume id must be non-empty without surrounding whitespace"
        )
    return value


def require_contract_inputs(values: dict[str, str]) -> dict[str, str]:
    """Validates opaque, volume-backed inputs without knowing product policy."""
    normalized: dict[str, str] = {}
    for name in sorted(values):
        if not name.startswith("NML_") or ENVIRONMENT_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid contract input environment name {name!r}")
        if name in RESERVED_RUNNER_ENVIRONMENT:
            raise ValueError(f"contract input may not override runner identity {name!r}")
        normalized[name] = require_workspace_path(f"contract input {name}", values[name])
    return normalized


def require_workspace_path(name: str, value: str | None) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty without surrounding whitespace")
    path = PurePosixPath(value)
    workspace = PurePosixPath(NETWORK_VOLUME_MOUNT_PATH)
    if not path.is_absolute() or path == workspace or workspace not in path.parents:
        raise ValueError(
            f"{name} must be an absolute child of {NETWORK_VOLUME_MOUNT_PATH}"
        )
    if ".." in path.parts or str(path) != value:
        raise ValueError(f"{name} must be canonical and must not contain '..'")
    return value


def ssh_endpoint(pod: dict[str, Any]) -> tuple[str, int] | None:
    runtime = pod.get("runtime")
    if not isinstance(runtime, dict):
        return None
    ports = runtime.get("ports")
    if not isinstance(ports, list):
        return None
    for candidate in ports:
        if not isinstance(candidate, dict):
            continue
        if (
            candidate.get("privatePort") == 22
            and str(candidate.get("type", "")).lower() == "tcp"
            and candidate.get("isIpPublic") is True
            and isinstance(candidate.get("ip"), str)
            and candidate["ip"]
            and type(candidate.get("publicPort")) is int
            and candidate["publicPort"] > 0
        ):
            return candidate["ip"], candidate["publicPort"]
    return None


def same_exact_image(left: str, right: str) -> bool:
    """Compares digest references while allowing Docker Hub's host normalization."""
    if EXACT_IMAGE.fullmatch(left) is None or EXACT_IMAGE.fullmatch(right) is None:
        return False
    left_name, left_digest = left.rsplit("@", 1)
    right_name, right_digest = right.rsplit("@", 1)
    return (
        left_name.removeprefix("docker.io/") == right_name.removeprefix("docker.io/")
        and left_digest == right_digest
    )


def validate_template(template: dict[str, Any], spec: TemplateSpec) -> None:
    expected = spec.payload()
    compared_fields = (
        "imageName",
        "containerDiskInGb",
        "dockerEntrypoint",
        "dockerStartCmd",
        "env",
        "isPublic",
        "isServerless",
        "ports",
        "volumeInGb",
        "volumeMountPath",
    )
    drift = [field for field in compared_fields if template.get(field) != expected[field]]
    if drift:
        raise ApiError(
            f"template {spec.name!r} drifts in {', '.join(drift)}; "
            "use a new name or explicitly replace it in RunPod"
        )
    require_string(template.get("id"), "template id")


def is_capacity_failure(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "capacity",
            "gpu type not found",
            "insufficient",
            "no available",
            "no instances",
            "no longer any instances available",
            "does not have the resources to deploy",
            "out of stock",
            "unable to find a machine",
        )
    )


def compact_errors(errors: list[Any]) -> str:
    messages = []
    for error in errors:
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            messages.append(error["message"])
        else:
            messages.append(json.dumps(error, sort_keys=True))
    return "; ".join(messages)


def require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ApiError(f"{context} is not a JSON object")
    return value


def require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ApiError(f"{context} is not a non-empty string")
    return value
