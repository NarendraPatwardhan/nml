"""One-owner lifecycle for NML device-contract Pods."""

from __future__ import annotations

import json
import secrets
import signal
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from api import (
    NETWORK_VOLUME_MOUNT_PATH,
    RunPodClient,
    require_contract_inputs,
    require_network_volume_id,
)
from lease import Lease, LeaseStore


RUNNER_PORT = 8080
RUNNER_SCHEMA_VERSION = 1
MAX_APPLICATION_RESPONSE_BYTES = 2 * 1024 * 1024
TERMINATION_CONFIRMATION_SECONDS = 90


class ControllerError(RuntimeError):
    pass


class ControllerInterrupted(ControllerError):
    pass


@dataclass(frozen=True)
class ContractRun:
    image: str
    image_digest: str
    source_commit: str
    source_dirty: bool
    gpu_types: list[str]
    gpu_count: int
    cloud: str
    data_center: str | None
    container_disk_gb: int
    contracts: list[str]
    per_contract_timeout_seconds: int
    total_timeout_seconds: int
    control_plane_timeout_seconds: int
    readiness_timeout_seconds: int
    template_id: str | None
    network_volume_id: str | None = None
    contract_inputs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.network_volume_id is not None:
            require_network_volume_id(self.network_volume_id)
            if self.data_center is None:
                raise ValueError("network-volume contract runs require a data center")
        inputs = require_contract_inputs(self.contract_inputs)
        if inputs and self.network_volume_id is None:
            raise ValueError("contract inputs require an attached network volume")
        object.__setattr__(self, "contract_inputs", inputs)
        object.__setattr__(self, "gpu_types", list(self.gpu_types))
        object.__setattr__(self, "contracts", list(self.contracts))


def execute_contracts(
    client: RunPodClient,
    store: LeaseStore,
    request: ContractRun,
) -> tuple[Lease, bool]:
    started = datetime.now(UTC)
    deadline = started + timedelta(
        seconds=request.control_plane_timeout_seconds
        + request.readiness_timeout_seconds
        + request.total_timeout_seconds
        + TERMINATION_CONFIRMATION_SECONDS
    )
    token = secrets.token_urlsafe(32)
    lease = Lease.create(
        image=request.image,
        image_digest=request.image_digest,
        source_commit=request.source_commit,
        source_dirty=request.source_dirty,
        requested_gpus=request.gpu_types,
        deadline_at=deadline,
        lease_token=token,
        template_id=request.template_id,
        network_volume_id=request.network_volume_id,
        network_volume_data_center=(
            request.data_center if request.network_volume_id else None
        ),
        network_volume_mount_path=(
            NETWORK_VOLUME_MOUNT_PATH if request.network_volume_id else None
        ),
        contract_inputs=request.contract_inputs,
    )
    store.save(lease)
    previous_handlers = install_signal_handlers()
    try:
        created = client.create_device_contract_pod(
            name=f"nml-contracts-{lease.lease_id[:8]}",
            image=request.image,
            gpu_types=request.gpu_types,
            gpu_count=request.gpu_count,
            cloud=request.cloud,
            container_disk_gb=request.container_disk_gb,
            lease_token=token,
            image_digest=request.image_digest,
            source_commit=request.source_commit,
            source_dirty=request.source_dirty,
            data_center=request.data_center,
            template_id=request.template_id,
            network_volume_id=request.network_volume_id,
            contract_inputs=request.contract_inputs,
        )
        lease.pod_id = created.pod_id
        lease.machine_id = created.machine_id
        lease.allocated_gpu = created.requested_gpu
        lease.application_url = (
            f"https://{created.pod_id}-{RUNNER_PORT}.proxy.runpod.net"
        )
        lease.record("provisioning", f"Pod {created.pod_id} allocated")
        store.save(lease)

        pod = wait_for_control_plane(
            client,
            lease,
            store,
            timeout=request.control_plane_timeout_seconds,
        )
        price = pod.get("costPerHr")
        if isinstance(price, (int, float)):
            lease.hourly_price = float(price)
        wait_for_readiness(
            lease,
            store,
            timeout=request.readiness_timeout_seconds,
        )
        submit_run(lease, request)
        lease.record("executing", "runner accepted the immutable contract selection")
        store.save(lease)
        result = wait_for_result(
            lease,
            timeout=request.total_timeout_seconds + 30,
        )
        validate_result_identity(lease, result)
        record_network_volume_result(lease, result)
        lease.terminal_result = result
        success = result.get("status") == "succeeded"
        lease.record(
            "contract_succeeded" if success else "contract_failed",
            f"runner returned terminal status {result.get('status')!r}",
        )
        store.save(lease)
        return lease, success
    except BaseException as error:
        lease.record("controller_failed", safe_error(error))
        store.save(lease)
        raise
    finally:
        ignore_termination_signals()
        try:
            if lease.pod_id is not None:
                terminate_and_confirm(client, lease, store)
        finally:
            restore_signal_handlers(previous_handlers)


def wait_for_control_plane(
    client: RunPodClient,
    lease: Lease,
    store: LeaseStore,
    *,
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pod = client.pod(required_pod_id(lease))
        if pod is None:
            raise ControllerError("RunPod lost the Pod before it became RUNNING")
        status = pod.get("desiredStatus")
        if status == "RUNNING" and isinstance(pod.get("runtime"), dict):
            lease.record("control_plane_running", "RunPod runtime metadata is present")
            store.save(lease)
            return pod
        if status in {"EXITED", "TERMINATED"}:
            raise ControllerError(f"Pod entered terminal control-plane state {status!r}")
        time.sleep(5)
    raise ControllerError(f"Pod did not reach RUNNING within {timeout} seconds")


def wait_for_readiness(lease: Lease, store: LeaseStore, *, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, payload = runner_request(lease, "GET", "/ready")
            if status == 200:
                validate_runner_identity(lease, payload)
                if payload.get("ready") is not True:
                    raise ControllerError("runner returned HTTP 200 without ready=true")
                validate_hardware_inventory(payload)
                lease.record("application_ready", "authenticated runner readiness passed")
                store.save(lease)
                return
            last_error = f"HTTP {status}: {payload}"
        except (ControllerError, urllib.error.URLError) as error:
            last_error = safe_error(error)
        time.sleep(5)
    raise ControllerError(
        f"runner did not become ready within {timeout} seconds: {last_error}"
    )


def submit_run(lease: Lease, request: ContractRun) -> None:
    body = {
        "contracts": request.contracts,
        "per_contract_timeout_seconds": request.per_contract_timeout_seconds,
        "total_timeout_seconds": request.total_timeout_seconds,
    }
    try:
        status, _ = runner_request(lease, "POST", "/run", body)
    except urllib.error.URLError:
        # A lost response is not permission to execute twice. The immutable
        # runner state tells us whether the first request took ownership.
        state_status, state = runner_request(lease, "GET", "/state")
        if state_status == 200 and state.get("state") in {"running", "terminal"}:
            return
        raise
    if status != 202:
        raise ControllerError(f"runner rejected the contract set with HTTP {status}")


def wait_for_result(lease: Lease, *, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, payload = runner_request(lease, "GET", "/result")
            if status == 200:
                return payload
            if status != 404:
                last_error = f"HTTP {status}: {payload}"
        except (ControllerError, urllib.error.URLError) as error:
            last_error = safe_error(error)
        time.sleep(5)
    raise ControllerError(f"runner result deadline expired: {last_error}")


def terminate_and_confirm(
    client: RunPodClient, lease: Lease, store: LeaseStore
) -> None:
    pod_id = required_pod_id(lease)
    lease.record("termination_pending", f"terminating Pod {pod_id}")
    store.save(lease)
    try:
        client.terminate(pod_id)
        deadline = time.monotonic() + TERMINATION_CONFIRMATION_SECONDS
        while time.monotonic() < deadline:
            pod = client.pod(pod_id)
            if pod is None or pod.get("desiredStatus") == "TERMINATED":
                lease.record("terminated", f"Pod {pod_id} termination confirmed")
                lease.cleanup_error = None
                store.save(lease)
                return
            time.sleep(3)
        raise ControllerError("termination was not confirmed before the cleanup deadline")
    except Exception as error:
        lease.cleanup_error = safe_error(error)
        lease.record(
            "orphaned",
            f"Pod {pod_id} may still be billable: {lease.cleanup_error}",
        )
        store.save(lease)
        raise ControllerError(
            f"Pod {pod_id} may still be billable; recover with terminate {lease.lease_id}"
        ) from error


def runner_request(
    lease: Lease,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    if lease.application_url is None:
        raise ControllerError("lease has no application URL")
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{lease.application_url}{path}",
        data=encoded,
        method=method,
        headers={
            "Authorization": f"Bearer {lease.lease_token}",
            "Content-Type": "application/json",
            "User-Agent": "nml-runpod/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
            payload = response.read(MAX_APPLICATION_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        payload = error.read(MAX_APPLICATION_RESPONSE_BYTES + 1)
    if len(payload) > MAX_APPLICATION_RESPONSE_BYTES:
        raise ControllerError("runner response exceeded the size limit")
    try:
        value = json.loads(payload) if payload else {}
    except json.JSONDecodeError as error:
        raise ControllerError(f"runner {path} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ControllerError(f"runner {path} response is not a JSON object")
    return status, value


def validate_runner_identity(lease: Lease, payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise ControllerError("runner uses an unsupported result schema")
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        raise ControllerError("runner readiness omitted artifact identity")
    expected = {
        "image_digest": lease.image_digest,
        "source_commit": lease.source_commit,
        "source_dirty": lease.source_dirty,
    }
    if artifact != expected:
        raise ControllerError(f"runner artifact identity {artifact!r} != {expected!r}")


def validate_result_identity(lease: Lease, result: dict[str, Any]) -> None:
    validate_runner_identity(lease, result)
    validate_hardware_inventory(result)
    if result.get("status") not in {"succeeded", "failed", "timed_out", "interrupted"}:
        raise ControllerError("runner result has an unknown terminal status")


def record_network_volume_result(lease: Lease, result: dict[str, Any]) -> None:
    """Adds controller-owned persistent-storage provenance to a runner result."""
    volume = lease.network_volume_identity()
    if volume is not None:
        result["network_volume"] = volume


def validate_hardware_inventory(payload: dict[str, Any]) -> None:
    hardware = payload.get("hardware")
    if not isinstance(hardware, list) or not hardware:
        raise ControllerError("runner result omitted its non-empty GPU inventory")
    for index, device in enumerate(hardware):
        if not isinstance(device, dict):
            raise ControllerError(f"runner GPU {index} is not an identity object")
        if type(device.get("index")) is not int or device["index"] < 0:
            raise ControllerError(f"runner GPU {index} has an invalid device index")
        for field in ("name", "uuid", "compute_capability", "driver_version"):
            if not isinstance(device.get(field), str) or not device[field]:
                raise ControllerError(f"runner GPU {index} has an invalid {field}")


def recover_termination(
    client: RunPodClient, store: LeaseStore, lease_id: str
) -> Lease:
    lease = store.load(lease_id)
    if lease.pod_id is None:
        raise ControllerError("lease never allocated a Pod")
    terminate_and_confirm(client, lease, store)
    return lease


def required_pod_id(lease: Lease) -> str:
    if lease.pod_id is None:
        raise ControllerError("lease has no Pod id")
    return lease.pod_id


def safe_error(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted by operator"
    return f"{type(error).__name__}: {error}"


def install_signal_handlers() -> dict[int, Any]:
    previous = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.getsignal(number)
        signal.signal(number, interrupt)
    return previous


def restore_signal_handlers(previous: dict[int, Any]) -> None:
    for number, handler in previous.items():
        signal.signal(number, handler)


def ignore_termination_signals() -> None:
    # Once a Pod exists, a second interrupt must not bypass the billable
    # cleanup boundary. The original interrupt is already recorded in the
    # lease; cleanup restores the operator's handlers before returning.
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, signal.SIG_IGN)


def interrupt(number: int, _frame: Any) -> None:
    raise ControllerInterrupted(f"received signal {number}")
