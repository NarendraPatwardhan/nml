"""Durable, concurrent RunPod lease records outside the source tree."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


SCHEMA_VERSION = 2


@dataclass
class Lease:
    lease_id: str
    state: str
    image: str
    image_digest: str
    source_commit: str
    source_dirty: bool
    requested_gpus: list[str]
    created_at: str
    deadline_at: str
    pod_id: str | None = None
    template_id: str | None = None
    network_volume_id: str | None = None
    network_volume_data_center: str | None = None
    network_volume_mount_path: str | None = None
    contract_inputs: dict[str, str] = field(default_factory=dict)
    machine_id: str | None = None
    allocated_gpu: str | None = None
    application_url: str | None = None
    hourly_price: float | None = None
    terminal_result: dict[str, Any] | None = None
    cleanup_error: str | None = None
    events: list[dict[str, str]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    lease_token: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.network_volume_id is None:
            if any(
                value is not None
                for value in (
                    self.network_volume_data_center,
                    self.network_volume_mount_path,
                    self.contract_inputs or None,
                )
            ):
                raise ValueError("network-volume metadata requires a volume id")
            return
        if self.network_volume_data_center is None:
            raise ValueError("network-volume lease omitted its data center")
        if self.network_volume_mount_path is None:
            raise ValueError("network-volume lease omitted its mount path")

    @classmethod
    def create(
        cls,
        *,
        image: str,
        image_digest: str,
        source_commit: str,
        source_dirty: bool,
        requested_gpus: list[str],
        deadline_at: datetime,
        lease_token: str,
        template_id: str | None = None,
        network_volume_id: str | None = None,
        network_volume_data_center: str | None = None,
        network_volume_mount_path: str | None = None,
        contract_inputs: dict[str, str] | None = None,
    ) -> "Lease":
        now = datetime.now(UTC)
        lease = cls(
            lease_id=str(uuid4()),
            state="allocating",
            image=image,
            image_digest=image_digest,
            source_commit=source_commit,
            source_dirty=source_dirty,
            requested_gpus=list(requested_gpus),
            created_at=isoformat(now),
            deadline_at=isoformat(deadline_at),
            template_id=template_id,
            network_volume_id=network_volume_id,
            network_volume_data_center=network_volume_data_center,
            network_volume_mount_path=network_volume_mount_path,
            contract_inputs=dict(contract_inputs or {}),
            lease_token=lease_token,
        )
        lease.record("allocating", "placement requested")
        return lease

    def record(self, state: str, detail: str) -> None:
        self.state = state
        self.events.append(
            {"at": isoformat(datetime.now(UTC)), "state": state, "detail": detail}
        )

    def network_volume_identity(self) -> dict[str, object] | None:
        if self.network_volume_id is None:
            return None
        assert self.network_volume_data_center is not None
        assert self.network_volume_mount_path is not None
        identity = {
            "id": self.network_volume_id,
            "data_center": self.network_volume_data_center,
            "mount_path": self.network_volume_mount_path,
        }
        if self.contract_inputs:
            identity["contract_inputs"] = dict(self.contract_inputs)
        return identity

    def public_record(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("lease_token", None)
        result["network_volume"] = self.network_volume_identity()
        return result


class LeaseStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_state_root()

    def save(self, lease: Lease) -> Path:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        destination = self.path(lease.lease_id)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{lease.lease_id}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(asdict(lease), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load(self, lease_id: str) -> Lease:
        validate_lease_id(lease_id)
        try:
            payload = json.loads(self.path(lease_id).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"unknown RunPod lease {lease_id!r}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"lease {lease_id!r} has an unsupported schema")
        return Lease(**payload)

    def list(self) -> list[Lease]:
        if not self.root.exists():
            return []
        leases = []
        for path in sorted(self.root.glob("*.json")):
            leases.append(self.load(path.stem))
        return leases

    def path(self, lease_id: str) -> Path:
        validate_lease_id(lease_id)
        return self.root / f"{lease_id}.json"


def default_state_root() -> Path:
    if value := os.environ.get("XDG_STATE_HOME"):
        return Path(value).expanduser() / "nml" / "runpod" / "leases"
    return Path.home() / ".local" / "state" / "nml" / "runpod" / "leases"


def validate_lease_id(value: str) -> None:
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"invalid lease id {value!r}") from error
    if str(parsed) != value:
        raise ValueError(f"lease id must use canonical lowercase UUID form: {value!r}")


def isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
