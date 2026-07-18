"""Produce the pinned GPT-OSS NVFP4 artifact on one ephemeral RunPod GPU.

The controller sends secrets over SSH rather than Pod configuration, follows
the converter's durable JSON-lines log, and owns Pod termination in a ``finally``
block. A small local state record preserves the Pod id for explicit recovery if
the controller process is killed before Python can execute that cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api import RunPodClient, ssh_endpoint


DEFAULT_IMAGE = (
    "docker.io/runpod/pytorch@"
    "sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35"
)
DEFAULT_GPUS = (
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
)
REMOTE_ROOT = "/workspace/nvfp4"
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=6",
    "-o",
    "ConnectTimeout=30",
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="convert, publish, and terminate the worker")
    run.add_argument("--hf-token", type=Path, default=Path("/mnt/workspace/hf.gptoss.key"))
    run.add_argument("--ssh-public-key", type=Path, default=Path.home() / ".ssh/id_ed25519.pub")
    run.add_argument("--ssh-private-key", type=Path, default=Path.home() / ".ssh/id_ed25519")
    run.add_argument("--source-manifest", type=Path, default=artifact_file("source.json"))
    run.add_argument(
        "--tensor-manifest", type=Path, default=artifact_file("source-tensors.json")
    )
    run.add_argument("--recipe", type=Path, default=artifact_file("recipe.json"))
    run.add_argument("--destination-repository", default="narendra747/gpt-oss-20b-nvfp4")
    run.add_argument("--image", default=DEFAULT_IMAGE)
    run.add_argument("--gpu", action="append", dest="gpus")
    run.add_argument("--cloud", choices=("SECURE", "COMMUNITY", "ALL"), default="COMMUNITY")
    run.add_argument("--data-center")
    run.add_argument("--container-disk-gb", type=positive_integer, default=100)
    run.add_argument("--chunk-rows", type=positive_integer, default=2048)
    run.add_argument("--provision-timeout", type=positive_integer, default=1200)
    run.add_argument("--conversion-timeout", type=positive_integer, default=14400)
    run.add_argument("--state", type=Path, default=default_state_file())
    run.set_defaults(handler=run_conversion)

    terminate = commands.add_parser(
        "terminate", help="terminate the Pod recorded by an interrupted controller"
    )
    terminate.add_argument("--state", type=Path, default=default_state_file())
    terminate.set_defaults(handler=terminate_stale)
    return root


def run_conversion(arguments: argparse.Namespace) -> int:
    require_regular_secret(arguments.hf_token, "Hugging Face token")
    require_regular_file(arguments.ssh_public_key, "SSH public key")
    require_regular_secret(arguments.ssh_private_key, "SSH private key")
    for path, label in (
        (arguments.source_manifest, "source manifest"),
        (arguments.tensor_manifest, "tensor manifest"),
        (arguments.recipe, "recipe"),
    ):
        require_regular_file(path, label)
    if arguments.state.exists():
        raise RuntimeError(
            f"state already exists at {arguments.state}; recover it with the terminate command"
        )

    client = RunPodClient(runpod_key_from_environment())
    created = None
    cleanup_error = None
    try:
        emit("placement_started", gpus=arguments.gpus or list(DEFAULT_GPUS))
        created = client.create_ssh_job_pod(
            name=f"gpt-oss-20b-nvfp4-{int(time.time())}",
            image=arguments.image,
            gpu_types=arguments.gpus or list(DEFAULT_GPUS),
            gpu_count=1,
            cloud=arguments.cloud,
            container_disk_gb=arguments.container_disk_gb,
            public_key=arguments.ssh_public_key.read_text(encoding="utf-8").strip(),
            data_center=arguments.data_center,
        )
        write_state(
            arguments.state,
            {
                "schema_version": 1,
                "pod_id": created.pod_id,
                "image": created.image,
                "gpu": created.requested_gpu,
                "created_at": timestamp(),
            },
        )
        emit(
            "pod_created",
            pod_id=created.pod_id,
            gpu=created.requested_gpu,
            image=created.image,
        )
        endpoint = wait_for_ssh_endpoint(
            client, created.pod_id, arguments.provision_timeout
        )
        update_state(arguments.state, ssh_ip=endpoint[0], ssh_port=endpoint[1])
        emit("ssh_mapped", ip=endpoint[0], port=endpoint[1])
        wait_for_ssh(endpoint, arguments.ssh_private_key, arguments.provision_timeout)
        upload_inputs(endpoint, arguments)
        run_ssh(
            endpoint,
            arguments.ssh_private_key,
            [
                "python",
                "-m",
                "venv",
                "--system-site-packages",
                f"{REMOTE_ROOT}/venv",
            ],
        )
        run_ssh(
            endpoint,
            arguments.ssh_private_key,
            [
                f"{REMOTE_ROOT}/venv/bin/python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                f"{REMOTE_ROOT}/requirements.txt",
            ],
        )
        launch_worker(endpoint, arguments)
        status = follow_conversion(
            endpoint,
            arguments.ssh_private_key,
            arguments.conversion_timeout,
        )
        if status.get("exit_code") != 0:
            raise RuntimeError(f"remote converter exited with {status.get('exit_code')!r}")
        emit("conversion_succeeded", repository=arguments.destination_repository)
        return 0
    finally:
        if created is not None:
            try:
                terminate_and_confirm(client, created.pod_id)
                arguments.state.unlink(missing_ok=True)
                emit("pod_terminated", pod_id=created.pod_id)
            except Exception as error:  # Preserve the recovery record on cleanup failure.
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                f"Pod may still be billable; recover with the terminate command: {cleanup_error}"
            ) from cleanup_error


def terminate_stale(arguments: argparse.Namespace) -> int:
    state = read_object(arguments.state)
    pod_id = state.get("pod_id")
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError("recovery state has no Pod id")
    terminate_and_confirm(RunPodClient(runpod_key_from_environment()), pod_id)
    arguments.state.unlink()
    emit("pod_terminated", pod_id=pod_id)
    return 0


def wait_for_ssh_endpoint(
    client: RunPodClient, pod_id: str, timeout: int
) -> tuple[str, int]:
    deadline = time.monotonic() + timeout
    previous_status = None
    while time.monotonic() < deadline:
        pod = client.pod(pod_id)
        if pod is None:
            raise RuntimeError("RunPod lost the Pod during provisioning")
        status = pod.get("desiredStatus")
        if status != previous_status:
            emit("pod_status", status=status)
            previous_status = status
        if status in {"EXITED", "TERMINATED"}:
            raise RuntimeError(f"Pod entered terminal state {status!r}")
        if endpoint := ssh_endpoint(pod):
            return endpoint
        time.sleep(5)
    raise RuntimeError(f"RunPod did not publish SSH mapping within {timeout} seconds")


def wait_for_ssh(endpoint: tuple[str, int], private_key: Path, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        completed = run_ssh(endpoint, private_key, ["true"], check=False, capture=True)
        if completed.returncode == 0:
            emit("ssh_ready")
            return
        last_error = completed.stderr.strip()
        time.sleep(5)
    raise RuntimeError(f"SSH did not become ready within {timeout} seconds: {last_error}")


def upload_inputs(endpoint: tuple[str, int], arguments: argparse.Namespace) -> None:
    run_ssh(endpoint, arguments.ssh_private_key, ["mkdir", "-p", REMOTE_ROOT])
    sources = (
        tool_file("convert.py"),
        tool_file("worker.py"),
        tool_file("requirements.txt"),
        arguments.source_manifest,
        arguments.tensor_manifest,
        arguments.recipe,
        arguments.hf_token,
    )
    command = [
        "scp",
        *SSH_OPTIONS,
        "-i",
        str(arguments.ssh_private_key),
        "-P",
        str(endpoint[1]),
        *(str(path) for path in sources),
        f"root@{endpoint[0]}:{REMOTE_ROOT}/",
    ]
    subprocess.run(command, check=True)
    # The local filename is intentionally not coupled to the operator's token
    # filename. The converter deletes this remote copy immediately after read.
    remote_token = f"{REMOTE_ROOT}/{arguments.hf_token.name}"
    run_ssh(endpoint, arguments.ssh_private_key, ["chmod", "600", remote_token])
    emit("inputs_uploaded", files=len(sources))


def launch_worker(endpoint: tuple[str, int], arguments: argparse.Namespace) -> None:
    converter_arguments = [
        "--source-manifest",
        f"{REMOTE_ROOT}/{arguments.source_manifest.name}",
        "--tensor-manifest",
        f"{REMOTE_ROOT}/{arguments.tensor_manifest.name}",
        "--recipe",
        f"{REMOTE_ROOT}/{arguments.recipe.name}",
        "--work-directory",
        f"{REMOTE_ROOT}/work",
        "--destination-repository",
        arguments.destination_repository,
        "--hf-token-file",
        f"{REMOTE_ROOT}/{arguments.hf_token.name}",
        "--chunk-rows",
        str(arguments.chunk_rows),
    ]
    worker = [
        f"{REMOTE_ROOT}/venv/bin/python",
        "worker.py",
        "--status",
        "worker-status.json",
        "--log",
        "conversion.log",
        "--",
        *converter_arguments,
    ]
    command = (
        f"cd {shlex.quote(REMOTE_ROOT)} && rm -f worker-status.json conversion.log "
        f"&& nohup {shlex.join(worker)} </dev/null >worker-launch.log 2>&1 &"
    )
    run_ssh(endpoint, arguments.ssh_private_key, ["bash", "-lc", command])
    emit("conversion_started")


def follow_conversion(
    endpoint: tuple[str, int], private_key: Path, timeout: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    emitted_lines = 0
    while time.monotonic() < deadline:
        log = run_ssh(
            endpoint,
            private_key,
            ["bash", "-lc", f"test -f {REMOTE_ROOT}/conversion.log && cat {REMOTE_ROOT}/conversion.log || true"],
            capture=True,
        ).stdout
        lines = log.splitlines()
        for line in lines[emitted_lines:]:
            print(line, flush=True)
        emitted_lines = len(lines)
        status = run_ssh(
            endpoint,
            private_key,
            ["bash", "-lc", f"test -f {REMOTE_ROOT}/worker-status.json && cat {REMOTE_ROOT}/worker-status.json"],
            check=False,
            capture=True,
        )
        if status.returncode == 0:
            return json.loads(status.stdout)
        time.sleep(10)
    raise RuntimeError(f"conversion did not finish within {timeout} seconds")


def run_ssh(
    endpoint: tuple[str, int],
    private_key: Path,
    remote_arguments: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        "ssh",
        *SSH_OPTIONS,
        "-i",
        str(private_key),
        "-p",
        str(endpoint[1]),
        f"root@{endpoint[0]}",
        shlex.join(remote_arguments),
    ]
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
    )


def terminate_and_confirm(client: RunPodClient, pod_id: str) -> None:
    client.terminate(pod_id)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        pod = client.pod(pod_id)
        if pod is None or pod.get("desiredStatus") == "TERMINATED":
            return
        time.sleep(3)
    raise RuntimeError("RunPod did not confirm termination within 90 seconds")


def require_regular_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")


def require_regular_secret(path: Path, label: str) -> None:
    require_regular_file(path, label)
    if not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"{label} is empty: {path}")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(f"{label} must not be accessible by group or other: {path}")


def runpod_key_from_environment() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        raise ValueError("RUNPOD_API_KEY must be set; the controller never reads it from disk")
    return value


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return value


def write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def update_state(path: Path, **updates: object) -> None:
    state = read_object(path)
    state.update(updates)
    write_state(path, state)


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def artifact_file(name: str) -> Path:
    return Path("artifacts/gpt-oss-20b-nvfp4") / name


def tool_file(name: str) -> Path:
    return Path("tools/nvfp4") / name


def default_state_file() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    return root / "nml/runpod/nvfp4-conversion.json"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.handler(arguments)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"nvfp4-runpod: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
