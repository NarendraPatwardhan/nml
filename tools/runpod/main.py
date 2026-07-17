"""Run exact NML OCI artifacts on ephemeral RunPod GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from api import RunPodClient, TemplateSpec, require_string
from controller import ContractRun, execute_contracts, recover_termination
from lease import LeaseStore


DEFAULT_GPU_TYPES = [
    "NVIDIA RTX A6000",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA A40",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 4090",
]
DEFAULT_CONTRACTS = [
    "flash_attention_device_capability",
    "cuda_runtime",
    "linear",
    "attention",
    "neural_ops",
    "execution_performance",
]
DIGEST_IMAGE = re.compile(r"^([^\s@]+)@(sha256:[0-9a-f]{64})$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("contracts", help="run permanent CUDA contracts")
    run.add_argument("--image", required=True, help="exact registry image@sha256:digest")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--source-dirty", action="store_true")
    run.add_argument("--gpu", action="append", dest="gpus")
    run.add_argument("--gpu-count", type=positive_integer, default=1)
    run.add_argument("--cloud", choices=("SECURE", "COMMUNITY", "ALL"), default="SECURE")
    run.add_argument("--data-center")
    run.add_argument("--container-disk-gb", type=positive_integer, default=20)
    run.add_argument(
        "--template-name",
        help="optional exact private device-contract template; direct creation is default",
    )
    run.add_argument("--contract", action="append", dest="contracts")
    run.add_argument("--per-contract-timeout", type=positive_integer, default=900)
    run.add_argument("--total-timeout", type=positive_integer, default=3600)
    run.add_argument("--control-plane-timeout", type=positive_integer, default=900)
    run.add_argument("--readiness-timeout", type=positive_integer, default=600)
    run.set_defaults(handler=run_contracts)

    template = commands.add_parser(
        "template", help="create or validate one exact private contract template"
    )
    template.add_argument("--name", required=True)
    template.add_argument("--image", required=True, help="exact registry image@sha256:digest")
    template.add_argument("--container-disk-gb", type=positive_integer, default=20)
    template.set_defaults(handler=ensure_template)

    status = commands.add_parser("status", help="show one durable lease and live Pod state")
    status.add_argument("lease_id")
    status.set_defaults(handler=show_status)

    leases = commands.add_parser("leases", help="list durable local lease records")
    leases.set_defaults(handler=list_leases)

    terminate = commands.add_parser("terminate", help="idempotently recover and terminate a Pod")
    terminate.add_argument("lease_id")
    terminate.set_defaults(handler=terminate_lease)
    return root


def run_contracts(arguments: argparse.Namespace) -> int:
    match = require_digest_image(arguments.image)
    if COMMIT.fullmatch(arguments.source_commit) is None:
        raise ValueError("--source-commit must be a lowercase full Git commit")
    client = client_from_environment()
    template_id = None
    if arguments.template_name:
        template, _ = client.ensure_template(
            TemplateSpec(
                name=arguments.template_name,
                image=arguments.image,
                container_disk_gb=arguments.container_disk_gb,
            )
        )
        template_id = require_string(template.get("id"), "template id")
    lease, success = execute_contracts(
        client,
        LeaseStore(),
        ContractRun(
            image=arguments.image,
            image_digest=match.group(2),
            source_commit=arguments.source_commit,
            source_dirty=arguments.source_dirty,
            gpu_types=arguments.gpus or DEFAULT_GPU_TYPES,
            gpu_count=arguments.gpu_count,
            cloud=arguments.cloud,
            data_center=arguments.data_center,
            container_disk_gb=arguments.container_disk_gb,
            contracts=arguments.contracts or DEFAULT_CONTRACTS,
            per_contract_timeout_seconds=arguments.per_contract_timeout,
            total_timeout_seconds=arguments.total_timeout,
            control_plane_timeout_seconds=arguments.control_plane_timeout,
            readiness_timeout_seconds=arguments.readiness_timeout,
            template_id=template_id,
        ),
    )
    print(json.dumps(public_lease(lease), indent=2, sort_keys=True))
    return 0 if success else 1


def ensure_template(arguments: argparse.Namespace) -> int:
    require_digest_image(arguments.image)
    template, created = client_from_environment().ensure_template(
        TemplateSpec(
            name=arguments.name,
            image=arguments.image,
            container_disk_gb=arguments.container_disk_gb,
        )
    )
    print(
        json.dumps(
            {
                "created": created,
                "id": require_string(template.get("id"), "template id"),
                "image": arguments.image,
                "name": arguments.name,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def show_status(arguments: argparse.Namespace) -> int:
    lease = LeaseStore().load(arguments.lease_id)
    output = {"lease": public_lease(lease), "pod": None}
    if lease.pod_id and lease.state != "terminated":
        output["pod"] = client_from_environment().pod(lease.pod_id)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def list_leases(_arguments: argparse.Namespace) -> int:
    print(
        json.dumps(
            [public_lease(lease) for lease in LeaseStore().list()],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def terminate_lease(arguments: argparse.Namespace) -> int:
    lease = recover_termination(
        client_from_environment(), LeaseStore(), arguments.lease_id
    )
    print(json.dumps(public_lease(lease), indent=2, sort_keys=True))
    return 0


def client_from_environment() -> RunPodClient:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise ValueError("RUNPOD_API_KEY must be set in the operator environment")
    return RunPodClient(key)


def public_lease(lease: object) -> dict[str, object]:
    result = dict(vars(lease))
    result.pop("lease_token", None)
    return result


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def require_digest_image(value: str) -> re.Match[str]:
    match = DIGEST_IMAGE.fullmatch(value)
    if match is None:
        raise ValueError("--image must be an exact registry image@sha256:digest, never a tag")
    return match


def main() -> int:
    arguments = parser().parse_args()
    try:
        return arguments.handler(arguments)
    except (ValueError, RuntimeError) as error:
        print(f"runpod: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
