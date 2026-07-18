"""Audit the published GPT-OSS NVFP4 artifact at its immutable revision.

Only bounded metadata is downloaded. SafeTensor payloads are identified by the
checked file digests produced during conversion; HTTP range reads recover each
shard header so every physical tensor can be matched to the frozen logical
manifest without downloading model weights again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NoReturn


HEADER_PREFIX_BYTES = 8
MAX_HEADER_BYTES = 100_000_000
DTYPE_BYTES = {"BF16": 2, "F32": 4, "U8": 1}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--published", type=Path, required=True)
    result.add_argument("--artifact-manifest", type=Path, required=True)
    result.add_argument("--source-tensors", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--check",
        action="store_true",
        help="fail if output differs instead of updating it",
    )
    return result


def main() -> int:
    arguments = parser().parse_args()
    published = read_object(arguments.published)
    artifact = read_object(arguments.artifact_manifest)
    sources = read_array(arguments.source_tensors)
    require_identity(published, artifact)

    base = (
        f"https://huggingface.co/{published['repository']}/resolve/"
        f"{published['revision']}"
    )
    remote_manifest = fetch(base, published["artifact_manifest_path"])
    if sha256_bytes(remote_manifest) != published["artifact_manifest_sha256"]:
        fail("published artifact manifest digest differs from the checked identity")
    if json.loads(remote_manifest) != artifact:
        fail("published artifact manifest differs from the checked local copy")

    index_bytes = fetch(base, "model.safetensors.index.json")
    index = json.loads(index_bytes)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        fail("published SafeTensor index has no weight_map")

    expected = expected_components(sources)
    actual_names = set(index["weight_map"])
    if actual_names != set(expected):
        fail(describe_name_difference(set(expected), actual_names))

    artifact_files = {record["path"]: record for record in artifact["files"]}
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)

    physical = []
    total_bytes = 0
    for shard in sorted(by_shard):
        file_record = artifact_files.get(shard)
        if file_record is None:
            fail(f"SafeTensor index names unchecked shard {shard!r}")
        header, remote_size, data_start = fetch_safetensors_header(base, shard)
        if remote_size != file_record["size"]:
            fail(f"published shard size differs for {shard!r}")
        header_names = set(header) - {"__metadata__"}
        if header_names != set(by_shard[shard]):
            fail(f"SafeTensor header and index disagree for {shard!r}")

        for name in sorted(header):
            metadata = header[name]
            expected_record = expected[name]
            validate_tensor(name, metadata, expected_record)
            start, end = metadata["data_offsets"]
            byte_length = end - start
            total_bytes += byte_length
            identity = {
                key: value
                for key, value in expected_record.items()
                if not key.startswith("expected_")
            }
            physical.append(
                {
                    **identity,
                    "name": name,
                    "shard": shard,
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                    "byte_length": byte_length,
                    "shard_data_offset": start,
                    "shard_absolute_offset": data_start + start,
                }
            )

    declared_total = index.get("metadata", {}).get("total_size")
    if declared_total != total_bytes:
        fail(
            f"SafeTensor index total_size {declared_total!r} differs from "
            f"physical tensor bytes {total_bytes}"
        )
    result = {
        "schema_version": 1,
        "repository": published["repository"],
        "revision": published["revision"],
        "artifact_manifest_sha256": published["artifact_manifest_sha256"],
        "recipe": artifact["recipe"],
        "logical_tensor_count": len(sources),
        "physical_tensor_count": len(physical),
        "physical_tensor_bytes": total_bytes,
        "tensors": physical,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_text(encoding="utf-8") != encoded:
            fail(f"checked physical inventory is stale: {arguments.output}")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "artifact_audited",
                "logical_tensors": len(sources),
                "physical_tensors": len(physical),
                "physical_bytes": total_bytes,
                "revision": published["revision"],
            },
            sort_keys=True,
        )
    )
    return 0


def require_identity(published: dict[str, Any], artifact: dict[str, Any]) -> None:
    if published.get("schema_version") != 1 or artifact.get("schema_version") != 1:
        fail("unsupported published or artifact manifest schema")
    required_strings = (
        "repository",
        "revision",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
    )
    for name in required_strings:
        if not isinstance(published.get(name), str) or not published[name]:
            fail(f"published identity has invalid {name!r}")
    if artifact.get("recipe") != "nml-nvfp4-weight-v1":
        fail("published artifact does not use the admitted NVFP4 recipe")


def expected_components(sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        name = source["name"]
        common = {
            "logical_name": name,
            "logical_shape": source["logical_shape"],
            "logical_dtype": source["source_dtype"],
            "logical_role": source["role"],
            "logical_mapping": source["logical_mapping"],
            "transpose": source["transpose"],
            "representation": source["target_representation"],
        }
        if source["target_representation"] == "dense":
            add_expected(
                result,
                name,
                {**common, "component_role": "values", "expected_dtype": source["source_dtype"], "expected_shape": source["logical_shape"]},
            )
            continue
        if source["target_representation"] != "nvfp4":
            fail(f"unsupported representation for {name!r}")
        shape = source["logical_shape"]
        if not shape or shape[-1] <= 0:
            fail(f"invalid NVFP4 logical shape for {name!r}")
        add_expected(
            result,
            f"{name}.payload",
            {
                **common,
                "component_role": "payload",
                "expected_dtype": "U8",
                "expected_shape": [*shape[:-1], (shape[-1] + 1) // 2],
            },
        )
        add_expected(
            result,
            f"{name}.block_scales",
            {
                **common,
                "component_role": "block_scales",
                "expected_dtype": "U8",
                "expected_shape": [*shape[:-1], (shape[-1] + 15) // 16],
            },
        )
        add_expected(
            result,
            f"{name}.global_scale",
            {
                **common,
                "component_role": "global_scale",
                "expected_dtype": "F32",
                "expected_shape": [],
            },
        )
    return result


def add_expected(result: dict[str, dict[str, Any]], name: str, record: dict[str, Any]) -> None:
    if name in result:
        fail(f"duplicate expected physical tensor {name!r}")
    result[name] = record


def fetch_safetensors_header(base: str, path: str) -> tuple[dict[str, Any], int, int]:
    prefix, total = fetch_range(base, path, 0, HEADER_PREFIX_BYTES - 1)
    if len(prefix) != HEADER_PREFIX_BYTES:
        fail(f"short SafeTensor prefix for {path!r}")
    header_length = int.from_bytes(prefix, "little")
    if header_length <= 0 or header_length > MAX_HEADER_BYTES:
        fail(f"invalid SafeTensor header length for {path!r}: {header_length}")
    header_bytes, second_total = fetch_range(
        base, path, HEADER_PREFIX_BYTES, HEADER_PREFIX_BYTES + header_length - 1
    )
    if second_total != total or len(header_bytes) != header_length:
        fail(f"inconsistent SafeTensor range response for {path!r}")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        fail(f"SafeTensor header is not an object for {path!r}")
    header.pop("__metadata__", None)
    return header, total, HEADER_PREFIX_BYTES + header_length


def fetch(base: str, path: str) -> bytes:
    request = urllib.request.Request(
        f"{base}/{path}", headers={"User-Agent": "nml-artifact-auditor/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(MAX_HEADER_BYTES + 1)
    except urllib.error.URLError as error:
        fail(f"failed to fetch {path!r}: {error}")


def fetch_range(base: str, path: str, start: int, end: int) -> tuple[bytes, int]:
    request = urllib.request.Request(
        f"{base}/{path}",
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "nml-artifact-auditor/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                fail(f"server ignored bounded range request for {path!r}")
            content_range = response.headers.get("Content-Range", "")
            prefix = f"bytes {start}-{end}/"
            if not content_range.startswith(prefix):
                fail(f"invalid Content-Range for {path!r}: {content_range!r}")
            total = int(content_range.removeprefix(prefix))
            return response.read(end - start + 2), total
    except (urllib.error.URLError, ValueError) as error:
        fail(f"failed to range-fetch {path!r}: {error}")


def validate_tensor(name: str, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual.get("dtype") != expected["expected_dtype"] or actual.get("shape") != expected["expected_shape"]:
        fail(f"physical dtype/shape differs for {name!r}")
    offsets = actual.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) and value >= 0 for value in offsets)
        or offsets[1] < offsets[0]
    ):
        fail(f"invalid data offsets for {name!r}")
    elements = 1
    for dimension in actual["shape"]:
        if not isinstance(dimension, int) or dimension < 0:
            fail(f"invalid physical dimension for {name!r}")
        elements *= dimension
    expected_bytes = elements * DTYPE_BYTES[actual["dtype"]]
    if offsets[1] - offsets[0] != expected_bytes:
        fail(f"physical byte extent differs for {name!r}")


def describe_name_difference(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"physical tensor inventory differs: missing={missing!r}, extra={extra!r}"


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        fail(f"{path} is not an array of JSON objects")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fail(message: str) -> NoReturn:
    raise SystemExit(f"NVFP4 artifact audit: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
