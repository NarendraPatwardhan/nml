"""Freeze bounded execution fixtures from the immutable NVFP4 artifact.

The extractor never downloads a shard. It reads exact component byte ranges
from the checked physical inventory, decodes them with a deliberately small
spec-level oracle, and records enough identity to make copied or stale bytes
unusable. The resulting fixture is permanent test data, not a model cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NoReturn


SELECTED_COLUMNS = (0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 127, 1023, 2047, 2879)
INPUT_MULTIPLIER = 17
INPUT_MODULUS = 251
INPUT_OFFSET = 125
INPUT_DIVISOR = 64.0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--published", type=Path, required=True)
    result.add_argument("--inventory", type=Path, required=True)
    result.add_argument("--tensor", required=True)
    result.add_argument("--rows", type=parse_rows, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--check", action="store_true")
    return result


def main() -> int:
    arguments = parser().parse_args()
    published = read_object(arguments.published)
    inventory = read_object(arguments.inventory)
    records = inventory.get("tensors")
    if not isinstance(records, list):
        fail("physical inventory has no tensor array")
    if (
        inventory.get("repository") != published.get("repository")
        or inventory.get("revision") != published.get("revision")
        or inventory.get("artifact_manifest_sha256")
        != published.get("artifact_manifest_sha256")
    ):
        fail("published identity and physical inventory disagree")

    components = {
        record.get("component_role"): record
        for record in records
        if record.get("logical_name") == arguments.tensor
    }
    if set(components) != {"payload", "block_scales", "global_scale"}:
        fail("selected tensor is not one complete NVFP4 representation")
    payload = components["payload"]
    scales = components["block_scales"]
    global_scale = components["global_scale"]
    logical_shape = payload.get("logical_shape")
    if (
        not isinstance(logical_shape, list)
        or len(logical_shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in logical_shape)
    ):
        fail("fixture tensor must be a positive rank-two projection")
    outputs, inputs = logical_shape
    if payload.get("shape") != [outputs, math.ceil(inputs / 2)]:
        fail("payload shape does not match the logical projection")
    if scales.get("shape") != [outputs, math.ceil(inputs / 16)]:
        fail("scale shape does not match the logical projection")
    if global_scale.get("shape") != [] or global_scale.get("dtype") != "F32":
        fail("global scale is not one F32 scalar")
    if any(row >= outputs for row in arguments.rows):
        fail("fixture row exceeds the logical output extent")
    if max(SELECTED_COLUMNS) >= inputs:
        fail("selected fixture column exceeds the logical input extent")

    base = (
        f"https://huggingface.co/{published['repository']}/resolve/"
        f"{published['revision']}"
    )
    global_bytes = fetch_component_range(base, global_scale, 0, 4)
    global_value = struct.unpack("<f", global_bytes)[0]
    if not math.isfinite(global_value) or global_value <= 0.0:
        fail("artifact global scale is not finite and positive")

    payload_width = math.ceil(inputs / 2)
    scale_width = math.ceil(inputs / 16)
    input_values = [
        f32(((index * INPUT_MULTIPLIER) % INPUT_MODULUS - INPUT_OFFSET) / INPUT_DIVISOR)
        for index in range(inputs)
    ]
    fixture_rows = []
    for row in arguments.rows:
        payload_bytes = fetch_component_range(
            base, payload, row * payload_width, payload_width
        )
        scale_bytes = fetch_component_range(
            base, scales, row * scale_width, scale_width
        )
        decoded = decode_row(payload_bytes, scale_bytes, global_value, inputs)
        decoded_bytes = b"".join(struct.pack("<f", value) for value in decoded)
        projection = math.fsum(
            float(input_value) * float(weight)
            for input_value, weight in zip(input_values, decoded, strict=True)
        )
        fixture_rows.append(
            {
                "row": row,
                "payload_hex": payload_bytes.hex(),
                "block_scales_hex": scale_bytes.hex(),
                "decoded_f32_sha256": hashlib.sha256(decoded_bytes).hexdigest(),
                "decoded_samples": [
                    {
                        "column": column,
                        "f32_bits": f"{struct.unpack('<I', struct.pack('<f', decoded[column]))[0]:08x}",
                    }
                    for column in SELECTED_COLUMNS
                ],
                "projection_f64": projection,
            }
        )

    result = {
        "schema_version": 1,
        "repository": published["repository"],
        "revision": published["revision"],
        "artifact_manifest_sha256": published["artifact_manifest_sha256"],
        "recipe": inventory["recipe"],
        "logical_name": arguments.tensor,
        "logical_shape": logical_shape,
        "logical_dtype": payload["logical_dtype"],
        "shard": payload["shard"],
        "global_scale_hex": global_bytes.hex(),
        "input_formula": {
            "multiplier": INPUT_MULTIPLIER,
            "modulus": INPUT_MODULUS,
            "offset": INPUT_OFFSET,
            "divisor": INPUT_DIVISOR,
        },
        "rows": fixture_rows,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_text(encoding="utf-8") != encoded:
            fail(f"checked execution fixture is stale: {arguments.output}")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "execution_fixture_checked" if arguments.check else "execution_fixture_written",
                "logical_name": arguments.tensor,
                "revision": published["revision"],
                "rows": arguments.rows,
            },
            sort_keys=True,
        )
    )
    return 0


def decode_row(payload: bytes, scales: bytes, global_scale: float, width: int) -> list[float]:
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    result = []
    for column in range(width):
        packed = payload[column // 2]
        code = packed & 0x0F if column % 2 == 0 else packed >> 4
        magnitude = magnitudes[code & 0x07]
        value = magnitude if code & 0x08 == 0 else -magnitude
        bits = scales[column // 16]
        if bits & 0x80 or bits == 0x7F:
            fail(f"invalid E4M3FN scale bits 0x{bits:02x}")
        exponent = (bits >> 3) & 0x0F
        fraction = bits & 0x07
        scale = (
            fraction * 2.0**-9
            if exponent == 0
            else (1.0 + fraction / 8.0) * 2.0 ** (exponent - 7)
        )
        scale = f32(scale)
        effective_scale = f32(scale * global_scale)
        result.append(f32(value * effective_scale))
    return result


def fetch_component_range(base: str, record: dict[str, Any], offset: int, length: int) -> bytes:
    absolute = record.get("shard_absolute_offset")
    extent = record.get("byte_length")
    shard = record.get("shard")
    if (
        not isinstance(absolute, int)
        or not isinstance(extent, int)
        or not isinstance(shard, str)
        or offset < 0
        or length <= 0
        or offset + length > extent
    ):
        fail("component range exceeds the checked physical record")
    start = absolute + offset
    end = start + length - 1
    request = urllib.request.Request(
        f"{base}/{shard}",
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "nml-nvfp4-fixture/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                fail(f"server ignored bounded range request for {shard!r}")
            expected_range = f"bytes {start}-{end}/"
            actual_range = response.headers.get("Content-Range", "")
            if not actual_range.startswith(expected_range):
                fail(f"invalid Content-Range for {shard!r}: {actual_range!r}")
            value = response.read(length + 1)
    except urllib.error.URLError as error:
        fail(f"failed to range-fetch {shard!r}: {error}")
    if len(value) != length:
        fail(f"short bounded range for {shard!r}: expected {length}, received {len(value)}")
    return value


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def parse_rows(value: str) -> list[int]:
    try:
        rows = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("rows must be comma-separated integers") from error
    if not rows or any(row < 0 for row in rows) or rows != sorted(set(rows)):
        raise argparse.ArgumentTypeError("rows must be unique, non-negative, and sorted")
    return rows


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def fail(message: str) -> NoReturn:
    raise SystemExit(f"NVFP4 fixture extraction: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
