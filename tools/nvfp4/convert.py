"""Convert the pinned GPT-OSS 20B BF16 source into NML NVFP4 recipe v2.

This is a deterministic artifact-production tool, not an inference runtime.
It validates the checked source manifest before interpreting weights, converts
one tensor in bounded CPU row chunks, validates its own packed output, and
publishes all output files in one Hugging Face commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3FN_MAX = 448.0
E2M1_MAX = 6.0
BLOCK_SIZE = 16
RUNTIME_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--tensor-manifest", type=Path, required=True)
    result.add_argument("--recipe", type=Path, required=True)
    result.add_argument("--work-directory", type=Path, required=True)
    result.add_argument("--destination-repository", required=True)
    result.add_argument("--hf-token-file", type=Path, required=True)
    result.add_argument("--chunk-rows", type=positive_integer, default=2048)
    result.add_argument("--resume", action="store_true")
    return result


def main() -> int:
    arguments = parser().parse_args()
    source_manifest = read_object(arguments.source_manifest)
    tensor_manifest = read_array(arguments.tensor_manifest)
    recipe = read_object(arguments.recipe)
    require_contract(source_manifest, tensor_manifest, recipe, arguments)

    token = arguments.hf_token_file.read_text(encoding="utf-8").strip()
    if not token:
        fail("Hugging Face token file is empty")
    arguments.hf_token_file.unlink()

    # Imports happen after contract validation so a malformed invocation does
    # not initialize the tensor runtime or perform network I/O.
    import torch
    from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    # Artifact production is a deterministic storage transformation, not an
    # inference workload. Keeping it CPU-only prevents checkpoint identity from
    # depending on accelerator availability or CUDA library state and lets the
    # remote build runner own the large download without renting a GPU.
    device = torch.device("cpu")
    print(
        json.dumps(
            {
                "event": "device",
                "kind": "cpu",
                "threads": torch.get_num_threads(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    source = arguments.work_directory / "source"
    output = arguments.work_directory / "output"
    source.mkdir(parents=True, exist_ok=True)
    if output.exists() and not arguments.resume:
        fail(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_records = {entry["path"]: entry for entry in source_manifest["files"]}
    expected_files = list(source_records)
    by_shard = group_tensor_manifest(tensor_manifest)
    metadata_files = [path for path in expected_files if path not in by_shard]
    print(json.dumps({"event": "download_started", "files": len(expected_files)}), flush=True)
    snapshot_download(
        repo_id=source_manifest["repository"],
        revision=source_manifest["revision"],
        local_dir=source,
        allow_patterns=metadata_files,
        max_workers=4,
        token=token,
    )
    verified = 0
    for name in metadata_files:
        verified += 1
        validate_source_file(
            source,
            source_records[name],
            verified,
            len(expected_files),
        )

    output_weight_map: dict[str, str] = {}
    output_files: list[dict[str, object]] = []
    for shard_index, shard_name in enumerate(sorted(by_shard), 1):
        # A BF16 source shard is around 5 GiB while its NVFP4 output is around
        # 1.4 GiB. Materialize, authenticate, and retire one source shard at a
        # time so disk usage is bounded by the final artifact plus one shard;
        # conversion must not require a 60+ GiB worker merely because the source
        # checkpoint is sharded.
        snapshot_download(
            repo_id=source_manifest["repository"],
            revision=source_manifest["revision"],
            local_dir=source,
            allow_patterns=[shard_name],
            max_workers=1,
            token=token,
        )
        verified += 1
        validate_source_file(
            source,
            source_records[shard_name],
            verified,
            len(expected_files),
        )
        validate_tensor_inventory(
            source,
            {shard_name: by_shard[shard_name]},
            safe_open,
        )
        destination = output / shard_name
        if destination.exists() and arguments.resume:
            print(
                json.dumps(
                    {"event": "shard_reused", "index": shard_index, "shard": shard_name},
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            convert_shard(
                source / shard_name,
                destination,
                by_shard[shard_name],
                arguments.chunk_rows,
                device,
                torch,
                safe_open,
                save_file,
            )
        records = inspect_output(destination, safe_open)
        for name in records:
            if name in output_weight_map:
                fail(f"duplicate output tensor {name!r}")
            output_weight_map[name] = shard_name
        output_files.append(file_record(destination))
        print(
            json.dumps(
                {
                    "event": "shard_complete",
                    "index": shard_index,
                    "total": len(by_shard),
                    "shard": shard_name,
                    "size": destination.stat().st_size,
                    "sha256": output_files[-1]["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        (source / shard_name).unlink()

    for name in RUNTIME_FILES:
        shutil.copyfile(source / name, output / name)
        output_files.append(file_record(output / name))

    index = {
        "metadata": {
            "total_size": sum(
                tensor_byte_length(record)
                for shard in sorted(by_shard)
                for record in inspect_output(output / shard, safe_open).values()
            ),
            "nml_recipe": recipe["recipe"],
        },
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    write_json(output / "model.safetensors.index.json", index)
    shutil.copyfile(arguments.source_manifest, output / "nml-source.json")
    shutil.copyfile(arguments.tensor_manifest, output / "nml-source-tensors.json")
    shutil.copyfile(arguments.recipe, output / "nml-nvfp4-recipe.json")
    write_model_card(output / "README.md", source_manifest, recipe)
    for name in (
        "README.md",
        "model.safetensors.index.json",
        "nml-source.json",
        "nml-source-tensors.json",
        "nml-nvfp4-recipe.json",
    ):
        output_files.append(file_record(output / name))

    artifact_manifest = {
        "schema_version": 1,
        "recipe": recipe["recipe"],
        "source_repository": source_manifest["repository"],
        "source_revision": source_manifest["revision"],
        "source_manifest_sha256": sha256(arguments.source_manifest),
        "tensor_manifest_sha256": sha256(arguments.tensor_manifest),
        "recipe_sha256": sha256(arguments.recipe),
        "converter": {
            "name": "nml-nvfp4-converter",
            "version": 2,
            "device": "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": importlib.metadata.version("numpy"),
            "safetensors": importlib.metadata.version("safetensors"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "script_sha256": sha256(Path(__file__)),
            "requirements_sha256": sha256(Path(__file__).with_name("requirements.txt")),
        },
        "files": sorted(output_files, key=lambda entry: str(entry["path"])),
    }
    write_json(output / "nml-artifact-manifest.json", artifact_manifest)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=arguments.destination_repository,
        repo_type="model",
        private=False,
        exist_ok=arguments.resume,
    )
    print(
        json.dumps(
            {
                "event": "upload_started",
                "repository": arguments.destination_repository,
                "bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # `upload_folder` delegates to `create_commit` with the client's default of
    # five concurrent file uploads. That is needlessly memory-hungry for this
    # artifact: nine 1.3 GiB LFS shards can be in flight together while the
    # converter's CPU allocator is still warm. Keep the publication atomic but
    # bound it to one file at a time. The Hub still deduplicates already
    # uploaded LFS objects when a remote runner has to retry this operation.
    operations = [
        CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path)
        for path in sorted(output.iterdir())
        if path.is_file()
    ]
    commit = api.create_commit(
        repo_id=arguments.destination_repository,
        repo_type="model",
        operations=operations,
        num_threads=1,
        commit_message=(
            "Publish deterministic NML NVFP4 conversion of "
            f"{source_manifest['repository']}@{source_manifest['revision']}"
        ),
    )
    info = api.repo_info(repo_id=arguments.destination_repository, repo_type="model")
    print(
        json.dumps(
            {
                "event": "published",
                "repository": arguments.destination_repository,
                "revision": info.sha,
                "commit_url": str(commit.commit_url),
                "artifact_manifest_sha256": sha256(output / "nml-artifact-manifest.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def convert_shard(
    source: Path,
    destination: Path,
    records: list[dict[str, Any]],
    chunk_rows: int,
    device: Any,
    torch: Any,
    safe_open: Any,
    save_file: Any,
) -> None:
    converted: dict[str, Any] = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for tensor_index, record in enumerate(records, 1):
            name = record["name"]
            tensor = logical_tensor(handle.get_tensor(name), record)
            if record["target_representation"] == "dense":
                converted[name] = tensor
            else:
                payload, scales, global_factor = quantize_tensor(
                    tensor, chunk_rows, device, torch
                )
                converted[f"{name}.payload"] = payload
                converted[f"{name}.block_scales"] = scales
                converted[f"{name}.global_scale"] = global_factor
            print(
                json.dumps(
                    {
                        "event": "tensor_converted",
                        "shard": source.name,
                        "index": tensor_index,
                        "total": len(records),
                        "name": name,
                        "representation": record["target_representation"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del tensor
    save_file(
        converted,
        destination,
        metadata={"format": "pt", "nml_recipe": "nml-nvfp4-weight-v2"},
    )


def logical_tensor(tensor: Any, record: dict[str, Any]) -> Any:
    """Maps source storage into NML's one output-major contraction layout.

    The source GPT-OSS expert tensors are input-major. Transposition happens
    before quantization so the representation block remains the contraction K
    axis; transposing packed v1 bytes would preserve the wrong scale geometry.
    """
    role = record["role"]
    if role in {"expert_gate_up_projection", "expert_down_projection"}:
        if tensor.ndim != 3:
            fail(f"{role} source tensor must have rank three")
        return tensor.permute(0, 2, 1).contiguous()
    return tensor.contiguous()


def quantize_tensor(tensor: Any, chunk_rows: int, device: Any, torch: Any) -> tuple[Any, Any, Any]:
    if tensor.dtype != torch.bfloat16 or tensor.ndim == 0 or tensor.shape[-1] == 0:
        fail(f"invalid NVFP4 source tensor dtype/shape: {tensor.dtype} {tuple(tensor.shape)}")
    width = tensor.shape[-1]
    rows = tensor.reshape(-1, width)

    maximum = torch.zeros((), dtype=torch.float32, device=device)
    for start in range(0, rows.shape[0], chunk_rows):
        values = rows[start : start + chunk_rows].to(device=device, dtype=torch.float32)
        if not torch.isfinite(values).all().item():
            fail("NVFP4 source tensor contains a non-finite value")
        maximum = torch.maximum(maximum, values.abs().amax())
    global_scale = float(maximum.item()) / (E4M3FN_MAX * E2M1_MAX)
    if global_scale == 0.0:
        global_scale = 1.0

    payload_chunks = []
    scale_chunks = []
    thresholds = torch.tensor(E2M1_THRESHOLDS, dtype=torch.float32, device=device)
    for start in range(0, rows.shape[0], chunk_rows):
        values = rows[start : start + chunk_rows].to(device=device, dtype=torch.float32)
        padding = (-width) % BLOCK_SIZE
        padded = torch.nn.functional.pad(values, (0, padding)) if padding else values
        blocks = padded.reshape(padded.shape[0], -1, BLOCK_SIZE)
        raw_scales = (blocks.abs().amax(dim=-1) / E2M1_MAX) / global_scale
        scale_bits = encode_e4m3fn(raw_scales, torch)
        block_scales = decode_e4m3fn(scale_bits, torch)
        effective = (block_scales * global_scale).unsqueeze(-1)
        normalized = torch.where(effective == 0.0, torch.zeros_like(blocks), blocks / effective)

        magnitude = normalized.abs()
        codes = torch.bucketize(magnitude, thresholds).to(torch.uint8)
        # `bucketize` chooses the lower code at a boundary. IEEE ties-to-even
        # instead chooses the upper code when the lower code is odd.
        for boundary_index in (1, 3, 5):
            boundary = thresholds[boundary_index]
            codes = torch.where(
                (magnitude == boundary) & (codes == boundary_index),
                codes + 1,
                codes,
            )
        codes |= torch.signbit(normalized).to(torch.uint8) << 3
        codes = codes.reshape(codes.shape[0], -1)[:, :width]
        if width & 1:
            codes = torch.nn.functional.pad(codes, (0, 1))
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        payload_chunks.append(packed.cpu())
        scale_chunks.append(scale_bits.cpu())

    payload = torch.cat(payload_chunks).reshape(*tensor.shape[:-1], (width + 1) // 2)
    scales = torch.cat(scale_chunks).reshape(*tensor.shape[:-1], (width + 15) // 16)
    global_factor = torch.tensor(global_scale, dtype=torch.float32)
    validate_quantized_tensor(tensor, payload, scales, global_factor, torch)
    return payload, scales, global_factor


def encode_e4m3fn(values: Any, torch: Any) -> Any:
    values = values.clamp(min=0.0, max=E4M3FN_MAX)
    subnormal = torch.round(values * 512.0).clamp(0, 8).to(torch.int16)
    safe = torch.where(values == 0.0, torch.ones_like(values), values)
    exponent = torch.floor(torch.log2(safe)).to(torch.int16).clamp(-6, 8)
    step = torch.pow(torch.tensor(2.0, device=values.device), exponent.to(torch.float32) - 3.0)
    significand = torch.round(values / step).to(torch.int16)
    carry = significand == 16
    exponent = exponent + carry.to(torch.int16)
    significand = torch.where(carry, torch.full_like(significand, 8), significand)
    normal = ((exponent + 7) << 3) | (significand - 8)
    bits = torch.where(values < 2.0**-6, subnormal, normal).clamp(0, 0x7E)
    return bits.to(torch.uint8)


def decode_e4m3fn(bits: Any, torch: Any) -> Any:
    integer = bits.to(torch.int16)
    exponent = integer >> 3
    fraction = integer & 0x07
    subnormal = fraction.to(torch.float32) * 2.0**-9
    normal = (1.0 + fraction.to(torch.float32) / 8.0) * torch.pow(
        torch.tensor(2.0, device=bits.device), exponent.to(torch.float32) - 7.0
    )
    return torch.where(exponent == 0, subnormal, normal)


def validate_quantized_tensor(source: Any, payload: Any, scales: Any, global_factor: Any, torch: Any) -> None:
    expected_payload = (*source.shape[:-1], (source.shape[-1] + 1) // 2)
    expected_scales = (*source.shape[:-1], (source.shape[-1] + 15) // 16)
    if payload.dtype != torch.uint8 or tuple(payload.shape) != expected_payload:
        fail("converter produced an invalid payload contract")
    if scales.dtype != torch.uint8 or tuple(scales.shape) != expected_scales:
        fail("converter produced an invalid block-scale contract")
    if global_factor.dtype != torch.float32 or global_factor.ndim != 0:
        fail("converter produced an invalid global-scale contract")
    if source.shape[-1] & 1 and torch.any(payload[..., -1] & 0xF0).item():
        fail("converter produced nonzero payload padding")
    if torch.any(scales == 0x7F).item() or torch.any(scales & 0x80).item():
        fail("converter produced an invalid E4M3FN scale")


def require_contract(
    source: dict[str, Any], tensors: list[dict[str, Any]], recipe: dict[str, Any], arguments: argparse.Namespace
) -> None:
    if source.get("schema_version") != 1 or recipe.get("schema_version") != 1:
        fail("unsupported source or recipe schema")
    if recipe.get("recipe") != "nml-nvfp4-weight-v2":
        fail("converter only implements nml-nvfp4-weight-v2")
    if recipe.get("source_repository") != source.get("repository") or recipe.get(
        "source_revision"
    ) != source.get("revision"):
        fail("source and recipe identities disagree")
    if recipe.get("destination_repository") != arguments.destination_repository:
        fail("destination repository differs from the checked recipe")
    if len(tensors) != source.get("tensor_count"):
        fail("tensor manifest count differs from the source contract")


def validate_source_file(
    root: Path,
    record: dict[str, Any],
    index: int,
    total: int,
) -> None:
    path = root / record["path"]
    if not path.is_file() or path.stat().st_size != record["size"]:
        fail(f"source file size mismatch: {record['path']}")
    actual = sha256(path)
    if actual != record["sha256"]:
        fail(f"source file hash mismatch: {record['path']}")
    print(
        json.dumps(
            {
                "event": "source_verified",
                "index": index,
                "total": total,
                "path": record["path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def group_tensor_manifest(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    names = set()
    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or name in names:
            fail(f"duplicate or invalid tensor manifest name: {name!r}")
        names.add(name)
        result[record["source_shard"]].append(record)
    for records_in_shard in result.values():
        records_in_shard.sort(key=lambda record: record["name"])
    return dict(result)


def validate_tensor_inventory(root: Path, by_shard: dict[str, list[dict[str, Any]]], safe_open: Any) -> None:
    for shard, expected in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {record["name"] for record in expected}:
                fail(f"tensor inventory differs in {shard}")
            for record in expected:
                tensor = handle.get_slice(record["name"])
                if tensor.get_dtype() != record["source_dtype"] or list(tensor.get_shape()) != record["logical_shape"]:
                    fail(f"tensor metadata differs for {record['name']}")


def inspect_output(path: Path, safe_open: Any) -> dict[str, dict[str, Any]]:
    result = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_slice(name)
            result[name] = {"dtype": tensor.get_dtype(), "shape": list(tensor.get_shape())}
    return result


def tensor_byte_length(record: dict[str, Any]) -> int:
    widths = {"U8": 1, "BF16": 2, "F32": 4}
    width = widths.get(record["dtype"])
    if width is None:
        fail(f"unexpected output dtype {record['dtype']}")
    elements = 1
    for dimension in record["shape"]:
        elements *= dimension
    return elements * width


def file_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_model_card(path: Path, source: dict[str, Any], recipe: dict[str, Any]) -> None:
    path.write_text(
        "---\nlicense: apache-2.0\nlibrary_name: nml\nbase_model: openai/gpt-oss-20b\n"
        "tags:\n- gpt_oss\n- nvfp4\n- nml\n---\n\n"
        "# GPT-OSS 20B NVFP4\n\n"
        "This is the deterministic NML NVFP4 recipe-v2 conversion of "
        f"`{source['repository']}@{source['revision']}`. It uses last-axis "
        "one-dimensional blocks of 16 weights, low-nibble-first E2M1 payloads, "
        "positive E4M3FN block scales, and one F32 global factor per quantized "
        "parameter. Small and sensitivity-critical tensors remain BF16 exactly "
        "as declared by `nml-source-tensors.json`.\n\n"
        "Conversion is CPU-only; exact converter and dependency provenance is "
        "recorded in `nml-artifact-manifest.json`. This artifact is intended for "
        "NML and is not mislabeled as the original "
        "GPT-OSS MXFP4 representation. Exact source hashes, tensor disposition, "
        f"and conversion semantics are included in the repository. Recipe: `{recipe['recipe']}`.\n",
        encoding="utf-8",
    )


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"nvfp4 conversion: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
