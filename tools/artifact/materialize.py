#!/usr/bin/env python3
"""Fully verify an artifact once and issue its local immutable receipt.

The manifest remains the cryptographic description of the artifact. This tool
is the only expensive materialization step: it hashes every declared file,
makes the verified files read-only, captures their filesystem identities, and
atomically writes the receipt consumed by product startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "nml-artifact-manifest.json"
RECEIPT_NAME = "nml-materialization.json"
RECEIPT_KIND = "nml.artifact.materialization"
RECEIPT_SCHEMA_VERSION = 1
MAX_CONTROL_BYTES = 1024 * 1024
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class MaterializationError(ValueError):
    """The artifact cannot be trusted as a complete materialization."""


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            value.update(chunk)
    return value.hexdigest()


def read_manifest(root: Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = root / MANIFEST_NAME
    require_regular_file(manifest_path, "artifact manifest")
    if manifest_path.stat().st_size > MAX_CONTROL_BYTES:
        raise MaterializationError("artifact manifest exceeds the control-file bound")
    actual = sha256(manifest_path)
    if actual != expected_sha256:
        raise MaterializationError(
            f"artifact manifest SHA-256 is {actual}, expected {expected_sha256}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"artifact manifest is invalid: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError("artifact manifest root must be an object")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise MaterializationError("artifact manifest must declare a nonempty files array")
    return manifest_path, value


def materialize(root: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    try:
        root_metadata = root.stat(follow_symlinks=False)
    except OSError as error:
        raise MaterializationError(f"cannot inspect artifact root {root}: {error}") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise MaterializationError(f"artifact root is not a real directory: {root}")
    if stat.S_IMODE(root_metadata.st_mode) & 0o022:
        raise MaterializationError(f"artifact root is group- or other-writable: {root}")
    root = root.resolve(strict=True)
    manifest_path, manifest = read_manifest(root, expected_manifest_sha256)
    entries = manifest["files"]
    verified: list[tuple[str, Path, int]] = []
    seen: set[str] = set()
    total_bytes = 0
    for index, raw in enumerate(entries, 1):
        if not isinstance(raw, dict):
            raise MaterializationError(f"artifact file entry {index} is not an object")
        relative = raw.get("path")
        expected_size = raw.get("size")
        expected_hash = raw.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise MaterializationError(f"artifact file entry {index} has no path")
        if relative in {MANIFEST_NAME, RECEIPT_NAME}:
            raise MaterializationError(
                f"local control file {relative!r} cannot be an artifact payload"
            )
        if relative in seen:
            raise MaterializationError(f"artifact manifest repeats {relative!r}")
        seen.add(relative)
        require_relative_path(relative)
        require_real_parent_directories(root, relative)
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
            raise MaterializationError(f"artifact file {relative!r} has an invalid size")
        if not is_sha256(expected_hash):
            raise MaterializationError(f"artifact file {relative!r} has an invalid SHA-256")
        path = root.joinpath(*PurePosixPath(relative).parts)
        require_regular_file(path, f"artifact file {relative!r}")
        size = path.stat().st_size
        if size != expected_size:
            raise MaterializationError(
                f"artifact file {relative!r} is {size} bytes, expected {expected_size}"
            )
        actual = sha256(path)
        if actual != expected_hash:
            raise MaterializationError(
                f"artifact file {relative!r} SHA-256 is {actual}, expected {expected_hash}"
            )
        total_bytes += size
        verified.append((relative, path, size))
        emit(
            "artifact_file_verified",
            file=relative,
            index=index,
            files=len(entries),
            bytes=size,
        )

    # Immutability is established only after every content hash succeeds. A
    # failed verification therefore never leaves a partially blessed receipt.
    for _, path, _ in verified:
        remove_write_permissions(path)
    remove_write_permissions(manifest_path)

    receipt_files = []
    for relative, path, size in verified:
        metadata = path.stat(follow_symlinks=False)
        receipt_files.append(
            {
                "path": relative,
                "size": size,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode": stat.S_IMODE(metadata.st_mode),
                "modified_unix_nanoseconds": metadata.st_mtime_ns,
                "changed_unix_nanoseconds": metadata.st_ctime_ns,
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "manifest_sha256": expected_manifest_sha256,
        "file_count": len(receipt_files),
        "total_bytes": total_bytes,
        "verified_at_unix_nanoseconds": time.time_ns(),
        "files": receipt_files,
    }
    write_receipt(root / RECEIPT_NAME, receipt)
    emit(
        "artifact_materialized",
        root=str(root),
        receipt=str(root / RECEIPT_NAME),
        manifest_sha256=expected_manifest_sha256,
        file_count=len(receipt_files),
        total_bytes=total_bytes,
    )
    return receipt


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def require_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MaterializationError(f"artifact path is not a clean relative path: {value!r}")


def require_real_parent_directories(root: Path, relative: str) -> None:
    """Reject a payload whose lexical path crosses a directory symlink.

    The artifact administrator is trusted against concurrent replacement, but
    the materialization boundary must still reject an accidentally mounted or
    linked subtree. The final component is checked separately as a regular,
    non-symlink file.
    """

    current = root
    for component in PurePosixPath(relative).parts[:-1]:
        current /= component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as error:
            raise MaterializationError(
                f"cannot inspect artifact directory {current}: {error}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise MaterializationError(
                f"artifact path crosses a non-directory or symlink: {current}"
            )


def require_regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise MaterializationError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise MaterializationError(f"{label} is not a regular file: {path}")


def remove_write_permissions(path: Path) -> None:
    mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    read_only_mode = mode & ~0o222
    if mode != read_only_mode:
        path.chmod(read_only_mode)


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("artifact_root", type=Path)
    root.add_argument("--expected-manifest-sha256", required=True)
    return root


def main() -> int:
    arguments = parser().parse_args()
    if not is_sha256(arguments.expected_manifest_sha256):
        raise MaterializationError("expected manifest SHA-256 must be lowercase hexadecimal")
    materialize(arguments.artifact_root, arguments.expected_manifest_sha256)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MaterializationError, OSError) as error:
        print(f"materialize: {error}", file=sys.stderr)
        raise SystemExit(1)
