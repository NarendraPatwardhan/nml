from __future__ import annotations

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

from materialize import (
    MANIFEST_NAME,
    RECEIPT_KIND,
    RECEIPT_NAME,
    MaterializationError,
    materialize,
)


class MaterializationContract(unittest.TestCase):
    def test_verified_files_become_read_only_and_receive_exact_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_hash = artifact(root, {"weights.bin": b"compact weights"})
            receipt = materialize(root, manifest_hash)

            self.assertEqual(receipt["kind"], RECEIPT_KIND)
            self.assertEqual(receipt["manifest_sha256"], manifest_hash)
            self.assertEqual(receipt["file_count"], 1)
            self.assertEqual(receipt["total_bytes"], len(b"compact weights"))
            record = receipt["files"][0]
            metadata = (root / "weights.bin").stat()
            self.assertEqual(record["device"], metadata.st_dev)
            self.assertEqual(record["inode"], metadata.st_ino)
            self.assertEqual(record["modified_unix_nanoseconds"], metadata.st_mtime_ns)
            self.assertEqual(record["changed_unix_nanoseconds"], metadata.st_ctime_ns)
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o444)
            self.assertEqual(stat.S_IMODE((root / RECEIPT_NAME).stat().st_mode), 0o444)

            # Re-verification is deliberate ingestion work and refreshes the
            # receipt without requiring writable artifact files.
            refreshed = materialize(root, manifest_hash)
            self.assertEqual(refreshed["files"], receipt["files"])

    def test_same_size_content_change_invalidates_full_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_hash = artifact(root, {"weights.bin": b"abcd"})
            materialize(root, manifest_hash)
            path = root / "weights.bin"
            path.chmod(0o644)
            path.write_bytes(b"wxyz")
            with self.assertRaisesRegex(MaterializationError, "SHA-256"):
                materialize(root, manifest_hash)

    def test_manifest_hash_is_the_materialization_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_hash = artifact(root, {"weights.bin": b"weights"})
            with self.assertRaisesRegex(MaterializationError, "manifest SHA-256"):
                materialize(root, "0" * 64)
            self.assertFalse((root / RECEIPT_NAME).exists())
            self.assertNotEqual(manifest_hash, "0" * 64)

    def test_manifest_cannot_escape_the_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.bin"
            outside.write_bytes(b"outside")
            manifest = {
                "files": [
                    {
                        "path": "../outside.bin",
                        "size": outside.stat().st_size,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                ]
            }
            manifest_path = root / MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            try:
                with self.assertRaisesRegex(MaterializationError, "clean relative path"):
                    materialize(root, manifest_hash)
            finally:
                outside.unlink(missing_ok=True)

    def test_symlink_root_is_not_a_materialization_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "artifact"
            root.mkdir()
            manifest_hash = artifact(root, {"weights.bin": b"weights"})
            alias = parent / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(MaterializationError, "not a real directory"):
                materialize(alias, manifest_hash)

    def test_payload_cannot_cross_an_intermediate_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "artifact"
            root.mkdir()
            outside = parent / "outside"
            outside.mkdir()
            payload = outside / "weights.bin"
            payload.write_bytes(b"weights")
            (root / "linked").symlink_to(outside, target_is_directory=True)
            manifest = {
                "files": [
                    {
                        "path": "linked/weights.bin",
                        "size": payload.stat().st_size,
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                    }
                ]
            }
            manifest_path = root / MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(
                MaterializationError, "crosses a non-directory or symlink"
            ):
                materialize(root, manifest_hash)

    def test_group_writable_root_cannot_issue_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_hash = artifact(root, {"weights.bin": b"weights"})
            root.chmod(0o770)
            with self.assertRaisesRegex(MaterializationError, "group- or other-writable"):
                materialize(root, manifest_hash)


def artifact(root: Path, files: dict[str, bytes]) -> str:
    entries = []
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        entries.append(
            {
                "path": name,
                "size": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps({"files": entries}, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
