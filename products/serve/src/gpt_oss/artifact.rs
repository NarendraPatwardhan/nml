//! Cheap launch-time validation of an ingestion-verified artifact.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};

pub(super) const RECEIPT_NAME: &str = "nml-materialization.json";
const RECEIPT_KIND: &str = "nml.artifact.materialization";
const RECEIPT_SCHEMA_VERSION: u32 = 1;
const MAX_RECEIPT_BYTES: u64 = 1024 * 1024;
const WRITE_BITS: u32 = 0o222;
const PERMISSION_BITS: u32 = 0o7777;

pub(super) struct ExpectedFile<'a> {
    pub(super) path: &'a str,
    pub(super) size: u64,
}

pub(super) fn validate_materialization<'a>(
    root: &Path,
    manifest_sha256: &str,
    expected_files: impl IntoIterator<Item = ExpectedFile<'a>>,
) -> Result<(), Error> {
    let root_metadata = fs::symlink_metadata(root).map_err(Error::Io)?;
    if !root_metadata.file_type().is_dir() {
        return Err(Error::Contract(format!(
            "artifact root is not a real directory: {}",
            root.display()
        )));
    }
    require_not_group_or_other_writable("artifact root", root_metadata.mode())?;

    let receipt_path = root.join(RECEIPT_NAME);
    let receipt_metadata = fs::symlink_metadata(&receipt_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Error::Contract(format!(
                "artifact materialization receipt is absent at {}; run the bounded artifact materializer before launch",
                receipt_path.display()
            ))
        } else {
            Error::Io(error)
        }
    })?;
    if !receipt_metadata.file_type().is_file() {
        return Err(Error::Contract(format!(
            "artifact materialization receipt is not a regular file: {}",
            receipt_path.display()
        )));
    }
    if receipt_metadata.len() > MAX_RECEIPT_BYTES {
        return Err(Error::Contract(
            "artifact materialization receipt exceeds the control-file bound".to_owned(),
        ));
    }
    require_read_only("artifact materialization receipt", receipt_metadata.mode())?;
    let receipt: Receipt = serde_json::from_slice(&fs::read(&receipt_path).map_err(Error::Io)?)
        .map_err(|error| {
            Error::Contract(format!(
                "artifact materialization receipt is invalid JSON: {error}"
            ))
        })?;
    if receipt.schema_version != RECEIPT_SCHEMA_VERSION || receipt.kind != RECEIPT_KIND {
        return Err(Error::Contract(
            "artifact materialization receipt uses an unsupported contract".to_owned(),
        ));
    }
    if receipt.manifest_sha256 != manifest_sha256 {
        return Err(Error::Contract(format!(
            "artifact materialization receipt names manifest {}, expected {manifest_sha256}",
            receipt.manifest_sha256
        )));
    }

    let mut expected = BTreeMap::new();
    let mut expected_total = 0u64;
    for file in expected_files {
        require_relative_path(file.path)?;
        if expected.insert(file.path, file.size).is_some() {
            return Err(Error::Contract(format!(
                "artifact manifest repeats {:?}",
                file.path
            )));
        }
        expected_total = expected_total.checked_add(file.size).ok_or_else(|| {
            Error::Contract("artifact manifest byte count overflows u64".to_owned())
        })?;
    }
    if receipt.verified_at_unix_nanoseconds <= 0
        || receipt.file_count != receipt.files.len()
        || receipt.file_count != expected.len()
        || receipt.total_bytes != expected_total
    {
        return Err(Error::Contract(
            "artifact materialization receipt accounting disagrees with the manifest".to_owned(),
        ));
    }

    let mut observed = BTreeMap::new();
    for file in receipt.files {
        require_relative_path(&file.path)?;
        if observed.insert(file.path.clone(), ()).is_some() {
            return Err(Error::Contract(format!(
                "artifact materialization receipt repeats {:?}",
                file.path
            )));
        }
        let expected_size = expected.get(file.path.as_str()).ok_or_else(|| {
            Error::Contract(format!(
                "artifact materialization receipt contains unlisted file {:?}",
                file.path
            ))
        })?;
        if file.size != *expected_size {
            return Err(Error::Contract(format!(
                "artifact materialization receipt gives {:?} {} bytes, expected {expected_size}",
                file.path, file.size
            )));
        }
        validate_file(root, &file)?;
    }
    if observed.len() != expected.len()
        || expected.keys().any(|path| !observed.contains_key(*path))
    {
        return Err(Error::Contract(
            "artifact materialization receipt omits a manifest file".to_owned(),
        ));
    }
    Ok(())
}

fn validate_file(root: &Path, receipt: &ReceiptFile) -> Result<(), Error> {
    require_real_parent_directories(root, &receipt.path)?;
    let path = root.join(&receipt.path);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        Error::Contract(format!(
            "cannot inspect materialized artifact file {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.file_type().is_file() {
        return Err(Error::Contract(format!(
            "materialized artifact file is not regular: {}",
            path.display()
        )));
    }
    require_read_only("materialized artifact file", metadata.mode())?;
    let mode = metadata.mode() & PERMISSION_BITS;
    let modified = unix_nanoseconds(metadata.mtime(), metadata.mtime_nsec())?;
    let changed = unix_nanoseconds(metadata.ctime(), metadata.ctime_nsec())?;
    let matches = metadata.len() == receipt.size
        && metadata.dev() == receipt.device
        && metadata.ino() == receipt.inode
        && mode == receipt.mode
        && modified == receipt.modified_unix_nanoseconds
        && changed == receipt.changed_unix_nanoseconds;
    if !matches {
        return Err(Error::Contract(format!(
            "materialized artifact file {:?} changed after content verification; rematerialize it before launch",
            receipt.path
        )));
    }
    Ok(())
}

fn require_real_parent_directories(root: &Path, relative: &str) -> Result<(), Error> {
    let mut current = root.to_path_buf();
    let mut components = Path::new(relative).components().peekable();
    while let Some(Component::Normal(component)) = components.next() {
        if components.peek().is_none() {
            break;
        }
        current.push(component);
        let metadata = fs::symlink_metadata(&current).map_err(|error| {
            Error::Contract(format!(
                "cannot inspect materialized artifact directory {}: {error}",
                current.display()
            ))
        })?;
        if !metadata.file_type().is_dir() {
            return Err(Error::Contract(format!(
                "materialized artifact path crosses a non-directory or symlink: {}",
                current.display()
            )));
        }
    }
    Ok(())
}

fn require_relative_path(value: &str) -> Result<(), Error> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::Contract(format!(
            "artifact path is not a clean relative path: {value:?}"
        )));
    }
    Ok(())
}

fn require_read_only(label: &str, mode: u32) -> Result<(), Error> {
    if mode & WRITE_BITS != 0 {
        return Err(Error::Contract(format!(
            "{label} is writable; rematerialize the artifact before launch"
        )));
    }
    Ok(())
}

fn require_not_group_or_other_writable(label: &str, mode: u32) -> Result<(), Error> {
    if mode & 0o022 != 0 {
        return Err(Error::Contract(format!(
            "{label} is group- or other-writable"
        )));
    }
    Ok(())
}

fn unix_nanoseconds(seconds: i64, nanoseconds: i64) -> Result<i64, Error> {
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds))
        .ok_or_else(|| Error::Contract("artifact file timestamp exceeds I64".to_owned()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u32,
    kind: String,
    manifest_sha256: String,
    file_count: usize,
    total_bytes: u64,
    verified_at_unix_nanoseconds: i64,
    files: Vec<ReceiptFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptFile {
    path: String,
    size: u64,
    device: u64,
    inode: u64,
    mode: u32,
    modified_unix_nanoseconds: i64,
    changed_unix_nanoseconds: i64,
}

#[derive(Debug)]
pub(super) enum Error {
    Io(std::io::Error),
    Contract(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Contract(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs::File;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn unchanged_read_only_materialization_is_accepted() {
        let fixture = Fixture::new();
        validate_materialization(
            &fixture.root,
            fixture.manifest,
            [ExpectedFile {
                path: fixture.name,
                size: 4,
            }],
        )
        .unwrap();
    }

    #[test]
    fn same_size_change_is_rejected_without_rehashing() {
        let fixture = Fixture::new();
        let path = fixture.root.join(fixture.name);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        fs::write(&path, b"wxyz").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o444)).unwrap();
        let error = validate_materialization(
            &fixture.root,
            fixture.manifest,
            [ExpectedFile {
                path: fixture.name,
                size: 4,
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("changed after content verification"));
    }

    #[test]
    fn writable_artifact_is_never_treated_as_materialized() {
        let fixture = Fixture::new();
        let path = fixture.root.join(fixture.name);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let error = validate_materialization(
            &fixture.root,
            fixture.manifest,
            [ExpectedFile {
                path: fixture.name,
                size: 4,
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("is writable"));
    }

    #[test]
    fn absent_receipt_requires_materialization_instead_of_rehashing() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.root.join(RECEIPT_NAME)).unwrap();
        let error = validate_materialization(
            &fixture.root,
            fixture.manifest,
            [ExpectedFile {
                path: fixture.name,
                size: 4,
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("receipt is absent"));
        assert!(error.to_string().contains("materializer"));
    }

    #[cfg(unix)]
    #[test]
    fn intermediate_directory_symlink_is_rejected_at_launch() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new();
        let outside = fixture.root.with_extension("outside");
        fs::create_dir(&outside).unwrap();
        let outside_file = outside.join(fixture.name);
        fs::write(&outside_file, b"abcd").unwrap();
        fs::set_permissions(&outside_file, fs::Permissions::from_mode(0o444)).unwrap();
        let metadata = fs::symlink_metadata(&outside_file).unwrap();
        let linked = fixture.root.join("linked");
        symlink(&outside, &linked).unwrap();
        let receipt_path = fixture.root.join(RECEIPT_NAME);
        fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o644)).unwrap();
        let receipt = json!({
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "kind": RECEIPT_KIND,
            "manifest_sha256": fixture.manifest,
            "file_count": 1,
            "total_bytes": 4,
            "verified_at_unix_nanoseconds": 1,
            "files": [{
                "path": "linked/weights.bin",
                "size": 4,
                "device": metadata.dev(),
                "inode": metadata.ino(),
                "mode": metadata.mode() & PERMISSION_BITS,
                "modified_unix_nanoseconds": unix_nanoseconds(metadata.mtime(), metadata.mtime_nsec()).unwrap(),
                "changed_unix_nanoseconds": unix_nanoseconds(metadata.ctime(), metadata.ctime_nsec()).unwrap(),
            }],
        });
        fs::write(&receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        fs::set_permissions(&receipt_path, fs::Permissions::from_mode(0o444)).unwrap();
        let error = validate_materialization(
            &fixture.root,
            fixture.manifest,
            [ExpectedFile {
                path: "linked/weights.bin",
                size: 4,
            }],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("crosses a non-directory or symlink"));
        fs::remove_dir_all(outside).unwrap();
    }

    struct Fixture {
        root: std::path::PathBuf,
        manifest: &'static str,
        name: &'static str,
    }

    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "nml-materialization-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let name = "weights.bin";
            let path = root.join(name);
            fs::write(&path, b"abcd").unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o444)).unwrap();
            let metadata = fs::symlink_metadata(&path).unwrap();
            let manifest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
            let receipt = json!({
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "kind": RECEIPT_KIND,
                "manifest_sha256": manifest,
                "file_count": 1,
                "total_bytes": 4,
                "verified_at_unix_nanoseconds": 1,
                "files": [{
                    "path": name,
                    "size": 4,
                    "device": metadata.dev(),
                    "inode": metadata.ino(),
                    "mode": metadata.mode() & PERMISSION_BITS,
                    "modified_unix_nanoseconds": unix_nanoseconds(metadata.mtime(), metadata.mtime_nsec()).unwrap(),
                    "changed_unix_nanoseconds": unix_nanoseconds(metadata.ctime(), metadata.ctime_nsec()).unwrap(),
                }],
            });
            let receipt_path = root.join(RECEIPT_NAME);
            let mut file = File::create(&receipt_path).unwrap();
            serde_json::to_writer(&mut file, &receipt).unwrap();
            file.write_all(b"\n").unwrap();
            drop(file);
            fs::set_permissions(receipt_path, fs::Permissions::from_mode(0o444)).unwrap();
            Self {
                root,
                manifest,
                name,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }
}
