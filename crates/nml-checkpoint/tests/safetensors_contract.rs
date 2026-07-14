use nml_checkpoint::safetensors::TensorRegistry;
use nml_types::{DType, Shape};
use safetensors::tensor::{Dtype, View};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::PathBuf;

struct TensorData {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for &TensorData {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

#[test]
fn registry_reads_single_and_indexed_files_without_persisting_a_copy() {
    let root = temporary_directory("registry");
    std::fs::create_dir_all(&root).unwrap();
    let weight = TensorData {
        dtype: Dtype::F16,
        shape: vec![2, 2],
        bytes: [1u16, 2, 3, 4]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
    };
    let bias = TensorData {
        dtype: Dtype::BF16,
        shape: vec![2],
        bytes: [5u16, 6].into_iter().flat_map(u16::to_le_bytes).collect(),
    };
    write_file(&root.join("a.safetensors"), [("weight", &weight)]);
    write_file(&root.join("b.safetensors"), [("bias", &bias)]);
    let index = serde_json::json!({
        "metadata": {"total_size": 12},
        "weight_map": {"weight": "a.safetensors", "bias": "b.safetensors"}
    });
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();

    let registry = TensorRegistry::from_path(&root).unwrap();
    assert_eq!(registry.names().collect::<Vec<_>>(), ["bias", "weight"]);
    assert_eq!(
        registry.shape("weight").unwrap(),
        Shape::new(DType::F16, &[2, 2]).unwrap()
    );
    assert_eq!(
        registry.read("bias").unwrap().contiguous_bytes().unwrap(),
        bias.bytes
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn unsupported_subbyte_dtype_is_rejected_before_tensor_loading() {
    let root = temporary_directory("dtype");
    std::fs::create_dir_all(&root).unwrap();
    let tensor = TensorData {
        dtype: Dtype::F8_E4M3,
        shape: vec![1],
        bytes: vec![0],
    };
    let path = root.join("model.safetensors");
    write_file(&path, [("unsupported", &tensor)]);
    assert!(TensorRegistry::from_path(path).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn malformed_extent_duplicate_names_and_escaping_shards_are_rejected() {
    let root = temporary_directory("malformed");
    std::fs::create_dir_all(&root).unwrap();
    let tensor = TensorData {
        dtype: Dtype::F16,
        shape: vec![1],
        bytes: 7u16.to_le_bytes().to_vec(),
    };

    let truncated = root.join("truncated.safetensors");
    let mut bytes = safetensors::serialize(BTreeMap::from([("x", &tensor)]), None).unwrap();
    bytes.pop();
    std::fs::write(&truncated, bytes).unwrap();
    assert!(TensorRegistry::from_path(&truncated).is_err());

    let duplicate = root.join("duplicate.safetensors");
    let header = br#"{"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&tensor.bytes);
    std::fs::write(&duplicate, bytes).unwrap();
    assert!(TensorRegistry::from_path(&duplicate).is_err());

    let outside = root.with_extension("outside.safetensors");
    write_file(&outside, [("x", &tensor)]);
    let outside_name = outside.file_name().unwrap().to_str().unwrap();
    let index = format!("{{\"weight_map\":{{\"x\":\"../{outside_name}\"}},\"metadata\":{{}}}}");
    std::fs::write(root.join("model.safetensors.index.json"), index).unwrap();
    assert!(TensorRegistry::from_path(&root).is_err());

    std::fs::remove_file(outside).unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn primary_names_win_and_multiple_fallback_aliases_are_ambiguous() {
    let root = temporary_directory("aliases");
    std::fs::create_dir_all(&root).unwrap();
    let tensor = TensorData {
        dtype: Dtype::F16,
        shape: vec![1],
        bytes: 9u16.to_le_bytes().to_vec(),
    };
    write_file(
        &root.join("model.safetensors"),
        [
            ("primary", &tensor),
            ("alias_a", &tensor),
            ("alias_b", &tensor),
        ],
    );
    let registry = TensorRegistry::from_path(&root).unwrap();
    let shape = Shape::new(DType::F16, &[1]).unwrap();
    let store = nml_checkpoint::io::TensorStore::new(registry);
    assert!(
        store
            .tensor("primary", shape, &["alias_a", "alias_b"])
            .is_ok()
    );
    assert!(
        store
            .tensor("absent", shape, &["alias_a", "alias_b"])
            .is_err()
    );
    std::fs::remove_dir_all(root).unwrap();
}

fn write_file<'a>(
    path: &std::path::Path,
    tensors: impl IntoIterator<Item = (&'a str, &'a TensorData)>,
) {
    let tensors = tensors.into_iter().collect::<BTreeMap<_, _>>();
    let bytes = safetensors::serialize(tensors, None).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn temporary_directory(label: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("nml-{label}-{}-{nonce}", std::process::id()))
}
