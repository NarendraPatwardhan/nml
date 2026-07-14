//! Product contract for persistent FP16/BF16 checkpoint parameters.

use nml_types::{BFloat16, DType, F16, Shape};
use safetensors::tensor::{Dtype as SafeDType, View};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const BATCH: usize = 3;
const INPUTS: usize = 4;
const OUTPUTS: usize = 3;

#[derive(nml::NmlStruct)]
struct Linear {
    weight: nml::Tensor,
    bias: Option<nml::Tensor>,
}

#[derive(nml::NmlStruct)]
struct TiedParameters {
    first: nml::Tensor,
    second: nml::Tensor,
}

struct TensorData {
    dtype: SafeDType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for &TensorData {
    fn dtype(&self) -> SafeDType {
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
fn persistent_linear_parameters_execute_repeatedly_from_real_checkpoints() {
    let platform = platform();
    for dtype in [DType::F16, DType::Bf16] {
        for with_bias in [false, true] {
            for sharded in [false, true] {
                run_variant(&platform, dtype, with_bias, sharded);
            }
        }
    }
    if platform.name() == "cpu" {
        tiled_cpu_placement_round_trips(&platform);
    }
    tied_parameters_load_once_and_share_storage(&platform);
    truncated_checkpoint_releases_in_flight_transfers(&platform);
    activation_donation_aliases_the_output(&platform);
}

fn platform() -> nml::Platform {
    match env!("NML_LINEAR_BACKEND") {
        "cpu" => nml::Platform::cpu().expect("CPU PJRT must initialize"),
        "cuda" => {
            // SAFETY: Bazel starts this test as a single-threaded process and
            // platform initialization precedes every other XLA/PJRT call.
            unsafe { nml::Platform::cuda() }.expect("CUDA PJRT must initialize on a supported GPU")
        }
        backend => panic!("unknown test backend {backend}"),
    }
}

fn run_variant(platform: &nml::Platform, dtype: DType, with_bias: bool, sharded: bool) {
    let root = temporary_directory(dtype, with_bias, sharded);
    std::fs::create_dir_all(&root).unwrap();
    let weight_values = [
        0.25, -0.5, 0.75, 1.0, -0.125, 0.375, -0.625, 0.875, 0.5, 0.25, -0.25, -0.5,
    ];
    let bias_values = [0.125, -0.25, 0.5];
    let weight = tensor_data(dtype, &[OUTPUTS, INPUTS], &weight_values);
    let bias = tensor_data(dtype, &[OUTPUTS], &bias_values);
    write_checkpoint(&root, &weight, with_bias.then_some(&bias), sharded);

    let registry = nml::safetensors::TensorRegistry::from_path(&root).unwrap();
    let store = nml::io::TensorStore::new(registry);
    let weight_shape = Shape::new(dtype, &[OUTPUTS as i64, INPUTS as i64]).unwrap();
    let bias_shape = Shape::new(dtype, &[OUTPUTS as i64]).unwrap();
    let input_shape = Shape::new(dtype, &[BATCH as i64, INPUTS as i64]).unwrap();
    let weight_tensor = store.tensor("weight", weight_shape, &[]).unwrap();
    let bias_tensor = if with_bias {
        Some(store.tensor("bias", bias_shape, &[]).unwrap())
    } else {
        store.maybe_tensor("bias", bias_shape, &[]).unwrap()
    };
    let model = Linear {
        weight: weight_tensor,
        bias: bias_tensor,
    };
    let input = store.activation("input", input_shape);
    let output = store.linear(input, model.weight, model.bias).unwrap();
    let progress = Arc::new(Mutex::new(Vec::new()));
    let progress_log = Arc::clone(&progress);
    let load_options = nml::io::LoadOptions::new(nml::Sharding::replicated())
        .parallelism(2)
        .unwrap()
        .staging(2, 8)
        .unwrap()
        .progress(move |completed, total| {
            progress_log.lock().unwrap().push((completed, total));
        });
    let parameters = store.load(&model, platform, &load_options).unwrap();
    let unique_parameters = usize::from(with_bias) + 1;
    assert_eq!(
        progress.lock().unwrap().last().copied(),
        Some((unique_parameters, unique_parameters))
    );
    let original_weight = parameters
        .weight
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .to_vec();
    let program = store.finish(&[("output".to_owned(), output)]).unwrap();
    let executable = platform.compile(&program).unwrap();
    let mut arguments = executable.args();
    arguments.set("weight", parameters.weight.clone()).unwrap();
    if let Some(bias) = &parameters.bias {
        arguments.set("bias", bias.clone()).unwrap();
    }
    arguments.bake().unwrap();

    let activation_sets = [
        [
            1.0, 0.5, -0.25, 0.125, -0.5, 0.75, 0.25, -1.0, 0.0, 0.125, 0.5, 1.0,
        ],
        [
            -0.75, 0.25, 1.0, 0.5, 0.25, -0.125, 0.875, -0.5, 1.0, -1.0, 0.5, 0.25,
        ],
        [
            0.0625, -0.125, 0.25, -0.5, 0.75, 0.625, -0.375, 0.125, -0.25, 0.5, 0.75, -1.0,
        ],
    ];
    for activations in activation_sets {
        let host = tensor_data(dtype, &[BATCH, INPUTS], &activations);
        let slice = nml::Slice::from_bytes(input_shape, &host.bytes).unwrap();
        let activation = platform
            .upload(&slice, nml::Sharding::replicated(), nml::Memory::Default)
            .unwrap();
        arguments.set("input", activation).unwrap();
        let results = arguments.call().unwrap();
        let actual = results.get("output").unwrap().to_slice().unwrap();
        let actual = actual.contiguous_bytes().unwrap();
        let expected = reference(
            &activations,
            &weight_values,
            with_bias.then_some(&bias_values),
        );
        assert_rounded_close(dtype, actual, &expected);
    }

    assert_eq!(
        parameters
            .weight
            .to_slice()
            .unwrap()
            .contiguous_bytes()
            .unwrap(),
        original_weight
    );
    std::fs::remove_dir_all(root).unwrap();
}

fn tiled_cpu_placement_round_trips(platform: &nml::Platform) {
    let shape = Shape::new(DType::F32, &[8, 4]).unwrap();
    let values = (0..32).map(|value| value as f32 - 7.5).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let slice = nml::Slice::from_bytes(shape, &bytes).unwrap();
    let buffer = platform
        .upload(
            &slice,
            nml::Sharding::tiled(&[4, 1]).unwrap(),
            nml::Memory::Default,
        )
        .unwrap();
    assert_eq!(buffer.byte_count().unwrap(), bytes.len());
    assert_eq!(
        buffer.to_slice().unwrap().contiguous_bytes().unwrap(),
        bytes
    );
}

fn tied_parameters_load_once_and_share_storage(platform: &nml::Platform) {
    let root = temporary_directory(DType::F16, false, false).with_extension("tied");
    std::fs::create_dir_all(&root).unwrap();
    let shape = Shape::new(DType::F16, &[OUTPUTS as i64, INPUTS as i64]).unwrap();
    let values = [
        0.25, -0.5, 0.75, 1.0, -0.125, 0.375, -0.625, 0.875, 0.5, 0.25, -0.25, -0.5,
    ];
    let tensor = tensor_data(DType::F16, &[OUTPUTS, INPUTS], &values);
    write_file(
        &root.join("model.safetensors"),
        BTreeMap::from([("shared", &tensor)]),
    );

    let registry = nml::safetensors::TensorRegistry::from_path(&root).unwrap();
    let store = nml::io::TensorStore::new(registry);
    let model = TiedParameters {
        first: store.tensor("first", shape, &["shared"]).unwrap(),
        second: store.tensor("second", shape, &["shared"]).unwrap(),
    };
    let progress = Arc::new(Mutex::new(Vec::new()));
    let progress_log = Arc::clone(&progress);
    // A one-tile mesh is still routed through the tiled checkpoint span
    // dispatcher. On the single installed CUDA device this exercises the real
    // tiled DMA implementation without inventing unavailable hardware.
    let options = nml::io::LoadOptions::new(nml::Sharding::tiled(&[1, 1]).unwrap())
        .staging(2, 8)
        .unwrap()
        .progress(move |completed, total| {
            progress_log.lock().unwrap().push((completed, total));
        });
    let parameters = store.load(&model, platform, &options).unwrap();
    assert_eq!(progress.lock().unwrap().as_slice(), &[(1, 1)]);
    assert!(!parameters.first.is_uniquely_owned());
    assert!(!parameters.second.is_uniquely_owned());

    // `Clone` is the tied/shared-storage operation. `copy` must allocate a
    // distinct physical buffer while preserving bytes and placement.
    let copied = parameters.first.copy().unwrap();
    assert!(copied.is_uniquely_owned());
    assert_eq!(
        copied.to_slice().unwrap().contiguous_bytes().unwrap(),
        tensor.bytes
    );
    assert!(parameters.first.clone().delete().is_err());
    copied.delete().unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

fn truncated_checkpoint_releases_in_flight_transfers(platform: &nml::Platform) {
    let root = temporary_directory(DType::F16, false, false).with_extension("truncated");
    std::fs::create_dir_all(&root).unwrap();
    let shape = Shape::new(DType::F16, &[OUTPUTS as i64, INPUTS as i64]).unwrap();
    let tensor = tensor_data(DType::F16, &[OUTPUTS, INPUTS], &[1.0; OUTPUTS * INPUTS]);
    let path = root.join("model.safetensors");
    write_file(&path, BTreeMap::from([("weight", &tensor)]));

    // Metadata is validated while the file is complete. Truncating afterward
    // exercises the loader's partial-read cleanup rather than parser rejection.
    let registry = nml::safetensors::TensorRegistry::from_path(&root).unwrap();
    let length = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(length - 12)
        .unwrap();
    let store = nml::io::TensorStore::new(registry);
    let model = Linear {
        weight: store.tensor("weight", shape, &[]).unwrap(),
        bias: None,
    };
    let options = nml::io::LoadOptions::new(nml::Sharding::replicated())
        .staging(2, 8)
        .unwrap();
    assert!(store.load(&model, platform, &options).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

fn activation_donation_aliases_the_output(platform: &nml::Platform) {
    let shape = Shape::new(DType::F32, &[8]).unwrap();
    let mut builder = nml_ir::ProgramBuilder::new();
    let left = builder.input("left", shape);
    let right = builder.input("right", shape);
    let sum = builder.add(left, right).unwrap();
    let sum = builder.reuse_buffer(sum, left).unwrap();
    let program = builder.finish_named(&[("sum".to_owned(), sum)]).unwrap();
    assert!(program.stablehlo().unwrap().contains("tf.aliasing_output"));
    let executable = platform.compile(&program).unwrap();

    let left_values = [1.0f32, -2.0, 3.5, 4.0, -5.25, 6.0, 7.0, -8.0];
    let right_values = [0.5f32, 1.0, -1.5, 2.0, 3.25, -4.0, 5.0, 6.0];
    let left_bytes = left_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let right_bytes = right_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    if platform.device_count().unwrap() > 1 {
        let single = platform
            .upload(
                &nml::Slice::from_bytes(shape, &left_bytes).unwrap(),
                nml::Sharding::single(),
                nml::Memory::Default,
            )
            .unwrap();
        let mut invalid = executable.args();
        assert!(invalid.set("left", single).is_err());
    }
    if platform.name() == "cpu" {
        let foreign_platform = nml::Platform::cpu().unwrap();
        let foreign = foreign_platform
            .upload(
                &nml::Slice::from_bytes(shape, &left_bytes).unwrap(),
                nml::Sharding::replicated(),
                nml::Memory::Default,
            )
            .unwrap();
        let mut invalid = executable.args();
        assert!(invalid.set("left", foreign).is_err());
    }
    let left = platform
        .upload(
            &nml::Slice::from_bytes(shape, &left_bytes).unwrap(),
            nml::Sharding::replicated(),
            nml::Memory::Default,
        )
        .unwrap();
    let right = platform
        .upload(
            &nml::Slice::from_bytes(shape, &right_bytes).unwrap(),
            nml::Sharding::replicated(),
            nml::Memory::Default,
        )
        .unwrap();
    let mut arguments = executable.args();
    arguments.set("left", left).unwrap();
    arguments.set("right", right).unwrap();
    let results = arguments.call().unwrap();
    let output = results.get("sum").unwrap().to_slice().unwrap();
    let actual = output
        .contiguous_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let expected = left_values
        .iter()
        .zip(right_values)
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    // Donation consumes the activation slot. Re-execution must bind a fresh
    // activation rather than accidentally reusing a deleted PJRT buffer.
    assert!(arguments.call().is_err());
}

fn reference(input: &[f32], weight: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let mut result = vec![0.0; BATCH * OUTPUTS];
    for batch in 0..BATCH {
        for output in 0..OUTPUTS {
            let mut value = bias.map_or(0.0, |values| values[output]);
            for input_axis in 0..INPUTS {
                value += input[batch * INPUTS + input_axis] * weight[output * INPUTS + input_axis];
            }
            result[batch * OUTPUTS + output] = value;
        }
    }
    result
}

fn assert_rounded_close(dtype: DType, actual: &[u8], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len() * 2);
    for (index, (bytes, expected)) in actual.chunks_exact(2).zip(expected).enumerate() {
        let actual_bits = u16::from_ne_bytes(bytes.try_into().unwrap());
        let expected_bits = match dtype {
            DType::F16 => F16::from_f32(*expected).to_bits(),
            DType::Bf16 => BFloat16::from_f32(*expected).to_bits(),
            _ => unreachable!(),
        };
        let actual_value = decode(dtype, actual_bits);
        let expected_value = decode(dtype, expected_bits);
        let ulps = ordered(actual_bits).abs_diff(ordered(expected_bits));
        assert!(
            ulps <= 4 || (actual_value - expected_value).abs() <= 1e-5,
            "element {index}: expected {expected_value} (0x{expected_bits:04x}), received {actual_value} (0x{actual_bits:04x}), distance {ulps} ULPs"
        );
    }
}

fn decode(dtype: DType, bits: u16) -> f32 {
    match dtype {
        DType::F16 => F16::from_bits(bits).to_f32(),
        DType::Bf16 => BFloat16::from_bits(bits).to_f32(),
        _ => unreachable!(),
    }
}

fn ordered(bits: u16) -> u16 {
    if bits & 0x8000 == 0 {
        bits | 0x8000
    } else {
        !bits
    }
}

fn tensor_data(dtype: DType, shape: &[usize], values: &[f32]) -> TensorData {
    let bytes = values
        .iter()
        .flat_map(|value| match dtype {
            DType::F16 => F16::from_f32(*value).to_bits().to_le_bytes(),
            DType::Bf16 => BFloat16::from_f32(*value).to_bits().to_le_bytes(),
            _ => unreachable!(),
        })
        .collect();
    TensorData {
        dtype: match dtype {
            DType::F16 => SafeDType::F16,
            DType::Bf16 => SafeDType::BF16,
            _ => unreachable!(),
        },
        shape: shape.to_vec(),
        bytes,
    }
}

fn write_checkpoint(root: &Path, weight: &TensorData, bias: Option<&TensorData>, sharded: bool) {
    if !sharded {
        let mut tensors = BTreeMap::from([("weight", weight)]);
        if let Some(bias) = bias {
            tensors.insert("bias", bias);
        }
        write_file(&root.join("model.safetensors"), tensors);
        return;
    }

    write_file(
        &root.join("model-00001-of-00002.safetensors"),
        BTreeMap::from([("weight", weight)]),
    );
    let mut weight_map = serde_json::Map::from_iter([(
        "weight".to_owned(),
        "model-00001-of-00002.safetensors".into(),
    )]);
    if let Some(bias) = bias {
        write_file(
            &root.join("model-00002-of-00002.safetensors"),
            BTreeMap::from([("bias", bias)]),
        );
        weight_map.insert("bias".to_owned(), "model-00002-of-00002.safetensors".into());
    }
    let index = serde_json::json!({"metadata": {}, "weight_map": weight_map});
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
}

fn write_file(path: &Path, tensors: BTreeMap<&str, &TensorData>) {
    std::fs::write(path, safetensors::serialize(tensors, None).unwrap()).unwrap();
}

fn temporary_directory(dtype: DType, bias: bool, sharded: bool) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "nml-linear-{dtype:?}-{bias}-{sharded}-{}-{nonce}",
        std::process::id()
    ))
}
