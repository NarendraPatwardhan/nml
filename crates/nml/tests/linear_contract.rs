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
const HIDDEN: usize = 5;
const MLP_BATCH: usize = 4;

#[derive(nml::NmlStruct)]
struct Linear {
    weight: nml::Tensor,
    bias: Option<nml::Tensor>,
}

#[derive(nml::NmlStruct)]
struct Mlp {
    first: Linear,
    second: Linear,
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
        checkpoint_backed_two_layer_mlp(&platform, dtype);
    }
    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        algebra_shape_and_activation_graph_executes(&platform, dtype);
    }
    if platform.name() == "cpu" {
        tiled_cpu_placement_round_trips(&platform);
        sharded_contraction_executes_with_compiler_communication(&platform);
    }
    tied_parameters_load_once_and_share_storage(&platform);
    truncated_checkpoint_releases_in_flight_transfers(&platform);
    activation_donation_aliases_the_output(&platform);
}

fn checkpoint_backed_two_layer_mlp(platform: &nml::Platform, dtype: DType) {
    let root = temporary_directory(dtype, true, false).with_extension("mlp");
    std::fs::create_dir_all(&root).unwrap();
    let first_weight_values = (0..HIDDEN * INPUTS)
        .map(|index| (index as f32 - 7.0) / 16.0)
        .collect::<Vec<_>>();
    let first_bias_values = [-0.125, 0.25, 0.0, 0.375, -0.25];
    let second_weight_values = (0..OUTPUTS * HIDDEN)
        .map(|index| (5.0 - index as f32) / 13.0)
        .collect::<Vec<_>>();
    let second_bias_values = [0.0625, -0.125, 0.25];
    let first_weight = tensor_data(dtype, &[HIDDEN, INPUTS], &first_weight_values);
    let first_bias = tensor_data(dtype, &[HIDDEN], &first_bias_values);
    let second_weight = tensor_data(dtype, &[OUTPUTS, HIDDEN], &second_weight_values);
    let second_bias = tensor_data(dtype, &[OUTPUTS], &second_bias_values);
    write_file(
        &root.join("model.safetensors"),
        BTreeMap::from([
            ("first.weight", &first_weight),
            ("first.bias", &first_bias),
            ("second.weight", &second_weight),
            ("second.bias", &second_bias),
        ]),
    );

    let registry = nml::safetensors::TensorRegistry::from_path(&root).unwrap();
    let data_axis = nml::AxisTag::new(91);
    let mesh_size = platform.device_count().unwrap();
    let placements = [
        (nml::Sharding::single(), None),
        (nml::Sharding::replicated(), None),
        (
            nml::Sharding::mesh(&[(data_axis, mesh_size)]).unwrap(),
            Some(data_axis),
        ),
    ];
    for (placement, partition) in placements {
        execute_checkpoint_backed_mlp(
            platform,
            registry.clone(),
            dtype,
            placement,
            partition,
            &first_weight_values,
            &first_bias_values,
            &second_weight_values,
            &second_bias_values,
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn execute_checkpoint_backed_mlp(
    platform: &nml::Platform,
    registry: nml::safetensors::TensorRegistry,
    dtype: DType,
    placement: nml::Sharding,
    partition: Option<nml::AxisTag>,
    first_weight_values: &[f32],
    first_bias_values: &[f32],
    second_weight_values: &[f32],
    second_bias_values: &[f32],
) {
    let store = nml::io::TensorStore::new(registry);
    let first_store = store.view("first");
    let second_store = store.view("second");
    let first = Linear {
        weight: first_store
            .tensor(
                "weight",
                Shape::new(dtype, &[HIDDEN as i64, INPUTS as i64]).unwrap(),
                &[],
            )
            .unwrap(),
        bias: Some(
            first_store
                .tensor("bias", Shape::new(dtype, &[HIDDEN as i64]).unwrap(), &[])
                .unwrap(),
        ),
    };
    let second = Linear {
        weight: second_store
            .tensor(
                "weight",
                Shape::new(dtype, &[OUTPUTS as i64, HIDDEN as i64]).unwrap(),
                &[],
            )
            .unwrap(),
        bias: Some(
            second_store
                .tensor("bias", Shape::new(dtype, &[OUTPUTS as i64]).unwrap(), &[])
                .unwrap(),
        ),
    };
    let model = Mlp { first, second };
    let mut input_shape = Shape::new(dtype, &[MLP_BATCH as i64, INPUTS as i64]).unwrap();
    if let Some(axis) = partition {
        input_shape = input_shape
            .with_partitions(&[nml::Partition::Sharded(axis), nml::Partition::Unspecified])
            .unwrap();
    }
    let input = store.activation("input", input_shape);
    let hidden = store
        .linear(input, model.first.weight, model.first.bias)
        .unwrap();
    let hidden = store.gelu(hidden).unwrap();
    let output = store
        .linear(hidden, model.second.weight, model.second.bias)
        .unwrap();
    let options = nml::io::LoadOptions::new(placement.clone());
    let parameters = store.load(&model, platform, &options).unwrap();
    drop(first_store);
    drop(second_store);
    let program = store.finish(&[("output".to_owned(), output)]).unwrap();
    let executable = platform.compile(&program, placement.clone()).unwrap();
    let mut arguments = executable.args();
    arguments
        .set("first.weight", parameters.first.weight.clone())
        .unwrap();
    arguments
        .set(
            "first.bias",
            parameters.first.bias.as_ref().unwrap().clone(),
        )
        .unwrap();
    arguments
        .set("second.weight", parameters.second.weight.clone())
        .unwrap();
    arguments
        .set(
            "second.bias",
            parameters.second.bias.as_ref().unwrap().clone(),
        )
        .unwrap();
    arguments.bake().unwrap();

    for invocation in 0..2 {
        let activations = (0..MLP_BATCH * INPUTS)
            .map(|index| (index as f32 - 6.0 + invocation as f32 * 0.75) / 7.0)
            .collect::<Vec<_>>();
        let host = tensor_data(dtype, &[MLP_BATCH, INPUTS], &activations);
        let activation = platform
            .upload(
                &nml::Slice::from_bytes(input_shape, &host.bytes).unwrap(),
                placement.clone(),
                nml::Memory::Default,
            )
            .unwrap();
        arguments.set("input", activation).unwrap();
        let results = arguments.call().unwrap();
        let actual = results.get("output").unwrap().to_slice().unwrap();
        let hidden = reference_layer(
            &activations,
            MLP_BATCH,
            INPUTS,
            HIDDEN,
            first_weight_values,
            first_bias_values,
        )
        .into_iter()
        .map(|value| gelu_reference(round_to_dtype(dtype, value)))
        .collect::<Vec<_>>();
        let expected = reference_layer(
            &hidden,
            MLP_BATCH,
            HIDDEN,
            OUTPUTS,
            second_weight_values,
            second_bias_values,
        );
        assert_nonlinear_close(dtype, actual.contiguous_bytes().unwrap(), &expected);
    }
}

fn sharded_contraction_executes_with_compiler_communication(platform: &nml::Platform) {
    let contract = nml::AxisTag::new(77);
    let left_shape = Shape::new(DType::F32, &[2, 8])
        .unwrap()
        .with_partitions(&[
            nml::Partition::Unspecified,
            nml::Partition::Sharded(contract),
        ])
        .unwrap();
    let right_shape = Shape::new(DType::F32, &[8, 3])
        .unwrap()
        .with_partitions(&[
            nml::Partition::Sharded(contract),
            nml::Partition::Unspecified,
        ])
        .unwrap();
    let mesh = nml::Sharding::mesh(&[(contract, 4)]).unwrap();
    let mut builder = nml_ir::ProgramBuilder::new();
    let left = builder.input("left", left_shape);
    let right = builder.input("right", right_shape);
    let product = builder.matmul(left, right).unwrap();
    let program = builder
        .finish_named(&[("product".to_owned(), product)])
        .unwrap();
    let executable = platform.compile(&program, mesh.clone()).unwrap();
    let left_values = (0..16)
        .map(|value| value as f32 / 8.0 - 1.0)
        .collect::<Vec<_>>();
    let right_values = (0..24)
        .map(|value| 0.5 - value as f32 / 32.0)
        .collect::<Vec<_>>();
    let left_slice = nml::Slice::from_typed(left_shape, &left_values).unwrap();
    let right_slice = nml::Slice::from_typed(right_shape, &right_values).unwrap();
    let mut arguments = executable.args();
    arguments
        .set(
            "left",
            platform
                .upload(&left_slice, mesh.clone(), nml::Memory::Default)
                .unwrap(),
        )
        .unwrap();
    arguments
        .set(
            "right",
            platform
                .upload(&right_slice, mesh, nml::Memory::Default)
                .unwrap(),
        )
        .unwrap();
    let results = arguments.call().unwrap();
    let output = results.get("product").unwrap().to_slice().unwrap();
    let actual = output.items::<f32>().unwrap();
    let mut expected = vec![0.0f32; 6];
    for batch in 0..2 {
        for output_axis in 0..3 {
            for contract_axis in 0..8 {
                expected[batch * 3 + output_axis] += left_values[batch * 8 + contract_axis]
                    * right_values[contract_axis * 3 + output_axis];
            }
        }
    }
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs());
    }
}

fn reference_layer(
    input: &[f32],
    batch_size: usize,
    input_size: usize,
    output_size: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut result = vec![0.0; batch_size * output_size];
    for batch in 0..batch_size {
        for output in 0..output_size {
            let mut value = bias[output];
            for input_axis in 0..input_size {
                value += input[batch * input_size + input_axis]
                    * weight[output * input_size + input_axis];
            }
            result[batch * output_size + output] = value;
        }
    }
    result
}

fn round_to_dtype(dtype: DType, value: f32) -> f32 {
    match dtype {
        DType::F16 => F16::from_f32(value).to_f32(),
        DType::Bf16 => BFloat16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

fn gelu_reference(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
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
    let executable = platform
        .compile(&program, nml::Sharding::replicated())
        .unwrap();
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
    let data_axis = nml::AxisTag::new(1);
    let shape = Shape::new(DType::F32, &[8, 4])
        .unwrap()
        .with_partitions(&[
            nml::Partition::Sharded(data_axis),
            nml::Partition::Unspecified,
        ])
        .unwrap();
    let values = (0..32).map(|value| value as f32 - 7.5).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let slice = nml::Slice::from_bytes(shape, &bytes).unwrap();
    let buffer = platform
        .upload(
            &slice,
            nml::Sharding::mesh(&[(data_axis, 4)]).unwrap(),
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
    // CUDA's one-device logical mesh still exercises the partitioned DMA
    // dispatcher. CPU uses all configured devices as replicas; its real mesh
    // loading contract is covered by the nonlinear model above.
    let placement = if platform.name() == "cuda" {
        nml::Sharding::mesh(&[(nml::AxisTag::new(1), 1)]).unwrap()
    } else {
        nml::Sharding::replicated()
    };
    let options = nml::io::LoadOptions::new(placement)
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
    let executable = platform
        .compile(&program, nml::Sharding::replicated())
        .unwrap();

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

fn algebra_shape_and_activation_graph_executes(platform: &nml::Platform, dtype: DType) {
    let shape = Shape::new(dtype, &[2, 3]).unwrap();
    let mut builder = nml_ir::ProgramBuilder::new();
    let input = builder.input("input", shape);
    let half = scalar(&mut builder, dtype, 0.5);
    let two = scalar(&mut builder, dtype, 2.0);
    let negative_half = builder.negate(half).unwrap();
    let arithmetic = builder.subtract(input, half).unwrap();
    let arithmetic = builder.multiply(arithmetic, two).unwrap();
    let arithmetic = builder.divide(arithmetic, two).unwrap();
    let arithmetic = builder.minimum(arithmetic, half).unwrap();
    let arithmetic = builder.maximum(arithmetic, negative_half).unwrap();
    let arithmetic = builder.negate(arithmetic).unwrap();

    let equal = builder.equal(input, half).unwrap();
    let not_equal = builder.not_equal(input, half).unwrap();
    let greater = builder.greater(input, half).unwrap();
    let greater_equal = builder.greater_equal(input, half).unwrap();
    let less = builder.less(input, half).unwrap();
    let less_equal = builder.less_equal(input, half).unwrap();
    let selected = builder.select(greater, input, arithmetic).unwrap();
    let converted = builder.convert(selected, DType::F32).unwrap();

    let reshaped = builder
        .reshape(input, Shape::new(dtype, &[3, 2]).unwrap())
        .unwrap();
    let transposed = builder.transpose(reshaped, &[1, 0]).unwrap();
    let exponential = builder.exp(input).unwrap();
    let logarithm = builder.log(input).unwrap();
    let square_root = builder.sqrt(input).unwrap();
    let reciprocal_root = builder.rsqrt(input).unwrap();
    let hyperbolic_tangent = builder.tanh(input).unwrap();
    let sine = builder.sin(input).unwrap();
    let cosine = builder.cos(input).unwrap();
    let relu = builder.relu(input).unwrap();
    let sigmoid = builder.sigmoid(input).unwrap();
    let silu = builder.silu(input).unwrap();
    let gelu = builder.gelu(input).unwrap();
    let leaky_relu = builder.leaky_relu(input, 0.125).unwrap();
    let quick_gelu = builder.quick_gelu(input).unwrap();
    let program = builder
        .finish_named(&[
            ("arithmetic".to_owned(), arithmetic),
            ("equal".to_owned(), equal),
            ("not_equal".to_owned(), not_equal),
            ("greater".to_owned(), greater),
            ("greater_equal".to_owned(), greater_equal),
            ("less".to_owned(), less),
            ("less_equal".to_owned(), less_equal),
            ("selected_f32".to_owned(), converted),
            ("transposed".to_owned(), transposed),
            ("exp".to_owned(), exponential),
            ("log".to_owned(), logarithm),
            ("sqrt".to_owned(), square_root),
            ("rsqrt".to_owned(), reciprocal_root),
            ("tanh".to_owned(), hyperbolic_tangent),
            ("sin".to_owned(), sine),
            ("cos".to_owned(), cosine),
            ("relu".to_owned(), relu),
            ("sigmoid".to_owned(), sigmoid),
            ("silu".to_owned(), silu),
            ("gelu".to_owned(), gelu),
            ("leaky_relu".to_owned(), leaky_relu),
            ("quick_gelu".to_owned(), quick_gelu),
        ])
        .unwrap();
    let placement = nml::Sharding::replicated();
    let executable = platform.compile(&program, placement.clone()).unwrap();
    let values = [0.25f32, 0.5, 0.75, 1.0, 1.25, 1.5];
    let bytes = encode(dtype, &values);
    let input = platform
        .upload(
            &nml::Slice::from_bytes(shape, &bytes).unwrap(),
            placement,
            nml::Memory::Default,
        )
        .unwrap();
    let mut arguments = executable.args();
    arguments.set("input", input).unwrap();
    let results = arguments.call().unwrap();

    let arithmetic = values.map(|value| -((value - 0.5).min(0.5).max(-0.5)));
    let selected = std::array::from_fn::<_, 6, _>(|index| {
        if values[index] > 0.5 {
            values[index]
        } else {
            arithmetic[index]
        }
    });
    assert_result_close(&results, "arithmetic", dtype, &arithmetic);
    assert_eq!(result_bytes(&results, "equal"), &[0, 1, 0, 0, 0, 0]);
    assert_eq!(result_bytes(&results, "not_equal"), &[1, 0, 1, 1, 1, 1]);
    assert_eq!(result_bytes(&results, "greater"), &[0, 0, 1, 1, 1, 1]);
    assert_eq!(result_bytes(&results, "greater_equal"), &[0, 1, 1, 1, 1, 1]);
    assert_eq!(result_bytes(&results, "less"), &[1, 0, 0, 0, 0, 0]);
    assert_eq!(result_bytes(&results, "less_equal"), &[1, 1, 0, 0, 0, 0]);
    assert_result_close(&results, "selected_f32", DType::F32, &selected);
    assert_result_close(
        &results,
        "transposed",
        dtype,
        &[0.25, 0.75, 1.25, 0.5, 1.0, 1.5],
    );
    let references: [(&str, fn(f32) -> f32); 13] = [
        ("exp", f32::exp),
        ("log", f32::ln),
        ("sqrt", f32::sqrt),
        ("rsqrt", |value| value.sqrt().recip()),
        ("tanh", f32::tanh),
        ("sin", f32::sin),
        ("cos", f32::cos),
        ("relu", |value| value.max(0.0)),
        ("sigmoid", |value| 1.0 / (1.0 + (-value).exp())),
        ("silu", |value| value / (1.0 + (-value).exp())),
        ("gelu", gelu_reference),
        ("leaky_relu", |value| value.max(0.125 * value)),
        ("quick_gelu", |value| value / (1.0 + (-1.702 * value).exp())),
    ];
    for (name, reference) in references {
        let expected = values.map(reference);
        assert_result_close(&results, name, dtype, &expected);
    }
}

fn scalar(builder: &mut nml_ir::ProgramBuilder, dtype: DType, value: f32) -> nml::Tensor {
    match dtype {
        DType::F32 => builder.scalar(value).unwrap(),
        DType::F16 => builder.scalar(F16::from_f32(value)).unwrap(),
        DType::Bf16 => builder.scalar(BFloat16::from_f32(value)).unwrap(),
        _ => unreachable!(),
    }
}

fn encode(dtype: DType, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| match dtype {
            DType::F32 => value.to_ne_bytes().to_vec(),
            DType::F16 => F16::from_f32(*value).to_bits().to_ne_bytes().to_vec(),
            DType::Bf16 => BFloat16::from_f32(*value).to_bits().to_ne_bytes().to_vec(),
            _ => unreachable!(),
        })
        .collect()
}

fn result_bytes(results: &nml::exe::Results, name: &str) -> Vec<u8> {
    results
        .get(name)
        .unwrap()
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .to_vec()
}

fn assert_result_close(results: &nml::exe::Results, name: &str, dtype: DType, expected: &[f32]) {
    let bytes = result_bytes(results, name);
    let actual = match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|bytes| F16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|bytes| {
                BFloat16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32()
            })
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    assert_eq!(actual.len(), expected.len());
    let tolerance = match dtype {
        DType::F32 => 2e-5,
        DType::F16 => 6e-3,
        DType::Bf16 => 3e-2,
        _ => unreachable!(),
    };
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance + tolerance * expected.abs(),
            "{name}[{index}]: expected {expected}, received {actual}"
        );
    }
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

fn assert_nonlinear_close(dtype: DType, actual: &[u8], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len() * 2);
    let tolerance = match dtype {
        DType::F16 => 2e-3,
        DType::Bf16 => 1.5e-2,
        _ => unreachable!(),
    };
    for (index, (bytes, expected)) in actual.chunks_exact(2).zip(expected).enumerate() {
        let actual = decode(dtype, u16::from_ne_bytes(bytes.try_into().unwrap()));
        assert!(
            (actual - expected).abs() <= tolerance + tolerance * expected.abs(),
            "nonlinear output {index}: expected {expected}, received {actual}"
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
