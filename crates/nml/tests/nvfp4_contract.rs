//! End-to-end contract for compact NVFP4 parameters across CPU and CUDA.
//!
//! Bazel compiles this source once per backend. Keeping the graph, compact
//! component upload, reference calculation, and numerical assertions identical
//! prevents CUDA acceptance from becoming a weaker kernel-shaped substitute
//! for the product semantics proved by the CPU reference path.

use nml_parameter::nvfp4::{dequantize_row, global_scale, quantize_row};
use nml_parameter::{ComponentRole, Parameter};
use nml_types::{BFloat16, DType, F16, Shape};

const INPUTS: usize = 16;
const OUTPUTS: usize = 5;

struct EncodedParameter {
    payload: Vec<u8>,
    block_scales: Vec<u8>,
    global: f32,
    decoded: Vec<f32>,
}

#[test]
fn compact_parameters_compile_bind_and_execute_through_pjrt() {
    let platform = platform();
    for dtype in [DType::F16, DType::Bf16] {
        execute_linear(&platform, dtype, 3, false);
        execute_linear(&platform, dtype, 3, true);
        execute_linear(&platform, dtype, 1, true);
        execute_embedding(&platform, dtype);
        execute_experts(&platform, dtype, 2);
        execute_experts(&platform, dtype, 1);
    }
}

fn platform() -> nml::Platform {
    match env!("NML_NVFP4_BACKEND") {
        "cpu" => nml::Platform::cpu().expect("CPU PJRT must initialize"),
        "cuda" => {
            // SAFETY: Bazel starts this contract as a single-threaded process;
            // CUDA platform initialization precedes every other XLA/PJRT call.
            unsafe { nml::Platform::cuda() }
                .expect("CUDA PJRT must initialize on a supported NVIDIA GPU")
        }
        backend => panic!("unknown NVFP4 contract backend {backend}"),
    }
}

fn execute_linear(
    platform: &nml::Platform,
    dtype: DType,
    rows: usize,
    with_bias: bool,
) {
    let input_shape = Shape::new(dtype, &[rows as i64, INPUTS as i64]).unwrap();
    let logical_weight_shape = Shape::new(dtype, &[OUTPUTS as i64, INPUTS as i64]).unwrap();
    let bias_shape = Shape::new(dtype, &[OUTPUTS as i64]).unwrap();
    let weight = Parameter::nvfp4("weight", "model.weight", logical_weight_shape).unwrap();
    let bias = with_bias.then(|| Parameter::dense("bias", "model.bias", bias_shape).unwrap());
    let mut builder = nml_ir::ProgramBuilder::new();
    let input = builder.input("input", input_shape);
    let output = builder.linear(input, &weight, bias.as_ref()).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let placement = nml::Sharding::single();
    let executable = platform.compile(&program, placement.clone()).unwrap();

    let input_values = (0..rows * INPUTS)
        .map(|index| (index as f32 - 19.0) / 13.0)
        .collect::<Vec<_>>();
    let source_weight = (0..OUTPUTS * INPUTS)
        .map(|index| (27.0 - index as f32) / 17.0)
        .collect::<Vec<_>>();
    let bias_values = [-0.5, -0.125, 0.0, 0.25, 0.75];
    let encoded_weight = encode_parameter(&source_weight, INPUTS);
    let no_bias = [0.0; OUTPUTS];
    let expected = reference(
        &input_values,
        &encoded_weight.decoded,
        if with_bias { &bias_values } else { &no_bias },
        rows,
    );

    let mut arguments = executable.args();
    set_buffer(
        platform,
        &mut arguments,
        "input",
        input_shape,
        &encode(dtype, &input_values),
        &placement,
    );
    let loaded_weight = upload_nvfp4(platform, weight, &encoded_weight, &placement);
    arguments.set_parameter(&loaded_weight).unwrap();

    if let Some(bias) = bias {
        let bias_bytes = encode(dtype, &bias_values);
        let bias_slice = nml::Slice::from_bytes(bias_shape, &bias_bytes).unwrap();
        let bias_buffer = platform
            .upload(&bias_slice, placement.clone(), nml::Memory::Default)
            .unwrap();
        let loaded_bias = nml::LoadedParameter::new(bias, vec![bias_buffer]).unwrap();
        arguments.set_parameter(&loaded_bias).unwrap();
    }
    arguments.bake().unwrap();

    let results = arguments.call().unwrap();
    let bytes = results
        .get("output")
        .unwrap()
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .to_vec();
    assert_close(dtype, &bytes, &expected);
}

fn execute_embedding(platform: &nml::Platform, dtype: DType) {
    const VOCABULARY: usize = 7;
    const WIDTH: usize = 17;
    let weight = Parameter::nvfp4_embedding(
        "embedding",
        "model.embedding",
        Shape::new(dtype, &[VOCABULARY as i64, WIDTH as i64]).unwrap(),
    )
    .unwrap();
    let index_shape = Shape::new(DType::I32, &[2, 2]).unwrap();
    let mut builder = nml_ir::ProgramBuilder::new();
    let indices = builder.input("indices", index_shape);
    let output = builder.token_embedding(&weight, indices).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let placement = nml::Sharding::single();
    let executable = platform.compile(&program, placement.clone()).unwrap();

    let source = (0..VOCABULARY * WIDTH)
        .map(|index| (index as f32 - 41.0) / 19.0)
        .collect::<Vec<_>>();
    let encoded = encode_parameter(&source, WIDTH);
    let indices = [6_i32, 0, 3, 3];
    let expected = indices
        .iter()
        .flat_map(|&index| {
            encoded.decoded[index as usize * WIDTH..(index as usize + 1) * WIDTH]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let index_bytes = indices
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();

    let mut arguments = executable.args();
    set_buffer(
        platform,
        &mut arguments,
        "indices",
        index_shape,
        &index_bytes,
        &placement,
    );
    let loaded = upload_nvfp4(platform, weight, &encoded, &placement);
    arguments.set_parameter(&loaded).unwrap();
    arguments.bake().unwrap();
    let results = arguments.call().unwrap();
    let bytes = results
        .get("output")
        .unwrap()
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .to_vec();
    assert_close(dtype, &bytes, &expected);
}

fn execute_experts(platform: &nml::Platform, dtype: DType, tokens: usize) {
    const EXPERTS: usize = 3;
    const HIDDEN: usize = 4;
    const INTERMEDIATE: usize = 5;
    const ROUTES: usize = 2;
    let gate = Parameter::nvfp4(
        "gate",
        "model.gate",
        Shape::new(
            dtype,
            &[EXPERTS as i64, (2 * INTERMEDIATE) as i64, HIDDEN as i64],
        )
        .unwrap(),
    )
    .unwrap();
    let down = Parameter::nvfp4(
        "down",
        "model.down",
        Shape::new(dtype, &[EXPERTS as i64, HIDDEN as i64, INTERMEDIATE as i64]).unwrap(),
    )
    .unwrap();
    let gate_bias = Parameter::dense(
        "gate_bias",
        "model.gate_bias",
        Shape::new(dtype, &[EXPERTS as i64, (2 * INTERMEDIATE) as i64]).unwrap(),
    )
    .unwrap();
    let down_bias = Parameter::dense(
        "down_bias",
        "model.down_bias",
        Shape::new(dtype, &[EXPERTS as i64, HIDDEN as i64]).unwrap(),
    )
    .unwrap();
    let hidden_shape = Shape::new(dtype, &[tokens as i64, HIDDEN as i64]).unwrap();
    let router_shape = Shape::new(DType::F32, &[tokens as i64, EXPERTS as i64]).unwrap();
    let mut builder = nml_ir::ProgramBuilder::new();
    let hidden = builder.input("hidden", hidden_shape);
    let router = builder.input("router", router_shape);
    let output = builder
        .routed_clamped_swiglu(hidden, router, &gate, &gate_bias, &down, &down_bias, ROUTES)
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let placement = nml::Sharding::single();
    let executable = platform.compile(&program, placement.clone()).unwrap();

    let hidden_values = (0..tokens * HIDDEN)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 4.0)
        .collect::<Vec<_>>();
    let router_values = (0..tokens * EXPERTS)
        .map(|index| ((index * 5 % 13) as f32 - 6.0) / 3.0)
        .collect::<Vec<_>>();
    let gate_source = (0..EXPERTS * HIDDEN * 2 * INTERMEDIATE)
        .map(|index| (index as f32 - 41.0) / 23.0)
        .collect::<Vec<_>>();
    let down_source = (0..EXPERTS * INTERMEDIATE * HIDDEN)
        .map(|index| (17.0 - index as f32) / 29.0)
        .collect::<Vec<_>>();
    let gate_encoded = encode_parameter(&gate_source, HIDDEN);
    let down_encoded = encode_parameter(&down_source, INTERMEDIATE);
    let gate_bias_values = (0..EXPERTS * 2 * INTERMEDIATE)
        .map(|index| (index as f32 - 9.0) / 31.0)
        .collect::<Vec<_>>();
    let down_bias_values = (0..EXPERTS * HIDDEN)
        .map(|index| (5.0 - index as f32) / 37.0)
        .collect::<Vec<_>>();
    let expected = reference_experts(
        dtype,
        &hidden_values,
        &router_values,
        &gate_encoded.decoded,
        &gate_bias_values,
        &down_encoded.decoded,
        &down_bias_values,
        tokens,
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        ROUTES,
    );

    let mut arguments = executable.args();
    set_buffer(
        platform,
        &mut arguments,
        "hidden",
        hidden_shape,
        &encode(dtype, &hidden_values),
        &placement,
    );
    let router_bytes = router_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    set_buffer(
        platform,
        &mut arguments,
        "router",
        router_shape,
        &router_bytes,
        &placement,
    );
    for loaded in [
        upload_nvfp4(platform, gate, &gate_encoded, &placement),
        upload_dense(
            platform,
            gate_bias,
            &encode(dtype, &gate_bias_values),
            &placement,
        ),
        upload_nvfp4(platform, down, &down_encoded, &placement),
        upload_dense(
            platform,
            down_bias,
            &encode(dtype, &down_bias_values),
            &placement,
        ),
    ] {
        arguments.set_parameter(&loaded).unwrap();
    }
    arguments.bake().unwrap();
    let results = arguments.call().unwrap();
    let bytes = results
        .get("output")
        .unwrap()
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .to_vec();
    assert_close(dtype, &bytes, &expected);
}

fn encode_parameter(values: &[f32], width: usize) -> EncodedParameter {
    assert_ne!(width, 0);
    assert_eq!(values.len() % width, 0);
    let global = global_scale(values).unwrap();
    let mut payload = Vec::new();
    let mut block_scales = Vec::new();
    let mut decoded = Vec::with_capacity(values.len());
    for row in values.chunks_exact(width) {
        let encoded = quantize_row(row, global).unwrap();
        payload.extend_from_slice(encoded.payload());
        block_scales.extend_from_slice(encoded.block_scales());
        decoded.extend(
            dequantize_row(encoded.payload(), encoded.block_scales(), global, width).unwrap(),
        );
    }
    EncodedParameter {
        payload,
        block_scales,
        global,
        decoded,
    }
}

fn upload_nvfp4(
    platform: &nml::Platform,
    parameter: Parameter,
    encoded: &EncodedParameter,
    placement: &nml::Sharding,
) -> nml::LoadedParameter {
    let global = encoded.global.to_ne_bytes();
    let contraction_major = parameter
        .nvfp4_spec()
        .is_some_and(|spec| spec.is_contraction_major());
    let components = parameter
        .components()
        .iter()
        .map(|component| {
            let bytes = match component.role() {
                ComponentRole::Payload if contraction_major => {
                    transpose_encoded_rows(&encoded.payload, parameter.shape(), 2)
                }
                ComponentRole::Payload => encoded.payload.clone(),
                ComponentRole::BlockScales if contraction_major => {
                    transpose_encoded_rows(&encoded.block_scales, parameter.shape(), 16)
                }
                ComponentRole::BlockScales => encoded.block_scales.clone(),
                ComponentRole::GlobalScale => global.to_vec(),
                ComponentRole::Values => unreachable!(),
            };
            let slice = nml::Slice::from_bytes(component.storage().shape(), &bytes).unwrap();
            platform
                .upload(&slice, placement.clone(), nml::Memory::Default)
                .unwrap()
        })
        .collect();
    nml::LoadedParameter::new(parameter, components).unwrap()
}

/// Converts the fixture codec's logical `[... N, encoded K]` rows into the
/// recipe-v3 contraction component order `[... encoded K, N]`. Production
/// artifacts perform this once in the converter; the test must model those
/// exact persisted bytes rather than relying on a shape-only reinterpretation.
fn transpose_encoded_rows(bytes: &[u8], logical: Shape, values_per_byte: usize) -> Vec<u8> {
    let dimensions = logical.dimensions();
    let outputs = usize::try_from(dimensions[dimensions.len() - 2]).unwrap();
    let inputs = usize::try_from(dimensions[dimensions.len() - 1]).unwrap();
    let encoded_k = inputs.div_ceil(values_per_byte);
    let batches = dimensions[..dimensions.len() - 2]
        .iter()
        .try_fold(1_usize, |product, &dimension| {
            product.checked_mul(usize::try_from(dimension).ok()?)
        })
        .unwrap();
    assert_eq!(bytes.len(), batches * outputs * encoded_k);
    let mut physical = vec![0_u8; bytes.len()];
    for batch in 0..batches {
        let base = batch * outputs * encoded_k;
        for output in 0..outputs {
            for k in 0..encoded_k {
                physical[base + k * outputs + output] = bytes[base + output * encoded_k + k];
            }
        }
    }
    physical
}

fn upload_dense(
    platform: &nml::Platform,
    parameter: Parameter,
    bytes: &[u8],
    placement: &nml::Sharding,
) -> nml::LoadedParameter {
    let slice = nml::Slice::from_bytes(parameter.shape(), bytes).unwrap();
    let buffer = platform
        .upload(&slice, placement.clone(), nml::Memory::Default)
        .unwrap();
    nml::LoadedParameter::new(parameter, vec![buffer]).unwrap()
}

fn set_buffer(
    platform: &nml::Platform,
    arguments: &mut nml::exe::Arguments<'_>,
    name: &str,
    shape: Shape,
    bytes: &[u8],
    placement: &nml::Sharding,
) {
    let slice = nml::Slice::from_bytes(shape, bytes).unwrap();
    arguments
        .set(
            name,
            platform
                .upload(&slice, placement.clone(), nml::Memory::Default)
                .unwrap(),
        )
        .unwrap();
}

fn encode(dtype: DType, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| match dtype {
            DType::F16 => F16::from_f32(*value).to_bits().to_ne_bytes(),
            DType::Bf16 => BFloat16::from_f32(*value).to_bits().to_ne_bytes(),
            _ => unreachable!(),
        })
        .collect()
}

fn reference(input: &[f32], weight: &[f32], bias: &[f32], rows: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * OUTPUTS];
    for row in 0..rows {
        for output_index in 0..OUTPUTS {
            let mut value = bias[output_index];
            for input_index in 0..INPUTS {
                value +=
                    input[row * INPUTS + input_index] * weight[output_index * INPUTS + input_index];
            }
            output[row * OUTPUTS + output_index] = value;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn reference_experts(
    dtype: DType,
    hidden: &[f32],
    router: &[f32],
    gate: &[f32],
    gate_bias: &[f32],
    down: &[f32],
    down_bias: &[f32],
    tokens: usize,
    experts: usize,
    hidden_size: usize,
    intermediate: usize,
    routes: usize,
) -> Vec<f32> {
    let hidden = hidden
        .iter()
        .map(|&value| round(dtype, value))
        .collect::<Vec<_>>();
    let gate_bias = gate_bias
        .iter()
        .map(|&value| round(dtype, value))
        .collect::<Vec<_>>();
    let down_bias = down_bias
        .iter()
        .map(|&value| round(dtype, value))
        .collect::<Vec<_>>();
    let doubled = 2 * intermediate;
    let mut output = vec![0.0f32; tokens * hidden_size];
    for token in 0..tokens {
        let mut order = (0..experts).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            router[token * experts + right]
                .total_cmp(&router[token * experts + left])
                .then_with(|| left.cmp(&right))
        });
        let selected = &order[..routes];
        let maximum = selected
            .iter()
            .map(|&expert| router[token * experts + expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator = selected
            .iter()
            .map(|&expert| (router[token * experts + expert] - maximum).exp())
            .sum::<f32>();
        for &expert in selected {
            let route_weight = round(
                dtype,
                (router[token * experts + expert] - maximum).exp() / denominator,
            );
            let mut projected = vec![0.0f32; doubled];
            for output_index in 0..doubled {
                let mut value = gate_bias[expert * doubled + output_index];
                for input_index in 0..hidden_size {
                    let row = expert * doubled + output_index;
                    value += hidden[token * hidden_size + input_index]
                        * gate[row * hidden_size + input_index];
                }
                projected[output_index] = value;
            }
            let activated = (0..intermediate)
                .map(|index| {
                    let gate = projected[2 * index].min(7.0);
                    let up = projected[2 * index + 1].clamp(-7.0, 7.0);
                    let swish = gate / (1.0 + (-1.702 * gate).exp());
                    (up + 1.0) * swish
                })
                .collect::<Vec<_>>();
            for output_index in 0..hidden_size {
                let mut value = down_bias[expert * hidden_size + output_index];
                for input_index in 0..intermediate {
                    let row = expert * hidden_size + output_index;
                    value += activated[input_index] * down[row * intermediate + input_index];
                }
                output[token * hidden_size + output_index] += route_weight * value;
            }
        }
    }
    output
}

fn round(dtype: DType, value: f32) -> f32 {
    match dtype {
        DType::F16 => F16::from_f32(value).to_f32(),
        DType::Bf16 => BFloat16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

fn assert_close(dtype: DType, bytes: &[u8], expected: &[f32]) {
    let actual = bytes
        .chunks_exact(2)
        .map(|bytes| {
            let bits = u16::from_ne_bytes(bytes.try_into().unwrap());
            match dtype {
                DType::F16 => F16::from_bits(bits).to_f32(),
                DType::Bf16 => BFloat16::from_bits(bits).to_f32(),
                _ => unreachable!(),
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    let tolerance = match dtype {
        DType::F16 => 1.0e-2,
        DType::Bf16 => 6.0e-2,
        _ => unreachable!(),
    };
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance + tolerance * expected.abs(),
            "output {index}: expected {expected}, received {actual}"
        );
    }
}
