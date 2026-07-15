//! Numerical product contract for model-enabling tensor and neural operations.

use nml_ir::ProgramBuilder;
use nml_types::{BFloat16, Complex64, DType, F16, Shape};

const ROWS: usize = 2;
const WIDTH: usize = 4;

#[test]
fn model_enabling_operations_execute_on_the_product_backends() {
    let platform = platform();
    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        execute_float_contract(&platform, dtype);
    }
    execute_complex_absolute_value(&platform);
}

fn execute_float_contract(platform: &nml::Platform, dtype: DType) {
    let shape = Shape::new(dtype, &[ROWS as i64, WIDTH as i64]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let gate = builder.input("gate", shape);
    let norm_weight = builder.input("norm_weight", Shape::new(dtype, &[WIDTH as i64]).unwrap());
    let norm_bias = builder.input("norm_bias", Shape::new(dtype, &[WIDTH as i64]).unwrap());
    let embedding_weight = builder.input(
        "embedding_weight",
        Shape::new(dtype, &[5, WIDTH as i64]).unwrap(),
    );
    let token_ids = builder.input("token_ids", Shape::new(DType::I32, &[2, 3]).unwrap());
    let scores = builder.input("scores", Shape::new(dtype, &[3, 5]).unwrap());

    let one = scalar(&mut builder, dtype, 1.0);
    let two = scalar(&mut builder, dtype, 2.0);
    let low = scalar(&mut builder, dtype, -1.0);
    let high = scalar(&mut builder, dtype, 2.0);
    let absolute = builder.abs(input).unwrap();
    let positive = builder.add(absolute, one).unwrap();
    let power = builder.power(positive, two).unwrap();
    let remainder = builder.remainder(input, two).unwrap();
    let clamped = builder.clamp(input, low, high).unwrap();
    let floor = builder.floor(input).unwrap();
    let ceil = builder.ceil(input).unwrap();
    let minimum = builder.reduce_min(input, &[1]).unwrap();
    let mean = builder.mean(input, &[1]).unwrap();
    let log_sum_exp = builder.log_sum_exp(input, &[1]).unwrap();
    let normalized = builder.normalize_variance(input, 1, 1e-5).unwrap();
    let layer_norm = builder
        .layer_norm(input, Some(norm_weight), Some(norm_bias), 1, 1e-5)
        .unwrap();
    let l2 = builder.normalize_l2(input, &[1], 1e-6).unwrap();
    let swiglu = builder.swiglu(gate, input).unwrap();
    let geglu = builder.geglu(gate, input).unwrap();
    let embedding = builder
        .token_embedding(embedding_weight, token_ids)
        .unwrap();
    let (maxima, indices) = builder.argmax(scores, 1).unwrap();

    let program = builder
        .finish_named(&[
            ("absolute".to_owned(), absolute),
            ("power".to_owned(), power),
            ("remainder".to_owned(), remainder),
            ("clamped".to_owned(), clamped),
            ("floor".to_owned(), floor),
            ("ceil".to_owned(), ceil),
            ("minimum".to_owned(), minimum),
            ("mean".to_owned(), mean),
            ("log_sum_exp".to_owned(), log_sum_exp),
            ("normalized".to_owned(), normalized),
            ("layer_norm".to_owned(), layer_norm),
            ("l2".to_owned(), l2),
            ("swiglu".to_owned(), swiglu),
            ("geglu".to_owned(), geglu),
            ("embedding".to_owned(), embedding),
            ("maxima".to_owned(), maxima),
            ("indices".to_owned(), indices),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let input = [-2.25, -0.5, 1.25, 3.75, 4.5, 1.5, -1.0, 2.0];
    let gate = [0.5, -1.0, 2.0, -0.25, 1.25, 0.75, -1.5, 2.5];
    let norm_weight = [1.0, 0.5, -1.0, 2.0];
    let norm_bias = [0.125, -0.25, 0.375, 0.0];
    let embedding_weight = (0..20)
        .map(|index| (index as f32 - 8.0) / 4.0)
        .collect::<Vec<_>>();
    let token_ids = [4, 0, 2, 1, 3, 1];
    let scores = [
        5.0,
        4.1,
        7.9,
        0.0,
        7.9,
        5.0,
        f32::NAN,
        7.9,
        0.0,
        f32::NAN,
        -1.0,
        -5.0,
        -0.5,
        -0.5,
        -2.0,
    ];
    let mut arguments = executable.args();
    set_float(platform, &mut arguments, "input", shape, &input);
    set_float(platform, &mut arguments, "gate", shape, &gate);
    set_float(
        platform,
        &mut arguments,
        "norm_weight",
        Shape::new(dtype, &[WIDTH as i64]).unwrap(),
        &norm_weight,
    );
    set_float(
        platform,
        &mut arguments,
        "norm_bias",
        Shape::new(dtype, &[WIDTH as i64]).unwrap(),
        &norm_bias,
    );
    set_float(
        platform,
        &mut arguments,
        "embedding_weight",
        Shape::new(dtype, &[5, WIDTH as i64]).unwrap(),
        &embedding_weight,
    );
    set_i32(
        platform,
        &mut arguments,
        "token_ids",
        Shape::new(DType::I32, &[2, 3]).unwrap(),
        &token_ids,
    );
    set_float(
        platform,
        &mut arguments,
        "scores",
        Shape::new(dtype, &[3, 5]).unwrap(),
        &scores,
    );
    let results = arguments.call().unwrap();

    let input = rounded(dtype, &input);
    let gate = rounded(dtype, &gate);
    let norm_weight = rounded(dtype, &norm_weight);
    let norm_bias = rounded(dtype, &norm_bias);
    assert_close(
        &results,
        "absolute",
        dtype,
        &input.iter().map(|value| value.abs()).collect::<Vec<_>>(),
    );
    assert_close(
        &results,
        "power",
        dtype,
        &input
            .iter()
            .map(|value| (value.abs() + 1.0).powi(2))
            .collect::<Vec<_>>(),
    );
    assert_close(
        &results,
        "remainder",
        dtype,
        &input.iter().map(|value| value % 2.0).collect::<Vec<_>>(),
    );
    assert_close(
        &results,
        "clamped",
        dtype,
        &input
            .iter()
            .map(|value| value.clamp(-1.0, 2.0))
            .collect::<Vec<_>>(),
    );
    assert_close(
        &results,
        "floor",
        dtype,
        &input.iter().map(|value| value.floor()).collect::<Vec<_>>(),
    );
    assert_close(
        &results,
        "ceil",
        dtype,
        &input.iter().map(|value| value.ceil()).collect::<Vec<_>>(),
    );
    let minimum = rowwise(&input, |row| {
        row.iter().copied().fold(f32::INFINITY, f32::min)
    });
    let mean = rowwise(&input, |row| row.iter().sum::<f32>() / row.len() as f32);
    let log_sum_exp = rowwise(&input, |row| {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        maximum
            + row
                .iter()
                .map(|value| (value - maximum).exp())
                .sum::<f32>()
                .ln()
    });
    assert_close(&results, "minimum", dtype, &minimum);
    assert_close(&results, "mean", dtype, &mean);
    assert_close(&results, "log_sum_exp", dtype, &log_sum_exp);

    let normalized = normalized_rows(&input, 1e-5);
    assert_close(&results, "normalized", dtype, &normalized);
    let layer_norm = normalized
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let normalized = round_to_dtype(dtype, *value);
            let scaled = round_to_dtype(dtype, normalized * norm_weight[index % WIDTH]);
            round_to_dtype(dtype, scaled + norm_bias[index % WIDTH])
        })
        .collect::<Vec<_>>();
    assert_close(&results, "layer_norm", dtype, &layer_norm);
    let l2 = input
        .chunks_exact(WIDTH)
        .flat_map(|row| {
            let inverse = (row.iter().map(|value| value * value).sum::<f32>() + 1e-6)
                .sqrt()
                .recip();
            row.iter().map(move |value| value * inverse)
        })
        .collect::<Vec<_>>();
    assert_close(&results, "l2", dtype, &l2);
    let swiglu = gate
        .iter()
        .zip(&input)
        .map(|(gate, value)| gate / (1.0 + (-gate).exp()) * value)
        .collect::<Vec<_>>();
    let geglu = gate
        .iter()
        .zip(&input)
        .map(|(gate, value)| gelu(*gate) * value)
        .collect::<Vec<_>>();
    assert_close(&results, "swiglu", dtype, &swiglu);
    assert_close(&results, "geglu", dtype, &geglu);

    let embedding_weight = rounded(dtype, &embedding_weight);
    let expected_embedding = token_ids
        .iter()
        .flat_map(|token| {
            let start = *token as usize * WIDTH;
            embedding_weight[start..start + WIDTH].iter().copied()
        })
        .collect::<Vec<_>>();
    assert_close(&results, "embedding", dtype, &expected_embedding);
    let maxima = decode_float(&results, "maxima", dtype);
    assert_close_values(dtype, &maxima[..1], &[round_to_dtype(dtype, 7.9)], "maxima");
    assert!(maxima[1].is_nan(), "argmax must propagate the first NaN");
    assert_close_values(
        dtype,
        &maxima[2..],
        &[round_to_dtype(dtype, -0.5)],
        "maxima",
    );
    assert_eq!(decode_i32(&results, "indices"), [2, 1, 2]);
}

fn execute_complex_absolute_value(platform: &nml::Platform) {
    let shape = Shape::new(DType::C64, &[3]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("complex", shape);
    let output = builder.abs(input).unwrap();
    assert_eq!(output.shape().dtype(), DType::F32);
    let program = builder
        .finish_named(&[("magnitude".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let values = [
        Complex64::new(3.0, 4.0),
        Complex64::new(-5.0, 12.0),
        Complex64::new(0.0, -2.5),
    ];
    let host = nml::Slice::from_typed(shape, &values).unwrap();
    let buffer = platform
        .upload(&host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let mut arguments = executable.args();
    arguments.set("complex", buffer).unwrap();
    let results = arguments.call().unwrap();
    assert_close(&results, "magnitude", DType::F32, &[5.0, 13.0, 2.5]);
}

fn normalized_rows(input: &[f32], epsilon: f32) -> Vec<f32> {
    input
        .chunks_exact(WIDTH)
        .flat_map(|row| {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let inverse = (variance + epsilon).sqrt().recip();
            row.iter().map(move |value| (value - mean) * inverse)
        })
        .collect()
}

fn rowwise(input: &[f32], operation: impl Fn(&[f32]) -> f32) -> Vec<f32> {
    input.chunks_exact(WIDTH).map(operation).collect()
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

fn scalar(builder: &mut ProgramBuilder, dtype: DType, value: f32) -> nml::Tensor {
    match dtype {
        DType::F32 => builder.scalar(value).unwrap(),
        DType::F16 => builder.scalar(F16::from_f32(value)).unwrap(),
        DType::Bf16 => builder.scalar(BFloat16::from_f32(value)).unwrap(),
        _ => unreachable!(),
    }
}

fn set_float(
    platform: &nml::Platform,
    arguments: &mut nml::exe::Arguments<'_>,
    name: &str,
    shape: Shape,
    values: &[f32],
) {
    let bytes = encode(shape.dtype(), values);
    let host = nml::Slice::from_bytes(shape, &bytes).unwrap();
    let buffer = platform
        .upload(&host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    arguments.set(name, buffer).unwrap();
}

fn set_i32(
    platform: &nml::Platform,
    arguments: &mut nml::exe::Arguments<'_>,
    name: &str,
    shape: Shape,
    values: &[i32],
) {
    let host = nml::Slice::from_typed(shape, values).unwrap();
    let buffer = platform
        .upload(&host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    arguments.set(name, buffer).unwrap();
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

fn rounded(dtype: DType, values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| round_to_dtype(dtype, *value))
        .collect()
}

fn round_to_dtype(dtype: DType, value: f32) -> f32 {
    match dtype {
        DType::F32 => value,
        DType::F16 => F16::from_f32(value).to_f32(),
        DType::Bf16 => BFloat16::from_f32(value).to_f32(),
        _ => unreachable!(),
    }
}

fn assert_close(results: &nml::exe::Results, name: &str, dtype: DType, expected: &[f32]) {
    let actual = decode_float(results, name, dtype);
    assert_close_values(dtype, &actual, expected, name);
}

fn assert_close_values(dtype: DType, actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    let tolerance = match dtype {
        DType::F32 => 3e-5,
        DType::F16 => 2e-2,
        DType::Bf16 => 7e-2,
        _ => unreachable!(),
    };
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance + tolerance * expected.abs(),
            "{name}[{index}]: expected {expected}, received {actual}"
        );
    }
}

fn decode_float(results: &nml::exe::Results, name: &str, dtype: DType) -> Vec<f32> {
    let slice = results.get(name).unwrap().to_slice().unwrap();
    let bytes = slice.contiguous_bytes().unwrap();
    match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|bytes| F16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|bytes| {
                BFloat16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32()
            })
            .collect(),
        _ => unreachable!(),
    }
}

fn decode_i32(results: &nml::exe::Results, name: &str) -> Vec<i32> {
    results
        .get(name)
        .unwrap()
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn platform() -> nml::Platform {
    match env!("NML_NEURAL_OPS_BACKEND") {
        "cpu" => nml::Platform::cpu_with_devices(1).unwrap(),
        "cuda" => {
            let runfiles = std::env::var("RUNFILES_DIR").unwrap();
            let relative = std::env::var("NML_CUDA_RUNTIME_RLOCATION").unwrap();
            // SAFETY: Bazel test processes have not created application
            // threads before platform initialization.
            unsafe {
                std::env::set_var(
                    "NML_CUDA_RUNTIME",
                    std::path::Path::new(&runfiles).join(relative),
                );
                nml::Platform::cuda().unwrap()
            }
        }
        backend => panic!("unknown neural-operations backend {backend}"),
    }
}
