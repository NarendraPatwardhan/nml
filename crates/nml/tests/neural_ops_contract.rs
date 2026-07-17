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
        execute_spatial_contract(&platform, dtype);
        execute_moe_contract(&platform, dtype);
        execute_gated_delta_net_contract(&platform, dtype);
    }
    execute_complex_absolute_value(&platform);
    execute_logical_contract(&platform);
    execute_bit_and_precision_contract(&platform);
    execute_structural_contract(&platform);
    execute_nd_indexing_contract(&platform);
    execute_ordering_random_and_sampling_contract(&platform);
    execute_single_device_collective_contract(&platform);
    if platform.name() == "cpu" {
        execute_expert_parallel_moe_contract(&platform);
    }
}

fn execute_float_contract(platform: &nml::Platform, dtype: DType) {
    let shape = Shape::new(dtype, &[ROWS as i64, WIDTH as i64]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let gate = builder.input("gate", shape);
    let norm_weight = builder.input("norm_weight", Shape::new(dtype, &[WIDTH as i64]).unwrap());
    let norm_bias = builder.input("norm_bias", Shape::new(dtype, &[WIDTH as i64]).unwrap());
    let embedding_parameter = nml::Parameter::dense(
        "embedding_weight",
        "embedding_weight",
        Shape::new(dtype, &[5, WIDTH as i64]).unwrap(),
    )
    .unwrap();
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
        .token_embedding(&embedding_parameter, token_ids)
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
    set_parameter_float(
        platform,
        &mut arguments,
        &embedding_parameter,
        &embedding_weight,
        nml::Sharding::single(),
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
    arguments.bake().unwrap();
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

fn execute_logical_contract(platform: &nml::Platform) {
    let shape = Shape::new(DType::I32, &[4]).unwrap();
    let mut builder = ProgramBuilder::new();
    let left = builder.input("logical_left", shape);
    let right = builder.input("logical_right", shape);
    let and = builder.logical_and(left, right).unwrap();
    let or = builder.logical_or(left, right).unwrap();
    let xor = builder.logical_xor(left, right).unwrap();
    let not = builder.logical_not(left).unwrap();
    let less = builder.less(left, right).unwrap();
    let equal = builder.equal(left, right).unwrap();
    let predicate_or = builder.logical_or(less, equal).unwrap();
    let predicate_not = builder.logical_not(less).unwrap();
    let program = builder
        .finish_named(&[
            ("and".to_owned(), and),
            ("or".to_owned(), or),
            ("xor".to_owned(), xor),
            ("not".to_owned(), not),
            ("predicate_or".to_owned(), predicate_or),
            ("predicate_not".to_owned(), predicate_not),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let left = [0, 0x00f0, -1, 0x5555_5555];
    let right = [0x000f, 0x000f, 0x000f, 0xaaaa_aaaau32 as i32];
    let mut arguments = executable.args();
    set_i32(platform, &mut arguments, "logical_left", shape, &left);
    set_i32(platform, &mut arguments, "logical_right", shape, &right);
    let results = arguments.call().unwrap();

    assert_eq!(decode_i32(&results, "and"), [0, 0, 0x000f, 0]);
    assert_eq!(decode_i32(&results, "or"), [0x000f, 0x00ff, -1, -1]);
    assert_eq!(decode_i32(&results, "xor"), [0x000f, 0x00ff, !0x000f, -1]);
    assert_eq!(decode_i32(&results, "not"), [!0, !0x00f0, 0, !0x5555_5555]);
    assert_eq!(
        decode_bool(&results, "predicate_or"),
        [true, false, true, false]
    );
    assert_eq!(
        decode_bool(&results, "predicate_not"),
        [false, true, false, true]
    );
}

fn execute_bit_and_precision_contract(platform: &nml::Platform) {
    let integer_shape = Shape::new(DType::I32, &[4]).unwrap();
    let float_shape = Shape::new(DType::F32, &[4]).unwrap();
    let mut builder = ProgramBuilder::new();
    let integers = builder.input("integers", integer_shape);
    let amounts = builder.input("amounts", integer_shape);
    let numerical = builder.input("numerical", float_shape);
    let classification = builder.input("classification", float_shape);

    let shifted_left = builder.shift_left(integers, amounts).unwrap();
    let shifted_arithmetic = builder.shift_right_arithmetic(integers, amounts).unwrap();
    let shifted_logical = builder.shift_right_logical(integers, amounts).unwrap();
    let leading_zeros = builder.count_leading_zeros(integers).unwrap();
    let population = builder.population_count(integers).unwrap();
    let bytes = builder.bitcast(integers, DType::U8).unwrap();
    let bitcast_roundtrip = builder.bitcast(bytes, DType::I32).unwrap();

    let finite = builder.is_finite(classification).unwrap();
    let sign = builder.sign(numerical).unwrap();
    let expm1 = builder.expm1(numerical).unwrap();
    let round_away = builder.round_nearest_away_from_zero(numerical).unwrap();
    let round_even = builder.round_nearest_even(numerical).unwrap();
    let reduced = builder.reduce_precision(numerical, 5, 10).unwrap();

    let program = builder
        .finish_named(&[
            ("shifted_left".to_owned(), shifted_left),
            ("shifted_arithmetic".to_owned(), shifted_arithmetic),
            ("shifted_logical".to_owned(), shifted_logical),
            ("leading_zeros".to_owned(), leading_zeros),
            ("population".to_owned(), population),
            ("bitcast_roundtrip".to_owned(), bitcast_roundtrip),
            ("finite".to_owned(), finite),
            ("sign".to_owned(), sign),
            ("expm1".to_owned(), expm1),
            ("round_away".to_owned(), round_away),
            ("round_even".to_owned(), round_even),
            ("reduced".to_owned(), reduced),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let integers = [0i32, 1, -16, i32::MIN];
    let amounts = [0i32, 1, 4, 31];
    let numerical = [-2.5f32, -1.5, 1.5, 2.5];
    let classification = [0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let mut arguments = executable.args();
    set_i32(
        platform,
        &mut arguments,
        "integers",
        integer_shape,
        &integers,
    );
    set_i32(platform, &mut arguments, "amounts", integer_shape, &amounts);
    set_float(
        platform,
        &mut arguments,
        "numerical",
        float_shape,
        &numerical,
    );
    set_float(
        platform,
        &mut arguments,
        "classification",
        float_shape,
        &classification,
    );
    let results = arguments.call().unwrap();

    assert_eq!(
        decode_i32(&results, "shifted_left"),
        integers
            .iter()
            .zip(amounts)
            .map(|(value, amount)| value.wrapping_shl(amount as u32))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        decode_i32(&results, "shifted_arithmetic"),
        integers
            .iter()
            .zip(amounts)
            .map(|(value, amount)| value.wrapping_shr(amount as u32))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        decode_i32(&results, "shifted_logical"),
        integers
            .iter()
            .zip(amounts)
            .map(|(value, amount)| ((*value as u32) >> amount) as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        decode_i32(&results, "leading_zeros"),
        integers
            .iter()
            .map(|value| value.leading_zeros() as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        decode_i32(&results, "population"),
        integers
            .iter()
            .map(|value| value.count_ones() as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        decode_i32(&results, "bitcast_roundtrip"),
        integers,
        "bitcast must preserve every element bit across width changes"
    );
    assert_eq!(decode_bool(&results, "finite"), [true, false, false, false]);
    assert_close(&results, "sign", DType::F32, &[-1.0, -1.0, 1.0, 1.0]);
    assert_close(&results, "expm1", DType::F32, &numerical.map(f32::exp_m1));
    assert_close(&results, "round_away", DType::F32, &[-3.0, -2.0, 2.0, 3.0]);
    assert_close(&results, "round_even", DType::F32, &[-2.0, -2.0, 2.0, 2.0]);
    let reduced = numerical.map(|value| F16::from_f32(value).to_f32());
    assert_close(&results, "reduced", DType::F32, &reduced);
}

fn execute_structural_contract(platform: &nml::Platform) {
    let input_shape = Shape::new(DType::F32, &[2, 3]).unwrap();
    let left_shape = Shape::new(DType::F32, &[2]).unwrap();
    let right_shape = Shape::new(DType::F32, &[3]).unwrap();
    let matrix_shape = Shape::new(DType::F32, &[2, 2]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("structural_input", input_shape);
    let left = builder.input("structural_left", left_shape);
    let right = builder.input("structural_right", right_shape);
    let positive_definite = builder.input("positive_definite", matrix_shape);
    let triangular_coefficient = builder.input("triangular_coefficient", matrix_shape);
    let triangular_rhs = builder.input("triangular_rhs", matrix_shape);
    let zero = builder.scalar(0.0f32).unwrap();

    let padded = builder.pad(input, zero, &[1, 0], &[0, 1], &[0, 1]).unwrap();
    let reversed = builder.reverse(input, &[0, 1]).unwrap();
    let inserted = builder
        .insert_axis(input, 1, nml::AxisTag::new(201))
        .unwrap();
    let squeezed = builder.squeeze(inserted, 1).unwrap();
    let stacked = builder
        .stack(&[input, input], 1, nml::AxisTag::new(202))
        .unwrap();
    let repeated = builder.repeat(input, 1, 2).unwrap();
    let stuttered = builder.stutter(input, 1, 2).unwrap();
    let split = builder.split(input, 1, &[1, 2]).unwrap();
    let chunks = builder.chunks(input, 1, 3).unwrap();
    let outer = builder.outer(left, right).unwrap();
    let diagonal = builder
        .diagonal(right, 0, nml::AxisTag::new(203), nml::AxisTag::new(204))
        .unwrap();
    let triangular = builder.triangular(input, 0, 1, 0).unwrap();
    let cartesian = builder
        .cartesian_product_stacked(&[left, right], nml::AxisTag::new(205))
        .unwrap();
    let rolled = builder.roll(input, 1, -1).unwrap();
    let barrier = builder.optimization_barrier(input).unwrap();
    let factor = builder.cholesky(positive_definite, true).unwrap();
    let solution = builder
        .triangular_solve(triangular_coefficient, triangular_rhs, true)
        .unwrap();

    let program = builder
        .finish_named(&[
            ("padded".to_owned(), padded),
            ("reversed".to_owned(), reversed),
            ("squeezed".to_owned(), squeezed),
            ("stacked".to_owned(), stacked),
            ("repeated".to_owned(), repeated),
            ("stuttered".to_owned(), stuttered),
            ("split_left".to_owned(), split[0]),
            ("split_right".to_owned(), split[1]),
            ("middle_chunk".to_owned(), chunks[1]),
            ("outer".to_owned(), outer),
            ("diagonal".to_owned(), diagonal),
            ("triangular".to_owned(), triangular),
            ("cartesian".to_owned(), cartesian),
            ("rolled".to_owned(), rolled),
            ("barrier".to_owned(), barrier),
            ("factor".to_owned(), factor),
            ("solution".to_owned(), solution),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left = [2.0f32, 3.0];
    let right = [4.0f32, 5.0, 6.0];
    let positive_definite = [4.0f32, 2.0, 2.0, 3.0];
    let triangular_coefficient = [2.0f32, 0.0, 1.0, 3.0];
    let triangular_rhs = [4.0f32, 2.0, 7.0, 5.0];
    let mut arguments = executable.args();
    set_float(
        platform,
        &mut arguments,
        "structural_input",
        input_shape,
        &input,
    );
    set_float(
        platform,
        &mut arguments,
        "structural_left",
        left_shape,
        &left,
    );
    set_float(
        platform,
        &mut arguments,
        "structural_right",
        right_shape,
        &right,
    );
    set_float(
        platform,
        &mut arguments,
        "positive_definite",
        matrix_shape,
        &positive_definite,
    );
    set_float(
        platform,
        &mut arguments,
        "triangular_coefficient",
        matrix_shape,
        &triangular_coefficient,
    );
    set_float(
        platform,
        &mut arguments,
        "triangular_rhs",
        matrix_shape,
        &triangular_rhs,
    );
    let results = arguments.call().unwrap();

    assert_close(
        &results,
        "padded",
        DType::F32,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0,
            0.0,
        ],
    );
    assert_close(
        &results,
        "reversed",
        DType::F32,
        &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    );
    assert_close(&results, "squeezed", DType::F32, &input);
    assert_close(
        &results,
        "stacked",
        DType::F32,
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
    );
    assert_close(
        &results,
        "repeated",
        DType::F32,
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
    );
    assert_close(
        &results,
        "stuttered",
        DType::F32,
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
    );
    assert_close(&results, "split_left", DType::F32, &[1.0, 4.0]);
    assert_close(&results, "split_right", DType::F32, &[2.0, 3.0, 5.0, 6.0]);
    assert_close(&results, "middle_chunk", DType::F32, &[2.0, 5.0]);
    assert_close(
        &results,
        "outer",
        DType::F32,
        &[8.0, 10.0, 12.0, 12.0, 15.0, 18.0],
    );
    assert_close(
        &results,
        "diagonal",
        DType::F32,
        &[4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0],
    );
    assert_close(
        &results,
        "triangular",
        DType::F32,
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0],
    );
    assert_close(
        &results,
        "cartesian",
        DType::F32,
        &[2.0, 4.0, 2.0, 5.0, 2.0, 6.0, 3.0, 4.0, 3.0, 5.0, 3.0, 6.0],
    );
    assert_close(
        &results,
        "rolled",
        DType::F32,
        &[2.0, 3.0, 1.0, 5.0, 6.0, 4.0],
    );
    assert_close(&results, "barrier", DType::F32, &input);
    let factor = decode_float(&results, "factor", DType::F32);
    assert_close_values(
        DType::F32,
        &[factor[0], factor[2], factor[3]],
        &[2.0, 1.0, 2.0f32.sqrt()],
        "factor lower triangle",
    );
    assert_close(
        &results,
        "solution",
        DType::F32,
        &[2.0, 1.0, 5.0 / 3.0, 4.0 / 3.0],
    );
}

fn execute_nd_indexing_contract(platform: &nml::Platform) {
    let input_shape = Shape::new(DType::F32, &[2, 3, 4]).unwrap();
    let gather_indices_shape = Shape::new(DType::I32, &[6, 2]).unwrap();
    let scatter_indices_shape = Shape::new(DType::I32, &[4, 2]).unwrap();
    let scatter_updates_shape = Shape::new(DType::F32, &[4, 3]).unwrap();
    let unique_indices_shape = Shape::new(DType::I32, &[2, 2]).unwrap();
    let unique_updates_shape = Shape::new(DType::F32, &[2, 3]).unwrap();
    let batched_indices_shape = Shape::new(DType::I32, &[2, 2, 1]).unwrap();
    let batched_updates_shape = Shape::new(DType::F32, &[2, 2, 4]).unwrap();
    let empty_indices_shape = Shape::new(DType::I32, &[0, 2]).unwrap();
    let empty_updates_shape = Shape::new(DType::F32, &[0, 3]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("nd_input", input_shape);
    let scatter_base = builder.input("scatter_base", input_shape);
    let gather_indices = builder.input("gather_indices", gather_indices_shape);
    let scatter_indices = builder.input("scatter_indices", scatter_indices_shape);
    let scatter_updates = builder.input("scatter_updates", scatter_updates_shape);
    let unique_indices = builder.input("unique_indices", unique_indices_shape);
    let unique_updates = builder.input("unique_updates", unique_updates_shape);
    let batched_indices = builder.input("batched_indices", batched_indices_shape);
    let batched_updates = builder.input("batched_updates", batched_updates_shape);
    let empty_indices = builder.input("empty_indices", empty_indices_shape);
    let empty_updates = builder.input("empty_updates", empty_updates_shape);
    let gathered = builder.gather_nd(input, gather_indices, &[0, 2]).unwrap();
    let batched = builder
        .gather_batched_nd(input, batched_indices, 1, &[1])
        .unwrap();
    let added = builder
        .scatter_add(scatter_base, scatter_indices, scatter_updates, &[0, 2])
        .unwrap();
    let updated = builder
        .scatter_update(scatter_base, unique_indices, unique_updates, &[0, 2])
        .unwrap();
    let batched_added = builder
        .scatter_add_batched(scatter_base, batched_indices, batched_updates, 1, &[1])
        .unwrap();
    let empty_added = builder
        .scatter_add(scatter_base, empty_indices, empty_updates, &[0, 2])
        .unwrap();
    let program = builder
        .finish_named(&[
            ("nd_gather".to_owned(), gathered),
            ("batched_gather".to_owned(), batched),
            ("scatter_add".to_owned(), added),
            ("scatter_update".to_owned(), updated),
            ("batched_scatter_add".to_owned(), batched_added),
            ("empty_scatter_add".to_owned(), empty_added),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let input = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let scatter_base = [0.0f32; 24];
    let gather_indices = [0, 0, 1, 3, 0, 2, 1, 1, 5, 0, -1, 2];
    let scatter_indices = [0, 0, 1, 3, 0, 0, 5, 0];
    let scatter_updates = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0,
    ];
    let unique_indices = [0, 2, 1, 1];
    let unique_updates = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let batched_indices = [2, 0, 1, 2];
    let batched_updates = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    // A compiled indexing graph is reusable. Rebinding fresh buffers on the
    // second call also proves that no update state leaked through PJRT.
    for _ in 0..2 {
        let mut arguments = executable.args();
        set_float(platform, &mut arguments, "nd_input", input_shape, &input);
        set_float(
            platform,
            &mut arguments,
            "scatter_base",
            input_shape,
            &scatter_base,
        );
        set_i32(
            platform,
            &mut arguments,
            "gather_indices",
            gather_indices_shape,
            &gather_indices,
        );
        set_i32(
            platform,
            &mut arguments,
            "scatter_indices",
            scatter_indices_shape,
            &scatter_indices,
        );
        set_float(
            platform,
            &mut arguments,
            "scatter_updates",
            scatter_updates_shape,
            &scatter_updates,
        );
        set_i32(
            platform,
            &mut arguments,
            "unique_indices",
            unique_indices_shape,
            &unique_indices,
        );
        set_float(
            platform,
            &mut arguments,
            "unique_updates",
            unique_updates_shape,
            &unique_updates,
        );
        set_i32(
            platform,
            &mut arguments,
            "batched_indices",
            batched_indices_shape,
            &batched_indices,
        );
        set_float(
            platform,
            &mut arguments,
            "batched_updates",
            batched_updates_shape,
            &batched_updates,
        );
        set_i32(
            platform,
            &mut arguments,
            "empty_indices",
            empty_indices_shape,
            &[],
        );
        set_float(
            platform,
            &mut arguments,
            "empty_updates",
            empty_updates_shape,
            &[],
        );
        let results = arguments.call().unwrap();

        assert_close(
            &results,
            "nd_gather",
            DType::F32,
            &[
                0.0, 4.0, 8.0, 15.0, 19.0, 23.0, 2.0, 6.0, 10.0, 13.0, 17.0, 21.0, 12.0, 16.0,
                20.0, 2.0, 6.0, 10.0,
            ],
        );
        assert_close(
            &results,
            "batched_gather",
            DType::F32,
            &[
                8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                23.0,
            ],
        );
        let mut expected_add = [0.0f32; 24];
        expected_add[0] = 11.0;
        expected_add[4] = 22.0;
        expected_add[8] = 33.0;
        expected_add[15] = 4.0;
        expected_add[19] = 5.0;
        expected_add[23] = 6.0;
        assert_close(&results, "scatter_add", DType::F32, &expected_add);
        let mut expected_update = [0.0f32; 24];
        expected_update[2] = 7.0;
        expected_update[6] = 8.0;
        expected_update[10] = 9.0;
        expected_update[13] = 10.0;
        expected_update[17] = 11.0;
        expected_update[21] = 12.0;
        assert_close(&results, "scatter_update", DType::F32, &expected_update);
        assert_close(
            &results,
            "batched_scatter_add",
            DType::F32,
            &[
                5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        );
        assert_close(&results, "empty_scatter_add", DType::F32, &scatter_base);
    }

    execute_scatter_donation_contract(platform);
}

fn execute_scatter_donation_contract(platform: &nml::Platform) {
    let base_shape = Shape::new(DType::F32, &[3, 2]).unwrap();
    let indices_shape = Shape::new(DType::I32, &[2, 1]).unwrap();
    let updates_shape = Shape::new(DType::F32, &[2, 2]).unwrap();
    let mut builder = ProgramBuilder::new();
    let base = builder.input("base", base_shape);
    let indices = builder.input("indices", indices_shape);
    let updates = builder.input("updates", updates_shape);
    let output = builder
        .scatter_update(base, indices, updates, &[0])
        .unwrap();
    let output = builder.reuse_buffer(output, base).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    for _ in 0..2 {
        let mut arguments = executable.args();
        set_float(platform, &mut arguments, "base", base_shape, &[0.0; 6]);
        set_i32(platform, &mut arguments, "indices", indices_shape, &[2, 0]);
        set_float(
            platform,
            &mut arguments,
            "updates",
            updates_shape,
            &[1.0, 2.0, 3.0, 4.0],
        );
        let results = arguments.call().unwrap();
        assert_close(
            &results,
            "output",
            DType::F32,
            &[3.0, 4.0, 0.0, 0.0, 1.0, 2.0],
        );
    }
}

fn execute_spatial_contract(platform: &nml::Platform, dtype: DType) {
    let sequence_shape = Shape::new(dtype, &[1, 1, 5]).unwrap();
    let conv1d_kernel_shape = Shape::new(dtype, &[1, 1, 3]).unwrap();
    let image_shape = Shape::new(dtype, &[1, 2, 3, 3]).unwrap();
    let grouped_kernel_shape = Shape::new(dtype, &[2, 1, 2, 2]).unwrap();
    let resize_shape = Shape::new(dtype, &[1, 1, 2, 2]).unwrap();
    let mut builder = ProgramBuilder::new();
    let sequence = builder.input("sequence", sequence_shape);
    let conv1d_kernel = builder.input("conv1d_kernel", conv1d_kernel_shape);
    let image = builder.input("image", image_shape);
    let grouped_kernel = builder.input("grouped_kernel", grouped_kernel_shape);
    let resize_input = builder.input("resize_input", resize_shape);

    let unit3 = [1, 1, 1];
    let cumulative = builder.cumulative_sum(sequence, 2).unwrap();
    let window_sum = builder
        .reduce_window_sum(
            sequence,
            &[1, 1, 3],
            &[1, 1, 2],
            &unit3,
            &unit3,
            &[[0, 0], [0, 0], [1, 1]],
        )
        .unwrap();
    let base_dilated = builder
        .reduce_window_sum(
            sequence,
            &[1, 1, 2],
            &unit3,
            &[1, 1, 2],
            &unit3,
            &[[0, 0]; 3],
        )
        .unwrap();
    let window_dilated = builder
        .reduce_window_sum(
            sequence,
            &[1, 1, 2],
            &unit3,
            &unit3,
            &[1, 1, 2],
            &[[0, 0]; 3],
        )
        .unwrap();
    let convolution_1d = builder
        .conv1d(sequence, conv1d_kernel, 1, [1, 1], 1, 1, 1)
        .unwrap();
    let convolution_2d = builder
        .conv2d(
            image,
            grouped_kernel,
            [1, 1],
            [[0, 0], [0, 0]],
            [1, 1],
            [1, 1],
            2,
        )
        .unwrap();
    let pooled = builder
        .max_pool2d(image, [2, 3], [2, 2], [2, 2], [[0, 0], [0, 0]])
        .unwrap();
    let nearest = builder.resize_nearest(resize_input, 3, 4).unwrap();
    let linear = builder.resize_linear(resize_input, 3, 4).unwrap();
    let bilinear = builder
        .resize_bilinear(resize_input, [2, 3], [3, 3])
        .unwrap();
    let cubic = builder.resize_cubic(resize_input, 3, 4).unwrap();
    let upsampled = builder.upsample_nearest(resize_input, &[2.0]).unwrap();

    let program = builder
        .finish_named(&[
            ("cumulative".to_owned(), cumulative),
            ("window_sum".to_owned(), window_sum),
            ("base_dilated".to_owned(), base_dilated),
            ("window_dilated".to_owned(), window_dilated),
            ("convolution_1d".to_owned(), convolution_1d),
            ("convolution_2d".to_owned(), convolution_2d),
            ("pooled".to_owned(), pooled),
            ("nearest".to_owned(), nearest),
            ("linear".to_owned(), linear),
            ("bilinear".to_owned(), bilinear),
            ("cubic".to_owned(), cubic),
            ("upsampled".to_owned(), upsampled),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let sequence = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let conv1d_kernel = [1.0f32, 0.0, -1.0];
    let image = (1..=18).map(|value| value as f32).collect::<Vec<_>>();
    let grouped_kernel = [1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let resize_input = [1.0f32, 2.0, 3.0, 4.0];
    let mut arguments = executable.args();
    set_float(
        platform,
        &mut arguments,
        "sequence",
        sequence_shape,
        &sequence,
    );
    set_float(
        platform,
        &mut arguments,
        "conv1d_kernel",
        conv1d_kernel_shape,
        &conv1d_kernel,
    );
    set_float(platform, &mut arguments, "image", image_shape, &image);
    set_float(
        platform,
        &mut arguments,
        "grouped_kernel",
        grouped_kernel_shape,
        &grouped_kernel,
    );
    set_float(
        platform,
        &mut arguments,
        "resize_input",
        resize_shape,
        &resize_input,
    );
    let results = arguments.call().unwrap();

    assert_close(&results, "cumulative", dtype, &[1.0, 3.0, 6.0, 10.0, 15.0]);
    assert_close(&results, "window_sum", dtype, &[3.0, 9.0, 9.0]);
    assert_close(
        &results,
        "base_dilated",
        dtype,
        &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0],
    );
    assert_close(&results, "window_dilated", dtype, &[4.0, 6.0, 8.0]);
    assert_close(
        &results,
        "convolution_1d",
        dtype,
        &[-2.0, -2.0, -2.0, -2.0, 4.0],
    );
    assert_close(
        &results,
        "convolution_2d",
        dtype,
        &[12.0, 16.0, 24.0, 28.0, 24.0, 26.0, 30.0, 32.0],
    );
    assert_close(&results, "pooled", dtype, &[5.0, 14.0]);
    assert_close(
        &results,
        "nearest",
        dtype,
        &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
    );
    assert_close(
        &results,
        "linear",
        dtype,
        &[1.0, 1.5, 2.0, 2.0, 3.0, 3.5, 4.0, 4.0],
    );
    assert_close(
        &results,
        "bilinear",
        dtype,
        &[
            1.0,
            5.0 / 3.0,
            2.0,
            7.0 / 3.0,
            3.0,
            10.0 / 3.0,
            3.0,
            11.0 / 3.0,
            4.0,
        ],
    );
    assert_close(
        &results,
        "cubic",
        dtype,
        &[1.0, 1.5, 2.0, 2.0625, 3.0, 3.5, 4.0, 4.0625],
    );
    assert_close(
        &results,
        "upsampled",
        dtype,
        &[
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ],
    );
}

fn execute_ordering_random_and_sampling_contract(platform: &nml::Platform) {
    const SAMPLE_ROWS: usize = 512;
    const VOCABULARY: usize = 6;

    let ordering_shape = Shape::new(DType::F32, &[2, 5]).unwrap();
    let random_shape = Shape::new(DType::F32, &[256]).unwrap();
    let logits_shape = Shape::new(DType::F32, &[SAMPLE_ROWS as i64, VOCABULARY as i64]).unwrap();
    let state_shape = Shape::new(DType::U64, &[2]).unwrap();
    let scalar_f32 = Shape::new(DType::F32, &[]).unwrap();
    let scalar_i32 = Shape::new(DType::I32, &[]).unwrap();

    let mut builder = ProgramBuilder::new();
    let ordering = builder.input("ordering", ordering_shape);
    let logits = builder.input("logits", logits_shape);
    let state_input = builder.input("state", state_shape);
    let top_k = builder.input("top_k", scalar_i32);
    let temperature = builder.input("temperature", scalar_f32);
    let top_p = builder.input("top_p", scalar_f32);
    let min_p = builder.input("min_p", scalar_f32);

    let (ascending, ascending_indices) = builder.sort(ordering, 1, false, true).unwrap();
    let (descending, descending_indices) = builder.sort(ordering, 1, true, false).unwrap();
    let (top_values, top_indices) = builder.top_k(ordering, 1, 3, true).unwrap();
    let greedy = builder.greedy_tokens(logits, 1).unwrap();
    let state = builder.random_state(state_input).unwrap();
    let (state, uniform) = builder
        .random_uniform(state, random_shape, -2.0, 3.0)
        .unwrap();
    let (state, normal) = builder
        .random_normal(state, random_shape, 1.0, 2.0)
        .unwrap();
    let (state, gumbel) = builder.random_gumbel(state, random_shape).unwrap();
    let (state, sampled) = builder
        .sample_tokens(logits, state, 1, 3, 0.8, 0.95, 0.0)
        .unwrap();
    let (state, dynamic_sampled) = builder
        .sample_tokens_dynamic(logits, state, 1, top_k, temperature, top_p, min_p, 4)
        .unwrap();
    let state = builder
        .reuse_buffer(state.into_tensor(), state_input)
        .unwrap();
    let program = builder
        .finish_named(&[
            ("ascending".to_owned(), ascending),
            ("ascending_indices".to_owned(), ascending_indices),
            ("descending".to_owned(), descending),
            ("descending_indices".to_owned(), descending_indices),
            ("top_values".to_owned(), top_values),
            ("top_indices".to_owned(), top_indices),
            ("greedy".to_owned(), greedy),
            ("uniform".to_owned(), uniform),
            ("normal".to_owned(), normal),
            ("gumbel".to_owned(), gumbel),
            ("sampled".to_owned(), sampled),
            ("dynamic_sampled".to_owned(), dynamic_sampled),
            ("state".to_owned(), state),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let ordering = [
        2.0,
        f32::NAN,
        1.0,
        2.0,
        -1.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let logits = (0..SAMPLE_ROWS)
        .flat_map(|_| [1.2f32, 0.6, 0.0, -0.6, -1.2, -1.8])
        .collect::<Vec<_>>();
    let seed = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
    let execute = || {
        let mut arguments = executable.args();
        set_float(
            platform,
            &mut arguments,
            "ordering",
            ordering_shape,
            &ordering,
        );
        set_float(platform, &mut arguments, "logits", logits_shape, &logits);
        set_u64(platform, &mut arguments, "state", state_shape, &seed);
        set_i32(platform, &mut arguments, "top_k", scalar_i32, &[2]);
        set_float(platform, &mut arguments, "temperature", scalar_f32, &[0.8]);
        set_float(platform, &mut arguments, "top_p", scalar_f32, &[0.95]);
        set_float(platform, &mut arguments, "min_p", scalar_f32, &[0.0]);
        arguments.call().unwrap()
    };
    let first = execute();
    let second = execute();

    assert_float_sequence(
        &decode_float(&first, "ascending", DType::F32),
        &[
            -1.0,
            1.0,
            2.0,
            2.0,
            f32::NAN,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NAN,
        ],
        "ascending",
    );
    assert_eq!(
        decode_i32(&first, "ascending_indices"),
        [4, 2, 0, 3, 1, 3, 0, 1, 2, 4]
    );
    assert_float_sequence(
        &decode_float(&first, "descending", DType::F32),
        &[
            f32::NAN,
            2.0,
            2.0,
            1.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            0.0,
            -0.0,
            f32::NEG_INFINITY,
        ],
        "descending",
    );
    assert_eq!(
        decode_i32(&first, "descending_indices"),
        [1, 0, 3, 2, 4, 4, 2, 0, 1, 3]
    );
    assert_float_sequence(
        &decode_float(&first, "top_values", DType::F32),
        &[f32::NAN, 2.0, 2.0, f32::NAN, f32::INFINITY, 0.0],
        "top_values",
    );
    assert_eq!(decode_i32(&first, "top_indices"), [1, 0, 3, 4, 2, 0]);
    assert!(decode_i32(&first, "greedy").iter().all(|token| *token == 0));

    let uniform = decode_float(&first, "uniform", DType::F32);
    let normal = decode_float(&first, "normal", DType::F32);
    let gumbel = decode_float(&first, "gumbel", DType::F32);
    assert!(uniform.iter().all(|value| (-2.0..3.0).contains(value)));
    assert!(normal.iter().all(|value| value.is_finite()));
    assert!(gumbel.iter().all(|value| value.is_finite()));
    assert!((mean(&uniform) - 0.5).abs() < 0.4, "uniform mean");
    assert!((mean(&normal) - 1.0).abs() < 0.5, "normal mean");
    assert_ne!(decode_u64(&first, "state"), seed);

    for name in ["uniform", "normal", "gumbel"] {
        assert_eq!(
            result_bytes(&first, name),
            result_bytes(&second, name),
            "{name} must replay exactly from the same state"
        );
    }
    assert_eq!(decode_u64(&first, "state"), decode_u64(&second, "state"));
    for name in ["sampled", "dynamic_sampled"] {
        assert_eq!(
            decode_i32(&first, name),
            decode_i32(&second, name),
            "{name}"
        );
    }
    let sampled = decode_i32(&first, "sampled");
    let dynamic_sampled = decode_i32(&first, "dynamic_sampled");
    assert!(sampled.iter().all(|token| (0..=2).contains(token)));
    assert!(dynamic_sampled.iter().all(|token| (0..=1).contains(token)));
    let first_count = sampled.iter().filter(|token| **token == 0).count();
    let third_count = sampled.iter().filter(|token| **token == 2).count();
    assert!(
        first_count > third_count,
        "sampling must respect logit ordering"
    );
}

fn execute_single_device_collective_contract(platform: &nml::Platform) {
    let shape = Shape::new(DType::F32, &[4]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let output = builder.all_reduce_sum(input).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let mut arguments = executable.args();
    set_float(
        platform,
        &mut arguments,
        "input",
        shape,
        &[1.0, -2.0, 3.5, 0.25],
    );
    let results = arguments.call().unwrap();
    assert_close(&results, "output", DType::F32, &[1.0, -2.0, 3.5, 0.25]);
}

fn execute_moe_contract(platform: &nml::Platform, dtype: DType) {
    const TOKENS: usize = 4;
    const HIDDEN: usize = 3;
    const EXPERTS: usize = 3;
    const INTERMEDIATE: usize = 2;

    let hidden_shape = Shape::new(dtype, &[TOKENS as i64, HIDDEN as i64]).unwrap();
    let router_shape = Shape::new(DType::F32, &[TOKENS as i64, EXPERTS as i64]).unwrap();
    let gate_up_shape = Shape::new(
        dtype,
        &[EXPERTS as i64, (2 * INTERMEDIATE) as i64, HIDDEN as i64],
    )
    .unwrap();
    let down_shape =
        Shape::new(dtype, &[EXPERTS as i64, HIDDEN as i64, INTERMEDIATE as i64]).unwrap();
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input("hidden", hidden_shape);
    let router = builder.input("router", router_shape);
    let gate_up_parameter = nml::Parameter::dense("gate_up", "gate_up", gate_up_shape).unwrap();
    let down_parameter = nml::Parameter::dense("down", "down", down_shape).unwrap();
    let swiglu = builder
        .moe_swiglu(hidden, router, &gate_up_parameter, &down_parameter, 2)
        .unwrap();
    let geglu = builder
        .moe_geglu(hidden, router, &gate_up_parameter, &down_parameter, 2)
        .unwrap();
    let reglu = builder
        .moe_reglu(hidden, router, &gate_up_parameter, &down_parameter, 2)
        .unwrap();
    let program = builder
        .finish_named(&[
            ("swiglu".to_owned(), swiglu),
            ("geglu".to_owned(), geglu),
            ("reglu".to_owned(), reglu),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let hidden = [
        0.5f32, -0.25, 0.75, -0.5, 1.0, 0.25, 0.125, 0.5, -0.75, 1.0, -0.5, 0.25,
    ];
    let router = [
        2.0f32, 1.0, -1.0, 0.5, 2.0, 1.5, -0.25, -0.25, 1.0, 1.5, 0.0, 1.0,
    ];
    let gate_up = (0..EXPERTS * 2 * INTERMEDIATE * HIDDEN)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 16.0)
        .collect::<Vec<_>>();
    let down = (0..EXPERTS * HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 5 % 13) as f32 - 6.0) / 12.0)
        .collect::<Vec<_>>();
    let mut arguments = executable.args();
    set_float(platform, &mut arguments, "hidden", hidden_shape, &hidden);
    set_float(platform, &mut arguments, "router", router_shape, &router);
    set_parameter_float(
        platform,
        &mut arguments,
        &gate_up_parameter,
        &gate_up,
        nml::Sharding::single(),
    );
    set_parameter_float(
        platform,
        &mut arguments,
        &down_parameter,
        &down,
        nml::Sharding::single(),
    );
    arguments.bake().unwrap();
    let results = arguments.call().unwrap();

    let hidden = rounded(dtype, &hidden);
    let gate_up = rounded(dtype, &gate_up);
    let down = rounded(dtype, &down);
    for (name, activation) in [
        ("swiglu", MoeReferenceActivation::Silu),
        ("geglu", MoeReferenceActivation::Gelu),
        ("reglu", MoeReferenceActivation::Relu),
    ] {
        let expected = moe_reference(
            &hidden,
            &router,
            &gate_up,
            &down,
            activation,
            TOKENS,
            HIDDEN,
            EXPERTS,
            INTERMEDIATE,
        );
        assert_close(&results, name, dtype, &expected);
    }
}

fn execute_expert_parallel_moe_contract(platform: &nml::Platform) {
    const TOKENS: usize = 4;
    const HIDDEN: usize = 3;
    const INTERMEDIATE: usize = 2;

    let experts = platform.device_count().unwrap();
    assert_eq!(experts, 4, "the CPU product mesh owns four expert shards");
    let expert_axis = nml::AxisTag::new(213);
    let mesh = nml::Sharding::mesh(&[(expert_axis, experts)]).unwrap();
    let hidden_shape = Shape::new(DType::F32, &[TOKENS as i64, HIDDEN as i64]).unwrap();
    let router_shape = Shape::new(DType::F32, &[TOKENS as i64, experts as i64]).unwrap();
    let expert_partitions = [
        nml::Partition::Sharded(expert_axis),
        nml::Partition::Replicated,
        nml::Partition::Replicated,
    ];
    let gate_up_shape = Shape::new(
        DType::F32,
        &[experts as i64, (2 * INTERMEDIATE) as i64, HIDDEN as i64],
    )
    .unwrap()
    .with_partitions(&expert_partitions)
    .unwrap();
    let down_shape = Shape::new(
        DType::F32,
        &[experts as i64, HIDDEN as i64, INTERMEDIATE as i64],
    )
    .unwrap()
    .with_partitions(&expert_partitions)
    .unwrap();

    let mut builder = ProgramBuilder::new();
    let hidden = builder.input("parallel_hidden", hidden_shape);
    let router = builder.input("parallel_router", router_shape);
    let gate_up_parameter =
        nml::Parameter::dense("parallel_gate_up", "parallel_gate_up", gate_up_shape).unwrap();
    let down_parameter =
        nml::Parameter::dense("parallel_down", "parallel_down", down_shape).unwrap();
    let output = builder
        .moe_swiglu(hidden, router, &gate_up_parameter, &down_parameter, 2)
        .unwrap();
    let program = builder
        .finish_named(&[("parallel_output".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, mesh.clone()).unwrap();

    let hidden = [
        0.5f32, -0.25, 0.75, -0.5, 1.0, 0.25, 0.125, 0.5, -0.75, 1.0, -0.5, 0.25,
    ];
    // Expert zero receives every token, experts one and two receive two each,
    // and expert three is empty. This exercises repeated assignments,
    // nonuniform load, and a completely empty local expert shard.
    let router = [
        4.0f32, 3.0, 0.0, -10.0, 4.0, 2.0, 1.0, -10.0, 4.0, 1.0, 3.0, -10.0, 4.0, 0.0, 2.0, -10.0,
    ];
    let gate_up = (0..experts * 2 * INTERMEDIATE * HIDDEN)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 16.0)
        .collect::<Vec<_>>();
    let down = (0..experts * HIDDEN * INTERMEDIATE)
        .map(|index| ((index * 5 % 13) as f32 - 6.0) / 12.0)
        .collect::<Vec<_>>();
    let mut arguments = executable.args();
    for (name, shape, values) in [
        ("parallel_hidden", hidden_shape, hidden.as_slice()),
        ("parallel_router", router_shape, router.as_slice()),
    ] {
        let host = nml::Slice::from_typed(shape, values).unwrap();
        let buffer = platform
            .upload(&host, mesh.clone(), nml::Memory::Default)
            .unwrap();
        arguments.set(name, buffer).unwrap();
    }
    set_parameter_float(
        platform,
        &mut arguments,
        &gate_up_parameter,
        &gate_up,
        mesh.clone(),
    );
    set_parameter_float(
        platform,
        &mut arguments,
        &down_parameter,
        &down,
        mesh.clone(),
    );
    arguments.bake().unwrap();
    let results = arguments.call().unwrap();
    let expected = moe_reference(
        &hidden,
        &router,
        &gate_up,
        &down,
        MoeReferenceActivation::Silu,
        TOKENS,
        HIDDEN,
        experts,
        INTERMEDIATE,
    );
    assert_close(&results, "parallel_output", DType::F32, &expected);
}

fn execute_gated_delta_net_contract(platform: &nml::Platform, dtype: DType) {
    let sequence_shape = Shape::new(dtype, &[2, 2, 2]).unwrap();
    let gate_shape = Shape::new(dtype, &[2, 2]).unwrap();
    let state_shape = Shape::new(dtype, &[2, 2, 2]).unwrap();
    let mut builder = ProgramBuilder::new();
    let queries = builder.input("delta_queries", sequence_shape);
    let keys = builder.input("delta_keys", sequence_shape);
    let values = builder.input("delta_values", sequence_shape);
    let alphas = builder.input("delta_alphas", gate_shape);
    let betas = builder.input("delta_betas", gate_shape);
    let state = builder.input("delta_state", state_shape);
    let (outputs, final_state) = builder
        .gated_delta_net(queries, keys, values, alphas, betas, state)
        .unwrap();
    let program = builder
        .finish_named(&[
            ("delta_outputs".to_owned(), outputs),
            ("delta_state".to_owned(), final_state),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();

    let queries = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0];
    let keys = [1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0];
    let values = [3.0, 1.0, 2.0, 4.0, 1.0, 5.0, 3.0, 0.0];
    let alphas = [0.5, 0.25, 0.8, 0.6];
    let betas = [1.0, 0.5, 0.75, 1.0];
    let initial = [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0];
    let mut arguments = executable.args();
    set_float(
        platform,
        &mut arguments,
        "delta_queries",
        sequence_shape,
        &queries,
    );
    set_float(
        platform,
        &mut arguments,
        "delta_keys",
        sequence_shape,
        &keys,
    );
    set_float(
        platform,
        &mut arguments,
        "delta_values",
        sequence_shape,
        &values,
    );
    set_float(
        platform,
        &mut arguments,
        "delta_alphas",
        gate_shape,
        &alphas,
    );
    set_float(platform, &mut arguments, "delta_betas", gate_shape, &betas);
    set_float(
        platform,
        &mut arguments,
        "delta_state",
        state_shape,
        &initial,
    );
    let results = arguments.call().unwrap();

    let (expected_outputs, expected_state) = gated_delta_net_reference(
        &rounded(dtype, &queries),
        &rounded(dtype, &keys),
        &rounded(dtype, &values),
        &rounded(dtype, &alphas),
        &rounded(dtype, &betas),
        &rounded(dtype, &initial),
        2,
        2,
        2,
        2,
    );
    assert_close(&results, "delta_outputs", dtype, &expected_outputs);
    assert_close(&results, "delta_state", dtype, &expected_state);
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_net_reference(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    alphas: &[f32],
    betas: &[f32],
    initial: &[f32],
    sequence: usize,
    heads: usize,
    value_size: usize,
    key_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = initial.to_vec();
    let mut outputs = vec![0.0; sequence * heads * value_size];
    for step in 0..sequence {
        for head in 0..heads {
            let gate = step * heads + head;
            let mut delta = vec![0.0; value_size];
            for value_axis in 0..value_size {
                let mut predicted = 0.0;
                for key_axis in 0..key_size {
                    predicted += state[(head * value_size + value_axis) * key_size + key_axis]
                        * keys[(step * heads + head) * key_size + key_axis];
                }
                delta[value_axis] = (values[(step * heads + head) * value_size + value_axis]
                    - predicted * alphas[gate])
                    * betas[gate];
            }
            for value_axis in 0..value_size {
                for key_axis in 0..key_size {
                    let index = (head * value_size + value_axis) * key_size + key_axis;
                    state[index] = state[index] * alphas[gate]
                        + delta[value_axis] * keys[(step * heads + head) * key_size + key_axis];
                }
            }
            for value_axis in 0..value_size {
                for key_axis in 0..key_size {
                    outputs[(step * heads + head) * value_size + value_axis] += state
                        [(head * value_size + value_axis) * key_size + key_axis]
                        * queries[(step * heads + head) * key_size + key_axis];
                }
            }
        }
    }
    (outputs, state)
}

#[derive(Clone, Copy)]
enum MoeReferenceActivation {
    Silu,
    Gelu,
    Relu,
}

#[allow(clippy::too_many_arguments)]
fn moe_reference(
    hidden: &[f32],
    router: &[f32],
    gate_up: &[f32],
    down: &[f32],
    activation: MoeReferenceActivation,
    tokens: usize,
    hidden_size: usize,
    experts: usize,
    intermediate: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; tokens * hidden_size];
    for token in 0..tokens {
        let logits = &router[token * experts..(token + 1) * experts];
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = probabilities.iter().sum::<f32>();
        probabilities
            .iter_mut()
            .for_each(|value| *value /= denominator);
        let mut routed = (0..experts).collect::<Vec<_>>();
        routed.sort_by(|left, right| {
            probabilities[*right]
                .total_cmp(&probabilities[*left])
                .then_with(|| left.cmp(right))
        });
        routed.truncate(2);
        let routed_sum = routed
            .iter()
            .map(|expert| probabilities[*expert])
            .sum::<f32>();
        for expert in routed {
            let mut projected = vec![0.0f32; 2 * intermediate];
            for output_axis in 0..2 * intermediate {
                for input_axis in 0..hidden_size {
                    projected[output_axis] += hidden[token * hidden_size + input_axis]
                        * gate_up
                            [(expert * 2 * intermediate + output_axis) * hidden_size + input_axis];
                }
            }
            let mut activated = vec![0.0f32; intermediate];
            for axis in 0..intermediate {
                let gate = match activation {
                    MoeReferenceActivation::Silu => {
                        projected[axis] / (1.0 + (-projected[axis]).exp())
                    }
                    MoeReferenceActivation::Gelu => gelu(projected[axis]),
                    MoeReferenceActivation::Relu => projected[axis].max(0.0),
                };
                activated[axis] = gate * projected[intermediate + axis];
            }
            let route_weight = probabilities[expert] / routed_sum;
            for output_axis in 0..hidden_size {
                for input_axis in 0..intermediate {
                    output[token * hidden_size + output_axis] += route_weight
                        * activated[input_axis]
                        * down[(expert * hidden_size + output_axis) * intermediate + input_axis];
                }
            }
        }
    }
    output
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

fn set_parameter_float(
    platform: &nml::Platform,
    arguments: &mut nml::exe::Arguments<'_>,
    parameter: &nml::Parameter,
    values: &[f32],
    sharding: nml::Sharding,
) {
    let shape = parameter.shape();
    let bytes = encode(shape.dtype(), values);
    let host = nml::Slice::from_bytes(shape, &bytes).unwrap();
    let buffer = platform
        .upload(&host, sharding, nml::Memory::Default)
        .unwrap();
    let loaded = nml::LoadedParameter::new(parameter.clone(), vec![buffer]).unwrap();
    arguments.set_parameter(&loaded).unwrap();
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

fn set_u64(
    platform: &nml::Platform,
    arguments: &mut nml::exe::Arguments<'_>,
    name: &str,
    shape: Shape,
    values: &[u64],
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

fn decode_u64(results: &nml::exe::Results, name: &str) -> Vec<u64> {
    result_bytes(results, name)
        .chunks_exact(8)
        .map(|bytes| u64::from_ne_bytes(bytes.try_into().unwrap()))
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

fn assert_float_sequence(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits(),
            "{name}[{index}]: expected {expected:?}, received {actual:?}"
        );
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn decode_bool(results: &nml::exe::Results, name: &str) -> Vec<bool> {
    results
        .get(name)
        .unwrap()
        .to_slice()
        .unwrap()
        .contiguous_bytes()
        .unwrap()
        .iter()
        .map(|value| *value != 0)
        .collect()
}

fn platform() -> nml::Platform {
    match env!("NML_NEURAL_OPS_BACKEND") {
        "cpu" => nml::Platform::cpu().unwrap(),
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
