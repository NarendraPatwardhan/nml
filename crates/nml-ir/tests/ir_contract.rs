use nml_ir::{
    AttentionOptions, ConvolutionOptions, Error, FftType, ProgramBuilder, RopeLayout, RopeOptions,
    RopeScaling,
};
use nml_mlir::Context;
use nml_parameter::Parameter;
use nml_tensor::Element;
use nml_types::{AxisTag, BFloat16, Complex64, Complex128, DType, F16, Layout, Partition, Shape};

fn parameter(name: &str, shape: Shape) -> Parameter {
    Parameter::dense(name, name, shape).unwrap()
}

#[test]
fn one_component_binding_cannot_alias_distinct_parameter_definitions() {
    let shape = Shape::new(DType::F32, &[2, 2]).unwrap();
    let first = Parameter::dense("weight", "first.weight", shape).unwrap();
    let second = Parameter::dense("weight", "second.weight", shape).unwrap();
    let mut builder = ProgramBuilder::new();
    builder.parameter_value(&first).unwrap();
    assert!(matches!(
        builder.parameter_value(&second),
        Err(Error::InvalidParameter(_))
    ));
}

#[test]
fn nvfp4_linear_is_one_semantic_operation_with_three_physical_components() {
    let logical_shape = Shape::new(DType::Bf16, &[4, 16]).unwrap();
    let weight = Parameter::nvfp4(
        "projection.weight",
        "model.projection.weight",
        logical_shape,
    )
    .unwrap();
    let bias = parameter("projection.bias", Shape::new(DType::Bf16, &[4]).unwrap());
    let mut builder = ProgramBuilder::new();
    let input = builder.input("hidden", Shape::new(DType::Bf16, &[2, 3, 16]).unwrap());
    let output = builder.linear(input, &weight, Some(&bias)).unwrap();
    assert_eq!(output.shape().dimensions(), &[2, 3, 4]);
    let program = builder.finish(&[output]).unwrap();
    let context = Context::new();
    let sm75 = program
        .module_with_sharding_cuda(&context, &nml_sharding::Sharding::single(), 24, 7, 5)
        .unwrap();
    sm75.verify().unwrap();
    let sm75 = sm75.text();
    assert_eq!(sm75.matches("nml.nvfp4.turing.linear").count(), 1, "{sm75}");
    assert!(!sm75.contains("__gpu$xla.gpu.triton"), "{sm75}");
    assert!(!sm75.contains("stablehlo.add"), "{sm75}");
    // Blackwell retains this explicitly named compact-weight emulation route
    // until the separate native block-scaled representation is prepared. It
    // must never regress to a dense expansion while native work is pending.
    for (major, minor) in [(8, 0), (9, 0), (10, 0)] {
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(
                &context,
                &nml_sharding::Sharding::single(),
                108,
                major,
                minor,
            )
            .unwrap();
        module.verify().unwrap();
        let text = module.text();
        assert_eq!(text.matches("__gpu$xla.gpu.triton").count(), 1, "{text}");
        assert!(text.contains("nvfp4_linear"), "{text}");
        assert!(!text.contains("nml.nvfp4.linear"), "{text}");
        assert!(!text.contains("stablehlo.add"), "{text}");
        assert!(text.contains("tensor<6x4xbf16>"), "{text}");
    }
    let text = program.stablehlo().unwrap();

    assert_eq!(text.matches("nml.nvfp4.linear").count(), 1, "{text}");
    assert!(text.contains("tensor<4x8xui8>"), "{text}");
    assert!(text.contains("tensor<4x1xui8>"), "{text}");
    assert!(text.contains("tensor<f32>"), "{text}");
    assert!(text.contains("api_version = 4 : i32"), "{text}");
    assert!(!text.contains("stablehlo.add"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();

    let bindings = program.inputs().collect::<Vec<_>>();
    assert_eq!(bindings.len(), 5);
    assert_eq!(bindings[0].0, "hidden");
    assert_eq!(bindings[1].0, "projection.weight.nvfp4.payload");
    assert_eq!(bindings[2].0, "projection.weight.nvfp4.block_scales");
    assert_eq!(bindings[3].0, "projection.weight.nvfp4.global_scale");
    assert_eq!(bindings[4].0, "projection.bias");

    let mut dense_only = ProgramBuilder::new();
    assert!(matches!(
        dense_only.parameter_value(&weight),
        Err(Error::InvalidParameter(message)) if message.contains("requires dense storage")
    ));
}

#[test]
fn nvfp4_decode_qkv_is_one_sm8x_launch_without_changing_portable_semantics() {
    fn projection(name: &str, outputs: i64) -> (Parameter, Parameter) {
        (
            Parameter::nvfp4(
                format!("{name}.weight"),
                format!("model.{name}.weight"),
                Shape::new(DType::Bf16, &[outputs, 16]).unwrap(),
            )
            .unwrap(),
            parameter(
                &format!("{name}.bias"),
                Shape::new(DType::Bf16, &[outputs]).unwrap(),
            ),
        )
    }

    fn program(rows: i64) -> nml_ir::Program {
        let (query_weight, query_bias) = projection("query", 32);
        let (key_weight, key_bias) = projection("key", 8);
        let (value_weight, value_bias) = projection("value", 8);
        let mut builder = ProgramBuilder::new();
        let input = builder.input(
            "hidden",
            Shape::new(DType::Bf16, &[1, rows, 16]).unwrap(),
        );
        let (query, key, value) = builder
            .linear_qkv(
                input,
                &query_weight,
                Some(&query_bias),
                &key_weight,
                Some(&key_bias),
                &value_weight,
                Some(&value_bias),
            )
            .unwrap();
        builder.finish(&[query, key, value]).unwrap()
    }

    let decode = program(1);
    let context = Context::new();
    let sm86 = decode
        .module_with_sharding_cuda(
            &context,
            &nml_sharding::Sharding::single(),
            84,
            8,
            6,
        )
        .unwrap();
    sm86.verify().unwrap();
    let sm86 = sm86.text();
    assert_eq!(sm86.matches("__gpu$xla.gpu.triton").count(), 1, "{sm86}");
    assert!(sm86.contains("nvfp4_qkv_gemv"), "{sm86}");
    assert!(!sm86.contains("nvfp4_linear_gemv"), "{sm86}");

    let context = Context::new();
    let sm75 = decode
        .module_with_sharding_cuda(
            &context,
            &nml_sharding::Sharding::single(),
            40,
            7,
            5,
        )
        .unwrap();
    sm75.verify().unwrap();
    let sm75 = sm75.text();
    assert_eq!(sm75.matches("nml.nvfp4.turing.linear").count(), 3, "{sm75}");
    assert!(!sm75.contains("nvfp4_qkv_gemv"), "{sm75}");

    let portable = decode.stablehlo().unwrap();
    assert_eq!(portable.matches("nml.nvfp4.linear").count(), 3, "{portable}");
    assert!(!portable.contains("nvfp4_qkv_gemv"), "{portable}");

    let context = Context::new();
    let prefill = program(2)
        .module_with_sharding_cuda(
            &context,
            &nml_sharding::Sharding::single(),
            84,
            8,
            6,
        )
        .unwrap();
    prefill.verify().unwrap();
    let prefill = prefill.text();
    assert_eq!(prefill.matches("__gpu$xla.gpu.triton").count(), 3, "{prefill}");
    assert!(!prefill.contains("nvfp4_qkv_gemv"), "{prefill}");
}

#[test]
fn nvfp4_linear_validates_optional_bias_before_authoring_the_compact_operation() {
    let dtype = DType::F16;
    let weight = Parameter::nvfp4(
        "projection.weight",
        "model.projection.weight",
        Shape::new(dtype, &[4, 16]).unwrap(),
    )
    .unwrap();
    let wrong_width = parameter("wrong_width", Shape::new(dtype, &[3]).unwrap());
    let wrong_rank = parameter("wrong_rank", Shape::new(dtype, &[1, 4]).unwrap());
    let wrong_dtype = parameter("wrong_dtype", Shape::new(DType::Bf16, &[4]).unwrap());
    let mut builder = ProgramBuilder::new();
    let input = builder.input("hidden", Shape::new(dtype, &[2, 16]).unwrap());

    assert!(matches!(
        builder.linear(input, &weight, Some(&wrong_width)),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(matches!(
        builder.linear(input, &weight, Some(&wrong_rank)),
        Err(Error::RankMismatch {
            operation: "linear bias",
            ..
        })
    ));
    assert!(matches!(
        builder.linear(input, &weight, Some(&wrong_dtype)),
        Err(Error::DTypeMismatch { .. })
    ));

    let output = builder.linear(input, &weight, None).unwrap();
    let text = builder.finish(&[output]).unwrap().stablehlo().unwrap();
    assert_eq!(text.matches("nml.nvfp4.linear").count(), 1, "{text}");
    assert!(!text.contains("stablehlo.add"), "{text}");
    assert!(!text.contains("wrong_width"), "{text}");
    assert!(!text.contains("wrong_rank"), "{text}");
    assert!(!text.contains("wrong_dtype"), "{text}");
}

#[test]
fn nvfp4_embedding_preserves_index_shape_and_decodes_only_selected_rows() {
    let weight = Parameter::nvfp4(
        "embedding.weight",
        "model.embedding.weight",
        Shape::new(DType::F16, &[32, 17]).unwrap(),
    )
    .unwrap();
    let mut builder = ProgramBuilder::new();
    let indices = builder.input("token_ids", Shape::new(DType::I32, &[2, 3]).unwrap());
    let output = builder.token_embedding(&weight, indices).unwrap();
    assert_eq!(output.shape().dtype(), DType::F16);
    assert_eq!(output.shape().dimensions(), &[2, 3, 17]);
    let program = builder.finish(&[output]).unwrap();
    let context = Context::new();
    let sm75 = program
        .module_with_sharding_cuda(&context, &nml_sharding::Sharding::single(), 24, 7, 5)
        .unwrap();
    sm75.verify().unwrap();
    let sm75 = sm75.text();
    assert_eq!(
        sm75.matches("nml.nvfp4.turing.embedding").count(),
        1,
        "{sm75}"
    );
    assert!(!sm75.contains("__gpu$xla.gpu.triton"), "{sm75}");
    for (major, minor) in [(8, 0), (9, 0), (10, 0)] {
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(
                &context,
                &nml_sharding::Sharding::single(),
                108,
                major,
                minor,
            )
            .unwrap();
        module.verify().unwrap();
        let text = module.text();
        assert_eq!(text.matches("__gpu$xla.gpu.triton").count(), 1, "{text}");
        assert!(text.contains("nvfp4_embedding"), "{text}");
        assert!(!text.contains("nml.nvfp4.embedding"), "{text}");
        assert!(text.contains("tensor<6x17xf16>"), "{text}");
    }
    let text = program.stablehlo().unwrap();
    assert_eq!(text.matches("nml.nvfp4.embedding").count(), 1, "{text}");
    assert!(text.contains("tensor<32x9xui8>"), "{text}");
    assert!(text.contains("tensor<32x2xui8>"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn clamped_swiglu_moe_keeps_model_semantics_above_weight_representation() {
    fn build(nvfp4: bool) -> nml_ir::Program {
        let dtype = DType::Bf16;
        let mut builder = ProgramBuilder::new();
        let hidden = builder.input("hidden", Shape::new(dtype, &[2, 4]).unwrap());
        let router = builder.input("router", Shape::new(DType::F32, &[2, 3]).unwrap());
        let gate_shape = Shape::new(dtype, &[3, 10, 4]).unwrap();
        let down_shape = Shape::new(dtype, &[3, 4, 5]).unwrap();
        let gate = if nvfp4 {
            Parameter::nvfp4("gate", "model.gate", gate_shape).unwrap()
        } else {
            parameter("gate", gate_shape)
        };
        let down = if nvfp4 {
            Parameter::nvfp4("down", "model.down", down_shape).unwrap()
        } else {
            parameter("down", down_shape)
        };
        let gate_bias = parameter("gate_bias", Shape::new(dtype, &[3, 10]).unwrap());
        let down_bias = parameter("down_bias", Shape::new(dtype, &[3, 4]).unwrap());
        let output = builder
            .routed_clamped_swiglu(hidden, router, &gate, &gate_bias, &down, &down_bias, 2)
            .unwrap();
        assert_eq!(output.shape().dimensions(), &[2, 4]);
        builder.finish(&[output]).unwrap()
    }

    let dense = build(false).stablehlo().unwrap();
    assert!(!dense.contains("nml.nvfp4"), "{dense}");
    assert!(dense.contains("stablehlo.dot_general"), "{dense}");
    assert!(dense.contains("stablehlo.slice"), "{dense}");
    assert!(dense.contains("stablehlo.clamp"), "{dense}");
    Context::new()
        .parse_module(&dense)
        .unwrap()
        .verify()
        .unwrap();

    let compact_program = build(true);
    let compact = compact_program.stablehlo().unwrap();
    assert_eq!(
        compact.matches("nml.nvfp4.routed_swiglu").count(),
        1,
        "{compact}"
    );
    assert!(compact.contains("stablehlo.sort"), "{compact}");
    assert!(compact.contains("tensor<3x10x2xui8>"), "{compact}");
    assert!(compact.contains("tensor<3x4x3xui8>"), "{compact}");
    Context::new()
        .parse_module(&compact)
        .unwrap()
        .verify()
        .unwrap();

    let context = Context::new();
    let sm75 = compact_program
        .module_with_sharding_cuda(&context, &nml_sharding::Sharding::single(), 24, 7, 5)
        .unwrap();
    sm75.verify().unwrap();
    let sm75 = sm75.text();
    assert_eq!(
        sm75.matches("nml.nvfp4.turing.expert_gate_up").count(),
        1,
        "{sm75}"
    );
    assert_eq!(
        sm75.matches("nml.nvfp4.turing.expert_down").count(),
        1,
        "{sm75}"
    );
    assert!(!sm75.contains("__gpu$xla.gpu.triton"), "{sm75}");
    assert!(!sm75.contains("nml.nvfp4.routed_swiglu"), "{sm75}");
    assert!(sm75.contains("stablehlo.reduce"), "{sm75}");

    for (major, minor) in [(8, 0), (9, 0), (10, 0)] {
        let context = Context::new();
        let module = compact_program
            .module_with_sharding_cuda(
                &context,
                &nml_sharding::Sharding::single(),
                108,
                major,
                minor,
            )
            .unwrap();
        module.verify().unwrap();
        let text = module.text();
        assert_eq!(text.matches("__gpu$xla.gpu.triton").count(), 2, "{text}");
        assert!(text.contains("nvfp4_grouped_gate_up"), "{text}");
        assert!(text.contains("nvfp4_grouped_down"), "{text}");
        assert!(!text.contains("nml.nvfp4.routed_swiglu"), "{text}");
        assert!(text.contains("stablehlo.reduce"), "{text}");
    }
}

#[test]
fn decode_shaped_moe_launches_only_selected_expert_blocks() {
    let dtype = DType::Bf16;
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input("hidden", Shape::new(dtype, &[1, 64]).unwrap());
    let router = builder.input("router", Shape::new(DType::F32, &[1, 32]).unwrap());
    let gate = Parameter::nvfp4(
        "gate",
        "model.gate",
        Shape::new(dtype, &[32, 128, 64]).unwrap(),
    )
    .unwrap();
    let gate_bias = parameter("gate_bias", Shape::new(dtype, &[32, 128]).unwrap());
    let down = Parameter::nvfp4(
        "down",
        "model.down",
        Shape::new(dtype, &[32, 64, 64]).unwrap(),
    )
    .unwrap();
    let down_bias = parameter("down_bias", Shape::new(dtype, &[32, 64]).unwrap());
    let output = builder
        .routed_clamped_swiglu(hidden, router, &gate, &gate_bias, &down, &down_bias, 4)
        .unwrap();
    let program = builder.finish(&[output]).unwrap();
    let text = program
        .module_with_sharding_cuda(
            &Context::new(),
            &nml_sharding::Sharding::single(),
            108,
            8,
            6,
        )
        .unwrap()
        .text();

    assert!(text.contains("tensor<64xi32>"), "{text}");
    assert!(text.contains("tensor<4xi32>"), "{text}");
    assert!(text.contains("tensor<16xi32>"), "{text}");
    assert!(text.contains("tensor<256xi32>"), "{text}");
    assert_eq!(
        text.matches("grid_x = 4 : i32").count(),
        2,
        "gate/up and down must launch exactly one block per selected route: {text}"
    );
    assert_eq!(text.matches("num_warps = 8 : i32").count(), 0, "{text}");
    assert_eq!(text.matches("num_warps = 4 : i32").count(), 2, "{text}");
    assert!(text.contains("stablehlo.pad"), "{text}");
    assert_eq!(text.matches("scf.if").count(), 2, "{text}");
}

#[test]
fn matmul_is_typed_deterministic_and_verified() {
    fn build() -> String {
        let mut builder = ProgramBuilder::new();
        let left = builder.input("left", Shape::new(DType::F32, &[3, 5]).unwrap());
        let right = builder.input("right", Shape::new(DType::F32, &[5, 4]).unwrap());
        let result = builder.matmul(left, right).unwrap();
        assert_eq!(result.shape().dimensions(), &[3, 4]);
        builder.finish(&[result]).unwrap().stablehlo().unwrap()
    }

    let first = build();
    let second = build();
    assert_eq!(first, second);
    assert!(first.contains("contracting_dims = [1] x [0]"), "{first}");
    let context = Context::new();
    context.parse_module(&first).unwrap().verify().unwrap();
}

#[test]
fn every_canonical_storage_dtype_builds_an_owned_scalar_constant() {
    fn verify<T: Element>(value: T) {
        let mut builder = ProgramBuilder::new();
        let constant = builder.scalar(value).unwrap();
        let module = builder.finish(&[constant]).unwrap().stablehlo().unwrap();
        assert!(module.contains("stablehlo.constant"));
    }

    verify(false);
    verify(-1i8);
    verify(-2i16);
    verify(-3i32);
    verify(-4i64);
    verify(1u8);
    verify(2u16);
    verify(3u32);
    verify(4u64);
    verify(F16::from_f32(0.5));
    verify(BFloat16::from_f32(-0.75));
    verify(1.25f32);
    verify(-2.5f64);
    verify(F16::from_f32(f32::INFINITY));
    verify(BFloat16::from_f32(f32::NAN));
    verify(f32::NEG_INFINITY);
    verify(f64::NAN);
    verify(Complex64 {
        real: 1.0,
        imaginary: -2.0,
    });
    verify(Complex128 {
        real: 3.0,
        imaginary: -4.0,
    });
}

#[test]
fn invalid_matmul_fails_before_mlir_construction() {
    let mut builder = ProgramBuilder::new();
    let left = builder.input("left", Shape::new(DType::F32, &[3, 5]).unwrap());
    let right = builder.input("right", Shape::new(DType::F32, &[6, 4]).unwrap());
    assert!(matches!(
        builder.matmul(left, right),
        Err(Error::DimensionMismatch { .. })
    ));
}

#[test]
fn linear_contracts_the_final_activation_axis_at_any_rank() {
    let mut builder = ProgramBuilder::new();
    let batch = AxisTag::new(11);
    let sequence = AxisTag::new(12);
    let model = AxisTag::new(13);
    let output = AxisTag::new(14);
    let input_shape = Shape::new(DType::Bf16, &[2, 3, 4])
        .unwrap()
        .with_axis_tags(&[batch, sequence, model])
        .unwrap()
        .with_partitions(&[
            Partition::Sharded(batch),
            Partition::Unspecified,
            Partition::Unspecified,
        ])
        .unwrap();
    let weight_shape = Shape::new(DType::Bf16, &[5, 4])
        .unwrap()
        .with_axis_tags(&[output, model])
        .unwrap();
    let input = builder.input("input", input_shape);
    let weight = parameter("weight", weight_shape);
    let bias = parameter("bias", Shape::new(DType::Bf16, &[5]).unwrap());
    let result = builder.linear(input, &weight, Some(&bias)).unwrap();

    assert_eq!(result.shape().dimensions(), &[2, 3, 5]);
    assert_eq!(result.shape().axis_tags(), &[batch, sequence, output]);
    assert_eq!(
        result.shape().partitions(),
        &[
            Partition::Sharded(batch),
            Partition::Unspecified,
            Partition::Unspecified,
        ]
    );

    let program = builder.finish(&[result]).unwrap();
    let mesh = nml_sharding::Sharding::mesh(&[(batch, 2)]).unwrap();
    let text = program.stablehlo_with_sharding(&mesh).unwrap();
    assert!(text.contains("contracting_dims = [2] x [1]"), "{text}");
    assert!(text.contains("dims = [2]"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn linear_rejects_scalar_activations_before_mlir_construction() {
    let mut builder = ProgramBuilder::new();
    let scalar = builder.input("scalar", Shape::new(DType::F32, &[]).unwrap());
    let weight = parameter("weight", Shape::new(DType::F32, &[2, 1]).unwrap());
    assert!(matches!(
        builder.linear(scalar, &weight, None),
        Err(Error::InvalidLinearAlgebra(_))
    ));
}

#[test]
fn attention_primitives_are_typed_and_verify_as_stablehlo() {
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[4, 3]).unwrap());
    let update = builder.input("update", Shape::new(DType::F32, &[1, 3]).unwrap());
    let indices = builder.input("indices", Shape::new(DType::I32, &[2]).unwrap());
    let weight = parameter("weight", Shape::new(DType::F32, &[3]).unwrap());
    let weight = builder.parameter_value(&weight).unwrap();
    let zero = builder.scalar(0i32).unwrap();
    let one = builder.scalar(1i32).unwrap();

    let prefix = builder.slice(input, &[0, 0], &[2, 3], &[1, 1]).unwrap();
    let rebuilt = builder.concatenate(&[prefix, prefix], 0).unwrap();
    assert_eq!(rebuilt.shape().dimensions(), &[4, 3]);
    let dynamic = builder.dynamic_slice(input, &[one, zero], &[2, 3]).unwrap();
    let updated = builder
        .dynamic_update_slice(input, update, &[one, zero])
        .unwrap();
    let gathered = builder.gather(input, indices, 0).unwrap();
    assert_eq!(gathered.shape().dimensions(), &[2, 3]);
    let positions = builder
        .iota(Shape::new(DType::I32, &[2, 3]).unwrap(), 1)
        .unwrap();
    let sum = builder.reduce_sum(input, &[0]).unwrap();
    let maximum = builder.reduce_max(input, &[1]).unwrap();
    let softmax = builder.softmax(input, 1).unwrap();
    let normalized = builder.rms_norm(input, Some(weight), 1, 1e-5).unwrap();

    let program = builder
        .finish(&[
            rebuilt, dynamic, updated, gathered, positions, sum, maximum, softmax, normalized,
        ])
        .unwrap();
    let text = program.stablehlo().unwrap();
    for operation in [
        "stablehlo.slice",
        "stablehlo.concatenate",
        "stablehlo.dynamic_slice",
        "stablehlo.dynamic_update_slice",
        "stablehlo.gather",
        "stablehlo.iota",
        "stablehlo.reduce",
        "stablehlo.exponential",
        "stablehlo.rsqrt",
    ] {
        assert!(text.contains(operation), "missing {operation} in {text}");
    }
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn attention_primitives_reject_invalid_contracts_before_mlir() {
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[4, 3]).unwrap());
    let bad_indices = builder.input("bad_indices", Shape::new(DType::F32, &[2]).unwrap());
    assert!(matches!(
        builder.gather(input, bad_indices, 0),
        Err(Error::InvalidIndexDType(DType::F32))
    ));
    assert!(matches!(
        builder.slice(input, &[0, 0], &[5, 3], &[1, 1]),
        Err(Error::InvalidSlice { axis: 0, .. })
    ));
    assert!(matches!(
        builder.concatenate(&[], 0),
        Err(Error::EmptyOperands("concatenate"))
    ));
    assert!(matches!(
        builder.reduce_max(input, &[2]),
        Err(Error::AxisOutOfBounds { .. })
    ));

    let tagged = Shape::new(DType::F32, &[4, 3])
        .unwrap()
        .with_axis_tags(&[AxisTag::new(91), AxisTag::new(92)])
        .unwrap();
    let input = builder.input("tagged_input", tagged);
    let update = builder.input("untagged_update", Shape::new(DType::F32, &[1, 3]).unwrap());
    let zero = builder.scalar(0i32).unwrap();
    assert!(matches!(
        builder.dynamic_update_slice(input, update, &[zero, zero]),
        Err(Error::MetadataMismatch {
            operation: "dynamic_update_slice",
            field: "axis tags",
        })
    ));
}

#[test]
fn model_enabling_operations_are_typed_and_verify_as_stablehlo() {
    let mut builder = ProgramBuilder::new();
    let shape = Shape::new(DType::F32, &[2, 4]).unwrap();
    let input = builder.input("input", shape);
    let gate = builder.input("gate", shape);
    let one = builder.scalar(1.0f32).unwrap();
    let two = builder.scalar(2.0f32).unwrap();
    let low = builder.scalar(-1.0f32).unwrap();
    let high = builder.scalar(1.0f32).unwrap();
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
    let l2 = builder.normalize_l2(input, &[1], 1e-12).unwrap();
    let norm_weight = parameter("norm_weight", Shape::new(DType::F32, &[4]).unwrap());
    let norm_weight = builder.parameter_value(&norm_weight).unwrap();
    let norm_bias = parameter("norm_bias", Shape::new(DType::F32, &[4]).unwrap());
    let norm_bias = builder.parameter_value(&norm_bias).unwrap();
    let layer_norm = builder
        .layer_norm(input, Some(norm_weight), Some(norm_bias), 1, 1e-5)
        .unwrap();
    let swiglu = builder.swiglu(gate, input).unwrap();
    let geglu = builder.geglu(gate, input).unwrap();

    let embedding_weight = parameter("embedding_weight", Shape::new(DType::F32, &[5, 4]).unwrap());
    let token_ids = builder.input("token_ids", Shape::new(DType::I32, &[2, 3]).unwrap());
    let embedding = builder
        .token_embedding(&embedding_weight, token_ids)
        .unwrap();
    assert_eq!(embedding.shape().dimensions(), &[2, 3, 4]);

    let scores = builder.input("scores", Shape::new(DType::F32, &[2, 5]).unwrap());
    let (maxima, indices) = builder.argmax(scores, 1).unwrap();
    assert_eq!(maxima.shape().dimensions(), &[2]);
    assert_eq!(indices.shape().dimensions(), &[2]);
    assert_eq!(indices.shape().dtype(), DType::I32);

    let complex = builder.input("complex", Shape::new(DType::C64, &[2]).unwrap());
    let magnitude = builder.abs(complex).unwrap();
    assert_eq!(magnitude.shape().dtype(), DType::F32);

    let program = builder
        .finish(&[
            power,
            remainder,
            clamped,
            floor,
            ceil,
            minimum,
            mean,
            log_sum_exp,
            normalized,
            l2,
            layer_norm,
            swiglu,
            geglu,
            embedding,
            maxima,
            indices,
            magnitude,
        ])
        .unwrap();
    let text = program.stablehlo().unwrap();
    for operation in [
        "stablehlo.abs",
        "stablehlo.power",
        "stablehlo.remainder",
        "stablehlo.clamp",
        "stablehlo.floor",
        "stablehlo.ceil",
        "stablehlo.gather",
        "stablehlo.reduce",
    ] {
        assert!(text.contains(operation), "missing {operation} in {text}");
    }
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn model_enabling_operations_reject_invalid_contracts_before_mlir() {
    let mut builder = ProgramBuilder::new();
    let floats = builder.input("floats", Shape::new(DType::F32, &[2, 4]).unwrap());
    let bools = builder.input("bools", Shape::new(DType::Bool, &[2, 4]).unwrap());
    let bad_weight = parameter("bad_weight", Shape::new(DType::F32, &[5]).unwrap());
    let good_weight = parameter("good_weight", Shape::new(DType::F32, &[5, 4]).unwrap());
    let bad_ids = builder.input("bad_ids", Shape::new(DType::F32, &[2]).unwrap());
    assert!(matches!(
        builder.abs(bools),
        Err(Error::UnsupportedDType {
            operation: "abs",
            dtype: DType::Bool,
        })
    ));
    assert!(matches!(
        builder.token_embedding(&bad_weight, bad_ids),
        Err(Error::RankMismatch { .. })
    ));
    assert!(matches!(
        builder.token_embedding(&good_weight, bad_ids),
        Err(Error::InvalidIndexDType(DType::F32))
    ));
    assert!(matches!(
        builder.layer_norm(floats, None, None, 2, 1e-5),
        Err(Error::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        builder.normalize_l2(floats, &[1], 0.0),
        Err(Error::InvalidNormalization(_))
    ));
    assert!(matches!(
        builder.normalize_l2(floats, &[], 1e-5),
        Err(Error::InvalidNormalization(_))
    ));
    assert!(matches!(
        builder.argmax(floats, 2),
        Err(Error::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        builder.argmax(bools, 1),
        Err(Error::UnsupportedDType {
            operation: "argmax",
            dtype: DType::Bool,
        })
    ));

    let huge = builder.input(
        "huge",
        Shape::new(DType::F32, &[i32::MAX as i64 + 1]).unwrap(),
    );
    let (_, indices) = builder.argmax(huge, 0).unwrap();
    assert_eq!(indices.shape().dtype(), DType::I64);
}

#[test]
fn logical_operations_are_typed_broadcasting_stablehlo() {
    let tagged = Shape::new(DType::I32, &[2, 4])
        .unwrap()
        .with_axis_tags(&[AxisTag::new(71), AxisTag::new(72)])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let left = builder.input("left", tagged);
    let right = builder.input("right", tagged);
    let scalar = builder.scalar(0x0fi32).unwrap();
    let and = builder.logical_and(left, scalar).unwrap();
    let or = builder.logical_or(left, right).unwrap();
    let xor = builder.logical_xor(left, right).unwrap();
    let not = builder.logical_not(left).unwrap();
    let predicate = builder.less(left, right).unwrap();
    let predicate_not = builder.logical_not(predicate).unwrap();
    assert_eq!(and.shape(), tagged);
    assert_eq!(predicate_not.shape().dtype(), DType::Bool);
    let floats = builder.input("floats", Shape::new(DType::F32, &[2]).unwrap());
    assert!(matches!(
        builder.logical_not(floats),
        Err(Error::UnsupportedDType {
            operation: "logical_not",
            dtype: DType::F32,
        })
    ));

    let text = builder
        .finish(&[and, or, xor, not, predicate_not])
        .unwrap()
        .stablehlo()
        .unwrap();
    for operation in [
        "stablehlo.and",
        "stablehlo.or",
        "stablehlo.xor",
        "stablehlo.not",
    ] {
        assert!(text.contains(operation), "missing {operation} in {text}");
    }
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn bit_and_precision_primitives_are_typed_verified_stablehlo() {
    let tagged = Shape::new(DType::I32, &[2, 4])
        .unwrap()
        .with_axis_tags(&[AxisTag::new(81), AxisTag::new(82)])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let integers = builder.input("integers", tagged);
    let shift_amount = builder.scalar(3i32).unwrap();
    let shifted_left = builder.shift_left(integers, shift_amount).unwrap();
    let shifted_arithmetic = builder
        .shift_right_arithmetic(integers, shift_amount)
        .unwrap();
    let shifted_logical = builder.shift_right_logical(integers, shift_amount).unwrap();
    let leading_zeros = builder.count_leading_zeros(integers).unwrap();
    let population = builder.population_count(integers).unwrap();
    let bytes = builder.bitcast(integers, DType::U8).unwrap();
    assert_eq!(bytes.shape().dimensions(), &[2, 4, 4]);
    assert_eq!(bytes.shape().axis_tags()[2], AxisTag::UNKNOWN);
    let reconstructed = builder.bitcast(bytes, DType::I32).unwrap();
    assert_eq!(reconstructed.shape(), tagged);

    let floats = builder.input("floats", Shape::new(DType::F32, &[2, 4]).unwrap());
    let finite = builder.is_finite(floats).unwrap();
    let sign = builder.sign(floats).unwrap();
    let expm1 = builder.expm1(floats).unwrap();
    let round_away = builder.round_nearest_away_from_zero(floats).unwrap();
    let round_even = builder.round_nearest_even(floats).unwrap();
    let reduced = builder.reduce_precision(floats, 5, 10).unwrap();
    assert_eq!(finite.shape().dtype(), DType::Bool);

    let text = builder
        .finish(&[
            shifted_left,
            shifted_arithmetic,
            shifted_logical,
            leading_zeros,
            population,
            reconstructed,
            finite,
            sign,
            expm1,
            round_away,
            round_even,
            reduced,
        ])
        .unwrap()
        .stablehlo()
        .unwrap();
    for operation in [
        "stablehlo.shift_left",
        "stablehlo.shift_right_arithmetic",
        "stablehlo.shift_right_logical",
        "stablehlo.count_leading_zeros",
        "stablehlo.popcnt",
        "stablehlo.bitcast_convert",
        "stablehlo.is_finite",
        "stablehlo.sign",
        "stablehlo.exponential_minus_one",
        "stablehlo.round_nearest_afz",
        "stablehlo.round_nearest_even",
        "stablehlo.reduce_precision",
    ] {
        assert!(text.contains(operation), "missing {operation} in {text}");
    }
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn bit_and_precision_primitives_reject_ambiguous_contracts() {
    use nml_types::Partition;

    let mut builder = ProgramBuilder::new();
    let floats = builder.input("floats", Shape::new(DType::F32, &[2]).unwrap());
    assert!(matches!(
        builder.population_count(floats),
        Err(Error::UnsupportedDType {
            operation: "population_count",
            dtype: DType::F32,
        })
    ));
    assert!(matches!(
        builder.reduce_precision(floats, 0, 10),
        Err(Error::InvalidPrecision { .. })
    ));

    let tagged_minor = Shape::new(DType::U8, &[2, 4])
        .unwrap()
        .with_axis_tags(&[AxisTag::new(90), AxisTag::new(91)])
        .unwrap();
    let tagged_minor = builder.input("tagged_minor", tagged_minor);
    assert!(matches!(
        builder.bitcast(tagged_minor, DType::I32),
        Err(Error::InvalidBitcast(_))
    ));

    let sharded_minor = Shape::new(DType::U8, &[2, 4])
        .unwrap()
        .with_partitions(&[Partition::Unspecified, Partition::Sharded(AxisTag::new(92))])
        .unwrap();
    let sharded_minor = builder.input("sharded_minor", sharded_minor);
    assert!(matches!(
        builder.bitcast(sharded_minor, DType::I32),
        Err(Error::InvalidBitcast(_))
    ));
}

#[test]
fn structural_construction_and_movement_are_typed_verified_compositions() {
    let row = AxisTag::new(101);
    let column = AxisTag::new(102);
    let shape = Shape::new(DType::F32, &[2, 3])
        .unwrap()
        .with_axis_tags(&[row, column])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let padding = builder.scalar(0.0f32).unwrap();
    let padded = builder
        .pad(input, padding, &[1, 0], &[0, 1], &[0, 1])
        .unwrap();
    assert_eq!(padded.shape().dimensions(), &[3, 6]);
    assert_eq!(padded.shape().axis_tags(), &[row, column]);
    let reversed = builder.reverse(input, &[0, 1]).unwrap();
    let inserted = builder.insert_axis(input, 1, AxisTag::new(103)).unwrap();
    let squeezed = builder.squeeze(inserted, 1).unwrap();
    assert_eq!(squeezed.shape(), shape);
    let stacked = builder
        .stack(&[input, input], 1, AxisTag::new(104))
        .unwrap();
    assert_eq!(stacked.shape().dimensions(), &[2, 2, 3]);
    let repeated = builder.repeat(input, 1, 2).unwrap();
    let stuttered = builder.stutter(input, 1, 2).unwrap();
    assert_eq!(repeated.shape().dimensions(), &[2, 6]);
    assert_eq!(stuttered.shape().dimensions(), &[2, 6]);
    let split = builder.split(input, 1, &[1, 2]).unwrap();
    let chunks = builder.chunks(input, 1, 3).unwrap();
    assert_eq!(split[0].shape().dimensions(), &[2, 1]);
    assert_eq!(chunks.len(), 3);

    let left = builder.input("left", Shape::new(DType::F32, &[2]).unwrap());
    let right = builder.input("right", Shape::new(DType::F32, &[3]).unwrap());
    let outer = builder.outer(left, right).unwrap();
    assert_eq!(outer.shape().dimensions(), &[2, 3]);
    let diagonal = builder
        .diagonal(right, 0, AxisTag::new(105), AxisTag::new(106))
        .unwrap();
    assert_eq!(diagonal.shape().dimensions(), &[3, 3]);
    let triangular = builder.triangular(input, 0, 1, 0).unwrap();
    let product = builder.cartesian_product(&[left, right]).unwrap();
    let product_stacked = builder
        .cartesian_product_stacked(&[left, right], AxisTag::new(107))
        .unwrap();
    assert_eq!(product[0].shape().dimensions(), &[2, 3]);
    assert_eq!(product_stacked.shape().dimensions(), &[2, 3, 2]);
    let rolled = builder.roll(input, 1, -1).unwrap();
    let barrier = builder.optimization_barrier(input).unwrap();

    let mut outputs = vec![
        padded,
        reversed,
        squeezed,
        stacked,
        repeated,
        stuttered,
        outer,
        diagonal,
        triangular,
        product_stacked,
        rolled,
        barrier,
    ];
    outputs.extend(split);
    outputs.extend(chunks);
    outputs.extend(product);
    let text = builder.finish(&outputs).unwrap().stablehlo().unwrap();
    for operation in [
        "stablehlo.pad",
        "stablehlo.reverse",
        "stablehlo.reshape",
        "stablehlo.broadcast_in_dim",
        "stablehlo.concatenate",
        "stablehlo.slice",
        "stablehlo.dot_general",
        "stablehlo.iota",
        "stablehlo.select",
        "stablehlo.optimization_barrier",
    ] {
        assert!(text.contains(operation), "missing {operation} in {text}");
    }
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn structural_compositions_reject_metadata_loss_and_ambiguous_shapes() {
    use nml_types::Partition;

    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[2, 3]).unwrap());
    let padding = builder.scalar(0.0f32).unwrap();
    assert!(matches!(
        builder.pad(input, padding, &[0], &[0], &[0]),
        Err(Error::InvalidStructure(_))
    ));
    assert!(matches!(
        builder.squeeze(input, 0),
        Err(Error::InvalidStructure(_))
    ));
    assert!(matches!(
        builder.split(input, 1, &[1, 1]),
        Err(Error::InvalidStructure(_))
    ));
    assert!(matches!(
        builder.chunks(input, 1, 2),
        Err(Error::InvalidStructure(_))
    ));

    let sharded = Shape::new(DType::F32, &[3])
        .unwrap()
        .with_partitions(&[Partition::Sharded(AxisTag::new(108))])
        .unwrap();
    let sharded = builder.input("sharded", sharded);
    assert!(matches!(
        builder.diagonal(sharded, 0, AxisTag::new(109), AxisTag::new(110)),
        Err(Error::InvalidStructure(_))
    ));
}

#[test]
fn retained_linear_algebra_is_batched_typed_and_verified() {
    let mut builder = ProgramBuilder::new();
    let coefficient = builder.input("coefficient", Shape::new(DType::F32, &[2, 3, 3]).unwrap());
    let right_hand_side = builder.input(
        "right_hand_side",
        Shape::new(DType::F32, &[2, 3, 2]).unwrap(),
    );
    let factor = builder.cholesky(coefficient, true).unwrap();
    let solution = builder
        .triangular_solve(coefficient, right_hand_side, true)
        .unwrap();
    assert_eq!(factor.shape(), coefficient.shape());
    assert_eq!(solution.shape(), right_hand_side.shape());
    let text = builder
        .finish(&[factor, solution])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(text.contains("stablehlo.cholesky"), "{text}");
    assert!(text.contains("stablehlo.triangular_solve"), "{text}");
    assert!(text.contains("NO_TRANSPOSE"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();

    let mut builder = ProgramBuilder::new();
    let nonsquare = builder.input("nonsquare", Shape::new(DType::F32, &[2, 3]).unwrap());
    assert!(matches!(
        builder.cholesky(nonsquare, true),
        Err(Error::InvalidLinearAlgebra(_))
    ));
    let integers = builder.input("integers", Shape::new(DType::I32, &[2, 2]).unwrap());
    assert!(matches!(
        builder.cholesky(integers, true),
        Err(Error::UnsupportedDType {
            operation: "cholesky",
            dtype: DType::I32,
        })
    ));
}

#[test]
fn window_convolution_pooling_and_resize_form_one_verified_substrate() {
    let batch = AxisTag::new(401);
    let feature = AxisTag::new(402);
    let height = AxisTag::new(403);
    let width = AxisTag::new(404);
    let output_feature = AxisTag::new(405);
    let input_shape = Shape::new(DType::F32, &[1, 5, 5, 2])
        .unwrap()
        .with_axis_tags(&[batch, height, width, feature])
        .unwrap();
    let kernel_shape = Shape::new(DType::F32, &[3, 3, 2, 4])
        .unwrap()
        .with_axis_tags(&[AxisTag::UNKNOWN, AxisTag::UNKNOWN, feature, output_feature])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("image", input_shape);
    let kernel = builder.input("kernel", kernel_shape);
    let convolution = builder
        .convolution(
            input,
            kernel,
            ConvolutionOptions {
                strides: &[1, 1],
                padding: &[[1, 1], [1, 1]],
                input_dilation: &[1, 1],
                kernel_dilation: &[1, 1],
                kernel_reversal: &[false, false],
                input_batch_axis: 0,
                input_feature_axis: 3,
                input_spatial_axes: &[1, 2],
                kernel_input_feature_axis: 2,
                kernel_output_feature_axis: 3,
                kernel_spatial_axes: &[0, 1],
                output_batch_axis: 0,
                output_feature_axis: 3,
                output_spatial_axes: &[1, 2],
                feature_groups: 1,
                batch_groups: 1,
            },
        )
        .unwrap();
    assert_eq!(convolution.shape().dimensions(), &[1, 5, 5, 4]);
    assert_eq!(
        convolution.shape().axis_tags(),
        &[batch, height, width, output_feature]
    );

    let pooled = builder
        .max_pool2d(input, [1, 2], [2, 2], [2, 2], [[0, 0], [0, 0]])
        .unwrap();
    assert_eq!(pooled.shape().dimensions(), &[1, 2, 2, 2]);
    assert_eq!(pooled.shape().axis_tags(), input_shape.axis_tags());
    let cumulative = builder.cumulative_sum(input, 2).unwrap();
    assert_eq!(cumulative.shape(), input_shape);
    let nearest = builder.resize_nearest(input, 2, 8).unwrap();
    let linear = builder.resize_linear(input, 2, 3).unwrap();
    let bilinear = builder.resize_bilinear(input, [1, 2], [3, 4]).unwrap();
    let cubic = builder.resize_cubic(input, 2, 3).unwrap();
    let upsampled = builder.upsample_nearest(input, &[2.0]).unwrap();
    assert_eq!(nearest.shape().dimensions(), &[1, 5, 8, 2]);
    assert_eq!(linear.shape().dimensions(), &[1, 5, 3, 2]);
    assert_eq!(bilinear.shape().dimensions(), &[1, 3, 4, 2]);
    assert_eq!(cubic.shape().dimensions(), &[1, 5, 3, 2]);
    // The compact convenience treats trailing rank-minus-two dimensions as
    // spatial. Explicit-axis resize covers layouts such as NHWC.
    assert_eq!(upsampled.shape().dimensions(), &[1, 5, 10, 4]);
    for resized in [nearest, linear, bilinear, cubic, upsampled] {
        assert_eq!(resized.shape().axis_tags(), input_shape.axis_tags());
    }

    let text = builder
        .finish(&[
            convolution,
            pooled,
            cumulative,
            nearest,
            linear,
            bilinear,
            cubic,
            upsampled,
        ])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(text.contains("stablehlo.convolution"), "{text}");
    assert!(text.contains("stablehlo.reduce_window"), "{text}");
    assert!(text.contains("stablehlo.gather"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn window_convolution_and_resize_reject_invalid_geometry_early() {
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[1, 2, 5]).unwrap());
    let kernel = builder.input("kernel", Shape::new(DType::F32, &[4, 2, 3]).unwrap());
    assert!(matches!(
        builder.reduce_window_sum(input, &[1, 1], &[1, 1], &[1, 1], &[1, 1], &[[0, 0]; 2]),
        Err(Error::InvalidWindow(_))
    ));
    assert!(matches!(
        builder.conv1d(input, kernel, 0, [0, 0], 1, 1, 1),
        Err(Error::InvalidWindow(_))
    ));
    assert!(matches!(
        builder.resize_nearest(input, 2, 0),
        Err(Error::InvalidResize(_))
    ));
    assert!(matches!(
        builder.resize_bilinear(input, [2, 2], [4, 4]),
        Err(Error::InvalidResize(_))
    ));
}

#[test]
fn ordering_and_explicit_random_state_are_typed_verified_graph_operations() {
    let mut builder = ProgramBuilder::new();
    let values = builder.input("values", Shape::new(DType::F32, &[2, 5]).unwrap());
    let state_input = builder.input("state", Shape::new(DType::U64, &[2]).unwrap());
    let (ascending, ascending_indices) = builder.sort(values, 1, false, true).unwrap();
    let (descending, descending_indices) = builder.sort(values, 1, true, false).unwrap();
    let argsorted = builder.argsort(values, 1, true, true).unwrap();
    let (top_values, top_indices) = builder.top_k(values, 1, 3, true).unwrap();
    assert_eq!(top_values.shape().dimensions(), &[2, 3]);
    assert_eq!(top_indices.shape().dimensions(), &[2, 3]);

    let state = builder.random_state(state_input).unwrap();
    let random_shape = Shape::new(DType::F32, &[2, 5]).unwrap();
    let (state, uniform) = builder
        .random_uniform(state, random_shape, -2.0, 3.0)
        .unwrap();
    let (state, normal) = builder
        .random_normal(state, random_shape, 1.0, 2.0)
        .unwrap();
    let (state, gumbel) = builder.random_gumbel(state, random_shape).unwrap();
    let state = builder
        .reuse_buffer(state.into_tensor(), state_input)
        .unwrap();
    let text = builder
        .finish(&[
            ascending,
            ascending_indices,
            descending,
            descending_indices,
            argsorted,
            top_values,
            top_indices,
            uniform,
            normal,
            gumbel,
            state,
        ])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(text.contains("stablehlo.sort"), "{text}");
    assert!(text.contains("is_stable = true"), "{text}");
    assert!(text.contains("is_stable = false"), "{text}");
    assert!(text.contains("stablehlo.rng_bit_generator"), "{text}");
    assert!(text.contains("tf.aliasing_output"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn token_sampling_composes_static_and_runtime_controls_with_one_state_chain() {
    let mut builder = ProgramBuilder::new();
    let logits = builder.input("logits", Shape::new(DType::F32, &[2, 8]).unwrap());
    let state_input = builder.input("state", Shape::new(DType::U64, &[2]).unwrap());
    let runtime_top_k = builder.input("top_k", Shape::new(DType::I32, &[]).unwrap());
    let runtime_temperature = builder.input("temperature", Shape::new(DType::F32, &[]).unwrap());
    let runtime_top_p = builder.input("top_p", Shape::new(DType::F32, &[]).unwrap());
    let runtime_min_p = builder.input("min_p", Shape::new(DType::F32, &[]).unwrap());

    let greedy = builder.greedy_tokens(logits, 1).unwrap();
    let state = builder.random_state(state_input).unwrap();
    let (state, static_sample) = builder
        .sample_tokens(logits, state, 1, 6, 0.8, 0.9, 0.05)
        .unwrap();
    let (state, dynamic_sample) = builder
        .sample_tokens_dynamic(
            logits,
            state,
            1,
            runtime_top_k,
            runtime_temperature,
            runtime_top_p,
            runtime_min_p,
            6,
        )
        .unwrap();
    assert_eq!(greedy.shape().dimensions(), &[2]);
    assert_eq!(static_sample.shape().dimensions(), &[2]);
    assert_eq!(dynamic_sample.shape().dimensions(), &[2]);
    let state = builder
        .reuse_buffer(state.into_tensor(), state_input)
        .unwrap();
    let text = builder
        .finish(&[greedy, static_sample, dynamic_sample, state])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(
        text.matches("stablehlo.rng_bit_generator").count() >= 2,
        "{text}"
    );
    assert!(text.contains("stablehlo.sort"), "{text}");
    assert!(text.contains("stablehlo.gather"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn ordering_and_random_generation_reject_invalid_contracts_early() {
    let mut builder = ProgramBuilder::new();
    let values = builder.input("values", Shape::new(DType::F32, &[2, 5]).unwrap());
    assert!(matches!(
        builder.top_k(values, 1, 0, true),
        Err(Error::InvalidSort(_))
    ));
    assert!(matches!(
        builder.top_k(values, 1, 6, true),
        Err(Error::InvalidSort(_))
    ));
    let bad_state = builder.input("bad_state", Shape::new(DType::U64, &[1]).unwrap());
    assert!(matches!(
        builder.random_state(bad_state),
        Err(Error::InvalidRandom(_))
    ));
    let state_input = builder.input("state", Shape::new(DType::U64, &[2]).unwrap());
    let state = builder.random_state(state_input).unwrap();
    assert!(matches!(
        builder.random_uniform(state, Shape::new(DType::F32, &[4]).unwrap(), 1.0, 1.0),
        Err(Error::InvalidRandom(_))
    ));

    let replay_input = builder.input("replay_state", Shape::new(DType::U64, &[2]).unwrap());
    let first = builder.random_state(replay_input).unwrap();
    let replay = builder.random_state(replay_input).unwrap();
    let _ = builder
        .random_bits(first, Shape::new(DType::U32, &[4]).unwrap())
        .unwrap();
    assert!(matches!(
        builder.random_bits(replay, Shape::new(DType::U32, &[4]).unwrap()),
        Err(Error::InvalidRandom(
            "random state has already been consumed"
        ))
    ));

    let state_input = builder.input("sample_state", Shape::new(DType::U64, &[2]).unwrap());
    for (top_k, temperature, top_p, min_p) in [
        (0, 1.0, 1.0, 0.0),
        (6, 1.0, 1.0, 0.0),
        (2, 0.0, 1.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (2, 1.0, 1.0, 1.1),
    ] {
        let state = builder.random_state(state_input).unwrap();
        assert!(matches!(
            builder.sample_tokens(values, state, 1, top_k, temperature, top_p, min_p),
            Err(Error::InvalidSampling(_))
        ));
    }
}

#[test]
fn nd_gather_and_typed_scatter_are_verified_without_public_mlir_configuration() {
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[2, 3, 4]).unwrap());
    let indices = builder.input("indices", Shape::new(DType::I32, &[5, 2]).unwrap());
    let updates = builder.input("updates", Shape::new(DType::F32, &[5, 3]).unwrap());
    let batched_indices = builder.input(
        "batched_indices",
        Shape::new(DType::I32, &[2, 5, 1]).unwrap(),
    );
    let batched_updates = builder.input(
        "batched_updates",
        Shape::new(DType::F32, &[2, 5, 4]).unwrap(),
    );
    let gathered = builder.gather_nd(input, indices, &[0, 2]).unwrap();
    assert_eq!(gathered.shape().dimensions(), &[5, 3]);
    let batched = builder
        .gather_batched_nd(input, batched_indices, 1, &[1])
        .unwrap();
    assert_eq!(batched.shape().dimensions(), &[2, 5, 4]);
    let updated = builder
        .scatter_update(input, indices, updates, &[0, 2])
        .unwrap();
    let added = builder
        .scatter_add(input, indices, updates, &[0, 2])
        .unwrap();
    let multiplied = builder
        .scatter_multiply(input, indices, updates, &[0, 2])
        .unwrap();
    let minimum = builder
        .scatter_minimum(input, indices, updates, &[0, 2])
        .unwrap();
    let maximum = builder
        .scatter_maximum(input, indices, updates, &[0, 2])
        .unwrap();
    let batched_updated = builder
        .scatter_update_batched(input, batched_indices, batched_updates, 1, &[1])
        .unwrap();
    let batched_added = builder
        .scatter_add_batched(input, batched_indices, batched_updates, 1, &[1])
        .unwrap();
    let promised = builder
        .scatter_update_with_promises(input, indices, updates, &[0, 2], true, true)
        .unwrap();
    for result in [updated, added, multiplied, minimum, maximum] {
        assert_eq!(result.shape(), input.shape());
    }

    let text = builder
        .finish(&[
            gathered,
            batched,
            updated,
            added,
            multiplied,
            minimum,
            maximum,
            batched_updated,
            batched_added,
            promised,
        ])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(text.contains("stablehlo.gather"), "{text}");
    assert_eq!(text.matches("\"stablehlo.scatter\"(").count(), 8, "{text}");
    assert!(text.contains("indices_are_sorted = true"), "{text}");
    assert!(text.contains("unique_indices = true"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn nd_indexing_rejects_structural_index_vector_and_update_errors() {
    use nml_types::Partition;

    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[2, 3, 4]).unwrap());
    let wrong_width = builder.input("wrong_width", Shape::new(DType::I32, &[5, 1]).unwrap());
    assert!(matches!(
        builder.gather_nd(input, wrong_width, &[0, 2]),
        Err(Error::InvalidIndexing(_))
    ));
    let sharded_vector = Shape::new(DType::I32, &[5, 2])
        .unwrap()
        .with_partitions(&[
            Partition::Unspecified,
            Partition::Sharded(AxisTag::new(301)),
        ])
        .unwrap();
    let sharded_vector = builder.input("sharded_vector", sharded_vector);
    assert!(matches!(
        builder.gather_nd(input, sharded_vector, &[0, 2]),
        Err(Error::InvalidIndexing(_))
    ));
    let indices = builder.input("indices", Shape::new(DType::I32, &[5, 2]).unwrap());
    let wrong_updates = builder.input("wrong_updates", Shape::new(DType::F32, &[5, 4]).unwrap());
    assert!(matches!(
        builder.scatter_add(input, indices, wrong_updates, &[0, 2]),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn rope_and_ordinary_attention_are_verified_compositions() {
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", Shape::new(DType::F16, &[2, 3, 4, 8]).unwrap());
    let key = builder.input("key", Shape::new(DType::F16, &[2, 5, 2, 8]).unwrap());
    let value = builder.input("value", Shape::new(DType::F16, &[2, 5, 2, 8]).unwrap());
    let query_positions =
        builder.input("query_positions", Shape::new(DType::I32, &[2, 3]).unwrap());
    let key_positions = builder.input("key_positions", Shape::new(DType::I32, &[2, 5]).unwrap());
    let interleaved = builder
        .rope(
            query,
            query_positions,
            RopeOptions {
                base: 10_000.0,
                rotary_dimensions: 8,
                layout: RopeLayout::Interleaved,
                scaling: RopeScaling::Default,
            },
        )
        .unwrap();
    let sequential = builder
        .rope(
            query,
            query_positions,
            RopeOptions {
                base: 500_000.0,
                rotary_dimensions: 4,
                layout: RopeLayout::Sequential,
                scaling: RopeScaling::Linear { factor: 4.0 },
            },
        )
        .unwrap();
    let attention = builder
        .attention(
            interleaved,
            key,
            value,
            query_positions,
            key_positions,
            None,
            AttentionOptions {
                causal: true,
                sliding_window: Some(4),
                scale: None,
            },
        )
        .unwrap();
    assert_eq!(attention.shape(), query.shape());
    let text = builder
        .finish(&[sequential, attention])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(text.contains("stablehlo.sine"));
    assert!(text.contains("stablehlo.cosine"));
    assert!(text.matches("stablehlo.dot_general").count() >= 2);
    assert!(text.contains("stablehlo.reduce"));
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn paged_attention_is_one_bounded_verified_stablehlo_loop() {
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", Shape::new(DType::F16, &[2, 3, 4, 8]).unwrap());
    let key_cache = builder.input("key_cache", Shape::new(DType::F16, &[7, 4, 2, 8]).unwrap());
    let value_cache = builder.input(
        "value_cache",
        Shape::new(DType::F16, &[7, 4, 2, 8]).unwrap(),
    );
    let page_table = builder.input("page_table", Shape::new(DType::I32, &[2, 5]).unwrap());
    let lengths = builder.input("lengths", Shape::new(DType::I32, &[2]).unwrap());
    let positions = builder.input("positions", Shape::new(DType::I32, &[2, 3]).unwrap());
    let output = builder
        .paged_attention(
            query,
            key_cache,
            value_cache,
            page_table,
            lengths,
            positions,
            None,
            AttentionOptions {
                causal: true,
                sliding_window: Some(6),
                scale: None,
            },
        )
        .unwrap();
    assert_eq!(output.shape(), query.shape());
    let text = builder.finish(&[output]).unwrap().stablehlo().unwrap();
    assert_eq!(text.matches("stablehlo.while").count(), 1, "{text}");
    assert_eq!(text.matches("\"stablehlo.gather\"").count(), 2, "{text}");
    assert!(text.matches("stablehlo.reduce").count() >= 2, "{text}");
    assert!(text.contains("tensor<2x2x2x3x4xf32>"), "{text}");
    assert!(
        !text.contains("tensor<2x20x2x8xf32>"),
        "paged lowering materialized a contiguous logical cache: {text}"
    );
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();
}

#[test]
fn cuda_paged_attention_lowers_complete_typed_triton_artifacts() {
    use nml_sharding::Sharding;

    fn lower(query_len: i64) -> String {
        let mut builder = ProgramBuilder::new();
        let query = builder.input(
            "query",
            Shape::new(DType::F16, &[2, query_len, 4, 64]).unwrap(),
        );
        let key_cache = builder.input(
            "key_cache",
            Shape::new(DType::F16, &[7, 16, 2, 64]).unwrap(),
        );
        let value_cache = builder.input(
            "value_cache",
            Shape::new(DType::F16, &[7, 16, 2, 64]).unwrap(),
        );
        let page_table = builder.input("page_table", Shape::new(DType::I32, &[2, 5]).unwrap());
        let lengths = builder.input("lengths", Shape::new(DType::I32, &[2]).unwrap());
        let positions = builder.input(
            "positions",
            Shape::new(DType::I32, &[2, query_len]).unwrap(),
        );
        let output = builder
            .paged_attention(
                query,
                key_cache,
                value_cache,
                page_table,
                lengths,
                positions,
                None,
                AttentionOptions {
                    causal: true,
                    sliding_window: Some(32),
                    scale: None,
                },
            )
            .unwrap();
        let program = builder.finish(&[output]).unwrap();
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(&context, &Sharding::single(), 30, 8, 0)
            .unwrap();
        module.verify().unwrap();
        let text = module.text();
        assert!(
            !module
                .portable_artifact(&nml_mlir::stablehlo_current_version())
                .unwrap()
                .is_empty()
        );
        text
    }

    let prefill = lower(3);
    assert_eq!(
        prefill.matches("__gpu$xla.gpu.triton").count(),
        1,
        "{prefill}"
    );
    assert!(prefill.contains("paged_attention_2d"), "{prefill}");
    assert!(!prefill.contains("stablehlo.while"), "{prefill}");

    let decode = lower(1);
    assert_eq!(
        decode.matches("__gpu$xla.gpu.triton").count(),
        2,
        "{decode}"
    );
    assert!(decode.contains("paged_attention_3d"), "{decode}");
    assert!(
        decode.contains("paged_attention_segment_reduction"),
        "{decode}"
    );
    assert!(decode.contains("tensor<2x4x16x64xf32>"), "{decode}");
    assert!(!decode.contains("stablehlo.while"), "{decode}");
}

#[test]
fn cuda_paged_attention_selects_only_upstream_supported_flash_variants() {
    use nml_sharding::Sharding;

    fn lower_with_options(
        page_size: i64,
        dtype: DType,
        capability: (u16, u16),
        options: AttentionOptions,
    ) -> String {
        let mut builder = ProgramBuilder::new();
        let query = builder.input("query", Shape::new(dtype, &[2, 3, 4, 64]).unwrap());
        let key_cache = builder.input(
            "key_cache",
            Shape::new(dtype, &[7, page_size, 2, 64]).unwrap(),
        );
        let value_cache = builder.input(
            "value_cache",
            Shape::new(dtype, &[7, page_size, 2, 64]).unwrap(),
        );
        let page_table = builder.input("page_table", Shape::new(DType::I32, &[2, 5]).unwrap());
        let lengths = builder.input("lengths", Shape::new(DType::I32, &[2]).unwrap());
        let positions = builder.input("positions", Shape::new(DType::I32, &[2, 3]).unwrap());
        let output = builder
            .paged_attention(
                query,
                key_cache,
                value_cache,
                page_table,
                lengths,
                positions,
                None,
                options,
            )
            .unwrap();
        let program = builder.finish(&[output]).unwrap();
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(
                &context,
                &Sharding::single(),
                80,
                capability.0,
                capability.1,
            )
            .unwrap();
        module.verify().unwrap();
        module.text()
    }

    fn lower(page_size: i64, dtype: DType, capability: (u16, u16)) -> String {
        lower_with_options(
            page_size,
            dtype,
            capability,
            AttentionOptions {
                causal: true,
                sliding_window: Some(32),
                scale: None,
            },
        )
    }

    let sm75 = lower(256, DType::F16, (7, 5));
    assert!(!sm75.contains("nml.flash_attention_2.paged"), "{sm75}");
    assert!(!sm75.contains("__gpu$xla.gpu.triton"), "{sm75}");
    assert!(sm75.contains("stablehlo.while"), "{sm75}");

    // Original upstream FA2 paged KV requires a page size divisible by 256.
    let sm80_small_page = lower(16, DType::F16, (8, 0));
    assert!(sm80_small_page.contains("__gpu$xla.gpu.triton"));
    assert!(!sm80_small_page.contains("nml.flash_attention_2.paged"));
    assert!(!sm80_small_page.contains("stablehlo.case"));

    let sm80 = lower(256, DType::Bf16, (8, 9));
    assert!(sm80.contains("nml.flash_attention_2.paged"), "{sm80}");
    assert!(sm80.contains("__gpu$xla.gpu.triton"), "{sm80}");
    assert!(sm80.contains("stablehlo.case"), "{sm80}");
    assert!(sm80.contains("tensor<2x4x3xf32>"), "{sm80}");

    let sm80_unmasked = lower_with_options(
        256,
        DType::F16,
        (8, 0),
        AttentionOptions {
            causal: false,
            sliding_window: None,
            scale: None,
        },
    );
    assert!(
        sm80_unmasked.contains("nml.flash_attention_2.paged"),
        "{sm80_unmasked}"
    );
    assert!(!sm80_unmasked.contains("stablehlo.case"), "{sm80_unmasked}");
    assert!(
        !sm80_unmasked.contains("__gpu$xla.gpu.triton"),
        "{sm80_unmasked}"
    );

    // FA3 accepts the ordinary page sizes used by NML, but only exact SM90.
    let sm90 = lower(16, DType::F16, (9, 0));
    assert!(sm90.contains("nml.flash_attention_3.paged"), "{sm90}");
    assert!(!sm90.contains("nml.flash_attention_2.paged"), "{sm90}");
    assert!(sm90.contains("stablehlo.case"), "{sm90}");
    assert!(sm90.contains("tensor<1xi32>"), "{sm90}");

    // Architectures outside the exact upstream FlashAttention binaries retain
    // the fused Triton implementation. They do not lose accelerated attention
    // merely because a device-specific FA adapter is unavailable.
    let sm91 = lower(16, DType::F16, (9, 1));
    assert!(!sm91.contains("nml.flash_attention_3.paged"));
    assert!(sm91.contains("__gpu$xla.gpu.triton"), "{sm91}");
    assert!(!sm91.contains("stablehlo.while"), "{sm91}");

    let sm100 = lower(16, DType::Bf16, (10, 0));
    assert!(!sm100.contains("nml.flash_attention_2.paged"), "{sm100}");
    assert!(!sm100.contains("nml.flash_attention_3.paged"), "{sm100}");
    assert!(sm100.contains("__gpu$xla.gpu.triton"), "{sm100}");
    assert!(!sm100.contains("stablehlo.while"), "{sm100}");

    let unsupported_dtype = lower(256, DType::F32, (8, 0));
    assert!(!unsupported_dtype.contains("nml.flash_attention_2.paged"));
    assert!(unsupported_dtype.contains("__gpu$xla.gpu.triton"));
    assert!(!unsupported_dtype.contains("stablehlo.case"));

    // Tiny heads are valid model semantics but cannot fill the retained
    // NVIDIA tt.dot K tile. They stay on exact StableHLO instead of entering a
    // kernel specialization whose physical dot geometry differs from QK.
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", Shape::new(DType::F32, &[1, 2, 2, 4]).unwrap());
    let key_cache = builder.input("key_cache", Shape::new(DType::F32, &[3, 2, 1, 4]).unwrap());
    let value_cache = builder.input(
        "value_cache",
        Shape::new(DType::F32, &[3, 2, 1, 4]).unwrap(),
    );
    let page_table = builder.input("page_table", Shape::new(DType::I32, &[1, 2]).unwrap());
    let lengths = builder.input("lengths", Shape::new(DType::I32, &[1]).unwrap());
    let positions = builder.input("positions", Shape::new(DType::I32, &[1, 2]).unwrap());
    let output = builder
        .paged_attention(
            query,
            key_cache,
            value_cache,
            page_table,
            lengths,
            positions,
            None,
            AttentionOptions::default(),
        )
        .unwrap();
    let program = builder.finish(&[output]).unwrap();
    let context = Context::new();
    let module = program
        .module_with_sharding_cuda(&context, &Sharding::single(), 80, 8, 6)
        .unwrap();
    let tiny_head = module.text();
    assert!(!tiny_head.contains("__gpu$xla.gpu.triton"), "{tiny_head}");
    assert!(tiny_head.contains("stablehlo.while"), "{tiny_head}");
}

#[test]
fn cuda_dense_attention_selects_flash_version_inside_its_exact_capability_contract() {
    use nml_sharding::Sharding;

    fn lower(capability_major: u16, dtype: DType) -> String {
        let mut builder = ProgramBuilder::new();
        let query = builder.input("query", Shape::new(dtype, &[2, 3, 4, 64]).unwrap());
        let key = builder.input("key", Shape::new(dtype, &[2, 5, 2, 64]).unwrap());
        let value = builder.input("value", Shape::new(dtype, &[2, 5, 2, 64]).unwrap());
        let query_positions =
            builder.input("query_positions", Shape::new(DType::I32, &[2, 3]).unwrap());
        let key_positions =
            builder.input("key_positions", Shape::new(DType::I32, &[2, 5]).unwrap());
        let output = builder
            .attention(
                query,
                key,
                value,
                query_positions,
                key_positions,
                None,
                AttentionOptions {
                    causal: true,
                    sliding_window: Some(4),
                    scale: None,
                },
            )
            .unwrap();
        let program = builder.finish(&[output]).unwrap();
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(&context, &Sharding::single(), 80, capability_major, 0)
            .unwrap();
        module.verify().unwrap();
        module.text()
    }

    let sm75 = lower(7, DType::F16);
    assert!(!sm75.contains("nml.flash_attention_2.forward"), "{sm75}");
    assert!(!sm75.contains("nml.flash_attention_3.forward"), "{sm75}");
    assert!(!sm75.contains("stablehlo.case"), "{sm75}");

    let sm80 = lower(8, DType::F16);
    assert!(sm80.contains("nml.flash_attention_2.forward"), "{sm80}");
    assert!(sm80.contains("stablehlo.case"), "{sm80}");
    assert!(sm80.contains("sliding_window = 4 : i32"), "{sm80}");
    assert!(sm80.contains("tensor<2x4x3xf32>"), "{sm80}");

    let unsupported_dtype = lower(8, DType::F32);
    assert!(
        !unsupported_dtype.contains("nml.flash_attention_2.forward"),
        "{unsupported_dtype}"
    );
    assert!(
        !unsupported_dtype.contains("nml.flash_attention_3.forward"),
        "{unsupported_dtype}"
    );

    // Hopper uses its distinct upstream ABI and is never mislabeled as FA2.
    let sm90 = lower(9, DType::F16);
    assert!(!sm90.contains("nml.flash_attention_2.forward"), "{sm90}");
    assert!(sm90.contains("nml.flash_attention_3.forward"), "{sm90}");
    assert!(sm90.contains("stablehlo.case"), "{sm90}");
    assert!(sm90.contains("tensor<1xi32>"), "{sm90}");

    let unbuilt_future_architecture = lower(10, DType::F16);
    assert!(
        !unbuilt_future_architecture.contains("nml.flash_attention_2.forward")
            && !unbuilt_future_architecture.contains("nml.flash_attention_3.forward"),
        "{unbuilt_future_architecture}"
    );
}

#[test]
fn learned_sinks_preserve_dense_flash_dispatch_and_use_the_exact_lse_epilogue() {
    use nml_sharding::Sharding;

    fn lower(capability_major: u16, with_sinks: bool) -> String {
        let mut builder = ProgramBuilder::new();
        let query = builder.input("query", Shape::new(DType::Bf16, &[2, 3, 4, 64]).unwrap());
        let key = builder.input("key", Shape::new(DType::Bf16, &[2, 5, 2, 64]).unwrap());
        let value = builder.input("value", Shape::new(DType::Bf16, &[2, 5, 2, 64]).unwrap());
        let query_positions =
            builder.input("query_positions", Shape::new(DType::I32, &[2, 3]).unwrap());
        let key_positions =
            builder.input("key_positions", Shape::new(DType::I32, &[2, 5]).unwrap());
        let sinks =
            with_sinks.then(|| builder.input("sinks", Shape::new(DType::Bf16, &[4]).unwrap()));
        let output = builder
            .attention(
                query,
                key,
                value,
                query_positions,
                key_positions,
                sinks,
                AttentionOptions {
                    causal: false,
                    sliding_window: None,
                    scale: None,
                },
            )
            .unwrap();
        let program = builder.finish(&[output]).unwrap();
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(&context, &Sharding::single(), 80, capability_major, 0)
            .unwrap();
        module.verify().unwrap();
        module.text()
    }

    for (capability, call) in [
        (8, "nml.flash_attention_2.forward"),
        (9, "nml.flash_attention_3.forward"),
    ] {
        let baseline = lower(capability, false);
        let with_sinks = lower(capability, true);
        assert_eq!(baseline.matches(call).count(), 1, "{baseline}");
        assert_eq!(with_sinks.matches(call).count(), 1, "{with_sinks}");
        assert!(!with_sinks.contains("stablehlo.case"), "{with_sinks}");
        assert!(
            !with_sinks.contains("stablehlo.dot_general"),
            "{with_sinks}"
        );
        assert_eq!(
            with_sinks.matches("stablehlo.logistic").count(),
            1,
            "the Flash LSE must supply the exact sink-normalizer correction:\n{with_sinks}"
        );
    }
}

#[test]
fn learned_sinks_preserve_each_optimized_paged_attention_dispatch() {
    use nml_sharding::Sharding;

    fn lower(
        query_length: i64,
        page_size: i64,
        capability: (u16, u16),
        with_sinks: bool,
        options: AttentionOptions,
    ) -> String {
        let mut builder = ProgramBuilder::new();
        let query = builder.input(
            "query",
            Shape::new(DType::Bf16, &[2, query_length, 4, 64]).unwrap(),
        );
        let key_cache = builder.input(
            "key_cache",
            Shape::new(DType::Bf16, &[7, page_size, 2, 64]).unwrap(),
        );
        let value_cache = builder.input(
            "value_cache",
            Shape::new(DType::Bf16, &[7, page_size, 2, 64]).unwrap(),
        );
        let page_table = builder.input("page_table", Shape::new(DType::I32, &[2, 5]).unwrap());
        let lengths = builder.input("lengths", Shape::new(DType::I32, &[2]).unwrap());
        let positions = builder.input(
            "positions",
            Shape::new(DType::I32, &[2, query_length]).unwrap(),
        );
        let sinks =
            with_sinks.then(|| builder.input("sinks", Shape::new(DType::Bf16, &[4]).unwrap()));
        let output = builder
            .paged_attention(
                query,
                key_cache,
                value_cache,
                page_table,
                lengths,
                positions,
                sinks,
                options,
            )
            .unwrap();
        let program = builder.finish(&[output]).unwrap();
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(
                &context,
                &Sharding::single(),
                80,
                capability.0,
                capability.1,
            )
            .unwrap();
        module.verify().unwrap();
        module.text()
    }

    let masked = AttentionOptions {
        causal: true,
        sliding_window: Some(32),
        scale: None,
    };
    for (query_length, expected_calls) in [(3, 1), (1, 2)] {
        let baseline = lower(query_length, 16, (8, 0), false, masked);
        let with_sinks = lower(query_length, 16, (8, 0), true, masked);
        assert_eq!(
            baseline.matches("__gpu$xla.gpu.triton").count(),
            expected_calls,
            "{baseline}"
        );
        assert_eq!(
            with_sinks.matches("__gpu$xla.gpu.triton").count(),
            expected_calls,
            "{with_sinks}"
        );
        assert!(!with_sinks.contains("stablehlo.while"), "{with_sinks}");
    }

    let unmasked = AttentionOptions {
        causal: false,
        sliding_window: None,
        scale: None,
    };
    for (page_size, capability, call) in [
        (256, (8, 0), "nml.flash_attention_2.paged"),
        (16, (9, 0), "nml.flash_attention_3.paged"),
    ] {
        let baseline = lower(3, page_size, capability, false, unmasked);
        let with_sinks = lower(3, page_size, capability, true, unmasked);
        assert_eq!(baseline.matches(call).count(), 1, "{baseline}");
        assert_eq!(with_sinks.matches(call).count(), 1, "{with_sinks}");
        assert!(!with_sinks.contains("__gpu$xla.gpu.triton"), "{with_sinks}");
        assert!(!with_sinks.contains("stablehlo.while"), "{with_sinks}");
        assert_eq!(
            with_sinks.matches("stablehlo.logistic").count(),
            1,
            "the paged Flash LSE must supply the exact sink-normalizer correction:\n{with_sinks}"
        );
    }
}

#[test]
fn learned_sink_identity_page_geometry_has_no_portable_cuda_branch() {
    use nml_sharding::Sharding;

    fn program(query_length: i64, capacity: i64) -> nml_ir::Program {
        let mut builder = ProgramBuilder::new();
        let query = builder.input(
            "query",
            Shape::new(DType::Bf16, &[1, query_length, 64, 64]).unwrap(),
        );
        let cache_shape = Shape::new(DType::Bf16, &[1, capacity, 8, 64]).unwrap();
        let key_cache = builder.input("key_cache", cache_shape);
        let value_cache = builder.input("value_cache", cache_shape);
        let page_table = builder
            .iota(Shape::new(DType::I32, &[1, 1]).unwrap(), 1)
            .unwrap();
        let lengths = builder.input("lengths", Shape::new(DType::I32, &[1]).unwrap());
        let positions = builder.input(
            "positions",
            Shape::new(DType::I32, &[1, query_length]).unwrap(),
        );
        let sinks = builder.input("sinks", Shape::new(DType::Bf16, &[64]).unwrap());
        let output = builder
            .paged_attention(
                query,
                key_cache,
                value_cache,
                page_table,
                lengths,
                positions,
                Some(sinks),
                AttentionOptions {
                    causal: true,
                    sliding_window: Some(128),
                    scale: None,
                },
            )
            .unwrap();
        builder.finish(&[output]).unwrap()
    }

    fn lower(program: &nml_ir::Program, capability: (u16, u16)) -> String {
        let context = Context::new();
        let module = program
            .module_with_sharding_cuda(
                &context,
                &Sharding::single(),
                108,
                capability.0,
                capability.1,
            )
            .unwrap();
        module.verify().unwrap();
        module.text()
    }

    // This representative learned-sink geometry uses grouped-query attention
    // and a 128-token local window. A bounded caller may view contiguous
    // donated cache storage as one identity page.
    let small = program(1, 8);
    for capability in [(8, 0), (9, 0), (10, 0)] {
        let text = lower(&small, capability);
        assert!(!text.contains("stablehlo.while"), "{text}");
        assert!(text.contains("__gpu$xla.gpu.triton"), "{text}");
        if capability == (9, 0) {
            assert!(text.contains("nml.flash_attention_3.paged"), "{text}");
        }
    }

    // A finite family whose single identity page is divisible by 256 admits
    // FA2. Arbitrary positions select its Triton alternate, never the portable
    // StableHLO page loop.
    let ampere = lower(&program(3, 256), (8, 0));
    assert!(ampere.contains("nml.flash_attention_2.paged"), "{ampere}");
    assert!(ampere.contains("__gpu$xla.gpu.triton"), "{ampere}");
    assert!(!ampere.contains("stablehlo.while"), "{ampere}");
}

#[test]
fn ordinary_attention_preserves_shardy_head_and_batch_placement() {
    use nml_sharding::Sharding;
    use nml_types::{AxisTag, Partition};

    let data = AxisTag::new(31);
    let model = AxisTag::new(32);
    let sequence = AxisTag::new(33);
    let head_dim = AxisTag::new(34);
    let query_shape = Shape::new(DType::F32, &[2, 3, 4, 8])
        .unwrap()
        .with_axis_tags(&[data, sequence, model, head_dim])
        .unwrap()
        .with_partitions(&[
            Partition::Sharded(data),
            Partition::Replicated,
            Partition::Sharded(model),
            Partition::Replicated,
        ])
        .unwrap();
    let key_shape = Shape::new(DType::F32, &[2, 5, 2, 8])
        .unwrap()
        .with_axis_tags(&[data, sequence, model, head_dim])
        .unwrap()
        .with_partitions(&[
            Partition::Sharded(data),
            Partition::Replicated,
            Partition::Sharded(model),
            Partition::Replicated,
        ])
        .unwrap();
    let position_shape = Shape::new(DType::I32, &[2, 3])
        .unwrap()
        .with_axis_tags(&[data, sequence])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Replicated])
        .unwrap();
    let key_position_shape = Shape::new(DType::I32, &[2, 5])
        .unwrap()
        .with_axis_tags(&[data, sequence])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Replicated])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", query_shape);
    let key = builder.input("key", key_shape);
    let value = builder.input("value", key_shape);
    let query_positions = builder.input("query_positions", position_shape);
    let key_positions = builder.input("key_positions", key_position_shape);
    let output = builder
        .attention(
            query,
            key,
            value,
            query_positions,
            key_positions,
            None,
            AttentionOptions::default(),
        )
        .unwrap();
    assert_eq!(output.shape().partitions(), query_shape.partitions());
    let program = builder.finish(&[output]).unwrap();
    let mesh = Sharding::mesh(&[(data, 2), (model, 2)]).unwrap();
    let text = program.stablehlo_with_sharding(&mesh).unwrap();
    assert!(text.contains("sdy.mesh"), "{text}");
    assert!(text.contains("sdy.sharding_constraint"), "{text}");
}

#[test]
fn paged_attention_preserves_shardy_head_and_batch_placement() {
    use nml_sharding::Sharding;
    use nml_types::{AxisTag, Partition};

    let data = AxisTag::new(41);
    let model = AxisTag::new(42);
    let sequence = AxisTag::new(43);
    let page = AxisTag::new(44);
    let head_dim = AxisTag::new(45);
    let query_shape = Shape::new(DType::F32, &[2, 3, 4, 8])
        .unwrap()
        .with_axis_tags(&[data, sequence, model, head_dim])
        .unwrap()
        .with_partitions(&[
            Partition::Sharded(data),
            Partition::Replicated,
            Partition::Sharded(model),
            Partition::Replicated,
        ])
        .unwrap();
    let cache_shape = Shape::new(DType::F32, &[4, 2, 2, 8])
        .unwrap()
        .with_axis_tags(&[page, sequence, model, head_dim])
        .unwrap()
        .with_partitions(&[
            Partition::Replicated,
            Partition::Replicated,
            Partition::Sharded(model),
            Partition::Replicated,
        ])
        .unwrap();
    let page_table_shape = Shape::new(DType::I32, &[2, 2])
        .unwrap()
        .with_axis_tags(&[data, page])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Replicated])
        .unwrap();
    let length_shape = Shape::new(DType::I32, &[2])
        .unwrap()
        .with_axis_tags(&[data])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data)])
        .unwrap();
    let position_shape = Shape::new(DType::I32, &[2, 3])
        .unwrap()
        .with_axis_tags(&[data, sequence])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Replicated])
        .unwrap();

    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", query_shape);
    let key_cache = builder.input("key_cache", cache_shape);
    let value_cache = builder.input("value_cache", cache_shape);
    let page_table = builder.input("page_table", page_table_shape);
    let sequence_lengths = builder.input("sequence_lengths", length_shape);
    let query_positions = builder.input("query_positions", position_shape);
    let output = builder
        .paged_attention(
            query,
            key_cache,
            value_cache,
            page_table,
            sequence_lengths,
            query_positions,
            None,
            AttentionOptions::default(),
        )
        .unwrap();
    assert_eq!(output.shape().partitions(), query_shape.partitions());

    let program = builder.finish(&[output]).unwrap();
    let mesh = Sharding::mesh(&[(data, 2), (model, 2)]).unwrap();
    let text = program.stablehlo_with_sharding(&mesh).unwrap();
    assert!(text.contains("sdy.mesh"), "{text}");
    assert!(text.contains("sdy.sharding_constraint"), "{text}");
}

#[test]
fn attention_geometry_is_rejected_before_mlir_construction() {
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", Shape::new(DType::F32, &[1, 2, 3, 8]).unwrap());
    let key = builder.input("key", Shape::new(DType::F32, &[1, 2, 2, 8]).unwrap());
    let value = builder.input("value", Shape::new(DType::F32, &[1, 2, 2, 8]).unwrap());
    let positions = builder.input("positions", Shape::new(DType::I32, &[1, 2]).unwrap());
    assert!(matches!(
        builder.attention(
            query,
            key,
            value,
            positions,
            positions,
            None,
            AttentionOptions::default(),
        ),
        Err(Error::InvalidAttention(_))
    ));
    assert!(matches!(
        builder.rope(
            query,
            positions,
            RopeOptions {
                base: 10_000.0,
                rotary_dimensions: 7,
                layout: RopeLayout::Interleaved,
                scaling: RopeScaling::Default,
            },
        ),
        Err(Error::InvalidRope(_))
    ));

    let mut zero_heads = ProgramBuilder::new();
    let query = zero_heads.input("query", Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap());
    let key = zero_heads.input("key", Shape::new(DType::F32, &[1, 1, 0, 4]).unwrap());
    let value = zero_heads.input("value", Shape::new(DType::F32, &[1, 1, 0, 4]).unwrap());
    let positions = zero_heads.input("positions", Shape::new(DType::I32, &[1, 1]).unwrap());
    assert!(matches!(
        zero_heads.attention(
            query,
            key,
            value,
            positions,
            positions,
            None,
            AttentionOptions::default(),
        ),
        Err(Error::InvalidAttention(_))
    ));

    let mut invalid_scale = ProgramBuilder::new();
    let query = invalid_scale.input("query", Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap());
    let key = invalid_scale.input("key", Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap());
    let value = invalid_scale.input("value", Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap());
    let positions = invalid_scale.input("positions", Shape::new(DType::I32, &[1, 1]).unwrap());
    assert!(matches!(
        invalid_scale.attention(
            query,
            key,
            value,
            positions,
            positions,
            None,
            AttentionOptions {
                scale: Some(f64::MAX),
                ..AttentionOptions::default()
            },
        ),
        Err(Error::InvalidAttention(_))
    ));

    let mut invalid_window = ProgramBuilder::new();
    let query = invalid_window.input("query", Shape::new(DType::F32, &[1, 1, 2, 4]).unwrap());
    let key = invalid_window.input("key", Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap());
    let value = invalid_window.input("value", Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap());
    let positions = invalid_window.input("positions", Shape::new(DType::I32, &[1, 1]).unwrap());
    assert!(matches!(
        invalid_window.attention(
            query,
            key,
            value,
            positions,
            positions,
            None,
            AttentionOptions {
                sliding_window: Some(0),
                ..AttentionOptions::default()
            },
        ),
        Err(Error::InvalidAttention(_))
    ));

    let key_cache =
        invalid_window.input("key_cache", Shape::new(DType::F32, &[2, 16, 1, 4]).unwrap());
    let value_cache = invalid_window.input(
        "value_cache",
        Shape::new(DType::F32, &[2, 16, 1, 4]).unwrap(),
    );
    let page_table = invalid_window.input("page_table", Shape::new(DType::I32, &[1, 1]).unwrap());
    let lengths = invalid_window.input("lengths", Shape::new(DType::I32, &[1]).unwrap());
    assert!(matches!(
        invalid_window.paged_attention(
            query,
            key_cache,
            value_cache,
            page_table,
            lengths,
            positions,
            None,
            AttentionOptions {
                sliding_window: Some(0),
                ..AttentionOptions::default()
            },
        ),
        Err(Error::InvalidAttention(_))
    ));
}

#[test]
fn validation_rejects_invalid_graphs_before_mlir_construction() {
    let mut left_builder = ProgramBuilder::new();
    let foreign = left_builder.input("foreign", Shape::new(DType::F32, &[2, 2]).unwrap());
    let mut right_builder = ProgramBuilder::new();
    let local = right_builder.input("local", Shape::new(DType::F32, &[2, 2]).unwrap());
    assert!(matches!(
        right_builder.matmul(foreign, local),
        Err(Error::ForeignTensor)
    ));

    let mut builder = ProgramBuilder::new();
    let rank_one = builder.input("rank_one", Shape::new(DType::F32, &[2]).unwrap());
    let matrix = builder.input("matrix", Shape::new(DType::F32, &[2, 2]).unwrap());
    assert!(matches!(
        builder.matmul(rank_one, matrix),
        Err(Error::RankMismatch { .. })
    ));

    let integer = builder.input("integer", Shape::new(DType::I32, &[2, 2]).unwrap());
    assert!(matches!(
        builder.matmul(matrix, integer),
        Err(Error::DTypeMismatch { .. })
    ));

    assert!(matches!(
        builder.dot_general(matrix, matrix, &[], &[], &[2], &[0]),
        Err(Error::AxisOutOfBounds { .. })
    ));
    assert!(matches!(
        builder.dot_general(matrix, matrix, &[], &[], &[1, 1], &[0, 1]),
        Err(Error::DuplicateAxis { .. })
    ));
    assert!(matches!(
        builder.dot_general(matrix, matrix, &[], &[], &[1], &[]),
        Err(Error::AxisCountMismatch)
    ));
    assert!(matches!(
        builder.dot_general(matrix, matrix, &[0], &[0], &[0], &[1]),
        Err(Error::DuplicateAxis { .. })
    ));

    let empty = ProgramBuilder::new();
    assert!(matches!(empty.finish(&[]), Err(Error::NoOutputs)));
}

#[test]
fn unsupported_physical_layout_is_rejected_before_mlir_construction() {
    let column_major = Shape::new(DType::F32, &[2, 2])
        .unwrap()
        .with_layout(Layout::from_minor_to_major(&[0, 1]).unwrap())
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("column_major", column_major);
    assert!(matches!(
        builder.finish(&[input]),
        Err(Error::UnsupportedLayout { .. })
    ));
}

#[test]
fn complex_and_fft_dtype_contracts_fail_before_mlir_construction() {
    let mut builder = ProgramBuilder::new();
    let integer = builder.input("integer", Shape::new(DType::I32, &[8]).unwrap());
    assert!(matches!(
        builder.complex(integer, integer),
        Err(Error::UnsupportedComplexInput(DType::I32))
    ));
    assert!(matches!(
        builder.real(integer),
        Err(Error::ExpectedComplex(DType::I32))
    ));
    assert!(matches!(
        builder.fft(integer, FftType::Rfft, &[8]),
        Err(Error::InvalidFft { .. })
    ));

    let signal = builder.input("signal", Shape::new(DType::F32, &[8]).unwrap());
    assert!(matches!(
        builder.fft(signal, FftType::Rfft, &[]),
        Err(Error::InvalidFft { .. })
    ));
    assert!(matches!(
        builder.fft(signal, FftType::Rfft, &[7]),
        Err(Error::InvalidFft { .. })
    ));

    let double = builder.input("double", Shape::new(DType::F64, &[8]).unwrap());
    assert!(matches!(
        builder.complex(signal, double),
        Err(Error::DTypeMismatch { .. })
    ));

    let shorter = builder.input("shorter", Shape::new(DType::F32, &[4]).unwrap());
    assert!(matches!(
        builder.complex(signal, shorter),
        Err(Error::ShapeMismatch { .. })
    ));

    let tagged = Shape::new(DType::F32, &[8])
        .unwrap()
        .with_axis_tags(&[nml_types::AxisTag::new(1)])
        .unwrap();
    let tagged = builder.input("tagged", tagged);
    assert!(matches!(
        builder.complex(signal, tagged),
        Err(Error::MetadataMismatch { .. })
    ));
}

#[test]
fn complex_real_and_imaginary_round_trip_verifies() {
    let mut builder = ProgramBuilder::new();
    let real = builder.input("real", Shape::new(DType::F32, &[4]).unwrap());
    let imaginary = builder.input("imaginary", Shape::new(DType::F32, &[4]).unwrap());
    let complex = builder.complex(real, imaginary).unwrap();
    assert_eq!(complex.shape().dtype(), DType::C64);
    let real_result = builder.real(complex).unwrap();
    let imaginary_result = builder.imaginary(complex).unwrap();
    let program = builder.finish(&[real_result, imaginary_result]).unwrap();
    let context = Context::new();
    program.module(&context).unwrap();
}

#[test]
fn type_changing_operations_preserve_tensor_metadata() {
    use nml_types::{AxisTag, Partition};

    let batch = AxisTag::new(1);
    let feature = AxisTag::new(2);
    let shape = Shape::new(DType::F32, &[2, 8])
        .unwrap()
        .with_axis_tags(&[batch, feature])
        .unwrap()
        .with_partitions(&[Partition::Replicated, Partition::Sharded(feature)])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let real = builder.input("real", shape);
    let imaginary = builder.input("imaginary", shape);
    let complex = builder.complex(real, imaginary).unwrap();
    assert_eq!(complex.shape().axis_tags(), shape.axis_tags());
    assert_eq!(complex.shape().partitions(), shape.partitions());
    let real = builder.real(complex).unwrap();
    assert_eq!(real.shape().axis_tags(), shape.axis_tags());
    assert_eq!(real.shape().partitions(), shape.partitions());
    let spectrum = builder.fft(real, FftType::Rfft, &[8]).unwrap();
    assert_eq!(spectrum.shape().axis_tags(), shape.axis_tags());
    assert_eq!(spectrum.shape().partitions(), shape.partitions());
}

#[test]
fn dot_general_derives_output_metadata_from_retained_axes() {
    use nml_types::{AxisTag, Partition};

    let batch = AxisTag::new(1);
    let row = AxisTag::new(2);
    let contract = AxisTag::new(3);
    let column = AxisTag::new(4);
    let left_shape = Shape::new(DType::F32, &[2, 3, 5])
        .unwrap()
        .with_axis_tags(&[batch, row, contract])
        .unwrap()
        .with_partitions(&[
            Partition::Replicated,
            Partition::Sharded(row),
            Partition::Unspecified,
        ])
        .unwrap();
    let right_shape = Shape::new(DType::F32, &[2, 5, 7])
        .unwrap()
        .with_axis_tags(&[batch, contract, column])
        .unwrap()
        .with_partitions(&[
            Partition::Replicated,
            Partition::Unspecified,
            Partition::Sharded(column),
        ])
        .unwrap();

    let mut builder = ProgramBuilder::new();
    let left = builder.input("left", left_shape);
    let right = builder.input("right", right_shape);
    let result = builder
        .dot_general(left, right, &[0], &[0], &[2], &[1])
        .unwrap();
    assert_eq!(result.shape().dimensions(), &[2, 3, 7]);
    assert_eq!(result.shape().axis_tags(), &[batch, row, column]);
    assert_eq!(
        result.shape().partitions(),
        &[
            Partition::Replicated,
            Partition::Sharded(row),
            Partition::Sharded(column),
        ]
    );

    let mismatched_batch = Shape::new(DType::F32, &[2, 5, 7])
        .unwrap()
        .with_axis_tags(&[AxisTag::new(99), contract, column])
        .unwrap();
    let mismatched_batch = builder.input("mismatched_batch", mismatched_batch);
    assert!(matches!(
        builder.dot_general(left, mismatched_batch, &[0], &[0], &[2], &[1]),
        Err(Error::MetadataMismatch { .. })
    ));
}

#[test]
fn fft_builders_are_typed_and_verified_without_claiming_execution_support() {
    let mut builder = ProgramBuilder::new();
    let signal = builder.input("signal", Shape::new(DType::F32, &[2, 8]).unwrap());
    let spectrum = builder.fft(signal, FftType::Rfft, &[8]).unwrap();
    assert_eq!(spectrum.shape().dtype(), DType::C64);
    assert_eq!(spectrum.shape().dimensions(), &[2, 5]);
    let reconstructed = builder.fft(spectrum, FftType::Irfft, &[8]).unwrap();
    assert_eq!(reconstructed.shape().dtype(), DType::F32);
    assert_eq!(reconstructed.shape().dimensions(), &[2, 8]);
    let program = builder.finish(&[reconstructed]).unwrap();
    let stablehlo = program.stablehlo().unwrap();
    assert!(
        stablehlo.contains("type = RFFT, length = [8]"),
        "{stablehlo}"
    );
    assert!(
        stablehlo.contains("type = IRFFT, length = [8]"),
        "{stablehlo}"
    );
    let context = Context::new();
    program.module(&context).unwrap();
}

#[test]
fn algebra_shape_transforms_and_activations_form_one_verified_graph() {
    use nml_tensor::Slice;
    use nml_types::{AxisTag, Partition};

    let batch = AxisTag::new(10);
    let sequence = AxisTag::new(11);
    let heads = AxisTag::new(12);
    let head_dim = AxisTag::new(13);
    let shape = Shape::new(DType::F32, &[2, 3, 4, 8])
        .unwrap()
        .with_axis_tags(&[batch, sequence, heads, head_dim])
        .unwrap()
        .with_partitions(&[
            Partition::Unspecified,
            Partition::Unspecified,
            Partition::Unspecified,
            Partition::Unspecified,
        ])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let scalar = builder.scalar(2.0f32).unwrap();
    let constant_shape = Shape::new(DType::F32, &[2]).unwrap();
    let constant_values = [1.0f32, -1.0];
    let constant = builder
        .constant(&Slice::from_typed(constant_shape, &constant_values).unwrap())
        .unwrap();
    assert_eq!(constant.shape(), constant_shape);

    let multiplied = builder.multiply(input, scalar).unwrap();
    let subtracted = builder.subtract(multiplied, scalar).unwrap();
    let divided = builder.divide(subtracted, scalar).unwrap();
    let predicate = builder.greater(divided, scalar).unwrap();
    let negated = builder.negate(divided).unwrap();
    let selected = builder.select(predicate, divided, negated).unwrap();
    let converted = builder.convert(selected, DType::F16).unwrap();
    let converted = builder.convert(converted, DType::F32).unwrap();
    let flattened_shape = Shape::new(DType::F32, &[6, 32])
        .unwrap()
        .with_axis_tags(&[batch, head_dim])
        .unwrap()
        .with_partitions(&[Partition::Unspecified, Partition::Unspecified])
        .unwrap();
    let flattened = builder.reshape(converted, flattened_shape).unwrap();
    let transposed = builder.transpose(flattened, &[1, 0]).unwrap();
    assert_eq!(transposed.shape().dimensions(), &[32, 6]);
    assert_eq!(transposed.shape().axis_tags(), &[head_dim, batch]);
    let relu = builder.relu(transposed).unwrap();
    let sigmoid = builder.sigmoid(relu).unwrap();
    let silu = builder.silu(sigmoid).unwrap();
    let gelu = builder.gelu(silu).unwrap();
    let quick = builder.quick_gelu(gelu).unwrap();
    let output = builder.leaky_relu(quick, 0.01).unwrap();
    let program = builder.finish(&[output]).unwrap();
    let text = program.stablehlo().unwrap();
    for operation in [
        "stablehlo.constant",
        "stablehlo.multiply",
        "stablehlo.subtract",
        "stablehlo.divide",
        "stablehlo.compare",
        "stablehlo.select",
        "stablehlo.convert",
        "stablehlo.reshape",
        "stablehlo.transpose",
        "stablehlo.logistic",
        "stablehlo.tanh",
    ] {
        assert!(text.contains(operation), "missing {operation} in {text}");
    }
}

#[test]
fn logical_mesh_lowers_shape_partitions_to_shardy() {
    use nml_sharding::Sharding;
    use nml_types::{AxisTag, Partition};

    let data = AxisTag::new(1);
    let model = AxisTag::new(2);
    let shape = Shape::new(DType::F32, &[8, 16])
        .unwrap()
        .with_axis_tags(&[data, model])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Replicated])
        .unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let constrained = builder
        .with_partitions(input, &[Partition::Sharded(data), Partition::Replicated])
        .unwrap();
    let program = builder.finish(&[constrained]).unwrap();
    let mesh = Sharding::mesh(&[(data, 2), (model, 2)]).unwrap();
    let context = Context::new();
    let module = program.module_with_sharding(&context, &mesh).unwrap();
    let text = module.text();
    assert!(text.contains("sdy.mesh"), "{text}");
    assert!(text.contains("\"axis_1\"=2"), "{text}");
    assert!(text.contains("\"axis_2\"=2"), "{text}");
    assert!(text.contains("sdy.sharding_constraint"), "{text}");
    assert!(text.contains("sdy.sharding"), "{text}");

    let absent = AxisTag::new(3);
    let invalid_shape = Shape::new(DType::F32, &[8])
        .unwrap()
        .with_partitions(&[Partition::Sharded(absent)])
        .unwrap();
    let mut invalid = ProgramBuilder::new();
    let input = invalid.input("input", invalid_shape);
    let invalid = invalid.finish(&[input]).unwrap();
    assert!(matches!(
        invalid.module_with_sharding(&context, &mesh),
        Err(nml_ir::Error::Sharding(
            nml_sharding::Error::MissingAxis(tag)
        )) if tag == absent
    ));
}

#[test]
fn typed_collectives_use_explicit_shardy_meshes_and_replicated_results() {
    use nml_sharding::Sharding;

    let mesh_axis = AxisTag::new(211);
    let shape = Shape::new(DType::F32, &[2, 4]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", shape);
    let sum = builder.all_reduce_sum(input).unwrap();
    let maximum = builder.all_reduce_max(input).unwrap();
    let minimum = builder.all_reduce_min(input).unwrap();
    assert!(
        sum.shape()
            .partitions()
            .iter()
            .all(|partition| *partition == Partition::Replicated)
    );
    let program = builder.finish(&[sum, maximum, minimum]).unwrap();
    let mesh = Sharding::mesh(&[(mesh_axis, 2)]).unwrap();
    let text = program.stablehlo_with_sharding(&mesh).unwrap();
    assert_eq!(text.matches("stablehlo.all_reduce").count(), 3, "{text}");
    assert!(text.contains("dense<[[0, 1]]>"), "{text}");
    assert!(text.contains("use_global_device_ids"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();

    let sharded_shape = shape
        .with_partitions(&[Partition::Sharded(mesh_axis), Partition::Replicated])
        .unwrap();
    let mut invalid = ProgramBuilder::new();
    let sharded = invalid.input("sharded", sharded_shape);
    assert!(matches!(
        invalid.all_reduce_sum(sharded),
        Err(Error::InvalidCollective(_))
    ));
    assert!(matches!(
        program.stablehlo_with_sharding(&Sharding::replicated()),
        Err(Error::InvalidCollective(_))
    ));
}

#[test]
fn portable_moe_routing_is_a_typed_backend_independent_graph() {
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input("hidden", Shape::new(DType::F32, &[3, 4]).unwrap());
    let router = builder.input("router", Shape::new(DType::F32, &[3, 3]).unwrap());
    let gate_up = parameter("gate_up", Shape::new(DType::F32, &[3, 10, 4]).unwrap());
    let down = parameter("down", Shape::new(DType::F32, &[3, 4, 5]).unwrap());
    let swiglu = builder
        .moe_swiglu(hidden, router, &gate_up, &down, 2)
        .unwrap();
    let geglu = builder
        .moe_geglu(hidden, router, &gate_up, &down, 2)
        .unwrap();
    let reglu = builder
        .moe_reglu(hidden, router, &gate_up, &down, 2)
        .unwrap();
    assert_eq!(swiglu.shape(), hidden.shape());
    assert_eq!(geglu.shape(), hidden.shape());
    assert_eq!(reglu.shape(), hidden.shape());
    let text = builder
        .finish(&[swiglu, geglu, reglu])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert!(text.contains("stablehlo.sort"), "{text}");
    assert!(text.contains("stablehlo.dot_general"), "{text}");
    assert!(text.contains("stablehlo.logistic"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();

    let mut invalid = ProgramBuilder::new();
    let hidden = invalid.input("hidden", Shape::new(DType::F32, &[3, 4]).unwrap());
    let router = invalid.input("router", Shape::new(DType::F32, &[3, 3]).unwrap());
    let gate_up = parameter("gate_up", Shape::new(DType::F32, &[3, 9, 4]).unwrap());
    let down = parameter("down", Shape::new(DType::F32, &[3, 4, 5]).unwrap());
    assert!(matches!(
        invalid.moe_swiglu(hidden, router, &gate_up, &down, 2),
        Err(Error::InvalidMoe(_))
    ));
}

#[test]
fn gated_delta_net_step_and_sequence_share_one_typed_recurrence() {
    let mut builder = ProgramBuilder::new();
    let queries = builder.input("queries", Shape::new(DType::F32, &[2, 2, 3]).unwrap());
    let keys = builder.input("keys", Shape::new(DType::F32, &[2, 2, 3]).unwrap());
    let values = builder.input("values", Shape::new(DType::F32, &[2, 2, 4]).unwrap());
    let alphas = builder.input("alphas", Shape::new(DType::F32, &[2, 2]).unwrap());
    let betas = builder.input("betas", Shape::new(DType::F32, &[2, 2]).unwrap());
    let state = builder.input("state", Shape::new(DType::F32, &[2, 4, 3]).unwrap());
    let (outputs, final_state) = builder
        .gated_delta_net(queries, keys, values, alphas, betas, state)
        .unwrap();
    assert_eq!(outputs.shape(), values.shape());
    assert_eq!(final_state.shape(), state.shape());
    let text = builder
        .finish(&[outputs, final_state])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert_eq!(text.matches("stablehlo.while").count(), 1, "{text}");
    assert_eq!(text.matches("stablehlo.dot_general").count(), 2, "{text}");
    assert!(text.contains("stablehlo.dynamic_slice"), "{text}");
    assert!(text.contains("stablehlo.dynamic_update_slice"), "{text}");
    Context::new()
        .parse_module(&text)
        .unwrap()
        .verify()
        .unwrap();

    // The recurrence is represented once regardless of the static sequence
    // length; long contexts must not produce proportionally larger graphs.
    let mut long = ProgramBuilder::new();
    let queries = long.input("queries", Shape::new(DType::F32, &[64, 2, 3]).unwrap());
    let keys = long.input("keys", Shape::new(DType::F32, &[64, 2, 3]).unwrap());
    let values = long.input("values", Shape::new(DType::F32, &[64, 2, 4]).unwrap());
    let alphas = long.input("alphas", Shape::new(DType::F32, &[64, 2]).unwrap());
    let betas = long.input("betas", Shape::new(DType::F32, &[64, 2]).unwrap());
    let state = long.input("state", Shape::new(DType::F32, &[2, 4, 3]).unwrap());
    let (outputs, final_state) = long
        .gated_delta_net(queries, keys, values, alphas, betas, state)
        .unwrap();
    let long_text = long
        .finish(&[outputs, final_state])
        .unwrap()
        .stablehlo()
        .unwrap();
    assert_eq!(long_text.matches("stablehlo.while").count(), 1);
    assert_eq!(long_text.matches("stablehlo.dot_general").count(), 2);

    let mut invalid = ProgramBuilder::new();
    let state = invalid.input("state", Shape::new(DType::F32, &[2, 4, 3]).unwrap());
    let query = invalid.input("query", Shape::new(DType::F32, &[2, 3]).unwrap());
    let key = invalid.input("key", Shape::new(DType::F32, &[2, 3]).unwrap());
    let value = invalid.input("value", Shape::new(DType::F32, &[2, 5]).unwrap());
    let alpha = invalid.input("alpha", Shape::new(DType::F32, &[2]).unwrap());
    let beta = invalid.input("beta", Shape::new(DType::F32, &[2]).unwrap());
    assert!(matches!(
        invalid.gated_delta_net_step(state, query, key, value, alpha, beta),
        Err(Error::InvalidStateSpace(_))
    ));

    let mut invalid_metadata = ProgramBuilder::new();
    let head = AxisTag::new(201);
    let value_axis = AxisTag::new(202);
    let key_axis = AxisTag::new(203);
    let wrong_value_axis = AxisTag::new(204);
    let state = invalid_metadata.input(
        "state",
        Shape::new(DType::F32, &[2, 4, 3])
            .unwrap()
            .with_axis_tags(&[head, value_axis, key_axis])
            .unwrap(),
    );
    let query = invalid_metadata.input(
        "query",
        Shape::new(DType::F32, &[2, 3])
            .unwrap()
            .with_axis_tags(&[head, key_axis])
            .unwrap(),
    );
    let key = invalid_metadata.input("key", query.shape());
    let value = invalid_metadata.input(
        "value",
        Shape::new(DType::F32, &[2, 4])
            .unwrap()
            .with_axis_tags(&[head, wrong_value_axis])
            .unwrap(),
    );
    let gate_shape = Shape::new(DType::F32, &[2])
        .unwrap()
        .with_axis_tags(&[head])
        .unwrap();
    let alpha = invalid_metadata.input("alpha", gate_shape);
    let beta = invalid_metadata.input("beta", gate_shape);
    assert!(matches!(
        invalid_metadata.gated_delta_net_step(state, query, key, value, alpha, beta),
        Err(Error::InvalidStateSpace(message)) if message.contains("key/value-axis metadata")
    ));
}

#[test]
fn ampere_moe_dispatch_is_private_and_capability_selected() {
    use nml_sharding::Sharding;

    fn program(dtype: DType) -> nml_ir::Program {
        let mut builder = ProgramBuilder::new();
        let hidden = builder.input("hidden", Shape::new(dtype, &[4, 32]).unwrap());
        let router = builder.input("router", Shape::new(DType::F32, &[4, 4]).unwrap());
        let gate_up = parameter("gate_up", Shape::new(dtype, &[4, 64, 32]).unwrap());
        let down = parameter("down", Shape::new(dtype, &[4, 32, 32]).unwrap());
        let output = builder
            .moe_swiglu(hidden, router, &gate_up, &down, 2)
            .unwrap();
        builder.finish(&[output]).unwrap()
    }

    let context = Context::new();
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let cuda = program(dtype)
            .module_with_sharding_cuda(&context, &Sharding::single(), 108, 8, 0)
            .unwrap()
            .text();
        assert_eq!(cuda.matches("__gpu$xla.gpu.triton").count(), 2, "{cuda}");
        assert!(cuda.contains("moe_grouped_gate_up"), "{cuda}");
        assert!(cuda.contains("moe_grouped_down"), "{cuda}");
    }

    let program = program(DType::F16);
    let sm75 = program
        .module_with_sharding_cuda(&context, &Sharding::single(), 24, 7, 5)
        .unwrap()
        .text();
    assert!(!sm75.contains("__gpu$xla.gpu.triton"), "{sm75}");
    assert!(sm75.contains("stablehlo.dot_general"), "{sm75}");

    let portable = program.stablehlo().unwrap();
    assert!(!portable.contains("__gpu$xla.gpu.triton"), "{portable}");
    assert!(portable.contains("stablehlo.dot_general"), "{portable}");
}

#[test]
fn expert_parallel_moe_derives_local_shards_inside_private_manual_computation() {
    use nml_sharding::Sharding;

    let data_axis = AxisTag::new(211);
    let expert_axis = AxisTag::new(212);
    let expert_partition = [
        Partition::Sharded(expert_axis),
        Partition::Replicated,
        Partition::Replicated,
    ];
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input(
        "hidden",
        Shape::new(DType::F16, &[4, 32])
            .unwrap()
            .with_partitions(&[Partition::Sharded(data_axis), Partition::Replicated])
            .unwrap(),
    );
    let router = builder.input(
        "router",
        Shape::new(DType::F32, &[4, 4])
            .unwrap()
            .with_partitions(&[Partition::Sharded(data_axis), Partition::Replicated])
            .unwrap(),
    );
    let gate_up = parameter(
        "gate_up",
        Shape::new(DType::F16, &[4, 64, 32])
            .unwrap()
            .with_partitions(&expert_partition)
            .unwrap(),
    );
    let down = parameter(
        "down",
        Shape::new(DType::F16, &[4, 32, 32])
            .unwrap()
            .with_partitions(&expert_partition)
            .unwrap(),
    );
    let output = builder
        .moe_swiglu(hidden, router, &gate_up, &down, 2)
        .unwrap();
    let program = builder.finish(&[output]).unwrap();
    let mesh = Sharding::mesh(&[(data_axis, 2), (expert_axis, 2)]).unwrap();
    let text = program
        .module_with_sharding_cuda(&Context::new(), &mesh, 108, 8, 0)
        .unwrap()
        .text();

    assert!(text.contains("sdy.manual_computation"), "{text}");
    assert!(text.contains("stablehlo.partition_id"), "{text}");
    assert_eq!(text.matches("__gpu$xla.gpu.triton").count(), 2, "{text}");
    assert_eq!(text.matches("stablehlo.all_reduce").count(), 1, "{text}");
    assert!(text.contains("dense<[[0, 1], [2, 3]]>"), "{text}");
    assert!(text.contains("tensor<2x64x32xf16>"), "{text}");
}

#[test]
fn expert_parallel_nvfp4_derives_local_components_inside_the_shared_manual_boundary() {
    use nml_sharding::Sharding;

    let data_axis = AxisTag::new(221);
    let expert_axis = AxisTag::new(222);
    let expert_partition = [
        Partition::Sharded(expert_axis),
        Partition::Replicated,
        Partition::Replicated,
    ];
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input(
        "hidden",
        Shape::new(DType::Bf16, &[4, 32])
            .unwrap()
            .with_partitions(&[Partition::Sharded(data_axis), Partition::Replicated])
            .unwrap(),
    );
    let router = builder.input(
        "router",
        Shape::new(DType::F32, &[4, 4])
            .unwrap()
            .with_partitions(&[Partition::Sharded(data_axis), Partition::Replicated])
            .unwrap(),
    );
    let gate_shape = Shape::new(DType::Bf16, &[4, 64, 32])
        .unwrap()
        .with_partitions(&expert_partition)
        .unwrap();
    let down_shape = Shape::new(DType::Bf16, &[4, 32, 32])
        .unwrap()
        .with_partitions(&expert_partition)
        .unwrap();
    let gate = Parameter::nvfp4("gate", "model.gate", gate_shape).unwrap();
    let down = Parameter::nvfp4("down", "model.down", down_shape).unwrap();
    let gate_bias = parameter(
        "gate_bias",
        Shape::new(DType::Bf16, &[4, 64])
            .unwrap()
            .with_partitions(&expert_partition[..2])
            .unwrap(),
    );
    let down_bias = parameter(
        "down_bias",
        Shape::new(DType::Bf16, &[4, 32])
            .unwrap()
            .with_partitions(&expert_partition[..2])
            .unwrap(),
    );
    let output = builder
        .routed_clamped_swiglu(hidden, router, &gate, &gate_bias, &down, &down_bias, 2)
        .unwrap();
    let program = builder.finish(&[output]).unwrap();
    let mesh = Sharding::mesh(&[(data_axis, 2), (expert_axis, 2)]).unwrap();
    let text = program
        .module_with_sharding_cuda(&Context::new(), &mesh, 108, 8, 0)
        .unwrap()
        .text();

    assert!(text.contains("sdy.manual_computation"), "{text}");
    assert!(text.contains("stablehlo.partition_id"), "{text}");
    assert_eq!(text.matches("__gpu$xla.gpu.triton").count(), 2, "{text}");
    assert!(text.contains("nvfp4_grouped_gate_up"), "{text}");
    assert!(text.contains("nvfp4_grouped_down"), "{text}");
    assert_eq!(text.matches("stablehlo.all_reduce").count(), 1, "{text}");
    assert!(text.contains("tensor<2x64x16xui8>"), "{text}");
    assert!(text.contains("tensor<2x64x2xui8>"), "{text}");
}
