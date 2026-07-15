use nml_ir::{
    AttentionOptions, Error, FftType, ProgramBuilder, RopeLayout, RopeOptions, RopeScaling,
};
use nml_mlir::Context;
use nml_tensor::Element;
use nml_types::{AxisTag, BFloat16, Complex128, Complex64, DType, Layout, Shape, F16};

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
fn attention_primitives_are_typed_and_verify_as_stablehlo() {
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", Shape::new(DType::F32, &[4, 3]).unwrap());
    let update = builder.input("update", Shape::new(DType::F32, &[1, 3]).unwrap());
    let indices = builder.input("indices", Shape::new(DType::I32, &[2]).unwrap());
    let weight = builder.parameter("weight", Shape::new(DType::F32, &[3]).unwrap());
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
    let norm_weight = builder.parameter("norm_weight", Shape::new(DType::F32, &[4]).unwrap());
    let norm_bias = builder.parameter("norm_bias", Shape::new(DType::F32, &[4]).unwrap());
    let layer_norm = builder
        .layer_norm(input, Some(norm_weight), Some(norm_bias), 1, 1e-5)
        .unwrap();
    let swiglu = builder.swiglu(gate, input).unwrap();
    let geglu = builder.geglu(gate, input).unwrap();

    let embedding_weight =
        builder.parameter("embedding_weight", Shape::new(DType::F32, &[5, 4]).unwrap());
    let token_ids = builder.input("token_ids", Shape::new(DType::I32, &[2, 3]).unwrap());
    let embedding = builder
        .token_embedding(embedding_weight, token_ids)
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
    let bad_weight = builder.input("bad_weight", Shape::new(DType::F32, &[5]).unwrap());
    let good_weight = builder.input("good_weight", Shape::new(DType::F32, &[5, 4]).unwrap());
    let bad_ids = builder.input("bad_ids", Shape::new(DType::F32, &[2]).unwrap());
    assert!(matches!(
        builder.abs(bools),
        Err(Error::UnsupportedDType {
            operation: "abs",
            dtype: DType::Bool,
        })
    ));
    assert!(matches!(
        builder.token_embedding(bad_weight, bad_ids),
        Err(Error::RankMismatch { .. })
    ));
    assert!(matches!(
        builder.token_embedding(good_weight, bad_ids),
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
        assert!(!module
            .portable_artifact(&nml_mlir::stablehlo_current_version())
            .unwrap()
            .is_empty());
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

    let unbuilt_sm91 = lower(16, DType::F16, (9, 1));
    assert!(!unbuilt_sm91.contains("nml.flash_attention_3.paged"));
    assert!(unbuilt_sm91.contains("stablehlo.while"), "{unbuilt_sm91}");

    let unsupported_dtype = lower(256, DType::F32, (8, 0));
    assert!(!unsupported_dtype.contains("nml.flash_attention_2.paged"));
    assert!(unsupported_dtype.contains("__gpu$xla.gpu.triton"));
    assert!(!unsupported_dtype.contains("stablehlo.case"));
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
