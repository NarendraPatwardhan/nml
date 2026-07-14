use nml_ir::{Error, FftType, ProgramBuilder};
use nml_mlir::Context;
use nml_tensor::Element;
use nml_types::{BFloat16, Complex64, Complex128, DType, F16, Layout, Shape};

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
