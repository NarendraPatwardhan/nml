use nml_ir::{Error, FftType, ProgramBuilder};
use nml_mlir::Context;
use nml_types::{DType, Layout, Shape};

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
