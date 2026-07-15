use nml_types::{
    AxisTag, BFloat16, Complex128, Complex64, DType, DTypeClass, Layout, Partition, Shape,
    ShapeError, F16, MAX_RANK,
};
use std::mem::{align_of, size_of};

#[test]
fn canonical_dtype_contract_is_exhaustive() {
    let expected = [
        (DType::Bool, DTypeClass::Boolean, 1, 1, "i1"),
        (DType::I8, DTypeClass::SignedInteger, 1, 1, "i8"),
        (DType::I16, DTypeClass::SignedInteger, 2, 2, "i16"),
        (DType::I32, DTypeClass::SignedInteger, 4, 4, "i32"),
        (DType::I64, DTypeClass::SignedInteger, 8, 8, "i64"),
        (DType::U8, DTypeClass::UnsignedInteger, 1, 1, "ui8"),
        (DType::U16, DTypeClass::UnsignedInteger, 2, 2, "ui16"),
        (DType::U32, DTypeClass::UnsignedInteger, 4, 4, "ui32"),
        (DType::U64, DTypeClass::UnsignedInteger, 8, 8, "ui64"),
        (DType::F16, DTypeClass::Float, 2, 2, "f16"),
        (DType::Bf16, DTypeClass::Float, 2, 2, "bf16"),
        (DType::F32, DTypeClass::Float, 4, 4, "f32"),
        (DType::F64, DTypeClass::Float, 8, 8, "f64"),
        (DType::C64, DTypeClass::Complex, 8, 4, "complex<f32>"),
        (DType::C128, DTypeClass::Complex, 16, 8, "complex<f64>"),
    ];
    assert_eq!(DType::ALL.len(), expected.len());
    for (actual, expected) in DType::ALL.into_iter().zip(expected) {
        assert_eq!(actual, expected.0);
        assert_eq!(actual.class(), expected.1);
        assert_eq!(actual.byte_width(), expected.2);
        assert_eq!(actual.alignment(), expected.3);
        assert_eq!(actual.stablehlo_spelling(), expected.4);
        assert_ne!(actual.stablehlo_spelling(), "index");
    }
}

#[test]
fn host_storage_layouts_match_dtype_contract() {
    assert_eq!((size_of::<F16>(), align_of::<F16>()), (2, 2));
    assert_eq!((size_of::<BFloat16>(), align_of::<BFloat16>()), (2, 2));
    assert_eq!((size_of::<Complex64>(), align_of::<Complex64>()), (8, 4));
    assert_eq!((size_of::<Complex128>(), align_of::<Complex128>()), (16, 8));
    assert_eq!(F16::from_bits(0x3c00).to_bits(), 0x3c00);
    assert_eq!(BFloat16::from_bits(0x3f80).to_bits(), 0x3f80);

    let value = Complex64::new(2.5, -7.0);
    assert_eq!(value.real, 2.5);
    assert_eq!(value.imaginary, -7.0);
}

#[test]
fn complex_values_are_deliberately_unordered() {
    assert!(DType::F32.require_ordering().is_ok());
    assert!(DType::C64.require_ordering().is_err());
    assert!(DType::C128.require_ordering().is_err());
}

#[test]
fn shape_carries_logical_partition_and_physical_metadata() {
    let batch = AxisTag::new(1);
    let model = AxisTag::new(2);
    let shape = Shape::new(DType::F32, &[3, 5])
        .unwrap()
        .with_axis_tags(&[batch, model])
        .unwrap()
        .with_partitions(&[Partition::Replicated, Partition::Sharded(model)])
        .unwrap()
        .with_layout(Layout::from_minor_to_major(&[0, 1]).unwrap())
        .unwrap();

    assert_eq!(shape.rank(), 2);
    assert_eq!(shape.dimensions(), &[3, 5]);
    assert_eq!(shape.axis_tags(), &[batch, model]);
    assert_eq!(
        shape.partitions(),
        &[Partition::Replicated, Partition::Sharded(model)]
    );
    assert_eq!(shape.layout().minor_to_major(), &[0, 1]);
    assert_eq!(shape.element_count().unwrap(), 15);
    assert_eq!(shape.byte_count().unwrap(), 60);

    let complex = shape.with_dtype(DType::C64);
    assert_eq!(complex.dtype(), DType::C64);
    assert_eq!(complex.dimensions(), shape.dimensions());
    assert_eq!(complex.axis_tags(), shape.axis_tags());
    assert_eq!(complex.partitions(), shape.partitions());
    assert_eq!(complex.layout(), shape.layout());
}

#[test]
fn scalar_empty_and_zero_dimension_shapes_are_distinct() {
    let scalar = Shape::new(DType::F64, &[]).unwrap();
    assert_eq!(scalar.element_count().unwrap(), 1);
    assert_eq!(scalar.byte_count().unwrap(), 8);

    let empty = Shape::new(DType::F32, &[2, 0, 4]).unwrap();
    assert_eq!(empty.element_count().unwrap(), 0);
    assert_eq!(empty.byte_count().unwrap(), 0);
    assert_eq!(empty.layout().minor_to_major(), &[2, 1, 0]);
}

#[test]
fn invalid_shape_metadata_is_rejected() {
    assert!(matches!(
        Shape::new(DType::F32, &[1; MAX_RANK + 1]),
        Err(ShapeError::RankTooLarge { .. })
    ));
    assert!(matches!(
        Shape::new(DType::F32, &[2, -1]),
        Err(ShapeError::NegativeDimension { axis: 1, .. })
    ));
    assert_eq!(
        Layout::from_minor_to_major(&[0, 0]),
        Err(ShapeError::InvalidLayout)
    );
    assert!(matches!(
        Shape::new(DType::F32, &[2, 3])
            .unwrap()
            .with_axis_tags(&[AxisTag::new(1)]),
        Err(ShapeError::MetadataRankMismatch { .. })
    ));
}

#[test]
fn element_and_byte_counts_are_checked() {
    let element_overflow = Shape::new(DType::F32, &[i64::MAX, i64::MAX]).unwrap();
    assert!(matches!(
        element_overflow.element_count(),
        Err(ShapeError::ElementCountOverflow { .. })
    ));

    let byte_overflow = Shape::new(DType::C128, &[i64::MAX]).unwrap();
    assert_eq!(byte_overflow.element_count().unwrap(), i64::MAX as usize);
    assert_eq!(
        byte_overflow.byte_count(),
        Err(ShapeError::ByteCountOverflow)
    );
}
