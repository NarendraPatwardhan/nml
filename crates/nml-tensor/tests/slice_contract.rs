use nml_tensor::{Error, Slice};
use nml_types::{BFloat16, DType, F16, Layout, Shape};
use std::mem::align_of;

#[test]
fn owned_and_borrowed_typed_storage_share_one_slice_api() {
    let shape = Shape::new(DType::F32, &[2, 3]).unwrap();
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let borrowed = Slice::from_typed(shape, &values).unwrap();
    assert_eq!(borrowed.items::<f32>().unwrap(), &values);
    assert!(!borrowed.is_mutable());

    let mut owned = Slice::alloc(shape).unwrap();
    assert!(owned.is_mutable());
    assert_eq!(owned.data_pointer().unwrap().addr() % align_of::<f32>(), 0);
    owned.copy_from(&borrowed).unwrap();
    assert_eq!(owned.items::<f32>().unwrap(), &values);
}

#[test]
fn strided_permuted_reversed_and_sub_views_copy_correctly() {
    let shape = Shape::new(DType::I32, &[2, 3]).unwrap();
    let values = [1i32, 2, 3, 4, 5, 6];
    let source = Slice::from_typed(shape, &values)
        .unwrap()
        .permute(&[1, 0])
        .unwrap()
        .reverse(0)
        .unwrap()
        .sub_slice(0, 1, 2)
        .unwrap();
    assert!(!source.is_contiguous());

    let dense = source.to_contiguous().unwrap();
    assert_eq!(dense.shape().dimensions(), &[2, 2]);
    // Materialization preserves the tensor's declared physical layout. The
    // permuted shape is column-major, so its dense backing order differs from
    // logical row-major iteration while still representing [[2, 5], [1, 4]].
    assert_eq!(dense.shape().layout().minor_to_major(), &[0, 1]);
    assert_eq!(dense.items::<i32>().unwrap(), &[2, 1, 5, 4]);
}

#[test]
fn physical_layout_controls_dense_strides() {
    let shape = Shape::new(DType::U16, &[2, 3])
        .unwrap()
        .with_layout(Layout::from_minor_to_major(&[0, 1]).unwrap())
        .unwrap();
    let bytes = [0u8; 12];
    let slice = Slice::from_bytes(shape, &bytes).unwrap();
    assert_eq!(slice.byte_strides(), &[2, 4]);
    assert!(slice.is_contiguous());
}

#[test]
fn scalar_and_empty_shapes_have_valid_storage() {
    let scalar = Slice::alloc(Shape::new(DType::F64, &[]).unwrap()).unwrap();
    assert_eq!(scalar.items::<f64>().unwrap(), &[0.0]);

    let empty = Slice::alloc(Shape::new(DType::F32, &[2, 0, 4]).unwrap()).unwrap();
    assert!(empty.items::<f32>().unwrap().is_empty());
}

#[test]
fn invalid_storage_and_views_are_rejected() {
    let shape = Shape::new(DType::F32, &[2]).unwrap();
    assert!(matches!(
        Slice::from_bytes(shape, &[0; 7]),
        Err(Error::ByteLength { .. })
    ));

    let values = [1.0f32, 2.0];
    assert!(matches!(
        Slice::from_typed(shape, &values)
            .unwrap()
            .sub_slice(0, 1, 2),
        Err(Error::InvalidSubSlice { .. })
    ));
    assert!(matches!(
        Slice::from_typed(shape, &values).unwrap().items::<u32>(),
        Err(Error::DTypeMismatch { .. })
    ));
}

#[test]
fn half_conversions_cover_normal_subnormal_special_and_ties() {
    for value in [
        0.0f32,
        -0.0,
        1.0,
        -2.5,
        65_504.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let half = F16::from_f32(value);
        if value.is_infinite() {
            assert_eq!(half.to_f32(), value);
        } else {
            assert_eq!(half.to_f32(), value);
        }
    }
    assert_eq!(F16::from_bits(1).to_f32(), 2.0f32.powi(-24));
    assert!(F16::from_f32(f32::NAN).to_f32().is_nan());

    for value in [0.0f32, -0.0, 1.0, -2.5, 123.5, f32::INFINITY] {
        assert_eq!(BFloat16::from_f32(value).to_f32(), value);
    }
    assert!(BFloat16::from_f32(f32::NAN).to_f32().is_nan());
}
