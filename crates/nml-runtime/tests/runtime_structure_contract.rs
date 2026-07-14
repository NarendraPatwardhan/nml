use nml_runtime::{CacheSpec, Sharding};
use nml_types::{AxisTag, DType, Partition, Shape};

#[test]
fn sharding_contract_rejects_invalid_product_and_exposes_compact_semantics() {
    let data = AxisTag::new(1);
    let model = AxisTag::new(2);
    assert!(Sharding::mesh(&[(data, 0)]).is_err());
    let tiled = Sharding::mesh(&[(data, 2), (model, 3)]).unwrap();
    assert_eq!(tiled.execution_count(6).unwrap(), 6);
    assert!(tiled.is_mesh());
    assert!(Sharding::replicated().is_replicated());

    let shape = Shape::new(DType::F16, &[4, 6])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Sharded(model)])
        .unwrap();
    assert_eq!(tiled.shard_shape(shape).unwrap().dimensions(), &[2, 2]);
}

#[test]
fn cache_specs_share_one_checked_dense_and_paged_contract() {
    let dense = CacheSpec::dense(DType::F16, 2, 128, 4, 64).unwrap();
    assert_eq!(dense.capacity(), 128);
    assert_eq!(
        dense.key_value_shape().unwrap().dimensions(),
        &[2, 128, 4, 64]
    );
    assert!(dense.page_table_shape().unwrap().is_none());

    let paged = CacheSpec::paged(DType::Bf16, 2, 48, 32, 16, 4, 64).unwrap();
    assert_eq!(paged.capacity(), 512);
    assert_eq!(
        paged.key_value_shape().unwrap().dimensions(),
        &[48, 16, 4, 64]
    );
    assert_eq!(
        paged.page_table_shape().unwrap().unwrap().dimensions(),
        &[2, 32]
    );
    assert!(CacheSpec::dense(DType::I32, 1, 1, 1, 1).is_err());
    assert!(CacheSpec::paged(DType::F32, 1, 0, 1, 1, 1, 1).is_err());
    if usize::BITS > 63 {
        assert!(CacheSpec::paged(DType::F32, 1, 1, i64::MAX as usize + 1, 1, 1, 1).is_err());
    }
}
