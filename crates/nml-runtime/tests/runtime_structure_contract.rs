use nml_runtime::Sharding;
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
