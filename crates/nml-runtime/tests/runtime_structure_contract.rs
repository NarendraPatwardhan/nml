use nml_runtime::Sharding;
use nml_types::{DType, Shape};

#[test]
fn sharding_contract_rejects_invalid_product_and_exposes_compact_semantics() {
    assert!(Sharding::tiled(&[2, 0]).is_err());
    let tiled = Sharding::tiled(&[2, 3]).unwrap();
    assert_eq!(tiled.shard_count(), 6);
    assert!(!tiled.is_replicated());
    assert!(Sharding::replicated().is_replicated());

    let _shape = Shape::new(DType::F16, &[4, 6]).unwrap();
}
