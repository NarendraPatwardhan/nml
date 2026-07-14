use nml_sharding::{Error, Sharding};
use nml_types::{AxisTag, DType, Layout, Partition, Shape};

#[test]
fn logical_mesh_validates_shapes_and_maps_canonical_ranges() {
    let data = AxisTag::new(1);
    let model = AxisTag::new(2);
    let mesh = Sharding::mesh(&[(data, 2), (model, 2)]).unwrap();
    let shape = Shape::new(DType::F32, &[8, 12])
        .unwrap()
        .with_partitions(&[Partition::Sharded(data), Partition::Sharded(model)])
        .unwrap();
    assert_eq!(mesh.compile_topology(4).unwrap(), (1, 4, 4));
    assert_eq!(mesh.shard_shape(shape).unwrap().dimensions(), &[4, 6]);
    assert_eq!(mesh.ranges(shape, 3).unwrap(), vec![(0, 4, 4), (1, 6, 6)]);

    let column_major = shape
        .with_layout(Layout::from_minor_to_major(&[0, 1]).unwrap())
        .unwrap();
    assert_eq!(
        mesh.shard_shape(column_major).unwrap().layout(),
        column_major.layout()
    );
}

#[test]
fn logical_mesh_rejects_ambiguous_or_impossible_placement() {
    let axis = AxisTag::new(7);
    assert_eq!(
        Sharding::mesh(&[(axis, 2), (axis, 2)]),
        Err(Error::DuplicateAxis(axis))
    );
    let mesh = Sharding::mesh(&[(axis, 2)]).unwrap();
    let unknown = Shape::new(DType::F32, &[4])
        .unwrap()
        .with_partitions(&[Partition::Sharded(AxisTag::UNKNOWN)])
        .unwrap();
    assert_eq!(mesh.validate_shape(unknown), Err(Error::UnknownAxis));
    let shape = Shape::new(DType::F32, &[3])
        .unwrap()
        .with_partitions(&[Partition::Sharded(axis)])
        .unwrap();
    assert!(matches!(
        mesh.validate_shape(shape),
        Err(Error::UnevenDimension { .. })
    ));
    assert_eq!(
        mesh.execution_count(4),
        Err(Error::DeviceCount {
            required: 2,
            available: 4
        })
    );

    if let Some(too_large) = usize::try_from(i64::MAX)
        .ok()
        .and_then(|maximum| maximum.checked_add(1))
    {
        assert_eq!(
            Sharding::mesh(&[(axis, too_large)]),
            Err(Error::AxisSizeOverflow {
                tag: axis,
                size: too_large,
            })
        );
    }
}
