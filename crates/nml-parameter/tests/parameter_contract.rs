use nml_parameter::{
    ComponentRole, Parameter, RepresentationKind, StorageEncoding,
    nvfp4::{decode_e2m1, decode_e4m3fn_scale, dequantize_row, global_scale, quantize_row},
};
use nml_sharding::Sharding;
use nml_types::{AxisTag, DType, Partition, Shape};

#[test]
fn dense_parameters_are_the_validated_one_component_case() {
    let shape = Shape::new(DType::Bf16, &[8, 16]).unwrap();
    let parameter = Parameter::dense("model.projection", "checkpoint.weight", shape).unwrap();
    assert_eq!(parameter.name(), "model.projection");
    assert_eq!(parameter.shape(), shape);
    assert_eq!(
        parameter.representation_id().kind(),
        RepresentationKind::Dense
    );
    assert_eq!(parameter.representation_id().version(), 1);
    assert_eq!(parameter.components().len(), 1);
    let component = &parameter.components()[0];
    assert_eq!(component.role(), ComponentRole::Values);
    assert_eq!(component.binding_name(), "model.projection");
    assert_eq!(component.artifact_name(), "checkpoint.weight");
    assert_eq!(
        component.storage().encoding(),
        StorageEncoding::Dense(DType::Bf16)
    );
    assert_eq!(component.storage().shape(), shape);
}

#[test]
fn parameter_identity_rejects_empty_names() {
    let shape = Shape::new(DType::F32, &[1]).unwrap();
    assert!(Parameter::dense("", "weight", shape).is_err());
    assert!(Parameter::dense("weight", "", shape).is_err());
}

#[test]
fn nvfp4_parameters_derive_exact_physical_components() {
    let shape = Shape::new(DType::Bf16, &[3, 17]).unwrap();
    let parameter = Parameter::nvfp4("model.projection", "checkpoint.projection", shape).unwrap();
    assert_eq!(
        parameter.representation_id().kind(),
        RepresentationKind::NvFp4
    );
    assert_eq!(parameter.representation_id().version(), 3);
    let spec = parameter.nvfp4_spec().unwrap();
    assert_eq!(spec.quantized_axis(), 1);
    assert_eq!(spec.block_size(), 16);
    assert!(spec.earlier_value_uses_low_nibble());

    let components = parameter.components();
    assert_eq!(components.len(), 3);
    assert_eq!(components[0].role(), ComponentRole::Payload);
    assert_eq!(
        components[0].artifact_name(),
        "checkpoint.projection.payload"
    );
    assert_eq!(
        components[0].storage().encoding(),
        StorageEncoding::PackedE2M1x2
    );
    assert_eq!(components[0].storage().shape().dtype(), DType::U8);
    assert_eq!(components[0].storage().shape().dimensions(), &[9, 3]);

    assert_eq!(components[1].role(), ComponentRole::BlockScales);
    assert_eq!(
        components[1].artifact_name(),
        "checkpoint.projection.block_scales"
    );
    assert_eq!(
        components[1].storage().encoding(),
        StorageEncoding::E4M3FnBits
    );
    assert_eq!(components[1].storage().shape().dimensions(), &[2, 3]);

    assert_eq!(components[2].role(), ComponentRole::GlobalScale);
    assert_eq!(components[2].storage().shape().dtype(), DType::F32);
    assert_eq!(components[2].storage().shape().dimensions(), &[]);
    assert!(parameter.dense_component().is_none());
}

#[test]
fn nvfp4_embeddings_retain_rowwise_lookup_storage() {
    let shape = Shape::new(DType::Bf16, &[3, 17]).unwrap();
    let parameter =
        Parameter::nvfp4_embedding("model.embedding", "checkpoint.embedding", shape).unwrap();
    assert_eq!(parameter.representation_id().version(), 3);
    assert_eq!(parameter.components()[0].storage().shape().dimensions(), &[3, 9]);
    assert_eq!(parameter.components()[1].storage().shape().dimensions(), &[3, 2]);
}

#[test]
fn e2m1_codec_exhaustively_matches_the_selected_bit_contract() {
    let expected: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for (code, expected) in expected.into_iter().enumerate() {
        let decoded = decode_e2m1(code as u8).unwrap();
        assert_eq!(decoded.to_bits(), expected.to_bits(), "code 0x{code:x}");
    }
    assert!(decode_e2m1(0x10).is_err());
}

#[test]
fn e4m3fn_scale_codec_covers_subnormal_normal_maximum_and_rejections() {
    assert_eq!(decode_e4m3fn_scale(0x00).unwrap(), 0.0);
    assert_eq!(decode_e4m3fn_scale(0x01).unwrap(), 2.0f32.powi(-9));
    assert_eq!(decode_e4m3fn_scale(0x38).unwrap(), 1.0);
    assert_eq!(decode_e4m3fn_scale(0x7e).unwrap(), 448.0);
    assert!(decode_e4m3fn_scale(0x7f).is_err());
    assert!(decode_e4m3fn_scale(0x80).is_err());
}

#[test]
fn row_codec_uses_low_nibble_first_and_rejects_nonzero_padding() {
    let values = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0,
    ];
    let global = global_scale(&values).unwrap();
    assert_eq!(global, 1.0 / 448.0);
    let encoded = quantize_row(&values, global).unwrap();
    assert_eq!(encoded.block_scales(), &[0x7e, 0x00]);
    assert_eq!(encoded.payload()[0], 0xef);
    assert_eq!(encoded.payload()[7], 0x76);
    assert_eq!(encoded.payload()[8], 0x00);
    let decoded = dequantize_row(
        encoded.payload(),
        encoded.block_scales(),
        global,
        values.len(),
    )
    .unwrap();
    for (actual, expected) in decoded.into_iter().zip(values) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    let mut invalid = encoded.payload().to_vec();
    *invalid.last_mut().unwrap() = 0x10;
    assert!(dequantize_row(&invalid, encoded.block_scales(), global, values.len()).is_err());
}

#[test]
fn nvfp4_rejects_values_that_cannot_form_the_selected_weight_recipe() {
    let scalar = Shape::new(DType::Bf16, &[]).unwrap();
    assert!(Parameter::nvfp4("weight", "weight", scalar).is_err());
    let integer = Shape::new(DType::I8, &[16]).unwrap();
    assert!(Parameter::nvfp4("weight", "weight", integer).is_err());
    let empty = Shape::new(DType::Bf16, &[2, 0]).unwrap();
    assert!(Parameter::nvfp4("weight", "weight", empty).is_err());
}

#[test]
fn nvfp4_sharding_derives_co_sharded_components_and_replicates_global_scale() {
    let tensor = AxisTag::new(71);
    let model = AxisTag::new(72);
    let shape = Shape::new(DType::Bf16, &[32, 2880])
        .unwrap()
        .with_axis_tags(&[tensor, model])
        .unwrap()
        .with_partitions(&[Partition::Sharded(tensor), Partition::Sharded(model)])
        .unwrap();
    let sharding = Sharding::mesh(&[(tensor, 2), (model, 2)]).unwrap();
    let parameter = Parameter::nvfp4("weight", "weight", shape).unwrap();

    parameter.validate_sharding(&sharding).unwrap();
    assert_eq!(
        sharding
            .shard_shape(parameter.components()[0].storage().shape())
            .unwrap()
            .dimensions(),
        &[720, 16]
    );
    assert_eq!(
        sharding
            .shard_shape(parameter.components()[1].storage().shape())
            .unwrap()
            .dimensions(),
        &[90, 16]
    );
    assert_eq!(
        sharding
            .replicated_mesh_axes(parameter.components()[2].storage().shape())
            .unwrap(),
        vec![tensor, model]
    );
}

#[test]
fn nvfp4_sharding_rejects_scale_block_splits_before_loading() {
    let model = AxisTag::new(81);
    let shape = Shape::new(DType::Bf16, &[4, 48])
        .unwrap()
        .with_axis_tags(&[AxisTag::UNKNOWN, model])
        .unwrap()
        .with_partitions(&[Partition::Unspecified, Partition::Sharded(model)])
        .unwrap();
    let parameter = Parameter::nvfp4("weight", "weight", shape).unwrap();
    let sharding = Sharding::mesh(&[(model, 2)]).unwrap();

    assert!(matches!(
        parameter.validate_sharding(&sharding),
        Err(nml_parameter::Error::MisalignedNvFp4Shard {
            axis: 1,
            logical_extent: 24,
            block_size: 16,
        })
    ));
}
