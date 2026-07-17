use nml_parameter::{ComponentRole, Parameter, RepresentationKind, StorageEncoding};
use nml_types::{DType, Shape};

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
