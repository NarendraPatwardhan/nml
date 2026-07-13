use nml::NmlStruct as _;
use nml_types::{DType, Shape};

#[derive(nml::NmlStruct)]
struct Layer {
    weight: nml::Tensor,
    bias: Option<nml::Tensor>,
    #[nml(skip)]
    label: &'static str,
}

#[derive(nml::NmlStruct)]
struct Model {
    layers: Vec<Layer>,
    tuple: (nml::Tensor, [nml::Tensor; 2]),
}

#[derive(nml::NmlStruct)]
enum Choice {
    Dense {
        value: nml::Tensor,
        #[nml(skip)]
        id: u32,
    },
    Pair(nml::Tensor, #[nml(skip)] bool, nml::Tensor),
    Empty,
}

#[test]
fn derive_visits_nested_models_in_deterministic_field_order() {
    let mut builder = nml_ir::ProgramBuilder::new();
    let shape = Shape::new(DType::F16, &[2, 2]).unwrap();
    let tensors = (0..6)
        .map(|index| builder.parameter(format!("p{index}"), shape))
        .collect::<Vec<_>>();
    let model = Model {
        layers: vec![Layer {
            weight: tensors[0],
            bias: Some(tensors[1]),
            label: "metadata is stripped",
        }],
        tuple: (tensors[2], [tensors[3], tensors[4]]),
    };
    assert_eq!(model.layers[0].label, "metadata is stripped");
    let mut paths = Vec::new();
    model.visit_tensors("model", &mut |path, _| paths.push(path.to_owned()));
    assert_eq!(
        paths,
        [
            "model.layers.0.weight",
            "model.layers.0.bias",
            "model.tuple.0",
            "model.tuple.1.0",
            "model.tuple.1.1",
        ]
    );

    let variants = [
        Choice::Dense {
            value: tensors[0],
            id: 7,
        },
        Choice::Pair(tensors[1], true, tensors[2]),
        Choice::Empty,
    ];
    assert!(matches!(&variants[0], Choice::Dense { id: 7, .. }));
    assert!(matches!(&variants[1], Choice::Pair(_, true, _)));
    let mut count = 0;
    for variant in &variants {
        variant.visit_tensors("choice", &mut |_, _| count += 1);
    }
    assert_eq!(count, 3);
}
