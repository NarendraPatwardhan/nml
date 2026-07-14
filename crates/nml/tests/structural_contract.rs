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
    boxed: Box<nml::Tensor>,
}

#[derive(nml::NmlStruct)]
struct TupleLayer(nml::Tensor, #[nml(skip)] &'static str, nml::Tensor);

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
        boxed: Box::new(tensors[5]),
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
            "model.boxed",
        ]
    );

    let tuple = TupleLayer(tensors[0], "metadata is stripped", tensors[1]);
    assert_eq!(tuple.1, "metadata is stripped");
    let mut tuple_paths = Vec::new();
    tuple.visit_tensors("tuple", &mut |path, _| tuple_paths.push(path.to_owned()));
    assert_eq!(tuple_paths, ["tuple.0", "tuple.2"]);

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

    let platform = nml::Platform::cpu().unwrap();
    let host_bytes = [0; 8];
    let host = nml::Slice::from_bytes(shape, &host_bytes).unwrap();
    let shared = platform
        .upload(&host, nml::Sharding::replicated(), nml::Memory::Default)
        .unwrap();
    let buffers = model
        .bufferize("model", &mut |_, _| Ok::<_, ()>(shared.clone()))
        .unwrap();
    let mut buffer_paths = Vec::new();
    Model::visit_buffers(&buffers, "model", &mut |path, _| {
        buffer_paths.push(path.to_owned())
    });
    assert_eq!(buffer_paths, paths);

    for variant in &variants {
        let buffers = variant
            .bufferize("choice", &mut |_, _| Ok::<_, ()>(shared.clone()))
            .unwrap();
        let mut visited = 0;
        Choice::visit_buffers(&buffers, "choice", &mut |_, _| visited += 1);
        assert_eq!(
            visited,
            match variant {
                Choice::Dense { .. } => 1,
                Choice::Pair(..) => 2,
                Choice::Empty => 0,
            }
        );
    }
}
