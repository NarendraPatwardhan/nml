use nml::ParameterTree as _;
use nml_types::{DType, Shape};

#[derive(nml::ParameterTree)]
struct Layer {
    weight: nml::Parameter,
    bias: Option<nml::Parameter>,
    #[nml(skip)]
    label: &'static str,
}

#[derive(nml::ParameterTree)]
struct Model {
    layers: Vec<Layer>,
    tuple: (nml::Parameter, [nml::Parameter; 2]),
    boxed: Box<nml::Parameter>,
}

#[derive(nml::ParameterTree)]
struct TupleLayer(nml::Parameter, #[nml(skip)] &'static str, nml::Parameter);

#[derive(nml::ParameterTree)]
enum Choice {
    Dense {
        value: nml::Parameter,
        #[nml(skip)]
        id: u32,
    },
    Pair(nml::Parameter, #[nml(skip)] bool, nml::Parameter),
    Empty,
}

mod derive_hygiene {
    // Product crates commonly expose their own one-parameter Result alias. The
    // derive must bind the standard two-parameter Result explicitly instead of
    // resolving this alias at its call site.
    pub type Result<T> = std::result::Result<T, ProductError>;

    #[derive(Debug)]
    pub struct ProductError;

    #[derive(nml::ParameterTree)]
    pub struct Model {
        pub weight: nml::Parameter,
    }
}

#[test]
fn derive_is_independent_of_call_site_result_aliases() {
    let shape = Shape::new(DType::F16, &[1]).unwrap();
    let model = derive_hygiene::Model {
        weight: nml::Parameter::dense("weight", "weight", shape).unwrap(),
    };
    let _uses_product_result: derive_hygiene::Result<()> = Ok(());
    let result = model.load_parameters("model", &mut |_, _| {
        Err::<nml::LoadedParameter, _>(derive_hygiene::ProductError)
    });
    assert!(result.is_err());
}

#[test]
fn derive_visits_nested_models_in_deterministic_field_order() {
    let shape = Shape::new(DType::F16, &[2, 2]).unwrap();
    let parameters = (0..6)
        .map(|index| {
            let name = format!("p{index}");
            nml::Parameter::dense(&name, &name, shape).unwrap()
        })
        .collect::<Vec<_>>();
    let model = Model {
        layers: vec![Layer {
            weight: parameters[0].clone(),
            bias: Some(parameters[1].clone()),
            label: "metadata is stripped",
        }],
        tuple: (
            parameters[2].clone(),
            [parameters[3].clone(), parameters[4].clone()],
        ),
        boxed: Box::new(parameters[5].clone()),
    };
    assert_eq!(model.layers[0].label, "metadata is stripped");
    let mut paths = Vec::new();
    model.visit_parameters("model", &mut |path, _| paths.push(path.to_owned()));
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

    let tuple = TupleLayer(
        parameters[0].clone(),
        "metadata is stripped",
        parameters[1].clone(),
    );
    assert_eq!(tuple.1, "metadata is stripped");
    let mut tuple_paths = Vec::new();
    tuple.visit_parameters("tuple", &mut |path, _| tuple_paths.push(path.to_owned()));
    assert_eq!(tuple_paths, ["tuple.0", "tuple.2"]);

    let variants = [
        Choice::Dense {
            value: parameters[0].clone(),
            id: 7,
        },
        Choice::Pair(parameters[1].clone(), true, parameters[2].clone()),
        Choice::Empty,
    ];
    assert!(matches!(&variants[0], Choice::Dense { id: 7, .. }));
    assert!(matches!(&variants[1], Choice::Pair(_, true, _)));
    let mut count = 0;
    for variant in &variants {
        variant.visit_parameters("choice", &mut |_, _| count += 1);
    }
    assert_eq!(count, 3);

    let platform = nml::Platform::cpu().unwrap();
    let host_bytes = [0; 8];
    let host = nml::Slice::from_bytes(shape, &host_bytes).unwrap();
    let shared = platform
        .upload(&host, nml::Sharding::replicated(), nml::Memory::Default)
        .unwrap();
    let loaded = model
        .load_parameters("model", &mut |_, parameter| {
            nml::LoadedParameter::new(parameter.clone(), vec![shared.clone()]).map_err(|_| ())
        })
        .unwrap();
    let mut loaded_paths = Vec::new();
    Model::visit_loaded(&loaded, "model", &mut |path, _| {
        loaded_paths.push(path.to_owned())
    });
    assert_eq!(loaded_paths, paths);

    for variant in &variants {
        let loaded = variant
            .load_parameters("choice", &mut |_, parameter| {
                nml::LoadedParameter::new(parameter.clone(), vec![shared.clone()]).map_err(|_| ())
            })
            .unwrap();
        let mut visited = 0;
        Choice::visit_loaded(&loaded, "choice", &mut |_, _| visited += 1);
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
