//! Exact logical parameter tree for the selected GPT-OSS 20B NVFP4 artifact.

use crate::config::{AttentionKind, Config};
use nml::io::ParameterSet;
use nml::{DataType, Graph, Parameter, ParameterTree, Shape, Tensor};
use std::error::Error as StdError;

type BoxError = Box<dyn StdError + Send + Sync>;
type Result<T> = std::result::Result<T, BoxError>;

#[derive(ParameterTree)]
struct Weight {
    weight: Parameter,
}

#[derive(ParameterTree)]
struct Projection {
    weight: Parameter,
    bias: Parameter,
}

#[derive(ParameterTree)]
struct Norm {
    weight: Parameter,
}

#[derive(ParameterTree)]
struct Attention {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    sinks: Parameter,
}

#[derive(ParameterTree)]
struct Router {
    weight: Parameter,
    bias: Parameter,
}

#[derive(ParameterTree)]
struct Experts {
    gate_up_proj: Parameter,
    gate_up_proj_bias: Parameter,
    down_proj: Parameter,
    down_proj_bias: Parameter,
}

#[derive(ParameterTree)]
struct Mlp {
    router: Router,
    experts: Experts,
}

#[derive(ParameterTree)]
struct DecoderLayer {
    input_layernorm: Norm,
    self_attn: Attention,
    post_attention_layernorm: Norm,
    mlp: Mlp,
}

#[derive(ParameterTree)]
struct Transformer {
    embed_tokens: Weight,
    layers: Vec<DecoderLayer>,
    norm: Norm,
}

/// The complete logical parameter ownership tree consumed by the model graph.
///
/// The type is public only within the package-private `//products/serve:gpt_oss`
/// Rust target. NML's facade does not expose model-specific checkpoint types.
#[derive(ParameterTree)]
pub struct Checkpoint {
    model: Transformer,
    lm_head: Weight,
}

/// Static graph family compiled for one prefill or decode sequence length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphSpec {
    sequence: usize,
    cache_capacity: usize,
    mode: GraphMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphMode {
    Prefill,
    Decode,
}

impl GraphSpec {
    pub fn prefill(sequence: usize, cache_capacity: usize) -> Result<Self> {
        Self::new(sequence, cache_capacity, GraphMode::Prefill)
    }

    pub fn decode(cache_capacity: usize) -> Result<Self> {
        Self::new(1, cache_capacity, GraphMode::Decode)
    }

    fn new(sequence: usize, cache_capacity: usize, mode: GraphMode) -> Result<Self> {
        if sequence == 0 {
            return Err(message("GPT-OSS graph sequence must be nonzero"));
        }
        if cache_capacity < sequence {
            return Err(message(
                "GPT-OSS cache capacity must be at least the graph sequence",
            ));
        }
        if cache_capacity > i32::MAX as usize {
            return Err(message(
                "GPT-OSS cache capacity exceeds the paged-attention I32 index domain",
            ));
        }
        Ok(Self {
            sequence,
            cache_capacity,
            mode,
        })
    }

    pub const fn sequence(self) -> usize {
        self.sequence
    }

    pub const fn cache_capacity(self) -> usize {
        self.cache_capacity
    }

    pub const fn mode(self) -> GraphMode {
        self.mode
    }
}

/// Last-token logits and the transactionally updated cache storage for each layer.
///
/// The donated buffers retain their dense, contiguous envelope for the current
/// model-neutral engine. Attention views that storage as identity-mapped pages;
/// reshaping the view neither copies the cache nor changes buffer ownership.
pub struct GraphOutputs {
    pub logits: Tensor,
    pub caches: Vec<(Tensor, Tensor)>,
}

/// Greedy token and transactionally updated caches consumed by the engine.
pub struct GreedyGraphOutputs {
    pub token: Tensor,
    pub caches: Vec<(Tensor, Tensor)>,
}

/// Resolves every parameter from the selected artifact without aliases.
///
/// The conversion manifest is the storage contract. Accepting aliases here
/// would allow a nearby checkpoint to masquerade as the pinned GPT-OSS
/// vertical and would weaken representation identity before loading begins.
pub fn declare(parameters: &ParameterSet, config: &Config) -> Result<Checkpoint> {
    declare_with(
        config,
        &mut |name, shape| {
            parameters
                .dense(name, shape, &[])
                .map_err(|error| Box::new(error) as BoxError)
        },
        &mut |name, shape| {
            parameters
                .nvfp4(name, shape, &[])
                .map_err(|error| Box::new(error) as BoxError)
        },
    )
}

/// Builds the exact single-request GPT-OSS inference graph.
///
/// The current serving engine compiles one prefill family and one single-token
/// decode family. Each request still owns contiguous donated storage, but the
/// model consumes it through paged attention now. Continuous batching can
/// replace only the identity page table and per-request storage owner without
/// changing the model block or its attention semantics.
pub fn build_graph(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    spec: GraphSpec,
) -> Result<GraphOutputs> {
    let sequence = dimension(spec.sequence)?;
    let cache_capacity = dimension(spec.cache_capacity)?;
    let hidden_size = dimension(config.hidden_size())?;
    let query_heads = dimension(config.query_heads())?;
    let key_value_heads = dimension(config.key_value_heads())?;
    let head_dim = dimension(config.head_dim())?;
    let token_shape = shape_for(DataType::I32, &[1, sequence])?;
    let tokens = graph.input("tokens", token_shape);
    let zero = nml(graph.scalar(0_i32))?;
    let (positions, cache_start, sequence_length) = match spec.mode() {
        GraphMode::Prefill => (
            nml(graph.iota(token_shape, 1))?,
            zero,
            nml(graph.scalar(i32::try_from(spec.sequence()).map_err(|_| {
                message("GPT-OSS prefill sequence exceeds the I32 attention domain")
            })?))?,
        ),
        GraphMode::Decode => {
            let position = graph.input("position", shape_for(DataType::I32, &[])?);
            let positions = nml(graph.reshape(position, token_shape))?;
            let one = nml(graph.scalar(1_i32))?;
            let sequence_length = nml(graph.add(position, one))?;
            (positions, position, sequence_length)
        }
    };
    let mut hidden = nml(graph.token_embedding(&checkpoint.model.embed_tokens.weight, tokens))?;
    let cache_shape = shape_for(
        DataType::Bf16,
        &[1, cache_capacity, key_value_heads, head_dim],
    )?;
    // A finite single-request graph owns one contiguous physical page. This is
    // exactly the donated cache buffer below, viewed without a copy. The future
    // continuous-batching owner can substitute a multi-page table while keeping
    // the model's paged-attention operation unchanged.
    let page_size = cache_capacity;
    let paged_cache_shape = shape_for(DataType::Bf16, &[1, page_size, key_value_heads, head_dim])?;
    let page_table = nml(graph.iota(shape_for(DataType::I32, &[1, 1])?, 1))?;
    let sequence_lengths = nml(graph.reshape(sequence_length, shape_for(DataType::I32, &[1])?))?;
    let rope = nml::attention::RopeOptions {
        base: config.rope_theta(),
        rotary_dimensions: config.head_dim(),
        layout: nml::attention::RopeLayout::Sequential,
        scaling: nml::attention::RopeScaling::Yarn {
            factor: config.rope_factor(),
            beta_fast: config.rope_beta_fast(),
            beta_slow: config.rope_beta_slow(),
            original_context: config.initial_context_length(),
            truncate: false,
            attention_factor: None,
        },
    };
    let mut caches = Vec::with_capacity(config.layers());

    for (index, layer) in checkpoint.model.layers.iter().enumerate() {
        let key_input = graph.input(format!("cache.{index}.key"), cache_shape);
        let value_input = graph.input(format!("cache.{index}.value"), cache_shape);

        let residual = hidden;
        let input_norm = nml(graph.parameter_value(&layer.input_layernorm.weight))?;
        hidden = nml(graph.rms_norm(hidden, Some(input_norm), 2, config.rms_norm_epsilon()))?;
        let query = nml(graph.linear(
            hidden,
            &layer.self_attn.q_proj.weight,
            Some(&layer.self_attn.q_proj.bias),
        ))?;
        let key = nml(graph.linear(
            hidden,
            &layer.self_attn.k_proj.weight,
            Some(&layer.self_attn.k_proj.bias),
        ))?;
        let value = nml(graph.linear(
            hidden,
            &layer.self_attn.v_proj.weight,
            Some(&layer.self_attn.v_proj.bias),
        ))?;
        let query = nml(graph.reshape(
            query,
            shape_for(DataType::Bf16, &[1, sequence, query_heads, head_dim])?,
        ))?;
        let key = nml(graph.reshape(
            key,
            shape_for(DataType::Bf16, &[1, sequence, key_value_heads, head_dim])?,
        ))?;
        let value = nml(graph.reshape(
            value,
            shape_for(DataType::Bf16, &[1, sequence, key_value_heads, head_dim])?,
        ))?;
        let query = nml(graph.rope(query, positions, rope))?;
        let key = nml(graph.rope(key, positions, rope))?;
        let key_cache =
            nml(graph.dynamic_update_slice(key_input, key, &[zero, cache_start, zero, zero]))?;
        let value_cache =
            nml(graph.dynamic_update_slice(value_input, value, &[zero, cache_start, zero, zero]))?;
        let key_cache = nml(graph.reuse_buffer(key_cache, key_input))?;
        let value_cache = nml(graph.reuse_buffer(value_cache, value_input))?;
        let paged_key_cache = nml(graph.reshape(key_cache, paged_cache_shape))?;
        let paged_value_cache = nml(graph.reshape(value_cache, paged_cache_shape))?;
        let sinks = nml(graph.parameter_value(&layer.self_attn.sinks))?;
        let sliding_window = match config.attention_kind(index) {
            Some(AttentionKind::SlidingAttention) => Some(config.sliding_window()),
            Some(AttentionKind::FullAttention) => None,
            None => return Err(message("GPT-OSS layer schedule is incomplete")),
        };
        let attention = nml(graph.paged_attention(
            query,
            paged_key_cache,
            paged_value_cache,
            page_table,
            sequence_lengths,
            positions,
            Some(sinks),
            nml::attention::Options {
                causal: true,
                sliding_window,
                scale: None,
            },
        ))?;
        let attention = nml(graph.reshape(
            attention,
            shape_for(
                DataType::Bf16,
                &[1, sequence, dimension(config.query_width())?],
            )?,
        ))?;
        let attention = nml(graph.linear(
            attention,
            &layer.self_attn.o_proj.weight,
            Some(&layer.self_attn.o_proj.bias),
        ))?;
        hidden = nml(graph.add(residual, attention))?;

        let residual = hidden;
        let post_norm = nml(graph.parameter_value(&layer.post_attention_layernorm.weight))?;
        hidden = nml(graph.rms_norm(hidden, Some(post_norm), 2, config.rms_norm_epsilon()))?;
        let routed =
            nml(graph.reshape(hidden, shape_for(DataType::Bf16, &[sequence, hidden_size])?))?;
        let router_logits = nml(graph.linear(
            routed,
            &layer.mlp.router.weight,
            Some(&layer.mlp.router.bias),
        ))?;
        let routed = nml(graph.moe_clamped_swiglu(
            routed,
            router_logits,
            &layer.mlp.experts.gate_up_proj,
            &layer.mlp.experts.gate_up_proj_bias,
            &layer.mlp.experts.down_proj,
            &layer.mlp.experts.down_proj_bias,
            config.experts_per_token(),
        ))?;
        let routed = nml(graph.reshape(
            routed,
            shape_for(DataType::Bf16, &[1, sequence, hidden_size])?,
        ))?;
        hidden = nml(graph.add(residual, routed))?;
        caches.push((key_cache, value_cache));
    }

    let final_norm = nml(graph.parameter_value(&checkpoint.model.norm.weight))?;
    hidden = nml(graph.rms_norm(hidden, Some(final_norm), 2, config.rms_norm_epsilon()))?;
    let last = nml(graph.slice(
        hidden,
        &[0, sequence - 1, 0],
        &[1, sequence, hidden_size],
        &[1, 1, 1],
    ))?;
    let last = nml(graph.reshape(last, shape_for(DataType::Bf16, &[1, hidden_size])?))?;
    let logits = nml(graph.linear(last, &checkpoint.lm_head.weight, None))?;
    Ok(GraphOutputs { logits, caches })
}

/// Builds the exact graph family and places deterministic greedy selection at
/// its model-neutral engine boundary. Logits remain internal to the compiled
/// executable; only one I32 token crosses PJRT per generation step.
pub fn build_greedy_graph(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    spec: GraphSpec,
) -> Result<GreedyGraphOutputs> {
    let outputs = build_graph(graph, checkpoint, config, spec)?;
    let (_, token) = nml(graph.argmax(outputs.logits, 1))?;
    Ok(GreedyGraphOutputs {
        token,
        caches: outputs.caches,
    })
}

fn declare_with(
    config: &Config,
    dense: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
    nvfp4: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
) -> Result<Checkpoint> {
    let hidden = config.hidden_size();
    let query = config.query_width();
    let key_value = config.key_value_width();
    let intermediate = config.intermediate_size();
    let doubled_intermediate = intermediate
        .checked_mul(2)
        .ok_or_else(|| message("GPT-OSS intermediate width overflows usize"))?;
    let experts = config.experts();
    let mut layers = Vec::with_capacity(config.layers());

    for index in 0..config.layers() {
        let prefix = format!("model.layers.{index}");
        let attention = format!("{prefix}.self_attn");
        let mlp = format!("{prefix}.mlp");
        let expert_prefix = format!("{mlp}.experts");
        layers.push(DecoderLayer {
            input_layernorm: Norm {
                weight: dense(
                    &format!("{prefix}.input_layernorm.weight"),
                    shape(&[hidden])?,
                )?,
            },
            self_attn: Attention {
                q_proj: projection(nvfp4, dense, &format!("{attention}.q_proj"), query, hidden)?,
                k_proj: projection(
                    nvfp4,
                    dense,
                    &format!("{attention}.k_proj"),
                    key_value,
                    hidden,
                )?,
                v_proj: projection(
                    nvfp4,
                    dense,
                    &format!("{attention}.v_proj"),
                    key_value,
                    hidden,
                )?,
                o_proj: projection(nvfp4, dense, &format!("{attention}.o_proj"), hidden, query)?,
                sinks: dense(
                    &format!("{attention}.sinks"),
                    shape(&[config.query_heads()])?,
                )?,
            },
            post_attention_layernorm: Norm {
                weight: dense(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    shape(&[hidden])?,
                )?,
            },
            mlp: Mlp {
                router: Router {
                    weight: dense(&format!("{mlp}.router.weight"), shape(&[experts, hidden])?)?,
                    bias: dense(&format!("{mlp}.router.bias"), shape(&[experts])?)?,
                },
                experts: Experts {
                    gate_up_proj: nvfp4(
                        &format!("{expert_prefix}.gate_up_proj"),
                        shape(&[experts, hidden, doubled_intermediate])?,
                    )?,
                    gate_up_proj_bias: dense(
                        &format!("{expert_prefix}.gate_up_proj_bias"),
                        shape(&[experts, doubled_intermediate])?,
                    )?,
                    down_proj: nvfp4(
                        &format!("{expert_prefix}.down_proj"),
                        shape(&[experts, intermediate, hidden])?,
                    )?,
                    down_proj_bias: dense(
                        &format!("{expert_prefix}.down_proj_bias"),
                        shape(&[experts, hidden])?,
                    )?,
                },
            },
        });
    }

    Ok(Checkpoint {
        model: Transformer {
            embed_tokens: Weight {
                weight: nvfp4(
                    "model.embed_tokens.weight",
                    shape(&[config.vocabulary(), hidden])?,
                )?,
            },
            layers,
            norm: Norm {
                weight: dense("model.norm.weight", shape(&[hidden])?)?,
            },
        },
        lm_head: Weight {
            weight: nvfp4("lm_head.weight", shape(&[config.vocabulary(), hidden])?)?,
        },
    })
}

fn projection(
    nvfp4: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
    dense: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
    prefix: &str,
    output: usize,
    input: usize,
) -> Result<Projection> {
    Ok(Projection {
        weight: nvfp4(&format!("{prefix}.weight"), shape(&[output, input])?)?,
        bias: dense(&format!("{prefix}.bias"), shape(&[output])?)?,
    })
}

fn shape(dimensions: &[usize]) -> Result<Shape> {
    let dimensions = dimensions
        .iter()
        .copied()
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| message("GPT-OSS parameter dimension exceeds I64"))?;
    Shape::new(DataType::Bf16, &dimensions).map_err(|error| Box::new(error) as BoxError)
}

fn dimension(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| message("GPT-OSS graph dimension exceeds I64"))
}

fn shape_for(dtype: DataType, dimensions: &[i64]) -> Result<Shape> {
    Shape::new(dtype, dimensions).map_err(|error| Box::new(error) as BoxError)
}

fn nml<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: StdError + Send + Sync + 'static,
{
    result.map_err(|error| Box::new(error) as BoxError)
}

fn message(message: &'static str) -> BoxError {
    Box::new(ContractError(message))
}

#[derive(Debug)]
struct ContractError(&'static str);

impl std::fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}

impl StdError for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;
    use nml::ParameterTree;
    use serde::Deserialize;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::Path;

    #[derive(Deserialize)]
    struct Manifest {
        logical_tensor_count: usize,
        physical_tensor_count: usize,
        tensors: Vec<ManifestTensor>,
    }

    #[derive(Deserialize)]
    struct ManifestTensor {
        logical_name: String,
        logical_dtype: String,
        logical_shape: Vec<i64>,
        name: String,
        representation: String,
    }

    #[test]
    fn structured_parameter_tree_matches_every_frozen_manifest_component() {
        let root = runfile_root();
        let config = Config::from_file(root.join("config.json")).unwrap();
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(root.join("output-tensors.json")).unwrap())
                .unwrap();

        let mut dense = |name: &str, shape: Shape| {
            Parameter::dense(name, name, shape).map_err(|error| Box::new(error) as BoxError)
        };
        let mut nvfp4 = |name: &str, shape: Shape| {
            Parameter::nvfp4(name, name, shape).map_err(|error| Box::new(error) as BoxError)
        };
        let checkpoint = declare_with(&config, &mut dense, &mut nvfp4).unwrap();
        let mut declared = BTreeMap::new();
        checkpoint.visit_parameters("", &mut |_, parameter| {
            let previous = declared.insert(parameter.name().to_owned(), parameter.clone());
            assert!(
                previous.is_none(),
                "duplicate declaration {}",
                parameter.name()
            );
        });

        let mut physical = BTreeMap::<String, BTreeSet<String>>::new();
        let mut logical = BTreeMap::<String, (&str, &[i64], &str)>::new();
        for tensor in &manifest.tensors {
            physical
                .entry(tensor.logical_name.clone())
                .or_default()
                .insert(tensor.name.clone());
            match logical.get(tensor.logical_name.as_str()) {
                Some(existing) => assert_eq!(
                    *existing,
                    (
                        tensor.logical_dtype.as_str(),
                        tensor.logical_shape.as_slice(),
                        tensor.representation.as_str(),
                    ),
                    "inconsistent physical records for {}",
                    tensor.logical_name
                ),
                None => {
                    logical.insert(
                        tensor.logical_name.clone(),
                        (
                            tensor.logical_dtype.as_str(),
                            tensor.logical_shape.as_slice(),
                            tensor.representation.as_str(),
                        ),
                    );
                }
            }
        }

        assert_eq!(manifest.logical_tensor_count, 411);
        assert_eq!(manifest.physical_tensor_count, 703);
        assert_eq!(manifest.tensors.len(), manifest.physical_tensor_count);
        assert_eq!(logical.len(), manifest.logical_tensor_count);
        assert_eq!(declared.len(), manifest.logical_tensor_count);
        assert_eq!(
            declared.keys().collect::<Vec<_>>(),
            logical.keys().collect::<Vec<_>>()
        );

        for (name, parameter) in declared {
            let (dtype, dimensions, representation) = logical[&name];
            assert_eq!(dtype, "BF16", "logical dtype mismatch for {name}");
            assert_eq!(
                parameter.shape().dimensions(),
                dimensions,
                "shape mismatch for {name}"
            );
            let expected_components = match representation {
                "dense" => 1,
                "nvfp4" => 3,
                other => panic!("unknown representation {other} for {name}"),
            };
            assert_eq!(parameter.components().len(), expected_components, "{name}");
            assert_eq!(
                parameter
                    .components()
                    .iter()
                    .map(|component| component.artifact_name().to_owned())
                    .collect::<BTreeSet<_>>(),
                physical[&name],
                "physical component mismatch for {name}"
            );
        }
    }

    #[test]
    fn graph_consumes_the_complete_parameter_tree_and_exposes_dense_cache_donation() {
        let root = runfile_root();
        let config = Config::from_file(root.join("config.json")).unwrap();
        let mut dense = |name: &str, shape: Shape| {
            Parameter::dense(name, name, shape).map_err(|error| Box::new(error) as BoxError)
        };
        let mut nvfp4 = |name: &str, shape: Shape| {
            Parameter::nvfp4(name, name, shape).map_err(|error| Box::new(error) as BoxError)
        };
        let checkpoint = declare_with(&config, &mut dense, &mut nvfp4).unwrap();
        let mut graph = Graph::new();
        let outputs = build_graph(
            &mut graph,
            &checkpoint,
            &config,
            GraphSpec::prefill(3, 8).unwrap(),
        )
        .unwrap();

        assert_eq!(outputs.logits.shape().dimensions(), [1, 201_088]);
        assert_eq!(outputs.logits.shape().dtype(), DataType::Bf16);
        assert_eq!(outputs.caches.len(), 24);

        let mut named_outputs = vec![("logits".to_owned(), outputs.logits)];
        for (layer, (key, value)) in outputs.caches.into_iter().enumerate() {
            named_outputs.push((format!("cache.{layer}.key"), key));
            named_outputs.push((format!("cache.{layer}.value"), value));
        }
        let program = graph.finish_named(&named_outputs).unwrap();
        let inputs = program.inputs().collect::<Vec<_>>();
        let parameter_components = inputs
            .iter()
            .filter(|(_, _, binding)| binding.is_parameter_component())
            .count();
        let activations = inputs.len() - parameter_components;
        assert_eq!(parameter_components, 703);
        assert_eq!(activations, 49);

        let aliases = program.output_aliases().collect::<Vec<_>>();
        assert_eq!(aliases.len(), 49);
        assert_eq!(aliases[0], None);
        assert!(aliases[1..].iter().all(Option::is_some));
        let stablehlo = program.stablehlo().unwrap();
        assert_eq!(
            stablehlo.matches("stablehlo.while").count(),
            config.layers(),
            "every GPT-OSS layer must consume its cache through paged attention"
        );
    }

    #[test]
    fn finite_prefill_and_decode_families_keep_sampling_and_cache_state_on_device() {
        let root = runfile_root();
        let config = Config::from_file(root.join("config.json")).unwrap();
        let mut dense = |name: &str, shape: Shape| {
            Parameter::dense(name, name, shape).map_err(|error| Box::new(error) as BoxError)
        };
        let mut nvfp4 = |name: &str, shape: Shape| {
            Parameter::nvfp4(name, name, shape).map_err(|error| Box::new(error) as BoxError)
        };
        let checkpoint = declare_with(&config, &mut dense, &mut nvfp4).unwrap();

        for spec in [
            GraphSpec::prefill(3, 8).unwrap(),
            GraphSpec::decode(8).unwrap(),
        ] {
            let mut graph = Graph::new();
            let outputs = build_greedy_graph(&mut graph, &checkpoint, &config, spec).unwrap();
            assert_eq!(outputs.token.shape().dtype(), DataType::I32);
            assert_eq!(outputs.token.shape().dimensions(), [1]);
            assert_eq!(outputs.caches.len(), 24);
            let mut named = vec![("token".to_owned(), outputs.token)];
            for (layer, (key, value)) in outputs.caches.into_iter().enumerate() {
                named.push((format!("cache.{layer}.key"), key));
                named.push((format!("cache.{layer}.value"), value));
            }
            let program = graph.finish_named(&named).unwrap();
            let input_names = program
                .inputs()
                .map(|(name, _, _)| name)
                .collect::<BTreeSet<_>>();
            assert!(input_names.contains("tokens"));
            match spec.mode() {
                GraphMode::Prefill => assert!(!input_names.contains("position")),
                GraphMode::Decode => assert!(input_names.contains("position")),
            }
            assert!(!input_names.contains("positions"));
            assert!(!input_names.contains("cache_start"));
            assert_eq!(program.output_aliases().filter(Option::is_some).count(), 48);
        }
    }

    fn runfile_root() -> std::path::PathBuf {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        Path::new(&runfiles).join("_main/artifacts/gpt-oss-20b-nvfp4")
    }
}
