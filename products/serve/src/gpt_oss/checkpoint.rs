//! Exact checkpoint schema and reusable executable-slot binding.

use super::config::{AttentionKind, Config};
use nml::exe::Arguments;
use nml::io::ParameterSet;
use nml::{Loaded, LoadedParameter, Parameter, ParameterTree, Shape};
use std::error::Error as StdError;

pub(super) type BoxError = Box<dyn StdError + Send + Sync>;
pub(super) type Result<T> = std::result::Result<T, BoxError>;

#[derive(ParameterTree)]
pub(super) struct Weight {
    pub(super) weight: Parameter,
}

#[derive(ParameterTree)]
pub(super) struct Projection {
    pub(super) weight: Parameter,
    pub(super) bias: Parameter,
}

#[derive(ParameterTree)]
pub(super) struct Norm {
    pub(super) weight: Parameter,
}

#[derive(ParameterTree)]
pub(super) struct Attention {
    pub(super) q_proj: Projection,
    pub(super) k_proj: Projection,
    pub(super) v_proj: Projection,
    pub(super) o_proj: Projection,
    pub(super) sinks: Parameter,
}

#[derive(ParameterTree)]
pub(super) struct Router {
    pub(super) weight: Parameter,
    pub(super) bias: Parameter,
}

#[derive(ParameterTree)]
pub(super) struct Experts {
    pub(super) gate_up_proj: Parameter,
    pub(super) gate_up_proj_bias: Parameter,
    pub(super) down_proj: Parameter,
    pub(super) down_proj_bias: Parameter,
}

#[derive(ParameterTree)]
pub(super) struct Mlp {
    pub(super) router: Router,
    pub(super) experts: Experts,
}

#[derive(ParameterTree)]
pub(super) struct DecoderLayer {
    pub(super) input_layernorm: Norm,
    pub(super) self_attn: Attention,
    pub(super) post_attention_layernorm: Norm,
    pub(super) mlp: Mlp,
}

#[derive(ParameterTree)]
pub(super) struct Transformer {
    pub(super) embed_tokens: Weight,
    pub(super) layers: Vec<DecoderLayer>,
    pub(super) norm: Norm,
}

/// Complete logical ownership of the selected artifact's parameters.
#[derive(ParameterTree)]
pub(super) struct Checkpoint {
    pub(super) model: Transformer,
    pub(super) lm_head: Weight,
}

pub(super) type LoadedCheckpoint = Loaded<Checkpoint>;
pub(super) type LoadedDecoderLayer = Loaded<DecoderLayer>;

/// Resolves the exact artifact records without aliases or format guessing.
pub(super) fn declare(parameters: &ParameterSet, config: &Config) -> Result<Checkpoint> {
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
        &mut |name, shape| {
            parameters
                .nvfp4_embedding(name, shape, &[])
                .map_err(|error| Box::new(error) as BoxError)
        },
    )
}

pub(super) fn representative_layer<'checkpoint>(
    checkpoint: &'checkpoint Checkpoint,
    config: &Config,
    kind: AttentionKind,
) -> Result<&'checkpoint DecoderLayer> {
    config
        .layer_types()
        .iter()
        .position(|candidate| *candidate == kind)
        .and_then(|index| checkpoint.model.layers.get(index))
        .ok_or_else(|| message("GPT-OSS schedule has no representative layer of the required kind"))
}

/// Binds one loaded parameter tree to the structurally corresponding slots of
/// a reusable executable. Checkpoint names are deliberately not executable
/// identities: representation and storage contracts remain the authority.
pub(super) fn bind_tree<T: ParameterTree>(
    arguments: &mut Arguments<'_>,
    slots: &T,
    loaded: &Loaded<T>,
) -> Result<()> {
    bind_tree_components(arguments, slots, loaded)?;
    arguments
        .bake()
        .map_err(|error| Box::new(error) as BoxError)?;
    Ok(())
}

/// Binds one tree without sealing the complete executable. Composite bounded
/// executables use this for each structural subtree, then call `bake` exactly
/// once after every parameter component has been installed.
pub(super) fn bind_tree_components<T: ParameterTree>(
    arguments: &mut Arguments<'_>,
    slots: &T,
    loaded: &Loaded<T>,
) -> Result<()> {
    let mut slot_parameters = Vec::<(String, Parameter)>::new();
    slots.visit_parameters("", &mut |path, parameter| {
        slot_parameters.push((path.to_owned(), parameter.clone()));
    });
    let mut loaded_parameters = Vec::<(String, LoadedParameter)>::new();
    T::visit_loaded(loaded, "", &mut |path, parameter| {
        loaded_parameters.push((path.to_owned(), parameter.clone()));
    });
    if slot_parameters.len() != loaded_parameters.len() {
        return Err(message("reusable parameter trees have different leaf counts"));
    }
    for ((slot_path, slot), (loaded_path, parameter)) in
        slot_parameters.into_iter().zip(loaded_parameters)
    {
        if slot_path != loaded_path {
            return Err(message("reusable parameter trees have different structure"));
        }
        arguments
            .set_parameter_slot(&slot, &parameter)
            .map_err(|error| Box::new(error) as BoxError)?;
    }
    Ok(())
}

pub(super) fn declare_with(
    config: &Config,
    dense: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
    nvfp4: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
    nvfp4_embedding: &mut impl FnMut(&str, Shape) -> Result<Parameter>,
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
                        shape(&[experts, doubled_intermediate, hidden])?,
                    )?,
                    gate_up_proj_bias: dense(
                        &format!("{expert_prefix}.gate_up_proj_bias"),
                        shape(&[experts, doubled_intermediate])?,
                    )?,
                    down_proj: nvfp4(
                        &format!("{expert_prefix}.down_proj"),
                        shape(&[experts, hidden, intermediate])?,
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
                weight: nvfp4_embedding(
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
    Shape::new(nml::DataType::Bf16, &dimensions)
        .map_err(|error| Box::new(error) as BoxError)
}

pub(super) fn message(message: &'static str) -> BoxError {
    Box::new(std::io::Error::other(message))
}
