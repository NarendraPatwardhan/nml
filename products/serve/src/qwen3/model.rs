//! Qwen3 graph construction.

use super::{Error, Result, config::Config, nml_result};
use crate::engine::{GraphKind, GraphOutputs};
use nml::io::ParameterSet;
use nml::{DataType, Graph, Parameter, ParameterTree, Shape};

#[derive(ParameterTree)]
struct Linear {
    weight: Parameter,
}

#[derive(ParameterTree)]
struct Norm {
    weight: Parameter,
}

#[derive(ParameterTree)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Norm,
    k_norm: Norm,
}

#[derive(ParameterTree)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
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
    embed_tokens: Linear,
    layers: Vec<DecoderLayer>,
    norm: Norm,
}

#[derive(ParameterTree)]
pub(crate) struct Checkpoint {
    model: Transformer,
}

pub(crate) fn declare(parameters: &ParameterSet, config: &Config) -> Result<Checkpoint> {
    let model = parameters.view("model");
    let layers = model.view("layers");
    let hidden = config.hidden_size;
    let query = config.query_width();
    let key_value = config.key_value_width();
    let mut declared_layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let layer = layers.view(&index.to_string());
        let attention = layer.view("self_attn");
        let mlp = layer.view("mlp");
        declared_layers.push(DecoderLayer {
            input_layernorm: norm(&layer.view("input_layernorm"), hidden)?,
            self_attn: Attention {
                q_proj: linear(&attention.view("q_proj"), query, hidden)?,
                k_proj: linear(&attention.view("k_proj"), key_value, hidden)?,
                v_proj: linear(&attention.view("v_proj"), key_value, hidden)?,
                o_proj: linear(&attention.view("o_proj"), hidden, query)?,
                q_norm: norm(&attention.view("q_norm"), config.head_dim)?,
                k_norm: norm(&attention.view("k_norm"), config.head_dim)?,
            },
            post_attention_layernorm: norm(&layer.view("post_attention_layernorm"), hidden)?,
            mlp: Mlp {
                gate_proj: linear(&mlp.view("gate_proj"), config.intermediate_size, hidden)?,
                up_proj: linear(&mlp.view("up_proj"), config.intermediate_size, hidden)?,
                down_proj: linear(&mlp.view("down_proj"), hidden, config.intermediate_size)?,
            },
        });
    }
    Ok(Checkpoint {
        model: Transformer {
            embed_tokens: Linear {
                weight: nml_result(model.dense(
                    "embed_tokens.weight",
                    shape(&[config.vocab_size, hidden])?,
                    &[],
                ))?,
            },
            layers: declared_layers,
            norm: norm(&model.view("norm"), hidden)?,
        },
    })
}

pub(crate) fn build_graph(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    sequence: usize,
    kind: GraphKind,
) -> Result<GraphOutputs> {
    if sequence == 0 {
        return Err(Error::contract(
            "Qwen3 graphs require a nonempty token sequence",
        ));
    }
    let batch = match kind {
        GraphKind::Prefill { batch, .. } | GraphKind::Decode { batch, .. } => batch,
    };
    if batch != 1 {
        return Err(Error::contract(
            "Qwen3 compatibility graphs currently require batch capacity one",
        ));
    }
    let token_shape = nml_result(Shape::new(
        DataType::I32,
        &[dimension(batch)?, dimension(sequence)?],
    ))?;
    let tokens = graph.input("tokens", token_shape);
    let positions = match kind {
        GraphKind::Prefill { .. } => nml_result(graph.iota(token_shape, 1))?,
        GraphKind::Decode { .. } => {
            let position = graph.input("position", nml_result(Shape::new(DataType::I32, &[]))?);
            nml_result(graph.reshape(position, token_shape))?
        }
    };
    let cache_capacity = match kind {
        GraphKind::Prefill { capacity, .. } | GraphKind::Decode { capacity, .. } => capacity,
    };
    if cache_capacity < sequence {
        return Err(Error::contract(
            "cache capacity is smaller than graph sequence",
        ));
    }

    let cache_shape = shape(&[
        1,
        cache_capacity,
        config.num_key_value_heads,
        config.head_dim,
    ])?;
    let zero = nml_result(graph.scalar(0i32))?;
    let mut hidden =
        nml_result(graph.token_embedding(&checkpoint.model.embed_tokens.weight, tokens))?;
    let mut cache_outputs = Vec::with_capacity(config.num_hidden_layers);
    for (index, layer) in checkpoint.model.layers.iter().enumerate() {
        let key_name = format!("cache.{index}.key");
        let value_name = format!("cache.{index}.value");
        let key_input = graph.input(&key_name, cache_shape);
        let value_input = graph.input(&value_name, cache_shape);

        let residual = hidden;
        let input_norm = nml_result(graph.parameter_value(&layer.input_layernorm.weight))?;
        hidden = nml_result(graph.rms_norm(hidden, Some(input_norm), 2, config.rms_norm_eps))?;
        let query = nml_result(graph.linear(hidden, &layer.self_attn.q_proj.weight, None))?;
        let key = nml_result(graph.linear(hidden, &layer.self_attn.k_proj.weight, None))?;
        let value = nml_result(graph.linear(hidden, &layer.self_attn.v_proj.weight, None))?;
        let query = nml_result(graph.reshape(
            query,
            shape(&[1, sequence, config.num_attention_heads, config.head_dim])?,
        ))?;
        let key = nml_result(graph.reshape(
            key,
            shape(&[1, sequence, config.num_key_value_heads, config.head_dim])?,
        ))?;
        let value = nml_result(graph.reshape(
            value,
            shape(&[1, sequence, config.num_key_value_heads, config.head_dim])?,
        ))?;
        let query_norm = nml_result(graph.parameter_value(&layer.self_attn.q_norm.weight))?;
        let query = nml_result(graph.rms_norm(query, Some(query_norm), 3, config.rms_norm_eps))?;
        let key_norm = nml_result(graph.parameter_value(&layer.self_attn.k_norm.weight))?;
        let key = nml_result(graph.rms_norm(key, Some(key_norm), 3, config.rms_norm_eps))?;
        let rope = nml::attention::RopeOptions {
            base: config.rope_theta,
            rotary_dimensions: config.head_dim,
            layout: nml::attention::RopeLayout::Sequential,
            scaling: nml::attention::RopeScaling::Default,
        };
        let query = nml_result(graph.rope(query, positions, rope))?;
        let key = nml_result(graph.rope(key, positions, rope))?;

        let cache_start = match kind {
            GraphKind::Prefill { .. } => zero,
            GraphKind::Decode { .. } => {
                // The decode graph's scalar input was reshaped into
                // `positions`; reshape it back without creating another input.
                nml_result(graph.reshape(positions, nml_result(Shape::new(DataType::I32, &[]))?))?
            }
        };
        let key_cache = nml_result(graph.dynamic_update_slice(
            key_input,
            key,
            &[zero, cache_start, zero, zero],
        ))?;
        let value_cache = nml_result(graph.dynamic_update_slice(
            value_input,
            value,
            &[zero, cache_start, zero, zero],
        ))?;
        let key_cache = nml_result(graph.reuse_buffer(key_cache, key_input))?;
        let value_cache = nml_result(graph.reuse_buffer(value_cache, value_input))?;

        let attention = match kind {
            GraphKind::Prefill { .. } => nml_result(graph.attention(
                query,
                key,
                value,
                positions,
                positions,
                None,
                nml::attention::Options::default(),
            ))?,
            GraphKind::Decode { capacity, .. } => {
                let key_positions = nml_result(graph.iota(
                    nml_result(Shape::new(DataType::I32, &[1, dimension(capacity)?]))?,
                    1,
                ))?;
                nml_result(graph.attention(
                    query,
                    key_cache,
                    value_cache,
                    positions,
                    key_positions,
                    None,
                    nml::attention::Options::default(),
                ))?
            }
        };
        let attention =
            nml_result(graph.reshape(attention, shape(&[1, sequence, config.query_width()])?))?;
        let attention = nml_result(graph.linear(attention, &layer.self_attn.o_proj.weight, None))?;
        hidden = nml_result(graph.add(residual, attention))?;

        let residual = hidden;
        let post_attention_norm =
            nml_result(graph.parameter_value(&layer.post_attention_layernorm.weight))?;
        hidden =
            nml_result(graph.rms_norm(hidden, Some(post_attention_norm), 2, config.rms_norm_eps))?;
        let gate = nml_result(graph.linear(hidden, &layer.mlp.gate_proj.weight, None))?;
        let up = nml_result(graph.linear(hidden, &layer.mlp.up_proj.weight, None))?;
        let gate = nml_result(graph.silu(gate))?;
        let gated = nml_result(graph.multiply(gate, up))?;
        let mlp = nml_result(graph.linear(gated, &layer.mlp.down_proj.weight, None))?;
        hidden = nml_result(graph.add(residual, mlp))?;
        cache_outputs.push((key_cache, value_cache));
    }

    let final_norm = nml_result(graph.parameter_value(&checkpoint.model.norm.weight))?;
    hidden = nml_result(graph.rms_norm(hidden, Some(final_norm), 2, config.rms_norm_eps))?;
    let last = nml_result(graph.slice(
        hidden,
        &[0, dimension(sequence - 1)?, 0],
        &[1, dimension(sequence)?, dimension(config.hidden_size)?],
        &[1, 1, 1],
    ))?;
    // The checkpoint redundantly stores `lm_head.weight`, but tied Qwen3
    // semantics require the exact embedding buffer. Reusing this symbol saves
    // one 311 MB upload for the 0.6B checkpoint and prevents silent drift.
    let logits = nml_result(graph.linear(last, &checkpoint.model.embed_tokens.weight, None))?;
    let (_, token) = nml_result(graph.argmax(logits, 2))?;
    Ok(GraphOutputs {
        token,
        caches: cache_outputs,
    })
}

fn linear(parameters: &ParameterSet, output: usize, input: usize) -> Result<Linear> {
    Ok(Linear {
        weight: nml_result(parameters.dense("weight", shape(&[output, input])?, &[]))?,
    })
}

fn norm(parameters: &ParameterSet, width: usize) -> Result<Norm> {
    Ok(Norm {
        weight: nml_result(parameters.dense("weight", shape(&[width])?, &[]))?,
    })
}

fn shape(dimensions: &[usize]) -> Result<Shape> {
    let dimensions = dimensions
        .iter()
        .map(|dimension| dimension_i64(*dimension))
        .collect::<Result<Vec<_>>>()?;
    nml_result(Shape::new(DataType::Bf16, &dimensions))
}

fn dimension(value: usize) -> Result<i64> {
    dimension_i64(value)
}

fn dimension_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::contract("Qwen3 dimension exceeds I64"))
}
