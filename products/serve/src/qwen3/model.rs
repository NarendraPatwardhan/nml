//! Qwen3 graph construction.

use super::{Error, Result, config::Config, nml_result};
use nml::io::TensorStore;
use nml::{DataType, NmlStruct, Shape, Tensor};

#[derive(NmlStruct)]
struct Linear {
    weight: Tensor,
}

#[derive(NmlStruct)]
struct Norm {
    weight: Tensor,
}

#[derive(NmlStruct)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Norm,
    k_norm: Norm,
}

#[derive(NmlStruct)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

#[derive(NmlStruct)]
struct DecoderLayer {
    input_layernorm: Norm,
    self_attn: Attention,
    post_attention_layernorm: Norm,
    mlp: Mlp,
}

#[derive(NmlStruct)]
struct Transformer {
    embed_tokens: Linear,
    layers: Vec<DecoderLayer>,
    norm: Norm,
}

#[derive(NmlStruct)]
pub(crate) struct Checkpoint {
    model: Transformer,
}

#[derive(Clone, Copy)]
pub(crate) enum GraphKind {
    Prefill { capacity: usize },
    Decode { capacity: usize },
}

pub(crate) struct GraphOutputs {
    pub(crate) token: Tensor,
    pub(crate) caches: Vec<(Tensor, Tensor)>,
}

pub(crate) fn declare(store: &TensorStore, config: &Config) -> Result<Checkpoint> {
    let model = store.view("model");
    let layers = model.view("layers");
    let hidden = config.hidden_size;
    let query = config.query_width();
    let key_value = config.key_value_width();
    let mut declared_layers = Vec::with_capacity(config.num_hidden_layers);
    for index in 0..config.num_hidden_layers {
        let layer = layers.layer(index);
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
                weight: nml_result(model.tensor(
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
    store: &TensorStore,
    checkpoint: &Checkpoint,
    config: &Config,
    sequence: usize,
    kind: GraphKind,
) -> Result<GraphOutputs> {
    if sequence == 0 {
        return Err(Error::Contract(
            "Qwen3 graphs require a nonempty token sequence",
        ));
    }
    let token_shape = nml_result(Shape::new(DataType::I32, &[1, dimension(sequence)?]))?;
    let tokens = store.activation("tokens", token_shape);
    let positions = match kind {
        GraphKind::Prefill { .. } => nml_result(store.iota(token_shape, 1))?,
        GraphKind::Decode { .. } => {
            let position =
                store.activation("position", nml_result(Shape::new(DataType::I32, &[]))?);
            nml_result(store.reshape(position, token_shape))?
        }
    };
    let cache_capacity = match kind {
        GraphKind::Prefill { capacity } | GraphKind::Decode { capacity } => capacity,
    };
    if cache_capacity < sequence {
        return Err(Error::Contract(
            "cache capacity is smaller than graph sequence",
        ));
    }

    let cache_shape = shape(&[
        1,
        cache_capacity,
        config.num_key_value_heads,
        config.head_dim,
    ])?;
    let zero = nml_result(store.scalar(0i32))?;
    let mut hidden =
        nml_result(store.token_embedding(checkpoint.model.embed_tokens.weight, tokens))?;
    let mut cache_outputs = Vec::with_capacity(config.num_hidden_layers);
    for (index, layer) in checkpoint.model.layers.iter().enumerate() {
        let key_name = format!("cache.{index}.key");
        let value_name = format!("cache.{index}.value");
        let key_input = store.activation(&key_name, cache_shape);
        let value_input = store.activation(&value_name, cache_shape);

        let residual = hidden;
        hidden = nml_result(store.rms_norm(
            hidden,
            Some(layer.input_layernorm.weight),
            2,
            config.rms_norm_eps,
        ))?;
        let query = nml_result(store.linear(hidden, layer.self_attn.q_proj.weight, None))?;
        let key = nml_result(store.linear(hidden, layer.self_attn.k_proj.weight, None))?;
        let value = nml_result(store.linear(hidden, layer.self_attn.v_proj.weight, None))?;
        let query = nml_result(store.reshape(
            query,
            shape(&[1, sequence, config.num_attention_heads, config.head_dim])?,
        ))?;
        let key = nml_result(store.reshape(
            key,
            shape(&[1, sequence, config.num_key_value_heads, config.head_dim])?,
        ))?;
        let value = nml_result(store.reshape(
            value,
            shape(&[1, sequence, config.num_key_value_heads, config.head_dim])?,
        ))?;
        let query = nml_result(store.rms_norm(
            query,
            Some(layer.self_attn.q_norm.weight),
            3,
            config.rms_norm_eps,
        ))?;
        let key = nml_result(store.rms_norm(
            key,
            Some(layer.self_attn.k_norm.weight),
            3,
            config.rms_norm_eps,
        ))?;
        let rope = nml::attention::RopeOptions {
            base: config.rope_theta,
            rotary_dimensions: config.head_dim,
            layout: nml::attention::RopeLayout::Sequential,
            scaling: nml::attention::RopeScaling::Default,
        };
        let query = nml_result(store.rope(query, positions, rope))?;
        let key = nml_result(store.rope(key, positions, rope))?;

        let cache_start = match kind {
            GraphKind::Prefill { .. } => zero,
            GraphKind::Decode { .. } => {
                // The decode graph's scalar input was reshaped into
                // `positions`; reshape it back without creating another input.
                nml_result(store.reshape(positions, nml_result(Shape::new(DataType::I32, &[]))?))?
            }
        };
        let key_cache = nml_result(store.dynamic_update_slice(
            key_input,
            key,
            &[zero, cache_start, zero, zero],
        ))?;
        let value_cache = nml_result(store.dynamic_update_slice(
            value_input,
            value,
            &[zero, cache_start, zero, zero],
        ))?;
        let key_cache = nml_result(store.reuse_buffer(key_cache, key_input))?;
        let value_cache = nml_result(store.reuse_buffer(value_cache, value_input))?;

        let attention = match kind {
            GraphKind::Prefill { .. } => nml_result(store.attention(
                query,
                key,
                value,
                positions,
                positions,
                nml::attention::Options::default(),
            ))?,
            GraphKind::Decode { capacity } => {
                let key_positions = nml_result(store.iota(
                    nml_result(Shape::new(DataType::I32, &[1, dimension(capacity)?]))?,
                    1,
                ))?;
                nml_result(store.attention(
                    query,
                    key_cache,
                    value_cache,
                    positions,
                    key_positions,
                    nml::attention::Options::default(),
                ))?
            }
        };
        let attention =
            nml_result(store.reshape(attention, shape(&[1, sequence, config.query_width()])?))?;
        let attention = nml_result(store.linear(attention, layer.self_attn.o_proj.weight, None))?;
        hidden = nml_result(store.add(residual, attention))?;

        let residual = hidden;
        hidden = nml_result(store.rms_norm(
            hidden,
            Some(layer.post_attention_layernorm.weight),
            2,
            config.rms_norm_eps,
        ))?;
        let gate = nml_result(store.linear(hidden, layer.mlp.gate_proj.weight, None))?;
        let up = nml_result(store.linear(hidden, layer.mlp.up_proj.weight, None))?;
        let gate = nml_result(store.silu(gate))?;
        let gated = nml_result(store.multiply(gate, up))?;
        let mlp = nml_result(store.linear(gated, layer.mlp.down_proj.weight, None))?;
        hidden = nml_result(store.add(residual, mlp))?;
        cache_outputs.push((key_cache, value_cache));
    }

    hidden = nml_result(store.rms_norm(
        hidden,
        Some(checkpoint.model.norm.weight),
        2,
        config.rms_norm_eps,
    ))?;
    let last = nml_result(store.slice(
        hidden,
        &[0, dimension(sequence - 1)?, 0],
        &[1, dimension(sequence)?, dimension(config.hidden_size)?],
        &[1, 1, 1],
    ))?;
    // The checkpoint redundantly stores `lm_head.weight`, but tied Qwen3
    // semantics require the exact embedding buffer. Reusing this symbol saves
    // one 311 MB upload for the 0.6B checkpoint and prevents silent drift.
    let logits = nml_result(store.linear(last, checkpoint.model.embed_tokens.weight, None))?;
    let (_, token) = nml_result(store.argmax(logits, 2))?;
    Ok(GraphOutputs {
        token,
        caches: cache_outputs,
    })
}

fn linear(store: &TensorStore, output: usize, input: usize) -> Result<Linear> {
    Ok(Linear {
        weight: nml_result(store.tensor("weight", shape(&[output, input])?, &[]))?,
    })
}

fn norm(store: &TensorStore, width: usize) -> Result<Norm> {
    Ok(Norm {
        weight: nml_result(store.tensor("weight", shape(&[width])?, &[]))?,
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
    i64::try_from(value).map_err(|_| Error::Contract("Qwen3 dimension exceeds I64"))
}
