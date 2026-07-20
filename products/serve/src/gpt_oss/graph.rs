//! Bounded GPT-OSS component graphs.
//!
//! The compiler sees one bounded fusion domain at a time: embedding, one full
//! prefill layer, or final projection and sampling. Decode uses a single fused
//! graph covering all 24 transformer layers, embedding, and the sampling head
//! to collapse recurring CUDA graph launch overhead. Prefill retains single-
//! layer components because each prefill step touches every KV cache exactly
//! once per layer and does not repeat per token.

use super::checkpoint::{message, BoxError, Checkpoint, DecoderLayer, Result};
use super::config::{AttentionKind, Config};
use nml::{DataType, Graph, Shape, Tensor};

// Sixteen-token physical pages provide useful allocation granularity for a
// future shared cache arena. CUDA compute-tile width is an independent kernel
// policy, so changing this product allocation constant cannot by itself create
// a wider, spill-heavy attention specialization.
pub(super) const CACHE_PAGE_SIZE: usize = 16;
pub(super) const MAXIMUM_TOP_K: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ShapeFamily {
    phase: Phase,
    sequence: usize,
    cache_capacity: usize,
}

impl ShapeFamily {
    pub(super) fn prefill(sequence: usize, cache_capacity: usize) -> Result<Self> {
        Self::new(Phase::Prefill, sequence, cache_capacity)
    }

    pub(super) fn decode(cache_capacity: usize) -> Result<Self> {
        Self::new(Phase::Decode, 1, cache_capacity)
    }

    fn new(phase: Phase, sequence: usize, cache_capacity: usize) -> Result<Self> {
        if sequence == 0 || cache_capacity == 0 {
            return Err(message("GPT-OSS execution dimensions must be nonzero"));
        }
        if sequence > cache_capacity {
            return Err(message("GPT-OSS prefill bucket exceeds cache capacity"));
        }
        if cache_capacity > i32::MAX as usize {
            return Err(message(
                "GPT-OSS cache capacity exceeds the I32 index domain",
            ));
        }
        if !cache_capacity.is_multiple_of(CACHE_PAGE_SIZE) {
            return Err(message(
                "GPT-OSS cache capacity must contain complete pages",
            ));
        }
        Ok(Self {
            phase,
            sequence,
            cache_capacity,
        })
    }

    pub(super) const fn phase(self) -> Phase {
        self.phase
    }

    pub(super) const fn sequence(self) -> usize {
        self.sequence
    }

    pub(super) const fn cache_capacity(self) -> usize {
        self.cache_capacity
    }

    pub(super) const fn page_count(self) -> usize {
        self.cache_capacity / CACHE_PAGE_SIZE
    }
}

pub(super) fn build_embedding(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    let tokens = graph.input(
        "tokens",
        shape(DataType::I32, &[1, dimension(family.sequence())?])?,
    );
    let hidden = nml(graph.token_embedding(&checkpoint.model.embed_tokens.weight, tokens))?;
    require_shape(
        hidden,
        DataType::Bf16,
        &[
            1,
            dimension(family.sequence())?,
            dimension(config.hidden_size())?,
        ],
    )?;
    Ok(vec![("hidden".to_owned(), hidden)])
}

pub(super) fn build_layer(
    graph: &mut Graph,
    layer: &DecoderLayer,
    config: &Config,
    family: ShapeFamily,
    attention_kind: AttentionKind,
) -> Result<Vec<(String, Tensor)>> {
    let hidden_shape = hidden_shape(config, family)?;
    let cache_shape = cache_shape(config, family)?;
    let hidden_input = graph.input("hidden", hidden_shape);
    let position = graph.input("position", shape(DataType::I32, &[])?);
    let sequence_lengths = sequence_lengths(graph, family, position)?;
    let page_table = graph.input("page_table", page_table_shape(family)?);
    let key_input = graph.input("cache.key", cache_shape);
    let value_input = graph.input("cache.value", cache_shape);
    let (hidden, key_cache, value_cache) = apply_layer(
        graph,
        layer,
        config,
        family,
        attention_kind,
        hidden_input,
        position,
        sequence_lengths,
        page_table,
        key_input,
        value_input,
    )?;
    let hidden = nml(graph.reuse_buffer(hidden, hidden_input))?;

    Ok(vec![
        ("hidden".to_owned(), hidden),
        ("cache.key".to_owned(), key_cache),
        ("cache.value".to_owned(), value_cache),
    ])
}

/// Builds the one decode fusion domain justified by GPT-OSS's immutable
/// alternating schedule: one sliding layer followed by one full layer.
///
/// This halves recurring PJRT graph submissions without making model depth a
/// compiler constant. The pair shares position and page-table inputs, owns two
/// independent donated cache pairs, and aliases the final hidden state back to
/// the original input. Prefill deliberately retains single-layer components.
// Kept for unit tests; production uses the fused `build_decode_full` instead.
#[allow(dead_code)]
pub(super) fn build_decode_layer_pair(
    graph: &mut Graph,
    sliding: &DecoderLayer,
    full: &DecoderLayer,
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    if family.phase() != Phase::Decode {
        return Err(message("GPT-OSS layer pairs are decode-only"));
    }
    let hidden_shape = hidden_shape(config, family)?;
    let cache_shape = cache_shape(config, family)?;
    let hidden_input = graph.input("hidden", hidden_shape);
    let position = graph.input("position", shape(DataType::I32, &[])?);
    let sequence_lengths = sequence_lengths(graph, family, position)?;
    let page_table = graph.input("page_table", page_table_shape(family)?);
    let sliding_key_input = graph.input("sliding.cache.key", cache_shape);
    let sliding_value_input = graph.input("sliding.cache.value", cache_shape);
    let full_key_input = graph.input("full.cache.key", cache_shape);
    let full_value_input = graph.input("full.cache.value", cache_shape);

    let (hidden, sliding_key, sliding_value) = apply_layer(
        graph,
        sliding,
        config,
        family,
        AttentionKind::SlidingAttention,
        hidden_input,
        position,
        sequence_lengths,
        page_table,
        sliding_key_input,
        sliding_value_input,
    )?;
    let (hidden, full_key, full_value) = apply_layer(
        graph,
        full,
        config,
        family,
        AttentionKind::FullAttention,
        hidden,
        position,
        sequence_lengths,
        page_table,
        full_key_input,
        full_value_input,
    )?;
    let hidden = nml(graph.reuse_buffer(hidden, hidden_input))?;
    Ok(vec![
        ("hidden".to_owned(), hidden),
        ("sliding.cache.key".to_owned(), sliding_key),
        ("sliding.cache.value".to_owned(), sliding_value),
        ("full.cache.key".to_owned(), full_key),
        ("full.cache.value".to_owned(), full_value),
    ])
}

/// Builds a single fused graph for the entire decode pipeline: embedding, all
/// 24 alternating layers, and the sampling head. This collapses the per-token
/// launch overhead from 14 CUDA graphs to 1, eliminating ~1.5 ms of recurring
/// cuGraphLaunch, cuGraphExecKernelNodeSetParams_v2, and PJRT dispatch cost.
///
/// Unlike `build_decode_layer_pair` this makes model depth a compiler constant
/// — the returned executable covers every transformer layer in one StableHLO
/// program. KV caches use per-layer input/output slots named `cache.{i}.key`
/// and `cache.{i}.value` so the runtime can donate each physical page table
/// independently.
pub fn build_decode_full(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    if family.phase() != Phase::Decode {
        return Err(message("GPT-OSS full decode graph is decode-only"));
    }
    let hidden_size = dimension(config.hidden_size())?;
    let cache_shape = cache_shape(config, family)?;
    let zero = nml(graph.scalar(0_i32))?;
    let one = nml(graph.scalar(1_i32))?;

    // ---- Embedding ----
    let tokens = graph.input(
        "tokens",
        shape(DataType::I32, &[1, dimension(family.sequence())?])?,
    );
    let mut hidden = nml(graph.token_embedding(&checkpoint.model.embed_tokens.weight, tokens))?;

    // ---- Shared layer inputs ----
    let position = graph.input("position", shape(DataType::I32, &[])?);
    let page_table = graph.input("page_table", page_table_shape(family)?);

    // Pre-compute positions once (XLA will CSE repeated computation anyway,
    // but computing it here is cleaner and avoids multiple iota/broadcast ops).
    let sequence_lengths = sequence_lengths(graph, family, position)?;

    // ---- Per-layer KV cache inputs and outputs ----
    // Create all 48 cache input slots upfront, then thread them through
    // apply_layer and collect the donated output tensors for the result tuple.
    let mut key_outputs: Vec<Tensor> = Vec::with_capacity(config.layers());
    let mut value_outputs: Vec<Tensor> = Vec::with_capacity(config.layers());

    for (i, layer) in checkpoint.model.layers.iter().enumerate() {
        let key_input = graph.input(&format!("cache.{i}.key"), cache_shape);
        let value_input = graph.input(&format!("cache.{i}.value"), cache_shape);

        let attention_kind = config.layer_types()[i];
        let (new_hidden, new_key, new_value) = apply_layer(
            graph,
            layer,
            config,
            family,
            attention_kind,
            hidden,
            position,
            sequence_lengths,
            page_table,
            key_input,
            value_input,
        )?;
        hidden = new_hidden;
        key_outputs.push(new_key);
        value_outputs.push(new_value);
    }

    // ---- Head (final norm, lm_head linear, sampling, position++) ----
    let last_index = graph.input("last_index", shape(DataType::I32, &[])?);
    let sampling_state_input = graph.input("sampling_state", shape(DataType::U64, &[2])?);
    let top_k = graph.input("top_k", shape(DataType::I32, &[])?);
    let temperature = graph.input("temperature", shape(DataType::F32, &[])?);
    let top_p = graph.input("top_p", shape(DataType::F32, &[])?);
    let min_p = graph.input("min_p", shape(DataType::F32, &[])?);

    let last = nml(graph.dynamic_slice(hidden, &[zero, last_index, zero], &[1, 1, hidden_size]))?;
    let final_norm = nml(graph.parameter_value(&checkpoint.model.norm.weight))?;
    let last = nml(graph.rms_norm(last, Some(final_norm), 2, config.rms_norm_epsilon()))?;
    let last = nml(graph.reshape(last, shape(DataType::Bf16, &[1, hidden_size])?))?;
    let logits = nml(graph.linear(last, &checkpoint.lm_head.weight, None))?;
    let sampling_state = nml(graph.random_state(sampling_state_input))?;
    let (sampling_state, token) = nml(graph.sample_tokens_dynamic(
        logits,
        sampling_state,
        1,
        top_k,
        temperature,
        top_p,
        min_p,
        MAXIMUM_TOP_K,
    ))?;
    let sampling_state =
        nml(graph.reuse_buffer(sampling_state.into_tensor(), sampling_state_input))?;
    let token = nml(graph.reshape(token, shape(DataType::I32, &[1, 1])?))?;
    let position_out = nml(graph.add(position, one))?;

    // ---- Collect outputs in execution order ----
    let mut outputs: Vec<(String, Tensor)> = vec![
        ("token".to_owned(), token),
        ("sampling_state".to_owned(), sampling_state),
        ("position".to_owned(), position_out),
    ];
    for i in 0..config.layers() {
        outputs.push((format!("cache.{i}.key"), key_outputs[i].clone()));
        outputs.push((format!("cache.{i}.value"), value_outputs[i].clone()));
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
fn apply_layer(
    graph: &mut Graph,
    layer: &DecoderLayer,
    config: &Config,
    family: ShapeFamily,
    attention_kind: AttentionKind,
    hidden_input: Tensor,
    position: Tensor,
    sequence_lengths: Tensor,
    page_table: Tensor,
    key_input: Tensor,
    value_input: Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let sequence = dimension(family.sequence())?;
    let hidden_size = dimension(config.hidden_size())?;
    let query_heads = dimension(config.query_heads())?;
    let key_value_heads = dimension(config.key_value_heads())?;
    let head_dim = dimension(config.head_dim())?;
    let page_count = dimension(family.page_count())?;
    let page_size = dimension(CACHE_PAGE_SIZE)?;
    let hidden_shape = hidden_shape(config, family)?;
    let cache_shape = shape(
        DataType::Bf16,
        &[page_count, page_size, key_value_heads, head_dim],
    )?;
    let zero = nml(graph.scalar(0_i32))?;

    let token_shape = shape(DataType::I32, &[1, sequence])?;
    let offsets = nml(graph.iota(token_shape, 1))?;
    let position_vector = nml(graph.broadcast_in_dim(position, token_shape, &[]))?;
    let positions = nml(graph.add(offsets, position_vector))?;

    let residual = hidden_input;
    let input_norm = nml(graph.parameter_value(&layer.input_layernorm.weight))?;
    let mut hidden =
        nml(graph.rms_norm(hidden_input, Some(input_norm), 2, config.rms_norm_epsilon()))?;
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
        shape(DataType::Bf16, &[1, sequence, query_heads, head_dim])?,
    ))?;
    let key = nml(graph.reshape(
        key,
        shape(DataType::Bf16, &[1, sequence, key_value_heads, head_dim])?,
    ))?;
    let value = nml(graph.reshape(
        value,
        shape(DataType::Bf16, &[1, sequence, key_value_heads, head_dim])?,
    ))?;
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
    let query = nml(graph.rope(query, positions, rope))?;
    let key = nml(graph.rope(key, positions, rope))?;

    let dense_cache_shape = shape(
        DataType::Bf16,
        &[
            1,
            dimension(family.cache_capacity())?,
            key_value_heads,
            head_dim,
        ],
    )?;
    let dense_key = nml(graph.reshape(key_input, dense_cache_shape))?;
    let dense_value = nml(graph.reshape(value_input, dense_cache_shape))?;
    let dense_key = nml(graph.dynamic_update_slice(dense_key, key, &[zero, position, zero, zero]))?;
    let dense_value =
        nml(graph.dynamic_update_slice(dense_value, value, &[zero, position, zero, zero]))?;
    let key_cache = nml(graph.reshape(dense_key, cache_shape))?;
    let value_cache = nml(graph.reshape(dense_value, cache_shape))?;
    let key_cache = nml(graph.reuse_buffer(key_cache, key_input))?;
    let value_cache = nml(graph.reuse_buffer(value_cache, value_input))?;

    let sinks = nml(graph.parameter_value(&layer.self_attn.sinks))?;
    let sliding_window = match attention_kind {
        AttentionKind::SlidingAttention => Some(config.sliding_window()),
        AttentionKind::FullAttention => None,
    };
    let attention = nml(graph.paged_attention(
        query,
        key_cache,
        value_cache,
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
        shape(
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
    let routed = nml(graph.reshape(hidden, shape(DataType::Bf16, &[sequence, hidden_size])?))?;
    let router_logits = nml(graph.linear(
        routed,
        &layer.mlp.router.weight,
        Some(&layer.mlp.router.bias),
    ))?;
    let routed = nml(graph.routed_clamped_swiglu(
        routed,
        router_logits,
        &layer.mlp.experts.gate_up_proj,
        &layer.mlp.experts.gate_up_proj_bias,
        &layer.mlp.experts.down_proj,
        &layer.mlp.experts.down_proj_bias,
        config.experts_per_token(),
    ))?;
    let routed = nml(graph.reshape(routed, hidden_shape))?;
    hidden = nml(graph.add(residual, routed))?;
    Ok((hidden, key_cache, value_cache))
}

pub(super) fn build_head(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    let sequence = dimension(family.sequence())?;
    let hidden_size = dimension(config.hidden_size())?;
    let hidden = graph.input(
        "hidden",
        shape(DataType::Bf16, &[1, sequence, hidden_size])?,
    );
    let last_index = graph.input("last_index", shape(DataType::I32, &[])?);
    let sampling_state_input = graph.input("sampling_state", shape(DataType::U64, &[2])?);
    let top_k = graph.input("top_k", shape(DataType::I32, &[])?);
    let temperature = graph.input("temperature", shape(DataType::F32, &[])?);
    let top_p = graph.input("top_p", shape(DataType::F32, &[])?);
    let min_p = graph.input("min_p", shape(DataType::F32, &[])?);
    let zero = nml(graph.scalar(0_i32))?;
    let last = nml(graph.dynamic_slice(hidden, &[zero, last_index, zero], &[1, 1, hidden_size]))?;
    let final_norm = nml(graph.parameter_value(&checkpoint.model.norm.weight))?;
    let last = nml(graph.rms_norm(last, Some(final_norm), 2, config.rms_norm_epsilon()))?;
    let last = nml(graph.reshape(last, shape(DataType::Bf16, &[1, hidden_size])?))?;
    let logits = nml(graph.linear(last, &checkpoint.lm_head.weight, None))?;
    let sampling_state = nml(graph.random_state(sampling_state_input))?;
    let (sampling_state, token) = nml(graph.sample_tokens_dynamic(
        logits,
        sampling_state,
        1,
        top_k,
        temperature,
        top_p,
        min_p,
        MAXIMUM_TOP_K,
    ))?;
    let sampling_state =
        nml(graph.reuse_buffer(sampling_state.into_tensor(), sampling_state_input))?;
    let token = nml(graph.reshape(token, shape(DataType::I32, &[1, 1])?))?;
    let mut outputs = vec![
        ("token".to_owned(), token),
        ("sampling_state".to_owned(), sampling_state),
    ];
    if family.phase() == Phase::Decode {
        let position = graph.input("position", shape(DataType::I32, &[])?);
        let one = nml(graph.scalar(1_i32))?;
        outputs.push(("position".to_owned(), nml(graph.add(position, one))?));
    }
    Ok(outputs)
}

pub(super) fn cache_shape(config: &Config, family: ShapeFamily) -> Result<Shape> {
    shape(
        DataType::Bf16,
        &[
            dimension(family.page_count())?,
            dimension(CACHE_PAGE_SIZE)?,
            dimension(config.key_value_heads())?,
            dimension(config.head_dim())?,
        ],
    )
}

fn hidden_shape(config: &Config, family: ShapeFamily) -> Result<Shape> {
    shape(
        DataType::Bf16,
        &[
            1,
            dimension(family.sequence())?,
            dimension(config.hidden_size())?,
        ],
    )
}

fn sequence_lengths(graph: &mut Graph, family: ShapeFamily, position: Tensor) -> Result<Tensor> {
    match family.phase() {
        Phase::Prefill => Ok(graph.input("sequence_lengths", shape(DataType::I32, &[1])?)),
        Phase::Decode => {
            let one = nml(graph.scalar(1_i32))?;
            let length = nml(graph.add(position, one))?;
            nml(graph.reshape(length, shape(DataType::I32, &[1])?))
        }
    }
}

pub(super) fn page_table_shape(family: ShapeFamily) -> Result<Shape> {
    shape(DataType::I32, &[1, dimension(family.page_count())?])
}

fn require_shape(tensor: Tensor, dtype: DataType, dimensions: &[i64]) -> Result<()> {
    if tensor.shape().dtype() != dtype || tensor.shape().dimensions() != dimensions {
        return Err(message("GPT-OSS component produced an unexpected shape"));
    }
    Ok(())
}

fn dimension(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| message("GPT-OSS dimension exceeds I64"))
}

fn shape(dtype: DataType, dimensions: &[i64]) -> Result<Shape> {
    Shape::new(dtype, dimensions).map_err(|error| Box::new(error) as BoxError)
}

fn nml<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: StdError + Send + Sync + 'static,
{
    result.map_err(|error| Box::new(error) as BoxError)
}

use std::error::Error as StdError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt_oss::checkpoint::{declare_with, representative_layer};
    use std::path::Path;

    macro_rules! finish {
        ($build:expr) => {{
            let mut graph = Graph::new();
            let outputs = $build(&mut graph).unwrap();
            graph.finish_named(&outputs).unwrap()
        }};
    }

    macro_rules! assert_contract {
        ($program:expr, $activations:expr, $parameters:expr, $outputs:expr, $aliases:expr $(,)?) => {{
            let program = &$program;
            assert_eq!(
                program
                    .inputs()
                    .filter(|(_, _, binding)| !binding.is_parameter_component())
                    .count(),
                $activations,
            );
            assert_eq!(
                program
                    .inputs()
                    .filter(|(_, _, binding)| binding.is_parameter_component())
                    .count(),
                $parameters,
            );
            assert_eq!(program.outputs().count(), $outputs);
            assert_eq!(program.output_aliases().collect::<Vec<_>>(), $aliases);
        }};
    }

    macro_rules! assert_single_layer_identity {
        ($program:expr, $layer:expr) => {{
            let expected = format!("model.layers.{}.", $layer);
            let layer_names = $program
                .input_names()
                .filter(|name| name.starts_with("model.layers."))
                .collect::<Vec<_>>();
            assert!(!layer_names.is_empty());
            assert!(layer_names.iter().all(|name| name.starts_with(&expected)));
        }};
    }

    #[test]
    fn component_programs_keep_model_depth_out_of_the_compiler_abi() {
        let config = selected_config();
        let checkpoint = synthetic_checkpoint(&config);
        let prefill = ShapeFamily::prefill(32, 512).unwrap();
        let decode = ShapeFamily::decode(512).unwrap();

        let embedding = finish!(|graph| build_embedding(graph, &checkpoint, &config, prefill));
        assert_contract!(embedding, 1, 3, 1, &[None]);

        let sliding =
            representative_layer(&checkpoint, &config, AttentionKind::SlidingAttention).unwrap();
        let prefill_layer = finish!(|graph| {
            build_layer(
                graph,
                sliding,
                &config,
                prefill,
                AttentionKind::SlidingAttention,
            )
        });
        assert_contract!(prefill_layer, 6, 29, 3, &[Some(0), Some(4), Some(5)],);
        assert!(prefill_layer
            .input_names()
            .any(|name| name == "sequence_lengths"));
        assert_single_layer_identity!(prefill_layer, 0);

        let full =
            representative_layer(&checkpoint, &config, AttentionKind::FullAttention).unwrap();
        let decode_layer =
            finish!(|graph| { build_decode_layer_pair(graph, sliding, full, &config, decode) });
        assert_contract!(
            decode_layer,
            7,
            58,
            5,
            &[Some(0), Some(3), Some(4), Some(5), Some(6)],
        );
        assert!(!decode_layer
            .input_names()
            .any(|name| name == "sequence_lengths"));
        let layer_names = decode_layer
            .input_names()
            .filter(|name| name.starts_with("model.layers."))
            .collect::<Vec<_>>();
        assert!(
            layer_names
                .iter()
                .all(|name| name.starts_with("model.layers.0.")
                    || name.starts_with("model.layers.1."))
        );
        assert!(layer_names
            .iter()
            .any(|name| name.starts_with("model.layers.0.")));
        assert!(layer_names
            .iter()
            .any(|name| name.starts_with("model.layers.1.")));

        let prefill_head = finish!(|graph| build_head(graph, &checkpoint, &config, prefill));
        assert_contract!(prefill_head, 7, 4, 2, &[None, Some(2)]);
        let decode_head = finish!(|graph| build_head(graph, &checkpoint, &config, decode));
        assert_contract!(decode_head, 8, 4, 3, &[None, Some(2), None]);
    }

    #[test]
    fn fused_decode_full_graph_contract() {
        let config = selected_config();
        let checkpoint = synthetic_checkpoint(&config);
        let decode = ShapeFamily::decode(512).unwrap();
        let layers = config.layers();

        let program = finish!(|graph| { build_decode_full(graph, &checkpoint, &config, decode) });

        let activations = program
            .inputs()
            .filter(|(_, _, binding)| !binding.is_parameter_component())
            .count();
        let parameters = program
            .inputs()
            .filter(|(_, _, binding)| binding.is_parameter_component())
            .count();
        let outputs = program.outputs().count();

        // 57 activation inputs: tokens + position + page_table + 48 caches
        // + last_index + sampling_state + top_k + temperature + top_p + min_p
        assert_eq!(activations, 57, "fused decode activation input count");
        // 703 parameter components: 24 layers × 29 + 3 (embed) + 1 (norm) + 3 (lm_head)
        assert_eq!(
            parameters,
            layers * 29 + 7,
            "fused decode parameter component count"
        );
        // 51 outputs: token + sampling_state + position + 48 caches
        assert_eq!(outputs, 3 + layers * 2, "fused decode output count");

        // Activation inputs are interleaved with parameter component slots.
        // Input ordering (0-indexed):
        //   0: tokens (activation)
        //   1-3: embed_tokens.weight (parameter, NVFP4 = 3 components)
        //   4: position (activation)
        //   5: page_table (activation)
        //   6 + 31*i: cache.{i}.key (activation)
        //   6 + 31*i + 1: cache.{i}.value (activation)
        //   8 + 31*i .. 36 + 31*i: layer i parameters (29 slots)
        //   750: last_index (activation)
        //   751: sampling_state (activation) — aliased to output sampling_state
        //   752-755: top_k, temperature, top_p, min_p (activation)
        //   756: norm.weight (parameter)
        //   757-759: lm_head.weight (parameter)
        let cache_base = 6usize;
        let stride = 31usize; // 2 cache slots + 29 parameter components per layer
        let sampling_state_index = 751usize;
        let mut expected_aliases: Vec<Option<usize>> = vec![None, Some(sampling_state_index), None];
        for i in 0..layers {
            expected_aliases.push(Some(cache_base + stride * i));
            expected_aliases.push(Some(cache_base + stride * i + 1));
        }
        assert_eq!(
            program.output_aliases().collect::<Vec<_>>(),
            expected_aliases,
            "fused decode output aliases",
        );

        // Verify all 24 layers' parameters appear in the input names
        let layer_names: Vec<_> = program
            .input_names()
            .filter(|name| name.starts_with("model.layers."))
            .collect();
        assert!(!layer_names.is_empty());
        for i in 0..layers {
            assert!(
                layer_names
                    .iter()
                    .any(|name| { name.starts_with(&format!("model.layers.{i}.")) }),
                "fused decode includes layer {i} parameters",
            );
        }

        // Verify cache inputs are named with per-layer indices
        for i in 0..layers {
            assert!(
                program
                    .input_names()
                    .any(|name| name == format!("cache.{i}.key")),
                "fused decode has cache.{i}.key input",
            );
            assert!(
                program
                    .input_names()
                    .any(|name| name == format!("cache.{i}.value")),
                "fused decode has cache.{i}.value input",
            );
        }
    }

    fn selected_config() -> Config {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        Config::from_file(
            Path::new(&runfiles).join("_main/artifacts/gpt-oss-20b-nvfp4/config.json"),
        )
        .unwrap()
    }

    fn synthetic_checkpoint(config: &Config) -> Checkpoint {
        declare_with(
            config,
            &mut |name, shape| {
                nml::Parameter::dense(name, name, shape)
                    .map_err(|error| Box::new(error) as BoxError)
            },
            &mut |name, shape| {
                nml::Parameter::nvfp4(name, name, shape)
                    .map_err(|error| Box::new(error) as BoxError)
            },
        )
        .unwrap()
    }
}
