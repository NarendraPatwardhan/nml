//! Bounded GPT-OSS component graphs.
//!
//! The compiler sees one bounded fusion domain at a time: embedding, one full
//! prefill layer, one alternating four-layer decode group, or final projection and
//! sampling. Model depth remains an execution-plan concern and never duplicates
//! the reusable component bodies in StableHLO.

use super::checkpoint::{BoxError, Checkpoint, DecoderLayer, Result, message};
use super::config::{AttentionKind, Config};
use nml::{DataType, Graph, Shape, Tensor};

// Sixteen-token physical pages provide useful allocation granularity for a
// future shared cache arena. CUDA compute-tile width is an independent kernel
// policy, so changing this product allocation constant cannot by itself create
// a wider, spill-heavy attention specialization.
pub(super) const CACHE_PAGE_SIZE: usize = 16;
pub(super) const MAXIMUM_TOP_K: usize = 64;
const LAYER_CONTROL_FIELDS: usize = 4;
const HEAD_I32_CONTROL_FIELDS: usize = 3;
const HEAD_F32_CONTROL_FIELDS: usize = 3;
pub(super) const BATCH_RESULT_BYTES_PER_ROW: usize =
    std::mem::size_of::<i32>() + 2 * std::mem::size_of::<u64>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ServingSlabLayout {
    token_offset: usize,
    layer_offset: usize,
    head_i32_offset: usize,
    sampling_state_offset: usize,
    head_f32_offset: usize,
    total_bytes: usize,
}

impl ServingSlabLayout {
    pub(super) fn for_family(family: ShapeFamily) -> Result<Self> {
        if !family.is_serving() {
            return Err(message("only serving families have a batch slab"));
        }
        let batch = family.batch();
        let token_offset = 0;
        let layer_offset = batch
            .checked_mul(family.sequence())
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| message("serving token slab size overflows usize"))?;
        let head_i32_offset = layer_offset
            .checked_add(
                batch
                    .checked_mul(LAYER_CONTROL_FIELDS + family.page_count())
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<i32>()))
                    .ok_or_else(|| message("serving layer slab size overflows usize"))?,
            )
            .ok_or_else(|| message("serving slab offset overflows usize"))?;
        let sampling_state_offset = head_i32_offset
            .checked_add(
                batch
                    .checked_mul(HEAD_I32_CONTROL_FIELDS)
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<i32>()))
                    .ok_or_else(|| message("serving head slab size overflows usize"))?,
            )
            .ok_or_else(|| message("serving slab offset overflows usize"))?;
        let head_f32_offset = sampling_state_offset
            .checked_add(
                batch
                    .checked_mul(2)
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<u64>()))
                    .ok_or_else(|| message("serving state slab size overflows usize"))?,
            )
            .ok_or_else(|| message("serving slab offset overflows usize"))?;
        let total_bytes = head_f32_offset
            .checked_add(
                batch
                    .checked_mul(HEAD_F32_CONTROL_FIELDS)
                    .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| message("serving sampling slab size overflows usize"))?,
            )
            .ok_or_else(|| message("serving slab size overflows usize"))?;
        Ok(Self {
            token_offset,
            layer_offset,
            head_i32_offset,
            sampling_state_offset,
            head_f32_offset,
            total_bytes,
        })
    }

    pub(super) const fn token_offset(self) -> usize {
        self.token_offset
    }

    pub(super) const fn layer_offset(self) -> usize {
        self.layer_offset
    }

    pub(super) const fn head_i32_offset(self) -> usize {
        self.head_i32_offset
    }

    pub(super) const fn sampling_state_offset(self) -> usize {
        self.sampling_state_offset
    }

    pub(super) const fn head_f32_offset(self) -> usize {
        self.head_f32_offset
    }

    pub(super) const fn total_bytes(self) -> usize {
        self.total_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ShapeFamily {
    phase: Phase,
    batch: usize,
    sequence: usize,
    cache_capacity: usize,
    physical_pages: usize,
    tensor_parallel: usize,
    serving: bool,
}

impl ShapeFamily {
    pub(super) fn prefill(
        sequence: usize,
        cache_capacity: usize,
        physical_pages: usize,
    ) -> Result<Self> {
        Self::new(
            Phase::Prefill,
            1,
            sequence,
            cache_capacity,
            physical_pages,
            1,
            false,
        )
    }

    pub(super) fn decode(cache_capacity: usize, physical_pages: usize) -> Result<Self> {
        Self::new(
            Phase::Decode,
            1,
            1,
            cache_capacity,
            physical_pages,
            1,
            false,
        )
    }

    pub(super) fn serving_prefill(
        batch: usize,
        query: usize,
        cache_capacity: usize,
        physical_pages: usize,
        tensor_parallel: usize,
    ) -> Result<Self> {
        Self::new(
            Phase::Prefill,
            batch,
            query,
            cache_capacity,
            physical_pages,
            tensor_parallel,
            true,
        )
    }

    pub(super) fn serving_decode(
        batch: usize,
        cache_capacity: usize,
        physical_pages: usize,
        tensor_parallel: usize,
    ) -> Result<Self> {
        Self::new(
            Phase::Decode,
            batch,
            1,
            cache_capacity,
            physical_pages,
            tensor_parallel,
            true,
        )
    }

    fn new(
        phase: Phase,
        batch: usize,
        sequence: usize,
        cache_capacity: usize,
        physical_pages: usize,
        tensor_parallel: usize,
        serving: bool,
    ) -> Result<Self> {
        if batch == 0 || sequence == 0 || cache_capacity == 0 || physical_pages == 0 {
            return Err(message("GPT-OSS execution dimensions must be nonzero"));
        }
        if !matches!(tensor_parallel, 1 | 2 | 4) {
            return Err(message("GPT-OSS tensor parallel degree must be 1, 2, or 4"));
        }
        if sequence > cache_capacity {
            return Err(message("GPT-OSS prefill bucket exceeds cache capacity"));
        }
        if cache_capacity > i32::MAX as usize {
            return Err(message(
                "GPT-OSS cache capacity exceeds the I32 index domain",
            ));
        }
        if physical_pages > i32::MAX as usize {
            return Err(message(
                "GPT-OSS physical cache page count exceeds the I32 index domain",
            ));
        }
        if !cache_capacity.is_multiple_of(CACHE_PAGE_SIZE) {
            return Err(message(
                "GPT-OSS cache capacity must contain complete pages",
            ));
        }
        Ok(Self {
            phase,
            batch,
            sequence,
            cache_capacity,
            physical_pages,
            tensor_parallel,
            serving,
        })
    }

    pub(super) const fn phase(self) -> Phase {
        self.phase
    }

    pub(super) const fn sequence(self) -> usize {
        self.sequence
    }

    pub(super) const fn batch(self) -> usize {
        self.batch
    }

    pub(super) const fn cache_capacity(self) -> usize {
        self.cache_capacity
    }

    pub(super) const fn page_count(self) -> usize {
        self.cache_capacity / CACHE_PAGE_SIZE
    }

    pub(super) const fn physical_pages(self) -> usize {
        self.physical_pages
    }

    pub(super) const fn is_serving(self) -> bool {
        self.serving
    }
}

pub(super) fn build_embedding(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    let batch = dimension(family.batch())?;
    let sequence = dimension(family.sequence())?;
    let token_shape = match family.phase() {
        Phase::Prefill => shape(DataType::I32, &[batch, sequence])?,
        Phase::Decode => shape(DataType::I32, &[batch])?,
    };
    let tokens = if family.is_serving() {
        let (slab, layout) = serving_slab_input(graph, family)?;
        let tokens = slab_typed_range(
            graph,
            slab,
            layout.token_offset(),
            family.batch(),
            family.sequence(),
            DataType::I32,
        )?;
        match family.phase() {
            Phase::Prefill => tokens,
            Phase::Decode => nml(graph.reshape(tokens, token_shape))?,
        }
    } else {
        graph.input("tokens", token_shape)
    };
    let hidden = nml(graph.token_embedding(&checkpoint.model.embed_tokens.weight, tokens))?;
    let hidden = match family.phase() {
        Phase::Prefill => hidden,
        Phase::Decode => nml(graph.reshape(
            hidden,
            shape(
                DataType::Bf16,
                &[batch, 1, dimension(config.hidden_size())?],
            )?,
        ))?,
    };
    require_shape(
        hidden,
        DataType::Bf16,
        &[
            dimension(family.batch())?,
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
    let (position, sequence_lengths, query_lengths, active_rows, page_table) =
        layer_inputs(graph, family)?;
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
        query_lengths,
        active_rows,
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

/// Builds one medium-grained decode fusion domain from two immutable
/// sliding/full pairs. Four layers halve the accepted pair-level submission
/// count without turning all model depth into one compiler monolith.
pub(super) fn build_decode_layer_group(
    graph: &mut Graph,
    layers: [&DecoderLayer; 4],
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    if family.phase() != Phase::Decode {
        return Err(message("GPT-OSS layer groups are decode-only"));
    }
    let hidden_shape = hidden_shape(config, family)?;
    let cache_shape = cache_shape(config, family)?;
    let hidden_input = graph.input("hidden", hidden_shape);
    let (position, sequence_lengths, query_lengths, active_rows, page_table) =
        layer_inputs(graph, family)?;
    let mut hidden = hidden_input;
    let mut outputs = Vec::with_capacity(9);
    for (index, layer) in layers.into_iter().enumerate() {
        let key_name = format!("layer{index}.cache.key");
        let value_name = format!("layer{index}.cache.value");
        let key_input = graph.input(&key_name, cache_shape);
        let value_input = graph.input(&value_name, cache_shape);
        let kind = if index.is_multiple_of(2) {
            AttentionKind::SlidingAttention
        } else {
            AttentionKind::FullAttention
        };
        let (next_hidden, key, value) = apply_layer(
            graph,
            layer,
            config,
            family,
            kind,
            hidden,
            position,
            sequence_lengths,
            query_lengths,
            active_rows,
            page_table,
            key_input,
            value_input,
        )?;
        hidden = next_hidden;
        outputs.push((key_name, key));
        outputs.push((value_name, value));
    }
    let hidden = nml(graph.reuse_buffer(hidden, hidden_input))?;
    outputs.insert(0, ("hidden".to_owned(), hidden));
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
    query_lengths: Tensor,
    active_rows: Tensor,
    page_table: Tensor,
    key_input: Tensor,
    value_input: Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch = dimension(family.batch())?;
    let sequence = dimension(family.sequence())?;
    let hidden_size = dimension(config.hidden_size())?;
    let query_heads = dimension(config.query_heads())?;
    let key_value_heads = dimension(config.key_value_heads())?;
    let head_dim = dimension(config.head_dim())?;
    let hidden_shape = hidden_shape(config, family)?;

    let token_shape = shape(DataType::I32, &[batch, sequence])?;
    let offsets = nml(graph.iota(token_shape, 1))?;
    let position_vector = nml(graph.broadcast_in_dim(position, token_shape, &[0]))?;
    let positions = nml(graph.add(offsets, position_vector))?;

    let residual = hidden_input;
    let input_norm = nml(graph.parameter_value(&layer.input_layernorm.weight))?;
    let mut hidden =
        nml(graph.rms_norm(hidden_input, Some(input_norm), 2, config.rms_norm_epsilon()))?;
    let (query, key, value) = nml(graph.linear_qkv(
        hidden,
        &layer.self_attn.q_proj.weight,
        Some(&layer.self_attn.q_proj.bias),
        &layer.self_attn.k_proj.weight,
        Some(&layer.self_attn.k_proj.bias),
        &layer.self_attn.v_proj.weight,
        Some(&layer.self_attn.v_proj.bias),
    ))?;
    let query = nml(graph.reshape(
        query,
        shape(DataType::Bf16, &[batch, sequence, query_heads, head_dim])?,
    ))?;
    let key = nml(graph.reshape(
        key,
        shape(
            DataType::Bf16,
            &[batch, sequence, key_value_heads, head_dim],
        )?,
    ))?;
    let value = nml(graph.reshape(
        value,
        shape(
            DataType::Bf16,
            &[batch, sequence, key_value_heads, head_dim],
        )?,
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

    let query_lengths_vector = query_lengths;
    let query_lengths = nml(graph.broadcast_in_dim(query_lengths_vector, token_shape, &[0]))?;
    let valid_queries = nml(graph.less(offsets, query_lengths))?;
    let active_queries = nml(graph.broadcast_in_dim(
        active_rows,
        shape(DataType::Bool, &[batch, sequence])?,
        &[0],
    ))?;
    let write_mask = nml(graph.logical_and(valid_queries, active_queries))?;
    let (key_cache, value_cache) = nml(graph.paged_cache_update_pair(
        key_input,
        value_input,
        key,
        value,
        page_table,
        position,
        query_lengths_vector,
        active_rows,
        write_mask,
    ))?;
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
            &[batch, sequence, dimension(config.query_width())?],
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
    let routed_tokens = batch
        .checked_mul(sequence)
        .ok_or_else(|| message("GPT-OSS routed token dimension overflows I64"))?;
    let routed = nml(graph.reshape(
        hidden,
        shape(DataType::Bf16, &[routed_tokens, hidden_size])?,
    ))?;
    let active_tokens = nml(graph.reshape(write_mask, shape(DataType::Bool, &[routed_tokens])?))?;
    let router_logits = nml(graph.linear(
        routed,
        &layer.mlp.router.weight,
        Some(&layer.mlp.router.bias),
    ))?;
    let routed = nml(graph.routed_clamped_swiglu_masked(
        routed,
        router_logits,
        &layer.mlp.experts.gate_up_proj,
        &layer.mlp.experts.gate_up_proj_bias,
        &layer.mlp.experts.down_proj,
        &layer.mlp.experts.down_proj_bias,
        config.experts_per_token(),
        active_tokens,
    ))?;
    let routed = nml(graph.reshape(routed, hidden_shape))?;
    hidden = nml(graph.add(residual, routed))?;
    let active_hidden =
        nml(graph.broadcast_in_dim(write_mask, hidden_shape.with_dtype(DataType::Bool), &[0, 1]))?;
    hidden = nml(graph.select(active_hidden, hidden, hidden_input))?;
    Ok((hidden, key_cache, value_cache))
}

pub(super) fn build_head(
    graph: &mut Graph,
    checkpoint: &Checkpoint,
    config: &Config,
    family: ShapeFamily,
) -> Result<Vec<(String, Tensor)>> {
    let batch = dimension(family.batch())?;
    let sequence = dimension(family.sequence())?;
    let hidden_size = dimension(config.hidden_size())?;
    let hidden = graph.input(
        "hidden",
        shape(DataType::Bf16, &[batch, sequence, hidden_size])?,
    );
    let (
        last_index,
        sampling_state_input,
        top_k,
        temperature,
        top_p,
        min_p,
        active_rows,
        serving_slab,
    ) = head_inputs(graph, family)?;
    let last = nml(graph.gather_batched_nd(hidden, last_index, 1, &[1]))?;
    let final_norm = nml(graph.parameter_value(&checkpoint.model.norm.weight))?;
    let last = nml(graph.rms_norm(last, Some(final_norm), 1, config.rms_norm_epsilon()))?;
    let logits = nml(graph.linear(last, &checkpoint.lm_head.weight, None))?;
    let (sampling_state, token) = nml(graph.sample_tokens_batched_dynamic(
        logits,
        sampling_state_input,
        top_k,
        temperature,
        top_p,
        min_p,
        active_rows,
        MAXIMUM_TOP_K,
    ))?;
    if family.is_serving() {
        let token_bytes = nml(graph.bitcast(token, DataType::U8))?;
        let state_bytes = nml(graph.bitcast(sampling_state, DataType::U8))?;
        let state_bytes = nml(graph.reshape(
            state_bytes,
            shape(
                DataType::U8,
                &[batch, dimension(2 * std::mem::size_of::<u64>())?],
            )?,
        ))?;
        let result = nml(graph.concatenate(&[token_bytes, state_bytes], 1))?;
        let mut outputs = vec![("batch_result".to_owned(), result)];
        if family.phase() == Phase::Decode {
            let (slab, layout) = serving_slab.expect("serving decode head retains its slab input");
            let next_slab = next_decode_slab(
                graph,
                family,
                slab,
                layout,
                token,
                sampling_state,
                active_rows,
            )?;
            outputs.push((
                "next_batch_slab".to_owned(),
                nml(graph.reuse_buffer(next_slab, slab))?,
            ));
        }
        return Ok(outputs);
    }

    let sampling_state = nml(graph.reuse_buffer(sampling_state, sampling_state_input))?;
    let mut outputs = vec![
        ("token".to_owned(), token),
        ("sampling_state".to_owned(), sampling_state),
    ];
    if family.phase() == Phase::Decode && !family.is_serving() {
        let position = graph.input("position", shape(DataType::I32, &[batch])?);
        let one = nml(graph.scalar(1_i32))?;
        let advanced = nml(graph.add(position, one))?;
        outputs.push((
            "position".to_owned(),
            nml(graph.select(active_rows, advanced, position))?,
        ));
    }
    Ok(outputs)
}

fn layer_inputs(
    graph: &mut Graph,
    family: ShapeFamily,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let batch = dimension(family.batch())?;
    if !family.is_serving() {
        let position = graph.input("position", shape(DataType::I32, &[batch])?);
        let sequence_lengths = sequence_lengths(graph, family, position)?;
        let query_lengths = graph.input("query_lengths", shape(DataType::I32, &[batch])?);
        let active_rows = graph.input("active_rows", shape(DataType::Bool, &[batch])?);
        let page_table = graph.input("page_table", page_table_shape(family)?);
        return Ok((
            position,
            sequence_lengths,
            query_lengths,
            active_rows,
            page_table,
        ));
    }

    let width_usize = LAYER_CONTROL_FIELDS + family.page_count();
    let width = dimension(width_usize)?;
    let (slab, layout) = serving_slab_input(graph, family)?;
    let control = slab_typed_range(
        graph,
        slab,
        layout.layer_offset(),
        family.batch(),
        width_usize,
        DataType::I32,
    )?;
    let position = control_vector(graph, control, 0, batch, DataType::I32)?;
    let sequence_lengths = control_vector(graph, control, 1, batch, DataType::I32)?;
    let query_lengths = control_vector(graph, control, 2, batch, DataType::I32)?;
    let active_i32 = control_vector(graph, control, 3, batch, DataType::I32)?;
    let zero = nml(graph.scalar(0_i32))?;
    let active_rows = nml(graph.not_equal(active_i32, zero))?;
    let page_table = nml(graph.slice(
        control,
        &[0, dimension(LAYER_CONTROL_FIELDS)?],
        &[batch, width],
        &[1, 1],
    ))?;
    Ok((
        position,
        sequence_lengths,
        query_lengths,
        active_rows,
        page_table,
    ))
}

fn head_inputs(
    graph: &mut Graph,
    family: ShapeFamily,
) -> Result<(
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Option<(Tensor, ServingSlabLayout)>,
)> {
    let batch = dimension(family.batch())?;
    if !family.is_serving() {
        return Ok((
            graph.input("last_index", shape(DataType::I32, &[batch, 1])?),
            graph.input("sampling_state", shape(DataType::U64, &[batch, 2])?),
            graph.input("top_k", shape(DataType::I32, &[batch])?),
            graph.input("temperature", shape(DataType::F32, &[batch])?),
            graph.input("top_p", shape(DataType::F32, &[batch])?),
            graph.input("min_p", shape(DataType::F32, &[batch])?),
            graph.input("active_rows", shape(DataType::Bool, &[batch])?),
            None,
        ));
    }

    let (slab, layout) = serving_slab_input(graph, family)?;
    let i32_control = slab_typed_range(
        graph,
        slab,
        layout.head_i32_offset(),
        family.batch(),
        HEAD_I32_CONTROL_FIELDS,
        DataType::I32,
    )?;
    let sampling_state = slab_typed_range(
        graph,
        slab,
        layout.sampling_state_offset(),
        family.batch(),
        2,
        DataType::U64,
    )?;
    let f32_control = slab_typed_range(
        graph,
        slab,
        layout.head_f32_offset(),
        family.batch(),
        HEAD_F32_CONTROL_FIELDS,
        DataType::F32,
    )?;
    let last_index = nml(graph.slice(i32_control, &[0, 0], &[batch, 1], &[1, 1]))?;
    let top_k = control_vector(graph, i32_control, 1, batch, DataType::I32)?;
    let active_i32 = control_vector(graph, i32_control, 2, batch, DataType::I32)?;
    let zero = nml(graph.scalar(0_i32))?;
    let active_rows = nml(graph.not_equal(active_i32, zero))?;
    let temperature = control_vector(graph, f32_control, 0, batch, DataType::F32)?;
    let top_p = control_vector(graph, f32_control, 1, batch, DataType::F32)?;
    let min_p = control_vector(graph, f32_control, 2, batch, DataType::F32)?;
    Ok((
        last_index,
        sampling_state,
        top_k,
        temperature,
        top_p,
        min_p,
        active_rows,
        Some((slab, layout)),
    ))
}

fn next_decode_slab(
    graph: &mut Graph,
    family: ShapeFamily,
    slab: Tensor,
    layout: ServingSlabLayout,
    token: Tensor,
    sampling_state: Tensor,
    active_rows: Tensor,
) -> Result<Tensor> {
    let batch = dimension(family.batch())?;
    let width_usize = LAYER_CONTROL_FIELDS + family.page_count();

    let token_bytes = nml(graph.bitcast(token, DataType::U8))?;
    let token_bytes = nml(graph.reshape(
        token_bytes,
        shape(DataType::U8, &[dimension(layout.layer_offset())?])?,
    ))?;

    let layer = slab_typed_range(
        graph,
        slab,
        layout.layer_offset(),
        family.batch(),
        width_usize,
        DataType::I32,
    )?;
    let position = control_vector(graph, layer, 0, batch, DataType::I32)?;
    let sequence_length = control_vector(graph, layer, 1, batch, DataType::I32)?;
    let one = nml(graph.scalar(1_i32))?;
    let advanced_position = nml(graph.add(position, one))?;
    let next_position = nml(graph.select(active_rows, advanced_position, position))?;
    let advanced_sequence_length = nml(graph.add(sequence_length, one))?;
    let next_sequence_length =
        nml(graph.select(active_rows, advanced_sequence_length, sequence_length))?;
    let next_position = nml(graph.reshape(next_position, shape(DataType::I32, &[batch, 1])?))?;
    let next_sequence_length =
        nml(graph.reshape(next_sequence_length, shape(DataType::I32, &[batch, 1])?))?;
    let next_positions_and_lengths =
        nml(graph.concatenate(&[next_position, next_sequence_length], 1))?;
    let state_bytes = nml(graph.bitcast(sampling_state, DataType::U8))?;
    let state_bytes = nml(graph.reshape(
        state_bytes,
        shape(
            DataType::U8,
            &[dimension(
                layout
                    .head_f32_offset()
                    .checked_sub(layout.sampling_state_offset())
                    .ok_or_else(|| message("serving state slab range underflows"))?,
            )?],
        )?,
    ))?;

    // The slab is donated to this result. Describe the complete state
    // transition as one set of disjoint byte ranges; the generic graph API
    // lowers it to one sorted, unique assignment scatter.
    let mut patches = Vec::with_capacity(family.batch() + 2);
    patches.push((0_i64, token_bytes));
    let row_control_bytes = 2 * std::mem::size_of::<i32>();
    let row_stride_bytes = width_usize
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| message("serving layer row byte count overflows"))?;
    for row in 0..family.batch() {
        let row_end = row
            .checked_add(1)
            .ok_or_else(|| message("serving batch row overflows"))?;
        let controls = nml(graph.slice(
            next_positions_and_lengths,
            &[dimension(row)?, 0],
            &[dimension(row_end)?, 2],
            &[1, 1],
        ))?;
        let controls = nml(graph.bitcast(controls, DataType::U8))?;
        let controls = nml(graph.reshape(
            controls,
            shape(DataType::U8, &[dimension(row_control_bytes)?])?,
        ))?;
        let offset = layout
            .layer_offset()
            .checked_add(
                row.checked_mul(row_stride_bytes)
                    .ok_or_else(|| message("serving layer row offset overflows"))?,
            )
            .ok_or_else(|| message("serving layer row offset overflows"))?;
        patches.push((
            i64::try_from(offset).map_err(|_| message("serving layer row offset exceeds I64"))?,
            controls,
        ));
    }
    patches.push((
        i64::try_from(layout.sampling_state_offset())
            .map_err(|_| message("serving sampling state offset exceeds I64"))?,
        state_bytes,
    ));
    nml(graph.patch_1d(slab, &patches))
}

fn control_vector(
    graph: &mut Graph,
    control: Tensor,
    column: usize,
    batch: i64,
    dtype: DataType,
) -> Result<Tensor> {
    let column = dimension(column)?;
    let sliced = nml(graph.slice(control, &[0, column], &[batch, column + 1], &[1, 1]))?;
    nml(graph.reshape(sliced, shape(dtype, &[batch])?))
}

fn serving_slab_input(
    graph: &mut Graph,
    family: ShapeFamily,
) -> Result<(Tensor, ServingSlabLayout)> {
    let layout = ServingSlabLayout::for_family(family)?;
    let slab = graph.input(
        "batch_slab",
        shape(DataType::U8, &[dimension(layout.total_bytes())?])?,
    );
    Ok((slab, layout))
}

fn slab_typed_range(
    graph: &mut Graph,
    slab: Tensor,
    byte_offset: usize,
    batch: usize,
    elements: usize,
    dtype: DataType,
) -> Result<Tensor> {
    let batch_elements = batch
        .checked_mul(elements)
        .ok_or_else(|| message("serving slab range overflows usize"))?;
    let batch = dimension(batch)?;
    let byte_width = dtype.byte_width();
    let byte_length = batch_elements
        .checked_mul(byte_width)
        .ok_or_else(|| message("serving slab range overflows usize"))?;
    let byte_limit = byte_offset
        .checked_add(byte_length)
        .ok_or_else(|| message("serving slab range overflows usize"))?;
    let bytes = nml(graph.slice(
        slab,
        &[dimension(byte_offset)?],
        &[dimension(byte_limit)?],
        &[1],
    ))?;
    let packed = nml(graph.reshape(
        bytes,
        shape(
            DataType::U8,
            &[batch, dimension(elements)?, dimension(byte_width)?],
        )?,
    ))?;
    nml(graph.bitcast(packed, dtype))
}

pub(super) fn cache_shape(config: &Config, family: ShapeFamily) -> Result<Shape> {
    shape(
        DataType::Bf16,
        &[
            dimension(family.physical_pages())?,
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
            dimension(family.batch())?,
            dimension(family.sequence())?,
            dimension(config.hidden_size())?,
        ],
    )
}

fn sequence_lengths(graph: &mut Graph, family: ShapeFamily, position: Tensor) -> Result<Tensor> {
    match family.phase() {
        Phase::Prefill => Ok(graph.input(
            "sequence_lengths",
            shape(DataType::I32, &[dimension(family.batch())?])?,
        )),
        Phase::Decode => {
            let one = nml(graph.scalar(1_i32))?;
            nml(graph.add(position, one))
        }
    }
}

pub(super) fn page_table_shape(family: ShapeFamily) -> Result<Shape> {
    shape(
        DataType::I32,
        &[dimension(family.batch())?, dimension(family.page_count())?],
    )
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
        let prefill = ShapeFamily::prefill(32, 512, 32).unwrap();
        let decode = ShapeFamily::decode(512, 32).unwrap();

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
        assert_contract!(prefill_layer, 8, 29, 3, &[Some(0), Some(6), Some(7)],);
        assert!(
            prefill_layer
                .input_names()
                .any(|name| name == "sequence_lengths")
        );
        assert_single_layer_identity!(prefill_layer, 0);

        let layers = &checkpoint.model.layers[..4];
        let decode_layer = finish!(|graph| {
            build_decode_layer_group(
                graph,
                [&layers[0], &layers[1], &layers[2], &layers[3]],
                &config,
                decode,
            )
        });
        assert_contract!(
            decode_layer,
            13,
            116,
            9,
            &[
                Some(0),
                Some(5),
                Some(6),
                Some(36),
                Some(37),
                Some(67),
                Some(68),
                Some(98),
                Some(99)
            ],
        );
        assert!(
            !decode_layer
                .input_names()
                .any(|name| name == "sequence_lengths")
        );
        let layer_names = decode_layer
            .input_names()
            .filter(|name| name.starts_with("model.layers."))
            .collect::<Vec<_>>();
        assert!(layer_names.iter().all(|name| {
            (0..4).any(|layer| name.starts_with(&format!("model.layers.{layer}.")))
        }));
        assert!(
            layer_names
                .iter()
                .any(|name| name.starts_with("model.layers.0."))
        );
        assert!(
            layer_names
                .iter()
                .any(|name| name.starts_with("model.layers.1."))
        );

        let prefill_head = finish!(|graph| build_head(graph, &checkpoint, &config, prefill));
        assert_contract!(prefill_head, 8, 4, 2, &[None, Some(2)]);
        let decode_head = finish!(|graph| build_head(graph, &checkpoint, &config, decode));
        assert_contract!(decode_head, 9, 4, 3, &[None, Some(2), None]);

        let batched = ShapeFamily::serving_decode(4, 512, 32, 1).unwrap();
        let batched_embedding =
            finish!(|graph| build_embedding(graph, &checkpoint, &config, batched));
        assert_contract!(batched_embedding, 1, 3, 1, &[None]);
        assert!(
            batched_embedding
                .input_names()
                .any(|name| name == "batch_slab")
        );
        assert_eq!(
            batched_embedding.outputs().next().unwrap().1.dimensions(),
            &[4, 1, config.hidden_size() as i64]
        );
        let batched_layer = finish!(|graph| {
            build_decode_layer_group(
                graph,
                [&layers[0], &layers[1], &layers[2], &layers[3]],
                &config,
                batched,
            )
        });
        assert_contract!(
            batched_layer,
            10,
            116,
            9,
            &[
                Some(0),
                Some(2),
                Some(3),
                Some(33),
                Some(34),
                Some(64),
                Some(65),
                Some(95),
                Some(96)
            ],
        );
        let batched_layer_inputs = batched_layer.input_names().collect::<Vec<_>>();
        assert!(batched_layer_inputs.contains(&"batch_slab"));
        assert!(!batched_layer_inputs.contains(&"position"));
        assert!(!batched_layer_inputs.contains(&"page_table"));
        let batched_head = finish!(|graph| build_head(graph, &checkpoint, &config, batched));
        assert_contract!(batched_head, 2, 4, 2, &[None, Some(1)]);
        let batched_head_inputs = batched_head.input_names().collect::<Vec<_>>();
        assert!(batched_head_inputs.contains(&"batch_slab"));
        assert!(!batched_head_inputs.contains(&"sampling_state"));
        assert!(!batched_head_inputs.contains(&"position"));
        let outputs = batched_head.outputs().collect::<Vec<_>>();
        assert_eq!(
            outputs[0].1.dimensions(),
            &[4, BATCH_RESULT_BYTES_PER_ROW as i64],
        );
        assert_eq!(
            outputs[1].1.dimensions(),
            &[ServingSlabLayout::for_family(batched)
                .unwrap()
                .total_bytes() as i64],
        );
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
