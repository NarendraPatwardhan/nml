//! CUDA unified paged attention expressed in the retained TTIR subset.
//!
//! This is the whole-sequence kernel used for prefill and sufficiently wide
//! decode launches. It follows the pinned ZML algorithm while removing FP8,
//! ALiBi, sink, soft-cap, and oneAPI branches that are outside NML's retained
//! contract. Cache pages remain physical and are never materialized as a dense
//! logical K/V tensor.

use super::{ArgumentKind, Builder, Comparison, DType, Error, Reduction, Value};

const LOG2_E: f64 = 1.442_695_040_888_963_4;

fn padded_power_of_two(value: i64) -> Option<i64> {
    u64::try_from(value)
        .ok()?
        .checked_next_power_of_two()?
        .try_into()
        .ok()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedAttention2dConfig {
    pub dtype: DType,
    pub num_query_heads: i64,
    pub queries_per_kv: i64,
    pub page_size: i64,
    pub tile_size: i64,
    pub head_size: i64,
    pub padded_head_size: i64,
    pub block_q: i64,
    pub block_m: i64,
    pub sliding_window: Option<i64>,
    pub causal: bool,
}

impl PagedAttention2dConfig {
    fn validate(self) -> Result<Self, Error> {
        let fits_i32 = |value: i64| i32::try_from(value).is_ok();
        if !matches!(self.dtype, DType::F16 | DType::Bf16 | DType::F32)
            || self.num_query_heads <= 0
            || self.queries_per_kv <= 0
            || self.num_query_heads % self.queries_per_kv != 0
            || self.page_size <= 0
            || self.tile_size <= 0
            || !(self.tile_size as u64).is_power_of_two()
            || self.head_size <= 0
            || self.padded_head_size < self.head_size
            || !(self.padded_head_size as u64).is_power_of_two()
            || self.block_q <= 0
            || padded_power_of_two(self.queries_per_kv)
                .is_none_or(|padded| self.block_q.checked_mul(padded) != Some(self.block_m))
            || !(self.block_m as u64).is_power_of_two()
            || self.sliding_window.is_some_and(|window| window <= 0)
            || [
                self.num_query_heads,
                self.queries_per_kv,
                self.page_size,
                self.tile_size,
                self.head_size,
                self.padded_head_size,
                self.block_q,
                self.block_m,
            ]
            .into_iter()
            .any(|value| !fits_i32(value))
        {
            return Err(Error::InvalidKernelSpec(
                "invalid retained 2D paged-attention specialization",
            ));
        }
        Ok(self)
    }
}

pub fn build_paged_attention_2d(config: PagedAttention2dConfig) -> Result<String, Error> {
    let config = config.validate()?;
    let mut builder = Builder::new("paged_attention_2d")?;
    let query = pointer(&mut builder, "query", config.dtype)?;
    let key_cache = pointer(&mut builder, "key_cache", config.dtype)?;
    let value_cache = pointer(&mut builder, "value_cache", config.dtype)?;
    let block_tables = pointer(&mut builder, "block_tables", DType::I32)?;
    let sequence_lengths = pointer(&mut builder, "sequence_lengths", DType::I32)?;
    let query_positions = pointer(&mut builder, "query_positions", DType::I32)?;
    let scale_pointer = pointer(&mut builder, "scale", DType::F32)?;
    let block_table_stride_pointer = pointer(&mut builder, "block_table_stride", DType::I64)?;
    let query_stride_0_pointer = pointer(&mut builder, "query_stride_0", DType::I64)?;
    let query_stride_1_pointer = pointer(&mut builder, "query_stride_1", DType::I64)?;
    let output_stride_0_pointer = pointer(&mut builder, "output_stride_0", DType::I64)?;
    let output_stride_1_pointer = pointer(&mut builder, "output_stride_1", DType::I64)?;
    let key_stride_0_pointer = pointer(&mut builder, "key_stride_0", DType::I64)?;
    let key_stride_1_pointer = pointer(&mut builder, "key_stride_1", DType::I64)?;
    let key_stride_2_pointer = pointer(&mut builder, "key_stride_2", DType::I64)?;
    let value_stride_0_pointer = pointer(&mut builder, "value_stride_0", DType::I64)?;
    let value_stride_1_pointer = pointer(&mut builder, "value_stride_1", DType::I64)?;
    let value_stride_2_pointer = pointer(&mut builder, "value_stride_2", DType::I64)?;
    let query_starts = pointer(&mut builder, "query_starts", DType::I32)?;
    let sequence_count_pointer = pointer(&mut builder, "sequence_count", DType::I32)?;
    let output = pointer(&mut builder, "output", config.dtype)?;

    let scale = builder.load(&scale_pointer)?;
    let block_table_stride = builder.load(&block_table_stride_pointer)?;
    let query_stride_0 = builder.load(&query_stride_0_pointer)?;
    let query_stride_1 = builder.load(&query_stride_1_pointer)?;
    let output_stride_0 = builder.load(&output_stride_0_pointer)?;
    let output_stride_1 = builder.load(&output_stride_1_pointer)?;
    let key_stride_0 = builder.load(&key_stride_0_pointer)?;
    let key_stride_1 = builder.load(&key_stride_1_pointer)?;
    let key_stride_2 = builder.load(&key_stride_2_pointer)?;
    let value_stride_0 = builder.load(&value_stride_0_pointer)?;
    let value_stride_1 = builder.load(&value_stride_1_pointer)?;
    let value_stride_2 = builder.load(&value_stride_2_pointer)?;
    let sequence_count = builder.load(&sequence_count_pointer)?;
    let kv_head = builder.program_id(0)?;
    let global_query_block = builder.program_id(1)?;
    let sequence = find_sequence(
        &mut builder,
        &query_starts,
        &global_query_block,
        &sequence_count,
        config.block_q,
        true,
    )?;
    let query_start = load_offset(&mut builder, &query_starts, &sequence)?;
    let block_q = integer(&mut builder, config.block_q, DType::I32)?;
    let query_block_start = builder.divide(&query_start, &block_q)?;
    let sequence_block_start = builder.add(&query_block_start, &sequence)?;
    let local_query_block = builder.subtract(&global_query_block, &sequence_block_start)?;
    let one_i32 = integer(&mut builder, 1, DType::I32)?;
    let next_sequence = builder.add(&sequence, &one_i32)?;
    let query_stop = load_offset(&mut builder, &query_starts, &next_sequence)?;
    let query_length = builder.subtract(&query_stop, &query_start)?;
    let local_query_offset = builder.multiply(&local_query_block, &block_q)?;
    let valid_block = builder.compare(Comparison::Less, &local_query_offset, &query_length)?;

    builder.if_only(&valid_block, |kernel| {
        emit_valid_block(
            kernel,
            config,
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &sequence_lengths,
            &query_positions,
            &scale,
            &block_table_stride,
            &query_stride_0,
            &query_stride_1,
            &key_stride_0,
            &key_stride_1,
            &key_stride_2,
            &value_stride_0,
            &value_stride_1,
            &value_stride_2,
            &kv_head,
            &sequence,
            &query_start,
            &query_length,
            &local_query_offset,
            AttentionOutput::Final {
                output: &output,
                stride_0: &output_stride_0,
                stride_1: &output_stride_1,
            },
        )
    })?;
    builder.return_void()?;
    builder.finish()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedAttention3dConfig {
    pub attention: PagedAttention2dConfig,
    pub segments: i64,
}

pub fn build_paged_attention_3d(config: PagedAttention3dConfig) -> Result<String, Error> {
    let attention = config.attention.validate()?;
    if config.segments <= 0
        || !(config.segments as u64).is_power_of_two()
        || i32::try_from(config.segments).is_err()
        || config
            .segments
            .checked_mul(attention.tile_size)
            .is_none_or(|value| i32::try_from(value).is_err())
    {
        return Err(Error::InvalidKernelSpec(
            "invalid split-K attention segment geometry",
        ));
    }
    let mut builder = Builder::new("paged_attention_3d")?;
    let query = pointer(&mut builder, "query", attention.dtype)?;
    let key_cache = pointer(&mut builder, "key_cache", attention.dtype)?;
    let value_cache = pointer(&mut builder, "value_cache", attention.dtype)?;
    let block_tables = pointer(&mut builder, "block_tables", DType::I32)?;
    let sequence_lengths = pointer(&mut builder, "sequence_lengths", DType::I32)?;
    let query_positions = pointer(&mut builder, "query_positions", DType::I32)?;
    let scale_pointer = pointer(&mut builder, "scale", DType::F32)?;
    let block_table_stride_pointer = pointer(&mut builder, "block_table_stride", DType::I64)?;
    let query_stride_0_pointer = pointer(&mut builder, "query_stride_0", DType::I64)?;
    let query_stride_1_pointer = pointer(&mut builder, "query_stride_1", DType::I64)?;
    let key_stride_0_pointer = pointer(&mut builder, "key_stride_0", DType::I64)?;
    let key_stride_1_pointer = pointer(&mut builder, "key_stride_1", DType::I64)?;
    let key_stride_2_pointer = pointer(&mut builder, "key_stride_2", DType::I64)?;
    let value_stride_0_pointer = pointer(&mut builder, "value_stride_0", DType::I64)?;
    let value_stride_1_pointer = pointer(&mut builder, "value_stride_1", DType::I64)?;
    let value_stride_2_pointer = pointer(&mut builder, "value_stride_2", DType::I64)?;
    let query_starts = pointer(&mut builder, "query_starts", DType::I32)?;
    let sequence_count_pointer = pointer(&mut builder, "sequence_count", DType::I32)?;
    let segment_values = pointer(&mut builder, "segment_values", DType::F32)?;
    let segment_maxima = pointer(&mut builder, "segment_maxima", DType::F32)?;
    let segment_sums = pointer(&mut builder, "segment_sums", DType::F32)?;

    let scale = builder.load(&scale_pointer)?;
    let block_table_stride = builder.load(&block_table_stride_pointer)?;
    let query_stride_0 = builder.load(&query_stride_0_pointer)?;
    let query_stride_1 = builder.load(&query_stride_1_pointer)?;
    let key_stride_0 = builder.load(&key_stride_0_pointer)?;
    let key_stride_1 = builder.load(&key_stride_1_pointer)?;
    let key_stride_2 = builder.load(&key_stride_2_pointer)?;
    let value_stride_0 = builder.load(&value_stride_0_pointer)?;
    let value_stride_1 = builder.load(&value_stride_1_pointer)?;
    let value_stride_2 = builder.load(&value_stride_2_pointer)?;
    let sequence_count = builder.load(&sequence_count_pointer)?;
    let global_query_block = builder.program_id(0)?;
    let kv_head = builder.program_id(1)?;
    let segment = builder.program_id(2)?;
    let sequence = find_sequence(
        &mut builder,
        &query_starts,
        &global_query_block,
        &sequence_count,
        attention.block_q,
        true,
    )?;
    let query_start = load_offset(&mut builder, &query_starts, &sequence)?;
    let block_q = integer(&mut builder, attention.block_q, DType::I32)?;
    let query_block_start = builder.divide(&query_start, &block_q)?;
    let sequence_block_start = builder.add(&query_block_start, &sequence)?;
    let local_query_block = builder.subtract(&global_query_block, &sequence_block_start)?;
    let one = integer(&mut builder, 1, DType::I32)?;
    let next_sequence = builder.add(&sequence, &one)?;
    let query_stop = load_offset(&mut builder, &query_starts, &next_sequence)?;
    let query_length = builder.subtract(&query_stop, &query_start)?;
    let local_query_offset = builder.multiply(&local_query_block, &block_q)?;
    let valid_query = builder.compare(Comparison::Less, &local_query_offset, &query_length)?;
    let sequence_length = load_offset(&mut builder, &sequence_lengths, &sequence)?;
    let segment_span = integer(
        &mut builder,
        config.segments * attention.tile_size,
        DType::I32,
    )?;
    let segment_span_minus_one = integer(
        &mut builder,
        config.segments * attention.tile_size - 1,
        DType::I32,
    )?;
    let padded_sequence = builder.add(&sequence_length, &segment_span_minus_one)?;
    let tiles_per_segment = builder.divide(&padded_sequence, &segment_span)?;
    let tile_size = integer(&mut builder, attention.tile_size, DType::I32)?;
    let segment_start = builder.multiply(&segment, &tiles_per_segment)?;
    let segment_start = builder.multiply(&segment_start, &tile_size)?;
    let valid_segment = builder.compare(Comparison::Less, &segment_start, &sequence_length)?;
    let valid_program = builder.bit_and(&valid_query, &valid_segment)?;

    builder.if_only(&valid_program, |kernel| {
        emit_valid_block(
            kernel,
            attention,
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &sequence_lengths,
            &query_positions,
            &scale,
            &block_table_stride,
            &query_stride_0,
            &query_stride_1,
            &key_stride_0,
            &key_stride_1,
            &key_stride_2,
            &value_stride_0,
            &value_stride_1,
            &value_stride_2,
            &kv_head,
            &sequence,
            &query_start,
            &query_length,
            &local_query_offset,
            AttentionOutput::Segment {
                values: &segment_values,
                maxima: &segment_maxima,
                sums: &segment_sums,
                index: &segment,
                count: config.segments,
            },
        )
    })?;
    builder.return_void()?;
    builder.finish()
}

#[derive(Clone, Copy)]
enum AttentionOutput<'a> {
    Final {
        output: &'a Value,
        stride_0: &'a Value,
        stride_1: &'a Value,
    },
    Segment {
        values: &'a Value,
        maxima: &'a Value,
        sums: &'a Value,
        index: &'a Value,
        count: i64,
    },
}

#[allow(clippy::too_many_arguments)]
fn emit_valid_block(
    kernel: &mut Builder,
    config: PagedAttention2dConfig,
    query: &Value,
    key_cache: &Value,
    value_cache: &Value,
    block_tables: &Value,
    sequence_lengths: &Value,
    query_position_values: &Value,
    scale: &Value,
    block_table_stride: &Value,
    query_stride_0: &Value,
    query_stride_1: &Value,
    key_stride_0: &Value,
    key_stride_1: &Value,
    key_stride_2: &Value,
    value_stride_0: &Value,
    value_stride_1: &Value,
    value_stride_2: &Value,
    kv_head: &Value,
    sequence: &Value,
    query_start: &Value,
    query_length: &Value,
    local_query_offset: &Value,
    attention_output: AttentionOutput<'_>,
) -> Result<(), Error> {
    let rows = kernel.range(0, config.block_m as i32)?;
    let dimensions = kernel.range(0, config.padded_head_size as i32)?;
    let tile = kernel.range(0, config.tile_size as i32)?;
    let queries_per_kv = integer(kernel, config.queries_per_kv, DType::I32)?;
    let padded_queries_per_kv = padded_power_of_two(config.queries_per_kv).ok_or(
        Error::InvalidKernelSpec("query-head group padding exceeds the retained index domain"),
    )?;
    let padded_queries_per_kv_value = integer(kernel, padded_queries_per_kv, DType::I32)?;
    let row_positions = kernel.divide(&rows, &padded_queries_per_kv_value)?;
    let query_slots = kernel.add(local_query_offset, &row_positions)?;
    let query_indices = kernel.add(query_start, &query_slots)?;
    let head_in_group = kernel.remainder(&rows, &padded_queries_per_kv_value)?;
    let first_query_head = kernel.multiply(kv_head, &queries_per_kv)?;
    let query_heads = kernel.add(&first_query_head, &head_in_group)?;
    let query_offset = offset_3d(
        kernel,
        &query_indices,
        &query_heads,
        &dimensions,
        query_stride_0,
        query_stride_1,
    )?;
    let dimension_mask = if config.padded_head_size == config.head_size {
        kernel.full_integer(&[config.padded_head_size], 1, DType::I1)?
    } else {
        let head_size = integer(kernel, config.head_size, DType::I32)?;
        kernel.compare(Comparison::Less, &dimensions, &head_size)?
    };
    let query_position_mask = kernel.compare(Comparison::Less, &query_slots, query_length)?;
    let position_addresses = kernel.add_pointer(query_position_values, &query_indices)?;
    let position_other = kernel.full_integer(&[config.block_m], 0, DType::I32)?;
    let query_positions =
        kernel.load_masked(&position_addresses, &query_position_mask, &position_other)?;
    let head_count = integer(kernel, config.num_query_heads, DType::I32)?;
    let query_head_mask_1d = kernel.compare(Comparison::Less, &query_heads, &head_count)?;
    let real_group_lane = kernel.compare(Comparison::Less, &head_in_group, &queries_per_kv)?;
    let query_head_mask_1d = kernel.bit_and(&query_head_mask_1d, &real_group_lane)?;
    let query_mask = kernel.mask_2d(&query_position_mask, &dimension_mask)?;
    let query_head_mask = kernel.expand_dimension(&query_head_mask_1d, 1)?;
    let query_mask = kernel.bit_and(&query_mask, &query_head_mask)?;
    let query_addresses = kernel.add_pointer(query, &query_offset)?;
    let query_zero = kernel.full_float(
        &[config.block_m, config.padded_head_size],
        0.0,
        config.dtype,
    )?;
    let query_tile = kernel.load_masked(&query_addresses, &query_mask, &query_zero)?;

    let sequence_i64 = kernel.cast(sequence, DType::I64)?;
    let block_table_base = kernel.multiply(&sequence_i64, block_table_stride)?;
    let block_table = kernel.add_pointer(block_tables, &block_table_base)?;
    let minus_infinity = kernel.full_float(&[config.block_m], f64::NEG_INFINITY, DType::F32)?;
    let denominator = kernel.full_float(&[config.block_m], 1.0, DType::F32)?;
    let accumulator =
        kernel.full_float(&[config.block_m, config.padded_head_size], 0.0, DType::F32)?;
    let sequence_length = load_offset(kernel, sequence_lengths, sequence)?;
    let maximum_prefix = if config.causal {
        let final_query_position = kernel.reduce(Reduction::Maximum, &query_positions, 0)?;
        let one = integer(kernel, 1, DType::I32)?;
        let end = kernel.add(&final_query_position, &one)?;
        kernel.minimum(&end, &sequence_length)?
    } else {
        sequence_length.clone()
    };
    let tile_size = integer(kernel, config.tile_size, DType::I32)?;
    let tile_minus_one = integer(kernel, config.tile_size - 1, DType::I32)?;
    let padded_prefix = kernel.add(&maximum_prefix, &tile_minus_one)?;
    let tile_count = kernel.divide(&padded_prefix, &tile_size)?;
    let zero_i32 = integer(kernel, 0, DType::I32)?;
    let (tile_start, tile_end) = match attention_output {
        AttentionOutput::Segment { index, count, .. } => {
            let segment_span = integer(kernel, count * config.tile_size, DType::I32)?;
            let segment_span_minus_one = integer(kernel, count * config.tile_size - 1, DType::I32)?;
            let padded_sequence = kernel.add(&sequence_length, &segment_span_minus_one)?;
            let tiles_per_segment = kernel.divide(&padded_sequence, &segment_span)?;
            let start = kernel.multiply(index, &tiles_per_segment)?;
            let one = integer(kernel, 1, DType::I32)?;
            let next_segment = kernel.add(index, &one)?;
            let end = kernel.multiply(&next_segment, &tiles_per_segment)?;
            let end = kernel.minimum(&end, &tile_count)?;
            (start, end)
        }
        AttentionOutput::Final { .. } => (zero_i32.clone(), tile_count.clone()),
    };
    let log2_e = kernel.float(LOG2_E, DType::F32)?;
    let qk_scale = kernel.multiply(scale, &log2_e)?;
    let loop_step = integer(kernel, 1, DType::I32)?;
    let carried = kernel.for_loop(
        &tile_start,
        &tile_end,
        &loop_step,
        &[minus_infinity, denominator, accumulator],
        |body, tile_index, carried| {
            let maximum = &carried[0];
            let denominator = &carried[1];
            let accumulator = &carried[2];
            let tile_base = body.multiply(&tile_index, &tile_size)?;
            let sequence_offset = body.add(&tile_base, &tile)?;
            let tile_mask = body.compare(Comparison::Less, &sequence_offset, &maximum_prefix)?;
            let page_size = integer(body, config.page_size, DType::I32)?;
            let page_offset = body.divide(&sequence_offset, &page_size)?;
            let page_addresses = body.add_pointer(&block_table, &page_offset)?;
            // A tile may extend past the sequence's final logical page. K/V
            // loads are masked below, but an unconditional page-table load
            // would already have read beyond a minimally sized table. Use
            // page zero only as an address placeholder for invalid lanes; the
            // same tile mask prevents those lanes from touching cache data.
            let invalid_page = body.full_integer(&[config.tile_size], 0, DType::I32)?;
            let physical_page = body.load_masked(&page_addresses, &tile_mask, &invalid_page)?;
            let physical_page = body.cast(&physical_page, DType::I64)?;
            let in_page = body.remainder(&sequence_offset, &page_size)?;
            let in_page_i64 = body.cast(&in_page, DType::I64)?;
            let kv_head_i64 = body.cast(kv_head, DType::I64)?;
            let dimensions_i64 = body.cast(&dimensions, DType::I64)?;
            let value_offset = cache_offset(
                body,
                &physical_page,
                &in_page_i64,
                &kv_head_i64,
                &dimensions_i64,
                value_stride_0,
                value_stride_1,
                value_stride_2,
                false,
            )?;
            let key_offset = cache_offset(
                body,
                &physical_page,
                &in_page_i64,
                &kv_head_i64,
                &dimensions_i64,
                key_stride_0,
                key_stride_1,
                key_stride_2,
                true,
            )?;
            let key_mask = body.mask_2d(&dimension_mask, &tile_mask)?;
            let value_mask = body.mask_2d(&tile_mask, &dimension_mask)?;
            let key_zero = body.full_float(
                &[config.padded_head_size, config.tile_size],
                0.0,
                config.dtype,
            )?;
            let value_zero = body.full_float(
                &[config.tile_size, config.padded_head_size],
                0.0,
                config.dtype,
            )?;
            let key_addresses = body.add_pointer(key_cache, &key_offset)?;
            let value_addresses = body.add_pointer(value_cache, &value_offset)?;
            let key_tile = body.load_masked(&key_addresses, &key_mask, &key_zero)?;
            let value_tile = body.load_masked(&value_addresses, &value_mask, &value_zero)?;
            let dot_zero = body.full_float(&[config.block_m, config.tile_size], 0.0, DType::F32)?;
            let dot = body.dot(&query_tile, &key_tile, &dot_zero)?;
            let mut scores = body.multiply(&dot, &qk_scale)?;
            let key_positions = body.expand_dimension(&sequence_offset, 0)?;
            let valid = if config.causal {
                let query_limits = body.expand_dimension(&query_positions, 1)?;
                let one = integer(body, 1, DType::I32)?;
                let query_limits = body.add(&query_limits, &one)?;
                body.compare(Comparison::Less, &key_positions, &query_limits)?
            } else {
                let tile_mask = body.expand_dimension(&tile_mask, 0)?;
                body.broadcast(&tile_mask, &[config.block_m, config.tile_size])?
            };
            let query_position_mask = body.expand_dimension(&query_position_mask, 1)?;
            let query_head_mask = body.expand_dimension(&query_head_mask_1d, 1)?;
            let query_mask = body.bit_and(&query_position_mask, &query_head_mask)?;
            let valid = body.bit_and(&valid, &query_mask)?;
            let invalid_scores = body.full_float(
                &[config.block_m, config.tile_size],
                f64::NEG_INFINITY,
                DType::F32,
            )?;
            scores = body.select(&valid, &scores, &invalid_scores)?;
            if let Some(window) = config.sliding_window {
                let distance = body.expand_dimension(&query_positions, 1)?;
                let distance = body.subtract(&distance, &key_positions)?;
                let window_value = integer(body, window, DType::I32)?;
                let mut in_window = body.compare(Comparison::Less, &distance, &window_value)?;
                if !config.causal {
                    // NML defines a noncausal window symmetrically around the
                    // authored query position. The retained ZML kernel only
                    // applied the lower (history) bound because its product
                    // callers use causal attention; keep NML's portable and
                    // Flash paths semantically identical here.
                    let negative_window = integer(body, -window, DType::I32)?;
                    let above_lower =
                        body.compare(Comparison::Greater, &distance, &negative_window)?;
                    in_window = body.bit_and(&in_window, &above_lower)?;
                }
                scores = body.select(&in_window, &scores, &invalid_scores)?;
            }
            let row_maximum = body.reduce(Reduction::Maximum, &scores, 1)?;
            let next_maximum = body.maximum(maximum, &row_maximum)?;
            let negative_infinity =
                body.full_float(&[config.block_m], f64::NEG_INFINITY, DType::F32)?;
            let finite = body.compare(Comparison::Greater, &next_maximum, &negative_infinity)?;
            let zero = body.full_float(&[config.block_m], 0.0, DType::F32)?;
            // Keep -infinity in the carried state for an all-masked tile. A
            // synthetic zero carried into a later valid tile would become an
            // incorrect maximum for entirely negative logits and could
            // underflow their exponentials. Zero is only a safe arithmetic
            // substitute for the current all-masked tile.
            let arithmetic_maximum = body.select(&finite, &next_maximum, &zero)?;
            let next_maximum_2d = body.expand_dimension(&arithmetic_maximum, 1)?;
            let centered_scores = body.subtract(&scores, &next_maximum_2d)?;
            let probabilities = body.exp2(&centered_scores)?;
            let tile_sum = body.reduce(Reduction::Sum, &probabilities, 1)?;
            let maximum_delta = body.subtract(maximum, &arithmetic_maximum)?;
            let alpha = body.exp2(&maximum_delta)?;
            let scaled_denominator = body.multiply(denominator, &alpha)?;
            let next_denominator = body.add(&scaled_denominator, &tile_sum)?;
            let alpha_2d = body.expand_dimension(&alpha, 1)?;
            let scaled_accumulator = body.multiply(accumulator, &alpha_2d)?;
            let probabilities = body.cast(&probabilities, config.dtype)?;
            let next_accumulator = body.dot(&probabilities, &value_tile, &scaled_accumulator)?;
            Ok(vec![next_maximum, next_denominator, next_accumulator])
        },
    )?;
    match attention_output {
        AttentionOutput::Final {
            output,
            stride_0,
            stride_1,
        } => {
            let denominator = kernel.expand_dimension(&carried[1], 1)?;
            let normalized_value = kernel.divide(&carried[2], &denominator)?;
            let zero = kernel.float(0.0, DType::F32)?;
            let sum_is_zero = kernel.compare(Comparison::Equal, &carried[1], &zero)?;
            let sum_is_zero = kernel.expand_dimension(&sum_is_zero, 1)?;
            let zero_output =
                kernel.full_float(&[config.block_m, config.padded_head_size], 0.0, DType::F32)?;
            let normalized = kernel.select(&sum_is_zero, &zero_output, &normalized_value)?;
            let output_value = kernel.cast(&normalized, config.dtype)?;
            let output_offset = offset_3d(
                kernel,
                &query_indices,
                &query_heads,
                &dimensions,
                stride_0,
                stride_1,
            )?;
            let output_addresses = kernel.add_pointer(output, &output_offset)?;
            kernel.store_masked(&output_addresses, &output_value, &query_mask)
        }
        AttentionOutput::Segment {
            values,
            maxima,
            sums,
            index,
            count,
        } => {
            let query_indices = kernel.cast(&query_indices, DType::I64)?;
            let query_heads = kernel.cast(&query_heads, DType::I64)?;
            let index_i64 = kernel.cast(index, DType::I64)?;
            let dimensions_i64 = kernel.cast(&dimensions, DType::I64)?;
            let token_stride = integer(
                kernel,
                config.num_query_heads * count * config.padded_head_size,
                DType::I64,
            )?;
            let head_stride = integer(kernel, count * config.padded_head_size, DType::I64)?;
            let segment_stride = integer(kernel, config.padded_head_size, DType::I64)?;
            let value_offset = offset_4d(
                kernel,
                &query_indices,
                &query_heads,
                &index_i64,
                &dimensions_i64,
                &token_stride,
                &head_stride,
                &segment_stride,
            )?;
            let value_addresses = kernel.add_pointer(values, &value_offset)?;
            kernel.store_masked(&value_addresses, &carried[2], &query_mask)?;

            let scalar_token_stride = integer(kernel, config.num_query_heads * count, DType::I64)?;
            let scalar_head_stride = integer(kernel, count, DType::I64)?;
            let token_offset = kernel.multiply(&query_indices, &scalar_token_stride)?;
            let head_offset = kernel.multiply(&query_heads, &scalar_head_stride)?;
            let scalar_offset = kernel.add(&token_offset, &head_offset)?;
            let scalar_offset = kernel.add(&scalar_offset, &index_i64)?;
            let scalar_mask = kernel.bit_and(&query_position_mask, &query_head_mask_1d)?;
            let maximum_addresses = kernel.add_pointer(maxima, &scalar_offset)?;
            kernel.store_masked(&maximum_addresses, &carried[0], &scalar_mask)?;
            let sum_addresses = kernel.add_pointer(sums, &scalar_offset)?;
            kernel.store_masked(&sum_addresses, &carried[1], &scalar_mask)
        }
    }
}

fn find_sequence(
    builder: &mut Builder,
    query_starts: &Value,
    target_block: &Value,
    sequence_count: &Value,
    block_q: i64,
    block_mode: bool,
) -> Result<Value, Error> {
    let left = integer(builder, 0, DType::I32)?;
    let carried = builder.while_loop(
        &[left, sequence_count.clone()],
        |before, carried| {
            let condition = before.compare(Comparison::Less, &carried[0], &carried[1])?;
            Ok((condition, carried.to_vec()))
        },
        |body, carried| {
            let two = integer(body, 2, DType::I32)?;
            let bounds_sum = body.add(&carried[0], &carried[1])?;
            let middle = body.divide(&bounds_sum, &two)?;
            let query_start = load_offset(body, query_starts, &middle)?;
            let block_q = integer(body, block_q, DType::I32)?;
            let boundary = if block_mode {
                let query_block = body.divide(&query_start, &block_q)?;
                body.add(&query_block, &middle)?
            } else {
                query_start
            };
            let advance = body.compare(Comparison::LessEqual, &boundary, target_block)?;
            let middle_for_else = middle.clone();
            body.if_then_else(
                &advance,
                |branch| {
                    let one = integer(branch, 1, DType::I32)?;
                    Ok(vec![branch.add(&middle, &one)?, carried[1].clone()])
                },
                |_| Ok(vec![carried[0].clone(), middle_for_else]),
            )
        },
    )?;
    let one = integer(builder, 1, DType::I32)?;
    builder.subtract(&carried[0], &one)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SegmentReductionConfig {
    pub output_dtype: DType,
    pub num_query_heads: i64,
    pub segments: i64,
    pub tile_size: i64,
    pub head_size: i64,
    pub padded_head_size: i64,
    pub block_q: i64,
}

impl SegmentReductionConfig {
    fn validate(self) -> Result<Self, Error> {
        let fits_i32 = |value: i64| i32::try_from(value).is_ok();
        if !matches!(self.output_dtype, DType::F16 | DType::Bf16 | DType::F32)
            || self.num_query_heads <= 0
            || self.segments <= 0
            || !(self.segments as u64).is_power_of_two()
            || self.tile_size <= 0
            || self.head_size <= 0
            || self.padded_head_size < self.head_size
            || !(self.padded_head_size as u64).is_power_of_two()
            || self.block_q <= 0
            || [
                self.num_query_heads,
                self.segments,
                self.tile_size,
                self.head_size,
                self.padded_head_size,
                self.block_q,
            ]
            .into_iter()
            .any(|value| !fits_i32(value))
            || self
                .segments
                .checked_mul(self.tile_size)
                .is_none_or(|value| !fits_i32(value))
        {
            return Err(Error::InvalidKernelSpec(
                "invalid paged-attention segment-reduction specialization",
            ));
        }
        Ok(self)
    }
}

pub fn build_segment_reduction(config: SegmentReductionConfig) -> Result<String, Error> {
    let config = config.validate()?;
    let mut builder = Builder::new("paged_attention_segment_reduction")?;
    let segment_output = pointer(&mut builder, "segment_output", DType::F32)?;
    let segment_maximum = pointer(&mut builder, "segment_maximum", DType::F32)?;
    let segment_sum = pointer(&mut builder, "segment_sum", DType::F32)?;
    let sequence_lengths = pointer(&mut builder, "sequence_lengths", DType::I32)?;
    let sequence_count_pointer = pointer(&mut builder, "sequence_count", DType::I32)?;
    let output_stride_0_pointer = pointer(&mut builder, "output_stride_0", DType::I64)?;
    let output_stride_1_pointer = pointer(&mut builder, "output_stride_1", DType::I64)?;
    let query_starts = pointer(&mut builder, "query_starts", DType::I32)?;
    let output = pointer(&mut builder, "output", config.output_dtype)?;

    let sequence_count = builder.load(&sequence_count_pointer)?;
    let output_stride_0 = builder.load(&output_stride_0_pointer)?;
    let output_stride_1 = builder.load(&output_stride_1_pointer)?;
    let query_token = builder.program_id(0)?;
    let query_head = builder.program_id(1)?;
    let sequence = find_sequence(
        &mut builder,
        &query_starts,
        &query_token,
        &sequence_count,
        config.block_q,
        false,
    )?;
    let sequence_length = load_offset(&mut builder, &sequence_lengths, &sequence)?;
    let segment_span = integer(&mut builder, config.segments * config.tile_size, DType::I32)?;
    let segment_span_minus_one = integer(
        &mut builder,
        config.segments * config.tile_size - 1,
        DType::I32,
    )?;
    let padded_length = builder.add(&sequence_length, &segment_span_minus_one)?;
    let tiles_per_segment = builder.divide(&padded_length, &segment_span)?;
    let tile_size = integer(&mut builder, config.tile_size, DType::I32)?;
    let segment_length = builder.multiply(&tiles_per_segment, &tile_size)?;
    let one = integer(&mut builder, 1, DType::I32)?;
    let segment_length_minus_one = builder.subtract(&segment_length, &one)?;
    let padded_sequence = builder.add(&sequence_length, &segment_length_minus_one)?;
    // Empty sequences have no producer programs in the split-K kernel. Keep
    // the reduction defined anyway: a unit divisor produces an empty segment
    // mask, and the stable maximum handling below reduces that state to zero.
    let safe_segment_length = builder.maximum(&segment_length, &one)?;
    let active_segments = builder.divide(&padded_sequence, &safe_segment_length)?;
    let segments = builder.range(0, config.segments as i32)?;
    let segment_mask = builder.compare(Comparison::Less, &segments, &active_segments)?;
    let dimensions = builder.range(0, config.padded_head_size as i32)?;
    let dimension_mask = if config.padded_head_size == config.head_size {
        builder.full_integer(&[config.padded_head_size], 1, DType::I1)?
    } else {
        let head_size = integer(&mut builder, config.head_size, DType::I32)?;
        builder.compare(Comparison::Less, &dimensions, &head_size)?
    };

    let query_token_i64 = builder.cast(&query_token, DType::I64)?;
    let query_head_i64 = builder.cast(&query_head, DType::I64)?;
    let segments_i64 = builder.cast(&segments, DType::I64)?;
    let token_stride = integer(
        &mut builder,
        config.num_query_heads * config.segments,
        DType::I64,
    )?;
    let head_stride = integer(&mut builder, config.segments, DType::I64)?;
    let token_offset = builder.multiply(&query_token_i64, &token_stride)?;
    let head_offset = builder.multiply(&query_head_i64, &head_stride)?;
    let segment_offset = builder.add(&token_offset, &head_offset)?;
    let segment_offset = builder.add(&segment_offset, &segments_i64)?;
    let maximum_addresses = builder.add_pointer(&segment_maximum, &segment_offset)?;
    let maximum_other = builder.full_float(&[config.segments], f64::NEG_INFINITY, DType::F32)?;
    let maxima = builder.load_masked(&maximum_addresses, &segment_mask, &maximum_other)?;
    let overall_maximum = builder.reduce(Reduction::Maximum, &maxima, 0)?;
    let negative_infinity = builder.float(f64::NEG_INFINITY, DType::F32)?;
    let finite = builder.compare(Comparison::Greater, &overall_maximum, &negative_infinity)?;
    let zero = builder.float(0.0, DType::F32)?;
    let overall_maximum = builder.select(&finite, &overall_maximum, &zero)?;
    let sum_addresses = builder.add_pointer(&segment_sum, &segment_offset)?;
    let sum_other = builder.full_float(&[config.segments], 0.0, DType::F32)?;
    let sums = builder.load_masked(&sum_addresses, &segment_mask, &sum_other)?;
    let maximum_delta = builder.subtract(&maxima, &overall_maximum)?;
    let rescale = builder.exp2(&maximum_delta)?;
    let sums = builder.multiply(&sums, &rescale)?;
    let overall_sum = builder.reduce(Reduction::Sum, &sums, 0)?;

    let output_token_stride = integer(
        &mut builder,
        config.num_query_heads * config.segments * config.padded_head_size,
        DType::I64,
    )?;
    let output_head_stride = integer(
        &mut builder,
        config.segments * config.padded_head_size,
        DType::I64,
    )?;
    let output_segment_stride = integer(&mut builder, config.padded_head_size, DType::I64)?;
    let output_token_base = builder.multiply(&query_token_i64, &output_token_stride)?;
    let output_head_base = builder.multiply(&query_head_i64, &output_head_stride)?;
    let output_base = builder.add(&output_token_base, &output_head_base)?;
    let segment_columns = builder.expand_dimension(&segments_i64, 1)?;
    let segment_columns = builder.multiply(&segment_columns, &output_segment_stride)?;
    let dimensions_i64 = builder.cast(&dimensions, DType::I64)?;
    let dimension_rows = builder.expand_dimension(&dimensions_i64, 0)?;
    let segment_output_offset = builder.add(&output_base, &segment_columns)?;
    let segment_output_offset = builder.add(&segment_output_offset, &dimension_rows)?;
    let segment_output_mask = builder.mask_2d(&segment_mask, &dimension_mask)?;
    let segment_output_addresses = builder.add_pointer(&segment_output, &segment_output_offset)?;
    let segment_output_other =
        builder.full_float(&[config.segments, config.padded_head_size], 0.0, DType::F32)?;
    let segment_values = builder.load_masked(
        &segment_output_addresses,
        &segment_output_mask,
        &segment_output_other,
    )?;
    let rescale = builder.expand_dimension(&rescale, 1)?;
    let segment_values = builder.multiply(&segment_values, &rescale)?;
    let accumulated = builder.reduce(Reduction::Sum, &segment_values, 0)?;
    let sum_is_zero = builder.compare(Comparison::Equal, &overall_sum, &zero)?;
    let zero_output = builder.full_float(&[config.padded_head_size], 0.0, DType::F32)?;
    let normalized = builder.divide(&accumulated, &overall_sum)?;
    let normalized = builder.select(&sum_is_zero, &zero_output, &normalized)?;
    let normalized = builder.cast(&normalized, config.output_dtype)?;

    let output_token = builder.multiply(&query_token_i64, &output_stride_0)?;
    let output_head = builder.multiply(&query_head_i64, &output_stride_1)?;
    let output_dimensions = builder.cast(&dimensions, DType::I64)?;
    let output_offset = builder.add(&output_token, &output_head)?;
    let output_offset = builder.add(&output_offset, &output_dimensions)?;
    let output_addresses = builder.add_pointer(&output, &output_offset)?;
    builder.store_masked(&output_addresses, &normalized, &dimension_mask)?;
    builder.return_void()?;
    builder.finish()
}

fn pointer(builder: &mut Builder, name: &str, element: DType) -> Result<Value, Error> {
    builder.argument(
        name,
        ArgumentKind::Pointer {
            element,
            address_space: 1,
        },
        None,
    )
}

fn integer(builder: &mut Builder, value: i64, dtype: DType) -> Result<Value, Error> {
    builder.integer(value, dtype)
}

fn load_offset(builder: &mut Builder, pointer: &Value, offset: &Value) -> Result<Value, Error> {
    let address = builder.add_pointer(pointer, offset)?;
    builder.load(&address)
}

fn offset_3d(
    builder: &mut Builder,
    first: &Value,
    second: &Value,
    third: &Value,
    first_stride: &Value,
    second_stride: &Value,
) -> Result<Value, Error> {
    let first = builder.cast(first, DType::I64)?;
    let second = builder.cast(second, DType::I64)?;
    let third = builder.cast(third, DType::I64)?;
    let first = builder.expand_dimension(&first, 1)?;
    let first = builder.multiply(&first, first_stride)?;
    let second = builder.expand_dimension(&second, 1)?;
    let second = builder.multiply(&second, second_stride)?;
    let third = builder.expand_dimension(&third, 0)?;
    let first_and_second = builder.add(&first, &second)?;
    builder.add(&first_and_second, &third)
}

#[allow(clippy::too_many_arguments)]
fn offset_4d(
    builder: &mut Builder,
    first: &Value,
    second: &Value,
    third: &Value,
    fourth: &Value,
    first_stride: &Value,
    second_stride: &Value,
    third_stride: &Value,
) -> Result<Value, Error> {
    let first = builder.expand_dimension(first, 1)?;
    let first = builder.multiply(&first, first_stride)?;
    let second = builder.expand_dimension(second, 1)?;
    let second = builder.multiply(&second, second_stride)?;
    let third = builder.multiply(third, third_stride)?;
    let fourth = builder.expand_dimension(fourth, 0)?;
    let offset = builder.add(&first, &second)?;
    let offset = builder.add(&offset, &third)?;
    builder.add(&offset, &fourth)
}

#[allow(clippy::too_many_arguments)]
fn cache_offset(
    builder: &mut Builder,
    page: &Value,
    in_page: &Value,
    head: &Value,
    dimension: &Value,
    page_stride: &Value,
    token_stride: &Value,
    head_stride: &Value,
    transpose: bool,
) -> Result<Value, Error> {
    let page = builder.multiply(page, page_stride)?;
    let token = builder.multiply(in_page, token_stride)?;
    let head = builder.multiply(head, head_stride)?;
    if transpose {
        let page = builder.expand_dimension(&page, 0)?;
        let token = builder.expand_dimension(&token, 0)?;
        let dimension = builder.expand_dimension(dimension, 1)?;
        let offset = builder.add(&page, &token)?;
        let offset = builder.add(&offset, &head)?;
        builder.add(&offset, &dimension)
    } else {
        let page = builder.expand_dimension(&page, 1)?;
        let token = builder.expand_dimension(&token, 1)?;
        let dimension = builder.expand_dimension(dimension, 0)?;
        let offset = builder.add(&page, &token)?;
        let offset = builder.add(&offset, &head)?;
        builder.add(&offset, &dimension)
    }
}
