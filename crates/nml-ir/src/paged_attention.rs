//! Internal lowering for the portable paged-attention product path.
//!
//! This operation is deliberately compound. Keeping its loop state private
//! avoids exposing StableHLO control-flow details through NML's model-authoring
//! API, while the emitted graph still uses ordinary, portable StableHLO.

use crate::{AttentionOptions, Error};
use nml_mlir::{
    Block, Context, Operation, Region, StableHloBinary, StableHloComparison,
    StableHloComparisonType, StableHloUnary, Type, Value,
};
use nml_types::{DType, Shape};

pub(crate) struct Inputs<'context> {
    pub query: Value<'context>,
    pub key_cache: Value<'context>,
    pub value_cache: Value<'context>,
    pub page_table: Value<'context>,
    pub sequence_lengths: Value<'context>,
    pub query_positions: Value<'context>,
    pub query_shape: Shape,
    pub cache_shape: Shape,
    pub page_table_shape: Shape,
    pub page_table_dtype: DType,
    pub sequence_lengths_dtype: DType,
    pub query_positions_dtype: DType,
    pub result_type: Type<'context>,
    pub options: AttentionOptions,
}

/// Emits one bounded page traversal. The loop carries immutable operands
/// explicitly because StableHLO regions must not capture values from above.
pub(crate) fn lower<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
) -> Result<Value<'context>, Error> {
    let [batch, query_len, query_heads, head_dim] = inputs.query_shape.dimensions() else {
        unreachable!("paged-attention query rank is validated when authored")
    };
    let [physical_pages, page_size, kv_heads, cache_head_dim] = inputs.cache_shape.dimensions()
    else {
        unreachable!("paged-attention cache rank is validated when authored")
    };
    debug_assert_eq!(head_dim, cache_head_dim);
    let logical_pages = inputs.page_table_shape.dimensions()[1];
    let groups = query_heads / kv_heads;

    let scalar_i64 = context.ranked_tensor_type(DType::I64, &[])?;
    let scalar_f32 = context.ranked_tensor_type(DType::F32, &[])?;
    let scalar_bool = context.ranked_tensor_type(DType::Bool, &[])?;
    let page_table_type = context.ranked_tensor_type(DType::I64, &[*batch, logical_pages])?;
    let lengths_type = context.ranked_tensor_type(DType::I64, &[*batch])?;
    let positions_type = context.ranked_tensor_type(DType::I64, &[*batch, *query_len])?;
    let query_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, *kv_heads, groups, *query_len, *head_dim],
    )?;
    let cache_type = context.ranked_tensor_type(
        DType::F32,
        &[*physical_pages, *page_size, *kv_heads, *head_dim],
    )?;
    let statistic_type =
        context.ranked_tensor_type(DType::F32, &[*batch, *kv_heads, groups, *query_len])?;
    let accumulator_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, *kv_heads, groups, *query_len, *head_dim],
    )?;

    let query = convert_if_needed(
        context,
        block,
        inputs.query,
        inputs.query_shape.dtype(),
        DType::F32,
        context.ranked_tensor_type(DType::F32, inputs.query_shape.dimensions())?,
    )?;
    let query = append_value(
        block,
        context.reshape(
            query,
            context.ranked_tensor_type(
                DType::F32,
                &[*batch, *query_len, *kv_heads, groups, *head_dim],
            )?,
        )?,
    )?;
    let query = append_value(
        block,
        context.transpose(query, query_type, &[0, 2, 3, 1, 4])?,
    )?;
    let key_cache = convert_if_needed(
        context,
        block,
        inputs.key_cache,
        inputs.cache_shape.dtype(),
        DType::F32,
        cache_type,
    )?;
    let value_cache = convert_if_needed(
        context,
        block,
        inputs.value_cache,
        inputs.cache_shape.dtype(),
        DType::F32,
        cache_type,
    )?;
    let page_table = convert_if_needed(
        context,
        block,
        inputs.page_table,
        inputs.page_table_dtype,
        DType::I64,
        page_table_type,
    )?;
    let sequence_lengths = convert_if_needed(
        context,
        block,
        inputs.sequence_lengths,
        inputs.sequence_lengths_dtype,
        DType::I64,
        lengths_type,
    )?;
    let query_positions = convert_if_needed(
        context,
        block,
        inputs.query_positions,
        inputs.query_positions_dtype,
        DType::I64,
        positions_type,
    )?;

    let counter = constant(context, block, scalar_i64, "0")?;
    let running_max = splat(
        context,
        block,
        scalar_f32,
        statistic_type,
        "-3.4028234663852886e+38",
    )?;
    let running_sum = splat(context, block, scalar_f32, statistic_type, "0.0")?;
    let accumulator = splat(context, block, scalar_f32, accumulator_type, "0.0")?;

    let state_types = [
        scalar_i64,
        statistic_type,
        statistic_type,
        accumulator_type,
        query_type,
        cache_type,
        cache_type,
        page_table_type,
        lengths_type,
        positions_type,
    ];
    let initial = [
        counter,
        running_max,
        running_sum,
        accumulator,
        query,
        key_cache,
        value_cache,
        page_table,
        sequence_lengths,
        query_positions,
    ];

    let condition = condition_region(
        context,
        &state_types,
        logical_pages,
        scalar_i64,
        scalar_bool,
    )?;
    let body = body_region(
        context,
        &state_types,
        LoopGeometry {
            batch: *batch,
            query_len: *query_len,
            groups,
            head_dim: *head_dim,
            page_size: *page_size,
            kv_heads: *kv_heads,
        },
        inputs.options,
    )?;
    let loop_operation = context.stablehlo_while(&initial, &state_types, condition, body)?;
    let final_sum = loop_operation.result(2)?;
    let final_accumulator = loop_operation.result(3)?;
    block.append_operation(loop_operation)?;

    let sum_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, *kv_heads, groups, *query_len, *head_dim],
    )?;
    let final_sum = append_value(
        block,
        context.broadcast_in_dim(final_sum, sum_type, &[0, 1, 2, 3])?,
    )?;
    let normalized = append_value(
        block,
        context.binary(
            StableHloBinary::Divide,
            final_accumulator,
            final_sum,
            accumulator_type,
        )?,
    )?;
    let zero = splat(context, block, scalar_f32, accumulator_type, "0.0")?;
    let zero_sum = splat(context, block, scalar_f32, sum_type, "0.0")?;
    let has_values = append_value(
        block,
        context.compare(
            final_sum,
            zero_sum,
            context.ranked_tensor_type(
                DType::Bool,
                &[*batch, *kv_heads, groups, *query_len, *head_dim],
            )?,
            StableHloComparison::Gt,
            StableHloComparisonType::Float,
        )?,
    )?;
    let normalized = append_value(
        block,
        context.select(has_values, normalized, zero, accumulator_type)?,
    )?;
    let transposed_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, *query_len, *kv_heads, groups, *head_dim],
    )?;
    let normalized = append_value(
        block,
        context.transpose(normalized, transposed_type, &[0, 3, 1, 2, 4])?,
    )?;
    let dense_f32_type = context.ranked_tensor_type(DType::F32, inputs.query_shape.dimensions())?;
    let normalized = append_value(block, context.reshape(normalized, dense_f32_type)?)?;
    if inputs.query_shape.dtype() == DType::F32 {
        append_value(block, context.reshape(normalized, inputs.result_type)?).map_err(Into::into)
    } else {
        append_value(block, context.convert(normalized, inputs.result_type)?).map_err(Into::into)
    }
}

#[derive(Clone, Copy)]
struct LoopGeometry {
    batch: i64,
    query_len: i64,
    groups: i64,
    head_dim: i64,
    page_size: i64,
    kv_heads: i64,
}

fn condition_region<'context>(
    context: &'context Context,
    state_types: &[Type<'context>],
    logical_pages: i64,
    scalar_i64: Type<'context>,
    scalar_bool: Type<'context>,
) -> Result<Region<'context>, Error> {
    let mut block = Block::new(context, state_types)?;
    let counter = block.argument(0)?;
    let limit = constant(context, &mut block, scalar_i64, &logical_pages.to_string())?;
    let condition = append_value(
        &mut block,
        context.compare(
            counter,
            limit,
            scalar_bool,
            StableHloComparison::Lt,
            StableHloComparisonType::Signed,
        )?,
    )?;
    block.append_operation(context.stablehlo_return(&[condition])?)?;
    let mut region = Region::new(context)?;
    region.append_block(block)?;
    Ok(region)
}

fn body_region<'context>(
    context: &'context Context,
    state_types: &[Type<'context>],
    geometry: LoopGeometry,
    options: AttentionOptions,
) -> Result<Region<'context>, Error> {
    let scalar_i64 = context.ranked_tensor_type(DType::I64, &[])?;
    let scalar_f32 = context.ranked_tensor_type(DType::F32, &[])?;
    let mut block = Block::new(context, state_types)?;
    let state = (0..state_types.len())
        .map(|index| block.argument(index))
        .collect::<Result<Vec<_>, _>>()?;
    let [
        counter,
        old_max,
        old_sum,
        old_accumulator,
        query,
        key_cache,
        value_cache,
        page_table,
        lengths,
        query_positions,
    ] = state.as_slice()
    else {
        unreachable!("portable paged-attention loop state is fixed")
    };

    let zero_i64 = constant(context, &mut block, scalar_i64, "0")?;
    let page_ids_matrix_type = context.ranked_tensor_type(DType::I64, &[geometry.batch, 1])?;
    let page_ids = append_value(
        &mut block,
        context.dynamic_slice(
            *page_table,
            &[zero_i64, *counter],
            page_ids_matrix_type,
            &[geometry.batch, 1],
        )?,
    )?;
    let page_ids_type = context.ranked_tensor_type(DType::I64, &[geometry.batch])?;
    let page_ids = append_value(&mut block, context.reshape(page_ids, page_ids_type)?)?;
    let page_size = constant(
        context,
        &mut block,
        scalar_i64,
        &geometry.page_size.to_string(),
    )?;
    let page_base = append_value(
        &mut block,
        context.binary(StableHloBinary::Multiply, *counter, page_size, scalar_i64)?,
    )?;
    let page_base_per_batch = append_value(
        &mut block,
        context.broadcast_in_dim(page_base, page_ids_type, &[])?,
    )?;
    let active_page_type = context.ranked_tensor_type(DType::Bool, &[geometry.batch])?;
    let active_page = append_value(
        &mut block,
        context.compare(
            page_base_per_batch,
            *lengths,
            active_page_type,
            StableHloComparison::Lt,
            StableHloComparisonType::Signed,
        )?,
    )?;
    // Inactive logical slots may remain -1 in the host page table. Gather
    // still needs an in-range index before the token-tail mask is applied.
    let safe_page = splat(context, &mut block, scalar_i64, page_ids_type, "0")?;
    let page_ids = append_value(
        &mut block,
        context.select(active_page, page_ids, safe_page, page_ids_type)?,
    )?;

    let gathered_type = context.ranked_tensor_type(
        DType::F32,
        &[
            geometry.batch,
            geometry.page_size,
            geometry.kv_heads,
            geometry.head_dim,
        ],
    )?;
    let key_page = gather_page(
        context,
        &mut block,
        *key_cache,
        page_ids,
        gathered_type,
        geometry,
    )?;
    let value_page = gather_page(
        context,
        &mut block,
        *value_cache,
        page_ids,
        gathered_type,
        geometry,
    )?;
    let page_type = context.ranked_tensor_type(
        DType::F32,
        &[
            geometry.batch,
            geometry.kv_heads,
            geometry.page_size,
            geometry.head_dim,
        ],
    )?;
    let key_page = append_value(
        &mut block,
        context.transpose(key_page, page_type, &[0, 2, 1, 3])?,
    )?;
    let value_page = append_value(
        &mut block,
        context.transpose(value_page, page_type, &[0, 2, 1, 3])?,
    )?;

    let scores_type = context.ranked_tensor_type(
        DType::F32,
        &[
            geometry.batch,
            geometry.kv_heads,
            geometry.groups,
            geometry.query_len,
            geometry.page_size,
        ],
    )?;
    let scores = append_value(
        &mut block,
        context.dot_general(*query, key_page, scores_type, &[0, 1], &[0, 1], &[4], &[3])?,
    )?;
    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (geometry.head_dim as f64).sqrt());
    let scale = splat(
        context,
        &mut block,
        scalar_f32,
        scores_type,
        &format!("{scale:.17e}"),
    )?;
    let scores = append_value(
        &mut block,
        context.binary(StableHloBinary::Multiply, scores, scale, scores_type)?,
    )?;

    let position_vector_type = context.ranked_tensor_type(DType::I64, &[geometry.page_size])?;
    let offsets = append_value(&mut block, context.iota(position_vector_type, 0)?)?;
    let page_base = append_value(
        &mut block,
        context.broadcast_in_dim(page_base, position_vector_type, &[])?,
    )?;
    let key_positions = append_value(
        &mut block,
        context.add(page_base, offsets, position_vector_type)?,
    )?;
    let position_tensor_type = context.ranked_tensor_type(
        DType::I64,
        &[
            geometry.batch,
            geometry.kv_heads,
            geometry.groups,
            geometry.query_len,
            geometry.page_size,
        ],
    )?;
    let key_positions = append_value(
        &mut block,
        context.broadcast_in_dim(key_positions, position_tensor_type, &[4])?,
    )?;
    let expanded_query_positions = append_value(
        &mut block,
        context.broadcast_in_dim(*query_positions, position_tensor_type, &[0, 3])?,
    )?;
    let expanded_lengths = append_value(
        &mut block,
        context.broadcast_in_dim(*lengths, position_tensor_type, &[0])?,
    )?;
    let mask_type = context.ranked_tensor_type(
        DType::Bool,
        &[
            geometry.batch,
            geometry.kv_heads,
            geometry.groups,
            geometry.query_len,
            geometry.page_size,
        ],
    )?;
    let mut valid = append_value(
        &mut block,
        context.compare(
            key_positions,
            expanded_lengths,
            mask_type,
            StableHloComparison::Lt,
            StableHloComparisonType::Signed,
        )?,
    )?;
    let invalid = splat(
        context,
        &mut block,
        context.ranked_tensor_type(DType::Bool, &[])?,
        mask_type,
        "false",
    )?;
    if options.causal {
        let causal = append_value(
            &mut block,
            context.compare(
                key_positions,
                expanded_query_positions,
                mask_type,
                StableHloComparison::Le,
                StableHloComparisonType::Signed,
            )?,
        )?;
        valid = append_value(
            &mut block,
            context.select(causal, valid, invalid, mask_type)?,
        )?;
    }
    if let Some(window) = options.sliding_window {
        let radius = i64::try_from(window - 1)
            .map_err(|_| Error::InvalidAttention("sliding window exceeds i64"))?;
        let radius = splat(
            context,
            &mut block,
            scalar_i64,
            position_tensor_type,
            &radius.to_string(),
        )?;
        let lower = append_value(
            &mut block,
            context.binary(
                StableHloBinary::Subtract,
                expanded_query_positions,
                radius,
                position_tensor_type,
            )?,
        )?;
        let within_lower = append_value(
            &mut block,
            context.compare(
                key_positions,
                lower,
                mask_type,
                StableHloComparison::Ge,
                StableHloComparisonType::Signed,
            )?,
        )?;
        valid = append_value(
            &mut block,
            context.select(within_lower, valid, invalid, mask_type)?,
        )?;
        if !options.causal {
            let upper = append_value(
                &mut block,
                context.add(expanded_query_positions, radius, position_tensor_type)?,
            )?;
            let within_upper = append_value(
                &mut block,
                context.compare(
                    key_positions,
                    upper,
                    mask_type,
                    StableHloComparison::Le,
                    StableHloComparisonType::Signed,
                )?,
            )?;
            valid = append_value(
                &mut block,
                context.select(within_upper, valid, invalid, mask_type)?,
            )?;
        }
    }

    let masked_value = splat(
        context,
        &mut block,
        scalar_f32,
        scores_type,
        "-3.4028234663852886e+38",
    )?;
    let scores = append_value(
        &mut block,
        context.select(valid, scores, masked_value, scores_type)?,
    )?;
    let statistic_type = context.ranked_tensor_type(
        DType::F32,
        &[
            geometry.batch,
            geometry.kv_heads,
            geometry.groups,
            geometry.query_len,
        ],
    )?;
    let page_max = reduce(
        context,
        &mut block,
        scores,
        statistic_type,
        &[4],
        Reduction::Maximum,
    )?;
    let new_max = append_value(
        &mut block,
        context.binary(StableHloBinary::Maximum, *old_max, page_max, statistic_type)?,
    )?;
    let old_delta = append_value(
        &mut block,
        context.binary(StableHloBinary::Subtract, *old_max, new_max, statistic_type)?,
    )?;
    let old_scale = append_value(
        &mut block,
        context.unary_math(StableHloUnary::Exponential, old_delta, statistic_type)?,
    )?;
    let new_max_scores = append_value(
        &mut block,
        context.broadcast_in_dim(new_max, scores_type, &[0, 1, 2, 3])?,
    )?;
    let page_delta = append_value(
        &mut block,
        context.binary(
            StableHloBinary::Subtract,
            scores,
            new_max_scores,
            scores_type,
        )?,
    )?;
    let page_weights = append_value(
        &mut block,
        context.unary_math(StableHloUnary::Exponential, page_delta, scores_type)?,
    )?;
    let zero_scores = splat(context, &mut block, scalar_f32, scores_type, "0.0")?;
    let page_weights = append_value(
        &mut block,
        context.select(valid, page_weights, zero_scores, scores_type)?,
    )?;
    let page_sum = reduce(
        context,
        &mut block,
        page_weights,
        statistic_type,
        &[4],
        Reduction::Sum,
    )?;
    let scaled_old_sum = append_value(
        &mut block,
        context.binary(
            StableHloBinary::Multiply,
            *old_sum,
            old_scale,
            statistic_type,
        )?,
    )?;
    let new_sum = append_value(
        &mut block,
        context.add(scaled_old_sum, page_sum, statistic_type)?,
    )?;

    let accumulator_type = context.ranked_tensor_type(
        DType::F32,
        &[
            geometry.batch,
            geometry.kv_heads,
            geometry.groups,
            geometry.query_len,
            geometry.head_dim,
        ],
    )?;
    let page_accumulator = append_value(
        &mut block,
        context.dot_general(
            page_weights,
            value_page,
            accumulator_type,
            &[0, 1],
            &[0, 1],
            &[4],
            &[2],
        )?,
    )?;
    let old_scale = append_value(
        &mut block,
        context.broadcast_in_dim(old_scale, accumulator_type, &[0, 1, 2, 3])?,
    )?;
    let old_accumulator = append_value(
        &mut block,
        context.binary(
            StableHloBinary::Multiply,
            *old_accumulator,
            old_scale,
            accumulator_type,
        )?,
    )?;
    let new_accumulator = append_value(
        &mut block,
        context.add(old_accumulator, page_accumulator, accumulator_type)?,
    )?;
    let one = constant(context, &mut block, scalar_i64, "1")?;
    let next_counter = append_value(&mut block, context.add(*counter, one, scalar_i64)?)?;
    block.append_operation(context.stablehlo_return(&[
        next_counter,
        new_max,
        new_sum,
        new_accumulator,
        *query,
        *key_cache,
        *value_cache,
        *page_table,
        *lengths,
        *query_positions,
    ])?)?;
    let mut region = Region::new(context)?;
    region.append_block(block)?;
    Ok(region)
}

fn gather_page<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    cache: Value<'context>,
    page_ids: Value<'context>,
    result_type: Type<'context>,
    geometry: LoopGeometry,
) -> Result<Value<'context>, Error> {
    append_value(
        block,
        context.gather(
            cache,
            page_ids,
            result_type,
            &[1, 2, 3],
            &[0],
            &[],
            &[],
            &[0],
            1,
            &[1, geometry.page_size, geometry.kv_heads, geometry.head_dim],
            false,
        )?,
    )
    .map_err(Into::into)
}

#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    Maximum,
}

fn reduce<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    input: Value<'context>,
    result_type: Type<'context>,
    dimensions: &[i64],
    reduction: Reduction,
) -> Result<Value<'context>, Error> {
    let scalar_type = context.ranked_tensor_type(DType::F32, &[])?;
    let identity = match reduction {
        Reduction::Sum => "0.0",
        Reduction::Maximum => "-3.4028234663852886e+38",
    };
    let init = constant(context, block, scalar_type, identity)?;
    let mut reduction_block = Block::new(context, &[scalar_type, scalar_type])?;
    let left = reduction_block.argument(0)?;
    let right = reduction_block.argument(1)?;
    let combine = match reduction {
        Reduction::Sum => context.add(left, right, scalar_type)?,
        Reduction::Maximum => context.binary(StableHloBinary::Maximum, left, right, scalar_type)?,
    };
    let combined = combine.result(0)?;
    reduction_block.append_operation(combine)?;
    reduction_block.append_operation(context.stablehlo_return(&[combined])?)?;
    let mut body = Region::new(context)?;
    body.append_block(reduction_block)?;
    append_value(
        block,
        context.reduce(input, init, result_type, dimensions, body)?,
    )
    .map_err(Into::into)
}

fn convert_if_needed<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    value: Value<'context>,
    source_dtype: DType,
    destination_dtype: DType,
    result_type: Type<'context>,
) -> Result<Value<'context>, Error> {
    if source_dtype == destination_dtype {
        Ok(value)
    } else {
        append_value(block, context.convert(value, result_type)?).map_err(Into::into)
    }
}

fn splat<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    scalar_type: Type<'context>,
    result_type: Type<'context>,
    literal: &str,
) -> Result<Value<'context>, Error> {
    let scalar = constant(context, block, scalar_type, literal)?;
    append_value(block, context.broadcast_in_dim(scalar, result_type, &[])?).map_err(Into::into)
}

fn constant<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    result_type: Type<'context>,
    literal: &str,
) -> Result<Value<'context>, Error> {
    let attribute = format!("dense<{literal}> : {}", result_type.text());
    append_value(
        block,
        context.constant(result_type, context.parse_attribute(&attribute)?)?,
    )
    .map_err(Into::into)
}

fn append_value<'context>(
    block: &mut Block<'context>,
    operation: Operation<'context>,
) -> Result<Value<'context>, nml_mlir::Error> {
    let value = operation.result(0)?;
    block.append_operation(operation)?;
    Ok(value)
}
