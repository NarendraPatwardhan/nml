//! Internal lowering for dense attention.
//!
//! The authored graph retains attention as one semantic operation so target
//! selection can happen after PJRT capability discovery.  The portable graph
//! remains complete and is emitted for CPU, unsupported CUDA devices, dtypes,
//! and layouts.  FA2 is only reachable for its exact dense ABI.

use crate::{attention_backend, AttentionOptions, Error};
use nml_mlir::{
    Block, Context, Region, StableHloBinary, StableHloComparison, StableHloComparisonType,
    StableHloUnary, Type, Value,
};
use nml_types::{DType, Shape};

pub(crate) struct Inputs<'context> {
    pub query: Value<'context>,
    pub key: Value<'context>,
    pub value: Value<'context>,
    pub query_positions: Value<'context>,
    pub key_positions: Value<'context>,
    pub query_shape: Shape,
    pub key_shape: Shape,
    pub query_positions_dtype: DType,
    pub key_positions_dtype: DType,
    pub result_type: Type<'context>,
    pub options: AttentionOptions,
}

pub(crate) fn lower_cuda<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
    capability_major: u16,
    capability_minor: u16,
) -> Result<Value<'context>, Error> {
    let head_dimension = inputs.query_shape.dimensions()[3];
    let backend = attention_backend::dense(
        inputs.query_shape.dtype(),
        head_dimension,
        capability_major,
        capability_minor,
    );
    if backend != attention_backend::Backend::Portable && !flash_index_geometry_supported(&inputs) {
        return lower(context, block, inputs);
    }
    match backend {
        attention_backend::Backend::Portable => lower(context, block, inputs),
        // FA3 has a distinct upstream ABI and is intentionally not routed
        // through FA2 even though both implement the same graph semantics.
        attention_backend::Backend::CudaFlash3 => lower_flash3(context, block, inputs),
        attention_backend::Backend::CudaFlash2 => lower_flash2(context, block, inputs),
        attention_backend::Backend::CudaTriton => {
            unreachable!("dense attention never selects Triton")
        }
    }
}

fn flash_index_geometry_supported(inputs: &Inputs<'_>) -> bool {
    let fits_i32 = |value: i64| i32::try_from(value).is_ok();
    let dimensions_fit = inputs
        .query_shape
        .dimensions()
        .iter()
        .chain(inputs.key_shape.dimensions())
        .copied()
        .all(fits_i32);
    let batch = inputs.query_shape.dimensions()[0];
    let query_length = inputs.query_shape.dimensions()[1];
    let key_length = inputs.key_shape.dimensions()[1];
    dimensions_fit
        && batch.checked_mul(query_length).is_some_and(fits_i32)
        && batch.checked_mul(key_length).is_some_and(fits_i32)
        && inputs
            .options
            .sliding_window
            .is_none_or(|window| i32::try_from(window).is_ok())
}

fn lower_flash3<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
) -> Result<Value<'context>, Error> {
    lower_flash(context, block, inputs, FlashVersion::Three)
}

fn lower_flash2<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
) -> Result<Value<'context>, Error> {
    lower_flash(context, block, inputs, FlashVersion::Two)
}

#[derive(Clone, Copy)]
enum FlashVersion {
    Two,
    Three,
}

fn lower_flash<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
    version: FlashVersion,
) -> Result<Value<'context>, Error> {
    if !inputs.options.causal && inputs.options.sliding_window.is_none() {
        return append_flash(context, block, &inputs, version);
    }

    // Upstream dense FlashAttention applies bottom-right aligned causal/local masks. NML
    // accepts arbitrary runtime positions, so the optimized branch is legal
    // only when K is [0..K) and Q is [K-Q..K). The other branch remains the
    // exact portable semantics rather than silently changing the mask.
    let predicate = canonical_positions(context, block, &inputs)?;
    let branch_index_type = context.ranked_tensor_type(DType::I32, &[])?;
    let branch_index = append_value(block, context.convert(predicate, branch_index_type)?)?;

    let mut portable_block = Block::new(context, &[])?;
    let portable = lower(context, &mut portable_block, clone_inputs(&inputs))?;
    portable_block.append_operation(context.stablehlo_return(&[portable])?)?;
    let mut portable_region = Region::new(context)?;
    portable_region.append_block(portable_block)?;

    let mut flash_block = Block::new(context, &[])?;
    let flash = append_flash(context, &mut flash_block, &inputs, version)?;
    flash_block.append_operation(context.stablehlo_return(&[flash])?)?;
    let mut flash_region = Region::new(context)?;
    flash_region.append_block(flash_block)?;

    let case = context.stablehlo_case(
        branch_index,
        &[inputs.result_type],
        vec![portable_region, flash_region],
    )?;
    let result = case.result(0)?;
    block.append_operation(case)?;
    Ok(result)
}

fn append_flash<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &Inputs<'context>,
    version: FlashVersion,
) -> Result<Value<'context>, Error> {
    let [batch, query_length, query_heads, head_dimension] = inputs.query_shape.dimensions() else {
        unreachable!("attention query rank is validated when authored")
    };
    let scale = inputs
        .options
        .scale
        .unwrap_or_else(|| 1.0 / (*head_dimension as f64).sqrt()) as f32;
    let sliding_window = inputs
        .options
        .sliding_window
        .map(|window| {
            i32::try_from(window).map_err(|_| Error::InvalidAttention("sliding window exceeds I32"))
        })
        .transpose()?
        .unwrap_or(-1);
    let lse_type =
        context.ranked_tensor_type(DType::F32, &[*batch, *query_heads, *query_length])?;
    let call = match version {
        FlashVersion::Two => context.flash_attention_2_custom_call(
            inputs.query,
            inputs.key,
            inputs.value,
            inputs.result_type,
            lse_type,
            scale,
            inputs.options.causal,
            sliding_window,
        )?,
        FlashVersion::Three => context.flash_attention_3_custom_call(
            inputs.query,
            inputs.key,
            inputs.value,
            inputs.result_type,
            lse_type,
            scale,
            inputs.options.causal,
            sliding_window,
        )?,
    };
    let output = call.result(0)?;
    block.append_operation(call)?;
    Ok(output)
}

fn canonical_positions<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &Inputs<'context>,
) -> Result<Value<'context>, Error> {
    let [batch, query_length, _, _] = inputs.query_shape.dimensions() else {
        unreachable!()
    };
    let key_length = inputs.key_shape.dimensions()[1];
    let query_type = context.ranked_tensor_type(DType::I64, &[*batch, *query_length])?;
    let key_type = context.ranked_tensor_type(DType::I64, &[*batch, key_length])?;
    let query_positions = convert_if_needed(
        context,
        block,
        inputs.query_positions,
        inputs.query_positions_dtype,
        DType::I64,
        query_type,
    )?;
    let key_positions = convert_if_needed(
        context,
        block,
        inputs.key_positions,
        inputs.key_positions_dtype,
        DType::I64,
        key_type,
    )?;
    let query_expected = append_value(block, context.iota(query_type, 1)?)?;
    let query_offset = splat(
        context,
        block,
        context.ranked_tensor_type(DType::I64, &[])?,
        query_type,
        &(key_length - query_length).to_string(),
    )?;
    let query_expected = append_value(
        block,
        context.add(query_expected, query_offset, query_type)?,
    )?;
    let key_expected = append_value(block, context.iota(key_type, 1)?)?;
    let query_matches = append_value(
        block,
        context.compare(
            query_positions,
            query_expected,
            context.ranked_tensor_type(DType::Bool, &[*batch, *query_length])?,
            StableHloComparison::Eq,
            StableHloComparisonType::Signed,
        )?,
    )?;
    let key_matches = append_value(
        block,
        context.compare(
            key_positions,
            key_expected,
            context.ranked_tensor_type(DType::Bool, &[*batch, key_length])?,
            StableHloComparison::Eq,
            StableHloComparisonType::Signed,
        )?,
    )?;
    let query_matches = reduce_bool_and(context, block, query_matches, &[0, 1])?;
    let key_matches = reduce_bool_and(context, block, key_matches, &[0, 1])?;
    let scalar_bool = context.ranked_tensor_type(DType::Bool, &[])?;
    append_value(
        block,
        context.binary(
            StableHloBinary::And,
            query_matches,
            key_matches,
            scalar_bool,
        )?,
    )
    .map_err(Into::into)
}

pub(crate) fn lower<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
) -> Result<Value<'context>, Error> {
    let [batch, query_length, query_heads, head_dimension] = inputs.query_shape.dimensions() else {
        unreachable!("attention query rank is validated when authored")
    };
    let key_length = inputs.key_shape.dimensions()[1];
    let key_value_heads = inputs.key_shape.dimensions()[2];
    let groups = query_heads / key_value_heads;
    let query_dense_type =
        context.ranked_tensor_type(DType::F32, inputs.query_shape.dimensions())?;
    let key_dense_type = context.ranked_tensor_type(DType::F32, inputs.key_shape.dimensions())?;
    let query = convert_if_needed(
        context,
        block,
        inputs.query,
        inputs.query_shape.dtype(),
        DType::F32,
        query_dense_type,
    )?;
    let key = convert_if_needed(
        context,
        block,
        inputs.key,
        inputs.key_shape.dtype(),
        DType::F32,
        key_dense_type,
    )?;
    let value = convert_if_needed(
        context,
        block,
        inputs.value,
        inputs.key_shape.dtype(),
        DType::F32,
        key_dense_type,
    )?;
    let grouped_query_type = context.ranked_tensor_type(
        DType::F32,
        &[
            *batch,
            *query_length,
            key_value_heads,
            groups,
            *head_dimension,
        ],
    )?;
    let query = append_value(block, context.reshape(query, grouped_query_type)?)?;
    let query_type = context.ranked_tensor_type(
        DType::F32,
        &[
            *batch,
            key_value_heads,
            groups,
            *query_length,
            *head_dimension,
        ],
    )?;
    let query = append_value(
        block,
        context.transpose(query, query_type, &[0, 2, 3, 1, 4])?,
    )?;
    let key_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, key_value_heads, key_length, *head_dimension],
    )?;
    let key = append_value(block, context.transpose(key, key_type, &[0, 2, 1, 3])?)?;
    let value = append_value(block, context.transpose(value, key_type, &[0, 2, 1, 3])?)?;
    let scores_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, key_value_heads, groups, *query_length, key_length],
    )?;
    let scores = append_value(
        block,
        context.dot_general(query, key, scores_type, &[0, 1], &[0, 1], &[4], &[3])?,
    )?;
    let scale = inputs
        .options
        .scale
        .unwrap_or_else(|| 1.0 / (*head_dimension as f64).sqrt());
    let scale = splat(
        context,
        block,
        context.ranked_tensor_type(DType::F32, &[])?,
        scores_type,
        &format!("{scale:.17e}"),
    )?;
    let mut scores = append_value(
        block,
        context.binary(StableHloBinary::Multiply, scores, scale, scores_type)?,
    )?;

    let positions_type = context.ranked_tensor_type(
        DType::I64,
        &[*batch, key_value_heads, groups, *query_length, key_length],
    )?;
    let query_positions_type = context.ranked_tensor_type(DType::I64, &[*batch, *query_length])?;
    let key_positions_type = context.ranked_tensor_type(DType::I64, &[*batch, key_length])?;
    let query_positions = convert_if_needed(
        context,
        block,
        inputs.query_positions,
        inputs.query_positions_dtype,
        DType::I64,
        query_positions_type,
    )?;
    let key_positions = convert_if_needed(
        context,
        block,
        inputs.key_positions,
        inputs.key_positions_dtype,
        DType::I64,
        key_positions_type,
    )?;
    let query_positions = append_value(
        block,
        context.broadcast_in_dim(query_positions, positions_type, &[0, 3])?,
    )?;
    let key_positions = append_value(
        block,
        context.broadcast_in_dim(key_positions, positions_type, &[0, 4])?,
    )?;
    let mask_type = context.ranked_tensor_type(
        DType::Bool,
        &[*batch, key_value_heads, groups, *query_length, key_length],
    )?;
    let scalar_bool = context.ranked_tensor_type(DType::Bool, &[])?;
    let mut valid = splat(context, block, scalar_bool, mask_type, "true")?;
    let invalid = splat(context, block, scalar_bool, mask_type, "false")?;
    if inputs.options.causal {
        let causal = append_value(
            block,
            context.compare(
                key_positions,
                query_positions,
                mask_type,
                StableHloComparison::Le,
                StableHloComparisonType::Signed,
            )?,
        )?;
        valid = append_value(block, context.select(causal, valid, invalid, mask_type)?)?;
    }
    if let Some(window) = inputs.options.sliding_window {
        let radius = i64::try_from(window - 1)
            .map_err(|_| Error::InvalidAttention("sliding window exceeds I64"))?;
        let radius = splat(
            context,
            block,
            context.ranked_tensor_type(DType::I64, &[])?,
            positions_type,
            &radius.to_string(),
        )?;
        let lower = append_value(
            block,
            context.binary(
                StableHloBinary::Subtract,
                query_positions,
                radius,
                positions_type,
            )?,
        )?;
        let within_lower = append_value(
            block,
            context.compare(
                key_positions,
                lower,
                mask_type,
                StableHloComparison::Ge,
                StableHloComparisonType::Signed,
            )?,
        )?;
        valid = append_value(
            block,
            context.select(within_lower, valid, invalid, mask_type)?,
        )?;
        if !inputs.options.causal {
            let upper = append_value(block, context.add(query_positions, radius, positions_type)?)?;
            let within_upper = append_value(
                block,
                context.compare(
                    key_positions,
                    upper,
                    mask_type,
                    StableHloComparison::Le,
                    StableHloComparisonType::Signed,
                )?,
            )?;
            valid = append_value(
                block,
                context.select(within_upper, valid, invalid, mask_type)?,
            )?;
        }
    }
    let masked = splat(
        context,
        block,
        context.ranked_tensor_type(DType::F32, &[])?,
        scores_type,
        "-3.4028234663852886e+38",
    )?;
    scores = append_value(block, context.select(valid, scores, masked, scores_type)?)?;
    let statistic_type = context.ranked_tensor_type(
        DType::F32,
        &[*batch, key_value_heads, groups, *query_length],
    )?;
    let maximum = reduce_f32(
        context,
        block,
        scores,
        statistic_type,
        &[4],
        Reduction::Maximum,
    )?;
    let maximum = append_value(
        block,
        context.broadcast_in_dim(maximum, scores_type, &[0, 1, 2, 3])?,
    )?;
    let shifted = append_value(
        block,
        context.binary(StableHloBinary::Subtract, scores, maximum, scores_type)?,
    )?;
    let weights = append_value(
        block,
        context.unary_math(StableHloUnary::Exponential, shifted, scores_type)?,
    )?;
    let zero_scores = splat(
        context,
        block,
        context.ranked_tensor_type(DType::F32, &[])?,
        scores_type,
        "0.0",
    )?;
    let weights = append_value(
        block,
        context.select(valid, weights, zero_scores, scores_type)?,
    )?;
    let denominator = reduce_f32(
        context,
        block,
        weights,
        statistic_type,
        &[4],
        Reduction::Sum,
    )?;
    let denominator = append_value(
        block,
        context.broadcast_in_dim(denominator, scores_type, &[0, 1, 2, 3])?,
    )?;
    let normalized = append_value(
        block,
        context.binary(StableHloBinary::Divide, weights, denominator, scores_type)?,
    )?;
    let has_values = append_value(
        block,
        context.compare(
            denominator,
            zero_scores,
            mask_type,
            StableHloComparison::Gt,
            StableHloComparisonType::Float,
        )?,
    )?;
    let weights = append_value(
        block,
        context.select(has_values, normalized, zero_scores, scores_type)?,
    )?;
    let output_type = context.ranked_tensor_type(
        DType::F32,
        &[
            *batch,
            key_value_heads,
            groups,
            *query_length,
            *head_dimension,
        ],
    )?;
    let output = append_value(
        block,
        context.dot_general(weights, value, output_type, &[0, 1], &[0, 1], &[4], &[2])?,
    )?;
    let output_transposed_type = context.ranked_tensor_type(
        DType::F32,
        &[
            *batch,
            *query_length,
            key_value_heads,
            groups,
            *head_dimension,
        ],
    )?;
    let output = append_value(
        block,
        context.transpose(output, output_transposed_type, &[0, 3, 1, 2, 4])?,
    )?;
    let output = append_value(block, context.reshape(output, query_dense_type)?)?;
    if inputs.query_shape.dtype() == DType::F32 {
        append_value(block, context.reshape(output, inputs.result_type)?).map_err(Into::into)
    } else {
        append_value(block, context.convert(output, inputs.result_type)?).map_err(Into::into)
    }
}

fn clone_inputs<'context>(inputs: &Inputs<'context>) -> Inputs<'context> {
    Inputs {
        query: inputs.query,
        key: inputs.key,
        value: inputs.value,
        query_positions: inputs.query_positions,
        key_positions: inputs.key_positions,
        query_shape: inputs.query_shape,
        key_shape: inputs.key_shape,
        query_positions_dtype: inputs.query_positions_dtype,
        key_positions_dtype: inputs.key_positions_dtype,
        result_type: inputs.result_type,
        options: inputs.options,
    }
}

#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    Maximum,
}

fn reduce_f32<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    input: Value<'context>,
    result_type: Type<'context>,
    dimensions: &[i64],
    reduction: Reduction,
) -> Result<Value<'context>, Error> {
    let scalar = context.ranked_tensor_type(DType::F32, &[])?;
    let init = constant(
        context,
        block,
        scalar,
        match reduction {
            Reduction::Sum => "0.0",
            Reduction::Maximum => "-3.4028234663852886e+38",
        },
    )?;
    let mut reduction_block = Block::new(context, &[scalar, scalar])?;
    let left = reduction_block.argument(0)?;
    let right = reduction_block.argument(1)?;
    let operation = match reduction {
        Reduction::Sum => context.add(left, right, scalar)?,
        Reduction::Maximum => context.binary(StableHloBinary::Maximum, left, right, scalar)?,
    };
    let result = operation.result(0)?;
    reduction_block.append_operation(operation)?;
    reduction_block.append_operation(context.stablehlo_return(&[result])?)?;
    let mut body = Region::new(context)?;
    body.append_block(reduction_block)?;
    append_value(
        block,
        context.reduce(input, init, result_type, dimensions, body)?,
    )
    .map_err(Into::into)
}

fn reduce_bool_and<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    input: Value<'context>,
    dimensions: &[i64],
) -> Result<Value<'context>, Error> {
    let scalar = context.ranked_tensor_type(DType::Bool, &[])?;
    let init = constant(context, block, scalar, "true")?;
    let mut reduction_block = Block::new(context, &[scalar, scalar])?;
    let left = reduction_block.argument(0)?;
    let right = reduction_block.argument(1)?;
    let operation = context.binary(StableHloBinary::And, left, right, scalar)?;
    let result = operation.result(0)?;
    reduction_block.append_operation(operation)?;
    reduction_block.append_operation(context.stablehlo_return(&[result])?)?;
    let mut body = Region::new(context)?;
    body.append_block(reduction_block)?;
    append_value(
        block,
        context.reduce(input, init, scalar, dimensions, body)?,
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
    let value = constant(context, block, scalar_type, literal)?;
    append_value(block, context.broadcast_in_dim(value, result_type, &[])?).map_err(Into::into)
}

fn constant<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    type_: Type<'context>,
    literal: &str,
) -> Result<Value<'context>, Error> {
    let attribute = context.parse_attribute(&format!("dense<{literal}> : {}", type_.text()))?;
    append_value(block, context.constant(type_, attribute)?).map_err(Into::into)
}

fn append_value<'context>(
    block: &mut Block<'context>,
    operation: nml_mlir::Operation<'context>,
) -> Result<Value<'context>, nml_mlir::Error> {
    let value = operation.result(0)?;
    block.append_operation(operation)?;
    Ok(value)
}
