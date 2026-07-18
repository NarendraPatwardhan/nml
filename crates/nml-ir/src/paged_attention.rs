//! Internal lowering for the paged-attention product path.
//!
//! This operation is deliberately compound. Keeping its loop state private
//! avoids exposing StableHLO control-flow details through NML's model-authoring
//! API. Target lowering selects complete StableHLO, retained Triton, FA2, or
//! FA3 under exact capability contracts. Learned sinks are native Triton state
//! and use FlashAttention's F32 LSE for an exact correction epilogue.

use crate::{
    AttentionOptions, Error, attention_backend, attention_sink,
    device_capabilities::CudaCapabilities,
};
use nml_kernel_triton::{
    AttentionGeometry, AttentionLaunch, DType as KernelDType, KernelLaunch, KernelSpec,
    PagedAttention2dConfig, PagedAttention3dConfig, SegmentReductionConfig, TensorSpec,
    build_paged_attention_2d, build_paged_attention_3d, build_segment_reduction,
    select_attention_launch,
};
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
    pub sinks: Option<Value<'context>>,
    pub query_shape: Shape,
    pub cache_shape: Shape,
    pub page_table_shape: Shape,
    pub page_table_dtype: DType,
    pub sequence_lengths_dtype: DType,
    pub query_positions_dtype: DType,
    pub result_type: Type<'context>,
    pub options: AttentionOptions,
}

/// Lowers the retained CUDA path to XLA's typed Triton custom call. Dtypes or
/// architectures outside the retained kernel envelope deliberately use the
/// portable implementation, so target selection never changes model meaning.
pub(crate) fn lower_triton<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    let query_dtype = inputs.query_shape.dtype();
    let head_dimension = inputs.query_shape.dimensions()[3];
    let page_size = inputs.cache_shape.dimensions()[1];
    let backend = attention_backend::paged(query_dtype, head_dimension, page_size, capabilities);
    // Both retained CUDA ABIs use I32 logical indices. Static geometries
    // outside that envelope remain valid NML graphs and therefore use the
    // I64 portable path instead of truncating a dimension in a kernel call.
    if backend != attention_backend::Backend::Portable && !cuda_index_geometry_supported(&inputs) {
        return lower(context, block, inputs);
    }
    match backend {
        attention_backend::Backend::Portable => return lower(context, block, inputs),
        attention_backend::Backend::CudaTriton => {}
        attention_backend::Backend::CudaFlash2 => {
            return lower_flash_or_triton(
                context,
                block,
                inputs,
                capabilities.core_count(),
                FlashVersion::Two,
            );
        }
        attention_backend::Backend::CudaFlash3 => {
            return lower_flash_or_triton(
                context,
                block,
                inputs,
                capabilities.core_count(),
                FlashVersion::Three,
            );
        }
    }

    lower_triton_kernel(context, block, inputs, capabilities.core_count())
}

fn cuda_index_geometry_supported(inputs: &Inputs<'_>) -> bool {
    let fits_i32 = |value: i64| i32::try_from(value).is_ok();
    let dimensions_fit = inputs
        .query_shape
        .dimensions()
        .iter()
        .chain(inputs.cache_shape.dimensions())
        .chain(inputs.page_table_shape.dimensions())
        .copied()
        .all(fits_i32);
    let [batch, query_length, query_heads, head_dimension] = inputs.query_shape.dimensions() else {
        return false;
    };
    let kv_heads = inputs.cache_shape.dimensions()[2];
    let page_size = inputs.cache_shape.dimensions()[1];
    let logical_pages = inputs.page_table_shape.dimensions()[1];
    let token_count = batch.checked_mul(*query_length);
    let starts_length = batch.checked_add(1);
    let padded_fits_i32 = |value: i64| {
        u64::try_from(value)
            .ok()
            .and_then(u64::checked_next_power_of_two)
            .and_then(|padded| i32::try_from(padded).ok())
            .is_some()
    };
    let head_group = query_heads.checked_div(kv_heads);
    dimensions_fit
        && token_count.is_some_and(fits_i32)
        && starts_length.is_some_and(fits_i32)
        && head_group.is_some_and(padded_fits_i32)
        && padded_fits_i32(*head_dimension)
        && padded_fits_i32(page_size)
        && logical_pages.checked_mul(page_size).is_some_and(fits_i32)
        && inputs
            .options
            .sliding_window
            .is_none_or(|window| i32::try_from(window).is_ok())
}

fn lower_triton_kernel<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
    core_count: usize,
) -> Result<Value<'context>, Error> {
    let query_dtype = inputs.query_shape.dtype();
    let [batch, query_len, query_heads, head_dim] = *inputs.query_shape.dimensions() else {
        unreachable!("paged-attention query rank is validated when authored")
    };
    let [_, page_size, kv_heads, _] = *inputs.cache_shape.dimensions() else {
        unreachable!("paged-attention cache rank is validated when authored")
    };
    let logical_pages = inputs.page_table_shape.dimensions()[1];
    let num_tokens = checked_product(batch, query_len, "CUDA attention token count")?;
    let geometry = AttentionGeometry {
        core_count,
        all_decode: query_len == 1,
        num_tokens: positive_usize(num_tokens, "CUDA attention token count")?,
        num_query_heads: positive_usize(query_heads, "CUDA attention query-head count")?,
        num_kv_heads: positive_usize(kv_heads, "CUDA attention KV-head count")?,
        head_dim: positive_usize(head_dim, "CUDA attention head dimension")?,
        batch_size: positive_usize(batch, "CUDA attention batch size")?,
        page_size: positive_usize(page_size, "CUDA attention page size")?,
        max_query_length: positive_usize(query_len, "CUDA attention query length")?,
    };
    let launch = select_attention_launch(geometry)
        .map_err(|_| Error::InvalidAttention("CUDA attention launch geometry is invalid"))?;
    let kernel_dtype = kernel_dtype(query_dtype)?;
    let queries_per_kv = query_heads / kv_heads;
    let padded_head_size =
        geometry
            .head_dim
            .checked_next_power_of_two()
            .ok_or(Error::InvalidAttention(
                "CUDA attention head padding overflows",
            ))?;
    let padded_head_size = kernel_i64(padded_head_size, "CUDA attention head padding")?;
    let sliding_window = inputs
        .options
        .sliding_window
        .map(|window| {
            i32::try_from(window)
                .map(i64::from)
                .map_err(|_| Error::InvalidAttention("CUDA attention window is too large"))
        })
        .transpose()?;
    let scale = inputs
        .options
        .scale
        .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());

    let index_i32 = |shape: &[i64]| context.ranked_tensor_type(DType::I32, shape);
    let page_table_type = index_i32(&[batch, logical_pages])?;
    let lengths_type = index_i32(&[batch])?;
    let positions_type = index_i32(&[batch, query_len])?;
    let page_table = convert_if_needed(
        context,
        block,
        inputs.page_table,
        inputs.page_table_dtype,
        DType::I32,
        page_table_type,
    )?;
    let sequence_lengths = convert_if_needed(
        context,
        block,
        inputs.sequence_lengths,
        inputs.sequence_lengths_dtype,
        DType::I32,
        lengths_type,
    )?;
    let query_positions = convert_if_needed(
        context,
        block,
        inputs.query_positions,
        inputs.query_positions_dtype,
        DType::I32,
        positions_type,
    )?;

    let scalar_f32 = context.ranked_tensor_type(DType::F32, &[])?;
    let scalar_i64 = context.ranked_tensor_type(DType::I64, &[])?;
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[])?;
    let starts_type = context.ranked_tensor_type(DType::I32, &[batch + 1])?;
    let scale = constant(context, block, scalar_f32, &format!("{scale:.17e}"))?;
    let block_table_stride = constant(context, block, scalar_i64, &logical_pages.to_string())?;
    let query_stride_0 = checked_product(query_heads, head_dim, "CUDA query stride")?;
    let query_stride_1 = head_dim;
    let cache_stride_0 = checked_product(
        page_size,
        checked_product(kv_heads, head_dim, "CUDA cache stride")?,
        "CUDA cache stride",
    )?;
    let cache_stride_1 = checked_product(kv_heads, head_dim, "CUDA cache stride")?;
    let cache_stride_2 = head_dim;
    let query_stride_0 = constant(context, block, scalar_i64, &query_stride_0.to_string())?;
    let query_stride_1 = constant(context, block, scalar_i64, &query_stride_1.to_string())?;
    let output_stride_0 = query_stride_0;
    let output_stride_1 = query_stride_1;
    let key_stride_0 = constant(context, block, scalar_i64, &cache_stride_0.to_string())?;
    let key_stride_1 = constant(context, block, scalar_i64, &cache_stride_1.to_string())?;
    let key_stride_2 = constant(context, block, scalar_i64, &cache_stride_2.to_string())?;
    let value_stride_0 = key_stride_0;
    let value_stride_1 = key_stride_1;
    let value_stride_2 = key_stride_2;
    let starts = (0..=batch)
        .map(|sequence| checked_product(sequence, query_len, "CUDA query starts"))
        .collect::<Result<Vec<_>, _>>()?;
    let starts_literal = format!(
        "[{}]",
        starts
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    let query_starts = constant(context, block, starts_type, &starts_literal)?;
    let sequence_count = constant(context, block, scalar_i32, &batch.to_string())?;

    let common_config = |block_m: usize,
                         block_q: usize,
                         tile_size: usize,
                         learned_sinks: bool|
     -> Result<PagedAttention2dConfig, Error> {
        Ok(PagedAttention2dConfig {
            dtype: kernel_dtype,
            num_query_heads: query_heads,
            queries_per_kv,
            page_size,
            tile_size: kernel_i64(tile_size, "CUDA attention tile size")?,
            head_size: head_dim,
            padded_head_size,
            block_q: kernel_i64(block_q, "CUDA attention query block")?,
            block_m: kernel_i64(block_m, "CUDA attention row block")?,
            sliding_window,
            causal: inputs.options.causal,
            learned_sinks,
        })
    };
    let tensor = |dtype, shape: &[i64]| TensorSpec::new(dtype, shape).map_err(kernel_error);
    let scalar = |dtype| tensor(dtype, &[]);
    let query_spec = tensor(kernel_dtype, &[batch, query_len, query_heads, head_dim])?;
    let cache_spec = tensor(kernel_dtype, inputs.cache_shape.dimensions())?;
    let page_table_spec = tensor(KernelDType::I32, &[batch, logical_pages])?;
    let lengths_spec = tensor(KernelDType::I32, &[batch])?;
    let positions_spec = tensor(KernelDType::I32, &[batch, query_len])?;
    let starts_spec = tensor(KernelDType::I32, &[batch + 1])?;

    let mut base_operands = vec![inputs.query];
    if let Some(sinks) = inputs.sinks {
        base_operands.push(sinks);
    }
    base_operands.extend([
        inputs.key_cache,
        inputs.value_cache,
        page_table,
        sequence_lengths,
        query_positions,
        scale,
        block_table_stride,
        query_stride_0,
        query_stride_1,
    ]);
    let cache_operands = [
        key_stride_0,
        key_stride_1,
        key_stride_2,
        value_stride_0,
        value_stride_1,
        value_stride_2,
    ];
    let mut base_specs = vec![query_spec.clone()];
    if inputs.sinks.is_some() {
        base_specs.push(tensor(kernel_dtype, &[query_heads])?);
    }
    base_specs.extend([
        cache_spec.clone(),
        cache_spec.clone(),
        page_table_spec,
        lengths_spec.clone(),
        positions_spec,
        scalar(KernelDType::F32)?,
        scalar(KernelDType::I64)?,
        scalar(KernelDType::I64)?,
        scalar(KernelDType::I64)?,
    ]);

    match launch {
        AttentionLaunch::TwoDimensional {
            block_m,
            block_q,
            tile_size,
            grid,
            warps,
            stages,
            ..
        } => {
            let config = common_config(block_m, block_q, tile_size, inputs.sinks.is_some())?;
            let ir = build_paged_attention_2d(config).map_err(kernel_error)?;
            let mut operands = base_operands;
            operands.extend([output_stride_0, output_stride_1]);
            operands.extend(cache_operands);
            operands.extend([query_starts, sequence_count]);
            let mut specs = base_specs;
            specs.extend([scalar(KernelDType::I64)?, scalar(KernelDType::I64)?]);
            specs.extend(
                (0..6)
                    .map(|_| scalar(KernelDType::I64))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            specs.extend([starts_spec, scalar(KernelDType::I32)?]);
            let specification =
                KernelSpec::new("paged_attention_2d", ir, specs, vec![query_spec], vec![])
                    .map_err(kernel_error)?;
            append_kernel(
                context,
                block,
                specification,
                &operands,
                kernel_launch(grid, warps, stages)?,
            )
        }
        AttentionLaunch::SplitK {
            block_m,
            block_q,
            tile_size,
            segments,
            attention_grid,
            attention_warps,
            attention_stages,
            reduction_grid,
            reduction_warps,
            reduction_stages,
            ..
        } => {
            let config = common_config(block_m, block_q, tile_size, false)?;
            let ir = build_paged_attention_3d(PagedAttention3dConfig {
                attention: config,
                segments: kernel_i64(segments, "CUDA attention segment count")?,
            })
            .map_err(kernel_error)?;
            let mut operands = base_operands;
            operands.extend(cache_operands);
            operands.extend([query_starts, sequence_count]);
            let mut specs = base_specs;
            specs.extend(
                (0..6)
                    .map(|_| scalar(KernelDType::I64))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            specs.extend([starts_spec.clone(), scalar(KernelDType::I32)?]);
            let segments_i64 = kernel_i64(segments, "CUDA attention segment count")?;
            let segment_values_spec = tensor(
                KernelDType::F32,
                &[num_tokens, query_heads, segments_i64, padded_head_size],
            )?;
            let segment_statistics_spec =
                tensor(KernelDType::F32, &[num_tokens, query_heads, segments_i64])?;
            let specification = KernelSpec::new(
                "paged_attention_3d",
                ir,
                specs,
                vec![
                    segment_values_spec.clone(),
                    segment_statistics_spec.clone(),
                    segment_statistics_spec.clone(),
                ],
                vec![],
            )
            .map_err(kernel_error)?;
            let operation = specification
                .lower(
                    context,
                    &operands,
                    kernel_launch(attention_grid, attention_warps, attention_stages)?,
                )
                .map_err(kernel_error)?;
            let segment_values = operation.result(0)?;
            let segment_maxima = operation.result(1)?;
            let segment_sums = operation.result(2)?;
            block.append_operation(operation)?;

            let reduction_ir = build_segment_reduction(SegmentReductionConfig {
                output_dtype: kernel_dtype,
                num_query_heads: query_heads,
                segments: segments_i64,
                tile_size: kernel_i64(tile_size, "CUDA attention tile size")?,
                head_size: head_dim,
                padded_head_size,
                block_q: kernel_i64(block_q, "CUDA attention query block")?,
                learned_sinks: inputs.sinks.is_some(),
            })
            .map_err(kernel_error)?;
            let mut reduction_operands = vec![segment_values, segment_maxima, segment_sums];
            if let Some(sinks) = inputs.sinks {
                reduction_operands.push(sinks);
            }
            reduction_operands.extend([
                sequence_lengths,
                sequence_count,
                output_stride_0,
                output_stride_1,
                query_starts,
            ]);
            let mut reduction_specs = vec![
                segment_values_spec,
                segment_statistics_spec.clone(),
                segment_statistics_spec,
            ];
            if inputs.sinks.is_some() {
                reduction_specs.push(tensor(kernel_dtype, &[query_heads])?);
            }
            reduction_specs.extend([
                lengths_spec,
                scalar(KernelDType::I32)?,
                scalar(KernelDType::I64)?,
                scalar(KernelDType::I64)?,
                starts_spec,
            ]);
            let reduction = KernelSpec::new(
                "paged_attention_segment_reduction",
                reduction_ir,
                reduction_specs,
                vec![query_spec],
                vec![],
            )
            .map_err(kernel_error)?;
            append_kernel(
                context,
                block,
                reduction,
                &reduction_operands,
                kernel_launch(reduction_grid, reduction_warps, reduction_stages)?,
            )
        }
    }
}

fn append_kernel<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    specification: KernelSpec,
    operands: &[Value<'context>],
    launch: KernelLaunch,
) -> Result<Value<'context>, Error> {
    let operation = specification
        .lower(context, operands, launch)
        .map_err(kernel_error)?;
    append_value(block, operation).map_err(Into::into)
}

#[derive(Clone, Copy)]
enum FlashVersion {
    Two,
    Three,
}

fn lower_flash_or_triton<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
    core_count: usize,
    version: FlashVersion,
) -> Result<Value<'context>, Error> {
    if !inputs.options.causal && inputs.options.sliding_window.is_none() {
        return append_paged_flash(context, block, &inputs, version);
    }
    // Upstream paged FlashAttention derives query positions by bottom-right
    // alignment against each sequence length. NML permits arbitrary positions,
    // so the upstream branch is selected at execution time only when that
    // implicit convention exactly matches the authored position tensor.
    let predicate = canonical_flash_positions(context, block, &inputs)?;
    let branch_index_type = context.ranked_tensor_type(DType::I32, &[])?;
    let branch_index = append_value(block, context.convert(predicate, branch_index_type)?)?;

    let mut triton_block = Block::new(context, &[])?;
    let triton = lower_triton_kernel(
        context,
        &mut triton_block,
        clone_inputs(&inputs),
        core_count,
    )?;
    triton_block.append_operation(context.stablehlo_return(&[triton])?)?;
    let mut triton_region = Region::new(context)?;
    triton_region.append_block(triton_block)?;

    let mut flash_block = Block::new(context, &[])?;
    let flash = append_paged_flash(context, &mut flash_block, &inputs, version)?;
    flash_block.append_operation(context.stablehlo_return(&[flash])?)?;
    let mut flash_region = Region::new(context)?;
    flash_region.append_block(flash_block)?;

    let case = context.stablehlo_case(
        branch_index,
        &[inputs.result_type],
        vec![triton_region, flash_region],
    )?;
    let result = case.result(0)?;
    block.append_operation(case)?;
    Ok(result)
}

fn append_paged_flash<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &Inputs<'context>,
    version: FlashVersion,
) -> Result<Value<'context>, Error> {
    let [batch, query_length, query_heads, head_dimension] = inputs.query_shape.dimensions() else {
        unreachable!()
    };
    let logical_pages = inputs.page_table_shape.dimensions()[1];
    let page_table_type = context.ranked_tensor_type(DType::I32, &[*batch, logical_pages])?;
    let lengths_type = context.ranked_tensor_type(DType::I32, &[*batch])?;
    let page_table = convert_if_needed(
        context,
        block,
        inputs.page_table,
        inputs.page_table_dtype,
        DType::I32,
        page_table_type,
    )?;
    let sequence_lengths = convert_if_needed(
        context,
        block,
        inputs.sequence_lengths,
        inputs.sequence_lengths_dtype,
        DType::I32,
        lengths_type,
    )?;
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
        FlashVersion::Two => context.paged_flash_attention_2_custom_call(
            inputs.query,
            inputs.key_cache,
            inputs.value_cache,
            page_table,
            sequence_lengths,
            inputs.result_type,
            lse_type,
            scale,
            inputs.options.causal,
            sliding_window,
        )?,
        FlashVersion::Three => context.paged_flash_attention_3_custom_call(
            inputs.query,
            inputs.key_cache,
            inputs.value_cache,
            page_table,
            sequence_lengths,
            inputs.result_type,
            lse_type,
            scale,
            inputs.options.causal,
            sliding_window,
        )?,
    };
    let output = call.result(0)?;
    let softmax_lse = call.result(1)?;
    block.append_operation(call)?;
    attention_sink::correct_flash_output(
        context,
        block,
        output,
        softmax_lse,
        inputs.sinks,
        inputs.query_shape.dtype(),
        *batch,
        *query_length,
        *query_heads,
        *head_dimension,
        inputs.result_type,
    )
}

fn canonical_flash_positions<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &Inputs<'context>,
) -> Result<Value<'context>, Error> {
    let [batch, query_length, _, _] = inputs.query_shape.dimensions() else {
        unreachable!()
    };
    let lengths_type = context.ranked_tensor_type(DType::I64, &[*batch])?;
    let positions_type = context.ranked_tensor_type(DType::I64, &[*batch, *query_length])?;
    let lengths = convert_if_needed(
        context,
        block,
        inputs.sequence_lengths,
        inputs.sequence_lengths_dtype,
        DType::I64,
        lengths_type,
    )?;
    let positions = convert_if_needed(
        context,
        block,
        inputs.query_positions,
        inputs.query_positions_dtype,
        DType::I64,
        positions_type,
    )?;
    let lengths = append_value(
        block,
        context.broadcast_in_dim(lengths, positions_type, &[0])?,
    )?;
    let scalar_i64 = context.ranked_tensor_type(DType::I64, &[])?;
    let query_length_value = splat(
        context,
        block,
        scalar_i64,
        positions_type,
        &query_length.to_string(),
    )?;
    let start = append_value(
        block,
        context.binary(
            StableHloBinary::Subtract,
            lengths,
            query_length_value,
            positions_type,
        )?,
    )?;
    let offsets = append_value(block, context.iota(positions_type, 1)?)?;
    let expected = append_value(block, context.add(start, offsets, positions_type)?)?;
    let matches = append_value(
        block,
        context.compare(
            positions,
            expected,
            context.ranked_tensor_type(DType::Bool, &[*batch, *query_length])?,
            StableHloComparison::Eq,
            StableHloComparisonType::Signed,
        )?,
    )?;
    reduce_bool_and(context, block, matches, &[0, 1])
}

fn clone_inputs<'context>(inputs: &Inputs<'context>) -> Inputs<'context> {
    Inputs {
        query: inputs.query,
        key_cache: inputs.key_cache,
        value_cache: inputs.value_cache,
        page_table: inputs.page_table,
        sequence_lengths: inputs.sequence_lengths,
        query_positions: inputs.query_positions,
        sinks: inputs.sinks,
        query_shape: inputs.query_shape,
        cache_shape: inputs.cache_shape,
        page_table_shape: inputs.page_table_shape,
        page_table_dtype: inputs.page_table_dtype,
        sequence_lengths_dtype: inputs.sequence_lengths_dtype,
        query_positions_dtype: inputs.query_positions_dtype,
        result_type: inputs.result_type,
        options: inputs.options,
    }
}

fn kernel_launch(grid: [usize; 3], warps: usize, stages: usize) -> Result<KernelLaunch, Error> {
    Ok(KernelLaunch {
        grid: [
            kernel_i32(grid[0], "CUDA launch grid")?,
            kernel_i32(grid[1], "CUDA launch grid")?,
            kernel_i32(grid[2], "CUDA launch grid")?,
        ],
        warps: kernel_i32(warps, "CUDA launch warp count")?,
        stages: kernel_i32(stages, "CUDA launch stage count")?,
    })
}

fn kernel_dtype(dtype: DType) -> Result<KernelDType, Error> {
    match dtype {
        DType::F16 => Ok(KernelDType::F16),
        DType::Bf16 => Ok(KernelDType::Bf16),
        DType::F32 => Ok(KernelDType::F32),
        _ => Err(Error::InvalidAttention(
            "CUDA Triton attention requires F16, BF16, or F32",
        )),
    }
}

fn positive_usize(value: i64, message: &'static str) -> Result<usize, Error> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value != 0)
        .ok_or(Error::InvalidAttention(message))
}

fn kernel_i64(value: usize, message: &'static str) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::InvalidAttention(message))
}

fn kernel_i32(value: usize, message: &'static str) -> Result<i32, Error> {
    i32::try_from(value).map_err(|_| Error::InvalidAttention(message))
}

fn checked_product(left: i64, right: i64, message: &'static str) -> Result<i64, Error> {
    left.checked_mul(right)
        .ok_or(Error::InvalidAttention(message))
}

fn kernel_error(error: nml_kernel_triton::Error) -> Error {
    match error {
        nml_kernel_triton::Error::Mlir(error) => Error::Mlir(error),
        _ => Error::InvalidAttention("CUDA Triton kernel construction failed"),
    }
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
    let (running_max, running_sum) = match inputs.sinks {
        Some(sinks) => (
            sink_statistics(
                context,
                block,
                sinks,
                inputs.query_shape.dtype(),
                *kv_heads,
                groups,
                statistic_type,
            )?,
            splat(context, block, scalar_f32, statistic_type, "1.0")?,
        ),
        None => (
            splat(
                context,
                block,
                scalar_f32,
                statistic_type,
                "-3.4028234663852886e+38",
            )?,
            splat(context, block, scalar_f32, statistic_type, "0.0")?,
        ),
    };
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

fn sink_statistics<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    sinks: Value<'context>,
    dtype: DType,
    key_value_heads: i64,
    groups: i64,
    statistic_type: Type<'context>,
) -> Result<Value<'context>, Error> {
    let query_heads = key_value_heads
        .checked_mul(groups)
        .ok_or(Error::InvalidAttention(
            "attention sink head geometry overflows",
        ))?;
    let dense_type = context.ranked_tensor_type(DType::F32, &[query_heads])?;
    let sinks = convert_if_needed(context, block, sinks, dtype, DType::F32, dense_type)?;
    let grouped_type = context.ranked_tensor_type(DType::F32, &[key_value_heads, groups])?;
    let sinks = append_value(block, context.reshape(sinks, grouped_type)?)?;
    append_value(
        block,
        context.broadcast_in_dim(sinks, statistic_type, &[1, 2])?,
    )
    .map_err(Into::into)
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
