//! CUDA lowering for the semantic paired paged-cache update.

use crate::Error;
use nml_kernel_triton::{
    DType as KernelDType, KernelLaunch, KernelSpec, OutputAlias, PagedCacheAppendConfig,
    TensorSpec, build_paged_cache_append,
};
use nml_mlir::{Block, Context, Value};
use nml_types::{DType, Shape};

pub(crate) struct Inputs<'context> {
    pub key_cache: Value<'context>,
    pub value_cache: Value<'context>,
    pub key_updates: Value<'context>,
    pub value_updates: Value<'context>,
    pub block_tables: Value<'context>,
    pub start_positions: Value<'context>,
    pub query_lengths: Value<'context>,
    pub active_rows: Value<'context>,
    pub write_mask: Value<'context>,
    pub cache_shape: Shape,
    pub updates_shape: Shape,
    pub block_tables_shape: Shape,
}

pub(crate) fn lower_triton<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
) -> Result<(Value<'context>, Value<'context>), Error> {
    let [physical_pages, page_size, heads, head_dim] = *inputs.cache_shape.dimensions() else {
        unreachable!("paired paged-cache update rank is validated when authored")
    };
    let [batch, query, _, _] = *inputs.updates_shape.dimensions() else {
        unreachable!("paired paged-cache update rank is validated when authored")
    };
    let logical_pages = inputs.block_tables_shape.dimensions()[1];
    let dtype = kernel_dtype(inputs.cache_shape.dtype())?;
    let block_elements = 128_i64;
    let head_elements = heads
        .checked_mul(head_dim)
        .ok_or(Error::InvalidIndexing(
            "paged-cache head element count overflows I64",
        ))?;
    let rows = batch
        .checked_mul(query)
        .ok_or(Error::InvalidIndexing(
            "paged-cache batch/query extent overflows I64",
        ))?;
    let element_programs = head_elements
        .checked_add(block_elements - 1)
        .and_then(|value| value.checked_div(block_elements))
        .ok_or(Error::InvalidIndexing(
            "paged-cache append grid overflows I64",
        ))?;
    let config = PagedCacheAppendConfig {
        dtype,
        batch,
        query,
        physical_pages,
        page_size,
        heads,
        head_dim,
        logical_pages,
        block_elements,
    };
    let cache = tensor(dtype, inputs.cache_shape.dimensions())?;
    let updates = tensor(dtype, inputs.updates_shape.dimensions())?;
    let block_tables = tensor(KernelDType::I32, inputs.block_tables_shape.dimensions())?;
    let batch_i32 = tensor(KernelDType::I32, &[batch])?;
    let batch_bool = tensor(KernelDType::I1, &[batch])?;
    let write_mask = tensor(KernelDType::I1, &[batch, query])?;
    let specification = KernelSpec::new(
        build_paged_cache_append(config).map_err(kernel_error)?,
        vec![
            cache.clone(),
            cache.clone(),
            updates.clone(),
            updates,
            block_tables,
            batch_i32.clone(),
            batch_i32,
            batch_bool,
            write_mask,
        ],
        vec![cache.clone(), cache],
        vec![
            OutputAlias {
                output: 0,
                input: 0,
            },
            OutputAlias {
                output: 1,
                input: 1,
            },
        ],
    )
    .map_err(kernel_error)?;
    let call = specification
        .lower(
            context,
            &[
                ("key_cache", inputs.key_cache),
                ("value_cache", inputs.value_cache),
                ("key_updates", inputs.key_updates),
                ("value_updates", inputs.value_updates),
                ("block_tables", inputs.block_tables),
                ("start_positions", inputs.start_positions),
                ("query_lengths", inputs.query_lengths),
                ("active_rows", inputs.active_rows),
                ("write_mask", inputs.write_mask),
            ],
            KernelLaunch {
                grid: [
                    i32::try_from(rows).map_err(|_| {
                        Error::InvalidIndexing("paged-cache row grid exceeds I32")
                    })?,
                    i32::try_from(element_programs).map_err(|_| {
                        Error::InvalidIndexing("paged-cache element grid exceeds I32")
                    })?,
                    1,
                ],
                warps: 4,
                stages: 1,
            },
        )
        .map_err(kernel_error)?;
    let key = call.result(0)?;
    let value = call.result(1)?;
    block.append_operation(call)?;
    Ok((key, value))
}

fn kernel_dtype(dtype: DType) -> Result<KernelDType, Error> {
    match dtype {
        DType::F16 => Ok(KernelDType::F16),
        DType::Bf16 => Ok(KernelDType::Bf16),
        DType::F32 => Ok(KernelDType::F32),
        _ => Err(Error::UnsupportedDType {
            operation: "CUDA paged-cache append",
            dtype,
        }),
    }
}

fn tensor(dtype: KernelDType, dimensions: &[i64]) -> Result<TensorSpec, Error> {
    TensorSpec::new(dtype, dimensions).map_err(kernel_error)
}

fn kernel_error(error: nml_kernel_triton::Error) -> Error {
    match error {
        nml_kernel_triton::Error::Mlir(error) => Error::Mlir(error),
        _ => Error::InvalidIndexing("CUDA paged-cache append construction failed"),
    }
}
