//! Exact learned-sink correction for optimized attention kernels.
//!
//! FlashAttention returns ordinary attention output and the F32 natural-log
//! softmax normalizer. A learned sink contributes `exp(sink)` to that
//! normalizer and contributes zero to the value numerator, so the exact result
//! is `output * sigmoid(lse - sink)`. Keeping this algebra in StableHLO lets
//! FA2 and FA3 retain their upstream kernels and adds only one elementwise
//! epilogue; model code and vendor ABIs remain free of one another.

use crate::Error;
use nml_mlir::{Block, Context, StableHloBinary, StableHloUnary, Type, Value};
use nml_types::DType;

#[allow(clippy::too_many_arguments)]
pub(crate) fn correct_flash_output<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    output: Value<'context>,
    softmax_lse: Value<'context>,
    sinks: Option<Value<'context>>,
    dtype: DType,
    batch: i64,
    query_length: i64,
    query_heads: i64,
    head_dimension: i64,
    result_type: Type<'context>,
) -> Result<Value<'context>, Error> {
    let Some(sinks) = sinks else {
        return Ok(output);
    };

    let sink_type = context.ranked_tensor_type(DType::F32, &[query_heads])?;
    let sinks = if dtype == DType::F32 {
        sinks
    } else {
        append_value(block, context.convert(sinks, sink_type)?)?
    };
    let statistic_type =
        context.ranked_tensor_type(DType::F32, &[batch, query_heads, query_length])?;
    let sinks = append_value(
        block,
        context.broadcast_in_dim(sinks, statistic_type, &[1])?,
    )?;
    let difference = append_value(
        block,
        context.binary(
            StableHloBinary::Subtract,
            softmax_lse,
            sinks,
            statistic_type,
        )?,
    )?;
    let correction = append_value(
        block,
        context.unary_math(StableHloUnary::Logistic, difference, statistic_type)?,
    )?;
    let transposed_type =
        context.ranked_tensor_type(DType::F32, &[batch, query_length, query_heads])?;
    let correction = append_value(
        block,
        context.transpose(correction, transposed_type, &[0, 2, 1])?,
    )?;
    let dense_type = context.ranked_tensor_type(
        DType::F32,
        &[batch, query_length, query_heads, head_dimension],
    )?;
    let correction = append_value(
        block,
        context.broadcast_in_dim(correction, dense_type, &[0, 1, 2])?,
    )?;
    let output = if dtype == DType::F32 {
        output
    } else {
        append_value(block, context.convert(output, dense_type)?)?
    };
    let corrected = append_value(
        block,
        context.binary(StableHloBinary::Multiply, output, correction, dense_type)?,
    )?;
    if dtype == DType::F32 {
        Ok(corrected)
    } else {
        append_value(block, context.convert(corrected, result_type)?).map_err(Into::into)
    }
}

fn append_value<'context>(
    block: &mut Block<'context>,
    operation: nml_mlir::Operation<'context>,
) -> Result<Value<'context>, nml_mlir::Error> {
    let value = operation.result(0)?;
    block.append_operation(operation)?;
    Ok(value)
}
