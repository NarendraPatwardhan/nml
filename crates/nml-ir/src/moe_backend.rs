//! Private CUDA specialization for NML's semantic MoE operation.
//!
//! StableHLO owns routing, deterministic assignment order, and CPU behavior.
//! CUDA replaces only the two expert projections, then reduces the selected
//! expert contributions with ordinary StableHLO. This keeps backend policy out
//! of model code and gives XLA a single explicit custom-call boundary per GEMM.

use crate::{Error, MoeActivation, device_capabilities::CudaCapabilities};
use nml_kernel_triton::{
    DType as KernelDType, GatedActivation, GroupedProjectionConfig, KernelLaunch, KernelSpec,
    TensorSpec, build_grouped_projection,
};
use nml_mlir::{Block, Context, Region, Type, Value};
use nml_types::{DType, Shape};

pub(crate) struct Inputs<'context> {
    pub hidden: Value<'context>,
    pub routing_weights: Value<'context>,
    pub gate_up_weights: Value<'context>,
    pub down_weights: Value<'context>,
    pub sorted_assignments: Value<'context>,
    pub block_experts: Value<'context>,
    pub active_blocks: Value<'context>,
    pub expert_offset: Option<Value<'context>>,
    pub hidden_shape: Shape,
    pub gate_up_shape: Shape,
    pub down_shape: Shape,
    pub schedule_shape: Shape,
    pub block_experts_shape: Shape,
    pub result_type: Type<'context>,
    pub activation: MoeActivation,
    pub experts_per_token: usize,
    pub block_size: usize,
}

pub(crate) fn supported(dtype: DType, capabilities: CudaCapabilities) -> bool {
    capabilities.supports_grouped_moe() && matches!(dtype, DType::F16 | DType::Bf16 | DType::F32)
}

pub(crate) fn lower<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Inputs<'context>,
) -> Result<Value<'context>, Error> {
    let dtype = kernel_dtype(inputs.hidden_shape.dtype())?;
    let tokens = inputs.hidden_shape.dimensions()[0];
    let hidden_size = inputs.hidden_shape.dimensions()[1];
    let experts_per_token = i64::try_from(inputs.experts_per_token)
        .map_err(|_| Error::InvalidMoe("experts per token exceeds I64"))?;
    let assignments = tokens
        .checked_mul(experts_per_token)
        .ok_or(Error::InvalidMoe("assignment count overflows"))?;
    let intermediate = inputs.down_shape.dimensions()[2];
    let local_experts = inputs.gate_up_shape.dimensions()[0];
    let gate_up_width = intermediate
        .checked_mul(2)
        .ok_or(Error::InvalidMoe("gate/up width overflows"))?;
    if inputs.gate_up_shape.dimensions()[1] != gate_up_width
        || inputs.gate_up_shape.dimensions()[2] != hidden_size
        || inputs.down_shape.dimensions()[1] != hidden_size
    {
        return Err(Error::InvalidMoe(
            "grouped expert logical dimensions are inconsistent",
        ));
    }
    let block_m = i64::try_from(inputs.block_size)
        .map_err(|_| Error::InvalidMoe("expert block size exceeds I64"))?;
    let block_n = 32_i64;
    let block_k = 32_i64;
    let block_count = inputs.block_experts_shape.dimensions()[0];

    let gate_up_config = GroupedProjectionConfig {
        dtype,
        assignments,
        input_size: hidden_size,
        output_size: intermediate,
        local_experts,
        source_row_divisor: experts_per_token,
        block_m,
        block_n,
        block_k,
        gated_activation: Some(match inputs.activation {
            MoeActivation::Silu => GatedActivation::Silu,
            MoeActivation::Gelu => GatedActivation::Gelu,
            MoeActivation::Relu => GatedActivation::Relu,
        }),
        multiply_routing_weight: false,
    };
    let gate_up_spec = KernelSpec::new(
        build_grouped_projection(gate_up_config).map_err(kernel_error)?,
        vec![
            tensor(dtype, inputs.hidden_shape.dimensions())?,
            tensor(KernelDType::I32, inputs.schedule_shape.dimensions())?,
            tensor(KernelDType::I32, inputs.block_experts_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
            tensor(dtype, inputs.gate_up_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
        ],
        vec![tensor(dtype, &[assignments, intermediate])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[])?;
    let expert_offset = match inputs.expert_offset {
        Some(value) => value,
        None => constant(context, block, scalar_i32, "0")?,
    };
    let gate_up_call = gate_up_spec
        .lower(
            context,
            &[
                ("input", inputs.hidden),
                ("sorted_assignments", inputs.sorted_assignments),
                ("block_experts", inputs.block_experts),
                ("active_blocks", inputs.active_blocks),
                ("weights", inputs.gate_up_weights),
                ("expert_offset", expert_offset),
            ],
            launch(block_count, intermediate, block_n)?,
        )
        .map_err(kernel_error)?;
    let gate_up = gate_up_call.result(0)?;
    block.append_operation(gate_up_call)?;

    let down_config = GroupedProjectionConfig {
        dtype,
        assignments,
        input_size: intermediate,
        output_size: hidden_size,
        local_experts,
        source_row_divisor: 1,
        block_m,
        block_n,
        block_k,
        gated_activation: None,
        multiply_routing_weight: true,
    };
    let down_spec = KernelSpec::new(
        build_grouped_projection(down_config).map_err(kernel_error)?,
        vec![
            tensor(dtype, &[assignments, intermediate])?,
            tensor(KernelDType::I32, inputs.schedule_shape.dimensions())?,
            tensor(KernelDType::I32, inputs.block_experts_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
            tensor(dtype, inputs.down_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
            tensor(dtype, &[tokens, experts_per_token])?,
        ],
        vec![tensor(dtype, &[assignments, hidden_size])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let down_call = down_spec
        .lower(
            context,
            &[
                ("input", gate_up),
                ("sorted_assignments", inputs.sorted_assignments),
                ("block_experts", inputs.block_experts),
                ("active_blocks", inputs.active_blocks),
                ("weights", inputs.down_weights),
                ("expert_offset", expert_offset),
                ("routing_weights", inputs.routing_weights),
            ],
            launch(block_count, hidden_size, block_n)?,
        )
        .map_err(kernel_error)?;
    let down = down_call.result(0)?;
    block.append_operation(down_call)?;

    let grouped_type = context.ranked_tensor_type(
        inputs.hidden_shape.dtype(),
        &[tokens, experts_per_token, hidden_size],
    )?;
    let grouped = append_value(block, context.reshape(down, grouped_type)?)?;
    let scalar_type = context.ranked_tensor_type(inputs.hidden_shape.dtype(), &[])?;
    let zero = constant(context, block, scalar_type, "0.0")?;
    let mut reduction_block = Block::new(context, &[scalar_type, scalar_type])?;
    let sum = context.add(
        reduction_block.argument(0)?,
        reduction_block.argument(1)?,
        scalar_type,
    )?;
    let sum_value = sum.result(0)?;
    reduction_block.append_operation(sum)?;
    reduction_block.append_operation(context.stablehlo_return(&[sum_value])?)?;
    let mut reduction = Region::new(context)?;
    reduction.append_block(reduction_block)?;
    append_value(
        block,
        context.reduce(grouped, zero, inputs.result_type, &[1], reduction)?,
    )
    .map_err(Into::into)
}

fn launch(blocks: i64, output_size: i64, block_n: i64) -> Result<KernelLaunch, Error> {
    let columns = output_size
        .checked_add(block_n - 1)
        .and_then(|value| value.checked_div(block_n))
        .ok_or(Error::InvalidMoe("expert projection launch grid overflows"))?;
    Ok(KernelLaunch {
        grid: [
            i32::try_from(blocks)
                .map_err(|_| Error::InvalidMoe("expert block count exceeds I32"))?,
            i32::try_from(columns)
                .map_err(|_| Error::InvalidMoe("expert output grid exceeds I32"))?,
            1,
        ],
        warps: 4,
        stages: 2,
    })
}

fn tensor(dtype: KernelDType, dimensions: &[i64]) -> Result<TensorSpec, Error> {
    TensorSpec::new(dtype, dimensions).map_err(kernel_error)
}

fn kernel_dtype(dtype: DType) -> Result<KernelDType, Error> {
    match dtype {
        DType::F16 => Ok(KernelDType::F16),
        DType::Bf16 => Ok(KernelDType::Bf16),
        DType::F32 => Ok(KernelDType::F32),
        _ => Err(Error::InvalidMoe(
            "CUDA grouped MoE requires F16, BF16, or F32",
        )),
    }
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

fn kernel_error(error: nml_kernel_triton::Error) -> Error {
    match error {
        nml_kernel_triton::Error::Mlir(error) => Error::Mlir(error),
        _ => Error::InvalidMoe("CUDA grouped expert kernel construction failed"),
    }
}
