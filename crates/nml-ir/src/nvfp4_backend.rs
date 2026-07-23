//! Private CUDA lowering for semantic NVFP4 operations.
//!
//! Model code sees an ordinary linear operation over a logical parameter.
//! This module is the only place where that operation becomes a compact-weight
//! W4A16 Triton kernel. The source payload and scale tensors are passed through
//! unchanged; decoding is tile-local inside the contraction kernel.

use crate::{Error, device_capabilities::CudaCapabilities};
use nml_kernel_triton::{
    DType as KernelDType, KernelLaunch, KernelSpec, NvFp4EmbeddingConfig,
    NvFp4GroupedProjectionConfig, NvFp4GroupedRole, NvFp4LinearConfig, NvFp4QkvConfig, TensorSpec,
    build_nvfp4_embedding, build_nvfp4_grouped_projection, build_nvfp4_linear, build_nvfp4_qkv,
};
use nml_mlir::{Block, Context, Region, Type, Value};
use nml_parameter::nvfp4::{decode_e2m1, decode_e4m3fn_scale};
use nml_types::{DType, Partition, Shape};

// These are finite kernel-family boundaries, not product batch-size policy.
// Dense projections have a proven rank-one GEMV body, a bounded row-minor
// wrapper around that exact body, and a tensor-core matrix path. Sparse
// selected-route projections remain GEMV-efficient through eight tokens
// because each route ordinarily owns only one useful row.
const DENSE_GEMV_ROWS: i64 = 1;
const FUSED_QKV_MAX_ROWS: i64 = 8;
const EMBEDDING_SCALAR_MAX_ROWS: i64 = 8;
const SPARSE_GEMV_MAX_TOKENS: i64 = 8;
const TENSOR_CORE_MINIMUM_M: i64 = 16;
const DECODE_BLOCK_N: i64 = 8;
const DECODE_BLOCK_K: i64 = 256;
const WIDE_GEMV_BLOCK_N: i64 = 32;
const LATENCY_BLOCK_N: i64 = 64;
const LATENCY_BLOCK_K: i64 = 128;
const THROUGHPUT_BLOCK_N: i64 = 128;
const THROUGHPUT_BLOCK_K: i64 = 64;
const EMBEDDING_WIDE_BLOCK_N: i64 = 64;
const EMBEDDING_NARROW_BLOCK_N: i64 = WIDE_GEMV_BLOCK_N;
const LOW_SM_COUNT: usize = 48;
const MINIMUM_ROW_GEMV_WAVES: i64 = 1;
const MINIMUM_MATRIX_ROW_UTILIZATION_DENOMINATOR: i64 = 4;

pub(crate) struct LinearInputs<'context> {
    pub activation: Value<'context>,
    pub payload: Value<'context>,
    pub block_scales: Value<'context>,
    pub global_scale: Value<'context>,
    pub bias: Option<Value<'context>>,
    pub activation_shape: Shape,
    pub payload_shape: Shape,
    pub block_scales_shape: Shape,
    pub global_scale_shape: Shape,
    pub bias_shape: Option<Shape>,
    pub result_shape: Shape,
    pub result_type: Type<'context>,
}

#[derive(Clone, Copy)]
pub(crate) struct QkvProjectionInputs<'context> {
    pub payload: Value<'context>,
    pub block_scales: Value<'context>,
    pub global_scale: Value<'context>,
    pub bias: Option<Value<'context>>,
    pub payload_shape: Shape,
    pub block_scales_shape: Shape,
    pub global_scale_shape: Shape,
    pub bias_shape: Option<Shape>,
    pub result_shape: Shape,
    pub result_type: Type<'context>,
}

pub(crate) struct QkvInputs<'context> {
    pub activation: Value<'context>,
    pub activation_shape: Shape,
    pub query: QkvProjectionInputs<'context>,
    pub key: QkvProjectionInputs<'context>,
    pub value: QkvProjectionInputs<'context>,
}

pub(crate) struct ExpertInputs<'context> {
    pub hidden: Value<'context>,
    pub routing_weights: Value<'context>,
    pub gate_payload: Value<'context>,
    pub gate_scales: Value<'context>,
    pub gate_global: Value<'context>,
    pub gate_bias: Value<'context>,
    pub down_payload: Value<'context>,
    pub down_scales: Value<'context>,
    pub down_global: Value<'context>,
    pub down_bias: Value<'context>,
    pub sorted_assignments: Value<'context>,
    pub block_experts: Value<'context>,
    pub active_blocks: Value<'context>,
    pub expert_offset: Option<Value<'context>>,
    pub hidden_shape: Shape,
    pub routing_shape: Shape,
    pub gate_payload_shape: Shape,
    pub gate_scales_shape: Shape,
    pub gate_global_shape: Shape,
    pub gate_bias_shape: Shape,
    pub down_payload_shape: Shape,
    pub down_scales_shape: Shape,
    pub down_global_shape: Shape,
    pub down_bias_shape: Shape,
    pub schedule_shape: Shape,
    pub block_experts_shape: Shape,
    pub result_type: Type<'context>,
    pub block_size: usize,
}

pub(crate) struct EmbeddingInputs<'context> {
    pub indices: Value<'context>,
    pub payload: Value<'context>,
    pub block_scales: Value<'context>,
    pub global_scale: Value<'context>,
    pub indices_shape: Shape,
    pub payload_shape: Shape,
    pub block_scales_shape: Shape,
    pub global_scale_shape: Shape,
    pub result_shape: Shape,
    pub result_type: Type<'context>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LinearPlan {
    config: NvFp4LinearConfig,
    warps: i32,
    stages: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GroupedPlan {
    block_n: i64,
    block_k: i64,
    gate_warps: i32,
    down_warps: i32,
    stages: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QkvPlan {
    block_n: i64,
    block_k: i64,
    warps: i32,
    stages: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EmbeddingPlan {
    block_m: i64,
    block_n: i64,
    warps: i32,
    stages: i32,
}

pub(crate) fn lower_linear<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: LinearInputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    require_unsharded(&inputs)?;
    if capabilities.supports_nvfp4_turing_custom_call() {
        return lower_turing_linear(context, block, inputs);
    }
    let plan = LinearPlan::new(
        inputs.activation_shape,
        inputs.result_shape,
        inputs.bias.is_some(),
        capabilities,
    )?;
    let rows = plan.config.rows;
    let outputs = plan.config.outputs;
    let output_shape = [rows, outputs];
    let mut argument_specs = vec![
        tensor(plan.config.dtype, inputs.activation_shape.dimensions())?,
        tensor(KernelDType::U8, inputs.payload_shape.dimensions())?,
        tensor(KernelDType::U8, inputs.block_scales_shape.dimensions())?,
        tensor(KernelDType::F32, inputs.global_scale_shape.dimensions())?,
    ];
    if let Some(shape) = inputs.bias_shape {
        argument_specs.push(tensor(plan.config.dtype, shape.dimensions())?);
    }
    let specification = KernelSpec::new(
        build_nvfp4_linear(plan.config).map_err(kernel_error)?,
        argument_specs,
        vec![tensor(plan.config.dtype, &output_shape)?],
        vec![],
    )
    .map_err(kernel_error)?;
    let mut arguments = vec![
        ("input", inputs.activation),
        ("payload", inputs.payload),
        ("block_scales", inputs.block_scales),
        ("global_scale", inputs.global_scale),
    ];
    if let Some(bias) = inputs.bias {
        arguments.push(("bias", bias));
    }
    let call = specification
        .lower(
            context,
            &arguments,
            KernelLaunch {
                grid: plan.config.launch_grid().map_err(kernel_error)?,
                warps: plan.warps,
                stages: plan.stages,
            },
        )
        .map_err(kernel_error)?;
    let result = call.result(0)?;
    block.append_operation(call)?;

    if result.type_().text() == inputs.result_type.text() {
        return Ok(result);
    }
    let reshape = context.reshape(result, inputs.result_type)?;
    let result = reshape.result(0)?;
    block.append_operation(reshape)?;
    Ok(result)
}

pub(crate) fn lower_qkv<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: QkvInputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<[Value<'context>; 3], Error> {
    let projections = [inputs.query, inputs.key, inputs.value];
    if capabilities.supports_nvfp4_turing_custom_call() {
        let mut results = Vec::with_capacity(3);
        for projection in projections {
            results.push(lower_linear(
                context,
                block,
                linear_inputs(inputs.activation, inputs.activation_shape, projection),
                capabilities,
            )?);
        }
        return results
            .try_into()
            .map_err(|_| Error::InvalidLinearAlgebra("NVFP4 QKV result arity changed"));
    }
    let plans = projections
        .iter()
        .map(|projection| {
            LinearPlan::new(
                inputs.activation_shape,
                projection.result_shape,
                projection.bias.is_some(),
                capabilities,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = plans[0];
    let fused_small_batch = first.config.rows <= FUSED_QKV_MAX_ROWS
        && plans.iter().all(|plan| {
            plan.config.rows == first.config.rows
                && plan.config.dtype == first.config.dtype
                && plan.config.inputs == first.config.inputs
                && plan.config.has_bias == first.config.has_bias
        });
    if !fused_small_batch {
        let mut results = Vec::with_capacity(3);
        for projection in projections {
            results.push(lower_linear(
                context,
                block,
                linear_inputs(inputs.activation, inputs.activation_shape, projection),
                capabilities,
            )?);
        }
        return results
            .try_into()
            .map_err(|_| Error::InvalidLinearAlgebra("NVFP4 QKV result arity changed"));
    }

    for projection in projections {
        require_unsharded(&linear_inputs(
            inputs.activation,
            inputs.activation_shape,
            projection,
        ))?;
    }
    let qkv_plan = QkvPlan::new(
        first.config.rows,
        first.config.inputs,
        [
            plans[0].config.outputs,
            plans[1].config.outputs,
            plans[2].config.outputs,
        ],
        capabilities,
    );
    let config = NvFp4QkvConfig {
        dtype: first.config.dtype,
        rows: first.config.rows,
        inputs: first.config.inputs,
        query_outputs: plans[0].config.outputs,
        key_outputs: plans[1].config.outputs,
        value_outputs: plans[2].config.outputs,
        block_n: qkv_plan.block_n,
        block_k: qkv_plan.block_k,
        has_bias: first.config.has_bias,
    };
    let mut argument_specs = vec![tensor(config.dtype, inputs.activation_shape.dimensions())?];
    for projection in projections {
        argument_specs.extend([
            tensor(KernelDType::U8, projection.payload_shape.dimensions())?,
            tensor(KernelDType::U8, projection.block_scales_shape.dimensions())?,
            tensor(KernelDType::F32, projection.global_scale_shape.dimensions())?,
        ]);
        if let Some(shape) = projection.bias_shape {
            argument_specs.push(tensor(config.dtype, shape.dimensions())?);
        }
    }
    let output_shapes = [
        [config.rows, config.query_outputs],
        [config.rows, config.key_outputs],
        [config.rows, config.value_outputs],
    ];
    let specification = KernelSpec::new(
        build_nvfp4_qkv(config).map_err(kernel_error)?,
        argument_specs,
        output_shapes
            .iter()
            .map(|shape| tensor(config.dtype, shape))
            .collect::<Result<Vec<_>, _>>()?,
        vec![],
    )
    .map_err(kernel_error)?;
    let names = ["query", "key", "value"];
    let mut arguments = vec![("input".to_owned(), inputs.activation)];
    for (name, projection) in names.into_iter().zip(projections) {
        arguments.extend([
            (projection_name(name, "payload"), projection.payload),
            (
                projection_name(name, "block_scales"),
                projection.block_scales,
            ),
            (
                projection_name(name, "global_scale"),
                projection.global_scale,
            ),
        ]);
        if let Some(bias) = projection.bias {
            arguments.push((projection_name(name, "bias"), bias));
        }
    }
    let borrowed_arguments = arguments
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    let call = specification
        .lower(
            context,
            &borrowed_arguments,
            KernelLaunch {
                grid: config.launch_grid().map_err(kernel_error)?,
                warps: qkv_plan.warps,
                stages: qkv_plan.stages,
            },
        )
        .map_err(kernel_error)?;
    let flat = [call.result(0)?, call.result(1)?, call.result(2)?];
    block.append_operation(call)?;

    let mut results = Vec::with_capacity(3);
    for (flat, projection) in flat.into_iter().zip(projections) {
        if flat.type_().text() == projection.result_type.text() {
            results.push(flat);
        } else {
            let reshape = context.reshape(flat, projection.result_type)?;
            let result = reshape.result(0)?;
            block.append_operation(reshape)?;
            results.push(result);
        }
    }
    results
        .try_into()
        .map_err(|_| Error::InvalidLinearAlgebra("NVFP4 QKV result arity changed"))
}

fn linear_inputs<'context>(
    activation: Value<'context>,
    activation_shape: Shape,
    projection: QkvProjectionInputs<'context>,
) -> LinearInputs<'context> {
    LinearInputs {
        activation,
        payload: projection.payload,
        block_scales: projection.block_scales,
        global_scale: projection.global_scale,
        bias: projection.bias,
        activation_shape,
        payload_shape: projection.payload_shape,
        block_scales_shape: projection.block_scales_shape,
        global_scale_shape: projection.global_scale_shape,
        bias_shape: projection.bias_shape,
        result_shape: projection.result_shape,
        result_type: projection.result_type,
    }
}

fn projection_name(projection: &str, component: &str) -> String {
    format!("{projection}_{component}")
}

pub(crate) fn lower_routed_swiglu<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: ExpertInputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    let use_turing_adapter = capabilities.supports_nvfp4_turing_custom_call();
    if use_turing_adapter {
        if inputs.expert_offset.is_some() {
            return Err(Error::UnsupportedTarget {
                operation: "NVFP4 routed clamped SwiGLU",
                target: "sharded CUDA SM75 execution".to_owned(),
                requirement: "the Turing grouped adapter currently owns one complete local expert set",
            });
        }
        require_unsharded_experts(&inputs)?;
    } else {
        require_triton_emulation(capabilities, "NVFP4 routed clamped SwiGLU")?;
        if inputs.expert_offset.is_none() {
            require_unsharded_experts(&inputs)?;
        }
    }
    let tokens = inputs.hidden_shape.dimensions()[0];
    let hidden_size = inputs.hidden_shape.dimensions()[1];
    let experts_per_token = inputs.routing_shape.dimensions()[1];
    let assignments = tokens
        .checked_mul(experts_per_token)
        .ok_or(Error::InvalidMoe("NVFP4 assignment count overflows"))?;
    let local_experts = inputs.gate_payload_shape.dimensions()[0];
    let gate_output = inputs.gate_bias_shape.dimensions()[1];
    let down_output = inputs.down_bias_shape.dimensions()[1];
    if gate_output % 2 != 0 {
        return Err(Error::InvalidMoe(
            "NVFP4 gate/up output width must contain equal gate and up halves",
        ));
    }
    let intermediate = gate_output / 2;
    let packed_hidden = hidden_size
        .checked_add(1)
        .ok_or(Error::InvalidMoe("NVFP4 hidden width overflows"))?
        / 2;
    let packed_intermediate = intermediate
        .checked_add(1)
        .ok_or(Error::InvalidMoe("NVFP4 intermediate width overflows"))?
        / 2;
    if inputs.gate_payload_shape.dimensions()[2] != packed_hidden
        || inputs.gate_payload_shape.dimensions()[1] != gate_output
        || inputs.down_payload_shape.dimensions()[1] != down_output
        || inputs.down_payload_shape.dimensions()[2] != packed_intermediate
        || down_output != hidden_size
    {
        return Err(Error::InvalidMoe(
            "NVFP4 grouped expert logical dimensions are inconsistent",
        ));
    }

    let down = if use_turing_adapter {
        lower_turing_experts(
            context,
            block,
            &inputs,
            assignments,
            intermediate,
            hidden_size,
        )?
    } else {
        lower_triton_experts(
            context,
            block,
            &inputs,
            assignments,
            experts_per_token,
            local_experts,
            intermediate,
            hidden_size,
            capabilities,
        )?
    };

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

fn lower_turing_experts<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &ExpertInputs<'context>,
    assignments: i64,
    intermediate: i64,
    hidden_size: i64,
) -> Result<Value<'context>, Error> {
    let activated_type =
        context.ranked_tensor_type(inputs.hidden_shape.dtype(), &[assignments, intermediate])?;
    let gate_call = context.ffi_custom_call(
        "nml.nvfp4.turing.expert_gate_up",
        &[
            inputs.hidden,
            inputs.sorted_assignments,
            inputs.block_experts,
            inputs.gate_payload,
            inputs.gate_scales,
            inputs.gate_global,
            inputs.gate_bias,
        ],
        &[activated_type],
    )?;
    let activated = gate_call.result(0)?;
    block.append_operation(gate_call)?;

    let weighted_type =
        context.ranked_tensor_type(inputs.hidden_shape.dtype(), &[assignments, hidden_size])?;
    let down_call = context.ffi_custom_call(
        "nml.nvfp4.turing.expert_down",
        &[
            activated,
            inputs.sorted_assignments,
            inputs.block_experts,
            inputs.down_payload,
            inputs.down_scales,
            inputs.down_global,
            inputs.down_bias,
            inputs.routing_weights,
        ],
        &[weighted_type],
    )?;
    let weighted = down_call.result(0)?;
    block.append_operation(down_call)?;
    Ok(weighted)
}

#[allow(clippy::too_many_arguments)]
fn lower_triton_experts<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &ExpertInputs<'context>,
    assignments: i64,
    experts_per_token: i64,
    local_experts: i64,
    intermediate: i64,
    hidden_size: i64,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    let dtype = kernel_dtype(inputs.hidden_shape.dtype(), "NVFP4 routed clamped SwiGLU")?;
    let block_m = i64::try_from(inputs.block_size)
        .map_err(|_| Error::InvalidMoe("NVFP4 expert block size exceeds I64"))?;
    let plan = GroupedPlan::new(
        inputs.hidden_shape.dimensions()[0],
        inputs.block_experts_shape.dimensions()[0],
        intermediate.max(hidden_size),
        capabilities,
    );
    let block_n = plan.block_n;
    let block_k = plan.block_k;
    let decode_codebooks = if inputs.hidden_shape.dimensions()[0] <= 8 {
        Some(decode_codebook_constants(context, block)?)
    } else {
        None
    };
    let expert_offset = match inputs.expert_offset {
        Some(value) => value,
        None => {
            let type_ = context.ranked_tensor_type(DType::I32, &[])?;
            constant(context, block, type_, "0")?
        }
    };

    let gate_config = NvFp4GroupedProjectionConfig {
        dtype,
        tokens: inputs.hidden_shape.dimensions()[0],
        assignments,
        input_size: hidden_size,
        output_size: intermediate,
        local_experts,
        source_row_divisor: experts_per_token,
        block_m,
        block_n,
        block_k,
        role: NvFp4GroupedRole::GateUpActivated,
    };
    let mut gate_specs = vec![
        tensor(dtype, inputs.hidden_shape.dimensions())?,
        tensor(KernelDType::I32, inputs.schedule_shape.dimensions())?,
        tensor(KernelDType::I32, inputs.block_experts_shape.dimensions())?,
        tensor(KernelDType::I32, &[])?,
        tensor(KernelDType::U8, inputs.gate_payload_shape.dimensions())?,
        tensor(KernelDType::U8, inputs.gate_scales_shape.dimensions())?,
    ];
    if decode_codebooks.is_some() {
        gate_specs.extend([
            tensor(KernelDType::I32, &[16])?,
            tensor(KernelDType::I32, &[256])?,
        ]);
    }
    gate_specs.extend([
        tensor(KernelDType::F32, inputs.gate_global_shape.dimensions())?,
        tensor(dtype, inputs.gate_bias_shape.dimensions())?,
        tensor(KernelDType::I32, &[])?,
    ]);
    let gate_specification = KernelSpec::new(
        build_nvfp4_grouped_projection(gate_config).map_err(kernel_error)?,
        gate_specs,
        vec![tensor(dtype, &[assignments, intermediate])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let mut gate_operands = vec![
        ("input", inputs.hidden),
        ("sorted_assignments", inputs.sorted_assignments),
        ("block_experts", inputs.block_experts),
        ("active_blocks", inputs.active_blocks),
        ("payload", inputs.gate_payload),
        ("block_scales", inputs.gate_scales),
    ];
    if let Some((e2m1_codebook, e4m3fn_codebook)) = decode_codebooks {
        gate_operands.extend([
            ("e2m1_codebook", e2m1_codebook),
            ("e4m3fn_codebook", e4m3fn_codebook),
        ]);
    }
    gate_operands.extend([
        ("global_scale", inputs.gate_global),
        ("bias", inputs.gate_bias),
        ("expert_offset", expert_offset),
    ]);
    let gate_call = gate_specification
        .lower(
            context,
            &gate_operands,
            grouped_launch(
                inputs.block_experts_shape.dimensions()[0],
                intermediate,
                block_n,
                plan.gate_warps,
                plan.stages,
            )?,
        )
        .map_err(kernel_error)?;
    let gate_output_value = gate_call.result(0)?;
    block.append_operation(gate_call)?;

    let down_config = NvFp4GroupedProjectionConfig {
        dtype,
        tokens: inputs.hidden_shape.dimensions()[0],
        assignments,
        input_size: intermediate,
        output_size: hidden_size,
        local_experts,
        source_row_divisor: 1,
        block_m,
        block_n,
        block_k,
        role: NvFp4GroupedRole::Down,
    };
    let mut down_specs = vec![
        tensor(dtype, &[assignments, intermediate])?,
        tensor(KernelDType::I32, inputs.schedule_shape.dimensions())?,
        tensor(KernelDType::I32, inputs.block_experts_shape.dimensions())?,
        tensor(KernelDType::I32, &[])?,
        tensor(KernelDType::U8, inputs.down_payload_shape.dimensions())?,
        tensor(KernelDType::U8, inputs.down_scales_shape.dimensions())?,
    ];
    if decode_codebooks.is_some() {
        down_specs.extend([
            tensor(KernelDType::I32, &[16])?,
            tensor(KernelDType::I32, &[256])?,
        ]);
    }
    down_specs.extend([
        tensor(KernelDType::F32, inputs.down_global_shape.dimensions())?,
        tensor(dtype, inputs.down_bias_shape.dimensions())?,
        tensor(KernelDType::I32, &[])?,
        tensor(dtype, inputs.routing_shape.dimensions())?,
    ]);
    let down_specification = KernelSpec::new(
        build_nvfp4_grouped_projection(down_config).map_err(kernel_error)?,
        down_specs,
        vec![tensor(dtype, &[assignments, hidden_size])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let mut down_operands = vec![
        ("input", gate_output_value),
        ("sorted_assignments", inputs.sorted_assignments),
        ("block_experts", inputs.block_experts),
        ("active_blocks", inputs.active_blocks),
        ("payload", inputs.down_payload),
        ("block_scales", inputs.down_scales),
    ];
    if let Some((e2m1_codebook, e4m3fn_codebook)) = decode_codebooks {
        down_operands.extend([
            ("e2m1_codebook", e2m1_codebook),
            ("e4m3fn_codebook", e4m3fn_codebook),
        ]);
    }
    down_operands.extend([
        ("global_scale", inputs.down_global),
        ("bias", inputs.down_bias),
        ("expert_offset", expert_offset),
        ("routing_weights", inputs.routing_weights),
    ]);
    let down_call = down_specification
        .lower(
            context,
            &down_operands,
            grouped_launch(
                inputs.block_experts_shape.dimensions()[0],
                hidden_size,
                block_n,
                plan.down_warps,
                plan.stages,
            )?,
        )
        .map_err(kernel_error)?;
    let down = down_call.result(0)?;
    block.append_operation(down_call)?;

    Ok(down)
}

pub(crate) fn lower_embedding<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: EmbeddingInputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    require_unsharded_embedding(&inputs)?;
    if capabilities.supports_nvfp4_turing_custom_call() {
        return lower_turing_embedding(context, block, inputs);
    }
    require_triton_emulation(capabilities, "NVFP4 embedding")?;
    let rows = inputs
        .indices_shape
        .dimensions()
        .iter()
        .try_fold(1_i64, |product, dimension| product.checked_mul(*dimension))
        .ok_or(Error::InvalidIndexing(
            "NVFP4 embedding index extent overflows I64",
        ))?;
    let vocabulary = inputs.payload_shape.dimensions()[0];
    let width = *inputs
        .result_shape
        .dimensions()
        .last()
        .ok_or(Error::InvalidIndexing(
            "NVFP4 embedding result must have a feature axis",
        ))?;
    let plan = EmbeddingPlan::new(rows, width, capabilities);
    let config = NvFp4EmbeddingConfig {
        dtype: kernel_dtype(inputs.result_shape.dtype(), "NVFP4 embedding")?,
        index_dtype: kernel_index_dtype(inputs.indices_shape.dtype())?,
        rows,
        vocabulary,
        width,
        block_m: plan.block_m,
        block_n: plan.block_n,
    };
    let specification = KernelSpec::new(
        build_nvfp4_embedding(config).map_err(kernel_error)?,
        vec![
            tensor(config.index_dtype, inputs.indices_shape.dimensions())?,
            tensor(KernelDType::U8, inputs.payload_shape.dimensions())?,
            tensor(KernelDType::U8, inputs.block_scales_shape.dimensions())?,
            tensor(KernelDType::F32, inputs.global_scale_shape.dimensions())?,
        ],
        vec![tensor(config.dtype, &[rows, width])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let call = specification
        .lower(
            context,
            &[
                ("indices", inputs.indices),
                ("payload", inputs.payload),
                ("block_scales", inputs.block_scales),
                ("global_scale", inputs.global_scale),
            ],
            KernelLaunch {
                grid: config.launch_grid().map_err(kernel_error)?,
                warps: plan.warps,
                stages: plan.stages,
            },
        )
        .map_err(kernel_error)?;
    let result = call.result(0)?;
    block.append_operation(call)?;
    if result.type_().text() == inputs.result_type.text() {
        return Ok(result);
    }
    let reshape = context.reshape(result, inputs.result_type)?;
    let result = reshape.result(0)?;
    block.append_operation(reshape)?;
    Ok(result)
}

fn lower_turing_linear<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: LinearInputs<'context>,
) -> Result<Value<'context>, Error> {
    let mut operands = vec![
        inputs.activation,
        inputs.payload,
        inputs.block_scales,
        inputs.global_scale,
    ];
    if let Some(bias) = inputs.bias {
        operands.push(bias);
    }
    let call =
        context.ffi_custom_call("nml.nvfp4.turing.linear", &operands, &[inputs.result_type])?;
    let result = call.result(0)?;
    block.append_operation(call)?;
    Ok(result)
}

fn lower_turing_embedding<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: EmbeddingInputs<'context>,
) -> Result<Value<'context>, Error> {
    let call = context.ffi_custom_call(
        "nml.nvfp4.turing.embedding",
        &[
            inputs.indices,
            inputs.payload,
            inputs.block_scales,
            inputs.global_scale,
        ],
        &[inputs.result_type],
    )?;
    let result = call.result(0)?;
    block.append_operation(call)?;
    Ok(result)
}

impl LinearPlan {
    fn new(
        activation: Shape,
        result: Shape,
        has_bias: bool,
        capabilities: CudaCapabilities,
    ) -> Result<Self, Error> {
        require_triton_emulation(capabilities, "NVFP4 linear")?;
        let dtype = kernel_dtype(activation.dtype(), "NVFP4 linear")?;
        let inputs = *activation
            .dimensions()
            .last()
            .ok_or(Error::InvalidLinearAlgebra(
                "NVFP4 linear activation must have rank",
            ))?;
        let outputs = *result
            .dimensions()
            .last()
            .ok_or(Error::InvalidLinearAlgebra(
                "NVFP4 linear result must have rank",
            ))?;
        let rows = activation.dimensions()[..activation.rank() - 1]
            .iter()
            .try_fold(1_i64, |product, dimension| product.checked_mul(*dimension))
            .ok_or(Error::InvalidLinearAlgebra(
                "NVFP4 linear batch geometry overflows I64",
            ))?;
        if rows <= 0 || inputs <= 0 || outputs <= 0 {
            return Err(Error::InvalidLinearAlgebra(
                "NVFP4 CUDA linear requires positive M, N, and K",
            ));
        }

        // The contraction authored by both compact families remains exactly
        // rank one. M=1 uses the accepted scalar grid. A bounded M>1 wrapper
        // may place rows in the minor grid dimension when an M16 matrix tile
        // would be at most one-quarter utilized and its N grid cannot fill
        // the latency target. Large-N projections such as the vocabulary head
        // retain the tensor-core family because their matrix grid is already
        // sufficient and useful rows can reuse each decoded weight tile.
        let scalar_gemv = rows == DENSE_GEMV_ROWS;
        let latency_grid = i64::try_from(capabilities.latency_grid_target()).unwrap_or(i64::MAX);
        let wide_gemv_grid = ceil_div_positive(outputs, WIDE_GEMV_BLOCK_N);
        let lost_parallelism = WIDE_GEMV_BLOCK_N / DECODE_BLOCK_N;
        let gemv_block_n =
            if wide_gemv_grid >= latency_grid.saturating_mul(lost_parallelism.saturating_mul(2)) {
                WIDE_GEMV_BLOCK_N
            } else {
                DECODE_BLOCK_N
            };
        let tensor_row_capacity =
            ceil_div_positive(rows, TENSOR_CORE_MINIMUM_M).saturating_mul(TENSOR_CORE_MINIMUM_M);
        let matrix_grid = ceil_div_positive(rows, TENSOR_CORE_MINIMUM_M)
            .saturating_mul(ceil_div_positive(outputs, LATENCY_BLOCK_N));
        let row_gemv_grid = rows.saturating_mul(ceil_div_positive(outputs, gemv_block_n));
        let row_gemv = !scalar_gemv
            && tensor_row_capacity
                >= rows.saturating_mul(MINIMUM_MATRIX_ROW_UTILIZATION_DENOMINATOR)
            && matrix_grid < latency_grid
            && row_gemv_grid
                >= i64::try_from(capabilities.multiprocessor_count())
                    .unwrap_or(i64::MAX)
                    .saturating_mul(MINIMUM_ROW_GEMV_WAVES);
        let compact_gemv = scalar_gemv || row_gemv;
        let block_m = if compact_gemv {
            1
        } else if rows <= 32 {
            TENSOR_CORE_MINIMUM_M
        } else if capabilities.supports_warp_group_mma() {
            64
        } else {
            32
        };
        let latency_sensitive = rows <= 32;
        Ok(Self {
            config: NvFp4LinearConfig {
                dtype,
                rows,
                outputs,
                inputs,
                block_m,
                block_n: if compact_gemv {
                    gemv_block_n
                } else if latency_sensitive {
                    LATENCY_BLOCK_N
                } else {
                    THROUGHPUT_BLOCK_N
                },
                block_k: if compact_gemv {
                    DECODE_BLOCK_K
                } else if latency_sensitive {
                    LATENCY_BLOCK_K
                } else {
                    THROUGHPUT_BLOCK_K
                },
                has_bias,
            },
            warps: if rows > 128 && capabilities.supports_warp_group_mma() {
                8
            } else {
                4
            },
            stages: if compact_gemv {
                1
            } else if latency_sensitive {
                4
            } else {
                3
            },
        })
    }
}

impl QkvPlan {
    fn new(rows: i64, inputs: i64, outputs: [i64; 3], capabilities: CudaCapabilities) -> Self {
        let latency_grid = i64::try_from(capabilities.latency_grid_target()).unwrap_or(i64::MAX);
        let programs_at_16 = outputs
            .into_iter()
            .map(|output| ceil_div_positive(output, 16))
            .sum::<i64>()
            .saturating_mul(rows);
        let block_n = if capabilities.supports_warp_group_mma()
            && programs_at_16 >= latency_grid.saturating_mul(2)
        {
            16
        } else {
            // Ampere's accepted decode family uses narrow output tiles to
            // expose enough independent memory transactions.
            8
        };
        Self {
            block_n,
            block_k: if inputs >= DECODE_BLOCK_K {
                DECODE_BLOCK_K
            } else {
                LATENCY_BLOCK_K
            },
            warps: 4,
            stages: 1,
        }
    }
}

impl EmbeddingPlan {
    /// Vocabulary rows selected by different tokens do not share weight data,
    /// so small-M embedding should expose each real row independently. For
    /// larger prompt chunks, choose the largest row tile that still supplies
    /// at least two waves of CTAs; column width and SM count decide whether a
    /// 64- or 32-column tile is sufficiently parallel.
    fn new(rows: i64, width: i64, capabilities: CudaCapabilities) -> Self {
        let multiprocessors =
            i64::try_from(capabilities.multiprocessor_count()).unwrap_or(i64::MAX);
        let two_waves = multiprocessors.saturating_mul(2);
        let grid = |block_m, block_n| {
            ceil_div_positive(rows, block_m).saturating_mul(ceil_div_positive(width, block_n))
        };
        let block_n = if grid(1, EMBEDDING_WIDE_BLOCK_N) >= multiprocessors {
            EMBEDDING_WIDE_BLOCK_N
        } else {
            EMBEDDING_NARROW_BLOCK_N
        };
        let block_m = if rows <= EMBEDDING_SCALAR_MAX_ROWS {
            1
        } else if grid(TENSOR_CORE_MINIMUM_M, block_n) >= two_waves {
            TENSOR_CORE_MINIMUM_M
        } else if grid(8, block_n) >= two_waves {
            8
        } else {
            4
        };
        Self {
            block_m,
            block_n,
            warps: 4,
            stages: if rows <= EMBEDDING_SCALAR_MAX_ROWS {
                1
            } else {
                2
            },
        }
    }
}

impl GroupedPlan {
    /// Selects from a finite, reviewable tile family. Through eight tokens,
    /// sparse routing rarely fills a 16-row expert tile, so selected-route
    /// GEMV avoids tensor-core padding and uses the proven decode geometry.
    /// Larger M exposes enough activation and expert-weight reuse for the
    /// grouped matrix family; sufficiently large batches employ eight warps
    /// only where the architecture exposes warp-group MMA.
    fn new(
        tokens: i64,
        scheduled_blocks: i64,
        maximum_output: i64,
        capabilities: CudaCapabilities,
    ) -> Self {
        if tokens <= SPARSE_GEMV_MAX_TOKENS {
            Self {
                block_n: DECODE_BLOCK_N,
                block_k: DECODE_BLOCK_K,
                gate_warps: 4,
                down_warps: 4,
                stages: 1,
            }
        } else {
            let grid_128 = scheduled_blocks
                .saturating_mul(ceil_div_positive(maximum_output, THROUGHPUT_BLOCK_N));
            let target = i64::try_from(capabilities.latency_grid_target()).unwrap_or(i64::MAX);
            let multiprocessors =
                i64::try_from(capabilities.multiprocessor_count()).unwrap_or(i64::MAX);
            // N=64/K=128 reduces the live decoded-weight tile and doubles the
            // CTA population. Retain it whenever N=128 would undersubscribe the
            // latency target or there is no more than one routed block per SM,
            // where register pressure dominates cross-row reuse.
            let narrow = scheduled_blocks <= multiprocessors || grid_128 < target;
            let block_n = if narrow {
                LATENCY_BLOCK_N
            } else {
                THROUGHPUT_BLOCK_N
            };
            let block_k = if narrow {
                LATENCY_BLOCK_K
            } else {
                THROUGHPUT_BLOCK_K
            };
            let large_throughput_family =
                scheduled_blocks > multiprocessors && capabilities.supports_warp_group_mma();
            Self {
                block_n,
                block_k,
                gate_warps: if large_throughput_family { 8 } else { 4 },
                down_warps: if large_throughput_family { 8 } else { 4 },
                stages: if scheduled_blocks.saturating_mul(2) < multiprocessors {
                    4
                } else if capabilities.multiprocessor_count() < LOW_SM_COUNT {
                    2
                } else {
                    3
                },
            }
        }
    }
}

fn ceil_div_positive(value: i64, divisor: i64) -> i64 {
    debug_assert!(value > 0 && divisor > 0);
    value / divisor + i64::from(value % divisor != 0)
}

fn require_unsharded(inputs: &LinearInputs<'_>) -> Result<(), Error> {
    if [
        inputs.activation_shape,
        inputs.payload_shape,
        inputs.block_scales_shape,
        inputs.global_scale_shape,
        inputs.result_shape,
    ]
    .iter()
    .flat_map(|shape| shape.partitions())
    .any(|partition| matches!(partition, Partition::Sharded(_)))
    {
        return Err(Error::UnsupportedTarget {
            operation: "NVFP4 linear",
            target: "sharded CUDA execution".to_owned(),
            requirement: "representation-aware local compact-weight geometry is not implemented",
        });
    }
    if inputs.bias_shape.is_some_and(|shape| {
        shape
            .partitions()
            .iter()
            .any(|partition| matches!(partition, Partition::Sharded(_)))
    }) {
        return Err(Error::UnsupportedTarget {
            operation: "NVFP4 linear",
            target: "sharded CUDA execution".to_owned(),
            requirement: "representation-aware local compact-weight geometry is not implemented",
        });
    }
    Ok(())
}

fn require_unsharded_experts(inputs: &ExpertInputs<'_>) -> Result<(), Error> {
    if [
        inputs.hidden_shape,
        inputs.routing_shape,
        inputs.gate_payload_shape,
        inputs.gate_scales_shape,
        inputs.gate_global_shape,
        inputs.gate_bias_shape,
        inputs.down_payload_shape,
        inputs.down_scales_shape,
        inputs.down_global_shape,
        inputs.down_bias_shape,
        inputs.schedule_shape,
        inputs.block_experts_shape,
    ]
    .iter()
    .flat_map(|shape| shape.partitions())
    .any(|partition| matches!(partition, Partition::Sharded(_)))
    {
        return Err(Error::UnsupportedTarget {
            operation: "NVFP4 routed clamped SwiGLU",
            target: "sharded CUDA execution".to_owned(),
            requirement: "representation-aware local expert component geometry is not implemented",
        });
    }
    Ok(())
}

fn require_unsharded_embedding(inputs: &EmbeddingInputs<'_>) -> Result<(), Error> {
    if [
        inputs.indices_shape,
        inputs.payload_shape,
        inputs.block_scales_shape,
        inputs.global_scale_shape,
        inputs.result_shape,
    ]
    .iter()
    .flat_map(|shape| shape.partitions())
    .any(|partition| matches!(partition, Partition::Sharded(_)))
    {
        return Err(Error::UnsupportedTarget {
            operation: "NVFP4 embedding",
            target: "sharded CUDA execution".to_owned(),
            requirement: "representation-aware vocabulary component geometry is not implemented",
        });
    }
    Ok(())
}

fn require_triton_emulation(
    capabilities: CudaCapabilities,
    operation: &'static str,
) -> Result<(), Error> {
    if capabilities.supports_nvfp4_triton_emulation() {
        return Ok(());
    }
    let (major, minor) = capabilities.compute_capability();
    Err(Error::UnsupportedTarget {
        operation,
        target: format!("CUDA SM{major}{minor}"),
        requirement: "the compact-weight XLA Triton emulation path requires SM80 or newer",
    })
}

fn grouped_launch(
    blocks: i64,
    output_size: i64,
    block_n: i64,
    warps: i32,
    stages: i32,
) -> Result<KernelLaunch, Error> {
    let columns = output_size
        .checked_add(block_n - 1)
        .and_then(|value| value.checked_div(block_n))
        .ok_or(Error::InvalidMoe("NVFP4 expert launch grid overflows"))?;
    Ok(KernelLaunch {
        grid: [
            i32::try_from(blocks)
                .map_err(|_| Error::InvalidMoe("NVFP4 expert block count exceeds I32"))?,
            i32::try_from(columns)
                .map_err(|_| Error::InvalidMoe("NVFP4 expert output grid exceeds I32"))?,
            1,
        ],
        warps,
        stages,
    })
}

fn kernel_dtype(dtype: DType, operation: &'static str) -> Result<KernelDType, Error> {
    match dtype {
        DType::F16 => Ok(KernelDType::F16),
        DType::Bf16 => Ok(KernelDType::Bf16),
        _ => Err(Error::UnsupportedDType { operation, dtype }),
    }
}

fn kernel_index_dtype(dtype: DType) -> Result<KernelDType, Error> {
    match dtype {
        DType::I32 => Ok(KernelDType::I32),
        DType::I64 => Ok(KernelDType::I64),
        _ => Err(Error::InvalidIndexDType(dtype)),
    }
}

fn tensor(dtype: KernelDType, dimensions: &[i64]) -> Result<TensorSpec, Error> {
    TensorSpec::new(dtype, dimensions).map_err(kernel_error)
}

fn decode_codebook_constants<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
) -> Result<(Value<'context>, Value<'context>), Error> {
    let mut e2m1_bits = Vec::with_capacity(16);
    for code in 0_u8..16 {
        let value = decode_e2m1(code)
            .map_err(|_| Error::InvalidLinearAlgebra("invalid E2M1 decode codebook"))?;
        e2m1_bits.push((value.to_bits() as i32).to_string());
    }
    let e2m1_literal = format!("[{}]", e2m1_bits.join(", "));
    let e2m1_type = context.ranked_tensor_type(DType::I32, &[16])?;
    let e2m1 = constant(context, block, e2m1_type, &e2m1_literal)?;

    let mut e4m3fn_bits = Vec::with_capacity(256);
    for bits in 0_u8..=u8::MAX {
        let value = decode_e4m3fn_scale(bits).unwrap_or(f32::NAN);
        e4m3fn_bits.push((value.to_bits() as i32).to_string());
    }
    let e4m3fn_literal = format!("[{}]", e4m3fn_bits.join(", "));
    let e4m3fn_type = context.ranked_tensor_type(DType::I32, &[256])?;
    let e4m3fn = constant(context, block, e4m3fn_type, &e4m3fn_literal)?;
    Ok((e2m1, e4m3fn))
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
        _ => Error::InvalidLinearAlgebra("NVFP4 CUDA linear kernel construction failed"),
    }
}
