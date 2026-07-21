//! Private CUDA lowering for semantic NVFP4 operations.
//!
//! Model code sees an ordinary linear operation over a logical parameter.
//! This module is the only place where that operation becomes a compact-weight
//! W4A16 Triton kernel. The source payload and scale tensors are passed through
//! unchanged; decoding is tile-local inside the contraction kernel.

use crate::{device_capabilities::CudaCapabilities, Error};
use nml_kernel_triton::{
    build_nvfp4_embedding, build_nvfp4_grouped_projection, build_nvfp4_linear,
    build_nvfp4_qkv, DType as KernelDType, KernelLaunch, KernelSpec, NvFp4EmbeddingConfig,
    NvFp4GroupedProjectionConfig, NvFp4GroupedRole, NvFp4LinearConfig, NvFp4QkvConfig,
    TensorSpec,
};
use nml_mlir::{Block, Context, Region, Type, Value};
use nml_types::{DType, Partition, Shape};

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
    let fused_decode = first.config.rows == 1
        && plans.iter().all(|plan| {
            plan.config.rows == 1
                && plan.config.dtype == first.config.dtype
                && plan.config.inputs == first.config.inputs
                && plan.config.block_n == first.config.block_n
                && plan.config.block_k == first.config.block_k
                && plan.config.has_bias == first.config.has_bias
                && plan.warps == first.warps
                && plan.stages == first.stages
        });
    if !fused_decode {
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
    let config = NvFp4QkvConfig {
        dtype: first.config.dtype,
        inputs: first.config.inputs,
        query_outputs: plans[0].config.outputs,
        key_outputs: plans[1].config.outputs,
        value_outputs: plans[2].config.outputs,
        block_n: first.config.block_n,
        block_k: first.config.block_k,
        has_bias: first.config.has_bias,
    };
    let mut argument_specs = vec![tensor(
        config.dtype,
        inputs.activation_shape.dimensions(),
    )?];
    for projection in projections {
        argument_specs.extend([
            tensor(KernelDType::U8, projection.payload_shape.dimensions())?,
            tensor(
                KernelDType::U8,
                projection.block_scales_shape.dimensions(),
            )?,
            tensor(
                KernelDType::F32,
                projection.global_scale_shape.dimensions(),
            )?,
        ]);
        if let Some(shape) = projection.bias_shape {
            argument_specs.push(tensor(config.dtype, shape.dimensions())?);
        }
    }
    let output_shapes = [
        [1, config.query_outputs],
        [1, config.key_outputs],
        [1, config.value_outputs],
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
                warps: first.warps,
                stages: first.stages,
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
                requirement:
                    "the Turing grouped adapter currently owns one complete local expert set",
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
) -> Result<Value<'context>, Error> {
    let dtype = kernel_dtype(inputs.hidden_shape.dtype(), "NVFP4 routed clamped SwiGLU")?;
    let block_m = i64::try_from(inputs.block_size)
        .map_err(|_| Error::InvalidMoe("NVFP4 expert block size exceeds I64"))?;
    let plan = GroupedPlan::new(inputs.hidden_shape.dimensions()[0]);
    let block_n = plan.block_n;
    let block_k = plan.block_k;
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
    let gate_specification = KernelSpec::new(
        build_nvfp4_grouped_projection(gate_config).map_err(kernel_error)?,
        vec![
            tensor(dtype, inputs.hidden_shape.dimensions())?,
            tensor(KernelDType::I32, inputs.schedule_shape.dimensions())?,
            tensor(KernelDType::I32, inputs.block_experts_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
            tensor(KernelDType::U8, inputs.gate_payload_shape.dimensions())?,
            tensor(KernelDType::U8, inputs.gate_scales_shape.dimensions())?,
            tensor(KernelDType::F32, inputs.gate_global_shape.dimensions())?,
            tensor(dtype, inputs.gate_bias_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
        ],
        vec![tensor(dtype, &[assignments, intermediate])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let gate_call = gate_specification
        .lower(
            context,
            &[
                ("input", inputs.hidden),
                ("sorted_assignments", inputs.sorted_assignments),
                ("block_experts", inputs.block_experts),
                ("active_blocks", inputs.active_blocks),
                ("payload", inputs.gate_payload),
                ("block_scales", inputs.gate_scales),
                ("global_scale", inputs.gate_global),
                ("bias", inputs.gate_bias),
                ("expert_offset", expert_offset),
            ],
            grouped_launch(
                inputs.block_experts_shape.dimensions()[0],
                intermediate,
                block_n,
                plan.warps,
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
    let down_specification = KernelSpec::new(
        build_nvfp4_grouped_projection(down_config).map_err(kernel_error)?,
        vec![
            tensor(dtype, &[assignments, intermediate])?,
            tensor(KernelDType::I32, inputs.schedule_shape.dimensions())?,
            tensor(KernelDType::I32, inputs.block_experts_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
            tensor(KernelDType::U8, inputs.down_payload_shape.dimensions())?,
            tensor(KernelDType::U8, inputs.down_scales_shape.dimensions())?,
            tensor(KernelDType::F32, inputs.down_global_shape.dimensions())?,
            tensor(dtype, inputs.down_bias_shape.dimensions())?,
            tensor(KernelDType::I32, &[])?,
            tensor(dtype, inputs.routing_shape.dimensions())?,
        ],
        vec![tensor(dtype, &[assignments, hidden_size])?],
        vec![],
    )
    .map_err(kernel_error)?;
    let down_call = down_specification
        .lower(
            context,
            &[
                ("input", gate_output_value),
                ("sorted_assignments", inputs.sorted_assignments),
                ("block_experts", inputs.block_experts),
                ("active_blocks", inputs.active_blocks),
                ("payload", inputs.down_payload),
                ("block_scales", inputs.down_scales),
                ("global_scale", inputs.down_global),
                ("bias", inputs.down_bias),
                ("expert_offset", expert_offset),
                ("routing_weights", inputs.routing_weights),
            ],
            grouped_launch(
                inputs.block_experts_shape.dimensions()[0],
                hidden_size,
                block_n,
                plan.warps,
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
    let config = NvFp4EmbeddingConfig {
        dtype: kernel_dtype(inputs.result_shape.dtype(), "NVFP4 embedding")?,
        index_dtype: kernel_index_dtype(inputs.indices_shape.dtype())?,
        rows,
        vocabulary,
        width,
        block_m: 16,
        block_n: 64,
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
                warps: 4,
                stages: 2,
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

        // Decode (M=1) GEMV is a memory-bandwidth-bound problem. Each program
        // handles 8 output columns over a 256-wide K tile with 4 warps and a
        // single pipeline stage — Recipe v2 proven geometry. Narrow tiles keep
        // grid blocks numerous (360 for N=2880), filling SM count to hide
        // memory latency. Fewer K-iterations reduce loop and decode overhead.
        // Prefill uses the tensor-core matrix family. Small batches get
        // narrower output tiles and deeper pipelines; large batches widen
        // the output tile and optionally use 8 warps for warp-group MMA.
        let block_m = if rows == 1 {
            1
        } else if rows <= 16 {
            16
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
                block_n: if rows == 1 {
                    if outputs >= 65_536 {
                        32
                    } else {
                        8
                    }
                } else if latency_sensitive {
                    64
                } else {
                    128
                },
                block_k: if rows == 1 {
                    256
                } else if latency_sensitive {
                    128
                } else {
                    64
                },
                has_bias,
            },
            warps: if rows > 128 && capabilities.supports_warp_group_mma() {
                8
            } else {
                4
            },
            stages: if rows == 1 {
                1
            } else if latency_sensitive {
                4
            } else {
                3
            },
        })
    }
}

impl GroupedPlan {
    /// Selects from a finite, reviewable tile family. Decode and small batches
    /// are dominated by expert-weight traffic, so they use a wider K tile and
    /// four pipeline stages. Larger M exposes activation reuse and uses wider
    /// output tiles; sufficiently large batches employ eight warps on every
    /// retained Triton-capable NVIDIA generation. These boundaries deliberately
    /// match ZML's built-in grouped-MoE policy.
    const fn new(tokens: i64) -> Self {
        if tokens == 1 {
            Self {
                block_n: 8,
                block_k: 256,
                warps: 4,
                stages: 1,
            }
        } else if tokens <= 32 {
            Self {
                block_n: 64,
                block_k: 128,
                warps: 4,
                stages: 4,
            }
        } else if tokens <= 64 {
            Self {
                block_n: 64,
                block_k: 128,
                warps: 4,
                stages: 3,
            }
        } else if tokens <= 128 {
            Self {
                block_n: 128,
                block_k: 64,
                warps: 4,
                stages: 3,
            }
        } else {
            Self {
                block_n: 128,
                block_k: 64,
                warps: 8,
                stages: 3,
            }
        }
    }
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
