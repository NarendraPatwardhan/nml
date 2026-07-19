//! Private CUDA lowering for semantic NVFP4 operations.
//!
//! Model code sees an ordinary linear operation over a logical parameter.
//! This module is the only place where that operation selects a compact CUDA
//! execution family: source-owned output-owner kernels for single-row decode,
//! or matrix-oriented CUDA/Triton lowering for larger row counts. The source
//! payload and scale tensors pass through unchanged and decode only inside the
//! selected contraction.

use crate::{Error, device_capabilities::CudaCapabilities};
use nml_kernel_triton::{
    DType as KernelDType, KernelLaunch, KernelSpec, NvFp4EmbeddingConfig,
    NvFp4GroupedProjectionConfig, NvFp4GroupedRole, NvFp4LinearConfig, TensorSpec,
    build_nvfp4_embedding, build_nvfp4_grouped_projection, build_nvfp4_linear,
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

pub(crate) struct ExpertInputs<'context> {
    pub hidden: Value<'context>,
    pub expert_ids: Value<'context>,
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
    pub router: Option<RouterInputs<'context>>,
    pub hidden_shape: Shape,
    pub expert_ids_shape: Shape,
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

#[derive(Clone, Copy)]
pub(crate) struct RouterInputs<'context> {
    pub weight: Value<'context>,
    pub bias: Value<'context>,
    pub weight_shape: Shape,
    pub bias_shape: Shape,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodePlan {
    target: &'static str,
}

pub(crate) fn lower_linear<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: LinearInputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    require_unsharded(&inputs)?;
    let rows = input_rows(inputs.activation_shape)?;
    if capabilities.supports_nvfp4_cuda_custom_call()
        && (rows == 1 || capabilities.requires_nvfp4_cuda_matrix())
    {
        return lower_cuda_linear(context, block, inputs, capabilities);
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

/// Lowers a semantic set of independent projections over one activation.
///
/// The grouped CUDA call is deliberately a decode specialization. Prefill and
/// unsupported representation details reuse the ordinary compact linear
/// lowering, so grouping never changes graph semantics or removes a more
/// capable matrix path.
pub(crate) fn lower_linear_group<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: Vec<LinearInputs<'context>>,
    capabilities: CudaCapabilities,
) -> Result<Vec<Value<'context>>, Error> {
    if inputs.len() != 3 {
        return Err(Error::InvalidLinearAlgebra(
            "the compact CUDA linear group requires exactly three projections",
        ));
    }
    let activation_shape = inputs[0].activation_shape;
    let rows = input_rows(activation_shape)?;
    let can_share_decode = rows == 1
        && capabilities.supports_nvfp4_cuda_custom_call()
        && inputs.iter().all(|input| {
            input.activation_shape == activation_shape
                && input.bias.is_some()
                && input.bias_shape.is_some()
        });
    if can_share_decode {
        for input in &inputs {
            require_unsharded(input)?;
        }
        let mut operands = vec![inputs[0].activation];
        let mut result_types = Vec::with_capacity(inputs.len());
        for input in &inputs {
            let bias = input.bias.ok_or(Error::InvalidLinearAlgebra(
                "grouped compact decode requires one bias per projection",
            ))?;
            operands.extend([
                input.payload,
                input.block_scales,
                input.global_scale,
                bias,
            ]);
            result_types.push(input.result_type);
        }
        let mut total_outputs = 0_i64;
        for input in &inputs {
            let outputs = input.result_shape.dimensions().last().copied().ok_or(
                Error::InvalidLinearAlgebra(
                    "grouped compact decode result must have an output axis",
                ),
            )?;
            total_outputs = total_outputs.checked_add(outputs).ok_or(
                Error::InvalidLinearAlgebra(
                    "grouped compact decode output extent overflows",
                ),
            )?;
        }
        let plan = DecodePlan::new(total_outputs, capabilities, true);
        let call = context.ffi_custom_call(plan.target, &operands, &result_types)?;
        let results = (0..inputs.len())
            .map(|index| call.result(index))
            .collect::<Result<Vec<_>, _>>()?;
        block.append_operation(call)?;
        return Ok(results);
    }

    let mut results = Vec::with_capacity(inputs.len());
    for input in inputs {
        results.push(lower_linear(context, block, input, capabilities)?);
    }
    Ok(results)
}

pub(crate) fn lower_linear_top64<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: LinearInputs<'context>,
    values_type: Type<'context>,
    indices_type: Type<'context>,
) -> Result<(Value<'context>, Value<'context>), Error> {
    require_unsharded(&inputs)?;
    if input_rows(inputs.activation_shape)? != 1 || inputs.bias.is_some() {
        return Err(Error::UnsupportedTarget {
            operation: "compact linear top-64",
            target: "pre-Blackwell CUDA decode".to_owned(),
            requirement: "one activation row and a bias-free compact projection",
        });
    }
    let outputs = *inputs
        .result_shape
        .dimensions()
        .last()
        .ok_or(Error::InvalidLinearAlgebra(
            "compact linear top-64 result must have an output axis",
        ))?;
    if outputs < 64 {
        return Err(Error::InvalidSort(
            "compact linear top-64 requires at least 64 outputs",
        ));
    }
    let groups = outputs
        .checked_add(127)
        .map(|extent| extent / 128)
        .ok_or(Error::InvalidSort(
            "compact linear top-64 workspace extent overflows",
        ))?;
    let workspace_values = context.ranked_tensor_type(DType::F32, &[groups, 64])?;
    let workspace_indices = context.ranked_tensor_type(DType::I32, &[groups, 64])?;
    let call = context.ffi_custom_call(
        "nml.nvfp4.cuda.linear_top64_m1",
        &[
            inputs.activation,
            inputs.payload,
            inputs.block_scales,
            inputs.global_scale,
        ],
        &[
            workspace_values,
            workspace_indices,
            workspace_values,
            workspace_indices,
            values_type,
            indices_type,
        ],
    )?;
    let values = call.result(4)?;
    let indices = call.result(5)?;
    block.append_operation(call)?;
    Ok((values, indices))
}

pub(crate) fn lower_routed_swiglu<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: ExpertInputs<'context>,
    capabilities: CudaCapabilities,
) -> Result<Value<'context>, Error> {
    let tokens = inputs.hidden_shape.dimensions()[0];
    let direct_decode = tokens == 1 && capabilities.supports_nvfp4_cuda_custom_call();
    let use_cuda_matrix = capabilities.requires_nvfp4_cuda_matrix();
    if use_cuda_matrix && !direct_decode {
        if inputs.expert_offset.is_some() {
            return Err(Error::UnsupportedTarget {
                operation: "NVFP4 routed clamped SwiGLU",
                target: "sharded CUDA SM75 execution".to_owned(),
                requirement: "the SM75 matrix adapter currently owns one complete local expert set",
            });
        }
        require_unsharded_experts(&inputs)?;
    } else if !direct_decode {
        require_triton_emulation(capabilities, "NVFP4 routed clamped SwiGLU")?;
        if inputs.expert_offset.is_none() {
            require_unsharded_experts(&inputs)?;
        }
    } else if inputs.expert_offset.is_none() {
        require_unsharded_experts(&inputs)?;
    }
    let hidden_size = inputs.hidden_shape.dimensions()[1];
    let experts_per_token = inputs.routing_shape.dimensions()[1];
    let assignments = tokens
        .checked_mul(experts_per_token)
        .ok_or(Error::InvalidMoe("NVFP4 assignment count overflows"))?;
    let local_experts = inputs.gate_payload_shape.dimensions()[0];
    let gate_input = inputs.gate_payload_shape.dimensions()[1];
    let gate_output = inputs.gate_bias_shape.dimensions()[1];
    let intermediate = inputs.down_payload_shape.dimensions()[1];
    let down_output = inputs.down_bias_shape.dimensions()[1];
    let expected_gate_output = intermediate.checked_mul(2).ok_or(Error::InvalidMoe(
        "NVFP4 expert intermediate width overflows",
    ))?;
    if gate_input != hidden_size
        || gate_output != expected_gate_output
        || down_output != hidden_size
    {
        return Err(Error::InvalidMoe(
            "NVFP4 grouped expert logical dimensions are inconsistent",
        ));
    }

    if direct_decode {
        return lower_cuda_direct_experts(
            context,
            block,
            &inputs,
            assignments,
            intermediate,
        );
    }

    let down = if use_cuda_matrix {
        lower_cuda_experts(
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

#[allow(clippy::too_many_arguments)]
fn lower_cuda_direct_experts<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: &ExpertInputs<'context>,
    routes: i64,
    intermediate: i64,
) -> Result<Value<'context>, Error> {
    if inputs.expert_ids_shape.dimensions() != [1, routes] {
        return Err(Error::InvalidMoe(
            "direct NVFP4 decode requires one row of route IDs",
        ));
    }
    let (expert_ids, routing_weights) = match inputs.router {
        Some(router) => {
            let [experts, router_inputs] = router.weight_shape.dimensions() else {
                return Err(Error::InvalidMoe(
                    "direct router weight must have shape [experts, hidden]",
                ));
            };
            if routes != 4
                || *experts < 4
                || *experts > 32
                || *router_inputs != inputs.hidden_shape.dimensions()[1]
                || router.bias_shape.dimensions() != [*experts]
                || router.weight_shape.dtype() != inputs.hidden_shape.dtype()
                || router.bias_shape.dtype() != inputs.hidden_shape.dtype()
                || router
                    .weight_shape
                    .partitions()
                    .iter()
                    .chain(router.bias_shape.partitions())
                    .any(|partition| matches!(partition, Partition::Sharded(_)))
            {
                return Err(Error::UnsupportedTarget {
                    operation: "direct NVFP4 MoE router",
                    target: "pre-Blackwell CUDA decode".to_owned(),
                    requirement:
                        "one dense replicated F16/BF16 router with 4 selected from at most 32 experts",
                });
            }
            let ids_type = context.ranked_tensor_type(DType::I32, &[1, routes])?;
            let weights_type =
                context.ranked_tensor_type(inputs.hidden_shape.dtype(), &[1, routes])?;
            let route_call = context.ffi_custom_call(
                "nml.nvfp4.cuda.route_top4_m1",
                &[inputs.hidden, router.weight, router.bias],
                &[ids_type, weights_type],
            )?;
            let ids = route_call.result(0)?;
            let weights = route_call.result(1)?;
            block.append_operation(route_call)?;
            (ids, weights)
        }
        None => (inputs.expert_ids, inputs.routing_weights),
    };
    let expert_offset = match inputs.expert_offset {
        Some(value) => value,
        None => {
            let scalar_i32 = context.ranked_tensor_type(DType::I32, &[])?;
            constant(context, block, scalar_i32, "0")?
        }
    };
    let activated_type =
        context.ranked_tensor_type(inputs.hidden_shape.dtype(), &[routes, intermediate])?;
    let gate_call = context.ffi_custom_call(
        "nml.nvfp4.cuda.expert_gate_up_m1",
        &[
            inputs.hidden,
            expert_ids,
            inputs.gate_payload,
            inputs.gate_scales,
            inputs.gate_global,
            inputs.gate_bias,
            expert_offset,
        ],
        &[activated_type],
    )?;
    let activated = gate_call.result(0)?;
    block.append_operation(gate_call)?;

    let down_call = context.ffi_custom_call(
        "nml.nvfp4.cuda.expert_down_m1",
        &[
            activated,
            expert_ids,
            inputs.down_payload,
            inputs.down_scales,
            inputs.down_global,
            inputs.down_bias,
            routing_weights,
            expert_offset,
        ],
        &[inputs.result_type],
    )?;
    let result = down_call.result(0)?;
    block.append_operation(down_call)?;
    Ok(result)
}

fn lower_cuda_experts<'context>(
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
        "nml.nvfp4.cuda.expert_gate_up",
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
        "nml.nvfp4.cuda.expert_down",
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
    if capabilities.requires_nvfp4_cuda_matrix() {
        return lower_cuda_embedding(context, block, inputs);
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

fn lower_cuda_linear<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: LinearInputs<'context>,
    capabilities: CudaCapabilities,
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
    let outputs = *inputs.result_shape.dimensions().last().ok_or(
        Error::InvalidLinearAlgebra("NVFP4 CUDA linear result must have an output axis"),
    )?;
    let target = if input_rows(inputs.activation_shape)? == 1 {
        DecodePlan::new(outputs, capabilities, false).target
    } else {
        // SM75 cannot use the Triton matrix path, so it retains the dedicated
        // CUDA matrix adapter. Keep its target identity distinct from the
        // latency-oriented M=1 family: executable cache keys and traces must
        // describe the execution regime truthfully.
        "nml.nvfp4.cuda.linear_matrix"
    };
    let call = context.ffi_custom_call(target, &operands, &[inputs.result_type])?;
    let result = call.result(0)?;
    block.append_operation(call)?;
    Ok(result)
}

impl DecodePlan {
    fn new(outputs: i64, capabilities: CudaCapabilities, grouped: bool) -> Self {
        // Eight output owners amortize activation staging when they still
        // provide at least three resident blocks per SM. Narrow projections
        // retain four owners to expose enough independently schedulable work.
        // Core count and target name make the finite choice part of the
        // compiler/cache identity rather than an execution-time device query.
        let blocks_with_eight = outputs.saturating_add(7) / 8;
        let occupancy_floor = i64::try_from(capabilities.core_count().saturating_mul(3))
            .unwrap_or(i64::MAX);
        let eight_warps = blocks_with_eight >= occupancy_floor;
        let target = match (grouped, eight_warps) {
            (false, false) => "nml.nvfp4.cuda.linear_m1_w4",
            (false, true) => "nml.nvfp4.cuda.linear_m1_w8",
            (true, false) => "nml.nvfp4.cuda.linear_group3_m1_w4",
            (true, true) => "nml.nvfp4.cuda.linear_group3_m1_w8",
        };
        Self { target }
    }
}

fn lower_cuda_embedding<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: EmbeddingInputs<'context>,
) -> Result<Value<'context>, Error> {
    let call = context.ffi_custom_call(
        "nml.nvfp4.cuda.embedding",
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

        // Decode is latency-sensitive at small M. Hopper-and-newer devices
        // retain a 64-row tile once enough rows exist for warp-group MMA;
        // Ampere/Ada use the smaller tile accepted by their ordinary tt.dot
        // lowering. This is private tuning, not an architecture-facing API.
        let block_m = if rows <= 16 {
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
                block_n: if latency_sensitive { 64 } else { 128 },
                block_k: if latency_sensitive { 128 } else { 64 },
                has_bias,
            },
            warps: if rows > 128 && capabilities.supports_warp_group_mma() {
                8
            } else {
                4
            },
            stages: if latency_sensitive { 4 } else { 3 },
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
        if tokens <= 32 {
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

fn input_rows(shape: Shape) -> Result<i64, Error> {
    if shape.rank() == 0 {
        return Err(Error::InvalidLinearAlgebra(
            "NVFP4 linear activation must have rank",
        ));
    }
    shape.dimensions()[..shape.rank() - 1]
        .iter()
        .try_fold(1_i64, |product, dimension| product.checked_mul(*dimension))
        .filter(|rows| *rows > 0)
        .ok_or(Error::InvalidLinearAlgebra(
            "NVFP4 linear row extent must be positive and fit I64",
        ))
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
        inputs.expert_ids_shape,
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
