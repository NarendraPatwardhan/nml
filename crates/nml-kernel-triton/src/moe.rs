//! Grouped expert projection for the private CUDA MoE specialization.
//!
//! Routing and assignment ordering are authored in StableHLO by `nml-ir`.
//! This kernel therefore has one job: consume aligned expert blocks and issue
//! tiled expert matrix multiplications. Keeping the schedule outside TTIR
//! avoids duplicating selection semantics in a backend-specific language.

use super::{ArgumentKind, Builder, Comparison, DType, Error};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatedActivation {
    Silu,
    Gelu,
    Relu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedProjectionConfig {
    pub dtype: DType,
    pub assignments: i64,
    pub input_size: i64,
    pub output_size: i64,
    pub local_experts: i64,
    pub source_row_divisor: i64,
    pub block_m: i64,
    pub block_n: i64,
    pub block_k: i64,
    pub gated_activation: Option<GatedActivation>,
    pub multiply_routing_weight: bool,
}

impl GroupedProjectionConfig {
    fn validate(self) -> Result<Self, Error> {
        let tiled = |value: i64| value > 0 && (value as u64).is_power_of_two();
        let fits_i32 = |value: i64| value > 0 && i32::try_from(value).is_ok();
        if !matches!(self.dtype, DType::F16 | DType::Bf16 | DType::F32)
            || !fits_i32(self.assignments)
            || !fits_i32(self.input_size)
            || !fits_i32(self.output_size)
            || !fits_i32(self.local_experts)
            || !fits_i32(self.source_row_divisor)
            || !tiled(self.block_m)
            || !tiled(self.block_n)
            || !tiled(self.block_k)
            || self.block_m > 128
            || self.block_n > 128
            || self.block_k > 128
            || self.input_size.checked_mul(self.output_size).is_none()
            || (self.gated_activation.is_some() && self.input_size.checked_mul(2).is_none())
        {
            return Err(Error::InvalidKernelSpec(
                "invalid grouped expert-projection specialization",
            ));
        }
        Ok(self)
    }
}

/// Builds either the gate/up or down expert projection. The function name is
/// part of XLA's custom-call ABI and intentionally fixed by the semantic role.
pub fn build_grouped_projection(config: GroupedProjectionConfig) -> Result<String, Error> {
    let config = config.validate()?;
    let name = if config.multiply_routing_weight {
        "moe_grouped_down"
    } else {
        "moe_grouped_gate_up"
    };
    let mut builder = Builder::new(name)?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let sorted_assignments = pointer(&mut builder, "sorted_assignments", DType::I32)?;
    let block_experts = pointer(&mut builder, "block_experts", DType::I32)?;
    let weights = pointer(&mut builder, "weights", config.dtype)?;
    let expert_offset_pointer = pointer(&mut builder, "expert_offset", DType::I32)?;
    let routing_weights = config
        .multiply_routing_weight
        .then(|| pointer(&mut builder, "routing_weights", config.dtype))
        .transpose()?;
    let output = pointer(&mut builder, "output", config.dtype)?;

    let block_index = builder.program_id(0)?;
    let output_block = builder.program_id(1)?;
    let row_lanes = builder.range(0, config.block_m as i32)?;
    let column_lanes = builder.range(0, config.block_n as i32)?;
    let block_m = builder.integer(config.block_m, DType::I32)?;
    let block_n = builder.integer(config.block_n, DType::I32)?;
    let row_start = builder.multiply(&block_index, &block_m)?;
    let column_start = builder.multiply(&output_block, &block_n)?;
    let schedule_positions = builder.add(&row_start, &row_lanes)?;
    let assignment_addresses = builder.add_pointer(&sorted_assignments, &schedule_positions)?;
    let assignment_sentinel =
        builder.full_integer(&[config.block_m], config.assignments, DType::I32)?;
    let assignments = builder.load(&assignment_addresses)?;
    let valid_assignment = builder.compare(Comparison::Less, &assignments, &assignment_sentinel)?;
    // A masked memory operation still needs an in-bounds address. The padded
    // schedule stores `assignments` as its sentinel, exactly one row beyond
    // the real assignment buffer. Clamp only the value used in addresses;
    // every semantic decision continues to use the original assignment and
    // `valid_assignment` mask.
    let last_assignment = builder.integer(config.assignments - 1, DType::I32)?;
    let address_assignment = builder.minimum(&assignments, &last_assignment)?;
    let expert_address = builder.add_pointer(&block_experts, &block_index)?;
    let global_expert = builder.load(&expert_address)?;
    let expert_offset = builder.load(&expert_offset_pointer)?;
    let expert = builder.subtract(&global_expert, &expert_offset)?;
    let zero_i32 = builder.integer(0, DType::I32)?;
    let after_first_expert = builder.compare(Comparison::GreaterEqual, &expert, &zero_i32)?;
    let local_experts = builder.integer(config.local_experts, DType::I32)?;
    let before_last_expert = builder.compare(Comparison::Less, &expert, &local_experts)?;
    let valid_expert = builder.bit_and(&after_first_expert, &before_last_expert)?;
    let compute_rows = builder.bit_and(&valid_assignment, &valid_expert)?;
    // Masked loads still require an in-bounds pointer. Clamp non-local and
    // padding-block expert ids to an arbitrary local expert; `compute_rows`
    // keeps their input tile zero, so the loaded weights cannot contribute.
    let last_local_expert = builder.integer(config.local_experts - 1, DType::I32)?;
    let address_expert = builder.maximum(&expert, &zero_i32)?;
    let address_expert = builder.minimum(&address_expert, &last_local_expert)?;
    let columns = builder.add(&column_start, &column_lanes)?;
    let output_size = builder.integer(config.output_size, DType::I32)?;
    let valid_columns = builder.compare(Comparison::Less, &columns, &output_size)?;

    let divisor = builder.integer(config.source_row_divisor, DType::I32)?;
    let source_rows = builder.divide(&address_assignment, &divisor)?;
    let source_rows = builder.cast(&source_rows, DType::I64)?;
    let input_size_i64 = builder.integer(config.input_size, DType::I64)?;
    let input_stride = if config.gated_activation.is_some() {
        config
            .input_size
            .checked_mul(2)
            .ok_or(Error::InvalidKernelSpec(
                "gated expert-projection input stride overflows",
            ))?
    } else {
        config.input_size
    };
    let input_stride = builder.integer(input_stride, DType::I64)?;
    let source_bases = builder.multiply(&source_rows, &input_stride)?;
    let source_bases = builder.expand_dimension(&source_bases, 1)?;

    let expert = builder.cast(&address_expert, DType::I64)?;
    let expert_stride =
        config
            .input_size
            .checked_mul(config.output_size)
            .ok_or(Error::InvalidKernelSpec(
                "grouped expert-projection weight stride overflows",
            ))?;
    let expert_stride = builder.integer(expert_stride, DType::I64)?;
    let expert_base = builder.multiply(&expert, &expert_stride)?;
    let columns_i64 = builder.cast(&columns, DType::I64)?;
    let weight_rows = builder.multiply(&columns_i64, &input_size_i64)?;
    let weight_rows = builder.add(&expert_base, &weight_rows)?;
    let weight_rows = builder.expand_dimension(&weight_rows, 0)?;

    let input_zero = builder.full_float(&[config.block_m, config.block_k], 0.0, config.dtype)?;
    let weight_zero = builder.full_float(&[config.block_k, config.block_n], 0.0, config.dtype)?;
    let accumulator = builder.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?;
    let k_lanes = builder.range(0, config.block_k as i32)?;
    let input_size = builder.integer(config.input_size, DType::I32)?;
    let k_lower = builder.integer(0, DType::I32)?;
    let k_step = builder.integer(config.block_k, DType::I32)?;
    // Keep one K-tile body in TTIR. Expanding one dot per tile makes compiler
    // work proportional to model width; Triton and ZML represent this as an
    // SCF loop so realistic hidden sizes remain compact specializations.
    let accumulated = builder.for_loop(
        &k_lower,
        &input_size,
        &k_step,
        std::slice::from_ref(&accumulator),
        |body, k_start, carried| {
            let k = body.add(&k_start, &k_lanes)?;
            let valid_k = body.compare(Comparison::Less, &k, &input_size)?;
            let input_mask = body.mask_2d(&compute_rows, &valid_k)?;
            let weight_mask = body.mask_2d(&valid_k, &valid_columns)?;

            let k_i64 = body.cast(&k, DType::I64)?;
            let input_columns = body.expand_dimension(&k_i64, 0)?;
            let input_offsets = body.add(&source_bases, &input_columns)?;
            let input_addresses = body.add_pointer(&input, &input_offsets)?;
            let mut input_tile = body.load_masked(&input_addresses, &input_mask, &input_zero)?;
            if let Some(activation) = config.gated_activation {
                let value_offset = body.integer(config.input_size, DType::I64)?;
                let value_offsets = body.add(&input_offsets, &value_offset)?;
                let value_addresses = body.add_pointer(&input, &value_offsets)?;
                let value_tile = body.load_masked(&value_addresses, &input_mask, &input_zero)?;
                input_tile =
                    gated_activation(body, input_tile, value_tile, activation, config.dtype)?;
            }

            let weight_inputs = body.expand_dimension(&k_i64, 1)?;
            let weight_offsets = body.add(&weight_rows, &weight_inputs)?;
            let weight_addresses = body.add_pointer(&weights, &weight_offsets)?;
            let weight_tile = body.load_masked(&weight_addresses, &weight_mask, &weight_zero)?;
            Ok(vec![body.dot(&input_tile, &weight_tile, &carried[0])?])
        },
    )?;
    let mut accumulator = accumulated[0].clone();

    if let Some(routing_weights) = routing_weights {
        let routing_addresses = builder.add_pointer(&routing_weights, &address_assignment)?;
        let routing_zero = builder.full_float(&[config.block_m], 0.0, config.dtype)?;
        let routing = builder.load_masked(&routing_addresses, &compute_rows, &routing_zero)?;
        let routing = builder.cast(&routing, DType::F32)?;
        let routing = builder.expand_dimension(&routing, 1)?;
        accumulator = builder.multiply(&accumulator, &routing)?;
    }

    let assignments_i64 = builder.cast(&address_assignment, DType::I64)?;
    let output_size_i64 = builder.integer(config.output_size, DType::I64)?;
    let output_rows = builder.multiply(&assignments_i64, &output_size_i64)?;
    let output_rows = builder.expand_dimension(&output_rows, 1)?;
    let columns_i64 = builder.cast(&columns, DType::I64)?;
    let columns_i64 = builder.expand_dimension(&columns_i64, 0)?;
    let output_offsets = builder.add(&output_rows, &columns_i64)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    // Every real assignment is written on every partition. Partitions that do
    // not own the assignment's expert write the zero accumulator, making the
    // following all-reduce independent of uninitialized output storage.
    let output_mask = builder.mask_2d(&valid_assignment, &valid_columns)?;
    let output_values = builder.cast(&accumulator, config.dtype)?;
    builder.store_masked(&output_addresses, &output_values, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}

fn gated_activation(
    builder: &mut Builder,
    gate: super::Value,
    value: super::Value,
    activation: GatedActivation,
    dtype: DType,
) -> Result<super::Value, Error> {
    let shape = match activation {
        GatedActivation::Relu => {
            let zero = builder.full_float_like(&gate, 0.0)?;
            builder.maximum(&gate, &zero)?
        }
        GatedActivation::Silu => {
            let gate_f32 = builder.cast(&gate, DType::F32)?;
            let scale = builder.full_float_like(&gate_f32, -std::f64::consts::LOG2_E)?;
            let exponent = builder.multiply(&gate_f32, &scale)?;
            let exponent = builder.exp2(&exponent)?;
            let one = builder.full_float_like(&gate_f32, 1.0)?;
            let denominator = builder.add(&one, &exponent)?;
            let sigmoid = builder.divide(&one, &denominator)?;
            let activated = builder.multiply(&gate_f32, &sigmoid)?;
            builder.cast(&activated, dtype)?
        }
        GatedActivation::Gelu => {
            let gate_f32 = builder.cast(&gate, DType::F32)?;
            let square = builder.multiply(&gate_f32, &gate_f32)?;
            let cube = builder.multiply(&square, &gate_f32)?;
            let correction = builder.full_float_like(&gate_f32, 0.044715)?;
            let correction = builder.multiply(&cube, &correction)?;
            let inner = builder.add(&gate_f32, &correction)?;
            let scale = builder.full_float_like(&gate_f32, 0.7978845608028654)?;
            let inner = builder.multiply(&inner, &scale)?;
            let inner = tanh_via_exp2(builder, &inner)?;
            let one = builder.full_float_like(&gate_f32, 1.0)?;
            let inner = builder.add(&one, &inner)?;
            let half = builder.full_float_like(&gate_f32, 0.5)?;
            let activated = builder.multiply(&gate_f32, &inner)?;
            let activated = builder.multiply(&activated, &half)?;
            builder.cast(&activated, dtype)?
        }
    };
    builder.multiply(&shape, &value)
}

/// Expresses tanh through the exp2 operation accepted by XLA's retained
/// Triton pipeline. The direct `math.tanh` TTIR operation verifies in MLIR but
/// is not legalizable by that pipeline. Evaluating the exponential at
/// `-2 * abs(x)` keeps the approximation bounded for both signs instead of
/// overflowing on large negative GELU inputs.
fn tanh_via_exp2(builder: &mut Builder, value: &super::Value) -> Result<super::Value, Error> {
    let negative_value = builder.negate(value)?;
    let magnitude = builder.maximum(value, &negative_value)?;
    let exponent_scale = builder.full_float_like(&magnitude, -2.0 * std::f64::consts::LOG2_E)?;
    let exponent = builder.multiply(&magnitude, &exponent_scale)?;
    let exponential = builder.exp2(&exponent)?;
    let one = builder.full_float_like(&magnitude, 1.0)?;
    let numerator = builder.subtract(&one, &exponential)?;
    let denominator = builder.add(&one, &exponential)?;
    let positive = builder.divide(&numerator, &denominator)?;
    let negative = builder.negate(&positive)?;
    let zero = builder.full_float_like(value, 0.0)?;
    let nonnegative = builder.compare(Comparison::GreaterEqual, value, &zero)?;
    builder.select(&nonnegative, &positive, &negative)
}

fn pointer(builder: &mut Builder, name: &str, dtype: DType) -> Result<super::Value, Error> {
    builder.argument(
        name,
        ArgumentKind::Pointer {
            element: dtype,
            address_space: 1,
        },
        Some(16),
    )
}
