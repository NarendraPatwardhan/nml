//! Fused compact-weight NVFP4 linear construction for pre-Blackwell CUDA.
//!
//! The source representation is row-major `[N, K]`: payload bytes and E4M3FN
//! scales share the N row, while the K tile is decoded directly into the
//! tensor-core operand. Only the current tile exists at activation precision.

use super::{ArgumentKind, Builder, Comparison, DType, Error};

const REPRESENTATION_BLOCK: i64 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4LinearConfig {
    pub dtype: DType,
    pub rows: i64,
    pub outputs: i64,
    pub inputs: i64,
    pub block_m: i64,
    pub block_n: i64,
    pub block_k: i64,
    pub has_bias: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NvFp4GroupedRole {
    GateUp,
    ClampedSwiGluDown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4GroupedProjectionConfig {
    pub dtype: DType,
    pub assignments: i64,
    pub input_size: i64,
    pub output_size: i64,
    pub local_experts: i64,
    pub source_row_divisor: i64,
    pub block_m: i64,
    pub block_n: i64,
    pub block_k: i64,
    pub role: NvFp4GroupedRole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4EmbeddingConfig {
    pub dtype: DType,
    pub index_dtype: DType,
    pub rows: i64,
    pub vocabulary: i64,
    pub width: i64,
    pub block_m: i64,
    pub block_n: i64,
}

impl NvFp4EmbeddingConfig {
    pub fn launch_grid(self) -> Result<[i32; 3], Error> {
        self.validate()?;
        Ok([
            i32::try_from(ceil_div(self.rows, self.block_m))
                .map_err(|_| Error::InvalidKernelSpec("embedding row grid exceeds I32"))?,
            i32::try_from(ceil_div(self.width, self.block_n))
                .map_err(|_| Error::InvalidKernelSpec("embedding column grid exceeds I32"))?,
            1,
        ])
    }

    fn validate(self) -> Result<Self, Error> {
        let tiled = |value: i64| value > 0 && (value as u64).is_power_of_two();
        if !matches!(self.dtype, DType::F16 | DType::Bf16)
            || !matches!(self.index_dtype, DType::I32 | DType::I64)
            || [self.rows, self.vocabulary, self.width]
                .into_iter()
                .any(|value| value <= 0)
            || !tiled(self.block_m)
            || !tiled(self.block_n)
            || self.block_m > 128
            || self.block_n > 128
        {
            return Err(Error::InvalidKernelSpec(
                "invalid compact embedding specialization",
            ));
        }
        Ok(self)
    }
}

impl NvFp4GroupedProjectionConfig {
    fn validate(self) -> Result<Self, Error> {
        let tiled = |value: i64| value > 0 && (value as u64).is_power_of_two();
        let fits_i32 = |value: i64| value > 0 && i32::try_from(value).is_ok();
        if !matches!(self.dtype, DType::F16 | DType::Bf16)
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
            || self.block_k % REPRESENTATION_BLOCK != 0
            || self.input_size.checked_mul(self.output_size).is_none()
        {
            return Err(Error::InvalidKernelSpec(
                "invalid compact grouped expert-projection specialization",
            ));
        }
        Ok(self)
    }
}

impl NvFp4LinearConfig {
    pub fn launch_grid(self) -> Result<[i32; 3], Error> {
        self.validate()?;
        let programs = ceil_div(self.rows, self.block_m)
            .checked_mul(ceil_div(self.outputs, self.block_n))
            .ok_or(Error::InvalidKernelSpec("NVFP4 launch grid overflows"))?;
        Ok([
            i32::try_from(programs)
                .map_err(|_| Error::InvalidKernelSpec("NVFP4 launch grid exceeds I32"))?,
            1,
            1,
        ])
    }

    fn validate(self) -> Result<(), Error> {
        if !matches!(self.dtype, DType::F16 | DType::Bf16)
            || [self.rows, self.outputs, self.inputs]
                .into_iter()
                .any(|value| value <= 0)
            || [self.block_m, self.block_n, self.block_k]
                .into_iter()
                .any(|value| value <= 0 || !(value as u64).is_power_of_two())
            || self.block_k % REPRESENTATION_BLOCK != 0
        {
            return Err(Error::InvalidKernelSpec(
                "NVFP4 linear requires F16/BF16, positive geometry, power-of-two tiles, and a K tile divisible by sixteen",
            ));
        }
        Ok(())
    }
}

pub fn build_nvfp4_linear(config: NvFp4LinearConfig) -> Result<String, Error> {
    config.validate()?;
    let block_m = i32::try_from(config.block_m)
        .map_err(|_| Error::InvalidKernelSpec("NVFP4 M tile exceeds I32"))?;
    let block_n = i32::try_from(config.block_n)
        .map_err(|_| Error::InvalidKernelSpec("NVFP4 N tile exceeds I32"))?;
    let block_k = i32::try_from(config.block_k)
        .map_err(|_| Error::InvalidKernelSpec("NVFP4 K tile exceeds I32"))?;
    let grid_n = ceil_div(config.outputs, config.block_n);
    let packed_width = ceil_div(config.inputs, 2);
    let scale_width = ceil_div(config.inputs, REPRESENTATION_BLOCK);

    let mut builder = Builder::new("nvfp4_linear")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = config
        .has_bias
        .then(|| pointer(&mut builder, "bias", config.dtype))
        .transpose()?;
    let output = pointer(&mut builder, "output", config.dtype)?;

    let program = builder.program_id(0)?;
    let grid_n_value = builder.integer(grid_n, DType::I32)?;
    let program_m = builder.divide(&program, &grid_n_value)?;
    let program_n = builder.remainder(&program, &grid_n_value)?;
    let block_m_value = builder.integer(config.block_m, DType::I32)?;
    let block_n_value = builder.integer(config.block_n, DType::I32)?;
    let row_base = builder.multiply(&program_m, &block_m_value)?;
    let output_base = builder.multiply(&program_n, &block_n_value)?;
    let row_range = builder.range(0, block_m)?;
    let rows = builder.add(&row_base, &row_range)?;
    let output_range = builder.range(0, block_n)?;
    let outputs = builder.add(&output_base, &output_range)?;
    let row_limit = builder.integer(config.rows, DType::I32)?;
    let output_limit = builder.integer(config.outputs, DType::I32)?;
    let valid_rows = builder.compare(Comparison::Less, &rows, &row_limit)?;
    let valid_outputs = builder.compare(Comparison::Less, &outputs, &output_limit)?;

    let lower = builder.integer(0, DType::I32)?;
    let upper = builder.integer(config.inputs, DType::I32)?;
    let step = builder.integer(config.block_k, DType::I32)?;
    let accumulator = builder.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?;
    let result = builder.for_loop(
        &lower,
        &upper,
        &step,
        &[accumulator],
        |builder, start, state| {
            let column_range = builder.range(0, block_k)?;
            let columns = builder.add(&start, &column_range)?;
            let input_limit = builder.integer(config.inputs, DType::I32)?;
            let valid_columns = builder.compare(Comparison::Less, &columns, &input_limit)?;

            let row_offsets = builder.multiply(&rows, &input_limit)?;
            let input_offsets = matrix_offsets(builder, &row_offsets, &columns)?;
            let input_pointer = builder.add_pointer(&input, &input_offsets)?;
            let input_mask = builder.mask_2d(&valid_rows, &valid_columns)?;
            let input_zero =
                builder.full_float(&[config.block_m, config.block_k], 0.0, config.dtype)?;
            let input_tile = builder.load_masked(&input_pointer, &input_mask, &input_zero)?;

            let two = builder.integer(2, DType::I32)?;
            let packed_columns = builder.divide(&columns, &two)?;
            let packed_width = builder.integer(packed_width, DType::I32)?;
            let payload_rows = builder.multiply(&outputs, &packed_width)?;
            let payload_offsets = matrix_offsets(builder, &packed_columns, &payload_rows)?;
            let payload_pointer = builder.add_pointer(&payload, &payload_offsets)?;
            let weight_mask = builder.mask_2d(&valid_columns, &valid_outputs)?;
            let payload_zero =
                builder.full_integer(&[config.block_k, config.block_n], 0, DType::U8)?;
            let packed = builder.load_masked(&payload_pointer, &weight_mask, &payload_zero)?;
            let parity = builder.remainder(&columns, &two)?;
            let four = builder.integer(4, DType::I32)?;
            let shift = builder.multiply(&parity, &four)?;
            let shift = builder.cast(&shift, DType::U8)?;
            let shift = builder.expand_dimension(&shift, 1)?;
            let code = builder.shift_right_logical(&packed, &shift)?;
            let nibble_mask =
                builder.full_integer(&[config.block_k, config.block_n], 0x0f, DType::U8)?;
            let code = builder.bit_and(&code, &nibble_mask)?;
            let values = decode_e2m1(builder, &code)?;

            let representation_block = builder.integer(REPRESENTATION_BLOCK, DType::I32)?;
            let scale_columns = builder.divide(&columns, &representation_block)?;
            let scale_width = builder.integer(scale_width, DType::I32)?;
            let scale_rows = builder.multiply(&outputs, &scale_width)?;
            let scale_offsets = matrix_offsets(builder, &scale_columns, &scale_rows)?;
            let scale_pointer = builder.add_pointer(&block_scales, &scale_offsets)?;
            let scale_zero =
                builder.full_integer(&[config.block_k, config.block_n], 0, DType::U8)?;
            let scales = builder.load_masked(&scale_pointer, &weight_mask, &scale_zero)?;
            let scales = decode_e4m3fn(builder, &scales)?;
            let global = builder.load(&global_scale)?;
            let scaled = builder.multiply(&values, &scales)?;
            let scaled = builder.multiply(&scaled, &global)?;
            let scaled = builder.cast(&scaled, config.dtype)?;
            Ok(vec![builder.dot(&input_tile, &scaled, &state[0])?])
        },
    )?[0]
        .clone();

    // Bias is part of the compact projection epilogue. Add it while the
    // contraction tile is still F32 so the optimized path never emits a
    // separate dense graph value or rounds the accumulator before the add.
    let result = if let Some(bias) = bias {
        let bias_pointer = builder.add_pointer(&bias, &outputs)?;
        let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
        let bias = builder.load_masked(&bias_pointer, &valid_outputs, &bias_zero)?;
        let bias = builder.cast(&bias, DType::F32)?;
        let bias = builder.expand_dimension(&bias, 0)?;
        builder.add(&result, &bias)?
    } else {
        result
    };
    let result = builder.cast(&result, config.dtype)?;
    let output_rows = builder.integer(config.outputs, DType::I32)?;
    let row_offsets = builder.multiply(&rows, &output_rows)?;
    let output_offsets = matrix_offsets(&mut builder, &row_offsets, &outputs)?;
    let output_pointer = builder.add_pointer(&output, &output_offsets)?;
    let output_mask = builder.mask_2d(&valid_rows, &valid_outputs)?;
    builder.store_masked(&output_pointer, &result, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}

pub fn build_nvfp4_embedding(config: NvFp4EmbeddingConfig) -> Result<String, Error> {
    let config = config.validate()?;
    let block_m = i32::try_from(config.block_m)
        .map_err(|_| Error::InvalidKernelSpec("embedding M tile exceeds I32"))?;
    let block_n = i32::try_from(config.block_n)
        .map_err(|_| Error::InvalidKernelSpec("embedding N tile exceeds I32"))?;
    let packed_width = ceil_div(config.width, 2);
    let scale_width = ceil_div(config.width, REPRESENTATION_BLOCK);

    let mut builder = Builder::new("nvfp4_embedding")?;
    let indices = pointer(&mut builder, "indices", config.index_dtype)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let output = pointer(&mut builder, "output", config.dtype)?;

    let row_program = builder.program_id(0)?;
    let column_program = builder.program_id(1)?;
    let block_m_value = builder.integer(config.block_m, DType::I32)?;
    let block_n_value = builder.integer(config.block_n, DType::I32)?;
    let row_base = builder.multiply(&row_program, &block_m_value)?;
    let column_base = builder.multiply(&column_program, &block_n_value)?;
    let row_lanes = builder.range(0, block_m)?;
    let column_lanes = builder.range(0, block_n)?;
    let rows = builder.add(&row_base, &row_lanes)?;
    let columns = builder.add(&column_base, &column_lanes)?;
    let row_limit = builder.integer(config.rows, DType::I32)?;
    let column_limit = builder.integer(config.width, DType::I32)?;
    let valid_rows = builder.compare(Comparison::Less, &rows, &row_limit)?;
    let valid_columns = builder.compare(Comparison::Less, &columns, &column_limit)?;

    let index_addresses = builder.add_pointer(&indices, &rows)?;
    let index_zero = builder.full_integer(&[config.block_m], 0, config.index_dtype)?;
    let token = builder.load_masked(&index_addresses, &valid_rows, &index_zero)?;
    let zero = builder.integer(0, config.index_dtype)?;
    let vocabulary = builder.integer(config.vocabulary, config.index_dtype)?;
    let nonnegative = builder.compare(Comparison::GreaterEqual, &token, &zero)?;
    let in_vocabulary = builder.compare(Comparison::Less, &token, &vocabulary)?;
    let valid_token = builder.bit_and(&nonnegative, &in_vocabulary)?;
    let valid_rows = builder.bit_and(&valid_rows, &valid_token)?;
    let mask = builder.mask_2d(&valid_rows, &valid_columns)?;
    let token = builder.cast(&token, DType::I64)?;
    let token = builder.expand_dimension(&token, 1)?;

    let packed_width_value = builder.integer(packed_width, DType::I64)?;
    let payload_rows = builder.multiply(&token, &packed_width_value)?;
    let two = builder.integer(2, DType::I32)?;
    let packed_columns = builder.divide(&columns, &two)?;
    let packed_columns = builder.cast(&packed_columns, DType::I64)?;
    let packed_columns = builder.expand_dimension(&packed_columns, 0)?;
    let payload_offsets = builder.add(&payload_rows, &packed_columns)?;
    let payload_addresses = builder.add_pointer(&payload, &payload_offsets)?;
    let payload_zero = builder.full_integer(&[config.block_m, config.block_n], 0, DType::U8)?;
    let packed = builder.load_masked(&payload_addresses, &mask, &payload_zero)?;
    let parity = builder.remainder(&columns, &two)?;
    let four = builder.integer(4, DType::I32)?;
    let shift = builder.multiply(&parity, &four)?;
    let shift = builder.cast(&shift, DType::U8)?;
    let shift = builder.expand_dimension(&shift, 0)?;
    let code = builder.shift_right_logical(&packed, &shift)?;
    let nibble_mask = builder.full_integer(&[config.block_m, config.block_n], 0x0f, DType::U8)?;
    let code = builder.bit_and(&code, &nibble_mask)?;
    let values = decode_e2m1(&mut builder, &code)?;

    let scale_width_value = builder.integer(scale_width, DType::I64)?;
    let scale_rows = builder.multiply(&token, &scale_width_value)?;
    let representation_block = builder.integer(REPRESENTATION_BLOCK, DType::I32)?;
    let scale_columns = builder.divide(&columns, &representation_block)?;
    let scale_columns = builder.cast(&scale_columns, DType::I64)?;
    let scale_columns = builder.expand_dimension(&scale_columns, 0)?;
    let scale_offsets = builder.add(&scale_rows, &scale_columns)?;
    let scale_addresses = builder.add_pointer(&block_scales, &scale_offsets)?;
    let scales = builder.load_masked(&scale_addresses, &mask, &payload_zero)?;
    let scales = decode_e4m3fn(&mut builder, &scales)?;
    let global = builder.load(&global_scale)?;
    let result = builder.multiply(&values, &scales)?;
    let result = builder.multiply(&result, &global)?;
    let result = builder.cast(&result, config.dtype)?;

    let output_width = builder.integer(config.width, DType::I32)?;
    let output_rows = builder.multiply(&rows, &output_width)?;
    let output_offsets = matrix_offsets(&mut builder, &output_rows, &columns)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    builder.store_masked(&output_addresses, &result, &mask)?;
    builder.return_void()?;
    builder.finish()
}

/// Builds one compact, input-major routed expert projection.
///
/// Routing is represented by the deterministic padded assignment schedule
/// authored in StableHLO. Gate/up writes interleaved biased channels. Down
/// consumes those channels, applies the selected clamped residual SwiGLU,
/// adds the per-expert bias, and applies the route weight.
pub fn build_nvfp4_grouped_projection(
    config: NvFp4GroupedProjectionConfig,
) -> Result<String, Error> {
    let config = config.validate()?;
    let name = match config.role {
        NvFp4GroupedRole::GateUp => "nvfp4_grouped_gate_up",
        NvFp4GroupedRole::ClampedSwiGluDown => "nvfp4_grouped_down",
    };
    let mut builder = Builder::new(name)?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let sorted_assignments = pointer(&mut builder, "sorted_assignments", DType::I32)?;
    let block_experts = pointer(&mut builder, "block_experts", DType::I32)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = pointer(&mut builder, "bias", config.dtype)?;
    let expert_offset_pointer = pointer(&mut builder, "expert_offset", DType::I32)?;
    let routing_weights = matches!(config.role, NvFp4GroupedRole::ClampedSwiGluDown)
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
    let last_local_expert = builder.integer(config.local_experts - 1, DType::I32)?;
    let address_expert = builder.maximum(&expert, &zero_i32)?;
    let address_expert = builder.minimum(&address_expert, &last_local_expert)?;

    let columns = builder.add(&column_start, &column_lanes)?;
    let output_size = builder.integer(config.output_size, DType::I32)?;
    let valid_columns = builder.compare(Comparison::Less, &columns, &output_size)?;
    let divisor = builder.integer(config.source_row_divisor, DType::I32)?;
    let source_rows = builder.divide(&address_assignment, &divisor)?;
    let source_rows = builder.cast(&source_rows, DType::I64)?;
    let input_stride_value = match config.role {
        NvFp4GroupedRole::GateUp => config.input_size,
        NvFp4GroupedRole::ClampedSwiGluDown => {
            config
                .input_size
                .checked_mul(2)
                .ok_or(Error::InvalidKernelSpec(
                    "clamped SwiGLU activation stride overflows",
                ))?
        }
    };
    let input_stride = builder.integer(input_stride_value, DType::I64)?;
    let source_bases = builder.multiply(&source_rows, &input_stride)?;
    let source_bases = builder.expand_dimension(&source_bases, 1)?;

    let packed_width = ceil_div(config.output_size, 2);
    let scale_width = ceil_div(config.output_size, REPRESENTATION_BLOCK);
    let expert = builder.cast(&address_expert, DType::I64)?;
    let payload_expert_stride =
        config
            .input_size
            .checked_mul(packed_width)
            .ok_or(Error::InvalidKernelSpec(
                "compact expert payload stride overflows",
            ))?;
    let scale_expert_stride =
        config
            .input_size
            .checked_mul(scale_width)
            .ok_or(Error::InvalidKernelSpec(
                "compact expert scale stride overflows",
            ))?;
    let payload_expert_stride = builder.integer(payload_expert_stride, DType::I64)?;
    let scale_expert_stride = builder.integer(scale_expert_stride, DType::I64)?;
    let payload_expert_base = builder.multiply(&expert, &payload_expert_stride)?;
    let scale_expert_base = builder.multiply(&expert, &scale_expert_stride)?;

    let input_zero = builder.full_float(&[config.block_m, config.block_k], 0.0, config.dtype)?;
    let payload_zero = builder.full_integer(&[config.block_k, config.block_n], 0, DType::U8)?;
    let accumulator = builder.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?;
    let k_lanes = builder.range(0, config.block_k as i32)?;
    let input_size = builder.integer(config.input_size, DType::I32)?;
    let k_lower = builder.integer(0, DType::I32)?;
    let k_step = builder.integer(config.block_k, DType::I32)?;
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

            let input_columns = match config.role {
                NvFp4GroupedRole::GateUp => body.expand_dimension(&k_i64, 0)?,
                NvFp4GroupedRole::ClampedSwiGluDown => {
                    let two = body.integer(2, DType::I64)?;
                    let interleaved = body.multiply(&k_i64, &two)?;
                    body.expand_dimension(&interleaved, 0)?
                }
            };
            let input_offsets = body.add(&source_bases, &input_columns)?;
            let input_addresses = body.add_pointer(&input, &input_offsets)?;
            let mut input_tile = body.load_masked(&input_addresses, &input_mask, &input_zero)?;
            if matches!(config.role, NvFp4GroupedRole::ClampedSwiGluDown) {
                let one = body.integer(1, DType::I64)?;
                let up_offsets = body.add(&input_offsets, &one)?;
                let up_addresses = body.add_pointer(&input, &up_offsets)?;
                let up = body.load_masked(&up_addresses, &input_mask, &input_zero)?;
                input_tile = clamped_residual_swiglu(body, input_tile, up, config.dtype)?;
            }

            let k_i64 = body.expand_dimension(&k_i64, 1)?;
            let packed_width_value = body.integer(packed_width, DType::I64)?;
            let payload_rows = body.multiply(&k_i64, &packed_width_value)?;
            let payload_rows = body.add(&payload_rows, &payload_expert_base)?;
            let two = body.integer(2, DType::I32)?;
            let packed_columns = body.divide(&columns, &two)?;
            let packed_columns = body.cast(&packed_columns, DType::I64)?;
            let packed_columns = body.expand_dimension(&packed_columns, 0)?;
            let payload_offsets = body.add(&payload_rows, &packed_columns)?;
            let payload_addresses = body.add_pointer(&payload, &payload_offsets)?;
            let packed = body.load_masked(&payload_addresses, &weight_mask, &payload_zero)?;
            let parity = body.remainder(&columns, &two)?;
            let four = body.integer(4, DType::I32)?;
            let shift = body.multiply(&parity, &four)?;
            let shift = body.cast(&shift, DType::U8)?;
            let shift = body.expand_dimension(&shift, 0)?;
            let code = body.shift_right_logical(&packed, &shift)?;
            let nibble_mask =
                body.full_integer(&[config.block_k, config.block_n], 0x0f, DType::U8)?;
            let code = body.bit_and(&code, &nibble_mask)?;
            let values = decode_e2m1(body, &code)?;

            let scale_width_value = body.integer(scale_width, DType::I64)?;
            let scale_rows = body.multiply(&k_i64, &scale_width_value)?;
            let scale_rows = body.add(&scale_rows, &scale_expert_base)?;
            let representation_block = body.integer(REPRESENTATION_BLOCK, DType::I32)?;
            let scale_columns = body.divide(&columns, &representation_block)?;
            let scale_columns = body.cast(&scale_columns, DType::I64)?;
            let scale_columns = body.expand_dimension(&scale_columns, 0)?;
            let scale_offsets = body.add(&scale_rows, &scale_columns)?;
            let scale_addresses = body.add_pointer(&block_scales, &scale_offsets)?;
            let scales = body.load_masked(&scale_addresses, &weight_mask, &payload_zero)?;
            let scales = decode_e4m3fn(body, &scales)?;
            let global = body.load(&global_scale)?;
            let weights = body.multiply(&values, &scales)?;
            let weights = body.multiply(&weights, &global)?;
            let weights = body.cast(&weights, config.dtype)?;
            Ok(vec![body.dot(&input_tile, &weights, &carried[0])?])
        },
    )?;
    let mut accumulator = accumulated[0].clone();

    let expert_i64 = builder.cast(&address_expert, DType::I64)?;
    let output_size_i64 = builder.integer(config.output_size, DType::I64)?;
    let bias_base = builder.multiply(&expert_i64, &output_size_i64)?;
    let columns_i64 = builder.cast(&columns, DType::I64)?;
    let bias_offsets = builder.add(&bias_base, &columns_i64)?;
    let bias_addresses = builder.add_pointer(&bias, &bias_offsets)?;
    let bias_mask = builder.bit_and(&valid_columns, &valid_expert)?;
    let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
    let bias_values = builder.load_masked(&bias_addresses, &bias_mask, &bias_zero)?;
    let bias_values = builder.cast(&bias_values, DType::F32)?;
    let bias_values = builder.expand_dimension(&bias_values, 0)?;
    accumulator = builder.add(&accumulator, &bias_values)?;

    if let Some(routing_weights) = routing_weights {
        let routing_addresses = builder.add_pointer(&routing_weights, &address_assignment)?;
        let routing_zero = builder.full_float(&[config.block_m], 0.0, config.dtype)?;
        let routing = builder.load_masked(&routing_addresses, &compute_rows, &routing_zero)?;
        let routing = builder.cast(&routing, DType::F32)?;
        let routing = builder.expand_dimension(&routing, 1)?;
        accumulator = builder.multiply(&accumulator, &routing)?;
    }

    let assignments_i64 = builder.cast(&address_assignment, DType::I64)?;
    let output_rows = builder.multiply(&assignments_i64, &output_size_i64)?;
    let output_rows = builder.expand_dimension(&output_rows, 1)?;
    let columns_i64 = builder.expand_dimension(&columns_i64, 0)?;
    let output_offsets = builder.add(&output_rows, &columns_i64)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    let output_mask = builder.mask_2d(&valid_assignment, &valid_columns)?;
    let output_values = builder.cast(&accumulator, config.dtype)?;
    builder.store_masked(&output_addresses, &output_values, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}

fn clamped_residual_swiglu(
    builder: &mut Builder,
    gate: super::Value,
    up: super::Value,
    dtype: DType,
) -> Result<super::Value, Error> {
    let gate = builder.cast(&gate, DType::F32)?;
    let up = builder.cast(&up, DType::F32)?;
    let gate_limit = builder.full_float_like(&gate, 7.0)?;
    let gate = builder.minimum(&gate, &gate_limit)?;
    let up_lower = builder.full_float_like(&up, -7.0)?;
    let up_upper = builder.full_float_like(&up, 7.0)?;
    let up = builder.maximum(&up, &up_lower)?;
    let up = builder.minimum(&up, &up_upper)?;
    let exponent_scale = builder.full_float_like(&gate, -1.702 * std::f64::consts::LOG2_E)?;
    let exponent = builder.multiply(&gate, &exponent_scale)?;
    let exponent = builder.exp2(&exponent)?;
    let one = builder.full_float_like(&gate, 1.0)?;
    let denominator = builder.add(&one, &exponent)?;
    let sigmoid = builder.divide(&one, &denominator)?;
    let swish = builder.multiply(&gate, &sigmoid)?;
    let residual = builder.add(&up, &one)?;
    let activated = builder.multiply(&residual, &swish)?;
    builder.cast(&activated, dtype)
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

fn matrix_offsets(
    builder: &mut Builder,
    rows: &super::Value,
    columns: &super::Value,
) -> Result<super::Value, Error> {
    let rows = builder.expand_dimension(rows, 1)?;
    let columns = builder.expand_dimension(columns, 0)?;
    builder.add(&rows, &columns)
}

fn decode_e2m1(builder: &mut Builder, code: &super::Value) -> Result<super::Value, Error> {
    let shape = code_shape(code)?;
    let magnitude_mask = builder.full_integer(&shape, 0x07, DType::U8)?;
    let magnitude = builder.bit_and(code, &magnitude_mask)?;
    let one_i8 = builder.full_integer(&shape, 1, DType::U8)?;
    let mantissa = builder.bit_and(&magnitude, &one_i8)?;
    let one_shift = builder.full_integer(&shape, 1, DType::U8)?;
    let group = builder.shift_right_logical(&magnitude, &one_shift)?;
    let magnitude_f32 = builder.cast(&magnitude, DType::F32)?;
    let half = builder.full_float(&shape, 0.5, DType::F32)?;
    let subnormal = builder.multiply(&magnitude_f32, &half)?;
    let group = builder.cast(&group, DType::F32)?;
    let one = builder.full_float(&shape, 1.0, DType::F32)?;
    let exponent = builder.subtract(&group, &one)?;
    let power = builder.exp2(&exponent)?;
    let mantissa = builder.cast(&mantissa, DType::F32)?;
    let mantissa = builder.multiply(&mantissa, &half)?;
    let significand = builder.add(&one, &mantissa)?;
    let normal = builder.multiply(&power, &significand)?;
    let two = builder.full_integer(&shape, 2, DType::U8)?;
    let is_subnormal = builder.compare(Comparison::Less, &magnitude, &two)?;
    let magnitude = builder.select(&is_subnormal, &subnormal, &normal)?;
    let sign_boundary = builder.full_integer(&shape, 8, DType::U8)?;
    let positive = builder.compare(Comparison::Less, code, &sign_boundary)?;
    let negative = builder.negate(&magnitude)?;
    builder.select(&positive, &magnitude, &negative)
}

fn decode_e4m3fn(builder: &mut Builder, bits: &super::Value) -> Result<super::Value, Error> {
    let shape = code_shape(bits)?;
    let three = builder.full_integer(&shape, 3, DType::U8)?;
    let exponent = builder.shift_right_logical(bits, &three)?;
    let exponent_mask = builder.full_integer(&shape, 0x0f, DType::U8)?;
    let exponent = builder.bit_and(&exponent, &exponent_mask)?;
    let fraction_mask = builder.full_integer(&shape, 0x07, DType::U8)?;
    let fraction = builder.bit_and(bits, &fraction_mask)?;
    let fraction = builder.cast(&fraction, DType::F32)?;
    let subnormal_factor = builder.full_float(&shape, 2.0f64.powi(-9), DType::F32)?;
    let subnormal = builder.multiply(&fraction, &subnormal_factor)?;
    let eight = builder.full_float(&shape, 8.0, DType::F32)?;
    let fraction = builder.divide(&fraction, &eight)?;
    let one = builder.full_float(&shape, 1.0, DType::F32)?;
    let significand = builder.add(&one, &fraction)?;
    let exponent_f32 = builder.cast(&exponent, DType::F32)?;
    let seven = builder.full_float(&shape, 7.0, DType::F32)?;
    let exponent_f32 = builder.subtract(&exponent_f32, &seven)?;
    let power = builder.exp2(&exponent_f32)?;
    let normal = builder.multiply(&significand, &power)?;
    let zero = builder.full_integer(&shape, 0, DType::U8)?;
    let is_zero_exponent = builder.compare(Comparison::Equal, &exponent, &zero)?;
    builder.select(&is_zero_exponent, &subnormal, &normal)
}

fn code_shape(value: &super::Value) -> Result<Vec<i64>, Error> {
    match &value.value_type {
        super::ValueType::Tensor {
            shape,
            pointer_address_space: None,
            ..
        } => Ok(shape.clone()),
        _ => Err(Error::ExpectedTensor),
    }
}

const fn ceil_div(value: i64, divisor: i64) -> i64 {
    (value + divisor - 1) / divisor
}
