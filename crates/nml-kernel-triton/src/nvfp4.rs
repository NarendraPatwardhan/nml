//! Fused compact-weight NVFP4 linear construction for pre-Blackwell CUDA.
//!
//! The source representation is row-major `[N, K]`: payload bytes and E4M3FN
//! scales share the N row, while the K tile is decoded directly into the
//! tensor-core operand. Only the current tile exists at activation precision.

use super::{ArgumentKind, Builder, Comparison, DType, Error, Kernel, Reduction, Value};

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
pub struct NvFp4QkvConfig {
    pub dtype: DType,
    pub inputs: i64,
    pub query_outputs: i64,
    pub key_outputs: i64,
    pub value_outputs: i64,
    pub block_n: i64,
    pub block_k: i64,
    pub has_bias: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NvFp4GroupedRole {
    GateUpActivated,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NvFp4GroupedProjectionConfig {
    pub dtype: DType,
    pub tokens: i64,
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
            || self.block_n % REPRESENTATION_BLOCK != 0
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
            || !fits_i32(self.tokens)
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
            || self.block_k > if self.tokens == 1 { 256 } else { 128 }
            || self.block_k % REPRESENTATION_BLOCK != 0
            || match self.role {
                NvFp4GroupedRole::GateUpActivated => {
                    self.block_n % 8 != 0
                        || self.tokens.checked_mul(self.source_row_divisor)
                            != Some(self.assignments)
                }
                NvFp4GroupedRole::Down => {
                    self.block_n
                        % (if self.tokens == 1 {
                            8
                        } else {
                            REPRESENTATION_BLOCK
                        })
                        != 0
                        || self.source_row_divisor != 1
                }
            }
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

impl NvFp4QkvConfig {
    pub fn launch_grid(self) -> Result<[i32; 3], Error> {
        self.validate()?;
        let programs = [
            self.query_outputs,
            self.key_outputs,
            self.value_outputs,
        ]
        .into_iter()
        .try_fold(0_i64, |programs, outputs| {
            programs.checked_add(ceil_div(outputs, self.block_n))
        })
        .ok_or(Error::InvalidKernelSpec("NVFP4 QKV launch grid overflows"))?;
        Ok([
            i32::try_from(programs)
                .map_err(|_| Error::InvalidKernelSpec("NVFP4 QKV launch grid exceeds I32"))?,
            1,
            1,
        ])
    }

    fn validate(self) -> Result<(), Error> {
        if !matches!(self.dtype, DType::F16 | DType::Bf16)
            || [
                self.inputs,
                self.query_outputs,
                self.key_outputs,
                self.value_outputs,
            ]
            .into_iter()
            .any(|value| value <= 0)
            || [self.block_n, self.block_k]
                .into_iter()
                .any(|value| value <= 0 || !(value as u64).is_power_of_two())
            || self.block_k % REPRESENTATION_BLOCK != 0
        {
            return Err(Error::InvalidKernelSpec(
                "NVFP4 QKV requires F16/BF16, positive geometry, power-of-two tiles, and a K tile divisible by sixteen",
            ));
        }
        Ok(())
    }
}

pub fn build_nvfp4_linear(config: NvFp4LinearConfig) -> Result<Kernel, Error> {
    config.validate()?;
    if config.rows == 1 {
        return build_nvfp4_linear_gemv(config);
    }
    build_nvfp4_linear_matrix(config)
}

/// Builds one autoregressive compact QKV launch over three independent source
/// tensors. The representation remains unchanged; only the scheduling domain
/// is combined so the two 64-CTA K/V tails share waves with Q on SM8x.
pub fn build_nvfp4_qkv(config: NvFp4QkvConfig) -> Result<Kernel, Error> {
    config.validate()?;
    let mut builder = Builder::new("nvfp4_qkv_gemv")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let query_payload = pointer(&mut builder, "query_payload", DType::U8)?;
    let query_block_scales = pointer(&mut builder, "query_block_scales", DType::U8)?;
    let query_global_scale = pointer(&mut builder, "query_global_scale", DType::F32)?;
    let query_bias = config
        .has_bias
        .then(|| pointer(&mut builder, "query_bias", config.dtype))
        .transpose()?;
    let key_payload = pointer(&mut builder, "key_payload", DType::U8)?;
    let key_block_scales = pointer(&mut builder, "key_block_scales", DType::U8)?;
    let key_global_scale = pointer(&mut builder, "key_global_scale", DType::F32)?;
    let key_bias = config
        .has_bias
        .then(|| pointer(&mut builder, "key_bias", config.dtype))
        .transpose()?;
    let value_payload = pointer(&mut builder, "value_payload", DType::U8)?;
    let value_block_scales = pointer(&mut builder, "value_block_scales", DType::U8)?;
    let value_global_scale = pointer(&mut builder, "value_global_scale", DType::F32)?;
    let value_bias = config
        .has_bias
        .then(|| pointer(&mut builder, "value_bias", config.dtype))
        .transpose()?;
    let query_output = pointer(&mut builder, "query_output", config.dtype)?;
    let key_output = pointer(&mut builder, "key_output", config.dtype)?;
    let value_output = pointer(&mut builder, "value_output", config.dtype)?;

    let program = builder.program_id(0)?;
    let query_programs = ceil_div(config.query_outputs, config.block_n);
    let key_programs = ceil_div(config.key_outputs, config.block_n);
    let query_limit = builder.integer(query_programs, DType::I32)?;
    let key_limit = builder.integer(
        query_programs
            .checked_add(key_programs)
            .ok_or(Error::InvalidKernelSpec("NVFP4 QKV program range overflows"))?,
        DType::I32,
    )?;

    let query_program = builder.compare(Comparison::Less, &program, &query_limit)?;
    builder.if_only(&query_program, |body| {
        build_nvfp4_gemv_projection(
            body,
            qkv_projection(config, config.query_outputs),
            &program,
            &input,
            &query_payload,
            &query_block_scales,
            &query_global_scale,
            query_bias.as_ref(),
            &query_output,
        )
    })?;

    let after_query = builder.compare(Comparison::GreaterEqual, &program, &query_limit)?;
    let before_value = builder.compare(Comparison::Less, &program, &key_limit)?;
    let key_program = builder.bit_and(&after_query, &before_value)?;
    builder.if_only(&key_program, |body| {
        let local_program = body.subtract(&program, &query_limit)?;
        build_nvfp4_gemv_projection(
            body,
            qkv_projection(config, config.key_outputs),
            &local_program,
            &input,
            &key_payload,
            &key_block_scales,
            &key_global_scale,
            key_bias.as_ref(),
            &key_output,
        )
    })?;

    let value_program = builder.compare(Comparison::GreaterEqual, &program, &key_limit)?;
    builder.if_only(&value_program, |body| {
        let local_program = body.subtract(&program, &key_limit)?;
        build_nvfp4_gemv_projection(
            body,
            qkv_projection(config, config.value_outputs),
            &local_program,
            &input,
            &value_payload,
            &value_block_scales,
            &value_global_scale,
            value_bias.as_ref(),
            &value_output,
        )
    })?;
    builder.return_void()?;
    builder.finish()
}

const fn qkv_projection(config: NvFp4QkvConfig, outputs: i64) -> NvFp4LinearConfig {
    NvFp4LinearConfig {
        dtype: config.dtype,
        rows: 1,
        outputs,
        inputs: config.inputs,
        block_m: 1,
        block_n: config.block_n,
        block_k: config.block_k,
        has_bias: config.has_bias,
    }
}

fn build_nvfp4_linear_matrix(config: NvFp4LinearConfig) -> Result<Kernel, Error> {
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
    // The representation carries one tensor-wide scale. Load it once per
    // program rather than once per reduction tile.
    let global = builder.load(&global_scale)?;
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

            let scale_tile = config.block_k / REPRESENTATION_BLOCK;
            let scale_range = builder.range(
                0,
                i32::try_from(scale_tile)
                    .map_err(|_| Error::InvalidKernelSpec("NVFP4 scale tile exceeds I32"))?,
            )?;
            let representation_block = builder.integer(REPRESENTATION_BLOCK, DType::I32)?;
            let scale_start = builder.divide(&start, &representation_block)?;
            let scale_columns = builder.add(&scale_start, &scale_range)?;
            let scale_limit = builder.integer(scale_width, DType::I32)?;
            let valid_scale_columns =
                builder.compare(Comparison::Less, &scale_columns, &scale_limit)?;
            let scale_width = builder.integer(scale_width, DType::I32)?;
            let scale_rows = builder.multiply(&outputs, &scale_width)?;
            let scale_offsets = matrix_offsets(builder, &scale_columns, &scale_rows)?;
            let scale_pointer = builder.add_pointer(&block_scales, &scale_offsets)?;
            let scale_mask = builder.mask_2d(&valid_scale_columns, &valid_outputs)?;
            let scale_zero = builder.full_integer(&[scale_tile, config.block_n], 0, DType::U8)?;
            let scales = builder.load_masked(&scale_pointer, &scale_mask, &scale_zero)?;
            let scales = decode_e4m3fn(builder, &scales)?;
            let scales = builder.reshape(&scales, &[scale_tile, 1, config.block_n])?;
            let scales =
                builder.broadcast(&scales, &[scale_tile, REPRESENTATION_BLOCK, config.block_n])?;
            let scales = builder.reshape(&scales, &[config.block_k, config.block_n])?;
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

/// Builds the autoregressive `M = 1` compact projection.
///
/// Tensor-core matrix tiles are intentionally not used here: promoting one
/// row to a sixteen-row dot wastes fifteen rows of arithmetic and register
/// state. One program owns an output tile, walks packed K pairs, reuses each
/// decoded block scale for all eight packed bytes that it governs, and reduces
/// directly into F32 output accumulators.
fn build_nvfp4_linear_gemv(config: NvFp4LinearConfig) -> Result<Kernel, Error> {
    let mut builder = Builder::new("nvfp4_linear_gemv")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = config
        .has_bias
        .then(|| pointer(&mut builder, "bias", config.dtype))
        .transpose()?;
    let output = pointer(&mut builder, "output", config.dtype)?;

    let output_program = builder.program_id(0)?;
    build_nvfp4_gemv_projection(
        &mut builder,
        config,
        &output_program,
        &input,
        &payload,
        &block_scales,
        &global_scale,
        bias.as_ref(),
        &output,
    )?;
    builder.return_void()?;
    builder.finish()
}

#[allow(clippy::too_many_arguments)]
fn build_nvfp4_gemv_projection(
    builder: &mut Builder,
    config: NvFp4LinearConfig,
    output_program: &Value,
    input: &Value,
    payload: &Value,
    block_scales: &Value,
    global_scale: &Value,
    bias: Option<&Value>,
    output: &Value,
) -> Result<(), Error> {
    let block_n = i32::try_from(config.block_n)
        .map_err(|_| Error::InvalidKernelSpec("NVFP4 N tile exceeds I32"))?;
    let packed_k = config.block_k / 2;
    let packed_k_i32 = i32::try_from(packed_k)
        .map_err(|_| Error::InvalidKernelSpec("NVFP4 packed K tile exceeds I32"))?;
    let scale_k = config.block_k / REPRESENTATION_BLOCK;
    let scale_k_i32 = i32::try_from(scale_k)
        .map_err(|_| Error::InvalidKernelSpec("NVFP4 scale K tile exceeds I32"))?;
    let packed_width = ceil_div(config.inputs, 2);
    let scale_width = ceil_div(config.inputs, REPRESENTATION_BLOCK);

    let block_n_value = builder.integer(config.block_n, DType::I32)?;
    let output_base = builder.multiply(output_program, &block_n_value)?;
    let output_lanes = builder.range(0, block_n)?;
    let outputs = builder.add(&output_base, &output_lanes)?;
    let output_limit = builder.integer(config.outputs, DType::I32)?;
    let valid_outputs = builder.compare(Comparison::Less, &outputs, &output_limit)?;

    let lower = builder.integer(0, DType::I32)?;
    let upper = builder.integer(config.inputs, DType::I32)?;
    let step = builder.integer(config.block_k, DType::I32)?;
    let global = builder.load(global_scale)?;
    let accumulator = builder.full_float(&[config.block_n], 0.0, DType::F32)?;
    let result = builder.for_loop(
        &lower,
        &upper,
        &step,
        &[accumulator],
        |body, start, carried| {
            let pair_lanes = body.range(0, packed_k_i32)?;
            let two_i32 = body.integer(2, DType::I32)?;
            let pair_offsets = body.multiply(&pair_lanes, &two_i32)?;
            let even_k = body.add(&start, &pair_offsets)?;
            let one_i32 = body.integer(1, DType::I32)?;
            let odd_k = body.add(&even_k, &one_i32)?;
            let input_limit = body.integer(config.inputs, DType::I32)?;
            let valid_even = body.compare(Comparison::Less, &even_k, &input_limit)?;
            let valid_odd = body.compare(Comparison::Less, &odd_k, &input_limit)?;

            let input_zero = body.full_float(&[packed_k], 0.0, config.dtype)?;
            let even_addresses = body.add_pointer(input, &even_k)?;
            let odd_addresses = body.add_pointer(input, &odd_k)?;
            let even = body.load_masked(&even_addresses, &valid_even, &input_zero)?;
            let odd = body.load_masked(&odd_addresses, &valid_odd, &input_zero)?;
            let even = body.cast(&even, DType::F32)?;
            let odd = body.cast(&odd, DType::F32)?;
            let even = body.expand_dimension(&even, 0)?;
            let odd = body.expand_dimension(&odd, 0)?;

            let packed_start = body.divide(&start, &two_i32)?;
            let packed_columns = body.add(&packed_start, &pair_lanes)?;
            let packed_width_value = body.integer(packed_width, DType::I32)?;
            let payload_rows = body.multiply(&outputs, &packed_width_value)?;
            // Weight rows own contiguous K. Keeping K as the final TTIR
            // dimension lets lanes in one warp issue adjacent byte loads.
            let payload_offsets = matrix_offsets(body, &payload_rows, &packed_columns)?;
            let payload_addresses = body.add_pointer(payload, &payload_offsets)?;
            let payload_mask = body.mask_2d(&valid_outputs, &valid_even)?;
            let payload_zero = body.full_integer(&[config.block_n, packed_k], 0, DType::U8)?;
            let packed =
                body.load_masked_streaming(&payload_addresses, &payload_mask, &payload_zero)?;
            let nibble_mask = body.full_integer(&[config.block_n, packed_k], 0x0f, DType::U8)?;
            let low_code = body.bit_and(&packed, &nibble_mask)?;
            let four = body.full_integer(&[config.block_n, packed_k], 4, DType::U8)?;
            let high_code = body.shift_right_logical(&packed, &four)?;
            let low = decode_e2m1(body, &low_code)?;
            let high = decode_e2m1(body, &high_code)?;

            let scale_lanes = body.range(0, scale_k_i32)?;
            let representation_block = body.integer(REPRESENTATION_BLOCK, DType::I32)?;
            let scale_start = body.divide(&start, &representation_block)?;
            let scale_columns = body.add(&scale_start, &scale_lanes)?;
            let scale_limit = body.integer(scale_width, DType::I32)?;
            let valid_scales = body.compare(Comparison::Less, &scale_columns, &scale_limit)?;
            let scale_width_value = body.integer(scale_width, DType::I32)?;
            let scale_rows = body.multiply(&outputs, &scale_width_value)?;
            let scale_offsets = matrix_offsets(body, &scale_rows, &scale_columns)?;
            let scale_addresses = body.add_pointer(block_scales, &scale_offsets)?;
            let scale_mask = body.mask_2d(&valid_outputs, &valid_scales)?;
            let scale_zero = body.full_integer(&[config.block_n, scale_k], 0, DType::U8)?;
            let scales = body.load_masked_streaming(&scale_addresses, &scale_mask, &scale_zero)?;
            let scales = decode_e4m3fn(body, &scales)?;
            let scales = body.reshape(&scales, &[config.block_n, scale_k, 1])?;
            let scales = body.broadcast(&scales, &[config.block_n, scale_k, 8])?;
            let scales = body.reshape(&scales, &[config.block_n, packed_k])?;
            let low = body.multiply(&low, &scales)?;
            let high = body.multiply(&high, &scales)?;
            let low = body.multiply(&low, &even)?;
            let high = body.multiply(&high, &odd)?;
            let products = body.add(&low, &high)?;
            let partial = body.reduce(Reduction::Sum, &products, 1)?;
            Ok(vec![body.add(&carried[0], &partial)?])
        },
    )?[0]
        .clone();
    // The tensor-wide scale is invariant across K. Apply it once after the
    // complete F32 reduction instead of to both decoded halves of every
    // packed weight.
    let result = builder.multiply(&result, &global)?;

    let result = if let Some(bias) = bias {
        let bias_addresses = builder.add_pointer(bias, &outputs)?;
        let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
        let bias = builder.load_masked(&bias_addresses, &valid_outputs, &bias_zero)?;
        let bias = builder.cast(&bias, DType::F32)?;
        builder.add(&result, &bias)?
    } else {
        result
    };
    let result = builder.cast(&result, config.dtype)?;
    let output_addresses = builder.add_pointer(output, &outputs)?;
    builder.store_masked(&output_addresses, &result, &valid_outputs)?;
    Ok(())
}

pub fn build_nvfp4_embedding(config: NvFp4EmbeddingConfig) -> Result<Kernel, Error> {
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
    let scale_tile = config.block_n / REPRESENTATION_BLOCK;
    let scale_lanes = builder.range(
        0,
        i32::try_from(scale_tile)
            .map_err(|_| Error::InvalidKernelSpec("embedding scale tile exceeds I32"))?,
    )?;
    let scale_base = builder.divide(&column_base, &representation_block)?;
    let scale_columns = builder.add(&scale_base, &scale_lanes)?;
    let scale_limit = builder.integer(scale_width, DType::I32)?;
    let valid_scale_columns = builder.compare(Comparison::Less, &scale_columns, &scale_limit)?;
    let scale_columns = builder.cast(&scale_columns, DType::I64)?;
    let scale_columns = builder.expand_dimension(&scale_columns, 0)?;
    let scale_offsets = builder.add(&scale_rows, &scale_columns)?;
    let scale_addresses = builder.add_pointer(&block_scales, &scale_offsets)?;
    let scale_mask = builder.mask_2d(&valid_rows, &valid_scale_columns)?;
    let scale_zero = builder.full_integer(&[config.block_m, scale_tile], 0, DType::U8)?;
    let scales = builder.load_masked(&scale_addresses, &scale_mask, &scale_zero)?;
    let scales = decode_e4m3fn(&mut builder, &scales)?;
    let scales = builder.reshape(&scales, &[config.block_m, scale_tile, 1])?;
    let scales = builder.broadcast(&scales, &[config.block_m, scale_tile, REPRESENTATION_BLOCK])?;
    let scales = builder.reshape(&scales, &[config.block_m, config.block_n])?;
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

/// Builds one compact, output-major routed expert projection.
///
/// Both prefill and decode observe the same expert boundary: gate/up performs
/// its paired contractions, adds paired biases, applies clamped residual
/// SwiGLU exactly once, and writes `[assignments, intermediate]`. Down consumes
/// that activated tensor and applies only its bias and routing epilogue.
pub fn build_nvfp4_grouped_projection(
    config: NvFp4GroupedProjectionConfig,
) -> Result<Kernel, Error> {
    let config = config.validate()?;
    match (config.tokens == 1, config.role) {
        (true, NvFp4GroupedRole::GateUpActivated) => build_grouped_gate_up_gemv(config),
        (true, NvFp4GroupedRole::Down) => build_grouped_down_gemv(config),
        (false, NvFp4GroupedRole::GateUpActivated) => build_grouped_gate_up_matrix(config),
        (false, NvFp4GroupedRole::Down) => build_grouped_down_matrix(config),
    }
}

struct GroupedMatrixSchedule {
    address_assignment: Value,
    address_expert: Value,
    execute_block: Value,
    store_rows: Value,
    compute_rows: Value,
    columns: Value,
    valid_columns: Value,
    source_bases: Value,
}

fn grouped_matrix_schedule(
    builder: &mut Builder,
    config: NvFp4GroupedProjectionConfig,
    sorted_assignments: &Value,
    block_experts: &Value,
    active_blocks_pointer: &Value,
    expert_offset_pointer: &Value,
) -> Result<GroupedMatrixSchedule, Error> {
    let block_index = builder.program_id(0)?;
    let output_block = builder.program_id(1)?;
    let active_blocks = builder.load(active_blocks_pointer)?;
    let active_block = builder.compare(Comparison::Less, &block_index, &active_blocks)?;

    let row_lanes = builder.range(0, config.block_m as i32)?;
    let block_m = builder.integer(config.block_m, DType::I32)?;
    let row_start = builder.multiply(&block_index, &block_m)?;
    let schedule_positions = builder.add(&row_start, &row_lanes)?;
    let assignment_addresses = builder.add_pointer(sorted_assignments, &schedule_positions)?;
    let assignments = builder.load(&assignment_addresses)?;
    let zero_rows = builder.full_integer(&[config.block_m], 0, DType::I32)?;
    let assignment_limit =
        builder.full_integer(&[config.block_m], config.assignments, DType::I32)?;
    let nonnegative = builder.compare(Comparison::GreaterEqual, &assignments, &zero_rows)?;
    let below_limit = builder.compare(Comparison::Less, &assignments, &assignment_limit)?;
    let valid_assignment = builder.bit_and(&nonnegative, &below_limit)?;
    let last_assignment =
        builder.full_integer(&[config.block_m], config.assignments - 1, DType::I32)?;
    let address_assignment = builder.maximum(&assignments, &zero_rows)?;
    let address_assignment = builder.minimum(&address_assignment, &last_assignment)?;

    let expert_address = builder.add_pointer(block_experts, &block_index)?;
    let global_expert = builder.load(&expert_address)?;
    let expert_offset = builder.load(expert_offset_pointer)?;
    let expert = builder.subtract(&global_expert, &expert_offset)?;
    let zero = builder.integer(0, DType::I32)?;
    let local_experts = builder.integer(config.local_experts, DType::I32)?;
    let after_first = builder.compare(Comparison::GreaterEqual, &expert, &zero)?;
    let before_last = builder.compare(Comparison::Less, &expert, &local_experts)?;
    let valid_expert = builder.bit_and(&after_first, &before_last)?;
    let execute_block = builder.bit_and(&active_block, &valid_expert)?;
    let store_rows = builder.bit_and(&valid_assignment, &active_block)?;
    let compute_rows = builder.bit_and(&store_rows, &valid_expert)?;
    let last_expert = builder.integer(config.local_experts - 1, DType::I32)?;
    let address_expert = builder.maximum(&expert, &zero)?;
    let address_expert = builder.minimum(&address_expert, &last_expert)?;

    let block_n = builder.integer(config.block_n, DType::I32)?;
    let column_start = builder.multiply(&output_block, &block_n)?;
    let column_lanes = builder.range(0, config.block_n as i32)?;
    let columns = builder.add(&column_start, &column_lanes)?;
    let output_size = builder.integer(config.output_size, DType::I32)?;
    let valid_columns = builder.compare(Comparison::Less, &columns, &output_size)?;

    let divisor = builder.integer(config.source_row_divisor, DType::I32)?;
    let source_rows = builder.divide(&address_assignment, &divisor)?;
    let source_rows = builder.cast(&source_rows, DType::I64)?;
    let input_stride = builder.integer(config.input_size, DType::I64)?;
    let source_bases = builder.multiply(&source_rows, &input_stride)?;
    let source_bases = builder.expand_dimension(&source_bases, 1)?;

    Ok(GroupedMatrixSchedule {
        address_assignment,
        address_expert,
        execute_block,
        store_rows,
        compute_rows,
        columns,
        valid_columns,
        source_bases,
    })
}

struct GroupedDecodeSchedule {
    address_assignment: Value,
    address_expert: Value,
    execute: Value,
    store: Value,
    source_base: Value,
    output_block: Value,
}

fn grouped_decode_schedule(
    builder: &mut Builder,
    config: NvFp4GroupedProjectionConfig,
    sorted_assignments: &Value,
    block_experts: &Value,
    active_blocks_pointer: &Value,
    expert_offset_pointer: &Value,
) -> Result<GroupedDecodeSchedule, Error> {
    let block_index = builder.program_id(0)?;
    let output_block = builder.program_id(1)?;
    let active_blocks = builder.load(active_blocks_pointer)?;
    let active_block = builder.compare(Comparison::Less, &block_index, &active_blocks)?;

    // With one token, top-k routing produces at most one assignment for each
    // selected expert. StableHLO still authors padded block-sized schedules;
    // the compact GEMV consumes only the first live slot of each active block.
    let block_m = builder.integer(config.block_m, DType::I32)?;
    let schedule_position = builder.multiply(&block_index, &block_m)?;
    let assignment_address = builder.add_pointer(sorted_assignments, &schedule_position)?;
    let assignment = builder.load(&assignment_address)?;
    let zero = builder.integer(0, DType::I32)?;
    let assignment_limit = builder.integer(config.assignments, DType::I32)?;
    let nonnegative = builder.compare(Comparison::GreaterEqual, &assignment, &zero)?;
    let below_limit = builder.compare(Comparison::Less, &assignment, &assignment_limit)?;
    let valid_assignment = builder.bit_and(&nonnegative, &below_limit)?;
    let last_assignment = builder.integer(config.assignments - 1, DType::I32)?;
    let address_assignment = builder.maximum(&assignment, &zero)?;
    let address_assignment = builder.minimum(&address_assignment, &last_assignment)?;

    let expert_address = builder.add_pointer(block_experts, &block_index)?;
    let global_expert = builder.load(&expert_address)?;
    let expert_offset = builder.load(expert_offset_pointer)?;
    let expert = builder.subtract(&global_expert, &expert_offset)?;
    let local_experts = builder.integer(config.local_experts, DType::I32)?;
    let after_first = builder.compare(Comparison::GreaterEqual, &expert, &zero)?;
    let before_last = builder.compare(Comparison::Less, &expert, &local_experts)?;
    let valid_expert = builder.bit_and(&after_first, &before_last)?;
    let store = builder.bit_and(&active_block, &valid_assignment)?;
    let execute = builder.bit_and(&store, &valid_expert)?;
    let last_expert = builder.integer(config.local_experts - 1, DType::I32)?;
    let address_expert = builder.maximum(&expert, &zero)?;
    let address_expert = builder.minimum(&address_expert, &last_expert)?;

    let source_divisor = builder.integer(config.source_row_divisor, DType::I32)?;
    let source_row = builder.divide(&address_assignment, &source_divisor)?;
    let source_row = builder.cast(&source_row, DType::I64)?;
    let input_stride = builder.integer(config.input_size, DType::I64)?;
    let source_base = builder.multiply(&source_row, &input_stride)?;

    Ok(GroupedDecodeSchedule {
        address_assignment,
        address_expert,
        execute,
        store,
        source_base,
        output_block,
    })
}

fn build_grouped_gate_up_matrix(config: NvFp4GroupedProjectionConfig) -> Result<Kernel, Error> {
    let logical_rows = config
        .output_size
        .checked_mul(2)
        .ok_or(Error::InvalidKernelSpec("gate/up logical width overflows"))?;
    let packed_width = ceil_div(config.input_size, 2);
    let scale_width = ceil_div(config.input_size, REPRESENTATION_BLOCK);
    let scale_k = config.block_k / REPRESENTATION_BLOCK;
    let scale_k_i32 = i32::try_from(scale_k)
        .map_err(|_| Error::InvalidKernelSpec("gate/up scale K tile exceeds I32"))?;

    let mut builder = Builder::new("nvfp4_grouped_gate_up")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let sorted_assignments = pointer(&mut builder, "sorted_assignments", DType::I32)?;
    let block_experts = pointer(&mut builder, "block_experts", DType::I32)?;
    let active_blocks = pointer(&mut builder, "active_blocks", DType::I32)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = pointer(&mut builder, "bias", config.dtype)?;
    let expert_offset = pointer(&mut builder, "expert_offset", DType::I32)?;
    let output = pointer(&mut builder, "output", config.dtype)?;
    let schedule = grouped_matrix_schedule(
        &mut builder,
        config,
        &sorted_assignments,
        &block_experts,
        &active_blocks,
        &expert_offset,
    )?;

    let payload_stride = logical_rows
        .checked_mul(packed_width)
        .ok_or(Error::InvalidKernelSpec("gate/up payload stride overflows"))?;
    let scale_stride = logical_rows
        .checked_mul(scale_width)
        .ok_or(Error::InvalidKernelSpec("gate/up scale stride overflows"))?;
    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let payload_stride = builder.integer(payload_stride, DType::I64)?;
    let scale_stride = builder.integer(scale_stride, DType::I64)?;
    let payload_expert_base = builder.multiply(&expert_i64, &payload_stride)?;
    let scale_expert_base = builder.multiply(&expert_i64, &scale_stride)?;

    let accumulators = builder.if_then_else(
        &schedule.execute_block,
        |body| {
            let global = body.load(&global_scale)?;
            let gate_accumulator =
                body.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?;
            let up_accumulator =
                body.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?;
            let lower = body.integer(0, DType::I32)?;
            let upper = body.integer(config.input_size, DType::I32)?;
            let step = body.integer(config.block_k, DType::I32)?;
            let k_lanes = body.range(0, config.block_k as i32)?;
            body.for_loop(
                &lower,
                &upper,
                &step,
                &[gate_accumulator, up_accumulator],
                |loop_body, start, carried| {
                    let k = loop_body.add(&start, &k_lanes)?;
                    let input_limit = loop_body.integer(config.input_size, DType::I32)?;
                    let valid_k = loop_body.compare(Comparison::Less, &k, &input_limit)?;
                    let input_mask = loop_body.mask_2d(&schedule.compute_rows, &valid_k)?;
                    let input_zero = loop_body.full_float(
                        &[config.block_m, config.block_k],
                        0.0,
                        config.dtype,
                    )?;
                    let k_i64 = loop_body.cast(&k, DType::I64)?;
                    let input_columns = loop_body.expand_dimension(&k_i64, 0)?;
                    let input_offsets = loop_body.add(&schedule.source_bases, &input_columns)?;
                    let input_addresses = loop_body.add_pointer(&input, &input_offsets)?;
                    let input_tile =
                        loop_body.load_masked(&input_addresses, &input_mask, &input_zero)?;

                    let two_i32 = loop_body.integer(2, DType::I32)?;
                    let packed_k = loop_body.divide(&k, &two_i32)?;
                    let packed_k_i64 = loop_body.cast(&packed_k, DType::I64)?;
                    let columns_i64 = loop_body.cast(&schedule.columns, DType::I64)?;
                    let two_i64 = loop_body.integer(2, DType::I64)?;
                    let gate_rows = loop_body.multiply(&columns_i64, &two_i64)?;
                    let one_i64 = loop_body.integer(1, DType::I64)?;
                    let up_rows = loop_body.add(&gate_rows, &one_i64)?;
                    let packed_width_value = loop_body.integer(packed_width, DType::I64)?;
                    let gate_row_offsets = loop_body.multiply(&gate_rows, &packed_width_value)?;
                    let up_row_offsets = loop_body.multiply(&up_rows, &packed_width_value)?;
                    let gate_bases = loop_body.add(&payload_expert_base, &gate_row_offsets)?;
                    let up_bases = loop_body.add(&payload_expert_base, &up_row_offsets)?;
                    let gate_offsets = matrix_offsets(loop_body, &packed_k_i64, &gate_bases)?;
                    let up_offsets = matrix_offsets(loop_body, &packed_k_i64, &up_bases)?;
                    let gate_addresses = loop_body.add_pointer(&payload, &gate_offsets)?;
                    let up_addresses = loop_body.add_pointer(&payload, &up_offsets)?;
                    let weight_mask = loop_body.mask_2d(&valid_k, &schedule.valid_columns)?;
                    let payload_zero =
                        loop_body.full_integer(&[config.block_k, config.block_n], 0, DType::U8)?;
                    let gate_packed =
                        loop_body.load_masked(&gate_addresses, &weight_mask, &payload_zero)?;
                    let up_packed =
                        loop_body.load_masked(&up_addresses, &weight_mask, &payload_zero)?;
                    let parity = loop_body.remainder(&k, &two_i32)?;
                    let four_i32 = loop_body.integer(4, DType::I32)?;
                    let shift = loop_body.multiply(&parity, &four_i32)?;
                    let shift = loop_body.cast(&shift, DType::U8)?;
                    let shift = loop_body.expand_dimension(&shift, 1)?;
                    let gate_code = loop_body.shift_right_logical(&gate_packed, &shift)?;
                    let up_code = loop_body.shift_right_logical(&up_packed, &shift)?;
                    let nibble = loop_body.full_integer(
                        &[config.block_k, config.block_n],
                        0x0f,
                        DType::U8,
                    )?;
                    let gate_code = loop_body.bit_and(&gate_code, &nibble)?;
                    let up_code = loop_body.bit_and(&up_code, &nibble)?;
                    let gate = decode_e2m1(loop_body, &gate_code)?;
                    let up = decode_e2m1(loop_body, &up_code)?;

                    let scale_lanes = loop_body.range(0, scale_k_i32)?;
                    let representation_block =
                        loop_body.integer(REPRESENTATION_BLOCK, DType::I32)?;
                    let scale_start = loop_body.divide(&start, &representation_block)?;
                    let scale_columns = loop_body.add(&scale_start, &scale_lanes)?;
                    let scale_limit = loop_body.integer(scale_width, DType::I32)?;
                    let valid_scales =
                        loop_body.compare(Comparison::Less, &scale_columns, &scale_limit)?;
                    let scale_columns_i64 = loop_body.cast(&scale_columns, DType::I64)?;
                    let scale_width_value = loop_body.integer(scale_width, DType::I64)?;
                    let gate_scale_row_offsets =
                        loop_body.multiply(&gate_rows, &scale_width_value)?;
                    let up_scale_row_offsets = loop_body.multiply(&up_rows, &scale_width_value)?;
                    let gate_scale_bases =
                        loop_body.add(&scale_expert_base, &gate_scale_row_offsets)?;
                    let up_scale_bases =
                        loop_body.add(&scale_expert_base, &up_scale_row_offsets)?;
                    let gate_scale_offsets =
                        matrix_offsets(loop_body, &scale_columns_i64, &gate_scale_bases)?;
                    let up_scale_offsets =
                        matrix_offsets(loop_body, &scale_columns_i64, &up_scale_bases)?;
                    let gate_scale_addresses =
                        loop_body.add_pointer(&block_scales, &gate_scale_offsets)?;
                    let up_scale_addresses =
                        loop_body.add_pointer(&block_scales, &up_scale_offsets)?;
                    let scale_mask = loop_body.mask_2d(&valid_scales, &schedule.valid_columns)?;
                    let scale_zero =
                        loop_body.full_integer(&[scale_k, config.block_n], 0, DType::U8)?;
                    let gate_scales =
                        loop_body.load_masked(&gate_scale_addresses, &scale_mask, &scale_zero)?;
                    let up_scales =
                        loop_body.load_masked(&up_scale_addresses, &scale_mask, &scale_zero)?;
                    let expand_scales = |builder: &mut Builder,
                                         scales: Value|
                     -> Result<Value, Error> {
                        let scales = decode_e4m3fn(builder, &scales)?;
                        let scales = builder.reshape(&scales, &[scale_k, 1, config.block_n])?;
                        let scales = builder
                            .broadcast(&scales, &[scale_k, REPRESENTATION_BLOCK, config.block_n])?;
                        builder.reshape(&scales, &[config.block_k, config.block_n])
                    };
                    let gate_scales = expand_scales(loop_body, gate_scales)?;
                    let up_scales = expand_scales(loop_body, up_scales)?;
                    let gate = loop_body.multiply(&gate, &gate_scales)?;
                    let up = loop_body.multiply(&up, &up_scales)?;
                    let gate = loop_body.multiply(&gate, &global)?;
                    let up = loop_body.multiply(&up, &global)?;
                    let gate = loop_body.cast(&gate, config.dtype)?;
                    let up = loop_body.cast(&up, config.dtype)?;
                    let gate = loop_body.dot(&input_tile, &gate, &carried[0])?;
                    let up = loop_body.dot(&input_tile, &up, &carried[1])?;
                    Ok(vec![gate, up])
                },
            )
        },
        |body| {
            Ok(vec![
                body.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?,
                body.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?,
            ])
        },
    )?;
    let mut gate = accumulators[0].clone();
    let mut up = accumulators[1].clone();

    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let logical_rows_value = builder.integer(logical_rows, DType::I64)?;
    let bias_base = builder.multiply(&expert_i64, &logical_rows_value)?;
    let columns_i64 = builder.cast(&schedule.columns, DType::I64)?;
    let two_i64 = builder.integer(2, DType::I64)?;
    let gate_columns = builder.multiply(&columns_i64, &two_i64)?;
    let one_i64 = builder.integer(1, DType::I64)?;
    let up_columns = builder.add(&gate_columns, &one_i64)?;
    let gate_offsets = builder.add(&bias_base, &gate_columns)?;
    let up_offsets = builder.add(&bias_base, &up_columns)?;
    let gate_addresses = builder.add_pointer(&bias, &gate_offsets)?;
    let up_addresses = builder.add_pointer(&bias, &up_offsets)?;
    let bias_mask = builder.bit_and(&schedule.valid_columns, &schedule.execute_block)?;
    let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
    let gate_bias = builder.load_masked(&gate_addresses, &bias_mask, &bias_zero)?;
    let up_bias = builder.load_masked(&up_addresses, &bias_mask, &bias_zero)?;
    let gate_bias = builder.cast(&gate_bias, DType::F32)?;
    let up_bias = builder.cast(&up_bias, DType::F32)?;
    let gate_bias = builder.expand_dimension(&gate_bias, 0)?;
    let up_bias = builder.expand_dimension(&up_bias, 0)?;
    gate = builder.add(&gate, &gate_bias)?;
    up = builder.add(&up, &up_bias)?;
    let activated = clamped_residual_swiglu(&mut builder, gate, up)?;

    let assignments_i64 = builder.cast(&schedule.address_assignment, DType::I64)?;
    let output_width = builder.integer(config.output_size, DType::I64)?;
    let output_rows = builder.multiply(&assignments_i64, &output_width)?;
    let output_offsets = matrix_offsets(&mut builder, &output_rows, &columns_i64)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    let output_mask = builder.mask_2d(&schedule.store_rows, &schedule.valid_columns)?;
    let activated = builder.cast(&activated, config.dtype)?;
    builder.store_masked(&output_addresses, &activated, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}

fn build_grouped_down_matrix(config: NvFp4GroupedProjectionConfig) -> Result<Kernel, Error> {
    let packed_width = ceil_div(config.input_size, 2);
    let scale_width = ceil_div(config.input_size, REPRESENTATION_BLOCK);
    let scale_k = config.block_k / REPRESENTATION_BLOCK;
    let scale_k_i32 = i32::try_from(scale_k)
        .map_err(|_| Error::InvalidKernelSpec("down scale K tile exceeds I32"))?;

    let mut builder = Builder::new("nvfp4_grouped_down")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let sorted_assignments = pointer(&mut builder, "sorted_assignments", DType::I32)?;
    let block_experts = pointer(&mut builder, "block_experts", DType::I32)?;
    let active_blocks = pointer(&mut builder, "active_blocks", DType::I32)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = pointer(&mut builder, "bias", config.dtype)?;
    let expert_offset = pointer(&mut builder, "expert_offset", DType::I32)?;
    let routing_weights = pointer(&mut builder, "routing_weights", config.dtype)?;
    let output = pointer(&mut builder, "output", config.dtype)?;
    let schedule = grouped_matrix_schedule(
        &mut builder,
        config,
        &sorted_assignments,
        &block_experts,
        &active_blocks,
        &expert_offset,
    )?;

    let payload_stride = config
        .output_size
        .checked_mul(packed_width)
        .ok_or(Error::InvalidKernelSpec("down payload stride overflows"))?;
    let scale_stride = config
        .output_size
        .checked_mul(scale_width)
        .ok_or(Error::InvalidKernelSpec("down scale stride overflows"))?;
    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let payload_stride = builder.integer(payload_stride, DType::I64)?;
    let scale_stride = builder.integer(scale_stride, DType::I64)?;
    let payload_expert_base = builder.multiply(&expert_i64, &payload_stride)?;
    let scale_expert_base = builder.multiply(&expert_i64, &scale_stride)?;

    let mut accumulator = builder
        .if_then_else(
            &schedule.execute_block,
            |body| {
                let global = body.load(&global_scale)?;
                let accumulator =
                    body.full_float(&[config.block_m, config.block_n], 0.0, DType::F32)?;
                let lower = body.integer(0, DType::I32)?;
                let upper = body.integer(config.input_size, DType::I32)?;
                let step = body.integer(config.block_k, DType::I32)?;
                let k_lanes = body.range(0, config.block_k as i32)?;
                let accumulated = body.for_loop(
                    &lower,
                    &upper,
                    &step,
                    &[accumulator],
                    |loop_body, start, carried| {
                        let k = loop_body.add(&start, &k_lanes)?;
                        let input_limit = loop_body.integer(config.input_size, DType::I32)?;
                        let valid_k = loop_body.compare(Comparison::Less, &k, &input_limit)?;
                        let input_mask = loop_body.mask_2d(&schedule.compute_rows, &valid_k)?;
                        let input_zero = loop_body.full_float(
                            &[config.block_m, config.block_k],
                            0.0,
                            config.dtype,
                        )?;
                        let k_i64 = loop_body.cast(&k, DType::I64)?;
                        let input_columns = loop_body.expand_dimension(&k_i64, 0)?;
                        let input_offsets =
                            loop_body.add(&schedule.source_bases, &input_columns)?;
                        let input_addresses = loop_body.add_pointer(&input, &input_offsets)?;
                        let input_tile =
                            loop_body.load_masked(&input_addresses, &input_mask, &input_zero)?;

                        let two_i32 = loop_body.integer(2, DType::I32)?;
                        let packed_k = loop_body.divide(&k, &two_i32)?;
                        let packed_k_i64 = loop_body.cast(&packed_k, DType::I64)?;
                        let columns_i64 = loop_body.cast(&schedule.columns, DType::I64)?;
                        let packed_width_value = loop_body.integer(packed_width, DType::I64)?;
                        let row_offsets = loop_body.multiply(&columns_i64, &packed_width_value)?;
                        let row_bases = loop_body.add(&payload_expert_base, &row_offsets)?;
                        let payload_offsets = matrix_offsets(loop_body, &packed_k_i64, &row_bases)?;
                        let payload_addresses =
                            loop_body.add_pointer(&payload, &payload_offsets)?;
                        let weight_mask = loop_body.mask_2d(&valid_k, &schedule.valid_columns)?;
                        let payload_zero = loop_body.full_integer(
                            &[config.block_k, config.block_n],
                            0,
                            DType::U8,
                        )?;
                        let packed = loop_body.load_masked(
                            &payload_addresses,
                            &weight_mask,
                            &payload_zero,
                        )?;
                        let parity = loop_body.remainder(&k, &two_i32)?;
                        let four_i32 = loop_body.integer(4, DType::I32)?;
                        let shift = loop_body.multiply(&parity, &four_i32)?;
                        let shift = loop_body.cast(&shift, DType::U8)?;
                        let shift = loop_body.expand_dimension(&shift, 1)?;
                        let code = loop_body.shift_right_logical(&packed, &shift)?;
                        let nibble = loop_body.full_integer(
                            &[config.block_k, config.block_n],
                            0x0f,
                            DType::U8,
                        )?;
                        let code = loop_body.bit_and(&code, &nibble)?;
                        let values = decode_e2m1(loop_body, &code)?;

                        let scale_lanes = loop_body.range(0, scale_k_i32)?;
                        let representation_block =
                            loop_body.integer(REPRESENTATION_BLOCK, DType::I32)?;
                        let scale_start = loop_body.divide(&start, &representation_block)?;
                        let scale_columns = loop_body.add(&scale_start, &scale_lanes)?;
                        let scale_limit = loop_body.integer(scale_width, DType::I32)?;
                        let valid_scales =
                            loop_body.compare(Comparison::Less, &scale_columns, &scale_limit)?;
                        let scale_columns_i64 = loop_body.cast(&scale_columns, DType::I64)?;
                        let scale_width_value = loop_body.integer(scale_width, DType::I64)?;
                        let scale_row_offsets =
                            loop_body.multiply(&columns_i64, &scale_width_value)?;
                        let scale_bases = loop_body.add(&scale_expert_base, &scale_row_offsets)?;
                        let scale_offsets =
                            matrix_offsets(loop_body, &scale_columns_i64, &scale_bases)?;
                        let scale_addresses =
                            loop_body.add_pointer(&block_scales, &scale_offsets)?;
                        let scale_mask =
                            loop_body.mask_2d(&valid_scales, &schedule.valid_columns)?;
                        let scale_zero =
                            loop_body.full_integer(&[scale_k, config.block_n], 0, DType::U8)?;
                        let scales =
                            loop_body.load_masked(&scale_addresses, &scale_mask, &scale_zero)?;
                        let scales = decode_e4m3fn(loop_body, &scales)?;
                        let scales = loop_body.reshape(&scales, &[scale_k, 1, config.block_n])?;
                        let scales = loop_body
                            .broadcast(&scales, &[scale_k, REPRESENTATION_BLOCK, config.block_n])?;
                        let scales =
                            loop_body.reshape(&scales, &[config.block_k, config.block_n])?;
                        let weights = loop_body.multiply(&values, &scales)?;
                        let weights = loop_body.multiply(&weights, &global)?;
                        let weights = loop_body.cast(&weights, config.dtype)?;
                        let result = loop_body.dot(&input_tile, &weights, &carried[0])?;
                        Ok(vec![result])
                    },
                )?;
                Ok(vec![accumulated[0].clone()])
            },
            |body| {
                Ok(vec![body.full_float(
                    &[config.block_m, config.block_n],
                    0.0,
                    DType::F32,
                )?])
            },
        )?
        .remove(0);

    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let output_width = builder.integer(config.output_size, DType::I64)?;
    let bias_base = builder.multiply(&expert_i64, &output_width)?;
    let columns_i64 = builder.cast(&schedule.columns, DType::I64)?;
    let bias_offsets = builder.add(&bias_base, &columns_i64)?;
    let bias_addresses = builder.add_pointer(&bias, &bias_offsets)?;
    let bias_mask = builder.bit_and(&schedule.valid_columns, &schedule.execute_block)?;
    let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
    let bias_values = builder.load_masked(&bias_addresses, &bias_mask, &bias_zero)?;
    let bias_values = builder.cast(&bias_values, DType::F32)?;
    let bias_values = builder.expand_dimension(&bias_values, 0)?;
    accumulator = builder.add(&accumulator, &bias_values)?;

    let routing_addresses = builder.add_pointer(&routing_weights, &schedule.address_assignment)?;
    let routing_zero = builder.full_float(&[config.block_m], 0.0, config.dtype)?;
    let routing = builder.load_masked(&routing_addresses, &schedule.compute_rows, &routing_zero)?;
    let routing = builder.cast(&routing, DType::F32)?;
    let routing = builder.expand_dimension(&routing, 1)?;
    accumulator = builder.multiply(&accumulator, &routing)?;

    let assignments_i64 = builder.cast(&schedule.address_assignment, DType::I64)?;
    let output_rows = builder.multiply(&assignments_i64, &output_width)?;
    let output_offsets = matrix_offsets(&mut builder, &output_rows, &columns_i64)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    let output_mask = builder.mask_2d(&schedule.store_rows, &schedule.valid_columns)?;
    let output_values = builder.cast(&accumulator, config.dtype)?;
    builder.store_masked(&output_addresses, &output_values, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}
fn build_grouped_gate_up_gemv(config: NvFp4GroupedProjectionConfig) -> Result<Kernel, Error> {
    let packed_k = config.block_k / 2;
    let packed_k_i32 = i32::try_from(packed_k)
        .map_err(|_| Error::InvalidKernelSpec("gate/up packed K tile exceeds I32"))?;
    let scale_k = config.block_k / REPRESENTATION_BLOCK;
    let scale_k_i32 = i32::try_from(scale_k)
        .map_err(|_| Error::InvalidKernelSpec("gate/up scale K tile exceeds I32"))?;
    let packed_width = ceil_div(config.input_size, 2);
    let scale_width = ceil_div(config.input_size, REPRESENTATION_BLOCK);
    let logical_rows = config
        .output_size
        .checked_mul(2)
        .ok_or(Error::InvalidKernelSpec("gate/up logical width overflows"))?;

    let mut builder = Builder::new("nvfp4_grouped_gate_up_gemv")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let sorted_assignments = pointer(&mut builder, "sorted_assignments", DType::I32)?;
    let block_experts = pointer(&mut builder, "block_experts", DType::I32)?;
    let active_blocks = pointer(&mut builder, "active_blocks", DType::I32)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = pointer(&mut builder, "bias", config.dtype)?;
    let expert_offset = pointer(&mut builder, "expert_offset", DType::I32)?;
    let output = pointer(&mut builder, "output", config.dtype)?;
    let schedule = grouped_decode_schedule(
        &mut builder,
        config,
        &sorted_assignments,
        &block_experts,
        &active_blocks,
        &expert_offset,
    )?;

    let block_n = builder.integer(config.block_n, DType::I32)?;
    let pair_start = builder.multiply(&schedule.output_block, &block_n)?;
    let pair_lanes = builder.range(0, config.block_n as i32)?;
    let pairs = builder.add(&pair_start, &pair_lanes)?;
    let output_limit = builder.integer(config.output_size, DType::I32)?;
    let valid_pairs = builder.compare(Comparison::Less, &pairs, &output_limit)?;
    let two_i32 = builder.integer(2, DType::I32)?;
    let gate_rows = builder.multiply(&pairs, &two_i32)?;
    let one_i32 = builder.integer(1, DType::I32)?;
    let up_rows = builder.add(&gate_rows, &one_i32)?;

    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let payload_stride = logical_rows
        .checked_mul(packed_width)
        .ok_or(Error::InvalidKernelSpec("gate/up payload stride overflows"))?;
    let scale_stride = logical_rows
        .checked_mul(scale_width)
        .ok_or(Error::InvalidKernelSpec("gate/up scale stride overflows"))?;
    let payload_stride = builder.integer(payload_stride, DType::I64)?;
    let scale_stride = builder.integer(scale_stride, DType::I64)?;
    let payload_expert_base = builder.multiply(&expert_i64, &payload_stride)?;
    let scale_expert_base = builder.multiply(&expert_i64, &scale_stride)?;

    let accumulators = builder.if_then_else(
        &schedule.execute,
        |body| {
            let gate_accumulator = body.full_float(&[config.block_n], 0.0, DType::F32)?;
            let up_accumulator = body.full_float(&[config.block_n], 0.0, DType::F32)?;
            let lower = body.integer(0, DType::I32)?;
            let upper = body.integer(config.input_size, DType::I32)?;
            let step = body.integer(config.block_k, DType::I32)?;
            body.for_loop(
                &lower,
                &upper,
                &step,
                &[gate_accumulator, up_accumulator],
                |loop_body, start, carried| {
                    let packed_lanes = loop_body.range(0, packed_k_i32)?;
                    let two = loop_body.integer(2, DType::I32)?;
                    let logical_offsets = loop_body.multiply(&packed_lanes, &two)?;
                    let even_k = loop_body.add(&start, &logical_offsets)?;
                    let one = loop_body.integer(1, DType::I32)?;
                    let odd_k = loop_body.add(&even_k, &one)?;
                    let input_limit = loop_body.integer(config.input_size, DType::I32)?;
                    let valid_even = loop_body.compare(Comparison::Less, &even_k, &input_limit)?;
                    let valid_odd = loop_body.compare(Comparison::Less, &odd_k, &input_limit)?;
                    let even_k_i64 = loop_body.cast(&even_k, DType::I64)?;
                    let odd_k_i64 = loop_body.cast(&odd_k, DType::I64)?;
                    let even_input_offset = loop_body.add(&schedule.source_base, &even_k_i64)?;
                    let odd_input_offset = loop_body.add(&schedule.source_base, &odd_k_i64)?;
                    let even_addresses = loop_body.add_pointer(&input, &even_input_offset)?;
                    let odd_addresses = loop_body.add_pointer(&input, &odd_input_offset)?;
                    let activation_zero = loop_body.full_float(&[packed_k], 0.0, config.dtype)?;
                    let even =
                        loop_body.load_masked(&even_addresses, &valid_even, &activation_zero)?;
                    let odd =
                        loop_body.load_masked(&odd_addresses, &valid_odd, &activation_zero)?;
                    let even = loop_body.cast(&even, DType::F32)?;
                    let odd = loop_body.cast(&odd, DType::F32)?;
                    let even = loop_body.expand_dimension(&even, 0)?;
                    let odd = loop_body.expand_dimension(&odd, 0)?;

                    let packed_start = loop_body.divide(&start, &two)?;
                    let packed_columns = loop_body.add(&packed_start, &packed_lanes)?;
                    let packed_columns_i64 = loop_body.cast(&packed_columns, DType::I64)?;
                    let packed_width_value = loop_body.integer(packed_width, DType::I64)?;
                    let gate_rows_i64 = loop_body.cast(&gate_rows, DType::I64)?;
                    let up_rows_i64 = loop_body.cast(&up_rows, DType::I64)?;
                    let gate_row_offsets =
                        loop_body.multiply(&gate_rows_i64, &packed_width_value)?;
                    let up_row_offsets = loop_body.multiply(&up_rows_i64, &packed_width_value)?;
                    let gate_bases = loop_body.add(&payload_expert_base, &gate_row_offsets)?;
                    let up_bases = loop_body.add(&payload_expert_base, &up_row_offsets)?;
                    let gate_offsets = matrix_offsets(loop_body, &gate_bases, &packed_columns_i64)?;
                    let up_offsets = matrix_offsets(loop_body, &up_bases, &packed_columns_i64)?;
                    let payload_mask = loop_body.mask_2d(&valid_pairs, &valid_even)?;
                    let payload_zero =
                        loop_body.full_integer(&[config.block_n, packed_k], 0, DType::U8)?;
                    let gate_addresses = loop_body.add_pointer(&payload, &gate_offsets)?;
                    let up_addresses = loop_body.add_pointer(&payload, &up_offsets)?;
                    let gate_packed = loop_body.load_masked_streaming(
                        &gate_addresses,
                        &payload_mask,
                        &payload_zero,
                    )?;
                    let up_packed = loop_body.load_masked_streaming(
                        &up_addresses,
                        &payload_mask,
                        &payload_zero,
                    )?;
                    let nibble =
                        loop_body.full_integer(&[config.block_n, packed_k], 0x0f, DType::U8)?;
                    let four = loop_body.full_integer(&[config.block_n, packed_k], 4, DType::U8)?;
                    let gate_low_bits = loop_body.bit_and(&gate_packed, &nibble)?;
                    let gate_high_bits = loop_body.shift_right_logical(&gate_packed, &four)?;
                    let up_low_bits = loop_body.bit_and(&up_packed, &nibble)?;
                    let up_high_bits = loop_body.shift_right_logical(&up_packed, &four)?;
                    let gate_low = decode_e2m1(loop_body, &gate_low_bits)?;
                    let gate_high = decode_e2m1(loop_body, &gate_high_bits)?;
                    let up_low = decode_e2m1(loop_body, &up_low_bits)?;
                    let up_high = decode_e2m1(loop_body, &up_high_bits)?;

                    let scale_lanes = loop_body.range(0, scale_k_i32)?;
                    let representation_block =
                        loop_body.integer(REPRESENTATION_BLOCK, DType::I32)?;
                    let scale_start = loop_body.divide(&start, &representation_block)?;
                    let scale_columns = loop_body.add(&scale_start, &scale_lanes)?;
                    let scale_columns_i64 = loop_body.cast(&scale_columns, DType::I64)?;
                    let scale_limit = loop_body.integer(scale_width, DType::I32)?;
                    let valid_scales =
                        loop_body.compare(Comparison::Less, &scale_columns, &scale_limit)?;
                    let scale_width_value = loop_body.integer(scale_width, DType::I64)?;
                    let gate_scale_row_offsets =
                        loop_body.multiply(&gate_rows_i64, &scale_width_value)?;
                    let up_scale_row_offsets =
                        loop_body.multiply(&up_rows_i64, &scale_width_value)?;
                    let gate_scale_bases =
                        loop_body.add(&scale_expert_base, &gate_scale_row_offsets)?;
                    let up_scale_bases =
                        loop_body.add(&scale_expert_base, &up_scale_row_offsets)?;
                    let scale_mask = loop_body.mask_2d(&valid_pairs, &valid_scales)?;
                    let scale_zero =
                        loop_body.full_integer(&[config.block_n, scale_k], 0, DType::U8)?;
                    let gate_scale_offsets =
                        matrix_offsets(loop_body, &gate_scale_bases, &scale_columns_i64)?;
                    let up_scale_offsets =
                        matrix_offsets(loop_body, &up_scale_bases, &scale_columns_i64)?;
                    let gate_scale_addresses =
                        loop_body.add_pointer(&block_scales, &gate_scale_offsets)?;
                    let up_scale_addresses =
                        loop_body.add_pointer(&block_scales, &up_scale_offsets)?;
                    let gate_scales = loop_body.load_masked_streaming(
                        &gate_scale_addresses,
                        &scale_mask,
                        &scale_zero,
                    )?;
                    let up_scales = loop_body.load_masked_streaming(
                        &up_scale_addresses,
                        &scale_mask,
                        &scale_zero,
                    )?;
                    let expand_scales = |builder: &mut Builder,
                                         scales: Value|
                     -> Result<Value, Error> {
                        let scales = decode_e4m3fn(builder, &scales)?;
                        let scales = builder.reshape(&scales, &[config.block_n, scale_k, 1])?;
                        let scales = builder.broadcast(&scales, &[config.block_n, scale_k, 8])?;
                        builder.reshape(&scales, &[config.block_n, packed_k])
                    };
                    let gate_scales = expand_scales(loop_body, gate_scales)?;
                    let up_scales = expand_scales(loop_body, up_scales)?;
                    let gate_low = loop_body.multiply(&gate_low, &gate_scales)?;
                    let gate_high = loop_body.multiply(&gate_high, &gate_scales)?;
                    let gate_even = loop_body.multiply(&gate_low, &even)?;
                    let gate_odd = loop_body.multiply(&gate_high, &odd)?;
                    let gate_products = loop_body.add(&gate_even, &gate_odd)?;
                    let up_low = loop_body.multiply(&up_low, &up_scales)?;
                    let up_high = loop_body.multiply(&up_high, &up_scales)?;
                    let up_even = loop_body.multiply(&up_low, &even)?;
                    let up_odd = loop_body.multiply(&up_high, &odd)?;
                    let up_products = loop_body.add(&up_even, &up_odd)?;
                    let gate = loop_body.reduce(Reduction::Sum, &gate_products, 1)?;
                    let up = loop_body.reduce(Reduction::Sum, &up_products, 1)?;
                    let gate = loop_body.add(&carried[0], &gate)?;
                    let up = loop_body.add(&carried[1], &up)?;
                    Ok(vec![gate, up])
                },
            )
        },
        |body| {
            Ok(vec![
                body.full_float(&[config.block_n], 0.0, DType::F32)?,
                body.full_float(&[config.block_n], 0.0, DType::F32)?,
            ])
        },
    )?;
    let global = builder.load(&global_scale)?;
    let mut gate = builder.multiply(&accumulators[0], &global)?;
    let mut up = builder.multiply(&accumulators[1], &global)?;

    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let bias_stride = builder.integer(logical_rows, DType::I64)?;
    let bias_base = builder.multiply(&expert_i64, &bias_stride)?;
    let gate_rows_i64 = builder.cast(&gate_rows, DType::I64)?;
    let up_rows_i64 = builder.cast(&up_rows, DType::I64)?;
    let bias_mask = builder.bit_and(&valid_pairs, &schedule.execute)?;
    let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
    let gate_bias_offsets = builder.add(&bias_base, &gate_rows_i64)?;
    let up_bias_offsets = builder.add(&bias_base, &up_rows_i64)?;
    let gate_bias_addresses = builder.add_pointer(&bias, &gate_bias_offsets)?;
    let up_bias_addresses = builder.add_pointer(&bias, &up_bias_offsets)?;
    let gate_bias = builder.load_masked(&gate_bias_addresses, &bias_mask, &bias_zero)?;
    let up_bias = builder.load_masked(&up_bias_addresses, &bias_mask, &bias_zero)?;
    let gate_bias = builder.cast(&gate_bias, DType::F32)?;
    let up_bias = builder.cast(&up_bias, DType::F32)?;
    gate = builder.add(&gate, &gate_bias)?;
    up = builder.add(&up, &up_bias)?;
    let activated = clamped_residual_swiglu(&mut builder, gate, up)?;

    let assignment_i64 = builder.cast(&schedule.address_assignment, DType::I64)?;
    let output_width = builder.integer(config.output_size, DType::I64)?;
    let output_base = builder.multiply(&assignment_i64, &output_width)?;
    let pairs_i64 = builder.cast(&pairs, DType::I64)?;
    let output_offsets = builder.add(&output_base, &pairs_i64)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    let output_mask = builder.bit_and(&valid_pairs, &schedule.store)?;
    let activated = builder.cast(&activated, config.dtype)?;
    builder.store_masked(&output_addresses, &activated, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}

fn build_grouped_down_gemv(config: NvFp4GroupedProjectionConfig) -> Result<Kernel, Error> {
    let packed_k = config.block_k / 2;
    let packed_k_i32 = i32::try_from(packed_k)
        .map_err(|_| Error::InvalidKernelSpec("down packed K tile exceeds I32"))?;
    let scale_k = config.block_k / REPRESENTATION_BLOCK;
    let scale_k_i32 = i32::try_from(scale_k)
        .map_err(|_| Error::InvalidKernelSpec("down scale K tile exceeds I32"))?;
    let packed_width = ceil_div(config.input_size, 2);
    let scale_width = ceil_div(config.input_size, REPRESENTATION_BLOCK);

    let mut builder = Builder::new("nvfp4_grouped_down_gemv")?;
    let input = pointer(&mut builder, "input", config.dtype)?;
    let sorted_assignments = pointer(&mut builder, "sorted_assignments", DType::I32)?;
    let block_experts = pointer(&mut builder, "block_experts", DType::I32)?;
    let active_blocks = pointer(&mut builder, "active_blocks", DType::I32)?;
    let payload = pointer(&mut builder, "payload", DType::U8)?;
    let block_scales = pointer(&mut builder, "block_scales", DType::U8)?;
    let global_scale = pointer(&mut builder, "global_scale", DType::F32)?;
    let bias = pointer(&mut builder, "bias", config.dtype)?;
    let expert_offset = pointer(&mut builder, "expert_offset", DType::I32)?;
    let routing_weights = pointer(&mut builder, "routing_weights", config.dtype)?;
    let output = pointer(&mut builder, "output", config.dtype)?;
    let schedule = grouped_decode_schedule(
        &mut builder,
        config,
        &sorted_assignments,
        &block_experts,
        &active_blocks,
        &expert_offset,
    )?;

    let block_n = builder.integer(config.block_n, DType::I32)?;
    let column_start = builder.multiply(&schedule.output_block, &block_n)?;
    let column_lanes = builder.range(0, config.block_n as i32)?;
    let columns = builder.add(&column_start, &column_lanes)?;
    let output_limit = builder.integer(config.output_size, DType::I32)?;
    let valid_columns = builder.compare(Comparison::Less, &columns, &output_limit)?;

    let expert_i64 = builder.cast(&schedule.address_expert, DType::I64)?;
    let payload_stride = config
        .output_size
        .checked_mul(packed_width)
        .ok_or(Error::InvalidKernelSpec("down payload stride overflows"))?;
    let scale_stride = config
        .output_size
        .checked_mul(scale_width)
        .ok_or(Error::InvalidKernelSpec("down scale stride overflows"))?;
    let payload_stride = builder.integer(payload_stride, DType::I64)?;
    let scale_stride = builder.integer(scale_stride, DType::I64)?;
    let payload_expert_base = builder.multiply(&expert_i64, &payload_stride)?;
    let scale_expert_base = builder.multiply(&expert_i64, &scale_stride)?;

    let accumulators = builder.if_then_else(
        &schedule.execute,
        |body| {
            let accumulator = body.full_float(&[config.block_n], 0.0, DType::F32)?;
            let lower = body.integer(0, DType::I32)?;
            let upper = body.integer(config.input_size, DType::I32)?;
            let step = body.integer(config.block_k, DType::I32)?;
            body.for_loop(
                &lower,
                &upper,
                &step,
                &[accumulator],
                |loop_body, start, carried| {
                    let packed_lanes = loop_body.range(0, packed_k_i32)?;
                    let two = loop_body.integer(2, DType::I32)?;
                    let logical_offsets = loop_body.multiply(&packed_lanes, &two)?;
                    let even_k = loop_body.add(&start, &logical_offsets)?;
                    let one = loop_body.integer(1, DType::I32)?;
                    let odd_k = loop_body.add(&even_k, &one)?;
                    let input_limit = loop_body.integer(config.input_size, DType::I32)?;
                    let valid_even = loop_body.compare(Comparison::Less, &even_k, &input_limit)?;
                    let valid_odd = loop_body.compare(Comparison::Less, &odd_k, &input_limit)?;
                    let activation_zero = loop_body.full_float(&[packed_k], 0.0, config.dtype)?;
                    let even_k_i64 = loop_body.cast(&even_k, DType::I64)?;
                    let odd_k_i64 = loop_body.cast(&odd_k, DType::I64)?;
                    let even_input_offset = loop_body.add(&schedule.source_base, &even_k_i64)?;
                    let odd_input_offset = loop_body.add(&schedule.source_base, &odd_k_i64)?;
                    let even_addresses = loop_body.add_pointer(&input, &even_input_offset)?;
                    let odd_addresses = loop_body.add_pointer(&input, &odd_input_offset)?;
                    let even =
                        loop_body.load_masked(&even_addresses, &valid_even, &activation_zero)?;
                    let odd =
                        loop_body.load_masked(&odd_addresses, &valid_odd, &activation_zero)?;
                    let even = loop_body.cast(&even, DType::F32)?;
                    let odd = loop_body.cast(&odd, DType::F32)?;
                    let even = loop_body.expand_dimension(&even, 0)?;
                    let odd = loop_body.expand_dimension(&odd, 0)?;

                    let packed_start = loop_body.divide(&start, &two)?;
                    let packed_columns = loop_body.add(&packed_start, &packed_lanes)?;
                    let packed_columns_i64 = loop_body.cast(&packed_columns, DType::I64)?;
                    let columns_i64 = loop_body.cast(&columns, DType::I64)?;
                    let packed_width_value = loop_body.integer(packed_width, DType::I64)?;
                    let row_offsets = loop_body.multiply(&columns_i64, &packed_width_value)?;
                    let row_bases = loop_body.add(&payload_expert_base, &row_offsets)?;
                    let payload_offsets =
                        matrix_offsets(loop_body, &row_bases, &packed_columns_i64)?;
                    let payload_mask = loop_body.mask_2d(&valid_columns, &valid_even)?;
                    let payload_zero =
                        loop_body.full_integer(&[config.block_n, packed_k], 0, DType::U8)?;
                    let payload_addresses = loop_body.add_pointer(&payload, &payload_offsets)?;
                    let packed = loop_body.load_masked_streaming(
                        &payload_addresses,
                        &payload_mask,
                        &payload_zero,
                    )?;
                    let nibble =
                        loop_body.full_integer(&[config.block_n, packed_k], 0x0f, DType::U8)?;
                    let four = loop_body.full_integer(&[config.block_n, packed_k], 4, DType::U8)?;
                    let low_bits = loop_body.bit_and(&packed, &nibble)?;
                    let high_bits = loop_body.shift_right_logical(&packed, &four)?;
                    let low = decode_e2m1(loop_body, &low_bits)?;
                    let high = decode_e2m1(loop_body, &high_bits)?;

                    let scale_lanes = loop_body.range(0, scale_k_i32)?;
                    let representation_block =
                        loop_body.integer(REPRESENTATION_BLOCK, DType::I32)?;
                    let scale_start = loop_body.divide(&start, &representation_block)?;
                    let scale_columns = loop_body.add(&scale_start, &scale_lanes)?;
                    let scale_columns_i64 = loop_body.cast(&scale_columns, DType::I64)?;
                    let scale_limit = loop_body.integer(scale_width, DType::I32)?;
                    let valid_scales =
                        loop_body.compare(Comparison::Less, &scale_columns, &scale_limit)?;
                    let scale_width_value = loop_body.integer(scale_width, DType::I64)?;
                    let scale_row_offsets = loop_body.multiply(&columns_i64, &scale_width_value)?;
                    let scale_bases = loop_body.add(&scale_expert_base, &scale_row_offsets)?;
                    let scale_offsets =
                        matrix_offsets(loop_body, &scale_bases, &scale_columns_i64)?;
                    let scale_mask = loop_body.mask_2d(&valid_columns, &valid_scales)?;
                    let scale_zero =
                        loop_body.full_integer(&[config.block_n, scale_k], 0, DType::U8)?;
                    let scale_addresses = loop_body.add_pointer(&block_scales, &scale_offsets)?;
                    let scales = loop_body.load_masked_streaming(
                        &scale_addresses,
                        &scale_mask,
                        &scale_zero,
                    )?;
                    let scales = decode_e4m3fn(loop_body, &scales)?;
                    let scales = loop_body.reshape(&scales, &[config.block_n, scale_k, 1])?;
                    let scales = loop_body.broadcast(&scales, &[config.block_n, scale_k, 8])?;
                    let scales = loop_body.reshape(&scales, &[config.block_n, packed_k])?;
                    let low = loop_body.multiply(&low, &scales)?;
                    let high = loop_body.multiply(&high, &scales)?;
                    let even_products = loop_body.multiply(&low, &even)?;
                    let odd_products = loop_body.multiply(&high, &odd)?;
                    let products = loop_body.add(&even_products, &odd_products)?;
                    let partial = loop_body.reduce(Reduction::Sum, &products, 1)?;
                    let result = loop_body.add(&carried[0], &partial)?;
                    Ok(vec![result])
                },
            )
        },
        |body| Ok(vec![body.full_float(&[config.block_n], 0.0, DType::F32)?]),
    )?;
    let global = builder.load(&global_scale)?;
    let mut result = builder.multiply(&accumulators[0], &global)?;
    let columns_i64 = builder.cast(&columns, DType::I64)?;
    let output_width = builder.integer(config.output_size, DType::I64)?;
    let bias_base = builder.multiply(&expert_i64, &output_width)?;
    let bias_mask = builder.bit_and(&valid_columns, &schedule.execute)?;
    let bias_zero = builder.full_float(&[config.block_n], 0.0, config.dtype)?;
    let bias_offsets = builder.add(&bias_base, &columns_i64)?;
    let bias_addresses = builder.add_pointer(&bias, &bias_offsets)?;
    let bias_value = builder.load_masked(&bias_addresses, &bias_mask, &bias_zero)?;
    let bias_value = builder.cast(&bias_value, DType::F32)?;
    result = builder.add(&result, &bias_value)?;
    let routing_addresses = builder.add_pointer(&routing_weights, &schedule.address_assignment)?;
    let routing_zero = builder.float(0.0, config.dtype)?;
    let routing = builder.load_masked(&routing_addresses, &schedule.execute, &routing_zero)?;
    let routing = builder.cast(&routing, DType::F32)?;
    result = builder.multiply(&result, &routing)?;

    let assignment_i64 = builder.cast(&schedule.address_assignment, DType::I64)?;
    let output_base = builder.multiply(&assignment_i64, &output_width)?;
    let output_offsets = builder.add(&output_base, &columns_i64)?;
    let output_mask = builder.bit_and(&valid_columns, &schedule.store)?;
    let result = builder.cast(&result, config.dtype)?;
    let output_addresses = builder.add_pointer(&output, &output_offsets)?;
    builder.store_masked(&output_addresses, &result, &output_mask)?;
    builder.return_void()?;
    builder.finish()
}

fn clamped_residual_swiglu(builder: &mut Builder, gate: Value, up: Value) -> Result<Value, Error> {
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
    builder.multiply(&residual, &swish)
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

    // For magnitudes two through seven, the exact IEEE F32 bit pattern is
    // `(magnitude + 252) << 22`. Zero and 0.5 are the only exceptions and are
    // exactly `magnitude * bits(0.5)`. This affine construction removes the
    // separate dynamic exponent and mantissa paths from every decoded lane.
    let magnitude_i32 = builder.cast(&magnitude, DType::I32)?;
    let normal_bias = builder.full_integer(&shape, 252, DType::I32)?;
    let normal_bits = builder.add(&magnitude_i32, &normal_bias)?;
    let normal_place = builder.full_integer(&shape, 1 << 22, DType::I32)?;
    let normal_bits = builder.multiply(&normal_bits, &normal_place)?;
    let half_bits = builder.full_integer(&shape, 0x3f00_0000, DType::I32)?;
    let small_bits = builder.multiply(&magnitude_i32, &half_bits)?;
    let two = builder.full_integer(&shape, 2, DType::U8)?;
    let is_small = builder.compare(Comparison::Less, &magnitude, &two)?;
    let magnitude_bits = builder.select(&is_small, &small_bits, &normal_bits)?;

    // Preserve E2M1 negative zero by transferring the code's sign into the
    // exact F32 sign bit before reinterpreting the integer tensor.
    let three = builder.full_integer(&shape, 3, DType::U8)?;
    let sign = builder.shift_right_logical(code, &three)?;
    let sign = builder.cast(&sign, DType::I32)?;
    let sign_place = builder.full_integer(&shape, i64::from(i32::MIN), DType::I32)?;
    let sign_bits = builder.multiply(&sign, &sign_place)?;
    let value_bits = builder.add(&magnitude_bits, &sign_bits)?;
    builder.bitcast(&value_bits, DType::F32)
}

fn decode_e4m3fn(builder: &mut Builder, bits: &super::Value) -> Result<super::Value, Error> {
    let shape = code_shape(bits)?;
    let payload_mask = builder.full_integer(&shape, 0x7f, DType::U8)?;
    let payload = builder.bit_and(bits, &payload_mask)?;
    let three = builder.full_integer(&shape, 3, DType::U8)?;
    let exponent = builder.shift_right_logical(&payload, &three)?;
    let fraction_mask = builder.full_integer(&shape, 0x07, DType::U8)?;
    let fraction = builder.bit_and(&payload, &fraction_mask)?;

    // A normal non-negative E4M3FN value becomes F32 by shifting its complete
    // seven-bit exponent/fraction payload into place and adding the bias
    // delta. Subnormals are exactly `fraction * 2^-9`; U8-to-F32 conversion is
    // exact for all eight cases and avoids a multi-branch bit construction.
    let payload_i32 = builder.cast(&payload, DType::I32)?;
    let payload_place = builder.full_integer(&shape, 1 << 20, DType::I32)?;
    let normal_bits = builder.multiply(&payload_i32, &payload_place)?;
    let bias_delta = builder.full_integer(&shape, 0x3c00_0000, DType::I32)?;
    let normal_bits = builder.add(&normal_bits, &bias_delta)?;
    let normal = builder.bitcast(&normal_bits, DType::F32)?;
    let fraction = builder.cast(&fraction, DType::F32)?;
    let subnormal_step = builder.full_float(&shape, 2.0_f64.powi(-9), DType::F32)?;
    let subnormal = builder.multiply(&fraction, &subnormal_step)?;
    let zero = builder.full_integer(&shape, 0, DType::U8)?;
    let is_zero_exponent = builder.compare(Comparison::Equal, &exponent, &zero)?;
    let value = builder.select(&is_zero_exponent, &subnormal, &normal)?;

    // The scalar representation rejects negative scales and E4M3FN NaN. A
    // malformed component cannot report an error from inside a custom call,
    // so poison those unreachable encodings with a canonical F32 NaN instead
    // of silently interpreting them as a valid positive scale.
    let invalid_boundary = builder.full_integer(&shape, 0x7f, DType::U8)?;
    let invalid = builder.compare(Comparison::GreaterEqual, bits, &invalid_boundary)?;
    let nan_bits = builder.full_integer(&shape, 0x7fc0_0000, DType::I32)?;
    let nan = builder.bitcast(&nan_bits, DType::F32)?;
    builder.select(&invalid, &nan, &value)
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
