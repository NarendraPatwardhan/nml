//! Bounded CPU execution over NML's compact NVFP4 representation.
//!
//! This crate owns representation execution, not a graph dtype. Operations
//! decode one row or contraction tile into registers/scratch and never create
//! a persistent dense weight. The portable implementation is both the product
//! fallback for x86-64/AArch64 and the correctness anchor for optimized CPU and
//! CUDA paths.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(nml_cuda)]
mod cuda;
mod ffi;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(nml_cuda)]
pub use cuda::register_cuda;
pub use ffi::register_cpu;

use nml_parameter::nvfp4::{decode_e2m1, decode_e4m3fn_scale};
use nml_types::{DType, Shape};
use std::error::Error as StdError;
use std::fmt;

const BLOCK_SIZE: usize = 16;

#[derive(Clone, Copy)]
enum CpuImplementation {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl CpuImplementation {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if x86::available() {
            return Self::Avx2;
        }
        Self::Portable
    }
}

/// A validated borrowed view of one compact logical weight.
///
/// It is intentionally not exported by the `nml` facade. Model authors pass a
/// logical `Parameter`; compiler/runtime lowering constructs this view at the
/// private CPU kernel boundary.
pub struct Weight<'a> {
    dimensions: Vec<usize>,
    row_width: usize,
    packed_width: usize,
    scale_width: usize,
    payload: &'a [u8],
    block_scales: &'a [u8],
    global_scale: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidLogicalDType(DType),
    InvalidLogicalShape,
    ExtentOverflow,
    PayloadExtent {
        expected: usize,
        actual: usize,
    },
    ScaleExtent {
        expected: usize,
        actual: usize,
    },
    InvalidGlobalScale,
    InvalidScale(u8),
    NonZeroPadding,
    ZeroScaleWithNonZeroValue,
    Rank {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    InputExtent {
        operation: &'static str,
        expected_multiple: usize,
        actual: usize,
    },
    OutputExtent {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        index: usize,
        extent: usize,
    },
    ExpertOutOfBounds {
        expert: usize,
        experts: usize,
    },
    IncompatibleExpertWeights,
    BiasExtent {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    RoutingExtent,
}

impl<'a> Weight<'a> {
    pub fn new(
        logical_shape: Shape,
        payload: &'a [u8],
        block_scales: &'a [u8],
        global_scale: f32,
    ) -> Result<Self, Error> {
        if !matches!(logical_shape.dtype(), DType::F16 | DType::Bf16) {
            return Err(Error::InvalidLogicalDType(logical_shape.dtype()));
        }
        if logical_shape.rank() == 0 {
            return Err(Error::InvalidLogicalShape);
        }
        let dimensions = logical_shape
            .dimensions()
            .iter()
            .map(|&dimension| usize::try_from(dimension).map_err(|_| Error::InvalidLogicalShape))
            .collect::<Result<Vec<_>, _>>()?;
        let row_width = *dimensions.last().ok_or(Error::InvalidLogicalShape)?;
        if row_width == 0 || !global_scale.is_finite() || global_scale <= 0.0 {
            return if row_width == 0 {
                Err(Error::InvalidLogicalShape)
            } else {
                Err(Error::InvalidGlobalScale)
            };
        }
        let rows = dimensions[..dimensions.len() - 1]
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or(Error::ExtentOverflow)?;
        let packed_width = row_width.div_ceil(2);
        let scale_width = row_width.div_ceil(BLOCK_SIZE);
        let expected_payload = rows
            .checked_mul(packed_width)
            .ok_or(Error::ExtentOverflow)?;
        let expected_scales = rows.checked_mul(scale_width).ok_or(Error::ExtentOverflow)?;
        if payload.len() != expected_payload {
            return Err(Error::PayloadExtent {
                expected: expected_payload,
                actual: payload.len(),
            });
        }
        if block_scales.len() != expected_scales {
            return Err(Error::ScaleExtent {
                expected: expected_scales,
                actual: block_scales.len(),
            });
        }

        for row in 0..rows {
            if row_width & 1 != 0 && payload[row * packed_width + packed_width - 1] & 0xf0 != 0 {
                return Err(Error::NonZeroPadding);
            }
            for block in 0..scale_width {
                let scale_bits = block_scales[row * scale_width + block];
                let scale =
                    decode_e4m3fn_scale(scale_bits).map_err(|_| Error::InvalidScale(scale_bits))?;
                if scale == 0.0 {
                    let start = block * BLOCK_SIZE;
                    let end = row_width.min(start + BLOCK_SIZE);
                    if (start..end)
                        .any(|column| nibble(payload, row * packed_width, column) & 0x07 != 0)
                    {
                        return Err(Error::ZeroScaleWithNonZeroValue);
                    }
                }
            }
        }

        Ok(Self {
            dimensions,
            row_width,
            packed_width,
            scale_width,
            payload,
            block_scales,
            global_scale,
        })
    }

    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    fn value(&self, row: usize, column: usize) -> f32 {
        let code = nibble(self.payload, row * self.packed_width, column);
        let scale =
            decode_e4m3fn_scale(self.block_scales[row * self.scale_width + column / BLOCK_SIZE])
                .expect("Weight construction validates all scale encodings");
        decode_e2m1(code).expect("a nibble always contains a valid E2M1 code")
            * (scale * self.global_scale)
    }
}

/// Decodes selected rows from a `[vocabulary, embedding]` weight.
pub fn embedding(weight: &Weight<'_>, indices: &[usize], output: &mut [f32]) -> Result<(), Error> {
    require_rank(weight, "embedding", 2)?;
    let vocabulary = weight.dimensions[0];
    let width = weight.dimensions[1];
    let expected = indices
        .len()
        .checked_mul(width)
        .ok_or(Error::ExtentOverflow)?;
    require_output("embedding", output, expected)?;
    let implementation = CpuImplementation::detect();
    for (destination, &index) in output.chunks_exact_mut(width).zip(indices) {
        if index >= vocabulary {
            return Err(Error::IndexOutOfBounds {
                index,
                extent: vocabulary,
            });
        }
        decode_row(weight, index, destination, implementation);
    }
    Ok(())
}

/// Computes `[..., K] * [N, K]^T` with F32 accumulation.
pub fn linear(
    input: &[f32],
    weight: &Weight<'_>,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<(), Error> {
    require_rank(weight, "linear", 2)?;
    let outputs = weight.dimensions[0];
    let inputs = weight.dimensions[1];
    if input.len() % inputs != 0 {
        return Err(Error::InputExtent {
            operation: "linear",
            expected_multiple: inputs,
            actual: input.len(),
        });
    }
    if let Some(bias) = bias
        && bias.len() != outputs
    {
        return Err(Error::BiasExtent {
            operation: "linear",
            expected: outputs,
            actual: bias.len(),
        });
    }
    let rows = input.len() / inputs;
    let expected = rows.checked_mul(outputs).ok_or(Error::ExtentOverflow)?;
    require_output("linear", output, expected)?;
    let implementation = CpuImplementation::detect();
    for (input_row, output_row) in input
        .chunks_exact(inputs)
        .zip(output.chunks_exact_mut(outputs))
    {
        for (output_index, destination) in output_row.iter_mut().enumerate() {
            *destination = bias.map_or(0.0, |values| values[output_index])
                + dot_row(weight, output_index, input_row, implementation);
        }
    }
    Ok(())
}

/// Applies an input-major `[experts, K, N]` projection to routed assignments.
///
/// `expert_indices` has one entry per input row. Empty experts require no
/// special case; uneven routing changes work distribution but not semantics.
pub fn grouped_projection(
    input: &[f32],
    expert_indices: &[usize],
    weight: &Weight<'_>,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<(), Error> {
    require_rank(weight, "grouped projection", 3)?;
    let experts = weight.dimensions[0];
    let inputs = weight.dimensions[1];
    let outputs = weight.dimensions[2];
    let expected_input = expert_indices
        .len()
        .checked_mul(inputs)
        .ok_or(Error::ExtentOverflow)?;
    if input.len() != expected_input {
        return Err(Error::InputExtent {
            operation: "grouped projection",
            expected_multiple: inputs,
            actual: input.len(),
        });
    }
    if let Some(bias) = bias {
        let expected = experts.checked_mul(outputs).ok_or(Error::ExtentOverflow)?;
        if bias.len() != expected {
            return Err(Error::BiasExtent {
                operation: "grouped projection",
                expected,
                actual: bias.len(),
            });
        }
    }
    let expected_output = expert_indices
        .len()
        .checked_mul(outputs)
        .ok_or(Error::ExtentOverflow)?;
    require_output("grouped projection", output, expected_output)?;
    let implementation = CpuImplementation::detect();
    for (assignment, &expert) in expert_indices.iter().enumerate() {
        if expert >= experts {
            return Err(Error::ExpertOutOfBounds { expert, experts });
        }
        project_expert(
            &input[assignment * inputs..(assignment + 1) * inputs],
            expert,
            weight,
            bias.map(|values| &values[expert * outputs..(expert + 1) * outputs]),
            &mut output[assignment * outputs..(assignment + 1) * outputs],
            implementation,
        );
    }
    Ok(())
}

/// Exact GPT-OSS routed expert MLP over compact gate/up and down weights.
///
/// Gate/up channels are interleaved. Gate values clamp only above `limit`, up
/// values clamp symmetrically, and the residual multiplicand is `(up + 1)`.
/// Routing weights are applied after the dense expert output and accumulated
/// into the owning token, matching the pinned GPT-OSS implementation.
pub fn gpt_oss_experts(
    hidden: &[f32],
    token_count: usize,
    router_indices: &[usize],
    routing_weights: &[f32],
    gate_up: &Weight<'_>,
    gate_up_bias: &[f32],
    down: &Weight<'_>,
    down_bias: &[f32],
    output: &mut [f32],
) -> Result<(), Error> {
    require_rank(gate_up, "GPT-OSS gate/up", 3)?;
    require_rank(down, "GPT-OSS down", 3)?;
    let experts = gate_up.dimensions[0];
    let hidden_size = gate_up.dimensions[1];
    let doubled_intermediate = gate_up.dimensions[2];
    if doubled_intermediate % 2 != 0
        || down.dimensions != [experts, doubled_intermediate / 2, hidden_size]
    {
        return Err(Error::IncompatibleExpertWeights);
    }
    let intermediate = doubled_intermediate / 2;
    if hidden.len()
        != token_count
            .checked_mul(hidden_size)
            .ok_or(Error::ExtentOverflow)?
        || router_indices.len() != routing_weights.len()
    {
        return Err(Error::RoutingExtent);
    }
    let gate_bias_extent = experts
        .checked_mul(doubled_intermediate)
        .ok_or(Error::ExtentOverflow)?;
    if gate_up_bias.len() != gate_bias_extent {
        return Err(Error::BiasExtent {
            operation: "GPT-OSS gate/up",
            expected: gate_bias_extent,
            actual: gate_up_bias.len(),
        });
    }
    let down_bias_extent = experts
        .checked_mul(hidden_size)
        .ok_or(Error::ExtentOverflow)?;
    if down_bias.len() != down_bias_extent {
        return Err(Error::BiasExtent {
            operation: "GPT-OSS down",
            expected: down_bias_extent,
            actual: down_bias.len(),
        });
    }
    require_output(
        "GPT-OSS experts",
        output,
        token_count
            .checked_mul(hidden_size)
            .ok_or(Error::ExtentOverflow)?,
    )?;
    output.fill(0.0);
    if token_count == 0 {
        if !router_indices.is_empty() {
            return Err(Error::RoutingExtent);
        }
        return Ok(());
    }
    if router_indices.is_empty() || router_indices.len() % token_count != 0 {
        return Err(Error::RoutingExtent);
    }
    let top_k = router_indices.len() / token_count;
    let mut gate_up_scratch = vec![0.0f32; doubled_intermediate];
    let mut activated = vec![0.0f32; intermediate];
    let mut down_scratch = vec![0.0f32; hidden_size];
    let implementation = CpuImplementation::detect();
    for token in 0..token_count {
        let input = &hidden[token * hidden_size..(token + 1) * hidden_size];
        for route in 0..top_k {
            let assignment = token * top_k + route;
            let expert = router_indices[assignment];
            if expert >= experts {
                return Err(Error::ExpertOutOfBounds { expert, experts });
            }
            project_expert(
                input,
                expert,
                gate_up,
                Some(
                    &gate_up_bias
                        [expert * doubled_intermediate..(expert + 1) * doubled_intermediate],
                ),
                &mut gate_up_scratch,
                implementation,
            );
            for index in 0..intermediate {
                let gate = gate_up_scratch[index * 2].min(7.0);
                let up = gate_up_scratch[index * 2 + 1].clamp(-7.0, 7.0);
                let swish = gate * (1.0 / (1.0 + (-1.702 * gate).exp()));
                activated[index] = (up + 1.0) * swish;
            }
            project_expert(
                &activated,
                expert,
                down,
                Some(&down_bias[expert * hidden_size..(expert + 1) * hidden_size]),
                &mut down_scratch,
                implementation,
            );
            let routing_weight = routing_weights[assignment];
            for (destination, value) in output[token * hidden_size..(token + 1) * hidden_size]
                .iter_mut()
                .zip(&down_scratch)
            {
                *destination += routing_weight * value;
            }
        }
    }
    Ok(())
}

fn project_expert(
    input: &[f32],
    expert: usize,
    weight: &Weight<'_>,
    bias: Option<&[f32]>,
    output: &mut [f32],
    implementation: CpuImplementation,
) {
    let inputs = weight.dimensions[1];
    if let Some(bias) = bias {
        output.copy_from_slice(bias);
    } else {
        output.fill(0.0);
    }
    for (input_index, &activation) in input.iter().enumerate().take(inputs) {
        let row = expert * inputs + input_index;
        axpy_row(weight, row, activation, output, implementation);
    }
}

fn decode_row(
    weight: &Weight<'_>,
    row: usize,
    output: &mut [f32],
    implementation: CpuImplementation,
) {
    #[cfg(target_arch = "x86_64")]
    if matches!(implementation, CpuImplementation::Avx2) {
        // SAFETY: the implementation is selected only after runtime AVX2
        // detection; public validation established row and output extents.
        unsafe { x86::decode_row(weight, row, output) };
        return;
    }
    for block_start in (0..weight.row_width).step_by(BLOCK_SIZE) {
        let block_end = weight.row_width.min(block_start + BLOCK_SIZE);
        for column in block_start..block_end {
            output[column] = weight.value(row, column);
        }
    }
}

fn dot_row(
    weight: &Weight<'_>,
    row: usize,
    input: &[f32],
    implementation: CpuImplementation,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if matches!(implementation, CpuImplementation::Avx2) {
        // SAFETY: runtime feature detection and the validated linear geometry
        // satisfy the AVX2 kernel contract.
        return unsafe { x86::dot_row(weight, row, input) };
    }
    input
        .iter()
        .enumerate()
        .map(|(column, activation)| activation * weight.value(row, column))
        .sum()
}

fn axpy_row(
    weight: &Weight<'_>,
    row: usize,
    activation: f32,
    output: &mut [f32],
    implementation: CpuImplementation,
) {
    #[cfg(target_arch = "x86_64")]
    if matches!(implementation, CpuImplementation::Avx2) {
        // SAFETY: runtime feature detection and grouped-projection validation
        // satisfy the AVX2 kernel contract.
        unsafe { x86::axpy_row(weight, row, activation, output) };
        return;
    }
    for (column, destination) in output.iter_mut().enumerate() {
        *destination += activation * weight.value(row, column);
    }
}

fn nibble(payload: &[u8], row_offset: usize, column: usize) -> u8 {
    let byte = payload[row_offset + column / 2];
    if column & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

fn require_rank(
    weight: &Weight<'_>,
    operation: &'static str,
    expected: usize,
) -> Result<(), Error> {
    if weight.dimensions.len() == expected {
        Ok(())
    } else {
        Err(Error::Rank {
            operation,
            expected,
            actual: weight.dimensions.len(),
        })
    }
}

fn require_output(operation: &'static str, output: &[f32], expected: usize) -> Result<(), Error> {
    if output.len() == expected {
        Ok(())
    } else {
        Err(Error::OutputExtent {
            operation,
            expected,
            actual: output.len(),
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogicalDType(dtype) => {
                write!(
                    formatter,
                    "NVFP4 logical weight dtype {dtype:?} is not F16 or BF16"
                )
            }
            Self::InvalidLogicalShape => formatter.write_str("invalid NVFP4 logical weight shape"),
            Self::ExtentOverflow => formatter.write_str("NVFP4 physical extent overflows usize"),
            Self::PayloadExtent { expected, actual } => write!(
                formatter,
                "NVFP4 payload extent mismatch: expected {expected} bytes, received {actual}"
            ),
            Self::ScaleExtent { expected, actual } => write!(
                formatter,
                "NVFP4 block-scale extent mismatch: expected {expected} bytes, received {actual}"
            ),
            Self::InvalidGlobalScale => {
                formatter.write_str("NVFP4 global scale must be finite and positive")
            }
            Self::InvalidScale(bits) => {
                write!(formatter, "invalid NVFP4 E4M3FN scale bits 0x{bits:02x}")
            }
            Self::NonZeroPadding => formatter.write_str("NVFP4 payload has nonzero edge padding"),
            Self::ZeroScaleWithNonZeroValue => {
                formatter.write_str("NVFP4 zero-scale block contains a nonzero E2M1 magnitude")
            }
            Self::Rank {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "{operation} requires a rank-{expected} NVFP4 weight, received rank {actual}"
            ),
            Self::InputExtent {
                operation,
                expected_multiple,
                actual,
            } => write!(
                formatter,
                "{operation} input extent {actual} does not match rows of {expected_multiple} values"
            ),
            Self::OutputExtent {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "{operation} output extent mismatch: expected {expected}, received {actual}"
            ),
            Self::IndexOutOfBounds { index, extent } => {
                write!(
                    formatter,
                    "embedding index {index} is outside vocabulary extent {extent}"
                )
            }
            Self::ExpertOutOfBounds { expert, experts } => {
                write!(
                    formatter,
                    "expert index {expert} is outside expert count {experts}"
                )
            }
            Self::IncompatibleExpertWeights => {
                formatter.write_str("GPT-OSS gate/up and down NVFP4 weight shapes are incompatible")
            }
            Self::BiasExtent {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "{operation} bias extent mismatch: expected {expected}, received {actual}"
            ),
            Self::RoutingExtent => formatter
                .write_str("GPT-OSS routing tensors do not match the token and hidden extents"),
        }
    }
}

impl StdError for Error {}
