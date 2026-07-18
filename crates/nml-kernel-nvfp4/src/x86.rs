//! AVX2 execution for compact NVFP4 rows.
//!
//! Eight packed bytes become sixteen E2M1 codes in registers. A gather from
//! the fixed sixteen-value codebook and one broadcast E4M3/global scale then
//! produce a complete NVFP4 block without a dense-weight allocation.

use super::{BLOCK_SIZE, Weight};
use nml_parameter::nvfp4::decode_e4m3fn_scale;
use std::arch::x86_64::*;

const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub(super) fn available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

/// Decodes exactly one complete sixteen-value representation block.
///
/// # Safety
///
/// The caller has checked AVX2 support and that `block_start..block_start+16`
/// lies inside the logical row. `Weight::new` has already validated the
/// corresponding packed and scale extents.
#[target_feature(enable = "avx2")]
unsafe fn decode_block(weight: &Weight<'_>, row: usize, block_start: usize) -> (__m256, __m256) {
    let payload_offset = row * weight.packed_width + block_start / 2;
    // SAFETY: the caller supplies a complete logical block, which occupies
    // exactly eight validated payload bytes.
    let packed = unsafe {
        _mm_loadl_epi64(
            weight
                .payload
                .as_ptr()
                .add(payload_offset)
                .cast::<__m128i>(),
        )
    };
    let nibble_mask = _mm_set1_epi8(0x0f);
    let low = _mm_and_si128(packed, nibble_mask);
    let high = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
    let interleaved = _mm_unpacklo_epi8(low, high);
    let first_codes = _mm256_cvtepu8_epi32(interleaved);
    let second_codes = _mm256_cvtepu8_epi32(_mm_srli_si128(interleaved, 8));
    let first = unsafe { _mm256_i32gather_ps(E2M1_VALUES.as_ptr(), first_codes, 4) };
    let second = unsafe { _mm256_i32gather_ps(E2M1_VALUES.as_ptr(), second_codes, 4) };
    let scale_bits = weight.block_scales[row * weight.scale_width + block_start / BLOCK_SIZE];
    let scale = decode_e4m3fn_scale(scale_bits)
        .expect("Weight construction validates E4M3FN scales")
        * weight.global_scale;
    let scale = _mm256_set1_ps(scale);
    (_mm256_mul_ps(first, scale), _mm256_mul_ps(second, scale))
}

/// # Safety
///
/// The caller has checked AVX2 support and supplied a validated row and output
/// slice whose length equals the logical row width.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn decode_row(weight: &Weight<'_>, row: usize, output: &mut [f32]) {
    let vector_width = weight.row_width / BLOCK_SIZE * BLOCK_SIZE;
    for block_start in (0..vector_width).step_by(BLOCK_SIZE) {
        let (first, second) = unsafe { decode_block(weight, row, block_start) };
        // SAFETY: each vector is stored inside the complete block promised by
        // the caller.
        unsafe {
            _mm256_storeu_ps(output.as_mut_ptr().add(block_start), first);
            _mm256_storeu_ps(output.as_mut_ptr().add(block_start + 8), second);
        }
    }
    for column in vector_width..weight.row_width {
        output[column] = weight.value(row, column);
    }
}

/// # Safety
///
/// The caller has checked AVX2 support and supplied a validated weight row and
/// an input slice with the same logical width.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot_row(weight: &Weight<'_>, row: usize, input: &[f32]) -> f32 {
    let vector_width = weight.row_width / BLOCK_SIZE * BLOCK_SIZE;
    let mut vector_sum = _mm256_setzero_ps();
    for block_start in (0..vector_width).step_by(BLOCK_SIZE) {
        let (first, second) = unsafe { decode_block(weight, row, block_start) };
        // SAFETY: the input covers the same complete block.
        let input_first = unsafe { _mm256_loadu_ps(input.as_ptr().add(block_start)) };
        let input_second = unsafe { _mm256_loadu_ps(input.as_ptr().add(block_start + 8)) };
        vector_sum = _mm256_add_ps(vector_sum, _mm256_mul_ps(input_first, first));
        vector_sum = _mm256_add_ps(vector_sum, _mm256_mul_ps(input_second, second));
    }
    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), vector_sum) };
    let mut sum = lanes.into_iter().sum::<f32>();
    for column in vector_width..weight.row_width {
        sum += input[column] * weight.value(row, column);
    }
    sum
}

/// Adds `activation * decoded(weight[row])` to one output row.
///
/// # Safety
///
/// The caller has checked AVX2 support and supplied a validated weight row and
/// output slice whose length equals the logical row width.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn axpy_row(
    weight: &Weight<'_>,
    row: usize,
    activation: f32,
    output: &mut [f32],
) {
    let vector_width = weight.row_width / BLOCK_SIZE * BLOCK_SIZE;
    let scalar_activation = activation;
    let activation = _mm256_set1_ps(activation);
    for block_start in (0..vector_width).step_by(BLOCK_SIZE) {
        let (first, second) = unsafe { decode_block(weight, row, block_start) };
        // SAFETY: loads and stores remain inside the complete output block.
        unsafe {
            let output_first = _mm256_loadu_ps(output.as_ptr().add(block_start));
            let output_second = _mm256_loadu_ps(output.as_ptr().add(block_start + 8));
            _mm256_storeu_ps(
                output.as_mut_ptr().add(block_start),
                _mm256_add_ps(output_first, _mm256_mul_ps(activation, first)),
            );
            _mm256_storeu_ps(
                output.as_mut_ptr().add(block_start + 8),
                _mm256_add_ps(output_second, _mm256_mul_ps(activation, second)),
            );
        }
    }
    for column in vector_width..weight.row_width {
        output[column] += scalar_activation * weight.value(row, column);
    }
}
