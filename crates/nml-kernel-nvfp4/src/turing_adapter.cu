#include "nml_nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <limits>

namespace {

constexpr int32_t kInvalidArgument = 1;
constexpr int32_t kLaunchFailure = 2;
constexpr int kWmmaTile = 16;

int32_t fail(int32_t code, const char *message, char *output, size_t capacity) {
  if (output != nullptr && capacity != 0) {
    std::snprintf(output, capacity, "%s", message);
  }
  return code;
}

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  const uint32_t magnitude = code & 0x07;
  const uint32_t magnitude_bits =
      magnitude < 2
          ? magnitude * 0x3f000000u
          : (((magnitude >> 1) + 126u) << 23) |
                ((magnitude & 1u) << 22);
  const uint32_t sign_bits = static_cast<uint32_t>(code & 0x08) << 28;
  return __uint_as_float(magnitude_bits | sign_bits);
}

__device__ __forceinline__ float decode_e4m3fn(uint8_t bits) {
  const uint32_t exponent = (bits >> 3) & 0x0f;
  const uint32_t fraction = bits & 0x07;
  if ((bits & 0x80) != 0 || (exponent == 0x0f && fraction == 0x07)) {
    return __uint_as_float(0x7fc00000u);
  }
  if (exponent == 0) {
    constexpr uint32_t subnormal_bits[8] = {
        0x00000000u, 0x3b000000u, 0x3b800000u, 0x3bc00000u,
        0x3c000000u, 0x3c200000u, 0x3c400000u, 0x3c600000u,
    };
    return __uint_as_float(subnormal_bits[fraction]);
  }
  return __uint_as_float(((exponent + 120u) << 23) | (fraction << 20));
}

template <typename T> __device__ __forceinline__ float load_float(T value);
template <> __device__ __forceinline__ float load_float(__half value) {
  return __half2float(value);
}
template <> __device__ __forceinline__ float load_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ void store_float(T *output, int64_t index,
                                            float value);
template <>
__device__ __forceinline__ void store_float(__half *output, int64_t index,
                                            float value) {
  output[index] = __float2half_rn(value);
}
template <>
__device__ __forceinline__ void store_float(__nv_bfloat16 *output,
                                            int64_t index, float value) {
  output[index] = __float2bfloat16(value);
}

template <typename Element>
__global__ void linear_kernel(const Element *__restrict__ activation,
                              const uint8_t *__restrict__ payload,
                              const uint8_t *__restrict__ block_scales,
                              const float *__restrict__ global_scale,
                              const Element *__restrict__ bias,
                              Element *__restrict__ output, int64_t rows,
                              int64_t outputs, int64_t inputs) {
  // One warp owns one 16x16 output tile. BF16 inputs are explicitly converted
  // tile-locally to F16 because Turing exposes no BF16 tensor-core operand.
  __shared__ __align__(16) __half left[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half right[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float completed[kWmmaTile * kWmmaTile];

  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kWmmaTile;
  const int64_t output_base = static_cast<int64_t>(blockIdx.x) * kWmmaTile;
  const int lane = threadIdx.x;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaTile, kWmmaTile,
                         kWmmaTile, float>
      accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);

  for (int64_t start = 0; start < inputs; start += kWmmaTile) {
    for (int index = lane; index < kWmmaTile * kWmmaTile; index += 32) {
      const int tile_row = index / kWmmaTile;
      const int tile_column = index % kWmmaTile;
      const int64_t row = row_base + tile_row;
      const int64_t activation_column = start + tile_column;
      left[index] = row < rows && activation_column < inputs
                        ? __float2half_rn(load_float(
                              activation[row * inputs + activation_column]))
                        : __float2half(0.0f);

      // Recipe v3 owns the exact [packed K, N] contraction order required by
      // this B tile. No launch-time transpose or prepared device copy exists.
      const int64_t output_column = output_base + tile_column;
      const int64_t weight_column = start + tile_row;
      float weight = 0.0f;
      if (output_column < outputs && weight_column < inputs) {
        const uint8_t packed =
            payload[(weight_column / 2) * outputs + output_column];
        const uint8_t code =
            static_cast<uint8_t>((packed >> ((weight_column & 1) * 4)) & 0x0f);
        const uint8_t scale_bits =
            block_scales[(weight_column / 16) * outputs + output_column];
        weight = decode_e2m1(code) * decode_e4m3fn(scale_bits) *
                 global_scale[0];
      }
      right[tile_column * kWmmaTile + tile_row] = __float2half_rn(weight);
    }
    __syncwarp();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::row_major>
        left_fragment;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::col_major>
        right_fragment;
    nvcuda::wmma::load_matrix_sync(left_fragment, left, kWmmaTile);
    nvcuda::wmma::load_matrix_sync(right_fragment, right, kWmmaTile);
    nvcuda::wmma::mma_sync(accumulator, left_fragment, right_fragment,
                           accumulator);
    __syncwarp();
  }

  nvcuda::wmma::store_matrix_sync(completed, accumulator, kWmmaTile,
                                  nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile; index += 32) {
    const int64_t row = row_base + index / kWmmaTile;
    const int64_t column = output_base + index % kWmmaTile;
    if (row < rows && column < outputs) {
      const float value = completed[index] +
                          (bias == nullptr ? 0.0f : load_float(bias[column]));
      store_float(output, row * outputs + column, value);
    }
  }
}

template <typename Element>
__global__ void linear_gemv_kernel(
    const Element *__restrict__ activation,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias, Element *__restrict__ output,
    int64_t outputs, int64_t inputs) {
  constexpr int kWarpsPerBlock = 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t output_column =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (output_column >= outputs) {
    return;
  }

  const int64_t packed_width = (inputs + 1) / 2;
  const float tensor_scale = global_scale[0];
  float accumulator = 0.0f;
  for (int64_t pair = lane; pair < packed_width; pair += 32) {
    const int64_t even = pair * 2;
    const uint8_t packed = payload[pair * outputs + output_column];
    float block_scale = 0.0f;
    if ((lane & 7) == 0) {
      block_scale = decode_e4m3fn(
          block_scales[(even / 16) * outputs + output_column]);
    }
    block_scale = __shfl_sync(0xffffffffu, block_scale, lane & ~7);
    const float scale = block_scale * tensor_scale;
    accumulator += load_float(activation[even]) *
                   decode_e2m1(packed & 0x0f) * scale;
    if (even + 1 < inputs) {
      accumulator += load_float(activation[even + 1]) *
                     decode_e2m1(packed >> 4) * scale;
    }
  }
  for (int offset = 16; offset != 0; offset >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
  }
  if (lane == 0) {
    accumulator += bias == nullptr ? 0.0f : load_float(bias[output_column]);
    store_float(output, output_column, accumulator);
  }
}

template <typename Index, typename Element>
__global__ void embedding_kernel(
    const Index *__restrict__ indices, const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale, Element *__restrict__ output,
    int64_t rows, int64_t vocabulary, int64_t width) {
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          static_cast<int64_t>(threadIdx.x);
  const int64_t extent = rows * width;
  if (element >= extent) {
    return;
  }
  const int64_t row = element / width;
  const int64_t column = element % width;
  const int64_t token = static_cast<int64_t>(indices[row]);
  if (token < 0 || token >= vocabulary) {
    // Host-side semantic validation cannot inspect device-resident indices.
    // Write a deterministic value; the semantic graph requires valid token IDs
    // and product callers validate them before upload.
    store_float(output, element, 0.0f);
    return;
  }
  const int64_t packed_width = (width + 1) / 2;
  const int64_t scale_width = (width + 15) / 16;
  const uint8_t packed = payload[token * packed_width + column / 2];
  const uint8_t code =
      static_cast<uint8_t>((packed >> ((column & 1) * 4)) & 0x0f);
  const uint8_t scale_bits =
      block_scales[token * scale_width + column / 16];
  store_float(output, element,
              decode_e2m1(code) * decode_e4m3fn(scale_bits) *
                  global_scale[0]);
}

__device__ __forceinline__ float compact_contraction_value(
    const uint8_t *payload, const uint8_t *block_scales,
    const float *global_scale, int64_t expert, int64_t output, int64_t input,
    int64_t outputs, int64_t inputs) {
  const int64_t packed_width = (inputs + 1) / 2;
  const int64_t scale_width = (inputs + 15) / 16;
  const uint8_t packed =
      payload[(expert * packed_width + input / 2) * outputs + output];
  const uint8_t code =
      static_cast<uint8_t>((packed >> ((input & 1) * 4)) & 0x0f);
  return decode_e2m1(code) *
         decode_e4m3fn(block_scales[
             (expert * scale_width + input / 16) * outputs + output]) *
         global_scale[0];
}

template <typename Element>
__global__ void expert_gate_up_kernel(
    const Element *__restrict__ hidden,
    const int32_t *__restrict__ sorted_assignments,
    const int32_t *__restrict__ block_experts,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias, Element *__restrict__ activated,
    int64_t assignments, int64_t schedule_positions, int64_t hidden_size,
    int64_t intermediate_size, int64_t experts, int64_t experts_per_token) {
  __shared__ __align__(16) __half left[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half gate_right[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half up_right[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float gate_complete[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float up_complete[kWmmaTile * kWmmaTile];

  const int lane = threadIdx.x;
  const int64_t schedule_base = static_cast<int64_t>(blockIdx.y) * kWmmaTile;
  const int64_t output_base = static_cast<int64_t>(blockIdx.x) * kWmmaTile;
  const int32_t expert = block_experts[blockIdx.y];
  // Schedule capacity is allowed to exceed runtime work. Invalid padding
  // blocks must leave before initializing fragments or touching compact
  // expert storage; masking their eventual stores is not a performance guard.
  if (expert < 0 || expert >= experts) {
    return;
  }
  const int64_t logical_width = 2 * intermediate_size;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaTile, kWmmaTile,
                         kWmmaTile, float>
      gate_accumulator;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaTile, kWmmaTile,
                         kWmmaTile, float>
      up_accumulator;
  nvcuda::wmma::fill_fragment(gate_accumulator, 0.0f);
  nvcuda::wmma::fill_fragment(up_accumulator, 0.0f);

  for (int64_t start = 0; start < hidden_size; start += kWmmaTile) {
    for (int index = lane; index < kWmmaTile * kWmmaTile; index += 32) {
      const int tile_row = index / kWmmaTile;
      const int tile_column = index % kWmmaTile;
      const int64_t slot = schedule_base + tile_row;
      const int32_t assignment =
          slot < schedule_positions ? sorted_assignments[slot] : -1;
      const int64_t input_column = start + tile_column;
      left[index] = assignment >= 0 && assignment < assignments &&
                            input_column < hidden_size
                        ? __float2half_rn(load_float(
                              hidden[(assignment / experts_per_token) *
                                         hidden_size +
                                     input_column]))
                        : __float2half(0.0f);

      const int64_t intermediate = output_base + tile_column;
      const int64_t weight_input = start + tile_row;
      float gate_weight = 0.0f;
      float up_weight = 0.0f;
      if (expert >= 0 && expert < experts && intermediate < intermediate_size &&
          weight_input < hidden_size) {
        gate_weight = compact_contraction_value(
            payload, block_scales, global_scale, expert, 2 * intermediate,
            weight_input, logical_width, hidden_size);
        up_weight = compact_contraction_value(
            payload, block_scales, global_scale, expert,
            2 * intermediate + 1, weight_input, logical_width, hidden_size);
      }
      const int right_index = tile_column * kWmmaTile + tile_row;
      gate_right[right_index] = __float2half_rn(gate_weight);
      up_right[right_index] = __float2half_rn(up_weight);
    }
    __syncwarp();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::row_major>
        left_fragment;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::col_major>
        gate_fragment;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::col_major>
        up_fragment;
    nvcuda::wmma::load_matrix_sync(left_fragment, left, kWmmaTile);
    nvcuda::wmma::load_matrix_sync(gate_fragment, gate_right, kWmmaTile);
    nvcuda::wmma::load_matrix_sync(up_fragment, up_right, kWmmaTile);
    nvcuda::wmma::mma_sync(gate_accumulator, left_fragment, gate_fragment,
                           gate_accumulator);
    nvcuda::wmma::mma_sync(up_accumulator, left_fragment, up_fragment,
                           up_accumulator);
    __syncwarp();
  }
  nvcuda::wmma::store_matrix_sync(gate_complete, gate_accumulator, kWmmaTile,
                                  nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync(up_complete, up_accumulator, kWmmaTile,
                                  nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile; index += 32) {
    const int64_t slot = schedule_base + index / kWmmaTile;
    const int32_t assignment =
        slot < schedule_positions ? sorted_assignments[slot] : -1;
    const int64_t intermediate = output_base + index % kWmmaTile;
    if (assignment >= 0 && assignment < assignments &&
        intermediate < intermediate_size && expert >= 0 && expert < experts) {
      const int64_t bias_base = static_cast<int64_t>(expert) * logical_width;
      const float gate = fminf(
          gate_complete[index] + load_float(bias[bias_base + 2 * intermediate]),
          7.0f);
      const float up = fminf(
          fmaxf(up_complete[index] +
                    load_float(bias[bias_base + 2 * intermediate + 1]),
                -7.0f),
          7.0f);
      const float swish = gate / (1.0f + expf(-1.702f * gate));
      store_float(activated, static_cast<int64_t>(assignment) *
                                 intermediate_size + intermediate,
                  (up + 1.0f) * swish);
    }
  }
}

template <typename Element>
__global__ void expert_down_kernel(
    const Element *__restrict__ activated,
    const int32_t *__restrict__ sorted_assignments,
    const int32_t *__restrict__ block_experts,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias,
    const Element *__restrict__ routing_weights,
    Element *__restrict__ weighted_output, int64_t assignments,
    int64_t schedule_positions, int64_t intermediate_size,
    int64_t hidden_size, int64_t experts) {
  __shared__ __align__(16) __half left[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half right[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float completed[kWmmaTile * kWmmaTile];
  const int lane = threadIdx.x;
  const int64_t schedule_base = static_cast<int64_t>(blockIdx.y) * kWmmaTile;
  const int64_t output_base = static_cast<int64_t>(blockIdx.x) * kWmmaTile;
  const int32_t expert = block_experts[blockIdx.y];
  if (expert < 0 || expert >= experts) {
    return;
  }
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaTile, kWmmaTile,
                         kWmmaTile, float>
      accumulator;
  nvcuda::wmma::fill_fragment(accumulator, 0.0f);
  for (int64_t start = 0; start < intermediate_size; start += kWmmaTile) {
    for (int index = lane; index < kWmmaTile * kWmmaTile; index += 32) {
      const int tile_row = index / kWmmaTile;
      const int tile_column = index % kWmmaTile;
      const int64_t slot = schedule_base + tile_row;
      const int32_t assignment =
          slot < schedule_positions ? sorted_assignments[slot] : -1;
      const int64_t input_column = start + tile_column;
      left[index] = assignment >= 0 && assignment < assignments &&
                            input_column < intermediate_size
                        ? __float2half_rn(load_float(
                              activated[static_cast<int64_t>(assignment) *
                                            intermediate_size +
                                        input_column]))
                        : __float2half(0.0f);
      const int64_t output_column = output_base + tile_column;
      const int64_t weight_input = start + tile_row;
      float weight = 0.0f;
      if (expert >= 0 && expert < experts && output_column < hidden_size &&
          weight_input < intermediate_size) {
        weight = compact_contraction_value(
            payload, block_scales, global_scale, expert, output_column,
            weight_input, hidden_size, intermediate_size);
      }
      right[tile_column * kWmmaTile + tile_row] = __float2half_rn(weight);
    }
    __syncwarp();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::row_major>
        left_fragment;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaTile, kWmmaTile,
                           kWmmaTile, __half, nvcuda::wmma::col_major>
        right_fragment;
    nvcuda::wmma::load_matrix_sync(left_fragment, left, kWmmaTile);
    nvcuda::wmma::load_matrix_sync(right_fragment, right, kWmmaTile);
    nvcuda::wmma::mma_sync(accumulator, left_fragment, right_fragment,
                           accumulator);
    __syncwarp();
  }
  nvcuda::wmma::store_matrix_sync(completed, accumulator, kWmmaTile,
                                  nvcuda::wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < kWmmaTile * kWmmaTile; index += 32) {
    const int64_t slot = schedule_base + index / kWmmaTile;
    const int32_t assignment =
        slot < schedule_positions ? sorted_assignments[slot] : -1;
    const int64_t output_column = output_base + index % kWmmaTile;
    if (assignment >= 0 && assignment < assignments &&
        output_column < hidden_size && expert >= 0 && expert < experts) {
      const float value =
          (completed[index] +
           load_float(bias[static_cast<int64_t>(expert) * hidden_size +
                           output_column])) *
          load_float(routing_weights[assignment]);
      store_float(weighted_output,
                  static_cast<int64_t>(assignment) * hidden_size +
                      output_column,
                  value);
    }
  }
}

template <typename Element>
__global__ void expert_gate_up_gemv_kernel(
    const Element *__restrict__ hidden,
    const int32_t *__restrict__ sorted_assignments,
    const int32_t *__restrict__ block_experts,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias, Element *__restrict__ activated,
    int64_t assignments, int64_t hidden_size, int64_t intermediate_size,
    int64_t experts, int64_t experts_per_token) {
  constexpr int kInputTile = 128;
  __shared__ float activation_tile[kInputTile];
  const int thread = threadIdx.x;
  const int32_t expert = block_experts[blockIdx.y];
  const int32_t assignment = sorted_assignments[blockIdx.y * kWmmaTile];
  if (expert < 0 || expert >= experts || assignment < 0 ||
      assignment >= assignments) {
    return;
  }
  const int64_t intermediate =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + thread;
  const int64_t logical_rows = 2 * intermediate_size;
  const int64_t packed_width = (hidden_size + 1) / 2;
  const int64_t scale_width = (hidden_size + 15) / 16;
  const float tensor_scale = global_scale[0];
  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  for (int64_t start = 0; start < hidden_size; start += kInputTile) {
    const int64_t load_column = start + thread;
    activation_tile[thread] =
        load_column < hidden_size
            ? load_float(hidden[(assignment / experts_per_token) * hidden_size +
                                load_column])
            : 0.0f;
    __syncthreads();
    if (intermediate < intermediate_size) {
      for (int block = 0;
           block < kInputTile && start + block < hidden_size; block += 16) {
        const int64_t scale_column = (start + block) / 16;
        const float gate_scale = decode_e4m3fn(block_scales[
            (static_cast<int64_t>(expert) * scale_width + scale_column) *
                logical_rows +
            2 * intermediate]);
        const float up_scale = decode_e4m3fn(block_scales[
            (static_cast<int64_t>(expert) * scale_width + scale_column) *
                logical_rows +
            2 * intermediate + 1]);
        for (int lane = 0;
             lane < 16 && block + lane < kInputTile &&
             start + block + lane < hidden_size;
             ++lane) {
          const int offset = block + lane;
          const int64_t input_column = start + offset;
          const int64_t packed_column = input_column / 2;
          const int shift = static_cast<int>((input_column & 1) * 4);
          const uint8_t gate_packed = payload[
              (static_cast<int64_t>(expert) * packed_width + packed_column) *
                  logical_rows +
              2 * intermediate];
          const uint8_t up_packed = payload[
              (static_cast<int64_t>(expert) * packed_width + packed_column) *
                  logical_rows +
              2 * intermediate + 1];
          const float activation = tensor_scale * activation_tile[offset];
          gate_accumulator += decode_e2m1((gate_packed >> shift) & 0x0f) *
                              gate_scale * activation;
          up_accumulator += decode_e2m1((up_packed >> shift) & 0x0f) *
                            up_scale * activation;
        }
      }
    }
    __syncthreads();
  }
  if (intermediate < intermediate_size) {
    const int64_t bias_base = static_cast<int64_t>(expert) *
                              (2 * intermediate_size) + 2 * intermediate;
    const float gate =
        fminf(gate_accumulator + load_float(bias[bias_base]), 7.0f);
    const float up = fminf(
        fmaxf(up_accumulator + load_float(bias[bias_base + 1]), -7.0f),
        7.0f);
    const float swish = gate / (1.0f + expf(-1.702f * gate));
    store_float(activated,
                static_cast<int64_t>(assignment) * intermediate_size +
                    intermediate,
                (up + 1.0f) * swish);
  }
}

template <typename Element>
__global__ void expert_down_gemv_kernel(
    const Element *__restrict__ activated,
    const int32_t *__restrict__ sorted_assignments,
    const int32_t *__restrict__ block_experts,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias,
    const Element *__restrict__ routing_weights,
    Element *__restrict__ weighted_output, int64_t assignments,
    int64_t intermediate_size, int64_t hidden_size, int64_t experts) {
  constexpr int kInputTile = 128;
  __shared__ float activation_tile[kInputTile];
  const int thread = threadIdx.x;
  const int32_t expert = block_experts[blockIdx.y];
  const int32_t assignment = sorted_assignments[blockIdx.y * kWmmaTile];
  if (expert < 0 || expert >= experts || assignment < 0 ||
      assignment >= assignments) {
    return;
  }
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + thread;
  const int64_t even = pair * 2;
  const int64_t odd = even + 1;
  const int64_t packed_width = (intermediate_size + 1) / 2;
  const int64_t scale_width = (intermediate_size + 15) / 16;
  const float tensor_scale = global_scale[0];
  float even_accumulator = 0.0f;
  float odd_accumulator = 0.0f;
  for (int64_t start = 0; start < intermediate_size; start += kInputTile) {
    const int64_t load_column = start + thread;
    activation_tile[thread] =
        load_column < intermediate_size
            ? load_float(activated[static_cast<int64_t>(assignment) *
                                       intermediate_size +
                                   load_column])
            : 0.0f;
    __syncthreads();
    if (even < hidden_size) {
      for (int block = 0;
           block < kInputTile && start + block < intermediate_size;
           block += 16) {
        const int64_t scale_column = (start + block) / 16;
        const float even_scale = decode_e4m3fn(block_scales[
            (static_cast<int64_t>(expert) * scale_width + scale_column) *
                hidden_size +
            even]);
        const float odd_scale = odd < hidden_size
                                    ? decode_e4m3fn(block_scales[
                                          (static_cast<int64_t>(expert) *
                                               scale_width +
                                           scale_column) *
                                                  hidden_size +
                                              odd])
                                    : 0.0f;
        for (int lane = 0;
             lane < 16 && block + lane < kInputTile &&
             start + block + lane < intermediate_size;
             ++lane) {
          const int offset = block + lane;
          const int64_t input_column = start + offset;
          const int64_t packed_column = input_column / 2;
          const int shift = static_cast<int>((input_column & 1) * 4);
          const uint8_t even_packed = payload[
              (static_cast<int64_t>(expert) * packed_width + packed_column) *
                  hidden_size +
              even];
          const float activation = tensor_scale * activation_tile[offset];
          even_accumulator += decode_e2m1((even_packed >> shift) & 0x0f) *
                              even_scale * activation;
          if (odd < hidden_size) {
            const uint8_t odd_packed = payload[
                (static_cast<int64_t>(expert) * packed_width + packed_column) *
                    hidden_size +
                odd];
            odd_accumulator += decode_e2m1((odd_packed >> shift) & 0x0f) *
                               odd_scale * activation;
          }
        }
      }
    }
    __syncthreads();
  }
  const float route = load_float(routing_weights[assignment]);
  const int64_t bias_base = static_cast<int64_t>(expert) * hidden_size;
  const int64_t output_base = static_cast<int64_t>(assignment) * hidden_size;
  if (even < hidden_size) {
    store_float(weighted_output, output_base + even,
                (even_accumulator + load_float(bias[bias_base + even])) *
                    route);
  }
  if (odd < hidden_size) {
    store_float(weighted_output, output_base + odd,
                (odd_accumulator + load_float(bias[bias_base + odd])) * route);
  }
}

bool valid_geometry(int64_t first, int64_t second, int64_t third) {
  return first > 0 && second > 0 && third > 0 &&
         first <= std::numeric_limits<int32_t>::max() &&
         second <= std::numeric_limits<int32_t>::max() &&
         third <= std::numeric_limits<int32_t>::max();
}

int32_t require_turing(char *message, size_t capacity) {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return fail(kLaunchFailure, cudaGetErrorString(status), message, capacity);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return fail(kLaunchFailure, cudaGetErrorString(status), message, capacity);
  }
  if (properties.major != 7 || properties.minor != 5) {
    return fail(kInvalidArgument,
                "the NVFP4 Turing adapter requires compute capability 7.5",
                message, capacity);
  }
  return 0;
}

int32_t launch_result(char *message, size_t capacity) {
  const cudaError_t status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? 0
             : fail(kLaunchFailure, cudaGetErrorString(status), message,
                    capacity);
}

} // namespace

extern "C" int32_t nml_nvfp4_turing_linear(
    const NmlNvFp4Linear *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr || request->struct_size < sizeof(NmlNvFp4Linear)) {
    return fail(kInvalidArgument, "truncated NVFP4 linear request",
                error_message, error_message_capacity);
  }
  if (request->activation == nullptr || request->payload == nullptr ||
      request->block_scales == nullptr || request->global_scale == nullptr ||
      request->output == nullptr || request->stream == nullptr ||
      !valid_geometry(request->rows, request->outputs, request->inputs)) {
    return fail(kInvalidArgument, "invalid NVFP4 linear request",
                error_message, error_message_capacity);
  }
  if (const int32_t status =
          require_turing(error_message, error_message_capacity)) {
    return status;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->rows == 1) {
    constexpr uint32_t kWarpsPerBlock = 8;
    const uint32_t blocks =
        static_cast<uint32_t>((request->outputs + kWarpsPerBlock - 1) /
                              kWarpsPerBlock);
#define NML_LAUNCH_LINEAR_GEMV(Element)                                       \
  linear_gemv_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(             \
      static_cast<const Element *>(request->activation), request->payload,    \
      request->block_scales, request->global_scale,                           \
      static_cast<const Element *>(request->bias),                            \
      static_cast<Element *>(request->output), request->outputs,              \
      request->inputs)
    if (request->dtype == NML_NVFP4_F16) {
      NML_LAUNCH_LINEAR_GEMV(__half);
    } else if (request->dtype == NML_NVFP4_BF16) {
      NML_LAUNCH_LINEAR_GEMV(__nv_bfloat16);
    } else {
      return fail(kInvalidArgument, "NVFP4 linear supports F16 and BF16 only",
                  error_message, error_message_capacity);
    }
#undef NML_LAUNCH_LINEAR_GEMV
    return launch_result(error_message, error_message_capacity);
  }

  const dim3 grid(static_cast<uint32_t>((request->outputs + 15) / 16),
                  static_cast<uint32_t>((request->rows + 15) / 16));
  if (request->dtype == NML_NVFP4_F16) {
    linear_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const __half *>(request->activation), request->payload,
        request->block_scales, request->global_scale,
        static_cast<const __half *>(request->bias),
        static_cast<__half *>(request->output), request->rows, request->outputs,
        request->inputs);
  } else if (request->dtype == NML_NVFP4_BF16) {
    linear_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(request->activation),
        request->payload, request->block_scales, request->global_scale,
        static_cast<const __nv_bfloat16 *>(request->bias),
        static_cast<__nv_bfloat16 *>(request->output), request->rows,
        request->outputs, request->inputs);
  } else {
    return fail(kInvalidArgument, "NVFP4 linear supports F16 and BF16 only",
                error_message, error_message_capacity);
  }
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_turing_embedding(
    const NmlNvFp4Embedding *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4Embedding)) {
    return fail(kInvalidArgument, "truncated NVFP4 embedding request",
                error_message, error_message_capacity);
  }
  if (request->indices == nullptr || request->payload == nullptr ||
      request->block_scales == nullptr || request->global_scale == nullptr ||
      request->output == nullptr || request->stream == nullptr ||
      !valid_geometry(request->rows, request->vocabulary, request->width)) {
    return fail(kInvalidArgument, "invalid NVFP4 embedding request",
                error_message, error_message_capacity);
  }
  if (request->rows > std::numeric_limits<int64_t>::max() / request->width) {
    return fail(kInvalidArgument, "NVFP4 embedding output extent overflows",
                error_message, error_message_capacity);
  }
  if (const int32_t status =
          require_turing(error_message, error_message_capacity)) {
    return status;
  }
  const int64_t extent = request->rows * request->width;
  const uint32_t blocks = static_cast<uint32_t>((extent + 255) / 256);
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
#define NML_LAUNCH_EMBEDDING(Index, Element)                                  \
  embedding_kernel<<<blocks, 256, 0, stream>>>(                              \
      static_cast<const Index *>(request->indices), request->payload,         \
      request->block_scales, request->global_scale,                           \
      static_cast<Element *>(request->output), request->rows,                 \
      request->vocabulary, request->width)
  if (request->dtype == NML_NVFP4_F16 && request->indices_are_i64 == 0) {
    NML_LAUNCH_EMBEDDING(int32_t, __half);
  } else if (request->dtype == NML_NVFP4_F16) {
    NML_LAUNCH_EMBEDDING(int64_t, __half);
  } else if (request->dtype == NML_NVFP4_BF16 &&
             request->indices_are_i64 == 0) {
    NML_LAUNCH_EMBEDDING(int32_t, __nv_bfloat16);
  } else if (request->dtype == NML_NVFP4_BF16) {
    NML_LAUNCH_EMBEDDING(int64_t, __nv_bfloat16);
  } else {
    return fail(kInvalidArgument,
                "NVFP4 embedding supports F16 and BF16 only", error_message,
                error_message_capacity);
  }
#undef NML_LAUNCH_EMBEDDING
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_turing_expert_gate_up(
    const NmlNvFp4ExpertGateUp *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4ExpertGateUp) ||
      request->hidden == nullptr || request->sorted_assignments == nullptr ||
      request->block_experts == nullptr || request->payload == nullptr ||
      request->block_scales == nullptr || request->global_scale == nullptr ||
      request->bias == nullptr || request->activated == nullptr ||
      request->stream == nullptr || request->block_size != kWmmaTile ||
      !valid_geometry(request->assignments, request->hidden_size,
                      request->intermediate_size) ||
      request->tokens <= 0 || request->experts <= 0 ||
      request->experts_per_token <= 0 ||
      request->tokens > std::numeric_limits<int64_t>::max() /
                            request->experts_per_token ||
      request->assignments != request->tokens * request->experts_per_token ||
      request->schedule_blocks <= 0 ||
      request->schedule_blocks >
          std::numeric_limits<int64_t>::max() / kWmmaTile ||
      request->schedule_positions != request->schedule_blocks * kWmmaTile) {
    return fail(kInvalidArgument, "invalid NVFP4 expert gate/up request",
                error_message, error_message_capacity);
  }
  if (const int32_t status =
          require_turing(error_message, error_message_capacity)) {
    return status;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->tokens == 1) {
    constexpr uint32_t kThreads = 128;
    const dim3 grid(
        static_cast<uint32_t>((request->intermediate_size + kThreads - 1) /
                              kThreads),
        static_cast<uint32_t>(request->schedule_blocks));
#define NML_LAUNCH_GATE_UP_GEMV(Element)                                      \
  expert_gate_up_gemv_kernel<<<grid, kThreads, 0, stream>>>(                  \
      static_cast<const Element *>(request->hidden),                           \
      request->sorted_assignments, request->block_experts, request->payload,   \
      request->block_scales, request->global_scale,                            \
      static_cast<const Element *>(request->bias),                             \
      static_cast<Element *>(request->activated), request->assignments,        \
      request->hidden_size, request->intermediate_size, request->experts,      \
      request->experts_per_token)
    if (request->dtype == NML_NVFP4_F16) {
      NML_LAUNCH_GATE_UP_GEMV(__half);
    } else if (request->dtype == NML_NVFP4_BF16) {
      NML_LAUNCH_GATE_UP_GEMV(__nv_bfloat16);
    } else {
      return fail(kInvalidArgument, "NVFP4 experts support F16 and BF16 only",
                  error_message, error_message_capacity);
    }
#undef NML_LAUNCH_GATE_UP_GEMV
    return launch_result(error_message, error_message_capacity);
  }

  const dim3 grid(
      static_cast<uint32_t>((request->intermediate_size + 15) / 16),
      static_cast<uint32_t>(request->schedule_blocks));
#define NML_LAUNCH_GATE_UP(Element)                                            \
  expert_gate_up_kernel<<<grid, 32, 0, stream>>>(                             \
      static_cast<const Element *>(request->hidden),                           \
      request->sorted_assignments, request->block_experts, request->payload,   \
      request->block_scales, request->global_scale,                            \
      static_cast<const Element *>(request->bias),                             \
      static_cast<Element *>(request->activated), request->assignments,        \
      request->schedule_positions, request->hidden_size,                       \
      request->intermediate_size, request->experts,                            \
      request->experts_per_token)
  if (request->dtype == NML_NVFP4_F16) {
    NML_LAUNCH_GATE_UP(__half);
  } else if (request->dtype == NML_NVFP4_BF16) {
    NML_LAUNCH_GATE_UP(__nv_bfloat16);
  } else {
    return fail(kInvalidArgument, "NVFP4 experts support F16 and BF16 only",
                error_message, error_message_capacity);
  }
#undef NML_LAUNCH_GATE_UP
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_turing_expert_down(
    const NmlNvFp4ExpertDown *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4ExpertDown) ||
      request->activated == nullptr ||
      request->sorted_assignments == nullptr ||
      request->block_experts == nullptr || request->payload == nullptr ||
      request->block_scales == nullptr || request->global_scale == nullptr ||
      request->bias == nullptr || request->routing_weights == nullptr ||
      request->weighted_output == nullptr || request->stream == nullptr ||
      request->block_size != kWmmaTile ||
      !valid_geometry(request->assignments, request->intermediate_size,
                      request->hidden_size) ||
      request->experts <= 0 || request->experts_per_token <= 0 ||
      request->schedule_blocks <= 0 ||
      request->schedule_blocks >
          std::numeric_limits<int64_t>::max() / kWmmaTile ||
      request->schedule_positions != request->schedule_blocks * kWmmaTile) {
    return fail(kInvalidArgument, "invalid NVFP4 expert down request",
                error_message, error_message_capacity);
  }
  if (const int32_t status =
          require_turing(error_message, error_message_capacity)) {
    return status;
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->assignments == request->experts_per_token) {
    constexpr uint32_t kThreads = 128;
    const int64_t pairs = (request->hidden_size + 1) / 2;
    const dim3 grid(
        static_cast<uint32_t>((pairs + kThreads - 1) / kThreads),
        static_cast<uint32_t>(request->schedule_blocks));
#define NML_LAUNCH_DOWN_GEMV(Element)                                         \
  expert_down_gemv_kernel<<<grid, kThreads, 0, stream>>>(                     \
      static_cast<const Element *>(request->activated),                        \
      request->sorted_assignments, request->block_experts, request->payload,   \
      request->block_scales, request->global_scale,                            \
      static_cast<const Element *>(request->bias),                             \
      static_cast<const Element *>(request->routing_weights),                  \
      static_cast<Element *>(request->weighted_output), request->assignments,  \
      request->intermediate_size, request->hidden_size, request->experts)
    if (request->dtype == NML_NVFP4_F16) {
      NML_LAUNCH_DOWN_GEMV(__half);
    } else if (request->dtype == NML_NVFP4_BF16) {
      NML_LAUNCH_DOWN_GEMV(__nv_bfloat16);
    } else {
      return fail(kInvalidArgument, "NVFP4 experts support F16 and BF16 only",
                  error_message, error_message_capacity);
    }
#undef NML_LAUNCH_DOWN_GEMV
    return launch_result(error_message, error_message_capacity);
  }

  const dim3 grid(static_cast<uint32_t>((request->hidden_size + 15) / 16),
                  static_cast<uint32_t>(request->schedule_blocks));
#define NML_LAUNCH_DOWN(Element)                                               \
  expert_down_kernel<<<grid, 32, 0, stream>>>(                                \
      static_cast<const Element *>(request->activated),                        \
      request->sorted_assignments, request->block_experts, request->payload,   \
      request->block_scales, request->global_scale,                            \
      static_cast<const Element *>(request->bias),                             \
      static_cast<const Element *>(request->routing_weights),                  \
      static_cast<Element *>(request->weighted_output), request->assignments,  \
      request->schedule_positions, request->intermediate_size,                 \
      request->hidden_size, request->experts)
  if (request->dtype == NML_NVFP4_F16) {
    NML_LAUNCH_DOWN(__half);
  } else if (request->dtype == NML_NVFP4_BF16) {
    NML_LAUNCH_DOWN(__nv_bfloat16);
  } else {
    return fail(kInvalidArgument, "NVFP4 experts support F16 and BF16 only",
                error_message, error_message_capacity);
  }
#undef NML_LAUNCH_DOWN
  return launch_result(error_message, error_message_capacity);
}
