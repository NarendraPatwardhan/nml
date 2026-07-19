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
__device__ __forceinline__ float round_to_element(float value) {
  Element rounded;
  store_float(&rounded, 0, value);
  return load_float(rounded);
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
  // tile-locally to F16 because CUDA exposes no BF16 tensor-core operand.
  __shared__ __align__(16) __half left[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) __half right[kWmmaTile * kWmmaTile];
  __shared__ __align__(16) float completed[kWmmaTile * kWmmaTile];

  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kWmmaTile;
  const int64_t output_base = static_cast<int64_t>(blockIdx.x) * kWmmaTile;
  const int lane = threadIdx.x;
  const int64_t packed_width = (inputs + 1) / 2;
  const int64_t scale_width = (inputs + 15) / 16;

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

      // WMMA's column-major B tile uses [K, N]. The checkpoint owns [N, K],
      // so its natural logical row is already the required matrix column.
      const int64_t output_column = output_base + tile_column;
      const int64_t weight_column = start + tile_row;
      float weight = 0.0f;
      if (output_column < outputs && weight_column < inputs) {
        const uint8_t packed =
            payload[output_column * packed_width + weight_column / 2];
        const uint8_t code =
            static_cast<uint8_t>((packed >> ((weight_column & 1) * 4)) & 0x0f);
        const uint8_t scale_bits =
            block_scales[output_column * scale_width + weight_column / 16];
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

template <typename Element, int WarpsPerBlock>
__global__ void linear_gemv_kernel(
    const Element *__restrict__ activation,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias, Element *__restrict__ output,
    int64_t outputs, int64_t inputs) {
  static_assert(WarpsPerBlock == 4 || WarpsPerBlock == 8);
  constexpr int kInputTile = WarpsPerBlock * 32;
  __shared__ float activation_tile[kInputTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t output_column =
      static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp;
  const bool valid_output = output_column < outputs;

  const int64_t packed_width = (inputs + 1) / 2;
  const int64_t scale_width = (inputs + 15) / 16;
  const float tensor_scale = global_scale[0];
  float accumulator = 0.0f;
  for (int64_t start = 0; start < inputs; start += kInputTile) {
    const int64_t activation_column = start + threadIdx.x;
    activation_tile[threadIdx.x] =
        activation_column < inputs ? load_float(activation[activation_column])
                                   : 0.0f;
    __syncthreads();
    const int64_t remaining = inputs - start;
    const int64_t tile_elements =
        remaining < kInputTile ? remaining : kInputTile;
    const int64_t tile_pairs = (tile_elements + 1) / 2;
    for (int64_t tile_pair = lane; tile_pair < tile_pairs; tile_pair += 32) {
      const int64_t even = start + tile_pair * 2;
      const int64_t pair = even / 2;
      const uint8_t packed =
          valid_output ? payload[output_column * packed_width + pair] : 0;
      float block_scale = 0.0f;
      if (valid_output && (lane & 7) == 0) {
        block_scale = decode_e4m3fn(
            block_scales[output_column * scale_width + even / 16]);
      }
      block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
      const float scale = block_scale * tensor_scale;
      accumulator += activation_tile[tile_pair * 2] *
                     decode_e2m1(packed & 0x0f) * scale;
      if (even + 1 < inputs) {
        accumulator += activation_tile[tile_pair * 2 + 1] *
                       decode_e2m1(packed >> 4) * scale;
      }
    }
    __syncthreads();
  }
  for (int offset = 16; offset != 0; offset >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
  }
  if (lane == 0 && valid_output) {
    accumulator += bias == nullptr ? 0.0f : load_float(bias[output_column]);
    store_float(output, output_column, accumulator);
  }
}

template <typename Element> struct LinearGroup3Device {
  const uint8_t *payloads[3];
  const uint8_t *block_scales[3];
  const float *global_scales[3];
  const Element *biases[3];
  Element *outputs[3];
  int64_t output_widths[3];
};

// Three independent projections of one activation share one output-owner
// launch and one staged activation tile. Projection boundaries are uniform at
// warp granularity for the representation's output widths, so each warp still
// issues naturally coalesced N-major payload transactions.
template <typename Element, int WarpsPerBlock>
__global__ void linear_group3_gemv_kernel(
    const Element *__restrict__ activation, LinearGroup3Device<Element> group,
    int64_t inputs) {
  static_assert(WarpsPerBlock == 4 || WarpsPerBlock == 8);
  constexpr int kInputTile = WarpsPerBlock * 32;
  __shared__ float activation_tile[kInputTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t global_output =
      static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp;
  const int64_t first_end = group.output_widths[0];
  const int64_t second_end = first_end + group.output_widths[1];
  const int64_t total_outputs = second_end + group.output_widths[2];
  const bool valid_output = global_output < total_outputs;
  const int projection = global_output < first_end
                             ? 0
                             : (global_output < second_end ? 1 : 2);
  const int64_t output_column =
      projection == 0
          ? global_output
          : (projection == 1 ? global_output - first_end
                             : global_output - second_end);
  const int64_t packed_width = (inputs + 1) / 2;
  const int64_t scale_width = (inputs + 15) / 16;
  const float tensor_scale =
      valid_output ? group.global_scales[projection][0] : 0.0f;
  float accumulator = 0.0f;
  for (int64_t start = 0; start < inputs; start += kInputTile) {
    const int64_t activation_column = start + threadIdx.x;
    activation_tile[threadIdx.x] =
        activation_column < inputs ? load_float(activation[activation_column])
                                   : 0.0f;
    __syncthreads();
    const int64_t remaining = inputs - start;
    const int64_t tile_elements =
        remaining < kInputTile ? remaining : kInputTile;
    const int64_t tile_pairs = (tile_elements + 1) / 2;
    for (int64_t tile_pair = lane; tile_pair < tile_pairs; tile_pair += 32) {
      const int64_t even = start + tile_pair * 2;
      const int64_t pair = even / 2;
      const uint8_t packed =
          valid_output
              ? group.payloads[projection][output_column * packed_width + pair]
              : 0;
      float block_scale = 0.0f;
      if (valid_output && (lane & 7) == 0) {
        block_scale = decode_e4m3fn(
            group.block_scales[projection]
                              [output_column * scale_width + even / 16]);
      }
      block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
      const float scale = block_scale * tensor_scale;
      accumulator += activation_tile[tile_pair * 2] *
                     decode_e2m1(packed & 0x0f) * scale;
      if (even + 1 < inputs) {
        accumulator += activation_tile[tile_pair * 2 + 1] *
                       decode_e2m1(packed >> 4) * scale;
      }
    }
    __syncthreads();
  }
  for (int offset = 16; offset != 0; offset >>= 1) {
    accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
  }
  if (lane == 0 && valid_output) {
    const Element *bias = group.biases[projection];
    accumulator += bias == nullptr ? 0.0f : load_float(bias[output_column]);
    store_float(group.outputs[projection], output_column, accumulator);
  }
}

// One block owns the complete single-token router. Eight warps share each
// activation tile and retain up to four expert accumulators apiece, covering
// the admitted maximum of 32 experts without rereading the activation once per
// expert. The thread-zero epilogue preserves the semantic activation-dtype
// rounding points around softmax, selection, and renormalization.
template <typename Element>
__global__ void route_top4_kernel(
    const Element *__restrict__ hidden, const Element *__restrict__ weight,
    const Element *__restrict__ bias, int32_t *__restrict__ expert_ids,
    Element *__restrict__ routing_weights, int64_t inputs, int64_t experts) {
  constexpr int kWarps = 8;
  constexpr int kTile = kWarps * 32;
  constexpr int kExpertsPerWarp = 4;
  __shared__ float activation_tile[kTile];
  __shared__ float logits[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float accumulators[kExpertsPerWarp] = {};

  for (int64_t start = 0; start < inputs; start += kTile) {
    const int64_t column = start + threadIdx.x;
    activation_tile[threadIdx.x] =
        column < inputs ? load_float(hidden[column]) : 0.0f;
    __syncthreads();
    const int64_t remaining = inputs - start;
    const int64_t tile_elements = remaining < kTile ? remaining : kTile;
    for (int group = 0; group < kExpertsPerWarp; ++group) {
      const int expert = warp + group * kWarps;
      if (expert >= experts) {
        continue;
      }
      for (int64_t offset = lane; offset < tile_elements; offset += 32) {
        accumulators[group] +=
            activation_tile[offset] *
            load_float(weight[static_cast<int64_t>(expert) * inputs + start +
                              offset]);
      }
    }
    __syncthreads();
  }

  for (int group = 0; group < kExpertsPerWarp; ++group) {
    float value = accumulators[group];
    for (int offset = 16; offset != 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    const int expert = warp + group * kWarps;
    if (lane == 0 && expert < experts) {
      logits[expert] =
          round_to_element<Element>(value + load_float(bias[expert]));
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float maximum = logits[0];
    for (int32_t expert = 1; expert < experts; ++expert) {
      maximum = fmaxf(maximum, logits[expert]);
    }
    float denominator = 0.0f;
    for (int32_t expert = 0; expert < experts; ++expert) {
      denominator += expf(logits[expert] - maximum);
    }
    float probabilities[32];
    for (int32_t expert = 0; expert < experts; ++expert) {
      probabilities[expert] =
          round_to_element<Element>(expf(logits[expert] - maximum) /
                                    denominator);
    }
    float selected_probabilities[4];
    int32_t selected_experts[4];
    for (int slot = 0; slot < 4; ++slot) {
      float best = -__int_as_float(0x7f800000);
      int32_t best_expert = -1;
      for (int32_t expert = 0; expert < experts; ++expert) {
        bool already_selected = false;
        for (int previous = 0; previous < slot; ++previous) {
          already_selected |= selected_experts[previous] == expert;
        }
        if (!already_selected &&
            (probabilities[expert] > best ||
             (probabilities[expert] == best &&
              (best_expert < 0 || expert < best_expert)))) {
          best = probabilities[expert];
          best_expert = expert;
        }
      }
      selected_probabilities[slot] = best;
      selected_experts[slot] = best_expert;
    }
    float selected_sum = 0.0f;
    for (int slot = 0; slot < 4; ++slot) {
      selected_sum += selected_probabilities[slot];
    }
    selected_sum = round_to_element<Element>(selected_sum);
    for (int slot = 0; slot < 4; ++slot) {
      expert_ids[slot] = selected_experts[slot];
      store_float(routing_weights, slot,
                  selected_probabilities[slot] / selected_sum);
    }
  }
}

__device__ __forceinline__ bool candidate_precedes(float left_value,
                                                   int32_t left_index,
                                                   float right_value,
                                                   int32_t right_index) {
  return left_value > right_value ||
         (left_value == right_value && left_index < right_index);
}

// Each block owns 128 consecutive output rows. A warp holds sixteen output
// accumulators while all eight warps share one activation tile, preserving
// N-major coalescing without writing the complete vocabulary-sized logits
// tensor. The block emits its exact best 64 candidates.
template <typename Element>
__global__ void linear_top64_candidates_kernel(
    const Element *__restrict__ activation,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias, float *__restrict__ candidate_values,
    int32_t *__restrict__ candidate_indices, int64_t outputs, int64_t inputs) {
  constexpr int kWarps = 8;
  constexpr int kOutputsPerWarp = 16;
  constexpr int kOutputsPerBlock = kWarps * kOutputsPerWarp;
  constexpr int kInputTile = kWarps * 32;
  __shared__ float activation_tile[kInputTile];
  __shared__ float block_logits[kOutputsPerBlock];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t output_base =
      static_cast<int64_t>(blockIdx.x) * kOutputsPerBlock +
      warp * kOutputsPerWarp;
  const int64_t packed_width = (inputs + 1) / 2;
  const int64_t scale_width = (inputs + 15) / 16;
  const float tensor_scale = global_scale[0];
  float accumulators[kOutputsPerWarp] = {};

  for (int64_t start = 0; start < inputs; start += kInputTile) {
    const int64_t activation_column = start + threadIdx.x;
    activation_tile[threadIdx.x] =
        activation_column < inputs ? load_float(activation[activation_column])
                                   : 0.0f;
    __syncthreads();
    const int64_t remaining = inputs - start;
    const int64_t tile_elements =
        remaining < kInputTile ? remaining : kInputTile;
    const int64_t tile_pairs = (tile_elements + 1) / 2;
#pragma unroll
    for (int local_output = 0; local_output < kOutputsPerWarp;
         ++local_output) {
      const int64_t output = output_base + local_output;
      if (output >= outputs) {
        continue;
      }
      for (int64_t tile_pair = lane; tile_pair < tile_pairs;
           tile_pair += 32) {
        const int64_t even = start + tile_pair * 2;
        const int64_t pair = even / 2;
        const uint8_t packed = payload[output * packed_width + pair];
        float block_scale = 0.0f;
        if ((lane & 7) == 0) {
          block_scale = decode_e4m3fn(
              block_scales[output * scale_width + even / 16]);
        }
        block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
        const float scale = block_scale * tensor_scale;
        accumulators[local_output] +=
            activation_tile[tile_pair * 2] *
            decode_e2m1(packed & 0x0f) * scale;
        if (even + 1 < inputs) {
          accumulators[local_output] +=
              activation_tile[tile_pair * 2 + 1] *
              decode_e2m1(packed >> 4) * scale;
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int local_output = 0; local_output < kOutputsPerWarp;
       ++local_output) {
    float value = accumulators[local_output];
    for (int offset = 16; offset != 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    const int64_t output = output_base + local_output;
    if (lane == 0) {
      float logit = -__int_as_float(0x7f800000);
      if (output < outputs) {
        logit = round_to_element<Element>(
            value + (bias == nullptr ? 0.0f : load_float(bias[output])));
      }
      block_logits[warp * kOutputsPerWarp + local_output] = logit;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int64_t block_output =
        static_cast<int64_t>(blockIdx.x) * kOutputsPerBlock;
    bool selected[kOutputsPerBlock] = {};
    for (int slot = 0; slot < 64; ++slot) {
      float best = -__int_as_float(0x7f800000);
      int32_t best_local = -1;
      int32_t best_index = -1;
      for (int local = 0; local < kOutputsPerBlock; ++local) {
        const int64_t global = block_output + local;
        if (global < outputs && !selected[local] &&
            (best_index < 0 ||
             candidate_precedes(block_logits[local], static_cast<int32_t>(global),
                                best, best_index))) {
          best = block_logits[local];
          best_local = local;
          best_index = static_cast<int32_t>(global);
        }
      }
      const int64_t destination = static_cast<int64_t>(blockIdx.x) * 64 + slot;
      candidate_values[destination] = best;
      candidate_indices[destination] = best_index;
      if (best_local >= 0) {
        selected[best_local] = true;
      }
    }
  }
}

// Merges eight sorted-or-unsorted top-64 lists into one exact top-64 list.
// Every candidate computes its deterministic global rank in parallel. Unique
// vocabulary indices make ranks unique even when logit values tie.
__global__ void merge_top64_kernel(
    const float *__restrict__ input_values,
    const int32_t *__restrict__ input_indices,
    float *__restrict__ output_values, int32_t *__restrict__ output_indices,
    int64_t input_groups) {
  constexpr int kListsPerBlock = 8;
  constexpr int kCandidatesPerList = 64;
  constexpr int kCandidates = kListsPerBlock * kCandidatesPerList;
  __shared__ float values[kCandidates];
  __shared__ int32_t indices[kCandidates];
  const int candidate = threadIdx.x;
  const int64_t first_group =
      static_cast<int64_t>(blockIdx.x) * kListsPerBlock;
  const int64_t source_group = first_group + candidate / kCandidatesPerList;
  const int source_slot = candidate % kCandidatesPerList;
  if (source_group < input_groups) {
    const int64_t source = source_group * kCandidatesPerList + source_slot;
    values[candidate] = input_values[source];
    indices[candidate] = input_indices[source];
  } else {
    values[candidate] = -__int_as_float(0x7f800000);
    indices[candidate] = -1;
  }
  if (candidate < kCandidatesPerList) {
    const int64_t destination =
        static_cast<int64_t>(blockIdx.x) * kCandidatesPerList + candidate;
    output_values[destination] = -__int_as_float(0x7f800000);
    output_indices[destination] = -1;
  }
  __syncthreads();

  const int32_t index = indices[candidate];
  if (index < 0) {
    return;
  }
  const float value = values[candidate];
  int rank = 0;
  for (int other = 0; other < kCandidates; ++other) {
    if (indices[other] >= 0 &&
        candidate_precedes(values[other], indices[other], value, index)) {
      ++rank;
    }
  }
  if (rank < kCandidatesPerList) {
    const int64_t destination =
        static_cast<int64_t>(blockIdx.x) * kCandidatesPerList + rank;
    output_values[destination] = value;
    output_indices[destination] = index;
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

__device__ __forceinline__ float compact_value(
    const uint8_t *payload, const uint8_t *block_scales,
    const float *global_scale, int64_t row, int64_t column, int64_t width) {
  const int64_t packed_width = (width + 1) / 2;
  const int64_t scale_width = (width + 15) / 16;
  const uint8_t packed = payload[row * packed_width + column / 2];
  const uint8_t code =
      static_cast<uint8_t>((packed >> ((column & 1) * 4)) & 0x0f);
  return decode_e2m1(code) *
         decode_e4m3fn(block_scales[row * scale_width + column / 16]) *
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
        const int64_t weight_row =
            static_cast<int64_t>(expert) * hidden_size + weight_input;
        gate_weight = compact_value(payload, block_scales, global_scale,
                                    weight_row, 2 * intermediate,
                                    logical_width);
        up_weight = compact_value(payload, block_scales, global_scale,
                                  weight_row, 2 * intermediate + 1,
                                  logical_width);
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
        const int64_t weight_row =
            static_cast<int64_t>(expert) * intermediate_size + weight_input;
        weight = compact_value(payload, block_scales, global_scale, weight_row,
                               output_column, hidden_size);
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
  const int lane = thread & 31;
  const int32_t expert = block_experts[blockIdx.y];
  const int32_t assignment = sorted_assignments[blockIdx.y * kWmmaTile];
  if (expert < 0 || expert >= experts || assignment < 0 ||
      assignment >= assignments) {
    return;
  }
  const int64_t intermediate =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + thread;
  const int64_t scale_width = (2 * intermediate_size + 15) / 16;
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
      for (int offset = 0; offset < kInputTile && start + offset < hidden_size;
           ++offset) {
        const int64_t weight_row =
            static_cast<int64_t>(expert) * hidden_size + start + offset;
        const uint8_t packed =
            payload[weight_row * intermediate_size + intermediate];
        float block_scale = 0.0f;
        if ((lane & 7) == 0) {
          block_scale = decode_e4m3fn(
              block_scales[weight_row * scale_width + intermediate / 8]);
        }
        block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
        const float scale = block_scale * tensor_scale * activation_tile[offset];
        gate_accumulator += decode_e2m1(packed & 0x0f) * scale;
        up_accumulator += decode_e2m1(packed >> 4) * scale;
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
  const int lane = thread & 31;
  const int32_t expert = block_experts[blockIdx.y];
  const int32_t assignment = sorted_assignments[blockIdx.y * kWmmaTile];
  if (expert < 0 || expert >= experts || assignment < 0 ||
      assignment >= assignments) {
    return;
  }
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + thread;
  const int64_t even = pair * 2;
  const int64_t odd = even + 1;
  const int64_t packed_width = (hidden_size + 1) / 2;
  const int64_t scale_width = (hidden_size + 15) / 16;
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
      for (int offset = 0;
           offset < kInputTile && start + offset < intermediate_size;
           ++offset) {
        const int64_t weight_row =
            static_cast<int64_t>(expert) * intermediate_size + start + offset;
        const uint8_t packed = payload[weight_row * packed_width + pair];
        float block_scale = 0.0f;
        if ((lane & 7) == 0) {
          block_scale = decode_e4m3fn(
              block_scales[weight_row * scale_width + pair / 8]);
        }
        block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
        const float scale = block_scale * tensor_scale * activation_tile[offset];
        even_accumulator += decode_e2m1(packed & 0x0f) * scale;
        if (odd < hidden_size) {
          odd_accumulator += decode_e2m1(packed >> 4) * scale;
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

// Decode already owns a compact top-k route list.  Keeping that list intact
// avoids constructing and interpreting the block-aligned matrix schedule for
// a single activation row.  Threads own consecutive intermediate columns, so
// each K-major checkpoint row is read as contiguous packed bytes.
template <typename Element>
__global__ void direct_expert_gate_up_kernel(
    const Element *__restrict__ hidden,
    const int32_t *__restrict__ expert_ids,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias, Element *__restrict__ activated,
    int64_t routes, int64_t hidden_size, int64_t intermediate_size,
    int64_t local_experts, const int32_t *__restrict__ expert_offset) {
  constexpr int kInputTile = 128;
  __shared__ float activation_tile[kInputTile];
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int64_t route = blockIdx.y;
  if (route >= routes) {
    return;
  }
  const int64_t expert =
      static_cast<int64_t>(expert_ids[route]) - expert_offset[0];
  const bool local_expert = expert >= 0 && expert < local_experts;
  const int64_t intermediate =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + thread;
  const int64_t scale_width = (2 * intermediate_size + 15) / 16;
  const float tensor_scale = global_scale[0];
  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  for (int64_t start = 0; start < hidden_size; start += kInputTile) {
    const int64_t load_column = start + thread;
    activation_tile[thread] =
        load_column < hidden_size ? load_float(hidden[load_column]) : 0.0f;
    __syncthreads();
    if (local_expert && intermediate < intermediate_size) {
      for (int offset = 0; offset < kInputTile && start + offset < hidden_size;
           ++offset) {
        const int64_t weight_row =
            expert * hidden_size + start + offset;
        const uint8_t packed =
            payload[weight_row * intermediate_size + intermediate];
        float block_scale = 0.0f;
        if ((lane & 7) == 0) {
          block_scale = decode_e4m3fn(
              block_scales[weight_row * scale_width + intermediate / 8]);
        }
        block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
        const float scale = block_scale * tensor_scale * activation_tile[offset];
        gate_accumulator += decode_e2m1(packed & 0x0f) * scale;
        up_accumulator += decode_e2m1(packed >> 4) * scale;
      }
    }
    __syncthreads();
  }
  if (intermediate >= intermediate_size) {
    return;
  }
  float value = 0.0f;
  if (local_expert) {
    const int64_t bias_base = expert * (2 * intermediate_size) +
                              2 * intermediate;
    const float gate =
        fminf(gate_accumulator + load_float(bias[bias_base]), 7.0f);
    const float up = fminf(
        fmaxf(up_accumulator + load_float(bias[bias_base + 1]), -7.0f),
        7.0f);
    value = (up + 1.0f) * gate / (1.0f + expf(-1.702f * gate));
  }
  store_float(activated, route * intermediate_size + intermediate, value);
}

// Every output owner accumulates the selected experts and performs routing
// reduction before its one final store.  This removes both the
// [routes, hidden] temporary and the following StableHLO reduction.
template <typename Element>
__global__ void direct_expert_down_kernel(
    const Element *__restrict__ activated,
    const int32_t *__restrict__ expert_ids,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ block_scales,
    const float *__restrict__ global_scale,
    const Element *__restrict__ bias,
    const Element *__restrict__ routing_weights, Element *__restrict__ output,
    int64_t routes, int64_t intermediate_size, int64_t hidden_size,
    int64_t local_experts, const int32_t *__restrict__ expert_offset) {
  constexpr int kInputTile = 128;
  __shared__ float activation_tile[kInputTile];
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + thread;
  const int64_t even = pair * 2;
  const int64_t odd = even + 1;
  const int64_t packed_width = (hidden_size + 1) / 2;
  const int64_t scale_width = (hidden_size + 15) / 16;
  const float tensor_scale = global_scale[0];
  float even_output = 0.0f;
  float odd_output = 0.0f;
  for (int64_t route = 0; route < routes; ++route) {
    const int64_t expert =
        static_cast<int64_t>(expert_ids[route]) - expert_offset[0];
    const bool local_expert = expert >= 0 && expert < local_experts;
    float even_accumulator = 0.0f;
    float odd_accumulator = 0.0f;
    for (int64_t start = 0; start < intermediate_size; start += kInputTile) {
      const int64_t load_column = start + thread;
      activation_tile[thread] =
          local_expert && load_column < intermediate_size
              ? load_float(activated[route * intermediate_size + load_column])
              : 0.0f;
      __syncthreads();
      if (local_expert && even < hidden_size) {
        for (int offset = 0;
             offset < kInputTile && start + offset < intermediate_size;
             ++offset) {
          const int64_t weight_row =
              expert * intermediate_size + start + offset;
          const uint8_t packed = payload[weight_row * packed_width + pair];
          float block_scale = 0.0f;
          if ((lane & 7) == 0) {
            block_scale = decode_e4m3fn(
                block_scales[weight_row * scale_width + pair / 8]);
          }
          block_scale = __shfl_sync(__activemask(), block_scale, lane & ~7);
          const float scale =
              block_scale * tensor_scale * activation_tile[offset];
          even_accumulator += decode_e2m1(packed & 0x0f) * scale;
          if (odd < hidden_size) {
            odd_accumulator += decode_e2m1(packed >> 4) * scale;
          }
        }
      }
      __syncthreads();
    }
    if (local_expert && even < hidden_size) {
      const float route_weight = load_float(routing_weights[route]);
      const int64_t bias_base = expert * hidden_size;
      even_output +=
          (even_accumulator + load_float(bias[bias_base + even])) * route_weight;
      if (odd < hidden_size) {
        odd_output +=
            (odd_accumulator + load_float(bias[bias_base + odd])) * route_weight;
      }
    }
  }
  if (even < hidden_size) {
    store_float(output, even, even_output);
  }
  if (odd < hidden_size) {
    store_float(output, odd, odd_output);
  }
}

bool valid_geometry(int64_t first, int64_t second, int64_t third) {
  return first > 0 && second > 0 && third > 0 &&
         first <= std::numeric_limits<int32_t>::max() &&
         second <= std::numeric_limits<int32_t>::max() &&
         third <= std::numeric_limits<int32_t>::max();
}

template <typename Element>
void launch_linear_group3(const NmlNvFp4LinearGroup3 *request,
                          cudaStream_t stream, int64_t total_outputs) {
  const uint32_t blocks =
      static_cast<uint32_t>((total_outputs + request->warps_per_block - 1) /
                            request->warps_per_block);
  const LinearGroup3Device<Element> device{
      {request->payloads[0], request->payloads[1], request->payloads[2]},
      {request->block_scales[0], request->block_scales[1],
       request->block_scales[2]},
      {request->global_scales[0], request->global_scales[1],
       request->global_scales[2]},
      {static_cast<const Element *>(request->biases[0]),
       static_cast<const Element *>(request->biases[1]),
       static_cast<const Element *>(request->biases[2])},
      {static_cast<Element *>(request->outputs[0]),
       static_cast<Element *>(request->outputs[1]),
       static_cast<Element *>(request->outputs[2])},
      {request->output_widths[0], request->output_widths[1],
       request->output_widths[2]},
  };
  if (request->warps_per_block == 4) {
    linear_group3_gemv_kernel<Element, 4><<<blocks, 4 * 32, 0, stream>>>(
        static_cast<const Element *>(request->activation), device,
        request->inputs);
  } else {
    linear_group3_gemv_kernel<Element, 8><<<blocks, 8 * 32, 0, stream>>>(
        static_cast<const Element *>(request->activation), device,
        request->inputs);
  }
}

template <typename Element>
void launch_linear_top64(const NmlNvFp4LinearTop64 *request,
                         cudaStream_t stream) {
  linear_top64_candidates_kernel<<<
      static_cast<uint32_t>(request->candidate_groups), 8 * 32, 0, stream>>>(
      static_cast<const Element *>(request->activation), request->payload,
      request->block_scales, request->global_scale,
      static_cast<const Element *>(request->bias), request->candidate_values_a,
      request->candidate_indices_a, request->outputs, request->inputs);

  const float *source_values = request->candidate_values_a;
  const int32_t *source_indices = request->candidate_indices_a;
  int64_t source_groups = request->candidate_groups;
  bool source_is_a = true;
  while (true) {
    const int64_t destination_groups = (source_groups + 7) / 8;
    float *destination_values;
    int32_t *destination_indices;
    if (destination_groups == 1) {
      destination_values = request->top_values;
      destination_indices = request->top_indices;
    } else if (source_is_a) {
      destination_values = request->candidate_values_b;
      destination_indices = request->candidate_indices_b;
    } else {
      destination_values = request->candidate_values_a;
      destination_indices = request->candidate_indices_a;
    }
    merge_top64_kernel<<<static_cast<uint32_t>(destination_groups), 8 * 64, 0,
                         stream>>>(source_values, source_indices,
                                  destination_values, destination_indices,
                                  source_groups);
    if (destination_groups == 1) {
      break;
    }
    source_values = destination_values;
    source_indices = destination_indices;
    source_groups = destination_groups;
    source_is_a = !source_is_a;
  }
}

int32_t launch_result(char *message, size_t capacity) {
  const cudaError_t status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? 0
             : fail(kLaunchFailure, cudaGetErrorString(status), message,
                    capacity);
}

} // namespace

extern "C" int32_t nml_nvfp4_cuda_linear(
    const NmlNvFp4Linear *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr || request->struct_size < sizeof(NmlNvFp4Linear)) {
    return fail(kInvalidArgument, "truncated NVFP4 linear request",
                error_message, error_message_capacity);
  }
  if (request->activation == nullptr || request->payload == nullptr ||
      request->block_scales == nullptr || request->global_scale == nullptr ||
      request->output == nullptr || request->stream == nullptr ||
      !valid_geometry(request->rows, request->outputs, request->inputs) ||
      (request->warps_per_block != 4 && request->warps_per_block != 8)) {
    return fail(kInvalidArgument, "invalid NVFP4 linear request",
                error_message, error_message_capacity);
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->rows == 1) {
    const uint32_t blocks =
        static_cast<uint32_t>((request->outputs + request->warps_per_block - 1) /
                              request->warps_per_block);
#define NML_LAUNCH_LINEAR_GEMV(Element, Warps)                                \
  linear_gemv_kernel<Element, Warps><<<blocks, Warps * 32, 0, stream>>>(      \
      static_cast<const Element *>(request->activation), request->payload,    \
      request->block_scales, request->global_scale,                           \
      static_cast<const Element *>(request->bias),                            \
      static_cast<Element *>(request->output), request->outputs,              \
      request->inputs)
    if (request->dtype == NML_NVFP4_F16) {
      if (request->warps_per_block == 4) {
        NML_LAUNCH_LINEAR_GEMV(__half, 4);
      } else {
        NML_LAUNCH_LINEAR_GEMV(__half, 8);
      }
    } else if (request->dtype == NML_NVFP4_BF16) {
      if (request->warps_per_block == 4) {
        NML_LAUNCH_LINEAR_GEMV(__nv_bfloat16, 4);
      } else {
        NML_LAUNCH_LINEAR_GEMV(__nv_bfloat16, 8);
      }
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

extern "C" int32_t nml_nvfp4_cuda_linear_group3(
    const NmlNvFp4LinearGroup3 *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4LinearGroup3) ||
      request->activation == nullptr || request->stream == nullptr ||
      request->inputs <= 0 ||
      (request->warps_per_block != 4 && request->warps_per_block != 8)) {
    return fail(kInvalidArgument, "invalid NVFP4 linear-group request",
                error_message, error_message_capacity);
  }
  int64_t total_outputs = 0;
  for (int index = 0; index < 3; ++index) {
    if (request->payloads[index] == nullptr ||
        request->block_scales[index] == nullptr ||
        request->global_scales[index] == nullptr ||
        request->biases[index] == nullptr ||
        request->outputs[index] == nullptr ||
        request->output_widths[index] <= 0 ||
        total_outputs > std::numeric_limits<int64_t>::max() -
                            request->output_widths[index]) {
      return fail(kInvalidArgument,
                  "invalid NVFP4 linear-group projection", error_message,
                  error_message_capacity);
    }
    total_outputs += request->output_widths[index];
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->dtype == NML_NVFP4_F16) {
    launch_linear_group3<__half>(request, stream, total_outputs);
  } else if (request->dtype == NML_NVFP4_BF16) {
    launch_linear_group3<__nv_bfloat16>(request, stream, total_outputs);
  } else {
    return fail(kInvalidArgument,
                "NVFP4 linear group supports F16 and BF16 only",
                error_message, error_message_capacity);
  }
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_cuda_route_top4(
    const NmlNvFp4RouteTop4 *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4RouteTop4) ||
      request->hidden == nullptr || request->weight == nullptr ||
      request->bias == nullptr || request->expert_ids == nullptr ||
      request->routing_weights == nullptr || request->stream == nullptr ||
      request->inputs <= 0 || request->experts < 4 ||
      request->experts > 32) {
    return fail(kInvalidArgument, "invalid direct top-four router request",
                error_message, error_message_capacity);
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->dtype == NML_NVFP4_F16) {
    route_top4_kernel<<<1, 8 * 32, 0, stream>>>(
        static_cast<const __half *>(request->hidden),
        static_cast<const __half *>(request->weight),
        static_cast<const __half *>(request->bias), request->expert_ids,
        static_cast<__half *>(request->routing_weights), request->inputs,
        request->experts);
  } else if (request->dtype == NML_NVFP4_BF16) {
    route_top4_kernel<<<1, 8 * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(request->hidden),
        static_cast<const __nv_bfloat16 *>(request->weight),
        static_cast<const __nv_bfloat16 *>(request->bias), request->expert_ids,
        static_cast<__nv_bfloat16 *>(request->routing_weights), request->inputs,
        request->experts);
  } else {
    return fail(kInvalidArgument, "direct router supports F16 and BF16 only",
                error_message, error_message_capacity);
  }
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_cuda_linear_top64(
    const NmlNvFp4LinearTop64 *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4LinearTop64) ||
      request->activation == nullptr || request->payload == nullptr ||
      request->block_scales == nullptr || request->global_scale == nullptr ||
      request->candidate_values_a == nullptr ||
      request->candidate_indices_a == nullptr ||
      request->candidate_values_b == nullptr ||
      request->candidate_indices_b == nullptr ||
      request->top_values == nullptr || request->top_indices == nullptr ||
      request->stream == nullptr || request->outputs < 64 ||
      request->outputs > std::numeric_limits<int32_t>::max() ||
      request->inputs <= 0 || request->candidate_groups <= 0 ||
      request->candidate_groups != (request->outputs + 127) / 128) {
    return fail(kInvalidArgument, "invalid compact linear top-64 request",
                error_message, error_message_capacity);
  }
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
  if (request->dtype == NML_NVFP4_F16) {
    launch_linear_top64<__half>(request, stream);
  } else if (request->dtype == NML_NVFP4_BF16) {
    launch_linear_top64<__nv_bfloat16>(request, stream);
  } else {
    return fail(kInvalidArgument,
                "compact linear top-64 supports F16 and BF16 only",
                error_message, error_message_capacity);
  }
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_cuda_embedding(
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

extern "C" int32_t nml_nvfp4_cuda_expert_gate_up(
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

extern "C" int32_t nml_nvfp4_cuda_expert_down(
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

extern "C" int32_t nml_nvfp4_cuda_direct_expert_gate_up(
    const NmlNvFp4DirectExpertGateUp *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4DirectExpertGateUp) ||
      request->hidden == nullptr || request->expert_ids == nullptr ||
      request->payload == nullptr || request->block_scales == nullptr ||
      request->global_scale == nullptr || request->bias == nullptr ||
      request->activated == nullptr || request->stream == nullptr ||
      !valid_geometry(request->routes, request->hidden_size,
                      request->intermediate_size) ||
      request->local_experts <= 0 || request->expert_offset == nullptr) {
    return fail(kInvalidArgument,
                "invalid direct NVFP4 expert gate/up request", error_message,
                error_message_capacity);
  }
  constexpr uint32_t kThreads = 128;
  const dim3 grid(
      static_cast<uint32_t>((request->intermediate_size + kThreads - 1) /
                            kThreads),
      static_cast<uint32_t>(request->routes));
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
#define NML_LAUNCH_DIRECT_GATE_UP(Element)                                    \
  direct_expert_gate_up_kernel<<<grid, kThreads, 0, stream>>>(                \
      static_cast<const Element *>(request->hidden), request->expert_ids,     \
      request->payload, request->block_scales, request->global_scale,         \
      static_cast<const Element *>(request->bias),                            \
      static_cast<Element *>(request->activated), request->routes,            \
      request->hidden_size, request->intermediate_size,                       \
      request->local_experts, request->expert_offset)
  if (request->dtype == NML_NVFP4_F16) {
    NML_LAUNCH_DIRECT_GATE_UP(__half);
  } else if (request->dtype == NML_NVFP4_BF16) {
    NML_LAUNCH_DIRECT_GATE_UP(__nv_bfloat16);
  } else {
    return fail(kInvalidArgument, "NVFP4 experts support F16 and BF16 only",
                error_message, error_message_capacity);
  }
#undef NML_LAUNCH_DIRECT_GATE_UP
  return launch_result(error_message, error_message_capacity);
}

extern "C" int32_t nml_nvfp4_cuda_direct_expert_down(
    const NmlNvFp4DirectExpertDown *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlNvFp4DirectExpertDown) ||
      request->activated == nullptr || request->expert_ids == nullptr ||
      request->payload == nullptr || request->block_scales == nullptr ||
      request->global_scale == nullptr || request->bias == nullptr ||
      request->routing_weights == nullptr || request->output == nullptr ||
      request->stream == nullptr ||
      !valid_geometry(request->routes, request->intermediate_size,
                      request->hidden_size) ||
      request->local_experts <= 0 || request->expert_offset == nullptr) {
    return fail(kInvalidArgument, "invalid direct NVFP4 expert down request",
                error_message, error_message_capacity);
  }
  constexpr uint32_t kThreads = 128;
  const int64_t output_pairs = (request->hidden_size + 1) / 2;
  const uint32_t blocks =
      static_cast<uint32_t>((output_pairs + kThreads - 1) / kThreads);
  cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
#define NML_LAUNCH_DIRECT_DOWN(Element)                                       \
  direct_expert_down_kernel<<<blocks, kThreads, 0, stream>>>(                 \
      static_cast<const Element *>(request->activated), request->expert_ids, \
      request->payload, request->block_scales, request->global_scale,         \
      static_cast<const Element *>(request->bias),                            \
      static_cast<const Element *>(request->routing_weights),                 \
      static_cast<Element *>(request->output), request->routes,               \
      request->intermediate_size, request->hidden_size,                       \
      request->local_experts, request->expert_offset)
  if (request->dtype == NML_NVFP4_F16) {
    NML_LAUNCH_DIRECT_DOWN(__half);
  } else if (request->dtype == NML_NVFP4_BF16) {
    NML_LAUNCH_DIRECT_DOWN(__nv_bfloat16);
  } else {
    return fail(kInvalidArgument, "NVFP4 experts support F16 and BF16 only",
                error_message, error_message_capacity);
  }
#undef NML_LAUNCH_DIRECT_DOWN
  return launch_result(error_message, error_message_capacity);
}
