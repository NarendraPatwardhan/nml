#include "nml_flash_attention.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>

#include <cutlass/numeric_types.h>

#include "adapter_common.h"
#include "flash.h"

namespace {

constexpr int32_t kInvalidArgument = 1;
constexpr int32_t kLaunchFailure = 2;

int32_t fail(int32_t code, const char *message, char *output, size_t capacity) {
  if (output != nullptr && capacity != 0) {
    std::snprintf(output, capacity, "%s", message);
  }
  return code;
}

bool valid_stride(int64_t stride) { return stride > 0; }

template <typename Element, int HeadDimension>
void launch(flash::Flash_fwd_params &params, cudaStream_t stream) {
  if (params.is_causal) {
    flash::run_mha_fwd_<Element, HeadDimension, true>(params, stream);
  } else {
    flash::run_mha_fwd_<Element, HeadDimension, false>(params, stream);
  }
}

template <typename Element>
void dispatch_head_dimension(flash::Flash_fwd_params &params,
                             cudaStream_t stream) {
  if (params.d <= 32) {
    launch<Element, 32>(params, stream);
  } else if (params.d <= 64) {
    launch<Element, 64>(params, stream);
  } else if (params.d <= 96) {
    launch<Element, 96>(params, stream);
  } else if (params.d <= 128) {
    launch<Element, 128>(params, stream);
  } else if (params.d <= 192) {
    launch<Element, 192>(params, stream);
  } else {
    launch<Element, 256>(params, stream);
  }
}

template <typename Element, int HeadDimension>
void launch_paged(flash::Flash_fwd_params &params, cudaStream_t stream) {
  if (params.is_causal) {
    flash::run_mha_fwd_splitkv_dispatch<Element, HeadDimension, true>(params,
                                                                      stream);
  } else {
    flash::run_mha_fwd_splitkv_dispatch<Element, HeadDimension, false>(params,
                                                                       stream);
  }
}

template <typename Element>
void dispatch_paged_head_dimension(flash::Flash_fwd_params &params,
                                   cudaStream_t stream) {
  if (params.d <= 32) {
    launch_paged<Element, 32>(params, stream);
  } else if (params.d <= 64) {
    launch_paged<Element, 64>(params, stream);
  } else if (params.d <= 96) {
    launch_paged<Element, 96>(params, stream);
  } else if (params.d <= 128) {
    launch_paged<Element, 128>(params, stream);
  } else if (params.d <= 192) {
    launch_paged<Element, 192>(params, stream);
  } else {
    launch_paged<Element, 256>(params, stream);
  }
}

} // namespace

extern "C" int32_t
nml_flash_attention_2_forward(const NmlFlashAttentionForward *request,
                              char *error_message,
                              size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlFlashAttentionForward)) {
    return fail(kInvalidArgument, "truncated FlashAttention request",
                error_message, error_message_capacity);
  }
  if (request->query == nullptr || request->key == nullptr ||
      request->value == nullptr || request->output == nullptr ||
      request->softmax_lse == nullptr || request->stream == nullptr) {
    return fail(kInvalidArgument,
                "FlashAttention requires non-null buffers and CUDA stream",
                error_message, error_message_capacity);
  }
  if (request->batch_size <= 0 || request->query_length <= 0 ||
      request->key_length <= 0 || request->query_heads <= 0 ||
      request->key_value_heads <= 0 || request->head_dimension <= 0 ||
      request->head_dimension > 256 || request->head_dimension % 8 != 0 ||
      request->query_heads % request->key_value_heads != 0) {
    return fail(kInvalidArgument, "unsupported FlashAttention geometry",
                error_message, error_message_capacity);
  }
  if (request->sliding_window == 0 || request->sliding_window < -1) {
    return fail(kInvalidArgument,
                "FlashAttention window must be -1 or positive", error_message,
                error_message_capacity);
  }
  if (!std::isfinite(request->scale) || request->scale <= 0.0f) {
    return fail(kInvalidArgument,
                "FlashAttention scale must be finite and positive",
                error_message, error_message_capacity);
  }
  const int64_t strides[] = {
      request->query_batch_stride, request->query_row_stride,
      request->query_head_stride,  request->key_batch_stride,
      request->key_row_stride,     request->key_head_stride,
      request->value_batch_stride, request->value_row_stride,
      request->value_head_stride,  request->output_batch_stride,
      request->output_row_stride,  request->output_head_stride,
  };
  for (int64_t stride : strides) {
    if (!valid_stride(stride)) {
      return fail(kInvalidArgument, "FlashAttention strides must be positive",
                  error_message, error_message_capacity);
    }
  }
  if (request->dtype != NML_FLASH_ATTENTION_F16 &&
      request->dtype != NML_FLASH_ATTENTION_BF16) {
    return fail(kInvalidArgument, "FlashAttention supports FP16 and BF16 only",
                error_message, error_message_capacity);
  }
  if (request->batch_size >
      std::numeric_limits<int32_t>::max() / request->query_length) {
    return fail(kInvalidArgument, "FlashAttention token count exceeds I32",
                error_message, error_message_capacity);
  }
  int32_t rounded_query_length = 0;
  int32_t rounded_key_length = 0;
  if (!nml::flash_attention::internal::round_sequence_length(
          request->query_length, &rounded_query_length) ||
      !nml::flash_attention::internal::round_sequence_length(
          request->key_length, &rounded_key_length)) {
    return fail(kInvalidArgument,
                "FlashAttention sequence length rounding exceeds I32",
                error_message, error_message_capacity);
  }

  flash::Flash_fwd_params params{};
  params.q_ptr = request->query;
  params.k_ptr = request->key;
  params.v_ptr = request->value;
  params.o_ptr = request->output;
  params.softmax_lse_ptr = request->softmax_lse;
  params.q_batch_stride = request->query_batch_stride;
  params.q_row_stride = request->query_row_stride;
  params.q_head_stride = request->query_head_stride;
  params.k_batch_stride = request->key_batch_stride;
  params.k_row_stride = request->key_row_stride;
  params.k_head_stride = request->key_head_stride;
  params.v_batch_stride = request->value_batch_stride;
  params.v_row_stride = request->value_row_stride;
  params.v_head_stride = request->value_head_stride;
  params.o_batch_stride = request->output_batch_stride;
  params.o_row_stride = request->output_row_stride;
  params.o_head_stride = request->output_head_stride;
  params.b = request->batch_size;
  params.seqlen_q = request->query_length;
  params.seqlen_k = request->key_length;
  params.h = request->query_heads;
  params.h_k = request->key_value_heads;
  params.h_h_k_ratio = request->query_heads / request->key_value_heads;
  params.d = request->head_dimension;
  params.total_q = request->batch_size * request->query_length;
  params.seqlen_q_rounded = rounded_query_length;
  params.seqlen_k_rounded = rounded_key_length;
  params.d_rounded = request->head_dimension <= 128
                         ? (request->head_dimension + 31) / 32 * 32
                         : (request->head_dimension + 63) / 64 * 64;
  params.scale_softmax = request->scale;
  params.scale_softmax_log2 = request->scale * 1.4426950408889634f;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = std::numeric_limits<uint8_t>::max();
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = request->scale;
  params.is_bf16 = request->dtype == NML_FLASH_ATTENTION_BF16;
  params.is_seqlens_k_cumulative = true;
  params.philox_args = {0, 0};

  if (request->causal != 0) {
    params.window_size_right = 0;
    params.window_size_left =
        request->sliding_window < 0 ? -1 : request->sliding_window - 1;
  } else if (request->sliding_window < 0) {
    params.window_size_left = -1;
    params.window_size_right = -1;
  } else {
    params.window_size_left = request->sliding_window - 1;
    params.window_size_right = request->sliding_window - 1;
  }
  params.is_causal =
      params.window_size_left < 0 && params.window_size_right == 0;

  try {
    int device = 0;
    int capability_major = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return fail(kLaunchFailure, cudaGetErrorString(status), error_message,
                  error_message_capacity);
    }
    status = cudaDeviceGetAttribute(&capability_major,
                                    cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) {
      return fail(kLaunchFailure, cudaGetErrorString(status), error_message,
                  error_message_capacity);
    }
    if (capability_major != 8) {
      return fail(kInvalidArgument,
                  "FlashAttention 2 requires CUDA compute capability SM80-SM89",
                  error_message, error_message_capacity);
    }
    auto stream = static_cast<cudaStream_t>(request->stream);
    if (params.is_bf16) {
      dispatch_head_dimension<cutlass::bfloat16_t>(params, stream);
    } else {
      dispatch_head_dimension<cutlass::half_t>(params, stream);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(kLaunchFailure, error.what(), error_message,
                error_message_capacity);
  } catch (...) {
    return fail(kLaunchFailure, "unknown FlashAttention launch failure",
                error_message, error_message_capacity);
  }
}

extern "C" int32_t nml_flash_attention_2_paged_forward(
    const NmlFlashAttentionPagedForward *request, char *error_message,
    size_t error_message_capacity) {
  if (request == nullptr ||
      request->struct_size < sizeof(NmlFlashAttentionPagedForward)) {
    return fail(kInvalidArgument, "truncated paged FlashAttention request",
                error_message, error_message_capacity);
  }
  if (request->query == nullptr || request->key_cache == nullptr ||
      request->value_cache == nullptr || request->page_table == nullptr ||
      request->sequence_lengths == nullptr || request->output == nullptr ||
      request->softmax_lse == nullptr || request->stream == nullptr) {
    return fail(
        kInvalidArgument,
        "paged FlashAttention requires non-null buffers and CUDA stream",
        error_message, error_message_capacity);
  }
  if (request->num_pages <= 0 || request->page_size <= 0 ||
      request->page_size % 256 != 0 || request->max_pages_per_sequence <= 0 ||
      request->batch_size <= 0 || request->query_length <= 0 ||
      request->query_heads <= 0 || request->key_value_heads <= 0 ||
      request->head_dimension <= 0 || request->head_dimension > 256 ||
      request->head_dimension % 8 != 0 ||
      request->query_heads % request->key_value_heads != 0) {
    return fail(kInvalidArgument, "unsupported paged FlashAttention 2 geometry",
                error_message, error_message_capacity);
  }
  if (request->sliding_window == 0 || request->sliding_window < -1 ||
      !std::isfinite(request->scale) || request->scale <= 0.0f) {
    return fail(kInvalidArgument,
                "invalid paged FlashAttention scale or sliding window",
                error_message, error_message_capacity);
  }
  const int64_t strides[] = {
      request->query_batch_stride,  request->query_row_stride,
      request->query_head_stride,   request->cache_page_stride,
      request->cache_row_stride,    request->cache_head_stride,
      request->output_batch_stride, request->output_row_stride,
      request->output_head_stride,  request->page_table_batch_stride,
  };
  for (int64_t stride : strides) {
    if (!valid_stride(stride)) {
      return fail(kInvalidArgument,
                  "paged FlashAttention strides must be positive",
                  error_message, error_message_capacity);
    }
  }
  if (request->dtype != NML_FLASH_ATTENTION_F16 &&
      request->dtype != NML_FLASH_ATTENTION_BF16) {
    return fail(kInvalidArgument,
                "paged FlashAttention supports FP16 and BF16 only",
                error_message, error_message_capacity);
  }
  if (request->max_pages_per_sequence >
          std::numeric_limits<int32_t>::max() / request->page_size ||
      request->batch_size >
          std::numeric_limits<int32_t>::max() / request->query_length) {
    return fail(kInvalidArgument, "paged FlashAttention dimensions exceed I32",
                error_message, error_message_capacity);
  }
  const int32_t key_length =
      request->max_pages_per_sequence * request->page_size;
  int32_t rounded_query_length = 0;
  int32_t rounded_key_length = 0;
  if (!nml::flash_attention::internal::round_sequence_length(
          request->query_length, &rounded_query_length) ||
      !nml::flash_attention::internal::round_sequence_length(
          key_length, &rounded_key_length)) {
    return fail(kInvalidArgument,
                "paged FlashAttention sequence length rounding exceeds I32",
                error_message, error_message_capacity);
  }

  flash::Flash_fwd_params params{};
  params.q_ptr = request->query;
  params.k_ptr = request->key_cache;
  params.v_ptr = request->value_cache;
  params.o_ptr = request->output;
  params.softmax_lse_ptr = request->softmax_lse;
  params.q_batch_stride = request->query_batch_stride;
  params.q_row_stride = request->query_row_stride;
  params.q_head_stride = request->query_head_stride;
  params.k_batch_stride = request->cache_page_stride;
  params.k_row_stride = request->cache_row_stride;
  params.k_head_stride = request->cache_head_stride;
  params.v_batch_stride = request->cache_page_stride;
  params.v_row_stride = request->cache_row_stride;
  params.v_head_stride = request->cache_head_stride;
  params.o_batch_stride = request->output_batch_stride;
  params.o_row_stride = request->output_row_stride;
  params.o_head_stride = request->output_head_stride;
  params.block_table = static_cast<int *>(request->page_table);
  params.block_table_batch_stride = request->page_table_batch_stride;
  params.seqused_k = static_cast<int *>(request->sequence_lengths);
  params.page_block_size = request->page_size;
  params.b = request->batch_size;
  params.seqlen_q = request->query_length;
  params.seqlen_k = key_length;
  params.h = request->query_heads;
  params.h_k = request->key_value_heads;
  params.h_h_k_ratio = request->query_heads / request->key_value_heads;
  params.d = request->head_dimension;
  params.total_q = request->batch_size * request->query_length;
  params.seqlen_q_rounded = rounded_query_length;
  params.seqlen_k_rounded = rounded_key_length;
  params.d_rounded = request->head_dimension <= 128
                         ? (request->head_dimension + 31) / 32 * 32
                         : (request->head_dimension + 63) / 64 * 64;
  params.scale_softmax = request->scale;
  params.scale_softmax_log2 = request->scale * 1.4426950408889634f;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = std::numeric_limits<uint8_t>::max();
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = request->scale;
  params.is_bf16 = request->dtype == NML_FLASH_ATTENTION_BF16;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = 1;

  int window_left =
      request->sliding_window < 0 ? -1 : request->sliding_window - 1;
  int window_right =
      request->causal != 0
          ? 0
          : (request->sliding_window < 0 ? -1 : request->sliding_window - 1);
  if (window_left >= params.seqlen_k) {
    window_left = -1;
  }
  if (window_right >= params.seqlen_k) {
    window_right = -1;
  }
  params.is_causal = window_left < 0 && window_right == 0;
  if (window_left < 0 && window_right >= 0) {
    window_left = params.seqlen_k;
  }
  if (window_left >= 0 && window_right < 0) {
    window_right = params.seqlen_k;
  }
  params.window_size_left = window_left;
  params.window_size_right = window_right;

  try {
    int device = 0;
    int capability_major = 0;
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) {
      status = cudaDeviceGetAttribute(
          &capability_major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (status != cudaSuccess) {
      return fail(kLaunchFailure, cudaGetErrorString(status), error_message,
                  error_message_capacity);
    }
    if (capability_major != 8) {
      return fail(kInvalidArgument,
                  "paged FlashAttention 2 requires CUDA compute capability "
                  "SM80-SM89",
                  error_message, error_message_capacity);
    }
    auto stream = static_cast<cudaStream_t>(request->stream);
    if (params.is_bf16) {
      dispatch_paged_head_dimension<cutlass::bfloat16_t>(params, stream);
    } else {
      dispatch_paged_head_dimension<cutlass::half_t>(params, stream);
    }
    return 0;
  } catch (const std::exception &error) {
    return fail(kLaunchFailure, error.what(), error_message,
                error_message_capacity);
  } catch (...) {
    return fail(kLaunchFailure, "unknown paged FlashAttention launch failure",
                error_message, error_message_capacity);
  }
}
