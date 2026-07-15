#include "nml_flash_attention.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace {

bool contains(const char *message, const char *expected) {
  return std::strstr(message, expected) != nullptr;
}

using Forward = int32_t (*)(const NmlFlashAttentionForward *, char *, size_t);
using PagedForward = int32_t (*)(const NmlFlashAttentionPagedForward *, char *,
                                 size_t);

template <typename Request, typename Function>
bool rejects(Function function, const Request &request, const char *expected) {
  char message[256] = {};
  return function(&request, message, sizeof(message)) == 1 &&
         contains(message, expected);
}

bool rejects_invalid_requests(Forward forward) {
  char message[256] = {};
  if (forward(nullptr, message, sizeof(message)) != 1 ||
      !contains(message, "truncated FlashAttention request")) {
    return false;
  }

  NmlFlashAttentionForward request{};
  request.struct_size = sizeof(request);
  message[0] = '\0';
  if (forward(&request, message, sizeof(message)) != 1 ||
      !contains(message, "requires non-null")) {
    return false;
  }

  // Non-null sentinel addresses are never dereferenced: invalid geometry must
  // be rejected before CUDA discovery or an upstream kernel launch.
  request.query = reinterpret_cast<void *>(1);
  request.key = reinterpret_cast<void *>(1);
  request.value = reinterpret_cast<void *>(1);
  request.output = reinterpret_cast<void *>(1);
  request.softmax_lse = reinterpret_cast<void *>(1);
  request.workspace = reinterpret_cast<void *>(1);
  request.stream = reinterpret_cast<void *>(1);
  request.batch_size = 1;
  request.query_length = 1;
  request.key_length = 1;
  request.query_heads = 1;
  request.key_value_heads = 1;
  request.head_dimension = 7;
  request.sliding_window = -1;
  request.scale = 1.0f;
  request.dtype = NML_FLASH_ATTENTION_F16;
  message[0] = '\0';
  if (forward(&request, message, sizeof(message)) != 1 ||
      !contains(message, "unsupported FlashAttention geometry")) {
    return false;
  }

  request.head_dimension = 64;
  request.query_batch_stride = 64;
  request.query_row_stride = 64;
  request.query_head_stride = 64;
  request.key_batch_stride = 64;
  request.key_row_stride = 64;
  request.key_head_stride = 64;
  request.value_batch_stride = 64;
  request.value_row_stride = 64;
  request.value_head_stride = 64;
  request.output_batch_stride = 64;
  request.output_row_stride = 64;
  request.output_head_stride = 64;

  request.sliding_window = 0;
  if (!rejects(forward, request, "window must be -1 or positive")) {
    return false;
  }
  request.sliding_window = -1;
  request.scale = 0.0f;
  if (!rejects(forward, request, "scale must be finite and positive")) {
    return false;
  }
  request.scale = 1.0f;
  request.query_batch_stride = 0;
  if (!rejects(forward, request, "strides must be positive")) {
    return false;
  }
  request.query_batch_stride = 64;
  request.dtype = static_cast<NmlFlashAttentionDType>(0);
  if (!rejects(forward, request, "supports FP16 and BF16 only")) {
    return false;
  }
  request.dtype = NML_FLASH_ATTENTION_F16;
  request.query_length = std::numeric_limits<int32_t>::max();
  if (!rejects(forward, request, "sequence length rounding exceeds I32")) {
    return false;
  }

  return true;
}

bool rejects_invalid_paged_requests(PagedForward forward) {
  char message[256] = {};
  if (forward(nullptr, message, sizeof(message)) != 1 ||
      !contains(message, "truncated paged FlashAttention request")) {
    return false;
  }

  NmlFlashAttentionPagedForward request{};
  request.struct_size = sizeof(request);
  message[0] = '\0';
  if (forward(&request, message, sizeof(message)) != 1 ||
      !contains(message, "requires non-null")) {
    return false;
  }

  // Invalid geometry is a durable host-side ABI contract. Sentinel addresses
  // prove both adapters reject it before device discovery or dereferencing.
  request.query = reinterpret_cast<void *>(1);
  request.key_cache = reinterpret_cast<void *>(1);
  request.value_cache = reinterpret_cast<void *>(1);
  request.page_table = reinterpret_cast<void *>(1);
  request.sequence_lengths = reinterpret_cast<void *>(1);
  request.output = reinterpret_cast<void *>(1);
  request.softmax_lse = reinterpret_cast<void *>(1);
  request.workspace = reinterpret_cast<void *>(1);
  request.stream = reinterpret_cast<void *>(1);
  request.num_pages = 1;
  request.page_size = 1;
  request.max_pages_per_sequence = 1;
  request.batch_size = 1;
  request.query_length = 1;
  request.query_heads = 1;
  request.key_value_heads = 1;
  request.head_dimension = 7;
  request.sliding_window = -1;
  request.scale = 1.0f;
  request.dtype = NML_FLASH_ATTENTION_F16;
  message[0] = '\0';
  if (forward(&request, message, sizeof(message)) != 1 ||
      !contains(message, "unsupported paged FlashAttention")) {
    return false;
  }

  request.page_size = 256;
  request.head_dimension = 64;
  request.query_batch_stride = 64;
  request.query_row_stride = 64;
  request.query_head_stride = 64;
  request.cache_page_stride = 256 * 64;
  request.cache_row_stride = 64;
  request.cache_head_stride = 64;
  request.output_batch_stride = 64;
  request.output_row_stride = 64;
  request.output_head_stride = 64;
  request.page_table_batch_stride = 1;

  request.sliding_window = 0;
  if (!rejects(forward, request,
               "invalid paged FlashAttention scale or sliding window")) {
    return false;
  }
  request.sliding_window = -1;
  request.scale = 0.0f;
  if (!rejects(forward, request,
               "invalid paged FlashAttention scale or sliding window")) {
    return false;
  }
  request.scale = 1.0f;
  request.cache_page_stride = 0;
  if (!rejects(forward, request, "strides must be positive")) {
    return false;
  }
  request.cache_page_stride = 256 * 64;
  request.dtype = static_cast<NmlFlashAttentionDType>(0);
  if (!rejects(forward, request, "supports FP16 and BF16 only")) {
    return false;
  }
  request.dtype = NML_FLASH_ATTENTION_F16;
  request.query_length = std::numeric_limits<int32_t>::max();
  if (!rejects(forward, request, "sequence length rounding exceeds I32")) {
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!rejects_invalid_requests(nml_flash_attention_2_forward)) {
    return 1;
  }
  if (!rejects_invalid_requests(nml_flash_attention_3_forward)) {
    return 2;
  }
  if (!rejects_invalid_paged_requests(nml_flash_attention_2_paged_forward)) {
    return 3;
  }
  if (!rejects_invalid_paged_requests(nml_flash_attention_3_paged_forward)) {
    return 4;
  }
  return 0;
}
