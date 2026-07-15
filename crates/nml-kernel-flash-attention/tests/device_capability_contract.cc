#include "nml_flash_attention.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr int32_t kInvalidArgument = 1;

bool contains(const char *message, const char *expected) {
  return std::strstr(message, expected) != nullptr;
}

NmlFlashAttentionForward dense_request() {
  NmlFlashAttentionForward request{};
  request.struct_size = sizeof(request);
  request.query = reinterpret_cast<void *>(1);
  request.key = reinterpret_cast<void *>(1);
  request.value = reinterpret_cast<void *>(1);
  request.output = reinterpret_cast<void *>(1);
  request.softmax_lse = reinterpret_cast<void *>(1);
  request.workspace = reinterpret_cast<void *>(1);
  request.stream = reinterpret_cast<void *>(1);
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
  request.batch_size = 1;
  request.query_length = 1;
  request.key_length = 1;
  request.query_heads = 1;
  request.key_value_heads = 1;
  request.head_dimension = 64;
  request.sliding_window = -1;
  request.scale = 0.125f;
  request.dtype = NML_FLASH_ATTENTION_F16;
  return request;
}

NmlFlashAttentionPagedForward paged_request() {
  NmlFlashAttentionPagedForward request{};
  request.struct_size = sizeof(request);
  request.query = reinterpret_cast<void *>(1);
  request.key_cache = reinterpret_cast<void *>(1);
  request.value_cache = reinterpret_cast<void *>(1);
  request.page_table = reinterpret_cast<void *>(1);
  request.sequence_lengths = reinterpret_cast<void *>(1);
  request.output = reinterpret_cast<void *>(1);
  request.softmax_lse = reinterpret_cast<void *>(1);
  request.workspace = reinterpret_cast<void *>(1);
  request.stream = reinterpret_cast<void *>(1);
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
  request.num_pages = 1;
  request.page_size = 256;
  request.max_pages_per_sequence = 1;
  request.batch_size = 1;
  request.query_length = 1;
  request.query_heads = 1;
  request.key_value_heads = 1;
  request.head_dimension = 64;
  request.sliding_window = -1;
  request.scale = 0.125f;
  request.dtype = NML_FLASH_ATTENTION_F16;
  return request;
}

template <typename Request, typename Forward>
bool rejects(Forward forward, const Request &request, const char *expected) {
  char message[256] = {};
  const int32_t status = forward(&request, message, sizeof(message));
  if (status == kInvalidArgument && contains(message, expected)) {
    return true;
  }
  std::fprintf(stderr,
               "expected capability rejection containing '%s'; "
               "received status %d and '%s'\n",
               expected, status, message);
  return false;
}

} // namespace

int main() {
  int device = 0;
  int major = 0;
  int minor = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                    device);
  }
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                    device);
  }
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cannot inspect active CUDA device: %s\n",
                 cudaGetErrorString(status));
    return 1;
  }

  const auto dense = dense_request();
  const auto paged = paged_request();
  if (major != 8 &&
      !rejects(nml_flash_attention_2_forward, dense,
               "FlashAttention 2 requires CUDA compute capability SM80-SM89")) {
    return 2;
  }
  if (major != 8 &&
      !rejects(nml_flash_attention_2_paged_forward, paged,
               "paged FlashAttention 2 requires CUDA compute capability "
               "SM80-SM89")) {
    return 3;
  }
  if ((major != 9 || minor != 0) &&
      !rejects(nml_flash_attention_3_forward, dense,
               "FlashAttention 3 requires CUDA compute capability SM90")) {
    return 4;
  }
  if ((major != 9 || minor != 0) &&
      !rejects(nml_flash_attention_3_paged_forward, paged,
               "paged FlashAttention 3 requires CUDA compute capability "
               "SM90")) {
    return 5;
  }
  return 0;
}
