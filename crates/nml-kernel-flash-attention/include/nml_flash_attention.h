#ifndef NML_FLASH_ATTENTION_H_
#define NML_FLASH_ATTENTION_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// This ABI describes borrowed device buffers only. Tensor ownership, backend
// selection, and workspace allocation stay with the XLA custom-call handler.
typedef enum NmlFlashAttentionDType {
  NML_FLASH_ATTENTION_F16 = 1,
  NML_FLASH_ATTENTION_BF16 = 2,
} NmlFlashAttentionDType;

typedef struct NmlFlashAttentionForward {
  size_t struct_size;
  void *query;
  void *key;
  void *value;
  void *output;
  void *softmax_lse;
  // FA3 uses one caller-owned I32 scheduler semaphore. FA2 ignores this field.
  void *workspace;
  void *stream;

  int64_t query_batch_stride;
  int64_t query_row_stride;
  int64_t query_head_stride;
  int64_t key_batch_stride;
  int64_t key_row_stride;
  int64_t key_head_stride;
  int64_t value_batch_stride;
  int64_t value_row_stride;
  int64_t value_head_stride;
  int64_t output_batch_stride;
  int64_t output_row_stride;
  int64_t output_head_stride;

  int32_t batch_size;
  int32_t query_length;
  int32_t key_length;
  int32_t query_heads;
  int32_t key_value_heads;
  int32_t head_dimension;
  int32_t sliding_window;
  float scale;
  NmlFlashAttentionDType dtype;
  uint8_t causal;
} NmlFlashAttentionForward;

typedef struct NmlFlashAttentionPagedForward {
  size_t struct_size;
  void *query;
  void *key_cache;
  void *value_cache;
  void *page_table;
  void *sequence_lengths;
  void *output;
  void *softmax_lse;
  // FA3 uses one caller-owned I32 scheduler semaphore. FA2 ignores this field.
  void *workspace;
  void *stream;

  int64_t query_batch_stride;
  int64_t query_row_stride;
  int64_t query_head_stride;
  int64_t cache_page_stride;
  int64_t cache_row_stride;
  int64_t cache_head_stride;
  int64_t output_batch_stride;
  int64_t output_row_stride;
  int64_t output_head_stride;
  int64_t page_table_batch_stride;

  int32_t num_pages;
  int32_t page_size;
  int32_t max_pages_per_sequence;
  int32_t batch_size;
  int32_t query_length;
  int32_t query_heads;
  int32_t key_value_heads;
  int32_t head_dimension;
  int32_t sliding_window;
  float scale;
  NmlFlashAttentionDType dtype;
  uint8_t causal;
} NmlFlashAttentionPagedForward;

// Returns zero after enqueueing the complete forward launch. On failure it
// writes a NUL-terminated diagnostic into caller-owned storage and returns a
// stable nonzero category; no exception crosses the C boundary.
int32_t nml_flash_attention_2_forward(const NmlFlashAttentionForward *request,
                                      char *error_message,
                                      size_t error_message_capacity);

int32_t nml_flash_attention_3_forward(const NmlFlashAttentionForward *request,
                                      char *error_message,
                                      size_t error_message_capacity);

int32_t nml_flash_attention_2_paged_forward(
    const NmlFlashAttentionPagedForward *request, char *error_message,
    size_t error_message_capacity);

int32_t nml_flash_attention_3_paged_forward(
    const NmlFlashAttentionPagedForward *request, char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NML_FLASH_ATTENTION_H_
