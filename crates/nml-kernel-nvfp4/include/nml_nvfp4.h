#ifndef NML_NVFP4_H_
#define NML_NVFP4_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NmlNvFp4DType {
  NML_NVFP4_F16 = 1,
  NML_NVFP4_BF16 = 2,
} NmlNvFp4DType;

// All pointers are borrowed device addresses owned by XLA. The adapter only
// enqueues work on `stream`; it never allocates or retains representation data.
typedef struct NmlNvFp4Linear {
  size_t struct_size;
  const void *activation;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  const void *bias;
  void *output;
  void *stream;
  int64_t rows;
  int64_t outputs;
  int64_t inputs;
  NmlNvFp4DType dtype;
} NmlNvFp4Linear;

typedef struct NmlNvFp4Embedding {
  size_t struct_size;
  const void *indices;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  void *output;
  void *stream;
  int64_t rows;
  int64_t vocabulary;
  int64_t width;
  NmlNvFp4DType dtype;
  uint8_t indices_are_i64;
} NmlNvFp4Embedding;

typedef struct NmlNvFp4ExpertGateUp {
  size_t struct_size;
  const void *hidden;
  const int32_t *sorted_assignments;
  const int32_t *block_experts;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  const void *bias;
  void *activated;
  void *stream;
  int64_t tokens;
  int64_t assignments;
  int64_t schedule_positions;
  int64_t schedule_blocks;
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t experts;
  int64_t experts_per_token;
  int32_t block_size;
  NmlNvFp4DType dtype;
} NmlNvFp4ExpertGateUp;

typedef struct NmlNvFp4ExpertDown {
  size_t struct_size;
  const void *activated;
  const int32_t *sorted_assignments;
  const int32_t *block_experts;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  const void *bias;
  const void *routing_weights;
  void *weighted_output;
  void *stream;
  int64_t assignments;
  int64_t schedule_positions;
  int64_t schedule_blocks;
  int64_t intermediate_size;
  int64_t hidden_size;
  int64_t experts;
  int64_t experts_per_token;
  int32_t block_size;
  NmlNvFp4DType dtype;
} NmlNvFp4ExpertDown;

int32_t nml_nvfp4_turing_linear(const NmlNvFp4Linear *request,
                                char *error_message,
                                size_t error_message_capacity);

int32_t nml_nvfp4_turing_embedding(const NmlNvFp4Embedding *request,
                                   char *error_message,
                                   size_t error_message_capacity);

int32_t nml_nvfp4_turing_expert_gate_up(
    const NmlNvFp4ExpertGateUp *request, char *error_message,
    size_t error_message_capacity);

int32_t nml_nvfp4_turing_expert_down(const NmlNvFp4ExpertDown *request,
                                     char *error_message,
                                     size_t error_message_capacity);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NML_NVFP4_H_
