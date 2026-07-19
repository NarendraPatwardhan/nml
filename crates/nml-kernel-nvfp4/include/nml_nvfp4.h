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
  uint32_t warps_per_block;
  NmlNvFp4DType dtype;
} NmlNvFp4Linear;

typedef struct NmlNvFp4LinearGroup3 {
  size_t struct_size;
  const void *activation;
  const uint8_t *payloads[3];
  const uint8_t *block_scales[3];
  const float *global_scales[3];
  const void *biases[3];
  void *outputs[3];
  void *stream;
  int64_t output_widths[3];
  int64_t inputs;
  uint32_t warps_per_block;
  NmlNvFp4DType dtype;
} NmlNvFp4LinearGroup3;

// Direct single-token MoE routing. The CUDA boundary preserves the semantic
// activation-dtype rounding of logits and probabilities before deterministic
// top-four selection and renormalization.
typedef struct NmlNvFp4RouteTop4 {
  size_t struct_size;
  const void *hidden;
  const void *weight;
  const void *bias;
  int32_t *expert_ids;
  void *routing_weights;
  void *stream;
  int64_t inputs;
  int64_t experts;
  NmlNvFp4DType dtype;
} NmlNvFp4RouteTop4;

// Exact streaming top-64 for one compact projection. Candidate workspaces are
// XLA-owned results of the enclosing custom call; the adapter never allocates
// or retains device memory.
typedef struct NmlNvFp4LinearTop64 {
  size_t struct_size;
  const void *activation;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  const void *bias;
  float *candidate_values_a;
  int32_t *candidate_indices_a;
  float *candidate_values_b;
  int32_t *candidate_indices_b;
  float *top_values;
  int32_t *top_indices;
  void *stream;
  int64_t outputs;
  int64_t inputs;
  int64_t candidate_groups;
  NmlNvFp4DType dtype;
} NmlNvFp4LinearTop64;

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

// Single-row expert decode bypasses the padded grouped schedule.  Top-k has
// already produced the route IDs and normalized weights, so this boundary
// carries exactly that semantic data and no matrix-oriented scheduling state.
typedef struct NmlNvFp4DirectExpertGateUp {
  size_t struct_size;
  const void *hidden;
  const int32_t *expert_ids;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  const void *bias;
  void *activated;
  void *stream;
  int64_t routes;
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t local_experts;
  const int32_t *expert_offset;
  NmlNvFp4DType dtype;
} NmlNvFp4DirectExpertGateUp;

typedef struct NmlNvFp4DirectExpertDown {
  size_t struct_size;
  const void *activated;
  const int32_t *expert_ids;
  const uint8_t *payload;
  const uint8_t *block_scales;
  const float *global_scale;
  const void *bias;
  const void *routing_weights;
  void *output;
  void *stream;
  int64_t routes;
  int64_t intermediate_size;
  int64_t hidden_size;
  int64_t local_experts;
  const int32_t *expert_offset;
  NmlNvFp4DType dtype;
} NmlNvFp4DirectExpertDown;

int32_t nml_nvfp4_cuda_linear(const NmlNvFp4Linear *request,
                                char *error_message,
                                size_t error_message_capacity);
int32_t nml_nvfp4_cuda_route_top4(const NmlNvFp4RouteTop4 *request,
                                  char *error_message,
                                  size_t error_message_capacity);
int32_t nml_nvfp4_cuda_linear_top64(const NmlNvFp4LinearTop64 *request,
                                    char *error_message,
                                    size_t error_message_capacity);

int32_t nml_nvfp4_cuda_linear_group3(const NmlNvFp4LinearGroup3 *request,
                                     char *error_message,
                                     size_t error_message_capacity);

int32_t nml_nvfp4_cuda_embedding(const NmlNvFp4Embedding *request,
                                   char *error_message,
                                   size_t error_message_capacity);

int32_t nml_nvfp4_cuda_expert_gate_up(
    const NmlNvFp4ExpertGateUp *request, char *error_message,
    size_t error_message_capacity);

int32_t nml_nvfp4_cuda_expert_down(const NmlNvFp4ExpertDown *request,
                                     char *error_message,
                                     size_t error_message_capacity);

int32_t nml_nvfp4_cuda_direct_expert_gate_up(
    const NmlNvFp4DirectExpertGateUp *request, char *error_message,
    size_t error_message_capacity);

int32_t nml_nvfp4_cuda_direct_expert_down(
    const NmlNvFp4DirectExpertDown *request, char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NML_NVFP4_H_
