#ifndef NML_XLA_H_
#define NML_XLA_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct NmlXlaCompileOptions {
  int32_t num_replicas;
  int32_t num_partitions;
  bool use_shardy_partitioner;
  bool enable_cuda_latency_hiding_scheduler;
  const int64_t* device_ids;
  size_t num_device_ids;
} NmlXlaCompileOptions;

typedef struct NmlXlaBytes {
  uint8_t* data;
  size_t size;
} NmlXlaBytes;

// Serializes XLA's pinned CompileOptionsProto using its generated upb API.
// Returns false on invalid input or allocation failure. The caller owns a
// successful byte range and releases it with nml_xla_bytes_destroy.
bool nml_xla_compile_options_serialize(const NmlXlaCompileOptions* options,
                                       NmlXlaBytes* output);
void nml_xla_bytes_destroy(NmlXlaBytes bytes);

#endif  // NML_XLA_H_
