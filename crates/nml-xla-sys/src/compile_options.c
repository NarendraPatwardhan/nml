#include "nml_xla.h"

#include <stdlib.h>
#include <string.h>

#include "upb/base/string_view.h"
#include "upb/message/map.h"
#include "upb/mem/arena.h"
#include "upb/wire/encode.h"
#include "xla/pjrt/proto/compile_options.upb.h"
#include "xla/xla_data.upb.h"

static bool set_bool_override(upb_Map* map, const char* name, bool value,
                              upb_Arena* arena) {
  xla_OptionOverrideProto* option = xla_OptionOverrideProto_new(arena);
  if (option == NULL) return false;
  xla_OptionOverrideProto_set_bool_field(option, value);
  upb_MessageValue key;
  memset(&key, 0, sizeof(key));
  key.str_val = upb_StringView_FromString(name);
  upb_MessageValue mapped;
  memset(&mapped, 0, sizeof(mapped));
  mapped.msg_val = (upb_Message*)option;
  return upb_Map_Set(map, key, mapped, arena);
}

bool nml_xla_compile_options_serialize(const NmlXlaCompileOptions* options,
                                       NmlXlaBytes* output) {
  if (options == NULL || output == NULL || options->num_replicas <= 0 ||
      options->num_partitions <= 0 ||
      options->num_device_ids !=
          (size_t)options->num_replicas * (size_t)options->num_partitions) {
    return false;
  }
  output->data = NULL;
  output->size = 0;
  upb_Arena* arena = upb_Arena_New();
  if (arena == NULL) return false;

  xla_CompileOptionsProto* compile = xla_CompileOptionsProto_new(arena);
  xla_ExecutableBuildOptionsProto* build =
      xla_ExecutableBuildOptionsProto_new(arena);
  xla_DeviceAssignmentProto* assignment = xla_DeviceAssignmentProto_new(arena);
  if (compile == NULL || build == NULL || assignment == NULL) goto failure;

  xla_ExecutableBuildOptionsProto_set_device_ordinal(build, -1);
  xla_ExecutableBuildOptionsProto_set_num_replicas(build,
                                                    options->num_replicas);
  xla_ExecutableBuildOptionsProto_set_num_partitions(build,
                                                      options->num_partitions);
  xla_ExecutableBuildOptionsProto_set_use_spmd_partitioning(build, true);
  xla_ExecutableBuildOptionsProto_set_use_shardy_partitioner(
      build, options->use_shardy_partitioner);

  xla_DeviceAssignmentProto_set_replica_count(assignment,
                                               options->num_replicas);
  xla_DeviceAssignmentProto_set_computation_count(assignment,
                                                   options->num_partitions);
  for (int32_t partition = 0; partition < options->num_partitions;
       ++partition) {
    xla_DeviceAssignmentProto_ComputationDevice* computation =
        xla_DeviceAssignmentProto_add_computation_devices(assignment, arena);
    if (computation == NULL) goto failure;
    for (int32_t replica = 0; replica < options->num_replicas; ++replica) {
      size_t index = (size_t)partition * (size_t)options->num_replicas +
                     (size_t)replica;
      if (!xla_DeviceAssignmentProto_ComputationDevice_add_replica_device_ids(
              computation, options->device_ids[index], arena)) {
        goto failure;
      }
    }
  }
  xla_ExecutableBuildOptionsProto_set_device_assignment(build, assignment);
  xla_CompileOptionsProto_set_executable_build_options(compile, build);

  if (options->enable_cuda_latency_hiding_scheduler) {
    upb_Map* overrides =
        _xla_CompileOptionsProto_env_option_overrides_mutable_upb_map(compile,
                                                                      arena);
    if (overrides == NULL ||
        !set_bool_override(overrides,
                           "xla_gpu_enable_latency_hiding_scheduler", true,
                           arena)) {
      goto failure;
    }
  }

  size_t encoded_size = 0;
  char* encoded = xla_CompileOptionsProto_serialize_ex(
      compile, kUpb_EncodeOption_Deterministic, arena, &encoded_size);
  if (encoded == NULL && encoded_size != 0) goto failure;
  output->data = (uint8_t*)malloc(encoded_size == 0 ? 1 : encoded_size);
  if (output->data == NULL) goto failure;
  memcpy(output->data, encoded, encoded_size);
  output->size = encoded_size;
  upb_Arena_Free(arena);
  return true;

failure:
  upb_Arena_Free(arena);
  return false;
}

void nml_xla_bytes_destroy(NmlXlaBytes bytes) { free(bytes.data); }
