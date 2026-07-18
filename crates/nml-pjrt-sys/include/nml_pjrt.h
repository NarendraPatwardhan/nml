#ifndef NML_PJRT_H_
#define NML_PJRT_H_

// Keep the plugin registration extension and the handler call-frame ABI in a
// single bindgen translation unit. They are two views of one pinned XLA/PJRT
// boundary and must never be generated from different XLA revisions.
#include "xla/ffi/api/c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"

#endif // NML_PJRT_H_
