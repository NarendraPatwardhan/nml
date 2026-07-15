#ifndef NML_MLIR_H_
#define NML_MLIR_H_

// This umbrella is intentionally small: bindgen sees only the stable C APIs
// that NML owns. C++ implementation headers remain behind their Bazel targets.
#include "shardy/integrations/c/attributes.h"
#include "shardy/integrations/c/dialect.h"
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Support.h>
#include <mlir-c/Transforms.h>
#include <stablehlo/integrations/c/StablehloDialect.h>
#include <stablehlo/integrations/c/StablehloDialectApi.h>
#include <stablehlo/integrations/c/StablehloAttributes.h>

// Triton is part of the XLA-pinned dependency graph.  Its project does not
// publish a complete C API, so NML owns a deliberately narrow bridge rather
// than exposing C++ headers to Rust.
#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, tt);

// MLIR defines these operations as C inline functions. Exporting two narrow
// wrappers keeps Rust from depending on the private representation of
// MlirLogicalResult or on bindgen's inline-function generation strategy.
bool nml_mlir_logical_result_is_success(MlirLogicalResult result);
MlirLogicalResult nml_mlir_logical_result_success(void);
MlirType nml_mlir_triton_pointer_type(MlirType pointee, int32_t address_space);
bool nml_mlir_type_is_triton_pointer(MlirType type);
MlirType nml_mlir_triton_tensor_descriptor_type(intptr_t rank,
                                                const int64_t *shape,
                                                MlirType element);
bool nml_mlir_type_is_triton_tensor_descriptor(MlirType type);
MlirAttribute nml_mlir_triton_program_dimension(MlirContext context,
                                                int32_t value);
MlirAttribute nml_mlir_triton_cache_modifier(MlirContext context,
                                             int32_t value);
MlirAttribute nml_mlir_triton_eviction_policy(MlirContext context,
                                              int32_t value);
MlirAttribute nml_mlir_triton_input_precision(MlirContext context,
                                              int32_t value);

#ifdef __cplusplus
}
#endif

#endif // NML_MLIR_H_
