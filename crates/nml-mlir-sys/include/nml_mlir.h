#ifndef NML_MLIR_H_
#define NML_MLIR_H_

// This umbrella is intentionally small: bindgen sees only the stable C APIs
// that NML owns. C++ implementation headers remain behind their Bazel targets.
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Support.h>
#include <mlir-c/Transforms.h>
#include "shardy/integrations/c/dialect.h"
#include "shardy/integrations/c/attributes.h"
#include <stablehlo/integrations/c/StablehloDialect.h>
#include <stablehlo/integrations/c/StablehloDialectApi.h>

#ifdef __cplusplus
extern "C" {
#endif

// MLIR defines these operations as C inline functions. Exporting two narrow
// wrappers keeps Rust from depending on the private representation of
// MlirLogicalResult or on bindgen's inline-function generation strategy.
bool nml_mlir_logical_result_is_success(MlirLogicalResult result);
MlirLogicalResult nml_mlir_logical_result_success(void);

#ifdef __cplusplus
}
#endif

#endif  // NML_MLIR_H_
