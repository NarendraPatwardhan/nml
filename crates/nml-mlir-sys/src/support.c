#include "nml_mlir.h"

bool nml_mlir_logical_result_is_success(MlirLogicalResult result) {
  return mlirLogicalResultIsSuccess(result);
}

MlirLogicalResult nml_mlir_logical_result_success(void) {
  return mlirLogicalResultSuccess();
}
