#include "nml_mlir.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

extern "C" {

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, tt, mlir::triton::TritonDialect)

MlirType nml_mlir_triton_pointer_type(MlirType pointee, int32_t address_space) {
  return wrap(mlir::triton::PointerType::get(unwrap(pointee), address_space));
}

bool nml_mlir_type_is_triton_pointer(MlirType type) {
  return mlir::isa<mlir::triton::PointerType>(unwrap(type));
}

MlirType nml_mlir_triton_tensor_descriptor_type(intptr_t rank,
                                                const int64_t *shape,
                                                MlirType element) {
  return wrap(
      mlir::triton::TensorDescType::get(llvm::ArrayRef<int64_t>(shape, rank),
                                        unwrap(element), mlir::Attribute{}));
}

bool nml_mlir_type_is_triton_tensor_descriptor(MlirType type) {
  return mlir::isa<mlir::triton::TensorDescType>(unwrap(type));
}

MlirAttribute nml_mlir_triton_program_dimension(MlirContext context,
                                                int32_t value) {
  return wrap(mlir::triton::ProgramIDDimAttr::get(
      unwrap(context), static_cast<mlir::triton::ProgramIDDim>(value)));
}

MlirAttribute nml_mlir_triton_cache_modifier(MlirContext context,
                                             int32_t value) {
  return wrap(mlir::triton::CacheModifierAttr::get(
      unwrap(context), static_cast<mlir::triton::CacheModifier>(value)));
}

MlirAttribute nml_mlir_triton_eviction_policy(MlirContext context,
                                              int32_t value) {
  return wrap(mlir::triton::EvictionPolicyAttr::get(
      unwrap(context), static_cast<mlir::triton::EvictionPolicy>(value)));
}

MlirAttribute nml_mlir_triton_input_precision(MlirContext context,
                                              int32_t value) {
  return wrap(mlir::triton::InputPrecisionAttr::get(
      unwrap(context), static_cast<mlir::triton::InputPrecision>(value)));
}

} // extern "C"
