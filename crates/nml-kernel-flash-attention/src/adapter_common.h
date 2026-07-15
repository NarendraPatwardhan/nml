#ifndef NML_FLASH_ATTENTION_ADAPTER_COMMON_H_
#define NML_FLASH_ATTENTION_ADAPTER_COMMON_H_

#include <cstdint>
#include <limits>

namespace nml::flash_attention::internal {

// Both upstream ABIs store rounded sequence lengths in signed 32-bit fields.
// Perform the arithmetic in I64 so hostile or corrupt FFI input cannot trigger
// signed overflow before the adapter has a chance to reject it.
inline bool round_sequence_length(int32_t length, int32_t *rounded) {
  constexpr int64_t kAlignment = 128;
  const int64_t wide_length = length;
  const int64_t wide_rounded =
      ((wide_length + kAlignment - 1) / kAlignment) * kAlignment;
  if (length <= 0 || wide_rounded > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  *rounded = static_cast<int32_t>(wide_rounded);
  return true;
}

} // namespace nml::flash_attention::internal

#endif // NML_FLASH_ATTENTION_ADAPTER_COMMON_H_
