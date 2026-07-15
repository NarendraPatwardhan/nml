"""Execution policy for Bazel's implicit test exec group."""

def _empty_test_toolchain_impl(_ctx):
    # Test rules use this toolchain only to select an execution platform. The
    # test executable and its runfiles remain the complete runtime payload.
    return [platform_common.ToolchainInfo()]

empty_test_toolchain = rule(
    implementation = _empty_test_toolchain_impl,
)
