"""Pinned CPU PJRT plugin repositories for every supported NML host."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_RELEASE = "manual-2026-07-03T00-10-30Z"
_PREFIX = "https://github.com/zml/pjrt-artifacts/releases/download/{}/".format(_RELEASE)

_BUILD = """\
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libpjrt_cpu",
    srcs = ["{library}"],
)
"""

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD.format(library = "libpjrt_cpu.so"),
        sha256 = "65e631db0f842845e7799d245a414b361a3c3e77bf4cc0547c20c71f28a9fd70",
        url = _PREFIX + "pjrt-cpu_linux-amd64.tar.gz",
    )
    http_archive(
        name = "libpjrt_cpu_linux_arm64",
        build_file_content = _BUILD.format(library = "libpjrt_cpu.so"),
        sha256 = "4c4d4021d4cde06a67a68d19d1d3ee4f0765ecb1f3a18f9610e55723095a51b1",
        url = _PREFIX + "pjrt-cpu_linux-arm64.tar.gz",
    )
    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD.format(library = "libpjrt_cpu.dylib"),
        sha256 = "14c85504d801c75fa8d157ce951a2644d8a8d7983346b3ac281aa7f64abf8390",
        url = _PREFIX + "pjrt-cpu_darwin-arm64.tar.gz",
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(implementation = _cpu_pjrt_plugin_impl)
