"""Hermetic repository containing the PJRT C ABI header from NML's XLA pin.

The header is self-contained apart from C standard-library headers. Fetching it
alone keeps the raw FFI boundary independent of XLA's compiler implementation
graph while preserving exact source identity. The GPU extension is included
because custom-call registration is part of NML's CUDA execution substrate.
"""

_XLA_COMMIT = "41370d1124c74d7b93a207136a636d8c631cbed9"
_PJRT_C_API_SHA256 = "9cd904fb8c92482d7a279cec5047f914ded9a55fb03d52f00492ff170e180706"
_PJRT_GPU_EXTENSION_SHA256 = "b9d73255cc297e0be60f350c03a5c4b96ed40e17c49fdfb404be8c055922a960"

def _pjrt_headers_repository_impl(rctx):
    path = "xla/pjrt/c/pjrt_c_api.h"
    rctx.download(
        output = path,
        sha256 = _PJRT_C_API_SHA256,
        url = "https://raw.githubusercontent.com/openxla/xla/{}/{}".format(_XLA_COMMIT, path),
    )
    gpu_path = "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
    rctx.download(
        output = gpu_path,
        sha256 = _PJRT_GPU_EXTENSION_SHA256,
        url = "https://raw.githubusercontent.com/openxla/xla/{}/{}".format(_XLA_COMMIT, gpu_path),
    )
    rctx.file("BUILD.bazel", """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "pjrt_c_api_headers",
    hdrs = [
        "xla/pjrt/c/pjrt_c_api.h",
        "xla/pjrt/c/pjrt_c_api_gpu_extension.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

exports_files([
    "xla/pjrt/c/pjrt_c_api.h",
    "xla/pjrt/c/pjrt_c_api_gpu_extension.h",
])
""")

_pjrt_headers_repository = repository_rule(
    implementation = _pjrt_headers_repository_impl,
)

def _pjrt_headers_impl(mctx):
    _pjrt_headers_repository(name = "xla_pjrt_headers")
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["xla_pjrt_headers"],
        root_module_direct_dev_deps = [],
    )

pjrt_headers = module_extension(implementation = _pjrt_headers_impl)
