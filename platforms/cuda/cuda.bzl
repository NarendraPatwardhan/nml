"""ZML-derived hermetic CUDA 13 package graph for NML's two Linux hosts."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

ARCHS = ["linux-x86_64", "linux-sbsa"]

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VERSION = "13.1.1"
CUDA_VARIANT = "cuda13.1"
CUDA_REDIST_JSON_SHA256 = "97cf605ccc4751825b1865f4af571c9b50dd29ffd13e9a38b296a9ecb1f0d422"

CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
CUDNN_VERSION = "9.19.1"
CUDNN_REDIST_JSON_SHA256 = "ee7bd6872b8611017bfc9ac99a4a71932652d1851b5917aa2c66bf29a12f8fd4"

NVSHMEM_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"
NVSHMEM_VERSION = "3.5.19"
NVSHMEM_REDIST_JSON_SHA256 = "6dced4193eb728542504b346cfb768da6e3de2abca0cded95fda3a69729994d2"

PJRT_CUDA_RELEASE = "manual-2026-07-03T00-10-30Z"

CUDA_COMPAT_FILES = [
    "libcuda.so.1",
    "libcudadebugger.so.1",
    "libnvidia-nvvm.so.4",
    "libnvidia-nvvm70.so.4",
    "libnvidia-ptxjitcompiler.so.1",
]

_BUILD_HEADER = """\
package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@nml//bazel:patchelf.bzl", "patchelf")
"""

def _filegroup(name, srcs):
    return """\
filegroup(
    name = {name},
    srcs = {srcs},
)
""".format(name = repr(name), srcs = repr(srcs))

def _cuda_package_builds():
    compat_rules = []
    for library in CUDA_COMPAT_FILES:
        compat_rules.append("""\
patchelf(
    name = {name},
    src = {src},
    local = True,
    soname = {soname},
    set_rpath = "$ORIGIN",
)
""".format(
            name = repr(library + ".patchelf"),
            src = repr("compat/" + library),
            soname = repr(library),
        ))
    compat_rules.append("""\
filegroup(
    name = "cuda_compat",
    srcs = [
        "compat/libnvidia-gpucomp.so.590.48.01",
        "compat/libnvidia-tileiras.so.590.48.01",
    ] + {patched} + select({{
        "@llvm//platforms/config:linux_x86_64": ["compat/libnvidia-pkcs11-openssl3.so.590.48.01"],
        "@llvm//platforms/config:linux_aarch64": [],
    }}),
)
""".format(patched = repr([":" + library + ".patchelf" for library in CUDA_COMPAT_FILES])))

    return {
        "cuda_nvml_dev": """\
cc_library(
    name = "nvml",
    hdrs = ["include/nvml.h"],
    includes = ["include"],
)
""",
        "cuda_cudart": """\
cc_library(
    name = "cuda",
    hdrs = ["include/cuda.h"],
    includes = ["include"],
)
""" + _filegroup("cuda_cudart", ["lib/libcudart.so.13"]),
        "cuda_cupti": _filegroup("cuda_cupti", ["lib/libcupti.so.13"]),
        "cuda_nvtx": """\
cc_library(
    name = "headers",
    hdrs = glob(["include/nvtx3/**"]),
    includes = ["include"],
)
""" + _filegroup("cuda_nvtx", ["lib/libnvtx3interop.so"]),
        "cuda_compat": "\n".join(compat_rules),
        "libcufft": _filegroup("libcufft", ["lib/libcufft.so.12"]),
        "libcusolver": _filegroup("libcusolver", ["lib/libcusolver.so.12"]),
        "libcusparse": _filegroup("libcusparse", ["lib/libcusparse.so.12"]),
        "libnvjitlink": _filegroup("libnvjitlink", ["lib/libnvJitLink.so.13"]),
        "cuda_nvcc": _filegroup("cuda_nvcc", ["bin/ptxas", "bin/nvlink"]) + """\
cc_import(
    name = "nvptxcompiler",
    static_library = "lib/libnvptxcompiler_static.a",
)
""",
        "libnvvm": _filegroup("libnvvm", ["nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc"]),
        "cuda_nvrtc": _filegroup("cuda_nvrtc", ["lib/libnvrtc.so.13", "lib/libnvrtc-builtins.so.13.1"]),
        "libcublas": _filegroup("libcublas", ["lib/libcublasLt.so.13", "lib/libcublas.so.13"]),
    }

CUDA_PACKAGES = _cuda_package_builds()

CUDNN_PACKAGES = {
    "cudnn": _filegroup("cudnn", [
        "lib/libcudnn.so.9",
        "lib/libcudnn_adv.so.9",
        "lib/libcudnn_ops.so.9",
        "lib/libcudnn_cnn.so.9",
        "lib/libcudnn_graph.so.9",
        "lib/libcudnn_engines_precompiled.so.9",
        "lib/libcudnn_engines_runtime_compiled.so.9",
        "lib/libcudnn_heuristic.so.9",
    ]),
}

NVSHMEM_PACKAGES = {
    "libnvshmem": _filegroup("libnvshmem", [
        "lib/libnvshmem_host.so.3",
        "lib/nvshmem_bootstrap_uid.so.3",
        "lib/nvshmem_transport_ibrc.so.4",
    ]),
}

# NVIDIA's redistributions also contain multi-gigabyte static development
# archives. NML's retained ZML runtime graph names only the files below, so
# extracting unrelated payload wastes repository disk without adding a runtime
# capability. Patterns include versioned shared-library targets as well as the
# stable sonames referenced by BUILD targets.
_REDIST_INCLUDES = {
    "cuda_nvml_dev": ["include/nvml.h"],
    "cuda_cudart": ["include/cuda.h", "lib/libcudart.so*"],
    "cuda_cupti": ["lib/libcupti.so*"],
    "cuda_nvtx": ["include/nvtx3/*", "lib/libnvtx3interop.so*"],
    "cuda_compat": ["compat/libcuda.so*", "compat/libcudadebugger.so*", "compat/libnvidia-*.so*"],
    "libcufft": ["lib/libcufft.so*"],
    "libcusolver": ["lib/libcusolver.so*"],
    "libcusparse": ["lib/libcusparse.so*"],
    "libnvjitlink": ["lib/libnvJitLink.so*"],
    "cuda_nvcc": ["bin/ptxas", "bin/nvlink"],
    "libnvvm": ["nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc"],
    "cuda_nvrtc": ["lib/libnvrtc.so*", "lib/libnvrtc-builtins.so*"],
    "libcublas": ["lib/libcublas.so*", "lib/libcublasLt.so*"],
    "cudnn": ["lib/libcudnn*.so*"],
    "libnvshmem": [
        "lib/libnvshmem_host.so*",
        "lib/nvshmem_bootstrap_uid.so*",
        "lib/nvshmem_transport_ibrc.so*",
    ],
}

_PJRT_CUDA_ASSETS = {
    "amd64": {
        "sha256": "6380f724fe21b25dc9231f3ec468ae92e39d21f4bae9377bfa8ff01972521e0d",
        "url": "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-cuda_linux-amd64.tar.gz",
    },
    "arm64": {
        "sha256": "98dc33f0740bc37f3a25611f5aad423cc46eb69679e4d45a22f7fc4dfa3efd8d",
        "url": "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-cuda_linux-arm64.tar.gz",
    },
}

_NCCL_ASSETS = {
    "amd64": {
        "sha256": "2a321629f49490e4e0122ecb578a4b4a6f89e72740dd988e04dfa4758fab7fc3",
        "url": "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.29.3-py3-none-manylinux_2_18_x86_64.whl",
    },
    "arm64": {
        "sha256": "eab9f5c565ab3326906f1d1b5be5773a174c2a1b47002faed76f9e957392f713",
        "url": "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.29.3-py3-none-manylinux_2_18_aarch64.whl",
    },
}

_ZLIB_ASSETS = {
    "amd64": {
        "path": "lib/x86_64-linux-gnu/libz.so.1",
        "sha256": "d7dd1d1411fedf27f5e27650a6eff20ef294077b568f4c8c5e51466dc7c08ce4",
        "url": "https://snapshot-cloudflare.debian.org/archive/debian/20250711T030400Z/pool/main/z/zlib/zlib1g_1.2.13.dfsg-1_amd64.deb",
    },
    "arm64": {
        "path": "lib/aarch64-linux-gnu/libz.so.1",
        "sha256": "52b8b8a145bbe1956bba82034f77022cbef0c3d0885c9e32d9817a7932fe1913",
        "url": "https://snapshot-cloudflare.debian.org/archive/debian/20250711T030400Z/pool/main/z/zlib/zlib1g_1.2.13.dfsg-1_arm64.deb",
    },
}

def _read_redist_json(mctx, prefix, version, sha256):
    output = ".{}.json".format(sha256)
    mctx.download(
        output = output,
        sha256 = sha256,
        url = prefix + "redistrib_{}.json".format(version),
    )
    return json.decode(mctx.read(output))

def _create_redist_repositories(packages, redist, prefix, variant):
    for package, build in packages.items():
        package_data = redist[package]
        for arch in ARCHS:
            arch_data = package_data.get(arch)
            if not arch_data:
                fail("{} has no CUDA redistribution for {}".format(package, arch))
            arch_data = arch_data.get(variant, None) or arch_data
            relative_path = arch_data["relative_path"]
            http_archive(
                name = package + "_" + arch.replace("-", "_"),
                build_file_content = _BUILD_HEADER + build,
                includes = _REDIST_INCLUDES[package],
                sha256 = arch_data["sha256"],
                strip_prefix = paths.basename(relative_path).replace(".tar.xz", ""),
                url = prefix + relative_path,
            )

def _cuda_impl(mctx):
    cuda_redist = _read_redist_json(mctx, CUDA_REDIST_PREFIX, CUDA_VERSION, CUDA_REDIST_JSON_SHA256)
    cudnn_redist = _read_redist_json(mctx, CUDNN_REDIST_PREFIX, CUDNN_VERSION, CUDNN_REDIST_JSON_SHA256)
    nvshmem_redist = _read_redist_json(mctx, NVSHMEM_REDIST_PREFIX, NVSHMEM_VERSION, NVSHMEM_REDIST_JSON_SHA256)

    _create_redist_repositories(CUDA_PACKAGES, cuda_redist, CUDA_REDIST_PREFIX, CUDA_VARIANT)
    _create_redist_repositories(CUDNN_PACKAGES, cudnn_redist, CUDNN_REDIST_PREFIX, "cuda13")
    _create_redist_repositories(NVSHMEM_PACKAGES, nvshmem_redist, NVSHMEM_REDIST_PREFIX, "cuda13")

    for arch, asset in _ZLIB_ASSETS.items():
        http_deb_archive(
            name = "zlib1g_linux_{}".format(arch),
            build_file_content = _BUILD_HEADER + _filegroup("zlib1g", [asset["path"]]),
            sha256 = asset["sha256"],
            urls = [asset["url"]],
        )

    for arch, asset in _NCCL_ASSETS.items():
        http_archive(
            name = "nccl_linux_{}".format(arch),
            build_file_content = _BUILD_HEADER + _filegroup("nccl", ["nvidia/nccl/lib/libnccl.so.2"]),
            sha256 = asset["sha256"],
            type = "zip",
            urls = [asset["url"]],
        )

    for arch, asset in _PJRT_CUDA_ASSETS.items():
        http_archive(
            name = "libpjrt_cuda_linux_{}".format(arch),
            build_file = Label("//platforms/cuda:libpjrt_cuda.BUILD.bazel"),
            sha256 = asset["sha256"],
            url = asset["url"].format(release = PJRT_CUDA_RELEASE),
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "cuda_nvml_dev_linux_sbsa",
            "cuda_nvml_dev_linux_x86_64",
            "cuda_nvtx_linux_sbsa",
            "cuda_nvtx_linux_x86_64",
            "libpjrt_cuda_linux_amd64",
            "libpjrt_cuda_linux_arm64",
            "zlib1g_linux_amd64",
            "zlib1g_linux_arm64",
        ],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(implementation = _cuda_impl)
