"""Pinned XLA source repository used by every compiler and PJRT ABI edge."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _xla_source_impl(module_ctx):
    git_repository(
        name = "xla",
        remote = "https://github.com/openxla/xla.git",
        commit = "41370d1124c74d7b93a207136a636d8c631cbed9",
        patches = ["//third_party/xla:cuda-root-path-local-defines.patch"],
        patch_args = ["-p1"],
    )
    return module_ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["xla"],
        root_module_direct_dev_deps = [],
    )

xla_source = module_extension(implementation = _xla_source_impl)
