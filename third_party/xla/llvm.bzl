"""Configures the LLVM/MLIR tree pinned by XLA rather than a second LLVM."""

load("@xla_llvm_raw//utils/bazel:configure.bzl", _llvm_configure = "llvm_configure")

def _llvm_impl(module_ctx):
    targets = {}
    for module in module_ctx.modules:
        for configuration in module.tags.configure:
            for target in configuration.targets:
                targets[target] = True
    _llvm_configure(
        name = "llvm-project",
        targets = targets.keys(),
    )
    return module_ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

llvm = module_extension(
    implementation = _llvm_impl,
    tag_classes = {
        "configure": tag_class(
            attrs = {"targets": attr.string_list(default = [])},
        ),
    },
)
