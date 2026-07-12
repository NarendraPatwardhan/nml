"""Shared host-compatibility policy for NML targets."""

def supported_host_compatible_with():
    """Rejects analysis on hosts outside NML's three supported OS/CPU pairs.

    The empty lists mean "no additional constraint" for supported hosts. Bazel's
    canonical incompatible constraint makes unsupported targets disappear from
    wildcard builds and fail clearly when requested directly.
    """
    return select({
        "//platforms:is_linux_x86_64": [],
        "//platforms:is_linux_aarch64": [],
        "//platforms:is_macos_aarch64": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
