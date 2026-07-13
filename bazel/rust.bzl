"""Thin Rust rule front-ends carrying NML-wide invariants.

These macros do not hide dependency edges or synthesize source lists. They only
centralize policy that must be identical for every product crate: Rust edition,
warning discipline, unsafe-block discipline, and supported-host enforcement.
Keeping the wrappers thin preserves ordinary rules_rust semantics and makes a
future native `link_deps` edge visible at its BUILD target.
"""

load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library", "rust_proc_macro", "rust_shared_library", "rust_test")
load("//platforms:defs.bzl", "supported_host_compatible_with")

_EDITION = "2024"

# Unsafe code is expected inside future FFI crates, so it is not forbidden at
# the build layer. Requiring an explicit unsafe block inside every unsafe
# function still makes each operation reviewable. Product crates may impose a
# stronger `#![forbid(unsafe_code)]` policy in their own crate root.
_COMMON_RUSTC_FLAGS = [
    "-Dwarnings",
    "-Dunsafe-op-in-unsafe-fn",
]

def nml_rust_library(
        name,
        rustc_flags = [],
        target_compatible_with = [],
        **kwargs):
    """Declares a Rust library governed by NML's repository invariants."""
    rust_library(
        name = name,
        edition = _EDITION,
        rustc_flags = _COMMON_RUSTC_FLAGS + rustc_flags,
        target_compatible_with = supported_host_compatible_with() + target_compatible_with,
        **kwargs
    )

def nml_rust_binary(
        name,
        rustc_flags = [],
        target_compatible_with = [],
        **kwargs):
    """Declares a product binary without changing normal rules_rust semantics."""
    rust_binary(
        name = name,
        edition = _EDITION,
        rustc_flags = _COMMON_RUSTC_FLAGS + rustc_flags,
        target_compatible_with = supported_host_compatible_with() + target_compatible_with,
        **kwargs
    )

def nml_rust_shared_library(
        name,
        rustc_flags = [],
        target_compatible_with = [],
        **kwargs):
    """Declares a C-compatible Rust shared library under NML policy."""
    rust_shared_library(
        name = name,
        edition = _EDITION,
        rustc_flags = _COMMON_RUSTC_FLAGS + rustc_flags,
        target_compatible_with = supported_host_compatible_with() + target_compatible_with,
        **kwargs
    )

def nml_rust_proc_macro(
        name,
        rustc_flags = [],
        target_compatible_with = [],
        **kwargs):
    """Declares a procedural macro with the same stable host policy as NML."""
    rust_proc_macro(
        name = name,
        edition = _EDITION,
        rustc_flags = _COMMON_RUSTC_FLAGS + rustc_flags,
        target_compatible_with = supported_host_compatible_with() + target_compatible_with,
        **kwargs
    )

def nml_rust_test(
        name,
        rustc_flags = [],
        target_compatible_with = [],
        **kwargs):
    """Declares a durable test target under the same host policy as its code."""
    rust_test(
        name = name,
        edition = _EDITION,
        rustc_flags = _COMMON_RUSTC_FLAGS + rustc_flags,
        target_compatible_with = supported_host_compatible_with() + target_compatible_with,
        **kwargs
    )
