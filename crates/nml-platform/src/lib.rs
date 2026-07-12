//! Compile-time identity for the host running NML.
//!
//! Artifact selection must never guess from filenames or silently fall back to
//! a different architecture. This crate gives future PJRT repository rules and
//! runtime diagnostics one vocabulary for the three supported host pairs.

#![forbid(unsafe_code)]

/// Operating systems on which NML is a product target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperatingSystem {
    Linux,
    MacOs,
}

/// CPU architectures on which NML is a product target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Architecture {
    X86_64,
    Aarch64,
}

/// PJRT backends compiled into this NML build.
///
/// This is deliberately not the same thing as [`Host`]. Linux can compile both
/// variants, and the runtime may discover devices from both plugins.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Cpu,
    Cuda,
}

/// Whether the CPU PJRT loader and package closure are selected by Bazel.
pub const CPU_ENABLED: bool = cfg!(nml_cpu);

/// Whether the CUDA PJRT loader and package closure are selected by Bazel.
pub const CUDA_ENABLED: bool = cfg!(nml_cuda);

impl Backend {
    /// Returns whether this backend was selected in the current Bazel graph.
    pub const fn is_enabled(self) -> bool {
        match self {
            Self::Cpu => CPU_ENABLED,
            Self::Cuda => CUDA_ENABLED,
        }
    }
}

/// The statically selected host contract for the current binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Host {
    pub operating_system: OperatingSystem,
    pub architecture: Architecture,
}

impl Host {
    /// Host selected by rustc's target triple.
    pub const CURRENT: Self = Self {
        operating_system: current_operating_system(),
        architecture: current_architecture(),
    };
}

#[cfg(target_os = "linux")]
const fn current_operating_system() -> OperatingSystem {
    OperatingSystem::Linux
}

#[cfg(target_os = "macos")]
const fn current_operating_system() -> OperatingSystem {
    OperatingSystem::MacOs
}

#[cfg(target_arch = "x86_64")]
const fn current_architecture() -> Architecture {
    Architecture::X86_64
}

#[cfg(target_arch = "aarch64")]
const fn current_architecture() -> Architecture {
    Architecture::Aarch64
}

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "aarch64"),
    all(target_os = "macos", target_arch = "aarch64"),
)))]
compile_error!(
    "NML supports Linux on x86-64/AArch64 and macOS on Apple Silicon; Intel macOS, Windows, and other host targets are unsupported"
);
