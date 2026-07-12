//! Executable proof that the Rust target and NML's host vocabulary agree.

use nml_platform::{
    Architecture, Backend, CPU_ENABLED, CUDA_ENABLED, Host, OperatingSystem,
};

#[test]
fn current_host_matches_the_compiler_target() {
    #[cfg(target_os = "linux")]
    assert_eq!(Host::CURRENT.operating_system, OperatingSystem::Linux);

    #[cfg(target_os = "macos")]
    assert_eq!(Host::CURRENT.operating_system, OperatingSystem::MacOs);

    #[cfg(target_arch = "x86_64")]
    assert_eq!(Host::CURRENT.architecture, Architecture::X86_64);

    #[cfg(target_arch = "aarch64")]
    assert_eq!(Host::CURRENT.architecture, Architecture::Aarch64);
}

#[test]
fn enabled_backends_match_the_bazel_configuration() {
    assert_eq!(CPU_ENABLED, cfg!(nml_cpu));
    assert_eq!(CUDA_ENABLED, cfg!(nml_cuda));
    assert_eq!(Backend::Cpu.is_enabled(), cfg!(nml_cpu));
    assert_eq!(Backend::Cuda.is_enabled(), cfg!(nml_cuda));
}
