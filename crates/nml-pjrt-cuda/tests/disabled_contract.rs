//! Contract for a build graph in which CUDA support is deliberately disabled.

#[test]
fn disabled_backend_has_no_runtime() {
    assert!(!nml_pjrt_cuda::is_enabled());
    // SAFETY: the disabled branch returns before process environment access.
    assert!(matches!(
        unsafe { nml_pjrt_cuda::Runtime::load() },
        Err(nml_pjrt_cuda::Error::Disabled)
    ));
}
