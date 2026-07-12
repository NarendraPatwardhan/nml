//! Contract for a build graph in which CPU support is deliberately disabled.

#[test]
fn disabled_backend_fails_before_runtime_resolution() {
    assert!(!nml_pjrt_cpu::is_enabled());
    assert!(matches!(
        nml_pjrt_cpu::load(),
        Err(nml_pjrt_cpu::LoadError::Unavailable)
    ));
}
