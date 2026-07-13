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

#[test]
fn cuda_13_capability_boundary_is_explicit() {
    use nml_pjrt_cuda::ComputeCapability;

    assert!(!ComputeCapability { major: 7, minor: 4 }.is_supported());
    assert!(ComputeCapability { major: 7, minor: 5 }.is_supported());
    assert!(
        ComputeCapability {
            major: 12,
            minor: 1
        }
        .is_supported()
    );
}

#[test]
fn invalid_allocator_policies_fail_before_plugin_creation() {
    use nml_pjrt_cuda::{Allocator, AllocatorOptions, ClientOptions, Error};

    for memory_fraction in [f32::NAN, -0.1, 1.1] {
        let options = ClientOptions {
            allocator: Allocator::Bfc(AllocatorOptions {
                memory_fraction,
                ..AllocatorOptions::default()
            }),
        };
        assert!(matches!(
            options.validate(),
            Err(Error::InvalidMemoryFraction(_))
        ));
    }

    let options = ClientOptions {
        allocator: Allocator::CudaAsync(AllocatorOptions {
            collective_memory_size_bytes: -1,
            ..AllocatorOptions::default()
        }),
    };
    assert!(matches!(
        options.validate(),
        Err(Error::NegativeCollectiveMemorySize(-1))
    ));
}
