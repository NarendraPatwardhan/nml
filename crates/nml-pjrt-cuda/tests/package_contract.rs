//! Contract for the complete hermetic CUDA runtime package.

use runfiles::Runfiles;

const DISTRIBUTION_RUNTIME_RLOCATION: &str = env!("NML_CUDA_DISTRIBUTION_RUNTIME_RLOCATION");
const SYSTEM_DRIVER_RUNTIME_RLOCATION: &str = env!("NML_CUDA_SYSTEM_DRIVER_RUNTIME_RLOCATION");

#[test]
fn distribution_runtime_contains_every_required_component() {
    let runfiles = Runfiles::create().expect("Bazel runfiles must initialize");
    let package = runfiles::rlocation!(runfiles, DISTRIBUTION_RUNTIME_RLOCATION)
        .expect("CUDA distribution runtime must be present in runfiles");

    assert_common_runtime(&package);

    for required in [
        "bin/driver_compatibility",
        "lib/compat/libcuda.so.1",
        "lib/compat/libcudadebugger.so.1",
        "lib/compat/libnvidia-gpucomp.so.590.48.01",
        "lib/compat/libnvidia-nvvm.so.4",
        "lib/compat/libnvidia-nvvm70.so.4",
        "lib/compat/libnvidia-pkcs11-openssl3.so.590.48.01",
        "lib/compat/libnvidia-ptxjitcompiler.so.1",
        "lib/compat/libnvidia-tileiras.so.590.48.01",
    ] {
        assert!(
            package.join(required).is_file(),
            "CUDA distribution runtime is missing {required}"
        );
    }
}

#[test]
fn system_driver_runtime_omits_only_the_forward_compatibility_overlay() {
    let runfiles = Runfiles::create().expect("Bazel runfiles must initialize");
    let package = runfiles::rlocation!(runfiles, SYSTEM_DRIVER_RUNTIME_RLOCATION)
        .expect("CUDA system-driver runtime must be present in runfiles");

    assert_common_runtime(&package);
    for excluded in ["bin/driver_compatibility", "lib/compat"] {
        assert!(
            !package.join(excluded).exists(),
            "CUDA system-driver runtime unexpectedly contains {excluded}"
        );
    }
}

fn assert_common_runtime(package: &std::path::Path) {
    for required in [
        "bin/nvlink",
        "bin/ptxas",
        "lib/libcublas.so.13",
        "lib/libcublasLt.so.13",
        "lib/libcudart.so.13",
        "lib/libcudnn.so.9",
        "lib/libcudnn_adv.so.9",
        "lib/libcudnn_cnn.so.9",
        "lib/libcudnn_engines_precompiled.so.9",
        "lib/libcudnn_engines_runtime_compiled.so.9",
        "lib/libcudnn_graph.so.9",
        "lib/libcudnn_heuristic.so.9",
        "lib/libcudnn_ops.so.9",
        "lib/libcufft.so.12",
        "lib/libcupti.so.13",
        "lib/libcusolver.so.12",
        "lib/libcusparse.so.12",
        "lib/libnccl.so.2",
        "lib/libnml_cuda.so.0",
        "lib/libnvJitLink.so.13",
        "lib/libnvrtc-builtins.so.13.1",
        "lib/libnvrtc.so.13",
        "lib/libnvshmem_host.so.3",
        "lib/libnvtx3interop.so",
        "lib/libpjrt_cuda.so",
        "lib/libz.so.1",
        "lib/nvshmem_bootstrap_uid.so.3",
        "lib/nvshmem_transport_ibrc.so.4",
        "nvvm/bin/cicc",
        "nvvm/libdevice/libdevice.10.bc",
    ] {
        assert!(
            package.join(required).is_file(),
            "CUDA runtime is missing common component {required}"
        );
    }
}
