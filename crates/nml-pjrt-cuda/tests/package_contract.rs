//! Contract for the complete hermetic CUDA runtime package.

use runfiles::Runfiles;

const CUDA_RUNTIME_RLOCATION: &str = env!("NML_CUDA_RUNTIME_CONTRACT_RLOCATION");

#[test]
fn packaged_cuda_runtime_contains_every_required_component() {
    let runfiles = Runfiles::create().expect("Bazel runfiles must initialize");
    let package = runfiles::rlocation!(runfiles, CUDA_RUNTIME_RLOCATION)
        .expect("CUDA runtime tree must be present in runfiles");

    for required in [
        "bin/driver_compatibility",
        "bin/nvlink",
        "bin/ptxas",
        "lib/compat/libcuda.so.1",
        "lib/compat/libcudadebugger.so.1",
        "lib/compat/libnvidia-gpucomp.so.590.48.01",
        "lib/compat/libnvidia-nvvm.so.4",
        "lib/compat/libnvidia-nvvm70.so.4",
        "lib/compat/libnvidia-pkcs11-openssl3.so.590.48.01",
        "lib/compat/libnvidia-ptxjitcompiler.so.1",
        "lib/compat/libnvidia-tileiras.so.590.48.01",
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
            "CUDA runtime is missing {required}"
        );
    }
}
