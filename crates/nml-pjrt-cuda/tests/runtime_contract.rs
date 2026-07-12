//! Runtime contract for the packaged CUDA PJRT backend on a real NVIDIA GPU.

#[test]
fn packaged_cuda_runtime_matches_the_host() {
    // SAFETY: Bazel starts this test with one application thread, and this is
    // the first operation that can initialize XLA or mutate its environment.
    let runtime = unsafe { nml_pjrt_cuda::Runtime::load() }
        .expect("packaged CUDA PJRT runtime requires an accessible NVIDIA GPU");
    assert!(runtime.directory().join("lib/libpjrt_cuda.so").is_file());
    runtime
        .custom_calls()
        .expect("CUDA PJRT must expose GPU custom-call registration");
    runtime
        .create_client(nml_pjrt_cuda::ClientOptions::default())
        .expect("CUDA client and every exposed compute capability must be supported");
}
