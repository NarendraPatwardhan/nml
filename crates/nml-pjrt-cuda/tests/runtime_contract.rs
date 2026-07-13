//! Runtime contract for the packaged CUDA PJRT backend on a real NVIDIA GPU.

use std::ffi::c_void;
use std::ptr::NonNull;

unsafe extern "C" fn typed_custom_call_stage(_call_frame: *mut c_void) -> *mut c_void {
    std::ptr::null_mut()
}

#[test]
fn packaged_cuda_runtime_matches_the_host() {
    // SAFETY: Bazel starts this test with one application thread, and this is
    // the first operation that can initialize XLA or mutate its environment.
    let runtime = unsafe { nml_pjrt_cuda::Runtime::load() }
        .expect("packaged CUDA PJRT runtime requires an accessible NVIDIA GPU");
    assert!(runtime.directory().join("lib/libpjrt_cuda.so").is_file());
    let custom_calls = runtime
        .custom_calls()
        .expect("CUDA PJRT must expose GPU custom-call registration");
    let handler = unsafe {
        nml_pjrt::GpuCustomCallHandler::from_address(
            NonNull::new(typed_custom_call_stage as *const () as *mut c_void).unwrap(),
        )
    };
    unsafe {
        custom_calls.register(
            "nml$runtime_contract",
            nml_pjrt::GpuCustomCallApi::Typed,
            nml_pjrt::GpuCustomCallHandlers {
                instantiate: Some(handler),
                prepare: Some(handler),
                initialize: Some(handler),
                execute: handler,
            },
        )
    }
    .expect("CUDA PJRT must register every typed custom-call lifecycle handler");
    let client = runtime
        .create_client(nml_pjrt_cuda::ClientOptions::default())
        .expect("CUDA client and every exposed compute capability must be supported");
    let devices = client.devices().expect("CUDA devices must be queryable");
    let values = [1.0f32, -2.5, 7.25, 0.0];
    let host_bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let transfer = client
        .buffer_from_host(
            &host_bytes,
            nml_types::Shape::new(nml_types::DType::F32, &[2, 2]).unwrap(),
            &devices[0],
        )
        .expect("CUDA host-to-device transfer must start");
    transfer
        .done
        .wait()
        .expect("CUDA host-to-device transfer must complete");
    assert_eq!(transfer.buffer.dtype().unwrap(), nml_types::DType::F32);
    assert_eq!(transfer.buffer.dimensions().unwrap(), &[2, 2]);
    let returned = transfer
        .buffer
        .to_host()
        .expect("CUDA device-to-host transfer must complete");
    assert_eq!(returned, host_bytes);

    execute_matmul_contract(&client, &devices[0]);
    execute_complex_contract(&client, &devices[0]);
}

fn execute_matmul_contract(client: &nml_pjrt::Client, device: &nml_pjrt::Device) {
    let mut builder = nml_ir::ProgramBuilder::new();
    let left = builder.input(
        "left",
        nml_types::Shape::new(nml_types::DType::F32, &[3, 5]).unwrap(),
    );
    let right = builder.input(
        "right",
        nml_types::Shape::new(nml_types::DType::F32, &[5, 4]).unwrap(),
    );
    let result = builder.matmul(left, right).unwrap();
    let program = builder.finish(&[result]).unwrap();
    let options =
        nml_xla::CompileOptions::single_device(device.id().unwrap(), nml_xla::Backend::Cuda)
            .unwrap();
    let executable = nml_compiler::compile(client, &program, &options)
        .expect("CUDA XLA compilation must succeed");

    let left_values: Vec<f32> = (0..15).map(|value| value as f32 / 7.0 - 1.0).collect();
    let right_values: Vec<f32> = (0..20).map(|value| value as f32 / 11.0 - 0.5).collect();
    let left_transfer = client
        .buffer_from_host(&f32_bytes(&left_values), left.shape(), device)
        .unwrap();
    let right_transfer = client
        .buffer_from_host(&f32_bytes(&right_values), right.shape(), device)
        .unwrap();
    left_transfer.done.wait().unwrap();
    right_transfer.done.wait().unwrap();
    let execution = executable
        .execute_one(
            &[&left_transfer.buffer, &right_transfer.buffer],
            Some(device),
        )
        .expect("CUDA executable launch must succeed");
    execution.complete.wait().unwrap();
    let actual = decode_f32(&execution.outputs[0].to_host().unwrap());
    let mut expected = vec![0.0f32; 12];
    for row in 0..3 {
        for column in 0..4 {
            expected[row * 4 + column] = (0..5)
                .map(|contract| {
                    left_values[row * 5 + contract] * right_values[contract * 4 + column]
                })
                .sum();
        }
    }
    assert_close(&actual, &expected);
}

fn execute_complex_contract(client: &nml_pjrt::Client, device: &nml_pjrt::Device) {
    let shape = nml_types::Shape::new(nml_types::DType::F32, &[4]).unwrap();
    let mut builder = nml_ir::ProgramBuilder::new();
    let real = builder.input("real", shape);
    let imaginary = builder.input("imaginary", shape);
    let complex = builder.complex(real, imaginary).unwrap();
    let real_result = builder.real(complex).unwrap();
    let imaginary_result = builder.imaginary(complex).unwrap();
    let program = builder.finish(&[real_result, imaginary_result]).unwrap();
    let options =
        nml_xla::CompileOptions::single_device(device.id().unwrap(), nml_xla::Backend::Cuda)
            .unwrap();
    let executable = nml_compiler::compile(client, &program, &options).unwrap();
    let real_values = [1.0f32, -2.0, 3.5, 0.25];
    let imaginary_values = [-4.0f32, 0.5, 8.0, -1.25];
    let real_transfer = client
        .buffer_from_host(&f32_bytes(&real_values), shape, device)
        .unwrap();
    let imaginary_transfer = client
        .buffer_from_host(&f32_bytes(&imaginary_values), shape, device)
        .unwrap();
    let execution = executable
        .execute_one(
            &[&real_transfer.buffer, &imaginary_transfer.buffer],
            Some(device),
        )
        .unwrap();
    execution.complete.wait().unwrap();
    assert_close(
        &decode_f32(&execution.outputs[0].to_host().unwrap()),
        &real_values,
    );
    assert_close(
        &decode_f32(&execution.outputs[1].to_host().unwrap()),
        &imaginary_values,
    );
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-4 + 1e-4 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected}, received {actual}, tolerance {tolerance}"
        );
    }
}
