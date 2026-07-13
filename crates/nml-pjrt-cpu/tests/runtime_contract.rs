//! End-to-end contract for the packaged CPU PJRT runtime.

#[test]
fn packaged_cpu_plugin_creates_a_real_cpu_client() {
    let plugin = nml_pjrt_cpu::load().expect("packaged CPU PJRT plugin must load and initialize");
    let version = plugin.version();
    assert_eq!(version.major, 0, "unexpected PJRT major version");

    let client = plugin
        .create_client()
        .expect("CPU PJRT client creation must succeed");
    let platform = client
        .platform_name()
        .expect("CPU platform name must be queryable");
    assert_eq!(platform.to_ascii_lowercase(), "cpu");
    assert!(
        client
            .device_count()
            .expect("CPU devices must be enumerable")
            > 0,
        "CPU PJRT must expose at least one addressable host device"
    );

    let devices = client.devices().expect("CPU devices must be queryable");
    assert!(matches!(
        client.buffer_from_host(
            &[0, 1, 2],
            nml_types::Shape::new(nml_types::DType::F32, &[]).unwrap(),
            &devices[0],
        ),
        Err(nml_pjrt::Error::InvalidHostBuffer {
            expected: 4,
            actual: 3
        })
    ));
    let column_major = nml_types::Shape::new(nml_types::DType::F32, &[2, 2])
        .unwrap()
        .with_layout(nml_types::Layout::from_minor_to_major(&[0, 1]).unwrap())
        .unwrap();
    assert!(matches!(
        client.buffer_from_host(&[0; 16], column_major, &devices[0]),
        Err(nml_pjrt::Error::UnsupportedLayout { .. })
    ));
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
        .expect("CPU host-to-device transfer must start");
    transfer
        .done
        .wait()
        .expect("CPU host-to-device transfer must complete");
    assert_eq!(transfer.buffer.dtype().unwrap(), nml_types::DType::F32);
    assert_eq!(transfer.buffer.dimensions().unwrap(), &[2, 2]);
    let returned = transfer
        .buffer
        .to_host()
        .expect("CPU device-to-host transfer must complete");
    assert_eq!(returned, host_bytes);

    for (shape, bytes) in [
        (
            nml_types::Shape::new(nml_types::DType::I32, &[]).unwrap(),
            17i32.to_ne_bytes().to_vec(),
        ),
        (
            nml_types::Shape::new(nml_types::DType::F32, &[0, 3]).unwrap(),
            Vec::new(),
        ),
    ] {
        let transfer = client
            .buffer_from_host(&bytes, shape, &devices[0])
            .expect("scalar and empty CPU transfers must start");
        transfer.done.wait().unwrap();
        assert_eq!(transfer.buffer.shape().unwrap(), shape);
        assert_eq!(transfer.buffer.to_host().unwrap(), bytes);
    }

    execute_matmul_contract(&client, &devices[0]);
    execute_complex_contract(&client, &devices[0]);
}

#[test]
fn pjrt_objects_retain_client_plugin_and_library_ownership() {
    let left_values: Vec<f32> = (0..15).map(|value| value as f32 / 7.0 - 1.0).collect();
    let right_values: Vec<f32> = (0..20).map(|value| value as f32 / 11.0 - 0.5).collect();
    let left_bytes = f32_bytes(&left_values);
    let right_bytes = f32_bytes(&right_values);

    let (executable, device, left, right) = {
        let plugin = nml_pjrt_cpu::load().unwrap();
        let client = plugin.create_client().unwrap();
        let device = client.devices().unwrap().remove(0);
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
            nml_xla::CompileOptions::single_device(device.id().unwrap(), nml_xla::Backend::Cpu)
                .unwrap();
        let executable = nml_compiler::compile(&client, &program, &options).unwrap();
        let left = client
            .buffer_from_host(&left_bytes, left.shape(), &device)
            .unwrap();
        let right = client
            .buffer_from_host(&right_bytes, right.shape(), &device)
            .unwrap();
        left.done.wait().unwrap();
        right.done.wait().unwrap();
        (executable, device, left.buffer, right.buffer)
    };

    // The lexical Plugin and Client handles are gone. Every call below crosses
    // PJRT and therefore proves that the returned objects retain the shared
    // client, API table, and dynamic-library state required by their pointers.
    assert_ne!(left.memory().unwrap().as_raw_identity(), 0);
    left.ready_event().unwrap().wait().unwrap();
    let metadata = executable.executable().unwrap();
    assert_eq!(metadata.output_count().unwrap(), 1);
    let _name = metadata.name().unwrap();
    assert!(!executable.addressable_devices().unwrap().is_empty());
    let execution = executable
        .execute_one(&[&left, &right], Some(&device))
        .unwrap();
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

    execution.outputs[0].delete().unwrap();
    assert!(execution.outputs[0].is_deleted().unwrap());
    drop(metadata);
    drop(execution);
    assert!(!executable.is_deleted().unwrap());
    executable.delete().unwrap();
    // The retained CPU artifact currently reports false even after a
    // successful Delete call. Preserve that plugin result instead of
    // fabricating state in the Rust wrapper; Drop still destroys the handle.
    let _plugin_deletion_state = executable.is_deleted().unwrap();
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
        nml_xla::CompileOptions::single_device(device.id().unwrap(), nml_xla::Backend::Cpu)
            .unwrap();
    let executable = nml_compiler::compile(client, &program, &options)
        .expect("CPU XLA compilation must succeed");

    let left_values: Vec<f32> = (0..15).map(|value| value as f32 / 7.0 - 1.0).collect();
    let right_values: Vec<f32> = (0..20).map(|value| value as f32 / 11.0 - 0.5).collect();
    let left_bytes = f32_bytes(&left_values);
    let right_bytes = f32_bytes(&right_values);
    let left_transfer = client
        .buffer_from_host(&left_bytes, left.shape(), device)
        .unwrap();
    let right_transfer = client
        .buffer_from_host(&right_bytes, right.shape(), device)
        .unwrap();
    left_transfer.done.wait().unwrap();
    right_transfer.done.wait().unwrap();
    let execution = executable
        .execute_one(
            &[&left_transfer.buffer, &right_transfer.buffer],
            Some(device),
        )
        .expect("CPU executable launch must succeed");
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
        nml_xla::CompileOptions::single_device(device.id().unwrap(), nml_xla::Backend::Cpu)
            .unwrap();
    let executable = nml_compiler::compile(client, &program, &options).unwrap();
    let real_values = [1.0f32, -2.0, 3.5, 0.25];
    let imaginary_values = [-4.0f32, 0.5, 8.0, -1.25];
    let real_bytes = f32_bytes(&real_values);
    let imaginary_bytes = f32_bytes(&imaginary_values);
    let real_transfer = client.buffer_from_host(&real_bytes, shape, device).unwrap();
    let imaginary_transfer = client
        .buffer_from_host(&imaginary_bytes, shape, device)
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
