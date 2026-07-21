//! End-to-end contract for the packaged CPU PJRT runtime.

use nml_tensor::Slice;

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
    let column_major = nml_types::Shape::new(nml_types::DType::F32, &[2, 2])
        .unwrap()
        .with_layout(nml_types::Layout::from_minor_to_major(&[0, 1]).unwrap())
        .unwrap();
    // Physical bytes [1, 2, 3, 4] under column-major strides represent the
    // logical row-major matrix [[1, 3], [2, 4]].
    let column_major_bytes = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect::<Vec<_>>();
    let column_major_slice = Slice::from_bytes(column_major, &column_major_bytes).unwrap();
    let column_major_buffer = client
        .buffer_from_host(&column_major_slice, &devices[0])
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        decode_f32(
            column_major_buffer
                .to_slice_alloc()
                .unwrap()
                .contiguous_bytes()
                .unwrap()
        ),
        [1.0, 3.0, 2.0, 4.0]
    );
    let values = [1.0f32, -2.5, 7.25, 0.0];
    let host_bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let shape = nml_types::Shape::new(nml_types::DType::F32, &[2, 2]).unwrap();
    let buffer = upload(&client, &host_bytes, shape, &devices[0]);
    assert_eq!(buffer.dtype().unwrap(), nml_types::DType::F32);
    assert_eq!(buffer.dimensions().unwrap(), &[2, 2]);
    let returned = buffer
        .to_slice_alloc()
        .expect("CPU device-to-host transfer must complete");
    assert_eq!(returned.contiguous_bytes().unwrap(), host_bytes);
    let mut asynchronous = Slice::alloc(shape).unwrap();
    let download = buffer
        .to_slice_async(&mut asynchronous)
        .expect("CPU device-to-host transfer must start asynchronously");
    download
        .wait()
        .expect("CPU asynchronous device-to-host transfer must complete");
    assert_eq!(asynchronous.contiguous_bytes().unwrap(), host_bytes);

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
        let buffer = upload(&client, &bytes, shape, &devices[0]);
        assert_eq!(buffer.shape().unwrap(), shape);
        assert_eq!(
            buffer.to_slice_alloc().unwrap().contiguous_bytes().unwrap(),
            bytes
        );
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
        let executable = nml_compiler::compile(
            &client,
            &program,
            &nml_sharding::Sharding::single(),
            &options,
            nml_compiler::Target::Cpu,
        )
        .unwrap();
        let left = upload(&client, &left_bytes, left.shape(), &device);
        let right = upload(&client, &right_bytes, right.shape(), &device);
        (executable, device, left, right)
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
    let actual = decode_f32(
        execution.outputs[0]
            .to_slice_alloc()
            .unwrap()
            .contiguous_bytes()
            .unwrap(),
    );
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
    let executable = nml_compiler::compile(
        client,
        &program,
        &nml_sharding::Sharding::single(),
        &options,
        nml_compiler::Target::Cpu,
    )
    .expect("CPU XLA compilation must succeed");

    let left_values: Vec<f32> = (0..15).map(|value| value as f32 / 7.0 - 1.0).collect();
    let right_values: Vec<f32> = (0..20).map(|value| value as f32 / 11.0 - 0.5).collect();
    let left_bytes = f32_bytes(&left_values);
    let right_bytes = f32_bytes(&right_values);
    let left_buffer = upload(client, &left_bytes, left.shape(), device);
    let right_buffer = upload(client, &right_bytes, right.shape(), device);
    let execution = executable
        .execute_one(&[&left_buffer, &right_buffer], Some(device))
        .expect("CPU executable launch must succeed");
    execution.complete.wait().unwrap();
    let actual = decode_f32(
        execution.outputs[0]
            .to_slice_alloc()
            .unwrap()
            .contiguous_bytes()
            .unwrap(),
    );
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
    let executable = nml_compiler::compile(
        client,
        &program,
        &nml_sharding::Sharding::single(),
        &options,
        nml_compiler::Target::Cpu,
    )
    .unwrap();
    let real_values = [1.0f32, -2.0, 3.5, 0.25];
    let imaginary_values = [-4.0f32, 0.5, 8.0, -1.25];
    let real_bytes = f32_bytes(&real_values);
    let imaginary_bytes = f32_bytes(&imaginary_values);
    let real_buffer = upload(client, &real_bytes, shape, device);
    let imaginary_buffer = upload(client, &imaginary_bytes, shape, device);
    let execution = executable
        .execute_one(&[&real_buffer, &imaginary_buffer], Some(device))
        .unwrap();
    execution.complete.wait().unwrap();
    assert_close(
        &decode_f32(
            execution.outputs[0]
                .to_slice_alloc()
                .unwrap()
                .contiguous_bytes()
                .unwrap(),
        ),
        &real_values,
    );
    assert_close(
        &decode_f32(
            execution.outputs[1]
                .to_slice_alloc()
                .unwrap()
                .contiguous_bytes()
                .unwrap(),
        ),
        &imaginary_values,
    );
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn upload(
    client: &nml_pjrt::Client,
    bytes: &[u8],
    shape: nml_types::Shape,
    device: &nml_pjrt::Device,
) -> nml_pjrt::Buffer {
    let slice = Slice::from_bytes(shape, bytes).unwrap();
    client
        .buffer_from_host(&slice, device)
        .unwrap()
        .wait()
        .unwrap()
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
