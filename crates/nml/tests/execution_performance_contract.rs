//! Durable phase-separated performance contract for the product execution path.
//!
//! The generous ceilings catch pathological regressions rather than ranking
//! machines. Exact tuning evidence is recorded per hardware; this contract
//! keeps compilation, transfer, first execution, and steady execution from
//! collapsing into one misleading wall-clock number.

use nml_ir::ProgramBuilder;
use nml_types::{DType, Shape};
use std::time::{Duration, Instant};

const BATCH: usize = 64;
const WIDTH: usize = 512;
const STEADY_RUNS: u32 = 5;

#[test]
fn representative_execution_has_phase_separated_bounds() {
    let platform = platform();
    let activation_shape = Shape::new(DType::F32, &[BATCH as i64, WIDTH as i64]).unwrap();
    let weight_shape = Shape::new(DType::F32, &[WIDTH as i64, WIDTH as i64]).unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", activation_shape);
    let weight = builder.parameter("weight", weight_shape);
    let projected = builder.linear(input, weight, None).unwrap();
    let activated = builder.silu(projected).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), activated)])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();

    let input_values = (0..BATCH * WIDTH)
        .map(|index| ((index * 13 % 127) as f32 - 63.0) / 64.0)
        .collect::<Vec<_>>();
    let weight_values = (0..WIDTH * WIDTH)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 512.0)
        .collect::<Vec<_>>();
    let input_host = nml::Slice::from_typed(activation_shape, &input_values).unwrap();
    let weight_host = nml::Slice::from_typed(weight_shape, &weight_values).unwrap();
    let upload_started = Instant::now();
    let input = platform
        .upload(&input_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let weight = platform
        .upload(&weight_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let upload = upload_started.elapsed();

    let mut arguments = executable.args();
    arguments.set("input", input).unwrap();
    arguments.set("weight", weight).unwrap().bake().unwrap();
    let first_started = Instant::now();
    let first = arguments.call().unwrap();
    let first_execution = first_started.elapsed();

    let steady_started = Instant::now();
    let mut final_results = None;
    for _ in 0..STEADY_RUNS {
        final_results = Some(arguments.call().unwrap());
    }
    let steady_total = steady_started.elapsed();
    let steady_average = steady_total / STEADY_RUNS;

    let download_started = Instant::now();
    let output = final_results
        .as_ref()
        .unwrap()
        .get("output")
        .unwrap()
        .to_slice()
        .unwrap();
    let download = download_started.elapsed();
    let output_bytes = output.contiguous_bytes().unwrap();
    assert_eq!(output_bytes.len(), BATCH * WIDTH * size_of::<f32>());
    assert!(output_bytes.chunks_exact(4).all(|bytes| {
        f32::from_le_bytes(bytes.try_into().expect("four-byte F32 chunk")).is_finite()
    }));

    report_phases(
        &platform,
        "language",
        compile,
        upload,
        first_execution,
        steady_average,
        download,
    );

    // Keep the first result live until after timing so its buffer lifecycle is
    // included in the same ownership contract as repeated execution.
    drop(first);

    measure_spatial_workloads(&platform);
    measure_moe_workload(&platform);
}

fn measure_spatial_workloads(platform: &nml::Platform) {
    let audio_shape = Shape::new(DType::F32, &[4, 16, 2048]).unwrap();
    let audio_kernel_shape = Shape::new(DType::F32, &[32, 16, 7]).unwrap();
    let image_shape = Shape::new(DType::F32, &[2, 16, 64, 64]).unwrap();
    let image_kernel_shape = Shape::new(DType::F32, &[32, 16, 3, 3]).unwrap();
    let mut builder = ProgramBuilder::new();
    let audio = builder.input("audio", audio_shape);
    let audio_kernel = builder.parameter("audio_kernel", audio_kernel_shape);
    let image = builder.input("image", image_shape);
    let image_kernel = builder.parameter("image_kernel", image_kernel_shape);
    let audio = builder
        .conv1d(audio, audio_kernel, 1, [3, 3], 1, 1, 1)
        .unwrap();
    let audio = builder.max_pool1d(audio, 2, 2, 2, [0, 0]).unwrap();
    let image = builder
        .conv2d(
            image,
            image_kernel,
            [1, 1],
            [[1, 1], [1, 1]],
            [1, 1],
            [1, 1],
            1,
        )
        .unwrap();
    let image = builder
        .max_pool2d(image, [2, 3], [2, 2], [2, 2], [[0, 0], [0, 0]])
        .unwrap();
    let program = builder
        .finish_named(&[
            ("audio_output".to_owned(), audio),
            ("image_output".to_owned(), image),
        ])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();
    let upload_started = Instant::now();
    let mut arguments = executable.args();
    for (name, shape) in [
        ("audio", audio_shape),
        ("audio_kernel", audio_kernel_shape),
        ("image", image_shape),
        ("image_kernel", image_kernel_shape),
    ] {
        let count = shape.element_count().unwrap();
        let values = (0..count)
            .map(|index| ((index * 11 % 97) as f32 - 48.0) / 128.0)
            .collect::<Vec<_>>();
        let host = nml::Slice::from_typed(shape, &values).unwrap();
        let buffer = platform
            .upload(&host, nml::Sharding::single(), nml::Memory::Default)
            .unwrap();
        arguments.set(name, buffer).unwrap();
    }
    arguments.bake().unwrap();
    let upload = upload_started.elapsed();
    let first_started = Instant::now();
    let first = arguments.call().unwrap();
    let first_execution = first_started.elapsed();
    let steady_started = Instant::now();
    let mut final_results = None;
    for _ in 0..3 {
        final_results = Some(arguments.call().unwrap());
    }
    let steady_average = steady_started.elapsed() / 3;
    let download_started = Instant::now();
    for output in ["audio_output", "image_output"] {
        assert_finite(final_results.as_ref().unwrap(), output);
    }
    let download = download_started.elapsed();
    report_phases(
        platform,
        "spatial",
        compile,
        upload,
        first_execution,
        steady_average,
        download,
    );
    drop(first);
}

fn measure_moe_workload(platform: &nml::Platform) {
    const TOKENS: usize = 64;
    const HIDDEN: usize = 128;
    const EXPERTS: usize = 8;
    const INTERMEDIATE: usize = 64;
    let hidden_shape = Shape::new(DType::F32, &[TOKENS as i64, HIDDEN as i64]).unwrap();
    let router_shape = Shape::new(DType::F32, &[TOKENS as i64, EXPERTS as i64]).unwrap();
    let gate_up_shape = Shape::new(
        DType::F32,
        &[EXPERTS as i64, (2 * INTERMEDIATE) as i64, HIDDEN as i64],
    )
    .unwrap();
    let down_shape = Shape::new(
        DType::F32,
        &[EXPERTS as i64, HIDDEN as i64, INTERMEDIATE as i64],
    )
    .unwrap();
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input("hidden", hidden_shape);
    let router = builder.input("router", router_shape);
    let gate_up = builder.parameter("gate_up", gate_up_shape);
    let down = builder.parameter("down", down_shape);
    let output = builder
        .moe_swiglu(hidden, router, gate_up, down, 2)
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();
    let upload_started = Instant::now();
    let mut arguments = executable.args();
    for (name, shape) in [
        ("hidden", hidden_shape),
        ("router", router_shape),
        ("gate_up", gate_up_shape),
        ("down", down_shape),
    ] {
        let count = shape.element_count().unwrap();
        let values = (0..count)
            .map(|index| ((index * 17 % 113) as f32 - 56.0) / 256.0)
            .collect::<Vec<_>>();
        let host = nml::Slice::from_typed(shape, &values).unwrap();
        let buffer = platform
            .upload(&host, nml::Sharding::single(), nml::Memory::Default)
            .unwrap();
        arguments.set(name, buffer).unwrap();
    }
    arguments.bake().unwrap();
    let upload = upload_started.elapsed();
    let first_started = Instant::now();
    let first = arguments.call().unwrap();
    let first_execution = first_started.elapsed();
    let steady_started = Instant::now();
    let mut final_results = None;
    for _ in 0..3 {
        final_results = Some(arguments.call().unwrap());
    }
    let steady_average = steady_started.elapsed() / 3;
    let download_started = Instant::now();
    assert_finite(final_results.as_ref().unwrap(), "output");
    let download = download_started.elapsed();
    report_phases(
        platform,
        "moe",
        compile,
        upload,
        first_execution,
        steady_average,
        download,
    );
    drop(first);
}

fn assert_finite(results: &nml::exe::Results, name: &str) {
    let output = results.get(name).unwrap().to_slice().unwrap();
    let bytes = output.contiguous_bytes().unwrap();
    assert!(
        bytes
            .chunks_exact(4)
            .all(|item| f32::from_le_bytes(item.try_into().unwrap()).is_finite()),
        "{name} contains a non-finite value"
    );
}

#[allow(clippy::too_many_arguments)]
fn report_phases(
    platform: &nml::Platform,
    workload: &str,
    compile: Duration,
    upload: Duration,
    first_execution: Duration,
    steady_average: Duration,
    download: Duration,
) {
    eprintln!(
        "nml-performance workload={} backend={} compile_ms={:.3} upload_ms={:.3} first_ms={:.3} steady_average_ms={:.3} download_ms={:.3}",
        workload,
        platform.name(),
        compile.as_secs_f64() * 1_000.0,
        upload.as_secs_f64() * 1_000.0,
        first_execution.as_secs_f64() * 1_000.0,
        steady_average.as_secs_f64() * 1_000.0,
        download.as_secs_f64() * 1_000.0,
    );
    assert_phase("compile", compile, Duration::from_secs(120));
    assert_phase("upload", upload, Duration::from_secs(30));
    assert_phase("first execution", first_execution, Duration::from_secs(30));
    assert_phase("steady execution", steady_average, Duration::from_secs(10));
    assert_phase("download", download, Duration::from_secs(30));
}

fn assert_phase(name: &str, actual: Duration, ceiling: Duration) {
    assert!(
        actual <= ceiling,
        "{name} took {actual:?}, exceeding regression ceiling {ceiling:?}"
    );
}

fn platform() -> nml::Platform {
    match env!("NML_PERFORMANCE_BACKEND") {
        "cpu" => nml::Platform::cpu_with_devices(1).unwrap(),
        "cuda" => {
            let runfiles = std::env::var("RUNFILES_DIR").unwrap();
            let relative = std::env::var("NML_CUDA_RUNTIME_RLOCATION").unwrap();
            // SAFETY: Bazel test processes have not created application
            // threads before platform initialization.
            unsafe {
                std::env::set_var(
                    "NML_CUDA_RUNTIME",
                    std::path::Path::new(&runfiles).join(relative),
                );
                nml::Platform::cuda().unwrap()
            }
        }
        backend => panic!("unknown performance backend {backend}"),
    }
}
