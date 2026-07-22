//! Durable phase-separated performance contract for the product execution path.
//!
//! The generous ceilings catch pathological regressions rather than ranking
//! machines. Exact tuning evidence is recorded per hardware; this contract
//! keeps compilation, transfer, first execution, and steady execution from
//! collapsing into one misleading wall-clock number.

use nml_ir::ProgramBuilder;
use nml_parameter::{ComponentRole, Parameter};
use nml_types::{BFloat16, DType, Shape};
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
    let weight = parameter("weight", weight_shape);
    let projected = builder.linear(input, &weight, None).unwrap();
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
    let weight_buffer = platform
        .upload(&weight_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let upload = upload_started.elapsed();

    let mut arguments = executable.args();
    arguments.set("input", input).unwrap();
    let loaded_weight = nml::LoadedParameter::new(weight, vec![weight_buffer]).unwrap();
    arguments
        .set_parameter(&loaded_weight)
        .unwrap()
        .bake()
        .unwrap();
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
    if env!("NML_PERFORMANCE_BACKEND") == "cuda" {
        measure_nvfp4_workloads(&platform);
    }
}

fn measure_spatial_workloads(platform: &nml::Platform) {
    let audio_shape = Shape::new(DType::F32, &[4, 16, 2048]).unwrap();
    let audio_kernel_shape = Shape::new(DType::F32, &[32, 16, 7]).unwrap();
    let image_shape = Shape::new(DType::F32, &[2, 16, 64, 64]).unwrap();
    let image_kernel_shape = Shape::new(DType::F32, &[32, 16, 3, 3]).unwrap();
    let mut builder = ProgramBuilder::new();
    let audio = builder.input("audio", audio_shape);
    let audio_parameter = parameter("audio_kernel", audio_kernel_shape);
    let audio_kernel = builder.parameter_value(&audio_parameter).unwrap();
    let image = builder.input("image", image_shape);
    let image_parameter = parameter("image_kernel", image_kernel_shape);
    let image_kernel = builder.parameter_value(&image_parameter).unwrap();
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
    for (name, shape) in [("audio", audio_shape), ("image", image_shape)] {
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
    set_generated_parameter(platform, &mut arguments, &audio_parameter, 11, 97, 128.0);
    set_generated_parameter(platform, &mut arguments, &image_parameter, 11, 97, 128.0);
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
    let gate_up = parameter("gate_up", gate_up_shape);
    let down = parameter("down", down_shape);
    let output = builder
        .moe_swiglu(hidden, router, &gate_up, &down, 2)
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();
    let upload_started = Instant::now();
    let mut arguments = executable.args();
    for (name, shape) in [("hidden", hidden_shape), ("router", router_shape)] {
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
    set_generated_parameter(platform, &mut arguments, &gate_up, 17, 113, 256.0);
    set_generated_parameter(platform, &mut arguments, &down, 17, 113, 256.0);
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

// These are production-scale dimensions, not toy multiples chosen to flatter
// a kernel. The ordinary CPU performance target deliberately does not run this
// CUDA acceptance family: its separately tracked CPU work needs
// architecture-specific ceilings and must not inherit GPU-shaped claims.
const LARGE_HIDDEN: usize = 2_880;
const LARGE_INTERMEDIATE: usize = 2_880;
const ROUTED_EXPERTS: usize = 32;
const ROUTES_PER_TOKEN: usize = 4;
const LARGE_VOCABULARY: usize = 201_088;
const NVFP4_SCALE: f32 = 1.0 / 1_024.0;

fn measure_nvfp4_workloads(platform: &nml::Platform) {
    measure_nvfp4_embedding(platform);
    measure_nvfp4_linear(platform, 1, "nvfp4_linear_decode_m1_k2880_n2880");
    measure_nvfp4_linear(platform, 128, "nvfp4_linear_prefill_m128_k2880_n2880");
    measure_nvfp4_grouped_moe(
        platform,
        1,
        "nvfp4_grouped_moe_decode_m1_experts32_top4_hidden2880_intermediate2880",
    );
    measure_nvfp4_grouped_moe(
        platform,
        16,
        "nvfp4_grouped_moe_tokens16_experts32_top4_hidden2880_intermediate2880",
    );
}

fn measure_nvfp4_embedding(platform: &nml::Platform) {
    const TOKENS: usize = 128;
    let index_shape = Shape::new(DType::I32, &[TOKENS as i64]).unwrap();
    let embedding = Parameter::nvfp4(
        "embedding",
        "model.embed_tokens.weight",
        Shape::new(
            DType::Bf16,
            &[LARGE_VOCABULARY as i64, LARGE_HIDDEN as i64],
        )
        .unwrap(),
    )
    .unwrap();
    let mut builder = ProgramBuilder::new();
    let indices = builder.input("indices", index_shape);
    let output = builder.token_embedding(&embedding, indices).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();
    let indices = (0..TOKENS)
        .map(|index| ((index * 1_543) % LARGE_VOCABULARY) as i32)
        .collect::<Vec<_>>();
    let index_host = nml::Slice::from_typed(index_shape, &indices).unwrap();
    let embedding_host = patterned_nvfp4(embedding);
    let upload_started = Instant::now();
    let index_buffer = platform
        .upload(&index_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let loaded_embedding = upload_parameter(platform, &embedding_host);
    let mut arguments = executable.args();
    arguments.set("indices", index_buffer).unwrap();
    arguments
        .set_parameter(&loaded_embedding)
        .unwrap()
        .bake()
        .unwrap();
    let upload = upload_started.elapsed();
    drop(embedding_host);

    let first_started = Instant::now();
    let first = arguments.call().unwrap();
    let first_execution = first_started.elapsed();
    let steady_started = Instant::now();
    let mut final_results = None;
    for _ in 0..STEADY_RUNS {
        final_results = Some(arguments.call().unwrap());
    }
    let steady_average = steady_started.elapsed() / STEADY_RUNS;
    let download_started = Instant::now();
    assert_bf16_finite(final_results.as_ref().unwrap(), "output");
    let download = download_started.elapsed();
    report_phases(
        platform,
        "nvfp4_embedding_vocab201088_width2880_tokens128",
        compile,
        upload,
        first_execution,
        steady_average,
        download,
    );
    drop(first);
}

fn measure_nvfp4_linear(platform: &nml::Platform, rows: usize, workload: &str) {
    let input_shape = Shape::new(DType::Bf16, &[rows as i64, LARGE_HIDDEN as i64]).unwrap();
    let weight = Parameter::nvfp4(
        "weight",
        "model.projection.weight",
        Shape::new(DType::Bf16, &[LARGE_HIDDEN as i64, LARGE_HIDDEN as i64]).unwrap(),
    )
    .unwrap();
    let mut builder = ProgramBuilder::new();
    let input = builder.input("input", input_shape);
    let output = builder.linear(input, &weight, None).unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();
    let input_values = patterned_bf16(rows * LARGE_HIDDEN);
    let input_host = nml::Slice::from_typed(input_shape, &input_values).unwrap();
    let weight_host = patterned_nvfp4(weight);
    let upload_started = Instant::now();
    let input_buffer = platform
        .upload(&input_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let loaded_weight = upload_parameter(platform, &weight_host);
    let mut arguments = executable.args();
    arguments.set("input", input_buffer).unwrap();
    arguments
        .set_parameter(&loaded_weight)
        .unwrap()
        .bake()
        .unwrap();
    let upload = upload_started.elapsed();
    drop(weight_host);

    let first_started = Instant::now();
    let first = arguments.call().unwrap();
    let first_execution = first_started.elapsed();
    let steady_started = Instant::now();
    let mut final_results = None;
    for _ in 0..STEADY_RUNS {
        final_results = Some(arguments.call().unwrap());
    }
    let steady_average = steady_started.elapsed() / STEADY_RUNS;
    let download_started = Instant::now();
    assert_bf16_finite(final_results.as_ref().unwrap(), "output");
    let download = download_started.elapsed();
    report_phases(
        platform,
        workload,
        compile,
        upload,
        first_execution,
        steady_average,
        download,
    );
    drop(first);
}

fn measure_nvfp4_grouped_moe(
    platform: &nml::Platform,
    tokens: usize,
    workload: &str,
) {
    let hidden_shape = Shape::new(DType::Bf16, &[tokens as i64, LARGE_HIDDEN as i64]).unwrap();
    let router_shape = Shape::new(DType::F32, &[tokens as i64, ROUTED_EXPERTS as i64]).unwrap();
    let gate = Parameter::nvfp4(
        "gate_up",
        "model.experts.gate_up_proj",
        Shape::new(
            DType::Bf16,
            &[
                ROUTED_EXPERTS as i64,
                LARGE_HIDDEN as i64,
                (2 * LARGE_INTERMEDIATE) as i64,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let down = Parameter::nvfp4(
        "down",
        "model.experts.down_proj",
        Shape::new(
            DType::Bf16,
            &[
                ROUTED_EXPERTS as i64,
                LARGE_INTERMEDIATE as i64,
                LARGE_HIDDEN as i64,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let gate_bias = Parameter::dense(
        "gate_bias",
        "model.experts.gate_up_proj_bias",
        Shape::new(
            DType::Bf16,
            &[ROUTED_EXPERTS as i64, (2 * LARGE_INTERMEDIATE) as i64],
        )
        .unwrap(),
    )
    .unwrap();
    let down_bias = Parameter::dense(
        "down_bias",
        "model.experts.down_proj_bias",
        Shape::new(
            DType::Bf16,
            &[ROUTED_EXPERTS as i64, LARGE_HIDDEN as i64],
        )
        .unwrap(),
    )
    .unwrap();
    let mut builder = ProgramBuilder::new();
    let hidden = builder.input("hidden", hidden_shape);
    let router = builder.input("router", router_shape);
    let output = builder
        .routed_clamped_swiglu(
            hidden,
            router,
            &gate,
            &gate_bias,
            &down,
            &down_bias,
            ROUTES_PER_TOKEN,
        )
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();

    let compile_started = Instant::now();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let compile = compile_started.elapsed();
    let hidden = patterned_bf16(tokens * LARGE_HIDDEN);
    let hidden_host = nml::Slice::from_typed(hidden_shape, &hidden).unwrap();
    let router = (0..tokens * ROUTED_EXPERTS)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) / 16.0)
        .collect::<Vec<_>>();
    let router_host = nml::Slice::from_typed(router_shape, &router).unwrap();
    let gate_host = patterned_nvfp4(gate);
    let gate_bias_host = zero_bf16_parameter(gate_bias);
    let down_host = patterned_nvfp4(down);
    let down_bias_host = zero_bf16_parameter(down_bias);
    let upload_started = Instant::now();
    let hidden_buffer = platform
        .upload(&hidden_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let router_buffer = platform
        .upload(&router_host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let loaded_gate = upload_parameter(platform, &gate_host);
    let loaded_gate_bias = upload_parameter(platform, &gate_bias_host);
    let loaded_down = upload_parameter(platform, &down_host);
    let loaded_down_bias = upload_parameter(platform, &down_bias_host);
    let mut arguments = executable.args();
    arguments.set("hidden", hidden_buffer).unwrap();
    arguments.set("router", router_buffer).unwrap();
    for parameter in [loaded_gate, loaded_gate_bias, loaded_down, loaded_down_bias] {
        arguments.set_parameter(&parameter).unwrap();
    }
    arguments.bake().unwrap();
    let upload = upload_started.elapsed();
    drop((gate_host, gate_bias_host, down_host, down_bias_host));

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
    assert_bf16_finite(final_results.as_ref().unwrap(), "output");
    let download = download_started.elapsed();
    report_phases(
        platform,
        workload,
        compile,
        upload,
        first_execution,
        steady_average,
        download,
    );
    drop(first);
}

struct HostParameter {
    parameter: Parameter,
    components: Vec<HostComponent>,
}

struct HostComponent {
    shape: Shape,
    bytes: Vec<u8>,
}

fn patterned_nvfp4(parameter: Parameter) -> HostParameter {
    let scale = nml_parameter::nvfp4::encode_e4m3fn_scale(1.0).unwrap();
    let global = NVFP4_SCALE.to_ne_bytes();
    let components = parameter
        .components()
        .iter()
        .map(|component| {
            let bytes = match component.role() {
                // Both nibbles encode +0.5. A small global factor keeps the
                // full production-scale contractions finite without changing storage
                // density or the compact-kernel path under measurement.
                ComponentRole::Payload => {
                    vec![0x11; component.storage().shape().element_count().unwrap()]
                }
                ComponentRole::BlockScales => {
                    vec![scale; component.storage().shape().element_count().unwrap()]
                }
                ComponentRole::GlobalScale => global.to_vec(),
                ComponentRole::Values => unreachable!(),
            };
            HostComponent {
                shape: component.storage().shape(),
                bytes,
            }
        })
        .collect();
    HostParameter {
        parameter,
        components,
    }
}

fn zero_bf16_parameter(parameter: Parameter) -> HostParameter {
    let bytes = vec![0; parameter.shape().element_count().unwrap() * size_of::<BFloat16>()];
    HostParameter {
        components: vec![HostComponent {
            shape: parameter.shape(),
            bytes,
        }],
        parameter,
    }
}

fn upload_parameter(platform: &nml::Platform, parameter: &HostParameter) -> nml::LoadedParameter {
    let buffers = parameter
        .components
        .iter()
        .map(|component| {
            let host = nml::Slice::from_bytes(component.shape, &component.bytes).unwrap();
            platform
                .upload(&host, nml::Sharding::single(), nml::Memory::Default)
                .unwrap()
        })
        .collect();
    nml::LoadedParameter::new(parameter.parameter.clone(), buffers).unwrap()
}

fn patterned_bf16(count: usize) -> Vec<BFloat16> {
    (0..count)
        .map(|index| BFloat16::from_f32(((index * 13 % 127) as f32 - 63.0) / 64.0))
        .collect()
}

fn assert_bf16_finite(results: &nml::exe::Results, name: &str) {
    let output = results.get(name).unwrap().to_slice().unwrap();
    let bytes = output.contiguous_bytes().unwrap();
    assert_eq!(bytes.len() % size_of::<BFloat16>(), 0);
    assert!(bytes.chunks_exact(2).all(|item| {
        BFloat16::from_bits(u16::from_ne_bytes(item.try_into().unwrap()))
            .to_f32()
            .is_finite()
    }));
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

fn parameter(name: &str, shape: Shape) -> nml::Parameter {
    nml::Parameter::dense(name, name, shape).unwrap()
}

fn set_generated_parameter(
    platform: &nml::Platform,
    arguments: &mut nml::exe::Arguments,
    parameter: &nml::Parameter,
    multiplier: usize,
    modulus: usize,
    divisor: f32,
) {
    let shape = parameter.shape();
    let center = (modulus / 2) as f32;
    let values = (0..shape.element_count().unwrap())
        .map(|index| ((index * multiplier % modulus) as f32 - center) / divisor)
        .collect::<Vec<_>>();
    let host = nml::Slice::from_typed(shape, &values).unwrap();
    let buffer = platform
        .upload(&host, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    let loaded = nml::LoadedParameter::new(parameter.clone(), vec![buffer]).unwrap();
    arguments.set_parameter(&loaded).unwrap();
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
