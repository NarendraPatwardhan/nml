use nml_serve::{
    Backend, CompilationProfile, Event, GenerationOptions, SamplingOptions, ServerConfig,
    ServerLimits, ServerProfile, SubmissionTimings, Timings,
};
use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("serve: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match Command::parse(std::env::args().skip(1))? {
        Command::Generate(cli) => generate(cli),
        Command::Serve(cli) => serve(cli),
        Command::Help => {
            println!("{}", usage());
            Ok(())
        }
    }
}

fn serve(cli: ServeCli) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .json()
        .try_init()?;
    let mut profile = ServerProfile::a40(cli.prefill_capacity, cli.cache_capacity);
    profile.cache_budget_bytes = cli.cache_budget_bytes;
    profile.cache_safety_bytes = cli.cache_safety_bytes;
    profile.tensor_parallel = cli.tensor_parallel;
    let config = ServerConfig {
        bind: cli.bind,
        model: cli.model,
        backend: cli.backend.public(),
        profile,
        limits: ServerLimits::default(),
        shutdown_grace: cli.shutdown_grace,
    };
    let backend = cli.backend;
    // Start performs the platform handshake before Tokio creates worker
    // threads. Compilation and residency continue on the named engine thread
    // while the runtime below binds the HTTP socket.
    let server = nml_serve::Server::start(config, move || match backend {
        BackendChoice::Cpu => nml::Platform::cpu().map_err(boxed),
        BackendChoice::Cuda => {
            // SAFETY: Server::start executes this closure on the engine thread
            // while the main thread is synchronously waiting, before the Tokio
            // runtime exists. No other application thread can access process
            // environment variables during CUDA's process-global loader.
            unsafe { nml::Platform::cuda() }.map_err(boxed)
        }
    })?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server.run())
}

fn generate(cli: GenerateCli) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let platform = match cli.backend {
        BackendChoice::Cpu => nml::Platform::cpu()?,
        BackendChoice::Cuda => {
            // SAFETY: diagnostic generation is single-threaded at
            // initialization and no XLA/PJRT API has run before this call.
            unsafe { nml::Platform::cuda() }?
        }
    };
    let mut stdout = std::io::stdout().lock();
    let profile = CompilationProfile {
        max_prompt_tokens: cli.prefill_capacity,
        max_sequence_tokens: cli.cache_capacity,
    };
    let generator = nml_serve::Generator::load(&platform, &cli.model, &[profile])?;
    let report = generator.generate(
        GenerationOptions {
            prompt: cli.prompt,
            max_new_tokens: cli.max_new_tokens,
            cache_capacity: Some(cli.cache_capacity),
            sampling: cli.sampling,
        },
        |event| -> std::io::Result<()> {
            match event {
                Event::ContentDelta { text, .. } => {
                    stdout.write_all(text.as_bytes())?;
                    stdout.flush()?;
                }
                Event::ToolCall {
                    recipient,
                    arguments,
                } => {
                    write!(
                        stdout,
                        "\n{{\"recipient\":{recipient:?},\"arguments\":{arguments}}}"
                    )?;
                    stdout.flush()?;
                }
                Event::Done { .. } => {}
            }
            Ok(())
        },
    )?;
    eprintln!();
    eprintln!(
        "serve: model {}, {} prompt tokens, {} generated tokens, cache capacity {}",
        report.model,
        report.prompt_tokens,
        report.generated_tokens.len(),
        report.cache_capacity
    );
    eprintln!(
        "serve: {} physical parameter components, source {} bytes, resident {} bytes, prepared {} bytes, peak staging {} bytes",
        report.physical_parameter_components,
        report.parameter_source_bytes,
        report.parameter_resident_bytes,
        report.parameter_prepared_bytes,
        report.parameter_peak_staging_bytes,
    );
    eprintln!(
        "serve: cache storage {} bytes, metadata capacity {} bytes, metadata uploaded {} bytes",
        report.cache_storage_bytes,
        report.cache_metadata_bytes,
        report.cache_metadata_upload_bytes,
    );
    print_timings(&report.timings);
    eprintln!("serve: token IDs {:?}", report.generated_tokens);
    Ok(())
}

fn print_timings(timings: &Timings) {
    for (name, duration) in [
        ("tokenization", timings.tokenization),
        ("artifact validation", timings.artifact_validation),
        ("prefill compilation", timings.prefill_compilation),
        ("decode compilation", timings.decode_compilation),
        ("parameter upload", timings.parameter_upload),
        ("cache allocation", timings.cache_allocation),
        ("cache metadata upload", timings.cache_metadata_upload),
        ("prompt upload", timings.prompt_upload),
        ("prefill execution", timings.prefill_execution),
        ("prefill download", timings.prefill_download),
        (
            "decode state initialization",
            timings.decode_state_initialization,
        ),
        ("first decode execution", timings.first_decode_execution),
        ("steady decode execution", timings.steady_decode_execution),
        ("decode download", timings.decode_download),
    ] {
        eprintln!("serve: {name:>24}: {:9.3} ms", duration.as_secs_f64() * 1e3);
    }
    print_submission("prefill submission", timings.prefill_submission);
    print_submission("first decode submission", timings.first_decode_submission);
    print_submission("steady decode submission", timings.steady_decode_submission);
}

fn print_submission(name: &str, timings: SubmissionTimings) {
    eprintln!(
        "serve: {name:>24}: embedding={:.3} ms sliding={:.3} ms full={:.3} ms pairs={:.3} ms head={:.3} ms",
        timings.embedding.as_secs_f64() * 1e3,
        timings.sliding_layers.as_secs_f64() * 1e3,
        timings.full_layers.as_secs_f64() * 1e3,
        timings.layer_pairs.as_secs_f64() * 1e3,
        timings.head.as_secs_f64() * 1e3,
    );
}

struct GenerateCli {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    prefill_capacity: usize,
    cache_capacity: usize,
    sampling: SamplingOptions,
    backend: BackendChoice,
}

struct ServeCli {
    model: PathBuf,
    bind: SocketAddr,
    prefill_capacity: usize,
    cache_capacity: usize,
    cache_budget_bytes: usize,
    cache_safety_bytes: usize,
    tensor_parallel: usize,
    shutdown_grace: Duration,
    backend: BackendChoice,
}

enum Command {
    Generate(GenerateCli),
    Serve(ServeCli),
    Help,
}

#[derive(Clone, Copy)]
enum BackendChoice {
    Cpu,
    Cuda,
}

impl BackendChoice {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!("unknown backend {other:?}; use cpu or cuda")),
        }
    }

    const fn public(self) -> Backend {
        match self {
            Self::Cpu => Backend::Cpu,
            Self::Cuda => Backend::Cuda,
        }
    }
}

impl Command {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut arguments = arguments.collect::<Vec<_>>();
        if arguments.is_empty() || matches!(arguments[0].as_str(), "--help" | "-h") {
            return Ok(Self::Help);
        }
        match arguments[0].as_str() {
            "serve" => Ok(Self::Serve(ServeCli::parse(arguments.drain(1..))?)),
            "generate" => Ok(Self::Generate(GenerateCli::parse(arguments.drain(1..))?)),
            first if first.starts_with('-') => {
                // Compatibility for the established RunPod acceptance harness.
                Ok(Self::Generate(GenerateCli::parse(arguments.into_iter())?))
            }
            other => Err(format!("unknown command {other:?}\n{}", usage())),
        }
    }
}

impl GenerateCli {
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut prompt = None;
        let mut max_new_tokens = 32usize;
        let mut prefill_capacity = None;
        let mut cache_capacity = None;
        let mut sampling = SamplingOptions::default();
        let mut backend = BackendChoice::Cpu;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--model" => model = Some(PathBuf::from(value(&mut arguments, "--model")?)),
                "--prompt" => prompt = Some(value(&mut arguments, "--prompt")?),
                "--max-new-tokens" => {
                    max_new_tokens = parse_usize(
                        &value(&mut arguments, "--max-new-tokens")?,
                        "--max-new-tokens",
                    )?;
                }
                "--prefill-capacity" => {
                    prefill_capacity = Some(parse_usize(
                        &value(&mut arguments, "--prefill-capacity")?,
                        "--prefill-capacity",
                    )?);
                }
                "--cache-capacity" => {
                    cache_capacity = Some(parse_usize(
                        &value(&mut arguments, "--cache-capacity")?,
                        "--cache-capacity",
                    )?);
                }
                "--seed" => {
                    let seed = parse_u64(&value(&mut arguments, "--seed")?, "--seed")?;
                    sampling.seed = [seed, seed ^ 0x9e37_79b9_7f4a_7c15];
                }
                "--temperature" => {
                    sampling.temperature =
                        parse_f32(&value(&mut arguments, "--temperature")?, "--temperature")?;
                }
                "--top-k" => {
                    sampling.top_k =
                        parse_usize(&value(&mut arguments, "--top-k")?, "--top-k")?;
                }
                "--top-p" => {
                    sampling.top_p = parse_f32(&value(&mut arguments, "--top-p")?, "--top-p")?;
                }
                "--min-p" => {
                    sampling.min_p = parse_f32(&value(&mut arguments, "--min-p")?, "--min-p")?;
                }
                "--backend" => {
                    backend = BackendChoice::parse(&value(&mut arguments, "--backend")?)?;
                }
                "--help" | "-h" => return Err(generate_usage().to_owned()),
                other => return Err(format!("unknown generation argument {other:?}\n{}", usage())),
            }
        }
        Ok(Self {
            model: model.ok_or_else(|| format!("--model is required\n{}", generate_usage()))?,
            prompt: prompt.ok_or_else(|| format!("--prompt is required\n{}", generate_usage()))?,
            max_new_tokens,
            prefill_capacity: prefill_capacity
                .ok_or_else(|| format!("--prefill-capacity is required\n{}", generate_usage()))?,
            cache_capacity: cache_capacity
                .ok_or_else(|| format!("--cache-capacity is required\n{}", generate_usage()))?,
            sampling,
            backend,
        })
    }
}

impl ServeCli {
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut model = None;
        let mut bind = "0.0.0.0:8000".parse().expect("default bind address is valid");
        let mut prefill_capacity = 4_096usize;
        let mut cache_capacity = 8_192usize;
        let mut cache_budget_bytes = 8 * 1024 * 1024 * 1024usize;
        let mut cache_safety_bytes = 512 * 1024 * 1024usize;
        let mut tensor_parallel = 1usize;
        let mut shutdown_grace = Duration::from_secs(30);
        let mut backend = BackendChoice::Cuda;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--model" => model = Some(PathBuf::from(value(&mut arguments, "--model")?)),
                "--bind" => {
                    let raw = value(&mut arguments, "--bind")?;
                    bind = raw
                        .parse()
                        .map_err(|_| format!("--bind requires an IP socket address, received {raw:?}"))?;
                }
                "--backend" => {
                    backend = BackendChoice::parse(&value(&mut arguments, "--backend")?)?;
                }
                "--prefill-capacity" => {
                    prefill_capacity = parse_usize(
                        &value(&mut arguments, "--prefill-capacity")?,
                        "--prefill-capacity",
                    )?;
                }
                "--cache-capacity" => {
                    cache_capacity = parse_usize(
                        &value(&mut arguments, "--cache-capacity")?,
                        "--cache-capacity",
                    )?;
                }
                "--cache-budget-bytes" => {
                    cache_budget_bytes = parse_usize(
                        &value(&mut arguments, "--cache-budget-bytes")?,
                        "--cache-budget-bytes",
                    )?;
                }
                "--cache-safety-bytes" => {
                    cache_safety_bytes = parse_usize(
                        &value(&mut arguments, "--cache-safety-bytes")?,
                        "--cache-safety-bytes",
                    )?;
                }
                "--tensor-parallel" => {
                    tensor_parallel = parse_usize(
                        &value(&mut arguments, "--tensor-parallel")?,
                        "--tensor-parallel",
                    )?;
                }
                "--shutdown-grace-seconds" => {
                    shutdown_grace = Duration::from_secs(parse_u64(
                        &value(&mut arguments, "--shutdown-grace-seconds")?,
                        "--shutdown-grace-seconds",
                    )?);
                }
                "--help" | "-h" => return Err(serve_usage().to_owned()),
                other => return Err(format!("unknown serve argument {other:?}\n{}", serve_usage())),
            }
        }
        Ok(Self {
            model: model.ok_or_else(|| format!("--model is required\n{}", serve_usage()))?,
            bind,
            prefill_capacity,
            cache_capacity,
            cache_budget_bytes,
            cache_safety_bytes,
            tensor_parallel,
            shutdown_grace,
            backend,
        })
    }
}

fn value(arguments: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_usize(value: &str, option: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{option} requires a nonnegative integer, received {value:?}"))
}

fn parse_u64(value: &str, option: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{option} requires an unsigned integer, received {value:?}"))
}

fn parse_f32(value: &str, option: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("{option} requires a floating-point value, received {value:?}"))
}

fn boxed(error: impl std::error::Error + Send + Sync + 'static) -> nml_serve::Error {
    Box::new(error)
}

fn usage() -> &'static str {
    "Usage:\n  nml-serve serve --model DIR [server options]\n  nml-serve generate --model DIR --prompt TEXT --prefill-capacity N --cache-capacity N [generation options]\n\nLegacy generation flags without the `generate` command remain accepted by the device harness."
}

fn serve_usage() -> &'static str {
    "Usage: nml-serve serve --model DIR [--bind IP:PORT] [--backend cpu|cuda] [--prefill-capacity N] [--cache-capacity N] [--cache-budget-bytes N] [--cache-safety-bytes N] [--tensor-parallel 1|2|4] [--shutdown-grace-seconds N]"
}

fn generate_usage() -> &'static str {
    "Usage: nml-serve generate --model DIR --prompt TEXT --prefill-capacity N --cache-capacity N [--max-new-tokens N] [--backend cpu|cuda] [--seed N] [--temperature F] [--top-k N] [--top-p F] [--min-p F]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_generation_and_named_server_commands_remain_distinct() {
        let legacy = Command::parse(
            [
                "--model", "m", "--prompt", "p", "--prefill-capacity", "16",
                "--cache-capacity", "32",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert!(matches!(legacy, Command::Generate(_)));
        let server = Command::parse(
            ["serve", "--model", "m"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert!(matches!(server, Command::Serve(_)));
    }
}
