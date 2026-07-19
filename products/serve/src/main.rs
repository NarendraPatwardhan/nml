use nml_serve::{
    CompilationProfile, Event, GenerationOptions, SamplingOptions, SubmissionTimings, Timings,
};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

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
    let Command::Run(cli) = Cli::parse(std::env::args().skip(1))? else {
        println!("{}", usage());
        return Ok(());
    };
    let platform = match cli.backend {
        Backend::Cpu => nml::Platform::cpu()?,
        Backend::Cuda => {
            // SAFETY: this product is single-threaded at initialization and no
            // XLA or PJRT API has been called before selecting the CUDA plugin.
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
        "serve: cache storage {} bytes, metadata {} bytes",
        report.cache_storage_bytes, report.cache_metadata_bytes,
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
        "serve: {name:>24}: embedding={:.3} ms sliding={:.3} ms full={:.3} ms segments={:.3} ms head={:.3} ms",
        timings.embedding.as_secs_f64() * 1e3,
        timings.sliding_layers.as_secs_f64() * 1e3,
        timings.full_layers.as_secs_f64() * 1e3,
        timings.decode_segments.as_secs_f64() * 1e3,
        timings.head.as_secs_f64() * 1e3,
    );
}

struct Cli {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    prefill_capacity: usize,
    cache_capacity: usize,
    sampling: SamplingOptions,
    backend: Backend,
}

enum Command {
    Run(Cli),
    Help,
}

#[derive(Clone, Copy)]
enum Backend {
    Cpu,
    Cuda,
}

impl Cli {
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Command, String> {
        let mut model = None;
        let mut prompt = None;
        let mut max_new_tokens = 32usize;
        let mut prefill_capacity = None;
        let mut cache_capacity = None;
        let mut sampling = SamplingOptions::default();
        let mut backend = Backend::Cpu;
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
                    backend = match value(&mut arguments, "--backend")?.as_str() {
                        "cpu" => Backend::Cpu,
                        "cuda" => Backend::Cuda,
                        other => return Err(format!("unknown backend {other:?}; use cpu or cuda")),
                    };
                }
                "--help" | "-h" => return Ok(Command::Help),
                other => return Err(format!("unknown argument {other:?}\n{}", usage())),
            }
        }
        Ok(Command::Run(Self {
            model: model.ok_or_else(|| format!("--model is required\n{}", usage()))?,
            prompt: prompt.ok_or_else(|| format!("--prompt is required\n{}", usage()))?,
            max_new_tokens,
            prefill_capacity: prefill_capacity
                .ok_or_else(|| format!("--prefill-capacity is required\n{}", usage()))?,
            cache_capacity: cache_capacity
                .ok_or_else(|| format!("--cache-capacity is required\n{}", usage()))?,
            sampling,
            backend,
        }))
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

fn usage() -> &'static str {
    "usage: serve --model PATH --prompt TEXT --prefill-capacity N --cache-capacity N [--max-new-tokens N] [--seed N] [--temperature F] [--top-k N] [--top-p F] [--min-p F] [--backend cpu|cuda]"
}
