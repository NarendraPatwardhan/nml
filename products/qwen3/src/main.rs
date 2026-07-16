use nml_qwen3::{GenerationOptions, Timings};
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("qwen3: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
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
    let report = nml_qwen3::generate(
        &platform,
        &GenerationOptions {
            model_directory: cli.model,
            prompt: cli.prompt,
            max_new_tokens: cli.max_new_tokens,
            cache_capacity: cli.cache_capacity,
        },
        &mut stdout,
    )?;
    eprintln!();
    eprintln!(
        "qwen3: {} prompt tokens, {} generated tokens, cache capacity {}",
        report.prompt_tokens,
        report.generated_tokens.len(),
        report.cache_capacity
    );
    print_timings(&report.timings);
    eprintln!("qwen3: token IDs {:?}", report.generated_tokens);
    Ok(())
}

fn print_timings(timings: &Timings) {
    for (name, duration) in [
        ("tokenization", timings.tokenization),
        ("prefill compilation", timings.prefill_compilation),
        ("decode compilation", timings.decode_compilation),
        ("parameter upload", timings.parameter_upload),
        ("cache allocation", timings.cache_allocation),
        ("cache metadata upload", timings.cache_metadata_upload),
        ("prompt upload", timings.prompt_upload),
        ("prefill execution", timings.prefill_execution),
        ("prefill download", timings.prefill_download),
        ("decode upload", timings.decode_upload),
        ("first decode execution", timings.first_decode_execution),
        ("steady decode execution", timings.steady_decode_execution),
        ("decode download", timings.decode_download),
    ] {
        eprintln!("qwen3: {name:>24}: {:9.3} ms", duration.as_secs_f64() * 1e3);
    }
}

struct Cli {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    cache_capacity: Option<usize>,
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
        let mut cache_capacity = None;
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
                "--cache-capacity" => {
                    cache_capacity = Some(parse_usize(
                        &value(&mut arguments, "--cache-capacity")?,
                        "--cache-capacity",
                    )?);
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
            cache_capacity,
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

fn usage() -> &'static str {
    "usage: qwen3 --model PATH --prompt TEXT [--max-new-tokens N] [--cache-capacity N] [--backend cpu|cuda]"
}
