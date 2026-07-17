//! Dense Qwen3 BF16 model adapter and compatibility generation API.

#![forbid(unsafe_code)]

mod config;
mod model;

use crate::engine::{
    self, CacheGeometry, Engine, GraphKind, GraphOutputs, Model, ModelIdentity, ModelPackage,
    ProtocolIdentity,
};
use config::Config;
use nml::io::TensorStore;
use nml::{DataType, Platform, Sharding};
use std::error::Error as StdError;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Inputs accepted by the legacy one-shot Qwen command.
pub struct GenerationOptions {
    pub model_directory: PathBuf,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub cache_capacity: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub tokenization: Duration,
    pub prefill_compilation: Duration,
    pub decode_compilation: Duration,
    pub parameter_upload: Duration,
    pub cache_allocation: Duration,
    pub cache_metadata_upload: Duration,
    pub prompt_upload: Duration,
    pub prefill_execution: Duration,
    pub prefill_download: Duration,
    pub decode_upload: Duration,
    pub first_decode_execution: Duration,
    pub steady_decode_execution: Duration,
    pub decode_download: Duration,
}

pub struct GenerationReport {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub cache_capacity: usize,
    pub timings: Timings,
}

/// Runs one compatibility request through the reusable private model engine.
///
/// The command intentionally retains this small API while the engine keeps
/// compiled executables and parameters independently of Qwen-specific public
/// types. The long-running server will own that same engine lifecycle directly.
pub fn generate(
    platform: &Platform,
    options: &GenerationOptions,
    output: &mut impl Write,
) -> Result<GenerationReport> {
    let package = ModelPackage::<Qwen3>::open(&options.model_directory)?;
    let started = std::time::Instant::now();
    let rendered = QwenProtocol::render_prompt(&options.prompt);
    let tokens = engine::external(package.protocol.protocol().tokenizer.encode(&rendered))?;
    let tokenization = started.elapsed();
    let request = package.protocol.prepare(
        tokens,
        options.max_new_tokens,
        options.cache_capacity,
        tokenization,
    )?;
    let mut decoder = engine::external(package.protocol.protocol().tokenizer.decoder())?;
    let mut engine = Engine::load(platform, package.definition, &request)?;
    let report = engine.generate(request, |token| {
        output.write_all(&engine::external(decoder.push(token))?)?;
        output.flush()?;
        Ok(())
    })?;
    output.write_all(&engine::external(decoder.finish())?)?;
    output.flush()?;
    Ok(GenerationReport {
        prompt_tokens: report.prompt_tokens,
        generated_tokens: report.generated_tokens,
        cache_capacity: report.cache_capacity,
        timings: Timings::from(report.timings),
    })
}

pub(crate) struct Qwen3;

pub(crate) struct QwenProtocol {
    tokenizer: nml::tokenizer::Tokenizer,
}

impl QwenProtocol {
    fn load(model_directory: &Path, configuration: &Config) -> engine::Result<Self> {
        let tokenizer = engine::external(nml::tokenizer::Tokenizer::from_file(
            model_directory.join("tokenizer.json"),
        ))?;
        for (text, expected) in [
            ("<|endoftext|>", configuration.bos_token_id),
            ("<|im_start|>", 151_644),
            ("<|im_end|>", configuration.eos_token_id),
        ] {
            let actual = tokenizer.token_id(text);
            if actual != Some(expected) {
                return Err(engine::Error::contract(format!(
                    "Qwen3 tokenizer maps {text:?} to {actual:?}, expected token {expected}"
                )));
            }
        }
        Ok(Self { tokenizer })
    }

    fn render_prompt(input: &str) -> String {
        format!(
            "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    }
}

impl Model for Qwen3 {
    type Configuration = Config;
    type Checkpoint = model::Checkpoint;
    type Protocol = QwenProtocol;

    const NAME: &'static str = "Qwen3";

    fn load_configuration(model_directory: &Path) -> engine::Result<Self::Configuration> {
        Config::from_file(model_directory.join("config.json"))
            .map_err(|error| engine::Error::model(Self::NAME, error))
    }

    fn identity(configuration: &Self::Configuration) -> ModelIdentity {
        ModelIdentity {
            architecture: "Qwen3ForCausalLM",
            representation: "bf16",
            context_limit: configuration.max_position_embeddings,
        }
    }

    fn load_protocol(
        model_directory: &Path,
        configuration: &Self::Configuration,
    ) -> engine::Result<Self::Protocol> {
        QwenProtocol::load(model_directory, configuration)
    }

    fn protocol_identity(_protocol: &Self::Protocol) -> ProtocolIdentity {
        ProtocolIdentity {
            tokenizer: "Qwen3 tokenizer.json",
            prompt: "qwen3-non-thinking-chat-v1",
        }
    }

    fn eos_token(configuration: &Self::Configuration) -> u32 {
        configuration.eos_token_id
    }

    fn cache_geometry(
        configuration: &Self::Configuration,
        batch_capacity: usize,
        token_capacity: usize,
    ) -> engine::Result<CacheGeometry> {
        CacheGeometry::dense(
            DataType::Bf16,
            configuration.num_hidden_layers,
            batch_capacity,
            token_capacity,
            configuration.num_key_value_heads,
            configuration.head_dim,
        )
    }

    fn placement(_configuration: &Self::Configuration) -> engine::Result<Sharding> {
        Ok(Sharding::single())
    }

    fn declare(
        store: &TensorStore,
        configuration: &Self::Configuration,
    ) -> engine::Result<Self::Checkpoint> {
        model::declare(store, configuration)
            .map_err(|error| engine::Error::model(Self::NAME, error))
    }

    fn build_graph(
        store: &TensorStore,
        checkpoint: &Self::Checkpoint,
        configuration: &Self::Configuration,
        sequence: usize,
        kind: GraphKind,
    ) -> engine::Result<GraphOutputs> {
        model::build_graph(store, checkpoint, configuration, sequence, kind)
            .map_err(|error| engine::Error::model(Self::NAME, error))
    }
}

impl From<engine::Timings> for Timings {
    fn from(timings: engine::Timings) -> Self {
        Self {
            tokenization: timings.tokenization,
            prefill_compilation: timings.prefill_compilation,
            decode_compilation: timings.decode_compilation,
            parameter_upload: timings.parameter_upload,
            cache_allocation: timings.cache_allocation,
            cache_metadata_upload: timings.cache_metadata_upload,
            prompt_upload: timings.prompt_upload,
            prefill_execution: timings.prefill_execution,
            prefill_download: timings.prefill_download,
            decode_upload: timings.decode_upload,
            first_decode_execution: timings.first_decode_execution,
            steady_decode_execution: timings.steady_decode_execution,
            decode_download: timings.decode_download,
        }
    }
}

pub(crate) fn nml_result<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: StdError + 'static,
{
    result.map_err(Error::model)
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(ErrorKind);

#[derive(Debug)]
enum ErrorKind {
    Engine(engine::Error),
    Model(Box<dyn StdError>),
    Contract(&'static str),
}

impl Error {
    pub(crate) fn contract(message: &'static str) -> Self {
        Self(ErrorKind::Contract(message))
    }

    fn model(error: impl StdError + 'static) -> Self {
        Self(ErrorKind::Model(Box::new(error)))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ErrorKind::Engine(error) => error.fmt(formatter),
            ErrorKind::Model(error) => error.fmt(formatter),
            ErrorKind::Contract(message) => {
                write!(formatter, "Qwen3 contract violation: {message}")
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match &self.0 {
            ErrorKind::Engine(error) => Some(error),
            ErrorKind::Model(error) => Some(&**error),
            ErrorKind::Contract(_) => None,
        }
    }
}

impl From<engine::Error> for Error {
    fn from(error: engine::Error) -> Self {
        Self(ErrorKind::Engine(error))
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::model(error)
    }
}
