//! Qwen3 model execution inside the serving product.

#![forbid(unsafe_code)]

mod config;
mod model;

use config::Config;
use model::{GraphKind, GraphOutputs};
use nml::attention::{Cache, CacheSpec};
use nml::exe::Arguments;
use nml::io::{LoadOptions, TensorStore};
use nml::{Buffer, DataType, Memory, NmlStruct, Platform, Shape, Sharding};
use std::error::Error as StdError;
use std::fmt;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

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

pub fn generate(
    platform: &Platform,
    options: &GenerationOptions,
    output: &mut impl Write,
) -> std::result::Result<GenerationReport, Error> {
    if options.max_new_tokens == 0 {
        return Err(Error::Contract("max_new_tokens must be positive"));
    }
    let config = Config::from_file(options.model_directory.join("config.json"))?;

    let tokenization_start = Instant::now();
    let tokenizer =
        nml::tokenizer::Tokenizer::from_file(options.model_directory.join("tokenizer.json"))?;
    validate_chat_tokens(&tokenizer, &config)?;
    let prompt = chat_prompt(&options.prompt);
    let prompt_tokens = tokenizer.encode(&prompt)?;
    let tokenization = tokenization_start.elapsed();
    if prompt_tokens.is_empty() {
        return Err(Error::Contract("the formatted prompt produced no tokens"));
    }
    let required_capacity = prompt_tokens
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or(Error::Contract(
            "prompt and generation length overflow usize",
        ))?;
    let capacity = options.cache_capacity.unwrap_or(required_capacity);
    if capacity < required_capacity {
        return Err(Error::Contract(
            "cache capacity must hold the prompt and complete generation bound",
        ));
    }
    if capacity > config.max_position_embeddings {
        return Err(Error::Contract(
            "cache capacity exceeds the checkpoint's maximum position embedding",
        ));
    }

    let registry = nml_result(nml::safetensors::TensorRegistry::from_path(
        &options.model_directory,
    ))?;
    let prefill_store = TensorStore::new(registry.clone());
    let prefill_model = model::declare(&prefill_store, &config)?;
    let prefill_outputs = model::build_graph(
        &prefill_store,
        &prefill_model,
        &config,
        prompt_tokens.len(),
        GraphKind::Prefill { capacity },
    )?;
    let prefill_program = nml_result(prefill_store.finish(&named_outputs(prefill_outputs)))?;

    let decode_store = TensorStore::new(registry.clone());
    let decode_model = model::declare(&decode_store, &config)?;
    let decode_outputs = model::build_graph(
        &decode_store,
        &decode_model,
        &config,
        1,
        GraphKind::Decode { capacity },
    )?;
    let decode_program = nml_result(decode_store.finish(&named_outputs(decode_outputs)))?;

    let placement = Sharding::single();
    let start = Instant::now();
    let prefill_executable = nml_result(platform.compile(&prefill_program, placement.clone()))?;
    let prefill_compilation = start.elapsed();
    let start = Instant::now();
    let decode_executable = nml_result(platform.compile(&decode_program, placement.clone()))?;
    let decode_compilation = start.elapsed();

    // Loading is deliberately after both compilations. The compiler never
    // competes with 1.2 GB of persistent Qwen3 parameter buffers for host
    // memory, and both executables receive clones of the same PJRT buffers.
    let load_store = TensorStore::new(registry);
    let load_model = model::declare(&load_store, &config)?;
    let loader_parallelism = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(4);
    let load_options =
        nml_result(LoadOptions::new(placement.clone()).parallelism(loader_parallelism))?;
    let start = Instant::now();
    let parameters = nml_result(load_store.load(&load_model, platform, &load_options))?;
    let parameter_upload = start.elapsed();

    let mut prefill_arguments = prefill_executable.args();
    bind_parameters::<model::Checkpoint>(&mut prefill_arguments, &parameters)?;
    let mut decode_arguments = decode_executable.args();
    bind_parameters::<model::Checkpoint>(&mut decode_arguments, &parameters)?;
    drop(parameters);

    let cache_spec = nml_result(CacheSpec::dense(
        DataType::Bf16,
        1,
        capacity,
        config.num_key_value_heads,
        config.head_dim,
    ))?;
    let start = Instant::now();
    let mut caches = (0..config.num_hidden_layers)
        .map(|_| {
            nml_result(Cache::allocate(
                platform,
                cache_spec,
                placement.clone(),
                Memory::Default,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let cache_allocation = start.elapsed();

    let prompt_i32 = prompt_tokens
        .iter()
        .map(|token| {
            i32::try_from(*token)
                .map_err(|_| Error::Contract("token ID exceeds the I32 graph domain"))
        })
        .collect::<Result<Vec<_>>>()?;
    let start = Instant::now();
    let prompt_buffer = upload_i32(
        platform,
        nml_result(Shape::new(
            DataType::I32,
            &[1, usize_i64(prompt_i32.len())?],
        ))?,
        &prompt_i32,
        &placement,
    )?;
    let prompt_upload = start.elapsed();
    nml_result(prefill_arguments.set("tokens", prompt_buffer))?;
    set_caches(&mut prefill_arguments, &mut caches)?;

    let start = Instant::now();
    let prefill_results = nml_result(prefill_arguments.call())?;
    let prefill_execution = start.elapsed();
    let (prefill_token, returned_caches) = split_results(prefill_results, caches.len())?;
    install_caches(&mut caches, returned_caches)?;
    let metadata_start = Instant::now();
    for cache in &mut caches {
        nml_result(cache.truncate(platform, 0, prompt_tokens.len()))?;
    }
    let mut cache_metadata_upload = metadata_start.elapsed();
    let start = Instant::now();
    let mut next_token = download_token(&prefill_token)?;
    let prefill_download = start.elapsed();

    let mut decoder = tokenizer.decoder()?;
    let mut generated_tokens = Vec::with_capacity(options.max_new_tokens);
    let mut decode_upload = Duration::ZERO;
    let mut first_decode_execution = Duration::ZERO;
    let mut steady_decode_execution = Duration::ZERO;
    let mut decode_download = Duration::ZERO;
    for generation_index in 0..options.max_new_tokens {
        if next_token == config.eos_token_id {
            break;
        }
        output.write_all(&decoder.push(next_token)?)?;
        output.flush()?;
        generated_tokens.push(next_token);
        if generation_index + 1 == options.max_new_tokens {
            break;
        }

        let position = prompt_tokens
            .len()
            .checked_add(generation_index)
            .ok_or(Error::Contract("decode position overflowed usize"))?;
        let upload_start = Instant::now();
        let token = i32::try_from(next_token)
            .map_err(|_| Error::Contract("generated token exceeds I32"))?;
        let token_buffer = upload_i32(
            platform,
            nml_result(Shape::new(DataType::I32, &[1, 1]))?,
            &[token],
            &placement,
        )?;
        let position_buffer = upload_i32(
            platform,
            nml_result(Shape::new(DataType::I32, &[]))?,
            &[i32::try_from(position)
                .map_err(|_| Error::Contract("decode position exceeds I32"))?],
            &placement,
        )?;
        decode_upload += upload_start.elapsed();
        nml_result(decode_arguments.set("tokens", token_buffer))?;
        nml_result(decode_arguments.set("position", position_buffer))?;
        set_caches(&mut decode_arguments, &mut caches)?;

        let execute_start = Instant::now();
        let results = nml_result(decode_arguments.call())?;
        let elapsed = execute_start.elapsed();
        if generation_index == 0 {
            first_decode_execution = elapsed;
        } else {
            steady_decode_execution += elapsed;
        }
        let (token, returned_caches) = split_results(results, caches.len())?;
        install_caches(&mut caches, returned_caches)?;
        let metadata_start = Instant::now();
        for cache in &mut caches {
            nml_result(cache.truncate(platform, 0, position + 1))?;
        }
        cache_metadata_upload += metadata_start.elapsed();
        let download_start = Instant::now();
        next_token = download_token(&token)?;
        decode_download += download_start.elapsed();
    }
    output.write_all(&decoder.finish()?)?;
    output.flush()?;

    Ok(GenerationReport {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens,
        cache_capacity: capacity,
        timings: Timings {
            tokenization,
            prefill_compilation,
            decode_compilation,
            parameter_upload,
            cache_allocation,
            cache_metadata_upload,
            prompt_upload,
            prefill_execution,
            prefill_download,
            decode_upload,
            first_decode_execution,
            steady_decode_execution,
            decode_download,
        },
    })
}

fn chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn validate_chat_tokens(tokenizer: &nml::tokenizer::Tokenizer, config: &Config) -> Result<()> {
    for (text, expected) in [
        ("<|endoftext|>", config.bos_token_id),
        ("<|im_start|>", 151_644),
        ("<|im_end|>", config.eos_token_id),
    ] {
        if tokenizer.token_id(text) != Some(expected) {
            return Err(Error::ChatToken {
                text,
                expected,
                actual: tokenizer.token_id(text),
            });
        }
    }
    Ok(())
}

fn named_outputs(outputs: GraphOutputs) -> Vec<(String, nml::Tensor)> {
    let mut named = Vec::with_capacity(1 + outputs.caches.len() * 2);
    named.push(("token".to_owned(), outputs.token));
    for (index, (key, value)) in outputs.caches.into_iter().enumerate() {
        named.push((format!("cache.{index}.key"), key));
        named.push((format!("cache.{index}.value"), value));
    }
    named
}

fn bind_parameters<T: NmlStruct>(
    arguments: &mut Arguments<'_>,
    parameters: &T::Buffers,
) -> Result<()> {
    let mut failure = None;
    T::visit_buffers(parameters, "", &mut |name, buffer| {
        if failure.is_none()
            && let Err(error) = arguments.set(name, buffer.clone())
        {
            failure = Some(Error::External(Box::new(error)));
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    nml_result(arguments.bake())?;
    Ok(())
}

fn set_caches(arguments: &mut Arguments<'_>, caches: &mut [Cache]) -> Result<()> {
    for (index, cache) in caches.iter_mut().enumerate() {
        let (key, value) = nml_result(cache.take_storage())?;
        nml_result(arguments.set(&format!("cache.{index}.key"), key))?;
        nml_result(arguments.set(&format!("cache.{index}.value"), value))?;
    }
    Ok(())
}

fn split_results(
    results: nml::exe::Results,
    cache_count: usize,
) -> Result<(Buffer, Vec<(Buffer, Buffer)>)> {
    let mut buffers = results.into_buffers().into_iter();
    let token = buffers
        .next()
        .ok_or(Error::Contract("Qwen3 execution returned no token buffer"))?;
    let mut caches = Vec::with_capacity(cache_count);
    for _ in 0..cache_count {
        let key = buffers
            .next()
            .ok_or(Error::Contract("Qwen3 execution omitted a key cache"))?;
        let value = buffers
            .next()
            .ok_or(Error::Contract("Qwen3 execution omitted a value cache"))?;
        caches.push((key, value));
    }
    if buffers.next().is_some() {
        return Err(Error::Contract("Qwen3 execution returned extra buffers"));
    }
    Ok((token, caches))
}

fn install_caches(caches: &mut [Cache], buffers: Vec<(Buffer, Buffer)>) -> Result<()> {
    if caches.len() != buffers.len() {
        return Err(Error::Contract(
            "returned cache count changed across execution",
        ));
    }
    for (cache, (key, value)) in caches.iter_mut().zip(buffers) {
        nml_result(cache.replace_storage(key, value))?;
    }
    Ok(())
}

fn upload_i32(
    platform: &Platform,
    shape: Shape,
    values: &[i32],
    placement: &Sharding,
) -> Result<Buffer> {
    let slice = nml_result(nml::Slice::from_typed(shape, values))?;
    nml_result(platform.upload(&slice, placement.clone(), Memory::Default))
}

fn download_token(buffer: &Buffer) -> Result<u32> {
    let slice = nml_result(buffer.to_slice())?;
    let values = nml_result(slice.items::<i32>())?;
    let [token] = values else {
        return Err(Error::Contract("Qwen3 token result is not scalar-shaped"));
    };
    u32::try_from(*token).map_err(|_| Error::Contract("Qwen3 produced a negative token ID"))
}

fn usize_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::Contract("dimension exceeds I64"))
}

pub(crate) fn nml_result<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: StdError + 'static,
{
    result.map_err(|error| Error::External(Box::new(error)))
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Config(String),
    Tokenizer(nml::tokenizer::Error),
    Io(std::io::Error),
    External(Box<dyn StdError>),
    Contract(&'static str),
    ChatToken {
        text: &'static str,
        expected: u32,
        actual: Option<u32>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => error.fmt(formatter),
            Self::Tokenizer(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
            Self::External(error) => error.fmt(formatter),
            Self::Contract(message) => write!(formatter, "Qwen3 contract violation: {message}"),
            Self::ChatToken {
                text,
                expected,
                actual,
            } => write!(
                formatter,
                "Qwen3 tokenizer maps {text:?} to {actual:?}, expected token {expected}"
            ),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Tokenizer(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::External(error) => Some(&**error),
            _ => None,
        }
    }
}

impl From<config::Error> for Error {
    fn from(error: config::Error) -> Self {
        Self::Config(error.to_string())
    }
}

impl From<nml::tokenizer::Error> for Error {
    fn from(error: nml::tokenizer::Error) -> Self {
        Self::Tokenizer(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
