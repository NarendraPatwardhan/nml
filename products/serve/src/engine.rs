//! Private model execution lifecycle shared by serving product models.

#![forbid(unsafe_code)]

use nml::attention::{Cache, CacheSpec};
use nml::exe::Arguments;
use nml::io::{LoadOptions, ParameterSet};
use nml::safetensors::TensorRegistry;
use nml::{Buffer, DataType, Graph, Loaded, Memory, ParameterTree, Platform, Shape, Sharding};
use std::error::Error as StdError;
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::time::{Duration, Instant};

/// The narrow contract a product model implements for the execution engine.
///
/// This trait is deliberately private. Model configuration, checkpoint, graph,
/// tokenizer, and sharding choices belong in `products/serve`; the scheduler
/// will consume `Engine` without exporting a model taxonomy through `nml`.
pub(crate) trait Model: Sized {
    type Configuration;
    type Checkpoint: ParameterTree;
    type Protocol;

    const NAME: &'static str;

    fn load_configuration(model_directory: &Path) -> Result<Self::Configuration>;
    fn identity(configuration: &Self::Configuration) -> ModelIdentity;
    fn load_protocol(
        model_directory: &Path,
        configuration: &Self::Configuration,
    ) -> Result<Self::Protocol>;
    fn protocol_identity(protocol: &Self::Protocol) -> ProtocolIdentity;
    fn eos_token(configuration: &Self::Configuration) -> u32;
    fn cache_geometry(
        configuration: &Self::Configuration,
        batch_capacity: usize,
        token_capacity: usize,
    ) -> Result<CacheGeometry>;
    fn placement(configuration: &Self::Configuration) -> Result<Sharding>;
    fn declare(
        parameters: &ParameterSet,
        configuration: &Self::Configuration,
    ) -> Result<Self::Checkpoint>;
    fn build_graph(
        graph: &mut Graph,
        checkpoint: &Self::Checkpoint,
        configuration: &Self::Configuration,
        sequence: usize,
        kind: GraphKind,
    ) -> Result<GraphOutputs>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModelIdentity {
    pub(crate) architecture: &'static str,
    pub(crate) representation: &'static str,
    pub(crate) context_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProtocolIdentity {
    pub(crate) tokenizer: &'static str,
    pub(crate) prompt: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheGeometry {
    dtype: DataType,
    layers: usize,
    batch_capacity: usize,
    token_capacity: usize,
    key_value_heads: usize,
    head_dimensions: usize,
}

impl CacheGeometry {
    pub(crate) fn dense(
        dtype: DataType,
        layers: usize,
        batch_capacity: usize,
        token_capacity: usize,
        key_value_heads: usize,
        head_dimensions: usize,
    ) -> Result<Self> {
        if layers == 0
            || batch_capacity == 0
            || token_capacity == 0
            || key_value_heads == 0
            || head_dimensions == 0
        {
            return Err(Error::contract(
                "cache geometry dimensions and layer count must be positive",
            ));
        }
        Ok(Self {
            dtype,
            layers,
            batch_capacity,
            token_capacity,
            key_value_heads,
            head_dimensions,
        })
    }

    fn spec(self) -> Result<CacheSpec> {
        external(CacheSpec::dense(
            self.dtype,
            self.batch_capacity,
            self.token_capacity,
            self.key_value_heads,
            self.head_dimensions,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphKind {
    Prefill { batch: usize, capacity: usize },
    Decode { batch: usize, capacity: usize },
}

pub(crate) struct GraphOutputs {
    pub(crate) token: nml::Tensor,
    pub(crate) caches: Vec<(nml::Tensor, nml::Tensor)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Sampling {
    Greedy,
}

pub(crate) struct ModelPackage<M: Model> {
    pub(crate) definition: PreparedModel<M>,
    pub(crate) protocol: ProtocolOwner<M>,
}

pub(crate) struct PreparedModel<M: Model> {
    configuration: M::Configuration,
    registry: TensorRegistry,
    identity: ModelIdentity,
    protocol: ProtocolIdentity,
}

pub(crate) struct ProtocolOwner<M: Model> {
    protocol: M::Protocol,
    identity: ProtocolIdentity,
    context_limit: usize,
    model: PhantomData<fn() -> M>,
}

impl<M: Model> ModelPackage<M> {
    pub(crate) fn open(model_directory: &Path) -> Result<Self> {
        let configuration = M::load_configuration(model_directory)?;
        let identity = M::identity(&configuration);
        validate_identity(M::NAME, &identity)?;
        let protocol = M::load_protocol(model_directory, &configuration)?;
        let protocol_identity = M::protocol_identity(&protocol);
        validate_protocol(M::NAME, &protocol_identity)?;
        let registry = external(TensorRegistry::from_path(model_directory))?;
        Ok(Self {
            definition: PreparedModel {
                configuration,
                registry,
                identity: identity.clone(),
                protocol: protocol_identity.clone(),
            },
            protocol: ProtocolOwner {
                protocol,
                identity: protocol_identity,
                context_limit: identity.context_limit,
                model: PhantomData,
            },
        })
    }
}

impl<M: Model> ProtocolOwner<M> {
    pub(crate) fn prepare(
        &self,
        tokens: Vec<u32>,
        max_new_tokens: usize,
        cache_capacity: Option<usize>,
        tokenization: Duration,
    ) -> Result<PreparedRequest> {
        let contract = |message: &str| {
            Error::contract(format!(
                "protocol {} with {}: {message}",
                self.identity.prompt, self.identity.tokenizer
            ))
        };
        if max_new_tokens == 0 {
            return Err(contract("max_new_tokens must be positive"));
        }
        if tokens.is_empty() {
            return Err(contract("the formatted prompt produced no tokens"));
        }
        let required_capacity = tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| contract("prompt and generation length overflow usize"))?;
        let cache_capacity = cache_capacity.unwrap_or(required_capacity);
        if cache_capacity < required_capacity {
            return Err(contract(
                "cache capacity must hold the prompt and complete generation bound",
            ));
        }
        if cache_capacity > self.context_limit {
            return Err(contract(
                "cache capacity exceeds the model's validated context limit",
            ));
        }
        Ok(PreparedRequest {
            tokens,
            max_new_tokens,
            cache_capacity,
            tokenization,
            sampling: Sampling::Greedy,
        })
    }

    pub(crate) fn protocol(&self) -> &M::Protocol {
        &self.protocol
    }
}

pub(crate) struct PreparedRequest {
    tokens: Vec<u32>,
    max_new_tokens: usize,
    cache_capacity: usize,
    tokenization: Duration,
    sampling: Sampling,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExecutionShape {
    batch_capacity: usize,
    prefill_tokens: usize,
    cache_capacity: usize,
}

impl ExecutionShape {
    fn for_request(request: &PreparedRequest) -> Self {
        Self {
            batch_capacity: 1,
            prefill_tokens: request.tokens.len(),
            cache_capacity: request.cache_capacity,
        }
    }

    fn validate_request(self, request: &PreparedRequest) -> Result<()> {
        if request.tokens.len() != self.prefill_tokens {
            return Err(Error::contract(
                "request token length is outside the compiled prefill family",
            ));
        }
        if request.cache_capacity != self.cache_capacity {
            return Err(Error::contract(
                "request cache capacity is outside the compiled executable family",
            ));
        }
        Ok(())
    }
}

struct ExecutableFamily {
    shape: ExecutionShape,
    prefill: nml::Exe,
    decode: nml::Exe,
}

/// One product-model lifecycle owned by the future serving engine thread.
///
/// Parameters and executables persist across requests. Per-request cache and
/// sampling state remain isolated, which keeps this boundary usable before the
/// server replaces dense compatibility caches with its global paged arena.
pub(crate) struct Engine<'platform, M: Model> {
    platform: &'platform Platform,
    model: PreparedModel<M>,
    placement: Sharding,
    cache_geometry: CacheGeometry,
    executables: ExecutableFamily,
    parameters: Loaded<M::Checkpoint>,
    startup_timings: StartupTimings,
}

impl<'platform, M: Model> Engine<'platform, M> {
    pub(crate) fn load(
        platform: &'platform Platform,
        model: PreparedModel<M>,
        request: &PreparedRequest,
    ) -> Result<Self> {
        let shape = ExecutionShape::for_request(request);
        let placement = M::placement(&model.configuration)?;
        let cache_geometry = M::cache_geometry(
            &model.configuration,
            shape.batch_capacity,
            shape.cache_capacity,
        )?;
        if cache_geometry.batch_capacity != shape.batch_capacity
            || cache_geometry.token_capacity != shape.cache_capacity
        {
            return Err(Error::contract(
                "model cache geometry disagrees with the executable family",
            ));
        }

        let parameter_set = ParameterSet::new(model.registry.clone());
        let checkpoint = M::declare(&parameter_set, &model.configuration)?;
        let (prefill, prefill_compilation) = compile_graph::<M>(
            platform,
            &placement,
            &model,
            &checkpoint,
            shape.prefill_tokens,
            GraphKind::Prefill {
                batch: shape.batch_capacity,
                capacity: shape.cache_capacity,
            },
        )?;
        let (decode, decode_compilation) = compile_graph::<M>(
            platform,
            &placement,
            &model,
            &checkpoint,
            1,
            GraphKind::Decode {
                batch: shape.batch_capacity,
                capacity: shape.cache_capacity,
            },
        )?;

        // Loading follows compilation so compiler memory does not compete with
        // the persistent checkpoint allocation. The stored buffers are cloned
        // into fresh argument sets for every request and never re-uploaded.
        let loader_parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(4);
        let load_options =
            external(LoadOptions::new(placement.clone()).parallelism(loader_parallelism))?;
        let started = Instant::now();
        let parameters = external(parameter_set.load(&checkpoint, platform, &load_options))?;
        let parameter_upload = started.elapsed();

        Ok(Self {
            platform,
            model,
            placement,
            cache_geometry,
            executables: ExecutableFamily {
                shape,
                prefill,
                decode,
            },
            parameters,
            startup_timings: StartupTimings {
                prefill_compilation,
                decode_compilation,
                parameter_upload,
            },
        })
    }

    pub(crate) fn generate(
        &mut self,
        request: PreparedRequest,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenerationReport> {
        self.executables
            .shape
            .validate_request(&request)
            .map_err(|source| Error::Request {
                architecture: self.model.identity.architecture,
                protocol: self.model.protocol.prompt,
                source: Box::new(source),
            })?;
        if request.sampling != Sampling::Greedy {
            return Err(Error::contract(
                "the current executable family supports greedy sampling only",
            ));
        }

        let mut prefill_arguments = self.executables.prefill.args();
        bind_parameters::<M::Checkpoint>(&mut prefill_arguments, &self.parameters)?;
        let mut decode_arguments = self.executables.decode.args();
        bind_parameters::<M::Checkpoint>(&mut decode_arguments, &self.parameters)?;

        let started = Instant::now();
        let cache_spec = self.cache_geometry.spec()?;
        let mut caches = (0..self.cache_geometry.layers)
            .map(|_| {
                external(Cache::allocate(
                    self.platform,
                    cache_spec,
                    self.placement.clone(),
                    Memory::Default,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let cache_allocation = started.elapsed();

        let prompt = request
            .tokens
            .iter()
            .map(|token| {
                i32::try_from(*token)
                    .map_err(|_| Error::contract("token ID exceeds the I32 graph domain"))
            })
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let prompt_buffer = upload_i32(
            self.platform,
            external(Shape::new(DataType::I32, &[1, usize_i64(prompt.len())?]))?,
            &prompt,
            &self.placement,
        )?;
        let prompt_upload = started.elapsed();
        external(prefill_arguments.set("tokens", prompt_buffer))?;
        set_caches(&mut prefill_arguments, &mut caches)?;

        let started = Instant::now();
        let prefill_results = external(prefill_arguments.call())?;
        let prefill_execution = started.elapsed();
        let (prefill_token, returned_caches) = split_results(prefill_results, caches.len())?;
        install_caches(&mut caches, returned_caches)?;
        let metadata_started = Instant::now();
        for cache in &mut caches {
            external(cache.truncate(self.platform, 0, request.tokens.len()))?;
        }
        let mut cache_metadata_upload = metadata_started.elapsed();
        let started = Instant::now();
        let mut next_token = download_token(&prefill_token)?;
        let prefill_download = started.elapsed();

        let mut generated_tokens = Vec::with_capacity(request.max_new_tokens);
        let mut decode_upload = Duration::ZERO;
        let mut first_decode_execution = Duration::ZERO;
        let mut steady_decode_execution = Duration::ZERO;
        let mut decode_download = Duration::ZERO;
        for generation_index in 0..request.max_new_tokens {
            if next_token == M::eos_token(&self.model.configuration) {
                break;
            }
            emit(next_token)?;
            generated_tokens.push(next_token);
            if generation_index + 1 == request.max_new_tokens {
                break;
            }

            let position = request
                .tokens
                .len()
                .checked_add(generation_index)
                .ok_or_else(|| Error::contract("decode position overflowed usize"))?;
            let upload_started = Instant::now();
            let token = i32::try_from(next_token)
                .map_err(|_| Error::contract("generated token exceeds I32"))?;
            let token_buffer = upload_i32(
                self.platform,
                external(Shape::new(DataType::I32, &[1, 1]))?,
                &[token],
                &self.placement,
            )?;
            let position_buffer = upload_i32(
                self.platform,
                external(Shape::new(DataType::I32, &[]))?,
                &[i32::try_from(position)
                    .map_err(|_| Error::contract("decode position exceeds I32"))?],
                &self.placement,
            )?;
            decode_upload += upload_started.elapsed();
            external(decode_arguments.set("tokens", token_buffer))?;
            external(decode_arguments.set("position", position_buffer))?;
            set_caches(&mut decode_arguments, &mut caches)?;

            let execute_started = Instant::now();
            let results = external(decode_arguments.call())?;
            let elapsed = execute_started.elapsed();
            if generation_index == 0 {
                first_decode_execution = elapsed;
            } else {
                steady_decode_execution += elapsed;
            }
            let (token, returned_caches) = split_results(results, caches.len())?;
            install_caches(&mut caches, returned_caches)?;
            let metadata_started = Instant::now();
            for cache in &mut caches {
                external(cache.truncate(self.platform, 0, position + 1))?;
            }
            cache_metadata_upload += metadata_started.elapsed();
            let download_started = Instant::now();
            next_token = download_token(&token)?;
            decode_download += download_started.elapsed();
        }
        Ok(GenerationReport {
            prompt_tokens: request.tokens.len(),
            generated_tokens,
            cache_capacity: request.cache_capacity,
            timings: Timings {
                tokenization: request.tokenization,
                prefill_compilation: self.startup_timings.prefill_compilation,
                decode_compilation: self.startup_timings.decode_compilation,
                parameter_upload: self.startup_timings.parameter_upload,
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
}

fn compile_graph<M: Model>(
    platform: &Platform,
    placement: &Sharding,
    model: &PreparedModel<M>,
    checkpoint: &M::Checkpoint,
    sequence: usize,
    kind: GraphKind,
) -> Result<(nml::Exe, Duration)> {
    let mut graph = Graph::new();
    let outputs = M::build_graph(&mut graph, checkpoint, &model.configuration, sequence, kind)?;
    let program = external(graph.finish_named(&named_outputs(outputs)))?;
    let started = Instant::now();
    let executable = external(platform.compile(&program, placement.clone()))?;
    Ok((executable, started.elapsed()))
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

fn bind_parameters<T: ParameterTree>(
    arguments: &mut Arguments<'_>,
    parameters: &Loaded<T>,
) -> Result<()> {
    let mut failure = None;
    T::visit_loaded(parameters, "", &mut |_name, parameter| {
        if failure.is_none()
            && let Err(error) = arguments.set_parameter(parameter)
        {
            failure = Some(Error::external(error));
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    external(arguments.bake())?;
    Ok(())
}

fn set_caches(arguments: &mut Arguments<'_>, caches: &mut [Cache]) -> Result<()> {
    for (index, cache) in caches.iter_mut().enumerate() {
        let (key, value) = external(cache.take_storage())?;
        external(arguments.set(&format!("cache.{index}.key"), key))?;
        external(arguments.set(&format!("cache.{index}.value"), value))?;
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
        .ok_or_else(|| Error::contract("model execution returned no token buffer"))?;
    let mut caches = Vec::with_capacity(cache_count);
    for _ in 0..cache_count {
        let key = buffers
            .next()
            .ok_or_else(|| Error::contract("model execution omitted a key cache"))?;
        let value = buffers
            .next()
            .ok_or_else(|| Error::contract("model execution omitted a value cache"))?;
        caches.push((key, value));
    }
    if buffers.next().is_some() {
        return Err(Error::contract("model execution returned extra buffers"));
    }
    Ok((token, caches))
}

fn install_caches(caches: &mut [Cache], buffers: Vec<(Buffer, Buffer)>) -> Result<()> {
    if caches.len() != buffers.len() {
        return Err(Error::contract(
            "returned cache count changed across execution",
        ));
    }
    for (cache, (key, value)) in caches.iter_mut().zip(buffers) {
        external(cache.replace_storage(key, value))?;
    }
    Ok(())
}

fn upload_i32(
    platform: &Platform,
    shape: Shape,
    values: &[i32],
    placement: &Sharding,
) -> Result<Buffer> {
    let slice = external(nml::Slice::from_typed(shape, values))?;
    external(platform.upload(&slice, placement.clone(), Memory::Default))
}

fn download_token(buffer: &Buffer) -> Result<u32> {
    let slice = external(buffer.to_slice())?;
    let values = external(slice.items::<i32>())?;
    let [token] = values else {
        return Err(Error::contract("model token result is not scalar-shaped"));
    };
    u32::try_from(*token).map_err(|_| Error::contract("model produced a negative token ID"))
}

fn usize_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::contract("dimension exceeds I64"))
}

fn validate_identity(model: &str, identity: &ModelIdentity) -> Result<()> {
    if identity.architecture.is_empty()
        || identity.representation.is_empty()
        || identity.context_limit == 0
    {
        return Err(Error::contract(format!(
            "{model} returned an incomplete model identity"
        )));
    }
    Ok(())
}

fn validate_protocol(model: &str, protocol: &ProtocolIdentity) -> Result<()> {
    if protocol.tokenizer.is_empty() || protocol.prompt.is_empty() {
        return Err(Error::contract(format!(
            "{model} returned an incomplete tokenizer/protocol identity"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Timings {
    pub(crate) tokenization: Duration,
    pub(crate) prefill_compilation: Duration,
    pub(crate) decode_compilation: Duration,
    pub(crate) parameter_upload: Duration,
    pub(crate) cache_allocation: Duration,
    pub(crate) cache_metadata_upload: Duration,
    pub(crate) prompt_upload: Duration,
    pub(crate) prefill_execution: Duration,
    pub(crate) prefill_download: Duration,
    pub(crate) decode_upload: Duration,
    pub(crate) first_decode_execution: Duration,
    pub(crate) steady_decode_execution: Duration,
    pub(crate) decode_download: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct StartupTimings {
    prefill_compilation: Duration,
    decode_compilation: Duration,
    parameter_upload: Duration,
}

pub(crate) struct GenerationReport {
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) cache_capacity: usize,
    pub(crate) timings: Timings,
}

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub(crate) enum Error {
    Contract(String),
    Request {
        architecture: &'static str,
        protocol: &'static str,
        source: Box<dyn StdError>,
    },
    Model {
        model: &'static str,
        source: Box<dyn StdError>,
    },
    External(Box<dyn StdError>),
    Io(std::io::Error),
}

impl Error {
    pub(crate) fn contract(message: impl Into<String>) -> Self {
        Self::Contract(message.into())
    }

    pub(crate) fn model(model: &'static str, source: impl StdError + 'static) -> Self {
        Self::Model {
            model,
            source: Box::new(source),
        }
    }

    pub(crate) fn external(source: impl StdError + 'static) -> Self {
        Self::External(Box::new(source))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(message) => {
                write!(formatter, "model-engine contract violation: {message}")
            }
            Self::Request {
                architecture,
                protocol,
                source,
            } => write!(
                formatter,
                "{architecture} request under {protocol}: {source}"
            ),
            Self::Model { model, source } => write!(formatter, "{model}: {source}"),
            Self::External(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Request { source, .. } => Some(&**source),
            Self::Model { source, .. } => Some(&**source),
            Self::External(error) => Some(&**error),
            Self::Io(error) => Some(error),
            Self::Contract(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub(crate) fn external<T, E>(result: std::result::Result<T, E>) -> Result<T>
where
    E: StdError + 'static,
{
    result.map_err(Error::external)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executable_family_accepts_only_its_compiled_request_shape() {
        let shape = ExecutionShape {
            batch_capacity: 1,
            prefill_tokens: 4,
            cache_capacity: 8,
        };
        let matching = PreparedRequest {
            tokens: vec![1, 2, 3, 4],
            max_new_tokens: 2,
            cache_capacity: 8,
            tokenization: Duration::ZERO,
            sampling: Sampling::Greedy,
        };
        assert!(shape.validate_request(&matching).is_ok());

        let different_prefill = PreparedRequest {
            tokens: vec![1, 2, 3],
            ..matching
        };
        assert!(
            shape
                .validate_request(&different_prefill)
                .unwrap_err()
                .to_string()
                .contains("compiled prefill family")
        );

        let different_cache = PreparedRequest {
            tokens: vec![1, 2, 3, 4],
            cache_capacity: 9,
            ..different_prefill
        };
        assert!(
            shape
                .validate_request(&different_cache)
                .unwrap_err()
                .to_string()
                .contains("compiled executable family")
        );
    }

    #[test]
    fn cache_geometry_rejects_incomplete_ownership_shape() {
        assert!(
            CacheGeometry::dense(DataType::Bf16, 0, 1, 8, 2, 4)
                .unwrap_err()
                .to_string()
                .contains("cache geometry")
        );
    }

    #[test]
    fn model_and_protocol_identity_must_be_complete() {
        let identity = ModelIdentity {
            architecture: "",
            representation: "bf16",
            context_limit: 8,
        };
        assert!(validate_identity("test", &identity).is_err());
        let protocol = ProtocolIdentity {
            tokenizer: "tokenizer",
            prompt: "",
        };
        assert!(validate_protocol("test", &protocol).is_err());
    }
}
