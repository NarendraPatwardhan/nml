//! Persistent compiled model and request-local execution state.

use super::checkpoint::{
    BoxError, Checkpoint, LoadedCheckpoint, LoadedDecoderLayer, Result, bind_tree, message,
    representative_layer,
};
use super::config::{AttentionKind, Config};
use super::graph::{
    CACHE_PAGE_SIZE, ShapeFamily, build_embedding, build_head, build_layer, cache_shape,
    page_table_shape,
};
use nml::exe::{Arguments, Results};
use nml::io::{LoadAccounting, LoadOptions, ParameterSet};
use nml::{Buffer, Exe, Graph, Memory, Platform, Shape, Sharding, Slice};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

const MIN_PREFILL_BUCKET: usize = 16;

pub(super) struct PreparedRequest {
    pub(super) tokens: Vec<u32>,
    pub(super) max_new_tokens: usize,
    pub(super) prefill_bucket: usize,
    pub(super) cache_capacity: usize,
    pub(super) tokenization: Duration,
}

impl PreparedRequest {
    pub(super) fn new(
        tokens: Vec<u32>,
        max_new_tokens: usize,
        requested_cache_capacity: Option<usize>,
        context_limit: usize,
        tokenization: Duration,
    ) -> Result<Self> {
        if tokens.is_empty() {
            return Err(message("GPT-OSS prompt token sequence is empty"));
        }
        let required = tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| message("GPT-OSS request length overflows usize"))?;
        let requested = requested_cache_capacity.unwrap_or(required.max(1));
        if requested < required {
            return Err(message("GPT-OSS cache capacity is smaller than the request"));
        }
        if requested > context_limit {
            return Err(message("GPT-OSS request exceeds the model context limit"));
        }
        let prefill_bucket = tokens
            .len()
            .max(MIN_PREFILL_BUCKET)
            .checked_next_power_of_two()
            .ok_or_else(|| message("GPT-OSS prefill bucket overflows usize"))?;
        let pages = requested
            .div_ceil(CACHE_PAGE_SIZE)
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| message("GPT-OSS cache page bucket overflows usize"))?;
        let cache_capacity = pages
            .checked_mul(CACHE_PAGE_SIZE)
            .ok_or_else(|| message("GPT-OSS cache capacity overflows usize"))?;
        if prefill_bucket > cache_capacity || cache_capacity > context_limit {
            return Err(message("GPT-OSS finite execution family exceeds model capacity"));
        }
        Ok(Self {
            tokens,
            max_new_tokens,
            prefill_bucket,
            cache_capacity,
            tokenization,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct StartupMetrics {
    pub(super) artifact_validation: Duration,
    pub(super) parameter_upload: Duration,
    pub(super) physical_parameter_components: usize,
    pub(super) parameter_source_bytes: usize,
    pub(super) parameter_resident_bytes: usize,
    pub(super) parameter_prepared_bytes: usize,
    pub(super) parameter_peak_staging_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RunMetrics {
    pub(super) prefill_compilation: Duration,
    pub(super) decode_compilation: Duration,
    pub(super) cache_allocation: Duration,
    pub(super) prompt_upload: Duration,
    pub(super) prefill_execution: Duration,
    pub(super) prefill_download: Duration,
    pub(super) decode_state_initialization: Duration,
    pub(super) first_decode_execution: Duration,
    pub(super) steady_decode_execution: Duration,
    pub(super) decode_download: Duration,
}

pub(super) struct RunReport {
    pub(super) generated_tokens: Vec<u32>,
    pub(super) cache_storage_bytes: usize,
    pub(super) cache_metadata_bytes: usize,
    pub(super) stopped: bool,
    pub(super) metrics: RunMetrics,
}

/// Model-wide state: one checkpoint allocation and a cache of bounded compiled
/// component families. Request state never enters this owner.
pub(super) struct CompiledModel<'platform> {
    platform: &'platform Platform,
    config: Config,
    checkpoint: Checkpoint,
    parameters: LoadedCheckpoint,
    placement: Sharding,
    families: BTreeMap<ShapeFamily, ComponentFamily>,
    startup: StartupMetrics,
}

impl<'platform> CompiledModel<'platform> {
    pub(super) fn load(
        platform: &'platform Platform,
        config: Config,
        parameter_set: ParameterSet,
        artifact_validation: Duration,
    ) -> Result<Self> {
        let checkpoint = super::checkpoint::declare(&parameter_set, &config)?;
        let placement = Sharding::single();
        let parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(4);
        let load_options = LoadOptions::new(placement.clone())
            .parallelism(parallelism)
            .map_err(boxed)?;
        let started = Instant::now();
        let (parameters, accounting) = parameter_set
            .load_accounted(&checkpoint, platform, &load_options)
            .map_err(boxed)?;
        let parameter_upload = started.elapsed();
        Ok(Self {
            platform,
            config,
            checkpoint,
            parameters,
            placement,
            families: BTreeMap::new(),
            startup: startup_metrics(artifact_validation, parameter_upload, accounting),
        })
    }

    pub(super) const fn config(&self) -> &Config {
        &self.config
    }

    pub(super) const fn startup(&self) -> StartupMetrics {
        self.startup
    }

    pub(super) fn generate(
        &mut self,
        request: &PreparedRequest,
        mut emit: impl FnMut(u32, bool) -> Result<()>,
    ) -> Result<RunReport> {
        if request.max_new_tokens == 0 {
            return Ok(RunReport {
                generated_tokens: Vec::new(),
                cache_storage_bytes: 0,
                cache_metadata_bytes: 0,
                stopped: false,
                metrics: RunMetrics::default(),
            });
        }
        let prefill = ShapeFamily::prefill(request.prefill_bucket, request.cache_capacity)?;
        let decode = ShapeFamily::decode(request.cache_capacity)?;
        let prefill_compilation = self.ensure_family(prefill)?;
        let decode_compilation = self.ensure_family(decode)?;

        let Self {
            platform,
            config,
            checkpoint,
            parameters,
            placement,
            families,
            ..
        } = self;
        let prefill_executables = families
            .get(&prefill)
            .ok_or_else(|| message("prefill executable family was not retained"))?;
        let decode_executables = families
            .get(&decode)
            .ok_or_else(|| message("decode executable family was not retained"))?;

        let mut prefill_embedding = prefill_executables.embedding.args();
        bind_embedding(&mut prefill_embedding, checkpoint, parameters)?;
        let mut prefill_layers = bind_layers(
            prefill_executables,
            checkpoint,
            parameters,
            config,
        )?;
        let mut prefill_head = prefill_executables.head.args();
        bind_head(&mut prefill_head, parameters)?;

        let mut decode_embedding = decode_executables.embedding.args();
        bind_embedding(&mut decode_embedding, checkpoint, parameters)?;
        let mut decode_layers = bind_layers(
            decode_executables,
            checkpoint,
            parameters,
            config,
        )?;
        let mut decode_head = decode_executables.head.args();
        bind_head(&mut decode_head, parameters)?;

        let allocation_started = Instant::now();
        let cache_tensor_shape = cache_shape(config, prefill)?;
        let mut caches = (0..config.layers())
            .map(|_| LayerCache::allocate(platform, cache_tensor_shape, placement))
            .collect::<Result<Vec<_>>>()?;
        let page_table = identity_page_table(platform, prefill, placement)?;
        let cache_allocation = allocation_started.elapsed();
        let cache_storage_bytes = cache_tensor_shape
            .byte_count()
            .map_err(boxed)?
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_mul(caches.len()))
            .ok_or_else(|| message("GPT-OSS cache storage accounting overflows usize"))?;
        let cache_metadata_bytes = page_table
            .byte_count()
            .map_err(boxed)?
            .checked_add(std::mem::size_of::<i32>())
            .ok_or_else(|| message("GPT-OSS cache metadata accounting overflows usize"))?;

        let prompt_upload_started = Instant::now();
        let mut padded = vec![0_i32; request.prefill_bucket];
        for (destination, token) in padded.iter_mut().zip(&request.tokens) {
            *destination = i32::try_from(*token)
                .map_err(|_| message("GPT-OSS token exceeds the I32 graph domain"))?;
        }
        let prompt = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1, usize_i64(request.prefill_bucket)?])
                .map_err(boxed)?,
            &padded,
            placement,
        )?;
        let prefill_position = upload_scalar(platform, 0, placement)?;
        let sequence_lengths = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1]).map_err(boxed)?,
            &[i32::try_from(request.tokens.len())
                .map_err(|_| message("GPT-OSS prompt length exceeds I32"))?],
            placement,
        )?;
        let last_index = upload_scalar(
            platform,
            i32::try_from(request.tokens.len() - 1)
                .map_err(|_| message("GPT-OSS last prompt index exceeds I32"))?,
            placement,
        )?;
        let prompt_upload = prompt_upload_started.elapsed();

        let prefill_started = Instant::now();
        prefill_embedding.set("tokens", prompt).map_err(boxed)?;
        let mut hidden = one(prefill_embedding.enqueue().map_err(boxed)?)?;
        for (arguments, cache) in prefill_layers.iter_mut().zip(&mut caches) {
            hidden = execute_layer(
                arguments,
                hidden,
                prefill_position.clone(),
                Some(sequence_lengths.clone()),
                page_table.clone(),
                cache,
            )?;
        }
        prefill_head.set("hidden", hidden).map_err(boxed)?;
        prefill_head
            .set("last_index", last_index)
            .map_err(boxed)?;
        let prefill_results = prefill_head.enqueue().map_err(boxed)?;
        let mut prefill_outputs = prefill_results.into_buffers();
        if prefill_outputs.len() != 1 {
            return Err(message("GPT-OSS prefill head returned an invalid result count"));
        }
        let mut token_buffer = prefill_outputs.remove(0);
        token_buffer.wait().map_err(boxed)?;
        let prefill_execution = prefill_started.elapsed();
        let download_started = Instant::now();
        let mut next_token = download_token(&token_buffer)?;
        let prefill_download = download_started.elapsed();

        let decode_state_started = Instant::now();
        let mut position = upload_scalar(
            platform,
            i32::try_from(request.tokens.len())
                .map_err(|_| message("GPT-OSS decode position exceeds I32"))?,
            placement,
        )?;
        let decode_last_index = upload_scalar(platform, 0, placement)?;
        let decode_state_initialization = decode_state_started.elapsed();

        let mut generated_tokens = Vec::with_capacity(request.max_new_tokens);
        let mut stopped = false;
        let mut first_decode_execution = Duration::ZERO;
        let mut steady_decode_execution = Duration::ZERO;
        let mut decode_download = Duration::ZERO;
        for generated_index in 0..request.max_new_tokens {
            let is_stop = super::protocol::is_stop_token(next_token);
            emit(next_token, is_stop)?;
            generated_tokens.push(next_token);
            if is_stop {
                stopped = true;
                break;
            }
            if generated_index + 1 == request.max_new_tokens {
                break;
            }

            let decode_started = Instant::now();
            decode_embedding
                .set("tokens", token_buffer)
                .map_err(boxed)?;
            let mut hidden = one(decode_embedding.enqueue().map_err(boxed)?)?;
            for (arguments, cache) in decode_layers.iter_mut().zip(&mut caches) {
                hidden = execute_layer(
                    arguments,
                    hidden,
                    position.clone(),
                    None,
                    page_table.clone(),
                    cache,
                )?;
            }
            decode_head.set("hidden", hidden).map_err(boxed)?;
            decode_head
                .set("last_index", decode_last_index.clone())
                .map_err(boxed)?;
            decode_head.set("position", position).map_err(boxed)?;
            let mut outputs = decode_head
                .enqueue()
                .map_err(boxed)?
                .into_buffers()
                .into_iter();
            token_buffer = outputs
                .next()
                .ok_or_else(|| message("GPT-OSS decode head omitted its token"))?;
            position = outputs
                .next()
                .ok_or_else(|| message("GPT-OSS decode head omitted its position"))?;
            if outputs.next().is_some() {
                return Err(message("GPT-OSS decode head returned extra buffers"));
            }
            token_buffer.wait().map_err(boxed)?;
            let execution_elapsed = decode_started.elapsed();
            if generated_index == 0 {
                first_decode_execution = execution_elapsed;
            } else {
                steady_decode_execution += execution_elapsed;
            }
            let download_started = Instant::now();
            next_token = download_token(&token_buffer)?;
            decode_download += download_started.elapsed();
        }

        Ok(RunReport {
            generated_tokens,
            cache_storage_bytes,
            cache_metadata_bytes,
            stopped,
            metrics: RunMetrics {
                prefill_compilation,
                decode_compilation,
                cache_allocation,
                prompt_upload,
                prefill_execution,
                prefill_download,
                decode_state_initialization,
                first_decode_execution,
                steady_decode_execution,
                decode_download,
            },
        })
    }

    fn ensure_family(&mut self, family: ShapeFamily) -> Result<Duration> {
        if self.families.contains_key(&family) {
            return Ok(Duration::ZERO);
        }
        let started = Instant::now();
        let compiled = ComponentFamily::compile(
            self.platform,
            &self.placement,
            &self.checkpoint,
            &self.config,
            family,
        )?;
        let elapsed = started.elapsed();
        self.families.insert(family, compiled);
        Ok(elapsed)
    }
}

struct ComponentFamily {
    embedding: Exe,
    sliding_layer: Exe,
    full_layer: Exe,
    head: Exe,
}

impl ComponentFamily {
    fn compile(
        platform: &Platform,
        placement: &Sharding,
        checkpoint: &Checkpoint,
        config: &Config,
        family: ShapeFamily,
    ) -> Result<Self> {
        // Deliberately compile bounded modules sequentially. This is startup
        // scheduling, not a global XLA flag, and has no execution-time cost.
        let embedding = compile(platform, placement, |graph| {
            build_embedding(graph, checkpoint, config, family)
        })?;
        let sliding = representative_layer(
            checkpoint,
            config,
            AttentionKind::SlidingAttention,
        )?;
        let sliding_layer = compile(platform, placement, |graph| {
            build_layer(
                graph,
                sliding,
                config,
                family,
                AttentionKind::SlidingAttention,
            )
        })?;
        let full = representative_layer(checkpoint, config, AttentionKind::FullAttention)?;
        let full_layer = compile(platform, placement, |graph| {
            build_layer(
                graph,
                full,
                config,
                family,
                AttentionKind::FullAttention,
            )
        })?;
        let head = compile(platform, placement, |graph| {
            build_head(graph, checkpoint, config, family)
        })?;
        Ok(Self {
            embedding,
            sliding_layer,
            full_layer,
            head,
        })
    }

    fn layer(&self, kind: AttentionKind) -> &Exe {
        match kind {
            AttentionKind::SlidingAttention => &self.sliding_layer,
            AttentionKind::FullAttention => &self.full_layer,
        }
    }
}

struct LayerCache {
    key: Option<Buffer>,
    value: Option<Buffer>,
}

impl LayerCache {
    fn allocate(platform: &Platform, shape: Shape, placement: &Sharding) -> Result<Self> {
        Ok(Self {
            key: Some(
                platform
                    .upload(&Slice::alloc(shape).map_err(boxed)?, placement.clone(), Memory::Default)
                    .map_err(boxed)?,
            ),
            value: Some(
                platform
                    .upload(&Slice::alloc(shape).map_err(boxed)?, placement.clone(), Memory::Default)
                    .map_err(boxed)?,
            ),
        })
    }

    fn take(&mut self) -> Result<(Buffer, Buffer)> {
        let key = self
            .key
            .take()
            .ok_or_else(|| message("GPT-OSS key cache is owned by an execution"))?;
        let value = self
            .value
            .take()
            .ok_or_else(|| message("GPT-OSS value cache is owned by an execution"))?;
        Ok((key, value))
    }

    fn install(&mut self, key: Buffer, value: Buffer) -> Result<()> {
        if self.key.is_some() || self.value.is_some() {
            return Err(message("GPT-OSS cache output would overwrite live storage"));
        }
        self.key = Some(key);
        self.value = Some(value);
        Ok(())
    }
}

fn bind_embedding(
    arguments: &mut Arguments<'_>,
    checkpoint: &Checkpoint,
    parameters: &LoadedCheckpoint,
) -> Result<()> {
    bind_tree(
        arguments,
        &checkpoint.model.embed_tokens,
        &parameters.model.embed_tokens,
    )
}

fn bind_head(
    arguments: &mut Arguments<'_>,
    parameters: &LoadedCheckpoint,
) -> Result<()> {
    arguments
        .set_parameter(&parameters.model.norm.weight)
        .map_err(boxed)?;
    arguments
        .set_parameter(&parameters.lm_head.weight)
        .map_err(boxed)?;
    arguments.bake().map_err(boxed)?;
    Ok(())
}

fn bind_layers<'family>(
    family: &'family ComponentFamily,
    checkpoint: &Checkpoint,
    parameters: &LoadedCheckpoint,
    config: &Config,
) -> Result<Vec<Arguments<'family>>> {
    if checkpoint.model.layers.len() != parameters.model.layers.len()
        || checkpoint.model.layers.len() != config.layer_types().len()
    {
        return Err(message("GPT-OSS layer schedule and checkpoint disagree"));
    }
    checkpoint
        .model
        .layers
        .iter()
        .zip(&parameters.model.layers)
        .zip(config.layer_types())
        .map(|((_, loaded), kind)| {
            let executable = family.layer(*kind);
            let slots = representative_layer(checkpoint, config, *kind)?;
            bind_layer(executable, slots, loaded)
        })
        .collect()
}

fn bind_layer<'family>(
    executable: &'family Exe,
    slots: &super::checkpoint::DecoderLayer,
    loaded: &LoadedDecoderLayer,
) -> Result<Arguments<'family>> {
    let mut arguments = executable.args();
    bind_tree(&mut arguments, slots, loaded)?;
    Ok(arguments)
}

fn execute_layer(
    arguments: &mut Arguments<'_>,
    hidden: Buffer,
    position: Buffer,
    sequence_lengths: Option<Buffer>,
    page_table: Buffer,
    cache: &mut LayerCache,
) -> Result<Buffer> {
    let (key, value) = cache.take()?;
    arguments.set("hidden", hidden).map_err(boxed)?;
    arguments.set("position", position).map_err(boxed)?;
    if let Some(lengths) = sequence_lengths {
        arguments
            .set("sequence_lengths", lengths)
            .map_err(boxed)?;
    }
    arguments.set("page_table", page_table).map_err(boxed)?;
    arguments.set("cache.key", key).map_err(boxed)?;
    arguments.set("cache.value", value).map_err(boxed)?;
    let mut outputs = arguments
        .enqueue()
        .map_err(boxed)?
        .into_buffers()
        .into_iter();
    let hidden = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer omitted hidden state"))?;
    let key = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer omitted key cache"))?;
    let value = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer omitted value cache"))?;
    if outputs.next().is_some() {
        return Err(message("GPT-OSS layer returned extra buffers"));
    }
    cache.install(key, value)?;
    Ok(hidden)
}

fn compile(
    platform: &Platform,
    placement: &Sharding,
    build: impl FnOnce(&mut Graph) -> Result<Vec<(String, nml::Tensor)>>,
) -> Result<Exe> {
    let mut graph = Graph::new();
    let outputs = build(&mut graph)?;
    let program = graph.finish_named(&outputs).map_err(boxed)?;
    platform
        .compile(&program, placement.clone())
        .map_err(boxed)
}

fn identity_page_table(
    platform: &Platform,
    family: ShapeFamily,
    placement: &Sharding,
) -> Result<Buffer> {
    let pages = (0..family.page_count())
        .map(|page| i32::try_from(page).map_err(|_| message("page index exceeds I32")))
        .collect::<Result<Vec<_>>>()?;
    upload_i32(platform, page_table_shape(family)?, &pages, placement)
}

fn upload_scalar(platform: &Platform, value: i32, placement: &Sharding) -> Result<Buffer> {
    upload_i32(
        platform,
        Shape::new(nml::DataType::I32, &[]).map_err(boxed)?,
        &[value],
        placement,
    )
}

fn upload_i32(
    platform: &Platform,
    shape: Shape,
    values: &[i32],
    placement: &Sharding,
) -> Result<Buffer> {
    let slice = Slice::from_typed(shape, values).map_err(boxed)?;
    platform
        .upload(&slice, placement.clone(), Memory::Default)
        .map_err(boxed)
}

fn download_token(buffer: &Buffer) -> Result<u32> {
    let slice = buffer.to_slice().map_err(boxed)?;
    let values = slice.items::<i32>().map_err(boxed)?;
    let [token] = values else {
        return Err(message("GPT-OSS token output is not scalar-shaped"));
    };
    u32::try_from(*token).map_err(|_| message("GPT-OSS produced a negative token"))
}

fn one(results: Results) -> Result<Buffer> {
    let mut buffers = results.into_buffers();
    if buffers.len() != 1 {
        return Err(message("GPT-OSS component returned an invalid result count"));
    }
    Ok(buffers.remove(0))
}

fn startup_metrics(
    artifact_validation: Duration,
    parameter_upload: Duration,
    accounting: LoadAccounting,
) -> StartupMetrics {
    StartupMetrics {
        artifact_validation,
        parameter_upload,
        physical_parameter_components: accounting.physical_components(),
        parameter_source_bytes: accounting.source_bytes(),
        parameter_resident_bytes: accounting.resident_bytes(),
        parameter_prepared_bytes: accounting.prepared_bytes(),
        parameter_peak_staging_bytes: accounting.peak_staging_bytes(),
    }
}

fn boxed<E>(error: E) -> BoxError
where
    E: std::error::Error + Send + Sync + 'static,
{
    Box::new(error)
}

fn usize_i64(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| message("GPT-OSS dimension exceeds I64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_select_finite_reusable_shape_families() {
        let request = PreparedRequest::new(
            vec![7; 17],
            5,
            None,
            131_072,
            Duration::from_millis(1),
        )
        .unwrap();
        assert_eq!(request.prefill_bucket, 32);
        assert_eq!(request.cache_capacity, CACHE_PAGE_SIZE);

        let request = PreparedRequest::new(
            vec![7; 17],
            5,
            Some(300),
            131_072,
            Duration::ZERO,
        )
        .unwrap();
        assert_eq!(request.prefill_bucket, 32);
        assert_eq!(request.cache_capacity, 2 * CACHE_PAGE_SIZE);
    }

    #[test]
    fn requests_reject_unrepresentable_or_undersized_families() {
        assert!(
            PreparedRequest::new(vec![], 1, None, 131_072, Duration::ZERO).is_err()
        );
        assert!(
            PreparedRequest::new(vec![1; 17], 5, Some(21), 131_072, Duration::ZERO).is_err()
        );
        assert!(
            PreparedRequest::new(vec![1; 17], 5, Some(131_073), 131_072, Duration::ZERO)
                .is_err()
        );
    }
}
