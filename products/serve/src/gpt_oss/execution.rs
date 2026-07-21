//! Definition, compilation, residency, and request-local execution state.

use super::checkpoint::{
    bind_tree, bind_tree_components, message, representative_layer, BoxError, Checkpoint,
    LoadedCheckpoint, LoadedDecoderLayer, Result,
};
use super::config::{AttentionKind, Config};
use super::graph::{
    build_decode_layer_pair, build_embedding, build_head, build_layer, cache_shape,
    page_table_shape, Phase, ShapeFamily, CACHE_PAGE_SIZE, MAXIMUM_TOP_K,
};
use crate::{CompilationProfile, SamplingOptions, SubmissionTimings};
use nml::exe::{Arguments, Results};
use nml::io::{LoadAccounting, LoadOptions, ParameterSet};
use nml::{Buffer, Exe, Graph, Memory, Platform, Shape, Sharding, Slice};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

const MIN_PREFILL_BUCKET: usize = 16;

pub(super) struct PreparedRequest {
    pub(super) tokens: Vec<u32>,
    pub(super) max_new_tokens: usize,
    pub(super) sampling: SamplingOptions,
    required_cache_capacity: usize,
    pub(super) tokenization: Duration,
}

impl PreparedRequest {
    pub(super) fn new(
        tokens: Vec<u32>,
        max_new_tokens: usize,
        requested_cache_capacity: Option<usize>,
        sampling: SamplingOptions,
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
            return Err(message(
                "GPT-OSS cache capacity is smaller than the request",
            ));
        }
        if requested > context_limit {
            return Err(message("GPT-OSS request exceeds the model context limit"));
        }
        if sampling.top_k == 0 || sampling.top_k > MAXIMUM_TOP_K {
            return Err(message(
                "GPT-OSS top-k must be between one and the compiled candidate capacity",
            ));
        }
        if !sampling.temperature.is_finite() || sampling.temperature <= 0.0 {
            return Err(message(
                "GPT-OSS sampling temperature must be finite and positive",
            ));
        }
        if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
            return Err(message("GPT-OSS top-p must be in (0, 1]"));
        }
        if !sampling.min_p.is_finite() || !(0.0..=1.0).contains(&sampling.min_p) {
            return Err(message("GPT-OSS min-p must be in [0, 1]"));
        }
        Ok(Self {
            tokens,
            max_new_tokens,
            sampling,
            required_cache_capacity: requested,
            tokenization,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ExecutionProfile {
    prefill: ShapeFamily,
    decode: ShapeFamily,
}

impl ExecutionProfile {
    fn new(profile: CompilationProfile, context_limit: usize) -> Result<Self> {
        if profile.max_prompt_tokens == 0 || profile.max_sequence_tokens == 0 {
            return Err(message(
                "GPT-OSS compilation profile capacities must be nonzero",
            ));
        }
        if profile.max_prompt_tokens > profile.max_sequence_tokens {
            return Err(message(
                "GPT-OSS compilation profile prompt capacity exceeds sequence capacity",
            ));
        }
        if profile.max_sequence_tokens > context_limit {
            return Err(message(
                "GPT-OSS compilation profile exceeds the model context limit",
            ));
        }
        let prefill_capacity = profile
            .max_prompt_tokens
            .max(MIN_PREFILL_BUCKET)
            .checked_next_power_of_two()
            .ok_or_else(|| message("GPT-OSS prefill profile overflows usize"))?;
        let cache_pages = profile
            .max_sequence_tokens
            .div_ceil(CACHE_PAGE_SIZE)
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| message("GPT-OSS cache profile overflows usize"))?;
        let cache_capacity = cache_pages
            .checked_mul(CACHE_PAGE_SIZE)
            .ok_or_else(|| message("GPT-OSS cache profile overflows usize"))?;
        if prefill_capacity > cache_capacity || cache_capacity > context_limit {
            return Err(message(
                "GPT-OSS normalized compilation profile exceeds model capacity",
            ));
        }
        Ok(Self {
            prefill: ShapeFamily::prefill(prefill_capacity, cache_capacity)?,
            decode: ShapeFamily::decode(cache_capacity)?,
        })
    }

    fn supports(self, request: &PreparedRequest) -> bool {
        self.prefill.sequence() >= request.tokens.len()
            && self.decode.cache_capacity() >= request.required_cache_capacity
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct StartupMetrics {
    pub(super) artifact_validation: Duration,
    pub(super) prefill_compilation: Duration,
    pub(super) decode_compilation: Duration,
    pub(super) parameter_upload: Duration,
    pub(super) physical_parameter_components: usize,
    pub(super) parameter_source_bytes: usize,
    pub(super) parameter_resident_bytes: usize,
    pub(super) parameter_prepared_bytes: usize,
    pub(super) parameter_peak_staging_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RunMetrics {
    pub(super) cache_allocation: Duration,
    pub(super) prompt_upload: Duration,
    pub(super) prefill_execution: Duration,
    pub(super) prefill_download: Duration,
    pub(super) decode_state_initialization: Duration,
    pub(super) first_decode_execution: Duration,
    pub(super) steady_decode_execution: Duration,
    pub(super) decode_download: Duration,
    pub(super) prefill_submission: SubmissionTimings,
    pub(super) first_decode_submission: SubmissionTimings,
    pub(super) steady_decode_submission: SubmissionTimings,
}

pub(super) struct RunReport {
    pub(super) generated_tokens: Vec<u32>,
    pub(super) cache_capacity: usize,
    pub(super) cache_storage_bytes: usize,
    pub(super) cache_metadata_bytes: usize,
    pub(super) stopped: bool,
    pub(super) metrics: RunMetrics,
}

/// Artifact-backed model description. Declaring this state opens checkpoint
/// metadata but deliberately allocates no device parameter buffers.
pub(super) struct ModelDefinition {
    config: Config,
    checkpoint: Checkpoint,
    parameter_set: ParameterSet,
    artifact_validation: Duration,
}

impl ModelDefinition {
    pub(super) fn declare(
        config: Config,
        parameter_set: ParameterSet,
        artifact_validation: Duration,
    ) -> Result<Self> {
        let checkpoint = super::checkpoint::declare(&parameter_set, &config)?;
        Ok(Self {
            config,
            checkpoint,
            parameter_set,
            artifact_validation,
        })
    }

    pub(super) fn compile<'platform>(
        self,
        platform: &'platform Platform,
        profiles: &[CompilationProfile],
    ) -> Result<CompiledDefinition<'platform>> {
        let plan = ExecutionPlan::compile(platform, &self.checkpoint, &self.config, profiles)?;
        Ok(CompiledDefinition {
            definition: self,
            plan,
        })
    }
}

/// Type-state transition proving that the complete execution plan exists while
/// the checkpoint is still metadata-only.
pub(super) struct CompiledDefinition<'platform> {
    definition: ModelDefinition,
    plan: ExecutionPlan<'platform>,
}

impl<'platform> CompiledDefinition<'platform> {
    pub(super) fn make_resident(self) -> Result<ResidentModel<'platform>> {
        let Self { definition, plan } = self;
        let ModelDefinition {
            config,
            checkpoint,
            parameter_set,
            artifact_validation,
        } = definition;
        let parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(4);
        let load_options = LoadOptions::new(plan.placement.clone())
            .parallelism(parallelism)
            .map_err(boxed)?;
        let started = Instant::now();
        let (parameters, accounting) = parameter_set
            .load_accounted(&checkpoint, plan.platform, &load_options)
            .map_err(boxed)?;
        let parameter_upload = started.elapsed();
        let startup = startup_metrics(
            artifact_validation,
            plan.prefill_compilation,
            plan.decode_compilation,
            parameter_upload,
            accounting,
        );
        Ok(ResidentModel {
            config,
            checkpoint,
            parameters,
            plan,
            startup,
        })
    }
}

/// Complete bounded executable plan. Construction compiles every configured
/// family while no model parameter buffers are resident.
struct ExecutionPlan<'platform> {
    platform: &'platform Platform,
    placement: Sharding,
    profiles: Vec<ExecutionProfile>,
    families: BTreeMap<ShapeFamily, ComponentFamily>,
    prefill_compilation: Duration,
    decode_compilation: Duration,
}

impl<'platform> ExecutionPlan<'platform> {
    fn compile(
        platform: &'platform Platform,
        checkpoint: &Checkpoint,
        config: &Config,
        requested_profiles: &[CompilationProfile],
    ) -> Result<Self> {
        let profiles = normalize_profiles(requested_profiles, config.context_limit())?;
        let families = profiles
            .iter()
            .flat_map(|profile| [profile.prefill, profile.decode])
            .collect::<BTreeSet<_>>();
        let placement = Sharding::single();
        let mut compiled = BTreeMap::new();
        let mut prefill_compilation = Duration::ZERO;
        let mut decode_compilation = Duration::ZERO;
        for family in families {
            let started = Instant::now();
            let components =
                ComponentFamily::compile(platform, &placement, checkpoint, config, family)?;
            match family.phase() {
                Phase::Prefill => prefill_compilation += started.elapsed(),
                Phase::Decode => decode_compilation += started.elapsed(),
            }
            compiled.insert(family, components);
        }
        Ok(Self {
            platform,
            placement,
            profiles: profiles.into_iter().collect(),
            families: compiled,
            prefill_compilation,
            decode_compilation,
        })
    }

    fn select(&self, request: &PreparedRequest) -> Result<ExecutionProfile> {
        select_profile(&self.profiles, request)
    }
}

fn normalize_profiles(
    requested: &[CompilationProfile],
    context_limit: usize,
) -> Result<BTreeSet<ExecutionProfile>> {
    if requested.is_empty() {
        return Err(message("GPT-OSS requires at least one compilation profile"));
    }
    requested
        .iter()
        .copied()
        .map(|profile| ExecutionProfile::new(profile, context_limit))
        .collect()
}

fn select_profile(
    profiles: &[ExecutionProfile],
    request: &PreparedRequest,
) -> Result<ExecutionProfile> {
    profiles
        .iter()
        .copied()
        .filter(|profile| profile.supports(request))
        .min_by_key(|profile| (profile.decode.cache_capacity(), profile.prefill.sequence()))
        .ok_or_else(|| message("GPT-OSS request is not covered by a compiled execution profile"))
}

/// Process-resident model. Its execution plan is complete before the first
/// checkpoint component is uploaded, and request execution cannot mutate it.
pub(super) struct ResidentModel<'platform> {
    config: Config,
    checkpoint: Checkpoint,
    parameters: LoadedCheckpoint,
    plan: ExecutionPlan<'platform>,
    startup: StartupMetrics,
}

impl ResidentModel<'_> {
    pub(super) const fn config(&self) -> &Config {
        &self.config
    }

    pub(super) const fn startup(&self) -> StartupMetrics {
        self.startup
    }

    pub(super) fn generate(
        &self,
        request: &PreparedRequest,
        mut emit: impl FnMut(u32, bool) -> Result<()>,
    ) -> Result<RunReport> {
        let profile = self.plan.select(request)?;
        let prefill = profile.prefill;
        let decode = profile.decode;
        let cache_capacity = decode.cache_capacity();
        if request.max_new_tokens == 0 {
            return Ok(RunReport {
                generated_tokens: Vec::new(),
                cache_capacity,
                cache_storage_bytes: 0,
                cache_metadata_bytes: 0,
                stopped: false,
                metrics: RunMetrics::default(),
            });
        }
        let platform = self.plan.platform;
        let placement = &self.plan.placement;
        let config = &self.config;
        let checkpoint = &self.checkpoint;
        let parameters = &self.parameters;
        let prefill_executables = self
            .plan
            .families
            .get(&prefill)
            .ok_or_else(|| message("prefill executable family was not retained"))?;
        let decode_executables = self
            .plan
            .families
            .get(&decode)
            .ok_or_else(|| message("decode executable family was not retained"))?;

        let mut prefill_embedding = prefill_executables.embedding.args();
        bind_embedding(&mut prefill_embedding, checkpoint, parameters)?;
        let mut prefill_layers = bind_layers(prefill_executables, checkpoint, parameters, config)?;
        let mut prefill_head = prefill_executables.head.args();
        bind_head(&mut prefill_head, parameters)?;

        let mut decode_embedding = decode_executables.embedding.args();
        bind_embedding(&mut decode_embedding, checkpoint, parameters)?;
        let mut decode_pairs =
            bind_decode_pairs(decode_executables, checkpoint, parameters, config)?;
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
        let mut padded = vec![0_i32; prefill.sequence()];
        for (destination, token) in padded.iter_mut().zip(&request.tokens) {
            *destination = i32::try_from(*token)
                .map_err(|_| message("GPT-OSS token exceeds the I32 graph domain"))?;
        }
        let prompt = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1, usize_i64(prefill.sequence())?]).map_err(boxed)?,
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
        let sampling_state = upload_u64(
            platform,
            Shape::new(nml::DataType::U64, &[2]).map_err(boxed)?,
            &request.sampling.seed,
            placement,
        )?;
        let top_k = upload_scalar(
            platform,
            i32::try_from(request.sampling.top_k)
                .map_err(|_| message("GPT-OSS top-k exceeds I32"))?,
            placement,
        )?;
        let temperature = upload_f32_scalar(platform, request.sampling.temperature, placement)?;
        let top_p = upload_f32_scalar(platform, request.sampling.top_p, placement)?;
        let min_p = upload_f32_scalar(platform, request.sampling.min_p, placement)?;
        for head in [&mut prefill_head, &mut decode_head] {
            head.set("top_k", top_k.clone()).map_err(boxed)?;
            head.set("temperature", temperature.clone())
                .map_err(boxed)?;
            head.set("top_p", top_p.clone()).map_err(boxed)?;
            head.set("min_p", min_p.clone()).map_err(boxed)?;
        }
        let prompt_upload = prompt_upload_started.elapsed();

        let prefill_started = Instant::now();
        let mut prefill_submission = SubmissionTimings::default();
        let submission_started = Instant::now();
        prefill_embedding.set("tokens", prompt).map_err(boxed)?;
        let mut hidden = one(prefill_embedding.enqueue().map_err(boxed)?)?;
        prefill_submission.embedding = submission_started.elapsed();
        for ((arguments, cache), kind) in prefill_layers
            .iter_mut()
            .zip(&mut caches)
            .zip(config.layer_types())
        {
            let (next_hidden, elapsed) = execute_layer(
                arguments,
                hidden,
                prefill_position.clone(),
                Some(sequence_lengths.clone()),
                page_table.clone(),
                cache,
            )?;
            hidden = next_hidden;
            record_layer_submission(&mut prefill_submission, *kind, elapsed);
        }
        let submission_started = Instant::now();
        prefill_head.set("hidden", hidden).map_err(boxed)?;
        prefill_head.set("last_index", last_index).map_err(boxed)?;
        prefill_head
            .set("sampling_state", sampling_state)
            .map_err(boxed)?;
        let mut prefill_outputs = prefill_head
            .enqueue()
            .map_err(boxed)?
            .into_buffers()
            .into_iter();
        let mut token_buffer = prefill_outputs
            .next()
            .ok_or_else(|| message("GPT-OSS prefill head omitted its token"))?;
        let mut sampling_state = prefill_outputs
            .next()
            .ok_or_else(|| message("GPT-OSS prefill head omitted its sampling state"))?;
        if prefill_outputs.next().is_some() {
            return Err(message("GPT-OSS prefill head returned extra buffers"));
        }
        prefill_submission.head = submission_started.elapsed();
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
        let mut first_decode_submission = SubmissionTimings::default();
        let mut steady_decode_submission = SubmissionTimings::default();
        let (first_decode_pair, remaining_decode_pairs) = decode_pairs
            .split_first_mut()
            .ok_or_else(|| message("GPT-OSS decode schedule contains no layer pair"))?;
        if caches.len() < 2 {
            return Err(message(
                "GPT-OSS decode schedule contains no first cache pair",
            ));
        }
        let (first_decode_caches, remaining_decode_caches) = caches.split_at_mut(2);
        let mut lookahead = None;
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
            let prefix = match lookahead.take() {
                Some(prefix) => prefix,
                None => enqueue_decode_prefix(
                    &mut decode_embedding,
                    first_decode_pair,
                    token_buffer.clone(),
                    position.clone(),
                    page_table.clone(),
                    first_decode_caches,
                )?,
            };
            let mut hidden = prefix.hidden;
            let mut decode_submission = prefix.submission;
            for (arguments, caches) in remaining_decode_pairs
                .iter_mut()
                .zip(remaining_decode_caches.chunks_exact_mut(2))
            {
                let (next_hidden, elapsed) = execute_layer_pair(
                    arguments,
                    hidden,
                    position.clone(),
                    page_table.clone(),
                    caches,
                )?;
                hidden = next_hidden;
                decode_submission.layer_pairs += elapsed;
            }
            let submission_started = Instant::now();
            decode_head.set("hidden", hidden).map_err(boxed)?;
            decode_head
                .set("last_index", decode_last_index.clone())
                .map_err(boxed)?;
            decode_head
                .set("sampling_state", sampling_state)
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
            sampling_state = outputs
                .next()
                .ok_or_else(|| message("GPT-OSS decode head omitted its sampling state"))?;
            position = outputs
                .next()
                .ok_or_else(|| message("GPT-OSS decode head omitted its position"))?;
            if outputs.next().is_some() {
                return Err(message("GPT-OSS decode head returned extra buffers"));
            }
            decode_submission.head = submission_started.elapsed();

            // The head outputs carry readiness dependencies, so the next token
            // can enter embedding and the first bounded layer pair before the
            // host observes it. This queues enough useful work to cover both
            // the head-to-embedding and embedding-to-first-pair bubbles while
            // limiting terminal speculation to two request-local layer caches.
            // Keep one token of budget beyond the speculative input: otherwise
            // the lookahead would compute a token the caller never requested.
            if should_enqueue_lookahead(generated_index, request.max_new_tokens) {
                lookahead = Some(enqueue_decode_prefix(
                    &mut decode_embedding,
                    first_decode_pair,
                    token_buffer.clone(),
                    position.clone(),
                    page_table.clone(),
                    first_decode_caches,
                )?);
            }
            token_buffer.wait().map_err(boxed)?;
            let execution_elapsed = decode_started.elapsed();
            if generated_index == 0 {
                first_decode_execution = execution_elapsed;
                first_decode_submission = decode_submission;
            } else {
                steady_decode_execution += execution_elapsed;
                add_submission(&mut steady_decode_submission, decode_submission);
            }
            let download_started = Instant::now();
            next_token = download_token(&token_buffer)?;
            decode_download += download_started.elapsed();
        }

        Ok(RunReport {
            generated_tokens,
            cache_capacity,
            cache_storage_bytes,
            cache_metadata_bytes,
            stopped,
            metrics: RunMetrics {
                cache_allocation,
                prompt_upload,
                prefill_execution,
                prefill_download,
                decode_state_initialization,
                first_decode_execution,
                steady_decode_execution,
                decode_download,
                prefill_submission,
                first_decode_submission,
                steady_decode_submission,
            },
        })
    }
}

struct ComponentFamily {
    embedding: Exe,
    layers: LayerExecutables,
    head: Exe,
}

enum LayerExecutables {
    Prefill { sliding: Exe, full: Exe },
    DecodePair(Exe),
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
        let sliding = representative_layer(checkpoint, config, AttentionKind::SlidingAttention)?;
        let full = representative_layer(checkpoint, config, AttentionKind::FullAttention)?;
        let layers = match family.phase() {
            Phase::Prefill => LayerExecutables::Prefill {
                sliding: compile(platform, placement, |graph| {
                    build_layer(
                        graph,
                        sliding,
                        config,
                        family,
                        AttentionKind::SlidingAttention,
                    )
                })?,
                full: compile(platform, placement, |graph| {
                    build_layer(graph, full, config, family, AttentionKind::FullAttention)
                })?,
            },
            Phase::Decode => LayerExecutables::DecodePair(compile(platform, placement, |graph| {
                build_decode_layer_pair(graph, sliding, full, config, family)
            })?),
        };
        let head = compile(platform, placement, |graph| {
            build_head(graph, checkpoint, config, family)
        })?;
        Ok(Self {
            embedding,
            layers,
            head,
        })
    }

    fn prefill_layer(&self, kind: AttentionKind) -> Result<&Exe> {
        let LayerExecutables::Prefill { sliding, full } = &self.layers else {
            return Err(message(
                "decode family does not contain single-layer executables",
            ));
        };
        Ok(match kind {
            AttentionKind::SlidingAttention => sliding,
            AttentionKind::FullAttention => full,
        })
    }

    fn decode_pair(&self) -> Result<&Exe> {
        match &self.layers {
            LayerExecutables::DecodePair(executable) => Ok(executable),
            LayerExecutables::Prefill { .. } => {
                Err(message("prefill family does not contain a decode pair"))
            }
        }
    }
}

struct LayerCache {
    key: Option<Buffer>,
    value: Option<Buffer>,
}

struct DecodePrefix {
    hidden: Buffer,
    submission: SubmissionTimings,
}

impl LayerCache {
    fn allocate(platform: &Platform, shape: Shape, placement: &Sharding) -> Result<Self> {
        Ok(Self {
            key: Some(
                platform
                    .upload(
                        &Slice::alloc(shape).map_err(boxed)?,
                        placement.clone(),
                        Memory::Default,
                    )
                    .map_err(boxed)?,
            ),
            value: Some(
                platform
                    .upload(
                        &Slice::alloc(shape).map_err(boxed)?,
                        placement.clone(),
                        Memory::Default,
                    )
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

fn bind_head(arguments: &mut Arguments<'_>, parameters: &LoadedCheckpoint) -> Result<()> {
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
            let executable = family.prefill_layer(*kind)?;
            let slots = representative_layer(checkpoint, config, *kind)?;
            bind_layer(executable, slots, loaded)
        })
        .collect()
}

fn bind_decode_pairs<'family>(
    family: &'family ComponentFamily,
    checkpoint: &Checkpoint,
    parameters: &LoadedCheckpoint,
    config: &Config,
) -> Result<Vec<Arguments<'family>>> {
    if checkpoint.model.layers.len() != parameters.model.layers.len()
        || checkpoint.model.layers.len() != config.layer_types().len()
        || !checkpoint.model.layers.len().is_multiple_of(2)
    {
        return Err(message(
            "GPT-OSS decode pairing requires one complete alternating layer schedule",
        ));
    }
    let slots = checkpoint.model.layers.get(0..2).ok_or_else(|| {
        message("GPT-OSS decode pairing requires sliding and full representatives")
    })?;
    parameters
        .model
        .layers
        .chunks_exact(2)
        .enumerate()
        .map(|(pair, loaded)| {
            let first = pair * 2;
            if config.layer_types()[first] != AttentionKind::SlidingAttention
                || config.layer_types()[first + 1] != AttentionKind::FullAttention
            {
                return Err(message(
                    "GPT-OSS decode pair violates the alternating schedule",
                ));
            }
            bind_layer_pair(family.decode_pair()?, slots, loaded)
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

fn bind_layer_pair<'family>(
    executable: &'family Exe,
    slots: &[super::checkpoint::DecoderLayer],
    loaded: &[LoadedDecoderLayer],
) -> Result<Arguments<'family>> {
    let [sliding_slot, full_slot] = slots else {
        return Err(message("GPT-OSS decode pair slot count is not two"));
    };
    let [sliding_loaded, full_loaded] = loaded else {
        return Err(message("GPT-OSS decode pair parameter count is not two"));
    };
    let mut arguments = executable.args();
    bind_tree_components(&mut arguments, sliding_slot, sliding_loaded)?;
    bind_tree_components(&mut arguments, full_slot, full_loaded)?;
    arguments.bake().map_err(boxed)?;
    Ok(arguments)
}

fn enqueue_decode_prefix(
    embedding: &mut Arguments<'_>,
    first_pair: &mut Arguments<'_>,
    token: Buffer,
    position: Buffer,
    page_table: Buffer,
    caches: &mut [LayerCache],
) -> Result<DecodePrefix> {
    let mut submission = SubmissionTimings::default();
    let started = Instant::now();
    embedding.set("tokens", token).map_err(boxed)?;
    let hidden = one(embedding.enqueue().map_err(boxed)?)?;
    submission.embedding = started.elapsed();
    let (hidden, elapsed) =
        execute_layer_pair(first_pair, hidden, position, page_table, caches)?;
    submission.layer_pairs = elapsed;
    Ok(DecodePrefix { hidden, submission })
}

const fn should_enqueue_lookahead(generated_index: usize, max_new_tokens: usize) -> bool {
    max_new_tokens.saturating_sub(generated_index) > 2
}

fn execute_layer(
    arguments: &mut Arguments<'_>,
    hidden: Buffer,
    position: Buffer,
    sequence_lengths: Option<Buffer>,
    page_table: Buffer,
    cache: &mut LayerCache,
) -> Result<(Buffer, Duration)> {
    let started = Instant::now();
    let (key, value) = cache.take()?;
    arguments.set("hidden", hidden).map_err(boxed)?;
    arguments.set("position", position).map_err(boxed)?;
    if let Some(lengths) = sequence_lengths {
        arguments.set("sequence_lengths", lengths).map_err(boxed)?;
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
    Ok((hidden, started.elapsed()))
}

fn execute_layer_pair(
    arguments: &mut Arguments<'_>,
    hidden: Buffer,
    position: Buffer,
    page_table: Buffer,
    caches: &mut [LayerCache],
) -> Result<(Buffer, Duration)> {
    let [sliding, full] = caches else {
        return Err(message(
            "GPT-OSS decode execution requires exactly two caches",
        ));
    };
    let started = Instant::now();
    let (sliding_key, sliding_value) = sliding.take()?;
    let (full_key, full_value) = full.take()?;
    arguments.set("hidden", hidden).map_err(boxed)?;
    arguments.set("position", position).map_err(boxed)?;
    arguments.set("page_table", page_table).map_err(boxed)?;
    arguments
        .set("sliding.cache.key", sliding_key)
        .map_err(boxed)?;
    arguments
        .set("sliding.cache.value", sliding_value)
        .map_err(boxed)?;
    arguments.set("full.cache.key", full_key).map_err(boxed)?;
    arguments
        .set("full.cache.value", full_value)
        .map_err(boxed)?;
    let mut outputs = arguments
        .enqueue()
        .map_err(boxed)?
        .into_buffers()
        .into_iter();
    let hidden = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer pair omitted hidden state"))?;
    let sliding_key = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer pair omitted sliding key cache"))?;
    let sliding_value = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer pair omitted sliding value cache"))?;
    let full_key = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer pair omitted full key cache"))?;
    let full_value = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS layer pair omitted full value cache"))?;
    if outputs.next().is_some() {
        return Err(message("GPT-OSS layer pair returned extra buffers"));
    }
    sliding.install(sliding_key, sliding_value)?;
    full.install(full_key, full_value)?;
    Ok((hidden, started.elapsed()))
}

fn record_layer_submission(
    timings: &mut SubmissionTimings,
    kind: AttentionKind,
    elapsed: Duration,
) {
    match kind {
        AttentionKind::SlidingAttention => timings.sliding_layers += elapsed,
        AttentionKind::FullAttention => timings.full_layers += elapsed,
    }
}

fn add_submission(total: &mut SubmissionTimings, value: SubmissionTimings) {
    total.embedding += value.embedding;
    total.sliding_layers += value.sliding_layers;
    total.full_layers += value.full_layers;
    total.layer_pairs += value.layer_pairs;
    total.head += value.head;
}

fn compile(
    platform: &Platform,
    placement: &Sharding,
    build: impl FnOnce(&mut Graph) -> Result<Vec<(String, nml::Tensor)>>,
) -> Result<Exe> {
    let mut graph = Graph::new();
    let outputs = build(&mut graph)?;
    let program = graph.finish_named(&outputs).map_err(boxed)?;
    platform.compile(&program, placement.clone()).map_err(boxed)
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

fn upload_u64(
    platform: &Platform,
    shape: Shape,
    values: &[u64],
    placement: &Sharding,
) -> Result<Buffer> {
    let slice = Slice::from_typed(shape, values).map_err(boxed)?;
    platform
        .upload(&slice, placement.clone(), Memory::Default)
        .map_err(boxed)
}

fn upload_f32_scalar(platform: &Platform, value: f32, placement: &Sharding) -> Result<Buffer> {
    let shape = Shape::new(nml::DataType::F32, &[]).map_err(boxed)?;
    let values = [value];
    let slice = Slice::from_typed(shape, &values).map_err(boxed)?;
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
        return Err(message(
            "GPT-OSS component returned an invalid result count",
        ));
    }
    Ok(buffers.remove(0))
}

fn startup_metrics(
    artifact_validation: Duration,
    prefill_compilation: Duration,
    decode_compilation: Duration,
    parameter_upload: Duration,
    accounting: LoadAccounting,
) -> StartupMetrics {
    StartupMetrics {
        artifact_validation,
        prefill_compilation,
        decode_compilation,
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
    fn decode_lookahead_never_crosses_the_visible_token_budget() {
        assert!(!should_enqueue_lookahead(0, 0));
        assert!(!should_enqueue_lookahead(0, 1));
        assert!(!should_enqueue_lookahead(0, 2));
        assert!(should_enqueue_lookahead(0, 3));
        assert!(!should_enqueue_lookahead(1, 3));
        assert!(should_enqueue_lookahead(0, 4));
        assert!(should_enqueue_lookahead(1, 4));
        assert!(!should_enqueue_lookahead(2, 4));
    }

    #[test]
    fn profiles_normalize_and_select_the_smallest_fitting_family() {
        let request = PreparedRequest::new(
            vec![7; 17],
            5,
            None,
            SamplingOptions::default(),
            131_072,
            Duration::from_millis(1),
        )
        .unwrap();
        let small = ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 17,
                max_sequence_tokens: 22,
            },
            131_072,
        )
        .unwrap();
        let large = ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 65,
                max_sequence_tokens: 300,
            },
            131_072,
        )
        .unwrap();
        assert_eq!(small.prefill.sequence(), 32);
        assert_eq!(small.decode.cache_capacity(), 2 * CACHE_PAGE_SIZE);
        assert_eq!(large.prefill.sequence(), 128);
        assert_eq!(large.decode.cache_capacity(), 32 * CACHE_PAGE_SIZE);
        assert_eq!(select_profile(&[large, small], &request).unwrap(), small);
    }

    #[test]
    fn equivalent_profiles_deduplicate_in_stable_capacity_order() {
        let profiles = normalize_profiles(
            &[
                CompilationProfile {
                    max_prompt_tokens: 17,
                    max_sequence_tokens: 257,
                },
                CompilationProfile {
                    max_prompt_tokens: 31,
                    max_sequence_tokens: 300,
                },
                CompilationProfile {
                    max_prompt_tokens: 65,
                    max_sequence_tokens: 513,
                },
            ],
            131_072,
        )
        .unwrap()
        .into_iter()
        .collect::<Vec<_>>();
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].prefill.sequence(), 32);
        assert_eq!(profiles[0].decode.cache_capacity(), 512);
        assert_eq!(profiles[1].prefill.sequence(), 128);
        assert_eq!(profiles[1].decode.cache_capacity(), 1_024);
    }

    #[test]
    fn requests_reject_unrepresentable_or_undersized_families() {
        assert!(normalize_profiles(&[], 131_072).is_err());
        assert!(PreparedRequest::new(
            vec![],
            1,
            None,
            SamplingOptions::default(),
            131_072,
            Duration::ZERO,
        )
        .is_err());
        for sampling in [
            SamplingOptions {
                top_k: 0,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_k: MAXIMUM_TOP_K + 1,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                temperature: 0.0,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                temperature: f32::NAN,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_p: 0.0,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                top_p: 1.01,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                min_p: -0.01,
                ..SamplingOptions::default()
            },
            SamplingOptions {
                min_p: 1.01,
                ..SamplingOptions::default()
            },
        ] {
            assert!(
                PreparedRequest::new(vec![1], 1, None, sampling, 131_072, Duration::ZERO,).is_err()
            );
        }
        assert!(PreparedRequest::new(
            vec![1; 17],
            5,
            Some(21),
            SamplingOptions::default(),
            131_072,
            Duration::ZERO,
        )
        .is_err());
        assert!(PreparedRequest::new(
            vec![1; 17],
            5,
            Some(131_073),
            SamplingOptions::default(),
            131_072,
            Duration::ZERO,
        )
        .is_err());
        assert!(ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 513,
                max_sequence_tokens: 512,
            },
            131_072,
        )
        .is_err());
        assert!(ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 16,
                max_sequence_tokens: 131_073,
            },
            131_072,
        )
        .is_err());
    }

    #[test]
    fn requests_outside_the_compiled_profiles_fail_instead_of_compiling() {
        let profile = ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 32,
                max_sequence_tokens: 256,
            },
            131_072,
        )
        .unwrap();
        let oversized_prompt = PreparedRequest::new(
            vec![1; 33],
            1,
            None,
            SamplingOptions::default(),
            131_072,
            Duration::ZERO,
        )
        .unwrap();
        assert!(select_profile(&[profile], &oversized_prompt).is_err());
        let oversized_sequence = PreparedRequest::new(
            vec![1; 16],
            241,
            None,
            SamplingOptions::default(),
            131_072,
            Duration::ZERO,
        )
        .unwrap();
        assert!(select_profile(&[profile], &oversized_sequence).is_err());
    }
}
