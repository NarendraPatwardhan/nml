//! Definition, compilation, residency, and request-local execution state.

use super::checkpoint::{
    bind_tree, bind_tree_components, message, representative_layer, BoxError, Checkpoint,
    LoadedCheckpoint, LoadedDecoderLayer, Result,
};
use super::config::{AttentionKind, Config};
use super::graph::{
    build_decode_layer_pair, build_embedding, build_head, build_layer, cache_shape,
    page_table_shape, Phase, ShapeFamily, BATCH_RESULT_BYTES_PER_ROW, CACHE_PAGE_SIZE,
    ServingSlabLayout, MAXIMUM_TOP_K,
};
use crate::{CompilationProfile, SamplingOptions, SubmissionTimings};
use nml::exe::{Arguments, Results};
use nml::io::{LoadAccounting, LoadOptions, ParameterSet};
use nml::{Buffer, Exe, Graph, Memory, Platform, Shape, Sharding, Slice};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

const MIN_PREFILL_BUCKET: usize = 16;

#[derive(Clone, Debug)]
pub(super) struct ServingCompileConfig {
    pub(super) batch_buckets: Vec<usize>,
    pub(super) prefill_query_buckets: Vec<usize>,
    pub(super) logical_cache_capacity: usize,
    pub(super) tensor_parallel: usize,
}

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
    fn new(
        profile: CompilationProfile,
        context_limit: usize,
        physical_pages: usize,
    ) -> Result<Self> {
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
            prefill: ShapeFamily::prefill(prefill_capacity, cache_capacity, physical_pages)?,
            decode: ShapeFamily::decode(cache_capacity, physical_pages)?,
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
    pub(super) compiled_families: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RunMetrics {
    pub(super) cache_allocation: Duration,
    pub(super) cache_metadata_upload: Duration,
    pub(super) cache_metadata_upload_bytes: usize,
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
    pub(super) cache_metadata_upload_bytes: usize,
    pub(super) metrics: RunMetrics,
}

pub(super) struct BatchInputs {
    pub(super) tokens: Vec<i32>,
    pub(super) positions: Vec<i32>,
    pub(super) sequence_lengths: Vec<i32>,
    pub(super) query_lengths: Vec<i32>,
    pub(super) active_rows: Vec<bool>,
    pub(super) sample_rows: Vec<bool>,
    pub(super) page_tables: Vec<i32>,
    pub(super) last_indices: Vec<i32>,
    pub(super) sampling_states: Vec<u64>,
    pub(super) top_k: Vec<i32>,
    pub(super) temperature: Vec<f32>,
    pub(super) top_p: Vec<f32>,
    pub(super) min_p: Vec<f32>,
}

pub(super) struct BatchOutputs {
    pub(super) tokens: Vec<i32>,
    pub(super) sampling_states: Vec<u64>,
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
        physical_pages: usize,
        serving: Option<&ServingCompileConfig>,
    ) -> Result<CompiledDefinition<'platform>> {
        let plan = ExecutionPlan::compile(
            platform,
            &self.checkpoint,
            &self.config,
            profiles,
            physical_pages,
            serving,
        )?;
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
            plan.families.len(),
        );
        let bound_families = plan
            .families
            .iter()
            .map(|(shape, family)| {
                BoundComponentFamily::bind(family, &checkpoint, &parameters, &config)
                    .map(|bound| (*shape, bound))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let allocation_started = Instant::now();
        let family = plan
            .families
            .keys()
            .next()
            .copied()
            .ok_or_else(|| message("GPT-OSS execution plan has no cache family"))?;
        let cache_tensor_shape = cache_shape(&config, family)?;
        let caches = (0..config.layers())
            .map(|_| LayerCache::allocate(plan.platform, cache_tensor_shape, &plan.placement))
            .collect::<Result<Vec<_>>>()?;
        let cache_allocation = allocation_started.elapsed();
        let cache_storage_bytes = cache_tensor_shape
            .byte_count()
            .map_err(boxed)?
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_mul(caches.len()))
            .ok_or_else(|| message("GPT-OSS cache storage accounting overflows usize"))?;
        Ok(ResidentModel {
            config,
            checkpoint,
            parameters,
            plan,
            startup,
            bound_families,
            caches,
            cache_allocation,
            cache_storage_bytes,
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
        physical_pages: usize,
        serving: Option<&ServingCompileConfig>,
    ) -> Result<Self> {
        let profiles = normalize_profiles(
            requested_profiles,
            config.context_limit(),
            physical_pages,
        )?;
        let mut families = profiles
            .iter()
            .flat_map(|profile| [profile.prefill, profile.decode])
            .collect::<BTreeSet<_>>();
        if let Some(serving) = serving {
            let cache_capacity = serving
                .logical_cache_capacity
                .div_ceil(CACHE_PAGE_SIZE)
                .checked_mul(CACHE_PAGE_SIZE)
                .ok_or_else(|| message("serving cache capacity overflows usize"))?;
            if cache_capacity > config.context_limit() {
                return Err(message(
                    "normalized serving cache capacity exceeds the model context limit",
                ));
            }
            for batch in &serving.batch_buckets {
                families.insert(ShapeFamily::serving_decode(
                    *batch,
                    cache_capacity,
                    physical_pages,
                    serving.tensor_parallel,
                )?);
                for query in &serving.prefill_query_buckets {
                    families.insert(ShapeFamily::serving_prefill(
                        *batch,
                        *query,
                        cache_capacity,
                        physical_pages,
                        serving.tensor_parallel,
                    )?);
                }
            }
        }
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
    physical_pages: usize,
) -> Result<BTreeSet<ExecutionProfile>> {
    if requested.is_empty() {
        return Err(message("GPT-OSS requires at least one compilation profile"));
    }
    requested
        .iter()
        .copied()
        .map(|profile| ExecutionProfile::new(profile, context_limit, physical_pages))
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
    bound_families: BTreeMap<ShapeFamily, BoundComponentFamily>,
    caches: Vec<LayerCache>,
    cache_allocation: Duration,
    cache_storage_bytes: usize,
}

impl ResidentModel<'_> {
    pub(super) const fn config(&self) -> &Config {
        &self.config
    }

    pub(super) const fn startup(&self) -> StartupMetrics {
        self.startup
    }

    pub(super) fn prefill_step(
        &mut self,
        execution: &mut RequestExecution,
    ) -> Result<Option<RawToken>> {
        execution.prefill_step(&mut self.caches)
    }

    pub(super) fn decode_step(
        &mut self,
        execution: &mut RequestExecution,
    ) -> Result<Option<RawToken>> {
        execution.decode_step(&mut self.caches)
    }

    pub(super) fn execute_batch(
        &mut self,
        phase: Phase,
        batch_capacity: usize,
        query_capacity: usize,
        input: BatchInputs,
    ) -> Result<BatchOutputs> {
        let family = self.batch_family(phase, batch_capacity, query_capacity)?;
        validate_batch_inputs(family, &input)?;
        let platform = self.plan.platform;
        let placement = &self.plan.placement;
        let slab_layout = ServingSlabLayout::for_family(family)?;
        let batch_slab = upload_u8(
            platform,
            Shape::new(nml::DataType::U8, &[usize_i64(slab_layout.total_bytes())?])
                .map_err(boxed)?,
            &pack_batch_slab(family, slab_layout, &input)?,
            placement,
        )?;

        let bound = self
            .bound_families
            .get_mut(&family)
            .expect("selected bound family exists");
        bound
            .embedding
            .set("batch_slab", batch_slab.clone())
            .map_err(boxed)?;
        let mut hidden = one(bound.embedding.enqueue().map_err(boxed)?)?;
        match &mut bound.layers {
            BoundLayerExecutables::Prefill { layers, kinds } => {
                for ((arguments, kind), cache) in
                    layers.iter_mut().zip(kinds.iter()).zip(&mut self.caches)
                {
                    let (next, _) = execute_serving_layer(
                        arguments,
                        hidden,
                        batch_slab.clone(),
                        cache,
                    )?;
                    hidden = next;
                    let _ = kind;
                }
            }
            BoundLayerExecutables::Decode { pairs } => {
                for (arguments, caches) in
                    pairs.iter_mut().zip(self.caches.chunks_exact_mut(2))
                {
                    let (next, _) = execute_serving_layer_pair(
                        arguments,
                        hidden,
                        batch_slab.clone(),
                        caches,
                    )?;
                    hidden = next;
                }
            }
        }
        bound.head.set("hidden", hidden).map_err(boxed)?;
        bound
            .head
            .set("batch_slab", batch_slab)
            .map_err(boxed)?;
        let mut results = bound
            .head
            .enqueue()
            .map_err(boxed)?
            .into_buffers()
            .into_iter();
        let result = results
            .next()
            .ok_or_else(|| message("batched head omitted its compact result"))?;
        if results.next().is_some() {
            return Err(message("batched head returned extra buffers"));
        }
        download_batch_outputs(&result, family.batch())
    }

    pub(super) fn batch_page_table_width(
        &self,
        phase: Phase,
        batch_capacity: usize,
        query_capacity: usize,
    ) -> Result<usize> {
        Ok(self
            .batch_family(phase, batch_capacity, query_capacity)?
            .page_count())
    }

    fn batch_family(
        &self,
        phase: Phase,
        batch_capacity: usize,
        query_capacity: usize,
    ) -> Result<ShapeFamily> {
        self.bound_families
            .keys()
            .copied()
            .filter(|family| {
                family.is_serving()
                    && family.phase() == phase
                    && family.batch() == batch_capacity
                    && family.sequence() == query_capacity
            })
            .min_by_key(|family| family.cache_capacity())
            .ok_or_else(|| message("requested batch is not covered by a compiled serving family"))
    }

    pub(super) fn install_page_table(
        &self,
        execution: &mut RequestExecution,
        pages: &[i32],
    ) -> Result<()> {
        if pages.len() != execution.page_table_width() {
            return Err(message(
                "GPT-OSS page table does not match the selected execution family",
            ));
        }
        let started = Instant::now();
        execution.page_table = Some(upload_i32(
            self.plan.platform,
            page_table_shape(execution.profile.prefill)?,
            pages,
            &self.plan.placement,
        )?);
        execution.metrics.cache_metadata_upload += started.elapsed();
        execution.metrics.cache_metadata_upload_bytes = execution
            .metrics
            .cache_metadata_upload_bytes
            .checked_add(
                pages
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| message("GPT-OSS metadata byte count overflows usize"))?,
            )
            .ok_or_else(|| message("GPT-OSS metadata byte count overflows usize"))?;
        Ok(())
    }

    /// Selects one already-compiled family and creates all request-local
    /// execution state without enqueueing a model graph. The returned request
    /// owns its prompt, sampling, K/V, and executable argument state.
    pub(super) fn prepare(&self, request: PreparedRequest) -> Result<RequestExecution> {
        let profile = self.plan.select(&request)?;
        let prefill = profile.prefill;
        let decode = profile.decode;
        let cache_capacity = decode.cache_capacity();
        if request.max_new_tokens == 0 {
            return Ok(RequestExecution::completed(
                profile,
                request.tokens.len(),
                self.cache_allocation,
                self.cache_storage_bytes,
            ));
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
        let prefill_layers = bind_layers(prefill_executables, checkpoint, parameters, config)?;
        let mut prefill_head = prefill_executables.head.args();
        bind_head(&mut prefill_head, parameters)?;

        let mut decode_embedding = decode_executables.embedding.args();
        bind_embedding(&mut decode_embedding, checkpoint, parameters)?;
        let decode_pairs =
            bind_decode_pairs(decode_executables, checkpoint, parameters, config)?;
        let mut decode_head = decode_executables.head.args();
        bind_head(&mut decode_head, parameters)?;

        let cache_metadata_bytes = page_table_shape(prefill)?
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
        let prefill_position = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1]).map_err(boxed)?,
            &[0],
            placement,
        )?;
        let metadata_started = Instant::now();
        let sequence_lengths = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1]).map_err(boxed)?,
            &[i32::try_from(request.tokens.len())
                .map_err(|_| message("GPT-OSS prompt length exceeds I32"))?],
            placement,
        )?;
        let sequence_metadata_upload = metadata_started.elapsed();
        let query_lengths = sequence_lengths.clone();
        let active_rows = upload_bool(
            platform,
            Shape::new(nml::DataType::Bool, &[1]).map_err(boxed)?,
            &[true],
            placement,
        )?;
        let last_index = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1, 1]).map_err(boxed)?,
            &[i32::try_from(request.tokens.len() - 1)
                .map_err(|_| message("GPT-OSS last prompt index exceeds I32"))?],
            placement,
        )?;
        let sampling_state = upload_u64(
            platform,
            Shape::new(nml::DataType::U64, &[1, 2]).map_err(boxed)?,
            &request.sampling.seed,
            placement,
        )?;
        let top_k = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1]).map_err(boxed)?,
            &[i32::try_from(request.sampling.top_k)
                .map_err(|_| message("GPT-OSS top-k exceeds I32"))?],
            placement,
        )?;
        let temperature = upload_f32_vector(platform, &[request.sampling.temperature], placement)?;
        let top_p = upload_f32_vector(platform, &[request.sampling.top_p], placement)?;
        let min_p = upload_f32_vector(platform, &[request.sampling.min_p], placement)?;
        for head in [&mut prefill_head, &mut decode_head] {
            head.set("top_k", top_k.clone()).map_err(boxed)?;
            head.set("temperature", temperature.clone())
                .map_err(boxed)?;
            head.set("top_p", top_p.clone()).map_err(boxed)?;
            head.set("min_p", min_p.clone()).map_err(boxed)?;
            head.set("active_rows", active_rows.clone()).map_err(boxed)?;
        }
        let prompt_upload = prompt_upload_started.elapsed();

        let decode_state_started = Instant::now();
        let position = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1]).map_err(boxed)?,
            &[i32::try_from(request.tokens.len())
                .map_err(|_| message("GPT-OSS decode position exceeds I32"))?],
            placement,
        )?;
        let decode_last_index = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1, 1]).map_err(boxed)?,
            &[0],
            placement,
        )?;
        let decode_query_lengths = upload_i32(
            platform,
            Shape::new(nml::DataType::I32, &[1]).map_err(boxed)?,
            &[1],
            placement,
        )?;
        let decode_state_initialization = decode_state_started.elapsed();

        if decode_pairs.len() < DECODE_LOOKAHEAD_PAIRS {
            return Err(message(
                "GPT-OSS decode schedule is shorter than the lookahead prefix",
            ));
        }
        let lookahead_cache_count = DECODE_LOOKAHEAD_PAIRS * 2;
        if config.layers() < lookahead_cache_count {
            return Err(message(
                "GPT-OSS decode cache schedule is shorter than the lookahead prefix",
            ));
        }
        Ok(RequestExecution {
            lifecycle: RequestLifecycle::new(request.max_new_tokens),
            profile,
            prompt_token_count: request.tokens.len(),
            cache_capacity,
            cache_storage_bytes: self.cache_storage_bytes,
            cache_metadata_bytes,
            metrics: RunMetrics {
                cache_allocation: self.cache_allocation,
                cache_metadata_upload: sequence_metadata_upload,
                cache_metadata_upload_bytes: std::mem::size_of::<i32>(),
                prompt_upload,
                prefill_execution: Duration::ZERO,
                prefill_download: Duration::ZERO,
                decode_state_initialization,
                first_decode_execution: Duration::ZERO,
                steady_decode_execution: Duration::ZERO,
                decode_download: Duration::ZERO,
                prefill_submission: SubmissionTimings::default(),
                first_decode_submission: SubmissionTimings::default(),
                steady_decode_submission: SubmissionTimings::default(),
            },
            page_table: None,
            prefill: Some(PrefillState {
                embedding: prefill_embedding,
                layers: prefill_layers,
                layer_types: config.layer_types().to_vec(),
                head: prefill_head,
                prompt,
                position: prefill_position,
                sequence_lengths,
                query_lengths,
                active_rows: active_rows.clone(),
                last_index,
                sampling_state,
                decode: PreparedDecodeState {
                    embedding: decode_embedding,
                    pairs: decode_pairs,
                    head: decode_head,
                    position,
                    query_lengths: decode_query_lengths,
                    active_rows,
                    last_index: decode_last_index,
                },
            }),
            decode: None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestPhase {
    PrefillPending,
    DecodeReady,
    Complete,
    Failed,
}

/// Pure visible-token state. Keeping this separate from device buffers makes
/// the prefill/decode/budget contract independently testable.
#[derive(Debug)]
struct RequestLifecycle {
    phase: RequestPhase,
    max_new_tokens: usize,
    generated_tokens: Vec<u32>,
    stopped: bool,
}

impl RequestLifecycle {
    fn new(max_new_tokens: usize) -> Self {
        Self {
            phase: if max_new_tokens == 0 {
                RequestPhase::Complete
            } else {
                RequestPhase::PrefillPending
            },
            max_new_tokens,
            generated_tokens: Vec::with_capacity(max_new_tokens),
            stopped: false,
        }
    }

    fn record_prefill(&mut self, token: u32, is_stop: bool) -> Result<RawToken> {
        if self.phase != RequestPhase::PrefillPending {
            return Err(message("GPT-OSS prefill step is not pending"));
        }
        self.record(token, is_stop)
    }

    fn record_decode(&mut self, token: u32, is_stop: bool) -> Result<RawToken> {
        if self.phase != RequestPhase::DecodeReady {
            return Err(message("GPT-OSS decode step is not ready"));
        }
        self.record(token, is_stop)
    }

    fn record(&mut self, token: u32, is_stop: bool) -> Result<RawToken> {
        if self.generated_tokens.len() >= self.max_new_tokens {
            return Err(message("GPT-OSS visible token budget is exhausted"));
        }
        self.generated_tokens.push(token);
        self.stopped = is_stop;
        self.phase = if is_stop || self.generated_tokens.len() == self.max_new_tokens {
            RequestPhase::Complete
        } else {
            RequestPhase::DecodeReady
        };
        Ok(RawToken { token, is_stop })
    }

    fn decode_generated_index(&self) -> Result<usize> {
        if self.phase != RequestPhase::DecodeReady {
            return Err(message("GPT-OSS decode step is not ready"));
        }
        self.generated_tokens
            .len()
            .checked_sub(1)
            .ok_or_else(|| message("GPT-OSS decode has no visible input token"))
    }

    fn fail(&mut self) {
        if self.phase != RequestPhase::Complete {
            self.phase = RequestPhase::Failed;
        }
    }
}

/// One raw token sampled by the engine. Harmony translation belongs to the
/// response/blocking adapter rather than this device execution object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RawToken {
    pub(crate) token: u32,
    pub(crate) is_stop: bool,
}

/// One prepared request. `prefill_step` is an explicit one-shot transition;
/// every successful `decode_step` performs exactly one additional token step.
pub(super) struct RequestExecution {
    lifecycle: RequestLifecycle,
    profile: ExecutionProfile,
    prompt_token_count: usize,
    cache_capacity: usize,
    cache_storage_bytes: usize,
    cache_metadata_bytes: usize,
    metrics: RunMetrics,
    page_table: Option<Buffer>,
    prefill: Option<PrefillState>,
    decode: Option<DecodeState>,
}

struct PrefillState {
    embedding: Arguments,
    layers: Vec<Arguments>,
    layer_types: Vec<AttentionKind>,
    head: Arguments,
    prompt: Buffer,
    position: Buffer,
    sequence_lengths: Buffer,
    query_lengths: Buffer,
    active_rows: Buffer,
    last_index: Buffer,
    sampling_state: Buffer,
    decode: PreparedDecodeState,
}

struct PreparedDecodeState {
    embedding: Arguments,
    pairs: Vec<Arguments>,
    head: Arguments,
    position: Buffer,
    query_lengths: Buffer,
    active_rows: Buffer,
    last_index: Buffer,
}

struct DecodeState {
    embedding: Arguments,
    pairs: Vec<Arguments>,
    head: Arguments,
    token_buffer: Buffer,
    sampling_state: Option<Buffer>,
    position: Option<Buffer>,
    query_lengths: Buffer,
    active_rows: Buffer,
    last_index: Buffer,
    token_download: Slice<'static>,
    lookahead: Option<DecodePrefix>,
}

impl RequestExecution {
    fn completed(
        profile: ExecutionProfile,
        prompt_token_count: usize,
        cache_allocation: Duration,
        cache_storage_bytes: usize,
    ) -> Self {
        Self {
            lifecycle: RequestLifecycle::new(0),
            profile,
            prompt_token_count,
            cache_capacity: profile.decode.cache_capacity(),
            cache_storage_bytes,
            cache_metadata_bytes: 0,
            metrics: RunMetrics {
                cache_allocation,
                ..RunMetrics::default()
            },
            page_table: None,
            prefill: None,
            decode: None,
        }
    }

    pub(super) fn is_complete(&self) -> bool {
        self.lifecycle.phase == RequestPhase::Complete
    }

    pub(super) const fn page_table_width(&self) -> usize {
        self.profile.decode.page_count()
    }

    pub(super) fn decode_cache_lookahead_tokens(&self) -> Result<usize> {
        let generated_index = self.lifecycle.decode_generated_index()?;
        Ok(1 + usize::from(should_enqueue_lookahead(
            generated_index,
            self.lifecycle.max_new_tokens,
        )))
    }

    pub(super) const fn stopped(&self) -> bool {
        self.lifecycle.stopped
    }

    /// Executes the selected bounded prefill family once. Chunked prefill will
    /// replace this full-prompt step in Milestone 3; until then its one-shot
    /// transition is explicit and cannot be repeated.
    fn prefill_step(
        &mut self,
        caches: &mut [LayerCache],
    ) -> Result<Option<RawToken>> {
        let result = self.try_prefill_step(caches);
        if result.is_err() {
            self.lifecycle.fail();
            self.prefill = None;
            self.decode = None;
        }
        result
    }

    fn try_prefill_step(&mut self, caches: &mut [LayerCache]) -> Result<Option<RawToken>> {
        if self.lifecycle.phase == RequestPhase::Complete
            && self.lifecycle.max_new_tokens == 0
        {
            return Ok(None);
        }
        if self.lifecycle.phase != RequestPhase::PrefillPending {
            return Err(message("GPT-OSS prefill step was already consumed"));
        }
        if self.prompt_token_count > self.profile.prefill.sequence() {
            return Err(message(
                "GPT-OSS prepared prompt exceeds its selected prefill family",
            ));
        }
        let mut prefill = self
            .prefill
            .take()
            .ok_or_else(|| message("GPT-OSS prefill state is not available"))?;
        let page_table = self
            .page_table
            .as_ref()
            .ok_or_else(|| message("GPT-OSS prepared request has no page table"))?
            .clone();

        let prefill_started = Instant::now();
        let mut submission = SubmissionTimings::default();
        let submission_started = Instant::now();
        prefill
            .embedding
            .set("tokens", prefill.prompt)
            .map_err(boxed)?;
        let mut hidden = one(prefill.embedding.enqueue().map_err(boxed)?)?;
        submission.embedding = submission_started.elapsed();
        for ((arguments, cache), kind) in prefill
            .layers
            .iter_mut()
            .zip(caches)
            .zip(&prefill.layer_types)
        {
            let (next_hidden, elapsed) = execute_layer(
                arguments,
                hidden,
                prefill.position.clone(),
                Some(prefill.sequence_lengths.clone()),
                prefill.query_lengths.clone(),
                prefill.active_rows.clone(),
                page_table.clone(),
                cache,
            )?;
            hidden = next_hidden;
            record_layer_submission(&mut submission, *kind, elapsed);
        }
        let submission_started = Instant::now();
        prefill.head.set("hidden", hidden).map_err(boxed)?;
        prefill
            .head
            .set("last_index", prefill.last_index)
            .map_err(boxed)?;
        prefill
            .head
            .set("sampling_state", prefill.sampling_state)
            .map_err(boxed)?;
        let mut outputs = prefill
            .head
            .enqueue()
            .map_err(boxed)?
            .into_buffers()
            .into_iter();
        let token_buffer = outputs
            .next()
            .ok_or_else(|| message("GPT-OSS prefill head omitted its token"))?;
        let sampling_state = outputs
            .next()
            .ok_or_else(|| message("GPT-OSS prefill head omitted its sampling state"))?;
        if outputs.next().is_some() {
            return Err(message("GPT-OSS prefill head returned extra buffers"));
        }
        submission.head = submission_started.elapsed();
        token_buffer.wait().map_err(boxed)?;
        self.metrics.prefill_execution = prefill_started.elapsed();
        self.metrics.prefill_submission = submission;

        let download_started = Instant::now();
        let token = download_token(&token_buffer)?;
        self.metrics.prefill_download = download_started.elapsed();
        let token_download = Slice::alloc(token_buffer.shape()).map_err(boxed)?;
        self.decode = Some(DecodeState {
            embedding: prefill.decode.embedding,
            pairs: prefill.decode.pairs,
            head: prefill.decode.head,
            token_buffer,
            sampling_state: Some(sampling_state),
            position: Some(prefill.decode.position),
            query_lengths: prefill.decode.query_lengths,
            active_rows: prefill.decode.active_rows,
            last_index: prefill.decode.last_index,
            token_download,
            lookahead: None,
        });
        let is_stop = super::protocol::is_stop_token(token);
        self.lifecycle.record_prefill(token, is_stop).map(Some)
    }

    /// Consumes the preceding device token and returns exactly one newly
    /// sampled raw token. A terminal or budget-ending token may have a bounded
    /// prefix in flight; `finalize` discards it without executing the suffix.
    fn decode_step(
        &mut self,
        caches: &mut [LayerCache],
    ) -> Result<Option<RawToken>> {
        let result = self.try_decode_step(caches);
        if result.is_err() {
            self.lifecycle.fail();
            self.prefill = None;
            self.decode = None;
        }
        result
    }

    fn try_decode_step(&mut self, caches: &mut [LayerCache]) -> Result<Option<RawToken>> {
        if self.is_complete() {
            return Ok(None);
        }
        let generated_index = self.lifecycle.decode_generated_index()?;
        let state = self
            .decode
            .as_mut()
            .ok_or_else(|| message("completed GPT-OSS request has no decode state"))?;
        let position = state
            .position
            .as_ref()
            .ok_or_else(|| message("GPT-OSS decode position is owned by an execution"))?
            .clone();
        let lookahead_cache_count = DECODE_LOOKAHEAD_PAIRS * 2;
        let (lookahead_pairs, remaining_pairs) =
            state.pairs.split_at_mut(DECODE_LOOKAHEAD_PAIRS);
        let (lookahead_caches, remaining_caches) =
            caches.split_at_mut(lookahead_cache_count);
        let page_table = self
            .page_table
            .as_ref()
            .ok_or_else(|| message("GPT-OSS prepared request has no page table"))?
            .clone();

        let decode_started = Instant::now();
        let prefix = match state.lookahead.take() {
            Some(prefix) => prefix,
            None => enqueue_decode_prefix(
                &mut state.embedding,
                lookahead_pairs,
                state.token_buffer.clone(),
                position.clone(),
                state.query_lengths.clone(),
                state.active_rows.clone(),
                page_table.clone(),
                lookahead_caches,
            )?,
        };
        let mut hidden = prefix.hidden;
        let mut submission = prefix.submission;
        for (arguments, caches) in remaining_pairs
            .iter_mut()
            .zip(remaining_caches.chunks_exact_mut(2))
        {
            let (next_hidden, elapsed) = execute_layer_pair(
                arguments,
                hidden,
                position.clone(),
                state.query_lengths.clone(),
                state.active_rows.clone(),
                page_table.clone(),
                caches,
            )?;
            hidden = next_hidden;
            submission.layer_pairs += elapsed;
        }
        let submission_started = Instant::now();
        state.head.set("hidden", hidden).map_err(boxed)?;
        state
            .head
            .set("last_index", state.last_index.clone())
            .map_err(boxed)?;
        state
            .head
            .set(
                "sampling_state",
                state.sampling_state.take().ok_or_else(|| {
                    message("GPT-OSS sampling state is owned by an execution")
                })?,
            )
            .map_err(boxed)?;
        state
            .head
            .set(
                "position",
                state
                    .position
                    .take()
                    .ok_or_else(|| message("GPT-OSS decode position is owned by an execution"))?,
            )
            .map_err(boxed)?;
        let mut outputs = state
            .head
            .enqueue()
            .map_err(boxed)?
            .into_buffers()
            .into_iter();
        state.token_buffer = outputs
            .next()
            .ok_or_else(|| message("GPT-OSS decode head omitted its token"))?;
        state.sampling_state = Some(
            outputs
                .next()
                .ok_or_else(|| message("GPT-OSS decode head omitted its sampling state"))?,
        );
        state.position = Some(
            outputs
                .next()
                .ok_or_else(|| message("GPT-OSS decode head omitted its position"))?,
        );
        if outputs.next().is_some() {
            return Err(message("GPT-OSS decode head returned extra buffers"));
        }
        submission.head = submission_started.elapsed();

        // Observe the token directly after the head. The bounded prefix below
        // remains an overlapping consumer of the same device token buffer.
        let pending_token = state
            .token_buffer
            .download_to(&mut state.token_download)
            .map_err(boxed)?;
        if should_enqueue_lookahead(generated_index, self.lifecycle.max_new_tokens) {
            state.lookahead = Some(enqueue_decode_prefix(
                &mut state.embedding,
                lookahead_pairs,
                state.token_buffer.clone(),
                state
                    .position
                    .as_ref()
                    .ok_or_else(|| message("GPT-OSS decode head omitted its position"))?
                    .clone(),
                state.query_lengths.clone(),
                state.active_rows.clone(),
                page_table,
                lookahead_caches,
            )?);
        }
        if !state.token_buffer.is_ready().map_err(boxed)? {
            state.token_buffer.wait().map_err(boxed)?;
        }
        let execution_elapsed = decode_started.elapsed();
        if generated_index == 0 {
            self.metrics.first_decode_execution = execution_elapsed;
            self.metrics.first_decode_submission = submission;
        } else {
            self.metrics.steady_decode_execution += execution_elapsed;
            add_submission(&mut self.metrics.steady_decode_submission, submission);
        }
        let download_started = Instant::now();
        pending_token.wait().map_err(boxed)?;
        let token = token_from_slice(&state.token_download)?;
        self.metrics.decode_download += download_started.elapsed();
        let is_stop = super::protocol::is_stop_token(token);
        self.lifecycle.record_decode(token, is_stop).map(Some)
    }

    pub(super) fn finalize(mut self) -> Result<RunReport> {
        if !self.is_complete() {
            return Err(message(
                "GPT-OSS request cannot finalize before a terminal token or visible budget",
            ));
        }
        // A stop token is observed after the bounded prefix was submitted.
        // Discard that prefix here: it is never extended through the remaining
        // pairs or head, preserving terminal-token semantics.
        if let Some(decode) = &mut self.decode {
            decode.lookahead.take();
        }
        Ok(RunReport {
            generated_tokens: self.lifecycle.generated_tokens,
            cache_capacity: self.cache_capacity,
            cache_storage_bytes: self.cache_storage_bytes,
            cache_metadata_bytes: self.cache_metadata_bytes,
            cache_metadata_upload_bytes: self.metrics.cache_metadata_upload_bytes,
            metrics: self.metrics,
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

struct BoundComponentFamily {
    embedding: Arguments,
    layers: BoundLayerExecutables,
    head: Arguments,
}

enum BoundLayerExecutables {
    Prefill {
        layers: Vec<Arguments>,
        kinds: Vec<AttentionKind>,
    },
    Decode {
        pairs: Vec<Arguments>,
    },
}

impl BoundComponentFamily {
    fn bind(
        family: &ComponentFamily,
        checkpoint: &Checkpoint,
        parameters: &LoadedCheckpoint,
        config: &Config,
    ) -> Result<Self> {
        let mut embedding = family.embedding.args();
        bind_embedding(&mut embedding, checkpoint, parameters)?;
        let mut head = family.head.args();
        bind_head(&mut head, parameters)?;
        let layers = match &family.layers {
            LayerExecutables::Prefill { .. } => BoundLayerExecutables::Prefill {
                layers: bind_layers(family, checkpoint, parameters, config)?,
                kinds: config.layer_types().to_vec(),
            },
            LayerExecutables::DecodePair(_) => BoundLayerExecutables::Decode {
                pairs: bind_decode_pairs(family, checkpoint, parameters, config)?,
            },
        };
        Ok(Self {
            embedding,
            layers,
            head,
        })
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

// Five layer pairs cover the pair-four host-submission bubble observed on A40
// while keeping terminal speculation bounded to ten of the model's 24 layers.
const DECODE_LOOKAHEAD_PAIRS: usize = 5;

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
    arguments: &mut Arguments,
    checkpoint: &Checkpoint,
    parameters: &LoadedCheckpoint,
) -> Result<()> {
    bind_tree(
        arguments,
        &checkpoint.model.embed_tokens,
        &parameters.model.embed_tokens,
    )
}

fn bind_head(arguments: &mut Arguments, parameters: &LoadedCheckpoint) -> Result<()> {
    arguments
        .set_parameter(&parameters.model.norm.weight)
        .map_err(boxed)?;
    arguments
        .set_parameter(&parameters.lm_head.weight)
        .map_err(boxed)?;
    arguments.bake().map_err(boxed)?;
    Ok(())
}

fn bind_layers(
    family: &ComponentFamily,
    checkpoint: &Checkpoint,
    parameters: &LoadedCheckpoint,
    config: &Config,
) -> Result<Vec<Arguments>> {
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

fn bind_decode_pairs(
    family: &ComponentFamily,
    checkpoint: &Checkpoint,
    parameters: &LoadedCheckpoint,
    config: &Config,
) -> Result<Vec<Arguments>> {
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

fn bind_layer(
    executable: &Exe,
    slots: &super::checkpoint::DecoderLayer,
    loaded: &LoadedDecoderLayer,
) -> Result<Arguments> {
    let mut arguments = executable.args();
    bind_tree(&mut arguments, slots, loaded)?;
    Ok(arguments)
}

fn bind_layer_pair(
    executable: &Exe,
    slots: &[super::checkpoint::DecoderLayer],
    loaded: &[LoadedDecoderLayer],
) -> Result<Arguments> {
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
    embedding: &mut Arguments,
    pairs: &mut [Arguments],
    token: Buffer,
    position: Buffer,
    query_lengths: Buffer,
    active_rows: Buffer,
    page_table: Buffer,
    caches: &mut [LayerCache],
) -> Result<DecodePrefix> {
    if pairs.len() != DECODE_LOOKAHEAD_PAIRS || caches.len() != pairs.len() * 2 {
        return Err(message("GPT-OSS decode lookahead schedule is invalid"));
    }
    let mut submission = SubmissionTimings::default();
    let started = Instant::now();
    embedding.set("tokens", token).map_err(boxed)?;
    let mut hidden = one(embedding.enqueue().map_err(boxed)?)?;
    submission.embedding = started.elapsed();
    for (pair, pair_caches) in pairs.iter_mut().zip(caches.chunks_exact_mut(2)) {
        let (next_hidden, elapsed) = execute_layer_pair(
            pair,
            hidden,
            position.clone(),
            query_lengths.clone(),
            active_rows.clone(),
            page_table.clone(),
            pair_caches,
        )?;
        hidden = next_hidden;
        submission.layer_pairs += elapsed;
    }
    Ok(DecodePrefix { hidden, submission })
}

const fn should_enqueue_lookahead(generated_index: usize, max_new_tokens: usize) -> bool {
    max_new_tokens.saturating_sub(generated_index) > 2
}

fn execute_layer(
    arguments: &mut Arguments,
    hidden: Buffer,
    position: Buffer,
    sequence_lengths: Option<Buffer>,
    query_lengths: Buffer,
    active_rows: Buffer,
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
    arguments
        .set("query_lengths", query_lengths)
        .map_err(boxed)?;
    arguments.set("active_rows", active_rows).map_err(boxed)?;
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
    arguments: &mut Arguments,
    hidden: Buffer,
    position: Buffer,
    query_lengths: Buffer,
    active_rows: Buffer,
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
    arguments
        .set("query_lengths", query_lengths)
        .map_err(boxed)?;
    arguments.set("active_rows", active_rows).map_err(boxed)?;
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

fn execute_serving_layer(
    arguments: &mut Arguments,
    hidden: Buffer,
    batch_slab: Buffer,
    cache: &mut LayerCache,
) -> Result<(Buffer, Duration)> {
    let started = Instant::now();
    let (key, value) = cache.take()?;
    arguments.set("hidden", hidden).map_err(boxed)?;
    arguments
        .set("batch_slab", batch_slab)
        .map_err(boxed)?;
    arguments.set("cache.key", key).map_err(boxed)?;
    arguments.set("cache.value", value).map_err(boxed)?;
    let mut outputs = arguments
        .enqueue()
        .map_err(boxed)?
        .into_buffers()
        .into_iter();
    let hidden = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer omitted hidden state"))?;
    let key = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer omitted key cache"))?;
    let value = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer omitted value cache"))?;
    if outputs.next().is_some() {
        return Err(message("GPT-OSS serving layer returned extra buffers"));
    }
    cache.install(key, value)?;
    Ok((hidden, started.elapsed()))
}

fn execute_serving_layer_pair(
    arguments: &mut Arguments,
    hidden: Buffer,
    batch_slab: Buffer,
    caches: &mut [LayerCache],
) -> Result<(Buffer, Duration)> {
    let [sliding, full] = caches else {
        return Err(message(
            "GPT-OSS serving decode execution requires exactly two caches",
        ));
    };
    let started = Instant::now();
    let (sliding_key, sliding_value) = sliding.take()?;
    let (full_key, full_value) = full.take()?;
    arguments.set("hidden", hidden).map_err(boxed)?;
    arguments
        .set("batch_slab", batch_slab)
        .map_err(boxed)?;
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
        .ok_or_else(|| message("GPT-OSS serving layer pair omitted hidden state"))?;
    let sliding_key = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer pair omitted sliding key cache"))?;
    let sliding_value = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer pair omitted sliding value cache"))?;
    let full_key = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer pair omitted full key cache"))?;
    let full_value = outputs
        .next()
        .ok_or_else(|| message("GPT-OSS serving layer pair omitted full value cache"))?;
    if outputs.next().is_some() {
        return Err(message(
            "GPT-OSS serving layer pair returned extra buffers",
        ));
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

fn upload_u8(
    platform: &Platform,
    shape: Shape,
    values: &[u8],
    placement: &Sharding,
) -> Result<Buffer> {
    let slice = Slice::from_typed(shape, values).map_err(boxed)?;
    platform
        .upload(&slice, placement.clone(), Memory::Default)
        .map_err(boxed)
}

fn upload_f32_vector(
    platform: &Platform,
    values: &[f32],
    placement: &Sharding,
) -> Result<Buffer> {
    let shape = Shape::new(
        nml::DataType::F32,
        &[i64::try_from(values.len()).map_err(|_| message("F32 upload length exceeds I64"))?],
    )
    .map_err(boxed)?;
    let slice = Slice::from_typed(shape, values).map_err(boxed)?;
    platform
        .upload(&slice, placement.clone(), Memory::Default)
        .map_err(boxed)
}

fn upload_bool(
    platform: &Platform,
    shape: Shape,
    values: &[bool],
    placement: &Sharding,
) -> Result<Buffer> {
    let slice = Slice::from_typed(shape, values).map_err(boxed)?;
    platform
        .upload(&slice, placement.clone(), Memory::Default)
        .map_err(boxed)
}

fn validate_batch_inputs(family: ShapeFamily, input: &BatchInputs) -> Result<()> {
    let batch = family.batch();
    let query = family.sequence();
    let tokens = match family.phase() {
        Phase::Prefill => batch
            .checked_mul(query)
            .ok_or_else(|| message("batch token shape overflows usize"))?,
        Phase::Decode => batch,
    };
    let table_entries = batch
        .checked_mul(family.page_count())
        .ok_or_else(|| message("batch page-table shape overflows usize"))?;
    if input.tokens.len() != tokens
        || input.positions.len() != batch
        || input.sequence_lengths.len() != batch
        || input.query_lengths.len() != batch
        || input.active_rows.len() != batch
        || input.sample_rows.len() != batch
        || input.page_tables.len() != table_entries
        || input.last_indices.len() != batch
        || input.sampling_states.len() != batch * 2
        || input.top_k.len() != batch
        || input.temperature.len() != batch
        || input.top_p.len() != batch
        || input.min_p.len() != batch
    {
        return Err(message("batch input vectors do not match the compiled family"));
    }
    for row in 0..batch {
        if input.sample_rows[row] && !input.active_rows[row] {
            return Err(message("only an active batch row may sample"));
        }
        if input.active_rows[row] {
            let query_length = usize::try_from(input.query_lengths[row])
                .map_err(|_| message("active query length is negative"))?;
            if query_length == 0 || query_length > query {
                return Err(message("active query length exceeds its compiled family"));
            }
            if input.positions[row] < 0
                || input.sequence_lengths[row]
                    < input.positions[row].saturating_add(input.query_lengths[row])
                || input.last_indices[row] < 0
                || input.last_indices[row] >= input.query_lengths[row]
            {
                return Err(message("active batch row has inconsistent positions or lengths"));
            }
        } else if input.positions[row] != 0
            || input.sequence_lengths[row] != 0
            || input.query_lengths[row] != 0
        {
            return Err(message("inactive batch rows must have zero positions and lengths"));
        }
    }
    Ok(())
}

fn pack_batch_slab(
    family: ShapeFamily,
    layout: ServingSlabLayout,
    input: &BatchInputs,
) -> Result<Vec<u8>> {
    let mut packed = Vec::with_capacity(layout.total_bytes());
    require_slab_offset(&packed, layout.token_offset())?;
    for token in &input.tokens {
        packed.extend_from_slice(&token.to_ne_bytes());
    }
    require_slab_offset(&packed, layout.layer_offset())?;
    for row in 0..family.batch() {
        for value in [
            input.positions[row],
            input.sequence_lengths[row],
            input.query_lengths[row],
            i32::from(input.active_rows[row]),
        ] {
            packed.extend_from_slice(&value.to_ne_bytes());
        }
        let page_start = row
            .checked_mul(family.page_count())
            .ok_or_else(|| message("serving page-table offset overflows usize"))?;
        for page in &input.page_tables[page_start..page_start + family.page_count()] {
            packed.extend_from_slice(&page.to_ne_bytes());
        }
    }
    require_slab_offset(&packed, layout.head_i32_offset())?;
    for row in 0..family.batch() {
        for value in [
            input.last_indices[row],
            input.top_k[row],
            i32::from(input.sample_rows[row]),
        ] {
            packed.extend_from_slice(&value.to_ne_bytes());
        }
    }
    require_slab_offset(&packed, layout.sampling_state_offset())?;
    for state in &input.sampling_states {
        packed.extend_from_slice(&state.to_ne_bytes());
    }
    require_slab_offset(&packed, layout.head_f32_offset())?;
    for row in 0..family.batch() {
        for value in [
            input.temperature[row],
            input.top_p[row],
            input.min_p[row],
        ] {
            packed.extend_from_slice(&value.to_ne_bytes());
        }
    }
    require_slab_offset(&packed, layout.total_bytes())?;
    Ok(packed)
}

fn require_slab_offset(bytes: &[u8], expected: usize) -> Result<()> {
    if bytes.len() != expected {
        return Err(message("serving batch-slab layout is inconsistent"));
    }
    Ok(())
}

fn download_batch_outputs(result: &Buffer, batch: usize) -> Result<BatchOutputs> {
    let mut result_slice = Slice::alloc(result.shape()).map_err(boxed)?;
    result
        .download_to(&mut result_slice)
        .map_err(boxed)?
        .wait()
        .map_err(boxed)?;
    decode_batch_output_bytes(result_slice.items::<u8>().map_err(boxed)?, batch)
}

fn decode_batch_output_bytes(bytes: &[u8], batch: usize) -> Result<BatchOutputs> {
    if bytes.len()
        != batch
            .checked_mul(BATCH_RESULT_BYTES_PER_ROW)
            .ok_or_else(|| message("serving result byte count overflows usize"))?
    {
        return Err(message("serving result buffer has an invalid byte count"));
    }
    let mut tokens = Vec::with_capacity(batch);
    let mut sampling_states = Vec::with_capacity(batch * 2);
    for row in bytes.chunks_exact(BATCH_RESULT_BYTES_PER_ROW) {
        tokens.push(i32::from_ne_bytes(
            row[..std::mem::size_of::<i32>()]
                .try_into()
                .expect("I32 result width is constant"),
        ));
        let states = &row[std::mem::size_of::<i32>()..];
        for state in states.chunks_exact(std::mem::size_of::<u64>()) {
            sampling_states.push(u64::from_ne_bytes(
                state.try_into().expect("U64 result width is constant"),
            ));
        }
    }
    Ok(BatchOutputs {
        tokens,
        sampling_states,
    })
}

fn download_token(buffer: &Buffer) -> Result<u32> {
    let slice = buffer.to_slice().map_err(boxed)?;
    token_from_slice(&slice)
}

fn token_from_slice(slice: &Slice<'_>) -> Result<u32> {
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
    compiled_families: usize,
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
        compiled_families,
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

    fn reference_visible_tokens(samples: &[(u32, bool)], budget: usize) -> Vec<u32> {
        samples
            .iter()
            .copied()
            .take(budget)
            .scan(false, |complete, (token, is_stop)| {
                if *complete {
                    return None;
                }
                *complete = is_stop;
                Some(token)
            })
            .collect()
    }

    fn stepped_visible_tokens(samples: &[(u32, bool)], budget: usize) -> Vec<u32> {
        let mut lifecycle = RequestLifecycle::new(budget);
        if budget == 0 {
            return lifecycle.generated_tokens;
        }
        let (token, is_stop) = samples[0];
        lifecycle.record_prefill(token, is_stop).unwrap();
        for &(token, is_stop) in &samples[1..] {
            if lifecycle.phase == RequestPhase::Complete {
                break;
            }
            lifecycle.record_decode(token, is_stop).unwrap();
        }
        lifecycle.generated_tokens
    }

    #[test]
    fn step_lifecycle_matches_the_complete_loop_for_every_terminal_shape() {
        for (samples, budget) in [
            (vec![(11, false), (12, false), (13, true), (14, false)], 8),
            (vec![(21, true), (22, false)], 8),
            (vec![(31, false), (32, false), (33, false)], 2),
            (vec![(41, false)], 1),
            (vec![(51, false)], 0),
        ] {
            assert_eq!(
                stepped_visible_tokens(&samples, budget),
                reference_visible_tokens(&samples, budget),
            );
        }
    }

    #[test]
    fn prefill_is_a_one_shot_transition_and_decode_requires_a_visible_input() {
        let mut lifecycle = RequestLifecycle::new(3);
        assert_eq!(lifecycle.phase, RequestPhase::PrefillPending);
        assert!(lifecycle.record_decode(1, false).is_err());
        assert_eq!(
            lifecycle.record_prefill(7, false).unwrap(),
            RawToken {
                token: 7,
                is_stop: false,
            },
        );
        assert_eq!(lifecycle.phase, RequestPhase::DecodeReady);
        assert_eq!(lifecycle.decode_generated_index().unwrap(), 0);
        assert!(lifecycle.record_prefill(8, false).is_err());
    }

    #[test]
    fn terminal_and_budget_tokens_make_all_later_decode_steps_ineligible() {
        let mut terminal = RequestLifecycle::new(4);
        terminal.record_prefill(1, false).unwrap();
        assert!(should_enqueue_lookahead(
            terminal.decode_generated_index().unwrap(),
            terminal.max_new_tokens,
        ));
        terminal.record_decode(2, true).unwrap();
        assert_eq!(terminal.phase, RequestPhase::Complete);
        assert!(terminal.stopped);
        assert!(terminal.decode_generated_index().is_err());
        assert!(terminal.record_decode(3, false).is_err());

        let mut bounded = RequestLifecycle::new(2);
        bounded.record_prefill(1, false).unwrap();
        assert!(!should_enqueue_lookahead(
            bounded.decode_generated_index().unwrap(),
            bounded.max_new_tokens,
        ));
        bounded.record_decode(2, false).unwrap();
        assert_eq!(bounded.phase, RequestPhase::Complete);
        assert!(!bounded.stopped);
    }

    #[test]
    fn zero_token_request_is_complete_without_prefill_or_decode() {
        let lifecycle = RequestLifecycle::new(0);
        assert_eq!(lifecycle.phase, RequestPhase::Complete);
        assert!(lifecycle.generated_tokens.is_empty());
        assert!(!lifecycle.stopped);
        assert!(lifecycle.decode_generated_index().is_err());
    }

    #[test]
    fn execution_failure_poisoning_is_terminal_but_not_a_success() {
        let mut lifecycle = RequestLifecycle::new(3);
        lifecycle.record_prefill(1, false).unwrap();
        lifecycle.fail();
        assert_eq!(lifecycle.phase, RequestPhase::Failed);
        assert!(lifecycle.record_decode(2, false).is_err());
        assert!(!lifecycle.stopped);

        let mut complete = RequestLifecycle::new(1);
        complete.record_prefill(1, false).unwrap();
        complete.fail();
        assert_eq!(complete.phase, RequestPhase::Complete);
    }

    #[test]
    fn decode_lookahead_never_crosses_the_visible_token_budget() {
        assert_eq!(DECODE_LOOKAHEAD_PAIRS, 5);
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
    fn serving_slab_packs_every_typed_input_into_one_columnar_transfer() {
        let family = ShapeFamily::serving_decode(2, 32, 16, 1).unwrap();
        let input = BatchInputs {
            tokens: vec![11, 22],
            positions: vec![4, 0],
            sequence_lengths: vec![5, 0],
            query_lengths: vec![1, 0],
            active_rows: vec![true, false],
            sample_rows: vec![true, false],
            page_tables: vec![7, 8, -1, -1],
            last_indices: vec![0, 0],
            sampling_states: vec![101, 102, 201, 202],
            top_k: vec![17, 1],
            temperature: vec![0.7, 1.0],
            top_p: vec![0.8, 1.0],
            min_p: vec![0.05, 0.0],
        };
        validate_batch_inputs(family, &input).unwrap();
        let layout = ServingSlabLayout::for_family(family).unwrap();
        let slab = pack_batch_slab(family, layout, &input).unwrap();
        assert_eq!(slab.len(), layout.total_bytes());
        let i32_at = |offset: usize| {
            i32::from_ne_bytes(slab[offset..offset + 4].try_into().unwrap())
        };
        let u64_at = |offset: usize| {
            u64::from_ne_bytes(slab[offset..offset + 8].try_into().unwrap())
        };
        let f32_at = |offset: usize| {
            f32::from_ne_bytes(slab[offset..offset + 4].try_into().unwrap())
        };
        assert_eq!(i32_at(layout.token_offset()), 11);
        assert_eq!(
            (0..6)
                .map(|index| i32_at(layout.layer_offset() + index * 4))
                .collect::<Vec<_>>(),
            [4, 5, 1, 1, 7, 8],
        );
        assert_eq!(
            (0..3)
                .map(|index| i32_at(layout.head_i32_offset() + index * 4))
                .collect::<Vec<_>>(),
            [0, 17, 1],
        );
        assert_eq!(u64_at(layout.sampling_state_offset()), 101);
        assert_eq!(u64_at(layout.sampling_state_offset() + 8), 102);
        assert_eq!(f32_at(layout.head_f32_offset()), 0.7);
        assert_eq!(f32_at(layout.head_f32_offset() + 4), 0.8);
        assert_eq!(f32_at(layout.head_f32_offset() + 8), 0.05);
        assert_eq!(i32_at(layout.token_offset() + 4), 22);
        assert_eq!(
            (0..6)
                .map(|index| i32_at(layout.layer_offset() + (6 + index) * 4))
                .collect::<Vec<_>>(),
            [0, 0, 0, 0, -1, -1],
        );
        assert_eq!(u64_at(layout.sampling_state_offset() + 16), 201);
        assert!(family.is_serving());
        assert!(!ShapeFamily::decode(32, 16).unwrap().is_serving());

        let mut result = Vec::new();
        for (token, states) in [(-1_i32, [101_u64, 102_u64]), (22, [201, 202])] {
            result.extend_from_slice(&token.to_ne_bytes());
            result.extend_from_slice(&states[0].to_ne_bytes());
            result.extend_from_slice(&states[1].to_ne_bytes());
        }
        let decoded = decode_batch_output_bytes(&result, 2).unwrap();
        assert_eq!(decoded.tokens, [-1, 22]);
        assert_eq!(decoded.sampling_states, [101, 102, 201, 202]);
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
            64,
        )
        .unwrap();
        let large = ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 65,
                max_sequence_tokens: 300,
            },
            131_072,
            64,
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
            64,
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
        assert!(normalize_profiles(&[], 131_072, 64).is_err());
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
            64,
        )
        .is_err());
        assert!(ExecutionProfile::new(
            CompilationProfile {
                max_prompt_tokens: 16,
                max_sequence_tokens: 131_073,
            },
            131_072,
            64,
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
            64,
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
