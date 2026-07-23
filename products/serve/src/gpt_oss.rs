//! GPT-OSS 20B product ownership.
//!
//! Artifact identity, Harmony, checkpoint structure, component graphs, and the
//! transformer schedule live here. Framework crates know only tensors,
//! parameters, compilation, buffers, and reusable executable slots.

#![forbid(unsafe_code)]

mod artifact;
mod checkpoint;
mod config;
mod execution;
mod graph;
pub(crate) mod protocol;

use crate::{CompilationProfile, SamplingOptions, SubmissionTimings};
use crate::server::cache::{
    CacheStats, FrozenCachePlan, PageManager, TargetCacheGeometry, TARGET_PAGE_SIZE,
};
use crate::server::contracts::{SequenceId, ServerProfile};
use checkpoint::{BoxError, Result};
use config::Config;
pub(crate) use execution::RawToken;
use execution::{
    BatchInputs, ModelDefinition, PreparedRequest, RequestExecution, ResidentModel, RunMetrics,
    ServingCompileConfig, StableDecodeBatchLane, StartupMetrics,
};
use nml::io::ParameterSet;
use nml::safetensors::TensorRegistry;
use protocol::{Conversation, HarmonyProtocol, Message, SystemContent};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::{Duration, Instant};

pub(crate) const MODEL_NAME: &str = "GPT-OSS 20B NVFP4";
const ARTIFACT_MANIFEST: &str = "nml-artifact-manifest.json";
const CHECKPOINT_INDEX: &str = "model.safetensors.index.json";
const DIRECT_CHECKPOINT: &str = "model.safetensors";
const ARTIFACT_MANIFEST_SHA256: &str =
    "3c36a89cbc0f908b3e782550fe32f3b6890ef3f857232d11710bc8e0dbcea71d";
const ARTIFACT_FILE_COUNT: usize = 20;
const ARTIFACT_TOTAL_BYTES: u64 = 11_805_934_204;
const ARTIFACT_RECIPE: &str = "nml-nvfp4-weight-v2";
const ARTIFACT_RECIPE_SHA256: &str =
    "68bad1480e9a68e4fa3d36c17315b8bcd5490e777cfd738f15c71101f6bb6603";
const SOURCE_MANIFEST_SHA256: &str =
    "4f9fd730e12e0535cf6788a11a9b1604749f4520738a5c7ea643e27bf4b5ccb1";
const TENSOR_MANIFEST_SHA256: &str =
    "fd7c6833d00eca158bc1145dc2577ad8d38d8f4ed977ef3e3dfc0c2a72ea5cae";
const SOURCE_REPOSITORY: &str = "unsloth/gpt-oss-20b-BF16";
const SOURCE_REVISION: &str = "cc89b3e7fd423253264883a80a4fa5abc619649f";
const CONVERTER_NAME: &str = "nml-nvfp4-converter";
const CONVERTER_VERSION: u32 = 2;
const CONVERTER_SCRIPT_SHA256: &str =
    "ca9a7a714798a8095d3eca1b873af8fed8637ba23b7ea984926c2c64de1c4079";
const CONVERTER_REQUIREMENTS_SHA256: &str =
    "f384757dfae59e89aa0dfad0ea75a651005a336437903981119636ed58de8c8e";

/// Persistent product model. Its complete execution plan is compiled before
/// its parameters become resident and is reused by every request.
pub(crate) struct Generator<'platform> {
    protocol: HarmonyProtocol,
    model: ResidentModel<'platform>,
    pages: PageManager,
    next_sequence: u64,
    stable_decode: StableDecodeLaneState,
}

#[derive(Default)]
struct StableDecodeLaneState {
    current: Option<StableDecodeBatch>,
}

struct StableDecodeBatch {
    descriptor: StableDecodeDescriptor,
    lane: StableDecodeBatchLane,
}

#[derive(Eq, PartialEq)]
struct StableDecodeDescriptor {
    members: Vec<SequenceId>,
    page_tables: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StableDecodeTransition {
    Reuse,
    Rebind { preserve_lookahead: bool },
}

impl StableDecodeDescriptor {
    fn transition_to(&self, replacement: &Self) -> StableDecodeTransition {
        if self == replacement {
            return StableDecodeTransition::Reuse;
        }
        let preserve_lookahead = self.members == replacement.members
            && self.page_tables.len() == replacement.page_tables.len()
            && self
                .page_tables
                .iter()
                .zip(&replacement.page_tables)
                .all(|(&old, &new)| old == new || (old == -1 && new >= 0));
        StableDecodeTransition::Rebind { preserve_lookahead }
    }
}

impl StableDecodeLaneState {
    fn transition_to(&self, replacement: &StableDecodeDescriptor) -> StableDecodeTransition {
        self.current.as_ref().map_or(
            StableDecodeTransition::Rebind {
                preserve_lookahead: false,
            },
            |current| current.descriptor.transition_to(replacement),
        )
    }

    fn install(
        &mut self,
        descriptor: StableDecodeDescriptor,
        mut lane: StableDecodeBatchLane,
    ) {
        let transition = self.transition_to(&descriptor);
        debug_assert_ne!(transition, StableDecodeTransition::Reuse);
        if transition
            == (StableDecodeTransition::Rebind {
                preserve_lookahead: true,
            })
        {
            let carried = lane.carry_lookahead_from(
                &mut self
                    .current
                    .as_mut()
                    .expect("lookahead-preserving transition has a resident lane")
                    .lane,
            );
            debug_assert!(
                carried,
                "lookahead-preserving transition must have pending prefix work",
            );
        }
        self.current = Some(StableDecodeBatch { descriptor, lane });
    }

    fn lane_mut(&mut self) -> Option<&mut StableDecodeBatchLane> {
        self.current.as_mut().map(|current| &mut current.lane)
    }

    fn clear(&mut self) {
        self.current.take();
    }
}

pub(crate) struct ProductReport {
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) cache_capacity: usize,
    pub(crate) physical_parameter_components: usize,
    pub(crate) parameter_source_bytes: usize,
    pub(crate) parameter_resident_bytes: usize,
    pub(crate) parameter_prepared_bytes: usize,
    pub(crate) parameter_peak_staging_bytes: usize,
    pub(crate) cache_storage_bytes: usize,
    pub(crate) cache_metadata_bytes: usize,
    pub(crate) cache_metadata_upload_bytes: usize,
    pub(crate) timings: ProductTimings,
}

pub(crate) struct ServerSession {
    sequence: SequenceId,
    prompt: Vec<u32>,
    prompt_cursor: usize,
    generated: Vec<u32>,
    max_new_tokens: usize,
    sampling: SamplingOptions,
    stopped: bool,
    released: bool,
}

impl ServerSession {
    pub(crate) const fn sequence(&self) -> SequenceId {
        self.sequence
    }

    pub(crate) fn prompt_remaining(&self) -> usize {
        self.prompt.len() - self.prompt_cursor
    }

    pub(crate) fn prefill_complete(&self) -> bool {
        self.prompt_cursor == self.prompt.len()
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.max_new_tokens == 0
            || self.stopped
            || self.generated.len() >= self.max_new_tokens
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.prompt.len()
    }

    pub(crate) fn completion_tokens(&self) -> usize {
        self.generated.len()
    }

    pub(crate) const fn stopped(&self) -> bool {
        self.stopped
    }
}

impl Generator<'_> {
    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.pages.stats()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ProductTimings {
    pub(crate) tokenization: Duration,
    pub(crate) artifact_validation: Duration,
    pub(crate) prefill_compilation: Duration,
    pub(crate) decode_compilation: Duration,
    pub(crate) parameter_upload: Duration,
    pub(crate) cache_allocation: Duration,
    pub(crate) cache_metadata_upload: Duration,
    pub(crate) prompt_upload: Duration,
    pub(crate) prefill_execution: Duration,
    pub(crate) prefill_download: Duration,
    pub(crate) decode_state_initialization: Duration,
    pub(crate) first_decode_execution: Duration,
    pub(crate) steady_decode_execution: Duration,
    pub(crate) decode_download: Duration,
    pub(crate) prefill_submission: SubmissionTimings,
    pub(crate) first_decode_submission: SubmissionTimings,
    pub(crate) steady_decode_submission: SubmissionTimings,
}

impl<'platform> Generator<'platform> {
    pub(crate) fn load(
        platform: &'platform nml::Platform,
        model_directory: &Path,
        profiles: &[CompilationProfile],
    ) -> Result<Self> {
        Self::load_sized(platform, model_directory, profiles, None, None)
    }

    pub(crate) fn load_server(
        platform: &'platform nml::Platform,
        model_directory: &Path,
        profile: &ServerProfile,
    ) -> Result<Self> {
        Self::load_sized(
            platform,
            model_directory,
            &profile.compilation_families,
            Some((profile.cache_budget_bytes, profile.cache_safety_bytes)),
            Some(ServingCompileConfig {
                batch_buckets: profile.batch_buckets.clone(),
                prefill_query_buckets: profile.prefill_query_buckets.clone(),
                logical_cache_capacity: profile.max_model_length,
                tensor_parallel: profile.tensor_parallel,
            }),
        )
    }

    fn load_sized(
        platform: &'platform nml::Platform,
        model_directory: &Path,
        profiles: &[CompilationProfile],
        cache_budget: Option<(usize, usize)>,
        serving: Option<ServingCompileConfig>,
    ) -> Result<Self> {
        let validation_started = Instant::now();
        validate_artifact(model_directory).map_err(boxed)?;
        let artifact_validation = validation_started.elapsed();
        let config = Config::from_file(model_directory.join("config.json")).map_err(boxed)?;
        let protocol = HarmonyProtocol::load(model_directory).map_err(boxed)?;
        let registry =
            TensorRegistry::from_path(model_directory.join(CHECKPOINT_INDEX)).map_err(boxed)?;
        let physical_pages = match cache_budget {
            Some((budget, safety)) => FrozenCachePlan::freeze(
                TargetCacheGeometry {
                    layers: config.layers(),
                    page_size: TARGET_PAGE_SIZE,
                    local_kv_heads: config.key_value_heads(),
                    head_dimension: config.head_dim(),
                    element_bytes: 2, // GPT-OSS target K/V is BF16.
                },
                budget,
                safety,
            )
            .map_err(boxed)?
            .physical_pages(),
            None => profiles
                .iter()
                .map(|profile| {
                    profile
                        .max_sequence_tokens
                        .div_ceil(graph::CACHE_PAGE_SIZE)
                        .max(1)
                        .checked_next_power_of_two()
                        .ok_or_else(|| {
                            checkpoint::message("GPT-OSS cache profile overflows usize")
                        })
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .ok_or_else(|| checkpoint::message("GPT-OSS requires a compilation profile"))?,
        };
        let definition =
            ModelDefinition::declare(config, ParameterSet::new(registry), artifact_validation)?;
        let model = definition
            .compile(platform, profiles, physical_pages, serving.as_ref())?
            .make_resident()?;
        let startup = model.startup();
        tracing::info!(
            compiled_families = startup.compiled_families,
            prefill_compilation_seconds = startup.prefill_compilation.as_secs_f64(),
            decode_compilation_seconds = startup.decode_compilation.as_secs_f64(),
            "compiled finite GPT-OSS serving families before parameter residency"
        );
        Ok(Self {
            protocol,
            model,
            pages: PageManager::new(physical_pages).map_err(boxed)?,
            next_sequence: 1,
            stable_decode: StableDecodeLaneState::default(),
        })
    }

    pub(crate) fn generate(
        &mut self,
        prompt: String,
        max_new_tokens: usize,
        cache_capacity: Option<usize>,
        sampling: SamplingOptions,
        mut emit: impl FnMut(protocol::Event) -> Result<()>,
    ) -> Result<ProductReport> {
        let conversation = Conversation::new([
            Message::System(SystemContent::default()),
            Message::user(prompt),
        ]);
        // This compatibility/diagnostic adapter, rather than the device
        // session, owns Harmony translation. Server response tasks can drive
        // the same step API with their own independent parser.
        let tokenization_started = Instant::now();
        let tokens = self
            .protocol
            .render_for_completion(&conversation)
            .map_err(boxed)?;
        let tokenization = tokenization_started.elapsed();
        let mut parser = self.protocol.parser();
        let mut session = self.prepare_tokens(
            tokens,
            max_new_tokens,
            cache_capacity,
            sampling,
            tokenization,
        )?;
        let generation = (|| {
            if let Some(raw) = self.prefill_step(&mut session)? {
                emit_raw_token(&mut parser, raw, &mut emit)?;
            }
            while !session.is_complete() {
                let raw = self.decode_step(&mut session)?.ok_or_else(|| {
                    checkpoint::message("GPT-OSS decode step omitted its raw token")
                })?;
                emit_raw_token(&mut parser, raw, &mut emit)?;
            }
            let stopped = session.stopped();
            let events = if stopped {
                parser.finish().map_err(boxed)?;
                Vec::new()
            } else {
                parser.truncate().map_err(boxed)?
            };
            Ok::<_, BoxError>(events)
        })();
        let events = match generation {
            Ok(events) => events,
            Err(error) => {
                self.cancel(&mut session)?;
                return Err(error);
            }
        };
        let report = self.finalize(session)?;
        for event in events {
            emit(event)?;
        }
        Ok(report)
    }

    /// Returns a cheaply cloned protocol owner for bounded CPU preparation and
    /// response-side parsing outside the dedicated model/PJRT thread.
    #[allow(dead_code)] // Consumed by the Tokio preparation path in Milestone 1.
    pub(crate) fn protocol(&self) -> HarmonyProtocol {
        self.protocol.clone()
    }

    /// Prepares an already rendered/tokenized request for engine execution.
    /// Family selection is limited to the startup plan and never compiles.
    pub(crate) fn prepare_tokens(
        &mut self,
        tokens: Vec<u32>,
        max_new_tokens: usize,
        cache_capacity: Option<usize>,
        sampling: SamplingOptions,
        tokenization: Duration,
    ) -> Result<ProductSession> {
        let request = PreparedRequest::new(
            tokens,
            max_new_tokens,
            cache_capacity,
            sampling,
            self.model.config().context_limit(),
            tokenization,
        )?;
        let prompt_tokens = request.tokens.len();
        let tokenization = request.tokenization;
        let mut execution = self.model.prepare(request)?;
        let sequence = if max_new_tokens == 0 {
            None
        } else {
            let sequence = SequenceId::new(self.next_sequence);
            self.next_sequence = self
                .next_sequence
                .checked_add(1)
                .ok_or_else(|| checkpoint::message("GPT-OSS sequence ID space is exhausted"))?;
            self.pages
                .reserve_tokens(
                    sequence,
                    prompt_tokens.checked_add(max_new_tokens).ok_or_else(|| {
                        checkpoint::message("GPT-OSS cache reservation overflows usize")
                    })?,
                )
                .map_err(boxed)?;
            let mut installed_page_table = Vec::new();
            let installed = (|| {
                self.pages
                    .append_tentative(sequence, prompt_tokens)
                    .map_err(boxed)?;
                self.install_page_table(
                    sequence,
                    &mut execution,
                    &mut installed_page_table,
                )
            })();
            if let Err(error) = installed {
                self.pages.release_sequence(sequence).map_err(boxed)?;
                return Err(error);
            }
            Some((sequence, installed_page_table))
        };
        let (sequence, installed_page_table) = match sequence {
            Some((sequence, installed_page_table)) => {
                (Some(sequence), installed_page_table)
            }
            None => (None, Vec::new()),
        };
        Ok(ProductSession {
            execution,
            sequence,
            installed_page_table,
            prompt_tokens,
            tokenization,
            startup: self.model.startup(),
        })
    }

    pub(crate) fn prepare_server_tokens(
        &mut self,
        tokens: Vec<u32>,
        max_new_tokens: usize,
        sampling: SamplingOptions,
    ) -> Result<ServerSession> {
        if tokens.is_empty() {
            return Err(checkpoint::message("GPT-OSS server prompt is empty"));
        }
        let maximum = tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| checkpoint::message("GPT-OSS server token budget overflows usize"))?;
        if maximum > self.model.config().context_limit() {
            return Err(checkpoint::message("GPT-OSS server request exceeds context"));
        }
        let sequence = SequenceId::new(self.next_sequence);
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| checkpoint::message("GPT-OSS sequence ID space is exhausted"))?;
        self.pages
            .reserve_tokens(sequence, maximum)
            .map_err(boxed)?;
        Ok(ServerSession {
            sequence,
            prompt: tokens,
            prompt_cursor: 0,
            generated: Vec::with_capacity(max_new_tokens),
            max_new_tokens,
            sampling,
            stopped: false,
            released: false,
        })
    }

    pub(crate) fn prefill_batch(
        &mut self,
        sessions: &mut [&mut ServerSession],
        chunk_tokens: &[usize],
        batch_capacity: usize,
        query_capacity: usize,
    ) -> Result<Vec<Option<RawToken>>> {
        self.stable_decode.clear();
        if sessions.len() != chunk_tokens.len() || sessions.len() > batch_capacity {
            return Err(checkpoint::message("prefill batch row count is inconsistent"));
        }
        let table_width = self.model.batch_page_table_width(
            graph::Phase::Prefill,
            batch_capacity,
            query_capacity,
        )?;
        let mut checkpoints = Vec::with_capacity(sessions.len());
        for (session, chunk) in sessions.iter_mut().zip(chunk_tokens) {
            if session.released
                || session.prefill_complete()
                || *chunk == 0
                || *chunk > query_capacity
                || *chunk > session.prompt_remaining()
            {
                return Err(checkpoint::message("invalid prefill batch transition"));
            }
            checkpoints.push(self.pages.checkpoint(session.sequence).map_err(boxed)?);
            if let Err(error) = self
                .pages
                .append_tentative(session.sequence, *chunk)
                .map_err(boxed)
            {
                for checkpoint in checkpoints {
                    self.pages.rollback(checkpoint).map_err(boxed)?;
                }
                return Err(error);
            }
        }
        let execution = (|| {
            let rows = sessions
                .iter()
                .map(|session| Some(session.sequence))
                .chain(std::iter::repeat(None))
                .take(batch_capacity)
                .collect::<Vec<_>>();
            let metadata = self
                .pages
                .compact_metadata(&rows, table_width)
                .map_err(boxed)?;
            let mut input = padded_batch_inputs(batch_capacity, query_capacity, metadata.block_tables);
            input.sequence_lengths = metadata.sequence_lengths;
            input.active_rows = metadata.active_rows;
            for (row, (session, chunk)) in sessions.iter().zip(chunk_tokens).enumerate() {
                let start = session.prompt_cursor;
                let end = start + *chunk;
                for (destination, token) in input.tokens
                    [row * query_capacity..row * query_capacity + *chunk]
                    .iter_mut()
                    .zip(&session.prompt[start..end])
                {
                    *destination = i32::try_from(*token)
                        .map_err(|_| checkpoint::message("prompt token exceeds I32"))?;
                }
                input.positions[row] = i32::try_from(start)
                    .map_err(|_| checkpoint::message("prefill position exceeds I32"))?;
                input.query_lengths[row] = i32::try_from(*chunk)
                    .map_err(|_| checkpoint::message("prefill chunk exceeds I32"))?;
                input.last_indices[row] = input.query_lengths[row] - 1;
                input.sample_rows[row] = end == session.prompt.len();
                install_sampling_row(&mut input, row, session)?;
            }
            self.model.execute_batch(
                graph::Phase::Prefill,
                batch_capacity,
                query_capacity,
                input,
            )
        })();
        let outputs = match execution {
            Ok(outputs) => outputs,
            Err(error) => {
                for checkpoint in checkpoints {
                    self.pages.rollback(checkpoint).map_err(boxed)?;
                }
                return Err(error);
            }
        };
        let mut raw = Vec::with_capacity(sessions.len());
        for (row, (session, chunk)) in sessions.iter_mut().zip(chunk_tokens).enumerate() {
            self.pages
                .commit(session.sequence, *chunk)
                .map_err(boxed)?;
            session.prompt_cursor += *chunk;
            session.sampling.seed.copy_from_slice(
                &outputs.sampling_states[row * 2..row * 2 + 2],
            );
            if session.prefill_complete() {
                let token = u32::try_from(outputs.tokens[row]).map_err(|_| {
                    checkpoint::message("active prefill sampling returned the padding sentinel")
                })?;
                session.generated.push(token);
                let is_stop = protocol::is_stop_token(token);
                session.stopped = is_stop;
                raw.push(Some(RawToken { token, is_stop }));
            } else {
                if outputs.tokens[row] != -1 {
                    return Err(checkpoint::message(
                        "partial prefill row unexpectedly sampled a token",
                    ));
                }
                raw.push(None);
            }
        }
        Ok(raw)
    }

    pub(crate) fn decode_batch(
        &mut self,
        sessions: &mut [&mut ServerSession],
        batch_capacity: usize,
    ) -> Result<Vec<RawToken>> {
        if sessions.is_empty() || sessions.len() > batch_capacity {
            return Err(checkpoint::message("decode batch row count is inconsistent"));
        }
        let query_capacity = 1;
        let table_width = self.model.batch_page_table_width(
            graph::Phase::Decode,
            batch_capacity,
            query_capacity,
        )?;
        let mut checkpoints = Vec::with_capacity(sessions.len());
        for session in sessions.iter_mut() {
            if session.released || !session.prefill_complete() || session.is_complete() {
                return Err(checkpoint::message("invalid decode batch transition"));
            }
            checkpoints.push(self.pages.checkpoint(session.sequence).map_err(boxed)?);
            let (committed, tentative) = self
                .pages
                .sequence_lengths(session.sequence)
                .map_err(boxed)?;
            // The generic stable lane submits the next token's first five
            // layer pairs before waiting for this token's download. Reserve
            // one uncommitted position ahead so that prefix always has a
            // valid physical page, including at 16-token boundaries.
            let required = committed
                .checked_add(2)
                .ok_or_else(|| checkpoint::message("decode lookahead length overflows usize"))?;
            let extension = required.saturating_sub(tentative);
            if let Err(error) = self
                .pages
                .append_tentative(session.sequence, extension)
                .map_err(boxed)
            {
                for checkpoint in checkpoints {
                    self.pages.rollback(checkpoint).map_err(boxed)?;
                }
                return Err(error);
            }
        }
        let execution = (|| {
            let rows = sessions
                .iter()
                .map(|session| Some(session.sequence))
                .chain(std::iter::repeat(None))
                .take(batch_capacity)
                .collect::<Vec<_>>();
            let metadata = self
                .pages
                .compact_metadata(&rows, table_width)
                .map_err(boxed)?;
            let members = sessions
                .iter()
                .map(|session| session.sequence)
                .collect::<Vec<_>>();
            let descriptor = StableDecodeDescriptor {
                members,
                page_tables: metadata.block_tables,
            };
            if self.stable_decode.transition_to(&descriptor) != StableDecodeTransition::Reuse {
                let mut input =
                    padded_batch_inputs(
                        batch_capacity,
                        query_capacity,
                        descriptor.page_tables.clone(),
                    );
                input.sequence_lengths = metadata.sequence_lengths;
                input.active_rows = metadata.active_rows;
                input.sample_rows[..sessions.len()].fill(true);
                for (row, session) in sessions.iter().enumerate() {
                    let token = *session.generated.last().ok_or_else(|| {
                        checkpoint::message("decode row has no visible input token")
                    })?;
                    input.tokens[row] = i32::try_from(token)
                        .map_err(|_| checkpoint::message("decode token exceeds I32"))?;
                    let position = session
                        .prompt
                        .len()
                        .checked_add(session.generated.len() - 1)
                        .ok_or_else(|| checkpoint::message("decode position overflows usize"))?;
                    input.positions[row] = i32::try_from(position)
                        .map_err(|_| checkpoint::message("decode position exceeds I32"))?;
                    // Tentative page allocation may be ahead of the visible
                    // sequence. The resident slab still attends only through
                    // the current visible token and advances itself on device.
                    input.sequence_lengths[row] = input.positions[row]
                        .checked_add(1)
                        .ok_or_else(|| checkpoint::message("decode length exceeds I32"))?;
                    input.query_lengths[row] = 1;
                    input.last_indices[row] = 0;
                    install_sampling_row(&mut input, row, session)?;
                }
                let lane = self
                    .model
                    .prepare_stable_decode_batch(batch_capacity, input)?;
                self.stable_decode.install(descriptor, lane);
            }
            self.model.execute_stable_decode_batch(
                self.stable_decode
                    .lane_mut()
                    .expect("stable decode lane was prepared"),
            )
        })();
        let outputs = match execution {
            Ok(outputs) => outputs,
            Err(error) => {
                self.stable_decode.clear();
                for checkpoint in checkpoints {
                    self.pages.rollback(checkpoint).map_err(boxed)?;
                }
                return Err(error);
            }
        };
        let mut raw = Vec::with_capacity(sessions.len());
        for (row, session) in sessions.iter_mut().enumerate() {
            self.pages.commit(session.sequence, 1).map_err(boxed)?;
            session.sampling.seed.copy_from_slice(
                &outputs.sampling_states[row * 2..row * 2 + 2],
            );
            let token = u32::try_from(outputs.tokens[row]).map_err(|_| {
                checkpoint::message("active decode sampling returned the padding sentinel")
            })?;
            session.generated.push(token);
            let is_stop = protocol::is_stop_token(token);
            session.stopped = is_stop;
            raw.push(RawToken { token, is_stop });
        }
        Ok(raw)
    }

    pub(crate) fn release_server_session(&mut self, session: &mut ServerSession) -> Result<bool> {
        self.stable_decode.clear();
        if session.released {
            return Ok(false);
        }
        session.released = true;
        self.pages.release_sequence(session.sequence).map_err(boxed)
    }

    pub(crate) fn prefill_step(
        &mut self,
        session: &mut ProductSession,
    ) -> Result<Option<RawToken>> {
        let result = self.model.prefill_step(&mut session.execution);
        match result {
            Ok(raw) => {
                if let Some(sequence) = session.sequence {
                    if let Err(error) = self
                        .pages
                        .commit(sequence, session.prompt_tokens)
                        .map_err(boxed)
                    {
                        self.cancel(session)?;
                        return Err(error);
                    }
                }
                Ok(raw)
            }
            Err(error) => {
                self.cancel(session)?;
                Err(error)
            }
        }
    }

    pub(crate) fn decode_step(
        &mut self,
        session: &mut ProductSession,
    ) -> Result<Option<RawToken>> {
        let Some(sequence) = session.sequence else {
            return Err(checkpoint::message(
                "completed GPT-OSS session has no decode cache reservation",
            ));
        };
        let result = (|| {
            let lookahead = session.execution.decode_cache_lookahead_tokens()?;
            let (committed, tentative) =
                self.pages.sequence_lengths(sequence).map_err(boxed)?;
            let required = committed
                .checked_add(lookahead)
                .ok_or_else(|| checkpoint::message("GPT-OSS cache lookahead overflows usize"))?;
            if tentative < required {
                self.pages
                    .append_tentative(sequence, required - tentative)
                    .map_err(boxed)?;
            }
            self.install_page_table(
                sequence,
                &mut session.execution,
                &mut session.installed_page_table,
            )?;
            let raw = self.model.decode_step(&mut session.execution)?;
            self.pages.commit(sequence, 1).map_err(boxed)?;
            Ok::<_, BoxError>(raw)
        })();
        match result {
            Ok(raw) => Ok(raw),
            Err(error) => {
                self.cancel(session)?;
                Err(error)
            }
        }
    }

    pub(crate) fn finalize(&mut self, session: ProductSession) -> Result<ProductReport> {
        let sequence = session.sequence;
        let report = session.into_report();
        if let Some(sequence) = sequence {
            self.pages.release_sequence(sequence).map_err(boxed)?;
        }
        report
    }

    pub(crate) fn cancel(&mut self, session: &mut ProductSession) -> Result<()> {
        if let Some(sequence) = session.sequence.take() {
            self.pages.release_sequence(sequence).map_err(boxed)?;
        }
        Ok(())
    }

    fn install_page_table(
        &mut self,
        sequence: SequenceId,
        execution: &mut RequestExecution,
        installed: &mut Vec<i32>,
    ) -> Result<()> {
        let metadata = self
            .pages
            .compact_metadata(&[Some(sequence)], execution.page_table_width())
            .map_err(boxed)?;
        if *installed == metadata.block_tables {
            return Ok(());
        }
        self.model
            .install_page_table(execution, &metadata.block_tables)?;
        *installed = metadata.block_tables;
        Ok(())
    }
}

fn padded_batch_inputs(
    batch: usize,
    query: usize,
    page_tables: Vec<i32>,
) -> BatchInputs {
    BatchInputs {
        tokens: vec![0; batch * query],
        positions: vec![0; batch],
        sequence_lengths: vec![0; batch],
        query_lengths: vec![0; batch],
        active_rows: vec![false; batch],
        sample_rows: vec![false; batch],
        page_tables,
        last_indices: vec![0; batch],
        sampling_states: vec![0; batch * 2],
        top_k: vec![1; batch],
        temperature: vec![1.0; batch],
        top_p: vec![1.0; batch],
        min_p: vec![0.0; batch],
    }
}

fn install_sampling_row(
    input: &mut BatchInputs,
    row: usize,
    session: &ServerSession,
) -> Result<()> {
    input.sampling_states[row * 2..row * 2 + 2].copy_from_slice(&session.sampling.seed);
    input.top_k[row] = i32::try_from(session.sampling.top_k)
        .map_err(|_| checkpoint::message("sampling top-k exceeds I32"))?;
    input.temperature[row] = session.sampling.temperature;
    input.top_p[row] = session.sampling.top_p;
    input.min_p[row] = session.sampling.min_p;
    Ok(())
}

fn emit_raw_token(
    parser: &mut protocol::HarmonyParser,
    raw: RawToken,
    emit: &mut impl FnMut(protocol::Event) -> Result<()>,
) -> Result<()> {
    if raw.is_stop != protocol::is_stop_token(raw.token) {
        return Err(checkpoint::message(
            "GPT-OSS raw token stop classification is inconsistent",
        ));
    }
    for event in parser.process(raw.token).map_err(boxed)? {
        emit(event)?;
    }
    Ok(())
}

/// Parser-independent engine-facing session. It emits raw model tokens only;
/// Harmony parsing is owned by the blocking adapter or an async response task.
pub(crate) struct ProductSession {
    execution: RequestExecution,
    sequence: Option<SequenceId>,
    installed_page_table: Vec<i32>,
    prompt_tokens: usize,
    tokenization: Duration,
    startup: StartupMetrics,
}

impl ProductSession {
    pub(crate) fn is_complete(&self) -> bool {
        self.execution.is_complete()
    }

    pub(crate) fn stopped(&self) -> bool {
        self.execution.stopped()
    }

    fn into_report(self) -> Result<ProductReport> {
        let run = self.execution.finalize()?;
        let startup = self.startup;
        Ok(ProductReport {
            prompt_tokens: self.prompt_tokens,
            generated_tokens: run.generated_tokens,
            cache_capacity: run.cache_capacity,
            physical_parameter_components: startup.physical_parameter_components,
            parameter_source_bytes: startup.parameter_source_bytes,
            parameter_resident_bytes: startup.parameter_resident_bytes,
            parameter_prepared_bytes: startup.parameter_prepared_bytes,
            parameter_peak_staging_bytes: startup.parameter_peak_staging_bytes,
            cache_storage_bytes: run.cache_storage_bytes,
            cache_metadata_bytes: run.cache_metadata_bytes,
            cache_metadata_upload_bytes: run.cache_metadata_upload_bytes,
            timings: timings(self.tokenization, startup, run.metrics),
        })
    }
}

fn timings(tokenization: Duration, startup: StartupMetrics, run: RunMetrics) -> ProductTimings {
    ProductTimings {
        tokenization,
        artifact_validation: startup.artifact_validation,
        prefill_compilation: startup.prefill_compilation,
        decode_compilation: startup.decode_compilation,
        parameter_upload: startup.parameter_upload,
        cache_allocation: run.cache_allocation,
        cache_metadata_upload: run.cache_metadata_upload,
        prompt_upload: run.prompt_upload,
        prefill_execution: run.prefill_execution,
        prefill_download: run.prefill_download,
        decode_state_initialization: run.decode_state_initialization,
        first_decode_execution: run.first_decode_execution,
        steady_decode_execution: run.steady_decode_execution,
        decode_download: run.decode_download,
        prefill_submission: run.prefill_submission,
        first_decode_submission: run.first_decode_submission,
        steady_decode_submission: run.steady_decode_submission,
    }
}

fn validate_artifact(model_directory: &Path) -> std::result::Result<(), ArtifactError> {
    let manifest_path = model_directory.join(ARTIFACT_MANIFEST);
    let manifest_metadata = std::fs::symlink_metadata(&manifest_path)?;
    if !manifest_metadata.file_type().is_file() {
        return Err(ArtifactError(format!(
            "artifact manifest is not a regular file: {}",
            manifest_path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if manifest_metadata.mode() & 0o222 != 0 {
            return Err(ArtifactError(
                "artifact manifest is writable; rematerialize the artifact before launch"
                    .to_owned(),
            ));
        }
    }
    let manifest_hash = sha256(&manifest_path)?;
    if manifest_hash != ARTIFACT_MANIFEST_SHA256 {
        return Err(ArtifactError(format!(
            "artifact manifest SHA-256 is {manifest_hash}, expected {ARTIFACT_MANIFEST_SHA256}"
        )));
    }
    let manifest: ArtifactManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?)).map_err(|error| {
            ArtifactError(format!("artifact manifest is invalid JSON: {error}"))
        })?;
    validate_manifest_identity(&manifest)?;
    validate_checkpoint_entries(
        &manifest.files,
        model_directory.join(DIRECT_CHECKPOINT).is_file(),
    )?;
    let total = manifest.files.iter().try_fold(0u64, |total, file| {
        if !is_sha256(&file.sha256) {
            return Err(ArtifactError(format!(
                "artifact file {:?} has an invalid SHA-256",
                file.path
            )));
        }
        total
            .checked_add(file.size)
            .ok_or_else(|| ArtifactError("artifact byte count overflows u64".to_owned()))
    })?;
    if total != ARTIFACT_TOTAL_BYTES {
        return Err(ArtifactError(format!(
            "artifact files total {total} bytes, expected {ARTIFACT_TOTAL_BYTES}"
        )));
    }
    artifact::validate_materialization(
        model_directory,
        ARTIFACT_MANIFEST_SHA256,
        manifest.files.iter().map(|file| artifact::ExpectedFile {
            path: &file.path,
            size: file.size,
        }),
    )?;
    Ok(())
}

fn validate_manifest_identity(
    manifest: &ArtifactManifest,
) -> std::result::Result<(), ArtifactError> {
    let converter = &manifest.converter;
    if converter.name != CONVERTER_NAME
        || converter.version != CONVERTER_VERSION
        || converter.device != "cpu"
        || converter.python != "3.12.11"
        || converter.torch != "2.8.0+cpu"
        || converter.numpy != "2.2.6"
        || converter.safetensors != "0.6.2"
        || converter.huggingface_hub != "0.34.4"
        || converter.script_sha256 != CONVERTER_SCRIPT_SHA256
        || converter.requirements_sha256 != CONVERTER_REQUIREMENTS_SHA256
        || manifest.schema_version != 1
        || manifest.recipe != ARTIFACT_RECIPE
        || manifest.recipe_sha256 != ARTIFACT_RECIPE_SHA256
        || manifest.source_manifest_sha256 != SOURCE_MANIFEST_SHA256
        || manifest.source_repository != SOURCE_REPOSITORY
        || manifest.source_revision != SOURCE_REVISION
        || manifest.tensor_manifest_sha256 != TENSOR_MANIFEST_SHA256
        || manifest.files.len() != ARTIFACT_FILE_COUNT
    {
        return Err(ArtifactError(
            "artifact manifest identity does not match the selected GPT-OSS revision".to_owned(),
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_checkpoint_entries(
    files: &[ArtifactFile],
    direct_checkpoint_exists: bool,
) -> std::result::Result<(), ArtifactError> {
    if !files.iter().any(|file| file.path == CHECKPOINT_INDEX) {
        return Err(ArtifactError(
            "artifact manifest does not list model.safetensors.index.json".to_owned(),
        ));
    }
    if direct_checkpoint_exists && !files.iter().any(|file| file.path == DIRECT_CHECKPOINT) {
        return Err(ArtifactError(
            "artifact contains an unlisted model.safetensors".to_owned(),
        ));
    }
    Ok(())
}

fn sha256(path: &Path) -> std::result::Result<String, ArtifactError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactManifest {
    converter: ConverterManifest,
    schema_version: u32,
    recipe: String,
    recipe_sha256: String,
    source_manifest_sha256: String,
    source_repository: String,
    source_revision: String,
    tensor_manifest_sha256: String,
    files: Vec<ArtifactFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConverterManifest {
    device: String,
    huggingface_hub: String,
    name: String,
    numpy: String,
    python: String,
    requirements_sha256: String,
    safetensors: String,
    script_sha256: String,
    torch: String,
    version: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactFile {
    path: String,
    sha256: String,
    size: u64,
}

#[derive(Debug)]
struct ArtifactError(String);

impl From<std::io::Error> for ArtifactError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<artifact::Error> for ArtifactError {
    fn from(error: artifact::Error) -> Self {
        Self(error.to_string())
    }
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ArtifactError {}

fn boxed<E>(error: E) -> BoxError
where
    E: std::error::Error + Send + Sync + 'static,
{
    Box::new(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_manifest_matches_the_product_contract() {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        let path =
            Path::new(&runfiles).join("_main/artifacts/gpt-oss-20b-nvfp4/artifact-manifest.json");
        let manifest: ArtifactManifest =
            serde_json::from_reader(BufReader::new(File::open(path).unwrap())).unwrap();
        validate_manifest_identity(&manifest).unwrap();
    }

    fn entry(path: &str) -> ArtifactFile {
        ArtifactFile {
            path: path.to_owned(),
            sha256: "0".repeat(64),
            size: 0,
        }
    }

    #[test]
    fn checkpoint_selection_requires_the_manifest_index() {
        let error = validate_checkpoint_entries(&[], false).unwrap_err();
        assert_eq!(
            error.to_string(),
            "artifact manifest does not list model.safetensors.index.json",
        );
    }

    #[test]
    fn checkpoint_selection_rejects_an_unlisted_direct_file() {
        let error = validate_checkpoint_entries(&[entry(CHECKPOINT_INDEX)], true).unwrap_err();
        assert_eq!(
            error.to_string(),
            "artifact contains an unlisted model.safetensors",
        );
    }

    #[test]
    fn checkpoint_selection_accepts_the_listed_index_without_a_direct_file() {
        validate_checkpoint_entries(&[entry(CHECKPOINT_INDEX)], false).unwrap();
    }

    #[test]
    fn stable_decode_page_extension_preserves_lookahead() {
        let current = StableDecodeDescriptor {
            members: vec![SequenceId::new(1), SequenceId::new(2)],
            page_tables: vec![3, 7, -1, -1, 11, -1],
        };
        let replacement = StableDecodeDescriptor {
            members: vec![SequenceId::new(1), SequenceId::new(2)],
            page_tables: vec![3, 7, 19, -1, 11, 23],
        };
        assert_eq!(
            current.transition_to(&replacement),
            StableDecodeTransition::Rebind {
                preserve_lookahead: true,
            },
        );
    }

    #[test]
    fn stable_decode_transition_distinguishes_reuse_from_invalidation() {
        let current = StableDecodeDescriptor {
            members: vec![SequenceId::new(1)],
            page_tables: vec![3, 7, -1, 11],
        };
        let exact = StableDecodeDescriptor {
            members: vec![SequenceId::new(1)],
            page_tables: vec![3, 7, -1, 11],
        };
        assert_eq!(
            current.transition_to(&exact),
            StableDecodeTransition::Reuse,
        );
        for replacement in [
            StableDecodeDescriptor {
                members: vec![SequenceId::new(1)],
                page_tables: vec![3, 19, -1, 11],
            },
            StableDecodeDescriptor {
                members: vec![SequenceId::new(1)],
                page_tables: vec![3, 7, -1],
            },
            StableDecodeDescriptor {
                members: vec![SequenceId::new(2)],
                page_tables: vec![3, 7, -1, 11],
            },
        ] {
            assert_eq!(
                current.transition_to(&replacement),
                StableDecodeTransition::Rebind {
                    preserve_lookahead: false,
                },
            );
        }
        let allocated = StableDecodeDescriptor {
            members: vec![SequenceId::new(1)],
            page_tables: vec![3, 7, 19, 11],
        };
        assert_eq!(
            allocated.transition_to(&current),
            StableDecodeTransition::Rebind {
                preserve_lookahead: false,
            },
        );
    }
}
