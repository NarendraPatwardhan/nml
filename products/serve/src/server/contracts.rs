//! HTTP-independent contracts shared by request preparation and the engine.
//!
//! Configuration is normalized once at startup. A prepared request is then
//! constructed only after its exact rendered tokens, sampling controls, and
//! deadline have passed the same limits. None of the types in this module
//! contains an HTTP status, response schema, socket, or framework error.

// These cross-milestone contracts intentionally land before the Tokio engine,
// paged arena, and speculative scheduler consume every field. Their permanent
// tests exercise the staged invariants in the meantime.
#![allow(dead_code)]

use crate::{CompilationProfile, SamplingOptions};
use std::error::Error as StdError;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_BATCH_BUCKETS: usize = 16;
const MAX_PREFILL_QUERY_BUCKETS: usize = 16;
const MAX_COMPILATION_FAMILIES: usize = 32;

/// A process-unique request identifier.
///
/// The high half is fixed for the lifetime of the process and the low half is
/// monotonic. Its text form contains only printable ASCII and is safe to use in
/// logs or a public response identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(u128);

impl RequestId {
    pub fn as_u128(self) -> u128 {
        self.0
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "chatcmpl-{:032x}", self.0)
    }
}

/// The sole request-ID allocator for one server process.
pub(crate) struct RequestIds {
    process: u64,
    next: AtomicU64,
}

impl RequestIds {
    pub(crate) fn new() -> Self {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        Self {
            process: epoch ^ u64::from(std::process::id()).rotate_left(32),
            next: AtomicU64::new(1),
        }
    }

    pub(crate) fn next(&self) -> RequestId {
        let sequence = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .expect("request identifier space exhausted");
        RequestId((u128::from(self.process) << 64) | u128::from(sequence))
    }
}

/// Engine-private identity for one independently scheduled token sequence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct SequenceId(u64);

impl SequenceId {
    pub(crate) const fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Client-visible generation terminal categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Cancelled,
}

impl FinishReason {
    /// Stable semantic spelling. HTTP adapters may map it to their own schema.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Why an otherwise live request was cancelled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelReason {
    ClientDisconnect,
    ClientBackpressure,
    Deadline,
    Shutdown,
    Explicit,
}

impl CancelReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ClientDisconnect => "client_disconnect",
            Self::ClientBackpressure => "client_backpressure",
            Self::Deadline => "deadline",
            Self::Shutdown => "shutdown",
            Self::Explicit => "explicit",
        }
    }
}

/// Stable engine failure categories. These names are not HTTP status codes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineErrorCode {
    InvalidRequest,
    QueueFull,
    NotReady,
    Cancelled,
    DeadlineExceeded,
    ExecutionFailed,
    ShuttingDown,
}

impl EngineErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::QueueFull => "queue_full",
            Self::NotReady => "not_ready",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::ExecutionFailed => "execution_failed",
            Self::ShuttingDown => "shutting_down",
        }
    }
}

/// An engine failure with a stable machine category and diagnostic detail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineError {
    code: EngineErrorCode,
    message: String,
}

impl EngineError {
    pub(crate) fn new(code: EngineErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub const fn code(&self) -> EngineErrorCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(EngineErrorCode::InvalidRequest, message)
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code.as_str(), self.message)
    }
}

impl StdError for EngineError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeculationPolicy {
    Disabled,
    Auto,
}

/// An absolute monotonic deadline validated from a bounded relative timeout.
///
/// `Instant`, rather than wall-clock time, makes elapsed-time comparisons
/// immune to NTP and administrator clock changes. There is deliberately no
/// public constructor from an unchecked absolute value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RequestDeadline(Instant);

impl RequestDeadline {
    pub(crate) fn after(
        now: Instant,
        timeout: Duration,
        limits: &ServerLimits,
    ) -> Result<Self, EngineError> {
        if timeout.is_zero() {
            return Err(EngineError::invalid_request(
                "request deadline must be positive",
            ));
        }
        if timeout > limits.max_request_duration {
            return Err(EngineError::invalid_request(format!(
                "request deadline exceeds the {:?} server limit",
                limits.max_request_duration
            )));
        }
        now.checked_add(timeout).map(Self).ok_or_else(|| {
            EngineError::invalid_request("request deadline overflows the monotonic clock")
        })
    }

    pub(crate) const fn instant(self) -> Instant {
        self.0
    }

    pub(crate) fn is_expired(self, now: Instant) -> bool {
        now >= self.0
    }
}

/// Exact process-level resource and request limits.
#[derive(Clone, Debug)]
pub struct ServerLimits {
    pub max_body_bytes: usize,
    pub max_prompt_tokens: usize,
    pub max_completion_tokens: usize,
    pub max_context_tokens: usize,
    pub command_queue_capacity: usize,
    pub request_event_capacity: usize,
    pub max_queued_requests: usize,
    pub max_active_sequences: usize,
    pub preparation_concurrency: usize,
    pub max_prefix_cache_salt_bytes: usize,
    pub max_tools: usize,
    pub max_tool_name_bytes: usize,
    pub max_tool_description_bytes: usize,
    pub max_tool_schema_bytes: usize,
    pub max_tool_call_id_bytes: usize,
    pub max_tool_arguments_bytes: usize,
    pub max_tool_result_bytes: usize,
    pub admission_timeout: Duration,
    pub max_request_duration: Duration,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_body_bytes: 2 * 1024 * 1024,
            max_prompt_tokens: 32_768,
            max_completion_tokens: 8_192,
            max_context_tokens: 131_072,
            command_queue_capacity: 1_024,
            request_event_capacity: 64,
            max_queued_requests: 1_024,
            max_active_sequences: 32,
            preparation_concurrency: 32,
            max_prefix_cache_salt_bytes: 256,
            max_tools: 64,
            max_tool_name_bytes: 128,
            max_tool_description_bytes: 4 * 1024,
            max_tool_schema_bytes: 64 * 1024,
            max_tool_call_id_bytes: 256,
            max_tool_arguments_bytes: 64 * 1024,
            max_tool_result_bytes: 256 * 1024,
            admission_timeout: Duration::from_secs(5),
            max_request_duration: Duration::from_secs(30 * 60),
        }
    }
}

impl ServerLimits {
    pub(crate) fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("max_body_bytes", self.max_body_bytes),
            ("max_prompt_tokens", self.max_prompt_tokens),
            ("max_completion_tokens", self.max_completion_tokens),
            ("max_context_tokens", self.max_context_tokens),
            ("command_queue_capacity", self.command_queue_capacity),
            ("request_event_capacity", self.request_event_capacity),
            ("max_queued_requests", self.max_queued_requests),
            ("max_active_sequences", self.max_active_sequences),
            ("preparation_concurrency", self.preparation_concurrency),
            (
                "max_prefix_cache_salt_bytes",
                self.max_prefix_cache_salt_bytes,
            ),
            ("max_tools", self.max_tools),
            ("max_tool_name_bytes", self.max_tool_name_bytes),
            (
                "max_tool_description_bytes",
                self.max_tool_description_bytes,
            ),
            ("max_tool_schema_bytes", self.max_tool_schema_bytes),
            ("max_tool_call_id_bytes", self.max_tool_call_id_bytes),
            ("max_tool_arguments_bytes", self.max_tool_arguments_bytes),
            ("max_tool_result_bytes", self.max_tool_result_bytes),
        ] {
            if value == 0 {
                return Err(format!("server limit {name} must be nonzero"));
            }
        }
        if self.max_prompt_tokens > self.max_context_tokens
            || self.max_completion_tokens > self.max_context_tokens
        {
            return Err("prompt/completion limits exceed the context limit".to_owned());
        }
        if self.admission_timeout.is_zero() {
            return Err("admission timeout must be nonzero".to_owned());
        }
        if self.max_request_duration.is_zero() {
            return Err("maximum request duration must be nonzero".to_owned());
        }
        if self.admission_timeout > self.max_request_duration {
            return Err("admission timeout exceeds maximum request duration".to_owned());
        }
        Ok(())
    }
}

/// Startup-time model execution and memory policy.
#[derive(Clone, Debug)]
pub struct ServerProfile {
    pub compilation_families: Vec<CompilationProfile>,
    pub batch_buckets: Vec<usize>,
    pub prefill_query_buckets: Vec<usize>,
    pub max_model_length: usize,
    pub max_batched_tokens: usize,
    pub max_prefill_chunk: usize,
    pub max_prefill_wait: Duration,
    pub cache_budget_bytes: usize,
    pub cache_safety_bytes: usize,
    pub tensor_parallel: usize,
    pub speculation: SpeculationPolicy,
}

impl ServerProfile {
    pub fn a40(prefill_capacity: usize, cache_capacity: usize) -> Self {
        Self {
            compilation_families: vec![CompilationProfile {
                max_prompt_tokens: prefill_capacity,
                max_sequence_tokens: cache_capacity,
            }],
            batch_buckets: vec![1, 2, 4, 8, 16, 32],
            prefill_query_buckets: vec![16, 64, 256],
            max_model_length: cache_capacity,
            max_batched_tokens: 4_096,
            max_prefill_chunk: 256,
            max_prefill_wait: Duration::from_millis(10),
            cache_budget_bytes: 8 * 1024 * 1024 * 1024,
            cache_safety_bytes: 512 * 1024 * 1024,
            tensor_parallel: 1,
            speculation: SpeculationPolicy::Disabled,
        }
    }

    /// Normalizes finite family lists and rejects an internally inconsistent
    /// profile before compilation, residency, cache allocation, or admission.
    pub(crate) fn validate(&mut self, limits: &ServerLimits) -> Result<(), String> {
        if self.compilation_families.is_empty() {
            return Err("server profile requires at least one compilation family".to_owned());
        }
        normalize_buckets(
            "batch",
            &mut self.batch_buckets,
            MAX_BATCH_BUCKETS,
            true,
        )?;
        normalize_buckets(
            "prefill query",
            &mut self.prefill_query_buckets,
            MAX_PREFILL_QUERY_BUCKETS,
            true,
        )?;

        let largest_batch = self.batch_buckets.last().copied().unwrap_or(0);
        if largest_batch > limits.max_active_sequences {
            return Err("largest batch bucket exceeds maximum active sequences".to_owned());
        }
        if self.max_model_length == 0 || self.max_model_length > limits.max_context_tokens {
            return Err("maximum model length exceeds the validated context limit".to_owned());
        }
        if self.max_batched_tokens == 0 || self.max_prefill_chunk == 0 {
            return Err("batched-token and prefill-chunk budgets must be nonzero".to_owned());
        }
        if self.max_prefill_wait.is_zero() {
            return Err("maximum prefill wait must be nonzero".to_owned());
        }
        if largest_batch > self.max_batched_tokens {
            return Err("largest decode batch exceeds the per-iteration token budget".to_owned());
        }
        if self.max_prefill_chunk > self.max_batched_tokens
            || self.max_prefill_chunk > self.max_model_length
        {
            return Err("prefill chunk exceeds the model or per-iteration token budget".to_owned());
        }
        if self
            .prefill_query_buckets
            .last()
            .is_some_and(|bucket| *bucket < self.max_prefill_chunk)
        {
            return Err("prefill query buckets do not cover the maximum prefill chunk".to_owned());
        }
        if self.cache_budget_bytes == 0 || self.cache_safety_bytes >= self.cache_budget_bytes {
            return Err("cache budget must exceed its safety reserve".to_owned());
        }
        if !matches!(self.tensor_parallel, 1 | 2 | 4) {
            return Err("tensor parallel degree must be exactly 1, 2, or 4".to_owned());
        }

        self.compilation_families.sort_by_key(|profile| {
            (profile.max_sequence_tokens, profile.max_prompt_tokens)
        });
        self.compilation_families.dedup();
        if self.compilation_families.len() > MAX_COMPILATION_FAMILIES {
            return Err(format!(
                "server profile has more than {MAX_COMPILATION_FAMILIES} compilation families"
            ));
        }
        if self.compilation_families.iter().any(|profile| {
            profile.max_prompt_tokens == 0
                || profile.max_sequence_tokens == 0
                || profile.max_prompt_tokens > profile.max_sequence_tokens
                || profile.max_prompt_tokens > limits.max_prompt_tokens
                || profile.max_sequence_tokens > self.max_model_length
        }) {
            return Err("compilation family exceeds the validated server profile".to_owned());
        }
        if self
            .compilation_families
            .iter()
            .all(|family| family.max_sequence_tokens < self.max_model_length)
        {
            return Err("no compilation family covers the maximum model length".to_owned());
        }
        Ok(())
    }

    fn supports(&self, prompt_tokens: usize, total_tokens: usize) -> bool {
        self.compilation_families.iter().any(|family| {
            prompt_tokens <= family.max_prompt_tokens
                && total_tokens <= family.max_sequence_tokens
        })
    }
}

fn normalize_buckets(
    name: &str,
    buckets: &mut Vec<usize>,
    maximum_count: usize,
    require_power_of_two: bool,
) -> Result<(), String> {
    if buckets.is_empty() || buckets.iter().any(|bucket| *bucket == 0) {
        return Err(format!("{name} buckets must be nonempty and nonzero"));
    }
    if require_power_of_two && buckets.iter().any(|bucket| !bucket.is_power_of_two()) {
        return Err(format!("{name} buckets must be powers of two"));
    }
    buckets.sort_unstable();
    buckets.dedup();
    if buckets.len() > maximum_count {
        return Err(format!(
            "{name} has more than {maximum_count} distinct buckets"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Consumed by the server CLI/engine startup in Milestone 1.
pub enum Backend {
    Cpu,
    Cuda,
}

#[derive(Clone, Debug)]
#[allow(dead_code)] // Consumed by the server CLI/engine startup in Milestone 1.
pub struct ServerConfig {
    pub bind: std::net::SocketAddr,
    pub model: PathBuf,
    pub backend: Backend,
    pub profile: ServerProfile,
    pub limits: ServerLimits,
    pub shutdown_grace: Duration,
}

impl ServerConfig {
    #[allow(dead_code)] // Consumed by the server CLI/engine startup in Milestone 1.
    pub fn validate(&mut self) -> Result<(), String> {
        self.limits.validate()?;
        self.profile.validate(&self.limits)?;
        if self.model.as_os_str().is_empty() {
            return Err("model directory must not be empty".to_owned());
        }
        if self.shutdown_grace.is_zero() {
            return Err("shutdown grace must be nonzero".to_owned());
        }
        Ok(())
    }
}

/// Immutable, post-tokenization input accepted by the engine.
///
/// Construction validates every size and profile boundary before the engine
/// can reserve pages or enqueue device work. GPT-OSS stop-token semantics stay
/// owned by its protocol/executor rather than being supplied by the client.
#[derive(Clone, Debug)]
pub(crate) struct PreparedInferenceRequest {
    id: RequestId,
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    sampling: SamplingOptions,
    deadline: Option<RequestDeadline>,
    prefix_cache_salt: Option<Vec<u8>>,
    speculation: SpeculationPolicy,
    request_body_bytes: usize,
    rendered_prompt_bytes: usize,
}

impl PreparedInferenceRequest {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: RequestId,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        sampling: SamplingOptions,
        deadline: Option<RequestDeadline>,
        prefix_cache_salt: Option<Vec<u8>>,
        speculation: SpeculationPolicy,
        request_body_bytes: usize,
        rendered_prompt_bytes: usize,
        limits: &ServerLimits,
        profile: &ServerProfile,
        now: Instant,
    ) -> Result<Self, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::invalid_request(
                "rendered prompt must contain at least one token",
            ));
        }
        if prompt_tokens.len() > limits.max_prompt_tokens {
            return Err(EngineError::invalid_request(
                "rendered prompt exceeds the prompt-token limit",
            ));
        }
        if max_new_tokens > limits.max_completion_tokens {
            return Err(EngineError::invalid_request(
                "completion exceeds the completion-token limit",
            ));
        }
        if request_body_bytes > limits.max_body_bytes {
            return Err(EngineError::invalid_request(
                "request exceeds the request-body byte limit",
            ));
        }
        let total_tokens = prompt_tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| EngineError::invalid_request("request token count overflows usize"))?;
        if total_tokens > limits.max_context_tokens || total_tokens > profile.max_model_length {
            return Err(EngineError::invalid_request(
                "prompt and completion exceed the model context limit",
            ));
        }
        if !profile.supports(prompt_tokens.len(), total_tokens) {
            return Err(EngineError::invalid_request(
                "request is not covered by a compiled execution family",
            ));
        }
        validate_sampling(sampling)?;
        if deadline.is_some_and(|deadline| deadline.is_expired(now)) {
            return Err(EngineError::new(
                EngineErrorCode::DeadlineExceeded,
                "request deadline elapsed before admission",
            ));
        }
        let prefix_cache_salt = prefix_cache_salt.filter(|salt| !salt.is_empty());
        if prefix_cache_salt
            .as_ref()
            .is_some_and(|salt| salt.len() > limits.max_prefix_cache_salt_bytes)
        {
            return Err(EngineError::invalid_request(
                "prefix cache salt exceeds the byte limit",
            ));
        }
        if speculation == SpeculationPolicy::Auto
            && profile.speculation == SpeculationPolicy::Disabled
        {
            return Err(EngineError::invalid_request(
                "speculative decoding is disabled by the server profile",
            ));
        }

        Ok(Self {
            id,
            prompt_tokens,
            max_new_tokens,
            sampling,
            deadline,
            prefix_cache_salt,
            speculation,
            request_body_bytes,
            rendered_prompt_bytes,
        })
    }

    pub(crate) const fn id(&self) -> RequestId {
        self.id
    }

    pub(crate) fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    pub(crate) const fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    pub(crate) const fn sampling(&self) -> SamplingOptions {
        self.sampling
    }

    pub(crate) const fn deadline(&self) -> Option<RequestDeadline> {
        self.deadline
    }

    pub(crate) fn prefix_cache_salt(&self) -> Option<&[u8]> {
        self.prefix_cache_salt.as_deref()
    }

    pub(crate) const fn speculation(&self) -> SpeculationPolicy {
        self.speculation
    }

    pub(crate) const fn request_body_bytes(&self) -> usize {
        self.request_body_bytes
    }

    pub(crate) const fn rendered_prompt_bytes(&self) -> usize {
        self.rendered_prompt_bytes
    }

    pub(crate) fn total_token_budget(&self) -> usize {
        // The constructor proved this addition cannot overflow.
        self.prompt_tokens.len() + self.max_new_tokens
    }
}

fn validate_sampling(sampling: SamplingOptions) -> Result<(), EngineError> {
    if !sampling.temperature.is_finite() || sampling.temperature <= 0.0 {
        return Err(EngineError::invalid_request(
            "temperature must be finite and positive",
        ));
    }
    if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
        return Err(EngineError::invalid_request("top_p must be in (0, 1]"));
    }
    if sampling.top_k == 0 || sampling.top_k > 64 {
        return Err(EngineError::invalid_request(
            "top_k must be between 1 and 64",
        ));
    }
    if !sampling.min_p.is_finite() || !(0.0..=1.0).contains(&sampling.min_p) {
        return Err(EngineError::invalid_request("min_p must be in [0, 1]"));
    }
    Ok(())
}

/// The first terminal cause wins even if cancellation and execution complete
/// concurrently at the engine boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalOutcome {
    Finished(FinishReason),
    Cancelled(CancelReason),
    Failed(EngineErrorCode),
}

/// Resource ownership held immediately before a terminal transition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TerminalOwnership {
    pub(crate) response: bool,
    pub(crate) target_cache: bool,
    pub(crate) draft_cache: bool,
    pub(crate) admission: bool,
    pub(crate) scheduler: bool,
}

/// Pure actions returned to the impure scheduler by a terminal transition.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct TerminalRelease {
    /// `Some` exactly once; the caller emits this before releasing `response`.
    pub(crate) emit: Option<TerminalOutcome>,
    pub(crate) response: bool,
    pub(crate) target_cache: bool,
    pub(crate) draft_cache: bool,
    pub(crate) admission: bool,
    pub(crate) scheduler: bool,
}

impl TerminalRelease {
    pub(crate) const fn is_noop(self) -> bool {
        self.emit.is_none()
            && !self.response
            && !self.target_cache
            && !self.draft_cache
            && !self.admission
            && !self.scheduler
    }
}

/// Idempotent terminal-state and resource-release ledger.
///
/// This type deliberately owns booleans rather than response senders, page
/// tables, or admission permits. The scheduler applies the returned actions in
/// order (emit, cache releases, admission/scheduler release, response drop),
/// while this pure transition makes duplicated calls harmless.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TerminalState {
    outcome: Option<TerminalOutcome>,
    ownership: TerminalOwnership,
}

impl TerminalState {
    pub(crate) const fn new(ownership: TerminalOwnership) -> Self {
        Self {
            outcome: None,
            ownership,
        }
    }

    pub(crate) const fn outcome(self) -> Option<TerminalOutcome> {
        self.outcome
    }

    pub(crate) fn finalize(&mut self, outcome: TerminalOutcome) -> TerminalRelease {
        if self.outcome.is_some() {
            return TerminalRelease::default();
        }
        self.outcome = Some(outcome);
        let ownership = std::mem::take(&mut self.ownership);
        TerminalRelease {
            emit: ownership.response.then_some(outcome),
            response: ownership.response,
            target_cache: ownership.target_cache,
            draft_cache: ownership.draft_cache,
            admission: ownership.admission,
            scheduler: ownership.scheduler,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_a40_control_profile_is_exact_and_has_a_promotion_floor() {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        let path = PathBuf::from(runfiles)
            .join("_main/products/serve/profiles/a40-single-stream-control-v1.json");
        let profile: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
        assert_eq!(profile["name"], "a40-single-stream-control-v1");
        assert_eq!(profile["source_commit"], "fb415a8dadd51a0053b9be314faa836e2b274721");
        assert_eq!(profile["workload"]["expected_prompt_tokens"], 106);
        assert_eq!(profile["workload"]["max_new_tokens"], 320);
        assert_eq!(profile["workload"]["prefill_capacity"], 256);
        assert_eq!(profile["workload"]["cache_capacity"], 512);
        assert_eq!(
            profile["accepted_baseline"]["decode_loop_tokens_per_second"],
            151.324,
        );
        assert_eq!(
            profile["promotion_floor"]["decode_loop_tokens_per_second"],
            150.0,
        );
    }

    #[test]
    fn profiles_are_finite_bounded_deduplicated_and_validated() {
        let limits = ServerLimits::default();
        let mut profile = ServerProfile::a40(4_096, 8_192);
        profile.batch_buckets.extend([8, 1]);
        profile.prefill_query_buckets.extend([64, 16]);
        profile
            .compilation_families
            .push(profile.compilation_families[0]);
        profile.validate(&limits).unwrap();
        assert_eq!(profile.batch_buckets, [1, 2, 4, 8, 16, 32]);
        assert_eq!(profile.prefill_query_buckets, [16, 64, 256]);
        assert_eq!(profile.compilation_families.len(), 1);

        for degree in [1, 2, 4] {
            let mut candidate = profile.clone();
            candidate.tensor_parallel = degree;
            candidate.validate(&limits).unwrap();
        }
        for degree in [0, 3, 8, 16] {
            let mut candidate = profile.clone();
            candidate.tensor_parallel = degree;
            assert!(candidate.validate(&limits).is_err());
        }

        let mut too_many_families = profile;
        too_many_families.compilation_families = (1..=MAX_COMPILATION_FAMILIES + 1)
            .map(|max_prompt_tokens| CompilationProfile {
                max_prompt_tokens,
                max_sequence_tokens: 8_192,
            })
            .collect();
        assert!(too_many_families.validate(&limits).is_err());
    }

    #[test]
    fn limits_and_deadlines_reject_zero_unbounded_and_overflowing_values() {
        let limits = ServerLimits::default();
        limits.validate().unwrap();
        let now = Instant::now();
        let deadline = RequestDeadline::after(now, Duration::from_secs(1), &limits).unwrap();
        assert!(!deadline.is_expired(now));
        assert!(deadline.is_expired(deadline.instant()));
        assert!(RequestDeadline::after(now, Duration::ZERO, &limits).is_err());
        assert!(RequestDeadline::after(
            now,
            limits.max_request_duration + Duration::from_nanos(1),
            &limits
        )
        .is_err());

        let mut invalid = limits.clone();
        invalid.request_event_capacity = 0;
        assert!(invalid.validate().is_err());
        invalid = limits.clone();
        invalid.admission_timeout = invalid.max_request_duration + Duration::from_nanos(1);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn prepared_requests_are_bounded_before_engine_admission() {
        let limits = ServerLimits::default();
        let mut profile = ServerProfile::a40(4_096, 8_192);
        profile.validate(&limits).unwrap();
        let now = Instant::now();
        let id = RequestIds::new().next();
        let request = PreparedInferenceRequest::new(
            id,
            vec![7; 128],
            64,
            SamplingOptions::default(),
            None,
            Some(Vec::new()),
            SpeculationPolicy::Disabled,
            512,
            768,
            &limits,
            &profile,
            now,
        )
        .unwrap();
        assert_eq!(request.id(), id);
        assert_eq!(request.prompt_tokens().len(), 128);
        assert_eq!(request.max_new_tokens(), 64);
        assert_eq!(request.sampling(), SamplingOptions::default());
        assert_eq!(request.deadline(), None);
        assert_eq!(request.prefix_cache_salt(), None);
        assert_eq!(request.speculation(), SpeculationPolicy::Disabled);
        assert_eq!(request.request_body_bytes(), 512);
        assert_eq!(request.rendered_prompt_bytes(), 768);
        assert_eq!(request.total_token_budget(), 192);

        let error = PreparedInferenceRequest::new(
            id,
            vec![7],
            1,
            SamplingOptions::default(),
            None,
            None,
            SpeculationPolicy::Disabled,
            limits.max_body_bytes + 1,
            1,
            &limits,
            &profile,
            now,
        )
        .unwrap_err();
        assert_eq!(error.code(), EngineErrorCode::InvalidRequest);

        let error = PreparedInferenceRequest::new(
            id,
            vec![7],
            1,
            SamplingOptions::default(),
            None,
            None,
            SpeculationPolicy::Auto,
            1,
            1,
            &limits,
            &profile,
            now,
        )
        .unwrap_err();
        assert_eq!(error.code(), EngineErrorCode::InvalidRequest);

        let error = PreparedInferenceRequest::new(
            id,
            vec![7; 4_096],
            4_097,
            SamplingOptions::default(),
            None,
            None,
            SpeculationPolicy::Auto,
            0,
            0,
            &limits,
            &profile,
            now,
        )
        .unwrap_err();
        assert_eq!(error.code(), EngineErrorCode::InvalidRequest);

        let mut overflow_limits = limits.clone();
        overflow_limits.max_completion_tokens = usize::MAX;
        overflow_limits.max_context_tokens = usize::MAX;
        let mut overflow_profile = profile.clone();
        overflow_profile.max_model_length = usize::MAX;
        overflow_profile.compilation_families = vec![CompilationProfile {
            max_prompt_tokens: 1,
            max_sequence_tokens: usize::MAX,
        }];
        let error = PreparedInferenceRequest::new(
            id,
            vec![7],
            usize::MAX,
            SamplingOptions::default(),
            None,
            None,
            SpeculationPolicy::Disabled,
            1,
            1,
            &overflow_limits,
            &overflow_profile,
            now,
        )
        .unwrap_err();
        assert_eq!(error.code(), EngineErrorCode::InvalidRequest);

        let elapsed = RequestDeadline::after(now, Duration::from_nanos(1), &limits).unwrap();
        let error = PreparedInferenceRequest::new(
            id,
            vec![7],
            0,
            SamplingOptions::default(),
            Some(elapsed),
            None,
            SpeculationPolicy::Disabled,
            1,
            1,
            &limits,
            &profile,
            elapsed.instant(),
        )
        .unwrap_err();
        assert_eq!(error.code(), EngineErrorCode::DeadlineExceeded);
    }

    #[test]
    fn terminal_transition_emits_and_releases_each_owner_exactly_once() {
        let ownership = TerminalOwnership {
            response: true,
            target_cache: true,
            draft_cache: true,
            admission: true,
            scheduler: true,
        };
        let expected = TerminalOutcome::Finished(FinishReason::Stop);
        let mut terminal = TerminalState::new(ownership);

        let first = terminal.finalize(expected);
        assert_eq!(first.emit, Some(expected));
        assert!(first.response);
        assert!(first.target_cache);
        assert!(first.draft_cache);
        assert!(first.admission);
        assert!(first.scheduler);

        let second = terminal.finalize(TerminalOutcome::Cancelled(CancelReason::Shutdown));
        let third = terminal.finalize(TerminalOutcome::Failed(EngineErrorCode::ExecutionFailed));
        assert!(second.is_noop());
        assert!(third.is_noop());
        assert_eq!(terminal.outcome(), Some(expected));

        let mut disconnected = TerminalState::new(TerminalOwnership {
            response: false,
            target_cache: true,
            draft_cache: false,
            admission: true,
            scheduler: true,
        });
        let release = disconnected.finalize(TerminalOutcome::Cancelled(
            CancelReason::ClientDisconnect,
        ));
        assert_eq!(release.emit, None);
        assert!(!release.response);
        assert!(release.target_cache);
        assert!(!release.draft_cache);
        assert!(release.admission);
        assert!(release.scheduler);
    }

    #[test]
    fn stable_semantic_names_do_not_depend_on_http_types() {
        assert_eq!(
            [
                FinishReason::Stop.as_str(),
                FinishReason::Length.as_str(),
                FinishReason::ToolCalls.as_str(),
                FinishReason::Cancelled.as_str(),
            ],
            ["stop", "length", "tool_calls", "cancelled"]
        );
        assert_eq!(
            [
                CancelReason::ClientDisconnect.as_str(),
                CancelReason::ClientBackpressure.as_str(),
                CancelReason::Deadline.as_str(),
                CancelReason::Shutdown.as_str(),
                CancelReason::Explicit.as_str(),
            ],
            [
                "client_disconnect",
                "client_backpressure",
                "deadline",
                "shutdown",
                "explicit",
            ]
        );
        assert_eq!(
            [
                EngineErrorCode::InvalidRequest.as_str(),
                EngineErrorCode::QueueFull.as_str(),
                EngineErrorCode::NotReady.as_str(),
                EngineErrorCode::Cancelled.as_str(),
                EngineErrorCode::DeadlineExceeded.as_str(),
                EngineErrorCode::ExecutionFailed.as_str(),
                EngineErrorCode::ShuttingDown.as_str(),
            ],
            [
                "invalid_request",
                "queue_full",
                "not_ready",
                "cancelled",
                "deadline_exceeded",
                "execution_failed",
                "shutting_down",
            ]
        );
        let error = EngineError::new(EngineErrorCode::NotReady, "warming");
        assert_eq!(error.code(), EngineErrorCode::NotReady);
        assert_eq!(error.message(), "warming");
        assert_eq!(error.to_string(), "not_ready: warming");
    }

    #[test]
    fn identifiers_are_monotonic_and_display_safe() {
        let ids = RequestIds::new();
        let first = ids.next();
        let second = ids.next();
        assert!(first < second);
        assert_eq!(first.as_u128() + 1, second.as_u128());
        assert!(first.to_string().starts_with("chatcmpl-"));
        assert_eq!(first.to_string().len(), 41);

        let sequence = SequenceId::new(42);
        assert_eq!(sequence.as_u64(), 42);
    }
}
