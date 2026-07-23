//! Axum lifecycle and OpenAI-shaped chat boundary.

use super::api::{
    ChatChoice, ChatChunk, ChatCompletionRequest, ChatCompletionResponse, ChunkChoice, ChunkDelta,
    ErrorBody, ErrorEnvelope, ResponseFunctionCall, ResponseMessage, ResponseToolCall, ToolKind,
    Usage,
};
use super::contracts::{
    CancelReason, EngineError, EngineErrorCode, PreparedInferenceRequest, RequestDeadline,
    RequestId, RequestIds, ServerConfig, SpeculationPolicy,
};
use super::engine::{self, EngineEvent, EngineHandle, Readiness};
use super::metrics::Metrics;
use crate::gpt_oss::protocol::{
    Channel, Event as HarmonyEvent, HarmonyParser, StopReason,
};
use crate::{Error, MODEL_NAME};
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::error_handling::HandleErrorLayer;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use std::convert::Infallible;
use std::future::IntoFuture;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;
use tower::{BoxError as TowerError, ServiceBuilder};

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
    config: Arc<ServerConfig>,
    request_ids: Arc<RequestIds>,
    preparation: Arc<Semaphore>,
    metrics: Metrics,
}

pub(crate) struct Server {
    config: ServerConfig,
    owner: super::engine::EngineOwner,
    metrics: Metrics,
}

impl Server {
pub(crate) fn start(
    mut config: ServerConfig,
    platform: impl FnOnce() -> Result<nml::Platform, Error> + Send + 'static,
) -> Result<Self, Error> {
    config
        .validate()
        .map_err(|error| Box::new(ConfigurationError(error)) as Error)?;
    let metrics = Metrics::new();
    let owner = engine::spawn(
        config.model.clone(),
        config.profile.clone(),
        config.limits.command_queue_capacity,
        config.limits.request_event_capacity,
        config.limits.max_queued_requests,
        config.limits.max_active_sequences,
        config.limits.admission_timeout,
        metrics.clone(),
        platform,
    )
    .map_err(|error| Box::new(error) as Error)?;
    Ok(Self {
        config,
        owner,
        metrics,
    })
}

pub(crate) async fn run(self) -> Result<(), Error> {
    let Self {
        config,
        owner,
        metrics,
    } = self;
    let state = AppState {
        engine: owner.handle.clone(),
        config: Arc::new(config.clone()),
        request_ids: Arc::new(RequestIds::new()),
        preparation: Arc::new(Semaphore::new(config.limits.preparation_concurrency)),
        metrics,
    };
    let maximum_in_flight = config
        .limits
        .max_queued_requests
        .checked_add(config.limits.preparation_concurrency)
        .ok_or_else(|| {
            Box::new(ConfigurationError(
                "HTTP concurrency limit overflows usize".to_owned(),
            )) as Error
        })?;
    let router = router(state, maximum_in_flight);
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    tracing::info!(bind = %config.bind, model = MODEL_NAME, "inference server listening");
    let shutdown = CancellationToken::new();
    let signal_shutdown = shutdown.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        signal_shutdown.cancel();
    });
    let server = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown.clone().cancelled_owned())
        .into_future();
    tokio::pin!(server);
    let server_result = tokio::select! {
        result = &mut server => result,
        () = shutdown.cancelled() => {
            match tokio::time::timeout(config.shutdown_grace, &mut server).await {
                Ok(result) => result,
                Err(_) => {
                    let shutdown_result = owner.shutdown().await;
                    shutdown_result.map_err(|error| Box::new(error) as Error)?;
                    return Err(Box::new(ConfigurationError(
                        "graceful HTTP shutdown exceeded its deadline".to_owned(),
                    )) as Error);
                }
            }
        }
    };
    let shutdown_result = owner.shutdown().await;
    server_result.map_err(|error| Box::new(error) as Error)?;
    shutdown_result.map_err(|error| Box::new(error) as Error)?;
    Ok(())
}
}

fn router(state: AppState, maximum_in_flight: usize) -> Router {
    let body_limit = state.config.limits.max_body_bytes;
    let middleware = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(middleware_error))
        .load_shed()
        .concurrency_limit(maximum_in_flight);
    Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(ready))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(middleware)
        .with_state(state)
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn middleware_error(_error: TowerError) -> Response {
    ApiError::overloaded("the HTTP admission boundary is saturated").into_response()
}

async fn ready(State(state): State<AppState>) -> Response {
    match state.engine.readiness() {
        Readiness::Ready => (StatusCode::OK, "ready\n").into_response(),
        Readiness::Starting => (StatusCode::SERVICE_UNAVAILABLE, "starting\n").into_response(),
        Readiness::Failed => (StatusCode::SERVICE_UNAVAILABLE, "startup_failed\n").into_response(),
        Readiness::ShuttingDown | Readiness::Stopped => {
            (StatusCode::SERVICE_UNAVAILABLE, "stopping\n").into_response()
        }
    }
}

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    if let Some(snapshot) = state.engine.snapshot().await {
        state.metrics.active.set(snapshot.active as i64);
        state.metrics.queued.set(snapshot.queued as i64);
        state
            .metrics
            .cache_total_pages
            .set(snapshot.cache_total_pages as i64);
        state
            .metrics
            .cache_free_pages
            .set(snapshot.cache_free_pages as i64);
        state
            .metrics
            .cache_reserved_pages
            .set(snapshot.cache_reserved_pages as i64);
    }
    match state.metrics.render() {
        Ok(body) => (
            [(header::CONTENT_TYPE, "application/openmetrics-text; version=1.0.0; charset=utf-8")],
            body,
        )
            .into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

async fn models() -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "nml"
        }]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    state.metrics.received.inc();
    match prepare_and_submit(&state, body, Instant::now()).await {
        Ok(submitted) if submitted.stream => stream_response(submitted, state.metrics).into_response(),
        Ok(submitted) => non_stream_response(submitted, state.metrics).await,
        Err(error) => error.into_response(),
    }
}

struct Submitted {
    id: RequestId,
    created: u64,
    prompt_tokens: usize,
    include_usage: bool,
    stream: bool,
    parser: HarmonyParser,
    events: mpsc::Receiver<EngineEvent>,
    cancellation: CancellationToken,
    engine: EngineHandle,
    started: Instant,
}

async fn prepare_and_submit(
    state: &AppState,
    body: Result<Bytes, BytesRejection>,
    started: Instant,
) -> Result<Submitted, ApiError> {
    let body = body.map_err(|_| ApiError::invalid("request body exceeds the configured limit"))?;
    let body_bytes = body.len();
    let request: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|_| ApiError::invalid("request body is not a supported chat completion"))?;
    let validated = request
        .validate(MODEL_NAME, &state.config.limits)
        .map_err(ApiError::invalid)?;
    let protocol = state
        .engine
        .protocol()
        .ok_or_else(|| ApiError::not_ready("the model is not ready"))?;
    let permit = tokio::time::timeout(
        state.config.limits.admission_timeout,
        Arc::clone(&state.preparation).acquire_owned(),
    )
    .await
    .map_err(|_| ApiError::overloaded("request preparation is saturated"))?
    .map_err(|_| ApiError::not_ready("request preparation is shutting down"))?;
    let conversation = validated.conversation;
    let (tokens, protocol) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let tokens = protocol.render_for_completion(&conversation)?;
        Ok::<_, crate::gpt_oss::protocol::Error>((tokens, protocol))
    })
    .await
    .map_err(|_| ApiError::internal())?
    .map_err(|_| ApiError::invalid("conversation cannot be rendered by the model protocol"))?;
    let now = Instant::now();
    let deadline = validated
        .deadline
        .map(|duration| RequestDeadline::after(now, duration, &state.config.limits))
        .transpose()
        .map_err(ApiError::from_engine)?;
    let id = state.request_ids.next();
    let rendered_prompt_bytes = tokens
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| ApiError::invalid("rendered prompt byte count overflows usize"))?;
    let prepared = PreparedInferenceRequest::new(
        id,
        tokens,
        validated.max_new_tokens,
        validated.sampling,
        deadline,
        validated.cache_salt.map(String::into_bytes),
        SpeculationPolicy::Disabled,
        body_bytes,
        rendered_prompt_bytes,
        &state.config.limits,
        &state.config.profile,
        now,
    )
    .map_err(ApiError::from_engine)?;
    let prompt_tokens = prepared.prompt_tokens().len();
    let family = state
        .config
        .profile
        .compilation_families
        .iter()
        .find(|family| {
            prompt_tokens <= family.max_prompt_tokens
                && prepared.total_token_budget() <= family.max_sequence_tokens
        })
        .copied()
        .expect("prepared request is covered by a compilation family");
    let cancellation = CancellationToken::new();
    let events = state
        .engine
        .submit(prepared, cancellation.clone())
        .map_err(|error| {
            if error.code() == EngineErrorCode::QueueFull {
                state.metrics.queue_full.inc();
            }
            ApiError::from_engine(error)
        })?;
    state.metrics.admitted.inc();
    tracing::info!(
        request_id = %id,
        state = "admitted",
        phase = "queued",
        prompt_tokens,
        max_new_tokens = validated.max_new_tokens,
        batch_family = 1,
        prefill_family = family.max_prompt_tokens,
        sequence_family = family.max_sequence_tokens,
        "chat request admitted"
    );
    Ok(Submitted {
        id,
        created: unix_seconds(),
        prompt_tokens,
        include_usage: validated.include_usage,
        stream: validated.stream,
        parser: protocol.parser(),
        events,
        cancellation,
        engine: state.engine.clone(),
        started,
    })
}

enum ResponseEvent {
    Role,
    Content(String),
    Reasoning(String),
    ToolCall(ResponseToolCall),
    Terminal { reason: &'static str, usage: Usage },
    Usage(Usage),
    Done,
    Failure(ApiError),
}

fn response_events(submitted: Submitted, metrics: Metrics) -> mpsc::Receiver<ResponseEvent> {
    let (sender, receiver) = mpsc::channel(64);
    tokio::spawn(async move {
        translate(submitted, sender, metrics).await;
    });
    receiver
}

async fn translate(
    mut submitted: Submitted,
    sender: mpsc::Sender<ResponseEvent>,
    metrics: Metrics,
) {
    let mut guard = CancelGuard::new(
        submitted.id,
        submitted.cancellation.clone(),
        submitted.engine.clone(),
    );
    if sender.send(ResponseEvent::Role).await.is_err() {
        return;
    }
    let mut stop_reason = None;
    let mut previous_token = None;
    while let Some(event) = submitted.events.recv().await {
        match event {
            EngineEvent::Raw(raw) => {
                let now = Instant::now();
                match previous_token.replace(now) {
                    None => metrics
                        .ttft_seconds
                        .observe(now.duration_since(submitted.started).as_secs_f64()),
                    Some(previous) => metrics
                        .tpot_seconds
                        .observe(now.duration_since(previous).as_secs_f64()),
                }
                match submitted.parser.process(raw.token) {
                Ok(events) => {
                    for event in events {
                        if let Some(response) = map_harmony_event(submitted.id, event, &mut stop_reason)
                        {
                            if sender.send(response).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                Err(_) => {
                    metrics.failed.inc();
                    let _ = sender.send(ResponseEvent::Failure(ApiError::internal())).await;
                    return;
                }
            }
            }
            EngineEvent::Complete(completion) => {
                let final_reason = if completion.stopped {
                    match submitted.parser.finish() {
                        Ok(reason) => reason,
                        Err(_) => {
                            metrics.failed.inc();
                            let _ = sender.send(ResponseEvent::Failure(ApiError::internal())).await;
                            return;
                        }
                    }
                } else {
                    match submitted.parser.truncate() {
                        Ok(events) => {
                            for event in events {
                                if let Some(response) =
                                    map_harmony_event(submitted.id, event, &mut stop_reason)
                                {
                                    if sender.send(response).await.is_err() {
                                        return;
                                    }
                                }
                            }
                            StopReason::Length
                        }
                        Err(_) => {
                            metrics.failed.inc();
                            let _ = sender.send(ResponseEvent::Failure(ApiError::internal())).await;
                            return;
                        }
                    }
                };
                if stop_reason.is_some_and(|observed| observed != final_reason) {
                    metrics.failed.inc();
                    let _ = sender.send(ResponseEvent::Failure(ApiError::internal())).await;
                    return;
                }
                let usage = Usage {
                    prompt_tokens: completion.prompt_tokens,
                    completion_tokens: completion.completion_tokens,
                    total_tokens: completion.prompt_tokens + completion.completion_tokens,
                };
                if sender
                    .send(ResponseEvent::Terminal {
                        reason: finish_reason(final_reason),
                        usage,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
                if submitted.include_usage && sender.send(ResponseEvent::Usage(usage)).await.is_err()
                {
                    return;
                }
                let _ = sender.send(ResponseEvent::Done).await;
                metrics.completed.inc();
                metrics
                    .request_seconds
                    .observe(submitted.started.elapsed().as_secs_f64());
                tracing::info!(
                    request_id = %submitted.id,
                    state = "terminal",
                    phase = "complete",
                    finish_reason = finish_reason(final_reason),
                    completion_tokens = completion.completion_tokens,
                    "chat request completed"
                );
                guard.disarm();
                return;
            }
            EngineEvent::Cancelled(reason) => {
                metrics.cancelled.inc();
                metrics
                    .request_seconds
                    .observe(submitted.started.elapsed().as_secs_f64());
                tracing::info!(
                    request_id = %submitted.id,
                    state = "terminal",
                    phase = "cancelled",
                    reason = reason.as_str(),
                    "chat request cancelled"
                );
                let _ = sender
                    .send(ResponseEvent::Failure(ApiError::cancelled(reason)))
                    .await;
                guard.disarm();
                return;
            }
            EngineEvent::Failed(error) => {
                metrics.failed.inc();
                metrics
                    .request_seconds
                    .observe(submitted.started.elapsed().as_secs_f64());
                tracing::error!(
                    request_id = %submitted.id,
                    state = "terminal",
                    phase = "failed",
                    error_code = error.code().as_str(),
                    error_message = error.message(),
                    "chat request failed"
                );
                let _ = sender
                    .send(ResponseEvent::Failure(ApiError::from_engine(error)))
                    .await;
                guard.disarm();
                return;
            }
        }
    }
    metrics.failed.inc();
    let _ = sender.send(ResponseEvent::Failure(ApiError::internal())).await;
}

fn map_harmony_event(
    id: RequestId,
    event: HarmonyEvent,
    stop_reason: &mut Option<StopReason>,
) -> Option<ResponseEvent> {
    match event {
        HarmonyEvent::ContentDelta { channel, text } => Some(match channel {
            Channel::Final => ResponseEvent::Content(text),
            Channel::Analysis | Channel::Commentary => ResponseEvent::Reasoning(text),
        }),
        HarmonyEvent::ToolCall(call) => Some(ResponseEvent::ToolCall(ResponseToolCall {
            id: format!("call_{:032x}_0", id.as_u128()),
            kind: ToolKind::Function,
            function: ResponseFunctionCall {
                name: call
                    .recipient
                    .strip_prefix("functions.")
                    .unwrap_or(&call.recipient)
                    .to_owned(),
                arguments: call.raw_arguments,
            },
        })),
        HarmonyEvent::Done(reason) => {
            *stop_reason = Some(reason);
            None
        }
        HarmonyEvent::Message(_) => None,
    }
}

fn stream_response(
    submitted: Submitted,
    metrics: Metrics,
) -> Sse<impl tokio_stream::Stream<Item = Result<SseEvent, Infallible>>> {
    let id = submitted.id.to_string();
    let created = submitted.created;
    let stream = ReceiverStream::new(response_events(submitted, metrics)).map(move |event| {
        let event = match event {
            ResponseEvent::Role => data_chunk(&id, created, ChunkDelta {
                role: Some("assistant"),
                ..ChunkDelta::default()
            }, None, None),
            ResponseEvent::Content(text) => data_chunk(&id, created, ChunkDelta {
                content: Some(text),
                ..ChunkDelta::default()
            }, None, None),
            ResponseEvent::Reasoning(text) => data_chunk(&id, created, ChunkDelta {
                reasoning: Some(text),
                ..ChunkDelta::default()
            }, None, None),
            ResponseEvent::ToolCall(call) => data_chunk(&id, created, ChunkDelta {
                tool_calls: vec![call],
                ..ChunkDelta::default()
            }, None, None),
            ResponseEvent::Terminal { reason, usage } => {
                data_chunk(&id, created, ChunkDelta::default(), Some(reason), Some(usage))
            }
            ResponseEvent::Usage(usage) => SseEvent::default().data(
                serde_json::to_string(&ChatChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: MODEL_NAME,
                    choices: Vec::new(),
                    usage: Some(usage),
                })
                .expect("chat chunk serialization is infallible"),
            ),
            ResponseEvent::Done => SseEvent::default().data("[DONE]"),
            ResponseEvent::Failure(error) => SseEvent::default().event("error").data(
                serde_json::to_string(&error.envelope())
                    .expect("error envelope serialization is infallible"),
            ),
        };
        Ok(event)
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn data_chunk(
    id: &str,
    created: u64,
    delta: ChunkDelta,
    finish_reason: Option<&'static str>,
    usage: Option<Usage>,
) -> SseEvent {
    SseEvent::default().data(
        serde_json::to_string(&ChatChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created,
            model: MODEL_NAME,
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage,
        })
        .expect("chat chunk serialization is infallible"),
    )
}

async fn non_stream_response(submitted: Submitted, metrics: Metrics) -> Response {
    let id = submitted.id.to_string();
    let created = submitted.created;
    let prompt_tokens = submitted.prompt_tokens;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut terminal = None;
    let mut events = response_events(submitted, metrics);
    while let Some(event) = events.recv().await {
        match event {
            ResponseEvent::Role | ResponseEvent::Done | ResponseEvent::Usage(_) => {}
            ResponseEvent::Content(delta) => content.push_str(&delta),
            ResponseEvent::Reasoning(delta) => reasoning.push_str(&delta),
            ResponseEvent::ToolCall(call) => tool_calls.push(call),
            ResponseEvent::Terminal { reason, usage } => terminal = Some((reason, usage)),
            ResponseEvent::Failure(error) => return error.into_response(),
        }
    }
    let Some((finish_reason, usage)) = terminal else {
        return ApiError::internal().into_response();
    };
    debug_assert_eq!(prompt_tokens, usage.prompt_tokens);
    Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: MODEL_NAME,
        choices: vec![ChatChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: (!content.is_empty()).then_some(content),
                reasoning: (!reasoning.is_empty()).then_some(reasoning),
                tool_calls,
            },
            finish_reason,
        }],
        usage,
    })
    .into_response()
}

fn finish_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Return => "stop",
        StopReason::ToolCall => "tool_calls",
        StopReason::Length => "length",
    }
}

struct CancelGuard {
    id: RequestId,
    cancellation: CancellationToken,
    engine: EngineHandle,
    armed: bool,
}

impl CancelGuard {
    fn new(id: RequestId, cancellation: CancellationToken, engine: EngineHandle) -> Self {
        Self {
            id,
            cancellation,
            engine,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
            self.engine.cancel(self.id, CancelReason::ClientDisconnect);
        }
    }
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request",
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn overloaded(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "queue_full",
            kind: "server_error",
            message: message.into(),
        }
    }

    fn not_ready(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "not_ready",
            kind: "server_error",
            message: message.into(),
        }
    }

    fn internal() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "execution_failed",
            kind: "server_error",
            message: "model execution failed".to_owned(),
        }
    }

    fn cancelled(reason: CancelReason) -> Self {
        match reason {
            CancelReason::Deadline => Self {
                status: StatusCode::REQUEST_TIMEOUT,
                code: "deadline_exceeded",
                kind: "server_error",
                message: "request deadline exceeded".to_owned(),
            },
            _ => Self {
                status: StatusCode::SERVICE_UNAVAILABLE,
                code: "cancelled",
                kind: "server_error",
                message: "request was cancelled".to_owned(),
            },
        }
    }

    fn from_engine(error: EngineError) -> Self {
        match error.code() {
            EngineErrorCode::InvalidRequest => Self::invalid(error.message()),
            EngineErrorCode::QueueFull => Self::overloaded("the inference queue is full"),
            EngineErrorCode::NotReady => Self::not_ready("the model is not ready"),
            EngineErrorCode::DeadlineExceeded => Self::cancelled(CancelReason::Deadline),
            EngineErrorCode::Cancelled => Self::cancelled(CancelReason::Explicit),
            EngineErrorCode::ShuttingDown => Self::not_ready("the server is shutting down"),
            EngineErrorCode::ExecutionFailed => Self::internal(),
        }
    }

    fn envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            error: ErrorBody {
                message: self.message.clone(),
                kind: self.kind,
                code: self.code,
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.envelope())).into_response()
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler must install");
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                let _ = result;
            }
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[derive(Debug)]
struct ConfigurationError(String);

impl std::fmt::Display for ConfigurationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ConfigurationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use std::path::PathBuf;
    use tower::ServiceExt;

    fn state(readiness: Readiness) -> AppState {
        let limits = super::super::contracts::ServerLimits::default();
        AppState {
            engine: EngineHandle::for_test(readiness, limits.request_event_capacity),
            config: Arc::new(ServerConfig {
                bind: "127.0.0.1:0".parse().unwrap(),
                model: PathBuf::from("/model"),
                backend: super::super::contracts::Backend::Cpu,
                profile: super::super::contracts::ServerProfile::a40(4_096, 8_192),
                limits: limits.clone(),
                shutdown_grace: Duration::from_secs(1),
            }),
            request_ids: Arc::new(RequestIds::new()),
            preparation: Arc::new(Semaphore::new(limits.preparation_concurrency)),
            metrics: Metrics::new(),
        }
    }

    #[test]
    fn engine_errors_have_stable_public_statuses_without_device_details() {
        let error = ApiError::from_engine(EngineError::new(
            EngineErrorCode::ExecutionFailed,
            "/secret/model/path: CUDA kernel launch failed",
        ));
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.code, "execution_failed");
        assert_eq!(error.message, "model execution failed");
    }

    #[test]
    fn finish_reasons_match_the_openai_domain() {
        assert_eq!(finish_reason(StopReason::Return), "stop");
        assert_eq!(finish_reason(StopReason::ToolCall), "tool_calls");
        assert_eq!(finish_reason(StopReason::Length), "length");
    }

    #[test]
    fn complete_tool_calls_receive_stable_prompt_independent_ids() {
        let id = RequestIds::new().next();
        let mut reason = None;
        let Some(ResponseEvent::ToolCall(call)) = map_harmony_event(
            id,
            HarmonyEvent::ToolCall(crate::gpt_oss::protocol::ParsedToolCall {
                recipient: "functions.lookup".to_owned(),
                arguments: serde_json::json!({"city": "Rome"}),
                raw_arguments: "{\"city\":\"Rome\"}".to_owned(),
            }),
            &mut reason,
        ) else {
            panic!("complete Harmony tool call did not map to an OpenAI call");
        };
        assert_eq!(call.id, format!("call_{:032x}_0", id.as_u128()));
        assert_eq!(call.function.name, "lookup");
        assert_eq!(call.function.arguments, "{\"city\":\"Rome\"}");
        assert_eq!(reason, None);
    }

    #[tokio::test]
    async fn health_readiness_and_model_routes_are_truthful_during_startup() {
        let app = router(state(Readiness::Starting), 4);
        let health = app
            .clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let ready = app
            .clone()
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
        let models = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(models.status(), StatusCode::OK);
        let body = to_bytes(models.into_body(), 64 * 1024).await.unwrap();
        assert!(std::str::from_utf8(&body).unwrap().contains(MODEL_NAME));
    }

    #[tokio::test]
    async fn chat_rejections_keep_the_openai_error_envelope() {
        let app = router(state(Readiness::Starting), 4);
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(
                        "{{\"model\":{model:?},\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}],\"unsupported\":true}}",
                        model = MODEL_NAME,
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
}
