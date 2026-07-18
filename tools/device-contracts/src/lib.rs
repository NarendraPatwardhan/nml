#![forbid(unsafe_code)]

use axum::Router;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Json, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use runfiles::Runfiles;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::ffi::OsString;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const SCHEMA_VERSION: u32 = 1;
const OUTPUT_TAIL_BYTES: usize = 64 * 1024;
const MAX_CONTRACT_TIMEOUT_SECONDS: u64 = 60 * 60;
const MAX_TOTAL_TIMEOUT_SECONDS: u64 = 3 * 60 * 60;

const CONTRACTS: [ContractDefinition; 7] = [
    ContractDefinition {
        name: "flash_attention_device_capability",
        rlocation: env!("NML_FLASH_ATTENTION_CAPABILITY_CONTRACT"),
        arguments: &[],
    },
    ContractDefinition {
        name: "cuda_runtime",
        rlocation: env!("NML_CUDA_RUNTIME_CONTRACT"),
        arguments: &[],
    },
    ContractDefinition {
        name: "linear",
        rlocation: env!("NML_LINEAR_CONTRACT"),
        arguments: &[],
    },
    ContractDefinition {
        name: "attention",
        rlocation: env!("NML_ATTENTION_CONTRACT"),
        arguments: &[],
    },
    ContractDefinition {
        name: "neural_ops",
        rlocation: env!("NML_NEURAL_OPS_CONTRACT"),
        arguments: &[],
    },
    ContractDefinition {
        name: "execution_performance",
        rlocation: env!("NML_EXECUTION_PERFORMANCE_CONTRACT"),
        // This executable is a Rust test binary. Device contracts invoke test
        // executables directly rather than through Bazel's test wrapper, so
        // the harness argument is part of the immutable contract definition.
        // Keeping it here ensures phase measurements survive into the lease
        // result instead of being swallowed by the harness's output capture.
        arguments: &["--nocapture"],
    },
    ContractDefinition {
        name: "nvfp4",
        rlocation: env!("NML_NVFP4_CONTRACT"),
        arguments: &[],
    },
];

#[derive(Clone, Copy)]
struct ContractDefinition {
    name: &'static str,
    rlocation: &'static str,
    arguments: &'static [&'static str],
}

#[derive(Clone)]
struct AppState {
    configuration: Arc<Configuration>,
    execution: Arc<Mutex<ExecutionState>>,
    task: Arc<Mutex<Option<JoinHandle<()>>>>,
    cancellation: CancellationToken,
}

struct Configuration {
    lease_token: String,
    identity: ArtifactIdentity,
    runfiles_directory: PathBuf,
    runtime_rlocation: &'static str,
    contracts: Vec<ResolvedContract>,
    hardware: Result<Vec<GpuIdentity>, String>,
}

#[derive(Clone)]
struct ResolvedContract {
    name: &'static str,
    path: PathBuf,
    arguments: &'static [&'static str],
}

enum ExecutionState {
    Idle,
    Running(RunningState),
    Terminal(Arc<ExecutionResult>),
}

#[derive(Clone, Serialize)]
struct RunningState {
    contracts: Vec<String>,
    started_unix_milliseconds: u128,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    contracts: Vec<String>,
    per_contract_timeout_seconds: u64,
    total_timeout_seconds: u64,
}

#[derive(Clone, Serialize)]
struct ArtifactIdentity {
    image_digest: String,
    source_commit: String,
    source_dirty: bool,
}

#[derive(Clone, Serialize)]
struct GpuIdentity {
    index: u32,
    name: String,
    uuid: String,
    compute_capability: String,
    driver_version: String,
}

#[derive(Serialize)]
struct LiveResponse {
    schema_version: u32,
    service: &'static str,
}

#[derive(Serialize)]
struct ReadinessResponse<'a> {
    schema_version: u32,
    ready: bool,
    artifact: &'a ArtifactIdentity,
    contracts: Vec<&'static str>,
    hardware: Option<&'a [GpuIdentity]>,
    hardware_failure: Option<&'a str>,
    runtime: RuntimeDiagnostics,
}

#[derive(Clone, Serialize)]
struct RuntimeDiagnostics {
    cuda_runtime_rlocation: &'static str,
    runfiles_directory: String,
}

#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum StateResponse {
    Idle,
    Running { execution: RunningState },
    Terminal { result: ExecutionResult },
}

#[derive(Clone, Serialize)]
struct ExecutionResult {
    schema_version: u32,
    status: ExecutionStatus,
    artifact: ArtifactIdentity,
    hardware: Vec<GpuIdentity>,
    runtime: RuntimeDiagnostics,
    started_unix_milliseconds: u128,
    duration_milliseconds: u128,
    contracts: Vec<ContractResult>,
    failure: Option<String>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionStatus {
    Succeeded,
    Failed,
    TimedOut,
    Interrupted,
}

#[derive(Clone, Serialize)]
struct ContractResult {
    name: String,
    status: ContractStatus,
    exit_code: Option<i32>,
    duration_milliseconds: u128,
    stdout: OutputTail,
    stderr: OutputTail,
    failure: Option<String>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum ContractStatus {
    Succeeded,
    Failed,
    TimedOut,
    Interrupted,
    SpawnFailed,
}

#[derive(Clone, Serialize)]
struct OutputTail {
    text: String,
    truncated: bool,
    observed_bytes: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    schema_version: u32,
    error: String,
}

pub async fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let configuration = Arc::new(Configuration::from_environment()?);
    let listen: SocketAddr = std::env::var("NML_CONTRACT_LISTEN")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_owned())
        .parse()?;
    let cancellation = CancellationToken::new();
    let state = AppState {
        configuration,
        execution: Arc::new(Mutex::new(ExecutionState::Idle)),
        task: Arc::new(Mutex::new(None)),
        cancellation: cancellation.clone(),
    };
    let application = Router::new()
        .route("/live", get(live))
        .route("/ready", get(ready))
        .route("/state", get(state_handler))
        .route("/run", post(run))
        .route("/result", get(result))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(listen).await?;
    eprintln!("run_device_contracts: listening on {listen}");
    axum::serve(listener, application)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    cancellation.cancel();
    if let Some(task) = state.task.lock().await.take() {
        let _ = task.await;
    }
    Ok(())
}

async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler must install on supported Linux hosts");
    tokio::select! {
        result = tokio::signal::ctrl_c() => {
            let _ = result;
        }
        _ = terminate.recv() => {}
    }
}

impl Configuration {
    fn from_environment() -> Result<Self, Box<dyn std::error::Error>> {
        let lease_token = required_environment("NML_CONTRACT_LEASE_TOKEN")?;
        if lease_token.len() < 32 {
            return Err("NML_CONTRACT_LEASE_TOKEN must contain at least 32 bytes".into());
        }
        let identity = ArtifactIdentity {
            image_digest: required_digest("NML_IMAGE_DIGEST")?,
            source_commit: required_commit("NML_SOURCE_COMMIT")?,
            source_dirty: required_boolean("NML_SOURCE_DIRTY")?,
        };
        let runfiles = Runfiles::create()?;
        let runfiles_directory = runfiles_directory()?;
        let contracts = CONTRACTS
            .iter()
            .map(|definition| {
                let path = runfiles.rlocation(definition.rlocation).ok_or_else(|| {
                    format!(
                        "contract {:?} has no runfiles entry {:?}",
                        definition.name, definition.rlocation
                    )
                })?;
                if !path.is_file() {
                    return Err(format!(
                        "contract {:?} is absent from runfiles at {}",
                        definition.name,
                        path.display()
                    ));
                }
                Ok(ResolvedContract {
                    name: definition.name,
                    path,
                    arguments: definition.arguments,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let runtime = runfiles
            .rlocation(env!("NML_CUDA_RUNTIME"))
            .ok_or_else(|| {
                format!(
                    "CUDA runtime has no runfiles entry {:?}",
                    env!("NML_CUDA_RUNTIME")
                )
            })?;
        if !runtime.exists() {
            return Err(format!("CUDA runtime is absent at {}", runtime.display()).into());
        }
        Ok(Self {
            lease_token,
            identity,
            runfiles_directory,
            runtime_rlocation: env!("NML_CUDA_RUNTIME"),
            contracts,
            hardware: query_hardware(),
        })
    }

    fn runtime_diagnostics(&self) -> RuntimeDiagnostics {
        RuntimeDiagnostics {
            cuda_runtime_rlocation: self.runtime_rlocation,
            runfiles_directory: self.runfiles_directory.display().to_string(),
        }
    }
}

async fn live() -> Json<LiveResponse> {
    Json(LiveResponse {
        schema_version: SCHEMA_VERSION,
        service: "nml-device-contract-runner",
    })
}

async fn ready(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let (hardware, hardware_failure) = match &state.configuration.hardware {
        Ok(hardware) => (Some(hardware.as_slice()), None),
        Err(failure) => (None, Some(failure.as_str())),
    };
    let ready = hardware.is_some();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(ReadinessResponse {
            schema_version: SCHEMA_VERSION,
            ready,
            artifact: &state.configuration.identity,
            contracts: state
                .configuration
                .contracts
                .iter()
                .map(|contract| contract.name)
                .collect(),
            hardware,
            hardware_failure,
            runtime: state.configuration.runtime_diagnostics(),
        }),
    )
        .into_response()
}

async fn state_handler(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let execution = state.execution.lock().await;
    Json(snapshot(&*execution)).into_response()
}

async fn result(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    match &*state.execution.lock().await {
        ExecutionState::Terminal(result) => Json((**result).clone()).into_response(),
        ExecutionState::Idle | ExecutionState::Running(_) => error(
            StatusCode::NOT_FOUND,
            "no terminal contract result is available",
        ),
    }
}

async fn run(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<RunRequest>, JsonRejection>,
) -> Response {
    if let Err(response) = authorize(&state, &headers) {
        return response;
    }
    let request = match payload {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return error(
                StatusCode::BAD_REQUEST,
                format!("invalid run request: {}", rejection.body_text()),
            );
        }
    };
    let selection = match validate_request(&state.configuration, &request) {
        Ok(selection) => selection,
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    if let Err(message) = &state.configuration.hardware {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("GPU discovery failed: {message}"),
        );
    }
    let started_unix_milliseconds = unix_milliseconds();
    let running = RunningState {
        contracts: selection
            .iter()
            .map(|contract| contract.name.to_owned())
            .collect(),
        started_unix_milliseconds,
    };
    {
        let mut execution = state.execution.lock().await;
        match &*execution {
            ExecutionState::Idle => *execution = ExecutionState::Running(running.clone()),
            ExecutionState::Running(_) | ExecutionState::Terminal(_) => {
                return error(
                    StatusCode::CONFLICT,
                    "this runner has already accepted its one contract execution",
                );
            }
        }
    }
    let execution_state = state.execution.clone();
    let configuration = state.configuration.clone();
    let cancellation = state.cancellation.clone();
    let task = tokio::spawn(async move {
        let result = execute(
            configuration,
            selection,
            request,
            started_unix_milliseconds,
            cancellation,
        )
        .await;
        *execution_state.lock().await = ExecutionState::Terminal(Arc::new(result));
    });
    *state.task.lock().await = Some(task);
    (StatusCode::ACCEPTED, Json(running)).into_response()
}

fn validate_request(
    configuration: &Configuration,
    request: &RunRequest,
) -> Result<Vec<ResolvedContract>, String> {
    if request.contracts.is_empty() {
        return Err("contracts must contain at least one allowlisted name".to_owned());
    }
    validate_deadline(
        "per_contract_timeout_seconds",
        request.per_contract_timeout_seconds,
        MAX_CONTRACT_TIMEOUT_SECONDS,
    )?;
    validate_deadline(
        "total_timeout_seconds",
        request.total_timeout_seconds,
        MAX_TOTAL_TIMEOUT_SECONDS,
    )?;
    let mut selection = Vec::with_capacity(request.contracts.len());
    for name in &request.contracts {
        if selection
            .iter()
            .any(|selected: &ResolvedContract| selected.name == name)
        {
            return Err(format!("contract {name:?} appears more than once"));
        }
        let contract = configuration
            .contracts
            .iter()
            .find(|contract| contract.name == name)
            .ok_or_else(|| format!("contract {name:?} is not in the build-time allowlist"))?;
        selection.push(contract.clone());
    }
    Ok(selection)
}

fn validate_deadline(name: &str, value: u64, maximum: u64) -> Result<(), String> {
    if value == 0 || value > maximum {
        return Err(format!("{name} must be between 1 and {maximum} seconds"));
    }
    Ok(())
}

async fn execute(
    configuration: Arc<Configuration>,
    contracts: Vec<ResolvedContract>,
    request: RunRequest,
    started_unix_milliseconds: u128,
    cancellation: CancellationToken,
) -> ExecutionResult {
    let started = Instant::now();
    let total_deadline = started + Duration::from_secs(request.total_timeout_seconds);
    let mut results = Vec::with_capacity(contracts.len());
    let mut terminal_status = ExecutionStatus::Succeeded;
    let mut failure = None;
    for contract in contracts {
        let now = Instant::now();
        if now >= total_deadline {
            terminal_status = ExecutionStatus::TimedOut;
            failure =
                Some("the total contract deadline expired before the next contract".to_owned());
            break;
        }
        let timeout = Duration::from_secs(request.per_contract_timeout_seconds)
            .min(total_deadline.duration_since(now));
        let result = execute_contract(&configuration, &contract, timeout, &cancellation).await;
        let status = result.status;
        results.push(result);
        match status {
            ContractStatus::Succeeded => {}
            ContractStatus::TimedOut => {
                terminal_status = ExecutionStatus::TimedOut;
                failure = Some(format!("contract {:?} timed out", contract.name));
                break;
            }
            ContractStatus::Interrupted => {
                terminal_status = ExecutionStatus::Interrupted;
                failure = Some(format!("contract {:?} was interrupted", contract.name));
                break;
            }
            ContractStatus::Failed | ContractStatus::SpawnFailed => {
                terminal_status = ExecutionStatus::Failed;
                failure = Some(format!("contract {:?} failed", contract.name));
                break;
            }
        }
    }
    ExecutionResult {
        schema_version: SCHEMA_VERSION,
        status: terminal_status,
        artifact: configuration.identity.clone(),
        hardware: configuration
            .hardware
            .as_ref()
            .expect("execute is entered only after successful GPU discovery")
            .clone(),
        runtime: configuration.runtime_diagnostics(),
        started_unix_milliseconds,
        duration_milliseconds: started.elapsed().as_millis(),
        contracts: results,
        failure,
    }
}

async fn execute_contract(
    configuration: &Configuration,
    contract: &ResolvedContract,
    timeout: Duration,
    cancellation: &CancellationToken,
) -> ContractResult {
    let started = Instant::now();
    let mut command = Command::new(&contract.path);
    command
        .args(contract.arguments)
        .env("RUNFILES_DIR", &configuration.runfiles_directory)
        .env("JAVA_RUNFILES", &configuration.runfiles_directory)
        .env(
            "NML_CUDA_RUNTIME_RLOCATION",
            configuration.runtime_rlocation,
        )
        .kill_on_drop(true)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return empty_contract_result(
                contract.name,
                ContractStatus::SpawnFailed,
                started.elapsed(),
                format!("failed to spawn {}: {error}", contract.path.display()),
            );
        }
    };
    let stdout_task = tokio::spawn(read_tail(child.stdout.take().expect("piped stdout")));
    let stderr_task = tokio::spawn(read_tail(child.stderr.take().expect("piped stderr")));
    let (status, exit_code, failure) = tokio::select! {
        status = child.wait() => match status {
            Ok(status) if status.success() => (ContractStatus::Succeeded, status.code(), None),
            Ok(status) => (
                ContractStatus::Failed,
                status.code(),
                Some(format!("contract exited with {status}")),
            ),
            Err(error) => (
                ContractStatus::Failed,
                None,
                Some(format!("failed while waiting for contract: {error}")),
            ),
        },
        () = tokio::time::sleep(timeout) => {
            let kill = child.kill().await;
            let exit_code = child.wait().await.ok().and_then(|status| status.code());
            (
                ContractStatus::TimedOut,
                exit_code,
                Some(match kill {
                    Ok(()) => format!("contract exceeded {} ms", timeout.as_millis()),
                    Err(error) => format!("contract exceeded {} ms and kill failed: {error}", timeout.as_millis()),
                }),
            )
        },
        () = cancellation.cancelled() => {
            let kill = child.kill().await;
            let exit_code = child.wait().await.ok().and_then(|status| status.code());
            (
                ContractStatus::Interrupted,
                exit_code,
                Some(match kill {
                    Ok(()) => "runner shutdown interrupted the contract".to_owned(),
                    Err(error) => format!("runner shutdown interrupted the contract and kill failed: {error}"),
                }),
            )
        },
    };
    let stdout = output_from_task(stdout_task, "stdout").await;
    let stderr = output_from_task(stderr_task, "stderr").await;
    ContractResult {
        name: contract.name.to_owned(),
        status,
        exit_code,
        duration_milliseconds: started.elapsed().as_millis(),
        stdout,
        stderr,
        failure,
    }
}

fn empty_contract_result(
    name: &str,
    status: ContractStatus,
    duration: Duration,
    failure: String,
) -> ContractResult {
    ContractResult {
        name: name.to_owned(),
        status,
        exit_code: None,
        duration_milliseconds: duration.as_millis(),
        stdout: OutputTail::empty(),
        stderr: OutputTail::empty(),
        failure: Some(failure),
    }
}

async fn read_tail(mut reader: impl AsyncRead + Unpin) -> OutputTail {
    let mut tail = VecDeque::with_capacity(OUTPUT_TAIL_BYTES);
    let mut observed_bytes = 0u64;
    let mut buffer = [0u8; 8192];
    loop {
        let count = match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) => {
                let marker = format!("\n<output read failed: {error}>");
                append_tail(&mut tail, marker.as_bytes());
                observed_bytes = observed_bytes.saturating_add(marker.len() as u64);
                break;
            }
        };
        observed_bytes = observed_bytes.saturating_add(count as u64);
        append_tail(&mut tail, &buffer[..count]);
    }
    OutputTail {
        text: String::from_utf8_lossy(&tail.into_iter().collect::<Vec<_>>()).into_owned(),
        truncated: observed_bytes > OUTPUT_TAIL_BYTES as u64,
        observed_bytes,
    }
}

fn append_tail(tail: &mut VecDeque<u8>, bytes: &[u8]) {
    let excess = tail
        .len()
        .saturating_add(bytes.len())
        .saturating_sub(OUTPUT_TAIL_BYTES);
    tail.drain(..excess.min(tail.len()));
    if bytes.len() > OUTPUT_TAIL_BYTES {
        tail.extend(&bytes[bytes.len() - OUTPUT_TAIL_BYTES..]);
    } else {
        tail.extend(bytes);
    }
}

async fn output_from_task(task: JoinHandle<OutputTail>, stream: &str) -> OutputTail {
    match task.await {
        Ok(output) => output,
        Err(error) => OutputTail {
            text: format!("<{stream} capture task failed: {error}>"),
            truncated: false,
            observed_bytes: 0,
        },
    }
}

impl OutputTail {
    fn empty() -> Self {
        Self {
            text: String::new(),
            truncated: false,
            observed_bytes: 0,
        }
    }
}

fn snapshot(state: &ExecutionState) -> StateResponse {
    match state {
        ExecutionState::Idle => StateResponse::Idle,
        ExecutionState::Running(execution) => StateResponse::Running {
            execution: execution.clone(),
        },
        ExecutionState::Terminal(result) => StateResponse::Terminal {
            result: (**result).clone(),
        },
    }
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let expected = format!("Bearer {}", state.configuration.lease_token);
    let authorized = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_equal(value.as_bytes(), expected.as_bytes()));
    if authorized {
        Ok(())
    } else {
        Err(error(StatusCode::UNAUTHORIZED, "invalid lease token"))
    }
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            schema_version: SCHEMA_VERSION,
            error: message.into(),
        }),
    )
        .into_response()
}

fn query_hardware() -> Result<Vec<GpuIdentity>, String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,uuid,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|error| format!("could not execute nvidia-smi: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvidia-smi exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("nvidia-smi returned non-UTF-8 output: {error}"))?;
    let devices = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_gpu)
        .collect::<Result<Vec<_>, _>>()?;
    if devices.is_empty() {
        return Err("nvidia-smi reported no visible GPU".to_owned());
    }
    Ok(devices)
}

fn parse_gpu(line: &str) -> Result<GpuIdentity, String> {
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    let [index, name, uuid, compute_capability, driver_version] = fields.as_slice() else {
        return Err(format!("unexpected nvidia-smi GPU row {line:?}"));
    };
    Ok(GpuIdentity {
        index: index
            .parse()
            .map_err(|_| format!("invalid GPU index in nvidia-smi row {line:?}"))?,
        name: (*name).to_owned(),
        uuid: (*uuid).to_owned(),
        compute_capability: (*compute_capability).to_owned(),
        driver_version: (*driver_version).to_owned(),
    })
}

fn runfiles_directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(directory) = std::env::var_os("RUNFILES_DIR") {
        return Ok(PathBuf::from(directory));
    }
    let executable = std::env::current_exe()?;
    let mut runfiles: OsString = executable.as_os_str().to_owned();
    runfiles.push(".runfiles");
    let runfiles = PathBuf::from(runfiles);
    if !runfiles.is_dir() {
        return Err(format!("runfiles directory is absent at {}", runfiles.display()).into());
    }
    Ok(runfiles)
}

fn required_environment(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let value = std::env::var(name).map_err(|_| format!("{name} is required"))?;
    if value.is_empty() {
        return Err(format!("{name} must not be empty").into());
    }
    Ok(value)
}

fn required_digest(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let value = required_environment(name)?;
    validate_digest(&value).ok_or_else(|| format!("{name} must be a sha256 OCI digest").into())
}

fn required_commit(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let value = required_environment(name)?;
    validate_commit(&value)
        .ok_or_else(|| format!("{name} must be a full 40-character Git commit").into())
}

fn required_boolean(name: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match required_environment(name)?.as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{name} must be exactly true or false").into()),
    }
}

fn validate_digest(value: &str) -> Option<String> {
    (value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().all(|byte| byte.is_ascii_hexdigit()))
    .then(|| value.to_ascii_lowercase())
}

fn validate_commit(value: &str) -> Option<String> {
    (value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

fn unix_milliseconds() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_tail_retains_only_the_declared_suffix() {
        let mut tail = VecDeque::new();
        append_tail(&mut tail, &vec![b'a'; OUTPUT_TAIL_BYTES - 2]);
        append_tail(&mut tail, b"bcdef");
        assert_eq!(tail.len(), OUTPUT_TAIL_BYTES);
        assert_eq!(
            tail.iter().rev().take(5).copied().collect::<Vec<_>>(),
            b"fedcb"
        );
    }

    #[test]
    fn gpu_rows_are_structurally_validated() {
        let gpu = parse_gpu("0, NVIDIA GeForce GTX 1660 Ti, GPU-1234, 7.5, 590.48.01")
            .expect("valid row");
        assert_eq!(gpu.index, 0);
        assert_eq!(gpu.compute_capability, "7.5");
        assert!(parse_gpu("0, too, short").is_err());
    }

    #[test]
    fn identity_fields_reject_ambiguous_values() {
        assert!(validate_digest(&format!("sha256:{}", "a".repeat(64))).is_some());
        assert!(validate_digest(&format!("sha512:{}", "a".repeat(64))).is_none());
        assert!(validate_commit(&"b".repeat(40)).is_some());
        assert!(validate_commit("short").is_none());
    }

    #[test]
    fn lease_comparison_includes_length() {
        assert!(constant_time_equal(b"same", b"same"));
        assert!(!constant_time_equal(b"same", b"same-prefix"));
        assert!(!constant_time_equal(b"same", b"different"));
    }

    #[test]
    fn performance_contract_preserves_phase_output() {
        let performance = CONTRACTS
            .iter()
            .find(|contract| contract.name == "execution_performance")
            .expect("performance contract is permanent");
        assert_eq!(performance.arguments, ["--nocapture"]);
        assert!(
            CONTRACTS
                .iter()
                .filter(|contract| contract.name != "execution_performance")
                .all(|contract| contract.arguments.is_empty())
        );
    }

    #[test]
    fn selection_is_allowlisted_unique_and_bounded() {
        let configuration = Configuration {
            lease_token: "x".repeat(32),
            identity: ArtifactIdentity {
                image_digest: format!("sha256:{}", "a".repeat(64)),
                source_commit: "b".repeat(40),
                source_dirty: true,
            },
            runfiles_directory: PathBuf::from("/runfiles"),
            runtime_rlocation: "runtime",
            contracts: vec![ResolvedContract {
                name: "cuda_runtime",
                path: PathBuf::from("/contract"),
                arguments: &[],
            }],
            hardware: Err("not consulted by request validation".to_owned()),
        };
        let request = |contracts: Vec<&str>, per_contract, total| RunRequest {
            contracts: contracts.into_iter().map(str::to_owned).collect(),
            per_contract_timeout_seconds: per_contract,
            total_timeout_seconds: total,
        };
        assert!(validate_request(&configuration, &request(vec!["cuda_runtime"], 10, 20)).is_ok());
        assert!(validate_request(&configuration, &request(vec!["unknown"], 10, 20)).is_err());
        assert!(
            validate_request(
                &configuration,
                &request(vec!["cuda_runtime", "cuda_runtime"], 10, 20),
            )
            .is_err()
        );
        assert!(validate_request(&configuration, &request(vec!["cuda_runtime"], 0, 20)).is_err());
        assert!(
            validate_request(
                &configuration,
                &request(vec!["cuda_runtime"], 10, MAX_TOTAL_TIMEOUT_SECONDS + 1),
            )
            .is_err()
        );
    }
}
