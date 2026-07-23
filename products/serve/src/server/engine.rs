//! Dedicated model/PJRT owner and bounded Tokio-facing command boundary.

use super::contracts::{CancelReason, EngineError, EngineErrorCode, PreparedInferenceRequest, RequestId};
use super::metrics::Metrics;
use super::scheduler::{BatchItem, BatchSubmission, Scheduler, SchedulerConfig};
use crate::gpt_oss::protocol::HarmonyProtocol;
use crate::gpt_oss::{Generator, RawToken, ServerSession};
use crate::Error;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;

const MAX_COMMANDS_PER_ITERATION: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Readiness {
    Starting,
    Ready,
    Failed,
    ShuttingDown,
    Stopped,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct EngineSnapshot {
    pub(crate) queued: usize,
    pub(crate) active: usize,
    pub(crate) completed: u64,
    pub(crate) cancelled: u64,
    pub(crate) failed: u64,
    pub(crate) cache_total_pages: usize,
    pub(crate) cache_free_pages: usize,
    pub(crate) cache_reserved_pages: usize,
}

pub(crate) enum EngineEvent {
    Raw(RawToken),
    Complete(EngineCompletion),
    Cancelled(CancelReason),
    Failed(EngineError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EngineCompletion {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) stopped: bool,
}

enum EngineCommand {
    Submit {
        request: PreparedInferenceRequest,
        cancellation: CancellationToken,
        events: mpsc::Sender<EngineEvent>,
    },
    Cancel {
        id: RequestId,
        reason: CancelReason,
    },
    Snapshot {
        reply: oneshot::Sender<EngineSnapshot>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

#[derive(Clone)]
pub(crate) struct EngineHandle {
    commands: mpsc::Sender<EngineCommand>,
    readiness: watch::Receiver<Readiness>,
    protocol: Arc<RwLock<Option<HarmonyProtocol>>>,
    event_capacity: usize,
}

pub(crate) struct EngineOwner {
    pub(crate) handle: EngineHandle,
    join: Option<JoinHandle<()>>,
}

impl EngineHandle {
    #[cfg(test)]
    pub(crate) fn for_test(readiness: Readiness, event_capacity: usize) -> Self {
        let (commands, _receiver) = mpsc::channel(1);
        let (_sender, readiness) = watch::channel(readiness);
        Self {
            commands,
            readiness,
            protocol: Arc::new(RwLock::new(None)),
            event_capacity,
        }
    }

    pub(crate) fn readiness(&self) -> Readiness {
        self.readiness.borrow().clone()
    }

    pub(crate) fn protocol(&self) -> Option<HarmonyProtocol> {
        self.protocol.read().ok()?.clone()
    }

    pub(crate) fn submit(
        &self,
        request: PreparedInferenceRequest,
        cancellation: CancellationToken,
    ) -> Result<mpsc::Receiver<EngineEvent>, EngineError> {
        if self.readiness() != Readiness::Ready {
            return Err(EngineError::new(
                EngineErrorCode::NotReady,
                "the model engine is not ready",
            ));
        }
        let (events, receiver) = mpsc::channel(self.event_capacity);
        self.commands
            .try_send(EngineCommand::Submit {
                request,
                cancellation,
                events,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => EngineError::new(
                    EngineErrorCode::QueueFull,
                    "the bounded engine command queue is full",
                ),
                mpsc::error::TrySendError::Closed(_) => EngineError::new(
                    EngineErrorCode::ShuttingDown,
                    "the model engine is shutting down",
                ),
            })?;
        Ok(receiver)
    }

    pub(crate) fn cancel(&self, id: RequestId, reason: CancelReason) {
        let _ = self.commands.try_send(EngineCommand::Cancel { id, reason });
    }

    pub(crate) async fn snapshot(&self) -> Option<EngineSnapshot> {
        let (reply, receiver) = oneshot::channel();
        self.commands
            .send(EngineCommand::Snapshot { reply })
            .await
            .ok()?;
        receiver.await.ok()
    }
}

impl EngineOwner {
    pub(crate) async fn shutdown(mut self) -> Result<(), EngineError> {
        let (reply, receiver) = oneshot::channel();
        let acknowledgement = self
            .handle
            .commands
            .send(EngineCommand::Shutdown { reply })
            .await;
        if acknowledgement.is_ok() {
            receiver.await.map_err(|_| {
                EngineError::new(
                    EngineErrorCode::ExecutionFailed,
                    "the model engine dropped its shutdown acknowledgement",
                )
            })?;
        }
        let join = self.join.take().expect("engine owner joins exactly once");
        tokio::task::spawn_blocking(move || join.join())
            .await
            .map_err(|error| {
                EngineError::new(
                    EngineErrorCode::ExecutionFailed,
                    format!("engine join task failed: {error}"),
                )
            })?
            .map_err(|_| {
                EngineError::new(
                    EngineErrorCode::ExecutionFailed,
                    "the model engine thread panicked",
                )
            })?;
        Ok(())
    }
}

pub(crate) fn spawn(
    model: std::path::PathBuf,
    profile: super::contracts::ServerProfile,
    command_capacity: usize,
    event_capacity: usize,
    max_queued: usize,
    max_active: usize,
    admission_timeout: Duration,
    metrics: Metrics,
    platform: impl FnOnce() -> Result<nml::Platform, Error> + Send + 'static,
) -> Result<EngineOwner, EngineError> {
    let (commands, receiver) = mpsc::channel(command_capacity);
    let (readiness_tx, readiness_rx) = watch::channel(Readiness::Starting);
    let (initialized_tx, initialized_rx) = std::sync::mpsc::sync_channel(0);
    let protocol = Arc::new(RwLock::new(None));
    let protocol_for_thread = Arc::clone(&protocol);
    let join = std::thread::Builder::new()
        .name("nml-model-engine".to_owned())
        .spawn(move || {
            let platform = match platform() {
                Ok(platform) => {
                    let _ = initialized_tx.send(Ok(()));
                    platform
                }
                Err(error) => {
                    let _ = initialized_tx.send(Err(error.to_string()));
                    let _ = readiness_tx.send(Readiness::Failed);
                    return;
                }
            };
            let result = (|| {
                let scheduler_config = SchedulerConfig {
                    batch_buckets: profile.batch_buckets.clone(),
                    prefill_query_buckets: profile.prefill_query_buckets.clone(),
                    max_active_sequences: max_active,
                    max_batched_tokens: profile.max_batched_tokens,
                    max_prefill_chunk: profile.max_prefill_chunk,
                    max_prefill_wait: profile.max_prefill_wait,
                };
                let generator = Generator::load_server(
                    &platform,
                    &model,
                    &profile,
                )?;
                let protocol = generator.protocol();
                *protocol_for_thread
                    .write()
                    .expect("protocol lock is not poisoned") = Some(protocol);
                let _ = readiness_tx.send(Readiness::Ready);
                run_loop(
                    generator,
                    receiver,
                    &readiness_tx,
                    max_queued,
                    admission_timeout,
                    scheduler_config,
                    metrics,
                );
                Ok::<(), Error>(())
            })();
            if result.is_err() {
                let _ = readiness_tx.send(Readiness::Failed);
            }
            // Generator, all request sessions, then Platform are destroyed on
            // this engine thread before the terminal readiness transition.
            *protocol_for_thread
                .write()
                .expect("protocol lock is not poisoned") = None;
            if result.is_ok() {
                let _ = readiness_tx.send(Readiness::Stopped);
            }
        })
        .map_err(|error| {
            EngineError::new(
                EngineErrorCode::ExecutionFailed,
                format!("failed to create the model engine thread: {error}"),
            )
        })?;
    if let Err(error) = initialized_rx.recv().map_err(|error| {
        EngineError::new(
            EngineErrorCode::ExecutionFailed,
            format!("model engine initialization handshake failed: {error}"),
        )
    })? {
        let _ = join.join();
        return Err(EngineError::new(EngineErrorCode::ExecutionFailed, error));
    }
    Ok(EngineOwner {
        handle: EngineHandle {
            commands,
            readiness: readiness_rx,
            protocol,
            event_capacity,
        },
        join: Some(join),
    })
}

struct Pending {
    request: PreparedInferenceRequest,
    cancellation: CancellationToken,
    events: mpsc::Sender<EngineEvent>,
    arrived: Instant,
}

struct Active {
    id: RequestId,
    deadline: Option<super::contracts::RequestDeadline>,
    cancellation: CancellationToken,
    events: mpsc::Sender<EngineEvent>,
    session: ServerSession,
}

fn run_loop(
    mut generator: Generator<'_>,
    mut commands: mpsc::Receiver<EngineCommand>,
    readiness: &watch::Sender<Readiness>,
    max_queued: usize,
    admission_timeout: Duration,
    scheduler_config: SchedulerConfig,
    metrics: Metrics,
) {
    let mut queue = VecDeque::<Pending>::new();
    let mut active = BTreeMap::<super::contracts::SequenceId, Active>::new();
    let mut scheduler = match Scheduler::new(scheduler_config) {
        Ok(scheduler) => scheduler,
        Err(_) => {
            let _ = readiness.send(Readiness::Failed);
            return;
        }
    };
    let mut snapshot = EngineSnapshot::default();
    let mut shutdown_reply = None;
    loop {
        let iteration_started = Instant::now();
        let mut drained = 0;
        while drained < MAX_COMMANDS_PER_ITERATION {
            match commands.try_recv() {
                Ok(command) => {
                    drained += 1;
                    if handle_command(
                        &mut generator,
                        command,
                        &mut queue,
                        &mut active,
                        &mut scheduler,
                        &mut snapshot,
                        max_queued,
                        &mut shutdown_reply,
                        readiness,
                    ) {
                        break;
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    shutdown_reply = Some(None);
                    break;
                }
            }
        }

        if shutdown_reply.is_some() {
            cancel_everything(
                &mut generator,
                &mut queue,
                &mut active,
                &mut scheduler,
                CancelReason::Shutdown,
                &mut snapshot,
            );
            if let Some(Some(reply)) = shutdown_reply.take() {
                let _ = reply.send(());
            }
            return;
        }

        cancel_expired(
            &mut generator,
            &mut active,
            &mut scheduler,
            &mut snapshot,
        );
        admit_pending(
            &mut generator,
            &mut queue,
            &mut active,
            &mut scheduler,
            admission_timeout,
            &mut snapshot,
        );
        let plan = match scheduler.plan(Instant::now()) {
            Ok(plan) => plan,
            Err(error) => {
                fail_everything(
                    &mut generator,
                    &mut queue,
                    &mut active,
                    &mut scheduler,
                    error.to_string(),
                    &mut snapshot,
                );
                let _ = readiness.send(Readiness::Failed);
                return;
            }
        };
        let mut submitted = false;
        if let Some(decode) = &plan.decode {
            submitted = true;
            if plan.stable_decode {
                execute_stable_decode(
                    &mut generator,
                    &mut active,
                    &mut scheduler,
                    decode,
                    &commands,
                    &mut snapshot,
                    &metrics,
                );
            } else {
                execute_decode(
                    &mut generator,
                    &mut active,
                    &mut scheduler,
                    decode,
                    &mut snapshot,
                    &metrics,
                );
            }
        }
        if let Some(prefill) = &plan.prefill {
            submitted = true;
            execute_prefill(
                &mut generator,
                &mut active,
                &mut scheduler,
                prefill,
                &mut snapshot,
                &metrics,
            );
        }

        snapshot.active = active.len();
        snapshot.queued = queue.len();
        if drained != 0 || submitted {
            metrics
                .engine_iteration_seconds
                .observe(iteration_started.elapsed().as_secs_f64());
        }
        if submitted || !active.is_empty() {
            continue;
        }
        match commands.blocking_recv() {
            Some(command) => {
                let _ = handle_command(
                    &mut generator,
                    command,
                    &mut queue,
                    &mut active,
                    &mut scheduler,
                    &mut snapshot,
                    max_queued,
                    &mut shutdown_reply,
                    readiness,
                );
            }
            None => {
                cancel_everything(
                    &mut generator,
                    &mut queue,
                    &mut active,
                    &mut scheduler,
                    CancelReason::Shutdown,
                    &mut snapshot,
                );
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_command(
    generator: &mut Generator<'_>,
    command: EngineCommand,
    queue: &mut VecDeque<Pending>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    snapshot: &mut EngineSnapshot,
    max_queued: usize,
    shutdown_reply: &mut Option<Option<oneshot::Sender<()>>>,
    readiness: &watch::Sender<Readiness>,
) -> bool {
    match command {
        EngineCommand::Submit {
            request,
            cancellation,
            events,
        } => {
            if queue.len() >= max_queued {
                let _ = events.try_send(EngineEvent::Failed(EngineError::new(
                    EngineErrorCode::QueueFull,
                    "the bounded engine admission queue is full",
                )));
            } else {
                queue.push_back(Pending {
                    request,
                    cancellation,
                    events,
                    arrived: Instant::now(),
                });
            }
        }
        EngineCommand::Cancel { id, reason } => {
            if let Some(sequence) = active
                .iter()
                .find_map(|(sequence, request)| (request.id == id).then_some(*sequence))
            {
                let request = active.remove(&sequence).expect("located active request exists");
                let _ = scheduler.cancel(sequence);
                cancel_active(generator, request, reason, snapshot);
            } else if let Some(index) = queue.iter().position(|request| request.request.id() == id) {
                let request = queue.remove(index).expect("located queued request exists");
                let _ = request.events.try_send(EngineEvent::Cancelled(reason));
                snapshot.cancelled += 1;
            }
        }
        EngineCommand::Snapshot { reply } => {
            snapshot.active = active.len();
            snapshot.queued = queue.len();
            let cache = generator.cache_stats();
            snapshot.cache_total_pages = cache.total_pages;
            snapshot.cache_free_pages = cache.free_pages;
            snapshot.cache_reserved_pages = cache.reserved_unallocated_pages;
            let _ = reply.send(*snapshot);
        }
        EngineCommand::Shutdown { reply } => {
            let _ = readiness.send(Readiness::ShuttingDown);
            *shutdown_reply = Some(Some(reply));
            return true;
        }
    }
    false
}

fn admit_pending(
    generator: &mut Generator<'_>,
    queue: &mut VecDeque<Pending>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    admission_timeout: Duration,
    snapshot: &mut EngineSnapshot,
) {
    while scheduler.can_admit() {
        let Some(pending) = queue.pop_front() else {
            return;
        };
        if pending.cancellation.is_cancelled() {
            let _ = pending
                .events
                .try_send(EngineEvent::Cancelled(CancelReason::ClientDisconnect));
            snapshot.cancelled += 1;
            continue;
        }
        let now = Instant::now();
        if pending
            .request
            .deadline()
            .is_some_and(|deadline| deadline.is_expired(now))
        {
            let _ = pending
                .events
                .try_send(EngineEvent::Cancelled(CancelReason::Deadline));
            snapshot.cancelled += 1;
            continue;
        }
        if now.saturating_duration_since(pending.arrived) >= admission_timeout {
            let _ = pending
                .events
                .try_send(EngineEvent::Failed(EngineError::new(
                    EngineErrorCode::QueueFull,
                    "request exceeded the bounded admission wait",
                )));
            snapshot.failed += 1;
            continue;
        }
        let mut session = match generator.prepare_server_tokens(
            pending.request.prompt_tokens().to_vec(),
            pending.request.max_new_tokens(),
            pending.request.sampling(),
        ) {
            Ok(session) => session,
            Err(error) => {
                if !active.is_empty()
                    && error
                        .downcast_ref::<super::cache::CacheError>()
                        .is_some_and(|error| {
                            matches!(error, super::cache::CacheError::InsufficientCapacity { .. })
                        })
                {
                    queue.push_front(pending);
                    return;
                }
                let _ = pending.events.try_send(EngineEvent::Failed(EngineError::new(
                    EngineErrorCode::ExecutionFailed,
                    error.to_string(),
                )));
                snapshot.failed += 1;
                continue;
            }
        };
        if session.is_complete() {
            let completion = EngineCompletion {
                prompt_tokens: session.prompt_tokens(),
                completion_tokens: 0,
                stopped: false,
            };
            let _ = generator.release_server_session(&mut session);
            if pending.events.try_send(EngineEvent::Complete(completion)).is_ok() {
                snapshot.completed += 1;
            } else {
                snapshot.cancelled += 1;
            }
            continue;
        }
        let sequence = session.sequence();
        let scheduled = scheduler
            .enqueue(
                sequence,
                session.prompt_tokens(),
                pending.request.total_token_budget(),
                pending.arrived,
            )
            .and_then(|_| scheduler.admit_reserved(sequence, now));
        if let Err(error) = scheduled {
            let _ = generator.release_server_session(&mut session);
            let _ = pending.events.try_send(EngineEvent::Failed(EngineError::new(
                EngineErrorCode::ExecutionFailed,
                error.to_string(),
            )));
            snapshot.failed += 1;
            continue;
        }
        active.insert(sequence, Active {
            id: pending.request.id(),
            deadline: pending.request.deadline(),
            cancellation: pending.cancellation,
            events: pending.events,
            session,
        });
    }
}

fn cancel_expired(
    generator: &mut Generator<'_>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    snapshot: &mut EngineSnapshot,
) {
    let now = Instant::now();
    let terminal = active
        .iter()
        .filter_map(|(sequence, request)| {
            if request.cancellation.is_cancelled() {
                Some((*sequence, CancelReason::ClientDisconnect))
            } else if request
                .deadline
                .is_some_and(|deadline| deadline.is_expired(now))
            {
                Some((*sequence, CancelReason::Deadline))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for (sequence, reason) in terminal {
        if let Some(request) = active.remove(&sequence) {
            let _ = scheduler.cancel(sequence);
            cancel_active(generator, request, reason, snapshot);
        }
    }
}

fn cancel_active(
    generator: &mut Generator<'_>,
    mut active: Active,
    reason: CancelReason,
    snapshot: &mut EngineSnapshot,
) {
    active.cancellation.cancel();
    if let Err(error) = generator.release_server_session(&mut active.session) {
        let _ = active.events.try_send(EngineEvent::Failed(EngineError::new(
            EngineErrorCode::ExecutionFailed,
            error.to_string(),
        )));
        snapshot.failed += 1;
        return;
    }
    let _ = active.events.try_send(EngineEvent::Cancelled(reason));
    snapshot.cancelled += 1;
}

fn take_rows(
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    items: &[BatchItem],
) -> Result<Vec<Active>, EngineError> {
    items
        .iter()
        .map(|item| {
            active.remove(&item.sequence).ok_or_else(|| {
                EngineError::new(
                    EngineErrorCode::ExecutionFailed,
                    "batch plan referenced a missing active request",
                )
            })
        })
        .collect()
}

/// Runs one complete, stable decode membership through the generic resident
/// batch lane. The scheduler is not re-entered between tokens while every row
/// survives and no queued work needs service. Any finish, cancellation,
/// backpressure event, page-family change, or command returns survivors to the
/// scheduler at the visible-token boundary.
fn execute_stable_decode(
    generator: &mut Generator<'_>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    submission: &BatchSubmission,
    commands: &mpsc::Receiver<EngineCommand>,
    snapshot: &mut EngineSnapshot,
    metrics: &Metrics,
) {
    let mut rows = match take_rows(active, &submission.items) {
        Ok(rows) => rows,
        Err(_) => {
            snapshot.failed += 1;
            return;
        }
    };
    loop {
        metrics.decode_batch_rows.observe(rows.len() as f64);
        let started = Instant::now();
        let result = {
            let mut sessions = rows
                .iter_mut()
                .map(|request| &mut request.session)
                .collect::<Vec<_>>();
            generator.decode_batch(&mut sessions, submission.family_capacity)
        };
        metrics
            .decode_batch_seconds
            .observe(started.elapsed().as_secs_f64());
        let raw = match result {
            Ok(raw) => raw,
            Err(error) => {
                fail_rows(
                    generator,
                    scheduler,
                    rows,
                    error.to_string(),
                    snapshot,
                );
                return;
            }
        };
        let now = Instant::now();
        let previous_rows = rows.len();
        let mut survivors = Vec::with_capacity(previous_rows);
        for (mut request, raw) in rows.into_iter().zip(raw) {
            let sequence = request.session.sequence();
            let delivered = request.events.try_send(EngineEvent::Raw(raw)).is_ok();
            if request.session.is_complete() || !delivered {
                if scheduler.complete_decode(sequence, true).is_err() {
                    fail_rows(
                        generator,
                        scheduler,
                        vec![request],
                        "stable decode result did not match its scheduler state".to_owned(),
                        snapshot,
                    );
                } else if delivered {
                    complete_active(generator, request, snapshot);
                } else {
                    request.cancellation.cancel();
                    let _ = generator.release_server_session(&mut request.session);
                    snapshot.cancelled += 1;
                }
                continue;
            }
            let cancellation = if request.cancellation.is_cancelled() {
                Some(CancelReason::ClientDisconnect)
            } else if request
                .deadline
                .is_some_and(|deadline| deadline.is_expired(now))
            {
                Some(CancelReason::Deadline)
            } else {
                None
            };
            if let Some(reason) = cancellation {
                if scheduler.cancel(sequence).is_err() {
                    fail_rows(
                        generator,
                        scheduler,
                        vec![request],
                        "stable decode cancellation did not match its scheduler state".to_owned(),
                        snapshot,
                    );
                } else {
                    cancel_active(generator, request, reason, snapshot);
                }
            } else {
                survivors.push(request);
            }
        }

        let membership_changed = survivors.len() != previous_rows;
        if membership_changed || commands.is_closed() || !commands.is_empty() {
            for request in survivors {
                let sequence = request.session.sequence();
                if scheduler.complete_decode(sequence, false).is_err() {
                    fail_rows(
                        generator,
                        scheduler,
                        vec![request],
                        "stable decode continuation did not match its scheduler state".to_owned(),
                        snapshot,
                    );
                } else {
                    active.insert(sequence, request);
                }
            }
            return;
        }
        rows = survivors;
        // All rows remain InFlightDecode. The next token consumes the donated
        // device slab directly without rebuilding or uploading the batch.
    }
}

fn execute_decode(
    generator: &mut Generator<'_>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    submission: &BatchSubmission,
    snapshot: &mut EngineSnapshot,
    metrics: &Metrics,
) {
    let mut rows = match take_rows(active, &submission.items) {
        Ok(rows) => rows,
        Err(error) => {
            snapshot.failed += 1;
            let _ = error;
            return;
        }
    };
    metrics.decode_batch_rows.observe(rows.len() as f64);
    let started = Instant::now();
    let result = {
        let mut sessions = rows
            .iter_mut()
            .map(|request| &mut request.session)
            .collect::<Vec<_>>();
        generator.decode_batch(&mut sessions, submission.family_capacity)
    };
    metrics
        .decode_batch_seconds
        .observe(started.elapsed().as_secs_f64());
    let raw = match result {
        Ok(raw) => raw,
        Err(error) => {
            fail_rows(generator, scheduler, rows, error.to_string(), snapshot);
            return;
        }
    };
    for (mut request, raw) in rows.into_iter().zip(raw) {
        let sequence = request.session.sequence();
        let delivered = request.events.try_send(EngineEvent::Raw(raw)).is_ok();
        let terminal = request.session.is_complete() || !delivered;
        if scheduler.complete_decode(sequence, terminal).is_err() {
            fail_rows(
                generator,
                scheduler,
                vec![request],
                "decode result did not match its scheduler plan".to_owned(),
                snapshot,
            );
            continue;
        }
        if terminal {
            if delivered {
                complete_active(generator, request, snapshot);
            } else {
                request.cancellation.cancel();
                let _ = generator.release_server_session(&mut request.session);
                snapshot.cancelled += 1;
            }
        } else {
            active.insert(sequence, request);
        }
    }
}

fn execute_prefill(
    generator: &mut Generator<'_>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    submission: &BatchSubmission,
    snapshot: &mut EngineSnapshot,
    metrics: &Metrics,
) {
    let mut rows = match take_rows(active, &submission.items) {
        Ok(rows) => rows,
        Err(_) => {
            snapshot.failed += 1;
            return;
        }
    };
    let chunks = submission.items.iter().map(|item| item.tokens).collect::<Vec<_>>();
    metrics.prefill_batch_rows.observe(rows.len() as f64);
    metrics
        .prefill_batch_tokens
        .observe(chunks.iter().sum::<usize>() as f64);
    let started = Instant::now();
    let result = {
        let mut sessions = rows
            .iter_mut()
            .map(|request| &mut request.session)
            .collect::<Vec<_>>();
        generator.prefill_batch(
            &mut sessions,
            &chunks,
            submission.family_capacity,
            submission.query_capacity,
        )
    };
    metrics
        .prefill_batch_seconds
        .observe(started.elapsed().as_secs_f64());
    let raw = match result {
        Ok(raw) => raw,
        Err(error) => {
            fail_rows(generator, scheduler, rows, error.to_string(), snapshot);
            return;
        }
    };
    for ((mut request, raw), item) in rows.into_iter().zip(raw).zip(&submission.items) {
        let sequence = request.session.sequence();
        if scheduler.complete_prefill(sequence, item.tokens).is_err() {
            fail_rows(
                generator,
                scheduler,
                vec![request],
                "prefill result did not match its scheduler plan".to_owned(),
                snapshot,
            );
            continue;
        }
        let delivered = raw
            .map(|raw| request.events.try_send(EngineEvent::Raw(raw)).is_ok())
            .unwrap_or(true);
        let terminal = request.session.is_complete() || !delivered;
        if terminal {
            let _ = scheduler.remove_terminal(sequence);
            if delivered {
                complete_active(generator, request, snapshot);
            } else {
                request.cancellation.cancel();
                let _ = generator.release_server_session(&mut request.session);
                snapshot.cancelled += 1;
            }
        } else {
            active.insert(sequence, request);
        }
    }
}

fn complete_active(
    generator: &mut Generator<'_>,
    mut active: Active,
    snapshot: &mut EngineSnapshot,
) {
    let completion = EngineCompletion {
        prompt_tokens: active.session.prompt_tokens(),
        completion_tokens: active.session.completion_tokens(),
        stopped: active.session.stopped(),
    };
    if let Err(error) = generator.release_server_session(&mut active.session) {
        let _ = active.events.try_send(EngineEvent::Failed(EngineError::new(
            EngineErrorCode::ExecutionFailed,
            error.to_string(),
        )));
        snapshot.failed += 1;
    } else if active.events.try_send(EngineEvent::Complete(completion)).is_ok() {
        snapshot.completed += 1;
    } else {
        snapshot.cancelled += 1;
    }
}

fn fail_rows(
    generator: &mut Generator<'_>,
    scheduler: &mut Scheduler,
    rows: Vec<Active>,
    message: String,
    snapshot: &mut EngineSnapshot,
) {
    for mut request in rows {
        let _ = scheduler.cancel(request.session.sequence());
        let _ = generator.release_server_session(&mut request.session);
        let _ = request.events.try_send(EngineEvent::Failed(EngineError::new(
            EngineErrorCode::ExecutionFailed,
            message.clone(),
        )));
        snapshot.failed += 1;
    }
}

fn cancel_everything(
    generator: &mut Generator<'_>,
    queue: &mut VecDeque<Pending>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    reason: CancelReason,
    snapshot: &mut EngineSnapshot,
) {
    for (sequence, request) in std::mem::take(active) {
        let _ = scheduler.cancel(sequence);
        cancel_active(generator, request, reason, snapshot);
    }
    for pending in queue.drain(..) {
        pending.cancellation.cancel();
        let _ = pending.events.try_send(EngineEvent::Cancelled(reason));
        snapshot.cancelled += 1;
    }
}

fn fail_everything(
    generator: &mut Generator<'_>,
    queue: &mut VecDeque<Pending>,
    active: &mut BTreeMap<super::contracts::SequenceId, Active>,
    scheduler: &mut Scheduler,
    message: String,
    snapshot: &mut EngineSnapshot,
) {
    let rows = std::mem::take(active).into_values().collect();
    fail_rows(generator, scheduler, rows, message.clone(), snapshot);
    for pending in queue.drain(..) {
        let _ = pending.events.try_send(EngineEvent::Failed(EngineError::new(
            EngineErrorCode::ExecutionFailed,
            message.clone(),
        )));
        snapshot.failed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_and_snapshot_domains_are_bounded() {
        let readiness = [
            Readiness::Starting,
            Readiness::Ready,
            Readiness::Failed,
            Readiness::ShuttingDown,
            Readiness::Stopped,
        ];
        assert_eq!(readiness.len(), 5);
        assert_eq!(EngineSnapshot::default().active, 0);
        assert_eq!(MAX_COMMANDS_PER_ITERATION, 64);
    }
}
