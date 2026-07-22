//! Deterministic host-side continuous-batching policy.
//!
//! This module owns scheduling identity and phase transitions, but deliberately
//! owns neither device buffers nor cache pages. Admission is a two-step
//! protocol: the caller inspects the oldest [`AdmissionCandidate`], obtains a
//! complete-lifetime page reservation from `PageManager`, and only then calls
//! [`Scheduler::admit_reserved`]. Consequently every sequence visible to a
//! prefill or decode plan already has credits that guarantee completion.

#![allow(dead_code)]

use super::contracts::SequenceId;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error as StdError;
use std::fmt;
use std::time::{Duration, Instant};

const SINGLE_SEQUENCE_LOOKAHEAD_PAIRS: usize = 5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerConfig {
    pub(crate) batch_buckets: Vec<usize>,
    pub(crate) prefill_query_buckets: Vec<usize>,
    pub(crate) max_active_sequences: usize,
    pub(crate) max_batched_tokens: usize,
    pub(crate) max_prefill_chunk: usize,
    pub(crate) max_prefill_wait: Duration,
}

impl SchedulerConfig {
    pub(crate) fn validate(&self) -> Result<(), SchedulerError> {
        if self.batch_buckets.is_empty()
            || self.prefill_query_buckets.is_empty()
            || self.max_active_sequences == 0
            || self.max_batched_tokens == 0
            || self.max_prefill_chunk == 0
            || self.max_prefill_wait.is_zero()
        {
            return Err(SchedulerError::InvalidConfig(
                "scheduler capacities and prefill wait must be nonzero",
            ));
        }
        if self.batch_buckets[0] != 1
            || self
                .batch_buckets
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(SchedulerError::InvalidConfig(
                "batch buckets must be strictly increasing and begin at one",
            ));
        }
        if self.prefill_query_buckets[0] == 0
            || self
                .prefill_query_buckets
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || *self.prefill_query_buckets.last().unwrap() < self.max_prefill_chunk
        {
            return Err(SchedulerError::InvalidConfig(
                "prefill query buckets must be increasing and cover the maximum chunk",
            ));
        }
        let largest = *self.batch_buckets.last().unwrap();
        if largest > self.max_active_sequences || largest > self.max_batched_tokens {
            return Err(SchedulerError::InvalidConfig(
                "largest batch exceeds the active-sequence or token budget",
            ));
        }
        if self.max_prefill_chunk > self.max_batched_tokens {
            return Err(SchedulerError::InvalidConfig(
                "prefill chunk exceeds the iteration token budget",
            ));
        }
        Ok(())
    }

    fn family_for(&self, rows: usize) -> Result<usize, SchedulerError> {
        self.batch_buckets
            .iter()
            .copied()
            .find(|capacity| *capacity >= rows)
            .ok_or(SchedulerError::NoBatchFamily { rows })
    }

    fn prefill_family_for(&self, tokens: usize) -> Result<usize, SchedulerError> {
        self.prefill_query_buckets
            .iter()
            .copied()
            .find(|capacity| *capacity >= tokens)
            .ok_or(SchedulerError::NoQueryFamily { tokens })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionCandidate {
    pub(crate) sequence: SequenceId,
    pub(crate) prompt_tokens: usize,
    pub(crate) maximum_sequence_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScheduledPhase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BatchItem {
    /// Stable sequence identity. Its position in `items` is only this
    /// submission's slot and is never retained as request identity.
    pub(crate) sequence: SequenceId,
    /// One for decode; a bounded positive query chunk for prefill.
    pub(crate) tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BatchSubmission {
    pub(crate) phase: ScheduledPhase,
    pub(crate) family_capacity: usize,
    pub(crate) query_capacity: usize,
    pub(crate) items: Vec<BatchItem>,
}

impl BatchSubmission {
    pub(crate) fn active_tokens(&self) -> usize {
        self.items.iter().map(|item| item.tokens).sum()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct BatchPlan {
    /// Decode remains the latency-sensitive first device submission.
    pub(crate) decode: Option<BatchSubmission>,
    /// Prefill is a separate submission, so a long prompt cannot sit in front
    /// of the decode rows selected in the same scheduler iteration.
    pub(crate) prefill: Option<BatchSubmission>,
    pub(crate) single_sequence_lookahead_pairs: usize,
}

impl BatchPlan {
    pub(crate) fn active_tokens(&self) -> usize {
        self.decode
            .iter()
            .chain(self.prefill.iter())
            .map(BatchSubmission::active_tokens)
            .sum()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.decode.is_none() && self.prefill.is_none()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CancellationState {
    Waiting,
    Prefill,
    Decode,
    InFlight(ScheduledPhase),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Waiting,
    Prefill,
    Decode,
    InFlightPrefill { planned_tokens: usize },
    InFlightDecode,
}

#[derive(Clone, Copy, Debug)]
struct SequenceState {
    prompt_tokens: usize,
    prompt_remaining: usize,
    maximum_sequence_tokens: usize,
    arrived: Instant,
    admitted: Option<Instant>,
    phase: Phase,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SchedulerError {
    InvalidConfig(&'static str),
    DuplicateSequence(SequenceId),
    UnknownSequence(SequenceId),
    NotWaiting(SequenceId),
    ActiveCapacityFull,
    NoBatchFamily { rows: usize },
    NoQueryFamily { tokens: usize },
    ResultDoesNotMatchPlan(SequenceId),
    ArithmeticOverflow,
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "invalid scheduler config: {message}"),
            Self::DuplicateSequence(sequence) => {
                write!(formatter, "duplicate scheduler sequence {}", sequence.as_u64())
            }
            Self::UnknownSequence(sequence) => {
                write!(formatter, "unknown scheduler sequence {}", sequence.as_u64())
            }
            Self::NotWaiting(sequence) => write!(
                formatter,
                "scheduler sequence {} is not waiting for admission",
                sequence.as_u64()
            ),
            Self::ActiveCapacityFull => write!(formatter, "scheduler active capacity is full"),
            Self::NoBatchFamily { rows } => {
                write!(formatter, "no compiled batch family can hold {rows} rows")
            }
            Self::NoQueryFamily { tokens } => {
                write!(formatter, "no compiled prefill family can hold {tokens} tokens")
            }
            Self::ResultDoesNotMatchPlan(sequence) => write!(
                formatter,
                "result does not match the in-flight plan for sequence {}",
                sequence.as_u64()
            ),
            Self::ArithmeticOverflow => write!(formatter, "scheduler accounting overflow"),
        }
    }
}

impl StdError for SchedulerError {}

/// FIFO phase queues plus request-identity state for dynamic batching.
pub(crate) struct Scheduler {
    config: SchedulerConfig,
    states: BTreeMap<SequenceId, SequenceState>,
    waiting: VecDeque<SequenceId>,
    prefill: VecDeque<SequenceId>,
    decode: VecDeque<SequenceId>,
    active: usize,
}

impl Scheduler {
    pub(crate) fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        config.validate()?;
        Ok(Self {
            config,
            states: BTreeMap::new(),
            waiting: VecDeque::new(),
            prefill: VecDeque::new(),
            decode: VecDeque::new(),
            active: 0,
        })
    }

    pub(crate) fn enqueue(
        &mut self,
        sequence: SequenceId,
        prompt_tokens: usize,
        maximum_sequence_tokens: usize,
        now: Instant,
    ) -> Result<(), SchedulerError> {
        if prompt_tokens == 0 || maximum_sequence_tokens < prompt_tokens {
            return Err(SchedulerError::InvalidConfig(
                "a request needs a nonempty prompt within its sequence budget",
            ));
        }
        if self.states.contains_key(&sequence) {
            return Err(SchedulerError::DuplicateSequence(sequence));
        }
        self.states.insert(
            sequence,
            SequenceState {
                prompt_tokens,
                prompt_remaining: prompt_tokens,
                maximum_sequence_tokens,
                arrived: now,
                admitted: None,
                phase: Phase::Waiting,
            },
        );
        self.waiting.push_back(sequence);
        Ok(())
    }

    /// Returns the oldest request without changing its admission state. The
    /// caller must reserve its complete cache budget before acknowledging it.
    pub(crate) fn next_admission(&self) -> Option<AdmissionCandidate> {
        let sequence = *self.waiting.front()?;
        let state = self.states.get(&sequence)?;
        Some(AdmissionCandidate {
            sequence,
            prompt_tokens: state.prompt_tokens,
            maximum_sequence_tokens: state.maximum_sequence_tokens,
        })
    }

    /// Acknowledges that the cache manager has granted complete-lifetime page
    /// credits. There is intentionally no method that admits an unreserved
    /// request.
    pub(crate) fn admit_reserved(
        &mut self,
        sequence: SequenceId,
        now: Instant,
    ) -> Result<(), SchedulerError> {
        if self.active >= self.config.max_active_sequences {
            return Err(SchedulerError::ActiveCapacityFull);
        }
        if self.waiting.front().copied() != Some(sequence) {
            return Err(SchedulerError::NotWaiting(sequence));
        }
        let state = self
            .states
            .get_mut(&sequence)
            .ok_or(SchedulerError::UnknownSequence(sequence))?;
        if state.phase != Phase::Waiting {
            return Err(SchedulerError::NotWaiting(sequence));
        }
        self.waiting.pop_front();
        state.phase = Phase::Prefill;
        state.admitted = Some(now);
        self.prefill.push_back(sequence);
        self.active += 1;
        Ok(())
    }

    /// Produces one immutable decode submission and one immutable prefill
    /// submission. Selected rows become in-flight until their exact results
    /// are applied or cancelled.
    pub(crate) fn plan(&mut self, now: Instant) -> Result<BatchPlan, SchedulerError> {
        let mut budget = self.config.max_batched_tokens;
        let aged_prefill = self.prefill.iter().copied().find(|sequence| {
            self.states
                .get(sequence)
                .and_then(|state| state.admitted)
                .is_some_and(|admitted| now.saturating_duration_since(admitted) >= self.config.max_prefill_wait)
        });
        // Reserve one real token of an aged prompt before filling the decode
        // budget. Query-family padding does not count as semantic token work.
        let reserved_prefill_tokens = usize::from(aged_prefill.is_some() && budget != 0);

        let decode_budget = budget.saturating_sub(reserved_prefill_tokens);
        let decode_rows = decode_budget
            .min(self.decode.len())
            .min(*self.config.batch_buckets.last().unwrap());
        let mut decode_items = Vec::with_capacity(decode_rows);
        for _ in 0..decode_rows {
            let sequence = self.decode.pop_front().unwrap();
            let state = self.states.get_mut(&sequence).unwrap();
            debug_assert_eq!(state.phase, Phase::Decode);
            state.phase = Phase::InFlightDecode;
            decode_items.push(BatchItem {
                sequence,
                tokens: 1,
            });
        }
        budget -= decode_items.len();

        let mut prefill_items = Vec::new();
        if let Some(sequence) = aged_prefill {
            if budget != 0 {
                self.take_prefill(sequence, &mut budget, &mut prefill_items)?;
            }
        }
        let maximum_rows = *self.config.batch_buckets.last().unwrap();
        while budget != 0 && prefill_items.len() < maximum_rows {
            let Some(sequence) = self.prefill.front().copied() else {
                break;
            };
            if prefill_items.iter().any(|item| item.sequence == sequence) {
                break;
            }
            self.take_prefill(sequence, &mut budget, &mut prefill_items)?;
        }

        let decode = if decode_items.is_empty() {
            None
        } else {
            Some(BatchSubmission {
                phase: ScheduledPhase::Decode,
                family_capacity: self.config.family_for(decode_items.len())?,
                query_capacity: 1,
                items: decode_items,
            })
        };
        let prefill = if prefill_items.is_empty() {
            None
        } else {
            Some(BatchSubmission {
                phase: ScheduledPhase::Prefill,
                family_capacity: self.config.family_for(prefill_items.len())?,
                query_capacity: self.config.prefill_family_for(
                    prefill_items.iter().map(|item| item.tokens).max().unwrap(),
                )?,
                items: prefill_items,
            })
        };
        let lookahead = if decode
            .as_ref()
            .is_some_and(|submission| submission.items.len() == 1)
            && prefill.is_none()
            && self.waiting.is_empty()
            && self.prefill.is_empty()
            && self.decode.is_empty()
            && self.active == 1
        {
            SINGLE_SEQUENCE_LOOKAHEAD_PAIRS
        } else {
            0
        };
        Ok(BatchPlan {
            decode,
            prefill,
            single_sequence_lookahead_pairs: lookahead,
        })
    }

    fn take_prefill(
        &mut self,
        sequence: SequenceId,
        budget: &mut usize,
        items: &mut Vec<BatchItem>,
    ) -> Result<(), SchedulerError> {
        let index = self
            .prefill
            .iter()
            .position(|candidate| *candidate == sequence)
            .ok_or(SchedulerError::UnknownSequence(sequence))?;
        self.prefill.remove(index);
        let state = self.states.get_mut(&sequence).unwrap();
        if state.phase != Phase::Prefill || state.prompt_remaining == 0 {
            return Err(SchedulerError::ResultDoesNotMatchPlan(sequence));
        }
        let tokens = state
            .prompt_remaining
            .min(self.config.max_prefill_chunk)
            .min(*budget);
        if tokens == 0 {
            self.prefill.insert(index, sequence);
            return Ok(());
        }
        state.phase = Phase::InFlightPrefill {
            planned_tokens: tokens,
        };
        items.push(BatchItem { sequence, tokens });
        *budget -= tokens;
        Ok(())
    }

    pub(crate) fn complete_prefill(
        &mut self,
        sequence: SequenceId,
        processed_tokens: usize,
    ) -> Result<bool, SchedulerError> {
        let state = self
            .states
            .get_mut(&sequence)
            .ok_or(SchedulerError::UnknownSequence(sequence))?;
        let Phase::InFlightPrefill { planned_tokens } = state.phase else {
            return Err(SchedulerError::ResultDoesNotMatchPlan(sequence));
        };
        if processed_tokens != planned_tokens || processed_tokens > state.prompt_remaining {
            return Err(SchedulerError::ResultDoesNotMatchPlan(sequence));
        }
        state.prompt_remaining -= processed_tokens;
        if state.prompt_remaining == 0 {
            state.phase = Phase::Decode;
            self.decode.push_back(sequence);
            Ok(true)
        } else {
            state.phase = Phase::Prefill;
            self.prefill.push_back(sequence);
            Ok(false)
        }
    }

    /// Requeues a nonterminal decode row at the tail. A terminal row leaves
    /// scheduler ownership immediately; cache/response cleanup remains the
    /// caller's single terminal path.
    pub(crate) fn complete_decode(
        &mut self,
        sequence: SequenceId,
        terminal: bool,
    ) -> Result<(), SchedulerError> {
        let state = self
            .states
            .get_mut(&sequence)
            .ok_or(SchedulerError::UnknownSequence(sequence))?;
        if state.phase != Phase::InFlightDecode {
            return Err(SchedulerError::ResultDoesNotMatchPlan(sequence));
        }
        if terminal {
            self.states.remove(&sequence);
            self.active -= 1;
        } else {
            state.phase = Phase::Decode;
            self.decode.push_back(sequence);
        }
        Ok(())
    }

    pub(crate) fn cancel(
        &mut self,
        sequence: SequenceId,
    ) -> Result<CancellationState, SchedulerError> {
        let state = self
            .states
            .remove(&sequence)
            .ok_or(SchedulerError::UnknownSequence(sequence))?;
        let cancelled = match state.phase {
            Phase::Waiting => {
                remove(&mut self.waiting, sequence);
                CancellationState::Waiting
            }
            Phase::Prefill => {
                remove(&mut self.prefill, sequence);
                self.active -= 1;
                CancellationState::Prefill
            }
            Phase::Decode => {
                remove(&mut self.decode, sequence);
                self.active -= 1;
                CancellationState::Decode
            }
            Phase::InFlightPrefill { .. } => {
                self.active -= 1;
                CancellationState::InFlight(ScheduledPhase::Prefill)
            }
            Phase::InFlightDecode => {
                self.active -= 1;
                CancellationState::InFlight(ScheduledPhase::Decode)
            }
        };
        Ok(cancelled)
    }

    /// Removes a terminal sequence after its successful result has already
    /// transitioned out of an in-flight phase.
    pub(crate) fn remove_terminal(
        &mut self,
        sequence: SequenceId,
    ) -> Result<(), SchedulerError> {
        let state = self
            .states
            .remove(&sequence)
            .ok_or(SchedulerError::UnknownSequence(sequence))?;
        match state.phase {
            Phase::Prefill => remove(&mut self.prefill, sequence),
            Phase::Decode => remove(&mut self.decode, sequence),
            Phase::Waiting => {
                self.states.insert(sequence, state);
                return Err(SchedulerError::NotWaiting(sequence));
            }
            Phase::InFlightPrefill { .. } | Phase::InFlightDecode => {
                self.states.insert(sequence, state);
                return Err(SchedulerError::ResultDoesNotMatchPlan(sequence));
            }
        }
        self.active -= 1;
        Ok(())
    }

    pub(crate) fn queued(&self) -> usize {
        self.waiting.len()
    }

    pub(crate) fn active(&self) -> usize {
        self.active
    }

    pub(crate) fn can_admit(&self) -> bool {
        self.active < self.config.max_active_sequences
    }

    pub(crate) fn phase_counts(&self) -> (usize, usize, usize) {
        (self.waiting.len(), self.prefill.len(), self.decode.len())
    }

    pub(crate) fn arrival(&self, sequence: SequenceId) -> Option<Instant> {
        self.states.get(&sequence).map(|state| state.arrived)
    }
}

fn remove(queue: &mut VecDeque<SequenceId>, sequence: SequenceId) {
    if let Some(index) = queue.iter().position(|candidate| *candidate == sequence) {
        queue.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::cache::{CacheError, PageManager};

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            batch_buckets: vec![1, 2, 4, 8],
            prefill_query_buckets: vec![1, 2, 4],
            max_active_sequences: 8,
            max_batched_tokens: 8,
            max_prefill_chunk: 4,
            max_prefill_wait: Duration::from_millis(10),
        }
    }

    fn id(value: u64) -> SequenceId {
        SequenceId::new(value)
    }

    fn admit(
        scheduler: &mut Scheduler,
        pages: &mut PageManager,
        sequence: SequenceId,
        now: Instant,
    ) -> Result<(), CacheError> {
        let candidate = scheduler.next_admission().unwrap();
        assert_eq!(candidate.sequence, sequence);
        pages.reserve_tokens(sequence, candidate.maximum_sequence_tokens)?;
        scheduler.admit_reserved(sequence, now).unwrap();
        Ok(())
    }

    #[test]
    fn config_and_smallest_family_are_bounded() {
        let scheduler = Scheduler::new(config()).unwrap();
        assert_eq!(scheduler.config.family_for(1).unwrap(), 1);
        assert_eq!(scheduler.config.family_for(3).unwrap(), 4);
        assert!(scheduler.config.family_for(9).is_err());

        let mut invalid = config();
        invalid.batch_buckets = vec![1, 4, 4];
        assert!(Scheduler::new(invalid).is_err());
    }

    #[test]
    fn admission_waits_until_complete_lifetime_pages_are_reserved() {
        let now = Instant::now();
        let mut scheduler = Scheduler::new(config()).unwrap();
        let mut pages = PageManager::new(2).unwrap();
        scheduler.enqueue(id(1), 16, 32, now).unwrap();
        scheduler.enqueue(id(2), 16, 32, now).unwrap();

        admit(&mut scheduler, &mut pages, id(1), now).unwrap();
        assert_eq!(scheduler.next_admission().unwrap().sequence, id(2));
        assert!(matches!(
            pages.reserve_tokens(id(2), 32),
            Err(CacheError::InsufficientCapacity { .. })
        ));
        assert_eq!(scheduler.phase_counts(), (1, 1, 0));
        assert_eq!(scheduler.active(), 1);

        scheduler.cancel(id(1)).unwrap();
        pages.release_sequence(id(1)).unwrap();
        admit(&mut scheduler, &mut pages, id(2), now).unwrap();
        assert_eq!(scheduler.phase_counts(), (0, 1, 0));
    }

    #[test]
    fn decode_is_first_but_aged_prefill_cannot_starve() {
        let now = Instant::now();
        let mut selected = config();
        selected.batch_buckets = vec![1, 2, 4];
        selected.max_batched_tokens = 4;
        selected.max_prefill_chunk = 4;
        let mut scheduler = Scheduler::new(selected).unwrap();
        let mut pages = PageManager::new(16).unwrap();

        for sequence in [id(1), id(2), id(3)] {
            scheduler.enqueue(sequence, 1, 16, now).unwrap();
            admit(&mut scheduler, &mut pages, sequence, now).unwrap();
        }
        let initial = scheduler.plan(now).unwrap();
        for item in &initial.prefill.unwrap().items {
            scheduler.complete_prefill(item.sequence, item.tokens).unwrap();
        }

        scheduler.enqueue(id(4), 8, 16, now).unwrap();
        admit(&mut scheduler, &mut pages, id(4), now).unwrap();
        let plan = scheduler
            .plan(now + Duration::from_millis(11))
            .unwrap();
        assert_eq!(plan.active_tokens(), 4);
        assert_eq!(plan.single_sequence_lookahead_pairs, 0);
        let decode = plan.decode.unwrap();
        let prefill = plan.prefill.unwrap();
        assert_eq!(decode.phase, ScheduledPhase::Decode);
        assert_eq!(decode.items.len(), 3);
        assert_eq!(prefill.items, [BatchItem { sequence: id(4), tokens: 1 }]);
    }

    #[test]
    fn chunks_round_robin_and_membership_repacks_every_iteration() {
        let now = Instant::now();
        let mut scheduler = Scheduler::new(config()).unwrap();
        let mut pages = PageManager::new(32).unwrap();
        for sequence in [id(10), id(20), id(30)] {
            scheduler.enqueue(sequence, 6, 16, now).unwrap();
            admit(&mut scheduler, &mut pages, sequence, now).unwrap();
        }

        let first = scheduler.plan(now).unwrap();
        let first = first.prefill.unwrap();
        assert_eq!(first.family_capacity, 2);
        assert_eq!(first.items[0], BatchItem { sequence: id(10), tokens: 4 });
        assert_eq!(first.items[1], BatchItem { sequence: id(20), tokens: 4 });
        for item in first.items {
            scheduler.complete_prefill(item.sequence, item.tokens).unwrap();
        }

        scheduler.cancel(id(20)).unwrap();
        pages.release_sequence(id(20)).unwrap();
        let second = scheduler.plan(now).unwrap().prefill.unwrap();
        assert_eq!(second.family_capacity, 2);
        assert_eq!(
            second.items,
            [
                BatchItem { sequence: id(30), tokens: 4 },
                BatchItem { sequence: id(10), tokens: 2 },
            ]
        );
        assert_eq!(second.items[1].sequence, id(10));
        // Slot one in the first batch became slot one in neither identity nor
        // lifecycle terms; only SequenceId links results to request state.
    }

    #[test]
    fn lookahead_is_exactly_the_idle_batch_one_fast_path() {
        let now = Instant::now();
        let mut scheduler = Scheduler::new(config()).unwrap();
        let mut pages = PageManager::new(8).unwrap();
        scheduler.enqueue(id(1), 1, 8, now).unwrap();
        admit(&mut scheduler, &mut pages, id(1), now).unwrap();
        let prefill = scheduler.plan(now).unwrap().prefill.unwrap();
        scheduler
            .complete_prefill(id(1), prefill.items[0].tokens)
            .unwrap();
        let decode = scheduler.plan(now).unwrap();
        assert_eq!(decode.decode.as_ref().unwrap().family_capacity, 1);
        assert_eq!(decode.single_sequence_lookahead_pairs, 5);
        scheduler.complete_decode(id(1), false).unwrap();

        scheduler.enqueue(id(2), 1, 8, now).unwrap();
        let with_waiter = scheduler.plan(now).unwrap();
        assert_eq!(with_waiter.single_sequence_lookahead_pairs, 0);
    }

    #[test]
    fn cancellation_is_total_in_every_scheduler_state() {
        let now = Instant::now();
        let mut scheduler = Scheduler::new(config()).unwrap();
        scheduler.enqueue(id(1), 4, 8, now).unwrap();
        assert_eq!(scheduler.cancel(id(1)).unwrap(), CancellationState::Waiting);

        scheduler.enqueue(id(2), 4, 8, now).unwrap();
        scheduler.admit_reserved(id(2), now).unwrap();
        assert_eq!(scheduler.cancel(id(2)).unwrap(), CancellationState::Prefill);

        scheduler.enqueue(id(3), 1, 8, now).unwrap();
        scheduler.admit_reserved(id(3), now).unwrap();
        let prefill = scheduler.plan(now).unwrap().prefill.unwrap();
        assert_eq!(
            scheduler.cancel(prefill.items[0].sequence).unwrap(),
            CancellationState::InFlight(ScheduledPhase::Prefill)
        );

        scheduler.enqueue(id(4), 1, 8, now).unwrap();
        scheduler.admit_reserved(id(4), now).unwrap();
        let prefill = scheduler.plan(now).unwrap().prefill.unwrap();
        scheduler
            .complete_prefill(id(4), prefill.items[0].tokens)
            .unwrap();
        let decode = scheduler.plan(now).unwrap().decode.unwrap();
        assert_eq!(
            scheduler.cancel(decode.items[0].sequence).unwrap(),
            CancellationState::InFlight(ScheduledPhase::Decode)
        );
        assert_eq!(scheduler.active(), 0);
        assert_eq!(scheduler.queued(), 0);
    }

    #[test]
    fn a_bad_result_cannot_advance_another_or_overrun_a_chunk() {
        let now = Instant::now();
        let mut scheduler = Scheduler::new(config()).unwrap();
        scheduler.enqueue(id(1), 8, 16, now).unwrap();
        scheduler.admit_reserved(id(1), now).unwrap();
        let item = scheduler.plan(now).unwrap().prefill.unwrap().items[0];
        assert!(scheduler.complete_prefill(id(1), item.tokens + 1).is_err());
        assert!(scheduler.complete_prefill(id(2), item.tokens).is_err());
        scheduler.complete_prefill(id(1), item.tokens).unwrap();
    }
}
