//! Exact host-side ownership for the process-wide paged KV arena.
//!
//! This module deliberately contains no device calls. It is the authority for
//! physical-page IDs, reservation credits, visibility lengths, sharing, and
//! reclamation. Device metadata is derived from this state and validated again
//! immediately before upload.

// Checkpoint/sealing operations are complete here before the prefix-cache and
// speculative schedulers begin consuming every transition.
#![allow(dead_code)]

use super::contracts::SequenceId;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error as StdError;
use std::fmt;

pub(crate) const TARGET_PAGE_SIZE: usize = 16;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct PageId(u32);

impl PageId {
    pub(crate) const fn as_u32(self) -> u32 {
        self.0
    }

    fn as_usize(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PageState {
    Free,
    Private {
        owner: SequenceId,
        committed: u8,
        tentative: u8,
    },
    Sealed {
        references: u32,
    },
}

#[derive(Clone, Debug)]
struct Reservation {
    /// Credits not yet converted into private physical pages.
    remaining_pages: usize,
    /// Physical pages this request may allocate over its complete lifetime.
    total_pages: usize,
    pages: Vec<PageId>,
    committed_tokens: usize,
    tentative_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CacheCheckpoint {
    sequence: SequenceId,
    pages: usize,
    committed_tokens: usize,
    tentative_tokens: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CacheStats {
    pub(crate) total_pages: usize,
    pub(crate) free_pages: usize,
    pub(crate) private_pages: usize,
    pub(crate) sealed_pages: usize,
    pub(crate) sealed_references: usize,
    pub(crate) reserved_unallocated_pages: usize,
    pub(crate) sequences: usize,
    pub(crate) committed_tokens: usize,
    pub(crate) tentative_tokens: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CacheError {
    ArithmeticOverflow,
    DuplicateSequence(SequenceId),
    UnknownSequence(SequenceId),
    InsufficientCapacity {
        requested_pages: usize,
        available_pages: usize,
    },
    ReservationExhausted(SequenceId),
    InvalidTransition(&'static str),
    InvalidMetadata(&'static str),
}

impl fmt::Display for CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticOverflow => write!(formatter, "cache accounting overflow"),
            Self::DuplicateSequence(sequence) => {
                write!(formatter, "duplicate cache sequence {}", sequence.as_u64())
            }
            Self::UnknownSequence(sequence) => {
                write!(formatter, "unknown cache sequence {}", sequence.as_u64())
            }
            Self::InsufficientCapacity {
                requested_pages,
                available_pages,
            } => write!(
                formatter,
                "cache admission needs {requested_pages} pages but only {available_pages} are unclaimed"
            ),
            Self::ReservationExhausted(sequence) => write!(
                formatter,
                "cache reservation for sequence {} is exhausted",
                sequence.as_u64()
            ),
            Self::InvalidTransition(message) => write!(formatter, "invalid cache transition: {message}"),
            Self::InvalidMetadata(message) => write!(formatter, "invalid cache metadata: {message}"),
        }
    }
}

impl StdError for CacheError {}

/// Frozen target-cache geometry shared by compilation and allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TargetCacheGeometry {
    pub(crate) layers: usize,
    pub(crate) page_size: usize,
    pub(crate) local_kv_heads: usize,
    pub(crate) head_dimension: usize,
    pub(crate) element_bytes: usize,
}

impl TargetCacheGeometry {
    pub(crate) fn bytes_per_physical_page(self) -> Result<usize, CacheError> {
        self.layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.page_size))
            .and_then(|value| value.checked_mul(self.local_kv_heads))
            .and_then(|value| value.checked_mul(self.head_dimension))
            .and_then(|value| value.checked_mul(self.element_bytes))
            .ok_or(CacheError::ArithmeticOverflow)
    }
}

/// Immutable page count selected before any serving family is compiled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrozenCachePlan {
    geometry: TargetCacheGeometry,
    physical_pages: usize,
    arena_bytes: usize,
    safety_bytes: usize,
}

impl FrozenCachePlan {
    pub(crate) fn freeze(
        geometry: TargetCacheGeometry,
        cache_budget_bytes: usize,
        safety_bytes: usize,
    ) -> Result<Self, CacheError> {
        if geometry.layers == 0
            || geometry.page_size != TARGET_PAGE_SIZE
            || geometry.local_kv_heads == 0
            || geometry.head_dimension == 0
            || geometry.element_bytes == 0
        {
            return Err(CacheError::InvalidTransition(
                "target cache geometry is incomplete or uses a non-16-token page",
            ));
        }
        let usable = cache_budget_bytes.checked_sub(safety_bytes).ok_or(
            CacheError::InvalidTransition("cache safety reserve exceeds the cache budget"),
        )?;
        let page_bytes = geometry.bytes_per_physical_page()?;
        let physical_pages = usable / page_bytes;
        if physical_pages == 0 {
            return Err(CacheError::InsufficientCapacity {
                requested_pages: 1,
                available_pages: 0,
            });
        }
        let arena_bytes = physical_pages
            .checked_mul(page_bytes)
            .ok_or(CacheError::ArithmeticOverflow)?;
        Ok(Self {
            geometry,
            physical_pages,
            arena_bytes,
            safety_bytes,
        })
    }

    pub(crate) const fn geometry(self) -> TargetCacheGeometry {
        self.geometry
    }

    pub(crate) const fn physical_pages(self) -> usize {
        self.physical_pages
    }

    pub(crate) const fn arena_bytes(self) -> usize {
        self.arena_bytes
    }

    pub(crate) fn required_remaining_bytes(self) -> Result<usize, CacheError> {
        self.arena_bytes
            .checked_add(self.safety_bytes)
            .ok_or(CacheError::ArithmeticOverflow)
    }

    /// Rechecks residency-era device accounting without changing the plan.
    pub(crate) fn verify_remaining_bytes(self, remaining_bytes: usize) -> Result<(), CacheError> {
        let required = self.required_remaining_bytes()?;
        if remaining_bytes < required {
            return Err(CacheError::InsufficientCapacity {
                requested_pages: self.physical_pages,
                available_pages: remaining_bytes / self.geometry.bytes_per_physical_page()?,
            });
        }
        Ok(())
    }
}

/// Compact, upload-ready batch metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CompactCacheMetadata {
    pub(crate) block_tables: Vec<i32>,
    pub(crate) sequence_lengths: Vec<i32>,
    pub(crate) active_rows: Vec<bool>,
    pub(crate) table_width: usize,
}

impl CompactCacheMetadata {
    pub(crate) fn upload_bytes(&self) -> Result<usize, CacheError> {
        self.block_tables
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .and_then(|bytes| {
                self.sequence_lengths
                    .len()
                    .checked_mul(std::mem::size_of::<i32>())
                    .and_then(|length_bytes| bytes.checked_add(length_bytes))
            })
            .and_then(|bytes| bytes.checked_add(self.active_rows.len()))
            .ok_or(CacheError::ArithmeticOverflow)
    }
}

/// Host authority for one physical page namespace shared by every target layer.
pub(crate) struct PageManager {
    page_size: usize,
    descriptors: Vec<PageState>,
    free: VecDeque<PageId>,
    reservations: BTreeMap<SequenceId, Reservation>,
}

impl PageManager {
    pub(crate) fn new(physical_pages: usize) -> Result<Self, CacheError> {
        if physical_pages == 0 || physical_pages > i32::MAX as usize {
            return Err(CacheError::InvalidTransition(
                "physical page count must fit the positive I32 domain",
            ));
        }
        let mut free = VecDeque::with_capacity(physical_pages);
        for page in 0..physical_pages {
            free.push_back(PageId(
                u32::try_from(page).map_err(|_| CacheError::ArithmeticOverflow)?,
            ));
        }
        Ok(Self {
            page_size: TARGET_PAGE_SIZE,
            descriptors: vec![PageState::Free; physical_pages],
            free,
            reservations: BTreeMap::new(),
        })
    }

    pub(crate) fn reserve_tokens(
        &mut self,
        sequence: SequenceId,
        token_budget: usize,
    ) -> Result<usize, CacheError> {
        let pages = token_budget
            .checked_add(self.page_size - 1)
            .ok_or(CacheError::ArithmeticOverflow)?
            / self.page_size;
        self.reserve_pages(sequence, pages)?;
        Ok(pages)
    }

    /// Reserves only future physical allocations. Shared sealed prefix pages do
    /// not consume these credits because they already occupy the arena.
    pub(crate) fn reserve_pages(
        &mut self,
        sequence: SequenceId,
        pages: usize,
    ) -> Result<(), CacheError> {
        if self.reservations.contains_key(&sequence) {
            return Err(CacheError::DuplicateSequence(sequence));
        }
        let available = self.unclaimed_pages();
        if pages > available {
            return Err(CacheError::InsufficientCapacity {
                requested_pages: pages,
                available_pages: available,
            });
        }
        self.reservations.insert(
            sequence,
            Reservation {
                remaining_pages: pages,
                total_pages: pages,
                pages: Vec::new(),
                committed_tokens: 0,
                tentative_tokens: 0,
            },
        );
        Ok(())
    }

    pub(crate) fn append_tentative(
        &mut self,
        sequence: SequenceId,
        mut tokens: usize,
    ) -> Result<(), CacheError> {
        self.require_sequence(sequence)?;
        while tokens != 0 {
            let tail = self
                .reservations
                .get(&sequence)
                .and_then(|reservation| reservation.pages.last().copied());
            let writable = tail.and_then(|page| match self.descriptors[page.as_usize()] {
                PageState::Private {
                    owner,
                    tentative,
                    ..
                } if owner == sequence && usize::from(tentative) < self.page_size => {
                    Some((page, self.page_size - usize::from(tentative)))
                }
                _ => None,
            });
            let (page, capacity) = match writable {
                Some(writable) => writable,
                None => (self.allocate_private(sequence)?, self.page_size),
            };
            let appended = tokens.min(capacity);
            let PageState::Private { tentative, .. } =
                &mut self.descriptors[page.as_usize()]
            else {
                unreachable!("the selected cache tail was private")
            };
            *tentative = tentative
                .checked_add(u8::try_from(appended).map_err(|_| CacheError::ArithmeticOverflow)?)
                .ok_or(CacheError::ArithmeticOverflow)?;
            let reservation = self.reservations.get_mut(&sequence).unwrap();
            reservation.tentative_tokens = reservation
                .tentative_tokens
                .checked_add(appended)
                .ok_or(CacheError::ArithmeticOverflow)?;
            tokens -= appended;
        }
        Ok(())
    }

    pub(crate) fn append_committed(
        &mut self,
        sequence: SequenceId,
        tokens: usize,
    ) -> Result<(), CacheError> {
        self.append_tentative(sequence, tokens)?;
        self.commit(sequence, tokens)
    }

    pub(crate) fn commit(
        &mut self,
        sequence: SequenceId,
        tokens: usize,
    ) -> Result<(), CacheError> {
        let reservation = self
            .reservations
            .get(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?;
        let target = reservation
            .committed_tokens
            .checked_add(tokens)
            .ok_or(CacheError::ArithmeticOverflow)?;
        if target > reservation.tentative_tokens {
            return Err(CacheError::InvalidTransition(
                "cannot commit tokens that were not tentatively appended",
            ));
        }
        let start = reservation.committed_tokens;
        let pages = reservation.pages.clone();
        for token in start..target {
            let logical_page = token / self.page_size;
            let within_page = token % self.page_size + 1;
            let page = pages[logical_page];
            match &mut self.descriptors[page.as_usize()] {
                PageState::Private {
                    owner, committed, ..
                } if *owner == sequence => {
                    *committed = (*committed).max(
                        u8::try_from(within_page).map_err(|_| CacheError::ArithmeticOverflow)?,
                    );
                }
                PageState::Sealed { .. } if within_page == self.page_size => {}
                _ => {
                    return Err(CacheError::InvalidTransition(
                        "committed range is not owned by the sequence",
                    ));
                }
            }
        }
        self.reservations.get_mut(&sequence).unwrap().committed_tokens = target;
        Ok(())
    }

    pub(crate) fn checkpoint(
        &self,
        sequence: SequenceId,
    ) -> Result<CacheCheckpoint, CacheError> {
        let reservation = self
            .reservations
            .get(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?;
        Ok(CacheCheckpoint {
            sequence,
            pages: reservation.pages.len(),
            committed_tokens: reservation.committed_tokens,
            tentative_tokens: reservation.tentative_tokens,
        })
    }

    /// Rolls back tentative writes only. Committed visibility is immutable at
    /// this boundary; use `truncate` for an explicit committed rollback.
    pub(crate) fn rollback(&mut self, checkpoint: CacheCheckpoint) -> Result<(), CacheError> {
        let current = self
            .reservations
            .get(&checkpoint.sequence)
            .ok_or(CacheError::UnknownSequence(checkpoint.sequence))?;
        if current.committed_tokens != checkpoint.committed_tokens
            || checkpoint.tentative_tokens > current.tentative_tokens
            || checkpoint.pages > current.pages.len()
        {
            return Err(CacheError::InvalidTransition(
                "checkpoint is stale or would roll back committed tokens",
            ));
        }
        self.shrink_private_tail(
            checkpoint.sequence,
            checkpoint.tentative_tokens,
            checkpoint.committed_tokens,
        )?;
        Ok(())
    }

    pub(crate) fn truncate(
        &mut self,
        sequence: SequenceId,
        committed_tokens: usize,
    ) -> Result<(), CacheError> {
        let current = self
            .reservations
            .get(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?;
        if committed_tokens > current.committed_tokens {
            return Err(CacheError::InvalidTransition(
                "truncate cannot increase the committed length",
            ));
        }
        let kept_pages = div_ceil(committed_tokens, self.page_size)?;
        if committed_tokens % self.page_size != 0 && kept_pages != 0 {
            let page = current.pages[kept_pages - 1];
            if matches!(self.descriptors[page.as_usize()], PageState::Sealed { .. }) {
                return Err(CacheError::InvalidTransition(
                    "cannot truncate inside an immutable sealed page",
                ));
            }
        }
        self.shrink_private_tail(sequence, committed_tokens, committed_tokens)
    }

    /// Converts every complete, fully committed private page into an immutable
    /// page with one reference owned by this sequence.
    pub(crate) fn seal_complete_pages(
        &mut self,
        sequence: SequenceId,
    ) -> Result<usize, CacheError> {
        let pages = self
            .reservations
            .get(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?
            .pages
            .clone();
        let mut sealed = 0;
        for page in pages {
            if matches!(
                self.descriptors[page.as_usize()],
                PageState::Private {
                    owner,
                    committed,
                    tentative,
                } if owner == sequence
                    && usize::from(committed) == self.page_size
                    && usize::from(tentative) == self.page_size
            ) {
                self.descriptors[page.as_usize()] = PageState::Sealed { references: 1 };
                sealed += 1;
            }
        }
        Ok(sealed)
    }

    /// Attaches one complete immutable page as the next logical prefix block.
    pub(crate) fn share_sealed(
        &mut self,
        sequence: SequenceId,
        page: PageId,
    ) -> Result<(), CacheError> {
        self.require_sequence(sequence)?;
        let descriptor = self
            .descriptors
            .get_mut(page.as_usize())
            .ok_or(CacheError::InvalidTransition("shared page ID is out of range"))?;
        let PageState::Sealed { references } = descriptor else {
            return Err(CacheError::InvalidTransition(
                "only a sealed page may be shared",
            ));
        };
        *references = references
            .checked_add(1)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let reservation = self.reservations.get_mut(&sequence).unwrap();
        if reservation.tentative_tokens != reservation.committed_tokens
            || !reservation.committed_tokens.is_multiple_of(self.page_size)
        {
            *references -= 1;
            return Err(CacheError::InvalidTransition(
                "shared pages may only extend a committed page-aligned prefix",
            ));
        }
        reservation.pages.push(page);
        reservation.committed_tokens = reservation
            .committed_tokens
            .checked_add(self.page_size)
            .ok_or(CacheError::ArithmeticOverflow)?;
        reservation.tentative_tokens = reservation.committed_tokens;
        Ok(())
    }

    /// Releases pages in reverse logical order. Repeated terminal release is a
    /// no-op so the terminal ledger can remain idempotent.
    pub(crate) fn release_sequence(
        &mut self,
        sequence: SequenceId,
    ) -> Result<bool, CacheError> {
        let Some(reservation) = self.reservations.remove(&sequence) else {
            return Ok(false);
        };
        for page in reservation.pages.into_iter().rev() {
            match self.descriptors[page.as_usize()] {
                PageState::Private { owner, .. } if owner == sequence => self.free_page(page),
                PageState::Sealed { references: 1 } => self.free_page(page),
                PageState::Sealed { references } if references > 1 => {
                    self.descriptors[page.as_usize()] = PageState::Sealed {
                        references: references - 1,
                    };
                }
                _ => {
                    return Err(CacheError::InvalidTransition(
                        "sequence release encountered a page it does not own",
                    ));
                }
            }
        }
        Ok(true)
    }

    pub(crate) fn compact_metadata(
        &self,
        rows: &[Option<SequenceId>],
        table_width: usize,
    ) -> Result<CompactCacheMetadata, CacheError> {
        if table_width == 0 {
            return Err(CacheError::InvalidMetadata(
                "block-table width must be nonzero",
            ));
        }
        let entries = rows
            .len()
            .checked_mul(table_width)
            .ok_or(CacheError::ArithmeticOverflow)?;
        let mut metadata = CompactCacheMetadata {
            block_tables: vec![-1; entries],
            sequence_lengths: vec![0; rows.len()],
            active_rows: vec![false; rows.len()],
            table_width,
        };
        for (row, sequence) in rows.iter().enumerate() {
            let Some(sequence) = sequence else {
                continue;
            };
            let reservation = self
                .reservations
                .get(sequence)
                .ok_or(CacheError::UnknownSequence(*sequence))?;
            let used_pages = div_ceil(reservation.tentative_tokens, self.page_size)?;
            if used_pages > table_width || used_pages != reservation.pages.len() {
                return Err(CacheError::InvalidMetadata(
                    "sequence does not fit the selected block-table width",
                ));
            }
            metadata.active_rows[row] = true;
            metadata.sequence_lengths[row] = i32::try_from(reservation.tentative_tokens)
                .map_err(|_| CacheError::InvalidMetadata("sequence length exceeds I32"))?;
            for (column, page) in reservation.pages.iter().enumerate() {
                metadata.block_tables[row * table_width + column] = i32::try_from(page.as_u32())
                    .map_err(|_| CacheError::InvalidMetadata("physical page ID exceeds I32"))?;
            }
        }
        self.validate_metadata(&metadata)?;
        Ok(metadata)
    }

    /// Validates all dynamic indexing values before any metadata upload.
    pub(crate) fn validate_metadata(
        &self,
        metadata: &CompactCacheMetadata,
    ) -> Result<(), CacheError> {
        let rows = metadata.active_rows.len();
        if metadata.table_width == 0
            || metadata.sequence_lengths.len() != rows
            || metadata.block_tables.len()
                != rows
                    .checked_mul(metadata.table_width)
                    .ok_or(CacheError::ArithmeticOverflow)?
        {
            return Err(CacheError::InvalidMetadata(
                "metadata vectors have incompatible shapes",
            ));
        }
        for row in 0..rows {
            let length = metadata.sequence_lengths[row];
            if length < 0 || (!metadata.active_rows[row] && length != 0) {
                return Err(CacheError::InvalidMetadata(
                    "inactive rows must have zero nonnegative length",
                ));
            }
            let used = if metadata.active_rows[row] {
                div_ceil(length as usize, self.page_size)?
            } else {
                0
            };
            if used > metadata.table_width {
                return Err(CacheError::InvalidMetadata(
                    "sequence length exceeds block-table width",
                ));
            }
            for column in 0..metadata.table_width {
                let page = metadata.block_tables[row * metadata.table_width + column];
                if column >= used {
                    if page != -1 {
                        return Err(CacheError::InvalidMetadata(
                            "unused block-table slots must contain -1",
                        ));
                    }
                    continue;
                }
                let page = usize::try_from(page).map_err(|_| {
                    CacheError::InvalidMetadata("used block-table page ID is negative")
                })?;
                if page >= self.descriptors.len()
                    || matches!(self.descriptors[page], PageState::Free)
                {
                    return Err(CacheError::InvalidMetadata(
                        "used block-table page ID is out of range or free",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn stats(&self) -> CacheStats {
        let mut stats = CacheStats {
            total_pages: self.descriptors.len(),
            free_pages: self.free.len(),
            sequences: self.reservations.len(),
            ..CacheStats::default()
        };
        for state in &self.descriptors {
            match state {
                PageState::Free => {}
                PageState::Private { .. } => stats.private_pages += 1,
                PageState::Sealed { references } => {
                    stats.sealed_pages += 1;
                    stats.sealed_references += *references as usize;
                }
            }
        }
        for reservation in self.reservations.values() {
            stats.reserved_unallocated_pages += reservation.remaining_pages;
            stats.committed_tokens += reservation.committed_tokens;
            stats.tentative_tokens += reservation.tentative_tokens;
        }
        stats
    }

    pub(crate) fn sequence_lengths(
        &self,
        sequence: SequenceId,
    ) -> Result<(usize, usize), CacheError> {
        let reservation = self
            .reservations
            .get(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?;
        Ok((reservation.committed_tokens, reservation.tentative_tokens))
    }

    pub(crate) fn page_table(
        &self,
        sequence: SequenceId,
    ) -> Result<&[PageId], CacheError> {
        self.reservations
            .get(&sequence)
            .map(|reservation| reservation.pages.as_slice())
            .ok_or(CacheError::UnknownSequence(sequence))
    }

    fn unclaimed_pages(&self) -> usize {
        let reserved = self
            .reservations
            .values()
            .map(|reservation| reservation.remaining_pages)
            .sum::<usize>();
        self.free.len().saturating_sub(reserved)
    }

    fn require_sequence(&self, sequence: SequenceId) -> Result<(), CacheError> {
        self.reservations
            .contains_key(&sequence)
            .then_some(())
            .ok_or(CacheError::UnknownSequence(sequence))
    }

    fn allocate_private(&mut self, sequence: SequenceId) -> Result<PageId, CacheError> {
        let reservation = self
            .reservations
            .get_mut(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?;
        if reservation.remaining_pages == 0 {
            return Err(CacheError::ReservationExhausted(sequence));
        }
        let page = self.free.pop_front().ok_or(CacheError::InvalidTransition(
            "reserved cache page is unexpectedly unavailable",
        ))?;
        reservation.remaining_pages -= 1;
        reservation.pages.push(page);
        self.descriptors[page.as_usize()] = PageState::Private {
            owner: sequence,
            committed: 0,
            tentative: 0,
        };
        Ok(page)
    }

    fn shrink_private_tail(
        &mut self,
        sequence: SequenceId,
        tentative_tokens: usize,
        committed_tokens: usize,
    ) -> Result<(), CacheError> {
        let keep_pages = div_ceil(tentative_tokens, self.page_size)?;
        let current_pages = self
            .reservations
            .get(&sequence)
            .ok_or(CacheError::UnknownSequence(sequence))?
            .pages
            .len();
        if keep_pages > current_pages || committed_tokens > tentative_tokens {
            return Err(CacheError::InvalidTransition(
                "rollback lengths exceed the current cache state",
            ));
        }
        while self.reservations.get(&sequence).unwrap().pages.len() > keep_pages {
            let page = self.reservations.get_mut(&sequence).unwrap().pages.pop().unwrap();
            match self.descriptors[page.as_usize()] {
                PageState::Private { owner, .. } if owner == sequence => {
                    self.free_page(page);
                    self.reservations.get_mut(&sequence).unwrap().remaining_pages += 1;
                }
                _ => {
                    return Err(CacheError::InvalidTransition(
                        "rollback may release only private tail pages",
                    ));
                }
            }
        }
        if keep_pages != 0 {
            let page = self.reservations[&sequence].pages[keep_pages - 1];
            let tentative_in_page = nonzero_tail(tentative_tokens, self.page_size);
            let committed_in_page = if committed_tokens / self.page_size == keep_pages - 1 {
                nonzero_tail(committed_tokens, self.page_size)
            } else {
                self.page_size
            };
            match &mut self.descriptors[page.as_usize()] {
                PageState::Private {
                    owner,
                    committed,
                    tentative,
                } if *owner == sequence => {
                    *committed = u8::try_from(committed_in_page)
                        .map_err(|_| CacheError::ArithmeticOverflow)?;
                    *tentative = u8::try_from(tentative_in_page)
                        .map_err(|_| CacheError::ArithmeticOverflow)?;
                }
                PageState::Sealed { .. }
                    if committed_in_page == self.page_size
                        && tentative_in_page == self.page_size => {}
                _ => {
                    return Err(CacheError::InvalidTransition(
                        "rollback would mutate an immutable or foreign page",
                    ));
                }
            }
        }
        let reservation = self.reservations.get_mut(&sequence).unwrap();
        reservation.committed_tokens = committed_tokens;
        reservation.tentative_tokens = tentative_tokens;
        debug_assert!(reservation.remaining_pages <= reservation.total_pages);
        Ok(())
    }

    fn free_page(&mut self, page: PageId) {
        self.descriptors[page.as_usize()] = PageState::Free;
        self.free.push_back(page);
    }
}

fn div_ceil(value: usize, divisor: usize) -> Result<usize, CacheError> {
    if value == 0 {
        return Ok(0);
    }
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or(CacheError::ArithmeticOverflow)
}

fn nonzero_tail(tokens: usize, page_size: usize) -> usize {
    let remainder = tokens % page_size;
    if remainder == 0 && tokens != 0 {
        page_size
    } else {
        remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_plan_uses_exact_whole_page_accounting() {
        let geometry = TargetCacheGeometry {
            layers: 24,
            page_size: 16,
            local_kv_heads: 8,
            head_dimension: 64,
            element_bytes: 2,
        };
        assert_eq!(geometry.bytes_per_physical_page().unwrap(), 786_432);
        let budget = 786_432 * 101 + 1_024;
        let plan = FrozenCachePlan::freeze(geometry, budget, 1_024).unwrap();
        assert_eq!(plan.physical_pages(), 101);
        assert_eq!(plan.arena_bytes(), 786_432 * 101);
        assert_eq!(plan.geometry(), geometry);
        assert!(plan
            .verify_remaining_bytes(plan.required_remaining_bytes().unwrap())
            .is_ok());
        assert!(plan
            .verify_remaining_bytes(plan.required_remaining_bytes().unwrap() - 1)
            .is_err());
    }

    #[test]
    fn reservations_rollback_sharing_metadata_and_release_are_exact() {
        let mut manager = PageManager::new(8).unwrap();
        let first = SequenceId::new(1);
        let second = SequenceId::new(2);
        manager.reserve_tokens(first, 32).unwrap();
        manager.append_committed(first, 32).unwrap();
        assert_eq!(manager.seal_complete_pages(first).unwrap(), 2);
        let shared = manager.page_table(first).unwrap()[0];

        manager.reserve_pages(second, 1).unwrap();
        manager.share_sealed(second, shared).unwrap();
        manager.append_committed(second, 8).unwrap();
        let checkpoint = manager.checkpoint(second).unwrap();
        manager.append_tentative(second, 8).unwrap();
        assert_eq!(manager.sequence_lengths(second).unwrap(), (24, 32));
        manager.rollback(checkpoint).unwrap();
        assert_eq!(manager.sequence_lengths(second).unwrap(), (24, 24));
        manager.truncate(second, 16).unwrap();
        assert_eq!(manager.sequence_lengths(second).unwrap(), (16, 16));

        let metadata = manager
            .compact_metadata(&[Some(first), None, Some(second)], 3)
            .unwrap();
        assert_eq!(metadata.active_rows, [true, false, true]);
        assert_eq!(metadata.sequence_lengths, [32, 0, 16]);
        assert_eq!(&metadata.block_tables[3..6], &[-1, -1, -1]);
        assert_eq!(metadata.upload_bytes().unwrap(), 9 * 4 + 3 * 4 + 3);
        let mut invalid = metadata.clone();
        invalid.block_tables[5] = 0;
        assert!(manager.validate_metadata(&invalid).is_err());

        assert!(manager.release_sequence(first).unwrap());
        let retained = manager.stats();
        assert_eq!(retained.sealed_pages, 1);
        assert_eq!(retained.sealed_references, 1);
        assert!(manager.release_sequence(second).unwrap());
        assert!(!manager.release_sequence(second).unwrap());
        assert_eq!(
            manager.stats(),
            CacheStats {
                total_pages: 8,
                free_pages: 8,
                ..CacheStats::default()
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct ReferenceSequence {
        capacity_tokens: usize,
        committed: usize,
        tentative: usize,
    }

    #[test]
    fn randomized_private_page_state_matches_a_simple_reference_model() {
        let mut manager = PageManager::new(64).unwrap();
        let mut reference = BTreeMap::<SequenceId, ReferenceSequence>::new();
        let mut random = 0x9e37_79b9_u64;
        for _ in 0..2_000 {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let sequence = SequenceId::new((random >> 8) % 8 + 1);
            match (random >> 4) % 5 {
                0 if !reference.contains_key(&sequence) => {
                    let pages = ((random >> 16) % 4 + 1) as usize;
                    if manager.reserve_pages(sequence, pages).is_ok() {
                        reference.insert(
                            sequence,
                            ReferenceSequence {
                                capacity_tokens: pages * TARGET_PAGE_SIZE,
                                committed: 0,
                                tentative: 0,
                            },
                        );
                    }
                }
                1 => {
                    if let Some(state) = reference.get_mut(&sequence) {
                        let tokens = ((random >> 20) % 12 + 1) as usize;
                        if state.tentative + tokens <= state.capacity_tokens {
                            manager.append_tentative(sequence, tokens).unwrap();
                            state.tentative += tokens;
                        }
                    }
                }
                2 => {
                    if let Some(state) = reference.get_mut(&sequence) {
                        let available = state.tentative - state.committed;
                        let tokens = ((random >> 24) as usize) % (available + 1);
                        manager.commit(sequence, tokens).unwrap();
                        state.committed += tokens;
                    }
                }
                3 => {
                    if let Some(state) = reference.get_mut(&sequence) {
                        let checkpoint = manager.checkpoint(sequence).unwrap();
                        let room = state.capacity_tokens - state.tentative;
                        let tokens = (((random >> 28) as usize) % 9).min(room);
                        manager.append_tentative(sequence, tokens).unwrap();
                        manager.rollback(checkpoint).unwrap();
                    }
                }
                4 => {
                    if reference.remove(&sequence).is_some() {
                        manager.release_sequence(sequence).unwrap();
                    }
                }
                _ => {}
            }

            let stats = manager.stats();
            let expected_allocated = reference
                .values()
                .map(|state| div_ceil(state.tentative, TARGET_PAGE_SIZE).unwrap())
                .sum::<usize>();
            let expected_credits = reference
                .values()
                .map(|state| {
                    state.capacity_tokens / TARGET_PAGE_SIZE
                        - div_ceil(state.tentative, TARGET_PAGE_SIZE).unwrap()
                })
                .sum::<usize>();
            assert_eq!(stats.private_pages, expected_allocated);
            assert_eq!(stats.free_pages, 64 - expected_allocated);
            assert_eq!(stats.reserved_unallocated_pages, expected_credits);
            assert_eq!(
                stats.committed_tokens,
                reference
                    .values()
                    .map(|state| state.committed)
                    .sum::<usize>()
            );
            assert_eq!(
                stats.tentative_tokens,
                reference
                    .values()
                    .map(|state| state.tentative)
                    .sum::<usize>()
            );
        }
        for sequence in reference.keys().copied().collect::<Vec<_>>() {
            manager.release_sequence(sequence).unwrap();
        }
        assert_eq!(manager.stats().free_pages, 64);
        assert_eq!(manager.stats().sequences, 0);
    }
}
