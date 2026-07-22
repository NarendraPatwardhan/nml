//! Safe ownership and streaming semantics for IREE Hugging Face tokenizers.

#![forbid(unsafe_op_in_unsafe_fn)]

use nml_tokenizer_sys as ffi;
use std::error::Error as StdError;
use std::ffi::CStr;
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};

const RESOURCE_EXHAUSTED: i32 = 8;
const MIN_TOKEN_CAPACITY: usize = 32;
const DECODE_CHUNK_BYTES: usize = 2048;

/// One immutable tokenizer parsed from a Hugging Face `tokenizer.json` file.
///
/// Clones retain the same allocation. Calls through clones and their decoder
/// sessions are serialized because upstream IREE does not publish a shared
/// tokenizer concurrency contract.
#[derive(Clone)]
pub struct Tokenizer {
    inner: Arc<TokenizerInner>,
}

struct TokenizerInner {
    // IREE does not publish a concurrency contract for a tokenizer shared by
    // independent encode/decode states. Keep the opaque allocation behind one
    // narrow gate instead of inferring thread safety from const-qualified C
    // parameters. Per-session state remains independently owned, but every
    // bridge operation that can observe the shared tokenizer is serialized.
    allocation: Mutex<TokenizerAllocation>,
}

struct TokenizerAllocation {
    raw: NonNull<ffi::nml_iree_tokenizer_t>,
}

// SAFETY: this value is only an owning handle. Moving it does not dereference
// the allocation, all dereferences are made while its containing `Mutex` is
// locked, and the bridge created the allocation with IREE's system allocator.
// No `Sync` claim is made for either the handle or the IREE allocation.
unsafe impl Send for TokenizerAllocation {}

impl TokenizerInner {
    fn lock(&self) -> MutexGuard<'_, TokenizerAllocation> {
        // A Rust panic cannot mutate the shared tokenizer: bridge calls do not
        // unwind and all mutable encoder/decoder state lives outside it. It is
        // therefore safe to recover the serialization gate after poisoning.
        self.allocation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(Error::Io)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(json: &[u8]) -> Result<Self, Error> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `json` remains readable for the call and `raw` is a valid
        // out-pointer. IREE copies all tokenizer data into its own allocation.
        let code = unsafe {
            ffi::nml_iree_tokenizer_from_huggingface_json(json.as_ptr(), json.len(), &mut raw)
        };
        check(code)?;
        let raw = NonNull::new(raw).ok_or(Error::NullResult("tokenizer"))?;
        Ok(Self {
            inner: Arc::new(TokenizerInner {
                allocation: Mutex::new(TokenizerAllocation { raw }),
            }),
        })
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        let allocation = self.inner.lock();
        let mut token_id = 0;
        // SAFETY: the tokenizer is alive, the UTF-8 string is readable for its
        // declared length, and the output points to initialized writable data.
        let found = unsafe {
            ffi::nml_iree_tokenizer_lookup(
                allocation.raw.as_ptr(),
                token.as_ptr(),
                token.len(),
                &mut token_id,
            )
        };
        found.then_some(token_id)
    }

    /// Encodes one complete input without imposing a prompt-length bound.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.encode_with_special_matching(text, true)
    }

    /// Encodes text while treating every special-token spelling as ordinary input.
    ///
    /// Protocol renderers use this for caller-controlled content and append their
    /// validated structural token IDs separately. This is IREE's equivalent of
    /// tiktoken's `encode_ordinary` and prevents delimiter injection.
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.encode_with_special_matching(text, false)
    }

    fn encode_with_special_matching(
        &self,
        text: &str,
        match_special_tokens: bool,
    ) -> Result<Vec<u32>, Error> {
        // The guard intentionally covers creation, all feed/finalize calls,
        // and `Encoder::drop`; the encoder may retain tokenizer references.
        let allocation = self.inner.lock();
        let mut raw = std::ptr::null_mut();
        // SAFETY: the tokenizer outlives the encoder and `raw` is writable.
        let code = unsafe {
            ffi::nml_iree_encoder_create(
                allocation.raw.as_ptr(),
                text.len(),
                match_special_tokens,
                &mut raw,
            )
        };
        check(code)?;
        let raw = NonNull::new(raw).ok_or(Error::NullResult("encoder"))?;
        let mut encoder = Encoder { raw };
        encoder.encode(text.as_bytes())
    }

    pub fn decoder(&self) -> Result<Decoder, Error> {
        let allocation = self.inner.lock();
        let mut raw = std::ptr::null_mut();
        // SAFETY: the returned decoder retains a shared tokenizer owner and
        // `raw` is a valid out-pointer.
        let code = unsafe { ffi::nml_iree_decoder_create(allocation.raw.as_ptr(), &mut raw) };
        check(code)?;
        let raw = NonNull::new(raw).ok_or(Error::NullResult("decoder"))?;
        Ok(Decoder {
            raw,
            finished: false,
            _tokenizer: Arc::clone(&self.inner),
        })
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, Error> {
        let mut decoder = self.decoder()?;
        let mut bytes = decoder.feed(token_ids)?;
        bytes.extend(decoder.finish()?);
        String::from_utf8(bytes).map_err(Error::Utf8)
    }
}

impl Drop for TokenizerAllocation {
    fn drop(&mut self) {
        // SAFETY: this pointer is uniquely owned and freed exactly once.
        unsafe { ffi::nml_iree_tokenizer_free(self.raw.as_ptr()) };
    }
}

struct Encoder {
    raw: NonNull<ffi::nml_iree_encoder_t>,
}

impl Encoder {
    fn encode(&mut self, text: &[u8]) -> Result<Vec<u32>, Error> {
        let mut tokens = Vec::new();
        reserve_more(&mut tokens, (text.len() / 4).max(MIN_TOKEN_CAPACITY))?;
        let mut remaining = text;
        while !remaining.is_empty() {
            let spare = tokens.spare_capacity_mut();
            if spare.is_empty() {
                let additional = MIN_TOKEN_CAPACITY.max(tokens.capacity());
                reserve_more(&mut tokens, additional)?;
                continue;
            }
            let capacity = spare.len();
            let mut consumed = 0usize;
            let mut produced = 0usize;
            // SAFETY: the input and spare output regions are valid for the
            // supplied lengths. The bridge guarantees produced <= capacity.
            let code = unsafe {
                ffi::nml_iree_encoder_feed(
                    self.raw.as_ptr(),
                    remaining.as_ptr(),
                    remaining.len(),
                    spare.as_mut_ptr().cast::<u32>(),
                    capacity,
                    &mut consumed,
                    &mut produced,
                )
            };
            if produced > capacity || consumed > remaining.len() {
                return Err(Error::InvalidProgress);
            }
            if code != 0 && code != RESOURCE_EXHAUSTED {
                return Err(iree_error(code));
            }
            // SAFETY: IREE initialized exactly `produced` token slots.
            let new_len = tokens
                .len()
                .checked_add(produced)
                .ok_or(Error::CapacityOverflow)?;
            // SAFETY: IREE initialized exactly `produced` token slots and the
            // checked length cannot exceed the spare region validated above.
            unsafe { tokens.set_len(new_len) };
            remaining = &remaining[consumed..];
            if consumed == 0 && produced == 0 {
                let additional = MIN_TOKEN_CAPACITY.max(tokens.capacity());
                reserve_more(&mut tokens, additional)?;
            }
        }

        // IREE finalization consumes state and cannot be retried. Its exact
        // pending-token bound is therefore reserved before the single call.
        // SAFETY: `raw` points to a live initialized encoder.
        let pending = unsafe { ffi::nml_iree_encoder_pending_token_bound(self.raw.as_ptr()) };
        tokens.try_reserve(pending).map_err(Error::Allocation)?;
        let spare = tokens.spare_capacity_mut();
        let mut produced = 0usize;
        // SAFETY: the pending bound guarantees enough writable token slots.
        let code = unsafe {
            ffi::nml_iree_encoder_finalize(
                self.raw.as_ptr(),
                spare.as_mut_ptr().cast::<u32>(),
                spare.len(),
                &mut produced,
            )
        };
        check(code)?;
        if produced > spare.len() {
            return Err(Error::InvalidProgress);
        }
        // SAFETY: IREE initialized exactly `produced` token slots.
        let new_len = tokens
            .len()
            .checked_add(produced)
            .ok_or(Error::CapacityOverflow)?;
        // SAFETY: IREE initialized exactly `produced` token slots and the
        // checked length cannot exceed the spare region validated above.
        unsafe { tokens.set_len(new_len) };
        Ok(tokens)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: this pointer is uniquely owned and freed exactly once.
        unsafe { ffi::nml_iree_encoder_free(self.raw.as_ptr()) };
    }
}

/// Owned stateful detokenization for token-at-a-time generation.
///
/// A decoder retains the tokenizer allocation that created it, so it can
/// safely outlive every public [`Tokenizer`] handle. Decoder operations join
/// the same narrow serialization gate as encoding and vocabulary lookup.
pub struct Decoder {
    raw: NonNull<ffi::nml_iree_decoder_t>,
    finished: bool,
    // Retains the immutable tokenizer allocation referenced by IREE's decoder
    // state. Field drop follows `Decoder::drop`, so the raw decoder is freed
    // before this final shared owner can release the tokenizer.
    _tokenizer: Arc<TokenizerInner>,
}

// SAFETY: the bridge decoder is a uniquely owned heap allocation containing
// only system-allocated storage plus a pointer to the retained immutable
// tokenizer. It has no creator-thread affinity, bridge diagnostics are
// `_Thread_local`, every operation requires `&mut self`, and each bridge call
// is additionally serialized by the retained tokenizer's mutex. Moving the
// unique owner between threads cannot introduce concurrent decoder access.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Feeds one token and returns all bytes now safe to stream. A fragment is
    /// not required to be valid UTF-8 until the complete stream is assembled.
    pub fn push(&mut self, token_id: u32) -> Result<Vec<u8>, Error> {
        self.feed(std::slice::from_ref(&token_id))
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        let _allocation = self._tokenizer.lock();
        // SAFETY: `raw` points to a live decoder and reset preserves storage.
        check(unsafe { ffi::nml_iree_decoder_reset(self.raw.as_ptr()) })?;
        self.finished = false;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, Error> {
        if self.finished {
            return Err(Error::DecoderFinished);
        }
        let _allocation = self._tokenizer.lock();
        let mut output = Vec::new();
        let mut chunk_bytes = 32usize;
        loop {
            let start = grow_zeroed(&mut output, chunk_bytes)?;
            let mut produced = 0usize;
            // SAFETY: the appended region is writable for `chunk_bytes` and
            // the decoder is alive for the call. Pinned IREE explicitly makes
            // RESOURCE_EXHAUSTED finalization resumable.
            let code = unsafe {
                ffi::nml_iree_decoder_finalize(
                    self.raw.as_ptr(),
                    output[start..].as_mut_ptr(),
                    chunk_bytes,
                    &mut produced,
                )
            };
            if produced > chunk_bytes {
                return Err(Error::InvalidProgress);
            }
            output.truncate(start.checked_add(produced).ok_or(Error::CapacityOverflow)?);
            match code {
                0 => {
                    self.finished = true;
                    return Ok(output);
                }
                RESOURCE_EXHAUSTED => {
                    chunk_bytes = chunk_bytes.checked_mul(2).ok_or(Error::CapacityOverflow)?;
                }
                code => return Err(iree_error(code)),
            }
        }
    }

    fn feed(&mut self, token_ids: &[u32]) -> Result<Vec<u8>, Error> {
        if self.finished {
            return Err(Error::DecoderFinished);
        }
        let _allocation = self._tokenizer.lock();
        let initial_capacity = token_ids
            .len()
            .max(1)
            .checked_mul(8)
            .ok_or(Error::CapacityOverflow)?;
        let mut output = Vec::new();
        reserve_more(&mut output, initial_capacity)?;
        let mut remaining = token_ids;
        let mut chunk_bytes = DECODE_CHUNK_BYTES;
        while !remaining.is_empty() {
            let start = grow_zeroed(&mut output, chunk_bytes)?;
            let mut consumed = 0usize;
            let mut produced = 0usize;
            // SAFETY: both input and output regions are valid for the supplied
            // lengths and the decoder is alive for the call.
            let code = unsafe {
                ffi::nml_iree_decoder_feed(
                    self.raw.as_ptr(),
                    remaining.as_ptr(),
                    remaining.len(),
                    output[start..].as_mut_ptr(),
                    chunk_bytes,
                    &mut consumed,
                    &mut produced,
                )
            };
            check(code)?;
            if produced > chunk_bytes || consumed > remaining.len() {
                return Err(Error::InvalidProgress);
            }
            output.truncate(start + produced);
            remaining = &remaining[consumed..];
            if consumed == 0 && produced == 0 {
                // IREE reports zero progress when the next token's complete
                // decoded text cannot fit. State is unchanged in that case,
                // so grow the buffer and retry instead of imposing an
                // undocumented maximum token-text length.
                chunk_bytes = chunk_bytes.checked_mul(2).ok_or(Error::CapacityOverflow)?;
            } else {
                chunk_bytes = DECODE_CHUNK_BYTES;
            }
        }
        Ok(output)
    }
}

fn reserve_more<T>(output: &mut Vec<T>, additional: usize) -> Result<(), Error> {
    output.try_reserve(additional).map_err(Error::Allocation)
}

fn grow_zeroed(output: &mut Vec<u8>, additional: usize) -> Result<usize, Error> {
    let start = output.len();
    let end = start
        .checked_add(additional)
        .ok_or(Error::CapacityOverflow)?;
    output.try_reserve(additional).map_err(Error::Allocation)?;
    output.resize(end, 0);
    Ok(start)
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _allocation = self._tokenizer.lock();
        // SAFETY: this pointer is uniquely owned and freed exactly once before
        // the retained tokenizer owner can be destroyed.
        unsafe { ffi::nml_iree_decoder_free(self.raw.as_ptr()) };
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Iree { code: i32, message: String },
    NullResult(&'static str),
    DecoderFinished,
    InvalidProgress,
    CapacityOverflow,
    Allocation(std::collections::TryReserveError),
    Utf8(std::string::FromUtf8Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Iree { code, message } => {
                write!(formatter, "IREE tokenizer status {code}: {message}")
            }
            Self::NullResult(owner) => write!(formatter, "IREE returned a null {owner}"),
            Self::DecoderFinished => {
                formatter.write_str("IREE tokenizer decoder is already finalized")
            }
            Self::InvalidProgress => {
                formatter.write_str("IREE tokenizer reported invalid or zero progress")
            }
            Self::CapacityOverflow => {
                formatter.write_str("tokenizer output capacity exceeds addressable memory")
            }
            Self::Allocation(error) => {
                write!(formatter, "unable to allocate tokenizer output: {error}")
            }
            Self::Utf8(error) => write!(formatter, "decoded text is not UTF-8: {error}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Allocation(error) => Some(error),
            Self::Utf8(error) => Some(error),
            _ => None,
        }
    }
}

fn check(code: i32) -> Result<(), Error> {
    if code == 0 {
        Ok(())
    } else {
        Err(iree_error(code))
    }
}

fn iree_error(code: i32) -> Error {
    // SAFETY: the bridge returns a permanent thread-local NUL-terminated
    // buffer whose contents remain valid until the next bridge call.
    let message = unsafe {
        let pointer = ffi::nml_iree_tokenizer_last_error();
        if pointer.is_null() {
            String::new()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    let message = if message.is_empty() {
        "no diagnostic was provided".to_owned()
    } else {
        message
    };
    Error::Iree { code, message }
}
