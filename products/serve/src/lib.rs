//! Qwen serving product built on NML's acceleration substrate.

#![forbid(unsafe_code)]

mod engine;

/// Qwen3 checkpoint, graph adapter, and one-shot compatibility API.
///
/// The serving scheduler will consume this engine without moving model-specific
/// types into NML's backend-independent public facade.
pub mod qwen3;
