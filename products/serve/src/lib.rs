//! Qwen serving product built on NML's acceleration substrate.

#![forbid(unsafe_code)]

/// Qwen3 checkpoint, graph, and single-request generation engine.
///
/// The serving scheduler will consume this engine without moving model-specific
/// types into NML's backend-independent public facade.
pub mod qwen3;
