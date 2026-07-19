//! Persistent GPT-OSS serving product over NML's acceleration substrate.

#![forbid(unsafe_code)]

mod gpt_oss;

use serde_json::Value;
use std::error::Error as StdError;
use std::path::Path;
use std::time::Duration;

pub type Error = Box<dyn StdError + Send + Sync>;

/// One request submitted to an already loaded GPT-OSS model.
pub struct GenerationOptions {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub cache_capacity: Option<usize>,
    pub sampling: SamplingOptions,
}

/// Runtime sampling controls for one generation request.
///
/// The candidate capacity remains a bounded compiled product contract; these
/// values select a deterministic distribution within it without compiling a
/// new executable. `top_k = 1` is the explicit greedy mode rather than an
/// implicit serving default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    pub seed: [u64; 2],
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            seed: [0x4e4d_4c2d_4750_544f, 0x5353_2d32_3042_0001],
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}

/// One bounded execution profile compiled before the model becomes resident.
///
/// Capacities are upper bounds. The product normalizes them to its finite
/// prefill and paged-cache buckets, deduplicates equivalent profiles, and
/// rejects requests that fit none of the compiled profiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompilationProfile {
    pub max_prompt_tokens: usize,
    pub max_sequence_tokens: usize,
}

/// Incremental Harmony events emitted by the product.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    ContentDelta { channel: &'static str, text: String },
    ToolCall { recipient: String, arguments: Value },
    Done { reason: &'static str },
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub tokenization: Duration,
    pub artifact_validation: Duration,
    pub prefill_compilation: Duration,
    pub decode_compilation: Duration,
    pub parameter_upload: Duration,
    pub cache_allocation: Duration,
    pub prompt_upload: Duration,
    pub prefill_execution: Duration,
    pub prefill_download: Duration,
    pub decode_state_initialization: Duration,
    pub first_decode_execution: Duration,
    pub steady_decode_execution: Duration,
    pub decode_download: Duration,
    /// Host time spent binding and submitting component executables during
    /// prefill. No device synchronization is introduced by these counters.
    pub prefill_submission: SubmissionTimings,
    /// Host submission time for the first decode step.
    pub first_decode_submission: SubmissionTimings,
    /// Aggregate host submission time for all later decode steps.
    pub steady_decode_submission: SubmissionTimings,
}

/// Host-side cost of crossing the reusable executable boundaries.
///
/// Layer values are accumulated across model depth. They are intentionally not
/// presented as GPU kernel timings because PJRT execution remains asynchronous
/// until the generated-token buffer is awaited.
#[derive(Clone, Copy, Debug, Default)]
pub struct SubmissionTimings {
    pub embedding: Duration,
    pub sliding_layers: Duration,
    pub full_layers: Duration,
    pub layer_pairs: Duration,
    pub head: Duration,
}

pub struct GenerationReport {
    pub model: &'static str,
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub cache_capacity: usize,
    pub physical_parameter_components: usize,
    pub parameter_source_bytes: usize,
    pub parameter_resident_bytes: usize,
    pub parameter_prepared_bytes: usize,
    pub parameter_peak_staging_bytes: usize,
    pub cache_storage_bytes: usize,
    pub cache_metadata_bytes: usize,
    pub timings: Timings,
}

/// One process-persistent model owner.
///
/// Artifact validation and tokenizer construction precede compilation of every
/// configured component family. Parameter upload begins only after compilation
/// succeeds. The resulting plan and parameters are retained across requests;
/// request tokens, parser state, K/V storage, and sampling state never escape a
/// generation call.
pub struct Generator<'platform> {
    inner: gpt_oss::Generator<'platform>,
}

impl<'platform> Generator<'platform> {
    pub fn load(
        platform: &'platform nml::Platform,
        model_directory: impl AsRef<Path>,
        profiles: &[CompilationProfile],
    ) -> Result<Self, Error> {
        Ok(Self {
            inner: gpt_oss::Generator::load(platform, model_directory.as_ref(), profiles)?,
        })
    }

    pub fn generate<E>(
        &self,
        options: GenerationOptions,
        mut emit: impl FnMut(Event) -> Result<(), E>,
    ) -> Result<GenerationReport, Error>
    where
        E: StdError + Send + Sync + 'static,
    {
        let report = self.inner.generate(
            options.prompt,
            options.max_new_tokens,
            options.cache_capacity,
            options.sampling,
            |event| {
                if let Some(event) = map_harmony_event(event) {
                    emit(event).map_err(|error| Box::new(error) as Error)?;
                }
                Ok(())
            },
        )?;
        Ok(GenerationReport {
            model: gpt_oss::MODEL_NAME,
            prompt_tokens: report.prompt_tokens,
            generated_tokens: report.generated_tokens,
            cache_capacity: report.cache_capacity,
            physical_parameter_components: report.physical_parameter_components,
            parameter_source_bytes: report.parameter_source_bytes,
            parameter_resident_bytes: report.parameter_resident_bytes,
            parameter_prepared_bytes: report.parameter_prepared_bytes,
            parameter_peak_staging_bytes: report.parameter_peak_staging_bytes,
            cache_storage_bytes: report.cache_storage_bytes,
            cache_metadata_bytes: report.cache_metadata_bytes,
            timings: timings(report.timings),
        })
    }
}

fn map_harmony_event(event: gpt_oss::protocol::Event) -> Option<Event> {
    use gpt_oss::protocol::{Channel, Event as HarmonyEvent, StopReason};
    match event {
        HarmonyEvent::ContentDelta { channel, text } => Some(Event::ContentDelta {
            channel: match channel {
                Channel::Analysis => "analysis",
                Channel::Commentary => "commentary",
                Channel::Final => "final",
            },
            text,
        }),
        HarmonyEvent::ToolCall(call) => Some(Event::ToolCall {
            recipient: call.recipient,
            arguments: call.arguments,
        }),
        HarmonyEvent::Done(reason) => Some(Event::Done {
            reason: match reason {
                StopReason::Return => "return",
                StopReason::ToolCall => "tool_call",
                StopReason::Length => "length",
            },
        }),
        HarmonyEvent::Message(_) => None,
    }
}

fn timings(value: gpt_oss::ProductTimings) -> Timings {
    Timings {
        tokenization: value.tokenization,
        artifact_validation: value.artifact_validation,
        prefill_compilation: value.prefill_compilation,
        decode_compilation: value.decode_compilation,
        parameter_upload: value.parameter_upload,
        cache_allocation: value.cache_allocation,
        prompt_upload: value.prompt_upload,
        prefill_execution: value.prefill_execution,
        prefill_download: value.prefill_download,
        decode_state_initialization: value.decode_state_initialization,
        first_decode_execution: value.first_decode_execution,
        steady_decode_execution: value.steady_decode_execution,
        decode_download: value.decode_download,
        prefill_submission: value.prefill_submission,
        first_decode_submission: value.first_decode_submission,
        steady_decode_submission: value.steady_decode_submission,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_harmony_generation_has_a_public_length_reason() {
        assert_eq!(
            map_harmony_event(gpt_oss::protocol::Event::Done(
                gpt_oss::protocol::StopReason::Length,
            )),
            Some(Event::Done { reason: "length" }),
        );
    }
}
