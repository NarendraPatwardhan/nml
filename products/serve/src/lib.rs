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
/// Artifact validation, tokenizer construction, parameter upload, and compiled
/// component families are retained across requests. Request tokens, parser
/// state, K/V storage, and sampling state never escape a generation call.
pub struct Generator<'platform> {
    inner: gpt_oss::Generator<'platform>,
}

impl<'platform> Generator<'platform> {
    pub fn load(
        platform: &'platform nml::Platform,
        model_directory: impl AsRef<Path>,
    ) -> Result<Self, Error> {
        Ok(Self {
            inner: gpt_oss::Generator::load(platform, model_directory.as_ref())?,
        })
    }

    pub fn generate<E>(
        &mut self,
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
