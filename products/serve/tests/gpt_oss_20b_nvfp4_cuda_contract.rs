use nml_serve::{
    CompilationProfile, Event, GenerationOptions, SamplingOptions, SubmissionTimings, Timings,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const MODEL_ENVIRONMENT: &str = "NML_GPT_OSS_MODEL";
const GENERATION_FIXTURE_ENVIRONMENT: &str = "NML_GPT_OSS_GENERATION_FIXTURE";
const ARTIFACT_MANIFEST: &str = "nml-artifact-manifest.json";
const ARTIFACT_MANIFEST_SHA256: &str =
    "ab4c8cbd4424c8fec95bf683c0efd04c9cd350ec2a26737408b5500e61003207";
const MODEL_IDENTITY: &str = "GPT-OSS 20B NVFP4";
const PROMPT: &str = "What is the capital of France?";
const MAX_NEW_TOKENS: usize = 32;
const MAX_PROMPT_TOKENS: usize = 128;
const MAX_SEQUENCE_TOKENS: usize = 256;
const SAMPLING: SamplingOptions = SamplingOptions {
    seed: [0x4e4d_4c2d_4750_544f, 0x5353_2d32_3042_0001],
    temperature: 1.0,
    top_k: 50,
    top_p: 1.0,
    min_p: 0.0,
};
const RETURN_TOKEN: u32 = 200_002;
const CALL_TOKEN: u32 = 200_012;
const PHYSICAL_PARAMETER_COMPONENTS: usize = 703;
const PARAMETER_SOURCE_BYTES: usize = 11_777_751_752;
const PARAMETER_PEAK_STAGING_BYTES: usize = 2 * 16 * 1024 * 1024;
const LAYERS: usize = 24;
const CACHE_PAGE_SIZE: usize = 16;
const KEY_VALUE_HEADS: usize = 8;
const HEAD_DIMENSION: usize = 64;
const BF16_BYTES: usize = 2;
const I32_BYTES: usize = 4;
const CONTRACT_MODE: &str = env!("NML_GPT_OSS_CONTRACT_MODE");

#[test]
fn full_checkpoint_executes_the_gpt_oss_nvfp4_cuda_product_contract() {
    let model_directory = required_absolute_path(MODEL_ENVIRONMENT);
    let fixture = match CONTRACT_MODE {
        "generation" => None,
        "acceptance" => {
            let fixture_path = required_absolute_path(GENERATION_FIXTURE_ENVIRONMENT);
            assert!(
                fixture_path.is_file(),
                "{GENERATION_FIXTURE_ENVIRONMENT} must name the independently checked generation fixture",
            );
            Some(load_checked_fixture(&fixture_path))
        }
        mode => panic!("unsupported GPT-OSS CUDA contract mode {mode:?}"),
    };
    assert_eq!(
        sha256(&model_directory.join(ARTIFACT_MANIFEST)),
        ARTIFACT_MANIFEST_SHA256,
        "the mounted model must be the immutable GPT-OSS 20B NVFP4 artifact",
    );

    // SAFETY: the contract process is single-threaded at initialization and no
    // XLA or PJRT API has run before the CUDA plugin is selected.
    let platform = unsafe { nml::Platform::cuda() }.expect("CUDA platform must initialize");
    let generator = nml_serve::Generator::load(
        &platform,
        &model_directory,
        &[CompilationProfile {
            max_prompt_tokens: MAX_PROMPT_TOKENS,
            max_sequence_tokens: MAX_SEQUENCE_TOKENS,
        }],
    )
    .expect("the immutable GPT-OSS checkpoint must compile before it loads");
    let mut events = Vec::new();
    let report = generator
        .generate(
            GenerationOptions {
                prompt: PROMPT.to_owned(),
                max_new_tokens: MAX_NEW_TOKENS,
                cache_capacity: None,
                sampling: SAMPLING,
            },
            |event| -> Result<(), std::convert::Infallible> {
                publish_completion(&event);
                events.push(ObservedEvent::from(event));
                Ok(())
            },
        )
        .expect("the full GPT-OSS checkpoint must generate on CUDA");

    assert_eq!(report.model, MODEL_IDENTITY);
    assert!(report.prompt_tokens > 0, "the fixed prompt must be tokenized");
    assert!(
        report.generated_tokens.len() >= 3,
        "acceptance generation must execute first and steady decode steps",
    );
    assert!(report.generated_tokens.len() <= MAX_NEW_TOKENS);
    assert!(
        report.cache_capacity >= report.prompt_tokens + MAX_NEW_TOKENS,
        "the finite cache family must cover the fixed prompt and generation bound",
    );
    assert_eq!(report.cache_capacity % CACHE_PAGE_SIZE, 0);
    assert!(report.cache_capacity.div_ceil(CACHE_PAGE_SIZE).is_power_of_two());
    assert_parameter_memory(&report);
    assert_cache_memory(&report);
    assert_runtime_events(&events, &report.generated_tokens);
    assert_runtime_timings(&report.timings, report.generated_tokens.len());
    if let Some(fixture) = &fixture {
        assert_checked_fixture(
            fixture,
            report.prompt_tokens,
            &report.generated_tokens,
            &events,
        );
    }
    println!();
    println!(
        "GPT-OSS completion: {} prompt tokens, {} generated tokens, {} resident bytes",
        report.prompt_tokens,
        report.generated_tokens.len(),
        report.parameter_resident_bytes,
    );
}

fn publish_completion(event: &Event) {
    match event {
        Event::ContentDelta { text, .. } => {
            print!("{text}");
            std::io::stdout()
                .flush()
                .expect("completion output must remain writable");
        }
        Event::ToolCall {
            recipient,
            arguments,
        } => {
            print!("{{\"recipient\":{recipient:?},\"arguments\":{arguments}}}");
            std::io::stdout()
                .flush()
                .expect("completion output must remain writable");
        }
        Event::Done { .. } => {}
    }
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ObservedEvent {
    ContentDelta { channel: String, text: String },
    ToolCall { recipient: String, arguments: Value },
    Done { reason: String },
}

impl From<Event> for ObservedEvent {
    fn from(event: Event) -> Self {
        match event {
            Event::ContentDelta { channel, text } => Self::ContentDelta {
                channel: channel.to_owned(),
                text,
            },
            Event::ToolCall {
                recipient,
                arguments,
            } => Self::ToolCall {
                recipient,
                arguments,
            },
            Event::Done { reason } => Self::Done {
                reason: reason.to_owned(),
            },
        }
    }
}

fn assert_runtime_events(events: &[ObservedEvent], generated_tokens: &[u32]) {
    assert!(!events.is_empty(), "generation must emit product events");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ObservedEvent::Done { .. }))
            .count(),
        1,
        "generation must emit exactly one terminal event",
    );
    let ObservedEvent::Done { reason } = events.last().expect("events are nonempty") else {
        panic!("the terminal product event must be last");
    };
    assert!(matches!(reason.as_str(), "return" | "tool_call" | "length"));
    match reason.as_str() {
        "return" => {
            assert_eq!(generated_tokens.last(), Some(&RETURN_TOKEN));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, ObservedEvent::ToolCall { .. }))
                    .count(),
                0,
                "a return termination must not publish a tool call",
            );
        }
        "tool_call" => {
            assert_eq!(generated_tokens.last(), Some(&CALL_TOKEN));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, ObservedEvent::ToolCall { .. }))
                    .count(),
                1,
                "tool-call termination must publish exactly one completed call",
            );
        }
        "length" => {
            assert_eq!(generated_tokens.len(), MAX_NEW_TOKENS);
            assert!(!matches!(
                generated_tokens.last(),
                Some(&RETURN_TOKEN) | Some(&CALL_TOKEN)
            ));
        }
        _ => unreachable!("the reason domain is checked above"),
    }
    assert!(
        events[..events.len() - 1].iter().any(|event| matches!(
            event,
            ObservedEvent::ContentDelta { text, .. } if !text.is_empty()
        ) || matches!(event, ObservedEvent::ToolCall { .. })),
        "the model must emit content or a tool call before completion",
    );
    for event in events {
        if let ObservedEvent::ContentDelta { channel, .. } = event {
            assert!(matches!(
                channel.as_str(),
                "analysis" | "commentary" | "final"
            ));
        }
    }
}

fn assert_parameter_memory(report: &nml_serve::GenerationReport) {
    assert_eq!(
        report.physical_parameter_components, PHYSICAL_PARAMETER_COMPONENTS,
        "the complete 411-parameter model must load all 703 compact components",
    );
    assert_eq!(report.parameter_source_bytes, PARAMETER_SOURCE_BYTES);
    assert_eq!(
        report.parameter_resident_bytes, report.parameter_source_bytes,
        "source-layout CUDA execution must preserve the compact source extent",
    );
    assert_eq!(
        report.parameter_prepared_bytes, 0,
        "source-layout execution must not retain a prepared or dense expansion",
    );
    assert_eq!(
        report.parameter_peak_staging_bytes, PARAMETER_PEAK_STAGING_BYTES,
        "CUDA loading must retain the declared two-lane 16 MiB staging bound",
    );
    assert!(
        report.parameter_peak_staging_bytes < report.parameter_source_bytes,
        "staging must remain bounded below the complete compact checkpoint",
    );
}

fn assert_cache_memory(report: &nml_serve::GenerationReport) {
    let pages = report.cache_capacity.div_ceil(CACHE_PAGE_SIZE);
    let expected_storage = LAYERS
        * 2
        * pages
        * CACHE_PAGE_SIZE
        * KEY_VALUE_HEADS
        * HEAD_DIMENSION
        * BF16_BYTES;
    let expected_metadata = (pages + 1) * I32_BYTES;
    assert_eq!(
        report.cache_storage_bytes, expected_storage,
        "cache storage must be 24 K/V pairs of fixed 16-token BF16 pages",
    );
    assert_eq!(
        report.cache_metadata_bytes, expected_metadata,
        "one shared I32 page table and sequence length must describe the request",
    );
}

fn assert_runtime_timings(timings: &Timings, generated_tokens: usize) {
    for (name, duration) in [
        ("tokenization", timings.tokenization),
        ("artifact validation", timings.artifact_validation),
        ("prefill compilation", timings.prefill_compilation),
        ("decode compilation", timings.decode_compilation),
        ("parameter upload", timings.parameter_upload),
        ("cache allocation", timings.cache_allocation),
        ("prompt upload", timings.prompt_upload),
        ("prefill execution", timings.prefill_execution),
        ("prefill download", timings.prefill_download),
    ] {
        assert_nonzero_timing(name, duration);
    }
    assert_submission("prefill", timings.prefill_submission);
    if generated_tokens > 1 {
        assert_nonzero_timing(
            "decode state initialization",
            timings.decode_state_initialization,
        );
        assert_nonzero_timing("first decode execution", timings.first_decode_execution);
        assert_nonzero_timing("decode download", timings.decode_download);
        assert_submission("first decode", timings.first_decode_submission);
    } else {
        assert_eq!(timings.decode_state_initialization, Duration::ZERO);
        assert_eq!(timings.first_decode_execution, Duration::ZERO);
        assert_eq!(timings.decode_download, Duration::ZERO);
    }
    if generated_tokens > 2 {
        assert_nonzero_timing("steady decode execution", timings.steady_decode_execution);
        assert_submission("steady decode", timings.steady_decode_submission);
    } else {
        assert_eq!(timings.steady_decode_execution, Duration::ZERO);
    }
}

fn assert_submission(name: &str, timings: SubmissionTimings) {
    for (component, duration) in [
        ("embedding", timings.embedding),
        ("sliding layers", timings.sliding_layers),
        ("full layers", timings.full_layers),
        ("head", timings.head),
    ] {
        assert_nonzero_timing(&format!("{name} {component} submission"), duration);
    }
}

fn assert_nonzero_timing(name: &str, duration: Duration) {
    assert!(!duration.is_zero(), "{name} timing must be observed");
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedGenerationFixture {
    schema_version: u32,
    artifact_manifest_sha256: String,
    model: String,
    prompt: String,
    max_new_tokens: usize,
    sampling: CheckedSamplingFixture,
    prompt_tokens: usize,
    generated_tokens: Vec<u32>,
    events: Vec<ObservedEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckedSamplingFixture {
    seed: [u64; 2],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
}

fn load_checked_fixture(path: &Path) -> CheckedGenerationFixture {
    let fixture: CheckedGenerationFixture = serde_json::from_reader(BufReader::new(
        File::open(path).expect("checked generation fixture must open"),
    ))
    .expect("checked generation fixture must be valid JSON");
    assert_eq!(fixture.schema_version, 2);
    assert_eq!(fixture.artifact_manifest_sha256, ARTIFACT_MANIFEST_SHA256);
    assert_eq!(fixture.model, MODEL_IDENTITY);
    assert_eq!(fixture.prompt, PROMPT);
    assert_eq!(fixture.max_new_tokens, MAX_NEW_TOKENS);
    assert_eq!(fixture.sampling.seed, SAMPLING.seed);
    assert_eq!(fixture.sampling.temperature, SAMPLING.temperature);
    assert_eq!(fixture.sampling.top_k, SAMPLING.top_k);
    assert_eq!(fixture.sampling.top_p, SAMPLING.top_p);
    assert_eq!(fixture.sampling.min_p, SAMPLING.min_p);
    fixture
}

fn assert_checked_fixture(
    fixture: &CheckedGenerationFixture,
    prompt_tokens: usize,
    generated_tokens: &[u32],
    events: &[ObservedEvent],
) {
    assert_eq!(prompt_tokens, fixture.prompt_tokens);
    assert_eq!(generated_tokens, fixture.generated_tokens);
    assert_eq!(events, fixture.events);
}

fn required_absolute_path(name: &str) -> PathBuf {
    let path = std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} is required for final product acceptance"));
    assert!(path.is_absolute(), "{name} must be an absolute path");
    path
}

fn sha256(path: &Path) -> String {
    let mut reader = BufReader::new(File::open(path).expect("artifact manifest must open"));
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .expect("artifact manifest must remain readable");
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
