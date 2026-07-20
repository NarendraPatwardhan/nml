//! GPT-OSS 20B product ownership.
//!
//! Artifact identity, Harmony, checkpoint structure, component graphs, and the
//! transformer schedule live here. Framework crates know only tensors,
//! parameters, compilation, buffers, and reusable executable slots.

#![forbid(unsafe_code)]

mod artifact;
mod checkpoint;
mod config;
mod execution;
mod graph;
pub(crate) mod protocol;

use checkpoint::{BoxError, Result};
use config::Config;
use crate::{CompilationProfile, SamplingOptions, SubmissionTimings};
use execution::{ModelDefinition, PreparedRequest, ResidentModel, RunMetrics, StartupMetrics};
use nml::io::ParameterSet;
use nml::safetensors::TensorRegistry;
use protocol::{Conversation, HarmonyProtocol, Message, SystemContent};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::{Duration, Instant};

pub(crate) const MODEL_NAME: &str = "GPT-OSS 20B NVFP4";
const ARTIFACT_MANIFEST: &str = "nml-artifact-manifest.json";
const CHECKPOINT_INDEX: &str = "model.safetensors.index.json";
const DIRECT_CHECKPOINT: &str = "model.safetensors";
const ARTIFACT_MANIFEST_SHA256: &str =
    "a3e8d0f77a85a9b1625c105fdb0853c285be0776efcfe44a1a2e3abd7ea286e9";
const ARTIFACT_FILE_COUNT: usize = 20;
// The manifest authenticates these twenty payload/metadata files. Its own
// 4,118 bytes are authenticated by ARTIFACT_MANIFEST_SHA256 and therefore are
// not counted a second time as a declared artifact file.
const ARTIFACT_TOTAL_BYTES: u64 = 11_805_934_314;
const ARTIFACT_RECIPE: &str = "nml-nvfp4-weight-v3";
const ARTIFACT_RECIPE_SHA256: &str =
    "7679e52e5139be4007e2efbe4c6028ec6ebcb5b370e48b6905e48b87f79677ee";
const SOURCE_MANIFEST_SHA256: &str =
    "4f9fd730e12e0535cf6788a11a9b1604749f4520738a5c7ea643e27bf4b5ccb1";
const TENSOR_MANIFEST_SHA256: &str =
    "fd7c6833d00eca158bc1145dc2577ad8d38d8f4ed977ef3e3dfc0c2a72ea5cae";
const SOURCE_REPOSITORY: &str = "unsloth/gpt-oss-20b-BF16";
const SOURCE_REVISION: &str = "cc89b3e7fd423253264883a80a4fa5abc619649f";
const CONVERTER_NAME: &str = "nml-nvfp4-converter";
const CONVERTER_VERSION: u32 = 3;
const CONVERTER_SCRIPT_SHA256: &str =
    "5b0626f16fbd7dfe5a0e914b46a7fdab84907a7d01b5ec720a6abf8b3e643fd5";
const CONVERTER_REQUIREMENTS_SHA256: &str =
    "f384757dfae59e89aa0dfad0ea75a651005a336437903981119636ed58de8c8e";

/// Persistent product model. Its complete execution plan is compiled before
/// its parameters become resident and is reused by every request.
pub(crate) struct Generator<'platform> {
    protocol: HarmonyProtocol,
    model: ResidentModel<'platform>,
}

pub(crate) struct ProductReport {
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) cache_capacity: usize,
    pub(crate) physical_parameter_components: usize,
    pub(crate) parameter_source_bytes: usize,
    pub(crate) parameter_resident_bytes: usize,
    pub(crate) parameter_prepared_bytes: usize,
    pub(crate) parameter_peak_staging_bytes: usize,
    pub(crate) cache_storage_bytes: usize,
    pub(crate) cache_metadata_bytes: usize,
    pub(crate) timings: ProductTimings,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ProductTimings {
    pub(crate) tokenization: Duration,
    pub(crate) artifact_validation: Duration,
    pub(crate) prefill_compilation: Duration,
    pub(crate) decode_compilation: Duration,
    pub(crate) parameter_upload: Duration,
    pub(crate) cache_allocation: Duration,
    pub(crate) prompt_upload: Duration,
    pub(crate) prefill_execution: Duration,
    pub(crate) prefill_download: Duration,
    pub(crate) decode_state_initialization: Duration,
    pub(crate) first_decode_execution: Duration,
    pub(crate) steady_decode_execution: Duration,
    pub(crate) decode_download: Duration,
    pub(crate) prefill_submission: SubmissionTimings,
    pub(crate) first_decode_submission: SubmissionTimings,
    pub(crate) steady_decode_submission: SubmissionTimings,
}

impl<'platform> Generator<'platform> {
    pub(crate) fn load(
        platform: &'platform nml::Platform,
        model_directory: &Path,
        profiles: &[CompilationProfile],
    ) -> Result<Self> {
        let validation_started = Instant::now();
        validate_artifact(model_directory).map_err(boxed)?;
        let artifact_validation = validation_started.elapsed();
        let config = Config::from_file(model_directory.join("config.json")).map_err(boxed)?;
        let protocol = HarmonyProtocol::load(model_directory).map_err(boxed)?;
        let registry = TensorRegistry::from_path(model_directory.join(CHECKPOINT_INDEX))
            .map_err(boxed)?;
        let definition = ModelDefinition::declare(
            config,
            ParameterSet::new(registry),
            artifact_validation,
        )?;
        let model = definition.compile(platform, profiles)?.make_resident()?;
        Ok(Self { protocol, model })
    }

    pub(crate) fn generate(
        &self,
        prompt: String,
        max_new_tokens: usize,
        cache_capacity: Option<usize>,
        sampling: SamplingOptions,
        mut emit: impl FnMut(protocol::Event) -> Result<()>,
    ) -> Result<ProductReport> {
        let tokenization_started = Instant::now();
        let conversation = Conversation::new([
            Message::System(SystemContent::default()),
            Message::user(prompt),
        ]);
        let tokens = self
            .protocol
            .render_for_completion(&conversation)
            .map_err(boxed)?;
        let tokenization = tokenization_started.elapsed();
        let request = PreparedRequest::new(
            tokens,
            max_new_tokens,
            cache_capacity,
            sampling,
            self.model.config().context_limit(),
            tokenization,
        )?;
        let prompt_tokens = request.tokens.len();
        let mut parser = self.protocol.parser();
        let run = self.model.generate(&request, |token, _is_stop| {
            for event in parser.process(token).map_err(boxed)? {
                emit(event)?;
            }
            Ok(())
        })?;
        if run.stopped {
            parser.finish().map_err(boxed)?;
        } else {
            for event in parser.truncate().map_err(boxed)? {
                emit(event)?;
            }
        }
        let startup = self.model.startup();
        Ok(ProductReport {
            prompt_tokens,
            generated_tokens: run.generated_tokens,
            cache_capacity: run.cache_capacity,
            physical_parameter_components: startup.physical_parameter_components,
            parameter_source_bytes: startup.parameter_source_bytes,
            parameter_resident_bytes: startup.parameter_resident_bytes,
            parameter_prepared_bytes: startup.parameter_prepared_bytes,
            parameter_peak_staging_bytes: startup.parameter_peak_staging_bytes,
            cache_storage_bytes: run.cache_storage_bytes,
            cache_metadata_bytes: run.cache_metadata_bytes,
            timings: timings(request.tokenization, startup, run.metrics),
        })
    }
}

fn timings(
    tokenization: Duration,
    startup: StartupMetrics,
    run: RunMetrics,
) -> ProductTimings {
    ProductTimings {
        tokenization,
        artifact_validation: startup.artifact_validation,
        prefill_compilation: startup.prefill_compilation,
        decode_compilation: startup.decode_compilation,
        parameter_upload: startup.parameter_upload,
        cache_allocation: run.cache_allocation,
        prompt_upload: run.prompt_upload,
        prefill_execution: run.prefill_execution,
        prefill_download: run.prefill_download,
        decode_state_initialization: run.decode_state_initialization,
        first_decode_execution: run.first_decode_execution,
        steady_decode_execution: run.steady_decode_execution,
        decode_download: run.decode_download,
        prefill_submission: run.prefill_submission,
        first_decode_submission: run.first_decode_submission,
        steady_decode_submission: run.steady_decode_submission,
    }
}

fn validate_artifact(model_directory: &Path) -> std::result::Result<(), ArtifactError> {
    let manifest_path = model_directory.join(ARTIFACT_MANIFEST);
    let manifest_metadata = std::fs::symlink_metadata(&manifest_path)?;
    if !manifest_metadata.file_type().is_file() {
        return Err(ArtifactError(format!(
            "artifact manifest is not a regular file: {}",
            manifest_path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if manifest_metadata.mode() & 0o222 != 0 {
            return Err(ArtifactError(
                "artifact manifest is writable; rematerialize the artifact before launch"
                    .to_owned(),
            ));
        }
    }
    let manifest_hash = sha256(&manifest_path)?;
    if manifest_hash != ARTIFACT_MANIFEST_SHA256 {
        return Err(ArtifactError(format!(
            "artifact manifest SHA-256 is {manifest_hash}, expected {ARTIFACT_MANIFEST_SHA256}"
        )));
    }
    let manifest: ArtifactManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?)).map_err(|error| {
            ArtifactError(format!("artifact manifest is invalid JSON: {error}"))
        })?;
    validate_manifest_identity(&manifest)?;
    validate_checkpoint_entries(
        &manifest.files,
        model_directory.join(DIRECT_CHECKPOINT).is_file(),
    )?;
    let total = manifest.files.iter().try_fold(0u64, |total, file| {
        if !is_sha256(&file.sha256) {
            return Err(ArtifactError(format!(
                "artifact file {:?} has an invalid SHA-256",
                file.path
            )));
        }
        total
            .checked_add(file.size)
            .ok_or_else(|| ArtifactError("artifact byte count overflows u64".to_owned()))
    })?;
    if total != ARTIFACT_TOTAL_BYTES {
        return Err(ArtifactError(format!(
            "artifact files total {total} bytes, expected {ARTIFACT_TOTAL_BYTES}"
        )));
    }
    artifact::validate_materialization(
        model_directory,
        ARTIFACT_MANIFEST_SHA256,
        manifest.files.iter().map(|file| artifact::ExpectedFile {
            path: &file.path,
            size: file.size,
        }),
    )?;
    Ok(())
}

fn validate_manifest_identity(
    manifest: &ArtifactManifest,
) -> std::result::Result<(), ArtifactError> {
    let converter = &manifest.converter;
    if converter.name != CONVERTER_NAME
        || converter.version != CONVERTER_VERSION
        || converter.device != "cpu"
        || converter.python != "3.12.11"
        || converter.torch != "2.8.0+cpu"
        || converter.numpy != "2.2.6"
        || converter.safetensors != "0.6.2"
        || converter.huggingface_hub != "0.34.4"
        || converter.script_sha256 != CONVERTER_SCRIPT_SHA256
        || converter.requirements_sha256 != CONVERTER_REQUIREMENTS_SHA256
        || manifest.schema_version != 1
        || manifest.recipe != ARTIFACT_RECIPE
        || manifest.recipe_sha256 != ARTIFACT_RECIPE_SHA256
        || manifest.source_manifest_sha256 != SOURCE_MANIFEST_SHA256
        || manifest.source_repository != SOURCE_REPOSITORY
        || manifest.source_revision != SOURCE_REVISION
        || manifest.tensor_manifest_sha256 != TENSOR_MANIFEST_SHA256
        || manifest.files.len() != ARTIFACT_FILE_COUNT
    {
        return Err(ArtifactError(
            "artifact manifest identity does not match the selected GPT-OSS revision".to_owned(),
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_checkpoint_entries(
    files: &[ArtifactFile],
    direct_checkpoint_exists: bool,
) -> std::result::Result<(), ArtifactError> {
    if !files.iter().any(|file| file.path == CHECKPOINT_INDEX) {
        return Err(ArtifactError(
            "artifact manifest does not list model.safetensors.index.json".to_owned(),
        ));
    }
    if direct_checkpoint_exists && !files.iter().any(|file| file.path == DIRECT_CHECKPOINT) {
        return Err(ArtifactError(
            "artifact contains an unlisted model.safetensors".to_owned(),
        ));
    }
    Ok(())
}

fn sha256(path: &Path) -> std::result::Result<String, ArtifactError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactManifest {
    converter: ConverterManifest,
    schema_version: u32,
    recipe: String,
    recipe_sha256: String,
    source_manifest_sha256: String,
    source_repository: String,
    source_revision: String,
    tensor_manifest_sha256: String,
    files: Vec<ArtifactFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConverterManifest {
    device: String,
    huggingface_hub: String,
    name: String,
    numpy: String,
    python: String,
    requirements_sha256: String,
    safetensors: String,
    script_sha256: String,
    torch: String,
    version: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactFile {
    path: String,
    sha256: String,
    size: u64,
}

#[derive(Debug)]
struct ArtifactError(String);

impl From<std::io::Error> for ArtifactError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<artifact::Error> for ArtifactError {
    fn from(error: artifact::Error) -> Self {
        Self(error.to_string())
    }
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ArtifactError {}

fn boxed<E>(error: E) -> BoxError
where
    E: std::error::Error + Send + Sync + 'static,
{
    Box::new(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_in_manifest_matches_the_product_contract() {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        let path = Path::new(&runfiles)
            .join("_main/artifacts/gpt-oss-20b-nvfp4/artifact-manifest.json");
        let manifest: ArtifactManifest =
            serde_json::from_reader(BufReader::new(File::open(path).unwrap())).unwrap();
        validate_manifest_identity(&manifest).unwrap();
    }

    fn entry(path: &str) -> ArtifactFile {
        ArtifactFile {
            path: path.to_owned(),
            sha256: "0".repeat(64),
            size: 0,
        }
    }

    #[test]
    fn checkpoint_selection_requires_the_manifest_index() {
        let error = validate_checkpoint_entries(&[], false).unwrap_err();
        assert_eq!(
            error.to_string(),
            "artifact manifest does not list model.safetensors.index.json",
        );
    }

    #[test]
    fn checkpoint_selection_rejects_an_unlisted_direct_file() {
        let error = validate_checkpoint_entries(&[entry(CHECKPOINT_INDEX)], true).unwrap_err();
        assert_eq!(
            error.to_string(),
            "artifact contains an unlisted model.safetensors",
        );
    }

    #[test]
    fn checkpoint_selection_accepts_the_listed_index_without_a_direct_file() {
        validate_checkpoint_entries(&[entry(CHECKPOINT_INDEX)], false).unwrap();
    }
}
