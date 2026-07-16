use nml_qwen3::GenerationOptions;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::path::PathBuf;

const MODEL_BYTES: u64 = 1_503_300_328;
const MODEL_SHA256: &str = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b";
const TOKENIZER_SHA256: &str = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";

#[test]
fn official_qwen3_0_6b_matches_the_pinned_bf16_oracle() {
    let model = std::env::var_os("NML_QWEN3_MODEL")
        .map(PathBuf::from)
        .expect("NML_QWEN3_MODEL must name the pinned Qwen3-0.6B directory");
    let weights = model.join("model.safetensors");
    assert_eq!(std::fs::metadata(&weights).unwrap().len(), MODEL_BYTES);
    assert_eq!(sha256(&weights), MODEL_SHA256);
    assert_eq!(sha256(&model.join("tokenizer.json")), TOKENIZER_SHA256);

    let platform = nml::Platform::cpu().unwrap();
    let mut text = Vec::new();
    let report = nml_qwen3::generate(
        &platform,
        &GenerationOptions {
            model_directory: model,
            prompt: "What is the capital of France?".to_owned(),
            max_new_tokens: 4,
            cache_capacity: None,
        },
        &mut text,
    )
    .unwrap();

    assert_eq!(report.prompt_tokens, 19);
    assert_eq!(report.generated_tokens, [785, 6722, 315, 9625]);
    assert_eq!(String::from_utf8(text).unwrap(), "The capital of France");
    assert!(!report.timings.prefill_compilation.is_zero());
    assert!(!report.timings.decode_compilation.is_zero());
    assert!(!report.timings.prefill_execution.is_zero());
    assert!(!report.timings.first_decode_execution.is_zero());
}

fn sha256(path: &Path) -> String {
    let mut reader = BufReader::new(File::open(path).unwrap());
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).unwrap();
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
