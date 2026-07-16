use serde::Deserialize;
use std::fmt;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    architectures: Vec<String>,
    attention_bias: bool,
    attention_dropout: f64,
    pub(crate) bos_token_id: u32,
    pub(crate) eos_token_id: u32,
    pub(crate) head_dim: usize,
    hidden_act: String,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    model_type: String,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    rope_scaling: Option<serde_json::Value>,
    pub(crate) rope_theta: f64,
    sliding_window: Option<usize>,
    tie_word_embeddings: bool,
    torch_dtype: String,
    use_cache: bool,
    use_sliding_window: bool,
    pub(crate) vocab_size: usize,
}

impl Config {
    pub(crate) fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(Error::Io)?;
        let config: Self = serde_json::from_slice(&bytes).map_err(Error::Json)?;
        config.validate()?;
        Ok(config)
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.model_type != "qwen3" || self.architectures.as_slice() != ["Qwen3ForCausalLM"] {
            return Err(Error::Unsupported(
                "expected the dense Qwen3ForCausalLM architecture",
            ));
        }
        if self.torch_dtype != "bfloat16" {
            return Err(Error::Unsupported(
                "the initial Qwen3 product contract requires bf16 checkpoint storage",
            ));
        }
        if self.hidden_act != "silu" {
            return Err(Error::Unsupported("Qwen3 requires the SiLU gated MLP"));
        }
        if self.attention_bias || self.attention_dropout != 0.0 {
            return Err(Error::Unsupported(
                "attention bias and dropout are not part of dense Qwen3 inference",
            ));
        }
        if !self.tie_word_embeddings {
            return Err(Error::Unsupported(
                "untied Qwen3 output embeddings are not supported by this model contract",
            ));
        }
        if !self.use_cache {
            return Err(Error::Unsupported(
                "Qwen3 generation requires persistent key/value state",
            ));
        }
        if self.use_sliding_window || self.sliding_window.is_some() {
            return Err(Error::Unsupported(
                "sliding-window Qwen3 variants require a distinct cache contract",
            ));
        }
        if self.rope_scaling.is_some() {
            return Err(Error::Unsupported(
                "scaled-RoPE Qwen3 variants require an explicit scaling contract",
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
        {
            return Err(Error::Invalid(
                "model dimensions, layer/head counts, vocabulary, and context must be positive",
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::Invalid(
                "query head count must be divisible by key/value head count",
            ));
        }
        if self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.head_dim)
                .is_none()
        {
            return Err(Error::Invalid("attention projection width overflows usize"));
        }
        if self.vocab_size > i32::MAX as usize
            || self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return Err(Error::Invalid(
                "vocabulary and special token IDs must fit the I32 graph index domain",
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(Error::Invalid(
                "RMS normalization epsilon must be finite and positive",
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::Invalid("RoPE theta must be finite and positive"));
        }
        Ok(())
    }

    pub(crate) fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub(crate) fn key_value_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[derive(Debug)]
pub(crate) enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(&'static str),
    Unsupported(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
            Self::Invalid(message) => write!(formatter, "invalid Qwen3 config: {message}"),
            Self::Unsupported(message) => {
                write!(formatter, "unsupported Qwen3 config: {message}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    fn official_config() -> Value {
        json!({
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "max_position_embeddings": 40960,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 0.000001,
            "rope_scaling": null,
            "rope_theta": 1000000.0,
            "sliding_window": null,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151936
        })
    }

    fn load(value: &Value) -> Result<Config, Error> {
        let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "nml-qwen3-config-{}-{sequence}.json",
            std::process::id()
        ));
        std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        let result = Config::from_file(&path);
        std::fs::remove_file(path).unwrap();
        result
    }

    #[test]
    fn official_dense_bf16_contract_is_accepted() {
        let config = load(&official_config()).unwrap();
        assert_eq!(config.query_width(), 2048);
        assert_eq!(config.key_value_width(), 1024);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.vocab_size, 151936);
    }

    #[test]
    fn unsupported_storage_and_cache_variants_fail_before_graph_construction() {
        for (field, replacement, expected) in [
            (
                "torch_dtype",
                json!("float16"),
                "requires bf16 checkpoint storage",
            ),
            (
                "tie_word_embeddings",
                json!(false),
                "untied Qwen3 output embeddings",
            ),
            (
                "use_sliding_window",
                json!(true),
                "sliding-window Qwen3 variants",
            ),
        ] {
            let mut value = official_config();
            value[field] = replacement;
            let error = load(&value).unwrap_err().to_string();
            assert!(error.contains(expected), "{field}: {error}");
        }
    }

    #[test]
    fn invalid_gqa_and_token_domains_are_rejected() {
        let mut gqa = official_config();
        gqa["num_key_value_heads"] = json!(6);
        assert!(
            load(&gqa)
                .unwrap_err()
                .to_string()
                .contains("query head count must be divisible")
        );

        let mut token = official_config();
        token["eos_token_id"] = json!(151936);
        assert!(
            load(&token)
                .unwrap_err()
                .to_string()
                .contains("special token IDs")
        );
    }
}
