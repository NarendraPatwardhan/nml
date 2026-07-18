//! Closed configuration contract for the selected GPT-OSS 20B artifact.

use serde::Deserialize;
use std::fmt;
use std::path::Path;

const LAYERS: usize = 24;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeScaling {
    beta_fast: f64,
    beta_slow: f64,
    factor: f64,
    original_max_position_embeddings: usize,
    rope_type: String,
    truncate: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    architectures: Vec<String>,
    attention_bias: bool,
    attention_dropout: f64,
    eos_token_id: u32,
    experts_per_token: usize,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    initial_context_length: usize,
    initializer_range: f64,
    intermediate_size: usize,
    layer_types: Vec<AttentionKind>,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    num_local_experts: usize,
    output_router_logits: bool,
    pad_token_id: u32,
    rms_norm_eps: f64,
    rope_scaling: RopeScaling,
    rope_theta: f64,
    router_aux_loss_coef: f64,
    sliding_window: usize,
    swiglu_limit: f64,
    tie_word_embeddings: bool,
    torch_dtype: String,
    transformers_version: String,
    use_cache: bool,
    vocab_size: usize,
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(Error::Io)?;
        Self::from_slice(&bytes)
    }

    fn from_slice(bytes: &[u8]) -> Result<Self, Error> {
        let config: Self = serde_json::from_slice(bytes).map_err(Error::Json)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.architectures.len() != 1 {
            return Err(Error::Contract(format!(
                "architectures is {:?}; selected GPT-OSS 20B requires exactly GptOssForCausalLM",
                self.architectures
            )));
        }
        exact(
            "architectures[0]",
            self.architectures[0].as_str(),
            "GptOssForCausalLM",
        )?;
        exact("model_type", self.model_type.as_str(), "gpt_oss")?;
        exact("torch_dtype", self.torch_dtype.as_str(), "bfloat16")?;
        exact(
            "transformers_version",
            self.transformers_version.as_str(),
            "4.56.0.dev0",
        )?;
        exact("attention_bias", self.attention_bias, true)?;
        exact("attention_dropout", self.attention_dropout, 0.0)?;
        exact("hidden_act", self.hidden_act.as_str(), "silu")?;
        exact("use_cache", self.use_cache, true)?;
        exact("tie_word_embeddings", self.tie_word_embeddings, false)?;
        exact("output_router_logits", self.output_router_logits, false)?;
        exact("hidden_size", self.hidden_size, 2_880)?;
        exact("intermediate_size", self.intermediate_size, 2_880)?;
        exact("num_hidden_layers", self.num_hidden_layers, LAYERS)?;
        exact("num_attention_heads", self.num_attention_heads, 64)?;
        exact("num_key_value_heads", self.num_key_value_heads, 8)?;
        exact("head_dim", self.head_dim, 64)?;
        exact("num_local_experts", self.num_local_experts, 32)?;
        exact("num_experts_per_tok", self.num_experts_per_tok, 4)?;
        exact(
            "experts_per_token",
            self.experts_per_token,
            self.num_experts_per_tok,
        )?;
        exact("vocab_size", self.vocab_size, 201_088)?;
        exact("pad_token_id", self.pad_token_id, 199_999)?;
        exact("eos_token_id", self.eos_token_id, 200_002)?;
        exact("initial_context_length", self.initial_context_length, 4_096)?;
        exact(
            "max_position_embeddings",
            self.max_position_embeddings,
            131_072,
        )?;
        exact("sliding_window", self.sliding_window, 128)?;
        exact("rms_norm_eps", self.rms_norm_eps, 1e-5)?;
        exact("rope_theta", self.rope_theta, 150_000.0)?;
        exact("swiglu_limit", self.swiglu_limit, 7.0)?;
        exact("initializer_range", self.initializer_range, 0.02)?;
        exact("router_aux_loss_coef", self.router_aux_loss_coef, 0.9)?;
        exact(
            "rope_scaling.rope_type",
            self.rope_scaling.rope_type.as_str(),
            "yarn",
        )?;
        exact("rope_scaling.factor", self.rope_scaling.factor, 32.0)?;
        exact("rope_scaling.beta_fast", self.rope_scaling.beta_fast, 32.0)?;
        exact("rope_scaling.beta_slow", self.rope_scaling.beta_slow, 1.0)?;
        exact(
            "rope_scaling.original_max_position_embeddings",
            self.rope_scaling.original_max_position_embeddings,
            self.initial_context_length,
        )?;
        exact("rope_scaling.truncate", self.rope_scaling.truncate, false)?;

        if self.layer_types.len() != self.num_hidden_layers {
            return Err(Error::Contract(format!(
                "layer_types has {} entries; expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for (index, actual) in self.layer_types.iter().copied().enumerate() {
            let expected = if index % 2 == 0 {
                AttentionKind::SlidingAttention
            } else {
                AttentionKind::FullAttention
            };
            if actual != expected {
                return Err(Error::Contract(format!(
                    "layer_types[{index}] is {actual:?}; expected {expected:?}"
                )));
            }
        }
        if self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self
                .num_local_experts
                .checked_mul(self.intermediate_size)
                .is_none()
            || self.intermediate_size.checked_mul(2).is_none()
        {
            return Err(Error::Contract(
                "model projection geometry overflows usize".to_owned(),
            ));
        }
        if self.vocab_size > i32::MAX as usize
            || self.pad_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
        {
            return Err(Error::Contract(
                "vocabulary and special tokens exceed the graph index domain".to_owned(),
            ));
        }
        Ok(())
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub const fn layers(&self) -> usize {
        self.num_hidden_layers
    }

    pub const fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub const fn key_value_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub const fn experts(&self) -> usize {
        self.num_local_experts
    }

    pub const fn experts_per_token(&self) -> usize {
        self.num_experts_per_tok
    }

    pub const fn vocabulary(&self) -> usize {
        self.vocab_size
    }

    pub const fn context_limit(&self) -> usize {
        self.max_position_embeddings
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn query_heads(&self) -> usize {
        self.num_attention_heads
    }

    pub const fn key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    pub const fn rms_norm_epsilon(&self) -> f64 {
        self.rms_norm_eps
    }

    pub const fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    pub const fn rope_factor(&self) -> f64 {
        self.rope_scaling.factor
    }

    pub const fn rope_beta_fast(&self) -> f64 {
        self.rope_scaling.beta_fast
    }

    pub const fn rope_beta_slow(&self) -> f64 {
        self.rope_scaling.beta_slow
    }

    pub const fn initial_context_length(&self) -> usize {
        self.initial_context_length
    }

    pub const fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    pub(super) fn layer_types(&self) -> &[AttentionKind] {
        &self.layer_types
    }
}

fn exact<T>(field: &str, actual: T, expected: T) -> Result<(), Error>
where
    T: fmt::Debug + PartialEq,
{
    if actual == expected {
        return Ok(());
    }
    Err(Error::Contract(format!(
        "{field} is {actual:?}; selected GPT-OSS 20B requires {expected:?}"
    )))
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Contract(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
            Self::Contract(message) => write!(formatter, "invalid GPT-OSS 20B config: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Contract(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn selected_config() -> Value {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        let path = Path::new(&runfiles).join("_main/artifacts/gpt-oss-20b-nvfp4/config.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    #[test]
    fn selected_artifact_configuration_is_exactly_admitted() {
        let runfiles = std::env::var_os("TEST_SRCDIR").expect("Bazel provides TEST_SRCDIR");
        let path = Path::new(&runfiles).join("_main/artifacts/gpt-oss-20b-nvfp4/config.json");
        let config = Config::from_file(path).unwrap();
        assert_eq!(config.layers(), 24);
        assert_eq!(config.query_width(), 4_096);
        assert_eq!(config.key_value_width(), 512);
        assert_eq!(config.experts(), 32);
        assert_eq!(config.experts_per_token(), 4);
        assert_eq!(config.vocabulary(), 201_088);
        assert_eq!(config.context_limit(), 131_072);
        assert_eq!(config.layer_types()[0], AttentionKind::SlidingAttention);
        assert_eq!(config.layer_types()[1], AttentionKind::FullAttention);
    }

    #[test]
    fn nearby_architecture_and_schedule_variants_are_rejected() {
        for (field, replacement, expected) in [
            ("num_local_experts", json!(16), "num_local_experts"),
            ("num_experts_per_tok", json!(2), "num_experts_per_tok"),
            ("torch_dtype", json!("float16"), "torch_dtype"),
            ("tie_word_embeddings", json!(true), "tie_word_embeddings"),
        ] {
            let mut value = selected_config();
            value[field] = replacement;
            let error = Config::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }

        let mut value = selected_config();
        value["layer_types"][7] = json!("sliding_attention");
        let error = Config::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("layer_types[7]"), "{error}");
    }

    #[test]
    fn unknown_configuration_fields_are_not_silently_ignored() {
        let mut value = selected_config();
        value["future_quantization_guess"] = json!(true);
        let error = Config::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("unknown field"), "{error}");
    }
}
