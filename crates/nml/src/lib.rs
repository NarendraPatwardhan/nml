//! NML's deliberately compact product surface.

#![forbid(unsafe_code)]

pub use nml_checkpoint::{io, safetensors};
pub use nml_derive::NmlStruct;
pub use nml_ir::Tensor;
pub use nml_runtime::{Buffer, Bufferized, Exe, Memory, NmlStruct, Platform, Sharding, exe};
pub use nml_tensor::Slice;
pub use nml_types::{AxisTag, DType as DataType, Partition, Shape};

pub mod attention {
    pub use nml_ir::{AttentionOptions as Options, RopeLayout, RopeOptions, RopeScaling};
    pub use nml_runtime::{Cache, CacheSpec};
}

pub mod tokenizer {
    pub use nml_tokenizer::{Decoder, Error, Tokenizer};
}
