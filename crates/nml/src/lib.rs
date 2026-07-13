//! NML's deliberately compact product surface.

#![forbid(unsafe_code)]

pub use nml_checkpoint::{io, safetensors};
pub use nml_derive::NmlStruct;
pub use nml_ir::Tensor;
pub use nml_runtime::{Buffer, Bufferized, Exe, Memory, NmlStruct, Platform, Sharding, exe};
pub use nml_tensor::Slice;
pub use nml_types::{DType as DataType, Shape};
