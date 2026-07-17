//! One private capability decision for every attention implementation.
//!
//! Backend names are deliberately absent from the model-authoring API.  The
//! graph describes attention semantics; lowering chooses an implementation
//! after PJRT has reported the actual CUDA device.  Keeping this decision in
//! one module prevents dense and paged attention from inventing subtly
//! different architecture floors.

use nml_types::DType;

// The retained Triton kernel expresses QK and PV as `tt.dot` operations.  Its
// NVIDIA tensor-core lowering requires a K tile of at least sixteen elements;
// smaller heads are still valid attention geometry, but belong on the exact
// StableHLO implementation rather than an under-filled accelerator tile.
const TRITON_MINIMUM_HEAD_DIMENSION: i64 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Backend {
    Portable,
    CudaTriton,
    CudaFlash2,
    CudaFlash3,
}

pub(crate) fn dense(
    dtype: DType,
    head_dimension: i64,
    capability_major: u16,
    capability_minor: u16,
) -> Backend {
    if !matches!(dtype, DType::F16 | DType::Bf16)
        || !(1..=256).contains(&head_dimension)
        || head_dimension % 8 != 0
    {
        return Backend::Portable;
    }
    match capability_major {
        9 if capability_minor == 0 => Backend::CudaFlash3,
        8 => Backend::CudaFlash2,
        _ => Backend::Portable,
    }
}

pub(crate) fn paged(
    dtype: DType,
    head_dimension: i64,
    page_size: i64,
    capability_major: u16,
    capability_minor: u16,
) -> Backend {
    let supported_cuda = capability_major == 8 || (capability_major == 9 && capability_minor == 0);
    if !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32)
        || !supported_cuda
        || head_dimension < TRITON_MINIMUM_HEAD_DIMENSION
    {
        return Backend::Portable;
    }
    if matches!(dtype, DType::F16 | DType::Bf16)
        && (1..=256).contains(&head_dimension)
        && head_dimension % 8 == 0
    {
        if capability_major == 9 && capability_minor == 0 {
            return Backend::CudaFlash3;
        }
        // Original-upstream FA2's paged split-K kernel requires physical
        // pages divisible by 256. Smaller NML pages stay on Triton rather than
        // changing the cache representation for one backend.
        if capability_major == 8 && page_size % 256 == 0 {
            return Backend::CudaFlash2;
        }
    }
    Backend::CudaTriton
}
