//! One compiler-private description of the device selected for lowering.
//!
//! Runtime discovery still crosses the crate boundary as the primitive values
//! reported by PJRT.  They are normalized here exactly once.  Semantic graph
//! code and individual kernel selectors consume named facts, never CUDA
//! version arithmetic or product names.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeviceCapabilities {
    Cpu,
    Cuda(CudaCapabilities),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaCapabilities {
    core_count: usize,
    major: u16,
    minor: u16,
}

impl DeviceCapabilities {
    pub(crate) const fn cuda(core_count: usize, major: u16, minor: u16) -> Self {
        Self::Cuda(CudaCapabilities {
            core_count,
            major,
            minor,
        })
    }

    pub(crate) const fn cuda_capabilities(self) -> Option<CudaCapabilities> {
        match self {
            Self::Cpu => None,
            Self::Cuda(capabilities) => Some(capabilities),
        }
    }
}

impl CudaCapabilities {
    pub(crate) const fn core_count(self) -> usize {
        self.core_count
    }

    pub(crate) const fn compute_capability(self) -> (u16, u16) {
        (self.major, self.minor)
    }

    // These hardware facts are part of the closed dispatch vocabulary. They
    // are consumed as the NVFP4 CUDA paths land; keeping them here prevents
    // those lowerings from reintroducing raw version arithmetic.
    #[allow(dead_code)]
    pub(crate) const fn supports_bf16_tensor_cores(self) -> bool {
        self.major >= 8
    }

    #[allow(dead_code)]
    pub(crate) const fn supports_fp8_tensor_cores(self) -> bool {
        self.major > 8 || (self.major == 8 && self.minor >= 9)
    }

    #[allow(dead_code)]
    pub(crate) const fn supports_native_nvfp4(self) -> bool {
        self.major >= 10
    }

    /// Warp-group MMA is a named tuning fact for Hopper and newer devices.
    /// Kernel policy may use it without inferring architecture identity from
    /// raw compute-capability numbers.
    pub(crate) const fn supports_warp_group_mma(self) -> bool {
        self.major >= 9
    }

    /// The pinned XLA/Triton path retained by NML is accepted from Ampere on.
    /// This is an implementation capability, not a statement that older GPUs
    /// cannot execute hand-written CUDA kernels.
    pub(crate) const fn supports_xla_triton(self) -> bool {
        self.major >= 8
    }

    pub(crate) const fn supports_nvfp4_triton_emulation(self) -> bool {
        self.supports_xla_triton() && self.supports_bf16_tensor_cores()
    }

    /// Turing's retained NVFP4 path is a dedicated typed CUDA custom call.
    /// Keeping this distinct from Triton support prevents an architecture
    /// version test from leaking into every semantic lowering.
    pub(crate) const fn supports_nvfp4_turing_custom_call(self) -> bool {
        self.major == 7 && self.minor == 5
    }

    pub(crate) const fn supports_flash_attention_2(self) -> bool {
        self.major == 8
    }

    pub(crate) const fn supports_flash_attention_3(self) -> bool {
        self.major == 9 && self.minor == 0
    }

    pub(crate) const fn supports_grouped_moe(self) -> bool {
        self.supports_xla_triton()
    }
}
