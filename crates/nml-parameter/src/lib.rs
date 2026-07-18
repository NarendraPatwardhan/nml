//! Logical model parameters and their physical representation components.
//!
//! `Tensor` and `Buffer` each remain one ordinary shaped value. A `Parameter`
//! instead describes one logical model value whose closed representation owns
//! one or more physical components. Dense storage is the one-component case.

#![forbid(unsafe_code)]

use nml_sharding::Sharding;
use nml_types::{DType, Shape, ShapeError};
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

/// Version of NML's selected rowwise weight-only NVFP4 recipe.
pub const NVFP4_RECIPE_VERSION: u16 = 1;
pub const NVFP4_BLOCK_SIZE: i64 = 16;
pub const NVFP4_VALUES_PER_PAYLOAD_BYTE: i64 = 2;

/// An immutable logical model value independent of any one compiled graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Parameter {
    spec: Arc<ParameterSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParameterSpec {
    logical_name: String,
    logical_shape: Shape,
    representation: RepresentationSpec,
}

/// The admitted representation recipes.
///
/// This enum stays closed deliberately. Adding a representation must make
/// loading, sharding, lowering, accounting, and diagnostics exhaustively
/// consider it rather than registering a partial runtime plugin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RepresentationSpec {
    Dense(DenseSpec),
    NvFp4(NvFp4Spec),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseSpec {
    component: ComponentSpec,
}

/// NML recipe v1: last-axis, one-dimensional NVFP4 weight scaling.
///
/// Every consecutive group of sixteen logical values owns one positive
/// E4M3FN block scale. Two E2M1 codes are packed low-nibble first, and one F32
/// global scale belongs to the complete logical parameter. These semantics are
/// source representation, not a Blackwell-prepared scale swizzle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NvFp4Spec {
    components: [ComponentSpec; 3],
    quantized_axis: u8,
}

/// Stable recipe identity used by executable and loaded-parameter contracts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RepresentationId {
    kind: RepresentationKind,
    version: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RepresentationKind {
    Dense,
    NvFp4,
}

/// One physical tensor required to reconstruct or execute a parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentSpec {
    role: ComponentRole,
    binding_name: String,
    artifact_name: String,
    storage: StorageSpec,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComponentRole {
    Values,
    Payload,
    BlockScales,
    GlobalScale,
}

/// Physical storage consumed by PJRT and kernels.
///
/// Encoded formats use ordinary byte-shaped PJRT buffers while retaining a
/// distinct encoding here. They do not become public graph dtypes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageSpec {
    encoding: StorageEncoding,
    shape: Shape,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StorageEncoding {
    Dense(DType),
    PackedE2M1x2,
    E4M3FnBits,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    EmptyLogicalName,
    EmptyArtifactName,
    InvalidNvFp4LogicalDType(DType),
    InvalidNvFp4Rank,
    EmptyNvFp4QuantizedAxis,
    MisalignedNvFp4Shard {
        axis: usize,
        logical_extent: i64,
        block_size: i64,
    },
    InvalidE2M1Code(u8),
    InvalidE4M3FnScale(u8),
    InvalidGlobalScale,
    NonFiniteValue,
    InvalidEncodedExtent {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    NonZeroPadding,
    Sharding(nml_sharding::Error),
    Shape(ShapeError),
}

impl Parameter {
    /// Creates the one-component dense representation after artifact binding.
    pub fn dense(
        logical_name: impl Into<String>,
        artifact_name: impl Into<String>,
        shape: Shape,
    ) -> Result<Self, Error> {
        let logical_name = require_name(logical_name.into(), Error::EmptyLogicalName)?;
        let artifact_name = require_name(artifact_name.into(), Error::EmptyArtifactName)?;
        let component = ComponentSpec {
            role: ComponentRole::Values,
            binding_name: logical_name.clone(),
            artifact_name,
            storage: StorageSpec {
                encoding: StorageEncoding::Dense(shape.dtype()),
                shape,
            },
        };
        Ok(Self {
            spec: Arc::new(ParameterSpec {
                logical_name,
                logical_shape: shape,
                representation: RepresentationSpec::Dense(DenseSpec { component }),
            }),
        })
    }

    /// Creates NML's selected compact rowwise NVFP4 representation.
    ///
    /// The converter and checkpoint manifest own `artifact_base`; the three
    /// physical records are named `<base>.payload`, `<base>.block_scales`, and
    /// `<base>.global_scale`. Their graph bindings are derived independently
    /// from the logical parameter name so checkpoint aliases cannot alter a
    /// compiled executable's semantic identity.
    pub fn nvfp4(
        logical_name: impl Into<String>,
        artifact_base: impl Into<String>,
        logical_shape: Shape,
    ) -> Result<Self, Error> {
        let logical_name = require_name(logical_name.into(), Error::EmptyLogicalName)?;
        let artifact_base = require_name(artifact_base.into(), Error::EmptyArtifactName)?;
        if !matches!(logical_shape.dtype(), DType::F16 | DType::Bf16) {
            return Err(Error::InvalidNvFp4LogicalDType(logical_shape.dtype()));
        }
        if logical_shape.rank() == 0 {
            return Err(Error::InvalidNvFp4Rank);
        }
        let quantized_axis = logical_shape.rank() - 1;
        let logical_extent = logical_shape.dimensions()[quantized_axis];
        if logical_extent == 0 {
            return Err(Error::EmptyNvFp4QuantizedAxis);
        }

        let payload_shape = encoded_shape(
            logical_shape,
            quantized_axis,
            ceil_div(logical_extent, NVFP4_VALUES_PER_PAYLOAD_BYTE),
            DType::U8,
        )?;
        let scale_shape = encoded_shape(
            logical_shape,
            quantized_axis,
            ceil_div(logical_extent, NVFP4_BLOCK_SIZE),
            DType::U8,
        )?;
        let global_shape = Shape::new(DType::F32, &[])?;

        let component = |role, suffix: &str, storage| ComponentSpec {
            role,
            binding_name: format!("{logical_name}.nvfp4.{suffix}"),
            artifact_name: format!("{artifact_base}.{suffix}"),
            storage,
        };
        let components = [
            component(
                ComponentRole::Payload,
                "payload",
                StorageSpec {
                    encoding: StorageEncoding::PackedE2M1x2,
                    shape: payload_shape,
                },
            ),
            component(
                ComponentRole::BlockScales,
                "block_scales",
                StorageSpec {
                    encoding: StorageEncoding::E4M3FnBits,
                    shape: scale_shape,
                },
            ),
            component(
                ComponentRole::GlobalScale,
                "global_scale",
                StorageSpec {
                    encoding: StorageEncoding::Dense(DType::F32),
                    shape: global_shape,
                },
            ),
        ];
        Ok(Self {
            spec: Arc::new(ParameterSpec {
                logical_name,
                logical_shape,
                representation: RepresentationSpec::NvFp4(NvFp4Spec {
                    components,
                    quantized_axis: quantized_axis as u8,
                }),
            }),
        })
    }

    pub fn name(&self) -> &str {
        &self.spec.logical_name
    }

    pub fn shape(&self) -> Shape {
        self.spec.logical_shape
    }

    pub fn representation(&self) -> &RepresentationSpec {
        &self.spec.representation
    }

    pub fn representation_id(&self) -> RepresentationId {
        self.spec.representation.id()
    }

    pub fn components(&self) -> &[ComponentSpec] {
        self.spec.representation.components()
    }

    pub fn dense_component(&self) -> Option<&ComponentSpec> {
        match &self.spec.representation {
            RepresentationSpec::Dense(spec) => Some(&spec.component),
            RepresentationSpec::NvFp4(_) => None,
        }
    }

    pub fn nvfp4_spec(&self) -> Option<&NvFp4Spec> {
        match &self.spec.representation {
            RepresentationSpec::Dense(_) => None,
            RepresentationSpec::NvFp4(spec) => Some(spec),
        }
    }

    /// Validates the representation's complete logical-to-physical placement.
    ///
    /// Component shapes are derived by the representation, not guessed by the
    /// loader. Ordinary sharding can then slice those physical shapes, provided
    /// a quantized-axis shard owns complete scale blocks. The scalar global
    /// factor has no partition and is consequently replicated across a mesh.
    pub fn validate_sharding(&self, sharding: &Sharding) -> Result<(), Error> {
        sharding
            .validate_shape(self.shape())
            .map_err(Error::Sharding)?;
        if let RepresentationSpec::NvFp4(spec) = self.representation() {
            let axis = spec.quantized_axis();
            let shard_extent = sharding
                .ranges(self.shape(), 0)
                .map_err(Error::Sharding)?
                .into_iter()
                .find_map(|(candidate, _, extent)| (candidate == axis).then_some(extent));
            if let Some(logical_extent) = shard_extent
                && logical_extent % NVFP4_BLOCK_SIZE != 0
            {
                return Err(Error::MisalignedNvFp4Shard {
                    axis,
                    logical_extent,
                    block_size: NVFP4_BLOCK_SIZE,
                });
            }
        }
        for component in self.components() {
            sharding
                .validate_shape(component.storage().shape())
                .map_err(Error::Sharding)?;
        }
        Ok(())
    }
}

impl RepresentationSpec {
    pub const fn id(&self) -> RepresentationId {
        match self {
            Self::Dense(_) => RepresentationId {
                kind: RepresentationKind::Dense,
                version: 1,
            },
            Self::NvFp4(_) => RepresentationId {
                kind: RepresentationKind::NvFp4,
                version: NVFP4_RECIPE_VERSION,
            },
        }
    }

    pub fn components(&self) -> &[ComponentSpec] {
        match self {
            Self::Dense(spec) => std::slice::from_ref(&spec.component),
            Self::NvFp4(spec) => &spec.components,
        }
    }
}

impl RepresentationId {
    pub const fn kind(self) -> RepresentationKind {
        self.kind
    }

    pub const fn version(self) -> u16 {
        self.version
    }
}

impl NvFp4Spec {
    pub const fn quantized_axis(&self) -> usize {
        self.quantized_axis as usize
    }

    pub const fn block_size(&self) -> usize {
        NVFP4_BLOCK_SIZE as usize
    }

    pub const fn earlier_value_uses_low_nibble(&self) -> bool {
        true
    }
}

impl ComponentSpec {
    pub const fn role(&self) -> ComponentRole {
        self.role
    }

    pub fn binding_name(&self) -> &str {
        &self.binding_name
    }

    pub fn artifact_name(&self) -> &str {
        &self.artifact_name
    }

    pub const fn storage(&self) -> StorageSpec {
        self.storage
    }
}

impl StorageSpec {
    pub const fn encoding(self) -> StorageEncoding {
        self.encoding
    }

    pub const fn shape(self) -> Shape {
        self.shape
    }
}

/// Exact scalar and row codec for NML NVFP4 recipe v1.
///
/// This is permanent product/reference code shared by checkpoint inspection
/// and the CPU implementation. It deliberately exposes no graph dtype.
pub mod nvfp4 {
    use super::{Error, NVFP4_BLOCK_SIZE};

    const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    const E4M3FN_MAX: f32 = 448.0;
    const E2M1_MAX: f32 = 6.0;

    #[derive(Clone, Debug, PartialEq)]
    pub struct EncodedRow {
        payload: Vec<u8>,
        block_scales: Vec<u8>,
    }

    impl EncodedRow {
        pub fn payload(&self) -> &[u8] {
            &self.payload
        }

        pub fn block_scales(&self) -> &[u8] {
            &self.block_scales
        }
    }

    pub fn decode_e2m1(code: u8) -> Result<f32, Error> {
        if code > 0x0f {
            return Err(Error::InvalidE2M1Code(code));
        }
        let magnitude = E2M1_MAGNITUDES[usize::from(code & 0x07)];
        Ok(if code & 0x08 == 0 {
            magnitude
        } else {
            -magnitude
        })
    }

    pub fn encode_e2m1(value: f32) -> Result<u8, Error> {
        if !value.is_finite() {
            return Err(Error::NonFiniteValue);
        }
        let sign = u8::from(value.is_sign_negative()) << 3;
        let magnitude = value.abs();
        let mut best = 0u8;
        let mut best_distance = f32::INFINITY;
        for (code, candidate) in E2M1_MAGNITUDES.iter().copied().enumerate() {
            let distance = (magnitude - candidate).abs();
            if distance < best_distance
                || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
            {
                best = code as u8;
                best_distance = distance;
            }
        }
        Ok(sign | best)
    }

    pub fn decode_e4m3fn_scale(bits: u8) -> Result<f32, Error> {
        if bits & 0x80 != 0 {
            return Err(Error::InvalidE4M3FnScale(bits));
        }
        let exponent = (bits >> 3) & 0x0f;
        let fraction = bits & 0x07;
        if exponent == 0x0f && fraction == 0x07 {
            return Err(Error::InvalidE4M3FnScale(bits));
        }
        if exponent == 0 {
            return Ok(f32::from(fraction) * 2.0f32.powi(-9));
        }
        Ok((1.0 + f32::from(fraction) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7))
    }

    pub fn encode_e4m3fn_scale(value: f32) -> Result<u8, Error> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::NonFiniteValue);
        }
        let mut best = 0u8;
        let mut best_distance = f32::INFINITY;
        for bits in 0u8..=0x7e {
            let candidate = decode_e4m3fn_scale(bits)?;
            let distance = (value - candidate).abs();
            if distance < best_distance
                || (distance == best_distance && bits & 1 == 0 && best & 1 != 0)
            {
                best = bits;
                best_distance = distance;
            }
        }
        Ok(best)
    }

    /// Computes `global_amax / (E4M3FN_MAX * E2M1_MAX)`.
    ///
    /// An all-zero tensor uses `1.0`; its block scales and payload remain zero.
    pub fn global_scale(values: &[f32]) -> Result<f32, Error> {
        let mut maximum = 0.0f32;
        for &value in values {
            if !value.is_finite() {
                return Err(Error::NonFiniteValue);
            }
            maximum = maximum.max(value.abs());
        }
        Ok(if maximum == 0.0 {
            1.0
        } else {
            maximum / (E4M3FN_MAX * E2M1_MAX)
        })
    }

    /// Quantizes one logical last-axis row using a tensor-global scale.
    pub fn quantize_row(values: &[f32], global_scale: f32) -> Result<EncodedRow, Error> {
        require_global_scale(global_scale)?;
        let mut payload = vec![0u8; values.len().div_ceil(2)];
        let mut block_scales = Vec::with_capacity(values.len().div_ceil(NVFP4_BLOCK_SIZE as usize));

        for (block_index, block) in values.chunks(NVFP4_BLOCK_SIZE as usize).enumerate() {
            let mut block_amax = 0.0f32;
            for &value in block {
                if !value.is_finite() {
                    return Err(Error::NonFiniteValue);
                }
                block_amax = block_amax.max(value.abs());
            }
            let unrounded_scale = (block_amax / E2M1_MAX) / global_scale;
            let scale_bits = encode_e4m3fn_scale(unrounded_scale)?;
            let block_scale = decode_e4m3fn_scale(scale_bits)?;
            block_scales.push(scale_bits);

            for (offset, &value) in block.iter().enumerate() {
                let logical_index = block_index * NVFP4_BLOCK_SIZE as usize + offset;
                let code = if block_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(value / (block_scale * global_scale))?
                };
                let packed = &mut payload[logical_index / 2];
                if logical_index & 1 == 0 {
                    *packed = code;
                } else {
                    *packed |= code << 4;
                }
            }
        }
        Ok(EncodedRow {
            payload,
            block_scales,
        })
    }

    pub fn dequantize_row(
        payload: &[u8],
        block_scales: &[u8],
        global_scale: f32,
        logical_length: usize,
    ) -> Result<Vec<f32>, Error> {
        require_global_scale(global_scale)?;
        let expected_payload = logical_length.div_ceil(2);
        if payload.len() != expected_payload {
            return Err(Error::InvalidEncodedExtent {
                component: "payload",
                expected: expected_payload,
                actual: payload.len(),
            });
        }
        let expected_scales = logical_length.div_ceil(NVFP4_BLOCK_SIZE as usize);
        if block_scales.len() != expected_scales {
            return Err(Error::InvalidEncodedExtent {
                component: "block scales",
                expected: expected_scales,
                actual: block_scales.len(),
            });
        }
        if logical_length & 1 != 0 && payload.last().is_some_and(|byte| byte & 0xf0 != 0) {
            return Err(Error::NonZeroPadding);
        }

        let mut result = Vec::with_capacity(logical_length);
        for index in 0..logical_length {
            let byte = payload[index / 2];
            let code = if index & 1 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let block_scale = decode_e4m3fn_scale(block_scales[index / NVFP4_BLOCK_SIZE as usize])?;
            if block_scale == 0.0 && code & 0x07 != 0 {
                return Err(Error::InvalidE4M3FnScale(
                    block_scales[index / NVFP4_BLOCK_SIZE as usize],
                ));
            }
            // Form the effective scale once, matching quantization's divisor.
            // Besides documenting the recipe's association explicitly, this
            // avoids an avoidable extra rounding between the block and global
            // factors in the CPU reference path.
            result.push(decode_e2m1(code)? * (block_scale * global_scale));
        }
        Ok(result)
    }

    fn require_global_scale(scale: f32) -> Result<(), Error> {
        if scale.is_finite() && scale > 0.0 {
            Ok(())
        } else {
            Err(Error::InvalidGlobalScale)
        }
    }
}

fn require_name(value: String, error: Error) -> Result<String, Error> {
    if value.is_empty() {
        Err(error)
    } else {
        Ok(value)
    }
}

fn ceil_div(value: i64, divisor: i64) -> i64 {
    value / divisor + i64::from(value % divisor != 0)
}

fn encoded_shape(
    logical: Shape,
    encoded_axis: usize,
    encoded_extent: i64,
    dtype: DType,
) -> Result<Shape, Error> {
    let mut dimensions = logical.dimensions().to_vec();
    dimensions[encoded_axis] = encoded_extent;
    Ok(Shape::new(dtype, &dimensions)?
        .with_axis_tags(logical.axis_tags())?
        .with_partitions(logical.partitions())?
        .with_layout(logical.layout())?)
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogicalName => formatter.write_str("parameter logical name is empty"),
            Self::EmptyArtifactName => formatter.write_str("parameter artifact name is empty"),
            Self::InvalidNvFp4LogicalDType(dtype) => {
                write!(
                    formatter,
                    "NVFP4 parameter logical dtype {dtype:?} is not F16 or BF16"
                )
            }
            Self::InvalidNvFp4Rank => {
                formatter.write_str("NVFP4 parameter must have rank at least one")
            }
            Self::EmptyNvFp4QuantizedAxis => {
                formatter.write_str("NVFP4 parameter quantized axis is empty")
            }
            Self::MisalignedNvFp4Shard {
                axis,
                logical_extent,
                block_size,
            } => write!(
                formatter,
                "NVFP4 logical axis {axis} shard extent {logical_extent} is not aligned to the {block_size}-value scale block"
            ),
            Self::InvalidE2M1Code(code) => write!(formatter, "invalid E2M1 code 0x{code:02x}"),
            Self::InvalidE4M3FnScale(bits) => {
                write!(
                    formatter,
                    "invalid non-negative E4M3FN scale bits 0x{bits:02x}"
                )
            }
            Self::InvalidGlobalScale => {
                formatter.write_str("NVFP4 global scale must be finite and positive")
            }
            Self::NonFiniteValue => formatter.write_str("NVFP4 source values must be finite"),
            Self::InvalidEncodedExtent {
                component,
                expected,
                actual,
            } => write!(
                formatter,
                "NVFP4 {component} extent mismatch: expected {expected} bytes, received {actual}"
            ),
            Self::NonZeroPadding => formatter.write_str("NVFP4 payload has nonzero edge padding"),
            Self::Sharding(error) => write!(formatter, "invalid parameter sharding: {error}"),
            Self::Shape(error) => write!(formatter, "invalid NVFP4 component shape: {error}"),
        }
    }
}

impl StdError for Error {}

impl From<ShapeError> for Error {
    fn from(error: ShapeError) -> Self {
        Self::Shape(error)
    }
}

impl From<nml_sharding::Error> for Error {
    fn from(error: nml_sharding::Error) -> Self {
        Self::Sharding(error)
    }
}
