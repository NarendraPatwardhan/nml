//! Canonical scalar and tensor metadata for every NML layer.
//!
//! This crate deliberately contains no MLIR or PJRT handles. A `DType` is a
//! runtime tensor element contract; MLIR's compiler-only `index` type therefore
//! cannot leak into host storage or executable buffer APIs through this model.

#![forbid(unsafe_code)]

use core::fmt;

/// Maximum tensor rank supported by the NML graph and runtime metadata model.
pub const MAX_RANK: usize = 8;

/// Stable storage for an IEEE 754 binary16 value.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("F16")
            .field(&format_args!("0x{:04x}", self.0))
            .finish()
    }
}

/// Stable storage for a bfloat16 value.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct BFloat16(u16);

impl BFloat16 {
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

impl fmt::Debug for BFloat16 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("BFloat16")
            .field(&format_args!("0x{:04x}", self.0))
            .finish()
    }
}

/// ABI-stable pair representation used by C64 and C128 host buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    pub real: T,
    pub imaginary: T,
}

impl<T> Complex<T> {
    pub const fn new(real: T, imaginary: T) -> Self {
        Self { real, imaginary }
    }
}

/// Host representation of StableHLO C64: two F32 components, 64 bits total.
pub type Complex64 = Complex<f32>;

/// Host representation of StableHLO C128: two F64 components, 128 bits total.
pub type Complex128 = Complex<f64>;

/// The ordinary tensor element types supported by NML.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    C64,
    C128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DTypeClass {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    Float,
    Complex,
}

impl DType {
    pub const ALL: [Self; 15] = [
        Self::Bool,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::F16,
        Self::Bf16,
        Self::F32,
        Self::F64,
        Self::C64,
        Self::C128,
    ];

    pub const fn class(self) -> DTypeClass {
        match self {
            Self::Bool => DTypeClass::Boolean,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => DTypeClass::SignedInteger,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => DTypeClass::UnsignedInteger,
            Self::F16 | Self::Bf16 | Self::F32 | Self::F64 => DTypeClass::Float,
            Self::C64 | Self::C128 => DTypeClass::Complex,
        }
    }

    pub const fn byte_width(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub const fn alignment(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 | Self::C64 => 4,
            Self::I64 | Self::U64 | Self::F64 | Self::C128 => 8,
        }
    }

    /// StableHLO's scalar type spelling inside a ranked tensor type.
    pub const fn stablehlo_spelling(self) -> &'static str {
        match self {
            Self::Bool => "i1",
            Self::I8 | Self::U8 => "i8",
            Self::I16 | Self::U16 => "i16",
            Self::I32 | Self::U32 => "i32",
            Self::I64 | Self::U64 => "i64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "complex<f32>",
            Self::C128 => "complex<f64>",
        }
    }

    pub const fn supports_ordering(self) -> bool {
        !matches!(self, Self::C64 | Self::C128)
    }

    pub const fn require_ordering(self) -> Result<(), TypeError> {
        if self.supports_ordering() {
            Ok(())
        } else {
            Err(TypeError::UnorderedDType(self))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TypeError {
    UnorderedDType(DType),
}

impl fmt::Display for TypeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnorderedDType(dtype) => {
                write!(formatter, "{dtype:?} has no total ordering")
            }
        }
    }
}

impl std::error::Error for TypeError {}

/// Stable identity for a logical tensor or mesh axis.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct AxisTag(u32);

impl AxisTag {
    pub const UNKNOWN: Self = Self(0);

    pub const fn new(identifier: u32) -> Self {
        Self(identifier)
    }

    pub const fn identifier(self) -> u32 {
        self.0
    }
}

/// Requested placement of one logical tensor dimension.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Partition {
    #[default]
    Unspecified,
    Replicated,
    Sharded(AxisTag),
}

/// Physical dimension order, expressed from minor to major as XLA does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    rank: u8,
    minor_to_major: [u8; MAX_RANK],
}

impl Layout {
    pub fn row_major(rank: usize) -> Result<Self, ShapeError> {
        if rank > MAX_RANK {
            return Err(ShapeError::RankTooLarge {
                rank,
                maximum: MAX_RANK,
            });
        }
        let mut order = [0; MAX_RANK];
        for (position, dimension) in (0..rank).rev().enumerate() {
            order[position] = dimension as u8;
        }
        Ok(Self {
            rank: rank as u8,
            minor_to_major: order,
        })
    }

    pub fn from_minor_to_major(order: &[u8]) -> Result<Self, ShapeError> {
        if order.len() > MAX_RANK {
            return Err(ShapeError::RankTooLarge {
                rank: order.len(),
                maximum: MAX_RANK,
            });
        }
        let mut seen = [false; MAX_RANK];
        let mut stored = [0; MAX_RANK];
        for (position, &dimension) in order.iter().enumerate() {
            let dimension = dimension as usize;
            if dimension >= order.len() || seen[dimension] {
                return Err(ShapeError::InvalidLayout);
            }
            seen[dimension] = true;
            stored[position] = dimension as u8;
        }
        Ok(Self {
            rank: order.len() as u8,
            minor_to_major: stored,
        })
    }

    pub fn minor_to_major(&self) -> &[u8] {
        &self.minor_to_major[..self.rank as usize]
    }
}

/// Static tensor shape together with logical and physical dimension metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    dtype: DType,
    rank: u8,
    dimensions: [i64; MAX_RANK],
    axis_tags: [AxisTag; MAX_RANK],
    partitions: [Partition; MAX_RANK],
    layout: Layout,
}

impl Shape {
    pub fn new(dtype: DType, dimensions: &[i64]) -> Result<Self, ShapeError> {
        if dimensions.len() > MAX_RANK {
            return Err(ShapeError::RankTooLarge {
                rank: dimensions.len(),
                maximum: MAX_RANK,
            });
        }
        let mut stored = [0; MAX_RANK];
        for (axis, &dimension) in dimensions.iter().enumerate() {
            if dimension < 0 {
                return Err(ShapeError::NegativeDimension { axis, dimension });
            }
            stored[axis] = dimension;
        }
        Ok(Self {
            dtype,
            rank: dimensions.len() as u8,
            dimensions: stored,
            axis_tags: [AxisTag::UNKNOWN; MAX_RANK],
            partitions: [Partition::Unspecified; MAX_RANK],
            layout: Layout::row_major(dimensions.len())?,
        })
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Changes only the element type, retaining logical axes, partitioning,
    /// dimensions, and physical layout.
    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub const fn rank(&self) -> usize {
        self.rank as usize
    }

    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions[..self.rank()]
    }

    pub fn axis_tags(&self) -> &[AxisTag] {
        &self.axis_tags[..self.rank()]
    }

    pub fn partitions(&self) -> &[Partition] {
        &self.partitions[..self.rank()]
    }

    pub const fn layout(&self) -> Layout {
        self.layout
    }

    pub fn with_axis_tags(mut self, tags: &[AxisTag]) -> Result<Self, ShapeError> {
        self.require_metadata_rank(tags.len(), "axis tags")?;
        self.axis_tags[..tags.len()].copy_from_slice(tags);
        Ok(self)
    }

    pub fn with_partitions(mut self, partitions: &[Partition]) -> Result<Self, ShapeError> {
        self.require_metadata_rank(partitions.len(), "partitions")?;
        self.partitions[..partitions.len()].copy_from_slice(partitions);
        Ok(self)
    }

    pub fn with_layout(mut self, layout: Layout) -> Result<Self, ShapeError> {
        self.require_metadata_rank(layout.rank as usize, "layout")?;
        self.layout = layout;
        Ok(self)
    }

    pub fn element_count(&self) -> Result<usize, ShapeError> {
        self.dimensions()
            .iter()
            .enumerate()
            .try_fold(1usize, |count, (axis, dimension)| {
                let dimension = usize::try_from(*dimension)
                    .map_err(|_| ShapeError::ElementCountOverflow { axis })?;
                count
                    .checked_mul(dimension)
                    .ok_or(ShapeError::ElementCountOverflow { axis })
            })
    }

    pub fn byte_count(&self) -> Result<usize, ShapeError> {
        self.element_count()?
            .checked_mul(self.dtype.byte_width())
            .ok_or(ShapeError::ByteCountOverflow)
    }

    fn require_metadata_rank(&self, actual: usize, field: &'static str) -> Result<(), ShapeError> {
        if actual == self.rank() {
            Ok(())
        } else {
            Err(ShapeError::MetadataRankMismatch {
                field,
                expected: self.rank(),
                actual,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShapeError {
    RankTooLarge {
        rank: usize,
        maximum: usize,
    },
    NegativeDimension {
        axis: usize,
        dimension: i64,
    },
    InvalidLayout,
    MetadataRankMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    ElementCountOverflow {
        axis: usize,
    },
    ByteCountOverflow,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RankTooLarge { rank, maximum } => {
                write!(formatter, "rank {rank} exceeds NML maximum rank {maximum}")
            }
            Self::NegativeDimension { axis, dimension } => {
                write!(formatter, "dimension {axis} is negative: {dimension}")
            }
            Self::InvalidLayout => formatter.write_str("layout is not a dimension permutation"),
            Self::MetadataRankMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} rank mismatch: expected {expected} entries, received {actual}"
            ),
            Self::ElementCountOverflow { axis } => {
                write!(formatter, "element count overflows at dimension {axis}")
            }
            Self::ByteCountOverflow => formatter.write_str("tensor byte count overflows usize"),
        }
    }
}

impl std::error::Error for ShapeError {}
