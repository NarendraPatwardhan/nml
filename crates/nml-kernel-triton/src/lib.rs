//! Private, typed construction of TTIR consumed by XLA's Triton custom call.
//!
//! A builder owns symbolic SSA values only while one kernel is authored.  Its
//! finish boundary reparses and verifies the complete module in an isolated
//! Triton MLIR context, then returns canonical text together with the exact
//! authored function ABI. The StableHLO graph sees neither raw MLIR pointers,
//! partially constructed TTIR, nor an independently reconstructed ABI.

#![forbid(unsafe_code)]

use nml_mlir::Context;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

mod moe;
mod nvfp4;
mod paged_attention;
mod specification;
mod unified_attention;

pub use moe::{GatedActivation, GroupedProjectionConfig, build_grouped_projection};
pub use nvfp4::{
    NvFp4EmbeddingConfig, NvFp4GroupedProjectionConfig, NvFp4GroupedRole, NvFp4LinearConfig,
    build_nvfp4_embedding, build_nvfp4_grouped_projection,
    build_nvfp4_grouped_projection_finalize, build_nvfp4_linear, build_nvfp4_linear_finalize,
};
pub use paged_attention::{AttentionGeometry, AttentionLaunch, select_attention_launch};
pub use specification::{KernelLaunch, KernelSpec, OutputAlias, TensorSpec};
pub use unified_attention::{
    PagedAttention2dConfig, PagedAttention3dConfig, SegmentReductionConfig,
    build_paged_attention_2d, build_paged_attention_3d, build_segment_reduction,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    I1,
    I8,
    U8,
    I16,
    I32,
    I64,
    F16,
    Bf16,
    F32,
    F64,
}

impl DType {
    const fn spelling(self) -> &'static str {
        match self {
            Self::I1 => "i1",
            Self::I8 | Self::U8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32 | Self::F64)
    }

    const fn bit_width(self) -> u8 {
        match self {
            Self::I1 => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::F32 => 32,
            Self::I64 | Self::F64 => 64,
        }
    }

    const fn is_unsigned(self) -> bool {
        matches!(self, Self::I1 | Self::U8)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgumentKind {
    Pointer { element: DType, address_space: i32 },
    Scalar(DType),
}

impl ArgumentKind {
    fn spelling(self) -> String {
        match self {
            Self::Pointer {
                element,
                address_space: 1,
            } => format!("!tt.ptr<{}>", element.spelling()),
            Self::Pointer {
                element,
                address_space,
            } => format!("!tt.ptr<{}, {address_space}>", element.spelling()),
            Self::Scalar(dtype) => dtype.spelling().to_owned(),
        }
    }
}

/// One verified TTIR module and the public function ABI that authored it.
///
/// The fields are intentionally private. A `Kernel` can only come from
/// [`Builder::finish`], so the canonical TTIR text and its typed argument list
/// cannot drift before [`KernelSpec`](crate::KernelSpec) binds tensor shapes to
/// the XLA custom call.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Kernel {
    name: String,
    ir: String,
    arguments: Vec<KernelArgument>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KernelArgument {
    name: String,
    kind: ArgumentKind,
}

impl Kernel {
    pub fn text(&self) -> &str {
        &self.ir
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn arguments(&self) -> &[KernelArgument] {
        &self.arguments
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.ir)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ValueType {
    Scalar(DType),
    Pointer {
        element: DType,
        address_space: i32,
    },
    Tensor {
        shape: Vec<i64>,
        element: DType,
        pointer_address_space: Option<i32>,
    },
}

impl ValueType {
    fn spelling(&self) -> String {
        match self {
            Self::Scalar(dtype) => dtype.spelling().to_owned(),
            Self::Pointer {
                element,
                address_space,
            } => ArgumentKind::Pointer {
                element: *element,
                address_space: *address_space,
            }
            .spelling(),
            Self::Tensor {
                shape,
                element,
                pointer_address_space,
            } => {
                let dimensions = shape
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                let element = pointer_address_space.map_or_else(
                    || element.spelling().to_owned(),
                    |address_space| {
                        ArgumentKind::Pointer {
                            element: *element,
                            address_space,
                        }
                        .spelling()
                    },
                );
                format!("tensor<{dimensions}x{element}>")
            }
        }
    }

    fn element(&self) -> DType {
        match self {
            Self::Scalar(dtype) | Self::Pointer { element: dtype, .. } => *dtype,
            Self::Tensor { element, .. } => *element,
        }
    }

    fn is_integer(&self) -> bool {
        !self.element().is_float()
    }

    fn loaded(&self) -> Option<Self> {
        match self {
            Self::Pointer { element, .. } => Some(Self::Scalar(*element)),
            Self::Tensor {
                shape,
                element,
                pointer_address_space: Some(_),
            } => Some(Self::Tensor {
                shape: shape.clone(),
                element: *element,
                pointer_address_space: None,
            }),
            _ => None,
        }
    }

    fn with_element(&self, element: DType) -> Result<Self, Error> {
        match self {
            Self::Scalar(_) => Ok(Self::Scalar(element)),
            Self::Tensor {
                shape,
                pointer_address_space: None,
                ..
            } => Ok(Self::Tensor {
                shape: shape.clone(),
                element,
                pointer_address_space: None,
            }),
            Self::Pointer { .. }
            | Self::Tensor {
                pointer_address_space: Some(_),
                ..
            } => Err(Error::TypeMismatch { operation: "cast" }),
        }
    }

    fn condition_type(&self) -> Self {
        match self {
            Self::Tensor { shape, .. } => Self::Tensor {
                shape: shape.clone(),
                element: DType::I1,
                pointer_address_space: None,
            },
            Self::Scalar(_) => Self::Scalar(DType::I1),
            Self::Pointer { .. } => Self::Scalar(DType::I1),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Value {
    owner: u64,
    id: String,
    value_type: ValueType,
}

static NEXT_BUILDER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Comparison {
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    Sum,
    Maximum,
}

/// Source-level cache intent for a Triton memory operation.
///
/// These values deliberately mirror the pinned Triton dialect rather than
/// exposing its integer attribute encoding to kernel authors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CacheModifier {
    Default,
    CacheAll,
    Streaming,
}

impl CacheModifier {
    const fn dialect_value(self) -> i32 {
        match self {
            Self::Default => 1,
            Self::CacheAll => 2,
            Self::Streaming => 5,
        }
    }
}

/// A typed load policy. Compact weights use PTX's streaming cache operator,
/// whose contract already gives the line first-eviction priority, while the
/// much smaller activation tile uses cache-all. We deliberately leave the
/// independent Triton eviction attribute at `normal`: on pre-Blackwell PTX,
/// combining `.cs` with `.evict_first` or `.ca` with `.evict_last` is illegal.
/// One semantic intent therefore lowers to one portable cache control instead
/// of two redundant target-specific hints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoadPolicy {
    cache: CacheModifier,
}

impl LoadPolicy {
    pub const DEFAULT: Self = Self {
        cache: CacheModifier::Default,
    };
    pub const STREAMING: Self = Self {
        cache: CacheModifier::Streaming,
    };
    pub const REUSED: Self = Self {
        cache: CacheModifier::CacheAll,
    };
}

/// Storage interpretation for one `tt.dot_scaled` operand.
///
/// Packed formats describe the values represented by an I8 tensor; they are
/// not general graph dtypes. The enum therefore lives only in this private
/// kernel-construction crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleDotElement {
    E4M3,
    E5M2,
    E2M3,
    E3M2,
    E2M1,
    Bf16,
    F16,
}

impl ScaleDotElement {
    const fn spelling(self) -> &'static str {
        match self {
            Self::E4M3 => "e4m3",
            Self::E5M2 => "e5m2",
            Self::E2M3 => "e2m3",
            Self::E3M2 => "e3m2",
            Self::E2M1 => "e2m1",
            Self::Bf16 => "bf16",
            Self::F16 => "fp16",
        }
    }
}

#[derive(Clone, Debug)]
struct Argument {
    name: String,
    kind: ArgumentKind,
    value: Value,
    divisibility: Option<i32>,
}

#[derive(Debug)]
pub enum Error {
    InvalidName(String),
    DuplicateArgument(String),
    InvalidDivisibility(i32),
    TypeMismatch { operation: &'static str },
    ExpectedPointer,
    ExpectedIntegerOffset,
    InvalidProgramDimension(u8),
    InvalidTensorShape(Vec<i64>),
    InvalidAxis { axis: usize, rank: usize },
    InvalidRange { start: i32, end: i32 },
    ExpectedTensor,
    ExpectedCondition,
    InvalidRegionYield { operation: &'static str },
    EmptyRegionState { operation: &'static str },
    ForeignValue,
    InvalidKernelSpec(&'static str),
    AlreadyTerminated,
    MissingTerminator,
    Mlir(nml_mlir::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName(name) => write!(formatter, "invalid Triton identifier {name:?}"),
            Self::DuplicateArgument(name) => {
                write!(formatter, "duplicate Triton argument {name:?}")
            }
            Self::InvalidDivisibility(value) => {
                write!(
                    formatter,
                    "Triton argument divisibility must be positive, got {value}"
                )
            }
            Self::TypeMismatch { operation } => {
                write!(
                    formatter,
                    "Triton {operation} operands have incompatible types"
                )
            }
            Self::ExpectedPointer => formatter.write_str("Triton operation requires a pointer"),
            Self::ExpectedIntegerOffset => {
                formatter.write_str("Triton pointer offset must be an integer scalar")
            }
            Self::InvalidProgramDimension(axis) => {
                write!(
                    formatter,
                    "Triton program dimension {axis} is outside x/y/z"
                )
            }
            Self::InvalidTensorShape(shape) => {
                write!(formatter, "invalid TTIR tensor shape {shape:?}")
            }
            Self::InvalidAxis { axis, rank } => {
                write!(formatter, "TTIR axis {axis} is invalid for rank {rank}")
            }
            Self::InvalidRange { start, end } => write!(
                formatter,
                "TTIR range [{start}, {end}) must have a positive power-of-two length"
            ),
            Self::ExpectedTensor => formatter.write_str("Triton operation requires a tensor"),
            Self::ExpectedCondition => {
                formatter.write_str("Triton selection requires a matching i1 condition")
            }
            Self::InvalidRegionYield { operation } => {
                write!(
                    formatter,
                    "Triton {operation} region yielded incompatible values"
                )
            }
            Self::EmptyRegionState { operation } => {
                write!(formatter, "Triton {operation} requires carried values")
            }
            Self::ForeignValue => {
                formatter.write_str("Triton value belongs to a different kernel builder")
            }
            Self::InvalidKernelSpec(message) => {
                write!(formatter, "invalid Triton kernel specification: {message}")
            }
            Self::AlreadyTerminated => formatter.write_str("Triton block is already terminated"),
            Self::MissingTerminator => formatter.write_str("Triton kernel has no tt.return"),
            Self::Mlir(error) => error.fmt(formatter),
        }
    }
}

impl StdError for Error {}

impl From<nml_mlir::Error> for Error {
    fn from(error: nml_mlir::Error) -> Self {
        Self::Mlir(error)
    }
}

pub struct Builder {
    owner: u64,
    name: String,
    arguments: Vec<Argument>,
    body: Vec<String>,
    next_value: usize,
    terminated: bool,
}

impl Builder {
    pub fn new(name: &str) -> Result<Self, Error> {
        require_identifier(name)?;
        Ok(Self {
            owner: NEXT_BUILDER_ID.fetch_add(1, Ordering::Relaxed),
            name: name.to_owned(),
            arguments: Vec::new(),
            body: Vec::new(),
            next_value: 0,
            terminated: false,
        })
    }

    pub fn argument(
        &mut self,
        name: &str,
        kind: ArgumentKind,
        divisibility: Option<i32>,
    ) -> Result<Value, Error> {
        self.require_open()?;
        require_identifier(name)?;
        if self.arguments.iter().any(|argument| argument.name == name) {
            return Err(Error::DuplicateArgument(name.to_owned()));
        }
        if let Some(divisibility) = divisibility {
            if divisibility <= 0 {
                return Err(Error::InvalidDivisibility(divisibility));
            }
        }
        let value_type = match kind {
            ArgumentKind::Pointer {
                element,
                address_space,
            } => ValueType::Pointer {
                element,
                address_space,
            },
            ArgumentKind::Scalar(dtype) => ValueType::Scalar(dtype),
        };
        let value = Value {
            owner: self.owner,
            id: format!("%{name}"),
            value_type,
        };
        self.arguments.push(Argument {
            name: name.to_owned(),
            kind,
            value: value.clone(),
            divisibility,
        });
        Ok(value)
    }

    pub fn program_id(&mut self, axis: u8) -> Result<Value, Error> {
        let dimension = match axis {
            0 => "x",
            1 => "y",
            2 => "z",
            other => return Err(Error::InvalidProgramDimension(other)),
        };
        self.emit_value(
            ValueType::Scalar(DType::I32),
            format!("tt.get_program_id {dimension} : i32"),
        )
    }

    pub fn range(&mut self, start: i32, end: i32) -> Result<Value, Error> {
        let length = end
            .checked_sub(start)
            .ok_or(Error::InvalidRange { start, end })?;
        if length <= 0 || !(length as u32).is_power_of_two() {
            return Err(Error::InvalidRange { start, end });
        }
        let value_type = ValueType::Tensor {
            shape: vec![i64::from(length)],
            element: DType::I32,
            pointer_address_space: None,
        };
        self.emit_value(
            value_type.clone(),
            format!(
                "\"tt.make_range\"() {{end = {end} : i32, start = {start} : i32}} : () -> {}",
                value_type.spelling()
            ),
        )
    }

    pub fn splat(&mut self, value: &Value, shape: &[i64]) -> Result<Value, Error> {
        self.require_values(&[value])?;
        validate_shape(shape)?;
        if !matches!(
            value.value_type,
            ValueType::Scalar(_) | ValueType::Pointer { .. }
        ) {
            return Err(Error::TypeMismatch { operation: "splat" });
        }
        let (element, pointer_address_space) = match &value.value_type {
            ValueType::Scalar(dtype) => (*dtype, None),
            ValueType::Pointer {
                element,
                address_space,
            } => (*element, Some(*address_space)),
            ValueType::Tensor { .. } => unreachable!(),
        };
        let result = ValueType::Tensor {
            shape: shape.to_vec(),
            element,
            pointer_address_space,
        };
        self.emit_value(
            result.clone(),
            format!(
                "\"tt.splat\"({}) : ({}) -> {}",
                value.id,
                value.value_type.spelling(),
                result.spelling()
            ),
        )
    }

    pub fn expand_dimension(&mut self, value: &Value, axis: usize) -> Result<Value, Error> {
        self.require_values(&[value])?;
        let ValueType::Tensor {
            shape,
            element,
            pointer_address_space,
        } = &value.value_type
        else {
            return Err(Error::ExpectedTensor);
        };
        if axis > shape.len() {
            return Err(Error::InvalidAxis {
                axis,
                rank: shape.len(),
            });
        }
        let mut result_shape = shape.clone();
        result_shape.insert(axis, 1);
        let result = ValueType::Tensor {
            shape: result_shape,
            element: *element,
            pointer_address_space: *pointer_address_space,
        };
        self.emit_value(
            result.clone(),
            format!(
                "\"tt.expand_dims\"({}) {{axis = {axis} : i32}} : ({}) -> {}",
                value.id,
                value.value_type.spelling(),
                result.spelling()
            ),
        )
    }

    pub fn broadcast(&mut self, value: &Value, shape: &[i64]) -> Result<Value, Error> {
        self.require_values(&[value])?;
        validate_shape(shape)?;
        let ValueType::Tensor {
            shape: source,
            element,
            pointer_address_space,
        } = &value.value_type
        else {
            return Err(Error::ExpectedTensor);
        };
        if source.len() != shape.len()
            || source
                .iter()
                .zip(shape)
                .any(|(from, to)| from != to && *from != 1)
        {
            return Err(Error::TypeMismatch {
                operation: "broadcast",
            });
        }
        let result = ValueType::Tensor {
            shape: shape.to_vec(),
            element: *element,
            pointer_address_space: *pointer_address_space,
        };
        self.emit_value(
            result.clone(),
            format!(
                "\"tt.broadcast\"({}) : ({}) -> {}",
                value.id,
                value.value_type.spelling(),
                result.spelling()
            ),
        )
    }

    /// Changes only a tensor's static dimension grouping.
    ///
    /// Compact kernels use this to load one physical scale per representation
    /// block and broadcast that value across its sixteen logical lanes. The
    /// element count is checked here; callers cannot use reshape as an
    /// indexing or storage-layout escape hatch.
    pub fn reshape(&mut self, value: &Value, shape: &[i64]) -> Result<Value, Error> {
        self.require_values(&[value])?;
        validate_shape(shape)?;
        let ValueType::Tensor {
            shape: source,
            element,
            pointer_address_space: None,
        } = &value.value_type
        else {
            return Err(Error::ExpectedTensor);
        };
        let source_elements = element_count(source).ok_or(Error::TypeMismatch {
            operation: "reshape",
        })?;
        let result_elements = element_count(shape).ok_or(Error::TypeMismatch {
            operation: "reshape",
        })?;
        if source_elements != result_elements {
            return Err(Error::TypeMismatch {
                operation: "reshape",
            });
        }
        let result = ValueType::Tensor {
            shape: shape.to_vec(),
            element: *element,
            pointer_address_space: None,
        };
        self.emit_value(
            result.clone(),
            format!(
                "\"tt.reshape\"({}) : ({}) -> {}",
                value.id,
                value.value_type.spelling(),
                result.spelling()
            ),
        )
    }

    pub fn integer(&mut self, value: i64, dtype: DType) -> Result<Value, Error> {
        if dtype.is_float() {
            return Err(Error::TypeMismatch {
                operation: "integer constant",
            });
        }
        self.emit_value(
            ValueType::Scalar(dtype),
            format!("arith.constant {value} : {}", dtype.spelling()),
        )
    }

    pub fn float(&mut self, value: f64, dtype: DType) -> Result<Value, Error> {
        if !dtype.is_float() {
            return Err(Error::TypeMismatch {
                operation: "floating constant",
            });
        }
        let literal = if value.is_finite() {
            // MLIR deliberately does not accept Rust's compact `1e0` form:
            // the decimal point distinguishes the literal from an SSA/token.
            format!("{value:.17e}")
        } else if value.is_infinite() {
            let negative = value.is_sign_negative();
            match dtype {
                DType::F16 => format!("0x{:04X}", if negative { 0xfc00u16 } else { 0x7c00 }),
                DType::Bf16 => format!("0x{:04X}", if negative { 0xff80u16 } else { 0x7f80 }),
                DType::F32 => format!(
                    "0x{:08X}",
                    if negative {
                        0xff80_0000u32
                    } else {
                        0x7f80_0000
                    }
                ),
                DType::F64 => format!(
                    "0x{:016X}",
                    if negative {
                        0xfff0_0000_0000_0000u64
                    } else {
                        0x7ff0_0000_0000_0000
                    }
                ),
                _ => unreachable!(),
            }
        } else {
            return Err(Error::TypeMismatch {
                operation: "NaN floating constant",
            });
        };
        self.emit_value(
            ValueType::Scalar(dtype),
            format!("arith.constant {literal} : {}", dtype.spelling()),
        )
    }

    pub fn add(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        self.binary("add", left, right)
    }

    pub fn multiply(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        self.binary("mul", left, right)
    }

    pub fn subtract(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        self.binary("sub", left, right)
    }

    pub fn divide(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        let operation = if left.value_type.element().is_float() {
            "div"
        } else if left.value_type.element().is_unsigned() {
            "divu"
        } else {
            "divs"
        };
        self.binary(operation, left, right)
    }

    pub fn remainder(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        let operation = if left.value_type.element().is_float() {
            "rem"
        } else if left.value_type.element().is_unsigned() {
            "remu"
        } else {
            "rems"
        };
        self.binary(operation, left, right)
    }

    pub fn negate(&mut self, value: &Value) -> Result<Value, Error> {
        self.require_values(&[value])?;
        if !value.value_type.element().is_float()
            || matches!(value.value_type, ValueType::Pointer { .. })
            || matches!(
                value.value_type,
                ValueType::Tensor {
                    pointer_address_space: Some(_),
                    ..
                }
            )
        {
            return Err(Error::TypeMismatch {
                operation: "negation",
            });
        }
        self.emit_value(
            value.value_type.clone(),
            format!("arith.negf {} : {}", value.id, value.value_type.spelling()),
        )
    }

    pub fn cast(&mut self, value: &Value, dtype: DType) -> Result<Value, Error> {
        self.require_values(&[value])?;
        let source = value.value_type.element();
        if source == dtype {
            return Ok(value.clone());
        }
        let result = value.value_type.with_element(dtype)?;
        let operation = match (source.is_float(), dtype.is_float()) {
            (true, true) if dtype.bit_width() > source.bit_width() => "extf",
            (true, true) => "truncf",
            (true, false) => "fptosi",
            (false, true) if source.is_unsigned() => "uitofp",
            (false, true) => "sitofp",
            (false, false) if dtype.bit_width() > source.bit_width() && source.is_unsigned() => {
                "extui"
            }
            (false, false) if dtype.bit_width() > source.bit_width() => "extsi",
            (false, false) => "trunci",
        };
        self.emit_value(
            result.clone(),
            format!(
                "arith.{operation} {} : {} to {}",
                value.id,
                value.value_type.spelling(),
                result.spelling()
            ),
        )
    }

    /// Reinterprets equal-width scalar or tensor elements without conversion.
    ///
    /// This is deliberately narrower than `cast`: pointers are never admitted
    /// and the source and destination widths must match. NVFP4 scale decoding
    /// uses it to construct exact IEEE F32 values from integer exponent and
    /// mantissa fields without transcendental arithmetic.
    pub fn bitcast(&mut self, value: &Value, dtype: DType) -> Result<Value, Error> {
        self.require_values(&[value])?;
        let source = value.value_type.element();
        if source == dtype {
            return Ok(value.clone());
        }
        if source.bit_width() != dtype.bit_width()
            || matches!(value.value_type, ValueType::Pointer { .. })
            || matches!(
                value.value_type,
                ValueType::Tensor {
                    pointer_address_space: Some(_),
                    ..
                }
            )
        {
            return Err(Error::TypeMismatch {
                operation: "bitcast",
            });
        }
        let result = value.value_type.with_element(dtype)?;
        self.emit_value(
            result.clone(),
            format!(
                "\"tt.bitcast\"({}) : ({}) -> {}",
                value.id,
                value.value_type.spelling(),
                result.spelling()
            ),
        )
    }

    pub fn bit_and(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        if !left.value_type.is_integer() || !right.value_type.is_integer() {
            return Err(Error::TypeMismatch {
                operation: "bitwise and",
            });
        }
        self.binary("and", left, right)
    }

    pub fn shift_right_logical(&mut self, value: &Value, amount: &Value) -> Result<Value, Error> {
        if !value.value_type.is_integer() || !amount.value_type.is_integer() {
            return Err(Error::TypeMismatch {
                operation: "logical shift right",
            });
        }
        self.binary("shru", value, amount)
    }

    pub fn minimum(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        self.ordered_extreme("min", left, right)
    }

    pub fn maximum(&mut self, left: &Value, right: &Value) -> Result<Value, Error> {
        self.ordered_extreme("max", left, right)
    }

    pub fn compare(
        &mut self,
        comparison: Comparison,
        left: &Value,
        right: &Value,
    ) -> Result<Value, Error> {
        let (left, right) = self.broadcast_pair(left, right, "comparison")?;
        let is_float = left.value_type.element().is_float();
        let predicate = match (is_float, comparison) {
            (true, Comparison::Equal) => "oeq",
            (true, Comparison::Less) => "olt",
            (true, Comparison::LessEqual) => "ole",
            (true, Comparison::Greater) => "ogt",
            (true, Comparison::GreaterEqual) => "oge",
            (false, Comparison::Equal) => "eq",
            (false, Comparison::Less) if left.value_type.element().is_unsigned() => "ult",
            (false, Comparison::LessEqual) if left.value_type.element().is_unsigned() => "ule",
            (false, Comparison::Greater) if left.value_type.element().is_unsigned() => "ugt",
            (false, Comparison::GreaterEqual) if left.value_type.element().is_unsigned() => "uge",
            (false, Comparison::Less) => "slt",
            (false, Comparison::LessEqual) => "sle",
            (false, Comparison::Greater) => "sgt",
            (false, Comparison::GreaterEqual) => "sge",
        };
        let result = match &left.value_type {
            ValueType::Scalar(_) => ValueType::Scalar(DType::I1),
            ValueType::Tensor { shape, .. } => ValueType::Tensor {
                shape: shape.clone(),
                element: DType::I1,
                pointer_address_space: None,
            },
            ValueType::Pointer { .. } => unreachable!(),
        };
        let operation = if is_float { "cmpf" } else { "cmpi" };
        self.emit_value(
            result,
            format!(
                "arith.{operation} {predicate}, {}, {} : {}",
                left.id,
                right.id,
                left.value_type.spelling()
            ),
        )
    }

    pub fn select(
        &mut self,
        condition: &Value,
        when_true: &Value,
        when_false: &Value,
    ) -> Result<Value, Error> {
        self.require_values(&[condition, when_true, when_false])?;
        let (when_true, when_false) = self.broadcast_pair(when_true, when_false, "selection")?;
        let expected_condition = match &when_true.value_type {
            ValueType::Scalar(_) => ValueType::Scalar(DType::I1),
            ValueType::Tensor { shape, .. } => ValueType::Tensor {
                shape: shape.clone(),
                element: DType::I1,
                pointer_address_space: None,
            },
            ValueType::Pointer { .. } => {
                return Err(Error::TypeMismatch {
                    operation: "selection",
                });
            }
        };
        let condition = if condition.value_type == expected_condition {
            condition.clone()
        } else if condition.value_type == ValueType::Scalar(DType::I1) {
            let ValueType::Tensor { shape, .. } = &expected_condition else {
                return Err(Error::ExpectedCondition);
            };
            self.splat(condition, shape)?
        } else if let (
            ValueType::Tensor {
                shape: source,
                element: DType::I1,
                pointer_address_space: None,
            },
            ValueType::Tensor { shape: target, .. },
        ) = (&condition.value_type, &expected_condition)
        {
            if source.len() != target.len()
                || source
                    .iter()
                    .zip(target)
                    .any(|(source, target)| source != target && *source != 1)
            {
                return Err(Error::ExpectedCondition);
            }
            self.broadcast(condition, target)?
        } else {
            return Err(Error::ExpectedCondition);
        };
        self.emit_value(
            when_true.value_type.clone(),
            format!(
                "\"arith.select\"({}, {}, {}) : ({}, {}, {}) -> {}",
                condition.id,
                when_true.id,
                when_false.id,
                condition.value_type.spelling(),
                when_true.value_type.spelling(),
                when_true.value_type.spelling(),
                when_true.value_type.spelling()
            ),
        )
    }

    pub fn exp2(&mut self, value: &Value) -> Result<Value, Error> {
        self.float_unary("exp2", value)
    }

    pub fn log2(&mut self, value: &Value) -> Result<Value, Error> {
        self.float_unary("log2", value)
    }

    pub fn sqrt(&mut self, value: &Value) -> Result<Value, Error> {
        self.float_unary("sqrt", value)
    }

    pub fn reduce(
        &mut self,
        reduction: Reduction,
        value: &Value,
        axis: usize,
    ) -> Result<Value, Error> {
        self.require_open()?;
        self.require_values(&[value])?;
        let ValueType::Tensor {
            shape,
            element,
            pointer_address_space: None,
        } = &value.value_type
        else {
            return Err(Error::ExpectedTensor);
        };
        if axis >= shape.len() {
            return Err(Error::InvalidAxis {
                axis,
                rank: shape.len(),
            });
        }
        let mut result_shape = shape.clone();
        result_shape.remove(axis);
        let result_type = if result_shape.is_empty() {
            ValueType::Scalar(*element)
        } else {
            ValueType::Tensor {
                shape: result_shape,
                element: *element,
                pointer_address_space: None,
            }
        };
        let result = self.next_value(result_type.clone());
        let left = self.next_value(ValueType::Scalar(*element));
        let right = self.next_value(ValueType::Scalar(*element));
        let combined = self.next_value(ValueType::Scalar(*element));
        let operation = match (reduction, element.is_float()) {
            (Reduction::Sum, true) => "addf",
            (Reduction::Sum, false) => "addi",
            (Reduction::Maximum, true) => "maxnumf",
            (Reduction::Maximum, false) => "maxsi",
        };
        self.body.push(format!(
            "{} = \"tt.reduce\"({}) ({{\n^bb0({}: {}, {}: {}):\n  {} = arith.{operation} {}, {} : {}\n  \"tt.reduce.return\"({}) : ({}) -> ()\n}}) {{axis = {axis} : i32}} : ({}) -> {}",
            result.id,
            value.id,
            left.id,
            element.spelling(),
            right.id,
            element.spelling(),
            combined.id,
            left.id,
            right.id,
            element.spelling(),
            combined.id,
            element.spelling(),
            value.value_type.spelling(),
            result_type.spelling(),
        ));
        Ok(result)
    }

    pub fn dot(
        &mut self,
        left: &Value,
        right: &Value,
        accumulator: &Value,
    ) -> Result<Value, Error> {
        self.require_values(&[left, right, accumulator])?;
        let (
            ValueType::Tensor {
                shape: left_shape,
                element: left_element,
                pointer_address_space: None,
            },
            ValueType::Tensor {
                shape: right_shape,
                element: right_element,
                pointer_address_space: None,
            },
            ValueType::Tensor {
                shape: accumulator_shape,
                element: accumulator_element,
                pointer_address_space: None,
            },
        ) = (&left.value_type, &right.value_type, &accumulator.value_type)
        else {
            return Err(Error::ExpectedTensor);
        };
        if left_shape.len() != 2
            || right_shape.len() != 2
            || accumulator_shape.len() != 2
            || left_shape[1] != right_shape[0]
            || accumulator_shape != &[left_shape[0], right_shape[1]]
            || left_element != right_element
            || !left_element.is_float()
            || *accumulator_element != DType::F32
        {
            return Err(Error::TypeMismatch { operation: "dot" });
        }
        self.emit_value(
            accumulator.value_type.clone(),
            format!(
                "\"tt.dot\"({}, {}, {}) {{inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32}} : ({}, {}, {}) -> {}",
                left.id,
                right.id,
                accumulator.id,
                left.value_type.spelling(),
                right.value_type.spelling(),
                accumulator.value_type.spelling(),
                accumulator.value_type.spelling(),
            ),
        )
    }

    /// Constructs the pinned Triton microscaling dot operation.
    ///
    /// Operand and scale layout constraints remain the caller's kernel-level
    /// responsibility; `finish` reparses and verifies the complete TTIR module
    /// so invalid packed geometries cannot reach XLA embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn dot_scaled(
        &mut self,
        left: &Value,
        right: &Value,
        accumulator: &Value,
        left_scale: Option<&Value>,
        right_scale: Option<&Value>,
        left_element: ScaleDotElement,
        right_element: ScaleDotElement,
        lhs_k_pack: bool,
        rhs_k_pack: bool,
    ) -> Result<Value, Error> {
        let mut operands = vec![left, right, accumulator];
        operands.extend(left_scale);
        operands.extend(right_scale);
        self.require_values(&operands)?;
        let ValueType::Tensor {
            element: DType::F32,
            pointer_address_space: None,
            ..
        } = &accumulator.value_type
        else {
            return Err(Error::TypeMismatch {
                operation: "scaled dot accumulator",
            });
        };
        for (value, element) in [(left, left_element), (right, right_element)] {
            let packed = matches!(element, ScaleDotElement::E2M1);
            if !matches!(
                &value.value_type,
                ValueType::Tensor {
                    pointer_address_space: None,
                    ..
                }
            ) || (packed && !matches!(value.value_type.element(), DType::I8 | DType::U8))
            {
                return Err(Error::TypeMismatch {
                    operation: "scaled dot operand",
                });
            }
        }
        for scale in [left_scale, right_scale].into_iter().flatten() {
            if !matches!(
                &scale.value_type,
                ValueType::Tensor {
                    element: DType::I8 | DType::U8 | DType::F32,
                    pointer_address_space: None,
                    ..
                }
            ) {
                return Err(Error::TypeMismatch {
                    operation: "scaled dot scale",
                });
            }
        }
        let scaled_operand = |value: &Value, scale: Option<&Value>| match scale {
            Some(scale) => format!("{} scale {}", value.id, scale.id),
            None => value.id.clone(),
        };
        let scaled_type = |value: &Value, scale: Option<&Value>| match scale {
            Some(scale) => format!(
                "{}, {}",
                value.value_type.spelling(),
                scale.value_type.spelling()
            ),
            None => value.value_type.spelling(),
        };
        self.emit_value(
            accumulator.value_type.clone(),
            format!(
                "tt.dot_scaled {}, {}, {} lhs = {} rhs = {} {{fastMath = false, lhs_k_pack = {lhs_k_pack}, rhs_k_pack = {rhs_k_pack}}} : {} * {} -> {}",
                scaled_operand(left, left_scale),
                scaled_operand(right, right_scale),
                accumulator.id,
                left_element.spelling(),
                right_element.spelling(),
                scaled_type(left, left_scale),
                scaled_type(right, right_scale),
                accumulator.value_type.spelling(),
            ),
        )
    }

    pub fn for_loop<F>(
        &mut self,
        lower: &Value,
        upper: &Value,
        step: &Value,
        initial: &[Value],
        body: F,
    ) -> Result<Vec<Value>, Error>
    where
        F: FnOnce(&mut Self, Value, &[Value]) -> Result<Vec<Value>, Error>,
    {
        self.require_open()?;
        self.require_values(&[lower, upper, step])?;
        self.require_values(&initial.iter().collect::<Vec<_>>())?;
        if lower.value_type != upper.value_type
            || lower.value_type != step.value_type
            || !matches!(lower.value_type, ValueType::Scalar(dtype) if !dtype.is_float())
        {
            return Err(Error::TypeMismatch {
                operation: "for loop bounds",
            });
        }
        if initial.is_empty() {
            return Err(Error::EmptyRegionState {
                operation: "scf.for",
            });
        }
        let induction = self.next_value(lower.value_type.clone());
        let carried = initial
            .iter()
            .map(|value| self.next_value(value.value_type.clone()))
            .collect::<Vec<_>>();
        let outer = std::mem::take(&mut self.body);
        let yielded = body(self, induction.clone(), &carried);
        let region_body = std::mem::replace(&mut self.body, outer);
        let yielded = yielded?;
        self.require_values(&yielded.iter().collect::<Vec<_>>())?;
        if !same_types(&yielded, initial) {
            return Err(Error::InvalidRegionYield {
                operation: "scf.for",
            });
        }
        let results = self.next_results(
            initial
                .iter()
                .map(|value| value.value_type.clone())
                .collect(),
        );
        let region_arguments = std::iter::once(format!(
            "{}: {}",
            induction.id,
            induction.value_type.spelling()
        ))
        .chain(
            carried
                .iter()
                .map(|value| format!("{}: {}", value.id, value.value_type.spelling())),
        )
        .collect::<Vec<_>>()
        .join(", ");
        let result_types = type_list(initial);
        self.body.push(format!(
            "{} = \"scf.for\"({}, {}, {}, {}) ({{\n^bb0({region_arguments}):\n{}\n  \"scf.yield\"({}) : ({result_types}) -> ()\n}}) : ({}, {}, {}, {result_types}) -> ({result_types})",
            result_binding(&results),
            lower.id,
            upper.id,
            step.id,
            value_list(initial),
            indent_lines(&region_body, 2),
            value_list(&yielded),
            lower.value_type.spelling(),
            upper.value_type.spelling(),
            step.value_type.spelling(),
        ));
        Ok(results)
    }

    pub fn if_then_else<T, E>(
        &mut self,
        condition: &Value,
        then_body: T,
        else_body: E,
    ) -> Result<Vec<Value>, Error>
    where
        T: FnOnce(&mut Self) -> Result<Vec<Value>, Error>,
        E: FnOnce(&mut Self) -> Result<Vec<Value>, Error>,
    {
        self.require_open()?;
        self.require_values(&[condition])?;
        if condition.value_type != ValueType::Scalar(DType::I1) {
            return Err(Error::ExpectedCondition);
        }
        let outer = std::mem::take(&mut self.body);
        let then_values = then_body(self);
        let then_region = std::mem::take(&mut self.body);
        let else_values = else_body(self);
        let else_region = std::mem::replace(&mut self.body, outer);
        let then_values = then_values?;
        let else_values = else_values?;
        self.require_values(&then_values.iter().collect::<Vec<_>>())?;
        self.require_values(&else_values.iter().collect::<Vec<_>>())?;
        if then_values.is_empty() || !same_types(&then_values, &else_values) {
            return Err(Error::InvalidRegionYield {
                operation: "scf.if",
            });
        }
        let results = self.next_results(
            then_values
                .iter()
                .map(|value| value.value_type.clone())
                .collect(),
        );
        let result_types = type_list(&then_values);
        self.body.push(format!(
            "{} = \"scf.if\"({}) ({{\n^bb0:\n{}\n  \"scf.yield\"({}) : ({result_types}) -> ()\n}}, {{\n^bb0:\n{}\n  \"scf.yield\"({}) : ({result_types}) -> ()\n}}) : (i1) -> ({result_types})",
            result_binding(&results),
            condition.id,
            indent_lines(&then_region, 2),
            value_list(&then_values),
            indent_lines(&else_region, 2),
            value_list(&else_values),
        ));
        Ok(results)
    }

    pub fn if_only<F>(&mut self, condition: &Value, body: F) -> Result<(), Error>
    where
        F: FnOnce(&mut Self) -> Result<(), Error>,
    {
        self.require_open()?;
        self.require_values(&[condition])?;
        if condition.value_type != ValueType::Scalar(DType::I1) {
            return Err(Error::ExpectedCondition);
        }
        let outer = std::mem::take(&mut self.body);
        let body_result = body(self);
        let region = std::mem::replace(&mut self.body, outer);
        body_result?;
        self.body.push(format!(
            "\"scf.if\"({}) ({{\n^bb0:\n{}\n  \"scf.yield\"() : () -> ()\n}}, {{}}) : (i1) -> ()",
            condition.id,
            indent_lines(&region, 2),
        ));
        Ok(())
    }

    pub fn full_float(&mut self, shape: &[i64], value: f64, dtype: DType) -> Result<Value, Error> {
        let scalar = self.float(value, dtype)?;
        self.splat(&scalar, shape)
    }

    pub fn full_float_like(&mut self, value: &Value, fill: f64) -> Result<Value, Error> {
        self.require_values(&[value])?;
        let dtype = value.value_type.element();
        if !dtype.is_float() {
            return Err(Error::TypeMismatch {
                operation: "floating fill",
            });
        }
        let scalar = self.float(fill, dtype)?;
        match &value.value_type {
            ValueType::Scalar(_) => Ok(scalar),
            ValueType::Tensor {
                shape,
                pointer_address_space: None,
                ..
            } => self.splat(&scalar, shape),
            ValueType::Pointer { .. }
            | ValueType::Tensor {
                pointer_address_space: Some(_),
                ..
            } => Err(Error::TypeMismatch {
                operation: "floating fill",
            }),
        }
    }

    pub fn full_integer(
        &mut self,
        shape: &[i64],
        value: i64,
        dtype: DType,
    ) -> Result<Value, Error> {
        let scalar = self.integer(value, dtype)?;
        self.splat(&scalar, shape)
    }

    pub fn mask_2d(&mut self, rows: &Value, columns: &Value) -> Result<Value, Error> {
        let rows = self.expand_dimension(rows, 1)?;
        let columns = self.expand_dimension(columns, 0)?;
        self.bit_and(&rows, &columns)
    }

    pub fn while_loop<C, B>(
        &mut self,
        initial: &[Value],
        condition: C,
        body: B,
    ) -> Result<Vec<Value>, Error>
    where
        C: FnOnce(&mut Self, &[Value]) -> Result<(Value, Vec<Value>), Error>,
        B: FnOnce(&mut Self, &[Value]) -> Result<Vec<Value>, Error>,
    {
        self.require_open()?;
        self.require_values(&initial.iter().collect::<Vec<_>>())?;
        if initial.is_empty() {
            return Err(Error::EmptyRegionState {
                operation: "scf.while",
            });
        }
        let before_arguments = initial
            .iter()
            .map(|value| self.next_value(value.value_type.clone()))
            .collect::<Vec<_>>();
        let outer = std::mem::take(&mut self.body);
        let condition_result = condition(self, &before_arguments);
        let before_region = std::mem::take(&mut self.body);
        let (condition_value, forwarded) = match condition_result {
            Ok(result) => result,
            Err(error) => {
                self.body = outer;
                return Err(error);
            }
        };
        if self.require_values(&[&condition_value]).is_err()
            || self
                .require_values(&forwarded.iter().collect::<Vec<_>>())
                .is_err()
        {
            self.body = outer;
            return Err(Error::ForeignValue);
        }
        if condition_value.value_type != ValueType::Scalar(DType::I1)
            || !same_types(&forwarded, initial)
        {
            self.body = outer;
            return Err(Error::InvalidRegionYield {
                operation: "scf.while condition",
            });
        }
        let after_arguments = forwarded
            .iter()
            .map(|value| self.next_value(value.value_type.clone()))
            .collect::<Vec<_>>();
        let body_result = body(self, &after_arguments);
        let after_region = std::mem::replace(&mut self.body, outer);
        let yielded = body_result?;
        self.require_values(&yielded.iter().collect::<Vec<_>>())?;
        if !same_types(&yielded, initial) {
            return Err(Error::InvalidRegionYield {
                operation: "scf.while body",
            });
        }
        let results = self.next_results(
            initial
                .iter()
                .map(|value| value.value_type.clone())
                .collect(),
        );
        let argument_types = type_list(initial);
        self.body.push(format!(
            "{} = \"scf.while\"({}) ({{\n^bb0({}):\n{}\n  \"scf.condition\"({}, {}) : (i1, {argument_types}) -> ()\n}}, {{\n^bb0({}):\n{}\n  \"scf.yield\"({}) : ({argument_types}) -> ()\n}}) : ({argument_types}) -> ({argument_types})",
            result_binding(&results),
            value_list(initial),
            typed_arguments(&before_arguments),
            indent_lines(&before_region, 2),
            condition_value.id,
            value_list(&forwarded),
            typed_arguments(&after_arguments),
            indent_lines(&after_region, 2),
            value_list(&yielded),
        ));
        Ok(results)
    }

    pub fn add_pointer(&mut self, pointer: &Value, offset: &Value) -> Result<Value, Error> {
        self.require_values(&[pointer, offset])?;
        if matches!(pointer.value_type, ValueType::Pointer { .. }) {
            if let ValueType::Tensor { shape, .. } = &offset.value_type {
                let pointer = self.splat(pointer, shape)?;
                return self.add_pointer(&pointer, offset);
            }
        }
        if let ValueType::Tensor { shape, .. } = &pointer.value_type {
            if matches!(offset.value_type, ValueType::Scalar(_)) {
                let offset = self.splat(offset, shape)?;
                return self.add_pointer(pointer, &offset);
            }
        }
        let (pointer_shape, element, address_space) = match &pointer.value_type {
            ValueType::Pointer {
                element,
                address_space,
            } => (None, *element, *address_space),
            ValueType::Tensor {
                shape,
                element,
                pointer_address_space: Some(address_space),
            } => (Some(shape.clone()), *element, *address_space),
            _ => return Err(Error::ExpectedPointer),
        };
        let offset_shape = match &offset.value_type {
            ValueType::Scalar(dtype) if !dtype.is_float() => None,
            ValueType::Tensor {
                shape,
                element,
                pointer_address_space: None,
            } if !element.is_float() => Some(shape.clone()),
            _ => return Err(Error::ExpectedIntegerOffset),
        };
        let result = match (pointer_shape, offset_shape) {
            (None, None) => ValueType::Pointer {
                element,
                address_space,
            },
            (Some(shape), None) | (None, Some(shape)) => ValueType::Tensor {
                shape,
                element,
                pointer_address_space: Some(address_space),
            },
            (Some(pointer), Some(offset)) if pointer == offset => ValueType::Tensor {
                shape: pointer,
                element,
                pointer_address_space: Some(address_space),
            },
            _ => {
                return Err(Error::TypeMismatch {
                    operation: "add pointer",
                });
            }
        };
        let pointer_type = pointer.value_type.spelling();
        let offset_type = offset.value_type.spelling();
        self.emit_value(
            result,
            format!(
                "tt.addptr {}, {} : {pointer_type}, {offset_type}",
                pointer.id, offset.id
            ),
        )
    }

    pub fn load(&mut self, pointer: &Value) -> Result<Value, Error> {
        self.require_values(&[pointer])?;
        let result = pointer.value_type.loaded().ok_or(Error::ExpectedPointer)?;
        self.emit_value(
            result,
            format!("tt.load {} : {}", pointer.id, pointer.value_type.spelling()),
        )
    }

    pub fn load_masked(
        &mut self,
        pointer: &Value,
        mask: &Value,
        other: &Value,
    ) -> Result<Value, Error> {
        self.load_masked_with(pointer, mask, other, LoadPolicy::DEFAULT)
    }

    pub(crate) fn load_masked_with(
        &mut self,
        pointer: &Value,
        mask: &Value,
        other: &Value,
        policy: LoadPolicy,
    ) -> Result<Value, Error> {
        self.require_values(&[pointer, mask, other])?;
        let result = pointer.value_type.loaded().ok_or(Error::ExpectedPointer)?;
        if mask.value_type != result.condition_type() || other.value_type != result {
            return Err(Error::TypeMismatch {
                operation: "masked load",
            });
        }
        self.emit_value(
            result.clone(),
            format!(
                "\"tt.load\"({}, {}, {}) <{{cache = {} : i32, evict = {} : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1>}}> : ({}, {}, {}) -> {}",
                pointer.id,
                mask.id,
                other.id,
                policy.cache.dialect_value(),
                1,
                pointer.value_type.spelling(),
                mask.value_type.spelling(),
                other.value_type.spelling(),
                result.spelling(),
            ),
        )
    }

    pub fn store(&mut self, pointer: &Value, value: &Value) -> Result<(), Error> {
        self.require_open()?;
        self.require_values(&[pointer, value])?;
        let expected = pointer.value_type.loaded().ok_or(Error::ExpectedPointer)?;
        if value.value_type != expected {
            return Err(Error::TypeMismatch { operation: "store" });
        }
        self.body.push(format!(
            "tt.store {}, {} : {}",
            pointer.id,
            value.id,
            pointer.value_type.spelling()
        ));
        Ok(())
    }

    pub fn store_masked(
        &mut self,
        pointer: &Value,
        value: &Value,
        mask: &Value,
    ) -> Result<(), Error> {
        self.require_open()?;
        self.require_values(&[pointer, value, mask])?;
        let expected = pointer.value_type.loaded().ok_or(Error::ExpectedPointer)?;
        if value.value_type != expected || mask.value_type != expected.condition_type() {
            return Err(Error::TypeMismatch {
                operation: "masked store",
            });
        }
        self.body.push(format!(
            "\"tt.store\"({}, {}, {}) <{{cache = 1 : i32, evict = 1 : i32}}> : ({}, {}, {}) -> ()",
            pointer.id,
            value.id,
            mask.id,
            pointer.value_type.spelling(),
            value.value_type.spelling(),
            mask.value_type.spelling(),
        ));
        Ok(())
    }

    pub fn return_void(&mut self) -> Result<(), Error> {
        self.require_open()?;
        self.body.push("tt.return".to_owned());
        self.terminated = true;
        Ok(())
    }

    pub fn finish(self) -> Result<Kernel, Error> {
        if !self.terminated {
            return Err(Error::MissingTerminator);
        }
        let arguments = self
            .arguments
            .iter()
            .map(|argument| {
                let divisibility = argument.divisibility.map_or_else(String::new, |value| {
                    format!(" {{tt.divisibility = {value} : i32}}")
                });
                format!(
                    "{}: {}{divisibility}",
                    argument.value.id,
                    argument.value.value_type.spelling()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let body = self
            .body
            .iter()
            .map(|line| format!("    {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "module {{\n  tt.func public @{}({arguments}) {{\n{body}\n  }}\n}}\n",
            self.name
        );
        let context = Context::new_ttir();
        let module = context.parse_module(&source)?;
        module.verify()?;
        Ok(Kernel {
            name: self.name,
            ir: module.text(),
            arguments: self
                .arguments
                .into_iter()
                .map(|argument| KernelArgument {
                    name: argument.name,
                    kind: argument.kind,
                })
                .collect(),
        })
    }

    fn binary(
        &mut self,
        operation: &'static str,
        left: &Value,
        right: &Value,
    ) -> Result<Value, Error> {
        self.require_values(&[left, right])?;
        let (left, right) = self.broadcast_pair(left, right, operation)?;
        let suffix = if left.value_type.element().is_float() {
            'f'
        } else {
            'i'
        };
        self.emit_value(
            left.value_type.clone(),
            format!(
                "arith.{operation}{suffix} {}, {} : {}",
                left.id,
                right.id,
                left.value_type.spelling()
            ),
        )
    }

    fn ordered_extreme(
        &mut self,
        operation: &'static str,
        left: &Value,
        right: &Value,
    ) -> Result<Value, Error> {
        self.require_values(&[left, right])?;
        let (left, right) = self.broadcast_pair(left, right, operation)?;
        let suffix = if left.value_type.element().is_float() {
            "numf"
        } else if left.value_type.element().is_unsigned() {
            "ui"
        } else {
            "si"
        };
        self.emit_value(
            left.value_type.clone(),
            format!(
                "arith.{operation}{suffix} {}, {} : {}",
                left.id,
                right.id,
                left.value_type.spelling()
            ),
        )
    }

    fn float_unary(&mut self, operation: &'static str, value: &Value) -> Result<Value, Error> {
        self.require_values(&[value])?;
        if !value.value_type.element().is_float()
            || matches!(value.value_type, ValueType::Pointer { .. })
            || matches!(
                value.value_type,
                ValueType::Tensor {
                    pointer_address_space: Some(_),
                    ..
                }
            )
        {
            return Err(Error::TypeMismatch { operation });
        }
        self.emit_value(
            value.value_type.clone(),
            format!(
                "math.{operation} {} : {}",
                value.id,
                value.value_type.spelling()
            ),
        )
    }

    fn broadcast_pair(
        &mut self,
        left: &Value,
        right: &Value,
        operation: &'static str,
    ) -> Result<(Value, Value), Error> {
        self.require_values(&[left, right])?;
        if left.value_type.element() != right.value_type.element()
            || matches!(left.value_type, ValueType::Pointer { .. })
            || matches!(right.value_type, ValueType::Pointer { .. })
            || matches!(
                left.value_type,
                ValueType::Tensor {
                    pointer_address_space: Some(_),
                    ..
                }
            )
            || matches!(
                right.value_type,
                ValueType::Tensor {
                    pointer_address_space: Some(_),
                    ..
                }
            )
        {
            return Err(Error::TypeMismatch { operation });
        }
        let left_shape = match &left.value_type {
            ValueType::Tensor { shape, .. } => Some(shape.as_slice()),
            ValueType::Scalar(_) => None,
            ValueType::Pointer { .. } => unreachable!(),
        };
        let right_shape = match &right.value_type {
            ValueType::Tensor { shape, .. } => Some(shape.as_slice()),
            ValueType::Scalar(_) => None,
            ValueType::Pointer { .. } => unreachable!(),
        };
        let target = match (left_shape, right_shape) {
            (None, None) => return Ok((left.clone(), right.clone())),
            (Some(shape), None) | (None, Some(shape)) => shape.to_vec(),
            (Some(left), Some(right)) if left.len() == right.len() => left
                .iter()
                .zip(right)
                .map(|(left, right)| match (*left, *right) {
                    (left, right) if left == right => Ok(left),
                    (1, right) => Ok(right),
                    (left, 1) => Ok(left),
                    _ => Err(Error::TypeMismatch { operation }),
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err(Error::TypeMismatch { operation }),
        };
        let left = match left_shape {
            None => self.splat(left, &target)?,
            Some(shape) if shape != target => self.broadcast(left, &target)?,
            Some(_) => left.clone(),
        };
        let right = match right_shape {
            None => self.splat(right, &target)?,
            Some(shape) if shape != target => self.broadcast(right, &target)?,
            Some(_) => right.clone(),
        };
        Ok((left, right))
    }

    fn emit_value(&mut self, value_type: ValueType, operation: String) -> Result<Value, Error> {
        self.require_open()?;
        let value = self.next_value(value_type);
        self.body.push(format!("{} = {operation}", value.id));
        Ok(value)
    }

    fn next_value(&mut self, value_type: ValueType) -> Value {
        let value = Value {
            owner: self.owner,
            id: format!("%{}", self.next_value),
            value_type,
        };
        self.next_value += 1;
        value
    }

    fn next_results(&mut self, value_types: Vec<ValueType>) -> Vec<Value> {
        if value_types.is_empty() {
            return Vec::new();
        }
        let base = self.next_value;
        self.next_value += 1;
        if value_types.len() == 1 {
            return vec![Value {
                owner: self.owner,
                id: format!("%{base}"),
                value_type: value_types.into_iter().next().unwrap(),
            }];
        }
        value_types
            .into_iter()
            .enumerate()
            .map(|(index, value_type)| Value {
                owner: self.owner,
                id: format!("%{base}#{index}"),
                value_type,
            })
            .collect()
    }

    fn require_open(&self) -> Result<(), Error> {
        if self.terminated {
            Err(Error::AlreadyTerminated)
        } else {
            Ok(())
        }
    }

    fn require_values(&self, values: &[&Value]) -> Result<(), Error> {
        if values.iter().any(|value| value.owner != self.owner) {
            Err(Error::ForeignValue)
        } else {
            Ok(())
        }
    }
}

fn require_identifier(name: &str) -> Result<(), Error> {
    let mut characters = name.chars();
    let valid_first = characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
    if !valid_first
        || !characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        Err(Error::InvalidName(name.to_owned()))
    } else {
        Ok(())
    }
}

fn validate_shape(shape: &[i64]) -> Result<(), Error> {
    if shape.is_empty() || shape.len() > 8 || shape.iter().any(|dimension| *dimension <= 0) {
        Err(Error::InvalidTensorShape(shape.to_vec()))
    } else {
        Ok(())
    }
}

fn element_count(shape: &[i64]) -> Option<i64> {
    shape
        .iter()
        .try_fold(1_i64, |product, dimension| product.checked_mul(*dimension))
}

fn same_types(left: &[Value], right: &[Value]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.value_type == right.value_type)
}

fn value_list(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| value.id.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn type_list(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| value.value_type.spelling())
        .collect::<Vec<_>>()
        .join(", ")
}

fn typed_arguments(values: &[Value]) -> String {
    values
        .iter()
        .map(|value| format!("{}: {}", value.id, value.value_type.spelling()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn result_binding(results: &[Value]) -> String {
    match results {
        [] => String::new(),
        [result] => result.id.clone(),
        [first, rest @ ..] => {
            let base = first.id.split('#').next().unwrap_or(&first.id);
            format!("{base}:{}", rest.len() + 1)
        }
    }
}

fn indent_lines(lines: &[String], spaces: usize) -> String {
    let indentation = " ".repeat(spaces);
    lines
        .iter()
        .flat_map(|line| line.lines())
        .map(|line| format!("{indentation}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn require_unique_names(names: &[&str]) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for &name in names {
        require_identifier(name)?;
        if !seen.insert(name) {
            return Err(Error::DuplicateArgument(name.to_owned()));
        }
    }
    Ok(())
}
