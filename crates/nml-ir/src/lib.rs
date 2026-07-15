//! Typed, deterministic construction of NML's StableHLO program subset.
//!
//! Validation happens while operations are authored. Consequently an invalid
//! graph never becomes an MLIR module and cannot reach XLA or a PJRT plugin.

#![forbid(unsafe_code)]

mod attention_backend;
mod moe_backend;
mod ordinary_attention;
mod paged_attention;

use nml_mlir::{
    Attribute as MlirAttribute, Block, Context, ConvolutionDimensionNumbers, ConvolutionWindow,
    Error as MlirError, Module, Operation as MlirOperation, Region, ShardyDimension,
    StableHloBinary, StableHloComparison, StableHloComparisonType, StableHloFftType,
    StableHloUnary, Type as MlirType, Value as MlirValue,
};
use nml_sharding::Sharding;
use nml_tensor::{Element, Slice};
use nml_types::{
    AxisTag, BFloat16, Complex128, Complex64, DType, DTypeClass, Layout, Partition, Shape,
    ShapeError, F16,
};
use std::collections::{HashMap, HashSet};
use std::error::Error as StdError;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PROGRAM_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tensor {
    program: u64,
    value: usize,
    shape: Shape,
}

impl Tensor {
    pub const fn shape(self) -> Shape {
        self.shape
    }
}

/// Linear graph-authoring handle for an explicit StableHLO random state.
/// It is intentionally not `Copy` or `Clone`: ordinary code must thread the
/// returned state instead of accidentally replaying a consumed one.
#[derive(Debug)]
pub struct RandomState {
    tensor: Tensor,
}

#[derive(Clone, Copy, Debug)]
struct MoeAssignmentPlan {
    sorted_assignments: Tensor,
    block_experts: Tensor,
    block_size: usize,
}

impl RandomState {
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOptions {
    pub causal: bool,
    pub sliding_window: Option<usize>,
    pub scale: Option<f64>,
}

/// Explicit tensor-axis and window contract for a 1D or 2D convolution.
/// Backend precision and StableHLO attribute types remain private.
#[derive(Clone, Copy, Debug)]
pub struct ConvolutionOptions<'a> {
    pub strides: &'a [i64],
    pub padding: &'a [[i64; 2]],
    pub input_dilation: &'a [i64],
    pub kernel_dilation: &'a [i64],
    pub kernel_reversal: &'a [bool],
    pub input_batch_axis: usize,
    pub input_feature_axis: usize,
    pub input_spatial_axes: &'a [usize],
    pub kernel_input_feature_axis: usize,
    pub kernel_output_feature_axis: usize,
    pub kernel_spatial_axes: &'a [usize],
    pub output_batch_axis: usize,
    pub output_feature_axis: usize,
    pub output_spatial_axes: &'a [usize],
    pub feature_groups: i64,
    pub batch_groups: i64,
}

impl Default for AttentionOptions {
    fn default() -> Self {
        Self {
            causal: true,
            sliding_window: None,
            scale: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeLayout {
    Interleaved,
    Sequential,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeScaling {
    Default,
    Linear {
        factor: f64,
    },
    Proportional {
        rotary_fraction: f64,
    },
    Llama3 {
        factor: f64,
        high_frequency_factor: f64,
        low_frequency_factor: f64,
        original_context: usize,
        truncate: bool,
    },
    Yarn {
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        original_context: usize,
        truncate: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeOptions {
    pub base: f64,
    pub rotary_dimensions: usize,
    pub layout: RopeLayout,
    pub scaling: RopeScaling,
}

impl RopeOptions {
    pub const fn new(rotary_dimensions: usize) -> Self {
        Self {
            base: 10_000.0,
            rotary_dimensions,
            layout: RopeLayout::Interleaved,
            scaling: RopeScaling::Default,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    ForeignTensor,
    DTypeMismatch {
        left: DType,
        right: DType,
    },
    RankMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    AxisOutOfBounds {
        side: &'static str,
        axis: usize,
        rank: usize,
    },
    DuplicateAxis {
        side: &'static str,
        axis: usize,
    },
    AxisCountMismatch,
    DimensionMismatch {
        left_axis: usize,
        right_axis: usize,
        left: i64,
        right: i64,
    },
    ShapeMismatch {
        operation: &'static str,
        left: Vec<i64>,
        right: Vec<i64>,
    },
    MetadataMismatch {
        operation: &'static str,
        field: &'static str,
    },
    UnsupportedComplexInput(DType),
    ExpectedComplex(DType),
    InvalidFft {
        kind: FftType,
        message: &'static str,
    },
    UnsupportedLayout {
        actual: Layout,
        expected: Layout,
    },
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },
    ElementCountMismatch {
        operation: &'static str,
        input: usize,
        output: usize,
    },
    EmptyOperands(&'static str),
    InvalidSlice {
        axis: usize,
        start: i64,
        limit: i64,
        stride: i64,
        dimension: i64,
    },
    InvalidIndexDType(DType),
    InvalidAttention(&'static str),
    InvalidRope(&'static str),
    InvalidNormalization(&'static str),
    InvalidReduction(&'static str),
    InvalidStructure(&'static str),
    InvalidLinearAlgebra(&'static str),
    InvalidIndexing(&'static str),
    InvalidWindow(&'static str),
    InvalidConvolution(&'static str),
    InvalidResize(&'static str),
    InvalidSort(&'static str),
    InvalidRandom(&'static str),
    InvalidSampling(&'static str),
    InvalidCollective(&'static str),
    InvalidMoe(&'static str),
    InvalidStateSpace(&'static str),
    InvalidBitcast(&'static str),
    InvalidPrecision {
        exponent_bits: i32,
        mantissa_bits: i32,
    },
    UpdateDimensionTooLarge {
        axis: usize,
        input: i64,
        update: i64,
    },
    Tensor(nml_tensor::Error),
    Sharding(nml_sharding::Error),
    NoOutputs,
    DuplicateInputName(String),
    DuplicateOutputName(String),
    AliasInputIsNotAnActivation(String),
    AliasShapeMismatch {
        input: Shape,
        output: Shape,
    },
    DuplicateOutputAlias,
    DuplicateInputAlias,
    Shape(ShapeError),
    Mlir(MlirError),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignTensor => formatter.write_str("tensor belongs to a different program"),
            Self::DTypeMismatch { left, right } => {
                write!(formatter, "dtype mismatch: {left:?} versus {right:?}")
            }
            Self::RankMismatch {
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "{operation} expects rank {expected}, received rank {actual}"
            ),
            Self::AxisOutOfBounds { side, axis, rank } => {
                write!(formatter, "{side} axis {axis} is outside rank {rank}")
            }
            Self::DuplicateAxis { side, axis } => {
                write!(formatter, "{side} axis {axis} is repeated")
            }
            Self::AxisCountMismatch => {
                formatter.write_str("left and right axis lists have different lengths")
            }
            Self::DimensionMismatch {
                left_axis,
                right_axis,
                left,
                right,
            } => write!(
                formatter,
                "dimension mismatch: left axis {left_axis} is {left}, right axis {right_axis} is {right}"
            ),
            Self::ShapeMismatch {
                operation,
                left,
                right,
            } => write!(
                formatter,
                "{operation} requires equal logical shapes, received {left:?} and {right:?}"
            ),
            Self::MetadataMismatch { operation, field } => {
                write!(formatter, "{operation} requires matching {field}")
            }
            Self::UnsupportedComplexInput(dtype) => {
                write!(
                    formatter,
                    "complex construction requires F32 or F64, received {dtype:?}"
                )
            }
            Self::ExpectedComplex(dtype) => {
                write!(
                    formatter,
                    "operation requires C64 or C128, received {dtype:?}"
                )
            }
            Self::InvalidFft { kind, message } => {
                write!(formatter, "{kind:?} is invalid: {message}")
            }
            Self::UnsupportedLayout { actual, expected } => write!(
                formatter,
                "physical layout {:?} is not supported; expected row-major {:?}",
                actual.minor_to_major(),
                expected.minor_to_major()
            ),
            Self::UnsupportedDType { operation, dtype } => {
                write!(formatter, "{operation} does not support {dtype:?}")
            }
            Self::ElementCountMismatch {
                operation,
                input,
                output,
            } => write!(
                formatter,
                "{operation} requires equal element counts, received {input} and {output}",
            ),
            Self::EmptyOperands(operation) => {
                write!(formatter, "{operation} requires at least one operand")
            }
            Self::InvalidSlice {
                axis,
                start,
                limit,
                stride,
                dimension,
            } => write!(
                formatter,
                "slice axis {axis} has invalid [{start}, {limit}) stride {stride} for dimension {dimension}",
            ),
            Self::InvalidIndexDType(dtype) => {
                write!(
                    formatter,
                    "index tensor must contain integers, received {dtype:?}"
                )
            }
            Self::InvalidAttention(message) => write!(formatter, "invalid attention: {message}"),
            Self::InvalidRope(message) => write!(formatter, "invalid RoPE: {message}"),
            Self::InvalidNormalization(message) => {
                write!(formatter, "invalid normalization: {message}")
            }
            Self::InvalidReduction(message) => write!(formatter, "invalid reduction: {message}"),
            Self::InvalidStructure(message) => write!(formatter, "invalid structure: {message}"),
            Self::InvalidLinearAlgebra(message) => {
                write!(formatter, "invalid linear algebra: {message}")
            }
            Self::InvalidIndexing(message) => write!(formatter, "invalid indexing: {message}"),
            Self::InvalidWindow(message) => write!(formatter, "invalid window: {message}"),
            Self::InvalidConvolution(message) => {
                write!(formatter, "invalid convolution: {message}")
            }
            Self::InvalidResize(message) => write!(formatter, "invalid resize: {message}"),
            Self::InvalidSort(message) => write!(formatter, "invalid sort: {message}"),
            Self::InvalidRandom(message) => write!(formatter, "invalid random state: {message}"),
            Self::InvalidSampling(message) => write!(formatter, "invalid sampling: {message}"),
            Self::InvalidCollective(message) => write!(formatter, "invalid collective: {message}"),
            Self::InvalidMoe(message) => write!(formatter, "invalid MoE: {message}"),
            Self::InvalidStateSpace(message) => {
                write!(formatter, "invalid state-space operation: {message}")
            }
            Self::InvalidBitcast(message) => write!(formatter, "invalid bitcast: {message}"),
            Self::InvalidPrecision {
                exponent_bits,
                mantissa_bits,
            } => write!(
                formatter,
                "invalid reduced precision: exponent bits must be positive and mantissa bits must be nonnegative, received ({exponent_bits}, {mantissa_bits})",
            ),
            Self::UpdateDimensionTooLarge {
                axis,
                input,
                update,
            } => write!(
                formatter,
                "dynamic update dimension {axis} is {update}, exceeding input dimension {input}",
            ),
            Self::Tensor(error) => error.fmt(formatter),
            Self::Sharding(error) => error.fmt(formatter),
            Self::NoOutputs => formatter.write_str("program must expose at least one output"),
            Self::DuplicateInputName(name) => write!(formatter, "duplicate input name {name:?}"),
            Self::DuplicateOutputName(name) => write!(formatter, "duplicate output name {name:?}"),
            Self::AliasInputIsNotAnActivation(name) => write!(
                formatter,
                "parameter {name:?} cannot donate storage; only uniquely owned activations may be aliased",
            ),
            Self::AliasShapeMismatch { input, output } => write!(
                formatter,
                "output alias shape mismatch: input {input:?}, output {output:?}",
            ),
            Self::DuplicateOutputAlias => {
                formatter.write_str("an output may alias only one executable input")
            }
            Self::DuplicateInputAlias => {
                formatter.write_str("an executable input may donate storage to only one output")
            }
            Self::Shape(error) => error.fmt(formatter),
            Self::Mlir(error) => error.fmt(formatter),
        }
    }
}

impl StdError for Error {}

impl From<ShapeError> for Error {
    fn from(error: ShapeError) -> Self {
        Self::Shape(error)
    }
}

impl From<MlirError> for Error {
    fn from(error: MlirError) -> Self {
        Self::Mlir(error)
    }
}

impl From<nml_tensor::Error> for Error {
    fn from(error: nml_tensor::Error) -> Self {
        Self::Tensor(error)
    }
}

impl From<nml_sharding::Error> for Error {
    fn from(error: nml_sharding::Error) -> Self {
        Self::Sharding(error)
    }
}

pub struct ProgramBuilder {
    identifier: u64,
    inputs: Vec<usize>,
    input_kinds: Vec<InputKind>,
    values: Vec<Value>,
    operations: Vec<Operation>,
    aliases: HashMap<usize, usize>,
    consumed_random_states: HashSet<usize>,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            identifier: NEXT_PROGRAM_ID.fetch_add(1, Ordering::Relaxed),
            inputs: Vec::new(),
            input_kinds: Vec::new(),
            values: Vec::new(),
            operations: Vec::new(),
            aliases: HashMap::new(),
            consumed_random_states: HashSet::new(),
        }
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        self.named_input(name, shape, InputKind::Activation)
    }

    pub fn parameter(&mut self, name: impl Into<String>, shape: Shape) -> Tensor {
        self.named_input(name, shape, InputKind::Parameter)
    }

    fn named_input(&mut self, name: impl Into<String>, shape: Shape, kind: InputKind) -> Tensor {
        let value = self.values.len();
        self.values.push(Value {
            name: name.into(),
            shape,
        });
        self.inputs.push(value);
        self.input_kinds.push(kind);
        self.tensor(value)
    }

    pub fn matmul(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.require_rank(left, "matmul left operand", 2)?;
        self.require_rank(right, "matmul right operand", 2)?;
        self.dot_general(left, right, &[], &[], &[1], &[0])
    }

    /// Computes a batched Cholesky factorization of positive-definite matrices.
    pub fn cholesky(&mut self, input: Tensor, lower: bool) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if input.shape.rank() < 2 {
            return Err(Error::InvalidLinearAlgebra(
                "cholesky requires at least two matrix dimensions",
            ));
        }
        if !matches!(
            input.shape.dtype().class(),
            DTypeClass::Float | DTypeClass::Complex
        ) {
            return Err(Error::UnsupportedDType {
                operation: "cholesky",
                dtype: input.shape.dtype(),
            });
        }
        let rank = input.shape.rank();
        if input.shape.dimensions()[rank - 2] != input.shape.dimensions()[rank - 1] {
            return Err(Error::InvalidLinearAlgebra(
                "cholesky requires square trailing matrix dimensions",
            ));
        }
        let result = self.push_value("cholesky", input.shape);
        self.operations.push(Operation::Cholesky {
            input: input.value,
            result: result.value,
            lower,
        });
        Ok(result)
    }

    /// Solves `coefficient * result = right_hand_side` for a triangular,
    /// non-unit coefficient matrix without transposing it.
    pub fn triangular_solve(
        &mut self,
        coefficient: Tensor,
        right_hand_side: Tensor,
        lower: bool,
    ) -> Result<Tensor, Error> {
        self.require_local(coefficient)?;
        self.require_local(right_hand_side)?;
        if coefficient.shape.dtype() != right_hand_side.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: coefficient.shape.dtype(),
                right: right_hand_side.shape.dtype(),
            });
        }
        if !matches!(
            coefficient.shape.dtype().class(),
            DTypeClass::Float | DTypeClass::Complex
        ) {
            return Err(Error::UnsupportedDType {
                operation: "triangular_solve",
                dtype: coefficient.shape.dtype(),
            });
        }
        if coefficient.shape.rank() < 2 || coefficient.shape.rank() != right_hand_side.shape.rank()
        {
            return Err(Error::InvalidLinearAlgebra(
                "triangular solve requires equal ranks with at least two dimensions",
            ));
        }
        let rank = coefficient.shape.rank();
        let matrix_size = coefficient.shape.dimensions()[rank - 1];
        if coefficient.shape.dimensions()[rank - 2] != matrix_size {
            return Err(Error::InvalidLinearAlgebra(
                "triangular coefficient matrix must be square",
            ));
        }
        if right_hand_side.shape.dimensions()[rank - 2] != matrix_size {
            return Err(Error::InvalidLinearAlgebra(
                "right-hand-side row dimension must match the coefficient matrix",
            ));
        }
        if coefficient.shape.dimensions()[..rank - 2]
            != right_hand_side.shape.dimensions()[..rank - 2]
        {
            return Err(Error::InvalidLinearAlgebra(
                "triangular solve batch dimensions must match",
            ));
        }
        if coefficient.shape.axis_tags()[..rank - 2]
            != right_hand_side.shape.axis_tags()[..rank - 2]
            || coefficient.shape.partitions()[..rank - 2]
                != right_hand_side.shape.partitions()[..rank - 2]
        {
            return Err(Error::MetadataMismatch {
                operation: "triangular_solve",
                field: "batch axis metadata",
            });
        }
        let result = self.push_value("triangular_solve", right_hand_side.shape);
        self.operations.push(Operation::TriangularSolve {
            coefficient: coefficient.value,
            right_hand_side: right_hand_side.value,
            result: result.value,
            lower,
        });
        Ok(result)
    }

    pub fn add(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Add)
    }

    pub fn constant(&mut self, value: &Slice<'_>) -> Result<Tensor, Error> {
        let value = value.to_contiguous()?;
        let shape = value.shape();
        require_supported_layout(shape)?;
        let literal = dense_literal(&value)?;
        let result = self.push_value("constant", shape);
        self.operations.push(Operation::Constant {
            result: result.value,
            literal,
        });
        Ok(result)
    }

    pub fn scalar<T: Element>(&mut self, value: T) -> Result<Tensor, Error> {
        let shape = Shape::new(T::DTYPE, &[])?;
        self.constant(&Slice::from_typed(shape, std::slice::from_ref(&value))?)
    }

    pub fn subtract(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Subtract)
    }

    pub fn multiply(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Multiply)
    }

    pub fn divide(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Divide)
    }

    pub fn power(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Power)
    }

    pub fn remainder(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Remainder)
    }

    pub fn minimum(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Minimum)
    }

    pub fn maximum(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Maximum)
    }

    /// Applies elementwise AND to boolean predicates or integer bit patterns.
    pub fn logical_and(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::And)
    }

    /// Applies elementwise OR to boolean predicates or integer bit patterns.
    pub fn logical_or(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Or)
    }

    /// Applies elementwise XOR to boolean predicates or integer bit patterns.
    pub fn logical_xor(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Xor)
    }

    /// Inverts each boolean predicate or integer bit pattern.
    pub fn logical_not(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_logical(input, "logical_not")?;
        self.unary(input, Unary::Not)
    }

    pub fn shift_left(&mut self, input: Tensor, amount: Tensor) -> Result<Tensor, Error> {
        self.binary(input, amount, Binary::ShiftLeft)
    }

    pub fn shift_right_arithmetic(
        &mut self,
        input: Tensor,
        amount: Tensor,
    ) -> Result<Tensor, Error> {
        self.binary(input, amount, Binary::ShiftRightArithmetic)
    }

    pub fn shift_right_logical(&mut self, input: Tensor, amount: Tensor) -> Result<Tensor, Error> {
        self.binary(input, amount, Binary::ShiftRightLogical)
    }

    /// Reinterprets element bits without performing a numerical conversion.
    ///
    /// StableHLO represents a width-changing bitcast by adding or removing the
    /// most-minor logical dimension. NML only removes an anonymous,
    /// non-sharded dimension, so bit reinterpretation cannot silently erase a
    /// model axis or a partition boundary.
    pub fn bitcast(&mut self, input: Tensor, dtype: DType) -> Result<Tensor, Error> {
        self.require_local(input)?;
        require_supported_layout(input.shape)?;
        if input.shape.dtype() == dtype {
            return Ok(input);
        }

        let input_bits = input.shape.dtype().bit_width();
        let output_bits = dtype.bit_width();
        let shape = if input_bits == output_bits {
            input.shape.with_dtype(dtype)
        } else if input_bits > output_bits {
            if input_bits % output_bits != 0 {
                return Err(Error::InvalidBitcast("element widths are not divisible"));
            }
            let mut dimensions = input.shape.dimensions().to_vec();
            dimensions.push((input_bits / output_bits) as i64);
            let mut tags = input.shape.axis_tags().to_vec();
            tags.push(AxisTag::UNKNOWN);
            let mut partitions = input.shape.partitions().to_vec();
            partitions.push(Partition::Unspecified);
            Shape::new(dtype, &dimensions)?
                .with_axis_tags(&tags)?
                .with_partitions(&partitions)?
        } else {
            if output_bits % input_bits != 0 {
                return Err(Error::InvalidBitcast("element widths are not divisible"));
            }
            let ratio = (output_bits / input_bits) as i64;
            let Some((&minor_dimension, dimensions)) = input.shape.dimensions().split_last() else {
                return Err(Error::InvalidBitcast(
                    "widening requires a trailing bit-pack dimension",
                ));
            };
            if minor_dimension != ratio {
                return Err(Error::InvalidBitcast(
                    "trailing dimension does not match the element-width ratio",
                ));
            }
            let minor_axis = input.shape.rank() - 1;
            if input.shape.axis_tags()[minor_axis] != AxisTag::UNKNOWN {
                return Err(Error::InvalidBitcast(
                    "widening cannot remove a tagged model axis",
                ));
            }
            if matches!(input.shape.partitions()[minor_axis], Partition::Sharded(_)) {
                return Err(Error::InvalidBitcast(
                    "widening cannot remove a sharded axis",
                ));
            }
            Shape::new(dtype, dimensions)?
                .with_axis_tags(&input.shape.axis_tags()[..minor_axis])?
                .with_partitions(&input.shape.partitions()[..minor_axis])?
        };

        let result = self.push_value("bitcast", shape);
        self.operations.push(Operation::Bitcast {
            input: input.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn count_leading_zeros(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.integer_unary(input, Unary::CountLeadingZeros)
    }

    pub fn population_count(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.integer_unary(input, Unary::PopulationCount)
    }

    pub fn is_finite(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_float(input, "is_finite")?;
        let result = self.push_value("is_finite", input.shape.with_dtype(DType::Bool));
        self.operations.push(Operation::Unary {
            input: input.value,
            result: result.value,
            operation: Unary::IsFinite,
        });
        Ok(result)
    }

    pub fn sign(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_local(input)?;
        match input.shape.dtype().class() {
            DTypeClass::SignedInteger | DTypeClass::Float | DTypeClass::Complex => {
                self.unary(input, Unary::Sign)
            }
            DTypeClass::Boolean | DTypeClass::UnsignedInteger => Err(Error::UnsupportedDType {
                operation: "sign",
                dtype: input.shape.dtype(),
            }),
        }
    }

    pub fn expm1(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Expm1)
    }

    pub fn round_nearest_away_from_zero(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::RoundNearestAwayFromZero)
    }

    pub fn round_nearest_even(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::RoundNearestEven)
    }

    pub fn reduce_precision(
        &mut self,
        input: Tensor,
        exponent_bits: i32,
        mantissa_bits: i32,
    ) -> Result<Tensor, Error> {
        self.require_float(input, "reduce_precision")?;
        if exponent_bits < 1 || mantissa_bits < 0 {
            return Err(Error::InvalidPrecision {
                exponent_bits,
                mantissa_bits,
            });
        }
        let result = self.push_value("reduce_precision", input.shape);
        self.operations.push(Operation::ReducePrecision {
            input: input.value,
            result: result.value,
            exponent_bits,
            mantissa_bits,
        });
        Ok(result)
    }

    pub fn clamp(
        &mut self,
        input: Tensor,
        minimum: Tensor,
        maximum: Tensor,
    ) -> Result<Tensor, Error> {
        let (input, minimum, shape) = self.elementwise_operands("clamp", input, minimum)?;
        let (input, maximum, result_shape) = self.elementwise_operands("clamp", input, maximum)?;
        require_matching_shape_metadata("clamp", shape, result_shape)?;
        if !result_shape.dtype().supports_ordering() || result_shape.dtype() == DType::Bool {
            return Err(Error::UnsupportedDType {
                operation: "clamp",
                dtype: result_shape.dtype(),
            });
        }
        let result = self.push_value("clamp", result_shape);
        self.operations.push(Operation::Clamp {
            minimum: minimum.value,
            input: input.value,
            maximum: maximum.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn negate(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_local(input)?;
        match input.shape.dtype().class() {
            DTypeClass::SignedInteger | DTypeClass::Float | DTypeClass::Complex => {}
            _ => {
                return Err(Error::UnsupportedDType {
                    operation: "negate",
                    dtype: input.shape.dtype(),
                });
            }
        }
        self.unary(input, Unary::Negate)
    }

    pub fn abs(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_local(input)?;
        let dtype = match input.shape.dtype().class() {
            DTypeClass::SignedInteger | DTypeClass::Float => input.shape.dtype(),
            DTypeClass::Complex => match input.shape.dtype() {
                DType::C64 => DType::F32,
                DType::C128 => DType::F64,
                _ => unreachable!("complex class contains only C64 and C128"),
            },
            DTypeClass::Boolean | DTypeClass::UnsignedInteger => {
                return Err(Error::UnsupportedDType {
                    operation: "abs",
                    dtype: input.shape.dtype(),
                });
            }
        };
        let result = self.push_value("abs", input.shape.with_dtype(dtype));
        self.operations.push(Operation::Unary {
            input: input.value,
            result: result.value,
            operation: Unary::Abs,
        });
        Ok(result)
    }

    pub fn equal(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.compare(left, right, Comparison::Eq)
    }

    pub fn not_equal(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.compare(left, right, Comparison::Ne)
    }

    pub fn greater(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.compare(left, right, Comparison::Gt)
    }

    pub fn greater_equal(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.compare(left, right, Comparison::Ge)
    }

    pub fn less(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.compare(left, right, Comparison::Lt)
    }

    pub fn less_equal(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.compare(left, right, Comparison::Le)
    }

    pub fn select(
        &mut self,
        predicate: Tensor,
        on_true: Tensor,
        on_false: Tensor,
    ) -> Result<Tensor, Error> {
        self.require_local(predicate)?;
        if predicate.shape.dtype() != DType::Bool {
            return Err(Error::UnsupportedDType {
                operation: "select predicate",
                dtype: predicate.shape.dtype(),
            });
        }
        let (on_true, on_false, shape) = self.elementwise_operands("select", on_true, on_false)?;
        let predicate = if predicate.shape.rank() == 0 && shape.rank() != 0 {
            self.broadcast_in_dim(predicate, shape.with_dtype(DType::Bool), &[])?
        } else {
            require_matching_shape_metadata(
                "select predicate",
                predicate.shape,
                shape.with_dtype(DType::Bool),
            )?;
            predicate
        };
        let result = self.push_value("select", shape);
        self.operations.push(Operation::Select {
            predicate: predicate.value,
            on_true: on_true.value,
            on_false: on_false.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn convert(&mut self, input: Tensor, dtype: DType) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if input.shape.dtype() == dtype {
            return Ok(input);
        }
        if matches!(input.shape.dtype(), DType::C64 | DType::C128)
            || matches!(dtype, DType::C64 | DType::C128)
        {
            return Err(Error::UnsupportedDType {
                operation: "convert",
                dtype,
            });
        }
        let result = self.push_value("convert", input.shape.with_dtype(dtype));
        self.operations.push(Operation::Convert {
            input: input.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn reshape(&mut self, input: Tensor, shape: Shape) -> Result<Tensor, Error> {
        self.require_local(input)?;
        require_supported_layout(shape)?;
        if input.shape.dtype() != shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: shape.dtype(),
            });
        }
        let input_elements = input.shape.element_count()?;
        let output_elements = shape.element_count()?;
        if input_elements != output_elements {
            return Err(Error::ElementCountMismatch {
                operation: "reshape",
                input: input_elements,
                output: output_elements,
            });
        }
        let result = self.push_value("reshape", shape);
        self.operations.push(Operation::Reshape {
            input: input.value,
            result: result.value,
        });
        let explicit_partition_change = shape
            .partitions()
            .iter()
            .any(|partition| *partition != Partition::Unspecified)
            && input.shape.partitions() != shape.partitions();
        if !explicit_partition_change {
            Ok(result)
        } else {
            let constrained = self.push_value("sharding_constraint", shape);
            self.operations.push(Operation::ShardingConstraint {
                input: result.value,
                result: constrained.value,
            });
            Ok(constrained)
        }
    }

    pub fn transpose(&mut self, input: Tensor, permutation: &[usize]) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(permutation, input.shape.rank(), "transpose")?;
        if permutation.len() != input.shape.rank() {
            return Err(Error::AxisCountMismatch);
        }
        let dimensions = permutation
            .iter()
            .map(|axis| input.shape.dimensions()[*axis])
            .collect::<Vec<_>>();
        let tags = permutation
            .iter()
            .map(|axis| input.shape.axis_tags()[*axis])
            .collect::<Vec<_>>();
        let partitions = permutation
            .iter()
            .map(|axis| input.shape.partitions()[*axis])
            .collect::<Vec<_>>();
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(&tags)?
            .with_partitions(&partitions)?;
        let result = self.push_value("transpose", shape);
        self.operations.push(Operation::Transpose {
            input: input.value,
            result: result.value,
            permutation: permutation.to_vec(),
        });
        Ok(result)
    }

    pub fn with_partitions(
        &mut self,
        input: Tensor,
        partitions: &[Partition],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        let shape = input.shape.with_partitions(partitions)?;
        let result = self.push_value("sharding_constraint", shape);
        self.operations.push(Operation::ShardingConstraint {
            input: input.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn exp(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Exp)
    }

    pub fn log(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Log)
    }

    pub fn sqrt(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Sqrt)
    }

    pub fn rsqrt(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Rsqrt)
    }

    pub fn tanh(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Tanh)
    }

    pub fn sin(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Sin)
    }

    pub fn cos(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Cos)
    }

    pub fn floor(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Floor)
    }

    pub fn ceil(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Ceil)
    }

    pub fn sigmoid(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.float_unary(input, Unary::Logistic)
    }

    pub fn relu(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_float(input, "relu")?;
        let zero = self.scalar_for(input.shape.dtype(), 0.0)?;
        self.maximum(input, zero)
    }

    pub fn silu(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_float(input, "silu")?;
        let sigmoid = self.sigmoid(input)?;
        self.multiply(input, sigmoid)
    }

    pub fn gelu(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_float(input, "gelu")?;
        let half = self.scalar_for(input.shape.dtype(), 0.5)?;
        let one = self.scalar_for(input.shape.dtype(), 1.0)?;
        let coefficient = self.scalar_for(input.shape.dtype(), 0.044715)?;
        let scale = self.scalar_for(input.shape.dtype(), 0.7978845608028654)?;
        let square = self.multiply(input, input)?;
        let cube = self.multiply(square, input)?;
        let correction = self.multiply(coefficient, cube)?;
        let inner = self.add(input, correction)?;
        let inner = self.multiply(scale, inner)?;
        let inner = self.tanh(inner)?;
        let inner = self.add(one, inner)?;
        let inner = self.multiply(input, inner)?;
        self.multiply(half, inner)
    }

    pub fn moe_swiglu(
        &mut self,
        hidden: Tensor,
        router_logits: Tensor,
        gate_up_weights: Tensor,
        down_weights: Tensor,
        experts_per_token: usize,
    ) -> Result<Tensor, Error> {
        self.moe_gated(
            hidden,
            router_logits,
            gate_up_weights,
            down_weights,
            experts_per_token,
            MoeActivation::Silu,
        )
    }

    pub fn moe_geglu(
        &mut self,
        hidden: Tensor,
        router_logits: Tensor,
        gate_up_weights: Tensor,
        down_weights: Tensor,
        experts_per_token: usize,
    ) -> Result<Tensor, Error> {
        self.moe_gated(
            hidden,
            router_logits,
            gate_up_weights,
            down_weights,
            experts_per_token,
            MoeActivation::Gelu,
        )
    }

    pub fn moe_reglu(
        &mut self,
        hidden: Tensor,
        router_logits: Tensor,
        gate_up_weights: Tensor,
        down_weights: Tensor,
        experts_per_token: usize,
    ) -> Result<Tensor, Error> {
        self.moe_gated(
            hidden,
            router_logits,
            gate_up_weights,
            down_weights,
            experts_per_token,
            MoeActivation::Relu,
        )
    }

    /// Executes one Gated DeltaNet recurrent update.
    ///
    /// Shapes are `state [heads, value, key]`, `query/key [heads, key]`,
    /// `value [heads, value]`, and `alpha/beta [heads]`. The returned pair is
    /// `(output [heads, value], next_state)` so no model-specific public state
    /// wrapper is needed.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net_step(
        &mut self,
        state: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        for tensor in [state, query, key, value, alpha, beta] {
            self.require_local(tensor)?;
            self.require_float(tensor, "gated_delta_net_step")?;
            if tensor.shape.dtype() != state.shape.dtype() {
                return Err(Error::InvalidStateSpace(
                    "state and step inputs must use one dtype",
                ));
            }
        }
        if state.shape.rank() != 3
            || query.shape.rank() != 2
            || key.shape.rank() != 2
            || value.shape.rank() != 2
            || alpha.shape.rank() != 1
            || beta.shape.rank() != 1
        {
            return Err(Error::InvalidStateSpace(
                "expected state [heads, value, key], query/key [heads, key], value [heads, value], and gates [heads]",
            ));
        }
        let [heads, value_size, key_size] = state.shape.dimensions() else {
            unreachable!("state rank was checked")
        };
        if query.shape.dimensions() != &[*heads, *key_size]
            || key.shape.dimensions() != &[*heads, *key_size]
            || value.shape.dimensions() != &[*heads, *value_size]
            || alpha.shape.dimensions() != &[*heads]
            || beta.shape.dimensions() != &[*heads]
        {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet head, key, and value dimensions are inconsistent",
            ));
        }
        let head_tag = state.shape.axis_tags()[0];
        let head_partition = state.shape.partitions()[0];
        if [query, key, value, alpha, beta].iter().any(|tensor| {
            tensor.shape.axis_tags()[0] != head_tag
                || tensor.shape.partitions()[0] != head_partition
        }) {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet head-axis metadata must agree",
            ));
        }
        if [query, key].iter().any(|tensor| {
            tensor.shape.axis_tags()[1] != state.shape.axis_tags()[2]
                || tensor.shape.partitions()[1] != state.shape.partitions()[2]
        }) || value.shape.axis_tags()[1] != state.shape.axis_tags()[1]
            || value.shape.partitions()[1] != state.shape.partitions()[1]
        {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet key/value-axis metadata must agree with the recurrent state",
            ));
        }

        let v_hat = self.dot_general(state, key, &[0], &[0], &[2], &[1])?;
        let alpha_values = self.broadcast_in_dim(alpha, v_hat.shape, &[0])?;
        let v_hat = self.multiply(v_hat, alpha_values)?;
        let delta = self.subtract(value, v_hat)?;
        let beta_values = self.broadcast_in_dim(beta, delta.shape, &[0])?;
        let delta = self.multiply(delta, beta_values)?;

        let delta = self.insert_axis(delta, 2, state.shape.axis_tags()[2])?;
        let delta = self.broadcast_in_dim(delta, state.shape, &[0, 1, 2])?;
        let key = self.insert_axis(key, 1, state.shape.axis_tags()[1])?;
        let key = self.broadcast_in_dim(key, state.shape, &[0, 1, 2])?;
        let correction = self.multiply(delta, key)?;
        let alpha_state = self.broadcast_in_dim(alpha, state.shape, &[0])?;
        let retained = self.multiply(state, alpha_state)?;
        let next_state = self.add(retained, correction)?;
        let output = self.dot_general(next_state, query, &[0], &[0], &[2], &[1])?;
        Ok((output, next_state))
    }

    /// Processes a statically shaped sequence with the same recurrence as
    /// [`Self::gated_delta_net_step`]. The public API remains a typed neural
    /// composite; lowering owns the loop privately so graph and compile size do
    /// not grow with the sequence length.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_net(
        &mut self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        alphas: Tensor,
        betas: Tensor,
        initial_state: Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        for tensor in [queries, keys, values, alphas, betas, initial_state] {
            self.require_local(tensor)?;
            self.require_float(tensor, "gated_delta_net")?;
            if tensor.shape.dtype() != initial_state.shape.dtype() {
                return Err(Error::InvalidStateSpace(
                    "state and sequence inputs must use one dtype",
                ));
            }
        }
        if queries.shape.rank() != 3
            || keys.shape.rank() != 3
            || values.shape.rank() != 3
            || alphas.shape.rank() != 2
            || betas.shape.rank() != 2
            || initial_state.shape.rank() != 3
        {
            return Err(Error::InvalidStateSpace(
                "expected sequence query/key/value rank 3, gates rank 2, and state rank 3",
            ));
        }
        let sequence = queries.shape.dimensions()[0];
        if sequence > i64::from(i32::MAX) {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet sequence length must fit its I32 loop index",
            ));
        }
        let heads = initial_state.shape.dimensions()[0];
        let value_size = initial_state.shape.dimensions()[1];
        let key_size = initial_state.shape.dimensions()[2];
        if queries.shape.dimensions() != &[sequence, heads, key_size]
            || keys.shape.dimensions() != &[sequence, heads, key_size]
            || values.shape.dimensions() != &[sequence, heads, value_size]
            || alphas.shape.dimensions() != &[sequence, heads]
            || betas.shape.dimensions() != &[sequence, heads]
        {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet sequence, head, key, and value dimensions are inconsistent",
            ));
        }
        let sequence_tag = queries.shape.axis_tags()[0];
        let sequence_partition = queries.shape.partitions()[0];
        if [keys, values, alphas, betas].iter().any(|tensor| {
            tensor.shape.axis_tags()[0] != sequence_tag
                || tensor.shape.partitions()[0] != sequence_partition
        }) {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet sequence-axis metadata must agree",
            ));
        }
        let head_tag = initial_state.shape.axis_tags()[0];
        let head_partition = initial_state.shape.partitions()[0];
        if [queries, keys, values, alphas, betas].iter().any(|tensor| {
            tensor.shape.axis_tags()[1] != head_tag
                || tensor.shape.partitions()[1] != head_partition
        }) {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet head-axis metadata must agree",
            ));
        }
        if [queries, keys].iter().any(|tensor| {
            tensor.shape.axis_tags()[2] != initial_state.shape.axis_tags()[2]
                || tensor.shape.partitions()[2] != initial_state.shape.partitions()[2]
        }) || values.shape.axis_tags()[2] != initial_state.shape.axis_tags()[1]
            || values.shape.partitions()[2] != initial_state.shape.partitions()[1]
        {
            return Err(Error::InvalidStateSpace(
                "Gated DeltaNet key/value-axis metadata must agree with the recurrent state",
            ));
        }
        let outputs = self.push_value("gated_delta_net_outputs", values.shape);
        let final_state = self.push_value("gated_delta_net_state", initial_state.shape);
        self.operations.push(Operation::GatedDeltaNet {
            queries: queries.value,
            keys: keys.value,
            values: values.value,
            alphas: alphas.value,
            betas: betas.value,
            initial_state: initial_state.value,
            outputs: outputs.value,
            final_state: final_state.value,
        });
        Ok((outputs, final_state))
    }

    pub fn leaky_relu(&mut self, input: Tensor, slope: f64) -> Result<Tensor, Error> {
        self.require_float(input, "leaky_relu")?;
        let zero = self.scalar_for(input.shape.dtype(), 0.0)?;
        let slope = self.scalar_for(input.shape.dtype(), slope)?;
        let negative = self.minimum(input, zero)?;
        let negative = self.multiply(slope, negative)?;
        let positive = self.maximum(input, zero)?;
        self.add(positive, negative)
    }

    pub fn quick_gelu(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_float(input, "quick_gelu")?;
        let scale = self.scalar_for(input.shape.dtype(), 1.702)?;
        let scaled = self.multiply(scale, input)?;
        let gate = self.sigmoid(scaled)?;
        self.multiply(input, gate)
    }

    pub fn swiglu(&mut self, gate: Tensor, value: Tensor) -> Result<Tensor, Error> {
        let gate = self.silu(gate)?;
        self.multiply(gate, value)
    }

    pub fn geglu(&mut self, gate: Tensor, value: Tensor) -> Result<Tensor, Error> {
        let gate = self.gelu(gate)?;
        self.multiply(gate, value)
    }

    pub fn broadcast_in_dim(
        &mut self,
        input: Tensor,
        result_shape: Shape,
        dimensions: &[usize],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        require_supported_layout(result_shape)?;
        if input.shape.dtype() != result_shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: result_shape.dtype(),
            });
        }
        if dimensions.len() != input.shape.rank() {
            return Err(Error::AxisCountMismatch);
        }
        validate_axes(dimensions, result_shape.rank(), "broadcast result")?;
        for (input_axis, &result_axis) in dimensions.iter().enumerate() {
            let input_dim = input.shape.dimensions()[input_axis];
            let result_dim = result_shape.dimensions()[result_axis];
            if input_dim != 1 && input_dim != result_dim {
                return Err(Error::DimensionMismatch {
                    left_axis: input_axis,
                    right_axis: result_axis,
                    left: input_dim,
                    right: result_dim,
                });
            }
        }
        let result = self.push_value("broadcast", result_shape);
        self.operations.push(Operation::BroadcastInDim {
            input: input.value,
            result: result.value,
            dimensions: dimensions.to_vec(),
        });
        Ok(result)
    }

    pub fn iota(&mut self, shape: Shape, axis: usize) -> Result<Tensor, Error> {
        require_supported_layout(shape)?;
        if axis >= shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "iota",
                axis,
                rank: shape.rank(),
            });
        }
        match shape.dtype().class() {
            DTypeClass::SignedInteger | DTypeClass::UnsignedInteger | DTypeClass::Float => {}
            _ => {
                return Err(Error::UnsupportedDType {
                    operation: "iota",
                    dtype: shape.dtype(),
                });
            }
        }
        let result = self.push_value("iota", shape);
        self.operations.push(Operation::Iota {
            result: result.value,
            axis,
        });
        Ok(result)
    }

    pub fn concatenate(&mut self, inputs: &[Tensor], axis: usize) -> Result<Tensor, Error> {
        let Some(first) = inputs.first().copied() else {
            return Err(Error::EmptyOperands("concatenate"));
        };
        self.require_local(first)?;
        if axis >= first.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "concatenate",
                axis,
                rank: first.shape.rank(),
            });
        }
        let mut dimensions = first.shape.dimensions().to_vec();
        for input in &inputs[1..] {
            self.require_local(*input)?;
            if input.shape.dtype() != first.shape.dtype() {
                return Err(Error::DTypeMismatch {
                    left: first.shape.dtype(),
                    right: input.shape.dtype(),
                });
            }
            if input.shape.rank() != first.shape.rank() {
                return Err(Error::RankMismatch {
                    operation: "concatenate",
                    expected: first.shape.rank(),
                    actual: input.shape.rank(),
                });
            }
            for dimension_axis in 0..first.shape.rank() {
                if dimension_axis != axis
                    && input.shape.dimensions()[dimension_axis]
                        != first.shape.dimensions()[dimension_axis]
                {
                    return Err(Error::DimensionMismatch {
                        left_axis: dimension_axis,
                        right_axis: dimension_axis,
                        left: first.shape.dimensions()[dimension_axis],
                        right: input.shape.dimensions()[dimension_axis],
                    });
                }
            }
            if input.shape.axis_tags() != first.shape.axis_tags() {
                return Err(Error::MetadataMismatch {
                    operation: "concatenate",
                    field: "axis tags",
                });
            }
            if input.shape.partitions() != first.shape.partitions() {
                return Err(Error::MetadataMismatch {
                    operation: "concatenate",
                    field: "partition metadata",
                });
            }
            dimensions[axis] = dimensions[axis]
                .checked_add(input.shape.dimensions()[axis])
                .ok_or(ShapeError::ElementCountOverflow { axis })?;
        }
        let shape = Shape::new(first.shape.dtype(), &dimensions)?
            .with_axis_tags(first.shape.axis_tags())?
            .with_partitions(first.shape.partitions())?;
        let result = self.push_value("concatenate", shape);
        self.operations.push(Operation::Concatenate {
            inputs: inputs.iter().map(|input| input.value).collect(),
            result: result.value,
            axis,
        });
        Ok(result)
    }

    pub fn pad(
        &mut self,
        input: Tensor,
        padding_value: Tensor,
        edge_low: &[i64],
        edge_high: &[i64],
        interior: &[i64],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(padding_value)?;
        let rank = input.shape.rank();
        if edge_low.len() != rank || edge_high.len() != rank || interior.len() != rank {
            return Err(Error::InvalidStructure(
                "pad configuration must contain one entry per input axis",
            ));
        }
        if padding_value.shape.rank() != 0 {
            return Err(Error::RankMismatch {
                operation: "pad value",
                expected: 0,
                actual: padding_value.shape.rank(),
            });
        }
        if padding_value.shape.dtype() != input.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: padding_value.shape.dtype(),
            });
        }

        let dimensions = input
            .shape
            .dimensions()
            .iter()
            .zip(edge_low)
            .zip(edge_high)
            .zip(interior)
            .map(|(((dimension, low), high), interior)| {
                if *interior < 0 {
                    return Err(Error::InvalidStructure(
                        "interior padding must be nonnegative",
                    ));
                }
                let gaps = dimension.saturating_sub(1);
                dimension
                    .checked_add(*low)
                    .and_then(|value| value.checked_add(*high))
                    .and_then(|value| {
                        gaps.checked_mul(*interior)
                            .and_then(|gap| value.checked_add(gap))
                    })
                    .filter(|value| *value >= 0)
                    .ok_or(Error::InvalidStructure(
                        "padding dimensions overflow or crop past an empty result",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(input.shape.axis_tags())?
            .with_partitions(input.shape.partitions())?;
        let result = self.push_value("pad", shape);
        self.operations.push(Operation::Pad {
            input: input.value,
            padding_value: padding_value.value,
            result: result.value,
            edge_low: edge_low.to_vec(),
            edge_high: edge_high.to_vec(),
            interior: interior.to_vec(),
        });
        Ok(result)
    }

    pub fn reverse(&mut self, input: Tensor, axes: &[usize]) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(axes, input.shape.rank(), "reverse")?;
        if axes.is_empty() {
            return Ok(input);
        }
        let result = self.push_value("reverse", input.shape);
        self.operations.push(Operation::Reverse {
            input: input.value,
            result: result.value,
            axes: axes.to_vec(),
        });
        Ok(result)
    }

    pub fn insert_axis(
        &mut self,
        input: Tensor,
        axis: usize,
        tag: AxisTag,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if axis > input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "insert_axis",
                axis,
                rank: input.shape.rank() + 1,
            });
        }
        let shape = inserted_axis_shape(input.shape, axis, 1, tag)?;
        self.reshape(input, shape)
    }

    pub fn squeeze(&mut self, input: Tensor, axis: usize) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "squeeze",
                axis,
                rank: input.shape.rank(),
            });
        }
        if input.shape.dimensions()[axis] != 1 {
            return Err(Error::InvalidStructure("squeeze requires a unit dimension"));
        }
        if matches!(input.shape.partitions()[axis], Partition::Sharded(_)) {
            return Err(Error::InvalidStructure(
                "squeeze cannot remove a sharded dimension",
            ));
        }
        let mut dimensions = input.shape.dimensions().to_vec();
        dimensions.remove(axis);
        let mut tags = input.shape.axis_tags().to_vec();
        tags.remove(axis);
        let mut partitions = input.shape.partitions().to_vec();
        partitions.remove(axis);
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(&tags)?
            .with_partitions(&partitions)?;
        self.reshape(input, shape)
    }

    pub fn stack(&mut self, inputs: &[Tensor], axis: usize, tag: AxisTag) -> Result<Tensor, Error> {
        let Some(&first) = inputs.first() else {
            return Err(Error::EmptyOperands("stack"));
        };
        let mut expanded = Vec::with_capacity(inputs.len());
        for &input in inputs {
            require_matching_shape_metadata("stack", first.shape, input.shape)?;
            expanded.push(self.insert_axis(input, axis, tag)?);
        }
        self.concatenate(&expanded, axis)
    }

    pub fn repeat(&mut self, input: Tensor, axis: usize, count: usize) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "repeat",
                axis,
                rank: input.shape.rank(),
            });
        }
        if count == 0 {
            return Err(Error::InvalidStructure("repeat count must be positive"));
        }
        if count == 1 {
            return Ok(input);
        }
        let count = i64::try_from(count)
            .map_err(|_| Error::InvalidStructure("repeat count exceeds i64"))?;
        let expanded_shape = inserted_axis_shape(input.shape, axis, 1, AxisTag::UNKNOWN)?;
        let mut broad_dimensions = expanded_shape.dimensions().to_vec();
        broad_dimensions[axis] = count;
        let broad_shape = Shape::new(input.shape.dtype(), &broad_dimensions)?
            .with_axis_tags(expanded_shape.axis_tags())?
            .with_partitions(expanded_shape.partitions())?;
        let mapping = (0..input.shape.rank())
            .map(|old_axis| {
                if old_axis < axis {
                    old_axis
                } else {
                    old_axis + 1
                }
            })
            .collect::<Vec<_>>();
        let broad = self.broadcast_in_dim(input, broad_shape, &mapping)?;
        let mut dimensions = input.shape.dimensions().to_vec();
        dimensions[axis] = dimensions[axis]
            .checked_mul(count)
            .ok_or(Error::InvalidStructure("repeat dimension overflows"))?;
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(input.shape.axis_tags())?
            .with_partitions(input.shape.partitions())?;
        self.reshape(broad, shape)
    }

    pub fn stutter(&mut self, input: Tensor, axis: usize, count: usize) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "stutter",
                axis,
                rank: input.shape.rank(),
            });
        }
        if count == 0 {
            return Err(Error::InvalidStructure("stutter count must be positive"));
        }
        if count == 1 {
            return Ok(input);
        }
        let count = i64::try_from(count)
            .map_err(|_| Error::InvalidStructure("stutter count exceeds i64"))?;
        let expanded_shape = inserted_axis_shape(input.shape, axis + 1, 1, AxisTag::UNKNOWN)?;
        let mut broad_dimensions = expanded_shape.dimensions().to_vec();
        broad_dimensions[axis + 1] = count;
        let broad_shape = Shape::new(input.shape.dtype(), &broad_dimensions)?
            .with_axis_tags(expanded_shape.axis_tags())?
            .with_partitions(expanded_shape.partitions())?;
        let mapping = (0..input.shape.rank())
            .map(|old_axis| {
                if old_axis <= axis {
                    old_axis
                } else {
                    old_axis + 1
                }
            })
            .collect::<Vec<_>>();
        let broad = self.broadcast_in_dim(input, broad_shape, &mapping)?;
        let mut dimensions = input.shape.dimensions().to_vec();
        dimensions[axis] = dimensions[axis]
            .checked_mul(count)
            .ok_or(Error::InvalidStructure("stutter dimension overflows"))?;
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(input.shape.axis_tags())?
            .with_partitions(input.shape.partitions())?;
        self.reshape(broad, shape)
    }

    pub fn split(
        &mut self,
        input: Tensor,
        axis: usize,
        sizes: &[i64],
    ) -> Result<Vec<Tensor>, Error> {
        self.require_local(input)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "split",
                axis,
                rank: input.shape.rank(),
            });
        }
        if sizes.is_empty() || sizes.iter().any(|size| *size < 0) {
            return Err(Error::InvalidStructure(
                "split sizes must be a nonempty list of nonnegative dimensions",
            ));
        }
        let total = sizes.iter().try_fold(0i64, |total, size| {
            total
                .checked_add(*size)
                .ok_or(Error::InvalidStructure("split dimensions overflow"))
        })?;
        if total != input.shape.dimensions()[axis] {
            return Err(Error::InvalidStructure(
                "split sizes must exactly cover the selected dimension",
            ));
        }
        let mut start = 0i64;
        sizes
            .iter()
            .map(|size| {
                let mut starts = vec![0; input.shape.rank()];
                let mut limits = input.shape.dimensions().to_vec();
                let strides = vec![1; input.shape.rank()];
                starts[axis] = start;
                start += *size;
                limits[axis] = start;
                self.slice(input, &starts, &limits, &strides)
            })
            .collect()
    }

    pub fn chunks(
        &mut self,
        input: Tensor,
        axis: usize,
        count: usize,
    ) -> Result<Vec<Tensor>, Error> {
        self.require_local(input)?;
        if count == 0 {
            return Err(Error::InvalidStructure("chunk count must be positive"));
        }
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "chunks",
                axis,
                rank: input.shape.rank(),
            });
        }
        let dimension = input.shape.dimensions()[axis];
        let count_i64 =
            i64::try_from(count).map_err(|_| Error::InvalidStructure("chunk count exceeds i64"))?;
        if dimension % count_i64 != 0 {
            return Err(Error::InvalidStructure(
                "exact chunks require a dimension divisible by the chunk count",
            ));
        }
        self.split(input, axis, &vec![dimension / count_i64; count])
    }

    pub fn outer(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.dot_general(left, right, &[], &[], &[], &[])
    }

    pub fn diagonal(
        &mut self,
        input: Tensor,
        axis: usize,
        row_tag: AxisTag,
        column_tag: AxisTag,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "diagonal",
                axis,
                rank: input.shape.rank(),
            });
        }
        if matches!(input.shape.partitions()[axis], Partition::Sharded(_)) {
            return Err(Error::InvalidStructure(
                "diagonal expansion requires an unsharded source axis",
            ));
        }
        if row_tag == column_tag && row_tag != AxisTag::UNKNOWN {
            return Err(Error::InvalidStructure(
                "diagonal row and column axes require distinct tags",
            ));
        }

        let dimension = input.shape.dimensions()[axis];
        let mut dimensions = input.shape.dimensions().to_vec();
        dimensions[axis] = dimension;
        dimensions.insert(axis + 1, dimension);
        let mut tags = input.shape.axis_tags().to_vec();
        tags[axis] = row_tag;
        tags.insert(axis + 1, column_tag);
        let mut partitions = input.shape.partitions().to_vec();
        partitions[axis] = Partition::Unspecified;
        partitions.insert(axis + 1, Partition::Unspecified);
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(&tags)?
            .with_partitions(&partitions)?;
        let mapping = (0..input.shape.rank())
            .map(|old_axis| {
                if old_axis <= axis {
                    old_axis
                } else {
                    old_axis + 1
                }
            })
            .collect::<Vec<_>>();
        let values = self.broadcast_in_dim(input, shape, &mapping)?;
        let index_shape = shape.with_dtype(DType::I64);
        let rows = self.iota(index_shape, axis)?;
        let columns = self.iota(index_shape, axis + 1)?;
        let predicate = self.equal(rows, columns)?;
        let zero = self.zero_for(input.shape.dtype())?;
        let zeros = self.broadcast_in_dim(zero, shape, &[])?;
        self.select(predicate, values, zeros)
    }

    pub fn triangular(
        &mut self,
        input: Tensor,
        row_axis: usize,
        column_axis: usize,
        upper_diagonals: i64,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(&[row_axis, column_axis], input.shape.rank(), "triangular")?;
        let index_shape = input.shape.with_dtype(DType::I64);
        let rows = self.iota(index_shape, row_axis)?;
        let columns = self.iota(index_shape, column_axis)?;
        let offset = self.scalar(upper_diagonals)?;
        let rows = self.add(rows, offset)?;
        let predicate = self.greater_equal(rows, columns)?;
        let zero = self.zero_for(input.shape.dtype())?;
        let zeros = self.broadcast_in_dim(zero, input.shape, &[])?;
        self.select(predicate, input, zeros)
    }

    pub fn cartesian_product(&mut self, vectors: &[Tensor]) -> Result<Vec<Tensor>, Error> {
        let Some(&first) = vectors.first() else {
            return Err(Error::EmptyOperands("cartesian_product"));
        };
        self.require_local(first)?;
        let mut dimensions = Vec::with_capacity(vectors.len());
        let mut tags = Vec::with_capacity(vectors.len());
        let mut partitions = Vec::with_capacity(vectors.len());
        for &vector in vectors {
            self.require_local(vector)?;
            if vector.shape.dtype() != first.shape.dtype() {
                return Err(Error::DTypeMismatch {
                    left: first.shape.dtype(),
                    right: vector.shape.dtype(),
                });
            }
            if vector.shape.rank() > 1 {
                return Err(Error::RankMismatch {
                    operation: "cartesian_product vector",
                    expected: 1,
                    actual: vector.shape.rank(),
                });
            }
            dimensions.push(vector.shape.dimensions().first().copied().unwrap_or(1));
            tags.push(
                vector
                    .shape
                    .axis_tags()
                    .first()
                    .copied()
                    .unwrap_or(AxisTag::UNKNOWN),
            );
            partitions.push(
                vector
                    .shape
                    .partitions()
                    .first()
                    .copied()
                    .unwrap_or(Partition::Unspecified),
            );
        }
        let shape = Shape::new(first.shape.dtype(), &dimensions)?
            .with_axis_tags(&tags)?
            .with_partitions(&partitions)?;
        vectors
            .iter()
            .enumerate()
            .map(|(axis, vector)| {
                let mapping = (vector.shape.rank() == 1).then_some(axis);
                self.broadcast_in_dim(*vector, shape, mapping.as_slice())
            })
            .collect()
    }

    pub fn cartesian_product_stacked(
        &mut self,
        vectors: &[Tensor],
        coordinate_tag: AxisTag,
    ) -> Result<Tensor, Error> {
        let product = self.cartesian_product(vectors)?;
        let axis = product
            .first()
            .ok_or(Error::EmptyOperands("cartesian_product_stacked"))?
            .shape
            .rank();
        self.stack(&product, axis, coordinate_tag)
    }

    pub fn roll(&mut self, input: Tensor, axis: usize, shift: i64) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "roll",
                axis,
                rank: input.shape.rank(),
            });
        }
        let dimension = input.shape.dimensions()[axis];
        if dimension == 0 {
            return Ok(input);
        }
        let shift = shift.rem_euclid(dimension);
        if shift == 0 {
            return Ok(input);
        }
        let mut starts = vec![0; input.shape.rank()];
        let mut limits = input.shape.dimensions().to_vec();
        let strides = vec![1; input.shape.rank()];
        starts[axis] = dimension - shift;
        let tail = self.slice(input, &starts, &limits, &strides)?;
        starts[axis] = 0;
        limits[axis] = dimension - shift;
        let head = self.slice(input, &starts, &limits, &strides)?;
        self.concatenate(&[tail, head], axis)
    }

    pub fn optimization_barrier(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.require_local(input)?;
        let result = self.push_value("optimization_barrier", input.shape);
        self.operations.push(Operation::OptimizationBarrier {
            input: input.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn slice(
        &mut self,
        input: Tensor,
        starts: &[i64],
        limits: &[i64],
        strides: &[i64],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if starts.len() != input.shape.rank()
            || limits.len() != input.shape.rank()
            || strides.len() != input.shape.rank()
        {
            return Err(Error::AxisCountMismatch);
        }
        let mut dimensions = Vec::with_capacity(input.shape.rank());
        for axis in 0..input.shape.rank() {
            let (start, limit, stride, dimension) = (
                starts[axis],
                limits[axis],
                strides[axis],
                input.shape.dimensions()[axis],
            );
            if start < 0 || limit < start || limit > dimension || stride <= 0 {
                return Err(Error::InvalidSlice {
                    axis,
                    start,
                    limit,
                    stride,
                    dimension,
                });
            }
            dimensions.push((limit - start + stride - 1) / stride);
        }
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(input.shape.axis_tags())?
            .with_partitions(input.shape.partitions())?;
        let result = self.push_value("slice", shape);
        self.operations.push(Operation::Slice {
            input: input.value,
            result: result.value,
            starts: starts.to_vec(),
            limits: limits.to_vec(),
            strides: strides.to_vec(),
        });
        Ok(result)
    }

    pub fn dynamic_slice(
        &mut self,
        input: Tensor,
        starts: &[Tensor],
        sizes: &[i64],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.validate_dynamic_starts(starts, input.shape.rank())?;
        if sizes.len() != input.shape.rank() {
            return Err(Error::AxisCountMismatch);
        }
        for (axis, (&size, &dimension)) in sizes.iter().zip(input.shape.dimensions()).enumerate() {
            if size < 0 || size > dimension {
                return Err(Error::InvalidSlice {
                    axis,
                    start: 0,
                    limit: size,
                    stride: 1,
                    dimension,
                });
            }
        }
        let shape = Shape::new(input.shape.dtype(), sizes)?
            .with_axis_tags(input.shape.axis_tags())?
            .with_partitions(input.shape.partitions())?;
        let result = self.push_value("dynamic_slice", shape);
        self.operations.push(Operation::DynamicSlice {
            input: input.value,
            starts: starts.iter().map(|start| start.value).collect(),
            result: result.value,
            sizes: sizes.to_vec(),
        });
        Ok(result)
    }

    pub fn dynamic_update_slice(
        &mut self,
        input: Tensor,
        update: Tensor,
        starts: &[Tensor],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(update)?;
        self.validate_dynamic_starts(starts, input.shape.rank())?;
        if input.shape.dtype() != update.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: update.shape.dtype(),
            });
        }
        if update.shape.rank() != input.shape.rank() {
            return Err(Error::RankMismatch {
                operation: "dynamic_update_slice",
                expected: input.shape.rank(),
                actual: update.shape.rank(),
            });
        }
        if update.shape.axis_tags() != input.shape.axis_tags() {
            return Err(Error::MetadataMismatch {
                operation: "dynamic_update_slice",
                field: "axis tags",
            });
        }
        if update.shape.partitions() != input.shape.partitions() {
            return Err(Error::MetadataMismatch {
                operation: "dynamic_update_slice",
                field: "partition metadata",
            });
        }
        for (axis, (&input_dimension, &update_dimension)) in input
            .shape
            .dimensions()
            .iter()
            .zip(update.shape.dimensions())
            .enumerate()
        {
            if update_dimension > input_dimension {
                return Err(Error::UpdateDimensionTooLarge {
                    axis,
                    input: input_dimension,
                    update: update_dimension,
                });
            }
        }
        let result = self.push_value("dynamic_update_slice", input.shape);
        self.operations.push(Operation::DynamicUpdateSlice {
            input: input.value,
            update: update.value,
            starts: starts.iter().map(|start| start.value).collect(),
            result: result.value,
        });
        Ok(result)
    }

    /// Gathers scalar indices from one operand axis. Index dimensions precede
    /// every retained operand dimension in the result.
    pub fn gather(&mut self, input: Tensor, indices: Tensor, axis: usize) -> Result<Tensor, Error> {
        self.gather_impl(input, indices, axis, 1, true)
    }

    /// Gathers fixed-size slices whose starting positions are supplied by an
    /// integer index tensor. The gathered operand axis is retained.
    pub fn gather_slices(
        &mut self,
        input: Tensor,
        indices: Tensor,
        axis: usize,
        slice_size: i64,
    ) -> Result<Tensor, Error> {
        self.gather_impl(input, indices, axis, slice_size, false)
    }

    /// Gathers scalar points addressed by vectors in the final indices axis.
    /// Index batch dimensions lead the result and operand axes not named in
    /// `axes` follow them in their original order.
    pub fn gather_nd(
        &mut self,
        input: Tensor,
        indices: Tensor,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(indices)?;
        require_index_dtype(indices.shape.dtype())?;
        let contract = nd_indexing_contract(input.shape, indices.shape, axes, "gather_nd")?;
        let result = self.push_value("gather_nd", contract.result_shape);
        let mut slice_sizes = input.shape.dimensions().to_vec();
        for axis in axes {
            slice_sizes[*axis] = 1;
        }
        self.operations.push(Operation::Gather {
            input: input.value,
            indices: indices.value,
            result: result.value,
            offset_dims: (contract.index_batch_rank
                ..contract.index_batch_rank + contract.retained_axes.len())
                .collect(),
            collapsed_slice_dims: sorted_axes(axes),
            operand_batching_dims: Vec::new(),
            start_indices_batching_dims: Vec::new(),
            start_index_map: axes.to_vec(),
            index_vector_dim: indices.shape.rank() - 1,
            slice_sizes,
            indices_are_sorted: false,
        });
        Ok(result)
    }

    /// Performs ND gather independently across equal leading batch axes.
    /// Batching is expressed as an axis count so StableHLO configuration does
    /// not leak into the public tensor API.
    pub fn gather_batched_nd(
        &mut self,
        input: Tensor,
        indices: Tensor,
        batch_axes: usize,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(indices)?;
        require_index_dtype(indices.shape.dtype())?;
        let contract = batched_nd_indexing_contract(
            input.shape,
            indices.shape,
            batch_axes,
            axes,
            "gather_batched_nd",
        )?;
        let result = self.push_value("gather_batched_nd", contract.result_shape);
        let mut slice_sizes = input.shape.dimensions().to_vec();
        for dimension in slice_sizes.iter_mut().take(batch_axes) {
            *dimension = 1;
        }
        for axis in axes {
            slice_sizes[*axis] = 1;
        }
        self.operations.push(Operation::Gather {
            input: input.value,
            indices: indices.value,
            result: result.value,
            offset_dims: (contract.index_batch_rank
                ..contract.index_batch_rank + contract.retained_axes.len())
                .collect(),
            collapsed_slice_dims: sorted_axes(axes),
            operand_batching_dims: (0..batch_axes).collect(),
            start_indices_batching_dims: (0..batch_axes).collect(),
            start_index_map: axes.to_vec(),
            index_vector_dim: indices.shape.rank() - 1,
            slice_sizes,
            indices_are_sorted: false,
        });
        Ok(result)
    }

    pub fn scatter_update(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_nd(input, indices, updates, axes, ScatterComputation::Update)
    }

    pub fn scatter_add(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_nd(input, indices, updates, axes, ScatterComputation::Add)
    }

    pub fn scatter_multiply(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_nd(input, indices, updates, axes, ScatterComputation::Multiply)
    }

    pub fn scatter_minimum(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_nd(input, indices, updates, axes, ScatterComputation::Minimum)
    }

    pub fn scatter_maximum(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_nd(input, indices, updates, axes, ScatterComputation::Maximum)
    }

    pub fn scatter_update_batched(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        batch_axes: usize,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_batched_nd(
            input,
            indices,
            updates,
            batch_axes,
            axes,
            ScatterComputation::Update,
            false,
            false,
        )
    }

    pub fn scatter_add_batched(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        batch_axes: usize,
        axes: &[usize],
    ) -> Result<Tensor, Error> {
        self.scatter_batched_nd(
            input,
            indices,
            updates,
            batch_axes,
            axes,
            ScatterComputation::Add,
            false,
            false,
        )
    }

    /// Assignment scatter with caller-proven StableHLO optimization promises.
    /// Incorrect promises can change which update wins, so ordinary code
    /// should use [`Self::scatter_update`].
    pub fn scatter_update_with_promises(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
        indices_are_sorted: bool,
        unique_indices: bool,
    ) -> Result<Tensor, Error> {
        self.scatter_batched_nd(
            input,
            indices,
            updates,
            0,
            axes,
            ScatterComputation::Update,
            indices_are_sorted,
            unique_indices,
        )
    }

    /// Gathers rows from a `[vocabulary, embedding]` weight without imposing a
    /// model-layer type or a particular checkpoint path.
    pub fn token_embedding(&mut self, weight: Tensor, indices: Tensor) -> Result<Tensor, Error> {
        self.require_rank(weight, "token embedding weight", 2)?;
        self.require_local(indices)?;
        require_index_dtype(indices.shape.dtype())?;
        self.gather(weight, indices, 0)
    }

    fn gather_impl(
        &mut self,
        input: Tensor,
        indices: Tensor,
        axis: usize,
        slice_size: i64,
        collapse_axis: bool,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(indices)?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "gather",
                axis,
                rank: input.shape.rank(),
            });
        }
        require_index_dtype(indices.shape.dtype())?;
        if slice_size <= 0 || slice_size > input.shape.dimensions()[axis] {
            return Err(Error::InvalidSlice {
                axis,
                start: 0,
                limit: slice_size,
                stride: 1,
                dimension: input.shape.dimensions()[axis],
            });
        }
        let retained_axes = (0..input.shape.rank())
            .filter(|candidate| !collapse_axis || *candidate != axis)
            .collect::<Vec<_>>();
        let dimensions = indices
            .shape
            .dimensions()
            .iter()
            .copied()
            .chain(retained_axes.iter().map(|retained| {
                if *retained == axis {
                    slice_size
                } else {
                    input.shape.dimensions()[*retained]
                }
            }))
            .collect::<Vec<_>>();
        let tags = indices
            .shape
            .axis_tags()
            .iter()
            .copied()
            .chain(
                retained_axes
                    .iter()
                    .map(|retained| input.shape.axis_tags()[*retained]),
            )
            .collect::<Vec<_>>();
        let partitions = indices
            .shape
            .partitions()
            .iter()
            .copied()
            .chain(
                retained_axes
                    .iter()
                    .map(|retained| input.shape.partitions()[*retained]),
            )
            .collect::<Vec<_>>();
        let shape = Shape::new(input.shape.dtype(), &dimensions)?
            .with_axis_tags(&tags)?
            .with_partitions(&partitions)?;
        let result = self.push_value("gather", shape);
        self.operations.push(Operation::Gather {
            input: input.value,
            indices: indices.value,
            result: result.value,
            offset_dims: (indices.shape.rank()..shape.rank()).collect(),
            collapsed_slice_dims: collapse_axis.then_some(axis).into_iter().collect(),
            operand_batching_dims: Vec::new(),
            start_indices_batching_dims: Vec::new(),
            start_index_map: vec![axis],
            index_vector_dim: indices.shape.rank(),
            slice_sizes: input
                .shape
                .dimensions()
                .iter()
                .enumerate()
                .map(|(input_axis, dimension)| {
                    if input_axis == axis {
                        slice_size
                    } else {
                        *dimension
                    }
                })
                .collect(),
            indices_are_sorted: false,
        });
        Ok(result)
    }

    fn scatter_nd(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        axes: &[usize],
        computation: ScatterComputation,
    ) -> Result<Tensor, Error> {
        self.scatter_batched_nd(input, indices, updates, 0, axes, computation, false, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_batched_nd(
        &mut self,
        input: Tensor,
        indices: Tensor,
        updates: Tensor,
        batch_axes: usize,
        axes: &[usize],
        computation: ScatterComputation,
        indices_are_sorted: bool,
        unique_indices: bool,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(indices)?;
        self.require_local(updates)?;
        require_index_dtype(indices.shape.dtype())?;
        let contract = batched_nd_indexing_contract(
            input.shape,
            indices.shape,
            batch_axes,
            axes,
            computation.name(),
        )?;
        require_matching_shape_metadata(computation.name(), updates.shape, contract.result_shape)?;
        if updates.shape.dtype() != input.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: updates.shape.dtype(),
            });
        }
        match computation {
            ScatterComputation::Update => {}
            ScatterComputation::Add | ScatterComputation::Multiply
                if input.shape.dtype() == DType::Bool =>
            {
                return Err(Error::UnsupportedDType {
                    operation: computation.name(),
                    dtype: input.shape.dtype(),
                });
            }
            ScatterComputation::Minimum | ScatterComputation::Maximum
                if input.shape.dtype() == DType::Bool
                    || !input.shape.dtype().supports_ordering() =>
            {
                return Err(Error::UnsupportedDType {
                    operation: computation.name(),
                    dtype: input.shape.dtype(),
                });
            }
            _ => {}
        }
        let result = self.push_value(computation.name(), input.shape);
        self.operations.push(Operation::Scatter {
            input: input.value,
            indices: indices.value,
            updates: updates.value,
            result: result.value,
            update_window_dims: (contract.index_batch_rank..updates.shape.rank()).collect(),
            inserted_window_dims: sorted_axes(axes),
            input_batching_dims: (0..batch_axes).collect(),
            scatter_indices_batching_dims: (0..batch_axes).collect(),
            scatter_dims_to_operand_dims: axes.to_vec(),
            index_vector_dim: indices.shape.rank() - 1,
            indices_are_sorted,
            unique_indices,
            computation,
        });
        Ok(result)
    }

    pub fn reduce_sum(&mut self, input: Tensor, axes: &[usize]) -> Result<Tensor, Error> {
        self.reduce(input, axes, Reduction::Sum)
    }

    pub fn reduce_max(&mut self, input: Tensor, axes: &[usize]) -> Result<Tensor, Error> {
        self.reduce(input, axes, Reduction::Maximum)
    }

    pub fn reduce_min(&mut self, input: Tensor, axes: &[usize]) -> Result<Tensor, Error> {
        self.reduce(input, axes, Reduction::Minimum)
    }

    pub fn all_reduce_sum(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.all_reduce(input, Reduction::Sum)
    }

    pub fn all_reduce_max(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.all_reduce(input, Reduction::Maximum)
    }

    pub fn all_reduce_min(&mut self, input: Tensor) -> Result<Tensor, Error> {
        self.all_reduce(input, Reduction::Minimum)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reduce_window_sum(
        &mut self,
        input: Tensor,
        window_dimensions: &[i64],
        window_strides: &[i64],
        base_dilations: &[i64],
        window_dilations: &[i64],
        padding: &[[i64; 2]],
    ) -> Result<Tensor, Error> {
        self.reduce_window(
            input,
            window_dimensions,
            window_strides,
            base_dilations,
            window_dilations,
            padding,
            Reduction::Sum,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reduce_window_max(
        &mut self,
        input: Tensor,
        window_dimensions: &[i64],
        window_strides: &[i64],
        base_dilations: &[i64],
        window_dilations: &[i64],
        padding: &[[i64; 2]],
    ) -> Result<Tensor, Error> {
        self.reduce_window(
            input,
            window_dimensions,
            window_strides,
            base_dilations,
            window_dilations,
            padding,
            Reduction::Maximum,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reduce_window_min(
        &mut self,
        input: Tensor,
        window_dimensions: &[i64],
        window_strides: &[i64],
        base_dilations: &[i64],
        window_dilations: &[i64],
        padding: &[[i64; 2]],
    ) -> Result<Tensor, Error> {
        self.reduce_window(
            input,
            window_dimensions,
            window_strides,
            base_dilations,
            window_dilations,
            padding,
            Reduction::Minimum,
        )
    }

    pub fn cumulative_sum(&mut self, input: Tensor, axis: usize) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), "cumulative_sum")?;
        if input.shape.dtype() == DType::Bool {
            return Err(Error::UnsupportedDType {
                operation: "cumulative_sum",
                dtype: input.shape.dtype(),
            });
        }
        let rank = input.shape.rank();
        let mut window_dimensions = vec![1; rank];
        window_dimensions[axis] = input.shape.dimensions()[axis];
        let unit = vec![1; rank];
        let mut padding = vec![[0, 0]; rank];
        padding[axis] = [input.shape.dimensions()[axis].saturating_sub(1), 0];
        self.reduce_window_sum(input, &window_dimensions, &unit, &unit, &unit, &padding)
    }

    pub fn max_pool1d(
        &mut self,
        input: Tensor,
        axis: usize,
        window: i64,
        stride: i64,
        padding: [i64; 2],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), "max_pool1d")?;
        let rank = input.shape.rank();
        let mut windows = vec![1; rank];
        windows[axis] = window;
        let mut strides = vec![1; rank];
        strides[axis] = stride;
        let unit = vec![1; rank];
        let mut paddings = vec![[0, 0]; rank];
        paddings[axis] = padding;
        self.reduce_window_max(input, &windows, &strides, &unit, &unit, &paddings)
    }

    pub fn max_pool2d(
        &mut self,
        input: Tensor,
        axes: [usize; 2],
        windows: [i64; 2],
        strides: [i64; 2],
        padding: [[i64; 2]; 2],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(&axes, input.shape.rank(), "max_pool2d")?;
        let rank = input.shape.rank();
        let mut full_windows = vec![1; rank];
        let mut full_strides = vec![1; rank];
        let mut full_padding = vec![[0, 0]; rank];
        for position in 0..2 {
            full_windows[axes[position]] = windows[position];
            full_strides[axes[position]] = strides[position];
            full_padding[axes[position]] = padding[position];
        }
        let unit = vec![1; rank];
        self.reduce_window_max(
            input,
            &full_windows,
            &full_strides,
            &unit,
            &unit,
            &full_padding,
        )
    }

    pub fn convolution(
        &mut self,
        input: Tensor,
        kernel: Tensor,
        options: ConvolutionOptions<'_>,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        self.require_local(kernel)?;
        let output_shape = convolution_output_shape(input.shape, kernel.shape, &options)?;
        let result = self.push_value("convolution", output_shape);
        self.operations.push(Operation::Convolution {
            input: input.value,
            kernel: kernel.value,
            result: result.value,
            strides: options.strides.to_vec(),
            padding: options.padding.to_vec(),
            input_dilation: options.input_dilation.to_vec(),
            kernel_dilation: options.kernel_dilation.to_vec(),
            kernel_reversal: options.kernel_reversal.to_vec(),
            input_batch_axis: options.input_batch_axis,
            input_feature_axis: options.input_feature_axis,
            input_spatial_axes: options.input_spatial_axes.to_vec(),
            kernel_input_feature_axis: options.kernel_input_feature_axis,
            kernel_output_feature_axis: options.kernel_output_feature_axis,
            kernel_spatial_axes: options.kernel_spatial_axes.to_vec(),
            output_batch_axis: options.output_batch_axis,
            output_feature_axis: options.output_feature_axis,
            output_spatial_axes: options.output_spatial_axes.to_vec(),
            feature_groups: options.feature_groups,
            batch_groups: options.batch_groups,
        });
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &mut self,
        input: Tensor,
        kernel: Tensor,
        stride: i64,
        padding: [i64; 2],
        input_dilation: i64,
        kernel_dilation: i64,
        feature_groups: i64,
    ) -> Result<Tensor, Error> {
        self.convolution(
            input,
            kernel,
            ConvolutionOptions {
                strides: &[stride],
                padding: &[padding],
                input_dilation: &[input_dilation],
                kernel_dilation: &[kernel_dilation],
                kernel_reversal: &[false],
                input_batch_axis: 0,
                input_feature_axis: 1,
                input_spatial_axes: &[2],
                kernel_input_feature_axis: 1,
                kernel_output_feature_axis: 0,
                kernel_spatial_axes: &[2],
                output_batch_axis: 0,
                output_feature_axis: 1,
                output_spatial_axes: &[2],
                feature_groups,
                batch_groups: 1,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        input: Tensor,
        kernel: Tensor,
        strides: [i64; 2],
        padding: [[i64; 2]; 2],
        input_dilation: [i64; 2],
        kernel_dilation: [i64; 2],
        feature_groups: i64,
    ) -> Result<Tensor, Error> {
        self.convolution(
            input,
            kernel,
            ConvolutionOptions {
                strides: &strides,
                padding: &padding,
                input_dilation: &input_dilation,
                kernel_dilation: &kernel_dilation,
                kernel_reversal: &[false, false],
                input_batch_axis: 0,
                input_feature_axis: 1,
                input_spatial_axes: &[2, 3],
                kernel_input_feature_axis: 1,
                kernel_output_feature_axis: 0,
                kernel_spatial_axes: &[2, 3],
                output_batch_axis: 0,
                output_feature_axis: 1,
                output_spatial_axes: &[2, 3],
                feature_groups,
                batch_groups: 1,
            },
        )
    }

    pub fn resize_nearest(
        &mut self,
        input: Tensor,
        axis: usize,
        new_length: i64,
    ) -> Result<Tensor, Error> {
        self.require_resize(input, axis, new_length, "resize_nearest")?;
        let old_length = input.shape.dimensions()[axis];
        let scale = old_length as f64 / new_length as f64;
        let indices = (0..new_length)
            .map(|position| {
                (((position as f64 + 0.5) * scale).floor() as i64).clamp(0, old_length - 1) as i32
            })
            .collect::<Vec<_>>();
        self.gather_resized_axis(input, axis, &indices)
    }

    pub fn resize_linear(
        &mut self,
        input: Tensor,
        axis: usize,
        new_length: i64,
    ) -> Result<Tensor, Error> {
        self.require_resize(input, axis, new_length, "resize_linear")?;
        let original_dtype = input.shape.dtype();
        let accumulation_dtype = interpolation_dtype(original_dtype)?;
        let input = self.convert(input, accumulation_dtype)?;
        let old_length = input.shape.dimensions()[axis];
        let scale = old_length as f64 / new_length as f64;
        let mut left_indices = Vec::with_capacity(new_length as usize);
        let mut right_indices = Vec::with_capacity(new_length as usize);
        let mut right_weights = Vec::with_capacity(new_length as usize);
        for position in 0..new_length {
            let coordinate = position as f64 * scale;
            let left = coordinate.floor() as i64;
            left_indices.push(left.clamp(0, old_length - 1) as i32);
            right_indices.push((left + 1).clamp(0, old_length - 1) as i32);
            right_weights.push(coordinate - coordinate.floor());
        }
        let left = self.gather_resized_axis(input, axis, &left_indices)?;
        let right = self.gather_resized_axis(input, axis, &right_indices)?;
        let right_weight = self.interpolation_weights(left.shape, axis, &right_weights)?;
        let one = self.scalar_for(accumulation_dtype, 1.0)?;
        let left_weight = self.subtract(one, right_weight)?;
        let left = self.multiply(left, left_weight)?;
        let right = self.multiply(right, right_weight)?;
        let result = self.add(left, right)?;
        self.convert(result, original_dtype)
    }

    pub fn resize_bilinear(
        &mut self,
        input: Tensor,
        axes: [usize; 2],
        new_lengths: [i64; 2],
    ) -> Result<Tensor, Error> {
        if axes[0] == axes[1] {
            return Err(Error::InvalidResize(
                "bilinear resize axes must be distinct",
            ));
        }
        let first = self.resize_linear(input, axes[0], new_lengths[0])?;
        self.resize_linear(first, axes[1], new_lengths[1])
    }

    pub fn resize_cubic(
        &mut self,
        input: Tensor,
        axis: usize,
        new_length: i64,
    ) -> Result<Tensor, Error> {
        self.require_resize(input, axis, new_length, "resize_cubic")?;
        let original_dtype = input.shape.dtype();
        let accumulation_dtype = interpolation_dtype(original_dtype)?;
        let input = self.convert(input, accumulation_dtype)?;
        let old_length = input.shape.dimensions()[axis];
        let scale = old_length as f64 / new_length as f64;
        let mut indices = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut weights = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for position in 0..new_length {
            let coordinate = position as f64 * scale;
            let base = coordinate.floor() as i64;
            let t = coordinate - coordinate.floor();
            let t2 = t * t;
            let t3 = t2 * t;
            let position_weights = [
                -0.5 * t + t2 - 0.5 * t3,
                1.0 - 2.5 * t2 + 1.5 * t3,
                0.5 * t + 2.0 * t2 - 1.5 * t3,
                -0.5 * t2 + 0.5 * t3,
            ];
            for neighbor in 0..4 {
                indices[neighbor]
                    .push((base + neighbor as i64 - 1).clamp(0, old_length - 1) as i32);
                weights[neighbor].push(position_weights[neighbor]);
            }
        }
        let mut result = None;
        for neighbor in 0..4 {
            let values = self.gather_resized_axis(input, axis, &indices[neighbor])?;
            let weight = self.interpolation_weights(values.shape, axis, &weights[neighbor])?;
            let weighted = self.multiply(values, weight)?;
            result = Some(match result {
                None => weighted,
                Some(result) => self.add(result, weighted)?,
            });
        }
        self.convert(
            result.expect("cubic interpolation always has four neighbors"),
            original_dtype,
        )
    }

    pub fn upsample_nearest(
        &mut self,
        input: Tensor,
        scale_factors: &[f64],
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if input.shape.rank() < 3 || input.shape.rank() > 5 {
            return Err(Error::InvalidResize(
                "nearest upsampling requires rank 3, 4, or 5",
            ));
        }
        let spatial_rank = input.shape.rank() - 2;
        if scale_factors.len() != 1 && scale_factors.len() != spatial_rank {
            return Err(Error::InvalidResize(
                "upsampling requires one shared scale or one per spatial axis",
            ));
        }
        let mut output = input;
        for spatial in 0..spatial_rank {
            let scale = scale_factors[if scale_factors.len() == 1 { 0 } else { spatial }];
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Error::InvalidResize(
                    "upsampling scale factors must be finite and positive",
                ));
            }
            let axis = spatial + 2;
            let new_length = (input.shape.dimensions()[axis] as f64 * scale).floor() as i64;
            output = self.resize_nearest(output, axis, new_length)?;
        }
        Ok(output)
    }

    /// Reduces floating-point axes with F32 accumulation for F16 and BF16.
    pub fn mean(&mut self, input: Tensor, axes: &[usize]) -> Result<Tensor, Error> {
        self.require_float(input, "mean")?;
        validate_axes(axes, input.shape.rank(), "mean")?;
        let count = axes.iter().try_fold(1i64, |count, axis| {
            count.checked_mul(input.shape.dimensions()[*axis])
        });
        let Some(count) = count else {
            return Err(Error::InvalidReduction("mean element count overflows i64"));
        };
        if count == 0 {
            return Err(Error::InvalidReduction("mean over an empty dimension"));
        }
        let accumulation = self.float_accumulation(input)?;
        let sum = self.reduce_sum(accumulation, axes)?;
        let count = self.scalar_for(accumulation.shape.dtype(), count as f64)?;
        let mean = self.divide(sum, count)?;
        self.convert(mean, input.shape.dtype())
    }

    /// Computes a maximum-shifted log-sum-exp and preserves `-inf` for rows
    /// containing only negative infinity.
    pub fn log_sum_exp(&mut self, input: Tensor, axes: &[usize]) -> Result<Tensor, Error> {
        self.require_float(input, "log_sum_exp")?;
        validate_axes(axes, input.shape.rank(), "log_sum_exp")?;
        let accumulation = self.float_accumulation(input)?;
        let maximum = self.reduce_max(accumulation, axes)?;
        let negative_infinity = self.scalar_for(accumulation.shape.dtype(), f64::NEG_INFINITY)?;
        let row_has_values = self.greater(maximum, negative_infinity)?;
        let zero = self.zero_for(accumulation.shape.dtype())?;
        let safe_maximum = self.select(row_has_values, maximum, zero)?;
        let broadcast_maximum = self.broadcast_reduction(safe_maximum, accumulation.shape, axes)?;
        let shifted = self.subtract(accumulation, broadcast_maximum)?;
        let exponentials = self.exp(shifted)?;
        let sum = self.reduce_sum(exponentials, axes)?;
        let logarithm = self.log(sum)?;
        let result = self.add(logarithm, safe_maximum)?;
        let result = self.select(row_has_values, result, negative_infinity)?;
        self.convert(result, input.shape.dtype())
    }

    /// Returns `(values, indices)` after removing `axis`. Ties select the first
    /// index and NaNs propagate with their first index, matching ZML.
    pub fn argmax(&mut self, input: Tensor, axis: usize) -> Result<(Tensor, Tensor), Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), "argmax")?;
        if !input.shape.dtype().supports_ordering() || input.shape.dtype() == DType::Bool {
            return Err(Error::UnsupportedDType {
                operation: "argmax",
                dtype: input.shape.dtype(),
            });
        }
        let dimension = input.shape.dimensions()[axis];
        if dimension == 0 {
            return Err(Error::InvalidReduction("argmax over an empty dimension"));
        }
        let index_dtype = if dimension <= i32::MAX as i64 {
            DType::I32
        } else {
            DType::I64
        };
        let indices = self.iota(input.shape.with_dtype(index_dtype), axis)?;
        let value_init = self.minimum_for(input.shape.dtype())?;
        let index_init = self.zero_for(index_dtype)?;
        let value_result = self.push_value(
            "argmax_value",
            reduced_shape(input.shape, &[axis], input.shape.dtype())?,
        );
        let index_result = self.push_value(
            "argmax_index",
            reduced_shape(input.shape, &[axis], index_dtype)?,
        );
        self.operations.push(Operation::ArgMax {
            input: input.value,
            indices: indices.value,
            value_init: value_init.value,
            index_init: index_init.value,
            value_result: value_result.value,
            index_result: index_result.value,
            axis,
        });
        Ok((value_result, index_result))
    }

    /// Sorts one tensor axis and carries original indices through the same
    /// total comparator. Equal values and equal NaNs use the lower original
    /// index, so both algorithm modes remain deterministic.
    pub fn sort(
        &mut self,
        input: Tensor,
        axis: usize,
        descending: bool,
        stable: bool,
    ) -> Result<(Tensor, Tensor), Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), "sort")?;
        if input.shape.dimensions()[axis] > i64::from(i32::MAX) {
            return Err(Error::InvalidSort("sorted axis does not fit i32 indices"));
        }
        if input.shape.dtype() == DType::Bool || !input.shape.dtype().supports_ordering() {
            return Err(Error::UnsupportedDType {
                operation: "sort",
                dtype: input.shape.dtype(),
            });
        }
        let indices = self.iota(input.shape.with_dtype(DType::I32), axis)?;
        let values_result = self.push_value("sorted_values", input.shape);
        let indices_result = self.push_value("sorted_indices", indices.shape);
        self.operations.push(Operation::Sort {
            input: input.value,
            indices: indices.value,
            values_result: values_result.value,
            indices_result: indices_result.value,
            axis,
            descending,
            stable,
        });
        Ok((values_result, indices_result))
    }

    pub fn argsort(
        &mut self,
        input: Tensor,
        axis: usize,
        descending: bool,
        stable: bool,
    ) -> Result<Tensor, Error> {
        self.sort(input, axis, descending, stable)
            .map(|(_, indices)| indices)
    }

    pub fn top_k(
        &mut self,
        input: Tensor,
        axis: usize,
        k: usize,
        descending: bool,
    ) -> Result<(Tensor, Tensor), Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), "top_k")?;
        let dimension = usize::try_from(input.shape.dimensions()[axis])
            .map_err(|_| Error::InvalidSort("top-k dimension does not fit usize"))?;
        if k == 0 || k > dimension {
            return Err(Error::InvalidSort(
                "top-k bound must be positive and no larger than its axis",
            ));
        }
        let (values, indices) = self.sort(input, axis, descending, false)?;
        let starts = vec![0; input.shape.rank()];
        let mut limits = input.shape.dimensions().to_vec();
        let strides = vec![1; input.shape.rank()];
        limits[axis] = k as i64;
        let values = self.slice(values, &starts, &limits, &strides)?;
        let indices = self.slice(indices, &starts, &limits, &strides)?;
        Ok((values, indices))
    }

    pub fn random_state(&self, tensor: Tensor) -> Result<RandomState, Error> {
        self.require_local(tensor)?;
        if tensor.shape.dtype() != DType::U64 || tensor.shape.dimensions() != [2] {
            return Err(Error::InvalidRandom(
                "random state must be a U64 tensor with shape [2]",
            ));
        }
        Ok(RandomState { tensor })
    }

    pub fn random_bits(
        &mut self,
        state: RandomState,
        shape: Shape,
    ) -> Result<(RandomState, Tensor), Error> {
        require_supported_layout(shape)?;
        if !matches!(shape.dtype(), DType::U32 | DType::U64) {
            return Err(Error::InvalidRandom(
                "random bits output must use U32 or U64 elements",
            ));
        }
        if !self.consumed_random_states.insert(state.tensor.value) {
            return Err(Error::InvalidRandom(
                "random state has already been consumed",
            ));
        }
        let state_result = self.push_value("random_state", state.tensor.shape);
        let output_result = self.push_value("random_bits", shape);
        self.operations.push(Operation::RngBitGenerator {
            state: state.tensor.value,
            state_result: state_result.value,
            output_result: output_result.value,
        });
        Ok((
            RandomState {
                tensor: state_result,
            },
            output_result,
        ))
    }

    pub fn random_uniform(
        &mut self,
        state: RandomState,
        shape: Shape,
        minimum: f64,
        maximum: f64,
    ) -> Result<(RandomState, Tensor), Error> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum >= maximum {
            return Err(Error::InvalidRandom(
                "uniform bounds must be finite and strictly increasing",
            ));
        }
        let dtype = interpolation_dtype(shape.dtype())?;
        let (unsigned_dtype, shift, exponent) = match dtype {
            DType::F32 => (DType::U32, 9u64, 0x3f80_0000u64),
            DType::F64 => (DType::U64, 12u64, 0x3ff0_0000_0000_0000u64),
            _ => unreachable!("interpolation dtype is F32 or F64"),
        };
        let bit_shape = shape.with_dtype(unsigned_dtype);
        let (state, bits) = self.random_bits(state, bit_shape)?;
        let shift = match unsigned_dtype {
            DType::U32 => self.scalar(shift as u32)?,
            DType::U64 => self.scalar(shift)?,
            _ => unreachable!(),
        };
        let bits = self.shift_right_logical(bits, shift)?;
        let exponent = match unsigned_dtype {
            DType::U32 => self.scalar(exponent as u32)?,
            DType::U64 => self.scalar(exponent)?,
            _ => unreachable!(),
        };
        let bits = self.logical_or(bits, exponent)?;
        let values = self.bitcast(bits, dtype)?;
        let one = self.scalar_for(dtype, 1.0)?;
        let values = self.subtract(values, one)?;
        let scale = self.scalar_for(dtype, maximum - minimum)?;
        let values = self.multiply(values, scale)?;
        let offset = self.scalar_for(dtype, minimum)?;
        let values = self.add(values, offset)?;
        Ok((state, self.convert(values, shape.dtype())?))
    }

    pub fn random_normal(
        &mut self,
        state: RandomState,
        shape: Shape,
        mean: f64,
        standard_deviation: f64,
    ) -> Result<(RandomState, Tensor), Error> {
        self.require_distribution_shape(shape, "random_normal")?;
        if !mean.is_finite() || !standard_deviation.is_finite() || standard_deviation <= 0.0 {
            return Err(Error::InvalidRandom(
                "normal mean must be finite and standard deviation positive",
            ));
        }
        let dtype = interpolation_dtype(shape.dtype())?;
        let accumulation_shape = shape.with_dtype(dtype);
        let epsilon = if dtype == DType::F64 {
            f64::EPSILON
        } else {
            f64::from(f32::EPSILON)
        };
        let (state, first) = self.random_uniform(state, accumulation_shape, epsilon, 1.0)?;
        let (state, second) = self.random_uniform(state, accumulation_shape, 0.0, 1.0)?;
        let logarithm = self.log(first)?;
        let minus_two = self.scalar_for(dtype, -2.0)?;
        let radius = self.multiply(logarithm, minus_two)?;
        let radius = self.sqrt(radius)?;
        let tau = self.scalar_for(dtype, std::f64::consts::TAU)?;
        let angle = self.multiply(second, tau)?;
        let direction = self.cos(angle)?;
        let values = self.multiply(radius, direction)?;
        let scale = self.scalar_for(dtype, standard_deviation)?;
        let values = self.multiply(values, scale)?;
        let offset = self.scalar_for(dtype, mean)?;
        let values = self.add(values, offset)?;
        Ok((state, self.convert(values, shape.dtype())?))
    }

    pub fn random_gumbel(
        &mut self,
        state: RandomState,
        shape: Shape,
    ) -> Result<(RandomState, Tensor), Error> {
        self.require_distribution_shape(shape, "random_gumbel")?;
        let dtype = interpolation_dtype(shape.dtype())?;
        let accumulation_shape = shape.with_dtype(dtype);
        let epsilon = if dtype == DType::F64 {
            f64::EPSILON
        } else {
            f64::from(f32::EPSILON)
        };
        let (state, uniform) = self.random_uniform(state, accumulation_shape, epsilon, 1.0)?;
        let logarithm = self.log(uniform)?;
        let negative = self.negate(logarithm)?;
        let logarithm = self.log(negative)?;
        let gumbel = self.negate(logarithm)?;
        Ok((state, self.convert(gumbel, shape.dtype())?))
    }

    pub fn greedy_tokens(&mut self, logits: Tensor, axis: usize) -> Result<Tensor, Error> {
        self.require_float(logits, "greedy_tokens")?;
        self.argmax(logits, axis).map(|(_, indices)| indices)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_tokens(
        &mut self,
        logits: Tensor,
        state: RandomState,
        axis: usize,
        top_k: usize,
        temperature: f64,
        top_p: f64,
        min_p: f64,
    ) -> Result<(RandomState, Tensor), Error> {
        self.require_float(logits, "sample_tokens")?;
        validate_axes(&[axis], logits.shape.rank(), "sample_tokens")?;
        let vocabulary = usize::try_from(logits.shape.dimensions()[axis])
            .map_err(|_| Error::InvalidSampling("vocabulary does not fit usize"))?;
        if top_k == 0 || top_k > vocabulary {
            return Err(Error::InvalidSampling(
                "top-k must be positive and no larger than the vocabulary",
            ));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Error::InvalidSampling(
                "temperature must be finite and positive",
            ));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(Error::InvalidSampling("top-p must be in (0, 1]"));
        }
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(Error::InvalidSampling("min-p must be in [0, 1]"));
        }
        if top_k == 1 {
            return Ok((state, self.greedy_tokens(logits, axis)?));
        }
        let maximum_top_k = top_k;
        let top_k = self.scalar(i32::try_from(maximum_top_k).map_err(|_| {
            Error::InvalidSampling("top-k does not fit the I32 sampling index contract")
        })?)?;
        let dtype = if matches!(logits.shape.dtype(), DType::F16 | DType::Bf16) {
            DType::F32
        } else {
            logits.shape.dtype()
        };
        let temperature = self.scalar_for(dtype, temperature)?;
        let top_p = self.scalar_for(dtype, top_p)?;
        let min_p = self.scalar_for(dtype, min_p)?;
        self.sample_tokens_dynamic(
            logits,
            state,
            axis,
            top_k,
            temperature,
            top_p,
            min_p,
            maximum_top_k,
        )
    }

    /// Samples with runtime scalar controls and a static candidate bound. The
    /// runtime controls are clamped to their safe domains; callers wanting
    /// invalid-option diagnostics should use [`Self::sample_tokens`].
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tokens_dynamic(
        &mut self,
        logits: Tensor,
        state: RandomState,
        axis: usize,
        top_k: Tensor,
        temperature: Tensor,
        top_p: Tensor,
        min_p: Tensor,
        maximum_top_k: usize,
    ) -> Result<(RandomState, Tensor), Error> {
        self.require_float(logits, "sample_tokens_dynamic")?;
        validate_axes(&[axis], logits.shape.rank(), "sample_tokens_dynamic")?;
        self.require_sampling_scalar(top_k, DType::I32, "top_k")?;
        for (value, name) in [
            (temperature, "temperature"),
            (top_p, "top_p"),
            (min_p, "min_p"),
        ] {
            self.require_local(value)?;
            if value.shape.rank() != 0 || value.shape.dtype().class() != DTypeClass::Float {
                return Err(Error::InvalidSampling(match name {
                    "temperature" => "temperature must be a floating-point scalar",
                    "top_p" => "top-p must be a floating-point scalar",
                    _ => "min-p must be a floating-point scalar",
                }));
            }
        }
        let vocabulary = usize::try_from(logits.shape.dimensions()[axis])
            .map_err(|_| Error::InvalidSampling("vocabulary does not fit usize"))?;
        if maximum_top_k == 0 || maximum_top_k > vocabulary || maximum_top_k > i32::MAX as usize {
            return Err(Error::InvalidSampling(
                "maximum top-k must be positive, fit I32, and not exceed the vocabulary",
            ));
        }

        let logits = self.move_axis_to_last(logits, axis)?;
        let candidate_axis = logits.shape.rank() - 1;
        let (values, original_indices) = self.top_k(logits, candidate_axis, maximum_top_k, true)?;
        let values = if matches!(values.shape.dtype(), DType::F16 | DType::Bf16) {
            self.convert(values, DType::F32)?
        } else {
            values
        };
        let one_i32 = self.scalar(1i32)?;
        let maximum_i32 = self.scalar(maximum_top_k as i32)?;
        let top_k = self.maximum(top_k, one_i32)?;
        let top_k = self.minimum(top_k, maximum_i32)?;
        let candidate_indices = self.iota(values.shape.with_dtype(DType::I32), candidate_axis)?;
        let outside_top_k = self.greater_equal(candidate_indices, top_k)?;

        let dtype = values.shape.dtype();
        let epsilon = self.scalar_for(
            dtype,
            if dtype == DType::F64 {
                f64::EPSILON
            } else {
                f64::from(f32::EPSILON)
            },
        )?;
        let one = self.scalar_for(dtype, 1.0)?;
        let zero = self.scalar_for(dtype, 0.0)?;
        let temperature = self.convert(temperature, dtype)?;
        let temperature = self.maximum(temperature, epsilon)?;
        let top_p = self.convert(top_p, dtype)?;
        let top_p = self.maximum(top_p, epsilon)?;
        let top_p = self.minimum(top_p, one)?;
        let min_p = self.convert(min_p, dtype)?;
        let min_p = self.maximum(min_p, zero)?;
        let min_p = self.minimum(min_p, one)?;

        let negative_infinity = self.scalar_for(dtype, f64::NEG_INFINITY)?;
        let filtered = self.select(outside_top_k, negative_infinity, values)?;
        let filtered = self.divide(filtered, temperature)?;
        let probabilities = self.softmax(filtered, candidate_axis)?;
        let cumulative = self.cumulative_sum(probabilities, candidate_axis)?;
        let cumulative_before = self.subtract(cumulative, probabilities)?;
        let within_top_p = self.less(cumulative_before, top_p)?;
        let starts = vec![0; probabilities.shape.rank()];
        let mut limits = probabilities.shape.dimensions().to_vec();
        let strides = vec![1; probabilities.shape.rank()];
        limits[candidate_axis] = 1;
        let maximum_probability = self.slice(probabilities, &starts, &limits, &strides)?;
        let threshold = self.multiply(maximum_probability, min_p)?;
        let threshold = self.broadcast_in_dim(
            threshold,
            probabilities.shape,
            &(0..probabilities.shape.rank()).collect::<Vec<_>>(),
        )?;
        let above_min_p = self.greater_equal(probabilities, threshold)?;
        let accepted = self.logical_and(within_top_p, above_min_p)?;
        let zero_i32 = self.scalar(0i32)?;
        let first = self.equal(candidate_indices, zero_i32)?;
        let accepted = self.logical_or(accepted, first)?;
        let filtered = self.select(accepted, filtered, negative_infinity)?;

        let (state, noise) = self.random_gumbel(state, filtered.shape)?;
        let scored = self.add(filtered, noise)?;
        let (_, selected) = self.argmax(scored, candidate_axis)?;
        let vector_indices = self.insert_axis(selected, selected.shape.rank(), AxisTag::UNKNOWN)?;
        let batch_axes = original_indices.shape.rank() - 1;
        let selected = self.gather_batched_nd(
            original_indices,
            vector_indices,
            batch_axes,
            &[candidate_axis],
        )?;
        Ok((state, selected))
    }

    pub fn softmax(&mut self, input: Tensor, axis: usize) -> Result<Tensor, Error> {
        self.require_float(input, "softmax")?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "softmax",
                axis,
                rank: input.shape.rank(),
            });
        }
        let accumulation = if matches!(input.shape.dtype(), DType::F16 | DType::Bf16) {
            self.convert(input, DType::F32)?
        } else {
            input
        };
        let maximum = self.reduce_max(accumulation, &[axis])?;
        let negative_infinity = self.scalar_for(accumulation.shape.dtype(), f64::NEG_INFINITY)?;
        let row_has_values = self.greater(maximum, negative_infinity)?;
        let broadcast_maximum = self.broadcast_reduction(maximum, accumulation.shape, &[axis])?;
        let shifted = self.subtract(accumulation, broadcast_maximum)?;
        let exponentials = self.exp(shifted)?;
        let denominator = self.reduce_sum(exponentials, &[axis])?;
        let denominator = self.broadcast_reduction(denominator, accumulation.shape, &[axis])?;
        let normalized = self.divide(exponentials, denominator)?;
        let normalized = self.convert(normalized, input.shape.dtype())?;
        let row_has_values =
            self.broadcast_reduction(row_has_values, input.shape.with_dtype(DType::Bool), &[axis])?;
        let zero = self.zero_for(input.shape.dtype())?;
        self.select(row_has_values, normalized, zero)
    }

    pub fn rms_norm(
        &mut self,
        input: Tensor,
        weight: Option<Tensor>,
        axis: usize,
        epsilon: f64,
    ) -> Result<Tensor, Error> {
        self.require_float(input, "rms_norm")?;
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "rms_norm",
                axis,
                rank: input.shape.rank(),
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidNormalization(
                "epsilon must be finite and positive",
            ));
        }
        let accumulation = if matches!(input.shape.dtype(), DType::F16 | DType::Bf16) {
            self.convert(input, DType::F32)?
        } else {
            input
        };
        let square = self.multiply(accumulation, accumulation)?;
        let sum = self.reduce_sum(square, &[axis])?;
        let count = self.scalar_for(
            accumulation.shape.dtype(),
            input.shape.dimensions()[axis] as f64,
        )?;
        let mean = self.divide(sum, count)?;
        let epsilon = self.scalar_for(accumulation.shape.dtype(), epsilon)?;
        let variance = self.add(mean, epsilon)?;
        let inverse = self.rsqrt(variance)?;
        let inverse = self.broadcast_reduction(inverse, accumulation.shape, &[axis])?;
        let normalized = self.multiply(accumulation, inverse)?;
        let normalized = self.convert(normalized, input.shape.dtype())?;
        let Some(weight) = weight else {
            return Ok(normalized);
        };
        self.require_local(weight)?;
        self.require_rank(weight, "rms_norm weight", 1)?;
        if weight.shape.dtype() != input.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: weight.shape.dtype(),
            });
        }
        if weight.shape.dimensions()[0] != input.shape.dimensions()[axis] {
            return Err(Error::DimensionMismatch {
                left_axis: 0,
                right_axis: axis,
                left: weight.shape.dimensions()[0],
                right: input.shape.dimensions()[axis],
            });
        }
        let weight = self.broadcast_in_dim(weight, input.shape, &[axis])?;
        self.multiply(normalized, weight)
    }

    /// Centers one axis and scales it by the reciprocal standard deviation.
    pub fn normalize_variance(
        &mut self,
        input: Tensor,
        axis: usize,
        epsilon: f64,
    ) -> Result<Tensor, Error> {
        self.require_float(input, "normalize_variance")?;
        self.validate_normalization_axis(input, axis, epsilon)?;
        let accumulation = self.float_accumulation(input)?;
        let sum = self.reduce_sum(accumulation, &[axis])?;
        let count = self.scalar_for(
            accumulation.shape.dtype(),
            input.shape.dimensions()[axis] as f64,
        )?;
        let mean = self.divide(sum, count)?;
        let mean = self.broadcast_reduction(mean, accumulation.shape, &[axis])?;
        let centered = self.subtract(accumulation, mean)?;
        let square = self.multiply(centered, centered)?;
        let variance = self.reduce_sum(square, &[axis])?;
        let variance = self.divide(variance, count)?;
        let epsilon = self.scalar_for(accumulation.shape.dtype(), epsilon)?;
        let variance = self.add(variance, epsilon)?;
        let inverse = self.rsqrt(variance)?;
        let inverse = self.broadcast_reduction(inverse, accumulation.shape, &[axis])?;
        let normalized = self.multiply(centered, inverse)?;
        self.convert(normalized, input.shape.dtype())
    }

    /// Applies variance normalization followed by optional one-dimensional
    /// scale and bias parameters along `axis`.
    pub fn layer_norm(
        &mut self,
        input: Tensor,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        axis: usize,
        epsilon: f64,
    ) -> Result<Tensor, Error> {
        let mut output = self.normalize_variance(input, axis, epsilon)?;
        if let Some(weight) = weight {
            let weight = self.normalization_parameter(input, weight, axis, "layer_norm weight")?;
            output = self.multiply(output, weight)?;
        }
        if let Some(bias) = bias {
            let bias = self.normalization_parameter(input, bias, axis, "layer_norm bias")?;
            output = self.add(output, bias)?;
        }
        Ok(output)
    }

    /// Scales values by the reciprocal L2 norm over arbitrary axes. Keeping
    /// axes explicit leaves vector, head, and feature normalization available
    /// without introducing separate public layer types.
    pub fn normalize_l2(
        &mut self,
        input: Tensor,
        axes: &[usize],
        epsilon: f64,
    ) -> Result<Tensor, Error> {
        self.require_float(input, "normalize_l2")?;
        validate_axes(axes, input.shape.rank(), "normalize_l2")?;
        if axes.is_empty() {
            return Err(Error::InvalidNormalization(
                "normalization requires at least one axis",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidNormalization(
                "epsilon must be finite and positive",
            ));
        }
        if axes.iter().any(|axis| input.shape.dimensions()[*axis] == 0) {
            return Err(Error::InvalidNormalization(
                "normalization dimensions must be nonempty",
            ));
        }
        let accumulation = self.float_accumulation(input)?;
        let square = self.multiply(accumulation, accumulation)?;
        let sum = self.reduce_sum(square, axes)?;
        let epsilon = self.scalar_for(accumulation.shape.dtype(), epsilon)?;
        let norm = self.add(sum, epsilon)?;
        let inverse = self.rsqrt(norm)?;
        let inverse = self.broadcast_reduction(inverse, accumulation.shape, axes)?;
        let normalized = self.multiply(accumulation, inverse)?;
        self.convert(normalized, input.shape.dtype())
    }

    /// Applies rotary position embeddings to `[batch, sequence, heads, head_dim]`.
    pub fn rope(
        &mut self,
        input: Tensor,
        positions: Tensor,
        options: RopeOptions,
    ) -> Result<Tensor, Error> {
        self.require_float(input, "rope")?;
        self.require_rank(input, "rope input", 4)?;
        self.require_local(positions)?;
        self.require_rank(positions, "rope positions", 2)?;
        require_index_dtype(positions.shape.dtype())?;
        if input.shape.dimensions()[0..2] != positions.shape.dimensions()[..] {
            return Err(Error::InvalidRope(
                "position dimensions must match batch and sequence",
            ));
        }
        if options.rotary_dimensions == 0
            || options.rotary_dimensions % 2 != 0
            || options.rotary_dimensions > input.shape.dimensions()[3] as usize
        {
            return Err(Error::InvalidRope(
                "rotary width must be positive, even, and no larger than head_dim",
            ));
        }
        if !options.base.is_finite() || options.base <= 0.0 {
            return Err(Error::InvalidRope("base must be finite and positive"));
        }

        let [batch, sequence, heads, head_dim] = input.shape.dimensions() else {
            unreachable!("rank was checked")
        };
        let rotary = options.rotary_dimensions as i64;
        let half = rotary / 2;
        let frequencies = rope_frequencies(options)?;
        let frequency_shape = Shape::new(DType::F32, &[half])?;
        let frequency_slice = Slice::from_typed(frequency_shape, &frequencies)?;
        let frequencies = self.constant(&frequency_slice)?;
        let positions = self.convert(positions, DType::F32)?;
        let angle_shape = Shape::new(DType::F32, &[*batch, *sequence, half])?;
        let positions = self.broadcast_in_dim(positions, angle_shape, &[0, 1])?;
        let frequencies = self.broadcast_in_dim(frequencies, angle_shape, &[2])?;
        let angles = self.multiply(positions, frequencies)?;
        let cosine = self.cos(angles)?;
        let sine = self.sin(angles)?;

        let input_f32 = self.convert(input, DType::F32)?;
        let rotary_input = self.slice(
            input_f32,
            &[0, 0, 0, 0],
            &[*batch, *sequence, *heads, rotary],
            &[1, 1, 1, 1],
        )?;
        let pair_shape = Shape::new(DType::F32, &[*batch, *sequence, *heads, half])?;
        let cosine = self.broadcast_in_dim(cosine, pair_shape, &[0, 1, 3])?;
        let sine = self.broadcast_in_dim(sine, pair_shape, &[0, 1, 3])?;
        let (first, second) = match options.layout {
            RopeLayout::Sequential => {
                let first = self.slice(
                    rotary_input,
                    &[0, 0, 0, 0],
                    &[*batch, *sequence, *heads, half],
                    &[1, 1, 1, 1],
                )?;
                let second = self.slice(
                    rotary_input,
                    &[0, 0, 0, half],
                    &[*batch, *sequence, *heads, rotary],
                    &[1, 1, 1, 1],
                )?;
                let first_cos = self.multiply(first, cosine)?;
                let second_sin = self.multiply(second, sine)?;
                let rotated_first = self.subtract(first_cos, second_sin)?;
                let second_cos = self.multiply(second, cosine)?;
                let first_sin = self.multiply(first, sine)?;
                let rotated_second = self.add(second_cos, first_sin)?;
                (rotated_first, rotated_second)
            }
            RopeLayout::Interleaved => {
                let even = self.slice(
                    rotary_input,
                    &[0, 0, 0, 0],
                    &[*batch, *sequence, *heads, rotary],
                    &[1, 1, 1, 2],
                )?;
                let odd = self.slice(
                    rotary_input,
                    &[0, 0, 0, 1],
                    &[*batch, *sequence, *heads, rotary],
                    &[1, 1, 1, 2],
                )?;
                let even_cos = self.multiply(even, cosine)?;
                let odd_sin = self.multiply(odd, sine)?;
                let rotated_even = self.subtract(even_cos, odd_sin)?;
                let odd_cos = self.multiply(odd, cosine)?;
                let even_sin = self.multiply(even, sine)?;
                let rotated_odd = self.add(odd_cos, even_sin)?;
                (rotated_even, rotated_odd)
            }
        };
        let rotated = match options.layout {
            RopeLayout::Sequential => self.concatenate(&[first, second], 3)?,
            RopeLayout::Interleaved => {
                let expanded = Shape::new(DType::F32, &[*batch, *sequence, *heads, half, 1])?;
                let first = self.reshape(first, expanded)?;
                let second = self.reshape(second, expanded)?;
                let pairs = self.concatenate(&[first, second], 4)?;
                self.reshape(
                    pairs,
                    Shape::new(DType::F32, &[*batch, *sequence, *heads, rotary])?,
                )?
            }
        };
        let rotated = if rotary == *head_dim {
            rotated
        } else {
            let tail = self.slice(
                input_f32,
                &[0, 0, 0, rotary],
                &[*batch, *sequence, *heads, *head_dim],
                &[1, 1, 1, 1],
            )?;
            self.concatenate(&[rotated, tail], 3)?
        };
        let rotated = self.convert(rotated, input.shape.dtype())?;
        self.reshape(rotated, input.shape)
    }

    /// Portable ordinary attention over dense K/V tensors.
    pub fn attention(
        &mut self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_positions: Tensor,
        key_positions: Tensor,
        options: AttentionOptions,
    ) -> Result<Tensor, Error> {
        for tensor in [query, key, value] {
            self.require_float(tensor, "attention")?;
            self.require_rank(tensor, "attention", 4)?;
        }
        self.require_local(query_positions)?;
        self.require_local(key_positions)?;
        self.require_rank(query_positions, "query positions", 2)?;
        self.require_rank(key_positions, "key positions", 2)?;
        require_index_dtype(query_positions.shape.dtype())?;
        require_index_dtype(key_positions.shape.dtype())?;
        if query.shape.dtype() != key.shape.dtype() || query.shape.dtype() != value.shape.dtype() {
            return Err(Error::InvalidAttention("Q, K, and V dtypes must match"));
        }
        if key.shape.dimensions() != value.shape.dimensions() {
            return Err(Error::InvalidAttention("K and V shapes must match"));
        }
        let [batch, query_len, query_heads, head_dim] = query.shape.dimensions() else {
            unreachable!("rank was checked")
        };
        let [key_batch, key_len, kv_heads, key_head_dim] = key.shape.dimensions() else {
            unreachable!("rank was checked")
        };
        if *batch <= 0
            || *query_len <= 0
            || *key_len <= 0
            || *query_heads <= 0
            || *kv_heads <= 0
            || *head_dim <= 0
        {
            return Err(Error::InvalidAttention(
                "batch, sequence, head count, and head dimension must be positive",
            ));
        }
        if batch != key_batch || head_dim != key_head_dim {
            return Err(Error::InvalidAttention(
                "Q and K/V batch and head dimensions must match",
            ));
        }
        if query_heads % kv_heads != 0 {
            return Err(Error::InvalidAttention(
                "query head count must be divisible by KV head count",
            ));
        }
        if query_positions.shape.dimensions() != &[*batch, *query_len]
            || key_positions.shape.dimensions() != &[*batch, *key_len]
        {
            return Err(Error::InvalidAttention(
                "position tensors must match query and key sequence shapes",
            ));
        }
        if options.sliding_window.is_some_and(|window| window <= 0) {
            return Err(Error::InvalidAttention(
                "sliding window must be positive when specified",
            ));
        }
        let scale = options
            .scale
            .unwrap_or_else(|| 1.0 / (*head_dim as f64).sqrt());
        let kernel_scale = scale as f32;
        if !scale.is_finite() || scale <= 0.0 || !kernel_scale.is_finite() || kernel_scale <= 0.0 {
            return Err(Error::InvalidAttention(
                "scale must be representable as positive finite F32",
            ));
        }

        let result = self.push_value("attention", query.shape);
        self.operations.push(Operation::Attention {
            query: query.value,
            key: key.value,
            value: value.value,
            query_positions: query_positions.value,
            key_positions: key_positions.value,
            result: result.value,
            options: AttentionOptions {
                scale: Some(scale),
                ..options
            },
        });
        Ok(result)
    }

    /// Portable blockwise paged attention. K/V storage is
    /// `[physical_pages, page_size, kv_heads, head_dim]`; the page table is
    /// `[batch, logical_pages]` and sequence lengths are `[batch]`.
    pub fn paged_attention(
        &mut self,
        query: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        page_table: Tensor,
        sequence_lengths: Tensor,
        query_positions: Tensor,
        options: AttentionOptions,
    ) -> Result<Tensor, Error> {
        for tensor in [query, key_cache, value_cache] {
            self.require_float(tensor, "paged_attention")?;
            self.require_rank(tensor, "paged_attention", 4)?;
        }
        for tensor in [page_table, sequence_lengths, query_positions] {
            self.require_local(tensor)?;
            require_index_dtype(tensor.shape.dtype())?;
        }
        self.require_rank(page_table, "page table", 2)?;
        self.require_rank(sequence_lengths, "sequence lengths", 1)?;
        self.require_rank(query_positions, "query positions", 2)?;
        if query.shape.dtype() != key_cache.shape.dtype()
            || query.shape.dtype() != value_cache.shape.dtype()
        {
            return Err(Error::InvalidAttention(
                "query and paged K/V cache dtypes must match",
            ));
        }
        if key_cache.shape.dimensions() != value_cache.shape.dimensions() {
            return Err(Error::InvalidAttention(
                "paged K and V cache shapes must match",
            ));
        }
        let [batch, query_len, query_heads, head_dim] = query.shape.dimensions() else {
            unreachable!("rank was checked")
        };
        let [physical_pages, page_size, kv_heads, cache_head_dim] = key_cache.shape.dimensions()
        else {
            unreachable!("rank was checked")
        };
        if *batch <= 0
            || *query_len <= 0
            || *query_heads <= 0
            || *head_dim <= 0
            || *physical_pages <= 0
            || *page_size <= 0
            || *kv_heads <= 0
        {
            return Err(Error::InvalidAttention(
                "batch, query length, page geometry, head count, and head dimension must be positive",
            ));
        }
        if head_dim != cache_head_dim || query_heads % kv_heads != 0 {
            return Err(Error::InvalidAttention(
                "paged query/KV head geometry is incompatible",
            ));
        }
        if page_table.shape.dimensions()[0] != *batch
            || sequence_lengths.shape.dimensions() != &[*batch]
            || query_positions.shape.dimensions() != &[*batch, *query_len]
        {
            return Err(Error::InvalidAttention(
                "page table, lengths, and positions must match query batch geometry",
            ));
        }
        if page_table.shape.dimensions()[1] <= 0 {
            return Err(Error::InvalidAttention(
                "page table must contain at least one logical page",
            ));
        }
        page_table.shape.dimensions()[1]
            .checked_mul(*page_size)
            .ok_or(Error::InvalidAttention(
                "logical cache capacity exceeds the I64 position domain",
            ))?;
        if options.sliding_window.is_some_and(|window| window <= 0) {
            return Err(Error::InvalidAttention(
                "sliding window must be positive when specified",
            ));
        }
        let scale = options
            .scale
            .unwrap_or_else(|| 1.0 / (*head_dim as f64).sqrt());
        let kernel_scale = scale as f32;
        if !scale.is_finite() || scale <= 0.0 || !kernel_scale.is_finite() || kernel_scale <= 0.0 {
            return Err(Error::InvalidAttention(
                "scale must be representable as positive finite F32",
            ));
        }
        let result = self.push_value("paged_attention", query.shape);
        self.operations.push(Operation::PagedAttention {
            query: query.value,
            key_cache: key_cache.value,
            value_cache: value_cache.value,
            page_table: page_table.value,
            sequence_lengths: sequence_lengths.value,
            query_positions: query_positions.value,
            result: result.value,
            options: AttentionOptions {
                scale: Some(scale),
                ..options
            },
        });
        Ok(result)
    }

    /// Conventional `[batch, in] * [out, in] + [out]` linear layer.
    pub fn linear(
        &mut self,
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Tensor, Error> {
        self.require_rank(input, "linear input", 2)?;
        self.require_rank(weight, "linear weight", 2)?;
        let product = self.dot_general(input, weight, &[], &[], &[1], &[1])?;
        let Some(bias) = bias else {
            return Ok(product);
        };
        self.require_rank(bias, "linear bias", 1)?;
        if bias.shape.dimensions()[0] != product.shape.dimensions()[1] {
            return Err(Error::DimensionMismatch {
                left_axis: 0,
                right_axis: 1,
                left: bias.shape.dimensions()[0],
                right: product.shape.dimensions()[1],
            });
        }
        let bias = self.broadcast_in_dim(bias, product.shape, &[1])?;
        self.add(product, bias)
    }

    pub fn dot_general(
        &mut self,
        left: Tensor,
        right: Tensor,
        left_batch: &[usize],
        right_batch: &[usize],
        left_contract: &[usize],
        right_contract: &[usize],
    ) -> Result<Tensor, Error> {
        self.require_local(left)?;
        self.require_local(right)?;
        if left.shape.dtype() != right.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: left.shape.dtype(),
                right: right.shape.dtype(),
            });
        }
        self.validate_axis_pairs(left, right, left_batch, right_batch, "batch")?;
        self.validate_axis_pairs(left, right, left_contract, right_contract, "contracting")?;
        ensure_disjoint(left_batch, left_contract, "left")?;
        ensure_disjoint(right_batch, right_contract, "right")?;

        let left_unselected = unselected_axes(left.shape.rank(), left_batch, left_contract);
        let right_unselected = unselected_axes(right.shape.rank(), right_batch, right_contract);
        let result_axes = left_batch
            .iter()
            .map(|&axis| (left.shape, axis))
            .chain(left_unselected.iter().map(|&axis| (left.shape, axis)))
            .chain(right_unselected.iter().map(|&axis| (right.shape, axis)))
            .collect::<Vec<_>>();
        let dimensions = result_axes
            .iter()
            .map(|(shape, axis)| shape.dimensions()[*axis])
            .collect::<Vec<_>>();
        let axis_tags = result_axes
            .iter()
            .map(|(shape, axis)| shape.axis_tags()[*axis])
            .collect::<Vec<_>>();
        let partitions = result_axes
            .iter()
            .map(|(shape, axis)| shape.partitions()[*axis])
            .collect::<Vec<_>>();
        let shape = Shape::new(left.shape.dtype(), &dimensions)?
            .with_axis_tags(&axis_tags)?
            .with_partitions(&partitions)?;
        let result = self.push_value("dot", shape);
        self.operations.push(Operation::DotGeneral {
            left: left.value,
            right: right.value,
            result: result.value,
            left_batch: left_batch.to_vec(),
            right_batch: right_batch.to_vec(),
            left_contract: left_contract.to_vec(),
            right_contract: right_contract.to_vec(),
        });
        Ok(result)
    }

    pub fn complex(&mut self, real: Tensor, imaginary: Tensor) -> Result<Tensor, Error> {
        self.require_local(real)?;
        self.require_local(imaginary)?;
        if real.shape.dtype() != imaginary.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: real.shape.dtype(),
                right: imaginary.shape.dtype(),
            });
        }
        require_matching_shape_metadata("complex", real.shape, imaginary.shape)?;
        let dtype = match real.shape.dtype() {
            DType::F32 => DType::C64,
            DType::F64 => DType::C128,
            other => return Err(Error::UnsupportedComplexInput(other)),
        };
        let shape = real.shape.with_dtype(dtype);
        let result = self.push_value("complex", shape);
        self.operations.push(Operation::Complex {
            real: real.value,
            imaginary: imaginary.value,
            result: result.value,
        });
        Ok(result)
    }

    pub fn real(&mut self, complex: Tensor) -> Result<Tensor, Error> {
        self.complex_component(complex, Component::Real)
    }

    pub fn imaginary(&mut self, complex: Tensor) -> Result<Tensor, Error> {
        self.complex_component(complex, Component::Imaginary)
    }

    /// Adds a one-, two-, or three-dimensional StableHLO FFT operation.
    pub fn fft(&mut self, input: Tensor, kind: FftType, lengths: &[i64]) -> Result<Tensor, Error> {
        self.require_local(input)?;
        if lengths.is_empty() || lengths.len() > 3 || lengths.len() > input.shape.rank() {
            return Err(Error::InvalidFft {
                kind,
                message: "transform rank must be between one and three and fit the input rank",
            });
        }
        if lengths.iter().any(|length| *length <= 0) {
            return Err(Error::InvalidFft {
                kind,
                message: "transform lengths must be positive",
            });
        }
        let output_dtype = match (kind, input.shape.dtype()) {
            (FftType::Fft | FftType::Ifft, DType::C64) => DType::C64,
            (FftType::Fft | FftType::Ifft, DType::C128) => DType::C128,
            (FftType::Rfft, DType::F32) => DType::C64,
            (FftType::Rfft, DType::F64) => DType::C128,
            (FftType::Irfft, DType::C64) => DType::F32,
            (FftType::Irfft, DType::C128) => DType::F64,
            _ => {
                return Err(Error::InvalidFft {
                    kind,
                    message: "dtype does not match the selected real or complex transform",
                });
            }
        };

        let first_axis = input.shape.rank() - lengths.len();
        let mut output_dimensions = input.shape.dimensions().to_vec();
        for (offset, &length) in lengths.iter().enumerate() {
            let axis = first_axis + offset;
            let final_axis = offset + 1 == lengths.len();
            let expected_input = if kind == FftType::Irfft && final_axis {
                length / 2 + 1
            } else {
                length
            };
            if input.shape.dimensions()[axis] != expected_input {
                return Err(Error::InvalidFft {
                    kind,
                    message: "trailing input dimensions do not match transform lengths",
                });
            }
            output_dimensions[axis] = if kind == FftType::Rfft && final_axis {
                length / 2 + 1
            } else {
                length
            };
        }
        let shape = Shape::new(output_dtype, &output_dimensions)?
            .with_axis_tags(input.shape.axis_tags())?
            .with_partitions(input.shape.partitions())?
            .with_layout(input.shape.layout())?;
        let result = self.push_value("fft", shape);
        self.operations.push(Operation::Fft {
            input: input.value,
            result: result.value,
            kind,
            lengths: lengths.to_vec(),
        });
        Ok(result)
    }

    /// Declares that an output consumes an activation's storage.
    ///
    /// This follows ZML's `reuseBuffer`: the declaration is carried to XLA as
    /// `tf.aliasing_output`, while runtime ownership separately ensures that a
    /// shared buffer or baked parameter can never be donated.
    pub fn reuse_buffer(&mut self, output: Tensor, input: Tensor) -> Result<Tensor, Error> {
        self.require_local(output)?;
        self.require_local(input)?;
        if output.shape != input.shape {
            return Err(Error::AliasShapeMismatch {
                input: input.shape,
                output: output.shape,
            });
        }
        let input_position = self
            .inputs
            .iter()
            .position(|value| *value == input.value)
            .ok_or(Error::ForeignTensor)?;
        if self.input_kinds[input_position] != InputKind::Activation {
            return Err(Error::AliasInputIsNotAnActivation(
                self.values[input.value].name.clone(),
            ));
        }
        if self.aliases.contains_key(&output.value) {
            return Err(Error::DuplicateOutputAlias);
        }
        if self
            .aliases
            .values()
            .any(|position| *position == input_position)
        {
            return Err(Error::DuplicateInputAlias);
        }
        self.aliases.insert(output.value, input_position);
        Ok(output)
    }

    pub fn finish(self, outputs: &[Tensor]) -> Result<Program, Error> {
        let named = outputs
            .iter()
            .enumerate()
            .map(|(index, tensor)| (format!("result{index}"), *tensor))
            .collect::<Vec<_>>();
        self.finish_named(&named)
    }

    pub fn finish_named(self, outputs: &[(String, Tensor)]) -> Result<Program, Error> {
        if outputs.is_empty() {
            return Err(Error::NoOutputs);
        }
        for (_, output) in outputs {
            self.require_local(*output)?;
        }
        require_unique_names(
            self.inputs
                .iter()
                .map(|&value| self.values[value].name.as_str()),
            Error::DuplicateInputName,
        )?;
        require_unique_names(
            outputs.iter().map(|(name, _)| name.as_str()),
            Error::DuplicateOutputName,
        )?;
        // StableHLO tensor types describe logical shapes, while the current
        // PJRT transfer boundary is deliberately dense row-major. Reject a
        // physical layout we cannot preserve instead of silently compiling a
        // graph whose host and device views disagree.
        for value in &self.values {
            require_supported_layout(value.shape)?;
        }
        let output_aliases = outputs
            .iter()
            .map(|(_, tensor)| self.aliases.get(&tensor.value).copied())
            .collect();
        Ok(Program {
            inputs: self.inputs,
            input_kinds: self.input_kinds,
            values: self.values,
            operations: self.operations,
            outputs: outputs.iter().map(|(_, tensor)| tensor.value).collect(),
            output_names: outputs.iter().map(|(name, _)| name.clone()).collect(),
            output_aliases,
        })
    }

    fn complex_component(&mut self, input: Tensor, component: Component) -> Result<Tensor, Error> {
        self.require_local(input)?;
        let dtype = match input.shape.dtype() {
            DType::C64 => DType::F32,
            DType::C128 => DType::F64,
            other => return Err(Error::ExpectedComplex(other)),
        };
        let shape = input.shape.with_dtype(dtype);
        let result = self.push_value(component.name(), shape);
        self.operations.push(Operation::Component {
            input: input.value,
            result: result.value,
            component,
        });
        Ok(result)
    }

    fn binary(&mut self, left: Tensor, right: Tensor, operation: Binary) -> Result<Tensor, Error> {
        let (left, right, shape) = self.elementwise_operands(operation.name(), left, right)?;
        match operation {
            Binary::And | Binary::Or | Binary::Xor => {
                self.require_logical(left, operation.name())?;
            }
            Binary::ShiftLeft | Binary::ShiftRightArithmetic | Binary::ShiftRightLogical => {
                self.require_integer(left, operation.name())?;
            }
            Binary::Minimum | Binary::Maximum => {
                if !shape.dtype().supports_ordering() || shape.dtype() == DType::Bool {
                    return Err(Error::UnsupportedDType {
                        operation: operation.name(),
                        dtype: shape.dtype(),
                    });
                }
            }
            Binary::Remainder if shape.dtype().class() == DTypeClass::Complex => {
                return Err(Error::UnsupportedDType {
                    operation: operation.name(),
                    dtype: shape.dtype(),
                });
            }
            _ if shape.dtype() == DType::Bool => {
                return Err(Error::UnsupportedDType {
                    operation: operation.name(),
                    dtype: shape.dtype(),
                });
            }
            _ => {}
        }
        let result = self.push_value(operation.name(), shape);
        self.operations.push(Operation::Binary {
            left: left.value,
            right: right.value,
            result: result.value,
            operation,
        });
        Ok(result)
    }

    fn elementwise_operands(
        &mut self,
        operation: &'static str,
        left: Tensor,
        right: Tensor,
    ) -> Result<(Tensor, Tensor, Shape), Error> {
        self.require_local(left)?;
        self.require_local(right)?;
        if left.shape.dtype() != right.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: left.shape.dtype(),
                right: right.shape.dtype(),
            });
        }
        if left.shape.rank() == 0 && right.shape.rank() != 0 {
            let left = self.broadcast_in_dim(left, right.shape, &[])?;
            return Ok((left, right, right.shape));
        }
        if right.shape.rank() == 0 && left.shape.rank() != 0 {
            let right = self.broadcast_in_dim(right, left.shape, &[])?;
            return Ok((left, right, left.shape));
        }
        require_matching_shape_metadata(operation, left.shape, right.shape)?;
        Ok((left, right, left.shape))
    }

    fn compare(
        &mut self,
        left: Tensor,
        right: Tensor,
        comparison: Comparison,
    ) -> Result<Tensor, Error> {
        let (left, right, shape) = self.elementwise_operands("compare", left, right)?;
        if matches!(shape.dtype(), DType::C64 | DType::C128) {
            return Err(Error::UnsupportedDType {
                operation: "compare",
                dtype: shape.dtype(),
            });
        }
        let result = self.push_value("compare", shape.with_dtype(DType::Bool));
        self.operations.push(Operation::Compare {
            left: left.value,
            right: right.value,
            result: result.value,
            comparison,
            input_dtype: shape.dtype(),
        });
        Ok(result)
    }

    fn unary(&mut self, input: Tensor, operation: Unary) -> Result<Tensor, Error> {
        self.require_local(input)?;
        let result = self.push_value(operation.name(), input.shape);
        self.operations.push(Operation::Unary {
            input: input.value,
            result: result.value,
            operation,
        });
        Ok(result)
    }

    fn float_unary(&mut self, input: Tensor, operation: Unary) -> Result<Tensor, Error> {
        self.require_float(input, operation.name())?;
        self.unary(input, operation)
    }

    fn integer_unary(&mut self, input: Tensor, operation: Unary) -> Result<Tensor, Error> {
        self.require_integer(input, operation.name())?;
        self.unary(input, operation)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_gated(
        &mut self,
        hidden: Tensor,
        router_logits: Tensor,
        gate_up_weights: Tensor,
        down_weights: Tensor,
        experts_per_token: usize,
        activation: MoeActivation,
    ) -> Result<Tensor, Error> {
        for tensor in [hidden, router_logits, gate_up_weights, down_weights] {
            self.require_local(tensor)?;
            self.require_float(tensor, "mixture_of_experts")?;
        }
        if hidden.shape.rank() != 2
            || router_logits.shape.rank() != 2
            || gate_up_weights.shape.rank() != 3
            || down_weights.shape.rank() != 3
        {
            return Err(Error::InvalidMoe(
                "expected hidden [tokens, hidden], router [tokens, experts], gate/up [experts, 2*intermediate, hidden], and down [experts, hidden, intermediate]",
            ));
        }
        if hidden.shape.dtype() != gate_up_weights.shape.dtype()
            || hidden.shape.dtype() != down_weights.shape.dtype()
        {
            return Err(Error::InvalidMoe(
                "hidden, gate/up, and down tensors must use one dtype",
            ));
        }
        let tokens = hidden.shape.dimensions()[0];
        let hidden_size = hidden.shape.dimensions()[1];
        let expert_count = router_logits.shape.dimensions()[1];
        let gate_up_width = gate_up_weights.shape.dimensions()[1];
        if tokens != router_logits.shape.dimensions()[0]
            || expert_count != gate_up_weights.shape.dimensions()[0]
            || expert_count != down_weights.shape.dimensions()[0]
            || gate_up_weights.shape.dimensions()[2] != hidden_size
            || down_weights.shape.dimensions()[1] != hidden_size
            || gate_up_width % 2 != 0
            || down_weights.shape.dimensions()[2] != gate_up_width / 2
        {
            return Err(Error::InvalidMoe("inconsistent MoE tensor dimensions"));
        }
        let experts = usize::try_from(expert_count)
            .map_err(|_| Error::InvalidMoe("expert count does not fit usize"))?;
        let experts_per_token_i64 = i64::try_from(experts_per_token)
            .map_err(|_| Error::InvalidMoe("experts per token exceeds I64"))?;
        if experts_per_token == 0 || experts_per_token > experts {
            return Err(Error::InvalidMoe(
                "experts per token must be positive and not exceed the expert count",
            ));
        }
        if hidden.shape.axis_tags()[0] != router_logits.shape.axis_tags()[0]
            || hidden.shape.partitions()[0] != router_logits.shape.partitions()[0]
        {
            return Err(Error::InvalidMoe(
                "hidden and router token axes must carry identical metadata",
            ));
        }
        if gate_up_weights.shape.partitions()[0] != down_weights.shape.partitions()[0] {
            return Err(Error::InvalidMoe(
                "gate/up and down expert axes must use identical partitioning",
            ));
        }
        if matches!(router_logits.shape.partitions()[1], Partition::Sharded(_)) {
            return Err(Error::InvalidMoe(
                "router logits stay replicated so top-k selection is global",
            ));
        }

        let probabilities = self.softmax(router_logits, 1)?;
        let (routing_weights, expert_ids) =
            self.top_k(probabilities, 1, experts_per_token, true)?;
        let normalizer = self.reduce_sum(routing_weights, &[1])?;
        let epsilon = self.scalar_for(
            routing_weights.shape.dtype(),
            if routing_weights.shape.dtype() == DType::F64 {
                f64::EPSILON
            } else {
                f64::from(f32::EPSILON)
            },
        )?;
        let normalizer = self.maximum(normalizer, epsilon)?;
        let normalizer = self.broadcast_in_dim(normalizer, routing_weights.shape, &[0])?;
        let routing_weights = self.divide(routing_weights, normalizer)?;
        let routing_weights = self.convert(routing_weights, hidden.shape.dtype())?;
        let assignment_plan = self.moe_assignment_plan(expert_ids, experts, 16)?;

        // Keep the expert dimension explicit throughout the portable graph.
        // Besides avoiding one graph copy per expert, this lets Shardy tile
        // both projections on that dimension and introduce only the final
        // cross-expert reduction.
        let intermediate = gate_up_width / 2;
        let projected = self.dot_general(hidden, gate_up_weights, &[], &[], &[1], &[2])?;
        let halves = self.split(projected, 2, &[intermediate, intermediate])?;
        let gate = match activation {
            MoeActivation::Silu => self.silu(halves[0])?,
            MoeActivation::Gelu => self.gelu(halves[0])?,
            MoeActivation::Relu => self.relu(halves[0])?,
        };
        let activated = self.multiply(gate, halves[1])?;
        let expert_outputs = self.dot_general(activated, down_weights, &[1], &[0], &[2], &[2])?;
        let expert_outputs = self.transpose(expert_outputs, &[1, 0, 2])?;

        // Expand the sparse top-k result to a dense token/expert weight map.
        // Routing remains globally replicated until after selection. The
        // broadcast result introduces the weights' expert partition, so the
        // dense map and expert projections carry identical metadata.
        let expert_partition = gate_up_weights.shape.partitions()[0];
        let expert_tag = gate_up_weights.shape.axis_tags()[0];
        let expert_ids_shape = Shape::new(DType::I32, &[expert_count])?
            .with_axis_tags(&[expert_tag])?
            .with_partitions(&[expert_partition])?;
        let all_expert_ids = self.iota(expert_ids_shape, 0)?;
        let selection_shape =
            Shape::new(DType::I32, &[tokens, experts_per_token_i64, expert_count])?
                .with_axis_tags(&[
                    expert_ids.shape.axis_tags()[0],
                    expert_ids.shape.axis_tags()[1],
                    expert_tag,
                ])?
                .with_partitions(&[
                    expert_ids.shape.partitions()[0],
                    expert_ids.shape.partitions()[1],
                    expert_partition,
                ])?;
        let all_expert_ids = self.broadcast_in_dim(all_expert_ids, selection_shape, &[2])?;
        let selected_ids = self.broadcast_in_dim(expert_ids, selection_shape, &[0, 1])?;
        let selected = self.equal(selected_ids, all_expert_ids)?;
        let routing_shape = selection_shape.with_dtype(hidden.shape.dtype());
        let selected_weights = self.broadcast_in_dim(routing_weights, routing_shape, &[0, 1])?;
        let zero = self.scalar_for(hidden.shape.dtype(), 0.0)?;
        let selected_weights = self.select(selected, selected_weights, zero)?;
        let dense_routing = self.reduce_sum(selected_weights, &[1])?;
        let routing = self.broadcast_in_dim(dense_routing, expert_outputs.shape, &[0, 1])?;
        let weighted = self.multiply(expert_outputs, routing)?;
        let portable_output = self.reduce_sum(weighted, &[1])?;
        let result = self.push_value("mixture_of_experts", portable_output.shape);
        self.operations.push(Operation::MoeDispatch {
            hidden: hidden.value,
            routing_weights: routing_weights.value,
            gate_up_weights: gate_up_weights.value,
            down_weights: down_weights.value,
            sorted_assignments: assignment_plan.sorted_assignments.value,
            block_experts: assignment_plan.block_experts.value,
            portable_output: portable_output.value,
            result: result.value,
            activation,
            experts_per_token,
            block_size: assignment_plan.block_size,
        });
        Ok(result)
    }

    /// Builds a stable, bounded token-to-expert schedule in ordinary StableHLO.
    /// Triton therefore owns only the grouped matrix multiplication, while the
    /// routing semantics remain shared by every backend and independently
    /// visible to the compiler.
    fn moe_assignment_plan(
        &mut self,
        expert_ids: Tensor,
        experts: usize,
        block_size: usize,
    ) -> Result<MoeAssignmentPlan, Error> {
        let assignments = expert_ids.shape.element_count().map_err(Error::Shape)?;
        let assignments = i64::try_from(assignments)
            .map_err(|_| Error::InvalidMoe("assignment count exceeds the shape contract"))?;
        let expert_count = i64::try_from(experts)
            .map_err(|_| Error::InvalidMoe("expert count exceeds the shape contract"))?;
        let block_size_i64 = i64::try_from(block_size)
            .ok()
            .filter(|value| *value > 0)
            .ok_or(Error::InvalidMoe("expert block size must be positive"))?;
        let max_blocks = assignments
            .checked_add(block_size_i64 - 1)
            .and_then(|value| value.checked_div(block_size_i64))
            .and_then(|value| value.checked_add(expert_count))
            .ok_or(Error::InvalidMoe("padded assignment schedule is too large"))?;
        let max_positions = max_blocks
            .checked_mul(block_size_i64)
            .ok_or(Error::InvalidMoe("padded assignment schedule is too large"))?;
        if assignments > i64::from(i32::MAX)
            || max_positions > i64::from(i32::MAX)
            || max_blocks > i64::from(i32::MAX)
        {
            return Err(Error::InvalidMoe(
                "MoE routing schedule must be addressable by i32 indices",
            ));
        }

        let flat_ids = self.reshape(expert_ids, Shape::new(DType::I32, &[assignments])?)?;
        let assignment_ids = self.iota(Shape::new(DType::I32, &[assignments])?, 0)?;
        let block_slots = self.iota(Shape::new(DType::I32, &[max_blocks])?, 0)?;
        let sentinel = self.scalar(i32::try_from(assignments).map_err(|_| {
            Error::InvalidMoe("assignment count must fit the I32 routing contract")
        })?)?;
        let mut sorted_assignments =
            self.broadcast_in_dim(sentinel, Shape::new(DType::I32, &[max_positions])?, &[])?;
        let invalid_expert = self.scalar(-1_i32)?;
        let mut block_experts =
            self.broadcast_in_dim(invalid_expert, Shape::new(DType::I32, &[max_blocks])?, &[])?;
        let mut padded_prefix = self.scalar(0_i32)?;
        let one = self.scalar(1_i32)?;
        let block = self.scalar(i32::try_from(block_size).map_err(|_| {
            Error::InvalidMoe("expert block size must fit the I32 routing contract")
        })?)?;
        let block_minus_one = self.scalar(i32::try_from(block_size - 1).map_err(|_| {
            Error::InvalidMoe("expert block size must fit the I32 routing contract")
        })?)?;
        let dropped_index = self.scalar(i32::try_from(max_positions).map_err(|_| {
            Error::InvalidMoe("padded schedule must fit the I32 routing contract")
        })?)?;

        for expert in 0..experts {
            let expert = self
                .scalar(i32::try_from(expert).map_err(|_| {
                    Error::InvalidMoe("expert id exceeds the I32 routing contract")
                })?)?;
            let assigned = self.equal(flat_ids, expert)?;
            let assigned_i32 = self.convert(assigned, DType::I32)?;
            let positions = self.cumulative_sum(assigned_i32, 0)?;
            let positions = self.subtract(positions, one)?;
            let count = self.reduce_sum(assigned_i32, &[0])?;
            let padded = self.add(count, block_minus_one)?;
            let padded = self.divide(padded, block)?;
            let padded = self.multiply(padded, block)?;

            let targets = self.add(padded_prefix, positions)?;
            let targets = self.select(assigned, targets, dropped_index)?;
            let targets = self.insert_axis(targets, 1, AxisTag::UNKNOWN)?;
            sorted_assignments =
                self.scatter_update(sorted_assignments, targets, assignment_ids, &[0])?;

            let block_start = self.divide(padded_prefix, block)?;
            let block_count = self.divide(padded, block)?;
            let block_end = self.add(block_start, block_count)?;
            let after_start = self.greater_equal(block_slots, block_start)?;
            let before_end = self.less(block_slots, block_end)?;
            let owns_block = self.logical_and(after_start, before_end)?;
            block_experts = self.select(owns_block, expert, block_experts)?;
            padded_prefix = self.add(padded_prefix, padded)?;
        }

        Ok(MoeAssignmentPlan {
            sorted_assignments,
            block_experts,
            block_size,
        })
    }

    fn reduce(
        &mut self,
        input: Tensor,
        axes: &[usize],
        reduction: Reduction,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(axes, input.shape.rank(), reduction.name())?;
        match reduction {
            Reduction::Sum if input.shape.dtype() == DType::Bool => {
                return Err(Error::UnsupportedDType {
                    operation: reduction.name(),
                    dtype: input.shape.dtype(),
                });
            }
            Reduction::Maximum | Reduction::Minimum
                if !input.shape.dtype().supports_ordering()
                    || input.shape.dtype() == DType::Bool =>
            {
                return Err(Error::UnsupportedDType {
                    operation: reduction.name(),
                    dtype: input.shape.dtype(),
                });
            }
            _ => {}
        }
        let shape = reduced_shape(input.shape, axes, input.shape.dtype())?;
        let init = match reduction {
            Reduction::Sum => self.zero_for(input.shape.dtype())?,
            Reduction::Maximum => self.minimum_for(input.shape.dtype())?,
            Reduction::Minimum => self.maximum_for(input.shape.dtype())?,
        };
        let result = self.push_value(reduction.name(), shape);
        self.operations.push(Operation::Reduce {
            input: input.value,
            init: init.value,
            result: result.value,
            axes: axes.to_vec(),
            reduction,
        });
        Ok(result)
    }

    fn all_reduce(&mut self, input: Tensor, reduction: Reduction) -> Result<Tensor, Error> {
        self.require_local(input)?;
        match reduction {
            Reduction::Sum if input.shape.dtype() == DType::Bool => {
                return Err(Error::UnsupportedDType {
                    operation: "all_reduce_sum",
                    dtype: input.shape.dtype(),
                });
            }
            Reduction::Maximum | Reduction::Minimum
                if input.shape.dtype() == DType::Bool
                    || !input.shape.dtype().supports_ordering() =>
            {
                return Err(Error::UnsupportedDType {
                    operation: "ordered all_reduce",
                    dtype: input.shape.dtype(),
                });
            }
            _ => {}
        }
        if input
            .shape
            .partitions()
            .iter()
            .any(|partition| matches!(partition, Partition::Sharded(_)))
        {
            return Err(Error::InvalidCollective(
                "all-reduce requires equal local tensor shapes; enter a manual computation before reducing sharded dimensions",
            ));
        }
        let partitions = vec![Partition::Replicated; input.shape.rank()];
        let shape = input.shape.with_partitions(&partitions)?;
        let result = self.push_value("all_reduce", shape);
        self.operations.push(Operation::AllReduce {
            input: input.value,
            result: result.value,
            reduction,
        });
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn reduce_window(
        &mut self,
        input: Tensor,
        window_dimensions: &[i64],
        window_strides: &[i64],
        base_dilations: &[i64],
        window_dilations: &[i64],
        padding: &[[i64; 2]],
        reduction: Reduction,
    ) -> Result<Tensor, Error> {
        self.require_local(input)?;
        let shape = reduce_window_output_shape(
            input.shape,
            window_dimensions,
            window_strides,
            base_dilations,
            window_dilations,
            padding,
        )?;
        match reduction {
            Reduction::Sum if input.shape.dtype() == DType::Bool => {
                return Err(Error::UnsupportedDType {
                    operation: "reduce_window_sum",
                    dtype: input.shape.dtype(),
                });
            }
            Reduction::Maximum | Reduction::Minimum
                if input.shape.dtype() == DType::Bool
                    || !input.shape.dtype().supports_ordering() =>
            {
                return Err(Error::UnsupportedDType {
                    operation: "ordered reduce_window",
                    dtype: input.shape.dtype(),
                });
            }
            _ => {}
        }
        let init = match reduction {
            Reduction::Sum => self.zero_for(input.shape.dtype())?,
            Reduction::Maximum => self.minimum_for(input.shape.dtype())?,
            Reduction::Minimum => self.maximum_for(input.shape.dtype())?,
        };
        let result = self.push_value("reduce_window", shape);
        self.operations.push(Operation::ReduceWindow {
            input: input.value,
            init: init.value,
            result: result.value,
            window_dimensions: window_dimensions.to_vec(),
            window_strides: window_strides.to_vec(),
            base_dilations: base_dilations.to_vec(),
            window_dilations: window_dilations.to_vec(),
            padding: padding.to_vec(),
            reduction,
        });
        Ok(result)
    }

    fn require_resize(
        &self,
        input: Tensor,
        axis: usize,
        new_length: i64,
        operation: &'static str,
    ) -> Result<(), Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), operation)?;
        let old_length = input.shape.dimensions()[axis];
        if old_length <= 0 || old_length > i64::from(i32::MAX) {
            return Err(Error::InvalidResize(
                "resized axes must be nonempty and addressable by i32 indices",
            ));
        }
        if new_length <= 0 || new_length > i64::from(i32::MAX) {
            return Err(Error::InvalidResize(
                "resized axis length must be positive and fit i32",
            ));
        }
        Ok(())
    }

    fn require_distribution_shape(
        &self,
        shape: Shape,
        operation: &'static str,
    ) -> Result<(), Error> {
        require_supported_layout(shape)?;
        if !matches!(
            shape.dtype(),
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64
        ) {
            return Err(Error::UnsupportedDType {
                operation,
                dtype: shape.dtype(),
            });
        }
        Ok(())
    }

    fn require_sampling_scalar(
        &self,
        value: Tensor,
        dtype: DType,
        name: &'static str,
    ) -> Result<(), Error> {
        self.require_local(value)?;
        if value.shape.rank() != 0 || value.shape.dtype() != dtype {
            return Err(Error::InvalidSampling(match name {
                "top_k" => "top-k must be a scalar I32 tensor",
                _ => "sampling option has the wrong scalar dtype",
            }));
        }
        Ok(())
    }

    fn move_axis_to_last(&mut self, input: Tensor, axis: usize) -> Result<Tensor, Error> {
        self.require_local(input)?;
        validate_axes(&[axis], input.shape.rank(), "move_axis_to_last")?;
        if axis + 1 == input.shape.rank() {
            return Ok(input);
        }
        let permutation = (0..input.shape.rank())
            .filter(|candidate| *candidate != axis)
            .chain(std::iter::once(axis))
            .collect::<Vec<_>>();
        self.transpose(input, &permutation)
    }

    fn gather_resized_axis(
        &mut self,
        input: Tensor,
        axis: usize,
        indices: &[i32],
    ) -> Result<Tensor, Error> {
        let index_shape = Shape::new(DType::I32, &[indices.len() as i64])?
            .with_axis_tags(&[input.shape.axis_tags()[axis]])?
            .with_partitions(&[input.shape.partitions()[axis]])?;
        let index_slice = Slice::from_typed(index_shape, indices)?;
        let indices = self.constant(&index_slice)?;
        let gathered = self.gather(input, indices, axis)?;
        let permutation = (0..input.shape.rank())
            .map(|output_axis| {
                if output_axis == axis {
                    0
                } else if output_axis < axis {
                    output_axis + 1
                } else {
                    output_axis
                }
            })
            .collect::<Vec<_>>();
        self.transpose(gathered, &permutation)
    }

    fn interpolation_weights(
        &mut self,
        result_shape: Shape,
        axis: usize,
        values: &[f64],
    ) -> Result<Tensor, Error> {
        let vector_shape = Shape::new(result_shape.dtype(), &[values.len() as i64])?
            .with_axis_tags(&[result_shape.axis_tags()[axis]])?
            .with_partitions(&[result_shape.partitions()[axis]])?;
        let vector = match result_shape.dtype() {
            DType::F32 => {
                let values = values.iter().map(|value| *value as f32).collect::<Vec<_>>();
                self.constant(&Slice::from_typed(vector_shape, &values)?)?
            }
            DType::F64 => self.constant(&Slice::from_typed(vector_shape, values)?)?,
            dtype => {
                return Err(Error::UnsupportedDType {
                    operation: "interpolation weights",
                    dtype,
                });
            }
        };
        self.broadcast_in_dim(vector, result_shape, &[axis])
    }

    fn broadcast_reduction(
        &mut self,
        input: Tensor,
        result_shape: Shape,
        reduced_axes: &[usize],
    ) -> Result<Tensor, Error> {
        let dimensions = (0..result_shape.rank())
            .filter(|axis| !reduced_axes.contains(axis))
            .collect::<Vec<_>>();
        self.broadcast_in_dim(input, result_shape, &dimensions)
    }

    fn validate_dynamic_starts(&self, starts: &[Tensor], rank: usize) -> Result<(), Error> {
        if starts.len() != rank {
            return Err(Error::AxisCountMismatch);
        }
        let mut dtype = None;
        for start in starts {
            self.require_local(*start)?;
            if start.shape.rank() != 0 {
                return Err(Error::RankMismatch {
                    operation: "dynamic slice start",
                    expected: 0,
                    actual: start.shape.rank(),
                });
            }
            require_index_dtype(start.shape.dtype())?;
            if let Some(dtype) = dtype {
                if dtype != start.shape.dtype() {
                    return Err(Error::DTypeMismatch {
                        left: dtype,
                        right: start.shape.dtype(),
                    });
                }
            } else {
                dtype = Some(start.shape.dtype());
            }
        }
        Ok(())
    }

    fn require_float(&self, input: Tensor, operation: &'static str) -> Result<(), Error> {
        self.require_local(input)?;
        if input.shape.dtype().class() == DTypeClass::Float {
            Ok(())
        } else {
            Err(Error::UnsupportedDType {
                operation,
                dtype: input.shape.dtype(),
            })
        }
    }

    fn require_logical(&self, input: Tensor, operation: &'static str) -> Result<(), Error> {
        self.require_local(input)?;
        match input.shape.dtype().class() {
            DTypeClass::Boolean | DTypeClass::SignedInteger | DTypeClass::UnsignedInteger => Ok(()),
            DTypeClass::Float | DTypeClass::Complex => Err(Error::UnsupportedDType {
                operation,
                dtype: input.shape.dtype(),
            }),
        }
    }

    fn require_integer(&self, input: Tensor, operation: &'static str) -> Result<(), Error> {
        self.require_local(input)?;
        match input.shape.dtype().class() {
            DTypeClass::SignedInteger | DTypeClass::UnsignedInteger => Ok(()),
            DTypeClass::Boolean | DTypeClass::Float | DTypeClass::Complex => {
                Err(Error::UnsupportedDType {
                    operation,
                    dtype: input.shape.dtype(),
                })
            }
        }
    }

    fn float_accumulation(&mut self, input: Tensor) -> Result<Tensor, Error> {
        if matches!(input.shape.dtype(), DType::F16 | DType::Bf16) {
            self.convert(input, DType::F32)
        } else {
            Ok(input)
        }
    }

    fn validate_normalization_axis(
        &self,
        input: Tensor,
        axis: usize,
        epsilon: f64,
    ) -> Result<(), Error> {
        if axis >= input.shape.rank() {
            return Err(Error::AxisOutOfBounds {
                side: "normalization",
                axis,
                rank: input.shape.rank(),
            });
        }
        if input.shape.dimensions()[axis] == 0 {
            return Err(Error::InvalidNormalization(
                "normalization dimension must be nonempty",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidNormalization(
                "epsilon must be finite and positive",
            ));
        }
        Ok(())
    }

    fn normalization_parameter(
        &mut self,
        input: Tensor,
        parameter: Tensor,
        axis: usize,
        operation: &'static str,
    ) -> Result<Tensor, Error> {
        self.require_local(parameter)?;
        self.require_rank(parameter, operation, 1)?;
        if parameter.shape.dtype() != input.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: input.shape.dtype(),
                right: parameter.shape.dtype(),
            });
        }
        if parameter.shape.dimensions()[0] != input.shape.dimensions()[axis] {
            return Err(Error::DimensionMismatch {
                left_axis: 0,
                right_axis: axis,
                left: parameter.shape.dimensions()[0],
                right: input.shape.dimensions()[axis],
            });
        }
        self.broadcast_in_dim(parameter, input.shape, &[axis])
    }

    fn scalar_for(&mut self, dtype: DType, value: f64) -> Result<Tensor, Error> {
        match dtype {
            DType::F16 => self.scalar(F16::from_f32(value as f32)),
            DType::Bf16 => self.scalar(BFloat16::from_f32(value as f32)),
            DType::F32 => self.scalar(value as f32),
            DType::F64 => self.scalar(value),
            _ => Err(Error::UnsupportedDType {
                operation: "floating-point scalar",
                dtype,
            }),
        }
    }

    fn zero_for(&mut self, dtype: DType) -> Result<Tensor, Error> {
        match dtype {
            DType::Bool => self.scalar(false),
            DType::I8 => self.scalar(0i8),
            DType::I16 => self.scalar(0i16),
            DType::I32 => self.scalar(0i32),
            DType::I64 => self.scalar(0i64),
            DType::U8 => self.scalar(0u8),
            DType::U16 => self.scalar(0u16),
            DType::U32 => self.scalar(0u32),
            DType::U64 => self.scalar(0u64),
            DType::F16 => self.scalar(F16::from_f32(0.0)),
            DType::Bf16 => self.scalar(BFloat16::from_f32(0.0)),
            DType::F32 => self.scalar(0.0f32),
            DType::F64 => self.scalar(0.0f64),
            DType::C64 => self.scalar(Complex64 {
                real: 0.0,
                imaginary: 0.0,
            }),
            DType::C128 => self.scalar(Complex128 {
                real: 0.0,
                imaginary: 0.0,
            }),
        }
    }

    fn minimum_for(&mut self, dtype: DType) -> Result<Tensor, Error> {
        match dtype {
            DType::I8 => self.scalar(i8::MIN),
            DType::I16 => self.scalar(i16::MIN),
            DType::I32 => self.scalar(i32::MIN),
            DType::I64 => self.scalar(i64::MIN),
            DType::U8 => self.scalar(u8::MIN),
            DType::U16 => self.scalar(u16::MIN),
            DType::U32 => self.scalar(u32::MIN),
            DType::U64 => self.scalar(u64::MIN),
            DType::F16 => self.scalar(F16::from_f32(f32::NEG_INFINITY)),
            DType::Bf16 => self.scalar(BFloat16::from_f32(f32::NEG_INFINITY)),
            DType::F32 => self.scalar(f32::NEG_INFINITY),
            DType::F64 => self.scalar(f64::NEG_INFINITY),
            _ => Err(Error::UnsupportedDType {
                operation: "reduce_max",
                dtype,
            }),
        }
    }

    fn maximum_for(&mut self, dtype: DType) -> Result<Tensor, Error> {
        match dtype {
            DType::I8 => self.scalar(i8::MAX),
            DType::I16 => self.scalar(i16::MAX),
            DType::I32 => self.scalar(i32::MAX),
            DType::I64 => self.scalar(i64::MAX),
            DType::U8 => self.scalar(u8::MAX),
            DType::U16 => self.scalar(u16::MAX),
            DType::U32 => self.scalar(u32::MAX),
            DType::U64 => self.scalar(u64::MAX),
            DType::F16 => self.scalar(F16::from_f32(f32::INFINITY)),
            DType::Bf16 => self.scalar(BFloat16::from_f32(f32::INFINITY)),
            DType::F32 => self.scalar(f32::INFINITY),
            DType::F64 => self.scalar(f64::INFINITY),
            _ => Err(Error::UnsupportedDType {
                operation: "reduce_min",
                dtype,
            }),
        }
    }

    fn validate_axis_pairs(
        &self,
        left: Tensor,
        right: Tensor,
        left_axes: &[usize],
        right_axes: &[usize],
        kind: &'static str,
    ) -> Result<(), Error> {
        if left_axes.len() != right_axes.len() {
            return Err(Error::AxisCountMismatch);
        }
        validate_axes(left_axes, left.shape.rank(), "left")?;
        validate_axes(right_axes, right.shape.rank(), "right")?;
        for (&left_axis, &right_axis) in left_axes.iter().zip(right_axes) {
            let left_dimension = left.shape.dimensions()[left_axis];
            let right_dimension = right.shape.dimensions()[right_axis];
            if left_dimension != right_dimension {
                return Err(Error::DimensionMismatch {
                    left_axis,
                    right_axis,
                    left: left_dimension,
                    right: right_dimension,
                });
            }
            if kind == "batch"
                && (left.shape.axis_tags()[left_axis] != right.shape.axis_tags()[right_axis]
                    || left.shape.partitions()[left_axis] != right.shape.partitions()[right_axis])
            {
                return Err(Error::MetadataMismatch {
                    operation: "dot_general",
                    field: "batch-axis metadata",
                });
            }
        }
        Ok(())
    }

    fn require_rank(
        &self,
        tensor: Tensor,
        operation: &'static str,
        expected: usize,
    ) -> Result<(), Error> {
        self.require_local(tensor)?;
        if tensor.shape.rank() == expected {
            Ok(())
        } else {
            Err(Error::RankMismatch {
                operation,
                expected,
                actual: tensor.shape.rank(),
            })
        }
    }

    fn require_local(&self, tensor: Tensor) -> Result<(), Error> {
        if tensor.program == self.identifier && tensor.value < self.values.len() {
            require_supported_layout(tensor.shape)
        } else {
            Err(Error::ForeignTensor)
        }
    }

    fn push_value(&mut self, name: &'static str, shape: Shape) -> Tensor {
        let value = self.values.len();
        self.values.push(Value {
            name: name.to_owned(),
            shape,
        });
        self.tensor(value)
    }

    fn tensor(&self, value: usize) -> Tensor {
        Tensor {
            program: self.identifier,
            value,
            shape: self.values[value].shape,
        }
    }
}

fn require_supported_layout(shape: Shape) -> Result<(), Error> {
    let expected = Layout::row_major(shape.rank())?;
    let actual = shape.layout();
    if actual == expected {
        Ok(())
    } else {
        Err(Error::UnsupportedLayout { actual, expected })
    }
}

fn require_index_dtype(dtype: DType) -> Result<(), Error> {
    match dtype.class() {
        DTypeClass::SignedInteger | DTypeClass::UnsignedInteger => Ok(()),
        _ => Err(Error::InvalidIndexDType(dtype)),
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn rope_frequencies(options: RopeOptions) -> Result<Vec<f32>, Error> {
    let half = options.rotary_dimensions / 2;
    let mut frequencies = (0..half)
        .map(|index| options.base.powf(-(index as f64) / half as f64))
        .collect::<Vec<_>>();
    match options.scaling {
        RopeScaling::Default => {}
        RopeScaling::Linear { factor } => {
            if !factor.is_finite() || factor <= 0.0 {
                return Err(Error::InvalidRope(
                    "linear scaling factor must be finite and positive",
                ));
            }
            for frequency in &mut frequencies {
                *frequency /= factor;
            }
        }
        RopeScaling::Proportional { rotary_fraction } => {
            if !rotary_fraction.is_finite() || !(0.0..=1.0).contains(&rotary_fraction) {
                return Err(Error::InvalidRope(
                    "proportional rotary fraction must be between zero and one",
                ));
            }
            let retained = (half as f64 * rotary_fraction).floor() as usize;
            frequencies[retained..].fill(0.0);
        }
        RopeScaling::Llama3 {
            factor,
            high_frequency_factor,
            low_frequency_factor,
            original_context,
            truncate,
        } => {
            validate_extended_rope(
                factor,
                high_frequency_factor,
                low_frequency_factor,
                original_context,
            )?;
            let context = original_context as f64;
            let high = high_frequency_factor * std::f64::consts::TAU / context;
            let low = low_frequency_factor * std::f64::consts::TAU / context;
            let downscale = factor.recip();
            for frequency in &mut frequencies {
                if *frequency < low {
                    *frequency *= downscale;
                } else if *frequency <= high {
                    let mut interpolation = (*frequency - low) / (high - low);
                    if truncate {
                        interpolation = interpolation.clamp(0.0, 1.0);
                    }
                    *frequency *= interpolation + (1.0 - interpolation) * downscale;
                }
            }
        }
        RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_context,
            truncate,
        } => {
            validate_extended_rope(factor, beta_fast, beta_slow, original_context)?;
            let context = original_context as f64;
            let high = beta_fast * std::f64::consts::TAU / context;
            let low = beta_slow * std::f64::consts::TAU / context;
            let mut low_index = -high.ln() / options.base.ln() * half as f64;
            let mut high_index = -low.ln() / options.base.ln() * half as f64;
            if truncate {
                low_index = low_index.floor();
                high_index = high_index.ceil();
            }
            if high_index <= low_index {
                return Err(Error::InvalidRope("Yarn transition interval is empty"));
            }
            let downscale = factor.recip();
            for (index, frequency) in frequencies.iter_mut().enumerate() {
                if *frequency < low {
                    *frequency *= downscale;
                } else if *frequency <= high {
                    let interpolation = (high_index - index as f64) / (high_index - low_index);
                    *frequency *= interpolation + (1.0 - interpolation) * downscale;
                }
            }
        }
    }
    Ok(frequencies.into_iter().map(|value| value as f32).collect())
}

fn validate_extended_rope(
    factor: f64,
    high: f64,
    low: f64,
    original_context: usize,
) -> Result<(), Error> {
    if !factor.is_finite()
        || factor <= 0.0
        || !high.is_finite()
        || !low.is_finite()
        || low <= 0.0
        || low >= high
        || original_context == 0
    {
        Err(Error::InvalidRope(
            "extended scaling requires a positive factor/context and ordered frequencies",
        ))
    } else {
        Ok(())
    }
}

pub struct Program {
    inputs: Vec<usize>,
    input_kinds: Vec<InputKind>,
    values: Vec<Value>,
    operations: Vec<Operation>,
    outputs: Vec<usize>,
    output_names: Vec<String>,
    output_aliases: Vec<Option<usize>>,
}

impl Program {
    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs
            .iter()
            .map(|value| self.values[*value].name.as_str())
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, Shape, InputKind)> {
        self.inputs.iter().enumerate().map(|(index, value)| {
            (
                self.values[*value].name.as_str(),
                self.values[*value].shape,
                self.input_kinds[index],
            )
        })
    }

    pub fn outputs(&self) -> impl Iterator<Item = (&str, Shape)> {
        self.outputs
            .iter()
            .enumerate()
            .map(|(index, value)| (self.output_names[index].as_str(), self.values[*value].shape))
    }

    pub fn output_aliases(&self) -> impl Iterator<Item = Option<usize>> + '_ {
        self.output_aliases.iter().copied()
    }

    /// Returns canonical text for single-device lowering.
    pub fn stablehlo(&self) -> Result<String, Error> {
        self.stablehlo_with_sharding(&Sharding::single())
    }

    /// Returns canonical text for the topology-specific module sent to XLA.
    pub fn stablehlo_with_sharding(&self, sharding: &Sharding) -> Result<String, Error> {
        let context = Context::new();
        Ok(self.module_with_sharding(&context, sharding)?.text())
    }

    pub fn module<'context>(&self, context: &'context Context) -> Result<Module<'context>, Error> {
        self.module_with_sharding(context, &Sharding::single())
    }

    pub fn module_with_sharding<'context>(
        &self,
        context: &'context Context,
        sharding: &Sharding,
    ) -> Result<Module<'context>, Error> {
        self.module_with_sharding_target(context, sharding, None)
    }

    #[doc(hidden)]
    pub fn module_with_sharding_cuda<'context>(
        &self,
        context: &'context Context,
        sharding: &Sharding,
        core_count: usize,
        capability_major: u16,
        capability_minor: u16,
    ) -> Result<Module<'context>, Error> {
        self.module_with_sharding_target(
            context,
            sharding,
            Some((core_count, capability_major, capability_minor)),
        )
    }

    fn module_with_sharding_target<'context>(
        &self,
        context: &'context Context,
        sharding: &Sharding,
        cuda: Option<(usize, u16, u16)>,
    ) -> Result<Module<'context>, Error> {
        for value in &self.values {
            sharding.validate_shape(value.shape)?;
        }
        let types = self
            .values
            .iter()
            .map(|value| context.ranked_tensor_type(value.shape.dtype(), value.shape.dimensions()))
            .collect::<Result<Vec<_>, _>>()?;
        let input_types = self
            .inputs
            .iter()
            .map(|value| types[*value])
            .collect::<Vec<_>>();
        let result_types = self
            .outputs
            .iter()
            .map(|value| types[*value])
            .collect::<Vec<_>>();
        let mut block = Block::new(context, &input_types)?;
        let mut values: Vec<Option<MlirValue<'context>>> = vec![None; self.values.len()];
        for (argument, &value) in self.inputs.iter().enumerate() {
            values[value] = Some(block.argument(argument)?);
        }

        for (operation_index, operation) in self.operations.iter().enumerate() {
            if let Operation::MoeDispatch {
                hidden,
                routing_weights,
                gate_up_weights,
                down_weights,
                sorted_assignments,
                block_experts,
                portable_output,
                result,
                activation,
                experts_per_token,
                block_size,
                ..
            } = operation
            {
                // The semantic graph is the CPU implementation. CUDA may
                // replace this private boundary with grouped expert kernels;
                // no backend selector enters the model-authoring API.
                if let Some((_, capability_major, _)) = cuda {
                    if moe_backend::supported(self.values[*hidden].shape.dtype(), capability_major)
                        && self.values[*hidden].shape.dimensions()[0] > 0
                    {
                        let lowered = match self.values[*gate_up_weights].shape.partitions()[0] {
                            Partition::Sharded(expert_axis) => lower_expert_parallel_moe(
                                context,
                                &mut block,
                                sharding,
                                [
                                    mlir_value(&values, *hidden),
                                    mlir_value(&values, *routing_weights),
                                    mlir_value(&values, *gate_up_weights),
                                    mlir_value(&values, *down_weights),
                                    mlir_value(&values, *sorted_assignments),
                                    mlir_value(&values, *block_experts),
                                ],
                                [
                                    self.values[*hidden].shape,
                                    self.values[*routing_weights].shape,
                                    self.values[*gate_up_weights].shape,
                                    self.values[*down_weights].shape,
                                    self.values[*sorted_assignments].shape,
                                    self.values[*block_experts].shape,
                                ],
                                types[*result],
                                self.values[*result].shape,
                                expert_axis,
                                *activation,
                                *experts_per_token,
                                *block_size,
                                operation_index,
                            )?,
                            _ => moe_backend::lower(
                                context,
                                &mut block,
                                moe_backend::Inputs {
                                    hidden: mlir_value(&values, *hidden),
                                    routing_weights: mlir_value(&values, *routing_weights),
                                    gate_up_weights: mlir_value(&values, *gate_up_weights),
                                    down_weights: mlir_value(&values, *down_weights),
                                    sorted_assignments: mlir_value(&values, *sorted_assignments),
                                    block_experts: mlir_value(&values, *block_experts),
                                    expert_offset: None,
                                    hidden_shape: self.values[*hidden].shape,
                                    gate_up_shape: self.values[*gate_up_weights].shape,
                                    down_shape: self.values[*down_weights].shape,
                                    schedule_shape: self.values[*sorted_assignments].shape,
                                    block_experts_shape: self.values[*block_experts].shape,
                                    result_type: types[*result],
                                    activation: *activation,
                                    experts_per_token: *experts_per_token,
                                    block_size: *block_size,
                                },
                            )?,
                        };
                        values[*result] = Some(constrain_compound_result(
                            context,
                            &mut block,
                            lowered,
                            types[*result],
                            sharding,
                            self.values[*result].shape,
                        )?);
                        continue;
                    }
                }
                values[*result] = Some(mlir_value(&values, *portable_output));
                continue;
            }
            if let Operation::GatedDeltaNet {
                queries,
                keys,
                values: sequence_values,
                alphas,
                betas,
                initial_state,
                outputs,
                final_state,
            } = operation
            {
                let (lowered_outputs, lowered_state) = lower_gated_delta_net(
                    context,
                    &mut block,
                    [
                        mlir_value(&values, *queries),
                        mlir_value(&values, *keys),
                        mlir_value(&values, *sequence_values),
                        mlir_value(&values, *alphas),
                        mlir_value(&values, *betas),
                        mlir_value(&values, *initial_state),
                    ],
                    [
                        self.values[*queries].shape,
                        self.values[*keys].shape,
                        self.values[*sequence_values].shape,
                        self.values[*alphas].shape,
                        self.values[*betas].shape,
                        self.values[*initial_state].shape,
                    ],
                    types[*outputs],
                    types[*final_state],
                )?;
                values[*outputs] = Some(constrain_compound_result(
                    context,
                    &mut block,
                    lowered_outputs,
                    types[*outputs],
                    sharding,
                    self.values[*outputs].shape,
                )?);
                values[*final_state] = Some(constrain_compound_result(
                    context,
                    &mut block,
                    lowered_state,
                    types[*final_state],
                    sharding,
                    self.values[*final_state].shape,
                )?);
                continue;
            }
            if let Operation::Attention {
                query,
                key,
                value,
                query_positions,
                key_positions,
                result,
                options,
            } = operation
            {
                let inputs = ordinary_attention::Inputs {
                    query: mlir_value(&values, *query),
                    key: mlir_value(&values, *key),
                    value: mlir_value(&values, *value),
                    query_positions: mlir_value(&values, *query_positions),
                    key_positions: mlir_value(&values, *key_positions),
                    query_shape: self.values[*query].shape,
                    key_shape: self.values[*key].shape,
                    query_positions_dtype: self.values[*query_positions].shape.dtype(),
                    key_positions_dtype: self.values[*key_positions].shape.dtype(),
                    result_type: types[*result],
                    options: *options,
                };
                let lowered = if let Some((_, major, minor)) = cuda {
                    ordinary_attention::lower_cuda(context, &mut block, inputs, major, minor)?
                } else {
                    ordinary_attention::lower(context, &mut block, inputs)?
                };
                values[*result] = Some(constrain_compound_result(
                    context,
                    &mut block,
                    lowered,
                    types[*result],
                    sharding,
                    self.values[*result].shape,
                )?);
                continue;
            }
            if let Operation::ArgMax {
                input,
                indices,
                value_init,
                index_init,
                value_result,
                index_result,
                axis,
            } = operation
            {
                let value_dtype = self.values[*input].shape.dtype();
                let index_dtype = self.values[*indices].shape.dtype();
                let value_scalar = context.ranked_tensor_type(value_dtype, &[])?;
                let index_scalar = context.ranked_tensor_type(index_dtype, &[])?;
                let bool_scalar = context.ranked_tensor_type(DType::Bool, &[])?;
                let mut reduction_block = Block::new(
                    context,
                    &[value_scalar, index_scalar, value_scalar, index_scalar],
                )?;
                let left_value = reduction_block.argument(0)?;
                let left_index = reduction_block.argument(1)?;
                let right_value = reduction_block.argument(2)?;
                let right_index = reduction_block.argument(3)?;

                let left_greater = context.compare(
                    left_value,
                    right_value,
                    bool_scalar,
                    StableHloComparison::Gt,
                    comparison_type(value_dtype),
                )?;
                let left_greater_value = left_greater.result(0)?;
                reduction_block.append_operation(left_greater)?;
                let left_nan = context.compare(
                    left_value,
                    left_value,
                    bool_scalar,
                    StableHloComparison::Ne,
                    comparison_type(value_dtype),
                )?;
                let left_nan_value = left_nan.result(0)?;
                reduction_block.append_operation(left_nan)?;
                let left_greater_or_nan = context.binary(
                    StableHloBinary::Or,
                    left_greater_value,
                    left_nan_value,
                    bool_scalar,
                )?;
                let left_greater_or_nan_value = left_greater_or_nan.result(0)?;
                reduction_block.append_operation(left_greater_or_nan)?;

                let values_equal = context.compare(
                    left_value,
                    right_value,
                    bool_scalar,
                    StableHloComparison::Eq,
                    comparison_type(value_dtype),
                )?;
                let values_equal_value = values_equal.result(0)?;
                reduction_block.append_operation(values_equal)?;
                let left_index_first = context.compare(
                    left_index,
                    right_index,
                    bool_scalar,
                    StableHloComparison::Lt,
                    comparison_type(index_dtype),
                )?;
                let left_index_first_value = left_index_first.result(0)?;
                reduction_block.append_operation(left_index_first)?;
                let equal_and_first = context.binary(
                    StableHloBinary::And,
                    values_equal_value,
                    left_index_first_value,
                    bool_scalar,
                )?;
                let equal_and_first_value = equal_and_first.result(0)?;
                reduction_block.append_operation(equal_and_first)?;
                let keep_left_index = context.binary(
                    StableHloBinary::Or,
                    left_greater_or_nan_value,
                    equal_and_first_value,
                    bool_scalar,
                )?;
                let keep_left_index_value = keep_left_index.result(0)?;
                reduction_block.append_operation(keep_left_index)?;

                let maximum = context.select(
                    left_greater_or_nan_value,
                    left_value,
                    right_value,
                    value_scalar,
                )?;
                let maximum_value = maximum.result(0)?;
                reduction_block.append_operation(maximum)?;
                let maximum_index =
                    context.select(keep_left_index_value, left_index, right_index, index_scalar)?;
                let maximum_index_value = maximum_index.result(0)?;
                reduction_block.append_operation(maximum_index)?;
                reduction_block.append_operation(
                    context.stablehlo_return(&[maximum_value, maximum_index_value])?,
                )?;
                let mut reduction_body = Region::new(context)?;
                reduction_body.append_block(reduction_block)?;
                let reduction = context.reduce_many(
                    &[mlir_value(&values, *input), mlir_value(&values, *indices)],
                    &[
                        mlir_value(&values, *value_init),
                        mlir_value(&values, *index_init),
                    ],
                    &[types[*value_result], types[*index_result]],
                    &[*axis as i64],
                    reduction_body,
                )?;
                values[*value_result] = Some(reduction.result(0)?);
                values[*index_result] = Some(reduction.result(1)?);
                block.append_operation(reduction)?;
                continue;
            }
            if let Operation::Sort {
                input,
                indices,
                values_result,
                indices_result,
                axis,
                descending,
                stable,
            } = operation
            {
                let sort = context.sort(
                    &[mlir_value(&values, *input), mlir_value(&values, *indices)],
                    &[types[*values_result], types[*indices_result]],
                    *axis as i64,
                    *stable,
                    sort_comparator_region(
                        context,
                        self.values[*input].shape.dtype(),
                        *descending,
                    )?,
                )?;
                values[*values_result] = Some(sort.result(0)?);
                values[*indices_result] = Some(sort.result(1)?);
                block.append_operation(sort)?;
                continue;
            }
            if let Operation::RngBitGenerator {
                state,
                state_result,
                output_result,
            } = operation
            {
                let generated = context.rng_bit_generator(
                    mlir_value(&values, *state),
                    types[*state_result],
                    types[*output_result],
                )?;
                values[*state_result] = Some(generated.result(0)?);
                values[*output_result] = Some(generated.result(1)?);
                block.append_operation(generated)?;
                continue;
            }
            if let Operation::PagedAttention {
                query,
                key_cache,
                value_cache,
                page_table,
                sequence_lengths,
                query_positions,
                result,
                options,
            } = operation
            {
                let inputs = paged_attention::Inputs {
                    query: mlir_value(&values, *query),
                    key_cache: mlir_value(&values, *key_cache),
                    value_cache: mlir_value(&values, *value_cache),
                    page_table: mlir_value(&values, *page_table),
                    sequence_lengths: mlir_value(&values, *sequence_lengths),
                    query_positions: mlir_value(&values, *query_positions),
                    query_shape: self.values[*query].shape,
                    cache_shape: self.values[*key_cache].shape,
                    page_table_shape: self.values[*page_table].shape,
                    page_table_dtype: self.values[*page_table].shape.dtype(),
                    sequence_lengths_dtype: self.values[*sequence_lengths].shape.dtype(),
                    query_positions_dtype: self.values[*query_positions].shape.dtype(),
                    result_type: types[*result],
                    options: *options,
                };
                let lowered = if let Some((core_count, major, minor)) = cuda {
                    paged_attention::lower_triton(
                        context, &mut block, inputs, core_count, major, minor,
                    )?
                } else {
                    paged_attention::lower(context, &mut block, inputs)?
                };
                values[*result] = Some(constrain_compound_result(
                    context,
                    &mut block,
                    lowered,
                    types[*result],
                    sharding,
                    self.values[*result].shape,
                )?);
                continue;
            }
            let (result, result_index) = match operation {
                Operation::Cholesky {
                    input,
                    result,
                    lower,
                } => (
                    context.cholesky(mlir_value(&values, *input), types[*result], *lower)?,
                    *result,
                ),
                Operation::TriangularSolve {
                    coefficient,
                    right_hand_side,
                    result,
                    lower,
                } => (
                    context.triangular_solve(
                        mlir_value(&values, *coefficient),
                        mlir_value(&values, *right_hand_side),
                        types[*result],
                        *lower,
                    )?,
                    *result,
                ),
                Operation::Constant { result, literal } => (
                    context.constant(types[*result], context.parse_attribute(literal)?)?,
                    *result,
                ),
                Operation::DotGeneral {
                    left,
                    right,
                    result,
                    left_batch,
                    right_batch,
                    left_contract,
                    right_contract,
                } => (
                    context.dot_general(
                        mlir_value(&values, *left),
                        mlir_value(&values, *right),
                        types[*result],
                        &axes_i64(left_batch),
                        &axes_i64(right_batch),
                        &axes_i64(left_contract),
                        &axes_i64(right_contract),
                    )?,
                    *result,
                ),
                Operation::Binary {
                    left,
                    right,
                    result,
                    operation,
                } => (
                    match operation {
                        Binary::Add => context.add(
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Subtract => context.binary(
                            StableHloBinary::Subtract,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Multiply => context.binary(
                            StableHloBinary::Multiply,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Divide => context.binary(
                            StableHloBinary::Divide,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Minimum => context.binary(
                            StableHloBinary::Minimum,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Maximum => context.binary(
                            StableHloBinary::Maximum,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Power => context.binary(
                            StableHloBinary::Power,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Remainder => context.binary(
                            StableHloBinary::Remainder,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::And => context.binary(
                            StableHloBinary::And,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Or => context.binary(
                            StableHloBinary::Or,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::Xor => context.binary(
                            StableHloBinary::Xor,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::ShiftLeft => context.binary(
                            StableHloBinary::ShiftLeft,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::ShiftRightArithmetic => context.binary(
                            StableHloBinary::ShiftRightArithmetic,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                        Binary::ShiftRightLogical => context.binary(
                            StableHloBinary::ShiftRightLogical,
                            mlir_value(&values, *left),
                            mlir_value(&values, *right),
                            types[*result],
                        )?,
                    },
                    *result,
                ),
                Operation::Clamp {
                    minimum,
                    input,
                    maximum,
                    result,
                } => (
                    context.clamp(
                        mlir_value(&values, *minimum),
                        mlir_value(&values, *input),
                        mlir_value(&values, *maximum),
                        types[*result],
                    )?,
                    *result,
                ),
                Operation::BroadcastInDim {
                    input,
                    result,
                    dimensions,
                } => (
                    context.broadcast_in_dim(
                        mlir_value(&values, *input),
                        types[*result],
                        &axes_i64(dimensions),
                    )?,
                    *result,
                ),
                Operation::Iota { result, axis } => {
                    (context.iota(types[*result], *axis as i64)?, *result)
                }
                Operation::Concatenate {
                    inputs,
                    result,
                    axis,
                } => (
                    context.concatenate(
                        &inputs
                            .iter()
                            .map(|input| mlir_value(&values, *input))
                            .collect::<Vec<_>>(),
                        types[*result],
                        *axis as i64,
                    )?,
                    *result,
                ),
                Operation::Pad {
                    input,
                    padding_value,
                    result,
                    edge_low,
                    edge_high,
                    interior,
                } => (
                    context.pad(
                        mlir_value(&values, *input),
                        mlir_value(&values, *padding_value),
                        types[*result],
                        edge_low,
                        edge_high,
                        interior,
                    )?,
                    *result,
                ),
                Operation::Reverse {
                    input,
                    result,
                    axes,
                } => (
                    context.reverse(
                        mlir_value(&values, *input),
                        types[*result],
                        &axes_i64(axes),
                    )?,
                    *result,
                ),
                Operation::Slice {
                    input,
                    result,
                    starts,
                    limits,
                    strides,
                } => (
                    context.slice(
                        mlir_value(&values, *input),
                        types[*result],
                        starts,
                        limits,
                        strides,
                    )?,
                    *result,
                ),
                Operation::DynamicSlice {
                    input,
                    starts,
                    result,
                    sizes,
                } => (
                    context.dynamic_slice(
                        mlir_value(&values, *input),
                        &starts
                            .iter()
                            .map(|start| mlir_value(&values, *start))
                            .collect::<Vec<_>>(),
                        types[*result],
                        sizes,
                    )?,
                    *result,
                ),
                Operation::DynamicUpdateSlice {
                    input,
                    update,
                    starts,
                    result,
                } => (
                    context.dynamic_update_slice(
                        mlir_value(&values, *input),
                        mlir_value(&values, *update),
                        &starts
                            .iter()
                            .map(|start| mlir_value(&values, *start))
                            .collect::<Vec<_>>(),
                        types[*result],
                    )?,
                    *result,
                ),
                Operation::Gather {
                    input,
                    indices,
                    result,
                    offset_dims,
                    collapsed_slice_dims,
                    operand_batching_dims,
                    start_indices_batching_dims,
                    start_index_map,
                    index_vector_dim,
                    slice_sizes,
                    indices_are_sorted,
                } => (
                    context.gather(
                        mlir_value(&values, *input),
                        mlir_value(&values, *indices),
                        types[*result],
                        &axes_i64(offset_dims),
                        &axes_i64(collapsed_slice_dims),
                        &axes_i64(operand_batching_dims),
                        &axes_i64(start_indices_batching_dims),
                        &axes_i64(start_index_map),
                        *index_vector_dim as i64,
                        slice_sizes,
                        *indices_are_sorted,
                    )?,
                    *result,
                ),
                Operation::Scatter {
                    input,
                    indices,
                    updates,
                    result,
                    update_window_dims,
                    inserted_window_dims,
                    input_batching_dims,
                    scatter_indices_batching_dims,
                    scatter_dims_to_operand_dims,
                    index_vector_dim,
                    indices_are_sorted,
                    unique_indices,
                    computation,
                } => {
                    let scalar_type =
                        context.ranked_tensor_type(self.values[*input].shape.dtype(), &[])?;
                    let mut update_block = Block::new(context, &[scalar_type, scalar_type])?;
                    let existing = update_block.argument(0)?;
                    let update = update_block.argument(1)?;
                    let returned = match computation {
                        ScatterComputation::Update => update,
                        ScatterComputation::Add => {
                            let operation = context.add(existing, update, scalar_type)?;
                            let returned = operation.result(0)?;
                            update_block.append_operation(operation)?;
                            returned
                        }
                        ScatterComputation::Multiply => {
                            let operation = context.binary(
                                StableHloBinary::Multiply,
                                existing,
                                update,
                                scalar_type,
                            )?;
                            let returned = operation.result(0)?;
                            update_block.append_operation(operation)?;
                            returned
                        }
                        ScatterComputation::Minimum => {
                            let operation = context.binary(
                                StableHloBinary::Minimum,
                                existing,
                                update,
                                scalar_type,
                            )?;
                            let returned = operation.result(0)?;
                            update_block.append_operation(operation)?;
                            returned
                        }
                        ScatterComputation::Maximum => {
                            let operation = context.binary(
                                StableHloBinary::Maximum,
                                existing,
                                update,
                                scalar_type,
                            )?;
                            let returned = operation.result(0)?;
                            update_block.append_operation(operation)?;
                            returned
                        }
                    };
                    update_block.append_operation(context.stablehlo_return(&[returned])?)?;
                    let mut update_computation = Region::new(context)?;
                    update_computation.append_block(update_block)?;
                    (
                        context.scatter(
                            mlir_value(&values, *input),
                            mlir_value(&values, *indices),
                            mlir_value(&values, *updates),
                            types[*result],
                            &axes_i64(update_window_dims),
                            &axes_i64(inserted_window_dims),
                            &axes_i64(input_batching_dims),
                            &axes_i64(scatter_indices_batching_dims),
                            &axes_i64(scatter_dims_to_operand_dims),
                            *index_vector_dim as i64,
                            *indices_are_sorted,
                            *unique_indices,
                            update_computation,
                        )?,
                        *result,
                    )
                }
                Operation::Reduce {
                    input,
                    init,
                    result,
                    axes,
                    reduction,
                } => (
                    context.reduce(
                        mlir_value(&values, *input),
                        mlir_value(&values, *init),
                        types[*result],
                        &axes_i64(axes),
                        reduction_region(context, self.values[*input].shape.dtype(), *reduction)?,
                    )?,
                    *result,
                ),
                Operation::AllReduce {
                    input,
                    result,
                    reduction,
                } => {
                    let device_count = if sharding.is_single() {
                        1
                    } else if sharding.is_mesh() {
                        sharding
                            .mesh_axes()
                            .try_fold(1usize, |count, (_, size)| count.checked_mul(size))
                            .ok_or(Error::InvalidCollective("mesh device count overflows"))?
                    } else {
                        return Err(Error::InvalidCollective(
                            "replicated placement has no compile-time device count; use an explicit Shardy mesh",
                        ));
                    };
                    let replica_group = (0..device_count)
                        .map(|device| {
                            i64::try_from(device)
                                .map_err(|_| Error::InvalidCollective("device id exceeds I64"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        context.all_reduce(
                            mlir_value(&values, *input),
                            types[*result],
                            &[replica_group],
                            collective_channel(operation_index)?,
                            reduction_region(
                                context,
                                self.values[*input].shape.dtype(),
                                *reduction,
                            )?,
                        )?,
                        *result,
                    )
                }
                Operation::ReduceWindow {
                    input,
                    init,
                    result,
                    window_dimensions,
                    window_strides,
                    base_dilations,
                    window_dilations,
                    padding,
                    reduction,
                } => (
                    context.reduce_window(
                        mlir_value(&values, *input),
                        mlir_value(&values, *init),
                        types[*result],
                        window_dimensions,
                        window_strides,
                        base_dilations,
                        window_dilations,
                        padding,
                        reduction_region(context, self.values[*input].shape.dtype(), *reduction)?,
                    )?,
                    *result,
                ),
                Operation::Convolution {
                    input,
                    kernel,
                    result,
                    strides,
                    padding,
                    input_dilation,
                    kernel_dilation,
                    kernel_reversal,
                    input_batch_axis,
                    input_feature_axis,
                    input_spatial_axes,
                    kernel_input_feature_axis,
                    kernel_output_feature_axis,
                    kernel_spatial_axes,
                    output_batch_axis,
                    output_feature_axis,
                    output_spatial_axes,
                    feature_groups,
                    batch_groups,
                } => (
                    context.convolution(
                        mlir_value(&values, *input),
                        mlir_value(&values, *kernel),
                        types[*result],
                        ConvolutionWindow {
                            strides,
                            padding,
                            lhs_dilation: input_dilation,
                            rhs_dilation: kernel_dilation,
                            reversal: kernel_reversal,
                        },
                        ConvolutionDimensionNumbers {
                            input_batch: *input_batch_axis as i64,
                            input_feature: *input_feature_axis as i64,
                            input_spatial: &axes_i64(input_spatial_axes),
                            kernel_input_feature: *kernel_input_feature_axis as i64,
                            kernel_output_feature: *kernel_output_feature_axis as i64,
                            kernel_spatial: &axes_i64(kernel_spatial_axes),
                            output_batch: *output_batch_axis as i64,
                            output_feature: *output_feature_axis as i64,
                            output_spatial: &axes_i64(output_spatial_axes),
                        },
                        *feature_groups,
                        *batch_groups,
                    )?,
                    *result,
                ),
                Operation::Complex {
                    real,
                    imaginary,
                    result,
                } => (
                    context.complex(
                        mlir_value(&values, *real),
                        mlir_value(&values, *imaginary),
                        types[*result],
                    )?,
                    *result,
                ),
                Operation::Component {
                    input,
                    result,
                    component,
                } => (
                    match component {
                        Component::Real => {
                            context.real(mlir_value(&values, *input), types[*result])?
                        }
                        Component::Imaginary => {
                            context.imaginary(mlir_value(&values, *input), types[*result])?
                        }
                    },
                    *result,
                ),
                Operation::Fft {
                    input,
                    result,
                    kind,
                    lengths,
                } => (
                    context.fft(
                        mlir_value(&values, *input),
                        types[*result],
                        match kind {
                            FftType::Fft => StableHloFftType::Fft,
                            FftType::Ifft => StableHloFftType::Ifft,
                            FftType::Rfft => StableHloFftType::Rfft,
                            FftType::Irfft => StableHloFftType::Irfft,
                        },
                        lengths,
                    )?,
                    *result,
                ),
                Operation::Compare {
                    left,
                    right,
                    result,
                    comparison,
                    input_dtype,
                } => (
                    context.compare(
                        mlir_value(&values, *left),
                        mlir_value(&values, *right),
                        types[*result],
                        match comparison {
                            Comparison::Eq => StableHloComparison::Eq,
                            Comparison::Ne => StableHloComparison::Ne,
                            Comparison::Ge => StableHloComparison::Ge,
                            Comparison::Gt => StableHloComparison::Gt,
                            Comparison::Le => StableHloComparison::Le,
                            Comparison::Lt => StableHloComparison::Lt,
                        },
                        comparison_type(*input_dtype),
                    )?,
                    *result,
                ),
                Operation::Select {
                    predicate,
                    on_true,
                    on_false,
                    result,
                } => (
                    context.select(
                        mlir_value(&values, *predicate),
                        mlir_value(&values, *on_true),
                        mlir_value(&values, *on_false),
                        types[*result],
                    )?,
                    *result,
                ),
                Operation::Convert { input, result } => (
                    context.convert(mlir_value(&values, *input), types[*result])?,
                    *result,
                ),
                Operation::Bitcast { input, result } => (
                    context.bitcast_convert(mlir_value(&values, *input), types[*result])?,
                    *result,
                ),
                Operation::ReducePrecision {
                    input,
                    result,
                    exponent_bits,
                    mantissa_bits,
                } => (
                    context.reduce_precision(
                        mlir_value(&values, *input),
                        types[*result],
                        *exponent_bits,
                        *mantissa_bits,
                    )?,
                    *result,
                ),
                Operation::Reshape { input, result } => (
                    context.reshape(mlir_value(&values, *input), types[*result])?,
                    *result,
                ),
                Operation::Transpose {
                    input,
                    result,
                    permutation,
                } => (
                    context.transpose(
                        mlir_value(&values, *input),
                        types[*result],
                        &axes_i64(permutation),
                    )?,
                    *result,
                ),
                Operation::Unary {
                    input,
                    result,
                    operation,
                } => (
                    context.unary_math(
                        match operation {
                            Unary::Negate => StableHloUnary::Negate,
                            Unary::Abs => StableHloUnary::Abs,
                            Unary::Exp => StableHloUnary::Exponential,
                            Unary::Log => StableHloUnary::Log,
                            Unary::Sqrt => StableHloUnary::Sqrt,
                            Unary::Rsqrt => StableHloUnary::Rsqrt,
                            Unary::Tanh => StableHloUnary::Tanh,
                            Unary::Sin => StableHloUnary::Sine,
                            Unary::Cos => StableHloUnary::Cosine,
                            Unary::Logistic => StableHloUnary::Logistic,
                            Unary::Floor => StableHloUnary::Floor,
                            Unary::Ceil => StableHloUnary::Ceil,
                            Unary::Not => StableHloUnary::Not,
                            Unary::CountLeadingZeros => StableHloUnary::CountLeadingZeros,
                            Unary::IsFinite => StableHloUnary::IsFinite,
                            Unary::PopulationCount => StableHloUnary::PopulationCount,
                            Unary::Sign => StableHloUnary::Sign,
                            Unary::Expm1 => StableHloUnary::ExponentialMinusOne,
                            Unary::RoundNearestAwayFromZero => {
                                StableHloUnary::RoundNearestAwayFromZero
                            }
                            Unary::RoundNearestEven => StableHloUnary::RoundNearestEven,
                        },
                        mlir_value(&values, *input),
                        types[*result],
                    )?,
                    *result,
                ),
                Operation::ShardingConstraint { input, result } => {
                    let attribute =
                        tensor_sharding_attribute(context, sharding, self.values[*result].shape)?
                            .ok_or_else(|| {
                            Error::Mlir(MlirError::InvalidAttribute {
                                source: "sharding constraint requires a logical mesh".to_owned(),
                            })
                        })?;
                    (
                        context.sharding_constraint(
                            mlir_value(&values, *input),
                            types[*result],
                            attribute,
                        )?,
                        *result,
                    )
                }
                Operation::OptimizationBarrier { input, result } => (
                    context.optimization_barrier(mlir_value(&values, *input), types[*result])?,
                    *result,
                ),
                Operation::PagedAttention { .. } => {
                    unreachable!("paged attention is lowered before scalar operations")
                }
                Operation::MoeDispatch { .. } => {
                    unreachable!("MoE dispatch is lowered before scalar operations")
                }
                Operation::GatedDeltaNet { .. } => {
                    unreachable!("Gated DeltaNet is lowered before scalar operations")
                }
                Operation::Attention { .. } => {
                    unreachable!("attention is lowered before scalar operations")
                }
                Operation::ArgMax { .. }
                | Operation::Sort { .. }
                | Operation::RngBitGenerator { .. } => {
                    unreachable!(
                        "multi-result operations are lowered before single-result operations"
                    )
                }
            };
            values[result_index] = Some(result.result(0)?);
            block.append_operation(result)?;
        }

        let returned = self
            .outputs
            .iter()
            .map(|value| mlir_value(&values, *value))
            .collect::<Vec<_>>();
        block.append_operation(context.return_operation(&returned)?)?;
        let mut body = Region::new(context)?;
        body.append_block(block)?;
        let mut input_aliases = vec![None; self.inputs.len()];
        for (output, input) in self.output_aliases.iter().enumerate() {
            if let Some(input) = input {
                input_aliases[*input] = Some(output);
            }
        }
        let input_shardings = self
            .inputs
            .iter()
            .map(|value| tensor_sharding_attribute(context, sharding, self.values[*value].shape))
            .collect::<Result<Vec<_>, _>>()?;
        let result_shardings = self
            .outputs
            .iter()
            .map(|value| tensor_sharding_attribute(context, sharding, self.values[*value].shape))
            .collect::<Result<Vec<_>, _>>()?;
        let function = context.function_with_attributes(
            "main",
            &input_types,
            &result_types,
            &input_aliases,
            &input_shardings,
            &result_shardings,
            body,
        )?;
        let mut module = context.empty_module()?;
        if sharding.is_mesh() {
            let axes = sharding
                .mesh_axes()
                .map(|(tag, size)| {
                    i64::try_from(size)
                        .map(|size| (axis_name(tag), size))
                        .map_err(|_| Error::InvalidStructure("mesh axis size exceeds I64"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let borrowed = axes
                .iter()
                .map(|(name, size)| (name.as_str(), *size))
                .collect::<Vec<_>>();
            let mesh = context.shardy_mesh(&borrowed, &[])?;
            module.append_operation(context.shardy_mesh_operation("nml_mesh", mesh)?)?;
        }
        module.append_operation(function)?;
        module.verify()?;
        Ok(module)
    }
}

#[derive(Debug)]
struct Value {
    name: String,
    shape: Shape,
}

#[derive(Debug)]
enum Operation {
    Cholesky {
        input: usize,
        result: usize,
        lower: bool,
    },
    TriangularSolve {
        coefficient: usize,
        right_hand_side: usize,
        result: usize,
        lower: bool,
    },
    DotGeneral {
        left: usize,
        right: usize,
        result: usize,
        left_batch: Vec<usize>,
        right_batch: Vec<usize>,
        left_contract: Vec<usize>,
        right_contract: Vec<usize>,
    },
    Constant {
        result: usize,
        literal: String,
    },
    Binary {
        left: usize,
        right: usize,
        result: usize,
        operation: Binary,
    },
    Clamp {
        minimum: usize,
        input: usize,
        maximum: usize,
        result: usize,
    },
    BroadcastInDim {
        input: usize,
        result: usize,
        dimensions: Vec<usize>,
    },
    Iota {
        result: usize,
        axis: usize,
    },
    Concatenate {
        inputs: Vec<usize>,
        result: usize,
        axis: usize,
    },
    Pad {
        input: usize,
        padding_value: usize,
        result: usize,
        edge_low: Vec<i64>,
        edge_high: Vec<i64>,
        interior: Vec<i64>,
    },
    Reverse {
        input: usize,
        result: usize,
        axes: Vec<usize>,
    },
    Slice {
        input: usize,
        result: usize,
        starts: Vec<i64>,
        limits: Vec<i64>,
        strides: Vec<i64>,
    },
    DynamicSlice {
        input: usize,
        starts: Vec<usize>,
        result: usize,
        sizes: Vec<i64>,
    },
    DynamicUpdateSlice {
        input: usize,
        update: usize,
        starts: Vec<usize>,
        result: usize,
    },
    Gather {
        input: usize,
        indices: usize,
        result: usize,
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        operand_batching_dims: Vec<usize>,
        start_indices_batching_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<i64>,
        indices_are_sorted: bool,
    },
    Scatter {
        input: usize,
        indices: usize,
        updates: usize,
        result: usize,
        update_window_dims: Vec<usize>,
        inserted_window_dims: Vec<usize>,
        input_batching_dims: Vec<usize>,
        scatter_indices_batching_dims: Vec<usize>,
        scatter_dims_to_operand_dims: Vec<usize>,
        index_vector_dim: usize,
        indices_are_sorted: bool,
        unique_indices: bool,
        computation: ScatterComputation,
    },
    Reduce {
        input: usize,
        init: usize,
        result: usize,
        axes: Vec<usize>,
        reduction: Reduction,
    },
    AllReduce {
        input: usize,
        result: usize,
        reduction: Reduction,
    },
    ReduceWindow {
        input: usize,
        init: usize,
        result: usize,
        window_dimensions: Vec<i64>,
        window_strides: Vec<i64>,
        base_dilations: Vec<i64>,
        window_dilations: Vec<i64>,
        padding: Vec<[i64; 2]>,
        reduction: Reduction,
    },
    Convolution {
        input: usize,
        kernel: usize,
        result: usize,
        strides: Vec<i64>,
        padding: Vec<[i64; 2]>,
        input_dilation: Vec<i64>,
        kernel_dilation: Vec<i64>,
        kernel_reversal: Vec<bool>,
        input_batch_axis: usize,
        input_feature_axis: usize,
        input_spatial_axes: Vec<usize>,
        kernel_input_feature_axis: usize,
        kernel_output_feature_axis: usize,
        kernel_spatial_axes: Vec<usize>,
        output_batch_axis: usize,
        output_feature_axis: usize,
        output_spatial_axes: Vec<usize>,
        feature_groups: i64,
        batch_groups: i64,
    },
    Sort {
        input: usize,
        indices: usize,
        values_result: usize,
        indices_result: usize,
        axis: usize,
        descending: bool,
        stable: bool,
    },
    RngBitGenerator {
        state: usize,
        state_result: usize,
        output_result: usize,
    },
    ArgMax {
        input: usize,
        indices: usize,
        value_init: usize,
        index_init: usize,
        value_result: usize,
        index_result: usize,
        axis: usize,
    },
    Attention {
        query: usize,
        key: usize,
        value: usize,
        query_positions: usize,
        key_positions: usize,
        result: usize,
        options: AttentionOptions,
    },
    PagedAttention {
        query: usize,
        key_cache: usize,
        value_cache: usize,
        page_table: usize,
        sequence_lengths: usize,
        query_positions: usize,
        result: usize,
        options: AttentionOptions,
    },
    GatedDeltaNet {
        queries: usize,
        keys: usize,
        values: usize,
        alphas: usize,
        betas: usize,
        initial_state: usize,
        outputs: usize,
        final_state: usize,
    },
    MoeDispatch {
        hidden: usize,
        routing_weights: usize,
        gate_up_weights: usize,
        down_weights: usize,
        sorted_assignments: usize,
        block_experts: usize,
        portable_output: usize,
        result: usize,
        activation: MoeActivation,
        experts_per_token: usize,
        block_size: usize,
    },
    Complex {
        real: usize,
        imaginary: usize,
        result: usize,
    },
    Component {
        input: usize,
        result: usize,
        component: Component,
    },
    Fft {
        input: usize,
        result: usize,
        kind: FftType,
        lengths: Vec<i64>,
    },
    Compare {
        left: usize,
        right: usize,
        result: usize,
        comparison: Comparison,
        input_dtype: DType,
    },
    Select {
        predicate: usize,
        on_true: usize,
        on_false: usize,
        result: usize,
    },
    Convert {
        input: usize,
        result: usize,
    },
    Bitcast {
        input: usize,
        result: usize,
    },
    ReducePrecision {
        input: usize,
        result: usize,
        exponent_bits: i32,
        mantissa_bits: i32,
    },
    Reshape {
        input: usize,
        result: usize,
    },
    Transpose {
        input: usize,
        result: usize,
        permutation: Vec<usize>,
    },
    Unary {
        input: usize,
        result: usize,
        operation: Unary,
    },
    ShardingConstraint {
        input: usize,
        result: usize,
    },
    OptimizationBarrier {
        input: usize,
        result: usize,
    },
}

#[derive(Clone, Copy, Debug)]
enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Minimum,
    Maximum,
    Power,
    Remainder,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRightArithmetic,
    ShiftRightLogical,
}

impl Binary {
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
            Self::Power => "power",
            Self::Remainder => "remainder",
            Self::And => "logical_and",
            Self::Or => "logical_or",
            Self::Xor => "logical_xor",
            Self::ShiftLeft => "shift_left",
            Self::ShiftRightArithmetic => "shift_right_arithmetic",
            Self::ShiftRightLogical => "shift_right_logical",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Comparison {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

#[derive(Clone, Copy, Debug)]
enum Unary {
    Negate,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Sin,
    Cos,
    Logistic,
    Floor,
    Ceil,
    Not,
    CountLeadingZeros,
    IsFinite,
    PopulationCount,
    Sign,
    Expm1,
    RoundNearestAwayFromZero,
    RoundNearestEven,
}

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Sum,
    Maximum,
    Minimum,
}

#[derive(Clone, Copy, Debug)]
enum MoeActivation {
    Silu,
    Gelu,
    Relu,
}

#[derive(Clone, Copy, Debug)]
enum ScatterComputation {
    Update,
    Add,
    Multiply,
    Minimum,
    Maximum,
}

impl ScatterComputation {
    const fn name(self) -> &'static str {
        match self {
            Self::Update => "scatter_update",
            Self::Add => "scatter_add",
            Self::Multiply => "scatter_multiply",
            Self::Minimum => "scatter_minimum",
            Self::Maximum => "scatter_maximum",
        }
    }
}

impl Reduction {
    const fn name(self) -> &'static str {
        match self {
            Self::Sum => "reduce_sum",
            Self::Maximum => "reduce_max",
            Self::Minimum => "reduce_min",
        }
    }
}

impl Unary {
    const fn name(self) -> &'static str {
        match self {
            Self::Negate => "negate",
            Self::Abs => "abs",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Tanh => "tanh",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Logistic => "sigmoid",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Not => "logical_not",
            Self::CountLeadingZeros => "count_leading_zeros",
            Self::IsFinite => "is_finite",
            Self::PopulationCount => "population_count",
            Self::Sign => "sign",
            Self::Expm1 => "expm1",
            Self::RoundNearestAwayFromZero => "round_nearest_away_from_zero",
            Self::RoundNearestEven => "round_nearest_even",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftType {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputKind {
    Activation,
    Parameter,
}

#[derive(Clone, Copy, Debug)]
enum Component {
    Real,
    Imaginary,
}

impl Component {
    const fn name(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Imaginary => "imaginary",
        }
    }
}

fn validate_axes(axes: &[usize], rank: usize, side: &'static str) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for &axis in axes {
        if axis >= rank {
            return Err(Error::AxisOutOfBounds { side, axis, rank });
        }
        if !seen.insert(axis) {
            return Err(Error::DuplicateAxis { side, axis });
        }
    }
    Ok(())
}

fn inserted_axis_shape(
    input: Shape,
    axis: usize,
    dimension: i64,
    tag: AxisTag,
) -> Result<Shape, Error> {
    let mut dimensions = input.dimensions().to_vec();
    dimensions.insert(axis, dimension);
    let mut tags = input.axis_tags().to_vec();
    tags.insert(axis, tag);
    let mut partitions = input.partitions().to_vec();
    partitions.insert(axis, Partition::Unspecified);
    Ok(Shape::new(input.dtype(), &dimensions)?
        .with_axis_tags(&tags)?
        .with_partitions(&partitions)?)
}

struct NdIndexingContract {
    result_shape: Shape,
    index_batch_rank: usize,
    retained_axes: Vec<usize>,
}

fn nd_indexing_contract(
    input: Shape,
    indices: Shape,
    axes: &[usize],
    operation: &'static str,
) -> Result<NdIndexingContract, Error> {
    if axes.is_empty() {
        return Err(Error::InvalidIndexing(
            "ND indexing requires at least one indexed operand axis",
        ));
    }
    validate_axes(axes, input.rank(), operation)?;
    if indices.rank() == 0 {
        return Err(Error::InvalidIndexing(
            "ND indices require a final index-vector dimension",
        ));
    }
    let vector_axis = indices.rank() - 1;
    if indices.dimensions()[vector_axis] != axes.len() as i64 {
        return Err(Error::InvalidIndexing(
            "final indices dimension must match the indexed-axis count",
        ));
    }
    if indices.axis_tags()[vector_axis] != AxisTag::UNKNOWN {
        return Err(Error::InvalidIndexing(
            "index-vector dimension must not be a tagged model axis",
        ));
    }
    if matches!(indices.partitions()[vector_axis], Partition::Sharded(_)) {
        return Err(Error::InvalidIndexing(
            "index-vector dimension cannot be sharded",
        ));
    }
    let retained_axes = (0..input.rank())
        .filter(|axis| !axes.contains(axis))
        .collect::<Vec<_>>();
    let dimensions = indices.dimensions()[..vector_axis]
        .iter()
        .copied()
        .chain(retained_axes.iter().map(|axis| input.dimensions()[*axis]))
        .collect::<Vec<_>>();
    let tags = indices.axis_tags()[..vector_axis]
        .iter()
        .copied()
        .chain(retained_axes.iter().map(|axis| input.axis_tags()[*axis]))
        .collect::<Vec<_>>();
    let partitions = indices.partitions()[..vector_axis]
        .iter()
        .copied()
        .chain(retained_axes.iter().map(|axis| input.partitions()[*axis]))
        .collect::<Vec<_>>();
    let result_shape = Shape::new(input.dtype(), &dimensions)?
        .with_axis_tags(&tags)?
        .with_partitions(&partitions)?;
    Ok(NdIndexingContract {
        result_shape,
        index_batch_rank: vector_axis,
        retained_axes,
    })
}

fn batched_nd_indexing_contract(
    input: Shape,
    indices: Shape,
    batch_axes: usize,
    axes: &[usize],
    operation: &'static str,
) -> Result<NdIndexingContract, Error> {
    if batch_axes == 0 {
        return nd_indexing_contract(input, indices, axes, operation);
    }
    if batch_axes >= input.rank() || indices.rank() <= batch_axes {
        return Err(Error::InvalidIndexing(
            "batched ND indexing requires leading batch axes and a final index vector",
        ));
    }
    if axes.iter().any(|axis| *axis < batch_axes) {
        return Err(Error::InvalidIndexing(
            "indexed axes must be distinct from leading batch axes",
        ));
    }
    validate_axes(axes, input.rank(), operation)?;
    for axis in 0..batch_axes {
        if input.dimensions()[axis] != indices.dimensions()[axis] {
            return Err(Error::DimensionMismatch {
                left_axis: axis,
                right_axis: axis,
                left: input.dimensions()[axis],
                right: indices.dimensions()[axis],
            });
        }
        if input.axis_tags()[axis] != indices.axis_tags()[axis]
            || input.partitions()[axis] != indices.partitions()[axis]
        {
            return Err(Error::MetadataMismatch {
                operation,
                field: "batch axis metadata",
            });
        }
    }
    let vector_axis = indices.rank() - 1;
    if axes.is_empty() || indices.dimensions()[vector_axis] != axes.len() as i64 {
        return Err(Error::InvalidIndexing(
            "final indices dimension must match the indexed-axis count",
        ));
    }
    if indices.axis_tags()[vector_axis] != AxisTag::UNKNOWN
        || matches!(indices.partitions()[vector_axis], Partition::Sharded(_))
    {
        return Err(Error::InvalidIndexing(
            "index-vector dimension must be anonymous and unsharded",
        ));
    }
    let retained_axes = (batch_axes..input.rank())
        .filter(|axis| !axes.contains(axis))
        .collect::<Vec<_>>();
    let dimensions = indices.dimensions()[..vector_axis]
        .iter()
        .copied()
        .chain(retained_axes.iter().map(|axis| input.dimensions()[*axis]))
        .collect::<Vec<_>>();
    let tags = indices.axis_tags()[..vector_axis]
        .iter()
        .copied()
        .chain(retained_axes.iter().map(|axis| input.axis_tags()[*axis]))
        .collect::<Vec<_>>();
    let partitions = indices.partitions()[..vector_axis]
        .iter()
        .copied()
        .chain(retained_axes.iter().map(|axis| input.partitions()[*axis]))
        .collect::<Vec<_>>();
    let result_shape = Shape::new(input.dtype(), &dimensions)?
        .with_axis_tags(&tags)?
        .with_partitions(&partitions)?;
    Ok(NdIndexingContract {
        result_shape,
        index_batch_rank: vector_axis,
        retained_axes,
    })
}

fn sorted_axes(axes: &[usize]) -> Vec<usize> {
    let mut sorted = axes.to_vec();
    sorted.sort_unstable();
    sorted
}

fn reduce_window_output_shape(
    input: Shape,
    window_dimensions: &[i64],
    window_strides: &[i64],
    base_dilations: &[i64],
    window_dilations: &[i64],
    padding: &[[i64; 2]],
) -> Result<Shape, Error> {
    let rank = input.rank();
    if rank == 0
        || window_dimensions.len() != rank
        || window_strides.len() != rank
        || base_dilations.len() != rank
        || window_dilations.len() != rank
        || padding.len() != rank
    {
        return Err(Error::InvalidWindow(
            "every window attribute must have one entry per input axis",
        ));
    }
    let mut dimensions = Vec::with_capacity(rank);
    for axis in 0..rank {
        dimensions.push(window_output_dimension(
            input.dimensions()[axis],
            window_dimensions[axis],
            window_strides[axis],
            base_dilations[axis],
            window_dilations[axis],
            padding[axis],
        )?);
    }
    Ok(Shape::new(input.dtype(), &dimensions)?
        .with_axis_tags(input.axis_tags())?
        .with_partitions(input.partitions())?
        .with_layout(input.layout())?)
}

fn window_output_dimension(
    input: i64,
    window: i64,
    stride: i64,
    base_dilation: i64,
    window_dilation: i64,
    padding: [i64; 2],
) -> Result<i64, Error> {
    if window <= 0 || stride <= 0 || base_dilation <= 0 || window_dilation <= 0 {
        return Err(Error::InvalidWindow(
            "window, stride, base dilation, and window dilation must be positive",
        ));
    }
    let dilated_input = if input == 0 {
        0
    } else {
        (i128::from(input) - 1) * i128::from(base_dilation) + 1
    };
    let dilated_window = (i128::from(window) - 1) * i128::from(window_dilation) + 1;
    let padded_input = dilated_input + i128::from(padding[0]) + i128::from(padding[1]);
    let output = if padded_input < dilated_window {
        0
    } else {
        (padded_input - dilated_window) / i128::from(stride) + 1
    };
    i64::try_from(output).map_err(|_| Error::InvalidWindow("output dimension overflows i64"))
}

fn convolution_output_shape(
    input: Shape,
    kernel: Shape,
    options: &ConvolutionOptions<'_>,
) -> Result<Shape, Error> {
    if input.dtype() != kernel.dtype() {
        return Err(Error::DTypeMismatch {
            left: input.dtype(),
            right: kernel.dtype(),
        });
    }
    if !matches!(input.dtype(), DType::F16 | DType::Bf16 | DType::F32) {
        return Err(Error::UnsupportedDType {
            operation: "convolution",
            dtype: input.dtype(),
        });
    }
    let rank = input.rank();
    if kernel.rank() != rank || !matches!(rank, 3 | 4) {
        return Err(Error::InvalidConvolution(
            "input and kernel must have equal rank 3 or 4",
        ));
    }
    let spatial_rank = rank - 2;
    if options.strides.len() != spatial_rank
        || options.padding.len() != spatial_rank
        || options.input_dilation.len() != spatial_rank
        || options.kernel_dilation.len() != spatial_rank
        || options.kernel_reversal.len() != spatial_rank
        || options.input_spatial_axes.len() != spatial_rank
        || options.kernel_spatial_axes.len() != spatial_rank
        || options.output_spatial_axes.len() != spatial_rank
    {
        return Err(Error::InvalidConvolution(
            "window and spatial-axis counts must equal rank minus two",
        ));
    }
    let input_axes = std::iter::once(options.input_batch_axis)
        .chain(std::iter::once(options.input_feature_axis))
        .chain(options.input_spatial_axes.iter().copied())
        .collect::<Vec<_>>();
    let kernel_axes = std::iter::once(options.kernel_input_feature_axis)
        .chain(std::iter::once(options.kernel_output_feature_axis))
        .chain(options.kernel_spatial_axes.iter().copied())
        .collect::<Vec<_>>();
    let output_axes = std::iter::once(options.output_batch_axis)
        .chain(std::iter::once(options.output_feature_axis))
        .chain(options.output_spatial_axes.iter().copied())
        .collect::<Vec<_>>();
    for axes in [&input_axes, &kernel_axes, &output_axes] {
        validate_axes(axes, rank, "convolution dimension numbers")?;
        if axes.len() != rank {
            return Err(Error::InvalidConvolution(
                "dimension numbers must classify every tensor axis",
            ));
        }
    }
    if options.feature_groups <= 0
        || options.batch_groups <= 0
        || (options.feature_groups > 1 && options.batch_groups > 1)
    {
        return Err(Error::InvalidConvolution(
            "group counts must be positive and only one group kind may exceed one",
        ));
    }
    let input_batch = input.dimensions()[options.input_batch_axis];
    let input_features = input.dimensions()[options.input_feature_axis];
    let kernel_input_features = kernel.dimensions()[options.kernel_input_feature_axis];
    let kernel_output_features = kernel.dimensions()[options.kernel_output_feature_axis];
    if input_batch % options.batch_groups != 0
        || input_features % options.feature_groups != 0
        || kernel_input_features != input_features / options.feature_groups
        || kernel_output_features % options.feature_groups != 0
        || kernel_output_features % options.batch_groups != 0
    {
        return Err(Error::InvalidConvolution(
            "feature or batch dimensions are incompatible with group counts",
        ));
    }

    let mut dimensions = vec![0; rank];
    let mut tags = vec![AxisTag::UNKNOWN; rank];
    let mut partitions = vec![Partition::Unspecified; rank];
    dimensions[options.output_batch_axis] = input_batch / options.batch_groups;
    tags[options.output_batch_axis] = input.axis_tags()[options.input_batch_axis];
    partitions[options.output_batch_axis] = input.partitions()[options.input_batch_axis];
    dimensions[options.output_feature_axis] = kernel_output_features;
    tags[options.output_feature_axis] = kernel.axis_tags()[options.kernel_output_feature_axis];
    partitions[options.output_feature_axis] =
        kernel.partitions()[options.kernel_output_feature_axis];
    for spatial in 0..spatial_rank {
        let input_axis = options.input_spatial_axes[spatial];
        let kernel_axis = options.kernel_spatial_axes[spatial];
        let output_axis = options.output_spatial_axes[spatial];
        dimensions[output_axis] = window_output_dimension(
            input.dimensions()[input_axis],
            kernel.dimensions()[kernel_axis],
            options.strides[spatial],
            options.input_dilation[spatial],
            options.kernel_dilation[spatial],
            options.padding[spatial],
        )?;
        tags[output_axis] = input.axis_tags()[input_axis];
        partitions[output_axis] = input.partitions()[input_axis];
    }
    Ok(Shape::new(input.dtype(), &dimensions)?
        .with_axis_tags(&tags)?
        .with_partitions(&partitions)?)
}

fn reduced_shape(input: Shape, axes: &[usize], dtype: DType) -> Result<Shape, Error> {
    let retained = (0..input.rank())
        .filter(|axis| !axes.contains(axis))
        .collect::<Vec<_>>();
    let dimensions = retained
        .iter()
        .map(|axis| input.dimensions()[*axis])
        .collect::<Vec<_>>();
    let tags = retained
        .iter()
        .map(|axis| input.axis_tags()[*axis])
        .collect::<Vec<_>>();
    let partitions = retained
        .iter()
        .map(|axis| input.partitions()[*axis])
        .collect::<Vec<_>>();
    Ok(Shape::new(dtype, &dimensions)?
        .with_axis_tags(&tags)?
        .with_partitions(&partitions)?)
}

fn comparison_type(dtype: DType) -> StableHloComparisonType {
    match dtype.class() {
        DTypeClass::Float => StableHloComparisonType::Float,
        DTypeClass::UnsignedInteger | DTypeClass::Boolean => StableHloComparisonType::Unsigned,
        DTypeClass::SignedInteger => StableHloComparisonType::Signed,
        DTypeClass::Complex => unreachable!("complex comparisons are rejected before lowering"),
    }
}

fn interpolation_dtype(dtype: DType) -> Result<DType, Error> {
    match dtype {
        DType::F64 => Ok(DType::F64),
        DType::F16
        | DType::Bf16
        | DType::F32
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::U8
        | DType::U16
        | DType::U32
        | DType::U64 => Ok(DType::F32),
        DType::Bool | DType::C64 | DType::C128 => Err(Error::UnsupportedDType {
            operation: "interpolation",
            dtype,
        }),
    }
}

fn require_unique_names<'a>(
    names: impl Iterator<Item = &'a str>,
    duplicate: fn(String) -> Error,
) -> Result<(), Error> {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(duplicate(name.to_owned()));
        }
    }
    Ok(())
}

fn ensure_disjoint(first: &[usize], second: &[usize], side: &'static str) -> Result<(), Error> {
    let first: HashSet<_> = first.iter().copied().collect();
    for &axis in second {
        if first.contains(&axis) {
            return Err(Error::DuplicateAxis { side, axis });
        }
    }
    Ok(())
}

fn unselected_axes(rank: usize, first: &[usize], second: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|axis| !first.contains(axis) && !second.contains(axis))
        .collect()
}

fn require_matching_shape_metadata(
    operation: &'static str,
    left: Shape,
    right: Shape,
) -> Result<(), Error> {
    if left.dimensions() != right.dimensions() {
        return Err(Error::ShapeMismatch {
            operation,
            left: left.dimensions().to_vec(),
            right: right.dimensions().to_vec(),
        });
    }
    for (field, matches) in [
        ("axis tags", left.axis_tags() == right.axis_tags()),
        (
            "partition metadata",
            left.partitions() == right.partitions(),
        ),
        ("physical layouts", left.layout() == right.layout()),
    ] {
        if !matches {
            return Err(Error::MetadataMismatch { operation, field });
        }
    }
    Ok(())
}

fn dense_literal(value: &Slice<'_>) -> Result<String, Error> {
    let elements = match value.dtype() {
        DType::Bool => value
            .items::<bool>()?
            .iter()
            .map(|value| value.to_string())
            .collect(),
        DType::I8 => decimal_elements(value.items::<i8>()?),
        DType::I16 => decimal_elements(value.items::<i16>()?),
        DType::I32 => decimal_elements(value.items::<i32>()?),
        DType::I64 => decimal_elements(value.items::<i64>()?),
        DType::U8 => decimal_elements(value.items::<u8>()?),
        DType::U16 => decimal_elements(value.items::<u16>()?),
        DType::U32 => decimal_elements(value.items::<u32>()?),
        DType::U64 => decimal_elements(value.items::<u64>()?),
        DType::F16 => value
            .items::<F16>()?
            .iter()
            .map(|value| float_literal_f16(*value))
            .collect(),
        DType::Bf16 => value
            .items::<BFloat16>()?
            .iter()
            .map(|value| float_literal_bf16(*value))
            .collect(),
        DType::F32 => value
            .items::<f32>()?
            .iter()
            .map(|value| float_literal_f32(*value))
            .collect(),
        DType::F64 => value
            .items::<f64>()?
            .iter()
            .map(|value| float_literal_f64(*value))
            .collect(),
        DType::C64 => value
            .items::<Complex64>()?
            .iter()
            .map(|value| {
                format!(
                    "({}, {})",
                    float_literal_f32(value.real),
                    float_literal_f32(value.imaginary)
                )
            })
            .collect(),
        DType::C128 => value
            .items::<Complex128>()?
            .iter()
            .map(|value| {
                format!(
                    "({}, {})",
                    float_literal_f64(value.real),
                    float_literal_f64(value.imaginary)
                )
            })
            .collect(),
    };
    let elements: Vec<String> = elements;
    let payload = match elements.as_slice() {
        [] => String::new(),
        [only] if value.shape().rank() == 0 => only.clone(),
        _ => format!("[{}]", elements.join(", ")),
    };
    let dimensions = value
        .shape()
        .dimensions()
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>();
    let tensor_type = if dimensions.is_empty() {
        format!("tensor<{}>", value.dtype().stablehlo_spelling())
    } else {
        format!(
            "tensor<{}x{}>",
            dimensions.join("x"),
            value.dtype().stablehlo_spelling()
        )
    };
    Ok(format!("dense<{payload}> : {tensor_type}"))
}

fn tensor_sharding_attribute<'context>(
    context: &'context Context,
    sharding: &Sharding,
    shape: Shape,
) -> Result<Option<MlirAttribute<'context>>, Error> {
    if !sharding.is_mesh() {
        return Ok(None);
    }
    let names = shape
        .partitions()
        .iter()
        .map(|partition| match partition {
            Partition::Sharded(tag) => Some(axis_name(*tag)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let dimensions = shape
        .partitions()
        .iter()
        .zip(&names)
        .map(|(partition, name)| match partition {
            Partition::Unspecified => ShardyDimension::Open,
            Partition::Replicated => ShardyDimension::Replicated,
            Partition::Sharded(_) => ShardyDimension::Sharded(
                name.as_deref().expect("sharded partition has an axis name"),
            ),
        })
        .collect::<Vec<_>>();
    let replicated = sharding
        .replicated_mesh_axes(shape)?
        .into_iter()
        .map(axis_name)
        .collect::<Vec<_>>();
    let replicated = replicated.iter().map(String::as_str).collect::<Vec<_>>();
    Ok(Some(context.shardy_tensor_sharding(
        "nml_mesh",
        &dimensions,
        &replicated,
    )?))
}

/// Compound operations lower into multiple primitive operations, so their
/// output placement cannot be attached by the generic one-operation lowering
/// path. Restore the semantic boundary explicitly and uniformly for every
/// compound lowering.
fn constrain_compound_result<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    value: MlirValue<'context>,
    type_: MlirType<'context>,
    sharding: &Sharding,
    shape: Shape,
) -> Result<MlirValue<'context>, Error> {
    let Some(attribute) = tensor_sharding_attribute(context, sharding, shape)? else {
        return Ok(value);
    };
    let constraint = context.sharding_constraint(value, type_, attribute)?;
    let constrained = constraint.result(0)?;
    block.append_operation(constraint)?;
    Ok(constrained)
}

/// Lowers the sequence recurrence as one StableHLO loop. Keeping the sequence
/// tensors in the loop state is deliberate: StableHLO regions do not capture
/// surrounding SSA values, while threading immutable values through the loop
/// keeps the operation valid and leaves loop-invariant forwarding to XLA.
fn lower_gated_delta_net<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    inputs: [MlirValue<'context>; 6],
    shapes: [Shape; 6],
    outputs_type: MlirType<'context>,
    state_type: MlirType<'context>,
) -> Result<(MlirValue<'context>, MlirValue<'context>), Error> {
    let dtype = shapes[5].dtype();
    let sequence = shapes[0].dimensions()[0];
    let heads = shapes[5].dimensions()[0];
    let value_size = shapes[5].dimensions()[1];
    let key_size = shapes[5].dimensions()[2];

    let step_type = context.ranked_tensor_type(DType::I32, &[])?;
    let bool_type = context.ranked_tensor_type(DType::Bool, &[])?;
    let scalar_type = context.ranked_tensor_type(dtype, &[])?;
    let query_slice_type = context.ranked_tensor_type(dtype, &[1, heads, key_size])?;
    let query_type = context.ranked_tensor_type(dtype, &[heads, key_size])?;
    let value_slice_type = context.ranked_tensor_type(dtype, &[1, heads, value_size])?;
    let value_type = context.ranked_tensor_type(dtype, &[heads, value_size])?;
    let gate_slice_type = context.ranked_tensor_type(dtype, &[1, heads])?;
    let gate_type = context.ranked_tensor_type(dtype, &[heads])?;
    let expanded_value_type = context.ranked_tensor_type(dtype, &[heads, value_size, 1])?;
    let expanded_key_type = context.ranked_tensor_type(dtype, &[heads, 1, key_size])?;

    let step = dense_scalar(context, block, step_type, "0")?;
    let zero = dense_scalar(context, block, scalar_type, "0.0")?;
    let outputs = append_mlir_result(block, context.broadcast_in_dim(zero, outputs_type, &[])?)?;
    let state_types = [
        step_type,
        inputs[0].type_(),
        inputs[1].type_(),
        inputs[2].type_(),
        inputs[3].type_(),
        inputs[4].type_(),
        state_type,
        outputs_type,
    ];
    let initial = [
        step, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], outputs,
    ];

    let mut condition_block = Block::new(context, &state_types)?;
    let condition_step = condition_block.argument(0)?;
    let sequence = dense_scalar(
        context,
        &mut condition_block,
        step_type,
        &sequence.to_string(),
    )?;
    let predicate = append_mlir_result(
        &mut condition_block,
        context.compare(
            condition_step,
            sequence,
            bool_type,
            StableHloComparison::Lt,
            StableHloComparisonType::Signed,
        )?,
    )?;
    condition_block.append_operation(context.stablehlo_return(&[predicate])?)?;
    let mut condition = Region::new(context)?;
    condition.append_block(condition_block)?;

    let mut body_block = Block::new(context, &state_types)?;
    let body_values = (0..state_types.len())
        .map(|index| body_block.argument(index))
        .collect::<Result<Vec<_>, _>>()?;
    let step = body_values[0];
    let zero_index = dense_scalar(context, &mut body_block, step_type, "0")?;
    let one_index = dense_scalar(context, &mut body_block, step_type, "1")?;
    let starts = [step, zero_index, zero_index];

    let query = append_mlir_result(
        &mut body_block,
        context.dynamic_slice(
            body_values[1],
            &starts,
            query_slice_type,
            &[1, heads, key_size],
        )?,
    )?;
    let query = append_mlir_result(&mut body_block, context.reshape(query, query_type)?)?;
    let key = append_mlir_result(
        &mut body_block,
        context.dynamic_slice(
            body_values[2],
            &starts,
            query_slice_type,
            &[1, heads, key_size],
        )?,
    )?;
    let key = append_mlir_result(&mut body_block, context.reshape(key, query_type)?)?;
    let value = append_mlir_result(
        &mut body_block,
        context.dynamic_slice(
            body_values[3],
            &starts,
            value_slice_type,
            &[1, heads, value_size],
        )?,
    )?;
    let value = append_mlir_result(&mut body_block, context.reshape(value, value_type)?)?;
    let gate_starts = [step, zero_index];
    let alpha = append_mlir_result(
        &mut body_block,
        context.dynamic_slice(body_values[4], &gate_starts, gate_slice_type, &[1, heads])?,
    )?;
    let alpha = append_mlir_result(&mut body_block, context.reshape(alpha, gate_type)?)?;
    let beta = append_mlir_result(
        &mut body_block,
        context.dynamic_slice(body_values[5], &gate_starts, gate_slice_type, &[1, heads])?,
    )?;
    let beta = append_mlir_result(&mut body_block, context.reshape(beta, gate_type)?)?;

    let state = body_values[6];
    let projected = append_mlir_result(
        &mut body_block,
        context.dot_general(state, key, value_type, &[0], &[0], &[2], &[1])?,
    )?;
    let alpha_values = append_mlir_result(
        &mut body_block,
        context.broadcast_in_dim(alpha, value_type, &[0])?,
    )?;
    let projected = append_mlir_result(
        &mut body_block,
        context.binary(
            StableHloBinary::Multiply,
            projected,
            alpha_values,
            value_type,
        )?,
    )?;
    let delta = append_mlir_result(
        &mut body_block,
        context.binary(StableHloBinary::Subtract, value, projected, value_type)?,
    )?;
    let beta_values = append_mlir_result(
        &mut body_block,
        context.broadcast_in_dim(beta, value_type, &[0])?,
    )?;
    let delta = append_mlir_result(
        &mut body_block,
        context.binary(StableHloBinary::Multiply, delta, beta_values, value_type)?,
    )?;

    let delta = append_mlir_result(
        &mut body_block,
        context.reshape(delta, expanded_value_type)?,
    )?;
    let delta = append_mlir_result(
        &mut body_block,
        context.broadcast_in_dim(delta, state_type, &[0, 1, 2])?,
    )?;
    let key = append_mlir_result(&mut body_block, context.reshape(key, expanded_key_type)?)?;
    let key = append_mlir_result(
        &mut body_block,
        context.broadcast_in_dim(key, state_type, &[0, 1, 2])?,
    )?;
    let correction = append_mlir_result(
        &mut body_block,
        context.binary(StableHloBinary::Multiply, delta, key, state_type)?,
    )?;
    let alpha_state = append_mlir_result(
        &mut body_block,
        context.broadcast_in_dim(alpha, state_type, &[0])?,
    )?;
    let retained = append_mlir_result(
        &mut body_block,
        context.binary(StableHloBinary::Multiply, state, alpha_state, state_type)?,
    )?;
    let next_state = append_mlir_result(
        &mut body_block,
        context.add(retained, correction, state_type)?,
    )?;
    let output = append_mlir_result(
        &mut body_block,
        context.dot_general(next_state, query, value_type, &[0], &[0], &[2], &[1])?,
    )?;
    let output = append_mlir_result(&mut body_block, context.reshape(output, value_slice_type)?)?;
    let next_outputs = append_mlir_result(
        &mut body_block,
        context.dynamic_update_slice(body_values[7], output, &starts, outputs_type)?,
    )?;
    let next_step = append_mlir_result(&mut body_block, context.add(step, one_index, step_type)?)?;
    body_block.append_operation(context.stablehlo_return(&[
        next_step,
        body_values[1],
        body_values[2],
        body_values[3],
        body_values[4],
        body_values[5],
        next_state,
        next_outputs,
    ])?)?;
    let mut body = Region::new(context)?;
    body.append_block(body_block)?;

    let operation = context.stablehlo_while(&initial, &state_types, condition, body)?;
    let final_state = operation.result(6)?;
    let outputs = operation.result(7)?;
    block.append_operation(operation)?;
    Ok((outputs, final_state))
}

fn dense_scalar<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    type_: MlirType<'context>,
    literal: &str,
) -> Result<MlirValue<'context>, Error> {
    let attribute = context.parse_attribute(&format!("dense<{literal}> : {}", type_.text()))?;
    append_mlir_result(block, context.constant(type_, attribute)?)
}

#[allow(clippy::too_many_arguments)]
fn lower_expert_parallel_moe<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    sharding: &Sharding,
    inputs: [MlirValue<'context>; 6],
    shapes: [Shape; 6],
    result_type: MlirType<'context>,
    result_shape: Shape,
    expert_axis: AxisTag,
    activation: MoeActivation,
    experts_per_token: usize,
    block_size: usize,
    operation_index: usize,
) -> Result<MlirValue<'context>, Error> {
    let mesh_axes = sharding.mesh_axes().collect::<Vec<_>>();
    let expert_axis_index = mesh_axes
        .iter()
        .position(|(tag, _)| *tag == expert_axis)
        .ok_or(Error::InvalidMoe(
            "expert partition references an absent mesh axis",
        ))?;
    let expert_partitions = mesh_axes[expert_axis_index].1;
    let local_shapes = shapes
        .iter()
        .map(|shape| manual_axis_local_shape(*shape, sharding, expert_axis))
        .collect::<Result<Vec<_>, _>>()?;
    let local_result_shape = manual_axis_local_shape(result_shape, sharding, expert_axis)?;
    let local_types = local_shapes
        .iter()
        .map(|shape| context.ranked_tensor_type(shape.dtype(), shape.dimensions()))
        .collect::<Result<Vec<_>, _>>()?;
    let local_result_type =
        context.ranked_tensor_type(local_result_shape.dtype(), local_result_shape.dimensions())?;

    let input_shardings = shapes
        .iter()
        .map(|shape| {
            tensor_sharding_attribute(context, sharding, *shape)?.ok_or(Error::InvalidMoe(
                "expert parallelism requires an explicit Shardy mesh",
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let input_shardings = context.shardy_per_value_sharding(&input_shardings)?;
    let result_sharding = tensor_sharding_attribute(context, sharding, result_shape)?.ok_or(
        Error::InvalidMoe("expert parallelism requires an explicit Shardy mesh"),
    )?;
    let result_shardings = context.shardy_per_value_sharding(&[result_sharding])?;
    let expert_axis_name = axis_name(expert_axis);
    let manual_axes = context.shardy_manual_axes(&[expert_axis_name.as_str()])?;

    let mut local_block = Block::new(context, &local_types)?;
    let local_inputs = (0..local_types.len())
        .map(|index| local_block.argument(index))
        .collect::<Result<Vec<_>, _>>()?;
    let partition_type = context.ranked_tensor_type(DType::U32, &[])?;
    let partition_id = append_mlir_result(&mut local_block, context.partition_id(partition_type)?)?;
    let later_product = mesh_axes[expert_axis_index + 1..]
        .iter()
        .try_fold(1usize, |product, (_, size)| product.checked_mul(*size))
        .ok_or(Error::InvalidMoe("mesh coordinate stride overflows"))?;
    let later = unsigned_scalar(context, &mut local_block, later_product)?;
    let expert_partitions_value = unsigned_scalar(context, &mut local_block, expert_partitions)?;
    let coordinate = append_mlir_result(
        &mut local_block,
        context.binary(StableHloBinary::Divide, partition_id, later, partition_type)?,
    )?;
    let coordinate = append_mlir_result(
        &mut local_block,
        context.binary(
            StableHloBinary::Remainder,
            coordinate,
            expert_partitions_value,
            partition_type,
        )?,
    )?;
    let local_experts = usize::try_from(local_shapes[2].dimensions()[0])
        .map_err(|_| Error::InvalidMoe("local expert count exceeds usize"))?;
    let local_experts_value = unsigned_scalar(context, &mut local_block, local_experts)?;
    let expert_offset = append_mlir_result(
        &mut local_block,
        context.binary(
            StableHloBinary::Multiply,
            coordinate,
            local_experts_value,
            partition_type,
        )?,
    )?;
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[])?;
    let expert_offset = append_mlir_result(
        &mut local_block,
        context.convert(expert_offset, scalar_i32)?,
    )?;

    let partial = moe_backend::lower(
        context,
        &mut local_block,
        moe_backend::Inputs {
            hidden: local_inputs[0],
            routing_weights: local_inputs[1],
            gate_up_weights: local_inputs[2],
            down_weights: local_inputs[3],
            sorted_assignments: local_inputs[4],
            block_experts: local_inputs[5],
            expert_offset: Some(expert_offset),
            hidden_shape: local_shapes[0],
            gate_up_shape: local_shapes[2],
            down_shape: local_shapes[3],
            schedule_shape: local_shapes[4],
            block_experts_shape: local_shapes[5],
            result_type: local_result_type,
            activation,
            experts_per_token,
            block_size,
        },
    )?;
    let replica_groups = expert_replica_groups(&mesh_axes, expert_axis_index)?;
    let reduce = context.all_reduce(
        partial,
        local_result_type,
        &replica_groups,
        collective_channel(operation_index)?,
        reduction_region(context, result_shape.dtype(), Reduction::Sum)?,
    )?;
    let reduced = reduce.result(0)?;
    local_block.append_operation(reduce)?;
    local_block.append_operation(context.shardy_return(&[reduced])?)?;
    let mut body = Region::new(context)?;
    body.append_block(local_block)?;
    let manual = context.shardy_manual_computation(
        &inputs,
        &[result_type],
        input_shardings,
        result_shardings,
        manual_axes,
        body,
    )?;
    let result = manual.result(0)?;
    block.append_operation(manual)?;
    Ok(result)
}

fn unsigned_scalar<'context>(
    context: &'context Context,
    block: &mut Block<'context>,
    value: usize,
) -> Result<MlirValue<'context>, Error> {
    let value =
        u32::try_from(value).map_err(|_| Error::InvalidMoe("mesh coordinate exceeds U32"))?;
    let type_ = context.ranked_tensor_type(DType::U32, &[])?;
    let attribute = context.parse_attribute(&format!("dense<{value}> : tensor<ui32>"))?;
    append_mlir_result(block, context.constant(type_, attribute)?)
}

fn manual_axis_local_shape(
    shape: Shape,
    sharding: &Sharding,
    manual_axis: AxisTag,
) -> Result<Shape, Error> {
    sharding.validate_shape(shape)?;
    let axis_size = sharding
        .mesh_axes()
        .find_map(|(axis, size)| (axis == manual_axis).then_some(size))
        .ok_or(Error::InvalidMoe(
            "manual computation references an absent mesh axis",
        ))?;
    let axis_size = i64::try_from(axis_size)
        .map_err(|_| Error::InvalidMoe("manual mesh axis size exceeds I64"))?;
    let dimensions = shape
        .dimensions()
        .iter()
        .zip(shape.partitions())
        .map(|(dimension, partition)| match partition {
            Partition::Sharded(axis) if *axis == manual_axis => *dimension / axis_size,
            _ => *dimension,
        })
        .collect::<Vec<_>>();
    Ok(Shape::new(shape.dtype(), &dimensions)?
        .with_axis_tags(shape.axis_tags())?
        .with_partitions(shape.partitions())?
        .with_layout(shape.layout())?)
}

fn expert_replica_groups(
    mesh_axes: &[(AxisTag, usize)],
    expert_axis_index: usize,
) -> Result<Vec<Vec<i64>>, Error> {
    let total = mesh_axes
        .iter()
        .try_fold(1usize, |product, (_, size)| product.checked_mul(*size))
        .ok_or(Error::InvalidMoe("mesh device count overflows"))?;
    let expert_partitions = mesh_axes[expert_axis_index].1;
    let later_product = mesh_axes[expert_axis_index + 1..]
        .iter()
        .try_fold(1usize, |product, (_, size)| product.checked_mul(*size))
        .ok_or(Error::InvalidMoe("mesh coordinate stride overflows"))?;
    let mut groups = Vec::with_capacity(total / expert_partitions);
    for root in 0..total {
        if (root / later_product) % expert_partitions != 0 {
            continue;
        }
        groups.push(
            (0..expert_partitions)
                .map(|coordinate| {
                    i64::try_from(root + coordinate * later_product)
                        .map_err(|_| Error::InvalidMoe("mesh device id exceeds I64"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    Ok(groups)
}

fn axis_name(tag: AxisTag) -> String {
    format!("axis_{}", tag.identifier())
}

fn decimal_elements<T: ToString>(values: &[T]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn float_literal_f16(value: F16) -> String {
    if value.to_f32().is_finite() {
        finite_float_literal(value.to_f32() as f64)
    } else {
        format!("0x{:04X}", value.to_bits())
    }
}

fn float_literal_bf16(value: BFloat16) -> String {
    if value.to_f32().is_finite() {
        finite_float_literal(value.to_f32() as f64)
    } else {
        format!("0x{:04X}", value.to_bits())
    }
}

fn float_literal_f32(value: f32) -> String {
    if value.is_finite() {
        finite_float_literal(value as f64)
    } else {
        format!("0x{:08X}", value.to_bits())
    }
}

fn float_literal_f64(value: f64) -> String {
    if value.is_finite() {
        finite_float_literal(value)
    } else {
        format!("0x{:016X}", value.to_bits())
    }
}

fn finite_float_literal(value: f64) -> String {
    debug_assert!(value.is_finite());
    format!("{value:.17e}")
}

fn reduction_region<'context>(
    context: &'context Context,
    dtype: DType,
    reduction: Reduction,
) -> Result<Region<'context>, Error> {
    let scalar_type = context.ranked_tensor_type(dtype, &[])?;
    let mut block = Block::new(context, &[scalar_type, scalar_type])?;
    let left = block.argument(0)?;
    let right = block.argument(1)?;
    let combine = match reduction {
        Reduction::Sum => context.add(left, right, scalar_type)?,
        Reduction::Maximum => context.binary(StableHloBinary::Maximum, left, right, scalar_type)?,
        Reduction::Minimum => context.binary(StableHloBinary::Minimum, left, right, scalar_type)?,
    };
    let combined = combine.result(0)?;
    block.append_operation(combine)?;
    block.append_operation(context.stablehlo_return(&[combined])?)?;
    let mut region = Region::new(context)?;
    region.append_block(block)?;
    Ok(region)
}

fn collective_channel(operation_index: usize) -> Result<i64, Error> {
    let channel = operation_index
        .checked_add(1)
        .ok_or(Error::InvalidCollective("channel id overflows"))?;
    i64::try_from(channel).map_err(|_| Error::InvalidCollective("channel id exceeds I64"))
}

fn append_mlir_result<'context>(
    block: &mut Block<'context>,
    operation: MlirOperation<'context>,
) -> Result<MlirValue<'context>, Error> {
    let result = operation.result(0)?;
    block.append_operation(operation)?;
    Ok(result)
}

fn sort_comparator_region<'context>(
    context: &'context Context,
    dtype: DType,
    descending: bool,
) -> Result<Region<'context>, Error> {
    let value_type = context.ranked_tensor_type(dtype, &[])?;
    let index_type = context.ranked_tensor_type(DType::I32, &[])?;
    let bool_type = context.ranked_tensor_type(DType::Bool, &[])?;
    let mut block = Block::new(context, &[value_type, value_type, index_type, index_type])?;
    let left = block.argument(0)?;
    let right = block.argument(1)?;
    let left_index = block.argument(2)?;
    let right_index = block.argument(3)?;
    let value_precedes = append_mlir_result(
        &mut block,
        context.compare(
            left,
            right,
            bool_type,
            if descending {
                StableHloComparison::Gt
            } else {
                StableHloComparison::Lt
            },
            comparison_type(dtype),
        )?,
    )?;
    let values_equal = append_mlir_result(
        &mut block,
        context.compare(
            left,
            right,
            bool_type,
            StableHloComparison::Eq,
            comparison_type(dtype),
        )?,
    )?;
    let lower_index = append_mlir_result(
        &mut block,
        context.compare(
            left_index,
            right_index,
            bool_type,
            StableHloComparison::Lt,
            StableHloComparisonType::Signed,
        )?,
    )?;
    let equal_with_lower_index = append_mlir_result(
        &mut block,
        context.binary(StableHloBinary::And, values_equal, lower_index, bool_type)?,
    )?;

    let precedes = if dtype.class() == DTypeClass::Float {
        let left_nan = append_mlir_result(
            &mut block,
            context.compare(
                left,
                left,
                bool_type,
                StableHloComparison::Ne,
                StableHloComparisonType::Float,
            )?,
        )?;
        let right_nan = append_mlir_result(
            &mut block,
            context.compare(
                right,
                right,
                bool_type,
                StableHloComparison::Ne,
                StableHloComparisonType::Float,
            )?,
        )?;
        let left_finite = append_mlir_result(
            &mut block,
            context.unary_math(StableHloUnary::Not, left_nan, bool_type)?,
        )?;
        let right_finite = append_mlir_result(
            &mut block,
            context.unary_math(StableHloUnary::Not, right_nan, bool_type)?,
        )?;
        let nan_precedes = append_mlir_result(
            &mut block,
            context.binary(
                StableHloBinary::And,
                if descending { left_nan } else { left_finite },
                if descending { right_finite } else { right_nan },
                bool_type,
            )?,
        )?;
        let both_nan = append_mlir_result(
            &mut block,
            context.binary(StableHloBinary::And, left_nan, right_nan, bool_type)?,
        )?;
        let nan_with_lower_index = append_mlir_result(
            &mut block,
            context.binary(StableHloBinary::And, both_nan, lower_index, bool_type)?,
        )?;
        let ties = append_mlir_result(
            &mut block,
            context.binary(
                StableHloBinary::Or,
                equal_with_lower_index,
                nan_with_lower_index,
                bool_type,
            )?,
        )?;
        let ordered_or_nan = append_mlir_result(
            &mut block,
            context.binary(StableHloBinary::Or, value_precedes, nan_precedes, bool_type)?,
        )?;
        append_mlir_result(
            &mut block,
            context.binary(StableHloBinary::Or, ordered_or_nan, ties, bool_type)?,
        )?
    } else {
        append_mlir_result(
            &mut block,
            context.binary(
                StableHloBinary::Or,
                value_precedes,
                equal_with_lower_index,
                bool_type,
            )?,
        )?
    };
    block.append_operation(context.stablehlo_return(&[precedes])?)?;
    let mut region = Region::new(context)?;
    region.append_block(block)?;
    Ok(region)
}

fn mlir_value<'context>(
    values: &[Option<MlirValue<'context>>],
    index: usize,
) -> MlirValue<'context> {
    values[index].expect("validated NML operation order must define every operand")
}

fn axes_i64(axes: &[usize]) -> Vec<i64> {
    axes.iter()
        .map(|axis| i64::try_from(*axis).expect("rank-8 axis always fits i64"))
        .collect()
}
