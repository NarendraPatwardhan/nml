//! Typed, deterministic construction of NML's StableHLO program subset.
//!
//! Validation happens while operations are authored. Consequently an invalid
//! graph never becomes an MLIR module and cannot reach XLA or a PJRT plugin.

#![forbid(unsafe_code)]

mod attention_backend;
mod ordinary_attention;
mod paged_attention;

use nml_mlir::{
    Attribute as MlirAttribute, Block, Context, Error as MlirError, Module, Region,
    ShardyDimension, StableHloBinary, StableHloComparison, StableHloComparisonType,
    StableHloFftType, StableHloUnary, Type as MlirType, Value as MlirValue,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionOptions {
    pub causal: bool,
    pub sliding_window: Option<usize>,
    pub scale: Option<f64>,
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
            axis,
            slice_size,
            collapse_axis,
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

        for operation in &self.operations {
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
                    axis,
                    slice_size,
                    collapse_axis,
                } => {
                    let input_rank = self.values[*input].shape.rank();
                    let indices_rank = self.values[*indices].shape.rank();
                    let retained_rank = input_rank - usize::from(*collapse_axis);
                    let offset_dims = (indices_rank..indices_rank + retained_rank)
                        .map(|dimension| dimension as i64)
                        .collect::<Vec<_>>();
                    let mut slice_sizes = self.values[*input].shape.dimensions().to_vec();
                    slice_sizes[*axis] = *slice_size;
                    let collapsed = collapse_axis.then_some(*axis as i64);
                    (
                        context.gather(
                            mlir_value(&values, *input),
                            mlir_value(&values, *indices),
                            types[*result],
                            &offset_dims,
                            collapsed.as_slice(),
                            &[],
                            &[],
                            &[*axis as i64],
                            indices_rank as i64,
                            &slice_sizes,
                            false,
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
                } => {
                    let scalar_type =
                        context.ranked_tensor_type(self.values[*input].shape.dtype(), &[])?;
                    let mut reduction_block = Block::new(context, &[scalar_type, scalar_type])?;
                    let left = reduction_block.argument(0)?;
                    let right = reduction_block.argument(1)?;
                    let combine = match reduction {
                        Reduction::Sum => context.add(left, right, scalar_type)?,
                        Reduction::Maximum => {
                            context.binary(StableHloBinary::Maximum, left, right, scalar_type)?
                        }
                        Reduction::Minimum => {
                            context.binary(StableHloBinary::Minimum, left, right, scalar_type)?
                        }
                    };
                    let combined = combine.result(0)?;
                    reduction_block.append_operation(combine)?;
                    reduction_block.append_operation(context.stablehlo_return(&[combined])?)?;
                    let mut reduction_body = Region::new(context)?;
                    reduction_body.append_block(reduction_block)?;
                    (
                        context.reduce(
                            mlir_value(&values, *input),
                            mlir_value(&values, *init),
                            types[*result],
                            &axes_i64(axes),
                            reduction_body,
                        )?,
                        *result,
                    )
                }
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
                Operation::PagedAttention { .. } => {
                    unreachable!("paged attention is lowered before scalar operations")
                }
                Operation::Attention { .. } => {
                    unreachable!("attention is lowered before scalar operations")
                }
                Operation::ArgMax { .. } => {
                    unreachable!("argmax is lowered before single-result operations")
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
                .map(|(tag, size)| (axis_name(tag), size as i64))
                .collect::<Vec<_>>();
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
        axis: usize,
        slice_size: i64,
        collapse_axis: bool,
    },
    Reduce {
        input: usize,
        init: usize,
        result: usize,
        axes: Vec<usize>,
        reduction: Reduction,
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
}

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Sum,
    Maximum,
    Minimum,
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
