//! Typed, deterministic construction of NML's StableHLO program subset.
//!
//! Validation happens while operations are authored. Consequently an invalid
//! graph never becomes an MLIR module and cannot reach XLA or a PJRT plugin.

#![forbid(unsafe_code)]

use nml_mlir::{
    Attribute as MlirAttribute, Block, Context, Error as MlirError, Module, Region,
    ShardyDimension, StableHloBinary, StableHloComparison, StableHloComparisonType,
    StableHloFftType, StableHloUnary, Value as MlirValue,
};
use nml_sharding::Sharding;
use nml_tensor::{Element, Slice};
use nml_types::{
    AxisTag, BFloat16, Complex64, Complex128, DType, DTypeClass, F16, Layout, Partition, Shape,
    ShapeError,
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

    pub fn minimum(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Minimum)
    }

    pub fn maximum(&mut self, left: Tensor, right: Tensor) -> Result<Tensor, Error> {
        self.binary(left, right, Binary::Maximum)
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

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
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
                    },
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
                        match input_dtype.class() {
                            DTypeClass::Float => StableHloComparisonType::Float,
                            DTypeClass::UnsignedInteger | DTypeClass::Boolean => {
                                StableHloComparisonType::Unsigned
                            }
                            DTypeClass::SignedInteger => StableHloComparisonType::Signed,
                            DTypeClass::Complex => unreachable!("complex comparisons are rejected"),
                        },
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
                            Unary::Exp => StableHloUnary::Exponential,
                            Unary::Log => StableHloUnary::Log,
                            Unary::Sqrt => StableHloUnary::Sqrt,
                            Unary::Rsqrt => StableHloUnary::Rsqrt,
                            Unary::Tanh => StableHloUnary::Tanh,
                            Unary::Sin => StableHloUnary::Sine,
                            Unary::Cos => StableHloUnary::Cosine,
                            Unary::Logistic => StableHloUnary::Logistic,
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
    BroadcastInDim {
        input: usize,
        result: usize,
        dimensions: Vec<usize>,
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
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Sin,
    Cos,
    Logistic,
}

impl Unary {
    const fn name(self) -> &'static str {
        match self {
            Self::Negate => "negate",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Tanh => "tanh",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Logistic => "sigmoid",
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
            .map(|value| float_literal(value.to_f32() as f64))
            .collect(),
        DType::Bf16 => value
            .items::<BFloat16>()?
            .iter()
            .map(|value| float_literal(value.to_f32() as f64))
            .collect(),
        DType::F32 => value
            .items::<f32>()?
            .iter()
            .map(|value| float_literal(*value as f64))
            .collect(),
        DType::F64 => value
            .items::<f64>()?
            .iter()
            .map(|value| float_literal(*value))
            .collect(),
        DType::C64 => value
            .items::<Complex64>()?
            .iter()
            .map(|value| {
                format!(
                    "({}, {})",
                    float_literal(value.real as f64),
                    float_literal(value.imaginary as f64)
                )
            })
            .collect(),
        DType::C128 => value
            .items::<Complex128>()?
            .iter()
            .map(|value| {
                format!(
                    "({}, {})",
                    float_literal(value.real),
                    float_literal(value.imaginary)
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

fn axis_name(tag: AxisTag) -> String {
    format!("axis_{}", tag.identifier())
}

fn decimal_elements<T: ToString>(values: &[T]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn float_literal(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else if value == f64::INFINITY {
        "inf".to_owned()
    } else if value == f64::NEG_INFINITY {
        "-inf".to_owned()
    } else {
        format!("{value:.17e}")
    }
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
