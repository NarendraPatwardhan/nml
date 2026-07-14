//! Typed, deterministic construction of NML's StableHLO program subset.
//!
//! Validation happens while operations are authored. Consequently an invalid
//! graph never becomes an MLIR module and cannot reach XLA or a PJRT plugin.

#![forbid(unsafe_code)]

use nml_mlir::{
    Block, Context, Error as MlirError, Module, Region, StableHloFftType, Value as MlirValue,
};
use nml_types::{DType, Layout, Shape, ShapeError};
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
        self.require_local(left)?;
        self.require_local(right)?;
        if left.shape.dtype() != right.shape.dtype() {
            return Err(Error::DTypeMismatch {
                left: left.shape.dtype(),
                right: right.shape.dtype(),
            });
        }
        require_matching_shape_metadata("add", left.shape, right.shape)?;
        let result = self.push_value("add", left.shape);
        self.operations.push(Operation::Add {
            left: left.value,
            right: right.value,
            result: result.value,
        });
        Ok(result)
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

    /// Returns MLIR's canonical text for the same owned module sent to XLA.
    pub fn stablehlo(&self) -> Result<String, Error> {
        let context = Context::new();
        Ok(self.module(&context)?.text())
    }

    pub fn module<'context>(&self, context: &'context Context) -> Result<Module<'context>, Error> {
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
                Operation::Add {
                    left,
                    right,
                    result,
                } => (
                    context.add(
                        mlir_value(&values, *left),
                        mlir_value(&values, *right),
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
        let function = context.function_with_input_aliases(
            "main",
            &input_types,
            &result_types,
            &input_aliases,
            body,
        )?;
        let mut module = context.empty_module()?;
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
    Add {
        left: usize,
        right: usize,
        result: usize,
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
