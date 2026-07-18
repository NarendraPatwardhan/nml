//! Typed boundary between an isolated TTIR module and its StableHLO call site.
//!
//! XLA's Triton custom call is string-configured, but NML does not let stringly
//! launch metadata leak upward.  This specification owns the immutable TTIR,
//! tensor ABI, layouts, aliases, and launch validation used to create the one
//! StableHLO operation.

use super::{DType, Error, require_identifier};
use nml_mlir::{
    Context, Operation, OutputOperandAlias, TritonCustomCall, Type, Value as MlirValue,
};
use nml_types::DType as NmlDType;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub dtype: DType,
    pub shape: Vec<i64>,
}

impl TensorSpec {
    pub fn new(dtype: DType, shape: &[i64]) -> Result<Self, Error> {
        if shape.len() > 8 || shape.iter().any(|dimension| *dimension <= 0) {
            return Err(Error::InvalidKernelSpec(
                "kernel ABI tensors must have rank at most eight and positive dimensions",
            ));
        }
        Ok(Self {
            dtype,
            shape: shape.to_vec(),
        })
    }

    fn mlir_type<'context>(&self, context: &'context Context) -> Result<Type<'context>, Error> {
        Ok(context.ranked_tensor_type(to_nml_dtype(self.dtype), &self.shape)?)
    }

    fn row_major_layout(&self) -> Vec<i64> {
        (0..self.shape.len() as i64).rev().collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputAlias {
    pub output: usize,
    pub input: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelLaunch {
    pub grid: [i32; 3],
    pub warps: i32,
    pub stages: i32,
}

impl KernelLaunch {
    fn validate(self) -> Result<Self, Error> {
        if self.grid.iter().any(|dimension| *dimension <= 0)
            || self.warps <= 0
            || !(self.warps as u32).is_power_of_two()
            || self.stages <= 0
        {
            return Err(Error::InvalidKernelSpec(
                "launch grid, power-of-two warp count, and stage count must be positive",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelSpec {
    name: String,
    ir: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    aliases: Vec<OutputAlias>,
}

impl KernelSpec {
    pub fn new(
        name: &str,
        ir: String,
        inputs: Vec<TensorSpec>,
        outputs: Vec<TensorSpec>,
        aliases: Vec<OutputAlias>,
    ) -> Result<Self, Error> {
        require_identifier(name)?;
        if inputs.is_empty() || outputs.is_empty() {
            return Err(Error::InvalidKernelSpec(
                "kernel input and output ABIs must be nonempty",
            ));
        }

        // Builder::finish already verifies authored kernels, but KernelSpec is
        // the independent boundary that embeds TTIR in StableHLO. Reparse here
        // so a malformed or name-mismatched string can never become an opaque
        // backend_config whose failure is deferred to XLA.
        let ttir_context = Context::new_ttir();
        let module = ttir_context.parse_module(&ir)?;
        module.verify()?;
        let ir = module.text();
        if ir.matches("tt.func public @").count() != 1
            || !ir.contains(&format!("tt.func public @{name}("))
        {
            return Err(Error::InvalidKernelSpec(
                "kernel IR must contain exactly one name-consistent public TTIR function",
            ));
        }
        for (index, alias) in aliases.iter().enumerate() {
            if alias.output >= outputs.len()
                || alias.input >= inputs.len()
                || aliases[..index]
                    .iter()
                    .any(|prior| prior.output == alias.output)
                || outputs[alias.output] != inputs[alias.input]
            {
                return Err(Error::InvalidKernelSpec(
                    "output aliases must be unique, in bounds, and tensor-compatible",
                ));
            }
        }
        Ok(Self {
            name: name.to_owned(),
            ir,
            inputs,
            outputs,
            aliases,
        })
    }

    pub fn lower<'context>(
        &self,
        context: &'context Context,
        operands: &[MlirValue<'context>],
        launch: KernelLaunch,
    ) -> Result<Operation<'context>, Error> {
        let launch = launch.validate()?;
        if operands.len() != self.inputs.len() {
            return Err(Error::InvalidKernelSpec(
                "kernel operand count does not match its typed input ABI",
            ));
        }
        for (operand, expected) in operands.iter().zip(&self.inputs) {
            let expected = expected.mlir_type(context)?;
            if operand.type_().text() != expected.text() {
                return Err(Error::InvalidKernelSpec(
                    "kernel operand type does not match its typed input ABI",
                ));
            }
        }

        let result_types = self
            .outputs
            .iter()
            .map(|output| output.mlir_type(context))
            .collect::<Result<Vec<_>, _>>()?;
        let operand_layout_storage = self
            .inputs
            .iter()
            .map(TensorSpec::row_major_layout)
            .collect::<Vec<_>>();
        let operand_layouts = operand_layout_storage
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let result_layout_storage = self
            .outputs
            .iter()
            .map(TensorSpec::row_major_layout)
            .collect::<Vec<_>>();
        let result_layouts = result_layout_storage
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let aliases = self
            .aliases
            .iter()
            .map(|alias| OutputOperandAlias {
                output_index: alias.output,
                operand_index: alias.input,
            })
            .collect::<Vec<_>>();

        Ok(context.triton_custom_call(
            operands,
            &result_types,
            TritonCustomCall {
                name: &self.name,
                ir: &self.ir,
                grid: launch.grid,
                num_stages: launch.stages,
                num_warps: launch.warps,
                operand_layouts: &operand_layouts,
                result_layouts: &result_layouts,
                output_operand_aliases: &aliases,
            },
        )?)
    }
}

const fn to_nml_dtype(dtype: DType) -> NmlDType {
    match dtype {
        DType::I1 => NmlDType::Bool,
        DType::I8 => NmlDType::I8,
        DType::U8 => NmlDType::U8,
        DType::I16 => NmlDType::I16,
        DType::I32 => NmlDType::I32,
        DType::I64 => NmlDType::I64,
        DType::F16 => NmlDType::F16,
        DType::Bf16 => NmlDType::Bf16,
        DType::F32 => NmlDType::F32,
        DType::F64 => NmlDType::F64,
    }
}
