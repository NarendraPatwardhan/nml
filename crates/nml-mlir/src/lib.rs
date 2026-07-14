//! Safe ownership for NML's pinned MLIR, StableHLO, and Shardy C APIs.
//!
//! The C API represents every object as a copyable pointer-sized value. This
//! crate restores the ownership distinctions that matter to Rust: contexts and
//! modules are unique owners, while locations and types are context-bounded
//! handles. Compiler-only index values are modeled separately from `DType`.

use nml_mlir_sys as sys;
use nml_types::DType;
use std::error::Error as StdError;
use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;

#[derive(Debug, Eq, PartialEq)]
pub enum Error {
    Parse {
        diagnostics: String,
    },
    Verification {
        diagnostics: String,
    },
    Bytecode {
        diagnostics: String,
    },
    PortableArtifact {
        diagnostics: String,
    },
    PassPipeline {
        diagnostics: String,
    },
    PassRun {
        diagnostics: String,
    },
    InvalidType(DType),
    InvalidAttribute {
        source: String,
    },
    NullObject(&'static str),
    OutOfBounds {
        object: &'static str,
        index: usize,
        count: usize,
    },
    ContextMismatch {
        object: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { diagnostics } => write!(formatter, "MLIR parse failed: {diagnostics}"),
            Self::Verification { diagnostics } => {
                write!(formatter, "MLIR verification failed: {diagnostics}")
            }
            Self::Bytecode { diagnostics } => {
                write!(
                    formatter,
                    "MLIR bytecode serialization failed: {diagnostics}"
                )
            }
            Self::PortableArtifact { diagnostics } => write!(
                formatter,
                "StableHLO portable-artifact serialization failed: {diagnostics}"
            ),
            Self::PassPipeline { diagnostics } => {
                write!(formatter, "MLIR pass pipeline is invalid: {diagnostics}")
            }
            Self::PassRun { diagnostics } => {
                write!(formatter, "MLIR pass execution failed: {diagnostics}")
            }
            Self::InvalidType(dtype) => write!(formatter, "MLIR rejected NML dtype {dtype:?}"),
            Self::InvalidAttribute { source } => {
                write!(formatter, "MLIR rejected attribute {source:?}")
            }
            Self::NullObject(object) => write!(formatter, "MLIR returned a null {object}"),
            Self::OutOfBounds {
                object,
                index,
                count,
            } => write!(
                formatter,
                "MLIR {object} index {index} is outside the available count {count}"
            ),
            Self::ContextMismatch { object } => {
                write!(formatter, "MLIR {object} belongs to a different context")
            }
        }
    }
}

impl StdError for Error {}

/// Unique owner of an MLIR context and its registered dialect universe.
pub struct Context {
    raw: sys::MlirContext,
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl Context {
    pub fn new() -> Self {
        // SAFETY: creation returns an independently owned context.
        let raw = unsafe { sys::mlirContextCreate() };
        // Reject unknown operations. A misspelled StableHLO operation must be
        // a construction error, not deferred until XLA compilation.
        unsafe { sys::mlirContextSetAllowUnregisteredDialects(raw, false) };
        for handle in [
            unsafe { sys::mlirGetDialectHandle__arith__() },
            unsafe { sys::mlirGetDialectHandle__func__() },
            unsafe { sys::mlirGetDialectHandle__stablehlo__() },
            unsafe { sys::mlirGetDialectHandle__sdy__() },
        ] {
            // SAFETY: every handle is process-static and the context is live.
            unsafe {
                sys::mlirDialectHandleRegisterDialect(handle, raw);
                sys::mlirDialectHandleLoadDialect(handle, raw);
            }
        }
        Self {
            raw,
            _not_send_or_sync: PhantomData,
        }
    }

    pub fn parse_module<'context>(&'context self, source: &str) -> Result<Module<'context>, Error> {
        let (raw, diagnostics) = self.capture_diagnostics(|| unsafe {
            sys::mlirModuleCreateParse(self.raw, string_ref(source.as_bytes()))
        });
        if raw.ptr.is_null() {
            Err(Error::Parse { diagnostics })
        } else {
            Ok(Module { raw, context: self })
        }
    }

    /// Creates an empty builtin module whose body owns appended operations.
    pub fn empty_module(&self) -> Result<Module<'_>, Error> {
        let raw = unsafe { sys::mlirModuleCreateEmpty(self.location().raw) };
        if raw.ptr.is_null() {
            Err(Error::NullObject("module"))
        } else {
            Ok(Module { raw, context: self })
        }
    }

    pub fn location(&self) -> Location<'_> {
        Location {
            raw: unsafe { sys::mlirLocationUnknownGet(self.raw) },
            _context: PhantomData,
        }
    }

    pub fn index_type(&self) -> IndexType<'_> {
        IndexType(Type {
            raw: unsafe { sys::mlirIndexTypeGet(self.raw) },
            context_id: self.context_id(),
            _context: PhantomData,
        })
    }

    pub fn dtype(&self, dtype: DType) -> Result<Type<'_>, Error> {
        let raw = unsafe {
            match dtype {
                DType::Bool => sys::mlirIntegerTypeGet(self.raw, 1),
                DType::I8 | DType::U8 => sys::mlirIntegerTypeGet(self.raw, 8),
                DType::I16 | DType::U16 => sys::mlirIntegerTypeGet(self.raw, 16),
                DType::I32 | DType::U32 => sys::mlirIntegerTypeGet(self.raw, 32),
                DType::I64 | DType::U64 => sys::mlirIntegerTypeGet(self.raw, 64),
                DType::F16 => sys::mlirF16TypeGet(self.raw),
                DType::Bf16 => sys::mlirBF16TypeGet(self.raw),
                DType::F32 => sys::mlirF32TypeGet(self.raw),
                DType::F64 => sys::mlirF64TypeGet(self.raw),
                DType::C64 => sys::mlirComplexTypeGet(sys::mlirF32TypeGet(self.raw)),
                DType::C128 => sys::mlirComplexTypeGet(sys::mlirF64TypeGet(self.raw)),
            }
        };
        if raw.ptr.is_null() {
            Err(Error::InvalidType(dtype))
        } else {
            Ok(Type {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn ranked_tensor_type<'context>(
        &'context self,
        dtype: DType,
        dimensions: &[i64],
    ) -> Result<Type<'context>, Error> {
        let element = self.dtype(dtype)?;
        let raw = unsafe {
            sys::mlirRankedTensorTypeGet(
                dimensions.len() as isize,
                dimensions.as_ptr(),
                element.raw,
                sys::MlirAttribute {
                    ptr: ptr::null_mut(),
                },
            )
        };
        if raw.ptr.is_null() {
            Err(Error::InvalidType(dtype))
        } else {
            Ok(Type {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn function_type<'context>(
        &'context self,
        inputs: &[Type<'context>],
        results: &[Type<'context>],
    ) -> Result<Type<'context>, Error> {
        self.require_contexts(
            inputs.iter().map(|value| value.context_id),
            "function input type",
        )?;
        self.require_contexts(
            results.iter().map(|value| value.context_id),
            "function result type",
        )?;
        let inputs: Vec<_> = inputs.iter().map(|value| value.raw).collect();
        let results: Vec<_> = results.iter().map(|value| value.raw).collect();
        let raw = unsafe {
            sys::mlirFunctionTypeGet(
                self.raw,
                inputs.len() as isize,
                inputs.as_ptr(),
                results.len() as isize,
                results.as_ptr(),
            )
        };
        if raw.ptr.is_null() {
            Err(Error::NullObject("function type"))
        } else {
            Ok(Type {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn parse_attribute(&self, source: &str) -> Result<Attribute<'_>, Error> {
        let raw = unsafe { sys::mlirAttributeParseGet(self.raw, string_ref(source.as_bytes())) };
        if raw.ptr.is_null() {
            Err(Error::InvalidAttribute {
                source: source.to_owned(),
            })
        } else {
            Ok(Attribute {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn string_attribute(&self, value: &str) -> Attribute<'_> {
        Attribute {
            raw: unsafe { sys::mlirStringAttrGet(self.raw, string_ref(value.as_bytes())) },
            context_id: self.context_id(),
            _context: PhantomData,
        }
    }

    pub fn type_attribute<'context>(
        &'context self,
        value: Type<'context>,
    ) -> Result<Attribute<'context>, Error> {
        self.require_context(value.context_id, "type attribute value")?;
        Ok(Attribute {
            raw: unsafe { sys::mlirTypeAttrGet(value.raw) },
            context_id: self.context_id(),
            _context: PhantomData,
        })
    }

    pub fn integer_attribute<'context>(
        &'context self,
        value_type: Type<'context>,
        value: i64,
    ) -> Result<Attribute<'context>, Error> {
        self.require_context(value_type.context_id, "integer attribute type")?;
        Ok(Attribute {
            raw: unsafe { sys::mlirIntegerAttrGet(value_type.raw, value) },
            context_id: self.context_id(),
            _context: PhantomData,
        })
    }

    pub fn named_attribute<'context>(
        &'context self,
        name: &str,
        value: Attribute<'context>,
    ) -> Result<NamedAttribute<'context>, Error> {
        self.require_context(value.context_id, "named attribute value")?;
        let identifier = unsafe { sys::mlirIdentifierGet(self.raw, string_ref(name.as_bytes())) };
        Ok(NamedAttribute {
            raw: unsafe { sys::mlirNamedAttributeGet(identifier, value.raw) },
            context_id: self.context_id(),
            _context: PhantomData,
        })
    }

    pub fn pass_manager(&self) -> PassManager<'_> {
        PassManager {
            raw: unsafe { sys::mlirPassManagerCreate(self.raw) },
            context: self,
        }
    }

    fn capture_diagnostics<T>(&self, operation: impl FnOnce() -> T) -> (T, String) {
        let mut bytes = Vec::<u8>::new();
        let identifier = unsafe {
            sys::mlirContextAttachDiagnosticHandler(
                self.raw,
                Some(diagnostic_handler),
                (&mut bytes as *mut Vec<u8>).cast(),
                None,
            )
        };
        let detach = DropGuard::new(|| unsafe {
            sys::mlirContextDetachDiagnosticHandler(self.raw, identifier)
        });
        let result = operation();
        drop(detach);
        (result, String::from_utf8_lossy(&bytes).trim().to_owned())
    }

    fn context_id(&self) -> usize {
        self.raw.ptr as usize
    }

    fn require_context(&self, actual: usize, object: &'static str) -> Result<(), Error> {
        if actual == self.context_id() {
            Ok(())
        } else {
            Err(Error::ContextMismatch { object })
        }
    }

    fn require_contexts(
        &self,
        contexts: impl IntoIterator<Item = usize>,
        object: &'static str,
    ) -> Result<(), Error> {
        for context in contexts {
            self.require_context(context, object)?;
        }
        Ok(())
    }
}

struct DropGuard<F: FnOnce()> {
    cleanup: Option<F>,
}

impl<F: FnOnce()> DropGuard<F> {
    fn new(cleanup: F) -> Self {
        Self {
            cleanup: Some(cleanup),
        }
    }
}

impl<F: FnOnce()> Drop for DropGuard<F> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IndexCastKind {
    Signed,
    Unsigned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableHloFftType {
    Fft,
    Ifft,
    Rfft,
    Irfft,
}

impl StableHloFftType {
    const fn spelling(self) -> &'static str {
        match self {
            Self::Fft => "FFT",
            Self::Ifft => "IFFT",
            Self::Rfft => "RFFT",
            Self::Irfft => "IRFFT",
        }
    }
}

impl Context {
    pub fn function<'context>(
        &'context self,
        name: &str,
        inputs: &[Type<'context>],
        results: &[Type<'context>],
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.function_with_input_aliases(name, inputs, results, &vec![None; inputs.len()], body)
    }

    /// Builds a function whose selected arguments may donate storage to an
    /// output. XLA consumes the same `tf.aliasing_output` argument attribute
    /// emitted by ZML; keeping it on the function boundary makes donation an
    /// executable ABI promise rather than an optimizer guess.
    pub fn function_with_input_aliases<'context>(
        &'context self,
        name: &str,
        inputs: &[Type<'context>],
        results: &[Type<'context>],
        input_aliases: &[Option<usize>],
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        if input_aliases.len() != inputs.len() {
            return Err(Error::OutOfBounds {
                object: "function alias input",
                index: input_aliases.len(),
                count: inputs.len(),
            });
        }
        if let Some(output) = input_aliases
            .iter()
            .flatten()
            .find(|output| **output >= results.len())
        {
            return Err(Error::OutOfBounds {
                object: "function alias output",
                index: *output,
                count: results.len(),
            });
        }
        let function_type = self.function_type(inputs, results)?;
        let mut attributes = vec![
            self.named_attribute("sym_name", self.string_attribute(name))?,
            self.named_attribute("function_type", self.type_attribute(function_type)?)?,
        ];
        if input_aliases.iter().any(Option::is_some) {
            let spelling = format!(
                "[{}]",
                input_aliases
                    .iter()
                    .map(|alias| alias.map_or_else(
                        || "{}".to_owned(),
                        |output| format!("{{tf.aliasing_output = {output} : i32}}"),
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            attributes.push(self.named_attribute("arg_attrs", self.parse_attribute(&spelling)?)?);
        }
        Operation::builder(self, "func.func")
            .attributes(&attributes)
            .region(body)
            .build()
    }

    pub fn return_operation<'context>(
        &'context self,
        values: &[Value<'context>],
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "func.return")
            .operands(values)
            .build()
    }

    pub fn constant<'context>(
        &'context self,
        result_type: Type<'context>,
        value: Attribute<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.constant")
            .results(&[result_type])
            .attributes(&[self.named_attribute("value", value)?])
            .build()
    }

    pub fn index_constant(&self, value: i64) -> Result<Operation<'_>, Error> {
        let result_type = self.index_type().0;
        Operation::builder(self, "arith.constant")
            .results(&[result_type])
            .attributes(&[
                self.named_attribute("value", self.integer_attribute(result_type, value)?)?
            ])
            .build()
    }

    pub fn index_cast<'context>(
        &'context self,
        value: Value<'context>,
        result_type: Type<'context>,
        kind: IndexCastKind,
    ) -> Result<Operation<'context>, Error> {
        let name = match kind {
            IndexCastKind::Signed => "arith.index_cast",
            IndexCastKind::Unsigned => "arith.index_castui",
        };
        Operation::builder(self, name)
            .results(&[result_type])
            .operands(&[value])
            .build()
    }

    pub fn dot_general<'context>(
        &'context self,
        left: Value<'context>,
        right: Value<'context>,
        result_type: Type<'context>,
        left_batch: &[i64],
        right_batch: &[i64],
        left_contract: &[i64],
        right_contract: &[i64],
    ) -> Result<Operation<'context>, Error> {
        let dimensions = format!(
            "#stablehlo.dot<lhs_batching_dimensions = {}, rhs_batching_dimensions = {}, \
             lhs_contracting_dimensions = {}, rhs_contracting_dimensions = {}>",
            i64_array(left_batch),
            i64_array(right_batch),
            i64_array(left_contract),
            i64_array(right_contract),
        );
        Operation::builder(self, "stablehlo.dot_general")
            .results(&[result_type])
            .operands(&[left, right])
            .attributes(&[
                self.named_attribute("dot_dimension_numbers", self.parse_attribute(&dimensions)?)?
            ])
            .build()
    }

    pub fn add<'context>(
        &'context self,
        left: Value<'context>,
        right: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.add")
            .results(&[result_type])
            .operands(&[left, right])
            .build()
    }

    pub fn broadcast_in_dim<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
        dimensions: &[i64],
    ) -> Result<Operation<'context>, Error> {
        let dimensions = format!("array<i64: {}>", comma_separated_i64(dimensions));
        Operation::builder(self, "stablehlo.broadcast_in_dim")
            .results(&[result_type])
            .operands(&[input])
            .attributes(&[
                self.named_attribute("broadcast_dimensions", self.parse_attribute(&dimensions)?)?
            ])
            .build()
    }

    pub fn complex<'context>(
        &'context self,
        real: Value<'context>,
        imaginary: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.complex")
            .results(&[result_type])
            .operands(&[real, imaginary])
            .build()
    }

    pub fn real<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.unary("stablehlo.real", input, result_type)
    }

    pub fn imaginary<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.unary("stablehlo.imag", input, result_type)
    }

    pub fn fft<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
        kind: StableHloFftType,
        lengths: &[i64],
    ) -> Result<Operation<'context>, Error> {
        let fft_type = format!("#stablehlo<fft_type {}>", kind.spelling());
        let fft_length = format!("array<i64: {}>", comma_separated_i64(lengths));
        Operation::builder(self, "stablehlo.fft")
            .results(&[result_type])
            .operands(&[input])
            .attributes(&[
                self.named_attribute("fft_type", self.parse_attribute(&fft_type)?)?,
                self.named_attribute("fft_length", self.parse_attribute(&fft_length)?)?,
            ])
            .build()
    }

    fn unary<'context>(
        &'context self,
        name: &str,
        input: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, name)
            .results(&[result_type])
            .operands(&[input])
            .build()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { sys::mlirContextDestroy(self.raw) };
    }
}

/// Unique owner of one top-level builtin module operation.
pub struct Module<'context> {
    raw: sys::MlirModule,
    context: &'context Context,
}

impl<'context> Module<'context> {
    pub fn append_operation(&mut self, operation: Operation<'context>) -> Result<(), Error> {
        self.context
            .require_context(operation.context_id, "module operation")?;
        let body = unsafe { sys::mlirModuleGetBody(self.raw) };
        unsafe { sys::mlirBlockAppendOwnedOperation(body, operation.into_raw()) };
        Ok(())
    }

    pub fn verify(&self) -> Result<(), Error> {
        let operation = unsafe { sys::mlirModuleGetOperation(self.raw) };
        let (valid, diagnostics) = self
            .context
            .capture_diagnostics(|| unsafe { sys::mlirOperationVerify(operation) });
        if valid {
            Ok(())
        } else {
            Err(Error::Verification { diagnostics })
        }
    }

    pub fn text(&self) -> String {
        let mut bytes = Vec::new();
        unsafe {
            sys::mlirOperationPrint(
                sys::mlirModuleGetOperation(self.raw),
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
            )
        };
        String::from_utf8(bytes).expect("MLIR textual form is UTF-8")
    }

    pub fn bytecode(&self) -> Result<Vec<u8>, Error> {
        let mut bytes = Vec::new();
        let configuration = unsafe { sys::mlirBytecodeWriterConfigCreate() };
        let (result, diagnostics) = self.context.capture_diagnostics(|| unsafe {
            sys::mlirOperationWriteBytecodeWithConfig(
                sys::mlirModuleGetOperation(self.raw),
                configuration,
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
            )
        });
        unsafe { sys::mlirBytecodeWriterConfigDestroy(configuration) };
        if unsafe { sys::nml_mlir_logical_result_is_success(result) } {
            Ok(bytes)
        } else {
            Err(Error::Bytecode { diagnostics })
        }
    }

    /// Serializes the module into StableHLO's versioned portable artifact.
    pub fn portable_artifact(&self, target_version: &str) -> Result<Vec<u8>, Error> {
        let mut bytes = Vec::new();
        let (result, diagnostics) = self.context.capture_diagnostics(|| unsafe {
            sys::stablehloSerializePortableArtifactFromModule(
                self.raw,
                string_ref(target_version.as_bytes()),
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
                true,
            )
        });
        if unsafe { sys::nml_mlir_logical_result_is_success(result) } {
            Ok(bytes)
        } else {
            Err(Error::PortableArtifact { diagnostics })
        }
    }
}

pub fn stablehlo_current_version() -> String {
    let mut bytes = Vec::new();
    unsafe {
        sys::stablehloGetCurrentVersion(Some(append_string), (&mut bytes as *mut Vec<u8>).cast())
    };
    String::from_utf8(bytes).expect("StableHLO version is UTF-8")
}

pub fn stablehlo_minimum_version() -> String {
    let mut bytes = Vec::new();
    unsafe {
        sys::stablehloGetMinimumVersion(Some(append_string), (&mut bytes as *mut Vec<u8>).cast())
    };
    String::from_utf8(bytes).expect("StableHLO version is UTF-8")
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlirModuleDestroy(self.raw) };
    }
}

/// Unique owner of an MLIR pass manager tied to its construction context.
pub struct PassManager<'context> {
    raw: sys::MlirPassManager,
    context: &'context Context,
}

impl<'context> PassManager<'context> {
    pub fn parse_pipeline(&mut self, pipeline: &str) -> Result<(), Error> {
        let mut diagnostics = Vec::new();
        let operation_manager = unsafe { sys::mlirPassManagerGetAsOpPassManager(self.raw) };
        let result = unsafe {
            sys::mlirParsePassPipeline(
                operation_manager,
                string_ref(pipeline.as_bytes()),
                Some(append_string),
                (&mut diagnostics as *mut Vec<u8>).cast(),
            )
        };
        if unsafe { sys::nml_mlir_logical_result_is_success(result) } {
            Ok(())
        } else {
            Err(Error::PassPipeline {
                diagnostics: String::from_utf8_lossy(&diagnostics).trim().to_owned(),
            })
        }
    }

    pub fn run(&mut self, module: &mut Module<'context>) -> Result<(), Error> {
        if !std::ptr::eq(self.context, module.context) {
            return Err(Error::ContextMismatch {
                object: "pass-manager module",
            });
        }
        let operation = unsafe { sys::mlirModuleGetOperation(module.raw) };
        let (result, diagnostics) = self
            .context
            .capture_diagnostics(|| unsafe { sys::mlirPassManagerRunOnOp(self.raw, operation) });
        if unsafe { sys::nml_mlir_logical_result_is_success(result) } {
            Ok(())
        } else {
            Err(Error::PassRun { diagnostics })
        }
    }
}

impl Drop for PassManager<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlirPassManagerDestroy(self.raw) };
    }
}

/// An MLIR region owned by Rust until it is transferred into an operation.
pub struct Region<'context> {
    raw: Option<sys::MlirRegion>,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

impl<'context> Region<'context> {
    pub fn new(_context: &'context Context) -> Result<Self, Error> {
        let raw = unsafe { sys::mlirRegionCreate() };
        if raw.ptr.is_null() {
            Err(Error::NullObject("region"))
        } else {
            Ok(Self {
                raw: Some(raw),
                context_id: _context.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn append_block(&mut self, block: Block<'context>) -> Result<(), Error> {
        if block.context_id != self.context_id {
            return Err(Error::ContextMismatch {
                object: "region block",
            });
        }
        unsafe {
            sys::mlirRegionAppendOwnedBlock(
                self.raw.expect("region ownership was transferred"),
                block.into_raw(),
            )
        };
        Ok(())
    }

    fn into_raw(mut self) -> sys::MlirRegion {
        self.raw.take().expect("region ownership was transferred")
    }
}

impl Drop for Region<'_> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { sys::mlirRegionDestroy(raw) };
        }
    }
}

/// An MLIR block owned by Rust until it is transferred into a region.
pub struct Block<'context> {
    raw: Option<sys::MlirBlock>,
    argument_count: usize,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

impl<'context> Block<'context> {
    pub fn new(context: &'context Context, arguments: &[Type<'context>]) -> Result<Self, Error> {
        context.require_contexts(
            arguments.iter().map(|argument| argument.context_id),
            "block argument type",
        )?;
        let argument_types: Vec<_> = arguments.iter().map(|value| value.raw).collect();
        let locations = vec![context.location().raw; arguments.len()];
        let raw = unsafe {
            sys::mlirBlockCreate(
                arguments.len() as isize,
                argument_types.as_ptr(),
                locations.as_ptr(),
            )
        };
        if raw.ptr.is_null() {
            Err(Error::NullObject("block"))
        } else {
            Ok(Self {
                raw: Some(raw),
                argument_count: arguments.len(),
                context_id: context.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn argument(&self, index: usize) -> Result<Value<'context>, Error> {
        if index >= self.argument_count {
            return Err(Error::OutOfBounds {
                object: "block argument",
                index,
                count: self.argument_count,
            });
        }
        Ok(Value {
            raw: unsafe {
                sys::mlirBlockGetArgument(
                    self.raw.expect("block ownership was transferred"),
                    index as isize,
                )
            },
            context_id: self.context_id,
            _context: PhantomData,
        })
    }

    pub fn append_operation(&mut self, operation: Operation<'context>) -> Result<(), Error> {
        if operation.context_id != self.context_id {
            return Err(Error::ContextMismatch {
                object: "block operation",
            });
        }
        unsafe {
            sys::mlirBlockAppendOwnedOperation(
                self.raw.expect("block ownership was transferred"),
                operation.into_raw(),
            )
        };
        Ok(())
    }

    fn into_raw(mut self) -> sys::MlirBlock {
        self.raw.take().expect("block ownership was transferred")
    }
}

impl Drop for Block<'_> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { sys::mlirBlockDestroy(raw) };
        }
    }
}

/// An MLIR operation owned by Rust until it is transferred into a block/module.
pub struct Operation<'context> {
    raw: Option<sys::MlirOperation>,
    result_count: usize,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

impl<'context> Operation<'context> {
    pub fn builder(context: &'context Context, name: &str) -> OperationBuilder<'context> {
        OperationBuilder {
            context,
            name: name.to_owned(),
            location: context.location(),
            results: Vec::new(),
            operands: Vec::new(),
            attributes: Vec::new(),
            regions: Vec::new(),
            infer_results: false,
        }
    }

    pub fn result(&self, index: usize) -> Result<Value<'context>, Error> {
        if index >= self.result_count {
            return Err(Error::OutOfBounds {
                object: "operation result",
                index,
                count: self.result_count,
            });
        }
        Ok(Value {
            raw: unsafe {
                sys::mlirOperationGetResult(
                    self.raw.expect("operation ownership was transferred"),
                    index as isize,
                )
            },
            context_id: self.context_id,
            _context: PhantomData,
        })
    }

    fn into_raw(mut self) -> sys::MlirOperation {
        self.raw
            .take()
            .expect("operation ownership was transferred")
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { sys::mlirOperationDestroy(raw) };
        }
    }
}

pub struct OperationBuilder<'context> {
    context: &'context Context,
    name: String,
    location: Location<'context>,
    results: Vec<Type<'context>>,
    operands: Vec<Value<'context>>,
    attributes: Vec<NamedAttribute<'context>>,
    regions: Vec<Region<'context>>,
    infer_results: bool,
}

impl<'context> OperationBuilder<'context> {
    pub fn results(mut self, results: &[Type<'context>]) -> Self {
        self.results.extend_from_slice(results);
        self
    }

    pub fn operands(mut self, operands: &[Value<'context>]) -> Self {
        self.operands.extend_from_slice(operands);
        self
    }

    pub fn attributes(mut self, attributes: &[NamedAttribute<'context>]) -> Self {
        self.attributes.extend_from_slice(attributes);
        self
    }

    pub fn region(mut self, region: Region<'context>) -> Self {
        self.regions.push(region);
        self
    }

    pub fn infer_results(mut self) -> Self {
        self.infer_results = true;
        self
    }

    pub fn build(self) -> Result<Operation<'context>, Error> {
        let context_id = self.context.context_id();
        for (object, actual) in self
            .results
            .iter()
            .map(|value| ("operation result type", value.context_id))
            .chain(
                self.operands
                    .iter()
                    .map(|value| ("operation operand", value.context_id)),
            )
            .chain(
                self.attributes
                    .iter()
                    .map(|value| ("operation attribute", value.context_id)),
            )
            .chain(
                self.regions
                    .iter()
                    .map(|value| ("operation region", value.context_id)),
            )
        {
            if actual != context_id {
                return Err(Error::ContextMismatch { object });
            }
        }
        let mut state = unsafe {
            sys::mlirOperationStateGet(string_ref(self.name.as_bytes()), self.location.raw)
        };
        let results: Vec<_> = self.results.iter().map(|value| value.raw).collect();
        let operands: Vec<_> = self.operands.iter().map(|value| value.raw).collect();
        let attributes: Vec<_> = self.attributes.iter().map(|value| value.raw).collect();
        unsafe {
            sys::mlirOperationStateAddResults(&mut state, results.len() as isize, results.as_ptr());
            sys::mlirOperationStateAddOperands(
                &mut state,
                operands.len() as isize,
                operands.as_ptr(),
            );
            sys::mlirOperationStateAddAttributes(
                &mut state,
                attributes.len() as isize,
                attributes.as_ptr(),
            );
        }
        let region_count = self.regions.len();
        if region_count != 0 {
            let regions: Vec<_> = self.regions.into_iter().map(Region::into_raw).collect();
            unsafe {
                sys::mlirOperationStateAddOwnedRegions(
                    &mut state,
                    regions.len() as isize,
                    regions.as_ptr(),
                )
            };
        }
        if self.infer_results {
            unsafe { sys::mlirOperationStateEnableResultTypeInference(&mut state) };
        }
        let raw = unsafe { sys::mlirOperationCreate(&mut state) };
        if raw.ptr.is_null() {
            Err(Error::NullObject("operation"))
        } else {
            let result_count = if self.infer_results {
                unsafe { sys::mlirOperationGetNumResults(raw) as usize }
            } else {
                results.len()
            };
            let _ = (self.context, region_count);
            Ok(Operation {
                raw: Some(raw),
                result_count,
                context_id,
                _context: PhantomData,
            })
        }
    }
}

#[derive(Clone, Copy)]
pub struct Value<'context> {
    raw: sys::MlirValue,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

impl Value<'_> {
    pub fn text(self) -> String {
        let mut bytes = Vec::new();
        unsafe {
            sys::mlirValuePrint(
                self.raw,
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
            )
        };
        String::from_utf8(bytes).expect("MLIR value text is UTF-8")
    }
}

#[derive(Clone, Copy)]
pub struct Attribute<'context> {
    raw: sys::MlirAttribute,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

impl Attribute<'_> {
    pub fn text(self) -> String {
        let mut bytes = Vec::new();
        unsafe {
            sys::mlirAttributePrint(
                self.raw,
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
            )
        };
        String::from_utf8(bytes).expect("MLIR attribute text is UTF-8")
    }
}

#[derive(Clone, Copy)]
pub struct NamedAttribute<'context> {
    raw: sys::MlirNamedAttribute,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

#[derive(Clone, Copy)]
pub struct Type<'context> {
    raw: sys::MlirType,
    context_id: usize,
    _context: PhantomData<&'context Context>,
}

impl Type<'_> {
    pub fn text(self) -> String {
        let mut bytes = Vec::new();
        unsafe {
            sys::mlirTypePrint(
                self.raw,
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
            )
        };
        String::from_utf8(bytes).expect("MLIR type text is UTF-8")
    }
}

/// Compiler-only MLIR index type. It intentionally has no conversion to
/// `DType`, host storage, or PJRT buffer element types.
#[derive(Clone, Copy)]
pub struct IndexType<'context>(Type<'context>);

impl IndexType<'_> {
    pub fn text(self) -> String {
        self.0.text()
    }
}

#[derive(Clone, Copy)]
pub struct Location<'context> {
    raw: sys::MlirLocation,
    _context: PhantomData<&'context Context>,
}

impl Location<'_> {
    pub fn text(self) -> String {
        let mut bytes = Vec::new();
        unsafe {
            sys::mlirLocationPrint(
                self.raw,
                Some(append_string),
                (&mut bytes as *mut Vec<u8>).cast(),
            )
        };
        String::from_utf8(bytes).expect("MLIR location text is UTF-8")
    }
}

unsafe extern "C" fn append_string(value: sys::MlirStringRef, user_data: *mut c_void) {
    let bytes = unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), value.length) };
    let output = unsafe { &mut *user_data.cast::<Vec<u8>>() };
    output.extend_from_slice(bytes);
}

unsafe extern "C" fn diagnostic_handler(
    diagnostic: sys::MlirDiagnostic,
    user_data: *mut c_void,
) -> sys::MlirLogicalResult {
    unsafe { sys::mlirDiagnosticPrint(diagnostic, Some(append_string), user_data) };
    let output = unsafe { &mut *user_data.cast::<Vec<u8>>() };
    output.push(b'\n');
    unsafe { sys::nml_mlir_logical_result_success() }
}

fn string_ref(bytes: &[u8]) -> sys::MlirStringRef {
    sys::MlirStringRef {
        data: bytes.as_ptr().cast(),
        length: bytes.len(),
    }
}

fn i64_array(values: &[i64]) -> String {
    format!("[{}]", comma_separated_i64(values))
}

fn comma_separated_i64(values: &[i64]) -> String {
    values
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}
