//! Safe ownership for NML's pinned MLIR, StableHLO, and Shardy C APIs.
//!
//! The C API represents every object as a copyable pointer-sized value. This
//! crate restores the ownership distinctions that matter to Rust: contexts and
//! modules are unique owners, while locations and types are context-bounded
//! handles. Compiler-only index values are modeled separately from `DType`.

use nml_mlir_sys as sys;
use nml_types::DType;
use std::collections::HashSet;
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
    InvalidOperation {
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
            Self::InvalidOperation { source } => {
                write!(formatter, "MLIR rejected operation contract: {source}")
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
        Self::with_dialects(&[
            unsafe { sys::mlirGetDialectHandle__arith__() },
            unsafe { sys::mlirGetDialectHandle__func__() },
            unsafe { sys::mlirGetDialectHandle__stablehlo__() },
            unsafe { sys::mlirGetDialectHandle__sdy__() },
        ])
    }

    /// Creates the isolated compiler context used while authoring one TTIR
    /// kernel.  TTIR never enters the long-lived StableHLO model context: the
    /// verified textual module is the only value crossing that boundary.
    pub fn new_ttir() -> Self {
        Self::with_dialects(&[
            unsafe { sys::mlirGetDialectHandle__arith__() },
            unsafe { sys::mlirGetDialectHandle__cf__() },
            unsafe { sys::mlirGetDialectHandle__func__() },
            unsafe { sys::mlirGetDialectHandle__math__() },
            unsafe { sys::mlirGetDialectHandle__scf__() },
            unsafe { sys::mlirGetDialectHandle__tt__() },
        ])
    }

    fn with_dialects(handles: &[sys::MlirDialectHandle]) -> Self {
        // SAFETY: creation returns an independently owned context.
        let raw = unsafe { sys::mlirContextCreate() };
        // Reject unknown operations. A misspelled StableHLO operation must be
        // a construction error, not deferred until XLA compilation.
        unsafe { sys::mlirContextSetAllowUnregisteredDialects(raw, false) };
        for &handle in handles {
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

    pub fn parse_type(&self, source: &str) -> Result<Type<'_>, Error> {
        let raw = unsafe { sys::mlirTypeParseGet(self.raw, string_ref(source.as_bytes())) };
        if raw.ptr.is_null() {
            Err(Error::InvalidOperation {
                source: format!("MLIR rejected type {source:?}"),
            })
        } else {
            Ok(Type {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn triton_pointer_type<'context>(
        &'context self,
        pointee: Type<'context>,
        address_space: i32,
    ) -> Result<Type<'context>, Error> {
        self.require_context(pointee.context_id, "Triton pointer pointee")?;
        let raw = unsafe { sys::nml_mlir_triton_pointer_type(pointee.raw, address_space) };
        if raw.ptr.is_null() {
            Err(Error::NullObject("Triton pointer type"))
        } else {
            Ok(Type {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn triton_tensor_descriptor_type<'context>(
        &'context self,
        shape: &[i64],
        element: Type<'context>,
    ) -> Result<Type<'context>, Error> {
        self.require_context(element.context_id, "Triton tensor descriptor element")?;
        let raw = unsafe {
            sys::nml_mlir_triton_tensor_descriptor_type(
                shape.len() as isize,
                shape.as_ptr(),
                element.raw,
            )
        };
        if raw.ptr.is_null() {
            Err(Error::NullObject("Triton tensor descriptor type"))
        } else {
            Ok(Type {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

    pub fn triton_program_dimension(&self, axis: u8) -> Result<Attribute<'_>, Error> {
        if axis > 2 {
            return Err(Error::InvalidOperation {
                source: format!("Triton program dimension {axis} is outside x/y/z"),
            });
        }
        Ok(self.triton_attribute(unsafe {
            sys::nml_mlir_triton_program_dimension(self.raw, i32::from(axis))
        }))
    }

    pub fn triton_cache_modifier(&self, value: i32) -> Result<Attribute<'_>, Error> {
        if !(1..=7).contains(&value) {
            return Err(Error::InvalidOperation {
                source: format!("invalid Triton cache modifier {value}"),
            });
        }
        Ok(self.triton_attribute(unsafe { sys::nml_mlir_triton_cache_modifier(self.raw, value) }))
    }

    pub fn triton_eviction_policy(&self, value: i32) -> Result<Attribute<'_>, Error> {
        if !(1..=3).contains(&value) {
            return Err(Error::InvalidOperation {
                source: format!("invalid Triton eviction policy {value}"),
            });
        }
        Ok(self.triton_attribute(unsafe { sys::nml_mlir_triton_eviction_policy(self.raw, value) }))
    }

    pub fn triton_input_precision(&self, value: i32) -> Result<Attribute<'_>, Error> {
        if !(0..=4).contains(&value) {
            return Err(Error::InvalidOperation {
                source: format!("invalid Triton input precision {value}"),
            });
        }
        Ok(self.triton_attribute(unsafe { sys::nml_mlir_triton_input_precision(self.raw, value) }))
    }

    fn triton_attribute(&self, raw: sys::MlirAttribute) -> Attribute<'_> {
        Attribute {
            raw,
            context_id: self.context_id(),
            _context: PhantomData,
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

    pub fn bool_attribute(&self, value: bool) -> Attribute<'_> {
        Attribute {
            raw: unsafe { sys::mlirBoolAttrGet(self.raw, i32::from(value)) },
            context_id: self.context_id(),
            _context: PhantomData,
        }
    }

    pub fn array_attribute<'context>(
        &'context self,
        values: &[Attribute<'context>],
    ) -> Result<Attribute<'context>, Error> {
        self.require_contexts(
            values.iter().map(|value| value.context_id),
            "array attribute value",
        )?;
        let raw_values = values.iter().map(|value| value.raw).collect::<Vec<_>>();
        Ok(Attribute {
            raw: unsafe {
                sys::mlirArrayAttrGet(self.raw, raw_values.len() as isize, raw_values.as_ptr())
            },
            context_id: self.context_id(),
            _context: PhantomData,
        })
    }

    pub fn dictionary_attribute<'context>(
        &'context self,
        values: &[NamedAttribute<'context>],
    ) -> Result<Attribute<'context>, Error> {
        self.require_contexts(
            values.iter().map(|value| value.context_id),
            "dictionary attribute value",
        )?;
        let raw_values = values.iter().map(|value| value.raw).collect::<Vec<_>>();
        Ok(Attribute {
            raw: unsafe {
                sys::mlirDictionaryAttrGet(self.raw, raw_values.len() as isize, raw_values.as_ptr())
            },
            context_id: self.context_id(),
            _context: PhantomData,
        })
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

    pub fn dense_index_attribute<'context>(
        &'context self,
        values: &[i64],
    ) -> Result<Attribute<'context>, Error> {
        let index = self.index_type().0;
        let dimensions = [values.len() as i64];
        let tensor = unsafe {
            sys::mlirRankedTensorTypeGet(
                1,
                dimensions.as_ptr(),
                index.raw,
                sys::MlirAttribute {
                    ptr: ptr::null_mut(),
                },
            )
        };
        if tensor.ptr.is_null() {
            return Err(Error::NullObject("dense index tensor type"));
        }
        let raw = unsafe {
            sys::mlirDenseElementsAttrInt64Get(tensor, values.len() as isize, values.as_ptr())
        };
        if raw.ptr.is_null() {
            Err(Error::NullObject("dense index attribute"))
        } else {
            Ok(Attribute {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShardyDimension<'a> {
    Open,
    Replicated,
    Sharded(&'a str),
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
    pub fn shardy_mesh<'context>(
        &'context self,
        axes: &[(&str, i64)],
        device_ids: &[i64],
    ) -> Result<Attribute<'context>, Error> {
        if axes.is_empty() {
            return Err(invalid_attribute("Shardy mesh has no axes"));
        }
        let mut axis_names = HashSet::new();
        let mut device_count = 1usize;
        for &(name, size) in axes {
            if name.is_empty() {
                return Err(invalid_attribute("Shardy mesh axis name is empty"));
            }
            if size <= 0 {
                return Err(invalid_attribute(format!(
                    "Shardy mesh axis {name:?} has non-positive size {size}"
                )));
            }
            if !axis_names.insert(name) {
                return Err(invalid_attribute(format!(
                    "Shardy mesh axis {name:?} is duplicated"
                )));
            }
            let size = usize::try_from(size)
                .map_err(|_| invalid_attribute("Shardy mesh size exceeds the host index range"))?;
            device_count = device_count
                .checked_mul(size)
                .ok_or_else(|| invalid_attribute("Shardy mesh device count overflows"))?;
        }
        if !device_ids.is_empty() && device_ids.len() != device_count {
            return Err(invalid_attribute(format!(
                "Shardy mesh needs {device_count} device ids, received {}",
                device_ids.len()
            )));
        }
        let mut unique_device_ids = HashSet::new();
        if let Some(id) = device_ids
            .iter()
            .find(|id| **id < 0 || !unique_device_ids.insert(**id))
        {
            return Err(invalid_attribute(format!(
                "Shardy mesh device id {id} is negative or duplicated"
            )));
        }
        let axes = axes
            .iter()
            .map(|(name, size)| unsafe {
                sys::sdyMeshAxisAttrGet(self.raw, string_ref(name.as_bytes()), *size)
            })
            .collect::<Vec<_>>();
        let raw = unsafe {
            sys::sdyMeshAttrGet(
                self.raw,
                axes.len() as isize,
                axes.as_ptr(),
                device_ids.len() as isize,
                device_ids.as_ptr(),
            )
        };
        self.attribute(raw, "Shardy mesh attribute")
    }

    pub fn shardy_tensor_sharding<'context>(
        &'context self,
        mesh: &str,
        dimensions: &[ShardyDimension<'_>],
        replicated_axes: &[&str],
    ) -> Result<Attribute<'context>, Error> {
        let null_attribute = sys::MlirAttribute {
            ptr: ptr::null_mut(),
        };
        let axis_ref = |name: &str| unsafe {
            sys::sdyAxisRefAttrGet(self.raw, string_ref(name.as_bytes()), null_attribute)
        };
        let dimensions = dimensions
            .iter()
            .map(|dimension| {
                let (axes, closed) = match dimension {
                    ShardyDimension::Open => (Vec::new(), false),
                    ShardyDimension::Replicated => (Vec::new(), true),
                    ShardyDimension::Sharded(axis) => (vec![axis_ref(axis)], true),
                };
                unsafe {
                    sys::sdyDimensionShardingAttrGet(
                        self.raw,
                        axes.len() as isize,
                        axes.as_ptr(),
                        closed,
                        -1,
                    )
                }
            })
            .collect::<Vec<_>>();
        let replicated_axes = replicated_axes
            .iter()
            .map(|axis| axis_ref(axis))
            .collect::<Vec<_>>();
        let mesh = unsafe { sys::mlirFlatSymbolRefAttrGet(self.raw, string_ref(mesh.as_bytes())) };
        let raw = unsafe {
            sys::sdyTensorShardingAttrGet(
                self.raw,
                mesh,
                dimensions.len() as isize,
                dimensions.as_ptr(),
                replicated_axes.len() as isize,
                replicated_axes.as_ptr(),
                0,
                ptr::null(),
            )
        };
        self.attribute(raw, "Shardy tensor sharding attribute")
    }

    pub fn shardy_mesh_operation<'context>(
        &'context self,
        name: &str,
        mesh: Attribute<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "sdy.mesh")
            .attributes(&[
                self.named_attribute("sym_name", self.string_attribute(name))?,
                self.named_attribute("mesh", mesh)?,
            ])
            .build()
    }

    pub fn sharding_constraint<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
        sharding: Attribute<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "sdy.sharding_constraint")
            .results(&[result_type])
            .operands(&[input])
            .attributes(&[self.named_attribute("sharding", sharding)?])
            .build()
    }

    pub fn shardy_per_value_sharding<'context>(
        &'context self,
        shardings: &[Attribute<'context>],
    ) -> Result<Attribute<'context>, Error> {
        self.require_contexts(
            shardings.iter().map(|sharding| sharding.context_id),
            "Shardy per-value sharding attribute",
        )?;
        let shardings = shardings
            .iter()
            .map(|sharding| sharding.raw)
            .collect::<Vec<_>>();
        let raw = unsafe {
            sys::sdyTensorShardingPerValueAttrGet(
                self.raw,
                shardings.len() as isize,
                shardings.as_ptr(),
            )
        };
        self.attribute(raw, "Shardy per-value sharding attribute")
    }

    pub fn shardy_manual_axes<'context>(
        &'context self,
        axes: &[&str],
    ) -> Result<Attribute<'context>, Error> {
        let axes = axes
            .iter()
            .map(|axis| self.string_attribute(axis).raw)
            .collect::<Vec<_>>();
        let raw =
            unsafe { sys::sdyManualAxesAttrGet(self.raw, axes.len() as isize, axes.as_ptr()) };
        self.attribute(raw, "Shardy manual axes attribute")
    }

    pub fn shardy_return<'context>(
        &'context self,
        values: &[Value<'context>],
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "sdy.return")
            .operands(values)
            .build()
    }

    /// Owns the manual-computation region until MLIR takes ownership of the
    /// completed operation. The local block types are intentionally supplied
    /// by the caller: they are physical shard types, whereas `results` are the
    /// corresponding global tensor types at the surrounding graph boundary.
    pub fn shardy_manual_computation<'context>(
        &'context self,
        inputs: &[Value<'context>],
        results: &[Type<'context>],
        input_shardings: Attribute<'context>,
        result_shardings: Attribute<'context>,
        manual_axes: Attribute<'context>,
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "sdy.manual_computation")
            .operands(inputs)
            .results(results)
            .attributes(&[
                self.named_attribute("in_shardings", input_shardings)?,
                self.named_attribute("out_shardings", result_shardings)?,
                self.named_attribute("manual_axes", manual_axes)?,
            ])
            .region(body)
            .build()
    }

    fn attribute<'context>(
        &'context self,
        raw: sys::MlirAttribute,
        object: &'static str,
    ) -> Result<Attribute<'context>, Error> {
        if raw.ptr.is_null() {
            Err(Error::NullObject(object))
        } else {
            Ok(Attribute {
                raw,
                context_id: self.context_id(),
                _context: PhantomData,
            })
        }
    }

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
        self.function_with_attributes(
            name,
            inputs,
            results,
            input_aliases,
            &vec![None; inputs.len()],
            &vec![None; results.len()],
            body,
        )
    }

    pub fn function_with_attributes<'context>(
        &'context self,
        name: &str,
        inputs: &[Type<'context>],
        results: &[Type<'context>],
        input_aliases: &[Option<usize>],
        input_shardings: &[Option<Attribute<'context>>],
        result_shardings: &[Option<Attribute<'context>>],
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        if input_aliases.len() != inputs.len() {
            return Err(Error::OutOfBounds {
                object: "function alias input",
                index: input_aliases.len(),
                count: inputs.len(),
            });
        }
        if input_shardings.len() != inputs.len() {
            return Err(Error::OutOfBounds {
                object: "function input sharding",
                index: input_shardings.len(),
                count: inputs.len(),
            });
        }
        if result_shardings.len() != results.len() {
            return Err(Error::OutOfBounds {
                object: "function result sharding",
                index: result_shardings.len(),
                count: results.len(),
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
        if input_aliases.iter().any(Option::is_some) || input_shardings.iter().any(Option::is_some)
        {
            let spelling = format!(
                "[{}]",
                input_aliases
                    .iter()
                    .zip(input_shardings)
                    .map(|(alias, sharding)| function_value_attributes(*alias, *sharding))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            attributes.push(self.named_attribute("arg_attrs", self.parse_attribute(&spelling)?)?);
        }
        if result_shardings.iter().any(Option::is_some) {
            let spelling = format!(
                "[{}]",
                result_shardings
                    .iter()
                    .map(|sharding| function_value_attributes(None, *sharding))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            attributes.push(self.named_attribute("res_attrs", self.parse_attribute(&spelling)?)?);
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

    pub fn binary<'context>(
        &'context self,
        operation: StableHloBinary,
        left: Value<'context>,
        right: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, operation.name())
            .results(&[result_type])
            .operands(&[left, right])
            .build()
    }

    pub fn clamp<'context>(
        &'context self,
        minimum: Value<'context>,
        input: Value<'context>,
        maximum: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.clamp")
            .results(&[result_type])
            .operands(&[minimum, input, maximum])
            .build()
    }

    pub fn compare<'context>(
        &'context self,
        left: Value<'context>,
        right: Value<'context>,
        result_type: Type<'context>,
        direction: StableHloComparison,
        comparison_type: StableHloComparisonType,
    ) -> Result<Operation<'context>, Error> {
        let direction = format!("#stablehlo<comparison_direction {}>", direction.spelling());
        let comparison_type = format!("#stablehlo<comparison_type {}>", comparison_type.spelling());
        Operation::builder(self, "stablehlo.compare")
            .results(&[result_type])
            .operands(&[left, right])
            .attributes(&[
                self.named_attribute("comparison_direction", self.parse_attribute(&direction)?)?,
                self.named_attribute("compare_type", self.parse_attribute(&comparison_type)?)?,
            ])
            .build()
    }

    pub fn select<'context>(
        &'context self,
        predicate: Value<'context>,
        on_true: Value<'context>,
        on_false: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.select")
            .results(&[result_type])
            .operands(&[predicate, on_true, on_false])
            .build()
    }

    pub fn convert<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.unary("stablehlo.convert", input, result_type)
    }

    pub fn reshape<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.unary("stablehlo.reshape", input, result_type)
    }

    pub fn transpose<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
        permutation: &[i64],
    ) -> Result<Operation<'context>, Error> {
        let permutation = format!("array<i64: {}>", comma_separated_i64(permutation));
        Operation::builder(self, "stablehlo.transpose")
            .results(&[result_type])
            .operands(&[input])
            .attributes(
                &[self.named_attribute("permutation", self.parse_attribute(&permutation)?)?],
            )
            .build()
    }

    pub fn unary_math<'context>(
        &'context self,
        operation: StableHloUnary,
        input: Value<'context>,
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.unary(operation.name(), input, result_type)
    }

    pub fn broadcast_in_dim<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
        dimensions: &[i64],
    ) -> Result<Operation<'context>, Error> {
        let dimensions = if dimensions.is_empty() {
            "array<i64>".to_owned()
        } else {
            format!("array<i64: {}>", comma_separated_i64(dimensions))
        };
        Operation::builder(self, "stablehlo.broadcast_in_dim")
            .results(&[result_type])
            .operands(&[input])
            .attributes(&[
                self.named_attribute("broadcast_dimensions", self.parse_attribute(&dimensions)?)?
            ])
            .build()
    }

    pub fn iota<'context>(
        &'context self,
        result_type: Type<'context>,
        dimension: i64,
    ) -> Result<Operation<'context>, Error> {
        let i64_type = self.dtype(DType::I64)?;
        Operation::builder(self, "stablehlo.iota")
            .results(&[result_type])
            .attributes(&[self.named_attribute(
                "iota_dimension",
                self.integer_attribute(i64_type, dimension)?,
            )?])
            .build()
    }

    pub fn concatenate<'context>(
        &'context self,
        inputs: &[Value<'context>],
        result_type: Type<'context>,
        dimension: i64,
    ) -> Result<Operation<'context>, Error> {
        let i64_type = self.dtype(DType::I64)?;
        Operation::builder(self, "stablehlo.concatenate")
            .results(&[result_type])
            .operands(inputs)
            .attributes(&[
                self.named_attribute("dimension", self.integer_attribute(i64_type, dimension)?)?
            ])
            .build()
    }

    pub fn slice<'context>(
        &'context self,
        input: Value<'context>,
        result_type: Type<'context>,
        starts: &[i64],
        limits: &[i64],
        strides: &[i64],
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.slice")
            .results(&[result_type])
            .operands(&[input])
            .attributes(&[
                self.named_attribute(
                    "start_indices",
                    self.parse_attribute(&dense_i64_array(starts))?,
                )?,
                self.named_attribute(
                    "limit_indices",
                    self.parse_attribute(&dense_i64_array(limits))?,
                )?,
                self.named_attribute("strides", self.parse_attribute(&dense_i64_array(strides))?)?,
            ])
            .build()
    }

    pub fn dynamic_slice<'context>(
        &'context self,
        input: Value<'context>,
        starts: &[Value<'context>],
        result_type: Type<'context>,
        sizes: &[i64],
    ) -> Result<Operation<'context>, Error> {
        let mut operands = Vec::with_capacity(starts.len() + 1);
        operands.push(input);
        operands.extend_from_slice(starts);
        Operation::builder(self, "stablehlo.dynamic_slice")
            .results(&[result_type])
            .operands(&operands)
            .attributes(&[self.named_attribute(
                "slice_sizes",
                self.parse_attribute(&dense_i64_array(sizes))?,
            )?])
            .build()
    }

    pub fn dynamic_update_slice<'context>(
        &'context self,
        input: Value<'context>,
        update: Value<'context>,
        starts: &[Value<'context>],
        result_type: Type<'context>,
    ) -> Result<Operation<'context>, Error> {
        let mut operands = Vec::with_capacity(starts.len() + 2);
        operands.extend([input, update]);
        operands.extend_from_slice(starts);
        Operation::builder(self, "stablehlo.dynamic_update_slice")
            .results(&[result_type])
            .operands(&operands)
            .build()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather<'context>(
        &'context self,
        input: Value<'context>,
        indices: Value<'context>,
        result_type: Type<'context>,
        offset_dims: &[i64],
        collapsed_slice_dims: &[i64],
        operand_batching_dims: &[i64],
        start_indices_batching_dims: &[i64],
        start_index_map: &[i64],
        index_vector_dim: i64,
        slice_sizes: &[i64],
        indices_are_sorted: bool,
    ) -> Result<Operation<'context>, Error> {
        let dimensions = format!(
            "#stablehlo.gather<offset_dims = {}, collapsed_slice_dims = {}, \
             operand_batching_dims = {}, start_indices_batching_dims = {}, \
             start_index_map = {}, index_vector_dim = {}>",
            i64_array(offset_dims),
            i64_array(collapsed_slice_dims),
            i64_array(operand_batching_dims),
            i64_array(start_indices_batching_dims),
            i64_array(start_index_map),
            index_vector_dim,
        );
        Operation::builder(self, "stablehlo.gather")
            .results(&[result_type])
            .operands(&[input, indices])
            .attributes(&[
                self.named_attribute("dimension_numbers", self.parse_attribute(&dimensions)?)?,
                self.named_attribute(
                    "slice_sizes",
                    self.parse_attribute(&dense_i64_array(slice_sizes))?,
                )?,
                self.named_attribute(
                    "indices_are_sorted",
                    self.parse_attribute(if indices_are_sorted { "true" } else { "false" })?,
                )?,
            ])
            .build()
    }

    pub fn reduce<'context>(
        &'context self,
        input: Value<'context>,
        init: Value<'context>,
        result_type: Type<'context>,
        dimensions: &[i64],
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        self.reduce_many(&[input], &[init], &[result_type], dimensions, body)
    }

    /// Builds StableHLO's variadic reduction form. Reducer block arguments are
    /// ordered as every left accumulator followed by every right value, which
    /// is the contract used by tuple reductions such as argmax.
    pub fn reduce_many<'context>(
        &'context self,
        inputs: &[Value<'context>],
        inits: &[Value<'context>],
        result_types: &[Type<'context>],
        dimensions: &[i64],
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        if inputs.is_empty() || inputs.len() != inits.len() || inputs.len() != result_types.len() {
            return Err(Error::InvalidOperation {
                source: format!(
                    "stablehlo.reduce requires equal nonzero input, init, and result counts; got {}, {}, and {}",
                    inputs.len(),
                    inits.len(),
                    result_types.len()
                ),
            });
        }
        let mut operands = Vec::with_capacity(inputs.len() + inits.len());
        operands.extend_from_slice(inputs);
        operands.extend_from_slice(inits);
        Operation::builder(self, "stablehlo.reduce")
            .results(result_types)
            .operands(&operands)
            .attributes(&[self.named_attribute(
                "dimensions",
                self.parse_attribute(&dense_i64_array(dimensions))?,
            )?])
            .region(body)
            .build()
    }

    pub fn stablehlo_return<'context>(
        &'context self,
        values: &[Value<'context>],
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.return")
            .operands(values)
            .build()
    }

    pub fn stablehlo_while<'context>(
        &'context self,
        initial: &[Value<'context>],
        result_types: &[Type<'context>],
        condition: Region<'context>,
        body: Region<'context>,
    ) -> Result<Operation<'context>, Error> {
        Operation::builder(self, "stablehlo.while")
            .results(result_types)
            .operands(initial)
            .region(condition)
            .region(body)
            .build()
    }

    pub fn stablehlo_case<'context>(
        &'context self,
        branch_index: Value<'context>,
        result_types: &[Type<'context>],
        branches: Vec<Region<'context>>,
    ) -> Result<Operation<'context>, Error> {
        if branches.is_empty() {
            return Err(Error::InvalidOperation {
                source: "stablehlo.case requires at least one branch".to_owned(),
            });
        }
        let mut builder = Operation::builder(self, "stablehlo.case")
            .results(result_types)
            .operands(&[branch_index]);
        for branch in branches {
            builder = builder.region(branch);
        }
        builder.build()
    }

    /// Builds NML's typed-FFI FlashAttention forward call.  The semantic
    /// decision remains in `nml-ir`; this narrow builder owns only the exact
    /// StableHLO ABI consumed by the process-lifetime handler.
    pub fn flash_attention_2_custom_call<'context>(
        &'context self,
        query: Value<'context>,
        key: Value<'context>,
        value: Value<'context>,
        output_type: Type<'context>,
        lse_type: Type<'context>,
        scale: f32,
        causal: bool,
        sliding_window: i32,
    ) -> Result<Operation<'context>, Error> {
        self.flash_attention_custom_call(
            "nml.flash_attention_2.forward",
            false,
            &[query, key, value],
            output_type,
            lse_type,
            scale,
            causal,
            sliding_window,
        )
    }

    pub fn flash_attention_3_custom_call<'context>(
        &'context self,
        query: Value<'context>,
        key: Value<'context>,
        value: Value<'context>,
        output_type: Type<'context>,
        lse_type: Type<'context>,
        scale: f32,
        causal: bool,
        sliding_window: i32,
    ) -> Result<Operation<'context>, Error> {
        self.flash_attention_custom_call(
            "nml.flash_attention_3.forward",
            true,
            &[query, key, value],
            output_type,
            lse_type,
            scale,
            causal,
            sliding_window,
        )
    }

    pub fn paged_flash_attention_2_custom_call<'context>(
        &'context self,
        query: Value<'context>,
        key_cache: Value<'context>,
        value_cache: Value<'context>,
        page_table: Value<'context>,
        sequence_lengths: Value<'context>,
        output_type: Type<'context>,
        lse_type: Type<'context>,
        scale: f32,
        causal: bool,
        sliding_window: i32,
    ) -> Result<Operation<'context>, Error> {
        self.flash_attention_custom_call(
            "nml.flash_attention_2.paged",
            false,
            &[query, key_cache, value_cache, page_table, sequence_lengths],
            output_type,
            lse_type,
            scale,
            causal,
            sliding_window,
        )
    }

    pub fn paged_flash_attention_3_custom_call<'context>(
        &'context self,
        query: Value<'context>,
        key_cache: Value<'context>,
        value_cache: Value<'context>,
        page_table: Value<'context>,
        sequence_lengths: Value<'context>,
        output_type: Type<'context>,
        lse_type: Type<'context>,
        scale: f32,
        causal: bool,
        sliding_window: i32,
    ) -> Result<Operation<'context>, Error> {
        self.flash_attention_custom_call(
            "nml.flash_attention_3.paged",
            true,
            &[query, key_cache, value_cache, page_table, sequence_lengths],
            output_type,
            lse_type,
            scale,
            causal,
            sliding_window,
        )
    }

    fn flash_attention_custom_call<'context>(
        &'context self,
        target: &'static str,
        scheduler_workspace: bool,
        operands: &[Value<'context>],
        output_type: Type<'context>,
        lse_type: Type<'context>,
        scale: f32,
        causal: bool,
        sliding_window: i32,
    ) -> Result<Operation<'context>, Error> {
        if !scale.is_finite() || scale <= 0.0 || sliding_window == 0 || sliding_window < -1 {
            return Err(Error::InvalidOperation {
                source: "invalid FlashAttention scale or sliding-window attribute".to_owned(),
            });
        }
        let i32_type = self.dtype(DType::I32)?;
        let backend_config = self.dictionary_attribute(&[
            self.named_attribute(
                "scale",
                self.parse_attribute(&format!("{scale:.9e} : f32"))?,
            )?,
            self.named_attribute("causal", self.bool_attribute(causal))?,
            self.named_attribute(
                "sliding_window",
                self.integer_attribute(i32_type, i64::from(sliding_window))?,
            )?,
        ])?;
        let operand_layouts = operands
            .iter()
            .map(|operand| row_major_layout_for_type(operand.type_()))
            .map(|layout| self.dense_index_attribute(&layout?))
            .collect::<Result<Vec<_>, _>>()?;
        let mut result_types = vec![output_type, lse_type];
        if scheduler_workspace {
            result_types.push(self.ranked_tensor_type(DType::I32, &[1])?);
        }
        let result_layouts = result_types
            .iter()
            .map(|type_| row_major_layout_for_type(*type_))
            .map(|layout| self.dense_index_attribute(&layout?))
            .collect::<Result<Vec<_>, _>>()?;

        Operation::builder(self, "stablehlo.custom_call")
            .results(&result_types)
            .operands(operands)
            .attributes(&[
                self.named_attribute("call_target_name", self.string_attribute(target))?,
                self.named_attribute("has_side_effect", self.bool_attribute(false))?,
                self.named_attribute("api_version", self.integer_attribute(i32_type, 4)?)?,
                self.named_attribute("backend_config", backend_config)?,
                self.named_attribute("operand_layouts", self.array_attribute(&operand_layouts)?)?,
                self.named_attribute("result_layouts", self.array_attribute(&result_layouts)?)?,
                self.named_attribute("output_operand_aliases", self.array_attribute(&[])?)?,
            ])
            .build()
    }

    /// Builds XLA's typed Triton custom call from already verified TTIR.
    /// Kernel authoring and verification belong to the isolated TTIR context;
    /// this method only embeds its immutable text in the StableHLO program.
    pub fn triton_custom_call<'context>(
        &'context self,
        operands: &[Value<'context>],
        result_types: &[Type<'context>],
        options: TritonCustomCall<'_>,
    ) -> Result<Operation<'context>, Error> {
        if options.name.is_empty()
            || options.ir.is_empty()
            || options.grid.iter().any(|value| *value <= 0)
            || options.num_stages <= 0
            || options.num_warps <= 0
            || options.operand_layouts.len() != operands.len()
            || options.result_layouts.len() != result_types.len()
        {
            return Err(Error::InvalidOperation {
                source: "invalid Triton custom-call launch or layout contract".to_owned(),
            });
        }
        for (index, alias) in options.output_operand_aliases.iter().enumerate() {
            if alias.output_index >= result_types.len()
                || alias.operand_index >= operands.len()
                || options.output_operand_aliases[..index]
                    .iter()
                    .any(|prior| prior.output_index == alias.output_index)
            {
                return Err(Error::InvalidOperation {
                    source: "invalid or duplicate Triton output/operand alias".to_owned(),
                });
            }
        }

        let i32_type = self.dtype(DType::I32)?;
        let backend_config = self.dictionary_attribute(&[
            self.named_attribute("name", self.string_attribute(options.name))?,
            self.named_attribute("ir", self.string_attribute(options.ir))?,
            self.named_attribute(
                "grid_x",
                self.integer_attribute(i32_type, i64::from(options.grid[0]))?,
            )?,
            self.named_attribute(
                "grid_y",
                self.integer_attribute(i32_type, i64::from(options.grid[1]))?,
            )?,
            self.named_attribute(
                "grid_z",
                self.integer_attribute(i32_type, i64::from(options.grid[2]))?,
            )?,
            self.named_attribute(
                "num_stages",
                self.integer_attribute(i32_type, i64::from(options.num_stages))?,
            )?,
            self.named_attribute(
                "num_warps",
                self.integer_attribute(i32_type, i64::from(options.num_warps))?,
            )?,
        ])?;
        let operand_layouts = options
            .operand_layouts
            .iter()
            .map(|layout| self.dense_index_attribute(layout))
            .collect::<Result<Vec<_>, _>>()?;
        let result_layouts = options
            .result_layouts
            .iter()
            .map(|layout| self.dense_index_attribute(layout))
            .collect::<Result<Vec<_>, _>>()?;
        let aliases = options
            .output_operand_aliases
            .iter()
            .map(|alias| {
                let output_tuple_indices = if result_types.len() == 1 {
                    String::from("[]")
                } else {
                    format!("[{}]", alias.output_index)
                };
                self.parse_attribute(&format!(
                    "#stablehlo.output_operand_alias<output_tuple_indices = \
                     {output_tuple_indices}, operand_index = {}, \
                     operand_tuple_indices = []>",
                    alias.operand_index
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Operation::builder(self, "stablehlo.custom_call")
            .results(result_types)
            .operands(operands)
            .attributes(&[
                self.named_attribute(
                    "call_target_name",
                    self.string_attribute("__gpu$xla.gpu.triton"),
                )?,
                self.named_attribute("has_side_effect", self.bool_attribute(false))?,
                self.named_attribute("api_version", self.integer_attribute(i32_type, 4)?)?,
                self.named_attribute("backend_config", backend_config)?,
                self.named_attribute("operand_layouts", self.array_attribute(&operand_layouts)?)?,
                self.named_attribute("result_layouts", self.array_attribute(&result_layouts)?)?,
                self.named_attribute("output_operand_aliases", self.array_attribute(&aliases)?)?,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputOperandAlias {
    pub output_index: usize,
    pub operand_index: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TritonCustomCall<'a> {
    pub name: &'a str,
    pub ir: &'a str,
    pub grid: [i32; 3],
    pub num_stages: i32,
    pub num_warps: i32,
    pub operand_layouts: &'a [&'a [i64]],
    pub result_layouts: &'a [&'a [i64]],
    pub output_operand_aliases: &'a [OutputOperandAlias],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableHloBinary {
    Subtract,
    Multiply,
    Divide,
    Minimum,
    Maximum,
    Power,
    Remainder,
    And,
    Or,
}

impl StableHloBinary {
    const fn name(self) -> &'static str {
        match self {
            Self::Subtract => "stablehlo.subtract",
            Self::Multiply => "stablehlo.multiply",
            Self::Divide => "stablehlo.divide",
            Self::Minimum => "stablehlo.minimum",
            Self::Maximum => "stablehlo.maximum",
            Self::Power => "stablehlo.power",
            Self::Remainder => "stablehlo.remainder",
            Self::And => "stablehlo.and",
            Self::Or => "stablehlo.or",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableHloUnary {
    Negate,
    Abs,
    Exponential,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Sine,
    Cosine,
    Logistic,
    Floor,
    Ceil,
}

impl StableHloUnary {
    const fn name(self) -> &'static str {
        match self {
            Self::Negate => "stablehlo.negate",
            Self::Abs => "stablehlo.abs",
            Self::Exponential => "stablehlo.exponential",
            Self::Log => "stablehlo.log",
            Self::Sqrt => "stablehlo.sqrt",
            Self::Rsqrt => "stablehlo.rsqrt",
            Self::Tanh => "stablehlo.tanh",
            Self::Sine => "stablehlo.sine",
            Self::Cosine => "stablehlo.cosine",
            Self::Logistic => "stablehlo.logistic",
            Self::Floor => "stablehlo.floor",
            Self::Ceil => "stablehlo.ceil",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableHloComparison {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}

impl StableHloComparison {
    const fn spelling(self) -> &'static str {
        match self {
            Self::Eq => "EQ",
            Self::Ne => "NE",
            Self::Ge => "GE",
            Self::Gt => "GT",
            Self::Le => "LE",
            Self::Lt => "LT",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableHloComparisonType {
    Float,
    Signed,
    Unsigned,
}

impl StableHloComparisonType {
    const fn spelling(self) -> &'static str {
        match self {
            Self::Float => "FLOAT",
            Self::Signed => "SIGNED",
            Self::Unsigned => "UNSIGNED",
        }
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

impl<'context> Value<'context> {
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

    pub fn type_(self) -> Type<'context> {
        Type {
            raw: unsafe { sys::mlirValueGetType(self.raw) },
            context_id: self.context_id,
            _context: PhantomData,
        }
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

    pub fn is_triton_pointer(self) -> bool {
        unsafe { sys::nml_mlir_type_is_triton_pointer(self.raw) }
    }

    pub fn is_triton_tensor_descriptor(self) -> bool {
        unsafe { sys::nml_mlir_type_is_triton_tensor_descriptor(self.raw) }
    }
}

fn row_major_layout_for_type(type_: Type<'_>) -> Result<Vec<i64>, Error> {
    // `mlirShapedTypeGetRank` requires a ranked shaped type. Keep that C API
    // precondition inside this safe wrapper rather than imposing it on callers.
    if !unsafe { sys::mlirTypeIsARankedTensor(type_.raw) } {
        return Err(Error::InvalidOperation {
            source: "custom-call layouts require ranked tensor types".to_owned(),
        });
    }
    let rank = unsafe { sys::mlirShapedTypeGetRank(type_.raw) } as i64;
    Ok((0..rank).rev().collect())
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

fn dense_i64_array(values: &[i64]) -> String {
    if values.is_empty() {
        "array<i64>".to_owned()
    } else {
        format!("array<i64: {}>", comma_separated_i64(values))
    }
}

fn comma_separated_i64(values: &[i64]) -> String {
    values
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn function_value_attributes(alias: Option<usize>, sharding: Option<Attribute<'_>>) -> String {
    let mut attributes = Vec::new();
    if let Some(output) = alias {
        attributes.push(format!("tf.aliasing_output = {output} : i32"));
    }
    if let Some(sharding) = sharding {
        attributes.push(format!("sdy.sharding = {}", sharding.text()));
    }
    format!("{{{}}}", attributes.join(", "))
}

fn invalid_attribute(source: impl Into<String>) -> Error {
    Error::InvalidAttribute {
        source: source.into(),
    }
}
