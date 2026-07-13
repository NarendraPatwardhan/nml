//! Ownership and error handling for the common PJRT plugin ABI.
//!
//! ZML has one broad PJRT wrapper shared by its platform implementations. NML
//! keeps that boundary: this crate knows how to load and call a PJRT C API and
//! its typed extension chain, but it does not know where CPU or CUDA plugins
//! are packaged. Platform crates own runtime selection and policy.

use nml_pjrt_sys as sys;
use nml_types::{DType, Layout, Shape};
use std::collections::HashSet;
use std::error::Error as StdError;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{offset_of, size_of, zeroed};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::{NonNull, addr_of};
use std::sync::Arc;

type PluginInitializeFn =
    unsafe extern "C" fn(*mut sys::PJRT_Plugin_Initialize_Args) -> *mut sys::PJRT_Error;
type PluginAttributesFn =
    unsafe extern "C" fn(*mut sys::PJRT_Plugin_Attributes_Args) -> *mut sys::PJRT_Error;
type ErrorDestroyFn = unsafe extern "C" fn(*mut sys::PJRT_Error_Destroy_Args);
type ErrorMessageFn = unsafe extern "C" fn(*mut sys::PJRT_Error_Message_Args);
type ErrorGetCodeFn =
    unsafe extern "C" fn(*mut sys::PJRT_Error_GetCode_Args) -> *mut sys::PJRT_Error;
type ClientCreateFn =
    unsafe extern "C" fn(*mut sys::PJRT_Client_Create_Args) -> *mut sys::PJRT_Error;
type ClientDestroyFn =
    unsafe extern "C" fn(*mut sys::PJRT_Client_Destroy_Args) -> *mut sys::PJRT_Error;
type ClientPlatformNameFn =
    unsafe extern "C" fn(*mut sys::PJRT_Client_PlatformName_Args) -> *mut sys::PJRT_Error;
type ClientDevicesFn =
    unsafe extern "C" fn(*mut sys::PJRT_Client_Devices_Args) -> *mut sys::PJRT_Error;
type DeviceGetDescriptionFn =
    unsafe extern "C" fn(*mut sys::PJRT_Device_GetDescription_Args) -> *mut sys::PJRT_Error;
type DeviceDescriptionAttributesFn =
    unsafe extern "C" fn(*mut sys::PJRT_DeviceDescription_Attributes_Args) -> *mut sys::PJRT_Error;
type DeviceDescriptionIdFn =
    unsafe extern "C" fn(*mut sys::PJRT_DeviceDescription_Id_Args) -> *mut sys::PJRT_Error;
type EventDestroyFn =
    unsafe extern "C" fn(*mut sys::PJRT_Event_Destroy_Args) -> *mut sys::PJRT_Error;
type EventIsReadyFn =
    unsafe extern "C" fn(*mut sys::PJRT_Event_IsReady_Args) -> *mut sys::PJRT_Error;
type EventAwaitFn = unsafe extern "C" fn(*mut sys::PJRT_Event_Await_Args) -> *mut sys::PJRT_Error;
type ClientCompileFn =
    unsafe extern "C" fn(*mut sys::PJRT_Client_Compile_Args) -> *mut sys::PJRT_Error;
type ClientBufferFromHostBufferFn =
    unsafe extern "C" fn(*mut sys::PJRT_Client_BufferFromHostBuffer_Args) -> *mut sys::PJRT_Error;
type BufferDestroyFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_Destroy_Args) -> *mut sys::PJRT_Error;
type BufferElementTypeFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_ElementType_Args) -> *mut sys::PJRT_Error;
type BufferDimensionsFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_Dimensions_Args) -> *mut sys::PJRT_Error;
type BufferToHostBufferFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_ToHostBuffer_Args) -> *mut sys::PJRT_Error;
type BufferReadyEventFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_ReadyEvent_Args) -> *mut sys::PJRT_Error;
type BufferDeleteFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_Delete_Args) -> *mut sys::PJRT_Error;
type BufferIsDeletedFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_IsDeleted_Args) -> *mut sys::PJRT_Error;
type BufferMemoryFn =
    unsafe extern "C" fn(*mut sys::PJRT_Buffer_Memory_Args) -> *mut sys::PJRT_Error;
type LoadedExecutableDestroyFn =
    unsafe extern "C" fn(*mut sys::PJRT_LoadedExecutable_Destroy_Args) -> *mut sys::PJRT_Error;
type LoadedExecutableGetExecutableFn = unsafe extern "C" fn(
    *mut sys::PJRT_LoadedExecutable_GetExecutable_Args,
) -> *mut sys::PJRT_Error;
type LoadedExecutableExecuteFn =
    unsafe extern "C" fn(*mut sys::PJRT_LoadedExecutable_Execute_Args) -> *mut sys::PJRT_Error;
type LoadedExecutableAddressableDevicesFn = unsafe extern "C" fn(
    *mut sys::PJRT_LoadedExecutable_AddressableDevices_Args,
) -> *mut sys::PJRT_Error;
type LoadedExecutableDeleteFn =
    unsafe extern "C" fn(*mut sys::PJRT_LoadedExecutable_Delete_Args) -> *mut sys::PJRT_Error;
type LoadedExecutableIsDeletedFn =
    unsafe extern "C" fn(*mut sys::PJRT_LoadedExecutable_IsDeleted_Args) -> *mut sys::PJRT_Error;
type ExecutableDestroyFn =
    unsafe extern "C" fn(*mut sys::PJRT_Executable_Destroy_Args) -> *mut sys::PJRT_Error;
type ExecutableNameFn =
    unsafe extern "C" fn(*mut sys::PJRT_Executable_Name_Args) -> *mut sys::PJRT_Error;
type ExecutableNumOutputsFn =
    unsafe extern "C" fn(*mut sys::PJRT_Executable_NumOutputs_Args) -> *mut sys::PJRT_Error;

/// Failures at the dynamic-library or PJRT ABI boundary.
#[derive(Debug)]
pub enum Error {
    InvalidLibraryPath(PathBuf),
    DynamicLibrary {
        path: PathBuf,
        message: String,
    },
    MissingSymbol {
        symbol: &'static str,
        message: String,
    },
    NullApi,
    TruncatedApi {
        required: usize,
        actual: usize,
    },
    IncompatibleApiMajor {
        expected: i32,
        actual: i32,
    },
    MissingFunction(&'static str),
    NullResult(&'static str),
    Pjrt {
        code: ErrorCode,
        message: String,
    },
    TruncatedExtension {
        extension: &'static str,
        required: usize,
        actual: usize,
    },
    CyclicExtensionChain,
    InvalidHostBuffer {
        expected: usize,
        actual: usize,
    },
    UnsupportedLayout {
        actual: Layout,
        expected: Layout,
    },
    ForeignClientObject(&'static str),
    UnknownBufferType(u32),
    NoAddressableDevice,
    TensorMetadataOverflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLibraryPath(path) => {
                write!(
                    f,
                    "PJRT library path contains a NUL byte: {}",
                    path.display()
                )
            }
            Self::DynamicLibrary { path, message } => {
                write!(
                    f,
                    "failed to load PJRT library {}: {message}",
                    path.display()
                )
            }
            Self::MissingSymbol { symbol, message } => {
                write!(f, "PJRT plugin does not export {symbol}: {message}")
            }
            Self::NullApi => f.write_str("PJRT plugin returned a null API pointer"),
            Self::TruncatedApi { required, actual } => write!(
                f,
                "PJRT API table is too small for NML's required common prefix: \
                 required {required} bytes, plugin provides {actual}"
            ),
            Self::IncompatibleApiMajor { expected, actual } => write!(
                f,
                "incompatible PJRT API major version: NML expects {expected}, plugin provides {actual}"
            ),
            Self::MissingFunction(name) => {
                write!(
                    f,
                    "PJRT API table does not provide required function {name}"
                )
            }
            Self::NullResult(name) => write!(f, "PJRT function {name} returned a null result"),
            Self::Pjrt { code, message } => write!(f, "PJRT error {code:?}: {message}"),
            Self::TruncatedExtension {
                extension,
                required,
                actual,
            } => write!(
                f,
                "PJRT {extension} extension is truncated: required {required} bytes, plugin provides {actual}"
            ),
            Self::CyclicExtensionChain => {
                f.write_str("PJRT plugin returned a cyclic extension chain")
            }
            Self::InvalidHostBuffer { expected, actual } => write!(
                f,
                "host buffer has {actual} bytes, tensor metadata requires {expected}"
            ),
            Self::UnsupportedLayout { actual, expected } => write!(
                f,
                "PJRT transfer does not support physical layout {:?}; expected row-major {:?}",
                actual.minor_to_major(),
                expected.minor_to_major()
            ),
            Self::ForeignClientObject(object) => {
                write!(f, "{object} belongs to a different PJRT client")
            }
            Self::UnknownBufferType(value) => {
                write!(f, "PJRT returned unsupported buffer element type {value}")
            }
            Self::NoAddressableDevice => f.write_str("PJRT executable has no addressable device"),
            Self::TensorMetadataOverflow => {
                f.write_str("tensor metadata exceeds host address space")
            }
        }
    }
}

impl StdError for Error {}

/// Stable PJRT status categories, preserving unknown future integer values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    Ok,
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    Unrecognized(i32),
}

impl From<i32> for ErrorCode {
    fn from(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::Cancelled,
            2 => Self::Unknown,
            3 => Self::InvalidArgument,
            4 => Self::DeadlineExceeded,
            5 => Self::NotFound,
            6 => Self::AlreadyExists,
            7 => Self::PermissionDenied,
            8 => Self::ResourceExhausted,
            9 => Self::FailedPrecondition,
            10 => Self::Aborted,
            11 => Self::OutOfRange,
            12 => Self::Unimplemented,
            13 => Self::Internal,
            14 => Self::Unavailable,
            15 => Self::DataLoss,
            16 => Self::Unauthenticated,
            other => Self::Unrecognized(other),
        }
    }
}

/// Version advertised by the loaded plugin's common PJRT API table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ApiVersion {
    pub major: i32,
    pub minor: i32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StableHloVersion {
    pub major: i64,
    pub minor: i64,
    pub patch: i64,
}

/// PJRT's two GPU custom-call ABIs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum GpuCustomCallApi {
    Untyped = 0,
    Typed = 1,
}

/// An opaque XLA GPU custom-call handler entry point.
///
/// PJRT deliberately types these as `void*`: the selected custom-call API and
/// XLA define the function signature, not the extension header. Keeping the
/// address opaque prevents this ABI boundary from inventing a false Rust
/// function type.
#[derive(Clone, Copy, Debug)]
pub struct GpuCustomCallHandler(NonNull<c_void>);

impl GpuCustomCallHandler {
    /// Wraps a handler address whose signature matches the selected GPU API.
    ///
    /// # Safety
    ///
    /// XLA may retain and invoke this address after registration. It must name
    /// a function with static lifetime and the exact ABI/signature required by
    /// `GpuCustomCallApi` for the handler stage in which it is installed.
    pub unsafe fn from_address(address: NonNull<c_void>) -> Self {
        Self(address)
    }

    fn as_ptr(self) -> *mut c_void {
        self.0.as_ptr()
    }
}

/// The lifecycle handlers registered for one GPU custom-call target.
#[derive(Clone, Copy, Debug)]
pub struct GpuCustomCallHandlers {
    pub instantiate: Option<GpuCustomCallHandler>,
    pub prepare: Option<GpuCustomCallHandler>,
    pub initialize: Option<GpuCustomCallHandler>,
    pub execute: GpuCustomCallHandler,
}

/// A validated GPU custom-call extension borrowed from a loaded plugin.
pub struct GpuCustomCalls {
    plugin: Plugin,
    register_fn: GpuRegisterCustomCallFn,
}

type GpuRegisterCustomCallFn =
    unsafe extern "C" fn(*mut sys::PJRT_Gpu_Register_Custom_Call_Args) -> *mut sys::PJRT_Error;

impl GpuCustomCalls {
    /// Registers a process-lifetime custom-call target with XLA's GPU backend.
    ///
    /// # Safety
    ///
    /// Every handler must satisfy the contract documented by
    /// `GpuCustomCallHandler::from_address`. Handler code and any state it
    /// references must remain valid for the lifetime of the PJRT plugin.
    pub unsafe fn register(
        &self,
        function_name: &str,
        api: GpuCustomCallApi,
        handlers: GpuCustomCallHandlers,
    ) -> Result<(), Error> {
        // SAFETY: zero initializes absent optional handler stages to null.
        let mut args: sys::PJRT_Gpu_Register_Custom_Call_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE as usize;
        args.function_name = function_name.as_ptr().cast();
        args.function_name_size = function_name.len();
        args.api_version = api as i32;
        args.handler_instantiate = optional_handler(handlers.instantiate);
        args.handler_prepare = optional_handler(handlers.prepare);
        args.handler_initialize = optional_handler(handlers.initialize);
        args.handler_execute = handlers.execute.as_ptr();
        // SAFETY: the extension table and function pointer were validated when
        // this handle was created; the caller upholds every handler contract.
        let error = unsafe { (self.register_fn)(&mut args) };
        self.plugin.into_result(error)
    }
}

fn optional_handler(handler: Option<GpuCustomCallHandler>) -> *mut c_void {
    handler.map_or(std::ptr::null_mut(), GpuCustomCallHandler::as_ptr)
}

/// A borrowed PJRT client-creation option.
///
/// Strings are length-delimited by PJRT and therefore need no allocation or
/// NUL terminator. The raw vector exists only for the duration of Client_Create.
#[derive(Clone, Copy, Debug)]
pub enum NamedValue<'a> {
    String { name: &'a str, value: &'a str },
    Int64 { name: &'a str, value: i64 },
    Float { name: &'a str, value: f32 },
    Bool { name: &'a str, value: bool },
}

/// A loaded and initialized PJRT plugin.
///
/// The dynamic-library handle and API pointer are one reference-counted
/// ownership unit. Every dependent object retains this state directly, so
/// dropping the loader handle cannot unload code behind a live PJRT object.
#[derive(Clone)]
pub struct Plugin {
    library: Arc<DynamicLibrary>,
    api: NonNull<sys::PJRT_Api>,
    api_size: usize,
    version: ApiVersion,
}

impl Plugin {
    /// Loads a library which is trusted to implement the PJRT C ABI.
    ///
    /// # Safety
    ///
    /// Loading a shared object executes its initializers. The caller must also
    /// ensure that an exported `GetPjrtApi` symbol really follows the pinned
    /// PJRT ABI. Platform loaders make this operation safe by resolving only
    /// hermetically pinned plugin artifacts from Bazel runfiles.
    pub unsafe fn load_trusted(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        // SAFETY: the caller accepts shared-object initialization and ABI trust
        // as documented above; DynamicLibrary retains the handle on success.
        let library = unsafe { DynamicLibrary::open(path)? };
        // SAFETY: GetPjrtApi is the required PJRT plugin entry point. The same
        // caller trust covers interpreting this address as its specified type.
        let get_api: unsafe extern "C" fn() -> *const sys::PJRT_Api =
            unsafe { library.symbol("GetPjrtApi")? };
        // SAFETY: calling the trusted entry point is part of the contract above.
        let api = NonNull::new(unsafe { get_api() }.cast_mut()).ok_or(Error::NullApi)?;

        // Read only the leading size word until the plugin proves that the
        // common prefix used below exists. This preserves PJRT's append-only
        // API-table compatibility model instead of requiring an exact minor.
        // SAFETY: every PJRT_Api starts with struct_size.
        let actual_size = unsafe { addr_of!((*api.as_ptr()).struct_size).read() };
        let required_size = offset_of!(sys::PJRT_Api, PJRT_Client_Devices) + size_of::<usize>();
        if actual_size < required_size {
            return Err(Error::TruncatedApi {
                required: required_size,
                actual: actual_size,
            });
        }

        // SAFETY: pjrt_api_version lies inside the checked common prefix.
        let raw_version = unsafe { addr_of!((*api.as_ptr()).pjrt_api_version).read() };
        let version = ApiVersion {
            major: raw_version.major_version,
            minor: raw_version.minor_version,
        };
        let expected_major = sys::PJRT_API_MAJOR as i32;
        if version.major != expected_major {
            return Err(Error::IncompatibleApiMajor {
                expected: expected_major,
                actual: version.major,
            });
        }

        let plugin = Self {
            library: Arc::new(library),
            api,
            api_size: actual_size,
            version,
        };
        plugin.validate_required_functions()?;
        plugin.initialize()?;
        Ok(plugin)
    }

    pub fn version(&self) -> ApiVersion {
        self.version
    }

    /// Maximum StableHLO portable-artifact version accepted by this plugin.
    pub fn stablehlo_version(&self) -> Result<Option<StableHloVersion>, Error> {
        let mut args: sys::PJRT_Plugin_Attributes_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Plugin_Attributes_Args_STRUCT_SIZE as usize;
        let error = unsafe { (self.plugin_attributes_fn()?)(&mut args) };
        self.into_result(error)?;
        if args.num_attributes != 0 && args.attributes.is_null() {
            return Err(Error::NullResult("PJRT_Plugin_Attributes"));
        }
        if args.num_attributes == 0 {
            return Ok(None);
        }
        let attributes =
            unsafe { std::slice::from_raw_parts(args.attributes, args.num_attributes) };
        for attribute in attributes {
            if copy_bytes(attribute.name, attribute.name_size)? != b"stablehlo_current_version" {
                continue;
            }
            if attribute.type_ != sys::PJRT_NamedValue_Type_PJRT_NamedValue_kInt64List
                || attribute.value_size != 3
            {
                return Ok(None);
            }
            let values = unsafe { attribute.__bindgen_anon_1.int64_array_value };
            if values.is_null() {
                return Err(Error::NullResult("stablehlo_current_version"));
            }
            let values = unsafe { std::slice::from_raw_parts(values, 3) };
            return Ok(Some(StableHloVersion {
                major: values[0],
                minor: values[1],
                patch: values[2],
            }));
        }
        Ok(None)
    }

    /// Finds and validates PJRT's GPU custom-call registration extension.
    ///
    /// Absence is not an ABI error: CPU plugins and older GPU plugins may not
    /// publish it. A present but truncated or cyclic extension list is rejected
    /// because invoking it would be undefined behavior.
    pub fn gpu_custom_calls(&self) -> Result<Option<GpuCustomCalls>, Error> {
        // SAFETY: extension_start is in the size-checked PJRT_Api prefix.
        let mut current = unsafe { addr_of!((*self.api.as_ptr()).extension_start).read() };
        let mut visited = HashSet::new();
        while let Some(extension) = NonNull::new(current) {
            if !visited.insert(extension.as_ptr() as usize) {
                return Err(Error::CyclicExtensionChain);
            }
            // SAFETY: a PJRT extension chain consists of PJRT_Extension_Base
            // nodes owned by the loaded plugin.
            let base = unsafe { extension.as_ref() };
            let base_required = sys::PJRT_Extension_Base_STRUCT_SIZE as usize;
            if base.struct_size < base_required {
                return Err(Error::TruncatedExtension {
                    extension: "base",
                    required: base_required,
                    actual: base.struct_size,
                });
            }
            if base.type_ == sys::PJRT_Extension_Type_PJRT_Extension_Type_Gpu_Custom_Call {
                let required = sys::PJRT_Gpu_Custom_Call_STRUCT_SIZE as usize;
                if base.struct_size < required {
                    return Err(Error::TruncatedExtension {
                        extension: "GPU custom-call",
                        required,
                        actual: base.struct_size,
                    });
                }
                // SAFETY: the type tag and complete struct size identify the
                // containing PJRT_Gpu_Custom_Call object.
                let gpu = unsafe { &*extension.as_ptr().cast::<sys::PJRT_Gpu_Custom_Call>() };
                let register_fn = gpu
                    .custom_call
                    .ok_or(Error::MissingFunction("PJRT_Gpu_Register_Custom_Call"))?;
                return Ok(Some(GpuCustomCalls {
                    plugin: self.clone(),
                    register_fn,
                }));
            }
            current = base.next;
        }
        Ok(None)
    }

    pub fn create_client(&self) -> Result<Client, Error> {
        self.create_client_with_options(&[])
    }

    pub fn create_client_with_options(&self, options: &[NamedValue<'_>]) -> Result<Client, Error> {
        let raw_options: Vec<_> = options.iter().map(named_value_to_raw).collect();
        // SAFETY: zero is the PJRT-defined absence value for every optional
        // callback/output field; struct_size opts into the complete pinned args.
        let mut args: sys::PJRT_Client_Create_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_Create_Args_STRUCT_SIZE as usize;
        args.create_options = if raw_options.is_empty() {
            std::ptr::null()
        } else {
            raw_options.as_ptr()
        };
        args.num_options = raw_options.len();
        // SAFETY: the API prefix and function pointer were validated at load;
        // args remains live and writable for the duration of the call.
        let error = unsafe { (self.client_create_fn()?)(&mut args) };
        self.into_result(error)?;
        let inner = NonNull::new(args.client).ok_or(Error::NullResult("PJRT_Client_Create"))?;
        Ok(Client {
            state: Arc::new(ClientState {
                plugin: self.clone(),
                inner,
                _not_send_or_sync: PhantomData,
            }),
        })
    }

    fn initialize(&self) -> Result<(), Error> {
        // SAFETY: see create_client; this argument has no optional payload.
        let mut args: sys::PJRT_Plugin_Initialize_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Plugin_Initialize_Args_STRUCT_SIZE as usize;
        // SAFETY: validated function pointer and live argument.
        let error = unsafe { (self.plugin_initialize_fn()?)(&mut args) };
        self.into_result(error)
    }

    fn validate_required_functions(&self) -> Result<(), Error> {
        self.error_destroy_fn()?;
        self.error_message_fn()?;
        self.error_get_code_fn()?;
        self.plugin_initialize_fn()?;
        self.plugin_attributes_fn()?;
        self.client_create_fn()?;
        self.client_destroy_fn()?;
        self.client_platform_name_fn()?;
        self.client_devices_fn()?;
        Ok(())
    }

    fn into_result(&self, error: *mut sys::PJRT_Error) -> Result<(), Error> {
        let Some(error) = NonNull::new(error) else {
            return Ok(());
        };

        // The message is borrowed from the error, so copy it before destroy.
        // SAFETY: the plugin produced this live PJRT_Error and the function
        // pointer is part of the checked API prefix.
        let mut message_args: sys::PJRT_Error_Message_Args = unsafe { zeroed() };
        message_args.struct_size = sys::PJRT_Error_Message_Args_STRUCT_SIZE as usize;
        message_args.error = error.as_ptr();
        // SAFETY: validated function pointer and initialized args.
        unsafe { (self.error_message_fn()?)(&mut message_args) };
        let message = if message_args.message.is_null() {
            String::new()
        } else {
            // SAFETY: PJRT guarantees message_size readable bytes for the
            // lifetime of error, which is still live here.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    message_args.message.cast::<u8>(),
                    message_args.message_size,
                )
            };
            String::from_utf8_lossy(bytes).into_owned()
        };

        // SAFETY: zero initializes code to PJRT_Error_Code_OK and all pointer
        // fields to absent; PJRT fills code on success.
        let mut code_args: sys::PJRT_Error_GetCode_Args = unsafe { zeroed() };
        code_args.struct_size = sys::PJRT_Error_GetCode_Args_STRUCT_SIZE as usize;
        code_args.error = error.as_ptr();
        // SAFETY: validated function pointer and initialized args.
        let code_error = unsafe { (self.error_get_code_fn()?)(&mut code_args) };
        let code = ErrorCode::from(code_args.code as i32);
        if let Some(code_error) = NonNull::new(code_error) {
            self.destroy_error(code_error);
        }
        self.destroy_error(error);

        Err(Error::Pjrt { code, message })
    }

    fn destroy_error(&self, error: NonNull<sys::PJRT_Error>) {
        // SAFETY: every error returned by this API must be destroyed through
        // the same API table. Destruction itself cannot report an error.
        let mut args: sys::PJRT_Error_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Error_Destroy_Args_STRUCT_SIZE as usize;
        args.error = error.as_ptr();
        if let Ok(function) = self.error_destroy_fn() {
            // SAFETY: validated function pointer and initialized args.
            unsafe { function(&mut args) };
        }
    }

    fn plugin_initialize_fn(&self) -> Result<PluginInitializeFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Plugin_Initialize).read() }
            .ok_or(Error::MissingFunction("PJRT_Plugin_Initialize"))
    }

    fn plugin_attributes_fn(&self) -> Result<PluginAttributesFn, Error> {
        self.require_function_field(
            offset_of!(sys::PJRT_Api, PJRT_Plugin_Attributes),
            "PJRT_Plugin_Attributes",
        )?;
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Plugin_Attributes).read() }
            .ok_or(Error::MissingFunction("PJRT_Plugin_Attributes"))
    }

    fn error_destroy_fn(&self) -> Result<ErrorDestroyFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Error_Destroy).read() }
            .ok_or(Error::MissingFunction("PJRT_Error_Destroy"))
    }

    fn error_message_fn(&self) -> Result<ErrorMessageFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Error_Message).read() }
            .ok_or(Error::MissingFunction("PJRT_Error_Message"))
    }

    fn error_get_code_fn(&self) -> Result<ErrorGetCodeFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Error_GetCode).read() }
            .ok_or(Error::MissingFunction("PJRT_Error_GetCode"))
    }

    fn client_create_fn(&self) -> Result<ClientCreateFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Client_Create).read() }
            .ok_or(Error::MissingFunction("PJRT_Client_Create"))
    }

    fn client_destroy_fn(&self) -> Result<ClientDestroyFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Client_Destroy).read() }
            .ok_or(Error::MissingFunction("PJRT_Client_Destroy"))
    }

    fn client_platform_name_fn(&self) -> Result<ClientPlatformNameFn, Error> {
        // SAFETY: field is within the size-checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Client_PlatformName).read() }
            .ok_or(Error::MissingFunction("PJRT_Client_PlatformName"))
    }

    fn client_devices_fn(&self) -> Result<ClientDevicesFn, Error> {
        // SAFETY: this is the final field in the checked prefix.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Client_Devices).read() }
            .ok_or(Error::MissingFunction("PJRT_Client_Devices"))
    }

    fn device_get_description_fn(&self) -> Result<DeviceGetDescriptionFn, Error> {
        self.require_function_field(
            offset_of!(sys::PJRT_Api, PJRT_Device_GetDescription),
            "PJRT_Device_GetDescription",
        )?;
        // SAFETY: require_function_field proved this pointer lies in the table.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_Device_GetDescription).read() }
            .ok_or(Error::MissingFunction("PJRT_Device_GetDescription"))
    }

    fn device_description_attributes_fn(&self) -> Result<DeviceDescriptionAttributesFn, Error> {
        self.require_function_field(
            offset_of!(sys::PJRT_Api, PJRT_DeviceDescription_Attributes),
            "PJRT_DeviceDescription_Attributes",
        )?;
        // SAFETY: require_function_field proved this pointer lies in the table.
        unsafe { addr_of!((*self.api.as_ptr()).PJRT_DeviceDescription_Attributes).read() }
            .ok_or(Error::MissingFunction("PJRT_DeviceDescription_Attributes"))
    }

    fn device_description_id_fn(&self) -> Result<DeviceDescriptionIdFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_DeviceDescription_Id),
            "PJRT_DeviceDescription_Id",
            |api| unsafe { addr_of!((*api).PJRT_DeviceDescription_Id).read() },
        )
    }

    fn event_destroy_fn(&self) -> Result<EventDestroyFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Event_Destroy),
            "PJRT_Event_Destroy",
            |api| unsafe { addr_of!((*api).PJRT_Event_Destroy).read() },
        )
    }
    fn event_is_ready_fn(&self) -> Result<EventIsReadyFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Event_IsReady),
            "PJRT_Event_IsReady",
            |api| unsafe { addr_of!((*api).PJRT_Event_IsReady).read() },
        )
    }
    fn event_await_fn(&self) -> Result<EventAwaitFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Event_Await),
            "PJRT_Event_Await",
            |api| unsafe { addr_of!((*api).PJRT_Event_Await).read() },
        )
    }
    fn client_compile_fn(&self) -> Result<ClientCompileFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Client_Compile),
            "PJRT_Client_Compile",
            |api| unsafe { addr_of!((*api).PJRT_Client_Compile).read() },
        )
    }
    fn client_buffer_from_host_buffer_fn(&self) -> Result<ClientBufferFromHostBufferFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Client_BufferFromHostBuffer),
            "PJRT_Client_BufferFromHostBuffer",
            |api| unsafe { addr_of!((*api).PJRT_Client_BufferFromHostBuffer).read() },
        )
    }
    fn buffer_destroy_fn(&self) -> Result<BufferDestroyFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_Destroy),
            "PJRT_Buffer_Destroy",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_Destroy).read() },
        )
    }
    fn buffer_element_type_fn(&self) -> Result<BufferElementTypeFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_ElementType),
            "PJRT_Buffer_ElementType",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_ElementType).read() },
        )
    }
    fn buffer_dimensions_fn(&self) -> Result<BufferDimensionsFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_Dimensions),
            "PJRT_Buffer_Dimensions",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_Dimensions).read() },
        )
    }
    fn buffer_to_host_buffer_fn(&self) -> Result<BufferToHostBufferFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_ToHostBuffer),
            "PJRT_Buffer_ToHostBuffer",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_ToHostBuffer).read() },
        )
    }
    fn buffer_ready_event_fn(&self) -> Result<BufferReadyEventFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_ReadyEvent),
            "PJRT_Buffer_ReadyEvent",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_ReadyEvent).read() },
        )
    }
    fn buffer_delete_fn(&self) -> Result<BufferDeleteFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_Delete),
            "PJRT_Buffer_Delete",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_Delete).read() },
        )
    }
    fn buffer_is_deleted_fn(&self) -> Result<BufferIsDeletedFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_IsDeleted),
            "PJRT_Buffer_IsDeleted",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_IsDeleted).read() },
        )
    }
    fn buffer_memory_fn(&self) -> Result<BufferMemoryFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Buffer_Memory),
            "PJRT_Buffer_Memory",
            |api| unsafe { addr_of!((*api).PJRT_Buffer_Memory).read() },
        )
    }
    fn loaded_executable_destroy_fn(&self) -> Result<LoadedExecutableDestroyFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_LoadedExecutable_Destroy),
            "PJRT_LoadedExecutable_Destroy",
            |api| unsafe { addr_of!((*api).PJRT_LoadedExecutable_Destroy).read() },
        )
    }
    fn loaded_executable_get_executable_fn(
        &self,
    ) -> Result<LoadedExecutableGetExecutableFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_LoadedExecutable_GetExecutable),
            "PJRT_LoadedExecutable_GetExecutable",
            |api| unsafe { addr_of!((*api).PJRT_LoadedExecutable_GetExecutable).read() },
        )
    }
    fn loaded_executable_execute_fn(&self) -> Result<LoadedExecutableExecuteFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_LoadedExecutable_Execute),
            "PJRT_LoadedExecutable_Execute",
            |api| unsafe { addr_of!((*api).PJRT_LoadedExecutable_Execute).read() },
        )
    }
    fn loaded_executable_addressable_devices_fn(
        &self,
    ) -> Result<LoadedExecutableAddressableDevicesFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_LoadedExecutable_AddressableDevices),
            "PJRT_LoadedExecutable_AddressableDevices",
            |api| unsafe { addr_of!((*api).PJRT_LoadedExecutable_AddressableDevices).read() },
        )
    }
    fn loaded_executable_delete_fn(&self) -> Result<LoadedExecutableDeleteFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_LoadedExecutable_Delete),
            "PJRT_LoadedExecutable_Delete",
            |api| unsafe { addr_of!((*api).PJRT_LoadedExecutable_Delete).read() },
        )
    }
    fn loaded_executable_is_deleted_fn(&self) -> Result<LoadedExecutableIsDeletedFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_LoadedExecutable_IsDeleted),
            "PJRT_LoadedExecutable_IsDeleted",
            |api| unsafe { addr_of!((*api).PJRT_LoadedExecutable_IsDeleted).read() },
        )
    }
    fn executable_destroy_fn(&self) -> Result<ExecutableDestroyFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Executable_Destroy),
            "PJRT_Executable_Destroy",
            |api| unsafe { addr_of!((*api).PJRT_Executable_Destroy).read() },
        )
    }
    fn executable_name_fn(&self) -> Result<ExecutableNameFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Executable_Name),
            "PJRT_Executable_Name",
            |api| unsafe { addr_of!((*api).PJRT_Executable_Name).read() },
        )
    }
    fn executable_num_outputs_fn(&self) -> Result<ExecutableNumOutputsFn, Error> {
        self.function(
            offset_of!(sys::PJRT_Api, PJRT_Executable_NumOutputs),
            "PJRT_Executable_NumOutputs",
            |api| unsafe { addr_of!((*api).PJRT_Executable_NumOutputs).read() },
        )
    }

    fn function<T: Copy>(
        &self,
        offset: usize,
        name: &'static str,
        read: impl FnOnce(*mut sys::PJRT_Api) -> Option<T>,
    ) -> Result<T, Error> {
        self.require_function_field(offset, name)?;
        read(self.api.as_ptr()).ok_or(Error::MissingFunction(name))
    }

    fn require_function_field(&self, offset: usize, name: &'static str) -> Result<(), Error> {
        if self.api_size < offset + size_of::<usize>() {
            Err(Error::MissingFunction(name))
        } else {
            Ok(())
        }
    }
}

impl fmt::Debug for Plugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plugin")
            .field("library", &self.library.path)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

/// A PJRT client whose final dependent object owns its destruction boundary.
#[derive(Clone)]
pub struct Client {
    state: Arc<ClientState>,
}

struct ClientState {
    plugin: Plugin,
    inner: NonNull<sys::PJRT_Client>,
    // PJRT thread-safety is an API/object-specific contract. Do not infer Send
    // or Sync for the whole client merely from its opaque pointer.
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl Client {
    pub fn stablehlo_version(&self) -> Result<Option<StableHloVersion>, Error> {
        self.state.plugin.stablehlo_version()
    }

    pub fn platform_name(&self) -> Result<String, Error> {
        // SAFETY: zero is valid for the output pointer and length.
        let mut args: sys::PJRT_Client_PlatformName_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_PlatformName_Args_STRUCT_SIZE as usize;
        args.client = self.state.inner.as_ptr();
        // SAFETY: validated function pointer and client/args owned by self.
        let error = unsafe { (self.state.plugin.client_platform_name_fn()?)(&mut args) };
        self.state.plugin.into_result(error)?;
        if args.platform_name.is_null() {
            return Err(Error::NullResult("PJRT_Client_PlatformName"));
        }
        // SAFETY: PJRT guarantees this client-owned byte range for the client's
        // lifetime. Return an owned string so the borrow does not escape FFI.
        let bytes = unsafe {
            std::slice::from_raw_parts(args.platform_name.cast::<u8>(), args.platform_name_size)
        };
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    pub fn device_count(&self) -> Result<usize, Error> {
        self.raw_devices().map(|(_, count)| count)
    }

    pub fn devices(&self) -> Result<Vec<Device>, Error> {
        let (devices, count) = self.raw_devices()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: raw_devices checked the non-null pointer and PJRT guarantees
        // count entries owned by the client.
        let devices = unsafe { std::slice::from_raw_parts(devices, count) };
        devices
            .iter()
            .map(|device| {
                NonNull::new(*device)
                    .map(|inner| Device {
                        client: self.clone(),
                        inner,
                    })
                    .ok_or(Error::NullResult("PJRT_Client_Devices entry"))
            })
            .collect()
    }

    /// Copies a dense row-major host tensor to a selected PJRT device.
    pub fn buffer_from_host(
        &self,
        data: &[u8],
        shape: Shape,
        device: &Device,
    ) -> Result<HostTransfer, Error> {
        self.require_own_device(device)?;
        let expected_layout =
            Layout::row_major(shape.rank()).map_err(|_| Error::TensorMetadataOverflow)?;
        if shape.layout() != expected_layout {
            return Err(Error::UnsupportedLayout {
                actual: shape.layout(),
                expected: expected_layout,
            });
        }
        let expected = shape
            .byte_count()
            .map_err(|_| Error::TensorMetadataOverflow)?;
        if data.len() != expected {
            return Err(Error::InvalidHostBuffer {
                expected,
                actual: data.len(),
            });
        }
        let mut args: sys::PJRT_Client_BufferFromHostBuffer_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE as usize;
        args.client = self.state.inner.as_ptr();
        args.data = data.as_ptr().cast();
        args.type_ = dtype_to_pjrt(shape.dtype());
        args.dims = shape.dimensions().as_ptr();
        args.num_dims = shape.rank();
        args.host_buffer_semantics =
            sys::PJRT_HostBufferSemantics_PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        args.device = device.inner.as_ptr();
        // A null memory and layout request the plugin's default memory and its
        // canonical dense layout, matching the dense host contract above.
        let error = unsafe { (self.state.plugin.client_buffer_from_host_buffer_fn()?)(&mut args) };
        self.state.plugin.into_result(error)?;
        // Wrap both independent outputs before validating either one. If a
        // broken plugin returns only one, the owned half is still destroyed
        // when this function reports the missing result.
        let buffer = NonNull::new(args.buffer).map(|buffer| Buffer::from_raw(self, buffer));
        let done = NonNull::new(args.done_with_host_buffer)
            .map(|event| Event::from_raw(&self.state.plugin, event));
        match (buffer, done) {
            (Some(buffer), Some(done)) => Ok(HostTransfer { buffer, done }),
            (None, _) => Err(Error::NullResult("PJRT_Client_BufferFromHostBuffer buffer")),
            (_, None) => Err(Error::NullResult("PJRT_Client_BufferFromHostBuffer event")),
        }
    }

    /// Compiles an MLIR/StableHLO program with serialized XLA options.
    pub fn compile(&self, mlir: &[u8], compile_options: &[u8]) -> Result<LoadedExecutable, Error> {
        let format = b"mlir";
        let program = sys::PJRT_Program {
            struct_size: sys::PJRT_Program_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            code: mlir.as_ptr().cast_mut().cast(),
            code_size: mlir.len(),
            format: format.as_ptr().cast(),
            format_size: format.len(),
        };
        let mut args: sys::PJRT_Client_Compile_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_Compile_Args_STRUCT_SIZE as usize;
        args.client = self.state.inner.as_ptr();
        args.program = &program;
        args.compile_options = compile_options.as_ptr().cast();
        args.compile_options_size = compile_options.len();
        let error = unsafe { (self.state.plugin.client_compile_fn()?)(&mut args) };
        self.state.plugin.into_result(error)?;
        let inner = NonNull::new(args.executable)
            .ok_or(Error::NullResult("PJRT_Client_Compile executable"))?;
        Ok(LoadedExecutable {
            client: self.clone(),
            inner,
        })
    }

    fn require_own_device(&self, device: &Device) -> Result<(), Error> {
        if Arc::ptr_eq(&self.state, &device.client.state) {
            Ok(())
        } else {
            Err(Error::ForeignClientObject("device"))
        }
    }

    fn raw_devices(&self) -> Result<(*const *mut sys::PJRT_Device, usize), Error> {
        // SAFETY: zero is valid for the output pointer and length.
        let mut args: sys::PJRT_Client_Devices_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_Devices_Args_STRUCT_SIZE as usize;
        args.client = self.state.inner.as_ptr();
        // SAFETY: validated function pointer and client/args owned by self.
        let error = unsafe { (self.state.plugin.client_devices_fn()?)(&mut args) };
        self.state.plugin.into_result(error)?;
        if args.num_devices != 0 && args.devices.is_null() {
            return Err(Error::NullResult("PJRT_Client_Devices"));
        }
        Ok((args.devices, args.num_devices))
    }
}

/// A client-owned device identity. The PJRT device itself is non-owning, while
/// the retained client keeps the pointer valid independently of lexical scope.
pub struct Device {
    client: Client,
    inner: NonNull<sys::PJRT_Device>,
}

impl Device {
    pub fn id(&self) -> Result<i64, Error> {
        let description = self.description()?;
        let mut args: sys::PJRT_DeviceDescription_Id_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_DeviceDescription_Id_Args_STRUCT_SIZE as usize;
        args.device_description = description.as_ptr();
        let error = unsafe { (self.client.state.plugin.device_description_id_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        Ok(args.id.into())
    }

    pub fn string_attribute(&self, requested_name: &str) -> Result<Option<String>, Error> {
        let description = self.description()?;

        // SAFETY: zero is valid for the output attributes pointer/count.
        let mut attributes_args: sys::PJRT_DeviceDescription_Attributes_Args = unsafe { zeroed() };
        attributes_args.struct_size =
            sys::PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE as usize;
        attributes_args.device_description = description.as_ptr();
        // SAFETY: function pointer is available and description/args are live.
        let error = unsafe {
            (self
                .client
                .state
                .plugin
                .device_description_attributes_fn()?)(&mut attributes_args)
        };
        self.client.state.plugin.into_result(error)?;
        if attributes_args.num_attributes != 0 && attributes_args.attributes.is_null() {
            return Err(Error::NullResult("PJRT_DeviceDescription_Attributes"));
        }
        if attributes_args.num_attributes == 0 {
            return Ok(None);
        }

        // SAFETY: PJRT guarantees num_attributes client-owned entries.
        let attributes = unsafe {
            std::slice::from_raw_parts(attributes_args.attributes, attributes_args.num_attributes)
        };
        for attribute in attributes {
            let name = copy_bytes(attribute.name, attribute.name_size)?;
            if name != requested_name.as_bytes() {
                continue;
            }
            if attribute.type_ != sys::PJRT_NamedValue_Type_PJRT_NamedValue_kString {
                return Ok(None);
            }
            // SAFETY: the discriminant above selects the string union member.
            let value = unsafe { attribute.__bindgen_anon_1.string_value };
            let bytes = copy_bytes(value, attribute.value_size)?;
            return Ok(Some(String::from_utf8_lossy(&bytes).into_owned()));
        }
        Ok(None)
    }

    fn description(&self) -> Result<NonNull<sys::PJRT_DeviceDescription>, Error> {
        let mut args: sys::PJRT_Device_GetDescription_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Device_GetDescription_Args_STRUCT_SIZE as usize;
        args.device = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.device_get_description_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        NonNull::new(args.device_description).ok_or(Error::NullResult("PJRT_Device_GetDescription"))
    }
}

/// Owned completion event returned by PJRT asynchronous operations.
pub struct Event {
    plugin: Plugin,
    inner: NonNull<sys::PJRT_Event>,
}

impl Event {
    fn from_raw(plugin: &Plugin, inner: NonNull<sys::PJRT_Event>) -> Self {
        Self {
            plugin: plugin.clone(),
            inner,
        }
    }

    pub fn is_ready(&self) -> Result<bool, Error> {
        let mut args: sys::PJRT_Event_IsReady_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Event_IsReady_Args_STRUCT_SIZE as usize;
        args.event = self.inner.as_ptr();
        let error = unsafe { (self.plugin.event_is_ready_fn()?)(&mut args) };
        self.plugin.into_result(error)?;
        Ok(args.is_ready)
    }

    pub fn wait(&self) -> Result<(), Error> {
        let mut args: sys::PJRT_Event_Await_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Event_Await_Args_STRUCT_SIZE as usize;
        args.event = self.inner.as_ptr();
        let error = unsafe { (self.plugin.event_await_fn()?)(&mut args) };
        self.plugin.into_result(error)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let mut args: sys::PJRT_Event_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Event_Destroy_Args_STRUCT_SIZE as usize;
        args.event = self.inner.as_ptr();
        if let Ok(function) = self.plugin.event_destroy_fn() {
            let error = unsafe { function(&mut args) };
            let _ = self.plugin.into_result(error);
        }
    }
}

/// Host-to-device transfer products. The buffer may be consumed immediately;
/// `done` reports when PJRT has completed its host-side transfer work.
pub struct HostTransfer {
    pub buffer: Buffer,
    pub done: Event,
}

/// Owned device buffer whose client and plugin necessarily remain live.
pub struct Buffer {
    client: Client,
    inner: NonNull<sys::PJRT_Buffer>,
}

impl Buffer {
    fn from_raw(client: &Client, inner: NonNull<sys::PJRT_Buffer>) -> Self {
        Self {
            client: client.clone(),
            inner,
        }
    }

    pub fn dtype(&self) -> Result<DType, Error> {
        let mut args: sys::PJRT_Buffer_ElementType_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_ElementType_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.buffer_element_type_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        dtype_from_pjrt(args.type_)
    }

    pub fn dimensions(&self) -> Result<Vec<i64>, Error> {
        let mut args: sys::PJRT_Buffer_Dimensions_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_Dimensions_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.buffer_dimensions_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        if args.num_dims != 0 && args.dims.is_null() {
            return Err(Error::NullResult("PJRT_Buffer_Dimensions"));
        }
        Ok(if args.num_dims == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(args.dims, args.num_dims) }.to_vec()
        })
    }

    pub fn shape(&self) -> Result<Shape, Error> {
        Shape::new(self.dtype()?, &self.dimensions()?).map_err(|_| Error::TensorMetadataOverflow)
    }

    pub fn ready_event(&self) -> Result<Event, Error> {
        let mut args: sys::PJRT_Buffer_ReadyEvent_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.buffer_ready_event_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        let inner = NonNull::new(args.event).ok_or(Error::NullResult("PJRT_Buffer_ReadyEvent"))?;
        Ok(Event::from_raw(&self.client.state.plugin, inner))
    }

    pub fn to_host(&self) -> Result<Vec<u8>, Error> {
        let byte_count = self
            .shape()?
            .byte_count()
            .map_err(|_| Error::TensorMetadataOverflow)?;
        let mut bytes = vec![0u8; byte_count];
        let mut args: sys::PJRT_Buffer_ToHostBuffer_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE as usize;
        args.src = self.inner.as_ptr();
        args.dst = bytes.as_mut_ptr().cast();
        args.dst_size = bytes.len();
        let error = unsafe { (self.client.state.plugin.buffer_to_host_buffer_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        let event =
            NonNull::new(args.event).ok_or(Error::NullResult("PJRT_Buffer_ToHostBuffer event"))?;
        Event::from_raw(&self.client.state.plugin, event).wait()?;
        Ok(bytes)
    }

    pub fn delete(&self) -> Result<(), Error> {
        let mut args: sys::PJRT_Buffer_Delete_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_Delete_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.buffer_delete_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)
    }

    pub fn is_deleted(&self) -> Result<bool, Error> {
        let mut args: sys::PJRT_Buffer_IsDeleted_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.buffer_is_deleted_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        Ok(args.is_deleted)
    }

    pub fn memory(&self) -> Result<Memory, Error> {
        let mut args: sys::PJRT_Buffer_Memory_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_Memory_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.buffer_memory_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        let inner = NonNull::new(args.memory).ok_or(Error::NullResult("PJRT_Buffer_Memory"))?;
        Ok(Memory {
            client: self.client.clone(),
            inner,
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let mut args: sys::PJRT_Buffer_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Buffer_Destroy_Args_STRUCT_SIZE as usize;
        args.buffer = self.inner.as_ptr();
        if let Ok(function) = self.client.state.plugin.buffer_destroy_fn() {
            let error = unsafe { function(&mut args) };
            let _ = self.client.state.plugin.into_result(error);
        }
    }
}

/// Client-owned memory space referenced by a buffer.
pub struct Memory {
    #[allow(dead_code)]
    client: Client,
    inner: NonNull<sys::PJRT_Memory>,
}

impl Memory {
    pub fn as_raw_identity(&self) -> usize {
        self.inner.as_ptr() as usize
    }
}

/// Owned unloaded executable metadata extracted from a loaded executable.
pub struct Executable {
    client: Client,
    inner: NonNull<sys::PJRT_Executable>,
}

impl Executable {
    pub fn name(&self) -> Result<String, Error> {
        let mut args: sys::PJRT_Executable_Name_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Executable_Name_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.executable_name_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        let bytes = copy_bytes(args.executable_name, args.executable_name_size)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn output_count(&self) -> Result<usize, Error> {
        let mut args: sys::PJRT_Executable_NumOutputs_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Executable_NumOutputs_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.executable_num_outputs_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        Ok(args.num_outputs)
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        let mut args: sys::PJRT_Executable_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Executable_Destroy_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        if let Ok(function) = self.client.state.plugin.executable_destroy_fn() {
            let error = unsafe { function(&mut args) };
            let _ = self.client.state.plugin.into_result(error);
        }
    }
}

/// Owned executable installed into one PJRT client.
pub struct LoadedExecutable {
    client: Client,
    inner: NonNull<sys::PJRT_LoadedExecutable>,
}

impl LoadedExecutable {
    pub fn executable(&self) -> Result<Executable, Error> {
        let mut args: sys::PJRT_LoadedExecutable_GetExecutable_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE as usize;
        args.loaded_executable = self.inner.as_ptr();
        let error = unsafe {
            (self
                .client
                .state
                .plugin
                .loaded_executable_get_executable_fn()?)(&mut args)
        };
        self.client.state.plugin.into_result(error)?;
        let inner = NonNull::new(args.executable)
            .ok_or(Error::NullResult("PJRT_LoadedExecutable_GetExecutable"))?;
        Ok(Executable {
            client: self.client.clone(),
            inner,
        })
    }

    pub fn addressable_devices(&self) -> Result<Vec<Device>, Error> {
        let mut args: sys::PJRT_LoadedExecutable_AddressableDevices_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        let error = unsafe {
            (self
                .client
                .state
                .plugin
                .loaded_executable_addressable_devices_fn()?)(&mut args)
        };
        self.client.state.plugin.into_result(error)?;
        if args.num_addressable_devices != 0 && args.addressable_devices.is_null() {
            return Err(Error::NullResult(
                "PJRT_LoadedExecutable_AddressableDevices",
            ));
        }
        if args.num_addressable_devices == 0 {
            return Ok(Vec::new());
        }
        unsafe {
            std::slice::from_raw_parts(args.addressable_devices, args.num_addressable_devices)
        }
        .iter()
        .map(|device| {
            Ok(Device {
                client: self.client.clone(),
                inner: NonNull::new(*device).ok_or(Error::NullResult(
                    "PJRT_LoadedExecutable_AddressableDevices entry",
                ))?,
            })
        })
        .collect()
    }

    pub fn execute_one(
        &self,
        inputs: &[&Buffer],
        device: Option<&Device>,
    ) -> Result<Execution, Error> {
        for input in inputs {
            if !Arc::ptr_eq(&self.client.state, &input.client.state) {
                return Err(Error::ForeignClientObject("buffer"));
            }
        }
        if let Some(device) = device {
            self.client.require_own_device(device)?;
        }
        let output_count = self.executable()?.output_count()?;
        let argument_pointers: Vec<_> = inputs.iter().map(|buffer| buffer.inner.as_ptr()).collect();
        let argument_lists = [argument_pointers.as_ptr()];
        let mut output_pointers = vec![std::ptr::null_mut(); output_count];
        let output_lists = [output_pointers.as_mut_ptr()];
        let mut complete_events = [std::ptr::null_mut()];
        let mut options: sys::PJRT_ExecuteOptions = unsafe { zeroed() };
        options.struct_size = sys::PJRT_ExecuteOptions_STRUCT_SIZE as usize;
        let mut args: sys::PJRT_LoadedExecutable_Execute_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        args.options = &mut options;
        args.argument_lists = argument_lists.as_ptr();
        args.num_devices = 1;
        args.num_args = inputs.len();
        args.output_lists = output_lists.as_ptr();
        args.device_complete_events = complete_events.as_mut_ptr();
        args.execute_device = device.map_or(std::ptr::null_mut(), |device| device.inner.as_ptr());
        let error =
            unsafe { (self.client.state.plugin.loaded_executable_execute_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        // PJRT transfers ownership of every non-null output on success. Adopt
        // all of them before reporting a malformed partial result so no later
        // entry leaks merely because an earlier entry was null.
        let outputs = output_pointers
            .into_iter()
            .map(|buffer| NonNull::new(buffer).map(|inner| Buffer::from_raw(&self.client, inner)))
            .collect::<Vec<_>>();
        let complete = NonNull::new(complete_events[0])
            .map(|event| Event::from_raw(&self.client.state.plugin, event));
        if outputs.iter().any(Option::is_none) {
            return Err(Error::NullResult("PJRT_LoadedExecutable_Execute output"));
        }
        let outputs = outputs.into_iter().flatten().collect();
        let complete = complete.ok_or(Error::NullResult("PJRT_LoadedExecutable_Execute event"))?;
        Ok(Execution { outputs, complete })
    }

    pub fn delete(&self) -> Result<(), Error> {
        let mut args: sys::PJRT_LoadedExecutable_Delete_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        let error = unsafe { (self.client.state.plugin.loaded_executable_delete_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)
    }

    pub fn is_deleted(&self) -> Result<bool, Error> {
        let mut args: sys::PJRT_LoadedExecutable_IsDeleted_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        let error =
            unsafe { (self.client.state.plugin.loaded_executable_is_deleted_fn()?)(&mut args) };
        self.client.state.plugin.into_result(error)?;
        Ok(args.is_deleted)
    }
}

impl Drop for LoadedExecutable {
    fn drop(&mut self) {
        let mut args: sys::PJRT_LoadedExecutable_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE as usize;
        args.executable = self.inner.as_ptr();
        if let Ok(function) = self.client.state.plugin.loaded_executable_destroy_fn() {
            let error = unsafe { function(&mut args) };
            let _ = self.client.state.plugin.into_result(error);
        }
    }
}

pub struct Execution {
    pub outputs: Vec<Buffer>,
    pub complete: Event,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

impl Drop for ClientState {
    fn drop(&mut self) {
        // SAFETY: zero initializes the extension pointer and self owns client.
        let mut args: sys::PJRT_Client_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_Destroy_Args_STRUCT_SIZE as usize;
        args.client = self.inner.as_ptr();
        if let Ok(function) = self.plugin.client_destroy_fn() {
            // SAFETY: validated function pointer and live owned client.
            let error = unsafe { function(&mut args) };
            let _ = self.plugin.into_result(error);
        }
    }
}

struct DynamicLibrary {
    handle: NonNull<c_void>,
    path: PathBuf,
}

impl DynamicLibrary {
    unsafe fn open(path: &Path) -> Result<Self, Error> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| Error::InvalidLibraryPath(path.to_owned()))?;
        let flags = if cfg!(target_os = "linux") {
            // Match ZML: GLOBAL exposes dependencies to later-loaded native
            // libraries and NODELETE keeps process-global XLA state resident.
            RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE
        } else {
            // Match ZML's macOS loader: lazy binding with local visibility.
            RTLD_LAZY | RTLD_LOCAL
        };
        clear_dlerror();
        // SAFETY: c_path is NUL-terminated and flags are valid for dlopen.
        let handle = NonNull::new(unsafe { dlopen(c_path.as_ptr(), flags) }).ok_or_else(|| {
            Error::DynamicLibrary {
                path: path.to_owned(),
                message: dlerror_message(),
            }
        })?;
        Ok(Self {
            handle,
            path: path.to_owned(),
        })
    }

    unsafe fn symbol<T: Copy>(&self, name: &'static str) -> Result<T, Error> {
        let c_name = CString::new(name).expect("static symbol names contain no NUL");
        clear_dlerror();
        // SAFETY: handle is live and c_name is NUL-terminated.
        let address = unsafe { dlsym(self.handle.as_ptr(), c_name.as_ptr()) };
        let error = dlerror_message_if_present();
        if address.is_null() || error.is_some() {
            return Err(Error::MissingSymbol {
                symbol: name,
                message: error.unwrap_or_else(|| "symbol address is null".to_owned()),
            });
        }
        assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
        // SAFETY: the caller supplies the symbol's ABI type, and the equal-size
        // assertion establishes representation width for this pointer copy.
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
}

// Deliberately no Drop implementation. ZML keeps PJRT libraries resident, and
// PJRT exposes initialization but no symmetric plugin-shutdown operation. A
// Rust wrapper therefore cannot prove that dlclose is safe after a client is
// destroyed: plugin-owned threads or process-global XLA state may remain. The
// operating system reclaims the handle at process exit; Linux additionally
// receives RTLD_NODELETE to make that contract explicit to the loader.

const RTLD_LAZY: c_int = 0x1;
#[cfg(target_os = "linux")]
const RTLD_GLOBAL: c_int = 0x100;
#[cfg(target_os = "macos")]
const RTLD_GLOBAL: c_int = 0;
#[cfg(target_os = "linux")]
const RTLD_NODELETE: c_int = 0x1000;
#[cfg(target_os = "macos")]
const RTLD_NODELETE: c_int = 0;
#[cfg(target_os = "linux")]
const RTLD_LOCAL: c_int = 0;
#[cfg(target_os = "macos")]
const RTLD_LOCAL: c_int = 0x4;

#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

fn clear_dlerror() {
    // SAFETY: dlerror has no preconditions and clears the calling thread's
    // dynamic-loader error state.
    let _ = unsafe { dlerror() };
}

fn dlerror_message_if_present() -> Option<String> {
    // SAFETY: a non-null result is a NUL-terminated thread-local error string.
    let error = unsafe { dlerror() };
    if error.is_null() {
        None
    } else {
        // SAFETY: non-null dlerror results are valid C strings until the next
        // dynamic-loader operation on this thread.
        Some(
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

fn dlerror_message() -> String {
    dlerror_message_if_present().unwrap_or_else(|| "dynamic loader returned no detail".to_owned())
}

fn named_value_to_raw(value: &NamedValue<'_>) -> sys::PJRT_NamedValue {
    // SAFETY: every field is overwritten below, including the active union
    // member selected by type_.
    let mut raw: sys::PJRT_NamedValue = unsafe { zeroed() };
    raw.struct_size = sys::PJRT_NamedValue_STRUCT_SIZE as usize;
    let (name, value_size) = match value {
        NamedValue::String { name, value } => {
            raw.type_ = sys::PJRT_NamedValue_Type_PJRT_NamedValue_kString;
            raw.__bindgen_anon_1 = sys::PJRT_NamedValue__bindgen_ty_1 {
                string_value: value.as_ptr().cast(),
            };
            (*name, value.len())
        }
        NamedValue::Int64 { name, value } => {
            raw.type_ = sys::PJRT_NamedValue_Type_PJRT_NamedValue_kInt64;
            raw.__bindgen_anon_1 = sys::PJRT_NamedValue__bindgen_ty_1 {
                int64_value: *value,
            };
            (*name, 1)
        }
        NamedValue::Float { name, value } => {
            raw.type_ = sys::PJRT_NamedValue_Type_PJRT_NamedValue_kFloat;
            raw.__bindgen_anon_1 = sys::PJRT_NamedValue__bindgen_ty_1 {
                float_value: *value,
            };
            (*name, 1)
        }
        NamedValue::Bool { name, value } => {
            raw.type_ = sys::PJRT_NamedValue_Type_PJRT_NamedValue_kBool;
            raw.__bindgen_anon_1 = sys::PJRT_NamedValue__bindgen_ty_1 { bool_value: *value };
            (*name, 1)
        }
    };
    raw.name = name.as_ptr().cast();
    raw.name_size = name.len();
    raw.value_size = value_size;
    raw
}

fn copy_bytes(pointer: *const c_char, length: usize) -> Result<Vec<u8>, Error> {
    if pointer.is_null() {
        if length == 0 {
            return Ok(Vec::new());
        }
        return Err(Error::NullResult("PJRT byte string"));
    }
    // SAFETY: callers pass PJRT-owned pointers with the accompanying length.
    // Copying prevents an FFI-owned borrow from escaping its owner.
    Ok(unsafe { std::slice::from_raw_parts(pointer.cast(), length) }.to_vec())
}

fn dtype_to_pjrt(dtype: DType) -> sys::PJRT_Buffer_Type {
    match dtype {
        DType::Bool => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_PRED,
        DType::I8 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S8,
        DType::I16 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S16,
        DType::I32 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S32,
        DType::I64 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S64,
        DType::U8 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U8,
        DType::U16 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U16,
        DType::U32 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U32,
        DType::U64 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U64,
        DType::F16 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_F16,
        DType::Bf16 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_BF16,
        DType::F32 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_F32,
        DType::F64 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_F64,
        DType::C64 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_C64,
        DType::C128 => sys::PJRT_Buffer_Type_PJRT_Buffer_Type_C128,
    }
}

fn dtype_from_pjrt(dtype: sys::PJRT_Buffer_Type) -> Result<DType, Error> {
    match dtype {
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_PRED => Ok(DType::Bool),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S8 => Ok(DType::I8),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S16 => Ok(DType::I16),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S32 => Ok(DType::I32),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_S64 => Ok(DType::I64),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U8 => Ok(DType::U8),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U16 => Ok(DType::U16),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U32 => Ok(DType::U32),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_U64 => Ok(DType::U64),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_F16 => Ok(DType::F16),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_BF16 => Ok(DType::Bf16),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_F32 => Ok(DType::F32),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_F64 => Ok(DType::F64),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_C64 => Ok(DType::C64),
        sys::PJRT_Buffer_Type_PJRT_Buffer_Type_C128 => Ok(DType::C128),
        other => Err(Error::UnknownBufferType(other)),
    }
}
