//! Ownership and error handling for the common PJRT plugin ABI.
//!
//! ZML has one broad PJRT wrapper shared by its platform implementations. NML
//! keeps that boundary: this crate knows how to load and call a PJRT C API and
//! its typed extension chain, but it does not know where CPU or CUDA plugins
//! are packaged. Platform crates own runtime selection and policy.

use nml_pjrt_sys as sys;
use std::collections::HashSet;
use std::error::Error as StdError;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::marker::PhantomData;
use std::mem::{offset_of, size_of, zeroed};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::ptr::{NonNull, addr_of};

type PluginInitializeFn =
    unsafe extern "C" fn(*mut sys::PJRT_Plugin_Initialize_Args) -> *mut sys::PJRT_Error;
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
pub struct GpuCustomCalls<'plugin> {
    plugin: &'plugin Plugin,
    register_fn: GpuRegisterCustomCallFn,
}

type GpuRegisterCustomCallFn =
    unsafe extern "C" fn(*mut sys::PJRT_Gpu_Register_Custom_Call_Args) -> *mut sys::PJRT_Error;

impl GpuCustomCalls<'_> {
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
/// The dynamic-library handle and API pointer are one ownership unit. Clients
/// borrow this value, so Rust prevents unloading a plugin while any PJRT object
/// created from it remains live.
pub struct Plugin {
    library: DynamicLibrary,
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
            library,
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

    /// Finds and validates PJRT's GPU custom-call registration extension.
    ///
    /// Absence is not an ABI error: CPU plugins and older GPU plugins may not
    /// publish it. A present but truncated or cyclic extension list is rejected
    /// because invoking it would be undefined behavior.
    pub fn gpu_custom_calls(&self) -> Result<Option<GpuCustomCalls<'_>>, Error> {
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
                    plugin: self,
                    register_fn,
                }));
            }
            current = base.next;
        }
        Ok(None)
    }

    pub fn create_client(&self) -> Result<Client<'_>, Error> {
        self.create_client_with_options(&[])
    }

    pub fn create_client_with_options(
        &self,
        options: &[NamedValue<'_>],
    ) -> Result<Client<'_>, Error> {
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
            plugin: self,
            inner,
            _not_send_or_sync: PhantomData,
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

/// A PJRT client tied to the lifetime of its originating plugin.
pub struct Client<'plugin> {
    plugin: &'plugin Plugin,
    inner: NonNull<sys::PJRT_Client>,
    // PJRT thread-safety is an API/object-specific contract. Do not infer Send
    // or Sync for the whole client merely from its opaque pointer.
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl<'plugin> Client<'plugin> {
    pub fn platform_name(&self) -> Result<String, Error> {
        // SAFETY: zero is valid for the output pointer and length.
        let mut args: sys::PJRT_Client_PlatformName_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_PlatformName_Args_STRUCT_SIZE as usize;
        args.client = self.inner.as_ptr();
        // SAFETY: validated function pointer and client/args owned by self.
        let error = unsafe { (self.plugin.client_platform_name_fn()?)(&mut args) };
        self.plugin.into_result(error)?;
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

    pub fn devices(&self) -> Result<Vec<Device<'_, 'plugin>>, Error> {
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
                        client: self,
                        inner,
                    })
                    .ok_or(Error::NullResult("PJRT_Client_Devices entry"))
            })
            .collect()
    }

    fn raw_devices(&self) -> Result<(*const *mut sys::PJRT_Device, usize), Error> {
        // SAFETY: zero is valid for the output pointer and length.
        let mut args: sys::PJRT_Client_Devices_Args = unsafe { zeroed() };
        args.struct_size = sys::PJRT_Client_Devices_Args_STRUCT_SIZE as usize;
        args.client = self.inner.as_ptr();
        // SAFETY: validated function pointer and client/args owned by self.
        let error = unsafe { (self.plugin.client_devices_fn()?)(&mut args) };
        self.plugin.into_result(error)?;
        if args.num_devices != 0 && args.devices.is_null() {
            return Err(Error::NullResult("PJRT_Client_Devices"));
        }
        Ok((args.devices, args.num_devices))
    }
}

/// A non-owning device whose lifetime is bounded by its PJRT client.
pub struct Device<'client, 'plugin> {
    client: &'client Client<'plugin>,
    inner: NonNull<sys::PJRT_Device>,
}

impl Device<'_, '_> {
    pub fn string_attribute(&self, requested_name: &str) -> Result<Option<String>, Error> {
        // SAFETY: zero is valid for the output description pointer.
        let mut description_args: sys::PJRT_Device_GetDescription_Args = unsafe { zeroed() };
        description_args.struct_size = sys::PJRT_Device_GetDescription_Args_STRUCT_SIZE as usize;
        description_args.device = self.inner.as_ptr();
        // SAFETY: function pointer is available and device/args are live.
        let error =
            unsafe { (self.client.plugin.device_get_description_fn()?)(&mut description_args) };
        self.client.plugin.into_result(error)?;
        let description = NonNull::new(description_args.device_description)
            .ok_or(Error::NullResult("PJRT_Device_GetDescription"))?;

        // SAFETY: zero is valid for the output attributes pointer/count.
        let mut attributes_args: sys::PJRT_DeviceDescription_Attributes_Args = unsafe { zeroed() };
        attributes_args.struct_size =
            sys::PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE as usize;
        attributes_args.device_description = description.as_ptr();
        // SAFETY: function pointer is available and description/args are live.
        let error = unsafe {
            (self.client.plugin.device_description_attributes_fn()?)(&mut attributes_args)
        };
        self.client.plugin.into_result(error)?;
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
}

impl fmt::Debug for Client<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

impl Drop for Client<'_> {
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
