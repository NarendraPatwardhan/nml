//! CUDA PJRT runtime initialization.
//!
//! This is the Rust adaptation of ZML's `platforms/cuda` loader. It preserves
//! ZML's runtime order because XLA observes process state while its shared
//! object is loaded: resolve the hermetic runtime, configure its CUDA data
//! directory, decide whether the packaged compatibility driver is required,
//! and only then initialize the PJRT plugin.

use nml_pjrt::{Client, GpuCustomCalls, NamedValue, Plugin};
use runfiles::Runfiles;
use std::error::Error as StdError;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::fs::File;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const CUDA_RUNTIME_RLOCATION: &str = env!("NML_CUDA_RUNTIME_RLOCATION");
const CUDA_LIBRARY_PATH_FRAGMENT: &str = "/cuda/";
const RTLD_NOW: c_int = 0x2;

/// Client allocation modes exposed by XLA's CUDA PJRT plugin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Allocator {
    /// Best-fit with coalescing, matching NML and ZML's default policy.
    Bfc(AllocatorOptions),
    /// CUDA's stream-ordered allocator.
    CudaAsync(AllocatorOptions),
    /// Direct platform allocation without a PJRT memory pool.
    Platform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AllocatorOptions {
    pub preallocate: bool,
    pub memory_fraction: f32,
    pub collective_memory_size_bytes: i64,
}

impl Default for AllocatorOptions {
    fn default() -> Self {
        Self {
            preallocate: true,
            memory_fraction: 0.90,
            collective_memory_size_bytes: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClientOptions {
    pub allocator: Allocator,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            allocator: Allocator::Bfc(AllocatorOptions::default()),
        }
    }
}

/// A compute capability reported by the CUDA PJRT device description.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComputeCapability {
    pub major: u16,
    pub minor: u16,
}

impl ComputeCapability {
    fn parse(value: &str) -> Option<Self> {
        let (major, minor) = value.split_once('.')?;
        if minor.contains('.') {
            return None;
        }
        Some(Self {
            major: major.parse().ok()?,
            minor: minor.parse().ok()?,
        })
    }
}

#[derive(Debug)]
pub enum Error {
    Disabled,
    NoNvidiaDevice,
    Runfiles(String),
    MissingRuntime(String),
    InvalidRuntimePath(PathBuf),
    CompatibilityDriver { path: PathBuf, message: String },
    Pjrt(nml_pjrt::Error),
    UnexpectedPlatform(String),
    NoCudaDevices,
    MissingComputeCapability(usize),
    InvalidComputeCapability { device_index: usize, value: String },
    MissingGpuCustomCallExtension,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => {
                f.write_str("the CUDA PJRT backend is disabled in this Bazel configuration")
            }
            Self::NoNvidiaDevice => f.write_str(
                "no readable NVIDIA device node was found at /dev/nvidiactl or /dev/dxg",
            ),
            Self::Runfiles(message) => write!(f, "failed to initialize Bazel runfiles: {message}"),
            Self::MissingRuntime(path) => {
                write!(f, "CUDA PJRT runtime is absent from Bazel runfiles: {path}")
            }
            Self::InvalidRuntimePath(path) => write!(
                f,
                "CUDA runtime path contains a NUL byte: {}",
                path.display()
            ),
            Self::CompatibilityDriver { path, message } => write!(
                f,
                "failed to load CUDA compatibility driver {}: {message}",
                path.display()
            ),
            Self::Pjrt(error) => error.fmt(f),
            Self::UnexpectedPlatform(platform) => {
                write!(f, "CUDA PJRT plugin reported platform {platform:?}")
            }
            Self::NoCudaDevices => f.write_str("CUDA PJRT plugin exposed no devices"),
            Self::MissingComputeCapability(index) => {
                write!(f, "CUDA device {index} has no compute_capability attribute")
            }
            Self::InvalidComputeCapability {
                device_index,
                value,
            } => write!(
                f,
                "CUDA device {device_index} reported invalid compute capability {value:?}"
            ),
            Self::MissingGpuCustomCallExtension => {
                f.write_str("CUDA PJRT plugin does not expose GPU custom-call registration")
            }
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Pjrt(error) => Some(error),
            _ => None,
        }
    }
}

impl From<nml_pjrt::Error> for Error {
    fn from(error: nml_pjrt::Error) -> Self {
        Self::Pjrt(error)
    }
}

/// Loaded CUDA PJRT code and the hermetic runtime directory it belongs to.
pub struct Runtime {
    plugin: Plugin,
    directory: PathBuf,
}

impl Runtime {
    /// Initializes CUDA's process-global XLA state and loads the PJRT plugin.
    ///
    /// # Safety
    ///
    /// This must run before the process starts any other thread that can read
    /// environment variables. XLA requires `XLA_FLAGS` to name the hermetic
    /// CUDA directory before its plugin is loaded. Rust cannot make mutation
    /// of a process environment safe in the presence of concurrent readers.
    pub unsafe fn load() -> Result<Self, Error> {
        if !is_enabled() {
            return Err(Error::Disabled);
        }
        if !has_nvidia_device() {
            return Err(Error::NoNvidiaDevice);
        }
        warn_for_external_cuda_libraries();

        let runfiles = Runfiles::create().map_err(|error| Error::Runfiles(error.to_string()))?;
        let directory = runfiles::rlocation!(runfiles, CUDA_RUNTIME_RLOCATION)
            .ok_or_else(|| Error::MissingRuntime(CUDA_RUNTIME_RLOCATION.to_owned()))?;
        if !directory.is_dir() {
            return Err(Error::MissingRuntime(directory.display().to_string()));
        }

        // SAFETY: the caller promises process-wide environment exclusivity.
        unsafe { configure_xla_flags(&directory) };
        if use_packaged_driver(&directory) {
            load_compatibility_driver(&directory)?;
        }

        let plugin_path = directory.join("lib/libpjrt_cuda.so");
        // SAFETY: the path is a file in the digest-pinned Bazel runtime tree.
        let plugin = unsafe { Plugin::load_trusted(plugin_path) }?;
        Ok(Self { plugin, directory })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Returns the CUDA plugin's validated custom-call registration interface.
    pub fn custom_calls(&self) -> Result<GpuCustomCalls<'_>, Error> {
        self.plugin
            .gpu_custom_calls()?
            .ok_or(Error::MissingGpuCustomCallExtension)
    }

    /// Creates a client using ZML's XLA GPU allocation policy and validates
    /// every device exposed by the pinned plugin.
    pub fn create_client(&self, options: ClientOptions) -> Result<Client<'_>, Error> {
        let mut values = Vec::with_capacity(4);
        match options.allocator {
            Allocator::Platform => values.push(NamedValue::String {
                name: "allocator",
                value: "platform",
            }),
            Allocator::Bfc(allocation) => write_allocator_options(&mut values, "bfc", allocation),
            Allocator::CudaAsync(allocation) => {
                write_allocator_options(&mut values, "cuda_async", allocation)
            }
        }
        let client = self.plugin.create_client_with_options(&values)?;
        compute_capabilities(&client)?;
        Ok(client)
    }
}

pub fn is_enabled() -> bool {
    !CUDA_RUNTIME_RLOCATION.is_empty()
}

fn write_allocator_options<'a>(
    values: &mut Vec<NamedValue<'a>>,
    name: &'a str,
    options: AllocatorOptions,
) {
    values.push(NamedValue::String {
        name: "allocator",
        value: name,
    });
    values.push(NamedValue::Bool {
        name: "preallocate",
        value: options.preallocate,
    });
    if options.memory_fraction > 0.0 {
        values.push(NamedValue::Float {
            name: "memory_fraction",
            value: options.memory_fraction,
        });
    }
    if options.collective_memory_size_bytes > 0 {
        values.push(NamedValue::Int64 {
            name: "collective_memory_size",
            value: options.collective_memory_size_bytes,
        });
    }
}

/// Returns every CUDA device capability, rejecting incomplete or malformed
/// device descriptions instead of silently choosing an unsupported code path.
pub fn compute_capabilities(client: &Client<'_>) -> Result<Vec<ComputeCapability>, Error> {
    let platform = client.platform_name()?;
    if !platform.eq_ignore_ascii_case("cuda") {
        return Err(Error::UnexpectedPlatform(platform));
    }
    let devices = client.devices()?;
    if devices.is_empty() {
        return Err(Error::NoCudaDevices);
    }
    devices
        .iter()
        .enumerate()
        .map(|(index, device)| {
            let value = device
                .string_attribute("compute_capability")?
                .ok_or(Error::MissingComputeCapability(index))?;
            ComputeCapability::parse(&value).ok_or(Error::InvalidComputeCapability {
                device_index: index,
                value,
            })
        })
        .collect()
}

fn has_nvidia_device() -> bool {
    ["/dev/nvidiactl", "/dev/dxg"]
        .iter()
        .any(|path| File::open(path).is_ok())
}

fn warn_for_external_cuda_libraries() {
    let Some(path) = std::env::var_os("LD_LIBRARY_PATH") else {
        return;
    };
    if path
        .to_string_lossy()
        .to_ascii_lowercase()
        .contains(CUDA_LIBRARY_PATH_FRAGMENT)
    {
        eprintln!(
            "warning: LD_LIBRARY_PATH contains {CUDA_LIBRARY_PATH_FRAGMENT}; external CUDA libraries can conflict with NML's hermetic runtime"
        );
    }
}

unsafe fn configure_xla_flags(runtime: &Path) {
    let existing = std::env::var_os("XLA_FLAGS").unwrap_or_default();
    let mut value = existing;
    if !value.is_empty() {
        value.push(" ");
    }
    value.push(format!(
        "--xla_gpu_cuda_data_dir={}",
        runtime.to_string_lossy()
    ));
    // SAFETY: this function inherits Runtime::load's process-wide exclusion.
    unsafe { std::env::set_var("XLA_FLAGS", value) };
}

fn use_packaged_driver(runtime: &Path) -> bool {
    let executable = runtime.join("bin/driver_compatibility");
    match Command::new(&executable)
        .current_dir(runtime)
        .status()
        .map(classify_driver_compatibility)
    {
        Ok(Some(required)) => required,
        Ok(None) => {
            eprintln!(
                "warning: CUDA driver compatibility executable returned an unexpected status; using the system driver"
            );
            false
        }
        Err(error) => {
            eprintln!(
                "warning: failed to execute {}: {error}; using the system driver",
                executable.display()
            );
            false
        }
    }
}

fn classify_driver_compatibility(status: ExitStatus) -> Option<bool> {
    match status.code()? {
        0 => Some(true),
        1 | 2 => Some(false),
        _ => None,
    }
}

fn load_compatibility_driver(runtime: &Path) -> Result<(), Error> {
    let path = runtime.join("lib/compat/libcuda.so.1");
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::InvalidRuntimePath(path.clone()))?;
    clear_dlerror();
    // SAFETY: c_path is terminated and RTLD_NOW is a valid loader flag. The
    // returned handle intentionally remains resident for the process lifetime,
    // matching ZML and the process-global CUDA driver model.
    let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return Err(Error::CompatibilityDriver {
            path,
            message: dlerror_message(),
        });
    }
    Ok(())
}

fn clear_dlerror() {
    // SAFETY: dlerror has no preconditions.
    let _ = unsafe { dlerror() };
}

fn dlerror_message() -> String {
    // SAFETY: a non-null result is a NUL-terminated thread-local string.
    let error = unsafe { dlerror() };
    if error.is_null() {
        "dynamic loader returned no detail".to_owned()
    } else {
        // SAFETY: checked non-null above.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
