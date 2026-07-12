//! Determines whether the packaged CUDA compatibility driver can initialize.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::process::ExitCode;
use std::{env, mem, ptr};

const CUDA_SUCCESS: c_int = 0;
const CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: c_int = 803;
const CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: c_int = 804;
const RTLD_LAZY: c_int = 0x1;

type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
type CuGetErrorString = unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int;

#[repr(u8)]
enum CompatibilityResult {
    Compatible = 0,
    CompatNotSupportedOnDevice = 1,
    SystemDriverMismatch = 2,
    UnexpectedError = 3,
}

fn main() -> ExitCode {
    ExitCode::from(run() as u8)
}

fn run() -> CompatibilityResult {
    let Some(executable_dir) = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_owned))
    else {
        eprintln!("unable to determine the CUDA driver compatibility executable directory");
        return CompatibilityResult::UnexpectedError;
    };
    let library_path = executable_dir.join("../lib/compat/libcuda.so.1");
    let Ok(c_path) = CString::new(library_path.as_os_str().as_bytes()) else {
        eprintln!("CUDA compatibility library path contains a NUL byte");
        return CompatibilityResult::UnexpectedError;
    };

    clear_dlerror();
    // SAFETY: c_path is NUL-terminated and RTLD_LAZY is a valid dlopen flag.
    let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_LAZY) };
    if handle.is_null() {
        eprintln!(
            "unable to open {}: {}",
            library_path.display(),
            dlerror_message()
        );
        return CompatibilityResult::UnexpectedError;
    }

    // SAFETY: the pinned CUDA driver ABI defines both symbol signatures.
    let Some(cu_init): Option<CuInit> = (unsafe { symbol(handle, "cuInit") }) else {
        return CompatibilityResult::UnexpectedError;
    };
    // SAFETY: the pinned CUDA driver ABI defines both symbol signatures.
    let Some(cu_get_error_string): Option<CuGetErrorString> =
        (unsafe { symbol(handle, "cuGetErrorString") })
    else {
        return CompatibilityResult::UnexpectedError;
    };

    // SAFETY: cuInit accepts flags=0 and has no caller-owned pointer arguments.
    match unsafe { cu_init(0) } {
        CUDA_SUCCESS => CompatibilityResult::Compatible,
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => CompatibilityResult::SystemDriverMismatch,
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => {
            CompatibilityResult::CompatNotSupportedOnDevice
        }
        code => {
            let mut message = ptr::null();
            // SAFETY: message is a valid writable output pointer.
            if unsafe { cu_get_error_string(code, &mut message) } == CUDA_SUCCESS
                && !message.is_null()
            {
                // SAFETY: successful CUDA error strings are NUL-terminated and process-owned.
                eprintln!(
                    "cuInit returned {code}: {}",
                    unsafe { CStr::from_ptr(message) }.to_string_lossy()
                );
            } else {
                eprintln!("cuInit returned unexpected error {code}");
            }
            CompatibilityResult::UnexpectedError
        }
    }
}

unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &'static str) -> Option<T> {
    let c_name = CString::new(name).expect("static CUDA symbol contains no NUL");
    clear_dlerror();
    // SAFETY: handle is a successful dlopen result and c_name is terminated.
    let address = unsafe { dlsym(handle, c_name.as_ptr()) };
    if address.is_null() {
        eprintln!("missing CUDA symbol {name}: {}", dlerror_message());
        return None;
    }
    assert_eq!(mem::size_of::<T>(), mem::size_of::<*mut c_void>());
    // SAFETY: the caller specifies the symbol's ABI type and widths match.
    Some(unsafe { mem::transmute_copy(&address) })
}

fn clear_dlerror() {
    // SAFETY: dlerror has no preconditions.
    let _ = unsafe { dlerror() };
}

fn dlerror_message() -> String {
    // SAFETY: a non-null dlerror result is a NUL-terminated thread-local string.
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
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
