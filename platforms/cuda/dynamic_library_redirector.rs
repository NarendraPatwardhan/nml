//! Interposes unversioned CUDA `dlopen` requests made by the PJRT plugin.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem::MaybeUninit;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

#[repr(C)]
struct DlInfo {
    filename: *const c_char,
    base: *mut c_void,
    symbol_name: *const c_char,
    symbol_address: *mut c_void,
}

/// CUDA PJRT's `dlopen` calls are renamed to this symbol by patchelf.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nml_cuda_dlopen(filename: *const c_char, flags: c_int) -> *mut c_void {
    if filename.is_null() {
        // SAFETY: forwarding dlopen(NULL, flags) preserves POSIX semantics.
        return unsafe { dlopen(filename, flags) };
    }

    // SAFETY: CUDA passes a NUL-terminated library name to dlopen.
    let requested = unsafe { CStr::from_ptr(filename) };
    let basename = Path::new(std::ffi::OsStr::from_bytes(requested.to_bytes()))
        .file_name()
        .and_then(|value| value.to_str());
    let replacement = basename.and_then(versioned_library_name);

    let rewritten = replacement
        .and_then(|name| own_library_directory().map(|directory| directory.join(name)))
        .and_then(|path| CString::new(path.as_os_str().as_bytes()).ok());
    let forwarded = rewritten.as_ref().map_or(filename, |path| path.as_ptr());
    // SAFETY: forwarded is either the caller's valid C string or a live CString.
    unsafe { dlopen(forwarded, flags) }
}

fn versioned_library_name(name: &str) -> Option<&'static str> {
    match name {
        "libcublas.so" => Some("libcublas.so.13"),
        "libcublasLt.so" => Some("libcublasLt.so.13"),
        "libcudart.so" => Some("libcudart.so.13"),
        "libcudnn.so" => Some("libcudnn.so.9"),
        "libcufft.so" => Some("libcufft.so.12"),
        "libcupti.so" => Some("libcupti.so.13"),
        "libcusolver.so" => Some("libcusolver.so.12"),
        "libcusparse.so" => Some("libcusparse.so.12"),
        "libnccl.so" => Some("libnccl.so.2"),
        _ => None,
    }
}

fn own_library_directory() -> Option<PathBuf> {
    let mut info = MaybeUninit::<DlInfo>::zeroed();
    // SAFETY: dladdr only reads the supplied code address and initializes info
    // on success. The exported function is guaranteed to live in this DSO.
    let found = unsafe {
        dladdr(
            nml_cuda_dlopen as *const () as *const c_void,
            info.as_mut_ptr(),
        )
    };
    if found == 0 {
        return None;
    }
    // SAFETY: dladdr succeeded and initialized info.
    let info = unsafe { info.assume_init() };
    if info.filename.is_null() {
        return None;
    }
    // SAFETY: dladdr returns a NUL-terminated path owned by the loader.
    let path = unsafe { CStr::from_ptr(info.filename) };
    Path::new(std::ffi::OsStr::from_bytes(path.to_bytes()))
        .parent()
        .map(Path::to_owned)
}

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dladdr(address: *const c_void, info: *mut DlInfo) -> c_int;
}
