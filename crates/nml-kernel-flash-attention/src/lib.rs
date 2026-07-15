//! Process-lifetime typed-FFI registration for NML's FlashAttention adapter.
//!
//! The public tensor API never sees these records. XLA owns buffers and the
//! CUDA stream; this crate validates the typed call frame, borrows those
//! resources for one launch, and translates every adapter failure back into an
//! XLA status.

#![forbid(unsafe_op_in_unsafe_fn)]

use nml_pjrt::{GpuCustomCallApi, GpuCustomCallHandler, GpuCustomCallHandlers, GpuCustomCalls};
use nml_pjrt_sys as sys;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem::{offset_of, size_of, zeroed};
use std::ptr::{NonNull, null_mut};
use std::sync::{Mutex, OnceLock};

const FA2_FORWARD_TARGET: &str = "nml.flash_attention_2.forward";
const FA3_FORWARD_TARGET: &str = "nml.flash_attention_3.forward";
const FA2_PAGED_FORWARD_TARGET: &str = "nml.flash_attention_2.paged";
const FA3_PAGED_FORWARD_TARGET: &str = "nml.flash_attention_3.paged";

static REGISTERED: OnceLock<Mutex<HashSet<(usize, &'static str)>>> = OnceLock::new();

// Bindgen deliberately leaves XLA_FFI_Api opaque: handlers receive a table
// owned by the loaded XLA runtime, not a table NML may construct.  The pinned
// C ABI nevertheless requires handlers to read its size-checked prefix.  Keep
// the smallest prefix that reaches Error_Create and Stream_Get, and reject an
// older/truncated table before reading either function pointer.
#[repr(C)]
struct FfiApiPrefix {
    struct_size: usize,
    extension_start: *mut sys::XLA_FFI_Extension_Base,
    api_version: sys::XLA_FFI_Api_Version,
    internal_api: *const sys::XLA_FFI_InternalApi,
    error_create: Option<
        unsafe extern "C" fn(*mut sys::XLA_FFI_Error_Create_Args) -> *mut sys::XLA_FFI_Error,
    >,
    error_get_message: Option<unsafe extern "C" fn(*mut sys::XLA_FFI_Error_GetMessage_Args)>,
    error_destroy: Option<unsafe extern "C" fn(*mut sys::XLA_FFI_Error_Destroy_Args)>,
    handler_register: *const c_void,
    stream_get:
        Option<unsafe extern "C" fn(*mut sys::XLA_FFI_Stream_Get_Args) -> *mut sys::XLA_FFI_Error>,
}

const COMMAND_BUFFER_COMPATIBLE: sys::XLA_FFI_Handler_Traits = 1;

/// Registers each FlashAttention forward handler once per loaded CUDA PJRT API
/// table. A failed handler is not recorded, so initialization reports the
/// original error and a later controlled retry remains possible.
pub fn register(custom_calls: &GpuCustomCalls) -> Result<(), nml_pjrt::Error> {
    let registered = REGISTERED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut registered = registered.lock().unwrap_or_else(|error| error.into_inner());
    let identity = custom_calls.plugin_identity();
    for (target, handler) in [
        (FA2_FORWARD_TARGET, fa2_forward as *const ()),
        (FA3_FORWARD_TARGET, fa3_forward as *const ()),
        (FA2_PAGED_FORWARD_TARGET, fa2_paged_forward as *const ()),
        (FA3_PAGED_FORWARD_TARGET, fa3_paged_forward as *const ()),
    ] {
        let key = (identity, target);
        if registered.contains(&key) {
            continue;
        }
        let address =
            NonNull::new(handler as *mut c_void).expect("a static function has a non-null address");
        // SAFETY: each function has the pinned XLA typed-FFI execute signature
        // and static lifetime. The linked adapters and handlers live for the
        // whole process, as required by PJRT's registration extension.
        unsafe {
            custom_calls.register(
                target,
                GpuCustomCallApi::Typed,
                GpuCustomCallHandlers {
                    instantiate: None,
                    prepare: None,
                    initialize: None,
                    execute: GpuCustomCallHandler::from_address(address),
                },
            )
        }?;
        registered.insert(key);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Version {
    Two,
    Three,
}

#[repr(C)]
struct ForwardRequest {
    struct_size: usize,
    query: *mut c_void,
    key: *mut c_void,
    value: *mut c_void,
    output: *mut c_void,
    softmax_lse: *mut c_void,
    workspace: *mut c_void,
    stream: *mut c_void,
    query_batch_stride: i64,
    query_row_stride: i64,
    query_head_stride: i64,
    key_batch_stride: i64,
    key_row_stride: i64,
    key_head_stride: i64,
    value_batch_stride: i64,
    value_row_stride: i64,
    value_head_stride: i64,
    output_batch_stride: i64,
    output_row_stride: i64,
    output_head_stride: i64,
    batch_size: i32,
    query_length: i32,
    key_length: i32,
    query_heads: i32,
    key_value_heads: i32,
    head_dimension: i32,
    sliding_window: i32,
    scale: f32,
    dtype: c_int,
    causal: u8,
}

#[repr(C)]
struct PagedForwardRequest {
    struct_size: usize,
    query: *mut c_void,
    key_cache: *mut c_void,
    value_cache: *mut c_void,
    page_table: *mut c_void,
    sequence_lengths: *mut c_void,
    output: *mut c_void,
    softmax_lse: *mut c_void,
    workspace: *mut c_void,
    stream: *mut c_void,
    query_batch_stride: i64,
    query_row_stride: i64,
    query_head_stride: i64,
    cache_page_stride: i64,
    cache_row_stride: i64,
    cache_head_stride: i64,
    output_batch_stride: i64,
    output_row_stride: i64,
    output_head_stride: i64,
    page_table_batch_stride: i64,
    num_pages: i32,
    page_size: i32,
    max_pages_per_sequence: i32,
    batch_size: i32,
    query_length: i32,
    query_heads: i32,
    key_value_heads: i32,
    head_dimension: i32,
    sliding_window: i32,
    scale: f32,
    dtype: c_int,
    causal: u8,
}

unsafe extern "C" {
    fn nml_flash_attention_2_forward(
        request: *const ForwardRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
    fn nml_flash_attention_3_forward(
        request: *const ForwardRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
    fn nml_flash_attention_2_paged_forward(
        request: *const PagedForwardRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
    fn nml_flash_attention_3_paged_forward(
        request: *const PagedForwardRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
}

unsafe extern "C" fn fa2_forward(raw: *mut sys::XLA_FFI_CallFrame) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, Version::Two, launch_dense) }
}

unsafe extern "C" fn fa3_forward(raw: *mut sys::XLA_FFI_CallFrame) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, Version::Three, launch_dense) }
}

unsafe extern "C" fn fa2_paged_forward(
    raw: *mut sys::XLA_FFI_CallFrame,
) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, Version::Two, launch_paged) }
}

unsafe extern "C" fn fa3_paged_forward(
    raw: *mut sys::XLA_FFI_CallFrame,
) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, Version::Three, launch_paged) }
}

type Launch = unsafe fn(&mut sys::XLA_FFI_CallFrame, Version) -> Result<(), HandlerFailure>;

unsafe fn execute(
    raw: *mut sys::XLA_FFI_CallFrame,
    version: Version,
    launch: Launch,
) -> *mut sys::XLA_FFI_Error {
    let Some(frame) = NonNull::new(raw) else {
        return null_mut();
    };
    // SAFETY: XLA calls the registered function with a live call frame.
    let frame = unsafe { &mut *frame.as_ptr() };
    if metadata_query(frame) {
        return null_mut();
    }
    match unsafe { launch(frame, version) } {
        Ok(()) => null_mut(),
        Err((code, message)) => ffi_error(frame, code, &message),
    }
}

fn metadata_query(frame: &mut sys::XLA_FFI_CallFrame) -> bool {
    let mut current = frame.extension_start;
    while let Some(extension) = NonNull::new(current) {
        // SAFETY: every extension begins with `struct_size`; read only that
        // word until it proves the complete base record is available.
        let actual = unsafe { extension.cast::<usize>().as_ptr().read() };
        if actual < sys::XLA_FFI_Extension_Base_STRUCT_SIZE as usize {
            return false;
        }
        // SAFETY: the size check above covers type and next.
        let extension = unsafe { extension.as_ref() };
        if extension.type_ != sys::XLA_FFI_Extension_Type_XLA_FFI_Extension_Metadata {
            current = extension.next;
            continue;
        }
        if actual < sys::XLA_FFI_Metadata_Extension_STRUCT_SIZE as usize {
            return true;
        }
        let metadata_extension = current.cast::<sys::XLA_FFI_Metadata_Extension>();
        // SAFETY: the matching type and size identify the metadata extension.
        let metadata_extension = unsafe { &mut *metadata_extension };
        let Some(metadata) = NonNull::new(metadata_extension.metadata) else {
            return true;
        };
        // SAFETY: metadata begins with a readable size word.
        let actual = unsafe { metadata.cast::<usize>().as_ptr().read() };
        if actual < sys::XLA_FFI_Metadata_STRUCT_SIZE as usize {
            return true;
        }
        // SAFETY: XLA supplies writable metadata storage for the registration
        // query and owns it beyond this call.
        let metadata = unsafe { &mut *metadata.as_ptr() };
        metadata.api_version.struct_size = sys::XLA_FFI_Api_Version_STRUCT_SIZE as usize;
        metadata.api_version.extension_start = null_mut();
        metadata.api_version.major_version = sys::XLA_FFI_API_MAJOR as i32;
        metadata.api_version.minor_version = sys::XLA_FFI_API_MINOR as i32;
        metadata.traits = COMMAND_BUFFER_COMPATIBLE;
        return true;
    }
    false
}

type HandlerFailure = (sys::XLA_FFI_Error_Code, String);

unsafe fn launch_dense(
    frame: &mut sys::XLA_FFI_CallFrame,
    version: Version,
) -> Result<(), HandlerFailure> {
    require_execute_frame(frame)?;
    let query = unsafe { argument(frame, 0, 3)? };
    let key = unsafe { argument(frame, 1, 3)? };
    let value = unsafe { argument(frame, 2, 3)? };
    let result_count = result_count(version);
    let output = unsafe { result(frame, 0, result_count)? };
    let softmax_lse = unsafe { result(frame, 1, result_count)? };
    let workspace = match version {
        Version::Two => None,
        Version::Three => Some(unsafe { result(frame, 2, result_count)? }),
    };

    let query_dims = dimensions(query)?;
    let key_dims = dimensions(key)?;
    let value_dims = dimensions(value)?;
    let output_dims = dimensions(output)?;
    let lse_dims = dimensions(softmax_lse)?;
    if query_dims.len() != 4 || key_dims.len() != 4 || value_dims.len() != 4 {
        return Err(invalid("FlashAttention Q, K, and V must be rank four"));
    }
    if key_dims != value_dims || query_dims != output_dims {
        return Err(invalid("FlashAttention buffer shapes are inconsistent"));
    }
    let [batch, query_length, query_heads, head_dimension] = query_dims else {
        unreachable!()
    };
    let [key_batch, key_length, key_value_heads, key_head_dimension] = key_dims else {
        unreachable!()
    };
    if batch != key_batch || head_dimension != key_head_dimension {
        return Err(invalid(
            "FlashAttention batch or head dimensions do not match",
        ));
    }
    if lse_dims != [*batch, *query_heads, *query_length] {
        return Err(invalid("FlashAttention LSE workspace has the wrong shape"));
    }
    let dtype = attention_dtype(query)?;
    if key.dtype != query.dtype || value.dtype != query.dtype || output.dtype != query.dtype {
        return Err(invalid(
            "FlashAttention Q, K, V, and output dtypes must match",
        ));
    }
    validate_auxiliary_results(softmax_lse, workspace)?;

    let scale = unsafe { scalar_f32(frame, "scale")? };
    let causal = unsafe { scalar_bool(frame, "causal")? };
    let sliding_window = unsafe { scalar_i32(frame, "sliding_window")? };
    let stream = unsafe { stream(frame)? };
    let q_row = checked_product(*query_heads, *head_dimension, "query row")?;
    let q_batch = checked_product(*query_length, q_row, "query batch")?;
    let kv_row = checked_product(*key_value_heads, *head_dimension, "key/value row")?;
    let kv_batch = checked_product(*key_length, kv_row, "key/value batch")?;

    let request = ForwardRequest {
        struct_size: size_of::<ForwardRequest>(),
        query: query.data,
        key: key.data,
        value: value.data,
        output: output.data,
        softmax_lse: softmax_lse.data,
        workspace: workspace.map_or(null_mut(), |workspace| workspace.data),
        stream,
        query_batch_stride: q_batch,
        query_row_stride: q_row,
        query_head_stride: *head_dimension,
        key_batch_stride: kv_batch,
        key_row_stride: kv_row,
        key_head_stride: *head_dimension,
        value_batch_stride: kv_batch,
        value_row_stride: kv_row,
        value_head_stride: *head_dimension,
        output_batch_stride: q_batch,
        output_row_stride: q_row,
        output_head_stride: *head_dimension,
        batch_size: to_i32(*batch, "batch size")?,
        query_length: to_i32(*query_length, "query length")?,
        key_length: to_i32(*key_length, "key length")?,
        query_heads: to_i32(*query_heads, "query head count")?,
        key_value_heads: to_i32(*key_value_heads, "KV head count")?,
        head_dimension: to_i32(*head_dimension, "head dimension")?,
        sliding_window,
        scale,
        dtype,
        causal: u8::from(causal),
    };
    let mut message = [0 as c_char; 512];
    // SAFETY: the request and diagnostic storage remain live for the complete
    // C call; every device pointer and stream is borrowed from this frame.
    let status = match version {
        Version::Two => unsafe {
            nml_flash_attention_2_forward(&request, message.as_mut_ptr(), message.len())
        },
        Version::Three => unsafe {
            nml_flash_attention_3_forward(&request, message.as_mut_ptr(), message.len())
        },
    };
    adapter_result(status, &message)
}

unsafe fn launch_paged(
    frame: &mut sys::XLA_FFI_CallFrame,
    version: Version,
) -> Result<(), HandlerFailure> {
    require_execute_frame(frame)?;
    let query = unsafe { argument(frame, 0, 5)? };
    let key_cache = unsafe { argument(frame, 1, 5)? };
    let value_cache = unsafe { argument(frame, 2, 5)? };
    let page_table = unsafe { argument(frame, 3, 5)? };
    let sequence_lengths = unsafe { argument(frame, 4, 5)? };
    let result_count = result_count(version);
    let output = unsafe { result(frame, 0, result_count)? };
    let softmax_lse = unsafe { result(frame, 1, result_count)? };
    let workspace = match version {
        Version::Two => None,
        Version::Three => Some(unsafe { result(frame, 2, result_count)? }),
    };

    let query_dims = dimensions(query)?;
    let key_dims = dimensions(key_cache)?;
    let value_dims = dimensions(value_cache)?;
    let page_table_dims = dimensions(page_table)?;
    let sequence_dims = dimensions(sequence_lengths)?;
    let output_dims = dimensions(output)?;
    let lse_dims = dimensions(softmax_lse)?;
    if query_dims.len() != 4 || key_dims.len() != 4 || value_dims.len() != 4 {
        return Err(invalid(
            "paged FlashAttention query and KV caches must be rank four",
        ));
    }
    if page_table_dims.len() != 2 || sequence_dims.len() != 1 {
        return Err(invalid(
            "paged FlashAttention page table and sequence lengths must be rank two and one",
        ));
    }
    if key_dims != value_dims || query_dims != output_dims {
        return Err(invalid(
            "paged FlashAttention buffer shapes are inconsistent",
        ));
    }
    let [batch, query_length, query_heads, head_dimension] = query_dims else {
        unreachable!()
    };
    let [num_pages, page_size, key_value_heads, cache_head_dimension] = key_dims else {
        unreachable!()
    };
    let [table_batch, max_pages_per_sequence] = page_table_dims else {
        unreachable!()
    };
    if batch != table_batch || sequence_dims != [*batch] || head_dimension != cache_head_dimension {
        return Err(invalid(
            "paged FlashAttention batch or head dimensions do not match",
        ));
    }
    if lse_dims != [*batch, *query_heads, *query_length] {
        return Err(invalid(
            "paged FlashAttention LSE workspace has the wrong shape",
        ));
    }
    let dtype = attention_dtype(query)?;
    if key_cache.dtype != query.dtype
        || value_cache.dtype != query.dtype
        || output.dtype != query.dtype
    {
        return Err(invalid(
            "paged FlashAttention query, KV caches, and output dtypes must match",
        ));
    }
    if page_table.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_S32
        || sequence_lengths.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_S32
    {
        return Err(invalid(
            "paged FlashAttention page table and sequence lengths must be I32",
        ));
    }
    validate_auxiliary_results(softmax_lse, workspace)?;

    let scale = unsafe { scalar_f32(frame, "scale")? };
    let causal = unsafe { scalar_bool(frame, "causal")? };
    let sliding_window = unsafe { scalar_i32(frame, "sliding_window")? };
    let stream = unsafe { stream(frame)? };
    let q_row = checked_product(*query_heads, *head_dimension, "query row")?;
    let q_batch = checked_product(*query_length, q_row, "query batch")?;
    let cache_row = checked_product(*key_value_heads, *head_dimension, "cache row")?;
    let cache_page = checked_product(*page_size, cache_row, "cache page")?;

    let request = PagedForwardRequest {
        struct_size: size_of::<PagedForwardRequest>(),
        query: query.data,
        key_cache: key_cache.data,
        value_cache: value_cache.data,
        page_table: page_table.data,
        sequence_lengths: sequence_lengths.data,
        output: output.data,
        softmax_lse: softmax_lse.data,
        workspace: workspace.map_or(null_mut(), |workspace| workspace.data),
        stream,
        query_batch_stride: q_batch,
        query_row_stride: q_row,
        query_head_stride: *head_dimension,
        cache_page_stride: cache_page,
        cache_row_stride: cache_row,
        cache_head_stride: *head_dimension,
        output_batch_stride: q_batch,
        output_row_stride: q_row,
        output_head_stride: *head_dimension,
        page_table_batch_stride: *max_pages_per_sequence,
        num_pages: to_i32(*num_pages, "page count")?,
        page_size: to_i32(*page_size, "page size")?,
        max_pages_per_sequence: to_i32(*max_pages_per_sequence, "maximum pages per sequence")?,
        batch_size: to_i32(*batch, "batch size")?,
        query_length: to_i32(*query_length, "query length")?,
        query_heads: to_i32(*query_heads, "query head count")?,
        key_value_heads: to_i32(*key_value_heads, "KV head count")?,
        head_dimension: to_i32(*head_dimension, "head dimension")?,
        sliding_window,
        scale,
        dtype,
        causal: u8::from(causal),
    };
    let mut message = [0 as c_char; 512];
    // SAFETY: the ABI record and diagnostic storage outlive the complete C
    // call, and every device address is borrowed from the current FFI frame.
    let status = match version {
        Version::Two => unsafe {
            nml_flash_attention_2_paged_forward(&request, message.as_mut_ptr(), message.len())
        },
        Version::Three => unsafe {
            nml_flash_attention_3_paged_forward(&request, message.as_mut_ptr(), message.len())
        },
    };
    adapter_result(status, &message)
}

fn require_execute_frame(frame: &sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_struct(
        frame.struct_size,
        sys::XLA_FFI_CallFrame_STRUCT_SIZE as usize,
        "call frame",
    )?;
    if frame.stage != sys::XLA_FFI_ExecutionStage_XLA_FFI_ExecutionStage_EXECUTE {
        return Err(invalid(
            "FlashAttention handler was called outside execute stage",
        ));
    }
    Ok(())
}

fn result_count(version: Version) -> usize {
    match version {
        Version::Two => 2,
        Version::Three => 3,
    }
}

fn attention_dtype(buffer: &sys::XLA_FFI_Buffer) -> Result<c_int, HandlerFailure> {
    match buffer.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 => Ok(1),
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16 => Ok(2),
        _ => Err(invalid("FlashAttention supports FP16 and BF16 only")),
    }
}

fn validate_auxiliary_results(
    softmax_lse: &sys::XLA_FFI_Buffer,
    workspace: Option<&sys::XLA_FFI_Buffer>,
) -> Result<(), HandlerFailure> {
    if softmax_lse.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32 {
        return Err(invalid("FlashAttention LSE workspace must be FP32"));
    }
    if let Some(workspace) = workspace {
        if workspace.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_S32
            || dimensions(workspace)? != [1]
        {
            return Err(invalid(
                "FlashAttention 3 scheduler workspace must be I32[1]",
            ));
        }
    }
    Ok(())
}

fn checked_product(left: i64, right: i64, name: &str) -> Result<i64, HandlerFailure> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(&format!("{name} stride overflows I64")))
}

fn to_i32(value: i64, name: &str) -> Result<i32, HandlerFailure> {
    i32::try_from(value).map_err(|_| invalid(&format!("{name} exceeds I32")))
}

fn adapter_result(status: i32, message: &[c_char]) -> Result<(), HandlerFailure> {
    if status == 0 {
        return Ok(());
    }
    let bytes = message
        .iter()
        .map(|byte| *byte as u8)
        .take_while(|byte| *byte != 0)
        .collect::<Vec<_>>();
    let message = String::from_utf8_lossy(&bytes).into_owned();
    Err((
        if status == 1 {
            sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INVALID_ARGUMENT
        } else {
            sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INTERNAL
        },
        message,
    ))
}

unsafe fn argument<'a>(
    frame: &'a sys::XLA_FFI_CallFrame,
    index: usize,
    expected: usize,
) -> Result<&'a sys::XLA_FFI_Buffer, HandlerFailure> {
    require_struct(
        frame.args.struct_size,
        sys::XLA_FFI_Args_STRUCT_SIZE as usize,
        "argument list",
    )?;
    if frame.args.size != expected as i64 || frame.args.types.is_null() || frame.args.args.is_null()
    {
        return Err(invalid(&format!(
            "FlashAttention expects exactly {expected} buffer arguments"
        )));
    }
    // SAFETY: sizes and base pointers were validated above.
    let kind = unsafe { *frame.args.types.add(index) };
    if kind != sys::XLA_FFI_ArgType_XLA_FFI_ArgType_BUFFER {
        return Err(invalid("FlashAttention arguments must be buffers"));
    }
    // SAFETY: XLA stores one live XLA_FFI_Buffer pointer per buffer argument.
    let pointer = unsafe { *frame.args.args.add(index) }.cast::<sys::XLA_FFI_Buffer>();
    let buffer = unsafe { pointer.as_ref() }
        .ok_or_else(|| invalid("FlashAttention received a null argument buffer"))?;
    require_struct(
        buffer.struct_size,
        sys::XLA_FFI_Buffer_STRUCT_SIZE as usize,
        "argument buffer",
    )?;
    Ok(buffer)
}

unsafe fn result<'a>(
    frame: &'a sys::XLA_FFI_CallFrame,
    index: usize,
    expected: usize,
) -> Result<&'a sys::XLA_FFI_Buffer, HandlerFailure> {
    require_struct(
        frame.rets.struct_size,
        sys::XLA_FFI_Rets_STRUCT_SIZE as usize,
        "result list",
    )?;
    if frame.rets.size != expected as i64 || frame.rets.types.is_null() || frame.rets.rets.is_null()
    {
        return Err(invalid(
            "FlashAttention received the wrong number of result buffers",
        ));
    }
    // SAFETY: sizes and base pointers were validated above.
    let kind = unsafe { *frame.rets.types.add(index) };
    if kind != sys::XLA_FFI_RetType_XLA_FFI_RetType_BUFFER {
        return Err(invalid("FlashAttention results must be buffers"));
    }
    // SAFETY: XLA stores one live XLA_FFI_Buffer pointer per buffer result.
    let pointer = unsafe { *frame.rets.rets.add(index) }.cast::<sys::XLA_FFI_Buffer>();
    let buffer = unsafe { pointer.as_ref() }
        .ok_or_else(|| invalid("FlashAttention received a null result buffer"))?;
    require_struct(
        buffer.struct_size,
        sys::XLA_FFI_Buffer_STRUCT_SIZE as usize,
        "result buffer",
    )?;
    Ok(buffer)
}

fn dimensions(buffer: &sys::XLA_FFI_Buffer) -> Result<&[i64], HandlerFailure> {
    if buffer.rank < 0 || (buffer.rank != 0 && buffer.dims.is_null()) || buffer.data.is_null() {
        return Err(invalid("FlashAttention received malformed buffer metadata"));
    }
    // SAFETY: XLA guarantees `dims` has `rank` entries for the duration of the
    // call frame; the checks above cover the zero-rank and null cases.
    Ok(unsafe { std::slice::from_raw_parts(buffer.dims, buffer.rank as usize) })
}

unsafe fn scalar_f32(frame: &sys::XLA_FFI_CallFrame, name: &str) -> Result<f32, HandlerFailure> {
    unsafe {
        scalar(frame, name, sys::XLA_FFI_DataType_XLA_FFI_DataType_F32)
            .map(|value| *(value as *const f32))
    }
}

unsafe fn scalar_i32(frame: &sys::XLA_FFI_CallFrame, name: &str) -> Result<i32, HandlerFailure> {
    unsafe {
        scalar(frame, name, sys::XLA_FFI_DataType_XLA_FFI_DataType_S32)
            .map(|value| *(value as *const i32))
    }
}

unsafe fn scalar_bool(frame: &sys::XLA_FFI_CallFrame, name: &str) -> Result<bool, HandlerFailure> {
    unsafe {
        scalar(frame, name, sys::XLA_FFI_DataType_XLA_FFI_DataType_PRED)
            .map(|value| *(value as *const u8) != 0)
    }
}

unsafe fn scalar(
    frame: &sys::XLA_FFI_CallFrame,
    expected_name: &str,
    expected_dtype: sys::XLA_FFI_DataType,
) -> Result<*const c_void, HandlerFailure> {
    require_struct(
        frame.attrs.struct_size,
        sys::XLA_FFI_Attrs_STRUCT_SIZE as usize,
        "attribute list",
    )?;
    if frame.attrs.size < 0
        || frame.attrs.types.is_null()
        || frame.attrs.names.is_null()
        || frame.attrs.attrs.is_null()
    {
        return Err(invalid("FlashAttention received malformed attributes"));
    }
    for index in 0..frame.attrs.size as usize {
        // SAFETY: all three arrays contain `size` entries.
        let name = unsafe { *frame.attrs.names.add(index) };
        let Some(name) = (unsafe { name.as_ref() }) else {
            continue;
        };
        if name.ptr.is_null() {
            continue;
        }
        // SAFETY: XLA owns the length-delimited attribute name for this call.
        let bytes = unsafe { std::slice::from_raw_parts(name.ptr.cast::<u8>(), name.len) };
        if bytes != expected_name.as_bytes() {
            continue;
        }
        // SAFETY: the type and attribute arrays share the same index.
        if unsafe { *frame.attrs.types.add(index) } != sys::XLA_FFI_AttrType_XLA_FFI_AttrType_SCALAR
        {
            return Err(invalid("FlashAttention attribute has the wrong kind"));
        }
        let pointer = unsafe { *frame.attrs.attrs.add(index) }.cast::<sys::XLA_FFI_Scalar>();
        let scalar = unsafe { pointer.as_ref() }
            .ok_or_else(|| invalid("FlashAttention scalar attribute is null"))?;
        if scalar.dtype != expected_dtype || scalar.value.is_null() {
            return Err(invalid(
                "FlashAttention scalar attribute has the wrong dtype",
            ));
        }
        return Ok(scalar.value.cast_const());
    }
    Err(invalid(&format!(
        "FlashAttention is missing attribute {expected_name:?}"
    )))
}

unsafe fn stream(frame: &sys::XLA_FFI_CallFrame) -> Result<*mut c_void, HandlerFailure> {
    let api = unsafe { ffi_api(frame)? };
    let get = api
        .stream_get
        .ok_or_else(|| invalid("XLA FFI API does not expose the CUDA stream"))?;
    // SAFETY: zero is the defined absence value for the extension and output.
    let mut args: sys::XLA_FFI_Stream_Get_Args = unsafe { zeroed() };
    args.struct_size = sys::XLA_FFI_Stream_Get_Args_STRUCT_SIZE as usize;
    args.ctx = frame.ctx;
    // SAFETY: `get` belongs to this frame's API and `args` is complete.
    let error = unsafe { get(&mut args) };
    if !error.is_null() {
        let message = unsafe { take_ffi_error(api, error) };
        return Err((
            sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INTERNAL,
            format!("XLA FFI could not provide a CUDA stream: {message}"),
        ));
    }
    if args.stream.is_null() {
        return Err(invalid("XLA FFI could not provide a CUDA stream"));
    }
    Ok(args.stream)
}

unsafe fn take_ffi_error(api: &FfiApiPrefix, error: *mut sys::XLA_FFI_Error) -> String {
    let message = if let Some(get_message) = api.error_get_message {
        // SAFETY: zero initializes the optional extension and output pointer.
        let mut args: sys::XLA_FFI_Error_GetMessage_Args = unsafe { zeroed() };
        args.struct_size = sys::XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE as usize;
        args.error = error;
        // SAFETY: the error remains owned by this function until it is
        // destroyed below and the API initializes a borrowed C string.
        unsafe { get_message(&mut args) };
        if args.message.is_null() {
            "unknown XLA FFI error".to_owned()
        } else {
            // SAFETY: the message belongs to the live error object.
            unsafe { CStr::from_ptr(args.message) }
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "unknown XLA FFI error".to_owned()
    };
    if let Some(destroy) = api.error_destroy {
        // SAFETY: zero initializes the optional extension pointer.
        let mut args: sys::XLA_FFI_Error_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::XLA_FFI_Error_Destroy_Args_STRUCT_SIZE as usize;
        args.error = error;
        // SAFETY: ownership of the non-null error transfers to Destroy once.
        unsafe { destroy(&mut args) };
    }
    message
}

fn require_struct(actual: usize, required: usize, name: &str) -> Result<(), HandlerFailure> {
    if actual < required {
        return Err(invalid(&format!(
            "truncated XLA FFI {name}: expected {required} bytes, received {actual}"
        )));
    }
    Ok(())
}

fn invalid(message: &str) -> HandlerFailure {
    (
        sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INVALID_ARGUMENT,
        message.to_owned(),
    )
}

fn ffi_error(
    frame: &sys::XLA_FFI_CallFrame,
    code: sys::XLA_FFI_Error_Code,
    message: &str,
) -> *mut sys::XLA_FFI_Error {
    let Ok(api) = (unsafe { ffi_api(frame) }) else {
        return null_mut();
    };
    let Some(create) = api.error_create else {
        return null_mut();
    };
    let message = CString::new(message)
        .unwrap_or_else(|_| CString::new("FlashAttention failure contained a NUL byte").unwrap());
    // SAFETY: zero initializes the optional extension pointer.
    let mut args: sys::XLA_FFI_Error_Create_Args = unsafe { zeroed() };
    args.struct_size = sys::XLA_FFI_Error_Create_Args_STRUCT_SIZE as usize;
    args.message = message.as_ptr();
    args.errc = code;
    // SAFETY: the API copies the message while the CString is live and returns
    // an error object whose ownership transfers to XLA.
    unsafe { create(&mut args) }
}

unsafe fn ffi_api(frame: &sys::XLA_FFI_CallFrame) -> Result<&FfiApiPrefix, HandlerFailure> {
    let pointer = NonNull::new(frame.api.cast_mut())
        .ok_or_else(|| invalid("FlashAttention call frame has no FFI API"))?;
    // SAFETY: every XLA FFI API begins with `struct_size`; reading that first
    // word is valid even when the producer implements an older prefix.
    let actual = unsafe { pointer.cast::<usize>().as_ptr().read() };
    let required = offset_of!(FfiApiPrefix, stream_get) + size_of::<*const c_void>();
    require_struct(actual, required, "API table")?;
    // SAFETY: the size check proves the pinned prefix is present and the XLA
    // runtime owns the immutable table for at least the duration of the call.
    Ok(unsafe { pointer.cast::<FfiApiPrefix>().as_ref() })
}
