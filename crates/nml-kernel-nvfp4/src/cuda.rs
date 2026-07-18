//! Typed XLA FFI boundary for the dedicated SM75 compact-weight kernels.
//!
//! The handler owns validation and CUDA-stream extraction; the linked adapter
//! owns only device code and launch diagnostics. This keeps XLA ABI details out
//! of CUDA C++ while preserving one representation and semantic graph.

use nml_pjrt::{GpuCustomCallApi, GpuCustomCallHandler, GpuCustomCallHandlers, GpuCustomCalls};
use nml_pjrt_sys as sys;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::mem::{offset_of, size_of, zeroed};
use std::ptr::{NonNull, null_mut};
use std::sync::{Mutex, OnceLock};

const LINEAR_TARGET: &str = "nml.nvfp4.turing.linear";
const EMBEDDING_TARGET: &str = "nml.nvfp4.turing.embedding";
const EXPERT_GATE_UP_TARGET: &str = "nml.nvfp4.turing.expert_gate_up";
const EXPERT_DOWN_TARGET: &str = "nml.nvfp4.turing.expert_down";
const COMMAND_BUFFER_COMPATIBLE: sys::XLA_FFI_Handler_Traits = 1;

static REGISTERED: OnceLock<Mutex<HashSet<(usize, &'static str)>>> = OnceLock::new();

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

#[repr(C)]
struct LinearRequest {
    struct_size: usize,
    activation: *const c_void,
    payload: *const u8,
    block_scales: *const u8,
    global_scale: *const f32,
    bias: *const c_void,
    output: *mut c_void,
    stream: *mut c_void,
    rows: i64,
    outputs: i64,
    inputs: i64,
    dtype: c_int,
}

#[repr(C)]
struct EmbeddingRequest {
    struct_size: usize,
    indices: *const c_void,
    payload: *const u8,
    block_scales: *const u8,
    global_scale: *const f32,
    output: *mut c_void,
    stream: *mut c_void,
    rows: i64,
    vocabulary: i64,
    width: i64,
    dtype: c_int,
    indices_are_i64: u8,
}

#[repr(C)]
struct ExpertGateUpRequest {
    struct_size: usize,
    hidden: *const c_void,
    sorted_assignments: *const i32,
    block_experts: *const i32,
    payload: *const u8,
    block_scales: *const u8,
    global_scale: *const f32,
    bias: *const c_void,
    activated: *mut c_void,
    stream: *mut c_void,
    tokens: i64,
    assignments: i64,
    schedule_positions: i64,
    schedule_blocks: i64,
    hidden_size: i64,
    intermediate_size: i64,
    experts: i64,
    experts_per_token: i64,
    block_size: i32,
    dtype: c_int,
}

#[repr(C)]
struct ExpertDownRequest {
    struct_size: usize,
    activated: *const c_void,
    sorted_assignments: *const i32,
    block_experts: *const i32,
    payload: *const u8,
    block_scales: *const u8,
    global_scale: *const f32,
    bias: *const c_void,
    routing_weights: *const c_void,
    weighted_output: *mut c_void,
    stream: *mut c_void,
    assignments: i64,
    schedule_positions: i64,
    schedule_blocks: i64,
    intermediate_size: i64,
    hidden_size: i64,
    experts: i64,
    experts_per_token: i64,
    block_size: i32,
    dtype: c_int,
}

unsafe extern "C" {
    fn nml_nvfp4_turing_linear(
        request: *const LinearRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
    fn nml_nvfp4_turing_embedding(
        request: *const EmbeddingRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
    fn nml_nvfp4_turing_expert_gate_up(
        request: *const ExpertGateUpRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
    fn nml_nvfp4_turing_expert_down(
        request: *const ExpertDownRequest,
        error_message: *mut c_char,
        error_message_capacity: usize,
    ) -> i32;
}

type HandlerFailure = (sys::XLA_FFI_Error_Code, String);

/// Registers the SM75 handlers once for each loaded CUDA PJRT API table.
pub fn register_cuda(custom_calls: &GpuCustomCalls) -> Result<(), nml_pjrt::Error> {
    let registered = REGISTERED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut registered = registered.lock().unwrap_or_else(|error| error.into_inner());
    for (target, handler) in [
        (LINEAR_TARGET, turing_linear as *const ()),
        (EMBEDDING_TARGET, turing_embedding as *const ()),
        (EXPERT_GATE_UP_TARGET, turing_expert_gate_up as *const ()),
        (EXPERT_DOWN_TARGET, turing_expert_down as *const ()),
    ] {
        let key = (custom_calls.plugin_identity(), target);
        if registered.contains(&key) {
            continue;
        }
        let address =
            NonNull::new(handler as *mut c_void).expect("a static function has a non-null address");
        // SAFETY: both functions implement the pinned typed-XLA-FFI ABI and
        // remain linked for the process lifetime.
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

unsafe extern "C" fn turing_linear(raw: *mut sys::XLA_FFI_CallFrame) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, launch_linear) }
}

unsafe extern "C" fn turing_embedding(raw: *mut sys::XLA_FFI_CallFrame) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, launch_embedding) }
}

unsafe extern "C" fn turing_expert_gate_up(
    raw: *mut sys::XLA_FFI_CallFrame,
) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, launch_expert_gate_up) }
}

unsafe extern "C" fn turing_expert_down(
    raw: *mut sys::XLA_FFI_CallFrame,
) -> *mut sys::XLA_FFI_Error {
    unsafe { execute(raw, launch_expert_down) }
}

type Launch = unsafe fn(&mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure>;

unsafe fn execute(raw: *mut sys::XLA_FFI_CallFrame, launch: Launch) -> *mut sys::XLA_FFI_Error {
    let Some(frame) = NonNull::new(raw) else {
        return null_mut();
    };
    // SAFETY: XLA invokes a registered handler with a live mutable call frame.
    let frame = unsafe { &mut *frame.as_ptr() };
    if metadata_query(frame) {
        return null_mut();
    }
    match unsafe { launch(frame) } {
        Ok(()) => null_mut(),
        Err((code, message)) => ffi_error(frame, code, &message),
    }
}

unsafe fn launch_linear(frame: &mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_execute_frame(frame, "linear")?;
    let argument_count = match frame.args.size {
        4 => 4,
        5 => 5,
        _ => {
            return Err(invalid(
                "NVFP4 Turing linear expects four arguments without bias or five with bias",
            ));
        }
    };
    let activation = unsafe { argument(frame, 0, argument_count)? };
    let payload = unsafe { argument(frame, 1, argument_count)? };
    let scales = unsafe { argument(frame, 2, argument_count)? };
    let global = unsafe { argument(frame, 3, argument_count)? };
    let bias = if argument_count == 5 {
        Some(unsafe { argument(frame, 4, argument_count)? })
    } else {
        None
    };
    let output = unsafe { result(frame, 0, 1)? };

    let activation_dimensions = dimensions(activation)?;
    let payload_dimensions = dimensions(payload)?;
    let scale_dimensions = dimensions(scales)?;
    let output_dimensions = dimensions(output)?;
    if activation_dimensions.is_empty()
        || payload_dimensions.len() != 2
        || scale_dimensions.len() != 2
        || output_dimensions.len() != activation_dimensions.len()
    {
        return Err(invalid("NVFP4 Turing linear received invalid ranks"));
    }
    let inputs = *activation_dimensions.last().unwrap();
    let outputs = payload_dimensions[0];
    let packed_inputs = ceil_div_positive(inputs, 2)
        .ok_or_else(|| invalid("NVFP4 Turing linear packed width overflows"))?;
    let scale_inputs = ceil_div_positive(inputs, 16)
        .ok_or_else(|| invalid("NVFP4 Turing linear scale width overflows"))?;
    if inputs <= 0
        || outputs <= 0
        || payload_dimensions[1] != packed_inputs
        || scale_dimensions != [outputs, scale_inputs]
        || output_dimensions[..output_dimensions.len() - 1]
            != activation_dimensions[..activation_dimensions.len() - 1]
        || output_dimensions[output_dimensions.len() - 1] != outputs
    {
        return Err(invalid(
            "NVFP4 Turing linear component and output shapes are inconsistent",
        ));
    }
    if payload.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || scales.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || global.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32
        || !dimensions(global)?.is_empty()
        || output.dtype != activation.dtype
    {
        return Err(invalid("NVFP4 Turing linear dtypes are inconsistent"));
    }
    let dtype = activation_dtype(activation)?;
    if let Some(bias) = bias
        && (dimensions(bias)? != [outputs] || bias.dtype != activation.dtype)
    {
        return Err(invalid(
            "NVFP4 Turing linear bias must be a length-N activation vector",
        ));
    }
    let rows = activation_dimensions[..activation_dimensions.len() - 1]
        .iter()
        .try_fold(1_i64, |count, &dimension| {
            count
                .checked_mul(dimension)
                .filter(|value| *value > 0)
                .ok_or_else(|| invalid("NVFP4 Turing linear row extent overflows"))
        })?;
    let request = LinearRequest {
        struct_size: size_of::<LinearRequest>(),
        activation: activation.data.cast_const(),
        payload: payload.data.cast_const().cast(),
        block_scales: scales.data.cast_const().cast(),
        global_scale: global.data.cast_const().cast(),
        bias: bias.map_or(std::ptr::null(), |bias| bias.data.cast_const()),
        output: output.data,
        stream: unsafe { stream(frame)? },
        rows,
        outputs,
        inputs,
        dtype,
    };
    call_adapter("linear", |message| unsafe {
        nml_nvfp4_turing_linear(&request, message.as_mut_ptr(), message.len())
    })
}

unsafe fn launch_embedding(frame: &mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_execute_frame(frame, "embedding")?;
    let indices = unsafe { argument(frame, 0, 4)? };
    let payload = unsafe { argument(frame, 1, 4)? };
    let scales = unsafe { argument(frame, 2, 4)? };
    let global = unsafe { argument(frame, 3, 4)? };
    let output = unsafe { result(frame, 0, 1)? };
    let index_dimensions = dimensions(indices)?;
    let payload_dimensions = dimensions(payload)?;
    let scale_dimensions = dimensions(scales)?;
    let output_dimensions = dimensions(output)?;
    if payload_dimensions.len() != 2
        || scale_dimensions.len() != 2
        || output_dimensions.len() != index_dimensions.len() + 1
        || output_dimensions[..index_dimensions.len()] != *index_dimensions
    {
        return Err(invalid("NVFP4 Turing embedding received invalid ranks"));
    }
    let vocabulary = payload_dimensions[0];
    let width = *output_dimensions.last().unwrap();
    let packed_width = ceil_div_positive(width, 2)
        .ok_or_else(|| invalid("NVFP4 Turing embedding packed width overflows"))?;
    let scale_width = ceil_div_positive(width, 16)
        .ok_or_else(|| invalid("NVFP4 Turing embedding scale width overflows"))?;
    if vocabulary <= 0
        || width <= 0
        || payload_dimensions[1] != packed_width
        || scale_dimensions != [vocabulary, scale_width]
        || payload.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || scales.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || global.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32
        || !dimensions(global)?.is_empty()
    {
        return Err(invalid(
            "NVFP4 Turing embedding component shapes or dtypes are inconsistent",
        ));
    }
    let indices_are_i64 = match indices.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_S32 => 0,
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_S64 => 1,
        _ => {
            return Err(invalid(
                "NVFP4 Turing embedding requires I32 or I64 indices",
            ));
        }
    };
    let rows = index_dimensions
        .iter()
        .try_fold(1_i64, |count, &dimension| {
            count
                .checked_mul(dimension)
                .filter(|value| *value > 0)
                .ok_or_else(|| invalid("NVFP4 Turing embedding index extent overflows"))
        })?;
    let request = EmbeddingRequest {
        struct_size: size_of::<EmbeddingRequest>(),
        indices: indices.data.cast_const(),
        payload: payload.data.cast_const().cast(),
        block_scales: scales.data.cast_const().cast(),
        global_scale: global.data.cast_const().cast(),
        output: output.data,
        stream: unsafe { stream(frame)? },
        rows,
        vocabulary,
        width,
        dtype: activation_dtype(output)?,
        indices_are_i64,
    };
    call_adapter("embedding", |message| unsafe {
        nml_nvfp4_turing_embedding(&request, message.as_mut_ptr(), message.len())
    })
}

unsafe fn launch_expert_gate_up(frame: &mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_execute_frame(frame, "expert gate/up")?;
    let hidden = unsafe { argument(frame, 0, 7)? };
    let schedule = unsafe { argument(frame, 1, 7)? };
    let block_experts = unsafe { argument(frame, 2, 7)? };
    let payload = unsafe { argument(frame, 3, 7)? };
    let scales = unsafe { argument(frame, 4, 7)? };
    let global = unsafe { argument(frame, 5, 7)? };
    let bias = unsafe { argument(frame, 6, 7)? };
    let activated = unsafe { result(frame, 0, 1)? };
    let hidden_dims = dimensions(hidden)?;
    let schedule_dims = dimensions(schedule)?;
    let block_dims = dimensions(block_experts)?;
    let payload_dims = dimensions(payload)?;
    let scale_dims = dimensions(scales)?;
    let bias_dims = dimensions(bias)?;
    let activated_dims = dimensions(activated)?;
    if hidden_dims.len() != 2
        || schedule_dims.len() != 1
        || block_dims.len() != 1
        || payload_dims.len() != 3
        || scale_dims.len() != 3
        || bias_dims.len() != 2
        || activated_dims.len() != 2
    {
        return Err(invalid(
            "NVFP4 Turing expert gate/up received invalid ranks",
        ));
    }
    let tokens = hidden_dims[0];
    let hidden_size = hidden_dims[1];
    let experts = payload_dims[0];
    let assignments = activated_dims[0];
    let intermediate = activated_dims[1];
    let doubled = intermediate
        .checked_mul(2)
        .ok_or_else(|| invalid("NVFP4 expert intermediate width overflows"))?;
    let routes = exact_positive_division(assignments, tokens, "expert route count")?;
    if tokens <= 0
        || hidden_size <= 0
        || experts <= 0
        || intermediate <= 0
        || payload_dims != [experts, hidden_size, intermediate]
        || scale_dims
            != [
                experts,
                hidden_size,
                ceil_div_positive(doubled, 16)
                    .ok_or_else(|| invalid("NVFP4 gate/up scale width overflows"))?,
            ]
        || bias_dims != [experts, doubled]
        || schedule_dims[0]
            != block_dims[0]
                .checked_mul(16)
                .ok_or_else(|| invalid("NVFP4 expert schedule extent overflows"))?
    {
        return Err(invalid(
            "NVFP4 Turing expert gate/up component shapes are inconsistent",
        ));
    }
    require_expert_dtypes(
        hidden,
        schedule,
        block_experts,
        payload,
        scales,
        global,
        bias,
        activated,
    )?;
    let request = ExpertGateUpRequest {
        struct_size: size_of::<ExpertGateUpRequest>(),
        hidden: hidden.data.cast_const(),
        sorted_assignments: schedule.data.cast_const().cast(),
        block_experts: block_experts.data.cast_const().cast(),
        payload: payload.data.cast_const().cast(),
        block_scales: scales.data.cast_const().cast(),
        global_scale: global.data.cast_const().cast(),
        bias: bias.data.cast_const(),
        activated: activated.data,
        stream: unsafe { stream(frame)? },
        tokens,
        assignments,
        schedule_positions: schedule_dims[0],
        schedule_blocks: block_dims[0],
        hidden_size,
        intermediate_size: intermediate,
        experts,
        experts_per_token: routes,
        block_size: 16,
        dtype: activation_dtype(hidden)?,
    };
    call_adapter("expert gate/up", |message| unsafe {
        nml_nvfp4_turing_expert_gate_up(&request, message.as_mut_ptr(), message.len())
    })
}

unsafe fn launch_expert_down(frame: &mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_execute_frame(frame, "expert down")?;
    let activated = unsafe { argument(frame, 0, 8)? };
    let schedule = unsafe { argument(frame, 1, 8)? };
    let block_experts = unsafe { argument(frame, 2, 8)? };
    let payload = unsafe { argument(frame, 3, 8)? };
    let scales = unsafe { argument(frame, 4, 8)? };
    let global = unsafe { argument(frame, 5, 8)? };
    let bias = unsafe { argument(frame, 6, 8)? };
    let routing = unsafe { argument(frame, 7, 8)? };
    let weighted = unsafe { result(frame, 0, 1)? };
    let activated_dims = dimensions(activated)?;
    let schedule_dims = dimensions(schedule)?;
    let block_dims = dimensions(block_experts)?;
    let payload_dims = dimensions(payload)?;
    let scale_dims = dimensions(scales)?;
    let bias_dims = dimensions(bias)?;
    let routing_dims = dimensions(routing)?;
    let weighted_dims = dimensions(weighted)?;
    if activated_dims.len() != 2
        || schedule_dims.len() != 1
        || block_dims.len() != 1
        || payload_dims.len() != 3
        || scale_dims.len() != 3
        || bias_dims.len() != 2
        || routing_dims.len() != 2
        || weighted_dims.len() != 2
    {
        return Err(invalid("NVFP4 Turing expert down received invalid ranks"));
    }
    let assignments = activated_dims[0];
    let intermediate = activated_dims[1];
    let experts = payload_dims[0];
    let hidden_size = weighted_dims[1];
    let routes = routing_dims[1];
    let expected_assignments = routing_dims[0]
        .checked_mul(routes)
        .ok_or_else(|| invalid("NVFP4 expert assignment count overflows"))?;
    if assignments <= 0
        || intermediate <= 0
        || experts <= 0
        || hidden_size <= 0
        || routes <= 0
        || assignments != expected_assignments
        || weighted_dims != [assignments, hidden_size]
        || payload_dims
            != [
                experts,
                intermediate,
                ceil_div_positive(hidden_size, 2)
                    .ok_or_else(|| invalid("NVFP4 expert down packed width overflows"))?,
            ]
        || scale_dims
            != [
                experts,
                intermediate,
                ceil_div_positive(hidden_size, 16)
                    .ok_or_else(|| invalid("NVFP4 expert down scale width overflows"))?,
            ]
        || bias_dims != [experts, hidden_size]
        || schedule_dims[0]
            != block_dims[0]
                .checked_mul(16)
                .ok_or_else(|| invalid("NVFP4 expert schedule extent overflows"))?
    {
        return Err(invalid(
            "NVFP4 Turing expert down component shapes are inconsistent",
        ));
    }
    require_expert_dtypes(
        activated,
        schedule,
        block_experts,
        payload,
        scales,
        global,
        bias,
        weighted,
    )?;
    if routing.dtype != activated.dtype {
        return Err(invalid(
            "NVFP4 Turing routing weights must match the activation dtype",
        ));
    }
    let request = ExpertDownRequest {
        struct_size: size_of::<ExpertDownRequest>(),
        activated: activated.data.cast_const(),
        sorted_assignments: schedule.data.cast_const().cast(),
        block_experts: block_experts.data.cast_const().cast(),
        payload: payload.data.cast_const().cast(),
        block_scales: scales.data.cast_const().cast(),
        global_scale: global.data.cast_const().cast(),
        bias: bias.data.cast_const(),
        routing_weights: routing.data.cast_const(),
        weighted_output: weighted.data,
        stream: unsafe { stream(frame)? },
        assignments,
        schedule_positions: schedule_dims[0],
        schedule_blocks: block_dims[0],
        intermediate_size: intermediate,
        hidden_size,
        experts,
        experts_per_token: routes,
        block_size: 16,
        dtype: activation_dtype(activated)?,
    };
    call_adapter("expert down", |message| unsafe {
        nml_nvfp4_turing_expert_down(&request, message.as_mut_ptr(), message.len())
    })
}

#[allow(clippy::too_many_arguments)]
fn require_expert_dtypes(
    activation: &sys::XLA_FFI_Buffer,
    schedule: &sys::XLA_FFI_Buffer,
    block_experts: &sys::XLA_FFI_Buffer,
    payload: &sys::XLA_FFI_Buffer,
    scales: &sys::XLA_FFI_Buffer,
    global: &sys::XLA_FFI_Buffer,
    bias: &sys::XLA_FFI_Buffer,
    output: &sys::XLA_FFI_Buffer,
) -> Result<(), HandlerFailure> {
    activation_dtype(activation)?;
    if schedule.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_S32
        || block_experts.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_S32
        || payload.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || scales.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || global.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32
        || !dimensions(global)?.is_empty()
        || bias.dtype != activation.dtype
        || output.dtype != activation.dtype
    {
        return Err(invalid(
            "NVFP4 Turing expert buffers have inconsistent dtypes",
        ));
    }
    Ok(())
}

fn exact_positive_division(
    numerator: i64,
    denominator: i64,
    name: &str,
) -> Result<i64, HandlerFailure> {
    if numerator <= 0 || denominator <= 0 || numerator % denominator != 0 {
        return Err(invalid(&format!("{name} is not a positive integer")));
    }
    Ok(numerator / denominator)
}

fn call_adapter(
    operation: &str,
    call: impl FnOnce(&mut [c_char; 512]) -> i32,
) -> Result<(), HandlerFailure> {
    let mut message = [0 as c_char; 512];
    let status = call(&mut message);
    if status == 0 {
        return Ok(());
    }
    let bytes = message
        .iter()
        .map(|byte| *byte as u8)
        .take_while(|byte| *byte != 0)
        .collect::<Vec<_>>();
    let detail = String::from_utf8_lossy(&bytes);
    Err((
        if status == 1 {
            sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INVALID_ARGUMENT
        } else {
            sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INTERNAL
        },
        format!("NVFP4 Turing {operation} launch failed: {detail}"),
    ))
}

fn require_execute_frame(
    frame: &sys::XLA_FFI_CallFrame,
    operation: &str,
) -> Result<(), HandlerFailure> {
    require_struct(
        frame.struct_size,
        sys::XLA_FFI_CallFrame_STRUCT_SIZE as usize,
        "call frame",
    )?;
    if frame.stage != sys::XLA_FFI_ExecutionStage_XLA_FFI_ExecutionStage_EXECUTE {
        return Err(invalid(&format!(
            "NVFP4 Turing {operation} was called outside execute stage"
        )));
    }
    Ok(())
}

fn activation_dtype(buffer: &sys::XLA_FFI_Buffer) -> Result<c_int, HandlerFailure> {
    match buffer.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 => Ok(1),
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16 => Ok(2),
        _ => Err(invalid("NVFP4 Turing kernels support F16 and BF16 only")),
    }
}

fn ceil_div_positive(value: i64, divisor: i64) -> Option<i64> {
    (value > 0 && divisor > 0)
        .then(|| value.checked_add(divisor - 1)?.checked_div(divisor))
        .flatten()
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
            "NVFP4 Turing operation expects exactly {expected} arguments"
        )));
    }
    if unsafe { *frame.args.types.add(index) } != sys::XLA_FFI_ArgType_XLA_FFI_ArgType_BUFFER {
        return Err(invalid("NVFP4 Turing arguments must be buffers"));
    }
    let pointer = unsafe { *frame.args.args.add(index) }.cast::<sys::XLA_FFI_Buffer>();
    let buffer = unsafe { pointer.as_ref() }.ok_or_else(|| invalid("null argument buffer"))?;
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
        return Err(invalid(&format!(
            "NVFP4 Turing operation expects exactly {expected} results"
        )));
    }
    if unsafe { *frame.rets.types.add(index) } != sys::XLA_FFI_RetType_XLA_FFI_RetType_BUFFER {
        return Err(invalid("NVFP4 Turing results must be buffers"));
    }
    let pointer = unsafe { *frame.rets.rets.add(index) }.cast::<sys::XLA_FFI_Buffer>();
    let buffer = unsafe { pointer.as_ref() }.ok_or_else(|| invalid("null result buffer"))?;
    require_struct(
        buffer.struct_size,
        sys::XLA_FFI_Buffer_STRUCT_SIZE as usize,
        "result buffer",
    )?;
    Ok(buffer)
}

fn dimensions(buffer: &sys::XLA_FFI_Buffer) -> Result<&[i64], HandlerFailure> {
    if buffer.rank < 0 || (buffer.rank != 0 && buffer.dims.is_null()) || buffer.data.is_null() {
        return Err(invalid(
            "NVFP4 Turing operation received malformed buffer metadata",
        ));
    }
    if buffer.rank == 0 {
        return Ok(&[]);
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.dims, buffer.rank as usize) })
}

fn metadata_query(frame: &mut sys::XLA_FFI_CallFrame) -> bool {
    let mut current = frame.extension_start;
    while let Some(extension) = NonNull::new(current) {
        let actual = unsafe { extension.cast::<usize>().as_ptr().read() };
        if actual < sys::XLA_FFI_Extension_Base_STRUCT_SIZE as usize {
            return false;
        }
        let extension = unsafe { extension.as_ref() };
        if extension.type_ != sys::XLA_FFI_Extension_Type_XLA_FFI_Extension_Metadata {
            current = extension.next;
            continue;
        }
        if actual < sys::XLA_FFI_Metadata_Extension_STRUCT_SIZE as usize {
            return true;
        }
        let metadata_extension = unsafe { &mut *current.cast::<sys::XLA_FFI_Metadata_Extension>() };
        let Some(metadata) = NonNull::new(metadata_extension.metadata) else {
            return true;
        };
        let actual = unsafe { metadata.cast::<usize>().as_ptr().read() };
        if actual < sys::XLA_FFI_Metadata_STRUCT_SIZE as usize {
            return true;
        }
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

unsafe fn stream(frame: &sys::XLA_FFI_CallFrame) -> Result<*mut c_void, HandlerFailure> {
    let api = unsafe { ffi_api(frame)? };
    let get = api
        .stream_get
        .ok_or_else(|| invalid("XLA FFI API does not expose the CUDA stream"))?;
    let mut args: sys::XLA_FFI_Stream_Get_Args = unsafe { zeroed() };
    args.struct_size = sys::XLA_FFI_Stream_Get_Args_STRUCT_SIZE as usize;
    args.ctx = frame.ctx;
    let error = unsafe { get(&mut args) };
    if !error.is_null() {
        let message = unsafe { take_ffi_error(api, error) };
        return Err((
            sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INTERNAL,
            format!("XLA FFI could not provide the CUDA stream: {message}"),
        ));
    }
    if args.stream.is_null() {
        return Err(invalid("XLA FFI returned a null CUDA stream"));
    }
    Ok(args.stream)
}

unsafe fn take_ffi_error(api: &FfiApiPrefix, error: *mut sys::XLA_FFI_Error) -> String {
    let message = if let Some(get_message) = api.error_get_message {
        let mut args: sys::XLA_FFI_Error_GetMessage_Args = unsafe { zeroed() };
        args.struct_size = sys::XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE as usize;
        args.error = error;
        unsafe { get_message(&mut args) };
        if args.message.is_null() {
            "unknown XLA FFI error".to_owned()
        } else {
            unsafe { CStr::from_ptr(args.message) }
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "unknown XLA FFI error".to_owned()
    };
    if let Some(destroy) = api.error_destroy {
        let mut args: sys::XLA_FFI_Error_Destroy_Args = unsafe { zeroed() };
        args.struct_size = sys::XLA_FFI_Error_Destroy_Args_STRUCT_SIZE as usize;
        args.error = error;
        unsafe { destroy(&mut args) };
    }
    message
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
        .unwrap_or_else(|_| CString::new("NVFP4 Turing failure contained a NUL byte").unwrap());
    let mut args: sys::XLA_FFI_Error_Create_Args = unsafe { zeroed() };
    args.struct_size = sys::XLA_FFI_Error_Create_Args_STRUCT_SIZE as usize;
    args.message = message.as_ptr();
    args.errc = code;
    unsafe { create(&mut args) }
}

unsafe fn ffi_api(frame: &sys::XLA_FFI_CallFrame) -> Result<&FfiApiPrefix, HandlerFailure> {
    let pointer = NonNull::new(frame.api.cast_mut())
        .ok_or_else(|| invalid("NVFP4 Turing call frame has no FFI API"))?;
    let actual = unsafe { pointer.cast::<usize>().as_ptr().read() };
    let required = offset_of!(FfiApiPrefix, stream_get) + size_of::<*const c_void>();
    require_struct(actual, required, "API table")?;
    Ok(unsafe { pointer.cast::<FfiApiPrefix>().as_ref() })
}

fn require_struct(actual: usize, required: usize, name: &str) -> Result<(), HandlerFailure> {
    if actual < required {
        Err(invalid(&format!(
            "truncated XLA FFI {name}: expected {required} bytes, received {actual}"
        )))
    } else {
        Ok(())
    }
}

fn invalid(message: &str) -> HandlerFailure {
    (
        sys::XLA_FFI_Error_Code_XLA_FFI_Error_Code_INVALID_ARGUMENT,
        message.to_owned(),
    )
}
