//! XLA typed-FFI adapter for the portable CPU linear kernel.

use crate::{Weight, embedding, linear, routed_clamped_swiglu};
use nml_pjrt::{Ffi, FfiHandler};
use nml_pjrt_sys as sys;
use nml_types::{BFloat16, DType, F16, Shape};
use std::collections::HashSet;
use std::ffi::{CString, c_void};
use std::mem::{offset_of, size_of, zeroed};
use std::ptr::{NonNull, null_mut};
use std::sync::{Mutex, OnceLock};

const LINEAR_TARGET: &str = "nml.nvfp4.linear";
const EMBEDDING_TARGET: &str = "nml.nvfp4.embedding";
const ROUTED_SWIGLU_TARGET: &str = "nml.nvfp4.routed_swiglu";
static REGISTERED: OnceLock<Mutex<HashSet<(usize, &'static str)>>> = OnceLock::new();

// Bindgen leaves this runtime-owned table opaque. The handler needs only the
// prefix through Error_Create, and validates `struct_size` before reading it.
#[repr(C)]
struct FfiApiPrefix {
    struct_size: usize,
    extension_start: *mut sys::XLA_FFI_Extension_Base,
    api_version: sys::XLA_FFI_Api_Version,
    internal_api: *const sys::XLA_FFI_InternalApi,
    error_create: Option<
        unsafe extern "C" fn(*mut sys::XLA_FFI_Error_Create_Args) -> *mut sys::XLA_FFI_Error,
    >,
}

type HandlerFailure = (sys::XLA_FFI_Error_Code, String);

/// Registers the portable CPU handler once per PJRT plugin API table.
pub fn register_cpu(ffi: &Ffi, platform_name: &str) -> Result<(), nml_pjrt::Error> {
    let registered = REGISTERED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut registered = registered.lock().unwrap_or_else(|error| error.into_inner());
    for (target, handler) in [
        (LINEAR_TARGET, cpu_linear as *const ()),
        (EMBEDDING_TARGET, cpu_embedding as *const ()),
        (ROUTED_SWIGLU_TARGET, cpu_routed_swiglu as *const ()),
    ] {
        let key = (ffi.plugin_identity(), target);
        if registered.contains(&key) {
            continue;
        }
        let address =
            NonNull::new(handler as *mut c_void).expect("a static function has a non-null address");
        // SAFETY: every entry above implements the pinned typed-FFI
        // signature, answers metadata queries, and has process lifetime.
        unsafe {
            ffi.register(
                target,
                platform_name,
                FfiHandler::from_address(address),
                false,
            )
        }?;
        registered.insert(key);
    }
    Ok(())
}

unsafe extern "C" fn cpu_linear(raw: *mut sys::XLA_FFI_CallFrame) -> *mut sys::XLA_FFI_Error {
    let Some(frame) = (unsafe { raw.as_mut() }) else {
        return null_mut();
    };
    if metadata_query(frame) {
        return null_mut();
    }
    match unsafe { execute_linear(frame) } {
        Ok(()) => null_mut(),
        Err((code, message)) => ffi_error(frame, code, &message),
    }
}

unsafe extern "C" fn cpu_embedding(raw: *mut sys::XLA_FFI_CallFrame) -> *mut sys::XLA_FFI_Error {
    let Some(frame) = (unsafe { raw.as_mut() }) else {
        return null_mut();
    };
    if metadata_query(frame) {
        return null_mut();
    }
    match unsafe { execute_embedding(frame) } {
        Ok(()) => null_mut(),
        Err((code, message)) => ffi_error(frame, code, &message),
    }
}

unsafe extern "C" fn cpu_routed_swiglu(
    raw: *mut sys::XLA_FFI_CallFrame,
) -> *mut sys::XLA_FFI_Error {
    let Some(frame) = (unsafe { raw.as_mut() }) else {
        return null_mut();
    };
    if metadata_query(frame) {
        return null_mut();
    }
    match unsafe { execute_routed_swiglu(frame) } {
        Ok(()) => null_mut(),
        Err((code, message)) => ffi_error(frame, code, &message),
    }
}

unsafe fn execute_linear(frame: &mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_struct(
        frame.struct_size,
        sys::XLA_FFI_CallFrame_STRUCT_SIZE as usize,
        "call frame",
    )?;
    if frame.stage != sys::XLA_FFI_ExecutionStage_XLA_FFI_ExecutionStage_EXECUTE {
        return Err(invalid("NVFP4 CPU linear was called outside execute stage"));
    }
    require_struct(
        frame.args.struct_size,
        sys::XLA_FFI_Args_STRUCT_SIZE as usize,
        "argument list",
    )?;
    let argument_count = match frame.args.size {
        4 => 4,
        5 => 5,
        _ => {
            return Err(invalid(
                "NVFP4 CPU linear expects four arguments without bias or five with bias",
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
        return Err(invalid("NVFP4 CPU linear received invalid ranks"));
    }
    let inputs = to_usize(*activation_dimensions.last().unwrap(), "input width")?;
    let outputs = to_usize(payload_dimensions[0], "output width")?;
    if inputs == 0
        || outputs == 0
        || payload_dimensions[1] != i64::try_from(inputs.div_ceil(2)).unwrap_or(i64::MAX)
        || scale_dimensions
            != [
                payload_dimensions[0],
                i64::try_from(inputs.div_ceil(16)).unwrap_or(i64::MAX),
            ]
        || output_dimensions[..output_dimensions.len() - 1]
            != activation_dimensions[..activation_dimensions.len() - 1]
        || output_dimensions[output_dimensions.len() - 1] != payload_dimensions[0]
    {
        return Err(invalid("NVFP4 CPU linear buffer shapes are inconsistent"));
    }
    if payload.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || scales.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || global.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32
        || !dimensions(global)?.is_empty()
        || output.dtype != activation.dtype
    {
        return Err(invalid("NVFP4 CPU linear buffer dtypes are inconsistent"));
    }
    if let Some(bias) = bias
        && (dimensions(bias)? != [payload_dimensions[0]] || bias.dtype != activation.dtype)
    {
        return Err(invalid(
            "NVFP4 CPU linear bias must be a length-N vector matching the activation dtype",
        ));
    }

    let mut rows = 1usize;
    for &dimension in &activation_dimensions[..activation_dimensions.len() - 1] {
        rows = rows
            .checked_mul(to_usize(dimension, "activation dimension")?)
            .ok_or_else(|| invalid("NVFP4 CPU linear activation extent overflows"))?;
    }
    let activation_count = rows
        .checked_mul(inputs)
        .ok_or_else(|| invalid("NVFP4 CPU linear activation extent overflows"))?;
    let output_count = rows
        .checked_mul(outputs)
        .ok_or_else(|| invalid("NVFP4 CPU linear output extent overflows"))?;
    let payload_count = outputs
        .checked_mul(inputs.div_ceil(2))
        .ok_or_else(|| invalid("NVFP4 CPU linear payload extent overflows"))?;
    let scale_count = outputs
        .checked_mul(inputs.div_ceil(16))
        .ok_or_else(|| invalid("NVFP4 CPU linear scale extent overflows"))?;
    let payload_values = unsafe { bytes(payload, payload_count)? };
    let scale_values = unsafe { bytes(scales, scale_count)? };
    if global.data.is_null() {
        return Err(invalid("NVFP4 CPU linear received a null global scale"));
    }
    let global_value = unsafe { *global.data.cast::<f32>() };
    let dtype = match activation.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 => DType::F16,
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16 => DType::Bf16,
        _ => {
            return Err(invalid(
                "NVFP4 CPU linear supports F16 and BF16 activations",
            ));
        }
    };
    let logical_shape = Shape::new(dtype, &[outputs as i64, inputs as i64])
        .map_err(|error| invalid(&error.to_string()))?;
    let weight = Weight::new(logical_shape, payload_values, scale_values, global_value)
        .map_err(|error| invalid(&error.to_string()))?;
    let input = read_activation(activation, activation_count)?;
    let bias = bias
        .map(|bias| read_activation(bias, outputs))
        .transpose()?;
    let mut computed = vec![0.0f32; output_count];
    linear(&input, &weight, bias.as_deref(), &mut computed)
        .map_err(|error| invalid(&error.to_string()))?;
    unsafe { write_output(output, &computed)? };
    Ok(())
}

unsafe fn execute_routed_swiglu(
    frame: &mut sys::XLA_FFI_CallFrame,
) -> Result<(), HandlerFailure> {
    require_struct(
        frame.struct_size,
        sys::XLA_FFI_CallFrame_STRUCT_SIZE as usize,
        "call frame",
    )?;
    if frame.stage != sys::XLA_FFI_ExecutionStage_XLA_FFI_ExecutionStage_EXECUTE {
        return Err(invalid(
            "NVFP4 routed clamped SwiGLU was called outside execute stage",
        ));
    }
    let hidden = unsafe { argument(frame, 0, 11)? };
    let router_indices = unsafe { argument(frame, 1, 11)? };
    let routing_weights = unsafe { argument(frame, 2, 11)? };
    let gate_payload = unsafe { argument(frame, 3, 11)? };
    let gate_scales = unsafe { argument(frame, 4, 11)? };
    let gate_global = unsafe { argument(frame, 5, 11)? };
    let gate_bias = unsafe { argument(frame, 6, 11)? };
    let down_payload = unsafe { argument(frame, 7, 11)? };
    let down_scales = unsafe { argument(frame, 8, 11)? };
    let down_global = unsafe { argument(frame, 9, 11)? };
    let down_bias = unsafe { argument(frame, 10, 11)? };
    let output = unsafe { result(frame, 0, 1)? };

    let hidden_dimensions = dimensions(hidden)?;
    let routing_dimensions = dimensions(router_indices)?;
    let gate_payload_dimensions = dimensions(gate_payload)?;
    let gate_scale_dimensions = dimensions(gate_scales)?;
    let gate_bias_dimensions = dimensions(gate_bias)?;
    let down_payload_dimensions = dimensions(down_payload)?;
    let down_scale_dimensions = dimensions(down_scales)?;
    let down_bias_dimensions = dimensions(down_bias)?;
    let output_dimensions = dimensions(output)?;
    if hidden_dimensions.len() != 2
        || routing_dimensions.len() != 2
        || gate_payload_dimensions.len() != 3
        || gate_scale_dimensions.len() != 3
        || gate_bias_dimensions.len() != 2
        || down_payload_dimensions.len() != 3
        || down_scale_dimensions.len() != 3
        || down_bias_dimensions.len() != 2
        || dimensions(routing_weights)? != routing_dimensions
        || output_dimensions != hidden_dimensions
    {
        return Err(invalid("NVFP4 routed clamped SwiGLU received invalid ranks"));
    }
    let tokens = to_usize(hidden_dimensions[0], "token count")?;
    let hidden_size = to_usize(hidden_dimensions[1], "hidden size")?;
    let experts = to_usize(gate_payload_dimensions[0], "expert count")?;
    let gate_inputs = hidden_size;
    let doubled_intermediate = to_usize(gate_bias_dimensions[1], "gate/up output width")?;
    if doubled_intermediate == 0 || doubled_intermediate % 2 != 0 {
        return Err(invalid(
            "NVFP4 clamped SwiGLU gate/up width must be positive and even",
        ));
    }
    let intermediate = doubled_intermediate / 2;
    if tokens != to_usize(routing_dimensions[0], "routing token count")?
        || hidden_size != gate_inputs
        || experts == 0
        || gate_payload_dimensions[1] != doubled_intermediate as i64
        || gate_payload_dimensions[2]
            != i64::try_from(hidden_size.div_ceil(2)).unwrap_or(i64::MAX)
        || gate_scale_dimensions
            != [
                gate_payload_dimensions[0],
                doubled_intermediate as i64,
                i64::try_from(hidden_size.div_ceil(16)).unwrap_or(i64::MAX),
            ]
        || gate_bias_dimensions != [gate_payload_dimensions[0], doubled_intermediate as i64]
        || down_payload_dimensions[0] != gate_payload_dimensions[0]
        || down_payload_dimensions[1] != hidden_size as i64
        || down_payload_dimensions[2]
            != i64::try_from(intermediate.div_ceil(2)).unwrap_or(i64::MAX)
        || down_scale_dimensions
            != [
                gate_payload_dimensions[0],
                hidden_size as i64,
                i64::try_from(intermediate.div_ceil(16)).unwrap_or(i64::MAX),
            ]
        || down_bias_dimensions != [gate_payload_dimensions[0], hidden_size as i64]
    {
        return Err(invalid(
            "NVFP4 routed clamped SwiGLU component shapes are inconsistent",
        ));
    }
    for buffer in [gate_payload, gate_scales, down_payload, down_scales] {
        if buffer.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8 {
            return Err(invalid(
                "NVFP4 routed SwiGLU payloads and block scales must use U8 buffers",
            ));
        }
    }
    for global in [gate_global, down_global] {
        if global.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32
            || !dimensions(global)?.is_empty()
            || global.data.is_null()
        {
            return Err(invalid(
                "NVFP4 routed SwiGLU global scales must be non-null F32 scalars",
            ));
        }
    }
    if hidden.dtype != routing_weights.dtype
        || hidden.dtype != gate_bias.dtype
        || hidden.dtype != down_bias.dtype
        || hidden.dtype != output.dtype
        || !matches!(
            hidden.dtype,
            sys::XLA_FFI_DataType_XLA_FFI_DataType_F16
                | sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16
        )
    {
        return Err(invalid(
            "NVFP4 routed SwiGLU activations and biases must share F16 or BF16",
        ));
    }

    let route_count = element_count(routing_dimensions, "routing")?;
    let hidden_count = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU hidden extent overflows"))?;
    let gate_payload_count = experts
        .checked_mul(doubled_intermediate)
        .and_then(|value| value.checked_mul(hidden_size.div_ceil(2)))
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU gate/up payload extent overflows"))?;
    let gate_scale_count = experts
        .checked_mul(doubled_intermediate)
        .and_then(|value| value.checked_mul(hidden_size.div_ceil(16)))
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU gate/up scale extent overflows"))?;
    let gate_bias_count = experts
        .checked_mul(doubled_intermediate)
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU gate/up bias extent overflows"))?;
    let down_payload_count = experts
        .checked_mul(hidden_size)
        .and_then(|value| value.checked_mul(intermediate.div_ceil(2)))
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU down payload extent overflows"))?;
    let down_scale_count = experts
        .checked_mul(hidden_size)
        .and_then(|value| value.checked_mul(intermediate.div_ceil(16)))
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU down scale extent overflows"))?;
    let down_bias_count = experts
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid("NVFP4 routed SwiGLU down bias extent overflows"))?;

    let dtype = if hidden.dtype == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 {
        DType::F16
    } else {
        DType::Bf16
    };
    let gate_shape = Shape::new(
        dtype,
        &[
            experts as i64,
            doubled_intermediate as i64,
            gate_inputs as i64,
        ],
    )
    .map_err(|error| invalid(&error.to_string()))?;
    let down_shape = Shape::new(
        dtype,
        &[experts as i64, hidden_size as i64, intermediate as i64],
    )
    .map_err(|error| invalid(&error.to_string()))?;
    let gate = Weight::new(
        gate_shape,
        unsafe { bytes(gate_payload, gate_payload_count)? },
        unsafe { bytes(gate_scales, gate_scale_count)? },
        unsafe { *gate_global.data.cast::<f32>() },
    )
    .map_err(|error| invalid(&error.to_string()))?;
    let down = Weight::new(
        down_shape,
        unsafe { bytes(down_payload, down_payload_count)? },
        unsafe { bytes(down_scales, down_scale_count)? },
        unsafe { *down_global.data.cast::<f32>() },
    )
    .map_err(|error| invalid(&error.to_string()))?;
    let hidden = read_activation(hidden, hidden_count)?;
    let indices = read_indices(router_indices, route_count)?;
    let routing_weights = read_activation(routing_weights, route_count)?;
    let gate_bias = read_activation(gate_bias, gate_bias_count)?;
    let down_bias = read_activation(down_bias, down_bias_count)?;
    let mut computed = vec![0.0f32; hidden_count];
    routed_clamped_swiglu(
        &hidden,
        tokens,
        &indices,
        &routing_weights,
        &gate,
        &gate_bias,
        &down,
        &down_bias,
        &mut computed,
    )
    .map_err(|error| invalid(&error.to_string()))?;
    unsafe { write_output(output, &computed)? };
    Ok(())
}

unsafe fn execute_embedding(frame: &mut sys::XLA_FFI_CallFrame) -> Result<(), HandlerFailure> {
    require_struct(
        frame.struct_size,
        sys::XLA_FFI_CallFrame_STRUCT_SIZE as usize,
        "call frame",
    )?;
    if frame.stage != sys::XLA_FFI_ExecutionStage_XLA_FFI_ExecutionStage_EXECUTE {
        return Err(invalid(
            "NVFP4 CPU embedding was called outside execute stage",
        ));
    }
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
        return Err(invalid("NVFP4 CPU embedding received invalid ranks"));
    }
    let vocabulary = to_usize(payload_dimensions[0], "vocabulary size")?;
    let width = to_usize(
        *output_dimensions
            .last()
            .expect("embedding output has one axis"),
        "embedding width",
    )?;
    if vocabulary == 0
        || width == 0
        || payload_dimensions[1] != i64::try_from(width.div_ceil(2)).unwrap_or(i64::MAX)
        || scale_dimensions
            != [
                payload_dimensions[0],
                i64::try_from(width.div_ceil(16)).unwrap_or(i64::MAX),
            ]
    {
        return Err(invalid(
            "NVFP4 CPU embedding component shapes are inconsistent",
        ));
    }
    if payload.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || scales.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_U8
        || global.dtype != sys::XLA_FFI_DataType_XLA_FFI_DataType_F32
        || !dimensions(global)?.is_empty()
        || !matches!(
            output.dtype,
            sys::XLA_FFI_DataType_XLA_FFI_DataType_F16
                | sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16
        )
    {
        return Err(invalid(
            "NVFP4 CPU embedding buffer dtypes are inconsistent",
        ));
    }
    let index_count = element_count(index_dimensions, "embedding index")?;
    let payload_count = vocabulary
        .checked_mul(width.div_ceil(2))
        .ok_or_else(|| invalid("NVFP4 CPU embedding payload extent overflows"))?;
    let scale_count = vocabulary
        .checked_mul(width.div_ceil(16))
        .ok_or_else(|| invalid("NVFP4 CPU embedding scale extent overflows"))?;
    let output_count = index_count
        .checked_mul(width)
        .ok_or_else(|| invalid("NVFP4 CPU embedding output extent overflows"))?;
    let payload_values = unsafe { bytes(payload, payload_count)? };
    let scale_values = unsafe { bytes(scales, scale_count)? };
    if global.data.is_null() {
        return Err(invalid("NVFP4 CPU embedding received a null global scale"));
    }
    let global_value = unsafe { *global.data.cast::<f32>() };
    let dtype = if output.dtype == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 {
        DType::F16
    } else {
        DType::Bf16
    };
    let logical_shape = Shape::new(dtype, &[vocabulary as i64, width as i64])
        .map_err(|error| invalid(&error.to_string()))?;
    let weight = Weight::new(logical_shape, payload_values, scale_values, global_value)
        .map_err(|error| invalid(&error.to_string()))?;
    let indices = read_indices(indices, index_count)?;
    let mut computed = vec![0.0f32; output_count];
    embedding(&weight, &indices, &mut computed).map_err(|error| invalid(&error.to_string()))?;
    unsafe { write_output(output, &computed)? };
    Ok(())
}

fn read_indices(buffer: &sys::XLA_FFI_Buffer, count: usize) -> Result<Vec<usize>, HandlerFailure> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if buffer.data.is_null() {
        return Err(invalid("NVFP4 CPU embedding received a null index buffer"));
    }
    match buffer.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_S32 => {
            let values = unsafe { std::slice::from_raw_parts(buffer.data.cast::<i32>(), count) };
            values
                .iter()
                .map(|&value| {
                    usize::try_from(value)
                        .map_err(|_| invalid("NVFP4 CPU embedding index is negative"))
                })
                .collect()
        }
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_S64 => {
            let values = unsafe { std::slice::from_raw_parts(buffer.data.cast::<i64>(), count) };
            values
                .iter()
                .map(|&value| {
                    usize::try_from(value).map_err(|_| {
                        invalid("NVFP4 CPU embedding index is negative or exceeds usize")
                    })
                })
                .collect()
        }
        _ => Err(invalid("NVFP4 CPU embedding supports I32 and I64 indices")),
    }
}

fn read_activation(buffer: &sys::XLA_FFI_Buffer, count: usize) -> Result<Vec<f32>, HandlerFailure> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if buffer.data.is_null() {
        return Err(invalid(
            "NVFP4 CPU operation received a null activation buffer",
        ));
    }
    match buffer.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 => {
            // SAFETY: XLA owns `count` contiguous F16 elements for this call.
            let values = unsafe { std::slice::from_raw_parts(buffer.data.cast::<F16>(), count) };
            Ok(values.iter().map(|value| value.to_f32()).collect())
        }
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16 => {
            // SAFETY: XLA owns `count` contiguous BF16 elements for this call.
            let values =
                unsafe { std::slice::from_raw_parts(buffer.data.cast::<BFloat16>(), count) };
            Ok(values.iter().map(|value| value.to_f32()).collect())
        }
        _ => Err(invalid(
            "NVFP4 CPU operations support F16 and BF16 activations",
        )),
    }
}

unsafe fn write_output(buffer: &sys::XLA_FFI_Buffer, values: &[f32]) -> Result<(), HandlerFailure> {
    if values.is_empty() {
        return Ok(());
    }
    if buffer.data.is_null() {
        return Err(invalid("NVFP4 CPU operation received a null output buffer"));
    }
    match buffer.dtype {
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_F16 => {
            // SAFETY: XLA owns `values.len()` writable F16 elements.
            let output =
                unsafe { std::slice::from_raw_parts_mut(buffer.data.cast::<F16>(), values.len()) };
            for (output, &value) in output.iter_mut().zip(values) {
                *output = F16::from_f32(value);
            }
        }
        value if value == sys::XLA_FFI_DataType_XLA_FFI_DataType_BF16 => {
            // SAFETY: XLA owns `values.len()` writable BF16 elements.
            let output = unsafe {
                std::slice::from_raw_parts_mut(buffer.data.cast::<BFloat16>(), values.len())
            };
            for (output, &value) in output.iter_mut().zip(values) {
                *output = BFloat16::from_f32(value);
            }
        }
        _ => {
            return Err(invalid("NVFP4 CPU operations support F16 and BF16 outputs"));
        }
    }
    Ok(())
}

unsafe fn bytes<'a>(
    buffer: &'a sys::XLA_FFI_Buffer,
    count: usize,
) -> Result<&'a [u8], HandlerFailure> {
    if count == 0 {
        return Ok(&[]);
    }
    if buffer.data.is_null() {
        return Err(invalid(
            "NVFP4 CPU operation received a null component buffer",
        ));
    }
    // SAFETY: the validated physical shape owns exactly `count` U8 elements.
    Ok(unsafe { std::slice::from_raw_parts(buffer.data.cast::<u8>(), count) })
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
            "NVFP4 CPU operation expects exactly {expected} buffer arguments"
        )));
    }
    // SAFETY: the validated arrays contain `expected` entries.
    if unsafe { *frame.args.types.add(index) } != sys::XLA_FFI_ArgType_XLA_FFI_ArgType_BUFFER {
        return Err(invalid("NVFP4 CPU operation arguments must be buffers"));
    }
    // SAFETY: XLA stores one live buffer pointer per argument.
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
            "NVFP4 CPU operation expects exactly {expected} result buffers"
        )));
    }
    // SAFETY: the validated arrays contain `expected` entries.
    if unsafe { *frame.rets.types.add(index) } != sys::XLA_FFI_RetType_XLA_FFI_RetType_BUFFER {
        return Err(invalid("NVFP4 CPU operation results must be buffers"));
    }
    // SAFETY: XLA stores one live buffer pointer per result.
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
    if buffer.rank < 0 || (buffer.rank != 0 && buffer.dims.is_null()) {
        return Err(invalid(
            "NVFP4 CPU operation received malformed buffer metadata",
        ));
    }
    if buffer.rank == 0 {
        return Ok(&[]);
    }
    // SAFETY: XLA guarantees `rank` readable dimension entries.
    Ok(unsafe { std::slice::from_raw_parts(buffer.dims, buffer.rank as usize) })
}

fn to_usize(value: i64, name: &str) -> Result<usize, HandlerFailure> {
    usize::try_from(value).map_err(|_| invalid(&format!("{name} is negative or exceeds usize")))
}

fn element_count(dimensions: &[i64], name: &str) -> Result<usize, HandlerFailure> {
    dimensions.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(to_usize(dimension, name)?)
            .ok_or_else(|| invalid(&format!("{name} extent overflows usize")))
    })
}

fn metadata_query(frame: &mut sys::XLA_FFI_CallFrame) -> bool {
    let mut current = frame.extension_start;
    while let Some(extension) = NonNull::new(current) {
        // SAFETY: every extension begins with a size word.
        let actual = unsafe { extension.cast::<usize>().as_ptr().read() };
        if actual < sys::XLA_FFI_Extension_Base_STRUCT_SIZE as usize {
            return false;
        }
        // SAFETY: the base prefix is size-checked.
        let extension = unsafe { extension.as_ref() };
        if extension.type_ != sys::XLA_FFI_Extension_Type_XLA_FFI_Extension_Metadata {
            current = extension.next;
            continue;
        }
        if actual < sys::XLA_FFI_Metadata_Extension_STRUCT_SIZE as usize {
            return true;
        }
        // SAFETY: type and size identify the metadata extension.
        let metadata_extension = unsafe { &mut *current.cast::<sys::XLA_FFI_Metadata_Extension>() };
        let Some(metadata) = NonNull::new(metadata_extension.metadata) else {
            return true;
        };
        // SAFETY: metadata begins with its size word.
        let actual = unsafe { metadata.cast::<usize>().as_ptr().read() };
        if actual < sys::XLA_FFI_Metadata_STRUCT_SIZE as usize {
            return true;
        }
        // SAFETY: XLA supplies writable metadata for this registration query.
        let metadata = unsafe { &mut *metadata.as_ptr() };
        metadata.api_version.struct_size = sys::XLA_FFI_Api_Version_STRUCT_SIZE as usize;
        metadata.api_version.extension_start = null_mut();
        metadata.api_version.major_version = sys::XLA_FFI_API_MAJOR as i32;
        metadata.api_version.minor_version = sys::XLA_FFI_API_MINOR as i32;
        metadata.traits = 0;
        return true;
    }
    false
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
    let message = CString::new(message).unwrap_or_else(|_| {
        CString::new("NVFP4 CPU handler failure contained a NUL byte").unwrap()
    });
    // SAFETY: zero initializes the optional extension pointer.
    let mut args: sys::XLA_FFI_Error_Create_Args = unsafe { zeroed() };
    args.struct_size = sys::XLA_FFI_Error_Create_Args_STRUCT_SIZE as usize;
    args.message = message.as_ptr();
    args.errc = code;
    // SAFETY: XLA copies the live message and takes ownership of the result.
    unsafe { create(&mut args) }
}

unsafe fn ffi_api(frame: &sys::XLA_FFI_CallFrame) -> Result<&FfiApiPrefix, HandlerFailure> {
    let pointer = NonNull::new(frame.api.cast_mut())
        .ok_or_else(|| invalid("NVFP4 CPU call frame has no FFI API"))?;
    // SAFETY: every API table begins with a readable size word.
    let actual = unsafe { pointer.cast::<usize>().as_ptr().read() };
    let required = offset_of!(FfiApiPrefix, error_create) + size_of::<*const c_void>();
    require_struct(actual, required, "API table")?;
    // SAFETY: the size check proves this immutable prefix is present.
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
