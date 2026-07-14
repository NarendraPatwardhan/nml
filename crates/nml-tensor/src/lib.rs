//! Typed host tensor storage and strided views.
//!
//! `Slice` deliberately mirrors ZML's one host-tensor concept. Borrowed,
//! mutable, and owned storage are implementation states of that concept rather
//! than separate public tensor abstractions.

use nml_types::{BFloat16, Complex64, Complex128, DType, F16, MAX_RANK, Shape, ShapeError};
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::error::Error as StdError;
use std::fmt;
use std::ptr::NonNull;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    const NATIVE: Self = if cfg!(target_endian = "little") {
        Self::Little
    } else {
        Self::Big
    };
}

enum Storage<'a> {
    Borrowed(&'a [u8]),
    BorrowedMut(&'a mut [u8]),
    Owned(AlignedBytes),
}

impl Storage<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::BorrowedMut(bytes) => bytes,
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }

    fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Borrowed(_) => None,
            Self::BorrowedMut(bytes) => Some(bytes),
            Self::Owned(bytes) => Some(bytes.as_mut_slice()),
        }
    }
}

struct AlignedBytes {
    pointer: NonNull<u8>,
    length: usize,
    alignment: usize,
}

// SAFETY: AlignedBytes uniquely owns its allocation. Moving it between
// threads transfers that ownership, and shared access exposes only &[u8]; all
// mutation still requires &mut self.
unsafe impl Send for AlignedBytes {}
// SAFETY: the allocation has no interior mutability and deallocation requires
// unique ownership, so concurrent shared reads are sound.
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    fn zeroed(length: usize, alignment: usize) -> Result<Self, Error> {
        let layout = Layout::from_size_align(length.max(1), alignment)
            .map_err(|_| Error::InvalidAlignment(alignment))?;
        // SAFETY: layout is non-zero and valid. The returned allocation is
        // owned by this value and released with the identical layout in Drop.
        let pointer =
            NonNull::new(unsafe { alloc(layout) }).unwrap_or_else(|| handle_alloc_error(layout));
        if length != 0 {
            // SAFETY: pointer owns at least `length` writable bytes.
            unsafe { pointer.as_ptr().write_bytes(0, length) };
        }
        Ok(Self {
            pointer,
            length,
            alignment,
        })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation remains live for self and contains length
        // initialized bytes (zeroed at creation or subsequently written).
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: &mut self proves unique access to the owned allocation.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.length.max(1), self.alignment)
            .expect("stored allocation layout remains valid");
        // SAFETY: pointer was allocated with this exact layout and has not
        // previously been deallocated.
        unsafe { dealloc(self.pointer.as_ptr(), layout) };
    }
}

/// A shaped host tensor or a view into shaped host tensor storage.
pub struct Slice<'a> {
    storage: Storage<'a>,
    shape: Shape,
    offset_bytes: usize,
    byte_strides: [i64; MAX_RANK],
    byte_order: ByteOrder,
}

impl<'a> Slice<'a> {
    pub fn from_bytes(shape: Shape, bytes: &'a [u8]) -> Result<Self, Error> {
        Self::from_storage(shape, Storage::Borrowed(bytes), ByteOrder::NATIVE)
    }

    pub fn from_bytes_mut(shape: Shape, bytes: &'a mut [u8]) -> Result<Self, Error> {
        Self::from_storage(shape, Storage::BorrowedMut(bytes), ByteOrder::NATIVE)
    }

    /// Wraps safetensors-compatible little-endian storage without converting
    /// it. PJRT transfer rejects it on a non-little-endian host.
    pub fn from_little_endian_bytes(shape: Shape, bytes: &'a [u8]) -> Result<Self, Error> {
        Self::from_storage(shape, Storage::Borrowed(bytes), ByteOrder::Little)
    }

    pub fn alloc(shape: Shape) -> Result<Slice<'static>, Error> {
        let length = shape.byte_count()?;
        let storage = Storage::Owned(AlignedBytes::zeroed(length, shape.dtype().alignment())?);
        Slice::from_storage(shape, storage, ByteOrder::NATIVE)
    }

    pub fn from_typed<T: Element>(shape: Shape, values: &'a [T]) -> Result<Self, Error> {
        require_element_type::<T>(shape.dtype())?;
        let actual = std::mem::size_of_val(values);
        let expected = shape.byte_count()?;
        if actual != expected {
            return Err(Error::ByteLength { expected, actual });
        }
        // SAFETY: T is sealed to plain tensor element representations, and the
        // produced byte borrow cannot outlive the source slice.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), actual) };
        Self::from_bytes(shape, bytes)
    }

    pub fn from_typed_mut<T: Element>(shape: Shape, values: &'a mut [T]) -> Result<Self, Error> {
        require_element_type::<T>(shape.dtype())?;
        let actual = std::mem::size_of_val(values);
        let expected = shape.byte_count()?;
        if actual != expected {
            return Err(Error::ByteLength { expected, actual });
        }
        // SAFETY: same representation contract as from_typed, with unique
        // source access retained by the returned lifetime.
        let bytes = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), actual) };
        Self::from_bytes_mut(shape, bytes)
    }

    fn from_storage(
        shape: Shape,
        storage: Storage<'a>,
        byte_order: ByteOrder,
    ) -> Result<Self, Error> {
        let expected = shape.byte_count()?;
        let actual = storage.bytes().len();
        if actual != expected {
            return Err(Error::ByteLength { expected, actual });
        }
        let byte_strides = dense_byte_strides(shape)?;
        let result = Self {
            storage,
            shape,
            offset_bytes: 0,
            byte_strides,
            byte_order,
        };
        result.validate_reachable_range()?;
        Ok(result)
    }

    pub const fn shape(&self) -> Shape {
        self.shape
    }

    pub const fn dtype(&self) -> DType {
        self.shape.dtype()
    }

    pub const fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn byte_strides(&self) -> &[i64] {
        &self.byte_strides[..self.shape.rank()]
    }

    pub fn is_mutable(&self) -> bool {
        !matches!(self.storage, Storage::Borrowed(_))
    }

    pub fn is_native_endian(&self) -> bool {
        self.byte_order == ByteOrder::NATIVE
    }

    pub fn is_contiguous(&self) -> bool {
        dense_byte_strides(self.shape).is_ok_and(|expected| {
            expected[..self.shape.rank()] == self.byte_strides[..self.shape.rank()]
        })
    }

    pub fn sub_slice(mut self, axis: usize, start: i64, length: i64) -> Result<Self, Error> {
        let dimension = *self
            .shape
            .dimensions()
            .get(axis)
            .ok_or(Error::AxisOutOfBounds {
                axis,
                rank: self.shape.rank(),
            })?;
        if start < 0 || length < 0 || start.checked_add(length).is_none_or(|end| end > dimension) {
            return Err(Error::InvalidSubSlice {
                axis,
                start,
                length,
                dimension,
            });
        }
        let delta = i128::from(start)
            .checked_mul(i128::from(self.byte_strides[axis]))
            .ok_or(Error::AddressOverflow)?;
        self.offset_bytes = add_signed(self.offset_bytes, delta)?;
        let mut dimensions = self.shape.dimensions().to_vec();
        dimensions[axis] = length;
        self.shape = Shape::new(self.shape.dtype(), &dimensions)?
            .with_axis_tags(self.shape.axis_tags())?
            .with_partitions(self.shape.partitions())?
            .with_layout(self.shape.layout())?;
        self.validate_reachable_range()?;
        Ok(self)
    }

    /// Creates an immutable view while retaining the original Slice. This is
    /// the form used to derive independently transferred device shards.
    pub fn sub_view(&self, axis: usize, start: i64, length: i64) -> Result<Slice<'_>, Error> {
        Slice {
            storage: Storage::Borrowed(self.storage.bytes()),
            shape: self.shape,
            offset_bytes: self.offset_bytes,
            byte_strides: self.byte_strides,
            byte_order: self.byte_order,
        }
        .sub_slice(axis, start, length)
    }

    pub fn region_view(&self, ranges: &[(usize, i64, i64)]) -> Result<Slice<'_>, Error> {
        let mut view = Slice {
            storage: Storage::Borrowed(self.storage.bytes()),
            shape: self.shape,
            offset_bytes: self.offset_bytes,
            byte_strides: self.byte_strides,
            byte_order: self.byte_order,
        };
        for &(axis, start, length) in ranges {
            view = view.sub_slice(axis, start, length)?;
        }
        Ok(view)
    }

    /// Creates a uniquely borrowed mutable view for shard reconstruction.
    pub fn sub_view_mut(
        &mut self,
        axis: usize,
        start: i64,
        length: i64,
    ) -> Result<Slice<'_>, Error> {
        let shape = self.shape;
        let offset_bytes = self.offset_bytes;
        let byte_strides = self.byte_strides;
        let byte_order = self.byte_order;
        Slice {
            storage: Storage::BorrowedMut(self.storage.bytes_mut().ok_or(Error::ImmutableStorage)?),
            shape,
            offset_bytes,
            byte_strides,
            byte_order,
        }
        .sub_slice(axis, start, length)
    }

    pub fn region_view_mut(&mut self, ranges: &[(usize, i64, i64)]) -> Result<Slice<'_>, Error> {
        let shape = self.shape;
        let offset_bytes = self.offset_bytes;
        let byte_strides = self.byte_strides;
        let byte_order = self.byte_order;
        let mut view = Slice {
            storage: Storage::BorrowedMut(self.storage.bytes_mut().ok_or(Error::ImmutableStorage)?),
            shape,
            offset_bytes,
            byte_strides,
            byte_order,
        };
        for &(axis, start, length) in ranges {
            view = view.sub_slice(axis, start, length)?;
        }
        Ok(view)
    }

    pub fn permute(mut self, permutation: &[usize]) -> Result<Self, Error> {
        let shape = self.shape.permuted(permutation)?;
        let old = self.byte_strides;
        for (new_axis, &old_axis) in permutation.iter().enumerate() {
            self.byte_strides[new_axis] = old[old_axis];
        }
        self.shape = shape;
        self.validate_reachable_range()?;
        Ok(self)
    }

    pub fn reverse(mut self, axis: usize) -> Result<Self, Error> {
        let dimension = *self
            .shape
            .dimensions()
            .get(axis)
            .ok_or(Error::AxisOutOfBounds {
                axis,
                rank: self.shape.rank(),
            })?;
        if dimension > 0 {
            let delta = i128::from(dimension - 1)
                .checked_mul(i128::from(self.byte_strides[axis]))
                .ok_or(Error::AddressOverflow)?;
            self.offset_bytes = add_signed(self.offset_bytes, delta)?;
        }
        self.byte_strides[axis] = self.byte_strides[axis]
            .checked_neg()
            .ok_or(Error::AddressOverflow)?;
        self.validate_reachable_range()?;
        Ok(self)
    }

    pub fn items<T: Element>(&self) -> Result<&[T], Error> {
        require_element_type::<T>(self.dtype())?;
        self.require_contiguous_native()?;
        let bytes = self.contiguous_bytes()?;
        require_alignment::<T>(bytes.as_ptr())?;
        if T::DTYPE == DType::Bool && bytes.iter().any(|byte| *byte > 1) {
            return Err(Error::InvalidBooleanStorage);
        }
        // SAFETY: dtype, byte count, alignment, endian, and bool validity were
        // checked; Element is sealed to plain tensor representations.
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr().cast(),
                bytes.len() / std::mem::size_of::<T>(),
            )
        })
    }

    pub fn items_mut<T: Element>(&mut self) -> Result<&mut [T], Error> {
        require_element_type::<T>(self.dtype())?;
        self.require_contiguous_native()?;
        let start = self.offset_bytes;
        let length = self.shape.byte_count()?;
        let bytes = self
            .storage
            .bytes_mut()
            .ok_or(Error::ImmutableStorage)?
            .get_mut(start..start + length)
            .ok_or(Error::AddressOutOfBounds)?;
        require_alignment::<T>(bytes.as_ptr())?;
        if T::DTYPE == DType::Bool && bytes.iter().any(|byte| *byte > 1) {
            return Err(Error::InvalidBooleanStorage);
        }
        // SAFETY: identical checks to items(), with unique mutable storage.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(
                bytes.as_mut_ptr().cast(),
                bytes.len() / std::mem::size_of::<T>(),
            )
        })
    }

    pub fn to_contiguous(&self) -> Result<Slice<'static>, Error> {
        let mut result = Slice::alloc(self.shape)?;
        result.copy_from(self)?;
        Ok(result)
    }

    pub fn copy_from(&mut self, source: &Slice<'_>) -> Result<(), Error> {
        if self.shape != source.shape {
            return Err(Error::ShapeMismatch {
                expected: self.shape,
                actual: source.shape,
            });
        }
        if self.byte_order != source.byte_order {
            return Err(Error::EndianMismatch);
        }
        let width = self.dtype().byte_width();
        let mut dense = vec![0u8; self.shape.byte_count()?];
        for_each_offset(source, |linear, offset| {
            dense[linear..linear + width]
                .copy_from_slice(&source.storage.bytes()[offset..offset + width]);
        })?;
        let destination_offsets = collect_offsets(self)?;
        let bytes = self.storage.bytes_mut().ok_or(Error::ImmutableStorage)?;
        for (linear, offset) in destination_offsets {
            bytes[offset..offset + width].copy_from_slice(&dense[linear..linear + width]);
        }
        Ok(())
    }

    pub fn contiguous_bytes(&self) -> Result<&[u8], Error> {
        if !self.is_contiguous() {
            return Err(Error::NonContiguous);
        }
        let length = self.shape.byte_count()?;
        self.storage
            .bytes()
            .get(self.offset_bytes..self.offset_bytes + length)
            .ok_or(Error::AddressOutOfBounds)
    }

    /// Returns the exact dense storage span for an explicit serialization or
    /// FFI boundary. Logical tensor code should prefer typed accessors.
    pub fn contiguous_bytes_mut(&mut self) -> Result<&mut [u8], Error> {
        if !self.is_contiguous() {
            return Err(Error::NonContiguous);
        }
        let start = self.offset_bytes;
        let length = self.shape.byte_count()?;
        self.storage
            .bytes_mut()
            .ok_or(Error::ImmutableStorage)?
            .get_mut(start..start + length)
            .ok_or(Error::AddressOutOfBounds)
    }

    pub fn data_pointer(&self) -> Result<*const u8, Error> {
        let bytes = self.storage.bytes();
        if self.offset_bytes > bytes.len() {
            return Err(Error::AddressOutOfBounds);
        }
        // SAFETY: offset is within or one-past the live allocation.
        Ok(unsafe { bytes.as_ptr().add(self.offset_bytes) })
    }

    fn require_contiguous_native(&self) -> Result<(), Error> {
        if !self.is_contiguous() {
            return Err(Error::NonContiguous);
        }
        if !self.is_native_endian() {
            return Err(Error::NonNativeEndian);
        }
        Ok(())
    }

    fn validate_reachable_range(&self) -> Result<(), Error> {
        if self.offset_bytes > self.storage.bytes().len() {
            return Err(Error::AddressOutOfBounds);
        }
        if self.shape.element_count()? == 0 {
            return Ok(());
        }
        let mut minimum = self.offset_bytes as i128;
        let mut maximum = minimum;
        for axis in 0..self.shape.rank() {
            let extent = i128::from(self.shape.dimensions()[axis] - 1)
                .checked_mul(i128::from(self.byte_strides[axis]))
                .ok_or(Error::AddressOverflow)?;
            if extent < 0 {
                minimum = minimum.checked_add(extent).ok_or(Error::AddressOverflow)?;
            } else {
                maximum = maximum.checked_add(extent).ok_or(Error::AddressOverflow)?;
            }
        }
        maximum = maximum
            .checked_add(self.dtype().byte_width() as i128)
            .ok_or(Error::AddressOverflow)?;
        if minimum < 0 || maximum > self.storage.bytes().len() as i128 {
            return Err(Error::AddressOutOfBounds);
        }
        Ok(())
    }
}

fn dense_byte_strides(shape: Shape) -> Result<[i64; MAX_RANK], Error> {
    let mut strides = [0i64; MAX_RANK];
    let mut running =
        i64::try_from(shape.dtype().byte_width()).map_err(|_| Error::AddressOverflow)?;
    for &axis in shape.layout().minor_to_major() {
        let axis = axis as usize;
        strides[axis] = running;
        running = running
            .checked_mul(shape.dimensions()[axis])
            .ok_or(Error::AddressOverflow)?;
    }
    Ok(strides)
}

fn add_signed(base: usize, delta: i128) -> Result<usize, Error> {
    let value = (base as i128)
        .checked_add(delta)
        .ok_or(Error::AddressOverflow)?;
    usize::try_from(value).map_err(|_| Error::AddressOverflow)
}

fn collect_offsets(slice: &Slice<'_>) -> Result<Vec<(usize, usize)>, Error> {
    let mut result = Vec::with_capacity(slice.shape.element_count()?);
    for_each_offset(slice, |linear, offset| result.push((linear, offset)))?;
    Ok(result)
}

fn for_each_offset(slice: &Slice<'_>, mut callback: impl FnMut(usize, usize)) -> Result<(), Error> {
    let elements = slice.shape.element_count()?;
    if elements == 0 {
        return Ok(());
    }
    let width = slice.dtype().byte_width();
    for linear_index in 0..elements {
        let mut remainder = linear_index;
        let mut offset = slice.offset_bytes as i128;
        for axis in (0..slice.shape.rank()).rev() {
            let dimension = usize::try_from(slice.shape.dimensions()[axis])
                .map_err(|_| Error::AddressOverflow)?;
            let coordinate = remainder % dimension;
            remainder /= dimension;
            offset = offset
                .checked_add((coordinate as i128) * i128::from(slice.byte_strides[axis]))
                .ok_or(Error::AddressOverflow)?;
        }
        let offset = usize::try_from(offset).map_err(|_| Error::AddressOverflow)?;
        callback(linear_index * width, offset);
    }
    Ok(())
}

fn require_alignment<T>(pointer: *const u8) -> Result<(), Error> {
    let required = std::mem::align_of::<T>();
    if pointer.addr().is_multiple_of(required) {
        Ok(())
    } else {
        Err(Error::MisalignedStorage { required })
    }
}

fn require_element_type<T: Element>(actual: DType) -> Result<(), Error> {
    if T::DTYPE == actual {
        Ok(())
    } else {
        Err(Error::DTypeMismatch {
            expected: T::DTYPE,
            actual,
        })
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Rust types with the exact in-memory representation of an NML tensor dtype.
/// The trait is sealed so downstream code cannot assert an unsound layout.
pub trait Element: sealed::Sealed + Copy + 'static {
    const DTYPE: DType;
}

macro_rules! elements {
    ($($type:ty => $dtype:expr),+ $(,)?) => {$ (
        impl sealed::Sealed for $type {}
        impl Element for $type {
            const DTYPE: DType = $dtype;
        }
    )+ };
}

elements! {
    bool => DType::Bool,
    i8 => DType::I8,
    i16 => DType::I16,
    i32 => DType::I32,
    i64 => DType::I64,
    u8 => DType::U8,
    u16 => DType::U16,
    u32 => DType::U32,
    u64 => DType::U64,
    F16 => DType::F16,
    BFloat16 => DType::Bf16,
    f32 => DType::F32,
    f64 => DType::F64,
    Complex64 => DType::C64,
    Complex128 => DType::C128,
}

#[derive(Debug)]
pub enum Error {
    Shape(ShapeError),
    InvalidAlignment(usize),
    ByteLength {
        expected: usize,
        actual: usize,
    },
    DTypeMismatch {
        expected: DType,
        actual: DType,
    },
    ShapeMismatch {
        expected: Shape,
        actual: Shape,
    },
    AxisOutOfBounds {
        axis: usize,
        rank: usize,
    },
    InvalidSubSlice {
        axis: usize,
        start: i64,
        length: i64,
        dimension: i64,
    },
    AddressOverflow,
    AddressOutOfBounds,
    MisalignedStorage {
        required: usize,
    },
    ImmutableStorage,
    NonContiguous,
    NonNativeEndian,
    EndianMismatch,
    InvalidBooleanStorage,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(error) => error.fmt(formatter),
            Self::InvalidAlignment(alignment) => write!(formatter, "invalid alignment {alignment}"),
            Self::ByteLength { expected, actual } => {
                write!(
                    formatter,
                    "tensor requires {expected} bytes, received {actual}"
                )
            }
            Self::DTypeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected:?} elements, received {actual:?}"
                )
            }
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "shape mismatch: expected {expected:?}, received {actual:?}"
                )
            }
            Self::AxisOutOfBounds { axis, rank } => {
                write!(formatter, "axis {axis} is outside rank {rank}")
            }
            Self::InvalidSubSlice {
                axis,
                start,
                length,
                dimension,
            } => write!(
                formatter,
                "slice axis {axis} range {start}..{} exceeds dimension {dimension}",
                start.saturating_add(*length)
            ),
            Self::AddressOverflow => {
                formatter.write_str("tensor view address arithmetic overflowed")
            }
            Self::AddressOutOfBounds => {
                formatter.write_str("tensor view reaches outside its backing storage")
            }
            Self::MisalignedStorage { required } => {
                write!(
                    formatter,
                    "tensor storage is not aligned to {required} bytes"
                )
            }
            Self::ImmutableStorage => formatter.write_str("tensor storage is immutable"),
            Self::NonContiguous => {
                formatter.write_str("operation requires a contiguous tensor view")
            }
            Self::NonNativeEndian => {
                formatter.write_str("operation requires native-endian tensor storage")
            }
            Self::EndianMismatch => formatter.write_str("source and destination byte order differ"),
            Self::InvalidBooleanStorage => {
                formatter.write_str("boolean tensor contains a byte other than 0 or 1")
            }
        }
    }
}

impl StdError for Error {}

impl From<ShapeError> for Error {
    fn from(error: ShapeError) -> Self {
        Self::Shape(error)
    }
}
