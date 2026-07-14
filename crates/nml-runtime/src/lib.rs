//! ZML-shaped persistent buffer and executable lifecycle.
//!
//! A public `Buffer` is one logical tensor. PJRT shards, transfer guards,
//! device identities, binding slots, and ownership counters remain private.

#![forbid(unsafe_op_in_unsafe_fn)]

use nml_ir::{InputKind, Program};
use nml_pjrt::{Client, Device, LoadedExecutable};
use nml_tensor::Slice;
use nml_types::Shape;
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Cpu,
    Cuda,
}

/// A loaded CPU or CUDA PJRT platform.
pub struct Platform {
    backend: Backend,
    client: Client,
}

impl Platform {
    /// Creates ZML's four-device CPU reference/performance topology.
    pub fn cpu() -> Result<Self, Error> {
        Self::cpu_with_devices(4)
    }

    pub fn cpu_with_devices(device_count: usize) -> Result<Self, Error> {
        if device_count == 0 || device_count > i64::MAX as usize {
            return Err(Error::InvalidCpuDeviceCount(device_count));
        }
        let plugin = nml_pjrt_cpu::load().map_err(|error| Error::Platform(error.to_string()))?;
        let client = plugin
            .create_client_with_options(&[nml_pjrt::NamedValue::Int64 {
                name: "cpu_device_count",
                value: device_count as i64,
            }])
            .map_err(Error::Pjrt)?;
        Ok(Self {
            backend: Backend::Cpu,
            client,
        })
    }

    /// Initializes the process-global CUDA runtime.
    ///
    /// # Safety
    /// This must be called before other application threads can read or mutate
    /// process environment variables, as required by the CUDA PJRT loader.
    #[cfg(target_os = "linux")]
    pub unsafe fn cuda() -> Result<Self, Error> {
        // SAFETY: the caller upholds the process-wide initialization contract.
        let runtime = unsafe { nml_pjrt_cuda::Runtime::load() }
            .map_err(|error| Error::Platform(error.to_string()))?;
        let client = runtime
            .create_client(nml_pjrt_cuda::ClientOptions::default())
            .map_err(|error| Error::Platform(error.to_string()))?;
        Ok(Self {
            backend: Backend::Cuda,
            client,
        })
    }

    pub fn name(&self) -> &'static str {
        match self.backend {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
        }
    }

    pub fn device_count(&self) -> Result<usize, Error> {
        self.client.device_count().map_err(Error::Pjrt)
    }

    pub fn upload(
        &self,
        slice: &Slice<'_>,
        sharding: Sharding,
        memory: Memory,
    ) -> Result<Buffer, Error> {
        sharding.validate(slice.shape())?;
        let devices = self.client.devices().map_err(Error::Pjrt)?;
        let shard_count = sharding.physical_shard_count(devices.len())?;
        let mut shards = Vec::with_capacity(shard_count);
        for (index, device) in devices.iter().take(shard_count).enumerate() {
            let ranges = sharding.ranges(slice.shape(), index)?;
            let view = slice.region_view(&ranges)?;
            let selected_memory = select_memory(device, memory)?;
            let transfer = self
                .client
                .buffer_from_host_in(&view, device, selected_memory.as_ref())
                .map_err(Error::Pjrt)?;
            shards.push(transfer.wait().map_err(Error::Pjrt)?);
        }
        Ok(Buffer {
            storage: Arc::new(BufferStorage { shards }),
            shape: slice.shape(),
            sharding,
            backend: self.backend,
            platform_id: self.client.as_raw_identity(),
            memory,
        })
    }

    /// Streams checkpoint bytes through CUDA's reusable mapped DMA lanes.
    /// CPU deliberately reads one aligned tensor and uses the ordinary
    /// immutable-host transfer path.
    #[doc(hidden)]
    pub fn upload_checkpoint_from(
        &self,
        shape: Shape,
        sharding: Sharding,
        memory: Memory,
        staging_buffers: usize,
        chunk_bytes: usize,
        mut read: impl FnMut(usize, &mut [u8]) -> std::io::Result<()>,
    ) -> Result<Buffer, Error> {
        if staging_buffers == 0 || chunk_bytes == 0 {
            return Err(Error::InvalidStaging {
                buffers: staging_buffers,
                chunk_bytes,
            });
        }
        sharding.validate(shape)?;
        let logical_bytes = shape.byte_count().map_err(nml_tensor::Error::from)?;
        if self.backend == Backend::Cpu || logical_bytes == 0 {
            let mut slice = Slice::alloc(shape)?;
            if logical_bytes != 0 {
                read(0, slice.contiguous_bytes_mut()?).map_err(Error::Io)?;
            }
            return self.upload(&slice, sharding, memory);
        }
        let devices = self.client.devices().map_err(Error::Pjrt)?;
        let shard_count = sharding.physical_shard_count(devices.len())?;
        if sharding.replicated || sharding.partitions.is_empty() {
            let selected_memories = devices
                .iter()
                .take(shard_count)
                .map(|device| match select_memory(device, memory)? {
                    Some(memory) => Ok(memory),
                    None => device.default_memory().map_err(Error::Pjrt),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let managers = selected_memories
                .iter()
                .map(|memory| {
                    self.client
                        .create_async_transfer_manager(&[shape], memory)
                        .map_err(Error::Pjrt)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let lane_bytes = chunk_bytes.min(logical_bytes).max(1);
            let mut lanes = (0..staging_buffers)
                .map(|_| {
                    self.client
                        .owned_dma_buffer(lane_bytes)
                        .map_err(Error::Pjrt)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pending = (0..staging_buffers)
                .map(|_| Vec::<nml_pjrt::OwnedDmaTransfer<'_>>::new())
                .collect::<Vec<_>>();
            for (chunk_index, byte_offset) in (0..logical_bytes).step_by(chunk_bytes).enumerate() {
                let lane = chunk_index % staging_buffers;
                for transfer in pending[lane].drain(..) {
                    transfer.wait().map_err(Error::Pjrt)?;
                }
                let length = chunk_bytes.min(logical_bytes - byte_offset);
                let staging =
                    Arc::get_mut(&mut lanes[lane]).expect("a completed DMA lane has one owner");
                read(byte_offset, &mut staging.bytes_mut()[..length]).map_err(Error::Io)?;
                let offset = i64::try_from(byte_offset).map_err(|_| Error::ByteCountOverflow)?;
                for manager in &managers {
                    pending[lane].push(
                        manager
                            .transfer_owned(
                                0,
                                Arc::clone(&lanes[lane]),
                                length,
                                offset,
                                byte_offset + length == logical_bytes,
                            )
                            .map_err(Error::Pjrt)?,
                    );
                }
            }
            for transfer in pending.into_iter().flatten() {
                transfer.wait().map_err(Error::Pjrt)?;
            }
            let shards = managers
                .iter()
                .map(|manager| manager.retrieve_buffer(0).map_err(Error::Pjrt))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Buffer {
                storage: Arc::new(BufferStorage { shards }),
                shape,
                sharding,
                backend: self.backend,
                platform_id: self.client.as_raw_identity(),
                memory,
            });
        }

        let shard_shape = sharding.shard_shape(shape)?;
        let shard_bytes = shard_shape.byte_count().map_err(nml_tensor::Error::from)?;
        let mut shards = Vec::with_capacity(shard_count);
        for (index, device) in devices.iter().take(shard_count).enumerate() {
            let selected_memory = match select_memory(device, memory)? {
                Some(memory) => memory,
                None => device.default_memory().map_err(Error::Pjrt)?,
            };
            let manager = self
                .client
                .create_async_transfer_manager(&[shard_shape], &selected_memory)
                .map_err(Error::Pjrt)?;
            let lane_bytes = chunk_bytes.min(shard_bytes).max(1);
            let mut lanes = (0..staging_buffers)
                .map(|_| {
                    self.client
                        .owned_dma_buffer(lane_bytes)
                        .map_err(Error::Pjrt)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut pending: Vec<Option<nml_pjrt::OwnedDmaTransfer<'_>>> =
                (0..staging_buffers).map(|_| None).collect();
            let ranges = sharding.ranges(shape, index)?;
            let mut destination_offset = 0usize;
            let mut transfer_index = 0usize;
            for_each_shard_span(shape, &ranges, |source_offset, span_length| {
                let mut consumed = 0usize;
                while consumed < span_length {
                    let length = chunk_bytes.min(span_length - consumed);
                    let lane = transfer_index % staging_buffers;
                    transfer_index += 1;
                    if let Some(transfer) = pending[lane].take() {
                        transfer.wait().map_err(Error::Pjrt)?;
                    }
                    let staging =
                        Arc::get_mut(&mut lanes[lane]).expect("a completed DMA lane has one owner");
                    read(source_offset + consumed, &mut staging.bytes_mut()[..length])
                        .map_err(Error::Io)?;
                    let offset =
                        i64::try_from(destination_offset).map_err(|_| Error::ByteCountOverflow)?;
                    destination_offset = destination_offset
                        .checked_add(length)
                        .ok_or(Error::ByteCountOverflow)?;
                    pending[lane] = Some(
                        manager
                            .transfer_owned(
                                0,
                                Arc::clone(&lanes[lane]),
                                length,
                                offset,
                                destination_offset == shard_bytes,
                            )
                            .map_err(Error::Pjrt)?,
                    );
                    consumed += length;
                }
                Ok(())
            })?;
            for transfer in pending.into_iter().flatten() {
                transfer.wait().map_err(Error::Pjrt)?;
            }
            if destination_offset != shard_bytes {
                return Err(Error::ByteCountOverflow);
            }
            shards.push(manager.retrieve_buffer(0).map_err(Error::Pjrt)?);
        }
        Ok(Buffer {
            storage: Arc::new(BufferStorage { shards }),
            shape,
            sharding,
            backend: self.backend,
            platform_id: self.client.as_raw_identity(),
            memory,
        })
    }

    pub fn compile(&self, program: &Program) -> Result<Exe, Error> {
        let devices = self.client.devices().map_err(Error::Pjrt)?;
        if devices.is_empty() {
            return Err(Error::NoDevices);
        }
        let backend = match self.backend {
            Backend::Cpu => nml_xla::Backend::Cpu,
            Backend::Cuda => nml_xla::Backend::Cuda,
        };
        // A platform is the physical replica mesh.  Tiled partitioning will
        // add a distinct logical mesh contract; until then every addressable
        // device executes the same program and receives one buffer shard.
        let device_ids = devices
            .iter()
            .map(|device| device.id().map_err(Error::Pjrt))
            .collect::<Result<Vec<_>, _>>()?;
        let replicas = u32::try_from(device_ids.len()).map_err(|_| Error::DeviceCount {
            required: device_ids.len(),
            available: u32::MAX as usize,
        })?;
        let options = nml_xla::CompileOptions::new(
            replicas,
            1,
            device_ids,
            nml_xla::Partitioner::Shardy,
            backend,
        )
        .map_err(Error::Xla)?;
        let loaded =
            nml_compiler::compile(&self.client, program, &options).map_err(Error::Compiler)?;
        Exe::new(
            self.backend,
            self.client.as_raw_identity(),
            loaded,
            program,
            devices.len(),
        )
    }
}

fn select_memory(device: &Device, requested: Memory) -> Result<Option<nml_pjrt::Memory>, Error> {
    let expected = match requested {
        Memory::Default => return Ok(None),
        Memory::Device => "device",
        Memory::HostPinned => "pinned_host",
        Memory::HostUnpinned => "unpinned_host",
    };
    for memory in device.addressable_memories().map_err(Error::Pjrt)? {
        if memory.kind().map_err(Error::Pjrt)? == expected {
            return Ok(Some(memory));
        }
    }
    Err(Error::UnsupportedMemory {
        requested,
        platform: device
            .string_attribute("device_kind")
            .map_err(Error::Pjrt)?
            .unwrap_or_else(|| "unknown device".to_owned()),
    })
}

/// ZML's stable memory selection vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Memory {
    Default,
    Device,
    HostPinned,
    HostUnpinned,
}

/// Logical tensor placement over the platform's canonical device order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sharding {
    partitions: Vec<usize>,
    replicated: bool,
}

impl Sharding {
    /// Places the complete logical tensor on the first canonical device.
    pub fn single() -> Self {
        Self {
            partitions: Vec::new(),
            replicated: false,
        }
    }

    pub fn replicated() -> Self {
        Self {
            partitions: Vec::new(),
            replicated: true,
        }
    }

    /// Partitions each logical axis by the corresponding positive factor.
    pub fn tiled(partitions: &[usize]) -> Result<Self, Error> {
        if partitions.contains(&0) {
            return Err(Error::InvalidSharding("partition factors must be positive"));
        }
        let _ = partitions.iter().try_fold(1usize, |count, partition| {
            count
                .checked_mul(*partition)
                .ok_or(Error::InvalidSharding("shard count overflows"))
        })?;
        Ok(Self {
            partitions: partitions.to_vec(),
            replicated: false,
        })
    }

    pub fn is_replicated(&self) -> bool {
        self.replicated
    }

    pub fn shard_count(&self) -> usize {
        if self.replicated {
            1
        } else {
            self.partitions.iter().product()
        }
    }

    fn validate(&self, shape: Shape) -> Result<(), Error> {
        if self.replicated {
            return Ok(());
        }
        if self.partitions.is_empty() {
            return Ok(());
        }
        if self.partitions.len() != shape.rank() {
            return Err(Error::ShardingRank {
                expected: shape.rank(),
                actual: self.partitions.len(),
            });
        }
        for (axis, (&dimension, &parts)) in
            shape.dimensions().iter().zip(&self.partitions).enumerate()
        {
            if dimension % parts as i64 != 0 {
                return Err(Error::UnevenSharding {
                    axis,
                    dimension,
                    partitions: parts,
                });
            }
        }
        Ok(())
    }

    fn physical_shard_count(&self, available: usize) -> Result<usize, Error> {
        let required = if self.replicated {
            available
        } else if self.partitions.is_empty() {
            1
        } else {
            self.shard_count()
        };
        if required == 0 || required > available {
            Err(Error::DeviceCount {
                required,
                available,
            })
        } else {
            Ok(required)
        }
    }

    fn shard_shape(&self, shape: Shape) -> Result<Shape, Error> {
        if self.replicated || self.partitions.is_empty() {
            return Ok(shape);
        }
        let dimensions = shape
            .dimensions()
            .iter()
            .zip(&self.partitions)
            .map(|(dimension, partitions)| *dimension / *partitions as i64)
            .collect::<Vec<_>>();
        Shape::new(shape.dtype(), &dimensions)
            .map_err(nml_tensor::Error::from)
            .map_err(Error::Tensor)
    }

    fn ranges(&self, shape: Shape, shard: usize) -> Result<Vec<(usize, i64, i64)>, Error> {
        if self.replicated {
            return Ok(Vec::new());
        }
        if self.partitions.is_empty() {
            return Ok(Vec::new());
        }
        let mut remainder = shard;
        let mut coordinates = vec![0usize; self.partitions.len()];
        for axis in (0..self.partitions.len()).rev() {
            coordinates[axis] = remainder % self.partitions[axis];
            remainder /= self.partitions[axis];
        }
        Ok(coordinates
            .iter()
            .enumerate()
            .filter_map(|(axis, coordinate)| {
                let parts = self.partitions[axis];
                if parts == 1 {
                    None
                } else {
                    let length = shape.dimensions()[axis] / parts as i64;
                    Some((axis, *coordinate as i64 * length, length))
                }
            })
            .collect())
    }
}

fn for_each_shard_span(
    shape: Shape,
    ranges: &[(usize, i64, i64)],
    mut visit: impl FnMut(usize, usize) -> Result<(), Error>,
) -> Result<(), Error> {
    let rank = shape.rank();
    let width = shape.dtype().byte_width();
    if rank == 0 {
        return visit(0, width);
    }
    let dimensions = shape
        .dimensions()
        .iter()
        .map(|dimension| usize::try_from(*dimension).map_err(|_| Error::ByteCountOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    let mut starts = vec![0usize; rank];
    let mut lengths = dimensions.clone();
    for &(axis, start, length) in ranges {
        starts[axis] = usize::try_from(start).map_err(|_| Error::ByteCountOverflow)?;
        lengths[axis] = usize::try_from(length).map_err(|_| Error::ByteCountOverflow)?;
    }
    if lengths.contains(&0) {
        return Ok(());
    }
    let row_count = lengths[..rank - 1]
        .iter()
        .try_fold(1usize, |count, length| {
            count.checked_mul(*length).ok_or(Error::ByteCountOverflow)
        })?;
    let span_length = lengths[rank - 1]
        .checked_mul(width)
        .ok_or(Error::ByteCountOverflow)?;
    for row in 0..row_count {
        let mut remainder = row;
        let mut coordinates = starts.clone();
        for axis in (0..rank - 1).rev() {
            coordinates[axis] += remainder % lengths[axis];
            remainder /= lengths[axis];
        }
        coordinates[rank - 1] = starts[rank - 1];
        let linear = coordinates.iter().zip(&dimensions).try_fold(
            0usize,
            |linear, (coordinate, dimension)| {
                linear
                    .checked_mul(*dimension)
                    .and_then(|linear| linear.checked_add(*coordinate))
                    .ok_or(Error::ByteCountOverflow)
            },
        )?;
        let source_offset = linear.checked_mul(width).ok_or(Error::ByteCountOverflow)?;
        visit(source_offset, span_length)?;
    }
    Ok(())
}

/// One logical, possibly sharded, persistent device tensor.
#[derive(Clone)]
pub struct Buffer {
    storage: Arc<BufferStorage>,
    shape: Shape,
    sharding: Sharding,
    backend: Backend,
    platform_id: usize,
    memory: Memory,
}

struct BufferStorage {
    shards: Vec<nml_pjrt::Buffer>,
}

impl Buffer {
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    pub fn sharding(&self) -> &Sharding {
        &self.sharding
    }

    pub const fn memory(&self) -> Memory {
        self.memory
    }

    pub fn platform_name(&self) -> &'static str {
        match self.backend {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
        }
    }

    pub fn is_ready(&self) -> Result<bool, Error> {
        for shard in &self.storage.shards {
            if !shard
                .ready_event()
                .map_err(Error::Pjrt)?
                .is_ready()
                .map_err(Error::Pjrt)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn wait(&self) -> Result<(), Error> {
        for shard in &self.storage.shards {
            shard
                .ready_event()
                .map_err(Error::Pjrt)?
                .wait()
                .map_err(Error::Pjrt)?;
        }
        Ok(())
    }

    pub fn byte_count(&self) -> Result<usize, Error> {
        self.storage.shards.iter().try_fold(0usize, |total, shard| {
            total
                .checked_add(shard.on_device_size_in_bytes().map_err(Error::Pjrt)?)
                .ok_or(Error::ByteCountOverflow)
        })
    }

    /// Allocates distinct physical storage with the same logical placement.
    /// Cloning a `Buffer` shares storage; this operation deliberately does not.
    pub fn copy(&self) -> Result<Self, Error> {
        let shards = self
            .storage
            .shards
            .iter()
            .map(|shard| {
                let memory = shard.memory().map_err(Error::Pjrt)?;
                shard.copy_to_memory(&memory).map_err(Error::Pjrt)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            storage: Arc::new(BufferStorage { shards }),
            shape: self.shape,
            sharding: self.sharding.clone(),
            backend: self.backend,
            platform_id: self.platform_id,
            memory: self.memory,
        })
    }

    pub fn to_slice(&self) -> Result<Slice<'static>, Error> {
        let mut result = Slice::alloc(self.shape)?;
        if self.sharding.replicated {
            self.storage.shards[0]
                .to_slice(&mut result)
                .map_err(Error::Pjrt)?;
            return Ok(result);
        }
        for (index, shard) in self.storage.shards.iter().enumerate() {
            let source = shard.to_slice_alloc().map_err(Error::Pjrt)?;
            let ranges = self.sharding.ranges(self.shape, index)?;
            let mut destination = result.region_view_mut(&ranges)?;
            destination.copy_from(&source)?;
        }
        Ok(result)
    }

    pub fn delete(self) -> Result<(), Error> {
        if !self.is_uniquely_owned() {
            return Err(Error::DeletionRequiresUniqueOwnership);
        }
        for shard in &self.storage.shards {
            shard.delete().map_err(Error::Pjrt)?;
        }
        Ok(())
    }

    pub fn is_uniquely_owned(&self) -> bool {
        Arc::strong_count(&self.storage) == 1
    }

    fn raw_shards(&self) -> &[nml_pjrt::Buffer] {
        &self.storage.shards
    }
}

/// Structural mapping from symbolic tensors to persistent buffers.
///
/// The derive macro generates companion storage privately in the model's own
/// module; NML exposes only this one trait and the `Bufferized<T>` mapping.
pub trait NmlStruct {
    type Buffers;

    fn visit_tensors(&self, prefix: &str, visitor: &mut dyn FnMut(&str, nml_ir::Tensor));

    fn visit_buffers(buffers: &Self::Buffers, prefix: &str, visitor: &mut dyn FnMut(&str, &Buffer));

    fn bufferize<E>(
        &self,
        prefix: &str,
        resolve: &mut impl FnMut(&str, nml_ir::Tensor) -> Result<Buffer, E>,
    ) -> Result<Self::Buffers, E>;
}

pub type Bufferized<T> = <T as NmlStruct>::Buffers;

/// A compiled executable with named, reusable argument bindings.
pub struct Exe {
    backend: Backend,
    platform_id: usize,
    loaded: LoadedExecutable,
    device_count: usize,
    inputs: Vec<Binding>,
    outputs: Vec<OutputBinding>,
}

#[derive(Clone)]
struct Binding {
    name: String,
    shape: Shape,
    kind: InputKind,
}

#[derive(Clone)]
struct OutputBinding {
    name: String,
    shape: Shape,
    alias_input: Option<usize>,
}

impl Exe {
    fn new(
        backend: Backend,
        platform_id: usize,
        loaded: LoadedExecutable,
        program: &Program,
        device_count: usize,
    ) -> Result<Self, Error> {
        let inputs = program
            .inputs()
            .map(|(name, shape, kind)| Binding {
                name: name.to_owned(),
                shape,
                kind,
            })
            .collect();
        let outputs = program
            .outputs()
            .zip(program.output_aliases())
            .map(|((name, shape), alias_input)| OutputBinding {
                name: name.to_owned(),
                shape,
                alias_input,
            })
            .collect();
        Ok(Self {
            backend,
            platform_id,
            loaded,
            device_count,
            inputs,
            outputs,
        })
    }

    pub fn args(&self) -> exe::Arguments<'_> {
        exe::Arguments {
            executable: self,
            slots: vec![None; self.inputs.len()],
            baked: vec![false; self.inputs.len()],
        }
    }

    pub fn results(&self, buffers: Vec<Buffer>) -> Result<exe::Results, Error> {
        if buffers.len() != self.outputs.len() {
            return Err(Error::ResultCount {
                expected: self.outputs.len(),
                actual: buffers.len(),
            });
        }
        Ok(exe::Results {
            names: self
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect(),
            buffers,
        })
    }
}

pub mod exe {
    use super::{Buffer, Error, Exe, InputKind};

    pub struct Arguments<'exe> {
        pub(super) executable: &'exe Exe,
        pub(super) slots: Vec<Option<Buffer>>,
        pub(super) baked: Vec<bool>,
    }

    impl Arguments<'_> {
        pub fn set(&mut self, name: &str, buffer: Buffer) -> Result<&mut Self, Error> {
            let index = self
                .executable
                .inputs
                .iter()
                .position(|binding| binding.name == name)
                .ok_or_else(|| Error::UnknownArgument(name.to_owned()))?;
            let binding = &self.executable.inputs[index];
            if binding.shape != buffer.shape {
                return Err(Error::ArgumentShape {
                    name: name.to_owned(),
                    expected: binding.shape,
                    actual: buffer.shape,
                });
            }
            if buffer.backend != self.executable.backend {
                return Err(Error::ArgumentPlatform(name.to_owned()));
            }
            if buffer.platform_id != self.executable.platform_id {
                return Err(Error::ArgumentPlatform(name.to_owned()));
            }
            if self.executable.device_count > 1 && !buffer.sharding.is_replicated() {
                return Err(Error::ArgumentSharding(name.to_owned()));
            }
            if self.baked[index] {
                return Err(Error::BakedArgument(name.to_owned()));
            }
            self.slots[index] = Some(buffer);
            Ok(self)
        }

        pub fn bake(&mut self) -> Result<&mut Self, Error> {
            for (index, binding) in self.executable.inputs.iter().enumerate() {
                if binding.kind == InputKind::Parameter {
                    if self.slots[index].is_none() {
                        return Err(Error::MissingArgument(binding.name.clone()));
                    }
                    self.baked[index] = true;
                }
            }
            Ok(self)
        }

        pub fn call(&mut self) -> Result<Results, Error> {
            for (index, binding) in self.executable.inputs.iter().enumerate() {
                if self.slots[index].is_none() {
                    return Err(Error::MissingArgument(binding.name.clone()));
                }
            }
            for (slot, binding) in self.slots.iter().zip(&self.executable.inputs) {
                let actual = slot.as_ref().expect("checked above").raw_shards().len();
                if actual != self.executable.device_count {
                    return Err(Error::ArgumentShardCount {
                        name: binding.name.clone(),
                        expected: self.executable.device_count,
                        actual,
                    });
                }
            }
            for output in &self.executable.outputs {
                if let Some(input) = output.alias_input {
                    let buffer = self.slots[input].as_ref().expect("checked above");
                    if !buffer.is_uniquely_owned() {
                        return Err(Error::DonationRequiresUniqueOwnership(
                            self.executable.inputs[input].name.clone(),
                        ));
                    }
                    if self.baked[input] {
                        return Err(Error::ParameterDonation(
                            self.executable.inputs[input].name.clone(),
                        ));
                    }
                }
            }
            let output_shardings = self
                .executable
                .outputs
                .iter()
                .map(|output| {
                    output.alias_input.map_or_else(
                        || {
                            if self.executable.device_count == 1 {
                                super::Sharding::single()
                            } else {
                                super::Sharding::replicated()
                            }
                        },
                        |input| {
                            self.slots[input]
                                .as_ref()
                                .expect("checked above")
                                .sharding
                                .clone()
                        },
                    )
                })
                .collect::<Vec<_>>();
            let output_memories = self
                .executable
                .outputs
                .iter()
                .map(|output| {
                    output.alias_input.map_or(super::Memory::Default, |input| {
                        self.slots[input].as_ref().expect("checked above").memory
                    })
                })
                .collect::<Vec<_>>();
            let raw = (0..self.executable.device_count)
                .map(|device| {
                    self.slots
                        .iter()
                        .map(|slot| &slot.as_ref().expect("checked above").raw_shards()[device])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let raw_slices = raw.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let execution = self
                .executable
                .loaded
                .execute(&raw_slices)
                .map_err(Error::Pjrt)?;
            for event in execution.complete {
                event.wait().map_err(Error::Pjrt)?;
            }
            let mut outputs = (0..self.executable.outputs.len())
                .map(|_| Vec::with_capacity(self.executable.device_count))
                .collect::<Vec<_>>();
            for device_outputs in execution.outputs {
                if device_outputs.len() != outputs.len() {
                    return Err(Error::ResultCount {
                        expected: outputs.len(),
                        actual: device_outputs.len(),
                    });
                }
                for (output, buffer) in outputs.iter_mut().zip(device_outputs) {
                    output.push(buffer);
                }
            }
            let buffers = outputs
                .into_iter()
                .zip(
                    self.executable
                        .outputs
                        .iter()
                        .zip(output_shardings)
                        .zip(output_memories),
                )
                .map(|(shards, ((output, sharding), memory))| Buffer {
                    storage: std::sync::Arc::new(super::BufferStorage { shards }),
                    shape: output.shape,
                    sharding,
                    backend: self.executable.backend,
                    platform_id: self.executable.platform_id,
                    memory,
                })
                .collect::<Vec<_>>();
            for output in &self.executable.outputs {
                if let Some(input) = output.alias_input {
                    self.slots[input] = None;
                }
            }
            self.executable.results(buffers)
        }
    }

    pub struct Results {
        pub(super) names: Vec<String>,
        pub(super) buffers: Vec<Buffer>,
    }

    impl Results {
        pub fn get(&self, name: &str) -> Option<&Buffer> {
            self.names
                .iter()
                .position(|candidate| candidate == name)
                .map(|index| &self.buffers[index])
        }

        pub fn into_buffers(self) -> Vec<Buffer> {
            self.buffers
        }
    }
}

#[derive(Debug)]
pub enum Error {
    InvalidCpuDeviceCount(usize),
    Platform(String),
    Pjrt(nml_pjrt::Error),
    Xla(nml_xla::Error),
    Compiler(nml_compiler::Error),
    Io(std::io::Error),
    Tensor(nml_tensor::Error),
    InvalidSharding(&'static str),
    ShardingRank {
        expected: usize,
        actual: usize,
    },
    UnevenSharding {
        axis: usize,
        dimension: i64,
        partitions: usize,
    },
    DeviceCount {
        required: usize,
        available: usize,
    },
    UnsupportedMemory {
        requested: Memory,
        platform: String,
    },
    NoDevices,
    ByteCountOverflow,
    InvalidStaging {
        buffers: usize,
        chunk_bytes: usize,
    },
    UnknownArgument(String),
    MissingArgument(String),
    BakedArgument(String),
    ArgumentShape {
        name: String,
        expected: Shape,
        actual: Shape,
    },
    ArgumentPlatform(String),
    ArgumentSharding(String),
    ArgumentShardCount {
        name: String,
        expected: usize,
        actual: usize,
    },
    DonationRequiresUniqueOwnership(String),
    ParameterDonation(String),
    DeletionRequiresUniqueOwnership,
    ResultCount {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCpuDeviceCount(count) => write!(f, "invalid CPU device count {count}"),
            Self::Platform(message) => f.write_str(message),
            Self::Pjrt(error) => error.fmt(f),
            Self::Xla(error) => error.fmt(f),
            Self::Compiler(error) => error.fmt(f),
            Self::Io(error) => error.fmt(f),
            Self::Tensor(error) => error.fmt(f),
            Self::InvalidSharding(message) => write!(f, "invalid sharding: {message}"),
            Self::ShardingRank { expected, actual } => write!(
                f,
                "sharding rank {actual} does not match tensor rank {expected}"
            ),
            Self::UnevenSharding {
                axis,
                dimension,
                partitions,
            } => write!(
                f,
                "dimension {axis} of size {dimension} is not divisible by {partitions} partitions"
            ),
            Self::DeviceCount {
                required,
                available,
            } => write!(
                f,
                "sharding requires {required} devices, platform exposes {available}"
            ),
            Self::UnsupportedMemory {
                requested,
                platform,
            } => write!(f, "memory {requested:?} is unsupported on {platform}"),
            Self::NoDevices => f.write_str("platform exposes no devices"),
            Self::ByteCountOverflow => f.write_str("physical buffer byte count overflows usize"),
            Self::InvalidStaging {
                buffers,
                chunk_bytes,
            } => write!(
                f,
                "invalid DMA staging pool: {buffers} buffers of {chunk_bytes} bytes"
            ),
            Self::UnknownArgument(name) => write!(f, "unknown executable argument {name:?}"),
            Self::MissingArgument(name) => write!(f, "missing executable argument {name:?}"),
            Self::BakedArgument(name) => write!(f, "baked parameter {name:?} cannot be replaced"),
            Self::ArgumentShape {
                name,
                expected,
                actual,
            } => write!(
                f,
                "argument {name:?} shape mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::ArgumentPlatform(name) => {
                write!(f, "argument {name:?} belongs to another platform")
            }
            Self::ArgumentSharding(name) => write!(
                f,
                "argument {name:?} is not replicated across the executable's device mesh"
            ),
            Self::ArgumentShardCount {
                name,
                expected,
                actual,
            } => write!(
                f,
                "argument {name:?} has {actual} device shards, executable requires {expected}"
            ),
            Self::DonationRequiresUniqueOwnership(name) => write!(
                f,
                "activation {name:?} must be uniquely owned before donation"
            ),
            Self::ParameterDonation(name) => {
                write!(f, "baked parameter {name:?} cannot be donated")
            }
            Self::DeletionRequiresUniqueOwnership => {
                f.write_str("shared buffer storage cannot be explicitly deleted")
            }
            Self::ResultCount { expected, actual } => write!(
                f,
                "executable returned {actual} results, expected {expected}"
            ),
        }
    }
}

impl StdError for Error {}

impl From<nml_tensor::Error> for Error {
    fn from(error: nml_tensor::Error) -> Self {
        Self::Tensor(error)
    }
}
