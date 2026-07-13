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
            memory,
        })
    }

    pub fn compile(&self, program: &Program) -> Result<Exe, Error> {
        let device = self
            .client
            .devices()
            .map_err(Error::Pjrt)?
            .into_iter()
            .next()
            .ok_or(Error::NoDevices)?;
        let backend = match self.backend {
            Backend::Cpu => nml_xla::Backend::Cpu,
            Backend::Cuda => nml_xla::Backend::Cuda,
        };
        let options =
            nml_xla::CompileOptions::single_device(device.id().map_err(Error::Pjrt)?, backend)
                .map_err(Error::Xla)?;
        let loaded =
            nml_compiler::compile(&self.client, program, &options).map_err(Error::Compiler)?;
        Exe::new(self.backend, loaded, program)
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

/// One logical, possibly sharded, persistent device tensor.
#[derive(Clone)]
pub struct Buffer {
    storage: Arc<BufferStorage>,
    shape: Shape,
    sharding: Sharding,
    backend: Backend,
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

    pub fn delete(&self) -> Result<(), Error> {
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
    loaded: LoadedExecutable,
    inputs: Vec<Binding>,
    outputs: Vec<(String, Shape)>,
}

#[derive(Clone)]
struct Binding {
    name: String,
    shape: Shape,
    kind: InputKind,
}

impl Exe {
    fn new(backend: Backend, loaded: LoadedExecutable, program: &Program) -> Result<Self, Error> {
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
            .map(|(name, shape)| (name.to_owned(), shape))
            .collect();
        Ok(Self {
            backend,
            loaded,
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
            names: self.outputs.iter().map(|(name, _)| name.clone()).collect(),
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

        pub fn call(&self) -> Result<Results, Error> {
            for (index, binding) in self.executable.inputs.iter().enumerate() {
                if self.slots[index].is_none() {
                    return Err(Error::MissingArgument(binding.name.clone()));
                }
            }
            if self.slots.iter().any(|slot| {
                slot.as_ref()
                    .is_some_and(|buffer| buffer.raw_shards().len() != 1)
            }) {
                return Err(Error::MultiDeviceExecutionNotYetRepresentable);
            }
            let raw = self
                .slots
                .iter()
                .map(|slot| &slot.as_ref().expect("checked above").raw_shards()[0])
                .collect::<Vec<_>>();
            let execution = self
                .executable
                .loaded
                .execute_one(&raw, None)
                .map_err(Error::Pjrt)?;
            execution.complete.wait().map_err(Error::Pjrt)?;
            let buffers = execution
                .outputs
                .into_iter()
                .zip(&self.executable.outputs)
                .map(|(buffer, (_, shape))| Buffer {
                    storage: std::sync::Arc::new(super::BufferStorage {
                        shards: vec![buffer],
                    }),
                    shape: *shape,
                    sharding: super::Sharding::tiled(&vec![1; shape.rank()])
                        .expect("unit sharding is valid"),
                    backend: self.executable.backend,
                    memory: super::Memory::Default,
                })
                .collect();
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
    UnknownArgument(String),
    MissingArgument(String),
    BakedArgument(String),
    ArgumentShape {
        name: String,
        expected: Shape,
        actual: Shape,
    },
    ArgumentPlatform(String),
    ResultCount {
        expected: usize,
        actual: usize,
    },
    MultiDeviceExecutionNotYetRepresentable,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCpuDeviceCount(count) => write!(f, "invalid CPU device count {count}"),
            Self::Platform(message) => f.write_str(message),
            Self::Pjrt(error) => error.fmt(f),
            Self::Xla(error) => error.fmt(f),
            Self::Compiler(error) => error.fmt(f),
            Self::Tensor(error) => error.fmt(f),
            Self::InvalidSharding(message) => write!(f, "invalid sharding: {message}"),
            Self::ShardingRank { expected, actual } => write!(f, "sharding rank {actual} does not match tensor rank {expected}"),
            Self::UnevenSharding { axis, dimension, partitions } => write!(f, "dimension {axis} of size {dimension} is not divisible by {partitions} partitions"),
            Self::DeviceCount { required, available } => write!(f, "sharding requires {required} devices, platform exposes {available}"),
            Self::UnsupportedMemory { requested, platform } => write!(f, "memory {requested:?} is unsupported on {platform}"),
            Self::NoDevices => f.write_str("platform exposes no devices"),
            Self::ByteCountOverflow => f.write_str("physical buffer byte count overflows usize"),
            Self::UnknownArgument(name) => write!(f, "unknown executable argument {name:?}"),
            Self::MissingArgument(name) => write!(f, "missing executable argument {name:?}"),
            Self::BakedArgument(name) => write!(f, "baked parameter {name:?} cannot be replaced"),
            Self::ArgumentShape { name, expected, actual } => write!(f, "argument {name:?} shape mismatch: expected {expected:?}, received {actual:?}"),
            Self::ArgumentPlatform(name) => write!(f, "argument {name:?} belongs to another platform"),
            Self::ResultCount { expected, actual } => write!(f, "executable returned {actual} results, expected {expected}"),
            Self::MultiDeviceExecutionNotYetRepresentable => f.write_str("multi-device executable argument flattening requires a multi-device compiled executable"),
        }
    }
}

impl StdError for Error {}

impl From<nml_tensor::Error> for Error {
    fn from(error: nml_tensor::Error) -> Self {
        Self::Tensor(error)
    }
}
