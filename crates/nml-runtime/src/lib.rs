//! ZML-shaped persistent buffer and executable lifecycle.
//!
//! A public `Buffer` is one logical tensor. PJRT shards, transfer guards,
//! device identities, binding slots, and ownership counters remain private.

#![forbid(unsafe_op_in_unsafe_fn)]

use nml_ir::{InputBinding, Program};
use nml_parameter::{Parameter, StorageSpec};
use nml_pjrt::{Client, Device, LoadedExecutable};
pub use nml_sharding::Sharding;
use nml_tensor::Slice;
use nml_types::{DType, DTypeClass, Shape};
use std::collections::{BTreeMap, HashMap};
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
    compiler_target: nml_compiler::Target,
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
        let platform_name = client.platform_name().map_err(Error::Pjrt)?;
        let ffi = plugin.ffi().map_err(Error::Pjrt)?.ok_or_else(|| {
            Error::Platform("CPU PJRT plugin does not expose typed FFI".to_owned())
        })?;
        nml_kernel_nvfp4::register_cpu(&ffi, &platform_name).map_err(Error::Pjrt)?;
        Ok(Self {
            backend: Backend::Cpu,
            client,
            compiler_target: nml_compiler::Target::Cpu,
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
        #[cfg(nml_cuda)]
        {
            let custom_calls = runtime
                .custom_calls()
                .map_err(|error| Error::Platform(error.to_string()))?;
            nml_kernel_flash_attention::register(&custom_calls)
                .map_err(|error| Error::Platform(error.to_string()))?;
            nml_kernel_nvfp4::register_cuda(&custom_calls)
                .map_err(|error| Error::Platform(error.to_string()))?;
        }
        let client = runtime
            .create_client(nml_pjrt_cuda::ClientOptions::default())
            .map_err(|error| Error::Platform(error.to_string()))?;
        let capabilities = nml_pjrt_cuda::compute_capabilities(&client)
            .map_err(|error| Error::Platform(error.to_string()))?;
        let capability = *capabilities.first().ok_or(Error::NoDevices)?;
        if let Some((index, actual)) = capabilities
            .iter()
            .enumerate()
            .find(|(_, actual)| **actual != capability)
        {
            // A single StableHLO module is compiled for every addressable
            // device. Selecting FA2 from the minimum capability would still
            // route that custom call onto an SM90 device, where its adapter
            // correctly rejects the launch. Until compilation is partitioned
            // per architecture, reject a heterogeneous client up front rather
            // than producing an executable that fails according to placement.
            return Err(Error::Platform(format!(
                "heterogeneous CUDA compute capabilities are unsupported: device 0 is SM{}{} but device {index} is SM{}{}",
                capability.major, capability.minor, actual.major, actual.minor
            )));
        }
        let mut core_count = usize::MAX;
        for (index, device) in client.devices().map_err(Error::Pjrt)?.iter().enumerate() {
            let value = device
                .int64_attribute("core_count")
                .map_err(Error::Pjrt)?
                .ok_or_else(|| {
                    Error::Platform(format!("CUDA device {index} has no core_count attribute"))
                })?;
            let value = usize::try_from(value)
                .ok()
                .filter(|value| *value != 0)
                .ok_or_else(|| {
                    Error::Platform(format!(
                        "CUDA device {index} has invalid core_count {value}"
                    ))
                })?;
            core_count = core_count.min(value);
        }
        Ok(Self {
            backend: Backend::Cuda,
            client,
            compiler_target: nml_compiler::Target::Cuda {
                core_count,
                capability_major: capability.major,
                capability_minor: capability.minor,
            },
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
        sharding
            .validate_shape(slice.shape())
            .map_err(Error::Sharding)?;
        let devices = self.client.devices().map_err(Error::Pjrt)?;
        let shard_count = sharding
            .execution_count(devices.len())
            .map_err(Error::Sharding)?;
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

    /// Streams one physical parameter component through CUDA's reusable
    /// mapped DMA lanes. The component storage contract, rather than a logical
    /// parameter shape, determines the PJRT buffer that is allocated.
    ///
    /// CPU deliberately reads one aligned component and uses the ordinary
    /// immutable-host transfer path.
    #[doc(hidden)]
    pub fn upload_component_from(
        &self,
        storage: StorageSpec,
        sharding: Sharding,
        memory: Memory,
        staging_buffers: usize,
        chunk_bytes: usize,
        mut read: impl FnMut(usize, &mut [u8]) -> std::io::Result<()>,
    ) -> Result<Buffer, Error> {
        let shape = storage.shape();
        if staging_buffers == 0 || chunk_bytes == 0 {
            return Err(Error::InvalidStaging {
                buffers: staging_buffers,
                chunk_bytes,
            });
        }
        sharding.validate_shape(shape).map_err(Error::Sharding)?;
        let logical_bytes = shape.byte_count().map_err(nml_tensor::Error::from)?;
        if self.backend == Backend::Cpu || logical_bytes == 0 {
            let mut slice = Slice::alloc(shape)?;
            if logical_bytes != 0 {
                read(0, slice.contiguous_bytes_mut()?).map_err(Error::Io)?;
            }
            return self.upload(&slice, sharding, memory);
        }
        let devices = self.client.devices().map_err(Error::Pjrt)?;
        let shard_count = sharding
            .execution_count(devices.len())
            .map_err(Error::Sharding)?;
        if sharding.is_replicated() || sharding.is_single() {
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

        let shard_shape = sharding.shard_shape(shape).map_err(Error::Sharding)?;
        let shard_bytes = shard_shape.byte_count().map_err(nml_tensor::Error::from)?;
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
                    .create_async_transfer_manager(&[shard_shape], memory)
                    .map_err(Error::Pjrt)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Mesh coordinates on axes unused by this tensor have identical file
        // spans. Group them so each checkpoint byte is read once, then fan the
        // same owned DMA lane out to every device that holds that replica.
        let mut groups = BTreeMap::<Vec<(usize, i64, i64)>, Vec<usize>>::new();
        for index in 0..shard_count {
            groups
                .entry(sharding.ranges(shape, index)?)
                .or_default()
                .push(index);
        }
        for (ranges, destinations) in groups {
            let lane_bytes = chunk_bytes.min(shard_bytes).max(1);
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
            let mut destination_offset = 0usize;
            let mut transfer_index = 0usize;
            for_each_shard_span(shape, &ranges, |source_offset, span_length| {
                let mut consumed = 0usize;
                while consumed < span_length {
                    let length = chunk_bytes.min(span_length - consumed);
                    let lane = transfer_index % staging_buffers;
                    transfer_index += 1;
                    for transfer in pending[lane].drain(..) {
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
                    for &destination in &destinations {
                        pending[lane].push(
                            managers[destination]
                                .transfer_owned(
                                    0,
                                    Arc::clone(&lanes[lane]),
                                    length,
                                    offset,
                                    destination_offset == shard_bytes,
                                )
                                .map_err(Error::Pjrt)?,
                        );
                    }
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
        }
        let shards = managers
            .iter()
            .map(|manager| manager.retrieve_buffer(0).map_err(Error::Pjrt))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Buffer {
            storage: Arc::new(BufferStorage { shards }),
            shape,
            sharding,
            backend: self.backend,
            platform_id: self.client.as_raw_identity(),
            memory,
        })
    }

    pub fn compile(&self, program: &Program, sharding: Sharding) -> Result<Exe, Error> {
        let devices = self.client.devices().map_err(Error::Pjrt)?;
        if devices.is_empty() {
            return Err(Error::NoDevices);
        }
        let backend = match self.backend {
            Backend::Cpu => nml_xla::Backend::Cpu,
            Backend::Cuda => nml_xla::Backend::Cuda,
        };
        let all_device_ids = devices
            .iter()
            .map(|device| device.id().map_err(Error::Pjrt))
            .collect::<Result<Vec<_>, _>>()?;
        let (replicas, partitions, execution_count) = sharding
            .compile_topology(all_device_ids.len())
            .map_err(Error::Sharding)?;
        let device_ids = all_device_ids[..execution_count].to_vec();
        let options = nml_xla::CompileOptions::new(replicas, partitions, device_ids, backend)
            .map_err(Error::Xla)?;
        let loaded = nml_compiler::compile(
            &self.client,
            program,
            &sharding,
            &options,
            self.compiler_target,
        )
        .map_err(Error::Compiler)?;
        Exe::new(
            self.backend,
            self.client.as_raw_identity(),
            loaded,
            program,
            execution_count,
            sharding,
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

/// Capacity and tensor geometry for one dense or paged K/V cache.
///
/// Constructors keep the layout choice explicit without exporting separate
/// backend-specific cache types. CUDA and CPU own the same logical state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheSpec {
    dtype: DType,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    layout: CacheLayout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CacheLayout {
    Dense {
        capacity: usize,
    },
    Paged {
        physical_pages: usize,
        logical_pages: usize,
        page_size: usize,
    },
}

impl CacheSpec {
    pub fn dense(
        dtype: DType,
        batch: usize,
        capacity: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, Error> {
        let spec = Self {
            dtype,
            batch,
            kv_heads,
            head_dim,
            layout: CacheLayout::Dense { capacity },
        };
        spec.validate()?;
        Ok(spec)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged(
        dtype: DType,
        batch: usize,
        physical_pages: usize,
        logical_pages: usize,
        page_size: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, Error> {
        let spec = Self {
            dtype,
            batch,
            kv_heads,
            head_dim,
            layout: CacheLayout::Paged {
                physical_pages,
                logical_pages,
                page_size,
            },
        };
        spec.validate()?;
        Ok(spec)
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn batch(self) -> usize {
        self.batch
    }

    pub const fn capacity(self) -> usize {
        match self.layout {
            CacheLayout::Dense { capacity } => capacity,
            CacheLayout::Paged {
                logical_pages,
                page_size,
                ..
            } => logical_pages * page_size,
        }
    }

    pub fn key_value_shape(self) -> Result<Shape, Error> {
        let dimensions = match self.layout {
            CacheLayout::Dense { capacity } => [
                to_i64(self.batch)?,
                to_i64(capacity)?,
                to_i64(self.kv_heads)?,
                to_i64(self.head_dim)?,
            ],
            CacheLayout::Paged {
                physical_pages,
                page_size,
                ..
            } => [
                to_i64(physical_pages)?,
                to_i64(page_size)?,
                to_i64(self.kv_heads)?,
                to_i64(self.head_dim)?,
            ],
        };
        Shape::new(self.dtype, &dimensions)
            .map_err(nml_tensor::Error::from)
            .map_err(Into::into)
    }

    pub fn page_table_shape(self) -> Result<Option<Shape>, Error> {
        match self.layout {
            CacheLayout::Dense { .. } => Ok(None),
            CacheLayout::Paged { logical_pages, .. } => Ok(Some(
                Shape::new(DType::I64, &[to_i64(self.batch)?, to_i64(logical_pages)?])
                    .map_err(nml_tensor::Error::from)?,
            )),
        }
    }

    pub fn lengths_shape(self) -> Result<Shape, Error> {
        Shape::new(DType::I64, &[to_i64(self.batch)?])
            .map_err(nml_tensor::Error::from)
            .map_err(Error::Tensor)
    }

    fn validate(self) -> Result<(), Error> {
        if self.dtype.class() != DTypeClass::Float {
            return Err(Error::InvalidCache("cache dtype must be floating point"));
        }
        if self.batch == 0 || self.kv_heads == 0 || self.head_dim == 0 {
            return Err(Error::InvalidCache(
                "batch, KV heads, and head dimension must be nonzero",
            ));
        }
        match self.layout {
            CacheLayout::Dense { capacity } if capacity == 0 => {
                Err(Error::InvalidCache("dense capacity must be nonzero"))
            }
            CacheLayout::Paged {
                physical_pages,
                logical_pages,
                page_size,
            } if physical_pages == 0 || logical_pages == 0 || page_size == 0 => Err(
                Error::InvalidCache("paged capacities and page size must be nonzero"),
            ),
            CacheLayout::Paged {
                logical_pages,
                page_size,
                ..
            } => {
                let capacity = logical_pages
                    .checked_mul(page_size)
                    .ok_or(Error::InvalidCache("logical capacity overflows usize"))?;
                to_i64(capacity).map_err(|_| {
                    Error::InvalidCache("logical capacity exceeds the I64 cache-index domain")
                })?;
                Ok(())
            }
            CacheLayout::Dense { .. } => Ok(()),
        }
    }
}

/// Persistent K/V device storage plus its small host-owned logical metadata.
///
/// `take_storage` and `replace_storage` make XLA donation explicit: a decode
/// step temporarily removes uniquely-owned K/V buffers, then installs the
/// aliased outputs without copying unaffected cache contents.
pub struct Cache {
    spec: CacheSpec,
    key: Option<Buffer>,
    value: Option<Buffer>,
    page_table: Option<Buffer>,
    lengths: Buffer,
    host_page_table: Vec<i64>,
    host_lengths: Vec<i64>,
}

impl Cache {
    pub fn allocate(
        platform: &Platform,
        spec: CacheSpec,
        sharding: Sharding,
        memory: Memory,
    ) -> Result<Self, Error> {
        spec.validate()?;
        let cache_shape = spec.key_value_shape()?;
        let key = platform.upload(&Slice::alloc(cache_shape)?, sharding.clone(), memory)?;
        let value = platform.upload(&Slice::alloc(cache_shape)?, sharding.clone(), memory)?;
        let host_lengths = vec![0i64; spec.batch];
        let lengths_shape = spec.lengths_shape()?;
        let lengths = platform.upload(
            &Slice::from_typed(lengths_shape, &host_lengths)?,
            sharding.clone(),
            memory,
        )?;
        let (host_page_table, page_table) = match spec.page_table_shape()? {
            None => (Vec::new(), None),
            Some(shape) => {
                let entries = shape.element_count().map_err(nml_tensor::Error::from)?;
                let table = vec![-1i64; entries];
                let buffer =
                    platform.upload(&Slice::from_typed(shape, &table)?, sharding, memory)?;
                (table, Some(buffer))
            }
        };
        Ok(Self {
            spec,
            key: Some(key),
            value: Some(value),
            page_table,
            lengths,
            host_page_table,
            host_lengths,
        })
    }

    pub const fn spec(&self) -> CacheSpec {
        self.spec
    }

    pub fn key(&self) -> Result<&Buffer, Error> {
        self.key.as_ref().ok_or(Error::CacheStorageUnavailable)
    }

    pub fn value(&self) -> Result<&Buffer, Error> {
        self.value.as_ref().ok_or(Error::CacheStorageUnavailable)
    }

    pub fn page_table(&self) -> Option<&Buffer> {
        self.page_table.as_ref()
    }

    pub const fn lengths(&self) -> &Buffer {
        &self.lengths
    }

    pub fn take_storage(&mut self) -> Result<(Buffer, Buffer), Error> {
        match (self.key.take(), self.value.take()) {
            (Some(key), Some(value)) => Ok((key, value)),
            (key, value) => {
                self.key = key;
                self.value = value;
                Err(Error::CacheStorageUnavailable)
            }
        }
    }

    pub fn replace_storage(&mut self, key: Buffer, value: Buffer) -> Result<(), Error> {
        if self.key.is_some() || self.value.is_some() {
            return Err(Error::CacheStorageAlreadyInstalled);
        }
        let expected = self.spec.key_value_shape()?;
        if key.shape != expected || value.shape != expected {
            return Err(Error::InvalidCache(
                "replacement K/V shape differs from cache spec",
            ));
        }
        if key.backend != self.lengths.backend
            || value.backend != self.lengths.backend
            || key.platform_id != self.lengths.platform_id
            || value.platform_id != self.lengths.platform_id
        {
            return Err(Error::InvalidCache(
                "replacement K/V storage belongs to another platform",
            ));
        }
        if key.sharding != self.lengths.sharding || value.sharding != self.lengths.sharding {
            return Err(Error::InvalidCache(
                "replacement K/V storage uses another sharding",
            ));
        }
        self.key = Some(key);
        self.value = Some(value);
        Ok(())
    }

    pub fn assign_page(
        &mut self,
        platform: &Platform,
        batch: usize,
        logical_page: usize,
        physical_page: usize,
    ) -> Result<(), Error> {
        let CacheLayout::Paged {
            physical_pages,
            logical_pages,
            ..
        } = self.spec.layout
        else {
            return Err(Error::InvalidCache("dense caches have no page table"));
        };
        if batch >= self.spec.batch
            || logical_page >= logical_pages
            || physical_page >= physical_pages
        {
            return Err(Error::InvalidCache(
                "page assignment is outside cache capacity",
            ));
        }
        self.require_platform(platform)?;
        let mut host_page_table = self.host_page_table.clone();
        host_page_table[batch * logical_pages + logical_page] = to_i64(physical_page)?;
        let shape = self
            .spec
            .page_table_shape()?
            .expect("paged cache has a page table shape");
        let page_table = platform.upload(
            &Slice::from_typed(shape, &host_page_table)?,
            self.lengths.sharding.clone(),
            self.lengths.memory,
        )?;
        self.host_page_table = host_page_table;
        self.page_table = Some(page_table);
        Ok(())
    }

    /// Moves the logical boundary in either direction. Growing is replay and
    /// is allowed only across pages already assigned to the batch.
    pub fn truncate(
        &mut self,
        platform: &Platform,
        batch: usize,
        length: usize,
    ) -> Result<(), Error> {
        if batch >= self.spec.batch || length > self.spec.capacity() {
            return Err(Error::InvalidCache(
                "sequence length is outside cache capacity",
            ));
        }
        self.require_platform(platform)?;
        if let CacheLayout::Paged {
            logical_pages,
            page_size,
            ..
        } = self.spec.layout
        {
            let required = length.div_ceil(page_size);
            let row = &self.host_page_table[batch * logical_pages..(batch + 1) * logical_pages];
            if row[..required].contains(&-1) {
                return Err(Error::InvalidCache(
                    "sequence length reaches an unassigned logical page",
                ));
            }
        }
        let mut host_lengths = self.host_lengths.clone();
        host_lengths[batch] = to_i64(length)?;
        let lengths = platform.upload(
            &Slice::from_typed(self.spec.lengths_shape()?, &host_lengths)?,
            self.lengths.sharding.clone(),
            self.lengths.memory,
        )?;
        self.host_lengths = host_lengths;
        self.lengths = lengths;
        Ok(())
    }

    fn require_platform(&self, platform: &Platform) -> Result<(), Error> {
        if self.lengths.backend != platform.backend
            || self.lengths.platform_id != platform.client.as_raw_identity()
        {
            Err(Error::InvalidCache("cache belongs to another platform"))
        } else {
            Ok(())
        }
    }
}

fn to_i64(value: usize) -> Result<i64, Error> {
    i64::try_from(value).map_err(|_| Error::InvalidCache("cache dimension exceeds i64"))
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

/// In-flight download of one unsharded logical buffer.
pub struct BufferDownload<'transfer> {
    transfer: nml_pjrt::HostDownload<'transfer>,
}

impl BufferDownload<'_> {
    pub fn wait(self) -> Result<(), Error> {
        self.transfer.wait().map_err(Error::Pjrt)
    }
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
        if self.sharding.is_replicated() {
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

    /// Starts a nonblocking download of an unsharded logical buffer.
    /// Sharded downloads require host-side placement assembly and therefore
    /// retain the synchronous `to_slice` path.
    pub fn download_to<'transfer>(
        &'transfer self,
        destination: &'transfer mut Slice<'_>,
    ) -> Result<BufferDownload<'transfer>, Error> {
        if !self.sharding.is_replicated() && !self.sharding.is_single() {
            return Err(Error::AsyncDownloadRequiresUnshardedPlacement);
        }
        let transfer = self.storage.shards[0]
            .to_slice_async(destination)
            .map_err(Error::Pjrt)?;
        Ok(BufferDownload { transfer })
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

/// Runtime ownership of every physical component for one logical parameter.
#[derive(Clone)]
pub struct LoadedParameter {
    parameter: Parameter,
    components: Vec<Buffer>,
}

impl LoadedParameter {
    pub fn new(parameter: Parameter, components: Vec<Buffer>) -> Result<Self, Error> {
        if parameter.components().len() != components.len() {
            return Err(Error::ParameterComponentCount {
                parameter: parameter.name().to_owned(),
                expected: parameter.components().len(),
                actual: components.len(),
            });
        }
        for (spec, buffer) in parameter.components().iter().zip(&components) {
            if spec.storage().shape() != buffer.shape() {
                return Err(Error::ArgumentShape {
                    name: spec.binding_name().to_owned(),
                    expected: spec.storage().shape(),
                    actual: buffer.shape(),
                });
            }
        }
        Ok(Self {
            parameter,
            components,
        })
    }

    pub fn parameter(&self) -> &Parameter {
        &self.parameter
    }

    pub fn components(&self) -> impl Iterator<Item = (&nml_parameter::ComponentSpec, &Buffer)> {
        self.parameter.components().iter().zip(&self.components)
    }

    pub fn component(&self, role: nml_parameter::ComponentRole) -> Option<&Buffer> {
        self.components()
            .find_map(|(spec, buffer)| (spec.role() == role).then_some(buffer))
    }
}

/// Structural mapping from logical parameters to loaded parameter owners.
pub trait ParameterTree {
    type Loaded;

    fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &Parameter));

    fn visit_loaded(
        loaded: &Self::Loaded,
        prefix: &str,
        visitor: &mut dyn FnMut(&str, &LoadedParameter),
    );

    fn load_parameters<E>(
        &self,
        prefix: &str,
        resolve: &mut impl FnMut(&str, &Parameter) -> Result<LoadedParameter, E>,
    ) -> Result<Self::Loaded, E>;
}

pub type Loaded<T> = <T as ParameterTree>::Loaded;

/// A compiled executable with named, reusable argument bindings.
#[derive(Clone)]
pub struct Exe {
    inner: Arc<ExeInner>,
}

struct ExeInner {
    backend: Backend,
    platform_id: usize,
    loaded: LoadedExecutable,
    device_count: usize,
    sharding: Sharding,
    inputs: Vec<Binding>,
    input_indices: HashMap<String, usize>,
    outputs: Vec<OutputBinding>,
    output_names: Arc<[String]>,
}

#[derive(Clone)]
struct Binding {
    name: String,
    shape: Shape,
    contract: InputBinding,
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
        sharding: Sharding,
    ) -> Result<Self, Error> {
        let inputs = program
            .inputs()
            .map(|(name, shape, contract)| Binding {
                name: name.to_owned(),
                shape,
                contract: contract.clone(),
            })
            .collect::<Vec<_>>();
        let input_indices = inputs
            .iter()
            .enumerate()
            .map(|(index, binding)| (binding.name.clone(), index))
            .collect();
        let outputs = program
            .outputs()
            .zip(program.output_aliases())
            .map(|((name, shape), alias_input)| OutputBinding {
                name: name.to_owned(),
                shape,
                alias_input,
            })
            .collect::<Vec<_>>();
        let output_names = outputs
            .iter()
            .map(|output| output.name.clone())
            .collect::<Vec<_>>()
            .into();
        Ok(Self {
            inner: Arc::new(ExeInner {
                backend,
                platform_id,
                loaded,
                device_count,
                sharding,
                inputs,
                input_indices,
                outputs,
                output_names,
            }),
        })
    }

    /// Creates an independently owned argument set.
    ///
    /// The returned bindings keep the loaded executable alive. This is
    /// intentional: long-lived inference sessions can retain baked arguments
    /// without borrowing the model object that originally compiled them.
    pub fn args(&self) -> exe::Arguments {
        exe::Arguments {
            executable: Arc::clone(&self.inner),
            slots: vec![None; self.inner.inputs.len()],
            baked: vec![false; self.inner.inputs.len()],
        }
    }
}

impl ExeInner {
    fn results(
        &self,
        buffers: Vec<Buffer>,
        completion: Vec<nml_pjrt::Event>,
    ) -> Result<exe::Results, Error> {
        if buffers.len() != self.outputs.len() {
            return Err(Error::ResultCount {
                expected: self.outputs.len(),
                actual: buffers.len(),
            });
        }
        Ok(exe::Results {
            names: Arc::clone(&self.output_names),
            buffers,
            completion,
        })
    }
}

pub mod exe {
    use super::{Arc, Buffer, Error, ExeInner, LoadedParameter, Parameter};

    /// Reusable executable bindings that own the executable they target.
    ///
    /// Arguments are deliberately not tied to an `Exe` borrow. The loaded
    /// executable is reference-counted together with its immutable binding
    /// manifest, while every argument set owns its mutable buffer slots.
    pub struct Arguments {
        pub(super) executable: Arc<ExeInner>,
        pub(super) slots: Vec<Option<Buffer>>,
        pub(super) baked: Vec<bool>,
    }

    impl Arguments {
        pub fn set(&mut self, name: &str, buffer: Buffer) -> Result<&mut Self, Error> {
            let index = self
                .executable
                .input_indices
                .get(name)
                .copied()
                .ok_or_else(|| Error::UnknownArgument(name.to_owned()))?;
            self.validate(index, &buffer)?;
            self.slots[index] = Some(buffer);
            Ok(self)
        }

        /// Releases one dynamic binding without changing the executable.
        ///
        /// Reusable inference pipelines must drop non-donated aliases before
        /// handing the same physical buffer to a later executable that donates
        /// it. Baked parameter bindings are immutable and cannot be released.
        pub fn unset(&mut self, name: &str) -> Result<Option<Buffer>, Error> {
            let index = self
                .executable
                .input_indices
                .get(name)
                .copied()
                .ok_or_else(|| Error::UnknownArgument(name.to_owned()))?;
            if self.baked[index] {
                return Err(Error::BakedArgument(name.to_owned()));
            }
            Ok(self.slots[index].take())
        }

        fn validate(&self, index: usize, buffer: &Buffer) -> Result<(), Error> {
            let binding = &self.executable.inputs[index];
            if binding.shape != buffer.shape {
                return Err(Error::ArgumentShape {
                    name: binding.name.clone(),
                    expected: binding.shape,
                    actual: buffer.shape,
                });
            }
            if buffer.backend != self.executable.backend {
                return Err(Error::ArgumentPlatform(binding.name.clone()));
            }
            if buffer.platform_id != self.executable.platform_id {
                return Err(Error::ArgumentPlatform(binding.name.clone()));
            }
            if buffer.sharding != self.executable.sharding {
                return Err(Error::ArgumentSharding(binding.name.clone()));
            }
            if self.baked[index] {
                return Err(Error::BakedArgument(binding.name.clone()));
            }
            Ok(())
        }

        /// Binds every physical component of one logical parameter and checks
        /// the executable's representation-aware component manifest.
        pub fn set_parameter(&mut self, parameter: &LoadedParameter) -> Result<&mut Self, Error> {
            self.set_parameter_slot(parameter.parameter(), parameter)
        }

        /// Binds loaded storage to a semantically compatible compiled slot.
        ///
        /// Checkpoint paths identify persistent storage, while slots identify
        /// roles in a reusable executable. Repeated blocks may therefore have
        /// distinct parameter names while sharing a slot contract. Rebinding is
        /// accepted only when logical shape, representation, component roles,
        /// physical storage, placement, and platform all agree.
        pub fn set_parameter_slot(
            &mut self,
            slot: &Parameter,
            parameter: &LoadedParameter,
        ) -> Result<&mut Self, Error> {
            if slot.shape() != parameter.parameter().shape()
                || slot.representation_id() != parameter.parameter().representation_id()
                || slot.components().len() != parameter.parameter().components().len()
            {
                return Err(Error::ParameterSlotContract {
                    slot: slot.name().to_owned(),
                    parameter: parameter.parameter().name().to_owned(),
                });
            }

            let mut assignments = Vec::with_capacity(slot.components().len());
            for component in slot.components() {
                let Some((actual, buffer)) = parameter
                    .components()
                    .find(|(actual, _)| actual.role() == component.role())
                else {
                    return Err(Error::ParameterSlotContract {
                        slot: slot.name().to_owned(),
                        parameter: parameter.parameter().name().to_owned(),
                    });
                };
                if actual.storage() != component.storage() {
                    return Err(Error::ParameterSlotContract {
                        slot: slot.name().to_owned(),
                        parameter: parameter.parameter().name().to_owned(),
                    });
                }
                let index = self
                    .executable
                    .input_indices
                    .get(component.binding_name())
                    .copied()
                    .ok_or_else(|| Error::UnknownArgument(component.binding_name().to_owned()))?;
                let binding = &self.executable.inputs[index];
                let nml_ir::InputBinding::ParameterComponent(contract) = &binding.contract else {
                    return Err(Error::ArgumentIsNotParameterComponent(
                        component.binding_name().to_owned(),
                    ));
                };
                if contract.parameter() != slot.name()
                    || contract.representation() != slot.representation_id()
                    || contract.role() != component.role()
                    || contract.storage() != component.storage()
                {
                    return Err(Error::ParameterContract {
                        parameter: slot.name().to_owned(),
                        component: component.binding_name().to_owned(),
                    });
                }
                if assignments
                    .iter()
                    .any(|(assigned, _): &(usize, Buffer)| *assigned == index)
                {
                    return Err(Error::ParameterContract {
                        parameter: slot.name().to_owned(),
                        component: component.binding_name().to_owned(),
                    });
                }
                self.validate(index, buffer)?;
                assignments.push((index, buffer.clone()));
            }
            // Validate every component before changing any slot. A malformed
            // multi-component parameter must never leave a partially rebound
            // executable behind.
            for (index, buffer) in assignments {
                self.slots[index] = Some(buffer);
            }
            Ok(self)
        }

        pub fn bake(&mut self) -> Result<&mut Self, Error> {
            for (index, binding) in self.executable.inputs.iter().enumerate() {
                if binding.contract.is_parameter_component() {
                    if self.slots[index].is_none() {
                        return Err(Error::MissingArgument(binding.name.clone()));
                    }
                    self.baked[index] = true;
                }
            }
            Ok(self)
        }

        /// Enqueues execution and returns dependency-carrying device buffers
        /// without synchronizing the host. Returned buffers can be supplied to
        /// another enqueue immediately; PJRT preserves readiness dependencies.
        pub fn enqueue(&mut self) -> Result<Results, Error> {
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
                        || self.executable.sharding.clone(),
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
            let completion = execution.complete;
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
            self.executable.results(buffers, completion)
        }

        /// Executes and waits for all addressable devices to complete.
        pub fn call(&mut self) -> Result<Results, Error> {
            self.enqueue()?.wait()
        }
    }

    pub struct Results {
        pub(super) names: Arc<[String]>,
        pub(super) buffers: Vec<Buffer>,
        pub(super) completion: Vec<nml_pjrt::Event>,
    }

    impl Results {
        pub fn wait(mut self) -> Result<Self, Error> {
            for event in &self.completion {
                event.wait().map_err(Error::Pjrt)?;
            }
            self.completion.clear();
            Ok(self)
        }

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
    Sharding(nml_sharding::Error),
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
    InvalidCache(&'static str),
    CacheStorageUnavailable,
    CacheStorageAlreadyInstalled,
    UnknownArgument(String),
    MissingArgument(String),
    BakedArgument(String),
    ArgumentIsNotParameterComponent(String),
    ParameterContract {
        parameter: String,
        component: String,
    },
    ParameterSlotContract {
        slot: String,
        parameter: String,
    },
    ParameterComponentCount {
        parameter: String,
        expected: usize,
        actual: usize,
    },
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
    AsyncDownloadRequiresUnshardedPlacement,
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
            Self::Sharding(error) => error.fmt(f),
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
            Self::InvalidCache(message) => write!(f, "invalid cache: {message}"),
            Self::CacheStorageUnavailable => {
                f.write_str("cache K/V storage is temporarily owned by an execution")
            }
            Self::CacheStorageAlreadyInstalled => f.write_str("cache already contains K/V storage"),
            Self::UnknownArgument(name) => write!(f, "unknown executable argument {name:?}"),
            Self::MissingArgument(name) => write!(f, "missing executable argument {name:?}"),
            Self::BakedArgument(name) => write!(f, "baked parameter {name:?} cannot be replaced"),
            Self::ArgumentIsNotParameterComponent(name) => {
                write!(
                    f,
                    "executable argument {name:?} is not a parameter component"
                )
            }
            Self::ParameterContract {
                parameter,
                component,
            } => write!(
                f,
                "loaded parameter {parameter:?} does not satisfy component binding {component:?}"
            ),
            Self::ParameterSlotContract { slot, parameter } => write!(
                f,
                "loaded parameter {parameter:?} is incompatible with executable slot {slot:?}"
            ),
            Self::ParameterComponentCount {
                parameter,
                expected,
                actual,
            } => write!(
                f,
                "loaded parameter {parameter:?} has {actual} components, expected {expected}"
            ),
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
                "argument {name:?} placement differs from the executable topology"
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
            Self::AsyncDownloadRequiresUnshardedPlacement => {
                f.write_str("asynchronous downloads require unsharded buffer placement")
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

impl From<nml_sharding::Error> for Error {
    fn from(error: nml_sharding::Error) -> Self {
        Self::Sharding(error)
    }
}
