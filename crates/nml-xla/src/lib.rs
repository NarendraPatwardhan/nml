//! Validated XLA compile options with deterministic generated-upb encoding.

use nml_xla_sys as sys;
use std::error::Error as StdError;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Partitioner {
    Shardy,
    Gspmd,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileOptions {
    replicas: u32,
    partitions: u32,
    device_ids: Vec<i64>,
    partitioner: Partitioner,
    backend: Backend,
}

impl CompileOptions {
    pub fn single_device(device_id: i64, backend: Backend) -> Result<Self, Error> {
        Self::new(1, 1, vec![device_id], Partitioner::Shardy, backend)
    }

    pub fn new(
        replicas: u32,
        partitions: u32,
        device_ids: Vec<i64>,
        partitioner: Partitioner,
        backend: Backend,
    ) -> Result<Self, Error> {
        if replicas == 0 || partitions == 0 {
            return Err(Error::ZeroTopology);
        }
        if replicas > i32::MAX as u32 || partitions > i32::MAX as u32 {
            return Err(Error::TopologyOverflow);
        }
        let expected = (replicas as usize)
            .checked_mul(partitions as usize)
            .ok_or(Error::TopologyOverflow)?;
        if device_ids.len() != expected {
            return Err(Error::DeviceCount {
                expected,
                actual: device_ids.len(),
            });
        }
        if let Some((index, &id)) = device_ids.iter().enumerate().find(|(_, id)| **id < 0) {
            return Err(Error::InvalidDeviceId { index, id });
        }
        Ok(Self {
            replicas,
            partitions,
            device_ids,
            partitioner,
            backend,
        })
    }

    pub fn serialize(&self) -> Result<Vec<u8>, Error> {
        let raw = sys::NmlXlaCompileOptions {
            num_replicas: i32::try_from(self.replicas).map_err(|_| Error::TopologyOverflow)?,
            num_partitions: i32::try_from(self.partitions).map_err(|_| Error::TopologyOverflow)?,
            use_shardy_partitioner: self.partitioner == Partitioner::Shardy,
            enable_cuda_latency_hiding_scheduler: self.backend == Backend::Cuda,
            device_ids: self.device_ids.as_ptr(),
            num_device_ids: self.device_ids.len(),
        };
        let mut bytes = sys::NmlXlaBytes {
            data: std::ptr::null_mut(),
            size: 0,
        };
        let success = unsafe { sys::nml_xla_compile_options_serialize(&raw, &mut bytes) };
        if !success || bytes.data.is_null() {
            if !bytes.data.is_null() {
                unsafe { sys::nml_xla_bytes_destroy(bytes) };
            }
            return Err(Error::Serialization);
        }
        let owned = unsafe { std::slice::from_raw_parts(bytes.data, bytes.size) }.to_vec();
        unsafe { sys::nml_xla_bytes_destroy(bytes) };
        Ok(owned)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    ZeroTopology,
    TopologyOverflow,
    DeviceCount { expected: usize, actual: usize },
    InvalidDeviceId { index: usize, id: i64 },
    Serialization,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTopology => formatter.write_str("replicas and partitions must be non-zero"),
            Self::TopologyOverflow => formatter.write_str("replica/partition topology overflows"),
            Self::DeviceCount { expected, actual } => write!(
                formatter,
                "device assignment needs {expected} ids, received {actual}"
            ),
            Self::InvalidDeviceId { index, id } => {
                write!(
                    formatter,
                    "device assignment entry {index} is negative: {id}"
                )
            }
            Self::Serialization => formatter.write_str("XLA upb option serialization failed"),
        }
    }
}

impl StdError for Error {}
