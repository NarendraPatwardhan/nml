//! One logical mesh contract shared by graph lowering and physical buffers.

#![forbid(unsafe_code)]

use nml_types::{AxisTag, Partition, Shape};
use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Sharding {
    placement: Placement,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Placement {
    Single,
    Replicated,
    Mesh(Vec<MeshAxis>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MeshAxis {
    tag: AxisTag,
    size: usize,
}

impl Sharding {
    pub const fn single() -> Self {
        Self {
            placement: Placement::Single,
        }
    }

    pub const fn replicated() -> Self {
        Self {
            placement: Placement::Replicated,
        }
    }

    pub fn mesh(axes: &[(AxisTag, usize)]) -> Result<Self, Error> {
        if axes.is_empty() {
            return Err(Error::EmptyMesh);
        }
        let mut seen = HashSet::new();
        let mut product = 1usize;
        let mut result = Vec::with_capacity(axes.len());
        for &(tag, size) in axes {
            if tag == AxisTag::UNKNOWN {
                return Err(Error::UnknownAxis);
            }
            if size == 0 {
                return Err(Error::ZeroAxis(tag));
            }
            if i64::try_from(size).is_err() {
                return Err(Error::AxisSizeOverflow { tag, size });
            }
            if !seen.insert(tag) {
                return Err(Error::DuplicateAxis(tag));
            }
            product = product.checked_mul(size).ok_or(Error::MeshSizeOverflow)?;
            result.push(MeshAxis { tag, size });
        }
        let _ = product;
        Ok(Self {
            placement: Placement::Mesh(result),
        })
    }

    pub fn is_single(&self) -> bool {
        matches!(self.placement, Placement::Single)
    }

    pub fn is_replicated(&self) -> bool {
        matches!(self.placement, Placement::Replicated)
    }

    pub fn is_mesh(&self) -> bool {
        matches!(self.placement, Placement::Mesh(_))
    }

    pub fn mesh_axes(&self) -> impl Iterator<Item = (AxisTag, usize)> + '_ {
        match &self.placement {
            Placement::Mesh(axes) => Some(axes.as_slice()),
            _ => None,
        }
        .into_iter()
        .flatten()
        .map(|axis| (axis.tag, axis.size))
    }

    pub fn execution_count(&self, available: usize) -> Result<usize, Error> {
        let required = match &self.placement {
            Placement::Single => 1,
            Placement::Replicated => available,
            Placement::Mesh(axes) => mesh_product(axes)?,
        };
        if required == 0 || required > available {
            return Err(Error::DeviceCount {
                required,
                available,
            });
        }
        if self.is_mesh() && required != available {
            return Err(Error::DeviceCount {
                required,
                available,
            });
        }
        Ok(required)
    }

    pub fn compile_topology(&self, available: usize) -> Result<(u32, u32, usize), Error> {
        let count = self.execution_count(available)?;
        let (replicas, partitions) = match &self.placement {
            Placement::Single => (1, 1),
            Placement::Replicated => (count, 1),
            Placement::Mesh(_) => (1, count),
        };
        Ok((
            u32::try_from(replicas).map_err(|_| Error::MeshSizeOverflow)?,
            u32::try_from(partitions).map_err(|_| Error::MeshSizeOverflow)?,
            count,
        ))
    }

    pub fn validate_shape(&self, shape: Shape) -> Result<(), Error> {
        let Placement::Mesh(axes) = &self.placement else {
            if shape
                .partitions()
                .iter()
                .any(|partition| matches!(partition, Partition::Sharded(_)))
            {
                return Err(Error::PartitionWithoutMesh);
            }
            return Ok(());
        };
        let mut consumed = HashSet::new();
        for (dimension, partition) in shape.dimensions().iter().zip(shape.partitions()) {
            let Partition::Sharded(tag) = partition else {
                continue;
            };
            if *tag == AxisTag::UNKNOWN {
                return Err(Error::UnknownAxis);
            }
            let axis = axes
                .iter()
                .find(|axis| axis.tag == *tag)
                .ok_or(Error::MissingAxis(*tag))?;
            if !consumed.insert(*tag) {
                return Err(Error::AxisConsumedTwice(*tag));
            }
            if *dimension % axis.size as i64 != 0 {
                return Err(Error::UnevenDimension {
                    tag: *tag,
                    dimension: *dimension,
                    partitions: axis.size,
                });
            }
        }
        Ok(())
    }

    pub fn shard_shape(&self, shape: Shape) -> Result<Shape, Error> {
        self.validate_shape(shape)?;
        let dimensions = shape
            .dimensions()
            .iter()
            .zip(shape.partitions())
            .map(|(dimension, partition)| match partition {
                Partition::Sharded(tag) => Ok(*dimension / self.axis_size(*tag)? as i64),
                _ => Ok(*dimension),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Shape::new(shape.dtype(), &dimensions)?
            .with_axis_tags(shape.axis_tags())?
            .with_partitions(shape.partitions())?
            .with_layout(shape.layout())?)
    }

    pub fn ranges(&self, shape: Shape, shard: usize) -> Result<Vec<(usize, i64, i64)>, Error> {
        self.validate_shape(shape)?;
        let Placement::Mesh(axes) = &self.placement else {
            return Ok(Vec::new());
        };
        let count = mesh_product(axes)?;
        if shard >= count {
            return Err(Error::ShardOutOfBounds { shard, count });
        }
        let mut remainder = shard;
        let mut coordinates = vec![0usize; axes.len()];
        for axis in (0..axes.len()).rev() {
            coordinates[axis] = remainder % axes[axis].size;
            remainder /= axes[axis].size;
        }
        shape
            .dimensions()
            .iter()
            .zip(shape.partitions())
            .enumerate()
            .filter_map(|(tensor_axis, (dimension, partition))| {
                let Partition::Sharded(tag) = partition else {
                    return None;
                };
                let mesh_axis = axes.iter().position(|axis| axis.tag == *tag).unwrap();
                let length = *dimension / axes[mesh_axis].size as i64;
                Some(Ok((
                    tensor_axis,
                    coordinates[mesh_axis] as i64 * length,
                    length,
                )))
            })
            .collect()
    }

    pub fn replicated_mesh_axes(&self, shape: Shape) -> Result<Vec<AxisTag>, Error> {
        self.validate_shape(shape)?;
        let used = shape
            .partitions()
            .iter()
            .filter_map(|partition| match partition {
                Partition::Sharded(tag) => Some(*tag),
                _ => None,
            })
            .collect::<HashSet<_>>();
        Ok(self
            .mesh_axes()
            .map(|(tag, _)| tag)
            .filter(|tag| !used.contains(tag))
            .collect())
    }

    fn axis_size(&self, tag: AxisTag) -> Result<usize, Error> {
        self.mesh_axes()
            .find_map(|(candidate, size)| (candidate == tag).then_some(size))
            .ok_or(Error::MissingAxis(tag))
    }
}

fn mesh_product(axes: &[MeshAxis]) -> Result<usize, Error> {
    axes.iter().try_fold(1usize, |product, axis| {
        product
            .checked_mul(axis.size)
            .ok_or(Error::MeshSizeOverflow)
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    EmptyMesh,
    UnknownAxis,
    ZeroAxis(AxisTag),
    DuplicateAxis(AxisTag),
    AxisSizeOverflow {
        tag: AxisTag,
        size: usize,
    },
    MissingAxis(AxisTag),
    AxisConsumedTwice(AxisTag),
    MeshSizeOverflow,
    PartitionWithoutMesh,
    UnevenDimension {
        tag: AxisTag,
        dimension: i64,
        partitions: usize,
    },
    DeviceCount {
        required: usize,
        available: usize,
    },
    ShardOutOfBounds {
        shard: usize,
        count: usize,
    },
    Shape(nml_types::ShapeError),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMesh => formatter.write_str("logical mesh must contain at least one axis"),
            Self::UnknownAxis => formatter.write_str("logical mesh cannot use AxisTag::UNKNOWN"),
            Self::ZeroAxis(tag) => write!(formatter, "logical mesh axis {tag:?} has size zero"),
            Self::DuplicateAxis(tag) => {
                write!(formatter, "logical mesh axis {tag:?} is duplicated")
            }
            Self::AxisSizeOverflow { tag, size } => write!(
                formatter,
                "logical mesh axis {tag:?} size {size} exceeds Shardy's i64 representation"
            ),
            Self::MissingAxis(tag) => {
                write!(formatter, "tensor references absent mesh axis {tag:?}")
            }
            Self::AxisConsumedTwice(tag) => write!(
                formatter,
                "mesh axis {tag:?} shards more than one tensor dimension"
            ),
            Self::MeshSizeOverflow => formatter.write_str("logical mesh device count overflows"),
            Self::PartitionWithoutMesh => {
                formatter.write_str("tensor requests sharding without a logical mesh")
            }
            Self::UnevenDimension {
                tag,
                dimension,
                partitions,
            } => write!(
                formatter,
                "dimension {dimension} is not divisible by {partitions} partitions on {tag:?}"
            ),
            Self::DeviceCount {
                required,
                available,
            } => write!(
                formatter,
                "logical mesh requires {required} devices, platform exposes {available}"
            ),
            Self::ShardOutOfBounds { shard, count } => {
                write!(formatter, "shard {shard} is outside shard count {count}")
            }
            Self::Shape(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for Error {}

impl From<nml_types::ShapeError> for Error {
    fn from(error: nml_types::ShapeError) -> Self {
        Self::Shape(error)
    }
}
