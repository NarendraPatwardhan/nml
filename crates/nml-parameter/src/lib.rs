//! Logical model parameters and their physical representation components.
//!
//! `Tensor` and `Buffer` each remain one ordinary shaped value. A `Parameter`
//! instead describes one logical model value whose closed representation owns
//! one or more physical components. Dense storage is the one-component case.

#![forbid(unsafe_code)]

use nml_types::{DType, Shape};
use std::error::Error as StdError;
use std::fmt;
use std::sync::Arc;

/// An immutable logical model value independent of any one compiled graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Parameter {
    spec: Arc<ParameterSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParameterSpec {
    logical_name: String,
    logical_shape: Shape,
    representation: RepresentationSpec,
}

/// The admitted representation recipes.
///
/// This enum stays closed deliberately. Adding a representation must make
/// loading, sharding, lowering, accounting, and diagnostics exhaustively
/// consider it rather than registering a partial runtime plugin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RepresentationSpec {
    Dense(DenseSpec),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseSpec {
    component: ComponentSpec,
}

/// Stable identity used by executable and loaded-parameter contracts.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RepresentationId {
    kind: RepresentationKind,
    version: u16,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RepresentationKind {
    Dense,
}

/// One physical tensor required to reconstruct or execute a parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentSpec {
    role: ComponentRole,
    binding_name: String,
    artifact_name: String,
    storage: StorageSpec,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComponentRole {
    Values,
}

/// Physical storage consumed by PJRT and kernels.
///
/// Encoded formats may use an ordinary byte-shaped buffer while retaining a
/// distinct encoding here. They do not become public graph dtypes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StorageSpec {
    encoding: StorageEncoding,
    shape: Shape,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StorageEncoding {
    Dense(DType),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    EmptyLogicalName,
    EmptyArtifactName,
}

impl Parameter {
    /// Creates the one-component dense representation after artifact binding.
    pub fn dense(
        logical_name: impl Into<String>,
        artifact_name: impl Into<String>,
        shape: Shape,
    ) -> Result<Self, Error> {
        let logical_name = logical_name.into();
        let artifact_name = artifact_name.into();
        if logical_name.is_empty() {
            return Err(Error::EmptyLogicalName);
        }
        if artifact_name.is_empty() {
            return Err(Error::EmptyArtifactName);
        }
        let component = ComponentSpec {
            role: ComponentRole::Values,
            binding_name: logical_name.clone(),
            artifact_name,
            storage: StorageSpec {
                encoding: StorageEncoding::Dense(shape.dtype()),
                shape,
            },
        };
        Ok(Self {
            spec: Arc::new(ParameterSpec {
                logical_name,
                logical_shape: shape,
                representation: RepresentationSpec::Dense(DenseSpec { component }),
            }),
        })
    }

    pub fn name(&self) -> &str {
        &self.spec.logical_name
    }

    pub fn shape(&self) -> Shape {
        self.spec.logical_shape
    }

    pub fn representation(&self) -> &RepresentationSpec {
        &self.spec.representation
    }

    pub fn representation_id(&self) -> RepresentationId {
        self.spec.representation.id()
    }

    pub fn components(&self) -> &[ComponentSpec] {
        self.spec.representation.components()
    }

    pub fn dense_component(&self) -> &ComponentSpec {
        match &self.spec.representation {
            RepresentationSpec::Dense(spec) => &spec.component,
        }
    }
}

impl RepresentationSpec {
    pub const fn id(&self) -> RepresentationId {
        match self {
            Self::Dense(_) => RepresentationId {
                kind: RepresentationKind::Dense,
                version: 1,
            },
        }
    }

    pub fn components(&self) -> &[ComponentSpec] {
        match self {
            Self::Dense(spec) => std::slice::from_ref(&spec.component),
        }
    }
}

impl RepresentationId {
    pub const fn kind(self) -> RepresentationKind {
        self.kind
    }

    pub const fn version(self) -> u16 {
        self.version
    }
}

impl ComponentSpec {
    pub const fn role(&self) -> ComponentRole {
        self.role
    }

    pub fn binding_name(&self) -> &str {
        &self.binding_name
    }

    pub fn artifact_name(&self) -> &str {
        &self.artifact_name
    }

    pub const fn storage(&self) -> StorageSpec {
        self.storage
    }
}

impl StorageSpec {
    pub const fn encoding(self) -> StorageEncoding {
        self.encoding
    }

    pub const fn shape(self) -> Shape {
        self.shape
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogicalName => formatter.write_str("parameter logical name is empty"),
            Self::EmptyArtifactName => formatter.write_str("parameter artifact name is empty"),
        }
    }
}

impl StdError for Error {}
