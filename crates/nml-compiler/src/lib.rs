//! Compilation orchestration shared unchanged by CPU and CUDA clients.
//!
//! Backends provide distinct PJRT plugins, but portable StableHLO negotiation,
//! XLA option serialization, and loaded-executable ownership are common.

#![forbid(unsafe_code)]

use nml_pjrt::{Client, LoadedExecutable, StableHloVersion};
use std::error::Error as StdError;
use std::fmt;

pub fn compile(
    client: &Client,
    program: &nml_ir::Program,
    options: &nml_xla::CompileOptions,
) -> Result<LoadedExecutable, Error> {
    let context = nml_mlir::Context::new();
    let module = program.module(&context)?;
    let version = negotiate_stablehlo_version(client.stablehlo_version()?)?;
    let artifact = module.portable_artifact(&version)?;
    let options = options.serialize()?;
    Ok(client.compile(&artifact, &options)?)
}

/// Chooses the newest portable StableHLO version understood by both sides.
///
/// A plugin older than the compiler's minimum is rejected explicitly. Passing
/// such a version to the serializer would otherwise turn a compatibility
/// failure into a misleading generic MLIR serialization error.
pub fn negotiate_stablehlo_version(plugin: Option<StableHloVersion>) -> Result<String, Error> {
    let current = nml_mlir::stablehlo_current_version();
    let current_parts = parse_version(&current)?;
    let minimum = nml_mlir::stablehlo_minimum_version();
    let minimum_parts = parse_version(&minimum)?;
    if minimum_parts > current_parts {
        return Err(Error::InvalidStableHloRange { minimum, current });
    }
    Ok(match plugin {
        Some(plugin) if plugin < minimum_parts => {
            return Err(Error::UnsupportedStableHloVersion {
                plugin,
                minimum: minimum_parts,
            });
        }
        Some(plugin) if plugin < current_parts => {
            format!("{}.{}.{}", plugin.major, plugin.minor, plugin.patch)
        }
        Some(_) => current,
        None => minimum,
    })
}

fn parse_version(version: &str) -> Result<StableHloVersion, Error> {
    let mut parts = version.split('.');
    let major = parts.next().and_then(|part| part.parse().ok());
    let minor = parts.next().and_then(|part| part.parse().ok());
    let patch = parts.next().and_then(|part| part.parse().ok());
    if parts.next().is_some() {
        return Err(Error::InvalidStableHloVersion(version.to_owned()));
    }
    match (major, minor, patch) {
        (Some(major), Some(minor), Some(patch)) => Ok(StableHloVersion {
            major,
            minor,
            patch,
        }),
        _ => Err(Error::InvalidStableHloVersion(version.to_owned())),
    }
}

#[derive(Debug)]
pub enum Error {
    Ir(nml_ir::Error),
    Mlir(nml_mlir::Error),
    Xla(nml_xla::Error),
    Pjrt(nml_pjrt::Error),
    InvalidStableHloVersion(String),
    InvalidStableHloRange {
        minimum: String,
        current: String,
    },
    UnsupportedStableHloVersion {
        plugin: StableHloVersion,
        minimum: StableHloVersion,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ir(error) => error.fmt(formatter),
            Self::Mlir(error) => error.fmt(formatter),
            Self::Xla(error) => error.fmt(formatter),
            Self::Pjrt(error) => error.fmt(formatter),
            Self::InvalidStableHloVersion(version) => {
                write!(formatter, "invalid StableHLO semantic version {version:?}")
            }
            Self::InvalidStableHloRange { minimum, current } => write!(
                formatter,
                "StableHLO compiler version range is invalid: minimum {minimum}, current {current}"
            ),
            Self::UnsupportedStableHloVersion { plugin, minimum } => write!(
                formatter,
                "PJRT accepts StableHLO {}.{}.{}, older than the compiler minimum {}.{}.{}",
                plugin.major,
                plugin.minor,
                plugin.patch,
                minimum.major,
                minimum.minor,
                minimum.patch,
            ),
        }
    }
}

impl StdError for Error {}

impl From<nml_ir::Error> for Error {
    fn from(error: nml_ir::Error) -> Self {
        Self::Ir(error)
    }
}
impl From<nml_mlir::Error> for Error {
    fn from(error: nml_mlir::Error) -> Self {
        Self::Mlir(error)
    }
}
impl From<nml_xla::Error> for Error {
    fn from(error: nml_xla::Error) -> Self {
        Self::Xla(error)
    }
}
impl From<nml_pjrt::Error> for Error {
    fn from(error: nml_pjrt::Error) -> Self {
        Self::Pjrt(error)
    }
}
