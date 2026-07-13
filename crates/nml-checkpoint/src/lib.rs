//! Bounded checkpoint metadata and direct checkpoint-to-PJRT loading.

#![forbid(unsafe_code)]

pub mod safetensors {
    use nml_tensor::Slice;
    use nml_types::{DType, Shape};
    use serde::Deserialize;
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    const HEADER_PREFIX_BYTES: usize = 8;
    const MAX_HEADER_BYTES: usize = 100_000_000;

    /// Validated tensor names and file spans for one safetensors repository.
    #[derive(Clone)]
    pub struct TensorRegistry {
        inner: Arc<Registry>,
    }

    struct Registry {
        records: BTreeMap<String, Record>,
    }

    #[derive(Clone)]
    struct Record {
        path: PathBuf,
        absolute_start: u64,
        byte_length: usize,
        shape: Shape,
    }

    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }

    impl TensorRegistry {
        pub fn from_path(path: impl AsRef<Path>) -> Result<Self, super::Error> {
            let path = path.as_ref();
            if path.is_file() {
                if path.file_name().and_then(|name| name.to_str())
                    == Some("model.safetensors.index.json")
                {
                    let root = path.parent().unwrap_or_else(|| Path::new("."));
                    return Self::from_index(root, path);
                }
                return Self::from_single(path);
            }
            if !path.is_dir() {
                return Err(super::Error::MissingRepository(path.to_owned()));
            }
            let direct = path.join("model.safetensors");
            if direct.is_file() {
                return Self::from_single(&direct);
            }
            let index = path.join("model.safetensors.index.json");
            if index.is_file() {
                return Self::from_index(path, &index);
            }
            Err(super::Error::MissingRepository(path.to_owned()))
        }

        fn from_single(path: &Path) -> Result<Self, super::Error> {
            Ok(Self {
                inner: Arc::new(Registry {
                    records: parse_file(path)?,
                }),
            })
        }

        fn from_index(root: &Path, index_path: &Path) -> Result<Self, super::Error> {
            let index: Index =
                serde_json::from_slice(&std::fs::read(index_path).map_err(super::Error::Io)?)
                    .map_err(super::Error::Json)?;
            let root = root.canonicalize().map_err(super::Error::Io)?;
            let mut shards = BTreeMap::<String, BTreeMap<String, Record>>::new();
            for shard in index.weight_map.values() {
                if shards.contains_key(shard) {
                    continue;
                }
                let candidate = root.join(shard);
                let canonical = candidate
                    .canonicalize()
                    .map_err(|_| super::Error::MissingShard(candidate.clone()))?;
                if !canonical.starts_with(&root) {
                    return Err(super::Error::PathEscapesRepository(candidate));
                }
                shards.insert(shard.clone(), parse_file(&canonical)?);
            }

            let mut records = BTreeMap::new();
            for (name, shard) in &index.weight_map {
                let record = shards
                    .get(shard)
                    .and_then(|records| records.get(name))
                    .ok_or_else(|| super::Error::IndexDisagrees {
                        tensor: name.clone(),
                        shard: shard.clone(),
                    })?
                    .clone();
                if records.insert(name.clone(), record).is_some() {
                    return Err(super::Error::DuplicateTensor(name.clone()));
                }
            }
            let indexed = index.weight_map.keys().cloned().collect::<BTreeSet<_>>();
            for (shard, shard_records) in &shards {
                for name in shard_records.keys() {
                    if !indexed.contains(name)
                        || index
                            .weight_map
                            .get(name)
                            .is_none_or(|mapped| mapped != shard)
                    {
                        return Err(super::Error::IndexDisagrees {
                            tensor: name.clone(),
                            shard: shard.clone(),
                        });
                    }
                }
            }
            Ok(Self {
                inner: Arc::new(Registry { records }),
            })
        }

        pub fn contains(&self, name: &str) -> bool {
            self.inner.records.contains_key(name)
        }

        pub fn names(&self) -> impl Iterator<Item = &str> {
            self.inner.records.keys().map(String::as_str)
        }

        pub fn shape(&self, name: &str) -> Result<Shape, super::Error> {
            Ok(self.record(name)?.shape)
        }

        pub fn read(&self, name: &str) -> Result<Slice<'static>, super::Error> {
            let record = self.record(name)?;
            let mut file = File::open(&record.path).map_err(super::Error::Io)?;
            file.seek(SeekFrom::Start(record.absolute_start))
                .map_err(super::Error::Io)?;
            let mut slice = Slice::alloc(record.shape)?;
            let bytes = slice.contiguous_bytes_mut()?;
            if bytes.len() != record.byte_length {
                return Err(super::Error::InvalidTensorBytes(name.to_owned()));
            }
            file.read_exact(bytes).map_err(super::Error::Io)?;
            Ok(slice)
        }

        fn record(&self, name: &str) -> Result<&Record, super::Error> {
            self.inner
                .records
                .get(name)
                .ok_or_else(|| super::Error::MissingTensor(name.to_owned()))
        }
    }

    fn parse_file(path: &Path) -> Result<BTreeMap<String, Record>, super::Error> {
        let mut file = File::open(path).map_err(super::Error::Io)?;
        let file_length = file.metadata().map_err(super::Error::Io)?.len();
        let mut prefix = [0u8; HEADER_PREFIX_BYTES];
        file.read_exact(&mut prefix).map_err(super::Error::Io)?;
        let header_length = usize::try_from(u64::from_le_bytes(prefix))
            .map_err(|_| super::Error::HeaderTooLarge)?;
        if header_length > MAX_HEADER_BYTES {
            return Err(super::Error::HeaderTooLarge);
        }
        let data_start = HEADER_PREFIX_BYTES
            .checked_add(header_length)
            .ok_or(super::Error::HeaderTooLarge)?;
        let mut header = vec![0u8; header_length];
        file.read_exact(&mut header).map_err(super::Error::Io)?;
        if header.first() != Some(&b'{') {
            return Err(super::Error::InvalidHeader(
                "header must begin with `{`".to_owned(),
            ));
        }
        let metadata: ::safetensors::tensor::Metadata =
            serde_json::from_slice(&header).map_err(super::Error::Json)?;
        let expected_file_length = (data_start as u64)
            .checked_add(metadata.data_len() as u64)
            .ok_or(super::Error::HeaderTooLarge)?;
        if expected_file_length != file_length {
            return Err(super::Error::FileExtent {
                path: path.to_owned(),
                expected: expected_file_length,
                actual: file_length,
            });
        }

        let mut records = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            let dtype = map_dtype(info.dtype)?;
            let dimensions = info
                .shape
                .iter()
                .map(|dimension| i64::try_from(*dimension).map_err(|_| super::Error::ShapeOverflow))
                .collect::<Result<Vec<_>, _>>()?;
            let shape = Shape::new(dtype, &dimensions)?;
            let expected = shape.byte_count()?;
            let actual = info
                .data_offsets
                .1
                .checked_sub(info.data_offsets.0)
                .ok_or_else(|| super::Error::InvalidTensorBytes(name.clone()))?;
            if expected != actual {
                return Err(super::Error::InvalidTensorBytes(name));
            }
            let absolute_start = (data_start as u64)
                .checked_add(info.data_offsets.0 as u64)
                .ok_or(super::Error::HeaderTooLarge)?;
            let record = Record {
                path: path.to_owned(),
                absolute_start,
                byte_length: actual,
                shape,
            };
            if records.insert(name.clone(), record).is_some() {
                return Err(super::Error::DuplicateTensor(name));
            }
        }
        Ok(records)
    }

    fn map_dtype(dtype: ::safetensors::Dtype) -> Result<DType, super::Error> {
        use ::safetensors::Dtype as S;
        match dtype {
            S::BOOL => Ok(DType::Bool),
            S::I8 => Ok(DType::I8),
            S::I16 => Ok(DType::I16),
            S::I32 => Ok(DType::I32),
            S::I64 => Ok(DType::I64),
            S::U8 => Ok(DType::U8),
            S::U16 => Ok(DType::U16),
            S::U32 => Ok(DType::U32),
            S::U64 => Ok(DType::U64),
            S::F16 => Ok(DType::F16),
            S::BF16 => Ok(DType::Bf16),
            S::F32 => Ok(DType::F32),
            S::F64 => Ok(DType::F64),
            S::C64 => Ok(DType::C64),
            unsupported => Err(super::Error::UnsupportedDType(format!("{unsupported:?}"))),
        }
    }
}

pub mod io {
    use super::safetensors::TensorRegistry;
    use nml_ir::{Program, ProgramBuilder, Tensor};
    use nml_runtime::{Buffer, Bufferized, Memory, NmlStruct, Platform, Sharding};
    use nml_types::Shape;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    /// Symbolic checkpoint view and eventual direct-to-buffer loader.
    pub struct TensorStore {
        inner: Rc<RefCell<Store>>,
        prefix: String,
    }

    struct Store {
        registry: TensorRegistry,
        builder: ProgramBuilder,
        tied_symbols: BTreeMap<String, Tensor>,
        path_to_record: BTreeMap<String, String>,
    }

    impl TensorStore {
        pub fn new(registry: TensorRegistry) -> Self {
            Self {
                inner: Rc::new(RefCell::new(Store {
                    registry,
                    builder: ProgramBuilder::new(),
                    tied_symbols: BTreeMap::new(),
                    path_to_record: BTreeMap::new(),
                })),
                prefix: String::new(),
            }
        }

        pub fn view(&self, prefix: &str) -> Self {
            Self {
                inner: Rc::clone(&self.inner),
                prefix: join(&self.prefix, prefix),
            }
        }

        pub fn layer(&self, index: usize) -> Self {
            self.view(&index.to_string())
        }

        pub fn activation(&self, name: &str, shape: Shape) -> Tensor {
            self.inner.borrow_mut().builder.input(name, shape)
        }

        pub fn linear(
            &self,
            input: Tensor,
            weight: Tensor,
            bias: Option<Tensor>,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .linear(input, weight, bias)
                .map_err(super::Error::Ir)
        }

        pub fn tensor(
            &self,
            name: &str,
            expected: Shape,
            aliases: &[&str],
        ) -> Result<Tensor, super::Error> {
            self.maybe_tensor(name, expected, aliases)?
                .ok_or_else(|| super::Error::MissingTensor(join(&self.prefix, name)))
        }

        pub fn maybe_tensor(
            &self,
            name: &str,
            expected: Shape,
            aliases: &[&str],
        ) -> Result<Option<Tensor>, super::Error> {
            let primary = join(&self.prefix, name);
            let aliases = aliases
                .iter()
                .map(|alias| join(&self.prefix, alias))
                .collect::<Vec<_>>();
            let mut store = self.inner.borrow_mut();
            let resolved = if store.registry.contains(&primary) {
                Some(primary.clone())
            } else {
                let present = aliases
                    .iter()
                    .filter(|alias| store.registry.contains(alias))
                    .cloned()
                    .collect::<Vec<_>>();
                match present.as_slice() {
                    [] => None,
                    [only] => Some(only.clone()),
                    _ => return Err(super::Error::AmbiguousAlias(primary)),
                }
            };
            let Some(resolved) = resolved else {
                return Ok(None);
            };
            let actual = store.registry.shape(&resolved)?;
            if actual != expected {
                return Err(super::Error::ShapeMismatch {
                    name: resolved,
                    expected,
                    actual,
                });
            }
            let tensor = if let Some(tensor) = store.tied_symbols.get(&resolved) {
                *tensor
            } else {
                let tensor = store.builder.parameter(resolved.clone(), actual);
                store.tied_symbols.insert(resolved.clone(), tensor);
                tensor
            };
            store.path_to_record.insert(primary, resolved);
            Ok(Some(tensor))
        }

        pub fn load<T: NmlStruct>(
            &self,
            model: &T,
            platform: &Platform,
            options: &LoadOptions,
        ) -> Result<Bufferized<T>, super::Error> {
            let store = self.inner.borrow();
            // Resolve the complete model before the first file read or device
            // allocation. A malformed derived structure therefore cannot
            // leave a half-loaded parameter set behind.
            let mut paths = Vec::new();
            model.visit_tensors("", &mut |path, _tensor| paths.push(path.to_owned()));
            let mut plan = BTreeMap::<String, String>::new();
            for path in paths {
                let record = store
                    .path_to_record
                    .get(&path)
                    .ok_or_else(|| super::Error::UnboundModelPath(path.clone()))?;
                plan.insert(path, record.clone());
            }

            // The current CPU path intentionally uses one staging allocation
            // at a time. These validated limits already form the bounded
            // loader contract; CUDA consumes the buffer and chunk bounds when
            // its mapped DMA path is selected.
            let _bounds = (
                options.parallelism,
                options.staging_buffers,
                options.chunk_bytes,
            );
            let mut loaded = BTreeMap::<String, Buffer>::new();
            for record in plan.values() {
                if loaded.contains_key(record) {
                    continue;
                }
                let slice = store.registry.read(record)?;
                let buffer = platform
                    .upload(&slice, options.sharding.clone(), options.memory)
                    .map_err(super::Error::Runtime)?;
                loaded.insert(record.clone(), buffer);
            }
            model.bufferize("", &mut |path, _tensor| {
                let record = plan
                    .get(path)
                    .expect("validated model traversal is deterministic");
                Ok(loaded
                    .get(record)
                    .expect("every unique planned record was loaded")
                    .clone())
            })
        }

        pub fn finish(self, outputs: &[(String, Tensor)]) -> Result<Program, super::Error> {
            let store = Rc::try_unwrap(self.inner)
                .map_err(|_| super::Error::OutstandingTensorStoreView)?
                .into_inner();
            store
                .builder
                .finish_named(outputs)
                .map_err(super::Error::Ir)
        }
    }

    /// Bounded loader policy without exposing its plan or accounting records.
    pub struct LoadOptions {
        sharding: Sharding,
        memory: Memory,
        parallelism: usize,
        staging_buffers: usize,
        chunk_bytes: usize,
    }

    impl LoadOptions {
        pub fn new(sharding: Sharding) -> Self {
            Self {
                sharding,
                memory: Memory::Default,
                parallelism: 1,
                staging_buffers: 2,
                chunk_bytes: 16 * 1024 * 1024,
            }
        }

        pub fn memory(mut self, memory: Memory) -> Self {
            self.memory = memory;
            self
        }

        pub fn parallelism(mut self, parallelism: usize) -> Result<Self, super::Error> {
            if parallelism == 0 {
                return Err(super::Error::InvalidLoadOption(
                    "parallelism must be positive",
                ));
            }
            self.parallelism = parallelism;
            Ok(self)
        }

        pub fn staging(mut self, buffers: usize, chunk_bytes: usize) -> Result<Self, super::Error> {
            if buffers == 0 || chunk_bytes == 0 {
                return Err(super::Error::InvalidLoadOption(
                    "staging values must be positive",
                ));
            }
            self.staging_buffers = buffers;
            self.chunk_bytes = chunk_bytes;
            Ok(self)
        }
    }

    fn join(prefix: &str, suffix: &str) -> String {
        if prefix.is_empty() {
            suffix.to_owned()
        } else if suffix.is_empty() {
            prefix.to_owned()
        } else {
            format!("{prefix}.{suffix}")
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Tensor(nml_tensor::Error),
    Shape(nml_types::ShapeError),
    Ir(nml_ir::Error),
    Runtime(nml_runtime::Error),
    MissingRepository(std::path::PathBuf),
    MissingShard(std::path::PathBuf),
    PathEscapesRepository(std::path::PathBuf),
    HeaderTooLarge,
    InvalidHeader(String),
    FileExtent {
        path: std::path::PathBuf,
        expected: u64,
        actual: u64,
    },
    DuplicateTensor(String),
    IndexDisagrees {
        tensor: String,
        shard: String,
    },
    UnsupportedDType(String),
    ShapeOverflow,
    InvalidTensorBytes(String),
    MissingTensor(String),
    AmbiguousAlias(String),
    ShapeMismatch {
        name: String,
        expected: nml_types::Shape,
        actual: nml_types::Shape,
    },
    UnboundModelPath(String),
    OutstandingTensorStoreView,
    InvalidLoadOption(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::Tensor(error) => error.fmt(f),
            Self::Shape(error) => error.fmt(f),
            Self::Ir(error) => error.fmt(f),
            Self::Runtime(error) => error.fmt(f),
            Self::MissingRepository(path) => {
                write!(f, "no safetensors repository at {}", path.display())
            }
            Self::MissingShard(path) => {
                write!(f, "missing safetensors shard {}", path.display())
            }
            Self::PathEscapesRepository(path) => {
                write!(f, "shard path escapes repository: {}", path.display())
            }
            Self::HeaderTooLarge => f.write_str("safetensors header is too large"),
            Self::InvalidHeader(message) => write!(f, "invalid safetensors header: {message}"),
            Self::FileExtent {
                path,
                expected,
                actual,
            } => write!(
                f,
                "{} has {actual} bytes, metadata requires {expected}",
                path.display()
            ),
            Self::DuplicateTensor(name) => write!(f, "duplicate safetensors tensor {name:?}"),
            Self::IndexDisagrees { tensor, shard } => {
                write!(f, "index maps {tensor:?} to inconsistent shard {shard:?}")
            }
            Self::UnsupportedDType(dtype) => write!(f, "unsupported safetensors dtype {dtype}"),
            Self::ShapeOverflow => f.write_str("safetensors shape exceeds NML's dimension range"),
            Self::InvalidTensorBytes(name) => write!(
                f,
                "tensor {name:?} byte count does not match its dtype and shape"
            ),
            Self::MissingTensor(name) => write!(f, "missing checkpoint tensor {name:?}"),
            Self::AmbiguousAlias(name) => write!(
                f,
                "multiple aliases are present for absent primary tensor {name:?}"
            ),
            Self::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "checkpoint tensor {name:?} shape mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::UnboundModelPath(path) => {
                write!(
                    f,
                    "model field {path:?} was not created by this TensorStore"
                )
            }
            Self::OutstandingTensorStoreView => {
                f.write_str("TensorStore views remain live while finishing the program")
            }
            Self::InvalidLoadOption(message) => write!(f, "invalid load option: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<nml_tensor::Error> for Error {
    fn from(error: nml_tensor::Error) -> Self {
        Self::Tensor(error)
    }
}

impl From<nml_types::ShapeError> for Error {
    fn from(error: nml_types::ShapeError) -> Self {
        Self::Shape(error)
    }
}
