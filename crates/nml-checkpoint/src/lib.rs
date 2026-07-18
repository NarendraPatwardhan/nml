//! Bounded checkpoint metadata and direct checkpoint-to-PJRT loading.

#![forbid(unsafe_code)]

pub mod safetensors {
    use nml_tensor::Slice;
    use nml_types::{DType, Shape};
    use serde::de::{IgnoredAny, MapAccess, Visitor};
    use serde::{Deserialize, Deserializer};
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

    pub(super) struct TensorReader {
        file: File,
        absolute_start: u64,
        byte_length: usize,
        name: String,
    }

    #[derive(Deserialize)]
    struct Index {
        #[serde(deserialize_with = "deserialize_unique_weight_map")]
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
            let index_bytes = read_bounded_metadata(index_path)?;
            let index: Index = serde_json::from_slice(&index_bytes).map_err(super::Error::Json)?;
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
            let shape = self.shape(name)?;
            self.read_with_shape(name, shape)
        }

        pub(super) fn read_with_shape(
            &self,
            name: &str,
            shape: Shape,
        ) -> Result<Slice<'static>, super::Error> {
            if !cfg!(target_endian = "little") {
                return Err(super::Error::NonNativeSafetensorsEndian);
            }
            let record = self.record(name)?;
            if !storage_compatible(record.shape, shape) {
                return Err(super::Error::ShapeMismatch {
                    name: name.to_owned(),
                    expected: shape,
                    actual: record.shape,
                });
            }
            let mut file = File::open(&record.path).map_err(super::Error::Io)?;
            file.seek(SeekFrom::Start(record.absolute_start))
                .map_err(super::Error::Io)?;
            // Axis tags and logical partitions belong to the model, not to the
            // safetensors file format. Allocate with the declared model shape
            // while reading the identical dense payload validated above.
            let mut slice = Slice::alloc(shape)?;
            let bytes = slice.contiguous_bytes_mut()?;
            if bytes.len() != record.byte_length {
                return Err(super::Error::InvalidTensorBytes(name.to_owned()));
            }
            file.read_exact(bytes).map_err(super::Error::Io)?;
            Ok(slice)
        }

        pub(super) fn reader(&self, name: &str) -> std::io::Result<TensorReader> {
            if !cfg!(target_endian = "little") {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "safetensors payload is not native-endian on this host",
                ));
            }
            let record = self.inner.records.get(name).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("missing checkpoint tensor {name:?}"),
                )
            })?;
            Ok(TensorReader {
                file: File::open(&record.path)?,
                absolute_start: record.absolute_start,
                byte_length: record.byte_length,
                name: name.to_owned(),
            })
        }

        fn record(&self, name: &str) -> Result<&Record, super::Error> {
            self.inner
                .records
                .get(name)
                .ok_or_else(|| super::Error::MissingTensor(name.to_owned()))
        }
    }

    pub(super) fn storage_compatible(stored: Shape, declared: Shape) -> bool {
        stored.dtype() == declared.dtype()
            && stored.dimensions() == declared.dimensions()
            && stored.layout() == declared.layout()
    }

    impl TensorReader {
        pub(super) fn read_at(
            &mut self,
            offset: usize,
            destination: &mut [u8],
        ) -> std::io::Result<()> {
            let end = offset.checked_add(destination.len()).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "tensor read overflows")
            })?;
            if end > self.byte_length {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("tensor {:?} read exceeds its validated extent", self.name),
                ));
            }
            let offset = u64::try_from(offset).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "tensor offset exceeds u64",
                )
            })?;
            let absolute = self.absolute_start.checked_add(offset).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "absolute tensor offset overflows",
                )
            })?;
            self.file.seek(SeekFrom::Start(absolute))?;
            self.file.read_exact(destination)
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
        validate_unique_top_level_names(&header)?;
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

    fn read_bounded_metadata(path: &Path) -> Result<Vec<u8>, super::Error> {
        let length = File::open(path)
            .and_then(|file| file.metadata())
            .map_err(super::Error::Io)?
            .len();
        if length > MAX_HEADER_BYTES as u64 {
            return Err(super::Error::MetadataTooLarge(path.to_owned()));
        }
        std::fs::read(path).map_err(super::Error::Io)
    }

    fn validate_unique_top_level_names(bytes: &[u8]) -> Result<(), super::Error> {
        struct UniqueNames;

        impl<'de> Visitor<'de> for UniqueNames {
            type Value = ();

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors metadata object with unique names")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
                let mut names = BTreeSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name.clone()) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate safetensors tensor name {name:?}"
                        )));
                    }
                    map.next_value::<IgnoredAny>()?;
                }
                Ok(())
            }
        }

        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        deserializer
            .deserialize_map(UniqueNames)
            .map_err(super::Error::Json)
    }

    fn deserialize_unique_weight_map<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<String, String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueWeightMap;

        impl<'de> Visitor<'de> for UniqueWeightMap {
            type Value = BTreeMap<String, String>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an index weight_map with unique tensor names")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = BTreeMap::new();
                while let Some((name, shard)) = map.next_entry::<String, String>()? {
                    if values.insert(name.clone(), shard).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate index tensor name {name:?}"
                        )));
                    }
                }
                Ok(values)
            }
        }

        deserializer.deserialize_map(UniqueWeightMap)
    }

    fn map_dtype(dtype: ::safetensors::Dtype) -> Result<DType, super::Error> {
        use safetensors::Dtype as S;
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

/// Checkpoint artifact resolution and physical parameter loading.
///
/// This layer deliberately has no graph builder. A `ParameterSet` may be used
/// to construct any number of graphs, while graph construction remains owned
/// by `nml_ir::ProgramBuilder` (exported by the facade as `Graph`).
pub mod io {
    use super::safetensors::TensorRegistry;
    use nml_parameter::{Parameter, StorageSpec};
    use nml_runtime::{Buffer, Loaded, LoadedParameter, Memory, ParameterTree, Platform, Sharding};
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// An immutable namespace over validated checkpoint artifacts.
    #[derive(Clone)]
    pub struct ParameterSet {
        registry: TensorRegistry,
        prefix: String,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct LoadAccounting {
        planned: usize,
        reads: usize,
        allocations: usize,
        uploads: usize,
        source_bytes: usize,
        resident_bytes: usize,
        prepared_bytes: usize,
        peak_staging_bytes: usize,
    }

    /// Owns every persistent buffer until the complete logical model has been
    /// reconstructed successfully. Any error drops the transaction and PJRT's
    /// buffer owners release all completed components; only `commit` permits
    /// their ownership to escape into `LoadedParameter` values.
    struct LoadTransaction {
        components: BTreeMap<String, Buffer>,
    }

    impl LoadTransaction {
        fn new() -> Self {
            Self {
                components: BTreeMap::new(),
            }
        }

        fn insert(&mut self, artifact: String, buffer: Buffer) {
            let previous = self.components.insert(artifact, buffer);
            debug_assert!(previous.is_none(), "load plan contains unique artifacts");
        }

        fn component(&self, artifact: &str) -> Option<&Buffer> {
            self.components.get(artifact)
        }

        fn commit(self) -> BTreeMap<String, Buffer> {
            self.components
        }
    }

    impl ParameterSet {
        pub fn new(registry: TensorRegistry) -> Self {
            Self {
                registry,
                prefix: String::new(),
            }
        }

        /// Returns another immutable view into the same artifact index.
        pub fn view(&self, prefix: &str) -> Self {
            Self {
                registry: self.registry.clone(),
                prefix: join(&self.prefix, prefix),
            }
        }

        pub fn dense(
            &self,
            name: &str,
            expected: nml_types::Shape,
            aliases: &[&str],
        ) -> Result<Parameter, super::Error> {
            self.maybe_dense(name, expected, aliases)?
                .ok_or_else(|| super::Error::MissingTensor(join(&self.prefix, name)))
        }

        pub fn maybe_dense(
            &self,
            name: &str,
            expected: nml_types::Shape,
            aliases: &[&str],
        ) -> Result<Option<Parameter>, super::Error> {
            let logical_name = join(&self.prefix, name);
            let aliases = aliases
                .iter()
                .map(|alias| join(&self.prefix, alias))
                .collect::<Vec<_>>();
            let artifact_name = if self.registry.contains(&logical_name) {
                Some(logical_name.clone())
            } else {
                let present = aliases
                    .iter()
                    .filter(|alias| self.registry.contains(alias))
                    .collect::<Vec<_>>();
                match present.as_slice() {
                    [] => None,
                    [only] => Some((*only).clone()),
                    _ => return Err(super::Error::AmbiguousAlias(logical_name)),
                }
            };
            let Some(artifact_name) = artifact_name else {
                return Ok(None);
            };
            let actual = self.registry.shape(&artifact_name)?;
            if !super::safetensors::storage_compatible(actual, expected) {
                return Err(super::Error::ShapeMismatch {
                    name: artifact_name,
                    expected,
                    actual,
                });
            }
            Parameter::dense(logical_name, artifact_name, expected)
                .map(Some)
                .map_err(super::Error::Parameter)
        }

        /// Resolves the three physical records of NML NVFP4 recipe v1.
        ///
        /// A base `<name>` owns `<name>.payload`, `<name>.block_scales`, and
        /// `<name>.global_scale`. Partially present bases are rejected rather
        /// than treated as a missing alias, because mixing components from two
        /// conversions would violate the parameter's representation identity.
        pub fn nvfp4(
            &self,
            name: &str,
            logical_shape: nml_types::Shape,
            aliases: &[&str],
        ) -> Result<Parameter, super::Error> {
            let logical_name = join(&self.prefix, name);
            let mut bases = Vec::with_capacity(aliases.len() + 1);
            bases.push(logical_name.clone());
            bases.extend(aliases.iter().map(|alias| join(&self.prefix, alias)));

            let mut complete = Vec::new();
            for base in bases {
                let parameter = Parameter::nvfp4(&logical_name, &base, logical_shape)
                    .map_err(super::Error::Parameter)?;
                let present = parameter
                    .components()
                    .iter()
                    .filter(|component| self.registry.contains(component.artifact_name()))
                    .count();
                if present == parameter.components().len() {
                    complete.push(parameter);
                } else if present != 0 {
                    return Err(super::Error::IncompleteParameterComponents(base));
                }
            }

            let parameter = match complete.as_slice() {
                [] => return Err(super::Error::MissingTensor(logical_name)),
                [only] => only.clone(),
                _ => return Err(super::Error::AmbiguousAlias(logical_name)),
            };
            for component in parameter.components() {
                let actual = self.registry.shape(component.artifact_name())?;
                let expected = component.storage().shape();
                if !super::safetensors::storage_compatible(actual, expected) {
                    return Err(super::Error::ShapeMismatch {
                        name: component.artifact_name().to_owned(),
                        expected,
                        actual,
                    });
                }
            }
            Ok(parameter)
        }

        pub fn load<T: ParameterTree>(
            &self,
            model: &T,
            platform: &Platform,
            options: &LoadOptions,
        ) -> Result<Loaded<T>, super::Error> {
            Ok(self.load_accounted(model, platform, options)?.0)
        }

        fn load_accounted<T: ParameterTree>(
            &self,
            model: &T,
            platform: &Platform,
            options: &LoadOptions,
        ) -> Result<(Loaded<T>, LoadAccounting), super::Error> {
            // Build and validate the complete physical load plan before the
            // first host allocation or device transfer.
            let mut records = BTreeMap::<String, StorageSpec>::new();
            let mut logical_parameters = BTreeMap::<String, Parameter>::new();
            let mut validation_error = None;
            model.visit_parameters("", &mut |_path, parameter| {
                if validation_error.is_some() {
                    return;
                }
                match logical_parameters.get(parameter.name()) {
                    Some(previous) if previous != parameter => {
                        validation_error = Some(super::Error::InconsistentParameterDefinition(
                            parameter.name().to_owned(),
                        ));
                        return;
                    }
                    Some(_) => {}
                    None => {
                        logical_parameters.insert(parameter.name().to_owned(), parameter.clone());
                    }
                }
                if let Err(error) = parameter.validate_sharding(&options.sharding) {
                    validation_error = Some(super::Error::Parameter(error));
                    return;
                }
                for component in parameter.components() {
                    let artifact = component.artifact_name();
                    let storage = component.storage();
                    match records.get(artifact) {
                        Some(previous) if *previous != storage => {
                            validation_error = Some(super::Error::InconsistentArtifactStorage {
                                name: artifact.to_owned(),
                                first: *previous,
                                second: storage,
                            });
                            return;
                        }
                        Some(_) => continue,
                        None => {}
                    }
                    match self.registry.shape(artifact) {
                        Ok(actual)
                            if super::safetensors::storage_compatible(actual, storage.shape()) => {}
                        Ok(actual) => {
                            validation_error = Some(super::Error::ShapeMismatch {
                                name: artifact.to_owned(),
                                expected: storage.shape(),
                                actual,
                            });
                            return;
                        }
                        Err(error) => {
                            validation_error = Some(error);
                            return;
                        }
                    }
                    records.insert(artifact.to_owned(), storage);
                }
            });
            if let Some(error) = validation_error {
                return Err(error);
            }

            let records = records.into_iter().collect::<Vec<_>>();
            let record_bytes = records
                .iter()
                .map(|(_, storage)| storage.shape().byte_count().map_err(super::Error::Shape))
                .collect::<Result<Vec<_>, _>>()?;
            let worker_count = options.parallelism.min(records.len()).max(1);
            let peak_staging_bytes = if platform.name() == "cuda" {
                record_bytes
                    .iter()
                    .map(|bytes| {
                        options
                            .staging_buffers
                            .checked_mul((*bytes).min(options.chunk_bytes))
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or(super::Error::InvalidLoadOption(
                        "staging byte bound overflows",
                    ))?
                    .into_iter()
                    .max()
                    .unwrap_or(0)
            } else {
                let mut largest = record_bytes.clone();
                largest.sort_unstable_by(|left, right| right.cmp(left));
                largest
                    .into_iter()
                    .take(worker_count)
                    .try_fold(0usize, usize::checked_add)
                    .ok_or(super::Error::InvalidLoadOption(
                        "staging byte bound overflows",
                    ))?
            };
            let mut accounting = LoadAccounting {
                planned: records.len(),
                source_bytes: record_bytes
                    .iter()
                    .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
                    .ok_or(super::Error::InvalidLoadOption(
                        "source byte accounting overflows",
                    ))?,
                peak_staging_bytes,
                ..LoadAccounting::default()
            };
            let mut transaction = LoadTransaction::new();

            if platform.name() == "cuda" {
                for (completed, (artifact, storage)) in records.iter().enumerate() {
                    let mut reader = self.registry.reader(artifact).map_err(super::Error::Io)?;
                    let buffer = platform
                        .upload_component_from(
                            *storage,
                            options.sharding.clone(),
                            options.memory,
                            options.staging_buffers,
                            options.chunk_bytes,
                            |offset, destination| reader.read_at(offset, destination),
                        )
                        .map_err(super::Error::Runtime)?;
                    accounting.reads += 1;
                    accounting.allocations += 1;
                    accounting.uploads += 1;
                    accounting.resident_bytes = accounting
                        .resident_bytes
                        .checked_add(buffer.byte_count().map_err(super::Error::Runtime)?)
                        .ok_or(super::Error::InvalidLoadOption(
                            "resident byte accounting overflows",
                        ))?;
                    transaction.insert(artifact.clone(), buffer);
                    if let Some(progress) = &options.progress {
                        progress(completed + 1, records.len());
                    }
                }
            } else {
                let next = AtomicUsize::new(0);
                std::thread::scope(|scope| -> Result<(), super::Error> {
                    // A rendezvous channel bounds live host components to the
                    // worker count. No worker reads another component until
                    // the main thread has accepted its current one.
                    let (sender, receiver) = std::sync::mpsc::sync_channel(0);
                    for _ in 0..worker_count {
                        let sender = sender.clone();
                        let registry = self.registry.clone();
                        let records = &records;
                        let next = &next;
                        scope.spawn(move || {
                            loop {
                                let index = next.fetch_add(1, Ordering::Relaxed);
                                let Some((artifact, storage)) = records.get(index) else {
                                    break;
                                };
                                if sender
                                    .send((
                                        artifact.clone(),
                                        registry.read_with_shape(artifact, storage.shape()),
                                    ))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        });
                    }
                    drop(sender);
                    for completed in 0..records.len() {
                        let (artifact, slice) = receiver
                            .recv()
                            .map_err(|_| super::Error::LoaderWorkerFailed)?;
                        let slice = slice?;
                        accounting.reads += 1;
                        let buffer = platform
                            .upload(&slice, options.sharding.clone(), options.memory)
                            .map_err(super::Error::Runtime)?;
                        accounting.allocations += 1;
                        accounting.uploads += 1;
                        accounting.resident_bytes = accounting
                            .resident_bytes
                            .checked_add(buffer.byte_count().map_err(super::Error::Runtime)?)
                            .ok_or(super::Error::InvalidLoadOption(
                                "resident byte accounting overflows",
                            ))?;
                        transaction.insert(artifact, buffer);
                        if let Some(progress) = &options.progress {
                            progress(completed + 1, records.len());
                        }
                    }
                    Ok(())
                })?;
            }

            debug_assert_eq!(accounting.reads, accounting.planned);
            debug_assert_eq!(accounting.allocations, accounting.planned);
            debug_assert_eq!(accounting.uploads, accounting.planned);
            // Source-layout execution performs no one-time repacking. Future
            // prepared layouts must account their additional persistent bytes
            // separately instead of hiding them in resident source storage.
            debug_assert_eq!(accounting.prepared_bytes, 0);
            let loaded_model = model.load_parameters("", &mut |_path, parameter| {
                let components = parameter
                    .components()
                    .iter()
                    .map(|component| {
                        transaction
                            .component(component.artifact_name())
                            .expect("validated physical component was loaded")
                            .clone()
                    })
                    .collect();
                LoadedParameter::new(parameter.clone(), components).map_err(super::Error::Runtime)
            })?;
            // This explicit commit documents the transactional ownership
            // boundary. The map's handles are tied-component owners and are
            // intentionally released after the loaded model holds its clones.
            drop(transaction.commit());
            Ok((loaded_model, accounting))
        }
    }

    /// Bounded loader policy without exposing private plans or accounting.
    pub struct LoadOptions {
        sharding: Sharding,
        memory: Memory,
        parallelism: usize,
        staging_buffers: usize,
        chunk_bytes: usize,
        progress: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    }

    impl LoadOptions {
        pub fn new(sharding: Sharding) -> Self {
            Self {
                sharding,
                memory: Memory::Default,
                parallelism: 1,
                staging_buffers: 2,
                chunk_bytes: 16 * 1024 * 1024,
                progress: None,
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

        /// Reports `(completed_components, total_unique_components)` after a
        /// component becomes persistent.
        pub fn progress(mut self, callback: impl Fn(usize, usize) + Send + Sync + 'static) -> Self {
            self.progress = Some(Arc::new(callback));
            self
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use nml_parameter::Parameter;
        use nml_runtime::ParameterTree;
        use nml_types::{DType, Shape};
        use safetensors::tensor::{Dtype, View};
        use std::borrow::Cow;
        use std::collections::BTreeMap;

        struct Pair {
            first: Parameter,
            second: Parameter,
        }

        struct LoadedPair {
            first: LoadedParameter,
            second: LoadedParameter,
        }

        impl ParameterTree for Pair {
            type Loaded = LoadedPair;

            fn visit_parameters(&self, prefix: &str, visitor: &mut dyn FnMut(&str, &Parameter)) {
                visitor(&join(prefix, "first"), &self.first);
                visitor(&join(prefix, "second"), &self.second);
            }

            fn visit_loaded(
                loaded: &Self::Loaded,
                prefix: &str,
                visitor: &mut dyn FnMut(&str, &LoadedParameter),
            ) {
                visitor(&join(prefix, "first"), &loaded.first);
                visitor(&join(prefix, "second"), &loaded.second);
            }

            fn load_parameters<E>(
                &self,
                prefix: &str,
                resolve: &mut impl FnMut(&str, &Parameter) -> Result<LoadedParameter, E>,
            ) -> Result<Self::Loaded, E> {
                Ok(LoadedPair {
                    first: resolve(&join(prefix, "first"), &self.first)?,
                    second: resolve(&join(prefix, "second"), &self.second)?,
                })
            }
        }

        struct TensorData(Vec<u8>);

        impl View for &TensorData {
            fn dtype(&self) -> Dtype {
                Dtype::F32
            }

            fn shape(&self) -> &[usize] {
                &[2]
            }

            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.0)
            }

            fn data_len(&self) -> usize {
                self.0.len()
            }
        }

        #[test]
        fn tied_components_are_planned_read_allocated_and_uploaded_once() {
            let root = std::env::temp_dir().join(format!(
                "nml-loader-accounting-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let tensor = TensorData(
                [1.0f32, -2.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            );
            let bytes =
                safetensors::serialize(BTreeMap::from([("shared", &tensor)]), None).unwrap();
            std::fs::write(root.join("model.safetensors"), bytes).unwrap();

            let registry = TensorRegistry::from_path(&root).unwrap();
            let parameters = ParameterSet::new(registry);
            let shape = Shape::new(DType::F32, &[2]).unwrap();
            let model = Pair {
                first: parameters.dense("first", shape, &["shared"]).unwrap(),
                second: parameters.dense("second", shape, &["shared"]).unwrap(),
            };
            let platform = Platform::cpu_with_devices(1).unwrap();
            let (loaded, accounting) = parameters
                .load_accounted(&model, &platform, &LoadOptions::new(Sharding::single()))
                .unwrap();

            assert_eq!(accounting.planned, 1);
            assert_eq!(accounting.reads, 1);
            assert_eq!(accounting.allocations, 1);
            assert_eq!(accounting.uploads, 1);
            assert_eq!(accounting.source_bytes, 8);
            assert_eq!(accounting.resident_bytes, 8);
            assert_eq!(accounting.prepared_bytes, 0);
            assert_eq!(accounting.peak_staging_bytes, 8);
            let first = loaded.first.components().next().unwrap().1;
            let second = loaded.second.components().next().unwrap().1;
            assert!(!first.is_uniquely_owned());
            assert!(!second.is_uniquely_owned());

            std::fs::remove_dir_all(root).unwrap();
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
    Parameter(nml_parameter::Error),
    Runtime(nml_runtime::Error),
    MissingRepository(std::path::PathBuf),
    MissingShard(std::path::PathBuf),
    PathEscapesRepository(std::path::PathBuf),
    HeaderTooLarge,
    MetadataTooLarge(std::path::PathBuf),
    NonNativeSafetensorsEndian,
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
    InconsistentArtifactStorage {
        name: String,
        first: nml_parameter::StorageSpec,
        second: nml_parameter::StorageSpec,
    },
    InconsistentParameterDefinition(String),
    IncompleteParameterComponents(String),
    InvalidLoadOption(&'static str),
    LoaderWorkerFailed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(f),
            Self::Json(error) => error.fmt(f),
            Self::Tensor(error) => error.fmt(f),
            Self::Shape(error) => error.fmt(f),
            Self::Parameter(error) => error.fmt(f),
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
            Self::MetadataTooLarge(path) => write!(f, "checkpoint metadata is too large: {}", path.display()),
            Self::NonNativeSafetensorsEndian => f.write_str("safetensors payloads are little-endian and cannot be transferred without conversion on this host"),
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
            Self::InconsistentArtifactStorage { name, first, second } => write!(
                f,
                "checkpoint artifact {name:?} is used with inconsistent physical storage contracts: {first:?} and {second:?}"
            ),
            Self::InconsistentParameterDefinition(name) => write!(
                f,
                "logical parameter {name:?} is declared with inconsistent representation or artifact binding"
            ),
            Self::IncompleteParameterComponents(name) => write!(
                f,
                "NVFP4 artifact base {name:?} has only some required physical components"
            ),
            Self::InvalidLoadOption(message) => write!(f, "invalid load option: {message}"),
            Self::LoaderWorkerFailed => {
                f.write_str("checkpoint reader terminated before completing the load plan")
            }
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
