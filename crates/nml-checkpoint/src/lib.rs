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

pub mod io {
    use super::safetensors::TensorRegistry;
    use nml_ir::{Program, ProgramBuilder, Tensor};
    use nml_runtime::{Buffer, Bufferized, Memory, NmlStruct, Platform, Sharding};
    use nml_tensor::{Element, Slice};
    use nml_types::{DType, Partition, Shape};
    use std::cell::RefCell;
    use std::collections::{BTreeMap, BTreeSet};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        record_shapes: BTreeMap<String, Shape>,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct LoadAccounting {
        planned: usize,
        reads: usize,
        allocations: usize,
        uploads: usize,
        peak_staging_bytes: usize,
    }

    impl TensorStore {
        pub fn new(registry: TensorRegistry) -> Self {
            Self {
                inner: Rc::new(RefCell::new(Store {
                    registry,
                    builder: ProgramBuilder::new(),
                    tied_symbols: BTreeMap::new(),
                    path_to_record: BTreeMap::new(),
                    record_shapes: BTreeMap::new(),
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

        pub fn constant(&self, value: &Slice<'_>) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .constant(value)
                .map_err(super::Error::Ir)
        }

        pub fn scalar<T: Element>(&self, value: T) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .scalar(value)
                .map_err(super::Error::Ir)
        }

        pub fn add(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .add(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn subtract(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .subtract(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn multiply(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .multiply(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn divide(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .divide(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn power(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .power(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn remainder(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .remainder(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn minimum(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .minimum(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn maximum(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .maximum(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn clamp(
            &self,
            input: Tensor,
            minimum: Tensor,
            maximum: Tensor,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .clamp(input, minimum, maximum)
                .map_err(super::Error::Ir)
        }

        pub fn negate(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .negate(input)
                .map_err(super::Error::Ir)
        }

        pub fn abs(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .abs(input)
                .map_err(super::Error::Ir)
        }

        pub fn equal(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .equal(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn not_equal(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .not_equal(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn greater(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .greater(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn greater_equal(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .greater_equal(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn less(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .less(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn less_equal(&self, left: Tensor, right: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .less_equal(left, right)
                .map_err(super::Error::Ir)
        }

        pub fn select(
            &self,
            predicate: Tensor,
            on_true: Tensor,
            on_false: Tensor,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .select(predicate, on_true, on_false)
                .map_err(super::Error::Ir)
        }

        pub fn convert(&self, input: Tensor, dtype: DType) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .convert(input, dtype)
                .map_err(super::Error::Ir)
        }

        pub fn reshape(&self, input: Tensor, shape: Shape) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .reshape(input, shape)
                .map_err(super::Error::Ir)
        }

        pub fn transpose(
            &self,
            input: Tensor,
            permutation: &[usize],
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .transpose(input, permutation)
                .map_err(super::Error::Ir)
        }

        pub fn broadcast_in_dim(
            &self,
            input: Tensor,
            shape: Shape,
            dimensions: &[usize],
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .broadcast_in_dim(input, shape, dimensions)
                .map_err(super::Error::Ir)
        }

        pub fn iota(&self, shape: Shape, axis: usize) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .iota(shape, axis)
                .map_err(super::Error::Ir)
        }

        pub fn concatenate(&self, inputs: &[Tensor], axis: usize) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .concatenate(inputs, axis)
                .map_err(super::Error::Ir)
        }

        pub fn slice(
            &self,
            input: Tensor,
            starts: &[i64],
            limits: &[i64],
            strides: &[i64],
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .slice(input, starts, limits, strides)
                .map_err(super::Error::Ir)
        }

        pub fn dynamic_slice(
            &self,
            input: Tensor,
            starts: &[Tensor],
            sizes: &[i64],
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .dynamic_slice(input, starts, sizes)
                .map_err(super::Error::Ir)
        }

        pub fn dynamic_update_slice(
            &self,
            input: Tensor,
            update: Tensor,
            starts: &[Tensor],
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .dynamic_update_slice(input, update, starts)
                .map_err(super::Error::Ir)
        }

        pub fn gather(
            &self,
            input: Tensor,
            indices: Tensor,
            axis: usize,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .gather(input, indices, axis)
                .map_err(super::Error::Ir)
        }

        pub fn gather_slices(
            &self,
            input: Tensor,
            indices: Tensor,
            axis: usize,
            slice_size: i64,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .gather_slices(input, indices, axis, slice_size)
                .map_err(super::Error::Ir)
        }

        pub fn token_embedding(
            &self,
            weight: Tensor,
            indices: Tensor,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .token_embedding(weight, indices)
                .map_err(super::Error::Ir)
        }

        pub fn reduce_sum(&self, input: Tensor, axes: &[usize]) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .reduce_sum(input, axes)
                .map_err(super::Error::Ir)
        }

        pub fn reduce_max(&self, input: Tensor, axes: &[usize]) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .reduce_max(input, axes)
                .map_err(super::Error::Ir)
        }

        pub fn reduce_min(&self, input: Tensor, axes: &[usize]) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .reduce_min(input, axes)
                .map_err(super::Error::Ir)
        }

        pub fn mean(&self, input: Tensor, axes: &[usize]) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .mean(input, axes)
                .map_err(super::Error::Ir)
        }

        pub fn log_sum_exp(&self, input: Tensor, axes: &[usize]) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .log_sum_exp(input, axes)
                .map_err(super::Error::Ir)
        }

        pub fn argmax(&self, input: Tensor, axis: usize) -> Result<(Tensor, Tensor), super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .argmax(input, axis)
                .map_err(super::Error::Ir)
        }

        pub fn softmax(&self, input: Tensor, axis: usize) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .softmax(input, axis)
                .map_err(super::Error::Ir)
        }

        pub fn rms_norm(
            &self,
            input: Tensor,
            weight: Option<Tensor>,
            axis: usize,
            epsilon: f64,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .rms_norm(input, weight, axis, epsilon)
                .map_err(super::Error::Ir)
        }

        pub fn normalize_variance(
            &self,
            input: Tensor,
            axis: usize,
            epsilon: f64,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .normalize_variance(input, axis, epsilon)
                .map_err(super::Error::Ir)
        }

        pub fn layer_norm(
            &self,
            input: Tensor,
            weight: Option<Tensor>,
            bias: Option<Tensor>,
            axis: usize,
            epsilon: f64,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .layer_norm(input, weight, bias, axis, epsilon)
                .map_err(super::Error::Ir)
        }

        pub fn normalize_l2(
            &self,
            input: Tensor,
            axes: &[usize],
            epsilon: f64,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .normalize_l2(input, axes, epsilon)
                .map_err(super::Error::Ir)
        }

        pub fn rope(
            &self,
            input: Tensor,
            positions: Tensor,
            options: nml_ir::RopeOptions,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .rope(input, positions, options)
                .map_err(super::Error::Ir)
        }

        pub fn attention(
            &self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            query_positions: Tensor,
            key_positions: Tensor,
            options: nml_ir::AttentionOptions,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .attention(query, key, value, query_positions, key_positions, options)
                .map_err(super::Error::Ir)
        }

        pub fn paged_attention(
            &self,
            query: Tensor,
            key_cache: Tensor,
            value_cache: Tensor,
            page_table: Tensor,
            sequence_lengths: Tensor,
            query_positions: Tensor,
            options: nml_ir::AttentionOptions,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    page_table,
                    sequence_lengths,
                    query_positions,
                    options,
                )
                .map_err(super::Error::Ir)
        }

        pub fn with_partitions(
            &self,
            input: Tensor,
            partitions: &[Partition],
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .with_partitions(input, partitions)
                .map_err(super::Error::Ir)
        }

        pub fn exp(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .exp(input)
                .map_err(super::Error::Ir)
        }

        pub fn log(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .log(input)
                .map_err(super::Error::Ir)
        }

        pub fn sqrt(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .sqrt(input)
                .map_err(super::Error::Ir)
        }

        pub fn rsqrt(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .rsqrt(input)
                .map_err(super::Error::Ir)
        }

        pub fn tanh(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .tanh(input)
                .map_err(super::Error::Ir)
        }

        pub fn sin(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .sin(input)
                .map_err(super::Error::Ir)
        }

        pub fn cos(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .cos(input)
                .map_err(super::Error::Ir)
        }

        pub fn floor(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .floor(input)
                .map_err(super::Error::Ir)
        }

        pub fn ceil(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .ceil(input)
                .map_err(super::Error::Ir)
        }

        pub fn relu(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .relu(input)
                .map_err(super::Error::Ir)
        }

        pub fn sigmoid(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .sigmoid(input)
                .map_err(super::Error::Ir)
        }

        pub fn silu(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .silu(input)
                .map_err(super::Error::Ir)
        }

        pub fn gelu(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .gelu(input)
                .map_err(super::Error::Ir)
        }

        pub fn leaky_relu(&self, input: Tensor, slope: f64) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .leaky_relu(input, slope)
                .map_err(super::Error::Ir)
        }

        pub fn quick_gelu(&self, input: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .quick_gelu(input)
                .map_err(super::Error::Ir)
        }

        pub fn swiglu(&self, gate: Tensor, value: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .swiglu(gate, value)
                .map_err(super::Error::Ir)
        }

        pub fn geglu(&self, gate: Tensor, value: Tensor) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .geglu(gate, value)
                .map_err(super::Error::Ir)
        }

        /// Declares ZML-style activation donation for an executable output.
        pub fn reuse_buffer(
            &self,
            output: Tensor,
            activation: Tensor,
        ) -> Result<Tensor, super::Error> {
            self.inner
                .borrow_mut()
                .builder
                .reuse_buffer(output, activation)
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
            if !super::safetensors::storage_compatible(actual, expected) {
                return Err(super::Error::ShapeMismatch {
                    name: resolved,
                    expected,
                    actual,
                });
            }
            let tensor = if let Some(tensor) = store.tied_symbols.get(&resolved) {
                let declared = store.record_shapes[&resolved];
                if declared != expected {
                    return Err(super::Error::ShapeMismatch {
                        name: resolved,
                        expected,
                        actual: declared,
                    });
                }
                *tensor
            } else {
                let tensor = store.builder.parameter(resolved.clone(), expected);
                store.tied_symbols.insert(resolved.clone(), tensor);
                store.record_shapes.insert(resolved.clone(), expected);
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
            Ok(self.load_accounted(model, platform, options)?.0)
        }

        fn load_accounted<T: NmlStruct>(
            &self,
            model: &T,
            platform: &Platform,
            options: &LoadOptions,
        ) -> Result<(Bufferized<T>, LoadAccounting), super::Error> {
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

            let records = plan
                .values()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|record| {
                    let shape = store.record_shapes[&record];
                    (record, shape)
                })
                .collect::<Vec<_>>();
            let record_bytes = records
                .iter()
                .map(|(_, shape)| shape.byte_count().map_err(super::Error::Shape))
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
                peak_staging_bytes,
                ..LoadAccounting::default()
            };
            let mut loaded = BTreeMap::<String, Buffer>::new();
            if platform.name() == "cuda" {
                for (completed, (record, shape)) in records.iter().enumerate() {
                    let mut reader = store.registry.reader(record).map_err(super::Error::Io)?;
                    let buffer = platform
                        .upload_checkpoint_from(
                            *shape,
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
                    loaded.insert(record.clone(), buffer);
                    if let Some(progress) = &options.progress {
                        progress(completed + 1, records.len());
                    }
                }
            } else {
                let next = AtomicUsize::new(0);
                std::thread::scope(|scope| -> Result<(), super::Error> {
                    // A rendezvous channel bounds live host tensors to the worker
                    // count: no worker can read the next tensor until the main
                    // thread has accepted and started uploading its current one.
                    let (sender, receiver) = std::sync::mpsc::sync_channel(0);
                    for _ in 0..worker_count {
                        let sender = sender.clone();
                        let registry = store.registry.clone();
                        let records = &records;
                        let next = &next;
                        scope.spawn(move || {
                            loop {
                                let index = next.fetch_add(1, Ordering::Relaxed);
                                let Some((record, shape)) = records.get(index) else {
                                    break;
                                };
                                if sender
                                    .send((
                                        record.clone(),
                                        registry.read_with_shape(record, *shape),
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
                        let (record, slice) = receiver
                            .recv()
                            .map_err(|_| super::Error::LoaderWorkerFailed)?;
                        let slice = slice?;
                        accounting.reads += 1;
                        let buffer = platform
                            .upload(&slice, options.sharding.clone(), options.memory)
                            .map_err(super::Error::Runtime)?;
                        accounting.allocations += 1;
                        accounting.uploads += 1;
                        loaded.insert(record, buffer);
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
            let buffers = model.bufferize("", &mut |path, _tensor| {
                let record = plan
                    .get(path)
                    .expect("validated model traversal is deterministic");
                Ok::<Buffer, super::Error>(
                    loaded
                        .get(record)
                        .expect("every unique planned record was loaded")
                        .clone(),
                )
            })?;
            Ok((buffers, accounting))
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

        /// Reports `(completed_unique_tensors, total_unique_tensors)` after a
        /// buffer becomes persistent. The callback never observes private load
        /// plans, paths, or allocation identities.
        pub fn progress(mut self, callback: impl Fn(usize, usize) + Send + Sync + 'static) -> Self {
            self.progress = Some(Arc::new(callback));
            self
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use nml_types::{DType, Shape};
        use safetensors::tensor::{Dtype, View};
        use std::borrow::Cow;

        struct Tied {
            first: Tensor,
            second: Tensor,
        }

        struct TiedBuffers {
            first: Buffer,
            second: Buffer,
        }

        impl NmlStruct for Tied {
            type Buffers = TiedBuffers;

            fn visit_tensors(&self, prefix: &str, visitor: &mut dyn FnMut(&str, Tensor)) {
                visitor(&join(prefix, "first"), self.first);
                visitor(&join(prefix, "second"), self.second);
            }

            fn visit_buffers(
                buffers: &Self::Buffers,
                prefix: &str,
                visitor: &mut dyn FnMut(&str, &Buffer),
            ) {
                visitor(&join(prefix, "first"), &buffers.first);
                visitor(&join(prefix, "second"), &buffers.second);
            }

            fn bufferize<E>(
                &self,
                prefix: &str,
                resolve: &mut impl FnMut(&str, Tensor) -> Result<Buffer, E>,
            ) -> Result<Self::Buffers, E> {
                Ok(TiedBuffers {
                    first: resolve(&join(prefix, "first"), self.first)?,
                    second: resolve(&join(prefix, "second"), self.second)?,
                })
            }
        }

        struct Fixture([u8; 8]);

        impl View for &Fixture {
            fn dtype(&self) -> Dtype {
                Dtype::F16
            }

            fn shape(&self) -> &[usize] {
                &[2, 2]
            }

            fn data(&self) -> Cow<'_, [u8]> {
                Cow::Borrowed(&self.0)
            }

            fn data_len(&self) -> usize {
                self.0.len()
            }
        }

        #[test]
        fn unique_storage_plan_is_accounted_once_for_tied_fields() {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "nml-loader-accounting-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let fixture = Fixture([0, 0, 0, 60, 0, 64, 0, 66]);
            let tensors = BTreeMap::from([("shared", &fixture)]);
            std::fs::write(
                root.join("model.safetensors"),
                ::safetensors::serialize(tensors, None).unwrap(),
            )
            .unwrap();

            let registry = TensorRegistry::from_path(&root).unwrap();
            let store = TensorStore::new(registry);
            let shape = Shape::new(DType::F16, &[2, 2]).unwrap();
            let model = Tied {
                first: store.tensor("first", shape, &["shared"]).unwrap(),
                second: store.tensor("second", shape, &["shared"]).unwrap(),
            };
            let platform = Platform::cpu().unwrap();
            let options = LoadOptions::new(Sharding::replicated());
            let (buffers, accounting) = store.load_accounted(&model, &platform, &options).unwrap();
            assert_eq!(accounting.planned, 1);
            assert_eq!(accounting.reads, 1);
            assert_eq!(accounting.allocations, 1);
            assert_eq!(accounting.uploads, 1);
            assert_eq!(accounting.peak_staging_bytes, 8);
            assert!(!buffers.first.is_uniquely_owned());
            assert!(!buffers.second.is_uniquely_owned());
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
    Ir(nml_ir::Error),
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
    UnboundModelPath(String),
    OutstandingTensorStoreView,
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
