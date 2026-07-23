//! Compact generic K/V append for a physical paged cache.
//!
//! One launch handles both K and V for every row in a static serving family.
//! Runtime row/query masks remove padding without changing the compiled
//! geometry, and output aliasing updates the process-wide cache in place.

use super::{ArgumentKind, Builder, Comparison, DType, Error, Kernel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedCacheAppendConfig {
    pub dtype: DType,
    pub batch: i64,
    pub query: i64,
    pub physical_pages: i64,
    pub page_size: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub logical_pages: i64,
    pub block_elements: i64,
}

impl PagedCacheAppendConfig {
    fn validate(self) -> Result<Self, Error> {
        let values = [
            self.batch,
            self.query,
            self.physical_pages,
            self.page_size,
            self.heads,
            self.head_dim,
            self.logical_pages,
            self.block_elements,
        ];
        if !matches!(self.dtype, DType::F16 | DType::Bf16 | DType::F32)
            || values
                .into_iter()
                .any(|value| value <= 0 || i32::try_from(value).is_err())
            || !(self.block_elements as u64).is_power_of_two()
            || self
                .heads
                .checked_mul(self.head_dim)
                .is_none_or(|elements| elements <= 0)
        {
            return Err(Error::InvalidKernelSpec(
                "invalid compact paged-cache append specialization",
            ));
        }
        Ok(self)
    }
}

pub fn build_paged_cache_append(config: PagedCacheAppendConfig) -> Result<Kernel, Error> {
    let config = config.validate()?;
    let mut builder = Builder::new("paged_cache_append")?;
    let _key_cache = pointer(&mut builder, "key_cache", config.dtype)?;
    let _value_cache = pointer(&mut builder, "value_cache", config.dtype)?;
    let key_updates = pointer(&mut builder, "key_updates", config.dtype)?;
    let value_updates = pointer(&mut builder, "value_updates", config.dtype)?;
    let block_tables = pointer(&mut builder, "block_tables", DType::I32)?;
    let start_positions = pointer(&mut builder, "start_positions", DType::I32)?;
    let query_lengths = pointer(&mut builder, "query_lengths", DType::I32)?;
    // StableHLO Bool is byte-addressed at the XLA custom-call boundary. Read
    // it as I8 and turn the stored 0/1 byte into a register predicate.
    let active_rows = pointer(&mut builder, "active_rows", DType::I8)?;
    let write_mask = pointer(&mut builder, "write_mask", DType::I8)?;
    let key_output = pointer(&mut builder, "key_output", config.dtype)?;
    let value_output = pointer(&mut builder, "value_output", config.dtype)?;

    let row = builder.program_id(0)?;
    let element_program = builder.program_id(1)?;
    let query = builder.integer(config.query, DType::I32)?;
    let batch_row = builder.divide(&row, &query)?;
    let query_offset = builder.remainder(&row, &query)?;

    let active_address = builder.add_pointer(&active_rows, &batch_row)?;
    let active = builder.load(&active_address)?;
    let zero_i8 = builder.integer(0, DType::I8)?;
    let active = builder.compare(Comparison::Greater, &active, &zero_i8)?;
    let length_address = builder.add_pointer(&query_lengths, &batch_row)?;
    let query_length = builder.load(&length_address)?;
    let within_query = builder.compare(Comparison::Less, &query_offset, &query_length)?;
    let write_address = builder.add_pointer(&write_mask, &row)?;
    let write = builder.load(&write_address)?;
    let write = builder.compare(Comparison::Greater, &write, &zero_i8)?;
    let enabled = builder.bit_and(&active, &within_query)?;
    let enabled = builder.bit_and(&enabled, &write)?;

    builder.if_only(&enabled, |body| {
        let start_address = body.add_pointer(&start_positions, &batch_row)?;
        let start = body.load(&start_address)?;
        let position = body.add(&start, &query_offset)?;
        let page_size = body.integer(config.page_size, DType::I32)?;
        let logical_page = body.divide(&position, &page_size)?;
        let page_offset = body.remainder(&position, &page_size)?;

        let logical_pages = body.integer(config.logical_pages, DType::I32)?;
        let table_row = body.multiply(&batch_row, &logical_pages)?;
        let table_offset = body.add(&table_row, &logical_page)?;
        let page_address = body.add_pointer(&block_tables, &table_offset)?;
        let physical_page = body.load(&page_address)?;

        let head_elements = config
            .heads
            .checked_mul(config.head_dim)
            .ok_or(Error::InvalidKernelSpec(
                "paged-cache head element count overflows",
            ))?;
        let block_elements = body.integer(config.block_elements, DType::I32)?;
        let element_start = body.multiply(&element_program, &block_elements)?;
        let lanes = body.range(
            0,
            i32::try_from(config.block_elements).map_err(|_| {
                Error::InvalidKernelSpec("paged-cache element block exceeds I32")
            })?,
        )?;
        let elements = body.add(&element_start, &lanes)?;
        let element_limit = body.integer(head_elements, DType::I32)?;
        let valid_elements = body.compare(Comparison::Less, &elements, &element_limit)?;

        let page_size_i64 = body.integer(config.page_size, DType::I64)?;
        let head_elements_i64 = body.integer(head_elements, DType::I64)?;
        let physical_page_i64 = body.cast(&physical_page, DType::I64)?;
        let page_offset_i64 = body.cast(&page_offset, DType::I64)?;
        let cache_page = body.multiply(&physical_page_i64, &page_size_i64)?;
        let cache_row = body.add(&cache_page, &page_offset_i64)?;
        let cache_base = body.multiply(&cache_row, &head_elements_i64)?;

        let row_i64 = body.cast(&row, DType::I64)?;
        let update_base = body.multiply(&row_i64, &head_elements_i64)?;
        let elements_i64 = body.cast(&elements, DType::I64)?;
        let cache_offsets = body.add(&cache_base, &elements_i64)?;
        let update_offsets = body.add(&update_base, &elements_i64)?;

        let key_update_addresses = body.add_pointer(&key_updates, &update_offsets)?;
        let value_update_addresses = body.add_pointer(&value_updates, &update_offsets)?;
        let zero = body.full_float(&[config.block_elements], 0.0, config.dtype)?;
        let key_values =
            body.load_masked(&key_update_addresses, &valid_elements, &zero)?;
        let value_values =
            body.load_masked(&value_update_addresses, &valid_elements, &zero)?;
        let key_addresses = body.add_pointer(&key_output, &cache_offsets)?;
        let value_addresses = body.add_pointer(&value_output, &cache_offsets)?;
        body.store_masked(&key_addresses, &key_values, &valid_elements)?;
        body.store_masked(&value_addresses, &value_values, &valid_elements)
    })?;
    builder.return_void()?;
    builder.finish()
}

fn pointer(builder: &mut Builder, name: &str, dtype: DType) -> Result<super::Value, Error> {
    builder.argument(
        name,
        ArgumentKind::Pointer {
            element: dtype,
            address_space: 1,
        },
        Some(16),
    )
}
