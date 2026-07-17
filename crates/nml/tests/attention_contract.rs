//! Numerical product contract shared by CPU and CUDA attention execution.

use nml_ir::{AttentionOptions, ProgramBuilder, RopeLayout, RopeOptions, RopeScaling};
use nml_types::{BFloat16, DType, F16, Shape};
use safetensors::tensor::{Dtype as SafeDType, View};
use std::borrow::Cow;
use std::collections::BTreeMap;

#[derive(nml::ParameterTree)]
struct AttentionProjectors {
    query: nml::Parameter,
    key: nml::Parameter,
    value: nml::Parameter,
}

struct TensorData {
    dtype: SafeDType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl View for &TensorData {
    fn dtype(&self) -> SafeDType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }

    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

const BATCH: usize = 1;
const QUERY_LEN: usize = 2;
const QUERY_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const KEY_LEN: usize = 3;
const PAGE_SIZE: usize = 2;
const PHYSICAL_PAGES: usize = 3;
const LOGICAL_PAGES: usize = 2;

#[test]
fn ordinary_and_portable_paged_attention_execute_the_same_semantics() {
    let platform = platform();
    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        execute_variant(&platform, dtype, true, None);
        execute_variant(&platform, dtype, false, Some(2));
    }
    execute_head_mapping(&platform, 2, 2); // MHA
    execute_head_mapping(&platform, 4, 2); // GQA
    rotary_embeddings_execute_both_layouts(&platform);
    fully_masked_ordinary_attention_returns_zero(&platform);
    empty_paged_context_returns_zero(&platform);
    cache_update_rollback_and_replay_preserve_persistent_storage(&platform);
    checkpoint_backed_attention_block_executes(&platform);
    if env!("NML_ATTENTION_BACKEND") == "cuda" {
        accelerated_cuda_attention_matches_dense_reference(&platform);
    }
}

fn fully_masked_ordinary_attention_returns_zero(platform: &nml::Platform) {
    let program = ordinary_program(DType::F32, AttentionOptions::default());
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let mut args = executable.args();
    set_float(
        platform,
        &mut args,
        "query",
        Shape::new(DType::F32, &[1, 2, 2, 4]).unwrap(),
        &[1.0; 16],
    );
    set_float(
        platform,
        &mut args,
        "key",
        Shape::new(DType::F32, &[1, 3, 1, 4]).unwrap(),
        &[2.0; 12],
    );
    set_float(
        platform,
        &mut args,
        "value",
        Shape::new(DType::F32, &[1, 3, 1, 4]).unwrap(),
        &[3.0; 12],
    );
    set_i32(platform, &mut args, "query_positions", &[1, 2], &[-2, -1]);
    set_i32(platform, &mut args, "key_positions", &[1, 3], &[0, 1, 2]);
    let output = decode(
        args.call()
            .unwrap()
            .get("output")
            .unwrap()
            .to_slice()
            .unwrap(),
    );
    assert!(output.iter().all(|value| *value == 0.0), "{output:?}");
}

/// Exercises the exact public API shapes that select FA2/FA3 and both Triton
/// launch families on compatible devices. The same contract deliberately runs
/// through the portable CUDA fallback on the repository's SM75 host; moving
/// the unchanged binary to SM8x or SM90 changes only the private lowering.
fn accelerated_cuda_attention_matches_dense_reference(platform: &nml::Platform) {
    execute_accelerated_dense(
        platform,
        DType::F16,
        AttentionOptions {
            causal: true,
            sliding_window: Some(4),
            scale: None,
        },
    );
    execute_accelerated_dense(
        platform,
        DType::Bf16,
        AttentionOptions {
            causal: false,
            sliding_window: None,
            scale: Some(0.17),
        },
    );

    for dtype in [DType::F16, DType::Bf16] {
        // Six query heads over two KV heads deliberately covers a non-power-of-
        // two GQA ratio. Page 16 selects Triton on SM8x and upstream FA3 on
        // SM90; prefill and decode select the 2D and split-K Triton families.
        execute_accelerated_paged(
            platform,
            dtype,
            16,
            3,
            AttentionOptions {
                causal: true,
                sliding_window: Some(4),
                scale: None,
            },
        );
        execute_accelerated_paged(
            platform,
            dtype,
            16,
            1,
            AttentionOptions {
                causal: false,
                sliding_window: Some(8),
                scale: Some(0.13),
            },
        );

        // Original-upstream FA2's paged path requires pages divisible by 256.
        // This pair therefore executes FA2 on SM8x and FA3 on SM90 without a
        // second cache representation or a test-only backend selector.
        execute_accelerated_paged(
            platform,
            dtype,
            256,
            3,
            AttentionOptions {
                causal: true,
                sliding_window: None,
                scale: None,
            },
        );
        execute_accelerated_paged(
            platform,
            dtype,
            256,
            1,
            AttentionOptions {
                causal: false,
                sliding_window: None,
                scale: None,
            },
        );
    }

    // F32 is outside upstream FlashAttention's retained ABI and consequently
    // keeps both Triton launch families observable on SM90 as well as SM8x.
    for query_length in [3, 1] {
        execute_accelerated_paged(
            platform,
            DType::F32,
            16,
            query_length,
            AttentionOptions {
                causal: true,
                sliding_window: Some(6),
                scale: None,
            },
        );
    }
}

fn execute_accelerated_dense(platform: &nml::Platform, dtype: DType, options: AttentionOptions) {
    const BATCHES: usize = 2;
    const QUERY_LENGTH: usize = 3;
    const KEY_LENGTH: usize = 7;
    const QUERY_HEADS: usize = 6;
    const KV_HEADS: usize = 2;
    const HEAD_DIMENSION: usize = 64;

    let query = generated_values(BATCHES * QUERY_LENGTH * QUERY_HEADS * HEAD_DIMENSION, 3);
    let key = generated_values(BATCHES * KEY_LENGTH * KV_HEADS * HEAD_DIMENSION, 11);
    let value = generated_values(BATCHES * KEY_LENGTH * KV_HEADS * HEAD_DIMENSION, 23);
    let query = round_values(dtype, &query);
    let key = round_values(dtype, &key);
    let value = round_values(dtype, &value);
    let query_positions = (0..BATCHES)
        .flat_map(|_| (KEY_LENGTH - QUERY_LENGTH..KEY_LENGTH).map(|value| value as i32))
        .collect::<Vec<_>>();
    let key_positions = (0..BATCHES)
        .flat_map(|_| (0..KEY_LENGTH).map(|value| value as i32))
        .collect::<Vec<_>>();
    let lengths = vec![KEY_LENGTH; BATCHES];
    let expected = reference_attention_geometry(
        &query,
        &key,
        &value,
        &query_positions,
        &lengths,
        BATCHES,
        QUERY_LENGTH,
        KEY_LENGTH,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIMENSION,
        options,
    );

    let mut builder = ProgramBuilder::new();
    let query_shape = Shape::new(
        dtype,
        &[
            BATCHES as i64,
            QUERY_LENGTH as i64,
            QUERY_HEADS as i64,
            HEAD_DIMENSION as i64,
        ],
    )
    .unwrap();
    let key_shape = Shape::new(
        dtype,
        &[
            BATCHES as i64,
            KEY_LENGTH as i64,
            KV_HEADS as i64,
            HEAD_DIMENSION as i64,
        ],
    )
    .unwrap();
    let query_tensor = builder.input("accelerated_query", query_shape);
    let key_tensor = builder.input("accelerated_key", key_shape);
    let value_tensor = builder.input("accelerated_value", key_shape);
    let query_position_tensor = builder.input(
        "accelerated_query_positions",
        Shape::new(DType::I32, &[BATCHES as i64, QUERY_LENGTH as i64]).unwrap(),
    );
    let key_position_tensor = builder.input(
        "accelerated_key_positions",
        Shape::new(DType::I32, &[BATCHES as i64, KEY_LENGTH as i64]).unwrap(),
    );
    let output = builder
        .attention(
            query_tensor,
            key_tensor,
            value_tensor,
            query_position_tensor,
            key_position_tensor,
            options,
        )
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let mut args = executable.args();
    set_float(
        platform,
        &mut args,
        "accelerated_query",
        query_shape,
        &query,
    );
    set_float(platform, &mut args, "accelerated_key", key_shape, &key);
    set_float(platform, &mut args, "accelerated_value", key_shape, &value);
    set_i32(
        platform,
        &mut args,
        "accelerated_query_positions",
        &[BATCHES as i64, QUERY_LENGTH as i64],
        &query_positions,
    );
    set_i32(
        platform,
        &mut args,
        "accelerated_key_positions",
        &[BATCHES as i64, KEY_LENGTH as i64],
        &key_positions,
    );
    for _ in 0..2 {
        let actual = decode(
            args.call()
                .unwrap()
                .get("output")
                .unwrap()
                .to_slice()
                .unwrap(),
        );
        assert_close(&actual, &expected, accelerated_tolerance(dtype));
    }
}

fn execute_accelerated_paged(
    platform: &nml::Platform,
    dtype: DType,
    page_size: usize,
    query_length: usize,
    options: AttentionOptions,
) {
    const BATCHES: usize = 2;
    const QUERY_HEADS: usize = 6;
    const KV_HEADS: usize = 2;
    const HEAD_DIMENSION: usize = 64;
    const PHYSICAL_PAGES: usize = 4;
    const LOGICAL_PAGES: usize = 2;

    let lengths = [page_size + 3, page_size + 1];
    let logical_capacity = page_size * LOGICAL_PAGES;
    let query = round_values(
        dtype,
        &generated_values(
            BATCHES * query_length * QUERY_HEADS * HEAD_DIMENSION,
            page_size + query_length,
        ),
    );
    // The two sequences intentionally share their first physical page. Their
    // logical KV values are therefore equal by token and differ only through
    // the batch-specific query, which makes sharing independently checkable.
    let logical_key = round_values(
        dtype,
        &generated_values(logical_capacity * KV_HEADS * HEAD_DIMENSION, 31),
    );
    let logical_value = round_values(
        dtype,
        &generated_values(logical_capacity * KV_HEADS * HEAD_DIMENSION, 47),
    );
    let mut dense_key = Vec::with_capacity(BATCHES * logical_key.len());
    let mut dense_value = Vec::with_capacity(BATCHES * logical_value.len());
    for _ in 0..BATCHES {
        dense_key.extend_from_slice(&logical_key);
        dense_value.extend_from_slice(&logical_value);
    }
    let query_positions = lengths
        .iter()
        .flat_map(|length| (*length - query_length..*length).map(|position| position as i32))
        .collect::<Vec<_>>();
    let expected = reference_attention_geometry(
        &query,
        &dense_key,
        &dense_value,
        &query_positions,
        &lengths,
        BATCHES,
        query_length,
        logical_capacity,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIMENSION,
        options,
    );

    let token_width = KV_HEADS * HEAD_DIMENSION;
    let mut key_cache = vec![91.0; PHYSICAL_PAGES * page_size * token_width];
    let mut value_cache = vec![-73.0; PHYSICAL_PAGES * page_size * token_width];
    let copy_page = |source: &[f32], target: &mut [f32], logical: usize, physical: usize| {
        let width = page_size * token_width;
        target[physical * width..(physical + 1) * width]
            .copy_from_slice(&source[logical * width..(logical + 1) * width]);
    };
    copy_page(&logical_key, &mut key_cache, 0, 2);
    copy_page(&logical_value, &mut value_cache, 0, 2);
    copy_page(&logical_key, &mut key_cache, 1, 0);
    copy_page(&logical_value, &mut value_cache, 1, 0);
    copy_page(&logical_key, &mut key_cache, 1, 3);
    copy_page(&logical_value, &mut value_cache, 1, 3);
    let page_table = [2, 0, 2, 3];

    let mut builder = ProgramBuilder::new();
    let query_shape = Shape::new(
        dtype,
        &[
            BATCHES as i64,
            query_length as i64,
            QUERY_HEADS as i64,
            HEAD_DIMENSION as i64,
        ],
    )
    .unwrap();
    let cache_shape = Shape::new(
        dtype,
        &[
            PHYSICAL_PAGES as i64,
            page_size as i64,
            KV_HEADS as i64,
            HEAD_DIMENSION as i64,
        ],
    )
    .unwrap();
    let query_tensor = builder.input("accelerated_query", query_shape);
    let key_tensor = builder.input("accelerated_key_cache", cache_shape);
    let value_tensor = builder.input("accelerated_value_cache", cache_shape);
    let table_tensor = builder.input(
        "accelerated_page_table",
        Shape::new(DType::I32, &[BATCHES as i64, LOGICAL_PAGES as i64]).unwrap(),
    );
    let length_tensor = builder.input(
        "accelerated_sequence_lengths",
        Shape::new(DType::I32, &[BATCHES as i64]).unwrap(),
    );
    let position_tensor = builder.input(
        "accelerated_query_positions",
        Shape::new(DType::I32, &[BATCHES as i64, query_length as i64]).unwrap(),
    );
    let output = builder
        .paged_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            table_tensor,
            length_tensor,
            position_tensor,
            options,
        )
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let mut args = executable.args();
    set_float(
        platform,
        &mut args,
        "accelerated_query",
        query_shape,
        &query,
    );
    set_float(
        platform,
        &mut args,
        "accelerated_key_cache",
        cache_shape,
        &key_cache,
    );
    set_float(
        platform,
        &mut args,
        "accelerated_value_cache",
        cache_shape,
        &value_cache,
    );
    set_i32(
        platform,
        &mut args,
        "accelerated_page_table",
        &[BATCHES as i64, LOGICAL_PAGES as i64],
        &page_table,
    );
    set_i32(
        platform,
        &mut args,
        "accelerated_sequence_lengths",
        &[BATCHES as i64],
        &lengths.map(|length| length as i32),
    );
    set_i32(
        platform,
        &mut args,
        "accelerated_query_positions",
        &[BATCHES as i64, query_length as i64],
        &query_positions,
    );
    for _ in 0..2 {
        let actual = decode(
            args.call()
                .unwrap()
                .get("output")
                .unwrap()
                .to_slice()
                .unwrap(),
        );
        assert_close(&actual, &expected, accelerated_tolerance(dtype));
    }
}

#[allow(clippy::too_many_arguments)]
fn reference_attention_geometry(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_positions: &[i32],
    key_lengths: &[usize],
    batches: usize,
    query_length: usize,
    key_capacity: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    options: AttentionOptions,
) -> Vec<f32> {
    let mut output = vec![0.0; batches * query_length * query_heads * head_dimension];
    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (head_dimension as f64).sqrt()) as f32;
    let queries_per_kv = query_heads / key_value_heads;
    for batch in 0..batches {
        for query_index in 0..query_length {
            let query_position = query_positions[batch * query_length + query_index];
            for query_head in 0..query_heads {
                let key_value_head = query_head / queries_per_kv;
                let mut scores = Vec::with_capacity(key_lengths[batch]);
                for key_index in 0..key_lengths[batch] {
                    let key_position = key_index as i32;
                    let distance = (key_position - query_position).unsigned_abs();
                    let valid = (!options.causal || key_position <= query_position)
                        && options
                            .sliding_window
                            .is_none_or(|window| distance < window as u32);
                    if !valid {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let query_offset = ((batch * query_length + query_index) * query_heads
                        + query_head)
                        * head_dimension;
                    let key_offset = ((batch * key_capacity + key_index) * key_value_heads
                        + key_value_head)
                        * head_dimension;
                    let score = (0..head_dimension)
                        .map(|dimension| {
                            query[query_offset + dimension] * key[key_offset + dimension]
                        })
                        .sum::<f32>()
                        * scale;
                    scores.push(score);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if maximum == f32::NEG_INFINITY {
                    continue;
                }
                let weights = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                let output_offset = ((batch * query_length + query_index) * query_heads
                    + query_head)
                    * head_dimension;
                for dimension in 0..head_dimension {
                    output[output_offset + dimension] = weights
                        .iter()
                        .enumerate()
                        .map(|(key_index, weight)| {
                            let value_offset = ((batch * key_capacity + key_index)
                                * key_value_heads
                                + key_value_head)
                                * head_dimension;
                            weight * value[value_offset + dimension]
                        })
                        .sum::<f32>()
                        / denominator;
                }
            }
        }
    }
    output
}

fn generated_values(length: usize, phase: usize) -> Vec<f32> {
    (0..length)
        .map(|index| (((index * 17 + phase) % 61) as f32 - 30.0) / 31.0)
        .collect()
}

fn accelerated_tolerance(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => 1e-2,
        DType::F16 => 8e-3,
        DType::Bf16 => 4e-2,
        _ => unreachable!(),
    }
}

fn rotary_embeddings_execute_both_layouts(platform: &nml::Platform) {
    let mut builder = ProgramBuilder::new();
    let shape = Shape::new(DType::F32, &[1, 2, 1, 4]).unwrap();
    let input = builder.input("input", shape);
    let positions = builder.input("positions", Shape::new(DType::I32, &[1, 2]).unwrap());
    let sequential = builder
        .rope(
            input,
            positions,
            RopeOptions {
                base: 10_000.0,
                rotary_dimensions: 4,
                layout: RopeLayout::Sequential,
                scaling: RopeScaling::Default,
            },
        )
        .unwrap();
    let interleaved = builder
        .rope(
            input,
            positions,
            RopeOptions {
                base: 10_000.0,
                rotary_dimensions: 4,
                layout: RopeLayout::Interleaved,
                scaling: RopeScaling::Default,
            },
        )
        .unwrap();
    let program = builder
        .finish_named(&[
            ("sequential".to_owned(), sequential),
            ("interleaved".to_owned(), interleaved),
        ])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let values = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut args = executable.args();
    set_float(platform, &mut args, "input", shape, &values);
    set_i32(platform, &mut args, "positions", &[1, 2], &[0, 1]);
    let results = args.call().unwrap();
    let sequential = decode(results.get("sequential").unwrap().to_slice().unwrap());
    let interleaved = decode(results.get("interleaved").unwrap().to_slice().unwrap());
    let (sin_one, cos_one) = 1.0f32.sin_cos();
    let (sin_small, cos_small) = 0.01f32.sin_cos();
    let expected_sequential = vec![
        0.0,
        1.0,
        2.0,
        3.0,
        4.0 * cos_one - 6.0 * sin_one,
        5.0 * cos_small - 7.0 * sin_small,
        6.0 * cos_one + 4.0 * sin_one,
        7.0 * cos_small + 5.0 * sin_small,
    ];
    let expected_interleaved = vec![
        0.0,
        1.0,
        2.0,
        3.0,
        4.0 * cos_one - 5.0 * sin_one,
        5.0 * cos_one + 4.0 * sin_one,
        6.0 * cos_small - 7.0 * sin_small,
        7.0 * cos_small + 6.0 * sin_small,
    ];
    assert_close(&sequential, &expected_sequential, 2e-5);
    assert_close(&interleaved, &expected_interleaved, 2e-5);
}

fn checkpoint_backed_attention_block_executes(platform: &nml::Platform) {
    for dtype in [DType::F16, DType::Bf16] {
        let root = temporary_directory(dtype);
        std::fs::create_dir_all(&root).unwrap();
        let query_weight = (0..8 * 8)
            .map(|index| if index / 8 == index % 8 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let key_weight = (0..4 * 8)
            .map(|index| if index % 8 == index / 8 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let value_weight = (0..4 * 8)
            .map(|index| if index % 8 == index / 8 + 4 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let query_data = tensor_data(dtype, &[8, 8], &query_weight);
        let key_data = tensor_data(dtype, &[4, 8], &key_weight);
        let value_data = tensor_data(dtype, &[4, 8], &value_weight);
        let tensors = BTreeMap::from([
            ("query", &query_data),
            ("key", &key_data),
            ("value", &value_data),
        ]);
        std::fs::write(
            root.join("model.safetensors"),
            safetensors::serialize(tensors, None).unwrap(),
        )
        .unwrap();

        let registry = nml::safetensors::TensorRegistry::from_path(&root).unwrap();
        let parameters = nml::io::ParameterSet::new(registry);
        let projectors = AttentionProjectors {
            query: parameters
                .dense("query", Shape::new(dtype, &[8, 8]).unwrap(), &[])
                .unwrap(),
            key: parameters
                .dense("key", Shape::new(dtype, &[4, 8]).unwrap(), &[])
                .unwrap(),
            value: parameters
                .dense("value", Shape::new(dtype, &[4, 8]).unwrap(), &[])
                .unwrap(),
        };
        let mut graph = nml::Graph::new();
        let input = graph.input("input", Shape::new(dtype, &[2, 8]).unwrap());
        let positions = graph.input("positions", Shape::new(DType::I32, &[1, 2]).unwrap());
        let query = graph.linear(input, &projectors.query, None).unwrap();
        let key = graph.linear(input, &projectors.key, None).unwrap();
        let value = graph.linear(input, &projectors.value, None).unwrap();
        let query = graph
            .reshape(query, Shape::new(dtype, &[1, 2, 2, 4]).unwrap())
            .unwrap();
        let key = graph
            .reshape(key, Shape::new(dtype, &[1, 2, 1, 4]).unwrap())
            .unwrap();
        let value = graph
            .reshape(value, Shape::new(dtype, &[1, 2, 1, 4]).unwrap())
            .unwrap();
        let output = graph
            .attention(
                query,
                key,
                value,
                positions,
                positions,
                AttentionOptions::default(),
            )
            .unwrap();
        let loaded = parameters
            .load(
                &projectors,
                platform,
                &nml::io::LoadOptions::new(nml::Sharding::single()),
            )
            .unwrap();
        let program = graph
            .finish_named(&[("output".to_owned(), output)])
            .unwrap();
        let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
        let input_values = (0..16)
            .map(|index| (index as f32 - 5.0) / 6.0)
            .collect::<Vec<_>>();
        let mut args = executable.args();
        args.set_parameter(&loaded.query).unwrap();
        args.set_parameter(&loaded.key).unwrap();
        args.set_parameter(&loaded.value).unwrap();
        args.bake().unwrap();
        set_float(
            platform,
            &mut args,
            "input",
            Shape::new(dtype, &[2, 8]).unwrap(),
            &input_values,
        );
        set_i32(platform, &mut args, "positions", &[1, 2], &[0, 1]);
        let output = decode(
            args.call()
                .unwrap()
                .get("output")
                .unwrap()
                .to_slice()
                .unwrap(),
        );
        assert!(output.iter().all(|value| value.is_finite()));
        std::fs::remove_dir_all(root).unwrap();
    }
}

fn empty_paged_context_returns_zero(platform: &nml::Platform) {
    let program = paged_program(DType::F32, AttentionOptions::default());
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let mut args = executable.args();
    set_float(
        platform,
        &mut args,
        "query",
        Shape::new(DType::F32, &[1, 2, 2, 4]).unwrap(),
        &[1.0; 16],
    );
    set_float(
        platform,
        &mut args,
        "key_cache",
        Shape::new(DType::F32, &[3, 2, 1, 4]).unwrap(),
        &[2.0; 24],
    );
    set_float(
        platform,
        &mut args,
        "value_cache",
        Shape::new(DType::F32, &[3, 2, 1, 4]).unwrap(),
        &[3.0; 24],
    );
    set_i32(platform, &mut args, "page_table", &[1, 2], &[-1, -1]);
    set_i32(platform, &mut args, "sequence_lengths", &[1], &[0]);
    set_i32(platform, &mut args, "query_positions", &[1, 2], &[0, 1]);
    let output = decode(
        args.call()
            .unwrap()
            .get("output")
            .unwrap()
            .to_slice()
            .unwrap(),
    );
    assert!(output.iter().all(|value| *value == 0.0), "{output:?}");
}

fn execute_head_mapping(platform: &nml::Platform, query_heads: usize, kv_heads: usize) {
    let query = (0..query_heads * HEAD_DIM)
        .map(|index| (index as f32 - 3.0) / 5.0)
        .collect::<Vec<_>>();
    let key = (0..2 * kv_heads * HEAD_DIM)
        .map(|index| (index as f32 - 4.0) / 7.0)
        .collect::<Vec<_>>();
    let value = (0..2 * kv_heads * HEAD_DIM)
        .map(|index| (6.0 - index as f32) / 8.0)
        .collect::<Vec<_>>();
    let mut builder = ProgramBuilder::new();
    let query_tensor = builder.input(
        "query",
        Shape::new(DType::F32, &[1, 1, query_heads as i64, 4]).unwrap(),
    );
    let key_tensor = builder.input(
        "key",
        Shape::new(DType::F32, &[1, 2, kv_heads as i64, 4]).unwrap(),
    );
    let value_tensor = builder.input(
        "value",
        Shape::new(DType::F32, &[1, 2, kv_heads as i64, 4]).unwrap(),
    );
    let query_positions =
        builder.input("query_positions", Shape::new(DType::I32, &[1, 1]).unwrap());
    let key_positions = builder.input("key_positions", Shape::new(DType::I32, &[1, 2]).unwrap());
    let output = builder
        .attention(
            query_tensor,
            key_tensor,
            value_tensor,
            query_positions,
            key_positions,
            AttentionOptions::default(),
        )
        .unwrap();
    let program = builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap();
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let mut args = executable.args();
    set_float(
        platform,
        &mut args,
        "query",
        Shape::new(DType::F32, &[1, 1, query_heads as i64, 4]).unwrap(),
        &query,
    );
    set_float(
        platform,
        &mut args,
        "key",
        Shape::new(DType::F32, &[1, 2, kv_heads as i64, 4]).unwrap(),
        &key,
    );
    set_float(
        platform,
        &mut args,
        "value",
        Shape::new(DType::F32, &[1, 2, kv_heads as i64, 4]).unwrap(),
        &value,
    );
    set_i32(platform, &mut args, "query_positions", &[1, 1], &[1]);
    set_i32(platform, &mut args, "key_positions", &[1, 2], &[0, 1]);
    let actual = decode(
        args.call()
            .unwrap()
            .get("output")
            .unwrap()
            .to_slice()
            .unwrap(),
    );

    let mut expected = vec![0.0f32; query_heads * HEAD_DIM];
    let groups = query_heads / kv_heads;
    for head in 0..query_heads {
        let kv_head = head / groups;
        let scores = (0..2)
            .map(|token| {
                (0..HEAD_DIM)
                    .map(|dimension| {
                        query[head * HEAD_DIM + dimension]
                            * key[(token * kv_heads + kv_head) * HEAD_DIM + dimension]
                    })
                    .sum::<f32>()
                    / (HEAD_DIM as f32).sqrt()
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        for dimension in 0..HEAD_DIM {
            expected[head * HEAD_DIM + dimension] = (0..2)
                .map(|token| {
                    weights[token] * value[(token * kv_heads + kv_head) * HEAD_DIM + dimension]
                })
                .sum::<f32>()
                / denominator;
        }
    }
    assert_close(&actual, &expected, 2e-5);
}

fn cache_update_rollback_and_replay_preserve_persistent_storage(platform: &nml::Platform) {
    let spec = nml::attention::CacheSpec::paged(DType::F32, 1, 3, 2, 2, 1, 4).unwrap();
    let mut cache = nml::attention::Cache::allocate(
        platform,
        spec,
        nml::Sharding::single(),
        nml::Memory::Default,
    )
    .unwrap();
    cache.assign_page(platform, 0, 0, 2).unwrap();
    cache.assign_page(platform, 0, 1, 0).unwrap();
    assert!(cache.assign_page(platform, 0, 2, 0).is_err());
    assert!(cache.truncate(platform, 0, 5).is_err());
    cache.truncate(platform, 0, spec.capacity()).unwrap();
    cache.assign_page(platform, 0, 1, 2).unwrap();
    cache.truncate(platform, 0, 3).unwrap();
    cache.assign_page(platform, 0, 1, 0).unwrap();
    cache.truncate(platform, 0, 3).unwrap();

    let mut builder = ProgramBuilder::new();
    let cache_shape = spec.key_value_shape().unwrap();
    let key_cache = builder.input("key_cache", cache_shape);
    let value_cache = builder.input("value_cache", cache_shape);
    let update_shape = Shape::new(DType::F32, &[1, 1, 1, 4]).unwrap();
    let key_update = builder.input("key_update", update_shape);
    let value_update = builder.input("value_update", update_shape);
    let page = builder.input("page", Shape::new(DType::I32, &[]).unwrap());
    let offset = builder.input("offset", Shape::new(DType::I32, &[]).unwrap());
    let zero = builder.scalar(0i32).unwrap();
    let key = builder
        .dynamic_update_slice(key_cache, key_update, &[page, offset, zero, zero])
        .unwrap();
    let value = builder
        .dynamic_update_slice(value_cache, value_update, &[page, offset, zero, zero])
        .unwrap();
    let key = builder.reuse_buffer(key, key_cache).unwrap();
    let value = builder.reuse_buffer(value, value_cache).unwrap();
    let program = builder
        .finish_named(&[("key".to_owned(), key), ("value".to_owned(), value)])
        .unwrap();
    assert_eq!(
        program.output_aliases().collect::<Vec<_>>(),
        vec![Some(0), Some(1)]
    );
    let executable = platform.compile(&program, nml::Sharding::single()).unwrap();
    let (key, value) = cache.take_storage().unwrap();
    let mut args = executable.args();
    args.set("key_cache", key).unwrap();
    args.set("value_cache", value).unwrap();
    set_float(
        platform,
        &mut args,
        "key_update",
        update_shape,
        &[1.0, 2.0, 3.0, 4.0],
    );
    set_float(
        platform,
        &mut args,
        "value_update",
        update_shape,
        &[-1.0, -2.0, -3.0, -4.0],
    );
    set_i32(platform, &mut args, "page", &[], &[2]);
    set_i32(platform, &mut args, "offset", &[], &[0]);
    let mut outputs = args.call().unwrap().into_buffers().into_iter();
    cache
        .replace_storage(outputs.next().unwrap(), outputs.next().unwrap())
        .unwrap();
    assert!(outputs.next().is_none());

    let (key, value) = cache.take_storage().unwrap();
    args.set("key_cache", key).unwrap();
    args.set("value_cache", value).unwrap();
    set_float(
        platform,
        &mut args,
        "key_update",
        update_shape,
        &[5.0, 6.0, 7.0, 8.0],
    );
    set_float(
        platform,
        &mut args,
        "value_update",
        update_shape,
        &[-5.0, -6.0, -7.0, -8.0],
    );
    set_i32(platform, &mut args, "page", &[], &[2]);
    set_i32(platform, &mut args, "offset", &[], &[1]);
    let mut outputs = args.call().unwrap().into_buffers().into_iter();
    cache
        .replace_storage(outputs.next().unwrap(), outputs.next().unwrap())
        .unwrap();
    assert!(outputs.next().is_none());

    let key = decode(cache.key().unwrap().to_slice().unwrap());
    let value = decode(cache.value().unwrap().to_slice().unwrap());
    assert_eq!(&key[16..20], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&value[16..20], &[-1.0, -2.0, -3.0, -4.0]);
    assert_eq!(&key[20..24], &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(&value[20..24], &[-5.0, -6.0, -7.0, -8.0]);
    assert!(key[..16].iter().all(|value| *value == 0.0));
    assert!(value[..16].iter().all(|value| *value == 0.0));

    cache.truncate(platform, 0, 1).unwrap();
    cache.truncate(platform, 0, 3).unwrap();
    assert_eq!(decode(cache.key().unwrap().to_slice().unwrap()), key);
    assert_eq!(decode(cache.value().unwrap().to_slice().unwrap()), value);

    cache.truncate(platform, 0, 1).unwrap();
    let mut builder = ProgramBuilder::new();
    let query_shape = Shape::new(DType::F32, &[1, 1, 2, 4]).unwrap();
    let query = builder.input("query", query_shape);
    let key_cache = builder.input("key_cache", cache_shape);
    let value_cache = builder.input("value_cache", cache_shape);
    let page_table = builder.input("page_table", spec.page_table_shape().unwrap().unwrap());
    let lengths = builder.input("sequence_lengths", spec.lengths_shape().unwrap());
    // Keep this lifecycle graph inside the optimized CUDA index ABI. It still
    // uses the portable fallback on SM75; the unchanged binary selects Triton
    // when the deferred hardware contract is eventually run on SM80/SM90.
    let positions = builder.input("query_positions", Shape::new(DType::I32, &[1, 1]).unwrap());
    let output = builder
        .paged_attention(
            query,
            key_cache,
            value_cache,
            page_table,
            lengths,
            positions,
            AttentionOptions::default(),
        )
        .unwrap();
    let executable = platform
        .compile(
            &builder
                .finish_named(&[("output".to_owned(), output)])
                .unwrap(),
            nml::Sharding::single(),
        )
        .unwrap();
    let mut args = executable.args();
    args.set("key_cache", cache.key().unwrap().clone()).unwrap();
    args.set("value_cache", cache.value().unwrap().clone())
        .unwrap();
    args.set("page_table", cache.page_table().unwrap().clone())
        .unwrap();
    args.set("sequence_lengths", cache.lengths().clone())
        .unwrap();
    set_float(platform, &mut args, "query", query_shape, &[1.0; 8]);
    set_i32(platform, &mut args, "query_positions", &[1, 1], &[0]);
    for _ in 0..2 {
        let output = decode(
            args.call()
                .unwrap()
                .get("output")
                .unwrap()
                .to_slice()
                .unwrap(),
        );
        assert_eq!(output, vec![-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0, -4.0]);
    }
}

fn execute_variant(
    platform: &nml::Platform,
    dtype: DType,
    causal: bool,
    sliding_window: Option<usize>,
) {
    let query = (0..BATCH * QUERY_LEN * QUERY_HEADS * HEAD_DIM)
        .map(|index| (index as f32 - 6.0) / 7.0)
        .collect::<Vec<_>>();
    let key = (0..BATCH * KEY_LEN * KV_HEADS * HEAD_DIM)
        .map(|index| (index as f32 - 4.0) / 6.0)
        .collect::<Vec<_>>();
    let value = (0..BATCH * KEY_LEN * KV_HEADS * HEAD_DIM)
        .map(|index| (7.0 - index as f32) / 5.0)
        .collect::<Vec<_>>();
    let query = round_values(dtype, &query);
    let key = round_values(dtype, &key);
    let value = round_values(dtype, &value);
    let query_positions = [1i32, 2];
    let key_positions = [0i32, 1, 2];
    let options = AttentionOptions {
        causal,
        sliding_window,
        scale: None,
    };
    let expected = reference_attention(
        &query,
        &key,
        &value,
        &query_positions,
        &key_positions,
        options,
    );

    let ordinary = ordinary_program(dtype, options);
    let ordinary = platform
        .compile(&ordinary, nml::Sharding::single())
        .unwrap();
    let mut args = ordinary.args();
    set_float(
        platform,
        &mut args,
        "query",
        Shape::new(dtype, &[1, 2, 2, 4]).unwrap(),
        &query,
    );
    set_float(
        platform,
        &mut args,
        "key",
        Shape::new(dtype, &[1, 3, 1, 4]).unwrap(),
        &key,
    );
    set_float(
        platform,
        &mut args,
        "value",
        Shape::new(dtype, &[1, 3, 1, 4]).unwrap(),
        &value,
    );
    set_i32(
        platform,
        &mut args,
        "query_positions",
        &[1, 2],
        &query_positions,
    );
    set_i32(
        platform,
        &mut args,
        "key_positions",
        &[1, 3],
        &key_positions,
    );
    let ordinary = args.call().unwrap();
    let ordinary = decode(ordinary.get("output").unwrap().to_slice().unwrap());

    // Logical page 0 lives in physical page 2 and logical page 1 in page 0.
    // Physical page 1 is unrelated storage and must never affect the result.
    let mut paged_key = vec![91.0f32; PHYSICAL_PAGES * PAGE_SIZE * KV_HEADS * HEAD_DIM];
    let mut paged_value = vec![-73.0f32; PHYSICAL_PAGES * PAGE_SIZE * KV_HEADS * HEAD_DIM];
    copy_token_range(&key, 0, 2, &mut paged_key, 2);
    copy_token_range(&value, 0, 2, &mut paged_value, 2);
    copy_token_range(&key, 2, 1, &mut paged_key, 0);
    copy_token_range(&value, 2, 1, &mut paged_value, 0);
    let paged = paged_program(dtype, options);
    let paged = platform.compile(&paged, nml::Sharding::single()).unwrap();
    let mut args = paged.args();
    set_float(
        platform,
        &mut args,
        "query",
        Shape::new(dtype, &[1, 2, 2, 4]).unwrap(),
        &query,
    );
    set_float(
        platform,
        &mut args,
        "key_cache",
        Shape::new(dtype, &[3, 2, 1, 4]).unwrap(),
        &paged_key,
    );
    set_float(
        platform,
        &mut args,
        "value_cache",
        Shape::new(dtype, &[3, 2, 1, 4]).unwrap(),
        &paged_value,
    );
    set_i32(
        platform,
        &mut args,
        "page_table",
        &[BATCH as i64, LOGICAL_PAGES as i64],
        &[2, 0],
    );
    set_i32(platform, &mut args, "sequence_lengths", &[1], &[3]);
    set_i32(
        platform,
        &mut args,
        "query_positions",
        &[1, 2],
        &query_positions,
    );
    let paged = args.call().unwrap();
    let paged = decode(paged.get("output").unwrap().to_slice().unwrap());

    let tolerance = match dtype {
        DType::F32 => 2e-5,
        DType::F16 => 3e-3,
        DType::Bf16 => 2e-2,
        _ => unreachable!(),
    };
    let case = format!("dtype={dtype:?}, causal={causal}, sliding_window={sliding_window:?}");
    assert_close_in(&ordinary, &expected, tolerance, &format!("ordinary {case}"));
    assert_close_in(&paged, &expected, tolerance, &format!("paged {case}"));
    assert_close_in(&paged, &ordinary, tolerance, &format!("cross-path {case}"));
}

fn ordinary_program(dtype: DType, options: AttentionOptions) -> nml_ir::Program {
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", Shape::new(dtype, &[1, 2, 2, 4]).unwrap());
    let key = builder.input("key", Shape::new(dtype, &[1, 3, 1, 4]).unwrap());
    let value = builder.input("value", Shape::new(dtype, &[1, 3, 1, 4]).unwrap());
    let query_positions =
        builder.input("query_positions", Shape::new(DType::I32, &[1, 2]).unwrap());
    let key_positions = builder.input("key_positions", Shape::new(DType::I32, &[1, 3]).unwrap());
    let output = builder
        .attention(query, key, value, query_positions, key_positions, options)
        .unwrap();
    builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap()
}

fn paged_program(dtype: DType, options: AttentionOptions) -> nml_ir::Program {
    let mut builder = ProgramBuilder::new();
    let query = builder.input("query", Shape::new(dtype, &[1, 2, 2, 4]).unwrap());
    let key = builder.input("key_cache", Shape::new(dtype, &[3, 2, 1, 4]).unwrap());
    let value = builder.input("value_cache", Shape::new(dtype, &[3, 2, 1, 4]).unwrap());
    let table = builder.input("page_table", Shape::new(DType::I32, &[1, 2]).unwrap());
    let lengths = builder.input("sequence_lengths", Shape::new(DType::I32, &[1]).unwrap());
    let positions = builder.input("query_positions", Shape::new(DType::I32, &[1, 2]).unwrap());
    let output = builder
        .paged_attention(query, key, value, table, lengths, positions, options)
        .unwrap();
    builder
        .finish_named(&[("output".to_owned(), output)])
        .unwrap()
}

fn set_float(
    platform: &nml::Platform,
    args: &mut nml::exe::Arguments<'_>,
    name: &str,
    shape: Shape,
    values: &[f32],
) {
    let buffer = match shape.dtype() {
        DType::F32 => upload_typed(platform, shape, values),
        DType::F16 => {
            let values = values
                .iter()
                .map(|value| F16::from_f32(*value))
                .collect::<Vec<_>>();
            upload_typed(platform, shape, &values)
        }
        DType::Bf16 => {
            let values = values
                .iter()
                .map(|value| BFloat16::from_f32(*value))
                .collect::<Vec<_>>();
            upload_typed(platform, shape, &values)
        }
        _ => unreachable!(),
    };
    args.set(name, buffer).unwrap();
}

fn upload_typed<T: nml_tensor::Element>(
    platform: &nml::Platform,
    shape: Shape,
    values: &[T],
) -> nml::Buffer {
    let slice = nml::Slice::from_typed(shape, values).unwrap();
    platform
        .upload(&slice, nml::Sharding::single(), nml::Memory::Default)
        .unwrap()
}

fn set_i32(
    platform: &nml::Platform,
    args: &mut nml::exe::Arguments<'_>,
    name: &str,
    dimensions: &[i64],
    values: &[i32],
) {
    let shape = Shape::new(DType::I32, dimensions).unwrap();
    let slice = nml::Slice::from_typed(shape, values).unwrap();
    let buffer = platform
        .upload(&slice, nml::Sharding::single(), nml::Memory::Default)
        .unwrap();
    args.set(name, buffer).unwrap();
}

fn decode(slice: nml::Slice<'_>) -> Vec<f32> {
    let bytes = slice.contiguous_bytes().unwrap();
    match slice.dtype() {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| F16::from_bits(u16::from_ne_bytes(chunk.try_into().unwrap())).to_f32())
            .collect(),
        DType::Bf16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                BFloat16::from_bits(u16::from_ne_bytes(chunk.try_into().unwrap())).to_f32()
            })
            .collect(),
        _ => unreachable!(),
    }
}

fn round_values(dtype: DType, values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| match dtype {
            DType::F32 => *value,
            DType::F16 => F16::from_f32(*value).to_f32(),
            DType::Bf16 => BFloat16::from_f32(*value).to_f32(),
            _ => unreachable!(),
        })
        .collect()
}

fn copy_token_range(source: &[f32], start: usize, count: usize, target: &mut [f32], page: usize) {
    let token_width = KV_HEADS * HEAD_DIM;
    let source = &source[start * token_width..(start + count) * token_width];
    let destination = page * PAGE_SIZE * token_width;
    target[destination..destination + source.len()].copy_from_slice(source);
}

fn reference_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    query_positions: &[i32],
    key_positions: &[i32],
    options: AttentionOptions,
) -> Vec<f32> {
    let mut output = vec![0.0; BATCH * QUERY_LEN * QUERY_HEADS * HEAD_DIM];
    let scale = options.scale.unwrap_or(1.0 / (HEAD_DIM as f64).sqrt()) as f32;
    for query_index in 0..QUERY_LEN {
        for head in 0..QUERY_HEADS {
            let mut scores = vec![f32::NEG_INFINITY; KEY_LEN];
            for key_index in 0..KEY_LEN {
                let distance =
                    (key_positions[key_index] - query_positions[query_index]).unsigned_abs();
                let valid = (!options.causal
                    || key_positions[key_index] <= query_positions[query_index])
                    && options
                        .sliding_window
                        .is_none_or(|window| distance < window as u32);
                if valid {
                    scores[key_index] = (0..HEAD_DIM)
                        .map(|dimension| {
                            query[(query_index * QUERY_HEADS + head) * HEAD_DIM + dimension]
                                * key[key_index * HEAD_DIM + dimension]
                        })
                        .sum::<f32>()
                        * scale;
                }
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            for dimension in 0..HEAD_DIM {
                output[(query_index * QUERY_HEADS + head) * HEAD_DIM + dimension] = weights
                    .iter()
                    .enumerate()
                    .map(|(key_index, weight)| weight * value[key_index * HEAD_DIM + dimension])
                    .sum::<f32>()
                    / denominator;
            }
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_close_in(actual, expected, tolerance, "attention result");
}

fn assert_close_in(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}, value {index}: expected {expected}, received {actual}, tolerance {tolerance}"
        );
    }
}

fn tensor_data(dtype: DType, shape: &[usize], values: &[f32]) -> TensorData {
    let bytes = values
        .iter()
        .flat_map(|value| match dtype {
            DType::F16 => F16::from_f32(*value).to_bits().to_le_bytes(),
            DType::Bf16 => BFloat16::from_f32(*value).to_bits().to_le_bytes(),
            _ => unreachable!(),
        })
        .collect();
    TensorData {
        dtype: match dtype {
            DType::F16 => SafeDType::F16,
            DType::Bf16 => SafeDType::BF16,
            _ => unreachable!(),
        },
        shape: shape.to_vec(),
        bytes,
    }
}

fn temporary_directory(dtype: DType) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "nml-attention-{dtype:?}-{}-{nonce}",
        std::process::id()
    ))
}

fn platform() -> nml::Platform {
    match env!("NML_ATTENTION_BACKEND") {
        "cpu" => nml::Platform::cpu_with_devices(1).unwrap(),
        "cuda" => {
            let runfiles = std::env::var("RUNFILES_DIR").unwrap();
            let relative = std::env::var("NML_CUDA_RUNTIME_RLOCATION").unwrap();
            // SAFETY: Bazel test processes have not created application
            // threads before platform initialization.
            unsafe {
                std::env::set_var(
                    "NML_CUDA_RUNTIME",
                    std::path::Path::new(&runfiles).join(relative),
                );
                nml::Platform::cuda().unwrap()
            }
        }
        backend => panic!("unknown attention backend {backend}"),
    }
}
