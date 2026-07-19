//! Deterministic launch geometry for the retained CUDA paged-attention path.
//!
//! Launch selection is deliberately separated from TTIR emission.  It is a
//! pure function of validated tensor geometry and the CUDA core count, which
//! makes specialization reproducible and keeps device queries out of kernels.

use std::error::Error as StdError;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionGeometry {
    pub core_count: usize,
    pub all_decode: bool,
    pub num_tokens: usize,
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub batch_size: usize,
    pub page_size: usize,
    pub max_query_length: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionLaunch {
    TwoDimensional {
        block_m: usize,
        block_q: usize,
        tile_size: usize,
        total_query_blocks: usize,
        grid: [usize; 3],
        warps: usize,
        stages: usize,
    },
    SplitK {
        block_m: usize,
        block_q: usize,
        tile_size: usize,
        total_query_blocks: usize,
        segments: usize,
        attention_grid: [usize; 3],
        attention_warps: usize,
        attention_stages: usize,
        reduction_grid: [usize; 3],
        reduction_warps: usize,
        reduction_stages: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchError {
    Zero(&'static str),
    InvalidHeadRatio { query_heads: usize, kv_heads: usize },
    Overflow(&'static str),
}

impl fmt::Display for LaunchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero(field) => write!(formatter, "paged-attention {field} must be nonzero"),
            Self::InvalidHeadRatio {
                query_heads,
                kv_heads,
            } => write!(
                formatter,
                "paged-attention query heads {query_heads} are not divisible by KV heads {kv_heads}"
            ),
            Self::Overflow(field) => {
                write!(formatter, "paged-attention {field} overflows host geometry")
            }
        }
    }
}

impl StdError for LaunchError {}

pub fn select_attention_launch(
    geometry: AttentionGeometry,
) -> Result<AttentionLaunch, LaunchError> {
    validate(geometry)?;
    let queries_per_kv = geometry.num_query_heads / geometry.num_kv_heads;
    let padded_queries_per_kv = queries_per_kv
        .checked_next_power_of_two()
        .ok_or(LaunchError::Overflow("query-head tile"))?;
    // `tt.make_range` requires a power-of-two extent, but NML's cache contract
    // permits any positive physical page size. Keep page addressing exact and
    // pad only the decode tile; tail lanes are masked by the kernel.
    let padded_page_size = geometry
        .page_size
        .checked_next_power_of_two()
        .ok_or(LaunchError::Overflow("decode tile"))?;
    // Page geometry describes cache allocation, not the amount of K/V state a
    // single program must retain. The addressing kernel already permits a tile
    // to cross page boundaries lane by lane. Cap the decode tile so a product
    // choosing coarse pages cannot accidentally manufacture a spill-heavy
    // attention specialization.
    let decode_tile_size = padded_page_size.min(64);
    let block_m = 16usize.max(padded_queries_per_kv);
    let block_q = block_m / padded_queries_per_kv;
    let total_query_blocks = geometry
        .num_tokens
        .checked_div(block_q)
        .and_then(|blocks| blocks.checked_add(geometry.batch_size))
        .ok_or(LaunchError::Overflow("query-block count"))?;

    // The pinned CUDA policy uses the whole-sequence kernel for prefill.  For
    // decode it switches back only when the batch supplies more than 128
    // programs across KV heads; smaller batches need split-K parallelism.
    let use_two_dimensional =
        !geometry.all_decode || geometry.batch_size > 128usize / geometry.num_kv_heads;
    if use_two_dimensional {
        return select_two_dimensional(
            geometry,
            padded_queries_per_kv,
            decode_tile_size,
            block_m,
            block_q,
            total_query_blocks,
        );
    }
    select_split_k(
        geometry,
        decode_tile_size,
        block_m,
        block_q,
        total_query_blocks,
    )
}

fn select_two_dimensional(
    geometry: AttentionGeometry,
    padded_queries_per_kv: usize,
    decode_tile_size: usize,
    mut block_m: usize,
    mut block_q: usize,
    mut total_query_blocks: usize,
) -> Result<AttentionLaunch, LaunchError> {
    let maximum_stages = if geometry.head_dim <= 128 { 4 } else { 2 };
    let (mut stages, mut warps, mut tile_size) = if geometry.all_decode {
        (3, 2, decode_tile_size)
    } else {
        (1, 2, 64)
    };
    if geometry.max_query_length >= 256 {
        let preferred_block_m = if geometry.head_dim >= 256 {
            tile_size = 16;
            64
        } else {
            128
        };
        block_m = preferred_block_m.max(padded_queries_per_kv);
        stages = 1;
        warps = 4;
        block_q = block_m / padded_queries_per_kv;
        total_query_blocks = geometry
            .num_tokens
            .checked_div(block_q)
            .and_then(|blocks| blocks.checked_add(geometry.batch_size))
            .ok_or(LaunchError::Overflow("query-block count"))?;
    }
    Ok(AttentionLaunch::TwoDimensional {
        block_m,
        block_q,
        tile_size,
        total_query_blocks,
        grid: [geometry.num_kv_heads, total_query_blocks, 1],
        warps,
        stages: stages.min(maximum_stages),
    })
}

fn select_split_k(
    geometry: AttentionGeometry,
    decode_tile_size: usize,
    block_m: usize,
    block_q: usize,
    total_query_blocks: usize,
) -> Result<AttentionLaunch, LaunchError> {
    let target_programs = geometry
        .core_count
        .checked_mul(4)
        .ok_or(LaunchError::Overflow("target program count"))?;
    let two_dimensional_programs = total_query_blocks
        .checked_mul(geometry.num_kv_heads)
        .ok_or(LaunchError::Overflow("2D program count"))?;
    let segments = target_programs
        .div_ceil(two_dimensional_programs)
        .checked_next_power_of_two()
        .ok_or(LaunchError::Overflow("segment count"))?
        .min(128)
        .min(16)
        .max(if decode_tile_size <= 16 { 16 } else { 8 });
    let reduction_warps = if segments == if decode_tile_size <= 16 { 16 } else { 8 } {
        1
    } else {
        2
    };

    Ok(AttentionLaunch::SplitK {
        block_m,
        block_q,
        tile_size: decode_tile_size,
        total_query_blocks,
        segments,
        attention_grid: [total_query_blocks, geometry.num_kv_heads, segments],
        attention_warps: 2,
        attention_stages: 1,
        reduction_grid: [geometry.num_tokens, geometry.num_query_heads, 1],
        reduction_warps,
        reduction_stages: 1,
    })
}

fn validate(geometry: AttentionGeometry) -> Result<(), LaunchError> {
    for (name, value) in [
        ("CUDA core count", geometry.core_count),
        ("token count", geometry.num_tokens),
        ("query-head count", geometry.num_query_heads),
        ("KV-head count", geometry.num_kv_heads),
        ("head dimension", geometry.head_dim),
        ("batch size", geometry.batch_size),
        ("page size", geometry.page_size),
        ("maximum query length", geometry.max_query_length),
    ] {
        if value == 0 {
            return Err(LaunchError::Zero(name));
        }
    }
    if geometry.num_query_heads % geometry.num_kv_heads != 0 {
        return Err(LaunchError::InvalidHeadRatio {
            query_heads: geometry.num_query_heads,
            kv_heads: geometry.num_kv_heads,
        });
    }
    Ok(())
}
