use nml_kernel_nvfp4::{embedding, gpt_oss_experts, grouped_projection, linear, Weight};
use nml_parameter::nvfp4::{global_scale, quantize_row};
use nml_types::{DType, Shape};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::Path;

const FIXTURE_RLOCATION: &str = "_main/artifacts/gpt-oss-20b-nvfp4/execution-fixture.json";

#[derive(Deserialize)]
struct ArtifactFixture {
    schema_version: u32,
    repository: String,
    revision: String,
    artifact_manifest_sha256: String,
    recipe: String,
    logical_name: String,
    logical_shape: [usize; 2],
    logical_dtype: String,
    global_scale_hex: String,
    input_formula: InputFormula,
    rows: Vec<FixtureRow>,
}

#[derive(Deserialize)]
struct InputFormula {
    multiplier: usize,
    modulus: usize,
    offset: isize,
    divisor: f32,
}

#[derive(Deserialize)]
struct FixtureRow {
    row: usize,
    payload_hex: String,
    block_scales_hex: String,
    decoded_f32_sha256: String,
    decoded_samples: Vec<DecodedSample>,
    projection_f64: f64,
}

#[derive(Deserialize)]
struct DecodedSample {
    column: usize,
    f32_bits: String,
}

struct Encoded {
    payload: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
}

fn encode(dimensions: &[usize], values: &[f32]) -> Encoded {
    let width = *dimensions.last().unwrap();
    assert_eq!(values.len(), dimensions.iter().product::<usize>());
    let global = global_scale(values).unwrap();
    let mut payload = Vec::new();
    let mut scales = Vec::new();
    for row in values.chunks_exact(width) {
        let encoded = quantize_row(row, global).unwrap();
        payload.extend_from_slice(encoded.payload());
        scales.extend_from_slice(encoded.block_scales());
    }
    Encoded {
        payload,
        scales,
        global,
    }
}

fn weight<'a>(dimensions: &[usize], encoded: &'a Encoded) -> Weight<'a> {
    let dimensions = dimensions
        .iter()
        .map(|&value| value as i64)
        .collect::<Vec<_>>();
    Weight::new(
        Shape::new(DType::Bf16, &dimensions).unwrap(),
        &encoded.payload,
        &encoded.scales,
        encoded.global,
    )
    .unwrap()
}

#[test]
fn immutable_artifact_rows_match_the_independent_execution_fixture() {
    let runfiles = std::env::var("RUNFILES_DIR").expect("Bazel must expose the runfiles tree");
    let fixture: ArtifactFixture = serde_json::from_slice(
        &std::fs::read(Path::new(&runfiles).join(FIXTURE_RLOCATION)).unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.repository, "narendra747/gpt-oss-20b-nvfp4");
    assert_eq!(fixture.revision, "729e9053f43c267636bfda3d6659c4141ff3ea1d");
    assert_eq!(
        fixture.artifact_manifest_sha256,
        "ab4c8cbd4424c8fec95bf683c0efd04c9cd350ec2a26737408b5500e61003207"
    );
    assert_eq!(fixture.recipe, "nml-nvfp4-weight-v1");
    assert_eq!(
        fixture.logical_name,
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(fixture.logical_shape, [4096, 2880]);
    assert_eq!(fixture.logical_dtype, "BF16");
    assert_eq!(
        fixture.rows.iter().map(|row| row.row).collect::<Vec<_>>(),
        [0, 1, 2047, 4095]
    );

    let width = fixture.logical_shape[1];
    let payload_width = width.div_ceil(2);
    let scale_width = width.div_ceil(16);
    let payload = fixture
        .rows
        .iter()
        .flat_map(|row| decode_hex(&row.payload_hex))
        .collect::<Vec<_>>();
    let scales = fixture
        .rows
        .iter()
        .flat_map(|row| decode_hex(&row.block_scales_hex))
        .collect::<Vec<_>>();
    assert_eq!(payload.len(), fixture.rows.len() * payload_width);
    assert_eq!(scales.len(), fixture.rows.len() * scale_width);
    let global_bytes: [u8; 4] = decode_hex(&fixture.global_scale_hex).try_into().unwrap();
    let compact = Weight::new(
        Shape::new(DType::Bf16, &[fixture.rows.len() as i64, width as i64]).unwrap(),
        &payload,
        &scales,
        f32::from_le_bytes(global_bytes),
    )
    .unwrap();

    let mut decoded = vec![0.0_f32; fixture.rows.len() * width];
    embedding(
        &compact,
        &(0..fixture.rows.len()).collect::<Vec<_>>(),
        &mut decoded,
    )
    .unwrap();
    for (fixture_row, actual) in fixture.rows.iter().zip(decoded.chunks_exact(width)) {
        let bytes = actual
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            fixture_row.decoded_f32_sha256,
            "decoded row {} differs from the immutable artifact fixture",
            fixture_row.row
        );
        for sample in &fixture_row.decoded_samples {
            assert_eq!(
                actual[sample.column].to_bits(),
                u32::from_str_radix(&sample.f32_bits, 16).unwrap(),
                "decoded row {}, column {} differs",
                fixture_row.row,
                sample.column
            );
        }
    }

    let formula = fixture.input_formula;
    let input = (0..width)
        .map(|index| {
            let value = (index * formula.multiplier) % formula.modulus;
            (value as isize - formula.offset) as f32 / formula.divisor
        })
        .collect::<Vec<_>>();
    let mut projected = vec![0.0_f32; fixture.rows.len()];
    linear(&input, &compact, None, &mut projected).unwrap();
    for ((actual, expected), row) in projected
        .into_iter()
        .zip(fixture.rows.iter().map(|row| row.projection_f64))
        .zip(fixture.rows.iter().map(|row| row.row))
    {
        assert!(
            (f64::from(actual) - expected).abs() < 2.0e-4,
            "artifact projection row {row}: actual={actual}, independent_f64={expected}"
        );
    }
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).unwrap();
            u8::from_str_radix(pair, 16).unwrap()
        })
        .collect()
}

fn oracle_value(encoded: &Encoded, width: usize, row: usize, column: usize) -> f64 {
    const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let packed_width = width.div_ceil(2);
    let byte = encoded.payload[row * packed_width + column / 2];
    let code = if column & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let magnitude = MAGNITUDES[usize::from(code & 0x07)];
    let value = if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let scale_bits = encoded.scales[row * width.div_ceil(16) + column / 16];
    let exponent = (scale_bits >> 3) & 0x0f;
    let fraction = scale_bits & 0x07;
    let scale = if exponent == 0 {
        f64::from(fraction) * 2.0f64.powi(-9)
    } else {
        (1.0 + f64::from(fraction) / 8.0) * 2.0f64.powi(i32::from(exponent) - 7)
    };
    value * scale * f64::from(encoded.global)
}

fn oracle_project(
    input: &[f32],
    expert: usize,
    inputs: usize,
    outputs: usize,
    encoded: &Encoded,
) -> Vec<f64> {
    (0..outputs)
        .map(|output| {
            (0..inputs).fold(0.0, |sum, input_index| {
                sum + f64::from(input[input_index])
                    * oracle_value(encoded, outputs, expert * inputs + input_index, output)
            })
        })
        .collect()
}

#[test]
fn embedding_decodes_only_selected_rows_including_odd_widths() {
    let values = (0..51)
        .map(|value| value as f32 / 8.0 - 3.0)
        .collect::<Vec<_>>();
    let encoded = encode(&[3, 17], &values);
    let weight = weight(&[3, 17], &encoded);
    let mut output = vec![0.0; 34];
    embedding(&weight, &[2, 0], &mut output).unwrap();

    for (output_row, source_row) in [2usize, 0].into_iter().enumerate() {
        for column in 0..17 {
            let actual = f64::from(output[output_row * 17 + column]);
            let expected = oracle_value(&encoded, 17, source_row, column);
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }
}

#[test]
fn linear_accumulates_compact_weights_without_dense_expansion() {
    let values = (0..68)
        .map(|value| value as f32 / 11.0 - 2.5)
        .collect::<Vec<_>>();
    let encoded = encode(&[4, 17], &values);
    let weight = weight(&[4, 17], &encoded);
    let input = (0..34)
        .map(|value| value as f32 / 13.0 - 1.0)
        .collect::<Vec<_>>();
    let bias = [0.5, -0.25, 1.0, -2.0];
    let mut output = vec![0.0; 8];
    linear(&input, &weight, Some(&bias), &mut output).unwrap();

    for row in 0..2 {
        for output_index in 0..4 {
            let expected = (0..17).fold(f64::from(bias[output_index]), |sum, input_index| {
                sum + f64::from(input[row * 17 + input_index])
                    * oracle_value(&encoded, 17, output_index, input_index)
            });
            assert!((f64::from(output[row * 4 + output_index]) - expected).abs() < 2.0e-5);
        }
    }
}

#[test]
fn grouped_projection_handles_uneven_and_empty_experts() {
    let values = (0..120)
        .map(|value| value as f32 / 17.0 - 3.0)
        .collect::<Vec<_>>();
    let encoded = encode(&[4, 3, 10], &values);
    let weight = weight(&[4, 3, 10], &encoded);
    let input = [1.0, 2.0, -1.0, -0.5, 1.5, 3.0, 2.0, 0.0, 0.25];
    let experts = [2, 2, 0];
    let bias = (0..40)
        .map(|value| value as f32 / 100.0)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; 30];
    grouped_projection(&input, &experts, &weight, Some(&bias), &mut output).unwrap();

    for assignment in 0..3 {
        for output_index in 0..10 {
            let expert = experts[assignment];
            let expected = (0..3).fold(
                f64::from(bias[expert * 10 + output_index]),
                |sum, input_index| {
                    let row = expert * 3 + input_index;
                    sum + f64::from(input[assignment * 3 + input_index])
                        * oracle_value(&encoded, 10, row, output_index)
                },
            );
            assert!((f64::from(output[assignment * 10 + output_index]) - expected).abs() < 2.0e-5);
        }
    }
}

#[test]
fn gpt_oss_experts_preserve_interleaved_clamped_residual_swiglu_semantics() {
    let gate_values = (0..48)
        .map(|value| value as f32 / 5.0 - 4.0)
        .collect::<Vec<_>>();
    let down_values = (0..24)
        .map(|value| value as f32 / 7.0 - 1.5)
        .collect::<Vec<_>>();
    let gate_encoded = encode(&[3, 2, 8], &gate_values);
    let down_encoded = encode(&[3, 4, 2], &down_values);
    let gate = weight(&[3, 2, 8], &gate_encoded);
    let down = weight(&[3, 4, 2], &down_encoded);
    let hidden = [1.0, -0.5, 0.25, 2.0];
    let router_indices = [0, 2, 2, 0];
    let routing_weights = [0.75, 0.25, 0.4, 0.6];
    let gate_bias = vec![0.0; 24];
    let down_bias = vec![0.0; 6];
    let mut output = vec![0.0; 4];
    gpt_oss_experts(
        &hidden,
        2,
        &router_indices,
        &routing_weights,
        &gate,
        &gate_bias,
        &down,
        &down_bias,
        &mut output,
    )
    .unwrap();

    let mut expected = [0.0f64; 4];
    for token in 0..2 {
        let token_hidden = &hidden[token * 2..token * 2 + 2];
        for route in 0..2 {
            let assignment = token * 2 + route;
            let expert = router_indices[assignment];
            let gate_up = oracle_project(token_hidden, expert, 2, 8, &gate_encoded);
            let activated = (0..4)
                .map(|index| {
                    let gate = gate_up[index * 2].min(7.0);
                    let up = gate_up[index * 2 + 1].clamp(-7.0, 7.0);
                    (up + 1.0) * gate * (1.0 / (1.0 + (-1.702 * gate).exp()))
                })
                .collect::<Vec<_>>();
            let down = (0..2)
                .map(|output| {
                    (0..4).fold(0.0, |sum, input_index| {
                        sum + activated[input_index]
                            * oracle_value(&down_encoded, 2, expert * 4 + input_index, output)
                    })
                })
                .collect::<Vec<_>>();
            for output_index in 0..2 {
                expected[token * 2 + output_index] +=
                    f64::from(routing_weights[assignment]) * down[output_index];
            }
        }
    }
    for (actual, expected) in output.into_iter().zip(expected) {
        assert!((f64::from(actual) - expected).abs() < 3.0e-5);
    }
}
