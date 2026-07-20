use nml_kernel_nvfp4::{
    Weight, embedding, grouped_projection, linear, routed_clamped_swiglu,
};
use nml_parameter::nvfp4::{global_scale, quantize_row};
use nml_types::{DType, Shape};
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

fn rowwise_weight<'a>(dimensions: &[usize], encoded: &'a Encoded) -> Weight<'a> {
    let dimensions = dimensions
        .iter()
        .map(|&value| value as i64)
        .collect::<Vec<_>>();
    Weight::rowwise(
        Shape::new(DType::Bf16, &dimensions).unwrap(),
        &encoded.payload,
        &encoded.scales,
        encoded.global,
    )
    .unwrap()
}

fn contraction_storage(dimensions: &[usize], encoded: &Encoded) -> Encoded {
    let inputs = *dimensions.last().unwrap();
    let outputs = dimensions[dimensions.len() - 2];
    let outer = dimensions[..dimensions.len() - 2].iter().product::<usize>();
    let packed_inputs = inputs.div_ceil(2);
    let scale_inputs = inputs.div_ceil(16);
    let mut payload = vec![0; encoded.payload.len()];
    let mut scales = vec![0; encoded.scales.len()];
    for batch in 0..outer {
        for output in 0..outputs {
            for input in 0..packed_inputs {
                payload[(batch * packed_inputs + input) * outputs + output] =
                    encoded.payload[(batch * outputs + output) * packed_inputs + input];
            }
            for input in 0..scale_inputs {
                scales[(batch * scale_inputs + input) * outputs + output] =
                    encoded.scales[(batch * outputs + output) * scale_inputs + input];
            }
        }
    }
    Encoded {
        payload,
        scales,
        global: encoded.global,
    }
}

fn contraction_weight<'a>(dimensions: &[usize], encoded: &'a Encoded) -> Weight<'a> {
    let dimensions = dimensions
        .iter()
        .map(|&value| value as i64)
        .collect::<Vec<_>>();
    Weight::contraction(
        Shape::new(DType::Bf16, &dimensions).unwrap(),
        &encoded.payload,
        &encoded.scales,
        encoded.global,
    )
    .unwrap()
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
                    * oracle_value(encoded, inputs, expert * outputs + output, input_index)
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
    let weight = rowwise_weight(&[3, 17], &encoded);
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
    let physical = contraction_storage(&[4, 17], &encoded);
    let weight = contraction_weight(&[4, 17], &physical);
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
    let encoded = encode(&[4, 10, 3], &values);
    let physical = contraction_storage(&[4, 10, 3], &encoded);
    let weight = contraction_weight(&[4, 10, 3], &physical);
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
                    let row = expert * 10 + output_index;
                    sum + f64::from(input[assignment * 3 + input_index])
                        * oracle_value(&encoded, 3, row, input_index)
                },
            );
            assert!((f64::from(output[assignment * 10 + output_index]) - expected).abs() < 2.0e-5);
        }
    }
}

#[test]
fn routed_experts_preserve_interleaved_clamped_residual_swiglu_semantics() {
    let gate_values = (0..48)
        .map(|value| value as f32 / 5.0 - 4.0)
        .collect::<Vec<_>>();
    let down_values = (0..24)
        .map(|value| value as f32 / 7.0 - 1.5)
        .collect::<Vec<_>>();
    let gate_encoded = encode(&[3, 8, 2], &gate_values);
    let down_encoded = encode(&[3, 2, 4], &down_values);
    let gate_physical = contraction_storage(&[3, 8, 2], &gate_encoded);
    let down_physical = contraction_storage(&[3, 2, 4], &down_encoded);
    let gate = contraction_weight(&[3, 8, 2], &gate_physical);
    let down = contraction_weight(&[3, 2, 4], &down_physical);
    let hidden = [1.0, -0.5, 0.25, 2.0];
    let router_indices = [0, 2, 2, 0];
    let routing_weights = [0.75, 0.25, 0.4, 0.6];
    let gate_bias = vec![0.0; 24];
    let down_bias = vec![0.0; 6];
    let mut output = vec![0.0; 4];
    routed_clamped_swiglu(
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
                            * oracle_value(&down_encoded, 4, expert * 2 + output, input_index)
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

#[test]
fn routed_experts_match_the_independent_oracle_across_bounded_shapes() {
    let mut state = 0x4e4d_4c4e_5646_5034u64;
    for &(tokens, hidden_size, experts, intermediate, top_k) in
        &[(1, 3, 4, 5, 2), (3, 5, 6, 7, 2), (4, 3, 7, 3, 3)]
    {
        let gate_values = generated_values(
            &mut state,
            experts * hidden_size * intermediate * 2,
            2.5,
        );
        let down_values = generated_values(
            &mut state,
            experts * intermediate * hidden_size,
            2.0,
        );
        let gate_encoded = encode(
            &[experts, intermediate * 2, hidden_size],
            &gate_values,
        );
        let down_encoded = encode(
            &[experts, hidden_size, intermediate],
            &down_values,
        );
        let gate_physical = contraction_storage(
            &[experts, intermediate * 2, hidden_size],
            &gate_encoded,
        );
        let down_physical =
            contraction_storage(&[experts, hidden_size, intermediate], &down_encoded);
        let gate = contraction_weight(
            &[experts, intermediate * 2, hidden_size],
            &gate_physical,
        );
        let down = contraction_weight(
            &[experts, hidden_size, intermediate],
            &down_physical,
        );
        let hidden = generated_values(&mut state, tokens * hidden_size, 1.25);
        let gate_bias = generated_values(&mut state, experts * intermediate * 2, 0.2);
        let down_bias = generated_values(&mut state, experts * hidden_size, 0.2);
        let mut router_indices = Vec::with_capacity(tokens * top_k);
        let mut routing_weights = Vec::with_capacity(tokens * top_k);
        for token in 0..tokens {
            let raw = (0..top_k)
                .map(|route| 0.25 + generated_unit(&mut state) + route as f32 * 0.1)
                .collect::<Vec<_>>();
            let normalization = raw.iter().sum::<f32>();
            for (route, value) in raw.into_iter().enumerate() {
                // Leave the final expert empty while deliberately repeating
                // others across tokens. This covers sparse, uneven routing.
                router_indices.push((token * (route + 1) + route * 2) % (experts - 1));
                routing_weights.push(value / normalization);
            }
        }

        let mut output = vec![0.0; tokens * hidden_size];
        routed_clamped_swiglu(
            &hidden,
            tokens,
            &router_indices,
            &routing_weights,
            &gate,
            &gate_bias,
            &down,
            &down_bias,
            &mut output,
        )
        .unwrap();

        let mut expected = vec![0.0f64; tokens * hidden_size];
        for token in 0..tokens {
            let token_hidden = &hidden[token * hidden_size..(token + 1) * hidden_size];
            for route in 0..top_k {
                let assignment = token * top_k + route;
                let expert = router_indices[assignment];
                let mut gate_up = oracle_project(
                    token_hidden,
                    expert,
                    hidden_size,
                    intermediate * 2,
                    &gate_encoded,
                );
                for (column, value) in gate_up.iter_mut().enumerate() {
                    *value += f64::from(gate_bias[expert * intermediate * 2 + column]);
                }
                let activated = (0..intermediate)
                    .map(|index| {
                        let gate = gate_up[index * 2].min(7.0);
                        let up = gate_up[index * 2 + 1].clamp(-7.0, 7.0);
                        (up + 1.0) * gate * (1.0 / (1.0 + (-1.702 * gate).exp()))
                    })
                    .collect::<Vec<_>>();
                let projected = (0..hidden_size)
                    .map(|output_index| {
                        (0..intermediate).fold(
                            f64::from(down_bias[expert * hidden_size + output_index]),
                            |sum, input_index| {
                                sum + activated[input_index]
                                    * oracle_value(
                                        &down_encoded,
                                        intermediate,
                                        expert * hidden_size + output_index,
                                        input_index,
                                    )
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                for output_index in 0..hidden_size {
                    expected[token * hidden_size + output_index] +=
                        f64::from(routing_weights[assignment]) * projected[output_index];
                }
            }
        }

        for (index, (actual, expected)) in output.into_iter().zip(expected).enumerate() {
            let tolerance = 3.0e-4 * expected.abs().max(1.0);
            assert!(
                (f64::from(actual) - expected).abs() <= tolerance,
                "shape tokens={tokens} hidden={hidden_size} experts={experts} intermediate={intermediate} top_k={top_k}, output {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }
}

fn generated_values(state: &mut u64, count: usize, magnitude: f32) -> Vec<f32> {
    (0..count)
        .map(|_| (generated_unit(state) * 2.0 - 1.0) * magnitude)
        .collect()
}

fn generated_unit(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32
}
