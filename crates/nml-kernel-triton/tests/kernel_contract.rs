use nml_kernel_triton::{
    build_grouped_projection, build_paged_attention_2d, build_paged_attention_3d,
    build_segment_reduction, select_attention_launch, ArgumentKind, AttentionGeometry,
    AttentionLaunch, Builder, Comparison, DType, Error, GatedActivation, GroupedProjectionConfig,
    KernelLaunch, KernelSpec, OutputAlias, PagedAttention2dConfig, PagedAttention3dConfig,
    Reduction, SegmentReductionConfig, TensorSpec,
};
use nml_mlir::{Block, Context, Region};

#[test]
fn named_typed_kernel_is_deterministic_and_verified() {
    fn emit() -> String {
        let mut builder = Builder::new("add_one").unwrap();
        let input = builder
            .argument(
                "input",
                ArgumentKind::Pointer {
                    element: DType::F32,
                    address_space: 1,
                },
                Some(16),
            )
            .unwrap();
        let output = builder
            .argument(
                "output",
                ArgumentKind::Pointer {
                    element: DType::F32,
                    address_space: 1,
                },
                Some(16),
            )
            .unwrap();
        let program = builder.program_id(0).unwrap();
        let input_at_program = builder.add_pointer(&input, &program).unwrap();
        let output_at_program = builder.add_pointer(&output, &program).unwrap();
        let value = builder.load(&input_at_program).unwrap();
        let one = builder.float(1.0, DType::F32).unwrap();
        let sum = builder.add(&value, &one).unwrap();
        builder.store(&output_at_program, &sum).unwrap();
        builder.return_void().unwrap();
        builder.finish().unwrap()
    }

    let first = emit();
    assert_eq!(first, emit());
    assert!(first.contains("tt.func public @add_one"), "{first}");
    assert!(first.contains("tt.get_program_id x"), "{first}");
    assert!(first.contains("tt.addptr"), "{first}");
    assert!(first.contains("tt.load"), "{first}");
    assert!(first.contains("arith.addf"), "{first}");
    assert!(first.contains("tt.store"), "{first}");
}

#[test]
fn grouped_expert_projections_are_verified_ttir() {
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let gate_up = build_grouped_projection(GroupedProjectionConfig {
            dtype,
            assignments: 32,
            input_size: 64,
            output_size: 128,
            local_experts: 4,
            source_row_divisor: 2,
            block_m: 16,
            block_n: 32,
            block_k: 32,
            gated_activation: None,
            multiply_routing_weight: false,
        })
        .unwrap();
        assert!(
            gate_up.contains("tt.func public @moe_grouped_gate_up"),
            "{gate_up}"
        );
        assert!(gate_up.contains("tt.dot"), "{gate_up}");
        assert!(gate_up.contains("scf.for"), "{gate_up}");
        assert_eq!(
            gate_up.matches("tt.dot").count(),
            1,
            "the K dimension must not be statically unrolled: {gate_up}"
        );
        assert!(gate_up.contains("tt.load"), "{gate_up}");
        assert!(gate_up.contains("tt.store"), "{gate_up}");
        assert!(gate_up.contains("arith.maxsi"), "{gate_up}");
        assert!(
            gate_up.matches("arith.minsi").count() >= 2,
            "assignment and expert addresses must both be clamped: {gate_up}"
        );

        for activation in [
            GatedActivation::Silu,
            GatedActivation::Gelu,
            GatedActivation::Relu,
        ] {
            let down = build_grouped_projection(GroupedProjectionConfig {
                dtype,
                assignments: 32,
                input_size: 64,
                output_size: 64,
                local_experts: 4,
                source_row_divisor: 1,
                block_m: 16,
                block_n: 32,
                block_k: 32,
                gated_activation: Some(activation),
                multiply_routing_weight: true,
            })
            .unwrap();
            assert!(down.contains("tt.func public @moe_grouped_down"), "{down}");
            assert!(down.contains("arith.mulf"), "{down}");
        }
    }

    assert!(build_grouped_projection(GroupedProjectionConfig {
        dtype: DType::F64,
        assignments: 32,
        input_size: 64,
        output_size: 64,
        local_experts: 4,
        source_row_divisor: 1,
        block_m: 16,
        block_n: 32,
        block_k: 32,
        gated_activation: None,
        multiply_routing_weight: false,
    })
    .is_err());
}

#[test]
fn invalid_kernel_contracts_fail_before_mlir() {
    assert!(matches!(
        Builder::new("not-a-symbol"),
        Err(Error::InvalidName(_))
    ));

    let mut builder = Builder::new("typed_failures").unwrap();
    let scalar = builder
        .argument("value", ArgumentKind::Scalar(DType::I32), None)
        .unwrap();
    assert!(matches!(
        builder.argument("value", ArgumentKind::Scalar(DType::I32), None),
        Err(Error::DuplicateArgument(_))
    ));
    assert!(matches!(builder.load(&scalar), Err(Error::ExpectedPointer)));
    assert!(matches!(builder.finish(), Err(Error::MissingTerminator)));

    let mut first = Builder::new("first").unwrap();
    let foreign = first.integer(1, DType::I32).unwrap();
    let mut second = Builder::new("second").unwrap();
    let local = second.integer(2, DType::I32).unwrap();
    assert!(matches!(
        second.add(&local, &foreign),
        Err(Error::ForeignValue)
    ));
}

#[test]
fn typed_kernel_spec_lowers_the_verified_artifact() {
    let mut builder = Builder::new("copy").unwrap();
    let alias_input = builder
        .argument(
            "input",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    let output = builder
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    let value = builder.load(&alias_input).unwrap();
    builder.store(&output, &value).unwrap();
    builder.return_void().unwrap();
    let tensor = TensorSpec::new(DType::F32, &[4]).unwrap();
    let kernel = KernelSpec::new(
        "copy",
        builder.finish().unwrap(),
        vec![tensor.clone()],
        vec![tensor],
        Vec::new(),
    )
    .unwrap();

    let context = Context::new();
    let tensor_type = context
        .ranked_tensor_type(nml_types::DType::F32, &[4])
        .unwrap();
    let mut block = Block::new(&context, &[tensor_type]).unwrap();
    let input = block.argument(0).unwrap();
    let call = kernel
        .lower(
            &context,
            &[input],
            KernelLaunch {
                grid: [4, 1, 1],
                warps: 1,
                stages: 1,
            },
        )
        .unwrap();
    let output = call.result(0).unwrap();
    block.append_operation(call).unwrap();
    block
        .append_operation(context.return_operation(&[output]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(block).unwrap();
    let function = context
        .function("typed_kernel", &[tensor_type], &[tensor_type], body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();
    let text = module.text();
    assert!(text.contains("__gpu$xla.gpu.triton"), "{text}");
    assert!(text.contains("grid_x = 4 : i32"), "{text}");

    let tensor = TensorSpec::new(DType::F32, &[4]).unwrap();
    assert!(matches!(
        KernelSpec::new(
            "copy",
            String::from("@copy("),
            vec![tensor.clone()],
            vec![tensor.clone()],
            Vec::new(),
        ),
        Err(Error::Mlir(_))
    ));

    let mut wrong_name = Builder::new("actual_name").unwrap();
    wrong_name.return_void().unwrap();
    assert!(matches!(
        KernelSpec::new(
            "declared_name",
            wrong_name.finish().unwrap(),
            vec![tensor.clone()],
            vec![tensor.clone()],
            Vec::new(),
        ),
        Err(Error::InvalidKernelSpec(_))
    ));

    let mut builder = Builder::new("alias_contract").unwrap();
    let alias_input = builder
        .argument(
            "input",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    let output = builder
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    let value = builder.load(&alias_input).unwrap();
    builder.store(&output, &value).unwrap();
    builder.return_void().unwrap();
    let tensor = TensorSpec::new(DType::F32, &[4]).unwrap();
    let bad_alias = KernelSpec::new(
        "alias_contract",
        builder.finish().unwrap(),
        vec![tensor.clone()],
        vec![tensor.clone()],
        vec![
            OutputAlias {
                output: 0,
                input: 0,
            },
            OutputAlias {
                output: 0,
                input: 0,
            },
        ],
    );
    assert!(matches!(bad_alias, Err(Error::InvalidKernelSpec(_))));

    let mut builder = Builder::new("launch_contract").unwrap();
    let input_pointer = builder
        .argument(
            "input",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    let output_pointer = builder
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    let value = builder.load(&input_pointer).unwrap();
    builder.store(&output_pointer, &value).unwrap();
    builder.return_void().unwrap();
    let kernel = KernelSpec::new(
        "launch_contract",
        builder.finish().unwrap(),
        vec![tensor.clone()],
        vec![tensor],
        Vec::new(),
    )
    .unwrap();
    let valid_launch = KernelLaunch {
        grid: [1, 1, 1],
        warps: 1,
        stages: 1,
    };
    assert!(matches!(
        kernel.lower(&context, &[], valid_launch),
        Err(Error::InvalidKernelSpec(_))
    ));
    for invalid_launch in [
        KernelLaunch {
            grid: [0, 1, 1],
            ..valid_launch
        },
        KernelLaunch {
            warps: 3,
            ..valid_launch
        },
        KernelLaunch {
            stages: 0,
            ..valid_launch
        },
    ] {
        assert!(matches!(
            kernel.lower(&context, &[input], invalid_launch),
            Err(Error::InvalidKernelSpec(_))
        ));
    }
}

#[test]
fn tensor_pointer_and_shape_operations_verify_as_ttir() {
    let mut builder = Builder::new("tensor_copy").unwrap();
    let input = builder
        .argument(
            "input",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            Some(16),
        )
        .unwrap();
    let output = builder
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            Some(16),
        )
        .unwrap();
    let offsets = builder.range(0, 16).unwrap();
    let columns = builder.expand_dimension(&offsets, 1).unwrap();
    let _tile = builder.broadcast(&columns, &[16, 16]).unwrap();
    let input_addresses = builder.add_pointer(&input, &offsets).unwrap();
    let output_addresses = builder.add_pointer(&output, &offsets).unwrap();
    let values = builder.load(&input_addresses).unwrap();
    let one = builder.float(1.0, DType::F32).unwrap();
    let ones = builder.splat(&one, &[16]).unwrap();
    let values = builder.add(&values, &ones).unwrap();
    let zero = builder.float(0.0, DType::F32).unwrap();
    let zeroes = builder.splat(&zero, &[16]).unwrap();
    let matrix = builder.splat(&one, &[16, 16]).unwrap();
    let accumulator = builder.splat(&zero, &[16, 16]).unwrap();
    let _product = builder.dot(&matrix, &matrix, &accumulator).unwrap();
    let positive = builder
        .compare(Comparison::Greater, &values, &zeroes)
        .unwrap();
    let exponent = builder.exp2(&values).unwrap();
    let bounded = builder.maximum(&exponent, &ones).unwrap();
    let _minimum = builder.minimum(&bounded, &ones).unwrap();
    let _product = builder.multiply(&bounded, &ones).unwrap();
    let _quotient = builder.divide(&bounded, &ones).unwrap();
    let _logarithm = builder.log2(&bounded).unwrap();
    let _square_root = builder.sqrt(&bounded).unwrap();
    let _conjunction = builder.bit_and(&positive, &positive).unwrap();
    let _maximum = builder.reduce(Reduction::Maximum, &bounded, 0).unwrap();
    let _sum = builder.reduce(Reduction::Sum, &bounded, 0).unwrap();
    let selected = builder.select(&positive, &bounded, &zeroes).unwrap();
    let cast = builder.cast(&selected, DType::F16).unwrap();
    let _restored = builder.cast(&cast, DType::F32).unwrap();
    let _negative = builder.negate(&selected).unwrap();
    let _remainder = builder.remainder(&offsets, &offsets).unwrap();
    let masked = builder
        .load_masked(&input_addresses, &positive, &zeroes)
        .unwrap();
    let lower = builder.integer(0, DType::I32).unwrap();
    let upper = builder.integer(4, DType::I32).unwrap();
    let step = builder.integer(1, DType::I32).unwrap();
    let _float_fill = builder.full_float(&[16], 0.0, DType::F32).unwrap();
    let _integer_fill = builder.full_integer(&[16], 0, DType::I32).unwrap();
    let _matrix_mask = builder.mask_2d(&offsets, &offsets).unwrap();
    let _loop_result = builder
        .for_loop(
            &lower,
            &upper,
            &step,
            std::slice::from_ref(&zero),
            |body, _, carried| Ok(vec![body.add(&carried[0], &one)?]),
        )
        .unwrap();
    let choose_loop_condition = builder
        .compare(Comparison::Greater, &upper, &lower)
        .unwrap();
    let choose_loop = builder
        .if_then_else(
            &choose_loop_condition,
            |branch| Ok(vec![branch.add(&upper, &step)?]),
            |branch| Ok(vec![branch.subtract(&upper, &step)?]),
        )
        .unwrap();
    builder
        .if_only(&choose_loop_condition, |branch| {
            let _ = branch.add(&upper, &step)?;
            Ok(())
        })
        .unwrap();
    let _while_result = builder
        .while_loop(
            &[lower.clone(), choose_loop[0].clone()],
            |before, carried| {
                let keep_going = before.compare(Comparison::Less, &carried[0], &carried[1])?;
                Ok((keep_going, carried.to_vec()))
            },
            |body, carried| Ok(vec![body.add(&carried[0], &step)?, carried[1].clone()]),
        )
        .unwrap();
    builder
        .store_masked(&output_addresses, &masked, &positive)
        .unwrap();
    builder.return_void().unwrap();
    let ttir = builder.finish().unwrap();
    assert!(ttir.contains("tt.make_range"), "{ttir}");
    assert!(ttir.contains("tt.expand_dims"), "{ttir}");
    assert!(ttir.contains("tt.broadcast"), "{ttir}");
    assert!(ttir.contains("arith.cmpf"), "{ttir}");
    assert!(ttir.contains("math.exp2"), "{ttir}");
    assert!(ttir.contains("math.log2"), "{ttir}");
    assert!(ttir.contains("math.sqrt"), "{ttir}");
    assert!(ttir.contains("arith.mulf"), "{ttir}");
    assert!(ttir.contains("arith.divf"), "{ttir}");
    assert!(ttir.contains("arith.andi"), "{ttir}");
    assert!(ttir.contains("arith.minnumf"), "{ttir}");
    assert!(ttir.contains("arith.maxnumf"), "{ttir}");
    assert!(ttir.contains("arith.select"), "{ttir}");
    assert!(ttir.contains("tt.reduce"), "{ttir}");
    assert!(ttir.contains("tt.reduce.return"), "{ttir}");
    assert!(ttir.contains("tt.dot"), "{ttir}");
    assert!(ttir.contains("scf.for"), "{ttir}");
    assert!(ttir.contains("scf.if"), "{ttir}");
    assert!(ttir.contains("scf.while"), "{ttir}");
    assert!(ttir.contains("scf.condition"), "{ttir}");
    assert!(ttir.contains("arith.truncf"), "{ttir}");
    assert!(ttir.contains("arith.negf"), "{ttir}");
    assert!(ttir.contains("arith.remsi"), "{ttir}");
    assert!(
        ttir.lines()
            .any(|line| line.contains("tt.load") && line.matches(',').count() == 2),
        "{ttir}"
    );
    assert!(
        ttir.lines()
            .any(|line| line.contains("tt.store") && line.matches(',').count() == 2),
        "{ttir}"
    );
    assert!(ttir.contains("tensor<16x!tt.ptr<f32>>"), "{ttir}");
}

fn geometry(all_decode: bool, batch_size: usize) -> AttentionGeometry {
    AttentionGeometry {
        core_count: 30,
        all_decode,
        num_tokens: batch_size,
        num_query_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        batch_size,
        page_size: 16,
        max_query_length: if all_decode { 1 } else { 128 },
    }
}

#[test]
fn cuda_attention_launch_policy_is_geometry_only() {
    assert_eq!(
        select_attention_launch(geometry(false, 4)).unwrap(),
        AttentionLaunch::TwoDimensional {
            block_m: 16,
            block_q: 4,
            tile_size: 64,
            total_query_blocks: 5,
            grid: [8, 5, 1],
            warps: 2,
            stages: 1,
        }
    );

    assert_eq!(
        select_attention_launch(geometry(true, 4)).unwrap(),
        AttentionLaunch::SplitK {
            block_m: 16,
            block_q: 4,
            tile_size: 16,
            total_query_blocks: 5,
            segments: 16,
            attention_grid: [5, 8, 16],
            attention_warps: 2,
            attention_stages: 1,
            reduction_grid: [4, 32, 1],
            reduction_warps: 1,
            reduction_stages: 1,
        }
    );

    let large_decode = geometry(true, 17);
    assert!(matches!(
        select_attention_launch(large_decode).unwrap(),
        AttentionLaunch::TwoDimensional {
            tile_size: 16,
            stages: 3,
            ..
        }
    ));

    let mut non_power_of_two_page = geometry(true, 4);
    non_power_of_two_page.page_size = 24;
    assert!(matches!(
        select_attention_launch(non_power_of_two_page).unwrap(),
        AttentionLaunch::SplitK { tile_size: 32, .. }
    ));
}

#[test]
fn non_power_of_two_gqa_uses_padded_masked_head_lanes() {
    let launch = select_attention_launch(AttentionGeometry {
        core_count: 80,
        all_decode: false,
        num_tokens: 8,
        num_query_heads: 6,
        num_kv_heads: 2,
        head_dim: 64,
        batch_size: 2,
        page_size: 16,
        max_query_length: 4,
    })
    .unwrap();
    let AttentionLaunch::TwoDimensional {
        block_m, block_q, ..
    } = launch
    else {
        panic!("prefill must use the 2D kernel");
    };
    assert_eq!((block_m, block_q), (16, 4));

    let ttir = build_paged_attention_2d(PagedAttention2dConfig {
        dtype: DType::F16,
        num_query_heads: 6,
        queries_per_kv: 3,
        page_size: 16,
        tile_size: 16,
        head_size: 64,
        padded_head_size: 64,
        block_q: 4,
        block_m: 16,
        sliding_window: None,
        causal: true,
    })
    .unwrap();
    assert!(ttir.contains("arith.constant 3 : i32"), "{ttir}");
    assert!(ttir.contains("arith.constant 4 : i32"), "{ttir}");
    assert!(ttir.contains("arith.andi"), "{ttir}");
}

#[test]
fn invalid_attention_geometry_never_reaches_ttir() {
    let mut invalid = geometry(false, 4);
    invalid.num_query_heads = 30;
    assert!(select_attention_launch(invalid)
        .unwrap_err()
        .to_string()
        .contains("not divisible"));

    invalid = geometry(false, 4);
    invalid.core_count = 0;
    assert!(select_attention_launch(invalid)
        .unwrap_err()
        .to_string()
        .contains("core count"));
}

#[test]
fn retained_paged_attention_2d_is_complete_verified_ttir() {
    let ttir = build_paged_attention_2d(PagedAttention2dConfig {
        dtype: DType::F16,
        num_query_heads: 8,
        queries_per_kv: 4,
        page_size: 16,
        tile_size: 16,
        head_size: 64,
        padded_head_size: 64,
        block_q: 4,
        block_m: 16,
        sliding_window: Some(128),
        causal: true,
    })
    .unwrap();
    for operation in [
        "scf.while",
        "scf.if",
        "scf.for",
        "tt.addptr",
        "tt.load",
        "tt.dot",
        "tt.reduce",
        "math.exp2",
        "tt.store",
    ] {
        assert!(ttir.contains(operation), "missing {operation}:\n{ttir}");
    }
    assert!(
        ttir.lines()
            .filter(|line| line.contains("tt.load") && line.matches(',').count() == 2)
            .count()
            >= 5,
        "query, query-position, page-table, key, and value loads must all be masked:\n{ttir}"
    );
    assert!(ttir.contains("tensor<16x64xf32>"), "{ttir}");
    assert!(ttir.contains("tensor<16x16xf16>"), "{ttir}");
    assert!(ttir.contains("arith.cmpf oeq"), "{ttir}");
}

#[test]
fn noncausal_sliding_window_has_both_position_bounds() {
    let ttir = build_paged_attention_2d(PagedAttention2dConfig {
        dtype: DType::F16,
        num_query_heads: 8,
        queries_per_kv: 4,
        page_size: 16,
        tile_size: 16,
        head_size: 64,
        padded_head_size: 64,
        block_q: 4,
        block_m: 16,
        sliding_window: Some(17),
        causal: false,
    })
    .unwrap();
    assert!(ttir.contains("arith.constant -17 : i32"), "{ttir}");
    assert!(ttir.contains("arith.cmpi sgt"), "{ttir}");
}

#[test]
fn split_k_segment_reduction_is_complete_verified_ttir() {
    let ttir = build_segment_reduction(SegmentReductionConfig {
        output_dtype: DType::Bf16,
        num_query_heads: 8,
        segments: 16,
        tile_size: 16,
        head_size: 80,
        padded_head_size: 128,
        block_q: 4,
    })
    .unwrap();
    assert!(
        ttir.contains("@paged_attention_segment_reduction"),
        "{ttir}"
    );
    assert!(ttir.matches("tt.reduce").count() >= 3, "{ttir}");
    assert!(ttir.contains("math.exp2"), "{ttir}");
    assert!(ttir.contains("tensor<16x128xf32>"), "{ttir}");
    assert!(ttir.contains("arith.truncf"), "{ttir}");
    assert!(ttir.contains("arith.maxsi"), "{ttir}");
    assert!(ttir.contains("arith.cmpf ogt"), "{ttir}");
    assert!(ttir.contains("tt.store"), "{ttir}");
}

#[test]
fn retained_paged_attention_3d_writes_fp32_segment_state() {
    let ttir = build_paged_attention_3d(PagedAttention3dConfig {
        attention: PagedAttention2dConfig {
            dtype: DType::Bf16,
            num_query_heads: 8,
            queries_per_kv: 4,
            page_size: 16,
            tile_size: 16,
            head_size: 80,
            padded_head_size: 128,
            block_q: 4,
            block_m: 16,
            sliding_window: Some(128),
            causal: true,
        },
        segments: 16,
    })
    .unwrap();
    assert!(ttir.contains("@paged_attention_3d"), "{ttir}");
    assert!(ttir.contains("tt.get_program_id z"), "{ttir}");
    assert!(ttir.contains("tensor<16x128xf32>"), "{ttir}");
    assert!(ttir.matches("tt.store").count() >= 3, "{ttir}");
    assert!(ttir.contains("tt.dot"), "{ttir}");
    assert!(ttir.contains("scf.for"), "{ttir}");
}
