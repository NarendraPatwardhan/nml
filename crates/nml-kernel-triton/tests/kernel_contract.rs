use nml_kernel_triton::{
    ArgumentKind, AttentionGeometry, AttentionLaunch, Builder, Comparison, DType, Error,
    GatedActivation, GroupedProjectionConfig, Kernel, KernelLaunch, KernelSpec,
    NvFp4EmbeddingConfig,
    NvFp4GroupedProjectionConfig, NvFp4GroupedRole, NvFp4LinearConfig, NvFp4QkvConfig, OutputAlias,
    PagedAttention2dConfig, PagedAttention3dConfig, Reduction, ScaleDotElement,
    SegmentReductionConfig, TensorSpec, build_grouped_projection, build_nvfp4_embedding,
    build_nvfp4_grouped_projection, build_nvfp4_linear, build_nvfp4_qkv, build_paged_attention_2d,
    build_paged_attention_3d, build_segment_reduction, select_attention_launch,
};
use nml_mlir::{Block, Context, Region};

#[test]
fn named_typed_kernel_is_deterministic_and_verified() {
    fn emit() -> Kernel {
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
    assert!(first.text().contains("tt.func public @add_one"), "{first}");
    assert!(first.text().contains("tt.get_program_id x"), "{first}");
    assert!(first.text().contains("tt.addptr"), "{first}");
    assert!(first.text().contains("tt.load"), "{first}");
    assert!(first.text().contains("arith.addf"), "{first}");
    assert!(first.text().contains("tt.store"), "{first}");
}

#[test]
fn nvfp4_linear_decodes_compact_tiles_inside_one_verified_kernel() {
    let config = NvFp4LinearConfig {
        dtype: DType::Bf16,
        rows: 17,
        outputs: 33,
        inputs: 80,
        block_m: 16,
        block_n: 32,
        block_k: 32,
        has_bias: true,
    };
    assert_eq!(config.launch_grid().unwrap(), [4, 1, 1]);
    let ttir = build_nvfp4_linear(config).unwrap();
    let ttir = ttir.text();
    for operation in [
        "@nvfp4_linear",
        "scf.for",
        "arith.shrui",
        "arith.andi",
        "tt.bitcast",
        "tt.reshape",
        "tt.dot",
        "tt.store",
    ] {
        assert!(ttir.contains(operation), "missing {operation}:\n{ttir}");
    }
    assert_eq!(ttir.matches("tt.dot").count(), 1, "{ttir}");
    assert!(!ttir.contains("math.exp2"), "{ttir}");
    assert!(
        ttir.rfind("arith.addf").unwrap() > ttir.find("tt.dot").unwrap(),
        "bias must be added after the contraction: {ttir}"
    );

    let bias_free = build_nvfp4_linear(NvFp4LinearConfig {
        has_bias: false,
        ..config
    })
    .unwrap();
    let bias_free = bias_free.text();
    let signature = ttir
        .lines()
        .find(|line| line.contains("tt.func public @nvfp4_linear"))
        .unwrap();
    let bias_free_signature = bias_free
        .lines()
        .find(|line| line.contains("tt.func public @nvfp4_linear"))
        .unwrap();
    assert_eq!(signature.matches("!tt.ptr<bf16>").count(), 3, "{signature}");
    assert_eq!(
        bias_free_signature.matches("!tt.ptr<bf16>").count(),
        2,
        "{bias_free_signature}"
    );
}

#[test]
fn nvfp4_decode_linear_uses_compact_gemv_without_dead_matrix_rows() {
    let config = NvFp4LinearConfig {
        dtype: DType::Bf16,
        rows: 1,
        outputs: 201_088,
        inputs: 2_880,
        block_m: 16,
        block_n: 64,
        block_k: 128,
        has_bias: false,
    };
    assert_eq!(config.launch_grid().unwrap(), [3_142, 1, 1]);
    let ttir = build_nvfp4_linear(config).unwrap();
    let ttir = ttir.text();
    assert!(ttir.contains("@nvfp4_linear_gemv"), "{ttir}");
    assert!(ttir.contains("tt.reduce"), "{ttir}");
    assert!(ttir.contains("tt.bitcast"), "{ttir}");
    assert!(ttir.contains("tt.reshape"), "{ttir}");
    assert!(!ttir.contains("tt.dot"), "{ttir}");
    assert!(!ttir.contains("math.exp2"), "{ttir}");
    assert!(
        ttir.rfind("arith.mulf").unwrap() > ttir.find("tt.reduce").unwrap(),
        "the tensor-wide scale must be applied after the K reduction: {ttir}"
    );
}

#[test]
fn nvfp4_decode_qkv_combines_three_projection_tails_in_one_grid() {
    let production = NvFp4QkvConfig {
        dtype: DType::Bf16,
        inputs: 2_880,
        query_outputs: 4_096,
        key_outputs: 512,
        value_outputs: 512,
        block_n: 8,
        block_k: 256,
        has_bias: true,
    };
    assert_eq!(production.launch_grid().unwrap(), [640, 1, 1]);

    let ttir = build_nvfp4_qkv(NvFp4QkvConfig {
        dtype: DType::Bf16,
        inputs: 80,
        query_outputs: 32,
        key_outputs: 8,
        value_outputs: 8,
        block_n: 8,
        block_k: 32,
        has_bias: true,
    })
    .unwrap();
    let ttir = ttir.text();
    assert!(ttir.contains("@nvfp4_qkv_gemv"), "{ttir}");
    assert_eq!(ttir.matches("scf.if ").count(), 3, "{ttir}");
    assert_eq!(ttir.matches("scf.for ").count(), 3, "{ttir}");
    assert_eq!(ttir.matches(" = \"tt.reduce\"").count(), 3, "{ttir}");
    assert_eq!(ttir.matches("tt.store").count(), 3, "{ttir}");
    assert!(!ttir.contains("tt.dot"), "{ttir}");
    assert!(!ttir.contains("math.exp2"), "{ttir}");
    let signature = ttir
        .lines()
        .find(|line| line.contains("tt.func public @nvfp4_qkv_gemv"))
        .unwrap();
    assert_eq!(signature.matches("!tt.ptr<bf16>").count(), 7, "{signature}");
    assert_eq!(signature.matches("!tt.ptr<i8>").count(), 6, "{signature}");
    assert_eq!(signature.matches("!tt.ptr<f32>").count(), 3, "{signature}");
}

#[test]
fn nvfp4_embedding_decodes_only_selected_compact_rows() {
    for index_dtype in [DType::I32, DType::I64] {
        let config = NvFp4EmbeddingConfig {
            dtype: DType::Bf16,
            index_dtype,
            rows: 7,
            vocabulary: 33,
            width: 80,
            block_m: 16,
            block_n: 32,
        };
        assert_eq!(config.launch_grid().unwrap(), [1, 3, 1]);
        let ttir = build_nvfp4_embedding(config).unwrap();
        let ttir = ttir.text();
        for operation in [
            "@nvfp4_embedding",
            "arith.shrui",
            "arith.andi",
            "tt.bitcast",
            "tt.reshape",
            "tt.load",
            "tt.store",
        ] {
            assert!(ttir.contains(operation), "missing {operation}:\n{ttir}");
        }
        assert!(!ttir.contains("tt.dot"), "{ttir}");
        assert!(!ttir.contains("math.exp2"), "{ttir}");
    }
}

#[test]
fn nvfp4_grouped_experts_keep_routing_and_decode_inside_verified_kernels() {
    for dtype in [DType::F16, DType::Bf16] {
        let gate_up = build_nvfp4_grouped_projection(NvFp4GroupedProjectionConfig {
            dtype,
            tokens: 16,
            assignments: 32,
            input_size: 64,
            output_size: 64,
            local_experts: 4,
            source_row_divisor: 2,
            block_m: 16,
            block_n: 32,
            block_k: 32,
            role: NvFp4GroupedRole::GateUpActivated,
        })
        .unwrap();
        let gate_up = gate_up.text();
        assert!(gate_up.contains("@nvfp4_grouped_gate_up"), "{gate_up}");
        assert!(gate_up.contains("arith.shrui"), "{gate_up}");
        assert!(gate_up.contains("math.exp2"), "{gate_up}");
        assert!(gate_up.contains("scf.if"), "{gate_up}");
        assert_eq!(gate_up.matches("tt.dot").count(), 2, "{gate_up}");
        assert_eq!(gate_up.matches("math.exp2").count(), 1, "{gate_up}");
        assert!(
            gate_up.matches("arith.minnumf").count() >= 2,
            "clamped SwiGLU must be applied in the gate/up epilogue: {gate_up}"
        );

        let down = build_nvfp4_grouped_projection(NvFp4GroupedProjectionConfig {
            dtype,
            tokens: 16,
            assignments: 32,
            input_size: 64,
            output_size: 64,
            local_experts: 4,
            source_row_divisor: 1,
            block_m: 16,
            block_n: 32,
            block_k: 32,
            role: NvFp4GroupedRole::Down,
        })
        .unwrap();
        let down = down.text();
        assert!(down.contains("@nvfp4_grouped_down"), "{down}");
        assert!(down.contains("arith.shrui"), "{down}");
        assert!(!down.contains("math.exp2"), "{down}");
        assert!(down.contains("scf.if"), "{down}");
        assert_eq!(down.matches("tt.dot").count(), 1, "{down}");
        assert!(
            down.matches("arith.minnumf").count() < 2,
            "down must consume the already-activated intermediate: {down}"
        );
    }
}

#[test]
fn nvfp4_decode_experts_use_selected_expert_gemv_kernels() {
    for dtype in [DType::F16, DType::Bf16] {
        let gate_up = build_nvfp4_grouped_projection(NvFp4GroupedProjectionConfig {
            dtype,
            tokens: 1,
            assignments: 4,
            input_size: 64,
            output_size: 64,
            local_experts: 4,
            source_row_divisor: 4,
            block_m: 16,
            block_n: 32,
            block_k: 32,
            role: NvFp4GroupedRole::GateUpActivated,
        })
        .unwrap();
        let gate_up = gate_up.text();
        assert!(gate_up.contains("@nvfp4_grouped_gate_up_gemv"), "{gate_up}");
        assert_eq!(gate_up.matches(" = \"tt.reduce\"").count(), 2, "{gate_up}");
        assert_eq!(gate_up.matches("math.exp2").count(), 1, "{gate_up}");
        assert!(!gate_up.contains("tt.dot"), "{gate_up}");
        assert!(
            gate_up.rfind("arith.mulf").unwrap() > gate_up.rfind("tt.reduce").unwrap(),
            "the tensor-wide scale must be applied after the K reduction: {gate_up}"
        );

        let down = build_nvfp4_grouped_projection(NvFp4GroupedProjectionConfig {
            dtype,
            tokens: 1,
            assignments: 4,
            input_size: 64,
            output_size: 64,
            local_experts: 4,
            source_row_divisor: 1,
            block_m: 16,
            block_n: 32,
            block_k: 32,
            role: NvFp4GroupedRole::Down,
        })
        .unwrap();
        let down = down.text();
        assert!(down.contains("@nvfp4_grouped_down_gemv"), "{down}");
        assert_eq!(down.matches(" = \"tt.reduce\"").count(), 1, "{down}");
        assert!(!down.contains("math.exp2"), "{down}");
        assert!(!down.contains("tt.dot"), "{down}");
        assert!(
            down.rfind("arith.mulf").unwrap() > down.find("tt.reduce").unwrap(),
            "the tensor-wide scale must be applied after the K reduction: {down}"
        );
    }
}

#[test]
fn microscaling_dot_surface_is_typed_and_verified_by_the_pinned_dialect() {
    let mut builder = Builder::new("nvfp4_scaled_dot_contract").unwrap();
    let left = builder.full_integer(&[128, 128], 0, DType::I8).unwrap();
    let left_scale = builder.full_integer(&[128, 4], 0, DType::I8).unwrap();
    let right = builder.full_integer(&[64, 128], 0, DType::I8).unwrap();
    let right_scale = builder.full_integer(&[128, 4], 0, DType::I8).unwrap();
    let accumulator = builder.full_float(&[128, 128], 0.0, DType::F32).unwrap();
    let result = builder
        .dot_scaled(
            &left,
            &right,
            &accumulator,
            Some(&left_scale),
            Some(&right_scale),
            ScaleDotElement::E4M3,
            ScaleDotElement::E2M1,
            true,
            true,
        )
        .unwrap();
    let zero = builder.full_float(&[128, 128], 0.0, DType::F32).unwrap();
    let _ = builder.add(&result, &zero).unwrap();
    builder.return_void().unwrap();
    let ttir = builder.finish().unwrap();
    let ttir = ttir.text();
    assert!(ttir.contains("tt.dot_scaled"), "{ttir}");
    assert!(ttir.contains("lhs = e4m3 rhs = e2m1"), "{ttir}");
}

#[test]
fn grouped_expert_projections_are_verified_ttir() {
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        for activation in [
            GatedActivation::Silu,
            GatedActivation::Gelu,
            GatedActivation::Relu,
        ] {
            let gate_up = build_grouped_projection(GroupedProjectionConfig {
                dtype,
                assignments: 32,
                input_size: 64,
                output_size: 64,
                local_experts: 4,
                source_row_divisor: 2,
                block_m: 16,
                block_n: 32,
                block_k: 32,
                gated_activation: Some(activation),
                multiply_routing_weight: false,
            })
            .unwrap();
            let gate_up = gate_up.text();
            assert!(
                gate_up.contains("tt.func public @moe_grouped_gate_up"),
                "{gate_up}"
            );
            assert!(gate_up.contains("scf.for"), "{gate_up}");
            assert!(gate_up.contains("scf.if"), "{gate_up}");
            assert_eq!(gate_up.matches("tt.dot").count(), 2, "{gate_up}");
            assert!(gate_up.contains("tt.load"), "{gate_up}");
            assert!(gate_up.contains("tt.store"), "{gate_up}");
            assert!(
                gate_up.find("scf.if") < gate_up.find("scf.for"),
                "inactive and non-local blocks must branch before contraction: {gate_up}"
            );
            if activation == GatedActivation::Gelu {
                assert!(gate_up.contains("math.exp2"), "{gate_up}");
                assert!(
                    !gate_up.contains("math.tanh"),
                    "XLA's retained Triton pipeline cannot legalize math.tanh: {gate_up}"
                );
            }
        }

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
            gated_activation: None,
            multiply_routing_weight: true,
        })
        .unwrap();
        let down = down.text();
        assert!(down.contains("tt.func public @moe_grouped_down"), "{down}");
        assert_eq!(down.matches("tt.dot").count(), 1, "{down}");
        assert!(!down.contains("math.exp2"), "{down}");
        assert!(down.contains("arith.mulf"), "{down}");
    }

    assert!(
        build_grouped_projection(GroupedProjectionConfig {
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
        .is_err()
    );
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

    let integer_tensor = second.full_integer(&[16], 0, DType::I32).unwrap();
    assert!(matches!(
        second.bitcast(&integer_tensor, DType::F16),
        Err(Error::TypeMismatch { operation: "bitcast" })
    ));
    assert!(matches!(
        second.reshape(&integer_tensor, &[2, 4]),
        Err(Error::TypeMismatch {
            operation: "reshape"
        })
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
            &[("input", input)],
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

    // TTIR and StableHLO used to carry independently authored argument lists.
    // Reject both count and type drift here so malformed custom calls cannot
    // reach XLA's unchecked LLVM argument annotation.
    let tensor = TensorSpec::new(DType::F32, &[4]).unwrap();
    let mut count_mismatch = Builder::new("count_mismatch").unwrap();
    count_mismatch
        .argument(
            "input",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    count_mismatch
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    count_mismatch.return_void().unwrap();
    assert!(matches!(
        KernelSpec::new(
            count_mismatch.finish().unwrap(),
            vec![tensor.clone(), tensor.clone()],
            vec![tensor.clone()],
            Vec::new(),
        ),
        Err(Error::InvalidKernelSpec(_))
    ));

    let mut type_mismatch = Builder::new("type_mismatch").unwrap();
    type_mismatch
        .argument(
            "input",
            ArgumentKind::Pointer {
                element: DType::I32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    type_mismatch
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    type_mismatch.return_void().unwrap();
    assert!(matches!(
        KernelSpec::new(
            type_mismatch.finish().unwrap(),
            vec![tensor.clone()],
            vec![tensor.clone()],
            Vec::new(),
        ),
        Err(Error::InvalidKernelSpec(_))
    ));

    let mut scalar_argument = Builder::new("scalar_argument").unwrap();
    scalar_argument
        .argument("input", ArgumentKind::Scalar(DType::F32), None)
        .unwrap();
    scalar_argument
        .argument(
            "output",
            ArgumentKind::Pointer {
                element: DType::F32,
                address_space: 1,
            },
            None,
        )
        .unwrap();
    scalar_argument.return_void().unwrap();
    assert!(matches!(
        KernelSpec::new(
            scalar_argument.finish().unwrap(),
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
    assert!(matches!(
        kernel.lower(&context, &[("wrong_input", input)], valid_launch),
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
            kernel.lower(&context, &[("input", input)], invalid_launch),
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
    let integer_bits = builder.full_integer(&[16], 0, DType::I32).unwrap();
    let reinterpreted = builder.bitcast(&integer_bits, DType::F32).unwrap();
    let _regrouped = builder.reshape(&reinterpreted, &[4, 4]).unwrap();
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
    let ttir = ttir.text();
    assert!(ttir.contains("tt.make_range"), "{ttir}");
    assert!(ttir.contains("tt.expand_dims"), "{ttir}");
    assert!(ttir.contains("tt.broadcast"), "{ttir}");
    assert!(ttir.contains("tt.reshape"), "{ttir}");
    assert!(ttir.contains("tt.bitcast"), "{ttir}");
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

    let mut wide_page = geometry(true, 4);
    wide_page.page_size = 256;
    assert!(matches!(
        select_attention_launch(wide_page).unwrap(),
        AttentionLaunch::SplitK { tile_size: 64, .. }
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
        learned_sinks: false,
    })
    .unwrap();
    let ttir = ttir.text();
    assert!(ttir.contains("arith.constant 3 : i32"), "{ttir}");
    assert!(ttir.contains("arith.constant 4 : i32"), "{ttir}");
    assert!(ttir.contains("arith.andi"), "{ttir}");
}

#[test]
fn invalid_attention_geometry_never_reaches_ttir() {
    let mut invalid = geometry(false, 4);
    invalid.num_query_heads = 30;
    assert!(
        select_attention_launch(invalid)
            .unwrap_err()
            .to_string()
            .contains("not divisible")
    );

    invalid = geometry(false, 4);
    invalid.core_count = 0;
    assert!(
        select_attention_launch(invalid)
            .unwrap_err()
            .to_string()
            .contains("core count")
    );
}

#[test]
fn sub_tile_head_dimensions_are_not_valid_ttir_specializations() {
    let result = build_paged_attention_2d(PagedAttention2dConfig {
        dtype: DType::F32,
        num_query_heads: 2,
        queries_per_kv: 2,
        page_size: 2,
        tile_size: 16,
        head_size: 4,
        padded_head_size: 4,
        block_q: 1,
        block_m: 2,
        sliding_window: None,
        causal: true,
        learned_sinks: false,
    });
    assert!(matches!(result, Err(Error::InvalidKernelSpec(_))));
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
        learned_sinks: false,
    })
    .unwrap();
    let ttir = ttir.text();
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
fn paged_attention_2d_initializes_online_softmax_from_learned_sinks() {
    let config = PagedAttention2dConfig {
        dtype: DType::Bf16,
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
        learned_sinks: true,
    };
    let ttir = build_paged_attention_2d(config).unwrap();
    let without_sinks = build_paged_attention_2d(PagedAttention2dConfig {
        learned_sinks: false,
        ..config
    })
    .unwrap();
    let ttir = ttir.text();
    let without_sinks = without_sinks.text();
    assert_eq!(
        ttir.matches("!tt.ptr<bf16>").count(),
        without_sinks.matches("!tt.ptr<bf16>").count() + 5,
        "the extra bf16 sink pointer must survive in the function type and its splat/addptr/load operations:\n{ttir}"
    );
    assert_eq!(
        ttir.matches("arith.extf").count(),
        without_sinks.matches("arith.extf").count() + 1,
        "the bf16 sink must be promoted to the F32 online-softmax state:\n{ttir}"
    );
    assert!(
        ttir.contains("1.44269502"),
        "sink logits must enter the kernel's base-two softmax domain:\n{ttir}"
    );
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
        learned_sinks: false,
    })
    .unwrap();
    let ttir = ttir.text();
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
        learned_sinks: false,
    })
    .unwrap();
    let ttir = ttir.text();
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
fn split_k_reduction_adds_one_learned_sink_to_the_global_denominator() {
    let config = SegmentReductionConfig {
        output_dtype: DType::Bf16,
        num_query_heads: 8,
        segments: 16,
        tile_size: 16,
        head_size: 80,
        padded_head_size: 128,
        block_q: 4,
        learned_sinks: true,
    };
    let ttir = build_segment_reduction(config).unwrap();
    let without_sinks = build_segment_reduction(SegmentReductionConfig {
        learned_sinks: false,
        ..config
    })
    .unwrap();
    let ttir = ttir.text();
    let without_sinks = without_sinks.text();
    assert_eq!(
        ttir.matches("!tt.ptr<bf16>").count(),
        without_sinks.matches("!tt.ptr<bf16>").count() + 3,
        "the extra bf16 sink pointer must survive in the function type and its load/addptr operations:\n{ttir}"
    );
    assert_eq!(
        ttir.matches("arith.maxnumf").count(),
        without_sinks.matches("arith.maxnumf").count() + 1,
        "the global maximum must include the learned sink exactly once:\n{ttir}"
    );
    assert_eq!(
        ttir.matches("math.exp2").count(),
        without_sinks.matches("math.exp2").count() + 1,
        "the denominator must include the learned sink exactly once:\n{ttir}"
    );
    assert!(ttir.contains("1.44269502"), "{ttir}");
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
            learned_sinks: false,
        },
        segments: 16,
    })
    .unwrap();
    let ttir = ttir.text();
    assert!(ttir.contains("@paged_attention_3d"), "{ttir}");
    assert!(ttir.contains("tt.get_program_id z"), "{ttir}");
    assert!(ttir.contains("tensor<16x128xf32>"), "{ttir}");
    assert!(ttir.matches("tt.store").count() >= 3, "{ttir}");
    assert!(ttir.contains("tt.dot"), "{ttir}");
    assert!(ttir.contains("scf.for"), "{ttir}");
}
