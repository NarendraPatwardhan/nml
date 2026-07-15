use nml_mlir::{
    stablehlo_current_version, Block, Context, Error, IndexCastKind, OutputOperandAlias, Region,
    ShardyDimension, TritonCustomCall,
};
use nml_types::DType;

#[test]
fn canonical_dtypes_and_index_have_distinct_mlir_contracts() {
    let context = Context::new();
    for dtype in DType::ALL {
        assert_eq!(
            context.dtype(dtype).unwrap().text(),
            dtype.stablehlo_spelling()
        );
    }
    assert_eq!(context.index_type().text(), "index");
    assert!(!DType::ALL
        .iter()
        .any(|dtype| dtype.stablehlo_spelling() == "index"));
}

#[test]
fn stablehlo_complex_module_verifies_and_serializes() {
    let context = Context::new();
    let module = context
        .parse_module(
            r#"module {
  func.func @complex_round_trip(%real: tensor<2xf32>, %imaginary: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    %complex = stablehlo.complex %real, %imaginary : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
    %real_result = stablehlo.real %complex : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    %imaginary_result = stablehlo.imag %complex : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    return %real_result, %imaginary_result : tensor<2xf32>, tensor<2xf32>
  }
}"#,
        )
        .unwrap();
    module.verify().unwrap();
    let text = module.text();
    assert!(text.contains("stablehlo.complex"));
    assert!(text.contains("complex<f32>"));
    let bytecode = module.bytecode().unwrap();
    assert!(bytecode.starts_with(b"ML\xefR"));
    let portable = module
        .portable_artifact(&stablehlo_current_version())
        .unwrap();
    assert!(!portable.is_empty());
}

#[test]
fn invalid_stablehlo_is_a_diagnostic_rich_error() {
    let context = Context::new();
    let error = match context.parse_module("module { stablehlo.not_an_operation }") {
        Ok(_) => panic!("unknown operation unexpectedly parsed"),
        Err(error) => error,
    };
    match error {
        Error::Parse { diagnostics } => {
            assert!(diagnostics.contains("not_an_operation"));
            assert!(!diagnostics.is_empty());
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn index_constants_and_both_index_casts_remain_compiler_internal() {
    let context = Context::new();
    let i64_type = context.dtype(DType::I64).unwrap();
    let mut block = Block::new(&context, &[]).unwrap();
    let index = context.index_constant(7).unwrap();
    let index_value = index.result(0).unwrap();
    block.append_operation(index).unwrap();
    let signed = context
        .index_cast(index_value, i64_type, IndexCastKind::Signed)
        .unwrap();
    let signed_value = signed.result(0).unwrap();
    block.append_operation(signed).unwrap();
    let unsigned = context
        .index_cast(index_value, i64_type, IndexCastKind::Unsigned)
        .unwrap();
    let unsigned_value = unsigned.result(0).unwrap();
    block.append_operation(unsigned).unwrap();
    block
        .append_operation(
            context
                .return_operation(&[signed_value, unsigned_value])
                .unwrap(),
        )
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(block).unwrap();
    let function = context
        .function("index_contract", &[], &[i64_type, i64_type], body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();
    assert!(module.text().contains("arith.index_cast"));
    assert!(module.text().contains("arith.index_castui"));
}

#[test]
fn stablehlo_constants_are_built_with_owned_operations() {
    let context = Context::new();
    let tensor_type = context.ranked_tensor_type(DType::F32, &[2]).unwrap();
    let value = context
        .parse_attribute("dense<[1.0, 2.0]> : tensor<2xf32>")
        .unwrap();
    let mut block = Block::new(&context, &[]).unwrap();
    let constant = context.constant(tensor_type, value).unwrap();
    let result = constant.result(0).unwrap();
    block.append_operation(constant).unwrap();
    block
        .append_operation(context.return_operation(&[result]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(block).unwrap();
    let function = context
        .function("constant", &[], &[tensor_type], body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();
    assert!(module.text().contains("stablehlo.constant"));

    assert!(matches!(
        context.parse_attribute("not-an-attribute"),
        Err(Error::InvalidAttribute { .. })
    ));
}

#[test]
fn stablehlo_while_owns_typed_regions_and_bytecode() {
    let context = Context::new();
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[]).unwrap();
    let scalar_bool = context.ranked_tensor_type(DType::Bool, &[]).unwrap();

    let mut condition_block = Block::new(&context, &[scalar_i32]).unwrap();
    let counter = condition_block.argument(0).unwrap();
    let limit_literal = context.parse_attribute("dense<4> : tensor<i32>").unwrap();
    let limit = context.constant(scalar_i32, limit_literal).unwrap();
    let limit_value = limit.result(0).unwrap();
    condition_block.append_operation(limit).unwrap();
    let predicate = context
        .compare(
            counter,
            limit_value,
            scalar_bool,
            nml_mlir::StableHloComparison::Lt,
            nml_mlir::StableHloComparisonType::Signed,
        )
        .unwrap();
    let predicate_value = predicate.result(0).unwrap();
    condition_block.append_operation(predicate).unwrap();
    condition_block
        .append_operation(context.stablehlo_return(&[predicate_value]).unwrap())
        .unwrap();
    let mut condition = Region::new(&context).unwrap();
    condition.append_block(condition_block).unwrap();

    let mut body_block = Block::new(&context, &[scalar_i32]).unwrap();
    let counter = body_block.argument(0).unwrap();
    let one_literal = context.parse_attribute("dense<1> : tensor<i32>").unwrap();
    let one = context.constant(scalar_i32, one_literal).unwrap();
    let one_value = one.result(0).unwrap();
    body_block.append_operation(one).unwrap();
    let incremented = context.add(counter, one_value, scalar_i32).unwrap();
    let incremented_value = incremented.result(0).unwrap();
    body_block.append_operation(incremented).unwrap();
    body_block
        .append_operation(context.stablehlo_return(&[incremented_value]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(body_block).unwrap();

    let mut function_block = Block::new(&context, &[]).unwrap();
    let zero_literal = context.parse_attribute("dense<0> : tensor<i32>").unwrap();
    let zero = context.constant(scalar_i32, zero_literal).unwrap();
    let zero_value = zero.result(0).unwrap();
    function_block.append_operation(zero).unwrap();
    let loop_operation = context
        .stablehlo_while(&[zero_value], &[scalar_i32], condition, body)
        .unwrap();
    let result = loop_operation.result(0).unwrap();
    function_block.append_operation(loop_operation).unwrap();
    function_block
        .append_operation(context.return_operation(&[result]).unwrap())
        .unwrap();
    let mut function_body = Region::new(&context).unwrap();
    function_body.append_block(function_block).unwrap();
    let function = context
        .function("while_contract", &[], &[scalar_i32], function_body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();
    let text = module.text();
    assert!(text.contains("stablehlo.while"));
    assert!(text.contains("stablehlo.return"));
    assert!(module.bytecode().unwrap().starts_with(b"ML\xefR"));
}

#[test]
fn stablehlo_while_rejects_an_unterminated_owned_region() {
    let context = Context::new();
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[]).unwrap();
    let scalar_bool = context.ranked_tensor_type(DType::Bool, &[]).unwrap();

    let mut condition_block = Block::new(&context, &[scalar_i32]).unwrap();
    let true_literal = context.parse_attribute("dense<true> : tensor<i1>").unwrap();
    let predicate = context.constant(scalar_bool, true_literal).unwrap();
    let predicate_value = predicate.result(0).unwrap();
    condition_block.append_operation(predicate).unwrap();
    condition_block
        .append_operation(context.stablehlo_return(&[predicate_value]).unwrap())
        .unwrap();
    let mut condition = Region::new(&context).unwrap();
    condition.append_block(condition_block).unwrap();

    let body_block = Block::new(&context, &[scalar_i32]).unwrap();
    let mut unterminated_body = Region::new(&context).unwrap();
    unterminated_body.append_block(body_block).unwrap();

    let mut function_block = Block::new(&context, &[]).unwrap();
    let zero_literal = context.parse_attribute("dense<0> : tensor<i32>").unwrap();
    let zero = context.constant(scalar_i32, zero_literal).unwrap();
    let zero_value = zero.result(0).unwrap();
    function_block.append_operation(zero).unwrap();
    let loop_operation = context
        .stablehlo_while(&[zero_value], &[scalar_i32], condition, unterminated_body)
        .unwrap();
    let result = loop_operation.result(0).unwrap();
    function_block.append_operation(loop_operation).unwrap();
    function_block
        .append_operation(context.return_operation(&[result]).unwrap())
        .unwrap();
    let mut function_body = Region::new(&context).unwrap();
    function_body.append_block(function_block).unwrap();
    let function = context
        .function("invalid_while", &[], &[scalar_i32], function_body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    assert!(matches!(module.verify(), Err(Error::Verification { .. })));
}

#[test]
fn stablehlo_control_flow_rejects_state_drift_before_mlir_construction() {
    let context = Context::new();
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[]).unwrap();
    let scalar_f32 = context.ranked_tensor_type(DType::F32, &[]).unwrap();
    let zero_literal = context.parse_attribute("dense<0> : tensor<i32>").unwrap();
    let zero = context.constant(scalar_i32, zero_literal).unwrap();
    let zero_value = zero.result(0).unwrap();

    let mut condition = Region::new(&context).unwrap();
    condition
        .append_block(Block::new(&context, &[scalar_f32]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(Block::new(&context, &[scalar_i32]).unwrap())
        .unwrap();
    assert!(matches!(
        context.stablehlo_while(&[zero_value], &[scalar_i32], condition, body),
        Err(Error::InvalidOperation { source })
            if source.contains("condition entry arguments")
    ));

    let mut empty_condition = Region::new(&context).unwrap();
    empty_condition
        .append_block(Block::new(&context, &[scalar_i32]).unwrap())
        .unwrap();
    let mut empty_body = Region::new(&context).unwrap();
    empty_body
        .append_block(Block::new(&context, &[scalar_i32]).unwrap())
        .unwrap();
    assert!(matches!(
        context.stablehlo_while(
            &[zero_value],
            &[scalar_i32, scalar_i32],
            empty_condition,
            empty_body,
        ),
        Err(Error::InvalidOperation { source }) if source.contains("equal nonzero")
    ));
}

#[test]
fn stablehlo_case_rejects_non_i32_indices_and_capturing_branches() {
    let context = Context::new();
    let scalar_i32 = context.ranked_tensor_type(DType::I32, &[]).unwrap();
    let scalar_i64 = context.ranked_tensor_type(DType::I64, &[]).unwrap();
    let index_literal = context.parse_attribute("dense<0> : tensor<i64>").unwrap();
    let index = context.constant(scalar_i64, index_literal).unwrap();
    let index_value = index.result(0).unwrap();
    let mut branch = Region::new(&context).unwrap();
    branch
        .append_block(Block::new(&context, &[]).unwrap())
        .unwrap();
    assert!(matches!(
        context.stablehlo_case(index_value, &[scalar_i32], vec![branch]),
        Err(Error::InvalidOperation { source }) if source.contains("tensor<i32>")
    ));

    let index_literal = context.parse_attribute("dense<0> : tensor<i32>").unwrap();
    let index = context.constant(scalar_i32, index_literal).unwrap();
    let index_value = index.result(0).unwrap();
    let mut capturing_branch = Region::new(&context).unwrap();
    capturing_branch
        .append_block(Block::new(&context, &[scalar_i32]).unwrap())
        .unwrap();
    assert!(matches!(
        context.stablehlo_case(index_value, &[scalar_i32], vec![capturing_branch]),
        Err(Error::InvalidOperation { source }) if source.contains("entry arguments")
    ));
}

#[test]
fn shardy_manual_computation_owns_a_verified_local_region() {
    let context = Context::new();
    let global = context.ranked_tensor_type(DType::F32, &[4]).unwrap();
    let local = context.ranked_tensor_type(DType::F32, &[2]).unwrap();
    let sharding = context
        .shardy_tensor_sharding("nml_mesh", &[ShardyDimension::Sharded("axis_1")], &[])
        .unwrap();
    let per_value = context.shardy_per_value_sharding(&[sharding]).unwrap();
    let manual_axes = context.shardy_manual_axes(&["axis_1"]).unwrap();

    let mut local_block = Block::new(&context, &[local]).unwrap();
    let local_value = local_block.argument(0).unwrap();
    local_block
        .append_operation(context.shardy_return(&[local_value]).unwrap())
        .unwrap();
    let mut manual_body = Region::new(&context).unwrap();
    manual_body.append_block(local_block).unwrap();

    let mut function_block = Block::new(&context, &[global]).unwrap();
    let global_value = function_block.argument(0).unwrap();
    let manual = context
        .shardy_manual_computation(
            &[global_value],
            &[global],
            per_value,
            per_value,
            manual_axes,
            manual_body,
        )
        .unwrap();
    let result = manual.result(0).unwrap();
    function_block.append_operation(manual).unwrap();
    function_block
        .append_operation(context.return_operation(&[result]).unwrap())
        .unwrap();
    let mut function_body = Region::new(&context).unwrap();
    function_body.append_block(function_block).unwrap();
    let function = context
        .function("manual_contract", &[global], &[global], function_body)
        .unwrap();

    let mesh = context.shardy_mesh(&[("axis_1", 2)], &[]).unwrap();
    let mut module = context.empty_module().unwrap();
    module
        .append_operation(context.shardy_mesh_operation("nml_mesh", mesh).unwrap())
        .unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();
    let text = module.text();
    assert!(text.contains("sdy.manual_computation"));
    assert!(text.contains("manual_axes"));
    assert!(text.contains("axis_1"));
    assert!(text.contains("sdy.return"));
}

#[test]
fn partition_identity_and_rectangular_collective_groups_are_typed() {
    let context = Context::new();
    let scalar_u32 = context.ranked_tensor_type(DType::U32, &[]).unwrap();
    let scalar_f32 = context.ranked_tensor_type(DType::F32, &[]).unwrap();
    let tensor = context.ranked_tensor_type(DType::F32, &[2]).unwrap();
    let mut function_block = Block::new(&context, &[tensor]).unwrap();

    let partition = context.partition_id(scalar_u32).unwrap();
    function_block.append_operation(partition).unwrap();

    let mut reducer_block = Block::new(&context, &[scalar_f32; 2]).unwrap();
    let sum = context
        .add(
            reducer_block.argument(0).unwrap(),
            reducer_block.argument(1).unwrap(),
            scalar_f32,
        )
        .unwrap();
    let sum_value = sum.result(0).unwrap();
    reducer_block.append_operation(sum).unwrap();
    reducer_block
        .append_operation(context.stablehlo_return(&[sum_value]).unwrap())
        .unwrap();
    let mut reducer = Region::new(&context).unwrap();
    reducer.append_block(reducer_block).unwrap();
    let reduce = context
        .all_reduce(
            function_block.argument(0).unwrap(),
            tensor,
            &[vec![0, 2], vec![1, 3]],
            7,
            reducer,
        )
        .unwrap();
    let reduced = reduce.result(0).unwrap();
    function_block.append_operation(reduce).unwrap();
    function_block
        .append_operation(context.return_operation(&[reduced]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(function_block).unwrap();
    let function = context
        .function("collective_groups", &[tensor], &[tensor], body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();
    let text = module.text();
    assert!(text.contains("stablehlo.partition_id"), "{text}");
    assert!(text.contains("dense<[[0, 2], [1, 3]]>"), "{text}");

    assert!(matches!(
        context.partition_id(context.ranked_tensor_type(DType::I32, &[]).unwrap()),
        Err(Error::InvalidOperation { .. })
    ));
}

#[test]
fn shardy_mesh_rejects_invalid_topology_before_calling_the_c_api() {
    let context = Context::new();
    assert!(matches!(
        context.shardy_mesh(&[], &[]),
        Err(Error::InvalidAttribute { .. })
    ));
    assert!(matches!(
        context.shardy_mesh(&[("data", 2), ("data", 2)], &[]),
        Err(Error::InvalidAttribute { .. })
    ));
    assert!(matches!(
        context.shardy_mesh(&[("data", 2)], &[0]),
        Err(Error::InvalidAttribute { .. })
    ));
    assert!(matches!(
        context.shardy_mesh(&[("data", 2)], &[0, 0]),
        Err(Error::InvalidAttribute { .. })
    ));
}

#[test]
fn pass_managers_are_owned_and_pipeline_errors_are_structured() {
    let context = Context::new();
    let mut module = context.parse_module("module {}").unwrap();
    let mut manager = context.pass_manager();
    manager.parse_pipeline("builtin.module()").unwrap();
    manager.run(&mut module).unwrap();

    let mut invalid = context.pass_manager();
    assert!(matches!(
        invalid.parse_pipeline("builtin.module(not-a-real-pass)"),
        Err(Error::PassPipeline { .. })
    ));
}

#[test]
fn safe_handles_reject_cross_context_composition() {
    use nml_mlir::Operation;

    let first = Context::new();
    let second = Context::new();
    let first_type = first.dtype(DType::F32).unwrap();

    assert!(matches!(
        second.function_type(&[first_type], &[]),
        Err(Error::ContextMismatch { .. })
    ));
    assert!(matches!(
        Operation::builder(&second, "test.cross_context")
            .results(&[first_type])
            .build(),
        Err(Error::ContextMismatch { .. })
    ));

    let first_operation = Operation::builder(&first, "func.return").build().unwrap();
    let mut second_module = second.empty_module().unwrap();
    assert!(matches!(
        second_module.append_operation(first_operation),
        Err(Error::ContextMismatch { .. })
    ));

    let mut first_manager = first.pass_manager();
    assert!(matches!(
        first_manager.run(&mut second_module),
        Err(Error::ContextMismatch { .. })
    ));
}

#[test]
fn ttir_context_owns_only_the_kernel_dialect_universe() {
    const KERNEL: &str = r#"
module {
  tt.func public @copy(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %value = tt.load %arg0 : !tt.ptr<f32>
    tt.store %arg1, %value : !tt.ptr<f32>
    tt.return
  }
}
"#;

    for _ in 0..8 {
        let context = Context::new_ttir();
        let f32_type = context.dtype(DType::F32).unwrap();
        let pointer = context.triton_pointer_type(f32_type, 1).unwrap();
        let descriptor = context
            .triton_tensor_descriptor_type(&[16, 32], f32_type)
            .unwrap();
        assert!(pointer.is_triton_pointer());
        assert_eq!(pointer.text(), "!tt.ptr<f32>");
        assert!(descriptor.is_triton_tensor_descriptor());
        assert_eq!(descriptor.text(), "!tt.tensordesc<16x32xf32>");
        assert_eq!(
            context.triton_program_dimension(0).unwrap().text(),
            "0 : i32"
        );
        assert_eq!(context.triton_cache_modifier(3).unwrap().text(), "3 : i32");
        assert_eq!(context.triton_eviction_policy(3).unwrap().text(), "3 : i32");
        assert_eq!(context.triton_input_precision(2).unwrap().text(), "2 : i32");

        let module = context.parse_module(KERNEL).unwrap();
        module.verify().unwrap();
        let text = module.text();
        assert!(text.contains("tt.func public @copy"), "{text}");
        assert!(text.contains("tt.load"), "{text}");
        assert!(text.contains("tt.store"), "{text}");
    }
}

#[test]
fn ttir_pointer_types_reject_foreign_contexts() {
    let first = Context::new_ttir();
    let second = Context::new_ttir();
    let foreign = first.dtype(DType::F32).unwrap();
    assert!(matches!(
        second.triton_pointer_type(foreign, 1),
        Err(Error::ContextMismatch {
            object: "Triton pointer pointee"
        })
    ));
    assert!(matches!(
        second.triton_tensor_descriptor_type(&[8], foreign),
        Err(Error::ContextMismatch {
            object: "Triton tensor descriptor element"
        })
    ));
    assert!(second.triton_program_dimension(3).is_err());
    assert!(second.triton_cache_modifier(0).is_err());
    assert!(second.triton_eviction_policy(4).is_err());
    assert!(second.triton_input_precision(5).is_err());
}

#[test]
fn triton_custom_call_has_xla_typed_ffi_contract() {
    const KERNEL: &str = r#"module {
  tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %value = tt.load %arg0 : !tt.ptr<f32>
    tt.store %arg1, %value : !tt.ptr<f32>
    tt.return
  }
}"#;

    let context = Context::new();
    let tensor = context.ranked_tensor_type(DType::F32, &[4]).unwrap();
    let mut block = Block::new(&context, &[tensor]).unwrap();
    let input = block.argument(0).unwrap();
    let call = context
        .triton_custom_call(
            &[input],
            &[tensor],
            TritonCustomCall {
                name: "copy",
                ir: KERNEL,
                grid: [1, 1, 1],
                num_stages: 1,
                num_warps: 1,
                operand_layouts: &[&[0]],
                result_layouts: &[&[0]],
                output_operand_aliases: &[OutputOperandAlias {
                    output_index: 0,
                    operand_index: 0,
                }],
            },
        )
        .unwrap();
    let result = call.result(0).unwrap();
    block.append_operation(call).unwrap();
    block
        .append_operation(context.return_operation(&[result]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(block).unwrap();
    let function = context
        .function("triton_custom_call", &[tensor], &[tensor], body)
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();

    let text = module.text();
    assert!(text.contains("__gpu$xla.gpu.triton"), "{text}");
    assert!(text.contains("#stablehlo.output_operand_alias"), "{text}");
    assert!(text.contains("grid_x = 1 : i32"), "{text}");
    assert!(text.contains("operand_layouts"), "{text}");
    assert!(text.contains("result_layouts"), "{text}");
}

#[test]
fn custom_call_layouts_reject_non_tensor_types_safely() {
    let context = Context::new();
    let scalar = context.dtype(DType::F32).unwrap();
    let block = Block::new(&context, &[scalar, scalar, scalar]).unwrap();
    let query = block.argument(0).unwrap();
    let key = block.argument(1).unwrap();
    let value = block.argument(2).unwrap();

    assert!(matches!(
        context.flash_attention_2_custom_call(
            query, key, value, scalar, scalar, 1.0, true, -1,
        ),
        Err(Error::InvalidOperation { source })
            if source == "custom-call layouts require ranked tensor types"
    ));
}

#[test]
fn paged_flash_custom_calls_have_narrow_typed_ffi_contracts() {
    let context = Context::new();
    let query = context
        .ranked_tensor_type(DType::F16, &[2, 3, 4, 64])
        .unwrap();
    let cache = context
        .ranked_tensor_type(DType::F16, &[7, 256, 2, 64])
        .unwrap();
    let page_table = context.ranked_tensor_type(DType::I32, &[2, 5]).unwrap();
    let lengths = context.ranked_tensor_type(DType::I32, &[2]).unwrap();
    let lse = context.ranked_tensor_type(DType::F32, &[2, 4, 3]).unwrap();
    let argument_types = [query, cache, cache, page_table, lengths];
    let mut block = Block::new(&context, &argument_types).unwrap();
    let arguments = (0..argument_types.len())
        .map(|index| block.argument(index).unwrap())
        .collect::<Vec<_>>();

    let fa2 = context
        .paged_flash_attention_2_custom_call(
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
            query,
            lse,
            0.125,
            true,
            32,
        )
        .unwrap();
    assert!(fa2.result(2).is_err());
    let fa2_output = fa2.result(0).unwrap();
    block.append_operation(fa2).unwrap();

    let fa3 = context
        .paged_flash_attention_3_custom_call(
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
            query,
            lse,
            0.125,
            true,
            32,
        )
        .unwrap();
    assert!(fa3.result(2).is_ok());
    let fa3_output = fa3.result(0).unwrap();
    block.append_operation(fa3).unwrap();
    block
        .append_operation(context.return_operation(&[fa2_output, fa3_output]).unwrap())
        .unwrap();
    let mut body = Region::new(&context).unwrap();
    body.append_block(block).unwrap();
    let function = context
        .function(
            "paged_flash_custom_calls",
            &argument_types,
            &[query, query],
            body,
        )
        .unwrap();
    let mut module = context.empty_module().unwrap();
    module.append_operation(function).unwrap();
    module.verify().unwrap();

    let text = module.text();
    assert!(text.contains("nml.flash_attention_2.paged"), "{text}");
    assert!(text.contains("nml.flash_attention_3.paged"), "{text}");
    assert!(text.contains("api_version = 4 : i32"), "{text}");
    assert!(text.contains("backend_config"), "{text}");
    assert!(text.contains("scale = 1.250000e-01 : f32"), "{text}");
    assert!(text.contains("sliding_window = 32 : i32"), "{text}");
    assert!(text.contains("tensor<1xi32>"), "{text}");
}
