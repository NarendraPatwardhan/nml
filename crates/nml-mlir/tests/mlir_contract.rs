use nml_mlir::{
    Block, Context, Error, IndexCastKind, Region, ShardyDimension, stablehlo_current_version,
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
    assert!(
        !DType::ALL
            .iter()
            .any(|dtype| dtype.stablehlo_spelling() == "index")
    );
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
