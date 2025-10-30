// RUN: bishengir-opt --hfusion-normalize-ops %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_hfusion_compare_neq_ops
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi1>)
// CHECK: %[[arg2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[arg3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<1024xi64>) outs(%[[arg2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[arg4:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[arg6:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[arg5:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg3]], %[[cst:.*]] : tensor<1024xf32>, f32) outs(%[[arg6]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[arg7:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg5]] : tensor<1024xi1>) outs(%[[arg4]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[arg7]]
func.func @test_hfusion_compare_neq_ops(
  %src1 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty() : tensor<1024xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %1 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----

// CHECK-LABEL: func.func @test_linalg_negf_mul
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[CST:.*]]: f32
// CHECK: %[[ZERO:.*]] : tensor<5x1xf32
// CHECK: %[[ONE:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0:.*]], %[[CST:.*]]: tensor<5x1xf32>, f32) outs(%[[ZERO:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[ONE:.*]]
func.func @test_linalg_negf_mul(%src: tensor<5x1xf32>) -> tensor<5x1xf32> {
  %x = tensor.empty() : tensor<5x1xf32>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<negf>} ins(%src : tensor<5x1xf32>) outs(%x : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_linalg_div_to_hfusion_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>)
// CHECK: %[[cast0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<5x1xf16>)
// CHECK: %[[rec:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[cast0:.*]]: tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[rec]] : tensor<5x1xf32>) outs({{.*}} : tensor<5x1xf16>)
// CHECK: return %[[cast1]]
func.func @test_linalg_div_to_hfusion_rec(%src: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %src : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %1 : tensor<5x1xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_rsqrt_to_hfusion_sqrt
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf32>)
// CHECK: %[[ONE:.*]]: tensor<5x1xf32>
// CHECK: %[[TWO:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<5x1xf32>) outs(%[[ONE:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[THREE:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[TWO:.*]]: tensor<5x1xf32>) outs(%[[ZERO:.*]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: return %[[THREE:.*]]
func.func @test_hfusion_rsqrt_to_hfusion_sqrt(%arg0: tensor<5x1xf32>) -> tensor<5x1xf32> {
    %0 = tensor.empty() : tensor<5x1xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
    return %1 : tensor<5x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_rsqrt_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf16>)
// CHECK: %[[EMPTY0:.*]]: tensor<16xf16>
// CHECK: %[[EMPTY1:.*]]: tensor<16xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<16xf16>) outs(%[[EMPTY1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY2:.*]]: tensor<16xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[EMPTY0:.*]] : tensor<16xf16>) outs(%[[EMPTY2:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY3:.*]]: tensor<16xf32>
// CHECK: %[[SQRT0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[CAST0:.*]]: tensor<16xf32>) outs(%[[EMPTY3:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[REC0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[SQRT0:.*]]: tensor<16xf32>) outs(%[[CAST1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[EMPTY4:.*]]: tensor<16xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[REC0:.*]] : tensor<16xf32>) outs(%[[EMPTY4:.*]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[REC0:.*]] : tensor<16xf16>
func.func @test_hfusion_rsqrt_f16(%arg0: tensor<16xf16>) -> tensor<16xf16> {
    %0 = tensor.empty() : tensor<16xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf16>) -> tensor<16xf16>
    return %1 : tensor<16xf16>
}

// -----

// CHECK: func.func @test_hfusion_rsqrt_to_hfusion_sqrt_dynshape(%[[ARG0:.*]]: tensor<5x?xf32>, %[[ARG1:.*]]: index)
// CHECK: %[[ONE:.*]]: tensor<5x?xf32>
// CHECK: %[[TWO:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<5x?xf32>) outs(%[[ONE:.*]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: %[[THREE:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%[[TWO:.*]]: tensor<5x?xf32>) outs(%[[ZERO:.*]] : tensor<5x?xf32>) -> tensor<5x?xf32>
// CHECK: return %[[THREE:.*]]
func.func @test_hfusion_rsqrt_to_hfusion_sqrt_dynshape(%s: tensor<5x?xf32>, %d : index) -> tensor<5x?xf32> {
    %0 = tensor.empty(%d) : tensor<5x?xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rsqrt>} ins(%s : tensor<5x?xf32>) outs(%0 : tensor<5x?xf32>) -> tensor<5x?xf32>
    return %1 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_normalize_rec_i16_to_f32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xi16>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi32>)
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<1x2xi32>) outs({{.*}} : tensor<1x2xi16>)
func.func @test_normalize_rec_i16_to_f32(%arg0 : tensor<1x2xi16>) -> tensor<1x2xi16> {
    %0 = tensor.empty() : tensor<1x2xi16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi16>) outs(%0 : tensor<1x2xi16>) -> tensor<1x2xi16>
    return %1 : tensor<1x2xi16>
}

// -----

// CHECK-LABEL: func.func @test_normalize_rec_i32_to_f32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xi32>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi32>)
func.func @test_normalize_rec_i32_to_f32(%arg0 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = tensor.empty() : tensor<1x2xi32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi32>) outs(%0 : tensor<1x2xi32>) -> tensor<1x2xi32>
    return %1 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func.func @test_normalize_rec_i64_to_f32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xi64>) outs({{.*}} : tensor<1x2xf32>)
// CHECK: hfusion.elemwise_unary
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<1x2xf32>) outs({{.*}} : tensor<1x2xi64>)
func.func @test_normalize_rec_i64_to_f32(%arg0 : tensor<1x2xi64>) -> tensor<1x2xi64> {
    %0 = tensor.empty() : tensor<1x2xi64>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg0 : tensor<1x2xi64>) outs(%0 : tensor<1x2xi64>) -> tensor<1x2xi64>
    return %1 : tensor<1x2xi64>
}

// -----

// CHECK-LABEL: func.func @test_linalg_floor_to_hfusion_cast
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf16>)
// CHECK: %[[DST0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[DST1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<1024xf16>) outs(%[[DST1]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DST2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[RES0]] : tensor<1024xf32>) outs(%[[DST2]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[RES1]] : tensor<1024xf32>) outs(%[[DST0]] : tensor<1024xf16>) -> tensor<1024xf16>
// CHECK: return %[[RES2]]
func.func @test_linalg_floor_to_hfusion_cast(%src: tensor<1024xf16>) -> tensor<1024xf16> {
    %dst = tensor.empty() : tensor<1024xf16>
    %res = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>} ins(%src : tensor<1024xf16>) outs(%dst : tensor<1024xf16>) -> tensor<1024xf16>
   return %res : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_mod
// CHECK-SAME: (%[[SRC0:.*]]: tensor<2048xi32>, %[[SRC1:.*]]: tensor<2048xi32>)
// CHECK: %[[C1:.*]] = arith.constant -1 : i32
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[LHS:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[LHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[SRC0]] : tensor<2048xi32>) outs(%[[LHS]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[RHS:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[RHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[SRC1]] : tensor<2048xi32>) outs(%[[RHS]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[RESFP:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[DIVFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSFP]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[RESFP]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[ARG2:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[ARG3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[DIVFP]] : tensor<2048xf32>) outs(%[[ARG2]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[ARG4:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[ARG5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG3]], %[[RHSFP]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[ARG4]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[ARG6:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[ARG7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[LHSFP]], %[[ARG5]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[ARG6]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[ARG8:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[ARG9:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG7]], %[[RHSFP]] : tensor<2048xf32>, tensor<2048xf32>) outs(%[[ARG8]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[ARG10:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[ARG11:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG7]], %[[CST]] : tensor<2048xf32>, f32) outs(%[[ARG10]] : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[ARG12:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[ARG13:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%[[DIVFP]], %[[CST]] : tensor<2048xf32>, f32) outs(%[[ARG12]] : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[ARG14:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[ARG15:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[ARG11]], %[[ARG13]] : tensor<2048xi1>, tensor<2048xi1>) outs(%[[ARG14]] : tensor<2048xi1>) -> tensor<2048xi1>
// CHECK: %[[ARG16:.*]] = tensor.empty() : tensor<2048xf32>
// CHECK: %[[ARG17:.*]] = hfusion.select ins(%[[ARG15]], %[[ARG7]], %[[ARG9]] : tensor<2048xi1>, tensor<2048xf32>, tensor<2048xf32>) outs(%[[ARG16]] : tensor<2048xf32>) -> tensor<2048xf32>
// CHECK: %[[ARG18:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[ARG19:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[ARG17]] : tensor<2048xf32>) outs(%[[ARG18]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK: %[[ARG20:.*]] = tensor.empty() : tensor<2048xi1>
// CHECK: %[[ARG21:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[ARG22:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[ARG23:.*]] = linalg.fill
// CHECK: %[[ARG24:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK: %[[ARG25:.*]] = hfusion.select 
// CHECK: return %[[ARG25]] : tensor<2048xi32>
func.func @test_hfusion_mod(%src0: tensor<2048xi32>, %src1: tensor<2048xi32>) -> tensor<2048xi32> {
    %3 = tensor.empty() : tensor<2048xi32>
    %4 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%src0, %src1 : tensor<2048xi32>, tensor<2048xi32>) outs(%3 : tensor<2048xi32>) -> tensor<2048xi32>
    return %4 : tensor<2048xi32>
}

// -----

// CHECK-LABEL: func.func @test_linalg_divi_to_divf
func.func @test_linalg_divi_to_divf(%arg0: tensor<48xi32>, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[LHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSFP]] : tensor<48xf32>, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----


// CHECK-LABEL: func.func @test_linalg_divi_to_divf_vs
func.func @test_linalg_divi_to_divf_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[LHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RHSCASTED:.*]] = arith.sitofp %arg1 : i32 to f32
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSCASTED]] : tensor<48xf32>, f32) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_linalg_divi_to_divf_sv
func.func @test_linalg_divi_to_divf_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHSCASTED:.*]] = arith.sitofp %arg0 : i32 to f32
    // CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSCASTED]], %[[RHSFP]] : f32, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----

// CHECK-LABEL: func.func @test_cast_op_tensor_i64_to_f16
// CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<23xf16>
// CHECK: %[[ONE:.*]] = tensor.empty() : tensor<23xf32>
// CHECK: %[[TWO:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg0:.*]] : tensor<23xi64>) outs(%[[ONE:.*]]: tensor<23xf32>) -> tensor<23xf32>
// CHECK: %[[THREE:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[TWO:.*]] : tensor<23xf32>) outs(%[[ZERO:.*]] : tensor<23xf16>) -> tensor<23xf16>
func.func @test_cast_op_tensor_i64_to_f16(%arg0: tensor<23xi64>, %arg1: tensor<f16>) -> tensor<23xf16> attributes {hacc.entry} {
    %cst = arith.constant 0.86956521739130432 : f64
    %0 = tensor.empty() : tensor<23xf16>
    %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<23xi64>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<f16>) outs(%0 : tensor<23xf16>) dimensions = [0]
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %broadcasted : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %3 = arith.truncf %cst : f64 to f16
    %4 = linalg.fill ins(%3 : f16) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %4 : tensor<23xf16>, tensor<23xf16>) outs(%0 : tensor<23xf16>) -> tensor<23xf16>
    return %5 : tensor<23xf16>
  }


// -----
// CHECK-LABEL: func.func @test_isinf
// CHECK: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NEGONE:.*]] = arith.constant -1 : i32
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NEGINF:.*]] = arith.constant -2139095040 : i32
// CHECK: %[[MASKVAL:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[MASKRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i32) outs(%[[MASKRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTEMPTY:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<8192xi32>, tensor<8192xi32>) outs(%[[VANDOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<8192xi32>, i32) outs(%[[VADDRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTADD:.*]] = hfusion.bitcast ins(%[[VADDOP]] : tensor<8192xi32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[VABSOP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[BITCASTADD]] : tensor<8192xf32>) outs(%[[BITCASTOUTPUT:.*]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[BITCASTABS:.*]] = hfusion.bitcast ins(%[[VABSOP]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[BITCASTABS]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[BITCASTOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMULOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VMINOP]], %[[NEGONE]] : tensor<8192xi32>, i32) outs(%[[VMINOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDOP1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VMULOP]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[VMULOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[TMPF:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[CASTF:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VADDOP1]] : tensor<8192xi32>) outs(%[[TMPF]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[OUT2:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0]] : tensor<8192xf32>, f32) outs(%[[OUT2]] : tensor<8192xi1>) -> tensor<8192xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<8192xi1>) outs(%[[OUT1]] : tensor<8192xi1>) -> tensor<8192xi1>
func.func @test_isinf() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xf32>
  %2 = hfusion.isinf %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}
// -----

// CHECK-LABEL: @lowering_cast_i64_to_bf16(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @lowering_cast_i64_to_bf16(%arg0: tensor<4x4xi64>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}
// -----

// CHECK-LABEL: @test_cast_i8_to_bf16
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xbf16>) -> tensor<4x4xbf16>
func.func @test_cast_i8_to_bf16(%arg0: tensor<4x4xi8>) -> tensor<4x4xbf16> {
  %0 = tensor.empty() : tensor<4x4xbf16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}
// -----

// CHECK-LABEL: func.func @test_cast_bf16_to_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xbf16>)
// CHECK: %[[arg1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[arg2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0]] : tensor<16xbf16>) outs(%[[arg1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[arg3:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[arg5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2]], %[[cst:.*]] : tensor<16xf32>, f32) outs(%[[arg5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4]] : tensor<16xi1>) outs(%[[arg3]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: return %[[arg6]]
func.func @test_cast_bf16_to_i1(%arg0: tensor<16xbf16>) -> tensor<16xi1> {
  %0 = tensor.empty() : tensor<16xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<16xbf16>) outs(%0 : tensor<16xi1>) -> tensor<16xi1>
  return %1 : tensor<16xi1>
}
// -----

// CHECK-LABEL: func.func @test_cast_f32_to_i1
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2:.*]], %[[cst:.*]] : tensor<2x256x12x257xf32>, f32) outs(%[[arg5:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// ChECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4:.*]] : tensor<2x256x12x257xi1>) outs(%[[arg3:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
func.func @test_cast_f32_to_i1(%arg0: tensor<2x256x12x257xf32>) -> tensor<2x256x12x257xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<2x256x12x257xi1>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x256x12x257xf32>) outs(%1 : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
  return %2 : tensor<2x256x12x257xi1>
}
// -----

// CHECK-LABEL: func.func @test_cast_f16_to_i1
// CHECK: %[[arg4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg2:.*]], %[[cst:.*]] : tensor<2x256x12x257xf16>, f16) outs(%[[arg5:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
// ChECK: %[[arg6:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[arg4:.*]] : tensor<2x256x12x257xi1>) outs(%[[arg3:.*]] : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
func.func @test_cast_f16_to_i1(%arg0: tensor<2x256x12x257xf16>) -> tensor<2x256x12x257xi1> attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %1 = tensor.empty() : tensor<2x256x12x257xi1>
  %2 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<2x256x12x257xf16>) outs(%1 : tensor<2x256x12x257xi1>) -> tensor<2x256x12x257xi1>
  return %2 : tensor<2x256x12x257xi1>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_elemwise_unary_absi(%dst: tensor<1xi32>) -> tensor<1xi32> {
  %src = arith.constant dense<5> : tensor<1xi32>
  // CHECK: %c5_i32 = arith.constant 5 : i32
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%c5_i32 : i32) outs(%arg0 : tensor<1xi32>) -> tensor<1xi32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%src : tensor<1xi32>) outs(%dst : tensor<1xi32>) -> tensor<1xi32>

  return %1 : tensor<1xi32>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_elemwise_unary_absi_rank0(%dst: tensor<i32>) -> tensor<i32> {
  %src = arith.constant dense<5> : tensor<i32>
  // CHECK: %c5_i32 = arith.constant 5 : i32
  // CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%c5_i32 : i32) outs(%arg0 : tensor<i32>) -> tensor<i32>
  %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<absi>} ins(%src : tensor<i32>) outs(%dst : tensor<i32>) -> tensor<i32>

  return %1 : tensor<i32>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_elemwise_binary(%src: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: %cst = arith.constant 5.000000e-01 : f32
  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  // CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %arg0 : f32, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%cst, %src : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_compare(%src: tensor<1xi64>) -> tensor<1xi1> {
  // CHECK: %c5_i64 = arith.constant 5 : i64
  // CHECK: %0 = tensor.empty() : tensor<1xi1>
  // CHECK: %1 = tensor.empty() : tensor<1xi1>
  // CHECK: %2 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %c5_i64 : tensor<1xi64>, i64) outs(%1 : tensor<1xi1>) -> tensor<1xi1>
  // CHECK: %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%2 : tensor<1xi1>) outs(%0 : tensor<1xi1>) -> tensor<1xi1>

  %cst = arith.constant dense<5> : tensor<1xi64>
  %dst = tensor.empty() : tensor<1xi1>

  %res = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%src, %cst : tensor<1xi64>, tensor<1xi64>)
    outs(%dst : tensor<1xi1>) -> tensor<1xi1>

  return %res : tensor<1xi1>
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_select(
  %src1 : memref<1xi1>, %src3 : memref<1xi32>, %dst : memref<1xi32>) {
  // CHECK: %c5_i32 = arith.constant 5 : i32
  // CHECK: hfusion.select ins(%arg0, %c5_i32, %arg1 : memref<1xi1>, i32, memref<1xi32>) outs(%arg2 : memref<1xi32>)

  %src2 = arith.constant dense<5> : memref<1xi32>

  hfusion.select
    ins(%src1, %src2, %src3 : memref<1xi1>, memref<1xi32>, memref<1xi32>)
    outs(%dst : memref<1xi32>)
  return
}

// -----
func.func @scalar_like_tensor_conversion_hfusion_cast() -> tensor<f16> {
    // CHECK: %cst = arith.constant 0.869565188 : f32
    // CHECK: %0 = tensor.empty() : tensor<f16>
    // CHECK: %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%cst : f32) outs(%0 : tensor<f16>) -> tensor<f16>

    %src = arith.constant dense<0.86956521739130432> : tensor<f32>
    %dst = tensor.empty() : tensor<f16>

    %res = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%src : tensor<f32>) outs(%dst : tensor<f16>) -> tensor<f16>
    return %res : tensor<f16>
}

// -----
func.func @scalar_like_tensor_conversion_linalg_elemwise_unary_abs(%output : tensor<f32>) -> tensor<f32> {
  // CHECK: %cst = arith.constant 5.100000e+00 : f32
  // CHECK: %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%cst : f32) outs(%arg0 : tensor<f32>) -> tensor<f32>
  %src = arith.constant dense<5.1> : tensor<f32>
  %0 = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
                              ins(%src: tensor<f32>) outs(%output: tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>
}

// -----
func.func @scalar_like_tensor_conversion_linalg_elemwise_binary(%src: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: %cst = arith.constant 5.000000e-01 : f32
  // CHECK: %0 = tensor.empty() : tensor<1xf32>
  // CHECK: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %arg0 : f32, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>

  %cst = arith.constant dense<0.5> : tensor<1xf32>
  %dst = tensor.empty() : tensor<1xf32>
  %res = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%cst, %src : tensor<1xf32>, tensor<1xf32>) outs(%dst : tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----
func.func @scalar_like_tensor_conversion_xori(%arg0: tensor<i8>) -> tensor<i8> {
  %c127_i8 = arith.constant 127 : i8
  %0 = tensor.empty() : tensor<i8>
  %2 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%c127_i8, %arg0 : i8, tensor<i8>) outs(%0 : tensor<i8>) -> tensor<i8>
  return %2 : tensor<i8>
}

// CHECK-LABEL:   func.func @scalar_like_tensor_conversion_xori(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<i8>) -> tensor<i8> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 127 : i8
// CHECK:           %[[VAL_2:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vor>} ins(%[[VAL_1]], %[[VAL_0]] : i8, tensor<i8>) outs(%[[VAL_2]] : tensor<i8>) -> tensor<i8>
// CHECK:           %[[VAL_4:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_5:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vand>} ins(%[[VAL_1]], %[[VAL_0]] : i8, tensor<i8>) outs(%[[VAL_4]] : tensor<i8>) -> tensor<i8>
// CHECK:           %[[VAL_6:.*]] = hfusion.elemwise_unary {fun = {{.*}}<vnot>} ins(%[[VAL_5]] : tensor<i8>) outs(%[[VAL_5]] : tensor<i8>) -> tensor<i8>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<i8>
// CHECK:           %[[VAL_8:.*]] = hfusion.elemwise_binary {fun = {{.*}}<vand>} ins(%[[VAL_6]], %[[VAL_3]] : tensor<i8>, tensor<i8>) outs(%[[VAL_7]] : tensor<i8>) -> tensor<i8>
// CHECK:           return %[[VAL_8]] : tensor<i8>
// CHECK:         }

// -----
func.func @scalar_like_tensor_conversion_linalg_brc(%init: tensor<4x1xf32>) -> tensor<4x1xf32> {
  // CHECK: %cst = arith.constant 5.000000e-01 : f32
  // CHECK: %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<4x1xf32>) -> tensor<4x1xf32>
  %input = arith.constant dense<0.5> : tensor<1xf32>
  %res = linalg.broadcast
      ins(%input: tensor<1xf32>)
      outs(%init: tensor<4x1xf32>)
      dimensions = [0]
  func.return %res : tensor<4x1xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log2
// CHECK: %[[CSTTWO:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_log2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log10
// CHECK: %[[CSTTEN:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTEN:.*]] : f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_log10(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log10>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log2_f16
// CHECK: %[[CSTTWO:.*]] : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<1024xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[CAST_RES1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0:.*]] : tensor<1024xf16>) outs(%[[EMPTY1:.*]]  : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[LOG_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[LOG_RES2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILL:.*]] : tensor<1024xf32>) outs(%[[EMPTY2:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DIV_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG_RES1:.*]], %[[LOG_RES2:.*]]: tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY3:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[CAST_RES2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY0:.*]]  : tensor<1024xf16>) -> tensor<1024xf16>
func.func @test_hfusion_elemwise_unary_log2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log2>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_log1p
// CHECK: %[[CSTONE:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[ADD_RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[ARG0:.*]],  %[[CSTONE:.*]] : tensor<1024xf32>, f32) outs(%[[EMPTY0:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[ADD_RES:.*]] : tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_log1p(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<log1p>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_exp2
// CHECK: %[[CSTLNTWO:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MUL_RES1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0:.*]],  %[[CSTLNTWO:.*]]: tensor<1024xf32>, f32) outs(%[[EMPTYO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL_RES1:.*]] : tensor<1024xf32>) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_exp2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_exp2_f16
// CHECK: %[[cast0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins({{.*}} : tensor<1024xf16>) outs({{.*}} : tensor<1024xf32>)
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast0]]
// CHECK: %[[exp:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul]] : tensor<1024xf32>)
// CHECK: %[[cast1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[exp]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>)
func.func @test_hfusion_elemwise_unary_exp2_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<exp2>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_expm1
// CHECK: %[[CSTLNTWO:.*]] : f32
// CHECK: %[[EMPTYO:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[EXP_RES1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[ARG0:.*]] : tensor<1024xf32>) outs(%[[EMPTYO:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[RES:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[EXP_RES1:.*]], %[[CSTLNTWO:.*]] : tensor<1024xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_unary_expm1(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_unary_expm1_f16
// CHECK: %[[cast0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins({{.*}} : tensor<1024xf16>) outs({{.*}} : tensor<1024xf32>)
// CHECK: %[[exp:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[cast0:.*]] : tensor<1024xf32>)
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[exp]]
// CHECK: %[[cast1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[sub]] : tensor<1024xf32>) outs({{.*}} : tensor<1024xf16>)
func.func @test_hfusion_elemwise_unary_expm1_f16(%arg0: tensor<1024xf16>) -> tensor<1024xf16> {
    %0 = tensor.empty() : tensor<1024xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<expm1>} ins(%arg0 : tensor<1024xf16>) outs(%0 : tensor<1024xf16>) -> tensor<1024xf16>
    return %1 : tensor<1024xf16>
}

// -----

// CHECK-LABEL: func.func @test_linalg_mul_div_by_one
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>, %[[arg1:.*]]: tensor<5x1xf16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg1]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>) outs(%[[empty]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[res]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
func.func @test_linalg_mul_div_by_one(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%cst, %arg0 : f16, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----
// CHECK-LABEL: func.func @test_linalg_mul_div_by_one_rec
// CHECK-SAME: (%[[arg0:.*]]: tensor<5x1xf16>, %[[arg1:.*]]: tensor<5x1xf16>)
// CHECK: %[[res:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[arg1]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[res]], %[[arg0]] : tensor<5x1xf16>, tensor<5x1xf16>)
func.func @test_linalg_mul_div_by_one_rec(%arg0: tensor<5x1xf16>, %arg1: tensor<5x1xf16>) -> tensor<5x1xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<5x1xf16>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %arg1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%2, %1 : tensor<5x1xf16>, tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
    return %3 : tensor<5x1xf16>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i8_hfusion_compare
// CHECK: %[[CAST0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi8>)
// CHECK: %[[CAST1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi8>)
// CHECK: %[[Veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CAST0]], %[[CAST1]] : tensor<16xf16>, tensor<16xf16>)
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq]] : tensor<16xi1>)
func.func @test_normalize_i8_hfusion_compare(%arg0: tensor<16xi8>, %arg1: tensor<16xi8>) -> tensor<16xi1> {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg1 : tensor<16xi8>, tensor<16xi8>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  return %res1 : tensor<16xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i32_hfusion_compare
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG0:.*]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: %[[Veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[ARG1:.*]], %[[ARG1:.*]] : tensor<16xi32>, tensor<16xi32>)
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq]] : tensor<16xi1>)
func.func @test_normalize_i32_hfusion_compare(%arg0: tensor<16xi32>, %arg1: tensor<16xi32>) -> (tensor<16xi1>, tensor<16xi1>) {
  %dst1 = tensor.empty() : tensor<16xi1>
  %dst2 = tensor.empty() : tensor<16xi1>
  %res1 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
    ins(%arg0, %arg1 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst1 : tensor<16xi1>) -> tensor<16xi1>
  %res2 = hfusion.compare {compare_fn = #hfusion.compare_fn<vne>}
    ins(%arg0, %arg0 : tensor<16xi32>, tensor<16xi32>)
    outs(%dst2 : tensor<16xi1>) -> tensor<16xi1>
  return %res1, %res2 : tensor<16xi1>, tensor<16xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i32_hfusion_compare_dynamic(
// CHECK: hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%[[ARG0:.*]], %[[ARG1:.*]] : tensor<?x?xi32>, tensor<?x?xi32>)
func.func @test_normalize_i32_hfusion_compare_dynamic(%arg0: tensor<?x?xi32>) -> (tensor<?x?xi32>, tensor<?x?xi1>) attributes {OperatorType = "Default", compute_capability = "", frontend_symbol = {input_0 = ["s93", "s94"], output_0 = ["s93", "s94"], output_1 = ["s93", "s94"]}, hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32319_i32 = arith.constant 32319 : i32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%arg0, %1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%0 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %3 = tensor.empty(%dim, %dim_0) : tensor<?x?xi32>
  %4 = linalg.fill ins(%c32319_i32 : i32) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%2, %4 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%3 : tensor<?x?xi32>) -> tensor<?x?xi32>
  %6 = tensor.empty(%dim, %dim_0) : tensor<?x?xi1>
  %7 = hfusion.compare {fun = #hfusion.compare_fn<veq>} ins(%arg0, %5 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%6 : tensor<?x?xi1>) -> tensor<?x?xi1>
  return %5, %7 : tensor<?x?xi32>, tensor<?x?xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_i8_to_f32
// CHECK: %[[cast0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1xi8>) outs({{.*}} : tensor<1xf16>)
// CHECK: %[[cast1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[cast0]] : tensor<1xf16>) outs({{.*}} : tensor<1xf32>)
// CHECK: %[[cast2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<1xi8>) outs({{.*}} : tensor<1xf16>)
// CHECK: %[[cast3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[cast2]] : tensor<1xf16>) outs({{.*}} : tensor<1xf32>)
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[cast1]], %[[cast3]] : tensor<1xf32>, tensor<1xf32>)
func.func @test_normalize_i8_to_f32(%arg0: memref<?xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) {
  %0 = tensor.empty() : tensor<1xf32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<1xi8>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg2 : tensor<1xi8>) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4 = tensor.empty() : tensor<1xf32>
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%1, %3 : tensor<1xf32>, tensor<1xf32>) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
  %6 = tensor.empty() : tensor<1xi8>
  %7 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%5 : tensor<1xf32>) outs(%6 : tensor<1xi8>) -> tensor<1xi8>
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
  bufferization.materialize_in_destination %7 in writable %reinterpret_cast : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
  return
}

// -----
// CHECK-LABEL: func.func @test_xori
// CHECK-SAME: (%[[arg0:.*]]: tensor<512xi16>, %[[arg1:.*]]: tensor<512xi16>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[vand:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[arg0]], %[[arg1]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty1]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[vnot:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[vand]] : tensor<512xi16>) outs(%[[vand]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<512xi16>
// CHECK: %[[xor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[vnot]], %[[vor]] : tensor<512xi16>, tensor<512xi16>) outs(%[[empty2]] : tensor<512xi16>) -> tensor<512xi16>
// CHECK: return %[[xor]] : tensor<512xi16>
func.func @test_xori(%arg0: tensor<512xi16>, %arg1: tensor<512xi16>) -> tensor<512xi16> {
    %0 = tensor.empty() : tensor<512xi16>
    %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vxor>} ins(%arg0, %arg1 : tensor<512xi16>, tensor<512xi16>) outs(%0 : tensor<512xi16>) -> tensor<512xi16>
    return %1 : tensor<512xi16>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_sin_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5x1xf32>) -> tensor<5x1xf32> {
// CHECK: %[[VAL_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[VAL_2:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[VAL_3:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL_5:.*]] = arith.constant -0.166666672 : f32
// CHECK: %[[VAL_6:.*]] = arith.constant 0.00833333377 : f32
// CHECK: %[[VAL_7:.*]] = arith.constant -1.98412701E-4 : f32
// CHECK: %[[VAL_8:.*]] = arith.constant 2.75573188E-6 : f32
// CHECK: %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAL_10:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[VAL_11:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[VAL_12:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[VAL_13:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[VAL_14:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[VAL_15:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_0]], %[[VAL_15]] : tensor<5x1xf32>, f32) outs(%[[VAL_16]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_18:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_19:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_17]] : tensor<5x1xf32>) outs(%[[VAL_18]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_21:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_14]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_0]], %[[VAL_21]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_23:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_13]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_22]], %[[VAL_23]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_12]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_24]], %[[VAL_25]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_11]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_26]], %[[VAL_27]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_10]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_28]], %[[VAL_29]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_30]], %[[VAL_9]] : tensor<5x1xf32>, f32) outs(%[[VAL_20]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_32:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_31]], %[[VAL_31]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_33]], %[[VAL_8]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_34]], %[[VAL_7]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_35]], %[[VAL_33]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_36]], %[[VAL_6]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_37]], %[[VAL_33]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_38]], %[[VAL_5]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_39]], %[[VAL_33]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_40]], %[[VAL_4]] : tensor<5x1xf32>, f32) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_41]], %[[VAL_31]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_32]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_43:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_3]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_45:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_46:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[VAL_44]] : tensor<5x1xf32>) outs(%[[VAL_45]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_46]], %[[VAL_2]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_19]], %[[VAL_1]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_49:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_47]], %[[VAL_48]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_49]], %[[VAL_4]] : tensor<5x1xf32>, f32) outs(%[[VAL_43]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_50]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_16]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK  return %[[VAL_51]] : tensor<5x1xf32>
// CHECK: }
func.func @test_hfusion_sin_ops(%arg0 : tensor<5x1xf32>) ->  tensor<5x1xf32> {
  %0 = tensor.empty() : tensor<5x1xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sin>} ins(%arg0 : tensor<5x1xf32>) outs(%0 : tensor<5x1xf32>) -> tensor<5x1xf32>
  return %ret : tensor<5x1xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_cos_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5x1xf16>) -> tensor<5x1xf16> {
// CHECK: %[[VAL_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[VAL_2:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAL_4:.*]] = arith.constant -0.166666672 : f32
// CHECK: %[[VAL_5:.*]] = arith.constant 0.00833333377 : f32
// CHECK: %[[VAL_6:.*]] = arith.constant -1.98412701E-4 : f32
// CHECK: %[[VAL_7:.*]] = arith.constant 2.75573188E-6 : f32
// CHECK: %[[VAL_8:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[VAL_9:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[VAL_10:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[VAL_11:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[VAL_12:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[VAL_13:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[VAL_14:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAL_15:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_17:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_0]] : tensor<5x1xf16>) outs(%[[VAL_16]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_18:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_17]], %[[VAL_15]] : tensor<5x1xf32>, f32) outs(%[[VAL_18]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_19]], %[[VAL_14]] : tensor<5x1xf32>, f32) outs(%[[VAL_18]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_21:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_22:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_20]] : tensor<5x1xf32>) outs(%[[VAL_21]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_13]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_17]], %[[VAL_24]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_12]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_25]], %[[VAL_26]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_11]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_27]], %[[VAL_28]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_10]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_29]], %[[VAL_30]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_9]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_31]], %[[VAL_32]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_33]], %[[VAL_8]] : tensor<5x1xf32>, f32) outs(%[[VAL_23]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_35:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_34]], %[[VAL_34]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_36]], %[[VAL_7]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_37]], %[[VAL_6]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_38]], %[[VAL_36]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_39]], %[[VAL_5]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_40]], %[[VAL_36]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[VAL_4]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_36]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[VAL_3]] : tensor<5x1xf32>, f32) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_44]], %[[VAL_34]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_35]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_46:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_14]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_48:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_49:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[VAL_47]] : tensor<5x1xf32>) outs(%[[VAL_48]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_49]], %[[VAL_2]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_22]], %[[VAL_1]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_50]], %[[VAL_51]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_52]], %[[VAL_3]] : tensor<5x1xf32>, f32) outs(%[[VAL_46]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_54:.*]] = tensor.empty() : tensor<5x1xf32>
// CHECK: %[[VAL_55:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_45]], %[[VAL_53]] : tensor<5x1xf32>, tensor<5x1xf32>) outs(%[[VAL_54]] : tensor<5x1xf32>) -> tensor<5x1xf32>
// CHECK: %[[VAL_56:.*]] = tensor.empty() : tensor<5x1xf16>
// CHECK: %[[VAL_57:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_55]] : tensor<5x1xf32>) outs(%[[VAL_56]] : tensor<5x1xf16>) -> tensor<5x1xf16>
// CHECK: return %[[VAL_57]] : tensor<5x1xf16>
// CHECK: }

func.func @test_hfusion_cos_ops(%arg0 : tensor<5x1xf16>) ->  tensor<5x1xf16> {
  %0 = tensor.empty() : tensor<5x1xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<cos>} ins(%arg0 : tensor<5x1xf16>) outs(%0 : tensor<5x1xf16>) -> tensor<5x1xf16>
  return %ret : tensor<5x1xf16>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_atan_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CONST_0:.*]] = arith.constant 2.16840434E-19 : f32
// CHECK: %[[CONST_1:.*]] = arith.constant 4.61168602E+18 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.785398185 : f32
// CHECK: %[[NEGONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 0.392699093 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 0.414213568 : f32
// CHECK: %[[CST_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_4:.*]] = arith.constant -0.333333343 : f32
// CHECK: %[[CST_5:.*]] = arith.constant 2.000000e-01 : f32
// CHECK: %[[CST_6:.*]] = arith.constant -0.142857149 : f32
// CHECK: %[[CST_7:.*]] = arith.constant 0.111111112 : f32
// CHECK: %[[CST_8:.*]] = arith.constant -0.0909090936 : f32
// CHECK: %[[CST_9:.*]] = arith.constant 0.0769230798 : f32
// CHECK: %[[LOWER_BOUND:.*]] = arith.constant -1.000000e+04 : f32
// CHECK: %[[UPPER_BOUND:.*]] = arith.constant 1.000000e+04 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_0]], %[[UPPER_BOUND]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_2]], %[[LOWER_BOUND]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_3]] : tensor<32xf32>) outs(%[[EMPTY1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_4]], %[[VAL_4]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_6]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_7]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_8]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_9]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_10]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_11]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_12]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_13]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_14]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_16]], %[[VAL_6]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_17]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_18]], %[[VAL_4]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_5]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_20:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_21:.*]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[VAL_20]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_22:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_23:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_4]], %[[VAL_21]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_22]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_23]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_22]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_25:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_4]], %[[VAL_21]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_26]], %[[VAL_24]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_28:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_27]] : tensor<32xf32>) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_29:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_28]], %[[VAL_28]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_30]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_31]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_33:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_32]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_33]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_34]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_35]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_36]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_37]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_38]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_40:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_39]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_40]], %[[VAL_30]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_42:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_42]], %[[VAL_28]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_29]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_44:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_25]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_45:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_19]], %[[VAL_44]] : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[VAL_46:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_4]], %[[NEGONE]] : tensor<32xf32>, f32)
// CHECK: %[[VAL_48:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_4]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_48]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_52:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_53:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_47]], %[[VAL_51]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_52]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_53]] : tensor<32xf32>) outs(%[[VAL_52]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_55:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_56:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54]], %[[VAL_54]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_57:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_56]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_58:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_57]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_59:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_58]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_60:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_59]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_61:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_60]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_62:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_61]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_63:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_62]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_64:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_63]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_65:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_64]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_66:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_65]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_67:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_66]], %[[VAL_56]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_68:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_67]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_69:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_68]], %[[VAL_54]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_55]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_70:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_71:.*]] = linalg.fill ins(%[[CST_2]] : f32) outs(%[[VAL_70]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_72:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_73:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_54]], %[[VAL_71]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_72]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_74:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_73]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_72]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_75:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_76:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_54]], %[[VAL_71]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_77:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_76]], %[[VAL_74]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_78:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_77]] : tensor<32xf32>) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_79:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_80:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_78]], %[[VAL_78]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_81:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_80]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_82:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_81]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_83:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_82]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_84:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_83]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_85:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_84]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_86:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_85]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_87:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_86]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_88:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_87]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_89:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_88]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_90:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_89]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_91:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_90]], %[[VAL_80]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_92:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_91]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_93:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_92]], %[[VAL_78]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_79]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_94:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_93]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_75]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_95:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_69]], %[[VAL_94]] : tensor<32xf32>, tensor<32xf32>)
// CHECK: %[[VAL_96:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_97:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_95]], %[[CST_0]] : tensor<32xf32>, f32) outs(%[[VAL_96]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_98:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_45]], %[[VAL_97]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY2]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_99:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_100:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_0]], %[[CONST_1]] : tensor<32xf32>, f32) outs(%[[VAL_99]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_101:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_102:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[VAL_100]] : tensor<32xf32>) outs(%[[VAL_101]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_103:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_102]], %[[CONST_0]] : tensor<32xf32>, f32) outs(%[[VAL_101]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_104:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_100]], %[[VAL_103]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_99]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_105:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_98]], %[[VAL_104]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY3]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK  return %[[VAL_105]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_atan_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<atan>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_tan_ops(
// CHECK-SAME: %[[ARG0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CST:.*]] = arith.constant -24.8048935 : f32
// CHECK: %[[CST_0:.*]] = arith.constant 61.2036247 : f32
// CHECK: %[[CST_1:.*]] = arith.constant -6.87115717 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 0.0698520839 : f32
// CHECK: %[[CST_3:.*]] = arith.constant -1.02906229E-13 : f32
// CHECK: %[[CST_4:.*]] = arith.constant 1.21644916E-10 : f32
// CHECK: %[[CST_5:.*]] = arith.constant 4.37113883E-8 : f32
// CHECK: %[[CST_6:.*]] = arith.constant -4.37113883E-8 : f32
// CHECK: %[[CST_7:.*]] = arith.constant 6.27711415E-7 : f32
// CHECK: %[[CST_8:.*]] = arith.constant -1.57079637 : f32
// CHECK: %[[CST_9:.*]] = arith.constant 1.57079637 : f32
// CHECK: %[[CST_10:.*]] = arith.constant 9.67025756E-4 : f32
// CHECK: %[[CST_11:.*]] = arith.constant 3.140625 : f32
// CHECK: %[[CST_12:.*]] = arith.constant 0.318309873 : f32
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0]], %[[CST_12]] : tensor<32xf32>, f32) outs(%[[VAL_0]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[VAL_1]] : tensor<32xf32>) outs(%[[VAL_2]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_11]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[ARG0]], %[[VAL_5]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_10]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_6]], %[[VAL_7]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_8]], %[[CST_9]] : tensor<32xf32>, f32) outs(%[[VAL_9]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_11:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_8]], %[[CST_8]] : tensor<32xf32>, f32) outs(%[[VAL_11]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_13:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_13]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_10]], %[[VAL_14]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_13]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_16:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_16]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_12]], %[[VAL_17]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_16]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_19:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_20:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_15]], %[[CST_6]] : tensor<32xf32>, f32) outs(%[[VAL_19]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_21:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_22:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_18]], %[[CST_5]] : tensor<32xf32>, f32) outs(%[[VAL_21]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_23:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_24:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_25:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_20]], %[[VAL_24]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_26:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_27:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_25]], %[[VAL_26]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_23]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_28:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_29:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_30:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_22]], %[[VAL_29]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_31:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_32:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_30]], %[[VAL_31]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_28]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_33:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_34:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_7]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_35:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_8]], %[[VAL_34]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_36:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_4]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_37:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_35]], %[[VAL_36]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_38:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST_3]] : tensor<32xf32>, f32) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_39:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[VAL_37]], %[[VAL_38]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_33]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_40:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_41:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_39]], %[[VAL_39]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_40]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_42:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_43:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_41]], %[[CST_2]] : tensor<32xf32>, f32) outs(%[[VAL_42]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_44:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_45:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_43]], %[[CST_1]] : tensor<32xf32>, f32) outs(%[[VAL_44]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_46:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_45]], %[[VAL_41]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_44]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_47:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_46]], %[[CST_0]] : tensor<32xf32>, f32) outs(%[[VAL_46]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_48:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_47]], %[[VAL_39]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_46]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_49:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_50:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_41]], %[[CST]] : tensor<32xf32>, f32) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_51:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_50]], %[[VAL_27]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_52:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_51]], %[[VAL_32]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_49]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_53:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_54:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_48]], %[[VAL_52]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_53]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_54]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_tan_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tan>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}
// -----

// CHECK-LABEL: func.func @test_hfusion_elemwise_erf
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xf32>)
// CHECK: %[[P5:.*]] = arith.constant 26267.2246 : f32
// CHECK: %[[P4:.*]] = arith.constant 13243.3662 : f32
// CHECK: %[[P3:.*]] = arith.constant 3023.12476 : f32
// CHECK: %[[P2:.*]] = arith.constant 398.569641 : f32
// CHECK: %[[P1:.*]] = arith.constant 31.2128582 : f32
// CHECK: %[[T5:.*]] = arith.constant 29639.3848 : f32
// CHECK: %[[T4:.*]] = arith.constant 5063.7915 : f32
// CHECK: %[[T3:.*]] = arith.constant 1393.80615 : f32
// CHECK: %[[T2:.*]] = arith.constant 101.62809 : f32
// CHECK: %[[T1:.*]] = arith.constant 7.55170154 : f32
// CHECK: %[[CST0:.*]] = arith.constant 0.0534437485 : f32
// CHECK: %[[LOWER_BOUND:.*]] = arith.constant -3.920000e+00 : f32
// CHECK: %[[UPPER_BOUND:.*]] = arith.constant 3.920000e+00 : f32
// CHECK: %[[NORM_SRC:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[MINOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[arg0]], %[[UPPER_BOUND]] : tensor<1024xf32>, f32) outs(%[[NORM_SRC]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[MAXOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINOP]], %[[LOWER_BOUND]] : tensor<1024xf32>, f32) outs(%[[NORM_SRC]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[SQURE_X_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SQURE_X:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXOP]], %[[MAXOP]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[SQURE_X_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NUMER_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[NUMER_INPUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[SQURE_X]], %[[CST0]] : tensor<1024xf32>, f32) outs(%[[NUMER_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NUMER_TMP:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[TMP0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[NUMER_INPUT]], %[[T1]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP0]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP1]], %[[T2]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP2]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP3]], %[[T3]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP4]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP6:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP5]], %[[T4]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP7:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP6]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP7]], %[[T5]] : tensor<1024xf32>, f32) outs(%[[NUMER_TMP]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[NUMER_RES_OP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXOP]], %[[TMP8]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[NUMER_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[DEMON_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[TMP11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[SQURE_X]], %[[P1]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP12:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP11]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP13:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP12]], %[[P2]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP14:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP13]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP15:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP14]], %[[P3]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP16:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP15]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP17:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP16]], %[[P4]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP18:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[TMP17]], %[[SQURE_X]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[TMP19:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[TMP18]], %[[P5]] : tensor<1024xf32>, f32) outs(%[[DEMON_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[ERF_RES:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[ERF_RES_OP:.*]]= linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[NUMER_RES_OP]], %[[TMP19]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[ERF_RES]] : tensor<1024xf32>) -> tensor<1024xf32>
func.func @test_hfusion_elemwise_erf(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<erf>} ins(%arg0 : tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
}

// -----
// CHECK-LABEL: func.func @test_i8_shift
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<200xf16>
// CHECK: %[[CAST0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0:.*]] : tensor<200xi8>) outs(%[[EMPTY0:.*]] : tensor<200xf16>) -> tensor<200xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<200xi16>
// CHECK: %[[CAST1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CAST0:.*]] : tensor<200xf16>) outs(%[[EMPTY1:.*]]: tensor<200xi16>) -> tensor<200xi16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<200xf16>
// CHECK: %[[CAST2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG1:.*]] : tensor<200xi8>) outs(%[[EMPTY2:.*]] : tensor<200xf16>) -> tensor<200xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<200xi16>
// CHECK: %[[CAST3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CAST2:.*]] : tensor<200xf16>) outs(%[[EMPTY3:.*]] : tensor<200xi16>) -> tensor<200xi16>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<200xi16>
// CHECK: %[[SHLI:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[CAST1:.*]], %[[CAST3:.*]] : tensor<200xi16>, tensor<200xi16>) outs(%[[EMPTY4:.*]] : tensor<200xi16>) -> tensor<200xi16>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<200xi8>
// CHECK: %[[RES:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%[[EMPTY4:.*]] : tensor<200xi16>) outs(%[[EMPTY6:.*]] : tensor<200xi8>) -> tensor<200xi8>
func.func @test_i8_shift(%arg0: tensor<200xi8>, %arg1: tensor<200xi8>) -> tensor<200xi8>{
  %0 = tensor.empty() : tensor<200xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%arg0, %arg1: tensor<200xi8>, tensor<200xi8>) outs(%0 : tensor<200xi8>) -> tensor<200xi8>
  return %1 : tensor<200xi8>
}

// -----
// CHECK-LABEL: func.func @test_i1_cast_i64
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<200x200xf32>
// CHECK: %[[CAST_F32:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ARG0:.*]] : tensor<200x200xi1>) outs(%[[EMPTY1:.*]] : tensor<200x200xf32>) -> tensor<200x200xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<200x200xi64>
// CHECK: %[[CAST_I64:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CAST_F32:.*]] : tensor<200x200xf32>) outs(%[[EMPTY2:.*]] : tensor<200x200xi64>) -> tensor<200x200xi64>
func.func @test_i1_cast_i64(%arg0: tensor<200x200xi1>) -> tensor<200x200xi64>{
  %0 = tensor.empty() : tensor<200x200xi64>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<200x200xi1>) outs(%0 : tensor<200x200xi64>) -> tensor<200x200xi64>
  return %1 : tensor<200x200xi64>
}

// -----

// CHECK-LABEL: @test_i16_cast_i32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi16>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @test_i16_cast_i32(%arg0: tensor<4x4xi16>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi16>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----

// CHECK-LABEL: @test_i8_cast_i32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x4xi8>) outs({{.*}} : tensor<4x4xf16>) -> tensor<4x4xf16>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xf16>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @test_i8_cast_i32(%arg0: tensor<4x4xi8>) -> tensor<4x4xi32> {
  %0 = tensor.empty() : tensor<4x4xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x4xi8>) outs(%0 : tensor<4x4xi32>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @test_i64_cast_i1
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<20x20xi64>) outs({{.*}} : tensor<20x20xf32>) -> tensor<20x20xf32>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<20x20xf32>, f32) outs({{.*}} : tensor<20x20xi1>) -> tensor<20x20xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<20x20xi1>)
func.func @test_i64_cast_i1(%arg0: tensor<20x20xi64>) -> tensor<20x20xi1>{
  %0 = tensor.empty() : tensor<20x20xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<20x20xi64>) outs(%0 : tensor<20x20xi1>) -> tensor<20x20xi1>
  return %1 : tensor<20x20xi1>
}

// -----

// CHECK-LABEL: @test_i1_cast_f32(
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x256xi1>) outs({{.*}} : tensor<4x256xf32>) -> tensor<4x256xf32>
func.func @test_i1_cast_f32(%arg0: tensor<4x256xi1>) -> tensor<4x256xf32> {
  %0 = tensor.empty() : tensor<4x256xf32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x256xi1>) outs(%0 : tensor<4x256xf32>) -> tensor<4x256xf32>
  return %1 : tensor<4x256xf32>
}

// -----
// CHECK-LABEL: func.func @test_i8_cast_i1
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<4x256xi8>) outs({{.*}} : tensor<4x256xf16>) -> tensor<4x256xf16>
// CHECK: hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins({{.*}}, %[[cst_0:.*]] : tensor<4x256xf16>, f16) outs({{.*}} : tensor<4x256xi1>) -> tensor<4x256xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[Veq:.*]] : tensor<4x256xi1>)
func.func @test_i8_cast_i1(%arg0: tensor<4x256xi8>) -> tensor<4x256xi1>{
  %0 = tensor.empty() : tensor<4x256xi1>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x256xi8>) outs(%0 : tensor<4x256xi1>) -> tensor<4x256xi1>
  return %1 : tensor<4x256xi1>
}

// -----

// CHECK-LABEL: @test_dyn_rec_mul
// CHECK: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[dim:.*]] = tensor.dim {{.*}}, %[[c0]] : tensor<?x14336xf16>
// CHECK: %[[empty:.*]] = tensor.empty(%[[dim]]) : tensor<?x14336xf16>
// CHECK: linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins({{.*}}, {{.*}} : tensor<?x14336xf16>, tensor<?x14336xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<?x14336xf16>) -> tensor<?x14336xf16>
func.func @test_dyn_rec_mul(%arg0: tensor<?x4096xf16>, %arg1: tensor<14336x4096xf16>, %arg2: tensor<?x14336xf16>) -> tensor<?x14336xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
    %0 = tensor.empty(%dim) : tensor<?x14336xf16>
    %1 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<14336x4096xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
    %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>, rec} ins(%arg2 : tensor<?x14336xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
    %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>, mul} ins(%1, %2 : tensor<?x14336xf16>, tensor<?x14336xf16>) outs(%0 : tensor<?x14336xf16>) -> tensor<?x14336xf16>
    return %3 : tensor<?x14336xf16>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_frexp
// CHECK: %[[CSTTWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CSTZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[CSTONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[FILLONE:.*]] = linalg.fill ins(%[[CSTONE:.*]] : f16) outs(%[[EMPTY0:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[ABS:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[ARG0:.*]] : tensor<10xf16>) outs(%[[EMPTY1:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[ABS:.*]] : tensor<10xf16>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[LOG1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[CAST1:.*]] : tensor<10xf32>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[FILLTWO:.*]] = linalg.fill ins(%[[CSTTWO:.*]] : f32) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[LOG2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[FILLTWO:.*]] : tensor<10xf32>) outs(%[[EMPTY3:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[DIV1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LOG1:.*]], %[[LOG2:.*]] : tensor<10xf32>, tensor<10xf32>) outs(%[[EMPTY4:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST2:.*]] = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%[[DIV1:.*]] : tensor<10xf32>) outs(%[[EMPTY2:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[EMPTY6:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[CAST2:.*]] : tensor<10xf16>) outs(%[[EMPTY6:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[EMPTY7:.*]] = tensor.empty() : tensor<10xf32>
// CHECK: %[[CAST4:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[CAST3:.*]] : tensor<10xf32>) outs(%[[EMPTY7:.*]] : tensor<10xf32>) -> tensor<10xf32>
// CHECK: %[[CAST5:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[CAST4:.*]] : tensor<10xf32>) outs(%[[EMPTY5:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY8:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[CAST5:.*]], %[[FILLONE:.*]] : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY8:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY9:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[SUB:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %23 : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY9:.*]] : tensor<10xf16>) -> tensor<10xf16>
// CHECK: %[[EMPTY10:.*]] = tensor.empty() : tensor<10xf16>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[ARG0:.*]], %[[SUB:.*]] : tensor<10xf16>, tensor<10xf16>) outs(%[[EMPTY10:.*]] : tensor<10xf16>) -> tensor<10xf16>
func.func @test_hfusion_frexp(%arg0: tensor<10xf16>) -> tensor<10xf16>{
  %cst_0 = arith.constant 0.000000e+00 : f16
  %cst_1 = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty() : tensor<10xf16>
  %1 = linalg.fill ins(%cst_1 : f16) outs(%0 : tensor<10xf16>) -> tensor<10xf16>
  %2 = tensor.empty() : tensor<10xf16>
  %3 = linalg.fill ins(%cst_0 : f16) outs(%2 : tensor<10xf16>) -> tensor<10xf16>
  %4 = tensor.empty() : tensor<10xf16>
  %5 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<ilogb>} ins(%arg0 : tensor<10xf16>) outs(%4 : tensor<10xf16>) -> tensor<10xf16>
  %6 = tensor.empty() : tensor<10xf16>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%5, %1 : tensor<10xf16>, tensor<10xf16>) outs(%6 : tensor<10xf16>) -> tensor<10xf16>
  %8 = tensor.empty() : tensor<10xi1>
  %9 = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%arg0, %3 : tensor<10xf16>, tensor<10xf16>) outs(%8 : tensor<10xi1>) -> tensor<10xi1>
  %10 = tensor.empty() : tensor<10xf16>
  %11 = hfusion.select ins(%9, %3, %7 : tensor<10xi1>, tensor<10xf16>, tensor<10xf16>) outs(%10 : tensor<10xf16>) -> tensor<10xf16>
  %12 = tensor.empty() : tensor<10xf16>
  %13 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%3, %7 : tensor<10xf16>, tensor<10xf16>) outs(%12 : tensor<10xf16>) -> tensor<10xf16>
  %14 = tensor.empty() : tensor<10xf16>
  %15 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ldexp>} ins(%arg0, %13 : tensor<10xf16>, tensor<10xf16>) outs(%14 : tensor<10xf16>) -> tensor<10xf16>
  return %15 : tensor<10xf16>
}
// -----

// CHECK-LABEL: func.func @test_linalg_sub_sv_to_muls_and_adds
// CHECK-SAME: (%[[arg0:.*]]: tensor<16xf32>)
// CHECK: %[[N1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[F5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[INIT0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[INIT1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0]], %[[N1]] : tensor<16xf32>, f32) outs(%[[INIT1]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[ADD:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[F5]], %[[MUL]] : f32, tensor<16xf32>) outs(%[[INIT0]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: return %[[ADD]]
func.func @test_linalg_sub_sv_to_muls_and_adds(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst = arith.constant 5.0 : f32
  %d = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%cst ,%arg0: f32, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %d : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_5
// CHECK: %[[cst_5:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[fill:.*]] = linalg.fill ins(%[[cst_5:.*]] : f32) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_hfusion_powf_5(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_5 = arith.constant dense<5.000000e+00> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_5: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_0
// CHECK: %[[cst_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1xf32>
// CHECK: %[[res:.*]] = linalg.fill ins(%[[cst_0:.*]] : f32) outs(%[[empty:.*]] : tensor<1xf32>) -> tensor<1xf32>
func.func @test_hfusion_powf_0(%arg0: tensor<1xf32>) -> tensor<1xf32>{
  %0 = tensor.empty(): tensor<1xf32>
  %cst_0 = arith.constant dense<0.0> : tensor<1xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_0: tensor<1xf32>, tensor<1xf32>) outs(%0: tensor<1xf32>) -> tensor<1xf32>
  return %res : tensor<1xf32>
}

// -----
// CHECK-LABEL: func.func @test_normalize_hfusion_powi_i64
func.func @test_normalize_hfusion_powi_i64(%arg0 : tensor<4x2x32xi64>, %arg1 : tensor<4x2x32xi64>) -> tensor<4x2x32xi64> {
  %0 = tensor.empty() : tensor<4x2x32xi64>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi64>, tensor<4x2x32xi64>) outs(%0: tensor<4x2x32xi64>) -> tensor<4x2x32xi64>
  return %1 : tensor<4x2x32xi64>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_f32
// CHECK: %[[cst_nan:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[cst:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[cst_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[cst_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[cst_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[c1_i32:.*]] = arith.constant -1 : i32
// CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[bitcast:.*]] = hfusion.bitcast ins(%arg0 : tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[shift:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[bitcast:.*]],  %[[c31_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[shift:.*]], %[[c1_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%arg1 : tensor<16xf32>) outs(%[[empty4]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast1]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq0]], %[[cmp_eq1]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty6]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<16xf32>) outs(%[[empty9]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[abs0]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty10]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[mul0]] : tensor<16xf32>) outs(%[[empty11]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast2]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty12]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty13:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[abs0]], %[[mul1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty13]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty14:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[add:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[sub]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty14]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty15:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq2:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[sub]], %[[cst_0]] : tensor<16xf32>, f32) outs(%[[empty15]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty16:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_ge:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%[[mul0]], %[[cst_0]] : tensor<16xf32>, f32) outs(%[[empty16]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty17:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[cmp_eq2]], %[[cmp_ge]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty17]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty18:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select0:.*]] = hfusion.select ins(%[[vor]], %[[sub]], %[[add]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%[[empty18]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty19:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[select0]], %[[cst_1]] : tensor<16xf32>, f32) outs(%[[empty19]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty20:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[add1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[mul2]], %[[cst_3]] : tensor<16xf32>, f32) outs(%[[empty20]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty21:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<16xf32>) outs(%[[empty21]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs1]] : tensor<16xf32>) outs(%[[empty7]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log0]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty8]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[exp0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul3]] : tensor<16xf32>) outs(%[[empty22]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty23:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[exp0]], %[[add1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty23]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty24:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty25:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty26:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty27:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<16xf32>) outs(%[[empty27]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs2]] : tensor<16xf32>) outs(%[[empty24]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log1]], %arg1 : tensor<16xf32>, tensor<16xf32>) outs(%[[empty25]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[exp1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul5]] : tensor<16xf32>) outs(%[[empty26]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty28:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select0:.*]] = hfusion.select ins(%[[vand0]], %[[mul4]], %[[exp1]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%[[empty28]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty29:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs3:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg0 : tensor<16xf32>) outs(%[[empty29]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty30:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq3:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs3]], %[[cst_3]] : tensor<16xf32>, f32) outs(%[[empty30]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty31:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%arg1 : tensor<16xf32>) outs(%[[empty31]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty32:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs4]], %[[cst]] : tensor<16xf32>, f32) outs(%[[empty32]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty33:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand1:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq3]], %[[cmp_eq4]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty33]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty34:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select1:.*]] = hfusion.select ins(%[[vand1]], %[[cst_3]], %[[select0]] : tensor<16xi1>, f32, tensor<16xf32>) outs(%[[empty34]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty37:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_lt0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: %[[empty38:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[empty39:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq6:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty40:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty41:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[empty42:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[empty43:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq7:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty44:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot1:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty45:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>}
// CHECK: %[[empty46:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq8:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty47:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty48:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[empty49:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand4:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[select2:.*]] = hfusion.select
// CHECK: %[[empty50:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[select3:.*]] = hfusion.select
// CHECK: return %[[select3]] : tensor<16xf32>
func.func @test_hfusion_powf_f32(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %arg1: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK-LABEL: func.func @test_hfusion_powf_cast_fill
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}
func.func @test_hfusion_powf_cast_fill(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.5 : f16
  %1 = tensor.empty(): tensor<16xf16>
  %2 = linalg.fill ins(%cst_1 : f16) outs(%1 : tensor<16xf16>) -> tensor<16xf16>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<16xf16>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %3: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_powf_f16
// CHECK: %[[cst:.*]] = arith.constant 2.13909504E+9 : f32
// CHECK: %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[cst_1:.*]] = arith.constant -2.000000e+00 : f32
// CHECK: %[[cst_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[cst_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[c1_i32:.*]] = arith.constant -1 : i32
// CHECK: %[[c31_i32:.*]] = arith.constant 31 : i32
// CHECK: %[[tmp0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast0_f32:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<16xf16>) outs(%[[tmp0:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[tmp1:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1_f32:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<16xf16>) outs(%[[tmp1:.*]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[bitcast:.*]] = hfusion.bitcast ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK: %[[shift:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[bitcast:.*]],  %[[c31_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[shift:.*]], %[[c1_i32:.*]] : tensor<16xi32>, i32) outs(%[[empty2]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>} ins(%[[cast1_f32:.*]] : tensor<16xf32>) outs(%[[empty4]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq1:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[cast1]], %[[cast1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty5]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand0:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq0]], %[[cmp_eq1]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty6]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty7:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty8:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty9:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast1_f32:.*]] : tensor<16xf32>) outs(%[[empty9]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty10:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[abs0]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty10]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty11:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[mul0]] : tensor<16xf32>) outs(%[[empty11]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty12:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[cast2]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty12]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty13:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[sub:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%[[abs0]], %[[mul1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty13]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty14:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[add:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[sub]], %[[cst_2]] : tensor<16xf32>, f32) outs(%[[empty14]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty15:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq2:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[sub]], %[[cst_0]] : tensor<16xf32>, f32) outs(%[[empty15]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty16:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_ge:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vge>} ins(%[[mul0]], %[[cst_0]] : tensor<16xf32>, f32) outs(%[[empty16]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty17:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vor:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vor>} ins(%[[cmp_eq2]], %[[cmp_ge]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty17]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty18:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select0:.*]] = hfusion.select ins(%[[vor]], %[[sub]], %[[add]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%[[empty18]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty19:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul2:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[select0]], %[[cst_1]] : tensor<16xf32>, f32) outs(%[[empty19]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty20:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[add1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[mul2]], %[[cst_3]] : tensor<16xf32>, f32) outs(%[[empty20]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty21:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty21]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs1]] : tensor<16xf32>) outs(%[[empty7]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul3:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log0]], %[[cast1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty8]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty22:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[exp0:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul3]] : tensor<16xf32>) outs(%[[empty22]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty23:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[mul4:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[exp0]], %[[add1]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty23]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty24:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty25:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty26:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[empty27:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs2:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty27]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[log1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[abs2]] : tensor<16xf32>) outs(%[[empty24]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[mul5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[log1]], %[[cast1_f32:.*]] : tensor<16xf32>, tensor<16xf32>) outs(%[[empty25]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[exp1:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[mul5]] : tensor<16xf32>) outs(%[[empty26]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty28:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select0:.*]] = hfusion.select ins(%[[vand0]], %[[mul4]], %[[exp1]] : tensor<16xi1>, tensor<16xf32>, tensor<16xf32>) outs(%[[empty28]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty29:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs3:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast0_f32:.*]] : tensor<16xf32>) outs(%[[empty29]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty30:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq3:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs3]], %[[cst_3]] : tensor<16xf32>, f32) outs(%[[empty30]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty31:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs4:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>} ins(%[[cast1_f32:.*]] : tensor<16xf32>) outs(%[[empty31]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty32:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq4:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[abs4]], %[[cst]] : tensor<16xf32>, f32) outs(%[[empty32]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty33:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand1:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[cmp_eq3]], %[[cmp_eq4]] : tensor<16xi1>, tensor<16xi1>) outs(%[[empty33]] : tensor<16xi1>) -> tensor<16xi1>
// CHECK: %[[empty34:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[select1:.*]] = hfusion.select ins(%[[vand1]], %[[cst_3]], %[[select0]] : tensor<16xi1>, f32, tensor<16xf32>) outs(%[[empty34]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: %[[empty37:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_lt0:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<vlt>}
// CHECK: %[[empty38:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[empty39:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq6:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty40:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot0:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty41:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[empty42:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[abs6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<abs>}
// CHECK: %[[empty43:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq7:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty44:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot1:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty45:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[cast3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>}
// CHECK: %[[empty46:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq8:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[empty47:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vnot2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}
// CHECK: %[[empty48:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[empty49:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[vand4:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>}
// CHECK: %[[select2:.*]] = hfusion.select
// CHECK: %[[empty50:.*]] = tensor.empty() : tensor<16xi1>
// CHECK: %[[cmp_eq9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>}
// CHECK: %[[select3:.*]] = hfusion.select
// CHECK: %[[res:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[select3:.*]] : tensor<16xf32>) outs(%[[empty30:.*]] : tensor<16xf16>) -> tensor<16xf16>
// CHECK: return %[[res:.*]] : tensor<16xf16>
func.func @test_hfusion_powf_f16(%arg0: tensor<16xf16>, %arg1: tensor<16xf16>) -> tensor<16xf16>{
  %0 = tensor.empty(): tensor<16xf16>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %arg1: tensor<16xf16>, tensor<16xf16>) outs(%0: tensor<16xf16>) -> tensor<16xf16>
  return %res : tensor<16xf16>
}

// -----

// CHECK-LABEL: func.func @test_normalize_hfusion_powi_i8
// CHECK: %[[VAL_0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x2x32xi8>) outs(%[[empty0:.*]] : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<4x2x32xf16>) outs(%[[empty1:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg1 : tensor<4x2x32xi8>) outs(%[[empty2:.*]] : tensor<4x2x32xf16>) -> tensor<4x2x32xf16>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_2:.*]] : tensor<4x2x32xf16>) outs(%[[empty3:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[_:.*]] : tensor<4x2x32xf32>) outs(%[[_:.*]] : tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
// CHECK: %[[result:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%[[_:.*]] : tensor<4x2x32xi32>) outs(%[[empty4:.*]] : tensor<4x2x32xi8>) -> tensor<4x2x32xi8>
// CHECK: return %[[result:.*]] : tensor<4x2x32xi8>

func.func @test_normalize_hfusion_powi_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x32xi8> {
  %0 = tensor.empty() : tensor<4x2x32xi8>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi8>, tensor<4x2x32xi8>) outs(%0: tensor<4x2x32xi8>) -> tensor<4x2x32xi8>
  return %1 : tensor<4x2x32xi8>
}

// -----

// CHECK-LABEL: func.func @test_normalize_hfusion_powi_i16
// CHECK: %[[VAL_0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<4x2x32xi16>) outs(%[[empty0:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<4x2x32xi16>) outs(%[[empty1:.*]] : tensor<4x2x32xf32>) -> tensor<4x2x32xf32>
// CHECK: %[[Result:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[empty2:.*]] : tensor<4x2x32xf32>) outs(%[[empty4:.*]] : tensor<4x2x32xi32>) -> tensor<4x2x32xi32>
// CHECK: %[[Result:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%[[empty4:.*]] : tensor<4x2x32xi32>) outs(%[[empty3:.*]] : tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
// CHECK: return %[[Result]] : tensor<4x2x32xi16>

func.func @test_normalize_hfusion_powi_i16(%arg0 : tensor<4x2x32xi16>, %arg1 : tensor<4x2x32xi16>) -> tensor<4x2x32xi16> {
  %0 = tensor.empty() : tensor<4x2x32xi16>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powi>} ins(%arg0,  %arg1: tensor<4x2x32xi16>, tensor<4x2x32xi16>) outs(%0: tensor<4x2x32xi16>) -> tensor<4x2x32xi16>
  return %1 : tensor<4x2x32xi16>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<floor>}
func.func @test_floordivsi(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<floordivsi>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}


// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_7:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<ceil>}
func.func @test_ceildivsi(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = hfusion.elemwise_binary {fun = #hfusion.binary_fn<ceildivsi>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_powf_half
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[res:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xf32>) -> tensor<16xf32>
func.func @test_hfusion_powf_half(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_1 = arith.constant 0.500000e+00 : f32
  %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0,  %1: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----
// CHECK-LABEL: func.func @test_hfusion_powf_const_dense
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<16xf32>
// CHECK: %[[res:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%[[arg0:.*]]: tensor<16xf32>) outs(%[[empty0:.*]] : tensor<16xf32>) -> tensor<16xf32>
func.func @test_hfusion_powf_const_dense(%arg0: tensor<16xf32>) -> tensor<16xf32>{
  %0 = tensor.empty(): tensor<16xf32>
  %cst_dense = arith.constant dense<0.500000e+00> : tensor<16xf32>
  %res = hfusion.elemwise_binary {fun = #hfusion.binary_fn<powf>} ins(%arg0, %cst_dense: tensor<16xf32>, tensor<16xf32>) outs(%0: tensor<16xf32>) -> tensor<16xf32>
  return %res : tensor<16xf32>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xi1>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xi1>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_0]] : tensor<4x2x64xi1>) outs(%[[VAL_3:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_6:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_1]] : tensor<4x2x64xi1>) outs(%[[VAL_5:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_7:.*]] = hfusion.interleave %[[VAL_4:.*]], %[[VAL_6:.*]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<4x2x128xi1>
// CHECK: %[[VAL_10:.*]] = tensor.empty() : tensor<4x2x128xi1>
// CHECK: %[[VAL_9:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[VAL_7:.*]], %{{.*}} : tensor<4x2x128xf16>, f16) outs(%[[VAL_10:.*]] : tensor<4x2x128xi1>) -> tensor<4x2x128xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VAL_9]] : tensor<4x2x128xi1>) outs(%[[VAL_8]] : tensor<4x2x128xi1>) -> tensor<4x2x128xi1>
func.func @test_interleave_i1() -> tensor<4x2x128xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = tensor.empty() : tensor<4x2x64xi1>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xi1>, tensor<4x2x64xi1> -> tensor<4x2x128xi1>
  return %2 : tensor<4x2x128xi1>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xi8>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xi8>
// CHECK: %[[VAL_3:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_4:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_0]] : tensor<4x2x64xi8>) outs(%[[VAL_3:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_5:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_6:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VAL_1]] : tensor<4x2x64xi8>) outs(%[[VAL_5:.*]] : tensor<4x2x64xf16>) -> tensor<4x2x64xf16>
// CHECK: %[[VAL_7:.*]] = hfusion.interleave %[[VAL_4:.*]], %[[VAL_6:.*]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
// CHECK: %[[VAL_8:.*]] = tensor.empty() : tensor<4x2x128xi8>
// CHECK: %[[VAL_9:.*]] = hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[VAL_7:.*]] : tensor<4x2x128xf16>) outs(%[[VAL_8:.*]] : tensor<4x2x128xi8>) -> tensor<4x2x128xi8>
func.func @test_interleave_i8() -> tensor<4x2x128xi8> {
  %0 = tensor.empty() : tensor<4x2x64xi8>
  %1 = tensor.empty() : tensor<4x2x64xi8>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xi8>, tensor<4x2x64xi8> -> tensor<4x2x128xi8>
  return %2 : tensor<4x2x128xi8>
}

// -----

// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<4x2x64xf16>
// CHECK: %[[VAL_2:.*]] = hfusion.interleave %[[VAL_0]], %[[VAL_1]] : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
func.func @test_interleave_f16() -> tensor<4x2x128xf16> {
  %0 = tensor.empty() : tensor<4x2x64xf16>
  %1 = tensor.empty() : tensor<4x2x64xf16>
  %2 = hfusion.interleave %0, %1 : tensor<4x2x64xf16>, tensor<4x2x64xf16> -> tensor<4x2x128xf16>
  return %2 : tensor<4x2x128xf16>
}

// -----

// CHECK-LABEL: @test_normalize_reduce_with_index_ra_to_ar
// CHECK: %[[res0:.*]] = tensor.empty() : tensor<32x128xf32>
// CHECK: %[[res1:.*]] = tensor.empty() : tensor<32x128xi32>
// CHECK: %[[tmp_buf0:.*]] = tensor.empty() : tensor<32x128x32xf32>
// CHECK: %[[transposed0:.*]] = linalg.transpose ins(%[[arg0:.*]] : tensor<32x32x128xf32>) outs(%[[tmp_buf0]] : tensor<32x128x32xf32>) permutation = [0, 2, 1]
// CHECK: %[[tmp_buf1:.*]] = tensor.empty() : tensor<32x128x32xi32>
// CHECK: %[[transposed1:.*]] = linalg.transpose ins(%[[arg1:.*]] : tensor<32x32x128xi32>) outs(%[[tmp_buf1]] : tensor<32x128x32xi32>) permutation = [0, 2, 1]
// CHECK: hfusion.reduce_with_index {tie_break_left = true} <min> ins(%[[transposed0]], %[[transposed1]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>) outs(%[[res0]], %[[res1]] : tensor<32x128xf32>, tensor<32x128xi32>) dimensions = [2]
func.func @test_normalize_reduce_with_index_ra_to_ar(%arg0: tensor<32x32x128xf32>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %true = arith.constant true
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xf32>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xf32>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----

// CHECK-LABEL: func.func @opt_cast_IToF_fill
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<24x32xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<24x32xf32>) -> tensor<24x32xf32>
// CHECK: return %[[FILL]] : tensor<24x32xf32>
func.func @opt_cast_IToF_fill() -> tensor<24x32xf32>{
  %c1_i32 = arith.constant 1 : i32
  %0 = tensor.empty() : tensor<24x32xf32>
  %1 = tensor.empty() : tensor<24x32xi32>
  %2 = linalg.fill ins(%c1_i32 : i32) outs(%1 : tensor<24x32xi32>) -> tensor<24x32xi32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<24x32xi32>) outs(%0 : tensor<24x32xf32>) -> tensor<24x32xf32>
  return %3 : tensor<24x32xf32>
}
// -----

// CHECK-LABEL: func.func @opt_cast_FToI_fill_rint
// CHECK: %[[CST:.*]] = arith.constant 1.500000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<f32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY0]] : tensor<f32>) -> tensor<f32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<i32>
// CHECK: %[[CAST:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<ceil>} ins(%[[FILL]] : tensor<f32>) outs(%[[EMPTY1]] : tensor<i32>) -> tensor<i32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<24x32xi32>
// CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[CAST]] : tensor<i32>) outs(%[[EMPTY2]] : tensor<24x32xi32>) dimensions = [0, 1]
// CHECK: return %[[BRC]] : tensor<24x32xi32>
func.func @opt_cast_FToI_fill_rint() -> tensor<24x32xi32>{
  %c1_i32 = arith.constant 1.5 : f32
  %0 = tensor.empty() : tensor<24x32xf32>
  %1 = tensor.empty() : tensor<24x32xi32>
  %2 = linalg.fill ins(%c1_i32 : f32) outs(%0 : tensor<24x32xf32>) -> tensor<24x32xf32>
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<ceil>} ins(%2 : tensor<24x32xf32>) outs(%1 : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %3 : tensor<24x32xi32>
}

// -----

// CHECK-LABEL: func.func @opt_cast_FToI_brc
// CHECK: %[[CST:.*]] = arith.constant dense<1.500000e+00> : tensor<32xf32>
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xi32>
// CHECK: %[[CAST:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[CST]] : tensor<32xf32>) outs(%[[EMPTY0]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<24x32xi32>
// CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[CAST]] : tensor<32xi32>) outs(%[[EMPTY1]] : tensor<24x32xi32>) dimensions = [0]
// CHECK: return %[[BRC]] : tensor<24x32xi32>
func.func @opt_cast_FToI_brc() -> tensor<24x32xi32>{
  %c1_f32 = arith.constant dense<1.5> : tensor<32xf32>
  %0 = tensor.empty() : tensor<24x32xi32>
  %1 = tensor.empty() : tensor<24x32xf32>
  %2 = linalg.broadcast ins(%c1_f32 : tensor<32xf32>) outs(%1 : tensor<24x32xf32>) dimensions=[0]
  %3 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%2 : tensor<24x32xf32>) outs(%0 : tensor<24x32xi32>) -> tensor<24x32xi32>
  return %3 : tensor<24x32xi32>
}

// -----

// CHECK-LABEL: @test_normalize_interleave_i1
// CHECK :  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_interleave_i1(%arg0 : tensor<4x2x32xi1>, %arg1 : tensor<4x2x32xi1>) -> tensor<4x2x64xi1> {
  %0 = tensor.empty() : tensor<4x2x64xi1>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi1>, tensor<4x2x32xi1> -> tensor<4x2x64xi1>
  return %1 : tensor<4x2x64xi1>
}

// -----

// CHECK-LABEL: @test_normalize_interleave_i8
// CHECK :  %[[res_f16:.*]] = hfusion.interleave %[[arg0_f16:.*]], %[[arg1_f16:.*]] : tensor<4x2x32xf16>, tensor<4x2x32xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_interleave_i8(%arg0 : tensor<4x2x32xi8>, %arg1 : tensor<4x2x32xi8>) -> tensor<4x2x64xi8> {
  %0 = tensor.empty() : tensor<4x2x64xi8>
  %1 = hfusion.interleave %arg0, %arg1 : tensor<4x2x32xi8>, tensor<4x2x32xi8> -> tensor<4x2x64xi8>
  return %1 : tensor<4x2x64xi8>
}

// -----

// CHECK-LABEL: @test_normalize_deinterleave_i8
// CHECK: %[[res_f16:.*]] = hfusion.deinterleave %[[cast_f16:.*]] channel<1> : tensor<4x2x128xf16> -> tensor<4x2x64xf16>
func.func @test_normalize_deinterleave_i8() -> tensor<4x2x64xi8> {
  %0 = tensor.empty() : tensor<4x2x128xi8>
  %1 = hfusion.deinterleave %0 channel<1> : tensor<4x2x128xi8> -> tensor<4x2x64xi8>
  return %1 : tensor<4x2x64xi8>
}

// -----


// CHECK-LABEL: func.func @test_hfusion_tanh_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<32xf32>) -> tensor<32xf32> {
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST0:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CST2:.*]] = arith.constant -8.800000e+00 : f32
// CHECK: %[[CST3:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_2:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[VAL_0]], %[[CST3]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[VAL_2]], %[[CST2]] : tensor<32xf32>, f32) outs(%[[VAL_1]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[VAL_3]], %[[CST1]] : tensor<32xf32>, f32) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_6:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[VAL_5]] : tensor<32xf32>) outs(%[[VAL_4]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_7:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_8:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_6]], %[[CST0]] : tensor<32xf32>, f32) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[VAL_9:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[VAL_10:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VAL_6]], %[[CST]] : tensor<32xf32>, f32) outs(%[[VAL_9]] : tensor<32xf32>) -> tensor<32xf32
// CHECK: %[[VAL_11:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[VAL_8]], %[[VAL_10]] : tensor<32xf32>, tensor<32xf32>) outs(%[[VAL_7]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: return %[[VAL_11]] : tensor<32xf32>
// CHECK: }
func.func @test_hfusion_tanh_ops(%arg0 : tensor<32xf32>) ->  tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg0 : tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %ret : tensor<32xf32>
}
// -----

// CHECK-LABEL: func.func @test_hfusion_tanh_ops_f16
// CHECK: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[CST_NEG1:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[CST_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[CST_NEG8:.*]] = arith.constant -8.800000e+00 : f32
// CHECK: %[[CST_8DOT8:.*]] = arith.constant 8.800000e+00 : f32
// CHECK: %[[EMPTY0:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[CAST0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[ARG0:.*]] : tensor<32xf16>) outs(%[[EMPTY0:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY1:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[MINF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<minf>} ins(%[[CAST0:.*]], %[[CST_8DOT8:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[MAXF:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<maxf>} ins(%[[MINF:.*]], %[[CST_NEG8:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY1:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY2:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[MUL:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[MAXF:.*]], %[[CST_2:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY2:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EXP:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>} ins(%[[MUL:.*]] : tensor<32xf32>) outs(%[[EMPTY2:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY3:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ADD0:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP:.*]], %[[CST_NEG1:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY3:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY4:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[ADD1:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[EXP:.*]], %[[CST_1:.*]] : tensor<32xf32>, f32) outs(%[[EMPTY4:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[DIV:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[ADD0:.*]], %[[ADD1:.*]] : tensor<32xf32>, tensor<32xf32>) outs(%[[EMPTY3:.*]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[EMPTY5:.*]] = tensor.empty() : tensor<32xf16>
// CHECK: %[[RES:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<round>} ins(%[[DIV:.*]] : tensor<32xf32>) outs(%[[EMPTY5:.*]] : tensor<32xf16>) -> tensor<32xf16>
func.func @test_hfusion_tanh_ops_f16(%arg0 : tensor<32xf16>) ->  tensor<32xf16> {
  %0 = tensor.empty() : tensor<32xf16>
  %ret = hfusion.elemwise_unary {fun = #hfusion.unary_fn<tanh>} ins(%arg0 : tensor<32xf16>) outs(%0 : tensor<32xf16>) -> tensor<32xf16>
  return %ret : tensor<32xf16>
}
// -----

// CHECK-LABEL: @normalize_mulext_i8_high_bits
// CHECK: %[[cst_8:.*]] = arith.constant 8 : i16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg0_f16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<4x2xi8>) outs(%[[empty0:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg0_i16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg0_f16:.*]] : tensor<4x2xf16>) outs(%[[empty1:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg1_f16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg1:.*]] : tensor<4x2xi8>) outs(%[[empty2:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg1_i16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg1_f16:.*]] : tensor<4x2xf16>) outs(%[[empty3:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0_i16:.*]], %[[arg1_i16:.*]] : tensor<4x2xi16>, tensor<4x2xi16>) outs(%[[empty4:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[res_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[mul:.*]], %[[cst_8:.*]] : tensor<4x2xi16>, i16) outs(%[[empty5:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
func.func @normalize_mulext_i8_high_bits(%arg0: tensor<4x2xi8>, %arg1: tensor<4x2xi8>) -> tensor<4x2xi8> {
  %low, %high = hfusion.mulext %arg0, %arg1 : tensor<4x2xi8>
  return %high : tensor<4x2xi8>
}

// -----

// CHECK-LABEL: @normalize_mulext_i8_low_bits
// CHECK: %[[cst_8:.*]] = arith.constant 8 : i16
// CHECK: %[[empty0:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg0_f16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<4x2xi8>) outs(%[[empty0:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty1:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg0_i16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg0_f16:.*]] : tensor<4x2xf16>) outs(%[[empty1:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty2:.*]] = tensor.empty() : tensor<4x2xf16>
// CHECK: %[[arg1_f16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg1:.*]] : tensor<4x2xi8>) outs(%[[empty2:.*]] : tensor<4x2xf16>) -> tensor<4x2xf16>
// CHECK: %[[empty3:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[arg1_i16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[arg1_f16:.*]] : tensor<4x2xf16>) outs(%[[empty3:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty4:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[mul:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%[[arg0_i16:.*]], %[[arg1_i16:.*]] : tensor<4x2xi16>, tensor<4x2xi16>) outs(%[[empty4:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty5:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[shl:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shli>} ins(%[[mul:.*]], %[[cst_8:.*]] : tensor<4x2xi16>, i16) outs(%[[empty5:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
// CHECK: %[[empty6:.*]] = tensor.empty() : tensor<4x2xi16>
// CHECK: %[[res_i16:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<shrsi>} ins(%[[shl:.*]], %c8_i16 : tensor<4x2xi16>, i16) outs(%[[empty6:.*]] : tensor<4x2xi16>) -> tensor<4x2xi16>
func.func @normalize_mulext_i8_low_bits(%arg0: tensor<4x2xi8>, %arg1: tensor<4x2xi8>) -> tensor<4x2xi8> {
  %low, %high = hfusion.mulext %arg0, %arg1 : tensor<4x2xi8>
  return  %low : tensor<4x2xi8>
}

// CHECK-LABEL: @normalize_vlog_f16_to_f32
// CHECK: %[[a0:.*]] = tensor.empty() : tensor<17x256xf16>
// CHECK: %[[a1:.*]] = tensor.empty() : tensor<17x256xf32>
// CHECK: %[[a2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg0:.*]] : tensor<17x256xf16>) outs(%[[a1]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a3:.*]] = tensor.empty() : tensor<17x256xf32>
// CHECK: %[[a4:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[a0]] : tensor<17x256xf16>) outs(%[[a3]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a5:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%[[a2]] : tensor<17x256xf32>) outs(%[[a4]] : tensor<17x256xf32>) -> tensor<17x256xf32>
// CHECK: %[[a6:.*]] = tensor.empty() : tensor<17x256xf16>
// CHECK: %[[a7:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[a5]] : tensor<17x256xf32>) outs(%[[a6]] : tensor<17x256xf16>) -> tensor<17x256xf16>
func.func @normalize_vlog_f16_to_f32(%arg0: tensor<17x256xf16>) -> tensor<17x256xf16> {
  %0 = tensor.empty() : tensor<17x256xf16>
  %1 = linalg.elemwise_unary {fun = #linalg.unary_fn<log>} ins(%arg0 : tensor<17x256xf16>) outs(%0 : tensor<17x256xf16>) -> tensor<17x256xf16>
  return %1 : tensor<17x256xf16>
}

// -----
// CHECK-LABEL: func.func @test_isnan
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[POSONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NEGINF:.*]] = arith.constant -2139095040 : i32
// CHECK: %[[MASKVAL:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[INPUT:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[MASKRES:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[VDUPOP:.*]] = linalg.fill ins(%[[MASKVAL]] : i32) outs(%[[MASKRES]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[BITCASTEMPTY:.*]] = tensor.empty() : tensor<8192xi32>
// CHECK: %[[BITCASTINPUT:.*]] = hfusion.bitcast ins(%[[INPUT]] : tensor<8192xf32>) outs(%[[BITCASTOUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VANDOP:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[BITCASTINPUT]], %[[VDUPOP]] : tensor<8192xi32>, tensor<8192xi32>) outs(%[[VANDOUTPUT:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VADDOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%[[VANDOP]], %[[NEGINF]] : tensor<8192xi32>, i32) outs(%[[VADDRES:.*]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMINOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<min_signed>} ins(%[[VADDOP]], %[[POSONE]] : tensor<8192xi32>, i32) outs(%[[VADDOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[VMAXOP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%[[VMINOP]], %[[ZERO]] : tensor<8192xi32>, i32) outs(%[[VMINOP]] : tensor<8192xi32>) -> tensor<8192xi32>
// CHECK: %[[TMPF:.*]] = tensor.empty() : tensor<8192xf32>
// CHECK: %[[CASTF:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[VMAXOP]] : tensor<8192xi32>) outs(%[[TMPF]] : tensor<8192xf32>) -> tensor<8192xf32>
// CHECK: %[[OUT1:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[OUT2:.*]] = tensor.empty() : tensor<8192xi1>
// CHECK: %[[RES:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[CASTF]], %[[CST0:.*]] : tensor<8192xf32>, f32) outs(%[[OUT2]] : tensor<8192xi1>) -> tensor<8192xi1>
// CHECK: %[[RES2:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[RES]] : tensor<8192xi1>) outs(%[[OUT1]] : tensor<8192xi1>) -> tensor<8192xi1>
func.func @test_isnan() -> tensor<8192xi1> {
  %0 = tensor.empty() : tensor<8192xf32>
  %2 = hfusion.isnan %0 : tensor<8192xf32> -> tensor<8192xi1>
  return %2 : tensor<8192xi1>
}

// -----
// CHECK-LABEL: func.func @test_divui
// CHECK: %[[VAL_0:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
// CHECK: %[[VAL_2:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_3:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
// CHECK: %[[VAL_4:.*]] = tensor.empty() : tensor<6x6xf32>
// CHECK: %[[VAL_5:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>}
// CHECK: %[[VAL_6:.*]] = tensor.empty() : tensor<6x6xi32>
// CHECK: %[[VAL_7:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>}
func.func @test_divui(%arg0: tensor<6x6xi32>, %arg1: tensor<6x6xi32>) -> tensor<6x6xi32> {
  %0 = tensor.empty() : tensor<6x6xi32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<6x6xi32>, tensor<6x6xi32>) outs(%0 : tensor<6x6xi32>) -> tensor<6x6xi32>
  return %1 : tensor<6x6xi32>
}

// -----
// CHECK-LABEL: func.func @test_divui_vs
func.func @test_divui_vs(%arg0: tensor<48xi32>, %arg1: i32) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[LHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<48xi32>) outs(%[[LHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RHSCASTED:.*]] = arith.sitofp %arg1 : i32 to f32
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSFP]], %[[RHSCASTED]] : tensor<48xf32>, f32) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : tensor<48xi32>, i32) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----
// CHECK-LABEL: func.func @test_divui_sv
func.func @test_divui_sv(%arg0: i32, %arg1: tensor<48xi32>) -> tensor<48xi32> {
    %res = tensor.empty() : tensor<48xi32>
    // CHECK: %[[LHSCASTED:.*]] = arith.sitofp %arg0 : i32 to f32
    // CHECK: %[[RHS:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RHSFP:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg1 : tensor<48xi32>) outs(%[[RHS]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<48xf32>
    // CHECK: %[[RESFP:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%[[LHSCASTED]], %[[RHSFP]] : f32, tensor<48xf32>) outs(%[[RES]] : tensor<48xf32>) -> tensor<48xf32>
    // CHECK: %[[RESINT:.*]] = tensor.empty() : tensor<48xi32>
    // CHECK: %[[RET:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%[[RESFP]] : tensor<48xf32>) outs(%[[RESINT]] : tensor<48xi32>) -> tensor<48xi32>
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div_unsigned>} ins(%arg0, %arg1 : i32, tensor<48xi32>) outs(%res : tensor<48xi32>) -> tensor<48xi32>
    return %0 : tensor<48xi32>
}

// -----
// CHECK-LABEL: @test_cast_f32_to_i16
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<4x4xf32>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_cast_f32_to_i16(%arg0: tensor<4x4xf32>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins(%arg0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----
// CHECK-LABEL: @test_cast_i64_to_i16
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi16>) -> tensor<4x4xi16>
func.func @test_cast_i64_to_i16(%arg0: tensor<4x4xi64>) -> tensor<4x4xi16> {
  %0 = tensor.empty() : tensor<4x4xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi16>) -> tensor<4x4xi16>
  return %1 : tensor<4x4xi16>
}

// -----
// CHECK-LABEL: @test_cast_i64_to_i8
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi64>) outs({{.*}} : tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<truncwithoverflow>} ins({{.*}} : tensor<4x4xi32>) outs({{.*}} : tensor<4x4xi8>) -> tensor<4x4xi8>
func.func @test_cast_i64_to_i8(%arg0: tensor<4x4xi64>) -> tensor<4x4xi8> {
  %0 = tensor.empty() : tensor<4x4xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<4x4xi64>) outs(%0 : tensor<4x4xi8>) -> tensor<4x4xi8>
  return %1 : tensor<4x4xi8>
}

// -----
// CHECK-LABEL: @test_broadcast_i1
// CHECK: %[[CAST16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<8xi1>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[BROADCAST16:.*]] = linalg.broadcast ins(%[[CAST16]] : tensor<8xf16>) outs({{.*}} : tensor<8x16xf16>) dimensions = [1] 
// CHECK: %[[VEQ:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[BROADCAST16]], {{.*}} : tensor<8x16xf16>, f16) outs({{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
// CHECK: hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[VEQ]] : tensor<8x16xi1>) outs(%{{.*}} : tensor<8x16xi1>) -> tensor<8x16xi1>
func.func @test_broadcast_i1(%arg0: tensor<8xi1>, %arg1: tensor<8x16xi1>) -> tensor<8x16xi1> {
  %0 = tensor.empty() : tensor<8x16xi1>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi1>)
    outs(%0 : tensor<8x16xi1>)
    dimensions = [1]
  return %1 : tensor<8x16xi1>
}

// -----
// CHECK-LABEL: @test_broadcast_i8
// CHECK: %[[CAST16:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<8xi8>) outs({{.*}} : tensor<8xf16>) -> tensor<8xf16>
// CHECK: %[[BROADCAST16:.*]] = linalg.broadcast ins(%[[CAST16]] : tensor<8xf16>) outs({{.*}} : tensor<8x16xf16>) dimensions = [1] 
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins(%[[BROADCAST16]] : tensor<8x16xf16>) outs({{.*}} : tensor<8x16xi8>) -> tensor<8x16xi8>
func.func @test_broadcast_i8(%arg0: tensor<8xi8>, %arg1: tensor<8x16xi8>) -> tensor<8x16xi8> {
  %0 = tensor.empty() : tensor<8x16xi8>
  %1 = linalg.broadcast
    ins(%arg0 : tensor<8xi8>)
    outs(%0 : tensor<8x16xi8>)
    dimensions = [1]
  return %1 : tensor<8x16xi8>
}

// -----

// CHECK-LABEL: @test_cumsum_f16
// CHECK: hfusion.cumsum %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [0] -> tensor<4x32xf32>
module {
  func.func @test_cumsum_f16(%arg0: tensor<4x64xf16>) -> tensor<4x32xf16> {
    %0 = tensor.empty() : tensor<4x32xf16>
    %1 = hfusion.cumsum %arg0 : tensor<4x64xf16> cum_dims = [0] -> tensor<4x32xf16>
    return %1 : tensor<4x32xf16>
  }
}

// -----

// CHECK-LABEL: @test_cumprod_f16
// CHECK: hfusion.cumprod %[[INPUT0:.*]] : tensor<4x64xf32> cum_dims = [1] -> tensor<4x32xf32>
module {
  func.func @test_cumprod_f16(%arg0: tensor<4x64xf16>) -> tensor<4x32xf16> {
    %0 = tensor.empty() : tensor<4x32xf16>
    %1 = hfusion.cumprod %arg0 : tensor<4x64xf16> cum_dims = [1] -> tensor<4x32xf16>
    return %1 : tensor<4x32xf16>
  }
}

// -----
// CHECK-LABEL: @test_reduce_with_index_i1_return_value
func.func @test_reduce_with_index_i1_return_value(%arg0: tensor<32x32x128xi1>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi1> {
  %0 = tensor.empty() : tensor<32x128xi1>
  %1 = tensor.empty() : tensor<32x128xi32>
  // CHECK %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true} <min>  ins(%[[arg0:.*]], %[[arg1:.*] : tensor<32x32x128xf16>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xi1>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xi1>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xi1>, tensor<32x128xi32>

  return %reduced#0 : tensor<32x128xi1>
}

// -----
// CHECK-LABEL: @test_reduce_with_index_i1_return_index
func.func @test_reduce_with_index_i1_return_index(%arg0: tensor<32x32x128xi1>, %arg1: tensor<32x32x128xi32>) -> tensor<32x128xi32> {
  %0 = tensor.empty() : tensor<32x128xi1>
  %1 = tensor.empty() : tensor<32x128xi32>
  // CHECK %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true} <min>  ins(%[[arg0:.*]], %[[arg1:.*] : tensor<32x32x128xf16>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xi1>, tensor<32x32x128xi32>)
                outs(%0, %1 : tensor<32x128xi1>, tensor<32x128xi32>)
                dimensions = [1] -> tensor<32x128xi1>, tensor<32x128xi32>

  return %reduced#1 : tensor<32x128xi32>
}

// -----
// CHECK-LABEL: @test_reduce_with_index_i64_return_index
func.func @test_reduce_with_index_i64_return_index(%arg0: tensor<32x32x128xf32>, %arg1: tensor<32x32x128xi64>) -> tensor<32x128xf32> {
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = tensor.empty() : tensor<32x128xi64>
  // CHECK: %[[reduce:.*]] = hfusion.reduce_with_index {tie_break_left = true} <min>  ins(%[[arg0:.*]], %[[arg1:.*]] : tensor<32x128x32xf32>, tensor<32x128x32xi32>
  %reduced:2 = hfusion.reduce_with_index {tie_break_left = true} <min>
                ins(%arg0, %arg1 : tensor<32x32x128xf32>, tensor<32x32x128xi64>)
                outs(%0, %1 : tensor<32x128xf32>, tensor<32x128xi64>)
                dimensions = [1] -> tensor<32x128xf32>, tensor<32x128xi64>

  return %reduced#0 : tensor<32x128xf32>
}

// -----
// CHECK-LABEL: @test_reduce_i1_addi
func.func @test_reduce_i1_addi(%arg0: tensor<16x32x64xi1>) -> tensor<16x64xi1> {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
  %reduce = linalg.reduce { arith.addi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce : tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_reduce_i1_muli
func.func @test_reduce_i1_muli(%arg0: tensor<16x32x64xi1>) -> tensor<16x64xi1> {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
  %reduce = linalg.reduce { arith.muli } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce : tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_reduce_i1_maxi
func.func @test_reduce_i1_maxi(%arg0: tensor<16x32x64xi1>) -> (tensor<16x64xi1>, tensor<16x64xi1>) {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.ori }
  %reduce = linalg.reduce { arith.maxui } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  %reduce_0 = linalg.reduce { arith.maxsi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce, %reduce_0 : tensor<16x64xi1>, tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_reduce_i1_mini
func.func @test_reduce_i1_mini(%arg0: tensor<16x32x64xi1>) -> (tensor<16x64xi1>, tensor<16x64xi1>) {
  %0 = tensor.empty() : tensor<16x64xi1>
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
  // CHECK: %[[reduce:.*]] = linalg.reduce { arith.andi }
  %reduce = linalg.reduce { arith.minui } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  %reduce_0 = linalg.reduce { arith.minsi } ins(%arg0 : tensor<16x32x64xi1>) outs(%0 : tensor<16x64xi1>) dimensions = [1]
  return %reduce, %reduce_0 : tensor<16x64xi1>, tensor<16x64xi1>
}

// -----
// CHECK-LABEL: @test_compare_i1
func.func @test_compare_i1(%arg0: tensor<16x32xi1>,%arg1: tensor<16x32xi1>,  %dst : tensor<16x32xi1>) -> (tensor<16x32xi1>) {
  // CHECK: %[[ret:.*]] = hfusion.compare  {compare_fn = #hfusion.compare_fn<vlt>} ins(%[[in1:.*]], %[[in2:.*]] : tensor<16x32xf16>, tensor<16x32xf16>)
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vlt>}
    ins(%arg0, %arg1 : tensor<16x32xi1>, tensor<16x32xi1>)
    outs(%dst : tensor<16x32xi1>)
    -> tensor<16x32xi1>
  return %ret : tensor<16x32xi1>
}

// -----
// CHECK-LABEL: @test_concat_i1
func.func @test_concat_i1(%arg0: tensor<2048xi1>, %arg1: tensor<2048xi1>) -> tensor<4096xi1> {
  // CHECK: %[[concat:.*]] = tensor.concat dim(0) %[[in1:.*]], %[[in2:.*]] : (tensor<2048xf16>, tensor<2048xf16>) -> tensor<4096xf16>
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<2048xi1>, tensor<2048xi1>) -> tensor<4096xi1>
  return %0 : tensor<4096xi1>
}

// -----
// CHECK-LABEL: @test_transpose_i1
func.func @test_transpose_i1() -> tensor<8x32xi1> {
  %src = tensor.empty() : tensor<32x8xi1>
  %dst = tensor.empty() : tensor<8x32xi1>
  // CHECK: %[[transposed:.*]] = linalg.transpose ins(%[[in1:.*]] : tensor<32x8xf16>) outs(%[[in2:.*]] : tensor<8x32xf16>) permutation = [1, 0]
  %transposed = linalg.transpose ins(%src : tensor<32x8xi1>) outs(%dst : tensor<8x32xi1>) permutation = [1, 0]
  return %transposed : tensor<8x32xi1>
}

// -----
// CHECK-LABEL: func.func @test_normalize_compare_neq_to_Not_eq
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi64>, %[[arg1:.*]]: tensor<1024xi64>, %[[arg2:.*]]: tensor<1024xi1>)
// CHECK: %[[empty:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK: %[[veq:.*]] = hfusion.compare {compare_fn = #hfusion.compare_fn<veq>} ins(%[[arg0]], %[[arg1]] : tensor<1024xi64>, tensor<1024xi64>) outs(%[[empty]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: %[[notOp:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>} ins(%[[veq]] : tensor<1024xi1>) outs(%[[arg2]] : tensor<1024xi1>) -> tensor<1024xi1>
// CHECK: return %[[notOp]]
func.func @test_normalize_compare_neq_to_Not_eq(
  %src1 : tensor<1024xi64>, %src2 : tensor<1024xi64>,  %dst : tensor<1024xi1>) ->  tensor<1024xi1> {
  %ret = hfusion.compare {compare_fn  = #hfusion.compare_fn<vne>}
    ins(%src1, %src2 : tensor<1024xi64>, tensor<1024xi64>)
    outs(%dst : tensor<1024xi1>)
    -> tensor<1024xi1>
  return %ret : tensor<1024xi1>
}

// -----
// CHECK-LABEL: @cast_f32_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @cast_f32_to_i16_with_overflow_mode(%arg0: tensor<16xf32>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----
// CHECK-LABEL: @cast_f32_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_f32_to_i8_with_overflow_mode(%arg0: tensor<16xf32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf32>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_f16_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_f16_to_i8_with_overflow_mode(%arg0: tensor<16xf16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_i64_to_i32_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xi32>) -> tensor<16xi32>
func.func @cast_i64_to_i32_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi32> {
  %0 = tensor.empty() : tensor<16xi32>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi32>) -> tensor<16xi32>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi32>
  return %1 : tensor<16xi32>
}

// -----
// CHECK-LABEL: @cast_i64_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @cast_i64_to_i16_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----
// CHECK-LABEL: @cast_i64_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi64>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_i64_to_i8_with_overflow_mode(%arg0: tensor<16xi64>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi64>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_i32_to_i16_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins({{.*}} : tensor<16xi32>) outs({{.*}} : tensor<16xi16>) -> tensor<16xi16>
func.func @cast_i32_to_i16_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi16> {
  %0 = tensor.empty() : tensor<16xi16>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi16>) -> tensor<16xi16>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi16>
  return %1 : tensor<16xi16>
}

// -----
// CHECK-LABEL: @cast_i32_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi32>) outs({{.*}} : tensor<16xf32>) -> tensor<16xf32>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf32>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_i32_to_i8_with_overflow_mode(%arg0: tensor<16xi32>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi32>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// -----
// CHECK-LABEL: @cast_i16_to_i8_with_overflow_mode(
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xi16>) outs({{.*}} : tensor<16xf16>) -> tensor<16xf16>
// CHECK: hfusion.cast {enable_overflow = false, round_mode = #hfusion.round_mode<trunc>} ins({{.*}} : tensor<16xf16>) outs({{.*}} : tensor<16xi8>) -> tensor<16xi8>
func.func @cast_i16_to_i8_with_overflow_mode(%arg0: tensor<16xi16>) -> tensor<16xi8> {
  %0 = tensor.empty() : tensor<16xi8>
  %1 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<16xi16>) outs(%0 : tensor<16xi8>) -> tensor<16xi8>
  annotation.mark %1 {overflow_mode = "saturate"} : tensor<16xi8>
  return %1 : tensor<16xi8>
}

// CHECK-LABEL: func.func @test_hfusion_muli_i1
// CHECK-SAME: (%[[arg0:.*]]: tensor<32xi1>, %[[arg1:.*]]: tensor<32xi1>)
// CHECK: %[[ZERO:.*]] : tensor<32xi1>
// CHECK: %[[ONE:.*]] = hfusion.elemwise_binary {fun = #hfusion.binary_fn<vand>} ins(%[[arg0:.*]], %[[arg1:.*]]: tensor<32xi1>, tensor<32xi1>) outs(%[[ZERO:.*]] : tensor<32xi1>) -> tensor<32xi1>
// CHECK: return %[[ONE:.*]]
func.func @test_hfusion_muli_i1(%src1: tensor<32xi1>, %src2: tensor<32xi1>) -> tensor<32xi1> {
  %x = tensor.empty() : tensor<32xi1>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%src1, %src2 : tensor<32xi1>, tensor<32xi1>) outs(%x : tensor<32xi1>) -> tensor<32xi1>
  return %1 : tensor<32xi1>
}

// CHECK-LABEL: func.func @test_hfusion_select_cast_i1_to_i16
// CHECK-SAME: %[[arg0:.*]]: tensor<32xi1>
// CHECK: %[[EMPTYI1:.*]] : tensor<32xi1>
// CHECK: %[[EMPTY1:.*]] : tensor<32xi16>
// CHECK: %[[CASTRES1:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg1:.*]] : tensor<32xi1>) outs(%[[EMPTY1:.*]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTY2:.*]] : tensor<32xi16>
// CHECK: %[[CASTRES2:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[arg2:.*]] : tensor<32xi1>) outs(%[[EMPTY2:.*]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[EMPTYI16:.*]] : tensor<32xi16>
// CHECK: %[[CASTRES0:.*]] = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>} ins(%[[EMPTYI1:.*]] : tensor<32xi1>) outs(%[[EMPTYI16:.*]] : tensor<32xi16>) -> tensor<32xi16>
// CHECK: %[[ONE:.*]] = hfusion.select ins(%[[arg0:.*]], %[[CASTRES1:.*]], %[[CASTRES2:.*]] : tensor<32xi1>, tensor<32xi16>, tensor<32xi16>) outs(%[[EMPTYI16:.*]] : tensor<32xi16>) -> tensor<32xi16>
func.func @test_hfusion_select_cast_i1_to_i16(%src0: tensor<32xi1>, %src1: tensor<32xi1>, %src2: tensor<32xi1>) -> tensor<32xi1>{
  %x = tensor.empty() : tensor<32xi1>
  %1 = hfusion.select ins(%src0, %src1, %src2 : tensor<32xi1>, tensor<32xi1>, tensor<32xi1>) outs(%x : tensor<32xi1>) -> tensor<32xi1>
  return %1 : tensor<32xi1>
}
