// RUN: bishengir-opt %s --bubble-up-extract-slice --split-input-file --cse --canonicalize | FileCheck %s
// RUN: bishengir-opt %s --bubble-up-extract-slice="aggressive=true" --split-input-file --cse --canonicalize | FileCheck %s -check-prefix=CHECK-AGGRESSIVE

// CHECK-LABEL: @reduce_example(
// CHECK: reduce
// CHECK-NOT: extract_slice
// CHECK: return
module {
  func.func @reduce_example(%arg0: tensor<5x4xf32>) -> tensor<2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4xf32>
    %reduced = linalg.reduce ins(%arg0 : tensor<5x4xf32>) outs(%0 : tensor<4xf32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %1 = arith.addf %in, %init : f32
        linalg.yield %1 : f32
      }
    %extracted_slice = tensor.extract_slice %reduced[1] [2] [1] : tensor<4xf32> to tensor<2xf32>
    return %extracted_slice : tensor<2xf32>
  }
}

// -----
// CHECK-LABEL: @broadcast_example(
// CHECK: broadcast
// CHECK-NOT: extract_slice
// CHECK: return
module {
  func.func @broadcast_example(%arg0: tensor<5x4xf32>) -> tensor<4x2x1x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<8x5x4x6xf32>
    %broadcasted = linalg.broadcast ins(%arg0 : tensor<5x4xf32>) outs(%0 : tensor<8x5x4x6xf32>) dimensions = [0, 3]
    %extracted_slice = tensor.extract_slice %broadcasted[0, 1, 0, 0] [4, 2, 1, 3] [1, 1, 1, 1] : tensor<8x5x4x6xf32> to tensor<4x2x1x3xf32>
    return %extracted_slice : tensor<4x2x1x3xf32>
  }
}
// -----
// CHECK-LABEL: @mlir_fused_mul_sum_119(
// CHECK: broadcast
// CHECK-SAME: tensor<1x256x128xf32>
// CHECK-SAME: tensor<1x24x256x128xf32>
// CHECK-SAME: dimensions
// CHECK-SAME: 1
// CHECK: return
func.func @mlir_fused_mul_sum_119(%arg0: tensor<1x24x333x128xbf16>, %arg1: tensor<333x128xf32>, %arg2: tensor<1x24x333x64x2xf32>, %arg3: tensor<1x24x256x128xbf16>, %arg4: tensor<1x24x256x1xf32>, %arg5: tensor<1x24x333x128xbf16>, %arg6: tensor<1x24x333x64x2xf32>, %arg7: tensor<1x24x256x128xbf16>, %arg8: tensor<1x24x256x1xf32>) -> (tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x24x333x128xf32>
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] output_shape [1, 333, 128] : tensor<333x128xf32> into tensor<1x333x128xf32>
  %broadcasted = linalg.broadcast ins(%expanded : tensor<1x333x128xf32>) outs(%0 : tensor<1x24x333x128xf32>) dimensions = [1]
  %collapsed = tensor.collapse_shape %arg2 [[0], [1], [2], [3, 4]] : tensor<1x24x333x64x2xf32> into tensor<1x24x333x128xf32>
  %extracted_slice = tensor.extract_slice %arg0[0, 0, 77, 0] [1, 24, 256, 128] [1, 1, 1, 1] : tensor<1x24x333x128xbf16> to tensor<1x24x256x128xbf16>
  %extracted_slice_0 = tensor.extract_slice %0[0, 0, 77, 0] [1, 24, 256, 128] [1, 1, 1, 1] : tensor<1x24x333x128xf32> to tensor<1x24x256x128xf32>
  %1 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice : tensor<1x24x256x128xbf16>) outs(%extracted_slice_0 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %extracted_slice_1 = tensor.extract_slice %broadcasted[0, 0, 77, 0] [1, 24, 256, 128] [1, 1, 1, 1] : tensor<1x24x333x128xf32> to tensor<1x24x256x128xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%1, %extracted_slice_1 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%extracted_slice_0 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %extracted_slice_2 = tensor.extract_slice %collapsed[0, 0, 77, 0] [1, 24, 256, 128] [1, 1, 1, 1] : tensor<1x24x333x128xf32> to tensor<1x24x256x128xf32>
  %3 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%2, %extracted_slice_2 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%extracted_slice_0 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %4 = tensor.empty() : tensor<1x24x256x128xf32>
  %5 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg3 : tensor<1x24x256x128xbf16>) outs(%4 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %collapsed_3 = tensor.collapse_shape %arg4 [[0], [1], [2, 3]] : tensor<1x24x256x1xf32> into tensor<1x24x256xf32>
  %broadcasted_4 = linalg.broadcast ins(%collapsed_3 : tensor<1x24x256xf32>) outs(%4 : tensor<1x24x256x128xf32>) dimensions = [3]
  %6 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%5, %broadcasted_4 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%4 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %7 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%3, %6 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%4 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %8 = tensor.empty() : tensor<128xf32>
  %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<128xf32>) -> tensor<128xf32>
  %reduced = linalg.reduce ins(%7 : tensor<1x24x256x128xf32>) outs(%9 : tensor<128xf32>) dimensions = [0, 1, 2]
    (%in: f32, %init: f32) {
      %16 = arith.addf %in, %init : f32
      linalg.yield %16 : f32
    }
  %expanded_5 = tensor.expand_shape %reduced [[0, 1, 2, 3]] output_shape [1, 1, 1, 128] : tensor<128xf32> into tensor<1x1x1x128xf32>
  %collapsed_6 = tensor.collapse_shape %arg6 [[0], [1], [2], [3, 4]] : tensor<1x24x333x64x2xf32> into tensor<1x24x333x128xf32>
  %extracted_slice_7 = tensor.extract_slice %arg5[0, 0, 77, 0] [1, 24, 256, 128] [1, 1, 1, 1] : tensor<1x24x333x128xbf16> to tensor<1x24x256x128xbf16>
  %10 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%extracted_slice_7 : tensor<1x24x256x128xbf16>) outs(%extracted_slice_0 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %11 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%10, %extracted_slice_1 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%extracted_slice_0 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %extracted_slice_8 = tensor.extract_slice %collapsed_6[0, 0, 77, 0] [1, 24, 256, 128] [1, 1, 1, 1] : tensor<1x24x333x128xf32> to tensor<1x24x256x128xf32>
  %12 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%11, %extracted_slice_8 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%extracted_slice_0 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %13 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%arg7 : tensor<1x24x256x128xbf16>) outs(%4 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %collapsed_9 = tensor.collapse_shape %arg8 [[0], [1], [2, 3]] : tensor<1x24x256x1xf32> into tensor<1x24x256xf32>
  %broadcasted_10 = linalg.broadcast ins(%collapsed_9 : tensor<1x24x256xf32>) outs(%4 : tensor<1x24x256x128xf32>) dimensions = [3]
  %14 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%13, %broadcasted_10 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%4 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %15 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%12, %14 : tensor<1x24x256x128xf32>, tensor<1x24x256x128xf32>) outs(%4 : tensor<1x24x256x128xf32>) -> tensor<1x24x256x128xf32>
  %reduced_11 = linalg.reduce ins(%15 : tensor<1x24x256x128xf32>) outs(%9 : tensor<128xf32>) dimensions = [0, 1, 2]
    (%in: f32, %init: f32) {
      %16 = arith.addf %in, %init : f32
      linalg.yield %16 : f32
    }
  %expanded_12 = tensor.expand_shape %reduced_11 [[0, 1, 2, 3]] output_shape [1, 1, 1, 128] : tensor<128xf32> into tensor<1x1x1x128xf32>
  return %expanded_5, %expanded_12 : tensor<1x1x1x128xf32>, tensor<1x1x1x128xf32>
}

// -----
// CHECK-LABEL: @aggressive_bubble_up_unaligned_extract_slice(
// CHECK: %[[cast0:.*]] = hfusion.cast
// CHECK: tensor.extract_slice %[[cast0]]
// CHECK: tensor.extract_slice %[[cast0]]
// CHECK-AGGRESSIVE-LABEL: @aggressive_bubble_up_unaligned_extract_slice(
// CHECK-AGGRESSIVE: %[[load0:.*]] = hfusion.load
// CHECK-AGGRESSIVE: %[[cast0:.*]] = hfusion.cast {{.*}} ins(%[[load0]] : tensor<2047xi64>)
// CHECK-AGGRESSIVE: %[[slice0:.*]] = tensor.extract_slice {{.*}}{{\[}}2046] {{\[}}1]
// CHECK-AGGRESSIVE: %[[load1:.*]] = hfusion.load ins(%[[slice0]] : tensor<1xi64>)
// CHECK-AGGRESSIVE: hfusion.cast {{.*}} ins(%[[load1]] : tensor<1xi64>)
// CHECK-AGGRESSIVE: %[[slice1:.*]] = tensor.extract_slice {{.*}}{{\[}}1] {{\[}}2046]
// CHECK-AGGRESSIVE: %[[load2:.*]] = hfusion.load ins(%[[slice1]] : tensor<2046xi64>)
// CHECK-AGGRESSIVE: hfusion.cast {{.*}} ins(%[[load2]] : tensor<2046xi64>)
func.func @aggressive_bubble_up_unaligned_extract_slice(%arg0: tensor<1x2047xi64>, %arg1: tensor<2047x2047xi32>) -> tensor<2047x2047xi32> {
  %0 = tensor.empty() : tensor<2047xi32>
  %1 = tensor.empty() : tensor<2047x2047xi32>
  %2 = tensor.empty() : tensor<2047xi64>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2047xi64> into tensor<2047xi64>
  %3 = hfusion.load ins(%collapsed : tensor<2047xi64>) outs(%2 : tensor<2047xi64>) -> tensor<2047xi64>
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<2047xi64>) outs(%0 : tensor<2047xi32>) -> tensor<2047xi32>
  %extracted_slice = tensor.extract_slice %4[2046] [1] [1] : tensor<2047xi32> to tensor<1xi32>
  %extracted_slice_0 = tensor.extract_slice %4[1] [2046] [1] : tensor<2047xi32> to tensor<2046xi32>
  %concat = tensor.concat dim(0) %extracted_slice_0, %extracted_slice : (tensor<2046xi32>, tensor<1xi32>) -> tensor<2047xi32>
  %broadcasted = linalg.broadcast ins(%concat : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [1]
  %broadcasted_1 = linalg.broadcast ins(%4 : tensor<2047xi32>) outs(%1 : tensor<2047x2047xi32>) dimensions = [0]
  %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<sub>} ins(%broadcasted, %broadcasted_1 : tensor<2047x2047xi32>, tensor<2047x2047xi32>) outs(%1 : tensor<2047x2047xi32>) -> tensor<2047x2047xi32>
  %6 = hfusion.store ins(%5 : tensor<2047x2047xi32>) outs(%arg1 : tensor<2047x2047xi32>) -> tensor<2047x2047xi32>
  return %6 : tensor<2047x2047xi32>
}

// -----
// CHECK-AGGRESSIVE-LABEL: @not_bubble_up_aligned_extract_slice(
// CHECK-AGGRESSIVE: %[[cast0:.*]] = hfusion.cast
// CHECK-AGGRESSIVE: tensor.extract_slice %[[cast0]]
// CHECK-AGGRESSIVE: tensor.extract_slice %[[cast0]]
func.func @not_bubble_up_aligned_extract_slice(%arg0: tensor<1x2047xi64>, %arg1: tensor<2048x2048xi32>) -> tensor<2048x2048xi32> {
  %0 = tensor.empty() : tensor<2047xi32>
  %1 = tensor.empty() : tensor<2048x2048xi32>
  %2 = tensor.empty() : tensor<2047xi64>
  %collapsed = tensor.collapse_shape %arg0 [[0, 1]] : tensor<1x2047xi64> into tensor<2047xi64>
  %3 = hfusion.load ins(%collapsed : tensor<2047xi64>) outs(%2 : tensor<2047xi64>) -> tensor<2047xi64>
  %4 = hfusion.cast {round_mode = #hfusion.round_mode<rint>} ins(%3 : tensor<2047xi64>) outs(%0 : tensor<2047xi32>) -> tensor<2047xi32>
  %extracted_slice = tensor.extract_slice %4[32] [1024] [1] : tensor<2047xi32> to tensor<1024xi32>
  %extracted_slice_0 = tensor.extract_slice %4[64] [1024] [1] : tensor<2047xi32> to tensor<1024xi32>
  %concat = tensor.concat dim(0) %extracted_slice_0, %extracted_slice : (tensor<1024xi32>, tensor<1024xi32>) -> tensor<2048xi32>
  %broadcasted = linalg.broadcast ins(%concat : tensor<2048xi32>) outs(%1 : tensor<2048x2048xi32>) dimensions = [1]
  %6 = hfusion.store ins(%broadcasted : tensor<2048x2048xi32>) outs(%arg1 : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  return %6 : tensor<2048x2048xi32>
}


// -----
// CHECK-AGGRESSIVE-LABEL: @bubble_up_dynamic(
// CHECK-AGGRESSIVE: %{{.*}} = linalg.reduce ins(%{{.*}} : tensor<?x1x32x128xf32>) outs(%{{.*}} : tensor<?x1x32xf32>) dimensions = [3]
// CHECK-AGGRESSIVE: %{{.*}} = linalg.broadcast ins(%{{.*}} : tensor<?x1x32xf32>) outs(%{{.*}} : tensor<?x1x32x128xf32>) dimensions = [3]
func.func @bubble_up_dynamic(%arg0: tensor<1x?x12288xf32>, %arg1: tensor<128xbf16>, %arg2: i64, %arg3: tensor<128xbf16>, %arg4: tensor<?x32x128xf32>, %arg5: tensor<?x1x32x128xf32>, %arg6: tensor<?x1x32x128xf32>, %arg7: tensor<?x1x32x128xf32>, %arg8: index, %arg9: index, %arg10: tensor<?x1x32x128xf32>, %arg11: tensor<?x1x32xf32>, %arg12: tensor<?x1x32xf32>, %arg13: f32, %arg14: f32, %arg15: f32) -> tensor<?x1x32x128xf32> attributes {enable_auto_mark_buffer_size, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hfusion.fusion_kind = #hfusion.fusion_kind<LAST_AXIS_PBR>} {
    %reduced = linalg.reduce ins(%arg5 : tensor<?x1x32x128xf32>) outs(%arg11 : tensor<?x1x32xf32>) dimensions = [3]
      (%in: f32, %init: f32) {
        %6 = arith.addf %in, %init : f32
        linalg.yield %6 : f32
      }
    %0 = linalg.elemwise_binary {fun = #linalg.binary_fn<div>} ins(%reduced, %arg14 : tensor<?x1x32xf32>, f32) outs(%arg12 : tensor<?x1x32xf32>) -> tensor<?x1x32xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg13 : tensor<?x1x32xf32>, f32) outs(%arg12 : tensor<?x1x32xf32>) -> tensor<?x1x32xf32>
    %2 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>} ins(%1 : tensor<?x1x32xf32>) outs(%arg12 : tensor<?x1x32xf32>) -> tensor<?x1x32xf32>
    %3 = hfusion.elemwise_unary {fun = #hfusion.unary_fn<rec>} ins(%2 : tensor<?x1x32xf32>) outs(%arg12 : tensor<?x1x32xf32>) -> tensor<?x1x32xf32>
    %broadcasted = linalg.broadcast ins(%3 : tensor<?x1x32xf32>) outs(%arg10 : tensor<?x1x32x128xf32>) dimensions = [3]
    %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%arg6, %broadcasted : tensor<?x1x32x128xf32>, tensor<?x1x32x128xf32>) outs(%arg10 : tensor<?x1x32x128xf32>) -> tensor<?x1x32x128xf32>
    %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} ins(%4, %arg7 : tensor<?x1x32x128xf32>, tensor<?x1x32x128xf32>) outs(%arg10 : tensor<?x1x32x128xf32>) -> tensor<?x1x32x128xf32>
    %extracted_slice = tensor.extract_slice %5[%arg9, 0, 0, 0] [%arg8, 1, 32, 128] [1, 1, 1, 1] : tensor<?x1x32x128xf32> to tensor<?x1x32x128xf32>
    return %extracted_slice : tensor<?x1x32x128xf32>
}
