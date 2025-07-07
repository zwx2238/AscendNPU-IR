// RUN: bishengir-opt -convert-math-to-hfusion %s | FileCheck %s

module {
  func.func @test_exp(%arg0 : tensor<6x6xf32>) -> tensor<6x6xf32> {
    // CHECK:       %[[EMPTY:.*]] = tensor.empty()
    // CHECK:       %[[RET:.*]] = linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}
    %ret = math.exp %arg0 : tensor<6x6xf32>
    return %ret : tensor<6x6xf32>
  }
}
