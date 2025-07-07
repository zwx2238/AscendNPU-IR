// RUN: bishengir-opt -convert-arith-to-hfusion %s | FileCheck %s

module {
  func.func @test_addf(%arg0 : tensor<6x6xf32>, %arg1 : tensor<6x6xf32>) -> tensor<6x6xf32> {
    // CHECK:       %[[EMPTY:.*]] = tensor.empty()
    // CHECK:       %[[RET:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    %ret = arith.addf %arg0, %arg1 : tensor<6x6xf32>
    return %ret : tensor<6x6xf32>
  }
}
