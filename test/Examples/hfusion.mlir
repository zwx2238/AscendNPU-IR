// RUN: bishengir-minimal-opt %s | FileCheck %s

// CHECK: test
module {
  func.func @test(%arg0 : tensor<16xf16>) -> tensor<16xf32> {
    %0 = tensor.empty() : tensor<16xf32>
    %1 = hfusion.cast ins(%arg0 : tensor<16xf16>) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    return %1 : tensor<16xf32>
  }
}