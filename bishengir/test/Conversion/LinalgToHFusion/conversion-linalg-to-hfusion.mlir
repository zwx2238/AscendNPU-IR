// RUN: bishengir-opt -convert-linalg-to-hfusion %s | FileCheck %s

module {
  func.func private @__hmf_reluDh(f16) -> f16 attributes {llvm.readnone}
  func.func @test_relu(%arg0 : tensor<6x6xf16>) -> tensor<6x6xf16> {
    // CHECK:       %[[RET:.*]] = hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}
    %ret = linalg.map { func.call {callee = @__hmf_reluDh} } ins(%arg0 : tensor<6x6xf16>) outs(%arg0 : tensor<6x6xf16>)
    return %ret : tensor<6x6xf16>
  }
}
