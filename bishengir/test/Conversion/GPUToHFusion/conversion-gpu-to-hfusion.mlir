// RUN: bishengir-opt -convert-gpu-to-hfusion %s | FileCheck %s

module {
  func.func @test_barrier() {
    gpu.barrier
    return
  }
  // CHECK: hfusion.barrier
}
