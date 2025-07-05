//===- HIVMPasses.td - Conversion pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_PASSES_H
#define BISHENGIR_CONVERSION_PASSES_H

include "mlir/Pass/PassBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"

//===----------------------------------------------------------------------===//
// ArithToHFusion
//===----------------------------------------------------------------------===//

def ConvertArithToHFusion : Pass<"convert-arith-to-hfusion"> {
  let summary = "Convert arith ops to HFusion ops";
  let constructor = "mlir::createArithToHFusionConversionPass()";
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect", "hfusion::HFusionDialect"];
}

//===----------------------------------------------------------------------===//
// MathToHFusion
//===----------------------------------------------------------------------===//

def ConvertMathToHFusion : Pass<"convert-math-to-hfusion"> {
  let summary = "Convert math ops to HFusion ops";
  let constructor = "mlir::createMathToHFusionConversionPass()";
  let dependentDialects = ["linalg::LinalgDialect", "hfusion::HFusionDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToHFusion
//===----------------------------------------------------------------------===//

def ConvertLinalgToHFusion : Pass<"convert-linalg-to-hfusion"> {
  let summary = "Convert linalg ops to HFusion ops";
  let constructor = "mlir::createLinalgToHFusionConversionPass()";
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect", "hfusion::HFusionDialect", "bufferization::BufferizationDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToHFusion
//===----------------------------------------------------------------------===//

def ConvertGPUToHFusion : Pass<"convert-gpu-to-hfusion"> {
  let summary = "Convert GPU ops to HFusion ops";
  let constructor = "mlir::createGPUToHFusionConversionPass()";
  let dependentDialects = ["gpu::GPUDialect", "hfusion::HFusionDialect"];
}

//===----------------------------------------------------------------------===//
// TensorToHFusion
//===----------------------------------------------------------------------===//

def ConvertTensorToHFusion : Pass<"convert-tensor-to-hfusion"> {
  let summary = "Convert tensor ops to Linalg/HFusion ops";
  let constructor = "mlir::createTensorToHFusionConversionPass()";
  let dependentDialects = ["tensor::TensorDialect", "hfusion::HFusionDialect",
                          "linalg::LinalgDialect"];
}

#endif // BISHENGIR_CONVERSION_PASSES_H
