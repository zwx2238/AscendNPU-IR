//===- bishengir-minimal-opt.cpp --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/InitAllDialects.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

/// This test includes the minimal amount of components for mlir-opt with
/// bishengir extension, that is the CoreIR, the printer/parser, the bytecode
/// reader/writer, the passmanagement infrastructure and all the
/// instrumentation.
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  bishengir::registerAllDialects(registry);
  mlir::registerAllDialects(registry);
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Minimal Standalone optimizer driver with bishengir \n",
      registry));
}
