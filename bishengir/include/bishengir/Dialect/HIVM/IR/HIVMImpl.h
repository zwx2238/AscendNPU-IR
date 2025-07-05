//===- HIVMImpl.h - HIVM implementation -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace hivm {

/// get operation core type
FailureOr<TCoreType> getCoreType(Operation *op);

} // namespace hivm
} // namespace mlir

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMIMPL_H