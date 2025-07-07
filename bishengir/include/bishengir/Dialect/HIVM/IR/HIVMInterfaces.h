/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file HIVMInterfaces.h
 * \brief HIVM dialect interface definitions
 */

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
#define BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H

#include "bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/ImplByScalarOpInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h"
#include "bishengir/Dialect/HIVM/Interfaces/OpPipeInterface.h"
#include "bishengir/Interfaces/AggregatedOpInterface.h"

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hivm {
/// Forward declarations.
class HIVMStructuredOp;
enum class TCoreType : uint32_t;
enum class IteratorType : uint32_t;
enum class AddressSpace : uint32_t;

namespace detail {

std::optional<TCoreType> queryCoreTypeHelper(Operation *op);

/// Return positions in `iteratorTypes` that match `iteratorTypeName`.
inline void findPositionsOfType(ArrayRef<IteratorType> iteratorTypes,
                                IteratorType iteratorTypeName,
                                SmallVectorImpl<int64_t> &res) {
  for (const auto &en : llvm::enumerate(iteratorTypes)) {
    if (en.value() == iteratorTypeName)
      res.push_back(en.index());
  }
}

/// Implementation to get HIVM Structured Op's memory effects.
void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    HIVMStructuredOp hivmOp);

// Implementation to get operands with or without extra buffer.
SmallVector<OpOperand *> getHIVMOperandsImpl(Operation *op,
                                             bool includeExtraBuffer = false);

// Implementation to get operand types with or without extra buffer.
SmallVector<Type> getHIVMOperandTypesImpl(Operation *op,
                                          bool includeExtraBuffer = false);

// Return mask of continuous axes of an operation
BitVector getContiguousAxesImpl(Operation *op);

// Implementation to get input operands with or without extra buffer.
SmallVector<OpOperand *>
getHIVMInputOperandsImpl(Operation *op, bool includeExtraBuffer = false);

/// Get Operands with target memref space
SmallVector<Value> getTargetSpaceOperandsImpl(Operation *op,
                                              AddressSpace hivmSpace,
                                              bool includeExtraBuffer);

// check if the operand is vector only at a specific index
bool isVectorOnlyOperandImpl(Operation *op, size_t idx);

} // namespace detail
} // namespace hivm
} // namespace mlir

#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMINTERFACES_H
