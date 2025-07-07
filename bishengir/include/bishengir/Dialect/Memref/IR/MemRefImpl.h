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
 * \file MemRefImpl.h
 * \brief MemRef Dialect Implementation
 */

#ifndef BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H
#define BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace memref {

/// Create memref.alloc op with the same type as source
Value createMemRefAllocOp(OpBuilder &builder, Location loc, Value source);

/// Create memref.alloc op with the same shape as source
/// but with element type targetElemType
Value createMemRefAllocOpWithTargetElemType(OpBuilder &builder, Location loc,
                                            Value source, Type targetElemType);

} // namespace memref
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_IR_MEMREFIMPL_H
