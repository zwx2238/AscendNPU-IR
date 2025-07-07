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
 * \file MathExt.h
 * \brief Extended math dialect.
 */

#ifndef BISHENGIR_DIALECT_MATH_IR_MATHEXT_H
#define BISHENGIR_DIALECT_MATH_IR_MATHEXT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/Dialect/Math/IR/Math.h"

//===----------------------------------------------------------------------===//
// Math Ext Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/MathExt/IR/MathExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Math Ext Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/MathExt/IR/MathExtOps.h.inc"

#endif // BISHENGIR_DIALECT_MATH_IR_MATHEXT_H
