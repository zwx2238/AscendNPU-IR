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
 * \file HFusion.h
 * \brief Hybrid Fusion dialect
 */

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H

#include "bishengir/Interfaces/AggregatedOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

namespace mlir {
namespace hfusion {
std::string generateLibraryCallName(Operation *op);
} // namespace hfusion
} // namespace mlir

//===----------------------------------------------------------------------===//
// HFusion Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusionOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// HFusion Enums
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HFusion/IR/HFusionEnums.h.inc"

//===----------------------------------------------------------------------===//
// HFusion Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionAttrs.h.inc"

//===----------------------------------------------------------------------===//
// HFusion Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionOps.h.inc"

#define GET_OP_CLASSES
#include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.h.inc"

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSION_H
