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
 * \file Annotation.h
 * \brief Annotation dialect
 */

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATION_H
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATION_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Annotation Dialect
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/AnnotationOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Annotation Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "bishengir/Dialect/Annotation/IR/AnnotationOps.h.inc"

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATION_H
