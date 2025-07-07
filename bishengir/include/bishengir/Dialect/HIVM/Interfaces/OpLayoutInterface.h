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
 * \file OpLayoutInterface.h
 * \brief Ops with Layout Information
 * \details This is the definition file for the OpLayoutInterface.
 */

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace hivm {
/// Forward declaration.
class DataLayoutAttr;
} // namespace hivm
} // namespace mlir

// Include the generated interface declarations.
#include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.h.inc"

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_H
