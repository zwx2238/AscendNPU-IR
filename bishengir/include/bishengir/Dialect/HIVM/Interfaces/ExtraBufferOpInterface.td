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
 * \file ExtraBufferOpInterface.td
 * \brief Ops with Temp. Buffer
 * \details This is the definition file for the ExtraBufferOpInterface.
 */

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ExtraBufferOpInterface : OpInterface<"ExtraBufferOpInterface"> {
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Allocate extra temp buffers for the current operation.
      }],
      /*retTy=*/"llvm::LogicalResult",
      /*methodName=*/"allocExtraBuffersIfPossible",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the extra temp buffers.
      }],
      /*retTy=*/"mlir::OperandRange",
      /*methodName=*/"getExtraBuffers",
      /*args=*/(ins)
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_EXTRABUFFEROPINTERFACE_TD
