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
 * \file OpLayoutInterface.td
 * \brief Ops with Layout Information
 * \details This is the definition file for the OpLayoutInterface.
 */

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD

include "mlir/IR/OpBase.td"

def OpLayoutInterface : OpInterface<"OpWithLayoutInterface"> {
  let description = [{
     Interface to query characteristics of HIVM Ops that have specific
     data layout requirements.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Returns the target data layouts for memref-typed operands.",
    /*retTy=*/"mlir::SmallVector<mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsTargetLayout",
    /*args=*/(ins)
    >,
    InterfaceMethod<
    /*desc=*/"Returns the current data layouts for memref-typed operands.",
    /*retTy=*/"mlir::SmallVector<mlir::hivm::DataLayoutAttr>",
    /*methodName=*/"getOperandsCurrentLayout",
    /*args=*/(ins)
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPLAYOUTINTERFACE_TD
