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
 * \file ImplByScalarOpInterface.td
 * \brief Ops to decompose to scalar
 * \details This is the definition file for the ImplByScalarOpInterface.
 */

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ImplByScalarOpInterface : OpInterface<"ImplByScalarOpInterface"> {
  let description = [{
    Interface for operations that needs to be lowered to scalar loops.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{ 
        Method to lower the operation to scalar loops.
      }],
      /*retType=*/"mlir::FailureOr<mlir::SmallVector<mlir::Value>>",
      /*methodName=*/"lowerToLoops",
      /*args=*/(ins
          "RewriterBase &":$b),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the operation should be lowered to scalar loops.
      }],
      /*retType=*/"bool",
      /*methodName=*/"shouldLowerToScalarLoops",
      /*args=*/(ins),
      /*methodBody=*/""
    >
  ];
}
#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD

