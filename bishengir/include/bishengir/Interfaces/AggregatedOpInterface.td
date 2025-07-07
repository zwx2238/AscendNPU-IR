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
 * \file AggregatedOpInterface.td
 * \brief Ops to decompose
 * \details This is the definition file for the BiShengIR
 *          custom AggregatedOpInterface.
 */

#ifndef BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD
#define BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD

include "mlir/IR/OpBase.td"

def BiShengIRAggregatedOpInterface : OpInterface<"BiShengIRAggregatedOpInterface"> {
  let description = [{
    Interface for decomposing aggregated operations into a sequence of simpler
    ops.
  }];
  let cppNamespace = "::bishengir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to decompose the operation into simpler operations.
      }],
      /*retType=*/"mlir::FailureOr<llvm::SmallVector<mlir::Value>>",
      /*methodName=*/"decomposeOperation",
      /*args=*/(ins
          "mlir::OpBuilder &":$b),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the phase for decomposing the operation.
      }],
      /*retType=*/"::bishengir::DecomposePhase",
      /*methodName=*/"getDecomposePhase",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::bishengir::DecomposePhase::NO_CONSTRAINT;
      }]
    >
  ];
}

#endif // BISHENGIR_DIALECT_BISHENGIR_INTERFACES_AGGREGATEDOPINTERFACE_TD
