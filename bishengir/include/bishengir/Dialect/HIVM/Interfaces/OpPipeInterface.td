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
 * \file OpPipeInterface.td
 * \brief Ops with Pipe Information
 * \details This is the definition file for the OpPipeInterface.
 */

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_OPPIPEINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_OPPIPEINTERFACE_TD

include "mlir/IR/OpBase.td"

def OpPipeInterface : OpInterface<"OpPipeInterface"> {
  let description = [{
   HIVM Op has three kinds of op:
     1. op with single pipe.
     2. macro op with input pipe and output pipe.
     3. op with no pipe attribute.
  }];

  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/"judge if it is op with single pipe.",
      /*retTy=*/"bool",
      /*methodName=*/"isSinglePipeOp",
      /*args=*/(ins),
      /*methodBody=*/[{
        return ConcreteOp::template hasTrait<OpTrait::SinglePipeOpTrait>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"judge if it is compound macro op.",
      /*retTy=*/"bool",
      /*methodName=*/"isMacroOp",
      /*args=*/(ins),
      /*methodBody=*/[{
        return ConcreteOp::template hasTrait<OpTrait::MacroOpTrait>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"get op pipe if it is op with single pipe.",
      /*retTy=*/"mlir::hivm::PIPE",
      /*methodName=*/"getPipe",
      /*args=*/(ins),
      /*methodBody=*/[{
        if constexpr(ConcreteOp::template hasTrait<OpTrait::SinglePipeOpTrait>()) {
          return $_op.getPipe();
        } else {
          return mlir::hivm::PIPE::PIPE_UNASSIGNED;
        }
      }]
    >,
    InterfaceMethod<
      /*desc=*/"get input pipe if it is macro op.",
      /*retTy=*/"mlir::hivm::PIPE",
      /*methodName=*/"getInPipe",
      /*args=*/(ins),
      /*methodBody=*/[{
        if constexpr(ConcreteOp::template hasTrait<OpTrait::MacroOpTrait>()) {
          return $_op.getInPipe();
        } else {
          return mlir::hivm::PIPE::PIPE_UNASSIGNED;
        }
      }]
    >,
    InterfaceMethod<
      /*desc=*/"get output pipe if it is macro op.",
      /*retTy=*/"mlir::hivm::PIPE",
      /*methodName=*/"getOutPipe",
      /*args=*/(ins),
      /*methodBody=*/[{
        if constexpr(ConcreteOp::template hasTrait<OpTrait::MacroOpTrait>()) {
          return $_op.getOutPipe();
        } else {
          return mlir::hivm::PIPE::PIPE_UNASSIGNED;
        }
      }]
    >,
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_OPPIPEINTERFACE_TD
