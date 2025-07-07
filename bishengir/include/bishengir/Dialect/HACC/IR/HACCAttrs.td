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
 * \file HACCAttrs.td
 * \brief HACC attributes definitions
 * \details This is the definition file for enumeration and attributes
 *          used in heterogeneous async computing call operations.
 */

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD

include "bishengir/Dialect/HACC/IR/HACCBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

//===----------------------------------------------------------------------===//
// HACC Enums and Attributes
//===----------------------------------------------------------------------===//

class HACC_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hacc";
}

class HACC_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HACC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Function Type
//===----------------------------------------------------------------------===//

def HACC_HOST : I32EnumAttrCase<"HOST", 1>;
def HACC_DEVICE : I32EnumAttrCase<"DEVICE", 2>;
def HACC_FuncTypeEnum : HACC_I32Enum<"HACCFuncType", "HACC Function Category", [
  HACC_HOST,
  HACC_DEVICE
]> {}

def HACC_FuncTypeAttr : HACC_Attr<"HACCFuncType", "function_kind"> {
  let parameters = (ins EnumParameter<HACC_FuncTypeEnum>:$function_kind);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HACC function type.
  }];
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCATTRS_TD
