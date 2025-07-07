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
 * \file BiShengIRAttrs.td
 * \brief BiShengIR general attr defs.
 * \details This is the definition file for BiShengIR general attributes.
 */

#ifndef BISHENGIR_INTERFACES_BISHENGIRATTRS_TD
#define BISHENGIR_INTERFACES_BISHENGIRATTRS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// BiShengIR general Enums and Attributes.
//===----------------------------------------------------------------------===//

class BiShengIR_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::bishengir";
}

// Decompose phase for AggregateOpInterface
def BiShengIR_DecomposePhase_Enum : BiShengIR_I32Enum<
  "DecomposePhase", "BiShengIR AggregateOpInterface Decompose Phase", [
    I32EnumAttrCase<"NO_CONSTRAINT", 0>,
    I32EnumAttrCase<"AFTER_HFUSION_FLATTEN", 1>,
    I32EnumAttrCase<"BEFORE_HIVM_STRIDE_ALIGNMENT", 2>,
    I32EnumAttrCase<"AFTER_HIVM_STRIDE_ALIGNMENT", 3>
]>;

#endif //BISHENGIR_INTERFACES_BISHENGIRATTRS_TD
