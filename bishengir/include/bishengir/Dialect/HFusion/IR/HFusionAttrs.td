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
 * \file HFusionAttrs.td
 * \brief HFusion dialect base support
 * \details This is the definition file for enumeration and attributes
 *          used in hybrid fusion operations.
 */

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD

include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// HFusion Attributes.
//===----------------------------------------------------------------------===//

class HFusion_Attr<string attrName, string attrMnemonic, list<Trait> traits = [
]> : AttrDef<HFusion_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def HFusion_ReduceWithIndexOpAttr : HFusion_Attr<"ReduceWithIndexKind", "reduce_with_index_kind"> {
  let parameters = (ins
    EnumParameter<ReduceWithIndexKind>:$reduce_with_index_kind
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    The kind of reduce with index.
  }];
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONATTRS_TD
