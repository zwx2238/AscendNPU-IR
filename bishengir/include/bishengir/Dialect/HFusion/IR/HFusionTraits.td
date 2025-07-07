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
 * \file HFusionTraits.td
 * \brief Traits used by HFusion ops.
 */

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONTRAITS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONTRAITS_TD

include "mlir/IR/OpBase.td"

/// Currently only attached to manually defined structured ops. However this may
/// be attached to HFusionStructuredBase_Op if suitable.
def ResultOnlyIfTensor
    : PredOpTrait<"Result must only exist if using tensor semantics",
                  CPred<"($_op.getNumResults() == 0) ^ "
                        "(cast<DestinationStyleOpInterface>(&$_op)."
                        "hasPureTensorSemantics())">>;

/// Check to make sure that `attr` is less than the rank of `shaped`. Used for
/// GatherOp.
class AttrWithinRankOf<string attr, string shaped>
    : PredOpTrait<attr#" must be less than rank of "#shaped,
                  CPred<"$" # attr # " < cast<ShapedType>($" # shaped #
                        ".getType()).getRank()">>;

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONTRAITS_TD
