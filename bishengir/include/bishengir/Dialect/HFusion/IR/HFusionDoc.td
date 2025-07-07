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
 * \file HFusionDoc.td
 * \brief HFusion documentation
 * \details This documentation files exists to circumvent
 *          limitations on mixing different .td files in cases
 *          one does not want to have all ops belong to the same
 *          logical unit. This file should only include other .td files
 *          only and be used for the purpose of generating documentation.
 */

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONDOC_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONDOC_TD

// Ops
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionOps.td"
include "bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td"

// Attributes
include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONDOC_TD
