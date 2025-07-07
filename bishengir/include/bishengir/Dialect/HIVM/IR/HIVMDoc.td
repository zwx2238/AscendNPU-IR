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
 * \file HIVMDoc.td
 * \brief HIVM documentation
 * \details This documentation files exists to circumvent
 *          limitations on mixing different .td files in cases
 *          one does not want to have all ops belong to the same
 *          logical unit. This file should only include other .td files
 *          only and be used for the purpose of generating documentation.
 */

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD

// Ops
include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMDMAOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMMacroOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td"
include "bishengir/Dialect/HIVM/IR/HIVMVectorOps.td"

// Attributes
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMDOC_TD
