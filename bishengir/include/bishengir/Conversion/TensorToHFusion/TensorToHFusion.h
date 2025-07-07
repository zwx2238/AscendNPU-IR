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
 * \file TensorToHFusion.h
 * \brief Tensor to HFusion conversion
 */

#ifndef BISHENGIR_CONVERSION_TENSORTOHFUSION_TENSORTOHFUSION_H
#define BISHENGIR_CONVERSION_TENSORTOHFUSION_TENSORTOHFUSION_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTTENSORTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"

namespace hfusion {
void populateTensorToHFusionConversionPatterns(RewritePatternSet &patterns);
} // namespace hfusion

/// Creates a pass to convert certain tensor ops to linalg/hfusion ops
std::unique_ptr<Pass> createTensorToHFusionConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TENSORTOHFUSION_TENSORTOHFUSION_H