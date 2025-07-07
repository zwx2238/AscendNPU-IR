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
 * \file GPUToHFusion.cpp
 * \brief Conversion from GPU to HFusion dialect
 */

#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

struct GPUBarrierToHFusionBarrierOp : OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hfusion::BarrierOp>(op);
    return success();
  }
};

void mlir::hfusion::populateGPUToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GPUBarrierToHFusionBarrierOp>(patterns.getContext());
}

namespace {
struct GPUToHFusionConversionPass
    : public impl::ConvertGPUToHFusionBase<GPUToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void GPUToHFusionConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<hfusion::HFusionDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();

  RewritePatternSet patterns(&getContext());
  populateGPUToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createGPUToHFusionConversionPass() {
  return std::make_unique<GPUToHFusionConversionPass>();
}
