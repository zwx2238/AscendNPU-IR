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
 * \file TensorToHFusion.cpp
 * \brief Conversion from Tensor to HFusion dialect
 */

#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

struct TensorSplatOpToLinalgFill : OpRewritePattern<tensor::SplatOp> {
  using OpRewritePattern<tensor::SplatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::SplatOp op,
                                PatternRewriter &rewriter) const final {
    auto retTensorType = op.getAggregate().getType();
    ValueRange dynamicSizes = op.getDynamicSizes();
    auto cstVal = op.getInput();
    Value emptyTensorOp = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), retTensorType, dynamicSizes);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{cstVal},
                                                ValueRange{emptyTensorOp});
    return success();
  }
};

void mlir::hfusion::populateTensorToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TensorSplatOpToLinalgFill>(patterns.getContext());
}

namespace {
struct TensorToHFusionConversionPass
    : public impl::ConvertTensorToHFusionBase<TensorToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void TensorToHFusionConversionPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<tensor::TensorDialect, linalg::LinalgDialect,
                         hfusion::HFusionDialect>();
  target.addIllegalOp<tensor::SplatOp>();
  RewritePatternSet patterns(&getContext());
  populateTensorToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createTensorToHFusionConversionPass() {
  return std::make_unique<TensorToHFusionConversionPass>();
}
