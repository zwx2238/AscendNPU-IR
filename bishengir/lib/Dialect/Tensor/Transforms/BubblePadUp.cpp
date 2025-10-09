//===- BubblePadUp.cpp ----------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "bishengir/Dialect/Tensor/Transforms/PropagateReshape/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "bubble-pad-up"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_BUBBLEPADUP
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::utils::debugger;

namespace mlir {

namespace tensor {
using namespace mlir::hfusion;
using namespace mlir::tensor::reshape_utils;
using namespace mlir::hfusion::reshape_utils;
using namespace mlir::utils::debugger;
namespace {
class BubblePadElementwise : public mlir::OpRewritePattern<tensor::PadOp> {
public:
  explicit BubblePadElementwise(MLIRContext *context)
      : OpRewritePattern<tensor::PadOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
BubblePadElementwise::matchAndRewrite(tensor::PadOp padOp,
                                      PatternRewriter &rewriter) const {
  Operation *srcOp = padOp.getSource().getDefiningOp();
  if (!isMarkedAsElementwiseUnaryOp(srcOp)) {
    LDBG("Pad source is not elemwise " << padOp.getSource());
    return failure();
  }
  // This returns a bit vector of size rank specifying which dims is padded
  auto paddedDims = padOp.getPaddedDims();
  auto srcShape = utils::getShape(padOp.getSource().getType());
  if (!paddedDims.back()) {
    LDBG("Padded dimension is not last element");
    return failure();
  }
  auto staticLowPad = padOp.getStaticLow();
  if (utils::areShapesAligned(
          {staticLowPad.back()},
          32)) // All shapes must be alined to 32 bit integer
    return failure();
  auto mixLowPad = padOp.getMixedLowPad();
  auto mixHighPad = padOp.getMixedHighPad();
  auto staticHighPad = padOp.getStaticHigh();
  SmallVector<Value> newOperands;
  for (auto opr : srcOp->getOperands()) {
    // Pad this
    if (isa<RankedTensorType>(opr.getType()) &&
        utils::getShape(opr.getType()) == srcShape) {
      rewriter.setInsertionPoint(srcOp);
      Type padType = tensor::PadOp::inferResultType(
          cast<RankedTensorType>(opr.getType()), staticLowPad, staticHighPad);
      tensor::PadOp newPadOp = rewriter.create<tensor::PadOp>(
          padOp.getLoc(), padType, opr, mixLowPad, mixHighPad);
      clonePadRegion(rewriter, padOp, newPadOp);
      newOperands.push_back(newPadOp.getResult());
    } else {
      newOperands.push_back(opr);
    }
  }
  updateDefiningOp(srcOp, rewriter, newOperands);
  return success();
}

class BubblePadUpPass : public impl::BubblePadUpBase<BubblePadUpPass> {
  using Base::Base;
  void runOnOperation() final;
};

void BubblePadUpPass::runOnOperation() {
  func::FuncOp f = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<BubblePadElementwise>(context);
  if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<Pass> createBubblePadUpPass() {
  return std::make_unique<BubblePadUpPass>();
}

} // namespace tensor
} // namespace mlir
