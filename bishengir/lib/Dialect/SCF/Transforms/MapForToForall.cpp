//===--------- MapForToForall.cpp -  Map scf.for to scf.forall ops --------===//
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
//
// This file implements a pass to map scf.for op to scf.forall ops.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/SCF/Transforms/Passes.h"
#include "bishengir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MAPFORTOFORALL
#include "bishengir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
static constexpr llvm::StringLiteral kMappingAttrName = "mapping";
static constexpr llvm::StringLiteral kMapForToForallAttrName =
    "map_for_to_forall";

struct ForToForallPass : public impl::MapForToForallBase<ForToForallPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

struct ForToForallRewritePattern : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult
  matchAndRewrite(scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttrOfType<UnitAttr>(kMapForToForallAttrName))
      return failure();

    std::optional<ArrayAttr> deviceMappingAttrs = std::nullopt;
    // If the mapping attribute exists beforehand, just use whatever's passed in
    if (op->hasAttrOfType<ArrayAttr>(kMappingAttrName))
      deviceMappingAttrs = op->getAttrOfType<ArrayAttr>(kMappingAttrName);
    // else if no mapping attribute exists, append a default one with no order
    else {
      deviceMappingAttrs = rewriter.getArrayAttr(
          {hivm::HIVMBlockMappingAttr::get(getContext())});
    }

    scf::ForallOp maybeResult = nullptr;
    DiagnosedSilenceableFailure diag = scf::utils::mapForToForallImpl(
        rewriter, op, deviceMappingAttrs, maybeResult);
    if (!diag.succeeded())
      return rewriter.notifyMatchFailure(op, diag.getMessage());

    rewriter.replaceOp(op, maybeResult);
    return success();
  }
};

void ForToForallPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<ForToForallRewritePattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> scf::createMapForToForallPass() {
  return std::make_unique<ForToForallPass>();
}
