//===- MergeConsecutiveInsertExtractSlice.cpp -------------------*- C++ -*-===//
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
#include "bishengir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_MERGECONSECUTIVEINSERTEXTRACTSLICE
#include "bishengir/Dialect/Tensor/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;

namespace {
struct MergeConsecutiveInsertExtractSlicePass
    : public impl::MergeConsecutiveInsertExtractSliceBase<
          MergeConsecutiveInsertExtractSlicePass> {
  void runOnOperation() override;
};
} // namespace

void MergeConsecutiveInsertExtractSlicePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  RewritePatternSet patterns(funcOp.getContext());
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<Pass>
mlir::tensor::createMergeConsecutiveInsertExtractSlicePass() {
  return std::make_unique<MergeConsecutiveInsertExtractSlicePass>();
}
