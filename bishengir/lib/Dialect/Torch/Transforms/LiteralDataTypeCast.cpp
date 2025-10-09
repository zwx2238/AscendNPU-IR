//===- LiteralDataTypeCast.cpp ------- LiteralDataTypeCast Pass -----------===//
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
// This file implements a pass to cast torch.literal from f64 to f32.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {
namespace torch {

#define GEN_PASS_DEF_LITERALDATATYPECAST
#include "bishengir/Dialect/Torch/Transforms/Passes.h.inc"

class Fp64ToFp32InLiteral : public OpRewritePattern<ValueTensorLiteralOp> {
public:
  using OpRewritePattern<ValueTensorLiteralOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ValueTensorLiteralOp op,
                                PatternRewriter &rewriter) const override {
    // Verify CompatibleTypes.
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    auto input = dyn_cast<DenseIntOrFPElementsAttr>(op.getValueAttr());
    if (!input) {
      return failure();
    }
    auto elemType = input.getElementType();
    if (!isa<Float64Type>(elemType)) {
      return failure();
    }
    auto maybeCastElem = convertFP64ToFP32(input, rewriter);
    if (failed(maybeCastElem)) {
      return failure();
    }
    auto castElem = *maybeCastElem;
    auto elemTy = cast<RankedTensorType>(castElem.getType());
    auto resultTy = ValueTensorType::get(op->getContext(), elemTy.getShape(),
                                         elemTy.getElementType());

    op.setValueAttr(castElem);
    rewriter.modifyOpInPlace(op, [&] { op->getResult(0).setType(resultTy); });
    return success();
  }

  FailureOr<DenseElementsAttr> convertFP64ToFP32(DenseIntOrFPElementsAttr attr,
                                                 Builder &builder) const {
    auto fp64Values = attr.getValues<double>();

    std::vector<float> fp32Values;
    fp32Values.reserve(fp64Values.size());
    for (double value : fp64Values) {
      fp32Values.push_back(static_cast<float>(value));
    }

    auto originalType = dyn_cast<RankedTensorType>(attr.getType());
    if (!originalType) {
      return failure();
    }
    auto fp32Type =
        RankedTensorType::get(originalType.getShape(), builder.getF32Type());

    return DenseFPElementsAttr::get(fp32Type, fp32Values);
  }
};

namespace {
class LiteralDataTypeCastPass
    : public impl::LiteralDataTypeCastBase<LiteralDataTypeCastPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<Fp64ToFp32InLiteral>(context);

    if (failed(applyPatternsGreedily(f, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLiteralDataTypeCastPass() {
  return std::make_unique<LiteralDataTypeCastPass>();
}

void registerLiteralDataTypeCast() {
  PassRegistration<LiteralDataTypeCastPass> reg;
}

} // namespace torch
} // namespace mlir