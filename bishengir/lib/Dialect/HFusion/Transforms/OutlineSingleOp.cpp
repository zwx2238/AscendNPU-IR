//===- OutlineSingleOp.cpp - Outline Single Operations --------------------===//
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

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlockOutliner.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-outline-single-op"

namespace mlir {
#define GEN_PASS_DEF_OUTLINESINGLEOP
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
using namespace hfusion::opfusion;
struct OutlineSingleOpPass
    : public impl::OutlineSingleOpBase<OutlineSingleOpPass> {
  explicit OutlineSingleOpPass(const OutlineSingleOpOptions &options)
      : OutlineSingleOpBase(options) {}

public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!hacc::utils::isHost(func))
      return;

    if (isReturnOnlyFunc(func)) {
      outlineAllReturnValue(func);
      return;
    }

    HFusionOpFusionOptions options;
    options.moveOutToParam = this->moveOutToParam;
    SmallVector<func::FuncOp> outlinedFuncs;
    if (failed(outlineSingleFusedFuncs(func, options, outlinedFuncs)))
      signalPassFailure();
  };

private:
  void outlineAllReturnValue(func::FuncOp calleeFunc) const {
    OpBuilder builder(calleeFunc);
    builder.setInsertionPointAfter(calleeFunc.getOperation());
    Operation *endTerminator =
        calleeFunc.getRegion().getBlocks().begin()->getTerminator();
    FunctionType outlinedFuncType = FunctionType::get(
        calleeFunc->getContext(), ValueRange(endTerminator->getOperands()),
        ValueRange(endTerminator->getOperands()));
    auto funcName = calleeFunc.getSymName().str() + "_single_outlined";
    auto outlinedFunc = builder.create<func::FuncOp>(
        calleeFunc->getLoc(), funcName, outlinedFuncType);

    // set attributes for the outlined device kernel
    // because the caller is host, the device kernel must be an entry kernel
    hacc::utils::setDeviceEntry(outlinedFunc);
    auto fusionKind = tryGetFusionKind(calleeFunc);
    if (fusionKind.has_value())
      trySetFusionKind(outlinedFunc, fusionKind.value());

    // add callsite and return op
    Block *entryBlock = outlinedFunc.addEntryBlock();
    builder.setInsertionPointToEnd(entryBlock);
    builder.create<func::ReturnOp>(outlinedFunc->getLoc(),
                                   outlinedFunc.getArguments());
    builder.setInsertionPoint(endTerminator);
    auto newCall = builder.create<func::CallOp>(
        endTerminator->getLoc(), endTerminator->getOperandTypes(), funcName,
        endTerminator->getOperands());
    endTerminator->setOperands(newCall->getResults());
  }

  bool isReturnOnlyFunc(func::FuncOp func) const {
    Region &region = func.getFunctionBody();
    if (!region.hasOneBlock()) {
      return false;
    }
    return llvm::all_of(region.front(), [&](Operation &op) {
      return !(FusibleHelper::isImportantPattern(&op) || isa<func::CallOp>(op));
    });
  }
};

} // namespace mlir

std::unique_ptr<Pass> mlir::hfusion::createOutlineSingleOpPass(
    const mlir::OutlineSingleOpOptions &options) {
  return std::make_unique<OutlineSingleOpPass>(options);
}
