//===- DropSymbols.cpp ----------------------------------------------------===//
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
// This file implements tensor.dim source replacer optimization
//
//===----------------------------------------------------------------------===//
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hfusion-drop-symbols"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_DROPSYMBOLS
#include "bishengir/Dialect/HFusion/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

namespace mlir {
namespace hfusion {

using namespace opfusion;

namespace {

Type dropSymbolsFromType(Type val) {
  auto rankedTensorTy = dyn_cast<RankedTensorType>(val);
  if (!rankedTensorTy)
    return val;
  auto newRankedTy = RankedTensorType::get(rankedTensorTy.getShape(),
                                           rankedTensorTy.getElementType());
  return newRankedTy;
}

SmallVector<Type> dropSymbolsFromType(TypeRange ty) {
  SmallVector<Type> ret;
  for (auto res : ty)
    ret.emplace_back(dropSymbolsFromType(res));
  return ret;
}

void dropSymbolsFromValue(Value val) {
  val.setType(dropSymbolsFromType(val.getType()));
}

void dropSymbolsFromValue(ValueRange val) {
  for (auto res : val)
    dropSymbolsFromValue(res);
}

/// or decode the argument’s shape.
void dropSymbolsFromFunc(func::FuncOp funcOp) {
  dropSymbolsFromValue(funcOp.getArguments());
  for (auto &block : funcOp.getBlocks())
    dropSymbolsFromValue(block.getArguments());
  for (auto &op : funcOp.getOps())
    dropSymbolsFromValue(op.getResults());
  auto oldFuncType = funcOp.getFunctionType();
  auto newFuncType = FunctionType::get(
      funcOp.getContext(), dropSymbolsFromType(oldFuncType.getInputs()),
      dropSymbolsFromType(oldFuncType.getResults()));
  funcOp.setFunctionType(newFuncType);
}

} // namespace
} // namespace hfusion

} // namespace mlir

struct DropSymbolsPass : public impl::DropSymbolsBase<DropSymbolsPass> {
  void runOnOperation() override {
    LDBG("Running DropSymbols");
    auto funcOp = getOperation();
    hfusion::dropSymbolsFromFunc(funcOp);
  }
};
std::unique_ptr<Pass> mlir::hfusion::createDropSymbolsPass() {
  return std::make_unique<DropSymbolsPass>();
}
