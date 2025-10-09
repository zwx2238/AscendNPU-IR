//===--ValueDependencyAnalyzer.cpp -------------------------------*- C++-*-===//
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

#include "bishengir/Dialect/Utils/UnionFind.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "bishengir/Dialect/Utils/ValueDependencyAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <queue>

#define DEBUG_TYPE "value-dependency-analyzer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// TODO: check memref indices whether operation affects or not (possible
// improvement)

namespace mlir {
namespace utils {

void ValueDependencyAnalyzer::reset() {
  // reset data
  valueList.clear();
  valueToIndexMap.clear();
}

void ValueDependencyAnalyzer::pushAllValues(Operation *parent) {
  auto pushValueIfNotExists = [this](Value val) -> void {
    if (valueToIndexMap.contains(val))
      return;
    valueToIndexMap[val] = static_cast<int>(valueList.size());
    valueList.push_back(val);
  };

  if (auto funcOp = dyn_cast<func::FuncOp>(parent)) {
    for (Value val : funcOp.getArguments())
      pushValueIfNotExists(val);
  }

  parent->walk<WalkOrder::PreOrder>(
      [&pushValueIfNotExists](Operation *op) -> WalkResult {
        // handles case for BlockArgument operands
        for (Value val : op->getOperands())
          pushValueIfNotExists(val);
        for (Value val : op->getResults())
          pushValueIfNotExists(val);
        return WalkResult::advance();
      });
}

void ValueDependencyAnalyzer::buildValueDependency(Operation *parent) {
  reset();
  pushAllValues(parent);
  dsu = UnionFindBase(valueList.size());

  parent->walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(op);
    if (!viewLikeOp)
      return WalkResult::advance();
    auto srcIdx = valueToIndexMap.at(viewLikeOp.getViewSource());

    for (OpResult result : viewLikeOp->getResults()) {
      // handle for aliases operation
      dsu.join(srcIdx, valueToIndexMap.at(result));
    }
    return WalkResult::advance();
  });
}

Value ValueDependencyAnalyzer::getAllocOf(Value value) {
  return valueList[dsu.find(valueToIndexMap.at(value))];
}

} // namespace utils
} // namespace mlir
