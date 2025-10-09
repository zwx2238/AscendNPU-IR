//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
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

#include "bishengir/Dialect/HFusion/Transforms/DecomposeOpInterfaceImpl.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace hfusion;

namespace {

// Convert multi-axis linalg.transpose to binary linalg.transpose
// linalg.tranpose(tensor(2x16x8x4x3xf32)) -> tensor(2x3x4x8x16xf32)
// permutation = [0,4,3,2,1] is normalized into
// tensor(2x16x8x4x3xf32) -> tensor(2x3x8x4x16xf32)  perm = [0,4,2,3,1]
// tensor(2x3x8x4x16xf32) -> tensor(2x3x4x8x16xf32)  perm = [0,1,3,2,4]
struct TransposeDecomposeInterface
    : public bishengir::BiShengIRAggregatedOpInterface::ExternalModel<
          TransposeDecomposeInterface, linalg::TransposeOp> {
  bool needDecompose(ArrayRef<int64_t> arr) const {
    int mismatch = 0;
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) {
      if (arr[i] != i)
        mismatch++;
    }
    return (mismatch > 2);
  }

  void calculateMinSwaps(ArrayRef<int64_t> perm,
                         SmallVector<SmallVector<int64_t, 2>> &swaps) const {
    int64_t N = static_cast<int64_t>(perm.size());
    SmallVector<std::pair<int64_t, int64_t>, 8> permIndexVec(N);
    for (int64_t i = 0; i < N; i++)
      permIndexVec[i] = {perm[i], i};

    std::sort(permIndexVec.begin(), permIndexVec.end());

    for (int64_t i = 0; i < N; i++) {
      if (permIndexVec[i].second == i)
        continue;
      swaps.push_back({i, permIndexVec[i].second});
      std::swap(permIndexVec[i], permIndexVec[permIndexVec[i].second]);
      i--;
    }
  }

  Value decomposeTransposeOp(linalg::TransposeOp op, OpBuilder &rewriter,
                             SmallVector<SmallVector<int64_t, 2>> swaps,
                             int64_t length) const {
    Value inputVal = op.getInput();
    auto inputTensor = dyn_cast<TensorType>(inputVal.getType());
    Type targetElemType = getElementTypeOrSelf(inputTensor);

    for (auto &swap : swaps) {
      auto curInputStaticShape =
          cast<ShapedType>(inputVal.getType()).getShape();
      SmallVector<int64_t> curOutputStaticShape;
      SmallVector<Value> curOutputDynamicShape;

      int64_t idx1 = swap[0], idx2 = swap[1];
      SmallVector<int64_t> tempPerm;
      // populate intermediate permutations and shapes
      for (int64_t i = 0; i < length; ++i) {
        int64_t targetIdx;
        if (i == idx1) {
          targetIdx = idx2;
        } else if (i == idx2) {
          targetIdx = idx1;
        } else {
          targetIdx = i;
        }

        tempPerm.push_back(targetIdx);
        if (curInputStaticShape[targetIdx] == ShapedType::kDynamic) {
          Operation *dynDimOp =
              rewriter.create<tensor::DimOp>(op->getLoc(), inputVal, targetIdx);
          curOutputStaticShape.push_back(ShapedType::kDynamic);
          curOutputDynamicShape.push_back(dynDimOp->getResults()[0]);
        } else {
          curOutputStaticShape.push_back(curInputStaticShape[targetIdx]);
        }
      }
      // create empty tensor holding the intermediate result shape
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), curOutputStaticShape, targetElemType,
          curOutputDynamicShape);
      auto intermediateTransposeOp = rewriter.create<linalg::TransposeOp>(
          op->getLoc(), inputVal, emptyTensor, tempPerm);
      inputVal = intermediateTransposeOp->getResult(0);
    }

    return inputVal;
  }

  FailureOr<SmallVector<Value>> decomposeOperation(Operation *op,
                                                   OpBuilder &rewriter) const {
    auto transposeOp = llvm::dyn_cast<mlir::linalg::TransposeOp>(op);
    if (!transposeOp)
      return failure();

    auto perm = transposeOp.getPermutation();
    // skip binary transpose
    if (!needDecompose(perm))
      return failure();

    // the order of swaps to be proceeded
    SmallVector<SmallVector<int64_t, 2>> swaps;
    calculateMinSwaps(perm, swaps);

    // Create tensor.empty and decomposed linalg.transpose Ops
    return SmallVector<Value>{
        decomposeTransposeOp(transposeOp, rewriter, swaps, perm.size())};
  }

  bishengir::DecomposePhase getDecomposePhase(Operation *op) const {
    return bishengir::DecomposePhase::AFTER_HFUSION_FLATTEN;
  }
};
} // namespace

void mlir::hfusion::registerDecomposeInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::TransposeOp::attachInterface<TransposeDecomposeInterface>(*ctx);
  });
}
