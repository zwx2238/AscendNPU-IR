//===- Utils.h - HFusion to HIVM Conversion Utilities ------------*- C++-*-===//
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

#ifndef BISHENGIR_CONVERSION_HFUSIONTOHIVM_UTILS_H
#define BISHENGIR_CONVERSION_HFUSIONTOHIVM_UTILS_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace hfusion_conversion_utils {

/// Get reassociation of expandShapeOp
/// eg.
///   tensor.expand_shape %arg0 [[0, 1], [2], [3, 4]] : tensor<4x5x3xf32> into
///                                                     tensor<4x1x5x3x1f32>
/// here outRank = 5, expandDims = {1, 4},
/// the result reassociation=[[0, 1], [2], [3, 4]]
SmallVector<SmallVector<int64_t, 2>>
getReAssociation(ArrayRef<int64_t> expandDims, int64_t outRank);

/// Use ins shapedType of linalg op to expand.
/// Insert 1 into shape in axis of dimsArr, and update strides if stridedLayout
/// exits.
///
/// \param shapedType ins shaped type of linalg ops
Type getExpandShapeOpResType(ShapedType shapedType, ArrayRef<int64_t> dimsArr);

template <typename ConcreteOp>
std::enable_if_t<
    std::is_same<ConcreteOp, linalg::ReduceOp>::value ||
        std::is_same<ConcreteOp, linalg::BroadcastOp>::value ||
        std::is_same<ConcreteOp, hfusion::ReduceWithIndexOp>::value,
    Value>
createExpandShapeOp(ConcreteOp op, PatternRewriter &rewriter, Value expandSrc,
                    ShapedType targetType) {
  auto dims = op.getDimensions();
  ShapedType shapedType = cast<ShapedType>(expandSrc.getType());
  Type expandShapeOpResTy = getExpandShapeOpResType(shapedType, dims);
  int64_t outRank = cast<ShapedType>(expandSrc.getType()).getRank();
  if (outRank != 0) {
    // This is not a rank-0 case. Thus turn to the normal case.
    outRank = targetType.getRank();
  }
  auto reassociation = getReAssociation(dims, outRank);

  const bool hasPureBuffer = op.hasPureBufferSemantics();
  assert(expandSrc && "expandSrc shouldn't null.");
  Value expandShapeOp =
      hasPureBuffer
          ? (Value)rewriter.create<memref::ExpandShapeOp>(
                op.getLoc(), expandShapeOpResTy, expandSrc, reassociation)
          : (Value)rewriter.create<tensor::ExpandShapeOp>(
                op.getLoc(), expandShapeOpResTy, expandSrc, reassociation);
  return expandShapeOp;
}

Value createCollapseShapeOp(PatternRewriter &rewriter, Location loc,
                            Value collapseSrc, Type resultType,
                            SmallVector<SmallVector<int64_t, 2>> collapseDims,
                            bool isPureTensor);

} // namespace hfusion_conversion_utils
} // namespace mlir

#endif // BISHENGIR_CONVERSION_HFUSIONTOHIVM_UTILS_H
