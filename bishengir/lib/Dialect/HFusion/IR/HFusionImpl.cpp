//===- HFusionImpl.cpp - Implementation of HFusion Dialect Ops --*- C++ -*-===//
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

#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"
#include "bishengir/Dialect/Utils/Util.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <variant>

#if BSPUB_DAVINCI_BISHENGIR
#include "mlir/Dialect/Linalg/IR/LinalgExtensions.h"
#endif

using namespace mlir;
using namespace mlir::hfusion;

Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType,
                      hfusion::RoundMode roundMode, std::optional<Value> dst,
                      bool enableOverflow) {
  Location loc = src.getLoc();
  if (!isa<TensorType>(src.getType())) {
    assert(src.getType().isIntOrIndexOrFloat());
    return convertScalarToDtype(builder, loc, src, targetElemType,
                                /*isUnsignedCast=*/false);
  }

  Value targetTensor;
  if (dst.has_value()) {
    targetTensor = dst.value();
  } else {
    targetTensor = utils::createEmptyOpWithTargetElemType(builder, loc, src,
                                                          targetElemType);
  }

  auto roundingAttr = builder.getAttr<hfusion::RoundModeAttr>(roundMode);
  auto enableOverflowVal = builder.getBoolAttr(enableOverflow);
  auto vcastOp = builder.create<hfusion::CastOp>(
      loc, SmallVector<Type>{targetTensor.getType()}, src, targetTensor,
      roundingAttr, enableOverflowVal);
  return vcastOp->getResult(0);
}

Value hfusion::castTo(OpBuilder &builder, Value src, Type targetElemType) {
  Type srcElemType = getElementTypeOrSelf(src.getType());
  hfusion::RoundMode rounding =
      mlir::utils::selectRoundMode<hfusion::RoundMode>(srcElemType,
                                                       targetElemType);
  return hfusion::castTo(builder, src, targetElemType, rounding);
}
