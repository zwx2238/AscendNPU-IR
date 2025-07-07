/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/*!
 * \file HFusionImpl.h
 * \brief HFusion implementation
 */

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir {
namespace hfusion {

template <typename BnaryOp, typename OpFun, typename OpFunAttr>
Operation *createBinaryOp(OpBuilder &builder, Location loc, OpFun opFn,
                          ValueRange inputs, ValueRange out) {
  auto attr = builder.getAttr<OpFunAttr>(opFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  return builder.create<BnaryOp>(loc, inputs, out, fnAttr);
}

template <typename UnaryOp, typename OpFun, typename OpFunAttr>
Operation *createUnaryOp(OpBuilder &builder, Location loc, OpFun opFn,
                         ValueRange inputs, ValueRange outs) {
  auto attr = builder.getAttr<OpFunAttr>(opFn);
  auto fnAttr = builder.getNamedAttr("fun", attr);
  return builder.create<UnaryOp>(loc, inputs, outs, fnAttr);
}

/// Cast `src` value to the specified element type and rounding mode.
///
/// `src` can be either tensor or scalar.
/// If it's a scalar, casting is done by arith dialect ops.
/// If it's a tensor, casting is done by `hfusion.cast` op. If `dst` is not
/// provided, the init value is a `tensor.empty` op. Otherwise, it's written
/// to `dst`.
Value castTo(OpBuilder &builder, Value src, Type targetElemType,
             hfusion::RoundMode roundMode,
             std::optional<Value> dst = std::nullopt,
             bool enableOverflow = true);

/// Cast `src` value to the specified element type.
/// Select rounding mode inside.
Value castTo(OpBuilder &builder, Value src, Type targetElemType);

} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONIMPL_H