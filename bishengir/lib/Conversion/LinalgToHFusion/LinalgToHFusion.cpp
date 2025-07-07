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
 * \file LinalgToHFusion.cpp
 * \brief Conversion from Linalg to HFusion dialect
 */

#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOHFUSION
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hfusion;

struct LinalgMapToHFusionPattern : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp op,
                                PatternRewriter &rewriter) const final {
    Region &mapper = op.getMapper();
    if (!mapper.hasOneBlock())
      return failure();
    Block &block = mapper.front();
    if (block.getOperations().size() != 2)
      return failure();
    auto &mapped = *block.getOperations().begin();
    auto callOp = dyn_cast<func::CallOp>(mapped);
    if (callOp == nullptr)
      return failure();
    StringRef funcName = callOp.getCallee();
    if (funcName.starts_with("__hmf_relu")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::relu);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_log1p")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::log1p);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_sqrt")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::sqrt);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_fabs")) {
      auto unaryAttr =
          rewriter.getAttr<linalg::UnaryFnAttr>(linalg::UnaryFn::abs);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_exp")) {
      auto unaryAttr =
          rewriter.getAttr<linalg::UnaryFnAttr>(linalg::UnaryFn::exp);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_rsqrt")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::rsqrt);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_log")) {
      auto unaryAttr =
          rewriter.getAttr<linalg::UnaryFnAttr>(linalg::UnaryFn::log);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_isinf")) {
      rewriter.replaceOpWithNewOp<hfusion::IsInfOp>(
          op, TypeRange(op.getResult()), ValueRange{op.getInputs()[0]});
      return success();
    }
    // TODO: funcName of is_nan op need to confirm
    if (funcName.starts_with("__hmf_isnan")) {
      rewriter.replaceOpWithNewOp<hfusion::IsNanOp>(
          op, TypeRange(op.getResult()), ValueRange{op.getInputs()[0]});
      return success();
    }
    if (funcName.starts_with("__hmf_recipf") ||
        funcName.starts_with("__hmf_recipDh")) {
      Type resultType = mlir::getElementTypeOrSelf(op.getInit().getType());
      auto constOne = rewriter.create<arith::ConstantOp>(
          op->getLoc(), rewriter.getFloatAttr(resultType, 1));
      auto emptyTensor = mlir::tensor::createTensorEmptyOp(
          rewriter, op->getLoc(), op.getInputs()[0]);
      auto fillOp = rewriter.create<linalg::FillOp>(
          op->getLoc(), ValueRange{constOne.getResult()},
          ValueRange{emptyTensor});
      auto binaryAttr =
          rewriter.getAttr<linalg::BinaryFnAttr>(linalg::BinaryFn::div);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<linalg::ElemwiseBinaryOp>(
          op, ValueRange{fillOp->getResults()[0], op.getInputs()[0]},
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_tanf") ||
        funcName.starts_with("__hmf_tanDh")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::tan);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_tanhf") ||
        funcName.starts_with("__hmf_tanhDh")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::tanh);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_atan")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::atan);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_ilogb")) {
      auto unaryAttr =
          rewriter.getAttr<hfusion::UnaryFnAttr>(hfusion::UnaryFn::ilogb);
      auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseUnaryOp>(
          op, op.getInputs(), ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_ldexp")) {
      auto binaryAttr =
          rewriter.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::ldexp);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
          op, ValueRange({op.getInputs()[0], op.getInputs()[1]}),
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_flip")) {
      // There is only one input which becomes the last dimension.
      // So, only need to do flip on the first input in the vector.
      rewriter.replaceOpWithNewOp<hfusion::FlipOp>(
          op, ValueRange{op.getInit()}, ValueRange{op.getInputs()[0]});
      return success();
    }
    if (funcName.starts_with("__hmf_powf")) {
      auto binaryAttr =
          rewriter.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::powf);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
          op, ValueRange({op.getInputs()[0], op.getInputs()[1]}),
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    if (funcName.starts_with("__hmf_powi")) {
      auto binaryAttr =
          rewriter.getAttr<hfusion::BinaryFnAttr>(hfusion::BinaryFn::powi);
      auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
      rewriter.replaceOpWithNewOp<hfusion::ElemwiseBinaryOp>(
          op, ValueRange({op.getInputs()[0], op.getInputs()[1]}),
          ValueRange{op.getInit()}, ArrayRef{fnAttr});
      return success();
    }
    return failure();
  }
};

struct LinalgGenericToHFusionArangePattern
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getOutputs().size() != 1 || op.getInputs().size() != 0)
      return failure();
    // Should iterate and store over the whole tensor/memref
    if (!llvm::all_of(op.getIteratorTypesArray(), linalg::isParallelIterator) ||
        !op.getIndexingMapsArray()[0].isIdentity())
      return failure();
    // Should only yield value from index
    if (!op.hasIndexSemantics() || !op.getBody()->getArgument(0).use_empty())
      return failure();

    Value target = op.getOutputs()[0];
    auto type = dyn_cast<ShapedType>(target.getType());
    if (type == nullptr || !type.getElementType().isIntOrFloat())
      return failure();

    // Note: currently, only 1-D arange is supported
    if (!type.hasRank() || type.getRank() != 1)
      return failure();
    auto yieldOp = *(op.getBody()->getOps<linalg::YieldOp>().begin());
    Value yieldVal = yieldOp.getValues()[0];
    auto castOp = yieldVal.getDefiningOp<arith::IndexCastOp>();
    if (castOp == nullptr)
      return failure();
    auto indexOp = castOp.getIn().getDefiningOp<linalg::IndexOp>();
    if (indexOp == nullptr)
      return failure();

#ifndef NDEBUG
    // Get the strides necessary from the dps init
    Value init = op.getDpsInitOperand(0)->get();
    auto shapedTy = dyn_cast<ShapedType>(init.getType());
    assert(shapedTy && "Expecting shaped type as output of arange");
#endif
    rewriter.replaceOpWithNewOp<hfusion::ArangeOp>(op, target);
    return success();
  }
};

// handle the atomic op in the form of linalg.generic
// use hfusion.store to represent the atomic op
// Input:
//  linalg.generic
//    ins(%subview_2, %extracted_slice : memref<?xf32>, tensor<?xf32>)
//    outs(%subview_2 : memref<?xf32>)
//      attrs = {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel",
//      MemSyncScope = "gpu"} {
//    ^bb0(%in: f32, %in_3: f32, %out: f32):
//      %output = arith.addf %in, %in_3 : f32
//      linalg.yield %output : f32
//    }
//
// Output:
//  %memref = bufferization.to_memref %extracted_slice : memref<?xf32>
//  hfusion.store {atomic_kind = #hfusion.atomic_kind<add>}
//    ins(%16 : memref<?xf32>)
//    outs(%subview_2 : memref<?xf32>)
struct AtomicLinalgGenericToHFusionStorePattern
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  std::optional<StringRef> getAtomicAttrRef(linalg::GenericOp op) const {
    StringAttr linalgAtomicRmwAttr =
        op->getAttrOfType<StringAttr>(StringRef("GenericAtomicRMW"));
    if (!linalgAtomicRmwAttr) {
      return std::nullopt;
    }

    return linalgAtomicRmwAttr.getValue();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto linalgAtomicRmwStr = getAtomicAttrRef(op);
    if (!linalgAtomicRmwStr.has_value()) {
      return failure();
    }

    auto atomicRef = *linalgAtomicRmwStr;
    hfusion::AtomicKindAttr atomicKind;
    auto *context = rewriter.getContext();
    if (atomicRef.ends_with("add")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::ADD);
    } else if (atomicRef.ends_with("max")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::MAX);
    } else if (atomicRef.ends_with("min")) {
      atomicKind = AtomicKindAttr::get(context, AtomicKind::MIN);
    } else {
      op.emitOpError("unsupported atomic operation: ");
      llvm_unreachable("Not implemented");
    }

    // hivm.copy only accept tensor/tensor or memref/memref as input/output
    // and the atomicRMW Op might be masked
    // need to turn the input tensor into the same type the dst memref has
    auto hfusionStoreOp = rewriter.create<hfusion::StoreOp>(
        op.getLoc(), ValueRange(op.getInputs()[1]),
        ValueRange(op.getInputs()[0]));
    hfusionStoreOp.setAtomicKindAttr(atomicKind);
    rewriter.eraseOp(op);
    return success();
  }
};

// To replace the linalg::reduceOp with attr of reduce_with_index
// with hfusion.reduce_with_index Op
// Input:
// %reduced:2 = linalg.reduce
//    ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>)
//    outs(%0, %1 : tensor<256xf32>, tensor<256xi32>)
//    dimensions = [1]  {reduce_mode = "max_with_index"}
//
// Output:
// %2:2 = hfusion.reduce_with_index <max>
//    ins(%arg0, %arg1 : tensor<256x64xf32>, tensor<256x64xi32>)
//    outs(%0, %1 : tensor<256xf32>, tensor<256xi32>)
//    dimensions = [1] -> tensor<256xf32>, tensor<256xi32>
struct LinalgToHFusionReduceWithIndex : OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const final {
    StringAttr linalgReduceAttr =
        op->getAttrOfType<StringAttr>(StringRef("reduce_mode"));
    if (!linalgReduceAttr) {
      return failure();
    }

    hfusion::ReduceWithIndexKind reduceKind;
    if (linalgReduceAttr == "max_with_index") {
      reduceKind = hfusion::ReduceWithIndexKind::MAX;
    } else if (linalgReduceAttr == "min_with_index") {
      reduceKind = hfusion::ReduceWithIndexKind::MIN;
    } else {
      return failure();
    }

    ValueRange inits = op.getInits();
    auto reduceKindAttr =
        ReduceWithIndexKindAttr::get(rewriter.getContext(), reduceKind);
    rewriter.replaceOpWithNewOp<hfusion::ReduceWithIndexOp>(
        op, TypeRange{inits[0].getType(), inits[1].getType()},
        /*input*/ op.getInputs(), /*outputValue&Index*/ inits, reduceKindAttr,
        op.getDimensionsAttr());
    return success();
  }
};

void mlir::hfusion::populateLinalgToHFusionConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LinalgMapToHFusionPattern, LinalgGenericToHFusionArangePattern,
               AtomicLinalgGenericToHFusionStorePattern,
               LinalgToHFusionReduceWithIndex>(patterns.getContext());
}

namespace {
struct LinalgToHFusionConversionPass
    : public impl::ConvertLinalgToHFusionBase<LinalgToHFusionConversionPass> {
  void runOnOperation() override;
};
} // namespace

void LinalgToHFusionConversionPass::runOnOperation() {
  auto *module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<memref::MemRefDialect, linalg::LinalgDialect,
                         bufferization::BufferizationDialect,
                         tensor::TensorDialect, hfusion::HFusionDialect>();
  // also add dialects that maybe created by hfusion dialect ops
  target.addLegalDialect<arith::ArithDialect, math::MathDialect>();
  target.addDynamicallyLegalOp<linalg::ReduceOp>([](Operation *op) {
    StringAttr linalgReduceAttr =
        op->getAttrOfType<StringAttr>(StringRef("reduce_mode"));
    return !linalgReduceAttr;
  });
  // Mark linalg.map and libclc func decls as illegal
  target.addIllegalOp<linalg::MapOp>();
  target.addIllegalOp<linalg::GenericOp>();

  RewritePatternSet patterns(&getContext());
  populateLinalgToHFusionConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createLinalgToHFusionConversionPass() {
  return std::make_unique<LinalgToHFusionConversionPass>();
}
