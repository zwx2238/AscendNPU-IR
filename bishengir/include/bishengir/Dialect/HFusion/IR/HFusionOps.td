//===- HFusionOps.td - HFusion op definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for hybrid fusion operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "bishengir/Interfaces/AggregatedOpInterface.td"
include "bishengir/Dialect/HFusion/IR/HFusionBase.td"
include "bishengir/Dialect/HFusion/IR/HFusionAttrs.td"

// Base class for HFusion dialect ops that do not correspond to library calls.
class HFusion_Op<string mnemonic, list<Trait> traits = []> :
    Op<HFusion_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// HFusion Types
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

//===----------------------------------------------------------------------===//
// HFusion Debug op definitions.
//===----------------------------------------------------------------------===//

def PrintOp : HFusion_Op<"print">,
  Arguments<(ins StrAttr:$prefix, BoolAttr:$hex,
             AnyTypeOf<[AnyInteger, AnyFloat, AnyRankedTensor]>:$arg)> {
  let summary = "Device-side print for debugging";
  let description = [{
    `hfusion.print` takes a literal string `prefix` and an argument of
    scalar or tensor that should be printed. The optional arg `hex` configs
    if printing in hex format.
  }];
  let assemblyFormat = [{
    $prefix attr-dict $arg `:` type($arg)
  }];
}

def BarrierOp : HFusion_Op<"barrier"> {
  let summary = "Synchronizes all pipelines of a core.";
  let description = [{
    The "barrier" op synchronizes all pipelines of a core.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// InterleaveOp
//===----------------------------------------------------------------------===//

def InterleaveOp : HFusion_Op<"interleave", [Pure, Commutative,
                                             SameOperandsAndResultRank,
                                             AllTypesMatch<["input"]>]> {
  let summary = [{
      Constructs one tensor by interleaving n input tensors.
      Only support n = 2 now.
  }];

  let description = [{
      Interleaves the values of n tensors along their last dimension.
      N tensors must have the same shape.
      Input tensors and output tensor must have same rank.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[AnyRankedTensor]>>:$input);
  let results = (outs AnyTypeOf<[AnyRankedTensor]>:$output);
  let extraClassDeclaration = [{
    static int64_t getInterLeaveChannelNums() {return 2;}
  }];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeinterleaveOp
//===----------------------------------------------------------------------===//

def DeinterleaveOp : HFusion_Op<"deinterleave", [Pure, Commutative,
                                                 SameOperandsAndResultRank,
                                                 AllTypesMatch<["output"]>]> {
  let summary = [{
    Constructs two tensors by deinterleaving an input tensor
  }];

  let description = [{
    The deinterleave operation constructs two tensors from a single input
    tensor. The first result tensor contains the elements from even indexes, and
    the second contains elements from odd indexes(both indexes mean the last
    dimension index). And it constrains the last dimension size of input must be
    multiples of 2.

    The channelIndex attribute controls the output behavior:
    * -1: Output all channels (returns two tensors, one with even indexes and one with odd indexes)
    * 0: Output only channel 0 (even indexes)
    * 1: Output only channel 1 (odd indexes)
  }];

  let extraClassDeclaration = [{
    static int64_t getDeInterLeaveChannelNum() { return 2; }
    int64_t getDeInterLeaveChannelIdx();
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedTensor]>:$input,
                       I64Attr:$channelIndex);
  let results = (outs Variadic<AnyTypeOf<[AnyRankedTensor]>>:$output);

  let assemblyFormat = "$input custom<HFusionDeinterleave>($channelIndex) attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FlipOp
//===----------------------------------------------------------------------===//

def FlipOp : HFusion_Op<"flip", [Pure, Commutative, AllTypesMatch<["output"]>]> {
  let summary = [{
      Flips a tensor x along the dimension dim.
  }];

  let description = [{
      Flips a tensor x along the dimension dim.
      currently only final dimension supported.
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedTensor]>:$input);
  let results = (outs AnyTypeOf<[AnyRankedTensor]>:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

def IsInfOp : HFusion_Op<"isinf", [Pure, SameOperandsAndResultRank,
                                   AllTypesMatch<["output"]>]> {
  let summary = [{
    Calculates whether elements of a float type tensor equal positive
    infinity or negative infinity.
  }];
  let description = [{
    Calculates whether elements of a float type tensor is infinity.
    Both positive infinity and negative infinity work.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsNanOp
//===----------------------------------------------------------------------===//

def IsNanOp : HFusion_Op<"isnan", [Pure, SameOperandsAndResultRank,
                                   AllTypesMatch<["output"]>]> {
  let summary = [{
    Calculates whether elements of a float type tensor is NAN.
  }];
  let description = [{
    Calculates whether elements of a float type tensor is NAN.
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

def IsFiniteOp : HFusion_Op<"isfinite", [Pure, SameOperandsAndResultRank,
                                         AllTypesMatch<["output"]>,
                                         DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
                                           ["decomposeOperation"]>]> {
  let summary = [{
    Calculates whether elements of a float type tensor is finite.
  }];
  let description = [{
    Calculates whether elements of a float type tensor is finite
    (i.e., not NaN or infinity).
  }];
  let arguments = (ins RankedTensorOf<[BF16, F16, F32]>:$input);
  let results = (outs RankedTensorOf<[I1]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONOPS_TD