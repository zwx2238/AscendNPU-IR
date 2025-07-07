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
 * \file HFusionBase.td
 * \brief Base definitions for hybrid fusion dialect
 * \details This is the definition file for base hybrid fusion support.
 */

#ifndef BISHENGIR_DIALECT_HFUSION_IR_HFUSIONBASE_TD
#define BISHENGIR_DIALECT_HFUSION_IR_HFUSIONBASE_TD

include "bishengir/Dialect/HFusion/IR/HFusionEnums.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def HFusion_Dialect : Dialect {
  let name = "hfusion";
  let description = [{
    Hybrid Fusion (HFusion) dialect.
  }];
  let cppNamespace = "::mlir::hfusion";
  let dependentDialects = [
    "hacc::HACCDialect",
    "linalg::LinalgDialect",
    "mathExt::MathExtDialect",
    "mesh::MeshDialect",
  ];
  let hasCanonicalizer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Define the function attribute enums matching the OpDSL functions.
def UnaryFnAttr : EnumAttr<HFusion_Dialect, UnaryFn, "unary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def BinaryFnAttr : EnumAttr<HFusion_Dialect, BinaryFn, "binary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def CompareFnAttr : EnumAttr<HFusion_Dialect, CompareFn, "compare_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TernaryFnAttr : EnumAttr<HFusion_Dialect, TernaryFn, "ternary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def TypeFnAttr : EnumAttr<HFusion_Dialect, TypeFn, "type_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def RoundModeAttr : EnumAttr<HFusion_Dialect, RoundMode, "round_mode"> {
  let assemblyFormat = "`<` $value `>`";
}
def AtomicKindAttr : EnumAttr<HFusion_Dialect, AtomicKind, "atomic_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // BISHENGIR_DIALECT_HFUSION_IR_HFUSIONBASE_TD
