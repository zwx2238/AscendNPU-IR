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
 * \file MathExtBase.td
 * \brief Base definitions for extended math dialect.
 */

#ifndef MATHEXT_BASE
#define MATHEXT_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def MathExt_Dialect : Dialect {
  let name = "mathExt";

  let cppNamespace = "::mlir::mathExt";

  let description = [{
    Extended Math dialect.
  }];
  let hasConstantMaterializer = 1;
}

#endif // MATHEXT_BASE
