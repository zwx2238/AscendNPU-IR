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
 * \file HACCBase.td
 * \brief Base definitions for HACC dialect
 * \details This is the definition file for base heterogeneous async
 *          computing call support.
 */

#ifndef BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD
#define BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD

include "mlir/IR/OpBase.td"

def HACC_Dialect : Dialect {
  let name = "hacc";
  let description = [{
    Heterogeneous Async Computing Call (HACC) dialect.
  }];
  let cppNamespace = "::mlir::hacc";
  let useDefaultAttributePrinterParser = 1;
}

#endif // BISHENGIR_DIALECT_HACC_IR_HACCBASE_TD
