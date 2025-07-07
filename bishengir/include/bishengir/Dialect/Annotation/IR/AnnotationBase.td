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
 * \file AnnotationBase.h
 * \brief Base definitions for annotation dialect
 */

#ifndef BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD
#define BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD

include "mlir/IR/OpBase.td"

def Annotation_Dialect : Dialect {
  let name = "annotation";
  let description = [{
    Annotation dialects for mark operations to
    define some extra attrs for a certain op:

    ```mlir
    // Mark extra attrs
    annotation.mark %a { attr-dict } : f64
    ```
  }];
  let cppNamespace = "::mlir::annotation";
}

#endif // BISHENGIR_DIALECT_ANNOTATION_IR_ANNOTATIONBASE_TD
