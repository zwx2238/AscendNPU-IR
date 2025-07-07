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
 * \file InitAllDialects.h
 * \brief BiShengIR Dialects Registration
 * \details This file defines a helper to trigger the registration of all
 *          bishengir related dialects to the system.
 */

#ifndef BISHENGIR_INITALLDIALECTS_H
#define BISHENGIR_INITALLDIALECTS_H

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace bishengir {

/// Add all the bishengir-specific dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::annotation::AnnotationDialect,
                  mlir::hacc::HACCDialect,
                  mlir::hfusion::HFusionDialect,
                  mlir::hivm::HIVMDialect,
                  mlir::mathExt::MathExtDialect>();
  // clang-format on
}

/// Append all the bishengir-specific dialects to the registry contained in the
/// given context.
inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  bishengir::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace bishengir

#endif // BISHENGIR_INITALLDIALECTS_H
