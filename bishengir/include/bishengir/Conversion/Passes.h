//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_CONVERSION_PASSES_H
#define BISHENGIR_CONVERSION_PASSES_H

#include "bishengir/Conversion/ArithToHFusion/ArithToHFusion.h"
#include "bishengir/Conversion/GPUToHFusion/GPUToHFusion.h"
#include "bishengir/Conversion/LinalgToHFusion/LinalgToHFusion.h"
#include "bishengir/Conversion/MathToHFusion/MathToHFusion.h"
#include "bishengir/Conversion/TensorToHFusion/TensorToHFusion.h"
#include "mlir/Pass/Pass.h"

namespace bishengir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Conversion/Passes.h.inc"

} // namespace bishengir

#endif // BISHENGIR_CONVERSION_PASSES_H
