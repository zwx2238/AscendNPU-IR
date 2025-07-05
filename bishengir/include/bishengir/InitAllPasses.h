//===- InitAllPasses.h - BiShengIR Passes Registration ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_INITALLPASSES_H
#define BISHENGIR_INITALLPASSES_H

#include "bishengir/Conversion/Passes.h"

namespace bishengir {

// This function may be called to register the bishengir-specific MLIR passes
// with the global registry.
inline void registerAllPasses() {
  // Conversion passes
  bishengir::registerConversionPasses();
}

} // namespace bishengir

#endif // BISHENGIR_INITALLPASSES_H
