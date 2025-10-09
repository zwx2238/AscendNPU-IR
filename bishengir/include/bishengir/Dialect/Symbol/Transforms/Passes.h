//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//
#ifndef BISHENGIR_DIALECT_SYMBOL_TRANSFORMS_PASSES_H
#define BISHENGIR_DIALECT_SYMBOL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"
namespace symbol {

/// Create a pass to propagate symbols
std::unique_ptr<Pass> createPropagateSymbolPass();

/// Create a pass to erase symbols
std::unique_ptr<Pass> createEraseSymbolPass();

/// Create a pass to convert bind_symbolic_shape to tensor encoding
std::unique_ptr<Pass> createSymbolToEncodingPass();

/// Create a pass to convert tensor encoding to bind_symbolic_shape
std::unique_ptr<Pass> createEncodingToSymbolPass();

/// Create a pass to replace symbol.symbolic_int with tensor.dim
std::unique_ptr<mlir::Pass> createUnfoldSymbolicIntPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "bishengir/Dialect/Symbol/Transforms/Passes.h.inc"
} // namespace symbol
} // namespace mlir

#endif // BISHENGIR_DIALECT_SYMBOL_TRANSFORMS_PASSES_H
