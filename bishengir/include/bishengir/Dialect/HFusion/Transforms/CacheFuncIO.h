//===----- CacheFuncIO.h - cache func input and output args -----*- C++ -*-===//
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

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_CACHEFUNCIO_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_CACHEFUNCIO_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace hfusion {
/// Apply caching to the input and output of the target function.
/// When \p annotate is true, the caching op will be annotated.
void cacheFuncIO(func::FuncOp funcOp, bool annotate = false,
                 bool writeUnique = false);
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_CACHEFUNCIO_H
