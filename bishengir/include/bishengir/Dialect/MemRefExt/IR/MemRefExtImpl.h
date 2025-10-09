//===- MemRefExtImpl.h - MemRefExt dialect Implementation -----------------===//
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

#ifndef BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXTIMPL_H
#define BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXTIMPL_H

#include "mlir/IR/Value.h"

namespace mlir {
namespace memref_ext {

/// Determine whether the current buffer is defined by one of the following:
///   - memref.alloc
///   - memref_ext.alloc_workspace
bool isDefiningOpAllocLike(Value operand);

} // namespace memref_ext
} // namespace mlir

#endif // BISHENGIR_DIALECT_MEMREF_IR_MEMREFEXTIMPL_H
