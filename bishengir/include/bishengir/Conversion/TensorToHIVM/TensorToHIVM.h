//===------- TensorToHIVM.h - Tensor to HIVM conversion ---------*- C++ -*-===//
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

#ifndef BISHENGIR_CONVERSION_TENSORTOHIVM_TENSORTOHIVM_H
#define BISHENGIR_CONVERSION_TENSORTOHIVM_TENSORTOHIVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTTENSORTOHIVM
#include "bishengir/Conversion/Passes.h.inc"

namespace hivm {
void populateTensorToHIVMConversionPatterns(RewritePatternSet &patterns);
} // namespace hivm

/// Creates a pass to convert certain tensor ops to hivm ops
std::unique_ptr<Pass> createTensorToHIVMConversionPass();

} // namespace mlir

#endif // BISHENGIR_CONVERSION_TENSORTOHIVM_TENSORTOHIVM_H