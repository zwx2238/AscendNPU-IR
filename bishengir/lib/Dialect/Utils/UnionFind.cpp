//===- UnionFind.h - Union Find/Disjoint Set Impl. --------------*- C++ -*-===//
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

#include "bishengir/Dialect/Utils/UnionFind.h"

#include "llvm/ADT/STLExtras.h"

int UnionFindBase::find(int x) {
  allocateMinimum(x);
  if (parent_[x] < 0)
    return x;
  return parent_[x] = find(parent_[x]);
}

bool UnionFindBase::join(int a, int b) {
  allocateMinimum(std::max(a, b));

  a = find(a);
  b = find(b);
  if (a != b) {
    if (parent_[a] > parent_[b])
      std::swap(a, b);
    parent_[a] += parent_[b];
    parent_[b] = a;
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
  }
  return true;
}

void UnionFindBase::allocateMinimum(size_t n) {
  if (n + 1 > parent_.size()) {
    parent_.resize(n + 1, -1);
    size_t oldSize = minIndex.size();
    minIndex.resize(n + 1, -1);
    for (size_t i = oldSize; i < n + 1; ++i) {
      assert(minIndex[i] == -1);
      minIndex[i] = static_cast<int>(i);
    }
  }
}
