//===- FusibleBlockAnalyzer.h ------------------------------------*- C++-*-===//
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

#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleBlock.h"
#include "bishengir/Dialect/HFusion/Transforms/OpFusion/FusibleHelper.h"
#include "bishengir/Dialect/HFusion/Transforms/Passes.h"

#ifndef BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCKANALYZER_H
#define BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCKANALYZER_H

namespace mlir {
namespace hfusion {
namespace opfusion {
class FusibleBlockAnalyzer {
public:
  FusibleBlockAnalyzer(Block &block, const FusibleHelper &fusibleHelper);

  SmallVector<SetVector<Operation *>> fuseBlock();
  FusibleBlocks getFusibleBlocks();
  bool isFusible();

private:
  using AdjacentEdge = DenseMap<size_t, bool>;
  using AdjacencyList = SmallVector<AdjacentEdge>;
  using ReverseAdjacencyList = SmallVector<DenseSet<size_t>>;

  int find(size_t x);
  int getSize(size_t x);
  bool reInitEdges();
  void mergeEdge(int nodeA, int nodeB);
  bool checkGroupRequirements(const SetVector<Operation *> &group);
  void join(int nodeA, int nodeB, bool isHorizontal = false);
  bool isRestrictedByDependency(int startNode, int endNode,
                                bool horizontal = false);
  bool isRestrictedByShapePivot(int nodeA, int nodeB);
  bool verifyRulesAndJoin(int nodeU, int nodeV, bool horizontal = false);
  void reinitTopoRank();

private:
  const FusibleHelper *fusibleHelper_;
  mutable SetVector<Operation *> ops_;
  DenseMap<Operation *, size_t> opIdx_;

  SmallVector<int, 8> disjointSet_;
  SmallVector<uint8_t, 8> setType_;

  // Record for reduce op specially
  SmallVector<int, 8> opMaxRank_;
  SmallVector<int, 8> opReduceDim_;

  SmallVector<int, 8> importantSize_;
  SmallVector<int, 8> shapePivot_;
  SmallVector<int, 8> topoRank_;

  AdjacencyList edge_;
  ReverseAdjacencyList revEdge_;

  int importantOpsNum_{0};
};
} // namespace opfusion
} // namespace hfusion
} // namespace mlir

#endif // BISHENGIR_DIALECT_HFUSION_TRANSFORMS_OPFUSION_FUSIBLEBLOCKANALYZER_H
