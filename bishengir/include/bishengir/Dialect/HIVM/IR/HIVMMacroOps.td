//===- HIVMMacroOps.td - HIVM dialect macro operations -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for HIVM dialect macro operations.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_IR_HIVMMACROOPS_TD
#define BISHENGIR_DIALECT_HIVM_IR_HIVMMACROOPS_TD

include "bishengir/Dialect/HIVM/IR/HIVMBase.td"
include "bishengir/Dialect/HIVM/IR/HIVMAttrs.td"
include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.td"
include "bishengir/Dialect/HIVM/Interfaces/OpLayoutInterface.td"
include "bishengir/Dialect/HIVM/IR/HIVMTraits.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// HIVM Macro Operations.
//===----------------------------------------------------------------------===//

class HIVM_MacroOp<string mnemonic, list<Trait> traits = []> :
  HIVM_StructuredOp<mnemonic, !listconcat(
    [MacroOpTrait,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>
    ],
    traits)
  >;

//===----------------------------------------------------------------------===//
// HIVM Local MMAD op definitions.
//===----------------------------------------------------------------------===//

class HIVM_LocalMmadOp<string mnemonic, list<Trait> traits = []> :
  HIVM_MacroOp<mnemonic, !listconcat(
    [MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">,
     CubeCoreTypeTrait, AttrSizedOperandSegments
    ], traits)> {
  let arguments = (ins Arg<TensorOrMemref, "Matrix A">:$a,
                       Arg<TensorOrMemref, "Matrix B">:$b,
                       Arg<I1, [{
                        The condition under which
                        data in L0C is cleared before use
                       }]>:$init_condition,
                       Index:$real_m,
                       Index:$real_k,
                       Index:$real_n,
                       Arg<TensorOrMemref, "Matrix C">:$c,
                       Optional<TensorOrMemref>:$per_channel_bias,
                       Arg<OptionalAttr<UnitAttr>, [{
                        Matrix A is transposed before load
                       }]>:$a_transpose,
                       Arg<OptionalAttr<UnitAttr>, [{
                        Matrix B is transposed before load
                       }]>:$b_transpose,
                       Arg<OptionalAttr<UnitAttr>, [{
                        MMAD HF32 mode control.If true, FP32 data will be rounded
                        to HF32 by CUBE before matrix multiplication, which achieves
                        double performance of MMAD.F32 but losing precision.
                       }]>:$enable_HF32,
                       Arg<Variadic<I64>, [{
                        Arguments required by mma_tile to insert 
                        synchronization instructions, they are 
                        managed by inject-sync passes.
                       }]>:$sync_related_args);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  code localMmadBaseDecls = [{
    // Implement functions necessary for DestinationStyleOpInterface.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getCMutable();
    }
    bool isInitConstant(std::optional<bool> cst = std::nullopt);
    void setInitCondition(Value init);
    MatmulBiasMode getMatmulBiasMode();
    bool shouldDecomposeBiasByElementAdd();
    int getNumSyncRelatedArgs();
  }];
  string localMmadBaseDes = [{
    The computation logic is:

    ```
    C = C + A x B + (optional) channel_bias
    ```
  }];
  let assemblyFormat = [{
    attr-dict `ins` `(`
    $a
    `,` $b
    `,` $init_condition
    `,` $real_m
    `,` $real_k
    `,` $real_n
    (`,` $per_channel_bias^)?
    `:`
    type($a)
    `,` type($b)
    `,` type($init_condition)
    `,` type($real_m)
    `,` type($real_k)
    `,` type($real_n)
    (`,` type($per_channel_bias)^)? `)`
    (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
    `outs` `(` $c `:` type($c) `)`
    (`->` type($result_tensors)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result_tensors,
                  "Value":$a,
                  "Value":$b,
                  "Value":$init_condition,
                  "Value":$real_m,
                  "Value":$real_k,
                  "Value":$real_n,
                  "Value":$c,
                  CArg<"Value", "{}">:$per_channel_bias,
                  CArg<"UnitAttr", "{}">:$a_transpose,
                  CArg<"UnitAttr", "{}">:$b_transpose,
                  CArg<"UnitAttr", "{}">:$enable_HF32)>
  ];

  let hasVerifier = 1;
}

def MmadL1Op : HIVM_LocalMmadOp<"mmadL1",
  [OpLayoutInterface]> {
  let summary = [{
    Matrix Multiply and Add Op with inputs from L1 memory hierarchy.
  }];
  let description = localMmadBaseDes # [{
    Note: the rank of A, B, and C Matrix must be two.
  }];
  let extraClassDeclaration = localMmadBaseDecls # [{
    static StringRef getOpName() { return "mma_tile"; }

    // Declare functions necessary for OpLayoutInterface.
    FailureOr<DataLayoutAttr> getOperandALayout();
    FailureOr<DataLayoutAttr> getOperandBLayout();
    FailureOr<DataLayoutAttr> getOperandCLayout();
    FailureOr<DataLayoutAttr> getOperandBiasLayout();
    SmallVector<DataLayoutAttr> getOperandsTargetLayout();
    SmallVector<DataLayoutAttr> getOperandsCurrentLayout();
  }];
}

def BatchMmadL1Op : HIVM_LocalMmadOp<"batchMmadL1"> {
  let summary = [{
    Batch Matrix Multiply and Add Op with inputs from L1 memory hierarchy.
  }];
  let description = localMmadBaseDes # [{
    Note: the rank of A, B, and C Matrix must be three, where the 0-th dimension
    being the batch dimension.
  }];
  let extraClassDeclaration = localMmadBaseDecls;
}

#endif // BISHENGIR_DIALECT_HIVM_IR_HIVMMACROOPS_TD
