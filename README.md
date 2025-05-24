# The AscendNPU IR (BiSheng IR) Project

## Where The AscendNPU IR (BiSheng IR) Is In CANN

![](./doc/pic/ascendnpu-ir-in-cann.png)

## Using AscendNPU IR (BiSheng IR)

### Installing pre-builts that are required to build BiShengIR

1. Extract the package (available in the release page) containing the pre-builts corresponding to your target machine to any location. After install, it should contain the following contents:

   ```bash
   ├── lib
     └── libBiShengIR.so     // used to build bishengir dialects
   └── bin
     └── bishengir-yaml-gen  // used to generate files from yaml
   ```

2. Set environment variable to the installed path:

  ```bash
  export BISHENG_IR_INSTALL_PATH = ...
  ```


### Building BiShengIR as an external LLVM Project

1. Find the version of LLVM that BiShengIR builds against. Check `cmake/llvm-release-tag.txt` to see the current version.
  
    For example, if it says: "llvm.18.1.3", it means that the version of BiShengIR you have builds against [LLVM](https://github.com/llvm/llvm-project/tree/llvmorg-18.1.3) release `llvmorg-18.1.3`.

2. `git checkout` LLVM at this revision. Optionally, make additional modifications to LLVM.

3. Add `bishengir` project as a third-party submodule to LLVM.

    ```bash
    git submodule add https://gitee.com/ascend/ascendnpu-ir.git third-party/bishengir
    ```

4. Apply `bishengir` related patch files to LLVM.

    ```bash
    git apply third-party/bishengir/cmake/bishengir.patch
    ```

5. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run the following to build the example:

    ```bash
    cd $HOME/llvm-project  # your clone of LLVM.
    mkdir build
    cd build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
      -DLLVM_INCLUDE_EXAMPLES=ON \
      -DLLVM_EXTERNAL_PROJECTS=bishengir \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DBISHENG_IR_INSTALL_PATH=${BISHENG_IR_INSTALL_PATH}
    ```

6. Run the following test case to see if the build is successful:

   ```bash
   ./bin/bishengir-minimal-opt ../third-party/bishengir/test/Examples/hfusion.mlir
   ```
