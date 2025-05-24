# 昇腾NPU IR（毕昇IR）项目

## 昇腾NPU IR（毕昇IR）在CANN中的位置

![](./doc/pic/ascendnpu-ir-in-cann.png)

## 使用昇腾NPU IR（毕昇IR）

### 安装构建毕昇IR所需的预编译组件

1. 将包含与您的目标机器对应的预编译组件的包（可在发布页面获取）解压到任意位置。在安装后，它应当包含如下内容：

   ```bash
   ├── lib
     └── libBiShengIR.so     // used to build bishengir dialects
   └── bin
     └── bishengir-yaml-gen  // used to generate files from yaml
   ```

2. 将环境变量设置为安装路径：

  ```bash
  export BISHENG_IR_INSTALL_PATH = ...
  ```


### 将毕昇IR构建为外部LLVM项目

1. 查找构建毕昇IR所依赖的LLVM版本。请查看`cmake/llvm-release-tag.txt`文件获取当前版本信息。
  
    例如，若显示"llvm.18.1.3"，意味着您当前版本的毕昇IR需要基于[LLVM](https://github.com/llvm/llvm-project/tree/llvmorg-18.1.3)的`llvmorg-18.1.3`发行版构建。

2. 使用`git checkout`命令签出到此版本。根据需要，您可以对LLVM进行额外修改。

3. 将`bishengir`项目作为第三方子模块添加到LLVM

    ```bash
    git submodule add https://gitee.com/ascend/ascendnpu-ir.git third-party/bishengir
    ```

4. 将`bishengir`相关补丁文件应用到LLVM中

    ```bash
    git apply third-party/bishengir/cmake/bishengir.patch
    ```

5. [构建LLVM](https://llvm.org/docs/CMake.html)。例如，您可以运行以下命令来构建示例：

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

6. 运行以下测试用例来查看构建是否成功：

   ```bash
   ./bin/bishengir-minimal-opt ../third-party/bishengir/test/Examples/hfusion.mlir
   ```
