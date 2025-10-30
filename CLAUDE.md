# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AscendNPU IR (BiShengIR) 是基于 MLIR 构建的昇腾 NPU 中间表示层,用于编译昇腾亲和算子。项目作为 LLVM 外部项目构建,提供多级抽象接口支持从高层算子到底层硬件指令的编译优化。

## 构建命令

### 初次构建
```bash
./build-tools/build.sh -o ./build --build-type Debug --apply-patches
```

### 后续构建
```bash
./build-tools/build.sh -o ./build --build-type Debug
```

### 手动构建 (高级用户)
```bash
mkdir -p build && cd build
cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$(pwd)/.." \
    -DBSPUB_DAVINCI_BISHENGIR=ON
ninja -j32
```

**重要参数:**
- `--apply-patches`: 首次编译时必须使用,启用第三方仓库扩展功能
- `-DBSPUB_DAVINCI_BISHENGIR=ON`: 必须启用,用于昇腾扩展功能
- `--build-torch-mlir`: 启用 Torch-MLIR 转换支持
- `--python-binding`: 启用 Python 绑定
- `--build-test`: 构建测试目标

### 环境要求
- CMake >= 3.28
- Ninja >= 1.12.0
- Clang >= 10
- 构建示例需要设置 `ASCEND_HOME_PATH` 环境变量

## 测试

### 运行所有测试
```bash
# 在 build 目录下
cmake --build . --target check-bishengir
```

### 使用 llvm-lit 运行测试
```bash
# 在 build 目录下
./bin/llvm-lit ../bishengir/test
```

### 运行单个测试
```bash
./bin/llvm-lit ../bishengir/test/path/to/test.mlir
```

## 代码架构

### 核心方言 (Dialects)

项目采用三层抽象架构,通过方言间的转换实现从高层表示到底层硬件指令的映射:

1. **HFusion (High-level Fusion)**: 高层算子方言
   - 提供硬件无关的结构化算子抽象 (如 matmul, conv, reduce 等)
   - 支持 Transform Dialect 扩展用于算子变换和优化
   - 路径: `bishengir/include/bishengir/Dialect/HFusion/`

2. **HIVM (Hardware Intermediate Virtual Machine)**: 硬件中间表示
   - 提供昇腾计算、数据搬运 (DMA)、同步操作的抽象
   - 包含向量操作、宏操作、流水线管理
   - 路径: `bishengir/include/bishengir/Dialect/HIVM/`

3. **HACC (Hardware Acceleration)**: 硬件加速层
   - 底层硬件指令映射
   - NPU 目标规格定义
   - 路径: `bishengir/include/bishengir/Dialect/HACC/`

### 其他重要方言

- **Annotation**: 元数据标注
- **Symbol**: 符号管理
- **MemRef/MemRefExt**: 内存引用扩展
- **Tensor**: 张量操作扩展
- **MathExt**: 数学操作扩展
- **Torch**: PyTorch 集成支持

### Conversion Passes (方言转换)

关键转换路径位于 `bishengir/lib/Conversion/`:

- `LinalgToHFusion`: Linalg 算子转换到 HFusion
- `TorchToHFusion`: PyTorch 算子转换到 HFusion
- `ArithToHFusion`/`MathToHFusion`: 算术/数学运算转换
- `HFusionToHIVM`: 高层算子 lowering 到硬件中间表示
- `TensorToHIVM`: Tensor 操作直接转换到 HIVM

典型编译流水线: **Torch/Linalg → HFusion → HIVM → HACC → 机器码**

### 主要工具

位于 `bishengir/tools/`:

- **bishengir-opt**: Pass 优化工具 (类似 mlir-opt)
- **bishengir-compile**: 端到端编译器
- **bishengir-lsp-server**: LSP 语言服务器
- **bishengir-target-spec-tblgen**: 目标规格 TableGen 工具
- **bishengir-hfusion-ods-gen**: HFusion 算子定义生成器

### TableGen 定义

所有 `*.td` 文件定义方言、算子、Pass、属性等:
- 算子定义: `*Ops.td`
- 基础定义: `*Base.td`
- 属性定义: `*Attrs.td`
- Pass 定义: `Passes.td`
- 接口定义: `*Interfaces.td`

## 开发工作流

### 添加新算子
1. 在相应方言的 `.td` 文件中定义算子 (如 `HFusionOps.td`)
2. 在 `lib/Dialect/*/IR/` 实现算子语义验证和 canonicalization
3. 在 `lib/Conversion/` 添加转换逻辑
4. 在 `test/Dialect/*/` 添加 FileCheck 测试

### 添加新 Pass
1. 在 `Passes.td` 中声明 Pass
2. 在 `lib/Dialect/*/Transforms/` 或 `lib/Conversion/` 实现
3. 在 `test/` 添加测试用例

### 运行单个 Pass 调试
```bash
./build/bin/bishengir-opt path/to/input.mlir -pass-name -debug
```

## 代码风格

- 使用 `.clang-format` 和 `.clang-tidy` 配置
- 遵循 LLVM/MLIR 编码规范
- TableGen 定义需包含详细文档注释

## 子模块管理

项目依赖 LLVM、Torch-MLIR 等第三方库作为 submodule:
```bash
# 初始化和更新子模块
git submodule update --init --recursive
```

## 构建选项说明

- `BISHENGIR_BUILD_EXAMPLES=ON`: 构建端到端示例 (需要 CANN 环境)
- `BISHENGIR_ENABLE_TORCH_CONVERSIONS`: 启用 Torch 转换 (自动检测)
- `BISHENGIR_BUILD_STANDALONE_IR_ONLY=ON`: 仅构建 IR 定义
- `MLIR_ENABLE_BINDINGS_PYTHON=ON`: 启用 Python 绑定
