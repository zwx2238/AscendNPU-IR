# BiShengIR 构建指南

本指南介绍如何在不同平台上编译 BiShengIR。

## 📋 目录

- [快速开始](#快速开始)
- [系统要求](#系统要求)
- [详细构建步骤](#详细构建步骤)
- [架构支持](#架构支持)
- [常见问题](#常见问题)
- [构建脚本说明](#构建脚本说明)

---

## 🚀 快速开始

### 使用自动构建脚本（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/zwx2238/AscendNPU-IR.git
cd AscendNPU-IR

# 2. 运行自动构建脚本
./build-tools/auto-build.sh
```

脚本会自动：
- ✅ 检查所有必要工具和版本
- ✅ 准备 LLVM 源码
- ✅ 应用所有补丁
- ✅ 配置构建
- ✅ 并行编译
- ✅ 运行测试验证

### 使用原始构建脚本

```bash
# 适合高级用户
./build-tools/build.sh -o ./build --build-type Release
```

---

## 💻 系统要求

### BiShengIR 工具说明

BiShengIR 提供三个主要工具，但**分发策略不同**：

| 工具 | 用途 | 系统安装包 | 源码编译 |
|------|------|-----------|---------|
| **bishengir-opt** | IR 转换和优化（开发工具） | ❌ 通常不包含 | ✅ 包含 |
| **bishengir-compile** | 端到端编译器（生产工具） | ✅ 包含 | ✅ 包含 |
| **bishengir-hivm-compile** | HIVM 后端（需要 NPU） | ✅ 包含 | ❌ 不包含 |

**重要说明**：
- 🏢 **官方安装包**：通常只包含 `bishengir-compile` 和 `bishengir-hivm-compile`
- 🔨 **源码编译**：会生成 `bishengir-opt` 和 `bishengir-compile`（不包含 `bishengir-hivm-compile`）
- 💡 **bishengir-opt** 是开发/调试工具，用于：
  - 验证 MLIR 语法
  - 单独运行各个 transformation pass
  - 查看中间 IR 表示
  - 开发和调试新的方言/pass

**如果你只需要编译功能**：使用系统安装的 `bishengir-compile` 即可

**如果你需要开发/调试**：需要从源码编译获得 `bishengir-opt`

### 必要工具

| 工具 | 最低版本 | 推荐版本 | 说明 |
|------|---------|---------|------|
| **CMake** | 3.28.0 | 最新 | 构建系统 |
| **Ninja** | 1.12.0 | 最新 | 构建工具 |
| **Clang** | 15.0 | 最新 | C/C++ 编译器 |
| **Git** | 2.0 | 最新 | 版本控制 |
| **Python** | 3.6 | 3.8+ | 用于安装工具 |

### 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **内存** | 8 GB | 16 GB+ |
| **磁盘空间** | 20 GB | 50 GB+ |
| **CPU 核心** | 4 核 | 8 核+ |

### 支持的平台

- ✅ **x86_64** (Intel/AMD 64位)
- ✅ **aarch64** (ARM 64位，如昇腾服务器)
- ✅ Linux (Ubuntu 20.04+, CentOS 8+)

---

## 📦 详细构建步骤

### 步骤 1: 安装依赖

#### Ubuntu/Debian

```bash
# 基础工具
sudo apt-get update
sudo apt-get install -y git clang

# 升级 CMake 和 Ninja (系统版本通常过旧)
pip install cmake --upgrade
pip install ninja --upgrade

# 验证版本
cmake --version  # 应该 >= 3.28
ninja --version  # 应该 >= 1.12
```

#### CentOS/RHEL

```bash
# 基础工具
sudo yum install -y git clang

# 升级工具
pip install cmake --upgrade
pip install ninja --upgrade
```

#### macOS

```bash
# 使用 Homebrew
brew install git cmake ninja llvm

# 设置 Clang
export PATH="/usr/local/opt/llvm/bin:$PATH"
```

### 步骤 2: 克隆仓库

```bash
git clone https://github.com/zwx2238/AscendNPU-IR.git
cd AscendNPU-IR
```

### 步骤 3: 准备 LLVM 源码

#### 选项 A: 使用 Git Submodule（官方方式，但较慢）

```bash
git submodule update --init --recursive third-party/llvm-project
```

#### 选项 B: 从本地复制（更快）

```bash
# 如果你已经有 llvm-project 仓库
cp -r ~/llvm-project third-party/llvm-project
```

#### 选项 C: 使用镜像站（中国用户推荐）

```bash
# 克隆 LLVM 镜像
git clone https://gitcode.com/gh_mirrors/ll/llvm-project.git third-party/llvm-project
```

### 步骤 4: 切换 LLVM 版本并应用补丁

```bash
cd third-party/llvm-project
git checkout cd708029e0b2869e80abe31ddb175f7c35361f90
cd ../..

# 应用 BiShengIR 补丁（66个补丁）
cd third-party/llvm-project
for patch in ../../build-tools/patches/llvm-project/*.patch; do
    git apply "$patch"
done
cd ../..
```

### 步骤 5: 配置构建

```bash
mkdir -p build
cd build

cmake -G Ninja ../third-party/llvm-project/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$(pwd)/.." \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=OFF
```

### 步骤 6: 编译

```bash
# 使用所有 CPU 核心编译
ninja -j$(nproc) bishengir-opt bishengir-compile

# 或指定核心数，例如 8 核
ninja -j8 bishengir-opt bishengir-compile
```

**编译时间参考**：
- 8 核 CPU: 约 30-60 分钟
- 16 核 CPU: 约 15-30 分钟
- aarch64: 可能需要更长时间

### 步骤 7: 验证

```bash
# 检查编译产物
ls -lh bin/bishengir-opt
ls -lh bin/bishengir-compile

# 测试工具
./bin/bishengir-opt --version
./bin/bishengir-opt ../examples/math_examples.mlir
```

---

## 🏗️ 架构支持

### x86_64 (Intel/AMD)

标准 x86_64 平台，编译最快。

```bash
# 检查架构
uname -m  # 应显示: x86_64

# 正常编译
./build-tools/auto-build.sh
```

### aarch64 (ARM64)

昇腾服务器通常使用 aarch64 架构。

```bash
# 检查架构
uname -m  # 应显示: aarch64

# aarch64 编译注意事项：
# 1. 确保使用 ARM 版本的 clang
# 2. 编译时间可能更长
# 3. 内存需求相同

# 编译
./build-tools/auto-build.sh
```

**aarch64 特殊说明**：
- ✅ 完全支持
- ⚠️ 编译时间可能是 x86_64 的 1.5-2 倍
- ✅ 运行时性能与架构优化相关
- ✅ 生成的工具可在同架构机器间移植

---

## ❓ 常见问题

### 1. CMake 版本过低

**错误**:
```
CMake 3.28.0 or higher is required. You are running version 3.22.1
```

**解决**:
```bash
pip install cmake --upgrade
cmake --version  # 验证
```

### 2. Ninja 版本过低

**错误**:
```
Ninja version too old, need at least 1.12.0
```

**解决**:
```bash
pip install ninja --upgrade
ninja --version  # 验证
```

### 3. 内存不足

**错误**:
```
c++: fatal error: Killed signal terminated program cc1plus
```

**解决**:
```bash
# 减少并行任务数
ninja -j4 bishengir-opt bishengir-compile

# 或增加 swap
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 4. LLVM 下载缓慢

**解决**:
```bash
# 方法 1: 使用镜像
git clone https://gitcode.com/gh_mirrors/ll/llvm-project.git third-party/llvm-project

# 方法 2: 从其他机器复制
scp -r user@other-machine:~/llvm-project third-party/

# 方法 3: 使用代理
git config --global http.proxy http://proxy:port
```

### 5. 补丁应用失败

**错误**:
```
error: patch failed: ...
```

**解决**:
```bash
# 补丁可能已应用，继续构建即可
cd third-party/llvm-project
git status  # 查看修改

# 或重置后重新应用
git reset --hard
git checkout cd708029e0b2869e80abe31ddb175f7c35361f90
for patch in ../../build-tools/patches/llvm-project/*.patch; do
    git apply "$patch"
done
```

### 6. aarch64 编译很慢

**正常现象**，可以：
- 使用更多 CPU 核心: `ninja -j16`
- 使用 Release 构建（已默认）
- 增加内存避免 swap
- 使用更快的存储（SSD）

### 7. 找不到 clang

**解决**:
```bash
# Ubuntu/Debian
sudo apt-get install clang

# CentOS/RHEL
sudo yum install clang

# macOS
brew install llvm
export PATH="/usr/local/opt/llvm/bin:$PATH"

# 验证
which clang
clang --version
```

---

## 📜 构建脚本说明

### `auto-build.sh`（推荐）

**特点**：
- ✅ 全自动化
- ✅ 详细的进度提示
- ✅ 自动检查依赖
- ✅ 架构检测
- ✅ 错误提示清晰
- ✅ 包含测试验证

**使用**：
```bash
./build-tools/auto-build.sh
```

### `build.sh`（原始脚本）

**特点**：
- 更底层的控制
- 需要手动准备依赖
- 适合高级用户

**使用**：
```bash
./build-tools/build.sh -o ./build --build-type Release
```

---

## 🎯 构建后步骤

### 1. 添加到 PATH（可选）

```bash
# 临时添加
export PATH="$PWD/build/bin:$PATH"

# 永久添加（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export PATH="/path/to/AscendNPU-IR/build/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 运行示例

```bash
# 运行演示
./examples/demo.sh

# 测试转换
./build/bin/bishengir-opt examples/math_examples.mlir --convert-math-to-hfusion

# 查看教程
cat examples/TUTORIAL.md
```

### 3. 开始开发

查看 [CLAUDE.md](../CLAUDE.md) 和 [examples/README.md](../examples/README.md)

---

## 📚 相关文档

- **[CLAUDE.md](../CLAUDE.md)** - 开发指南
- **[examples/TUTORIAL.md](../examples/TUTORIAL.md)** - 入门教程
- **[examples/QUICK_REFERENCE.md](../examples/QUICK_REFERENCE.md)** - 快速参考
- **[examples/HIVM_COMPILE_GUIDE.md](../examples/HIVM_COMPILE_GUIDE.md)** - 编译指南

---

## 💡 提示

1. **首次编译**需要较长时间（30-60分钟），后续增量编译会很快
2. **aarch64** 平台编译时间会更长，请耐心等待
3. 如果编译失败，查看错误信息并参考**常见问题**部分
4. 编译成功后，工具可以在**没有 NPU 硬件**的机器上运行 MLIR 转换
5. **bishengir-hivm-compile** 需要从有 NPU 的机器获取

---

## 🆘 获取帮助

如果遇到问题：

1. 查看本文档的**常见问题**部分
2. 查看 [GitHub Issues](https://github.com/zwx2238/AscendNPU-IR/issues)
3. 运行 `./build-tools/auto-build.sh` 查看详细错误信息

---

**祝编译顺利！** 🎉
