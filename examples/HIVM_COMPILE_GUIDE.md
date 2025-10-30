# BiShengIR HIVM 编译指南

## 目录
1. [工具说明](#工具说明)
2. [环境设置](#环境设置)
3. [基础用法](#基础用法)
4. [完整编译流程](#完整编译流程)
5. [常见选项](#常见选项)
6. [测试脚本](#测试脚本)

---

## 工具说明

BiShengIR 提供三个主要工具：

### 1. `bishengir-opt`
- **用途**: MLIR IR 转换和优化
- **位置**: `build/bin/bishengir-opt`
- **功能**:
  - 验证 MLIR 语法
  - 方言间转换（Math → HFusion → HIVM）
  - 应用各种优化 pass

### 2. `bishengir-compile`
- **用途**: 端到端编译器前端
- **位置**: `build/bin/bishengir-compile`
- **功能**:
  - 高层 IR 编译
  - 调用 bishengir-hivm-compile 后端
  - 生成 LLVM IR 或目标代码

### 3. `bishengir-hivm-compile`
- **用途**: HIVM 后端编译器
- **位置**: 需要从其他机器复制或安装
- **功能**:
  - HIVM → HACC 转换
  - 生成昇腾 NPU 特定代码
  - 硬件相关优化

---

## 环境设置

### 方法 1: 添加到 PATH（推荐）

```bash
# 假设你从另一台机器复制了 bishengir-hivm-compile 到 /path/to/bin
export PATH=/path/to/bin:$PATH

# 验证
which bishengir-hivm-compile
# 应该输出: /path/to/bin/bishengir-hivm-compile
```

### 方法 2: 设置 BISHENG_INSTALL_PATH

```bash
# 如果 bishengir-hivm-compile 在 /opt/bisheng/bin/ 下
export BISHENG_INSTALL_PATH=/opt/bisheng

# bishengir-compile 会自动查找 $BISHENG_INSTALL_PATH/bin/bishengir-hivm-compile
```

### 方法 3: 创建符号链接

```bash
# 复制到系统路径
sudo cp /path/to/bishengir-hivm-compile /usr/local/bin/

# 或创建符号链接
sudo ln -s /path/to/bishengir-hivm-compile /usr/local/bin/
```

### 验证安装

```bash
# 检查是否可用
command -v bishengir-hivm-compile
bishengir-hivm-compile --help
```

---

## 基础用法

### 1. 验证 HIVM IR

```bash
./build/bin/bishengir-opt examples/hivm_add.mlir
```

### 2. 基础编译（无需 bishengir-hivm-compile）

```bash
./build/bin/bishengir-compile \
  -enable-lir-compile=false \
  examples/hivm_add.mlir \
  -o output.ll
```

### 3. 完整 HIVM 编译（需要 bishengir-hivm-compile）

```bash
./build/bin/bishengir-compile \
  -enable-hivm-compile=true \
  -target=Ascend910B1 \
  examples/hivm_add.mlir \
  -o output.ll
```

---

## 完整编译流程

### 流程图

```
源代码 (Math/Arith)
    ↓ bishengir-opt --convert-*-to-hfusion
HFusion IR
    ↓ bishengir-opt --convert-hfusion-to-hivm
HIVM IR
    ↓ bishengir-compile -enable-hivm-compile=true
LLVM IR
    ↓ llc
目标代码 (.o)
```

### 示例：从 Arith 到目标代码

```bash
# 步骤 1: Arith → HFusion → HIVM
./build/bin/bishengir-opt examples/simple_add.mlir \
  --convert-arith-to-hfusion \
  --convert-hfusion-to-hivm \
  > /tmp/hivm.mlir

# 步骤 2: 验证生成的 HIVM IR
./build/bin/bishengir-opt /tmp/hivm.mlir

# 步骤 3: HIVM → LLVM IR（需要 bishengir-hivm-compile）
./build/bin/bishengir-compile \
  -enable-hivm-compile=true \
  -target=Ascend910B1 \
  /tmp/hivm.mlir \
  -o /tmp/output.ll

# 步骤 4: LLVM IR → 目标代码
llc /tmp/output.ll -o /tmp/output.o
```

---

## 常见选项

### bishengir-compile 选项

#### 编译模式
```bash
-enable-hivm-compile=true        # 启用 HIVM 编译（需要 bishengir-hivm-compile）
-enable-lir-compile=false        # 禁用 LIR 编译
-enable-hfusion-compile=true     # 启用 HFusion 编译
```

#### 目标设备
```bash
-target=Ascend910B1              # Ascend 910B1
-target=Ascend910B2              # Ascend 910B2
-target=Ascend910B3              # Ascend 910B3
-target=Ascend910B4              # Ascend 910B4
```

#### 优化选项
```bash
-enable-auto-multi-buffer=true           # 自动多缓冲优化
-enable-static-bare-ptr=true             # 静态裸指针优化
-enable-deterministic-computing=true     # 确定性计算
-enable-ops-reorder=true                 # 操作重排序优化
```

#### 调试选项
```bash
-enable-debug-info=true                  # 生成调试信息
-enable-sanitizer=true                   # 启用内存检查
-bishengir-print-ir-after=all            # 打印所有 pass 后的 IR
```

#### HIVM 特定优化
```bash
-enable-hivm-auto-cv-balance=true        # CV 流水线平衡
-enable-hivm-auto-storage-align=true     # 存储对齐
-enable-hivm-global-workspace-reuse=true # 全局工作空间复用
-set-workspace-multibuffer=4             # 设置工作空间多缓冲数量
```

---

## 测试脚本

### 1. 快速测试

```bash
# 运行快速测试脚本
./examples/quick_test.sh
```

该脚本会：
- 验证 HIVM IR
- 尝试基础编译
- 尝试完整 HIVM 编译（如果 bishengir-hivm-compile 可用）

### 2. 完整测试

```bash
# 运行完整测试脚本
./examples/test_hivm_compile.sh
```

该脚本会：
- 检查所有工具
- 显示输入文件内容
- 执行多种编译配置
- 显示编译选项示例
- 提供设置指南

### 3. 基础演示

```bash
# 运行入门演示
./examples/demo.sh
```

---

## 命令速查表

### IR 转换
```bash
# Math → HFusion
bishengir-opt input.mlir --convert-math-to-hfusion

# Arith → HFusion
bishengir-opt input.mlir --convert-arith-to-hfusion

# HFusion → HIVM
bishengir-opt input.mlir --convert-hfusion-to-hivm

# 完整链路
bishengir-opt input.mlir \
  --convert-arith-to-hfusion \
  --convert-hfusion-to-hivm
```

### 编译
```bash
# 基础编译
bishengir-compile -enable-lir-compile=false input.mlir -o output.ll

# HIVM 编译
bishengir-compile \
  -enable-hivm-compile=true \
  -target=Ascend910B1 \
  input.mlir \
  -o output.ll

# 带优化
bishengir-compile \
  -enable-hivm-compile=true \
  -target=Ascend910B1 \
  -enable-auto-multi-buffer=true \
  -enable-deterministic-computing=true \
  input.mlir \
  -o output.ll
```

### 调试
```bash
# 查看所有转换过程
bishengir-compile \
  -bishengir-print-ir-after=all \
  input.mlir \
  -o output.ll 2>&1 | less

# 查看特定 pass 后的 IR
bishengir-opt input.mlir \
  --convert-arith-to-hfusion \
  --mlir-print-ir-after-all
```

---

## 故障排查

### 问题 1: bishengir-hivm-compile 未找到

**错误信息**:
```
[ERROR] Cannot find bishengir-hivm-compile under $PATH
```

**解决方案**:
```bash
# 检查是否在 PATH 中
which bishengir-hivm-compile

# 如果没有，添加到 PATH
export PATH=/path/to/bin:$PATH

# 或设置 BISHENG_INSTALL_PATH
export BISHENG_INSTALL_PATH=/path/to/bisheng/install
```

### 问题 2: 编译失败

**检查清单**:
1. 确保输入 MLIR 文件语法正确
   ```bash
   bishengir-opt input.mlir
   ```

2. 检查是否有必要的属性
   ```mlir
   // DEVICE 函数必须有这些属性
   attributes {
     hacc.entry,
     hacc.function_kind = #hacc.function_kind<DEVICE>
   }
   ```

3. 使用调试选项查看详细信息
   ```bash
   bishengir-compile \
     -enable-hivm-compile=true \
     -bishengir-print-ir-after=all \
     input.mlir \
     -o output.ll 2>&1 | tee debug.log
   ```

### 问题 3: 目标设备不支持

**检查支持的目标**:
```bash
bishengir-compile --help | grep -A 10 "target="
```

**可用目标**:
- Ascend910B1, Ascend910B2, Ascend910B3, Ascend910B4
- Ascend910_9362, Ascend910_9372
- Ascend910_9381, Ascend910_9382
- Ascend910_9391, Ascend910_9392

---

## 示例文件

### 1. `hivm_add.mlir` - HIVM 向量加法
完整的 HIVM 层代码，包含：
- 全局内存 (gm) 输入/输出
- UB 缓冲区分配
- load/vadd/store 操作

### 2. `simple_add.mlir` - Arith 加法
高层 Arith 方言代码，可转换到 HIVM

### 3. `math_examples.mlir` - Math 操作
Math 方言示例（exp, sqrt）

### 4. `arith_examples.mlir` - Arith 操作
完整的算术运算示例（add, mul, sub, div）

---

## 更多资源

- **教程**: `examples/TUTORIAL.md`
- **快速参考**: `examples/QUICK_REFERENCE.md`
- **开发指南**: `CLAUDE.md`
- **测试用例**: `bishengir/test/`
  - `bishengir/test/Conversion/` - 转换测试
  - `bishengir/test/Dialect/HIVM/` - HIVM 方言测试
  - `bishengir/test/bishengir-compile/` - 编译测试

---

## 快速开始

```bash
# 1. 设置环境（假设你已经复制了 bishengir-hivm-compile）
export PATH=/path/to/bisheng/bin:$PATH

# 2. 验证安装
which bishengir-hivm-compile

# 3. 运行快速测试
./examples/quick_test.sh

# 4. 如果成功，尝试完整测试
./examples/test_hivm_compile.sh

# 5. 手动编译示例
./build/bin/bishengir-compile \
  -enable-hivm-compile=true \
  -target=Ascend910B1 \
  examples/hivm_add.mlir \
  -o /tmp/output.ll

# 6. 查看生成的 LLVM IR
cat /tmp/output.ll
```

---

**提示**: 如果你没有 `bishengir-hivm-compile`，仍然可以：
1. 使用 `bishengir-opt` 进行 IR 转换（Math → HFusion → HIVM）
2. 使用 `bishengir-compile -enable-lir-compile=false` 进行基础编译
3. 学习 MLIR 方言和转换 pass

只有在需要生成最终 NPU 二进制代码时才必须使用 `bishengir-hivm-compile`。
