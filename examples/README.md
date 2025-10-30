# BiShengIR 示例和测试脚本

本目录包含 BiShengIR 的示例代码、测试脚本和教程文档。

## 📁 文件结构

### 📘 文档
- **`TUTORIAL.md`** - 完整入门教程
- **`QUICK_REFERENCE.md`** - 快速参考卡
- **`HIVM_COMPILE_GUIDE.md`** - HIVM 编译详细指南

### 🧪 MLIR 示例
- **`math_examples.mlir`** - Math 方言示例（exp, sqrt）
- **`arith_examples.mlir`** - Arith 方言示例（add, mul, sub, div）
- **`simple_add.mlir`** - 简单加法示例
- **`hivm_add.mlir`** - 完整的 HIVM 层向量加法示例

### 🚀 测试脚本
- **`demo.sh`** - 基础演示脚本（入门推荐）
- **`quick_test.sh`** - HIVM 编译快速测试
- **`test_hivm_compile.sh`** - HIVM 编译完整测试

---

## 🎯 快速开始

### 1. 基础演示（推荐入门）

```bash
# 运行基础演示
./examples/demo.sh
```

这个脚本会演示：
- ✅ 工具版本检查
- ✅ MLIR 文件验证
- ✅ Math → HFusion 转换
- ✅ 带规范化的转换
- ✅ 支持的方言列表

### 2. IR 转换示例

```bash
# Math → HFusion
./build/bin/bishengir-opt examples/math_examples.mlir \
  --convert-math-to-hfusion

# Arith → HFusion → HIVM
./build/bin/bishengir-opt examples/simple_add.mlir \
  --convert-arith-to-hfusion \
  --convert-hfusion-to-hivm

# 查看详细转换过程
./build/bin/bishengir-opt examples/arith_examples.mlir \
  --convert-arith-to-hfusion \
  --convert-hfusion-to-hivm \
  --mlir-print-ir-after-all
```

### 3. HIVM 编译测试（需要 bishengir-hivm-compile）

如果你从另一台机器获得了 `bishengir-hivm-compile`：

```bash
# 设置环境
export PATH=/path/to/bisheng/bin:$PATH

# 运行快速测试
./examples/quick_test.sh

# 运行完整测试
./examples/test_hivm_compile.sh
```

---

## 📚 学习路径

### 🟢 初学者
1. 阅读 `TUTORIAL.md`
2. 运行 `./examples/demo.sh`
3. 查看 `math_examples.mlir` 和 `arith_examples.mlir`
4. 尝试修改示例并运行转换

### 🟡 中级
1. 学习 `QUICK_REFERENCE.md` 中的命令
2. 阅读 `hivm_add.mlir` 理解 HIVM 层
3. 探索 `bishengir/test/Conversion/` 下的测试用例
4. 尝试编写自己的 MLIR 代码

### 🔴 高级
1. 阅读 `HIVM_COMPILE_GUIDE.md`
2. 设置 `bishengir-hivm-compile` 并测试完整编译
3. 研究 `bishengir/test/bishengir-compile/` 下的编译测试
4. 探索优化选项和性能调优

---

## 🔧 工具说明

### bishengir-opt
**IR 转换和优化工具**

```bash
# 验证语法
./build/bin/bishengir-opt input.mlir

# 应用转换
./build/bin/bishengir-opt input.mlir --convert-math-to-hfusion

# 查看帮助
./build/bin/bishengir-opt --help
```

### bishengir-compile
**端到端编译器**

```bash
# 基础编译（不需要 bishengir-hivm-compile）
./build/bin/bishengir-compile \
  -enable-lir-compile=false \
  input.mlir \
  -o output.ll

# 完整 HIVM 编译（需要 bishengir-hivm-compile）
./build/bin/bishengir-compile \
  -enable-hivm-compile=true \
  -target=Ascend910B1 \
  input.mlir \
  -o output.ll
```

### bishengir-hivm-compile
**HIVM 后端编译器**（需要单独获取）

详见 `HIVM_COMPILE_GUIDE.md`

---

## 📖 方言层次

```
┌─────────────────────────────────────┐
│  Math / Arith (标准 MLIR 方言)      │  <-- 高层抽象
│  - math.exp, math.sqrt              │
│  - arith.addf, arith.mulf           │
└─────────────────────────────────────┘
              ↓ convert-*-to-hfusion
┌─────────────────────────────────────┐
│  HFusion (算子级抽象)                │  <-- 硬件无关
│  - hfusion.elemwise_unary            │
│  - hfusion.elemwise_binary           │
└─────────────────────────────────────┘
              ↓ convert-hfusion-to-hivm
┌─────────────────────────────────────┐
│  HIVM (硬件中间表示)                 │  <-- 硬件相关
│  - hivm.hir.vadd, hivm.hir.vmul     │
│  - hivm.hir.load, hivm.hir.store    │
│  - 地址空间: gm, ub, l0              │
└─────────────────────────────────────┘
              ↓ bishengir-hivm-compile
┌─────────────────────────────────────┐
│  HACC (硬件加速指令)                 │  <-- NPU 指令
└─────────────────────────────────────┘
```

---

## 🎓 示例代码解析

### Math 方言示例
```mlir
// math_examples.mlir
func.func @test_exp(%arg0: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = math.exp %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}
```

**转换到 HIVM**:
```bash
./build/bin/bishengir-opt examples/math_examples.mlir \
  --convert-math-to-hfusion \
  --convert-hfusion-to-hivm
```

**结果**:
```mlir
func.func @test_exp(%arg0: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %0 = tensor.empty() : tensor<6x6xf32>
  %1 = hivm.hir.vexp ins(%arg0 : tensor<6x6xf32>)
                      outs(%0 : tensor<6x6xf32>) -> tensor<6x6xf32>
  return %1 : tensor<6x6xf32>
}
```

### Arith 方言示例
```mlir
// simple_add.mlir
func.func @test_add(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = arith.addf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}
```

**转换**:
```bash
./build/bin/bishengir-opt examples/simple_add.mlir \
  --convert-arith-to-hfusion \
  --convert-hfusion-to-hivm
```

### HIVM 层示例
```mlir
// hivm_add.mlir - 完整的 HIVM 层代码
func.func @add(
  %arg0: memref<16xf32, #hivm.address_space<gm>>,
  %arg1: memref<16xf32, #hivm.address_space<gm>>,
  %arg2: memref<16xf32, #hivm.address_space<gm>>
) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // 分配 UB 缓冲区
  %buf0 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>

  // 从全局内存加载
  hivm.hir.load ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                outs(%buf0 : memref<16xf32, #hivm.address_space<ub>>)

  // 向量加法
  %buf1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>)
                outs(%buf1 : memref<16xf32, #hivm.address_space<ub>>)

  %result = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  hivm.hir.vadd ins(%buf0, %buf1 : memref<16xf32, #hivm.address_space<ub>>,
                                    memref<16xf32, #hivm.address_space<ub>>)
                outs(%result : memref<16xf32, #hivm.address_space<ub>>)

  // 存储回全局内存
  hivm.hir.store ins(%result : memref<16xf32, #hivm.address_space<ub>>)
                 outs(%arg2 : memref<16xf32, #hivm.address_space<gm>>)
  return
}
```

---

## 🐛 故障排查

### 问题：pass 名称错误
❌ 错误：`--hfusion-to-hivm`
✅ 正确：`--convert-hfusion-to-hivm`

所有转换 pass 都使用 `--convert-X-to-Y` 格式。

### 问题：bishengir-hivm-compile 未找到
```
[ERROR] Cannot find bishengir-hivm-compile under $PATH
```

**解决方案**：
```bash
# 方法 1: 添加到 PATH
export PATH=/path/to/bisheng/bin:$PATH

# 方法 2: 设置安装路径
export BISHENG_INSTALL_PATH=/path/to/bisheng/install

# 方法 3: 创建符号链接
sudo ln -s /path/to/bishengir-hivm-compile /usr/local/bin/
```

### 问题：操作不存在
❌ `math.addi` 不存在（Math 方言没有加法）
✅ 使用 `arith.addf` 进行浮点加法
✅ 使用 `arith.addi` 进行整数加法

---

## 📞 获取帮助

```bash
# 查看工具帮助
./build/bin/bishengir-opt --help
./build/bin/bishengir-compile --help

# 查看支持的方言
./build/bin/bishengir-opt --help | grep "Available Dialects"

# 查看所有 pass
./build/bin/bishengir-opt --help | grep "convert"

# 浏览测试用例
ls bishengir/test/Dialect/
ls bishengir/test/Conversion/
ls bishengir/test/bishengir-compile/
```

---

## ✨ 总结

**不需要硬件也能做的事**：
- ✅ 学习 MLIR 和方言系统
- ✅ 编写和验证 MLIR 代码
- ✅ 执行 IR 转换（Math → HFusion → HIVM）
- ✅ 理解编译器 pass 和优化
- ✅ 查看生成的 IR

**需要 bishengir-hivm-compile 的事**：
- ⚠️ 生成最终的 NPU 二进制代码
- ⚠️ 完整的 HIVM → HACC 编译
- ⚠️ 硬件特定的代码生成

**开始你的学习之旅**：
1. 📖 阅读 `TUTORIAL.md`
2. 🚀 运行 `./examples/demo.sh`
3. 🧪 尝试修改示例文件
4. 📚 查看 `QUICK_REFERENCE.md`
5. 🔧 探索更多测试用例

祝学习愉快！🎉
