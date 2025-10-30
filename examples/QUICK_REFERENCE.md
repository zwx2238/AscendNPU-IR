# BiShengIR 快速参考卡

## 构建的工具

```bash
cd /home/zwx/triton-ascend-examples/AscendNPU-IR

# 主要工具
./build/bin/bishengir-opt       # MLIR 优化工具 (114MB)
./build/bin/bishengir-compile   # 端到端编译器 (105MB)
```

## 常用命令

### 1. 验证 MLIR 文件
```bash
./build/bin/bishengir-opt input.mlir
```

### 2. 查看帮助
```bash
# 查看所有 pass
./build/bin/bishengir-opt --help

# 查看版本
./build/bin/bishengir-opt --version

# 查看支持的方言
./build/bin/bishengir-opt --help | grep "Available Dialects"
```

### 3. 运行转换
```bash
# Math 转 HFusion
./build/bin/bishengir-opt input.mlir --convert-math-to-hfusion

# 带规范化
./build/bin/bishengir-opt input.mlir \
  --convert-math-to-hfusion \
  --canonicalize

# 查看每个 pass 后的结果
./build/bin/bishengir-opt input.mlir \
  --convert-math-to-hfusion \
  --mlir-print-ir-after-all
```

## 快速示例

### 例子1: Math 操作
```mlir
// math_exp.mlir
func.func @test(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %ret = math.exp %arg0 : tensor<1024xf32>
  return %ret : tensor<1024xf32>
}
```

运行：
```bash
./build/bin/bishengir-opt math_exp.mlir --convert-math-to-hfusion
```

### 例子2: HIVM 向量加法
见 `examples/simple_add.mlir` 和 `bishengir/test/Integration/HIVM/VecAdd/add.mlir`

## 方言层次

```
┌─────────────────────────────────────┐
│  Math / Linalg (标准 MLIR 方言)     │  <-- 高层抽象
└─────────────────────────────────────┘
              ↓ convert-*-to-hfusion
┌─────────────────────────────────────┐
│  HFusion (算子级抽象)                │  <-- 硬件无关
│  - elemwise_unary/binary             │
│  - matmul, conv, reduce              │
└─────────────────────────────────────┘
              ↓ hfusion-to-hivm
┌─────────────────────────────────────┐
│  HIVM (硬件中间表示)                 │  <-- 硬件相关
│  - vadd, vmul (向量运算)             │
│  - load/store (数据搬运)             │
│  - 地址空间: gm, ub, l0              │
└─────────────────────────────────────┘
              ↓ hivm-to-hacc
┌─────────────────────────────────────┐
│  HACC (硬件加速指令)                 │  <-- NPU 指令
└─────────────────────────────────────┘
```

## 关键 Pass

| Pass | 功能 |
|------|------|
| `--convert-math-to-hfusion` | Math 算子 → HFusion |
| `--convert-linalg-to-hfusion` | Linalg 算子 → HFusion |
| `--convert-arith-to-hfusion` | 算术运算 → HFusion |
| `--convert-tensor-to-hivm` | Tensor 操作 → HIVM |
| `--canonicalize` | 规范化优化 |
| `--hfusion-decompose` | 分解复合算子 |

## 学习资源

```bash
# 测试用例
ls bishengir/test/Dialect/        # 方言定义测试
ls bishengir/test/Conversion/     # 转换测试
ls bishengir/test/Integration/    # 端到端示例

# 文档
cat examples/TUTORIAL.md          # 入门教程
cat CLAUDE.md                      # 开发指南

# 示例代码
cat examples/math_examples.mlir   # Math 操作示例
cat examples/simple_add.mlir       # Linalg 操作示例
```

## 调试技巧

```bash
# 1. 逐步验证
./build/bin/bishengir-opt input.mlir  # 先验证语法

# 2. 查看转换结果
./build/bin/bishengir-opt input.mlir --convert-math-to-hfusion

# 3. 查看详细过程
./build/bin/bishengir-opt input.mlir \
  --convert-math-to-hfusion \
  --mlir-print-ir-after-all 2>&1 | less

# 4. 只看特定 pass 之后
./build/bin/bishengir-opt input.mlir \
  --convert-math-to-hfusion \
  --mlir-print-ir-after=convert-math-to-hfusion
```

## 常见错误

### 错误1: Pass 不识别
```
Unknown command line argument '--xxx'
```
**解决**: 检查 pass 名称，使用 `--help` 查看正确名称

### 错误2: 方言未注册
```
'xxx.yyy' op is not registered
```
**解决**: 确保使用了 `bishengir-opt` 而不是标准的 `mlir-opt`

### 错误3: 转换失败
```
failed to legalize operation
```
**解决**:
1. 检查输入 IR 是否合法
2. 确认该操作是否支持转换
3. 查看 `bishengir/test/Conversion/` 中的示例

## 下一步

1. ✅ 运行 `examples/math_examples.mlir` 熟悉基本转换
2. ✅ 阅读 `examples/TUTORIAL.md` 了解详细概念
3. ⏭️ 查看 `bishengir/test/` 下的更多示例
4. ⏭️ 尝试编写自己的 MLIR 代码并运行转换
