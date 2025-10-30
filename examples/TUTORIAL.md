# BiShengIR 入门教程

## 1. 简介

BiShengIR 是基于 MLIR 的昇腾 NPU 编译器 IR，提供三层抽象：

```
Math/Linalg (高层)
    ↓  convert-*-to-hfusion
HFusion (算子层)
    ↓  convert-hfusion-to-hivm
HIVM (硬件抽象层)
    ↓  hivm-to-hacc
HACC (硬件指令层)
```

## 2. 第一个例子：Math 操作转换

### 2.1 创建输入文件 (math_exp.mlir)

```mlir
// 使用 math.exp 计算指数函数
func.func @test_exp(%arg0: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = math.exp %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}
```

### 2.2 运行转换

```bash
# 转换 Math 操作到 HFusion
./build/bin/bishengir-opt math_exp.mlir --convert-math-to-hfusion
```

**输出**：会看到 `math.exp` 被转换为 `hfusion.elemwise_unary`

### 2.3 查看所有方言

```bash
./build/bin/bishengir-opt --help | grep "Available Dialects"
```

可以看到支持的方言包括：`hfusion`, `hivm`, `hacc` 等

## 3. HIVM 层例子：向量加法

这是一个低层的 HIVM 例子，展示了硬件抽象层的操作：

```mlir
// add.mlir - HIVM 层的向量加法
module {
  func.func @add(
    %arg0: memref<16xi16, #hivm.address_space<gm>>,  // 全局内存输入1
    %arg1: memref<16xi16, #hivm.address_space<gm>>,  // 全局内存输入2
    %arg2: memref<16xi16, #hivm.address_space<gm>>   // 全局内存输出
  ) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {

    // 在 UB (Unified Buffer) 中分配缓冲区
    %alloc = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>

    // 从全局内存加载到 UB
    hivm.hir.load ins(%arg0 : memref<16xi16, #hivm.address_space<gm>>)
                  outs(%alloc : memref<16xi16, #hivm.address_space<ub>>)

    %alloc_0 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xi16, #hivm.address_space<gm>>)
                  outs(%alloc_0 : memref<16xi16, #hivm.address_space<ub>>)

    // 向量加法
    %alloc_1 = memref.alloc() : memref<16xi16, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%alloc, %alloc_0 : memref<16xi16, #hivm.address_space<ub>>,
                                          memref<16xi16, #hivm.address_space<ub>>)
                  outs(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)

    // 存储回全局内存
    hivm.hir.store ins(%alloc_1 : memref<16xi16, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<16xi16, #hivm.address_space<gm>>)
    return
  }
}
```

### 验证代码

```bash
./build/bin/bishengir-opt add.mlir
```

## 4. 常用 Pass 说明

### 4.1 转换 Pass

- `--convert-math-to-hfusion` - Math 算子转 HFusion
- `--convert-linalg-to-hfusion` - Linalg 算子转 HFusion
- `--convert-arith-to-hfusion` - 算术运算转 HFusion
- `--convert-tensor-to-hivm` - Tensor 操作转 HIVM
- `--hfusion-to-hivm` - HFusion 转 HIVM

### 4.2 优化 Pass

- `--linalg-fuse-elementwise-ops` - 融合逐元素操作
- `--hfusion-decompose` - 分解复合算子
- `--canonicalize` - 规范化

### 4.3 编译选项

- `--enable-hfusion-compile` - 启用 HFusion 编译
- `--enable-hivm-compile` - 启用 HIVM 编译

## 5. 完整工作流程示例

```bash
# 创建一个简单的 math 操作文件
cat > exp_example.mlir << 'EOF'
func.func @compute(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = math.exp %arg0 : tensor<1024xf32>
  %1 = math.sqrt %0 : tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
EOF

# 步骤1: 验证语法
./build/bin/bishengir-opt exp_example.mlir

# 步骤2: 转换到 HFusion
./build/bin/bishengir-opt exp_example.mlir --convert-math-to-hfusion

# 步骤3: 应用规范化
./build/bin/bishengir-opt exp_example.mlir \
  --convert-math-to-hfusion \
  --canonicalize
```

## 6. 查看帮助和调试

```bash
# 查看所有 pass
./build/bin/bishengir-opt --help

# 查看特定 pass 的选项
./build/bin/bishengir-opt --help | grep -A 3 "convert-math-to-hfusion"

# 启用调试输出
./build/bin/bishengir-opt input.mlir --convert-math-to-hfusion --mlir-print-ir-after-all

# 查看版本
./build/bin/bishengir-opt --version
```

## 7. 关键概念

### 7.1 地址空间 (Address Space)

在 HIVM 中，不同的内存有不同的地址空间标记：
- `gm` (Global Memory) - 全局内存/DDR
- `ub` (Unified Buffer) - 片上统一缓存
- `l0` - L0 缓存

### 7.2 操作分类

- **Load/Store**: 数据搬运 (`hivm.hir.load`, `hivm.hir.store`)
- **Vector Ops**: 向量运算 (`hivm.hir.vadd`, `hivm.hir.vmul` 等)
- **Element-wise**: 逐元素操作 (`hfusion.elemwise_unary`, `hfusion.elemwise_binary`)

## 8. 下一步学习

1. 查看 `bishengir/test/Dialect/` 下的测试用例
2. 查看 `bishengir/test/Conversion/` 下的转换示例
3. 查看 `bishengir/test/Integration/` 下的端到端示例
4. 阅读 `docs/` 目录下的文档

## 9. 常见问题

**Q: 如何知道某个操作支持哪些转换？**
A: 查看 `bishengir/test/Conversion/` 目录下对应的测试文件

**Q: 如何调试 pass 失败？**
A: 使用 `--mlir-print-ir-after-all` 查看每个 pass 后的 IR 变化

**Q: 编译需要硬件吗？**
A: MLIR 层的转换不需要硬件，但最终的 HACC 层代码执行需要昇腾 NPU
