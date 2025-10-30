#!/bin/bash
# BiShengIR 快速编译测试
# 假设你已经将 bishengir-hivm-compile 添加到 PATH 或设置了 BISHENG_INSTALL_PATH

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================="
echo "BiShengIR HIVM 编译快速测试"
echo "========================================="
echo ""

# 输入输出文件
INPUT="examples/hivm_add.mlir"
OUTPUT="/tmp/hivm_add_output.ll"

# 1. 验证输入
echo "1️⃣  验证 HIVM IR..."
./build/bin/bishengir-opt "$INPUT" > /dev/null
echo "✓ 验证成功"
echo ""

# 2. 编译（不使用 HIVM 编译器）
echo "2️⃣  基础编译（无需 bishengir-hivm-compile）..."
./build/bin/bishengir-compile \
  -enable-lir-compile=false \
  "$INPUT" \
  -o "$OUTPUT" && echo "✓ 编译成功: $OUTPUT" || echo "❌ 编译失败"
echo ""

# 3. 完整编译（需要 bishengir-hivm-compile）
echo "3️⃣  HIVM 编译（需要 bishengir-hivm-compile）..."
if command -v bishengir-hivm-compile &> /dev/null; then
    echo "✓ 找到 bishengir-hivm-compile: $(which bishengir-hivm-compile)"
    echo ""

    OUTPUT_HIVM="/tmp/hivm_add_full.ll"

    echo "正在编译..."
    ./build/bin/bishengir-compile \
      -enable-hivm-compile=true \
      -target=Ascend910B1 \
      "$INPUT" \
      -o "$OUTPUT_HIVM" && echo "✓ HIVM 编译成功: $OUTPUT_HIVM" || echo "❌ HIVM 编译失败"
else
    echo "⚠️  未找到 bishengir-hivm-compile"
    echo ""
    echo "请设置环境变量："
    echo "  export PATH=/path/to/bisheng/bin:\$PATH"
    echo "或"
    echo "  export BISHENG_INSTALL_PATH=/path/to/bisheng/install"
fi
echo ""

echo "========================================="
echo "✅ 测试完成"
echo "========================================="
