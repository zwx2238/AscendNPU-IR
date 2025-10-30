#!/bin/bash
# BiShengIR 快速编译测试
# 支持使用本地编译版本或系统安装版本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================="
echo "BiShengIR HIVM 编译快速测试"
echo "========================================="
echo ""

# 检查 bishengir-opt
if [ -f "./build/bin/bishengir-opt" ]; then
    BISHENGIR_OPT="./build/bin/bishengir-opt"
    echo "✓ 使用本地 bishengir-opt: $BISHENGIR_OPT"
elif command -v bishengir-opt &> /dev/null; then
    BISHENGIR_OPT="bishengir-opt"
    echo "✓ 使用系统 bishengir-opt: $(which bishengir-opt)"
else
    echo "❌ 错误: bishengir-opt 未找到"
    echo "   请确保 bishengir-opt 在 PATH 中或已编译到 ./build/bin/"
    exit 1
fi

# 检查 bishengir-compile
if [ -f "./build/bin/bishengir-compile" ]; then
    BISHENGIR_COMPILE="./build/bin/bishengir-compile"
    echo "✓ 使用本地 bishengir-compile: $BISHENGIR_COMPILE"
elif command -v bishengir-compile &> /dev/null; then
    BISHENGIR_COMPILE="bishengir-compile"
    echo "✓ 使用系统 bishengir-compile: $(which bishengir-compile)"
else
    echo "❌ 错误: bishengir-compile 未找到"
    echo "   请确保 bishengir-compile 在 PATH 中或已编译到 ./build/bin/"
    exit 1
fi
echo ""

# 输入输出文件
INPUT="examples/hivm_add.mlir"
OUTPUT="/tmp/hivm_add_output.ll"

# 1. 验证输入
echo "1️⃣  验证 HIVM IR..."
$BISHENGIR_OPT "$INPUT" > /dev/null
echo "✓ 验证成功"
echo ""

# 2. 编译（不使用 HIVM 编译器）
echo "2️⃣  基础编译（无需 bishengir-hivm-compile）..."
$BISHENGIR_COMPILE \
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
    $BISHENGIR_COMPILE \
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
