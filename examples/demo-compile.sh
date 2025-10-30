#!/bin/bash
# BiShengIR bishengir-compile 演示脚本
# 适用于只有 bishengir-compile 的系统（没有 bishengir-opt）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================="
echo "BiShengIR Compile 工具演示"
echo "========================================="
echo ""
echo "注意: 本脚本使用 bishengir-compile"
echo "      不需要 bishengir-opt"
echo ""

# 检查 bishengir-compile
echo "1️⃣  检查 bishengir-compile..."
if [ -f "./build/bin/bishengir-compile" ]; then
    BISHENGIR_COMPILE="./build/bin/bishengir-compile"
    echo "✓ 使用本地编译版本: $BISHENGIR_COMPILE"
elif command -v bishengir-compile &> /dev/null; then
    BISHENGIR_COMPILE="bishengir-compile"
    echo "✓ 使用系统安装版本: $(which bishengir-compile)"
else
    echo "❌ 错误: bishengir-compile 未找到"
    echo "   请确保 bishengir-compile 在 PATH 中"
    exit 1
fi
echo ""

# 显示版本
echo "2️⃣  工具版本："
$BISHENGIR_COMPILE --version | head -3
echo ""

# 示例1: 编译 Math 示例
echo "3️⃣  示例 1: 编译 Math 示例到 LLVM IR"
echo ""
INPUT_FILE="examples/math_examples.mlir"
OUTPUT_FILE="/tmp/bishengir_compile_demo_math.ll"

echo "输入: $INPUT_FILE"
echo "输出: $OUTPUT_FILE"
echo ""

echo "命令:"
echo "  $BISHENGIR_COMPILE -enable-lir-compile=false \\"
echo "    $INPUT_FILE \\"
echo "    -o $OUTPUT_FILE"
echo ""

if $BISHENGIR_COMPILE \
    -enable-lir-compile=false \
    "$INPUT_FILE" \
    -o "$OUTPUT_FILE" 2>&1; then
    echo "✓ 编译成功！"
    echo ""
    echo "生成的 LLVM IR (前20行):"
    echo "-------------------------------------------"
    head -20 "$OUTPUT_FILE"
    echo "..."
    echo "-------------------------------------------"
else
    echo "⚠️  编译失败"
    echo ""
    echo "可能原因:"
    echo "  - 需要 bishengir-hivm-compile 后端"
    echo "  - 输入文件格式不正确"
fi
echo ""

# 示例2: 使用 HIVM 编译（如果有 bishengir-hivm-compile）
echo "4️⃣  示例 2: 完整 HIVM 编译（需要 bishengir-hivm-compile）"
echo ""

if command -v bishengir-hivm-compile &> /dev/null; then
    echo "✓ 找到 bishengir-hivm-compile"
    echo ""

    INPUT_HIVM="examples/hivm_add.mlir"
    OUTPUT_HIVM="/tmp/bishengir_compile_demo_hivm.ll"

    echo "输入: $INPUT_HIVM"
    echo "输出: $OUTPUT_HIVM"
    echo ""

    echo "命令:"
    echo "  $BISHENGIR_COMPILE -enable-hivm-compile=true \\"
    echo "    -target=Ascend910B1 \\"
    echo "    $INPUT_HIVM \\"
    echo "    -o $OUTPUT_HIVM"
    echo ""

    if $BISHENGIR_COMPILE \
        -enable-hivm-compile=true \
        -target=Ascend910B1 \
        "$INPUT_HIVM" \
        -o "$OUTPUT_HIVM" 2>&1; then
        echo "✓ HIVM 编译成功！"
        echo ""
        echo "生成的 LLVM IR (前20行):"
        echo "-------------------------------------------"
        head -20 "$OUTPUT_HIVM"
        echo "..."
        echo "-------------------------------------------"
    else
        echo "❌ HIVM 编译失败"
    fi
else
    echo "⚠️  bishengir-hivm-compile 不可用，跳过此示例"
    echo ""
    echo "设置方法:"
    echo "  export PATH=/path/to/bisheng/bin:\$PATH"
fi
echo ""

# 总结
echo "========================================="
echo "✅ 演示完成！"
echo "========================================="
echo ""

echo "📚 bishengir-compile 常用选项:"
echo ""
echo "1. 基础编译（不需要 NPU 后端）:"
echo "   $BISHENGIR_COMPILE -enable-lir-compile=false input.mlir -o output.ll"
echo ""
echo "2. HIVM 完整编译（需要 bishengir-hivm-compile）:"
echo "   $BISHENGIR_COMPILE -enable-hivm-compile=true -target=Ascend910B1 input.mlir -o output.ll"
echo ""
echo "3. 带优化:"
echo "   $BISHENGIR_COMPILE -enable-auto-multi-buffer=true input.mlir -o output.ll"
echo ""
echo "4. 查看中间 IR:"
echo "   $BISHENGIR_COMPILE -bishengir-print-ir-after=all input.mlir -o output.ll 2>&1 | less"
echo ""
echo "5. 查看帮助:"
echo "   $BISHENGIR_COMPILE --help"
echo ""

echo "💡 提示:"
echo ""
echo "- bishengir-compile 是端到端编译器，适合生产使用"
echo "- 如需调试 IR 转换，建议编译 bishengir-opt (开发工具)"
echo "- 查看完整编译指南: cat examples/HIVM_COMPILE_GUIDE.md"
echo ""

echo "🎯 下一步:"
echo ""
echo "1. 查看更多编译选项:"
echo "   $BISHENGIR_COMPILE --help | less"
echo ""
echo "2. 尝试编译自己的 MLIR 文件"
echo ""
echo "3. 如需开发调试工具 (bishengir-opt):"
echo "   ./build-tools/auto-build.sh"
echo ""
