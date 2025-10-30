#!/bin/bash
# BiShengIR HIVM 编译测试脚本
# 用于测试 HIVM → HACC 的完整编译流程

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "BiShengIR HIVM 编译测试"
echo "========================================="
echo ""

# ============================================
# 1. 检查工具
# ============================================
echo "1️⃣  检查必要工具..."
echo ""

# 检查 bishengir-opt - 优先本地，否则系统
if [ -f "$PROJECT_ROOT/build/bin/bishengir-opt" ]; then
    BISHENGIR_OPT="$PROJECT_ROOT/build/bin/bishengir-opt"
    echo "✓ bishengir-opt (本地): $BISHENGIR_OPT"
elif command -v bishengir-opt &> /dev/null; then
    BISHENGIR_OPT="bishengir-opt"
    echo "✓ bishengir-opt (系统): $(which bishengir-opt)"
else
    echo "❌ 错误: bishengir-opt 未找到"
    echo "   请编译或确保 bishengir-opt 在 PATH 中"
    exit 1
fi

# 检查 bishengir-compile - 优先本地，否则系统
if [ -f "$PROJECT_ROOT/build/bin/bishengir-compile" ]; then
    BISHENGIR_COMPILE="$PROJECT_ROOT/build/bin/bishengir-compile"
    echo "✓ bishengir-compile (本地): $BISHENGIR_COMPILE"
elif command -v bishengir-compile &> /dev/null; then
    BISHENGIR_COMPILE="bishengir-compile"
    echo "✓ bishengir-compile (系统): $(which bishengir-compile)"
else
    echo "❌ 错误: bishengir-compile 未找到"
    echo "   请编译或确保 bishengir-compile 在 PATH 中"
    exit 1
fi

# 检查 bishengir-hivm-compile (需要手动提供)
if ! command -v bishengir-hivm-compile &> /dev/null; then
    echo "⚠️  警告: bishengir-hivm-compile 不在 PATH 中"
    echo "   请设置环境变量："
    echo "   export PATH=/path/to/bishengir-hivm-compile:\$PATH"
    echo ""
    echo "   或者设置 BISHENG_INSTALL_PATH："
    echo "   export BISHENG_INSTALL_PATH=/path/to/bisheng/install"
    echo ""
else
    echo "✓ bishengir-hivm-compile: $(which bishengir-hivm-compile)"
fi
echo ""

# ============================================
# 2. 准备输入文件
# ============================================
echo "2️⃣  准备测试文件..."
echo ""

INPUT_FILE="$SCRIPT_DIR/hivm_add.mlir"
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi
echo "✓ 输入文件: $INPUT_FILE"
echo ""

# ============================================
# 3. 显示输入文件内容
# ============================================
echo "3️⃣  输入文件内容："
echo "-------------------------------------------"
cat "$INPUT_FILE"
echo "-------------------------------------------"
echo ""

# ============================================
# 4. 步骤 1: 使用 bishengir-opt 验证和转换
# ============================================
echo "4️⃣  步骤 1: 使用 bishengir-opt 验证 HIVM IR..."
echo ""

TMP_DIR="/tmp/bishengir_test_$$"
mkdir -p "$TMP_DIR"

echo "命令: bishengir-opt $INPUT_FILE"
$BISHENGIR_OPT "$INPUT_FILE" > "$TMP_DIR/validated.mlir"
echo "✓ HIVM IR 验证成功"
echo ""

# ============================================
# 5. 步骤 2: 使用 bishengir-compile 编译
# ============================================
echo "5️⃣  步骤 2: 使用 bishengir-compile 编译到 LLVM IR..."
echo ""

OUTPUT_LL="$TMP_DIR/output.ll"

echo "方法 A: 使用 bishengir-compile（不启用 LIR 编译）"
echo "命令: bishengir-compile -enable-lir-compile=false $INPUT_FILE -o $OUTPUT_LL"
echo ""

if $BISHENGIR_COMPILE \
    -enable-lir-compile=false \
    "$INPUT_FILE" \
    -o "$OUTPUT_LL" 2>&1; then
    echo "✓ 编译成功"
    echo "  输出文件: $OUTPUT_LL"

    if [ -f "$OUTPUT_LL" ]; then
        echo ""
        echo "生成的 LLVM IR (前50行):"
        echo "-------------------------------------------"
        head -50 "$OUTPUT_LL"
        echo "..."
        echo "-------------------------------------------"
    fi
else
    echo "⚠️  编译失败（可能需要 bishengir-hivm-compile）"
fi
echo ""

# ============================================
# 6. 步骤 3: 使用 bishengir-compile 启用 HIVM 编译
# ============================================
echo "6️⃣  步骤 3: 使用 bishengir-compile 启用 HIVM 编译..."
echo ""

OUTPUT_HIVM_LL="$TMP_DIR/output_hivm.ll"

echo "方法 B: 启用 HIVM 编译（需要 bishengir-hivm-compile）"
echo "命令: bishengir-compile -enable-hivm-compile=true -target=Ascend910B1 $INPUT_FILE -o $OUTPUT_HIVM_LL"
echo ""

if command -v bishengir-hivm-compile &> /dev/null; then
    if $BISHENGIR_COMPILE \
        -enable-hivm-compile=true \
        -target=Ascend910B1 \
        "$INPUT_FILE" \
        -o "$OUTPUT_HIVM_LL" 2>&1; then
        echo "✓ HIVM 编译成功"
        echo "  输出文件: $OUTPUT_HIVM_LL"

        if [ -f "$OUTPUT_HIVM_LL" ]; then
            echo ""
            echo "生成的 LLVM IR (前50行):"
            echo "-------------------------------------------"
            head -50 "$OUTPUT_HIVM_LL"
            echo "..."
            echo "-------------------------------------------"
        fi
    else
        echo "❌ HIVM 编译失败"
    fi
else
    echo "⚠️  跳过（bishengir-hivm-compile 不可用）"
fi
echo ""

# ============================================
# 7. 步骤 4: 更多编译选项示例
# ============================================
echo "7️⃣  其他编译选项示例："
echo ""

echo "# 基础编译（无优化）"
echo "bishengir-compile -enable-lir-compile=false input.mlir -o output.ll"
echo ""

echo "# 启用 HIVM 编译 + 指定目标设备"
echo "bishengir-compile -enable-hivm-compile=true -target=Ascend910B1 input.mlir -o output.ll"
echo ""

echo "# 启用多缓冲优化"
echo "bishengir-compile -enable-auto-multi-buffer=true input.mlir -o output.ll"
echo ""

echo "# 启用调试信息"
echo "bishengir-compile -enable-debug-info=true input.mlir -o output.ll"
echo ""

echo "# 查看 IR 转换过程"
echo "bishengir-compile -bishengir-print-ir-after=all input.mlir -o output.ll"
echo ""

echo "# 启用静态裸指针优化"
echo "bishengir-compile -enable-static-bare-ptr=true input.mlir -o output.ll"
echo ""

# ============================================
# 8. 使用 bishengir-hivm-compile 的说明
# ============================================
echo "8️⃣  如何设置 bishengir-hivm-compile："
echo ""

cat << 'EOF'
bishengir-hivm-compile 是 HIVM → HACC 的后端编译器。

设置方法：

方法 1: 添加到 PATH
  export PATH=/path/to/bisheng/bin:$PATH
  # 例如：
  export PATH=/usr/local/bisheng/bin:$PATH

方法 2: 设置 BISHENG_INSTALL_PATH
  export BISHENG_INSTALL_PATH=/path/to/bisheng/install
  # bishengir-compile 会自动查找 $BISHENG_INSTALL_PATH/bin/bishengir-hivm-compile

方法 3: 创建符号链接
  ln -s /path/to/bishengir-hivm-compile /usr/local/bin/

验证安装:
  which bishengir-hivm-compile
  bishengir-hivm-compile --help
EOF
echo ""

# ============================================
# 9. 完整编译流程总结
# ============================================
echo "========================================="
echo "✅ 测试完成！"
echo "========================================="
echo ""

echo "📚 完整编译流程："
echo ""
echo "1. Math/Arith → HFusion:"
echo "   bishengir-opt input.mlir --convert-arith-to-hfusion"
echo ""
echo "2. HFusion → HIVM:"
echo "   bishengir-opt input.mlir --convert-arith-to-hfusion --convert-hfusion-to-hivm"
echo ""
echo "3. HIVM → LLVM IR (需要 bishengir-hivm-compile):"
echo "   bishengir-compile -enable-hivm-compile=true -target=Ascend910B1 input.mlir -o output.ll"
echo ""
echo "4. LLVM IR → 目标代码:"
echo "   llc output.ll -o output.o"
echo ""

echo "🔍 调试技巧："
echo ""
echo "# 查看每个 pass 后的 IR"
echo "bishengir-compile -bishengir-print-ir-after=all input.mlir 2>&1 | less"
echo ""
echo "# 保存中间结果"
echo "bishengir-opt input.mlir --convert-arith-to-hfusion > hfusion.mlir"
echo "bishengir-opt hfusion.mlir --convert-hfusion-to-hivm > hivm.mlir"
echo "bishengir-compile -enable-hivm-compile=true hivm.mlir -o output.ll"
echo ""

echo "📂 生成的文件位置: $TMP_DIR"
echo "   - validated.mlir: 验证后的 HIVM IR"
echo "   - output.ll: 编译后的 LLVM IR"
echo ""

echo "💡 提示："
echo "   如果需要在另一台机器测试，请："
echo "   1. 复制 bishengir-hivm-compile 到本机"
echo "   2. 设置 PATH 或 BISHENG_INSTALL_PATH"
echo "   3. 重新运行此脚本"
echo ""
