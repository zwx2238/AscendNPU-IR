#!/bin/bash
# BiShengIR 演示脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================="
echo "BiShengIR 入门演示"
echo "========================================="
echo ""

# 检查工具
echo "✓ 检查 bishengir-opt 工具..."
if [ ! -f "./build/bin/bishengir-opt" ]; then
    echo "❌ 错误: bishengir-opt 未找到"
    echo "   请先运行构建：./build-tools/build.sh -o ./build --build-type Release"
    exit 1
fi

# 显示版本
echo ""
echo "1️⃣  工具版本："
./build/bin/bishengir-opt --version
echo ""

# 示例1: 验证 MLIR 文件
echo "2️⃣  验证 MLIR 文件语法："
echo "   命令: bishengir-opt examples/math_examples.mlir"
echo ""
./build/bin/bishengir-opt examples/math_examples.mlir > /tmp/bishengir_demo_1.txt
echo "   ✓ 语法正确！"
echo ""

# 示例2: Math 到 HFusion 转换
echo "3️⃣  演示 Math → HFusion 转换："
echo "   命令: bishengir-opt examples/math_examples.mlir --convert-math-to-hfusion"
echo ""
echo "   转换结果（部分）："
./build/bin/bishengir-opt examples/math_examples.mlir --convert-math-to-hfusion | head -15
echo "   ..."
echo ""
echo "   注意："
echo "   - math.exp → linalg.elemwise_unary {fun = #linalg.unary_fn<exp>}"
echo "   - math.sqrt → hfusion.elemwise_unary {fun = #hfusion.unary_fn<sqrt>}"
echo ""

# 示例3: 带规范化
echo "4️⃣  演示带规范化的转换："
echo "   命令: bishengir-opt --convert-math-to-hfusion --canonicalize"
echo ""
./build/bin/bishengir-opt examples/math_examples.mlir \
    --convert-math-to-hfusion \
    --canonicalize > /tmp/bishengir_demo_2.txt
echo "   ✓ 转换并优化完成"
echo ""

# 查看支持的方言
echo "5️⃣  支持的方言（部分）："
./build/bin/bishengir-opt --help | grep "Available Dialects" | \
    sed 's/Available Dialects: //' | \
    tr ',' '\n' | head -20 | sed 's/^/   - /'
echo "   ... 等更多方言"
echo ""

# 总结
echo "========================================="
echo "✅ 演示完成！"
echo "========================================="
echo ""
echo "📚 后续学习建议："
echo ""
echo "1. 阅读详细教程："
echo "   cat examples/TUTORIAL.md"
echo ""
echo "2. 查看快速参考："
echo "   cat examples/QUICK_REFERENCE.md"
echo ""
echo "3. 自己尝试转换："
echo "   ./build/bin/bishengir-opt examples/math_examples.mlir --convert-math-to-hfusion"
echo ""
echo "4. 探索测试用例："
echo "   ls bishengir/test/Conversion/"
echo "   ls bishengir/test/Dialect/"
echo ""
echo "5. 查看所有可用的 pass："
echo "   ./build/bin/bishengir-opt --help | less"
echo ""
echo "🎯 关键命令提醒："
echo "   bishengir-opt <input.mlir>                     # 验证语法"
echo "   bishengir-opt <input.mlir> --convert-*-to-*    # 运行转换"
echo "   bishengir-opt --help                           # 查看帮助"
echo ""
