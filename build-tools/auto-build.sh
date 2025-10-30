#!/bin/bash
# BiShengIR 编译脚本
# 适用于 x86_64 和 aarch64 架构
# 基于成功的编译经验整理

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}BiShengIR 自动编译脚本${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 检测架构
ARCH=$(uname -m)
echo -e "${GREEN}✓${NC} 检测到架构: ${YELLOW}$ARCH${NC}"

# 检测操作系统
OS=$(uname -s)
echo -e "${GREEN}✓${NC} 操作系统: ${YELLOW}$OS${NC}"
echo ""

# 项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}✓${NC} 项目目录: ${YELLOW}$PROJECT_ROOT${NC}"
echo ""

# ============================================
# 1. 检查必要工具
# ============================================
echo -e "${BLUE}1️⃣  检查必要工具...${NC}"
echo ""

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1: $(which $1)"
        if [ "$2" != "" ]; then
            echo -e "   版本: $($1 $2 2>&1 | head -1)"
        fi
        return 0
    else
        echo -e "${RED}✗${NC} $1: 未找到"
        return 1
    fi
}

MISSING_TOOLS=()

# 检查必要工具
if ! check_command "git" "--version"; then
    MISSING_TOOLS+=("git")
fi

if ! check_command "cmake" "--version"; then
    MISSING_TOOLS+=("cmake")
fi

if ! check_command "ninja" "--version"; then
    MISSING_TOOLS+=("ninja")
fi

if ! check_command "clang" "--version"; then
    MISSING_TOOLS+=("clang")
fi

if ! check_command "clang++" "--version"; then
    MISSING_TOOLS+=("clang++")
fi

echo ""

# 如果有缺失的工具，提供安装建议
if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "${RED}❌ 缺少以下工具:${NC}"
    for tool in "${MISSING_TOOLS[@]}"; do
        echo "   - $tool"
    done
    echo ""
    echo -e "${YELLOW}📦 安装建议:${NC}"
    echo ""

    if [ "$OS" = "Linux" ]; then
        echo "Ubuntu/Debian:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install -y git clang ninja-build"
        echo "  pip install cmake --upgrade  # cmake 需要 >= 3.28"
        echo "  pip install ninja --upgrade  # ninja 需要 >= 1.12"
        echo ""
        echo "CentOS/RHEL:"
        echo "  sudo yum install -y git clang ninja-build"
        echo "  pip install cmake --upgrade"
        echo "  pip install ninja --upgrade"
    fi
    echo ""
    exit 1
fi

# 检查版本要求
echo -e "${BLUE}检查版本要求...${NC}"

# 检查 CMake 版本 (需要 >= 3.28)
CMAKE_VERSION=$(cmake --version | head -1 | grep -oP '\d+\.\d+\.\d+')
CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)

if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 28 ]); then
    echo -e "${YELLOW}⚠️  CMake 版本过低: $CMAKE_VERSION (需要 >= 3.28)${NC}"
    echo "   请升级: pip install cmake --upgrade"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓${NC} CMake 版本: $CMAKE_VERSION"
fi

# 检查 Ninja 版本 (需要 >= 1.12)
NINJA_VERSION=$(ninja --version)
NINJA_MAJOR=$(echo $NINJA_VERSION | cut -d. -f1)
NINJA_MINOR=$(echo $NINJA_VERSION | cut -d. -f2)

if [ "$NINJA_MAJOR" -lt 1 ] || ([ "$NINJA_MAJOR" -eq 1 ] && [ "$NINJA_MINOR" -lt 12 ]); then
    echo -e "${YELLOW}⚠️  Ninja 版本过低: $NINJA_VERSION (需要 >= 1.12)${NC}"
    echo "   请升级: pip install ninja --upgrade"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓${NC} Ninja 版本: $NINJA_VERSION"
fi

echo ""

# ============================================
# 2. 准备 LLVM 源码
# ============================================
echo -e "${BLUE}2️⃣  准备 LLVM 源码...${NC}"
echo ""

LLVM_DIR="$PROJECT_ROOT/third-party/llvm-project"

if [ -d "$LLVM_DIR" ] && [ -f "$LLVM_DIR/llvm/CMakeLists.txt" ]; then
    echo -e "${GREEN}✓${NC} LLVM 源码已存在: $LLVM_DIR"
else
    echo -e "${YELLOW}⚠️  LLVM 源码不存在，开始下载...${NC}"
    echo ""

    # 提供多种下载方式
    echo "选择下载方式:"
    echo "  1. 使用 git submodule (官方，较慢)"
    echo "  2. 从本地复制 (如果你有 llvm-project 仓库)"
    echo ""
    read -p "请选择 [1/2]: " choice

    case $choice in
        1)
            echo "使用 git submodule 下载..."
            git submodule update --init --recursive third-party/llvm-project
            ;;
        2)
            read -p "请输入 llvm-project 源码路径: " LLVM_SRC
            if [ -d "$LLVM_SRC" ]; then
                echo "从 $LLVM_SRC 复制..."
                cp -r "$LLVM_SRC" "$LLVM_DIR"
            else
                echo -e "${RED}❌ 路径不存在: $LLVM_SRC${NC}"
                exit 1
            fi
            ;;
        *)
            echo -e "${RED}❌ 无效选择${NC}"
            exit 1
            ;;
    esac
fi

# 切换到正确的 LLVM 提交
echo ""
echo "切换 LLVM 到正确的提交..."
cd "$LLVM_DIR"
LLVM_COMMIT="cd708029e0b2869e80abe31ddb175f7c35361f90"
git checkout $LLVM_COMMIT
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓${NC} LLVM 提交: $LLVM_COMMIT"
echo ""

# ============================================
# 3. 应用补丁
# ============================================
echo -e "${BLUE}3️⃣  应用 LLVM 补丁...${NC}"
echo ""

PATCH_DIR="$PROJECT_ROOT/build-tools/patches/llvm-project"
if [ -d "$PATCH_DIR" ]; then
    echo "找到补丁目录: $PATCH_DIR"
    PATCH_COUNT=$(ls -1 "$PATCH_DIR"/*.patch 2>/dev/null | wc -l)
    echo "补丁数量: $PATCH_COUNT"
    echo ""

    if [ $PATCH_COUNT -gt 0 ]; then
        echo "应用补丁..."
        cd "$LLVM_DIR"

        for patch in "$PATCH_DIR"/*.patch; do
            echo "应用: $(basename $patch)"
            git apply "$patch" || {
                echo -e "${YELLOW}⚠️  补丁可能已应用: $(basename $patch)${NC}"
            }
        done

        cd "$PROJECT_ROOT"
        echo -e "${GREEN}✓${NC} 补丁应用完成"
    fi
else
    echo -e "${YELLOW}⚠️  未找到补丁目录${NC}"
fi
echo ""

# ============================================
# 4. 配置构建
# ============================================
echo -e "${BLUE}4️⃣  配置构建...${NC}"
echo ""

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

echo "构建配置:"
echo "  架构: $ARCH"
echo "  构建目录: $BUILD_DIR"
echo "  构建类型: Release"
echo "  编译器: clang/clang++"
echo ""

cd "$BUILD_DIR"

echo "运行 CMake..."
cmake -G Ninja ../third-party/llvm-project/llvm \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$PROJECT_ROOT" \
    -DBSPUB_DAVINCI_BISHENGIR=ON \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=OFF

echo -e "${GREEN}✓${NC} CMake 配置完成"
echo ""

# ============================================
# 5. 编译
# ============================================
echo -e "${BLUE}5️⃣  开始编译...${NC}"
echo ""

# 检测 CPU 核心数
if [ "$OS" = "Linux" ]; then
    NPROC=$(nproc)
elif [ "$OS" = "Darwin" ]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

echo "使用 $NPROC 个并行任务"
echo ""

# 编译主要目标
echo "编译 bishengir-opt 和 bishengir-compile..."
ninja -j $NPROC bishengir-opt bishengir-compile

echo ""
echo -e "${GREEN}✓${NC} 编译完成"
echo ""

# ============================================
# 6. 验证编译结果
# ============================================
echo -e "${BLUE}6️⃣  验证编译结果...${NC}"
echo ""

if [ -f "$BUILD_DIR/bin/bishengir-opt" ]; then
    SIZE=$(du -h "$BUILD_DIR/bin/bishengir-opt" | cut -f1)
    echo -e "${GREEN}✓${NC} bishengir-opt: $BUILD_DIR/bin/bishengir-opt ($SIZE)"
    echo "   版本信息:"
    "$BUILD_DIR/bin/bishengir-opt" --version | head -3 | sed 's/^/     /'
else
    echo -e "${RED}✗${NC} bishengir-opt 未找到"
fi

echo ""

if [ -f "$BUILD_DIR/bin/bishengir-compile" ]; then
    SIZE=$(du -h "$BUILD_DIR/bin/bishengir-compile" | cut -f1)
    echo -e "${GREEN}✓${NC} bishengir-compile: $BUILD_DIR/bin/bishengir-compile ($SIZE)"
else
    echo -e "${RED}✗${NC} bishengir-compile 未找到"
fi

echo ""

# ============================================
# 7. 运行测试
# ============================================
echo -e "${BLUE}7️⃣  运行快速测试...${NC}"
echo ""

TEST_FILE="$PROJECT_ROOT/examples/math_examples.mlir"
if [ -f "$TEST_FILE" ]; then
    echo "测试文件: $TEST_FILE"
    echo ""
    echo "测试 1: 验证语法"
    if "$BUILD_DIR/bin/bishengir-opt" "$TEST_FILE" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 语法验证通过"
    else
        echo -e "${RED}✗${NC} 语法验证失败"
    fi

    echo ""
    echo "测试 2: Math → HFusion 转换"
    if "$BUILD_DIR/bin/bishengir-opt" "$TEST_FILE" --convert-math-to-hfusion > /tmp/test_output.mlir 2>&1; then
        echo -e "${GREEN}✓${NC} 转换成功"
        echo "   输出: /tmp/test_output.mlir"
    else
        echo -e "${RED}✗${NC} 转换失败"
    fi
fi

echo ""

# ============================================
# 8. 总结
# ============================================
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✅ 编译完成！${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

echo -e "${BLUE}📦 构建产物:${NC}"
echo "  bishengir-opt:     $BUILD_DIR/bin/bishengir-opt"
echo "  bishengir-compile: $BUILD_DIR/bin/bishengir-compile"
echo ""

echo -e "${BLUE}🚀 后续步骤:${NC}"
echo ""
echo "1. 运行演示脚本:"
echo "   ./examples/demo.sh"
echo ""
echo "2. 测试 IR 转换:"
echo "   $BUILD_DIR/bin/bishengir-opt examples/math_examples.mlir --convert-math-to-hfusion"
echo ""
echo "3. 查看教程:"
echo "   cat examples/TUTORIAL.md"
echo ""
echo "4. 添加到 PATH (可选):"
echo "   export PATH=$BUILD_DIR/bin:\$PATH"
echo ""

echo -e "${BLUE}💡 提示:${NC}"
echo "  - 如果需要 bishengir-hivm-compile，请从有 NPU 的机器获取"
echo "  - 本地编译的工具可以在没有 NPU 的机器上运行 MLIR 转换"
echo "  - 查看 examples/README.md 了解更多信息"
echo ""
