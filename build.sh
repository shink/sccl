#!/bin/bash

set -e

cd "$(dirname "$0")"

# 清理旧构建目录
rm -rf build
mkdir build
cd build

# CMake 配置 + 编译
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 验证产物
echo ""
echo "Build OK:"
echo "  Library: $(pwd)/src/libsccl.a"
echo "  Test:    $(pwd)/test/test_init"
