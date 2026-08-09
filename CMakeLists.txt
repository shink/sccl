cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(sccl CXX)

# 采用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SCCL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 添加子目录
add_subdirectory(src)
add_subdirectory(test)
