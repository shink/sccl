cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(sccl C)

# 采用 C11 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(SCCL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 添加子目录
add_subdirectory(src)
add_subdirectory(test)
