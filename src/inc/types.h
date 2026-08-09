#pragma once

#include <cstdio>

#include "sccl.h"

// 分支预测提示宏：__builtin_expect 在 C++17 中仍然有效
// （[[likely]]/[[unlikely]] 属性是 C++20 特性，C++17 尚不可用）
#define SCCL_LIKELY(cond) __builtin_expect(!!(cond), 1)
#define SCCL_UNLIKELY(cond) __builtin_expect(!!(cond), 0)

// 条件检查宏：cond 为真时向 stderr 打印错误信息并返回 SCCL_ERROR_INTERNAL。
// 供内部函数快速上报错误，不调用 exit/abort。
// 注意：必须保持宏形式（因为需要从调用方 return），无法替换为 inline 函数。
#define SCCL_CHECK(cond, msg, ...)                                             \
    do                                                                         \
    {                                                                          \
        if (SCCL_UNLIKELY(cond))                                               \
        {                                                                      \
            std::fprintf(stderr, "[SCCL] ERROR: " msg "\n", ##__VA_ARGS__);    \
            return SCCL_ERROR_INTERNAL;                                        \
        }                                                                      \
    } while (0)
