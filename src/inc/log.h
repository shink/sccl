#pragma once

#include <cstdio>

// 日志宏：输出到 stderr，格式为 [SCCL][rank X] LEVEL: msg
// rank 未设置时为 -1，表示尚未初始化

// C++17 inline 变量：所有 TU 共享同一实例。
// 修复原 C 代码的 ODR 问题：static 变量在每个 .c 文件中各有一份副本，
// 导致 scclLogSetRank 只影响 communicator.c 的副本，allgather.c/tcp.c
// 的副本始终为 -1。
inline int sccl_log_rank = -1;

// 设置当前进程的 rank，用于日志前缀
inline void scclLogSetRank(int rank)
{
    sccl_log_rank = rank;
}

#define SCCL_LOG_INFO(fmt, ...)                                                \
    std::fprintf(stderr, "[SCCL][rank %d] INFO: " fmt "\n", sccl_log_rank,     \
                 ##__VA_ARGS__)

#define SCCL_LOG_WARN(fmt, ...)                                                \
    std::fprintf(stderr, "[SCCL][rank %d] WARN: " fmt "\n", sccl_log_rank,     \
                 ##__VA_ARGS__)

#define SCCL_LOG_ERROR(fmt, ...)                                               \
    std::fprintf(stderr, "[SCCL][rank %d] ERROR: " fmt "\n", sccl_log_rank,    \
                 ##__VA_ARGS__)
