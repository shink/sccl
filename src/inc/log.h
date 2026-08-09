#ifndef SCCL_INC_LOG_H
#define SCCL_INC_LOG_H

#include <stdio.h>

// 日志宏：输出到 stderr，格式为 [SCCL][rank X] LEVEL: msg
// rank 未设置时为 -1，表示尚未初始化

// 进程内全局 rank，由 scclLogSetRank 在初始化时设置
static int sccl_log_rank = -1;

// 设置当前进程的 rank，用于日志前缀
static inline void scclLogSetRank(int rank)
{
    sccl_log_rank = rank;
}

#define SCCL_LOG_INFO(fmt, ...)                                                \
    fprintf(stderr, "[SCCL][rank %d] INFO: " fmt "\n", sccl_log_rank,          \
            ##__VA_ARGS__)

#define SCCL_LOG_WARN(fmt, ...)                                                \
    fprintf(stderr, "[SCCL][rank %d] WARN: " fmt "\n", sccl_log_rank,          \
            ##__VA_ARGS__)

#define SCCL_LOG_ERROR(fmt, ...)                                               \
    fprintf(stderr, "[SCCL][rank %d] ERROR: " fmt "\n", sccl_log_rank,         \
            ##__VA_ARGS__)

#endif // SCCL_INC_LOG_H
