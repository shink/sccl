#define _DEFAULT_SOURCE

#include "communicator.h"
#include "log.h"
#include "sccl.h"
#include "tcp.h"
#include "types.h"

#include <string.h>

// 获取数据类型大小（字节）
static size_t dtype_size(scclDataType_t dtype)
{
    switch (dtype)
    {
    case SCCL_INT32:
    case SCCL_UINT32:
        return 4;
    default:
        return 0;
    }
}

// AllGather：每个 rank 贡献 sendbuff 的 count 个元素，
// 所有 rank 在 recvbuff 中收集全部 nRanks*count 个元素（按 rank 顺序排列）
// 算法：非 root 发送数据给 root，root 汇总后广播给所有 rank
scclResult_t ScclAllGather(const void *sendbuff, void *recvbuff, size_t count,
                           scclDataType_t datatype, scclComm_t comm)
{
    SCCL_CHECK(!sendbuff || !recvbuff || !comm, "invalid argument");

    struct scclComm *c = (struct scclComm *)comm;
    size_t elemSize = dtype_size(datatype);
    SCCL_CHECK(elemSize == 0, "invalid datatype");
    SCCL_CHECK(count == 0, "count is 0");

    size_t chunkSize = count * elemSize;
    size_t totalSize = (size_t)c->nRanks * chunkSize;
    scclResult_t ret;

    if (c->rank == c->rootRank)
    {
        // root 路径：先拷贝自己的数据，再接收每个非 root rank
        // 的数据，最后广播完整结果
        memcpy((char *)recvbuff + (size_t)c->rootRank * chunkSize, sendbuff,
               chunkSize);

        for (uint32_t r = 0; r < c->nRanks; r++)
        {
            if (r == c->rootRank)
                continue;
            char *dst = (char *)recvbuff + (size_t)r * chunkSize;
            ret = sccl_tcp_recvall(c->connFds[r], dst, chunkSize);
            if (ret != SCCL_SUCCESS)
            {
                SCCL_LOG_ERROR("recv from rank %u failed", r);
                return ret;
            }
        }

        // 广播完整结果给所有非 root rank
        for (uint32_t r = 0; r < c->nRanks; r++)
        {
            if (r == c->rootRank)
                continue;
            ret = sccl_tcp_sendall(c->connFds[r], recvbuff, totalSize);
            if (ret != SCCL_SUCCESS)
            {
                SCCL_LOG_ERROR("send to rank %u failed", r);
                return ret;
            }
        }
    }
    else
    {
        // 非 root 路径：发送自己的数据给 root，然后接收完整结果
        ret = sccl_tcp_sendall(c->connFds[c->rootRank], sendbuff, chunkSize);
        if (ret != SCCL_SUCCESS)
        {
            SCCL_LOG_ERROR("send to root failed");
            return ret;
        }

        ret = sccl_tcp_recvall(c->connFds[c->rootRank], recvbuff, totalSize);
        if (ret != SCCL_SUCCESS)
        {
            SCCL_LOG_ERROR("recv from root failed");
            return ret;
        }
    }

    SCCL_LOG_INFO("allgather done: rank=%u count=%zu", c->rank, count);
    return SCCL_SUCCESS;
}
