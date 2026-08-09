#pragma once

#include <stddef.h>
#include <stdint.h>

#include "sccl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // root 节点调用：创建 TCP 监听 socket，填充 rootInfo（含 rootRank）
    // rootRank 为本进程（root）的 rank，将写入 rootInfo 供所有节点判断 root
    // 身份
    scclResult_t HcclGetRootInfo(scclRootInfo_t *rootInfo, uint32_t rootRank);

    // 各节点调用：根据 rootInfo 建立通信域
    scclResult_t ScclCommInitRootInfo(uint32_t nRanks,
                                      const scclRootInfo_t *rootInfo,
                                      uint32_t rank, scclComm_t *comm);

    // 销毁通信域
    scclResult_t ScclCommDestroy(scclComm_t comm);

    // 集合通信算子 - AllGather
    // 每个 rank 贡献 sendbuff 的 count 个元素，
    // 所有 rank 在 recvbuff 中收集全部 nRanks*count 个元素（按 rank 顺序排列）
    scclResult_t ScclAllGather(const void *sendbuff, void *recvbuff,
                               size_t count, scclDataType_t datatype,
                               scclComm_t comm);

#ifdef __cplusplus
}
#endif
