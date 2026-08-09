#ifndef SCCL_COMM_COMMUNICATOR_H
#define SCCL_COMM_COMMUNICATOR_H

#include <assert.h>
#include <stdint.h>

// root 节点信息：固定 64 字节，便于带外跨进程传递
// 字段排列保证自然对齐：magic(8) ip(46) port(2) rootRank(4@off56) reserved(4)
struct scclRootInfo
{
    uint64_t magic;      // 随机魔数，校验 rootInfo 有效性
    char ip[46];         // root 节点 IP
    uint16_t port;       // root 节点监听端口
    uint32_t rootRank;   // root 节点的 rank
    uint8_t reserved[4]; // 预留字段（凑齐 64 字节）
};
static_assert(sizeof(struct scclRootInfo) == 64,
              "scclRootInfo must be 64 bytes");

// 节点地址信息（用于 wire 传输）
struct scclPeerAddr
{
    uint32_t rank;
    char ip[46];
    uint16_t port;
};

// 通信域结构（内部实现，对公共 API 不透明）
struct scclComm
{
    uint32_t rank;     // 本节点 rank
    uint32_t nRanks;   // 总节点数
    uint32_t rootRank; // root 节点 rank（从 rootInfo 读取，非硬编码 0）
    int listenFd;      // 本节点监听 fd
    int *connFds;      // 连接 fd 数组（按 rank 索引，connFds[rootRank]==-1）
    int nConnFds;      // 连接 fd 数组大小（== nRanks）
    struct scclPeerAddr *peerAddrs; // 所有节点地址表
};

#endif // SCCL_COMM_COMMUNICATOR_H