#pragma once

#include <cassert>
#include <cstdint>
#include <unistd.h>
#include <vector>

// root 节点信息：固定 64 字节，便于带外跨进程传递
// 字段排列保证自然对齐：magic(8) ip(46) port(2) rootRank(4@off56) reserved(4)
// 注意：必须保持 POD/standard-layout，因为此结构通过 socket 和文件跨进程传递。
struct scclRootInfo
{
    uint64_t magic;      // 随机魔数，校验 rootInfo 有效性
    char ip[46];         // root 节点 IP
    uint16_t port;       // root 节点监听端口
    uint32_t rootRank;   // root 节点的 rank
    uint8_t reserved[4]; // 预留字段（凑齐 64 字节）
};
static_assert(sizeof(scclRootInfo) == 64, "scclRootInfo must be 64 bytes");

// 节点地址信息（用于 wire 传输，必须保持 POD）
struct scclPeerAddr
{
    uint32_t rank;
    char ip[46];
    uint16_t port;
};

// 通信域结构（内部实现，对公共 API 不透明）
// C++17 改造：使用 std::vector 管理动态数组，析构函数自动关闭 fd（RAII），
// 消除原 C 代码中的 goto fail 手动清理模式。
struct scclComm
{
    uint32_t rank = 0;     // 本节点 rank
    uint32_t nRanks = 0;   // 总节点数
    uint32_t rootRank = 0; // root 节点 rank
    int listenFd = -1;     // 本节点监听 fd
    std::vector<int>
        connFds; // 连接 fd 数组（按 rank 索引，connFds[rootRank]==-1）
    std::vector<scclPeerAddr> peerAddrs; // 所有节点地址表

    // RAII：析构时自动关闭所有打开的 fd
    ~scclComm()
    {
        if (listenFd >= 0)
            close(listenFd);
        for (int fd : connFds)
            if (fd >= 0)
                close(fd);
    }

    // 不可拷贝（持有 fd 资源）
    scclComm(const scclComm &) = delete;
    scclComm &operator=(const scclComm &) = delete;

    // 默认构造
    scclComm() = default;
};
