#define _DEFAULT_SOURCE

#include "communicator.h"
#include "log.h"
#include "sccl.h"
#include "tcp.h"
#include "types.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ifaddrs.h>
#include <memory>
#include <net/if.h>
#include <unistd.h>

namespace
{
// root 的监听 fd：HcclGetRootInfo 创建，init_root 消费
int g_rootListenFd = -1;

// 获取本机非 loopback IPv4 地址
scclResult_t get_local_ip(char *ip, size_t ipLen)
{
    struct ifaddrs *ifap = nullptr;
    if (getifaddrs(&ifap) != 0)
    {
        SCCL_LOG_ERROR("getifaddrs failed: %s", std::strerror(errno));
        return SCCL_ERROR_SOCKET;
    }

    for (struct ifaddrs *ifa = ifap; ifa; ifa = ifa->ifa_next)
    {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET)
            continue;
        if (ifa->ifa_flags & IFF_LOOPBACK)
            continue;

        auto *sa = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
        if (inet_ntop(AF_INET, &sa->sin_addr, ip,
                      static_cast<socklen_t>(ipLen)))
        {
            freeifaddrs(ifap);
            return SCCL_SUCCESS;
        }
    }
    freeifaddrs(ifap);
    SCCL_LOG_ERROR("no non-loopback IPv4 interface found");
    return SCCL_ERROR_SOCKET;
}

// root 路径：接受 nRanks-1 个连接，收集地址，广播完整表
// RAII：fd 直接存入 comm->connFds，失败时由 scclComm 析构函数自动关闭
scclResult_t init_root(uint32_t nRanks, const scclRootInfo *info,
                       scclComm *comm)
{
    SCCL_CHECK(g_rootListenFd < 0, "root listen fd not initialized");
    SCCL_CHECK(info->rootRank >= nRanks, "rootRank %u >= nRanks %u",
               info->rootRank, nRanks);

    uint32_t rootRank = info->rootRank;

    // 预分配向量，fd 初始化为 -1
    comm->connFds.assign(nRanks, -1);
    comm->peerAddrs.resize(nRanks);

    // root 自身地址
    comm->peerAddrs[rootRank].rank = rootRank;
    std::memcpy(comm->peerAddrs[rootRank].ip, info->ip, sizeof(info->ip));
    comm->peerAddrs[rootRank].port = info->port;

    // 接受 nRanks-1 个连接，按 rank 存入对应槽位
    for (uint32_t i = 0; i < nRanks - 1; i++)
    {
        char ip[46] = {0};
        int fd = -1;
        scclResult_t ret = sccl_tcp_accept(g_rootListenFd, ip, sizeof(ip), &fd);
        if (ret != SCCL_SUCCESS)
        {
            // comm 析构函数会关闭已接受的 fd
            close(g_rootListenFd);
            g_rootListenFd = -1;
            return ret;
        }

        scclPeerAddr peer;
        ret = sccl_tcp_recvall(fd, &peer, sizeof(peer));
        if (ret != SCCL_SUCCESS)
        {
            close(fd);
            close(g_rootListenFd);
            g_rootListenFd = -1;
            return ret;
        }
        if (peer.rank == rootRank || peer.rank >= nRanks)
        {
            SCCL_LOG_ERROR("invalid rank %u (rootRank=%u, nRanks=%u)",
                           peer.rank, rootRank, nRanks);
            close(fd);
            close(g_rootListenFd);
            g_rootListenFd = -1;
            return SCCL_ERROR_BOOTSTRAP;
        }
        comm->peerAddrs[peer.rank] = peer;
        comm->connFds[peer.rank] = fd;
        SCCL_LOG_INFO("accepted rank %u from %s:%d", peer.rank, ip, peer.port);
    }

    // 广播完整地址表给所有已连接 rank
    for (uint32_t r = 0; r < nRanks; r++)
    {
        if (r == rootRank)
            continue;
        scclResult_t ret =
            sccl_tcp_sendall(comm->connFds[r], comm->peerAddrs.data(),
                             nRanks * sizeof(scclPeerAddr));
        if (ret != SCCL_SUCCESS)
        {
            close(g_rootListenFd);
            g_rootListenFd = -1;
            return ret;
        }
    }

    close(g_rootListenFd);
    g_rootListenFd = -1;

    comm->rank = rootRank;
    comm->nRanks = nRanks;
    comm->rootRank = rootRank;
    comm->listenFd = -1;
    SCCL_LOG_INFO("comm init done: rootRank=%u nRanks=%u", rootRank, nRanks);
    return SCCL_SUCCESS;
}

// 非 root 路径：连接 root，发送地址，接收完整表
// RAII：fd 直接存入 comm，失败时由 scclComm 析构函数自动关闭
scclResult_t init_nonroot(uint32_t nRanks, const scclRootInfo *info,
                          uint32_t rank, scclComm *comm)
{
    SCCL_CHECK(info->rootRank >= nRanks, "rootRank %u >= nRanks %u",
               info->rootRank, nRanks);

    uint32_t rootRank = info->rootRank;

    // 预分配向量
    comm->connFds.assign(nRanks, -1);
    comm->peerAddrs.resize(nRanks);

    // 创建自己的监听 socket（数据通路用）
    int myPort = 0;
    scclResult_t ret = sccl_tcp_listen(&comm->listenFd, &myPort);
    if (ret != SCCL_SUCCESS)
        return ret;

    char myIp[46] = {0};
    ret = get_local_ip(myIp, sizeof(myIp));
    if (ret != SCCL_SUCCESS)
        return ret;

    // 连接 root
    ret = sccl_tcp_connect(info->ip, info->port, &comm->connFds[rootRank]);
    if (ret != SCCL_SUCCESS)
        return ret;

    // 发送自己的地址给 root
    scclPeerAddr me{};
    me.rank = rank;
    me.port = static_cast<uint16_t>(myPort);
    std::memcpy(me.ip, myIp, sizeof(me.ip));
    ret = sccl_tcp_sendall(comm->connFds[rootRank], &me, sizeof(me));
    if (ret != SCCL_SUCCESS)
        return ret;

    // 接收完整地址表
    ret = sccl_tcp_recvall(comm->connFds[rootRank], comm->peerAddrs.data(),
                           nRanks * sizeof(scclPeerAddr));
    if (ret != SCCL_SUCCESS)
        return ret;

    comm->rank = rank;
    comm->nRanks = nRanks;
    comm->rootRank = rootRank;
    SCCL_LOG_INFO("comm init done: rank=%u rootRank=%u nRanks=%u", rank,
                  rootRank, nRanks);
    return SCCL_SUCCESS;
}
} // namespace

// root 节点调用：创建监听 socket，填充 rootInfo（含 rootRank）
// 注意：rootInfo 使用 calloc 分配，因为公共 API 约定调用方用 free() 释放。
scclResult_t HcclGetRootInfo(scclRootInfo_t *rootInfo, uint32_t rootRank)
{
    try
    {
        SCCL_CHECK(!rootInfo, "rootInfo pointer is NULL");

        auto *info =
            static_cast<scclRootInfo *>(std::calloc(1, SCCL_ROOT_INFO_BYTES));
        SCCL_CHECK(!info, "calloc failed");
        *rootInfo = reinterpret_cast<scclRootInfo_t>(info);

        info->rootRank = rootRank;

        // 随机魔数：优先 /dev/urandom，回退 time^pid
        FILE *f = std::fopen("/dev/urandom", "rb");
        if (f)
        {
            size_t nrd = std::fread(&info->magic, 1, sizeof(uint64_t), f);
            if (nrd != sizeof(uint64_t))
            {
                info->magic = static_cast<uint64_t>(std::time(nullptr)) ^
                              static_cast<uint64_t>(getpid());
            }
            std::fclose(f);
        }
        else
        {
            info->magic = static_cast<uint64_t>(std::time(nullptr)) ^
                          static_cast<uint64_t>(getpid());
        }

        // 创建监听 socket，OS 分配临时端口
        int port = 0;
        scclResult_t ret = sccl_tcp_listen(&g_rootListenFd, &port);
        if (ret != SCCL_SUCCESS)
        {
            std::free(info);
            *rootInfo = nullptr;
            return ret;
        }

        ret = get_local_ip(info->ip, sizeof(info->ip));
        if (ret != SCCL_SUCCESS)
        {
            close(g_rootListenFd);
            g_rootListenFd = -1;
            std::free(info);
            *rootInfo = nullptr;
            return ret;
        }

        info->port = static_cast<uint16_t>(port);
        SCCL_LOG_INFO("root rank=%u listening on %s:%d", rootRank, info->ip,
                      static_cast<int>(info->port));
        return SCCL_SUCCESS;
    }
    catch (const std::exception &e)
    {
        SCCL_LOG_ERROR("HcclGetRootInfo: %s", e.what());
        return SCCL_ERROR_INTERNAL;
    }
    catch (...)
    {
        SCCL_LOG_ERROR("HcclGetRootInfo: unknown exception");
        return SCCL_ERROR_INTERNAL;
    }
}

// 各节点调用：根据 rootInfo 建立通信域
scclResult_t ScclCommInitRootInfo(uint32_t nRanks,
                                  const scclRootInfo_t *rootInfo, uint32_t rank,
                                  scclComm_t *comm)
{
    try
    {
        SCCL_CHECK(!comm, "comm is NULL");
        SCCL_CHECK(!rootInfo || !*rootInfo, "rootInfo is NULL");
        SCCL_CHECK(nRanks == 0, "nRanks is 0");
        SCCL_CHECK(rank >= nRanks, "rank %u >= nRanks %u", rank, nRanks);

        scclLogSetRank(static_cast<int>(rank));

        // RAII：unique_ptr 确保 init 失败时自动析构（关闭 fd、释放内存）
        auto c = std::make_unique<scclComm>();

        const auto *info = reinterpret_cast<const scclRootInfo *>(*rootInfo);
        scclResult_t ret = (rank == info->rootRank)
                               ? init_root(nRanks, info, c.get())
                               : init_nonroot(nRanks, info, rank, c.get());
        if (ret != SCCL_SUCCESS)
        {
            // c 自动析构，RAII 清理所有资源
            *comm = nullptr;
            return ret;
        }

        *comm = reinterpret_cast<scclComm_t>(c.release());
        return SCCL_SUCCESS;
    }
    catch (const std::exception &e)
    {
        SCCL_LOG_ERROR("ScclCommInitRootInfo: %s", e.what());
        *comm = nullptr;
        return SCCL_ERROR_INTERNAL;
    }
    catch (...)
    {
        SCCL_LOG_ERROR("ScclCommInitRootInfo: unknown exception");
        *comm = nullptr;
        return SCCL_ERROR_INTERNAL;
    }
}

// 销毁通信域：delete 触发 scclComm 析构函数（RAII 关闭 fd + 释放 vector 内存）
scclResult_t ScclCommDestroy(scclComm_t comm)
{
    try
    {
        SCCL_CHECK(!comm, "comm is NULL");
        // delete 触发 ~scclComm()：关闭所有 fd，释放 vector 内存
        delete reinterpret_cast<scclComm *>(comm);
        return SCCL_SUCCESS;
    }
    catch (const std::exception &e)
    {
        SCCL_LOG_ERROR("ScclCommDestroy: %s", e.what());
        return SCCL_ERROR_INTERNAL;
    }
    catch (...)
    {
        SCCL_LOG_ERROR("ScclCommDestroy: unknown exception");
        return SCCL_ERROR_INTERNAL;
    }
}
