#define _DEFAULT_SOURCE

#include "communicator.h"
#include "log.h"
#include "sccl.h"
#include "tcp.h"
#include "types.h"

#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// root 的监听 fd：HcclGetRootInfo 创建，ScclCommInitRootInfo(rank==rootRank)
// 消费
static int g_rootListenFd = -1;

// 获取本机非 loopback IPv4 地址
static scclResult_t get_local_ip(char *ip, size_t ipLen)
{
    struct ifaddrs *ifap, *ifa;
    if (getifaddrs(&ifap) != 0)
    {
        SCCL_LOG_ERROR("getifaddrs failed: %s", strerror(errno));
        return SCCL_ERROR_SOCKET;
    }
    for (ifa = ifap; ifa; ifa = ifa->ifa_next)
    {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET)
            continue;
        if (ifa->ifa_flags & IFF_LOOPBACK)
            continue;
        struct sockaddr_in *sa = (struct sockaddr_in *)ifa->ifa_addr;
        if (inet_ntop(AF_INET, &sa->sin_addr, ip, (socklen_t)ipLen))
        {
            freeifaddrs(ifap);
            return SCCL_SUCCESS;
        }
    }
    freeifaddrs(ifap);
    SCCL_LOG_ERROR("no non-loopback IPv4 interface found");
    return SCCL_ERROR_SOCKET;
}

// root 节点：创建监听 socket，填充 rootInfo（含 rootRank）
scclResult_t HcclGetRootInfo(scclRootInfo_t *rootInfo, uint32_t rootRank)
{
    SCCL_CHECK(!rootInfo, "rootInfo pointer is NULL");

    struct scclRootInfo *info = calloc(1, SCCL_ROOT_INFO_BYTES);
    SCCL_CHECK(!info, "calloc failed");
    *rootInfo = (scclRootInfo_t)info;

    info->rootRank = rootRank;

    // 随机魔数：优先 /dev/urandom，回退 time^pid
    FILE *f = fopen("/dev/urandom", "rb");
    if (f)
    {
        size_t nrd = fread(&info->magic, 1, sizeof(uint64_t), f);
        if (nrd != sizeof(uint64_t))
        {
            info->magic = (uint64_t)time(NULL) ^ (uint64_t)getpid();
        }
        fclose(f);
    }
    else
    {
        info->magic = (uint64_t)time(NULL) ^ (uint64_t)getpid();
    }

    // 创建监听 socket，OS 分配临时端口
    int port = 0;
    scclResult_t ret = sccl_tcp_listen(&g_rootListenFd, &port);
    if (ret != SCCL_SUCCESS)
    {
        free(info);
        *rootInfo = NULL;
        return ret;
    }

    ret = get_local_ip(info->ip, sizeof(info->ip));
    if (ret != SCCL_SUCCESS)
    {
        close(g_rootListenFd);
        g_rootListenFd = -1;
        free(info);
        *rootInfo = NULL;
        return ret;
    }

    info->port = (uint16_t)port;
    SCCL_LOG_INFO("root rank=%u listening on %s:%d", rootRank, info->ip,
                  (int)info->port);
    return SCCL_SUCCESS;
}

// root 路径：接受 nRanks-1 个连接，收集地址，广播完整表
static scclResult_t init_root(uint32_t nRanks, const struct scclRootInfo *info,
                              struct scclComm *comm)
{
    SCCL_CHECK(g_rootListenFd < 0, "root listen fd not initialized");
    SCCL_CHECK(info->rootRank >= nRanks, "rootRank %u >= nRanks %u",
               info->rootRank, nRanks);

    uint32_t rootRank = info->rootRank;
    struct scclPeerAddr *peers = calloc(nRanks, sizeof(struct scclPeerAddr));
    SCCL_CHECK(!peers, "calloc peers failed");

    peers[rootRank].rank = rootRank;
    memcpy(peers[rootRank].ip, info->ip, sizeof(peers[rootRank].ip));
    peers[rootRank].port = info->port;

    // connFds indexed by rank: connFds[rootRank]==-1 (root has no self-fd),
    // connFds[r]=fd to rank r. AllGather relies on this indexing.
    int *fds = calloc(nRanks, sizeof(int));
    if (!fds)
    {
        free(peers);
        return SCCL_ERROR_INTERNAL;
    }
    for (uint32_t i = 0; i < nRanks; i++)
        fds[i] = -1;

    scclResult_t ret = SCCL_SUCCESS;

    // 接受 nRanks-1 个连接，按 rank 存入对应槽位
    for (uint32_t i = 0; i < nRanks - 1; i++)
    {
        char ip[46] = {0};
        int fd = -1;
        ret = sccl_tcp_accept(g_rootListenFd, ip, sizeof(ip), &fd);
        if (ret != SCCL_SUCCESS)
            goto fail;

        struct scclPeerAddr peer;
        ret = sccl_tcp_recvall(fd, &peer, sizeof(peer));
        if (ret != SCCL_SUCCESS)
        {
            close(fd);
            goto fail;
        }
        if (peer.rank == rootRank || peer.rank >= nRanks)
        {
            SCCL_LOG_ERROR("invalid rank %u (rootRank=%u, nRanks=%u)",
                           peer.rank, rootRank, nRanks);
            close(fd);
            ret = SCCL_ERROR_BOOTSTRAP;
            goto fail;
        }
        peers[peer.rank] = peer;
        fds[peer.rank] = fd;
        SCCL_LOG_INFO("accepted rank %u from %s:%d", peer.rank, ip, peer.port);
    }

    // 广播完整地址表给所有已连接 rank
    for (uint32_t r = 0; r < nRanks; r++)
    {
        if (r == rootRank)
            continue;
        ret = sccl_tcp_sendall(fds[r], peers,
                               nRanks * sizeof(struct scclPeerAddr));
        if (ret != SCCL_SUCCESS)
            goto fail;
    }

    close(g_rootListenFd);
    g_rootListenFd = -1;

    comm->rank = rootRank;
    comm->nRanks = nRanks;
    comm->rootRank = rootRank;
    comm->listenFd = -1;
    comm->connFds = fds;
    comm->nConnFds = (int)nRanks;
    comm->peerAddrs = peers;
    SCCL_LOG_INFO("comm init done: rootRank=%u nRanks=%u", rootRank, nRanks);
    return SCCL_SUCCESS;

fail:
    for (uint32_t i = 0; i < nRanks; i++)
        if (fds[i] >= 0)
            close(fds[i]);
    free(fds);
    free(peers);
    if (g_rootListenFd >= 0)
    {
        close(g_rootListenFd);
        g_rootListenFd = -1;
    }
    return ret;
}

// 非 root 路径：连接 root，发送地址，接收完整表
static scclResult_t init_nonroot(uint32_t nRanks,
                                 const struct scclRootInfo *info, uint32_t rank,
                                 struct scclComm *comm)
{
    SCCL_CHECK(info->rootRank >= nRanks, "rootRank %u >= nRanks %u",
               info->rootRank, nRanks);

    uint32_t rootRank = info->rootRank;
    int myFd = -1, rootFd = -1;
    struct scclPeerAddr *peers = NULL;

    // 创建自己的监听 socket（Phase 2 数据通路用）
    int myPort = 0;
    scclResult_t ret = sccl_tcp_listen(&myFd, &myPort);
    if (ret != SCCL_SUCCESS)
        goto fail;

    char myIp[46] = {0};
    ret = get_local_ip(myIp, sizeof(myIp));
    if (ret != SCCL_SUCCESS)
        goto fail;

    // 连接 root
    ret = sccl_tcp_connect(info->ip, info->port, &rootFd);
    if (ret != SCCL_SUCCESS)
        goto fail;

    // 发送自己的地址给 root
    struct scclPeerAddr me = {.rank = rank, .port = (uint16_t)myPort};
    memcpy(me.ip, myIp, sizeof(me.ip));
    ret = sccl_tcp_sendall(rootFd, &me, sizeof(me));
    if (ret != SCCL_SUCCESS)
        goto fail;

    // 接收完整地址表
    peers = calloc(nRanks, sizeof(struct scclPeerAddr));
    if (!peers)
    {
        ret = SCCL_ERROR_INTERNAL;
        goto fail;
    }
    ret = sccl_tcp_recvall(rootFd, peers, nRanks * sizeof(struct scclPeerAddr));
    if (ret != SCCL_SUCCESS)
        goto fail;

    // connFds indexed by rank: only connFds[rootRank] is set (fd to root),
    // rest -1. Matches init_root's indexing for uniform AllGather access.
    comm->connFds = calloc(nRanks, sizeof(int));
    if (!comm->connFds)
    {
        ret = SCCL_ERROR_INTERNAL;
        goto fail;
    }
    for (uint32_t i = 0; i < nRanks; i++)
        comm->connFds[i] = -1;
    comm->connFds[rootRank] = rootFd;
    rootFd = -1;
    comm->rank = rank;
    comm->nRanks = nRanks;
    comm->rootRank = rootRank;
    comm->listenFd = myFd;
    myFd = -1;
    comm->nConnFds = (int)nRanks;
    comm->peerAddrs = peers;
    SCCL_LOG_INFO("comm init done: rank=%u rootRank=%u nRanks=%u", rank,
                  rootRank, nRanks);
    return SCCL_SUCCESS;

fail:
    if (rootFd >= 0)
        close(rootFd);
    if (myFd >= 0)
        close(myFd);
    free(peers);
    return ret;
}

// 各节点调用：根据 rootInfo 建立通信域
scclResult_t ScclCommInitRootInfo(uint32_t nRanks,
                                  const scclRootInfo_t *rootInfo, uint32_t rank,
                                  scclComm_t *comm)
{
    SCCL_CHECK(!comm, "comm is NULL");
    SCCL_CHECK(!rootInfo || !*rootInfo, "rootInfo is NULL");
    SCCL_CHECK(nRanks == 0, "nRanks is 0");
    SCCL_CHECK(rank >= nRanks, "rank %u >= nRanks %u", rank, nRanks);

    scclLogSetRank((int)rank);

    struct scclComm *c = calloc(1, sizeof(struct scclComm));
    SCCL_CHECK(!c, "calloc comm failed");

    const struct scclRootInfo *info = (const struct scclRootInfo *)*rootInfo;
    scclResult_t ret = (rank == info->rootRank)
                           ? init_root(nRanks, info, c)
                           : init_nonroot(nRanks, info, rank, c);
    if (ret != SCCL_SUCCESS)
    {
        free(c);
        *comm = NULL;
        return ret;
    }

    *comm = (scclComm_t)c;
    return SCCL_SUCCESS;
}

// 销毁通信域：关闭 fd，释放全部内存
scclResult_t ScclCommDestroy(scclComm_t comm)
{
    SCCL_CHECK(!comm, "comm is NULL");
    struct scclComm *c = (struct scclComm *)comm;

    if (c->listenFd >= 0)
        close(c->listenFd);
    if (c->connFds)
    {
        for (int i = 0; i < c->nConnFds; i++)
            if (c->connFds[i] >= 0)
                close(c->connFds[i]);
        free(c->connFds);
    }
    free(c->peerAddrs);
    free(c);
    return SCCL_SUCCESS;
}
