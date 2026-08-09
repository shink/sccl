#include "tcp.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

scclResult_t sccl_tcp_listen(int *outFd, int *outPort)
{
    // 创建 IPv4 TCP socket：阻塞模式
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        SCCL_LOG_ERROR("socket() failed: %s", std::strerror(errno));
        return SCCL_ERROR_SOCKET;
    }

    // SO_REUSEADDR：快速重启时复用 TIME_WAIT 端口
    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
    {
        SCCL_LOG_ERROR("setsockopt(SO_REUSEADDR) failed: %s",
                       std::strerror(errno));
        close(fd);
        return SCCL_ERROR_SOCKET;
    }

    // 绑定 INADDR_ANY:0 —— port=0 让 OS 分配空闲临时端口
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(0);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0)
    {
        SCCL_LOG_ERROR("bind() failed: %s", std::strerror(errno));
        close(fd);
        return SCCL_ERROR_SOCKET;
    }

    // backlog=64：bootstrap 阶段所有 rank 几乎同时连入
    if (listen(fd, 64) < 0)
    {
        SCCL_LOG_ERROR("listen() failed: %s", std::strerror(errno));
        close(fd);
        return SCCL_ERROR_SOCKET;
    }

    // getsockname 取 OS 实际分配的端口
    struct sockaddr_in local;
    socklen_t localLen = sizeof(local);
    std::memset(&local, 0, sizeof(local));
    if (getsockname(fd, reinterpret_cast<struct sockaddr *>(&local),
                    &localLen) < 0)
    {
        SCCL_LOG_ERROR("getsockname() failed: %s", std::strerror(errno));
        close(fd);
        return SCCL_ERROR_SOCKET;
    }

    *outFd = fd;
    *outPort = ntohs(local.sin_port);
    return SCCL_SUCCESS;
}

scclResult_t sccl_tcp_accept(int listenFd, char *outIp, size_t ipLen,
                             int *outFd)
{
    // accept 返回时即带客户端地址
    struct sockaddr_in client;
    socklen_t clientLen = sizeof(client);
    std::memset(&client, 0, sizeof(client));

    int fd = accept(listenFd, reinterpret_cast<struct sockaddr *>(&client),
                    &clientLen);
    if (fd < 0)
    {
        SCCL_LOG_ERROR("accept() failed: %s", std::strerror(errno));
        return SCCL_ERROR_SOCKET;
    }

    // inet_ntop 线程安全
    if (outIp != nullptr && ipLen > 0)
    {
        if (inet_ntop(AF_INET, &client.sin_addr, outIp,
                      static_cast<socklen_t>(ipLen)) == nullptr)
        {
            SCCL_LOG_ERROR("inet_ntop() failed: %s", std::strerror(errno));
            close(fd);
            return SCCL_ERROR_SOCKET;
        }
    }

    *outFd = fd;
    return SCCL_SUCCESS;
}

scclResult_t sccl_tcp_connect(const char *ip, int port, int *outFd)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        SCCL_LOG_ERROR("socket() failed: %s", std::strerror(errno));
        return SCCL_ERROR_SOCKET;
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (inet_pton(AF_INET, ip, &addr.sin_addr) <= 0)
    {
        SCCL_LOG_ERROR("inet_pton(%s) failed: %s", ip, std::strerror(errno));
        close(fd);
        return SCCL_ERROR_SOCKET;
    }

    if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) <
        0)
    {
        SCCL_LOG_ERROR("connect(%s:%d) failed: %s", ip, port,
                       std::strerror(errno));
        close(fd);
        return SCCL_ERROR_SOCKET;
    }

    *outFd = fd;
    return SCCL_SUCCESS;
}

scclResult_t sccl_tcp_sendall(int fd, const void *buf, size_t n)
{
    // TCP 是字节流，send 可能短写，必须循环至全部发出
    const auto *p = static_cast<const char *>(buf);
    size_t sent = 0;
    while (sent < n)
    {
        ssize_t w = send(fd, p + sent, n - sent, 0);
        if (w < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            SCCL_LOG_ERROR("send() failed: %s", std::strerror(errno));
            return SCCL_ERROR_SOCKET;
        }
        sent += static_cast<size_t>(w);
    }
    return SCCL_SUCCESS;
}

scclResult_t sccl_tcp_recvall(int fd, void *buf, size_t n)
{
    // recv 可能短读，循环直到收满 n 字节
    auto *p = static_cast<char *>(buf);
    size_t got = 0;
    while (got < n)
    {
        ssize_t r = recv(fd, p + got, n - got, 0);
        if (r < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            SCCL_LOG_ERROR("recv() failed: %s", std::strerror(errno));
            return SCCL_ERROR_SOCKET;
        }
        if (r == 0)
        {
            // 对端关闭但数据未收满，视为错误
            SCCL_LOG_ERROR("recv() peer closed before %zu bytes", n);
            return SCCL_ERROR_SOCKET;
        }
        got += static_cast<size_t>(r);
    }
    return SCCL_SUCCESS;
}
