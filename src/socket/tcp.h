#pragma once

#include <stddef.h>
#include <stdint.h>

#include "log.h"
#include "sccl.h"

// 绑定 INADDR_ANY:0，让 OS 分配端口，返回监听 fd 和实际端口
scclResult_t sccl_tcp_listen(int *outFd, int *outPort);

// 接受一个连接，填充客户端 IP 字符串（inet_ntop，线程安全）
scclResult_t sccl_tcp_accept(int listenFd, char *outIp, size_t ipLen,
                             int *outFd);

// 连接到指定 IP:port
scclResult_t sccl_tcp_connect(const char *ip, int port, int *outFd);

// 循环发送直到 n 字节全部发出（处理短写与 EINTR）
scclResult_t sccl_tcp_sendall(int fd, const void *buf, size_t n);

// 循环接收直到 n 字节全部收到（处理短读与 EINTR）
scclResult_t sccl_tcp_recvall(int fd, void *buf, size_t n);
