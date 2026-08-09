#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // 通信域（不透明句柄，内部实现见 src/comm/communicator.h）
    struct scclComm;
    typedef struct scclComm *scclComm_t;

    // root 节点信息（不透明句柄，内部实现见 src/comm/communicator.h）
    struct scclRootInfo;
    typedef struct scclRootInfo *scclRootInfo_t;

    // rootInfo 的字节大小（用于跨进程传递）
    // 注意：保持 #define 而非 constexpr，因为此头文件需兼容 C 消费者
#define SCCL_ROOT_INFO_BYTES 64

    // 错误码
    typedef enum
    {
        SCCL_SUCCESS = 0,
        SCCL_ERROR_PARAM = 1,
        SCCL_ERROR_INTERNAL = 2,
        SCCL_ERROR_SOCKET = 3,
        SCCL_ERROR_BOOTSTRAP = 4
    } scclResult_t;

    // 数据类型
    typedef enum
    {
        SCCL_INT32 = 0,
        SCCL_UINT32 = 1
    } scclDataType_t;

#ifdef __cplusplus
}
#endif
