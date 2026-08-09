# 定义动态库
add_library(sccl SHARED)

target_sources(sccl PRIVATE
    comm/communicator.cpp
    ops/allgather.cpp
    socket/tcp.cpp
)

target_include_directories(sccl
    PUBLIC
        ${SCCL_DIR}/include
        ${SCCL_DIR}/include/sccl
    PRIVATE
        ${SCCL_DIR}/src/inc
        ${SCCL_DIR}/src/comm
        ${SCCL_DIR}/src/ops
        ${SCCL_DIR}/src/socket
)

target_compile_options(sccl PRIVATE
    -Werror
    -Wextra
    -Wall
)
