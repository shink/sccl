# 冒烟测试可执行文件
add_executable(test_init test_init.c)
target_link_libraries(test_init sccl)
target_include_directories(test_init PRIVATE
    ${CMAKE_SOURCE_DIR}/src/inc
)

# AllGather 测试
add_executable(test_allgather test_allgather.c)
target_link_libraries(test_allgather sccl)
target_include_directories(test_allgather PRIVATE
    ${CMAKE_SOURCE_DIR}/src/inc
)

# AllGather 扩展测试（INT32/UINT32 × count=1/4/8/1024 × np=2/4/8 × 非零 root）
add_executable(test_allgather_ext test_allgather_ext.c)
target_link_libraries(test_allgather_ext sccl)
target_include_directories(test_allgather_ext PRIVATE
    ${CMAKE_SOURCE_DIR}/src/inc
)
