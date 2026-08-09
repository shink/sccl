#define _POSIX_C_SOURCE 199309L

#include "sccl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

constexpr size_t COUNT = 4;
constexpr const char *ROOTINFO_PATH = "/tmp/sccl_rootinfo";
constexpr const char *ROOTINFO_TMP = "/tmp/sccl_rootinfo.tmp";
constexpr int POLL_INTERVAL_MS = 100;
constexpr int POLL_MAX_RETRIES = 100;

static void discover_rank_and_size(uint32_t &rank, uint32_t &size)
{
    const char *r = std::getenv("OMPI_COMM_WORLD_RANK");
    const char *s = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!r)
        r = std::getenv("PMI_RANK");
    if (!s)
        s = std::getenv("PMI_SIZE");
    if (!r || !s)
    {
        rank = 0;
        size = 1;
        return;
    }
    rank = static_cast<uint32_t>(std::atoi(r));
    size = static_cast<uint32_t>(std::atoi(s));
}

static int write_rootinfo(const scclRootInfo_t rootInfo)
{
    {
        std::ofstream f(ROOTINFO_TMP, std::ios::binary);
        if (!f)
            return -1;
        f.write(reinterpret_cast<const char *>(rootInfo), SCCL_ROOT_INFO_BYTES);
        if (!f)
            return -1;
    }
    return std::rename(ROOTINFO_TMP, ROOTINFO_PATH);
}

static int read_rootinfo(scclRootInfo_t rootInfo)
{
    for (int i = 0; i < POLL_MAX_RETRIES; i++)
    {
        {
            std::ifstream f(ROOTINFO_PATH, std::ios::binary);
            if (f)
            {
                f.read(reinterpret_cast<char *>(rootInfo),
                       SCCL_ROOT_INFO_BYTES);
                if (f && f.gcount() ==
                             static_cast<std::streamsize>(SCCL_ROOT_INFO_BYTES))
                    return 0;
            }
        }
        struct timespec ts = {POLL_INTERVAL_MS / 1000,
                              static_cast<long>(POLL_INTERVAL_MS % 1000) *
                                  1000000L};
        nanosleep(&ts, nullptr);
    }
    return -1;
}

int main()
{
    uint32_t rank = 0, nRanks = 0;
    discover_rank_and_size(rank, nRanks);
    std::fprintf(stderr, "[test] rank=%u nRanks=%u starting\n", rank, nRanks);

    scclRootInfo_t rootInfo = nullptr;

    if (rank == 0)
    {
        std::remove(ROOTINFO_PATH);
        std::remove(ROOTINFO_TMP);

        scclResult_t ret = HcclGetRootInfo(&rootInfo, 0);
        if (ret != SCCL_SUCCESS)
        {
            std::fprintf(stderr, "[test][rank 0] HcclGetRootInfo failed: %d\n",
                         ret);
            return 1;
        }
        if (write_rootinfo(rootInfo) != 0)
        {
            std::free(rootInfo);
            return 1;
        }
    }
    else
    {
        rootInfo =
            static_cast<scclRootInfo_t>(std::malloc(SCCL_ROOT_INFO_BYTES));
        if (!rootInfo)
            return 1;
        if (read_rootinfo(rootInfo) != 0)
        {
            std::free(rootInfo);
            return 1;
        }
    }

    scclComm_t comm = nullptr;
    scclResult_t ret = ScclCommInitRootInfo(nRanks, &rootInfo, rank, &comm);
    if (ret != SCCL_SUCCESS)
    {
        std::fprintf(stderr, "[test][rank %u] init failed: %d\n", rank, ret);
        std::free(rootInfo);
        return 1;
    }

    // sendbuf：栈上数组，COUNT 编译期已知
    std::vector<int32_t> sendbuf(COUNT);
    for (size_t i = 0; i < COUNT; i++)
        sendbuf[i] = static_cast<int32_t>(rank);

    // recvbuf：堆分配，RAII 自动释放
    std::vector<int32_t> recvbuf(static_cast<size_t>(nRanks) * COUNT);

    ret =
        ScclAllGather(sendbuf.data(), recvbuf.data(), COUNT, SCCL_INT32, comm);
    if (ret != SCCL_SUCCESS)
    {
        std::fprintf(stderr, "[test][rank %u] AllGather failed: %d\n", rank,
                     ret);
        ScclCommDestroy(comm);
        std::free(rootInfo);
        return 1;
    }

    int errors = 0;
    for (uint32_t r = 0; r < nRanks; r++)
    {
        for (size_t i = 0; i < COUNT; i++)
        {
            int32_t expected = static_cast<int32_t>(r);
            int32_t actual = recvbuf[r * COUNT + i];
            if (actual != expected)
            {
                std::fprintf(stderr,
                             "[test][rank %u] mismatch at [%u][%zu]: got %d, "
                             "expected %d\n",
                             rank, r, i, actual, expected);
                errors++;
            }
        }
    }

    if (errors == 0)
    {
        std::fprintf(stderr,
                     "[test][rank %u] allgather verified: %u ranks x %zu "
                     "elements\n",
                     rank, nRanks, COUNT);
    }
    else
    {
        std::fprintf(stderr, "[test][rank %u] allgather FAILED: %d errors\n",
                     rank, errors);
    }

    ScclCommDestroy(comm);
    std::free(rootInfo);

    return errors > 0 ? 1 : 0;
}
