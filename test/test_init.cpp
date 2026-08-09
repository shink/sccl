#define _POSIX_C_SOURCE 199309L

#include "sccl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>

constexpr const char *ROOTINFO_PATH = "/tmp/sccl_rootinfo";
constexpr const char *ROOTINFO_TMP = "/tmp/sccl_rootinfo.tmp";
constexpr int POLL_INTERVAL_MS = 100;
constexpr int POLL_MAX_RETRIES = 100;

static void discover_rank_and_size(uint32_t &rank, uint32_t &size)
{
    const char *rankStr = std::getenv("OMPI_COMM_WORLD_RANK");
    const char *sizeStr = std::getenv("OMPI_COMM_WORLD_SIZE");

    if (rankStr == nullptr)
        rankStr = std::getenv("PMI_RANK");
    if (sizeStr == nullptr)
        sizeStr = std::getenv("PMI_SIZE");

    if (rankStr == nullptr || sizeStr == nullptr)
    {
        rank = 0;
        size = 1;
        std::fprintf(stderr,
                     "[test] no MPI env vars found, running as singleton\n");
        return;
    }

    rank = static_cast<uint32_t>(std::atoi(rankStr));
    size = static_cast<uint32_t>(std::atoi(sizeStr));
}

static int write_rootinfo(const scclRootInfo_t rootInfo)
{
    {
        std::ofstream f(ROOTINFO_TMP, std::ios::binary);
        if (!f)
        {
            std::fprintf(stderr,
                         "[test][rank 0] failed to open %s for writing\n",
                         ROOTINFO_TMP);
            return -1;
        }
        f.write(reinterpret_cast<const char *>(rootInfo), SCCL_ROOT_INFO_BYTES);
        if (!f)
        {
            std::fprintf(stderr, "[test][rank 0] write failed\n");
            return -1;
        }
    } // f 析构时关闭文件
    if (std::rename(ROOTINFO_TMP, ROOTINFO_PATH) != 0)
    {
        std::fprintf(stderr, "[test][rank 0] rename %s -> %s failed\n",
                     ROOTINFO_TMP, ROOTINFO_PATH);
        return -1;
    }
    return 0;
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
        std::fprintf(stderr, "[test][rank 0] rootInfo written to %s\n",
                     ROOTINFO_PATH);
    }
    else
    {
        rootInfo =
            static_cast<scclRootInfo_t>(std::malloc(SCCL_ROOT_INFO_BYTES));
        if (rootInfo == nullptr)
        {
            std::fprintf(stderr, "[test][rank %u] malloc failed\n", rank);
            return 1;
        }

        if (read_rootinfo(rootInfo) != 0)
        {
            std::fprintf(stderr, "[test][rank %u] timeout waiting for %s\n",
                         rank, ROOTINFO_PATH);
            std::free(rootInfo);
            return 1;
        }
        std::fprintf(stderr, "[test][rank %u] rootInfo read from %s\n", rank,
                     ROOTINFO_PATH);
    }

    scclComm_t comm = nullptr;
    scclResult_t ret = ScclCommInitRootInfo(nRanks, &rootInfo, rank, &comm);
    if (ret != SCCL_SUCCESS)
    {
        std::fprintf(stderr,
                     "[test][rank %u] ScclCommInitRootInfo failed: %d\n", rank,
                     ret);
        std::free(rootInfo);
        return 1;
    }

    std::fprintf(stderr, "[test][rank %u] comm ready: nRanks=%u\n", rank,
                 nRanks);

    ScclCommDestroy(comm);
    std::free(rootInfo);

    std::fprintf(stderr, "[test][rank %u] done\n", rank);
    return 0;
}
