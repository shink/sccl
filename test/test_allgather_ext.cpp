#define _POSIX_C_SOURCE 199309L

#include "sccl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

constexpr const char *ROOTINFO_PATH = "/tmp/sccl_rootinfo_ext";
constexpr const char *ROOTINFO_TMP = "/tmp/sccl_rootinfo_ext.tmp";
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

static int subtest_int32(uint32_t rank, uint32_t nRanks, size_t count,
                         scclComm_t comm, const char *label)
{
    std::vector<int32_t> sendbuf(count);
    std::vector<int32_t> recvbuf(static_cast<size_t>(nRanks) * count);

    for (size_t i = 0; i < count; i++)
        sendbuf[i] =
            static_cast<int32_t>(rank * 1000 + static_cast<uint32_t>(i));

    scclResult_t ret =
        ScclAllGather(sendbuf.data(), recvbuf.data(), count, SCCL_INT32, comm);
    if (ret != SCCL_SUCCESS)
    {
        std::fprintf(stderr, "[ext][rank %u] %s AllGather failed: %d\n", rank,
                     label, ret);
        return -1;
    }

    int errors = 0;
    for (uint32_t r = 0; r < nRanks; r++)
    {
        for (size_t i = 0; i < count; i++)
        {
            int32_t expected =
                static_cast<int32_t>(r * 1000 + static_cast<uint32_t>(i));
            int32_t actual = recvbuf[r * count + i];
            if (actual != expected)
            {
                if (errors < 5)
                    std::fprintf(stderr,
                                 "[ext][rank %u] %s mismatch [%u][%zu]: got "
                                 "%d, expected %d\n",
                                 rank, label, r, i, actual, expected);
                errors++;
            }
        }
    }
    return errors;
}

static int subtest_uint32(uint32_t rank, uint32_t nRanks, size_t count,
                          scclComm_t comm, const char *label)
{
    std::vector<uint32_t> sendbuf(count);
    std::vector<uint32_t> recvbuf(static_cast<size_t>(nRanks) * count);

    for (size_t i = 0; i < count; i++)
        sendbuf[i] = rank * 100000u + static_cast<uint32_t>(i);

    scclResult_t ret =
        ScclAllGather(sendbuf.data(), recvbuf.data(), count, SCCL_UINT32, comm);
    if (ret != SCCL_SUCCESS)
    {
        std::fprintf(stderr, "[ext][rank %u] %s AllGather failed: %d\n", rank,
                     label, ret);
        return -1;
    }

    int errors = 0;
    for (uint32_t r = 0; r < nRanks; r++)
    {
        for (size_t i = 0; i < count; i++)
        {
            uint32_t expected = r * 100000u + static_cast<uint32_t>(i);
            uint32_t actual = recvbuf[r * count + i];
            if (actual != expected)
            {
                if (errors < 5)
                    std::fprintf(stderr,
                                 "[ext][rank %u] %s mismatch [%u][%zu]: got "
                                 "%u, expected %u\n",
                                 rank, label, r, i, actual, expected);
                errors++;
            }
        }
    }
    return errors;
}

int main(int argc, char *argv[])
{
    uint32_t rank = 0, nRanks = 0;
    discover_rank_and_size(rank, nRanks);

    uint32_t rootRank =
        (argc > 1) ? static_cast<uint32_t>(std::atoi(argv[1])) : nRanks - 1;
    if (rootRank >= nRanks)
    {
        std::fprintf(stderr, "[ext][rank %u] rootRank %u >= nRanks %u\n", rank,
                     rootRank, nRanks);
        return 1;
    }

    std::fprintf(stderr, "[ext] rank=%u nRanks=%u rootRank=%u starting\n", rank,
                 nRanks, rootRank);

    scclRootInfo_t rootInfo = nullptr;

    if (rank == rootRank)
    {
        std::remove(ROOTINFO_PATH);
        std::remove(ROOTINFO_TMP);

        scclResult_t ret = HcclGetRootInfo(&rootInfo, rootRank);
        if (ret != SCCL_SUCCESS)
        {
            std::fprintf(stderr, "[ext][rank %u] HcclGetRootInfo failed: %d\n",
                         rank, ret);
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
        std::fprintf(stderr, "[ext][rank %u] init failed: %d\n", rank, ret);
        std::free(rootInfo);
        return 1;
    }

    int total_errors = 0;
    total_errors += subtest_int32(rank, nRanks, 1, comm, "T1-int32-c1");
    total_errors += subtest_int32(rank, nRanks, 1024, comm, "T2-int32-c1024");
    total_errors += subtest_uint32(rank, nRanks, 4, comm, "T3-uint32-c4");
    total_errors += subtest_uint32(rank, nRanks, 1024, comm, "T4-uint32-c1024");
    total_errors += subtest_int32(rank, nRanks, 8, comm, "T5a-int32-c8");
    total_errors += subtest_int32(rank, nRanks, 8, comm, "T5b-int32-c8");

    if (total_errors == 0)
    {
        std::fprintf(stderr,
                     "[ext][rank %u] ALL SUBTESTS PASS: nRanks=%u rootRank=%u, "
                     "6 cases\n",
                     rank, nRanks, rootRank);
    }
    else
    {
        std::fprintf(stderr,
                     "[ext][rank %u] SUBTESTS FAILED: %d total errors\n", rank,
                     total_errors);
    }

    ScclCommDestroy(comm);
    std::free(rootInfo);
    return total_errors > 0 ? 1 : 0;
}
