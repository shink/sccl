#define _POSIX_C_SOURCE 199309L

#include "sccl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROOTINFO_PATH "/tmp/sccl_rootinfo_ext"
#define ROOTINFO_TMP "/tmp/sccl_rootinfo_ext.tmp"
#define POLL_INTERVAL_MS 100
#define POLL_MAX_RETRIES 100

static int discover_rank_and_size(uint32_t *rank, uint32_t *size)
{
    const char *r = getenv("OMPI_COMM_WORLD_RANK");
    const char *s = getenv("OMPI_COMM_WORLD_SIZE");
    if (!r)
        r = getenv("PMI_RANK");
    if (!s)
        s = getenv("PMI_SIZE");
    if (!r || !s)
    {
        *rank = 0;
        *size = 1;
        return 0;
    }
    *rank = (uint32_t)atoi(r);
    *size = (uint32_t)atoi(s);
    return 0;
}

static int write_rootinfo(const scclRootInfo_t rootInfo)
{
    FILE *f = fopen(ROOTINFO_TMP, "wb");
    if (!f)
        return -1;
    size_t n = fwrite((const void *)rootInfo, 1, SCCL_ROOT_INFO_BYTES, f);
    fclose(f);
    if (n != SCCL_ROOT_INFO_BYTES)
        return -1;
    return rename(ROOTINFO_TMP, ROOTINFO_PATH);
}

static int read_rootinfo(scclRootInfo_t rootInfo)
{
    for (int i = 0; i < POLL_MAX_RETRIES; i++)
    {
        FILE *f = fopen(ROOTINFO_PATH, "rb");
        if (f)
        {
            size_t n = fread((void *)rootInfo, 1, SCCL_ROOT_INFO_BYTES, f);
            fclose(f);
            if (n == SCCL_ROOT_INFO_BYTES)
                return 0;
        }
        struct timespec ts = {.tv_sec = POLL_INTERVAL_MS / 1000,
                              .tv_nsec =
                                  (long)(POLL_INTERVAL_MS % 1000) * 1000000L};
        nanosleep(&ts, NULL);
    }
    return -1;
}

static int subtest_int32(uint32_t rank, uint32_t nRanks, size_t count,
                         scclComm_t comm, const char *label)
{
    int32_t *sendbuf = malloc(count * sizeof(int32_t));
    int32_t *recvbuf = malloc((size_t)nRanks * count * sizeof(int32_t));
    if (!sendbuf || !recvbuf)
    {
        free(sendbuf);
        free(recvbuf);
        fprintf(stderr, "[ext][rank %u] %s alloc failed\n", rank, label);
        return -1;
    }

    for (size_t i = 0; i < count; i++)
        sendbuf[i] = (int32_t)(rank * 1000 + (uint32_t)i);

    scclResult_t ret = ScclAllGather(sendbuf, recvbuf, count, SCCL_INT32, comm);
    if (ret != SCCL_SUCCESS)
    {
        fprintf(stderr, "[ext][rank %u] %s AllGather failed: %d\n", rank, label,
                ret);
        free(sendbuf);
        free(recvbuf);
        return -1;
    }

    int errors = 0;
    for (uint32_t r = 0; r < nRanks; r++)
    {
        for (size_t i = 0; i < count; i++)
        {
            int32_t expected = (int32_t)(r * 1000 + (uint32_t)i);
            int32_t actual = recvbuf[r * count + i];
            if (actual != expected)
            {
                if (errors < 5)
                    fprintf(stderr,
                            "[ext][rank %u] %s mismatch [%u][%zu]: got %d, "
                            "expected %d\n",
                            rank, label, r, i, actual, expected);
                errors++;
            }
        }
    }

    free(sendbuf);
    free(recvbuf);
    return errors;
}

static int subtest_uint32(uint32_t rank, uint32_t nRanks, size_t count,
                          scclComm_t comm, const char *label)
{
    uint32_t *sendbuf = malloc(count * sizeof(uint32_t));
    uint32_t *recvbuf = malloc((size_t)nRanks * count * sizeof(uint32_t));
    if (!sendbuf || !recvbuf)
    {
        free(sendbuf);
        free(recvbuf);
        fprintf(stderr, "[ext][rank %u] %s alloc failed\n", rank, label);
        return -1;
    }

    for (size_t i = 0; i < count; i++)
        sendbuf[i] = (uint32_t)(rank * 100000u + (uint32_t)i);

    scclResult_t ret =
        ScclAllGather(sendbuf, recvbuf, count, SCCL_UINT32, comm);
    if (ret != SCCL_SUCCESS)
    {
        fprintf(stderr, "[ext][rank %u] %s AllGather failed: %d\n", rank, label,
                ret);
        free(sendbuf);
        free(recvbuf);
        return -1;
    }

    int errors = 0;
    for (uint32_t r = 0; r < nRanks; r++)
    {
        for (size_t i = 0; i < count; i++)
        {
            uint32_t expected = (uint32_t)(r * 100000u + (uint32_t)i);
            uint32_t actual = recvbuf[r * count + i];
            if (actual != expected)
            {
                if (errors < 5)
                    fprintf(stderr,
                            "[ext][rank %u] %s mismatch [%u][%zu]: got %u, "
                            "expected %u\n",
                            rank, label, r, i, actual, expected);
                errors++;
            }
        }
    }

    free(sendbuf);
    free(recvbuf);
    return errors;
}

int main(int argc, char *argv[])
{
    uint32_t rank, nRanks;
    discover_rank_and_size(&rank, &nRanks);

    uint32_t rootRank = (argc > 1) ? (uint32_t)atoi(argv[1]) : nRanks - 1;
    if (rootRank >= nRanks)
    {
        fprintf(stderr, "[ext][rank %u] rootRank %u >= nRanks %u\n", rank,
                rootRank, nRanks);
        return 1;
    }

    fprintf(stderr, "[ext] rank=%u nRanks=%u rootRank=%u starting\n", rank,
            nRanks, rootRank);

    scclRootInfo_t rootInfo = NULL;

    if (rank == rootRank)
    {
        remove(ROOTINFO_PATH);
        remove(ROOTINFO_TMP);

        scclResult_t ret = HcclGetRootInfo(&rootInfo, rootRank);
        if (ret != SCCL_SUCCESS)
        {
            fprintf(stderr, "[ext][rank %u] HcclGetRootInfo failed: %d\n", rank,
                    ret);
            return 1;
        }
        if (write_rootinfo(rootInfo) != 0)
        {
            free(rootInfo);
            return 1;
        }
    }
    else
    {
        rootInfo = (scclRootInfo_t)malloc(SCCL_ROOT_INFO_BYTES);
        if (!rootInfo)
            return 1;
        if (read_rootinfo(rootInfo) != 0)
        {
            free(rootInfo);
            return 1;
        }
    }

    scclComm_t comm = NULL;
    scclResult_t ret = ScclCommInitRootInfo(nRanks, &rootInfo, rank, &comm);
    if (ret != SCCL_SUCCESS)
    {
        fprintf(stderr, "[ext][rank %u] init failed: %d\n", rank, ret);
        free(rootInfo);
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
        fprintf(stderr,
                "[ext][rank %u] ALL SUBTESTS PASS: nRanks=%u rootRank=%u, "
                "6 cases\n",
                rank, nRanks, rootRank);
    }
    else
    {
        fprintf(stderr, "[ext][rank %u] SUBTESTS FAILED: %d total errors\n",
                rank, total_errors);
    }

    ScclCommDestroy(comm);
    free(rootInfo);
    return total_errors > 0 ? 1 : 0;
}
