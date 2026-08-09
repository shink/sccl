#define _POSIX_C_SOURCE 199309L

#include "sccl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define COUNT 4
#define ROOTINFO_PATH "/tmp/sccl_rootinfo"
#define ROOTINFO_TMP "/tmp/sccl_rootinfo.tmp"
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

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    uint32_t rank, nRanks;
    discover_rank_and_size(&rank, &nRanks);
    fprintf(stderr, "[test] rank=%u nRanks=%u starting\n", rank, nRanks);

    scclRootInfo_t rootInfo = NULL;

    if (rank == 0)
    {
        remove(ROOTINFO_PATH);
        remove(ROOTINFO_TMP);

        scclResult_t ret = HcclGetRootInfo(&rootInfo, 0);
        if (ret != SCCL_SUCCESS)
        {
            fprintf(stderr, "[test][rank 0] HcclGetRootInfo failed: %d\n", ret);
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
        fprintf(stderr, "[test][rank %u] init failed: %d\n", rank, ret);
        free(rootInfo);
        return 1;
    }

    int32_t sendbuf[COUNT];
    for (int i = 0; i < COUNT; i++)
        sendbuf[i] = (int32_t)rank;

    int32_t *recvbuf = malloc((size_t)nRanks * COUNT * sizeof(int32_t));
    if (!recvbuf)
    {
        ScclCommDestroy(comm);
        free(rootInfo);
        return 1;
    }

    ret = ScclAllGather(sendbuf, recvbuf, COUNT, SCCL_INT32, comm);
    if (ret != SCCL_SUCCESS)
    {
        fprintf(stderr, "[test][rank %u] AllGather failed: %d\n", rank, ret);
        free(recvbuf);
        ScclCommDestroy(comm);
        free(rootInfo);
        return 1;
    }

    int errors = 0;
    for (uint32_t r = 0; r < nRanks; r++)
    {
        for (int i = 0; i < COUNT; i++)
        {
            int32_t expected = (int32_t)r;
            int32_t actual = recvbuf[r * COUNT + i];
            if (actual != expected)
            {
                fprintf(stderr,
                        "[test][rank %u] mismatch at [%u][%d]: got %d, "
                        "expected %d\n",
                        rank, r, i, actual, expected);
                errors++;
            }
        }
    }

    if (errors == 0)
    {
        fprintf(stderr,
                "[test][rank %u] allgather verified: %u ranks x %d elements\n",
                rank, nRanks, COUNT);
    }
    else
    {
        fprintf(stderr, "[test][rank %u] allgather FAILED: %d errors\n", rank,
                errors);
    }

    free(recvbuf);
    ScclCommDestroy(comm);
    free(rootInfo);

    return errors > 0 ? 1 : 0;
}
