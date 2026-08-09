#define _POSIX_C_SOURCE 199309L

#include "sccl.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROOTINFO_PATH "/tmp/sccl_rootinfo"
#define ROOTINFO_TMP "/tmp/sccl_rootinfo.tmp"
#define POLL_INTERVAL_MS 100
#define POLL_MAX_RETRIES 100

static int discover_rank_and_size(uint32_t *rank, uint32_t *size)
{
    const char *rankStr = getenv("OMPI_COMM_WORLD_RANK");
    const char *sizeStr = getenv("OMPI_COMM_WORLD_SIZE");

    if (rankStr == NULL)
        rankStr = getenv("PMI_RANK");
    if (sizeStr == NULL)
        sizeStr = getenv("PMI_SIZE");

    if (rankStr == NULL || sizeStr == NULL)
    {
        *rank = 0;
        *size = 1;
        fprintf(stderr, "[test] no MPI env vars found, running as singleton\n");
        return 0;
    }

    *rank = (uint32_t)atoi(rankStr);
    *size = (uint32_t)atoi(sizeStr);
    return 0;
}

static int write_rootinfo(const scclRootInfo_t rootInfo)
{
    FILE *f = fopen(ROOTINFO_TMP, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "[test][rank 0] failed to open %s for writing\n",
                ROOTINFO_TMP);
        return -1;
    }
    size_t n = fwrite((const void *)rootInfo, 1, SCCL_ROOT_INFO_BYTES, f);
    fclose(f);
    if (n != SCCL_ROOT_INFO_BYTES)
    {
        fprintf(stderr, "[test][rank 0] short write: %zu bytes\n", n);
        return -1;
    }
    if (rename(ROOTINFO_TMP, ROOTINFO_PATH) != 0)
    {
        fprintf(stderr, "[test][rank 0] rename %s -> %s failed\n", ROOTINFO_TMP,
                ROOTINFO_PATH);
        return -1;
    }
    return 0;
}

static int read_rootinfo(scclRootInfo_t rootInfo)
{
    for (int i = 0; i < POLL_MAX_RETRIES; i++)
    {
        FILE *f = fopen(ROOTINFO_PATH, "rb");
        if (f != NULL)
        {
            size_t n = fread((void *)rootInfo, 1, SCCL_ROOT_INFO_BYTES, f);
            fclose(f);
            if (n == SCCL_ROOT_INFO_BYTES)
                return 0;
            fprintf(stderr, "[test] short read: %zu bytes, retrying\n", n);
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
        fprintf(stderr, "[test][rank 0] rootInfo written to %s\n",
                ROOTINFO_PATH);
    }
    else
    {
        rootInfo = (scclRootInfo_t)malloc(SCCL_ROOT_INFO_BYTES);
        if (rootInfo == NULL)
        {
            fprintf(stderr, "[test][rank %u] malloc failed\n", rank);
            return 1;
        }

        if (read_rootinfo(rootInfo) != 0)
        {
            fprintf(stderr, "[test][rank %u] timeout waiting for %s\n", rank,
                    ROOTINFO_PATH);
            free(rootInfo);
            return 1;
        }
        fprintf(stderr, "[test][rank %u] rootInfo read from %s\n", rank,
                ROOTINFO_PATH);
    }

    scclComm_t comm = NULL;
    scclResult_t ret = ScclCommInitRootInfo(nRanks, &rootInfo, rank, &comm);
    if (ret != SCCL_SUCCESS)
    {
        fprintf(stderr, "[test][rank %u] ScclCommInitRootInfo failed: %d\n",
                rank, ret);
        free(rootInfo);
        return 1;
    }

    fprintf(stderr, "[test][rank %u] comm ready: nRanks=%u\n", rank, nRanks);

    ScclCommDestroy(comm);
    free(rootInfo);

    fprintf(stderr, "[test][rank %u] done\n", rank);
    return 0;
}
