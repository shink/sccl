#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d build ]; then
    echo "Error: build directory not found. Run ./build.sh first."
    exit 1
fi

rm -f /tmp/sccl_rootinfo /tmp/sccl_rootinfo.tmp

echo "Running mpirun -np 4 ./build/test/test_allgather ..."
mpirun --oversubscribe -np 4 ./build/test/test_allgather 2>test/allgather_log.txt

COUNT=$(grep -c "allgather verified" test/allgather_log.txt)
if [ "$COUNT" -ne 4 ]; then
    echo "FAIL: expected 4 'allgather verified' lines, got $COUNT"
    cat test/allgather_log.txt
    exit 1
fi

echo "PASS: all 4 ranks verified allgather"
