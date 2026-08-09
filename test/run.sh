#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d build ]; then
    echo "Error: build directory not found. Run ./build.sh first."
    exit 1
fi

rm -f /tmp/sccl_rootinfo /tmp/sccl_rootinfo.tmp

echo "Running mpirun -np 4 ./build/test/test_init ..."
mpirun --oversubscribe -np 4 ./build/test/test_init 2>test/last_run.log

COUNT=$(grep -c "comm ready" test/last_run.log)
if [ "$COUNT" -ne 4 ]; then
    echo "FAIL: expected 4 'comm ready' lines, got $COUNT"
    cat test/last_run.log
    exit 1
fi

echo "PASS: all 4 ranks reported 'comm ready'"
