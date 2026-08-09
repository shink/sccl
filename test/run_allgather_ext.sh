#!/bin/bash
set -e

cd "$(dirname "$0")/.."

if [ ! -d build ]; then
    echo "Error: build directory not found. Run ./build.sh first."
    exit 1
fi

PASS=0
FAIL=0

for NP in 2 4 8; do
    ROOTS="0 $((NP - 1))"
    if [ "$NP" -gt 2 ]; then
        ROOTS="0 1 $((NP - 1))"
    fi
    for ROOT in $ROOTS; do
        rm -f /tmp/sccl_rootinfo_ext /tmp/sccl_rootinfo_ext.tmp
        echo "Running mpirun -np $NP ./build/test/test_allgather_ext $ROOT ..."
        if mpirun --oversubscribe -np $NP ./build/test/test_allgather_ext \
                "$ROOT" 2>test/allgather_ext_log_${NP}_${ROOT}.txt; then
            COUNT=$(grep -c "ALL SUBTESTS PASS" test/allgather_ext_log_${NP}_${ROOT}.txt)
            if [ "$COUNT" -eq "$NP" ]; then
                echo "  np=$NP rootRank=$ROOT: PASS (all $NP ranks, 6 cases each)"
                PASS=$((PASS + 1))
            else
                echo "  np=$NP rootRank=$ROOT: FAIL (expected $NP PASS lines, got $COUNT)"
                cat test/allgather_ext_log_${NP}_${ROOT}.txt
                FAIL=$((FAIL + 1))
            fi
        else
            echo "  np=$NP rootRank=$ROOT: FAIL (mpirun exit nonzero)"
            cat test/allgather_ext_log_${NP}_${ROOT}.txt
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "Summary: $PASS/$((PASS + FAIL)) configurations passed"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "PASS: all configurations verified allgather"
