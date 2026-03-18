#!/bin/bash
#SBATCH --job-name=mooncake-xpu-test
#SBATCH --partition=bmg-B60
#SBATCH --nodelist=anbmg-c01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-xpu-test-%j.out
#SBATCH --error=slurm-xpu-test-%j.err
#
# Mooncake Intel XPU test job
# Usage:
#   sbatch scripts/slurm_xpu_test.sh              # build + test
#   sbatch scripts/slurm_xpu_test.sh --test-only   # test only (skip build)
#
# For multi-node testing:
#   Change --nodelist to anbmg-c[01-02] and --nodes=2

set -e

# ── Environment setup ───────────────────────────────────────────────────
source /opt/intel/oneapi/setvars.sh 2>/dev/null || true

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate mooncake-xpu

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

BUILD_DIR="build-xpu"
WORKSPACE_DIR="$SLURM_SUBMIT_DIR"
cd "$WORKSPACE_DIR"

echo "=== Mooncake XPU Test ==="
echo "Host:      $(hostname)"
echo "Job ID:    $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node:      $SLURM_NODELIST"
echo "Date:      $(date)"
echo "CONDA_PREFIX=$CONDA_PREFIX"

# ── Check for Intel GPUs ────────────────────────────────────────────────
echo ""
echo "=== Intel GPU Detection ==="
if command -v xpu-smi &>/dev/null; then
    xpu-smi discovery 2>/dev/null || echo "xpu-smi discovery failed (non-fatal)"
elif [ -d /sys/class/drm ]; then
    echo "DRM devices:"
    ls -la /sys/class/drm/card*/device/vendor 2>/dev/null | head -5
    for f in /sys/class/drm/card*/device/vendor; do
        vendor=$(cat "$f" 2>/dev/null)
        if [ "$vendor" = "0x8086" ]; then
            card=$(echo "$f" | grep -oP 'card\d+')
            echo "  $card: Intel GPU (vendor=$vendor)"
        fi
    done
fi

# ── Build ───────────────────────────────────────────────────────────────
if [ "$1" != "--test-only" ]; then
    echo ""
    echo "=== Building Mooncake with XPU support ==="
    bash scripts/build_xpu.sh "$BUILD_DIR"
else
    echo ""
    echo "=== Skipping build (--test-only) ==="
fi

# ── Run tests ───────────────────────────────────────────────────────────
echo ""
echo "=== Running XPU unit tests ==="
cd "$BUILD_DIR"

# Run XPU-specific test
if [ -f mooncake-transfer-engine/tests/xpu_transport_test ]; then
    echo "--- xpu_transport_test ---"
    ./mooncake-transfer-engine/tests/xpu_transport_test --gtest_output=xml:xpu_test_results.xml
    echo "xpu_transport_test: PASSED"
else
    echo "ERROR: xpu_transport_test binary not found"
    exit 1
fi

# Run generic unit tests (transport_uint_test, etc.)
echo ""
echo "--- transport_uint_test ---"
if [ -f mooncake-transfer-engine/tests/transport_uint_test ]; then
    ./mooncake-transfer-engine/tests/transport_uint_test
    echo "transport_uint_test: PASSED"
fi

echo ""
echo "=== All tests complete ==="
echo "Finished: $(date)"
