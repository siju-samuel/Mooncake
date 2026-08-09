#!/bin/bash
# Build Mooncake with Intel XPU (Level Zero) support
# Usage: ./scripts/build_xpu.sh [build_dir]
#
# Prerequisites:
#   - conda activate mooncake-xpu
#   - source /opt/intel/oneapi/setvars.sh  (for oneAPI 2025.2+)
#
# Environment:
#   CONDA_PREFIX must point to a conda env with level-zero-devel, glog, gtest, etc.

set -e
set -x

BUILD_DIR="${1:-build-xpu}"

# Verify conda env is active
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: No conda env active. Run: conda activate mooncake-xpu"
    exit 1
fi

# Verify Level Zero headers are available
if [ ! -f "$CONDA_PREFIX/include/level_zero/ze_api.h" ]; then
    echo "ERROR: Level Zero headers not found in $CONDA_PREFIX/include/level_zero/"
    echo "Install: conda install -c conda-forge level-zero-devel"
    exit 1
fi

# Use GCC from conda for C++20 support
export CC=$(which x86_64-conda-linux-gnu-gcc 2>/dev/null || which gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++ 2>/dev/null || which g++)

# Conda cross-compilers use their own sysroot and may not see system headers
# (e.g. /usr/include/infiniband/verbs.h) or libraries (e.g. /usr/lib64/libnuma).
# Add them explicitly so RDMA and libnuma are found.
SYS_INCLUDE_FLAGS="-isystem /usr/include"
SYS_LINK_FLAGS="-L/usr/lib64 -Wl,-rpath-link,/usr/lib64"

echo "Using CC=$CC CXX=$CXX"
echo "CONDA_PREFIX=$CONDA_PREFIX"

cmake -B "$BUILD_DIR" -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DUSE_XPU=ON \
    -DUSE_TCP=ON \
    -DUSE_HTTP=ON \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_STORE=OFF \
    -DWITH_P2P_STORE=OFF \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_INCLUDE_PATH="$CONDA_PREFIX/include;/usr/include" \
    -DCMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib;/usr/lib64" \
    -DCMAKE_CXX_FLAGS="$SYS_INCLUDE_FLAGS" \
    -DCMAKE_C_FLAGS="$SYS_INCLUDE_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$SYS_LINK_FLAGS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$SYS_LINK_FLAGS" \
    -DCMAKE_MODULE_LINKER_FLAGS="$SYS_LINK_FLAGS"

cmake --build "$BUILD_DIR" -j "$(nproc)"

echo ""
echo "Build complete: $BUILD_DIR"
echo "Run tests with: cd $BUILD_DIR && ctest --output-on-failure"
