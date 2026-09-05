// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>

#include "cuda_alike.h"
#include "gpu_vendor/xpu.h"

// Helper: returns true if at least one Intel XPU device is reachable via
// Level Zero on the current host. Tests that require GPU hardware call this
// in SetUp() and GTEST_SKIP() when it returns false so that the suite runs
// cleanly on head-nodes and CI machines that lack GPUs.
static bool hasXpuDevice() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    return true;
}

class XpuTransportTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        google::InitGoogleLogging("XpuTransportTest");
        FLAGS_logtostderr = 1;
        gpu_available_ = hasXpuDevice();
    }
    static void TearDownTestSuite() { google::ShutdownGoogleLogging(); }

    void SetUp() override {
        if (!gpu_available_) {
            GTEST_SKIP() << "No Intel XPU device available on this host";
        }
    }

    static bool gpu_available_;
};

bool XpuTransportTest::gpu_available_ = false;

// Test Level Zero initialization via cuda-alike shim
TEST_F(XpuTransportTest, LevelZeroInit) {
    // If we got here, SetUp() confirmed a device is available.
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    ASSERT_EQ(err, cudaSuccess) << "Level Zero initialization failed";
    ASSERT_GT(count, 0) << "No Intel XPU devices found";
    LOG(INFO) << "Found " << count << " Intel XPU device(s)";
}

// Test device selection
TEST_F(XpuTransportTest, SetDevice) {
    auto err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to set device 0";
}

// Test device memory allocation and free
TEST_F(XpuTransportTest, DeviceAllocFree) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    void *ptr = nullptr;
    auto err = cudaMalloc(&ptr, 4096);
    ASSERT_EQ(err, cudaSuccess) << "cudaMalloc (zeMemAllocDevice) failed";
    ASSERT_NE(ptr, nullptr);

    err = cudaFree(ptr);
    ASSERT_EQ(err, cudaSuccess) << "cudaFree (zeMemFree) failed";
}

// Test host memory allocation
TEST_F(XpuTransportTest, HostAlloc) {
    void *ptr = nullptr;
    auto err = cudaHostAlloc(&ptr, 4096, 0);
    ASSERT_EQ(err, cudaSuccess) << "cudaHostAlloc (zeMemAllocHost) failed";
    ASSERT_NE(ptr, nullptr);

    // Host memory should be writable
    std::memset(ptr, 0xAB, 4096);

    err = cudaFree(ptr);
    ASSERT_EQ(err, cudaSuccess);
}

// Test pointer attributes for device memory
TEST_F(XpuTransportTest, PointerAttributesDevice) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    void *ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&ptr, 4096), cudaSuccess);

    cudaPointerAttributes attrs;
    auto err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess)
        << "cudaPointerGetAttributes failed for device memory";
    ASSERT_EQ(attrs.type, cudaMemoryTypeDevice)
        << "Expected device memory type";

    cudaFree(ptr);
}

// Test pointer attributes for host memory
TEST_F(XpuTransportTest, PointerAttributesHost) {
    void *ptr = nullptr;
    ASSERT_EQ(cudaHostAlloc(&ptr, 4096, 0), cudaSuccess);

    cudaPointerAttributes attrs;
    auto err = cudaPointerGetAttributes(&attrs, ptr);
    ASSERT_EQ(err, cudaSuccess)
        << "cudaPointerGetAttributes failed for host memory";
    ASSERT_EQ(attrs.type, cudaMemoryTypeHost) << "Expected host memory type";

    cudaFree(ptr);
}

// Test memcpy host-to-device and device-to-host
TEST_F(XpuTransportTest, MemcpyHostDeviceRoundtrip) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const size_t size = 1024;
    std::vector<uint8_t> src(size, 0);
    std::vector<uint8_t> dst(size, 0);

    // Fill source with pattern
    for (size_t i = 0; i < size; ++i) src[i] = static_cast<uint8_t>(i & 0xFF);

    void *dev_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_ptr, size), cudaSuccess);

    // Host -> Device
    auto err = cudaMemcpy(dev_ptr, src.data(), size, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess) << "H2D memcpy failed";

    // Device -> Host
    err = cudaMemcpy(dst.data(), dev_ptr, size, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess) << "D2H memcpy failed";

    EXPECT_EQ(src, dst) << "Data mismatch after H2D + D2H roundtrip";

    cudaFree(dev_ptr);
}

// Test isDeviceMemory helper
TEST_F(XpuTransportTest, IsDeviceMemory) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    void *dev_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_ptr, 4096), cudaSuccess);

    EXPECT_TRUE(mooncake::xpu::isDeviceMemory(dev_ptr))
        << "Device pointer not recognized as device memory";

    void *host_ptr = nullptr;
    ASSERT_EQ(cudaHostAlloc(&host_ptr, 4096, 0), cudaSuccess);

    EXPECT_FALSE(mooncake::xpu::isDeviceMemory(host_ptr))
        << "Host pointer incorrectly identified as device memory";

    // Stack memory should not be device memory
    int stack_var = 42;
    EXPECT_FALSE(mooncake::xpu::isDeviceMemory(&stack_var));

    cudaFree(dev_ptr);
    cudaFree(host_ptr);
}

// Test DMA-BUF export for RDMA registration
TEST_F(XpuTransportTest, ExportDmaBufFd) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const size_t size = 4096;
    void *dev_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_ptr, size), cudaSuccess);

    int fd = mooncake::xpu::exportDmaBufFd(dev_ptr, size);
    EXPECT_GE(fd, 0) << "DMA-BUF fd export failed (fd=" << fd << ")";

    if (fd >= 0) {
        close(fd);
    }

    cudaFree(dev_ptr);
}

// exportDmaBufFd() must hand back an fd the caller owns, so exporting the same
// allocation twice yields two independent descriptors and closing one does not
// invalidate the other. A driver-owned fd would come back identical both times.
TEST_F(XpuTransportTest, ExportDmaBufFdIsCallerOwned) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const size_t size = 4096;
    void *dev_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_ptr, size), cudaSuccess);

    int fd1 = mooncake::xpu::exportDmaBufFd(dev_ptr, size);
    int fd2 = mooncake::xpu::exportDmaBufFd(dev_ptr, size);
    ASSERT_GE(fd1, 0);
    ASSERT_GE(fd2, 0);
    EXPECT_NE(fd1, fd2) << "exports must not alias the same descriptor";

    close(fd1);
    // fd2 must still be valid after fd1 is closed.
    EXPECT_EQ(fcntl(fd2, F_GETFD) >= 0, true)
        << "fd2 invalidated by closing fd1";
    close(fd2);

    cudaFree(dev_ptr);
}

// A pointer inside a larger allocation must report the allocation base and a
// matching offset — the RDMA dmabuf path registers at base+offset, so a wrong
// base silently registers the wrong bytes.
TEST_F(XpuTransportTest, GetAllocBaseReportsOffset) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const size_t size = 1 << 20;
    void *dev_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&dev_ptr, size), cudaSuccess);

    void *base = nullptr;
    size_t alloc_size = 0;
    ASSERT_EQ(mooncake::xpu::getAllocBase(dev_ptr, &base, &alloc_size), 0);
    EXPECT_EQ(base, dev_ptr);
    EXPECT_GE(alloc_size, size);

    // An interior pointer resolves to the same base, with a real offset.
    const size_t kOffset = 4096;
    void *interior = static_cast<char *>(dev_ptr) + kOffset;
    void *base2 = nullptr;
    size_t alloc_size2 = 0;
    ASSERT_EQ(mooncake::xpu::getAllocBase(interior, &base2, &alloc_size2), 0);
    EXPECT_EQ(base2, dev_ptr);
    EXPECT_EQ(static_cast<char *>(interior) - static_cast<char *>(base2),
              (ptrdiff_t)kOffset);

    cudaFree(dev_ptr);
}

// Test multiple allocations and frees
TEST_F(XpuTransportTest, MultipleAllocFree) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr int N = 16;
    void *ptrs[N] = {};

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(cudaMalloc(&ptrs[i], 4096 * (i + 1)), cudaSuccess);
        ASSERT_NE(ptrs[i], nullptr);
    }

    // All should be device memory
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(mooncake::xpu::isDeviceMemory(ptrs[i]));
    }

    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(cudaFree(ptrs[i]), cudaSuccess);
    }
}

// Test GPU_PREFIX is set correctly (no GPU needed — compile-time constant)
TEST(XpuCompileTest, GpuPrefix) {
    std::string prefix = GPU_PREFIX;
    EXPECT_EQ(prefix, "xpu:")
        << "GPU_PREFIX should be 'xpu:' when USE_XPU is defined";
}
