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

#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

// ---------- GPU prefix for Mooncake location strings ----------
const static std::string GPU_PREFIX = "xpu:";

// ---------- Error types ----------
typedef int cudaError_t;
#define cudaSuccess 0

// ---------- Memory type enum ----------
enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
};

// ---------- Pointer attributes ----------
struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

// ---------- Copy direction enum ----------
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

// ---------- Stream / Event opaque types ----------
typedef void *cudaStream_t;
typedef void *cudaEvent_t;

// ---------- XPU runtime context (singleton) ----------
// Internal state managed in xpu.cpp; these functions provide
// the cuda-alike API surface that the rest of Mooncake uses.

// Device management
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceCount(int *count);

// Memory management
cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream);

// Host pinned memory
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t cudaFreeHost(void *ptr);
#define cudaHostAllocMapped 0x02

// Pointer queries
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                     const void *ptr);
const char *cudaGetErrorString(cudaError_t error);

// Device PCI bus ID (for topology discovery)
cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);

// ---------- XPU-specific helpers for RDMA DMA-BUF ----------
// These are NOT cuda-alike; they are called explicitly under #ifdef USE_XPU.
namespace mooncake {
namespace xpu {

// Initialise Level Zero (idempotent). Returns 0 on success.
int ensureInitialized();

// Get the Level Zero context handle (created once on init).
ze_context_handle_t getContext();

// Get the Level Zero device handle for the given ordinal.
ze_device_handle_t getDevice(int ordinal);

// Query the base address and size of the allocation containing `ptr`.
// Needed because a tensor may sit at an offset inside a larger allocation
// (PyTorch's caching allocator packs several tensors per block).
// Returns 0 on success, -1 on failure.
int getAllocBase(const void *ptr, void **base, size_t *size);

// Export a DMA-BUF file descriptor for a device allocation.
// Returns >=0 fd on success, -1 on failure. The fd is OWNED BY THE CALLER,
// which must close() it when done. `size` is ignored: the exported dma_buf
// always covers the whole allocation.
int exportDmaBufFd(void *devPtr, size_t size);

// Query whether `ptr` is XPU device memory.
bool isDeviceMemory(const void *ptr);

}  // namespace xpu
}  // namespace mooncake
