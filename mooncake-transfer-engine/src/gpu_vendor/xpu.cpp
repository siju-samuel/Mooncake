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

// Intel XPU (Level Zero) implementation of the cuda-alike API surface.
// This file is compiled only when USE_XPU is defined.

#include "gpu_vendor/xpu.h"

#include <glog/logging.h>
#include <level_zero/ze_api.h>

#include <cstdlib>
#include <unistd.h>  // dup()

#include <cstring>
#include <mutex>
#include <vector>

namespace {

// Thread-local current device index (matches CUDA per-thread semantics).
static thread_local int tl_current_device = 0;

// ---- Singleton Level Zero runtime state ----
struct ZeRuntime {
    bool initialized = false;
    ze_driver_handle_t driver = nullptr;
    ze_context_handle_t context = nullptr;
    std::vector<ze_device_handle_t> devices;
    // Per-device immediate command list for synchronous copies.
    std::vector<ze_command_list_handle_t> imm_cmd_lists;
    std::mutex mu;

    ~ZeRuntime() {
        for (auto cl : imm_cmd_lists) {
            if (cl) zeCommandListDestroy(cl);
        }
        if (context) zeContextDestroy(context);
    }
};

static ZeRuntime &runtime() {
    static ZeRuntime rt;
    return rt;
}

static int initRuntime() {
    auto &rt = runtime();
    std::lock_guard<std::mutex> lock(rt.mu);
    if (rt.initialized) return 0;

    ze_result_t res = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeInit failed: 0x" << std::hex << res;
        return -1;
    }

    // Get the first GPU driver.
    uint32_t driver_count = 0;
    res = zeDriverGet(&driver_count, nullptr);
    if (res != ZE_RESULT_SUCCESS || driver_count == 0) {
        LOG(ERROR) << "No Level Zero GPU drivers found (res=0x" << std::hex
                   << res << ", count=" << std::dec << driver_count << ")";
        return -1;
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    res = zeDriverGet(&driver_count, drivers.data());
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeDriverGet (fill) failed: 0x" << std::hex << res;
        return -1;
    }
    rt.driver = drivers[0];

    // Create context.
    ze_context_desc_t ctx_desc = {};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    res = zeContextCreate(rt.driver, &ctx_desc, &rt.context);
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeContextCreate failed: 0x" << std::hex << res;
        return -1;
    }

    // Enumerate GPU devices.
    uint32_t device_count = 0;
    res = zeDeviceGet(rt.driver, &device_count, nullptr);
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeDeviceGet (count) failed: 0x" << std::hex << res;
        return -1;
    }
    if (device_count == 0) {
        LOG(WARNING) << "No Level Zero GPU devices found";
        rt.initialized = true;
        return 0;
    }
    rt.devices.resize(device_count);
    res = zeDeviceGet(rt.driver, &device_count, rt.devices.data());
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeDeviceGet (fill) failed: 0x" << std::hex << res;
        return -1;
    }

    // Create one immediate command list per device for sync memcpy.
    rt.imm_cmd_lists.resize(device_count, nullptr);
    for (uint32_t i = 0; i < device_count; ++i) {
        ze_command_queue_desc_t cq_desc = {};
        cq_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        cq_desc.ordinal = 0;
        cq_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
        res = zeCommandListCreateImmediate(rt.context, rt.devices[i], &cq_desc,
                                           &rt.imm_cmd_lists[i]);
        if (res != ZE_RESULT_SUCCESS) {
            LOG(ERROR) << "zeCommandListCreateImmediate failed for device " << i
                       << ": 0x" << std::hex << res;
            return -1;
        }
    }

    rt.initialized = true;
    LOG(INFO) << "Level Zero XPU runtime initialized: " << device_count
              << " device(s)";
    return 0;
}

}  // anonymous namespace

// ========================================================================
// cuda-alike API
// ========================================================================

cudaError_t cudaSetDevice(int device) {
    if (initRuntime()) return -1;
    auto &rt = runtime();
    if (device < 0 || device >= (int)rt.devices.size()) return -1;
    tl_current_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
    if (initRuntime()) return -1;
    *device = tl_current_device;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count) {
    if (initRuntime()) return -1;
    *count = (int)runtime().devices.size();
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (initRuntime()) return -1;
    auto &rt = runtime();
    ze_device_mem_alloc_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    desc.ordinal = 0;
    ze_result_t res = zeMemAllocDevice(rt.context, &desc, size, 64,
                                       rt.devices[tl_current_device], devPtr);
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeMemAllocDevice failed: 0x" << std::hex << res;
        return -1;
    }
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) return cudaSuccess;
    if (initRuntime()) return -1;
    ze_result_t res = zeMemFree(runtime().context, devPtr);
    return (res == ZE_RESULT_SUCCESS) ? cudaSuccess : -1;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
    (void)kind;  // Level Zero figures out direction automatically.
    if (initRuntime()) return -1;
    auto &rt = runtime();
    int dev = tl_current_device;
    ze_result_t res = zeCommandListAppendMemoryCopy(
        rt.imm_cmd_lists[dev], dst, src, count, nullptr, 0, nullptr);
    return (res == ZE_RESULT_SUCCESS) ? cudaSuccess : -1;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    // For now, fall back to synchronous copy.
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    (void)flags;
    if (initRuntime()) return -1;
    auto &rt = runtime();
    ze_host_mem_alloc_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    ze_result_t res = zeMemAllocHost(rt.context, &desc, size, 64, pHost);
    return (res == ZE_RESULT_SUCCESS) ? cudaSuccess : -1;
}

cudaError_t cudaFreeHost(void *ptr) {
    if (!ptr) return cudaSuccess;
    if (initRuntime()) return -1;
    ze_result_t res = zeMemFree(runtime().context, ptr);
    return (res == ZE_RESULT_SUCCESS) ? cudaSuccess : -1;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                     const void *ptr) {
    if (initRuntime()) return -1;
    auto &rt = runtime();
    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t alloc_device = nullptr;
    ze_result_t res =
        zeMemGetAllocProperties(rt.context, ptr, &props, &alloc_device);
    if (res != ZE_RESULT_SUCCESS) {
        // Level Zero doesn't recognise this pointer.
        attributes->type = cudaMemoryTypeUnregistered;
        attributes->device = -1;
        attributes->devicePointer = nullptr;
        attributes->hostPointer = nullptr;
        return cudaSuccess;
    }

    if (props.type == ZE_MEMORY_TYPE_UNKNOWN) {
        // Not an L0 allocation — treat as unregistered (likely stack/heap).
        attributes->type = cudaMemoryTypeUnregistered;
        attributes->device = -1;
        attributes->devicePointer = nullptr;
        attributes->hostPointer = nullptr;
        return cudaSuccess;
    }

    switch (props.type) {
        case ZE_MEMORY_TYPE_DEVICE:
            attributes->type = cudaMemoryTypeDevice;
            attributes->devicePointer = const_cast<void *>(ptr);
            attributes->hostPointer = nullptr;
            // Find device ordinal.
            attributes->device = 0;
            for (size_t i = 0; i < rt.devices.size(); ++i) {
                if (rt.devices[i] == alloc_device) {
                    attributes->device = (int)i;
                    break;
                }
            }
            break;
        case ZE_MEMORY_TYPE_HOST:
        case ZE_MEMORY_TYPE_SHARED:
            attributes->type = cudaMemoryTypeHost;
            attributes->device = 0;
            attributes->devicePointer = nullptr;
            attributes->hostPointer = const_cast<void *>(ptr);
            break;
        default:
            attributes->type = cudaMemoryTypeUnregistered;
            attributes->device = -1;
            attributes->devicePointer = nullptr;
            attributes->hostPointer = nullptr;
            break;
    }
    return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t error) {
    if (error == cudaSuccess) return "success";
    return "Level Zero error";
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (initRuntime()) return -1;
    auto &rt = runtime();
    if (device < 0 || device >= (int)rt.devices.size()) return -1;

    ze_device_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ze_result_t res = zeDeviceGetProperties(rt.devices[device], &props);
    if (res != ZE_RESULT_SUCCESS) return -1;

    ze_pci_ext_properties_t pci_props = {};
    pci_props.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    res = zeDevicePciGetPropertiesExt(rt.devices[device], &pci_props);
    if (res == ZE_RESULT_SUCCESS) {
        snprintf(pciBusId, len, "%04x:%02x:%02x.%01x", pci_props.address.domain,
                 pci_props.address.bus, pci_props.address.device,
                 pci_props.address.function);
        return cudaSuccess;
    }

    // Fallback: use the device UUID or a placeholder.
    snprintf(pciBusId, len, "0000:00:%02x.0", device);
    return cudaSuccess;
}

// ========================================================================
// mooncake::xpu helpers
// ========================================================================

namespace mooncake {
namespace xpu {

int ensureInitialized() { return initRuntime(); }

ze_context_handle_t getContext() {
    initRuntime();
    return runtime().context;
}

ze_device_handle_t getDevice(int ordinal) {
    initRuntime();
    auto &rt = runtime();
    if (ordinal < 0 || ordinal >= (int)rt.devices.size()) return nullptr;
    return rt.devices[ordinal];
}

bool isDeviceMemory(const void *ptr) {
    if (initRuntime()) return false;
    auto &rt = runtime();
    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_result_t res = zeMemGetAllocProperties(rt.context, ptr, &props, nullptr);
    return (res == ZE_RESULT_SUCCESS && props.type == ZE_MEMORY_TYPE_DEVICE);
}

int getAllocBase(const void *ptr, void **base, size_t *size) {
    if (initRuntime()) return -1;
    auto &rt = runtime();
    ze_result_t res = zeMemGetAddressRange(rt.context, ptr, base, size);
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeMemGetAddressRange failed for " << ptr << ": 0x"
                   << std::hex << res;
        return -1;
    }
    return 0;
}

int exportDmaBufFd(void *devPtr, size_t size) {
    (void)size;  // the dma_buf covers the whole allocation
    if (initRuntime()) return -1;
    auto &rt = runtime();
    ze_ipc_mem_handle_t ipc_handle;
    ze_result_t res = zeMemGetIpcHandle(rt.context, devPtr, &ipc_handle);
    if (res != ZE_RESULT_SUCCESS) {
        LOG(ERROR) << "zeMemGetIpcHandle failed: 0x" << std::hex << res;
        return -1;
    }
    // The IPC handle on Linux with the xe/i915 driver carries a dma_buf fd in
    // the first sizeof(int) bytes of the opaque handle data.
    int driver_fd = -1;
    memcpy(&driver_fd, ipc_handle.data, sizeof(driver_fd));
    if (driver_fd < 0) {
        LOG(ERROR) << "DMA-BUF fd extraction failed (fd=" << driver_fd << ")";
        zeMemPutIpcHandle(rt.context, ipc_handle);
        return -1;
    }
    // The fd inside the IPC handle belongs to the driver: zeMemPutIpcHandle()
    // closes it. Callers here own what we return and close() it once every NIC
    // has registered, so hand back a dup() and release the driver's reference
    // immediately. Returning driver_fd directly would leak the handle and later
    // double-close the same descriptor number.
    int fd = dup(driver_fd);
    if (fd < 0) {
        PLOG(ERROR) << "dup() of dma_buf fd " << driver_fd << " failed";
    }
    zeMemPutIpcHandle(rt.context, ipc_handle);
    return fd;
}

}  // namespace xpu
}  // namespace mooncake
