# RFC: Intel XPU (GPU) Support for Mooncake

| Field | Value |
|-------|-------|
| **Title** | Intel XPU Accelerator Support for Mooncake Transfer Engine, Store, and EP |
| **Author** | *[Your Name / GitHub Handle]* |
| **Status** | Draft |
| **Created** | 2026-03-16 |
| **Target Components** | mooncake-transfer-engine, mooncake-store, mooncake-ep, mooncake-pg, mooncake-integration, mooncake-wheel |

---

## Summary

This RFC proposes adding first-class Intel XPU (CRI/JGS) support to Mooncake. Intel XPUs use the oneAPI/SYCL programming model and the Level Zero runtime — a fundamentally different stack from CUDA/HIP/MUSA. This document outlines the integration strategy across all Mooncake layers: transfer engine transports, memory management, device detection, Python allocators, the EP (Expert Parallelism) kernel layer, the PG (Process Group) backend, and the build system.

The approach follows the established pattern Mooncake uses for non-NVIDIA accelerators (Ascend NPU, AMD HIP, Moore Threads MUSA), minimizing code churn while enabling Intel GPU hardware for distributed KV-cache serving in LLM inference workloads.

---

## Motivation

**Why Intel XPU?**

1. **Market availability**: Intel Data Center GPU Max (PVC) are deployed in HPC/AI clusters (Aurora supercomputer, etc.). LLM serving workloads are expanding to these platforms. JGS/TGS are new upcoming GPUs

2. **PyTorch ecosystem maturity**: PyTorch provides `torch.xpu` device support, XPU-aware `DistributedDataParallel`, and SYCL kernel compilation — making upstream framework integration feasible.

3. **RDMA compatibility**: Intel XPUs support PCIe peer-to-peer and DMA-BUF for GPUDirect-style RDMA, aligning with Mooncake's existing `ibv_reg_dmabuf_mr()` path.

4. **Competitive parity**: Mooncake already supports NVIDIA (CUDA), AMD (HIP), Moore Threads (MUSA), and Huawei (Ascend). Intel is the remaining major accelerator vendor without support.

5. **oneAPI/Level Zero maturity**: The Level Zero API provides low-level device control (memory allocation, P2P, events, command queues) comparable to CUDA Driver API, making high-performance transport implementations practical.

---

## Background: Mooncake Architecture

Mooncake's hardware abstraction has well-defined extension points. The following summarizes the layers a new accelerator must integrate with:

### Transfer Engine (Legacy Path)

```
TransferEngine (public API)
  └── MultiTransport (transport router)
        ├── RdmaTransport       ← always enabled
        ├── TcpTransport        ← USE_TCP
        ├── NvlinkTransport     ← USE_MNNVL (NVIDIA)
        ├── HipTransport        ← USE_MNNVL + USE_HIP (AMD)
        ├── AscendDirectTransport ← USE_ASCEND_DIRECT
        ├── EfaTransport        ← USE_EFA
        ├── CxlTransport        ← USE_CXL
        ├── BarexTransport      ← USE_BAREX (Moore Threads)
        └── [IntelXpuTransport] ← USE_XPU (proposed)
```

Each transport inherits from the `Transport` base class (`mooncake-transfer-engine/include/transport/transport.h`) and implements:
- `install()` — initialization
- `registerLocalMemory()` / `unregisterLocalMemory()` — memory region management
- `submitTransfer()` — data movement
- `getTransferStatus()` — completion polling

### Transfer Engine (TENT — New Path)

TENT provides a cleaner `device_plugin_t` C interface (`mooncake-transfer-engine/tent/include/tent/device_plugin.h`) with function pointers for:
- `alloc()` / `free()` — device memory management
- `memcpy_sync()` — synchronous copy
- `query_location()` — address-to-device mapping
- `get_device_count()` / `get_device_pci_bus_id()` — enumeration

A CUDA plugin already exists at `mooncake-transfer-engine/tent/plugins/cuda/`. The Intel XPU plugin would follow the same structure.

### GPU Memory Detection

`mooncake-transfer-engine/src/memory_location.cpp` uses `cudaPointerGetAttributes()` to distinguish GPU vs CPU memory. Intel XPU equivalent: `syclext::get_pointer_type()` or Level Zero `zeMemGetAddressRange()`.

### GPU Vendor Header Layer

`mooncake-transfer-engine/include/cuda_alike.h` provides compile-time vendor selection:
```cpp
#ifdef USE_CUDA    → #include <cuda.h>
#elif USE_HIP      → #include "gpu_vendor/hip.h"
#elif USE_MUSA     → #include "gpu_vendor/musa.h"
#elif USE_UBSHMEM  → #include "gpu_vendor/ubshmem.h"
// proposed:
#elif USE_XPU      → #include "gpu_vendor/xpu.h"
```

### Python Allocator Layer

`mooncake-integration/allocator.py` wraps C++ shared libraries for fabric memory allocation (e.g., `nvlink_allocator.so`). The Ascend NPU has its own allocator at `mooncake-integration/allocator_ascend_npu.py`.

### EP (Expert Parallelism) Layer

`mooncake-ep/` implements GPU-initiated RDMA for MoE dispatch/combine using CUDA kernels and IBGDA (InfiniBand GPU Direct Async). This is the most CUDA-coupled component.

### PG (Process Group) Layer

`mooncake-pg/` implements a PyTorch `c10d::Backend` with CUDA reduce kernels in `mooncake_worker.cu`.

---

## Detailed Design

### Phase 1: Core Transfer Engine Support

#### 1.1 Build System — `USE_XPU` Flag

**File**: `mooncake-common/common.cmake`

Add alongside existing GPU options:

```cmake
option(USE_XPU "option for enabling gpu features for Intel XPU" OFF)

if (USE_XPU)
  # Intel oneAPI Level Zero
  find_package(LevelZero REQUIRED)
  # Intel SYCL compiler (icpx)
  find_path(SYCL_INCLUDE_DIR sycl/sycl.hpp
    HINTS $ENV{ONEAPI_ROOT}/compiler/latest/include
          /opt/intel/oneapi/compiler/latest/include)
  find_library(ZE_LOADER_LIB ze_loader
    HINTS $ENV{ONEAPI_ROOT}/lib
          /opt/intel/oneapi/lib)

  if (NOT SYCL_INCLUDE_DIR OR NOT ZE_LOADER_LIB)
    message(FATAL_ERROR "Intel oneAPI/Level Zero not found. "
      "Set ONEAPI_ROOT or install intel-oneapi-devel.")
  endif()

  include_directories(${SYCL_INCLUDE_DIR} ${LevelZero_INCLUDE_DIRS})
  link_directories(${LevelZero_LIBRARY_DIRS})
  add_compile_definitions(USE_XPU)
  message(STATUS "Intel XPU (oneAPI/Level Zero) support is enabled")
endif()
```

**Mutual exclusivity guard** (same pattern as existing GPU flags):
```cmake
# At most one GPU vendor active
set(_gpu_count 0)
foreach(_flag USE_CUDA USE_HIP USE_MUSA USE_XPU)
  if(${${_flag}})
    math(EXPR _gpu_count "${_gpu_count} + 1")
  endif()
endforeach()
if(_gpu_count GREATER 1)
  message(FATAL_ERROR "Only one GPU vendor flag may be enabled at a time.")
endif()
```

#### 1.2 GPU Vendor Header — `gpu_vendor/xpu.h`

**New file**: `mooncake-transfer-engine/include/gpu_vendor/xpu.h`

Maps CUDA API names to Intel XPU equivalents via Level Zero / SYCL runtime:

```cpp
#pragma once
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <string>

// ---------- Device management ----------
#define cudaSetDevice(dev)           xpuSetDevice(dev)
#define cudaGetDeviceCount(count)    xpuGetDeviceCount(count)

// ---------- Memory management ----------
#define cudaMalloc(ptr, size)        xpuMalloc(ptr, size)
#define cudaFree(ptr)                xpuFree(ptr)
#define cudaMemcpy(dst, src, sz, k)  xpuMemcpy(dst, src, sz, k)
#define cudaMemcpyDeviceToHost       XPU_MEMCPY_D2H
#define cudaMemcpyHostToDevice       XPU_MEMCPY_H2D
#define cudaMemcpyDeviceToDevice     XPU_MEMCPY_D2D

// ---------- Pointer attributes ----------
#define cudaPointerGetAttributes(attr, ptr)  xpuPointerGetAttributes(attr, ptr)

// ---------- Stream / Event ----------
typedef void* cudaStream_t;  // wraps sycl::queue*
typedef void* cudaEvent_t;   // wraps sycl::event*
#define cudaSuccess              0
typedef int cudaError_t;

// ---------- Memory type enum ----------
#define cudaMemoryTypeDevice     2
#define cudaMemoryTypeHost       1

struct cudaPointerAttributes {
    int type;     // cudaMemoryTypeDevice or cudaMemoryTypeHost
    int device;   // device ordinal
    void* devicePointer;
};

// ---------- GPU prefix for location strings ----------
const static std::string GPU_PREFIX = "xpu:";

// Implementation in gpu_vendor/xpu.cpp (or inline):
int xpuSetDevice(int dev);
int xpuGetDeviceCount(int* count);
int xpuMalloc(void** ptr, size_t size);
int xpuFree(void* ptr);
int xpuMemcpy(void* dst, const void* src, size_t size, int kind);
int xpuPointerGetAttributes(cudaPointerAttributes* attr, const void* ptr);
```

The implementation file (`gpu_vendor/xpu.cpp`) wraps Level Zero calls:

```cpp
// Uses ze_driver_handle_t, ze_device_handle_t, zeMemAllocDevice, etc.
// Thread-local device selection state, cached device handles.
// zeMemGetAllocProperties() for pointer attribute queries.
```

#### 1.3 `cuda_alike.h` Integration

**File**: `mooncake-transfer-engine/include/cuda_alike.h`

Add `USE_XPU` branch:

```cpp
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include "gpu_vendor/hip.h"
#elif defined(USE_MUSA)
#include "gpu_vendor/musa.h"
#elif defined(USE_UBSHMEM)
#include "gpu_vendor/ubshmem.h"
#elif defined(USE_XPU)
#include "gpu_vendor/xpu.h"
#endif

#if !defined(USE_HIP) && !defined(USE_MUSA) && !defined(USE_UBSHMEM) && !defined(USE_XPU)
const static std::string GPU_PREFIX = "cuda:";
#endif
```

#### 1.4 Memory Location Detection

**File**: `mooncake-transfer-engine/src/memory_location.cpp`

The existing code gates on `USE_CUDA || USE_MUSA || USE_HIP`. Add `USE_XPU`:

```cpp
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) || defined(USE_XPU)
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, start);
    // ... existing GPU detection logic works via gpu_vendor/xpu.h mapping
#endif
```

Because `gpu_vendor/xpu.h` remaps `cudaPointerGetAttributes` to `xpuPointerGetAttributes`, this code compiles and works transparently.

#### 1.5 RDMA Transport — DMA-BUF for Intel XPU

**File**: `mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp`

The existing RDMA memory registration already supports DMA-BUF (`ibv_reg_dmabuf_mr`) for GPU memory without nvidia-peermem. Intel XPU supports DMA-BUF export via Level Zero:

```cpp
// In registerMemoryRegion(), add XPU DMA-BUF path:
#ifdef USE_XPU
if (isXpuDeviceMemory(addr)) {
    // Export DMA-BUF fd from Level Zero
    ze_external_memory_export_fd_t export_fd = {};
    export_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
    export_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    zeMemGetIpcHandle(ze_context, addr, &ipc_handle);
    // or use ze_external_memory_export_fd extension

    int dmabuf_fd = export_fd.fd;
    struct ibv_mr *mr = ibv_reg_dmabuf_mr(
        pd_, 0, length, (uint64_t)addr, dmabuf_fd, access);
    // ...
}
#endif
```

**Key prerequisite**: The Linux kernel must support i915/xe DMA-BUF export, and the RDMA NIC driver (mlx5, etc.) must support `ibv_reg_dmabuf_mr` — both are standard on modern kernels (5.12+) and MLNX_OFED 5.5+.

#### 1.6 TCP Transport — XPU Memory Staging

**File**: `mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp`

The `isCudaMemory()` helper already gates on `USE_CUDA || USE_HIP || USE_MUSA`. Extend:

```cpp
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_HIP) || defined(USE_XPU)
static bool isCudaMemory(void *addr) {
    cudaPointerAttributes attributes;
    // works transparently via gpu_vendor/xpu.h mapping
    auto status = cudaPointerGetAttributes(&attributes, addr);
    if (status != cudaSuccess) return false;
    return (attributes.type == cudaMemoryTypeDevice);
}
#endif
```

#### 1.7 XPU P2P Transport (Optional — Phase 1 Stretch Goal)

For multi-node Intel XPU-to-XPU transfers without RDMA (analogous to NVLink transport), a new `XpuTransport` class could leverage:

- **Intra-node**: Level Zero P2P copy (`zeCommandListAppendMemoryCopy` between devices on same node)
- **Inter-node**: Intel Xe Link fabric (if available on GPU Max 1550 with Xe Link bridges)

**New directory**: `mooncake-transfer-engine/include/transport/xpu_transport/`

```cpp
class XpuTransport : public Transport {
public:
    int install(std::string &local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo) override;

    int registerLocalMemory(void *addr, size_t length,
                            const std::string &location,
                            bool remote_accessible,
                            bool update_metadata) override;

    int unregisterLocalMemory(void *addr, bool update_metadata) override;

    int registerLocalMemoryBatch(
        const std::vector<BufferEntry> &buffer_list,
        const std::string &location) override;

    Status submitTransfer(BatchID batch_id,
        const std::vector<TransferRequest> &entries) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
        TransferStatus &status) override;

    const char *getName() const override { return "xpu"; }

    // XPU-specific: fabric memory allocation
    static void *allocatePinnedLocalMemory(size_t size);
    static void freePinnedLocalMemory(void *ptr);

private:
    ze_context_handle_t ze_context_;
    std::vector<ze_device_handle_t> devices_;
    std::vector<ze_command_queue_handle_t> queues_;
};
```

**Registration in `multi_transport.cpp`**:

```cpp
#ifdef USE_XPU
    else if (std::string(proto) == "xpu") {
        transport = new XpuTransport();
    }
#endif
```

**Transport `CMakeLists.txt`**:

```cmake
if (USE_XPU)
    add_subdirectory(xpu_transport)
    target_sources(transport PUBLIC $<TARGET_OBJECTS:xpu_transport>)
    target_link_libraries(transport PRIVATE ze_loader)
endif()
```

### Phase 2: TENT Device Plugin

#### 2.1 XPU Device Plugin

**New file**: `mooncake-transfer-engine/tent/plugins/xpu/xpu_plugin.cpp`

Following the CUDA plugin pattern at `tent/plugins/cuda/cuda_plugin.cpp`:

```cpp
#include "tent/device_plugin.h"
#include <level_zero/ze_api.h>
#include <cstring>
#include <vector>

struct XpuPluginContext {
    ze_driver_handle_t driver;
    std::vector<ze_device_handle_t> devices;
    ze_context_handle_t context;
};

static void* xpu_create_plugin() {
    auto* ctx = new XpuPluginContext();
    zeInit(0);
    uint32_t driver_count = 1;
    zeDriverGet(&driver_count, &ctx->driver);
    uint32_t device_count = 0;
    zeDeviceGet(ctx->driver, &device_count, nullptr);
    ctx->devices.resize(device_count);
    zeDeviceGet(ctx->driver, &device_count, ctx->devices.data());
    ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    zeContextCreate(ctx->driver, &ctx_desc, &ctx->context);
    return ctx;
}

static int xpu_alloc(void* handle, void** pptr, size_t size,
                     const char* location) {
    auto* ctx = (XpuPluginContext*)handle;
    int dev_idx = parse_device_index(location);  // "xpu:0" → 0
    ze_device_mem_alloc_desc_t desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    return zeMemAllocDevice(ctx->context, &desc, size, 64,
                            ctx->devices[dev_idx], pptr);
}

static int xpu_free(void* handle, void* ptr, size_t size) {
    auto* ctx = (XpuPluginContext*)handle;
    return zeMemFree(ctx->context, ptr);
}

static int xpu_memcpy_sync(void* handle, void* dst, void* src,
                           size_t length) {
    auto* ctx = (XpuPluginContext*)handle;
    ze_command_list_handle_t cmd_list;
    ze_command_list_desc_t desc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
    zeCommandListCreateImmediate(ctx->context, ctx->devices[0],
                                 &desc, &cmd_list);
    zeCommandListAppendMemoryCopy(cmd_list, dst, src, length,
                                   nullptr, 0, nullptr);
    zeCommandListDestroy(cmd_list);
    return 0;
}

static int xpu_query_location(void* handle, void* addr, size_t size,
                              location_t* buf, size_t buf_count) {
    auto* ctx = (XpuPluginContext*)handle;
    ze_memory_allocation_properties_t props = {
        ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES};
    ze_device_handle_t alloc_device;
    zeMemGetAllocProperties(ctx->context, addr, &props, &alloc_device);

    if (props.type == ZE_MEMORY_TYPE_DEVICE) {
        // Find device index
        for (size_t i = 0; i < ctx->devices.size(); i++) {
            if (ctx->devices[i] == alloc_device) {
                snprintf(buf[0].location, LOCATION_LEN, "xpu:%zu", i);
                buf[0].start = addr;
                buf[0].length = size;
                return 1;
            }
        }
    }
    snprintf(buf[0].location, LOCATION_LEN, "cpu:0");
    buf[0].start = addr;
    buf[0].length = size;
    return 1;
}

static int xpu_get_device_count(void* handle) {
    auto* ctx = (XpuPluginContext*)handle;
    return (int)ctx->devices.size();
}

static int xpu_get_device_pci_bus_id(void* handle, int device_index,
                                     char* pci_bus_id, size_t len) {
    auto* ctx = (XpuPluginContext*)handle;
    ze_pci_ext_properties_t pci = {
        ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
    zeDevicePciGetPropertiesExt(ctx->devices[device_index], &pci);
    snprintf(pci_bus_id, len, "%04x:%02x:%02x.%01x",
             pci.address.domain, pci.address.bus,
             pci.address.device, pci.address.function);
    return 0;
}

extern "C" int tent_register_device_plugin(device_plugin_t* out) {
    out->class_name = "xpu";
    out->create_plugin = xpu_create_plugin;
    out->destroy_plugin = [](void* h) -> int {
        auto* ctx = (XpuPluginContext*)h;
        zeContextDestroy(ctx->context);
        delete ctx;
        return 0;
    };
    out->alloc = xpu_alloc;
    out->free = xpu_free;
    out->memcpy_sync = xpu_memcpy_sync;
    out->query_location = xpu_query_location;
    out->get_device_count = xpu_get_device_count;
    out->get_device_pci_bus_id = xpu_get_device_pci_bus_id;
    return 0;
}
```

### Phase 3: Python Integration Layer

#### 3.1 XPU Memory Allocator — `allocator_intel_xpu.py`

**New file**: `mooncake-integration/allocator_intel_xpu.py`

Following the pattern of `allocator_ascend_npu.py`:

```python
import ctypes
import logging
import threading
from enum import IntEnum
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class MemoryBackend(IntEnum):
    UNKNOWN = -1
    USE_XPU_MALLOC = 0
    USE_ZE_MEM_CREATE = 1
    UNSUPPORTED = 2


class XpuFabricAllocator:
    """Intel XPU fabric memory allocator for Mooncake transfer engine."""

    _lock = threading.Lock()
    _supports_fabric: int = MemoryBackend.UNKNOWN
    _allocator_cache: dict = {}

    @classmethod
    def detect_mem_backend(cls) -> MemoryBackend:
        if cls._supports_fabric != MemoryBackend.UNKNOWN:
            return cls._supports_fabric

        try:
            lib = ctypes.CDLL("xpu_fabric_allocator.so")
            lib.mc_probe_xpu_fabric_support.restype = ctypes.c_int
            lib.mc_probe_xpu_fabric_support.argtypes = [ctypes.c_int]
            result = lib.mc_probe_xpu_fabric_support(0)
            cls._supports_fabric = MemoryBackend(result)
        except OSError:
            logger.warning("xpu_fabric_allocator.so not found")
            cls._supports_fabric = MemoryBackend.UNSUPPORTED

        return cls._supports_fabric

    @classmethod
    def get_allocator(cls, device: torch.device):
        device_index = device.index or 0
        with cls._lock:
            if device_index in cls._allocator_cache:
                return cls._allocator_cache[device_index]

            backend = cls.detect_mem_backend()
            if backend == MemoryBackend.UNSUPPORTED:
                logger.warning("XPU fabric memory not supported, "
                               "falling back to default allocator")
                return None

            # Use PyTorch built-in XPU pluggable allocator (torch.xpu)
            from torch.xpu.memory import XPUPluggableAllocator

            alloc = XPUPluggableAllocator(
                "xpu_fabric_allocator.so",
                "mc_xpu_fabric_malloc",
                "mc_xpu_fabric_free",
            )
            cls._allocator_cache[device_index] = alloc
            return alloc
```

#### 3.2 XPU Fabric Allocator — C++ Shared Library

**New file**: `mooncake-transfer-engine/xpu-allocator/xpu_fabric_allocator.cpp`

```cpp
#include <level_zero/ze_api.h>
#include <cstdlib>
#include <cstring>

extern "C" {

int mc_probe_xpu_fabric_support(int device_id) {
    if (zeInit(0) != ZE_RESULT_SUCCESS) return 2;  // UNSUPPORTED

    uint32_t driver_count = 1;
    ze_driver_handle_t driver;
    zeDriverGet(&driver_count, &driver);

    uint32_t device_count = 0;
    zeDeviceGet(driver, &device_count, nullptr);
    if ((uint32_t)device_id >= device_count) return 2;

    std::vector<ze_device_handle_t> devices(device_count);
    zeDeviceGet(driver, &device_count, devices.data());

    // Probe: try allocating device memory
    ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t context;
    zeContextCreate(driver, &ctx_desc, &context);

    ze_device_mem_alloc_desc_t alloc_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    void* test_ptr = nullptr;
    auto result = zeMemAllocDevice(
        context, &alloc_desc, 4096, 64, devices[device_id], &test_ptr);

    if (test_ptr) zeMemFree(context, test_ptr);
    zeContextDestroy(context);

    return (result == ZE_RESULT_SUCCESS) ? 0 : 2;
}

void* mc_xpu_fabric_malloc(ssize_t size, int device, void* queue) {
    ze_driver_handle_t driver;
    uint32_t driver_count = 1;
    zeDriverGet(&driver_count, &driver);

    uint32_t device_count = 0;
    zeDeviceGet(driver, &device_count, nullptr);
    std::vector<ze_device_handle_t> devices(device_count);
    zeDeviceGet(driver, &device_count, devices.data());

    ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t context;
    zeContextCreate(driver, &ctx_desc, &context);

    ze_device_mem_alloc_desc_t alloc_desc = {
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    alloc_desc.ordinal = 0;

    void* ptr = nullptr;
    size_t aligned_size = (size + 4095) & ~4095;  // 4K aligned
    zeMemAllocDevice(context, &alloc_desc, aligned_size, 64,
                     devices[device], &ptr);
    return ptr;
}

void mc_xpu_fabric_free(void* ptr, ssize_t size, int device,
                        void* queue) {
    ze_driver_handle_t driver;
    uint32_t driver_count = 1;
    zeDriverGet(&driver_count, &driver);

    ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t context;
    zeContextCreate(driver, &ctx_desc, &context);

    zeMemFree(context, ptr);
    zeContextDestroy(context);
}

}  // extern "C"
```

> **Note**: Production implementation should cache `ze_driver_handle_t`, `ze_context_handle_t`, and device handles in thread-local or global singletons rather than re-initializing per call.

### Phase 4: EP (Expert Parallelism) Layer

The EP layer (`mooncake-ep/`) is the most CUDA-coupled component, with:
- CUDA kernels (`.cu` / `.cuh` files) for MoE dispatch/combine
- IBGDA (InfiniBand GPU Direct Async) via MLX5 adapter
- `cudaStream_t`-based async orchestration

#### 4.1 Approach: SYCL Kernel Port

Intel XPU kernels use SYCL instead of CUDA. The dispatch/combine kernels in `mooncake_ep_api.cuh` and `mooncake_ep_launch.cuh` must be ported:

| CUDA Concept | Intel XPU Equivalent |
|---|---|
| `__global__` kernel | `sycl::handler::parallel_for` |
| `cudaStream_t` | `sycl::queue` |
| `cudaEvent_t` | `sycl::event` |
| `__shared__` memory | `sycl::local_accessor` |
| `atomicAdd` | `sycl::atomic_ref<>::fetch_add` |
| `__syncthreads()` | `sycl::group_barrier` |
| Cooperative launch | `sycl::nd_range` with work-group barriers |
| `cudaLaunchCooperativeKernel` | `queue.submit` with `nd_range` |

**New files**:
- `mooncake-ep/include/mooncake_ep_api_xpu.hpp` — SYCL dispatch/combine kernels
- `mooncake-ep/include/mooncake_ep_launch_xpu.hpp` — launch configuration
- `mooncake-ep/src/mooncake_ep_kernel_xpu.cpp` — SYCL kernel implementations

IBGDA (GPU Direct Async RDMA) is NVIDIA-specific (mlx5 DV). For Intel XPU, EP transfers fall back to **CPU-mediated RDMA** with `zeMemcpy` staging, or future Intel-native GPU-initiated RDMA if supported.

#### 4.2 EP Buffer Adaptation

**File**: `mooncake-ep/include/mooncake_ep_buffer.h`

The `MooncakeEpBuffer` class uses `ibv_mr*` and `mlx5gda_qp*` for GPU Direct RDMA. For Intel XPU:

- Replace `cudaHostAlloc` with `zeMemAllocHost` for pinned host buffers
- Replace `cudaMalloc` with `zeMemAllocDevice` for device buffers
- DMA-BUF export for RDMA registration instead of GDR

### Phase 5: PG (Process Group) Backend

#### 5.1 SYCL Reduce Kernels

**File**: `mooncake-pg/src/mooncake_worker.cu` → new `mooncake_worker_xpu.cpp`

Port CUDA reduce kernels to SYCL:

```cpp
// SYCL equivalent of reduceKernel
template <typename T>
void launchReduceKernelXpu(sycl::queue& q, T* dst, const T* src,
                           size_t count, size_t num_ranks,
                           c10d::ReduceOp::RedOpType op) {
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
        T val = dst[i];
        for (size_t r = 1; r < num_ranks; r++) {
            T other = src[r * count + i];
            switch (op) {
                case c10d::ReduceOp::SUM:     val += other; break;
                case c10d::ReduceOp::PRODUCT:  val *= other; break;
                case c10d::ReduceOp::MIN:      val = sycl::min(val, other); break;
                case c10d::ReduceOp::MAX:      val = sycl::max(val, other); break;
            }
        }
        dst[i] = val;
    });
}
```

#### 5.2 Device Detection in Worker

```cpp
// Replace cudaGetDeviceCount with Level Zero enumeration
#ifdef USE_XPU
    zeInit(0);
    uint32_t deviceCount = 0;
    ze_driver_handle_t driver;
    uint32_t driverCount = 1;
    zeDriverGet(&driverCount, &driver);
    zeDeviceGet(driver, &deviceCount, nullptr);
    if (deviceCount > 0) {
        // Allocate host-pinned buffers via zeMemAllocHost
    }
#endif
```

### Phase 6: Packaging & Wheel

#### 6.1 mooncake-wheel Updates

**File**: `mooncake-wheel/mooncake/ep.py` / `pg.py`

Add XPU backend detection:

```python
def _get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"
```

#### 6.2 Build Script

**File**: `scripts/build_wheel.sh`

Add XPU wheel variant:

```bash
if [[ "$DEVICE" == "xpu" ]]; then
    CMAKE_ARGS="-DUSE_XPU=ON -DUSE_TCP=ON -DUSE_HTTP=ON"
fi
```

---

## Slice Union Extension

**File**: `mooncake-transfer-engine/include/transport/transport.h`

Add XPU-specific fields to the `Slice` union:

```cpp
union {
    struct { /* rdma fields */ } rdma;
    struct { /* tcp fields */ } tcp;
    // ... existing fields ...

    struct {
        uint64_t dest_addr;
        ze_memory_allocation_properties_t alloc_props;
    } xpu;
};
```

---

## File Change Summary

| Action | Path | Description |
|--------|------|-------------|
| **Modify** | `mooncake-common/common.cmake` | Add `USE_XPU` option + Level Zero detection |
| **Create** | `mooncake-transfer-engine/include/gpu_vendor/xpu.h` | CUDA→Level Zero API mapping header |
| **Create** | `mooncake-transfer-engine/include/gpu_vendor/xpu.cpp` | Level Zero wrapper implementations |
| **Modify** | `mooncake-transfer-engine/include/cuda_alike.h` | Add `USE_XPU` branch |
| **Modify** | `mooncake-transfer-engine/src/memory_location.cpp` | Add `USE_XPU` to GPU detection guard |
| **Modify** | `mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp` | DMA-BUF path for XPU memory |
| **Modify** | `mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp` | Add `USE_XPU` to GPU memory guard |
| **Create** | `mooncake-transfer-engine/include/transport/xpu_transport/` | XPU P2P transport (optional) |
| **Modify** | `mooncake-transfer-engine/src/multi_transport.cpp` | Register `"xpu"` protocol |
| **Modify** | `mooncake-transfer-engine/include/transport/transport.h` | Add `xpu` to Slice union |
| **Create** | `mooncake-transfer-engine/tent/plugins/xpu/xpu_plugin.cpp` | TENT device plugin for XPU |
| **Create** | `mooncake-transfer-engine/xpu-allocator/xpu_fabric_allocator.cpp` | XPU fabric memory allocator |
| **Create** | `mooncake-integration/allocator_intel_xpu.py` | Python allocator wrapper |
| **Create** | `mooncake-ep/include/mooncake_ep_api_xpu.hpp` | SYCL EP dispatch/combine kernels |
| **Create** | `mooncake-ep/src/mooncake_ep_kernel_xpu.cpp` | SYCL kernel implementations |
| **Create** | `mooncake-pg/src/mooncake_worker_xpu.cpp` | SYCL reduce kernels |
| **Modify** | `mooncake-wheel/mooncake/ep.py` | XPU device detection |
| **Modify** | `scripts/build_wheel.sh` | XPU build variant |
| **Modify** | `dependencies.sh` | Add oneAPI/Level Zero dependency notes |
| **Create** | `mooncake-transfer-engine/tests/xpu_transport_test.cpp` | XPU transport tests |

---

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| **Intel oneAPI Base Toolkit** | 2025.0+ | SYCL compiler (`icpx`), MKL, oneDNN |
| **Level Zero Loader** | 1.17+ | `ze_loader` shared library |
| **Intel GPU driver** | `i915` or `xe` kernel module | Device access, DMA-BUF |
| **PyTorch** | 2.5+ | Native `torch.xpu` device support and XPU pluggable allocator |
| **Linux kernel** | 5.17+ (xe driver) or 5.12+ (i915 with DG2+) | DMA-BUF, P2P |
| **MLNX_OFED / rdma-core** | 5.5+ | `ibv_reg_dmabuf_mr` support |

---

## Phased Implementation Plan

### Phase 1 — Core Transport (Weeks 1–4)
- [ ] `USE_XPU` build flag + CMake Level Zero detection
- [ ] `gpu_vendor/xpu.h` mapping header + implementation
- [ ] `cuda_alike.h` + `memory_location.cpp` integration
- [ ] RDMA transport DMA-BUF registration for XPU memory
- [ ] TCP transport XPU memory staging
- [ ] Basic transfer engine test: CPU↔XPU, XPU↔XPU via RDMA
- [ ] CI environment with Intel GPU (or Level Zero software emulation)

### Phase 2 — TENT Plugin + Allocator (Weeks 3–5)
- [ ] TENT `xpu_plugin.cpp` device plugin
- [ ] `xpu_fabric_allocator.so` C++ shared library
- [ ] `allocator_intel_xpu.py` Python wrapper
- [ ] Integration tests with `mooncake-store`

### Phase 3 — PG Backend (Weeks 4–6)
- [ ] `mooncake_worker_xpu.cpp` SYCL reduce kernels
- [ ] PyTorch `c10d` backend integration with `torch.xpu`
- [ ] Distributed all-reduce / all-gather validation

### Phase 4 — EP Layer (Weeks 5–8)
- [ ] SYCL port of dispatch/combine kernels
- [ ] EP buffer management with Level Zero memory APIs
- [ ] CPU-mediated RDMA path (fallback from IBGDA)
- [ ] MoE workload benchmarking

### Phase 5 — Polish & Release (Weeks 7–10)
- [ ] Wheel packaging (`mooncake-xpu` variant)
- [ ] Documentation (getting_started, deployment guides)
- [ ] Performance benchmarking vs CUDA baseline
- [ ] CI/CD pipeline for Intel XPU

---

## Testing Strategy

### Unit Tests

| Test | Coverage |
|------|----------|
| `xpu_memory_location_test` | `xpuPointerGetAttributes` → location string mapping |
| `xpu_rdma_registration_test` | DMA-BUF export + `ibv_reg_dmabuf_mr` for XPU buffers |
| `xpu_transport_test` | End-to-end RDMA read/write with XPU source/dest |
| `xpu_tcp_staging_test` | XPU → host staging → TCP send/recv |
| `xpu_tent_plugin_test` | TENT device plugin lifecycle, alloc/free/memcpy |
| `xpu_allocator_test` | Python allocator probe + allocation |

### Integration Tests

| Test | Coverage |
|------|----------|
| `xpu_store_put_get` | mooncake-store with XPU-resident KV-cache buffers |
| `xpu_pg_allreduce` | Distributed reduce across XPU ranks |
| `xpu_ep_dispatch` | MoE expert dispatch on XPU devices |
| `xpu_multi_transport` | Simultaneous RDMA + TCP with XPU memory |

### Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| XPU↔Host bandwidth | PCIe Gen5 x16 theoretical: ~64 GB/s |
| XPU↔XPU (same node, Xe Link) | Up to 128 GB/s (GPU Max 1550) |
| XPU↔Remote XPU (RDMA) | Line rate of RDMA NIC (100/200/400 Gbps) |
| KV-cache transfer latency | Sub-millisecond for typical KV block sizes |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **DMA-BUF maturity on Intel xe driver** | RDMA registration may fail on older kernels | Pin minimum kernel version (5.17+); provide fallback to host-staging path |
| **No IBGDA equivalent for Intel** | EP layer cannot do GPU-initiated RDMA | CPU-mediated RDMA path (same pattern as Ascend `HeterogeneousRdmaTransport`) |
| **SYCL kernel performance vs CUDA** | EP/PG kernels may be slower | Profile early; leverage oneDNN primitives where possible |
| **PyTorch XPU allocator API stability** | `torch.xpu.memory.XPUPluggableAllocator` is relatively new | Pin minimum PyTorch version (2.5+); add fallback to `torch.xpu.empty()` |
| **Limited CI GPU availability** | Cannot run full test suite in CI | Use Level Zero null driver for build/unit tests; dedicated Intel GPU nodes for integration tests |
| **Xe Link availability** | P2P transport only works on GPU Max 1550 | Make XPU P2P transport optional; RDMA path works on all Intel GPUs |

---

## Alternatives Considered

### 1. HIP-SYCL Compatibility Layer (hipSYCL / AdaptiveCpp)
**Rejected**: While AdaptiveCpp can target Intel GPUs via SYCL, it adds a translation layer that reduces performance and complicates debugging. Native Level Zero integration is preferred for a production transport.

### 2. OpenCL Backend
**Rejected**: OpenCL lacks the low-level memory management (DMA-BUF export, P2P, fabric memory) needed for high-performance RDMA integration. Level Zero is the correct abstraction level.

### 3. Unified `cuda_alike.h` Approach Only (No Dedicated Transport)
**Partially adopted**: Phase 1 uses the `cuda_alike.h` mapping for memory detection and TCP staging. But a dedicated `XpuTransport` is needed for P2P and Xe Link scenarios beyond RDMA.

---

## Open Questions

1. **Xe Link topology discovery**: How should Mooncake's topology configuration (`transport_config.json`) represent Xe Link interconnects? Should we reuse the NVLink topology format or define a new one?

2. **Multi-tile support**: Intel GPU Max 1550 has 2 tiles per device. Should each tile appear as a separate device (`xpu:0`, `xpu:1`) or as sub-devices of a single device? Level Zero supports both models.

3. **PyTorch version pinning**: What is the minimum PyTorch version that guarantees a stable `torch.xpu.memory.XPUPluggableAllocator` API? Should we provide a pure Level Zero C++ fallback path for non-PyTorch use cases?

4. **EP priorities**: Given that IBGDA has no Intel equivalent, should EP Phase 4 be deferred until Intel provides GPU-initiated RDMA, or should the CPU-mediated path be prioritized?

5. **Upstream coordination**: Should this work coordinate with the PyTorch core team for `c10d` XPU backend integration, or be self-contained within Mooncake?

---

## References

- [Mooncake Architecture Paper (FAST '25)](https://arxiv.org/abs/2407.00079)
- [Intel oneAPI Level Zero Specification](https://spec.oneapi.io/level-zero/latest/index.html)
- [Level Zero Memory Management](https://spec.oneapi.io/level-zero/latest/core/api.html#memory)
- [DMA-BUF for Intel GPUs](https://www.kernel.org/doc/html/latest/gpu/drm-mm.html)
- [PyTorch XPU Backend](https://pytorch.org/docs/stable/notes/cuda.html) (`torch.xpu` native support)
- [Mooncake Ascend NPU Support (Prior Art)](mooncake-transfer-engine/include/transport/ascend_transport/)
- [Mooncake AMD HIP Support (Prior Art)](mooncake-transfer-engine/include/transport/hip_transport/)
- [TENT Device Plugin Interface](mooncake-transfer-engine/tent/include/tent/device_plugin.h)
