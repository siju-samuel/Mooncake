"""Intel XPU transport for Mooncake EP.

The Mooncake EP kernels are CUDA-only: they need IBGDA (mlx5 device doorbells)
for inter-node traffic and CUDA IPC / NVLink peer pointers for intra-node
traffic.  Level Zero exposes neither, so on Intel XPU the EP data path is
provided here instead.

Two transports are available, tried in order:

``ishmem``
    The DeepEP XPU port (``deep_ep``, Intel ISHMEM + SYCL kernels).  Its
    low-latency dispatch/combine kernels use the *same* masked buffer layout
    and the *same* 5-tuple handle as Mooncake EP, so they can back
    ``Buffer.dispatch`` / ``Buffer.combine`` directly.

``collective``
    ``None`` is returned and ``Buffer`` falls back to its own pure
    ``torch.distributed`` implementation, which is device-agnostic and needs no
    native extension.

Select explicitly with ``MOONCAKE_EP_XPU_TRANSPORT=ishmem|collective``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Size of one BF16 element; EP buffers are BF16 on XPU (no FP8 dispatch).
_BF16_SIZE = 2
# Per-token metadata Mooncake's BufferPair reserves (2 * sizeof(int4)).
_METADATA_BYTES = 32


def ep_buffer_size_hint(
    num_max_dispatch_tokens_per_rank: int,
    hidden: int,
    num_ranks: int,
    num_experts: int,
) -> int:
    """Symmetric-buffer byte count, mirroring Mooncake's ``BufferPair``.

    Ported from ``mooncake-ep/include/mooncake_ep_buffer.h`` so the value does
    not depend on the CUDA extension being importable:

        signal = num_experts * 4
        data   = num_experts * T * (2 * sizeof(int4) + hidden * 2)
        total  = 4 * signal + 4 * data      (2 buffers, send + recv each)
    """
    signaling = num_experts * 4
    payload = (
        num_experts
        * num_max_dispatch_tokens_per_rank
        * (_METADATA_BYTES + hidden * _BF16_SIZE)
    )
    return 4 * signaling + 4 * payload


class _StreamOrderedEvent:
    """No-op event for work already ordered on the current stream."""

    def current_stream_wait(self) -> None:
        pass

    def synchronize(self) -> None:
        torch.xpu.synchronize()


class _IshmemTransport:
    """Backs Mooncake EP dispatch/combine with the DeepEP XPU (ISHMEM) kernels.

    ``deep_ep.Buffer.low_latency_dispatch`` returns
    ``(recv_x, recv_count, handle, event, hook)`` where ``handle`` is
    ``(src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden,
    num_experts)`` — the identical layout Mooncake EP uses, so the handle
    passes through untouched.
    """

    name = "ishmem"

    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        import deep_ep

        self._deep_ep = deep_ep
        self.group = group
        self.device = device
        self.group_size = group.size()
        self.rank = group.rank()
        # deep_ep.Buffer is sized per (tokens, hidden, experts); it is created
        # lazily on the first dispatch, when those are known.
        self._buffer = None
        self._buffer_key: Optional[Tuple[int, int, int]] = None

    def _get_buffer(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ):
        key = (num_max_dispatch_tokens_per_rank, hidden, num_experts)
        if self._buffer is not None and self._buffer_key == key:
            return self._buffer
        if self._buffer is not None:
            raise RuntimeError(
                "Mooncake EP (XPU/ISHMEM) buffer was created for "
                f"{self._buffer_key} but dispatch was called with {key}. "
                "The symmetric heap cannot be resized after creation."
            )

        num_rdma_bytes = self._deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, self.group_size, num_experts
        )
        self._buffer = self._deep_ep.Buffer(
            self.group,
            0,
            num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=max(1, num_experts // self.group_size),
        )
        self._buffer_key = key
        logger.info(
            "Mooncake EP: ISHMEM transport ready (ep_size=%d, tokens=%d, "
            "hidden=%d, experts=%d, %d bytes)",
            self.group_size,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
            num_rdma_bytes,
        )
        return self._buffer

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        use_fp8: bool,
        async_finish: bool,
        return_recv_hook: bool,
    ):
        assert not use_fp8, "FP8 dispatch is not supported on Intel XPU"
        buffer = self._get_buffer(
            num_max_dispatch_tokens_per_rank, x.size(1), num_experts
        )
        recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=False,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
        )
        src_info, layout_range = handle[0], handle[1]
        return (
            recv_x,
            None,  # no FP8 scales
            recv_count,
            src_info,
            layout_range,
            self._unwrap_event(event),
            hook if return_recv_hook else (lambda: None),
        )

    def combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool,
        async_finish: bool,
        return_recv_hook: bool,
        out: Optional[torch.Tensor],
    ):
        buffer = self._get_buffer(handle[2], handle[3], handle[4])
        combined_x, event, hook = buffer.low_latency_combine(
            x,
            topk_idx,
            topk_weights,
            handle,
            zero_copy=zero_copy,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            out=out,
        )
        return (
            combined_x,
            self._unwrap_event(event),
            hook if return_recv_hook else (lambda: None),
        )

    @staticmethod
    def _unwrap_event(event: Any) -> Any:
        """Return an object with ``current_stream_wait()``.

        ``deep_ep`` wraps its native handle in its own ``EventOverlap``, and
        leaves the inner handle ``None`` when ``async_finish=False`` (the work
        is already ordered on the current stream).  Mooncake's ``Buffer`` wraps
        whatever we return in *its* ``EventOverlap``, whose
        ``current_stream_wait()`` asserts the inner event is not None — so
        substitute a no-op waiter in that case.
        """
        inner = getattr(event, "event", event)
        return inner if inner is not None else _StreamOrderedEvent()

    def get_next_combine_buffer(self, handle: tuple) -> torch.Tensor:
        buffer = self._get_buffer(handle[2], handle[3], handle[4])
        return buffer.get_next_low_latency_combine_buffer(handle)

    def update_ep_member(self) -> None:
        # Membership changes are re-read from active_ranks per dispatch; the
        # ISHMEM symmetric heap itself is not rebuilt.
        pass


class _CollectiveTransport:
    """Vectorised all-to-all EP over plain ``torch.distributed`` collectives.

    Same masked layout and handle as the ISHMEM path, but needs no launcher and
    no native extension — it works wherever the process group works.

    Why not ``Buffer._fallback_dispatch``: that reference implementation loops
    in Python over every local expert and every peer rank, calling ``.item()``
    and allocating small tensors inside the loop. With 64 local experts that is
    thousands of tiny device allocations plus host syncs per MoE layer, which
    exhausts Level Zero resources on XPU (UR_RESULT_ERROR_OUT_OF_RESOURCES).
    Here the whole permutation is computed with vectorised index math instead:
    a fixed number of collectives and kernels per call, no host syncs.
    """

    name = "collective"

    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        self.group = group
        self.device = device
        self.group_size = group.size()
        self.rank = group.rank()
        self._next_combine_buffer: Optional[torch.Tensor] = None

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        use_fp8: bool,
        async_finish: bool,
        return_recv_hook: bool,
    ):
        assert not use_fp8, "FP8 dispatch is not supported on Intel XPU"
        num_tokens, hidden = x.shape
        num_ranks = self.group_size
        num_local = num_experts // num_ranks
        capacity = num_ranks * num_max_dispatch_tokens_per_rank

        # Every rank contributes the same token count in sglang's masked path,
        # so gather straight into one tensor (no per-rank size exchange).
        all_x = torch.empty((num_ranks, num_tokens, hidden), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(all_x, x.contiguous(), group=self.group)
        all_topk = torch.empty(
            (num_ranks, num_tokens, topk_idx.size(1)),
            dtype=topk_idx.dtype,
            device=x.device,
        )
        dist.all_gather_into_tensor(all_topk, topk_idx.contiguous(), group=self.group)

        # Tokens routed to each of *this* rank's experts, ordered by source rank
        # then token index — the order the combine kernel's layout_range assumes.
        first_expert = self.rank * num_local
        # routed[e, r, t] = token t of rank r goes to local expert e
        routed = (
            all_topk.unsqueeze(0) == (first_expert + torch.arange(
                num_local, device=x.device, dtype=topk_idx.dtype
            )).view(num_local, 1, 1, 1)
        ).any(-1)

        flat = routed.reshape(num_local, num_ranks * num_tokens)
        # Rank-major running position of each selected token within its expert.
        slot = flat.cumsum(dim=1) - 1
        keep = flat & (slot < capacity)

        recv_count = keep.sum(dim=1).to(torch.int32)
        e_idx, src_flat = keep.nonzero(as_tuple=True)
        dst = slot[e_idx, src_flat]

        recv_x = torch.zeros(
            (num_local, capacity, hidden), dtype=x.dtype, device=x.device
        )
        flat_x = all_x.reshape(num_ranks * num_tokens, hidden)
        recv_x[e_idx, dst] = flat_x[src_flat]

        # src_info holds the *token index within its source rank*.
        src_info = torch.full(
            (num_local, capacity), -1, dtype=torch.int32, device=x.device
        )
        src_info[e_idx, dst] = (src_flat % num_tokens).to(torch.int32)

        # layout_range[e, r] = (begin << 32) | count, begin/count over recv_x.
        per_rank = keep.reshape(num_local, num_ranks, num_tokens).sum(dim=2)
        begin = per_rank.cumsum(dim=1) - per_rank
        layout_range = ((begin.to(torch.int64) << 32) | per_rank.to(torch.int64)) * (
            per_rank > 0
        )

        # Allocated on demand by get_next_combine_buffer(): it is only used by
        # the zero-copy combine path, and eagerly reserving a second
        # capacity-sized buffer here costs as much as recv_x itself.
        return (
            recv_x,
            None,  # no FP8 scales
            recv_count,
            src_info,
            layout_range,
            _StreamOrderedEvent(),
            lambda: None,
        )

    def combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        zero_copy: bool,
        async_finish: bool,
        return_recv_hook: bool,
        out: Optional[torch.Tensor],
    ):
        src_info, layout_range, _, hidden, num_experts = handle
        num_ranks = self.group_size
        num_local = num_experts // num_ranks
        num_tokens = topk_idx.size(0)
        if zero_copy:
            expert_out = self.get_next_combine_buffer(handle)
        else:
            expert_out = x
        if expert_out.dtype != torch.bfloat16:
            expert_out = expert_out.to(torch.bfloat16)

        all_topk = torch.empty(
            (num_ranks, num_tokens, topk_idx.size(1)),
            dtype=topk_idx.dtype,
            device=topk_idx.device,
        )
        dist.all_gather_into_tensor(all_topk, topk_idx.contiguous(), group=self.group)
        all_w = torch.empty(
            (num_ranks, num_tokens, topk_weights.size(1)),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        dist.all_gather_into_tensor(
            all_w, topk_weights.contiguous(), group=self.group
        )

        # Rebuild (expert, slot) -> (src_rank, token) from the handle. Only the
        # first counts[e, r] slots of each (expert, src_rank) span hold real
        # tokens; the rest of the capacity-sized buffer is padding. Work on the
        # occupied slots alone -- gathering the whole buffer would allocate
        # hundreds of MB of transients per layer and exhaust Level Zero.
        counts = (layout_range & 0xFFFFFFFF).to(torch.int64)  # [num_local, num_ranks]
        begins = (layout_range >> 32) & 0xFFFFFFFF
        capacity = src_info.size(1)
        max_slot = int(counts.max().item()) if counts.numel() else 0
        if max_slot == 0:
            combined = torch.zeros(
                (num_tokens, hidden), dtype=torch.bfloat16, device=x.device
            )
            if out is not None:
                out.copy_(combined)
                combined = out
            return combined, _StreamOrderedEvent(), (lambda: None)

        # [num_local, num_ranks, max_slot] view of the occupied slots.
        off = torch.arange(max_slot, device=src_info.device)
        keep = off.view(1, 1, -1) < counts.unsqueeze(-1)
        slot = (begins.unsqueeze(-1) + off.view(1, 1, -1)).clamp_(max=capacity - 1)

        e_idx, r_sel, k_idx = keep.nonzero(as_tuple=True)
        s_idx = slot[e_idx, r_sel, k_idx]
        t_sel = src_info[e_idx, s_idx].to(torch.int64)
        # Guard against a stale/padded src_info entry indexing out of range.
        ok = (t_sel >= 0) & (t_sel < num_tokens)
        if not bool(ok.all()):
            e_idx, r_sel, s_idx, t_sel = (
                e_idx[ok],
                r_sel[ok],
                s_idx[ok],
                t_sel[ok],
            )
        expert_id = self.rank * num_local + e_idx

        # This expert's routing weight for that token (0 if not routed there).
        w = (
            all_w[r_sel, t_sel] * (all_topk[r_sel, t_sel] == expert_id.unsqueeze(1))
        ).sum(dim=1, keepdim=True)

        send = torch.zeros(
            (num_ranks, num_tokens, hidden), dtype=torch.bfloat16, device=x.device
        )
        # Accumulate in bf16 to halve the transient footprint; each destination
        # token sums at most num_topk contributions, so the error stays within
        # the bf16 rounding the rest of this path already accepts.
        send.reshape(num_ranks * num_tokens, hidden).index_add_(
            0,
            r_sel * num_tokens + t_sel,
            expert_out[e_idx, s_idx] * w.to(torch.bfloat16),
        )
        dist.all_reduce(send, group=self.group)
        combined = send[self.rank]

        if out is not None:
            out.copy_(combined)
            combined = out
        return combined, _StreamOrderedEvent(), (lambda: None)

    def get_next_combine_buffer(self, handle: tuple) -> torch.Tensor:
        num_experts, hidden = handle[4], handle[3]
        num_local = num_experts // self.group_size
        capacity = self.group_size * handle[2]
        shape = (num_local, capacity, hidden)
        if (
            self._next_combine_buffer is None
            or tuple(self._next_combine_buffer.shape) != shape
        ):
            self._next_combine_buffer = torch.empty(
                shape, dtype=torch.bfloat16, device=self.device
            )
        return self._next_combine_buffer

    def update_ep_member(self) -> None:
        pass


def _transport_choice() -> str:
    return os.getenv("MOONCAKE_EP_XPU_TRANSPORT", "auto").strip().lower()


def _ishmem_device_pinning_ok() -> bool:
    """True if this process sees exactly one GPU, as ISHMEM requires.

    ISHMEM (verified on 8x Arc Pro B60) works fine without an MPI launcher — it
    bootstraps from a unique id exchanged over the c10d Store — but every rank
    must own one device. ``ZE_AFFINITY_MASK`` must be set *before* the Level Zero
    driver initialises, i.e. before the first XPU call in the process, so it
    cannot be fixed up from here.
    """
    mask = os.getenv("ZE_AFFINITY_MASK", "")
    return bool(mask) and len([p for p in mask.split(",") if p.strip()]) == 1


def make_xpu_backend(
    group: dist.ProcessGroup,
    num_ep_buffer_bytes: int,
    device: torch.device,
) -> Optional[Any]:
    """Build the XPU EP transport, or ``None`` to use the collective fallback."""
    choice = _transport_choice()
    if choice in {"collective", "fallback"}:
        logger.info("Mooncake EP: using vectorised collective transport (XPU)")
        return _CollectiveTransport(group, device)
    if choice == "reference":
        # Buffer's own pure-collective implementation. Correct but allocation-
        # heavy: it can exhaust Level Zero resources with many local experts.
        logger.info("Mooncake EP: using Buffer reference collective fallback (XPU)")
        return None
    if choice not in {"auto", "ishmem", "deep_ep", "deepep"}:
        raise ValueError(
            f"Unknown MOONCAKE_EP_XPU_TRANSPORT={choice!r}; expected one of: "
            "auto, ishmem, collective, reference"
        )

    if choice == "auto" and not _ishmem_device_pinning_ok():
        # ISHMEM resolves peer pointers per PE and needs one GPU per rank. Under
        # mpirun it sets ZE_AFFINITY_MASK itself; with a spawn launcher nothing
        # does, and several ranks then share a device and deadlock in dispatch.
        # Only auto-select ISHMEM once the mask is in place.
        logger.info(
            "Mooncake EP: ZE_AFFINITY_MASK not set to a single device, so the "
            "ISHMEM transport is unsafe here; using the vectorised collective "
            "transport. Pin one GPU per rank (ZE_AFFINITY_MASK=<local_rank>) "
            "before process start, or force MOONCAKE_EP_XPU_TRANSPORT=ishmem."
        )
        return _CollectiveTransport(group, device)

    try:
        return _IshmemTransport(group, device)
    except Exception as exc:  # pragma: no cover - depends on local install
        if choice != "auto":
            raise
        logger.warning(
            "Mooncake EP: ISHMEM transport unavailable (%s); "
            "using vectorised collective transport",
            exc,
        )
        return _CollectiveTransport(group, device)
