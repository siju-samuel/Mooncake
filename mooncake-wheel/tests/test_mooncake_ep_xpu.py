"""Verify mooncake.mooncake_ep_buffer.Buffer works on Intel XPU.

Exercises the exact API sglang's MooncakeEPDispatcher calls, for both
transports (ISHMEM kernels and the torch.distributed collective fallback),
and checks combine output against a CPU reference.
"""

import os
import sys

_use_mpi = "LOCAL_RANK" not in os.environ
if _use_mpi:
    from mpi4py import MPI

os.environ["TORCH_SYMMMEM"] = "XPU"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.environ.get("MC_PORT", "29601"))

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

if _use_mpi:
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
else:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world)

dev = int(os.environ.get("MPI_LOCALRANKID", rank))
if os.environ.get("ZE_AFFINITY_MASK"):
    dev = 0
torch.xpu.set_device(dev)
dist.init_process_group(backend="xccl", rank=rank, world_size=world)

from mooncake.mooncake_ep_buffer import Buffer

NUM_EXPERTS = int(os.environ.get("MC_EXPERTS", "16"))
HIDDEN = int(os.environ.get("MC_HIDDEN", "512"))
TOPK = int(os.environ.get("MC_TOPK", "2"))
NTOK = int(os.environ.get("MC_NTOK", "16"))
MAXTOK = int(os.environ.get("MC_MAXTOK", "64"))
device = torch.device("xpu", dev)


def log(*a):
    if rank == 0:
        print(*a, flush=True)


def run(transport: str) -> bool:
    os.environ["MOONCAKE_EP_XPU_TRANSPORT"] = transport
    log(f"\n===== transport={transport} =====")

    nbytes = Buffer.get_ep_buffer_size_hint(MAXTOK, HIDDEN, world, NUM_EXPERTS)
    log(f"get_ep_buffer_size_hint -> {nbytes}")
    buf = Buffer(dist.group.WORLD, nbytes)
    actual = "collective" if buf._xpu_backend is None else buf._xpu_backend.name
    log(f"Buffer created; transport={actual}, use_fallback={buf._use_fallback}")

    torch.manual_seed(1234 + rank)
    x = torch.randn((NTOK, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_idx = torch.stack(
        [torch.randperm(NUM_EXPERTS, device=device)[:TOPK] for _ in range(NTOK)]
    ).to(torch.int64)
    topk_w = torch.rand((NTOK, TOPK), dtype=torch.float32, device=device)
    active_ranks = torch.ones((world,), dtype=torch.int32, device=device)

    # Exactly the call sglang's _dispatch_core makes (use_fp8=True -> BF16 on XPU).
    recv_x, recv_cnt, handle, event, hook = buf.dispatch(
        x,
        topk_idx,
        active_ranks,
        MAXTOK,
        NUM_EXPERTS,
        -1,
        use_fp8=True,
        async_finish=False,
        return_recv_hook=False,
    )
    event.current_stream_wait()
    assert not isinstance(recv_x, tuple), "XPU must return BF16, not an FP8 tuple"
    assert recv_x.dtype == torch.bfloat16, recv_x.dtype
    n_local = NUM_EXPERTS // world
    assert recv_x.shape[0] == n_local, (recv_x.shape, n_local)
    assert recv_cnt.shape == (n_local,), recv_cnt.shape
    assert len(handle) == 5, handle
    log(f"dispatch OK recv_x={tuple(recv_x.shape)} {recv_x.dtype} cnt={recv_cnt.tolist()}")

    # Expert compute: identity, so combine must reproduce sum(w) * x per token.
    combined, event2, hook2 = buf.combine(
        recv_x,
        topk_idx,
        topk_w,
        active_ranks,
        -1,
        handle,
        async_finish=False,
        return_recv_hook=False,
    )
    event2.current_stream_wait()
    assert combined.shape == (NTOK, HIDDEN), combined.shape
    assert torch.isfinite(combined.float()).all(), "combine produced non-finite values"

    expect = (x.float() * topk_w.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    diff = (combined.float() - expect.float()).abs()
    rel = (diff.sum() / expect.float().abs().sum().clamp(min=1e-6)).item()
    ok = rel < 0.02
    log(f"combine OK shape={tuple(combined.shape)} rel_err={rel:.5f} -> "
        f"{'PASS' if ok else 'FAIL'}")

    # Idempotence / reuse: a second round must also work.
    recv_x2, recv_cnt2, handle2, ev3, _ = buf.dispatch(
        x, topk_idx, active_ranks, MAXTOK, NUM_EXPERTS, -1,
        use_fp8=True, async_finish=False, return_recv_hook=False,
    )
    ev3.current_stream_wait()
    combined2, ev4, _ = buf.combine(
        recv_x2, topk_idx, topk_w, active_ranks, -1, handle2,
        async_finish=False, return_recv_hook=False,
    )
    ev4.current_stream_wait()
    same = torch.equal(combined, combined2)
    log(f"second round OK, deterministic={same}")

    buf.update_ep_member()
    log("update_ep_member OK")
    return ok


results = {}
for t in os.environ.get("MC_TRANSPORTS", "ishmem,collective").split(","):
    t = t.strip()
    if not t:
        continue
    try:
        results[t] = run(t)
    except Exception as exc:
        import traceback
        if rank == 0:
            traceback.print_exc()
        results[t] = False
    dist.barrier()

if rank == 0:
    print("\n===== SUMMARY =====", flush=True)
    for t, ok in results.items():
        print(f"  {t:12s} {'PASS' if ok else 'FAIL'}", flush=True)
    print("ALL_PASS" if results and all(results.values()) else "SOME_FAILED", flush=True)

dist.barrier()
os._exit(0 if results and all(results.values()) else 1)
