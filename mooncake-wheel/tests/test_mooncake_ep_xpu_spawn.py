"""Mooncake EP over ISHMEM with NO MPI launcher (torch.multiprocessing spawn).

This is the launcher model sglang uses for its TP workers, so it demonstrates
that the fast ISHMEM transport is usable in-server: ISHMEM bootstraps from a
unique id exchanged over the c10d Store, no mpirun required.

The one hard requirement is that each rank owns a single GPU. ISHMEM resolves
peer pointers per PE, and if several ranks share a device dispatch deadlocks.
Under mpirun ISHMEM sets ZE_AFFINITY_MASK itself; here the worker sets it before
its first XPU call (Level Zero reads the mask at driver init).

Run: python test_mooncake_ep_xpu_spawn.py    (MC_WORLD=2 by default)
"""

import os
import sys

os.environ.setdefault("TORCH_SYMMMEM", "XPU")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.environ.get("MC_PORT", "30101"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = int(os.environ.get("MC_WORLD", "2"))
NUM_EXPERTS = 16
HIDDEN = 512
TOPK = 2
NTOK = 16
MAXTOK = 64


def worker(rank: int, world: int, q):
    stage = "start"
    try:
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        os.environ["LOCAL_RANK"] = str(rank)
        if os.environ.get("MC_PIN_DEVICE", "1") == "1":
            os.environ["ZE_AFFINITY_MASK"] = str(rank)
            dev = 0
        else:
            dev = rank
        torch.xpu.set_device(dev)
        device = torch.device("xpu", dev)

        stage = "init_process_group"
        dist.init_process_group(
            backend="xccl", rank=rank, world_size=world, device_id=device
        )
        print(f"[r{rank}] STAGE ok: {stage}", flush=True)

        stage = "import deep_ep"
        import deep_ep

        print(f"[r{rank}] STAGE ok: {stage}", flush=True)

        stage = "Buffer ctor (ishmem init + symmetric alloc + barrier)"
        nbytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            MAXTOK, HIDDEN, world, NUM_EXPERTS
        )
        buf = deep_ep.Buffer(
            dist.group.WORLD, 0, nbytes, low_latency_mode=True,
            num_qps_per_rank=max(1, NUM_EXPERTS // world),
        )
        print(f"[r{rank}] STAGE ok: {stage}", flush=True)

        stage = "torch.distributed barrier after ctor"
        dist.barrier()
        print(f"[r{rank}] STAGE ok: {stage}", flush=True)

        x = torch.randn((NTOK, HIDDEN), dtype=torch.bfloat16, device=device)
        topk_idx = torch.randint(
            0, NUM_EXPERTS, (NTOK, TOPK), dtype=torch.int64, device=device
        )

        stage = "low_latency_dispatch"
        recv_x, recv_cnt, handle, ev, _ = buf.low_latency_dispatch(
            x, topk_idx, MAXTOK, NUM_EXPERTS, use_fp8=False
        )
        torch.xpu.synchronize()
        print(f"[r{rank}] STAGE ok: {stage} cnt={recv_cnt.tolist()}", flush=True)

        stage = "low_latency_combine"
        tw = torch.ones((NTOK, TOPK), dtype=torch.float32, device=device)
        out, _, _ = buf.low_latency_combine(recv_x, topk_idx, tw, handle)
        torch.xpu.synchronize()
        print(f"[r{rank}] STAGE ok: {stage} out={tuple(out.shape)}", flush=True)

        q.put((rank, True, "all stages", ""))
    except Exception as exc:
        import traceback

        q.put((rank, False, stage, traceback.format_exc()[-1200:]))
    finally:
        # mp.Queue hands off to a background feeder thread, so the item is not
        # on the pipe yet. os._exit() skips that flush and the parent would time
        # out despite every stage passing — close() waits for the feeder.
        # _exit (not sys.exit) is still required: ISHMEM hangs in its atexit
        # finalize, which would wedge the process after a clean run.
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
        os._exit(0)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, WORLD, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    results = []
    for _ in range(WORLD):
        try:
            results.append(q.get(timeout=120))
        except Exception:
            results.append((-1, False, "TIMEOUT (see last STAGE ok above)", ""))
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    ok = all(r[1] for r in results) and len(results) == WORLD
    for rank, good, stage, err in results:
        print(f"rank {rank}: {'PASS' if good else 'FAIL'} last_stage={stage}")
        if err:
            print(err)
    print(f"STAGE_TEST_{'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)
