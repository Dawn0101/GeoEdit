"""Benchmark the partial-OT solver on realistic grids (GPU latency / memory).

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/bench_uot.py
"""

from __future__ import annotations

import argparse
import time

import torch

from geoedit.uot import estimate_sigma, window_radius


def bench(h, w, d, tau, s_p, eps, iters, tol, device, dtype, repeats=5):
    n = h * w
    gen = torch.Generator(device=device).manual_seed(0)
    src = torch.randn(n, d, generator=gen, device=device, dtype=dtype)
    tgt = src + 0.35 * torch.randn(n, d, generator=gen, device=device, dtype=dtype)
    radius = window_radius(tau, s_p)

    torch.cuda.synchronize(device) if device.type == "cuda" else None
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    estimate_sigma(src, tgt, h, w, tau=tau, s_p=s_p, eps=eps, iters=iters, tol=tol, radius=radius)

    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        res = estimate_sigma(
            src, tgt, h, w, tau=tau, s_p=s_p, eps=eps, iters=iters, tol=tol, radius=radius
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    print(
        f"{h:>3}x{w:<3} D={d:<4} tau={tau:<5g} s_p={s_p:<4g} eps={eps:<5g} "
        f"R={radius:<2d} K={(2 * radius + 1) ** 2:<4d} "
        f"iters={res.iters:<4d} conv={str(res.converged):<5s} "
        f"median={sorted(times)[len(times) // 2] * 1e3:7.1f} ms  "
        f"edit_frac={res.edit_fraction:.3f}  peak={peak:6.0f} MiB"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}\n")

    # Default E2 configuration family (64x64 grid, the latent resolution).
    for tau, s_p in ((1.0, 2.0), (4.0, 2.0), (8.0, 2.0), (16.0, 2.0)):
        bench(64, 64, 36, tau, s_p, 0.1, 300, 1e-5, device, dtype, args.repeats)
    print()
    for tau, s_p in ((1.0, 2.0), (4.0, 2.0), (16.0, 2.0)):
        bench(64, 64, 804, tau, s_p, 0.1, 300, 1e-5, device, dtype, args.repeats)
    print()
    bench(64, 64, 36, 4.0, 2.0, 0.1, 300, 1e-5, device, dtype, args.repeats)
    for tau in (1.0, 4.0):
        bench(32, 32, 768, tau, 2.0, 0.1, 300, 1e-5, device, dtype, args.repeats)
    print()
    # Pointwise ablation R=0 must be essentially free.
    bench(64, 64, 36, 4.0, 2.0, 0.1, 300, 1e-5, device, dtype, args.repeats)


if __name__ == "__main__":
    main()
