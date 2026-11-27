"""P6 Stage 0: cache ``mu_src`` so the attraction term ``A = mu_tar - x_s`` becomes free.

``results/p3/a/R`` stores ``R = mu_tar - mu_src`` but **not** ``mu_src``, so the source-side
prediction error ``e_src = mu_src - x_s`` -- exactly the quantity the SDS/attraction term adds back
-- was never measured.  One extra UNet forward per (image, seed) at ``t_h`` recovers it, which is
still inside the 4-NFE budget: the two forwards at ``t_h`` were always part of Ours-4, only their
source branch was thrown away.

Cached keys (fp32, identical noise to P3a so the numbers line up bit-for-bit):

``t0.9_s0..s3``  four seeds at ``t_h`` -- needed to ask whether ``A`` is *more repeatable* than ``R``
``t0.5_s0``      the seed used for the mask -- only for the "outer transition with R(0.5)" option

Outputs ``results/p6/a/musrc/{image_id}.pt``; sharded over GPUs, ``--only-missing`` resumable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml, now  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

PAIRS: List[Tuple[float, int]] = [(0.9, 0), (0.9, 1), (0.9, 2), (0.9, 3), (0.5, 0)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    indices = [i for k, i in enumerate(range(len(dataset))) if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out_dir = ROOT / base["output_root"] / "p6" / "a" / "musrc"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[musrc shard {args.shard}/{args.num_shards}] {len(indices)} images on {device}", flush=True)

    a_by_t = {t: model.ab(t)[0] for t, _ in PAIRS}
    b_by_t = {t: model.ab(t)[1] for t, _ in PAIRS}
    start = time.perf_counter()
    done = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        out_path = out_dir / f"{sample.image_id}.pt"
        if args.only_missing and out_path.exists():
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt",
                         weights_only=True).float().to(model.device)
        emb_src = model.embed([sample.src_prompt])
        eps = torch.stack([edit_ops.make_eps(sample.image_id, s, model.device, model.unet_dtype)
                           for _, s in PAIRS], dim=0)                      # [5,4,64,64]
        t_vec = torch.tensor([t for t, _ in PAIRS], device=model.device, dtype=torch.float32)
        a_vec = torch.tensor([a_by_t[t] for t, _ in PAIRS], device=model.device,
                             dtype=x_s.dtype).view(-1, 1, 1, 1)
        b_vec = torch.tensor([b_by_t[t] for t, _ in PAIRS], device=model.device,
                             dtype=x_s.dtype).view(-1, 1, 1, 1)
        z = a_vec * x_s.expand(len(PAIRS), -1, -1, -1) + b_vec * eps
        with torch.no_grad():
            mu_src = model.x0_pred(z, t_vec, emb_src.expand(len(PAIRS), -1, -1))
        bundle = {f"t{t:g}_s{s}": mu_src[k].float().cpu() for k, (t, s) in enumerate(PAIRS)}

        # cheap consistency check: mu_src + R must equal the target prediction, so |R + mu_src|
        # is the t_h target side we never stored.  We only check that nothing is degenerate.
        if not bool(torch.isfinite(bundle["t0.9_s0"]).all()):
            raise RuntimeError(f"{sample.image_id}: non-finite mu_src")
        tmp = out_path.with_suffix(".tmp")
        torch.save(bundle, tmp)
        tmp.replace(out_path)
        done += 1
        if done % 50 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / done:.2f}s/img", flush=True)

    print(f"[musrc shard {args.shard}] done={done} in {(time.perf_counter() - start) / 60:.1f} min "
          f"({now()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
