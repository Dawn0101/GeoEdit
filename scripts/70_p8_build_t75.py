"""P8: build the t=0.75 branch (the timestep ChordEdit's chord field is 86% weighted on).

`results/p3/a/R` has t in {0.9, 0.7, 0.5} only, so the chord field
``u = (delta*R(t_s) + t_s*R(t_s-delta)) / (t_s + delta) = 0.143*R(0.9) + 0.857*R(0.75)``
cannot be assembled.  Two UNet forwards per image (source + target at t=0.75) fix that; the mask
is free (a Gaussian on the single-draw ``||R(0.75)||``).

Saves ``results/p8/a/t75/{image_id}.pt`` with ``R75`` and ``mu_src75`` (fp32), plus the same for
``t=0.9`` in the P1 cache already.  200 images x 2 forwards ~ 3 minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import load_yaml, now  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

T75 = 0.75


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="all 700 images instead of the eval set")
    parser.add_argument("--only-missing", action="store_true")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    if args.all:
        indices = list(range(len(dataset)))
    else:
        import numpy as np
        pool = list(range(len(dataset)))
        indices = [pool[i] for i in np.linspace(0, len(pool) - 1, args.n).round().astype(int).tolist()]

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out_dir = ROOT / base["output_root"] / "p8" / "a" / "t75"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[t75] {len(indices)} images on {args.device}", flush=True)

    start = time.perf_counter()
    done = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        path = out_dir / f"{sample.image_id}.pt"
        if args.only_missing and path.exists():
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt",
                         weights_only=True).float().to(model.device)
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        a, b = model.ab(T75)
        z = float(a) * x_s + float(b) * eps
        with torch.no_grad():
            mu_src = model.x0_pred(z, T75, emb_src)
            mu_tar = model.x0_pred(z, T75, emb_tar)
        bundle = {"R75": (mu_tar - mu_src).float().cpu(), "mu_src75": mu_src.float().cpu(),
                  "mu_tar75": mu_tar.float().cpu()}
        tmp = path.with_suffix(".tmp")
        torch.save(bundle, tmp)
        tmp.replace(path)
        done += 1
        if done % 50 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / done:.3f}s/img", flush=True)
    print(f"[t75] done={done} in {(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
