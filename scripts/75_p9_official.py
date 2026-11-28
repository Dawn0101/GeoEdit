"""P9 step 2: run our mask + SDS on top of the **official ChordEdit pipeline**.

The base is ``geoedit.chord.ChordEditor`` (a thin wrapper over
``third_party/ChordEdit/pipeline_chord.py``), so the transport field ``u`` and the ``t_c=0.3`` prox
are the official ones, with ChordEdit's own noise draw (``seed=42``, CUDA RNG, identical for every
image).  We only add two things *in latent space*:

* ``M``: a quantile-saturated mask from the transport field's low-noise residual.  Primary is
  ``q60/q90`` on ``||R(0.75)||`` (background 60% of pixels exactly zero, per the P9 spec); the
  alternative ``M`` from ``||R(0.5)||`` (P5's L5, sigma=2) is run as a separate variant.
* the core-confined attraction ``k * M^2 * A(0.75)`` with ``A = mu_tar(0.75) - x_s``, combined as the
  convex mix ``delta = M*(u + k*M^2*A)/(1 + k*M^2)`` so that ``M=1`` gives the official displacement.

Variants: ``off_k0`` (M=1, k=0 -> must equal B1_chord), ``m075_k0``, ``m075_k05``, ``m075_k10``,
``m05_k0`` (mask from t=0.5).  The composite uses the *un-squared* M.

Outputs ``results/p9/frontier/per_image_shard{i}.csv`` and (``--mode dump``) images for the A/B.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.chord import ChordEditor  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import attention_term  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.masks import gaussian_blur  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402

FIELDS = ["image_id", "category", "variant", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist", "struct_dist_unedit", "clip_target_edit", "mask_area", "M_out_gt",
          "delta_norm"]
SCORE_KEYS = FIELDS[4:10]


def qmask(E: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    ql = torch.quantile(E.flatten(), lo)
    qh = torch.quantile(E.flatten(), hi)
    return ((E - ql) / (qh - ql + 1e-8)).clamp(0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["frontier", "dump"], default="frontier")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default="off_k0,m075_k0,m075_k05,m075_k10,m05_k0")
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    variants = args.variants.split(",")
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = list(range(len(dataset)))
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, 200).round().astype(int).tolist()]
    indices = [i for k, i in enumerate(eval_idx) if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    a_root = ROOT / base["output_root"] / "p3" / "a"
    t75_root = ROOT / base["output_root"] / "p8" / "a" / "t75"
    out_dir = ROOT / base["output_root"] / "p9"
    (out_dir / "frontier").mkdir(parents=True, exist_ok=True)
    img_root = ROOT / "cache" / "images" / "p9"
    ed = ChordEditor(device=args.device)
    metrics = PieMetrics(device=args.device)
    print(f"[p9 {args.mode} shard {args.shard}/{args.num_shards}] {len(indices)} images, "
          f"variants {variants} on {args.device}", flush=True)

    if args.mode == "dump":
        if args.manifest:
            pd.read_csv(args.manifest)  # only used for side-effect-free bookkeeping
        for position, index in enumerate(indices):
            sample = dataset[index]
            outs = run_image(ed, sample, a_root, t75_root, variants)
            for name, (lat, _info) in outs.items():
                d = img_root / name
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{sample.image_id}.png"
                if not p.exists():
                    from PIL import Image
                    Image.fromarray(ed.decode(lat)[0]).save(p)
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(indices)}", flush=True)
        print(f"[p9 dump] done ({now()})")
        return 0

    csv_path = out_dir / "frontier" / (f"per_image_shard{args.shard}.csv" if args.num_shards > 1
                                       else "per_image.csv")
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
        done = set(zip(prev["image_id"], prev["variant"]))
    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        todo = [v for v in variants if (sample.image_id, v) not in done]
        if not todo:
            continue
        outs = run_image(ed, sample, a_root, t75_root, todo)
        names = list(outs.keys())
        decoded = [ed.decode(outs[n][0])[0] for n in names]
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=4)
        rows = [{"image_id": sample.image_id, "category": sample.category, "variant": n, "nfe": 5,
                 **{k: s[k] for k in SCORE_KEYS},
                 "mask_area": outs[n][1].get("mask_area", 1.0),
                 "M_out_gt": outs[n][1].get("M_out_gt", 0.0),
                 "delta_norm": outs[n][1].get("delta_norm", 0.0)}
                for n, s in zip(names, scored)]
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img eta "
                  f"{(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[p9 frontier shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


@torch.no_grad()
def run_image(ed, sample, a_root, t75_root, variants):
    """Official transport + optional mask/SDS composition + official prox, for each variant."""
    image = sample.image()
    tr = ed.transport(image, sample.src_prompt, sample.tar_prompt)
    x_s = tr["x_src"].float()
    y = tr["y"].float()
    noise = tr["noise"].float()
    cond = ed.encode_prompt(sample.tar_prompt)
    u = y - x_s                                   # official displacement (step_scale = 1.0)
    d75 = torch.load(t75_root / f"{sample.image_id}.pt", weights_only=True)
    R75 = (d75["R75"][0] if d75["R75"].dim() == 4 else d75["R75"]).float().to(x_s.device)
    mu75 = (d75["mu_src75"][0] if d75["mu_src75"].dim() == 4 else d75["mu_src75"]).float().to(x_s.device)
    A75 = attention_term(R75.unsqueeze(0), mu75.unsqueeze(0), x_s)[0]
    R5 = torch.load(a_root / "R" / f"{sample.image_id}.pt",
                    weights_only=True)["t0.5_s0"].float().to(x_s.device)
    R5 = R5[0] if R5.dim() == 4 else R5
    gt = torch.from_numpy(sample.mask64 > 0.5)
    E75 = gaussian_blur(R75.norm(dim=0), 2.0)
    E5 = gaussian_blur(R5.norm(dim=0), 2.0)

    out = {}
    for name in variants:
        if name.startswith("off"):
            M = torch.ones_like(E75)
            k, src_mask = 0.0, E75
        elif name.startswith("m075"):
            src_mask, k = E75, (0.5 if "k05" in name else 1.0 if "k10" in name else 0.0)
            M = qmask(src_mask, 0.60, 0.90)
        elif name.startswith("m05"):
            src_mask, k = E5, 0.0
            M = qmask(src_mask, 0.60, 0.90)
        else:
            raise ValueError(name)
        denom = 1.0 + k * (M ** 2)
        delta = M * (u[0] + k * (M ** 2) * A75) / denom
        y_new = (x_s + delta.unsqueeze(0)).to(y.dtype)
        prox = ed.prox(y_new, cond, noise)
        if name.startswith("off"):
            lat = prox
        else:
            lat = (1.0 - M) * x_s + M * prox
        info = {"mask_area": float((M > 0.5).float().mean()),
                "M_out_gt": float(M[~gt].mean()) if bool((~gt).any()) else 0.0,
                "delta_norm": float(delta.norm())}
        out[name] = (lat, info)
    return out


if __name__ == "__main__":
    sys.exit(main())
