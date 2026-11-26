"""P3d-D1: average vs pick vs hybrid, judged on a **sensitive** semantic metric (CLIP-D).

Handoff: ``docs/handoff/v3_p3d.md`` §2.  No new UNet forwards: every latent is built from the
fp32 residuals cached by ``30_p3a_build_and_eval.py --prep`` (4 seeds at t=0.9 and at t=0.5).

Variants (``lam`` grid from ``configs/p3.yaml``, NFE in the row):

===========  ========================================================  ====
``V0``       ``R0`` (single seed)                                        2
``V0s1..3``  the other three seeds (basis for the oracle and S3)        2
``V1``       ``mean_4(R9)``                                             8
``V1x``      ``mean_8(R9, R5)`` -- same NFE as the 16-NFE variants      16
``V12``      ``M5 * R0``                                                10
``S3_5``     ``R_i*``, ``i* = argmin_i ||(1-M5) * R9_i||^2``            16
``S3_9``     same with ``M9`` (no extra forwards)                       8
``H1_5``     ``M5*R_i* + (1-M5)*mean_4``                                16
``H1_9``     same with ``M9``                                           8
``H0_5``     same but ``i* = 0`` (separates "pick" from "spatial split") 16
===========  ========================================================  ====

``M`` is the continuous DiffEdit-style mask (N=4 mean ``||R||``, divided by its max).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "variant", "lam", "nfe", "psnr", "lpips", "ssim",
          "clip_target_whole", "clip_target_edit", "clip_d", "delta_norm_over_lam"]
NFE = {"V0": 2, "V0s1": 2, "V0s2": 2, "V0s3": 2, "V1": 8, "V1x": 16, "V12": 10,
       "S3_5": 16, "S3_9": 8, "H1_5": 16, "H1_9": 8, "H0_5": 16}


def apply(w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return w.unsqueeze(0).unsqueeze(0) * r


def diffedit_cont(Rs: torch.Tensor) -> torch.Tensor:
    energy = Rs.norm(dim=1).mean(dim=0)
    return energy / energy.max().clamp_min(1e-12)


def clip_d(feats, src_feat, tar_text, src_text):
    di = feats - src_feat
    dt = (tar_text - src_text).expand_as(di)
    return (di / di.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            * dt / dt.norm(dim=-1, keepdim=True).clamp_min(1e-8)).sum(dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--only-missing", action="store_true",
                        help="process the images absent from the union of all shard CSVs")
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"
    lam_list = [float(v) for v in p3["lam_list"]]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    if args.only_missing:
        import csv
        seen = set()
        for path in sorted((ROOT / base["output_root"] / "p3d" / "d1" / "shards").glob("shard*.csv")):
            with path.open() as fh:
                seen |= {row["image_id"] for row in csv.DictReader(fh)}
        indices = [i for i in range(len(dataset)) if dataset.image_ids[i] not in seen]
        indices = [i for k, i in enumerate(indices) if k % num_shards == args.shard]
        print(f"[d1 top-up] {len(indices)} missing images on this shard", flush=True)
    else:
        indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"], vae_only=True)
    metrics = PieMetrics(device=device)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out = ROOT / base["output_root"] / "p3d" / "d1" / "shards"
    csv_path = out / f"shard{args.shard}.csv"
    done = set()
    if csv_path.exists():
        import csv
        with csv_path.open() as fh:
            done = {row["image_id"] for row in csv.DictReader(fh)}
    print(f"[d1 shard {args.shard}/{num_shards}] {len(indices)} images on {device}"
          f"{f', resuming ({len(done)})' if done else ''}", flush=True)

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        if sample.image_id in done:
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
        b = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R9 = torch.stack([b[f"t0.9_s{s}"] for s in range(4)], 0).float().to(model.device)
        R5 = torch.stack([b[f"t0.5_s{s}"] for s in range(4)], 0).float().to(model.device)
        m4 = R9.mean(dim=0, keepdim=True)
        m8 = torch.cat([R9, R5], 0).mean(dim=0, keepdim=True)
        M5 = diffedit_cont(R5)
        M9 = diffedit_cont(R9)
        i5 = int(torch.argmin(((R9 * (1.0 - M5).unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=(1, 2, 3))))
        i9 = int(torch.argmin(((R9 * (1.0 - M9).unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=(1, 2, 3))))

        deltas = {
            "V0": R9[0], "V0s1": R9[1], "V0s2": R9[2], "V0s3": R9[3],
            "V1": m4, "V1x": m8,
            "V12": apply(M5, R9[0]),
            "S3_5": R9[i5], "S3_9": R9[i9],
            "H1_5": apply(M5, R9[i5]) + apply(1.0 - M5, m4),
            "H1_9": apply(M9, R9[i9]) + apply(1.0 - M9, m4),
            "H0_5": apply(M5, R9[0]) + apply(1.0 - M5, m4),
        }
        latents, meta = [{"variant": "recon", "lam": 0.0, "latent": x_s, "dn": 0.0}], None
        for name, delta in deltas.items():
            dnorm = float(delta.norm())
            for lam in lam_list:
                latents.append({"variant": name, "lam": lam, "latent": x_s + float(lam) * delta,
                                "dn": dnorm / float(lam)})

        decoded: List[np.ndarray] = []
        for k in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[k:k + args.batch]], dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        score = metrics.evaluate_many(source_np := np.asarray(sample.image()), decoded,
                                      sample.mask512, sample.src_prompt, sample.tar_prompt,
                                      chunk=args.eval_batch)
        src_feat = metrics._image_features([source_np])
        feats = metrics._image_features(decoded)
        cd = clip_d(feats, src_feat, metrics._text_features(sample.tar_prompt),
                    metrics._text_features(sample.src_prompt)).float().cpu()

        rows = []
        for m, s, c in zip(latents, score, cd):
            rows.append({"image_id": sample.image_id, "category": sample.category,
                         "variant": m["variant"], "lam": m["lam"], "nfe": NFE.get(m["variant"], 0),
                         "psnr": s["psnr_unedit"], "lpips": s["lpips_unedit"],
                         "ssim": s["ssim_unedit"],
                         "clip_target_whole": s["clip_target_whole"],
                         "clip_target_edit": s["clip_target_edit"],
                         "clip_d": float(c), "delta_norm_over_lam": m["dn"]})
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 20 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img "
                  f"eta {(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)

    print(f"[d1 shard {args.shard}] done={processed} in {(time.perf_counter() - start) / 60:.1f} min ({now()})",
          flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
