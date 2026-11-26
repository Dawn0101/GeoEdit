"""P3d-D2: is the model's own likelihood-ratio map a usable verifier?

D2a (per pixel): start from a *plain* edit ``x1 = x_s + R0`` at t=0.9, lambda=1 (no SNR gating),
then compute ``dl = l(x1) - l(x_s)`` at ``t_r = 0.5`` for K in {2, 8} and report the continuous
AUROC of ``dl``, ``|dl|`` and ``l(x_s)`` against the GT latent mask.

D2b (whole image): 4 candidates ``x_i = x_s + R_i`` (lambda=1) scored with **common random
numbers** (the same eps for every candidate), ``L_i = sum_{p in M5} l_p(x_i)``; report the
within-image Spearman correlation of ``L_i`` with CLIP-Edited / CLIP-D, and the selection gain
of ``argmax L`` over seed 0 (next to the random and CLIP-oracle references).

Outputs
-------
``results/p3d/d2/pixel_shard{i}.csv``   one row per (image, K)
``results/p3d/d2/whole_shard{i}.csv``   one row per (image, candidate seed)
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

from geoedit import masks, reward  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402
from geoedit.edit_ops import make_eps  # noqa: E402

PIXEL_FIELDS = ["image_id", "category", "K", "auroc_signed", "auroc_abs", "auroc_lsrc",
                "area_frac_m5"]
WHOLE_FIELDS = ["image_id", "category", "seed", "L", "clip_target_edit", "clip_d", "psnr"]


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
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--k-list", type=int, nargs="+", default=[2, 8])
    parser.add_argument("--lam", type=float, default=1.0)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"
    t_edit, t_ref = 0.9, 0.5

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    ids = [str(i) for i in p3["p3c"]["image_ids"]]
    picked = [i for k, i in enumerate(ids) if k % num_shards == args.shard]
    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    metrics = PieMetrics(device=device)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out = ROOT / base["output_root"] / "p3d" / "d2"
    print(f"[d2 shard {args.shard}/{num_shards}] {len(picked)} images on {device}", flush=True)

    start = time.perf_counter()
    for position, image_id in enumerate(picked):
        sample = dataset[image_id]
        x_s = torch.load(a_root / "xs" / f"{image_id}.pt", weights_only=True).float().to(model.device)
        b = torch.load(a_root / "R" / f"{image_id}.pt", weights_only=True)
        R9 = torch.stack([b[f"t{t_edit:g}_s{s}"] for s in range(4)], 0).float().to(model.device)
        R5 = torch.stack([b[f"t{t_ref:g}_s{s}"] for s in range(4)], 0).float().to(model.device)
        mask64 = sample.mask64
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        energy5 = R5.norm(dim=1).mean(dim=0)
        M5 = masks.to_numpy(energy5 / energy5.max().clamp_min(1e-12))
        x1 = x_s + args.lam * R9[0]

        pixel_rows = []
        for K in args.k_list:
            eps_list = [make_eps(image_id, 10 + j, model.device, model.unet_dtype) for j in range(K)]
            l_src = reward.lr_map(model, x_s, t_ref, eps_list, emb_src, emb_tar)
            l_new = reward.lr_map(model, x1, t_ref, eps_list, emb_src, emb_tar)
            d_l = (l_new - l_src)[0]
            pixel_rows.append({
                "image_id": image_id, "category": sample.category, "K": K,
                "auroc_signed": masks.auroc(masks.to_numpy(d_l), mask64),
                "auroc_abs": masks.auroc(np.abs(masks.to_numpy(d_l)), mask64),
                "auroc_lsrc": masks.auroc(masks.to_numpy(l_src[0]), mask64),
                "area_frac_m5": float((M5 > 0.5).mean()),
            })
        append_rows(out / "shards" / f"pixel_shard{args.shard}.csv", pixel_rows, PIXEL_FIELDS)

        # D2b: four candidates, common random numbers
        eps_common = [make_eps(image_id, 20 + j, model.device, model.unet_dtype) for j in range(4)]
        L = []
        for i in range(4):
            cand = x_s + args.lam * R9[i]
            l_map = reward.lr_map(model, cand, t_ref, eps_common, emb_src, emb_tar)
            L.append(float((l_map[0] * torch.from_numpy(M5).float().to(model.device)).sum()))
        cands = [x_s + args.lam * R9[i] for i in range(4)]
        decoded: List[np.ndarray] = []
        for k in range(0, 4, args.batch):
            decoded.extend(model.decode(torch.cat(cands[k:k + args.batch], dim=0)))
        source_np = np.asarray(sample.image())
        scored = metrics.evaluate_many(source_np, decoded, sample.mask512,
                                       sample.src_prompt, sample.tar_prompt, chunk=4)
        feats = metrics._image_features(decoded)
        cd = clip_d(feats, metrics._image_features([source_np]),
                    metrics._text_features(sample.tar_prompt),
                    metrics._text_features(sample.src_prompt)).float().cpu()
        rows = [{"image_id": image_id, "category": sample.category, "seed": i, "L": L[i],
                 "clip_target_edit": scored[i]["clip_target_edit"],
                 "clip_d": float(cd[i]), "psnr": scored[i]["psnr_unedit"]} for i in range(4)]
        append_rows(out / "shards" / f"whole_shard{args.shard}.csv", rows, WHOLE_FIELDS)

        if (position + 1) % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(picked)} {el / (position + 1):.1f}s/img", flush=True)

    print(f"[d2 shard {args.shard}] done in {(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
