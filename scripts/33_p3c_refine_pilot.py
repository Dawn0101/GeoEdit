"""P3c: coarse mask -> likelihood-ratio refinement (doc §5, exploratory, no judgement).

For the 100 images listed in ``configs/p3.yaml`` (``p3c.image_ids``), at ``t = 0.9``:

1. coarse ``w0 = snr_weight`` (N=4, c=1) and the edit ``x1 = x_s + 1 * w0 * R0``;
2. the P3a residuals are reused (no UNet needed for step 1-2);
3. likelihood-ratio maps at ``t_r = 0.5`` for ``x_s`` and ``x1`` with seeds ``{10, 11}``
   (8 UNet forwards), giving ``dL = l(x1) - l(x_s)``;
4. refine ``w1 = w0 * 1[box3(dL) > 0]`` -- drop the pixels that did not become more
   target-like -- and edit again with ``w1`` at ``lambda in {0.5, 1, 1.5, 2}``;
5. IoU / AUROC of ``w0`` and ``w1`` against ``mask64``, plus 12 mask pictures.

Outputs
-------
``results/p3/c/pilot.csv``      one row per (image, mask_type, lam)
``results/p3/c/masks_png/``     12 fixed images: source | GT | w0 | w1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops, masks, reward  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "mask_type", "lam", "psnr", "lpips", "mse", "ssim",
          "struct_dist", "clip_target_whole", "clip_target_edit", "clip_source_whole",
          "auroc", "iou"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=8)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    c_cfg = p3["p3c"]
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    t_coarse = float(c_cfg["t_coarse"])
    t_refine = float(c_cfg["t_refine"])
    seeds_refine = [int(s) for s in c_cfg["seeds"]]
    lam_list = [float(v) for v in c_cfg["lam_list"]]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    ids = [str(i) for i in c_cfg["image_ids"]]
    picked = [ids[i] for i in range(len(ids)) if i % num_shards == args.shard]
    if args.limit:
        picked = picked[: args.limit]
    vis_ids = set(ids[:12])

    out = ROOT / base["output_root"] / "p3" / "c"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"shard{args.shard}.csv"
    png_dir = out / "masks_png"
    if args.shard == 0:
        png_dir.mkdir(parents=True, exist_ok=True)

    model = SDTurbo(model_root=base["model_root"], device=device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    metrics = PieMetrics(device=device)
    a_root = ROOT / base["output_root"] / "p3" / "a"
    print(f"[p3c shard {args.shard}/{num_shards}] {len(picked)} images on {device}", flush=True)

    start = time.perf_counter()
    for position, image_id in enumerate(picked):
        sample = dataset[image_id]
        x_s = torch.load(a_root / "xs" / f"{image_id}.pt", map_location="cpu",
                         weights_only=True).float().to(model.device)
        bundle = torch.load(a_root / "R" / f"{image_id}.pt", map_location="cpu", weights_only=True)
        Rs = torch.stack([bundle[f"t{t_coarse:g}_s{s}"] for s in p3["seeds"]], dim=0).float().to(model.device)
        mask64 = torch.from_numpy(sample.mask64).float().to(model.device)
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])
        source = np.asarray(sample.image())

        w0 = masks.snr_weight(Rs, c=float(p3["c_main"]), pool=int(p3["pool"]))[0]
        R0 = Rs[0]
        x1 = x_s + apply(w0, R0)

        eps_refine = [edit_ops.make_eps(image_id, s, model.device, model.unet_dtype)
                      for s in seeds_refine]
        l_src = reward.lr_map(model, x_s, t_refine, eps_refine, emb_src, emb_tar)
        l_new = reward.lr_map(model, x1, t_refine, eps_refine, emb_src, emb_tar)
        delta_l = l_new - l_src
        w1 = reward.refine(w0, delta_l, pool=3)

        gt = masks.to_numpy(mask64)
        rows: List[Dict] = []
        latents = [{"mask_type": "recon", "lam": 0.0, "latent": x_s}]
        for name, w in (("w0", w0), ("w1", w1)):
            for lam in lam_list:
                latents.append({"mask_type": name, "lam": lam, "latent": x_s + float(lam) * apply(w, R0)})

        decoded: List[np.ndarray] = []
        for chunk_start in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[chunk_start:chunk_start + args.batch]],
                              dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        scored_all = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                           sample.tar_prompt, chunk=args.eval_batch)
        quality = {"recon": (float("nan"), float("nan"))}
        for name, w in (("w0", w0), ("w1", w1)):
            w_np = masks.to_numpy(w)
            quality[name] = (masks.auroc(w_np, gt), masks.iou(w_np, gt))
        quality["gt"] = (masks.auroc(gt, gt), masks.iou(gt, gt))

        for meta, scored in zip(latents, scored_all):
            auroc, iou = quality[meta["mask_type"]]
            rows.append({"image_id": image_id, "category": sample.category,
                         "mask_type": meta["mask_type"], "lam": meta["lam"],
                         "psnr": scored["psnr_unedit"], "lpips": scored["lpips_unedit"],
                         "mse": scored["mse_unedit"], "ssim": scored["ssim_unedit"],
                         "struct_dist": scored["struct_dist"],
                         "clip_target_whole": scored["clip_target_whole"],
                         "clip_target_edit": scored["clip_target_edit"],
                         "clip_source_whole": scored["clip_source_whole"],
                         "auroc": auroc, "iou": iou})
        append_rows(csv_path, rows, FIELDS)

        if image_id in vis_ids:
            strip = np.concatenate([
                np.asarray(sample.image().resize((256, 256))),
                (mask64.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy() * 255).astype(np.uint8).repeat(8, 0).repeat(8, 1)[:256, :256],
                (w0.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy() * 255).astype(np.uint8).repeat(8, 0).repeat(8, 1)[:256, :256],
                (w1.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy() * 255).astype(np.uint8).repeat(8, 0).repeat(8, 1)[:256, :256],
            ], axis=1)
            Image.fromarray(strip).save(png_dir / f"{image_id}_masks.png")

        if (position + 1) % 10 == 0:
            print(f"  {position + 1}/{len(picked)} {np.round((time.perf_counter() - start) / (position + 1), 1)}s/img",
                  flush=True)

    print(f"[p3c shard {args.shard}] done {len(picked)} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


def apply(weight: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    return weight.unsqueeze(0).unsqueeze(0) * res


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
