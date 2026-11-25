"""P3a: build every variant's latent from the stored residuals, decode, evaluate.

Two modes (same script, so the sharding logic is shared):

``--prep``
    Recompute the residual ``R`` for all ``t in {0.9, 0.7, 0.5}`` and ``seed in 0..3`` in
    **fp32** and cache ``x_s`` too (``results/p3/a/{R,xs}/{image_id}.pt``).  P1 saved the
    residuals as fp16 and P3a decides between masks whose difference is small, so the extra
    ~1.1 GB is worth it.  Also reports how far the fp32 values are from P1's fp16 files.
    Costs 2 batched UNet forwards per image (all 12 (t, seed) pairs at once).

*(default)*
    Read one image's ``R``/``x_s``, build the variants of ``configs/p3.yaml`` for its ``t``,
    decode in small batches and score each latent with the official PIE-Bench metrics.
    **No UNet is loaded** (``vae_only``), which is what lets 4 processes share one GPU.

Outputs
-------
``results/p3/a/shards/shard{i}.csv``       one row per (image, t, variant, lam)
``results/p3/a/shards/shard{i}_mask.csv``  one row per (image, t, mask type)
``results/p3/a/png/``                      fixed 12 images, t=0.9, lam=1 (qualitative only)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops, masks  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now, read_rows, write_manifest  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402
from geoedit.noise_ops import select_draw  # noqa: E402

FIELDS = ["image_id", "category", "t", "variant", "lam", "nfe",
          "psnr", "lpips", "mse", "ssim", "struct_dist",
          "clip_target_whole", "clip_target_edit", "clip_source_whole",
          "mask_auroc", "mask_iou"]
MASK_FIELDS = ["image_id", "category", "t", "mask_type", "auroc", "iou"]
PREP_FIELDS = ["image_id", "category", "t", "seed", "res_norm", "fp16_max_abs", "fp16_rel"]

#: §3.3 "c 的敏感性": only at the main noise level, and never used to pick a result
C_SENSITIVITY = {"V4c05": 0.5, "V4c2": 2.0}

#: which mask a variant's weight comes from (for the per-row mask_auroc / mask_iou columns)
VARIANT_MASK = {"V2": "gt", "V3": "gt", "V4": "w4", "V5": "w4", "V6": "w2",
                "V7": "diffedit", "V8": "energy", "V10": "ct", "V11": "w4",
                "V4c05": "w4", "V4c2": "w4"}


def save_pt(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def apply_weight(weight: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """``w[64,64] * R[1,4,64,64]`` with the weight broadcast over the 4 channels."""
    return weight.unsqueeze(0).unsqueeze(0) * res


# --------------------------------------------------------------------------- #
# --prep
# --------------------------------------------------------------------------- #
def run_prep(args, base: dict, p3: dict, dataset: PIEDataset, indices: List[int]) -> int:
    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"])
    t_list = [float(t) for t in p3["t_list"]]
    seeds = [int(s) for s in p3["seeds"]]
    pairs = [(t, s) for t in t_list for s in seeds]
    a_by_t = {t: model.ab(t)[0] for t in t_list}
    b_by_t = {t: model.ab(t)[1] for t in t_list}

    out = ROOT / base["output_root"] / "p3" / "a"
    rows: List[Dict] = []
    start = time.perf_counter()
    done = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        r_path = out / "R" / f"{sample.image_id}.pt"
        if args.resume and r_path.exists():
            continue
        x_s = model.encode([sample.image()])
        emb_src = model.embed([sample.src_prompt])
        emb_tar = model.embed([sample.tar_prompt])

        eps = torch.stack([edit_ops.make_eps(sample.image_id, s, model.device, model.unet_dtype)
                           for _, s in pairs], dim=0)                       # [12,4,64,64]
        t_vec = torch.tensor([t for t, _ in pairs], device=model.device, dtype=torch.float32)
        a_vec = torch.tensor([a_by_t[t] for t, _ in pairs], device=model.device,
                             dtype=x_s.dtype).view(-1, 1, 1, 1)
        b_vec = torch.tensor([b_by_t[t] for t, _ in pairs], device=model.device,
                             dtype=x_s.dtype).view(-1, 1, 1, 1)
        z = a_vec * x_s.expand(len(pairs), -1, -1, -1) + b_vec * eps
        mu_src = model.x0_pred(z, t_vec, emb_src.expand(len(pairs), -1, -1))
        mu_tar = model.x0_pred(z, t_vec, emb_tar.expand(len(pairs), -1, -1))
        res = (mu_tar - mu_src).float().cpu()

        bundle = {}
        for k, (t, s) in enumerate(pairs):
            bundle[f"t{t:g}_s{s}"] = res[k]
            row = {"image_id": sample.image_id, "category": sample.category, "t": t, "seed": s,
                   "res_norm": float(res[k].norm().item()), "fp16_max_abs": "", "fp16_rel": ""}
            old = ROOT / base["output_root"] / "p1" / "latents" / f"{sample.image_id}_t{t:g}_s{s}.pt"
            if old.exists():
                old_t = torch.load(old, map_location="cpu", weights_only=True).float()[0]
                diff = float((res[k] - old_t).abs().max())
                row["fp16_max_abs"] = diff
                row["fp16_rel"] = diff / max(float(res[k].abs().max()), 1e-12)
            rows.append(row)
        save_pt(r_path, bundle)
        save_pt(out / "xs" / f"{sample.image_id}.pt", x_s.float().cpu())
        done += 1
        if args.log_every and done % args.log_every == 0:
            elapsed = time.perf_counter() - start
            print(f"  [prep {args.shard}] {position + 1}/{len(indices)} {elapsed / done:.2f}s/img "
                  f"eta {(len(indices) - position - 1) * elapsed / done / 60:.1f} min", flush=True)

    append_rows(out / "shards" / f"prep_shard{args.shard}.csv", rows, PREP_FIELDS)
    print(f"[prep shard {args.shard}] {len(indices)} images, {done} computed, "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# build + evaluate
# --------------------------------------------------------------------------- #
def build_masks(Rs: torch.Tensor, Rs_fine: Optional[torch.Tensor], mask64: torch.Tensor,
                c_main: float, pool: int, n_small: int, need_cross_t: bool):
    """All masks of §2.1 as ``{name: (weight, auroc, iou)}`` (``gt`` included)."""
    weights: Dict[str, torch.Tensor] = {
        "w4": masks.snr_weight(Rs, c=c_main, pool=pool)[0],
        "w2": masks.snr_weight(Rs[:n_small], c=c_main, pool=pool)[0],
        "diffedit": masks.diffedit_mask(Rs),
        "energy": masks.energy_mask(Rs[0]),
    }
    for name, value in C_SENSITIVITY.items():
        weights[name] = masks.snr_weight(Rs, c=value, pool=pool)[0]
    if need_cross_t:
        if Rs_fine is None:
            raise ValueError("V10 needs the t=0.5 residuals")
        weights["ct"] = weights["w4"] * masks.snr_weight(Rs_fine, c=c_main, pool=pool)[0]

    gt = masks.to_numpy(mask64)
    quality: Dict[str, Tuple[torch.Tensor, float, float]] = {"gt": (mask64, masks.auroc(gt, gt), masks.iou(gt, gt))}
    for name, w in weights.items():
        w_np = masks.to_numpy(w)
        quality[name] = (w, masks.auroc(w_np, gt), masks.iou(w_np, gt))
    return weights, quality


def build_variants(names: List[str], weights: Dict[str, torch.Tensor], Rs: torch.Tensor,
                   m_all: torch.Tensor, mask64: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``{variant: delta}`` for one noise level; every variant is defined in doc §3.1."""
    R0 = Rs[0]
    out: Dict[str, torch.Tensor] = {}
    for name in names:
        if name == "V0":
            out[name] = R0
        elif name == "V1":
            out[name] = m_all
        elif name == "V2":
            out[name] = apply_weight(mask64, R0)
        elif name == "V3":
            out[name] = apply_weight(mask64, m_all)
        elif name == "V4":
            out[name] = apply_weight(weights["w4"], R0)
        elif name == "V5":
            out[name] = apply_weight(weights["w4"], m_all)
        elif name == "V6":
            out[name] = apply_weight(weights["w2"], R0)
        elif name == "V7":
            out[name] = apply_weight(weights["diffedit"], R0)
        elif name == "V8":
            out[name] = apply_weight(weights["energy"], R0)
        elif name == "V10":
            out[name] = apply_weight(weights["ct"], R0)
        elif name == "V11":
            out[name] = apply_weight(weights["w4"], Rs[select_draw(Rs, weights["w4"])])
        elif name in C_SENSITIVITY:
            out[name] = apply_weight(weights[name], R0)
        else:
            raise ValueError(f"unknown variant {name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=4, help="VAE decode batch (peak ~1.5 GB/item)")
    parser.add_argument("--eval-batch", type=int, default=8, help="CLIP batch inside evaluate_many")
    parser.add_argument("--prep", action="store_true", help="recompute fp32 residuals + cache x_s")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    args.device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    out = ROOT / base["output_root"] / "p3" / "a"
    if args.prep:
        if args.shard == 0:
            write_manifest(out / "manifest_prep.json",
                           {"stage": "p3a_prep", "configs": {"base": base, "p3": p3},
                            "num_shards": num_shards, "dtype": "fp32"})
        return run_prep(args, base, p3, dataset, indices)

    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"], vae_only=True)
    metrics = PieMetrics(device=args.device)

    shard_dir = out / "shards"
    shard_csv = shard_dir / f"shard{args.shard}.csv"
    mask_csv = shard_dir / f"shard{args.shard}_mask.csv"
    png_dir = out / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    if args.shard == 0:
        write_manifest(out / "manifest.json", {
            "stage": "p3a_build_and_eval", "configs": {"base": base, "p3": p3},
            "num_shards": num_shards, "batch": args.batch,
            "device_policy": "cuda:{gpus[shard % len(gpus)]}"})

    done_ids = {row["image_id"] for row in read_rows(shard_csv)}
    print(f"[p3a shard {args.shard}/{num_shards}] {len(indices)} images on {args.device}"
          f"{f', resuming ({len(done_ids)} done)' if done_ids else ''}", flush=True)

    t_list = [float(t) for t in p3["t_list"]]
    lam_list = [float(v) for v in p3["lam_list"]]
    variants_by_t = {float(k): list(v) for k, v in p3["variants"].items()}
    seeds = [int(s) for s in p3["seeds"]]
    vis_ids = set(p3["p3c"]["image_ids"][: int(p3["vis_n"])])

    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        if sample.image_id in done_ids:
            continue
        x_s = torch.load(out / "xs" / f"{sample.image_id}.pt", map_location="cpu",
                         weights_only=True).float()
        bundle = torch.load(out / "R" / f"{sample.image_id}.pt", map_location="cpu", weights_only=True)
        Rs_by_t = {t: torch.stack([bundle[f"t{t:g}_s{s}"] for s in seeds], dim=0).float()
                   for t in t_list}
        mask64 = torch.from_numpy(sample.mask64).float()
        source = np.asarray(sample.image())

        rows: List[Dict] = []
        mask_rows: List[Dict] = []
        latents: List[Tuple[Dict, torch.Tensor]] = [
            ({"t": 0.0, "variant": "recon", "lam": 0.0, "auroc": "", "iou": ""}, x_s.clone())
        ]

        for t in t_list:
            names = variants_by_t.get(t)
            if not names:
                continue
            names = list(names) + [k for k in C_SENSITIVITY if t == 0.9]
            Rs = Rs_by_t[t]
            weights, quality = build_masks(
                Rs, Rs_by_t.get(0.5), mask64, c_main=float(p3["c_main"]), pool=int(p3["pool"]),
                n_small=int(p3["n_small"]), need_cross_t=("V10" in names))
            for mask_type, (_w, auc, iou) in quality.items():
                mask_rows.append({"image_id": sample.image_id, "category": sample.category,
                                  "t": t, "mask_type": mask_type, "auroc": auc, "iou": iou})
            m_all = Rs.mean(dim=0, keepdim=True)
            deltas = build_variants(names, weights, Rs, m_all, mask64)
            for variant, delta in deltas.items():
                mask_type = VARIANT_MASK.get(variant)
                auc, iou = ("", "") if not mask_type else (quality[mask_type][1], quality[mask_type][2])
                for lam in lam_list:
                    latents.append(({"t": t, "variant": variant, "lam": lam,
                                     "auroc": auc, "iou": iou},
                                    x_s + float(lam) * delta))

        decoded_all: List[np.ndarray] = []
        for chunk_start in range(0, len(latents), args.batch):
            stack = torch.cat([c[1] for c in latents[chunk_start:chunk_start + args.batch]],
                              dim=0).to(model.device)
            decoded_all.extend(model.decode(stack))
        scored_all = metrics.evaluate_many(source, decoded_all, sample.mask512,
                                           sample.src_prompt, sample.tar_prompt,
                                           chunk=args.eval_batch)
        for (meta, _latent), output, scored in zip(latents, decoded_all, scored_all):
            rows.append({
                "image_id": sample.image_id, "category": sample.category,
                "t": meta["t"], "variant": meta["variant"], "lam": meta["lam"], "nfe": 0,
                "psnr": scored["psnr_unedit"], "lpips": scored["lpips_unedit"],
                "mse": scored["mse_unedit"], "ssim": scored["ssim_unedit"],
                "struct_dist": scored["struct_dist"],
                "clip_target_whole": scored["clip_target_whole"],
                "clip_target_edit": scored["clip_target_edit"],
                "clip_source_whole": scored["clip_source_whole"],
                "mask_auroc": meta["auroc"], "mask_iou": meta["iou"],
            })
            if (sample.image_id in vis_ids and meta["t"] == 0.9 and meta["lam"] == 1.0
                    and meta["variant"] in {"recon", "V0", "V2", "V4"}):
                Image.fromarray(output).save(
                    png_dir / f"{sample.image_id}_{meta['variant']}.jpg", quality=92)

        append_rows(shard_csv, rows, FIELDS)
        append_rows(mask_csv, mask_rows, MASK_FIELDS)
        processed += 1
        if args.log_every and processed % args.log_every == 0:
            elapsed = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {elapsed / processed:.1f}s/img "
                  f"eta {(len(indices) - position - 1) * elapsed / processed / 60:.1f} min",
                  flush=True)

    print(f"[p3a shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
