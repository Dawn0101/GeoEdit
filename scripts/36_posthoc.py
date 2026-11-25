"""Post-hoc experiments requested after the P3a preview (everything here is **not**
pre-registered; it is reported in its own section of the report).

Three things:

``--part grid``   new variants on all 700 images at t=0.9, 6 lambdas:
                  ``V12a`` = ``E_{0.5} * R_0.9``   (energy mask from t=0.5, content from t=0.9;
                                                    threshold 0.5 * max, the same recipe as V8)
                  ``V12b`` = same with the mask thresholded at the **median** energy (about
                            half the frame, closer to the GT area of 0.34)
                  ``V2d``  = dilated GT mask (``avgpool8(mask512) > 0``) * R_0.9 -- does the
                            CLIP ceiling come back to V0 when the oracle mask is dilated?
                  plus ``V0`` / ``V2`` / ``V4`` re-scored so that CLIP-D is available for the
                  same rows.  CLIP-D = cos(E(I_edit) - E(I_src), E(t_tar) - E(t_src)).

``--part bestof`` 200 fixed images, t=0.9, lambda in {1, 1.5}: decode seeds 0-3 and compare four
                  selection rules (oracle PSNR / oracle CLIP-E / non-oracle off-mask energy /
                  random) against seed 0, for the plain residual and for the V12a latent.

Usage (12 shards on the GPUs of configs/base.yaml)::

    python scripts/36_posthoc.py --part grid   --shard 0 --num-shards 12
    python scripts/36_posthoc.py --part bestof --shard 0 --num-shards 12
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

from geoedit import masks  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

GRID_FIELDS = ["image_id", "category", "t", "variant", "lam",
               "psnr", "lpips", "mse", "ssim", "struct_dist",
               "clip_target_whole", "clip_target_edit", "clip_source_whole", "clip_d"]
BEST_FIELDS = ["image_id", "category", "kind", "lam", "rule", "seed",
               "psnr", "clip_target_edit", "clip_d",
               "d_psnr", "d_clip", "d_clipd", "picked_by"]


def apply(weight: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    return weight.unsqueeze(0).unsqueeze(0) * res


def dilated_mask(sample, mask512_grid) -> torch.Tensor:
    """``avgpool8(mask512) > 0``: any 8x8 block that touches the GT mask is kept."""
    pooled = sample.mask512.astype(np.float32).reshape(64, 8, 64, 8).max(axis=(1, 3))
    return torch.from_numpy(pooled).float()


def energy_weight(Rs: torch.Tensor, mode: str) -> torch.Tensor:
    """Support of the mean energy: threshold at ``0.5 * max`` or at the median."""
    energy = Rs.norm(dim=1).mean(dim=0)
    thr = 0.5 * energy.max() if mode == "half_max" else energy.median()
    return (energy >= thr).float()


def clip_d(feats: torch.Tensor, src_feat: torch.Tensor,
           tar_text: torch.Tensor, src_text: torch.Tensor) -> torch.Tensor:
    di = feats - src_feat
    dt = (tar_text - src_text).expand_as(di)
    return (di / di.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            * dt / dt.norm(dim=-1, keepdim=True).clamp_min(1e-8)).sum(dim=-1)


def run_grid(args, base, p3, dataset, indices) -> int:
    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"], vae_only=True)
    metrics = PieMetrics(device=args.device)
    lam_list = [float(v) for v in p3["lam_list"]]
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out = ROOT / base["output_root"] / "p3" / "posthoc"
    csv_path = out / "shards" / f"grid_shard{args.shard}.csv"

    done = {row["image_id"] for row in _read(csv_path)}
    print(f"[grid shard {args.shard}] {len(indices)} images on {args.device}", flush=True)
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
        mask64 = torch.from_numpy(sample.mask64).float().to(model.device)
        source = np.asarray(sample.image())

        w5 = energy_weight(R5, "half_max")
        w5m = energy_weight(R5, "median")
        src_feat = metrics._image_features([source])
        tar_text = metrics._text_features(sample.tar_prompt)
        src_text = metrics._text_features(sample.src_prompt)

        deltas = {
            "V0": R9[0],
            "V2": apply(mask64, R9[0]),
            "V2d": apply(dilated_mask(sample, None).to(model.device), R9[0]),
            "V4": apply(masks.snr_weight(R9, c=float(p3["c_main"]), pool=int(p3["pool"]))[0], R9[0]),
            "V12a": apply(w5, R9[0]),
            "V12b": apply(w5m, R9[0]),
        }
        latents = [{"variant": "recon", "lam": 0.0, "latent": x_s}]
        for name, delta in deltas.items():
            for lam in lam_list:
                latents.append({"variant": name, "lam": lam, "latent": x_s + float(lam) * delta})

        decoded: List[np.ndarray] = []
        for k in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[k:k + args.batch]], dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        scored = metrics.evaluate_many(source, decoded, sample.mask512,
                                      sample.src_prompt, sample.tar_prompt, chunk=args.eval_batch)
        feats = metrics._image_features(decoded)
        cd = clip_d(feats, src_feat, tar_text, src_text).float().cpu()

        rows = []
        for meta, row, value in zip(latents, scored, cd):
            rows.append({"image_id": sample.image_id, "category": sample.category, "t": 0.9,
                         "variant": meta["variant"], "lam": meta["lam"],
                         "psnr": row["psnr_unedit"], "lpips": row["lpips_unedit"],
                         "mse": row["mse_unedit"], "ssim": row["ssim_unedit"],
                         "struct_dist": row["struct_dist"],
                         "clip_target_whole": row["clip_target_whole"],
                         "clip_target_edit": row["clip_target_edit"],
                         "clip_source_whole": row["clip_source_whole"],
                         "clip_d": float(value)})
        append_rows(csv_path, rows, GRID_FIELDS)
        processed += 1
        if processed % 20 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img "
                  f"eta {(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[grid shard {args.shard}] done={processed} in {(time.perf_counter() - start) / 60:.1f} min",
          flush=True)
    return 0


def run_bestof(args, base, p3, dataset, indices) -> int:
    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision=base["vae_precision"],
                    text_precision=base["text_precision"], vae_only=True)
    metrics = PieMetrics(device=args.device)
    lam_list = [float(v) for v in args.bestof_lam]
    a_root = ROOT / base["output_root"] / "p3" / "a"
    out = ROOT / base["output_root"] / "p3" / "posthoc"
    csv_path = out / "shards" / f"bestof_shard{args.shard}.csv"
    done = {row["image_id"] for row in _read(csv_path)}
    print(f"[bestof shard {args.shard}] {len(indices)} images on {args.device}", flush=True)

    start = time.perf_counter()
    for position, index in enumerate(indices):
        sample = dataset[index]
        if sample.image_id in done:
            continue
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
        b = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R9 = torch.stack([b[f"t0.9_s{s}"] for s in range(4)], 0).float().to(model.device)
        R5 = torch.stack([b[f"t0.5_s{s}"] for s in range(4)], 0).float().to(model.device)
        w5 = energy_weight(R5, "half_max").to(model.device)
        mask64 = torch.from_numpy(sample.mask64).float().to(model.device)
        source = np.asarray(sample.image())
        src_feat = metrics._image_features([source])
        tar_text = metrics._text_features(sample.tar_prompt)
        src_text = metrics._text_features(sample.src_prompt)

        latents, meta = [], []
        for kind in ("plain", "v12"):
            base_R = R9 if kind == "plain" else R9 * w5.unsqueeze(0).unsqueeze(0)
            for lam in lam_list:
                for seed in range(4):
                    latents.append({"kind": kind, "lam": lam, "seed": seed,
                                    "latent": x_s + float(lam) * base_R[seed]})
                    meta.append(None)

        decoded: List[np.ndarray] = []
        for k in range(0, len(latents), args.batch):
            stack = torch.cat([c["latent"] for c in latents[k:k + args.batch]], dim=0).to(model.device)
            decoded.extend(model.decode(stack))
        scored = metrics.evaluate_many(source, decoded, sample.mask512,
                                      sample.src_prompt, sample.tar_prompt, chunk=args.eval_batch)
        feats = metrics._image_features(decoded)
        cd = clip_d(feats, src_feat, tar_text, src_text).float().cpu()

        # non-oracle criterion (iii): off-mask energy of the t=0.9 residual
        off = 1.0 - w5
        off_energy = ((R9 * off.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=(1, 2, 3)).cpu().numpy()
        rng = np.random.default_rng(abs(hash(sample.image_id)) % (2 ** 31))

        rows = []
        for kind in ("plain", "v12"):
            for lam in lam_list:
                block = [(latents[i], scored[i], float(cd[i])) for i in range(len(latents))
                         if latents[i]["kind"] == kind and latents[i]["lam"] == lam]
                psnr = np.array([s["psnr_unedit"] for _, s, _ in block])
                clips = np.array([s["clip_target_edit"] for _, s, _ in block])
                clipds = np.array([c for _, _, c in block])
                rules = {
                    "seed0": 0,
                    "oracle_psnr": int(np.argmax(psnr)),
                    "oracle_clip": int(np.argmax(clips)),
                    "offmask_energy": int(np.argmin(off_energy)),
                    "random": int(rng.integers(0, 4)),
                }
                for rule, pick in rules.items():
                    rows.append({"image_id": sample.image_id, "category": sample.category,
                                 "kind": kind, "lam": lam, "rule": rule, "seed": int(pick),
                                 "psnr": float(psnr[pick]), "clip_target_edit": float(clips[pick]),
                                 "clip_d": float(clipds[pick]),
                                 "d_psnr": float(psnr[pick] - psnr[0]),
                                 "d_clip": float(clips[pick] - clips[0]),
                                 "d_clipd": float(clipds[pick] - clipds[0]),
                                 "picked_by": rule})
        append_rows(csv_path, rows, BEST_FIELDS)
        if (position + 1) % 20 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / (position + 1):.1f}s/img", flush=True)
    print(f"[bestof shard {args.shard}] done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    return 0


def _read(path: Path) -> List[Dict]:
    from geoedit.io_utils import read_rows

    return read_rows(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--p3-config", default="configs/p3.yaml")
    parser.add_argument("--part", choices=["grid", "bestof"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--bestof-n", type=int, default=200)
    parser.add_argument("--bestof-lam", type=float, nargs="+", default=[1.0, 1.5])
    args = parser.parse_args()

    base = load_yaml(ROOT / args.config)
    p3 = load_yaml(ROOT / args.p3_config)
    num_shards = int(args.num_shards or base["num_shards"])
    gpus = [int(g) for g in base["gpus"]]
    args.device = args.device or f"cuda:{gpus[args.shard % len(gpus)]}"
    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")

    if args.part == "bestof":
        excluded = set(str(c) for c in p3["exclude_categories"])
        pool = [i for i in range(len(dataset)) if dataset[i].category not in excluded]
        picked = [pool[i] for i in np.linspace(0, len(pool) - 1, int(args.bestof_n)).round().astype(int).tolist()]
        indices = [i for k, i in enumerate(picked) if k % num_shards == args.shard]
        return run_bestof(args, base, p3, dataset, indices)

    indices = [i for i in range(len(dataset)) if i % num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]
    return run_grid(args, base, p3, dataset, indices)


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
