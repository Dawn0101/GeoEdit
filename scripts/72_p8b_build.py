"""P8b: the corrected Ours-5 -- per the three bugs found in the P7 implementation.

What was wrong (all three verified against ``deepseek_talk/p7/p7_frontier_summary.csv``):

1. the L5 mask is max-normalised, and the composite squared it, so the **mean** ``M^2`` inside the
   ground-truth region was **0.155** -- 84.5% of the edit was blended back to ``x_s``.  Worse, the
   resulting CLIP-Edited (20.13) was *below* the unedited source (20.26).
2. ``delta`` was renormalised to ``lam*||R_h||`` where ``R_h`` is **unmasked**, while ``delta_raw``
   is masked: an in-mask amplification of **4.56x** at every lambda (11.6x at the matched point).
3. the direction gate (``gamma_dir`` = 0.0278) multiplied ``u`` but not ``A``, so the DDS term was
   effectively switched off; ``o5_nogate`` already showed no-gate is better.

This build therefore uses the clean recipe:

    u   = 0.143*R(0.9) + 0.857*R(0.75)                       # ChordEdit's chord weights
    E   = Gaussian(||R(0.75)||, sigma=2)
    M   = clip((E - q30) / (q85 - q30), 0, 1)                # quantile saturation, per image
    A   = mu_tar(0.75) - x_s
    dl  = lam * M * (u + k*M^2*A) / (1 + k*M^2)              # convex mix, 1:1 with ChordEdit at M=1
    prox= x0(a_0.3*(x_s+dl) + b_0.3*eps, 0.3, c_tar)         # the 5th forward
    out = (1-M)*x_s + M*prox                                 # *un-squared* M in the composite

``k=0`` degenerates to ``lam*M*u``: exactly ChordEdit's displacement where ``M=1``, no amplification.

The audit (``mask_audit.csv``) records the area of ``M>0.5`` and ``mean M`` inside/outside the GT
region for every image, so the 20-35% target can be checked rather than assumed.
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

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import attention_term, prox_latent  # noqa: E402
from geoedit.io_utils import append_rows, load_yaml, now, write_json  # noqa: E402
from geoedit.masks import gaussian_blur  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "run", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist_unedit", "struct_dist", "clip_target_edit", "clip_target_whole",
          "clip_source_whole", "delta_norm", "inmask_norm", "mask_area", "lam", "k", "prox"]
SCORE_KEYS = ("psnr_unedit", "lpips_unedit", "ssim_unedit", "struct_dist_unedit", "struct_dist",
              "clip_target_edit", "clip_target_whole", "clip_source_whole")
W_CHORD, T_CHORD, PROX_T, SIGMA = 0.143, 0.75, 0.3, 2.0


def chord_field(R9, R75):
    return W_CHORD * R9.float() + (1.0 - W_CHORD) * R75.float()


def quantile_mask(R75, lo=0.30, hi=0.85):
    """Quantile-saturated mask: object interior -> 1, background -> 0 (no 1.5% pinhole)."""
    e = gaussian_blur(R75.float().norm(dim=0), SIGMA)
    ql = torch.quantile(e.flatten(), lo)
    qh = torch.quantile(e.flatten(), hi)
    return ((e - ql) / (qh - ql + 1e-8)).clamp(0.0, 1.0)


def build(run, x_s, R9, R75, mu75, M, lam, model, eps, emb_tar):
    u = chord_field(R9, R75)
    A = attention_term(R75, mu75, x_s)
    k = {"b_k0": 0.0, "b_k05": 0.5, "b_k1": 1.0}.get(run, 1.0)
    prox = "none" if run.endswith("noprox") else ("full" if run.endswith("fullprox") else "masked")
    denom = 1.0 + k * (M ** 2)
    delta = float(lam) * M * (u + k * (M ** 2) * A) / denom
    y = x_s + delta
    info = {"delta_norm": float(delta.norm()), "inmask_norm": float(delta.norm())}
    if prox == "none":
        return y, info
    pr = prox_latent(model, y, PROX_T, emb_tar, eps)
    if prox == "full":
        return pr, info
    return (1.0 - M) * x_s + M * pr, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["frontier", "dump"], default="frontier")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    base = load_yaml(ROOT / args.config)
    p8 = load_yaml(ROOT / "configs/p8b.yaml")
    lams = [float(x) for x in p8["lam_list"]]
    runs = [v["name"] for v in p8["variants"]]

    dataset = PIEDataset(pie_root=Path(base["pie_root"]), cache_dir=ROOT / base["cache_root"] / "pie")
    pool = list(range(len(dataset)))
    eval_idx = [pool[i] for i in np.linspace(0, len(pool) - 1, int(p8["n_images"]))
                .round().astype(int).tolist()]
    indices = [i for k, i in enumerate(eval_idx) if k % args.num_shards == args.shard]
    if args.limit:
        indices = indices[: args.limit]

    a_root = ROOT / base["output_root"] / "p3" / "a"
    t75_root = ROOT / base["output_root"] / "p8" / "a" / "t75"
    out_dir = ROOT / base["output_root"] / "p8b"
    (out_dir / "frontier").mkdir(parents=True, exist_ok=True)
    img_root = ROOT / "cache" / "images" / "p8b"
    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32")
    print(f"[p8b {args.mode} shard {args.shard}/{args.num_shards}] {len(indices)} images, "
          f"{len(runs)} variants x {len(lams)} lam on {args.device}", flush=True)

    if args.mode == "dump":
        spec = pd.read_csv(args.manifest)
        rows = [(str(r["run"]), float(r["lam"]), str(r["target"])) for _, r in spec.iterrows()]
        for position, index in enumerate(indices):
            sample = dataset[index]
            x_s, R9, R75, mu75 = load_inputs(sample, a_root, t75_root, model)
            M = quantile_mask(R75)
            eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
            emb_tar = model.embed([sample.tar_prompt])
            for run, lam, target in rows:
                d = img_root / f"{run}_{target}"
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{sample.image_id}.png"
                if path.exists():
                    continue
                latent, _ = build(run, x_s, R9, R75, mu75, M, lam, model, eps, emb_tar)
                from PIL import Image
                Image.fromarray(model.decode(latent.unsqueeze(0))[0]).save(path)
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(indices)}", flush=True)
        print(f"[p8b dump] done ({now()})")
        return 0

    metrics = PieMetrics(device=args.device)
    csv_path = out_dir / "frontier" / (f"per_image_shard{args.shard}.csv" if args.num_shards > 1
                                       else "per_image.csv")
    audit_path = out_dir / "frontier" / "mask_audit.csv"
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
        done = set(zip(prev["image_id"], prev["run"], prev["lam"].astype(float)))
    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        x_s, R9, R75, mu75 = load_inputs(sample, a_root, t75_root, model)
        M = quantile_mask(R75)
        gt = torch.from_numpy(sample.mask64 > 0.5)
        append_rows(audit_path, [{"image_id": sample.image_id,
                                  "area_gt05": float((M > 0.5).float().mean()),
                                  "M_in_gt": float(M[gt].mean()) if bool(gt.any()) else float("nan"),
                                  "M_out_gt": float(M[~gt].mean()), "gt_area": float(gt.float().mean()),
                                  "R75_norm": float(R75.norm()), "u_norm": float(chord_field(R9, R75).norm())}],
                     ["image_id", "area_gt05", "M_in_gt", "M_out_gt", "gt_area", "R75_norm", "u_norm"])
        eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        emb_tar = model.embed([sample.tar_prompt])
        entries: List[dict] = [{"run": "src", "latent": x_s, "lam": 0.0, "nfe": 0, "k": 0.0,
                                "prox": "none", "delta_norm": 0.0, "inmask_norm": 0.0}]
        for run in runs:
            nfe = 4 if run.endswith("noprox") else 5
            for lam in lams:
                lat, info = build(run, x_s, R9, R75, mu75, M, lam, model, eps, emb_tar)
                entries.append({"run": run, "latent": lat, "lam": lam, "nfe": nfe,
                                "k": {"b_k0": 0.0, "b_k05": 0.5}.get(run, 1.0),
                                "prox": ("none" if run.endswith("noprox") else
                                         "full" if run.endswith("fullprox") else "masked"),
                                "delta_norm": info["delta_norm"],
                                "inmask_norm": info["inmask_norm"]})
        todo = [e for e in entries if (sample.image_id, e["run"], e["lam"]) not in done]
        if not todo:
            continue
        decoded = []
        for i in range(0, len(todo), 6):
            decoded.extend(model.decode(torch.stack([e["latent"] for e in todo[i:i + 6]])))
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=8)
        rows = []
        for e, s in zip(todo, scored):
            row = {"image_id": sample.image_id, "category": sample.category, "run": e["run"],
                   "nfe": e["nfe"], "delta_norm": e["delta_norm"], "inmask_norm": e["inmask_norm"],
                   "mask_area": float((M > 0.5).float().mean()), "lam": e["lam"], "k": e["k"],
                   "prox": e["prox"]}
            row.update({key: s[key] for key in SCORE_KEYS})
            rows.append(row)
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img eta "
                  f"{(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[p8b frontier shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


def load_inputs(sample, a_root, t75_root, model):
    x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
    x_s = x_s[0] if x_s.dim() == 4 else x_s
    R9 = torch.load(a_root / "R" / f"{sample.image_id}.pt",
                    weights_only=True)["t0.9_s0"].float().to(model.device)
    d = torch.load(t75_root / f"{sample.image_id}.pt", weights_only=True)
    R75 = (d["R75"][0] if d["R75"].dim() == 4 else d["R75"]).float().to(model.device)
    mu75 = (d["mu_src75"][0] if d["mu_src75"].dim() == 4 else d["mu_src75"]).float().to(model.device)
    return x_s, R9, R75, mu75


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
