"""P8: the decisive 5-4 NFE comparison with ChordEdit's *own* chord field.

What changed vs P7 (the two diagnoses):

* the base field is now the chord ``u = 0.143*R(0.9) + 0.857*R(0.75)`` (ChordEdit's weights,
  from ``(delta*R(t_s) + t_s*R(t_s-delta))/(t_s+delta)`` with ``t_s=0.9, delta=0.15``), instead of
  ``R(0.9)`` alone -- P3a measured ``R(0.9)`` to be 97% noise, and this field is 86% ``R(0.75)``;
* the mask is ``Gaussian(||R(0.75)||, sigma=2)`` (0 extra NFE).  Honest note: this single-seed
  mask reaches median AUROC **0.755**, below the 0.857 of the t=0.5 mask used in P6/P7 -- the
  trade-off of staying inside 4 forwards.  Free alternatives are also measured
  (``M_chord`` on the chord energy: 0.772);
* the SDS term uses ``A(0.75) = mu_tar(0.75) - x_s`` (the *same* timestep as the field, not 0.9);
* the displacement is renormalised to ``lam*||u_chord||``, so the lambda grid is a pure amplitude
  knob and every variant competes for the same budget.

Variants (all 4 forwards + optionally the 5th prox forward):

``c_k0``            ``lam*M * u``                                    masked prox   (5 NFE)
``c_k05``, ``c_k1`` ``lam*M * (u + k*M^2*A75)``                       masked prox   (5 NFE)
``c_k1_noprox``     same as c_k1, no 5th forward                        (4 NFE)
``c_k1_fullprox``   same as c_k1, prox composited over the whole image  (5 NFE)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoedit import edit_ops  # noqa: E402
from geoedit.data import PIEDataset  # noqa: E402
from geoedit.edits import (attention_term, energy_normalise, l5_mask, masked_prox,  # noqa: E402
                           prox_latent)
from geoedit.io_utils import append_rows, load_yaml, now, write_json  # noqa: E402
from geoedit.metrics import PieMetrics  # noqa: E402
from geoedit.model import SDTurbo  # noqa: E402

FIELDS = ["image_id", "category", "run", "nfe", "psnr_unedit", "lpips_unedit", "ssim_unedit",
          "struct_dist_unedit", "struct_dist", "clip_target_edit", "clip_target_whole",
          "clip_source_whole", "delta_norm", "inmask_norm", "lam", "k", "prox"]
SCORE_KEYS = ("psnr_unedit", "lpips_unedit", "ssim_unedit", "struct_dist_unedit", "struct_dist",
              "clip_target_edit", "clip_target_whole", "clip_source_whole")
CHORD_W = 0.143  # delta/(t_s+delta) with t_s=0.9, delta=0.15
PROX_T = 0.3
MASK_SIGMA = 2.0


def chord_field(R9: torch.Tensor, R75: torch.Tensor) -> torch.Tensor:
    return CHORD_W * R9.float() + (1.0 - CHORD_W) * R75.float()


def build(run: str, x_s, R9, R75, mu_src75, M, mask_gt, lam, p8, model, eps, emb_tar):
    u = chord_field(R9, R75)
    A75 = attention_term(R75, mu_src75, x_s)
    k = {"c_k0": 0.0, "c_k05": 0.5, "c_k1": 1.0}.get(run, 1.0)
    prox = "none" if run.endswith("noprox") else ("full" if run.endswith("fullprox") else "masked")
    # `c_plain`: the ChordEdit replica through our own code path -- no mask on the field, no SDS,
    # full-image prox.  If this does NOT reach ChordEdit's CLIP-Edited, the gap is in our field or
    # pipeline rather than in the masking policy.
    if run == "c_plain":
        k, prox = 0.0, "full"
        M = torch.ones_like(M)
    delta_raw = float(lam) * M * (u + k * (M ** 2) * A75)
    delta = energy_normalise(delta_raw, u, lam)
    y = x_s + delta
    inmask = float((delta * mask_gt.unsqueeze(0)).norm())
    if prox == "none":
        return y, {"delta_norm": float(delta.norm()), "inmask_norm": inmask}
    pr = prox_latent(model, y, PROX_T, emb_tar, eps)
    out = pr if prox == "full" else masked_prox(x_s, y, pr, M ** 2)
    return out, {"delta_norm": float(delta.norm()), "inmask_norm": inmask}


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
    p8 = load_yaml(ROOT / "configs/p8.yaml")
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
    out_dir = ROOT / base["output_root"] / "p8"
    (out_dir / "frontier").mkdir(parents=True, exist_ok=True)
    img_root = ROOT / "cache" / "images" / "p8"
    model = SDTurbo(model_root=base["model_root"], device=args.device,
                    unet_precision=base["unet_precision"], vae_precision="fp32",
                    text_precision="fp32")
    print(f"[p8 {args.mode} shard {args.shard}/{args.num_shards}] {len(indices)} images, "
          f"{len(runs)} variants x {len(lams)} lam on {args.device}", flush=True)

    if args.mode == "dump":
        spec = pd.read_csv(args.manifest)
        rows = [(str(r["run"]), float(r["lam"]), str(r["target"])) for _, r in spec.iterrows()]
        for position, index in enumerate(indices):
            sample = dataset[index]
            x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
            x_s = x_s[0] if x_s.dim() == 4 else x_s
            bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
            R9 = bundle["t0.9_s0"].float().to(model.device)
            d75 = torch.load(t75_root / f"{sample.image_id}.pt", weights_only=True)
            R75 = d75["R75"][0].float().to(model.device) if d75["R75"].dim() == 4 else d75["R75"].float().to(model.device)
            mu75 = d75["mu_src75"][0].float().to(model.device) if d75["mu_src75"].dim() == 4 else d75["mu_src75"].float().to(model.device)
            M = l5_mask(R75, MASK_SIGMA)
            mask_gt = torch.from_numpy(sample.mask64).float().to(model.device)
            eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
            emb_tar = model.embed([sample.tar_prompt])
            for run, lam, target in rows:
                d = img_root / f"{run}_{target}"
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"{sample.image_id}.png"
                if path.exists():
                    continue
                latent, _ = build(run, x_s, R9, R75, mu75, M, mask_gt, lam, p8, model, eps, emb_tar)
                from PIL import Image
                Image.fromarray(model.decode(latent.unsqueeze(0))[0]).save(path)
            if (position + 1) % 25 == 0:
                print(f"  {position + 1}/{len(indices)}", flush=True)
        print(f"[p8 dump] done ({now()})")
        return 0

    metrics = PieMetrics(device=args.device)
    csv_path = out_dir / "frontier" / (f"per_image_shard{args.shard}.csv" if args.num_shards > 1
                                       else "per_image.csv")
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path, dtype={"image_id": str}, on_bad_lines="skip")
        done = set(zip(prev["image_id"], prev["run"], prev["lam"].astype(float)))
    start = time.perf_counter()
    processed = 0
    for position, index in enumerate(indices):
        sample = dataset[index]
        x_s = torch.load(a_root / "xs" / f"{sample.image_id}.pt", weights_only=True).float().to(model.device)
        x_s = x_s[0] if x_s.dim() == 4 else x_s
        bundle = torch.load(a_root / "R" / f"{sample.image_id}.pt", weights_only=True)
        R9 = bundle["t0.9_s0"].float().to(model.device)
        d75 = torch.load(t75_root / f"{sample.image_id}.pt", weights_only=True)
        R75 = d75["R75"][0].float().to(model.device) if d75["R75"].dim() == 4 else d75["R75"].float().to(model.device)
        mu75 = d75["mu_src75"][0].float().to(model.device) if d75["mu_src75"].dim() == 4 else d75["mu_src75"].float().to(model.device)
        M = l5_mask(R75, MASK_SIGMA)
        mask_gt = torch.from_numpy(sample.mask64).float().to(model.device)
        eps = edit_ops.make_eps(sample.image_id, 0, model.device, model.unet_dtype)
        emb_tar = model.embed([sample.tar_prompt])

        entries: List[dict] = [{"run": "src", "latent": x_s, "lam": 0.0, "nfe": 0, "k": 0.0,
                                "prox": "none", "delta_norm": 0.0, "inmask_norm": 0.0}]
        for run in runs:
            nfe = 4 if run.endswith("noprox") else 5
            for lam in lams:
                lat, info = build(run, x_s, R9, R75, mu75, M, mask_gt, lam, p8, model, eps, emb_tar)
                entries.append({"run": run, "latent": lat, "lam": lam, "nfe": nfe,
                                "k": {"c_k0": 0.0, "c_k05": 0.5}.get(run, 1.0),
                                "prox": ("none" if run.endswith("noprox") else
                                         "full" if run.endswith("fullprox") else "masked"),
                                "delta_norm": info["delta_norm"],
                                "inmask_norm": info["inmask_norm"]})
        todo = [e for e in entries if (sample.image_id, e["run"], e["lam"]) not in done]
        if not todo:
            continue
        decoded = []
        for i in range(0, len(todo), 6):
            stack = torch.stack([e["latent"] for e in todo[i:i + 6]])
            decoded.extend(model.decode(stack))
        source = np.asarray(sample.image())
        scored = metrics.evaluate_many(source, decoded, sample.mask512, sample.src_prompt,
                                       sample.tar_prompt, chunk=8)
        rows = []
        for e, s in zip(todo, scored):
            row = {"image_id": sample.image_id, "category": sample.category, "run": e["run"],
                   "nfe": e["nfe"], "delta_norm": e["delta_norm"], "inmask_norm": e["inmask_norm"],
                   "lam": e["lam"], "k": e["k"], "prox": e["prox"]}
            row.update({key: s[key] for key in SCORE_KEYS})
            rows.append(row)
        append_rows(csv_path, rows, FIELDS)
        processed += 1
        if processed % 10 == 0:
            el = time.perf_counter() - start
            print(f"  {position + 1}/{len(indices)} {el / processed:.1f}s/img eta "
                  f"{(len(indices) - position - 1) * el / processed / 60:.1f} min", flush=True)
    print(f"[p8 frontier shard {args.shard}] done={processed} in "
          f"{(time.perf_counter() - start) / 60:.1f} min ({now()})", flush=True)
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
