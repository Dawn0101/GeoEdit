"""E1: is the proposal's background deviation geometric misalignment or repaint/drift?

This is the diagnostic that decides whether *any* spatially tolerant mask (block matching,
partial OT, or otherwise) can help at all.  It is method-agnostic: it only looks at the
ChordEdit proposals ``y`` (B0, transport only) and ``y_prox`` (B1, after the proximal
step), both already cached.

For every 8x8 *latent* patch that lies inside the GT background (mask complement, eroded
by 2 latent cells) we compute

``r0``    ``mean((y - x_src)^2)`` at zero displacement            (latent, per element)
``r*``    min over displacements in ``[-3, 3]^2`` of the same     (block matching)
``gain``  ``1 - r*/r0``: how much of the residual a rigid shift explains
``dino``  1 - cosine similarity of DINOv2 patch features, decoded images

plus two diagnostics that are not in the original plan but directly answer the question:

``drift_ratio``  share of the residual energy that survives a low-pass filter, i.e.
                 low-frequency colour drift vs high-frequency repaint.  Block matching
                 cannot fix drift; a feature-space threshold with colour-normalised
                 inputs can.
``fp_rate``      fraction of *background* tokens whose pointwise cost already exceeds
                 ``2 tau`` for ``2 tau in {2, 4, 8}``: the false-positive rate of the
                 pointwise (``R = 0``) mask that all spatially tolerant variants must beat.

Usage::

    CUDA_VISIBLE_DEVICES=3 python scripts/run_e1.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from geoedit.cache import RESULTS, image_path, load_proposal
from geoedit.features import EditFeatureExtractor, block_scale
from geoedit.pie import DEFAULT_PIE_ROOT, load_mapping, mask_decode

PATCH = 8  # latent cells per patch (64 / 8 = 8x8 patches)
MAX_SHIFT = 3
ERODE = 2
TAUS = (1.0, 2.0, 4.0)  # 2*tau in {2, 4, 8}


def box_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, applied independently per channel."""
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    channels = x.shape[1]
    wx = w.view(1, 1, 1, k).expand(channels, 1, 1, k)
    wy = w.view(1, 1, k, 1).expand(channels, 1, k, 1)
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="replicate"), wx, groups=channels)
    return F.conv2d(x, wy, groups=channels)


def patch_means(x: torch.Tensor) -> torch.Tensor:
    """``[1, C, 64, 64]`` -> ``[8, 8]`` patch means over channels and cells."""
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    return x.mean(dim=1).reshape(h // PATCH, PATCH, w // PATCH, PATCH).mean(dim=(1, 3))


def shifted(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """``x`` shifted so that output[p] == x[p + (dy, dx)]`` (replicate at the border)."""
    pad = MAX_SHIFT
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    h, w = x.shape[-2], x.shape[-1]
    return padded[..., pad + dy: pad + dy + h, pad + dx: pad + dx + w]


def background_patches(mask_latent: torch.Tensor) -> torch.Tensor:
    """``[8, 8]`` bool: patch is fully background, and safe under a +/-SHIFT displacement."""
    bg = (mask_latent < 0.5).float()  # [64, 64]
    bg_patch = bg.reshape(8, PATCH, 8, PATCH).min(dim=3).values.min(dim=1).values  # [8, 8]
    safe = torch.ones(8, 8, dtype=torch.bool, device=mask_latent.device)
    edge = MAX_SHIFT / PATCH  # patches that could move out of bounds
    safe[0, :] = safe[-1, :] = False
    safe[:, 0] = safe[:, -1] = False
    del edge
    return (bg_patch > 0.5) & safe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie-root", type=Path, default=DEFAULT_PIE_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "E1")
    args = ap.parse_args()

    device = torch.device(args.device)
    mapping = load_mapping(args.pie_root)
    image_ids = sorted(mapping)
    if args.limit:
        image_ids = image_ids[: args.limit]

    feat = EditFeatureExtractor("lat", device=str(device), grid=64)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # DINO on decoded images, lazily constructed (2.3 GB of weights).
    from geoedit.features import DinoExtractor

    dino = DinoExtractor(device=str(device))
    dino_grid = dino.grid  # 32 for 448 / 14
    dino_patch_per_latent = PATCH * dino_grid // 64  # 4 DINO patches per 8x8 latent patch

    rows: List[Dict[str, float]] = []
    for position, image_id in enumerate(image_ids):
        sample = mapping[image_id]
        proposal = load_proposal(image_id)
        x_src = torch.from_numpy(proposal["x_src"])[None].to(device)

        source_pil = Image.open(sample.image_path).convert("RGB")
        source_img = np.asarray(source_pil)
        mask = mask_decode(sample.rle_mask, image_shape=(source_img.shape[0], source_img.shape[1]))
        mask_lat = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None].to(device),
                                 size=(64, 64), mode="area")[0]
        if ERODE > 0:
            k = 2 * ERODE + 1
            mask_er = -F.max_pool2d(-F.pad(mask_lat[None], (ERODE,) * 4, mode="replicate"),
                                    kernel_size=k, stride=1)[0]
        else:
            mask_er = mask_lat
        bg_patch = background_patches(mask_er)

        for run_id in ("B0_chord_noprox", "B1_chord"):
            y = torch.from_numpy(proposal["y" if run_id.startswith("B0") else "y_prox"])[None].to(device)

            residual = (y - x_src) ** 2  # [1, 4, 64, 64]
            r0 = patch_means(residual)  # [8, 8]

            stack = []
            shifts = [(dy, dx) for dy in range(-MAX_SHIFT, MAX_SHIFT + 1)
                      for dx in range(-MAX_SHIFT, MAX_SHIFT + 1)]
            for dy, dx in shifts:
                stack.append(patch_means((y - shifted(x_src, dy, dx)) ** 2))
            stack = torch.stack(stack)  # [49, 8, 8]
            r_best, arg = stack.min(dim=0)
            flat_arg = arg.flatten().tolist()
            shift_y = torch.tensor([shifts[int(a)][0] for a in flat_arg])
            shift_x = torch.tensor([shifts[int(a)][1] for a in flat_arg])
            gain = 1.0 - r_best / r0.clamp_min(1e-12)

            # low-pass share of the residual: colour drift (low freq) vs repaint (high freq)
            low = box_blur(y - x_src, 2.0)
            drift = patch_means(low ** 2) / r0.clamp_min(1e-12)

            # DINO cosine on the decoded pair, averaged inside each 8x8 latent patch
            out_img = np.asarray(Image.open(image_path(run_id, image_id)).convert("RGB"))
            tokens = dino([source_img, out_img], dino_grid)  # [2, 1024, D], L2-normalised
            cos = (tokens[0] * tokens[1]).sum(-1).reshape(dino_grid, dino_grid)
            cos_p = cos.reshape(8, dino_patch_per_latent, 8, dino_patch_per_latent).mean(dim=(1, 3))
            dino_dist = 1.0 - cos_p

            # pointwise false positives on background: same calibration as estimate_sigma
            bundle = feat(x_src, y)
            s_f = block_scale(bundle.src, bundle.tgt)
            cost = ((bundle.src - bundle.tgt) ** 2).sum(-1).reshape(64, 64) / s_f ** 2
            bg_tokens = mask_er[0, 0] < 0.5
            fp = {f"fp_{2 * tau:g}": float((cost[bg_tokens] > 2 * tau).float().mean())
                  for tau in TAUS}

            for i in range(8):
                for j in range(8):
                    if not bool(bg_patch[i, j]):
                        continue
                    rows.append({
                        "image_id": image_id,
                        "editing_type_id": sample.editing_type_id,
                        "proposal": run_id,
                        "patch_y": i,
                        "patch_x": j,
                        "r0": float(r0[i, j]),
                        "r_best": float(r_best[i, j]),
                        "gain": float(gain[i, j]),
                        "shift_y": int(shift_y[i * 8 + j]),
                        "shift_x": int(shift_x[i * 8 + j]),
                        "drift_ratio": float(drift[i, j]),
                        "dino_dist": float(dino_dist[i, j]),
                        **fp,
                    })
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(image_ids)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "per_patch.csv", index=False)

    report = {"n_images": len(image_ids), "n_patches": int(len(frame))}
    for proposal_id, group in frame.groupby("proposal"):
        shift_mag = np.hypot(group["shift_y"], group["shift_x"])
        moved = shift_mag > 0
        significant = moved & (group["gain"] > 0.2)
        entry = {
            "n_patches": int(len(group)),
            "frac_moved": float(moved.mean()),
            "frac_significant": float(significant.mean()),
            "frac_gain_gt_0_2": float((group["gain"] > 0.2).mean()),
            "r0_p25_p50_p75_p90": [float(v) for v in group["r0"].quantile([.25, .5, .75, .9])],
            "gain_p25_p50_p75_p90": [float(v) for v in group["gain"].quantile([.25, .5, .75, .9])],
            "drift_ratio_p25_p50_p75_p90": [float(v) for v in
                                           group["drift_ratio"].quantile([.25, .5, .75, .9])],
            "dino_dist_p25_p50_p75_p90": [float(v) for v in
                                         group["dino_dist"].quantile([.25, .5, .75, .9])],
            "fp_rate": {c: float(group[c].mean()) for c in group.columns if c.startswith("fp_")},
            "by_type": {},
        }
        for type_id, sub in group.groupby("editing_type_id"):
            sub_moved = np.hypot(sub["shift_y"], sub["shift_x"]) > 0
            entry["by_type"][str(type_id)] = {
                "n_patches": int(len(sub)),
                "frac_moved": float(sub_moved.mean()),
                "frac_significant": float((sub_moved & (sub["gain"] > 0.2)).mean()),
                "r0_p50": float(sub["r0"].median()),
                "gain_p50": float(sub["gain"].median()),
                "drift_p50": float(sub["drift_ratio"].median()),
                "fp_rate": {c: float(sub[c].mean()) for c in sub.columns if c.startswith("fp_")},
            }
        report[proposal_id] = entry

    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("B0_chord_noprox", "B1_chord")},
                     indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
