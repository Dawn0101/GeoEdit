"""Score one run directory with the official PIE-Bench metrics and write the run bundle.

Output layout (task document section 8)::

    results/<run_id>/per_image.csv
    results/<run_id>/config.json
    results/<run_id>/summary.json

``per_image.csv`` columns::

    image_id, editing_type_id, struct_dist, psnr, lpips, mse, ssim,
    clip_whole, clip_edited, psnr_unedit, lpips_unedit, mse_unedit, ssim_unedit,
    mask_iou, edit_area_frac, gt_area_frac

``mask_iou`` / ``edit_area_frac`` are only filled when the run also provides an edit
mask (OT runs); for proposal baselines they stay empty.

Usage::

    python scripts/eval_run.py --run-id B1_chord
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from geoedit.cache import RESULTS, image_path
from geoedit.metrics import PieMetrics
from geoedit.pie import DEFAULT_PIE_ROOT, load_mapping, mask_decode

COLUMNS = [
    "image_id",
    "editing_type_id",
    "struct_dist",
    "psnr",
    "lpips",
    "mse",
    "ssim",
    "clip_whole",
    "clip_edited",
    "psnr_unedit",
    "lpips_unedit",
    "mse_unedit",
    "ssim_unedit",
    "mask_iou",
    "edit_area_frac",
    "gt_area_frac",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--pie-root", type=Path, default=DEFAULT_PIE_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pred-root", type=Path, default=None,
                    help="defaults to cache/images/<run_id>")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to results/<run_id>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    mapping = load_mapping(args.pie_root)
    pred_root = args.pred_root or image_path(args.run_id, "x").parent
    out_dir = args.out_dir or (RESULTS / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = PieMetrics(device=args.device)

    image_ids = sorted(mapping)
    if args.limit:
        image_ids = image_ids[: args.limit]

    rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    t_start = time.perf_counter()
    for position, image_id in enumerate(image_ids):
        pred_path = pred_root / f"{image_id}.jpg"
        if not pred_path.exists():
            skipped.append(image_id)
            continue
        sample = mapping[image_id]
        source = np.asarray(Image.open(sample.image_path).convert("RGB"))
        output = np.asarray(Image.open(pred_path).convert("RGB"))
        mask = mask_decode(sample.rle_mask, image_shape=(source.shape[0], source.shape[1]))
        background = 1.0 - mask

        row: Dict[str, object] = {"image_id": image_id, "editing_type_id": sample.editing_type_id}
        if background.sum() == 0:
            # Official evaluator returns "nan" when the background is empty (style edits).
            row.update({k: float("nan") for k in
                        ("psnr", "lpips", "mse", "ssim", "psnr_unedit", "lpips_unedit",
                         "mse_unedit", "ssim_unedit", "struct_dist", "clip_whole",
                         "clip_edited")})
        else:
            row.update(metrics.evaluate(
                source, output, mask, sample.original_prompt, sample.editing_prompt
            ))
        row["gt_area_frac"] = float(mask.mean())
        row["mask_iou"] = np.nan
        row["edit_area_frac"] = np.nan
        rows.append(row)

        if args.log_every and (position + 1) % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {position + 1}/{len(image_ids)} {elapsed / max(1, len(rows)) * 1e3:.0f} ms/img",
                  flush=True)

    if not rows:
        raise SystemExit(f"no predictions found under {pred_root}")

    frame = pd.DataFrame(rows)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[COLUMNS]
    frame.to_csv(out_dir / "per_image.csv", index=False)

    summary = {
        "run_id": args.run_id,
        "n_images": int(len(frame)),
        "n_skipped": len(skipped),
        "overall": {c: _mean(frame[c]) for c in COLUMNS if c not in ("image_id", "editing_type_id")},
        "by_type": {
            str(type_id): {
                "n": int(len(group)),
                **{c: _mean(group[c]) for c in COLUMNS
                   if c not in ("image_id", "editing_type_id")},
            }
            for type_id, group in frame.groupby("editing_type_id")
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "run_id": args.run_id,
        "pred_root": str(pred_root),
        "pie_root": str(args.pie_root),
        "seed": 42,
        "metrics": "PnPInversion evaluation/matrics_calculator.py (unmodified)",
    }, indent=2) + "\n")

    print(f"\n{args.run_id}: n={len(frame)} skipped={len(skipped)}")
    for key in ("psnr_unedit", "lpips_unedit", "clip_edited", "clip_whole", "struct_dist"):
        print(f"  {key:14s} {summary['overall'][key]:8.3f}")


def _mean(series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


if __name__ == "__main__":
    with __import__("torch").no_grad():
        main()
