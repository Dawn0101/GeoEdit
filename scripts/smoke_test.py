"""End-to-end smoke test: ChordEdit transport -> prox -> decode -> official PIE-Bench metrics.

Run it after any env / model change.  It is the T0 acceptance gate: a truncated weight
file, a wrong scheduler, a broken VAE scaling or a missing metric dependency all show up
here in under two minutes instead of 700 images into a run.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test.py --per-type 1 --verify-official
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geoedit.chord import ChordEditor
from geoedit.metrics import PieMetrics
from geoedit.pie import DEFAULT_PIE_ROOT, load_mapping, mask_decode, select_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie-root", type=Path, default=DEFAULT_PIE_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-type", type=int, default=1)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-official", action="store_true",
                    help="compare our transport against the official __call__ path")
    args = ap.parse_args()

    mapping = load_mapping(args.pie_root)
    samples = select_samples(mapping, per_type=args.per_type, types=args.types)
    print(f"PIE-Bench: {len(mapping)} samples, testing {len(samples)}")

    editor = ChordEditor(device=args.device, seed=args.seed)
    metrics = PieMetrics(device=args.device)

    rows = []
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        mask = mask_decode(sample.rle_mask, image_shape=(image.size[1], image.size[0]))
        source = np.asarray(image)

        state = editor.transport(image, sample.original_prompt, sample.editing_prompt, seed=args.seed)
        y = state["y"]
        noise = state["noise"]
        x_src = state["x_src"]

        if args.verify_official:
            official = editor.pipe(
                image,
                source_prompt=sample.original_prompt,
                target_prompt=sample.editing_prompt,
                edit_config={**editor.edit_config, "cleanup": False},
                seed=args.seed,
                output_type="pt",
            )
            gap = float((official.latents.to(args.device) - y).abs().max())
            print(f"  [verify] max |our y - official latents| = {gap:.3e}")
            assert gap < 1e-5, "our transport does not match the official pipeline"

        tgt_embed = editor.encode_prompt(sample.editing_prompt)
        y_prox = editor.prox(y, tgt_embed, noise)

        decoded = editor.decode(torch.cat([y, y_prox], dim=0))
        out_b0, out_b1 = decoded

        row = {"image_id": sample.image_id, "type": sample.editing_type_id}
        row.update({f"B0_{k}": v for k, v in metrics.evaluate(
            source, out_b0, mask, sample.original_prompt, sample.editing_prompt
        ).items()})
        row.update({f"B1_{k}": v for k, v in metrics.evaluate(
            source, out_b1, mask, sample.original_prompt, sample.editing_prompt
        ).items()})
        rows.append(row)

        latent_drift = float(((y - x_src) ** 2).mean().sqrt())
        print(f"  {sample.image_id} type={sample.editing_type_id} "
              f"t={state['seconds'] * 1e3:.0f}ms latent_rms={latent_drift:.3f} "
              f"PSNR_unedit B0={row['B0_psnr_unedit']:.2f} B1={row['B1_psnr_unedit']:.2f} "
              f"CLIP_E B0={row['B0_clip_edited']:.2f} B1={row['B1_clip_edited']:.2f}")

    keys = [k for k in rows[0] if k.startswith(("B0_", "B1_"))]
    print("\n=== mean over {} samples ===".format(len(rows)))
    for key in keys:
        values = [r[key] for r in rows]
        print(f"  {key:28s} {np.mean(values):8.3f}")

    for column in ("B0_psnr_unedit", "B1_psnr_unedit", "B1_clip_edited"):
        print(f"\n[{column}] paper-scale sanity: "
              f"{np.mean([r[column] for r in rows]):.2f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
