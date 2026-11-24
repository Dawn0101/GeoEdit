"""Build the ChordEdit proposal cache for all of PIE-Bench (T2).

Per image we store both baselines at once, because they share the transport step:

* ``B0_chord_noprox`` = ``x_src + step_scale * u_hat``                 (1 NFE)
* ``B1_chord``        = ``prox(B0, t_c, c_tar)``                       (2 NFE)

``x_src``, ``y``, ``y_prox`` and the sampler ``noise`` go into ``cache/proposals``;
the decoded JPEGs go into ``cache/images/<run_id>/`` for the metric pass.  The script is
sharded and resumable, so it can be spread over several free GPUs and re-run after a
crash without redoing finished images.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/build_cache.py --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geoedit.cache import has_proposal, image_dir, image_path, save_proposal
from geoedit.chord import ChordEditor
from geoedit.pie import DEFAULT_PIE_ROOT, load_mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pie-root", type=Path, default=DEFAULT_PIE_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step-scale", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    mapping = load_mapping(args.pie_root)
    image_ids = sorted(mapping)
    image_ids = [i for idx, i in enumerate(image_ids) if idx % args.num_shards == args.shard]
    if args.limit:
        image_ids = image_ids[: args.limit]

    print(f"shard {args.shard}/{args.num_shards}: {len(image_ids)} images on {args.device}")

    editor = ChordEditor(
        device=args.device,
        seed=args.seed,
        edit_config={"step_scale": args.step_scale},
    )
    for run_id in ("B0_chord_noprox", "B1_chord"):
        image_dir(run_id).mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    t_start = time.perf_counter()
    for position, image_id in enumerate(image_ids):
        if has_proposal(image_id) and image_path("B1_chord", image_id).exists():
            skipped += 1
            continue
        sample = mapping[image_id]
        try:
            image = Image.open(sample.image_path).convert("RGB")
            state = editor.transport(
                image, sample.original_prompt, sample.editing_prompt, seed=args.seed
            )
            y, noise = state["y"], state["noise"]

            t0 = time.perf_counter()
            cond = editor.encode_prompt(sample.editing_prompt)
            y_prox = editor.prox(y, cond, noise)
            if editor.device.type == "cuda":
                torch.cuda.synchronize(editor.device)
            prox_seconds = time.perf_counter() - t0

            decoded = editor.decode(torch.cat([y, y_prox], dim=0))
            Image.fromarray(decoded[0]).save(image_path("B0_chord_noprox", image_id), quality=95)
            Image.fromarray(decoded[1]).save(image_path("B1_chord", image_id), quality=95)

            save_proposal(
                image_id,
                {
                    "x_src": state["x_src"][0].cpu().numpy(),
                    "y": y[0].cpu().numpy(),
                    "y_prox": y_prox[0].cpu().numpy(),
                    "noise": noise[0].cpu().numpy(),
                },
                {"transport_s": float(state["seconds"]), "prox_s": float(prox_seconds),
                 "seed": float(args.seed)},
            )
            done += 1
            if args.log_every and (position + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - t_start
                per_image = elapsed / max(1, done + skipped)
                left = (len(image_ids) - position - 1) * per_image
                print(f"  {position + 1}/{len(image_ids)} done={done} skipped={skipped} "
                      f"{per_image * 1e3:.0f} ms/img, eta {left / 60:.1f} min", flush=True)
        except Exception:  # keep going; report at the end
            failed += 1
            print(f"  FAILED {image_id}")
            traceback.print_exc()

    elapsed = time.perf_counter() - t_start
    print(f"shard {args.shard}: done={done} skipped={skipped} failed={failed} "
          f"in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    with torch.no_grad():
        main()
